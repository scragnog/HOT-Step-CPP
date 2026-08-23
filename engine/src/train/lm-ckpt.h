#pragma once
// lm-ckpt.h — per-layer gradient checkpointing + chunked cross-entropy: the
// low-VRAM LM training path that makes 4B fit in ~12 GB.
//
// docs/plans/2026-07-28-lm-4b-training.md §3.3 - §3.7
//
// Why this exists (D1/D2). ggml_build_backward_expand computes the ACTIVATION
// gradient of mul_mat(W, x) as ggml_out_prod(W, transpose(grad)); out_prod is
// F32-only on CUDA and GGML_ABORTs for BF16 on CPU. So every frozen projection
// on the gradient path must be F32 *while its graph runs*. In a whole-model
// graph all L layers' F32 copies are live at once (16 GB at 4B); with one
// checkpoint segment per layer exactly ONE is (385 MiB). That single fact is
// what unlocks 4B.
//
// Shape of a micro-step (§3.5):
//
//   P1  embedding      -> C[lo]                              grads = false
//   P2  collect        C[l] -> C[l+1], l = lo .. hi-2        grads = false
//   P3  tail forward   C[hi-1] -> t_H (layer hi-1 + final norm)
//   P4  clear t_G
//   P5  chunked CE head: per-chunk logits/loss/dL-dh into t_G, no autodiff
//   P7  backward segments l = hi-1 .. lo: recompute the layer from C[l], build
//       the surrogate loss SUM(Y (.) dY), backward it, accumulate LoRA grads
//       into opt.acc[] and dL/dC[l] into the other Gh ping-pong buffer.
//
// Why the surrogate is exact: ggml_sum's backward is repeat(grad); with
// grad = t_one = 1.0 that is an all-ones [H,S], and ggml_mul's backward w.r.t.
// Y is mul(ones, dY) = dY — exact in fp32. So the segment sees precisely the
// upstream gradient and the parameter gradients equal a whole-graph backward's
// by the chain rule.
//
// D13 — the recomputed forward MUST use the same LmLayerOpts as the collect
// pass. Skipping the F32 cast in the collect pass is faster and silently turns
// checkpointing from an identity into a ~1e-3 approximation.
//
// lm-optim.h is NOT modified (D8): lm_ckpt_fill_gacc() below reads only the
// public LmOptim::param_slot / LmOptim::acc members.

#include "train/lm-common.h"
#include "train/lm-data.h"
#include "train/lm-graph.h"
#include "train/lm-optim.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ─── §6.1 Phase-0 instrumentation: the Lever B build gate ───────────────────
//
// Micro-batching's ENTIRE win is amortising per-graph HOST cost (graph build +
// sched_reset + split planning) and kernel-launch overhead over B samples,
// minus padding waste — the GEMMs themselves get no more efficient (S7). So the
// design made the lever conditional on measuring that host fraction FIRST:
// build Lever B only if (build + planning + launch) / total >= 10 % at 0.6B.
//
// This is stderr-only and off unless HOTSTEP_LM_PHASE_TIMER is set in the
// environment: no new flag, no contract change, and when disabled the only cost
// is a predictable branch. The numbers a run produces are untouched, which
// §6.0's byte-identity gate depends on.
// The decision rule names THREE amortisable terms — build, planning and launch.
// An earlier cut of this timer bucketed ggml_backend_sched_graph_compute()
// wholesale into `compute` and then reported `AMORTISABLE(build+reset)`, i.e. it
// answered a narrower question than the one the gate asks. Both missing terms
// live inside that one call, and both are reachable through PUBLIC api:
//
//   ggml_backend_sched_graph_compute(sched, gf)                      is exactly
//     = ggml_backend_sched_graph_compute_async(sched, gf)            [submit]
//     + ggml_backend_sched_synchronize(sched)                        [GPU tail]
//   …and compute_async's first act is ggml_backend_sched_alloc_graph()
//     = split_graph + gallocr_alloc_graph                            [plan]
//   (ggml-backend.cpp:1882-1901), which is legal to hoist because the async
//   entry point simply skips it when `is_alloc` is already set.
//
// So the split below is a bookkeeping change, not a behaviour change, and it is
// only taken when the timer is on.
//
// `submit` is an UPPER bound on kernel-launch overhead: CUDA launches are
// asynchronous, so it is pure host submission cost UNLESS the queue backs up, in
// which case it also absorbs GPU wait. Upper-bound-high is the honest direction
// for a gate deciding whether to build a lever — if even the upper bound leaves
// the total under the bar, the closure is safe.
struct LmPhaseTimer {
    bool      on         = false;
    long long reset_us   = 0;  // ggml_backend_sched_reset
    long long plan_us    = 0;  // split_graph + gallocr (ggml_backend_sched_alloc_graph)
    long long submit_us  = 0;  // compute_async after alloc: kernel launch, UPPER bound
    long long sync_us    = 0;  // ggml_backend_sched_synchronize: the GPU itself
    long long total_us   = 0;
    int       graphs     = 0;
    int       steps      = 0;

    void begin_step() { on = true; }

    void report(const char * tag) const {
        if (!steps) {
            return;
        }
        const double t   = (double) total_us / (double) steps / 1000.0;
        const double r   = (double) reset_us / (double) steps / 1000.0;
        const double pl  = (double) plan_us / (double) steps / 1000.0;
        const double sb  = (double) submit_us / (double) steps / 1000.0;
        const double gpu = (double) sync_us / (double) steps / 1000.0;
        // Everything that is not sched_reset and not sched_graph_compute: graph
        // construction, ggml_free, input uploads, buffer clears. An UPPER bound
        // on what batching could amortise (the mask/token uploads in here are
        // per-micro-step, not per-graph, and would only partly amortise).
        const double b   = t - r - pl - sb - gpu;
        // THE NUMBER THE GATE IS ABOUT: every per-graph host term batching would
        // spread over B samples. Bar is >= 10 % at 0.6B.
        const double am  = b + r + pl + sb;
        fprintf(stderr,
                "[phase0] %s  steps=%d graphs/step=%.1f | build(host)=%.2f ms (%.2f%%) "
                "sched_reset=%.2f ms (%.2f%%) plan=%.2f ms (%.2f%%) submit=%.2f ms (%.2f%%) "
                "gpu_sync=%.2f ms (%.2f%%) | total=%.2f ms/micro-step | "
                "AMORTISABLE(build+reset+plan+submit)=%.2f%% vs the 10%% bar -> %s\n",
                tag, steps, (double) graphs / (double) steps, b, 100.0 * b / t, r, 100.0 * r / t, pl, 100.0 * pl / t,
                sb, 100.0 * sb / t, gpu, 100.0 * gpu / t, t, 100.0 * am / t,
                100.0 * am / t >= 10.0 ? "BUILD LEVER B" : "CLOSE LEVER B");
    }
};

static LmPhaseTimer g_lm_phase;

static inline bool lm_phase_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * e = getenv("HOTSTEP_LM_PHASE_TIMER");
        cached         = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached != 0;
}

// One sched run, phase-split when the timer is on and byte-for-byte the shipped
// single call when it is off.
static inline bool lm_phase_compute(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    if (!lm_phase_enabled()) {
        return ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    }
    const int64_t t0 = ggml_time_us();
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        g_lm_phase.plan_us += ggml_time_us() - t0;
        g_lm_phase.graphs++;
        return false;
    }
    const int64_t t1 = ggml_time_us();
    const ggml_status st = ggml_backend_sched_graph_compute_async(sched, gf);
    const int64_t t2 = ggml_time_us();
    ggml_backend_sched_synchronize(sched);
    const int64_t t3 = ggml_time_us();
    g_lm_phase.plan_us += t1 - t0;
    g_lm_phase.submit_us += t2 - t1;
    g_lm_phase.sync_us += t3 - t2;
    g_lm_phase.graphs++;
    return st == GGML_STATUS_SUCCESS;
}

// ─── configuration + persistent state ───────────────────────────────────────

struct LmCkptCfg {
    int chunk           = 128;  // trained positions per CE chunk
    int attn_head_block = 0;    // q heads per attention block (0 = off)
    int s_max           = 0;    // longest accepted sequence
    int layer_lo        = 0;
    int layer_hi        = 0;    // exclusive; <= 0 means cfg.n_layers

    // Lever A (--weights bf16). Default OFF => lm_ckpt_layer_opts() returns a
    // default LmLayerOpts and every graph below is emitted verbatim (§6.0).
    bool weights_bf16   = false;

    // ── OUTPUT-HEAD OVERRIDE (MiniMax-Music3) ──────────────────────────────
    //
    // ACE's planner LM ties its head to embed_tokens and scores the whole
    // vocabulary. MM3 ships an UNTIED head and only ever scores a contiguous
    // slice of it (semantic codes + EOS: 16,389 of 200,000 rows), so the
    // chunked CE head takes a weight, a first row and a row count.
    //
    // ALL THREE DEFAULT TO THE TIED FULL-VOCAB CASE, so an ACE run emits
    // byte-identical graphs — the same discipline as Lever A above. Note this
    // is the PARENT tensor plus an offset, not a pre-made view: t_embT is
    // built by a host-side transpose that reads the parent's buffer directly.
    ggml_tensor * head_w    = nullptr;  // nullptr => lm->embed_tokens
    int64_t       head_row0 = 0;        // first scored row of head_w
    int           head_v    = 0;        // 0 => cfg.vocab_size
};

// The scored head and its width, resolved. One place, so no call site can
// disagree about which of the two cases it is in.
static inline ggml_tensor * lm_ckpt_head_src(const Qwen3LM * lm, const LmCkptCfg & cfg) {
    return cfg.head_w ? cfg.head_w : lm->embed_tokens;
}
static inline int lm_ckpt_head_width(const Qwen3LM * lm, const LmCkptCfg & cfg) {
    return cfg.head_v > 0 ? cfg.head_v : lm->cfg.vocab_size;
}

// Each group gets its OWN backend buffer because ggml_backend_buffer_clear()
// is whole-buffer.
struct LmCkptState {
    Qwen3LM * lm = nullptr;
    LmCkptCfg cfg;

    ggml_context *        ctx_ckpt = nullptr;  // C[0 .. L-1]                     never cleared
    ggml_backend_buffer_t buf_ckpt = nullptr;
    ggml_context *        ctx_gh0  = nullptr;  // Gh[0] == t_G                    cleared per micro-step
    ggml_backend_buffer_t buf_gh0  = nullptr;
    ggml_context *        ctx_gh1  = nullptr;  // Gh[1]                           cleared per segment
    ggml_backend_buffer_t buf_gh1  = nullptr;
    ggml_context *        ctx_misc = nullptr;  // t_H + t_zero_attn               cleared once
    ggml_backend_buffer_t buf_misc = nullptr;
    ggml_context *        ctx_embt = nullptr;  // t_embT                          written once
    ggml_backend_buffer_t buf_embt = nullptr;
    ggml_context *        ctx_labc = nullptr;  // t_labc                          sparse set/clear
    ggml_backend_buffer_t buf_labc = nullptr;

    std::vector<ggml_tensor *> C;
    ggml_tensor *              Gh[2]       = { nullptr, nullptr };
    ggml_tensor *              t_H         = nullptr;
    ggml_tensor *              t_zero_attn = nullptr;
    ggml_tensor *              t_embT      = nullptr;
    ggml_tensor *              t_labc      = nullptr;

    std::vector<uint8_t>       arena;  // one reused 32 MiB graph arena
    std::vector<ggml_tensor *> gacc;
    std::vector<int32_t>       pos_scratch;
    int                        last_mask_S = 0;

    size_t fixed_bytes() const {
        size_t b = 0;
        const ggml_backend_buffer_t bufs[6] = { buf_ckpt, buf_gh0, buf_gh1, buf_misc, buf_embt, buf_labc };
        for (int i = 0; i < 6; i++) {
            if (bufs[i]) {
                b += ggml_backend_buffer_get_size(bufs[i]);
            }
        }
        return b;
    }
};

static void lm_ckpt_free(LmCkptState * st) {
    ggml_backend_buffer_t bufs[6] = { st->buf_ckpt, st->buf_gh0, st->buf_gh1, st->buf_misc, st->buf_embt, st->buf_labc };
    for (int i = 0; i < 6; i++) {
        if (bufs[i]) {
            ggml_backend_buffer_free(bufs[i]);
        }
    }
    ggml_context * ctxs[6] = { st->ctx_ckpt, st->ctx_gh0, st->ctx_gh1, st->ctx_misc, st->ctx_embt, st->ctx_labc };
    for (int i = 0; i < 6; i++) {
        if (ctxs[i]) {
            ggml_free(ctxs[i]);
        }
    }
    *st = LmCkptState{};
}

// The low-VRAM path never mirrors, so the "is this base trainable?" check that
// lm_build_f32_mirror() performs on the way past has to be done explicitly.
// Same wording as the mirror's, so the server sees the same fatal message.
// `allow_quantized` is OFF by default so every existing caller keeps today's
// behaviour verbatim.
//
// When it is on, the claim being made is narrow and worth stating: the DEFAULT
// lm_linear path emits `mul_mat(qwen3_f32(w), x)`, i.e. an in-graph cast to F32
// that gallocr frees with the segment. So the backward never sees the quantized
// tensor at all — it sees the cast's F32 output, and out_prod is satisfied.
// That is QLoRA's dequantize-per-matmul, using machinery this trainer already
// had for its BF16/F16 bases.
//
// It does NOT hold under Lever A (`--weights bf16`), which deliberately feeds
// the RAW weight to mul_mat to reach the BF16 tensor cores. A quantized weight
// there would hit the transpose path that block-quantized types cannot take, so
// the caller must not combine the two.
static bool lm_ckpt_check_base(Qwen3LM * lm, std::string * err, bool allow_quantized = false) {
    const int L = lm->cfg.n_layers;
    for (int i = 0; i < L; i++) {
        const Qwen3Layer & ly = lm->layers[i];
        if (!ly.q_proj || !ly.k_proj || !ly.v_proj || !ly.o_proj || !ly.gate_proj || !ly.up_proj || !ly.down_proj) {
            char b[160];
            snprintf(b, sizeof(b), "layer %d has fused projections — the trainer requires an unfused load", i);
            *err = b;
            return false;
        }
    }
    std::vector<ggml_tensor *> all;
    all.push_back(lm->embed_tokens);
    all.push_back(lm->final_norm);
    for (int i = 0; i < L; i++) {
        Qwen3Layer & ly = lm->layers[i];
        ggml_tensor * t[11] = { ly.input_layernorm, ly.post_attn_layernorm, ly.q_proj,  ly.k_proj,
                                ly.v_proj,          ly.o_proj,              ly.q_norm,  ly.k_norm,
                                ly.gate_proj,       ly.up_proj,             ly.down_proj };
        for (int j = 0; j < 11; j++) {
            all.push_back(t[j]);
        }
    }
    for (size_t i = 0; i < all.size(); i++) {
        ggml_tensor * s = all[i];
        if (!s) {
            continue;
        }
        const bool plain = s->type == GGML_TYPE_F32 || s->type == GGML_TYPE_BF16
                        || s->type == GGML_TYPE_F16;
        // A quantized weight is acceptable only if it can be dequantized by the
        // in-graph cast, which needs a type with a defined to_float and a real
        // block size (i.e. an ordinary k-quant or Q8_0, not something exotic).
        const ggml_type_traits * tr = ggml_get_type_traits(s->type);
        const bool castable = allow_quantized && tr && tr->to_float != nullptr;
        if (!plain && !castable) {
            char b[256];
            snprintf(b, sizeof(b),
                     "base weight '%s' is %s — LM training needs a BF16/F16/F32 base, or a quantized one "
                     "with the dequantizing cast enabled (ggml_out_prod is F32-only)",
                     s->name, ggml_type_name(s->type));
            *err = b;
            return false;
        }
    }
    return true;
}

// Allocate after sample building, when S_max / V are known.
static bool lm_ckpt_alloc(LmCkptState * st, Qwen3LM * lm, const LmCkptCfg & cfg, std::string * err) {
    const Qwen3LMConfig & c = lm->cfg;
    const int             L = c.n_layers, H = c.hidden_size;
    const int             D = c.head_dim, Nh = c.n_heads;
    // V is the SCORED width, which is the vocabulary only in the tied case.
    const int             V = lm_ckpt_head_width(lm, cfg);

    st->lm  = lm;
    st->cfg = cfg;
    if (st->cfg.layer_hi <= 0) {
        st->cfg.layer_hi = L;
    }
    const int S = st->cfg.s_max;
    if (S <= 0 || st->cfg.chunk <= 0) {
        *err = "invalid checkpoint configuration (s_max / chunk)";
        return false;
    }
    if (!lm_ckpt_head_block_ok(c, st->cfg.attn_head_block)) {
        char b[160];
        snprintf(b, sizeof(b), "--attn-head-block %d does not divide n_heads %d with n_kv_heads %d",
                 st->cfg.attn_head_block, c.n_heads, c.n_kv_heads);
        *err = b;
        return false;
    }

    auto mkctx = [](int n) {
        ggml_init_params p = { (size_t) (n + 8) * ggml_tensor_overhead(), nullptr, true };
        return ggml_init(p);
    };

    st->ctx_ckpt = mkctx(L);
    st->ctx_gh0  = mkctx(1);
    st->ctx_gh1  = mkctx(1);
    st->ctx_misc = mkctx(2);
    st->ctx_embt = mkctx(1);
    st->ctx_labc = mkctx(1);
    if (!st->ctx_ckpt || !st->ctx_gh0 || !st->ctx_gh1 || !st->ctx_misc || !st->ctx_embt || !st->ctx_labc) {
        *err = "cannot create the checkpoint contexts";
        return false;
    }

    st->C.assign((size_t) L, nullptr);
    for (int l = 0; l < L; l++) {
        st->C[(size_t) l] = ggml_new_tensor_2d(st->ctx_ckpt, GGML_TYPE_F32, H, S);
        char nm[32];
        snprintf(nm, sizeof(nm), "ckpt.%d", l);
        ggml_set_name(st->C[(size_t) l], nm);
        // Legal on a pre-allocated leaf: ggml_set_param only asserts op == NONE.
        // Being a PARAM makes it a NODE (not a leaf) of the segment graph, which
        // is what lets lm_ckpt_fill_gacc() hand it a persistent dX buffer.
        ggml_set_param(st->C[(size_t) l]);
        ggml_set_input(st->C[(size_t) l]);
    }

    st->Gh[0] = ggml_new_tensor_2d(st->ctx_gh0, GGML_TYPE_F32, H, S);
    st->Gh[1] = ggml_new_tensor_2d(st->ctx_gh1, GGML_TYPE_F32, H, S);
    ggml_set_name(st->Gh[0], "ckpt.G0");
    ggml_set_name(st->Gh[1], "ckpt.G1");
    ggml_set_input(st->Gh[0]);
    ggml_set_input(st->Gh[1]);

    st->t_H = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, H, S);
    ggml_set_name(st->t_H, "ckpt.H");
    ggml_set_input(st->t_H);
    if (st->cfg.attn_head_block > 0) {
        st->t_zero_attn = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, (int64_t) Nh * D, S);
        ggml_set_name(st->t_zero_attn, "ckpt.attn_zero");
        ggml_set_input(st->t_zero_attn);
    }

    // Dtype stays the base's own (D4): the chunked head never runs out_prod on
    // it, so no F32 copy is needed — 1,060 MiB instead of 2,120 MiB at 4B.
    // A quantized head cannot BE t_embT: the transpose below is a host-side
    // element move, and block-quantized rows do not survive one. Dequantize to
    // F16 instead — half the bytes of F32, and this tensor feeds exactly one
    // mul_mat (the chunked head's dL/dh), never a backward.
    {
        ggml_tensor *      hsrc = lm_ckpt_head_src(lm, cfg);
        const ggml_type_traits * htr = ggml_get_type_traits(hsrc->type);
        const bool         hq   = htr && htr->is_quantized;
        st->t_embT = ggml_new_tensor_2d(st->ctx_embt, hq ? GGML_TYPE_F16 : hsrc->type, V, H);
    }
    ggml_set_name(st->t_embT, "ckpt.embT");
    ggml_set_input(st->t_embT);

    st->t_labc = ggml_new_tensor_2d(st->ctx_labc, GGML_TYPE_F32, V, st->cfg.chunk);
    ggml_set_name(st->t_labc, "ckpt.labels");
    ggml_set_input(st->t_labc);

    st->buf_ckpt = ggml_backend_alloc_ctx_tensors(st->ctx_ckpt, lm->backend);
    st->buf_gh0  = ggml_backend_alloc_ctx_tensors(st->ctx_gh0, lm->backend);
    st->buf_gh1  = ggml_backend_alloc_ctx_tensors(st->ctx_gh1, lm->backend);
    st->buf_misc = ggml_backend_alloc_ctx_tensors(st->ctx_misc, lm->backend);
    st->buf_embt = ggml_backend_alloc_ctx_tensors(st->ctx_embt, lm->backend);
    st->buf_labc = ggml_backend_alloc_ctx_tensors(st->ctx_labc, lm->backend);
    if (!st->buf_ckpt || !st->buf_gh0 || !st->buf_gh1 || !st->buf_misc || !st->buf_embt || !st->buf_labc) {
        *err = "checkpoint buffer allocation failed";
        return false;
    }
    ggml_backend_buffer_clear(st->buf_ckpt, 0);
    ggml_backend_buffer_clear(st->buf_gh0, 0);
    ggml_backend_buffer_clear(st->buf_gh1, 0);
    ggml_backend_buffer_clear(st->buf_misc, 0);  // t_zero_attn: once, then NEVER again
    ggml_backend_buffer_clear(st->buf_labc, 0);

    st->arena.resize((size_t) 32 << 20);
    st->pos_scratch.resize((size_t) S);

    fprintf(stderr,
            "[train-lm] low-vram state: %d checkpoints [%d,%d] f32 (%.1f MB) + embT %s (%.1f MB) + labels [%d,%d] "
            "(%.1f MB) = %.1f MB\n",
            L, H, S, (double) L * H * S * 4.0 / 1048576.0, ggml_type_name(st->t_embT->type),
            ggml_nbytes(st->t_embT) / 1048576.0, V, st->cfg.chunk, ggml_nbytes(st->t_labc) / 1048576.0,
            st->fixed_bytes() / 1048576.0);
    return true;
}

// Host-side [H,V] -> [V,H] scatter at ggml_type_size() granularity, dtype
// preserved. PyTorch gets lm_head.weight.T free as a cuBLAS transa flag; ggml's
// mul_mat reduces over ne0 only, so a physical transpose is unavoidable (§6.2).
static bool lm_ckpt_build_embed_t(LmCkptState * st, std::string * err) {
    Qwen3LM *     lm  = st->lm;
    ggml_tensor * src_t = lm_ckpt_head_src(lm, st->cfg);
    const int     H   = lm->cfg.hidden_size;
    const int     V   = lm_ckpt_head_width(lm, st->cfg);
    const int64_t row0 = st->cfg.head_w ? st->cfg.head_row0 : 0;
    if (row0 + (int64_t) V > src_t->ne[1]) {
        *err = "the scored head slice runs past the head tensor";
        return false;
    }

    const ggml_type_traits * tr = ggml_get_type_traits(src_t->type);
    const bool               quantized = tr && tr->is_quantized;

    if (quantized) {
        // Dequantize row by row, then transpose into F16. Rows are contiguous
        // and H is a multiple of every block size in use, so a row is a whole
        // number of blocks and can be converted on its own — which keeps the
        // scratch at one row instead of the whole 16k x 4096 slice.
        if (!tr->to_float) {
            *err = std::string("the output head is ") + ggml_type_name(src_t->type)
                 + ", which has no dequantizer";
            return false;
        }
        if (st->t_embT->type != GGML_TYPE_F16) {
            *err = "embedT must be F16 for a quantized head";
            return false;
        }
        const size_t row_bytes = (size_t) (H / ggml_blck_size(src_t->type)) * ggml_type_size(src_t->type);
        std::vector<uint8_t> qrow(row_bytes);
        std::vector<float>   frow((size_t) H);
        std::vector<ggml_fp16_t> dst((size_t) V * (size_t) H);
        for (int64_t v = 0; v < V; v++) {
            ggml_backend_tensor_get(src_t, qrow.data(), (size_t) (row0 + v) * row_bytes, row_bytes);
            tr->to_float(qrow.data(), frow.data(), H);
            for (int64_t h = 0; h < H; h++) {
                dst[(size_t) (h * (int64_t) V + v)] = ggml_fp32_to_fp16(frow[(size_t) h]);
            }
        }
        ggml_backend_tensor_set(st->t_embT, dst.data(), 0, dst.size() * sizeof(ggml_fp16_t));
        return true;
    }

    if (st->t_embT->type != src_t->type) {
        *err = "embedT dtype mismatch";
        return false;
    }
    const size_t ts = ggml_type_size(src_t->type);
    // Read only the scored rows. In the tied case row0 == 0 and V == vocab, so
    // this is the whole tensor and the behaviour is unchanged.
    std::vector<uint8_t> src((size_t) V * (size_t) H * ts), dst(ggml_nbytes(st->t_embT));
    ggml_backend_tensor_get(src_t, src.data(), (size_t) row0 * (size_t) H * ts, src.size());
    for (int64_t v = 0; v < V; v++) {
        for (int64_t h = 0; h < H; h++) {
            memcpy(&dst[(size_t) (h * (int64_t) V + v) * ts], &src[(size_t) (v * (int64_t) H + h) * ts], ts);
        }
    }
    ggml_backend_tensor_set(st->t_embT, dst.data(), 0, dst.size());
    return true;
}

// ─── the only new autodiff plumbing (D8) ────────────────────────────────────
//
// PARAM-flag node classes in a segment graph, IN PRIORITY ORDER:
//   1. the segment's checkpoint input  -> g_out  (its dL/dX; NOT in param_slot)
//   2. a LoRA A/B tensor               -> o->acc[param_slot[nd]]
//   3. the surrogate loss node         -> t_one  (NOT o->t_lossgrad — D9)
//
// `ckpt_in` carries the PARAM flag AND is absent from param_slot, so the order
// of the first two branches is load-bearing: swapped, lm_optim's GGML_ASSERT
// equivalent fires on every segment.
static void lm_ckpt_fill_gacc(const LmOptim * o, ggml_cgraph * gf, ggml_tensor * ckpt_in, ggml_tensor * g_out,
                              ggml_tensor * t_one, std::vector<ggml_tensor *> * gacc) {
    const int n = ggml_graph_n_nodes(gf);
    gacc->assign((size_t) n, nullptr);
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(gf, i);
        if (nd == ckpt_in) {
            (*gacc)[(size_t) i] = g_out;
            continue;
        }
        if (nd->flags & GGML_TENSOR_FLAG_PARAM) {
            auto it = o->param_slot.find(nd);
            GGML_ASSERT(it != o->param_slot.end());
            (*gacc)[(size_t) i] = o->acc[(size_t) it->second];
            continue;
        }
        if (nd->flags & GGML_TENSOR_FLAG_LOSS) {
            (*gacc)[(size_t) i] = t_one;
        }
    }
}

// ─── one micro-step ─────────────────────────────────────────────────────────

struct LmCkptRun {
    Qwen3LM *            lm    = nullptr;
    LmOptim *            opt   = nullptr;
    ggml_backend_sched_t sched = nullptr;
    LmCkptState *        st    = nullptr;

    ggml_tensor * t_tok = nullptr;
    ggml_tensor * t_pos = nullptr;
    ggml_tensor * t_msk = nullptr;
    ggml_tensor * t_gs  = nullptr;  // [1] per-chunk upstream scalar
    ggml_tensor * t_one = nullptr;  // [1] == 1.0f, the segment surrogate's loss grad

    int grad_accum = 1;

    // Self-test only (§3.6): the un-chunked CE head. Needs a full [V, s_tr]
    // label buffer, which is exactly why D4 exists — never a production path.
    bool          naive_head    = false;
    ggml_tensor * t_lab_full    = nullptr;
    // Self-test only: cast embed_tokens / embT to F32 in the head so the
    // chunked and naive heads are numerically comparable (T11 isolates
    // chunking from the BF16 GEMM). Production leaves the head in the base's
    // own dtype, exactly as Side-Step does.
    bool          head_f32_embed = false;

    // ── INPUT-EMBEDDING OVERRIDE (MiniMax-Music3) ──────────────────────────
    //
    // ACE's LM is fed token ids and P1 is `get_rows(embed_tokens, t_tok)`. An
    // MM3 audio frame is (token_embd[semantic] + SUM_c audio_embd[code_c]) *
    // num_codebooks^-0.5, built from two tables in two different FILES, so the
    // caller builds the [H, S] input itself. nullptr keeps the token path
    // exactly as it was; `user` carries whatever the builder needs.
    ggml_tensor * (*embed_build)(ggml_context * ctx, LmCkptRun & r, int S) = nullptr;
    void *          embed_user = nullptr;

    /** Stop after the CE head — no backward segments, no gradient accumulation.
     *
     *  This is how held-out EVALUATION runs without a second graph: the forward
     *  phases and the buffers they use are the ones training already allocated,
     *  so an eval costs compute and NOT VRAM. That matters here more than it
     *  sounds — an MM3 run sits at ~30 GB of a 32 GB card, so a separate
     *  forward-only graph could be the allocation that tips it into paging.
     *
     *  The head still writes dL/dh into Gh[0] as a side effect; nothing reads
     *  it, and P4 clears that buffer at the start of every micro-step. */
    bool forward_only = false;

    // ── PRIOR-PRESERVATION CAPTURE ─────────────────────────────────────────
    //
    // With capture_k > 0 the chunked head reads its logits back and appends the
    // top-K of each supervised position to capture_idx/capture_p. That is how a
    // regularisation target is produced: the frozen base model's OWN answer,
    // recorded once, so a later step can be told "keep predicting this" instead
    // of "learn this other song too".
    //
    // It must run while the adapter is provably inert — at init B is zero, so
    // the LoRA contributes exactly nothing and the capture is the true base
    // distribution. After a single optimizer step that is no longer true, which
    // is why the result is cached to disk rather than recomputed on resume.
    int                    capture_k   = 0;
    std::vector<int32_t> * capture_idx = nullptr;
    std::vector<float>   * capture_p   = nullptr;
};

static inline void lm_ckpt_upload_mask(LmCkptRun & r, int S) {
    if (S == r.st->last_mask_S) {
        return;
    }
    std::vector<float> m;
    lm_causal_mask(S, &m);
    ggml_backend_tensor_set(r.t_msk, m.data(), 0, m.size() * sizeof(float));
    r.st->last_mask_S = S;
}

static inline LmLayerOpts lm_ckpt_layer_opts(const LmCkptState & st) {
    LmLayerOpts o;
    o.cast_weights    = true;
    o.attn_head_block = st.cfg.attn_head_block;
    o.attn_zero       = st.t_zero_attn;
    o.weights_bf16    = st.cfg.weights_bf16;
    // o.wt stays nullptr: P2/P3 are forward-only, so a transposed weight there
    // would be a node nothing consumes. Only the P7 backward segment sets it.
    return o;
}

/** Read one chunk's logits back and append each row's top-K as a normalised
 *  distribution.
 *
 *  Softmax is taken over the WHOLE scored width and only then truncated to K,
 *  so p reflects the base's real confidence — renormalising the top-K to sum to
 *  1 afterwards would turn a flat, uncertain prediction into a confident one and
 *  teach the adapter to be more certain than the model it is protecting. The
 *  leftover mass is simply dropped: rows sum to <= 1, and the cross-entropy
 *  treats the missing mass as unconstrained, which is exactly right — we have no
 *  opinion about the 16,000 codes the base considered and rejected.
 *
 *  K = 64 at 16,385 classes typically captures well over 99% of the mass; the
 *  producer logs the measured coverage so a pathological case is visible rather
 *  than assumed. */
#define LM_CAPTURE_K_MAX 256

static void lm_ckpt_capture_topk(LmCkptRun & r, ggml_tensor * lg, int V, int Sc) {
    static std::vector<float> row;              // reused; V floats
    const int K = r.capture_k;
    row.resize((size_t) V);
    for (int i = 0; i < Sc; i++) {
        ggml_backend_tensor_get(lg, row.data(), (size_t) i * (size_t) V * sizeof(float),
                                (size_t) V * sizeof(float));
        float mx = -INFINITY;
        for (int v = 0; v < V; v++) {
            if (row[v] > mx) mx = row[v];
        }
        double sum = 0.0;
        for (int v = 0; v < V; v++) {
            const double e = exp((double) row[v] - (double) mx);
            row[v] = (float) e;
            sum += e;
        }
        const float inv = (float) (1.0 / (sum > 0.0 ? sum : 1.0));

        // Insertion selection into a K-sized running top list. One pass over V
        // with an early-out on the smallest kept value; at K=64 / V=16k that is
        // nothing next to the matmul that produced these logits, and it needs no
        // per-position allocation. K is bounded at the CLI (LM_CAPTURE_K_MAX).
        int   best_i[LM_CAPTURE_K_MAX];
        float best_v[LM_CAPTURE_K_MAX];
        const int kk = K < LM_CAPTURE_K_MAX ? K : LM_CAPTURE_K_MAX;
        for (int k = 0; k < kk; k++) { best_i[k] = -1; best_v[k] = -INFINITY; }
        for (int v = 0; v < V; v++) {
            const float x = row[v];
            if (x <= best_v[kk - 1]) {
                continue;
            }
            int j = kk - 1;
            while (j > 0 && best_v[j - 1] < x) {
                best_v[j] = best_v[j - 1];
                best_i[j] = best_i[j - 1];
                j--;
            }
            best_v[j] = x;
            best_i[j] = v;
        }
        for (int k = 0; k < kk; k++) {
            r.capture_idx->push_back(best_i[k] >= 0 ? best_i[k] : 0);
            r.capture_p->push_back(best_i[k] >= 0 ? best_v[k] * inv : 0.0f);
        }
    }
}

// P5 (production): per-chunk graphs, no autodiff, disjoint t_G column writes.
static bool lm_ckpt_head_chunked(LmCkptRun & r, const LmSample & s, bool count_loss, double * ce_out) {
    LmCkptState &         st   = *r.st;
    const Qwen3LMConfig & c    = r.lm->cfg;
    const int             H    = c.hidden_size;
    const int             V    = lm_ckpt_head_width(r.lm, st.cfg);   // scored width, not the vocab
    const int             s_tr = s.s_tr, CH = st.cfg.chunk;
    const int             GA   = std::max(1, r.grad_accum);

    double ce = 0.0;
    for (int i = 0; i < s_tr; i += CH) {
        const int Sc = std::min(CH, s_tr - i);

        // D9: gs = Sc / (s_tr * grad_accum). ggml_cross_entropy_loss_back
        // computes (softmax - onehot) * gs / nrows with nrows == Sc, i.e.
        // (softmax - onehot) / (s_tr * grad_accum) — exactly Side-Step's
        // (loss / (n_tok * grad_accum)).backward(). The trunk surrogate's loss
        // gradient is therefore 1.0, NOT 1/grad_accum.
        const float gs = (float) Sc / ((float) s_tr * (float) GA);
        ggml_backend_tensor_set(r.t_gs, &gs, 0, sizeof(float));

        LmChunkLabelGuard guard(st.t_labc, s, i, Sc, V);

        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 1024, /*grads=*/false);

        const size_t  col = (size_t) (s.n_masked - 1 + i);  // [P] L6a offset
        ggml_tensor * hd  = ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, Sc, st.t_H->nb[1], col * st.t_H->nb[1]));
        ggml_tensor * hw  = lm_ckpt_head_src(r.lm, st.cfg);
        if (st.cfg.head_w) {
            hw = ggml_view_2d(ctx, hw, hw->ne[0], (int64_t) V, hw->nb[1],
                              (size_t) st.cfg.head_row0 * hw->nb[1]);
        }
        ggml_tensor * emb = r.head_f32_embed ? qwen3_f32(ctx, hw) : hw;
        ggml_tensor * ebt = r.head_f32_embed ? qwen3_f32(ctx, st.t_embT) : st.t_embT;
        ggml_tensor * lg  = ggml_mul_mat(ctx, emb, hd);  // [V, Sc]
        ggml_tensor * lb  = ggml_view_2d(ctx, st.t_labc, V, Sc, st.t_labc->nb[1], 0);
        ggml_tensor * lc  = ggml_cross_entropy_loss(ctx, lg, lb);  // scalar, mean over Sc rows
        ggml_set_output(lc);
        ggml_tensor * dl = ggml_cross_entropy_loss_back(ctx, r.t_gs, lg, lb);  // (grad, logits, labels)
        ggml_tensor * dh = ggml_mul_mat(ctx, ebt, dl);                         // [H, Sc]
        ggml_tensor * gv = ggml_view_2d(ctx, st.Gh[0], H, Sc, st.Gh[0]->nb[1], col * st.Gh[0]->nb[1]);
        if (r.capture_k > 0) {
            // The logits are an intermediate; without this the arena is free to
            // reuse their memory before the readback.
            ggml_set_output(lg);
            ggml_build_forward_expand(gf, lg);
        }
        ggml_build_forward_expand(gf, lc);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, dh, gv));

        const bool tm = lm_phase_enabled();
        const int64_t p0 = tm ? ggml_time_us() : 0;
        ggml_backend_sched_reset(r.sched);
        if (tm) {
            g_lm_phase.reset_us += ggml_time_us() - p0;
        }
        const bool ok = lm_phase_compute(r.sched, gf);
        if (ok && count_loss) {
            float lv = 0.0f;
            ggml_backend_tensor_get(lc, &lv, 0, sizeof(float));
            ce += (double) lv * (double) Sc / (double) s_tr;
        }
        if (ok && r.capture_k > 0) {
            lm_ckpt_capture_topk(r, lg, V, Sc);
        }
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }
    if (count_loss && ce_out) {
        *ce_out = ce;
    }
    return true;
}

// §3.6, SELF-TEST ONLY: one grads = true graph over the whole trained span, with
// t_H temporarily ggml_set_param'd so its gradient lands in t_G. Isolates
// "checkpointing" from "chunked loss" in the verification ladder.
static bool lm_ckpt_head_naive(LmCkptRun & r, const LmSample & s, bool count_loss, double * ce_out) {
    LmCkptState &         st   = *r.st;
    const Qwen3LMConfig & c    = r.lm->cfg;
    // Tied-head only, on purpose: this is the self-test rung that isolates
    // checkpointing from chunked loss, and it has no reason to grow a second
    // case. Loud rather than silently scoring the wrong rows.
    GGML_ASSERT(st.cfg.head_w == nullptr && "lm_ckpt_head_naive is tied-head only");
    const int             H    = c.hidden_size, V = c.vocab_size;
    const int             s_tr = s.s_tr;
    GGML_ASSERT(r.t_lab_full != nullptr);

    ggml_set_param(st.t_H);

    ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 4096, /*grads=*/true);

    const size_t  col = (size_t) (s.n_masked - 1);
    ggml_tensor * hd  = ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, s_tr, st.t_H->nb[1], col * st.t_H->nb[1]));
    // The cast is MANDATORY here, not optional: mul_mat's activation backward is
    // ggml_out_prod(src0, ...), which is F32-only.
    ggml_tensor * logits = ggml_mul_mat(ctx, qwen3_f32(ctx, r.lm->embed_tokens), hd);
    ggml_tensor * labv   = ggml_view_2d(ctx, r.t_lab_full, V, s_tr, r.t_lab_full->nb[1], 0);
    ggml_tensor * loss   = ggml_cross_entropy_loss(ctx, logits, labv);
    ggml_set_loss(loss);
    ggml_set_output(loss);
    ggml_build_forward_expand(gf, loss);

    lm_ckpt_fill_gacc(r.opt, gf, st.t_H, st.Gh[0], r.opt->t_lossgrad, &st.gacc);
    ggml_build_backward_expand(ctx, gf, st.gacc.data());

    bool ok = false;
    {
        LmLabelGuard guard(r.t_lab_full, s.targets.data(), s_tr, V);
        ggml_backend_sched_reset(r.sched);
        ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && count_loss && ce_out) {
            float ce = 0.0f;
            ggml_backend_tensor_get(loss, &ce, 0, sizeof(float));
            *ce_out = (double) ce;
        }
    }
    ggml_free(ctx);
    st.t_H->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
    return ok;
}

// Build (but do not run) one P7 segment to size the scheduler.
// `counts` (T16) makes the Lever A surgery REPORT instead of abort, so the
// self-test can print the tripwire numbers rather than take the process down.
static int lm_ckpt_probe_segment_nodes(LmCkptRun & r, int S, LmBf16Counts * counts = nullptr) {
    LmCkptState &         st = *r.st;
    const Qwen3LMConfig & c  = r.lm->cfg;
    const int             H  = c.hidden_size;
    const int             l  = st.cfg.layer_hi - 1;

    // MIRRORS P7 EXACTLY, including the Lever A surgery, so the scheduler is
    // sized against the real graph. Any drift here shows up as a mid-run
    // reallocation (and T13's leak gate catches it).
    LmLayerOpts      opts = lm_ckpt_layer_opts(st);
    LmWtCollect      wc;
    if (opts.weights_bf16) {
        opts.wt = &wc;
    }
    ggml_init_params  ip   = { st.arena.size(), st.arena.data(), true };
    ggml_context *    ctx  = ggml_init(ip);
    ggml_cgraph *     gf   = ggml_new_graph_custom(ctx, 8192, /*grads=*/true);

    ggml_tensor * pos_v = ggml_view_1d(ctx, r.t_pos, S, 0);
    ggml_tensor * mask  = ggml_view_2d(ctx, r.t_msk, S, S, (size_t) S * sizeof(float), 0);
    ggml_tensor * X     = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
    ggml_tensor * Y     = lm_train_layer(ctx, c, &r.lm->layers[l], X, pos_v, mask, S, opts);
    Y                   = lm_rms(ctx, Y, r.lm->final_norm, c.rms_norm_eps);
    lm_bf16_expand_wt(gf, wc);  // §3.2: BEFORE the loss root, or Wt lands after its consumer
    ggml_tensor * dY    = ggml_view_2d(ctx, st.Gh[0], H, S, st.Gh[0]->nb[1], 0);
    ggml_tensor * Lsur  = ggml_sum(ctx, ggml_mul(ctx, Y, dY));
    ggml_set_loss(Lsur);
    ggml_set_output(Lsur);
    ggml_build_forward_expand(gf, Lsur);
    lm_ckpt_fill_gacc(r.opt, gf, st.C[(size_t) l], st.Gh[1], r.t_one, &st.gacc);
    ggml_build_backward_expand(ctx, gf, st.gacc.data());
    if (opts.weights_bf16) {
        if (counts) {
            *counts = lm_bf16_rewrite_segment(gf, wc);
        } else {
            lm_bf16_finish_segment(gf, wc);
        }
    }
    const int n = ggml_graph_n_nodes(gf);
    ggml_free(ctx);
    return n;
}

static bool lm_ckpt_micro_step(LmCkptRun & r, const LmSample & s, bool count_loss, double * ce_out) {
    Qwen3LM &             lm = *r.lm;
    LmCkptState &         st = *r.st;
    const Qwen3LMConfig & c  = lm.cfg;
    const int             H  = c.hidden_size;
    const int             S  = (int) s.tokens.size();
    const int             Lo = st.cfg.layer_lo, Hi = st.cfg.layer_hi;

    GGML_ASSERT(S <= st.cfg.s_max && Hi > Lo);

    // ── P0: inputs ───────────────────────────────────────────────────────
    lm_ckpt_upload_mask(r, S);
    // `tokens` still SIZES the sequence, but with an embedding override its
    // contents are meaningless and t_tok is never read — do not pretend.
    if (!r.embed_build) {
        ggml_backend_tensor_set(r.t_tok, s.tokens.data(), 0, (size_t) S * 4);
    }
    for (int i = 0; i < S; i++) {
        st.pos_scratch[(size_t) i] = i;
    }
    ggml_backend_tensor_set(r.t_pos, st.pos_scratch.data(), 0, (size_t) S * 4);

    const LmLayerOpts opts = lm_ckpt_layer_opts(st);

    // §6.1: `build` is derived as total - reset - plan - submit - sync, so it absorbs every
    // remaining host cost (graph construction, ggml_free, input uploads, buffer
    // clears). That makes it an UPPER bound on what batching could amortise —
    // which is the honest way round for a gate whose job is to decide whether
    // the lever is worth building.
    const bool    phase_tm = lm_phase_enabled();
    const int64_t step_t0  = phase_tm ? ggml_time_us() : 0;

    auto run_graph = [&](ggml_cgraph * gf) -> bool {
        const int64_t p0 = phase_tm ? ggml_time_us() : 0;
        ggml_backend_sched_reset(r.sched);
        if (phase_tm) {
            g_lm_phase.reset_us += ggml_time_us() - p0;
        }
        return lm_phase_compute(r.sched, gf);
    };

    // ── P1: embedding -> C[Lo] ───────────────────────────────────────────
    {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 64, /*grads=*/false);
        // get_rows on a BF16 source yields F32 (exact upcast) and needs no
        // backward: embed_tokens is frozen and GET_ROWS' row indices are on
        // ggml's "not differentiable" list. The override (MM3 frames) is held
        // to the same contract — it must be frozen and gradient-free.
        ggml_tensor * h0 = r.embed_build
                             ? r.embed_build(ctx, r, S)
                             : ggml_get_rows(ctx, lm.embed_tokens, ggml_view_1d(ctx, r.t_tok, S, 0));
        ggml_build_forward_expand(gf,
                                  ggml_cpy(ctx, h0, ggml_view_2d(ctx, st.C[(size_t) Lo], H, S,
                                                                 st.C[(size_t) Lo]->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // ── P2: forward-collect, l = Lo .. Hi-2 ──────────────────────────────
    for (int l = Lo; l < Hi - 1; l++) {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 2048, /*grads=*/false);
        ggml_tensor *    pv  = ggml_view_1d(ctx, r.t_pos, S, 0);
        ggml_tensor *    mv  = ggml_view_2d(ctx, r.t_msk, S, S, (size_t) S * sizeof(float), 0);
        ggml_tensor *    X   = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
        // D13: SAME opts as the P7 recompute. Not an optimisation opportunity.
        ggml_tensor * Y = lm_train_layer(ctx, c, &lm.layers[l], X, pv, mv, S, opts);
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx, Y, ggml_view_2d(ctx, st.C[(size_t) (l + 1)], H, S, st.C[(size_t) (l + 1)]->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // ── P3: tail forward (layer Hi-1 + final norm) -> t_H ────────────────
    {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 2048, /*grads=*/false);
        ggml_tensor *    pv  = ggml_view_1d(ctx, r.t_pos, S, 0);
        ggml_tensor *    mv  = ggml_view_2d(ctx, r.t_msk, S, S, (size_t) S * sizeof(float), 0);
        ggml_tensor *    X   = ggml_view_2d(ctx, st.C[(size_t) (Hi - 1)], H, S, st.C[(size_t) (Hi - 1)]->nb[1], 0);
        ggml_tensor *    Y   = lm_train_layer(ctx, c, &lm.layers[Hi - 1], X, pv, mv, S, opts);
        ggml_tensor *    hN  = lm_rms(ctx, Y, lm.final_norm, c.rms_norm_eps);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, hN, ggml_view_2d(ctx, st.t_H, H, S, st.t_H->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // ── P4: t_G := 0 (masked columns must stay exactly zero) ─────────────
    ggml_backend_buffer_clear(st.buf_gh0, 0);

    // ── P5: CE head ──────────────────────────────────────────────────────
    if (r.naive_head) {
        if (!lm_ckpt_head_naive(r, s, count_loss, ce_out)) {
            return false;
        }
    } else {
        if (!lm_ckpt_head_chunked(r, s, count_loss, ce_out)) {
            return false;
        }
    }

    // Evaluation stops here: the loss is what it came for.
    if (r.forward_only) {
        return true;
    }

    // ── P6/P7: backward segments, l = Hi-1 .. Lo ─────────────────────────
    int cur = 0;  // Gh[0] == t_G, already seeded by the head
    for (int l = Hi - 1; l >= Lo; l--) {
        const int nxt = 1 - cur;
        // MANDATORY: ggml_acc_or_set accumulates INTO a supplied grad_acc, so
        // the destination must start at zero.
        ggml_backend_buffer_clear(nxt == 0 ? st.buf_gh0 : st.buf_gh1, 0);

        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 8192, /*grads=*/true);

        // Lever A: a FRESH collect per segment. The transposes are in-graph and
        // gallocr frees them with the segment (S4), so nothing survives the loop.
        LmLayerOpts sopts = opts;
        LmWtCollect wc;
        if (sopts.weights_bf16) {
            sopts.wt = &wc;
        }

        ggml_tensor * pv = ggml_view_1d(ctx, r.t_pos, S, 0);
        ggml_tensor * mv = ggml_view_2d(ctx, r.t_msk, S, S, (size_t) S * sizeof(float), 0);
        ggml_tensor * X  = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
        ggml_tensor * Y  = lm_train_layer(ctx, c, &lm.layers[l], X, pv, mv, S, sopts);
        if (l == Hi - 1) {
            Y = lm_rms(ctx, Y, lm.final_norm, c.rms_norm_eps);
        }
        // §3.2 step 2, and the one thing that is easy to get wrong:
        // ggml_build_forward_expand APPENDS, so every cont(transpose(W)) must
        // enter the graph BEFORE the loss root — otherwise the rewritten
        // backward node would consume a Wt that is computed after it, silently.
        lm_bf16_expand_wt(gf, wc);

        ggml_tensor * dY   = ggml_view_2d(ctx, st.Gh[cur], H, S, st.Gh[cur]->nb[1], 0);
        ggml_tensor * Lsur = ggml_sum(ctx, ggml_mul(ctx, Y, dY));  // L' = SUM(Y (.) dY)
        ggml_set_loss(Lsur);
        ggml_set_output(Lsur);
        ggml_build_forward_expand(gf, Lsur);

        lm_ckpt_fill_gacc(r.opt, gf, st.C[(size_t) l], st.Gh[nxt], r.t_one, &st.gacc);
        ggml_build_backward_expand(ctx, gf, st.gacc.data());

        // S18: exactly 7 rewritten, 0 skipped, 0 base-weight out_prod left, or
        // GGML_ABORT. A silent no-op flag would be a TF32 run labelled "bf16".
        if (sopts.weights_bf16) {
            lm_bf16_finish_segment(gf, wc);
        }

        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
        cur = nxt;
    }
    // Gh[cur] now holds dL/d(embedding output) and is DISCARDED: embed_tokens is
    // frozen and carries no LoRA site.
    if (phase_tm) {
        g_lm_phase.total_us += ggml_time_us() - step_t0;
        g_lm_phase.steps++;
        if (g_lm_phase.steps % 20 == 0) {
            char tag[64];
            snprintf(tag, sizeof(tag), "S=%d", S);
            g_lm_phase.report(tag);
        }
    }
    return true;
}
