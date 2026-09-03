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
#include "train/lm-kvprefix.h"
#include "train/lm-prefix.h"
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

    // Trainable KV prefix (train/lm-prefix.h). nullptr = off, byte-identical.
    // Every row of the mask sees the whole prefix (pfx_lo = 0, n_prompt = 0):
    // this prefix is conditioning the caption must read, unlike the frozen
    // history below, which the caption is deliberately blind to.
    const LmPrefix * pfx = nullptr;

    // ── --attn flash (D1/D3, docs/plans/2026-09-02-lm-flash-attn.md) ───────
    //
    // Copied straight into LmLayerOpts by lm_ckpt_layer_opts(), which is the
    // ONLY place the segment graphs get their options — so P2 (forward
    // collect), P3 (tail forward), P7 (backward segment) and
    // lm_ckpt_probe_segment_nodes() cannot disagree about the attention
    // formulation. D13's "the recomputed forward must use the same
    // LmLayerOpts as the collect pass" already depended on that; this rides on
    // the same guarantee.
    //
    // Default OFF, and off is byte-identical. MM3's trainer shares this struct
    // and never sets either field (D12), so it keeps its F32 mask and its exact
    // graph in R2.
    //
    // `attn_flash` also implies attn_head_block == 0 — the combination has no
    // fused arm and lm_attn_head_blocked asserts on it. The CLI refuses the pair
    // at exit 2 and lm_ckpt_default_head_block(c, true) returns 0.
    bool      attn_flash = false;
    ggml_prec attn_prec  = GGML_PREC_DEFAULT;

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

    // Rank dropout mask, [r] F32, owned by the caller. nullptr = off. Passed
    // straight through to LmLayerOpts, which is what makes it satisfy D13: the
    // collect and recompute passes read the same tensor, so they see the same
    // subnetwork. The caller must leave it all-ones (or clear it) for eval and
    // for any forward whose output is treated as the frozen base.
    ggml_tensor * rank_mask = nullptr;

    // ── FROZEN KV PREFIX (train/lm-kvprefix.h) ─────────────────────────────
    //
    // Non-null => every layer graph attends to `kv->n` stored history columns
    // in front of the window, and the mask becomes rectangular. The caller
    // fills the store before the micro-step; nothing here writes to it.
    //
    // nullptr (the default) => square self-attention, emitted verbatim.
    LmKvPrefix * kv = nullptr;

    // ── LIVE TEACHER (--reg-teacher live, 2026-09-03) ──────────────────────
    //
    // Allocates the two persistent tensors the live prior-preservation path
    // needs and NOTHING else changes: `t_teach_h`, an [H, s_max] F32 stash of
    // one forward pass's final hidden states, and `t_codemask`, a [V,1] F32
    // additive mask that is 0 inside [teacher_lo, teacher_hi) and -INF outside.
    //
    // WHY A HIDDEN-STATE STASH AND NOT A PROBABILITY BUFFER. The teacher's
    // answer is a [V, s_tr] distribution — 2.6 GB at ACE's 217k head and a
    // 3000-frame song, and 786 MB even restricted to the audio-code range. The
    // hidden states it is computed FROM are [H, S]: 42 MB at 4B. So the teacher
    // pass runs P1-P3 with the adapter off, stashes t_H, and the student's
    // chunked head re-derives that chunk's teacher logits from the stash with
    // one extra head GEMM. Nothing crosses PCIe and nothing is held per-token.
    //
    // Default off: `live_teacher` false allocates neither tensor, so an
    // untouched run's fixed_bytes() and every graph in this header are exactly
    // what they were.
    bool live_teacher = false;
    int  teacher_lo   = 0;  // first scored class the teacher softmax spans
    int  teacher_hi   = 0;  // one past the last; <= lo means "the whole width"
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
    // --reg-teacher live only (LmCkptCfg::live_teacher); both stay null
    // otherwise, and every consumer tests them rather than the flag.
    ggml_tensor *              t_teach_h   = nullptr;  // [H, s_max] adapter-off hidden states
    ggml_tensor *              t_codemask  = nullptr;  // [1, V] 0 in range, -1e30 outside
    ggml_tensor *              t_teach_sel = nullptr;  // [1, chunk] 1 at code positions, 0 elsewhere

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
    st->ctx_misc = mkctx(5);  // t_H, t_zero_attn, + the three live-teacher tensors
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
    // D3: t_zero_attn is the permanently-zero ggml_acc base the head-blocked
    // path reassembles into. Flash mode never blocks, so it is never allocated
    // (Nh*D*S*4 = 57 MB at 4B/S 3500 that the flash path does not spend). The
    // guard is the head-block test alone because attn_flash forces it to 0;
    // asserting that here rather than relying on it.
    GGML_ASSERT(!(st->cfg.attn_flash && st->cfg.attn_head_block > 0) &&
                "--attn flash with --attn-head-block > 0 is refused at the CLI");
    if (st->cfg.attn_head_block > 0) {
        st->t_zero_attn = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, (int64_t) Nh * D, S);
        ggml_set_name(st->t_zero_attn, "ckpt.attn_zero");
        ggml_set_input(st->t_zero_attn);
    }

    // Live teacher (--reg-teacher live). All three tensors are additive and
    // only exist when the mode is on.
    if (st->cfg.live_teacher) {
        if (st->cfg.teacher_hi <= st->cfg.teacher_lo) {
            st->cfg.teacher_lo = 0;
            st->cfg.teacher_hi = V;
        }
        st->cfg.teacher_lo = std::max(0, std::min(st->cfg.teacher_lo, V));
        st->cfg.teacher_hi = std::max(st->cfg.teacher_lo, std::min(st->cfg.teacher_hi, V));
        if (st->cfg.teacher_hi <= st->cfg.teacher_lo) {
            *err = "the live teacher's class range is empty after clamping to the scored width";
            return false;
        }
        st->t_teach_h = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, H, S);
        ggml_set_name(st->t_teach_h, "ckpt.teachH");
        ggml_set_input(st->t_teach_h);
        // [1, V] rather than [V, 1]: it is the LEFT operand of an outer product
        // with the per-position selector below, and ggml_mul_mat reduces over
        // ne0. See lm_ckpt_teacher_labels for why the mask is per position.
        st->t_codemask = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, 1, V);
        ggml_set_name(st->t_codemask, "ckpt.codemask");
        ggml_set_input(st->t_codemask);
        st->t_teach_sel = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, 1, st->cfg.chunk);
        ggml_set_name(st->t_teach_sel, "ckpt.teachSel");
        ggml_set_input(st->t_teach_sel);
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

    // The additive class mask, uploaded ONCE — the clear above has just zeroed
    // it, so this must follow it.
    //
    // -1e30f, NOT -INF: the mask is scaled by the per-position selector (0 or
    // 1) before it is added, and 0 * -INF is NaN while 0 * -1e30 is exactly 0.
    // As an added logit -1e30 is -INF for every practical purpose — after the
    // softmax's max-subtraction expf() underflows to a hard zero.
    if (st->t_codemask) {
        std::vector<float> cm((size_t) V, -1e30f);
        std::fill(cm.begin() + st->cfg.teacher_lo, cm.begin() + st->cfg.teacher_hi, 0.0f);
        ggml_backend_tensor_set(st->t_codemask, cm.data(), 0, cm.size() * sizeof(float));
        fprintf(stderr,
                "[train-lm] live teacher: hidden stash [%d,%d] f32 (%.1f MB) + class mask over [%d,%d) of %d "
                "(%.1f MB)\n",
                H, S, (double) H * S * 4.0 / 1048576.0, st->cfg.teacher_lo, st->cfg.teacher_hi, V,
                (double) V * 4.0 / 1048576.0);
    }

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
    // F32 (exact) or F16 (--attn flash, D4) — allocate it with lm_mask_alloc and
    // upload through lm_ckpt_upload_mask, which converts. Every view of it in
    // this header sizes its row stride with ggml_element_size(), so the F32 case
    // is byte-identical to the shipped one.
    ggml_tensor * t_msk = nullptr;
    ggml_tensor * t_gs  = nullptr;  // [1] per-chunk upstream scalar
    ggml_tensor * t_one = nullptr;  // [1] == 1.0f, the segment surrogate's loss grad

    int grad_accum = 1;

    /** Leave t_pos alone — the caller has already uploaded the positions.
     *
     *  The micro-step's own fill is `0 .. S-1`, which is correct for a trainer
     *  whose sequence IS the whole example. MM3's is not: a crop taken at frame
     *  c0 must be presented at RoPE position P + c0, and `--crop-anchor song`
     *  computes exactly that and uploads it — and this function then silently
     *  overwrote it. Every MM3 LM adapter trained before 2026-08-26 therefore
     *  saw each crop as if it were the song's opening.
     *
     *  Default false, so the ACE trainer is byte-identical. */
    bool pos_external = false;

    // Optional extra loss head, run AFTER the CE head has filled t_G and
    // BEFORE the backward segments consume it. The hook may ADD gradient
    // contributions into t_G (st->Gh[0]) at any supervised column; the
    // segment backward then carries them to the adapter with no further
    // changes here. Introduced for the MM3 acoustic (depth-decoder) loss;
    // null for every other trainer, which keeps ACE byte-identical.
    bool (*aux_head)(LmCkptRun &, const LmSample &, void *) = nullptr;
    void * aux_user = nullptr;

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

    /** The override still reads t_tok, so keep uploading the ids to it.
     *
     *  MM3's override builds frames from its own code tensors and never touches
     *  t_tok, which is why P0 stops maintaining it as soon as embed_build is
     *  set. An ACE artist-token hook is the opposite: it IS the default
     *  get_rows plus an acc, so without this it gathers rows out of a stale
     *  buffer and produces a confidently wrong forward — measured as loss 24.5
     *  against the naive path's 9.35 on the same data, with no error anywhere. */
    bool embed_uses_tok = false;

    /** The embedding stage is NO LONGER frozen — run a backward over it (P1B).
     *
     *  P1 is forward-only and grads=false, on the stated contract that
     *  embed_tokens is frozen and GET_ROWS' indices are not differentiable. An
     *  artist token (textual inversion) breaks that: `embed_build` then contains
     *  a real parameter, and without this flag it would receive exactly zero
     *  gradient while the loss fell perfectly convincingly on the adapter.
     *
     *  The gradient P1B needs already exists and was already being thrown away —
     *  the segment loop ends with Gh[cur] holding dL/d(embedding output). This
     *  flag just stops discarding it.
     *
     *  Default false, so every trainer that has no embedding parameter emits the
     *  graph it always did and pays one predictable branch. */
    bool embed_trainable = false;

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

    // ── LIVE TEACHER (--reg-teacher live) ──────────────────────────────────
    //
    // Three per-pass switches, all default-off. A live reg micro-step is two
    // calls to lm_ckpt_micro_step in this order:
    //
    //   1. teacher: adapter_off = true, teacher_fill = true, forward_only =
    //      true. Runs P1-P3 only, copies t_H into st->t_teach_h and returns
    //      BEFORE P4 — so it clears no gradient buffer, builds no backward and
    //      accumulates nothing. The student pass immediately after overwrites
    //      C[] and t_H, which is why the stash exists.
    //   2. student: teacher_use = true. The chunked head derives each chunk's
    //      teacher distribution from the stash and writes it into t_labc
    //      instead of reading the sample's own labels.
    //
    // `adapter_off` is per PASS and reaches P2/P3/P7 through the single
    // lm_ckpt_layer_opts() call in lm_ckpt_micro_step, so D13 still holds: one
    // micro-step cannot disagree with itself about the adapter.
    bool adapter_off  = false;
    bool teacher_fill = false;
    bool teacher_use  = false;

    // One-shot diagnostic hook. When set, the chunked head keeps each chunk's
    // logits alive and hands them to this callback after the chunk computes.
    // Used by the step-0 teacher==student check in train/lm-train-run.h, which
    // needs the base's own full-width logits to compute the entropy the live
    // reg CE must equal. Null on every training path.
    void (*logit_sink)(LmCkptRun &, ggml_tensor * lg, int V, int Sc, int off, void * user) = nullptr;
    void * logit_user = nullptr;
};

// Rows of the attention mask, i.e. how many keys a query can reach: S on the
// square path, kv->n + S once a frozen prefix sits in front of the window.
static inline int lm_ckpt_n_kv(const LmCkptState & st, int S) {
    if (st.cfg.kv) {
        return st.cfg.kv->n + S;
    }
    if (st.cfg.pfx) {
        return st.cfg.pfx->n + S;
    }
    return S;
}

static inline void lm_ckpt_upload_mask(LmCkptRun & r, int S, int n_prompt) {
    std::vector<float> m;
    if (r.st->cfg.kv) {
        // The prefix length moves with the crop, so there is nothing stable to
        // cache on — rebuild every micro-step. It is a host memset of a few
        // tens of MB against a multi-second step.
        //
        // The store's stream is [prompt ; history], and the window carries its
        // own prompt, so the prompt's stored columns are skipped: pfx_lo =
        // n_prompt. Prompt ROWS stay blind to the whole prefix, which is what
        // keeps their hidden states identical to the no-prefix case.
        lm_causal_mask_prefix(r.st->cfg.kv->n, n_prompt, S, n_prompt, &m);
        // lm_mask_set converts to F16 when the buffer is F16 (--attn flash, D4)
        // and is a plain ggml_backend_tensor_set of the same bytes when it is
        // F32 — which every current caller of this path is.
        lm_mask_set(r.t_msk, m);
        r.st->last_mask_S = -1;
        return;
    }
    if (S == r.st->last_mask_S) {
        return;
    }
    if (r.st->cfg.pfx) {
        lm_causal_mask_prefix(r.st->cfg.pfx->n, /*pfx_lo=*/0, S, /*n_prompt=*/0, &m);
    } else {
        lm_causal_mask(S, &m);
    }
    lm_mask_set(r.t_msk, m);
    r.st->last_mask_S = S;
}

// The per-layer view of the shared opts: identical to `base` unless a frozen
// prefix is configured, in which case it names that layer's store.
static inline LmLayerOpts lm_ckpt_layer_kv(const LmCkptState & st, const LmLayerOpts & base, int l) {
    if (st.cfg.pfx) {
        LmLayerOpts o = base;
        o.pfx_k       = st.cfg.pfx->k[(size_t) l];
        o.pfx_v       = st.cfg.pfx->v[(size_t) l];
        o.pfx_zero    = st.cfg.pfx->zero;
        o.pfx_n       = st.cfg.pfx->n;
        return o;
    }
    if (!st.cfg.kv) {
        return base;
    }
    LmLayerOpts o = base;
    o.kv_k        = st.cfg.kv->k[(size_t) l];
    o.kv_v        = st.cfg.kv->v[(size_t) l];
    o.kv_pfx      = st.cfg.kv->n;
    return o;
}

// `adapter_off` is a PER-PASS argument rather than an LmCkptCfg field on
// purpose: it is the one option that legitimately differs between two graphs of
// the same run (the live teacher's forward vs the student's), and a cfg field
// would have to be mutated and restored around every teacher pass — exactly the
// shape of bug D13 exists to prevent. Defaulted, so every existing call site is
// unchanged.
static inline LmLayerOpts lm_ckpt_layer_opts(const LmCkptState & st, bool adapter_off = false) {
    LmLayerOpts o;
    o.adapter_off     = adapter_off;
    o.cast_weights    = true;
    o.attn_head_block = st.cfg.attn_head_block;
    o.attn_zero       = st.t_zero_attn;
    o.weights_bf16    = st.cfg.weights_bf16;
    o.rank_mask       = st.cfg.rank_mask;
    o.attn_flash      = st.cfg.attn_flash;
    o.attn_prec       = st.cfg.attn_prec;
    // o.wt stays nullptr: P2/P3 are forward-only, so a transposed weight there
    // would be a node nothing consumes. Only the P7 backward segment sets it.
    return o;
}

/** Read one chunk's logits back and append each row's top-K, as probabilities
 *  under the FULL softmax.
 *
 *  Softmax is taken over the WHOLE scored width and only then truncated to K, so
 *  what lands here is the base's real confidence in those K codes and the row
 *  sums to the captured mass, not to 1. That is deliberate, and it is a CAPTURE
 *  format, not a training label:
 *
 *    * the row sum is the coverage diagnostic — how much of the base this K
 *      actually recorded — and it is what gets cached to disk and reported;
 *    * a consumer that feeds these rows to ggml_cross_entropy_loss MUST
 *      bring each row to sum 1 first. ggml's backward is
 *      (softmax(z) - labels) * d/nr, which is the true gradient only for labels
 *      summing to 1; an unnormalised row adds (1 - S) * softmax_j, i.e. the
 *      gradient of a logit-scale term that changes no probability at all.
 *
 *  HOW the row is completed matters. Rescaling the K to sum to 1 makes the
 *  target SHARPER than the base — the dropped tail is asked for probability 0
 *  and the surviving K absorb its mass; at MM3's >99% coverage that is a
 *  rounding error, at ACE's 18-24% (K=64 over 217,204) it is a different and
 *  peakier teacher. The ACE trainer therefore keeps the K at their captured
 *  probabilities and spreads the missing mass flat over the audio-code range
 *  (LmSample::soft_tail in train/lm-data.h; train/lm-train-run.h), and the
 *  producer logs the measured coverage and warns below 50%. */
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

/** LIVE TEACHER: fill t_labc with one chunk's frozen-base distribution.
 *
 *  Runs entirely on the device — the alternative was reading V floats per
 *  position back to the host (869 KB x s_tr, ~2.6 GB per reg step at 4B) and
 *  exp()-ing 229 M values single-threaded, which is seconds per step against
 *  the ~1 ms this graph costs. There is no host round trip and no per-token
 *  host buffer anywhere in the live path.
 *
 *  Four nodes and a copy:
 *
 *    lg  = mul_mat(head, teach_h[:, chunk])  the base's logits for this chunk,
 *                                            re-derived from the stash the
 *                                            adapter-off pass left behind
 *    msk = mul_mat(codemask, sel)            outer product: -1e30 outside
 *                                            [teacher_lo, teacher_hi) at the
 *                                            positions sel marks, 0 elsewhere
 *    soft_max(lg + msk)                      -> rows sum to 1
 *    cpy -> t_labc[:, 0..Sc)                 the dense [V, Sc] label block the
 *                                            student's CE head reads
 *
 *  WHY THE MASK IS PER POSITION AND NOT A CONSTANT. The obvious reading of
 *  "softmax over the audio-code range only" is a [V,1] mask broadcast over the
 *  chunk — and it is wrong here, because with `--loss-on-cot` (the default) the
 *  supervised span of a reg row starts at the <think> token, so a large share
 *  of its positions are CoT YAML TEXT, not codes. Measured on the boston reg
 *  corpus at 4B: the base puts a mean of only 8.5% of its probability mass on
 *  the code range across supervised positions (the step-0 check's reg CE
 *  exceeded the teacher's own entropy by 2.47 nats), which is what a set of
 *  positions that are mostly not code positions looks like. A code-only target
 *  there is not "the base restricted"; it is a renormalisation of 65,535
 *  logits the base has driven to near-zero — arbitrary noise — and worse, it
 *  asks for total code probability 1.0 at a position where the base emits
 *  text. On one micro-step in --reg-every that is a standing push toward
 *  emitting codes instead of a plan, i.e. the exact degeneracy prior
 *  preservation exists to prevent.
 *
 *  So: `sel[i]` is 1 exactly when the row's own ground-truth token at that
 *  position IS an audio code, and the range mask applies only there. At every
 *  other position the teacher is the base's full-width distribution, unaltered.
 *  Both are the frozen base's real answer; neither is invented.
 *
 *  `off` is the chunk's first supervised position and is threaded through the
 *  SAME `s.n_masked - 1 + off` column arithmetic the student head uses, so a
 *  teacher row cannot land against the wrong student position. `s.targets[off
 *  + i]` is the label for the same position — one indexing convention, used by
 *  both.
 *
 *  Rows summing to 1 is what makes ggml's cross-entropy backward
 *  (softmax - labels) the true gradient — the reason the cached path has to
 *  renormalise its top-K with a flat tail, and the reason this path needs no
 *  tail at all: it IS a full distribution. */
static bool lm_ckpt_teacher_labels(LmCkptRun & r, const LmSample & s, int off, int Sc, int V) {
    LmCkptState &         st = *r.st;
    const Qwen3LMConfig & c  = r.lm->cfg;
    const int             H  = c.hidden_size;
    GGML_ASSERT(st.t_teach_h && st.t_codemask && st.t_teach_sel && "--reg-teacher live was not allocated");
    GGML_ASSERT(off + Sc <= (int) s.targets.size() && "the live teacher needs the row's own targets");

    // Which of this chunk's positions are code positions. Sc floats over PCIe.
    {
        std::vector<float> sel((size_t) Sc, 0.0f);
        for (int i = 0; i < Sc; i++) {
            const int32_t t = s.targets[(size_t) (off + i)];
            sel[(size_t) i] = (t >= st.cfg.teacher_lo && t < st.cfg.teacher_hi) ? 1.0f : 0.0f;
        }
        ggml_backend_tensor_set(st.t_teach_sel, sel.data(), 0, sel.size() * sizeof(float));
    }

    ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 64, /*grads=*/false);

    const size_t  col = (size_t) (s.n_masked - 1 + off);  // [P] L6a, same as the student head
    ggml_tensor * hd =
        ggml_cont(ctx, ggml_view_2d(ctx, st.t_teach_h, H, Sc, st.t_teach_h->nb[1], col * st.t_teach_h->nb[1]));
    ggml_tensor * hw = lm_ckpt_head_src(r.lm, st.cfg);
    if (st.cfg.head_w) {
        hw = ggml_view_2d(ctx, hw, hw->ne[0], (int64_t) V, hw->nb[1], (size_t) st.cfg.head_row0 * hw->nb[1]);
    }
    ggml_tensor * emb = r.head_f32_embed ? qwen3_f32(ctx, hw) : hw;
    ggml_tensor * lg  = ggml_mul_mat(ctx, emb, hd);  // [V, Sc]
    // [1,V] x [1,Sc] -> [V,Sc]: the k = 1 outer product of the class mask with
    // the position selector.
    ggml_tensor * selv = ggml_view_2d(ctx, st.t_teach_sel, 1, Sc, st.t_teach_sel->nb[1], 0);
    ggml_tensor * msk  = ggml_mul_mat(ctx, st.t_codemask, selv);
    ggml_tensor * pr   = ggml_soft_max(ctx, ggml_add(ctx, lg, msk));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, pr, ggml_view_2d(ctx, st.t_labc, V, Sc, st.t_labc->nb[1], 0)));

    ggml_backend_sched_reset(r.sched);
    const bool ok = lm_phase_compute(r.sched, gf);
    ggml_free(ctx);
    return ok;
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

        // --reg-teacher live: the frozen base's own distribution for THIS
        // chunk, written straight into t_labc on the device. The guard is then
        // told the buffer is externally owned so it neither writes nor clears
        // it; the whole buffer is cleared once when the head is done, which is
        // what the next hard-label step's "starts from zero" assumption needs.
        if (r.teacher_use && !lm_ckpt_teacher_labels(r, s, i, Sc, V)) {
            ggml_backend_buffer_clear(st.buf_labc, 0);
            return false;
        }
        LmChunkLabelGuard guard(st.t_labc, s, i, Sc, V, /*external=*/r.teacher_use);

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
        if (r.capture_k > 0 || r.logit_sink) {
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
        if (ok && r.logit_sink) {
            r.logit_sink(r, lg, V, Sc, i, r.logit_user);
        }
        ggml_free(ctx);
        if (!ok) {
            if (r.teacher_use) {
                ggml_backend_buffer_clear(st.buf_labc, 0);
            }
            return false;
        }
    }
    // The live teacher wrote dense blocks the guard did not clear. One
    // whole-buffer clear per micro-step (a device memset of V*chunk floats)
    // restores the invariant every other label path assumes: t_labc is zero
    // before its first write.
    if (r.teacher_use) {
        ggml_backend_buffer_clear(st.buf_labc, 0);
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
//
// `op_hash` (self-test T22) is filled with lm_graph_op_hash(gf) when non-null.
// Additive and default-null: it reads the finished graph and changes nothing.
static int lm_ckpt_probe_segment_nodes(LmCkptRun & r, int S, LmBf16Counts * counts = nullptr,
                                       uint64_t * op_hash = nullptr) {
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
    // Worst case is a FULL prefix: it adds the splice nodes to every segment.
    const int     nkv   = st.cfg.kv ? (int) st.cfg.kv->q_max + S : (st.cfg.pfx ? st.cfg.pfx->n + S : S);
    if (st.cfg.kv) {
        opts.kv_k   = st.cfg.kv->k[(size_t) l];
        opts.kv_v   = st.cfg.kv->v[(size_t) l];
        opts.kv_pfx = (int) st.cfg.kv->q_max;
    } else if (st.cfg.pfx) {
        opts.pfx_k    = st.cfg.pfx->k[(size_t) l];
        opts.pfx_v    = st.cfg.pfx->v[(size_t) l];
        opts.pfx_zero = st.cfg.pfx->zero;
        opts.pfx_n    = st.cfg.pfx->n;
    }
    ggml_tensor * mask  = ggml_view_2d(ctx, r.t_msk, nkv, S, (size_t) nkv * ggml_element_size(r.t_msk), 0);
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
    if (op_hash) {
        *op_hash = lm_graph_op_hash(gf);
    }
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
    lm_ckpt_upload_mask(r, S, s.n_prompt >= 0 ? s.n_prompt : s.n_masked);
    const int NKV = lm_ckpt_n_kv(st, S);
    // `tokens` still SIZES the sequence, but with an embedding override its
    // contents are meaningless and t_tok is never read — do not pretend.
    if (!r.embed_build || r.embed_uses_tok) {
        ggml_backend_tensor_set(r.t_tok, s.tokens.data(), 0, (size_t) S * 4);
    }
    if (!r.pos_external) {
        for (int i = 0; i < S; i++) {
            st.pos_scratch[(size_t) i] = i;
        }
        ggml_backend_tensor_set(r.t_pos, st.pos_scratch.data(), 0, (size_t) S * 4);
    }

    // ONE opts for the whole micro-step (D13). `r.adapter_off` is the only
    // field that can differ between micro-steps of the same run, and it is read
    // exactly here, so P2, P3 and P7 cannot disagree about the adapter.
    const LmLayerOpts opts = lm_ckpt_layer_opts(st, r.adapter_off);

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
        ggml_tensor *    mv  = ggml_view_2d(ctx, r.t_msk, NKV, S, (size_t) NKV * ggml_element_size(r.t_msk), 0);
        ggml_tensor *    X   = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
        // D13: SAME opts as the P7 recompute. Not an optimisation opportunity.
        ggml_tensor * Y = lm_train_layer(ctx, c, &lm.layers[l], X, pv, mv, S, lm_ckpt_layer_kv(st, opts, l));
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
        ggml_tensor *    mv  = ggml_view_2d(ctx, r.t_msk, NKV, S, (size_t) NKV * ggml_element_size(r.t_msk), 0);
        ggml_tensor *    X   = ggml_view_2d(ctx, st.C[(size_t) (Hi - 1)], H, S, st.C[(size_t) (Hi - 1)]->nb[1], 0);
        ggml_tensor *    Y   = lm_train_layer(ctx, c, &lm.layers[Hi - 1], X, pv, mv, S, lm_ckpt_layer_kv(st, opts, Hi - 1));
        ggml_tensor *    hN  = lm_rms(ctx, Y, lm.final_norm, c.rms_norm_eps);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, hN, ggml_view_2d(ctx, st.t_H, H, S, st.t_H->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // ── LIVE TEACHER: the pass ends here ─────────────────────────────────
    //
    // Stash the adapter-off final hidden states and get out BEFORE P4. That
    // ordering is the contract: this pass clears no gradient buffer, runs no
    // head, builds no backward and touches no accumulator, so the student pass
    // that follows starts from exactly the state it would have started from
    // without it. The only thing it leaves behind is C[] / t_H, which the
    // student's own P1-P3 overwrites — hence the stash.
    if (r.teacher_fill) {
        GGML_ASSERT(r.st->t_teach_h && "teacher_fill without --reg-teacher live allocation");
        ggml_backend_tensor_copy(st.t_H, st.t_teach_h);  // device-to-device, same shape
        return true;
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

    // Auxiliary loss head (see the field comment): adds into t_G between the
    // CE head and the backward segments — the only window where that is legal.
    if (r.aux_head && !r.aux_head(r, s, r.aux_user)) {
        return false;
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
        ggml_tensor * mv = ggml_view_2d(ctx, r.t_msk, NKV, S, (size_t) NKV * ggml_element_size(r.t_msk), 0);
        ggml_tensor * X  = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
        ggml_tensor * Y  = lm_train_layer(ctx, c, &lm.layers[l], X, pv, mv, S, lm_ckpt_layer_kv(st, sopts, l));
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
    // ── P1B: backward over the embedding stage ───────────────────────────
    //
    // Gh[cur] holds dL/d(embedding output). It is DISCARDED in the ordinary
    // case — embed_tokens is frozen and carries no LoRA site — but when
    // `embed_trainable` says the builder contains a parameter (an artist token),
    // that discarded tensor is precisely the seed its gradient needs.
    //
    // Same surrogate as every segment above: L' = SUM(h0 (.) dY) has gradient
    // w.r.t. h0 of exactly dY, so backprop from L' gives the parameter its true
    // gradient and the loss seed is 1.0, NOT the micro-batch scale — dY already
    // carries it. Seeding again would square it.
    //
    // No ckpt_in/g_out here: nothing below the embedding needs a gradient, so
    // fill_gacc is handed nullptr for both and its `nd == ckpt_in` test simply
    // never fires.
    if (r.embed_trainable && r.embed_build && !r.forward_only) {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 512, /*grads=*/true);
        ggml_tensor *    h0  = r.embed_build(ctx, r, S);
        ggml_tensor *    dY  = ggml_view_2d(ctx, st.Gh[cur], H, S, st.Gh[cur]->nb[1], 0);
        ggml_tensor *    Lsur = ggml_sum(ctx, ggml_mul(ctx, h0, dY));
        ggml_set_loss(Lsur);
        ggml_set_output(Lsur);
        ggml_build_forward_expand(gf, Lsur);
        lm_ckpt_fill_gacc(r.opt, gf, nullptr, nullptr, r.t_one, &st.gacc);
        ggml_build_backward_expand(ctx, gf, st.gacc.data());
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }
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
