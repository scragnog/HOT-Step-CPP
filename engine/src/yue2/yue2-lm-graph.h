#pragma once
// yue2/yue2-lm-graph.h — YuE2 AR (causal) forward graph. Milestones M2/M3
// per docs/plans/yue2/06-engine-port-plan.md §3.
//
// HOT-Step file (does not exist upstream). SCOPE OF THIS FILE: the AR half of
// one decoder layer (blk.N.* — never the nar_* Mixture-of-Transformers twin,
// which has no consumer yet), assembled into a teacher-forced, FULL-SEQUENCE,
// SINGLE-ROW forward (no CFG batching — see the note below on why not) that
// returns full-vocab logits and/or all-30-layer hidden-state taps at whatever
// positions the caller asks for. This is bring-up/parity tooling for
// `yue2-probe --ar-parity` against docs/plans/yue2/02-fixture-schema.md v1.4
// fixtures — NOT the production decode loop (incremental per-token decode,
// a persistent multi-step KV cache, and real CFG all belong to a later
// milestone once generation itself is being built).
//
// ── What's reused from engine/src/minimax/mm3-lm-graph.h, and why it's safe ──
//
// YuE2's AR half is Qwen3-shaped (03-reference-numerics.md §2.2 confirms the
// exact same op order MM3's own Qwen3-8B LM uses): RMSNorm -> Q/K/V proj ->
// per-head QK-RMSNorm (dim=head_dim, eps 1e-6, BEFORE RoPE, never applied to
// V) -> NeoX half-split RoPE -> KV-cache write via ggml_set_rows (destination
// rows supplied as data) -> attention -> residual add -> RMSNorm -> SwiGLU
// (down(silu(gate)*up)) -> residual add. `yue2_ar_block` below copies that
// shape verbatim from `mm3_lm_block`, stripped of everything MM3-only that
// YuE2 v1 has no equivalent for: RUNTIME LoRA/LoKr adapters (still none — but
// note that since phase 3 of docs/plans/yue2/08-nar-lora-trainer.md, YuE2 does
// have MERGE-at-load adapters in yue2-adapter.h, which bake the delta into the
// weights before wctx_alloc and so need no graph support at all), the
// artist-token soft prompt, the ensemble-take/CFG batch axis,
// and the mul-fold-rows decode-step optimization (fold_rows only pays off at
// T=1 per-step decode, which this one-shot full-sequence forward never does).
// `yue2_lm_attn_f32` is `mm3_lm_attn_f32` with the batch axis dropped (always
// B=1 here) and the alignment-probe output tap removed (nothing reads it yet).
//
// ── Where this MUST diverge from mm3-lm-graph.h (load-bearing) ──
//
// MM3's 2-row CFG batch (mm3-lm-graph.h note A) shares one `kv_pos`/position
// counter across both rows because MM3's cond/uncond prompts are the SAME
// length by construction. YuE2's are not: `negative_prefix()` drops the
// [Tags]/style/[Lyrics]/lyrics block entirely (03-reference-numerics.md §1.4,
// engine-port-plan.md §0/§3), so the positive and negative branches sit at
// DIFFERENT lengths and therefore different absolute RoPE positions at the
// "same" decode step. The engine-port-plan's own recommendation (§3, design
// 2) is followed here: CFG is two INDEPENDENT single-row forward calls, each
// with its own KV cache and its own local position stream 0..T-1 — never one
// batched ne3=2 graph. `yue2_ar_forward` is written as a self-contained
// single-sequence forward for exactly this reason; a caller wanting the CFG
// comparison calls it twice (yue2-probe --ar-parity does, for the semantic
// stage), once per branch's own `final_ids`.
//
// ── Precision policy ──
//
// F32 activations throughout (RMSNorm's own internal mean-square reduction is
// already fp32 per the reference, 03-reference-numerics.md §2.2 — everything
// else here matches that by running the whole graph at F32), BF16 weights
// (the GGUF's NATIVE policy, loaded as-is by yue2-model.h), F16 KV cache
// (same trade as mm3-lm-graph.h note G: the reference itself runs end-to-end
// BF16, so F32 activations are strictly MORE precise than the thing this is
// validated against, never less).
//
// ── One-shot KV cache, no persistence, no padding, no NaN risk ──
//
// Unlike MM3's LM graph (whose KV cache spans many decode steps and must
// therefore be padded ahead of the written region, then explicitly zeroed to
// avoid a masked-but-uninitialized F16 NaN, per mm3-lm-graph.h note B), this
// module's cache is sized to EXACTLY T (this call's sequence length) and
// every one of its T rows is written by this same call's `ggml_set_rows`
// before anything reads it — there is no unwritten padding region to zero.
// The cache is allocated fresh per call and freed at the end of the call; no
// state survives across `yue2_ar_forward` invocations. This is deliberately
// the simplest correct thing for a bring-up/parity tool, not a performance
// design — a production incremental decode loop needs MM3's own persistent,
// padded, bucketed cache shape instead.

#include "yue2-imatrix.h"
#include "yue2-model.h"

#include "backend.h"
#include "ggml.h"
#include "hot-step-build-flags.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 28 layers x ~20 nodes (no adapter/CFG-batch/fold-rows branching to inflate
// it) + up to 30 hidden-tap gathers + embed/head-gather/logits plumbing.
// Loose on purpose, same posture as MM3_LM_MAX_NODES.
#define YUE2_LM_MAX_NODES 4096

// YUE2_LM_NO_FLASH=1 forces the manual F32 soft_max attention path — a
// parity-debug escape hatch (mirrors MM3_LM_NO_FLASH), not a production knob.
static inline bool yue2_lm_use_flash(ggml_backend_t backend) {
    if (HOT_STEP_FA_DISABLED) {
        return false;
    }
    if (!backend || strcmp(ggml_backend_name(backend), "CPU") == 0) {
        return false;  // the manual path works everywhere; flash CUDA kernels do not run on CPU
    }
    static const bool disabled_env = [] {
        const char * e = std::getenv("YUE2_LM_NO_FLASH");
        return e && e[0] && e[0] != '0';
    }();
    return !disabled_env;
}

static ggml_tensor * yue2_lm_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Manual F32 attention fallback. q [D,T,Nh,1], k/v [D,n_kv,Nkv,1] (cache
// views) -> [D,Nh,T,1]. Copy of mm3_lm_attn_f32's shape (mm3-lm-graph.h),
// batch axis dropped (YuE2's AR forward here is always a single row) and the
// alignment-probe score tap removed (nothing consumes it yet).
static ggml_tensor * yue2_lm_attn_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                      ggml_tensor * mask, float scale) {
    if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) {
        k = ggml_cast(ctx, k, GGML_TYPE_F32);
        v = ggml_cast(ctx, v, GGML_TYPE_F32);
    }
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);                     // [n_kv, T, Nh, 1]
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));       // [n_kv, D, Nkv, 1]
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);                // [D, T, Nh, 1]
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));           // [D, Nh, T, 1]
}

// One AR transformer block (blk.N.* — the shared AR half, never the nar_*
// twin). h [H,T] -> [H,T]. Copy of mm3_lm_block's math (mm3-lm-graph.h),
// stripped of runtime adapters, the ensemble/CFG batch axis, and the
// mul-fold-rows decode optimization — none of it exists (or applies) here.
static ggml_tensor * yue2_ar_block(ggml_context * ctx, ggml_cgraph * gf, const Yue2LmConfig & c,
                                   const Yue2LmLayer & w, ggml_tensor * h, ggml_tensor * positions, ggml_tensor * mask,
                                   ggml_tensor * rows, ggml_tensor * kcache, ggml_tensor * vcache, int64_t n_kv_pad,
                                   bool use_flash) {
    const int64_t H   = (int64_t) c.embedding_length;
    const int64_t D   = (int64_t) c.key_length;
    const int64_t Nh  = (int64_t) c.head_count;
    const int64_t Nkv = (int64_t) c.head_count_kv;
    const int64_t T   = h->ne[1];

    ggml_tensor * n = yue2_lm_rms(ctx, h, w.attn_norm, c.rms_eps);

    ggml_tensor * q = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.attn_q, n), D, Nh, T, 1);   // [D,Nh,T,1]
    ggml_tensor * k = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.attn_k, n), D, Nkv, T, 1);  // [D,Nkv,T,1]
    ggml_tensor * v = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.attn_v, n), D, Nkv, T, 1);

    // Per-head QK RMSNorm (dim=head_dim=128, eps 1e-6), strictly BEFORE RoPE.
    // No norm on V (03-reference-numerics.md §2.2, Attention.project_qkv).
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_eps), w.attn_q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_eps), w.attn_k_norm);

    // NeoX half-split rotation (x1=first half, x2=second half — the
    // reference's own convention, §2.2's _apply_rotary), theta = rope_freq_base
    // (1e6). Position_ids == cache_position always in this one-shot,
    // non-CFG-batched forward (§2.2's own "unexercised padding branch" note —
    // there is no padding here, so this is exactly the traced behavior, not an
    // approximation of it).
    q = ggml_rope_ext(ctx, q, positions, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                       0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                       0.0f, 0.0f);

    ggml_tensor * k_w = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [D,T,Nkv,1]
    ggml_tensor * v_w = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kcache, k_w, rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, vcache, v_w, rows));

    ggml_tensor * k_win =
        ggml_view_4d(ctx, kcache, D, n_kv_pad, Nkv, 1, kcache->nb[1], kcache->nb[2], kcache->nb[3], 0);
    ggml_tensor * v_win =
        ggml_view_4d(ctx, vcache, D, n_kv_pad, Nkv, 1, vcache->nb[1], vcache->nb[2], vcache->nb[3], 0);

    ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D,T,Nh,1]

    // yue2.attention.softmax_scale KV, already 128^-0.5 (03-reference-numerics.md
    // §2.8 — no call site overrides PyTorch's default in either model).
    const float   scale = c.softmax_scale;
    ggml_tensor * attn;
    if (use_flash) {
        attn = ggml_flash_attn_ext(ctx, q4, k_win, v_win, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = yue2_lm_attn_f32(ctx, q4, k_win, v_win, mask, scale);
    }
    attn = ggml_reshape_2d(ctx, attn, H, T);  // [D,Nh,T,1] -> [H,T]

    h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.attn_output, attn));

    // SwiGLU: down(silu(gate) * up). No bias anywhere (03-reference-numerics.md §2.2).
    ggml_tensor * n2   = yue2_lm_rms(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, w.ffn_gate, n2));
    ggml_tensor * up   = ggml_mul_mat(ctx, w.ffn_up, n2);
    return ggml_add(ctx, h, ggml_mul_mat(ctx, w.ffn_down, ggml_mul(ctx, gate, up)));
}

// ── Public API ───────────────────────────────────────────────────────────

struct Yue2ArForwardRequest {
    std::vector<int32_t> ids;              // T token ids, the exact sequence to teacher-force
    std::vector<int64_t> logit_positions;   // rows (0-indexed into ids) to compute full-vocab lm_head logits at;
                                             // logits at row p are "the forward's output that PREDICTS token p+1"
    std::vector<int64_t> hidden_positions;  // rows to gather all 30 layer-taps' hidden states at
                                             // (0=embed_tokens output, 1..28=post-block i-1, 29=final norm,
                                             // per 02-fixture-schema.md §2.7)
};

struct Yue2ArForwardResult {
    int64_t T = 0;
    int64_t V = 0;  // vocab_size, only meaningful if logits is non-empty
    int64_t H = 0;  // embedding_length, only meaningful if hidden is non-empty
    std::vector<float> logits;  // [logit_positions.size(), V] row-major (position-major)
    std::vector<float> hidden;  // [hidden_positions.size(), 30, H] row-major, per §2.7's layer-axis convention
};

// One self-contained teacher-forced forward: builds a fresh KV cache and a
// fresh compute graph sized exactly to req.ids.size(), runs the full 28-layer
// AR stack under a full causal mask, and reads back logits/hidden-state taps
// ONLY at the requested positions (never the whole sequence — the lm_head
// matmul and the 30-way hidden gather are both restricted to the requested
// rows, which is what keeps a ~9-position pinned-fixture check cheap even at
// T in the thousands). Every call is independent: no KV cache or graph state
// survives past the call, matching this module's one-shot/bring-up scope
// (see the file header for why, and what a production decode loop needs
// instead). `m.lm_resident` must already be true (yue2_load_parts).
static bool yue2_ar_forward(const Yue2Model & m, const Yue2ArForwardRequest & req, Yue2ArForwardResult * out,
                            std::string * err) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    const int64_t T = (int64_t) req.ids.size();
    if (T <= 0) {
        if (err) {
            *err = "yue2_ar_forward: empty ids";
        }
        return false;
    }
    const Yue2LmConfig & c   = m.lm_cfg;
    const int64_t        H   = (int64_t) c.embedding_length;
    const int64_t        D   = (int64_t) c.key_length;
    const int64_t        Nkv = (int64_t) c.head_count_kv;
    const int64_t        V   = (int64_t) c.vocab_size;
    const int             L   = (int) c.block_count;
    const int64_t         Kh  = (int64_t) req.hidden_positions.size();
    const int64_t         Kl  = (int64_t) req.logit_positions.size();

    // ── one-shot KV cache: exactly T rows, every row written by this call ──
    ggml_init_params kv_ip = { (size_t) (L * 2) * ggml_tensor_overhead() + 1024, NULL, /*no_alloc*/ true };
    ggml_context *   kv_ctx = ggml_init(kv_ip);
    if (!kv_ctx) {
        if (err) {
            *err = "ggml_init failed for the YuE2 AR KV cache context";
        }
        return false;
    }
    std::vector<ggml_tensor *> kv_k((size_t) L, nullptr), kv_v((size_t) L, nullptr);
    for (int i = 0; i < L; i++) {
        kv_k[(size_t) i] = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, D, T, Nkv, 1);
        kv_v[(size_t) i] = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F16, D, T, Nkv, 1);
    }
    ggml_backend_buffer_t kv_buf = ggml_backend_alloc_ctx_tensors(kv_ctx, m.backend);
    if (!kv_buf) {
        ggml_free(kv_ctx);
        if (err) {
            *err = "backend buffer allocation failed for the YuE2 AR KV cache (out of VRAM?)";
        }
        return false;
    }
    // Every row [0,T) is written by this same call's ggml_set_rows before
    // anything reads it (see file header) — this clear is defensive, not
    // load-bearing, but it is nearly free next to the compute below and it
    // matches mm3-lm-graph.h's own posture of never trusting uninitialized
    // F16 bit patterns near a softmax.
    ggml_backend_buffer_clear(kv_buf, 0);

    // ── compute graph ──
    const size_t ctx_bytes =
        ggml_tensor_overhead() * (YUE2_LM_MAX_NODES + 256) + ggml_graph_overhead_custom(YUE2_LM_MAX_NODES, false);
    uint8_t * gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!gbuf) {
        ggml_backend_buffer_free(kv_buf);
        ggml_free(kv_ctx);
        if (err) {
            *err = "out of host memory allocating the YuE2 AR compute graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(gbuf);
        ggml_backend_buffer_free(kv_buf);
        ggml_free(kv_ctx);
        if (err) {
            *err = "ggml_init failed for the YuE2 AR compute graph context";
        }
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, YUE2_LM_MAX_NODES, false);

    ggml_tensor * in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(in_ids, "yue2_ar_ids");
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(in_pos, "yue2_ar_positions");
    ggml_set_input(in_pos);
    ggml_tensor * in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    ggml_set_name(in_rows, "yue2_ar_kv_rows");
    ggml_set_input(in_rows);
    ggml_tensor * in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);  // [n_kv=T, T]
    ggml_set_name(in_mask, "yue2_ar_mask");
    ggml_set_input(in_mask);

    ggml_tensor * in_hidx = nullptr;
    if (Kh > 0) {
        in_hidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Kh);
        ggml_set_name(in_hidx, "yue2_ar_hidden_idx");
        ggml_set_input(in_hidx);
    }
    ggml_tensor * in_lidx = nullptr;
    if (Kl > 0) {
        in_lidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Kl);
        ggml_set_name(in_lidx, "yue2_ar_logit_idx");
        ggml_set_input(in_lidx);
    }

    ggml_tensor * h = ggml_get_rows(ctx, m.lm.token_embd, in_ids);  // [H,T]

    // 30-slot layer-tap axis, per 02-fixture-schema.md §2.7: 0=embed_tokens
    // output, 1..28=post-block i-1, 29=final norm. Every slot is a
    // full-sequence [H,T] tensor kept alive purely by the gather node reading
    // it below (ggml keeps a node's graph dependencies alive automatically —
    // nothing here needs an explicit "keep" beyond that reference).
    std::vector<ggml_tensor *> taps((size_t) L + 2, nullptr);
    taps[0] = h;
    const bool use_flash = yue2_lm_use_flash(m.backend);
    for (int i = 0; i < L; i++) {
        h        = yue2_ar_block(ctx, gf, c, m.lm.blk[(size_t) i], h, in_pos, in_mask, in_rows, kv_k[(size_t) i],
                                 kv_v[(size_t) i], T, use_flash);
        taps[(size_t) i + 1] = h;
    }
    ggml_tensor * h_final = yue2_lm_rms(ctx, h, m.lm.output_norm, c.rms_eps);
    taps[(size_t) L + 1]  = h_final;

    std::vector<ggml_tensor *> hidden_taps;
    if (Kh > 0) {
        hidden_taps.assign(taps.size(), nullptr);
        for (size_t l = 0; l < taps.size(); l++) {
            ggml_tensor * g = ggml_get_rows(ctx, taps[l], in_hidx);  // [H,Kh]
            ggml_set_name(g, ("yue2_ar_hidden_tap_" + std::to_string(l)).c_str());
            ggml_set_output(g);
            ggml_build_forward_expand(gf, g);
            hidden_taps[l] = g;
        }
    }
    ggml_tensor * out_logits = nullptr;
    if (Kl > 0) {
        ggml_tensor * gathered = ggml_get_rows(ctx, h_final, in_lidx);       // [H,Kl]
        out_logits             = ggml_mul_mat(ctx, m.lm.output, gathered);  // [V,Kl] — norm already applied to
                                                                             // the FULL sequence before this
                                                                             // gather, matching Backbone.forward()
                                                                             // returning self.norm(x) over every
                                                                             // position before YuE2ForCausalLM
                                                                             // slices the requested rows
                                                                             // (03-reference-numerics.md §2.2).
        ggml_set_name(out_logits, "yue2_ar_logits");
        ggml_set_output(out_logits);
        ggml_build_forward_expand(gf, out_logits);
    }

    BackendPair           bp    = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
    ggml_backend_sched_t sched = backend_sched_new(bp, YUE2_LM_MAX_NODES);
    bool                 ok    = sched && ggml_backend_sched_alloc_graph(sched, gf);
    if (!ok) {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        ggml_free(ctx);
        free(gbuf);
        ggml_backend_buffer_free(kv_buf);
        ggml_free(kv_ctx);
        if (err) {
            *err = "YuE2 AR graph allocation failed (out of VRAM?) at T=" + std::to_string((long long) T);
        }
        return false;
    }

    // ── host-side inputs ──
    ggml_backend_tensor_set(in_ids, req.ids.data(), 0, (size_t) T * sizeof(int32_t));

    std::vector<int32_t> pos_host((size_t) T);
    std::vector<int64_t> rows_host((size_t) T);
    for (int64_t i = 0; i < T; i++) {
        pos_host[(size_t) i]  = (int32_t) i;  // position_ids == cache_position always here (§2.2)
        rows_host[(size_t) i] = i;
    }
    ggml_backend_tensor_set(in_pos, pos_host.data(), 0, (size_t) T * sizeof(int32_t));
    ggml_backend_tensor_set(in_rows, rows_host.data(), 0, (size_t) T * sizeof(int64_t));

    std::vector<uint16_t> mask_host((size_t) (T * T));
    for (int64_t i = 0; i < T; i++) {       // i = query position
        for (int64_t j = 0; j < T; j++) {   // j = key position
            mask_host[(size_t) (i * T + j)] = ggml_fp32_to_fp16(j <= i ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(in_mask, mask_host.data(), 0, (size_t) (T * T) * sizeof(uint16_t));

    if (in_hidx) {
        std::vector<int32_t> hidx((size_t) Kh);
        for (int64_t i = 0; i < Kh; i++) {
            hidx[(size_t) i] = (int32_t) req.hidden_positions[(size_t) i];
        }
        ggml_backend_tensor_set(in_hidx, hidx.data(), 0, (size_t) Kh * sizeof(int32_t));
    }
    if (in_lidx) {
        std::vector<int32_t> lidx((size_t) Kl);
        for (int64_t i = 0; i < Kl; i++) {
            lidx[(size_t) i] = (int32_t) req.logit_positions[(size_t) i];
        }
        ggml_backend_tensor_set(in_lidx, lidx.data(), 0, (size_t) Kl * sizeof(int32_t));
    }

    ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (!ok) {
        if (err) {
            *err = "YuE2 AR graph compute failed";
        }
    } else {
        out->T = T;
        if (Kh > 0) {
            out->H = H;
            out->hidden.assign((size_t) (Kh * (int64_t) taps.size() * H), 0.0f);
            const int64_t n_layers = (int64_t) taps.size();  // 30
            std::vector<float> layer_buf((size_t) (Kh * H));
            for (int64_t l = 0; l < n_layers; l++) {
                ggml_backend_tensor_get(hidden_taps[(size_t) l], layer_buf.data(), 0,
                                        (size_t) (Kh * H) * sizeof(float));
                for (int64_t k = 0; k < Kh; k++) {
                    memcpy(out->hidden.data() + (size_t) (k * n_layers * H + l * H),
                           layer_buf.data() + (size_t) (k * H), (size_t) H * sizeof(float));
                }
            }
        }
        if (Kl > 0) {
            out->V = V;
            out->logits.assign((size_t) (Kl * V), 0.0f);
            ggml_backend_tensor_get(out_logits, out->logits.data(), 0, (size_t) (Kl * V) * sizeof(float));
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    free(gbuf);
    ggml_backend_buffer_free(kv_buf);
    ggml_free(kv_ctx);
    return ok;
}

// ── Persistent KV-cache decode (M4) ─────────────────────────────────────────
//
// yue2_ar_forward() above is a ONE-SHOT full-sequence forward: it allocates a
// KV cache sized exactly to that one call's T and throws it away at the end
// — correct for teacher-forced parity checking (M2/M3), but not the shape a
// real incremental generation loop uses. This section adds the other half:
// a KV cache that SURVIVES across calls (`Yue2ArKvCache`), a multi-token
// `yue2_ar_prefill()` that writes the first N rows of it, and a single-token
// `yue2_ar_decode_step()` that appends one row at a time — the actual
// production decode shape (prefill the prompt once, then one token per
// step), teacher-forceable for parity against M2's one-shot forward and
// against the fixture's own `StaticKVCache` dump (kv_layer0/kv_layer27,
// docs/plans/yue2/02-fixture-schema.md §5 step 7), or free-running for
// engine/tools/yue2-probe.cpp's --generate smoke test.
//
// Layout: `[D, capacity, Nkv, 1]` per layer, F16, exactly like
// yue2_ar_forward's own throwaway cache — contiguous, so a head's first `n`
// rows are a single contiguous span at byte offset `h*nb[2]`, which is what
// `yue2_ar_kv_cache_dump_layer` below relies on to reproduce the fixture's
// `[num_kv_heads, seq, head_dim]` (head-major) flatten order with a single
// `ggml_backend_tensor_get` per head — no transpose needed, because ggml's
// own `[D,T,Nkv,1]` axis order already puts head_dim fastest, then seq, then
// kv-head slowest, which IS `StaticKVCache`'s head-major convention
// (03-reference-numerics.md §2.6) read back linearly.
//
// One cache instance is only ever used by ONE branch (positive or negative)
// — CFG's two independent forward calls (yue2-lm-graph.h's file header, and
// engine-port-plan.md §3 design 2) means a caller doing CFG'd decode owns
// two separate `Yue2ArKvCache` instances, never one shared cache with a
// batch axis.

struct Yue2ArKvCache {
    int64_t                    capacity = 0;  // total rows this cache can ever hold
    int64_t                    filled   = 0;  // rows [0,filled) are valid/written
    ggml_context *              ctx      = nullptr;
    ggml_backend_buffer_t       buf      = nullptr;
    std::vector<ggml_tensor *> k, v;          // one F16 [D,capacity,Nkv,1] tensor per layer
};

static bool yue2_ar_kv_cache_alloc(const Yue2Model & m, int64_t capacity, Yue2ArKvCache * out, std::string * err) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    if (capacity <= 0) {
        if (err) {
            *err = "yue2_ar_kv_cache_alloc: capacity must be > 0";
        }
        return false;
    }
    const Yue2LmConfig & c   = m.lm_cfg;
    const int64_t         D   = (int64_t) c.key_length;
    const int64_t         Nkv = (int64_t) c.head_count_kv;
    const int             L   = (int) c.block_count;

    ggml_init_params ip = { (size_t) (L * 2) * ggml_tensor_overhead() + 1024, NULL, /*no_alloc*/ true };
    out->ctx             = ggml_init(ip);
    if (!out->ctx) {
        if (err) {
            *err = "ggml_init failed for the persistent YuE2 KV cache context";
        }
        return false;
    }
    out->k.assign((size_t) L, nullptr);
    out->v.assign((size_t) L, nullptr);
    for (int i = 0; i < L; i++) {
        out->k[(size_t) i] = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F16, D, capacity, Nkv, 1);
        out->v[(size_t) i] = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F16, D, capacity, Nkv, 1);
    }
    out->buf = ggml_backend_alloc_ctx_tensors(out->ctx, m.backend);
    if (!out->buf) {
        ggml_free(out->ctx);
        out->ctx = nullptr;
        if (err) {
            *err = "backend buffer allocation failed for the persistent YuE2 KV cache (out of VRAM?)";
        }
        return false;
    }
    ggml_backend_buffer_clear(out->buf, 0);  // defensive — every row gets written before any read, same posture as yue2_ar_forward's cache
    out->capacity = capacity;
    out->filled   = 0;
    return true;
}

static void yue2_ar_kv_cache_free(Yue2ArKvCache * c) {
    if (c->buf) {
        ggml_backend_buffer_free(c->buf);
        c->buf = nullptr;
    }
    if (c->ctx) {
        ggml_free(c->ctx);
        c->ctx = nullptr;
    }
    c->k.clear();
    c->v.clear();
    c->capacity = 0;
    c->filled   = 0;
}

// Reads back rows [0,n) of layer `layer`'s cached K/V, widened to f32, in
// StaticKVCache's own head-major flatten order ([num_kv_heads, n, head_dim],
// h slowest / t / d fastest) — directly comparable to the fixture's
// kv_layer{0,27}_{k,v}.bin after that file's own bf16-widen (both are the
// same flatten order; see the file-header note on why no transpose is
// needed). `n` must be <= cache.filled.
static bool yue2_ar_kv_cache_dump_layer(const Yue2ArKvCache & cache, int layer, int64_t n, std::vector<float> * k_out,
                                        std::vector<float> * v_out) {
    if (layer < 0 || (size_t) layer >= cache.k.size() || n <= 0 || n > cache.filled) {
        return false;
    }
    ggml_tensor * kt  = cache.k[(size_t) layer];
    ggml_tensor * vt  = cache.v[(size_t) layer];
    const int64_t D   = kt->ne[0];
    const int64_t Nkv = kt->ne[2];
    k_out->resize((size_t) (Nkv * n * D));
    v_out->resize((size_t) (Nkv * n * D));
    std::vector<uint16_t> tmp((size_t) (n * D));
    for (int64_t h = 0; h < Nkv; h++) {
        ggml_backend_tensor_get(kt, tmp.data(), (size_t) h * kt->nb[2], (size_t) (n * D) * sizeof(uint16_t));
        for (size_t i = 0; i < tmp.size(); i++) {
            (*k_out)[(size_t) (h * n * D) + i] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &tmp[i]);
        }
        ggml_backend_tensor_get(vt, tmp.data(), (size_t) h * vt->nb[2], (size_t) (n * D) * sizeof(uint16_t));
        for (size_t i = 0; i < tmp.size(); i++) {
            (*v_out)[(size_t) (h * n * D) + i] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &tmp[i]);
        }
    }
    return true;
}

// Multi-token prefill: runs `ids.size()` tokens through the full causal AR
// stack exactly like yue2_ar_forward()'s body, EXCEPT the K/V it writes land
// in `cache` (rows [0,ids.size())) instead of a throwaway per-call buffer,
// and `cache.filled` is advanced on success so a later yue2_ar_decode_step()
// call knows where to continue. Requested logit/hidden positions must be
// row indices into `ids` (same convention as Yue2ArForwardRequest). Must be
// the FIRST call made against a freshly-allocated `cache` (filled must be 0).
static bool yue2_ar_prefill(const Yue2Model & m, Yue2ArKvCache & cache, const std::vector<int32_t> & ids,
                            const std::vector<int64_t> & logit_positions,
                            const std::vector<int64_t> & hidden_positions, Yue2ArForwardResult * out,
                            std::string * err) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    if (cache.filled != 0) {
        if (err) {
            *err = "yue2_ar_prefill: cache is not empty (filled != 0) -- prefill must be the first call";
        }
        return false;
    }
    const int64_t T = (int64_t) ids.size();
    if (T <= 0 || T > cache.capacity) {
        if (err) {
            *err = "yue2_ar_prefill: ids.size() must be in (0, cache.capacity]";
        }
        return false;
    }
    const Yue2LmConfig & c   = m.lm_cfg;
    const int64_t         H   = (int64_t) c.embedding_length;
    const int64_t         V   = (int64_t) c.vocab_size;
    const int             L   = (int) c.block_count;
    const int64_t         Kh  = (int64_t) hidden_positions.size();
    const int64_t         Kl  = (int64_t) logit_positions.size();

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (YUE2_LM_MAX_NODES + 256) + ggml_graph_overhead_custom(YUE2_LM_MAX_NODES, false);
    uint8_t * gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!gbuf) {
        if (err) {
            *err = "out of host memory allocating the YuE2 prefill compute graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(gbuf);
        if (err) {
            *err = "ggml_init failed for the YuE2 prefill compute graph context";
        }
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, YUE2_LM_MAX_NODES, false);

    ggml_tensor * in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(in_pos);
    ggml_tensor * in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    ggml_set_input(in_rows);
    ggml_tensor * in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);  // [n_kv=T, T], causal
    ggml_set_input(in_mask);

    ggml_tensor * in_hidx = nullptr;
    if (Kh > 0) {
        in_hidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Kh);
        ggml_set_input(in_hidx);
    }
    ggml_tensor * in_lidx = nullptr;
    if (Kl > 0) {
        in_lidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Kl);
        ggml_set_input(in_lidx);
    }

    ggml_tensor * h = ggml_get_rows(ctx, m.lm.token_embd, in_ids);

    std::vector<ggml_tensor *> taps((size_t) L + 2, nullptr);
    taps[0]              = h;
    const bool use_flash = yue2_lm_use_flash(m.backend);
    for (int i = 0; i < L; i++) {
        h = yue2_ar_block(ctx, gf, c, m.lm.blk[(size_t) i], h, in_pos, in_mask, in_rows, cache.k[(size_t) i],
                          cache.v[(size_t) i], T, use_flash);
        taps[(size_t) i + 1] = h;
    }
    ggml_tensor * h_final = yue2_lm_rms(ctx, h, m.lm.output_norm, c.rms_eps);
    taps[(size_t) L + 1]  = h_final;

    std::vector<ggml_tensor *> hidden_taps;
    if (Kh > 0) {
        hidden_taps.assign(taps.size(), nullptr);
        for (size_t l = 0; l < taps.size(); l++) {
            ggml_tensor * g = ggml_get_rows(ctx, taps[l], in_hidx);
            ggml_set_output(g);
            ggml_build_forward_expand(gf, g);
            hidden_taps[l] = g;
        }
    }
    ggml_tensor * out_logits = nullptr;
    if (Kl > 0) {
        ggml_tensor * gathered = ggml_get_rows(ctx, h_final, in_lidx);
        out_logits             = ggml_mul_mat(ctx, m.lm.output, gathered);
        ggml_set_output(out_logits);
        ggml_build_forward_expand(gf, out_logits);
    }

    BackendPair           bp    = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
    ggml_backend_sched_t sched = backend_sched_new(bp, YUE2_LM_MAX_NODES);
    bool                 ok    = sched && ggml_backend_sched_alloc_graph(sched, gf);
    if (!ok) {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        ggml_free(ctx);
        free(gbuf);
        if (err) {
            *err = "YuE2 prefill graph allocation failed (out of VRAM?) at T=" + std::to_string((long long) T);
        }
        return false;
    }
    yue2_imatrix_hook(sched);

    ggml_backend_tensor_set(in_ids, ids.data(), 0, (size_t) T * sizeof(int32_t));
    std::vector<int32_t> pos_host((size_t) T);
    std::vector<int64_t> rows_host((size_t) T);
    for (int64_t i = 0; i < T; i++) {
        pos_host[(size_t) i]  = (int32_t) i;
        rows_host[(size_t) i] = i;
    }
    ggml_backend_tensor_set(in_pos, pos_host.data(), 0, (size_t) T * sizeof(int32_t));
    ggml_backend_tensor_set(in_rows, rows_host.data(), 0, (size_t) T * sizeof(int64_t));

    std::vector<uint16_t> mask_host((size_t) (T * T));
    for (int64_t i = 0; i < T; i++) {
        for (int64_t j = 0; j < T; j++) {
            mask_host[(size_t) (i * T + j)] = ggml_fp32_to_fp16(j <= i ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(in_mask, mask_host.data(), 0, (size_t) (T * T) * sizeof(uint16_t));

    if (in_hidx) {
        std::vector<int32_t> hidx((size_t) Kh);
        for (int64_t i = 0; i < Kh; i++) {
            hidx[(size_t) i] = (int32_t) hidden_positions[(size_t) i];
        }
        ggml_backend_tensor_set(in_hidx, hidx.data(), 0, (size_t) Kh * sizeof(int32_t));
    }
    if (in_lidx) {
        std::vector<int32_t> lidx((size_t) Kl);
        for (int64_t i = 0; i < Kl; i++) {
            lidx[(size_t) i] = (int32_t) logit_positions[(size_t) i];
        }
        ggml_backend_tensor_set(in_lidx, lidx.data(), 0, (size_t) Kl * sizeof(int32_t));
    }

    ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (!ok) {
        if (err) {
            *err = "YuE2 prefill graph compute failed";
        }
    } else {
        cache.filled = T;
        out->T       = T;
        if (Kh > 0) {
            out->H = H;
            out->hidden.assign((size_t) (Kh * (int64_t) taps.size() * H), 0.0f);
            const int64_t       n_layers = (int64_t) taps.size();
            std::vector<float> layer_buf((size_t) (Kh * H));
            for (int64_t l = 0; l < n_layers; l++) {
                ggml_backend_tensor_get(hidden_taps[(size_t) l], layer_buf.data(), 0,
                                        (size_t) (Kh * H) * sizeof(float));
                for (int64_t k = 0; k < Kh; k++) {
                    memcpy(out->hidden.data() + (size_t) (k * n_layers * H + l * H),
                           layer_buf.data() + (size_t) (k * H), (size_t) H * sizeof(float));
                }
            }
        }
        if (Kl > 0) {
            out->V = V;
            out->logits.assign((size_t) (Kl * V), 0.0f);
            ggml_backend_tensor_get(out_logits, out->logits.data(), 0, (size_t) (Kl * V) * sizeof(float));
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    free(gbuf);
    return ok;
}

// Single-token decode step: feeds ONE token at absolute position
// `cache.filled` (the next unfilled slot), attends over cache rows
// `[0,cache.filled]` inclusive (its own row is written before the attention
// call runs, matching the reference's own scatter-before-attend convention,
// 03-reference-numerics.md §2.7), and returns the full 184704-wide logits row
// this call's own position predicts. Advances `cache.filled` by 1 on success.
static bool yue2_ar_decode_step(const Yue2Model & m, Yue2ArKvCache & cache, int32_t token_id,
                                std::vector<float> * logits_out, std::string * err) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    if (cache.filled >= cache.capacity) {
        if (err) {
            *err = "yue2_ar_decode_step: cache is full (filled == capacity)";
        }
        return false;
    }
    const int64_t         pos      = cache.filled;
    const int64_t         n_kv_pad = pos + 1;
    const Yue2LmConfig & c        = m.lm_cfg;
    const int64_t         V        = (int64_t) c.vocab_size;
    const int             L        = (int) c.block_count;

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (YUE2_LM_MAX_NODES + 256) + ggml_graph_overhead_custom(YUE2_LM_MAX_NODES, false);
    uint8_t * gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!gbuf) {
        if (err) {
            *err = "out of host memory allocating the YuE2 decode-step compute graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(gbuf);
        if (err) {
            *err = "ggml_init failed for the YuE2 decode-step compute graph context";
        }
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, YUE2_LM_MAX_NODES, false);

    ggml_tensor * in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(in_pos);
    ggml_tensor * in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_set_input(in_rows);
    // [n_kv_pad, 1] -- one query row, fully visible over [0,pos] (every key
    // position this view exposes IS <= pos by construction: n_kv_pad==pos+1
    // and the view always starts at row 0 of the cache), so an all-zero mask
    // is already correct -- no upper-triangular restriction is needed for a
    // single trailing query row.
    ggml_tensor * in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, 1);
    ggml_set_input(in_mask);

    ggml_tensor * h = ggml_get_rows(ctx, m.lm.token_embd, in_ids);  // [H,1]
    const bool use_flash = yue2_lm_use_flash(m.backend);
    for (int i = 0; i < L; i++) {
        h = yue2_ar_block(ctx, gf, c, m.lm.blk[(size_t) i], h, in_pos, in_mask, in_rows, cache.k[(size_t) i],
                          cache.v[(size_t) i], n_kv_pad, use_flash);
    }
    ggml_tensor * h_final    = yue2_lm_rms(ctx, h, m.lm.output_norm, c.rms_eps);
    ggml_tensor * out_logits = ggml_mul_mat(ctx, m.lm.output, h_final);  // [V,1]
    ggml_set_output(out_logits);
    ggml_build_forward_expand(gf, out_logits);

    BackendPair           bp    = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
    ggml_backend_sched_t sched = backend_sched_new(bp, YUE2_LM_MAX_NODES);
    bool                 ok    = sched && ggml_backend_sched_alloc_graph(sched, gf);
    if (!ok) {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        ggml_free(ctx);
        free(gbuf);
        if (err) {
            *err = "YuE2 decode-step graph allocation failed (out of VRAM?) at pos=" + std::to_string((long long) pos);
        }
        return false;
    }
    yue2_imatrix_hook(sched);

    const int32_t id32 = token_id;
    ggml_backend_tensor_set(in_ids, &id32, 0, sizeof(int32_t));
    const int32_t pos32 = (int32_t) pos;
    ggml_backend_tensor_set(in_pos, &pos32, 0, sizeof(int32_t));
    const int64_t row64 = pos;
    ggml_backend_tensor_set(in_rows, &row64, 0, sizeof(int64_t));
    std::vector<uint16_t> mask_host((size_t) n_kv_pad, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(in_mask, mask_host.data(), 0, (size_t) n_kv_pad * sizeof(uint16_t));

    ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        cache.filled = pos + 1;
        logits_out->assign((size_t) V, 0.0f);
        ggml_backend_tensor_get(out_logits, logits_out->data(), 0, (size_t) V * sizeof(float));
    } else if (err) {
        *err = "YuE2 decode-step graph compute failed";
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    free(gbuf);
    return ok;
}
