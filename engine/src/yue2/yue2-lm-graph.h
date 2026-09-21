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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 28 layers x ~20 nodes (no adapter/CFG-batch/fold-rows branching to inflate
// it) + up to 30 hidden-tap gathers + embed/head-gather/logits plumbing.
// Loose on purpose, same posture as MM3_LM_MAX_NODES.
#define YUE2_LM_MAX_NODES 4096

struct Yue2ArStepProfile {
    int64_t calls = 0;
    double graph_ms = 0.0;
    double alloc_ms = 0.0;
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
    double cleanup_ms = 0.0;
};

static thread_local Yue2ArStepProfile g_yue2_ar_step_profile;

static bool yue2_ar_step_profile_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("YUE2_AR_PROFILE");
        return e && e[0] && e[0] != '0';
    }();
    return enabled;
}

static void yue2_ar_step_profile_reset() {
    if (yue2_ar_step_profile_enabled()) g_yue2_ar_step_profile = {};
}

static void yue2_ar_step_profile_log(const char * stage) {
    if (!yue2_ar_step_profile_enabled()) return;
    const auto & p = g_yue2_ar_step_profile;
    fprintf(stderr, "[YuE2-AR-Profile] %s calls=%lld graph=%.1f alloc=%.1f upload=%.1f compute=%.1f readback=%.1f cleanup=%.1f ms\n",
            stage, (long long) p.calls, p.graph_ms, p.alloc_ms, p.upload_ms, p.compute_ms,
            p.readback_ms, p.cleanup_ms);
}

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

static ggml_tensor * yue2_lm_match_type(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    return x->type == w->type ? w : ggml_cast(ctx, w, x->type);
}

static ggml_tensor * yue2_lm_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    ggml_tensor * normalized = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normalized, yue2_lm_match_type(ctx, normalized, w));
}

static ggml_tensor * yue2_convrot_linear(ggml_context * ctx, const Yue2AitkConvRotLinear & w,
                                          ggml_tensor * x) {
    if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx, x, GGML_TYPE_F32);
    return ggml_convrot8(ctx, w.weight_i8, x, w.scales_f32, nullptr, w.rotation, true);
}

static ggml_tensor * yue2_convrot_rows(ggml_context * ctx, ggml_tensor * fused,
                                        int64_t first, int64_t count) {
    GGML_ASSERT(fused->type == GGML_TYPE_F32 && first >= 0 && first + count <= fused->ne[0]);
    return ggml_cont(ctx, ggml_view_2d(ctx, fused, count, fused->ne[1], fused->nb[1],
                                        (size_t) first * sizeof(float)));
}

static ggml_tensor * yue2_convrot_adapt(ggml_context * ctx, const Yue2Model * model,
                                        int layer, bool nar, Yue2ConvRotSite site,
                                        ggml_tensor * x, ggml_tensor * base) {
    if (!model || model->convrot_adapters.empty()) return base;
    ggml_tensor * result = base;
    ggml_tensor * input = x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    input = ggml_bf16_round(ctx, input);
    for (const auto & adapter : model->convrot_adapters) {
        if (adapter->nar() != nar) continue;
        const Yue2ConvRotLora & lora = adapter->site(layer, site);
        if (lora.scale == 0.0f) continue;
        ggml_tensor * ax = ggml_mul_mat(ctx, lora.a, input);
        ggml_tensor * delta = ggml_mul_mat(ctx, lora.b, ax);
        delta = ggml_bf16_round(ctx, ggml_scale(ctx, delta, lora.scale));
        result = ggml_bf16_round(ctx, ggml_add(ctx, result, delta));
    }
    return result;
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
// twin). Copy of mm3_lm_block's math (mm3-lm-graph.h), stripped of runtime
// adapters and the mul-fold-rows decode optimization.
//
// Two layouts share this function (doc 30 §core):
//   batched == false  prefill/one-shot: h [H,T] is T consecutive tokens of ONE
//                     cache set (`set`); K/V land in rows `rows[T]` of that
//                     set's view and the block attends over rows [0,n_kv) of it.
//   batched == true   lockstep decode: h [H,S] is ONE token per set for every
//                     set of the cache; positions/rows/mask are per set, K/V
//                     land at rows[1,1,S] of the full [D,cap,Nkv,S] cache and
//                     each set attends over its own rows [0,n_kv) under its
//                     own mask column ([n_kv,1,1,S]). At S=1 this is the exact
//                     tensor shape the single-row decode always used.
static ggml_tensor * yue2_ar_block(ggml_context * ctx, ggml_cgraph * gf, const Yue2LmConfig & c,
                                   const Yue2LmLayer & w, ggml_tensor * h, ggml_tensor * positions, ggml_tensor * mask,
                                   ggml_tensor * rows, ggml_tensor * kcache, ggml_tensor * vcache, int64_t n_kv,
                                   bool use_flash, const Yue2Model * cm = nullptr, int layer = -1,
                                   int64_t set = 0, bool batched = false) {
    const Yue2AitkLayerWeights * cw = cm && cm->convrot && layer >= 0 ?
        &cm->convrot->ar().layers[(size_t) layer] : nullptr;
    const int64_t H   = (int64_t) c.embedding_length;
    const int64_t D   = (int64_t) c.key_length;
    const int64_t Nh  = (int64_t) c.head_count;
    const int64_t Nkv = (int64_t) c.head_count_kv;
    const int64_t T   = h->ne[1];  // tokens (prefill) or sets (batched decode)
    const int64_t S   = batched ? T : 1;

    ggml_tensor * n = yue2_lm_rms(ctx, h, w.attn_norm, c.rms_eps);

    ggml_tensor * qkv = cw ? yue2_convrot_linear(ctx, cw->qkv, n) : nullptr;
    ggml_tensor * q0 = cw ? yue2_convrot_rows(ctx, qkv, 0, H) : ggml_mul_mat(ctx, w.attn_q, n);
    ggml_tensor * k0 = cw ? yue2_convrot_rows(ctx, qkv, H, D*Nkv) : ggml_mul_mat(ctx, w.attn_k, n);
    ggml_tensor * v0 = cw ? yue2_convrot_rows(ctx, qkv, H+D*Nkv, D*Nkv) : ggml_mul_mat(ctx, w.attn_v, n);
    if (cw) {
        q0 = yue2_convrot_adapt(ctx, cm, layer, false, YUE2_CR_Q, n, q0);
        k0 = yue2_convrot_adapt(ctx, cm, layer, false, YUE2_CR_K, n, k0);
        v0 = yue2_convrot_adapt(ctx, cm, layer, false, YUE2_CR_V, n, v0);
    }
    ggml_tensor * q = ggml_reshape_4d(ctx, q0, D, Nh, T, 1);
    ggml_tensor * k = ggml_reshape_4d(ctx, k0, D, Nkv, T, 1);
    ggml_tensor * v = ggml_reshape_4d(ctx, v0, D, Nkv, T, 1);

    // Per-head QK RMSNorm (dim=head_dim=128, eps 1e-6), strictly BEFORE RoPE.
    // No norm on V (03-reference-numerics.md §2.2, Attention.project_qkv).
    q = yue2_lm_rms(ctx, q, w.attn_q_norm, c.rms_eps);
    k = yue2_lm_rms(ctx, k, w.attn_k_norm, c.rms_eps);

    // NeoX half-split rotation (x1=first half, x2=second half — the
    // reference's own convention, §2.2's _apply_rotary), theta = rope_freq_base
    // (1e6). `positions` has one entry per token (prefill) or per set
    // (batched decode) — which is what lets cond and uncond sets sit at
    // different absolute positions in one graph.
    q = ggml_rope_ext(ctx, q, positions, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                       0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                       0.0f, 0.0f);

    ggml_tensor * k_win;
    ggml_tensor * v_win;
    ggml_tensor * q4;
    if (!batched) {
        // [D,T,Nkv,1] into rows `rows` of set `set`.
        ggml_tensor * k_w = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * v_w = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_tensor * k_set = ggml_view_4d(ctx, kcache, D, kcache->ne[1], Nkv, 1, kcache->nb[1], kcache->nb[2],
                                           kcache->nb[3], (size_t) set * kcache->nb[3]);
        ggml_tensor * v_set = ggml_view_4d(ctx, vcache, D, vcache->ne[1], Nkv, 1, vcache->nb[1], vcache->nb[2],
                                           vcache->nb[3], (size_t) set * vcache->nb[3]);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_set, k_w, rows));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_set, v_w, rows));
        k_win = ggml_view_4d(ctx, kcache, D, n_kv, Nkv, 1, kcache->nb[1], kcache->nb[2], kcache->nb[3],
                             (size_t) set * kcache->nb[3]);
        v_win = ggml_view_4d(ctx, vcache, D, n_kv, Nkv, 1, vcache->nb[1], vcache->nb[2], vcache->nb[3],
                             (size_t) set * vcache->nb[3]);
        q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D,T,Nh,1]
    } else {
        // [D,Nkv,S,1] -> [D,1,Nkv,S]: one row per set, set on the batch axis.
        ggml_tensor * k_w = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 3, 1));
        ggml_tensor * v_w = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 3, 1));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, kcache, k_w, rows));  // rows [1,1,S]
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, vcache, v_w, rows));
        k_win = ggml_view_4d(ctx, kcache, D, n_kv, Nkv, S, kcache->nb[1], kcache->nb[2], kcache->nb[3], 0);
        v_win = ggml_view_4d(ctx, vcache, D, n_kv, Nkv, S, vcache->nb[1], vcache->nb[2], vcache->nb[3], 0);
        q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 3, 1));  // [D,1,Nh,S]
    }

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
    attn = ggml_reshape_2d(ctx, attn, H, T);  // [D,Nh,T,1] or [D,Nh,1,S] -> [H,T]

    ggml_tensor * projected = cw ? yue2_convrot_linear(ctx, cw->output, attn) :
                                   ggml_mul_mat(ctx, w.attn_output, attn);
    if (cw) projected = yue2_convrot_adapt(ctx, cm, layer, false, YUE2_CR_O, attn, projected);
    h = ggml_add(ctx, h, projected);

    // SwiGLU: down(silu(gate) * up). No bias anywhere (03-reference-numerics.md §2.2).
    ggml_tensor * n2   = yue2_lm_rms(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gate_up = cw ? yue2_convrot_linear(ctx, cw->gate_up, n2) : nullptr;
    ggml_tensor * gate0 = cw ? yue2_convrot_rows(ctx, gate_up, 0, c.feed_forward_length) :
                              ggml_mul_mat(ctx, w.ffn_gate, n2);
    ggml_tensor * up = cw ? yue2_convrot_rows(ctx, gate_up, c.feed_forward_length, c.feed_forward_length) :
                            ggml_mul_mat(ctx, w.ffn_up, n2);
    if (cw) {
        gate0 = yue2_convrot_adapt(ctx, cm, layer, false, YUE2_CR_GATE, n2, gate0);
        up = yue2_convrot_adapt(ctx, cm, layer, false, YUE2_CR_UP, n2, up);
    }
    ggml_tensor * gate = ggml_silu(ctx, gate0);
    ggml_tensor * activated = ggml_mul(ctx, gate, up);
    ggml_tensor * down = cw ? yue2_convrot_linear(ctx, cw->down, activated) :
                              ggml_mul_mat(ctx, w.ffn_down, activated);
    if (cw) down = yue2_convrot_adapt(ctx, cm, layer, false, YUE2_CR_DOWN, activated, down);
    return ggml_add(ctx, h, down);
}

// The lm_head, optionally restricted to vocab rows [head_lo, head_lo+head_n)
// (doc 30 #1 / upstream 4a1de08f): a stage only ever samples from its own
// content range plus its end token, which sit next to each other in the
// vocabulary, so the rest of the 184704-row matmul and its readback are
// wasted. A whole-row range of a contiguous [H,V] weight is itself contiguous,
// so mul_mat takes the view as-is. head_n == 0 means the full vocabulary.
// ConvRot heads are never sliced (int8 + rotation layout), and neither is a
// head under imatrix collection (the hook matches weights by name).
static ggml_tensor * yue2_lm_head(ggml_context * ctx, const Yue2Model & m, ggml_tensor * x, int64_t head_lo,
                                  int64_t head_n) {
    if (m.convrot) return yue2_convrot_linear(ctx, m.convrot->lm_head(), x);
    ggml_tensor * w = m.lm.output;
    if (head_n > 0 && !g_yue2_imatrix.armed) {
        GGML_ASSERT(head_lo >= 0 && head_lo + head_n <= w->ne[1]);
        w = ggml_view_2d(ctx, w, w->ne[0], head_n, w->nb[1], (size_t) head_lo * w->nb[1]);
    }
    return ggml_mul_mat(ctx, w, x);
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
                                 kv_v[(size_t) i], T, use_flash,
                                 &m, i);
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
        out_logits             = m.convrot ? yue2_convrot_linear(ctx, m.convrot->lm_head(), gathered) :
                                            ggml_mul_mat(ctx, m.lm.output, gathered);  // [V,Kl]
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

// ── Persistent KV-cache decode (M4, batched since doc 30) ───────────────────
//
// yue2_ar_forward() above is a ONE-SHOT full-sequence forward: it allocates a
// KV cache sized exactly to that one call's T and throws it away at the end
// — correct for teacher-forced parity checking (M2/M3), but not the shape a
// real incremental generation loop uses. This section adds the other half:
// a KV cache that SURVIVES across calls (`Yue2ArKvCache`), a multi-token
// `yue2_ar_prefill()` that appends N rows to one set of it, and a lockstep
// `yue2_ar_decode_batch()` that appends one row to EVERY set in one graph.
//
// Layout: `[D, capacity, Nkv, S]` per layer, F16. S "sets" are S independent
// sequences (docs/plans/yue2/30-upstream-backports.md): the cond/uncond
// pair of a guided decode, or the B songs of a batch, or both. Each set has
// its own `filled[s]` row count and its own absolute positions, so sets of
// different prompt lengths decode in the same graph — the constraint that
// made this file's header reject a batch axis for CFG (yue2_negative_prefix
// drops the whole lyric block, so the two branches never sit at the same
// position) is gone: positions are per set, not per graph.
//
// Within one set, a head's first `n` rows are one contiguous span at byte
// offset `s*nb[3] + h*nb[2]`, which is what `yue2_ar_kv_cache_dump_layer`
// relies on to reproduce the fixture's `[num_kv_heads, seq, head_dim]`
// (head-major) flatten order with a single `ggml_backend_tensor_get` per
// head.
//
// `filled[s]` may be lowered (`yue2_ar_kv_cache_trim`): rows past it stay
// allocated and masked out, and the next prefill on that set overwrites
// them. That is how an acoustic chunk keeps the prompt rows it shares with
// the semantic stage and forwards only the tail it is missing (doc 30 #2).

struct Yue2ArDecodeGraph {
    int64_t bucket  = 0;
    int64_t n_sets  = 0;
    int64_t head_lo = -1;
    int64_t head_n  = -1;
    uint8_t * gbuf = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_tensor * in_ids = nullptr;
    ggml_tensor * in_pos = nullptr;
    ggml_tensor * in_rows = nullptr;
    ggml_tensor * in_mask = nullptr;
    ggml_tensor * out_logits = nullptr;
    std::vector<uint16_t> mask_host;
};

static void yue2_ar_decode_graph_free(Yue2ArDecodeGraph * d) {
    if (d->sched) ggml_backend_sched_free(d->sched);
    if (d->ctx) ggml_free(d->ctx);
    free(d->gbuf);
    *d = {};
}

struct Yue2ArKvCache {
    int64_t                    capacity = 0;  // rows per set
    int64_t                    n_sets   = 0;
    std::vector<int64_t>       filled;        // per set: rows [0,filled[s]) are valid
    // lm_head window applied by prefill/decode on this cache (doc 30 #1).
    // 0 = full vocabulary. Logits come back as [head_n] per set, indexed from
    // head_lo; the sampler is told the same base.
    int64_t                    head_lo  = 0;
    int64_t                    head_n   = 0;
    ggml_context *              ctx      = nullptr;
    ggml_backend_buffer_t       buf      = nullptr;
    std::vector<ggml_tensor *> k, v;          // one F16 [D,capacity,Nkv,S] tensor per layer
    Yue2ArDecodeGraph dec;                   // stable within each (bucket, S, head window)
};

static bool yue2_ar_kv_cache_alloc(const Yue2Model & m, int64_t capacity, Yue2ArKvCache * out, std::string * err,
                                   int64_t n_sets = 1) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    if (capacity <= 0 || n_sets <= 0) {
        if (err) {
            *err = "yue2_ar_kv_cache_alloc: capacity and n_sets must be > 0";
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
        out->k[(size_t) i] = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F16, D, capacity, Nkv, n_sets);
        out->v[(size_t) i] = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F16, D, capacity, Nkv, n_sets);
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
    out->n_sets   = n_sets;
    out->filled.assign((size_t) n_sets, 0);
    out->head_lo  = 0;
    out->head_n   = 0;
    return true;
}

static void yue2_ar_kv_cache_free(Yue2ArKvCache * c) {
    yue2_ar_decode_graph_free(&c->dec);
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
    c->n_sets   = 0;
    c->filled.clear();
}

// Forget rows [n, filled[set]) of one set. Nothing is freed or zeroed: the
// rows stay masked until the next prefill on this set overwrites them.
static inline void yue2_ar_kv_cache_trim(Yue2ArKvCache & c, int64_t set, int64_t n) {
    GGML_ASSERT(set >= 0 && set < c.n_sets && n >= 0 && n <= c.filled[(size_t) set]);
    c.filled[(size_t) set] = n;
}

// Replicate rows [0,n) of set `src_set` of `src` into set `dst_set` of `dst`
// on the device (one ggml_cpy per layer tensor, no host round trip) and mark
// the destination set filled to n. Both caches may be the same object. A
// prefix shared by two sets — the prompt of B songs, the cond and uncond
// halves of a guided plan, a song's prompt and each of its acoustic chunks —
// costs one prefill this way instead of one per set.
static bool yue2_ar_kv_cache_copy_rows(const Yue2Model & m, const Yue2ArKvCache & src, int64_t src_set,
                                       Yue2ArKvCache & dst, int64_t dst_set, int64_t n, std::string * err) {
    if (src_set < 0 || dst_set < 0 || src_set >= src.n_sets || dst_set >= dst.n_sets ||
        (&src == &dst && src_set == dst_set) || n <= 0 || n > src.filled[(size_t) src_set] || n > dst.capacity ||
        src.k.size() != dst.k.size()) {
        if (err) *err = "yue2_ar_kv_cache_copy_rows: bad set index or row count";
        return false;
    }
    const int L = (int) src.k.size();
    // Per layer: two source views, two destination views, two cpy nodes — six
    // graph nodes, not two (a view is a node once the graph expands it) — and
    // the scheduler's hash set must also hold the leaves: both caches' K and V
    // tensors when src and dst differ (4 per layer).
    const size_t n_nodes   = (size_t) L * 16 + 64;
    const size_t ctx_bytes = ggml_tensor_overhead() * (n_nodes + 16) + ggml_graph_overhead_custom(n_nodes, false);
    std::vector<uint8_t> gbuf(ctx_bytes);
    ggml_init_params ip = { ctx_bytes, gbuf.data(), /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "yue2_ar_kv_cache_copy_rows: ggml_init failed";
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false);
    for (int i = 0; i < L; i++) {
        ggml_tensor * st[2] = { src.k[(size_t) i], src.v[(size_t) i] };
        ggml_tensor * dt[2] = { dst.k[(size_t) i], dst.v[(size_t) i] };
        for (int j = 0; j < 2; j++) {
            ggml_tensor * x = st[j];
            ggml_tensor * y = dt[j];
            ggml_tensor * a = ggml_view_4d(ctx, x, x->ne[0], n, x->ne[2], 1, x->nb[1], x->nb[2], x->nb[3],
                                           (size_t) src_set * x->nb[3]);
            ggml_tensor * b = ggml_view_4d(ctx, y, y->ne[0], n, y->ne[2], 1, y->nb[1], y->nb[2], y->nb[3],
                                           (size_t) dst_set * y->nb[3]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, a, b));
        }
    }
    // Through the scheduler, not ggml_backend_graph_compute: gallocr is what
    // gives the cache views a buffer (ggml_backend_view_init), and CUDA's
    // graph runner dereferences src->buffer.
    BackendPair bp = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
    ggml_backend_sched_t sched = backend_sched_new(bp, (int) n_nodes);
    bool ok = sched && ggml_backend_sched_alloc_graph(sched, gf) &&
              ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
    if (!ok) {
        if (err) *err = "yue2_ar_kv_cache_copy_rows: graph compute failed";
        return false;
    }
    dst.filled[(size_t) dst_set] = n;
    return true;
}

static inline bool yue2_ar_kv_cache_copy_set(const Yue2Model & m, Yue2ArKvCache & c, int64_t src, int64_t dst,
                                             int64_t n, std::string * err) {
    return yue2_ar_kv_cache_copy_rows(m, c, src, c, dst, n, err);
}

// Reads back rows [0,n) of layer `layer`'s cached K/V for set `set`, widened
// to f32, in StaticKVCache's own head-major flatten order ([num_kv_heads, n,
// head_dim], h slowest / t / d fastest) — directly comparable to the
// fixture's kv_layer{0,27}_{k,v}.bin after that file's own bf16-widen (both
// are the same flatten order; see the section note on why no transpose is
// needed). `n` must be <= cache.filled[set].
static bool yue2_ar_kv_cache_dump_layer(const Yue2ArKvCache & cache, int layer, int64_t n, std::vector<float> * k_out,
                                        std::vector<float> * v_out, int64_t set = 0) {
    if (layer < 0 || (size_t) layer >= cache.k.size() || set < 0 || set >= cache.n_sets || n <= 0 ||
        n > cache.filled[(size_t) set]) {
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
        ggml_backend_tensor_get(kt, tmp.data(), (size_t) set * kt->nb[3] + (size_t) h * kt->nb[2],
                                (size_t) (n * D) * sizeof(uint16_t));
        for (size_t i = 0; i < tmp.size(); i++) {
            (*k_out)[(size_t) (h * n * D) + i] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &tmp[i]);
        }
        ggml_backend_tensor_get(vt, tmp.data(), (size_t) set * vt->nb[3] + (size_t) h * vt->nb[2],
                                (size_t) (n * D) * sizeof(uint16_t));
        for (size_t i = 0; i < tmp.size(); i++) {
            (*v_out)[(size_t) (h * n * D) + i] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &tmp[i]);
        }
    }
    return true;
}

// Multi-token prefill: runs `ids.size()` tokens through the full causal AR
// stack exactly like yue2_ar_forward()'s body, EXCEPT the K/V it writes land
// in set `set` of `cache` at rows [filled[set], filled[set]+T) — absolute
// positions continue from filled[set] — and `filled[set]` advances on
// success so a later decode step knows where to continue. On an empty set
// this is the classic whole-prompt prefill; on a non-empty one it appends
// (doc 30 #2). Requested logit/hidden positions are row indices into `ids`
// (same convention as Yue2ArForwardRequest); logits are [Kl, head_n or V]
// under the cache's head window.
static bool yue2_ar_prefill(const Yue2Model & m, Yue2ArKvCache & cache, const std::vector<int32_t> & ids,
                            const std::vector<int64_t> & logit_positions,
                            const std::vector<int64_t> & hidden_positions, Yue2ArForwardResult * out,
                            std::string * err, int64_t set = 0) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    if (set < 0 || set >= cache.n_sets) {
        if (err) {
            *err = "yue2_ar_prefill: set index out of range";
        }
        return false;
    }
    const int64_t base = cache.filled[(size_t) set];
    const int64_t T    = (int64_t) ids.size();
    if (T <= 0 || base + T > cache.capacity) {
        if (err) {
            *err = "yue2_ar_prefill: ids.size() must be in (0, cache.capacity - filled]";
        }
        return false;
    }
    const int64_t n_kv = base + T;
    const Yue2LmConfig & c   = m.lm_cfg;
    const int64_t         H   = (int64_t) c.embedding_length;
    const int64_t         V   = (int64_t) c.vocab_size;
    const int             L   = (int) c.block_count;
    const int64_t         Kh  = (int64_t) hidden_positions.size();
    const int64_t         Kl  = (int64_t) logit_positions.size();
    const int64_t         head_n = (cache.head_n > 0 && !m.convrot && !g_yue2_imatrix.armed) ? cache.head_n : V;
    const int64_t         head_lo = head_n == V ? 0 : cache.head_lo;

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
    ggml_tensor * in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, T);  // [n_kv, T], causal over the whole set
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
                          cache.v[(size_t) i], n_kv, use_flash, &m, i, set, /*batched=*/false);
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
        out_logits             = yue2_lm_head(ctx, m, gathered, head_lo, head_n);
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
        pos_host[(size_t) i]  = (int32_t) (base + i);
        rows_host[(size_t) i] = base + i;
    }
    ggml_backend_tensor_set(in_pos, pos_host.data(), 0, (size_t) T * sizeof(int32_t));
    ggml_backend_tensor_set(in_rows, rows_host.data(), 0, (size_t) T * sizeof(int64_t));

    std::vector<uint16_t> mask_host((size_t) (n_kv * T));
    const uint16_t zero = ggml_fp32_to_fp16(0.0f);
    const uint16_t hidden = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t i = 0; i < T; i++) {          // i = query row (absolute position base+i)
        for (int64_t j = 0; j < n_kv; j++) {   // j = key position
            mask_host[(size_t) (i * n_kv + j)] = j <= base + i ? zero : hidden;
        }
    }
    ggml_backend_tensor_set(in_mask, mask_host.data(), 0, (size_t) (n_kv * T) * sizeof(uint16_t));

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
        cache.filled[(size_t) set] = n_kv;
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
            out->V = head_n;
            out->logits.assign((size_t) (Kl * head_n), 0.0f);
            ggml_backend_tensor_get(out_logits, out->logits.data(), 0, (size_t) (Kl * head_n) * sizeof(float));
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    free(gbuf);
    return ok;
}

// Lockstep decode: feeds ONE token per set (`ids[s]` at absolute position
// `filled[s]`, the set's next unfilled slot), each set attending over its own
// rows `[0,filled[s]]` inclusive (its own row is written before the attention
// runs, matching the reference's scatter-before-attend convention,
// 03-reference-numerics.md §2.7), and returns the logits row every set's own
// position predicts: `logits_out` is [S, head_n] (head_n == V without a head
// window). Advances every `filled[s]` by 1 on success. The graph is built
// once per (KV bucket, S, head window) and replayed with new inputs.
static bool yue2_ar_decode_batch(const Yue2Model & m, Yue2ArKvCache & cache, const int32_t * ids,
                                 std::vector<float> * logits_out, std::string * err) {
    const bool profile = yue2_ar_step_profile_enabled();
    auto tick_start = std::chrono::steady_clock::now();
    auto tick = [&](double & ms) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        ms += std::chrono::duration<double, std::milli>(now - tick_start).count();
        tick_start = now;
    };
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    const int64_t S = cache.n_sets;
    int64_t pos_max = 0;
    for (int64_t s = 0; s < S; s++) {
        if (cache.filled[(size_t) s] >= cache.capacity) {
            if (err) {
                *err = "yue2_ar_decode_batch: cache set is full (filled == capacity)";
            }
            return false;
        }
        pos_max = std::max(pos_max, cache.filled[(size_t) s]);
    }
    int64_t n_kv_pad = 1;
    // Padded buckets may alter sampling through small attention differences.
    // Exact span remains available for numerical comparisons.
    static const bool exact_span = std::getenv("YUE2_AR_EXACT_SPAN") != nullptr;
    if (exact_span) {
        n_kv_pad = pos_max + 1;
    } else {
        static const bool power_of_two_bucket = std::getenv("YUE2_AR_BUCKET_POW2") != nullptr;
        if (!power_of_two_bucket) {
            // GGML flash attention's KV stride is 256. Fixed-width buckets keep
            // CUDA graph replay while reducing masked work at long positions.
            n_kv_pad = ((pos_max + 256) / 256) * 256;
        } else {
            while (n_kv_pad < pos_max + 1 && n_kv_pad < cache.capacity) n_kv_pad *= 2;
        }
        static const bool odd_bucket = std::getenv("YUE2_AR_BUCKET_ODD") != nullptr;
        if (odd_bucket && n_kv_pad < cache.capacity) ++n_kv_pad;
        n_kv_pad = std::min(n_kv_pad, cache.capacity);
    }
    const Yue2LmConfig & c        = m.lm_cfg;
    const int64_t         V        = (int64_t) c.vocab_size;
    const int             L        = (int) c.block_count;
    const int64_t         head_n   = (cache.head_n > 0 && !m.convrot && !g_yue2_imatrix.armed) ? cache.head_n : V;
    const int64_t         head_lo  = head_n == V ? 0 : cache.head_lo;
    Yue2ArDecodeGraph & d = cache.dec;
    static const bool no_reuse = std::getenv("YUE2_AR_NO_REUSE") != nullptr;
    if (no_reuse) yue2_ar_decode_graph_free(&d);
    if (d.bucket != n_kv_pad || d.n_sets != S || d.head_lo != head_lo || d.head_n != head_n) {
        yue2_ar_decode_graph_free(&d);
        const size_t ctx_bytes =
            ggml_tensor_overhead() * (YUE2_LM_MAX_NODES + 256) + ggml_graph_overhead_custom(YUE2_LM_MAX_NODES, false);
        d.gbuf = (uint8_t *) malloc(ctx_bytes);
        if (!d.gbuf) {
            if (err) *err = "out of host memory allocating the YuE2 decode-step compute graph context";
            return false;
        }
        ggml_init_params ip = { ctx_bytes, d.gbuf, /*no_alloc*/ true };
        d.ctx = ggml_init(ip);
        if (!d.ctx) {
            yue2_ar_decode_graph_free(&d);
            if (err) *err = "ggml_init failed for the YuE2 decode-step compute graph context";
            return false;
        }
        d.gf = ggml_new_graph_custom(d.ctx, YUE2_LM_MAX_NODES, false);
        d.in_ids = ggml_new_tensor_1d(d.ctx, GGML_TYPE_I32, S);
        ggml_set_input(d.in_ids);
        d.in_pos = ggml_new_tensor_1d(d.ctx, GGML_TYPE_I32, S);
        ggml_set_input(d.in_pos);
        d.in_rows = ggml_new_tensor_3d(d.ctx, GGML_TYPE_I64, 1, 1, S);  // one destination row per set
        ggml_set_input(d.in_rows);
        // Padding remains invisible until its row is written by a later step.
        // One mask column per set: sets sit at different positions.
        d.in_mask = ggml_new_tensor_4d(d.ctx, GGML_TYPE_F16, n_kv_pad, 1, 1, S);
        ggml_set_input(d.in_mask);

        ggml_tensor * h = ggml_get_rows(d.ctx, m.lm.token_embd, d.in_ids);  // [H,S]
        const bool use_flash = yue2_lm_use_flash(m.backend);
        for (int i = 0; i < L; i++) {
            h = yue2_ar_block(d.ctx, d.gf, c, m.lm.blk[(size_t) i], h, d.in_pos, d.in_mask, d.in_rows,
                              cache.k[(size_t) i], cache.v[(size_t) i], n_kv_pad, use_flash, &m, i, 0,
                              /*batched=*/true);
        }
        ggml_tensor * h_final = yue2_lm_rms(d.ctx, h, m.lm.output_norm, c.rms_eps);
        d.out_logits = yue2_lm_head(d.ctx, m, h_final, head_lo, head_n);  // [head_n, S]
        ggml_set_output(d.out_logits);
        ggml_build_forward_expand(d.gf, d.out_logits);
        d.mask_host.resize((size_t) (n_kv_pad * S));
        d.bucket  = n_kv_pad;
        d.n_sets  = S;
        d.head_lo = head_lo;
        d.head_n  = head_n;
        tick(g_yue2_ar_step_profile.graph_ms);

        BackendPair bp = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
        d.sched = backend_sched_new(bp, YUE2_LM_MAX_NODES);
        if (!d.sched || !ggml_backend_sched_alloc_graph(d.sched, d.gf)) {
            yue2_ar_decode_graph_free(&d);
            if (err) *err = "YuE2 decode-step graph allocation failed (out of VRAM?) at pos=" +
                             std::to_string((long long) pos_max);
            return false;
        }
        tick(g_yue2_ar_step_profile.alloc_ms);
    }

    std::vector<int32_t> pos_host((size_t) S);
    std::vector<int64_t> rows_host((size_t) S);
    const uint16_t zero = ggml_fp32_to_fp16(0.0f);
    const uint16_t hidden = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t s = 0; s < S; s++) {
        const int64_t pos = cache.filled[(size_t) s];
        pos_host[(size_t) s]  = (int32_t) pos;
        rows_host[(size_t) s] = pos;
        auto col = d.mask_host.begin() + (size_t) (s * n_kv_pad);
        std::fill(col, col + (size_t) (pos + 1), zero);
        std::fill(col + (size_t) (pos + 1), col + (size_t) n_kv_pad, hidden);
    }
    ggml_backend_tensor_set(d.in_ids, ids, 0, (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(d.in_pos, pos_host.data(), 0, (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(d.in_rows, rows_host.data(), 0, (size_t) S * sizeof(int64_t));
    ggml_backend_tensor_set(d.in_mask, d.mask_host.data(), 0, d.mask_host.size() * sizeof(uint16_t));
    tick(g_yue2_ar_step_profile.upload_ms);

    yue2_imatrix_hook(d.sched);
    bool ok = ggml_backend_sched_graph_compute(d.sched, d.gf) == GGML_STATUS_SUCCESS;
    tick(g_yue2_ar_step_profile.compute_ms);
    if (ok) {
        for (int64_t s = 0; s < S; s++) cache.filled[(size_t) s] += 1;
        logits_out->assign((size_t) (S * head_n), 0.0f);
        ggml_backend_tensor_get(d.out_logits, logits_out->data(), 0, (size_t) (S * head_n) * sizeof(float));
        tick(g_yue2_ar_step_profile.readback_ms);
    } else if (err) {
        *err = "YuE2 decode-step graph compute failed";
    }

    tick(g_yue2_ar_step_profile.cleanup_ms);
    if (profile && ok) g_yue2_ar_step_profile.calls++;
    return ok;
}

// Single-set convenience (the probe, the trainers, every S=1 caller): one
// token into set 0, one logits row back.
static bool yue2_ar_decode_step(const Yue2Model & m, Yue2ArKvCache & cache, int32_t token_id,
                                std::vector<float> * logits_out, std::string * err) {
    if (cache.n_sets != 1) {
        if (err) *err = "yue2_ar_decode_step: cache has more than one set — use yue2_ar_decode_batch";
        return false;
    }
    return yue2_ar_decode_batch(m, cache, &token_id, logits_out, err);
}
