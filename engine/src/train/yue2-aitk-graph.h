#pragma once

// New recipe graph; no Legacy graph helpers or Legacy adapter parameterization.
// Activations are F32 storage with explicit BF16 value/gradient boundaries.
// These builders still require numerical qualification against the reference.
#include "yue2-aitk-model.h"
#include "yue2-aitk-lora.h"
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>

struct Yue2AitkGraphConfig {
    int64_t hidden = 2048, intermediate = 6144;
    int64_t heads = 16, kv_heads = 8, head_dim = 128;
    float rms_eps = 1e-6f;
    // Use the existing tensor-core training kernels. Strict scalar attention
    // remains available for arithmetic diagnostics, not the normal trainer.
    ggml_prec attention_precision = (std::getenv("YUE2_AITK_STRICT_F32") ||
        std::getenv("YUE2_AITK_BASELINE")) ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
};

struct Yue2AitkBlockNorms {
    ggml_tensor * input = nullptr;
    ggml_tensor * post_attention = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
};

struct Yue2AitkBlockResult {
    ggml_tensor * hidden = nullptr;
    // Post-RoPE K and unrotated V, [D,S,Nkv,1]. A runner copies these into
    // independent non-param buffers before NAR; never connect AR to NAR backward.
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
    // Packed O+LSE node of the fused attention (nullptr on the exact-math path).
    // The forward can copy it into the tape; the backward recompute reuses it.
    ggml_tensor * attention_packed = nullptr;
};

namespace yue2_aitk_graph {
// Set once by the runner: true when the backend has no FLASH_ATTN_TRAIN
// (Vulkan, Metal). The exact mul_mat -> soft_max -> mul_mat attention takes
// its place; same maths, but it keeps the [keys, S, heads] probabilities.
inline bool & math_attention() { static bool on = false; return on; }
// Set once by the runner: true only when the backend accepts BF16 Q/K/V
// flash attention in both directions (S2c; CUDA sm_80+). Elsewhere the BF16
// cast below never happens and the dispatch is the F32/TF32 one.
inline bool & bf16_attention_supported() { static bool on = false; return on; }
// Training precision (S2a, 2026-10-13). Default: every training GEMM is a
// BF16 x BF16 tensor-core product with FP32 accumulation and F32 output. The
// inputs are already BF16-rounded, so each product is exact and only the
// summation order differs from the F32 path; trainable F32 factors are
// cast to BF16 first, as the PyTorch reference does. Quantized GGUF weights
// and the frozen companion decoder (mm_f32) keep the F32 path.
// YUE2_AITK_PRECISION=strict-f32 restores the previous dispatch
// (GGML_PREC_F32 GEMMs, today's attention) for every recipe; the runner
// validates the value and logs which one is in force.
inline const char * precision_name() {
    const char * v = std::getenv("YUE2_AITK_PRECISION");
    return v && *v ? v : "bf16";
}
inline bool precision_valid() {
    const std::string v = precision_name();
    return v == "bf16" || v == "strict-f32";
}
inline bool & strict_f32() { static bool on = std::string(precision_name()) == "strict-f32"; return on; }
// GEMMs built so far by path; the runner logs each step's share in train.jsonl.
struct GemmCounts { int64_t bf16 = 0, f32 = 0; };
inline GemmCounts & gemm_counts() { static GemmCounts c; return c; }
// The one training GEMM: w [K, N] (weight or factor), x [K, S] F32.
inline ggml_tensor * mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    const bool tensor_core = !strict_f32() && (w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_F32);
    if (!tensor_core) {
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        ++gemm_counts().f32;
        return y;
    }
    ++gemm_counts().bf16;
    return ggml_mul_mat(ctx, w->type == GGML_TYPE_BF16 ? w : ggml_cast(ctx, w, GGML_TYPE_BF16), x);
}
// The frozen companion decoder stays on the F32 path on both settings: its
// stored F32 factors are not BF16-representable, so a BF16 cast would change
// the decoder the adapter trains against (NAR gradients moved ~4%; S2a,
// RESEARCH/YUE2_S2A_TOLERANCES.md).
inline ggml_tensor * mm_f32(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    ++gemm_counts().f32;
    return y;
}
inline ggml_tensor * f32(ggml_context * ctx, ggml_tensor * x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
}
inline ggml_tensor * round(ggml_context * ctx, ggml_tensor * x) {
    return ggml_bf16_round(ctx, x);
}
inline ggml_tensor * mul_bf16(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    // Separate input edges also round each derivative before fan-in. Merely
    // rounding the forward result would accumulate F32 products in backward.
    return round(ctx, ggml_mul(ctx, round(ctx, a), round(ctx, b)));
}
inline ggml_tensor * silu_bf16(ggml_context * ctx, ggml_tensor * x) {
    return round(ctx, ggml_silu(ctx, round(ctx, x)));
}
inline ggml_tensor * rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps) {
    // RMSNorm explicitly normalizes in F32, casts, then multiplies BF16 weight.
    return mul_bf16(ctx, round(ctx, ggml_rms_norm(ctx, round(ctx, x), eps)), f32(ctx, weight));
}
// Stack [n_i, S] F32 tensors on ne0. CONCAT has no backward, so the parts are
// accumulated into a zero canvas, as join_halves below does for two halves.
inline ggml_tensor * join_rows(ggml_context * ctx, const std::vector<ggml_tensor *> & parts) {
    if (parts.size() == 1) return parts[0];
    int64_t total = 0;
    for (ggml_tensor * p : parts) {
        GGML_ASSERT(p->type == GGML_TYPE_F32 && p->ne[1] == parts[0]->ne[1] && p->ne[2] == 1 && p->ne[3] == 1);
        total += p->ne[0];
    }
    GGML_ASSERT(total % parts[0]->ne[0] == 0);  // ggml_repeat builds the canvas
    ggml_tensor * shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total, parts[0]->ne[1]);
    ggml_tensor * out = ggml_scale(ctx, ggml_repeat(ctx, parts[0], shape), 0.0f);
    size_t offset = 0;
    for (ggml_tensor * p : parts) {
        out = ggml_acc(ctx, out, p, out->nb[1], out->nb[2], out->nb[3], offset);
        offset += size_t(p->ne[0]) * sizeof(float);
    }
    return out;
}
inline ggml_tensor * linear(ggml_context * ctx, const Yue2AitkConvRotLinear & w,
                            ggml_tensor * x, const Yue2AitkFusedLora * adapter = nullptr,
                            ggml_tensor * bias = nullptr) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == w.cols);
    ggml_tensor * base;
    if (w.dense) {
        // Companion full-weight replacement (llm2vae): F32 maths, BF16 output.
        base = mm_f32(ctx, w.dense, round(ctx, x));
        if (bias) base = ggml_add(ctx, base, f32(ctx, bias));
        base = round(ctx, base);
    } else if (!w.parts.empty()) {
        // GGUF base: the same BF16 value boundaries as ConvRot's compute_bf16
        // (rounded input, bias and output), over the GGUF weight in its own type.
        ggml_tensor * xr = round(ctx, x);
        std::vector<ggml_tensor *> ys;
        for (ggml_tensor * p : w.parts) {
            ys.push_back(mm(ctx, p, xr));
        }
        base = join_rows(ctx, ys);
        if (bias) base = ggml_add(ctx, base, round(ctx, f32(ctx, bias)));
        base = round(ctx, base);
    } else {
        base = ggml_convrot8(ctx, w.weight_i8, x, w.scales_f32,
                             bias ? f32(ctx, bias) : nullptr, w.rotation, true);
    }
    if (w.frozen_a) {
        // Companion decoder LoRA: frozen, same boundaries as the trainable branch.
        ggml_tensor * ca = mm_f32(ctx, w.frozen_a, round(ctx, x));
        ggml_tensor * cd = mm_f32(ctx, w.frozen_b, ca);
        base = round(ctx, ggml_add(ctx, base, round(ctx, cd)));
    }
    if (!adapter) return base;
    if (adapter->is_lokr()) {
        // Same boundary as the LoRA branch: BF16-rounded input, BF16 delta.
        // lokr-apply.h is shared with the DiT trainer, so the BF16 factor
        // copies are made here (w1 scaled in F32 first); strict keeps F32.
        // [out_k, out_l*S] is the same bytes as [out, S].
        LokrApplySite site = adapter->lokr;
        if (!strict_f32()) {
            auto bf16 = [&](ggml_tensor * t) { return t ? ggml_cast(ctx, t, GGML_TYPE_BF16) : nullptr; };
            site.w1 = bf16(site.scale == 1.0f ? site.w1 : ggml_scale(ctx, site.w1, site.scale));
            site.scale = 1.0f;
            site.w2 = bf16(site.w2); site.w2_a = bf16(site.w2_a); site.w2_b = bf16(site.w2_b);
        }
        gemm_counts().bf16 += strict_f32() ? 0 : 1;
        gemm_counts().f32 += strict_f32() ? 1 : 0;
        ggml_tensor * d = lokr_apply_delta(ctx, round(ctx, x), site);
        ggml_tensor * delta = round(ctx, ggml_reshape_2d(ctx, d, base->ne[0], base->ne[1]));
        return round(ctx, ggml_add(ctx, base, delta));
    }
    // FP32 trainables; the products use their BF16 casts (strict: F32 maths).
    // Only the branch output is BF16-rounded.
    ggml_tensor * ax = mm(ctx, adapter->a, round(ctx, x));
    ggml_tensor * delta = mm(ctx, adapter->b, ax);
    // x * 1.0f is exact: alpha == rank skips the scale pass and its backward.
    // Bit-identical either way, so this has no rollback switch.
    delta = round(ctx, adapter->scale == 1.0f ? delta : ggml_scale(ctx, delta, adapter->scale));
    return round(ctx, ggml_add(ctx, base, delta));
}
inline ggml_tensor * slice_rows(ggml_context * ctx, ggml_tensor * x, int64_t first, int64_t width) {
    GGML_ASSERT(first >= 0 && width > 0 && first + width <= x->ne[0]);
    return ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, x, width, x->ne[1], x->nb[1],
                                                       size_t(first) * sizeof(float))), width, x->ne[1]);
}
// Output rows [first, first + width) of a [in, out] weight: a contiguous view.
inline ggml_tensor * weight_rows(ggml_context * ctx, ggml_tensor * w, int64_t first, int64_t width) {
    GGML_ASSERT(first >= 0 && width > 0 && first + width <= w->ne[1]);
    return ggml_view_2d(ctx, w, w->ne[0], width, w->nb[1], size_t(first) * w->nb[1]);
}
// YUE2_AITK_FUSED_SLICE=1 keeps the fused linear + output slices in bf16 mode.
inline bool fused_slice() {
    static const bool on = [] { const char * v = std::getenv("YUE2_AITK_FUSED_SLICE"); return v && *v && *v != '0'; }();
    return on;
}
// A fused linear (q|k|v, gate|up) split into its outputs, each [width, S].
// Over one fused GGUF weight with a LoRA (or no) adapter, each output is its
// own GEMM over a row view of the weight, of the companion B and of the LoRA
// B, with the same BF16 boundaries as linear(); the fused output is never
// built, so neither are its slice copies nor, in the backward, the zero
// canvas their gradients accumulate into. Anything else runs linear() and
// slices it.
inline std::vector<ggml_tensor *> linear_split(ggml_context * ctx, const Yue2AitkConvRotLinear & w,
                                               ggml_tensor * x, const Yue2AitkFusedLora * adapter,
                                               const std::vector<int64_t> & widths) {
    std::vector<ggml_tensor *> out;
    if (strict_f32() || fused_slice() || w.dense || w.parts.size() != 1 || (adapter && adapter->is_lokr())) {
        ggml_tensor * y = linear(ctx, w, x, adapter);
        int64_t first = 0;
        for (int64_t width : widths) { out.push_back(slice_rows(ctx, y, first, width)); first += width; }
        return out;
    }
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == w.cols);
    ggml_tensor * xr = round(ctx, x);
    ggml_tensor * ca = w.frozen_a ? mm_f32(ctx, w.frozen_a, round(ctx, x)) : nullptr;
    ggml_tensor * ax = adapter ? mm(ctx, adapter->a, round(ctx, x)) : nullptr;
    int64_t first = 0;
    for (int64_t width : widths) {
        ggml_tensor * base = round(ctx, mm(ctx, weight_rows(ctx, w.parts[0], first, width), xr));
        if (ca) base = round(ctx, ggml_add(ctx, base, round(ctx, mm_f32(ctx, weight_rows(ctx, w.frozen_b, first, width), ca))));
        if (ax) {
            ggml_tensor * delta = mm(ctx, weight_rows(ctx, adapter->b, first, width), ax);
            delta = round(ctx, adapter->scale == 1.0f ? delta : ggml_scale(ctx, delta, adapter->scale));
            base = round(ctx, ggml_add(ctx, base, delta));
        }
        out.push_back(base);
        first += width;
    }
    GGML_ASSERT(first == w.rows);
    return out;
}
inline ggml_tensor * join_halves(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));
    // CONCAT has no backward in this GGML revision. Accumulating disjoint
    // halves into a zero canvas avoids SET backward's noncontiguous NEG path.
    ggml_tensor * shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2*a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
    ggml_tensor * canvas = ggml_scale(ctx, ggml_repeat(ctx, a, shape), 0.0f);
    ggml_tensor * first = ggml_acc(ctx, canvas, a, canvas->nb[1], canvas->nb[2], canvas->nb[3], 0);
    ggml_tensor * both = ggml_acc(ctx, first, b, canvas->nb[1], canvas->nb[2], canvas->nb[3],
                                  size_t(a->ne[0]) * sizeof(float));
    return ggml_reshape_4d(ctx, both, shape->ne[0], shape->ne[1], shape->ne[2], shape->ne[3]);
}
// YUE2_AITK_ROPE_TABLE=1 keeps the BF16 table RoPE below in bf16 mode too.
inline bool rope_table() {
    static const bool on = [] { const char * v = std::getenv("YUE2_AITK_ROPE_TABLE"); return v && *v && *v != '0'; }();
    return on;
}
inline bool qk_prescale() {
    static const bool on = [] { const char * v = std::getenv("YUE2_AITK_QK_PRESCALE"); return v && *v && *v != '0'; }();
    return on;
}
inline ggml_tensor * rope(ggml_context * ctx, ggml_tensor * x,
                          ggml_tensor * cosine, ggml_tensor * sine, ggml_tensor * positions = nullptr) {
    if (positions && !strict_f32() && !rope_table()) {
        // One NeoX RoPE op with F32 trig and one BF16 round of its output, the
        // op the YuE2 inference graph runs (theta 1e6, as trig() below). The
        // table path's per-product rounds go, and with them a dozen launches
        // per call and their backward.
        return round(ctx, ggml_rope_ext(ctx, x, positions, nullptr, int(x->ne[0]), GGML_ROPE_TYPE_NEOX, 0,
                                        1000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f));
    }
    // x [D,heads,S,1], trig [D/2,1,S,1], already BF16-rounded constants.
    const int64_t half = x->ne[0] / 2;
    GGML_ASSERT(x->ne[0] % 2 == 0 && cosine->ne[0] == half && sine->ne[0] == half);
    ggml_tensor * a = ggml_cont(ctx, ggml_view_4d(ctx, x, half, x->ne[1], x->ne[2], 1,
                                                x->nb[1], x->nb[2], x->nb[3], 0));
    ggml_tensor * b = ggml_cont(ctx, ggml_view_4d(ctx, x, half, x->ne[1], x->ne[2], 1,
                                                x->nb[1], x->nb[2], x->nb[3], size_t(half)*sizeof(float)));
    // Reference eager PyTorch rounds each multiplication before add/subtract.
    ggml_tensor * ac = mul_bf16(ctx, a, cosine);
    ggml_tensor * bs = mul_bf16(ctx, b, sine);
    ggml_tensor * bc = mul_bf16(ctx, b, cosine);
    ggml_tensor * as = mul_bf16(ctx, a, sine);
    return join_halves(ctx, round(ctx, ggml_sub(ctx, ac, bs)), round(ctx, ggml_add(ctx, bc, as)));
}

inline Yue2AitkBlockResult block(ggml_context * ctx, const Yue2AitkGraphConfig & c,
                                 const Yue2AitkLayerWeights & w, const Yue2AitkBlockNorms & norms,
                                 const Yue2AitkLayerAdapters * adapters, ggml_tensor * h,
                                 ggml_tensor * cosine, ggml_tensor * sine, ggml_tensor * mask,
                                 ggml_tensor * prefix_k_canvas = nullptr,
                                 ggml_tensor * prefix_v_canvas = nullptr, int64_t prefix_length = 0,
                                 int64_t causal_hint = -1, ggml_tensor * positions = nullptr,
                                 ggml_tensor * saved_attention = nullptr,
                                 bool unmasked_attention = false) {
    const int64_t S = h->ne[1], D = c.head_dim, Q = D*c.heads, KV = D*c.kv_heads;
    GGML_ASSERT(c.hidden == Q && h->ne[0] == c.hidden && prefix_length >= 0);
    GGML_ASSERT((prefix_k_canvas == nullptr) == (prefix_v_canvas == nullptr));
    h = round(ctx, h);
    ggml_tensor * n = rms(ctx, h, norms.input, c.rms_eps);
    const std::vector<ggml_tensor *> qkv = linear_split(ctx, w.qkv, n, adapters ? &adapters->qkv : nullptr, {Q, KV, KV});
    ggml_tensor * q = ggml_reshape_4d(ctx, qkv[0], D, c.heads, S, 1);
    ggml_tensor * k = ggml_reshape_4d(ctx, qkv[1], D, c.kv_heads, S, 1);
    ggml_tensor * v = ggml_reshape_4d(ctx, qkv[2], D, c.kv_heads, S, 1);
    q = rope(ctx, rms(ctx, q, norms.q, c.rms_eps), cosine, sine, positions);
    k = rope(ctx, rms(ctx, k, norms.k, c.rms_eps), cosine, sine, positions);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    Yue2AitkBlockResult out; out.key = k; out.value = v;
    if (prefix_k_canvas) {
        GGML_ASSERT(prefix_k_canvas->ne[0] == D && prefix_k_canvas->ne[1] == prefix_length+S &&
                    prefix_k_canvas->ne[2] == c.kv_heads && ggml_are_same_shape(prefix_k_canvas, prefix_v_canvas));
        GGML_ASSERT(!(prefix_k_canvas->flags & GGML_TENSOR_FLAG_PARAM) &&
                    !(prefix_v_canvas->flags & GGML_TENSOR_FLAG_PARAM));
        GGML_ASSERT(prefix_k_canvas->op == GGML_OP_NONE && prefix_v_canvas->op == GGML_OP_NONE);
        // Callers supply detached prefix followed by S zero rows in each canvas.
        k = ggml_reshape_4d(ctx, ggml_cont(ctx, k), D, S, c.kv_heads, 1);
        v = ggml_reshape_4d(ctx, ggml_cont(ctx, v), D, S, c.kv_heads, 1);
        const size_t offset = size_t(prefix_length) * prefix_k_canvas->nb[1];
        k = ggml_reshape_4d(ctx, ggml_set(ctx, prefix_k_canvas, k, prefix_k_canvas->nb[1],
                    prefix_k_canvas->nb[2], prefix_k_canvas->nb[3], offset), D, prefix_length+S, c.kv_heads, 1);
        v = ggml_reshape_4d(ctx, ggml_set(ctx, prefix_v_canvas, v, prefix_v_canvas->nb[1],
                    prefix_v_canvas->nb[2], prefix_v_canvas->nb[3], offset), D, prefix_length+S, c.kv_heads, 1);
    }
    // Torch math SDPA scales Q and K independently before the dot product.
    // Post-dot scaling differs at BF16 rounding boundaries and is amplified
    // by subsequent activation quantization. Keep these products in F32.
    const float root_scale = std::sqrt(1.0f/std::sqrt(float(D)));
    // BF16 attention can instead take the BF16 q and k as they are and apply
    // 1/sqrt(D) to the F32 scores (YUE2_AITK_QK_PRESCALE=1 keeps the above).
    const bool fold_scale = !math_attention() && !strict_f32() && c.attention_precision == GGML_PREC_DEFAULT &&
        bf16_attention_supported() && !qk_prescale();
    if (!fold_scale) {
        q = ggml_scale(ctx, ggml_cont(ctx, q), root_scale);
        k = ggml_scale(ctx, ggml_cont(ctx, k), root_scale);
    }
    ggml_tensor * attention;
#ifndef YUE2_AITK_DIAGNOSTIC_MATH_ATTENTION
    if (math_attention())
#endif
    {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    ggml_tensor * probabilities = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);
    ggml_tensor * weighted = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), probabilities);
    ggml_mul_mat_set_prec(weighted, GGML_PREC_F32);
    attention = round(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, weighted, 0, 2, 1, 3)), Q, S));
    }
#ifndef YUE2_AITK_DIAGNOSTIC_MATH_ATTENTION
    else {
    // S2c: BF16 attention is gated on a runner-verified capability, not just
    // the requested precision. strict-f32, YUE2_AITK_STRICT_F32/BASELINE and an
    // unsupported backend all keep the F32/TF32 dispatch with no cast node.
    const bool bf16_attention = !strict_f32() && c.attention_precision == GGML_PREC_DEFAULT &&
        bf16_attention_supported();
    ggml_tensor * attn_q = bf16_attention ? ggml_cast(ctx, q, GGML_TYPE_BF16) : q;
    ggml_tensor * attn_k = bf16_attention ? ggml_cast(ctx, k, GGML_TYPE_BF16) : k;
    ggml_tensor * attn_v = bf16_attention ? ggml_cast(ctx, v, GGML_TYPE_BF16) : v;
    // unmasked_attention: every key is visible (the NAR). No mask and no causal hint then,
    // so the Metal attention backward may take the dS-materialisation path.
    ggml_tensor * fa_mask = unmasked_attention ? nullptr : mask;
    if (unmasked_attention) causal_hint = -1;
    ggml_tensor * packed = ggml_flash_attn_train(ctx, attn_q, attn_k, attn_v, fa_mask, fold_scale ? root_scale*root_scale : 1.0f);
    ggml_flash_attn_train_set_prec(packed, bf16_attention ? GGML_PREC_BF16 : c.attention_precision);
    // The mask is exactly "key j visible iff j <= causal_hint + row" (the
    // executor's make_mask), so CUDA skips reading it; the prefix rows' dK/dV
    // go into the detached canvas, whose SET backward never reads them.
    if (causal_hint >= 0) ggml_flash_attn_train_set_causal(packed, int32_t(causal_hint));
    if (prefix_k_canvas && prefix_length > 0) ggml_flash_attn_train_set_kv_grad_start(packed, int32_t(prefix_length));
    // A saved copy of this node's forward result (same inputs, same op params)
    // lets the Metal backend copy instead of recompute. Type/shape are checked
    // here so a mismatched slot silently falls back to the recompute.
    if (saved_attention && saved_attention->type == GGML_TYPE_F32 && ggml_is_contiguous(saved_attention) &&
        ggml_nelements(saved_attention) == ggml_nelements(packed)) ggml_flash_attn_train_set_saved(packed, saved_attention);
    out.attention_packed = packed;
    attention = round(ctx, ggml_reshape_2d(ctx, ggml_flash_attn_train_get_o(ctx, packed), Q, S));
    }
#endif
    h = round(ctx, ggml_add(ctx, h, linear(ctx, w.output, attention, adapters ? &adapters->output : nullptr)));
    n = rms(ctx, h, norms.post_attention, c.rms_eps);
    const std::vector<ggml_tensor *> gu = linear_split(ctx, w.gate_up, n, adapters ? &adapters->gate_up : nullptr,
                                                       {c.intermediate, c.intermediate});
    ggml_tensor * gate = silu_bf16(ctx, gu[0]);
    ggml_tensor * up = gu[1];
    ggml_tensor * activation = mul_bf16(ctx, gate, up);
    out.hidden = round(ctx, ggml_add(ctx, h, linear(ctx, w.down, activation, adapters ? &adapters->down : nullptr)));
    return out;
}
} // namespace yue2_aitk_graph
