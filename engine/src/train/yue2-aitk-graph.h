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
};

namespace yue2_aitk_graph {
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
inline ggml_tensor * linear(ggml_context * ctx, const Yue2AitkConvRotLinear & w,
                            ggml_tensor * x, const Yue2AitkFusedLora * adapter = nullptr,
                            ggml_tensor * bias = nullptr) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == w.cols);
    ggml_tensor * base = ggml_convrot8(ctx, w.weight_i8, x, w.scales_f32,
                                      bias ? f32(ctx, bias) : nullptr, w.rotation, true);
    if (!adapter) return base;
    // FP32 trainables and FP32 LoRA math; only the branch output is BF16.
    ggml_tensor * ax = ggml_mul_mat(ctx, adapter->a, round(ctx, x));
    ggml_mul_mat_set_prec(ax, GGML_PREC_F32);
    ggml_tensor * delta = ggml_mul_mat(ctx, adapter->b, ax);
    ggml_mul_mat_set_prec(delta, GGML_PREC_F32);
    delta = round(ctx, ggml_scale(ctx, delta, adapter->scale));
    return round(ctx, ggml_add(ctx, base, delta));
}
inline ggml_tensor * slice_rows(ggml_context * ctx, ggml_tensor * x, int64_t first, int64_t width) {
    GGML_ASSERT(first >= 0 && width > 0 && first + width <= x->ne[0]);
    return ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, x, width, x->ne[1], x->nb[1],
                                                       size_t(first) * sizeof(float))), width, x->ne[1]);
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
inline ggml_tensor * rope(ggml_context * ctx, ggml_tensor * x,
                          ggml_tensor * cosine, ggml_tensor * sine) {
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
                                 ggml_tensor * prefix_v_canvas = nullptr, int64_t prefix_length = 0) {
    const int64_t S = h->ne[1], D = c.head_dim, Q = D*c.heads, KV = D*c.kv_heads;
    GGML_ASSERT(c.hidden == Q && h->ne[0] == c.hidden && prefix_length >= 0);
    GGML_ASSERT((prefix_k_canvas == nullptr) == (prefix_v_canvas == nullptr));
    h = round(ctx, h);
    ggml_tensor * n = rms(ctx, h, norms.input, c.rms_eps);
    ggml_tensor * qkv = linear(ctx, w.qkv, n, adapters ? &adapters->qkv : nullptr);
    ggml_tensor * q = ggml_reshape_4d(ctx, slice_rows(ctx, qkv, 0, Q), D, c.heads, S, 1);
    ggml_tensor * k = ggml_reshape_4d(ctx, slice_rows(ctx, qkv, Q, KV), D, c.kv_heads, S, 1);
    ggml_tensor * v = ggml_reshape_4d(ctx, slice_rows(ctx, qkv, Q+KV, KV), D, c.kv_heads, S, 1);
    q = rope(ctx, rms(ctx, q, norms.q, c.rms_eps), cosine, sine);
    k = rope(ctx, rms(ctx, k, norms.k, c.rms_eps), cosine, sine);
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
    q = ggml_scale(ctx, ggml_cont(ctx, q), root_scale);
    k = ggml_scale(ctx, ggml_cont(ctx, k), root_scale);
#ifdef YUE2_AITK_DIAGNOSTIC_MATH_ATTENTION
    // Development-only quadratic graph to isolate arithmetic differences.
    // Never enable this diagnostic in a shipped training runtime.
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    ggml_tensor * probabilities = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);
    ggml_tensor * weighted = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), probabilities);
    ggml_mul_mat_set_prec(weighted, GGML_PREC_F32);
    ggml_tensor * attention = round(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, weighted, 0, 2, 1, 3)), Q, S));
#else
    ggml_tensor * packed = ggml_flash_attn_train(ctx, q, k, v, mask, 1.0f);
    ggml_flash_attn_train_set_prec(packed, c.attention_precision);
    ggml_tensor * attention = round(ctx, ggml_reshape_2d(ctx, ggml_flash_attn_train_get_o(ctx, packed), Q, S));
#endif
    h = round(ctx, ggml_add(ctx, h, linear(ctx, w.output, attention, adapters ? &adapters->output : nullptr)));
    n = rms(ctx, h, norms.post_attention, c.rms_eps);
    ggml_tensor * gu = linear(ctx, w.gate_up, n, adapters ? &adapters->gate_up : nullptr);
    ggml_tensor * gate = silu_bf16(ctx, slice_rows(ctx, gu, 0, c.intermediate));
    ggml_tensor * up = slice_rows(ctx, gu, c.intermediate, c.intermediate);
    ggml_tensor * activation = mul_bf16(ctx, gate, up);
    out.hidden = round(ctx, ggml_add(ctx, h, linear(ctx, w.down, activation, adapters ? &adapters->down : nullptr)));
    return out;
}
} // namespace yue2_aitk_graph
