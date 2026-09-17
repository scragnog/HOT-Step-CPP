#pragma once

// AI Toolkit-compatible YuE2 fused LoRA allocation seam.
//
// This is deliberately separate from the Legacy AR/NAR adapter makers. The
// reference trains one shared input factor for fused QKV and gate/up sites;
// the output factor is sliced for each projection at use time.

#include "../../ggml/include/ggml.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct Yue2AitkDims {
    int64_t hidden = 0;
    int64_t q_width = 0;
    int64_t kv_width = 0;
    int64_t feed_forward = 0;
};

struct Yue2AitkFusedLora {
    ggml_tensor * a = nullptr; // [input, rank], shared by all output slices
    ggml_tensor * b = nullptr; // [rank, fused_output]
    int64_t input_width = 0;
    int64_t output_width = 0;
    int64_t rank = 0;
    float scale = 1.0f;

    bool valid() const { return a && b && input_width > 0 && output_width > 0 && rank > 0; }
};

struct Yue2AitkLayerAdapters {
    Yue2AitkFusedLora qkv;      // q, k and v slices of one [rank, Q+K+K] B
    Yue2AitkFusedLora output;   // [rank, H]
    Yue2AitkFusedLora gate_up;  // gate and up slices of one [rank, F+F] B
    Yue2AitkFusedLora down;     // [rank, H]
};

struct Yue2AitkExpertAdapters {
    std::vector<Yue2AitkLayerAdapters> layers;
    std::vector<ggml_tensor *> params; // each unique A/B tensor exactly once
};

// Allocate one fused adapter collection. `params` receives eight tensors per
// layer (four A/B pairs), rather than fourteen tensors from the Legacy split
// maker. Both factors are F32 trainables and are marked with ggml_set_param.
bool yue2_aitk_make_expert_adapters(
    ggml_context *ctx,
    const Yue2AitkDims &dims,
    int n_layers,
    int64_t rank,
    float alpha,
    Yue2AitkExpertAdapters *out,
    const std::string &expert = "ar");

size_t yue2_aitk_unique_param_count(const Yue2AitkExpertAdapters &adapters);

// Return scale * B_slice * (A * x), where x is [input, columns]. `offset`
// addresses output rows in the fused B tensor and must stay within its fused
// output width. This helper creates views only; it never allocates factors.
ggml_tensor *yue2_aitk_fused_delta(
    ggml_context *ctx,
    ggml_tensor *x,
    const Yue2AitkFusedLora &site,
    int64_t offset,
    int64_t width);

// Inline implementation keeps this seam usable by the trainer's header-only
// graph builders while keeping Legacy makers untouched.
namespace yue2_aitk_detail {
inline bool valid_dims(const Yue2AitkDims &d) {
    return d.hidden > 0 && d.hidden <= 65536 && d.q_width > 0 && d.q_width <= 65536 &&
        d.kv_width > 0 && d.kv_width <= 65536 && d.feed_forward > 0 && d.feed_forward <= 262144;
}
inline bool make_site(ggml_context *ctx, int64_t input, int64_t output, int64_t rank,
                     float scale, const std::string &name, Yue2AitkFusedLora *site,
                     std::vector<ggml_tensor *> *params) {
    site->a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input, rank);
    site->b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, output);
    if (!site->a || !site->b) return false;
    site->input_width = input; site->output_width = output; site->rank = rank; site->scale = scale;
    ggml_set_name(site->a, (name + ".A").c_str()); ggml_set_name(site->b, (name + ".B").c_str());
    ggml_set_param(site->a); ggml_set_param(site->b);
    params->push_back(site->a); params->push_back(site->b);
    return true;
}
} // namespace yue2_aitk_detail

inline bool yue2_aitk_make_expert_adapters(ggml_context *ctx, const Yue2AitkDims &dims,
                                           int n_layers, int64_t rank, float alpha,
                                           Yue2AitkExpertAdapters *out, const std::string &expert) {
    if (!ctx || !out || !yue2_aitk_detail::valid_dims(dims) || n_layers <= 0 || n_layers > 256 || rank <= 0 || rank > 65536 ||
        (expert != "ar" && expert != "nar") ||
        !std::isfinite(alpha) || alpha <= 0.0f) return false;
    const float scale = alpha / static_cast<float>(rank);
    out->layers.clear(); out->params.clear(); out->layers.resize(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        const std::string prefix = expert + ".blk." + std::to_string(layer) + ".";
        auto &dst = out->layers[static_cast<size_t>(layer)];
        if (!yue2_aitk_detail::make_site(ctx, dims.hidden, dims.q_width + 2 * dims.kv_width, rank, scale, prefix + "attn_qkv", &dst.qkv, &out->params) ||
            !yue2_aitk_detail::make_site(ctx, dims.q_width, dims.hidden, rank, scale, prefix + "attn_output", &dst.output, &out->params) ||
            !yue2_aitk_detail::make_site(ctx, dims.hidden, 2 * dims.feed_forward, rank, scale, prefix + "ffn_gate_up", &dst.gate_up, &out->params) ||
            !yue2_aitk_detail::make_site(ctx, dims.feed_forward, dims.hidden, rank, scale, prefix + "ffn_down", &dst.down, &out->params)) {
            out->layers.clear(); out->params.clear(); return false;
        }
    }
    return true;
}

inline size_t yue2_aitk_unique_param_count(const Yue2AitkExpertAdapters &adapters) {
    std::vector<const ggml_tensor *> seen;
    for (ggml_tensor *tensor : adapters.params) {
        if (!tensor) continue;
        bool duplicate = false;
        for (const ggml_tensor *prior : seen) if (prior == tensor) { duplicate = true; break; }
        if (!duplicate) seen.push_back(tensor);
    }
    return seen.size();
}

inline ggml_tensor *yue2_aitk_fused_delta(ggml_context *ctx, ggml_tensor *x,
                                          const Yue2AitkFusedLora &site,
                                          int64_t offset, int64_t width) {
    if (!ctx || !x || !site.valid() || offset < 0 || width <= 0 || offset > site.output_width - width || x->ne[0] != site.input_width) return nullptr;
    ggml_tensor *ax = ggml_mul_mat(ctx, site.a, x);
    if (!ax) return nullptr;
    ggml_tensor *b_slice = ggml_view_2d(ctx, site.b, site.rank, width, site.b->nb[1], static_cast<size_t>(offset) * site.b->nb[1]);
    if (!b_slice) return nullptr;
    ggml_tensor *delta = ggml_mul_mat(ctx, b_slice, ax);
    return delta ? ggml_scale(ctx, delta, site.scale) : nullptr;
}
