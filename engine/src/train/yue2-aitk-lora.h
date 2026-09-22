#pragma once

// AI Toolkit-compatible YuE2 fused LoRA allocation seam.
//
// This is deliberately separate from the Legacy AR/NAR adapter makers. The
// reference trains one shared input factor for fused QKV and gate/up sites;
// the output factor is sliced for each projection at use time.

#include "../../ggml/include/ggml.h"
#include "../lokr-apply.h"
#include "../lokr-common.h"

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

// One adapted site: the whole fused base matrix (qkv, o, gate_up or down)
// carries ONE delta. Either a LoRA pair (a, b) or a LoKr factor set (lokr);
// is_lokr() tells which, and every consumer (graph, slots, exporters)
// branches on it. The struct keeps its LoRA-era name because it is the type
// every file names.
struct Yue2AitkFusedLora {
    ggml_tensor * a = nullptr; // [input, rank], shared by all output slices
    ggml_tensor * b = nullptr; // [rank, fused_output]
    int64_t input_width = 0;
    int64_t output_width = 0;
    int64_t rank = 0;
    float scale = 1.0f;
    // LoKr: kron(w1, w2) over the same fused matrix. lokr.scale carries
    // alpha/dim (forced to 1 on monolithic-w2 sites, LyCORIS K6).
    LokrApplySite lokr;

    bool is_lokr() const { return lokr.w1 != nullptr; }
    bool valid() const {
        if (input_width <= 0 || output_width <= 0) return false;
        return is_lokr() ? lokr.valid() : (a && b && rank > 0);
    }
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

// One trainable factor of one site. The trainer's slot list, the block
// executor's gradient download and the exporter all walk the SAME sequence
// (site order qkv, o, gate_up, down; factors in the order pushed here), so a
// new parameterization only has to add its factors in one place.
// `zero_init` marks the factor that starts at zero so dW = 0 at step 0.
struct Yue2AitkSiteParam {
    ggml_tensor * tensor = nullptr;
    const char *  suffix = "";  // PEFT-style tensor suffix, e.g. ".lora_A.weight"
    bool          zero_init = false;
};
constexpr int kYue2AitkSites = 4;
inline const Yue2AitkFusedLora & yue2_aitk_site(const Yue2AitkLayerAdapters & l, int s) {
    return s == 0 ? l.qkv : s == 1 ? l.output : s == 2 ? l.gate_up : l.down;
}
inline void yue2_aitk_site_params(const Yue2AitkFusedLora & site, std::vector<Yue2AitkSiteParam> * out) {
    if (site.is_lokr()) {
        // Same init as dit-adapter-lokr.h: w1 and w2_a kaiming, w2 / w2_b zero.
        out->push_back({site.lokr.w1, ".lokr_w1", false});
        if (site.lokr.mono) {
            out->push_back({site.lokr.w2, ".lokr_w2", true});
        } else {
            out->push_back({site.lokr.w2_a, ".lokr_w2_a", false});
            out->push_back({site.lokr.w2_b, ".lokr_w2_b", true});
        }
        return;
    }
    out->push_back({site.a, ".lora_A.weight", false});
    out->push_back({site.b, ".lora_B.weight", true});
}
inline std::vector<ggml_tensor *> yue2_aitk_layer_params(const Yue2AitkLayerAdapters & l) {
    std::vector<Yue2AitkSiteParam> params;
    for (int s = 0; s < kYue2AitkSites; ++s) yue2_aitk_site_params(yue2_aitk_site(l, s), &params);
    std::vector<ggml_tensor *> out;
    out.reserve(params.size());
    for (const auto & p : params) out.push_back(p.tensor);
    return out;
}

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

// ── LoKr sites ──────────────────────────────────────────────────────────────
//
// The native split export (yue2-aitk-native-adapter-io.h) slices the fused
// qkv / gate_up deltas into per-projection files by ROW. kron row index is
// l*out_k + k, so a row range [r0, r1) is kron(w1[r0/out_k : r1/out_k], w2)
// exactly when out_k divides both r0 and r1. Refuse a factor that does not
// land on the q/k/v and gate/up boundaries rather than export something the
// loader would silently mis-slice.
inline bool yue2_aitk_lokr_slices_ok(const Yue2AitkDims & d, int factor, std::string * why) {
    int64_t out_l = 0, out_k = 0;
    lokr_factorization(d.q_width + 2 * d.kv_width, factor, &out_l, &out_k);
    if (d.q_width % out_k != 0 || d.kv_width % out_k != 0) {
        if (why) *why = "--lokr-factor " + std::to_string(factor) + " splits the fused qkv output into " + std::to_string(out_l) + "x" + std::to_string(out_k) + ", and " + std::to_string(out_k) + " does not divide the q/k/v boundaries";
        return false;
    }
    lokr_factorization(2 * d.feed_forward, factor, &out_l, &out_k);
    if (d.feed_forward % out_k != 0) {
        if (why) *why = "--lokr-factor " + std::to_string(factor) + " splits the fused gate_up output into " + std::to_string(out_l) + "x" + std::to_string(out_k) + ", and " + std::to_string(out_k) + " does not divide the gate/up boundary";
        return false;
    }
    return true;
}

namespace yue2_aitk_detail {
inline bool make_lokr_site(ggml_context *ctx, int64_t input, int64_t output, int dim, int factor,
                          float alpha, const std::string &name, Yue2AitkFusedLora *site,
                          std::vector<ggml_tensor *> *params) {
    LokrApplySite & k = site->lokr;
    lokr_factorization(output, factor, &k.out_l, &k.out_k);
    lokr_factorization(input, factor, &k.in_m, &k.in_n);
    if (k.out_l * k.out_k != output || k.in_m * k.in_n != input) return false;
    k.dim  = dim;
    k.mono = lokr_w2_mono(dim, k.out_k, k.in_n);
    // K6: LyCORIS forces alpha = dim (scale 1) when both factors are monolithic.
    const float a_eff = k.mono ? (float) dim : alpha;
    k.scale = a_eff / (float) dim;
    k.order = lokr_pick_order(k.mono, k.in_m, k.in_n, k.out_l, k.out_k, dim);
    k.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_m, k.out_l);
    if (!k.w1) return false;
    ggml_set_name(k.w1, (name + ".w1").c_str()); ggml_set_param(k.w1); params->push_back(k.w1);
    if (k.mono) {
        k.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_n, k.out_k);
        if (!k.w2) return false;
        ggml_set_name(k.w2, (name + ".w2").c_str()); ggml_set_param(k.w2); params->push_back(k.w2);
    } else {
        k.w2_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, k.out_k);
        k.w2_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_n, dim);
        if (!k.w2_a || !k.w2_b) return false;
        ggml_set_name(k.w2_a, (name + ".w2_a").c_str()); ggml_set_param(k.w2_a); params->push_back(k.w2_a);
        ggml_set_name(k.w2_b, (name + ".w2_b").c_str()); ggml_set_param(k.w2_b); params->push_back(k.w2_b);
    }
    site->input_width = input; site->output_width = output; site->rank = dim; site->scale = k.scale;
    return true;
}
} // namespace yue2_aitk_detail

// LoKr twin of yue2_aitk_make_expert_adapters: the same four fused sites per
// layer, each carrying kron factors instead of an A/B pair.
inline bool yue2_aitk_make_expert_adapters_lokr(ggml_context *ctx, const Yue2AitkDims &dims,
                                                int n_layers, int dim, int factor, float alpha,
                                                Yue2AitkExpertAdapters *out, const std::string &expert) {
    if (!ctx || !out || !yue2_aitk_detail::valid_dims(dims) || n_layers <= 0 || n_layers > 256 ||
        dim <= 0 || dim > 65536 || factor == 0 || factor < -1 || factor > 65536 ||
        (expert != "ar" && expert != "nar") || !std::isfinite(alpha) || alpha <= 0.0f ||
        !yue2_aitk_lokr_slices_ok(dims, factor, nullptr)) return false;
    out->layers.clear(); out->params.clear(); out->layers.resize(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        const std::string prefix = expert + ".blk." + std::to_string(layer) + ".";
        auto &dst = out->layers[static_cast<size_t>(layer)];
        if (!yue2_aitk_detail::make_lokr_site(ctx, dims.hidden, dims.q_width + 2 * dims.kv_width, dim, factor, alpha, prefix + "attn_qkv", &dst.qkv, &out->params) ||
            !yue2_aitk_detail::make_lokr_site(ctx, dims.q_width, dims.hidden, dim, factor, alpha, prefix + "attn_output", &dst.output, &out->params) ||
            !yue2_aitk_detail::make_lokr_site(ctx, dims.hidden, 2 * dims.feed_forward, dim, factor, alpha, prefix + "ffn_gate_up", &dst.gate_up, &out->params) ||
            !yue2_aitk_detail::make_lokr_site(ctx, dims.feed_forward, dims.hidden, dim, factor, alpha, prefix + "ffn_down", &dst.down, &out->params)) {
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
    if (!ctx || !x || !site.valid() || site.is_lokr() || offset < 0 || width <= 0 || offset > site.output_width - width || x->ne[0] != site.input_width) return nullptr;
    ggml_tensor *ax = ggml_mul_mat(ctx, site.a, x);
    if (!ax) return nullptr;
    ggml_tensor *b_slice = ggml_view_2d(ctx, site.b, site.rank, width, site.b->nb[1], static_cast<size_t>(offset) * site.b->nb[1]);
    if (!b_slice) return nullptr;
    ggml_tensor *delta = ggml_mul_mat(ctx, b_slice, ax);
    return delta ? ggml_scale(ctx, delta, site.scale) : nullptr;
}
