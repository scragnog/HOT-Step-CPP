#pragma once

// Reusable, per-block execution seam for the staged AI Toolkit YuE2 graph.
// The backend and frozen/LoRA tensors are caller-owned. Each call owns a
// temporary no-alloc graph context and backend buffer, and performs no update.

#include "yue2-aitk-graph.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

enum class Yue2AitkMaskMode { causal, noncausal };

struct Yue2AitkBlockForwardHost {
    std::vector<float> hidden;
    std::vector<float> key;
    std::vector<float> value;
};

struct Yue2AitkBlockBackwardHost {
    std::vector<float> dx;
    std::vector<float> qkv_dA, qkv_dB;
    std::vector<float> output_dA, output_dB;
    std::vector<float> gate_up_dA, gate_up_dB;
    std::vector<float> down_dA, down_dB;
};

namespace yue2_aitk_executor_detail {
struct Runtime {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Runtime() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

inline bool fail(std::string * error, const char * message) {
    if (error) *error = message;
    return false;
}
inline bool fail(std::string * error, const std::string & message) {
    if (error) *error = message;
    return false;
}
inline bool checked_count(int64_t a, int64_t b, size_t * out) {
    if (a <= 0 || b <= 0 || static_cast<uint64_t>(a) > (std::numeric_limits<size_t>::max)() / static_cast<uint64_t>(b)) return false;
    *out = static_cast<size_t>(a) * static_cast<size_t>(b); return true;
}
inline bool finite_all(const float * p, size_t n) {
    if (!p) return false;
    for (size_t i = 0; i < n; ++i) if (!std::isfinite(p[i])) return false;
    return true;
}
inline bool make_runtime(ggml_backend_t backend, size_t mem_size, Runtime * out, std::string * error) {
    if (!backend || !out) return fail(error, "backend/runtime output is null");
    ggml_init_params params{}; params.mem_size = mem_size; params.no_alloc = true;
    out->ctx = ggml_init(params); if (!out->ctx) return fail(error, "failed to create block graph context");
    return true;
}
inline ggml_tensor * make_mask(ggml_context * ctx, int64_t sequence, int64_t prefix,
                               Yue2AitkMaskMode mode, std::vector<uint16_t> * host,
                               std::string * error) {
    const int64_t keys = prefix + sequence;
    if (sequence <= 0 || prefix < 0 || keys <= 0) return nullptr;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, sequence);
    if (!mask) return nullptr;
    if (!host) return nullptr;
    host->resize(static_cast<size_t>(keys * sequence));
    for (int64_t row = 0; row < sequence; ++row) for (int64_t col = 0; col < keys; ++col) {
        const bool visible = mode == Yue2AitkMaskMode::noncausal || col <= prefix + row;
        (*host)[static_cast<size_t>(row * keys + col)] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
    }
    ggml_set_name(mask, "aitk_block_mask");
    (void) error;
    return mask;
}
inline bool supports_all(ggml_backend_t backend, ggml_cgraph * graph, std::string * error) {
    const int n = ggml_graph_n_nodes(graph); ggml_tensor ** nodes = ggml_graph_nodes(graph);
    for (int i = 0; i < n; ++i) if (!ggml_backend_supports_op(backend, nodes[i]))
        return fail(error, "backend does not support block graph node " + std::to_string(i));
    return true;
}
inline bool validate_inputs(ggml_backend_t backend, const Yue2AitkGraphConfig & c,
                            const float * x, const float * cosine, const float * sine,
                            int64_t sequence, std::string * error) {
    if (!backend || !x || !cosine || !sine) return fail(error, "null block input/backend");
    if (c.hidden <= 0 || c.head_dim <= 0 || c.heads <= 0 || c.kv_heads <= 0 || sequence <= 0 || c.hidden != c.heads*c.head_dim)
        return fail(error, "invalid block dimensions");
    size_t hs=0, ts=0; if (!checked_count(c.hidden, sequence, &hs) || !checked_count(c.head_dim/2, sequence, &ts)) return fail(error, "block size overflow");
    if (!finite_all(x, hs) || !finite_all(cosine, ts) || !finite_all(sine, ts)) return fail(error, "non-finite block input");
    return true;
}
inline bool allocate(Runtime * r, ggml_backend_t backend, std::string * error) {
    r->buffer = ggml_backend_alloc_ctx_tensors(r->ctx, backend);
    return r->buffer ? true : fail(error, "failed to allocate block graph tensors");
}
inline bool setup_inputs(ggml_backend_t backend, ggml_tensor * h, ggml_tensor * cosine,
                         ggml_tensor * sine, ggml_tensor * mask, const float * x,
                         const float * cos_data, const float * sin_data,
                         const std::vector<uint16_t> & mask_data) {
    ggml_backend_tensor_set(h, x, 0, ggml_nbytes(h));
    ggml_backend_tensor_set(cosine, cos_data, 0, ggml_nbytes(cosine));
    ggml_backend_tensor_set(sine, sin_data, 0, ggml_nbytes(sine));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));
    (void) backend; return true;
}
} // namespace yue2_aitk_executor_detail

class Yue2AitkBlockExecutor {
public:
    static bool forward(ggml_backend_t backend, const Yue2AitkGraphConfig & config,
                        const Yue2AitkLayerWeights & weights, const Yue2AitkBlockNorms & norms,
                        const Yue2AitkLayerAdapters * adapters, const float * x,
                        int64_t sequence, const float * cosine, const float * sine,
                        Yue2AitkMaskMode mask_mode, Yue2AitkBlockForwardHost * result,
                        ggml_tensor * prefix_k_canvas = nullptr, ggml_tensor * prefix_v_canvas = nullptr,
                        int64_t prefix_length = 0, std::string * error = nullptr) {
        using namespace yue2_aitk_executor_detail;
        if (!result) return fail(error, "forward result is null");
        if ((prefix_k_canvas == nullptr) != (prefix_v_canvas == nullptr)) return fail(error, "prefix canvases must be paired");
        if (!validate_inputs(backend, config, x, cosine, sine, sequence, error)) return false;
        Runtime r; if (!make_runtime(backend, 64ull*1024ull*1024ull, &r, error)) return false;
        ggml_tensor * h = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, config.hidden, sequence);
        ggml_tensor * cos_t = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, config.head_dim/2, 1, sequence, 1);
        ggml_tensor * sin_t = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, config.head_dim/2, 1, sequence, 1);
        std::vector<uint16_t> mask_data; ggml_tensor * mask = make_mask(r.ctx, sequence, prefix_length, mask_mode, &mask_data, error);
        if (!h || !cos_t || !sin_t || !mask) return fail(error, "forward tensor allocation failed");
        Yue2AitkBlockResult out = yue2_aitk_graph::block(r.ctx, config, weights, norms, adapters, h, cos_t, sin_t, mask, prefix_k_canvas, prefix_v_canvas, prefix_length);
        ggml_cgraph * graph = ggml_new_graph_custom(r.ctx, 32768, true); if (!graph) return fail(error, "forward graph allocation failed");
        ggml_tensor * key_cont = out.key ? ggml_cont(r.ctx, out.key) : nullptr;
        ggml_tensor * value_cont = out.value ? ggml_cont(r.ctx, out.value) : nullptr;
        ggml_build_forward_expand(graph, out.hidden); if (key_cont) ggml_build_forward_expand(graph, key_cont); if (value_cont) ggml_build_forward_expand(graph, value_cont);
        if (!supports_all(backend, graph, error)) return false;
        if (!allocate(&r, backend, error)) return false;
        setup_inputs(backend, h, cos_t, sin_t, mask, x, cosine, sine, mask_data);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return fail(error, "forward graph compute failed");
        result->hidden.resize(static_cast<size_t>(config.hidden*sequence)); ggml_backend_tensor_get(out.hidden, result->hidden.data(), 0, ggml_nbytes(out.hidden));
        result->key.clear(); result->value.clear();
        if (key_cont && value_cont) { result->key.resize(ggml_nelements(key_cont)); result->value.resize(ggml_nelements(value_cont)); ggml_backend_tensor_get(key_cont,result->key.data(),0,ggml_nbytes(key_cont)); ggml_backend_tensor_get(value_cont,result->value.data(),0,ggml_nbytes(value_cont)); }
        return true;
    }

    static bool backward(ggml_backend_t backend, const Yue2AitkGraphConfig & config,
                         const Yue2AitkLayerWeights & weights, const Yue2AitkBlockNorms & norms,
                         const Yue2AitkLayerAdapters * adapters, const float * x, const float * dy,
                         int64_t sequence, const float * cosine, const float * sine,
                         Yue2AitkMaskMode mask_mode, Yue2AitkBlockBackwardHost * result,
                         ggml_tensor * prefix_k_canvas = nullptr, ggml_tensor * prefix_v_canvas = nullptr,
                         int64_t prefix_length = 0, std::string * error = nullptr) {
        using namespace yue2_aitk_executor_detail;
        if (!result || !adapters) return fail(error, "backward result/adapters is null");
        size_t hs=0; if (!checked_count(config.hidden, sequence, &hs) || !finite_all(dy, hs)) return fail(error, "invalid/non-finite upstream gradient");
        if ((prefix_k_canvas == nullptr) != (prefix_v_canvas == nullptr)) return fail(error, "prefix canvases must be paired");
        if (!validate_inputs(backend, config, x, cosine, sine, sequence, error)) return false;
        Runtime r; if (!make_runtime(backend, 64ull*1024ull*1024ull, &r, error)) return false;
        ggml_tensor * h=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,config.hidden,sequence), * upstream=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,config.hidden,sequence);
        std::vector<uint16_t> mask_data; ggml_tensor * cos_t=ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,config.head_dim/2,1,sequence,1), * sin_t=ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,config.head_dim/2,1,sequence,1), * mask=make_mask(r.ctx,sequence,prefix_length,mask_mode,&mask_data,error);
        if(!h||!upstream||!cos_t||!sin_t||!mask) return fail(error,"backward tensor allocation failed");
        ggml_set_param(h);
        Yue2AitkBlockResult out=yue2_aitk_graph::block(r.ctx,config,weights,norms,adapters,h,cos_t,sin_t,mask,prefix_k_canvas,prefix_v_canvas,prefix_length);
        ggml_tensor * loss=ggml_sum(r.ctx,ggml_mul(r.ctx,out.hidden,upstream)); ggml_cgraph * graph=ggml_new_graph_custom(r.ctx,65536,true); if(!loss||!graph) return fail(error,"backward graph allocation failed");
        ggml_build_forward_expand(graph,loss); ggml_set_loss(loss); ggml_build_backward_expand(r.ctx,graph,nullptr); if(!supports_all(backend,graph,error)) return false;
        if(!allocate(&r,backend,error)) return false; setup_inputs(backend,h,cos_t,sin_t,mask,x,cosine,sine,mask_data); ggml_backend_tensor_set(upstream,dy,0,ggml_nbytes(upstream)); ggml_graph_reset(graph);
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return fail(error,"backward graph compute failed");
        auto get=[&](ggml_tensor*t,std::vector<float>&v,const char*n)->bool{ggml_tensor*g=ggml_graph_get_grad(graph,t);if(!g)return fail(error,std::string("missing gradient: ")+n);v.resize(ggml_nbytes(g)/sizeof(float));ggml_backend_tensor_get(g,v.data(),0,ggml_nbytes(g));return true;};
        if(!get(h,result->dx,"x")||!get(adapters->qkv.a,result->qkv_dA,"qkv A")||!get(adapters->qkv.b,result->qkv_dB,"qkv B")||!get(adapters->output.a,result->output_dA,"output A")||!get(adapters->output.b,result->output_dB,"output B")||!get(adapters->gate_up.a,result->gate_up_dA,"gate_up A")||!get(adapters->gate_up.b,result->gate_up_dB,"gate_up B")||!get(adapters->down.a,result->down_dA,"down A")||!get(adapters->down.b,result->down_dB,"down B")) return false;
        return true;
    }
};
