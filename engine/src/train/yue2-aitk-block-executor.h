#pragma once

// Reusable, per-block execution seam for the staged AI Toolkit YuE2 graph.
// The backend and frozen/LoRA tensors are caller-owned. Each call owns a
// temporary no-alloc graph context and backend buffer, and performs no update.

#include "yue2-aitk-graph.h"
#include "yue2-aitk-host-profile.h"
#include "yue2-aitk-backend.h"
#include "yue2-aitk-bf16-storage.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "ggml-alloc.h"

enum class Yue2AitkMaskMode { causal, noncausal };

struct Yue2AitkBlockForwardHost {
    std::vector<float> hidden;
    std::vector<float> key;
    std::vector<float> value;
    bool attention_saved = false; // packed O+LSE was copied into device_attn
};

struct Yue2AitkBlockBackwardHost {
    std::vector<float> dx;
    // One gradient per trainable factor, in yue2_aitk_layer_params order.
    std::vector<std::vector<float>> params;
};

namespace yue2_aitk_executor_detail {
struct TensorFlagsGuard {
    ggml_tensor * tensor;
    int32_t flags;
    explicit TensorFlagsGuard(ggml_tensor * t) : tensor(t), flags(t ? t->flags : 0) {}
    ~TensorFlagsGuard() { if (tensor) tensor->flags = flags; }
};
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
// Same bytes? `data` alone is not an identity: Vulkan hands out offsets from a
// fixed fake base per backend buffer, and a large context is split across
// several buffers, so distinct tensors can share a `data` value.
inline bool same_memory(const ggml_tensor * a, const ggml_tensor * b) {
    return a->buffer == b->buffer && a->data == b->data;
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
// make_mask's rule as the attention op's causal hint: key j is visible iff
// j <= hint + row (noncausal: every key, so any hint >= keys - 1).
inline int64_t causal_hint(Yue2AitkMaskMode mode, int64_t prefix, int64_t sequence) {
    return mode == Yue2AitkMaskMode::noncausal ? prefix + sequence : prefix;
}
inline bool supports_all(ggml_backend_t backend, ggml_cgraph * graph, std::string * error) {
    const int n = ggml_graph_n_nodes(graph); ggml_tensor ** nodes = ggml_graph_nodes(graph);
    std::string what;
    for (int i = 0; i < n; ++i) if (!ggml_backend_supports_op(backend, nodes[i])) {
        std::string one = std::string(ggml_op_desc(nodes[i])) + " -> " + ggml_type_name(nodes[i]->type);
        for (int s = 0; s < GGML_MAX_SRC && nodes[i]->src[s]; ++s)
            one += std::string(s ? ", " : " from ") + ggml_type_name(nodes[i]->src[s]->type) + (ggml_is_contiguous(nodes[i]->src[s]) ? "" : " (non-contiguous)");
        if (what.find(one) == std::string::npos) what += (what.empty() ? "" : "; ") + one;
    }
    return what.empty() ? true : fail(error, "backend does not support block graph node(s): " + what);
}
inline bool validate_inputs(ggml_backend_t backend, const Yue2AitkGraphConfig & c,
                            const float * x, const float * cosine, const float * sine,
                            int64_t sequence, std::string * error, bool device_x = false,
                            bool checked_constants = false) {
    if (!backend || (!x && !device_x) || !cosine || !sine) return fail(error, "null block input/backend");
    if (c.hidden <= 0 || c.head_dim <= 0 || c.heads <= 0 || c.kv_heads <= 0 || sequence <= 0 || c.hidden != c.heads*c.head_dim)
        return fail(error, "invalid block dimensions");
    size_t hs=0, ts=0; if (!checked_count(c.hidden, sequence, &hs) || !checked_count(c.head_dim/2, sequence, &ts)) return fail(error, "block size overflow");
    // Device-resident stack inputs are validated by the producer before upload;
    // downloading them here would defeat the fast path. Constants remain host
    // inputs unless the shared constants object owns their device copies.
    // Shared block constants were checked once when they were built.
    if ((!device_x && !finite_all(x, hs)) ||
        (!checked_constants && (!finite_all(cosine, ts) || !finite_all(sine, ts)))) return fail(error, "non-finite block input");
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

// One allocation plan and one set of constants per stack, shared by its layers.
struct Yue2AitkBlockWorkspace {
    ggml_gallocr_t allocator = nullptr;
    ~Yue2AitkBlockWorkspace() { if (allocator) ggml_gallocr_free(allocator); }
    bool allocate(ggml_backend_t backend, ggml_cgraph * graph, std::string * error) {
        if (!allocator) allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return (allocator && ggml_gallocr_alloc_graph(allocator, graph)) ||
            yue2_aitk_executor_detail::fail(error, "block workspace allocation failed");
    }
};
struct Yue2AitkBlockConstants {
    yue2_aitk_executor_detail::Runtime runtime;
    ggml_tensor * cosine = nullptr, * sine = nullptr, * mask = nullptr;
    // Absolute RoPE positions prefix..prefix+count-1, for ggml_rope_ext.
    ggml_tensor * positions = nullptr;
    // The backward graphs' loss seed: dL/dL = 1. Their loss gradient
    // accumulator points here instead of being set by ggml_graph_reset, a
    // blocking 4-byte upload per block (YUE2_AITK_BLOCK_RESET=1 restores it).
    ggml_tensor * one = nullptr;
    bool make(ggml_backend_t backend, const Yue2AitkGraphConfig & c,
              int64_t count, int64_t prefix, Yue2AitkMaskMode mode,
              const float * cos_data, const float * sin_data, std::string * error) {
        using namespace yue2_aitk_executor_detail;
        YUE2_PROF("constants: mask build + upload");
        if (!make_runtime(backend, 6*ggml_tensor_overhead()+1024, &runtime, error)) return false;
        cosine = ggml_new_tensor_4d(runtime.ctx, GGML_TYPE_F32, c.head_dim/2, 1, count, 1);
        sine = ggml_new_tensor_4d(runtime.ctx, GGML_TYPE_F32, c.head_dim/2, 1, count, 1);
        std::vector<uint16_t> host;
        mask = make_mask(runtime.ctx, count, prefix, mode, &host, error);
        one = ggml_new_tensor_1d(runtime.ctx, GGML_TYPE_F32, 1);
        positions = ggml_new_tensor_1d(runtime.ctx, GGML_TYPE_I32, count);
        if (!mask || !cosine || !sine || !one || !positions || !allocate(&runtime, backend, error)) return false;
        std::vector<int32_t> pos(static_cast<size_t>(count));
        for (int64_t t = 0; t < count; ++t) pos[size_t(t)] = int32_t(prefix + t);
        ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
        const float onef = 1.0f;
        ggml_backend_tensor_set(one, &onef, 0, sizeof(onef));
        ggml_backend_tensor_set(cosine, cos_data, 0, ggml_nbytes(cosine));
        ggml_backend_tensor_set(sine, sin_data, 0, ggml_nbytes(sine));
        ggml_backend_tensor_set(mask, host.data(), 0, ggml_nbytes(mask));
        return true;
    }
};

class Yue2AitkBlockExecutor {
public:
    static bool forward(ggml_backend_t backend, const Yue2AitkGraphConfig & config,
                        const Yue2AitkLayerWeights & weights, const Yue2AitkBlockNorms & norms,
                        const Yue2AitkLayerAdapters * adapters, const float * x,
                        int64_t sequence, const float * cosine, const float * sine,
                        Yue2AitkMaskMode mask_mode, Yue2AitkBlockForwardHost * result,
                        ggml_tensor * prefix_k_canvas = nullptr, ggml_tensor * prefix_v_canvas = nullptr,
                        int64_t prefix_length = 0, std::string * error = nullptr,
                        bool capture_kv = true, Yue2AitkBlockConstants * constants = nullptr,
                        Yue2AitkBlockWorkspace * workspace = nullptr,
                        ggml_tensor * device_input = nullptr, ggml_tensor * device_output = nullptr,
                        ggml_tensor * capture_k = nullptr, ggml_tensor * capture_v = nullptr,
                        bool async = false, ggml_tensor * device_attn = nullptr) {
        using namespace yue2_aitk_executor_detail;
        if (!result) return fail(error, "forward result is null");
        result->attention_saved = false;
        if ((capture_k == nullptr) != (capture_v == nullptr)) return fail(error, "device K/V capture canvases must be paired");
        if ((prefix_k_canvas == nullptr) != (prefix_v_canvas == nullptr)) return fail(error, "prefix canvases must be paired");
        yue2_aitk_prof::Lap lap;
        if (!validate_inputs(backend, config, x, cosine, sine, sequence, error, device_input != nullptr, constants != nullptr)) return false;
        lap.mark("fwd: validate inputs (host finite checks)");
        Runtime r; if (!make_runtime(backend, 64ull*1024ull*1024ull, &r, error)) return false;
        lap.mark("fwd: ggml_init 64 MiB context");
        auto valid_device = [&](ggml_tensor * t) {
            return !t || (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
                          t->ne[0] == config.hidden && t->ne[1] == sequence &&
                          t->ne[2] == 1 && t->ne[3] == 1 && t->data);
        };
        if (!valid_device(device_input) || !valid_device(device_output)) return fail(error, "invalid device block buffer shape/type");
        if (device_input && (!constants || (device_output && same_memory(device_input, device_output))))
            return fail(error, "device input needs shared constants and a distinct output");
        ggml_tensor * h = device_input ? device_input : ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, config.hidden, sequence);
        ggml_tensor * cos_t = constants ? constants->cosine : ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, config.head_dim/2, 1, sequence, 1);
        ggml_tensor * sin_t = constants ? constants->sine : ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, config.head_dim/2, 1, sequence, 1);
        std::vector<uint16_t> mask_data; ggml_tensor * mask = constants ? constants->mask : make_mask(r.ctx, sequence, prefix_length, mask_mode, &mask_data, error);
        if (!h || !cos_t || !sin_t || !mask) return fail(error, "forward tensor allocation failed");
        Yue2AitkBlockResult out = yue2_aitk_graph::block(r.ctx, config, weights, norms, adapters, h, cos_t, sin_t, mask, prefix_k_canvas, prefix_v_canvas, prefix_length, causal_hint(mask_mode, prefix_length, sequence), constants ? constants->positions : nullptr);
        ggml_cgraph * graph = ggml_new_graph_custom(r.ctx, 32768, true); if (!graph) return fail(error, "forward graph allocation failed");
        ggml_tensor * key_cont = capture_kv && out.key ? ggml_cont(r.ctx, out.key) : nullptr;
        ggml_tensor * value_cont = capture_kv && out.value ? ggml_cont(r.ctx, out.value) : nullptr;
        ggml_tensor * hidden_copy = device_output ? ggml_cpy(r.ctx, out.hidden, device_output) : nullptr;
        // Keep the fused attention's packed O+LSE on the tape for the backward recompute.
        ggml_tensor * attn_copy = nullptr;
        if (device_attn && out.attention_packed && device_attn->type == GGML_TYPE_F32 && ggml_is_contiguous(device_attn) &&
            device_attn->data && ggml_nelements(device_attn) == ggml_nelements(out.attention_packed))
            attn_copy = ggml_cpy(r.ctx, out.attention_packed, device_attn);
        if (attn_copy) ggml_build_forward_expand(graph, attn_copy);
        ggml_build_forward_expand(graph, hidden_copy ? hidden_copy : out.hidden); if (key_cont) ggml_build_forward_expand(graph, key_cont); if (value_cont) ggml_build_forward_expand(graph, value_cont);
        if (capture_k) {
            // The leading `sequence` rows of each [D, rows, Nkv] canvas take
            // this block's post-RoPE K and its V; the caller zeroed the rest.
            if (!out.key || !out.value || capture_k->ne[0] != out.key->ne[0] || capture_k->ne[1] < sequence ||
                capture_k->ne[2] != out.key->ne[2] || !ggml_are_same_shape(capture_k, capture_v))
                return fail(error, "device K/V capture canvas has the wrong shape");
            auto rows = [&](ggml_tensor * canvas) {
                return ggml_view_3d(r.ctx, canvas, canvas->ne[0], sequence, canvas->ne[2], canvas->nb[1], canvas->nb[2], 0);
            };
            ggml_build_forward_expand(graph, ggml_cpy(r.ctx, out.key, rows(capture_k)));
            ggml_build_forward_expand(graph, ggml_cpy(r.ctx, out.value, rows(capture_v)));
        }
        if (yue2_aitk_bf16_storage::enabled() && yue2_aitk_is_cuda(backend) && !yue2_aitk_graph::strict_f32())
            yue2_aitk_bf16_storage::apply(graph, {hidden_copy ? nullptr : out.hidden, key_cont, value_cont});
        lap.mark("fwd: graph build");
        if (!supports_all(backend, graph, error)) return false;
        lap.mark("fwd: supports_op scan");
        if (workspace) {
            ggml_set_input(h); ggml_set_output(hidden_copy ? hidden_copy : out.hidden);
            if (key_cont) ggml_set_output(key_cont);
            if (value_cont) ggml_set_output(value_cont);
            if (attn_copy) ggml_set_output(attn_copy);
            if (!workspace->allocate(backend, graph, error)) return false;
        } else if (!allocate(&r, backend, error)) return false;
        lap.mark("fwd: gallocr/allocate");
        if (!device_input) {
            if (constants) ggml_backend_tensor_set(h, x, 0, ggml_nbytes(h));
            else setup_inputs(backend, h, cos_t, sin_t, mask, x, cosine, sine, mask_data);
        }
        lap.mark("fwd: input upload");
        // async: the caller keeps every input and output on the device and
        // synchronizes before it reads anything back; a block with a host
        // readback still waits here.
        const bool launch_async = async && device_output && !key_cont && !value_cont;
        if ((launch_async ? ggml_backend_graph_compute_async(backend, graph) : ggml_backend_graph_compute(backend, graph)) != GGML_STATUS_SUCCESS)
            return fail(error, "forward graph compute failed");
        lap.mark("fwd: graph_compute (launch)");
        result->attention_saved = attn_copy != nullptr;
        result->hidden.clear();
        if (!device_output) { result->hidden.resize(static_cast<size_t>(config.hidden*sequence)); ggml_backend_tensor_get(out.hidden, result->hidden.data(), 0, ggml_nbytes(out.hidden)); }
        result->key.clear(); result->value.clear();
        if (key_cont && value_cont) { result->key.resize(ggml_nelements(key_cont)); result->value.resize(ggml_nelements(value_cont)); ggml_backend_tensor_get(key_cont,result->key.data(),0,ggml_nbytes(key_cont)); ggml_backend_tensor_get(value_cont,result->value.data(),0,ggml_nbytes(value_cont)); }
        lap.mark("fwd: readback (waits for GPU)");
        return true;
    }

    static bool backward(ggml_backend_t backend, const Yue2AitkGraphConfig & config,
                         const Yue2AitkLayerWeights & weights, const Yue2AitkBlockNorms & norms,
                         const Yue2AitkLayerAdapters * adapters, const float * x, const float * dy,
                         int64_t sequence, const float * cosine, const float * sine,
                         Yue2AitkMaskMode mask_mode, Yue2AitkBlockBackwardHost * result,
                         ggml_tensor * prefix_k_canvas = nullptr, ggml_tensor * prefix_v_canvas = nullptr,
                         int64_t prefix_length = 0, std::string * error = nullptr,
                         Yue2AitkBlockConstants * constants = nullptr,
                         Yue2AitkBlockWorkspace * workspace = nullptr,
                         ggml_tensor * device_input = nullptr, ggml_tensor * device_upstream = nullptr,
                         ggml_tensor * device_dx = nullptr,
                         const std::vector<ggml_tensor *> * gradient_sinks = nullptr,
                         bool accumulate_sinks = false, bool async = false,
                         ggml_tensor * device_attn = nullptr) {
        using namespace yue2_aitk_executor_detail;
        yue2_aitk_prof::Lap lap;
        if (!result || !adapters) return fail(error, "backward result/adapters is null");
        size_t hs=0; if (!checked_count(config.hidden, sequence, &hs) || ((!device_upstream || dy) && !finite_all(dy, hs))) return fail(error, "invalid/non-finite upstream gradient");
        if ((prefix_k_canvas == nullptr) != (prefix_v_canvas == nullptr)) return fail(error, "prefix canvases must be paired");
        if (!validate_inputs(backend, config, x, cosine, sine, sequence, error, device_input != nullptr, constants != nullptr)) return false;
        lap.mark("bwd: validate inputs (host finite checks)");
        Runtime r; if (!make_runtime(backend, 64ull*1024ull*1024ull, &r, error)) return false;
        lap.mark("bwd: ggml_init 64 MiB context");
        auto valid_device = [&](ggml_tensor * t) {
            return !t || (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
                          t->ne[0] == config.hidden && t->ne[1] == sequence &&
                          t->ne[2] == 1 && t->ne[3] == 1 && t->data);
        };
        if (!valid_device(device_input) || !valid_device(device_upstream) || !valid_device(device_dx)) return fail(error, "invalid device backward buffer shape/type");
        if (device_input && !constants) return fail(error, "device input needs shared constants");
        if (device_dx && ((device_input && same_memory(device_dx, device_input)) ||
                          (device_upstream && same_memory(device_dx, device_upstream))))
            return fail(error, "device input gradient needs a distinct output");
        ggml_tensor * h=device_input ? device_input : ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,config.hidden,sequence), * upstream=device_upstream ? device_upstream : ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,config.hidden,sequence);
        std::vector<uint16_t> mask_data;
        ggml_tensor * cos_t=constants ? constants->cosine : ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,config.head_dim/2,1,sequence,1);
        ggml_tensor * sin_t=constants ? constants->sine : ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,config.head_dim/2,1,sequence,1);
        ggml_tensor * mask=constants ? constants->mask : make_mask(r.ctx,sequence,prefix_length,mask_mode,&mask_data,error);
        if(!h||!upstream||!cos_t||!sin_t||!mask) return fail(error,"backward tensor allocation failed");
        TensorFlagsGuard input_flags(device_input);
        // Saved packed O+LSE from the forward (a plain device leaf, like the other device tensors).
        if (device_attn && !(device_attn->type == GGML_TYPE_F32 && ggml_is_contiguous(device_attn) && device_attn->data)) device_attn = nullptr;
        TensorFlagsGuard attn_flags(device_attn);
        ggml_set_param(h);
        Yue2AitkBlockResult out=yue2_aitk_graph::block(r.ctx,config,weights,norms,adapters,h,cos_t,sin_t,mask,prefix_k_canvas,prefix_v_canvas,prefix_length,causal_hint(mask_mode,prefix_length,sequence),constants ? constants->positions : nullptr,device_attn);
        ggml_tensor * loss=ggml_sum(r.ctx,ggml_mul(r.ctx,out.hidden,upstream)); ggml_cgraph * graph=ggml_new_graph_custom(r.ctx,65536,true); if(!loss||!graph) return fail(error,"backward graph allocation failed");
        ggml_build_forward_expand(graph,loss); ggml_set_loss(loss);
        lap.mark("bwd: graph build");
        ggml_build_backward_expand(r.ctx,graph,nullptr);
        lap.mark("bwd: ggml_build_backward_expand");
        ggml_tensor * dx_copy = nullptr;
        static const bool block_reset = std::getenv("YUE2_AITK_BLOCK_RESET") != nullptr;
        const bool seed_from_constant = workspace && constants && constants->one && !block_reset;
        if (seed_from_constant) {
            ggml_tensor * seed = ggml_graph_get_grad_acc(graph, loss);
            if (!seed || !ggml_is_scalar(seed) || seed->type != GGML_TYPE_F32) return fail(error, "unexpected loss gradient accumulator");
            // Read only: the loss feeds no node, so nothing accumulates into it.
            if (ggml_backend_tensor_alloc(constants->one->buffer, seed, constants->one->data) != GGML_STATUS_SUCCESS)
                return fail(error, "loss seed binding failed");
        }
        if (device_dx) { ggml_tensor * gh = ggml_graph_get_grad(graph, h); if (!gh) return fail(error, "missing block input gradient"); dx_copy = ggml_cpy(r.ctx, gh, device_dx); ggml_set_output(dx_copy); ggml_build_forward_expand(graph, dx_copy); }
        const std::vector<ggml_tensor *> layer_params = yue2_aitk_layer_params(*adapters);
        // Device gradient slots (S2b): the graph writes each parameter gradient
        // into its slot, or adds it for --grad-accum, and nothing is read back.
        // The finite check moves to the clip, which reads every slot.
        if (gradient_sinks) {
            if (gradient_sinks->size() != layer_params.size()) return fail(error, "gradient slot count mismatch");
            for (size_t i = 0; i < layer_params.size(); ++i) {
                ggml_tensor * g = ggml_graph_get_grad(graph, layer_params[i]), * sink = (*gradient_sinks)[i];
                if (!g || !sink || !ggml_are_same_shape(g, sink) || g->type != GGML_TYPE_F32 || sink->type != GGML_TYPE_F32)
                    return fail(error, "invalid gradient slot");
                ggml_tensor * store = ggml_cpy(r.ctx, accumulate_sinks ? ggml_add(r.ctx, sink, g) : g, sink);
                ggml_set_output(store); ggml_build_forward_expand(graph, store);
            }
        }
        if (yue2_aitk_bf16_storage::enabled() && yue2_aitk_is_cuda(backend) && !yue2_aitk_graph::strict_f32())
            yue2_aitk_bf16_storage::apply(graph, {device_dx ? nullptr : ggml_graph_get_grad(graph, h)});
        if(!supports_all(backend,graph,error)) return false;
        lap.mark("bwd: supports_op scan");
        if (workspace) {
            ggml_set_input(h); ggml_set_input(upstream);
            std::vector<ggml_tensor *> outputs = layer_params; outputs.insert(outputs.begin(), h);
            for (ggml_tensor * p : outputs) {
                ggml_tensor * g = ggml_graph_get_grad(graph, p);
                if (!g) return fail(error, "missing block gradient");
                ggml_set_output(g);
            }
            if (!workspace->allocate(backend, graph, error)) return false;
        } else if(!allocate(&r,backend,error)) return false;
        lap.mark("bwd: gallocr/allocate");
        if (!device_input) { if (constants) ggml_backend_tensor_set(h,x,0,ggml_nbytes(h)); else setup_inputs(backend,h,cos_t,sin_t,mask,x,cosine,sine,mask_data); }
        if (!device_upstream) ggml_backend_tensor_set(upstream,dy,0,ggml_nbytes(upstream));
        lap.mark("bwd: input upload");
        const bool launch_async = async && device_dx && gradient_sinks;
        if (!seed_from_constant) {
            // ggml_graph_reset uploads on another stream: let in-flight blocks finish first.
            if (launch_async) ggml_backend_synchronize(backend);
            ggml_graph_reset(graph);
        }
        lap.mark("bwd: ggml_graph_reset");
        if((launch_async ? ggml_backend_graph_compute_async(backend,graph) : ggml_backend_graph_compute(backend,graph))!=GGML_STATUS_SUCCESS) return fail(error,"backward graph compute failed");
        lap.mark("bwd: graph_compute (launch)");
        auto get=[&](ggml_tensor*t,std::vector<float>&v,const char*n)->bool{ggml_tensor*g=ggml_graph_get_grad(graph,t);if(!g)return fail(error,std::string("missing gradient: ")+n);v.resize(ggml_nbytes(g)/sizeof(float));ggml_backend_tensor_get(g,v.data(),0,ggml_nbytes(g));return true;};
        if(!device_dx && !get(h,result->dx,"x")) return false;
        result->params.resize(gradient_sinks ? 0 : layer_params.size());
        if (!gradient_sinks) for(size_t i=0;i<layer_params.size();++i) if(!get(layer_params[i],result->params[i],ggml_get_name(layer_params[i]))) return false;
        lap.mark("bwd: gradient readback (waits for GPU)");
        return true;
    }
};
