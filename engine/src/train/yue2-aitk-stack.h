#pragma once
#include "yue2-aitk-block-executor.h"
#include "st-write.h"
#include <cstring>
#include <functional>
#include <cstdlib>

// Recompute one block at a time. Keep layer checkpoints on the backend by
// default; the diagnostic host tape packs the same rounded values as BF16.
struct Yue2AitkPrefixHost {
    int64_t length = 0;
    std::vector<std::vector<float>> keys, values; // [D,length,Nkv] per layer
};
struct Yue2AitkStackTape {
    Yue2AitkStackTape() = default;
    Yue2AitkStackTape(const Yue2AitkStackTape &) = delete;
    Yue2AitkStackTape & operator=(const Yue2AitkStackTape &) = delete;
    int64_t length = 0;
    bool nar = false;
    std::vector<std::vector<uint16_t>> layer_inputs;
    std::vector<float> final_hidden; // before final expert norm
    // Fast path storage. Tensor objects and their backend buffer live with the
    // tape, so a backward pass can walk all layer inputs without BF16 host
    // widen/upload cycles. The baseline path never initializes these fields.
    ggml_context * device_ctx = nullptr;
    ggml_backend_buffer_t device_buffer = nullptr;
    std::vector<ggml_tensor *> device_layers;
    ggml_tensor * device_grad[2] = {nullptr, nullptr};
    bool device_tape = false;
    bool device_saved = false;
    void clear_device() {
        if (device_buffer) ggml_backend_buffer_free(device_buffer);
        if (device_ctx) ggml_free(device_ctx);
        device_buffer = nullptr; device_ctx = nullptr; device_layers.clear();
        device_grad[0] = device_grad[1] = nullptr; device_tape = false; device_saved = false;
    }
    ~Yue2AitkStackTape() {
        clear_device();
    }
};

namespace yue2_aitk_stack {
inline bool fast_execution() { return std::getenv("YUE2_AITK_BASELINE") == nullptr; }
inline std::vector<float> widen(const std::vector<uint16_t> & source) {
    std::vector<float> result(source.size());
    for (size_t i=0; i<source.size(); ++i) {
        const uint32_t bits=uint32_t(source[i])<<16;
        std::memcpy(&result[i], &bits, sizeof(float));
    }
    return result;
}
inline std::vector<uint16_t> pack(const std::vector<float> & source) {
    std::vector<uint16_t> result(source.size());
    for (size_t i=0; i<source.size(); ++i) result[i]=stw_f32_to_bf16(source[i]);
    return result;
}
inline void trig(const Yue2AitkGraphConfig & c, int64_t count, int64_t offset,
                  std::vector<float> & cosine, std::vector<float> & sine) {
    const int64_t half=c.head_dim/2;
    cosine.resize(size_t(count*half)); sine.resize(cosine.size());
    for (int64_t t=0;t<count;++t) for (int64_t i=0;i<half;++i) {
        const float frequency=1.0f/std::pow(1000000.0f,float(2*i)/float(c.head_dim));
        const float angle=float(t+offset)*frequency;
        const uint32_t cb=uint32_t(stw_f32_to_bf16(std::cos(angle)))<<16;
        const uint32_t sb=uint32_t(stw_f32_to_bf16(std::sin(angle)))<<16;
        std::memcpy(&cosine[size_t(t*half+i)],&cb,4);
        std::memcpy(&sine[size_t(t*half+i)],&sb,4);
    }
}
inline Yue2AitkBlockNorms norms(const Yue2AitkModel & model, bool nar, size_t layer) {
    const std::string prefix=std::string(nar ? "model.diffusion_model.model.layers." : "text_encoders.model.layers.")+
                              std::to_string(layer)+".";
    return {model.ordinary((prefix+"input_layernorm.weight").c_str()),
            model.ordinary((prefix+"post_attention_layernorm.weight").c_str()),
            model.ordinary((prefix+"self_attn.q_norm.weight").c_str()),
            model.ordinary((prefix+"self_attn.k_norm.weight").c_str())};
}
inline bool valid_norms(const Yue2AitkBlockNorms & n) { return n.input && n.post_attention && n.q && n.k; }
inline bool make_device_tape(ggml_backend_t backend, const Yue2AitkGraphConfig & c,
                             size_t layers, int64_t length, bool save,
                             Yue2AitkStackTape * tape, std::string * error) {
    using yue2_aitk_executor_detail::fail;
    if (!backend || !tape || length <= 0 || c.hidden <= 0) return fail(error, "invalid device tape request");
    const size_t count = save ? layers + 1 : 2;
    const size_t overhead = (count + 2) * ggml_tensor_overhead() + 4096;
    ggml_init_params p{}; p.mem_size = overhead; p.no_alloc = true;
    tape->device_ctx = ggml_init(p);
    if (!tape->device_ctx) return fail(error, "device tape context allocation failed");
    tape->device_layers.resize(count);
    for (size_t i = 0; i < count; ++i) {
        tape->device_layers[i] = ggml_new_tensor_2d(tape->device_ctx, GGML_TYPE_F32, c.hidden, length);
        if (!tape->device_layers[i]) return fail(error, "device tape tensor allocation failed");
    }
    tape->device_grad[0] = ggml_new_tensor_2d(tape->device_ctx, GGML_TYPE_F32, c.hidden, length);
    tape->device_grad[1] = ggml_new_tensor_2d(tape->device_ctx, GGML_TYPE_F32, c.hidden, length);
    if (!tape->device_grad[0] || !tape->device_grad[1]) return fail(error, "device gradient tensor allocation failed");
    tape->device_buffer = ggml_backend_alloc_ctx_tensors(tape->device_ctx, backend);
    if (!tape->device_buffer) return fail(error, "device tape backend allocation failed");
    tape->device_tape = true;
    tape->device_saved = save;
    return true;
}
struct PrefixCanvas {
    yue2_aitk_executor_detail::Runtime runtime;
    ggml_tensor * k=nullptr, * v=nullptr;
    bool make(ggml_backend_t backend, const Yue2AitkGraphConfig & c,
              const Yue2AitkPrefixHost * prefix, size_t layer, int64_t count, std::string * error) {
        if (!prefix) return true;
        using namespace yue2_aitk_executor_detail;
        if (prefix->length<=0 || prefix->length+count>24576 || layer>=prefix->keys.size() || layer>=prefix->values.size())
            return fail(error,"invalid detached AR prefix dimensions");
        const size_t expected=size_t(c.head_dim*c.kv_heads*prefix->length);
        if (prefix->keys[layer].size()!=expected || prefix->values[layer].size()!=expected)
            return fail(error,"AR prefix payload has wrong size");
        if (!make_runtime(backend, 4*ggml_tensor_overhead()+1024,&runtime,error)) return false;
        k=ggml_new_tensor_4d(runtime.ctx,GGML_TYPE_F32,c.head_dim,prefix->length+count,c.kv_heads,1);
        v=ggml_new_tensor_4d(runtime.ctx,GGML_TYPE_F32,c.head_dim,prefix->length+count,c.kv_heads,1);
        if (!allocate(&runtime,backend,error)) return false;
        ggml_backend_buffer_clear(runtime.buffer,0);
        for (int64_t head=0;head<c.kv_heads;++head) {
            const size_t begin=size_t(head*c.head_dim*prefix->length);
            const size_t bytes=size_t(c.head_dim*prefix->length)*sizeof(float);
            ggml_backend_tensor_set(k,prefix->keys[layer].data()+begin,size_t(head)*k->nb[2],bytes);
            ggml_backend_tensor_set(v,prefix->values[layer].data()+begin,size_t(head)*v->nb[2],bytes);
        }
        return true;
    }
};

inline bool forward(ggml_backend_t backend, const Yue2AitkModel & model,
                     const Yue2AitkExpertAdapters * adapters, bool nar,
                     const std::vector<float> & initial, int64_t length,
                     const Yue2AitkPrefixHost * prefix, bool save_tape,
                     Yue2AitkStackTape * tape, Yue2AitkPrefixHost * capture_prefix,
                     std::string * error=nullptr) {
    using namespace yue2_aitk_executor_detail;
    const auto & expert=nar?model.nar():model.ar();
    Yue2AitkGraphConfig c;
    if (!tape || length<=0 || length>24576 || initial.size()!=size_t(length*c.hidden) ||
        (adapters && adapters->layers.size()!=expert.layers.size()) || (nar && !prefix))
        return fail(error,"invalid expert forward request");
    tape->clear_device();
    tape->length=length; tape->nar=nar; tape->layer_inputs.clear(); tape->final_hidden.clear();
    if (capture_prefix) { capture_prefix->length=length; capture_prefix->keys.clear(); capture_prefix->values.clear(); }
    std::vector<float> cosine,sine;
    trig(c,length,prefix?prefix->length:0,cosine,sine);
    Yue2AitkBlockConstants constants;
    Yue2AitkBlockWorkspace workspace;
    const bool fast = fast_execution();
    const bool device = fast && backend != nullptr && std::getenv("YUE2_AITK_HOST_TAPE") == nullptr;
    if (device && !make_device_tape(backend, c, expert.layers.size(), length, save_tape, tape, error)) return false;
    if (fast && !constants.make(backend,c,length,prefix?prefix->length:0,
            nar?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal,cosine.data(),sine.data(),error)) return false;
    std::vector<float> hidden=initial;
    if (device) ggml_backend_tensor_set(tape->device_layers[0], initial.data(), 0, ggml_nbytes(tape->device_layers[0]));
    for (size_t layer=0;layer<expert.layers.size();++layer) {
        if (!device && save_tape) tape->layer_inputs.push_back(pack(hidden));
        PrefixCanvas canvas;
        if (!canvas.make(backend,c,prefix,layer,length,error)) return false;
        const Yue2AitkBlockNorms layer_norms = norms(model,nar,layer);
        if (!valid_norms(layer_norms)) return fail(error,"expert layer norm weights are missing");
        Yue2AitkBlockForwardHost out;
        ggml_tensor * device_in = device ? tape->device_layers[save_tape ? layer : (layer & 1)] : nullptr;
        ggml_tensor * device_out = device ? tape->device_layers[save_tape ? layer + 1 : ((layer + 1) & 1)] : nullptr;
        if (!Yue2AitkBlockExecutor::forward(backend,c,expert.layers[layer],layer_norms,
                adapters?&adapters->layers[layer]:nullptr,device ? nullptr : hidden.data(),length,cosine.data(),sine.data(),
                nar?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal,&out,canvas.k,canvas.v,
                prefix?prefix->length:0,error,!fast || capture_prefix != nullptr,
                fast?&constants:nullptr,fast?&workspace:nullptr,device_in,device_out)) return false;
        if (!device) hidden=std::move(out.hidden);
        if (capture_prefix) { capture_prefix->keys.push_back(std::move(out.key)); capture_prefix->values.push_back(std::move(out.value)); }
    }
    if (device) {
        const ggml_tensor * final_tensor = tape->device_layers[save_tape ? expert.layers.size() : (expert.layers.size() & 1)];
        tape->final_hidden.resize(static_cast<size_t>(c.hidden * length));
        ggml_backend_tensor_get(final_tensor, tape->final_hidden.data(), 0, ggml_nbytes(final_tensor));
        if (!save_tape) tape->clear_device();
    } else tape->final_hidden=std::move(hidden);
    return true;
}

using StoreGradients=std::function<bool(size_t,const Yue2AitkBlockBackwardHost &,std::string *)>;
inline bool backward(ggml_backend_t backend,const Yue2AitkModel & model,
                      const Yue2AitkExpertAdapters & adapters,const Yue2AitkStackTape & tape,
                      const Yue2AitkPrefixHost * prefix,std::vector<float> gradient,
                      const StoreGradients & store,std::string * error=nullptr) {
    using namespace yue2_aitk_executor_detail;
    const auto & expert=tape.nar?model.nar():model.ar();
    Yue2AitkGraphConfig c;
    const bool fast = fast_execution();
    const bool device = fast && tape.device_tape;
    if (!store || ((!device && tape.layer_inputs.size()!=expert.layers.size()) ||
        (device && (!tape.device_saved || tape.device_layers.size()!=expert.layers.size()+1))) ||
        adapters.layers.size()!=expert.layers.size() ||
        gradient.size()!=size_t(tape.length*c.hidden)) return fail(error,"invalid expert backward tape");
    std::vector<float> cosine,sine;
    trig(c,tape.length,prefix?prefix->length:0,cosine,sine);
    Yue2AitkBlockConstants constants;
    Yue2AitkBlockWorkspace workspace;
    if (fast && !constants.make(backend,c,tape.length,prefix?prefix->length:0,
            tape.nar?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal,cosine.data(),sine.data(),error)) return false;
    if (device) {
        ggml_backend_tensor_set(tape.device_grad[0], gradient.data(), 0, ggml_nbytes(tape.device_grad[0]));
        int ping = 0;
        for (size_t end=expert.layers.size(); end>0; --end) {
            const size_t layer = end - 1;
            PrefixCanvas canvas;
            if (!canvas.make(backend,c,prefix,layer,tape.length,error)) return false;
            const Yue2AitkBlockNorms layer_norms = norms(model,tape.nar,layer);
            if (!valid_norms(layer_norms)) return fail(error,"expert layer norm weights are missing");
            Yue2AitkBlockBackwardHost out;
            const size_t input_index = layer;
            const int next = ping ^ 1;
            if (!Yue2AitkBlockExecutor::backward(backend,c,expert.layers[layer],layer_norms,
                    &adapters.layers[layer],nullptr,nullptr,tape.length,cosine.data(),sine.data(),
                    tape.nar?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal,&out,canvas.k,canvas.v,
                    prefix?prefix->length:0,error,fast?&constants:nullptr,fast?&workspace:nullptr,
                    tape.device_layers[input_index],tape.device_grad[ping],tape.device_grad[next])) return false;
            if (!store(layer,out,error)) return false;
            ping = next;
        }
        return true;
    }
    for (size_t end=expert.layers.size();end>0;--end) {
        const size_t layer=end-1;
        auto input=widen(tape.layer_inputs[layer]);
        PrefixCanvas canvas;
        if (!canvas.make(backend,c,prefix,layer,tape.length,error)) return false;
        const Yue2AitkBlockNorms layer_norms = norms(model,tape.nar,layer);
        if (!valid_norms(layer_norms)) return fail(error,"expert layer norm weights are missing");
        Yue2AitkBlockBackwardHost out;
        if (!Yue2AitkBlockExecutor::backward(backend,c,expert.layers[layer],layer_norms,
                &adapters.layers[layer],input.data(),gradient.data(),tape.length,cosine.data(),sine.data(),
                tape.nar?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal,&out,canvas.k,canvas.v,
                prefix?prefix->length:0,error,fast?&constants:nullptr,fast?&workspace:nullptr)) return false;
        if (!store(layer,out,error)) return false;
        gradient=std::move(out.dx);
    }
    return true;
}
} // namespace yue2_aitk_stack
