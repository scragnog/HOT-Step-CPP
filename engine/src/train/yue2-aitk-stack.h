#pragma once
#include "yue2-aitk-block-executor.h"
#include "st-write.h"
#include <cstring>
#include <functional>
#include <cstdlib>

// Layer-by-layer recomputation keeps one block graph resident. Checkpoints are
// host BF16 values; this is an initial bounded-memory execution path, not yet
// a performance-qualified replacement for device checkpoint scheduling.
struct Yue2AitkPrefixHost {
    int64_t length = 0;
    std::vector<std::vector<float>> keys, values; // [D,length,Nkv] per layer
};
struct Yue2AitkStackTape {
    int64_t length = 0;
    bool nar = false;
    std::vector<std::vector<uint16_t>> layer_inputs;
    std::vector<float> final_hidden; // before final expert norm
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
    tape->length=length; tape->nar=nar; tape->layer_inputs.clear(); tape->final_hidden.clear();
    if (capture_prefix) { capture_prefix->length=length; capture_prefix->keys.clear(); capture_prefix->values.clear(); }
    std::vector<float> cosine,sine;
    trig(c,length,prefix?prefix->length:0,cosine,sine);
    Yue2AitkBlockConstants constants;
    Yue2AitkBlockWorkspace workspace;
    const bool fast = fast_execution();
    if (fast && !constants.make(backend,c,length,prefix?prefix->length:0,
            nar?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal,cosine.data(),sine.data(),error)) return false;
    std::vector<float> hidden=initial;
    for (size_t layer=0;layer<expert.layers.size();++layer) {
        if (save_tape) tape->layer_inputs.push_back(pack(hidden));
        PrefixCanvas canvas;
        if (!canvas.make(backend,c,prefix,layer,length,error)) return false;
        const Yue2AitkBlockNorms layer_norms = norms(model,nar,layer);
        if (!valid_norms(layer_norms)) return fail(error,"expert layer norm weights are missing");
        Yue2AitkBlockForwardHost out;
        if (!Yue2AitkBlockExecutor::forward(backend,c,expert.layers[layer],layer_norms,
                adapters?&adapters->layers[layer]:nullptr,hidden.data(),length,cosine.data(),sine.data(),
                nar?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal,&out,canvas.k,canvas.v,
                prefix?prefix->length:0,error,!fast || capture_prefix != nullptr,
                fast?&constants:nullptr,fast?&workspace:nullptr)) return false;
        hidden=std::move(out.hidden);
        if (capture_prefix) { capture_prefix->keys.push_back(std::move(out.key)); capture_prefix->values.push_back(std::move(out.value)); }
    }
    tape->final_hidden=std::move(hidden);
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
    if (!store || tape.layer_inputs.size()!=expert.layers.size() || adapters.layers.size()!=expert.layers.size() ||
        gradient.size()!=size_t(tape.length*c.hidden)) return fail(error,"invalid expert backward tape");
    std::vector<float> cosine,sine;
    trig(c,tape.length,prefix?prefix->length:0,cosine,sine);
    Yue2AitkBlockConstants constants;
    Yue2AitkBlockWorkspace workspace;
    const bool fast = fast_execution();
    if (fast && !constants.make(backend,c,tape.length,prefix?prefix->length:0,
            tape.nar?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal,cosine.data(),sine.data(),error)) return false;
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
