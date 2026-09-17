#pragma once
#include "yue2-aitk-stack.h"
#include "yue2-aitk-endpoints.h"
#include "yue2-aitk-train-state.h"
#include "yue2-aitk-head-loss.h"
#include "yue2-aitk-optimizer.h"
#include "yue2-aitk-batch.h"

// One explicit, deterministic joint update. The runner supplies sampled noise,
// timestep and target so this seam can be compared without conflating RNGs.
// No prefix is retained across calls, and neither expert updates before both
// objectives have supplied all adapter gradients.
namespace yue2_aitk_joint {
struct Metrics { double ar_ce=0, ar_kl=0, nar_mse=0, gradient_norm=0; int step=0; };
struct Input {
    const yue2_aitk::Batch * batch=nullptr;
    std::vector<float> noisy_latents, flow_target; // [crop frames,64]
    float timestep=0; // reference normalized BF16-valued timestep
};
using Progress=std::function<void(const char *)>;
inline bool run(ggml_backend_t backend, const Yue2AitkModel & model,
                Yue2AitkTrainState & state, yue2_aitk::Optimizer & optimizer,
                const Input & input, Metrics * metrics, std::string * error,
                const Progress & progress={}) {
    using yue2_aitk_executor_detail::fail;
    constexpr size_t H=2048,C=64;
    if (!input.batch || !metrics || !state.initialized() || !std::isfinite(input.timestep) ||
        input.timestep<0 || input.timestep>1) return fail(error,"invalid joint update input");
    const auto & batch=*input.batch;
    const size_t frames=batch.nar.semantic_tokens.size(), N=batch.ar.target_ids.size();
    if (!frames || frames>1500 || !N || input.noisy_latents.size()!=frames*C ||
        input.flow_target.size()!=frames*C || batch.ar.prediction_positions.size()!=N ||
        batch.ar.input_ids.empty() || batch.nar.ar.input_ids.empty() ||
        !yue2_aitk_executor_detail::finite_all(input.noisy_latents.data(),input.noisy_latents.size()) ||
        !yue2_aitk_executor_detail::finite_all(input.flow_target.data(),input.flow_target.size()))
        return fail(error,"invalid joint update sequence or flow tensors");
    for(size_t i=0;i<N;++i) if(batch.ar.prediction_positions[i]>=batch.ar.input_ids.size() ||
        batch.ar.target_ids[i]<0 || batch.ar.target_ids[i]>=yue2_aitk::kVocabSize)
        return fail(error,"invalid AR prediction position or target");
    const auto notify=[&](const char * stage){if(progress)progress(stage);};
    Metrics result;
    notify("AR adapted forward");
    Yue2AitkEndpointHost embeds;
    if(!Yue2AitkEndpoints::token_embedding(backend,model,batch.ar.input_ids.data(),batch.ar.input_ids.size(),&embeds,error)) return false;
    Yue2AitkStackTape ar;
    if(!yue2_aitk_stack::forward(backend,model,&state.ar_adapters(),false,embeds.values,
        batch.ar.input_ids.size(),nullptr,true,&ar,nullptr,error)) return false;
    // Final norm's small graph currently shares a forward/backward entry point.
    std::vector<float> zeros(ar.final_hidden.size(),0.0f);
    Yue2AitkEndpointHost adapted_norm;
    if(!Yue2AitkEndpoints::ar_final_norm(backend,model,ar.final_hidden.data(),zeros.data(),ar.length,&adapted_norm,error)) return false;
    notify("AR frozen teacher forward");
    Yue2AitkStackTape base;
    if(!yue2_aitk_stack::forward(backend,model,nullptr,false,embeds.values,ar.length,nullptr,false,&base,nullptr,error)) return false;
    Yue2AitkEndpointHost base_norm;
    if(!Yue2AitkEndpoints::ar_final_norm(backend,model,base.final_hidden.data(),zeros.data(),base.length,&base_norm,error)) return false;
    std::vector<float> selected(N*H),selected_base(N*H);
    std::vector<uint32_t> targets(N);
    for(size_t i=0;i<N;++i) {
        const size_t row=batch.ar.prediction_positions[i];
        std::copy_n(adapted_norm.values.data()+row*H,H,selected.data()+i*H);
        std::copy_n(base_norm.values.data()+row*H,H,selected_base.data()+i*H);
        targets[i]=uint32_t(batch.ar.target_ids[i]);
    }
    std::vector<uint16_t> selected_grad(N*H); float ce=0,kl=0;
    yue2_aitk_head_loss::Request head;
    head.backend=backend;head.head=&model.lm_head();head.adapted_hidden=selected.data();
    head.base_hidden=selected_base.data();head.targets=targets.data();head.positions=N;
    head.hidden=H;head.kl_weight=0.2f;head.adapted_hidden_grad_bf16=selected_grad.data();head.ce_sum=&ce;head.kl_sum=&kl;
    notify("AR CE and KL backward");
    if(yue2_aitk_head_loss::compute(head,error)!=yue2_aitk_head_loss::Status::success) return false;
    result.ar_ce=double(ce)/N;result.ar_kl=double(kl)/N;
    notify("AR transformer backward");
    for(size_t i=0;i<N;++i) for(size_t d=0;d<H;++d)
        zeros[batch.ar.prediction_positions[i]*H+d]=ggml_bf16_to_fp32(ggml_bf16_t{selected_grad[i*H+d]});
    if(!Yue2AitkEndpoints::ar_final_norm(backend,model,ar.final_hidden.data(),zeros.data(),ar.length,&adapted_norm,error)) return false;
    if(!yue2_aitk_stack::backward(backend,model,state.ar_adapters(),ar,nullptr,std::move(adapted_norm.dx),
        [&](size_t layer,const Yue2AitkBlockBackwardHost & g,std::string * why){return state.upload_gradients(false,int(layer),g,why);},error)) return false;
    // Recompute with the current adapted AR. This cache is detached host data.
    // It is local to this step and cannot survive the optimizer update below.
    notify("Refresh detached AR conditioning");
    Yue2AitkEndpointHost condition;
    if(!Yue2AitkEndpoints::token_embedding(backend,model,batch.nar.ar.input_ids.data(),batch.nar.ar.input_ids.size(),&condition,error)) return false;
    Yue2AitkStackTape condition_tape; Yue2AitkPrefixHost prefix;
    if(!yue2_aitk_stack::forward(backend,model,&state.ar_adapters(),false,condition.values,
        batch.nar.ar.input_ids.size(),nullptr,false,&condition_tape,&prefix,error)) return false;
    notify("NAR forward");
    Yue2AitkEndpointHost frontend;
    if(!Yue2AitkEndpoints::nar_frontend(backend,model,input.noisy_latents.data(),frames,input.timestep,&frontend,error)) return false;
    Yue2AitkStackTape nar;
    if(!yue2_aitk_stack::forward(backend,model,&state.nar_adapters(),true,frontend.values,frames+2,&prefix,true,&nar,nullptr,error)) return false;
    std::vector<float> flow_grad((frames+2)*C,0.0f); Yue2AitkEndpointHost prediction;
    if(!Yue2AitkEndpoints::nar_final(backend,model,nar.final_hidden.data(),flow_grad.data(),nar.length,&prediction,error)) return false;
    for(size_t i=0;i<frames*C;++i) {
        const float diff=prediction.values[C+i]-input.flow_target[i];
        result.nar_mse+=double(diff)*diff;
        flow_grad[C+i]=2.0f*diff/float(frames*C);
    }
    result.nar_mse/=double(frames*C);
    notify("NAR flow backward");
    if(!Yue2AitkEndpoints::nar_final(backend,model,nar.final_hidden.data(),flow_grad.data(),nar.length,&prediction,error)) return false;
    if(!yue2_aitk_stack::backward(backend,model,state.nar_adapters(),nar,&prefix,std::move(prediction.dx),
        [&](size_t layer,const Yue2AitkBlockBackwardHost & g,std::string * why){return state.upload_gradients(true,int(layer),g,why);},error)) return false;
    if(!std::isfinite(result.ar_ce)||!std::isfinite(result.ar_kl)||!std::isfinite(result.nar_mse)) return fail(error,"nonfinite joint loss");
    notify("Joint clipping and AdamW8bit update");
    if(!state.clip_gradients(1.0f,&result.gradient_norm,error)) return false;
    ggml_backend_synchronize(backend);
    try { optimizer.step_once(yue2_aitk::StepConfig{}); }
    catch(const std::exception & e){return fail(error,e.what());}
    result.step=optimizer.step(); *metrics=result;
    return true;
}
} // namespace yue2_aitk_joint
