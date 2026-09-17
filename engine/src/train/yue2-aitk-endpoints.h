#pragma once

// Native YuE2 endpoint seams. These are intentionally small graph calls:
// embedding lookup, AR final RMSNorm, NAR latent frontend, and NAR final
// projection. They own only temporary graph state and never update weights.

#include "yue2-aitk-graph.h"
#include "yue2-aitk-model.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace yue2_aitk_endpoints_detail {
struct Runtime { ggml_context * ctx=nullptr; ggml_backend_buffer_t buffer=nullptr; ~Runtime(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);} };
inline bool fail(std::string *e,const char*m){if(e)*e=m;return false;} inline bool fail(std::string*e,const std::string&m){if(e)*e=m;return false;}
inline bool finite(const float*p,size_t n){if(!p)return false;for(size_t i=0;i<n;++i)if(!std::isfinite(p[i]))return false;return true;}
inline bool init(ggml_backend_t b,Runtime*r,std::string*e){if(!b||!r)return fail(e,"null endpoint backend");ggml_init_params p{};p.mem_size=32ull*1024ull*1024ull;p.no_alloc=true;r->ctx=ggml_init(p);return r->ctx?true:fail(e,"endpoint context allocation failed");}
inline bool alloc(Runtime*r,ggml_backend_t b,std::string*e){r->buffer=ggml_backend_alloc_ctx_tensors(r->ctx,b);return r->buffer?true:fail(e,"endpoint backend allocation failed");}
inline bool support(ggml_backend_t b,ggml_cgraph*g,std::string*e){auto n=ggml_graph_n_nodes(g);auto v=ggml_graph_nodes(g);for(int i=0;i<n;++i)if(!ggml_backend_supports_op(b,v[i]))return fail(e,"endpoint graph node is unsupported");return true;}
inline bool compute(ggml_backend_t b,ggml_cgraph*g,std::string*e){ggml_graph_reset(g);return ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS?true:fail(e,"endpoint graph compute failed");}
inline ggml_tensor*f32(ggml_context*c,ggml_tensor*x){return x->type==GGML_TYPE_F32?x:ggml_cast(c,x,GGML_TYPE_F32);}
inline bool shape2(const float*p,int64_t a,int64_t b){return a>0&&b>0&&p&&finite(p,size_t(a)*size_t(b));}
}

struct Yue2AitkEndpointHost { std::vector<float> values; std::vector<float> dx; };

class Yue2AitkEndpoints {
public:
    static bool token_embedding(ggml_backend_t backend, const Yue2AitkModel & model,
                                const int32_t * ids, int64_t count, Yue2AitkEndpointHost * out,
                                std::string * error=nullptr) {
        using namespace yue2_aitk_endpoints_detail; if(!out||!ids||count<=0||count>100000)return fail(error,"invalid embedding request");
        ggml_tensor* emb=model.embedding(); if(!emb||emb->ne[0]!=2048)return fail(error,"embedding is unavailable"); Runtime r;if(!init(backend,&r,error))return false;
        for (int64_t i=0;i<count;++i) if (ids[i] < 0 || ids[i] >= emb->ne[1]) return fail(error,"embedding id is out of range");
        ggml_tensor* id=ggml_new_tensor_1d(r.ctx,GGML_TYPE_I32,count); if(!id)return fail(error,"embedding id allocation failed"); ggml_tensor* y=ggml_get_rows(r.ctx,emb,id); if(!y)return fail(error,"embedding lookup construction failed"); y=f32(r.ctx,y);
        ggml_cgraph*g=ggml_new_graph_custom(r.ctx,2048,true);if(!g)return fail(error,"embedding graph allocation failed");ggml_build_forward_expand(g,y);if(!support(backend,g,error)||!alloc(&r,backend,error))return false;ggml_backend_tensor_set(id,ids,0,ggml_nbytes(id));if(!compute(backend,g,error))return false;
        out->values.resize(ggml_nelements(y));ggml_backend_tensor_get(y,out->values.data(),0,ggml_nbytes(y));return true;
    }

    static bool ar_final_norm(ggml_backend_t backend, const Yue2AitkModel & model,
                              const float * hidden, const float * dy, int64_t tokens,
                              Yue2AitkEndpointHost * out, std::string * error=nullptr) {
        using namespace yue2_aitk_endpoints_detail; if(!out||!shape2(hidden,2048,tokens)||!shape2(dy,2048,tokens)||!model.ar().final_norm)return fail(error,"invalid AR final norm request"); Runtime r;if(!init(backend,&r,error))return false;
        ggml_tensor*x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,2048,tokens),*up=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,2048,tokens);if(!x||!up)return fail(error,"AR final norm input allocation failed");ggml_set_param(x);
        ggml_tensor*y=yue2_aitk_graph::rms(r.ctx,yue2_aitk_graph::round(r.ctx,x),model.ar().final_norm,1e-6f);ggml_tensor*loss=ggml_sum(r.ctx,ggml_mul(r.ctx,y,up));ggml_cgraph*g=ggml_new_graph_custom(r.ctx,4096,true);if(!loss||!g)return fail(error,"AR final norm graph allocation failed");ggml_build_forward_expand(g,loss);ggml_set_loss(loss);ggml_build_backward_expand(r.ctx,g,nullptr);if(!support(backend,g,error)||!alloc(&r,backend,error))return false;ggml_backend_tensor_set(x,hidden,0,ggml_nbytes(x));ggml_backend_tensor_set(up,dy,0,ggml_nbytes(up));if(!compute(backend,g,error))return false;
        out->values.resize(ggml_nelements(y));ggml_backend_tensor_get(y,out->values.data(),0,ggml_nbytes(y));ggml_tensor*gx=ggml_graph_get_grad(g,x);if(!gx)return fail(error,"AR final norm gradient missing");out->dx.resize(ggml_nelements(gx));ggml_backend_tensor_get(gx,out->dx.data(),0,ggml_nbytes(gx));return true;
    }

    static bool nar_frontend(ggml_backend_t backend, const Yue2AitkModel & model,
                             const float * latents, int64_t frames, float timestep,
                             Yue2AitkEndpointHost * out, std::string * error=nullptr) {
        using namespace yue2_aitk_endpoints_detail; if(!out||!shape2(latents,64,frames)||!std::isfinite(timestep))return fail(error,"invalid NAR frontend request");ggml_tensor*vae=model.ordinary("model.diffusion_model.vae2llm.weight"),*vb=model.ordinary("model.diffusion_model.vae2llm.bias"),*tb0=model.ordinary("model.diffusion_model.time_embedder.mlp.0.bias"),*tb2=model.ordinary("model.diffusion_model.time_embedder.mlp.2.bias"),*pos=model.ordinary("model.diffusion_model.latent_pos_embed.pe");if(!vae||!vb||!tb0||!tb2||!pos)return fail(error,"NAR frontend weights unavailable");const int64_t n=frames+2;Runtime r;if(!init(backend,&r,error))return false;ggml_tensor*in=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,64,n);if(!in)return fail(error,"NAR frontend input allocation failed");std::vector<float> padded(size_t(64*n),0.0f),freq(256);for(int64_t j=0;j<frames;++j)for(int64_t i=0;i<64;++i)padded[size_t(i+64*(j+1))]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(latents[size_t(i+64*j)]));const float tbf=ggml_bf16_to_fp32(ggml_fp32_to_bf16(timestep));for(int i=0;i<128;++i){float f=std::exp(-std::log(10000.0f)*float(i)/128.0f);freq[i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::cos(tbf*f)));freq[128+i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::sin(tbf*f)));} // Toolkit casts the time embedding input to BF16
        ggml_tensor*time_in=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,256,1);if(!time_in)return fail(error,"timestep allocation failed");
        ggml_tensor*x=yue2_aitk_graph::round(r.ctx,ggml_add(r.ctx,ggml_mul_mat(r.ctx,vae,in),f32(r.ctx,vb))); // rounded BF16 vae2llm
        ggml_tensor*tm=yue2_aitk_graph::linear(r.ctx,model.time0(),time_in,nullptr,tb0);tm=yue2_aitk_graph::round(r.ctx,ggml_silu(r.ctx,tm));tm=yue2_aitk_graph::linear(r.ctx,model.time2(),tm,nullptr,tb2);ggml_tensor*tm2=ggml_repeat(r.ctx,tm, x);x=yue2_aitk_graph::round(r.ctx,ggml_add(r.ctx,x,tm2));ggml_tensor*pv=ggml_cont(r.ctx,ggml_view_2d(r.ctx,pos,2048,n,pos->nb[1],0));x=yue2_aitk_graph::round(r.ctx,ggml_add(r.ctx,x,f32(r.ctx,pv)));
        ggml_cgraph*g=ggml_new_graph_custom(r.ctx,4096,true);if(!g)return fail(error,"NAR frontend graph allocation failed");ggml_build_forward_expand(g,x);if(!support(backend,g,error)||!alloc(&r,backend,error))return false;ggml_backend_tensor_set(in,padded.data(),0,ggml_nbytes(in));std::vector<float>time_host(256,0);for(int i=0;i<128;++i){time_host[i]=freq[i];time_host[128+i]=freq[128+i];}ggml_backend_tensor_set(time_in,time_host.data(),0,ggml_nbytes(time_in));if(!compute(backend,g,error))return false;out->values.resize(ggml_nelements(x));ggml_backend_tensor_get(x,out->values.data(),0,ggml_nbytes(x));return true;
    }

    static bool nar_final(ggml_backend_t backend, const Yue2AitkModel & model,
                          const float * hidden, const float * dy, int64_t tokens,
                          Yue2AitkEndpointHost * out, std::string * error=nullptr) {
        using namespace yue2_aitk_endpoints_detail; if(!out||!shape2(hidden,2048,tokens)||!shape2(dy,64,tokens)||!model.nar().final_norm)return fail(error,"invalid NAR final request");ggml_tensor*bias=model.ordinary("model.diffusion_model.llm2vae.bias");if(!bias)return fail(error,"NAR output bias unavailable");Runtime r;if(!init(backend,&r,error))return false;ggml_tensor*x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,2048,tokens),*up=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,64,tokens);if(!x||!up)return fail(error,"NAR final input allocation failed");ggml_set_param(x);ggml_tensor*n=yue2_aitk_graph::rms(r.ctx,yue2_aitk_graph::round(r.ctx,x),model.nar().final_norm,1e-6f);ggml_tensor*y=yue2_aitk_graph::linear(r.ctx,model.llm2vae(),n,nullptr,bias);ggml_tensor*loss=ggml_sum(r.ctx,ggml_mul(r.ctx,y,up));ggml_cgraph*g=ggml_new_graph_custom(r.ctx,8192,true);if(!loss||!g)return fail(error,"NAR final graph allocation failed");ggml_build_forward_expand(g,loss);ggml_set_loss(loss);ggml_build_backward_expand(r.ctx,g,nullptr);if(!support(backend,g,error)||!alloc(&r,backend,error))return false;ggml_backend_tensor_set(x,hidden,0,ggml_nbytes(x));ggml_backend_tensor_set(up,dy,0,ggml_nbytes(up));if(!compute(backend,g,error))return false;out->values.resize(ggml_nelements(y));ggml_backend_tensor_get(y,out->values.data(),0,ggml_nbytes(y));ggml_tensor*gx=ggml_graph_get_grad(g,x);if(!gx)return fail(error,"NAR final gradient missing");out->dx.resize(ggml_nelements(gx));ggml_backend_tensor_get(gx,out->dx.data(),0,ggml_nbytes(gx));return true;
    }
};






