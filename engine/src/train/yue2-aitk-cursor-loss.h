#pragma once
#include "yue2-aitk-block-executor.h"
#include <utility>

// Cursor gradients enter the same post-final-norm buffer as CE/KL. Only
// semantic prediction rows are queries; ABC and END targets are excluded.
namespace yue2_aitk_cursor_loss {
struct Request {
    ggml_backend_t backend = nullptr;
    ggml_tensor * head = nullptr, * head_gradient = nullptr;
    const std::vector<float> * hidden = nullptr;
    std::vector<float> * hidden_gradient = nullptr;
    int64_t hidden_size = 2048, sequence = 0, lyric_start = 0, lyric_count = 0, audio_start = 0;
    // Half-open lyric-token intervals. Each frame has a uniform target over
    // its word's overlapping tokens, matching the Legacy cursor objective.
    const std::vector<std::pair<int32_t,int32_t>> * frame_tokens = nullptr;
    float weight = 0.08f;
};

inline bool compute(const Request & r, double * mean_loss, std::string * error) {
    using namespace yue2_aitk_executor_detail;
    const int64_t H=r.hidden_size,L=r.lyric_count;
    const size_t frames=r.frame_tokens?r.frame_tokens->size():0;
    if (!mean_loss || !r.backend || !r.head || !r.head_gradient || !r.hidden || !r.hidden_gradient ||
        H<=0 || L<=0 || !frames || r.sequence<=0 || r.lyric_start<0 || r.audio_start<0 ||
        r.lyric_start+L>r.sequence || r.audio_start+int64_t(frames)>r.sequence ||
        r.hidden->size()!=size_t(H*r.sequence) || r.hidden_gradient->size()!=r.hidden->size() ||
        r.head->type!=GGML_TYPE_F32 || r.head->ne[0]!=H || r.head->ne[1]!=H ||
        r.head_gradient->type!=GGML_TYPE_F32 || !ggml_are_same_shape(r.head,r.head_gradient) ||
        !std::isfinite(r.weight) || r.weight<=0) return fail(error,"invalid cursor loss inputs");
    for (const auto & span : *r.frame_tokens)
        if(span.first<0 || span.second<=span.first || span.second>L) return fail(error,"cursor token range is invalid");
    Runtime constants;
    if(!make_runtime(r.backend,8*ggml_tensor_overhead()+1024,&constants,error))return false;
    ggml_tensor * keys=ggml_new_tensor_2d(constants.ctx,GGML_TYPE_F32,H,L);
    ggml_tensor * key_grad=ggml_new_tensor_2d(constants.ctx,GGML_TYPE_F32,H,L);
    if(!allocate(&constants,r.backend,error))return false;
    ggml_backend_tensor_set(keys,r.hidden->data()+r.lyric_start*H,0,ggml_nbytes(keys));
    ggml_backend_tensor_memset(key_grad,0,0,ggml_nbytes(key_grad));
    ggml_backend_tensor_memset(r.head_gradient,0,0,ggml_nbytes(r.head_gradient));
    struct Chunk {
        Runtime runtime;
        ggml_cgraph * graph=nullptr;
        ggml_tensor * query=nullptr,*target=nullptr,*loss=nullptr,*query_grad=nullptr;
        bool make(const Request & r,ggml_tensor * keys,ggml_tensor * key_grad,int64_t count,size_t total,std::string * error) {
            if(!make_runtime(r.backend,2*1024*1024,&runtime,error))return false;
            auto * ctx=runtime.ctx;
            const int64_t H=r.hidden_size,L=r.lyric_count;
            query=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,H,count);
            target=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,L,count);
            auto * upstream=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);
            auto * q=ggml_mul_mat(ctx,r.head,query);
            const float scale=1.0f/std::sqrt(float(H));
            auto * scores=ggml_scale(ctx,ggml_mul_mat(ctx,keys,q),scale);
            loss=ggml_cross_entropy_loss(ctx,scores,target);
            auto * ds=ggml_scale(ctx,ggml_cross_entropy_loss_back(ctx,upstream,scores,target),scale);
            auto * dq=ggml_mul_mat(ctx,ggml_cont(ctx,ggml_transpose(ctx,keys)),ds);
            query_grad=ggml_mul_mat(ctx,ggml_cont(ctx,ggml_transpose(ctx,r.head)),dq);
            auto * dk=ggml_mul_mat(ctx,ggml_cont(ctx,ggml_transpose(ctx,q)),ggml_cont(ctx,ggml_transpose(ctx,ds)));
            auto * dw=ggml_mul_mat(ctx,ggml_cont(ctx,ggml_transpose(ctx,query)),ggml_cont(ctx,ggml_transpose(ctx,dq)));
            graph=ggml_new_graph_custom(ctx,256,false);
            ggml_build_forward_expand(graph,loss);
            ggml_build_forward_expand(graph,query_grad);
            ggml_build_forward_expand(graph,ggml_cpy(ctx,ggml_add(ctx,key_grad,dk),key_grad));
            ggml_build_forward_expand(graph,ggml_cpy(ctx,ggml_add(ctx,r.head_gradient,dw),r.head_gradient));
            if(!supports_all(r.backend,graph,error)||!allocate(&runtime,r.backend,error))return false;
            const float gradient_scale=r.weight*float(count)/float(total);
            ggml_backend_tensor_set(upstream,&gradient_scale,0,sizeof(float));
            return true;
        }
    } full,tail;
    *mean_loss=0;
    std::vector<float> targets,query_gradient;
    for(size_t offset=0;offset<frames;offset+=128) {
        const size_t count=(std::min)(size_t(128),frames-offset);
        Chunk & chunk=count==128?full:tail;
        if(!chunk.graph&&!chunk.make(r,keys,key_grad,int64_t(count),frames,error))return false;
        targets.assign(size_t(L)*count,0);
        for(size_t row=0;row<count;++row) {
            const auto span=(*r.frame_tokens)[offset+row];
            const float mass=1.0f/float(span.second-span.first);
            for(int32_t token=span.first;token<span.second;++token) targets[row*size_t(L)+size_t(token)]=mass;
        }
        ggml_backend_tensor_set(chunk.query,r.hidden->data()+(size_t(r.audio_start)+offset)*size_t(H),0,ggml_nbytes(chunk.query));
        ggml_backend_tensor_set(chunk.target,targets.data(),0,ggml_nbytes(chunk.target));
        if(ggml_backend_graph_compute(r.backend,chunk.graph)!=GGML_STATUS_SUCCESS)return fail(error,"cursor graph compute failed");
        float loss=0; ggml_backend_tensor_get(chunk.loss,&loss,0,sizeof(float));
        if(!std::isfinite(loss))return fail(error,"nonfinite cursor loss");
        *mean_loss+=double(loss)*double(count)/double(frames);
        query_gradient.resize(size_t(H)*count);
        ggml_backend_tensor_get(chunk.query_grad,query_gradient.data(),0,ggml_nbytes(chunk.query_grad));
        for(size_t i=0;i<query_gradient.size();++i)
            (*r.hidden_gradient)[(size_t(r.audio_start)+offset)*size_t(H)+i]+=query_gradient[i];
    }
    std::vector<float> key_gradient(size_t(H*L));
    ggml_backend_tensor_get(key_grad,key_gradient.data(),0,ggml_nbytes(key_grad));
    for(size_t i=0;i<key_gradient.size();++i) (*r.hidden_gradient)[size_t(r.lyric_start*H)+i]+=key_gradient[i];
    return true;
}
} // namespace yue2_aitk_cursor_loss
