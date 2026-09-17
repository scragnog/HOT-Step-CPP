#include "../../engine/src/train/yue2-aitk-cursor-loss.h"
#include <cstdio>
#include <stdexcept>

static void require(bool ok,const char * message) { if(!ok) throw std::runtime_error(message); }
static double reference(const std::vector<float> & h,const std::vector<float> & w,
                        const std::vector<std::pair<int32_t,int32_t>> & spans) {
    double total=0;
    for(size_t frame=0;frame<spans.size();++frame) {
        double query[4]={},scores[3]={};
        for(int out=0;out<4;++out)for(int in=0;in<4;++in)query[out]+=w[out*4+in]*double(h[(4+frame)*4+in]);
        for(int token=0;token<3;++token)for(int d=0;d<4;++d)scores[token]+=query[d]*h[token*4+d]*0.5;
        const double max=(std::max)({scores[0],scores[1],scores[2]});
        double sum=0;for(double score:scores)sum+=std::exp(score-max);
        const auto span=spans[frame];
        for(int token=span.first;token<span.second;++token)total+=(max+std::log(sum)-scores[token])/double(span.second-span.first);
    }
    return total/double(spans.size());
}
static void check(ggml_backend_t backend,size_t frames) {
    using namespace yue2_aitk_executor_detail;
    Runtime state; std::string error;
    require(make_runtime(backend,1024*1024,&state,&error),error.c_str());
    auto * w=ggml_new_tensor_2d(state.ctx,GGML_TYPE_F32,4,4);
    auto * dw=ggml_new_tensor_2d(state.ctx,GGML_TYPE_F32,4,4);
    require(allocate(&state,backend,&error),error.c_str());
    std::vector<float> weights(16),hidden((frames+4)*4),gradient(hidden.size(),0),head_grad(16);
    for(size_t i=0;i<weights.size();++i) weights[i]=0.03f*float(i+1);
    for(size_t i=0;i<hidden.size();++i) hidden[i]=0.3f*std::sin(float(i)*0.37f);
    std::vector<std::pair<int32_t,int32_t>> spans;
    for(size_t i=0;i<frames;++i)spans.emplace_back(int(i%2),int(i%2)+2);
    ggml_backend_tensor_set(w,weights.data(),0,ggml_nbytes(w));
    yue2_aitk_cursor_loss::Request req;
    req.backend=backend;req.head=w;req.head_gradient=dw;req.hidden=&hidden;req.hidden_gradient=&gradient;
    req.hidden_size=4;req.sequence=frames+4;req.lyric_start=0;req.lyric_count=3;req.audio_start=4;
    req.frame_tokens=&spans;req.weight=0.08f;
    double loss=0;require(yue2_aitk_cursor_loss::compute(req,&loss,&error),error.c_str());
    std::printf("loss %.10f reference %.10f\n",loss,reference(hidden,weights,spans)); require(std::abs(loss-reference(hidden,weights,spans))<1e-5,"cursor mean loss differs from double reference");
    ggml_backend_tensor_get(dw,head_grad.data(),0,ggml_nbytes(dw));
    auto fd=[&](std::vector<float> & values,size_t index) {
        const float old=values[index],eps=0.002f;values[index]=old+eps;
        const double plus=reference(hidden,weights,spans);values[index]=old-eps;
        const double minus=reference(hidden,weights,spans);values[index]=old;
        return 0.08*(plus-minus)/(2*eps);
    };
    for(size_t index: {size_t(1),size_t(5),size_t(14)})
        require(std::abs(fd(weights,index)-head_grad[index])<2e-6,"cursor head gradient differs from finite difference");
    for(size_t index: {size_t(1),size_t(9),size_t(17),hidden.size()-2})
        require(std::abs(fd(hidden,index)-gradient[index])<2e-6,"cursor query/key gradient differs from finite difference");
    require(std::abs(gradient[1])>1e-8,"lyric-key gradient is zero");
    require(std::abs(gradient[17])>1e-8,"audio-query gradient is zero");
    require(gradient[12]==0,"unrelated hidden row received gradient");
    std::printf("cursor loss/gradient check passed: %zu frames\n",frames);
}
int main(int argc,char ** argv) {
    try {
        ggml_backend_load_all_from_path("engine/build/Release");
        auto * device=ggml_backend_dev_by_name(argc>1?argv[1]:"CPU");
        require(device!=nullptr,"requested backend unavailable");
        auto backend=ggml_backend_dev_init(device,nullptr);
        require(backend!=nullptr,"backend init failed");
        check(backend,5);check(backend,133);
        ggml_backend_free(backend);return 0;
    } catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}

