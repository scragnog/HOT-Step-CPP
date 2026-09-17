// Native endpoint contract probe. Root runs this through the guarded CUDA
// launcher; this source intentionally does not launch a process itself.
#include "../../engine/src/train/yue2-aitk-endpoints.h"
#include "../../engine/src/train/yue2-aitk-embedding-dequant.h"
#include "../../engine/ggml/include/ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "../../engine/ggml/include/ggml-cuda.h"
#endif
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
struct BackendGuard { ggml_backend_t p=nullptr; ~BackendGuard(){if(p)ggml_backend_free(p);} };
[[noreturn]] void fail(const std::string &s){throw std::runtime_error(s);}
void save(const fs::path&p,const std::vector<float>&v){std::ofstream o(p,std::ios::binary);if(!o)fail("cannot create "+p.string());o.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(float)));}
void check(const std::vector<float>&v,const char*name,bool bf16=true){if(v.empty())fail(std::string(name)+" is empty");for(float x:v){if(!std::isfinite(x))fail(std::string(name)+" contains non-finite output");if(bf16){const float r=ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));if(r!=x)fail(std::string(name)+" is not BF16-valued");}}}

int main(int argc,char**argv){
    try {
        if(argc!=3){std::cerr<<"usage: test_yue2_aitk_endpoints CHECKPOINT NEW_OUTPUT_DIR\n";return 2;}
        const fs::path checkpoint=argv[1], out=argv[2]; if(!fs::is_regular_file(checkpoint))fail("checkpoint missing"); if(fs::exists(out))fail("output directory already exists"); fs::create_directories(out);
#ifndef GGML_USE_CUDA
        fail("compile with GGML_USE_CUDA");
#else
        BackendGuard backend{ggml_backend_cuda_init(0)}; if(!backend.p)fail("CUDA backend unavailable");
        Yue2AitkModel model; std::string error;
        if(!model.load(checkpoint.string().c_str(),backend.p,yue2_aitk_load_embedding_bf16,&error))fail("model load: "+error);
        const int32_t ids[3]={0,1,152011}; Yue2AitkEndpointHost emb;
        if(!Yue2AitkEndpoints::token_embedding(backend.p,model,ids,3,&emb,&error))fail("embedding: "+error);check(emb.values,"embedding");save(out/"embedding.f32",emb.values);
        std::vector<float> ar_dy(emb.values.size(),0.0f);for(size_t i=0;i<ar_dy.size();++i)ar_dy[i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(float((int(i%17)-8))*0.007f));Yue2AitkEndpointHost ar;
        if(!Yue2AitkEndpoints::ar_final_norm(backend.p,model,emb.values.data(),ar_dy.data(),3,&ar,&error))fail("AR final: "+error);check(ar.values,"AR output");check(ar.dx,"AR dx",false);save(out/"ar_final.f32",ar.values);save(out/"ar_dx.f32",ar.dx);
        const int64_t frames=2;std::vector<float> latent(size_t(frames*64));for(size_t i=0;i<latent.size();++i)latent[i]=float((int(i%13)-6))*0.03125f;Yue2AitkEndpointHost front;
        if(!Yue2AitkEndpoints::nar_frontend(backend.p,model,latent.data(),frames,0.375f,&front,&error))fail("NAR frontend: "+error);check(front.values,"NAR frontend");save(out/"nar_frontend.f32",front.values);
        std::vector<float> nar_dy(size_t((frames+2)*64));for(size_t i=0;i<nar_dy.size();++i)nar_dy[i]=ggml_bf16_to_fp32(ggml_fp32_to_bf16(float((int(i%11)-5))*0.005f));Yue2AitkEndpointHost nar;
        if(!Yue2AitkEndpoints::nar_final(backend.p,model,front.values.data(),nar_dy.data(),frames+2,&nar,&error))fail("NAR final: "+error);check(nar.values,"NAR output");check(nar.dx,"NAR dx",false);save(out/"nar_final.f32",nar.values);save(out/"nar_dx.f32",nar.dx);
        std::ofstream meta(out/"metrics.json");meta<<"{\"embedding_elements\":"<<emb.values.size()<<",\"ar_elements\":"<<ar.values.size()<<",\"nar_frontend_elements\":"<<front.values.size()<<",\"nar_elements\":"<<nar.values.size()<<"}\n";return 0;
#endif
    } catch(const std::exception&e){std::cerr<<"test_yue2_aitk_endpoints: "<<e.what()<<"\n";return 1;}
}

