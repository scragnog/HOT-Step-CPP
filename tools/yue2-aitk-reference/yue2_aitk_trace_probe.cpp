#include "../../engine/src/train/yue2-aitk-stack.h"
#include "../../engine/src/train/yue2-aitk-endpoints.h"
#include "../../engine/src/train/yue2-aitk-embedding-dequant.h"
#include "ggml-cuda.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc,char ** argv) {
  try {
    if(argc!=3 && argc!=4) throw std::runtime_error("trace_probe CHECKPOINT NEW_DIR [--prescale|DETAIL_LAYER]");
    const int detail_layer=argc==4 && std::string(argv[3])!="--prescale"?std::stoi(argv[3]):5;
    if(detail_layer<0||detail_layer>=28)throw std::runtime_error("invalid layer");
    namespace fs=std::filesystem; const fs::path out=argv[2];
    if(fs::exists(out))throw std::runtime_error("output exists");fs::create_directories(out);
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");
#else
    setenv("NVIDIA_TF32_OVERRIDE","0",1);
#endif
    struct Backend{ggml_backend_t p;~Backend(){if(p)ggml_backend_free(p);}} backend{ggml_backend_cuda_init(0)};
    if(!backend.p)throw std::runtime_error("CUDA unavailable");
    Yue2AitkModel model;std::string error;
    if(!model.load(argv[1],backend.p,yue2_aitk_load_embedding_bf16,&error))throw std::runtime_error(error);
    auto write=[&](const std::string &name,const std::vector<float>&v){
      std::ofstream f(out/(name+".f32"),std::ios::binary);
      f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));f.close();
      if(!f)throw std::runtime_error("trace write failed");
    };
    auto tape_write=[&](const char*expert,const Yue2AitkStackTape&t){
      for(size_t i=0;i<28;++i)write(std::string(expert)+".layer"+std::to_string(i),i==27?t.final_hidden:yue2_aitk_stack::widen(t.layer_inputs[i+1]));
    };
    const int32_t ids[]={1,2,151848,151851,151865,151895,151852};
    Yue2AitkEndpointHost emb;
    if(!Yue2AitkEndpoints::token_embedding(backend.p,model,ids,7,&emb,&error))throw std::runtime_error(error);
    write("embedding",emb.values);
    Yue2AitkStackTape ar;Yue2AitkPrefixHost prefix;
    if(!yue2_aitk_stack::forward(backend.p,model,nullptr,false,emb.values,7,nullptr,true,&ar,&prefix,&error))throw std::runtime_error(error);
    tape_write("ar",ar);
    {
      using namespace yue2_aitk_executor_detail;
      Runtime r;Yue2AitkGraphConfig c;
      if(!make_runtime(backend.p,64u*1024u*1024u,&r,&error))throw std::runtime_error(error);
      auto *x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,2048,7);
      auto *co=ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,64,1,7,1);
      auto *si=ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,64,1,7,1);
      std::vector<uint16_t> mask_values;
      auto *mask=make_mask(r.ctx,7,0,Yue2AitkMaskMode::causal,&mask_values,&error);
      auto layer=yue2_aitk_graph::block(r.ctx,c,model.ar().layers[detail_layer],yue2_aitk_stack::norms(model,false,detail_layer),nullptr,x,co,si,mask);
      auto *g=ggml_new_graph_custom(r.ctx,8192,false);ggml_build_forward_expand(g,layer.hidden);
      if(argc==4 && std::string(argv[3])=="--prescale"){
        for(int n=0;n<ggml_graph_n_nodes(g);++n){
          auto *node=ggml_graph_nodes(g)[n];
          if(node->op==GGML_OP_FLASH_ATTN_TRAIN){
            const float s=std::sqrt(1.0f/std::sqrt(128.0f)),one=1;
            node->src[0]=ggml_scale(r.ctx,ggml_cont(r.ctx,node->src[0]),s);
            node->src[1]=ggml_scale(r.ctx,ggml_cont(r.ctx,node->src[1]),s);
            std::memcpy(node->op_params,&one,sizeof(one));
          }
        }
        g=ggml_new_graph_custom(r.ctx,8192,false);ggml_build_forward_expand(g,layer.hidden);
      }
      if(!supports_all(backend.p,g,&error)||!allocate(&r,backend.p,&error))throw std::runtime_error(error);
      std::vector<float> cv,sv;yue2_aitk_stack::trig(c,7,0,cv,sv);
      auto xv=yue2_aitk_stack::widen(ar.layer_inputs[detail_layer]);
      setup_inputs(backend.p,x,co,si,mask,xv.data(),cv.data(),sv.data(),mask_values);
      if(ggml_backend_graph_compute(backend.p,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("detail graph failed");
      const char *sites[]={"qkv","o","gate_up","down"};int site=0;
      for(int n=0;n<ggml_graph_n_nodes(g);++n){
        auto *node=ggml_graph_nodes(g)[n];
        if(node->op==GGML_OP_CONVROT8){
          for(int which=0;which<2;++which){
            auto *t=which?node:node->src[1];std::vector<float> v(ggml_nelements(t));
            ggml_backend_tensor_get(t,v.data(),0,ggml_nbytes(t));
            write(std::string("ar.detail5.")+sites[site]+(which?".output":".input"),v);
          }++site;
        }
        if(node->type==GGML_TYPE_F32&&ggml_is_contiguous(node)){
          std::vector<float> v(ggml_nelements(node));ggml_backend_tensor_get(node,v.data(),0,ggml_nbytes(node));
          write(std::string("ar.node5.")+std::to_string(n)+"."+ggml_op_name(node->op),v);
        }
      }
    }
    Yue2AitkEndpointHost norm;
    std::vector<float> zero(7*2048,0);
    if(!Yue2AitkEndpoints::ar_final_norm(backend.p,model,ar.final_hidden.data(),zero.data(),7,&norm,&error))throw std::runtime_error(error);
    write("ar.norm",norm.values);
    std::vector<float> noisy(128);
    for(size_t i=0;i<128;++i)noisy[i]=.625f*float(int(i%13)-6)*.03125f+.375f*float(int(i%17)-8)*.0625f;
    Yue2AitkEndpointHost front;
    if(!Yue2AitkEndpoints::nar_frontend(backend.p,model,noisy.data(),2,.375f,&front,&error))throw std::runtime_error(error);
    write("nar.frontend",front.values);
    Yue2AitkStackTape nar;
    if(!yue2_aitk_stack::forward(backend.p,model,nullptr,true,front.values,4,&prefix,true,&nar,nullptr,&error))throw std::runtime_error(error);
    tape_write("nar",nar);
    std::vector<float> dy(4*64,0);Yue2AitkEndpointHost pred;
    if(!Yue2AitkEndpoints::nar_final(backend.p,model,nar.final_hidden.data(),dy.data(),4,&pred,&error))throw std::runtime_error(error);
    write("prediction",std::vector<float>(pred.values.begin()+64,pred.values.end()-64));
    std::cout<<"Saved frozen native layer trace\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
