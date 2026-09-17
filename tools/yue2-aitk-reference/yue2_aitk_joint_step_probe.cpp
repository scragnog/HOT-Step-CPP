// Development-only two-step integration fixture, not an album training runner.
#include "../../engine/src/train/yue2-aitk-joint-step.h"
#include "../../engine/src/train/yue2-aitk-resume.h"
#include "../../engine/src/train/yue2-aitk-embedding-dequant.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc,char ** argv) {
    try {
        if(argc!=3 && !(argc==4 && std::string(argv[3])=="--parity-export")) throw std::runtime_error("usage: joint_step_probe CHECKPOINT NEW_OUTPUT_DIR [--parity-export]");
        namespace fs=std::filesystem;
        const fs::path out=argv[2];
        if(fs::exists(out)) throw std::runtime_error("output directory already exists");
        fs::create_directories(out);
        // Must precede CUDA initialization. GGML's F32 nodes alone still use
        // cuBLAS handles configured for TF32.
#ifdef _WIN32
        if(_putenv_s("NVIDIA_TF32_OVERRIDE","0")) throw std::runtime_error("cannot set strict math policy");
#else
        if(setenv("NVIDIA_TF32_OVERRIDE","0",1)) throw std::runtime_error("cannot set strict math policy");
#endif
        struct Backend {ggml_backend_t p;~Backend(){if(p)ggml_backend_free(p);}} backend{ggml_backend_cuda_init(0)};
        if(!backend.p) throw std::runtime_error("CUDA unavailable");
        std::string error; Yue2AitkModel model; Yue2AitkTrainState state;
        std::cout<<"Load checkpoint and initialize both experts' adapters"<<std::endl;
        if(!model.load(argv[1],backend.p,yue2_aitk_load_embedding_bf16,&error) ||
           !state.initialize(backend.p,20260917,&error)) throw std::runtime_error(error);
        std::vector<yue2_aitk::ParameterSpec> specs;
        for(const auto & p:state.named_tensors()) specs.push_back({p.name,p.parameter,p.gradient});
        yue2_aitk::Optimizer optimizer(backend.p,0,std::move(specs));
        if(argc==4 && !yue2_aitk::yue2_aitk_write_resume((out/"initial.resume").string().c_str(),optimizer.capture(),"synthetic-joint-v1"))
            throw std::runtime_error("cannot export initial F32 state");
        yue2_aitk::SongInput song; song.semantic_tokens={12,42};song.latents.resize(128);
        for(size_t i=0;i<128;++i)song.latents[i]=float(int(i%13)-6)*0.03125f;
        yue2_aitk::PromptInput prompt;prompt.retained_prefix_ids={1,2};prompt.dropped_prefix_ids={1,2};prompt.retain_abc=false;
        yue2_aitk::Batch batch;
        if(!yue2_aitk::build(prompt,song,{0,2},0,&batch,&error)) throw std::runtime_error(error);
        yue2_aitk_joint::Input input;input.batch=&batch;input.timestep=0.375f;
        input.noisy_latents.resize(128);input.flow_target.resize(128);
        for(size_t i=0;i<128;++i) {
            const float noise=float(int(i%17)-8)*0.0625f;
            input.noisy_latents[i]=0.625f*song.latents[i]+0.375f*noise;
            input.flow_target[i]=noise-song.latents[i];
        }
        std::ofstream metrics(out/"metrics.jsonl"); if(!metrics)throw std::runtime_error("cannot create metrics file");
        for(int step=0;step<2;++step) {
            yue2_aitk_joint::Metrics m;
            if(!yue2_aitk_joint::run(backend.p,model,state,optimizer,input,&m,&error,
                [step](const char * stage){std::cout<<"step "<<step+1<<": "<<stage<<std::endl;})) throw std::runtime_error(error);
            metrics<<std::setprecision(17)<<"{\"step\":"<<m.step<<",\"ce\":"<<m.ar_ce<<",\"kl\":"<<m.ar_kl
                <<",\"mse\":"<<m.nar_mse<<",\"gradient_norm\":"<<m.gradient_norm<<"}\n";metrics.flush();
            if(!(m.gradient_norm>0))throw std::runtime_error("joint gradients are zero");
            if(step==0 && std::abs(m.ar_kl)>1e-5)throw std::runtime_error("zero-B adapter differs from frozen teacher");
            if(argc==4 && !yue2_aitk::yue2_aitk_write_resume((out/("step"+std::to_string(step+1)+".resume")).string().c_str(),optimizer.capture(),"synthetic-joint-v1"))
                throw std::runtime_error("cannot export updated F32 state");
        }
        if(!state.export_snapshot((out/"adapter-step2.safetensors").string().c_str(),2,&error))throw std::runtime_error(error);
        std::cout<<"two native joint steps and fused adapter export completed"<<std::endl;
        return 0;
    }catch(const std::exception & e){std::cerr<<"joint step probe: "<<e.what()<<std::endl;return 1;}
}
