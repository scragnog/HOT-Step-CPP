// Standalone CUDA-only contract probe for the staged YuE2 block graph.
// The root agent runs this against a fresh output directory; this translation
// unit deliberately has no checkpoint or engine integration side effects.

#include "../../engine/ggml/include/ggml.h"
#include "../../engine/ggml/include/ggml-alloc.h"
#include "../../engine/ggml/include/ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "../../engine/ggml/include/ggml-cuda.h"
#endif

#include "../../engine/src/train/yue2-aitk-graph.h"
#include "../../engine/src/train/yue2-aitk-block-executor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr int H = 256, F = 512, Q = 256, KV = 128, D = 64, NH = 4, NKV = 2, S = 3, R = 4;

struct Ctx { ggml_context * p = nullptr; ~Ctx() { if (p) ggml_free(p); } };
struct Backend { ggml_backend_t p = nullptr; ~Backend() { if (p) ggml_backend_free(p); } };
struct Buffer { ggml_backend_buffer_t p = nullptr; ~Buffer() { if (p) ggml_backend_buffer_free(p); } };
[[noreturn]] void fail(const std::string & s) { throw std::runtime_error(s); }

std::vector<uint8_t> bytes(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) fail("cannot open " + p.string());
    in.seekg(0, std::ios::end);
    const auto n = in.tellg();
    if (n < 0) fail("cannot size " + p.string());
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> v(static_cast<size_t>(n));
    if (!v.empty()) in.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size()));
    if (!in) fail("truncated " + p.string());
    return v;
}
template <typename T> std::vector<T> load(const fs::path & d, const char * name, size_t n) {
    auto v = bytes(d / name);
    if (v.size() != n * sizeof(T)) fail(std::string("wrong size for ") + name);
    std::vector<T> out(n); std::memcpy(out.data(), v.data(), v.size()); return out;
}
float bf16f(uint16_t x) { uint32_t u = uint32_t(x) << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f32bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); const uint32_t l = u & 0xffffu; return uint16_t((u >> 16) + (l > 0x8000u || (l == 0x8000u && ((u >> 16) & 1)))); }
void set(ggml_tensor * t, ggml_backend_t b, const void * p) { ggml_backend_tensor_set(t, p, 0, ggml_nbytes(t)); }
ggml_tensor * t2(ggml_context * c, ggml_type ty, int64_t x, int64_t y) { return ggml_new_tensor_2d(c, ty, x, y); }

struct Metric { double max_abs = 0, sum_abs = 0, l2 = 0, cosine = 0; size_t max_i = 0; };
Metric compare(const std::vector<float> & got, const std::vector<float> & ref) {
    if (got.size() != ref.size()) fail("comparison size mismatch");
    Metric m;
    double dot = 0, gn = 0, rn = 0;
    for (size_t i = 0; i < got.size(); ++i) { if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) fail("non-finite comparison value"); const double e = std::abs(double(got[i]) - double(ref[i])); m.sum_abs += e; m.l2 += e*e; dot += double(got[i])*ref[i]; gn += double(got[i])*got[i]; rn += double(ref[i])*ref[i]; if (e > m.max_abs) { m.max_abs = e; m.max_i = i; } }
    m.l2 = std::sqrt(m.l2); m.cosine = (gn > 0 && rn > 0) ? dot / std::sqrt(gn*rn) : 0;
    return m;
}
void write_f32(const fs::path & p, const std::vector<float> & v) { std::ofstream o(p, std::ios::binary); if (!o) fail("cannot create " + p.string()); o.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size()*4)); if (!o) fail("write failed"); }

int run(const fs::path & fixture, const fs::path & out, int prefix = 0) {
#ifndef GGML_USE_CUDA
    fail("compile with GGML_USE_CUDA");
#else
    if (!fs::is_directory(fixture)) fail("fixture directory missing");
    if (fs::exists(out)) fail("output directory already exists: " + out.string());
    fs::create_directories(out);

    const auto xbf = load<uint16_t>(fixture, "x.bin", H*S);
    const auto dybf = load<uint16_t>(fixture, "upstream_dy.bin", H*S);
    const auto fwdrefbf = load<uint16_t>(fixture, "forward.bin", H*S);
    const auto dxrefbf = load<uint16_t>(fixture, "dx.bin", H*S);
    const auto iwbf = load<uint16_t>(fixture, "input_layernorm_weight.bin", H);
    const auto pwbf = load<uint16_t>(fixture, "post_attention_layernorm_weight.bin", H);
    const auto qnbf = load<uint16_t>(fixture, "q_norm_weight.bin", D);
    const auto knbf = load<uint16_t>(fixture, "k_norm_weight.bin", D);
    const auto cosv = load<float>(fixture, "cos.bin", S*D/2);
    const auto sinv = load<float>(fixture, "sin.bin", S*D/2);

    Backend cuda{ggml_backend_cuda_init(0)}; if (!cuda.p) fail("CUDA backend unavailable");
    ggml_init_params ip{}; ip.mem_size = 512ull*1024ull*1024ull; ip.no_alloc = true;
    Ctx ctx{ggml_init(ip)}; if (!ctx.p) fail("ggml_init failed");

    ggml_tensor * h = t2(ctx.p, GGML_TYPE_F32, H, S), * dy = t2(ctx.p, GGML_TYPE_F32, H, S);
    ggml_tensor * in_norm = ggml_new_tensor_1d(ctx.p, GGML_TYPE_BF16, H), * post_norm = ggml_new_tensor_1d(ctx.p, GGML_TYPE_BF16, H);
    ggml_tensor * q_norm = ggml_new_tensor_1d(ctx.p, GGML_TYPE_BF16, D), * k_norm = ggml_new_tensor_1d(ctx.p, GGML_TYPE_BF16, D);
    ggml_tensor * cosine = ggml_new_tensor_4d(ctx.p, GGML_TYPE_F32, D/2, 1, S, 1), * sine = ggml_new_tensor_4d(ctx.p, GGML_TYPE_F32, D/2, 1, S, 1);
    if (prefix < 0 || prefix > 100) fail("invalid fixture prefix length");
    ggml_tensor * pk = prefix ? ggml_new_tensor_3d(ctx.p,GGML_TYPE_F32,D,S+prefix,NKV) : nullptr;
    ggml_tensor * pv = prefix ? ggml_new_tensor_3d(ctx.p,GGML_TYPE_F32,D,S+prefix,NKV) : nullptr;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx.p, GGML_TYPE_F16, S+prefix, S);
    if (!h || !dy || !in_norm || !post_norm || !q_norm || !k_norm || !cosine || !sine || !mask) fail("input tensor allocation failed");
    ggml_set_param(h);
    std::vector<float> xf(H*S), dyf(H*S), maskf((S+prefix)*S);
    for (size_t i=0;i<xf.size();++i) { xf[i]=bf16f(xbf[i]); dyf[i]=bf16f(dybf[i]); }
    for (int row=0; row<S; ++row) for (int col=0; col<S+prefix; ++col) maskf[row*(S+prefix)+col] = prefix || col <= row ? 0.0f : -INFINITY;
    std::vector<uint16_t> maskh(maskf.size()); for (size_t i=0;i<maskh.size();++i) maskh[i]=ggml_fp32_to_fp16(maskf[i]);
    std::vector<float> trigc(S*D/2), trigs(S*D/2); for(size_t i=0;i<trigc.size();++i){trigc[i]=bf16f(f32bf16(cosv[i]));trigs[i]=bf16f(f32bf16(sinv[i]));}

    auto make_linear = [&](int rows, int cols, const char * stem) {
        Yue2AitkConvRotLinear w; w.rows=rows; w.cols=cols; w.rotation=256;
        w.weight_i8=t2(ctx.p, GGML_TYPE_I8, cols, rows); w.scales_f32=ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, rows);
        if (!w.weight_i8 || !w.scales_f32) fail("linear allocation failed");
        w.weight_i8->name[0]=0; ggml_set_name(w.weight_i8, stem); return w;
    };
    Yue2AitkLayerWeights w{make_linear(Q+2*KV,H,"qkv"),make_linear(H,Q,"output"),make_linear(2*F,H,"gate_up"),make_linear(H,F,"down")};
    auto make_lora = [&](int input, int output, const char * stem) {
        Yue2AitkFusedLora a; a.input_width=input; a.output_width=output; a.rank=R; a.scale=1.0f;
        a.a=t2(ctx.p,GGML_TYPE_F32,input,R); a.b=t2(ctx.p,GGML_TYPE_F32,R,output); if(!a.a||!a.b) fail("LoRA allocation failed");
        ggml_set_param(a.a); ggml_set_param(a.b); ggml_set_name(a.a,(std::string(stem)+".A").c_str()); ggml_set_name(a.b,(std::string(stem)+".B").c_str()); return a;
    };
    Yue2AitkLayerAdapters ad{make_lora(H,Q+2*KV,"qkv"),make_lora(Q,H,"output"),make_lora(H,2*F,"gate_up"),make_lora(F,H,"down")};
    auto qdata=[&](const char*n,size_t z){return load<int8_t>(fixture,n,z);}; auto f32=[&](const char*n,size_t z){return load<float>(fixture,n,z);};
    auto put = [&](ggml_tensor*t,const void*p){set(t,cuda.p,p);};
    auto fill_linear=[&](Yue2AitkConvRotLinear & z,const char* q,const char*s,int rows,int cols){auto a=qdata(q,size_t(rows)*cols);auto b=f32(s,rows);put(z.weight_i8,a.data());put(z.scales_f32,b.data());};
    auto fill_lora=[&](Yue2AitkFusedLora & z,const char*an,const char*bn){auto a=f32(an,size_t(R)*z.input_width),b=f32(bn,size_t(z.output_width)*R);put(z.a,a.data());put(z.b,b.data());};

    Yue2AitkGraphConfig c; c.hidden=H;c.intermediate=F;c.heads=NH;c.kv_heads=NKV;c.head_dim=D;c.rms_eps=1e-6f;c.attention_precision=GGML_PREC_F32;
    Yue2AitkBlockNorms norms{in_norm,post_norm,q_norm,k_norm}; Yue2AitkBlockResult r=yue2_aitk_graph::block(ctx.p,c,w,norms,&ad,h,cosine,sine,mask,pk,pv,prefix);
    ggml_tensor * loss=ggml_sum(ctx.p,ggml_mul(ctx.p,r.hidden,dy)); ggml_cgraph * graph=ggml_new_graph_custom(ctx.p,32768,true); if(!loss||!graph) fail("graph allocation failed");
    ggml_build_forward_expand(graph,loss); ggml_set_loss(loss); ggml_build_backward_expand(ctx.p,graph,nullptr);
    const int node_count = ggml_graph_n_nodes(graph); ggml_tensor ** nodes = ggml_graph_nodes(graph);
    for(int i=0;i<node_count;++i) if(!ggml_backend_supports_op(cuda.p,nodes[i])) {
        const auto * t = nodes[i];
        std::ostringstream detail; detail << "CUDA unsupported node " << i << " " << ggml_op_name(t->op) << " " << t->name << " type=" << ggml_type_name(t->type);
        if (t->op == GGML_OP_UNARY) detail << " unary=" << ggml_unary_op_name(ggml_get_unary_op(t));
        for (int s=0;s<GGML_MAX_SRC;++s) if(t->src[s]) { detail << " src" << s << "=" << ggml_type_name(t->src[s]->type) << "["; for(int d=0;d<4;++d) detail << t->src[s]->ne[d] << ","; detail << "]"; }
        fail(detail.str());
    }
    Buffer buf{ggml_backend_alloc_ctx_tensors(ctx.p,cuda.p)}; if(!buf.p) fail("backend allocation failed");
    put(h,xf.data()); put(dy,dyf.data()); put(in_norm,iwbf.data()); put(post_norm,pwbf.data()); put(q_norm,qnbf.data()); put(k_norm,knbf.data()); put(cosine,trigc.data()); put(sine,trigs.data()); put(mask,maskh.data());
    if (prefix) {
        auto fill_prefix = [&](ggml_tensor * t, const char * file) {
            auto data=load<uint16_t>(fixture,file,size_t(D)*prefix*NKV);
            std::vector<float> canvas(size_t(D)*(prefix+S)*NKV,0.0f);
            for(int head=0;head<NKV;++head) for(int i=0;i<D*prefix;++i) canvas[size_t(head)*D*(prefix+S)+i]=bf16f(data[size_t(head)*D*prefix+i]);
            put(t,canvas.data());
        };
        fill_prefix(pk,"prefix_k.bin"); fill_prefix(pv,"prefix_v.bin");
    }
    fill_linear(w.qkv,"qkv_qdata.bin","qkv_scales.bin",Q+2*KV,H); fill_linear(w.output,"output_qdata.bin","output_scales.bin",H,Q); fill_linear(w.gate_up,"gate_up_qdata.bin","gate_up_scales.bin",2*F,H); fill_linear(w.down,"down_qdata.bin","down_scales.bin",H,F);
    fill_lora(ad.qkv,"qkv_lora_A.bin","qkv_lora_B.bin"); fill_lora(ad.output,"output_lora_A.bin","output_lora_B.bin"); fill_lora(ad.gate_up,"gate_up_lora_A.bin","gate_up_lora_B.bin"); fill_lora(ad.down,"down_lora_A.bin","down_lora_B.bin");
    ggml_graph_reset(graph);
    if(ggml_backend_graph_compute(cuda.p,graph)!=GGML_STATUS_SUCCESS) fail("CUDA graph compute failed");
    std::vector<float> got(H*S), gotdx(H*S), gotqa(R*H), gotqb((Q+2*KV)*R), gotoa(R*Q), gotob(H*R), gotga(R*H), gotgb(2*F*R), gotda(R*F), gotdb(H*R);
    ggml_backend_tensor_get(r.hidden,got.data(),0,ggml_nbytes(r.hidden)); auto gx=ggml_graph_get_grad(graph,h); if(!gx) fail("missing dx"); ggml_backend_tensor_get(gx,gotdx.data(),0,ggml_nbytes(gx));
    auto grad=[&](ggml_tensor*t,std::vector<float>&v){auto g=ggml_graph_get_grad(graph,t);if(!g)fail("missing LoRA gradient");ggml_backend_tensor_get(g,v.data(),0,ggml_nbytes(g));}; grad(ad.qkv.a,gotqa);grad(ad.qkv.b,gotqb);grad(ad.output.a,gotoa);grad(ad.output.b,gotob);grad(ad.gate_up.a,gotga);grad(ad.gate_up.b,gotgb);grad(ad.down.a,gotda);grad(ad.down.b,gotdb);
    auto ref=[&](const char*n,size_t z){auto b=load<uint16_t>(fixture,n,z);std::vector<float>v(z);for(size_t i=0;i<z;++i)v[i]=bf16f(b[i]);return v;};
    const auto rf=ref("forward.bin",H*S), rdx=ref("dx.bin",H*S); const auto ma=compare(got,rf), md=compare(gotdx,rdx);
    const auto mqa=compare(gotqa,load<float>(fixture,"qkv_dA.bin",gotqa.size()));
    const auto mqb=compare(gotqb,load<float>(fixture,"qkv_dB.bin",gotqb.size()));
    const auto moa=compare(gotoa,load<float>(fixture,"output_dA.bin",gotoa.size()));
    const auto mob=compare(gotob,load<float>(fixture,"output_dB.bin",gotob.size()));
    const auto mga=compare(gotga,load<float>(fixture,"gate_up_dA.bin",gotga.size()));
    const auto mgb=compare(gotgb,load<float>(fixture,"gate_up_dB.bin",gotgb.size()));
    const auto mda=compare(gotda,load<float>(fixture,"down_dA.bin",gotda.size()));
    const auto mdb=compare(gotdb,load<float>(fixture,"down_dB.bin",gotdb.size()));
    write_f32(out/"hidden.f32",got);write_f32(out/"dx.f32",gotdx);write_f32(out/"qkv_dA.f32",gotqa);write_f32(out/"qkv_dB.f32",gotqb);write_f32(out/"output_dA.f32",gotoa);write_f32(out/"output_dB.f32",gotob);write_f32(out/"gate_up_dA.f32",gotga);write_f32(out/"gate_up_dB.f32",gotgb);write_f32(out/"down_dA.f32",gotda);write_f32(out/"down_dB.f32",gotdb);
    Yue2AitkBlockForwardHost ef; Yue2AitkBlockBackwardHost eb; std::string why;
    const auto mode=prefix?Yue2AitkMaskMode::noncausal:Yue2AitkMaskMode::causal;
    if (!Yue2AitkBlockExecutor::forward(cuda.p,c,w,norms,&ad,xf.data(),S,trigc.data(),trigs.data(),mode,&ef,pk,pv,prefix,&why)) fail("executor forward: "+why);
    if (!Yue2AitkBlockExecutor::backward(cuda.p,c,w,norms,&ad,xf.data(),dyf.data(),S,trigc.data(),trigs.data(),mode,&eb,pk,pv,prefix,&why)) fail("executor backward: "+why);
    if (prefix && (ggml_graph_get_grad(graph,pk) || ggml_graph_get_grad(graph,pv))) fail("detached prefix acquired a gradient");
    if (ef.hidden!=got || eb.dx!=gotdx || eb.qkv_dA!=gotqa || eb.qkv_dB!=gotqb || eb.output_dA!=gotoa || eb.output_dB!=gotob || eb.gate_up_dA!=gotga || eb.gate_up_dB!=gotgb || eb.down_dA!=gotda || eb.down_dB!=gotdb) fail("executor differs from direct block graph");
    std::cout << "executor forward/backward agrees exactly with direct graph\n";
    std::ofstream j(out/"metrics.json"); j<<"{\"backend\":\"cuda\",\"nodes\":"<<node_count<<",\"hidden_max_abs\":"<<std::setprecision(17)<<ma.max_abs<<",\"hidden_l2\":"<<ma.l2<<",\"hidden_cosine\":"<<ma.cosine<<",\"dx_max_abs\":"<<md.max_abs<<",\"dx_l2\":"<<md.l2<<",\"dx_cosine\":"<<md.cosine<<",\"qkv_dA_max_abs\":"<<mqa.max_abs<<",\"qkv_dB_max_abs\":"<<mqb.max_abs<<",\"output_dA_max_abs\":"<<moa.max_abs<<",\"output_dB_max_abs\":"<<mob.max_abs<<",\"gate_up_dA_max_abs\":"<<mga.max_abs<<",\"gate_up_dB_max_abs\":"<<mgb.max_abs<<",\"down_dA_max_abs\":"<<mda.max_abs<<",\"down_dB_max_abs\":"<<mdb.max_abs<<",\"hidden_max_index\":"<<ma.max_i<<",\"dx_max_index\":"<<md.max_i<<"}\n";
    return 0;
#endif
}
}

int main(int argc,char**argv){try{if(argc!=3 && argc!=4){std::cerr<<"usage: yue2_aitk_block_probe FIXTURE_DIR NEW_OUTPUT_DIR [PREFIX_LENGTH]\n";return 2;}return run(argv[1],argv[2],argc==4?std::stoi(argv[3]):0);}catch(const std::exception&e){std::cerr<<"yue2_aitk_block_probe: "<<e.what()<<"\n";return 1;}}
