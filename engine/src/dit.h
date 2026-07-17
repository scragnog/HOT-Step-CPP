#pragma once
// dit.h: ACE-Step DiT (Diffusion Transformer) via ggml compute graph
// Ported from Python ACE-Step-1.5 reference. Same weights, loaded from GGUF.
//
// Architecture: 24-layer transformer with AdaLN, GQA self-attn + cross-attn, SwiGLU MLP.
// Flow matching: 8 Euler steps (turbo schedule).
//
// ggml ops used: rms_norm, mul_mat, rope_ext, flash_attn_ext, swiglu_split,
//                conv_transpose_1d, add, mul, scale, view, reshape, permute.

#include "adapter-merge.h"
#include "convrot.h"
#include "hot-step-build-flags.h"
#include "adapter-runtime.h"
#include "backend.h"
#include "config-json.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "timer.h"
#include "weight-source.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

// Config (populated from GGUF metadata or config.json by dit_ggml_load)
// DiTGGMLConfig is defined in config-json.h

// Layer weights
struct DiTGGMLTembWeights {
    struct ggml_tensor * linear_1_w;   // [256, hidden]
    struct ggml_tensor * linear_1_b;   // [hidden]
    struct ggml_tensor * linear_2_w;   // [hidden, hidden]
    struct ggml_tensor * linear_2_b;   // [hidden]
    struct ggml_tensor * time_proj_w;  // [hidden, 6*hidden]
    struct ggml_tensor * time_proj_b;  // [6*hidden]

    // HOT-Step ConvRot: activation-rotation group size per linear (0 = weight
    // not rotated). See DiTGGML::convrot.
    int rot_lin1 = 0;
    int rot_lin2 = 0;
    int rot_proj = 0;
};

struct DiTGGMLLayer {
    // Self-attention
    struct ggml_tensor * self_attn_norm;  // [hidden]
    struct ggml_tensor * sa_qkv;          // [hidden, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * sa_qk;           // [hidden, (Nh+Nkv)*D] partial QK fused (or NULL)
    struct ggml_tensor * sa_q_proj;       // separate fallback (NULL when any fusion active)
    struct ggml_tensor * sa_k_proj;
    struct ggml_tensor * sa_v_proj;
    struct ggml_tensor * sa_q_norm;  // [head_dim]
    struct ggml_tensor * sa_k_norm;  // [head_dim]
    struct ggml_tensor * sa_o_proj;  // [n_heads*head_dim, hidden]

    // Cross-attention
    struct ggml_tensor * cross_attn_norm;  // [hidden]
    struct ggml_tensor * ca_qkv;           // [hidden, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * ca_q_proj;        // separate (always for cross-attn with mixed types)
    struct ggml_tensor * ca_kv;            // [hidden, 2*Nkv*D] fused KV (or NULL)
    struct ggml_tensor * ca_k_proj;
    struct ggml_tensor * ca_v_proj;
    struct ggml_tensor * ca_q_norm;  // [head_dim]
    struct ggml_tensor * ca_k_norm;  // [head_dim]
    struct ggml_tensor * ca_o_proj;  // [n_heads*head_dim, hidden]

    // MLP
    struct ggml_tensor * mlp_norm;   // [hidden]
    struct ggml_tensor * gate_up;    // [hidden, 2*intermediate] fused (or NULL)
    struct ggml_tensor * gate_proj;  // [hidden, intermediate] (fallback if types differ)
    struct ggml_tensor * up_proj;    // [hidden, intermediate] (fallback if types differ)
    struct ggml_tensor * down_proj;  // [intermediate, hidden]

    // AdaLN scale-shift table: [6*hidden] (6 rows of [hidden])
    struct ggml_tensor * scale_shift_table;  // [hidden, 6] in ggml layout

    int layer_type;                          // 0=sliding, 1=full

    // HOT-Step ConvRot: activation-rotation group size per linear site
    // (0 = weight not rotated). Fused sites (qkv/qk/kv/gate_up) require a
    // uniform group across their parts — enforced at load.
    int rot_sa    = 0;  // self_attn q/k/v (input: norm_sa)
    int rot_sa_o  = 0;  // self_attn o_proj (input: attn)
    int rot_ca_q  = 0;  // cross_attn q_proj (input: norm_ca)
    int rot_ca_kv = 0;  // cross_attn k/v_proj (input: enc)
    int rot_ca_o  = 0;  // cross_attn o_proj (input: attn)
    int rot_mlp   = 0;  // mlp gate/up (input: norm_ffn)
    int rot_down  = 0;  // mlp down_proj (input: ff)
};

// Full model
#define DIT_GGML_MAX_LAYERS 32

struct DiTGGML {
    DiTGGMLConfig cfg;

    // Timestep embeddings
    DiTGGMLTembWeights time_embed;
    DiTGGMLTembWeights time_embed_r;

    // proj_in: Conv1d(in_channels, hidden, kernel=2, stride=2)
    struct ggml_tensor * proj_in_w;  // [in_ch*P, H] pre-permuted F32
    struct ggml_tensor * proj_in_b;  // [hidden]

    // condition_embedder: Linear(encoder_H, decoder_H)
    struct ggml_tensor * cond_emb_w;  // [encoder_H, decoder_H] projects encoder to decoder space
    struct ggml_tensor * cond_emb_b;  // [decoder_H]

    // Layers
    DiTGGMLLayer layers[DIT_GGML_MAX_LAYERS];

    // Output
    struct ggml_tensor * norm_out;         // [hidden]
    struct ggml_tensor * out_scale_shift;  // [hidden, 2] in ggml layout
    struct ggml_tensor * proj_out_w;       // [H, out_ch*P] pre-permuted+transposed F32
    struct ggml_tensor * proj_out_b;       // [out_channels]

    // CFG (classifier-free guidance, used by base/sft models)
    struct ggml_tensor * null_condition_emb;  // [hidden] or NULL if not present

    // Backend
    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    bool                 use_flash_attn;

    // Weight storage
    WeightCtx wctx;

    // Pre-allocated constant for AdaLN (1+scale) fusion
    struct ggml_tensor * scalar_one;  // [1] = 1.0f, broadcast in ggml_add

    // Runtime LoRA: precomputed BF16 delta tensors applied at inference.
    // `lora` is the single summed delta set (normal multi-adapter stacking).
    DiTLoRA lora;
    // Per-section masking (regional LoRA): one DiTLoRA per adapter, kept separate
    // (not summed) so each can be gated by its own per-frame mask in the graph.
    // Non-empty only when adapter_sections is active. `lora_masks` are the
    // per-adapter [1, S, 1] mask input tensors, created per graph build and
    // uploaded by the sampler.
    std::vector<DiTLoRA>          loras;
    std::vector<struct ggml_tensor *> lora_masks;

    // ─── HOT-Step ConvRot (group-wise Hadamard-rotated int8 weights) ───
    // GGUF KV "acestep.convrot_map" ("name:group;name:group;...") marks weights
    // stored PRE-ROTATED offline (W' = W·H per input-dim group of size G, as
    // written by convert_to_quant --convrot and repacked by convert-comfy-int8.py).
    // At inference the same rotation is applied to that linear's input:
    //   y = W'·rot(x) == W·x   because H is symmetric orthogonal (H·H = I).
    // H is ConvRot's "regular" Hadamard (H4 Kronecker powers, power-of-4 sizes)
    // — NOT the Sylvester matrix, so ggml's FWHT fast path
    // (GGML_HINT_SRC0_IS_HADAMARD) must NOT be hinted; rotation runs as a plain
    // grouped mul_mat against the [G, G] matrices below.
    struct {
        bool active   = false;
        int  cond_emb = 0;                            // condition_embedder group (input: enc_hidden)
        std::map<int, struct ggml_tensor *> hmats;    // group size -> [G, G] F32
    } convrot;
};

// ─── HOT-Step ConvRot helpers ───

// Parse "tensor.name:group;tensor.name:group;..." into a name→group map.
static void dit_convrot_parse_map(const char * s, std::unordered_map<std::string, int> & out) {
    if (!s) return;
    const char * p = s;
    while (*p) {
        const char * colon = strchr(p, ':');
        if (!colon) break;
        const char * semi = strchr(colon, ';');
        std::string name(p, colon - p);
        int         g = atoi(colon + 1);
        if (!name.empty() && g > 0) out[name] = g;
        if (!semi) break;
        p = semi + 1;
    }
}

// Build the normalized ConvRot "regular" Hadamard matrix tensor for group
// size G (see convrot.h). Row-major; H is symmetric so ggml_mul_mat(H, x)
// applies exactly rotate_activation(x). Returns nullptr if G is unsupported.
static struct ggml_tensor * dit_convrot_build_h(WeightCtx * wctx, int G) {
    std::vector<float> hdata;
    if (!convrot_build_h_data(G, hdata)) return nullptr;

    size_t n   = (size_t) G * G;
    auto   buf = std::make_unique<float[]>(n);
    memcpy(buf.get(), hdata.data(), n * sizeof(float));

    struct ggml_tensor * t = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, G, G);
    char nm[32];
    snprintf(nm, sizeof(nm), "convrot_h%d", G);
    ggml_set_name(t, nm);
    wctx->pending.push_back({ t, buf.get(), n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return t;
}

// Helper: check if path ends with .gguf
static bool dit_ends_with_gguf(const char * path) {
    size_t len = strlen(path);
    return len >= 5 && strcmp(path + len - 5, ".gguf") == 0;
}

// Helper: check if path ends with .onnx OR is a directory containing .onnx files
static bool dit_ends_with_onnx(const char * path) {
    size_t len = strlen(path);
    if (len >= 5 && strcmp(path + len - 5, ".onnx") == 0) {
        return true;
    }
    // Check if path is a directory containing any .onnx file
    // (e.g. registry stores "models/onnx/dit-fp8" not "models/onnx/dit-fp8/dit_fp8.onnx")
    struct stat st;
    if (stat(path, &st) == 0 && (st.st_mode & S_IFDIR)) {
#ifdef _WIN32
        std::string pattern = std::string(path) + "\\*.onnx";
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            FindClose(h);
            return true;
        }
#else
        DIR * d = opendir(path);
        if (d) {
            struct dirent * ent;
            while ((ent = readdir(d)) != nullptr) {
                const char * name = ent->d_name;
                size_t nlen = strlen(name);
                if (nlen >= 5 && strcmp(name + nlen - 5, ".onnx") == 0) {
                    closedir(d);
                    return true;
                }
            }
            closedir(d);
        }
#endif
    }
    return false;
}

// Helper: get sidecar directory for a model path
// For .onnx files: parent directory (sidecars live alongside the ONNX file)
// For directories (ONNX subdirs or safetensors): the directory itself
static std::string dit_sidecar_dir(const char * path) {
    size_t len = strlen(path);
    if (len >= 5 && strcmp(path + len - 5, ".onnx") == 0) {
        std::string p(path);
        auto sep = p.find_last_of("/\\");
        return (sep != std::string::npos) ? p.substr(0, sep) : ".";
    }
    return std::string(path);
}

// Load timestep embedding weights
static void dit_ggml_load_temb(DiTGGMLTembWeights * w,
                               WeightCtx *          wctx,
                               const WeightSource & ws,
                               const std::string &  prefix) {
    w->linear_1_w  = ws_load_tensor(wctx, ws, prefix + ".linear_1.weight");
    w->linear_1_b  = ws_load_tensor_f32(wctx, ws, prefix + ".linear_1.bias");
    w->linear_2_w  = ws_load_tensor(wctx, ws, prefix + ".linear_2.weight");
    w->linear_2_b  = ws_load_tensor_f32(wctx, ws, prefix + ".linear_2.bias");
    w->time_proj_w = ws_load_tensor(wctx, ws, prefix + ".time_proj.weight");
    w->time_proj_b = ws_load_tensor_f32(wctx, ws, prefix + ".time_proj.bias");
}

// Load proj_in weight: [H, in_ch, P] -> pre-permuted 2D [in_ch*P, H] F32
// Eliminates runtime permute+cont in the compute graph.
static struct ggml_tensor * dit_load_proj_in_w(WeightCtx *         wctx,
                                               const WeightSource & ws,
                                               const std::string & name,
                                               int                 H,
                                               int                 in_ch,
                                               int                 P) {
    ggml_type    src_type;
    const void * raw = ws.data(name.c_str(), src_type);
    if (!raw) {
        fprintf(stderr, "[WeightSource] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }

    struct ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, in_ch * P, H);
    ggml_set_name(dst, name.c_str());

    size_t  n    = (size_t) in_ch * P * H;
    auto    buf  = std::make_unique<float[]>(n);
    float * data = buf.get();

    // src ggml [P, in_ch, H]: elem(p, ic, h) = raw[h*P*in_ch + ic*P + p]
    // dst ggml [in_ch*P, H]:  elem(j, h)     = data[h*in_ch*P + j]  where j = p*in_ch + ic
    auto cvt = [&](auto read_fn) {
        for (int h = 0; h < H; h++) {
            for (int ic = 0; ic < in_ch; ic++) {
                for (int p = 0; p < P; p++) {
                    data[h * in_ch * P + p * in_ch + ic] = read_fn(h * P * in_ch + ic * P + p);
                }
            }
        }
    };
    if (src_type == GGML_TYPE_BF16) {
        const uint16_t * s = (const uint16_t *) raw;
        cvt([&](int i) { return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &s[i]); });
    } else if (src_type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) raw;
        cvt([&](int i) { return ggml_fp16_to_fp32(s[i]); });
    } else if (src_type == GGML_TYPE_F32) {
        const float * s = (const float *) raw;
        cvt([&](int i) { return s[i]; });
    } else {
        fprintf(stderr, "[WeightSource] FATAL: unsupported type %d for '%s' in proj_in pre-permute\n", src_type, name.c_str());
        exit(1);
    }
    wctx->pending.push_back({ dst, data, n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return dst;
}

// Load proj_out weight: [H, out_ch, P] -> pre-permuted+transposed 2D [H, out_ch*P] F32
// Eliminates runtime permute+cont+transpose+cont in the compute graph.
static struct ggml_tensor * dit_load_proj_out_w(WeightCtx *          wctx,
                                                const WeightSource & ws,
                                                const std::string &  name,
                                                int                  H,
                                                int                  out_ch,
                                                int                  P) {
    ggml_type    src_type;
    const void * raw = ws.data(name.c_str(), src_type);
    if (!raw) {
        fprintf(stderr, "[WeightSource] FATAL: tensor '%s' not found\n", name.c_str());
        exit(1);
    }

    struct ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, H, out_ch * P);
    ggml_set_name(dst, name.c_str());

    size_t  n    = (size_t) out_ch * P * H;
    auto    buf  = std::make_unique<float[]>(n);
    float * data = buf.get();

    // src ggml [P, out_ch, H]: elem(p, oc, h) = raw[h*P*out_ch + oc*P + p]
    // dst ggml [H, out_ch*P]:  elem(h, j)     = data[j*H + h]  where j = p*out_ch + oc
    auto cvt = [&](auto read_fn) {
        for (int h = 0; h < H; h++) {
            for (int oc = 0; oc < out_ch; oc++) {
                for (int p = 0; p < P; p++) {
                    data[(p * out_ch + oc) * H + h] = read_fn(h * P * out_ch + oc * P + p);
                }
            }
        }
    };
    if (src_type == GGML_TYPE_BF16) {
        const uint16_t * s = (const uint16_t *) raw;
        cvt([&](int i) { return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &s[i]); });
    } else if (src_type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) raw;
        cvt([&](int i) { return ggml_fp16_to_fp32(s[i]); });
    } else if (src_type == GGML_TYPE_F32) {
        const float * s = (const float *) raw;
        cvt([&](int i) { return s[i]; });
    } else {
        fprintf(stderr, "[WeightSource] FATAL: unsupported type %d for '%s' in proj_out pre-permute\n", src_type,
                name.c_str());
        exit(1);
    }
    wctx->pending.push_back({ dst, data, n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return dst;
}

static void dit_ggml_free(DiTGGML * m); // defined below; used by load-failure cleanup

// Load full DiT model from GGUF or safetensors
static bool dit_ggml_load(DiTGGML *    m,
                          const char * path,
                          const char * adapter_path  = nullptr,
                          float        adapter_scale = 1.0f,
                          const char * rebase_source = nullptr,
                          float        rebase_beta   = 0.0f) {
    // Backend init. flash_attn_ext accumulates in F16 on CPU, causing audible
    // drift over 24 layers x 8 steps: use F32 manual attention on CPU instead.
    BackendPair bp    = backend_init("DiT");
    m->backend        = bp.backend;
    m->cpu_backend    = bp.cpu_backend;
    // Scheduler hash-set must hold n_nodes + n_leafs. Per-section masking loads N
    // separate adapters (each ~360 extra delta leaf tensors) and multiplies the
    // per-projection LoRA nodes, so scale the budget with the adapter count to
    // match the graph node budget (dit-graph.h graph_cap). Normal single/summed
    // loads keep the default 8192. Uses the intended stack size (m->loras isn't
    // populated yet at this point in the load).
    int sched_nodes = 8192;
    if (!g_hotstep_params.adapter_sections.empty() && g_hotstep_params.adapters.size() >= 2)
        sched_nodes = 8192 + (int) g_hotstep_params.adapters.size() * 4096;
    m->sched          = backend_sched_new(bp, sched_nodes);
    m->use_flash_attn = bp.has_gpu && !HOT_STEP_FA_DISABLED;

    // Detect format: .gguf → GGUF path, directory → safetensors path
    bool is_st = !dit_ends_with_gguf(path);

    GGUFModel gf = {};
    STMulti   sm = {};
    WeightSource ws;

    if (is_st) {
        if (!st_multi_open(&sm, path)) {
            fprintf(stderr, "[DiT] FATAL: cannot open safetensors in %s\n", path);
            return false;
        }
        ws.is_st = true;
        ws.sm    = &sm;
        // No name prefix needed for DiT — names match safetensors directly
    } else {
        if (!gf_load(&gf, path)) {
            fprintf(stderr, "[Load] FATAL: cannot load %s\n", path);
            return false;
        }
        ws.gf = &gf;
    }

    // config from GGUF metadata or config.json sidecar
    DiTGGMLConfig & cfg = m->cfg;
    if (is_st) {
        // Read config from config.json sidecar
        std::string cfg_path = std::string(path) + WS_SEP + "config.json";
        if (!config_json_load_dit(&cfg, cfg_path.c_str())) {
            fprintf(stderr, "[DiT] FATAL: cannot read config.json from %s\n", cfg_path.c_str());
            st_multi_close(&sm);
            return false;
        }
    } else {
        // config from GGUF metadata (all keys required)
        cfg.n_layers          = (int) gf_get_u32(gf, "acestep-dit.block_count");
        cfg.hidden_size       = (int) gf_get_u32(gf, "acestep-dit.embedding_length");
        cfg.intermediate_size = (int) gf_get_u32(gf, "acestep-dit.feed_forward_length");
        cfg.n_heads           = (int) gf_get_u32(gf, "acestep-dit.attention.head_count");
        cfg.n_kv_heads        = (int) gf_get_u32(gf, "acestep-dit.attention.head_count_kv");
        cfg.head_dim          = (int) gf_get_u32(gf, "acestep-dit.attention.key_length");
        cfg.in_channels       = (int) gf_get_u32(gf, "acestep.in_channels");
        cfg.out_channels      = (int) gf_get_u32(gf, "acestep.audio_acoustic_hidden_dim");
        cfg.patch_size        = (int) gf_get_u32(gf, "acestep.patch_size");
        cfg.sliding_window    = (int) gf_get_u32(gf, "acestep.sliding_window");
        cfg.rope_theta        = gf_get_f32(gf, "acestep-dit.rope.freq_base");
        cfg.rms_norm_eps      = gf_get_f32(gf, "acestep-dit.attention.layer_norm_rms_epsilon");
    }

    if (!cfg.n_layers || !cfg.hidden_size || !cfg.intermediate_size || !cfg.n_heads || !cfg.n_kv_heads ||
        !cfg.head_dim || !cfg.in_channels || !cfg.out_channels || !cfg.patch_size || !cfg.sliding_window ||
        cfg.rope_theta <= 0.0f || cfg.rms_norm_eps <= 0.0f) {
        fprintf(stderr, "[Load] FATAL: incomplete DiT config in %s\n", path);
        if (is_st) { st_multi_close(&sm); } else { gf_close(&gf); }
        return false;
    }

    // HOT-Step ConvRot: optional rotation map (GGUF KV; safetensors DiTs never carry it)
    std::unordered_map<std::string, int> rotmap;
    if (!is_st) {
        dit_convrot_parse_map(gf_get_str(gf, "acestep.convrot_map"), rotmap);
    }
    std::map<int, int> rot_sizes;  // distinct group sizes -> count
    for (const auto & kv : rotmap) rot_sizes[kv.second]++;

    // tensor count: temb(6*2) + proj_in(2) + cond_emb(2) + layers(19*N) + output(4) + null_cond(1) + scalar_one(1)
    int n_tensors = 6 * 2 + 2 + 2 + 19 * cfg.n_layers + 4 + 1 + 1 + (int) rot_sizes.size();
    wctx_init(&m->wctx, n_tensors);

    if (!rotmap.empty()) {
        m->convrot.active = true;
        for (const auto & gs : rot_sizes) {
            struct ggml_tensor * h = dit_convrot_build_h(&m->wctx, gs.first);
            if (!h) {
                fprintf(stderr, "[DiT] FATAL: convrot group size %d unsupported (power of 4, <= 4096 only)\n", gs.first);
                if (is_st) { st_multi_close(&sm); } else { gf_close(&gf); }
                return false;
            }
            m->convrot.hmats[gs.first] = h;
        }
        fprintf(stderr, "[DiT] ConvRot active: %zu rotated weights, %zu group size(s)\n",
                rotmap.size(), rot_sizes.size());
    }
    // group lookup for one weight name (0 = not rotated)
    auto rotg = [&rotmap](const std::string & n) -> int {
        auto it = rotmap.find(n);
        return it == rotmap.end() ? 0 : it->second;
    };
    // group lookup for a fused site: all parts must agree (offline converter
    // guarantees this; a mismatch means a hand-edited map — fail loudly)
    bool rot_uniform_ok = true;
    auto rotg_fused = [&](const std::string & a, const std::string & b, const std::string & c) -> int {
        int ga = rotg(a), gb = rotg(b);
        int gc = c.empty() ? gb : rotg(c);
        if (ga != gb || gb != gc) {
            fprintf(stderr, "[DiT] FATAL: convrot group mismatch across fused site %s (%d/%d/%d)\n",
                    a.c_str(), ga, gb, gc);
            rot_uniform_ok = false;
        }
        return ga;
    };

    // Timestep embeddings
    dit_ggml_load_temb(&m->time_embed, &m->wctx, ws, "decoder.time_embed");
    dit_ggml_load_temb(&m->time_embed_r, &m->wctx, ws, "decoder.time_embed_r");
    m->time_embed.rot_lin1   = rotg("decoder.time_embed.linear_1.weight");
    m->time_embed.rot_lin2   = rotg("decoder.time_embed.linear_2.weight");
    m->time_embed.rot_proj   = rotg("decoder.time_embed.time_proj.weight");
    m->time_embed_r.rot_lin1 = rotg("decoder.time_embed_r.linear_1.weight");
    m->time_embed_r.rot_lin2 = rotg("decoder.time_embed_r.linear_2.weight");
    m->time_embed_r.rot_proj = rotg("decoder.time_embed_r.time_proj.weight");

    // proj_in: Conv1d weight [hidden, in_ch, patch_size]
    // Pre-permuted to 2D [in_ch*P, H] F32 at load time
    m->proj_in_w =
        dit_load_proj_in_w(&m->wctx, ws, "decoder.proj_in.1.weight", cfg.hidden_size, cfg.in_channels, cfg.patch_size);
    m->proj_in_b = ws_load_tensor_f32(&m->wctx, ws, "decoder.proj_in.1.bias");

    // condition_embedder
    m->cond_emb_w = ws_load_tensor(&m->wctx, ws, "decoder.condition_embedder.weight");
    m->cond_emb_b = ws_load_tensor_f32(&m->wctx, ws, "decoder.condition_embedder.bias");
    m->convrot.cond_emb = rotg("decoder.condition_embedder.weight");

    // Layers
    for (int i = 0; i < cfg.n_layers; i++) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "decoder.layers.%d", i);
        std::string    p(prefix);
        DiTGGMLLayer & ly = m->layers[i];

        // Self-attention: try full QKV, partial QK, separate
        // HOT-Step: Runtime LoRA and merge_hq need individual projections (no fusion)
        // Runtime: deltas are applied per-projection in the compute graph
        // Merge HQ: projections are promoted to F32, incompatible with fused BF16 tensors
        bool skip_fusion = (adapter_path && 
            (g_hotstep_params.adapter_mode == "runtime" || 
             g_hotstep_params.adapter_mode == "merge"));
        ly.self_attn_norm = ws_load_tensor_f32(&m->wctx, ws, p + ".self_attn_norm.weight");
        if (!skip_fusion) {
        ly.sa_qkv = ws_load_qkv_fused(&m->wctx, ws, p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight",
                                      p + ".self_attn.v_proj.weight");
        } else {
            ly.sa_qkv = nullptr;
            if (i == 0) fprintf(stderr, "[DiT] Skipping QKV/gate_up fusion (runtime LoRA needs individual projections)\n");
        }
        if (!ly.sa_qkv) {
            // Try Q+K fusion (same input, often same type in K-quants)
            if (!skip_fusion) {
            ly.sa_qk = ws_load_pair_fused(&m->wctx, ws, p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight");
            } else {
                ly.sa_qk = nullptr;
            }
            if (ly.sa_qk) {
                ly.sa_v_proj = ws_load_tensor(&m->wctx, ws, p + ".self_attn.v_proj.weight");
                if (i == 0) {
                    fprintf(stderr, "[DiT] Self-attn: Q+K fused, V separate\n");
                }
            } else {
                ly.sa_q_proj = ws_load_tensor(&m->wctx, ws, p + ".self_attn.q_proj.weight");
                ly.sa_k_proj = ws_load_tensor(&m->wctx, ws, p + ".self_attn.k_proj.weight");
                ly.sa_v_proj = ws_load_tensor(&m->wctx, ws, p + ".self_attn.v_proj.weight");
                if (i == 0) {
                    fprintf(stderr, "[DiT] Self-attn: all separate%s\n",
                            skip_fusion ? " (runtime LoRA)" : " (3 types differ)");
                }
            }
        } else {
            if (i == 0) {
                fprintf(stderr, "[DiT] Self-attn: Q+K+V fused\n");
            }
        }
        ly.sa_q_norm = ws_load_tensor_f32(&m->wctx, ws, p + ".self_attn.q_norm.weight");
        ly.sa_k_norm = ws_load_tensor_f32(&m->wctx, ws, p + ".self_attn.k_norm.weight");
        ly.sa_o_proj = ws_load_tensor(&m->wctx, ws, p + ".self_attn.o_proj.weight");

        // Cross-attention: try full QKV, K+V fused, separate
        ly.cross_attn_norm = ws_load_tensor_f32(&m->wctx, ws, p + ".cross_attn_norm.weight");
        if (!skip_fusion) {
        ly.ca_qkv = ws_load_qkv_fused(&m->wctx, ws, p + ".cross_attn.q_proj.weight", p + ".cross_attn.k_proj.weight",
                                      p + ".cross_attn.v_proj.weight");
        } else {
            ly.ca_qkv = nullptr;
        }
        if (!ly.ca_qkv) {
            ly.ca_q_proj = ws_load_tensor(&m->wctx, ws, p + ".cross_attn.q_proj.weight");
            // Try K+V fusion (same input enc, may share type)
            if (!skip_fusion) {
            ly.ca_kv =
                ws_load_pair_fused(&m->wctx, ws, p + ".cross_attn.k_proj.weight", p + ".cross_attn.v_proj.weight");
            } else {
                ly.ca_kv = nullptr;
            }
            if (ly.ca_kv) {
                if (i == 0) {
                    fprintf(stderr, "[DiT] Cross-attn: Q separate, K+V fused\n");
                }
            } else {
                ly.ca_k_proj = ws_load_tensor(&m->wctx, ws, p + ".cross_attn.k_proj.weight");
                ly.ca_v_proj = ws_load_tensor(&m->wctx, ws, p + ".cross_attn.v_proj.weight");
                if (i == 0) {
                    fprintf(stderr, "[DiT] Cross-attn: all separate%s\n",
                            skip_fusion ? " (runtime LoRA)" : "");
                }
            }
        } else {
            if (i == 0) {
                fprintf(stderr, "[DiT] Cross-attn: Q+K+V fused\n");
            }
        }
        ly.ca_q_norm = ws_load_tensor_f32(&m->wctx, ws, p + ".cross_attn.q_norm.weight");
        ly.ca_k_norm = ws_load_tensor_f32(&m->wctx, ws, p + ".cross_attn.k_norm.weight");
        ly.ca_o_proj = ws_load_tensor(&m->wctx, ws, p + ".cross_attn.o_proj.weight");

        // MLP: try gate+up fusion (same input, same pattern as QKV)
        ly.mlp_norm = ws_load_tensor_f32(&m->wctx, ws, p + ".mlp_norm.weight");
        if (!skip_fusion) {
        ly.gate_up  = ws_load_pair_fused(&m->wctx, ws, p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight");
        } else {
            ly.gate_up = nullptr;
        }
        if (ly.gate_up) {
            if (i == 0) {
                fprintf(stderr, "[DiT] MLP: gate+up fused\n");
            }
        } else {
            ly.gate_proj = ws_load_tensor(&m->wctx, ws, p + ".mlp.gate_proj.weight");
            ly.up_proj   = ws_load_tensor(&m->wctx, ws, p + ".mlp.up_proj.weight");
            if (i == 0) {
                fprintf(stderr, "[DiT] MLP: gate+up separate%s\n",
                        skip_fusion ? " (runtime LoRA)" : " (types differ)");
            }
        }
        ly.down_proj = ws_load_tensor(&m->wctx, ws, p + ".mlp.down_proj.weight");

        // AdaLN scale_shift_table [1, 6, hidden] in GGUF
        ly.scale_shift_table = ws_load_tensor_f32(&m->wctx, ws, p + ".scale_shift_table");

        ly.layer_type = (i % 2 == 0) ? 0 : 1;  // 0=sliding, 1=full

        // ConvRot per-site groups. Fused sites need one uniform group; q/k/v and
        // gate/up share in_features so the offline converter always agrees here.
        if (m->convrot.active) {
            ly.rot_sa    = rotg_fused(p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight",
                                      p + ".self_attn.v_proj.weight");
            ly.rot_sa_o  = rotg(p + ".self_attn.o_proj.weight");
            ly.rot_ca_q  = rotg(p + ".cross_attn.q_proj.weight");
            ly.rot_ca_kv = rotg_fused(p + ".cross_attn.k_proj.weight", p + ".cross_attn.v_proj.weight", "");
            ly.rot_ca_o  = rotg(p + ".cross_attn.o_proj.weight");
            // full ca_qkv fusion additionally requires q == kv
            if (ly.ca_qkv && ly.rot_ca_q != ly.rot_ca_kv) {
                fprintf(stderr, "[DiT] FATAL: convrot group mismatch across fused ca_qkv (layer %d: %d vs %d)\n",
                        i, ly.rot_ca_q, ly.rot_ca_kv);
                rot_uniform_ok = false;
            }
            ly.rot_mlp  = rotg_fused(p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight", "");
            ly.rot_down = rotg(p + ".mlp.down_proj.weight");
        }
    }

    if (!rot_uniform_ok) {
        if (is_st) { st_multi_close(&sm); } else { gf_close(&gf); }
        return false;
    }

    // Output
    m->norm_out        = ws_load_tensor_f32(&m->wctx, ws, "decoder.norm_out.weight");
    m->out_scale_shift = ws_load_tensor_f32(&m->wctx, ws, "decoder.scale_shift_table");
    m->proj_out_w = dit_load_proj_out_w(&m->wctx, ws, "decoder.proj_out.1.weight", cfg.hidden_size, cfg.out_channels,
                                        cfg.patch_size);
    m->proj_out_b = ws_load_tensor_f32(&m->wctx, ws, "decoder.proj_out.1.bias");

    // Null condition embedding for CFG (base/sft models; turbo has it but unused at inference)
    m->null_condition_emb = ws_try_load_tensor(&m->wctx, ws, "null_condition_emb");
    if (m->null_condition_emb) {
        fprintf(stderr, "[Load] null_condition_emb found (CFG available)\n");
    }

    // Scalar constant for AdaLN (1+scale) fusion
    static const float one_val = 1.0f;
    m->scalar_one              = ggml_new_tensor_1d(m->wctx.ctx, GGML_TYPE_F32, 1);
    m->wctx.pending.push_back({ m->scalar_one, &one_val, sizeof(float), 0 });

    // Merge adapter deltas into projection weights (before GPU upload and QKV fusion)
    // HOT-Step: skip merge in runtime mode — runtime adapter loaded after wctx_alloc
    if (adapter_path) {
        bool runtime_mode = (g_hotstep_params.adapter_mode == "runtime");
        // ConvRot weights live in rotated space; merge-mode deltas are unrotated
        // and would corrupt them. Runtime mode is safe (deltas run as separate
        // matmuls on the UNROTATED activations in the graph).
        if (m->convrot.active && !runtime_mode) {
            fprintf(stderr, "[Adapter] FATAL: adapter merge mode is not supported on ConvRot models — use adapter_mode=runtime\n");
            if (is_st) { st_multi_close(&sm); } else { gf_close(&gf); }
            return false;
        }
        if (!runtime_mode) {
            // Auto-detect promote_f32: NVFP4/MXFP4 stay in native quant (saves ~13 GB
            // VRAM vs F32 promotion). GGML CUDA has full dequant kernels for these types
            // and ggml_quantize_chunk handles host requantization.
            bool promote_f32 = true;
            {
                ggml_type t = ws.type("decoder.layers.0.self_attn.q_proj.weight");
                if (t == GGML_TYPE_NVFP4 || t == GGML_TYPE_MXFP4) {
                    promote_f32 = false;
                    fprintf(stderr, "[Adapter] FP4 model detected — merge will requant to native type (no F32 promotion)\n");
                } else if (g_hotstep_params.adapter_merge_lowvram) {
                    // Opt-in "Merge (low VRAM)": store merged weights in the base's
                    // native quant instead of F32. Same tensor selection as HQ merge
                    // (adapter_hq_should_skip is widened to cover this flag).
                    promote_f32 = false;
                    fprintf(stderr, "[Adapter] Low-VRAM merge — merged weights re-encoded to native type (no F32 promotion)\n");
                }
            }
            Timer adapter_timer;
            // Multi-adapter stack: merge each adapter sequentially. adapter_merge
            // reads each tensor's CURRENT (post-prior-merge) value, so deltas
            // accumulate: W <- W + s1*d1 + s2*d2 + ... (see adapter-merge.h).
            //
            // Basin re-base is applied to the FIRST adapter only. Re-base nudges
            // the loaded base T toward the adapter's home base S before adding the
            // delta (base <- base + beta*(S - base)); at beta=1 that REPLACES the
            // base with S. Re-applying it per adapter would reset the running base
            // every merge and discard all earlier adapters (only the last would
            // survive). So we re-base T once up front, then stack every adapter's
            // delta on the nudged base. When the stack is empty, fall back to the
            // single legacy adapter path.
            const std::vector<AdapterSpec> & stack = g_hotstep_params.adapters;
            bool merge_ok = false;
            if (!stack.empty()) {
                int si = 0;
                for (const auto & a : stack) {
                    const char * rb_src  = (si == 0) ? rebase_source : nullptr;
                    float        rb_beta = (si == 0) ? rebase_beta   : 0.0f;
                    if (adapter_merge(&m->wctx, ws, a.path.c_str(), a.scale, m->backend, promote_f32, rb_src, rb_beta)) {
                        merge_ok = true;
                    } else {
                        fprintf(stderr, "[Adapter] WARNING: stack adapter %d merged no tensors: %s\n", si, a.path.c_str());
                    }
                    si++;
                }
            } else {
                merge_ok = adapter_merge(&m->wctx, ws, adapter_path, adapter_scale, m->backend, promote_f32, rebase_source, rebase_beta);
            }
            if (!merge_ok) {
                fprintf(stderr, "[Adapter] FATAL: no tensors merged (model mismatch)\n");
                if (is_st) { st_multi_close(&sm); } else { gf_close(&gf); }
                return false;
            }
            fprintf(stderr, "[Adapter] Merge time: %.1f ms (%zu adapter%s)\n", adapter_timer.ms(),
                    stack.empty() ? (size_t) 1 : stack.size(), (stack.size() == 1) ? "" : "s");
        } else {
            fprintf(stderr, "[Adapter] mode=runtime, deferring delta precompute\n");
        }
    }

    // Allocate backend buffer and copy weights
    if (!wctx_alloc(&m->wctx, m->backend)) {
        if (is_st) { st_multi_close(&sm); } else { gf_close(&gf); }
        return false;
    }

    // HOT-Step: load runtime adapter AFTER wctx_alloc (base weights on GPU first)
    if (adapter_path) {
        bool runtime_mode = (g_hotstep_params.adapter_mode == "runtime");
        if (runtime_mode) {
            Timer rt_timer;
            // Multi-adapter stack: precompute each adapter's deltas and SUM them
            // per projection into a single delta set (constant per-step cost,
            // VRAM flat regardless of stack depth). Empty stack => single adapter.
            const std::vector<AdapterSpec> & stack = g_hotstep_params.adapters;
            bool rt_ok;
            if (!g_hotstep_params.adapter_sections.empty() && stack.size() >= 2) {
                // Per-section masking: load each adapter into its OWN DiTLoRA
                // (not summed) so the graph can gate each with a per-frame mask.
                // N× VRAM vs the summed path — the price of per-section control.
                //
                // Basin re-base is NOT supported here: the nudge is an always-on
                // base correction, but every per-section delta is gated by a
                // per-frame mask — folding it in would fade the correction with
                // the section weights and duplicate it once per adapter.
                if (rebase_source && rebase_source[0] && rebase_beta != 0.0f) {
                    fprintf(stderr, "[Adapter-RT] WARNING: basin re-base is not supported with per-section masking — skipping nudge\n");
                }
                m->loras.clear();
                m->loras.resize(stack.size());
                // ALL adapters must load — a partial stack would silently generate
                // with some adapters missing (and get cached that way, see below).
                rt_ok = true;
                for (size_t i = 0; i < stack.size(); i++) {
                    // Load each adapter UNIT-scaled (scale 1.0): the per-frame
                    // section mask carries the full effective scale, so baking the
                    // stack scale into the delta too would double-scale it. Group
                    // scales still apply (they gate weight groups, not sections).
                    std::vector<AdapterSpec> one{ AdapterSpec{ stack[i].path, 1.0f } };
                    if (!adapter_load_runtime_stack(&m->loras[i], &m->wctx, ws, one,
                                                    g_hotstep_params.adapter_group_scales, m->backend)) {
                        fprintf(stderr, "[Adapter-RT] ERROR: section adapter %zu loaded no deltas: %s\n",
                                i, stack[i].path.c_str());
                        rt_ok = false;
                    }
                }
                fprintf(stderr, "[Adapter-RT] Per-section load: %zu adapters kept separate\n", m->loras.size());
            } else if (!stack.empty()) {
                rt_ok = adapter_load_runtime_stack(&m->lora, &m->wctx, ws, stack,
                                                   g_hotstep_params.adapter_group_scales, m->backend,
                                                   rebase_source, rebase_beta,
                                                   m->convrot.active ? &rotmap : nullptr);
            } else {
                rt_ok = adapter_load_runtime(&m->lora, &m->wctx, ws, adapter_path, adapter_scale,
                                             g_hotstep_params.adapter_group_scales, m->backend,
                                             rebase_source, rebase_beta,
                                             m->convrot.active ? &rotmap : nullptr);
            }
            if (!rt_ok) {
                // FAIL the load, matching the merge path. Returning success here
                // would install an adapter-LESS model in the store under the
                // adapter-bearing cache key — every later request with these
                // adapters would silently cache-hit base-model output. wctx is
                // already on the GPU at this point, so free the model's resources
                // (the store's failure path only does `delete m`).
                fprintf(stderr, "[Adapter-RT] FATAL: runtime adapter load failed\n");
                if (is_st) { st_multi_close(&sm); } else { gf_close(&gf); }
                dit_ggml_free(m);
                return false;
            }
            fprintf(stderr, "[Adapter-RT] Load time: %.1f ms (%zu adapter%s)\n", rt_timer.ms(),
                    stack.empty() ? (size_t) 1 : stack.size(), (stack.size() == 1) ? "" : "s");
        }
    }

    if (is_st) { st_multi_close(&sm); } else { gf_close(&gf); }

    fprintf(stderr, "[Load] DiT: %d layers, H=%d, Nh=%d/%d, D=%d%s\n", cfg.n_layers, cfg.hidden_size, cfg.n_heads,
            cfg.n_kv_heads, cfg.head_dim, is_st ? " (safetensors)" : " (GGUF)");
    return true;
}

static void dit_ggml_free(DiTGGML * m) {
    dit_lora_free(&m->lora);
    for (auto & lr : m->loras) {
        dit_lora_free(&lr);
    }
    m->loras.clear();
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}

// Read DiT config from GGUF metadata or config.json without loading any tensor weights.
// Used by the orchestrator to keep patch_size, in_channels, out_channels
// accessible during text encoding while the DiT itself is not yet loaded.
// Returns true on success, false on I/O or missing key.
static bool dit_ggml_load_config(DiTGGMLConfig * cfg, const char * path) {
    bool is_st = !dit_ends_with_gguf(path);

    if (is_st) {
        // For ONNX files: config.json is in the parent directory
        // For safetensors dirs: config.json is inside the directory
        std::string sidecar_dir = dit_sidecar_dir(path);
        std::string cfg_path = sidecar_dir + WS_SEP + "config.json";
        if (!config_json_load_dit(cfg, cfg_path.c_str())) {
            fprintf(stderr, "[Load] FATAL: cannot read config from %s\n", cfg_path.c_str());
            return false;
        }
    } else {
        GGUFModel gf;
        if (!gf_load(&gf, path)) {
            fprintf(stderr, "[Load] FATAL: cannot load %s\n", path);
            return false;
        }
        cfg->n_layers          = (int) gf_get_u32(gf, "acestep-dit.block_count");
        cfg->hidden_size       = (int) gf_get_u32(gf, "acestep-dit.embedding_length");
        cfg->intermediate_size = (int) gf_get_u32(gf, "acestep-dit.feed_forward_length");
        cfg->n_heads           = (int) gf_get_u32(gf, "acestep-dit.attention.head_count");
        cfg->n_kv_heads        = (int) gf_get_u32(gf, "acestep-dit.attention.head_count_kv");
        cfg->head_dim          = (int) gf_get_u32(gf, "acestep-dit.attention.key_length");
        cfg->in_channels       = (int) gf_get_u32(gf, "acestep.in_channels");
        cfg->out_channels      = (int) gf_get_u32(gf, "acestep.audio_acoustic_hidden_dim");
        cfg->patch_size        = (int) gf_get_u32(gf, "acestep.patch_size");
        cfg->sliding_window    = (int) gf_get_u32(gf, "acestep.sliding_window");
        cfg->rope_theta        = gf_get_f32(gf, "acestep-dit.rope.freq_base");
        cfg->rms_norm_eps      = gf_get_f32(gf, "acestep-dit.attention.layer_norm_rms_epsilon");
        gf_close(&gf);
    }

    if (!cfg->n_layers || !cfg->hidden_size || !cfg->intermediate_size || !cfg->n_heads || !cfg->n_kv_heads ||
        !cfg->head_dim || !cfg->in_channels || !cfg->out_channels || !cfg->patch_size || !cfg->sliding_window ||
        cfg->rope_theta <= 0.0f || cfg->rms_norm_eps <= 0.0f) {
        fprintf(stderr, "[Load] FATAL: incomplete DiT config in %s\n", path);
        return false;
    }
    return true;
}
