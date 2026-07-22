// sa3-same-ggml.h: Stable Audio 3 SAME-L autoencoder (encoder + decoder) via ggml
//
// Ports the SA3 pretransform autoencoder (stabilityai/SAME-L): a transformer-based
// audio autoencoder. Weights: engine/models/sa3-same-enc-F16.gguf (arch
// "sa3-same-enc", tensors "encoder.*" + "bottleneck.*") and
// sa3-same-dec-F16.gguf (arch "sa3-same-dec", "decoder.*" + "bottleneck.*").
// F16 (not BF16) on purpose: the decoder's sinusoidal FF blocks amplify weight
// rounding noise layer over layer (bf16 lands at parity cosine ~0.9987 on the
// decoder; f16 ~0.99998). Config: GGUF metadata "sa3.config_json" ->
// model.pretransform.config.
//
// Both directions process exactly ONE 128-latent chunk (524288 samples);
// chunking/overlap-trim happens outside, exactly like the ONNX graphs exported
// by tools/onnx-export/export_sa3_same.py (which wrapped ae.encode/ae.decode).
// All stochastic paths (bottleneck noise_regularize, resampler mask_noise) are
// disabled = the deterministic paths the goldens were dumped with.
//
// Architecture (verified against stable_audio_3/models/*.py):
//   Encoder (AudioAutoencoder.encode, autoencoders.py L405-449):
//     1. PatchedPretransform.encode (pretransforms.py L72-77): pure reshape
//        "b c (l h) -> b (c h) l" with patch_size h=256 -> (b, 512, 2048).
//        No oversampling / postfilter for this config (config has only
//        {patch_size:256, channels:2}). Done CPU-side here.
//     2. SAMEEncoder (autoencoders.py L225-288): ONE TransformerResamplingBlock
//        (in 512, out 1536, stride 16, depth 12) then Linear(1536 -> 256).
//     3. SoftNormBottleneck.encode (bottleneck.py L22-48):
//        x = (x * scaling_factor + bias) / running_std   (auto_scale, freeze).
//   Decoder (AudioAutoencoder.decode, L451-494): bottleneck.decode =
//     x * running_std (L50-52; noise_regularize path zeroed), then SAMEDecoder
//     (L290-348): Linear(256 -> 1536), ONE TransformerResamplingBlock
//     (in 1536, out 512, stride 16, type decoder, depth 12, sinusoidal_blocks 8),
//     then PatchedPretransform.decode reshape "b (c h) l -> b c (l h)" (L78-84).
//
// TransformerResamplingBlock (autoencoders.py L34-222), sliding_window=[1,1]:
//   - input_seg_size/output_seg_size/sub_chunk_size (L90-96): encoder 16/1/17,
//     decoder 1/16/17. sliding_window_seq = [win*(stride+1)] = [17,17] (L84-88).
//   - _zero_pad_modulo_sequence to input_seg_size (L125, L138): no-op at
//     2048 frames / 128 latents (verified: 2048%16==0, 128%1==0).
//   - mapping = weight_norm Conv1d kernel 1 with bias (L50; conv_mapping false,
//     mapping_bias true). Encoder maps BEFORE the transformer, decoder AFTER
//     (L126, L218). Weight norm fused at load: w = g * v / ||v||_rows.
//   - fold 'b (n c) d -> (b n) c d' with c=input_seg_size, append new_tokens
//     (variable_stride=true: parameter is (1,1,dim), expanded to output_seg_size
//     along seq, L75 + L140-141), cat on seq (L148), unfold back to
//     'b (n c) d' (L153) -> one sequence of 128*(16+1) = 2176 tokens.
//   - transformer layers with self_attention_flash_sliding_window=[17,17]
//     (L202-206). The exported/golden path (flash + flex unavailable) is
//     chunked-halo / masked SDPA (transformer.py L118-159, L678-697): band mask
//     delta = kv - q, keep -17 <= delta <= 17 (symmetric, INCLUSIVE), softmax
//     scale 1/sqrt(64). Implemented as a manual F32 attention with that mask.
//   - extract: fold to sub_chunks of 17, keep LAST output_seg_size tokens
//     (encoder 1, decoder 16), 'b d (n c)' (L214-216).
//
// TransformerBlock (transformer.py L859-1068), per layer (no global cond):
//   x = x + to_out(attn(dyt_pre(x)));  x = x + ff(dyt_ff(x))     (L1052, L1065)
//   - norm_type 'dyt' = DynamicTanh (L325-334): gamma * tanh(alpha*x) + beta,
//     alpha a learned SCALAR (eps kwarg ignored by DynamicTanh).
//   - attention (L523-820), differential=true: to_qkv = Linear(dim, 5*dim,
//     no bias), chunk order q,k,v,q_diff,k_diff (L738); qk_norm 'dyt' =
//     DynamicTanh(64) on head dim, SAME q_norm applied to q and q_diff, k_norm
//     to k and k_diff (L740-741 stack + L751); RoPE AFTER qk-norm (L755-781).
//     Differential output = attn(q,k,v) - attn(q_diff,k_diff,v) (L790-795):
//     plain subtraction, shared v, no lambda/groupnorm in this implementation.
//     to_out = Linear(dim, dim, no bias) (zero-INITIALIZED, trained weights).
//   - RoPE: RotaryEmbedding(dim_heads//2 = 32) (L976) -> PARTIAL rotary over
//     the first 32 of 64 dims (apply_rotary_pos_emb L302-322, rot_dim=32).
//     rotate_half = half-split pairing j <-> j+16 within the 32 (L296-299),
//     inv_freq_j = 10000^(-j/16) (L255) == ggml NEOX rope with n_dims=32.
//   - ff mult 3 (inner 4608), GLU (L421-451): proj = Linear(dim, 2*inner,
//     bias=True), split x|gate, x * act(gate); act = SiLU, or sin(pi*x)
//     (Sin, L446-451) when sinusoidal; then Linear(inner, dim, bias=True).
//     Decoder sinusoidal blocks: (depth - i) < sinusoidal_blocks
//     (autoencoders.py L61) -> with depth 12, sinusoidal_blocks 8: layers 5..11.

#pragma once
#include <string>  // must precede backend.h (uses std::wstring on Windows)

#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "yyjson.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define SA3_SAME_MAX_LAYERS 16

struct SA3SameConfig {
    int   dim;          // 1536 (c_mults[0] * channels)
    int   io_dim;       // 512 (patched channels = 2 * patch_size)
    int   latent_dim;   // 256
    int   stride;       // 16
    int   depth;        // 12 transformer layers
    int   dim_heads;    // 64
    int   n_heads;      // 24
    int   patch_size;   // 256
    int   win_left;     // 17 = sliding_window[0] * (stride + 1)
    int   win_right;    // 17
    int   sinusoidal_blocks;  // decoder only: 8
    float rope_theta;   // 10000
};

struct SA3SameLayer {
    // DynamicTanh norms: scalar alpha read at load, gamma/beta [dim] f32
    float                pre_alpha, ffn_alpha, qn_alpha, kn_alpha;
    struct ggml_tensor * pre_gamma;
    struct ggml_tensor * pre_beta;
    struct ggml_tensor * ffn_gamma;
    struct ggml_tensor * ffn_beta;
    struct ggml_tensor * qn_gamma;  // [64]
    struct ggml_tensor * qn_beta;
    struct ggml_tensor * kn_gamma;
    struct ggml_tensor * kn_beta;

    struct ggml_tensor * to_qkv;  // [dim, 5*dim] bf16
    struct ggml_tensor * to_out;  // [dim, dim]

    struct ggml_tensor * ff0_w;  // [dim, 2*inner]
    struct ggml_tensor * ff0_b;  // [2*inner] f32
    struct ggml_tensor * ff2_w;  // [inner, dim]
    struct ggml_tensor * ff2_b;  // [dim] f32

    bool sinusoidal;  // sin(pi*x) gate activation instead of SiLU
};

struct SA3Same {
    bool          is_encoder;
    SA3SameConfig cfg;
    SA3SameLayer  layers[SA3_SAME_MAX_LAYERS];

    struct ggml_tensor * mapping_w;   // fused WN conv1x1, f32: enc [512,1536], dec [1536,512]
    struct ggml_tensor * mapping_b;   // f32 [out]
    struct ggml_tensor * new_tokens;  // f32 [dim]
    struct ggml_tensor * proj_w;      // enc: layers.2 [1536,256]; dec: layers.1 [256,1536]
    struct ggml_tensor * proj_b;      // f32
    struct ggml_tensor * bn_mul;      // encoder only: scaling_factor / running_std, f32 [256]
    struct ggml_tensor * bn_add;      // encoder only: bias / running_std, f32 [256]
    float                running_std; // decoder: latents * running_std

    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx            wctx;

    // Debug: if >= 0, also dump the token sequence [dim, S] after this many
    // transformer layers (0 = the folded input incl. new_tokens) into
    // debug_out (token-major, matches torch (b, n, d) contiguous).
    int                debug_stage = -1;
    std::vector<float> debug_out;
};

// ── load helpers ────────────────────────────────────────────────────────────

static inline float sa3_same_read_elem(const void * data, size_t idx, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return ((const float *) data)[idx];
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(((const ggml_bf16_t *) data)[idx]);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(((const ggml_fp16_t *) data)[idx]);
        default:
            fprintf(stderr, "[SA3-SAME] FATAL: unsupported tensor type %d\n", (int) type);
            exit(1);
    }
}

// Read a whole GGUF tensor into f32 (CPU vector), any of f32/bf16/f16.
static bool sa3_same_read_f32(const GGUFModel & gf, const std::string & name, std::vector<float> & out) {
    struct ggml_tensor * meta = ggml_get_tensor(gf.meta, name.c_str());
    const void *         data = gf_get_data(gf, name.c_str());
    if (!meta || !data) {
        return false;
    }
    size_t n = (size_t) ggml_nelements(meta);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        out[i] = sa3_same_read_elem(data, i, meta->type);
    }
    return true;
}

// Read a scalar (1-element) tensor.
static float sa3_same_read_scalar(const GGUFModel & gf, const std::string & name) {
    std::vector<float> v;
    if (!sa3_same_read_f32(gf, name, v) || v.empty()) {
        fprintf(stderr, "[SA3-SAME] FATAL: scalar tensor '%s' not found\n", name.c_str());
        exit(1);
    }
    return v[0];
}

// Stage a CPU-built f32 vector as a weight tensor [ne0, ne1].
static struct ggml_tensor * sa3_same_stage_f32(WeightCtx * wctx, const char * name, int64_t ne0, int64_t ne1,
                                               const std::vector<float> & data) {
    GGML_ASSERT((int64_t) data.size() == ne0 * ne1);
    struct ggml_tensor * t = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    auto buf = std::make_unique<float[]>(data.size());
    memcpy(buf.get(), data.data(), data.size() * sizeof(float));
    wctx->pending.push_back({ t, buf.get(), data.size() * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return t;
}

// Fuse torch weight_norm for a kernel-1 Conv1d: w = g * v / ||v|| where the
// norm is per output channel over (IC, K). GGUF: weight_v ggml [1, IC, OC],
// weight_g [1, 1, OC]. Result: f32 [IC, OC] ready for ggml_mul_mat (same
// math as engine/src/vae.h vae_fuse_wn; torch adds no epsilon, 1e-12 here
// is negligible).
static struct ggml_tensor * sa3_same_load_wn_conv1x1(WeightCtx * wctx, const GGUFModel & gf,
                                                     const std::string & pfx) {
    struct ggml_tensor * vm = ggml_get_tensor(gf.meta, (pfx + ".weight_v").c_str());
    struct ggml_tensor * gm = ggml_get_tensor(gf.meta, (pfx + ".weight_g").c_str());
    const void *         vd = gf_get_data(gf, (pfx + ".weight_v").c_str());
    const void *         gd = gf_get_data(gf, (pfx + ".weight_g").c_str());
    if (!vm || !gm || !vd || !gd || vm->ne[0] != 1) {
        fprintf(stderr, "[SA3-SAME] FATAL: bad weight_norm tensors for '%s'\n", pfx.c_str());
        exit(1);
    }
    int64_t IC = vm->ne[1];
    int64_t OC = vm->ne[2];

    std::vector<float> w((size_t) IC * OC);
    for (int64_t oc = 0; oc < OC; oc++) {
        float g   = sa3_same_read_elem(gd, (size_t) oc, gm->type);
        float nsq = 0.0f;
        for (int64_t ic = 0; ic < IC; ic++) {
            float v = sa3_same_read_elem(vd, (size_t) (oc * IC + ic), vm->type);
            nsq += v * v;
        }
        float s = g / (sqrtf(nsq) + 1e-12f);
        for (int64_t ic = 0; ic < IC; ic++) {
            float v                       = sa3_same_read_elem(vd, (size_t) (oc * IC + ic), vm->type);
            w[(size_t) (oc * IC + ic)] = v * s;
        }
    }
    return sa3_same_stage_f32(wctx, (pfx + ".weight").c_str(), IC, OC, w);
}

// ── config ──────────────────────────────────────────────────────────────────

// Parse model.pretransform.config from the full SA3 model_config.json stored
// in GGUF metadata "sa3.config_json".
static void sa3_same_parse_config(SA3SameConfig * c, const char * config_json, bool is_encoder) {
    // Defaults: SAME-L
    c->dim               = 1536;
    c->io_dim            = 512;
    c->latent_dim        = 256;
    c->stride            = 16;
    c->depth             = 12;
    c->dim_heads         = 64;
    c->patch_size        = 256;
    c->sinusoidal_blocks = is_encoder ? 0 : 8;
    c->rope_theta        = 10000.0f;
    int win[2]           = { 1, 1 };

    yyjson_doc * doc = NULL;
    if (config_json && config_json[0]) {
        doc = yyjson_read(config_json, strlen(config_json), 0);
    }
    if (doc) {
        yyjson_val * root = yyjson_doc_get_root(doc);
        yyjson_val * model = root ? yyjson_obj_get(root, "model") : NULL;
        yyjson_val * pt   = model ? yyjson_obj_get(model, "pretransform") : NULL;
        yyjson_val * ae   = pt ? yyjson_obj_get(pt, "config") : NULL;
        // AE-only configs put encoder/decoder at model top level
        if (!ae || !yyjson_obj_get(ae, "encoder")) {
            if (model && yyjson_obj_get(model, "encoder")) {
                ae = model;
            }
        }
        if (ae) {
            yyjson_val * side = yyjson_obj_get(ae, is_encoder ? "encoder" : "decoder");
            yyjson_val * sc   = side ? yyjson_obj_get(side, "config") : NULL;
            if (sc) {
                yyjson_val * v;
                int channels = 256, c_mult = 6;
                if ((v = yyjson_obj_get(sc, "channels")) && yyjson_is_int(v)) {
                    channels = (int) yyjson_get_int(v);
                }
                if ((v = yyjson_obj_get(sc, "c_mults")) && yyjson_is_arr(v) && yyjson_arr_size(v) == 1) {
                    c_mult = (int) yyjson_get_int(yyjson_arr_get(v, 0));
                }
                c->dim = channels * c_mult;
                if ((v = yyjson_obj_get(sc, is_encoder ? "in_channels" : "out_channels")) && yyjson_is_int(v)) {
                    c->io_dim = (int) yyjson_get_int(v);
                }
                if ((v = yyjson_obj_get(sc, "latent_dim")) && yyjson_is_int(v)) {
                    c->latent_dim = (int) yyjson_get_int(v);
                }
                if ((v = yyjson_obj_get(sc, "strides")) && yyjson_is_arr(v) && yyjson_arr_size(v) == 1) {
                    c->stride = (int) yyjson_get_int(yyjson_arr_get(v, 0));
                }
                if ((v = yyjson_obj_get(sc, "transformer_depths")) && yyjson_is_arr(v) && yyjson_arr_size(v) == 1) {
                    c->depth = (int) yyjson_get_int(yyjson_arr_get(v, 0));
                }
                if ((v = yyjson_obj_get(sc, "dim_heads")) && yyjson_is_int(v)) {
                    c->dim_heads = (int) yyjson_get_int(v);
                }
                if ((v = yyjson_obj_get(sc, "sinusoidal_blocks")) && yyjson_is_arr(v) && yyjson_arr_size(v) == 1) {
                    c->sinusoidal_blocks = (int) yyjson_get_int(yyjson_arr_get(v, 0));
                }
                if ((v = yyjson_obj_get(sc, "sliding_window")) && yyjson_is_arr(v) && yyjson_arr_size(v) == 2) {
                    win[0] = (int) yyjson_get_int(yyjson_arr_get(v, 0));
                    win[1] = (int) yyjson_get_int(yyjson_arr_get(v, 1));
                }
            }
            yyjson_val * ptc = yyjson_obj_get(ae, "pretransform");
            ptc              = ptc ? yyjson_obj_get(ptc, "config") : NULL;
            yyjson_val * ps  = ptc ? yyjson_obj_get(ptc, "patch_size") : NULL;
            if (ps && yyjson_is_int(ps)) {
                c->patch_size = (int) yyjson_get_int(ps);
            }
        } else {
            fprintf(stderr, "[SA3-SAME] WARNING: config json missing pretransform config, using SAME-L defaults\n");
        }
        yyjson_doc_free(doc);
    } else {
        fprintf(stderr, "[SA3-SAME] WARNING: no config json, using SAME-L defaults\n");
    }

    c->n_heads   = c->dim / c->dim_heads;
    // autoencoders.py L84-88: window sizes in TOKENS of the folded sequence
    c->win_left  = win[0] * (c->stride + 1);
    c->win_right = win[1] * (c->stride + 1);
}

// ── loading ─────────────────────────────────────────────────────────────────

static bool sa3_same_load(SA3Same * m, const char * gguf_path, bool is_encoder) {
    m->is_encoder  = is_encoder;
    BackendPair bp = backend_init(is_encoder ? "SA3-SAME-Enc" : "SA3-SAME-Dec");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 8192);

    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[SA3-SAME] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    sa3_same_parse_config(&m->cfg, gf_get_str(gf, "sa3.config_json"), is_encoder);
    const SA3SameConfig & c = m->cfg;
    if (c.depth > SA3_SAME_MAX_LAYERS) {
        fprintf(stderr, "[SA3-SAME] FATAL: %d layers > max %d\n", c.depth, SA3_SAME_MAX_LAYERS);
        gf_close(&gf);
        return false;
    }

    wctx_init(&m->wctx, 16 + c.depth * 14);

    const std::string block = is_encoder ? "encoder.layers.0" : "decoder.layers.3";
    const std::string proj  = is_encoder ? "encoder.layers.2" : "decoder.layers.1";

    m->mapping_w = sa3_same_load_wn_conv1x1(&m->wctx, gf, block + ".mapping");
    m->mapping_b = gf_load_tensor_f32(&m->wctx, gf, block + ".mapping.bias");

    // new_tokens: torch (1, 1, dim) -> flat [dim] f32
    {
        std::vector<float> nt;
        if (!sa3_same_read_f32(gf, block + ".new_tokens", nt) || (int) nt.size() != c.dim) {
            fprintf(stderr, "[SA3-SAME] FATAL: bad new_tokens\n");
            gf_close(&gf);
            return false;
        }
        m->new_tokens = sa3_same_stage_f32(&m->wctx, "new_tokens", c.dim, 1, nt);
    }

    m->proj_w = gf_load_tensor_f32(&m->wctx, gf, proj + ".weight");
    m->proj_b = gf_load_tensor_f32(&m->wctx, gf, proj + ".bias");

    // Bottleneck (bottleneck.py): encode (x*sf + b)/std, decode x*std.
    m->running_std = sa3_same_read_scalar(gf, "bottleneck.running_std");
    if (is_encoder) {
        std::vector<float> sf, bb;
        if (!sa3_same_read_f32(gf, "bottleneck.scaling_factor", sf) ||
            !sa3_same_read_f32(gf, "bottleneck.bias", bb) || (int) sf.size() != c.latent_dim ||
            (int) bb.size() != c.latent_dim) {
            fprintf(stderr, "[SA3-SAME] FATAL: bad bottleneck tensors\n");
            gf_close(&gf);
            return false;
        }
        for (int i = 0; i < c.latent_dim; i++) {
            sf[(size_t) i] /= m->running_std;
            bb[(size_t) i] /= m->running_std;
        }
        m->bn_mul = sa3_same_stage_f32(&m->wctx, "bn_mul", c.latent_dim, 1, sf);
        m->bn_add = sa3_same_stage_f32(&m->wctx, "bn_add", c.latent_dim, 1, bb);
    } else {
        m->bn_mul = m->bn_add = NULL;
    }

    for (int i = 0; i < c.depth; i++) {
        char pfx[96];
        snprintf(pfx, sizeof(pfx), "%s.transformers.%d", block.c_str(), i);
        std::string    p  = pfx;
        SA3SameLayer * ly = &m->layers[i];

        ly->pre_alpha = sa3_same_read_scalar(gf, p + ".pre_norm.alpha");
        ly->pre_gamma = gf_load_tensor_f32(&m->wctx, gf, p + ".pre_norm.gamma");
        ly->pre_beta  = gf_load_tensor_f32(&m->wctx, gf, p + ".pre_norm.beta");
        ly->ffn_alpha = sa3_same_read_scalar(gf, p + ".ff_norm.alpha");
        ly->ffn_gamma = gf_load_tensor_f32(&m->wctx, gf, p + ".ff_norm.gamma");
        ly->ffn_beta  = gf_load_tensor_f32(&m->wctx, gf, p + ".ff_norm.beta");
        ly->qn_alpha  = sa3_same_read_scalar(gf, p + ".self_attn.q_norm.alpha");
        ly->qn_gamma  = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.q_norm.gamma");
        ly->qn_beta   = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.q_norm.beta");
        ly->kn_alpha  = sa3_same_read_scalar(gf, p + ".self_attn.k_norm.alpha");
        ly->kn_gamma  = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.k_norm.gamma");
        ly->kn_beta   = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.k_norm.beta");

        // Projections are expanded to F32 at load: with 16-bit weights the
        // backend runs 16-bit GEMMs (activations rounded per matmul), and the
        // resulting per-layer noise measurably drifts through the decoder's
        // 12 layers (sinusoidal FF amplification) — parity-first module.
        ly->to_qkv = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.to_qkv.weight");
        ly->to_out = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.to_out.weight");

        ly->ff0_w = gf_load_tensor_f32(&m->wctx, gf, p + ".ff.ff.0.proj.weight");
        ly->ff0_b = gf_load_tensor_f32(&m->wctx, gf, p + ".ff.ff.0.proj.bias");
        ly->ff2_w = gf_load_tensor_f32(&m->wctx, gf, p + ".ff.ff.2.weight");
        ly->ff2_b = gf_load_tensor_f32(&m->wctx, gf, p + ".ff.ff.2.bias");

        // autoencoders.py L61: sinusoidal iff (depth - i) < sinusoidal_blocks
        ly->sinusoidal = (c.depth - i) < c.sinusoidal_blocks;
    }

    fprintf(stderr, "[Load] SA3-SAME-%s: dim=%d, %dL, heads=%dx%d, stride=%d, win=[%d,%d], std=%.4f\n",
            is_encoder ? "Enc" : "Dec", c.dim, c.depth, c.n_heads, c.dim_heads, c.stride, c.win_left,
            c.win_right, m->running_std);

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    return true;
}

// ── graph builders ──────────────────────────────────────────────────────────

// DynamicTanh (transformer.py L325-334): gamma * tanh(alpha * x) + beta.
// gamma/beta broadcast over ne0.
static struct ggml_tensor * sa3_same_dyt(struct ggml_context * ctx, struct ggml_tensor * x, float alpha,
                                         struct ggml_tensor * gamma, struct ggml_tensor * beta) {
    struct ggml_tensor * t = ggml_tanh(ctx, ggml_scale(ctx, x, alpha));
    return ggml_add(ctx, ggml_mul(ctx, t, gamma), beta);
}

// One attention map: q, k [D, S, H] f32 (cont) -> out [D, S, H].
// vt: [S, D, H] (cont transpose of v). mask: [S, S] f32 band (0 / -inf).
static struct ggml_tensor * sa3_same_attn_map(struct ggml_context * ctx, struct ggml_tensor * q,
                                              struct ggml_tensor * k, struct ggml_tensor * vt,
                                              struct ggml_tensor * mask, float scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);                    // [S_kv, S_q, H]
    struct ggml_tensor * probs  = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    return ggml_mul_mat(ctx, vt, probs);                                      // [D, S_q, H]
}

// One TransformerBlock on x [dim, S]. positions [S] i32, mask [S, S] f32.
static struct ggml_tensor * sa3_same_build_layer(struct ggml_context * ctx, const SA3SameConfig & c,
                                                 SA3SameLayer * ly, struct ggml_tensor * x,
                                                 struct ggml_tensor * positions, struct ggml_tensor * mask,
                                                 int64_t S) {
    const int   D     = c.dim_heads;
    const int   H     = c.n_heads;
    const int   dim   = c.dim;
    const float scale = 1.0f / sqrtf((float) D);  // SDPA default scale
    const int   rot   = D / 2;                    // partial rotary: 32 of 64 dims

    // ── self-attention branch ──
    struct ggml_tensor * xn  = sa3_same_dyt(ctx, x, ly->pre_alpha, ly->pre_gamma, ly->pre_beta);
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, ly->to_qkv, xn);  // [5*dim, S]

    // chunk order (transformer.py L738): q, k, v, q_diff, k_diff
    auto slice = [&](int i) {
        struct ggml_tensor * v =
            ggml_view_2d(ctx, qkv, dim, S, qkv->nb[1], (size_t) i * dim * sizeof(float));
        return ggml_reshape_3d(ctx, ggml_cont(ctx, v), D, H, S);
    };
    struct ggml_tensor * q  = slice(0);
    struct ggml_tensor * k  = slice(1);
    struct ggml_tensor * v  = slice(2);
    struct ggml_tensor * qd = slice(3);
    struct ggml_tensor * kd = slice(4);

    // qk-norm 'dyt' on the head dim, shared params for main/diff (L740-751)
    q  = sa3_same_dyt(ctx, q, ly->qn_alpha, ly->qn_gamma, ly->qn_beta);
    qd = sa3_same_dyt(ctx, qd, ly->qn_alpha, ly->qn_gamma, ly->qn_beta);
    k  = sa3_same_dyt(ctx, k, ly->kn_alpha, ly->kn_gamma, ly->kn_beta);
    kd = sa3_same_dyt(ctx, kd, ly->kn_alpha, ly->kn_gamma, ly->kn_beta);

    // RoPE after qk-norm; NEOX pairing, n_dims = 32 (partial), theta 10000
    auto rope = [&](struct ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, positions, NULL, rot, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f,
                             0.0f, 1.0f, 0.0f, 0.0f);
    };
    q  = rope(q);
    k  = rope(k);
    qd = rope(qd);
    kd = rope(kd);

    // [D, H, S] -> [D, S, H] (cont for mul_mat)
    auto heads_out = [&](struct ggml_tensor * t) {
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
    };
    q  = heads_out(q);
    k  = heads_out(k);
    qd = heads_out(qd);
    kd = heads_out(kd);

    struct ggml_tensor * vp = heads_out(v);                            // [D, S, H]
    struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vp)); // [S, D, H]

    // Differential attention (L790-795): attn(q,k,v) - attn(qd,kd,v), shared v
    struct ggml_tensor * o1  = sa3_same_attn_map(ctx, q, k, vt, mask, scale);
    struct ggml_tensor * o2  = sa3_same_attn_map(ctx, qd, kd, vt, mask, scale);
    struct ggml_tensor * out = ggml_sub(ctx, o1, o2);                  // [D, S, H]

    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));          // [D, H, S]
    out = ggml_reshape_2d(ctx, out, dim, S);
    out = ggml_mul_mat(ctx, ly->to_out, out);                          // [dim, S]
    x   = ggml_add(ctx, x, out);

    // ── feed-forward branch: GLU(mult 3), act(gate) = SiLU or sin(pi*x) ──
    const int64_t        inner = (int64_t) dim * 3;
    struct ggml_tensor * y     = sa3_same_dyt(ctx, x, ly->ffn_alpha, ly->ffn_gamma, ly->ffn_beta);
    struct ggml_tensor * p     = ggml_mul_mat(ctx, ly->ff0_w, y);      // [2*inner, S]
    p                          = ggml_add(ctx, p, ly->ff0_b);
    struct ggml_tensor * a =
        ggml_cont(ctx, ggml_view_2d(ctx, p, inner, S, p->nb[1], 0));
    struct ggml_tensor * g =
        ggml_cont(ctx, ggml_view_2d(ctx, p, inner, S, p->nb[1], (size_t) inner * sizeof(float)));
    struct ggml_tensor * act;
    if (ly->sinusoidal) {
        act = ggml_sin(ctx, ggml_scale(ctx, g, 3.14159265359f));  // Sin (transformer.py L446-451)
    } else {
        act = ggml_silu(ctx, g);
    }
    struct ggml_tensor * h = ggml_mul(ctx, a, act);                    // x * act(gate)
    h                      = ggml_mul_mat(ctx, ly->ff2_w, h);          // [dim, S]
    h                      = ggml_add(ctx, h, ly->ff2_b);
    return ggml_add(ctx, x, h);
}

// ── forward ─────────────────────────────────────────────────────────────────
//
// Encoder: input  = audio, torch layout [1, 2, n_latents*patch*stride... ] i.e.
//          [2, n_latents*4096] f32; output = latents [latent_dim, n_latents]
//          (torch (1, C, L) contiguous).
// Decoder: input = latents [latent_dim, n_latents]; output = audio
//          [2, n_latents*4096].
static void sa3_same_forward(SA3Same * m, const float * input, float * output, int n_latents) {
    const SA3SameConfig & c = m->cfg;
    const int64_t F = (int64_t) n_latents * c.stride;        // patch frames (2048)
    const int64_t S = (int64_t) n_latents * (c.stride + 1);  // folded tokens (2176)
    const int64_t dim = c.dim;
    const int     n_audio_ch = c.io_dim / c.patch_size;      // 2
    const int64_t n_samples  = F * c.patch_size;

    size_t                  ctx_size = (size_t) 16384 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);
    struct ggml_cgraph *    graph    = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Band mask (transformer.py L118-127 / L143-158): keep kv - q in
    // [-win_left, +win_right], INCLUSIVE both ends; -inf outside.
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    struct ggml_tensor * x_in;
    struct ggml_tensor * seq;  // [dim, S]

    if (m->is_encoder) {
        // patched audio, token-major: [io_dim, F]
        x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.io_dim, F);
        ggml_set_name(x_in, "audio_patched");
        ggml_set_input(x_in);

        // mapping (WNConv1d k=1) BEFORE the transformer (autoencoders.py L126)
        struct ggml_tensor * h = ggml_mul_mat(ctx, m->mapping_w, x_in);  // [dim, F]
        h                      = ggml_add(ctx, h, m->mapping_b);

        // fold into groups of stride, append 1 new token per group (L139-153)
        struct ggml_tensor * xr = ggml_reshape_2d(ctx, h, dim * c.stride, n_latents);
        struct ggml_tensor * nt = ggml_repeat_4d(ctx, m->new_tokens, dim, n_latents, 1, 1);
        seq = ggml_concat(ctx, xr, ggml_reshape_2d(ctx, nt, dim, n_latents), 0);
        seq = ggml_reshape_2d(ctx, seq, dim, S);
    } else {
        // latents, token-major: [latent_dim, n_latents]
        x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.latent_dim, n_latents);
        ggml_set_name(x_in, "latents");
        ggml_set_input(x_in);

        // bottleneck.decode: x * running_std (bottleneck.py L50-52)
        struct ggml_tensor * h = ggml_scale(ctx, x_in, m->running_std);
        // Linear(latent_dim -> dim) (decoder.layers.1)
        h = ggml_mul_mat(ctx, m->proj_w, h);  // [dim, n_latents]
        h = ggml_add(ctx, h, m->proj_b);

        // per group: 1 latent token then stride new tokens (L139-153)
        struct ggml_tensor * nt = ggml_repeat_4d(ctx, m->new_tokens, dim, (int64_t) c.stride * n_latents, 1, 1);
        nt  = ggml_reshape_2d(ctx, nt, dim * c.stride, n_latents);
        seq = ggml_concat(ctx, h, nt, 0);  // h is [dim, n_latents] == [dim*1, n_latents]
        seq = ggml_reshape_2d(ctx, seq, dim, S);
    }

    struct ggml_tensor * dbg = NULL;
    if (m->debug_stage == 0) {
        dbg = seq;
    }
    for (int i = 0; i < c.depth; i++) {
        seq = sa3_same_build_layer(ctx, c, &m->layers[i], seq, positions, mask, S);
        if (m->debug_stage == i + 1) {
            dbg = seq;
        }
    }
    if (dbg) {
        dbg = ggml_cont(ctx, dbg);
        ggml_set_name(dbg, "debug_stage");
        ggml_set_output(dbg);
    }

    // extract output segments: fold to sub_chunks of stride+1, keep the LAST
    // output_seg_size tokens (autoencoders.py L214-216)
    const int out_seg = m->is_encoder ? 1 : c.stride;
    struct ggml_tensor * folded = ggml_reshape_2d(ctx, seq, dim * (c.stride + 1), n_latents);
    struct ggml_tensor * tail   = ggml_view_2d(ctx, folded, dim * out_seg, n_latents, folded->nb[1],
                                               (size_t) dim * (c.stride + 1 - out_seg) * sizeof(float));
    struct ggml_tensor * xo     = ggml_reshape_2d(ctx, ggml_cont(ctx, tail), dim, (int64_t) out_seg * n_latents);

    struct ggml_tensor * out;
    if (m->is_encoder) {
        // Linear(dim -> latent_dim), then bottleneck.encode
        out = ggml_mul_mat(ctx, m->proj_w, xo);  // [latent_dim, n_latents]
        out = ggml_add(ctx, out, m->proj_b);
        // (x * scaling_factor + bias) / running_std, folded into bn_mul/bn_add
        out = ggml_add(ctx, ggml_mul(ctx, out, m->bn_mul), m->bn_add);
    } else {
        // mapping (WNConv1d k=1) AFTER the transformer (autoencoders.py L218)
        out = ggml_mul_mat(ctx, m->mapping_w, xo);  // [io_dim, F]
        out = ggml_add(ctx, out, m->mapping_b);
    }
    ggml_set_name(out, "output");
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);
    if (dbg) {
        ggml_build_forward_expand(graph, dbg);
    }

    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        fprintf(stderr, "[SA3-SAME] FATAL: failed to allocate graph (S=%lld)\n", (long long) S);
        exit(1);
    }

    // inputs
    {
        std::vector<int32_t> pos((size_t) S);
        for (int64_t i = 0; i < S; i++) {
            pos[(size_t) i] = (int32_t) i;
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, (size_t) S * sizeof(int32_t));
    }
    {
        std::vector<float> md((size_t) S * S);
        for (int64_t qi = 0; qi < S; qi++) {
            for (int64_t kv = 0; kv < S; kv++) {
                int64_t delta = kv - qi;
                md[(size_t) (qi * S + kv)] =
                    (delta >= -(int64_t) c.win_left && delta <= (int64_t) c.win_right) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask, md.data(), 0, (size_t) S * S * sizeof(float));
    }
    if (m->is_encoder) {
        // PatchedPretransform.encode (pretransforms.py L76): "b c (l h) -> b (c h) l"
        // token-major upload: x[l][c*patch + h] = audio[c][l*patch + h]
        std::vector<float> xp((size_t) c.io_dim * F);
        for (int64_t l = 0; l < F; l++) {
            for (int ch = 0; ch < n_audio_ch; ch++) {
                const float * src = input + (size_t) ch * n_samples + (size_t) l * c.patch_size;
                float *       dst = xp.data() + (size_t) l * c.io_dim + (size_t) ch * c.patch_size;
                memcpy(dst, src, (size_t) c.patch_size * sizeof(float));
            }
        }
        ggml_backend_tensor_set(x_in, xp.data(), 0, xp.size() * sizeof(float));
    } else {
        // latents torch (1, C, L) -> token-major [C, L]
        std::vector<float> lp((size_t) c.latent_dim * n_latents);
        for (int64_t l = 0; l < n_latents; l++) {
            for (int ch = 0; ch < c.latent_dim; ch++) {
                lp[(size_t) (l * c.latent_dim + ch)] = input[(size_t) ch * n_latents + l];
            }
        }
        ggml_backend_tensor_set(x_in, lp.data(), 0, lp.size() * sizeof(float));
    }

    ggml_backend_sched_graph_compute(m->sched, graph);

    if (dbg) {
        m->debug_out.resize((size_t) dim * S);
        ggml_backend_tensor_get(dbg, m->debug_out.data(), 0, m->debug_out.size() * sizeof(float));
    }

    if (m->is_encoder) {
        // [latent_dim, n_latents] token-major -> torch (1, C, L)
        std::vector<float> lat((size_t) c.latent_dim * n_latents);
        ggml_backend_tensor_get(out, lat.data(), 0, lat.size() * sizeof(float));
        for (int64_t l = 0; l < n_latents; l++) {
            for (int ch = 0; ch < c.latent_dim; ch++) {
                output[(size_t) ch * n_latents + l] = lat[(size_t) (l * c.latent_dim + ch)];
            }
        }
    } else {
        // [io_dim, F] token-major -> PatchedPretransform.decode reshape
        // "b (c h) l -> b c (l h)" (pretransforms.py L79)
        std::vector<float> xp((size_t) c.io_dim * F);
        ggml_backend_tensor_get(out, xp.data(), 0, xp.size() * sizeof(float));
        for (int64_t l = 0; l < F; l++) {
            for (int ch = 0; ch < n_audio_ch; ch++) {
                const float * src = xp.data() + (size_t) l * c.io_dim + (size_t) ch * c.patch_size;
                float *       dst = output + (size_t) ch * n_samples + (size_t) l * c.patch_size;
                memcpy(dst, src, (size_t) c.patch_size * sizeof(float));
            }
        }
    }

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

static void sa3_same_free(SA3Same * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
