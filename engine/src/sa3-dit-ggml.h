// sa3-dit-ggml.h: Stable Audio 3 medium DiT (1.45B DiffusionTransformer) via ggml
//
// Ports the SA3 diffusion core: one _forward pass of DiffusionTransformer
// (stable_audio_3/models/dit.py L179-309) — exactly the graph exported by
// tools/onnx-export/export_sa3_dit.py. The rf_denoiser objective keeps ALL
// solver math outside the model: sampling.py L181/L211/L268 treat the model
// output as v with denoised = x - t*v, so this module returns the raw
// _forward output.
//
// Weights: engine/models/sa3-dit-BF16.gguf (arch "sa3-dit"). Tensor names are
// the DiffusionTransformer state_dict keys under "model." (the DiTWrapper
// prefix as stored), e.g. "model.transformer.layers.0.self_attn.to_qkv.weight",
// plus "conditioner.*" extras used by sa3-t5gemma-enc.h. Config: GGUF metadata
// "sa3.config_json" -> model.diffusion.config.
//
// Architecture (verified against stable_audio_3/models/dit.py + transformer.py
// and the medium model_config.json: io_channels 256, embed_dim 1536, depth 24,
// num_heads 24 -> dim_heads 64, cond_token_dim 768, global_cond_dim 768,
// local_add_cond_dim 257, global_cond_type "adaLN", timestep_features_type
// "expo", attn_kwargs {qk_norm: "rms", differential: true}, norm_type
// "rms_norm" (force_fp32), ff_kwargs {mult: 4.0}, num_memory_tokens 64;
// patch_size default 1 (dit.py L16), timestep_cond_type default "global"
// (dit.py L28), timestep_features_logsnr default False (dit.py L33),
// project_cond_tokens/project_global_cond default True (dit.py L19/L21)):
//
//  _forward (dit.py L179-309):
//   1. cross_attn_cond = to_cond_embed(cross): Linear(768->1536, no bias),
//      SiLU, Linear(1536->1536, no bias)                       (dit.py L70-74, L197-198)
//   2. global_embed    = to_global_embed(glob): same shape family, no bias
//                                                              (dit.py L81-85, L200-202)
//   3. local_add_cond rearranged "b c t -> b t c"              (dit.py L224-225)
//   4. timestep_embed = to_timestep_embed(timestep_features(t[:,None])):
//      ExpoFourierFeatures(256, 0.5, 10000) (dit.py L47-48; blocks.py L50-82:
//      ramp = linspace(0,1,128), freqs = exp(ramp*(ln 1e4 - ln 0.5) + ln 0.5),
//      args = t*freqs*2*pi, feat = [cos|sin]), then Linear(256->1536, bias) +
//      SiLU + Linear(1536->1536, bias)                         (dit.py L58-62, L239)
//   5. timestep_cond_type "global": global_embed += timestep_embed (dit.py L243-245)
//   6. x = preprocess_conv(x) + x, Conv1d(256,256,1,bias=False) (dit.py L133-134, L264)
//   7. rearrange b c t -> b t c; patch_size 1 => no patch fold  (dit.py L266, L273-274)
//   8. adaLN: extra_args["global_cond"] = global_embed          (dit.py L270-271)
//   9. ContinuousTransformer (transformer.py L1070-1272):
//      - project_in Linear(256->1536, no bias)                  (L1105, L1184)
//      - memory tokens (64,1536) prepended to the sequence      (L1118-1120, L1193-1195)
//        -> S = 64 + T; memory tokens participate in ALL layers and are
//        dropped before project_out                             (L1265)
//      - RoPE: RotaryEmbedding(max(64//2,32)=32)                (L1109) over
//        positions 0..S-1 incl. memory tokens (L1197-1198); partial rotary,
//        first 32 of 64 head dims, NEOX half-split pairing, theta 10000
//        (L239-322); applied AFTER qk-norm in Attention (L755-781); NOT
//        applied to cross-attention (cross_attn_rotary_pos_emb=False, L1113,
//        cross call passes no rope: L1034-1035/L1057-1058)
//      - global_cond_embedder: Linear(1536->1536,bias) + SiLU +
//        Linear(1536->9216,bias), applied ONCE per forward      (L1130-1136, L1210-1211)
//      - padding_mask extended with an always-valid prefix for the 64 memory
//        tokens (L1216-1227). In the exported no-flash/no-flex path the mask
//        is applied by V-ZEROING ONLY (L675-677): padded keys still receive
//        softmax probability, but their values are zero; then plain
//        unmasked SDPA (L699, scale 1/sqrt(64)). Cross-attn gets no mask
//        (padding_mask only passed to self_attn: L1026/L1052).
//   10. TransformerBlock x24, adaLN branch (transformer.py L1018-1049):
//      ssg = to_scale_shift_gate[9216] + global_cond, chunk 6 ->
//        scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff (L1020)
//      a) h = pre_norm(x); h = h*(1+scale_self)+shift_self; h = self_attn(h);
//         h = h*sigmoid(1 - gate_self); x = x + h              (L1023-1029)
//         pre_norm = RMSNorm(1536, force_fp32, eps 1e-5): x*rsqrt(mean+eps)*gamma
//         (L392-410; norm_kwargs only sets force_fp32, eps default 1e-5)
//         self_attn differential (L523-820): to_qkv = Linear(1536, 5*1536,
//         no bias), chunk order q,k,v,q_diff,k_diff (L738); qk_norm "rms" =
//         RMSNorm(64, eps=1e-6) on the head dim, q_norm shared by q/q_diff,
//         k_norm by k/k_diff (L740-741 stack + L751); RoPE after qk-norm;
//         out = attn(q,k,v) - attn(q_diff,k_diff,v): plain subtraction,
//         SHARED v, no lambda/groupnorm (L790-795); to_out Linear no bias.
//      b) x = x + cross_attn(cross_attend_norm(x), context)    (L1031-1035)
//         NO adaLN modulation and NO gate on the cross branch. Differential
//         cross-attn: to_q = Linear(1536, 2*1536, no bias) chunk q,q_diff
//         (L549, L724); to_kv = Linear(1536, 3*1536, no bias) chunk
//         k,k_diff,v (L550, L727); same rms qk-norm; no rope; no mask.
//      c) x = x + left_pad(to_local_embed(local_add_cond))     (L978-982, L1040)
//         to_local_embed = Linear(257->1536,bias)+SiLU+Linear(1536->1536,bias)
//         (L946-951); _left_pad_to_match zero-pads the 64 memory positions
//         at the FRONT (L77-88) so only the T data tokens are shifted.
//      d) h = ff_norm(x); h = h*(1+scale_ff)+shift_ff; h = ff(h);
//         h = h*sigmoid(1 - gate_ff); x = x + h                (L1043-1049)
//         ff = GLU (mult 4 -> inner 6144): proj Linear(1536->12288, bias),
//         chunk x|gate, x * SiLU(gate) (L421-444, L471), then
//         Linear(6144->1536, bias) (L485).
//   11. x = x[:, 64:, :]; project_out Linear(1536->256, no bias) (L1265-1267)
//   12. rearrange b t c -> b c t (prepend_length 0)             (dit.py L299)
//   13. output = postprocess_conv(output) + output              (dit.py L135-136, L304)
//
// Cross-attention runs on EVERY layer (final_cross_attn_ix default -1,
// transformer.py L1081, L1143).
//
// Batch 1 only; the graph is built per call for the requested T (the caller
// rebuilds per shape like the other engine GGML modules).

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

#define SA3_DIT_MAX_LAYERS 32

struct SA3DiTConfig {
    int   io_channels;     // 256
    int   dim;             // 1536 (embed_dim)
    int   depth;           // 24
    int   n_heads;         // 24
    int   dim_heads;       // 64
    int   cond_token_dim;  // 768 (raw prompt tokens; projected to dim)
    int   global_cond_dim; // 768
    int   local_cond_dim;  // 257
    int   n_memory_tokens; // 64
    int   patch_size;      // 1
    int   ts_feat_dim;     // 256 (timestep_features_dim)
    int   ff_inner;        // 6144 (dim * ff mult 4)
    int   rope_dims;       // 32 (partial rotary)
    float rope_theta;      // 10000
    float norm_eps;        // 1e-5 (block RMSNorms)
    float qk_norm_eps;     // 1e-6 (attention head RMSNorms)
    float ts_min_freq;     // 0.5 (ExpoFourierFeatures)
    float ts_max_freq;     // 10000
};

struct SA3DiTLayer {
    struct ggml_tensor * pre_norm_g;   // f32 [dim]
    struct ggml_tensor * ca_norm_g;    // f32 [dim] (cross_attend_norm)
    struct ggml_tensor * ff_norm_g;    // f32 [dim]
    struct ggml_tensor * ssg;          // f32 [6*dim] (to_scale_shift_gate)

    struct ggml_tensor * attn_qkv;     // [dim, 5*dim]
    struct ggml_tensor * attn_out;     // [dim, dim]
    struct ggml_tensor * attn_qn_g;    // f32 [dim_heads]
    struct ggml_tensor * attn_kn_g;    // f32 [dim_heads]

    struct ggml_tensor * cross_q;      // [dim, 2*dim]
    struct ggml_tensor * cross_kv;     // [dim, 3*dim]
    struct ggml_tensor * cross_out;    // [dim, dim]
    struct ggml_tensor * cross_qn_g;   // f32 [dim_heads]
    struct ggml_tensor * cross_kn_g;   // f32 [dim_heads]

    struct ggml_tensor * ff0_w;        // [dim, 2*inner]
    struct ggml_tensor * ff0_b;        // f32 [2*inner]
    struct ggml_tensor * ff2_w;        // [inner, dim]
    struct ggml_tensor * ff2_b;        // f32 [dim]

    struct ggml_tensor * loc0_w;       // [local_cond_dim, dim]
    struct ggml_tensor * loc0_b;       // f32 [dim]
    struct ggml_tensor * loc2_w;       // [dim, dim]
    struct ggml_tensor * loc2_b;       // f32 [dim]
};

struct SA3DiT {
    SA3DiTConfig cfg;
    SA3DiTLayer  layers[SA3_DIT_MAX_LAYERS];

    struct ggml_tensor * pre_conv_w;   // [io, io] (Conv1d k=1, no bias)
    struct ggml_tensor * post_conv_w;  // [io, io]
    struct ggml_tensor * cond0_w;      // to_cond_embed.0 [768, dim]
    struct ggml_tensor * cond2_w;      // to_cond_embed.2 [dim, dim]
    struct ggml_tensor * glob0_w;      // to_global_embed.0 [768, dim]
    struct ggml_tensor * glob2_w;      // to_global_embed.2 [dim, dim]
    struct ggml_tensor * ts0_w;        // to_timestep_embed.0 [256, dim]
    struct ggml_tensor * ts0_b;        // f32 [dim]
    struct ggml_tensor * ts2_w;        // [dim, dim]
    struct ggml_tensor * ts2_b;        // f32 [dim]
    struct ggml_tensor * gc0_w;        // global_cond_embedder.0 [dim, dim]
    struct ggml_tensor * gc0_b;        // f32 [dim]
    struct ggml_tensor * gc2_w;        // global_cond_embedder.2 [dim, 6*dim]
    struct ggml_tensor * gc2_b;        // f32 [6*dim]
    struct ggml_tensor * proj_in_w;    // transformer.project_in [io, dim]
    struct ggml_tensor * proj_out_w;   // transformer.project_out [dim, io]
    struct ggml_tensor * memory_toks;  // f32 [dim, n_memory_tokens]

    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx            wctx;

    // Debug (SA3_DIT_STAGE): if >= 0, also dump the token sequence [dim, S]
    // after this many transformer layers (0 = post-memory-prepend input) into
    // debug_out (token-major, matches torch (b, n, d) contiguous).
    int                debug_stage = -1;
    std::vector<float> debug_out;
};

// ── config ──────────────────────────────────────────────────────────────────

// Parse model.diffusion.config from the full SA3 model_config.json in GGUF
// metadata "sa3.config_json".
static void sa3_dit_parse_config(SA3DiTConfig * c, const char * config_json) {
    // Defaults: stable-audio-3-medium
    c->io_channels     = 256;
    c->dim             = 1536;
    c->depth           = 24;
    c->n_heads         = 24;
    c->cond_token_dim  = 768;
    c->global_cond_dim = 768;
    c->local_cond_dim  = 257;
    c->n_memory_tokens = 64;
    c->patch_size      = 1;
    c->ts_feat_dim     = 256;
    float ff_mult      = 4.0f;
    c->rope_theta      = 10000.0f;
    c->norm_eps        = 1e-5f;
    c->qk_norm_eps     = 1e-6f;
    c->ts_min_freq     = 0.5f;
    c->ts_max_freq     = 10000.0f;

    yyjson_doc * doc = NULL;
    if (config_json && config_json[0]) {
        doc = yyjson_read(config_json, strlen(config_json), 0);
    }
    if (doc) {
        yyjson_val * root  = yyjson_doc_get_root(doc);
        yyjson_val * model = root ? yyjson_obj_get(root, "model") : NULL;
        yyjson_val * diff  = model ? yyjson_obj_get(model, "diffusion") : NULL;
        yyjson_val * dc    = diff ? yyjson_obj_get(diff, "config") : NULL;
        if (dc) {
            yyjson_val * v;
            if ((v = yyjson_obj_get(dc, "io_channels")) && yyjson_is_int(v)) {
                c->io_channels = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "embed_dim")) && yyjson_is_int(v)) {
                c->dim = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "depth")) && yyjson_is_int(v)) {
                c->depth = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "num_heads")) && yyjson_is_int(v)) {
                c->n_heads = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "cond_token_dim")) && yyjson_is_int(v)) {
                c->cond_token_dim = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "global_cond_dim")) && yyjson_is_int(v)) {
                c->global_cond_dim = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "local_add_cond_dim")) && yyjson_is_int(v)) {
                c->local_cond_dim = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "num_memory_tokens")) && yyjson_is_int(v)) {
                c->n_memory_tokens = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "patch_size")) && yyjson_is_int(v)) {
                c->patch_size = (int) yyjson_get_int(v);
            }
            if ((v = yyjson_obj_get(dc, "timestep_features_dim")) && yyjson_is_int(v)) {
                c->ts_feat_dim = (int) yyjson_get_int(v);
            }
            yyjson_val * ffk = yyjson_obj_get(dc, "ff_kwargs");
            if (ffk && (v = yyjson_obj_get(ffk, "mult")) && yyjson_is_num(v)) {
                ff_mult = (float) yyjson_get_num(v);
            }
        } else {
            fprintf(stderr, "[SA3-DiT] WARNING: config json missing diffusion config, using medium defaults\n");
        }
        yyjson_doc_free(doc);
    } else {
        fprintf(stderr, "[SA3-DiT] WARNING: no config json, using medium defaults\n");
    }

    c->dim_heads = c->dim / c->n_heads;
    c->ff_inner  = (int) ((float) c->dim * ff_mult);          // FeedForward L467: int(dim * mult)
    c->rope_dims = c->dim_heads / 2 >= 32 ? c->dim_heads / 2 : 32;  // RotaryEmbedding(max(dh//2, 32)), L1109

    if (c->patch_size != 1) {
        fprintf(stderr, "[SA3-DiT] FATAL: patch_size %d != 1 not implemented\n", c->patch_size);
        exit(1);
    }
}

// ── loading ─────────────────────────────────────────────────────────────────

static bool sa3_dit_load(SA3DiT * m, const char * gguf_path) {
    BackendPair bp = backend_init("SA3-DiT");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 8192);

    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[SA3-DiT] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    sa3_dit_parse_config(&m->cfg, gf_get_str(gf, "sa3.config_json"));
    const SA3DiTConfig & c = m->cfg;
    if (c.depth > SA3_DIT_MAX_LAYERS) {
        fprintf(stderr, "[SA3-DiT] FATAL: %d layers > max %d\n", c.depth, SA3_DIT_MAX_LAYERS);
        gf_close(&gf);
        return false;
    }

    wctx_init(&m->wctx, 20 + c.depth * 21);

    // Conv1d k=1 weights: torch (out, in, 1) -> ggml (1, in, out); reload as [in, out].
    {
        const int64_t conv_shape[2] = { c.io_channels, c.io_channels };
        m->pre_conv_w  = gf_load_tensor(&m->wctx, gf, "model.preprocess_conv.weight", conv_shape, 2);
        m->post_conv_w = gf_load_tensor(&m->wctx, gf, "model.postprocess_conv.weight", conv_shape, 2);
    }

    m->cond0_w = gf_load_tensor(&m->wctx, gf, "model.to_cond_embed.0.weight");
    m->cond2_w = gf_load_tensor(&m->wctx, gf, "model.to_cond_embed.2.weight");
    m->glob0_w = gf_load_tensor(&m->wctx, gf, "model.to_global_embed.0.weight");
    m->glob2_w = gf_load_tensor(&m->wctx, gf, "model.to_global_embed.2.weight");
    m->ts0_w   = gf_load_tensor(&m->wctx, gf, "model.to_timestep_embed.0.weight");
    m->ts0_b   = gf_load_tensor_f32(&m->wctx, gf, "model.to_timestep_embed.0.bias");
    m->ts2_w   = gf_load_tensor(&m->wctx, gf, "model.to_timestep_embed.2.weight");
    m->ts2_b   = gf_load_tensor_f32(&m->wctx, gf, "model.to_timestep_embed.2.bias");
    m->gc0_w   = gf_load_tensor(&m->wctx, gf, "model.transformer.global_cond_embedder.0.weight");
    m->gc0_b   = gf_load_tensor_f32(&m->wctx, gf, "model.transformer.global_cond_embedder.0.bias");
    m->gc2_w   = gf_load_tensor(&m->wctx, gf, "model.transformer.global_cond_embedder.2.weight");
    m->gc2_b   = gf_load_tensor_f32(&m->wctx, gf, "model.transformer.global_cond_embedder.2.bias");

    m->proj_in_w   = gf_load_tensor(&m->wctx, gf, "model.transformer.project_in.weight");
    m->proj_out_w  = gf_load_tensor(&m->wctx, gf, "model.transformer.project_out.weight");
    // f32: concatenated with the f32 activation sequence (ggml_concat needs same type)
    m->memory_toks = gf_load_tensor_f32(&m->wctx, gf, "model.transformer.memory_tokens");

    for (int i = 0; i < c.depth; i++) {
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "model.transformer.layers.%d", i);
        std::string   p  = pfx;
        SA3DiTLayer * ly = &m->layers[i];

        ly->pre_norm_g = gf_load_tensor_f32(&m->wctx, gf, p + ".pre_norm.gamma");
        ly->ca_norm_g  = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attend_norm.gamma");
        ly->ff_norm_g  = gf_load_tensor_f32(&m->wctx, gf, p + ".ff_norm.gamma");
        ly->ssg        = gf_load_tensor_f32(&m->wctx, gf, p + ".to_scale_shift_gate");

        ly->attn_qkv  = gf_load_tensor(&m->wctx, gf, p + ".self_attn.to_qkv.weight");
        ly->attn_out  = gf_load_tensor(&m->wctx, gf, p + ".self_attn.to_out.weight");
        ly->attn_qn_g = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.q_norm.gamma");
        ly->attn_kn_g = gf_load_tensor_f32(&m->wctx, gf, p + ".self_attn.k_norm.gamma");

        ly->cross_q    = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.to_q.weight");
        ly->cross_kv   = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.to_kv.weight");
        ly->cross_out  = gf_load_tensor(&m->wctx, gf, p + ".cross_attn.to_out.weight");
        ly->cross_qn_g = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attn.q_norm.gamma");
        ly->cross_kn_g = gf_load_tensor_f32(&m->wctx, gf, p + ".cross_attn.k_norm.gamma");

        ly->ff0_w = gf_load_tensor(&m->wctx, gf, p + ".ff.ff.0.proj.weight");
        ly->ff0_b = gf_load_tensor_f32(&m->wctx, gf, p + ".ff.ff.0.proj.bias");
        ly->ff2_w = gf_load_tensor(&m->wctx, gf, p + ".ff.ff.2.weight");
        ly->ff2_b = gf_load_tensor_f32(&m->wctx, gf, p + ".ff.ff.2.bias");

        ly->loc0_w = gf_load_tensor(&m->wctx, gf, p + ".to_local_embed.0.weight");
        ly->loc0_b = gf_load_tensor_f32(&m->wctx, gf, p + ".to_local_embed.0.bias");
        ly->loc2_w = gf_load_tensor(&m->wctx, gf, p + ".to_local_embed.2.weight");
        ly->loc2_b = gf_load_tensor_f32(&m->wctx, gf, p + ".to_local_embed.2.bias");
    }

    fprintf(stderr, "[Load] SA3-DiT: dim=%d, %dL, heads=%dx%d, io=%d, mem=%d, ff=%d, rope=%d\n", c.dim,
            c.depth, c.n_heads, c.dim_heads, c.io_channels, c.n_memory_tokens, c.ff_inner, c.rope_dims);

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    return true;
}

// ── graph builders ──────────────────────────────────────────────────────────

// RMSNorm (transformer.py L392-410): x * rsqrt(mean(x^2) + eps) * gamma.
// gamma broadcasts over ne0.
static struct ggml_tensor * sa3_dit_rms(struct ggml_context * ctx, struct ggml_tensor * x,
                                        struct ggml_tensor * gamma, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), gamma);
}

// One attention map: q, k [D, S_q/S_kv, H] f32 (cont), vt [S_kv, D, H]
// (cont transpose of v) -> out [D, S_q, H]. No mask (padding is V-zeroed).
static struct ggml_tensor * sa3_dit_attn_map(struct ggml_context * ctx, struct ggml_tensor * q,
                                             struct ggml_tensor * k, struct ggml_tensor * vt, float scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);                        // [S_kv, S_q, H]
    struct ggml_tensor * probs  = ggml_soft_max_ext(ctx, scores, NULL, scale, 0.0f);
    return ggml_mul_mat(ctx, vt, probs);                                          // [D, S_q, H]
}

// One TransformerBlock (adaLN branch) on x [dim, S].
// gcond: [6*dim, 1] f32 (global_cond_embedder output).
// positions: [S] i32. vmask: [1, 1, S] f32 (1 valid / 0 padded, memory=1).
// cross: [dim, S_c] projected conditioning tokens.
// local: [dim, T] per-layer-INDEPENDENT? no — local input is shared; the
//        per-layer projection happens here. local_in: [local_cond_dim, T].
// zeros_mem: [dim, n_mem] f32 zeros (front padding for the local embed).
static struct ggml_tensor * sa3_dit_build_layer(struct ggml_context * ctx, const SA3DiTConfig & c,
                                                SA3DiTLayer * ly, struct ggml_tensor * x,
                                                struct ggml_tensor * gcond, struct ggml_tensor * positions,
                                                struct ggml_tensor * vmask, struct ggml_tensor * cross,
                                                struct ggml_tensor * local_in, struct ggml_tensor * zeros_mem,
                                                int64_t S, int64_t S_c) {
    const int     D     = c.dim_heads;
    const int     H     = c.n_heads;
    const int64_t dim   = c.dim;
    const float   scale = 1.0f / sqrtf((float) D);  // SDPA default scale

    // adaLN vectors (transformer.py L1020): param + global_cond, chunk 6
    struct ggml_tensor * ssg = ggml_add(ctx, ly->ssg, gcond);  // [6*dim, 1]
    auto chunk = [&](int i) {
        return ggml_cont(ctx, ggml_view_1d(ctx, ssg, dim, (size_t) i * dim * sizeof(float)));
    };
    struct ggml_tensor * scale_self = chunk(0);
    struct ggml_tensor * shift_self = chunk(1);
    struct ggml_tensor * gate_self  = chunk(2);
    struct ggml_tensor * scale_ff   = chunk(3);
    struct ggml_tensor * shift_ff   = chunk(4);
    struct ggml_tensor * gate_ff    = chunk(5);

    // sigmoid(1 - gate) (L1027, L1047)
    struct ggml_tensor * sg_self = ggml_sigmoid(ctx, ggml_scale_bias(ctx, gate_self, -1.0f, 1.0f));
    struct ggml_tensor * sg_ff   = ggml_sigmoid(ctx, ggml_scale_bias(ctx, gate_ff, -1.0f, 1.0f));

    // x*(1+s)+b == x + x*s + b
    auto mod = [&](struct ggml_tensor * t, struct ggml_tensor * s, struct ggml_tensor * b) {
        return ggml_add(ctx, ggml_add(ctx, t, ggml_mul(ctx, t, s)), b);
    };

    // [dim, S] slice i of a fused projection -> heads [D, H, S]
    auto slice_heads = [&](struct ggml_tensor * fused, int i, int64_t n) {
        struct ggml_tensor * v =
            ggml_view_2d(ctx, fused, dim, n, fused->nb[1], (size_t) i * dim * sizeof(float));
        return ggml_reshape_3d(ctx, ggml_cont(ctx, v), D, H, n);
    };
    // [D, H, S] -> [D, S, H] cont
    auto heads_mid = [&](struct ggml_tensor * t) {
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
    };

    // ── self-attention (adaLN-modulated, gated) ──
    {
        struct ggml_tensor * xn = sa3_dit_rms(ctx, x, ly->pre_norm_g, c.norm_eps);
        xn                      = mod(xn, scale_self, shift_self);

        struct ggml_tensor * qkv = ggml_mul_mat(ctx, ly->attn_qkv, xn);  // [5*dim, S]
        struct ggml_tensor * q   = slice_heads(qkv, 0, S);
        struct ggml_tensor * k   = slice_heads(qkv, 1, S);
        struct ggml_tensor * v   = slice_heads(qkv, 2, S);
        struct ggml_tensor * qd  = slice_heads(qkv, 3, S);
        struct ggml_tensor * kd  = slice_heads(qkv, 4, S);

        // qk-norm "rms" on the head dim, shared params for main/diff (L740-751)
        q  = sa3_dit_rms(ctx, q, ly->attn_qn_g, c.qk_norm_eps);
        qd = sa3_dit_rms(ctx, qd, ly->attn_qn_g, c.qk_norm_eps);
        k  = sa3_dit_rms(ctx, k, ly->attn_kn_g, c.qk_norm_eps);
        kd = sa3_dit_rms(ctx, kd, ly->attn_kn_g, c.qk_norm_eps);

        // RoPE after qk-norm; NEOX pairing, partial (32 of 64 dims), theta 10000
        auto rope = [&](struct ggml_tensor * t) {
            return ggml_rope_ext(ctx, t, positions, NULL, c.rope_dims, GGML_ROPE_TYPE_NEOX, 0,
                                 c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        };
        q  = rope(q);
        k  = rope(k);
        qd = rope(qd);
        kd = rope(kd);

        // padding: V-zeroing only (transformer.py L675-677); softmax stays unmasked
        v = ggml_mul(ctx, v, vmask);

        q  = heads_mid(q);
        k  = heads_mid(k);
        qd = heads_mid(qd);
        kd = heads_mid(kd);
        struct ggml_tensor * vp = heads_mid(v);                             // [D, S, H]
        struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vp));  // [S, D, H]

        // Differential (L790-795): attn(q,k,v) - attn(qd,kd,v), shared v
        struct ggml_tensor * o1  = sa3_dit_attn_map(ctx, q, k, vt, scale);
        struct ggml_tensor * o2  = sa3_dit_attn_map(ctx, qd, kd, vt, scale);
        struct ggml_tensor * out = ggml_sub(ctx, o1, o2);                   // [D, S, H]

        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));           // [D, H, S]
        out = ggml_reshape_2d(ctx, out, dim, S);
        out = ggml_mul_mat(ctx, ly->attn_out, out);                         // [dim, S]
        out = ggml_mul(ctx, out, sg_self);                                  // * sigmoid(1 - gate)
        x   = ggml_add(ctx, x, out);
    }

    // ── cross-attention (ungated, unmodulated; L1031-1035) ──
    {
        struct ggml_tensor * xn = sa3_dit_rms(ctx, x, ly->ca_norm_g, c.norm_eps);

        struct ggml_tensor * qq = ggml_mul_mat(ctx, ly->cross_q, xn);      // [2*dim, S]
        struct ggml_tensor * kv = ggml_mul_mat(ctx, ly->cross_kv, cross);  // [3*dim, S_c]
        struct ggml_tensor * q  = slice_heads(qq, 0, S);
        struct ggml_tensor * qd = slice_heads(qq, 1, S);
        struct ggml_tensor * k  = slice_heads(kv, 0, S_c);                 // chunk order k, k_diff, v (L727)
        struct ggml_tensor * kd = slice_heads(kv, 1, S_c);
        struct ggml_tensor * v  = slice_heads(kv, 2, S_c);

        q  = sa3_dit_rms(ctx, q, ly->cross_qn_g, c.qk_norm_eps);
        qd = sa3_dit_rms(ctx, qd, ly->cross_qn_g, c.qk_norm_eps);
        k  = sa3_dit_rms(ctx, k, ly->cross_kn_g, c.qk_norm_eps);
        kd = sa3_dit_rms(ctx, kd, ly->cross_kn_g, c.qk_norm_eps);
        // no RoPE on cross-attention (cross_attn_rotary_pos_emb=False)

        q  = heads_mid(q);
        qd = heads_mid(qd);
        k  = heads_mid(k);
        kd = heads_mid(kd);
        struct ggml_tensor * vp = heads_mid(v);                             // [D, S_c, H]
        struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, vp));  // [S_c, D, H]

        struct ggml_tensor * o1  = sa3_dit_attn_map(ctx, q, k, vt, scale);
        struct ggml_tensor * o2  = sa3_dit_attn_map(ctx, qd, kd, vt, scale);
        struct ggml_tensor * out = ggml_sub(ctx, o1, o2);

        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        out = ggml_reshape_2d(ctx, out, dim, S);
        out = ggml_mul_mat(ctx, ly->cross_out, out);
        x   = ggml_add(ctx, x, out);
    }

    // ── local additive conditioning (L978-982, L1040) ──
    {
        struct ggml_tensor * le = ggml_mul_mat(ctx, ly->loc0_w, local_in);  // [dim, T]
        le                      = ggml_add(ctx, le, ly->loc0_b);
        le                      = ggml_silu(ctx, le);
        le                      = ggml_mul_mat(ctx, ly->loc2_w, le);
        le                      = ggml_add(ctx, le, ly->loc2_b);
        // _left_pad_to_match: zero rows for the memory tokens at the FRONT
        struct ggml_tensor * padded = ggml_concat(ctx, zeros_mem, le, 1);   // [dim, S]
        x                           = ggml_add(ctx, x, padded);
    }

    // ── feed-forward (adaLN-modulated, gated): GLU mult 4, SiLU gate ──
    {
        struct ggml_tensor * y = sa3_dit_rms(ctx, x, ly->ff_norm_g, c.norm_eps);
        y                      = mod(y, scale_ff, shift_ff);

        struct ggml_tensor * p = ggml_mul_mat(ctx, ly->ff0_w, y);  // [2*inner, S]
        p                      = ggml_add(ctx, p, ly->ff0_b);
        const int64_t        inner = c.ff_inner;
        struct ggml_tensor * a =
            ggml_cont(ctx, ggml_view_2d(ctx, p, inner, S, p->nb[1], 0));
        struct ggml_tensor * g =
            ggml_cont(ctx, ggml_view_2d(ctx, p, inner, S, p->nb[1], (size_t) inner * sizeof(float)));
        struct ggml_tensor * h = ggml_mul(ctx, a, ggml_silu(ctx, g));  // x * SiLU(gate) (L443-444)
        h                      = ggml_mul_mat(ctx, ly->ff2_w, h);      // [dim, S]
        h                      = ggml_add(ctx, h, ly->ff2_b);
        h                      = ggml_mul(ctx, h, sg_ff);
        x                      = ggml_add(ctx, x, h);
    }

    return x;
}

// ── forward ─────────────────────────────────────────────────────────────────
//
// One DiT forward (batch 1). Layouts are torch-contiguous:
//   x            [io_channels, T]      (torch (1, C, T))
//   t            scalar in [0, 1]
//   cross        [S_c, cond_token_dim] (torch (1, S_c, 768))
//   glob         [global_cond_dim]     (torch (1, 768))
//   local        [local_cond_dim, T]   (torch (1, 257, T))
//   padding_mask [T] u8, 1 = valid
//   out          [io_channels, T]      (torch (1, C, T))
static void sa3_dit_forward(SA3DiT * m, const float * x_in_p, float t_val, const float * cross_p,
                            int64_t S_c, const float * glob_p, const float * local_p,
                            const uint8_t * padding_mask, int64_t T, float * out_p) {
    const SA3DiTConfig & c = m->cfg;
    const int64_t dim   = c.dim;
    const int64_t io    = c.io_channels;
    const int64_t n_mem = c.n_memory_tokens;
    const int64_t S     = n_mem + T;

    size_t                  ctx_size = (size_t) 16384 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);
    struct ggml_cgraph *    graph    = ggml_new_graph_custom(ctx, 8192, false);

    // ── inputs ──
    struct ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, io, T);
    ggml_set_name(x_in, "x");
    ggml_set_input(x_in);
    struct ggml_tensor * cross_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.cond_token_dim, S_c);
    ggml_set_name(cross_in, "cross");
    ggml_set_input(cross_in);
    struct ggml_tensor * glob_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.global_cond_dim, 1);
    ggml_set_name(glob_in, "glob");
    ggml_set_input(glob_in);
    struct ggml_tensor * local_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.local_cond_dim, T);
    ggml_set_name(local_in, "local");
    ggml_set_input(local_in);
    struct ggml_tensor * tfeat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ts_feat_dim, 1);
    ggml_set_name(tfeat, "tfeat");
    ggml_set_input(tfeat);
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    struct ggml_tensor * vmask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, S);
    ggml_set_name(vmask, "vmask");
    ggml_set_input(vmask);
    struct ggml_tensor * zeros_mem = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_mem);
    ggml_set_name(zeros_mem, "zeros_mem");
    ggml_set_input(zeros_mem);

    // ── conditioning projections (dit.py L197-202, L239-245) ──
    // cross: to_cond_embed = Linear/SiLU/Linear, no bias
    struct ggml_tensor * cross = ggml_mul_mat(ctx, m->cond0_w, cross_in);  // [dim, S_c]
    cross                      = ggml_silu(ctx, cross);
    cross                      = ggml_mul_mat(ctx, m->cond2_w, cross);

    // global: to_global_embed, no bias
    struct ggml_tensor * glob = ggml_mul_mat(ctx, m->glob0_w, glob_in);    // [dim, 1]
    glob                      = ggml_silu(ctx, glob);
    glob                      = ggml_mul_mat(ctx, m->glob2_w, glob);

    // timestep: to_timestep_embed on ExpoFourier features (CPU-computed)
    struct ggml_tensor * ts = ggml_mul_mat(ctx, m->ts0_w, tfeat);          // [dim, 1]
    ts                      = ggml_add(ctx, ts, m->ts0_b);
    ts                      = ggml_silu(ctx, ts);
    ts                      = ggml_mul_mat(ctx, m->ts2_w, ts);
    ts                      = ggml_add(ctx, ts, m->ts2_b);

    // timestep_cond_type "global": global_embed += timestep_embed (L243-245)
    glob = ggml_add(ctx, glob, ts);

    // transformer.global_cond_embedder (transformer.py L1130-1136, L1210-1211)
    struct ggml_tensor * gcond = ggml_mul_mat(ctx, m->gc0_w, glob);        // [dim, 1]
    gcond                      = ggml_add(ctx, gcond, m->gc0_b);
    gcond                      = ggml_silu(ctx, gcond);
    gcond                      = ggml_mul_mat(ctx, m->gc2_w, gcond);       // [6*dim, 1]
    gcond                      = ggml_add(ctx, gcond, m->gc2_b);

    // ── input path: preprocess conv residual + project_in + memory tokens ──
    struct ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, m->pre_conv_w, x_in), x_in);  // (dit.py L264)
    h                      = ggml_mul_mat(ctx, m->proj_in_w, h);                           // [dim, T]
    struct ggml_tensor * seq = ggml_concat(ctx, m->memory_toks, h, 1);                     // [dim, S]

    struct ggml_tensor * dbg = NULL;
    if (m->debug_stage == 0) {
        dbg = seq;
    }
    for (int i = 0; i < c.depth; i++) {
        seq = sa3_dit_build_layer(ctx, c, &m->layers[i], seq, gcond, positions, vmask, cross, local_in,
                                  zeros_mem, S, S_c);
        if (m->debug_stage == i + 1) {
            dbg = seq;
        }
    }
    if (dbg) {
        dbg = ggml_cont(ctx, dbg);
        ggml_set_name(dbg, "debug_stage");
        ggml_set_output(dbg);
    }

    // drop memory tokens (transformer.py L1265), project_out, postprocess conv
    struct ggml_tensor * tail =
        ggml_cont(ctx, ggml_view_2d(ctx, seq, dim, T, seq->nb[1], (size_t) n_mem * seq->nb[1]));
    struct ggml_tensor * out = ggml_mul_mat(ctx, m->proj_out_w, tail);          // [io, T]
    out                      = ggml_add(ctx, ggml_mul_mat(ctx, m->post_conv_w, out), out);  // (dit.py L304)
    ggml_set_name(out, "v");
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);
    if (dbg) {
        ggml_build_forward_expand(graph, dbg);
    }

    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        fprintf(stderr, "[SA3-DiT] FATAL: failed to allocate graph (T=%lld)\n", (long long) T);
        exit(1);
    }

    // ── upload inputs ──
    {
        // x torch (1, C, T) -> token-major [C, T]
        std::vector<float> buf((size_t) io * T);
        for (int64_t tt = 0; tt < T; tt++) {
            for (int64_t ch = 0; ch < io; ch++) {
                buf[(size_t) (tt * io + ch)] = x_in_p[(size_t) ch * T + tt];
            }
        }
        ggml_backend_tensor_set(x_in, buf.data(), 0, buf.size() * sizeof(float));
    }
    // cross torch (1, S_c, 768) is already token-major [768, S_c]
    ggml_backend_tensor_set(cross_in, cross_p, 0, (size_t) c.cond_token_dim * S_c * sizeof(float));
    ggml_backend_tensor_set(glob_in, glob_p, 0, (size_t) c.global_cond_dim * sizeof(float));
    {
        // local torch (1, 257, T) -> rearrange "b c t -> b t c" (dit.py L225)
        std::vector<float> buf((size_t) c.local_cond_dim * T);
        for (int64_t tt = 0; tt < T; tt++) {
            for (int64_t ch = 0; ch < c.local_cond_dim; ch++) {
                buf[(size_t) (tt * c.local_cond_dim + ch)] = local_p[(size_t) ch * T + tt];
            }
        }
        ggml_backend_tensor_set(local_in, buf.data(), 0, buf.size() * sizeof(float));
    }
    {
        // ExpoFourierFeatures(256, 0.5, 10000) on raw t (blocks.py L50-82;
        // timestep_features_logsnr=False so t is used directly, dit.py L235).
        const int          half = c.ts_feat_dim / 2;
        std::vector<float> feat((size_t) c.ts_feat_dim);
        const double       log_min = log((double) c.ts_min_freq);
        const double       log_max = log((double) c.ts_max_freq);
        for (int i = 0; i < half; i++) {
            double ramp = (half > 1) ? (double) i / (double) (half - 1) : 0.0;
            float  freq = (float) exp(ramp * (log_max - log_min) + log_min);
            float  arg  = t_val * freq;
            arg         = arg * 2.0f;
            arg         = arg * (float) 3.14159265358979323846;
            feat[(size_t) i]          = cosf(arg);  // cos first (blocks.py L80)
            feat[(size_t) (half + i)] = sinf(arg);
        }
        ggml_backend_tensor_set(tfeat, feat.data(), 0, feat.size() * sizeof(float));
    }
    {
        std::vector<int32_t> pos((size_t) S);
        for (int64_t i = 0; i < S; i++) {
            pos[(size_t) i] = (int32_t) i;
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, (size_t) S * sizeof(int32_t));
    }
    {
        // extended padding mask: memory tokens always valid (transformer.py L1222-1225)
        std::vector<float> vm((size_t) S, 1.0f);
        for (int64_t tt = 0; tt < T; tt++) {
            vm[(size_t) (n_mem + tt)] = padding_mask[tt] ? 1.0f : 0.0f;
        }
        ggml_backend_tensor_set(vmask, vm.data(), 0, (size_t) S * sizeof(float));
    }
    {
        std::vector<float> z((size_t) dim * n_mem, 0.0f);
        ggml_backend_tensor_set(zeros_mem, z.data(), 0, z.size() * sizeof(float));
    }

    ggml_backend_sched_graph_compute(m->sched, graph);

    if (dbg) {
        m->debug_out.resize((size_t) dim * S);
        ggml_backend_tensor_get(dbg, m->debug_out.data(), 0, m->debug_out.size() * sizeof(float));
    }

    {
        // [io, T] token-major -> torch (1, C, T)
        std::vector<float> buf((size_t) io * T);
        ggml_backend_tensor_get(out, buf.data(), 0, buf.size() * sizeof(float));
        for (int64_t tt = 0; tt < T; tt++) {
            for (int64_t ch = 0; ch < io; ch++) {
                out_p[(size_t) ch * T + tt] = buf[(size_t) (tt * io + ch)];
            }
        }
    }

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

static void sa3_dit_free(SA3DiT * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}
