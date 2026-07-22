// sa3-t5gemma-enc.h: Stable Audio 3 text conditioner (T5Gemma encoder) via ggml
//
// Ports the SA3 "prompt" conditioner: a BIDIRECTIONAL T5Gemma encoder
// (google/t5gemma-b-b-ul2, encoder half only) followed by the SA3 learned
// padding substitution. Weights: engine/models/sa3-text-enc-BF16.gguf
// (arch "sa3-t5gemma", HF tensor names under "model.encoder.*" plus
// "conditioner.conditioners.prompt.padding_embedding").
//
// Architecture (verified against transformers/models/t5gemma/modeling_t5gemma.py):
//   - embed lookup, then scale by sqrt(hidden_size)            (L700-701)
//   - per layer (T5GemmaEncoderLayer.forward, L426-452):
//       h = residual + post_self_attn_norm(self_attn(pre_self_attn_norm(h)))
//       h = residual + post_ff_norm(mlp(pre_ff_norm(h)))
//     (both PRE and POST sublayer RMSNorms; post-norm applied BEFORE the
//      residual add)
//   - RMSNorm is Gemma-style: x * rsqrt(mean(x^2)+eps) * (1 + weight),
//     computed in f32 (L60-74)
//   - MLP is GeGLU: down( gelu_tanh(gate(x)) * up(x) ), hidden_activation
//     "gelu_pytorch_tanh" (L92-96 + config)
//   - attention: no q/k/v bias, NO qk-norm, query scale =
//     query_pre_attn_scalar^-0.5 (L254), attn-logit softcapping
//     tanh(logits/50)*50 applied AFTER scaling and BEFORE the mask (L226-233)
//   - RoPE: rotate_half = NEOX half-split pairing (L164-168), theta 10000,
//     dims = head_dim (L137-146)
//   - MHA not GQA for this checkpoint: 12 heads == 12 kv heads
//   - alternating sliding/full attention layers (config layer_types).
//     Encoder is bidirectional; sliding window is symmetric AND inclusive:
//     |q - kv| <= sliding_window (masking_utils.py L121-131). With
//     max_length 256 << window 4096 this is a no-op but is implemented anyway.
//   - final RMSNorm (L714)
// Post-encoder (stable_audio_3/models/conditioners.py L41-68, L267):
//   padding_mode "learned": output rows at attention_mask==0 positions are
//   REPLACED by the learned padding_embedding vector. proj_out is Identity.
//
// Also included: the tiny SA3 "seconds_total" conditioner (NumberConditioner,
// min_val 0 / max_val 384, expo Fourier features + Linear(256->768)); weights
// are loaded from the sa3-dit GGUF and evaluated in plain C++ on the CPU.

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

#define SA3_T5G_MAX_LAYERS 24

struct SA3T5GemmaConfig {
    int   hidden_size;        // 768
    int   intermediate_size;  // 2048
    int   n_heads;            // 12
    int   n_kv_heads;         // 12
    int   head_dim;           // 64
    int   n_layers;           // 12
    float rope_theta;         // 10000
    float rms_norm_eps;       // 1e-6
    float attn_softcap;       // 50.0 (attn_logit_softcapping)
    float query_scale;        // query_pre_attn_scalar^-0.5 = 64^-0.5
    int   sliding_window;     // 4096
    bool  layer_is_sliding[SA3_T5G_MAX_LAYERS];
};

struct SA3T5GemmaLayer {
    struct ggml_tensor * pre_self_attn_ln;   // [H] f32 (raw weight; graph applies 1+w)
    struct ggml_tensor * post_self_attn_ln;  // [H]
    struct ggml_tensor * pre_ff_ln;          // [H]
    struct ggml_tensor * post_ff_ln;         // [H]

    struct ggml_tensor * q_proj;  // [H, Nh*D]
    struct ggml_tensor * k_proj;  // [H, Nkv*D]
    struct ggml_tensor * v_proj;  // [H, Nkv*D]
    struct ggml_tensor * o_proj;  // [Nh*D, H]

    struct ggml_tensor * gate_proj;  // [H, FFN]
    struct ggml_tensor * up_proj;    // [H, FFN]
    struct ggml_tensor * down_proj;  // [FFN, H]
};

struct SA3T5GemmaEnc {
    SA3T5GemmaConfig cfg;
    SA3T5GemmaLayer  layers[SA3_T5G_MAX_LAYERS];

    struct ggml_tensor * embed_tokens;  // [H, V] bf16
    struct ggml_tensor * final_norm;    // [H] f32

    // SA3 learned padding embedding, kept on CPU: substituted into the output
    // at padded positions after the encoder runs.
    std::vector<float> padding_embedding;  // [H]

    ggml_backend_t       backend;
    ggml_backend_t       cpu_backend;
    ggml_backend_sched_t sched;
    WeightCtx            wctx;

    // Debug: if >0, truncate the graph after this many layers (skip final norm
    // and padding substitution) for layer-by-layer parity work.
    int debug_n_layers = 0;
};

// Gemma RMSNorm: n = x * rsqrt(mean(x^2)+eps); out = n * (1 + w) = n + n*w
static struct ggml_tensor * sa3_t5g_rms_norm(struct ggml_context * ctx,
                                             struct ggml_tensor *  x,
                                             struct ggml_tensor *  w,
                                             float                 eps) {
    struct ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_add(ctx, n, ggml_mul(ctx, n, w));
}

// Self-attention with logit softcapping. x: [H, S] -> [H, S]
// mask: [S, S] f32 (0 / -inf), broadcast over heads.
static struct ggml_tensor * sa3_t5g_build_self_attn(struct ggml_context *    ctx,
                                                    const SA3T5GemmaConfig & c,
                                                    SA3T5GemmaLayer *        ly,
                                                    struct ggml_tensor *     x,
                                                    struct ggml_tensor *     positions,  // [S] i32
                                                    struct ggml_tensor *     mask,       // [S, S] f32
                                                    int                      S) {
    const int D   = c.head_dim;
    const int Nh  = c.n_heads;
    const int Nkv = c.n_kv_heads;

    struct ggml_tensor * q = ggml_mul_mat(ctx, ly->q_proj, x);  // [Nh*D, S]
    struct ggml_tensor * k = ggml_mul_mat(ctx, ly->k_proj, x);  // [Nkv*D, S]
    struct ggml_tensor * v = ggml_mul_mat(ctx, ly->v_proj, x);  // [Nkv*D, S]

    q = ggml_reshape_3d(ctx, q, D, Nh, S);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S);
    v = ggml_reshape_3d(ctx, v, D, Nkv, S);

    // RoPE: NEOX half-split pairing (HF rotate_half), theta = c.rope_theta
    q = ggml_rope_ext(ctx, q, positions, NULL, D, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // [D, X, S] -> [D, S, X]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Manual F32 attention (precision-first; model is small). GQA note: this
    // checkpoint has Nh == Nkv so no kv-head broadcast is needed; guard anyway.
    GGML_ASSERT(Nh == Nkv && "sa3-t5gemma: GQA repeat not implemented (Nh != Nkv)");

    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // [S_kv, S_q, Nh]

    // scale, then softcap: tanh(logits * scale / cap) * cap, then mask+softmax
    if (c.attn_softcap > 0.0f) {
        scores = ggml_scale(ctx, scores, c.query_scale / c.attn_softcap);
        scores = ggml_tanh(ctx, scores);
        scores = ggml_scale(ctx, scores, c.attn_softcap);
        scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);
    } else {
        scores = ggml_soft_max_ext(ctx, scores, mask, c.query_scale, 0.0f);
    }

    struct ggml_tensor * vt  = ggml_cont(ctx, ggml_transpose(ctx, v));  // [S_kv, D, Nh]
    struct ggml_tensor * out = ggml_mul_mat(ctx, vt, scores);           // [D, S_q, Nh]
    out                      = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));  // [D, Nh, S]
    out                      = ggml_reshape_2d(ctx, out, Nh * D, S);

    return ggml_mul_mat(ctx, ly->o_proj, out);  // [H, S]
}

// GeGLU MLP: down( gelu_tanh(gate(x)) * up(x) )
static struct ggml_tensor * sa3_t5g_build_mlp(struct ggml_context * ctx, SA3T5GemmaLayer * ly, struct ggml_tensor * x) {
    struct ggml_tensor * gate = ggml_mul_mat(ctx, ly->gate_proj, x);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, ly->up_proj, x);
    // ggml_gelu == tanh-approximation GELU == HF "gelu_pytorch_tanh"
    struct ggml_tensor * ff = ggml_geglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, ly->down_proj, ff);
}

static struct ggml_tensor * sa3_t5g_build_layer(struct ggml_context *    ctx,
                                                const SA3T5GemmaConfig & c,
                                                SA3T5GemmaLayer *        ly,
                                                struct ggml_tensor *     hidden,
                                                struct ggml_tensor *     positions,
                                                struct ggml_tensor *     mask,
                                                int                      S) {
    // h = h + post_norm(attn(pre_norm(h)))
    struct ggml_tensor * t = sa3_t5g_rms_norm(ctx, hidden, ly->pre_self_attn_ln, c.rms_norm_eps);
    t                      = sa3_t5g_build_self_attn(ctx, c, ly, t, positions, mask, S);
    t                      = sa3_t5g_rms_norm(ctx, t, ly->post_self_attn_ln, c.rms_norm_eps);
    hidden                 = ggml_add(ctx, hidden, t);

    // h = h + post_norm(mlp(pre_norm(h)))
    t      = sa3_t5g_rms_norm(ctx, hidden, ly->pre_ff_ln, c.rms_norm_eps);
    t      = sa3_t5g_build_mlp(ctx, ly, t);
    t      = sa3_t5g_rms_norm(ctx, t, ly->post_ff_ln, c.rms_norm_eps);
    hidden = ggml_add(ctx, hidden, t);

    return hidden;
}

// Parse the encoder section of the HF t5gemma config.json (stored verbatim in
// GGUF metadata key "sa3.config_json").
static void sa3_t5g_parse_config(SA3T5GemmaConfig * c, const char * config_json) {
    // Defaults: t5gemma-b-b-ul2 encoder
    c->hidden_size       = 768;
    c->intermediate_size = 2048;
    c->n_heads           = 12;
    c->n_kv_heads        = 12;
    c->head_dim          = 64;
    c->n_layers          = 12;
    c->rope_theta        = 10000.0f;
    c->rms_norm_eps      = 1e-6f;
    c->attn_softcap      = 50.0f;
    c->query_scale       = 1.0f / sqrtf(64.0f);
    c->sliding_window    = 4096;
    for (int i = 0; i < SA3_T5G_MAX_LAYERS; i++) {
        c->layer_is_sliding[i] = (i % 2) == 0;
    }

    if (!config_json || !config_json[0]) {
        fprintf(stderr, "[SA3-T5G] WARNING: no config json, using t5gemma-b-b-ul2 defaults\n");
        return;
    }
    yyjson_doc * doc = yyjson_read(config_json, strlen(config_json), 0);
    if (!doc) {
        fprintf(stderr, "[SA3-T5G] WARNING: cannot parse config json, using defaults\n");
        return;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * enc  = root ? yyjson_obj_get(root, "encoder") : NULL;
    if (enc && yyjson_is_obj(enc)) {
        yyjson_val * v;
        if ((v = yyjson_obj_get(enc, "hidden_size")) && yyjson_is_int(v)) {
            c->hidden_size = (int) yyjson_get_int(v);
        }
        if ((v = yyjson_obj_get(enc, "intermediate_size")) && yyjson_is_int(v)) {
            c->intermediate_size = (int) yyjson_get_int(v);
        }
        if ((v = yyjson_obj_get(enc, "num_attention_heads")) && yyjson_is_int(v)) {
            c->n_heads = (int) yyjson_get_int(v);
        }
        if ((v = yyjson_obj_get(enc, "num_key_value_heads")) && yyjson_is_int(v)) {
            c->n_kv_heads = (int) yyjson_get_int(v);
        }
        if ((v = yyjson_obj_get(enc, "head_dim")) && yyjson_is_int(v)) {
            c->head_dim = (int) yyjson_get_int(v);
        }
        if ((v = yyjson_obj_get(enc, "num_hidden_layers")) && yyjson_is_int(v)) {
            c->n_layers = (int) yyjson_get_int(v);
        }
        if ((v = yyjson_obj_get(enc, "rope_theta")) && yyjson_is_num(v)) {
            c->rope_theta = (float) yyjson_get_num(v);
        }
        if ((v = yyjson_obj_get(enc, "rms_norm_eps")) && yyjson_is_num(v)) {
            c->rms_norm_eps = (float) yyjson_get_num(v);
        }
        if ((v = yyjson_obj_get(enc, "attn_logit_softcapping"))) {
            c->attn_softcap = yyjson_is_num(v) ? (float) yyjson_get_num(v) : 0.0f;
        }
        if ((v = yyjson_obj_get(enc, "query_pre_attn_scalar")) && yyjson_is_num(v)) {
            c->query_scale = 1.0f / sqrtf((float) yyjson_get_num(v));
        }
        if ((v = yyjson_obj_get(enc, "sliding_window")) && yyjson_is_int(v)) {
            c->sliding_window = (int) yyjson_get_int(v);
        }
        if ((v = yyjson_obj_get(enc, "layer_types")) && yyjson_is_arr(v)) {
            size_t       idx, max;
            yyjson_val * lt;
            yyjson_arr_foreach(v, idx, max, lt) {
                if (idx < SA3_T5G_MAX_LAYERS && yyjson_is_str(lt)) {
                    c->layer_is_sliding[idx] = strcmp(yyjson_get_str(lt), "sliding_attention") == 0;
                }
            }
        }
    } else {
        fprintf(stderr, "[SA3-T5G] WARNING: config json has no 'encoder' object, using defaults\n");
    }
    yyjson_doc_free(doc);
}

static bool sa3_t5gemma_load(SA3T5GemmaEnc * m, const char * gguf_path) {
    BackendPair bp = backend_init("SA3-T5Gemma");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 4096);

    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[SA3-T5G] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    sa3_t5g_parse_config(&m->cfg, gf_get_str(gf, "sa3.config_json"));
    if (m->cfg.n_layers > SA3_T5G_MAX_LAYERS) {
        fprintf(stderr, "[SA3-T5G] FATAL: %d layers > max %d\n", m->cfg.n_layers, SA3_T5G_MAX_LAYERS);
        gf_close(&gf);
        return false;
    }

    // embed(1) + final_norm(1) + 12 layers * 11 tensors
    wctx_init(&m->wctx, 2 + m->cfg.n_layers * 11);

    m->embed_tokens = gf_load_tensor(&m->wctx, gf, "model.encoder.embed_tokens.weight");
    m->final_norm   = gf_load_tensor_f32(&m->wctx, gf, "model.encoder.norm.weight");

    for (int i = 0; i < m->cfg.n_layers; i++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "model.encoder.layers.%d", i);
        std::string       p  = prefix;
        SA3T5GemmaLayer * ly = &m->layers[i];

        ly->pre_self_attn_ln  = gf_load_tensor_f32(&m->wctx, gf, p + ".pre_self_attn_layernorm.weight");
        ly->post_self_attn_ln = gf_load_tensor_f32(&m->wctx, gf, p + ".post_self_attn_layernorm.weight");
        ly->pre_ff_ln         = gf_load_tensor_f32(&m->wctx, gf, p + ".pre_feedforward_layernorm.weight");
        ly->post_ff_ln        = gf_load_tensor_f32(&m->wctx, gf, p + ".post_feedforward_layernorm.weight");

        ly->q_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.q_proj.weight");
        ly->k_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.k_proj.weight");
        ly->v_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.v_proj.weight");
        ly->o_proj = gf_load_tensor(&m->wctx, gf, p + ".self_attn.o_proj.weight");

        ly->gate_proj = gf_load_tensor(&m->wctx, gf, p + ".mlp.gate_proj.weight");
        ly->up_proj   = gf_load_tensor(&m->wctx, gf, p + ".mlp.up_proj.weight");
        ly->down_proj = gf_load_tensor(&m->wctx, gf, p + ".mlp.down_proj.weight");
    }

    // Learned padding embedding (F32 1D [H]) -> CPU copy
    {
        const char * name = "conditioner.conditioners.prompt.padding_embedding";
        struct ggml_tensor * meta = ggml_get_tensor(gf.meta, name);
        const void *         data = gf_get_data(gf, name);
        if (!meta || !data || meta->type != GGML_TYPE_F32 || (int) meta->ne[0] != m->cfg.hidden_size) {
            fprintf(stderr, "[SA3-T5G] FATAL: bad or missing tensor '%s'\n", name);
            gf_close(&gf);
            return false;
        }
        m->padding_embedding.resize((size_t) m->cfg.hidden_size);
        memcpy(m->padding_embedding.data(), data, (size_t) m->cfg.hidden_size * sizeof(float));
    }

    fprintf(stderr, "[Load] SA3-T5Gemma: %dL, H=%d, Nh=%d/%d, D=%d, softcap=%.1f\n", m->cfg.n_layers,
            m->cfg.hidden_size, m->cfg.n_heads, m->cfg.n_kv_heads, m->cfg.head_dim, m->cfg.attn_softcap);

    if (!wctx_alloc(&m->wctx, m->backend)) {
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    return true;
}

// Forward: token IDs + attention mask -> conditioner embeddings.
// token_ids: [S] int32 (CPU)
// attn_mask: [S] uint8, 1 = valid token, 0 = padding (CPU)
// output:    [S * H] float (CPU, caller-allocated), token-major: out[s*H + h].
// The learned padding substitution is applied (rows at attn_mask==0 are the
// padding_embedding vector), matching the SA3 conditioner output.
static void sa3_t5gemma_forward(SA3T5GemmaEnc * m,
                                const int32_t * token_ids,
                                const uint8_t * attn_mask,
                                int             S,
                                float *         output) {
    const SA3T5GemmaConfig & c = m->cfg;
    const int                H = c.hidden_size;

    size_t                  ctx_size = 2048 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params gp       = { ctx_size, NULL, true };
    struct ggml_context *   ctx      = ggml_init(gp);
    struct ggml_cgraph *    graph    = ggml_new_graph_custom(ctx, 4096, false);

    struct ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(t_ids, "token_ids");
    ggml_set_input(t_ids);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Two bidirectional masks (0 / -inf), f32 [S_kv, S_q]:
    //   full:    key valid
    //   sliding: key valid AND |q - kv| <= sliding_window (inclusive)
    struct ggml_tensor * mask_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S);
    ggml_set_name(mask_full, "mask_full");
    ggml_set_input(mask_full);
    bool need_sliding = false;
    for (int i = 0; i < c.n_layers; i++) {
        need_sliding |= c.layer_is_sliding[i];
    }
    need_sliding                  = need_sliding && S > c.sliding_window + 1;
    struct ggml_tensor * mask_swa = mask_full;
    if (need_sliding) {
        mask_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S);
        ggml_set_name(mask_swa, "mask_swa");
        ggml_set_input(mask_swa);
    }

    // Embedding lookup [H, S] (f32 out of get_rows), scaled by sqrt(H)
    struct ggml_tensor * hidden = ggml_get_rows(ctx, m->embed_tokens, t_ids);
    hidden                      = ggml_scale(ctx, hidden, sqrtf((float) H));

    int n_layers = (m->debug_n_layers > 0 && m->debug_n_layers < c.n_layers) ? m->debug_n_layers : c.n_layers;
    for (int i = 0; i < n_layers; i++) {
        struct ggml_tensor * mask = c.layer_is_sliding[i] ? mask_swa : mask_full;
        hidden                    = sa3_t5g_build_layer(ctx, c, &m->layers[i], hidden, positions, mask, S);
    }
    struct ggml_tensor * out = hidden;
    if (n_layers == c.n_layers) {
        out = sa3_t5g_rms_norm(ctx, hidden, m->final_norm, c.rms_norm_eps);
    }
    ggml_set_name(out, "output");
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        fprintf(stderr, "[SA3-T5G] FATAL: failed to allocate graph (%d tokens)\n", S);
        exit(1);
    }

    ggml_backend_tensor_set(t_ids, token_ids, 0, (size_t) S * sizeof(int32_t));
    {
        std::vector<int32_t> pos(S);
        for (int i = 0; i < S; i++) {
            pos[i] = i;
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, (size_t) S * sizeof(int32_t));
    }
    {
        std::vector<float> md((size_t) S * S);
        for (int q = 0; q < S; q++) {
            for (int kv = 0; kv < S; kv++) {
                md[(size_t) q * S + kv] = attn_mask[kv] ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask_full, md.data(), 0, (size_t) S * S * sizeof(float));
        if (need_sliding) {
            for (int q = 0; q < S; q++) {
                for (int kv = 0; kv < S; kv++) {
                    int d = q > kv ? q - kv : kv - q;
                    if (d > c.sliding_window) {
                        md[(size_t) q * S + kv] = -INFINITY;
                    }
                }
            }
            ggml_backend_tensor_set(mask_swa, md.data(), 0, (size_t) S * S * sizeof(float));
        }
    }

    ggml_backend_sched_graph_compute(m->sched, graph);
    ggml_backend_tensor_get(out, output, 0, (size_t) H * S * sizeof(float));

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);

    // SA3 learned padding substitution (conditioners.py apply_padding, mode
    // "learned"): padded rows become the padding_embedding vector.
    if (m->debug_n_layers <= 0) {
        for (int s = 0; s < S; s++) {
            if (!attn_mask[s]) {
                memcpy(output + (size_t) s * H, m->padding_embedding.data(), (size_t) H * sizeof(float));
            }
        }
    }
}

static void sa3_t5gemma_free(SA3T5GemmaEnc * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    backend_release(m->backend, m->cpu_backend);
    wctx_free(&m->wctx);
    *m = {};
}

// ── SA3 seconds embedder ────────────────────────────────────────────────────
//
// NumberConditioner (conditioners.py L121-155): seconds are clamped to
// [min_val, max_val] = [0, 384] (SA3 medium model_config.json) and normalized
// to [0,1], then fed to NumberEmbedder with fourier_features_type "expo":
//   ExpoFourierFeatures (blocks.py L50-82): half = dim/2 = 128,
//     ramp  = linspace(0, 1, 128)
//     freqs = exp(ramp * (ln(10000) - ln(0.5)) + ln(0.5))
//     args  = t * freqs * 2 * pi
//     feat  = concat(cos(args), sin(args))       // cos FIRST, then sin
//   then Linear(256 -> 768) with bias.
// Weights (sa3-dit GGUF, arch "sa3-dit"):
//   conditioner.conditioners.seconds_total.embedder.embedding.1.weight [256,768] bf16
//   conditioner.conditioners.seconds_total.embedder.embedding.1.bias   [768]     f32
// Tiny model -> evaluated in plain C++ on the CPU (no ggml graph).

struct SA3SecondsEmbedder {
    int                in_dim  = 256;
    int                out_dim = 768;
    float              min_val = 0.0f;
    float              max_val = 384.0f;
    float              min_freq = 0.5f;
    float              max_freq = 10000.0f;
    std::vector<float> weight;  // [out_dim * in_dim], row-major per output
    std::vector<float> bias;    // [out_dim]
};

// Loads the two tensors from the sa3-dit GGUF (path to sa3-dit-*.gguf).
static bool sa3_seconds_embedder_load(SA3SecondsEmbedder * e, const char * dit_gguf_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, dit_gguf_path)) {
        fprintf(stderr, "[SA3-Seconds] FATAL: cannot load %s\n", dit_gguf_path);
        return false;
    }
    const char * w_name = "conditioner.conditioners.seconds_total.embedder.embedding.1.weight";
    const char * b_name = "conditioner.conditioners.seconds_total.embedder.embedding.1.bias";

    struct ggml_tensor * wm = ggml_get_tensor(gf.meta, w_name);
    struct ggml_tensor * bm = ggml_get_tensor(gf.meta, b_name);
    const void *         wd = gf_get_data(gf, w_name);
    const void *         bd = gf_get_data(gf, b_name);
    if (!wm || !bm || !wd || !bd) {
        fprintf(stderr, "[SA3-Seconds] FATAL: embedder tensors not found in %s\n", dit_gguf_path);
        gf_close(&gf);
        return false;
    }
    e->in_dim  = (int) wm->ne[0];
    e->out_dim = (int) wm->ne[1];
    size_t n   = (size_t) e->in_dim * e->out_dim;
    e->weight.resize(n);
    if (wm->type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) wd;
        for (size_t i = 0; i < n; i++) {
            e->weight[i] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &p[i]);
        }
    } else if (wm->type == GGML_TYPE_F32) {
        memcpy(e->weight.data(), wd, n * sizeof(float));
    } else if (wm->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) wd, e->weight.data(), (int) n);
    } else {
        fprintf(stderr, "[SA3-Seconds] FATAL: unsupported weight type %d\n", (int) wm->type);
        gf_close(&gf);
        return false;
    }
    e->bias.resize((size_t) e->out_dim);
    if (bm->type == GGML_TYPE_F32) {
        memcpy(e->bias.data(), bd, (size_t) e->out_dim * sizeof(float));
    } else if (bm->type == GGML_TYPE_BF16) {
        const uint16_t * p = (const uint16_t *) bd;
        for (int i = 0; i < e->out_dim; i++) {
            e->bias[(size_t) i] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &p[i]);
        }
    } else {
        fprintf(stderr, "[SA3-Seconds] FATAL: unsupported bias type %d\n", (int) bm->type);
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);
    fprintf(stderr, "[Load] SA3-Seconds: Linear(%d -> %d), range [%.0f, %.0f]\n", e->in_dim, e->out_dim, e->min_val,
            e->max_val);
    return true;
}

// seconds -> [out_dim] embedding (caller-allocated out).
static void sa3_seconds_embed(const SA3SecondsEmbedder & e, float seconds, float * out) {
    // clamp + normalize (NumberConditioner.forward)
    float t = seconds;
    if (t < e.min_val) {
        t = e.min_val;
    }
    if (t > e.max_val) {
        t = e.max_val;
    }
    t = (t - e.min_val) / (e.max_val - e.min_val);

    // Expo Fourier features. ramp/freqs mirror torch.linspace + exp in double,
    // then the arg product follows the source expression order in f32:
    // args = (t * freqs) * 2 * pi
    const int          half = e.in_dim / 2;
    std::vector<float> feat((size_t) e.in_dim);
    const double       log_min = log((double) e.min_freq);
    const double       log_max = log((double) e.max_freq);
    for (int i = 0; i < half; i++) {
        double ramp = (half > 1) ? (double) i / (double) (half - 1) : 0.0;
        float  freq = (float) exp(ramp * (log_max - log_min) + log_min);
        float  arg  = t * freq;
        arg         = arg * 2.0f;
        arg         = arg * (float) 3.14159265358979323846;
        feat[(size_t) i]        = cosf(arg);  // cos first
        feat[(size_t) (half + i)] = sinf(arg);  // then sin
    }

    // Linear(256 -> 768)
    for (int o = 0; o < e.out_dim; o++) {
        const float * w   = e.weight.data() + (size_t) o * e.in_dim;
        double        acc = e.bias[(size_t) o];
        for (int i = 0; i < e.in_dim; i++) {
            acc += (double) w[i] * (double) feat[(size_t) i];
        }
        out[o] = (float) acc;
    }
}
