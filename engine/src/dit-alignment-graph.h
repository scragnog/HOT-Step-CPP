#pragma once
// dit-alignment-graph.h: Modified DiT forward for cross-attention score extraction
//
// Builds a partial DiT graph (layers 0..max_layer) with manual cross-attention
// on target layers to materialize attention weights. Used by LRC Phase 3.

#include "alignment-config.h"
#include "dit-graph.h"
#include "dit.h"
#include "philox.h"

#include <cstring>
#include <string>
#include <vector>

// Build cross-attention with explicit score output for a target layer.
// Same as dit_ggml_build_cross_attn but splits the attention into
// Q*K^T -> softmax (tagged output) -> V multiply.
// Returns: {attn_output [H,S,N], scores_per_head vector}
struct CrossAttnWithScores {
    struct ggml_tensor *              attn_output;
    std::vector<struct ggml_tensor *> head_scores;  // [enc_S, S, 1, N] per target head
};

static CrossAttnWithScores dit_build_cross_attn_scored(
    struct ggml_context * ctx,
    DiTGGML *             m,
    DiTGGMLLayer *        ly,
    struct ggml_tensor *  norm_ca,
    struct ggml_tensor *  enc,
    struct ggml_tensor *  ca_mask,
    int                   S,
    int                   enc_S,
    int                   N,
    int                   layer_idx,
    const std::vector<int> & target_heads) {

    CrossAttnWithScores result;
    DiTGGMLConfig & c   = m->cfg;
    int             D   = c.head_dim;
    int             Nh  = c.n_heads;
    int             Nkv = c.n_kv_heads;

    // QKV projections (same logic as dit_ggml_build_cross_attn)
    int                 q_dim  = Nh * D;
    int                 kv_dim = Nkv * D;
    struct ggml_tensor *q, *k, *v;
    if (ly->ca_qkv) {
        struct ggml_tensor * w_q  = ggml_view_2d(ctx, ly->ca_qkv, ly->ca_qkv->ne[0], q_dim, ly->ca_qkv->nb[1], 0);
        struct ggml_tensor * w_kv = ggml_view_2d(ctx, ly->ca_qkv, ly->ca_qkv->ne[0], 2 * kv_dim, ly->ca_qkv->nb[1],
                                                 (size_t) q_dim * ly->ca_qkv->nb[1]);
        q                         = ggml_mul_mat(ctx, w_q, norm_ca);
        struct ggml_tensor * kv   = ggml_mul_mat(ctx, w_kv, enc);
        k = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], 0));
        v = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], (size_t) kv_dim * kv->nb[0]));
    } else if (ly->ca_kv) {
        q                       = dit_ggml_linear(ctx, ly->ca_q_proj, norm_ca);
        struct ggml_tensor * kv = ggml_mul_mat(ctx, ly->ca_kv, enc);
        k = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], 0));
        v = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], (size_t) kv_dim * kv->nb[0]));
    } else {
        q = dit_ggml_linear(ctx, ly->ca_q_proj, norm_ca);
        k = dit_ggml_linear(ctx, ly->ca_k_proj, enc);
        v = dit_ggml_linear(ctx, ly->ca_v_proj, enc);
    }

    // Reshape + permute to [D, seq, heads, N]
    q = ggml_permute(ctx, ggml_reshape_4d(ctx, q, D, Nh, S, N), 0, 2, 1, 3);
    k = ggml_permute(ctx, ggml_reshape_4d(ctx, k, D, Nkv, enc_S, N), 0, 2, 1, 3);
    v = ggml_permute(ctx, ggml_reshape_4d(ctx, v, D, Nkv, enc_S, N), 0, 2, 1, 3);

    // QK-norm
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_q_norm));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), dit_ggml_f32(ctx, ly->ca_k_norm));

    float scale = 1.0f / sqrtf((float) D);

    // Manual attention: compute full scores, tag target heads as outputs
    // Q: [D, S, Nh, N], K: [D, enc_S, Nkv, N]
    // scores = Q^T * K: [enc_S, S, Nh, N] (ggml mul_mat: [D,enc_S,Nkv,N]^T * [D,S,Nh,N])
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // [enc_S, S, Nh, N]
    scores = ggml_soft_max_ext(ctx, scores, ca_mask, scale, 0.0f);

    // Tag target heads as named outputs
    for (int head_idx : target_heads) {
        if (head_idx >= Nh) continue;
        // Extract single head: view [enc_S, S, 1, N] at offset head_idx
        size_t head_offset = (size_t) head_idx * scores->nb[2];
        struct ggml_tensor * head_scores = ggml_view_4d(ctx, scores,
            enc_S, S, 1, N,
            scores->nb[1], scores->nb[2], scores->nb[3],
            head_offset);
        head_scores = ggml_cont(ctx, head_scores);

        char name[64];
        snprintf(name, sizeof(name), "ca_L%d_H%d", layer_idx, head_idx);
        ggml_set_name(head_scores, name);
        ggml_set_output(head_scores);
        result.head_scores.push_back(head_scores);
    }

    // V multiply: attn_out = scores * V^T
    struct ggml_tensor * vt   = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * attn = ggml_mul_mat(ctx, vt, scores);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, N);

    result.attn_output = dit_ggml_linear(ctx, ly->ca_o_proj, attn);
    return result;
}

// Build a partial alignment layer: same as dit_ggml_build_layer but uses
// dit_build_cross_attn_scored for target layers.
static struct ggml_tensor * dit_build_alignment_layer(
    struct ggml_context *       ctx,
    DiTGGML *                   m,
    int                         layer_idx,
    struct ggml_tensor *        hidden,
    struct ggml_tensor *        tproj,
    struct ggml_tensor *        enc,
    struct ggml_tensor *        positions,
    struct ggml_tensor *        sa_mask,
    struct ggml_tensor *        ca_mask,
    int S, int enc_S, int N,
    const std::vector<int> *    target_heads,  // NULL = not a target layer
    std::vector<struct ggml_tensor *> & score_outputs) {

    DiTGGMLConfig & c  = m->cfg;
    DiTGGMLLayer *  ly = &m->layers[layer_idx];
    int             H  = c.hidden_size;

    // AdaLN
    struct ggml_tensor * ss = ly->scale_shift_table;
    if (ss->type != GGML_TYPE_F32) ss = ggml_cast(ctx, ss, GGML_TYPE_F32);
    struct ggml_tensor * ss_flat = ggml_reshape_1d(ctx, ss, 6 * H);
    struct ggml_tensor * adaln   = ggml_add(ctx, ss_flat, tproj);

    size_t Hb = H * sizeof(float);
    struct ggml_tensor * shift_sa  = ggml_view_1d(ctx, adaln, H, 0 * Hb);
    struct ggml_tensor * scale_sa  = ggml_view_1d(ctx, adaln, H, 1 * Hb);
    struct ggml_tensor * gate_sa   = ggml_view_1d(ctx, adaln, H, 2 * Hb);
    struct ggml_tensor * shift_ffn = ggml_view_1d(ctx, adaln, H, 3 * Hb);
    struct ggml_tensor * scale_ffn = ggml_view_1d(ctx, adaln, H, 4 * Hb);
    struct ggml_tensor * gate_ffn  = ggml_view_1d(ctx, adaln, H, 5 * Hb);

    // Self-attention
    struct ggml_tensor * residual = hidden;
    struct ggml_tensor * norm_sa  = dit_ggml_rms_norm_weighted(ctx, hidden, ly->self_attn_norm, c.rms_norm_eps);
    norm_sa = dit_ggml_adaln(ctx, norm_sa, scale_sa, shift_sa, m->scalar_one);
    struct ggml_tensor * sa_out = dit_ggml_build_self_attn(ctx, m, ly, norm_sa, positions, sa_mask, S, N, -1, nullptr);
    hidden = dit_ggml_gated_add(ctx, residual, sa_out, gate_sa);

    // Cross-attention
    if (enc) {
        struct ggml_tensor * norm_ca = dit_ggml_rms_norm_weighted(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);

        if (target_heads && !target_heads->empty()) {
            // Scored cross-attention: extracts attention weights for target heads
            auto ca_result = dit_build_cross_attn_scored(ctx, m, ly, norm_ca, enc, ca_mask,
                                                          S, enc_S, N, layer_idx, *target_heads);
            hidden = ggml_add(ctx, hidden, ca_result.attn_output);
            for (auto * s : ca_result.head_scores) {
                score_outputs.push_back(s);
            }
        } else {
            // Normal cross-attention (non-target layers)
            struct ggml_tensor * ca_out = dit_ggml_build_cross_attn(ctx, m, ly, norm_ca, enc, positions, ca_mask, S, enc_S, N);
            hidden = ggml_add(ctx, hidden, ca_out);
        }
    }

    // FFN
    residual = hidden;
    struct ggml_tensor * norm_ffn = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    norm_ffn = dit_ggml_adaln(ctx, norm_ffn, scale_ffn, shift_ffn, m->scalar_one);
    struct ggml_tensor * ffn_out = dit_ggml_build_mlp(ctx, m, ly, norm_ffn, S);
    hidden = dit_ggml_gated_add(ctx, residual, ffn_out, gate_ffn);

    return hidden;
}

// Extract cross-attention alignment scores from the DiT.
// Builds a partial graph (layers 0..max_layer), runs one forward pass at t_last,
// reads attention scores for configured layer/head pairs.
//
// out_scores: [total_heads, enc_S, S] f32, caller-allocated.
// Returns 0 on success, -1 on error.
static int dit_alignment_extract(
    DiTGGML *              dit,
    const AlignmentConfig & cfg,
    const float * output,      // [T * Oc] DiT output (pred_latent), batch 0 only
    const float * context,     // [T * ctx_ch] context latents, batch 0 only
    const float * enc_hidden,  // [enc_S * H_cond] encoder hidden states, batch 0 only
    int T, int S, int enc_S,
    int num_steps,
    float * out_scores) {

    if (!cfg.valid || cfg.total_heads <= 0) {
        fprintf(stderr, "[Align-Extract] ERROR: invalid alignment config\n");
        return -1;
    }

    DiTGGMLConfig & c = dit->cfg;
    int P   = c.patch_size;
    int N   = 1;  // alignment always batch=1
    int Oc  = c.out_channels;
    int Ic  = c.in_channels;

    // Compute t_last and xt
    float t_last = 1.0f / (float) num_steps;

    // Generate alignment noise (seed 42, matching Python)
    std::vector<float> align_noise(Oc * T);
    philox_randn(42, align_noise.data(), Oc * T, true);

    // xt = t_last * noise + (1 - t_last) * output
    std::vector<float> xt(Oc * T);
    for (int i = 0; i < Oc * T; i++) {
        xt[i] = t_last * align_noise[i] + (1.0f - t_last) * output[i];
    }

    // Build input_latents: [in_channels, T, 1] = concat(context, xt)
    int ctx_ch = Ic - Oc;  // 64 typically
    std::vector<float> input_latents(Ic * T);
    for (int t = 0; t < T; t++) {
        // context channels first, then xt channels
        memcpy(input_latents.data() + t * Ic, context + t * ctx_ch, ctx_ch * sizeof(float));
        memcpy(input_latents.data() + t * Ic + ctx_ch, xt.data() + t * Oc, Oc * sizeof(float));
    }

    // Build alignment graph
    size_t ctx_size = ggml_tensor_overhead() * 4096 + 256 * 1024 * 1024;
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        fprintf(stderr, "[Align-Extract] ERROR: ggml_init failed\n");
        return -1;
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    // Input tensors
    struct ggml_tensor * t_input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Ic, T, N);
    ggml_set_name(t_input, "input_latents"); ggml_set_input(t_input);

    int H_enc = (int) dit->cond_emb_w->ne[0];
    struct ggml_tensor * t_enc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H_enc, enc_S, N);
    ggml_set_name(t_enc, "enc_hidden"); ggml_set_input(t_enc);

    struct ggml_tensor * t_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t_t, "t"); ggml_set_input(t_t);

    struct ggml_tensor * t_tr = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t_tr, "t_r"); ggml_set_input(t_tr);

    struct ggml_tensor * t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S * N);
    ggml_set_name(t_pos, "positions"); ggml_set_input(t_pos);

    struct ggml_tensor * sa_mask_sw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, N);
    ggml_set_name(sa_mask_sw, "sa_mask_sw"); ggml_set_input(sa_mask_sw);

    struct ggml_tensor * sa_mask_pad = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, N);
    ggml_set_name(sa_mask_pad, "sa_mask_pad"); ggml_set_input(sa_mask_pad);

    struct ggml_tensor * ca_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, enc_S, S, 1, N);
    ggml_set_name(ca_mask, "ca_mask"); ggml_set_input(ca_mask);

    // Timestep embeddings
    struct ggml_tensor * tproj, * temb;
    {
        struct ggml_tensor * tproj_t;
        struct ggml_tensor * temb_t = dit_ggml_build_temb(ctx, &dit->time_embed, t_t, &tproj_t, "_t");
        struct ggml_tensor * t_diff = ggml_sub(ctx, t_t, t_tr);
        struct ggml_tensor * tproj_r;
        struct ggml_tensor * temb_r = dit_ggml_build_temb(ctx, &dit->time_embed_r, t_diff, &tproj_r, "_r");
        temb  = ggml_add(ctx, temb_t, temb_r);
        tproj = ggml_add(ctx, tproj_t, tproj_r);
    }

    // proj_in
    struct ggml_tensor * patched = ggml_reshape_3d(ctx, t_input, Ic * P, S, N);
    struct ggml_tensor * hidden  = dit_ggml_linear_bias(ctx, dit->proj_in_w, dit->proj_in_b, patched);

    // Condition embedder
    struct ggml_tensor * enc = dit_ggml_linear_bias(ctx, dit->cond_emb_w, dit->cond_emb_b, t_enc);

    // Build target layer lookup
    std::vector<std::vector<int>> layer_heads(cfg.max_layer + 1);
    for (const auto & lh : cfg.targets) {
        if (lh.layer <= cfg.max_layer) {
            layer_heads[lh.layer] = lh.heads;
        }
    }

    // Layers 0..max_layer (early exit)
    std::vector<struct ggml_tensor *> score_outputs;
    for (int i = 0; i <= cfg.max_layer && i < c.n_layers; i++) {
        struct ggml_tensor * sa_mask = (dit->layers[i].layer_type == 0) ? sa_mask_sw : sa_mask_pad;
        const std::vector<int> * heads = (i < (int) layer_heads.size() && !layer_heads[i].empty())
                                         ? &layer_heads[i] : nullptr;
        hidden = dit_build_alignment_layer(ctx, dit, i, hidden, tproj, enc, t_pos,
                                           sa_mask, ca_mask, S, enc_S, N,
                                           heads, score_outputs);
    }

    // Build the graph from all score outputs
    for (auto * s : score_outputs) {
        ggml_build_forward_expand(gf, s);
    }
    // Also need hidden to be computed (drives dependencies)
    ggml_build_forward_expand(gf, hidden);

    // Allocate and compute
    ggml_backend_sched_reset(dit->sched);
    if (!ggml_backend_sched_alloc_graph(dit->sched, gf)) {
        fprintf(stderr, "[Align-Extract] ERROR: graph alloc failed\n");
        ggml_free(ctx);
        return -1;
    }

    // Set input data
    ggml_backend_tensor_set(t_input, input_latents.data(), 0, Ic * T * sizeof(float));
    ggml_backend_tensor_set(t_enc, enc_hidden, 0, H_enc * enc_S * sizeof(float));
    ggml_backend_tensor_set(t_t, &t_last, 0, sizeof(float));
    ggml_backend_tensor_set(t_tr, &t_last, 0, sizeof(float));  // t_r = t for alignment

    // Positions: 0..S-1
    std::vector<int> pos_data(S);
    for (int i = 0; i < S; i++) pos_data[i] = i;
    ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * sizeof(int));

    // Masks: all zeros (no padding for single sample)
    std::vector<uint16_t> zero_sa(S * S, 0);
    ggml_backend_tensor_set(sa_mask_sw, zero_sa.data(), 0, S * S * sizeof(uint16_t));
    ggml_backend_tensor_set(sa_mask_pad, zero_sa.data(), 0, S * S * sizeof(uint16_t));
    std::vector<uint16_t> zero_ca(enc_S * S, 0);
    ggml_backend_tensor_set(ca_mask, zero_ca.data(), 0, enc_S * S * sizeof(uint16_t));

    // Compute
    Timer timer;
    ggml_backend_sched_graph_compute(dit->sched, gf);
    fprintf(stderr, "[Align-Extract] Graph compute: %.1f ms (%d layers)\n", timer.ms(), cfg.max_layer + 1);

    // Read back scores
    int score_idx = 0;
    for (const auto & lh : cfg.targets) {
        for (int head : lh.heads) {
            char name[64];
            snprintf(name, sizeof(name), "ca_L%d_H%d", lh.layer, head);
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (!t) {
                fprintf(stderr, "[Align-Extract] WARNING: tensor '%s' not found\n", name);
                memset(out_scores + score_idx * enc_S * S, 0, enc_S * S * sizeof(float));
            } else {
                // Tensor is [enc_S, S, 1, N] — read as [enc_S * S] floats
                // We need [S, enc_S] (transposed: frames × tokens) but Python transposes
                // at a later stage. Store as [enc_S, S] here.
                ggml_backend_tensor_get(t, out_scores + score_idx * enc_S * S, 0, enc_S * S * sizeof(float));
            }
            score_idx++;
        }
    }

    ggml_free(ctx);
    fprintf(stderr, "[Align-Extract] Extracted %d/%d head scores\n", score_idx, cfg.total_heads);
    return 0;
}
