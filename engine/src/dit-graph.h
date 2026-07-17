#pragma once
// dit-graph.h: DiT compute graph construction (ggml)
//
// Graph builders: timestep embedding, self-attention, cross-attention,
// MLP, per-layer composition, and full N-step diffusion graph.
//
// ggml tensor layout reminder:
//   [S, H] in math = ne[0]=H, ne[1]=S in ggml
//   [Nh, S, D] in math = ne[0]=D, ne[1]=S, ne[2]=Nh in ggml

#include "dit.h"

#include <cmath>
#include <cstdlib>

// Helper: ensure tensor is f32 (cast if bf16/f16)
static struct ggml_tensor * dit_ggml_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return t;
    }
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

// Helper: RMSNorm + weight multiply
static struct ggml_tensor * dit_ggml_rms_norm_weighted(struct ggml_context * ctx,
                                                       struct ggml_tensor *  x,       // [H, S]
                                                       struct ggml_tensor *  weight,  // [H]
                                                       float                 eps) {
    struct ggml_tensor * norm = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, norm, dit_ggml_f32(ctx, weight));
}

// Helper: Linear layer (no bias)
// weight: [in, out] in ggml (= [out, in] in PyTorch)
// input:  [in, S]
// output: [out, S]
static struct ggml_tensor * dit_ggml_linear(struct ggml_context * ctx,
                                            struct ggml_tensor *  weight,
                                            struct ggml_tensor *  input) {
    return ggml_mul_mat(ctx, weight, input);
}

// ─── HOT-Step ConvRot: group-wise Hadamard rotation of activations ───
// For weights stored pre-rotated (W' = W·H per input-dim group of size g),
// apply the matching rotation to the activation: rot(x)[grp] = H·x[grp].
// H is symmetric orthogonal, so W'·rot(x) == W·x. g == 0 → passthrough.
// x: [in, ...] with in % g == 0 (guaranteed by the offline converter).
// LoRA/adapter deltas are in UNROTATED space — they must keep consuming the
// raw input, never the rotated one.
static struct ggml_tensor * dit_convrot_rotate(struct ggml_context * ctx,
                                               DiTGGML *             m,
                                               struct ggml_tensor *  x,
                                               int                   g) {
    if (g <= 0) {
        return x;
    }
    auto it = m->convrot.hmats.find(g);
    if (it == m->convrot.hmats.end()) {
        return x;  // unreachable: load fails on unknown group sizes
    }
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    const int64_t        cols = ggml_nelements(x) / g;
    struct ggml_tensor * xg   = ggml_reshape_2d(ctx, x, g, cols);
    struct ggml_tensor * xr   = ggml_mul_mat(ctx, it->second, xg);
    return ggml_reshape_4d(ctx, xr, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

// Apply a slot's runtime adapter components onto y: the optional full-size
// delta (summed deltas, basin re-base correction, Conv1d fallbacks) plus any
// lowrank factor units. LoRA units add B@(A@x) — batched dims ride mul_mat
// directly. LoKr units apply (w1 ⊗ w2)@x via the Kronecker identity
// (HOTSTEP_KRON_TEST-validated choreography, docs/plans/lowrank-runtime-
// adapters.md §8) — trailing dims flatten into the column axis and reshape
// back. Everything consumes the RAW input (unrotated space, like full deltas).
static struct ggml_tensor * dit_lora_apply_units(struct ggml_context * ctx,
                                                 const DiTLoRADelta *  sd,
                                                 struct ggml_tensor *  y,
                                                 struct ggml_tensor *  input) {
    if (!sd) {
        return y;
    }
    if (sd->delta) {
        y = ggml_add(ctx, y, ggml_mul_mat(ctx, sd->delta, input));
    }
    if (sd->units.empty()) {
        return y;
    }
    struct ggml_tensor * xin  = ggml_is_contiguous(input) ? input : ggml_cont(ctx, input);
    const int64_t        cols = ggml_nelements(input) / input->ne[0];
    for (const DiTLoRAFactorUnit & u : sd->units) {
        struct ggml_tensor * dy = nullptr;
        if (u.a && u.b) {
            dy = ggml_mul_mat(ctx, u.b, ggml_mul_mat(ctx, u.a, xin));
        } else if (u.k1 && u.k2) {
            const int64_t        bb = u.k1->ne[0], aa = u.k1->ne[1];
            const int64_t        dd = u.k2->ne[0], cc = u.k2->ne[1];
            struct ggml_tensor * X2 = ggml_reshape_2d(ctx, xin, dd, bb * cols);
            struct ggml_tensor * T3 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, u.k2, X2), cc, bb, cols);
            struct ggml_tensor * P2 =
                ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, T3, 1, 0, 2, 3)), bb, cc * cols);
            struct ggml_tensor * Y3 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, u.k1, P2), aa, cc, cols);
            struct ggml_tensor * YP = ggml_cont(ctx, ggml_permute(ctx, Y3, 1, 0, 2, 3));  // [cc, aa, cols]
            dy = ggml_reshape_4d(ctx, YP, aa * cc, input->ne[1], input->ne[2], input->ne[3]);
        }
        if (dy) {
            y = ggml_add(ctx, y, dy);
        }
    }
    return y;
}

// Helper: Linear layer with runtime LoRA slot (full delta and/or factor units)
// y = W@x + slot components  (sd NULL → plain linear)
// base_input: ConvRot-rotated input for the base weight (NULL → use input);
// adapter components always consume the raw input (they are unrotated).
static struct ggml_tensor * dit_ggml_linear_lora(struct ggml_context * ctx,
                                                 struct ggml_tensor *  weight,
                                                 const DiTLoRADelta *  sd,
                                                 struct ggml_tensor *  input,
                                                 struct ggml_tensor *  base_input = nullptr) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, weight, base_input ? base_input : input);
    return dit_lora_apply_units(ctx, sd, y, input);
}

// Helper: Linear layer with bias and runtime LoRA slot
static struct ggml_tensor * dit_ggml_linear_bias_lora(struct ggml_context * ctx,
                                                      struct ggml_tensor *  weight,
                                                      const DiTLoRADelta *  sd,
                                                      struct ggml_tensor *  bias,
                                                      struct ggml_tensor *  input,
                                                      struct ggml_tensor *  base_input = nullptr) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, base_input ? base_input : input);
    out                      = dit_lora_apply_units(ctx, sd, out, input);
    return ggml_add(ctx, out, dit_ggml_f32(ctx, bias));
}

// ─── Per-section adapter masking (regional LoRA) ───
//
// When active, each adapter's delta is applied SEPARATELY (not summed) and gated:
//  - frame-indexed projections (self-attn q/k/v/o, MLP, cross-attn q/o, proj_in):
//    multiplied by that adapter's per-frame mask `masks[i]` ([1,S,1]).
//  - token/global projections (cross-attn k/v, cond_emb): scaled by that adapter's
//    scalar mean section weight `means[i]` (a per-frame mask can't apply — wrong axis).
// See docs/plans/per-section-adapter-masking.md. `loras`/`masks`/`means` are all
// indexed by adapter (same order as g_hotstep_params.adapters).
struct DiTLoRASectionCtx {
    const std::vector<DiTLoRA> *              loras = nullptr;
    const std::vector<struct ggml_tensor *> * masks = nullptr;
    const std::vector<float> *                means = nullptr;
};

// Debug toggles to isolate the per-section path:
//   HOTSTEP_SECTION_NOMASK — apply every section adapter's delta UNMASKED (full
//     strength, no per-frame gating). If adapters become audible, the mask
//     multiply is the culprit; if still nil, the separate-delta wiring is.
static const bool g_hotstep_section_nomask = (std::getenv("HOTSTEP_SECTION_NOMASK") != nullptr);

// Apply a per-layer projection with LoRA. Summed path (sect==nullptr): y = W@x
// + the slot's components (full delta and/or lowrank factor units). Section
// path: y = W@x + Σ_i gate_i·(delta_i@x), gate = mask (frame-indexed) or mean
// scalar (otherwise). Sections never carry factor units — the engine forces
// plain runtime mode on the per-section path.
static struct ggml_tensor * dit_lora_apply_layer(struct ggml_context *      ctx,
                                                 struct ggml_tensor *       weight,
                                                 struct ggml_tensor *       input,
                                                 const DiTLoRADelta *       single_sd,
                                                 const DiTLoRASectionCtx *  sect,
                                                 int                        layer_idx,
                                                 DiTLoRADelta DiTLoRALayer::* slot,
                                                 bool                       frame_masked,
                                                 struct ggml_tensor *       base_input = nullptr) {
    // base_input: ConvRot-rotated input for the base weight (NULL → input).
    // All adapter deltas below stay on the raw input (unrotated space).
    struct ggml_tensor * y = ggml_mul_mat(ctx, weight, base_input ? base_input : input);
    if (sect && sect->loras) {
        for (size_t i = 0; i < sect->loras->size(); i++) {
            const DiTLoRA & lr = (*sect->loras)[i];
            if (!lr.active || layer_idx < 0 || layer_idx >= DIT_LORA_MAX_LAYERS) continue;
            struct ggml_tensor * d = (lr.layers[layer_idx].*slot).delta;
            if (!d) continue;
            struct ggml_tensor * dy = ggml_mul_mat(ctx, d, input);
            if (g_hotstep_section_nomask) {
                // debug: apply unmasked (full strength)
            } else if (frame_masked) {
                if (sect->masks && i < sect->masks->size() && (*sect->masks)[i])
                    dy = ggml_mul(ctx, dy, (*sect->masks)[i]);
            } else if (sect->means && i < sect->means->size()) {
                dy = ggml_scale(ctx, dy, (*sect->means)[i]);
            }
            y = ggml_add(ctx, y, dy);
        }
    } else if (single_sd) {
        y = dit_lora_apply_units(ctx, single_sd, y, input);
    }
    return y;
}

// Apply a global (non-layer) projection delta onto an already-computed base_y.
// slot is a DiTLoRA member (proj_in / cond_emb). frame_masked as above.
static struct ggml_tensor * dit_lora_apply_global(struct ggml_context *      ctx,
                                                  struct ggml_tensor *       base_y,
                                                  struct ggml_tensor *       input,
                                                  const DiTLoRADelta *       single_sd,
                                                  const DiTLoRASectionCtx *  sect,
                                                  DiTLoRADelta DiTLoRA::*     slot,
                                                  bool                       frame_masked) {
    struct ggml_tensor * y = base_y;
    if (sect && sect->loras) {
        for (size_t i = 0; i < sect->loras->size(); i++) {
            const DiTLoRA & lr = (*sect->loras)[i];
            if (!lr.active) continue;
            struct ggml_tensor * d = (lr.*slot).delta;
            if (!d) continue;
            struct ggml_tensor * dy = ggml_mul_mat(ctx, d, input);
            if (g_hotstep_section_nomask) {
                // debug: apply unmasked (full strength)
            } else if (frame_masked) {
                if (sect->masks && i < sect->masks->size() && (*sect->masks)[i])
                    dy = ggml_mul(ctx, dy, (*sect->masks)[i]);
            } else if (sect->means && i < sect->means->size()) {
                dy = ggml_scale(ctx, dy, (*sect->means)[i]);
            }
            y = ggml_add(ctx, y, dy);
        }
    } else if (single_sd) {
        y = dit_lora_apply_units(ctx, single_sd, y, input);
    }
    return y;
}

// Helper: Linear layer with bias
static struct ggml_tensor * dit_ggml_linear_bias(struct ggml_context * ctx,
                                                 struct ggml_tensor *  weight,
                                                 struct ggml_tensor *  bias,
                                                 struct ggml_tensor *  input) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    return ggml_add(ctx, out, dit_ggml_f32(ctx, bias));
}

// Helper: AdaLN modulate
// out = norm * (1 + scale) + shift
// norm: [H, S], scale: [H], shift: [H]
static struct ggml_tensor * dit_ggml_adaln(struct ggml_context * ctx,
                                           struct ggml_tensor *  norm,
                                           struct ggml_tensor *  scale,
                                           struct ggml_tensor *  shift,
                                           struct ggml_tensor *  one) {
    // norm * (1 + scale) + shift
    // one is [1] = 1.0, broadcasts to [H]; avoids expensive [H,S,N] add
    struct ggml_tensor * one_plus_s = ggml_add(ctx, scale, one);        // [H] + [1] -> [H]
    struct ggml_tensor * scaled     = ggml_mul(ctx, norm, one_plus_s);  // [H,S,N]
    return ggml_add(ctx, scaled, shift);                                // [H,S,N]
}

// Helper: Gated residual
// out = residual + x * gate
// residual: [H, S], x: [H, S], gate: [H]
// NOTE: no sigmoid, gate is a raw scaling factor (matches Python reference)
static struct ggml_tensor * dit_ggml_gated_add(struct ggml_context * ctx,
                                               struct ggml_tensor *  residual,
                                               struct ggml_tensor *  x,
                                               struct ggml_tensor *  gate) {
    struct ggml_tensor * gated = ggml_mul(ctx, x, gate);  // broadcast [H] over [H,S]
    return ggml_add(ctx, residual, gated);
}

// Build timestep embedding subgraph
// t_scalar: [1] f32, returns temb [H] and *out_tproj [6H]
// suffix: "_t" or "_r" for naming intermediate tensors
static struct ggml_tensor * dit_ggml_build_temb(struct ggml_context * ctx,
                                                DiTGGML *             m,
                                                DiTGGMLTembWeights *  w,
                                                struct ggml_tensor *  t_scalar,
                                                struct ggml_tensor ** out_tproj,
                                                DiTLoRADelta *        delta_lin1,
                                                DiTLoRADelta *        delta_lin2,
                                                DiTLoRADelta *        delta_proj,
                                                const char *          suffix = "") {
    // scale timestep by 1000 (diffusion convention, matches Python)
    struct ggml_tensor * t_scaled = ggml_scale(ctx, t_scalar, 1000.0f);

    // sinusoidal embedding: [1] -> [256]
    struct ggml_tensor * sinusoidal = ggml_timestep_embedding(ctx, t_scaled, 256, 10000);
    {
        char name[64];
        snprintf(name, sizeof(name), "sinusoidal%s", suffix);
        ggml_set_name(sinusoidal, name);
        ggml_set_output(sinusoidal);
    }

    // linear1 + silu: [256] -> [H]
    struct ggml_tensor * h = dit_ggml_linear_bias_lora(ctx, w->linear_1_w, delta_lin1, w->linear_1_b, sinusoidal,
                                                       dit_convrot_rotate(ctx, m, sinusoidal, w->rot_lin1));
    {
        char name[64];
        snprintf(name, sizeof(name), "temb_lin1%s", suffix);
        ggml_set_name(h, name);
        ggml_set_output(h);
    }

    h = ggml_silu(ctx, h);

    // linear2: [H] -> [H]
    struct ggml_tensor * temb = dit_ggml_linear_bias_lora(ctx, w->linear_2_w, delta_lin2, w->linear_2_b, h,
                                                          dit_convrot_rotate(ctx, m, h, w->rot_lin2));

    // silu + proj: [H] -> [6H]
    struct ggml_tensor * h2 = ggml_silu(ctx, temb);
    *out_tproj              = dit_ggml_linear_bias_lora(ctx, w->time_proj_w, delta_proj, w->time_proj_b, h2,
                                                        dit_convrot_rotate(ctx, m, h2, w->rot_proj));

    return temb;  // [H] (used for output adaln)
}

// F32 manual attention (fallback when flash_attn_ext is not available or imprecise).
// Q: [D, S, Nh], K: [D, S_kv, Nkv], V: [D, S_kv, Nkv]
// mask: [S_kv, S] F16 or NULL, scale: 1/sqrt(D)
// Returns: [D, Nh, S] (same layout as flash_attn_ext output)
static struct ggml_tensor * dit_attn_f32(struct ggml_context * ctx,
                                         struct ggml_tensor *  q,
                                         struct ggml_tensor *  k,
                                         struct ggml_tensor *  v,
                                         struct ggml_tensor *  mask,
                                         float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// Build self-attention sub-graph for a single layer.
// norm_sa: [H, S, N] pre-normalized + AdaLN-modulated hidden state
// Returns: output [H, S, N] (self-attention output, NOT added to residual yet)
static struct ggml_tensor * dit_ggml_build_self_attn(
    struct ggml_context * ctx,
    DiTGGML *             m,
    DiTGGMLLayer *        ly,
    struct ggml_tensor *  norm_sa,    // [H, S, N] pre-normalized + AdaLN-modulated
    struct ggml_tensor *  positions,  // [S*N] int32 position indices for RoPE
    struct ggml_tensor *  mask,       // [S, S] or NULL (sliding window mask)
    int                   S,
    int                   N,
    int                   layer_idx = -1,
    DiTLoRALayer *        lora      = nullptr,
    const DiTLoRASectionCtx * sect  = nullptr) {
    DiTGGMLConfig & c   = m->cfg;
    int             D   = c.head_dim;
    int             Nh  = c.n_heads;
    int             Nkv = c.n_kv_heads;

    // 1) QKV projections (full fused, QK partial, separate)
    // ConvRot: q/k/v share one rotated input; adapter deltas keep raw norm_sa.
    struct ggml_tensor * norm_sa_rot = dit_convrot_rotate(ctx, m, norm_sa, ly->rot_sa);
    struct ggml_tensor *q, *k, *v;
    int                 q_dim  = Nh * D;
    int                 kv_dim = Nkv * D;
    if (ly->sa_qkv) {
        struct ggml_tensor * qkv = dit_ggml_linear(ctx, ly->sa_qkv, norm_sa_rot);
        q                        = ggml_cont(ctx, ggml_view_3d(ctx, qkv, q_dim, S, N, qkv->nb[1], qkv->nb[2], 0));
        k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, kv_dim, S, N, qkv->nb[1], qkv->nb[2], (size_t) q_dim * qkv->nb[0]));
        v = ggml_cont(
            ctx, ggml_view_3d(ctx, qkv, kv_dim, S, N, qkv->nb[1], qkv->nb[2], (size_t) (q_dim + kv_dim) * qkv->nb[0]));
    } else if (ly->sa_qk) {
        struct ggml_tensor * qk = dit_ggml_linear(ctx, ly->sa_qk, norm_sa_rot);
        q                       = ggml_cont(ctx, ggml_view_3d(ctx, qk, q_dim, S, N, qk->nb[1], qk->nb[2], 0));
        k = ggml_cont(ctx, ggml_view_3d(ctx, qk, kv_dim, S, N, qk->nb[1], qk->nb[2], (size_t) q_dim * qk->nb[0]));
        v = dit_ggml_linear(ctx, ly->sa_v_proj, norm_sa_rot);
    } else {
        q = dit_lora_apply_layer(ctx, ly->sa_q_proj, norm_sa, lora ? &lora->sa_q : nullptr, sect, layer_idx, &DiTLoRALayer::sa_q, true, norm_sa_rot);
        k = dit_lora_apply_layer(ctx, ly->sa_k_proj, norm_sa, lora ? &lora->sa_k : nullptr, sect, layer_idx, &DiTLoRALayer::sa_k, true, norm_sa_rot);
        v = dit_lora_apply_layer(ctx, ly->sa_v_proj, norm_sa, lora ? &lora->sa_v : nullptr, sect, layer_idx, &DiTLoRALayer::sa_v, true, norm_sa_rot);
    }

    // 2) Reshape to heads: [Nh*D, S, N] -> [D, Nh, S, N]
    //    Rope merges S*N then restores 4D. Permute to flash_attn layout after rope.
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);
    v = ggml_reshape_4d(ctx, v, D, Nkv, S, N);

    // 4) QK-Norm: per-head RMSNorm on D dimension
    //    [D, Nh, S] rms_norm operates on ne[0]=D
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, dit_ggml_f32(ctx, ly->sa_q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, dit_ggml_f32(ctx, ly->sa_k_norm));

    // 5) RoPE (bidirectional, sequential positions)
    //    ggml_rope_ext asserts ne[2] == positions.ne[0].
    //    With batch N>1, positions has S*N elements (repeated [0..S-1] per batch).
    //    Merge S and N before rope, then restore 4D after.
    q = ggml_reshape_3d(ctx, q, D, Nh, S * N);
    k = ggml_reshape_3d(ctx, k, D, Nkv, S * N);
    q = ggml_rope_ext(ctx, q, positions, NULL, D, 2 /*mode=NEOX*/, 0 /*n_ctx_orig*/, c.rope_theta, 1.0f /*freq_scale*/,
                      0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    k = ggml_reshape_4d(ctx, k, D, Nkv, S, N);

    if (layer_idx == 0) {
        ggml_set_name(q, "layer0_q_after_rope");
        ggml_set_output(q);
        ggml_set_name(k, "layer0_k_after_rope");
        ggml_set_output(k);
    }

    // 6) Permute for flash_attn_ext: [D, Nh, S, N] -> [D, S, Nh, N]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // 7) Attention (flash on GPU, F32 manual on CPU)
    //    Q[D, S, Nh, N], K[D, S, Nkv, N], V[D, S, Nkv, N]
    float scale = 1.0f / sqrtf((float) D);

    // K/V come in F32 from mul_mat (no KV cache here). Cast to F16 before FA,
    // mirroring llama.cpp build_attn_mha for graphs without a KV cache.
    if (m->use_flash_attn) {
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
        }
        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }
    }

    struct ggml_tensor * attn = m->use_flash_attn ? ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f) :
                                                    dit_attn_f32(ctx, q, k, v, mask, scale);
    if (m->use_flash_attn) {
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    }

    // Both return [D, Nh, S, N]
    // Reshape: [D, Nh, S, N] -> [D*Nh, S, N] = [H, S, N]
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, N);

    if (layer_idx == 0) {
        ggml_set_name(attn, "layer0_attn_out");
        ggml_set_output(attn);
    }

    // 8) O projection: [Nh*D, S, N] -> [H, S, N]
    struct ggml_tensor * out = dit_lora_apply_layer(ctx, ly->sa_o_proj, attn, lora ? &lora->sa_o : nullptr, sect, layer_idx, &DiTLoRALayer::sa_o, true,
                                                    dit_convrot_rotate(ctx, m, attn, ly->rot_sa_o));
    return out;
}

// Build MLP sub-graph: SwiGLU
// norm_ffn: [H, S, N] pre-normalized + AdaLN-modulated hidden state
// Returns: output [H, S, N]
static struct ggml_tensor * dit_ggml_build_mlp(struct ggml_context * ctx,
                                               DiTGGML *             m,
                                               DiTGGMLLayer *        ly,
                                               struct ggml_tensor *  norm_ffn,
                                               int                   S,
                                               DiTLoRALayer *        lora = nullptr,
                                               int                   layer_idx = -1,
                                               const DiTLoRASectionCtx * sect = nullptr) {
    struct ggml_tensor * ff;
    struct ggml_tensor * norm_ffn_rot = dit_convrot_rotate(ctx, m, norm_ffn, ly->rot_mlp);
    if (ly->gate_up) {
        // Fused: single matmul [H, 2*I] x [H, S, N] -> [2*I, S, N], then swiglu splits ne[0]
        struct ggml_tensor * gu = dit_ggml_linear(ctx, ly->gate_up, norm_ffn_rot);
        ff                      = ggml_swiglu(ctx, gu);
    } else {
        // Separate: two matmuls + split swiglu
        struct ggml_tensor * gate = dit_lora_apply_layer(ctx, ly->gate_proj, norm_ffn, lora ? &lora->gate : nullptr, sect, layer_idx, &DiTLoRALayer::gate, true, norm_ffn_rot);
        struct ggml_tensor * up   = dit_lora_apply_layer(ctx, ly->up_proj, norm_ffn, lora ? &lora->up : nullptr, sect, layer_idx, &DiTLoRALayer::up, true, norm_ffn_rot);
        ff                        = ggml_swiglu_split(ctx, gate, up);
    }

    // Down projection: [I, S] -> [H, S]
    return dit_lora_apply_layer(ctx, ly->down_proj, ff, lora ? &lora->down : nullptr, sect, layer_idx, &DiTLoRALayer::down, true,
                                dit_convrot_rotate(ctx, m, ff, ly->rot_down));
}

// Build cross-attention sub-graph for a single layer.
// norm_ca: [H, S, N] pre-normalized hidden state (Q source)
// enc:     [H, enc_S, N] condition-embedded encoder states (K/V source)
// Returns: output [H, S, N] (NOT added to residual yet)
static struct ggml_tensor * dit_ggml_build_cross_attn(struct ggml_context * ctx,
                                                      DiTGGML *             m,
                                                      DiTGGMLLayer *        ly,
                                                      struct ggml_tensor *  norm_ca,    // [H, S, N]
                                                      struct ggml_tensor *  enc,        // [H, enc_S, N]
                                                      struct ggml_tensor *  positions,  // unused, kept for consistency
                                                      struct ggml_tensor *  mask,       // [enc_S, S, 1, N] F16 or NULL
                                                      int                   S,
                                                      int                   enc_S,
                                                      int                   N,
                                                      DiTLoRALayer *        lora = nullptr,
                                                      int                   layer_idx = -1,
                                                      const DiTLoRASectionCtx * sect = nullptr) {
    DiTGGMLConfig & c   = m->cfg;
    int             D   = c.head_dim;
    int             Nh  = c.n_heads;
    int             Nkv = c.n_kv_heads;

    (void) positions;  // cross-attn has no RoPE

    // Q from hidden, KV from encoder (full fused, Q+KV partial, separate)
    // ConvRot: rotate each input for its rotated weight; deltas keep raw inputs.
    struct ggml_tensor * norm_ca_rot = dit_convrot_rotate(ctx, m, norm_ca, ly->rot_ca_q);
    struct ggml_tensor * enc_rot     = dit_convrot_rotate(ctx, m, enc, ly->rot_ca_kv);
    int                 q_dim  = Nh * D;
    int                 kv_dim = Nkv * D;
    struct ggml_tensor *q, *k, *v;
    if (ly->ca_qkv) {
        // Full QKV fused: split Q from hidden, KV from enc via weight views
        struct ggml_tensor * w_q  = ggml_view_2d(ctx, ly->ca_qkv, ly->ca_qkv->ne[0], q_dim, ly->ca_qkv->nb[1], 0);
        struct ggml_tensor * w_kv = ggml_view_2d(ctx, ly->ca_qkv, ly->ca_qkv->ne[0], 2 * kv_dim, ly->ca_qkv->nb[1],
                                                 (size_t) q_dim * ly->ca_qkv->nb[1]);
        q                         = ggml_mul_mat(ctx, w_q, norm_ca_rot);
        struct ggml_tensor * kv   = ggml_mul_mat(ctx, w_kv, enc_rot);
        k                         = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], 0));
        v = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], (size_t) kv_dim * kv->nb[0]));
    } else if (ly->ca_kv) {
        // Q separate, K+V fused
        q                       = dit_ggml_linear(ctx, ly->ca_q_proj, norm_ca_rot);
        struct ggml_tensor * kv = ggml_mul_mat(ctx, ly->ca_kv, enc_rot);
        k                       = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], 0));
        v = ggml_cont(ctx, ggml_view_3d(ctx, kv, kv_dim, enc_S, N, kv->nb[1], kv->nb[2], (size_t) kv_dim * kv->nb[0]));
    } else {
        // ca_q is frame-indexed ([q_dim, S]) → per-frame mask. ca_k/ca_v come from
        // the encoder ([kv_dim, enc_S], token-indexed) → mean-scaled, not masked.
        q = dit_lora_apply_layer(ctx, ly->ca_q_proj, norm_ca, lora ? &lora->ca_q : nullptr, sect, layer_idx, &DiTLoRALayer::ca_q, true, norm_ca_rot);
        k = dit_lora_apply_layer(ctx, ly->ca_k_proj, enc, lora ? &lora->ca_k : nullptr, sect, layer_idx, &DiTLoRALayer::ca_k, false, enc_rot);
        v = dit_lora_apply_layer(ctx, ly->ca_v_proj, enc, lora ? &lora->ca_v : nullptr, sect, layer_idx, &DiTLoRALayer::ca_v, false, enc_rot);
    }

    // reshape to [D, heads, seq, N] then permute to [D, seq, heads, N]
    q = ggml_reshape_4d(ctx, q, D, Nh, S, N);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, S, Nh, N]

    k = ggml_reshape_4d(ctx, k, D, Nkv, enc_S, N);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);  // [D, enc_S, Nkv, N]

    v = ggml_reshape_4d(ctx, v, D, Nkv, enc_S, N);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);  // [D, enc_S, Nkv, N]

    // QK-norm (per head)
    q = ggml_rms_norm(ctx, q, c.rms_norm_eps);
    q = ggml_mul(ctx, q, dit_ggml_f32(ctx, ly->ca_q_norm));
    k = ggml_rms_norm(ctx, k, c.rms_norm_eps);
    k = ggml_mul(ctx, k, dit_ggml_f32(ctx, ly->ca_k_norm));

    // no RoPE for cross-attention
    // mask blocks padding positions in encoder hidden states
    float scale = 1.0f / sqrtf((float) D);

    // K/V come in F32 from mul_mat (no KV cache here). Cast to F16 before FA,
    // mirroring llama.cpp build_attn_mha for graphs without a KV cache.
    if (m->use_flash_attn) {
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
        }
        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }
    }

    struct ggml_tensor * attn = m->use_flash_attn ? ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f) :
                                                    dit_attn_f32(ctx, q, k, v, mask, scale);
    if (m->use_flash_attn) {
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    }

    // Attention output: [D, Nh, S, N], reshape to [H, S, N]
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, N);

    // O projection ([H, S], frame-indexed)
    return dit_lora_apply_layer(ctx, ly->ca_o_proj, attn, lora ? &lora->ca_o : nullptr, sect, layer_idx, &DiTLoRALayer::ca_o, true,
                                dit_convrot_rotate(ctx, m, attn, ly->rot_ca_o));
}

// Build one full DiT layer (AdaLN + self-attn + cross-attn + FFN + gated residuals)
// hidden: [H, S, N], tproj: [6H] (combined timestep projection)
// enc: [H, enc_S, N] (condition-embedded encoder states, or NULL to skip cross-attn)
// sa_mask: [S, S, 1, N] self-attention sliding window mask, or NULL (full attention)
// ca_mask: [enc_S, S, 1, N] cross-attention mask (encoder padding), or NULL
// Returns: updated hidden [H, S, N]
static struct ggml_tensor * dit_ggml_build_layer(struct ggml_context * ctx,
                                                 DiTGGML *             m,
                                                 int                   layer_idx,
                                                 struct ggml_tensor *  hidden,     // [H, S, N]
                                                 struct ggml_tensor *  tproj,      // [6H] f32 combined temb projection
                                                 struct ggml_tensor *  enc,        // [H, enc_S, N] or NULL
                                                 struct ggml_tensor *  positions,  // [S] int32
                                                 struct ggml_tensor *  sa_mask,    // [S, S, 1, N] or NULL
                                                 struct ggml_tensor *  ca_mask,    // [enc_S, S, 1, N] or NULL
                                                 int                   S,
                                                 int                   enc_S,
                                                 int                   N,
                                                 DiTLoRALayer *        lora = nullptr,
                                                 const DiTLoRASectionCtx * sect = nullptr) {
    DiTGGMLConfig & c  = m->cfg;
    DiTGGMLLayer *  ly = &m->layers[layer_idx];
    int             H  = c.hidden_size;

    // AdaLN: scale_shift_table [6, H] + tproj [6H] -> 6 vectors of [H]
    // scale_shift_table is stored as bf16, cast to f32 for arithmetic
    struct ggml_tensor * ss = ly->scale_shift_table;
    if (ss->type != GGML_TYPE_F32) {
        ss = ggml_cast(ctx, ss, GGML_TYPE_F32);
    }
    // flatten [H, 6] -> [6H] (ggml ne[0]=H, ne[1]=6, contiguous = 6H floats)
    struct ggml_tensor * ss_flat = ggml_reshape_1d(ctx, ss, 6 * H);
    struct ggml_tensor * adaln   = ggml_add(ctx, ss_flat, tproj);  // [6H] f32

    // extract 6 modulation vectors [H] each
    size_t               Hb        = H * sizeof(float);
    struct ggml_tensor * shift_sa  = ggml_view_1d(ctx, adaln, H, 0 * Hb);
    struct ggml_tensor * scale_sa  = ggml_view_1d(ctx, adaln, H, 1 * Hb);
    struct ggml_tensor * gate_sa   = ggml_view_1d(ctx, adaln, H, 2 * Hb);
    struct ggml_tensor * shift_ffn = ggml_view_1d(ctx, adaln, H, 3 * Hb);
    struct ggml_tensor * scale_ffn = ggml_view_1d(ctx, adaln, H, 4 * Hb);
    struct ggml_tensor * gate_ffn  = ggml_view_1d(ctx, adaln, H, 5 * Hb);

    // Self-attention with AdaLN + gated residual
    struct ggml_tensor * residual = hidden;
    struct ggml_tensor * norm_sa  = dit_ggml_rms_norm_weighted(ctx, hidden, ly->self_attn_norm, c.rms_norm_eps);
    norm_sa                       = dit_ggml_adaln(ctx, norm_sa, scale_sa, shift_sa, m->scalar_one);

    if (layer_idx == 0) {
        ggml_set_name(norm_sa, "layer0_sa_input");
        ggml_set_output(norm_sa);
    }

    // sa_mask is pre-selected by the caller (sliding window for layer_type=0, NULL for layer_type=1)
    struct ggml_tensor * sa_out = dit_ggml_build_self_attn(ctx, m, ly, norm_sa, positions, sa_mask, S, N, layer_idx, lora, sect);

    if (layer_idx == 0) {
        ggml_set_name(sa_out, "layer0_sa_output");
        ggml_set_output(sa_out);
    }

    hidden = dit_ggml_gated_add(ctx, residual, sa_out, gate_sa);

    if (layer_idx == 0) {
        ggml_set_name(hidden, "layer0_after_self_attn");
        ggml_set_output(hidden);
    }

    // Cross-attention (no gate, simple residual add)
    if (enc) {
        struct ggml_tensor * norm_ca = dit_ggml_rms_norm_weighted(ctx, hidden, ly->cross_attn_norm, c.rms_norm_eps);
        struct ggml_tensor * ca_out =
            dit_ggml_build_cross_attn(ctx, m, ly, norm_ca, enc, positions, ca_mask, S, enc_S, N, lora, layer_idx, sect);
        hidden = ggml_add(ctx, hidden, ca_out);
    }

    if (layer_idx == 0) {
        ggml_set_name(hidden, "layer0_after_cross_attn");
        ggml_set_output(hidden);
    }

    // FFN with AdaLN + gated residual
    residual                      = hidden;
    struct ggml_tensor * norm_ffn = dit_ggml_rms_norm_weighted(ctx, hidden, ly->mlp_norm, c.rms_norm_eps);
    norm_ffn                      = dit_ggml_adaln(ctx, norm_ffn, scale_ffn, shift_ffn, m->scalar_one);
    struct ggml_tensor * ffn_out  = dit_ggml_build_mlp(ctx, m, ly, norm_ffn, S, lora, layer_idx, sect);
    hidden                        = dit_ggml_gated_add(ctx, residual, ffn_out, gate_ffn);

    return hidden;
}

// Build the full DiT forward graph (all layers).
// Returns the final output tensor (velocity prediction).
// N = batch size (number of samples to denoise in parallel).
//
// Graph inputs (ggml [ne0, ne1, ne2] notation):
//   "input_latents"   [in_channels, T, N]  concat(context_latents, xt) per sample
//   "enc_hidden"      [H, enc_S, N]        text encoder hidden states (N copies)
//   "t"               [1] f32              flow matching timestep (shared)
//   "t_r"             [1] f32              reference timestep (shared)
//   "positions"       [S*N] i32            position indices 0..S-1 repeated N times
//   "sa_mask_sw"      [S, S, 1, N] f16     self-attn sliding window (ne0=KV, ne1=Q)
//   "ca_mask"         [enc_S, S, 1, N] f16 cross-attn, enc padding  (ne0=KV, ne1=Q)
//
// Graph outputs:
//   "velocity"        [out_channels, T, N]  predicted flow velocity
static struct ggml_cgraph * dit_ggml_build_graph(DiTGGML *             m,
                                                 struct ggml_context * ctx,
                                                 int                   T,           // temporal length (before patching)
                                                 int                   enc_S,       // encoder sequence length
                                                 int                   N,           // batch size
                                                 struct ggml_tensor ** p_input,     // [out] input tensor to fill
                                                 struct ggml_tensor ** p_output) {  // [out] output tensor to read

    DiTGGMLConfig & c = m->cfg;
    int             S = T / c.patch_size;  // sequence length after patching
    int             H = c.hidden_size;
    int             P = c.patch_size;

    // Node budget scales with the separate per-adapter delta sets (per-section
    // masking); must match the ctx sizing in the sampler.
    size_t graph_cap = 8192 + m->loras.size() * 4096 + dit_lora_unit_nodes(&m->lora);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_cap, false);

    // Inputs

    // Concatenated latent: [in_channels, T, N] per sample
    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.in_channels, T, N);
    ggml_set_name(input, "input_latents");
    ggml_set_input(input);
    *p_input = input;

    // Encoder hidden states: [H_enc, enc_S, N]
    // H_enc comes from the condition_embedder input dimension (2048 for both 2B and XL).
    // The condition_embedder projects H_enc -> H (decoder) via cond_emb_w.
    int                  H_enc      = (int) m->cond_emb_w->ne[0];
    struct ggml_tensor * enc_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H_enc, enc_S, N);
    ggml_set_name(enc_hidden, "enc_hidden");
    ggml_set_input(enc_hidden);

    // Timesteps: scalars
    struct ggml_tensor * t_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t_val, "t");
    ggml_set_input(t_val);

    struct ggml_tensor * tr_val = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(tr_val, "t_r");
    ggml_set_input(tr_val);

    // Position indices for RoPE: [N*S] with values [0..S-1] repeated N times.
    // The CUDA rope kernel indexes positions by channel_x = row / ne1 which
    // linearizes (ne2, ne3) = (S, N). Batch b reads pos[b*S + s], so we must
    // repeat the sequence for each batch element.
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S * N);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Attention masks: F16, 4D [ne0, ne1, 1, N].
    // flash_attn_ext and soft_max_ext both expect [ne0=KV_len, ne1=Q_len, 1, N].
    // Must be 4D: CUDA flash_attn_mask_to_KV_max offsets by batch*nb[3],
    // so ne[3] must equal N.
    //
    // sa_mask_sw:  [S, S, 1, N]      self-attn sliding window (layer_type=0)
    // ca_mask:     [enc_S, S, 1, N]  cross-attn (encoder padding)
    //
    // Full-attention layers (layer_type=1) run unmasked: the batch shares one T,
    // so there is no temporal padding to block.
    struct ggml_tensor * sa_mask_sw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, N);
    ggml_set_name(sa_mask_sw, "sa_mask_sw");
    ggml_set_input(sa_mask_sw);

    struct ggml_tensor * ca_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, enc_S, S, 1, N);
    ggml_set_name(ca_mask, "ca_mask");
    ggml_set_input(ca_mask);

    // Per-section adapter masking setup: one [1,S,1] mask input per adapter (gates
    // frame-indexed projections) + a size-weighted mean section weight per adapter
    // (scales token/global projections that a per-frame mask can't apply to). Only
    // when the section path loaded separate per-adapter deltas into m->loras.
    m->lora_masks.clear();
    std::vector<float> lora_means;
    DiTLoRASectionCtx  sect_storage;
    const DiTLoRASectionCtx * sect = nullptr;
    const bool section_mode = (!m->loras.empty() && !g_hotstep_params.adapter_sections.empty());
    if (section_mode) {
        const size_t Nad = m->loras.size();
        sect_storage.loras = &m->loras;
        // NOMASK debug: skip creating the mask/mean gates entirely (deltas applied
        // unmasked). Creating input tensors that no graph op consumes leaves them
        // without a backend buffer, and the sampler's mask upload would then hit
        // GGML_ASSERT(buf != NULL). So only build the gates when they're used.
        if (!g_hotstep_section_nomask) {
            m->lora_masks.resize(Nad, nullptr);
            for (size_t i = 0; i < Nad; i++) {
                struct ggml_tensor * mk = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, S, 1);
                char nm[32];
                snprintf(nm, sizeof(nm), "lora_mask_%zu", i);
                ggml_set_name(mk, nm);
                ggml_set_input(mk);
                m->lora_masks[i] = mk;
            }
            lora_means.assign(Nad, 0.0f);
            double tot = 0.0;
            for (const auto & s : g_hotstep_params.adapter_sections) tot += (s.size > 0.0f ? s.size : 1.0f);
            if (tot <= 0.0) tot = 1.0;
            for (const auto & s : g_hotstep_params.adapter_sections) {
                const float w = (s.size > 0.0f ? s.size : 1.0f);
                for (size_t i = 0; i < Nad && i < s.weights.size(); i++)
                    lora_means[i] += (float) (w * s.weights[i] / tot);
            }
            sect_storage.masks = &m->lora_masks;
            sect_storage.means = &lora_means;
        }
        sect = &sect_storage;
    }

    // 1) Timestep embeddings
    struct ggml_tensor * tproj;

    struct ggml_tensor * temb;

    {
        struct ggml_tensor * tproj_t;
        struct ggml_tensor * temb_t = dit_ggml_build_temb(ctx, m, &m->time_embed, t_val, &tproj_t,
                                                          m->lora.active ? &m->lora.time_embed_linear_1 : nullptr,
                                                          m->lora.active ? &m->lora.time_embed_linear_2 : nullptr,
                                                          m->lora.active ? &m->lora.time_embed_time_proj : nullptr,
                                                          "_t");
        ggml_set_name(temb_t, "temb_t");
        ggml_set_output(temb_t);

        struct ggml_tensor * tproj_r;
        // Python passes (t - t_r) to time_embed_r, not t_r directly
        // In turbo mode t = t_r, so input is 0
        struct ggml_tensor * t_diff = ggml_sub(ctx, t_val, tr_val);
        struct ggml_tensor * temb_r = dit_ggml_build_temb(ctx, m, &m->time_embed_r, t_diff, &tproj_r,
                                                          m->lora.active ? &m->lora.time_embed_r_linear_1 : nullptr,
                                                          m->lora.active ? &m->lora.time_embed_r_linear_2 : nullptr,
                                                          m->lora.active ? &m->lora.time_embed_r_time_proj : nullptr,
                                                          "_r");
        ggml_set_name(temb_r, "temb_r");
        ggml_set_output(temb_r);

        // combine: temb = temb_t + temb_r [H], tproj = tproj_t + tproj_r [6H]
        temb = ggml_add(ctx, temb_t, temb_r);
        ggml_set_name(temb, "temb");
        ggml_set_output(temb);
        tproj = ggml_add(ctx, tproj_t, tproj_r);
        ggml_set_name(tproj, "tproj");
        ggml_set_output(tproj);
    }

    // 2) proj_in: patchify + linear (weight pre-permuted at load time)
    ggml_set_name(input, "proj_in_input");
    ggml_set_output(input);
    struct ggml_tensor * patched = ggml_reshape_3d(ctx, input, c.in_channels * P, S, N);
    struct ggml_tensor * hidden  = dit_ggml_linear_bias(ctx, m->proj_in_w, m->proj_in_b, patched);
    if (section_mode) {
        hidden = dit_lora_apply_global(ctx, hidden, patched, nullptr, sect, &DiTLoRA::proj_in, true);
    } else if (m->lora.active) {
        hidden = dit_lora_apply_units(ctx, &m->lora.proj_in, hidden, patched);
    }
    ggml_set_name(hidden, "hidden_after_proj_in");
    ggml_set_output(hidden);

    // 3) Condition embedder: project encoder hidden states
    // ConvRot: base weight consumes the rotated input; the LoRA delta paths
    // below keep consuming the raw enc_hidden (deltas are unrotated).
    struct ggml_tensor * enc = dit_ggml_linear_bias(ctx, m->cond_emb_w, m->cond_emb_b,
                                                    dit_convrot_rotate(ctx, m, enc_hidden, m->convrot.cond_emb));
    if (section_mode) {
        // cond_emb is token/global-indexed → mean-scaled, not per-frame masked.
        enc = dit_lora_apply_global(ctx, enc, enc_hidden, nullptr, sect, &DiTLoRA::cond_emb, false);
    } else if (m->lora.active) {
        enc = dit_lora_apply_units(ctx, &m->lora.cond_emb, enc, enc_hidden);
    }
    ggml_set_name(enc, "enc_after_cond_emb");
    ggml_set_output(enc);

    // 4) Transformer layers
    for (int i = 0; i < c.n_layers; i++) {
        // layer_type=0 (sliding window): sa_mask_sw, layer_type=1 (full): unmasked
        struct ggml_tensor * sa_mask = (m->layers[i].layer_type == 0) ? sa_mask_sw : nullptr;
        DiTLoRALayer * lora_ly = (m->lora.active && i < DIT_LORA_MAX_LAYERS) ? &m->lora.layers[i] : nullptr;
        hidden = dit_ggml_build_layer(ctx, m, i, hidden, tproj, enc, positions, sa_mask, ca_mask, S, enc_S, N, lora_ly, sect);
        // Debug dumps at key layers: 0, 6, 12, 18, last
        if (i == 0 || i == 6 || i == 12 || i == 18 || i == c.n_layers - 1) {
            char lname[64];
            snprintf(lname, sizeof(lname), "hidden_after_layer%d", i);
            ggml_set_name(hidden, lname);
            ggml_set_output(hidden);
        }
    }

    // 5) Output: AdaLN + proj_out
    // out_scale_shift: [H, 2] -> cast to f32 if bf16, flatten to [2H]
    struct ggml_tensor * oss = m->out_scale_shift;
    if (oss->type != GGML_TYPE_F32) {
        oss = ggml_cast(ctx, oss, GGML_TYPE_F32);
    }
    struct ggml_tensor * oss_flat = ggml_reshape_1d(ctx, oss, 2 * H);

    size_t               Hb        = H * sizeof(float);
    struct ggml_tensor * out_shift = ggml_view_1d(ctx, oss_flat, H, 0);
    struct ggml_tensor * out_scale = ggml_view_1d(ctx, oss_flat, H, Hb);
    out_shift                      = ggml_add(ctx, out_shift, temb);
    out_scale                      = ggml_add(ctx, out_scale, temb);

    struct ggml_tensor * norm_out = dit_ggml_rms_norm_weighted(ctx, hidden, m->norm_out, c.rms_norm_eps);
    norm_out                      = dit_ggml_adaln(ctx, norm_out, out_scale, out_shift, m->scalar_one);

    // proj_out: weight pre-permuted+transposed at load time to [H, out_ch*P] F32
    struct ggml_tensor * output = dit_ggml_linear_bias(ctx, m->proj_out_w, m->proj_out_b, norm_out);
    output                      = ggml_reshape_3d(ctx, output, c.out_channels, T, N);

    ggml_set_name(output, "velocity");
    ggml_set_output(output);
    *p_output = output;

    ggml_build_forward_expand(gf, output);

    return gf;
}
