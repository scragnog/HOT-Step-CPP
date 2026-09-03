#pragma once
// lm-graph.h — trainable, cache-free, unfused Qwen3 LM forward.
//
// Five load-bearing properties (plan §3.4), all validated by the Phase-0 spike:
//
//  1. NO KV cache        — GGML_OP_SET_ROWS has no backward. Never call
//                          qw3lm_forward*.
//  2. NO flash attention — GGML_OP_FLASH_ATTN_EXT has no backward. We use
//                          qwen3_attn_f32() with an explicit [S,S] causal
//                          0/-INF F32 mask. Exact: soft_max_ext's backward
//                          uses the softmax OUTPUT, so the mask and scale are
//                          correctly baked in and masked positions get exactly
//                          zero gradient.
//  3. NO fused QKV / gate_up — g_qwen3_load_no_fuse = true around qw3lm_load()
//                          (L18), restored afterwards.
//  4. ggml_swiglu_split() — the fused form has no backward.
//  5. F32 weight mirror, allocated once, BF16 buffer released — ggml_out_prod
//     (the backward of mul_mat w.r.t. its ACTIVATION input) is F32-only on
//     CUDA and GGML_ABORTs for BF16 on CPU. A quantized base therefore cannot
//     be trained against and is rejected at mirror time.
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §3.4

#include "qwen3-lm.h"
#include "lokr-common.h"
#include "qwen3-lora.h"
#include "train/lm-bf16.h"  // LmWtCollect — Lever A (--weights bf16)
#include "train/lm-common.h"
#include "train/flash-prec.h"  // dit_flash_probe / dit_flash_prec_label (--attn flash)

#include <string>
#include <vector>

// ─── BF16/F16 -> F32 weight mirror ──────────────────────────────────────────

struct LmF32Mirror {
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    size_t                bytes = 0;
    int                   n_tensors = 0;
};

static void lm_mirror_free(LmF32Mirror * M) {
    if (M->buf) {
        ggml_backend_buffer_free(M->buf);
    }
    if (M->ctx) {
        ggml_free(M->ctx);
    }
    *M = LmF32Mirror{};
}

// Mirror every weight the training graph touches into F32, then drop the
// original weight buffer. `err` is set on failure (unsupported dtype = a
// quantized base -> caller emits fatal reason "model-load").
static bool lm_build_f32_mirror(Qwen3LM * lm, LmF32Mirror * M, std::string * err) {
    const int L = lm->cfg.n_layers;
    const int n = 2 + L * 11 + 16;

    ggml_init_params p = { (size_t) n * ggml_tensor_overhead() + 4096, nullptr, true };
    M->ctx             = ggml_init(p);
    if (!M->ctx) {
        *err = "cannot create the F32 mirror context";
        return false;
    }

    struct Slot {
        ggml_tensor ** slot;  // pointer to the Qwen3LM field
        ggml_tensor *  src;
        ggml_tensor *  dst;
    };
    std::vector<Slot> slots;
    slots.reserve((size_t) (2 + L * 11));

    auto add = [&](ggml_tensor ** field) {
        if (!field || !*field) {
            return;  // fused slots are NULL under no-fuse; nothing to mirror
        }
        ggml_tensor * s = *field;
        ggml_tensor * d = ggml_new_tensor_4d(M->ctx, GGML_TYPE_F32, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
        ggml_set_name(d, s->name);
        M->bytes += ggml_nbytes(d);
        slots.push_back({ field, s, d });
    };

    add(&lm->embed_tokens);
    add(&lm->final_norm);
    for (int i = 0; i < L; i++) {
        Qwen3Layer & ly = lm->layers[i];
        add(&ly.input_layernorm);
        add(&ly.post_attn_layernorm);
        add(&ly.q_proj);
        add(&ly.k_proj);
        add(&ly.v_proj);
        add(&ly.o_proj);
        add(&ly.q_norm);
        add(&ly.k_norm);
        add(&ly.gate_proj);
        add(&ly.up_proj);
        add(&ly.down_proj);
    }

    // Reject fused / missing projections up front — the trainer needs every
    // LoRA site individually addressable.
    for (int i = 0; i < L; i++) {
        const Qwen3Layer & ly = lm->layers[i];
        if (!ly.q_proj || !ly.k_proj || !ly.v_proj || !ly.o_proj || !ly.gate_proj || !ly.up_proj || !ly.down_proj) {
            char b[160];
            snprintf(b, sizeof(b), "layer %d has fused projections — the trainer requires an unfused load", i);
            *err = b;
            return false;
        }
    }

    M->buf = ggml_backend_alloc_ctx_tensors(M->ctx, lm->backend);
    if (!M->buf) {
        char b[160];
        snprintf(b, sizeof(b), "F32 weight mirror allocation failed (%.1f MB)", M->bytes / 1048576.0);
        *err = b;
        return false;
    }
    ggml_backend_buffer_set_usage(M->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (size_t i = 0; i < slots.size(); i++) {
        ggml_tensor * s  = slots[i].src;
        ggml_tensor * d  = slots[i].dst;
        const size_t  ne = (size_t) ggml_nelements(s);
        raw.resize(ggml_nbytes(s));
        ggml_backend_tensor_get(s, raw.data(), 0, raw.size());

        if (s->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(d, raw.data(), 0, raw.size());
        } else if (s->type == GGML_TYPE_BF16) {
            f32.resize(ne);
            const uint16_t * u = (const uint16_t *) raw.data();
            for (size_t j = 0; j < ne; j++) {
                const uint32_t b = (uint32_t) u[j] << 16;
                float          v;
                memcpy(&v, &b, 4);
                f32[j] = v;
            }
            ggml_backend_tensor_set(d, f32.data(), 0, ne * sizeof(float));
        } else if (s->type == GGML_TYPE_F16) {
            f32.resize(ne);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) ne);
            ggml_backend_tensor_set(d, f32.data(), 0, ne * sizeof(float));
        } else {
            char b[256];
            snprintf(b, sizeof(b),
                     "base weight '%s' is %s — LM training needs a BF16/F16/F32 base "
                     "(quantized bases cannot be trained: ggml_out_prod is F32-only)",
                     s->name, ggml_type_name(s->type));
            *err = b;
            return false;
        }
        *slots[i].slot = d;
    }

    M->n_tensors = (int) slots.size();

    // Drop the original weight buffer: the mirror fully replaces it.
    if (lm->wctx.buffer) {
        ggml_backend_buffer_free(lm->wctx.buffer);
        lm->wctx.buffer = nullptr;
    }
    fprintf(stderr, "[train-lm] F32 weight mirror: %d tensors, %.1f MB (base weight buffer released)\n", M->n_tensors,
            M->bytes / 1048576.0);
    return true;
}

// ─── trainable LoRA bank ────────────────────────────────────────────────────

struct LmLora {
    ggml_context *             ctx = nullptr;
    ggml_backend_buffer_t      buf = nullptr;
    QwLoraLayer                layers[QW3LM_MAX_LAYERS];
    std::vector<ggml_tensor *> params;  // A0,B0,A1,B1,… in layer-major slot order
    int                        rank      = 0;
    int                        layer_lo  = 0;
    int                        layer_hi  = 0;
    float                      alpha     = 0.0f;
    float                      scale     = 1.0f;
    size_t                     n_params  = 0;

    // LoKr (2026-07-30). The factors themselves hang off QwLoraPair, so the
    // graph, the optimizer, the checkpoint driver and the training loop are all
    // untouched — lm_linear dispatches on pr->has_lokr() and everything
    // downstream just sees a different set of ggml_set_param'd tensors.
    bool  is_lokr        = false;
    int   lokr_dim       = 0;
    int   lokr_factor    = 0;
    float lokr_alpha     = 0.0f;
    bool  lokr_decompose = true;
};

static void lm_lora_free(LmLora * L) {
    if (L->buf) {
        ggml_backend_buffer_free(L->buf);
    }
    if (L->ctx) {
        ggml_free(L->ctx);
    }
    L->buf = nullptr;
    L->ctx = nullptr;
    L->params.clear();
}

static void lm_slot_dims(const Qwen3LMConfig & c, int slot, int * in_dim, int * out_dim) {
    const int H = c.hidden_size, D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads, F = c.intermediate_size;
    switch (slot) {
        case QW_LORA_Q:    *in_dim = H;      *out_dim = Nh * D;  break;
        case QW_LORA_K:    *in_dim = H;      *out_dim = Nkv * D; break;
        case QW_LORA_V:    *in_dim = H;      *out_dim = Nkv * D; break;
        case QW_LORA_O:    *in_dim = Nh * D; *out_dim = H;       break;
        case QW_LORA_GATE: *in_dim = H;      *out_dim = F;       break;
        case QW_LORA_UP:   *in_dim = H;      *out_dim = F;       break;
        default:           *in_dim = F;      *out_dim = H;       break;  // DOWN
    }
}

static const char * lm_slot_peft_name(int slot) {
    switch (slot) {
        case QW_LORA_Q:    return "self_attn.q_proj";
        case QW_LORA_K:    return "self_attn.k_proj";
        case QW_LORA_V:    return "self_attn.v_proj";
        case QW_LORA_O:    return "self_attn.o_proj";
        case QW_LORA_GATE: return "mlp.gate_proj";
        case QW_LORA_UP:   return "mlp.up_proj";
        default:           return "mlp.down_proj";
    }
}

// True PEFT init: A ~ N(0, 1/sqrt(in)), B = 0. b_sigma > 0 breaks that (the
// self-test needs it: with B == 0, dL/dA is identically zero by construction).
// The LoRA tensors live in their OWN plain buffer — ggml_opt_step_adamw writes
// into them, so they must not share the mirror's WEIGHTS-usage buffer.
static bool lm_lora_init(LmLora * L, Qwen3LM * lm, int layer_lo, int layer_hi, int rank, float alpha, uint64_t seed,
                         float b_sigma, std::string * err) {
    const Qwen3LMConfig & c = lm->cfg;
    L->rank     = rank;
    L->alpha    = alpha;
    L->scale    = alpha / (float) rank;
    L->layer_lo = layer_lo;
    L->layer_hi = layer_hi;

    const int n_lay = layer_hi - layer_lo;
    const int n_ten = n_lay * QW_LORA_NSLOTS * 2;
    ggml_init_params p = { (size_t) (n_ten + 8) * ggml_tensor_overhead(), nullptr, true };
    L->ctx             = ggml_init(p);
    if (!L->ctx) {
        *err = "cannot create the LoRA context";
        return false;
    }

    for (int i = 0; i < QW3LM_MAX_LAYERS; i++) {
        L->layers[i] = QwLoraLayer{};
    }

    for (int l = layer_lo; l < layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            int in_dim = 0, out_dim = 0;
            lm_slot_dims(c, s, &in_dim, &out_dim);
            QwLoraPair & pr = L->layers[l].p[s];
            pr.A            = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, in_dim, rank);
            pr.B            = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, rank, out_dim);
            pr.scale        = L->scale;
            char nm[96];
            snprintf(nm, sizeof(nm), "L%d.s%d.lora_A", l, s);
            ggml_set_name(pr.A, nm);
            snprintf(nm, sizeof(nm), "L%d.s%d.lora_B", l, s);
            ggml_set_name(pr.B, nm);
            ggml_set_param(pr.A);
            ggml_set_param(pr.B);
            L->params.push_back(pr.A);
            L->params.push_back(pr.B);
        }
    }

    L->buf = ggml_backend_alloc_ctx_tensors(L->ctx, lm->backend);
    if (!L->buf) {
        *err = "LoRA parameter buffer allocation failed";
        return false;
    }

    LmRng rng;
    lm_rng_seed(&rng, seed);
    size_t             n_par = 0;
    std::vector<float> a, b;
    for (int l = layer_lo; l < layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair & pr = L->layers[l].p[s];
            a.assign((size_t) ggml_nelements(pr.A), 0.0f);
            b.assign((size_t) ggml_nelements(pr.B), 0.0f);
            lm_rng_fill_normal(&rng, a, 1.0f / sqrtf((float) pr.A->ne[0]));
            if (b_sigma > 0.0f) {
                lm_rng_fill_normal(&rng, b, b_sigma);
            }
            ggml_backend_tensor_set(pr.A, a.data(), 0, a.size() * sizeof(float));
            ggml_backend_tensor_set(pr.B, b.data(), 0, b.size() * sizeof(float));
            n_par += a.size() + b.size();
        }
        lm->layers[l].lora = &L->layers[l];
    }
    L->n_params = n_par;
    fprintf(stderr, "[train-lm] LoRA layers %d..%d rank %d alpha %.0f -> %zu trainable params (%.1f MB f32)\n",
            layer_lo, layer_hi, rank, (double) alpha, n_par, (double) n_par * 4.0 / 1048576.0);
    return true;
}

// ─── LoKr parameterization (2026-07-30) ─────────────────────────────────────
//
// LyCORIS parity via the shared rules in lokr-common.h, so a file written here
// has the same layout as one written by the DiT trainer and is read by the same
// adapter-merge.h kron path.
//
//   (out_l, out_k) = factorization(out, factor)
//   (in_m,  in_n)  = factorization(in,  factor)
//   dW = kron(w1, w2) * (alpha / dim),  w1 ggml [in_m, out_l], w2 [in_n, out_k]
//
// Init mirrors the DiT's: the delta must start at EXACTLY zero or the first
// step perturbs a base that has not been trained toward yet — monolithic w2 is
// zeroed and w1 kaiming; factorized w2_a is kaiming and w2_b zeroed.
static bool lm_lokr_init(LmLora * L, Qwen3LM * lm, int layer_lo, int layer_hi, int dim, float alpha, int factor,
                         bool decompose_both, uint64_t seed, std::string * err) {
    const Qwen3LMConfig & c = lm->cfg;
    L->is_lokr        = true;
    L->lokr_dim       = dim;
    L->lokr_factor    = factor;
    L->lokr_alpha     = alpha;
    L->lokr_decompose = decompose_both;
    L->layer_lo       = layer_lo;
    L->layer_hi       = layer_hi;

    const int n_lay = layer_hi - layer_lo;
    const int n_ten = n_lay * QW_LORA_NSLOTS * 3;  // w1 + (w2 | w2_a + w2_b)
    ggml_init_params p = { (size_t) (n_ten + 8) * ggml_tensor_overhead(), nullptr, true };
    L->ctx             = ggml_init(p);
    if (!L->ctx) {
        *err = "cannot create the LoKr context";
        return false;
    }
    for (int i = 0; i < QW3LM_MAX_LAYERS; i++) {
        L->layers[i] = QwLoraLayer{};
    }

    int n_mono = 0, n_fact = 0;
    for (int l = layer_lo; l < layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            int in_dim = 0, out_dim = 0;
            lm_slot_dims(c, s, &in_dim, &out_dim);
            QwLoraPair & pr = L->layers[l].p[s];
            lokr_factorization(out_dim, factor, &pr.out_l, &pr.out_k);
            lokr_factorization(in_dim, factor, &pr.in_m, &pr.in_n);

            // Refuse rather than train something that cannot be exported: the
            // kron of the chosen factors has to reconstruct the base shape
            // exactly, or the adapter is unloadable.
            if (pr.out_l * pr.out_k != (int64_t) out_dim || pr.in_m * pr.in_n != (int64_t) in_dim) {
                char b[192];
                snprintf(b, sizeof(b), "layer %d slot %s does not factorize: base [%d,%d] but kron gives [%lld,%lld]",
                         l, lm_slot_peft_name(s), in_dim, out_dim, (long long) (pr.in_m * pr.in_n),
                         (long long) (pr.out_l * pr.out_k));
                *err = b;
                return false;
            }

            const bool mono = lokr_w2_mono(dim, pr.out_k, pr.in_n);
            // LyCORIS forces alpha = dim when both factors are monolithic, which
            // makes the in-graph scale exactly 1 (DiT K6).
            float a_eff = (alpha == 0.0f) ? (float) dim : alpha;
            if (mono) {
                a_eff = (float) dim;
            }
            pr.lokr_scale = a_eff / (float) dim;

            char nm[128];
            pr.w1 = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, pr.in_m, pr.out_l);
            snprintf(nm, sizeof(nm), "L%d.%s.lokr_w1", l, lm_slot_peft_name(s));
            ggml_set_name(pr.w1, nm);
            ggml_set_param(pr.w1);
            L->params.push_back(pr.w1);
            if (mono) {
                pr.w2 = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, pr.in_n, pr.out_k);
                snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2", l, lm_slot_peft_name(s));
                ggml_set_name(pr.w2, nm);
                ggml_set_param(pr.w2);
                L->params.push_back(pr.w2);
                n_mono++;
            } else {
                pr.w2_a = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, dim, pr.out_k);
                pr.w2_b = ggml_new_tensor_2d(L->ctx, GGML_TYPE_F32, pr.in_n, dim);
                snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2_a", l, lm_slot_peft_name(s));
                ggml_set_name(pr.w2_a, nm);
                snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2_b", l, lm_slot_peft_name(s));
                ggml_set_name(pr.w2_b, nm);
                ggml_set_param(pr.w2_a);
                ggml_set_param(pr.w2_b);
                L->params.push_back(pr.w2_a);
                L->params.push_back(pr.w2_b);
                n_fact++;
            }
        }
    }

    L->buf = ggml_backend_alloc_ctx_tensors(L->ctx, lm->backend);
    if (!L->buf) {
        *err = "LoKr parameter buffer allocation failed";
        return false;
    }

    LmRng rng;
    lm_rng_seed(&rng, seed);
    std::vector<float> v;
    size_t             n_par = 0;
    for (int l = layer_lo; l < layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair & pr = L->layers[l].p[s];
            auto kaiming = [&](ggml_tensor * t) {
                v.assign((size_t) ggml_nelements(t), 0.0f);
                lm_rng_fill_normal(&rng, v, 1.0f / sqrtf((float) t->ne[0]));
                ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
                n_par += v.size();
            };
            auto zeros = [&](ggml_tensor * t) {
                v.assign((size_t) ggml_nelements(t), 0.0f);
                ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
                n_par += v.size();
            };
            kaiming(pr.w1);
            if (pr.w2) {
                zeros(pr.w2);
            } else {
                kaiming(pr.w2_a);
                zeros(pr.w2_b);
            }
        }
        // ATTACH, exactly as lm_lora_init does. Without this the model's layers
        // carry no adapter pointer, the graph applies nothing, every parameter
        // is unreachable, and ggml_build_backward_expand aborts with "no
        // trainable parameters found" — which is what it did.
        lm->layers[l].lora = &L->layers[l];
    }
    L->n_params = n_par;
    fprintf(stderr, "[train-lm] LoKr: dim %d factor %d alpha %.4g, %d monolithic / %d factorized w2, %zu params\n",
            dim, factor, (double) alpha, n_mono, n_fact, n_par);
    return true;
}


// Detach the LoRA slots from the model (used by the self-test between phases).
static void lm_lora_detach(LmLora * L, Qwen3LM * lm) {
    for (int l = L->layer_lo; l < L->layer_hi; l++) {
        lm->layers[l].lora = nullptr;
    }
}

// ─── layer build options (low-VRAM path, 4B plan §1.1 / D3 / D6) ────────────
//
// A default-constructed LmLayerOpts reproduces the shipped naive graph EXACTLY:
//   * cast_weights is documentation only — lm_linear()/lm_rms() route every
//     weight through qwen3_f32(), which returns an already-F32 tensor unchanged
//     (qwen3-enc.h:87-92), so the naive path (F32 mirror) emits no cast node.
//   * attn_head_block == 0 selects the verbatim whole-head attention span.
// Destination views plus the ggml_cpy nodes that fill them; see
// LmLayerOpts::kv_cap and train/lm-kvprefix.h.
struct LmKvCapture {
    ggml_tensor *              k_dst = nullptr;  // [Nkv*D, S] view of the store
    ggml_tensor *              v_dst = nullptr;
    std::vector<ggml_tensor *> nodes;
};

struct LmLayerOpts {
    bool          cast_weights    = false;    // true when the base is BF16/F16 and
                                              // the per-segment F32 window is live
    int           attn_head_block = 0;        // q heads per attention block; 0 = off
    ggml_tensor * attn_zero       = nullptr;  // persistent, permanently-zero
                                              // [Nh*D, S_max] F32 base for ggml_acc

    // ── Lever A (2026-07-28 speed-levers plan §3.1) ──────────────────────
    // Both default OFF, and with them off lm_linear() takes the shipped branch
    // verbatim — §6.0 byte-identity depends on that.
    bool          weights_bf16 = false;    // run the 7 projections in BF16
    LmWtCollect * wt           = nullptr;  // per-graph W -> cont(transpose(W)) map;
                                           // nullptr in the P2/P3 forward-collect
                                           // graphs, which have no backward and
                                           // would only carry dead nodes

    // ── RANK DROPOUT ───────────────────────────────────────────────────────
    //
    // An [r] or [r,1] F32 mask multiplied into the LoRA bottleneck: survivors
    // carry 1/(1-p), dropped components carry 0. nullptr = off, and with it off
    // lm_linear emits the shipped graph verbatim.
    //
    // WHY THE BOTTLENECK AND NOT THE INPUT. PEFT's lora_dropout drops elements
    // of x, whose mask is [hidden, S] — 88 MB per module at S 5400, times 252
    // modules, per step. Masking rank components instead needs r floats per
    // step and is a real regulariser (LyCORIS calls it rank_dropout): each step
    // trains a random lower-rank subnetwork. It is NOT the same thing as
    // lora_dropout and should not be described as such.
    //
    // ONE MASK, SHARED BY EVERY MODULE. Per-module masks would need a pair
    // index threaded into lm_linear, and QwLoraPair is shared with the
    // inference runtime. Sharing correlates the dropout across modules — a
    // coarser regulariser than independent masks, closer to "train a random
    // rank-(1-p)r subnetwork this step". Stated because it is a real
    // difference, not because it is a defect.
    //
    // CHECKPOINT SAFETY IS BY CONSTRUCTION: this is a persistent tensor
    // uploaded once per micro-step, and D13 already requires the recomputed
    // forward to use the same LmLayerOpts as the collect pass. Both passes
    // therefore read the same bytes. A freshly-drawn mask per pass would
    // silently compute gradients for a different network than the loss.
    ggml_tensor * rank_mask = nullptr;

    // ── FROZEN KV PREFIX (train/lm-kvprefix.h) ─────────────────────────────
    //
    // Optional no-grad history in front of the trained window. kv_k / kv_v are
    // [Nkv*D, >= kv_pfx + S] F32 leaves whose first kv_pfx columns hold the
    // prefix's post-QK-norm, post-RoPE K and its raw V, and whose remaining
    // columns are ZERO — so splicing the window's own K/V in with ggml_acc is
    // exactly a copy.
    //
    // WHY ACC AND NOT CONCAT. GGML_OP_CONCAT has no backward (the note above
    // lm_attn_head_blocked); GGML_OP_ACC does, and its src1 gradient is a view
    // of the incoming gradient. The store is a frozen leaf with no grad, so the
    // prefix adds no backward nodes at all — which is the entire point of it.
    //
    // The caller sets these PER LAYER. nullptr => the shipped square
    // self-attention, emitted verbatim.
    ggml_tensor * kv_k   = nullptr;
    ggml_tensor * kv_v   = nullptr;
    int           kv_pfx = 0;

    // PREFILL CAPTURE. Set by the prefix pass only: lm_train_layer appends two
    // ggml_cpy nodes that write THIS call's post-QK-norm post-RoPE K and its
    // raw V into the store, and the caller expands them into the graph (the
    // same collector shape as LmWtCollect — lm_train_layer has no graph).
    LmKvCapture * kv_cap = nullptr;

    // ── FUSED ATTENTION (--attn flash, D1/D2) ──────────────────────────────
    //
    // false (the default) means the whole-head branch of lm_train_layer emits
    // the ORIGINAL qwen3_attn_f32 call and nothing else in this header moves —
    // that is what G0's byte-identity gate checks, and it is why the switch is a
    // ternary at one call site rather than a restructure.
    //
    // true routes that site through lm_attn_flash: GGML_OP_FLASH_ATTN_TRAIN plus
    // the _BACK node ggml's autodiff builds from it, so the [S_kv,S,Nh] softmax
    // is never materialised and attention memory becomes linear in S.
    //
    // `attn_prec` is the ARITHMETIC REQUEST, set on the forward node only;
    // ggml_compute_backward copies it onto the _BACK node it builds, which is
    // load-bearing (a backward rounding differently from its forward is not
    // rounding error, it is a different function). GGML_PREC_DEFAULT resolves to
    // the TF32 kernels on sm_80+, GGML_PREC_F32 pins the scalar v1 kernels.
    // NOTE the zero-init trap: GGML_PREC_DEFAULT == 0, so "unset" IS "tf32 where
    // available" — which is why the run log records the RESOLVED precision from
    // dit_flash_prec_label() and not just the flag (skill §2 point 6).
    //
    // The head-blocked branch has no fused arm and asserts; the CLI refuses the
    // pair (--attn flash with --attn-head-block > 0 is exit 2, D3).
    bool      attn_flash = false;
    ggml_prec attn_prec  = GGML_PREC_DEFAULT;

    // ── ADAPTER OFF (--reg-teacher live, 2026-09-03) ───────────────────────
    //
    // true makes lm_linear() emit the frozen base projection and NOTHING else:
    // no LoRA A/B branch, no LoKr kron delta. It is how the live-teacher pass
    // asks "what would the base have said here?" from inside a run whose
    // adapter is already trained, without a second model load and without
    // touching the adapter tensors.
    //
    // Default false, and false emits the shipped graph verbatim — the branch
    // below wraps the existing delta block rather than restructuring it, so a
    // flags-off run's node sequence is unchanged (the standing byte-identity
    // rule for every lever in this header).
    //
    // It applies to the whole pass, not per module: a half-disabled adapter is
    // not the base, and there is no use for one.
    bool      adapter_off = false;
};

// The fused drop-in for qwen3_attn_f32 at the whole-head site. Same arguments,
// and the SAME result shape and layout: ggml_flash_attn_train_get_o is a
// CONTIGUOUS [D,Nh,S] view, which is exactly what qwen3_attn_f32's closing
// ggml_cont(ggml_permute(...)) produces — so the call site's
// ggml_reshape_2d(attn, Nh*D, S) accepts it unchanged (B1).
//
// Note what is NOT here: no ggml_cont on q/k/v. They arrive as permuted views
// and the op reads them through nb[1..3] on purpose (only nb[0] == 4 is
// required); materialising them would hand back part of the saving.
//
// Mirrors dit_attn_flash in train/dit-train-graph.h op for op.
static ggml_tensor * lm_attn_flash(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                   ggml_tensor * mask, float scale, ggml_prec prec) {
    ggml_tensor * packed = ggml_flash_attn_train(ctx, q, k, v, mask, scale);
    ggml_flash_attn_train_set_prec(packed, prec);
    return ggml_flash_attn_train_get_o(ctx, packed);  // [D,Nh,S]
}

// F32 bytes of ONE transformer layer's trainable-graph weights (7 projections +
// 4 norms) — the size of the per-segment F32 window (§3.8 `w_layer_f32`).
static size_t lm_layer_weight_bytes(const Qwen3LMConfig & c) {
    const size_t H = (size_t) c.hidden_size, F = (size_t) c.intermediate_size;
    const size_t D = (size_t) c.head_dim, Nh = (size_t) c.n_heads, Nkv = (size_t) c.n_kv_heads;
    size_t       p = 0;
    p += H * (Nh * D);          // q_proj
    p += H * (Nkv * D) * 2;     // k_proj, v_proj
    p += (Nh * D) * H;          // o_proj
    p += H * F * 2;             // gate_proj, up_proj
    p += F * H;                 // down_proj
    p += 2 * H + 2 * D;         // input/post-attn layernorm + q_norm/k_norm
    return p * sizeof(float);
}

// Bytes of ONE transformer layer's SEVEN PROJECTIONS in `t` — the size of the
// per-segment transposed window under Lever A (§3.5). The 4 norms are excluded
// on purpose: lm_rms() is untouched by the lever, so they keep their F32 cast
// (21 KiB at 4B, charged separately in the model's slack).
//
// 4B [D]: P_proj = 100,925,440 params -> 192.5 MiB at BF16 vs 385.0 MiB for the
// F32 window it replaces.
static size_t lm_layer_proj_bytes(const Qwen3LMConfig & c, ggml_type t) {
    const size_t H = (size_t) c.hidden_size, F = (size_t) c.intermediate_size;
    const size_t D = (size_t) c.head_dim, Nh = (size_t) c.n_heads, Nkv = (size_t) c.n_kv_heads;
    size_t       p = 0;
    p += H * (Nh * D);          // q_proj
    p += H * (Nkv * D) * 2;     // k_proj, v_proj
    p += (Nh * D) * H;          // o_proj
    p += H * F * 2;             // gate_proj, up_proj
    p += F * H;                 // down_proj
    return p * ggml_type_size(t) / (size_t) ggml_blck_size(t);
}

// Bytes actually held by the loaded base weight buffer (BF16 in low-VRAM mode).
static size_t lm_base_weight_bytes(const Qwen3LM & lm) {
    return lm.wctx.buffer ? ggml_backend_buffer_get_size(lm.wctx.buffer) : 0;
}

// ─── graph builders ─────────────────────────────────────────────────────────

static ggml_tensor * lm_linear(ggml_context * ctx, ggml_tensor * w, const QwLoraPair * pr, ggml_tensor * x,
                               const LmLayerOpts & opts) {
    ggml_tensor * y;
    if (!opts.weights_bf16 || w->type == GGML_TYPE_F32) {
        // SHIPPED PATH — byte-identical.
        // qwen3_f32() is a NO-OP for an F32 weight, so the naive path's graph is
        // byte-identical to the shipped one. On a BF16 base it emits the in-graph
        // cast that ggml_out_prod (the mul_mat activation backward) requires — and
        // ggml_gallocr frees it with the segment. Plan D2/D3.
        y = ggml_mul_mat(ctx, qwen3_f32(ctx, w), x);
    } else {
        // Lever A. src0 stays BF16 -> ggml-cuda.cu's cublasGemmEx(CUDA_R_16BF,
        // CUBLAS_COMPUTE_32F) instead of cublasSgemm/TF32.
        y = ggml_mul_mat(ctx, w, x);
        if (opts.wt && !opts.wt->map.count(w)) {
            // [K,N] -> [N,K], SAME dtype. Consumed ONLY by the rewritten backward
            // node (lm_bf16_rewrite_outprod). ~0.041 ms against a ~0.9 ms GEMM
            // [M]; gallocr frees it with the segment, so there is no persistent
            // transposed bank (S4 — one would be a second 7,991 MiB at 4B).
            ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, w));
            ggml_format_name(wt, "%s.T", w->name);
            opts.wt->map[w] = wt;
            opts.wt->nodes.push_back(wt);
        }
    }
    // opts.adapter_off short-circuits BOTH parameterizations (--reg-teacher
    // live's frozen-base pass). Guarding the whole block rather than each arm
    // is deliberate: a future third adapter form added below would otherwise be
    // silently left on in the teacher pass, and the teacher would quietly stop
    // being the base.
    if (!opts.adapter_off) {
        if (pr && pr->A && pr->B) {
            ggml_tensor * t = ggml_mul_mat(ctx, pr->A, x);  // [r, S]
            if (opts.rank_mask) {
                t = ggml_mul(ctx, t, opts.rank_mask);       // [r,1] broadcasts over S
            }
            t               = ggml_scale(ctx, t, pr->scale);
            y               = ggml_add(ctx, y, ggml_mul_mat(ctx, pr->B, t));
        } else if (pr && pr->has_lokr()) {
            // Shared with the inference runtime (qwen3-lora.h) on purpose: one kron
            // implementation, so the ne2 lesson cannot be learned twice.
            y = qwen3_lokr_delta(ctx, pr, x, y);
        }
    }
    return y;
}

static ggml_tensor * lm_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), qwen3_f32(ctx, w));
}

// ─── head-blocked attention (plan §3.7, D6/D7) ──────────────────────────────
//
// Replaces the reshape -> qk-norm -> rope -> permute -> qwen3_attn_f32 ->
// reshape span with `Nh/gq` independent blocks, so at most `3 * gq * S^2 * 4`
// bytes of [S,S] score/softmax state is live at once instead of `3 * Nh * S^2`.
// Blocks are reassembled with ggml_acc into a persistent ZERO base:
// Splice a window's K (or V) into the tail of a frozen prefix store and return
// the whole [Nkv*D, n_pfx + S] span as one tensor. The store's tail columns are
// zero, so the ggml_acc is a copy; see LmLayerOpts::kv_k for why it is an acc.
static ggml_tensor * lm_kv_splice(ggml_context * ctx, ggml_tensor * store, ggml_tensor * win, int64_t row,
                                  int64_t n_pfx, int64_t S) {
    GGML_ASSERT(store->ne[0] == row && store->ne[1] >= n_pfx + S);
    ggml_tensor * base = ggml_view_2d(ctx, store, row, n_pfx + S, store->nb[1], 0);
    ggml_tensor * w2   = ggml_reshape_2d(ctx, win, row, S);
    return ggml_acc(ctx, base, w2, base->nb[1], base->nb[2], base->nb[3], (size_t) n_pfx * store->nb[1]);
}

// GGML_OP_CONCAT has no backward (it would hit ggml_compute_backward's
// GGML_ABORT default), GGML_OP_ACC does, and its src1 branch already inserts a
// ggml_cont before ggml_reshape so the strided gradient view is legal.
//
// o_proj (and its LoRA) then runs ONCE on the assembled [Nh*D, S] tensor.
static ggml_tensor * lm_attn_head_blocked(ggml_context * ctx, const Qwen3LMConfig & c, Qwen3Layer * ly,
                                          ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * positions,
                                          ggml_tensor * mask, int S, const LmLayerOpts & opts) {
    const int D = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    const int gq = opts.attn_head_block;

    // The blocked path ropes per head-block, so sharing one roped K with a
    // prefix store would mean hoisting that out and changing the shipped graph.
    // MM3 never sets attn_head_block; refuse rather than differ silently.
    GGML_ASSERT(opts.kv_k == nullptr && opts.kv_cap == nullptr &&
                "attn_head_block is not supported with a frozen KV prefix");
    // D3: there is no fused head-blocked arm, and there is no reason to build
    // one — blocking exists to cap the 3*hb*S^2 score/softmax transient, which
    // the fused op does not have. The CLI refuses the pair at exit 2 and
    // lm_ckpt_default_head_block returns 0 under flash, so reaching here means a
    // caller constructed the combination directly.
    GGML_ASSERT(!opts.attn_flash && "--attn flash is not supported with --attn-head-block > 0");
    GGML_ASSERT(gq > 0 && gq < Nh && Nh % gq == 0);
    GGML_ASSERT(((int64_t) gq * (int64_t) Nkv) % (int64_t) Nh == 0);
    const int gkv = gq * Nkv / Nh;
    GGML_ASSERT(gkv >= 1);
    GGML_ASSERT(opts.attn_zero != nullptr);
    GGML_ASSERT(opts.attn_zero->ne[0] == (int64_t) Nh * (int64_t) D && opts.attn_zero->ne[1] >= (int64_t) S);

    // Contiguous ne1-prefix view of the permanently-zero base. Starting from a
    // garbage leaf would poison the rows the first ggml_acc does not write.
    ggml_tensor * acc = ggml_view_2d(ctx, opts.attn_zero, (int64_t) Nh * D, S, opts.attn_zero->nb[1], 0);

    for (int b = 0; b < Nh / gq; b++) {
        ggml_tensor * qb =
            ggml_cont(ctx, ggml_view_2d(ctx, q, (int64_t) gq * D, S, q->nb[1], (size_t) b * gq * D * sizeof(float)));
        ggml_tensor * kb =
            ggml_cont(ctx, ggml_view_2d(ctx, k, (int64_t) gkv * D, S, k->nb[1], (size_t) b * gkv * D * sizeof(float)));
        ggml_tensor * vb =
            ggml_cont(ctx, ggml_view_2d(ctx, v, (int64_t) gkv * D, S, v->nb[1], (size_t) b * gkv * D * sizeof(float)));

        qb = ggml_reshape_3d(ctx, qb, D, gq, S);
        kb = ggml_reshape_3d(ctx, kb, D, gkv, S);
        vb = ggml_reshape_3d(ctx, vb, D, gkv, S);

        qb = ggml_mul(ctx, ggml_rms_norm(ctx, qb, c.rms_norm_eps), qwen3_f32(ctx, ly->q_norm));
        kb = ggml_mul(ctx, ggml_rms_norm(ctx, kb, c.rms_norm_eps), qwen3_f32(ctx, ly->k_norm));

        qb = ggml_rope_ext(ctx, qb, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        kb = ggml_rope_ext(ctx, kb, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        qb = ggml_permute(ctx, qb, 0, 2, 1, 3);
        kb = ggml_permute(ctx, kb, 0, 2, 1, 3);
        vb = ggml_permute(ctx, vb, 0, 2, 1, 3);

        ggml_tensor * ab = qwen3_attn_f32(ctx, qb, kb, vb, mask, 1.0f / sqrtf((float) D));  // [D, gq, S]
        ab               = ggml_reshape_2d(ctx, ab, (int64_t) gq * D, S);
        acc = ggml_acc(ctx, acc, ab, acc->nb[1], acc->nb[2], acc->nb[3], (size_t) b * gq * D * sizeof(float));
    }
    return acc;  // [Nh*D, S]
}

// One transformer layer, cache-free, manual attention. hidden: [H, S]
static ggml_tensor * lm_train_layer(ggml_context * ctx, const Qwen3LMConfig & c, Qwen3Layer * ly, ggml_tensor * hidden,
                                    ggml_tensor * positions, ggml_tensor * mask, int S, const LmLayerOpts & opts) {
    const int           D  = c.head_dim, Nh = c.n_heads, Nkv = c.n_kv_heads;
    const QwLoraLayer * ll = ly->lora;

    ggml_tensor * x = lm_rms(ctx, hidden, ly->input_layernorm, c.rms_norm_eps);

    ggml_tensor * q = lm_linear(ctx, ly->q_proj, qwen3_lora_slot(ll, QW_LORA_Q), x, opts);
    ggml_tensor * k = lm_linear(ctx, ly->k_proj, qwen3_lora_slot(ll, QW_LORA_K), x, opts);
    ggml_tensor * v = lm_linear(ctx, ly->v_proj, qwen3_lora_slot(ll, QW_LORA_V), x, opts);

    ggml_tensor * attn = nullptr;
    if (opts.attn_head_block > 0 && opts.attn_head_block < Nh) {
        attn = lm_attn_head_blocked(ctx, c, ly, q, k, v, positions, mask, S, opts);
    } else {
        const int64_t n_pfx = opts.kv_k ? (int64_t) opts.kv_pfx : 0;
        const int64_t n_kv  = n_pfx + S;

        q = ggml_reshape_3d(ctx, q, D, Nh, S);
        k = ggml_reshape_3d(ctx, k, D, Nkv, S);
        v = ggml_reshape_3d(ctx, v, D, Nkv, S);

        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_norm_eps), qwen3_f32(ctx, ly->q_norm));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_norm_eps), qwen3_f32(ctx, ly->k_norm));

        q = ggml_rope_ext(ctx, q, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, positions, NULL, D, 2, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Prefill: record what this call computed BEFORE the splice, so a later
        // window sees exactly the K/V a single long crop would have produced.
        if (opts.kv_cap) {
            opts.kv_cap->nodes.push_back(
                ggml_cpy(ctx, ggml_reshape_2d(ctx, k, (int64_t) Nkv * D, S), opts.kv_cap->k_dst));
            opts.kv_cap->nodes.push_back(
                ggml_cpy(ctx, ggml_reshape_2d(ctx, v, (int64_t) Nkv * D, S), opts.kv_cap->v_dst));
        }

        // The window's K/V are spliced in AFTER RoPE, because the store already
        // holds the prefix roped at ITS positions — which is what makes the
        // relative offsets seen here identical to a single long crop's.
        if (opts.kv_k) {
            k = ggml_reshape_3d(ctx, lm_kv_splice(ctx, opts.kv_k, k, (int64_t) Nkv * D, n_pfx, S), D, Nkv, n_kv);
            v = ggml_reshape_3d(ctx, lm_kv_splice(ctx, opts.kv_v, v, (int64_t) Nkv * D, n_pfx, S), D, Nkv, n_kv);
        }

        q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, S,    Nh ]
        k = ggml_permute(ctx, k, 0, 2, 1, 3);  // [D, n_kv, Nkv]
        v = ggml_permute(ctx, v, 0, 2, 1, 3);

        // D1/D2 — THE ONE MODE SWITCH. The false arm is the original call,
        // untouched, so with the flag off this layer emits the byte-identical
        // node sequence it emitted before the flag existed.
        attn = opts.attn_flash
                   ? lm_attn_flash(ctx, q, k, v, mask, 1.0f / sqrtf((float) D), opts.attn_prec)
                   : qwen3_attn_f32(ctx, q, k, v, mask, 1.0f / sqrtf((float) D));
        attn = ggml_reshape_2d(ctx, attn, Nh * D, S);
    }

    ggml_tensor * ao = lm_linear(ctx, ly->o_proj, qwen3_lora_slot(ll, QW_LORA_O), attn, opts);
    hidden           = ggml_add(ctx, hidden, ao);

    ggml_tensor * xm = lm_rms(ctx, hidden, ly->post_attn_layernorm, c.rms_norm_eps);
    ggml_tensor * g  = lm_linear(ctx, ly->gate_proj, qwen3_lora_slot(ll, QW_LORA_GATE), xm, opts);
    ggml_tensor * u  = lm_linear(ctx, ly->up_proj, qwen3_lora_slot(ll, QW_LORA_UP), xm, opts);
    ggml_tensor * ff = ggml_swiglu_split(ctx, g, u);  // fused ggml_swiglu has NO backward
    ggml_tensor * dn = lm_linear(ctx, ly->down_proj, qwen3_lora_slot(ll, QW_LORA_DOWN), ff, opts);
    return ggml_add(ctx, hidden, dn);
}

// Shipped 7-argument form — a default LmLayerOpts, i.e. the naive graph.
static inline ggml_tensor * lm_train_layer(ggml_context * ctx, const Qwen3LMConfig & c, Qwen3Layer * ly,
                                           ggml_tensor * hidden, ggml_tensor * positions, ggml_tensor * mask, int S) {
    const LmLayerOpts opts;
    return lm_train_layer(ctx, c, ly, hidden, positions, mask, S, opts);
}

// `t_msk_flat` is the persistent [S_MAX*S_MAX] mask buffer; the graph views a
// CONTIGUOUS [S, S] block of it, because ggml_soft_max_ext asserts
// ggml_is_contiguous(mask) && mask->ne[0] == scores->ne[0] — and so does
// ggml_flash_attn_train, which additionally requires F16.
//
// The row stride is `S * ggml_element_size(t_msk_flat)`, not `S * 4`: under
// --attn flash the buffer is F16 with the same flat layout (D4, lm_mask_alloc).
// For an F32 buffer the two expressions are the same number, which is what keeps
// exact mode byte-identical.
//
// ─── Artist token (textual inversion) ───────────────────────────────────────
//
// A [H, k] parameter added onto the embeddings of k contiguous positions —
// the placeholder ids that prompt.h spliced into the caption span. Learning a
// DELTA rather than a replacement means zero-init starts the run at a real
// token the model already understands, and it needs no `ggml_set` semantics:
// ggml_acc's backward hands src1 a strided view of the output gradient at the
// same offset, which is exactly this parameter's gradient.
//
// Nothing here touches a weight. The base model, and any LoRA layered on top of
// it, are untouched — which is the whole reason this parameterization cannot
// produce the loop/early-EOS degeneracies that adapter training can.
struct LmArtistTok {
    ggml_tensor * t   = nullptr;  // [H, k] F32 param
    int           k   = 0;
    int           off = -1;       // first placeholder position in the sequence

    bool active() const { return t && k > 0 && off >= 0; }
};

// layer_lo/layer_hi let the self-test build a 2-layer slice (T5).
static ggml_tensor * lm_build_trunk(ggml_context * ctx, Qwen3LM * lm, ggml_tensor * tokens, ggml_tensor * pos,
                                    ggml_tensor * t_msk_flat, int S, int layer_lo, int layer_hi,
                                    const LmLayerOpts & opts, const LmArtistTok * art = nullptr) {
    const Qwen3LMConfig & c = lm->cfg;

    ggml_tensor * tok_v = (tokens->ne[0] == S) ? tokens : ggml_view_1d(ctx, tokens, S, 0);
    ggml_tensor * pos_v = (pos->ne[0] == S) ? pos : ggml_view_1d(ctx, pos, S, 0);

    ggml_tensor * mask = ggml_view_2d(ctx, t_msk_flat, S, S, (size_t) S * ggml_element_size(t_msk_flat), 0);
    ggml_tensor * h    = ggml_get_rows(ctx, lm->embed_tokens, tok_v);  // [H, S]
    if (art && art->active()) {
        GGML_ASSERT(art->off + art->k <= S);
        GGML_ASSERT(art->t->ne[0] == h->ne[0] && art->t->ne[1] == art->k);
        h = ggml_acc(ctx, h, art->t, h->nb[1], h->nb[2], h->nb[3], (size_t) art->off * h->nb[1]);
    }
    for (int l = layer_lo; l < layer_hi; l++) {
        h = lm_train_layer(ctx, c, &lm->layers[l], h, pos_v, mask, S, opts);
    }
    return lm_rms(ctx, h, lm->final_norm, c.rms_norm_eps);
}

static ggml_tensor * lm_build_trunk(ggml_context * ctx, Qwen3LM * lm, ggml_tensor * tokens, ggml_tensor * pos,
                                    ggml_tensor * t_msk_flat, int S, int layer_lo, int layer_hi) {
    const Qwen3LMConfig & c = lm->cfg;

    // `tokens` / `pos` are persistent buffers sized for the LONGEST sample, so
    // every shorter song must view only its own prefix — otherwise get_rows
    // emits [H, max_seq] and the per-head reshape trips
    // GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2).
    ggml_tensor * tok_v = (tokens->ne[0] == S) ? tokens : ggml_view_1d(ctx, tokens, S, 0);
    ggml_tensor * pos_v = (pos->ne[0] == S) ? pos : ggml_view_1d(ctx, pos, S, 0);

    ggml_tensor * mask = ggml_view_2d(ctx, t_msk_flat, S, S, (size_t) S * ggml_element_size(t_msk_flat), 0);
    ggml_tensor * h    = ggml_get_rows(ctx, lm->embed_tokens, tok_v);  // [H, S]
    for (int l = layer_lo; l < layer_hi; l++) {
        h = lm_train_layer(ctx, c, &lm->layers[l], h, pos_v, mask, S);
    }
    return lm_rms(ctx, h, lm->final_norm, c.rms_norm_eps);
}

static inline ggml_tensor * lm_build_trunk(ggml_context * ctx, Qwen3LM * lm, ggml_tensor * tokens, ggml_tensor * pos,
                                           ggml_tensor * t_msk_flat, int S) {
    return lm_build_trunk(ctx, lm, tokens, pos, t_msk_flat, S, 0, lm->cfg.n_layers);
}

// Same trunk, entered from INPUTS-EMBEDS [H, S] instead of token ids.
//
// ACE's planner LM is fed tokens and nothing else, so the id form above is the
// only entry it ever needed. MiniMax-Music3 is not: an audio frame's input is
// (token_embd[semantic] + SUM_c audio_embd[code_c]) * num_codebooks^-0.5, built
// from two tables in two different files, so the embedding is computed by the
// caller and handed in whole (train/mm3-lm-load.h, tools/ace-train.cpp
// mm3-lm-loss). Nothing below the embedding differs, which is the entire point
// of splitting here rather than forking the trunk.
// The opts overload exists for symmetry with lm_build_trunk; note that
// cast_weights is DOCUMENTATION ONLY (see the LmLayerOpts note above —
// lm_linear routes every weight through qwen3_f32, which returns an already-F32
// tensor unchanged), so it does not switch precision and cannot be used to make
// two paths numerically comparable.
static ggml_tensor * lm_build_trunk_embeds(ggml_context * ctx, Qwen3LM * lm, ggml_tensor * h, ggml_tensor * pos,
                                           ggml_tensor * t_msk_flat, int S, int layer_lo, int layer_hi,
                                           const LmLayerOpts & opts) {
    const Qwen3LMConfig & c = lm->cfg;

    ggml_tensor * pos_v = (pos->ne[0] == S) ? pos : ggml_view_1d(ctx, pos, S, 0);
    ggml_tensor * mask  = ggml_view_2d(ctx, t_msk_flat, S, S, (size_t) S * ggml_element_size(t_msk_flat), 0);
    for (int l = layer_lo; l < layer_hi; l++) {
        h = lm_train_layer(ctx, c, &lm->layers[l], h, pos_v, mask, S, opts);
    }
    return lm_rms(ctx, h, lm->final_norm, c.rms_norm_eps);
}

static inline ggml_tensor * lm_build_trunk_embeds(ggml_context * ctx, Qwen3LM * lm, ggml_tensor * h,
                                                  ggml_tensor * pos, ggml_tensor * t_msk_flat, int S,
                                                  int layer_lo, int layer_hi) {
    return lm_build_trunk_embeds(ctx, lm, h, pos, t_msk_flat, S, layer_lo, layer_hi, LmLayerOpts{});
}

static inline ggml_tensor * lm_build_trunk_embeds(ggml_context * ctx, Qwen3LM * lm, ggml_tensor * h,
                                                  ggml_tensor * pos, ggml_tensor * t_msk_flat, int S) {
    return lm_build_trunk_embeds(ctx, lm, h, pos, t_msk_flat, S, 0, lm->cfg.n_layers, LmLayerOpts{});
}
