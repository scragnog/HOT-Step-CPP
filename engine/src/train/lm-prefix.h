#pragma once
// lm-prefix.h — trainable per-layer K/V prefixes (prefix tuning / P-tuning v2)
// for the planner LM.
//
// WHAT IT IS. n learned key/value pairs per layer, sitting in front of the real
// sequence in every attention. Where the artist token (V2) learns k vectors at
// ONE point — the input embedding — this learns 2 * L * n vectors at depth, and
// steers the computation layer by layer. On the 0.6B (H=1024, 28L) that is 56x
// the parameters of a k=8 token; on MM3 (H=4096, 36L) 72x. Still no weight is
// touched: the base model, and any LoRA on it, are bit-identical to before, so
// it keeps both properties that made tokens attractive — it cannot damage the
// base, and it composes with a LoRA with no merge arithmetic.
//
// HOW IT RIDES THE EXISTING MACHINERY. lm-kvprefix.h already teaches
// lm_train_layer to attend over [prefix ; window] through LmLayerOpts::kv_k/kv_v
// — a store whose first kv_pfx columns are prefix K/V and whose tail is zero,
// with the window's own K/V spliced in by ggml_acc. The only difference here is
// that the prefix columns are PARAMETERS rather than a frozen history:
//
//     store = acc(zero_base, P_k, off 0)        <- P_k is src1: it gets a gradient
//     k     = acc(store,     K_win, off n)      <- unchanged lm_kv_splice
//
// GGML_OP_ACC has a backward for src1 (a strided view of the output gradient),
// and GGML_OP_CONCAT has none, which is why every join in this path is an acc.
//
// THREE CONTRACTS, stated because each is a silent failure if missed:
//
//   1. Prefix K is stored POST-QK-NORM and WITHOUT RoPE. The frozen store holds
//      roped keys at their true positions; a learned prefix has no position,
//      exactly like HF prefix-tuning's raw past_key_values. The window's own K
//      is roped as usual. Inference must write P_k into the cache raw, never
//      rope it.
//   2. Every row sees the whole prefix, INCLUDING the caption rows. The frozen
//      prefix blanks the caption's view of it (that prefix is audio history the
//      caption precedes); this prefix is conditioning, and blinding the caption
//      to it would train a prefix the prompt cannot read. lm_causal_mask_prefix
//      with pfx_lo = 0 and n_prompt = 0.
//   3. Window positions are 0..S-1, unshifted. The prefix occupies KV columns
//      [0, n) but no RoPE positions. Inference must reproduce this: kv_pos
//      starts at n, positions are computed from kv_pos - n.
//
// EXACT ATTENTION ONLY. S_kv != S is outside what the --attn flash probe covers
// (lm-train-run.h's note about the frozen prefix says the same). Refuse rather
// than emit an unprobed fused graph.
//
// INIT. Small normal on BOTH K and V, sigma 0.02. Zero-initialising V would give
// K exactly zero gradient forever — dL/dK is proportional to V through the
// attention output — and the run would look like "the prefix never moved".

#include "qwen3-lm.h"
#include "train/lm-common.h"

#include <cmath>
#include <string>
#include <vector>

struct LmPrefix {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // Indexed by ABSOLUTE layer; nullptr outside [layer_lo, layer_hi).
    std::vector<ggml_tensor *> k, v;  // [Nkv*D, n] F32 params
    std::vector<ggml_tensor *> params;

    // Persistent zero base for the acc, [Nkv*D, n + s_max]. One is enough: acc
    // is not in-place, so every layer reads the same zeros and none writes them.
    ggml_tensor * zero = nullptr;

    int     n        = 0;
    int     layer_lo = 0, layer_hi = 0;
    int64_t row      = 0;  // Nkv * D
    size_t  n_params = 0;

    bool active() const { return n > 0 && !k.empty(); }
};

static void lm_prefix_free(LmPrefix * P) {
    if (P->buf) {
        ggml_backend_buffer_free(P->buf);
    }
    if (P->ctx) {
        ggml_free(P->ctx);
    }
    *P = LmPrefix{};
}

static size_t lm_prefix_bytes(const Qwen3LMConfig & c, int layer_lo, int layer_hi, int n, int64_t s_max) {
    const int64_t row = (int64_t) c.n_kv_heads * (int64_t) c.head_dim;
    const int64_t nl  = (int64_t) (layer_hi - layer_lo);
    return (size_t) nl * 2u * (size_t) row * (size_t) n * sizeof(float)  // params
         + (size_t) row * (size_t) (n + s_max) * sizeof(float);          // zero base
}

static bool lm_prefix_init(LmPrefix * P, Qwen3LM * lm, int layer_lo, int layer_hi, int n, int64_t s_max,
                           uint64_t seed, float sigma, std::string * err) {
    const Qwen3LMConfig & c = lm->cfg;
    if (n <= 0 || layer_hi <= layer_lo || s_max <= 0) {
        *err = "invalid prefix configuration (n / layers / s_max)";
        return false;
    }
    P->n        = n;
    P->layer_lo = layer_lo;
    P->layer_hi = layer_hi;
    P->row      = (int64_t) c.n_kv_heads * (int64_t) c.head_dim;

    const int        nl = layer_hi - layer_lo;
    ggml_init_params p  = { (size_t) (2 * nl + 4) * ggml_tensor_overhead(), nullptr, true };
    P->ctx              = ggml_init(p);
    if (!P->ctx) {
        *err = "cannot create the prefix context";
        return false;
    }
    P->k.assign((size_t) c.n_layers, nullptr);
    P->v.assign((size_t) c.n_layers, nullptr);
    for (int l = layer_lo; l < layer_hi; l++) {
        char nm[48];
        P->k[(size_t) l] = ggml_new_tensor_2d(P->ctx, GGML_TYPE_F32, P->row, n);
        snprintf(nm, sizeof(nm), "L%d.prefix_k", l);
        ggml_set_name(P->k[(size_t) l], nm);
        ggml_set_param(P->k[(size_t) l]);
        P->v[(size_t) l] = ggml_new_tensor_2d(P->ctx, GGML_TYPE_F32, P->row, n);
        snprintf(nm, sizeof(nm), "L%d.prefix_v", l);
        ggml_set_name(P->v[(size_t) l], nm);
        ggml_set_param(P->v[(size_t) l]);
        P->params.push_back(P->k[(size_t) l]);
        P->params.push_back(P->v[(size_t) l]);
    }
    P->zero = ggml_new_tensor_2d(P->ctx, GGML_TYPE_F32, P->row, (int64_t) n + s_max);
    ggml_set_name(P->zero, "prefix.zero");
    ggml_set_input(P->zero);

    P->buf = ggml_backend_alloc_ctx_tensors(P->ctx, lm->backend);
    if (!P->buf) {
        *err = "prefix parameter buffer allocation failed";
        return false;
    }
    ggml_backend_buffer_clear(P->buf, 0);  // zero base: once, then never written

    LmRng rng;
    lm_rng_seed(&rng, seed ^ 0x9E3779B97F4A7C15ull);
    std::vector<float> tmp;
    for (int l = layer_lo; l < layer_hi; l++) {
        for (int which = 0; which < 2; which++) {
            ggml_tensor * t = which ? P->v[(size_t) l] : P->k[(size_t) l];
            tmp.assign((size_t) ggml_nelements(t), 0.0f);
            lm_rng_fill_normal(&rng, tmp, sigma);
            ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size() * sizeof(float));
            P->n_params += tmp.size();
        }
    }
    return true;
}
