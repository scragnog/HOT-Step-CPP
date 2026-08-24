#pragma once
// minimax/mm3-lm-adapter.h — runtime LoRA adapters for the MM3 global LM.
//
// HOT-Step file (does not exist upstream). Loads a PEFT-format LM LoRA
// (SimpleTuner `--minimax_music_train_component=language_model` checkpoints:
// keys `language_model.model.layers.N.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,
// down}_proj}.lora_{A,B}.weight`) and applies it as RUNTIME low-rank deltas
// inside mm3-lm-graph.h — the base weights are never modified, so per-group
// scales are live per generation with no reload.
//
// Why runtime deltas and not a load-time merge: the whole point (validated by
// the 2026-08-20 ablation grid on the alk3 r256 adapter) is DIALING groups at
// render time — attention carries the plan/genre, the MLPs carry vocal
// identity AND the fidelity damage, and attention 1.0 / MLP 0.5 was the ear
// winner. A merge would freeze one setting into 17 GB of resident weights.
//
// Scale model (multiplicative, all default 1.0):
//     effective(module, layer) = global
//                              * (attn | mlp)          by module kind
//                              * (early | mid | late)  by layer/12
// PEFT's own alpha/rank scaling: SimpleTuner LM checkpoints carry NO alpha
// tensors and train with alpha == rank, so the baked base scale is 1.0. If a
// per-module `.alpha` scalar IS present (comfy-style exports), it is honoured
// as alpha/rank.
//
// Cost: r256 f16 adapter ≈ 1.5 GB resident, streamed per AR token on top of
// the 17.2 GB base — expect ≈ +9 % on the LM step at rank 256, proportionally
// less at lower ranks. VRAM and step cost both scale linearly with rank.
//
// Scales are baked into cached graphs as constants; mm3-lm-graph.h owns an
// `adapter_epoch` and rebuilds its slots when (adapter, scales) change. That
// is once per generation, not per frame — rebuild cost is graph construction
// only.

#include "backend.h"
#include "safetensors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#define MM3_LM_ADAPTER_LAYERS  36
#define MM3_LM_ADAPTER_MODULES 7

enum MM3LmAdapterModule {
    MM3_LM_ADAPTER_Q = 0,
    MM3_LM_ADAPTER_K,
    MM3_LM_ADAPTER_V,
    MM3_LM_ADAPTER_O,
    MM3_LM_ADAPTER_GATE,
    MM3_LM_ADAPTER_UP,
    MM3_LM_ADAPTER_DOWN,
};

static const char * MM3_LM_ADAPTER_MODULE_KEY[MM3_LM_ADAPTER_MODULES] = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj",
};

struct MM3LmAdapterScales {
    float global = 1.0f;
    float attn   = 1.0f;
    float mlp    = 1.0f;
    float early  = 1.0f;  // layers 0..11
    float mid    = 1.0f;  // layers 12..23
    float late   = 1.0f;  // layers 24..35

    bool operator==(const MM3LmAdapterScales & o) const {
        return global == o.global && attn == o.attn && mlp == o.mlp && early == o.early && mid == o.mid &&
               late == o.late;
    }

    bool operator!=(const MM3LmAdapterScales & o) const { return !(*this == o); }
};

// Which component of a module a safetensors key carries. PEFT LoRA files use
// A/B; LyCORIS LoKr files use w1 plus either a monolithic w2 or the w2_a/w2_b
// pair, and a scalar alpha.
enum MM3LmAdapterComp {
    MM3_LM_COMP_NONE = 0,
    MM3_LM_COMP_A,
    MM3_LM_COMP_B,
    MM3_LM_COMP_W1,
    MM3_LM_COMP_W2,
    MM3_LM_COMP_W2A,
    MM3_LM_COMP_W2B,
    MM3_LM_COMP_ALPHA,
};

struct MM3LmAdapterPair {
    ggml_tensor * a          = nullptr;  // [in, r]  (f16)
    ggml_tensor * b          = nullptr;  // [r, out] (f16)
    float         base_scale = 1.0f;     // alpha/rank when an alpha scalar exists, else 1

    // ── LoKr (dW = kron(w1, w2) * lokr_scale) ──────────────────────────────
    // ggml extents, mirroring lm_lokr_init: w1 [in_m, out_l], w2 [in_n, out_k],
    // w2_a [dim, out_k], w2_b [in_n, dim]. Geometry is read back off the
    // tensors so nothing here has to parse the LyCORIS config blob.
    ggml_tensor * w1         = nullptr;
    ggml_tensor * w2         = nullptr;  // monolithic
    ggml_tensor * w2_a       = nullptr;  // factorized
    ggml_tensor * w2_b       = nullptr;
    int64_t       in_m = 0, in_n = 0, out_l = 0, out_k = 0;
    float         lokr_scale = 1.0f;
    float         alpha_raw  = 0.0f;     // the file's alpha scalar, pre-division
    bool          has_alpha  = false;

    bool has_lora() const { return a && b; }
    bool has_lokr() const { return w1 && (w2 || (w2_a && w2_b)); }
};

struct MM3LmAdapter {
    std::string      path;
    int64_t          mtime = 0;
    int              rank  = 0;   // LoRA only; 0 for LoKr
    bool             is_lokr = false;
    MM3LmAdapterPair mods[MM3_LM_ADAPTER_LAYERS][MM3_LM_ADAPTER_MODULES];
    int              n_loaded = 0;

    ggml_context *        ctx         = nullptr;
    ggml_backend_buffer_t buf         = nullptr;
    BackendPair           bp          = {};
    bool                  backend_ref = false;

    // effective scale for one module instance under the request's dials
    float effective(int layer, int module, const MM3LmAdapterScales & s) const {
        const MM3LmAdapterPair & p = mods[layer][module];
        float                    v = s.global * p.base_scale;
        v *= (module <= MM3_LM_ADAPTER_O) ? s.attn : s.mlp;
        v *= (layer < 12) ? s.early : (layer < 24) ? s.mid : s.late;
        return v;
    }
};

static void mm3_lm_adapter_free(MM3LmAdapter * ad) {
    if (!ad) {
        return;
    }
    if (ad->buf) {
        ggml_backend_buffer_free(ad->buf);
    }
    if (ad->ctx) {
        ggml_free(ad->ctx);
    }
    if (ad->backend_ref) {
        backend_release(ad->bp.backend, ad->bp.cpu_backend);
    }
    delete ad;
}

// Accept keys with or without the `language_model.` prefix, and PEFT's
// `.default.` infix (present after PeftModel round-trips).
static bool mm3_lm_adapter_parse_key(const std::string & key, int * layer, int * module, bool * is_a) {
    const char * s = key.c_str();
    if (strncmp(s, "language_model.", 15) == 0) {
        s += 15;
    }
    if (strncmp(s, "base_model.model.", 17) == 0) {
        s += 17;
    }
    if (strncmp(s, "model.layers.", 13) != 0) {
        return false;
    }
    s += 13;
    char * end   = nullptr;
    long   lyr   = strtol(s, &end, 10);
    if (end == s || *end != '.' || lyr < 0 || lyr >= MM3_LM_ADAPTER_LAYERS) {
        return false;
    }
    s = end + 1;
    int mod = -1;
    for (int m = 0; m < MM3_LM_ADAPTER_MODULES; m++) {
        size_t n = strlen(MM3_LM_ADAPTER_MODULE_KEY[m]);
        if (strncmp(s, MM3_LM_ADAPTER_MODULE_KEY[m], n) == 0 && s[n] == '.') {
            mod = m;
            s += n + 1;
            break;
        }
    }
    if (mod < 0) {
        return false;
    }
    // remainder: lora_A.weight / lora_B.weight, optionally lora_A.default.weight
    if (strncmp(s, "lora_A.", 7) == 0) {
        *is_a = true;
    } else if (strncmp(s, "lora_B.", 7) == 0) {
        *is_a = false;
    } else {
        return false;
    }
    *layer  = (int) lyr;
    *module = mod;
    return true;
}

// LyCORIS LoKr keys, as written by lm_export_lokr:
//   lycoris_layers_<L>_<site with dots as underscores>.lokr_w1 | .lokr_w2
//                                                     | .lokr_w2_a | .lokr_w2_b
//                                                     | .alpha
static MM3LmAdapterComp mm3_lm_adapter_parse_lokr(const std::string & key, int * layer, int * module) {
    const char * s = key.c_str();
    if (strncmp(s, "lycoris_layers_", 15) != 0) {
        return MM3_LM_COMP_NONE;
    }
    s += 15;
    char * end = nullptr;
    long   lyr = strtol(s, &end, 10);
    if (end == s || *end != '_' || lyr < 0 || lyr >= MM3_LM_ADAPTER_LAYERS) {
        return MM3_LM_COMP_NONE;
    }
    s = end + 1;
    // Site name with '.' replaced by '_', so compare against the module keys
    // under the same substitution rather than keeping a second table.
    int mod = -1;
    for (int m = 0; m < MM3_LM_ADAPTER_MODULES; m++) {
        std::string want(MM3_LM_ADAPTER_MODULE_KEY[m]);
        for (size_t i = 0; i < want.size(); i++) {
            if (want[i] == '.') {
                want[i] = '_';
            }
        }
        if (strncmp(s, want.c_str(), want.size()) == 0 && s[want.size()] == '.') {
            mod = m;
            s += want.size() + 1;
            break;
        }
    }
    if (mod < 0) {
        return MM3_LM_COMP_NONE;
    }
    MM3LmAdapterComp c = MM3_LM_COMP_NONE;
    // Longest first: lokr_w2_a / lokr_w2_b must not be swallowed by lokr_w2.
    if (strcmp(s, "lokr_w2_a") == 0) {
        c = MM3_LM_COMP_W2A;
    } else if (strcmp(s, "lokr_w2_b") == 0) {
        c = MM3_LM_COMP_W2B;
    } else if (strcmp(s, "lokr_w1") == 0) {
        c = MM3_LM_COMP_W1;
    } else if (strcmp(s, "lokr_w2") == 0) {
        c = MM3_LM_COMP_W2;
    } else if (strcmp(s, "alpha") == 0) {
        c = MM3_LM_COMP_ALPHA;
    } else {
        return MM3_LM_COMP_NONE;
    }
    *layer  = (int) lyr;
    *module = mod;
    return c;
}

// One parse for both layouts. MM3_LM_COMP_NONE means "not an LM adapter key".
static MM3LmAdapterComp mm3_lm_adapter_parse_any(const std::string & key, int * layer, int * module) {
    bool is_a = false;
    if (mm3_lm_adapter_parse_key(key, layer, module, &is_a)) {
        return is_a ? MM3_LM_COMP_A : MM3_LM_COMP_B;
    }
    return mm3_lm_adapter_parse_lokr(key, layer, module);
}

// Where a component lands in the pair. ALPHA is a scalar, not a stored tensor,
// so it has no slot and is handled during upload.
static ggml_tensor ** mm3_lm_pair_slot(MM3LmAdapterPair & p, MM3LmAdapterComp c) {
    switch (c) {
        case MM3_LM_COMP_A:   return &p.a;
        case MM3_LM_COMP_B:   return &p.b;
        case MM3_LM_COMP_W1:  return &p.w1;
        case MM3_LM_COMP_W2:  return &p.w2;
        case MM3_LM_COMP_W2A: return &p.w2_a;
        case MM3_LM_COMP_W2B: return &p.w2_b;
        default:              return nullptr;
    }
}

// Load a PEFT LM LoRA. Acquires its own backend reference (same shared pool
// as every other module). Returns nullptr with a message on any structural
// problem — a half-loaded adapter is worse than none (the LM-echo whitelist
// lesson: silently dropping modules changes what the adapter IS).
static MM3LmAdapter * mm3_lm_adapter_load(const char * path, std::string * err) {
    STFile st;
    if (!st_open(&st, path)) {
        if (err) {
            *err = std::string("cannot open adapter: ") + path;
        }
        return nullptr;
    }

    struct stat sb {};

    stat(path, &sb);

    MM3LmAdapter * ad = new MM3LmAdapter();
    ad->path          = path;
    ad->mtime         = (int64_t) sb.st_mtime;
    ad->bp            = backend_init("MM3-LM-Adapter");
    ad->backend_ref   = true;
    ggml_backend_t backend = ad->bp.backend ? ad->bp.backend : ad->bp.cpu_backend;

    // Pass 1: count matched pairs so the ggml context can be sized exactly.
    int matched = 0;   // tensor-backed components only (alpha is a scalar)
    int n_alpha = 0;
    for (const STEntry & e : st.entries) {
        int                    layer, module;
        const MM3LmAdapterComp c = mm3_lm_adapter_parse_any(e.name, &layer, &module);
        if (c == MM3_LM_COMP_ALPHA) {
            n_alpha++;
        } else if (c != MM3_LM_COMP_NONE) {
            matched++;
        }
    }
    if (matched == 0) {
        st_close(&st);
        if (err) {
            *err = "no language_model LoRA/LoKr keys found (is this a DiT adapter?)";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }
    (void) n_alpha;

    ggml_init_params ip = {
        /*mem_size   =*/(size_t) (matched + 2) * ggml_tensor_overhead(),
        /*mem_buffer =*/nullptr,
        /*no_alloc   =*/true,
    };
    ad->ctx = ggml_init(ip);

    // Pass 2: create tensors (f16, torch row-major maps directly onto ggml
    // [ne0 = innermost]): A [r, in] -> ggml [in, r]; B [out, r] -> ggml [r, out].
    for (const STEntry & e : st.entries) {
        int                    layer, module;
        const MM3LmAdapterComp c = mm3_lm_adapter_parse_any(e.name, &layer, &module);
        if (c == MM3_LM_COMP_NONE || c == MM3_LM_COMP_ALPHA) {
            continue;   // alpha is read during upload; it gets no tensor
        }
        if (e.n_dims != 2) {
            if (err) {
                *err = "unexpected adapter tensor rank on " + e.name;
            }
            st_close(&st);
            mm3_lm_adapter_free(ad);
            return nullptr;
        }
        // torch shape[0] = rows, shape[1] = cols; ggml ne0 is the innermost.
        // LoRA:  A [r, in]     -> [in, r]      B [out, r]   -> [r, out]
        // LoKr:  w1 [out_l, in_m] -> [in_m, out_l]; w2 [out_k, in_n] -> [in_n, out_k]
        //        w2_a [out_k, dim] -> [dim, out_k];  w2_b [dim, in_n] -> [in_n, dim]
        const int64_t ne0 = e.shape[1];
        const int64_t ne1 = e.shape[0];
        ggml_tensor * t   = ggml_new_tensor_2d(ad->ctx, GGML_TYPE_F16, ne0, ne1);
        ggml_set_name(t, e.name.c_str());
        MM3LmAdapterPair & p    = ad->mods[layer][module];
        ggml_tensor **     slot = mm3_lm_pair_slot(p, c);
        if (slot) {
            *slot = t;
        }
        if (c == MM3_LM_COMP_A) {
            ad->rank = (int) ne1;
        }
    }

    ad->buf = ggml_backend_alloc_ctx_tensors(ad->ctx, backend);
    if (!ad->buf) {
        st_close(&st);
        if (err) {
            *err = "adapter VRAM allocation failed";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }

    // Pass 3: upload, converting any dtype to f16 via f32.
    std::vector<float>      f32;
    std::vector<ggml_fp16_t> f16;
    for (const STEntry & e : st.entries) {
        int                    layer, module;
        const MM3LmAdapterComp c = mm3_lm_adapter_parse_any(e.name, &layer, &module);
        if (c == MM3_LM_COMP_NONE) {
            continue;
        }
        MM3LmAdapterPair & p = ad->mods[layer][module];
        if (c == MM3_LM_COMP_ALPHA) {
            // A scalar (or 1-element tensor). Stored, not uploaded.
            float av = 0.0f;
            if (adapter_to_f32(st_data(st, e), &av, 1, e.dtype)) {
                p.alpha_raw = av;
                p.has_alpha = true;
            }
            continue;
        }
        ggml_tensor ** slot = mm3_lm_pair_slot(p, c);
        ggml_tensor *  t    = slot ? *slot : nullptr;
        if (!t) {
            continue;
        }
        const int64_t n = ggml_nelements(t);
        f32.resize((size_t) n);
        if (!adapter_to_f32(st_data(st, e), f32.data(), n, e.dtype)) {
            st_close(&st);
            if (err) {
                *err = "unsupported dtype " + e.dtype + " on " + e.name;
            }
            mm3_lm_adapter_free(ad);
            return nullptr;
        }
        f16.resize((size_t) n);
        ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
        ggml_backend_tensor_set(t, f16.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
    }
    st_close(&st);

    // Pass 4: validate pairing + count. Every module with an A must have a B.
    int n_lokr = 0;
    for (int l = 0; l < MM3_LM_ADAPTER_LAYERS; l++) {
        for (int m = 0; m < MM3_LM_ADAPTER_MODULES; m++) {
            MM3LmAdapterPair & p = ad->mods[l][m];
            if ((p.a == nullptr) != (p.b == nullptr)) {
                if (err) {
                    *err = "unpaired lora_A/lora_B at layer " + std::to_string(l);
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            // A LoKr module needs w1 and exactly one of {w2} or {w2_a, w2_b}.
            // Half a module is worse than none: it would silently apply a
            // different delta than the one that was trained.
            const bool any_lokr = p.w1 || p.w2 || p.w2_a || p.w2_b;
            if (any_lokr && !p.has_lokr()) {
                if (err) {
                    *err = "incomplete LoKr module at layer " + std::to_string(l) + " (need w1 plus w2 or w2_a/w2_b)";
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            if (p.has_lokr()) {
                // scale = alpha / dim. Monolithic w2 carries no dim, but
                // LyCORIS forces alpha == dim there, so the scale is exactly 1.
                // Factorized: w2_a is [dim, out_k] in ggml, so dim is ne[0].
                if (p.w2_a && p.has_alpha) {
                    const double dim = (double) p.w2_a->ne[0];
                    p.lokr_scale     = dim > 0.0 ? (float) (p.alpha_raw / dim) : 1.0f;
                } else {
                    p.lokr_scale = 1.0f;
                }
                // Geometry for the Kronecker apply, straight off the tensors.
                p.in_m  = p.w1->ne[0];
                p.out_l = p.w1->ne[1];
                p.in_n  = p.w2 ? p.w2->ne[0] : p.w2_b->ne[0];
                p.out_k = p.w2 ? p.w2->ne[1] : p.w2_a->ne[1];
                n_lokr++;
            }
            if (p.a || p.has_lokr()) {
                ad->n_loaded++;
            }
        }
    }
    ad->is_lokr = n_lokr > 0;
    if (ad->is_lokr && ad->rank != 0) {
        if (err) {
            *err = "adapter mixes LoRA and LoKr modules";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }
    fprintf(stderr, "[MM3] LM adapter loaded: %s (%d modules, %s, %.1f MB)\n", path, ad->n_loaded,
            ad->is_lokr ? "LoKr" : "LoRA", (double) ggml_backend_buffer_get_size(ad->buf) / 1e6);
    return ad;
}
