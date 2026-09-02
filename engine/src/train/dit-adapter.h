#pragma once
// dit-adapter.h — the parameterization seam (plan §3.6 / D22).
//
// `DitAdapter` owns ONLY site enumeration, apply(), params() and export. The
// graph (dit-train-graph.h), the loop (dit-train-run.h), the optimizer
// (lm-optim.h), the VRAM model and every artefact are parameterization-agnostic,
// so a LoKR implementation slots in by implementing four methods and adding one
// --adapter-type value. Its Kron-matvec gradient path, out_prod shapes and VRAM
// profile are entirely unmeasured — do NOT quote this plan's numbers as LoKR
// numbers.
//
// Shape/layout (verified end-to-end through the production merge path, D18):
//   ggml A [in, r]  == row-major [r, in]   == PEFT lora_A   -> STW shape {r, in}
//   ggml B [r, out] == row-major [out, r]  == PEFT lora_B   -> STW shape {out, r}
// adapter-runtime.h:116-141 checks in_feat == base->ne[0], out_feat == base->ne[1].
//
// alpha/r is applied IN-GRAPH; export writes RAW A and B and the engine
// re-applies alpha/r from adapter_config.json (D19). Never bake it into B.

#include "dit.h"
#include "train/dit-node-profile.h"
#include "train/lm-common.h"
#include "train/st-write.h"
#include "version.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

// ─── sites ──────────────────────────────────────────────────────────────────

enum DitSite {
    DIT_SA_Q = 0,
    DIT_SA_K,
    DIT_SA_V,
    DIT_SA_O,
    DIT_CA_Q,
    DIT_CA_K,
    DIT_CA_V,
    DIT_CA_O,
    DIT_MLP_GATE,
    DIT_MLP_UP,
    DIT_MLP_DOWN,
    DIT_NSITES
};

#define DIT_NSITES_ATTN 8

static const char * dit_site_peft(int s) {
    switch (s) {
        case DIT_SA_Q:     return "self_attn.q_proj";
        case DIT_SA_K:     return "self_attn.k_proj";
        case DIT_SA_V:     return "self_attn.v_proj";
        case DIT_SA_O:     return "self_attn.o_proj";
        case DIT_CA_Q:     return "cross_attn.q_proj";
        case DIT_CA_K:     return "cross_attn.k_proj";
        case DIT_CA_V:     return "cross_attn.v_proj";
        case DIT_CA_O:     return "cross_attn.o_proj";
        case DIT_MLP_GATE: return "mlp.gate_proj";
        case DIT_MLP_UP:   return "mlp.up_proj";
        default:           return "mlp.down_proj";
    }
}

// Site name used by the node profiler's construction scopes (dit-node-profile.h).
// Layer-independent on purpose: 32 layers collapse onto 11 rows.
static const char * dit_site_scope(int s) {
    static const char * const n[DIT_NSITES] = { "apply.sa_q",      "apply.sa_k",    "apply.sa_v",
                                                "apply.sa_o",      "apply.ca_q",    "apply.ca_k",
                                                "apply.ca_v",      "apply.ca_o",    "apply.mlp_gate",
                                                "apply.mlp_up",    "apply.mlp_down" };
    return (s >= 0 && s < DIT_NSITES) ? n[s] : "apply.unknown";
}

static ggml_tensor * dit_site_weight(DiTGGMLLayer * ly, int s) {
    switch (s) {
        case DIT_SA_Q:     return ly->sa_q_proj;
        case DIT_SA_K:     return ly->sa_k_proj;
        case DIT_SA_V:     return ly->sa_v_proj;
        case DIT_SA_O:     return ly->sa_o_proj;
        case DIT_CA_Q:     return ly->ca_q_proj;
        case DIT_CA_K:     return ly->ca_k_proj;
        case DIT_CA_V:     return ly->ca_v_proj;
        case DIT_CA_O:     return ly->ca_o_proj;
        case DIT_MLP_GATE: return ly->gate_proj;
        case DIT_MLP_UP:   return ly->up_proj;
        default:           return ly->down_proj;
    }
}

// ─── config + export metadata ───────────────────────────────────────────────

struct DitAdapterCfg {
    int      rank       = 16;
    float    alpha      = 32.0f;
    bool     target_mlp = false;
    uint64_t seed       = 42;
    float    b_sigma    = 0.0f;  // >0 only for the finite-difference self-test
    // LoKR (dit-adapter-lokr.h); ignored by the LoRA implementation.
    int   lokr_dim            = 512;
    float lokr_alpha          = 512.0f;
    int   lokr_factor         = 6;
    bool  lokr_decompose_both = true;
};

struct DitExportMeta {
    std::string base_model_path;  // adapter_config.json base_model_name_or_path
    std::string producer;         // "ace-train <version>"
    // Trigger word embedded in __metadata__ (empty = none).
    // docs/plans/2026-07-28-adapter-trigger-embedding.md §2.1
    std::string trigger, trigger_position;
};

struct DitExportResult {
    int       tensors = 0;
    long long bytes   = 0;
};

// ─── the seam ───────────────────────────────────────────────────────────────

struct DitAdapter {
    virtual ~DitAdapter() = default;
    virtual bool init(DiTGGML * m, ggml_backend_t backend, int lo, int hi, const DitAdapterCfg & cfg,
                      std::string * err)                                                            = 0;
    virtual const std::vector<ggml_tensor *> & params() const                                       = 0;
    /** In-graph: y = W*x + delta(x). Returns W*x unchanged when the site is untrained. */
    virtual ggml_tensor * apply(ggml_context * ctx, ggml_tensor * w, int layer, int site,
                                ggml_tensor * x) const                                              = 0;
    /** apply(), wrapped in the node profiler's per-site construction scope. The
     *  graph builder calls THIS one so every adapter/base-projection node can be
     *  attributed without shape guessing; with DIT_PROFILE_NODES unset the scope
     *  is one null-pointer test in and one out, and the emitted graph is
     *  identical (a scope tags tensors, it never creates any). */
    inline ggml_tensor *  applyP(ggml_context * ctx, ggml_tensor * w, int layer, int site, ggml_tensor * x) const {
        DnpScope _dnp(ctx, dit_site_scope(site));
        return apply(ctx, w, layer, site, x);
    }
    virtual bool          exportDir(const char * dir, const DitExportMeta & meta, DitExportResult * res,
                                    std::string * err) const                                        = 0;
    virtual size_t        nParams() const                                                           = 0;
    virtual int           nSites() const                                                            = 0;
    virtual const char *  typeName() const                                                          = 0;
    virtual void          free()                                                                    = 0;
};

// ─── LoRA ───────────────────────────────────────────────────────────────────

struct DitLoraPair {
    ggml_tensor * A = nullptr;  // [in, r]
    ggml_tensor * B = nullptr;  // [r, out]
};

struct DitAdapterLora final : DitAdapter {
    ggml_context *                                ctx = nullptr;
    ggml_backend_buffer_t                         buf = nullptr;
    std::vector<std::array<DitLoraPair, DIT_NSITES>> layers;  // indexed by (layer - lo)
    std::vector<ggml_tensor *>                    par;
    int    lo = 0, hi = 0, rank = 0, n_sites = DIT_NSITES_ATTN;
    float  alpha = 0.0f, scale = 1.0f;
    size_t n_params = 0;

    const std::vector<ggml_tensor *> & params() const override { return par; }
    size_t                             nParams() const override { return n_params; }
    int                                nSites() const override { return n_sites; }
    const char *                       typeName() const override { return "lora"; }

    void free() override {
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        layers.clear();
        par.clear();
        n_params = 0;
    }

    const DitLoraPair * pair(int layer, int site) const {
        if (layer < lo || layer >= hi || site < 0 || site >= n_sites) {
            return nullptr;
        }
        const DitLoraPair & p = layers[(size_t) (layer - lo)][(size_t) site];
        return (p.A && p.B) ? &p : nullptr;
    }

    bool init(DiTGGML * m, ggml_backend_t backend, int lo_, int hi_, const DitAdapterCfg & cfg,
              std::string * err) override {
        lo      = lo_;
        hi      = hi_;
        rank    = cfg.rank;
        alpha   = cfg.alpha;
        scale   = cfg.alpha / (float) cfg.rank;
        n_sites = cfg.target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
        layers.assign((size_t) (hi - lo), std::array<DitLoraPair, DIT_NSITES>{});

        const int n_ten = (hi - lo) * n_sites * 2;
        {
            ggml_init_params p = { (size_t) (n_ten + 8) * ggml_tensor_overhead(), nullptr, true };
            ctx                = ggml_init(p);
        }
        if (!ctx) {
            *err = "cannot create the LoRA context";
            return false;
        }

        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                ggml_tensor * w = dit_site_weight(&m->layers[l], s);
                if (!w) {
                    char b[128];
                    snprintf(b, sizeof(b), "layer %d site %s has no weight", l, dit_site_peft(s));
                    *err = b;
                    return false;
                }
                DitLoraPair & pr = layers[(size_t) (l - lo)][(size_t) s];
                pr.A             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], rank);
                pr.B             = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, w->ne[1]);
                char nm[112];
                snprintf(nm, sizeof(nm), "L%d.%s.lora_A", l, dit_site_peft(s));
                ggml_set_name(pr.A, nm);
                snprintf(nm, sizeof(nm), "L%d.%s.lora_B", l, dit_site_peft(s));
                ggml_set_name(pr.B, nm);
                ggml_set_param(pr.A);
                ggml_set_param(pr.B);
                par.push_back(pr.A);
                par.push_back(pr.B);

                // Exportability is CHECKED, not assumed (§3.6). Round 1's LoRA sat
                // on the fused sa_qkv, which is not expressible as a PEFT q/k/v
                // adapter at all — this is the check that would have caught it.
                if (pr.A->ne[0] != w->ne[0] || pr.B->ne[1] != w->ne[1] || pr.A->ne[1] != rank ||
                    pr.B->ne[0] != rank) {
                    char b[224];
                    snprintf(b, sizeof(b),
                             "layer %d site %s is not PEFT-shaped: base[%lld,%lld] A[%lld,%lld] B[%lld,%lld] r=%d", l,
                             dit_site_peft(s), (long long) w->ne[0], (long long) w->ne[1], (long long) pr.A->ne[0],
                             (long long) pr.A->ne[1], (long long) pr.B->ne[0], (long long) pr.B->ne[1], rank);
                    *err = b;
                    return false;
                }
            }
        }

        // LoRA tensors live in their OWN backend buffer, never the mirror's:
        // ggml_opt_step_adamw writes into them and the mirror is USAGE_WEIGHTS.
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            *err = "LoRA parameter buffer allocation failed";
            return false;
        }

        LmRng rng;
        lm_rng_seed(&rng, cfg.seed);
        std::vector<float> a, b;
        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                DitLoraPair & pr = layers[(size_t) (l - lo)][(size_t) s];
                a.assign((size_t) ggml_nelements(pr.A), 0.0f);
                b.assign((size_t) ggml_nelements(pr.B), 0.0f);
                lm_rng_fill_normal(&rng, a, 1.0f / sqrtf((float) pr.A->ne[0]));  // A ~ N(0, 1/sqrt(in))
                if (cfg.b_sigma > 0.0f) {
                    lm_rng_fill_normal(&rng, b, cfg.b_sigma);
                }
                ggml_backend_tensor_set(pr.A, a.data(), 0, a.size() * sizeof(float));
                ggml_backend_tensor_set(pr.B, b.data(), 0, b.size() * sizeof(float));  // true PEFT init: B = 0
                n_params += a.size() + b.size();
            }
        }
        return true;
    }

    ggml_tensor * apply(ggml_context * ctx_g, ggml_tensor * w, int layer, int site, ggml_tensor * x) const override {
        ggml_tensor *       y = ggml_mul_mat(ctx_g, w, x);
        const DitLoraPair * p = pair(layer, site);
        if (p) {
            ggml_tensor * t = ggml_scale(ctx_g, ggml_mul_mat(ctx_g, p->A, x), scale);  // alpha/rank in-graph
            y               = ggml_add(ctx_g, y, ggml_mul_mat(ctx_g, p->B, t));
        }
        return y;
    }

    // §2.3: layer-major, site order q,k,v,o (self), q,k,v,o (cross), gate,up,down;
    // lora_A before lora_B. Deterministic byte layout.
    bool exportDir(const char * dir, const DitExportMeta & meta, DitExportResult * res,
                   std::string * err) const override {
        const std::string d(dir);
        if (!pm_mkdir_p(d)) {
            *err = "cannot create " + d;
            return false;
        }
        // adapter_config.json is written FIRST (D19): a missing/unparseable file
        // silently degrades the scale to 1.0 with only a warning.
        if (!dit_write_adapter_config(d, rank, (int) (alpha + 0.5f), n_sites == DIT_NSITES, meta.base_model_path)) {
            *err = "cannot write adapter_config.json in " + d;
            return false;
        }

        std::vector<STWTensor>          tensors;
        std::vector<std::vector<float>> store;
        tensors.reserve((size_t) (hi - lo) * (size_t) n_sites * 2);
        store.reserve(tensors.capacity());

        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                const DitLoraPair & pr = layers[(size_t) (l - lo)][(size_t) s];
                ggml_tensor *       both[2] = { pr.A, pr.B };
                const char *        suffix[2] = { "lora_A", "lora_B" };
                for (int k = 0; k < 2; k++) {
                    ggml_tensor * t = both[k];
                    store.push_back(std::vector<float>((size_t) ggml_nelements(t)));
                    std::vector<float> & bufv = store.back();
                    ggml_backend_tensor_get(t, bufv.data(), 0, bufv.size() * sizeof(float));
                    char nm[208];
                    snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.%s.weight", l, dit_site_peft(s),
                             suffix[k]);
                    STWTensor st;
                    st.name  = nm;
                    st.shape = { t->ne[1], t->ne[0] };  // row-major (rows, cols)
                    st.data  = bufv.data();
                    tensors.push_back(st);
                }
            }
        }
        // `store` reallocates as it grows; re-point once it has stopped moving.
        GGML_ASSERT(store.size() == tensors.size());
        for (size_t i = 0; i < tensors.size(); i++) {
            tensors[i].data = store[i].data();
        }

        std::vector<std::pair<std::string, std::string>> md;
        md.push_back({ "format", "pt" });
        md.push_back({ "producer", meta.producer });
        md.push_back({ "hot_step_dit_trainer", "v1" });
        // Trigger word (all three keys together or none — §2.1). Unknown metadata
        // keys are ignored by every other safetensors consumer, so the adapter
        // stays loadable by ComfyUI / Side-Step / PEFT exactly as before.
        if (!meta.trigger.empty()) {
            md.push_back({ "hot_step_trigger", meta.trigger });
            md.push_back({ "hot_step_trigger_position",
                           meta.trigger_position.empty() ? std::string("prepend") : meta.trigger_position });
            md.push_back({ "modelspec.trigger_phrase", meta.trigger });
        }

        const std::string sf = lm_join(d, "adapter_model.safetensors");
        // BF16 for the same reason as the LoKR writer (dit-adapter-lokr.h): a
        // DiT LoRA r128 is 1.2 GB in F32 and every consumer of a PEFT adapter
        // takes BF16. The LM trainer's PEFT export stays F32 — Side-Step's LM
        // adapters are F32, and matching them keeps the sizes comparable.
        if (!st_write_file(sf.c_str(), tensors, md, STW_BF16)) {
            *err = "cannot write " + sf;
            return false;
        }
        if (res) {
            long long bytes = 0;
            pm_stat_file(sf, &bytes, NULL);
            res->tensors = (int) tensors.size();
            res->bytes   = bytes;
        }
        return true;
    }

    // ─── adapter_config.json (§2.3 literal) ─────────────────────────────────
    static bool dit_write_adapter_config(const std::string & dir, int r, int lora_alpha, bool target_mlp,
                                         const std::string & base_model) {
        std::string j;
        char        b[96];
        j += "{\n";
        j += "  \"peft_type\": \"LORA\",\n";
        j += "  \"task_type\": null,\n";
        j += "  \"auto_mapping\": null,\n";
        j += "  \"base_model_name_or_path\": \"" + lm_json_escape(base_model) + "\",\n";
        snprintf(b, sizeof(b), "  \"r\": %d,\n", r);
        j += b;
        snprintf(b, sizeof(b), "  \"lora_alpha\": %d,\n", lora_alpha);
        j += b;
        j += "  \"lora_dropout\": 0.0,\n";
        j += "  \"bias\": \"none\",\n";
        j += "  \"fan_in_fan_out\": false,\n";
        j += "  \"inference_mode\": true,\n";
        j += "  \"init_lora_weights\": true,\n";
        j += "  \"use_dora\": false,\n";
        j += "  \"use_rslora\": false,\n";
        j += "  \"rank_pattern\": {},\n";
        j += "  \"alpha_pattern\": {},\n";
        j += "  \"modules_to_save\": null,\n";
        j += "  \"layers_to_transform\": null,\n";
        j += "  \"layers_pattern\": null,\n";
        j += "  \"layer_replication\": null,\n";
        j += "  \"loftq_config\": {},\n";
        j += "  \"megatron_config\": null,\n";
        j += "  \"megatron_core\": \"megatron.core\",\n";
        j += "  \"revision\": null,\n";
        j += "  \"target_modules\": [\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"";
        if (target_mlp) {
            j += ",\"gate_proj\",\"up_proj\",\"down_proj\"";
        }
        j += "]\n}\n";
        return pm_write_atomic(lm_join(dir, "adapter_config.json"), j);
    }
};

// Expected parameter count for the assertion in dit-train-run.h (§3.6 table).
static size_t dit_lora_expected_params(const DiTGGMLConfig & c, int lo, int hi, int rank, bool target_mlp) {
    const int64_t H = c.hidden_size, Q = (int64_t) c.n_heads * c.head_dim, KV = (int64_t) c.n_kv_heads * c.head_dim,
                  F = c.intermediate_size;
    // per site: in*r + r*out
    int64_t per_layer = 0;
    per_layer += H * rank + (int64_t) rank * Q;   // sa q
    per_layer += H * rank + (int64_t) rank * KV;  // sa k
    per_layer += H * rank + (int64_t) rank * KV;  // sa v
    per_layer += Q * rank + (int64_t) rank * H;   // sa o
    per_layer += H * rank + (int64_t) rank * Q;   // ca q
    per_layer += H * rank + (int64_t) rank * KV;  // ca k
    per_layer += H * rank + (int64_t) rank * KV;  // ca v
    per_layer += Q * rank + (int64_t) rank * H;   // ca o
    if (target_mlp) {
        per_layer += H * rank + (int64_t) rank * F;  // gate
        per_layer += H * rank + (int64_t) rank * F;  // up
        per_layer += F * rank + (int64_t) rank * H;  // down
    }
    return (size_t) (per_layer * (int64_t) (hi - lo));
}
