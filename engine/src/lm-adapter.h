#pragma once
// lm-adapter.h: runtime LoRA loading for the 5Hz planner LM.
//
// Loads a PEFT LoRA adapter (directory with adapter_model.safetensors +
// adapter_config.json, or a bare .safetensors file) and stages the A/B
// pairs on the LM's backend.  Applied at graph-build time via
// qwen3_linear_lora() — never merged, so the base LM can be any quant
// (Q8_0/Q5_K/NVFP4/...) with zero requantization loss.  Per-token cost at
// rank 16 is negligible (two rank-r matmuls per adapted projection).
//
// Trained by Side-Step's `sidestep lm-train` (sidestep_engine/lm/), which
// replicates the engine's exact prompt format (engine/src/prompt.h).
//
// PEFT tensor naming handled:
//   base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
//   (any prefix before "model.layers." is ignored)
// Supported slots: q/k/v/o_proj, gate/up/down_proj.  Anything else
// (embed_tokens, lm_head) is skipped with a warning.
//
// Local HOT-Step feature — not upstream acestep.cpp.

#include "artist-token-runtime.h"  // the token half of a unified adapter file
#include "qwen3-lora.h"
#include "safetensors.h"
#include "yyjson.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct LMLora {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    QwLoraLayer           layers[QWEN3_LORA_MAX_LAYERS];
    int                   n_tensors = 0;
    int                   max_layer = -1;
    std::string           path;
    float                 user_scale = 1.0f;

    // ── Soft-prompt halves that travel in the same file (train/lm-export.h) ──
    //
    // The artist token is installed into the process-wide runtime
    // (artist-token-runtime.h) at load and cleared at free; this flag says this
    // adapter is the one that owns it, so freeing a DIFFERENT adapter cannot
    // evict a token that came from elsewhere.
    bool owns_artist_token = false;

    // Per-layer K/V prefix, kept on the HOST as F32 and written into a KV set
    // by qw3lm_seed_prefix at Phase-2 reset. Indexed by absolute layer; empty
    // vectors outside [pfx_lo, pfx_hi). Row layout [Nkv*D] per column, n columns.
    std::vector<std::vector<float>> pfx_k, pfx_v;
    int                             pfx_n  = 0;
    int                             pfx_lo = 0, pfx_hi = 0;
    int64_t                         pfx_row = 0;  // Nkv * D, as written
};

// Map a module substring to a slot. Order matters: check longer names first.
static bool lm_adapter_slot_for(const std::string & name, QwLoraSlot * out) {
    struct { const char * pat; QwLoraSlot slot; } map[] = {
        { ".self_attn.q_proj.", QW_LORA_Q },
        { ".self_attn.k_proj.", QW_LORA_K },
        { ".self_attn.v_proj.", QW_LORA_V },
        { ".self_attn.o_proj.", QW_LORA_O },
        { ".mlp.gate_proj.", QW_LORA_GATE },
        { ".mlp.up_proj.", QW_LORA_UP },
        { ".mlp.down_proj.", QW_LORA_DOWN },
    };
    for (auto & m : map) {
        if (name.find(m.pat) != std::string::npos) {
            *out = m.slot;
            return true;
        }
    }
    return false;
}

// Parse "…model.layers.<N>…" → layer index, or -1.
static int lm_adapter_layer_for(const std::string & name) {
    size_t p = name.find("model.layers.");
    if (p == std::string::npos) return -1;
    p += strlen("model.layers.");
    if (p >= name.size() || name[p] < '0' || name[p] > '9') return -1;
    return atoi(name.c_str() + p);
}

// Read lora_alpha and r from adapter_config.json. Returns alpha/r ratio
// factor, or -1 when the config is missing/unparseable (caller falls back
// to per-tensor alpha=r, i.e. factor 1.0, with a warning).
static float lm_adapter_read_alpha_ratio(const std::string & dir) {
    std::string  cfg_path = dir + "/adapter_config.json";
    yyjson_doc * doc      = yyjson_read_file(cfg_path.c_str(), 0, NULL, NULL);
    if (!doc) return -1.0f;
    yyjson_val * root  = yyjson_doc_get_root(doc);
    double       alpha = 0.0, r = 0.0;
    bool         rslora = false;
    if (root && yyjson_is_obj(root)) {
        yyjson_val * a = yyjson_obj_get(root, "lora_alpha");
        yyjson_val * rv = yyjson_obj_get(root, "r");
        yyjson_val * rs = yyjson_obj_get(root, "use_rslora");
        if (a && yyjson_is_num(a)) alpha = yyjson_get_num(a);
        if (rv && yyjson_is_num(rv)) r = yyjson_get_num(rv);
        if (rs && yyjson_is_true(rs)) rslora = true;
    }
    yyjson_doc_free(doc);
    // rsLoRA (use_rslora): the trainer applied alpha/sqrt(r) in-graph, so the
    // runtime must too, or the adapter comes in at sqrt(r) times the wrong
    // strength — 11x at r128 — with nothing in the log to say so.
    if (alpha > 0.0 && r > 0.0) return (float) (rslora ? alpha / sqrt(r) : alpha / r);
    return -1.0f;
}

static void lm_adapter_free(LMLora * l) {
    if (!l) return;
    if (l->owns_artist_token) {
        artist_token_clear();  // never leave a stale token behind an adapter switch
    }
    if (l->buf) ggml_backend_buffer_free(l->buf);
    if (l->ctx) ggml_free(l->ctx);
    delete l;
}

// Load a PEFT LoRA for the LM and stage its tensors on `backend`.
// ─── LoKr support (2026-07-30) ──────────────────────────────────────────────
//
// Reads the LyCORIS layout the LM trainer writes (train/lm-export.h), which is
// byte-layout identical to the DiT's:
//
//   lycoris_layers_<L>_<site>.alpha        [1]
//   lycoris_layers_<L>_<site>.lokr_w1      [out_l, in_m]
//   lycoris_layers_<L>_<site>.lokr_w2      [out_k, in_n]   monolithic
//   lycoris_layers_<L>_<site>.lokr_w2_a    [out_k, dim]    factorized
//   lycoris_layers_<L>_<site>.lokr_w2_b    [dim,   in_n]
//
// NO METADATA IS REQUIRED, deliberately: safetensors.h skips __metadata__, and
// everything needed is derivable from the shapes plus one LyCORIS rule —
// alpha is FORCED to dim when both factors are monolithic, so the scale is
// exactly 1 there, and a factorized site gets dim from w2_a's own ne0.
static bool lm_adapter_lokr_slot(const std::string & site, QwLoraSlot * out) {
    if (site == "self_attn_q_proj") { *out = QW_LORA_Q;    return true; }
    if (site == "self_attn_k_proj") { *out = QW_LORA_K;    return true; }
    if (site == "self_attn_v_proj") { *out = QW_LORA_V;    return true; }
    if (site == "self_attn_o_proj") { *out = QW_LORA_O;    return true; }
    if (site == "mlp_gate_proj")    { *out = QW_LORA_GATE; return true; }
    if (site == "mlp_up_proj")      { *out = QW_LORA_UP;   return true; }
    if (site == "mlp_down_proj")    { *out = QW_LORA_DOWN; return true; }
    return false;
}

// "lycoris_layers_12_self_attn_q_proj.lokr_w2" -> layer 12, slot Q, "lokr_w2".
// Returns false for anything that is not a LoKr key at all. `site_out` is
// filled even when the slot is unknown, so the caller can tell a DiT adapter
// (which carries cross_attn sites) from a corrupt one.
static bool lm_adapter_lokr_parse(const std::string & name, int * layer_out, QwLoraSlot * slot_out,
                                  std::string * suffix_out, std::string * site_out) {
    static const std::string PFX = "lycoris_layers_";
    if (name.compare(0, PFX.size(), PFX) != 0) {
        return false;
    }
    size_t i = PFX.size();
    int    layer = 0;
    if (i >= name.size() || !isdigit((unsigned char) name[i])) {
        return false;
    }
    while (i < name.size() && isdigit((unsigned char) name[i])) {
        layer = layer * 10 + (name[i] - '0');
        i++;
    }
    if (i >= name.size() || name[i] != '_') {
        return false;
    }
    const size_t dot = name.find('.', i);
    if (dot == std::string::npos) {
        return false;
    }
    *site_out   = name.substr(i + 1, dot - i - 1);
    *suffix_out = name.substr(dot + 1);
    *layer_out  = layer;
    return lm_adapter_lokr_slot(*site_out, slot_out);
}

// Read a [1] alpha tensor, whatever dtype it was written in.
static float lm_adapter_read_alpha(const STFile & st, const STEntry & e) {
    const void * d = st_data(st, e);
    if (e.dtype == "F32") {
        float v = 0.0f;
        memcpy(&v, d, sizeof(float));
        return v;
    }
    if (e.dtype == "BF16") {
        uint16_t b = 0;
        memcpy(&b, d, sizeof(uint16_t));
        const uint32_t u = (uint32_t) b << 16;
        float          v = 0.0f;
        memcpy(&v, &u, sizeof(float));
        return v;
    }
    return 0.0f;
}

// Returns nullptr on failure (caller must treat this as a load failure —
// never fall back silently to the base LM under an adapter-bearing key).
// ─── The soft-prompt half of a unified adapter file ──────────────────────────
//
// train/lm-export.h writes hot_step.artist_token.{vec,meta} and
// hot_step.prefix.{L<l>.k,L<l>.v,meta} into the same adapter_model.safetensors
// as the LoRA pairs. Read them here, while `st` is still open, so a token or a
// prefix loads with the adapter it was trained with and never has to be found
// separately. Absent entries are simply absent — an adapter written before this
// existed loads exactly as it did.
//
// site: 1 = as15_lm (this runtime), 2 = mm3_lm. An mm3 token loaded into the
// ACE LM would add a vector that means nothing; refuse it by name.
static void lm_adapter_read_soft_prompt(const STFile & st, LMLora * l) {
    const STEntry * vec  = st_find(st, "hot_step.artist_token.vec");
    const STEntry * meta = st_find(st, "hot_step.artist_token.meta");
    if (vec && meta && meta->n_dims == 1 && meta->shape[0] == 4 && meta->dtype == "F32" && vec->dtype == "F32" &&
        vec->n_dims == 2) {
        const float * m = (const float *) st_data(st, *meta);
        const int k = (int) m[0], placeholder = (int) m[1], hidden = (int) m[2], site = (int) m[3];
        if (site != 1) {
            fprintf(stderr, "[LM-Adapter] artist token in %s is site %d, not as15_lm — ignored\n", l->path.c_str(),
                    site);
        } else if (vec->shape[0] != k || vec->shape[1] != hidden || k < 1 || placeholder < 0) {
            fprintf(stderr, "[LM-Adapter] artist token in %s has inconsistent shape/meta — ignored\n",
                    l->path.c_str());
        } else {
            std::string nm = l->path;
            size_t      sl = nm.find_last_of("/\\");
            if (sl != std::string::npos) {
                nm = nm.substr(sl + 1);
            }
            artist_token_set((const float *) st_data(st, *vec), k, placeholder, hidden, "as15_lm", nm.c_str());
            l->owns_artist_token = true;
        }
    }

    const STEntry * pm = st_find(st, "hot_step.prefix.meta");
    if (pm && pm->n_dims == 1 && pm->shape[0] == 4 && pm->dtype == "F32") {
        const float * m  = (const float *) st_data(st, *pm);
        const int     n  = (int) m[0];
        const int64_t row = (int64_t) m[1];
        const int     lo = (int) m[2], hi = (int) m[3];
        if (n > 0 && row > 0 && hi > lo && lo >= 0 && hi <= QWEN3_LORA_MAX_LAYERS) {
            l->pfx_k.assign((size_t) hi, {});
            l->pfx_v.assign((size_t) hi, {});
            int got = 0;
            for (int ly = lo; ly < hi; ly++) {
                char nk[64], nv[64];
                snprintf(nk, sizeof(nk), "hot_step.prefix.L%d.k", ly);
                snprintf(nv, sizeof(nv), "hot_step.prefix.L%d.v", ly);
                const STEntry * ek = st_find(st, nk);
                const STEntry * ev = st_find(st, nv);
                if (!ek || !ev || ek->dtype != "F32" || ev->dtype != "F32" || ek->n_dims != 2 ||
                    ek->shape[0] != n || ek->shape[1] != row || ev->shape[0] != n || ev->shape[1] != row) {
                    fprintf(stderr, "[LM-Adapter] prefix layer %d missing or malformed in %s — prefix ignored\n", ly,
                            l->path.c_str());
                    got = -1;
                    break;
                }
                const float * pk = (const float *) st_data(st, *ek);
                const float * pv = (const float *) st_data(st, *ev);
                l->pfx_k[(size_t) ly].assign(pk, pk + (size_t) n * (size_t) row);
                l->pfx_v[(size_t) ly].assign(pv, pv + (size_t) n * (size_t) row);
                got++;
            }
            if (got > 0) {
                l->pfx_n   = n;
                l->pfx_lo  = lo;
                l->pfx_hi  = hi;
                l->pfx_row = row;
                fprintf(stderr, "[LM-Adapter] prefix installed: n=%d over layers [%d,%d), row %lld\n", n, lo, hi,
                        (long long) row);
            } else {
                l->pfx_k.clear();
                l->pfx_v.clear();
            }
        }
    }
}

static LMLora * lm_adapter_load(const char * path, float user_scale, ggml_backend_t backend) {
    std::string p = path;
    std::string dir;
    std::string sf_path;
    bool        is_file = p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0;
    if (is_file) {
        sf_path = p;
        size_t slash = p.find_last_of("/\\");
        dir = (slash == std::string::npos) ? "." : p.substr(0, slash);
    } else {
        dir = p;
        sf_path = p + "/adapter_model.safetensors";
        // A LoKr adapter dir has no PEFT file at all — the weights live in
        // lokr_weights.safetensors and there is deliberately no
        // adapter_config.json (alpha rides the per-module tensors instead).
        FILE * probe = fopen(sf_path.c_str(), "rb");
        if (probe) {
            fclose(probe);
        } else {
            sf_path = p + "/lokr_weights.safetensors";
        }
    }

    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) {
        fprintf(stderr, "[LM-Adapter] FATAL: cannot open %s\n", sf_path.c_str());
        return nullptr;
    }

    float alpha_ratio = lm_adapter_read_alpha_ratio(dir);
    bool  ratio_from_cfg = alpha_ratio > 0.0f;

    LMLora * l = new LMLora();
    l->path = p;
    l->user_scale = user_scale;

    // ── LoKr? ────────────────────────────────────────────────────────────
    int  skipped = 0;
    bool has_lokr = false, has_cross_attn = false;
    for (const STEntry & e : st.entries) {
        if (e.name.find(".lokr_w1") != std::string::npos) {
            has_lokr = true;
        }
        if (e.name.find("cross_attn") != std::string::npos) {
            has_cross_attn = true;
        }
    }
    if (has_lokr) {
        // A DiT LoKr uses the SAME lycoris_layers_<L>_<site> stems and differs
        // only in its site set, so without this a DiT adapter would load into
        // the LM's slots and quietly compute nonsense. cross_attn exists only on
        // the DiT.
        if (has_cross_attn) {
            fprintf(stderr, "[LM-Adapter] FATAL: %s carries cross_attn sites - that is a DiT LoKr, not an LM one\n",
                    sf_path.c_str());
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        struct LkPending { const STEntry * e; ggml_tensor ** dst; };
        std::vector<LkPending> lkp;
        std::vector<float>     alphas((size_t) QWEN3_LORA_MAX_LAYERS * QW_LORA_NSLOTS, -1.0f);
        int                    n_ten = 0;
        for (const STEntry & e : st.entries) {
            int         layer = -1;
            QwLoraSlot  slot;
            std::string sfx, site;
            if (lm_adapter_lokr_parse(e.name, &layer, &slot, &sfx, &site) && layer >= 0
                && layer < QWEN3_LORA_MAX_LAYERS && sfx != "alpha") {
                n_ten++;
            }
        }
        struct ggml_init_params gp2 = { ggml_tensor_overhead() * (size_t) (n_ten + 8), NULL, true };
        l->ctx                      = ggml_init(gp2);

        for (const STEntry & e : st.entries) {
            int         layer = -1;
            QwLoraSlot  slot;
            std::string sfx, site;
            if (!lm_adapter_lokr_parse(e.name, &layer, &slot, &sfx, &site)) {
                skipped++;
                continue;
            }
            if (layer < 0 || layer >= QWEN3_LORA_MAX_LAYERS) {
                skipped++;
                continue;
            }
            QwLoraPair & pr = l->layers[layer].p[slot];
            if (sfx == "alpha") {
                alphas[(size_t) layer * QW_LORA_NSLOTS + (size_t) slot] = lm_adapter_read_alpha(st, e);
                continue;
            }
            if (e.n_dims != 2) {
                fprintf(stderr, "[LM-Adapter] FATAL: %s has %d dims (want 2)\n", e.name.c_str(), e.n_dims);
                st_close(&st);
                lm_adapter_free(l);
                return nullptr;
            }
            const ggml_type ty = st_ggml_type(e);
            if (ty == GGML_TYPE_COUNT) {
                fprintf(stderr, "[LM-Adapter] FATAL: %s has unsupported dtype %s\n", e.name.c_str(),
                        e.dtype.c_str());
                st_close(&st);
                lm_adapter_free(l);
                return nullptr;
            }
            // torch [rows, cols] row-major -> ggml ne0=cols, ne1=rows
            ggml_tensor *  t   = ggml_new_tensor_2d(l->ctx, ty, e.shape[1], e.shape[0]);
            ggml_tensor ** dst = nullptr;
            if (sfx == "lokr_w1") {
                dst = &pr.w1;
            } else if (sfx == "lokr_w2") {
                dst = &pr.w2;
            } else if (sfx == "lokr_w2_a") {
                dst = &pr.w2_a;
            } else if (sfx == "lokr_w2_b") {
                dst = &pr.w2_b;
            }
            if (!dst) {
                skipped++;
                continue;
            }
            *dst = t;
            if (layer > l->max_layer) {
                l->max_layer = layer;
            }
            lkp.push_back({ &e, dst });
        }

        l->buf = ggml_backend_alloc_ctx_tensors(l->ctx, backend);
        if (!l->buf) {
            fprintf(stderr, "[LM-Adapter] FATAL: backend alloc failed for %d LoKr tensors\n", (int) lkp.size());
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        for (auto & pd : lkp) {
            ggml_backend_tensor_set(*pd.dst, st_data(st, *pd.e), 0, ggml_nbytes(*pd.dst));
        }
        l->n_tensors = (int) lkp.size();
        lm_adapter_read_soft_prompt(st, l);  // a LoKr adapter can carry the token half too
        st_close(&st);

        int sites = 0, mono = 0, fact = 0;
        for (int i = 0; i <= l->max_layer; i++) {
            for (int sl = 0; sl < QW_LORA_NSLOTS; sl++) {
                QwLoraPair & pr = l->layers[i].p[sl];
                if (!pr.w1) {
                    continue;
                }
                if (!pr.w2 && !(pr.w2_a && pr.w2_b)) {
                    fprintf(stderr, "[LM-Adapter] FATAL: layer %d slot %d has lokr_w1 but no usable w2\n", i, sl);
                    lm_adapter_free(l);
                    return nullptr;
                }
                // Factor dims come straight from the tensors: w1 is ggml
                // [in_m, out_l], w2 [in_n, out_k], w2_a [dim, out_k],
                // w2_b [in_n, dim].
                pr.in_m  = pr.w1->ne[0];
                pr.out_l = pr.w1->ne[1];
                if (pr.w2) {
                    pr.in_n  = pr.w2->ne[0];
                    pr.out_k = pr.w2->ne[1];
                    // LyCORIS forces alpha == dim when both factors are
                    // monolithic, so the scale is exactly 1 and no metadata is
                    // needed to recover it.
                    pr.lokr_scale = 1.0f * user_scale;
                    mono++;
                } else {
                    pr.in_n  = pr.w2_b->ne[0];
                    pr.out_k = pr.w2_a->ne[1];
                    const float dim = (float) pr.w2_a->ne[0];
                    const float a   = alphas[(size_t) i * QW_LORA_NSLOTS + (size_t) sl];
                    pr.lokr_scale   = ((a > 0.0f && dim > 0.0f) ? (a / dim) : 1.0f) * user_scale;
                    fact++;
                }
                sites++;
            }
        }
        if (sites == 0) {
            fprintf(stderr, "[LM-Adapter] FATAL: no usable LoKr sites in %s\n", sf_path.c_str());
            lm_adapter_free(l);
            return nullptr;
        }
        fprintf(stderr,
                "[LM-Adapter] Loaded %s: LoKr, %d sites across %d layers (%d monolithic / %d factorized), "
                "user scale=%.2f%s\n",
                p.c_str(), sites, l->max_layer + 1, mono, fact, user_scale,
                skipped ? " (some non-LoKr tensors skipped)" : "");
        return l;
    }

    // Pass 1: count usable tensors
    int usable = 0;
    for (const STEntry & e : st.entries) {
        QwLoraSlot slot;
        int        layer = lm_adapter_layer_for(e.name);
        bool is_ab = e.name.find(".lora_A.") != std::string::npos ||
                     e.name.find(".lora_B.") != std::string::npos;
        if (layer >= 0 && layer < QWEN3_LORA_MAX_LAYERS && is_ab && lm_adapter_slot_for(e.name, &slot)) {
            usable++;
        } else {
            skipped++;
        }
    }
    if (usable == 0) {
        fprintf(stderr, "[LM-Adapter] FATAL: no usable lora_A/lora_B tensors in %s (%d entries)\n",
                sf_path.c_str(), (int) st.entries.size());
        st_close(&st);
        lm_adapter_free(l);
        return nullptr;
    }

    struct ggml_init_params gp = { ggml_tensor_overhead() * (size_t) usable, NULL, true };
    l->ctx = ggml_init(gp);

    // Pass 2: create tensors
    struct Pending { const STEntry * e; struct ggml_tensor * t; };
    std::vector<Pending> pending;
    pending.reserve(usable);
    for (const STEntry & e : st.entries) {
        QwLoraSlot slot;
        int        layer = lm_adapter_layer_for(e.name);
        bool  is_a = e.name.find(".lora_A.") != std::string::npos;
        bool  is_b = e.name.find(".lora_B.") != std::string::npos;
        if (layer < 0 || layer >= QWEN3_LORA_MAX_LAYERS || (!is_a && !is_b) ||
            !lm_adapter_slot_for(e.name, &slot)) {
            continue;
        }
        if (e.n_dims != 2) {
            fprintf(stderr, "[LM-Adapter] FATAL: %s has %d dims (want 2)\n", e.name.c_str(), e.n_dims);
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        ggml_type ty = st_ggml_type(e);
        if (ty == GGML_TYPE_COUNT) {
            fprintf(stderr, "[LM-Adapter] FATAL: %s has unsupported dtype %s\n", e.name.c_str(), e.dtype.c_str());
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        // torch [rows, cols] row-major -> ggml ne0=cols, ne1=rows
        struct ggml_tensor * t = ggml_new_tensor_2d(l->ctx, ty, e.shape[1], e.shape[0]);
        QwLoraPair & pair = l->layers[layer].p[slot];
        if (is_a) pair.A = t; else pair.B = t;
        if (layer > l->max_layer) l->max_layer = layer;
        pending.push_back({ &e, t });
    }

    l->buf = ggml_backend_alloc_ctx_tensors(l->ctx, backend);
    if (!l->buf) {
        fprintf(stderr, "[LM-Adapter] FATAL: backend alloc failed for %d tensors\n", (int) pending.size());
        st_close(&st);
        lm_adapter_free(l);
        return nullptr;
    }
    for (auto & pd : pending) {
        ggml_backend_tensor_set(pd.t, st_data(st, *pd.e), 0, ggml_nbytes(pd.t));
    }
    l->n_tensors = (int) pending.size();
    lm_adapter_read_soft_prompt(st, l);  // token / prefix halves, while the file is still mapped
    st_close(&st);

    // Finalize pairs: both halves present, per-pair scale = ratio * user_scale
    // (ratio from adapter_config.json, else alpha=r fallback -> 1.0).
    if (!ratio_from_cfg) {
        fprintf(stderr, "[LM-Adapter] WARNING: no adapter_config.json next to %s — "
                        "assuming lora_alpha == r (scale factor 1.0)\n", sf_path.c_str());
        alpha_ratio = 1.0f;
    }
    int pairs = 0;
    for (int i = 0; i <= l->max_layer; i++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair & pr = l->layers[i].p[s];
            if ((pr.A == nullptr) != (pr.B == nullptr)) {
                fprintf(stderr, "[LM-Adapter] FATAL: layer %d slot %d has only one of lora_A/lora_B\n", i, s);
                lm_adapter_free(l);
                return nullptr;
            }
            if (pr.A) {
                pr.scale = alpha_ratio * user_scale;
                pairs++;
            }
        }
    }

    fprintf(stderr, "[LM-Adapter] Loaded %s: %d pairs across %d layers, alpha/r=%.3f, user scale=%.2f%s\n",
            p.c_str(), pairs, l->max_layer + 1, alpha_ratio, user_scale,
            skipped ? " (some non-projection tensors skipped)" : "");
    return l;
}
