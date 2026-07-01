#pragma once
// adapter-runtime.h: Runtime LoRA/LoKr adapter loading
//
// Instead of merging adapter deltas into base weights (slow for K-quants due
// to CPU re-quantization), this module precomputes the delta matrix on GPU
// and stores it as BF16 in VRAM. At inference time, dit-graph.h applies:
//   y = W@x + delta@x
// where W stays in native quant and delta is BF16.
//
// Supports both LoRA (A/B factorized) and LoKr (Kronecker product) adapters.
// Reuses the same safetensors parsing and delta graph construction from
// adapter-merge.h — only the merge step is replaced with delta storage.

#include "adapter-cancel.h"
#include "adapter-merge.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "timer.h"
#include "weight-source.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Cooperative cancel for the LoKr/LoRA delta precompute loop lives in
// adapter-cancel.h so ace-server.cpp can poke at it without pulling all of
// adapter-runtime.h's transitive ggml/safetensors deps. See that header.

// Per-projection runtime LoRA delta (stored as BF16 tensor in VRAM)
struct DiTLoRADelta {
    struct ggml_tensor * delta = nullptr;  // [in, out] BF16, or NULL
};

// Per-layer LoRA deltas for all adapted projections
struct DiTLoRALayer {
    DiTLoRADelta sa_q, sa_k, sa_v, sa_o;     // self-attention
    DiTLoRADelta ca_q, ca_k, ca_v, ca_o;     // cross-attention
    DiTLoRADelta gate, up, down;              // MLP
};

#define DIT_LORA_MAX_LAYERS 32

// Staged delta: pairs a tensor pointer with its F32 data for upload after buffer allocation
struct DiTLoRAStagedDelta {
    struct ggml_tensor * tensor;
    std::vector<float>   f32_data;
};

// Runtime LoRA storage: holds all precomputed delta tensors
struct DiTLoRA {
    bool                            active = false;
    DiTLoRALayer                    layers[DIT_LORA_MAX_LAYERS];
    DiTLoRADelta                    proj_in;
    DiTLoRADelta                    cond_emb;
    DiTLoRADelta                    time_embed_linear_1;
    DiTLoRADelta                    time_embed_linear_2;
    DiTLoRADelta                    time_embed_time_proj;
    DiTLoRADelta                    time_embed_r_linear_1;
    DiTLoRADelta                    time_embed_r_linear_2;
    DiTLoRADelta                    time_embed_r_time_proj;

    struct ggml_context *           ctx    = nullptr;  // owns the delta tensors
    ggml_backend_buffer_t           buffer = nullptr;  // single buffer for all deltas
    std::vector<DiTLoRAStagedDelta> staged;            // temp F32 data awaiting BF16 upload
};

static void dit_lora_free(DiTLoRA * lora) {
    if (lora->buffer) {
        ggml_backend_buffer_free(lora->buffer);
    }
    if (lora->ctx) {
        ggml_free(lora->ctx);
    }
    *lora = {};
}

// Map a GGUF tensor name to the corresponding DiTLoRADelta slot.
// Returns NULL if the name doesn't correspond to an adapted projection.
static DiTLoRADelta * dit_lora_slot(DiTLoRA * lora, const std::string & gguf_name) {
    if (gguf_name == "decoder.proj_in.1.weight") return &lora->proj_in;
    if (gguf_name == "decoder.condition_embedder.weight") return &lora->cond_emb;
    
    if (gguf_name == "decoder.time_embed.linear_1.weight") return &lora->time_embed_linear_1;
    if (gguf_name == "decoder.time_embed.linear_2.weight") return &lora->time_embed_linear_2;
    if (gguf_name == "decoder.time_embed.time_proj.weight") return &lora->time_embed_time_proj;
    if (gguf_name == "decoder.time_embed_r.linear_1.weight") return &lora->time_embed_r_linear_1;
    if (gguf_name == "decoder.time_embed_r.linear_2.weight") return &lora->time_embed_r_linear_2;
    if (gguf_name == "decoder.time_embed_r.time_proj.weight") return &lora->time_embed_r_time_proj;

    // Parse: "decoder.layers.<N>.<block>.<proj>.weight"
    const char * p = gguf_name.c_str();
    if (strncmp(p, "decoder.layers.", 15) != 0) return nullptr;
    p += 15;

    char * end = nullptr;
    int layer = (int) strtol(p, &end, 10);
    if (end == p || layer < 0 || layer >= DIT_LORA_MAX_LAYERS) return nullptr;
    p = end;
    if (*p != '.') return nullptr;
    p++;

    DiTLoRALayer & ly = lora->layers[layer];

    if (strncmp(p, "self_attn.", 10) == 0) {
        p += 10;
        if (strncmp(p, "q_proj.weight", 13) == 0) return &ly.sa_q;
        if (strncmp(p, "k_proj.weight", 13) == 0) return &ly.sa_k;
        if (strncmp(p, "v_proj.weight", 13) == 0) return &ly.sa_v;
        if (strncmp(p, "o_proj.weight", 13) == 0) return &ly.sa_o;
    } else if (strncmp(p, "cross_attn.", 11) == 0) {
        p += 11;
        if (strncmp(p, "q_proj.weight", 13) == 0) return &ly.ca_q;
        if (strncmp(p, "k_proj.weight", 13) == 0) return &ly.ca_k;
        if (strncmp(p, "v_proj.weight", 13) == 0) return &ly.ca_v;
        if (strncmp(p, "o_proj.weight", 13) == 0) return &ly.ca_o;
    } else if (strncmp(p, "mlp.", 4) == 0) {
        p += 4;
        if (strncmp(p, "gate_proj.weight", 16) == 0) return &ly.gate;
        if (strncmp(p, "up_proj.weight", 14) == 0) return &ly.up;
        if (strncmp(p, "down_proj.weight", 16) == 0) return &ly.down;
    }
    return nullptr;
}

// adapter_compute_delta is defined in adapter-merge.h (shared with split merge path)

// Choose the VRAM storage type for a runtime delta from g_hotstep_params, honoring
// per-tensor block-size constraints (Q4_K needs ne0 % 256 == 0, Q8_0 needs
// ne0 % 32 == 0, where ne0 = in_features is the mul_mat contraction dim). Falls
// back to BF16 when the requested quant can't tile this tensor. Quantizing the
// precomputed deltas cuts VRAM (~½ for Q8_0, ~¼ for Q4_K) so many stacked adapters
// fit; quality impact is small since the base is usually already 4-bit.
static inline ggml_type adapter_runtime_storage_type(int64_t ne0) {
    // Q4_0 / Q8_0 both quantize per 32-block (fast, no optimisation pass — unlike
    // Q4_K which runs a slow per-superblock search that stalls load for minutes).
    const std::string & q = g_hotstep_params.adapter_runtime_quant;
    if ((ne0 % 32) == 0) {
        if (q == "q4_0" || q == "q4_k") return GGML_TYPE_Q4_0;  // accept legacy "q4_k" alias
        if (q == "q8_0")                return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_BF16;
}

// Stage a precomputed delta: create the storage tensor in lora context, store F32
// data for later (quantized) upload. The tensor and data are paired so upload
// order doesn't matter.
static void adapter_stage_delta(DiTLoRA * lora, DiTLoRADelta * slot,
                                 const std::string & gguf_name,
                                 int64_t ne0, int64_t ne1,
                                 std::vector<float> && delta_f32) {
    // Multi-adapter stacking: if this projection already has a staged delta from
    // an earlier adapter in the stack, sum into it (same base tensor => identical
    // shape) rather than allocating a second tensor. This keeps per-step inference
    // cost and VRAM constant regardless of how many adapters are stacked.
    if (slot->delta != nullptr) {
        for (auto & sd : lora->staged) {
            if (sd.tensor == slot->delta) {
                if (sd.f32_data.size() == delta_f32.size()) {
                    for (size_t i = 0; i < delta_f32.size(); i++) {
                        sd.f32_data[i] += delta_f32[i];
                    }
                } else {
                    fprintf(stderr, "[Adapter-RT] WARNING: delta shape mismatch stacking %s (%zu vs %zu), skipping add\n",
                            gguf_name.c_str(), sd.f32_data.size(), delta_f32.size());
                }
                return;
            }
        }
        // slot->delta set but not in staged (shouldn't happen) — fall through.
    }
    char tname[128];
    snprintf(tname, sizeof(tname), "lora_%s", gguf_name.c_str());
    slot->delta = ggml_new_tensor_2d(lora->ctx, adapter_runtime_storage_type(ne0), ne0, ne1);
    ggml_set_name(slot->delta, tname);
    lora->staged.push_back({ slot->delta, std::move(delta_f32) });
}

// ─── LoRA runtime loading ───

static bool adapter_runtime_lora(DiTLoRA *                  lora,
                                  WeightCtx *                wctx,
                                  const WeightSource &       ws,
                                  const STFile &             st,
                                  const std::string &        cfg_dir,
                                  float                      scale,
                                  const AdapterGroupScales & gs,
                                  ggml_backend_t             backend) {
    int alpha_cfg = adapter_read_alpha(cfg_dir.c_str());

    std::map<std::string, const STEntry *> a_map, b_map;
    std::map<std::string, float>           alpha_map;
    for (const auto & e : st.entries) {
        const char * alpha_suffix = ".alpha";
        size_t       slen         = strlen(alpha_suffix);
        if (e.name.size() > slen && e.name.compare(e.name.size() - slen, slen, alpha_suffix) == 0 && e.dtype == "F32" &&
            e.n_dims == 0) {
            std::string fake_key = e.name.substr(0, e.name.size() - slen) + ".lora_.x";
            std::string base     = lora_base_name(fake_key);
            if (!base.empty()) {
                float val = 0.0f;
                memcpy(&val, st_data(st, e), sizeof(float));
                alpha_map[base] = val;
            }
            continue;
        }
        std::string base = lora_base_name(e.name);
        if (base.empty()) continue;
        if (lora_is_a(e.name))      a_map[base] = &e;
        else if (lora_is_b(e.name)) b_map[base] = &e;
    }

    int  merged = 0, skipped = 0;
    bool cancelled = false;

    // Determine precision-rounding type for delta computation.
    // BF16 is ideal (matches storage format) but requires a bf16→f32 copy
    // shader.  Vulkan lacks this, so fall back to F16 (more mantissa bits,
    // so the delta is actually slightly more precise — negligible difference).
    ggml_type round_type = GGML_TYPE_BF16;
    {
        std::string bname = ggml_backend_name(backend);
        if (bname.find("Vulkan") != std::string::npos) {
            round_type = GGML_TYPE_F16;
            fprintf(stderr, "[Adapter-RT] Vulkan backend: using F16 precision rounding\n");
        }
    }

    for (const auto & kv : a_map) {
        // Cooperative cancel: check between every delta. The dominant cost
        // inside the body is adapter_compute_delta (GPU graph build + run),
        // which is non-trivial to interrupt — best we can do is stop
        // queueing more work.  Sub-100ms granularity in practice.
        if (adapter_cancel_requested()) {
            fprintf(stderr, "[Adapter-RT] LoRA: cancelled at delta %d (merged=%d, skipped=%d)\n",
                    merged + skipped, merged, skipped);
            cancelled = true;
            break;
        }

        const std::string & gguf_name = kv.first;
        const STEntry *     ea        = kv.second;

        auto it = b_map.find(gguf_name);
        if (it == b_map.end()) {
            fprintf(stderr, "[Adapter-RT] WARNING: no lora_B for %s, skipping\n", gguf_name.c_str());
            skipped++; continue;
        }
        const STEntry * eb = it->second;

        // Check base tensor exists
        if (!ws.exists(gguf_name.c_str())) {
            fprintf(stderr, "[Adapter-RT] WARNING: tensor %s not found in base model, skipping\n", gguf_name.c_str());
            skipped++; continue;
        }
        int n_dims; int64_t ne_arr[4];
        ws.shape(gguf_name.c_str(), n_dims, ne_arr);
        int64_t ne0 = ne_arr[0], ne1 = ne_arr[1];

        // Conv1d pre-permuted check
        if (n_dims >= 3 && wctx && wctx->ctx) {
            for (struct ggml_tensor * t = ggml_get_first_tensor(wctx->ctx); t != nullptr; t = ggml_get_next_tensor(wctx->ctx, t)) {
                if (t->name && gguf_name == t->name) {
                    ne0 = t->ne[0];
                    ne1 = t->ne[1];
                    fprintf(stderr, "[Adapter-RT] Conv1d %s: using pre-permuted shape [%lld, %lld]\n",
                            gguf_name.c_str(), (long long) ne0, (long long) ne1);
                    break;
                }
            }
        }

        DiTLoRADelta * slot = dit_lora_slot(lora, gguf_name);
        if (!slot) {
            fprintf(stderr, "[Adapter-RT] INFO: no runtime slot for %s (non-layer weight, merge-only)\n",
                    gguf_name.c_str());
            skipped++; continue;
        }

        int64_t rank = ea->shape[0], in_feat = ea->shape[1], out_feat = eb->shape[0];
        if (eb->shape[1] != rank) {
            fprintf(stderr, "[Adapter-RT] WARNING: rank mismatch A=%lld vs B=%lld for %s\n", (long long) rank,
                    (long long) eb->shape[1], gguf_name.c_str());
            skipped++; continue;
        }

        // Conv1d expansion check
        int64_t conv_expand = 1;
        if (in_feat == ne0 && out_feat == ne1) {
            // exact match
        } else if (out_feat == ne1 && ne0 > 0 && (ne0 % in_feat) == 0) {
            conv_expand = ne0 / in_feat;
            fprintf(stderr, "[Adapter-RT] Conv1d %s: tiling LoRA delta [%lld, %lld] x%lld -> [%lld, %lld]\n",
                    gguf_name.c_str(), (long long) in_feat, (long long) out_feat,
                    (long long) conv_expand, (long long) ne0, (long long) ne1);
        } else {
            fprintf(stderr, "[Adapter-RT] WARNING: shape mismatch for %s: adapter [%lld,%lld] (rank=%lld) vs GGUF [%lld,%lld]\n",
                    gguf_name.c_str(), (long long) out_feat, (long long) in_feat, (long long) rank,
                    (long long) ne1, (long long) ne0);
            skipped++; continue;
        }

        float alpha;
        auto alpha_it = alpha_map.find(gguf_name);
        if (alpha_it != alpha_map.end())   alpha = alpha_it->second;
        else if (alpha_cfg > 0)            alpha = (float) alpha_cfg;
        else                               alpha = (float) rank;

        float g_scale = adapter_group_scale_for(gs, adapter_determine_group(gguf_name));
        float scaling = (alpha / (float) rank) * scale * g_scale;

        // Per-tensor detail suppressed (uncomment for debugging)
        // fprintf(stderr, "[Adapter-RT]   %s → scaling=%.4f\n", gguf_name.c_str(), scaling);

        int64_t a_nel = rank * in_feat, b_nel = out_feat * rank;
        std::vector<float> a_f32((size_t) a_nel), b_f32((size_t) b_nel);
        if (!adapter_to_f32(st_data(st, *ea), a_f32.data(), a_nel, ea->dtype) ||
            !adapter_to_f32(st_data(st, *eb), b_f32.data(), b_nel, eb->dtype)) {
            fprintf(stderr, "[Adapter-RT] WARNING: unsupported dtype (A=%s, B=%s) for %s, skipping\n",
                    ea->dtype.c_str(), eb->dtype.c_str(), gguf_name.c_str());
            skipped++; continue;
        }

        auto build = [&](struct ggml_context * ctx) {
            struct ggml_tensor * ta     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, rank);
            struct ggml_tensor * tb     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out_feat);
            struct ggml_tensor * ta_br  = ggml_cast(ctx, ggml_cast(ctx, ta, round_type), GGML_TYPE_F32);
            struct ggml_tensor * tb_br  = ggml_cast(ctx, ggml_cast(ctx, tb, round_type), GGML_TYPE_F32);
            struct ggml_tensor * ta_t   = ggml_cont(ctx, ggml_transpose(ctx, ta_br));
            struct ggml_tensor * tdelta = ggml_scale(ctx, ggml_mul_mat(ctx, ta_t, tb_br), scaling);

            // Conv1d expansion: tile delta [in_ch, out] -> [in_ch*P, out]
            if (conv_expand > 1) {
                struct ggml_tensor * ttile = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat * conv_expand, out_feat);
                tdelta = ggml_repeat(ctx, tdelta, ttile);
            }

            adapter_delta_build db;
            db.tdelta = tdelta;
            db.upload = [=]() {
                ggml_backend_tensor_set(ta, a_f32.data(), 0, (size_t) a_nel * sizeof(float));
                ggml_backend_tensor_set(tb, b_f32.data(), 0, (size_t) b_nel * sizeof(float));
            };
            return db;
        };

        std::vector<float> delta_f32;
        if (!adapter_compute_delta(build, ne0, ne1, backend, delta_f32)) {
            fprintf(stderr, "[Adapter-RT] WARNING: GPU delta compute failed for %s, skipping\n", gguf_name.c_str());
            skipped++; continue;
        }

        adapter_stage_delta(lora, slot, gguf_name, ne0, ne1, std::move(delta_f32));
        merged++;
    }

    if (cancelled) {
        return false;
    }
    fprintf(stderr, "[Adapter-RT] LoRA: %d deltas precomputed (%d skipped), scale=%.2f\n", merged, skipped, scale);
    return merged > 0;
}

// ─── LoKr runtime loading ───

static bool adapter_runtime_lokr(DiTLoRA *                  lora,
                                  WeightCtx *                wctx,
                                  const WeightSource &       ws,
                                  const STFile &             st,
                                  float                      user_scale,
                                  const AdapterGroupScales & gs,
                                  ggml_backend_t             backend) {
    struct LoKrEntry {
        const STEntry * w1 = nullptr, * w2 = nullptr, * w2_a = nullptr, * w2_b = nullptr;
        const STEntry * alpha = nullptr, * dora_scale = nullptr;
    };

    std::map<std::string, LoKrEntry> modules;
    for (const auto & e : st.entries) {
        std::string prefix, suffix;
        if (!adapter_split_suffix(e.name, &prefix, &suffix)) continue;
        if (prefix.compare(0, 8, "lycoris_") != 0) continue;
        LoKrEntry & m = modules[prefix];
        if (suffix == "lokr_w1")         m.w1 = &e;
        else if (suffix == "lokr_w2")    m.w2 = &e;
        else if (suffix == "lokr_w2_a")  m.w2_a = &e;
        else if (suffix == "lokr_w2_b")  m.w2_b = &e;
        else if (suffix == "alpha")      m.alpha = &e;
        else if (suffix == "dora_scale") m.dora_scale = &e;
    }

    // DoRA detection: dora_scale is a per-row multiplicative rescale of (W+ΔW)
    // that cannot be expressed as a precomputed additive delta. The MERGE path
    // applies it; this runtime path does not — warn loudly rather than produce
    // silently-wrong output.
    {
        int dora_n = 0;
        for (const auto & kv : modules) {
            if (kv.second.dora_scale) dora_n++;
        }
        if (dora_n > 0) {
            fprintf(stderr, "[Adapter-RT] WARNING: %d LoKr module(s) carry dora_scale — DoRA rescaling is NOT applied in runtime mode; output will differ from merge mode. Use merge mode for DoRA adapters.\n",
                    dora_n);
        }
    }

    std::unordered_map<std::string, std::string> name_map = lokr_build_reverse_map(ws);
    int  lokr_dim = adapter_read_lokr_dim(st);
    int  merged = 0, skipped = 0;
    bool cancelled = false;

    for (const auto & kv : modules) {
        // Cooperative cancel: this is the ~17 s hot loop. Check every iter so
        // a wrapper-side cancel during cold start aborts in <100 ms instead of
        // waiting for all 352 deltas to finish.
        if (adapter_cancel_requested()) {
            fprintf(stderr, "[Adapter-RT] LoKr: cancelled at delta %d/%zu (merged=%d, skipped=%d)\n",
                    merged + skipped, modules.size(), merged, skipped);
            cancelled = true;
            break;
        }

        const std::string & lyc_prefix = kv.first;
        const LoKrEntry &   m          = kv.second;

        bool has_factor = (m.w2_a && m.w2_b);
        bool has_mono   = (m.w2 != nullptr);
        if (!m.w1 || !m.alpha || has_factor == has_mono) {
            fprintf(stderr, "[Adapter-RT] WARNING: incomplete/ambiguous LoKr module %s (w1=%d alpha=%d factor=%d mono=%d), skipping\n",
                    lyc_prefix.c_str(), !!m.w1, !!m.alpha, has_factor, has_mono);
            skipped++; continue;
        }

        auto nm_it = name_map.find(lyc_prefix);
        if (nm_it == name_map.end()) {
            fprintf(stderr, "[Adapter-RT] WARNING: no GGUF tensor mapping for %s, skipping\n", lyc_prefix.c_str());
            skipped++; continue;
        }
        const std::string & gguf_name = nm_it->second;

        if (!ws.exists(gguf_name.c_str())) {
            fprintf(stderr, "[Adapter-RT] WARNING: tensor %s not found in base model, skipping\n", gguf_name.c_str());
            skipped++; continue;
        }
        int n_dims; int64_t ne_arr[4];
        ws.shape(gguf_name.c_str(), n_dims, ne_arr);
        int64_t ne0 = ne_arr[0], ne1 = ne_arr[1];

        // Conv1d pre-permuted check
        if (n_dims >= 3 && wctx && wctx->ctx) {
            for (struct ggml_tensor * t = ggml_get_first_tensor(wctx->ctx); t != nullptr; t = ggml_get_next_tensor(wctx->ctx, t)) {
                if (t->name && gguf_name == t->name) {
                    ne0 = t->ne[0];
                    ne1 = t->ne[1];
                    fprintf(stderr, "[Adapter-RT] Conv1d %s: using pre-permuted shape [%lld, %lld]\n",
                            gguf_name.c_str(), (long long) ne0, (long long) ne1);
                    break;
                }
            }
        }

        DiTLoRADelta * slot = dit_lora_slot(lora, gguf_name);
        if (!slot) {
            fprintf(stderr, "[Adapter-RT] INFO: no runtime slot for %s (non-layer weight, merge-only)\n",
                    gguf_name.c_str());
            skipped++; continue;
        }

        // LoKr shapes
        int64_t a = m.w1->shape[0], b = m.w1->shape[1], c, d, r;
        if (has_factor) {
            c = m.w2_a->shape[0]; r = m.w2_a->shape[1]; d = m.w2_b->shape[1];
            if (r != m.w2_b->shape[0]) {
                fprintf(stderr, "[Adapter-RT] WARNING: LoKr rank mismatch w2_a=%lld vs w2_b=%lld for %s\n",
                        (long long) r, (long long) m.w2_b->shape[0], lyc_prefix.c_str());
                skipped++; continue;
            }
        } else {
            c = m.w2->shape[0]; d = m.w2->shape[1];
            if (lokr_dim <= 0) {
                fprintf(stderr, "[Adapter-RT] WARNING: monolithic LoKr %s needs __metadata__.lokr_config.linear_dim, skipping\n",
                        lyc_prefix.c_str());
                skipped++; continue;
            }
            r = lokr_dim;
        }

        // Conv1d expansion check
        int64_t conv_expand = 1;
        if (a * c == ne1 && b * d == ne0) {
            // exact match
        } else if (a * c == ne1 && ne0 > 0 && (ne0 % (b * d)) == 0) {
            conv_expand = ne0 / (b * d);
            fprintf(stderr, "[Adapter-RT] Conv1d %s: tiling LoKr delta [%lld, %lld] x%lld -> [%lld, %lld]\n",
                    gguf_name.c_str(), (long long)(b * d), (long long)(a * c),
                    (long long) conv_expand, (long long) ne0, (long long) ne1);
        } else {
            fprintf(stderr, "[Adapter-RT] WARNING: LoKr shape mismatch for %s: kron(%lldx%lld, %lldx%lld) = %lldx%lld vs GGUF [%lld,%lld]\n",
                    gguf_name.c_str(), (long long) a, (long long) b, (long long) c, (long long) d,
                    (long long)(a*c), (long long)(b*d), (long long) ne1, (long long) ne0);
            skipped++; continue;
        }

        float alpha_val = 0.0f;
        if (!adapter_to_f32(st_data(st, *m.alpha), &alpha_val, 1, m.alpha->dtype)) {
            fprintf(stderr, "[Adapter-RT] WARNING: unsupported alpha dtype %s for %s, skipping\n",
                    m.alpha->dtype.c_str(), lyc_prefix.c_str());
            skipped++; continue;
        }

        int64_t w1_nel = a * b;
        std::vector<float> w1_f32((size_t) w1_nel);
        if (!adapter_to_f32(st_data(st, *m.w1), w1_f32.data(), w1_nel, m.w1->dtype)) {
            fprintf(stderr, "[Adapter-RT] WARNING: unsupported w1 dtype %s for %s, skipping\n",
                    m.w1->dtype.c_str(), lyc_prefix.c_str());
            skipped++; continue;
        }

        int64_t w2_nel = 0, w2a_nel = 0, w2b_nel = 0;
        std::vector<float> w2_f32, w2a_f32, w2b_f32;
        if (has_factor) {
            w2a_nel = c * r; w2b_nel = r * d;
            w2a_f32.resize((size_t) w2a_nel); w2b_f32.resize((size_t) w2b_nel);
            if (!adapter_to_f32(st_data(st, *m.w2_a), w2a_f32.data(), w2a_nel, m.w2_a->dtype) ||
                !adapter_to_f32(st_data(st, *m.w2_b), w2b_f32.data(), w2b_nel, m.w2_b->dtype)) {
                fprintf(stderr, "[Adapter-RT] WARNING: unsupported w2 factor dtype for %s, skipping\n", lyc_prefix.c_str());
                skipped++; continue;
            }
        } else {
            w2_nel = c * d; w2_f32.resize((size_t) w2_nel);
            if (!adapter_to_f32(st_data(st, *m.w2), w2_f32.data(), w2_nel, m.w2->dtype)) {
                fprintf(stderr, "[Adapter-RT] WARNING: unsupported w2 dtype %s for %s, skipping\n",
                        m.w2->dtype.c_str(), lyc_prefix.c_str());
                skipped++; continue;
            }
        }

        float g_scale = adapter_group_scale_for(gs, adapter_determine_group(gguf_name));
        float scaling = (alpha_val / (float) r) * g_scale;
        // Per-tensor detail suppressed (uncomment for debugging)
        // fprintf(stderr, "[Adapter-RT]   %s → scaling=%.4f\n", gguf_name.c_str(), scaling);

        // Build the same Kronecker product graph as adapter_merge_lokr
        auto build = [&](struct ggml_context * ctx) {
            struct ggml_tensor * tw1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b, a);
            struct ggml_tensor * tw2, * tw2_src = nullptr, * tw2a = nullptr, * tw2b = nullptr;
            if (has_factor) {
                tw2a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, c);
                tw2b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, r);
                tw2  = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, tw2b)), tw2a);
            } else {
                tw2_src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, c);
                tw2     = tw2_src;
            }

            struct ggml_tensor * tw1_s    = ggml_scale(ctx, tw1, scaling * user_scale);
            struct ggml_tensor * tw1_flat = ggml_reshape_2d(ctx, tw1_s, 1, a * b);
            struct ggml_tensor * tw2_flat = ggml_reshape_2d(ctx, tw2, 1, c * d);
            struct ggml_tensor * touter   = ggml_mul_mat(ctx, tw1_flat, tw2_flat);
            struct ggml_tensor * t4d      = ggml_reshape_4d(ctx, touter, b, a, d, c);
            struct ggml_tensor * tperm    = ggml_permute(ctx, t4d, 1, 3, 0, 2);
            struct ggml_tensor * tdelta   = ggml_reshape_2d(ctx, ggml_cont(ctx, tperm), b * d, a * c);

            // Conv1d expansion: tile delta [in_ch, out] -> [in_ch*P, out]
            if (conv_expand > 1) {
                struct ggml_tensor * ttile = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b * d * conv_expand, a * c);
                tdelta = ggml_repeat(ctx, tdelta, ttile);
            }

            adapter_delta_build db;
            db.tdelta = tdelta;
            db.upload = [=]() {
                ggml_backend_tensor_set(tw1, w1_f32.data(), 0, (size_t) w1_nel * sizeof(float));
                if (has_factor) {
                    ggml_backend_tensor_set(tw2a, w2a_f32.data(), 0, (size_t) w2a_nel * sizeof(float));
                    ggml_backend_tensor_set(tw2b, w2b_f32.data(), 0, (size_t) w2b_nel * sizeof(float));
                } else {
                    ggml_backend_tensor_set(tw2_src, w2_f32.data(), 0, (size_t) w2_nel * sizeof(float));
                }
            };
            return db;
        };

        std::vector<float> delta_f32;
        if (!adapter_compute_delta(build, ne0, ne1, backend, delta_f32)) {
            fprintf(stderr, "[Adapter-RT] WARNING: GPU delta compute failed for %s, skipping\n", gguf_name.c_str());
            skipped++; continue;
        }

        adapter_stage_delta(lora, slot, gguf_name, ne0, ne1, std::move(delta_f32));
        merged++;
    }

    if (cancelled) {
        return false;
    }
    fprintf(stderr, "[Adapter-RT] LoKr: %d deltas precomputed (%d skipped), scale=%.2f\n",
            merged, skipped, user_scale);
    return merged > 0;
}

// ─── Per-file staging ───

// Open one adapter (PEFT dir or flat file) and STAGE its per-projection deltas
// into lora->ctx / lora->staged. Requires lora->ctx already initialised. Does
// not allocate the backend buffer or upload — that happens once after the whole
// stack is staged (see adapter_load_runtime_stack). Deltas for a projection
// already staged by an earlier adapter are summed in (adapter_stage_delta).
static bool adapter_runtime_stage_file(DiTLoRA *                  lora,
                                       WeightCtx *                wctx,
                                       const WeightSource &       ws,
                                       const char *               adapter_path,
                                       float                      adapter_scale,
                                       const AdapterGroupScales & gs,
                                       ggml_backend_t             backend) {
    // Load safetensors (same logic as adapter_merge)
    std::string path_str(adapter_path);
    STFile st;
    bool   loaded = false;

    // Try as directory first (PEFT format)
    std::string st_path = path_str + "/adapter_model.safetensors";
    if (st_open(&st, st_path.c_str())) {
        loaded = true;
    }
    if (!loaded) {
        st_path = path_str + "/adapter_model.bin.safetensors";
        if (st_open(&st, st_path.c_str())) {
            loaded = true;
        }
    }
    // Try as flat file
    if (!loaded && st_open(&st, adapter_path)) {
        loaded  = true;
        st_path = adapter_path;
    }
    if (!loaded) {
        fprintf(stderr, "[Adapter-RT] FATAL: cannot open adapter at %s\n", adapter_path);
        return false;
    }

    fprintf(stderr, "[Adapter-RT] Loading runtime adapter from %s (scale=%.2f)\n", st_path.c_str(), adapter_scale);

    bool ok;
    if (adapter_detect_lokr(st)) {
        ok = adapter_runtime_lokr(lora, wctx, ws, st, adapter_scale, gs, backend);
    } else {
        std::string cfg_dir;
        size_t slash = path_str.find_last_of("/\\");
        cfg_dir = (slash != std::string::npos) ? path_str.substr(0, slash) : ".";
        if (path_str.find(".safetensors") != std::string::npos) {
            // flat file — config dir is same as file dir
        } else {
            cfg_dir = path_str;  // directory format
        }
        ok = adapter_runtime_lora(lora, wctx, ws, st, cfg_dir, adapter_scale, gs, backend);
    }

    st_close(&st);
    return ok;
}

// Finalise a staged stack: allocate one backend buffer for all delta tensors and
// upload them (F32 staging -> BF16). Frees the ctx and returns false on failure.
static bool adapter_runtime_finalize(DiTLoRA * lora, ggml_backend_t backend) {
    if (lora->staged.empty()) {
        fprintf(stderr, "[Adapter-RT] WARNING: no deltas computed\n");
        if (lora->ctx) { ggml_free(lora->ctx); lora->ctx = nullptr; }
        return false;
    }

    // Allocate one backend buffer for all delta tensors
    lora->buffer = ggml_backend_alloc_ctx_tensors(lora->ctx, backend);
    if (!lora->buffer) {
        fprintf(stderr, "[Adapter-RT] FATAL: failed to allocate lora buffer\n");
        ggml_free(lora->ctx);
        lora->ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_set_usage(lora->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Upload: quantize/convert each staged F32 delta into its storage type
    // (BF16 / Q8_0 / Q4_0). Quantization is CPU-bound and per-tensor independent,
    // so do it in PARALLEL into host buffers first, then upload serially (GPU
    // calls must be serialized). A single-threaded pass over 360 large tensors
    // stalls the load for many seconds; threading keeps it snappy.
    Timer  qtimer;
    size_t n_staged = lora->staged.size();
    std::vector<std::vector<char>> host_bufs(n_staged);
    std::vector<size_t>            host_nbytes(n_staged, 0);
    auto quant_worker = [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; j++) {
            struct ggml_tensor * t   = lora->staged[j].tensor;
            const float *        src = lora->staged[j].f32_data.data();
            int64_t              nel = ggml_nelements(t);
            if (t->type == GGML_TYPE_BF16) {
                host_nbytes[j] = (size_t) nel * sizeof(ggml_bf16_t);
                host_bufs[j].resize(host_nbytes[j]);
                ggml_fp32_to_bf16_row(src, (ggml_bf16_t *) host_bufs[j].data(), nel);
            } else {
                int64_t ne0 = t->ne[0], ne1 = t->ne[1];
                host_nbytes[j] = ggml_row_size(t->type, ne0) * (size_t) ne1;
                host_bufs[j].resize(host_nbytes[j]);
                ggml_quantize_chunk(t->type, src, host_bufs[j].data(), 0, ne1, ne0, nullptr);
            }
        }
    };
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > n_staged) nthreads = (unsigned) std::max<size_t>(1, n_staged);
    if (nthreads <= 1 || n_staged == 0) {
        quant_worker(0, n_staged);
    } else {
        std::vector<std::thread> pool;
        size_t chunk = (n_staged + nthreads - 1) / nthreads;
        for (unsigned ti = 0; ti < nthreads; ti++) {
            size_t b = ti * chunk, e = std::min(n_staged, b + chunk);
            if (b < e) pool.emplace_back(quant_worker, b, e);
        }
        for (auto & th : pool) th.join();
    }
    size_t total_bytes = 0;
    int    n_quant     = 0;
    for (size_t j = 0; j < n_staged; j++) {
        ggml_backend_tensor_set(lora->staged[j].tensor, host_bufs[j].data(), 0, host_nbytes[j]);
        total_bytes += host_nbytes[j];
        if (lora->staged[j].tensor->type != GGML_TYPE_BF16) n_quant++;
    }
    fprintf(stderr, "[Adapter-RT] Quantize+upload: %.1f ms (%u threads)\n", qtimer.ms(), nthreads);

    size_t n_deltas = lora->staged.size();
    lora->staged.clear();
    lora->active = true;

    fprintf(stderr, "[Adapter-RT] Loaded %zu deltas (%.1f MB, %d quantized to %s) into VRAM\n",
            n_deltas, (float) total_bytes / (1024 * 1024), n_quant,
            g_hotstep_params.adapter_runtime_quant.c_str());
    return true;
}

// ─── Main entry points ───

// Multi-adapter stack: stage every adapter's deltas into one DiTLoRA, summing
// per projection, then allocate + upload once. Per-step inference cost and VRAM
// are independent of stack depth (deltas collapse into one set per projection).
static bool adapter_load_runtime_stack(DiTLoRA *                       lora,
                                       WeightCtx *                     wctx,
                                       const WeightSource &            ws,
                                       const std::vector<AdapterSpec> & stack,
                                       const AdapterGroupScales &      gs,
                                       ggml_backend_t                  backend) {
    if (stack.empty()) {
        fprintf(stderr, "[Adapter-RT] WARNING: empty adapter stack\n");
        return false;
    }

    // One context sized for the union of slots. Slots are shared across adapters
    // (deltas accumulate into the same tensor), so the per-adapter slot count
    // bounds the total — sizing as for a single adapter is sufficient.
    int max_deltas = DIT_LORA_MAX_LAYERS * 11 + 32;
    size_t ctx_size = (size_t) max_deltas * ggml_tensor_overhead() + 4096;
    struct ggml_init_params params = { ctx_size, NULL, true };
    lora->ctx = ggml_init(params);
    if (!lora->ctx) {
        fprintf(stderr, "[Adapter-RT] FATAL: failed to init lora context\n");
        return false;
    }

    int staged_ok = 0;
    for (size_t i = 0; i < stack.size(); i++) {
        if (adapter_runtime_stage_file(lora, wctx, ws, stack[i].path.c_str(), stack[i].scale, gs, backend)) {
            staged_ok++;
        } else {
            fprintf(stderr, "[Adapter-RT] WARNING: stack adapter %zu staged no deltas: %s\n", i, stack[i].path.c_str());
        }
        if (adapter_cancel_requested()) {
            fprintf(stderr, "[Adapter-RT] stack staging cancelled at adapter %zu/%zu\n", i + 1, stack.size());
            if (lora->ctx) { ggml_free(lora->ctx); lora->ctx = nullptr; }
            lora->staged.clear();
            return false;
        }
    }

    if (staged_ok == 0) {
        if (lora->ctx) { ggml_free(lora->ctx); lora->ctx = nullptr; }
        lora->staged.clear();
        return false;
    }

    return adapter_runtime_finalize(lora, backend);
}

// Single-adapter convenience wrapper (legacy call site / CLI / warm).
static bool adapter_load_runtime(DiTLoRA *                  lora,
                                  WeightCtx *                wctx,
                                  const WeightSource &       ws,
                                  const char *               adapter_path,
                                  float                      adapter_scale,
                                  const AdapterGroupScales & gs,
                                  ggml_backend_t             backend) {
    std::vector<AdapterSpec> single{ AdapterSpec{ std::string(adapter_path), adapter_scale } };
    return adapter_load_runtime_stack(lora, wctx, ws, single, gs, backend);
}
