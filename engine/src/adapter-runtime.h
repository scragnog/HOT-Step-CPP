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
#include "convrot.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "timer.h"
#include "weight-source.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <thread>
#include <unordered_map>
#include <vector>

// Cooperative cancel for the LoKr/LoRA delta precompute loop lives in
// adapter-cancel.h so ace-server.cpp can poke at it without pulling all of
// adapter-runtime.h's transitive ggml/safetensors deps. See that header.

// Low-rank factor unit (adapter_mode == "runtime_lowrank"): the adapter's raw
// factors kept in VRAM instead of the materialized full-size delta product.
// LoRA:  y += B@(A@x)          — a [in, r], b [r, out], scaling pre-folded into b.
// LoKr:  y += (w1 ⊗ w2)@x      — k1 [b1, a1], k2 [d2, c2], scaling pre-folded
//        into k1; applied via the Kronecker identity (validated by
//        HOTSTEP_KRON_TEST, see docs/plans/lowrank-runtime-adapters.md §8):
//        reshape → mul_mat(k2) → permute/cont → mul_mat(k1) → permute/cont.
// Exactly one of (a,b) or (k1,k2) is set. ~20-40× less VRAM than a full delta.
struct DiTLoRAFactorUnit {
    struct ggml_tensor * a  = nullptr;  // LoRA A  (ggml [in, r], BF16)
    struct ggml_tensor * b  = nullptr;  // LoRA B  (ggml [r, out], BF16, scale folded)
    struct ggml_tensor * k1 = nullptr;  // LoKr w1 (ggml [b1, a1], BF16, scale folded)
    struct ggml_tensor * k2 = nullptr;  // LoKr w2 (ggml [d2, c2], BF16)
};

// Per-projection runtime LoRA delta (stored as BF16/Q8_0/Q4_0 tensor in VRAM).
// In lowrank mode `delta` holds only the (optional) full-size components that
// cannot be expressed as factors — the basin re-base correction and Conv1d-tiled
// fallbacks — while `units` carries the adapter factors (one unit per adapter).
struct DiTLoRADelta {
    struct ggml_tensor *           delta = nullptr;  // [in, out], or NULL
    std::vector<DiTLoRAFactorUnit> units;            // lowrank mode only
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

// ─── Low-rank runtime mode (adapter_mode == "runtime_lowrank") ───

static inline bool adapter_runtime_lowrank_active() {
    return g_hotstep_params.adapter_mode == "runtime_lowrank";
}

// Stage one factor tensor: BF16 tensor in lora->ctx + F32 payload for finalize.
// Named "lrf_*" so adapter_runtime_rebase (which folds β·(S−T) into staged
// tensors named "lora_*") never touches factors — the base correction is
// full-rank and rides the zero-initialized correction deltas instead.
static struct ggml_tensor * adapter_stage_factor(DiTLoRA * lora, const char * kind,
                                                 const std::string & gguf_name,
                                                 int64_t ne0, int64_t ne1,
                                                 std::vector<float> && f32) {
    char tname[128];
    snprintf(tname, sizeof(tname), "lrf_%s_%s", kind, gguf_name.c_str());
    struct ggml_tensor * t = ggml_new_tensor_2d(lora->ctx, GGML_TYPE_BF16, ne0, ne1);
    ggml_set_name(t, tname);
    lora->staged.push_back({ t, std::move(f32) });
    return t;
}

// Zero-filled full-size correction delta for a slot: created for the FIRST stack
// adapter's slots when basin re-base is active in lowrank mode, so that
// adapter_runtime_rebase has a full-rank tensor to fold β·(S−T) into (factors
// can't carry it). Quantized like any runtime delta (adapter_runtime_quant).
static void adapter_stage_zero_correction(DiTLoRA * lora, DiTLoRADelta * slot,
                                          const std::string & gguf_name,
                                          int64_t ne0, int64_t ne1) {
    if (slot->delta) {
        return;  // already has a full-size component (e.g. Conv1d fallback)
    }
    adapter_stage_delta(lora, slot, gguf_name, ne0, ne1,
                        std::vector<float>((size_t) (ne0 * ne1), 0.0f));
}

// Graph node budget contribution of factor units (LoKr apply ≈ 9 nodes/unit,
// LoRA ≈ 3; use the LoKr bound plus slack). Callers add this to graph_cap and
// the scheduler hash-set size so lowrank stacks never overflow the graph.
static size_t dit_lora_unit_nodes(const DiTLoRA * lora) {
    if (!lora || !lora->active) return 0;
    size_t units = 0;
    for (int i = 0; i < DIT_LORA_MAX_LAYERS; i++) {
        const DiTLoRALayer & ly = lora->layers[i];
        const DiTLoRADelta * slots[11] = { &ly.sa_q, &ly.sa_k, &ly.sa_v, &ly.sa_o,
                                           &ly.ca_q, &ly.ca_k, &ly.ca_v, &ly.ca_o,
                                           &ly.gate, &ly.up, &ly.down };
        for (const DiTLoRADelta * s : slots) units += s->units.size();
    }
    units += lora->proj_in.units.size() + lora->cond_emb.units.size()
           + lora->time_embed_linear_1.units.size() + lora->time_embed_linear_2.units.size()
           + lora->time_embed_time_proj.units.size() + lora->time_embed_r_linear_1.units.size()
           + lora->time_embed_r_linear_2.units.size() + lora->time_embed_r_time_proj.units.size();
    return units * 12;
}

// ─── LoRA runtime loading ───

static bool adapter_runtime_lora(DiTLoRA *                  lora,
                                  WeightCtx *                wctx,
                                  const WeightSource &       ws,
                                  const STFile &             st,
                                  const std::string &        cfg_dir,
                                  float                      scale,
                                  const AdapterGroupScales & gs,
                                  ggml_backend_t             backend,
                                  bool                       zero_corr = false) {
    int alpha_cfg = adapter_read_alpha(cfg_dir.c_str());

    std::map<std::string, const STEntry *> a_map, b_map;
    std::map<std::string, float>           alpha_map;
    adapter_read_alpha_pattern(cfg_dir.c_str(), alpha_map);
    int                                    dora_n = 0;
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
        else if (lora_is_magnitude(e.name)) dora_n++;
    }
    if (dora_n > 0) {
        fprintf(stderr,
                "[Adapter-RT] WARNING: %d module(s) carry lora_magnitude_vector (PEFT DoRA) — DoRA rescaling is NOT applied in runtime mode; output will differ from merge mode. Use merge mode for DoRA adapters.\n",
                dora_n);
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

        // Low-rank mode: keep the raw factors in VRAM and apply B@(A@x) in the
        // graph — the full-size delta product is never materialized. Conv1d
        // tiled tensors (conv_expand > 1) keep the full-delta path: tiling a
        // factor pair across the patch dim has no low-rank form.
        if (adapter_runtime_lowrank_active() && conv_expand == 1) {
            if (zero_corr) {
                adapter_stage_zero_correction(lora, slot, gguf_name, ne0, ne1);
            }
            for (int64_t bi = 0; bi < b_nel; bi++) b_f32[(size_t) bi] *= scaling;
            DiTLoRAFactorUnit u;
            u.a = adapter_stage_factor(lora, "a", gguf_name, in_feat, rank, std::move(a_f32));
            u.b = adapter_stage_factor(lora, "b", gguf_name, rank, out_feat, std::move(b_f32));
            slot->units.push_back(u);
            merged++;
            continue;
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
                                  ggml_backend_t             backend,
                                  bool                       zero_corr = false) {
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

        // Low-rank mode: store w1/w2 and apply (w1 ⊗ w2)@x via the Kronecker
        // identity in the graph — the kron product is never materialized.
        // Conv1d-tiled tensors keep the full-delta path (same as LoRA above).
        if (adapter_runtime_lowrank_active() && conv_expand == 1) {
            if (zero_corr) {
                adapter_stage_zero_correction(lora, slot, gguf_name, ne0, ne1);
            }
            // Densify a factored w2 host-side (tiny: c×r×d products).
            if (has_factor) {
                w2_nel = c * d;
                w2_f32.assign((size_t) w2_nel, 0.0f);
                for (int64_t ci = 0; ci < c; ci++)
                    for (int64_t ki = 0; ki < r; ki++) {
                        float wa = w2a_f32[(size_t) (ci * r + ki)];
                        if (wa == 0.0f) continue;
                        const float * wb = &w2b_f32[(size_t) (ki * d)];
                        float *       wo = &w2_f32[(size_t) (ci * d)];
                        for (int64_t j = 0; j < d; j++) wo[j] += wa * wb[j];
                    }
            }
            float fold = scaling * user_scale;
            for (int64_t i1 = 0; i1 < w1_nel; i1++) w1_f32[(size_t) i1] *= fold;
            DiTLoRAFactorUnit u;
            u.k1 = adapter_stage_factor(lora, "k1", gguf_name, b, a, std::move(w1_f32));
            u.k2 = adapter_stage_factor(lora, "k2", gguf_name, d, c, std::move(w2_f32));
            slot->units.push_back(u);
            merged++;
            continue;
        }

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
                                       ggml_backend_t             backend,
                                       bool                       zero_corr = false) {
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
        ok = adapter_runtime_lokr(lora, wctx, ws, st, adapter_scale, gs, backend, zero_corr);
    } else {
        std::string cfg_dir;
        size_t slash = path_str.find_last_of("/\\");
        cfg_dir = (slash != std::string::npos) ? path_str.substr(0, slash) : ".";
        if (path_str.find(".safetensors") != std::string::npos) {
            // flat file — config dir is same as file dir
        } else {
            cfg_dir = path_str;  // directory format
        }
        ok = adapter_runtime_lora(lora, wctx, ws, st, cfg_dir, adapter_scale, gs, backend, zero_corr);
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

// ─── Basin re-base (runtime mode) ───
//
// Merge mode nudges the loaded base T toward the adapter's training base S
// before adding deltas: W' = T + beta*(S - T) + sum(s_i * d_i). Runtime mode
// never rewrites base weights, but matmul distributes over addition, so the
// SAME output falls out of folding beta*(S - T) into the staged delta sum:
//   y = T@x + (beta*(S - T) + sum(s_i * d_i))@x
// Zero extra VRAM and zero extra per-step cost — the nudge rides the existing
// delta tensors. Must run ONCE PER STACK over the FIRST adapter's staged slots
// (merge parity: merge threads rebase into the first adapter's merge only; a
// per-adapter nudge would multiply the base correction by the stack depth).
//
// S is read with adapter_rebase_fetch (2D linear, shape-matched tensors only —
// same skips as merge). T is dequantized host-side from the still-open weight
// source, so the nudge is exact against what actually sits on the GPU.
//
// ConvRot models: T is stored PRE-ROTATED (W·H per input-dim group), but S and
// the deltas live in unrotated space and the deltas consume unrotated
// activations at inference. Nudging with raw T would bake a rotated-space term
// into the delta — audible as total corruption (found the hard way 2026-07-15).
// convrot_groups (tensor name → group size) marks which T's must be unrotated
// host-side (fast radix-4 transform; H·H = I) before computing beta*(S - T).
static void adapter_runtime_rebase(DiTLoRA *            lora,
                                   const WeightSource & ws,
                                   const char *         rebase_source,
                                   float                rebase_beta,
                                   const std::unordered_map<std::string, int> * convrot_groups = nullptr) {
    if (!rebase_source || !rebase_source[0] || rebase_beta == 0.0f || lora->staged.empty()) {
        return;
    }

    // Resolve a model DIRECTORY to its model.safetensors (mirrors adapter_merge)
    std::string rb_path = rebase_source;
    struct stat rsb;
    if (stat(rb_path.c_str(), &rsb) == 0 && S_ISDIR(rsb.st_mode)) {
        rb_path += "/model.safetensors";
    }
    STFile rebase_st = {};
    if (!st_open(&rebase_st, rb_path.c_str())) {
        fprintf(stderr, "[Adapter-RT] WARNING: basin re-base source unreadable: %s (continuing without nudge)\n",
                rb_path.c_str());
        return;
    }

    // Per-slot work is independent (S/T reads are pure mmap pointer math), so
    // split across threads like the finalize quantizer — a single-threaded pass
    // over ~360 large tensors adds seconds to the load.
    Timer            rbtimer;
    size_t           n_staged = lora->staged.size();
    std::atomic<int> nudged{ 0 };
    auto             rebase_worker = [&](size_t begin, size_t end) {
        std::vector<float> sbuf, tbuf;
        for (size_t j = begin; j < end; j++) {
            if (adapter_cancel_requested()) {
                return;
            }
            DiTLoRAStagedDelta & sd    = lora->staged[j];
            const char *         tname = ggml_get_name(sd.tensor);
            if (strncmp(tname, "lora_", 5) != 0) {
                continue;  // staged tensors are named "lora_<gguf_name>"
            }
            const char * gguf_name = tname + 5;
            int64_t      ne0       = sd.tensor->ne[0];
            int64_t      ne1       = sd.tensor->ne[1];
            int64_t      nel       = ne0 * ne1;
            const float * s = adapter_rebase_fetch(&rebase_st, rebase_beta, gguf_name, ne0, ne1, sbuf);
            if (!s) {
                continue;  // not in S / conv / shape mismatch — same skips as merge
            }
            if ((int64_t) sd.f32_data.size() != nel) {
                continue;
            }
            ggml_type    ttype = GGML_TYPE_F32;
            const void * tdata = ws.data(gguf_name, ttype);
            if (!tdata) {
                continue;
            }
            tbuf.resize((size_t) nel);
            if (ttype == GGML_TYPE_F32) {
                memcpy(tbuf.data(), tdata, (size_t) nel * sizeof(float));
            } else if (ttype == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) tdata, tbuf.data(), nel);
            } else if (ttype == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) tdata, tbuf.data(), nel);
            } else {
                const struct ggml_type_traits * traits = ggml_get_type_traits(ttype);
                if (!traits || !traits->to_float) {
                    fprintf(stderr, "[Adapter-RT] WARNING: no host dequant for type %d, re-base skipping %s\n",
                            (int) ttype, gguf_name);
                    continue;
                }
                traits->to_float(tdata, tbuf.data(), nel);
            }
            // ConvRot: bring T back to unrotated space before the nudge
            if (convrot_groups) {
                auto it = convrot_groups->find(gguf_name);
                if (it != convrot_groups->end()) {
                    convrot_transform_rows(tbuf.data(), ne1, ne0, it->second);
                }
            }
            float * d = sd.f32_data.data();
            for (int64_t i = 0; i < nel; i++) {
                d[i] += rebase_beta * (s[i] - tbuf[i]);
            }
            nudged.fetch_add(1, std::memory_order_relaxed);
        }
    };
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > n_staged) nthreads = (unsigned) std::max<size_t>(1, n_staged);
    if (nthreads <= 1) {
        rebase_worker(0, n_staged);
    } else {
        std::vector<std::thread> pool;
        size_t chunk = (n_staged + nthreads - 1) / nthreads;
        for (unsigned ti = 0; ti < nthreads; ti++) {
            size_t b = ti * chunk, e = std::min(n_staged, b + chunk);
            if (b < e) pool.emplace_back(rebase_worker, b, e);
        }
        for (auto & th : pool) th.join();
    }
    st_close(&rebase_st);
    fprintf(stderr, "[Adapter-RT] Basin re-base: nudged %d/%zu deltas toward training base, beta=%.2f (%.1f ms)\n",
            nudged.load(), n_staged, rebase_beta, rbtimer.ms());
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
                                       ggml_backend_t                  backend,
                                       const char *                    rebase_source = nullptr,
                                       float                           rebase_beta = 0.0f,
                                       const std::unordered_map<std::string, int> * convrot_groups = nullptr) {
    if (stack.empty()) {
        fprintf(stderr, "[Adapter-RT] WARNING: empty adapter stack\n");
        return false;
    }

    // One context sized for the union of slots. Slots are shared across adapters
    // (deltas accumulate into the same tensor), so the per-adapter slot count
    // bounds the total — sizing as for a single adapter is sufficient.
    int max_deltas = DIT_LORA_MAX_LAYERS * 11 + 32;
    // Lowrank: each adapter stages up to 2 factor tensors per slot (not summed),
    // plus optional zero-correction deltas — size the ctx for the whole stack.
    if (adapter_runtime_lowrank_active()) {
        max_deltas *= (int) (2 * stack.size() + 1);
    }
    size_t ctx_size = (size_t) max_deltas * ggml_tensor_overhead() + 4096;
    struct ggml_init_params params = { ctx_size, NULL, true };
    lora->ctx = ggml_init(params);
    if (!lora->ctx) {
        fprintf(stderr, "[Adapter-RT] FATAL: failed to init lora context\n");
        return false;
    }

    // Lowrank + basin re-base: the correction is full-rank and can't ride the
    // factors, so the FIRST adapter's slots get zero-filled full-size deltas
    // for adapter_runtime_rebase to fold β·(S−T) into (once per stack).
    bool rebase_active = rebase_source && rebase_source[0] && rebase_beta != 0.0f;
    int staged_ok = 0;
    for (size_t i = 0; i < stack.size(); i++) {
        bool zero_corr = adapter_runtime_lowrank_active() && rebase_active && i == 0;
        if (adapter_runtime_stage_file(lora, wctx, ws, stack[i].path.c_str(), stack[i].scale, gs, backend, zero_corr)) {
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
        // Basin re-base: once per stack, over the FIRST adapter's staged slots
        // only — later adapters sum their deltas on top, exactly matching the
        // merge path's nudge-then-stack order (dit.h). See adapter_runtime_rebase.
        if (i == 0) {
            adapter_runtime_rebase(lora, ws, rebase_source, rebase_beta, convrot_groups);
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
                                  ggml_backend_t             backend,
                                  const char *               rebase_source = nullptr,
                                  float                      rebase_beta = 0.0f,
                                  const std::unordered_map<std::string, int> * convrot_groups = nullptr) {
    std::vector<AdapterSpec> single{ AdapterSpec{ std::string(adapter_path), adapter_scale } };
    return adapter_load_runtime_stack(lora, wctx, ws, single, gs, backend, rebase_source, rebase_beta, convrot_groups);
}
