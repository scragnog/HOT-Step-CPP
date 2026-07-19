#pragma once
// adapter-trt.h: LoRA/LoKr adapter merge for TRT engine via IRefitter
//
// Reads adapter safetensors (PEFT, ComfyUI, or LyCORIS LoKr format), maps
// adapter keys to TRT weight names, computes merged weights (base + scale *
// delta) on CPU in fp32, casts back to bf16, and calls dit_trt_refit_adapter()
// to hot-swap weights in the live TRT engine.
//
// Unlike the GGML merge path (adapter-merge.h) which operates on PendingCopy
// staging buffers before GPU upload, this operates on the live TRT engine
// via the IRefitter API — enabling adapter switching without engine reload.
//
// Supports both LoRA (A/B factorized) and LoKr (Kronecker product) adapters.
// Delta computation uses the GGML GPU backend (adapter_compute_delta from
// adapter-runtime.h) for GPU-accelerated Kronecker products and matmuls.

#ifdef HOT_STEP_TRT

#include "adapter-merge.h"    // lora_base_name(), adapter_read_alpha(), adapter_detect_lokr(), etc.
#include "adapter-runtime.h"  // adapter_compute_delta() — GPU delta computation
#include "backend.h"          // g_backend_cache — shared GGML CUDA backend
#include "dit-trt.h"          // DitTrt, dit_trt_refit_adapter/base
#include "hot-step-params.h"  // adapter_group_scales
#include "safetensors.h"      // STFile, st_open, st_data

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ── BF16 conversion (same as hot-step-sampler-trt.h) ────────────────────────
// Duplicated here to keep headers self-contained. BF16 is just the top 16 bits
// of FP32 with rounding.

static inline uint16_t trt_fp32_to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u += 0x7FFF + ((u >> 16) & 1);  // round to nearest even
    return (uint16_t)(u >> 16);
}

static inline float trt_bf16_to_fp32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float result;
    memcpy(&result, &u, 4);
    return result;
}

// Map an adapter base name to a TRT weight name.
//
// Adapter keys use "decoder." prefix (e.g. "decoder.layers.0.self_attn.q_proj.weight").
// The renamed ONNX uses "dit." prefix (e.g. "dit.layers.0.self_attn.q_proj.weight").
// Returns the matching key from base_weights, or "" if not found.
static std::string find_trt_weight_name(
    const DitTrt& ctx,
    const std::string& gguf_name  // e.g. "decoder.layers.0.self_attn.q_proj.weight"
) {
    // Try exact match
    if (ctx.base_weights.count(gguf_name)) return gguf_name;

    // decoder.X → dit.X
    if (gguf_name.compare(0, 8, "decoder.") == 0) {
        std::string dit_name = "dit." + gguf_name.substr(8);
        if (ctx.base_weights.count(dit_name)) return dit_name;
    }

    // Try dit. prefix on raw path
    std::string dit_prefixed = "dit." + gguf_name;
    if (ctx.base_weights.count(dit_prefixed)) return dit_prefixed;

    return "";
}

// ── CPU merge helper (used by LoRA path only) ───────────────────────────────
// The LoKr path uses GPU-side merge instead (base+delta→bf16 on GPU).
static void trt_merge_delta(
    const DitTrt& ctx,
    const std::string& trt_name,
    const std::vector<uint16_t>& base_data,
    const float* delta_f32,
    int64_t ne0, int64_t ne1,
    float scaling,
    std::unordered_map<std::string, std::vector<uint16_t>>& merged_storage,
    std::unordered_map<std::string, const void*>& merged_ptrs
) {
    int64_t nel = ne0 * ne1;
    bool is_transposed = ctx.weights_transposed.count(trt_name) > 0;
    std::vector<uint16_t> merged_bf16((size_t)nel);

    if (is_transposed) {
        for (int64_t in_idx = 0; in_idx < ne0; in_idx++) {
            for (int64_t out_idx = 0; out_idx < ne1; out_idx++) {
                int64_t base_idx  = in_idx * ne1 + out_idx;
                int64_t delta_idx = out_idx * ne0 + in_idx;
                float base_val = trt_bf16_to_fp32(base_data[base_idx]);
                float merged_val = base_val + scaling * delta_f32[delta_idx];
                merged_bf16[base_idx] = trt_fp32_to_bf16(merged_val);
            }
        }
    } else {
        for (int64_t j = 0; j < nel; j++) {
            float base_val = trt_bf16_to_fp32(base_data[j]);
            float merged_val = base_val + scaling * delta_f32[j];
            merged_bf16[j] = trt_fp32_to_bf16(merged_val);
        }
    }

    merged_storage[trt_name] = std::move(merged_bf16);
    merged_ptrs[trt_name] = merged_storage[trt_name].data();
}

// DoRA variant of trt_merge_delta: merged = (base + delta) * s_row with
// s_row = user * (ds[row] / (||base + delta||_row + eps)) + (1 - user).
// The delta carries only the alpha/rank factor (LyCORIS convention, same
// math as the DoRA branch in adapter_merge_on_backend).
static void trt_merge_delta_dora(
    const DitTrt& ctx,
    const std::string& trt_name,
    const std::vector<uint16_t>& base_data,
    const float* delta_f32,
    const float* ds,
    int64_t ne0, int64_t ne1,   // ne0 = in_feat, ne1 = out_feat
    float user_scale,
    std::unordered_map<std::string, std::vector<uint16_t>>& merged_storage,
    std::unordered_map<std::string, const void*>& merged_ptrs
) {
    // torch.finfo(torch.bfloat16).eps, matches adapter_merge_on_backend
    const float eps = 7.8125e-3f;
    int64_t nel = ne0 * ne1;
    bool is_transposed = ctx.weights_transposed.count(trt_name) > 0;
    std::vector<uint16_t> merged_bf16((size_t)nel);
    std::vector<float> row((size_t)ne0);

    for (int64_t out_idx = 0; out_idx < ne1; out_idx++) {
        float sum_sq = 0.0f;
        for (int64_t in_idx = 0; in_idx < ne0; in_idx++) {
            int64_t base_idx = is_transposed ? in_idx * ne1 + out_idx : out_idx * ne0 + in_idx;
            float v = trt_bf16_to_fp32(base_data[base_idx]) + delta_f32[out_idx * ne0 + in_idx];
            row[(size_t)in_idx] = v;
            sum_sq += v * v;
        }
        float s = ds[out_idx] / (sqrtf(sum_sq) + eps);
        if (user_scale != 1.0f) { s = s * user_scale + (1.0f - user_scale); }
        for (int64_t in_idx = 0; in_idx < ne0; in_idx++) {
            int64_t base_idx = is_transposed ? in_idx * ne1 + out_idx : out_idx * ne0 + in_idx;
            merged_bf16[base_idx] = trt_fp32_to_bf16(row[(size_t)in_idx] * s);
        }
    }

    merged_storage[trt_name] = std::move(merged_bf16);
    merged_ptrs[trt_name] = merged_storage[trt_name].data();
}

// ── Build TRT reverse map for LoKr ──────────────────────────────────────────
// Maps lycoris prefix (e.g. "lycoris_layers_0_self_attn_q_proj")
// to TRT weight name (e.g. "dit.layers.0.self_attn.q_proj.weight")
// by iterating base_weights keys.
static std::unordered_map<std::string, std::string> lokr_build_trt_reverse_map(
    const DitTrt& ctx
) {
    std::unordered_map<std::string, std::string> out;
    for (const auto& [trt_name, _] : ctx.base_weights) {
        // TRT names are "dit.X.weight" — build lycoris prefix from "X"
        // Strip "dit." prefix and ".weight" suffix
        if (trt_name.compare(0, 4, "dit.") != 0) continue;
        size_t wsuf = trt_name.rfind(".weight");
        if (wsuf == std::string::npos) continue;
        std::string path = trt_name.substr(4, wsuf - 4);
        for (char& ch : path) { if (ch == '.') ch = '_'; }
        out["lycoris_" + path] = trt_name;
    }
    return out;
}

// ── Get GGML backend for GPU delta computation ──────────────────────────────
// In TRT-only mode, the shared g_backend_cache may be empty (GGML DiT never
// loads). We lazily init our own CUDA backend for adapter delta computation.
static ggml_backend_t adapter_trt_get_backend() {
    // Try shared backend first (if LM or other GGML module initialized it)
    if (g_backend_refs > 0 && g_backend_cache.backend) {
        return g_backend_cache.backend;
    }

    // Lazily init our own backend
    static ggml_backend_t s_adapter_backend = nullptr;
    if (!s_adapter_backend) {
        ggml_backend_load_all();
        s_adapter_backend = ggml_backend_init_best();
        if (s_adapter_backend) {
            fprintf(stderr, "[Adapter-TRT] Initialized %s backend for delta computation\n",
                    ggml_backend_name(s_adapter_backend));
        } else {
            // Last resort: CPU
            s_adapter_backend = cpu_backend_new(backend_cpu_n_threads());
            if (s_adapter_backend) {
                fprintf(stderr, "[Adapter-TRT] Using CPU backend for delta computation\n");
            }
        }
    }
    return s_adapter_backend;
}

// ── LoRA merge path ─────────────────────────────────────────────────────────
static int adapter_trt_apply_lora(
    DitTrt* ctx,
    const STFile& st,
    const std::string& cfg_dir,
    float adapter_scale,
    ggml_backend_t backend,
    std::unordered_map<std::string, std::vector<uint16_t>>& merged_storage,
    std::unordered_map<std::string, const void*>& merged_ptrs
) {
    int alpha_cfg = adapter_read_alpha(cfg_dir.c_str());

    std::map<std::string, const STEntry*> a_map, b_map, ds_map;
    std::map<std::string, float> alpha_map;
    adapter_read_alpha_pattern(cfg_dir.c_str(), alpha_map);

    for (const auto& e : st.entries) {
        const char* alpha_suffix = ".alpha";
        size_t slen = strlen(alpha_suffix);
        if (e.name.size() > slen &&
            e.name.compare(e.name.size() - slen, slen, alpha_suffix) == 0 &&
            e.dtype == "F32" && e.n_dims == 0) {
            std::string fake_key = e.name.substr(0, e.name.size() - slen) + ".lora_.x";
            std::string base = lora_base_name(fake_key);
            if (!base.empty()) {
                float val = 0.0f;
                memcpy(&val, st_data(st, e), sizeof(float));
                alpha_map[base] = val;
            }
            continue;
        }

        std::string base = lora_base_name(e.name);
        if (base.empty()) continue;
        if (lora_is_a(e.name)) a_map[base] = &e;
        else if (lora_is_b(e.name)) b_map[base] = &e;
        else if (lora_is_magnitude(e.name)) ds_map[base] = &e;
    }

    fprintf(stderr, "[Adapter-TRT] LoRA: found %zu A/B pairs (%zu with DoRA magnitude)\n",
            a_map.size(), ds_map.size());
    int merged = 0, skipped = 0, dora_merged = 0;

    // Use BF16 precision rounding (same as adapter-runtime.h)
    ggml_type round_type = GGML_TYPE_BF16;

    for (const auto& [gguf_name, ea] : a_map) {
        auto it = b_map.find(gguf_name);
        if (it == b_map.end()) { skipped++; continue; }
        const STEntry* eb = it->second;

        std::string trt_name = find_trt_weight_name(*ctx, gguf_name);
        if (trt_name.empty()) { skipped++; continue; }

        const auto& base_data = ctx->base_weights.at(trt_name);

        int64_t rank     = ea->shape[0];
        int64_t in_feat  = ea->shape[1];
        int64_t out_feat = eb->shape[0];

        if (eb->shape[1] != rank) {
            fprintf(stderr, "[Adapter-TRT] Rank mismatch for %s\n", gguf_name.c_str());
            skipped++; continue;
        }

        int64_t nel = in_feat * out_feat;
        if (nel != (int64_t)base_data.size()) {
            fprintf(stderr, "[Adapter-TRT] Size mismatch for %s: LoRA %lldx%lld=%lld vs base %zu\n",
                    gguf_name.c_str(), (long long)out_feat, (long long)in_feat,
                    (long long)nel, base_data.size());
            skipped++; continue;
        }

        float alpha;
        auto alpha_it = alpha_map.find(gguf_name);
        if (alpha_it != alpha_map.end())   alpha = alpha_it->second;
        else if (alpha_cfg > 0)            alpha = (float)alpha_cfg;
        else                               alpha = (float)rank;

        float scaling = (alpha / (float)rank) * adapter_scale;
        float g_scale = adapter_group_scale_for(
            g_hotstep_params.adapter_group_scales,
            adapter_determine_group(gguf_name));
        scaling *= g_scale;

        // PEFT DoRA: optional magnitude vector, one value per output row.
        // Delta keeps only alpha/rank; the user strength blends the per-row
        // rescale inside trt_merge_delta_dora (same as adapter-merge.h).
        const float* ds_ptr = nullptr;
        std::vector<float> ds_f32;
        float effective_user_scale = 1.0f;
        auto ds_it = ds_map.find(gguf_name);
        if (ds_it != ds_map.end()) {
            const STEntry* eds = ds_it->second;
            int64_t ds_nel = 1;
            for (int di = 0; di < eds->n_dims; di++) ds_nel *= eds->shape[di];
            if (ds_nel != out_feat) {
                fprintf(stderr, "[Adapter-TRT] WARNING: lora_magnitude_vector nel %lld != out rows %lld for %s, merging without DoRA rescale\n",
                        (long long)ds_nel, (long long)out_feat, gguf_name.c_str());
            } else {
                ds_f32.resize((size_t)ds_nel);
                if (!adapter_to_f32(st_data(st, *eds), ds_f32.data(), ds_nel, eds->dtype)) {
                    fprintf(stderr, "[Adapter-TRT] WARNING: unsupported lora_magnitude_vector dtype %s for %s, merging without DoRA rescale\n",
                            eds->dtype.c_str(), gguf_name.c_str());
                } else {
                    ds_ptr = ds_f32.data();
                    scaling = alpha / (float)rank;
                    effective_user_scale = adapter_scale * g_scale;
                }
            }
        }

        // Load A and B to fp32
        int64_t a_nel = rank * in_feat;
        int64_t b_nel = out_feat * rank;
        std::vector<float> a_f32((size_t)a_nel);
        std::vector<float> b_f32((size_t)b_nel);

        if (!adapter_to_f32(st_data(st, *ea), a_f32.data(), a_nel, ea->dtype)) { skipped++; continue; }
        if (!adapter_to_f32(st_data(st, *eb), b_f32.data(), b_nel, eb->dtype)) { skipped++; continue; }

        // GPU-accelerated delta computation (same graph as adapter-runtime.h)
        auto build = [&](struct ggml_context* gctx) {
            struct ggml_tensor* ta    = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, in_feat, rank);
            struct ggml_tensor* tb    = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, rank, out_feat);
            struct ggml_tensor* ta_br = ggml_cast(gctx, ggml_cast(gctx, ta, round_type), GGML_TYPE_F32);
            struct ggml_tensor* tb_br = ggml_cast(gctx, ggml_cast(gctx, tb, round_type), GGML_TYPE_F32);
            struct ggml_tensor* ta_t  = ggml_cont(gctx, ggml_transpose(gctx, ta_br));
            struct ggml_tensor* tdelta = ggml_scale(gctx, ggml_mul_mat(gctx, ta_t, tb_br), scaling);
            adapter_delta_build db;
            db.tdelta = tdelta;
            db.upload = [=]() {
                ggml_backend_tensor_set(ta, a_f32.data(), 0, (size_t)a_nel * sizeof(float));
                ggml_backend_tensor_set(tb, b_f32.data(), 0, (size_t)b_nel * sizeof(float));
            };
            return db;
        };

        std::vector<float> delta_f32;
        if (!adapter_compute_delta(build, in_feat, out_feat, backend, delta_f32)) {
            fprintf(stderr, "[Adapter-TRT] WARNING: GPU delta compute failed for %s\n", gguf_name.c_str());
            skipped++; continue;
        }

        if (ds_ptr) {
            trt_merge_delta_dora(*ctx, trt_name, base_data, delta_f32.data(), ds_ptr,
                                 in_feat, out_feat, effective_user_scale,
                                 merged_storage, merged_ptrs);
            dora_merged++;
        } else {
            trt_merge_delta(*ctx, trt_name, base_data, delta_f32.data(),
                            in_feat, out_feat, 1.0f,  // scaling already baked into delta
                            merged_storage, merged_ptrs);
        }
        merged++;
    }

    fprintf(stderr, "[Adapter-TRT] LoRA: %d merged (%d with DoRA), %d skipped\n", merged, dora_merged, skipped);
    return merged;
}

// ── LoKr merge path (batched GPU computation) ───────────────────────────────
// All LoKr Kronecker products are built as subgraphs in a single GGML context,
// allocated in one GPU buffer, computed in one graph dispatch, and downloaded
// in one pass. This avoids 359× separate cudaMalloc/cudaFree cycles that
// caused ~14s of CUDA driver overhead.
static int adapter_trt_apply_lokr(
    DitTrt* ctx,
    const STFile& st,
    float adapter_scale,
    ggml_backend_t backend,
    std::unordered_map<std::string, std::vector<uint16_t>>& merged_storage,
    std::unordered_map<std::string, const void*>& merged_ptrs
) {
    struct LoKrEntry {
        const STEntry* w1 = nullptr;
        const STEntry* w2 = nullptr;
        const STEntry* w2_a = nullptr;
        const STEntry* w2_b = nullptr;
        const STEntry* alpha = nullptr;
    };

    std::map<std::string, LoKrEntry> modules;
    for (const auto& e : st.entries) {
        std::string prefix, suffix;
        if (!adapter_split_suffix(e.name, &prefix, &suffix)) continue;
        if (prefix.compare(0, 8, "lycoris_") != 0) continue;
        LoKrEntry& m = modules[prefix];
        if (suffix == "lokr_w1")         m.w1 = &e;
        else if (suffix == "lokr_w2")    m.w2 = &e;
        else if (suffix == "lokr_w2_a")  m.w2_a = &e;
        else if (suffix == "lokr_w2_b")  m.w2_b = &e;
        else if (suffix == "alpha")      m.alpha = &e;
    }

    std::unordered_map<std::string, std::string> name_map = lokr_build_trt_reverse_map(*ctx);
    int lokr_dim = adapter_read_lokr_dim(st);
    int skipped = 0;

    fprintf(stderr, "[Adapter-TRT] LoKr: found %zu modules, %zu TRT name mappings\n",
            modules.size(), name_map.size());

    // ── Phase 1: Validate and collect all mergeable entries ──────────────
    struct BatchEntry {
        std::string trt_name;
        int64_t a, b, c, d, r;
        int64_t in_feat, out_feat;
        float scaling;
        bool has_factor;
        bool is_transposed;

        // Host factor data (kept alive for upload)
        std::vector<float> w1_f32;
        std::vector<float> w2a_f32, w2b_f32, w2_f32;
        int64_t w1_nel, w2a_nel, w2b_nel, w2_nel;

        // Graph tensor pointers (filled in Phase 2)
        struct ggml_tensor* tw1     = nullptr;
        struct ggml_tensor* tw2a    = nullptr;
        struct ggml_tensor* tw2b    = nullptr;
        struct ggml_tensor* tw2_src = nullptr;
        struct ggml_tensor* tbase   = nullptr;   // base bf16 weight
        struct ggml_tensor* tmerged = nullptr;    // final merged bf16 output
    };

    std::vector<BatchEntry> batch;
    batch.reserve(modules.size());

    for (const auto& [lyc_prefix, m] : modules) {
        bool has_factor = (m.w2_a && m.w2_b);
        bool has_mono   = (m.w2 != nullptr);
        if (!m.w1 || !m.alpha || has_factor == has_mono) {
            fprintf(stderr, "[Adapter-TRT] WARNING: incomplete LoKr module %s, skipping\n",
                    lyc_prefix.c_str());
            skipped++; continue;
        }

        auto nm_it = name_map.find(lyc_prefix);
        if (nm_it == name_map.end()) {
            fprintf(stderr, "[Adapter-TRT] WARNING: no TRT weight for %s, skipping\n",
                    lyc_prefix.c_str());
            skipped++; continue;
        }
        const std::string& trt_name = nm_it->second;

        if (ctx->base_weights.count(trt_name) == 0) {
            skipped++; continue;
        }
        const auto& base_data = ctx->base_weights.at(trt_name);

        BatchEntry e;
        e.trt_name = trt_name;
        e.a = m.w1->shape[0]; e.b = m.w1->shape[1];
        e.has_factor = has_factor;
        e.is_transposed = ctx->weights_transposed.count(trt_name) > 0;

        if (has_factor) {
            e.c = m.w2_a->shape[0]; e.r = m.w2_a->shape[1]; e.d = m.w2_b->shape[1];
            if (e.r != m.w2_b->shape[0]) {
                fprintf(stderr, "[Adapter-TRT] WARNING: LoKr rank mismatch for %s\n",
                        lyc_prefix.c_str());
                skipped++; continue;
            }
        } else {
            e.c = m.w2->shape[0]; e.d = m.w2->shape[1];
            if (lokr_dim <= 0) {
                fprintf(stderr, "[Adapter-TRT] WARNING: monolithic LoKr %s needs lokr_config.linear_dim\n",
                        lyc_prefix.c_str());
                skipped++; continue;
            }
            e.r = lokr_dim;
        }

        e.out_feat = e.a * e.c;
        e.in_feat  = e.b * e.d;
        int64_t nel = e.out_feat * e.in_feat;
        if (nel != (int64_t)base_data.size()) {
            fprintf(stderr, "[Adapter-TRT] WARNING: LoKr size mismatch for %s: kron=%lldx%lld=%lld vs base=%zu\n",
                    lyc_prefix.c_str(), (long long)e.out_feat, (long long)e.in_feat,
                    (long long)nel, base_data.size());
            skipped++; continue;
        }

        // Read alpha
        float alpha_val = 0.0f;
        if (!adapter_to_f32(st_data(st, *m.alpha), &alpha_val, 1, m.alpha->dtype)) {
            skipped++; continue;
        }
        e.scaling = (alpha_val / (float)e.r) * adapter_scale;

        // Group scaling
        std::string gguf_name = trt_name;
        if (gguf_name.compare(0, 4, "dit.") == 0) {
            gguf_name = "decoder." + gguf_name.substr(4);
        }
        float g_scale = adapter_group_scale_for(
            g_hotstep_params.adapter_group_scales,
            adapter_determine_group(gguf_name));
        e.scaling *= g_scale;

        // Load w1
        e.w1_nel = e.a * e.b;
        e.w1_f32.resize((size_t)e.w1_nel);
        if (!adapter_to_f32(st_data(st, *m.w1), e.w1_f32.data(), e.w1_nel, m.w1->dtype)) {
            skipped++; continue;
        }

        // Load w2 factors
        if (has_factor) {
            e.w2a_nel = e.c * e.r; e.w2b_nel = e.r * e.d;
            e.w2a_f32.resize((size_t)e.w2a_nel); e.w2b_f32.resize((size_t)e.w2b_nel);
            if (!adapter_to_f32(st_data(st, *m.w2_a), e.w2a_f32.data(), e.w2a_nel, m.w2_a->dtype) ||
                !adapter_to_f32(st_data(st, *m.w2_b), e.w2b_f32.data(), e.w2b_nel, m.w2_b->dtype)) {
                skipped++; continue;
            }
        } else {
            e.w2_nel = e.c * e.d; e.w2_f32.resize((size_t)e.w2_nel);
            if (!adapter_to_f32(st_data(st, *m.w2), e.w2_f32.data(), e.w2_nel, m.w2->dtype)) {
                skipped++; continue;
            }
        }

        batch.push_back(std::move(e));
    }

    if (batch.empty()) {
        fprintf(stderr, "[Adapter-TRT] LoKr: 0 merged, %d skipped\n", skipped);
        return 0;
    }

    fprintf(stderr, "[Adapter-TRT] LoKr: %zu valid entries, computing in micro-batches...\n", batch.size());

    // ── Phase 2-4: Process in micro-batches with GPU-side merge ─────────
    // Each chunk: build Kronecker delta + merge (base+delta→bf16) ON GPU,
    // then download the final bf16 result. No CPU merge loop needed.
    const size_t CHUNK_SIZE = 32;
    auto t_total = std::chrono::steady_clock::now();
    int merged = 0;

    for (size_t chunk_start = 0; chunk_start < batch.size(); chunk_start += CHUNK_SIZE) {
        size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, batch.size());
        size_t chunk_n   = chunk_end - chunk_start;

        // Extra tensors per entry for GPU merge: tbase(bf16) + cast_f32 + add + cast_bf16
        // + possible transpose = ~18 tensors total per entry
        size_t max_tensors = chunk_n * 20 + 32;
        size_t max_nodes   = chunk_n * 16 + 32;
        size_t meta = ggml_tensor_overhead() * max_tensors +
                      ggml_graph_overhead_custom(max_nodes, false) +
                      64 * 1024;

        struct ggml_init_params gparams = { meta, NULL, true };
        struct ggml_context* gctx = ggml_init(gparams);
        if (!gctx) {
            fprintf(stderr, "[Adapter-TRT] FATAL: failed to init chunk graph context\n");
            break;
        }

        struct ggml_cgraph* graph = ggml_new_graph_custom(gctx, max_nodes, false);

        for (size_t i = chunk_start; i < chunk_end; i++) {
            auto& e = batch[i];

            // ── Kronecker delta subgraph (same as before) ──
            e.tw1 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, e.b, e.a);

            struct ggml_tensor* tw2;
            if (e.has_factor) {
                e.tw2a = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, e.r, e.c);
                e.tw2b = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, e.d, e.r);
                tw2 = ggml_mul_mat(gctx, ggml_cont(gctx, ggml_transpose(gctx, e.tw2b)), e.tw2a);
            } else {
                e.tw2_src = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, e.d, e.c);
                tw2 = e.tw2_src;
            }

            struct ggml_tensor* tw1_s    = ggml_scale(gctx, e.tw1, e.scaling);
            struct ggml_tensor* tw1_flat = ggml_reshape_2d(gctx, tw1_s, 1, e.a * e.b);
            struct ggml_tensor* tw2_flat = ggml_reshape_2d(gctx, tw2, 1, e.c * e.d);
            struct ggml_tensor* touter   = ggml_mul_mat(gctx, tw1_flat, tw2_flat);
            struct ggml_tensor* t4d      = ggml_reshape_4d(gctx, touter, e.b, e.a, e.d, e.c);
            struct ggml_tensor* tperm    = ggml_permute(gctx, t4d, 1, 3, 0, 2);
            // tdelta: ne=(in_feat, out_feat) = [out, in] row-major
            struct ggml_tensor* tdelta   = ggml_reshape_2d(gctx, ggml_cont(gctx, tperm), e.b * e.d, e.a * e.c);

            // ── GPU merge: base_bf16 + delta → merged_bf16 ──
            // Base weight is bf16 from TRT cache. Upload it, cast to f32, add delta, cast back.
            //
            // Layout handling:
            // - tdelta is GGML [in_feat, out_feat] = row-major [out, in]
            // - Non-transposed base: also [out, in] → element-wise add
            // - Transposed base: [in, out] → transpose delta before adding
            int64_t nel = e.in_feat * e.out_feat;

            if (e.is_transposed) {
                // Base is [in, out] = GGML ne=(out_feat, in_feat)
                e.tbase = ggml_new_tensor_2d(gctx, GGML_TYPE_BF16, e.out_feat, e.in_feat);
                struct ggml_tensor* tbase_f32 = ggml_cast(gctx, e.tbase, GGML_TYPE_F32);
                // Transpose delta from [in_feat, out_feat] to [out_feat, in_feat]
                struct ggml_tensor* tdelta_t = ggml_cont(gctx, ggml_transpose(gctx, tdelta));
                struct ggml_tensor* tsum = ggml_add(gctx, tbase_f32, tdelta_t);
                e.tmerged = ggml_cast(gctx, tsum, GGML_TYPE_BF16);
            } else {
                // Base is [out, in] = GGML ne=(in_feat, out_feat) — same as delta
                e.tbase = ggml_new_tensor_2d(gctx, GGML_TYPE_BF16, e.in_feat, e.out_feat);
                struct ggml_tensor* tbase_f32 = ggml_cast(gctx, e.tbase, GGML_TYPE_F32);
                struct ggml_tensor* tsum = ggml_add(gctx, tbase_f32, tdelta);
                e.tmerged = ggml_cast(gctx, tsum, GGML_TYPE_BF16);
            }

            ggml_build_forward_expand(graph, e.tmerged);
        }

        // Allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
        if (!buf) {
            fprintf(stderr, "[Adapter-TRT] FATAL: failed to allocate chunk GPU buffer\n");
            ggml_free(gctx);
            break;
        }

        // Upload: adapter factors + base weights
        for (size_t i = chunk_start; i < chunk_end; i++) {
            auto& e = batch[i];
            ggml_backend_tensor_set(e.tw1, e.w1_f32.data(), 0, (size_t)e.w1_nel * sizeof(float));
            if (e.has_factor) {
                ggml_backend_tensor_set(e.tw2a, e.w2a_f32.data(), 0, (size_t)e.w2a_nel * sizeof(float));
                ggml_backend_tensor_set(e.tw2b, e.w2b_f32.data(), 0, (size_t)e.w2b_nel * sizeof(float));
            } else {
                ggml_backend_tensor_set(e.tw2_src, e.w2_f32.data(), 0, (size_t)e.w2_nel * sizeof(float));
            }
            // Upload base bf16 weight
            const auto& base_data = ctx->base_weights.at(e.trt_name);
            ggml_backend_tensor_set(e.tbase, base_data.data(), 0,
                                    base_data.size() * sizeof(uint16_t));
        }

        // Compute (delta + merge all on GPU)
        ggml_backend_graph_compute(backend, graph);

        // Download merged bf16 results directly
        for (size_t i = chunk_start; i < chunk_end; i++) {
            auto& e = batch[i];
            int64_t nel = e.in_feat * e.out_feat;
            std::vector<uint16_t> merged_bf16((size_t)nel);
            ggml_backend_tensor_get(e.tmerged, merged_bf16.data(), 0,
                                    (size_t)nel * sizeof(uint16_t));

            merged_storage[e.trt_name] = std::move(merged_bf16);
            merged_ptrs[e.trt_name] = merged_storage[e.trt_name].data();
            merged++;
        }

        // Free this chunk's GPU buffer
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
    }

    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_total).count();
    fprintf(stderr, "[Adapter-TRT] Batched GPU merge: %lld ms (%d weights, %zu-entry chunks)\n",
            (long long)total_ms, merged, CHUNK_SIZE);
    fflush(stderr);

    fprintf(stderr, "[Adapter-TRT] LoKr: %d merged, %d skipped\n", merged, skipped);
    return merged;
}

// ── Main entry point ────────────────────────────────────────────────────────
// Apply a LoRA or LoKr adapter to the TRT engine.
//
// adapter_path: directory containing adapter_model.safetensors + adapter_config.json,
//               or a single .safetensors file
// adapter_scale: user-provided scale multiplier (typically 1.0)
//
// Returns refit time in ms, or -1 on failure.
static int64_t adapter_trt_apply(
    DitTrt* ctx,
    const char* adapter_path,
    float adapter_scale
) {
    // Resolve safetensors file path
    std::string st_path;
    std::string cfg_dir;
    struct stat st_stat;
    if (stat(adapter_path, &st_stat) != 0) {
        fprintf(stderr, "[Adapter-TRT] Path not found: %s\n", adapter_path);
        return -1;
    }
    if (S_ISDIR(st_stat.st_mode)) {
        st_path = std::string(adapter_path) + "/adapter_model.safetensors";
        cfg_dir = adapter_path;
    } else {
        st_path = adapter_path;
        size_t slash = st_path.find_last_of("/\\");
        cfg_dir = (slash != std::string::npos) ? st_path.substr(0, slash) : ".";
    }

    // Open safetensors
    STFile st;
    if (!st_open(&st, st_path.c_str())) {
        fprintf(stderr, "[Adapter-TRT] Failed to open: %s\n", st_path.c_str());
        return -1;
    }
    fprintf(stderr, "[Adapter-TRT] Loading adapter: %s\n", st_path.c_str());

    // Get GGML GPU backend for delta computation
    ggml_backend_t backend = adapter_trt_get_backend();
    if (!backend) {
        fprintf(stderr, "[Adapter-TRT] FATAL: no compute backend available\n");
        st_close(&st);
        return -1;
    }
    fprintf(stderr, "[Adapter-TRT] Using %s backend for delta computation\n",
            ggml_backend_name(backend));

    // Compute merged weights
    std::unordered_map<std::string, std::vector<uint16_t>> merged_storage;
    std::unordered_map<std::string, const void*> merged_ptrs;
    int merged = 0;

    if (adapter_detect_lokr(st)) {
        merged = adapter_trt_apply_lokr(ctx, st, adapter_scale, backend,
                                        merged_storage, merged_ptrs);
    } else {
        merged = adapter_trt_apply_lora(ctx, st, cfg_dir, adapter_scale, backend,
                                        merged_storage, merged_ptrs);
    }

    st_close(&st);

    fprintf(stderr, "[Adapter-TRT] Total: %d merged weights ready for refit\n", merged);

    if (merged == 0) {
        fprintf(stderr, "[Adapter-TRT] No weights matched TRT engine — adapter has no effect\n");
        return 0;
    }

    // Apply via TRT IRefitter
    return dit_trt_refit_adapter(ctx, adapter_path, merged_ptrs);
}

// Revert to base model (remove adapter)
static int64_t adapter_trt_revert(DitTrt* ctx) {
    return dit_trt_refit_base(ctx);
}

#endif // HOT_STEP_TRT
