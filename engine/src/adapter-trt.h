#pragma once
// adapter-trt.h: LoRA adapter merge for TRT engine via IRefitter
//
// Reads LoRA safetensors (PEFT or ComfyUI format), maps adapter keys to TRT
// weight names, computes merged weights (base + scale * delta) on CPU in fp32,
// casts back to bf16, and calls dit_trt_refit_adapter() to hot-swap weights.
//
// Unlike the GGML merge path (adapter-merge.h) which operates on PendingCopy
// staging buffers before GPU upload, this operates on the live TRT engine
// via the IRefitter API — enabling adapter switching without engine reload.
//
// Only LoRA is supported (not LoKr). LoKr adapters are rare for this model
// and would require significantly more complex delta computation.

#ifdef HOT_STEP_TRT

#include "adapter-merge.h"   // lora_base_name(), adapter_read_alpha(), etc.
#include "dit-trt.h"         // DitTrt, dit_trt_refit_adapter/base
#include "hot-step-params.h" // adapter_group_scales
#include "safetensors.h"     // STFile, st_open, st_data

#include <cmath>
#include <cstdio>
#include <cstring>
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

// CPU matrix multiply: C[M,N] = B[M,K] @ A[K,N] (row-major)
// Small enough for adapter LoRA factors (rank is typically 4-64).
static void matmul_f32(const float* B, const float* A, float* C,
                       int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += B[m * K + k] * A[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Build the mapping from TRT weight name → base_weights key.
// TRT weight names from dynamo export use the module path directly, e.g.:
//   "dit.layers.0.self_attn.q_proj.weight"
// The GGML GGUF convention uses "decoder." prefix:
//   "decoder.layers.0.self_attn.q_proj.weight"
// We need to try both forms when looking up in the base_weights cache.
//
// Returns the matching key from base_weights, or "" if not found.
static std::string find_trt_weight_name(
    const DitTrt& ctx,
    const std::string& gguf_name  // e.g. "decoder.layers.0.self_attn.q_proj.weight"
) {
    // Try exact match first
    if (ctx.base_weights.count(gguf_name)) {
        return gguf_name;
    }

    // Try without "decoder." prefix (TRT/dynamo uses "dit." prefix)
    if (gguf_name.compare(0, 8, "decoder.") == 0) {
        std::string dit_name = "dit." + gguf_name.substr(8);
        if (ctx.base_weights.count(dit_name)) {
            return dit_name;
        }
    }

    // Try adding "dit." prefix to raw path
    std::string dit_prefixed = "dit." + gguf_name;
    if (ctx.base_weights.count(dit_prefixed)) {
        return dit_prefixed;
    }

    return "";
}

// Apply a LoRA adapter to the TRT engine.
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
        // config dir is parent of the file
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

    // Read alpha from config
    int alpha_cfg = adapter_read_alpha(cfg_dir.c_str());

    // Group lora_A and lora_B by base tensor name (reuses GGML key mapping)
    std::map<std::string, const STEntry*> a_map, b_map;
    std::map<std::string, float> alpha_map;

    for (const auto& e : st.entries) {
        // Per-tensor alpha
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
    }

    fprintf(stderr, "[Adapter-TRT] Found %zu LoRA pairs\n", a_map.size());

    if (a_map.empty()) {
        fprintf(stderr, "[Adapter-TRT] No LoRA pairs found in adapter\n");
        st_close(&st);
        return -1;
    }

    // Compute merged weights for each LoRA pair
    std::unordered_map<std::string, std::vector<uint16_t>> merged_storage;
    std::unordered_map<std::string, const void*> merged_ptrs;
    int merged = 0, skipped = 0;

    for (const auto& [gguf_name, ea] : a_map) {
        auto it = b_map.find(gguf_name);
        if (it == b_map.end()) {
            skipped++;
            continue;
        }
        const STEntry* eb = it->second;

        // Find this weight in TRT base cache
        std::string trt_name = find_trt_weight_name(*ctx, gguf_name);
        if (trt_name.empty()) {
            // Not a refittable weight — skip silently (many LoRA targets
            // like norm weights may not be in the TRT engine)
            skipped++;
            continue;
        }

        const auto& base_data = ctx->base_weights.at(trt_name);

        // LoRA shapes (safetensors row-major):
        //   A: [rank, in_features]
        //   B: [out_features, rank]
        int64_t rank     = ea->shape[0];
        int64_t in_feat  = ea->shape[1];
        int64_t out_feat = eb->shape[0];

        if (eb->shape[1] != rank) {
            fprintf(stderr, "[Adapter-TRT] Rank mismatch for %s: A=%lld B=%lld\n",
                    gguf_name.c_str(), (long long)rank, (long long)eb->shape[1]);
            skipped++;
            continue;
        }

        int64_t nel = in_feat * out_feat;
        if (nel != (int64_t)base_data.size()) {
            fprintf(stderr, "[Adapter-TRT] Size mismatch for %s: LoRA %lldx%lld=%lld vs base %zu\n",
                    gguf_name.c_str(), (long long)out_feat, (long long)in_feat,
                    (long long)nel, base_data.size());
            skipped++;
            continue;
        }

        // Compute scaling: (alpha / rank) * adapter_scale * group_scale
        float alpha;
        auto alpha_it = alpha_map.find(gguf_name);
        if (alpha_it != alpha_map.end()) {
            alpha = alpha_it->second;
        } else if (alpha_cfg > 0) {
            alpha = (float)alpha_cfg;
        } else {
            alpha = (float)rank;
        }
        float scaling = (alpha / (float)rank) * adapter_scale;

        // HOT-Step per-group scaling
        float g_scale = adapter_group_scale_for(
            g_hotstep_params.adapter_group_scales,
            adapter_determine_group(gguf_name));
        scaling *= g_scale;

        // Load A and B to fp32
        int64_t a_nel = rank * in_feat;
        int64_t b_nel = out_feat * rank;
        std::vector<float> a_f32((size_t)a_nel);
        std::vector<float> b_f32((size_t)b_nel);

        if (!adapter_to_f32(st_data(st, *ea), a_f32.data(), a_nel, ea->dtype)) {
            skipped++;
            continue;
        }
        if (!adapter_to_f32(st_data(st, *eb), b_f32.data(), b_nel, eb->dtype)) {
            skipped++;
            continue;
        }

        // Compute delta = B @ A  [out_feat, in_feat]
        std::vector<float> delta((size_t)nel);
        matmul_f32(b_f32.data(), a_f32.data(), delta.data(),
                   (int)out_feat, (int)rank, (int)in_feat);

        // Merge: base_bf16 + scaling * delta → bf16
        // Convert base from bf16 to fp32, add scaled delta, convert back
        std::vector<uint16_t> merged_bf16((size_t)nel);
        for (int64_t j = 0; j < nel; j++) {
            float base_val = trt_bf16_to_fp32(base_data[j]);
            float merged_val = base_val + scaling * delta[j];
            merged_bf16[j] = trt_fp32_to_bf16(merged_val);
        }

        merged_storage[trt_name] = std::move(merged_bf16);
        merged_ptrs[trt_name] = merged_storage[trt_name].data();
        merged++;
    }

    st_close(&st);

    fprintf(stderr, "[Adapter-TRT] Computed %d merged weights (skipped %d)\n",
            merged, skipped);

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
