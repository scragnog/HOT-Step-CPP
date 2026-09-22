#pragma once

// Bounded AI Toolkit YuE2 fused-LoRA safetensors export seam.
//
// The reference file contains two expert namespaces, 28 layers, four fused
// sites per layer, and one A/B pair per site: 448 BF16 tensors in total.
// Optimizer state and resume data are intentionally outside this format.

#include "st-write.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

struct Yue2AitkF32Matrix {
    std::string  name;
    int64_t      rows = 0;
    int64_t      cols = 0;
    const float *data = nullptr;
};

namespace yue2_aitk_adapter_io_detail {

inline bool site_dims(const char *site, int64_t *input, int64_t *output) {
    if (!site || !input || !output) return false;
    const std::string name(site);
    if (name == "self_attn.qkv_proj") { *input = 2048; *output = 4096; return true; }
    if (name == "self_attn.o_proj")   { *input = 2048; *output = 2048; return true; }
    if (name == "mlp.gate_up_proj")   { *input = 2048; *output = 12288; return true; }
    if (name == "mlp.down_proj")      { *input = 6144; *output = 2048; return true; }
    return false;
}

inline bool output_exists(const char *path) {
    if (!path || !*path) return true;
    FILE *f = hs_fopen(path, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

inline std::string alpha_string(float alpha) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(alpha));
    return std::string(buf);
}

inline std::string unique_sibling(const char *path) {
    static std::atomic<uint64_t> serial{0};
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return std::string(path) + ".__aitk_writing__" + std::to_string(now) +
           "_" + std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
}

// Publish without replacing an existing destination. The unique sibling was
// fully written first, so a failed publish leaves no partial output.
inline bool publish_no_replace(const std::string &temporary, const char *path) {
#ifdef _WIN32
    return MoveFileExW(hs_widen(temporary).c_str(), hs_widen(path).c_str(),
                       MOVEFILE_WRITE_THROUGH) != 0;
#else
    if (::link(temporary.c_str(), path) != 0) return false;
    return ::unlink(temporary.c_str()) == 0;
#endif
}

} // namespace yue2_aitk_adapter_io_detail

// Write a complete reference-compatible fused adapter. The caller owns every
// data pointer for the duration of this call. Output is refused if it exists.
inline bool yue2_aitk_write_fused_lora(
    const std::vector<Yue2AitkF32Matrix> &factors,
    int64_t                              rank,
    float                                alpha,
    int64_t                              steps,
    const char *                         output_path) {
    using namespace yue2_aitk_adapter_io_detail;

    constexpr int kLayers = 28;
    constexpr int kSites = 4;
    constexpr size_t kTensorCount = 2u * kLayers * kSites * 2u;
    if (!output_path || !*output_path || output_exists(output_path) ||
        factors.size() != kTensorCount || rank <= 0 || rank > 65536 ||
        steps < 0 || !std::isfinite(alpha) || alpha <= 0.0f) {
        return false;
    }

    std::unordered_set<std::string> seen;
    seen.reserve(factors.size());
    std::vector<STWTensor> tensors;
    tensors.reserve(factors.size());

    for (const char *expert : {"diffusion_model", "text_encoders"}) {
        for (int layer = 0; layer < kLayers; ++layer) {
            for (const char *site : {"self_attn.qkv_proj", "self_attn.o_proj",
                                     "mlp.gate_up_proj", "mlp.down_proj"}) {
                const std::string prefix = std::string(expert) + ".model.layers." +
                                            std::to_string(layer) + "." + site + ".lora_";
                for (const char *factor : {"A.weight", "B.weight"}) {
                    const std::string name = prefix + factor;
                    size_t found = factors.size();
                    for (size_t i = 0; i < factors.size(); ++i) {
                        if (factors[i].name == name) {
                            if (found != factors.size()) return false;
                            found = i;
                        }
                    }
                    if (found == factors.size() || !seen.insert(name).second) return false;
                    const Yue2AitkF32Matrix &src = factors[found];
                    const bool is_a = factor[0] == 'A';
                    int64_t site_input = 0;
                    int64_t site_output = 0;
                    if (!site_dims(site, &site_input, &site_output)) return false;
                    const int64_t expected_rows = is_a ? rank : site_output;
                    const int64_t expected_cols = is_a ? site_input : rank;
                    if (src.rows != expected_rows || src.cols != expected_cols ||
                        src.rows <= 0 || src.cols <= 0 || !src.data) {
                        return false;
                    }
                    tensors.push_back({src.name, {src.rows, src.cols}, src.data, -1});
                }
            }
        }
    }

    if (seen.size() != kTensorCount || tensors.size() != kTensorCount) return false;
    const std::vector<std::pair<std::string, std::string>> metadata = {
        {"format", "yue2-aitk-fused-lora-v1"},
        {"rank", std::to_string(rank)},
        {"alpha", alpha_string(alpha)},
        {"steps", std::to_string(steps)},
    };
    const std::string temporary = unique_sibling(output_path);
    if (!st_write_file(temporary.c_str(), tensors, metadata, STW_BF16)) {
        hs_remove(temporary);
        return false;
    }
    const bool published = publish_no_replace(temporary, output_path);
    if (!published) hs_remove(temporary);
    return published;
}

// The LoKr twin. No reference trainer defines a fused LoKr layout, so this
// writes the trainer's own slot inventory verbatim (`<expert>.model.layers.N.
// <site>.lokr_w1` / `.lokr_w2` / `.lokr_w2_a` / `.lokr_w2_b`) so a checkpoint
// directory keeps its adapter.safetensors either way. The loadable files are
// the native split ones (yue2-aitk-native-adapter-io.h).
inline bool yue2_aitk_write_fused_lokr(
    const std::vector<Yue2AitkF32Matrix> &factors,
    int                                  dim,
    int                                  factor,
    float                                alpha,
    int64_t                              steps,
    const char *                         output_path) {
    using namespace yue2_aitk_adapter_io_detail;
    if (!output_path || !*output_path || output_exists(output_path) || factors.empty() ||
        dim <= 0 || factor == 0 || steps < 0 || !std::isfinite(alpha) || alpha <= 0.0f) return false;
    std::unordered_set<std::string> seen;
    std::vector<STWTensor> tensors;
    tensors.reserve(factors.size());
    for (const Yue2AitkF32Matrix &f : factors) {
        if (!f.data || f.rows <= 0 || f.cols <= 0 || f.name.find(".lokr_w") == std::string::npos ||
            !seen.insert(f.name).second) return false;
        tensors.push_back({f.name, {f.rows, f.cols}, f.data, -1});
    }
    const std::vector<std::pair<std::string, std::string>> metadata = {
        {"format", "yue2-aitk-fused-lokr-v1"},
        {"lokr_dim", std::to_string(dim)},
        {"lokr_factor", std::to_string(factor)},
        {"alpha", alpha_string(alpha)},
        {"steps", std::to_string(steps)},
    };
    const std::string temporary = unique_sibling(output_path);
    if (!st_write_file(temporary.c_str(), tensors, metadata, STW_BF16)) {
        hs_remove(temporary);
        return false;
    }
    const bool published = publish_no_replace(temporary, output_path);
    if (!published) hs_remove(temporary);
    return published;
}
