#pragma once

// Native export bridge for the AI Toolkit YuE2 fused adapter.
//
// The AI Toolkit has two fused linears (qkv_proj and gate_up_proj). HOT-Step's
// loader consumes separate native GGUF sites. This bridge writes two ordinary
// safetensors files, one AR and one NAR, without expanding rank: split B rows
// are copied into each destination site and the shared A is duplicated.

#include "yue2-aitk-adapter-io.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using Yue2AitkNativeMatrix = Yue2AitkF32Matrix;

namespace yue2_aitk_native_detail {
inline bool exists(const char * path) {
    if (!path || !*path) return true;
    FILE * f = hs_fopen(path, "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

inline std::string temp_path(const char * path) {
    static std::atomic<uint64_t> serial{0};
    const auto now = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    return std::string(path) + ".__native_aitk_writing__" + std::to_string(now) +
           "_" + std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
}

inline bool publish_new(const std::string & temp, const char * path) {
#ifdef _WIN32
    return MoveFileExW(hs_widen(temp).c_str(), hs_widen(path).c_str(), MOVEFILE_WRITE_THROUGH) != 0;
#else
    if (::link(temp.c_str(), path) != 0) return false;
    return ::unlink(temp.c_str()) == 0;
#endif
}

inline bool dims(const std::string & site, int64_t * input, int64_t * output) {
    if (site == "self_attn.qkv_proj") { *input = 2048; *output = 4096; return true; }
    if (site == "self_attn.o_proj") { *input = 2048; *output = 2048; return true; }
    if (site == "mlp.gate_up_proj") { *input = 2048; *output = 12288; return true; }
    if (site == "mlp.down_proj") { *input = 6144; *output = 2048; return true; }
    return false;
}

inline bool add_pair(std::vector<STWTensor> & out, const std::string & module,
                     const char * target, const Yue2AitkNativeMatrix & a,
                     const Yue2AitkNativeMatrix & b, int64_t rank,
                     int64_t first_row, int64_t row_count) {
    if (a.rows != rank || a.cols <= 0 || b.rows < first_row + row_count || b.cols != rank ||
        !a.data || !b.data) return false;
    std::vector<int64_t> ashape{rank, a.cols};
    std::vector<int64_t> bshape{row_count, rank};
    // The returned STWTensors borrow caller storage. A split A is represented
    // by the same pointer in each file; st_write_file consumes it immediately.
    out.push_back({"yue2." + module + ".lora_A.weight", ashape, a.data, -1});
    out.push_back({"yue2." + module + ".lora_B.weight", bshape,
                   b.data + first_row * rank, -1});
    return true;
}

inline bool write_one(const char * path, const std::vector<STWTensor> & tensors,
                      const char * expert, int64_t rank, float alpha, int64_t steps, const std::string & trigger,
                      float caption_dropout = 0.0f) {
    if (exists(path) || tensors.empty() || !std::isfinite(alpha) || alpha <= 0.0f || rank <= 0) return false;
    std::vector<std::pair<std::string, std::string>> md = {
        {"format", std::string("yue2-") + expert + "-lora-v1"},
        {"rank", std::to_string(rank)}, {"alpha", std::to_string(alpha)},
        {"steps", std::to_string(steps)}, {"yue2_adapter_layout", "native_split_v1"},
    };
    if (!trigger.empty()) { md.push_back({"trigger", trigger}); md.push_back({"style_template", "upstream"}); }
    // Same key the legacy AR trainer writes; generate.ts reads it to know the
    // bare trigger is a prompt this adapter actually trained on.
    if (caption_dropout > 0.0f) {
        char buf[32]; snprintf(buf, sizeof(buf), "%.3f", (double) caption_dropout);
        md.push_back({"caption_dropout", buf});
    }
    const std::string tmp = temp_path(path);
    if (!st_write_file(tmp.c_str(), tensors, md, STW_BF16)) { hs_remove(tmp); return false; }
    if (!publish_new(tmp, path)) { hs_remove(tmp); return false; }
    return true;
}
} // namespace yue2_aitk_native_detail

// Export the complete 448-factor AI Toolkit inventory into separate native
// adapters. `diffusion_model` is the NAR expert; `text_encoders` is AR. The
// input inventory remains borrowed for the duration of this call.
inline bool yue2_aitk_write_native_split(const std::vector<Yue2AitkNativeMatrix> & factors,
                                         int64_t rank, float alpha, int64_t steps,
                                         const char * ar_path, const char * nar_path, const std::string & trigger = {},
                                         float caption_dropout = 0.0f) {
    using namespace yue2_aitk_native_detail;
    constexpr int layers = 28;
    constexpr size_t expected = 2u * layers * 4u * 2u;
    if (factors.size() != expected || rank <= 0 || !std::isfinite(alpha) || alpha <= 0.0f ||
        steps < 0 || !ar_path || !nar_path || std::string(ar_path) == nar_path ||
        exists(ar_path) || exists(nar_path)) return false;

    for (const auto & f : factors) {
        if (!f.data || f.rows <= 0 || f.cols <= 0 || f.rows > 12288 || f.cols > 6144) return false;
        for (int64_t i = 0; i < f.rows * f.cols; ++i) if (!std::isfinite(f.data[i])) return false;
    }
    std::vector<STWTensor> ar, nar;
    ar.reserve(expected * 3u); nar.reserve(expected * 3u);
    std::unordered_set<std::string> seen;
    seen.reserve(factors.size());
    for (const char * expert : {"diffusion_model", "text_encoders"}) {
        std::vector<STWTensor> & out = std::string(expert) == "text_encoders" ? ar : nar;
        for (int layer = 0; layer < layers; ++layer) {
            for (const char * site_c : {"self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"}) {
                const std::string site(site_c);
                int64_t input = 0, output = 0;
                if (!dims(site, &input, &output)) return false;
                const std::string base = std::string(expert) + ".model.layers." + std::to_string(layer) + "." + site + ".lora_";
                const Yue2AitkNativeMatrix * a = nullptr, * b = nullptr;
                for (const auto & f : factors) {
                    if (f.name == base + "A.weight") a = &f;
                    if (f.name == base + "B.weight") b = &f;
                }
                if (!a || !b || !seen.insert(base).second || a->rows != rank || a->cols != input ||
                    b->rows != output || b->cols != rank || !a->data || !b->data) return false;
                const std::string native = std::string("blk.") + std::to_string(layer) + "." +
                    (std::string(expert) == "text_encoders" ? "" : "nar_");
                if (site == "self_attn.qkv_proj") {
                    if (!add_pair(out, native + "attn_q", "", *a, *b, rank, 0, 2048) ||
                        !add_pair(out, native + "attn_k", "", *a, *b, rank, 2048, 1024) ||
                        !add_pair(out, native + "attn_v", "", *a, *b, rank, 3072, 1024)) return false;
                } else if (site == "self_attn.o_proj") {
                    if (!add_pair(out, native + "attn_output", "", *a, *b, rank, 0, 2048)) return false;
                } else if (site == "mlp.gate_up_proj") {
                    const std::string ffn = std::string("blk.") + std::to_string(layer) + "." +
                        (std::string(expert) == "text_encoders" ? "ffn_" : "nar_ffn_");
                    if (!add_pair(out, ffn + "gate", "", *a, *b, rank, 0, 6144) ||
                        !add_pair(out, ffn + "up", "", *a, *b, rank, 6144, 6144)) return false;
                } else {
                    const std::string ffn = std::string("blk.") + std::to_string(layer) + "." +
                        (std::string(expert) == "text_encoders" ? "ffn_down" : "nar_ffn_down");
                    if (!add_pair(out, ffn, "", *a, *b, rank, 0, 2048)) return false;
                }
            }
        }
    }
    if (!write_one(ar_path, ar, "ar", rank, alpha, steps, trigger, caption_dropout)) return false;
    if (!write_one(nar_path, nar, "nar", rank, alpha, steps, trigger, caption_dropout)) {
        hs_remove(ar_path);
        return false;
    }
    return true;
}
