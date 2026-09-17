// Standalone CUDA probe for the staged YuE2 head-loss seam.
// Build-only artifact: the guarded launcher/root agent owns execution.
#include "../../engine/src/train/yue2-aitk-head-loss.h"
#include "../../engine/ggml/include/ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
constexpr size_t kRows = 184704, kWidth = 256, kPositions = 129;

template <class T>
std::vector<T> read_exact(const fs::path & path, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
        throw std::runtime_error("fixture size overflow: " + path.string());
    const size_t bytes = count * sizeof(T);
    if (!fs::is_regular_file(path) || fs::file_size(path) != bytes)
        throw std::runtime_error("unexpected fixture size: " + path.string());
    std::vector<T> out(count);
    std::ifstream in(path, std::ios::binary);
    if (!in || !in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("fixture read failed: " + path.string());
    return out;
}

std::vector<uint32_t> read_targets(const fs::path & path) {
    auto raw = read_exact<uint32_t>(path, kPositions);
    std::vector<uint32_t> out(kPositions);
    for (size_t i = 0; i < kPositions; ++i) {
        if (raw[i] >= static_cast<uint32_t>(kRows))
            throw std::runtime_error("fixture target is outside vocabulary");
        out[i] = raw[i];
    }
    return out;
}

std::array<double, 4> read_metrics(const fs::path & path) {
    auto v = read_exact<double>(path, 4);
    return {v[0], v[1], v[2], v[3]};
}

float bf16_to_f32(uint16_t x) {
    uint32_t bits = static_cast<uint32_t>(x) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct GradMetrics { double l2 = 0, max_abs = 0, cosine = 0, dot = 0; size_t nonfinite = 0; };

GradMetrics compare_grad(const std::vector<uint16_t> & expected,
                         const std::vector<uint16_t> & actual) {
    GradMetrics m;
    double ea = 0, aa = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float e = bf16_to_f32(expected[i]);
        const float a = bf16_to_f32(actual[i]);
        if (!std::isfinite(a) || !std::isfinite(e)) { ++m.nonfinite; continue; }
        const double d = static_cast<double>(a) - e;
        m.l2 += d * d; m.max_abs = std::max(m.max_abs, std::abs(d));
        m.dot += static_cast<double>(a) * e; ea += static_cast<double>(e) * e;
        aa += static_cast<double>(a) * a;
    }
    m.l2 = std::sqrt(m.l2);
    m.cosine = m.dot / std::sqrt(std::max(1e-30, ea * aa));
    return m;
}
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const fs::path fixture = argv[1], output = argv[2];
    if (fs::exists(output)) throw std::runtime_error("refusing existing output directory");
    fs::create_directory(output);

    auto q = read_exact<int8_t>(fixture / "weight_i8.bin", kRows * kWidth);
    auto scales = read_exact<float>(fixture / "scales_f32.bin", kRows);
    auto adapted = read_exact<float>(fixture / "adapted_hidden_f32.bin", kPositions * kWidth);
    auto base = read_exact<float>(fixture / "base_hidden_f32.bin", kPositions * kWidth);
    auto targets = read_targets(fixture / "targets_u32.bin");
    auto expected = read_exact<uint16_t>(fixture / "expected_hidden_grad_bf16.bin", kPositions * kWidth);
    const auto reference = read_metrics(fixture / "reference_metrics.bin");
    for (float x : adapted) if (!std::isfinite(x)) throw std::runtime_error("nonfinite adapted hidden");
    for (float x : base) if (!std::isfinite(x)) throw std::runtime_error("nonfinite base hidden");

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) throw std::runtime_error("CUDA backend unavailable");
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 + 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("GGML metadata allocation failed");
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kWidth, kRows);
    ggml_tensor * scales_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kRows);
    if (!weight || !scales_t) throw std::runtime_error("head metadata allocation failed");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer || !ggml_backend_buffer_get_base(buffer))
        throw std::runtime_error("head backend allocation failed");
    ggml_backend_tensor_set(weight, q.data(), 0, q.size());
    ggml_backend_tensor_set(scales_t, scales.data(), 0, scales.size() * sizeof(float));
    Yue2AitkConvRotLinear head{weight, scales_t, 256, static_cast<int64_t>(kRows), static_cast<int64_t>(kWidth)};
    std::vector<uint16_t> actual(kPositions * kWidth);
    float ce = 0, kl = 0;
    yue2_aitk_head_loss::Request request{backend, &head, adapted.data(), base.data(), targets.data(),
                                         kPositions, kWidth, 0.2, actual.data(), &ce, &kl};
    std::string error;
    const auto status = yue2_aitk_head_loss::compute(request, &error);
    if (status != yue2_aitk_head_loss::Status::success)
        throw std::runtime_error("head-loss compute failed: " + error);
    const auto gm = compare_grad(expected, actual);
    fs::create_directories(output);
    std::ofstream(output / "actual_hidden_grad_bf16.bin", std::ios::binary).write(
        reinterpret_cast<const char *>(actual.data()), static_cast<std::streamsize>(actual.size() * sizeof(uint16_t)));
    std::ofstream report(output / "metrics.json");
    report << std::setprecision(17) << "{\n"
           << "  \"reference_ce_sum\": " << reference[0] << ",\n  \"actual_ce_sum\": " << ce << ",\n"
           << "  \"reference_kl_sum\": " << reference[1] << ",\n  \"actual_kl_sum\": " << kl << ",\n"
           << "  \"ce_abs_error\": " << std::abs(ce - reference[0]) << ",\n"
           << "  \"kl_abs_error\": " << std::abs(kl - reference[1]) << ",\n"
           << "  \"grad_l2\": " << gm.l2 << ",\n  \"grad_max_abs\": " << gm.max_abs << ",\n"
           << "  \"grad_cosine\": " << gm.cosine << ",\n  \"grad_dot\": " << gm.dot << ",\n"
           << "  \"grad_nonfinite\": " << gm.nonfinite << "\n}\n";
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
