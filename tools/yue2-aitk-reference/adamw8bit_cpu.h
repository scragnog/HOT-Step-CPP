#pragma once
// Portable fixture reader for the bounded AdamW8bit parity probe.
//
// This header intentionally does not contain a guessed optimizer update. The
// bitsandbytes CUDA update kernel is not shipped in the reference checkout;
// replacing it with CPU AdamW would produce a false parity result.
// Formula source: bitsandbytes tag 0.49.2, commit
// f0e6ca31b32c4744a9cee4e31610b25796cbf778, csrc/kernels.cu
// kOptimizerStatic8bit2StateBlockwise. Upstream is MIT licensed; this scalar
// translation is maintained as a separately marked diagnostic foundation.

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aitk_adamw8bit {

struct Array {
    std::string name;
    uint32_t dtype = 0; // 1=f32, 2=u8, 3=i32, 4=f64
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
};

struct Fixture {
    uint32_t version = 0;
    std::vector<Array> arrays;
};

template <typename T> inline T read(std::istream & in) {
    T value{};
    if (!in.read(reinterpret_cast<char *>(&value), sizeof(value))) throw std::runtime_error("truncated optimizer fixture");
    return value;
}

inline Fixture read_fixture(std::istream & in) {
    char magic[8]{};
    if (!in.read(magic, sizeof(magic)) || std::string(magic, 8) != "AITKOPT1")
        throw std::runtime_error("invalid AITKOPT1 fixture magic");
    Fixture f;
    f.version = read<uint32_t>(in);
    const uint32_t count = read<uint32_t>(in);
    if (f.version != 1 || count > 100000) throw std::runtime_error("unsupported optimizer fixture header");
    for (uint32_t i = 0; i < count; ++i) {
        Array a;
        const uint32_t name_len = read<uint32_t>(in);
        if (name_len == 0 || name_len > 4096) throw std::runtime_error("invalid optimizer array name");
        a.name.resize(name_len);
        if (!in.read(a.name.data(), name_len)) throw std::runtime_error("truncated optimizer array name");
        a.dtype = read<uint32_t>(in);
        const uint32_t ndim = read<uint32_t>(in);
        if (ndim > 8) throw std::runtime_error("optimizer array rank exceeds limit");
        uint64_t elements = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            const uint64_t n = read<uint64_t>(in);
            if (n == 0 || n > (1ull << 34) || elements > (1ull << 34) / n)
                throw std::runtime_error("invalid optimizer array shape");
            a.shape.push_back(n);
            elements *= n;
        }
        const uint64_t bytes = read<uint64_t>(in);
        const uint64_t item = a.dtype == 1 ? 4 : a.dtype == 2 ? 1 : a.dtype == 3 ? 4 : a.dtype == 4 ? 8 : 0;
        if (!item || bytes != elements * item || bytes > (1ull << 36))
            throw std::runtime_error("invalid optimizer array payload");
        a.bytes.resize(static_cast<size_t>(bytes));
        if (!in.read(reinterpret_cast<char *>(a.bytes.data()), static_cast<std::streamsize>(bytes)))
            throw std::runtime_error("truncated optimizer array payload");
        f.arrays.push_back(std::move(a));
    }
    return f;
}

inline bool has(const Fixture & f, const std::string & name) {
    for (const auto & a : f.arrays) if (a.name == name) return true;
    return false;
}

inline const Array & get(const Fixture & f, const std::string & name) {
    for (const auto & a : f.arrays) if (a.name == name) return a;
    throw std::runtime_error("missing optimizer array: " + name);
}

inline float f32(const Array & a, size_t i) {
    float v; std::memcpy(&v, a.bytes.data() + i * sizeof(float), sizeof(float)); return v;
}
inline void put_f32(Array & a, size_t i, float v) { std::memcpy(a.bytes.data() + i * sizeof(float), &v, sizeof(float)); }

inline std::vector<float> dynamic_map(bool signed_map) {
    std::vector<float> out;
    const int max_exp = 7, non_sign = 7;
    for (int i = 0; i < max_exp; ++i) {
        const int count = signed_map ? int(std::pow(2.0, i + non_sign - max_exp) + 1)
                                     : int(std::pow(2.0, i + non_sign - max_exp + 1) + 1);
        for (int j = 0; j + 1 < count; ++j) {
            const float lo = 0.1f + 0.9f * j / float(count - 1);
            const float hi = 0.1f + 0.9f * (j + 1) / float(count - 1);
            const float v = std::pow(10.0f, float(-(max_exp - 1) + i)) * (lo + hi) * 0.5f;
            out.push_back(v); if (signed_map) out.push_back(-v);
        }
    }
    // For the v0.49.2 8-bit map max_exponent_bits == non_sign_bits, so the
    // upstream additional-items branch contributes zero entries.
    out.push_back(0.0f); out.push_back(1.0f);
    while (out.size() < 256) out.push_back(0.0f);
    std::sort(out.begin(), out.end());
    return out;
}

inline uint8_t quantize(float x, const std::vector<float> & code) {
    int lo = 0, hi = 255;
    while (hi - lo > 1) { const int mid = (lo + hi) / 2; if (x > code[mid]) lo = mid; else hi = mid; }
    return uint8_t(x > (code[lo] + code[hi]) * 0.5f ? hi : lo);
}

// Exact scalar translation of csrc/kernels.cu kOptimizerStatic8bit2StateBlockwise
// for FP32 parameters. Returns the maximum absolute parameter error against
// the expected arrays embedded in a converted fixture.
inline double replay_fp32(const Fixture & fixture, int steps = 5) {
    const auto smap = dynamic_map(true), umap = dynamic_map(false);
    const float lr = 1e-4f, b1 = .9f, b2 = .999f, eps = 1e-6f, wd = 1e-4f;
    double worst = 0.0;
    for (const char * side : {"small", "large"}) {
        const std::string s(side); const size_t n = get(fixture, "parameter_" + s + "_initial").bytes.size() / 4;
        std::vector<float> p(n), m(n), v(n); std::vector<uint8_t> q1(n), q2(n);
        const auto & initial = get(fixture, "parameter_" + s + "_initial");
        for (size_t i = 0; i < n; ++i) p[i] = f32(initial, i);
        std::vector<float> a1((n + 255) / 256), a2((n + 255) / 256);
        for (int step = 1; step <= steps; ++step) {
            const auto & g = get(fixture, "gradient_" + s + "_step_" + std::to_string(step - 1));
            for (size_t block = 0; block < a1.size(); ++block) {
                const size_t begin = block * 256, end = std::min(begin + 256, n);
                float ma = 0.0f, va = 0.0f;
                for (size_t i = begin; i < end; ++i) {
                    const float grad = f32(g, i);
                    const float old_m = n < 4096 ? m[i] : smap[q1[i]] * a1[block];
                    const float old_v = n < 4096 ? v[i] : umap[q2[i]] * a2[block];
                    m[i] = old_m * b1 + grad * (1 - b1); v[i] = old_v * b2 + grad * grad * (1 - b2);
                    ma = std::max(ma, std::abs(m[i])); va = std::max(va, std::abs(v[i]));
                }
                a1[block] = ma; a2[block] = va;
                const float c1 = 1 - std::pow(b1, float(step)), c2 = std::sqrt(1 - std::pow(b2, float(step)));
                const float step_size = -lr * c2 / c1;
                for (size_t i = begin; i < end; ++i) {
                    p[i] += step_size * (m[i] / (std::sqrt(v[i]) + c2 * eps)); p[i] *= 1 - lr * wd;
                    if (n >= 4096) {
                        q1[i] = quantize(a1[block] ? m[i] / a1[block] : 0, smap);
                        q2[i] = quantize(a2[block] ? v[i] / a2[block] : 0, umap);
                        if (std::signbit(smap[q1[i]]) != std::signbit(m[i])) q1[i] = uint8_t(std::max(0, std::min(255, int(q1[i]) + (m[i] > 0 ? 1 : -1))));
                    }
                }
            }
            const auto & expected = get(fixture, "parameter_" + s + "_after_step_" + std::to_string(step));
            for (size_t i = 0; i < n; ++i) worst = std::max(worst, double(std::abs(p[i] - f32(expected, i))));
        }
    }
    return worst;
}

} // namespace aitk_adamw8bit
