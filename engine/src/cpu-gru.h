#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cpu_gru {

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// One GRU direction over precomputed input gates [n, 3H] -> y columns [off, off+H) of a strided output buffer.
inline void dir(int H, const float * weight, const float * bias, const float * gx, int64_t n, bool reverse, float * y,
                int64_t y_stride, int64_t y_off) {
    std::vector<float> h((size_t) H, 0.0f), gh((size_t) 3 * H);
    for (int64_t s = 0; s < n; s++) {
        const int64_t t  = reverse ? n - 1 - s : s;
        const float * gi = gx + t * 3 * H;  // W_ih x + b_ih, precomputed
        for (int o = 0; o < 3 * H; o++) {
            const float * wr  = weight + (size_t) o * H;
            float         acc = 0.0f;
            for (int i = 0; i < H; i++) {
                acc += wr[i] * h[(size_t) i];
            }
            gh[(size_t) o] = acc + bias[(size_t) o];
        }
        float * yr = y + t * y_stride + y_off;
        for (int i = 0; i < H; i++) {
            const float r  = sigmoid(gi[i] + gh[(size_t) i]);
            const float z  = sigmoid(gi[H + i] + gh[(size_t) (H + i)]);
            const float nn = std::tanh(gi[2 * H + i] + r * gh[(size_t) (2 * H + i)]);
            h[(size_t) i]  = (1.0f - z) * nn + z * h[(size_t) i];
            yr[i]          = h[(size_t) i];
        }
    }
}

}  // namespace cpu_gru
