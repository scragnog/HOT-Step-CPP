#pragma once

#include "philox.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// Mix before cover/repaint initialization. Zero leaves the original bytes alone.
static inline void retake_noise(float * noise, int n, int64_t seed, float variance) {
    if (variance <= 0.0f) return;
    std::vector<float> other(n);
    philox_randn(seed, other.data(), n, /*bf16_round=*/true);
    if (variance >= 1.0f) {
        std::memcpy(noise, other.data(), n * sizeof(float));
        return;
    }
    const double angle = variance * (3.14159265358979323846 / 2);
    const float a = (float) std::cos(angle), b = (float) std::sin(angle);
    for (int i = 0; i < n; ++i) noise[i] = a * noise[i] + b * other[i];
}
