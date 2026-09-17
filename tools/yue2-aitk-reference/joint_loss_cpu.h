#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace yue2_joint_loss {

struct Result {
    double ce = 0.0;
    double kl = 0.0;
    std::vector<float> gradient; // d(CE + kl_weight * KL(base || adapted))/d adapted logits
};

inline Result loss_and_gradient(const std::vector<float> & adapted, const std::vector<float> & base,
                                const std::vector<unsigned> & targets, std::size_t positions,
                                std::size_t vocab, std::size_t chunk = 512, float kl_weight = 0.2f) {
    if (positions == 0 || vocab == 0 || adapted.size() != positions * vocab || base.size() != adapted.size() ||
        targets.size() != positions || chunk == 0) {
        throw std::invalid_argument("invalid AR loss fixture dimensions");
    }
    for (unsigned t : targets) if (t >= vocab) throw std::invalid_argument("AR target outside vocabulary");
    Result out;
    out.gradient.assign(adapted.size(), 0.0f);
    for (std::size_t begin = 0; begin < positions; begin += chunk) {
        const std::size_t end = (begin + chunk < positions) ? begin + chunk : positions;
        for (std::size_t row = begin; row < end; ++row) {
            const float *a = adapted.data() + row * vocab;
            const float *b = base.data() + row * vocab;
            float amax = -std::numeric_limits<float>::infinity();
            float bmax = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < vocab; ++j) { amax = std::max(amax, a[j]); bmax = std::max(bmax, b[j]); }
            double asum = 0.0, bsum = 0.0;
            for (std::size_t j = 0; j < vocab; ++j) { asum += std::exp(double(a[j] - amax)); bsum += std::exp(double(b[j] - bmax)); }
            const double alogz = double(amax) + std::log(asum);
            const double blogz = double(bmax) + std::log(bsum);
            out.ce += -double(a[targets[row]]) + alogz;
            for (std::size_t j = 0; j < vocab; ++j) {
                const double p = std::exp(double(a[j]) - alogz);
                const double q = std::exp(double(b[j]) - blogz); // detached teacher: no base gradient is returned
                out.kl += q * ((double(b[j]) - blogz) - (double(a[j]) - alogz));
                out.gradient[row * vocab + j] = float((p - (j == targets[row] ? 1.0 : 0.0)) + kl_weight * (p - q));
            }
        }
    }
    const double inv_n = 1.0 / double(positions);
    out.ce *= inv_n;
    out.kl *= inv_n;
    for (float &g : out.gradient) g = float(double(g) * inv_n);
    return out;
}

} // namespace yue2_joint_loss
