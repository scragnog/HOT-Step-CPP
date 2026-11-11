#pragma once

#include <algorithm>
#include <cmath>

// ACE-Step 1.5 adg_forward: per-frame angle guidance in clean-latent space.
// Layout [T, Oc]. The reference clips at 3.14/6, not an exact pi/6.
static inline void adg_forward(const float * xt, const float * cond, const float * uncond,
                               float scale, float sigma, float * result, int Oc, int T) {
    if (sigma < 1e-4f) {
        for (int i = 0; i < Oc * T; ++i) result[i] = uncond[i] + scale * (cond[i] - uncond[i]);
        return;
    }
    const double weight = std::max(0.0, (double) scale - 1.0) + 1e-3;
    for (int t = 0; t < T; ++t) {
        const int off = t * Oc;
        double cc = 0, uu = 0, cu = 0, du = 0;
        for (int c = 0; c < Oc; ++c) {
            const double a = xt[off+c] - sigma * cond[off+c];
            const double b = xt[off+c] - sigma * uncond[off+c];
            cc += a*a; uu += b*b; cu += a*b; du += (a-b)*b;
        }
        const double cosine = std::clamp(cu / ((std::sqrt(cc)+1e-8)*(std::sqrt(uu)+1e-8)), -0.99999, 0.99999);
        const double theta = std::acos(cosine);
        const double angle = std::min(weight * theta, 3.14 / 6);
        const double sine = std::sin(theta);
        const double factor = sine > 1e-3 ? std::sin(angle) / sine : weight;
        for (int c = 0; c < Oc; ++c) {
            const double a = xt[off+c] - sigma * cond[off+c];
            const double b = xt[off+c] - sigma * uncond[off+c];
            const double perp = a - b - du / (uu + 1e-8) * b;
            result[off+c] = (float) ((xt[off+c] - (std::cos(angle)*a + factor*perp)) / (sigma + 1e-8));
        }
    }
}

static inline bool cfg_in_interval(float sigma, float start, float end) {
    return sigma >= start && sigma <= end;
}
