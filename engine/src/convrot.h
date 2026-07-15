#pragma once
// convrot.h: ConvRot "regular" Hadamard helpers (HOT-Step).
//
// ConvRot int8 models (convert_to_quant --convrot / ComfyUI >= 0.27) store
// linear weights pre-rotated per input-dim group: W' = W·H, where H is the
// REGULAR Hadamard of the group size — Kronecker powers of the 4x4 matrix
// below (power-of-4 sizes only), normalized by 1/sqrt(G). H is symmetric
// orthogonal (H·H = I), so the same transform both rotates and unrotates.
//
// Shared by the DiT loader (builds the full [G, G] matrix for the GPU-side
// activation rotation mul_mat) and the adapter runtime basin re-base (host-
// side unrotation of rotated base weights via the fast radix-4 transform).
//
// NOTE: this is NOT the Sylvester/Walsh matrix — ggml's FWHT kernels compute
// a different transform and must not be used for ConvRot data.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// H4 from ConvRot Theorem 3.3: every row/col sums to 2 (no all-ones row).
// Symmetric; columns equal rows.
static const float CONVROT_H4[16] = { 1, 1, 1, -1,   1, 1, -1, 1,   1, -1, 1, 1,   -1, 1, 1, 1 };

static inline bool convrot_is_pow4(int g) {
    if (g < 4) return false;
    while (g > 1) {
        if (g % 4 != 0) return false;
        g /= 4;
    }
    return true;
}

// Full normalized H_G as row-major [G, G] data (H_{4^{k+1}} = H_{4^k} ⊗ H4,
// matching convert_to_quant utils/convrot.py build_hadamard). Power-of-4 G
// only; returns false otherwise.
static inline bool convrot_build_h_data(int G, std::vector<float> & out) {
    if (!convrot_is_pow4(G) || G > 4096) return false;
    out.assign((size_t) G * G, 0.0f);
    std::vector<float> tmp((size_t) G * G);
    memcpy(out.data(), CONVROT_H4, sizeof(CONVROT_H4));
    int cur = 4;
    while (cur < G) {
        int next = cur * 4;  // kron(H_cur, H4)
        for (int i = 0; i < cur; i++) {
            for (int j = 0; j < cur; j++) {
                float a = out[(size_t) i * cur + j];
                for (int k = 0; k < 4; k++) {
                    for (int l = 0; l < 4; l++) {
                        tmp[(size_t) (i * 4 + k) * next + (j * 4 + l)] = a * CONVROT_H4[k * 4 + l];
                    }
                }
            }
        }
        memcpy(out.data(), tmp.data(), (size_t) next * next * sizeof(float));
        cur = next;
    }
    const float norm = 1.0f / sqrtf((float) G);
    for (size_t i = 0; i < (size_t) G * G; i++) out[i] *= norm;
    return true;
}

// In-place fast transform of one length-G group: v <- v·H_G.
// Radix-4 butterflies over the Kronecker structure — O(G·log4 G) instead of
// O(G²). Self-inverse (H·H = I), so this both rotates and unrotates.
static inline void convrot_transform_group(float * v, int G) {
    for (int stride = 1; stride < G; stride *= 4) {
        int block = stride * 4;
        for (int base = 0; base < G; base += block) {
            for (int off = 0; off < stride; off++) {
                float * p  = v + base + off;
                float   x0 = p[0], x1 = p[stride], x2 = p[2 * stride], x3 = p[3 * stride];
                p[0]          = x0 + x1 + x2 - x3;
                p[stride]     = x0 + x1 - x2 + x3;
                p[2 * stride] = x0 - x1 + x2 + x3;
                p[3 * stride] = -x0 + x1 + x2 + x3;
            }
        }
    }
    const float norm = 1.0f / sqrtf((float) G);
    for (int i = 0; i < G; i++) v[i] *= norm;
}

// Row-major (rows, cols) weight with groups of size g along cols:
// every row's every group is transformed in place.
static inline void convrot_transform_rows(float * w, int64_t rows, int64_t cols, int g) {
    if (g <= 0 || cols % g != 0) return;
    for (int64_t r = 0; r < rows; r++) {
        float * row = w + r * cols;
        for (int64_t c = 0; c < cols; c += g) {
            convrot_transform_group(row + c, g);
        }
    }
}
