#pragma once
// dwt-haar.h: Haar DWT/IDWT 1D + DCW low-band correction.
//
// Implements the sampler-side Differential Correction in Wavelet domain
// from: Meng Yu, Lei Sun, Jianhao Zeng, Xiangxiang Chu, Kun Zhan.
// "Elucidating the SNR-t Bias of Diffusion Probabilistic Models",
// CVPR 2026. arXiv:2604.16044. https://github.com/AMAP-ML/DCW
//
// Recipe (mode "low", wavelet "haar"):
//   denoised = xt_before - vt * t_curr       (predicted clean sample)
//   xL, xH = DWT(xt)
//   yL, _  = DWT(denoised)
//   xL    += (t_curr * scaler) * (xL - yL)
//   xt     = IDWT(xL, xH)
//
// Haar 1D orthonormal kernels:
//   low-pass  h = [ 1/sqrt(2),  1/sqrt(2) ]
//   high-pass g = [ 1/sqrt(2), -1/sqrt(2) ]
// Zero-pad on odd T, trim on reconstruction (matches pytorch_wavelets mode='zero').

#include <cmath>
#include <vector>

// Forward Haar DWT along the T axis of a [T, C] buffer.
// out_L and out_H each hold Tl = (T + 1) / 2 frames of C channels.
// src layout: src[t * C + c]. out layout: out_{L,H}[tl * C + c].
static inline void dwt_haar_fwd_tc(const float * src, int T, int C, float * out_L, float * out_H) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    int         Tl        = (T + 1) / 2;
    for (int tl = 0; tl < Tl; tl++) {
        int           i0    = 2 * tl;
        int           i1    = 2 * tl + 1;
        const float * a     = src + i0 * C;
        // Zero-padded high index on odd T: pretend src[i1] = 0.
        bool          has_b = (i1 < T);
        float *       L     = out_L + tl * C;
        float *       H     = out_H + tl * C;
        if (has_b) {
            const float * b = src + i1 * C;
            for (int c = 0; c < C; c++) {
                L[c] = (a[c] + b[c]) * inv_sqrt2;
                H[c] = (a[c] - b[c]) * inv_sqrt2;
            }
        } else {
            for (int c = 0; c < C; c++) {
                L[c] = a[c] * inv_sqrt2;
                H[c] = a[c] * inv_sqrt2;
            }
        }
    }
}

// Inverse Haar IDWT along the T axis, reconstructing exactly T frames.
// L and H must each hold Tl = (T + 1) / 2 frames of C channels.
static inline void dwt_haar_inv_tc(const float * L, const float * H, int T, int C, float * out) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    int         Tl        = (T + 1) / 2;
    for (int tl = 0; tl < Tl; tl++) {
        int           i0 = 2 * tl;
        int           i1 = 2 * tl + 1;
        const float * Lc = L + tl * C;
        const float * Hc = H + tl * C;
        float *       a  = out + i0 * C;
        for (int c = 0; c < C; c++) {
            a[c] = (Lc[c] + Hc[c]) * inv_sqrt2;
        }
        if (i1 < T) {
            float * b = out + i1 * C;
            for (int c = 0; c < C; c++) {
                b[c] = (Lc[c] - Hc[c]) * inv_sqrt2;
            }
        }
        // i1 >= T: odd T, last destination frame doesn't exist, discard.
    }
}

// DCW mode "low" correction on xt using Haar wavelet, in place.
// Pushes xt's low-frequency band away from denoised's low-frequency band.
// Formula: xL += effective_scaler * (xL - yL), then xt = IDWT(xL, xH).
// xt and denoised are [T, C] single-sample buffers.
// Tmp buffers must each provide Tl * C floats, Tl = (T + 1) / 2.
static inline void dcw_haar_low_inplace(float *       xt,
                                        const float * denoised,
                                        int           T,
                                        int           C,
                                        float         effective_scaler,
                                        float *       tmp_xL,
                                        float *       tmp_xH,
                                        float *       tmp_yL,
                                        float *       tmp_yH) {
    if (effective_scaler == 0.0f) {
        return;
    }
    int Tl = (T + 1) / 2;
    dwt_haar_fwd_tc(xt, T, C, tmp_xL, tmp_xH);
    dwt_haar_fwd_tc(denoised, T, C, tmp_yL, tmp_yH);
    int n_L = Tl * C;
    for (int i = 0; i < n_L; i++) {
        tmp_xL[i] = tmp_xL[i] + effective_scaler * (tmp_xL[i] - tmp_yL[i]);
    }
    dwt_haar_inv_tc(tmp_xL, tmp_xH, T, C, xt);
}

// DCW mode "high" correction on xt using Haar wavelet, in place.
// Pushes xt's high-frequency band away from denoised's high-frequency band.
// Formula: xH += effective_scaler * (xH - yH), then xt = IDWT(xL, xH).
static inline void dcw_haar_high_inplace(float *       xt,
                                         const float * denoised,
                                         int           T,
                                         int           C,
                                         float         effective_scaler,
                                         float *       tmp_xL,
                                         float *       tmp_xH,
                                         float *       tmp_yL,
                                         float *       tmp_yH) {
    if (effective_scaler == 0.0f) {
        return;
    }
    int Tl  = (T + 1) / 2;
    int n_H = Tl * C;
    dwt_haar_fwd_tc(xt, T, C, tmp_xL, tmp_xH);
    dwt_haar_fwd_tc(denoised, T, C, tmp_yL, tmp_yH);
    for (int i = 0; i < n_H; i++) {
        tmp_xH[i] = tmp_xH[i] + effective_scaler * (tmp_xH[i] - tmp_yH[i]);
    }
    dwt_haar_inv_tc(tmp_xL, tmp_xH, T, C, xt);
}

// DCW mode "double" correction on xt using Haar wavelet, in place.
// Applies independent correction to both bands with distinct scalers.
// Formula: xL += low_s * (xL - yL);  xH += high_s * (xH - yH).
static inline void dcw_haar_double_inplace(float *       xt,
                                           const float * denoised,
                                           int           T,
                                           int           C,
                                           float         low_scaler,
                                           float         high_scaler,
                                           float *       tmp_xL,
                                           float *       tmp_xH,
                                           float *       tmp_yL,
                                           float *       tmp_yH) {
    if (low_scaler == 0.0f && high_scaler == 0.0f) {
        return;
    }
    int Tl  = (T + 1) / 2;
    int n_L = Tl * C;
    int n_H = Tl * C;
    dwt_haar_fwd_tc(xt, T, C, tmp_xL, tmp_xH);
    dwt_haar_fwd_tc(denoised, T, C, tmp_yL, tmp_yH);
    if (low_scaler != 0.0f) {
        for (int i = 0; i < n_L; i++) {
            tmp_xL[i] = tmp_xL[i] + low_scaler * (tmp_xL[i] - tmp_yL[i]);
        }
    }
    if (high_scaler != 0.0f) {
        for (int i = 0; i < n_H; i++) {
            tmp_xH[i] = tmp_xH[i] + high_scaler * (tmp_xH[i] - tmp_yH[i]);
        }
    }
    dwt_haar_inv_tc(tmp_xL, tmp_xH, T, C, xt);
}

// DCW mode "pix" correction on xt, in place. No wavelet transform.
// Pixel/latent-space differential correction, ablation baseline.
// Formula: xt += effective_scaler * (xt - denoised).
static inline void dcw_pix_inplace(float * xt, const float * denoised, int T, int C, float effective_scaler) {
    if (effective_scaler == 0.0f) {
        return;
    }
    int n = T * C;
    for (int i = 0; i < n; i++) {
        xt[i] = xt[i] + effective_scaler * (xt[i] - denoised[i]);
    }
}
