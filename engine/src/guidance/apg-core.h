#pragma once
// apg-core.h: APG (Adaptive Projected Guidance) primitives
//
// Extracted from dit-sampler.h so guidance modes can share the core functions.
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py
//
// NOTE: Python reference uses .double() but empirical testing shows F32 produces
// audibly identical results while halving memory bandwidth for APG computation.

#include <cmath>
#include <cstring>
#include <vector>

// Momentum buffer for APG running average smoothing.
// Matches Python APGMomentumBuffer with momentum=-0.75
struct APGMomentumBuffer {
    float               momentum;
    std::vector<float>  running_average;
    bool                initialized;

    APGMomentumBuffer(float m = -0.75f) : momentum(m), initialized(false) {}

    void update(const float * values, int n) {
        if (!initialized) {
            running_average.assign(values, values + n);
            initialized = true;
        } else {
            for (int i = 0; i < n; i++) {
                running_average[i] = values[i] + momentum * running_average[i];
            }
        }
    }
};

// Pre-allocated workspace for APG forward pass. Eliminates heap
// allocation per step. Allocate once before the denoising loop, reuse every step.
struct APGWorkspace {
    std::vector<float> diff;
    std::vector<float> pred_cond_copy;
    std::vector<float> par;
    std::vector<float> orth;

    void resize(int n) {
        diff.resize(n);
        pred_cond_copy.resize(n);
        par.resize(n);
        orth.resize(n);
    }
};

// project(v0, v1, dims=[1]): decompose v0 into parallel + orthogonal w.r.t. v1
// Layout: memory [T, Oc] time-major (ggml ne=[Oc, T]).
// Python dims=[1] on [B,T,C] = normalize/project per channel over T dimension.
// In memory [T, Oc] layout: for each channel c, operate over all T time frames.
static void apg_project(const float * v0, const float * v1,
                         float * out_par, float * out_orth,
                         int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        float norm2 = 0.0f;
        for (int t = 0; t < T; t++) {
            norm2 += v1[t * Oc + c] * v1[t * Oc + c];
        }
        float inv_norm = (norm2 > 1e-30f) ? (1.0f / sqrtf(norm2)) : 0.0f;

        float dot = 0.0f;
        for (int t = 0; t < T; t++) {
            dot += v0[t * Oc + c] * (v1[t * Oc + c] * inv_norm);
        }

        for (int t = 0; t < T; t++) {
            int   idx    = t * Oc + c;
            float v1n    = v1[idx] * inv_norm;
            out_par[idx]  = dot * v1n;
            out_orth[idx] = v0[idx] - out_par[idx];
        }
    }
}

// APG forward matching Python apg_forward():
//   1. diff = cond - uncond
//   2. momentum.update(diff); diff = running_average
//   3. norm clip: per-channel L2 over T (dims=[1]), clip to norm_threshold=2.5
//   4. project(diff, pred_COND) -> (parallel, orthogonal)
//   5. result = pred_cond + (scale - 1) * orthogonal
//
// Overload with pre-allocated workspace (preferred — avoids heap allocs per step).
static void apg_forward(const float *       pred_cond,
                        const float *       pred_uncond,
                        float               guidance_scale,
                        APGMomentumBuffer & mbuf,
                        float *             result,
                        int                 Oc,
                        int                 T,
                        float               norm_threshold,
                        APGWorkspace &      ws) {
    int n = Oc * T;

    // 1. diff = cond - uncond
    for (int i = 0; i < n; i++) {
        ws.diff[i] = pred_cond[i] - pred_uncond[i];
    }

    // 2. momentum update, then use smoothed diff
    mbuf.update(ws.diff.data(), n);
    memcpy(ws.diff.data(), mbuf.running_average.data(), n * sizeof(float));

    // 3. norm clipping: per-channel L2 over T (dims=[1]), clip to threshold
    if (norm_threshold > 0.0f) {
        for (int c = 0; c < Oc; c++) {
            float norm2 = 0.0f;
            for (int t = 0; t < T; t++) {
                norm2 += ws.diff[t * Oc + c] * ws.diff[t * Oc + c];
            }
            float norm = sqrtf(norm2 > 0.0f ? norm2 : 0.0f);
            float s    = (norm > 1e-30f) ? fminf(1.0f, norm_threshold / norm) : 1.0f;
            if (s < 1.0f) {
                for (int t = 0; t < T; t++) {
                    ws.diff[t * Oc + c] *= s;
                }
            }
        }
    }

    // 4. project(diff, pred_COND) -> orthogonal component
    memcpy(ws.pred_cond_copy.data(), pred_cond, n * sizeof(float));
    apg_project(ws.diff.data(), ws.pred_cond_copy.data(), ws.par.data(), ws.orth.data(), Oc, T);

    // 5. result = pred_cond + (scale - 1) * orthogonal
    float w = guidance_scale - 1.0f;
    for (int i = 0; i < n; i++) {
        result[i] = pred_cond[i] + w * ws.orth[i];
    }
}

// Legacy overload: allocates workspace internally (for callers that don't pre-allocate).
static void apg_forward(const float *       pred_cond,
                        const float *       pred_uncond,
                        float               guidance_scale,
                        APGMomentumBuffer & mbuf,
                        float *             result,
                        int                 Oc,
                        int                 T,
                        float               norm_threshold = 2.5f) {
    APGWorkspace ws;
    ws.resize(Oc * T);
    apg_forward(pred_cond, pred_uncond, guidance_scale, mbuf, result, Oc, T, norm_threshold, ws);
}
