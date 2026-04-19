#pragma once
// guidance-implementations.h: all guidance mode implementations
//
// All modes route through apg_forward() for the core APG pipeline
// (momentum smoothing + norm thresholding + perpendicular projection).
// Modes differ by pre-processing (scale adjustment) or post-processing.
//
// Matches Python acestep/core/generation/guidance.py

#include "guidance-interface.h"

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ═══════════════════════════════════════════════════════════════════════════
// APG — Adaptive Perpendicular Guidance (native upstream default)
// ═══════════════════════════════════════════════════════════════════════════
static void guidance_apg(const float *       pred_cond,
                         const float *       pred_uncond,
                         float               guidance_scale,
                         APGMomentumBuffer & mbuf,
                         float *             result,
                         int                 Oc,
                         int                 T,
                         const GuidanceCtx & /*ctx*/,
                         float               norm_threshold) {
    apg_forward(pred_cond, pred_uncond, guidance_scale, mbuf, result, Oc, T, norm_threshold);
}


// ═══════════════════════════════════════════════════════════════════════════
// CFG++ — Step-scaled guidance for large steps
//
// Uses APG with an effective scale reduced by the step-size-to-sigma ratio.
// Prevents over-correction when dt is large relative to t_curr.
// ═══════════════════════════════════════════════════════════════════════════
static void guidance_cfg_pp(const float *       pred_cond,
                            const float *       pred_uncond,
                            float               guidance_scale,
                            APGMomentumBuffer & mbuf,
                            float *             result,
                            int                 Oc,
                            int                 T,
                            const GuidanceCtx & ctx,
                            float               norm_threshold) {
    float effective_scale = guidance_scale;
    if (ctx.t_curr > 1e-6f) {
        float step_scale = fabsf(ctx.dt) / ctx.t_curr;
        effective_scale  = 1.0f + (guidance_scale - 1.0f) * step_scale;
    }
    apg_forward(pred_cond, pred_uncond, effective_scale, mbuf, result, Oc, T, norm_threshold);
}


// ═══════════════════════════════════════════════════════════════════════════
// Dynamic CFG — Cosine-decaying guidance schedule
//
// Full guidance early in generation (structure), reduced guidance later
// (fine detail preservation). Uses cos^0.5 decay.
// ═══════════════════════════════════════════════════════════════════════════
static void guidance_dynamic_cfg(const float *       pred_cond,
                                 const float *       pred_uncond,
                                 float               guidance_scale,
                                 APGMomentumBuffer & mbuf,
                                 float *             result,
                                 int                 Oc,
                                 int                 T,
                                 const GuidanceCtx & ctx,
                                 float               norm_threshold) {
    const float power = 0.5f;
    float progress = (float) ctx.step_idx / fmaxf((float) (ctx.total_steps - 1), 1.0f);
    // cosf(π/2) in float32 can return a tiny negative (-4e-8) due to precision.
    // powf(negative, 0.5) = NaN, so clamp to 0.
    float decay    = powf(fmaxf(cosf((float) M_PI / 2.0f * progress), 0.0f), power);
    float effective_scale = 1.0f + (guidance_scale - 1.0f) * decay;
    apg_forward(pred_cond, pred_uncond, effective_scale, mbuf, result, Oc, T, norm_threshold);
}


// ═══════════════════════════════════════════════════════════════════════════
// Rescaled CFG — Std-matched post-processing
//
// Runs APG at full scale, then rescales the output to match the conditional
// prediction's standard deviation. Prevents over-saturation from high guidance.
// Blends rescaled and raw output with phi factor.
// ═══════════════════════════════════════════════════════════════════════════
static void guidance_rescaled_cfg(const float *       pred_cond,
                                  const float *       pred_uncond,
                                  float               guidance_scale,
                                  APGMomentumBuffer & mbuf,
                                  float *             result,
                                  int                 Oc,
                                  int                 T,
                                  const GuidanceCtx & /*ctx*/,
                                  float               norm_threshold) {
    int n = Oc * T;
    float phi = (guidance_scale > 4.0f) ? 0.95f : 0.7f;

    // Run standard APG
    apg_forward(pred_cond, pred_uncond, guidance_scale, mbuf, result, Oc, T, norm_threshold);

    // Compute std of conditional prediction (over all elements)
    double sum_cond = 0.0, sum2_cond = 0.0;
    double sum_guided = 0.0, sum2_guided = 0.0;
    for (int i = 0; i < n; i++) {
        double c = (double) pred_cond[i];
        double g = (double) result[i];
        sum_cond += c;   sum2_cond += c * c;
        sum_guided += g; sum2_guided += g * g;
    }
    double mean_cond   = sum_cond / n;
    double mean_guided = sum_guided / n;
    double var_cond    = sum2_cond / n - mean_cond * mean_cond;
    double var_guided  = sum2_guided / n - mean_guided * mean_guided;
    double std_cond    = (var_cond > 0.0)   ? sqrt(var_cond)   : 1e-5;
    double std_guided  = (var_guided > 0.0) ? sqrt(var_guided) : 1e-5;

    // Rescale to match conditional std
    double factor = std_cond / (std_guided + 1e-5);
    for (int i = 0; i < n; i++) {
        float rescaled = (float) ((double) result[i] * factor);
        // Blend rescaled and raw
        result[i] = phi * rescaled + (1.0f - phi) * result[i];
    }
}
