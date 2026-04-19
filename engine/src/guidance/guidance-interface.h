#pragma once
// guidance-interface.h: universal guidance mode contract for ACE-Step
//
// A guidance mode defines HOW conditional and unconditional model predictions
// are combined via CFG. All modes route through APG core (momentum + projection),
// but differ in scale adjustment or post-processing.
//
// Interface:
//   guidance_fn(pred_cond, pred_uncond, scale, mbuf, result, Oc, T, ctx)

#include "apg-core.h"

// Context passed to guidance functions — carries per-step metadata.
struct GuidanceCtx {
    int   step_idx;     // current step index (0-based)
    int   total_steps;  // total number of diffusion steps
    float dt;           // t_curr - t_next (timestep delta)
    float t_curr;       // current timestep (sigma)
};

// Guidance function signature.
// pred_cond/pred_uncond: model outputs [n_per], conditional and unconditional
// guidance_scale: user-set CFG scale
// mbuf: per-batch APG momentum buffer (maintained across steps)
// result: output [n_per] — the guided velocity
// Oc, T: channel count and time frames
// ctx: step context for modes that use it
// norm_threshold: APG norm clipping threshold (default 2.5)
using GuidanceFn = void (*)(const float *       pred_cond,
                            const float *       pred_uncond,
                            float               guidance_scale,
                            APGMomentumBuffer & mbuf,
                            float *             result,
                            int                 Oc,
                            int                 T,
                            const GuidanceCtx & ctx,
                            float               norm_threshold);
