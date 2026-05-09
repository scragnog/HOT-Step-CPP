-- apg.lua: Adaptive Perpendicular Guidance (default)
-- Routes through the native C++ apg_forward() for momentum smoothing,
-- per-channel norm thresholding, and perpendicular projection.
-- In practice, the engine takes the native C++ path for APG directly,
-- but this Lua implementation exists as a correct fallback.

guidance = {
    name        = "apg",
    display     = "APG",
    description = "Adaptive perpendicular guidance (default)",
    params      = {
        { key = "momentum", type = "slider", label = "Momentum",
          default = 0.75, min = -1, max = 1, step = 0.01,
          hint = "APG momentum coefficient (negative = adaptive)" },
        { key = "norm_threshold", type = "slider", label = "Norm Threshold",
          default = 2.5, min = 0, max = 10, step = 0.1,
          hint = "APG norm clipping threshold" },
    },
}

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    -- Route through native APG C++ implementation
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
end
