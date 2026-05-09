-- rescaled_cfg.lua: Std-matched post-processing guidance
-- Runs APG at full scale, then rescales output to match conditional std.
-- Routes through native APG (momentum + projection + norm thresholding),
-- then applies std-matching post-processing.

guidance = {
    name        = "rescaled_cfg",
    display     = "Rescaled CFG",
    description = "Std-matched to prevent saturation",
}

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n = Oc * T
    local phi = (guidance_scale > 4.0) and 0.95 or 0.7

    -- Run APG at full guidance scale first
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)

    -- Compute std of conditional prediction and guided output
    local sum_c, sum2_c = 0, 0
    local sum_g, sum2_g = 0, 0
    for i = 0, n - 1 do
        local c = pred_cond[i]
        local g = result[i]
        sum_c = sum_c + c;  sum2_c = sum2_c + c * c
        sum_g = sum_g + g;  sum2_g = sum2_g + g * g
    end
    local mean_c = sum_c / n
    local mean_g = sum_g / n
    local var_c = sum2_c / n - mean_c * mean_c
    local var_g = sum2_g / n - mean_g * mean_g
    local std_c = (var_c > 0) and math.sqrt(var_c) or 1e-5
    local std_g = (var_g > 0) and math.sqrt(var_g) or 1e-5

    -- Rescale to match conditional std, blend with raw APG output
    local factor = std_c / (std_g + 1e-5)
    for i = 0, n - 1 do
        local rescaled = result[i] * factor
        result[i] = phi * rescaled + (1 - phi) * result[i]
    end
end
