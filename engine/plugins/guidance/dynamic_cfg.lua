-- dynamic_cfg.lua: Cosine-decaying guidance schedule
-- Full guidance early (structure), reduced guidance late (fine detail).
-- Routes through native APG (momentum + projection + norm thresholding).

guidance = {
    name        = "dynamic_cfg",
    display     = "Dynamic CFG",
    description = "Cosine-decaying guidance schedule with APG projection",
}

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local power = 0.5
    local progress = (step_idx or 0) / math.max((total_steps or 1) - 1, 1)
    local cos_val = math.max(math.cos(math.pi / 2 * progress), 0)
    local decay = cos_val ^ power
    local effective_scale = 1.0 + (guidance_scale - 1.0) * decay

    -- Route through native APG with the decayed scale
    apg(pred_cond, pred_uncond, effective_scale, result, Oc, T, norm_threshold)
end
