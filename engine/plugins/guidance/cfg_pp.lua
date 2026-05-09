-- cfg_pp.lua: CFG++ — Step-scaled guidance for large steps
-- Reduces effective scale by the step-size-to-sigma ratio.
-- Routes through native APG (momentum + projection + norm thresholding).

guidance = {
    name        = "cfg_pp",
    display     = "CFG++",
    description = "Step-scaled guidance for few-step models",
}

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    -- Effective scale adjusted by step ratio (t_curr and dt are globals)
    local effective_scale = guidance_scale
    if t_curr and t_curr > 1e-6 and dt then
        local step_scale = math.abs(dt) / t_curr
        effective_scale = 1.0 + (guidance_scale - 1.0) * step_scale
    end

    -- Route through native APG with the adjusted scale
    apg(pred_cond, pred_uncond, effective_scale, result, Oc, T, norm_threshold)
end
