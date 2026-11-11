-- ACE samplers dispatch this mode through guidance/adg-core.h, as for APG.
guidance = {
    name = "adg",
    display = "ADG (ACE-Step)",
    description = "Angle-based Dynamic Guidance for ACE-Step base models. Rotates clean-latent estimates with a clipped angle. Other backends use ordinary CFG.",
    params = {},
}

-- Other backend bridges have no current latent in their guide() contract.
function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T)
    for i = 1, Oc * T do
        result[i] = pred_uncond[i] + guidance_scale * (pred_cond[i] - pred_uncond[i])
    end
end
