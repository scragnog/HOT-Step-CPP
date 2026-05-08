-- dpm2m_ada.lua: DPM++ 2M Adaptive (step-ratio-corrected AB2)
-- Adjusts AB2 coefficients based on step size ratio for non-uniform schedules.

solver = {
    name        = "dpm2m_ada",
    display     = "DPM++ 2M Adaptive",
    description = "Step-ratio-corrected AB2 for non-uniform schedules",
    nfe         = 1,
    order       = 2,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
}

local prev_vt = nil
local prev_dt = 0

function step(xt, vt, t_curr, t_prev, n)
    local dt = t_curr - t_prev

    if prev_vt and prev_dt > 0 then
        -- Step-ratio corrected AB2
        local r  = dt / prev_dt
        local c1 = 1.0 + r / 2.0
        local c0 = r / 2.0
        for i = 0, n - 1 do
            local v_eff = c1 * vt[i] - c0 * prev_vt[i]
            xt[i] = xt[i] - v_eff * dt
        end
    else
        -- First step: Euler
        for i = 0, n - 1 do
            xt[i] = xt[i] - vt[i] * dt
        end
    end

    prev_vt = {}
    for i = 0, n - 1 do prev_vt[i] = vt[i] end
    prev_dt = dt
end
