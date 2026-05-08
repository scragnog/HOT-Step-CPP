-- dpm3m.lua: DPM++ 3M solver (Adams-Bashforth 3, 1 NFE, stateful)
-- Uses two previous velocities for 3rd order correction.

solver = {
    name        = "dpm3m",
    display     = "DPM++ 3M",
    description = "3rd order Adams-Bashforth multistep (1 NFE)",
    nfe         = 1,
    order       = 3,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
}

local prev_vt = nil
local prev_prev_vt = nil

function step(xt, vt, t_curr, t_prev, n)
    local dt = t_curr - t_prev

    if prev_vt and prev_prev_vt then
        -- AB3: v_eff = (23/12)*vt - (16/12)*prev - (5/12)*prev_prev
        for i = 0, n - 1 do
            local v_eff = (23/12) * vt[i] - (16/12) * prev_vt[i] + (5/12) * prev_prev_vt[i]
            xt[i] = xt[i] - v_eff * dt
        end
    elseif prev_vt then
        -- AB2 fallback
        for i = 0, n - 1 do
            local v_eff = 1.5 * vt[i] - 0.5 * prev_vt[i]
            xt[i] = xt[i] - v_eff * dt
        end
    else
        -- Euler fallback
        for i = 0, n - 1 do
            xt[i] = xt[i] - vt[i] * dt
        end
    end

    -- Shift history
    prev_prev_vt = prev_vt
    prev_vt = {}
    for i = 0, n - 1 do prev_vt[i] = vt[i] end
end
