-- dpm2m.lua: DPM++ 2M solver (Adams-Bashforth 2, 1 NFE, stateful)
-- Uses previous step's velocity for 2nd order correction.
-- First step falls back to Euler.

solver = {
    name        = "dpm2m",
    display     = "DPM++ 2M",
    description = "2nd order Adams-Bashforth multistep (1 NFE)",
    nfe         = 1,
    order       = 2,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
}

-- Persistent state across steps (Lua tables survive between calls)
local prev_vt = nil

function step(xt, vt, t_curr, t_prev, n)
    local dt = t_curr - t_prev

    if prev_vt then
        -- AB2: v_eff = 1.5 * vt - 0.5 * prev_vt
        for i = 0, n - 1 do
            local v_eff = 1.5 * vt[i] - 0.5 * prev_vt[i]
            xt[i] = xt[i] - v_eff * dt
        end
    else
        -- First step: Euler
        for i = 0, n - 1 do
            xt[i] = xt[i] - vt[i] * dt
        end
    end

    -- Save velocity for next step
    prev_vt = {}
    for i = 0, n - 1 do prev_vt[i] = vt[i] end
end
