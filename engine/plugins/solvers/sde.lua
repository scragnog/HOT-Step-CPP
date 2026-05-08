-- sde.lua: Stochastic Differential Equation Euler solver (1 NFE)
-- Predicts x0 then re-noises with Philox random numbers.
-- Note: requires philox_randn C helper exposed to Lua.
-- Falls back to ODE Euler if no seed info available.

solver = {
    name        = "sde",
    display     = "SDE (Stochastic)",
    description = "SDE Euler with Philox re-noising for diversity",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = false,
    stochastic  = true,
}

function step(xt, vt, t_curr, t_prev, n)
    -- SDE needs philox_randn which is a C helper.
    -- For now, fall back to ODE Euler. The C++ wrapper will handle
    -- the SDE-specific logic (philox noise injection) when it detects
    -- this plugin is stochastic and seeds are available.
    local dt = t_curr - t_prev
    for i = 0, n - 1 do
        local x0 = xt[i] - vt[i] * t_curr
        -- Without philox, do deterministic ODE step
        xt[i] = xt[i] - vt[i] * dt
    end
end
