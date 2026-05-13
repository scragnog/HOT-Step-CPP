-- euler_fullloop.lua: Full-loop Euler ODE solver (test plugin)
-- Functionally identical to the built-in Euler solver but exercises
-- the full-loop plugin path (owns_loop = true).
-- Used to validate the lua_call_solver_loop() bridge.

solver = {
    name        = "euler_fullloop",
    display     = "Euler (Full Loop Test)",
    description = "Euler ODE via full-loop API (test/validation plugin)",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = false,
    stochastic  = false,
    owns_loop   = true,
}

function sample(xt, vt_buf, schedule, n, model_fn)
    local ns = #schedule
    for i = 1, ns do
        local t_curr = schedule[i]

        -- Evaluate model at current state
        model_fn(xt, t_curr)
        -- vt_buf now contains velocity

        if i < ns then
            -- Normal step: Euler update
            local t_next = schedule[i + 1]
            local dt = t_curr - t_next
            for j = 0, n - 1 do
                xt[j] = xt[j] - vt_buf[j] * dt
            end
            -- Report step (triggers DCW, repaint, progress)
            if on_step(i - 1, t_curr, t_next) then return end
        else
            -- Final step: predict x0
            for j = 0, n - 1 do
                xt[j] = xt[j] - vt_buf[j] * t_curr
            end
        end
    end
end
