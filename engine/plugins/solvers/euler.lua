-- euler.lua: First-order Euler ODE solver
-- Single evaluation, simplest possible solver.
-- xt_next = xt - vt * (t_curr - t_prev)

solver = {
    name        = "euler",
    display     = "Euler (ODE)",
    description = "First-order Euler step (default)",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = false,
    stochastic  = false,
}

function step(xt, vt, t_curr, t_prev, n)
    local dt = t_curr - t_prev
    for i = 0, n - 1 do
        xt[i] = xt[i] - vt[i] * dt
    end
end
