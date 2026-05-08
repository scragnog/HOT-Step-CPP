-- heun.lua: Heun's method (improved Euler / explicit trapezoidal)
-- 2 NFE: evaluate at t_curr, predict, evaluate at t_prev, average.

solver = {
    name        = "heun",
    display     = "Heun (2 NFE)",
    description = "Second-order predictor-corrector",
    nfe         = 2,
    order       = 2,
    needs_model = true,
    stateful    = false,
    stochastic  = false,
}

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    local dt = t_curr - t_prev

    -- Predict: xt_pred = xt - vt * dt
    for i = 0, n - 1 do
        xt[i] = xt[i] - vt[i] * dt
    end

    -- Correct: evaluate at (xt_pred, t_prev)
    model_fn(xt, t_prev)

    -- Average: xt = xt + 0.5 * (vt_buf - vt) * dt
    -- Note: xt is already xt_pred = xt_orig - vt * dt
    -- We want: xt_orig - 0.5*(vt + vt_buf)*dt
    -- = (xt + vt*dt) - 0.5*(vt + vt_buf)*dt
    -- = xt + 0.5*(vt - vt_buf)*dt
    for i = 0, n - 1 do
        xt[i] = xt[i] + 0.5 * (vt[i] - vt_buf[i]) * dt
    end
end
