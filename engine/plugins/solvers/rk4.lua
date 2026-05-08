-- rk4.lua: Classic 4th-order Runge-Kutta solver
-- 4 NFE per step, excellent accuracy for smooth flows.

solver = {
    name        = "rk4",
    display     = "RK4 (4 NFE)",
    description = "Classic 4th-order Runge-Kutta",
    nfe         = 4,
    order       = 4,
    needs_model = true,
    stateful    = false,
    stochastic  = false,
}

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    local dt = t_curr - t_prev
    local t_mid = t_curr - 0.5 * dt

    -- k1 = vt (already evaluated)
    -- Save k1 and original xt
    local k1 = {}
    local xt_orig = {}
    for i = 0, n - 1 do
        k1[i] = vt[i]
        xt_orig[i] = xt[i]
    end

    -- k2: evaluate at midpoint using k1
    for i = 0, n - 1 do
        xt[i] = xt_orig[i] - 0.5 * k1[i] * dt
    end
    model_fn(xt, t_mid)
    local k2 = {}
    for i = 0, n - 1 do k2[i] = vt_buf[i] end

    -- k3: evaluate at midpoint using k2
    for i = 0, n - 1 do
        xt[i] = xt_orig[i] - 0.5 * k2[i] * dt
    end
    model_fn(xt, t_mid)
    local k3 = {}
    for i = 0, n - 1 do k3[i] = vt_buf[i] end

    -- k4: evaluate at endpoint using k3
    for i = 0, n - 1 do
        xt[i] = xt_orig[i] - k3[i] * dt
    end
    model_fn(xt, t_prev)

    -- Combine: xt = xt_orig - (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    for i = 0, n - 1 do
        xt[i] = xt_orig[i] - (k1[i] + 2*k2[i] + 2*k3[i] + vt_buf[i]) * dt / 6.0
    end
end
