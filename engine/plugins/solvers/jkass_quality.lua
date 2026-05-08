-- jkass_quality.lua: JKASS Quality solver (2 NFE)
-- Heun with derivative averaging.

solver = {
    name        = "jkass_quality",
    display     = "JKASS Quality (2 NFE)",
    description = "Heun predictor-corrector with derivative averaging",
    accent      = "amber",
    nfe         = 2,
    order       = 2,
    needs_model = true,
    stateful    = false,
    stochastic  = false,
}

function step(xt, vt, t_curr, t_prev, n, model_fn, vt_buf)
    local dt = t_curr - t_prev

    if t_prev <= 0 then
        -- Fallback to Euler for final step
        for i = 0, n - 1 do xt[i] = xt[i] - vt[i] * dt end
        return
    end

    -- Save k1
    local k1 = {}
    for i = 0, n - 1 do k1[i] = vt[i] end

    -- Euler predictor
    for i = 0, n - 1 do xt[i] = xt[i] - vt[i] * dt end

    -- Second evaluation
    model_fn(xt, t_prev)

    -- Correct: undo Euler, apply averaged derivative
    for i = 0, n - 1 do
        xt[i] = xt[i] + k1[i] * dt  -- undo Euler
        xt[i] = xt[i] - 0.5 * (k1[i] + vt_buf[i]) * dt  -- Heun average
    end
end
