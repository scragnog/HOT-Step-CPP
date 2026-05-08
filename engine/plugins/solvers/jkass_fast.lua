-- jkass_fast.lua: JKASS Fast solver (1 NFE, stateful)
-- Euler with momentum blending, frequency damping, and temporal smoothing.
-- Port from jeankassio/JK-AceStep-Nodes.

solver = {
    name        = "jkass_fast",
    display     = "JKASS Fast",
    description = "Euler with beat stability, frequency damping, and temporal smoothing",
    accent      = "amber",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    params      = {
        { key = "beat_stability",    type = "slider", label = "Beat Stability",
          default = 0.25, min = 0, max = 1, step = 0.01,
          hint = "Momentum blend with previous step (0=off, 1=full momentum)" },
        { key = "frequency_damping", type = "slider", label = "Frequency Damping",
          default = 0.4, min = 0, max = 5, step = 0.1,
          hint = "Attenuate high-frequency bins (0=off)" },
        { key = "temporal_smoothing", type = "slider", label = "Temporal Smoothing",
          default = 0.13, min = 0, max = 1, step = 0.01,
          hint = "1D blur across time axis (0=off)" },
    },
}

local prev_delta = nil

-- Frequency damping: exponential decay across channel dimension
local function apply_frequency_damping(data, offset, T, Oc, damping)
    if damping <= 0 then return end
    local freq_mult = {}
    for c = 0, Oc - 1 do
        local freq = c / (Oc - 1)
        freq_mult[c] = math.exp(-damping * freq * freq)
    end
    for t = 0, T - 1 do
        for c = 0, Oc - 1 do
            local idx = offset + t * Oc + c
            data[idx] = data[idx] * freq_mult[c]
        end
    end
end

-- Temporal smoothing: [0.25, 0.5, 0.25] blur across time axis
local function apply_temporal_smoothing(data, offset, T, Oc, strength)
    if strength <= 0 or T < 3 then return end
    local smoothed = {}
    for c = 0, Oc - 1 do
        for t = 0, T - 1 do
            local t_prev = (t > 0) and (t - 1) or 1
            local t_next = (t < T - 1) and (t + 1) or (T - 2)
            local v_prev = data[offset + t_prev * Oc + c]
            local v_curr = data[offset + t * Oc + c]
            local v_next = data[offset + t_next * Oc + c]
            smoothed[t * Oc + c] = 0.25 * v_prev + 0.5 * v_curr + 0.25 * v_next
        end
    end
    for i = 0, T * Oc - 1 do
        data[offset + i] = (1 - strength) * data[offset + i] + strength * smoothed[i]
    end
end

function step(xt, vt, t_curr, t_prev, n)
    local dt = t_curr - t_prev

    -- Read params (injected by C++ before each call)
    local bs  = params and params.beat_stability or 0.25
    local fd  = params and params.frequency_damping or 0.4
    local ts  = params and params.temporal_smoothing or 0.13

    -- Copy velocity as working delta
    local delta = {}
    for i = 0, n - 1 do delta[i] = vt[i] end

    -- Beat stability: momentum blend
    if prev_delta and bs > 0 then
        for i = 0, n - 1 do
            delta[i] = (1 - bs) * delta[i] + bs * prev_delta[i]
        end
    end

    -- Save for next step
    prev_delta = {}
    for i = 0, n - 1 do prev_delta[i] = delta[i] end

    -- Frequency damping (per batch item, Oc=64 for ACE-Step)
    if fd > 0 and n_per and n_per > 0 then
        local Oc = 64
        local T = n_per / Oc
        for b = 0, batch_n - 1 do
            apply_frequency_damping(delta, b * n_per, T, Oc, fd)
        end
    end

    -- Temporal smoothing (per batch item)
    if ts > 0 and n_per and n_per > 0 then
        local Oc = 64
        local T = n_per / Oc
        for b = 0, batch_n - 1 do
            apply_temporal_smoothing(delta, b * n_per, T, Oc, ts)
        end
    end

    -- Euler step with modified delta
    for i = 0, n - 1 do
        xt[i] = xt[i] - delta[i] * dt
    end
end
