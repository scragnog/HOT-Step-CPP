-- Evaluation port of the WassersteinFlowSolver shared by Alex Allan,
-- 2026-09-16. Redistribution permission for this port was confirmed by the
-- HOT-Step project owner on 2026-09-20.
--
-- The input is a flat time-major latent. This follows the shared Python
-- implementation's elementwise "spectral" energy; it does not take an FFT.

solver = {
    name = "md_wasserstein_yue2",
    display = "MD Wasserstein Flow (YuE2)",
    description = "One-evaluation Euler step with the shared proximal latent regularizer. Tuned for YuE2 NAR.",
    nfe = 1,
    order = 1,
    needs_model = false,
    stateful = false,
    params = {
        { key = "tau", type = "slider", label = "Proximal Scale", default = 1.0, min = 0.1, max = 5.0, step = 0.05 },
        { key = "spectral_weight", type = "slider", label = "Element Energy Weight", default = 0.1, min = 0, max = 1, step = 0.01 },
        { key = "rms_weight", type = "slider", label = "RMS Weight", default = 0.05, min = 0, max = 1, step = 0.01 },
        { key = "rms_target", type = "slider", label = "RMS Target (0 = Auto)", default = 0, min = 0, max = 3, step = 0.01 },
        { key = "sigma_gate", type = "slider", label = "Sigma Gate", default = 0.3, min = 0, max = 0.99, step = 0.01 },
        { key = "prox_iterations", type = "slider", label = "Proximal Iterations", default = 1, min = 1, max = 4, step = 1 },
        { key = "orthogonal_proj", type = "toggle", label = "Transverse Projection", default = true },
        { key = "latent_rms", type = "slider", label = "Auto Target RMS", default = 0.97, min = 0.1, max = 3, step = 0.01 },
    },
}

local EPS = 1e-8
local next_x = {}
local gradient = {}

local function setting(key, fallback)
    if params and params[key] ~= nil then return params[key] end
    return fallback
end

function step(xt, vt, t_curr, t_prev, n)
    if n <= 0 then return end
    local dt = t_prev - t_curr
    local base_disp_sq = 0
    for i = 0, n - 1 do
        local delta = dt * vt[i]
        next_x[i] = xt[i] + delta
        base_disp_sq = base_disp_sq + delta * delta
    end

    local tau = setting("tau", 1.0)
    local spectral_weight = setting("spectral_weight", 0.1)
    local rms_weight = setting("rms_weight", 0.05)
    local rms_target = setting("rms_target", 0)
    if rms_target == 0 then rms_target = setting("latent_rms", 0.97) end
    local gate = math.min(setting("sigma_gate", 0.3), 0.99)
    local iterations = math.floor(setting("prox_iterations", 1))
    local project = setting("orthogonal_proj", true)

    local strength = 1
    if t_curr > gate then
        strength = math.max(0, 1 - (t_curr - gate) / (1 - gate + EPS))
    end
    if strength > 0.01 and (spectral_weight > 0 or rms_weight > 0) then
        local effective_step = math.abs(dt) / (tau + EPS) * strength
        local max_disp = math.sqrt(base_disp_sq)
        for _ = 1, iterations do
            local sum_e = 0
            for i = 0, n - 1 do
                sum_e = sum_e + next_x[i] * next_x[i] + EPS
            end
            local inv_mean = 1 / (sum_e / n + EPS)
            local rms = math.sqrt(sum_e / n)
            local rms_scale = 2 * (rms - rms_target) / (n * rms + EPS)
            rms_scale = math.max(-1, math.min(1, rms_scale))

            local dot_gv, norm_v_sq = 0, EPS
            for i = 0, n - 1 do
                local g = 0
                if spectral_weight > 0 then
                    local x = next_x[i]
                    local raw = 2 * x * (inv_mean - 1 / (x * x + EPS)) / n
                    g = spectral_weight * math.max(-5, math.min(5, raw))
                end
                gradient[i] = g
                if project and spectral_weight > 0 then
                    dot_gv = dot_gv + g * vt[i]
                    norm_v_sq = norm_v_sq + vt[i] * vt[i]
                end
            end
            local projection = project and spectral_weight > 0 and dot_gv / norm_v_sq or 0
            local reg_disp_sq = 0
            for i = 0, n - 1 do
                local g = gradient[i] - projection * vt[i]
                if rms_weight > 0 then g = g + rms_weight * rms_scale * next_x[i] end
                gradient[i] = g
                local d = effective_step * g
                reg_disp_sq = reg_disp_sq + d * d
            end
            local actual_disp = math.sqrt(reg_disp_sq)
            local trust = 1
            if actual_disp > max_disp and actual_disp > EPS then
                trust = max_disp / actual_disp
            end
            for i = 0, n - 1 do
                next_x[i] = next_x[i] - effective_step * trust * gradient[i]
            end
        end
    end

    local finite = true
    for i = 0, n - 1 do
        if next_x[i] ~= next_x[i] or next_x[i] == math.huge or next_x[i] == -math.huge then
            finite = false
            break
        end
    end
    for i = 0, n - 1 do
        xt[i] = finite and next_x[i] or (xt[i] + dt * vt[i])
    end
end
