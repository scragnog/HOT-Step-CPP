--[[
md_storm_core.lua
STORM -- Stabilized Taylor Oscillation with Runge-Kutta Memory
V4: Commons integration + relational decomposition

© 2026 Alexander Allan (MDMAchine) | A&E Concepts
GPL v3
--]]

local C = require("md_solver_commons")

solver = {
    name        = "md_storm_V4",
    display     = "MD STORM V4",
    description = "Adaptive STORK/DPM++3M hybrid with relational velocity decomposition, pseudo-LTE error estimation, look-back SNR smoother, stiffness dispatch.",
    accent      = "cyan",
    nfe         = 0,
    order       = 5,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    owns_loop   = true,
    params      = {
        -- ── Pseudo-LTE Research Controls ──
        { key = "attenuation_k", type = "slider", label = "Attenuation K",
          default = 25.0, min = 1.0, max = 30.0, step = 0.5,
          hint = "Exponential decay factor for highest-order memory term in pseudo-LTE estimation." },
        { key = "hyst_downgrade", type = "slider", label = "Downgrade Threshold",
          default = 0.40, min = 0.10, max = 0.80, step = 0.02,
          hint = "Pseudo-LTE above which solver drops order. Lower = more cautious." },
        { key = "hyst_upgrade", type = "slider", label = "Upgrade Threshold",
          default = 0.25, min = 0.05, max = 0.40, step = 0.01,
          hint = "Pseudo-LTE below which solver regains trust and considers upgrading order." },
        { key = "stability_window", type = "slider", label = "Stability Window",
          default = 4, min = 1, max = 8, step = 1,
          hint = "Consecutive stable steps required before order upgrade. Higher = more conservative." },

        -- ── Stiffness Detection ──
        { key = "stiffness_threshold", type = "slider", label = "Stiffness Threshold",
          default = 0.15, min = 0.05, max = 0.50, step = 0.01,
          hint = "Base threshold for STORK/DPM++ dispatch. Lower = more STORK (precise), higher = more DPM++ (smooth)." },
        { key = "stiffness_hysteresis", type = "slider", label = "Stiffness Hysteresis",
          default = 0.05, min = 0.0, max = 0.20, step = 0.01,
          hint = "Dead zone preventing rapid mode switching. Higher = stickier dispatch." },
        { key = "stiffness_ema", type = "slider", label = "Stiffness EMA",
          default = 0.4, min = 0.05, max = 0.8, step = 0.05,
          hint = "EMA smoothing for stiffness ratio. Lower = more reactive, higher = more stable." },

        -- ── Look-Back Smoother ──
        { key = "look_back_lambda", type = "slider", label = "Look-Back Lambda",
          default = 0.5, min = 0.0, max = 1.0, step = 0.01,
          hint = "Inter-step smoothing strength. 0=off (raw). 0.15=standard. Higher = smoother but softer detail." },
        { key = "look_back_snr_power", type = "slider", label = "Look-Back SNR Power",
          default = 1.2, min = 0.5, max = 3.0, step = 0.1,
          hint = "Concentrates smoothing on early noisy steps. Higher = heavier early smoothing, leaves late detail alone." },

        -- ── Solver Order & Cache ──
        { key = "rk_order", type = "select", label = "Precision Level",
          default = "auto",
          options = {
            { value = "auto", label = "Auto (Recommended)" },
            { value = "2",    label = "Low (RK2)" },
            { value = "3",    label = "Medium (RK3)" },
            { value = "4",    label = "High (RK4)" },
            { value = "5",    label = "Maximum (RK5)" },
          },
          hint = "Max STORK order. Auto ramps up as cache fills. DPM++3M always uses order 3." },
        { key = "cache_depth", type = "slider", label = "Cache Depth",
          default = 5, min = 2, max = 10, step = 1,
          hint = "Velocity history size. More = higher order available, diminishing returns past 5." },

        -- ── Diagnostics ──
        { key = "telemetry", type = "toggle", label = "Output Telemetry",
          default = false,
          hint = "Print JSON diagnostic logs to console at generation end." },
        { key = "verbose", type = "toggle", label = "Verbose Logging",
          default = true,
          hint = "Print per-step solver decisions to console (debug)." },
        { key = "relational_weight", type = "slider", label = "Relational Weight",
          default = 0.0, min = 0.0, max = 1.0, step = 0.05,
          hint = "Barbour Best Matching velocity decomposition. 0 = off. 0.3-0.5 = balanced." },
        { key = "relational_sigma_power", type = "slider", label = "Relational Sigma Decay",
          default = 1.0, min = 0.25, max = 4.0, step = 0.25,
          hint = "How fast relational weight fades. 1.0 = linear." },
    },
}

-- ─────────────────────────────────────────────────────────────────────────────
-- HELPERS (aliased from md_solver_commons)
-- ─────────────────────────────────────────────────────────────────────────────
local EPSILON     = C.EPSILON
local fa_to_tbl   = C.fa_to_tbl
local tbl_to_fa   = C.tbl_to_fa
local vec_norm    = C.vec_norm
local vec_sub_norm = C.vec_sub_norm
local vec_dot     = C.vec_dot
local vec_clone   = C.vec_clone
local has_nan_inf_tbl = C.has_nan_inf
local clamp       = C.clamp

-- ─────────────────────────────────────────────────────────────────────────────
-- LOOK-BACK SMOOTHER (arXiv:2602.09449)
-- ─────────────────────────────────────────────────────────────────────────────
local function look_back_smooth(x_curr, x_prev, sigma_curr, sigma_max, lambda_base, snr_power, n)
    if x_prev == nil then return x_curr, 0.0 end
    local ratio = math.min(sigma_curr / math.max(sigma_max, 1e-8), 1.0)
    local lam   = lambda_base * math.max(ratio ^ snr_power, 0.15)
    local out   = {}
    for i = 0, n - 1 do out[i] = (1.0 - lam) * x_curr[i] + lam * x_prev[i] end
    return out, lam
end

-- ─────────────────────────────────────────────────────────────────────────────
-- STIFFNESS DETECTION (from deployed STORM v3.0)
-- ─────────────────────────────────────────────────────────────────────────────
local function compute_stiffness(v_curr, v_cache, step_idx, baseline, threshold, ema_alpha, n_calib, n)
    if #v_cache < 1 then return true, baseline, nil end

    local v_prev     = v_cache[#v_cache].v
    local norm_delta = vec_sub_norm(v_curr, v_prev, n)
    local norm_curr  = vec_norm(v_curr, n) + 1e-8
    local raw_ratio  = norm_delta / norm_curr

    local prev_ema = baseline.ema or raw_ratio
    local smoothed = ema_alpha * raw_ratio + (1.0 - ema_alpha) * prev_ema
    baseline.ema   = smoothed

    local dot     = vec_dot(v_curr, v_prev, n)
    local nc      = vec_norm(v_curr, n)
    local np_     = vec_norm(v_prev, n)
    local cos_sim = dot / (nc * np_ + 1e-8)

    if step_idx < n_calib then
        baseline.sum        = (baseline.sum   or 0.0) + smoothed
        baseline.count      = (baseline.count or 0)   + 1
        baseline.last_ratio = smoothed
        return true, baseline, cos_sim
    end

    local bmean    = baseline.sum / math.max(baseline.count, 1)
    local adap_thr = threshold * (bmean / 0.15)
    adap_thr       = clamp(adap_thr, 0.05, 0.50)

    local stiff = smoothed > adap_thr
    baseline.last_ratio     = smoothed
    baseline.last_threshold = adap_thr
    return stiff, baseline, cos_sim
end

-- ─────────────────────────────────────────────────────────────────────────────
-- STORK MULTI-ORDER (AB2-AB5, cosine-similarity damping, from deployed STORM)
-- ─────────────────────────────────────────────────────────────────────────────
local function stork_step(v_cache, x, sigma_curr, sigma_next, v_curr, rk_order, n)
    local dt      = sigma_next - sigma_curr
    local n_cache = #v_cache

    local actual_order
    if rk_order == "auto" then
        actual_order = (n_cache >= 1) and math.min(n_cache + 1, 5) or 1
    else
        actual_order = (n_cache >= 1) and math.min(tonumber(rk_order), n_cache + 1) or 1
    end
    actual_order = math.max(actual_order, 1)

    -- Euler fallback
    if n_cache < 1 or actual_order <= 1 then
        local x_next = {}
        for i = 0, n - 1 do x_next[i] = x[i] + dt * v_curr[i] end
        return x_next, 1
    end

    local e0         = v_cache[#v_cache]
    local v_prev_0   = e0.v
    local sigma_prev = e0.sigma

    -- Cosine-similarity damping (deployed STORM's approach)
    local dot     = vec_dot(v_curr, v_prev_0, n)
    local nc      = vec_norm(v_curr, n)
    local np_     = vec_norm(v_prev_0, n)
    local cos_sim = dot / (nc * np_ + 1e-8)
    local damping = clamp(cos_sim, 0.0, 1.0)

    local denom = sigma_curr - sigma_prev
    if math.abs(denom) < 1e-8 then
        local x_next = {}
        for i = 0, n - 1 do x_next[i] = x[i] + dt * v_curr[i] end
        return x_next, 2
    end
    local alpha = (sigma_next - sigma_curr) / denom

    local x_next = {}

    if actual_order == 2 then
        for i = 0, n - 1 do
            local v_extrap = v_curr[i] + (alpha * damping) * (v_curr[i] - v_prev_0[i])
            x_next[i]     = x[i] + dt * (0.5 * v_curr[i] + 0.5 * v_extrap)
        end

    elseif actual_order == 3 and n_cache >= 2 then
        local v1, s1 = v_cache[#v_cache].v,     v_cache[#v_cache].sigma
        local v2, s2 = v_cache[#v_cache - 1].v, v_cache[#v_cache - 1].sigma
        local h  = sigma_curr - s1
        local h1 = s1 - s2
        if math.abs(h) < 1e-8 or math.abs(h1) < 1e-8 then
            for i = 0, n - 1 do
                local ve = v_curr[i] + (alpha * damping) * (v_curr[i] - v1[i])
                x_next[i] = x[i] + dt * (0.5 * v_curr[i] + 0.5 * ve)
            end
            actual_order = 2
        else
            local c0 = 1.0 + (dt / (2.0 * h)) + (dt ^ 2 / (3.0 * h * h1))
            local c1 = -(dt / (2.0 * h)) * (1.0 + dt / h1)
            local c2 = (dt ^ 2) / (3.0 * h * h1)
            for i = 0, n - 1 do
                local v_pred = c0 * v_curr[i] + c1 * v1[i] + c2 * v2[i]
                x_next[i]   = x[i] + dt * (v_curr[i] + damping * (v_pred - v_curr[i]))
            end
        end

    elseif actual_order == 4 and n_cache >= 3 then
        local v1, s1 = v_cache[#v_cache].v,     v_cache[#v_cache].sigma
        local v2, s2 = v_cache[#v_cache - 1].v, v_cache[#v_cache - 1].sigma
        local v3, s3 = v_cache[#v_cache - 2].v, v_cache[#v_cache - 2].sigma
        local h  = sigma_curr - s1
        local h1 = s1 - s2
        local h2 = s2 - s3
        if math.abs(h) < 1e-8 or math.abs(h1) < 1e-8 or math.abs(h2) < 1e-8 then
            local c0 = 1.0 + (dt / (2.0 * h)) + (dt ^ 2 / (3.0 * h * h1))
            local c1 = -(dt / (2.0 * h)) * (1.0 + dt / h1)
            local c2 = (dt ^ 2) / (3.0 * h * h1)
            for i = 0, n - 1 do
                local vp = c0 * v_curr[i] + c1 * v1[i] + c2 * v2[i]
                x_next[i] = x[i] + dt * (v_curr[i] + damping * (vp - v_curr[i]))
            end
            actual_order = 3
        else
            local c0 = 1.0 + dt/(2.0*h) + dt^2/(3.0*h*h1) + dt^3/(4.0*h*h1*h2)
            local c1 = -(dt/(2.0*h)) * (1.0 + dt/h1 + dt^2/(2.0*h1*h2))
            local c2 = (dt^2/(3.0*h*h1)) * (1.0 + dt/(2.0*h2))
            local c3 = -(dt^3) / (4.0*h*h1*h2)
            for i = 0, n - 1 do
                local vp = c0*v_curr[i] + c1*v1[i] + c2*v2[i] + c3*v3[i]
                x_next[i] = x[i] + dt * (v_curr[i] + damping * (vp - v_curr[i]))
            end
        end

    elseif actual_order >= 5 and n_cache >= 4 then
        local v1, s1 = v_cache[#v_cache].v,     v_cache[#v_cache].sigma
        local v2, s2 = v_cache[#v_cache - 1].v, v_cache[#v_cache - 1].sigma
        local v3, s3 = v_cache[#v_cache - 2].v, v_cache[#v_cache - 2].sigma
        local v4, s4 = v_cache[#v_cache - 3].v, v_cache[#v_cache - 3].sigma
        local h  = sigma_curr - s1
        local h1 = s1 - s2
        local h2 = s2 - s3
        local h3 = s3 - s4
        if math.abs(h) < 1e-8 or math.abs(h1) < 1e-8 or math.abs(h2) < 1e-8 or math.abs(h3) < 1e-8 then
            local c0 = 1.0 + dt/(2.0*h) + dt^2/(3.0*h*h1) + dt^3/(4.0*h*h1*h2)
            local c1 = -(dt/(2.0*h)) * (1.0 + dt/h1 + dt^2/(2.0*h1*h2))
            local c2 = (dt^2/(3.0*h*h1)) * (1.0 + dt/(2.0*h2))
            local c3 = -(dt^3) / (4.0*h*h1*h2)
            for i = 0, n - 1 do
                local vp = c0*v_curr[i] + c1*v1[i] + c2*v2[i] + c3*v3[i]
                x_next[i] = x[i] + dt * (v_curr[i] + damping * (vp - v_curr[i]))
            end
            actual_order = 4
        else
            local c0 = 1.0 + dt/(2.0*h) + dt^2/(3.0*h*h1) + dt^3/(4.0*h*h1*h2) + dt^4/(5.0*h*h1*h2*h3)
            local c1 = -(dt/(2.0*h)) * (1.0 + dt/h1 + dt^2/(2.0*h1*h2) + dt^3/(3.0*h1*h2*h3))
            local c2 = (dt^2/(3.0*h*h1)) * (1.0 + dt/(2.0*h2) + dt^2/(3.0*h2*h3))
            local c3 = -(dt^3/(4.0*h*h1*h2)) * (1.0 + dt/(2.0*h3))
            local c4 = dt^4 / (5.0*h*h1*h2*h3)
            for i = 0, n - 1 do
                local vp = c0*v_curr[i] + c1*v1[i] + c2*v2[i] + c3*v3[i] + c4*v4[i]
                x_next[i] = x[i] + dt * (v_curr[i] + damping * (vp - v_curr[i]))
            end
            actual_order = 5
        end

    else
        -- Fallback AB2
        for i = 0, n - 1 do
            local ve = v_curr[i] + (alpha * damping) * (v_curr[i] - v_prev_0[i])
            x_next[i] = x[i] + dt * (0.5 * v_curr[i] + 0.5 * ve)
        end
        actual_order = 2
    end

    return x_next, actual_order
end

-- ─────────────────────────────────────────────────────────────────────────────
-- DPM++3M (smooth schedule path, from deployed STORM)
-- ─────────────────────────────────────────────────────────────────────────────
local function dpmpp3m_step(v_cache, x, sigma_curr, sigma_next, v_curr, n)
    local dt     = sigma_next - sigma_curr
    local x_next = {}

    if #v_cache >= 2 then
        local v1, s1 = v_cache[#v_cache].v,     v_cache[#v_cache].sigma
        local v2, s2 = v_cache[#v_cache - 1].v, v_cache[#v_cache - 1].sigma
        local h  = sigma_curr - s1
        local h1 = s1 - s2
        if math.abs(h) < 1e-8 or math.abs(h1) < 1e-8 then
            for i = 0, n - 1 do x_next[i] = x[i] + dt * v_curr[i] end
        else
            local cc = 1.0 + (dt / (2.0 * h)) + (dt ^ 2 / (3.0 * h * h1))
            local c1 = -(dt / (2.0 * h)) * (1.0 + dt / h1)
            local c2 = (dt ^ 2) / (3.0 * h * h1)
            for i = 0, n - 1 do x_next[i] = x[i] + dt * (cc * v_curr[i] + c1 * v1[i] + c2 * v2[i]) end
        end
    elseif #v_cache >= 1 then
        local v1, s1 = v_cache[#v_cache].v, v_cache[#v_cache].sigma
        local h = sigma_curr - s1
        if math.abs(h) < 1e-8 then
            for i = 0, n - 1 do x_next[i] = x[i] + dt * v_curr[i] end
        else
            for i = 0, n - 1 do x_next[i] = x[i] + dt * (v_curr[i] + (dt / (2.0 * h)) * (v_curr[i] - v1[i])) end
        end
    else
        for i = 0, n - 1 do x_next[i] = x[i] + dt * v_curr[i] end
    end

    return x_next
end

-- ─────────────────────────────────────────────────────────────────────────────
-- PSEUDO-LTE ESTIMATION (Phase 1 research addition)
-- Computes kinetic-floored relative error from the highest-order AB term.
-- Returns rel_epsilon and attenuation weight w_t for telemetry.
-- ─────────────────────────────────────────────────────────────────────────────
local function estimate_pseudo_lte(v_cache, v_curr, ema_vel, k_atten, n)
    local n_cache = #v_cache
    if n_cache < 1 then return 0.0, 1.0 end

    -- Use the oldest cached velocity as the "highest order contribution" proxy
    local v_oldest = v_cache[1].v
    local oldest_norm = vec_norm(v_oldest, n)

    -- Extrapolation norm: current velocity (proxy for full polynomial magnitude)
    local extrap_norm = vec_norm(v_curr, n)

    local safe_den = math.max(extrap_norm, ema_vel * 0.5)
    local rel_epsilon = oldest_norm / (safe_den + EPSILON)
    local w_t = math.exp(-k_atten * rel_epsilon)

    return rel_epsilon, w_t
end

-- ─────────────────────────────────────────────────────────────────────────────
-- SAMPLE — Full-loop entry point
-- ─────────────────────────────────────────────────────────────────────────────
function sample(xt, vt_buf, schedule, n, model_fn)
    local p = params or {}

    -- Pseudo-LTE research params
    local k_atten      = p.attenuation_k     or 10.0
    local thr_down     = p.hyst_downgrade     or 0.40
    local thr_up       = p.hyst_upgrade       or 0.15
    local stab_window  = math.floor(p.stability_window or 3)

    -- Stiffness detection params
    local stiff_thr    = p.stiffness_threshold or 0.15
    local hyst         = p.stiffness_hysteresis or 0.05
    local ema_a        = p.stiffness_ema       or 0.3

    -- Look-back smoother params
    local lb_lambda    = p.look_back_lambda    or 0.15
    local lb_snr_pow   = p.look_back_snr_power or 1.5

    -- Solver order & cache params
    local rk_order     = p.rk_order            or "auto"
    local depth_max    = math.floor(p.cache_depth or 5)

    -- Diagnostics
    local do_tele      = p.telemetry           or false
    local verbose      = p.verbose             or false
    local rw           = C.num_param(p, "relational_weight", 0.0)
    local rw_sig_pow   = C.num_param(p, "relational_sigma_power", 1.0)

    -- Derived constants
    local calib_frac   = 0.12
    local ns           = #schedule
    -- Engine schedule has NO trailing 0 (fix ported from 46c081e): iterate all ns
    -- entries so the last iteration gets sigma_next = 0.0 and the terminal branch
    -- performs the final x0 projection. With ns - 1 that branch is dead code and
    -- the output keeps ~final-sigma noise.
    local n_steps      = ns
    if n_steps < 1 then return end

    local v_cache      = {}
    local baseline     = { sum = 0.0, count = 0 }
    local sigma_max    = schedule[1]
    local n_calib      = math.max(2, math.min(5, math.floor(n_steps * calib_frac)))
    local lb_enabled   = (lb_lambda > 0)

    -- Pseudo-LTE state
    local current_order     = 1
    local stability_counter = 0
    local ema_vel           = 0.0
    local telemetry_data    = {}

    if verbose then
        print(string.format("[STORM] Schedule: %d steps | Calib: %d | RK: %s | Cache: %d | LB: %.2f^%.1f",
            n_steps, n_calib, tostring(rk_order), depth_max, lb_lambda, lb_snr_pow))
    end

    -- Working copy
    local x = fa_to_tbl(xt, n)

    -- Seed x_prev for look-back (jittered copy, same as deployed STORM)
    local x_prev_lb = nil
    if lb_enabled then
        x_prev_lb = {}
        for i = 0, n - 1 do
            local u1 = math.max(1e-12, math.random())
            local u2 = math.random()
            local r  = (sigma_max * 0.1) * math.sqrt(-2.0 * math.log(u1))
            x_prev_lb[i] = x[i] + r * math.cos(2 * math.pi * u2)
        end
    end

    -- Model eval helper
    local function eval_at(x_tbl, t_val)
        tbl_to_fa(x_tbl, xt, n)
        model_fn(xt, t_val)
        return fa_to_tbl(vt_buf, n)
    end

    for i = 1, n_steps do
        local sigma_curr = schedule[i]
        local sigma_next = (i < ns) and schedule[i + 1] or 0.0

        -- Terminal step: Euler denoise to x0
        if sigma_next == 0.0 then
            local v_final = eval_at(x, sigma_curr)
            for j = 0, n - 1 do x[j] = x[j] - v_final[j] * sigma_curr end
            if verbose then
                print(string.format("[STORM] Step %02d: FINAL (Euler terminal)", i - 1))
            end
            break
        end

        -- Snapshot for look-back (before this step modifies x)
        local x_prev_lb_before = nil
        if lb_enabled then x_prev_lb_before = vec_clone(x, n) end

        -- Evaluate velocity
        local v_curr = eval_at(x, sigma_curr)

        -- Relational decomposition
        local sigma_ratio = clamp(sigma_curr / math.max(sigma_max, EPSILON), 0.0, 1.0)
        if rw > 0 then
            C.apply_relational(v_curr, n, 1, n, sigma_ratio, sigma_max,
                rw, rw_sig_pow, false, 0.85, x)
        end

        local cur_vel_norm = vec_norm(v_curr, n)

        -- Update kinetic floor EMA
        if i == 1 then ema_vel = cur_vel_norm
        else ema_vel = 0.8 * ema_vel + 0.2 * cur_vel_norm end

        -- ── Stiffness detection (deployed STORM) ──
        local stiff, cos_sim_out
        if #v_cache >= 1 then
            stiff, baseline, cos_sim_out = compute_stiffness(
                v_curr, v_cache, i - 1, baseline, stiff_thr, ema_a, n_calib, n)
        else
            stiff, cos_sim_out = true, nil
        end

        -- Hysteresis: prevent rapid STORK↔DPM++ switching
        local prev_mode = baseline.prev_mode or "STORK"
        if prev_mode == "DPM++" and not stiff then
            if (baseline.last_ratio or 0) > (baseline.last_threshold or stiff_thr) + hyst then
                stiff = true
            end
        end

        -- ── Dispatch: STORK (stiff) or DPM++3M (smooth) ──
        local x_next, actual_order, mode
        if stiff then
            x_next, actual_order = stork_step(v_cache, x, sigma_curr, sigma_next, v_curr, rk_order, n)
            mode = "STORK"
        else
            x_next = dpmpp3m_step(v_cache, x, sigma_curr, sigma_next, v_curr, n)
            mode         = "DPM++"
            actual_order = 3
        end

        -- ── Pseudo-LTE estimation (Phase 1 research) ──
        local rel_epsilon, w_t = estimate_pseudo_lte(v_cache, v_curr, ema_vel, k_atten, n)

        -- Asymmetric hysteresis order management
        if rel_epsilon > thr_down then
            current_order = math.max(1, current_order - 1)
            stability_counter = 0
        elseif rel_epsilon < thr_up then
            stability_counter = stability_counter + 1
            if stability_counter >= stab_window then
                current_order = math.min(current_order + 1, 5)
                stability_counter = 0
            end
        else
            stability_counter = 0
        end

        -- Verbose logging
        if verbose then
            local lr  = baseline.last_ratio     or 0.0
            local lt  = baseline.last_threshold or stiff_thr
            local cs  = cos_sim_out and string.format("%.4f", cos_sim_out) or "N/A"
            local tag = (stiff and prev_mode == "DPM++") and " -> CURVATURE SPIKE" or ""
            print(string.format("[STORM] Step %02d: %-5s RK%d | Stiff: %.3f/%.3f | cos: %s | LTE: %.4f w=%.3f ord=%d%s",
                i - 1, mode, actual_order, lr, lt, cs, rel_epsilon, w_t, current_order, tag))
        end

        -- Telemetry
        if do_tele then
            table.insert(telemetry_data, string.format(
                '{"step":%d,"sigma":%.4f,"mode":"%s","rk_order":%d,"stiff_ratio":%.5f,"stiff_thr":%.5f,"cos_sim":%s,"rel_eps":%.5f,"w_t":%.5f,"lte_order":%d,"vel_norm":%.5f}',
                i, sigma_curr, mode, actual_order,
                baseline.last_ratio or 0, baseline.last_threshold or stiff_thr,
                cos_sim_out and string.format("%.5f", cos_sim_out) or "null",
                rel_epsilon, w_t, current_order, cur_vel_norm))
        end

        -- ── NaN guard ──
        if has_nan_inf_tbl(x_next, n) then
            print(string.format("[STORM] NaN/Inf at step %d. Flushing cache, Euler fallback.", i - 1))
            local dt = sigma_next - sigma_curr
            x_next = {}
            for j = 0, n - 1 do x_next[j] = x[j] + dt * v_curr[j] end
            v_cache = {}
            baseline.prev_mode = "STORK"
            current_order = 1
            stability_counter = 0
        end

        -- Update cache
        table.insert(v_cache, { v = v_curr, sigma = sigma_curr })
        while #v_cache > depth_max do table.remove(v_cache, 1) end
        baseline.prev_mode = mode

        x = x_next

        -- ── Look-Back smoothing ──
        if lb_enabled and x_prev_lb ~= nil then
            local lam
            x, lam = look_back_smooth(x, x_prev_lb, sigma_curr, sigma_max, lb_lambda, lb_snr_pow, n)
            if verbose then
                print(string.format("[STORM] LookBack lambda=%.4f @ sigma=%.3f", lam, sigma_curr))
            end
        end
        x_prev_lb = x_prev_lb_before

        -- Write back for on_step hooks (DCW, repaint)
        tbl_to_fa(x, xt, n)
        tbl_to_fa(v_curr, vt_buf, n)

        -- Report step
        if on_step(i - 1, sigma_curr, sigma_next) then return end

        -- Re-read in case hooks modified xt
        x = fa_to_tbl(xt, n)
    end

    -- Write final x0
    tbl_to_fa(x, xt, n)

    if do_tele then
        print("\nSTORM_DATA:[" .. table.concat(telemetry_data, ",") .. "]\n")
    end
end
