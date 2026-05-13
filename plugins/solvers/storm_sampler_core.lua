--[[
storm_sampler_core.lua
STORM -- Stabilized Taylor Oscillation with Runge-Kutta Memory
Adaptive hybrid solver: STORK (stiff) + DPM++3M (stable), per-step dispatch

© 2026 Alexander Allan (MDMAchine) | A&E Concepts
GPL v3 -- Public version. Gradient norm stiffness detection only.

Adapted for HOT-Step full-loop plugin API (owns_loop = true).
All data uses 0-indexed FloatArray or 0-indexed Lua tables.

Version: 3.0.0  (HOT-Step plugin port from v2.1.0)
--]]

solver = {
    name        = "storm",
    display     = "STORM",
    description = "Adaptive STORK/DPM++3M hybrid with stiffness detection",
    accent      = "cyan",
    nfe         = 0,
    order       = 5,
    needs_model = false,
    stateful    = true,
    stochastic  = false,
    owns_loop   = true,
    params      = {
        { key = "stiffness_threshold", type = "slider", label = "Detail Sensitivity",
          default = 0.15, min = 0.05, max = 0.50, step = 0.01,
          hint = "How aggressively complex passages get extra precision. Lower = more careful on transients and busy sections, higher = faster but looser" },
        { key = "look_back_lambda", type = "slider", label = "Coherence Smoothing",
          default = 0.35, min = 0, max = 1, step = 0.01,
          hint = "Blends each step with previous ones for smoother output. 0 = off (raw), higher = more coherent but softer detail" },
        { key = "look_back_snr_power", type = "slider", label = "Early-Step Focus",
          default = 1.5, min = 0.5, max = 3, step = 0.1,
          hint = "Concentrates smoothing on early noisy steps (structure). Higher = smooths structure more, leaves fine detail alone" },
        { key = "rk_order", type = "select", label = "Precision Level",
          default = "auto",
          options = {
            { value = "auto", label = "Auto (Recommended)" },
            { value = "2",    label = "Low (RK2)" },
            { value = "3",    label = "Medium (RK3)" },
            { value = "4",    label = "High (RK4)" },
            { value = "5",    label = "Maximum (RK5)" },
          },
          hint = "Solver accuracy per step. Auto ramps up gradually. Higher = cleaner but more compute per step" },
        { key = "cache_depth", type = "slider", label = "History Memory",
          default = 5, min = 2, max = 10, step = 1,
          hint = "How many previous steps the solver remembers. More = smoother multi-step blending, but diminishing returns past 5" },
        { key = "verbose", type = "toggle", label = "Verbose Logging",
          default = false,
          hint = "Print per-step solver decisions to the console (debug)" },
    },
}

-- ─────────────────────────────────────────────────────────────────────────────
-- HELPERS: FloatArray ↔ Lua table (0-indexed)
-- ─────────────────────────────────────────────────────────────────────────────

local function fa_to_tbl(fa, n)
    local t = {}
    for i = 0, n - 1 do t[i] = fa[i] end
    return t
end

local function tbl_to_fa(t, fa, n)
    for i = 0, n - 1 do fa[i] = t[i] end
end

-- ─────────────────────────────────────────────────────────────────────────────
-- MATH HELPERS (0-indexed Lua tables)
-- ─────────────────────────────────────────────────────────────────────────────

local function vec_norm(v, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + v[i] * v[i] end
    return math.sqrt(s)
end

local function vec_sub_norm(a, b, n)
    local s = 0.0
    for i = 0, n - 1 do local d = a[i] - b[i]; s = s + d * d end
    return math.sqrt(s)
end

local function vec_dot(a, b, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + a[i] * b[i] end
    return s
end

local function vec_clone(v, n)
    local c = {}
    for i = 0, n - 1 do c[i] = v[i] end
    return c
end

local function has_nan_inf_tbl(v, n)
    for i = 0, n - 1 do
        if v[i] ~= v[i] or math.abs(v[i]) == math.huge then return true end
    end
    return false
end

local function clamp(x, lo, hi) return math.max(lo, math.min(hi, x)) end

local function randn_iso(n, scale)
    local out = {}
    for i = 0, n - 1, 2 do
        local u1 = math.max(1e-12, math.random())
        local u2 = math.random()
        local r  = scale * math.sqrt(-2.0 * math.log(u1))
        out[i]   = r * math.cos(2 * math.pi * u2)
        if i + 1 < n then
            out[i + 1] = r * math.sin(2 * math.pi * u2)
        end
    end
    return out
end

-- ─────────────────────────────────────────────────────────────────────────────
-- LOOK-BACK SMOOTHER (arXiv:2602.09449)
-- ─────────────────────────────────────────────────────────────────────────────

local function look_back_smooth(x_curr, x_prev, sigma_curr, sigma_max, lambda_base, snr_power, n)
    if x_prev == nil then return x_curr, 0.0 end
    local ratio = math.min(sigma_curr / math.max(sigma_max, 1e-8), 1.0)
    local lam   = lambda_base * (ratio ^ snr_power)
    local out   = {}
    for i = 0, n - 1 do out[i] = (1.0 - lam) * x_curr[i] + lam * x_prev[i] end
    return out, lam
end

-- ─────────────────────────────────────────────────────────────────────────────
-- STIFFNESS DETECTION
-- ─────────────────────────────────────────────────────────────────────────────

local function compute_stiffness(v_curr, v_cache, step_idx, baseline, threshold, ema_alpha, n_calib, n)
    threshold = threshold or 0.15
    ema_alpha = ema_alpha or 0.3
    n_calib   = n_calib   or 4

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
-- STORK MULTI-ORDER (AB2-AB5, single NFE, cached derivatives)
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

    if n_cache < 1 or actual_order <= 1 then
        local x_next = {}
        for i = 0, n - 1 do x_next[i] = x[i] + dt * v_curr[i] end
        return x_next, 1
    end

    local e0         = v_cache[#v_cache]
    local v_prev_0   = e0.v
    local sigma_prev = e0.sigma

    -- Curvature damping
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
            local c0 = 1.0 + dt / (2.0 * h) + dt ^ 2 / (3.0 * h * h1) + dt ^ 3 / (4.0 * h * h1 * h2)
            local c1 = -(dt / (2.0 * h)) * (1.0 + dt / h1 + dt ^ 2 / (2.0 * h1 * h2))
            local c2 = (dt ^ 2 / (3.0 * h * h1)) * (1.0 + dt / (2.0 * h2))
            local c3 = -(dt ^ 3) / (4.0 * h * h1 * h2)
            for i = 0, n - 1 do
                local vp = c0 * v_curr[i] + c1 * v1[i] + c2 * v2[i] + c3 * v3[i]
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
            local c0 = 1.0 + dt / (2.0 * h) + dt ^ 2 / (3.0 * h * h1) + dt ^ 3 / (4.0 * h * h1 * h2)
            local c1 = -(dt / (2.0 * h)) * (1.0 + dt / h1 + dt ^ 2 / (2.0 * h1 * h2))
            local c2 = (dt ^ 2 / (3.0 * h * h1)) * (1.0 + dt / (2.0 * h2))
            local c3 = -(dt ^ 3) / (4.0 * h * h1 * h2)
            for i = 0, n - 1 do
                local vp = c0 * v_curr[i] + c1 * v1[i] + c2 * v2[i] + c3 * v3[i]
                x_next[i] = x[i] + dt * (v_curr[i] + damping * (vp - v_curr[i]))
            end
            actual_order = 4
        else
            local c0 = 1.0 + dt / (2.0 * h) + dt ^ 2 / (3.0 * h * h1) + dt ^ 3 / (4.0 * h * h1 * h2) + dt ^ 4 / (5.0 * h * h1 * h2 * h3)
            local c1 = -(dt / (2.0 * h)) * (1.0 + dt / h1 + dt ^ 2 / (2.0 * h1 * h2) + dt ^ 3 / (3.0 * h1 * h2 * h3))
            local c2 = (dt ^ 2 / (3.0 * h * h1)) * (1.0 + dt / (2.0 * h2) + dt ^ 2 / (3.0 * h2 * h3))
            local c3 = -(dt ^ 3 / (4.0 * h * h1 * h2)) * (1.0 + dt / (2.0 * h3))
            local c4 = dt ^ 4 / (5.0 * h * h1 * h2 * h3)
            for i = 0, n - 1 do
                local vp = c0 * v_curr[i] + c1 * v1[i] + c2 * v2[i] + c3 * v3[i] + c4 * v4[i]
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
-- DPM++3M -- smooth schedule path
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
-- SAMPLE — Full-loop entry point
-- ─────────────────────────────────────────────────────────────────────────────

function sample(xt, vt_buf, schedule, n, model_fn)
    -- Read params
    local p = params or {}
    local thr           = p.stiffness_threshold or 0.15
    local lb_lambda     = p.look_back_lambda    or 0.35
    local lb_snr_pow    = p.look_back_snr_power or 1.5
    local rk_order      = p.rk_order            or "auto"
    local depth_max     = p.cache_depth         or 5
    local verbose       = p.verbose             or false

    local hyst      = 0.05
    local ema_a     = 0.3
    local calib_frac = 0.12

    local ns        = #schedule
    local n_steps   = ns - 1
    if n_steps < 1 then return end

    local v_cache   = {}
    local baseline  = { sum = 0.0, count = 0 }
    local sigma_max = schedule[1]
    local n_calib   = math.max(2, math.min(5, math.floor(n_steps * calib_frac)))
    local lb_enabled = (lb_lambda > 0)

    if verbose then
        print(string.format("[STORM] Schedule: %d steps | Calib: %d | RK: %s | Cache: %d",
            n_steps, n_calib, tostring(rk_order), depth_max))
    end

    -- Working copy of xt as a Lua table (we write back to FloatArray at each step)
    local x = fa_to_tbl(xt, n)

    -- Seed x_prev for look-back
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

    -- Helper: evaluate model and return velocity as Lua table
    local function eval_model(x_tbl)
        tbl_to_fa(x_tbl, xt, n)
        model_fn(xt, 0)  -- dummy t, we'll set it properly below
        return fa_to_tbl(vt_buf, n)
    end

    -- Proper eval: writes x_tbl to xt, calls model at t_val, returns velocity table
    local function eval_at(x_tbl, t_val)
        tbl_to_fa(x_tbl, xt, n)
        model_fn(xt, t_val)
        return fa_to_tbl(vt_buf, n)
    end

    for i = 1, n_steps do
        local sigma_curr = schedule[i]
        local sigma_next = (i < ns) and schedule[i + 1] or 0.0

        -- Terminal step
        if sigma_next == 0.0 then
            local v_final = eval_at(x, sigma_curr)
            for j = 0, n - 1 do x[j] = x[j] - v_final[j] * sigma_curr end
            if verbose then
                print(string.format("[STORM] Step %02d: FINAL (Euler terminal)", i - 1))
            end
            break
        end

        local x_prev_lb_before = nil
        if lb_enabled then x_prev_lb_before = vec_clone(x, n) end

        -- Evaluate velocity
        local v_curr = eval_at(x, sigma_curr)

        -- Stiffness detection
        local stiff, cos_sim_out
        if #v_cache >= 1 then
            stiff, baseline, cos_sim_out = compute_stiffness(
                v_curr, v_cache, i - 1, baseline, thr, ema_a, n_calib, n)
        else
            stiff, cos_sim_out = true, nil
        end

        -- Hysteresis
        local prev_mode = baseline.prev_mode or "STORK"
        if prev_mode == "DPM++" and not stiff then
            if (baseline.last_ratio or 0) > (baseline.last_threshold or thr) + hyst then
                stiff = true
            end
        end

        -- Dispatch
        local x_next, actual_order, mode
        if stiff then
            x_next, actual_order = stork_step(v_cache, x, sigma_curr, sigma_next, v_curr, rk_order, n)
            mode = "STORK"
        else
            x_next = dpmpp3m_step(v_cache, x, sigma_curr, sigma_next, v_curr, n)
            mode         = "DPM++"
            actual_order = 3
        end

        -- Verbose
        if verbose then
            local lr  = baseline.last_ratio     or 0.0
            local lt  = baseline.last_threshold or thr
            local cs  = cos_sim_out and string.format("%.4f", cos_sim_out) or "N/A"
            local tag = (stiff and prev_mode == "DPM++") and " -> CURVATURE SPIKE" or ""
            print(string.format("[STORM] Step %02d: %-5s RK%d | Ratio: %.3f | Thr: %.3f | cos: %s%s",
                i - 1, mode, actual_order, lr, lt, cs, tag))
        end

        -- NaN guard
        if has_nan_inf_tbl(x_next, n) then
            print(string.format("[STORM] NaN/Inf at step %d. Flushing cache.", i - 1))
            local dt = sigma_next - sigma_curr
            x_next = {}
            for j = 0, n - 1 do x_next[j] = x[j] + dt * v_curr[j] end
            v_cache = {}
            baseline.prev_mode = "STORK"
            actual_order = 1
        end

        -- Update cache
        table.insert(v_cache, { v = v_curr, sigma = sigma_curr })
        while #v_cache > depth_max do table.remove(v_cache, 1) end
        baseline.prev_mode = mode

        x = x_next

        -- Look-Back smoothing
        if lb_enabled and x_prev_lb ~= nil then
            local lam
            x, lam = look_back_smooth(x, x_prev_lb, sigma_curr, sigma_max, lb_lambda, lb_snr_pow, n)
            if verbose then
                print(string.format("[STORM] LookBack λ=%.4f @ σ=%.3f", lam, sigma_curr))
            end
        end
        x_prev_lb = x_prev_lb_before

        -- Write x back to xt FloatArray for on_step hooks (DCW, repaint)
        tbl_to_fa(x, xt, n)
        -- Write velocity to vt_buf for DCW
        tbl_to_fa(v_curr, vt_buf, n)

        -- Report step (engine hooks: DCW, repaint, progress)
        if on_step(i - 1, sigma_curr, sigma_next) then return end

        -- Re-read xt in case hooks modified it (DCW, repaint)
        x = fa_to_tbl(xt, n)
    end

    -- Write final x0 to xt
    tbl_to_fa(x, xt, n)
end
