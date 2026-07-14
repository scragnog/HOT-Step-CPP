-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD Vortex V1 -- Vorticity Damping Sampler
-- MDMAchine | A&E Concepts (c) 2026
--
-- Multi-scale curl proxy detects rotational velocity energy. Enstrophy EMA
-- triggers targeted vortex shedding (direct damping of vortical component).
-- Euler advance with cleaned velocity. owns_loop = true. Single NFE.
-- ============================================================================

local C = require("md_solver_commons")

-- ── CURL PROXY (per-batch) ──────────────────────────────────────────────────

local function compute_curl_proxy(v_curr, v_prev, off, cnt, multi_scale)
    local curl = {}
    for i = 0, cnt - 1 do curl[i] = 0.0 end

    for i = 1, cnt - 2 do
        local grad_curr = (v_curr[off + i + 1] - v_curr[off + i - 1]) * 0.5
        local grad_prev = (v_prev[off + i + 1] - v_prev[off + i - 1]) * 0.5
        curl[i] = curl[i] + (grad_curr - grad_prev)
    end

    if multi_scale then
        for i = 2, cnt - 3 do
            local grad_curr = (v_curr[off + i + 2] - v_curr[off + i - 2]) * 0.25
            local grad_prev = (v_prev[off + i + 2] - v_prev[off + i - 2]) * 0.25
            curl[i] = curl[i] + 0.5 * (grad_curr - grad_prev)
        end
        for i = 4, cnt - 5 do
            local grad_curr = (v_curr[off + i + 4] - v_curr[off + i - 4]) * 0.125
            local grad_prev = (v_prev[off + i + 4] - v_prev[off + i - 4]) * 0.125
            curl[i] = curl[i] + 0.25 * (grad_curr - grad_prev)
        end
    end

    local enstrophy, curl_max = 0.0, 0.0
    for i = 0, cnt - 1 do
        enstrophy = enstrophy + curl[i] * curl[i]
        local ac = math.abs(curl[i])
        if ac > curl_max then curl_max = ac end
    end
    return curl, enstrophy / math.max(cnt, 1), curl_max
end

local function apply_shedding(v_out, off, cnt, curl, enstrophy_ema,
                               threshold, strength, progressive)
    if strength < 1e-6 or enstrophy_ema <= threshold then return false end

    local eff_strength = strength
    if progressive then
        local overshoot = C.clamp((enstrophy_ema - threshold) / (threshold + C.EPSILON), 0.0, 2.0)
        eff_strength = strength * (overshoot / 2.0)
    end
    if eff_strength < 1e-6 then return false end

    local correction = {}
    correction[0] = 0.0
    for i = 1, cnt - 1 do correction[i] = correction[i - 1] + curl[i] end

    local corr_energy = 0.0
    for i = 0, cnt - 1 do corr_energy = corr_energy + correction[i] * correction[i] end
    corr_energy = math.sqrt(corr_energy / math.max(cnt, 1) + C.EPSILON)

    local scale = eff_strength / (corr_energy + C.EPSILON)
    local v_rms = 0.0
    for i = 0, cnt - 1 do v_rms = v_rms + v_out[off + i] * v_out[off + i] end
    v_rms = math.sqrt(v_rms / math.max(cnt, 1) + C.EPSILON)
    local max_corr = 0.05 * v_rms

    for i = 0, cnt - 1 do
        v_out[off + i] = v_out[off + i] - C.clamp(correction[i] * scale, -max_corr, max_corr)
    end
    return true
end

-- ── SOLVER DEFINITION ───────────────────────────────────────────────────────

solver = {
    name        = "md_vortex_v1",
    display     = "MD Vortex V1",
    description = "Vorticity damping sampler. Multi-scale curl proxy, enstrophy tracking, targeted shedding. Batch-aware, shared anchor stack.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = true,
    owns_loop   = true,
    params      = {
        { key = "shedding_strength", type = "slider", label = "Shedding Strength",
          default = 0.30, min = 0.0, max = 1.0, step = 0.05,
          hint = "Vortical energy damping. 0 = monitor only." },
        { key = "enstrophy_threshold", type = "slider", label = "Enstrophy Threshold",
          default = 0.02, min = 0.001, max = 0.2, step = 0.001,
          hint = "EMA level triggering shedding. Calibrate with verbose=true." },
        { key = "enstrophy_ema_alpha", type = "slider", label = "Enstrophy EMA Alpha",
          default = 0.1, min = 0.02, max = 0.5, step = 0.02,
          hint = "Tracker responsiveness." },
        { key = "multi_scale", type = "toggle", label = "Multi-Scale Curl",
          default = true, hint = "Adjacent + skip-2 + skip-4 gradient changes." },
        { key = "sigma_gate", type = "slider", label = "Sigma Gate",
          default = 0.9, min = 0.5, max = 1.0, step = 0.05,
          hint = "Sigma fraction above which shedding is disabled." },
        { key = "progressive_shedding", type = "toggle", label = "Progressive Shedding",
          default = true, hint = "Strength scales with enstrophy overshoot." },
    },
}

C.append_common_params(solver.params)

-- ── SAMPLE ──────────────────────────────────────────────────────────────────

function sample(xt, vt_buf, schedule, n, model_fn)
    local p = params or {}
    local B, NPB = C.get_batch_routing(n)

    local shed_str     = C.num_param(p, "shedding_strength", 0.30)
    local enst_thresh  = C.num_param(p, "enstrophy_threshold", 0.02)
    local enst_alpha   = C.num_param(p, "enstrophy_ema_alpha", 0.1)
    local f_multi      = C.bool_param(p, "multi_scale", true)
    local sigma_gate   = C.num_param(p, "sigma_gate", 0.9)
    local f_prog       = C.bool_param(p, "progressive_shedding", true)
    local opts         = C.read_common_opts(p)
    local state        = C.new_state()

    -- Engine schedule has NO trailing 0 (fix ported from 46c081e): iterate all ns
    -- entries so the last iteration gets sigma_next = 0.0 and the terminal branch
    -- performs the final x0 projection. With ns - 1 that branch is dead code and
    -- the output keeps ~final-sigma noise.
    local ns, n_steps = #schedule, #schedule
    if n_steps < 1 then return end

    local sigma_max = schedule[1]
    local v_prev = nil
    local enst_ema = {}
    for b = 0, B - 1 do enst_ema[b] = 0.0 end

    local x = C.fa_to_tbl(xt, n)

    if opts.verbose then
        print(string.format("[VORTEX V1] Schedule: %d steps | B=%d NPB=%d | shed=%.2f thresh=%.4f",
            n_steps, B, NPB, shed_str, enst_thresh))
    end

    for i = 1, n_steps do
        local sigma_curr  = schedule[i]
        local sigma_next  = (i < ns) and schedule[i + 1] or 0.0
        local step_idx    = i - 1
        local sigma_ratio = C.clamp(sigma_curr / math.max(sigma_max, C.EPSILON), 0.0, 1.0)

        if sigma_next == 0.0 then
            C.tbl_to_fa(x, xt, n)
            model_fn(xt, sigma_curr)
            local v_final = C.fa_to_tbl(vt_buf, n)
            for j = 0, n - 1 do x[j] = x[j] - v_final[j] * sigma_curr end
            break
        end

        C.tbl_to_fa(x, xt, n)
        model_fn(xt, sigma_curr)
        local v_curr = C.fa_to_tbl(vt_buf, n)
        local dt = sigma_next - sigma_curr

        -- Relational decomposition
        if opts.rw > 0 then
            C.apply_relational(v_curr, n, B, NPB, sigma_ratio, sigma_max,
                opts.rw, opts.rw_sigma_pow, opts.drift_on, opts.drift_thr, x)
        end

        local shedding_active, max_enst, max_curl = false, 0.0, 0.0

        if v_prev ~= nil and sigma_ratio < sigma_gate then
            local v_shed = C.vec_clone(v_curr, n)
            for b = 0, B - 1 do
                local off = b * NPB
                local curl, enstrophy, curl_max = compute_curl_proxy(v_curr, v_prev, off, NPB, f_multi)
                enst_ema[b] = (1.0 - enst_alpha) * enst_ema[b] + enst_alpha * enstrophy
                if enst_ema[b] > max_enst then max_enst = enst_ema[b] end
                if curl_max > max_curl then max_curl = curl_max end
                if apply_shedding(v_shed, off, NPB, curl, enst_ema[b], enst_thresh, shed_str, f_prog) then
                    shedding_active = true
                end
            end
            if not C.has_nan_inf(v_shed, n) then v_curr = v_shed end
        end

        v_prev = C.vec_clone(v_curr, n)

        local x_new = {}
        for j = 0, n - 1 do x_new[j] = x[j] + dt * v_curr[j] end
        if C.has_nan_inf(x_new, n) then
            local v_raw = C.fa_to_tbl(vt_buf, n)
            for j = 0, n - 1 do x_new[j] = x[j] + dt * v_raw[j] end
        end

        opts.sigma_next = sigma_next
        opts.step_idx   = step_idx
        C.post_advance(x_new, n, B, NPB, sigma_ratio, opts, state)

        if opts.verbose then
            print(string.format("[VORTEX V1] step %02d | enst_ema=%.5f %s | curl_max=%.4f | rms=%.3f",
                step_idx, enst_ema[0] or 0, shedding_active and "SHEDDING" or "quiet",
                max_curl, C.rms(x_new, n)))
        end

        x = x_new
        C.tbl_to_fa(x, xt, n)
        C.tbl_to_fa(v_curr, vt_buf, n)
        if on_step(step_idx, sigma_curr, sigma_next) then return end
        x = C.fa_to_tbl(xt, n)
    end

    C.tbl_to_fa(x, xt, n)
end
