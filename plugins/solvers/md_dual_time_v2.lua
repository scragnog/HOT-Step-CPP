-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD Dual-Time V2 -- Inner Convergence Loop Sampler
-- MDMAchine | A&E Concepts (c) 2026
--
-- Zero-NFE iterative refinement using cached (v, sigma, x) tuples.
-- Inverse-distance + sigma-proximity velocity interpolation. Sigma-adaptive
-- inner blend protects early vocal separation. owns_loop = true. Single NFE.
-- ============================================================================

local C = require("md_solver_commons")

-- ── VELOCITY INTERPOLATION (per-batch) ──────────────────────────────────────

local function interpolate_velocity_batch(x_cand, off, cnt, cache, cache_len, sigma_curr, sigma_wt)
    local v_interp = {}
    for i = 0, cnt - 1 do v_interp[i] = 0.0 end
    local total_weight = 0.0

    for k = 1, cache_len do
        local entry = cache[k]
        local dist_sq = 0.0
        for i = 0, cnt - 1 do
            local d = x_cand[off + i] - entry.x[off + i]
            dist_sq = dist_sq + d * d
        end
        local pos_dist = math.sqrt(dist_sq / math.max(cnt, 1) + C.EPSILON)
        local sig_factor = 1.0 / (1.0 + sigma_wt * math.abs(sigma_curr - entry.sigma))
        local w = sig_factor / (pos_dist + C.EPSILON)
        total_weight = total_weight + w
        for i = 0, cnt - 1 do v_interp[i] = v_interp[i] + w * entry.v[off + i] end
    end

    if total_weight > C.EPSILON then
        local inv_w = 1.0 / total_weight
        for i = 0, cnt - 1 do v_interp[i] = v_interp[i] * inv_w end
    end
    return v_interp
end

-- ── INNER CONVERGENCE LOOP ──────────────────────────────────────────────────

local function inner_loop(x_start, x_euler, v_curr, dt, n, B, NPB,
                           cache, cache_len, sigma_curr, sigma_wt,
                           max_inner, conv_thresh, relaxation)
    local x_cand = C.vec_clone(x_euler, n)
    local init_resid, final_resid = 0.0, 0.0
    local converged = false
    local iters_used = 0

    for k = 1, max_inner do
        iters_used = k

        local v_interp_full = C.vec_clone(v_curr, n)
        for b = 0, B - 1 do
            local off = b * NPB
            local v_batch = interpolate_velocity_batch(
                x_cand, off, NPB, cache, cache_len, sigma_curr, sigma_wt)
            for j = 0, NPB - 1 do v_interp_full[off + j] = v_batch[j] end
        end

        local x_new = {}
        for j = 0, n - 1 do x_new[j] = x_start[j] + dt * v_interp_full[j] end

        local corr_rms = 0.0
        for j = 0, n - 1 do
            local c = x_new[j] - x_cand[j]
            corr_rms = corr_rms + c * c
        end
        corr_rms = math.sqrt(corr_rms / math.max(n, 1))

        if k == 1 then init_resid = corr_rms end
        final_resid = corr_rms

        for j = 0, n - 1 do
            x_cand[j] = x_cand[j] + relaxation * (x_new[j] - x_cand[j])
        end

        if C.has_nan_inf(x_cand, n) then
            for j = 0, n - 1 do x_cand[j] = x_euler[j] end
            break
        end

        if corr_rms < conv_thresh then converged = true; break end
    end

    return x_cand, iters_used, converged, init_resid, final_resid
end

-- ── SOLVER DEFINITION ───────────────────────────────────────────────────────

solver = {
    name        = "md_dual_time_v2",
    display     = "MD Dual-Time V2",
    description = "Inner convergence loop sampler. Zero-NFE velocity history interpolation. Sigma-adaptive inner blend. Batch-aware, shared anchor stack.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = true,
    owns_loop   = true,
    params      = {
        { key = "max_inner", type = "slider", label = "Max Inner Iterations",
          default = 3, min = 1, max = 10, step = 1,
          hint = "Pseudo-time iterations per step." },
        { key = "cache_depth", type = "slider", label = "Cache Depth",
          default = 6, min = 2, max = 12, step = 1,
          hint = "Number of (v, sigma, x) tuples stored." },
        { key = "convergence_threshold", type = "slider", label = "Convergence Threshold",
          default = 0.005, min = 0.0005, max = 0.1, step = 0.0005,
          hint = "Per-element RMS for early exit." },
        { key = "sigma_weight", type = "slider", label = "Sigma Proximity Weight",
          default = 4.0, min = 0.0, max = 8.0, step = 0.25,
          hint = "Sigma proximity influence. Higher = less phase ghosting." },
        { key = "relaxation", type = "slider", label = "Relaxation Factor",
          default = 0.45, min = 0.1, max = 1.0, step = 0.05,
          hint = "Inner loop step size. Lower = less phase ghosting." },
        { key = "sigma_gate", type = "slider", label = "Sigma Gate",
          default = 0.9, min = 0.5, max = 1.0, step = 0.05,
          hint = "Sigma fraction above which inner loop is disabled." },
        { key = "inner_blend", type = "slider", label = "Inner Blend",
          default = 0.4, min = 0.0, max = 1.0, step = 0.05,
          hint = "Max Euler/converged mix. Sigma-adaptive: near-zero early, ramps quadratically." },
    },
}

C.append_common_params(solver.params)

-- ── SAMPLE ──────────────────────────────────────────────────────────────────

function sample(xt, vt_buf, schedule, n, model_fn)
    local p = params or {}
    local B, NPB = C.get_batch_routing(n)

    local max_inner    = math.floor(C.num_param(p, "max_inner", 3))
    local cache_depth  = math.floor(C.num_param(p, "cache_depth", 6))
    local conv_thresh  = C.num_param(p, "convergence_threshold", 0.005)
    local sigma_wt     = C.num_param(p, "sigma_weight", 4.0)
    local relaxation   = C.num_param(p, "relaxation", 0.45)
    local sigma_gate   = C.num_param(p, "sigma_gate", 0.9)
    local inner_blend  = C.num_param(p, "inner_blend", 0.4)
    local opts         = C.read_common_opts(p)
    local state        = C.new_state()

    -- Engine schedule has NO trailing 0 (fix ported from 46c081e): iterate all ns
    -- entries so the last iteration gets sigma_next = 0.0 and the terminal branch
    -- performs the final x0 projection. With ns - 1 that branch is dead code and
    -- the output keeps ~final-sigma noise.
    local ns, n_steps = #schedule, #schedule
    if n_steps < 1 then return end

    local sigma_max = schedule[1]
    local cache, cache_len, cache_pos, cache_max = {}, 0, 0, cache_depth
    for k = 1, cache_depth do cache[k] = nil end

    local x = C.fa_to_tbl(xt, n)

    if opts.verbose then
        print(string.format("[DUAL-TIME V2] Schedule: %d steps | B=%d NPB=%d | inner=%d cache=%d",
            n_steps, B, NPB, max_inner, cache_depth))
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

        local x_euler = {}
        for j = 0, n - 1 do x_euler[j] = x[j] + dt * v_curr[j] end

        local x_new = x_euler
        local iters_used, converged, init_resid, final_resid = 0, false, 0.0, 0.0

        if cache_len >= 2 and sigma_ratio < sigma_gate then
            local cache_ordered = {}
            for k = 1, cache_len do
                local idx = ((cache_pos - cache_len + k - 1) % cache_max) + 1
                cache_ordered[k] = cache[idx]
            end

            x_new, iters_used, converged, init_resid, final_resid = inner_loop(
                x, x_euler, v_curr, dt, n, B, NPB,
                cache_ordered, cache_len, sigma_curr, sigma_wt,
                max_inner, conv_thresh, relaxation)

            -- Sigma-adaptive inner blend
            local blend_ramp = (1.0 - sigma_ratio) * (1.0 - sigma_ratio)
            local eff_blend = inner_blend * blend_ramp
            if eff_blend > 1e-6 and eff_blend < 1.0 - 1e-6 then
                for j = 0, n - 1 do
                    x_new[j] = (1.0 - eff_blend) * x_euler[j] + eff_blend * x_new[j]
                end
            elseif eff_blend <= 1e-6 then
                for j = 0, n - 1 do x_new[j] = x_euler[j] end
            end
        end

        -- Cache push
        cache_pos = (cache_pos % cache_max) + 1
        cache[cache_pos] = { v = C.vec_clone(v_curr, n), sigma = sigma_curr, x = C.vec_clone(x, n) }
        if cache_len < cache_max then cache_len = cache_len + 1 end

        if C.has_nan_inf(x_new, n) then
            for j = 0, n - 1 do x_new[j] = x_euler[j] end
        end

        opts.sigma_next = sigma_next
        opts.step_idx   = step_idx
        C.post_advance(x_new, n, B, NPB, sigma_ratio, opts, state)

        if opts.verbose then
            print(string.format("[DUAL-TIME V2] step %02d | inner=%d/%d %s | resid %.5f->%.5f | rms=%.3f",
                step_idx, iters_used, max_inner,
                converged and "CONV" or (iters_used > 0 and "max" or "skip"),
                init_resid, final_resid, C.rms(x_new, n)))
        end

        x = x_new
        C.tbl_to_fa(x, xt, n)
        C.tbl_to_fa(v_curr, vt_buf, n)
        if on_step(step_idx, sigma_curr, sigma_next) then return end
        x = C.fa_to_tbl(xt, n)
    end

    C.tbl_to_fa(x, xt, n)
end
