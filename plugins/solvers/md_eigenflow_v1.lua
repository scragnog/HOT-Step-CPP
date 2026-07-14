-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD Eigenflow V1 -- PCA Trajectory Filtering Sampler
-- MDMAchine | A&E Concepts (c) 2026
--
-- PCA trajectory filtering via power iteration on velocity history window.
-- Separates dominant denoising direction from oscillatory noise.
-- Euler advance with filtered velocity. owns_loop = true. Single NFE.
-- ============================================================================

local C = require("md_solver_commons")

-- ── POWER ITERATION (per-batch) ─────────────────────────────────────────────

local function power_iteration_batch(window, win_len, off, cnt, M, iters)
    local eigvecs, eigvals = {}, {}

    for m = 1, M do
        local q = {}
        for i = 0, cnt - 1 do q[i] = window[1][off + i] end

        for prev = 1, m - 1 do
            local d = 0.0
            for i = 0, cnt - 1 do d = d + q[i] * eigvecs[prev][i] end
            for i = 0, cnt - 1 do q[i] = q[i] - d * eigvecs[prev][i] end
        end

        for _iter = 1, iters do
            local Cq = {}
            for i = 0, cnt - 1 do Cq[i] = 0.0 end

            for k = 1, win_len do
                local dot = 0.0
                for i = 0, cnt - 1 do dot = dot + window[k][off + i] * q[i] end
                local scale = dot / win_len
                for i = 0, cnt - 1 do Cq[i] = Cq[i] + window[k][off + i] * scale end
            end

            for prev = 1, m - 1 do
                local d = 0.0
                for i = 0, cnt - 1 do d = d + Cq[i] * eigvecs[prev][i] end
                for i = 0, cnt - 1 do Cq[i] = Cq[i] - d * eigvecs[prev][i] end
            end

            local nrm = 0.0
            for i = 0, cnt - 1 do nrm = nrm + Cq[i] * Cq[i] end
            nrm = math.sqrt(nrm + C.EPSILON)
            for i = 0, cnt - 1 do q[i] = Cq[i] / nrm end
        end

        local lam = 0.0
        for k = 1, win_len do
            local dot = 0.0
            for i = 0, cnt - 1 do dot = dot + window[k][off + i] * q[i] end
            lam = lam + dot * dot
        end
        eigvecs[m] = q
        eigvals[m] = lam / win_len
    end

    return eigvecs, eigvals
end

local function filter_velocity_batch(v_curr, off, cnt, eigvecs, M, ratio)
    local projections = {}
    for m = 1, M do
        local dot = 0.0
        for i = 0, cnt - 1 do dot = dot + v_curr[off + i] * eigvecs[m][i] end
        projections[m] = dot
    end

    local filtered = {}
    for i = 0, cnt - 1 do
        local dominant = 0.0
        for m = 1, M do dominant = dominant + projections[m] * eigvecs[m][i] end
        filtered[i] = dominant + ratio * (v_curr[off + i] - dominant)
    end
    return filtered
end

-- ── SOLVER DEFINITION ───────────────────────────────────────────────────────

solver = {
    name        = "md_eigenflow_v1",
    display     = "MD Eigenflow V1",
    description = "PCA trajectory filtering sampler. Power iteration on velocity history, batch-aware, shared anchor stack.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = true,
    owns_loop   = true,
    params      = {
        { key = "window_size", type = "slider", label = "Velocity Window Size",
          default = 6, min = 3, max = 12, step = 1,
          hint = "Velocity snapshots in sliding window." },
        { key = "num_modes", type = "slider", label = "Principal Modes",
          default = 2, min = 1, max = 4, step = 1,
          hint = "Dominant eigenvectors to keep. 1 = aggressive, 3+ = conservative." },
        { key = "power_iterations", type = "slider", label = "Power Iterations",
          default = 4, min = 2, max = 8, step = 1,
          hint = "Convergence iterations for power method." },
        { key = "eigenflow_ratio", type = "slider", label = "Eigenflow Ratio",
          default = 0.3, min = 0.0, max = 1.0, step = 0.05,
          hint = "Residual to keep. 0 = pure dominant mode. 1 = passthrough (Euler)." },
        { key = "sigma_warmup", type = "slider", label = "Sigma Warmup",
          default = 0.85, min = 0.5, max = 1.0, step = 0.05,
          hint = "Sigma fraction above which filtering is disabled." },
        { key = "adaptive_ratio", type = "toggle", label = "Adaptive Ratio",
          default = true,
          hint = "Modulates eigenflow_ratio by dominance ratio." },
    },
}

C.append_common_params(solver.params)

-- ── SAMPLE ──────────────────────────────────────────────────────────────────

function sample(xt, vt_buf, schedule, n, model_fn)
    local p = params or {}
    local B, NPB = C.get_batch_routing(n)

    local win_size     = math.floor(C.num_param(p, "window_size", 6))
    local num_modes    = math.floor(C.num_param(p, "num_modes", 2))
    local pw_iters     = math.floor(C.num_param(p, "power_iterations", 4))
    local ef_ratio     = C.num_param(p, "eigenflow_ratio", 0.3)
    local sigma_warmup = C.num_param(p, "sigma_warmup", 0.85)
    local f_adaptive   = C.bool_param(p, "adaptive_ratio", true)
    local opts         = C.read_common_opts(p)
    local state        = C.new_state()

    -- Engine schedule has NO trailing 0 (fix ported from 46c081e): iterate all ns
    -- entries so the last iteration gets sigma_next = 0.0 and the terminal branch
    -- performs the final x0 projection. With ns - 1 that branch is dead code and
    -- the output keeps ~final-sigma noise.
    local ns, n_steps = #schedule, #schedule
    if n_steps < 1 then return end

    local sigma_max = schedule[1]
    local v_window, v_win_len, v_win_pos = {}, 0, 0
    for k = 1, win_size do v_window[k] = nil end

    local x = C.fa_to_tbl(xt, n)

    if opts.verbose then
        print(string.format("[EIGENFLOW V1] Schedule: %d steps | B=%d NPB=%d n=%d | win=%d modes=%d ratio=%.2f",
            n_steps, B, NPB, n, win_size, num_modes, ef_ratio))
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

        -- Relational decomposition (shape/scale cleanup on velocity)
        if opts.rw > 0 then
            C.apply_relational(v_curr, n, B, NPB, sigma_ratio, sigma_max,
                opts.rw, opts.rw_sigma_pow, opts.drift_on, opts.drift_thr, x)
        end

        -- Push into ring buffer
        v_win_pos = (v_win_pos % win_size) + 1
        v_window[v_win_pos] = C.vec_clone(v_curr, n)
        if v_win_len < win_size then v_win_len = v_win_len + 1 end

        -- Build ordered window
        local win_ordered = {}
        for k = 1, v_win_len do
            local idx = ((v_win_pos - v_win_len + k - 1) % win_size) + 1
            win_ordered[k] = v_window[idx]
        end

        -- Eigenflow filtering
        local v_use = v_curr
        local filtered = false
        local dominance_ratio = 0.0

        if v_win_len >= win_size and sigma_ratio < sigma_warmup then
            local actual_modes = math.min(num_modes, win_size - 1)
            local v_filtered = C.vec_clone(v_curr, n)

            for b = 0, B - 1 do
                local off = b * NPB
                local eigvecs, eigvals = power_iteration_batch(
                    win_ordered, v_win_len, off, NPB, actual_modes, pw_iters)

                if actual_modes >= 2 and eigvals[2] > C.EPSILON then
                    local dr = eigvals[1] / eigvals[2]
                    if dr > dominance_ratio then dominance_ratio = dr end
                end

                local eff_ratio = ef_ratio
                if f_adaptive and dominance_ratio > 1.0 then
                    eff_ratio = ef_ratio * C.clamp(1.0 / math.sqrt(dominance_ratio), 0.1, 1.0)
                end

                local batch_filtered = filter_velocity_batch(
                    v_curr, off, NPB, eigvecs, actual_modes, eff_ratio)
                for j = 0, NPB - 1 do v_filtered[off + j] = batch_filtered[j] end
            end

            if not C.has_nan_inf(v_filtered, n) then
                v_use = v_filtered
                filtered = true
            end
        end

        -- Euler advance
        local x_new = {}
        for j = 0, n - 1 do x_new[j] = x[j] + dt * v_use[j] end

        if C.has_nan_inf(x_new, n) then
            for j = 0, n - 1 do x_new[j] = x[j] + dt * v_curr[j] end
        end

        -- Post-advance stack
        opts.sigma_next = sigma_next
        opts.step_idx   = step_idx
        C.post_advance(x_new, n, B, NPB, sigma_ratio, opts, state)

        if opts.verbose then
            print(string.format("[EIGENFLOW V1] step %02d | %s | dom=%.2f | rms=%.3f",
                step_idx, filtered and "FILTERED" or "raw", dominance_ratio, C.rms(x_new, n)))
        end

        x = x_new
        C.tbl_to_fa(x, xt, n)
        C.tbl_to_fa(v_curr, vt_buf, n)
        if on_step(step_idx, sigma_curr, sigma_next) then return end
        x = C.fa_to_tbl(xt, n)
    end

    C.tbl_to_fa(x, xt, n)
end
