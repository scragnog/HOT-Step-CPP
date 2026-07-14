-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD Hamiltonian V2 -- Energy-Conserving Momentum-Augmented Sampler
-- MDMAchine | A&E Concepts (c) 2026
--
-- Euler-primary architecture with momentum correction layer, sigma-adaptive
-- decay, confidence gating, spectral momentum, Hamiltonian energy tracking.
-- Two look-backs (primary + post-step). owns_loop = true. Single NFE.
-- ============================================================================

local C = require("md_solver_commons")

-- ── HAMILTONIAN ENERGY ──────────────────────────────────────────────────────

local function kinetic_energy(p, mass, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + p[i] * p[i] end
    return 0.5 * s / mass
end

local function potential_energy(x, v_curr, sigma_ratio, n)
    return -C.vec_dot(v_curr, x, n) * sigma_ratio
end

-- ── SOLVER DEFINITION ───────────────────────────────────────────────────────

solver = {
    name        = "md_hamiltonian_v2",
    display     = "MD Hamiltonian V2",
    description = "Energy-conserving momentum-augmented sampler. Euler + momentum correction, spectral weighting, Hamiltonian tracking. Shared anchor stack.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = true,
    owns_loop   = true,
    params      = {
        -- Momentum
        { key = "momentum_weight", type = "slider", label = "Momentum Weight",
          default = 0.20, min = 0.0, max = 0.8, step = 0.05,
          hint = "Momentum blend. 0 = pure Euler. Scaled by confidence gating and sigma fadeout." },
        { key = "momentum_decay", type = "slider", label = "Momentum Decay",
          default = 0.85, min = 0.0, max = 0.99, step = 0.01,
          hint = "Step-to-step carry-over. Sigma-adaptive." },
        { key = "momentum_ema_alpha", type = "slider", label = "Momentum EMA Alpha",
          default = 0.3, min = 0.05, max = 0.8, step = 0.05,
          hint = "Velocity absorption rate. Sigma-adaptive." },
        { key = "mass", type = "slider", label = "Particle Mass",
          default = 1.0, min = 0.1, max = 5.0, step = 0.1,
          hint = "Inertial mass." },
        -- Energy
        { key = "energy_tolerance", type = "slider", label = "Energy Tolerance",
          default = 0.05, min = 0.005, max = 0.5, step = 0.005,
          hint = "Hamiltonian drift before Metropolis correction." },
        { key = "correction_strength", type = "slider", label = "Correction Strength",
          default = 0.7, min = 0.0, max = 1.0, step = 0.05,
          hint = "Metropolis momentum rescale. 0 = monitor only." },
        { key = "energy_tracking", type = "select", label = "Energy Tracking",
          default = "adaptive",
          options = {
            { value = "fixed",    label = "Fixed" },
            { value = "adaptive", label = "Adaptive" },
            { value = "monitor",  label = "Monitor Only" },
          },
          hint = "How H reference evolves." },
        -- Confidence
        { key = "confidence_floor", type = "slider", label = "Confidence Floor",
          default = 0.2, min = 0.0, max = 0.8, step = 0.05, hint = "Min alignment for momentum." },
        { key = "confidence_ceiling", type = "slider", label = "Confidence Ceiling",
          default = 0.7, min = 0.3, max = 1.0, step = 0.05, hint = "Full momentum alignment." },
        -- Spectral momentum
        { key = "spectral_momentum", type = "toggle", label = "Spectral Momentum",
          default = true, hint = "Per-batch 4-band momentum weighting." },
        { key = "spectral_hi_boost", type = "slider", label = "Spectral HF Boost",
          default = 1.4, min = 1.0, max = 4.0, step = 0.1, hint = "HF momentum multiplier." },
        { key = "spectral_mid_cut", type = "slider", label = "Spectral Mid Cut",
          default = 0.6, min = 0.1, max = 1.0, step = 0.05, hint = "Mid momentum multiplier." },
        -- Post-step look-back (secondary)
        { key = "post_look_back", type = "slider", label = "Post-Step Look-Back",
          default = 0.0, min = 0.0, max = 0.6, step = 0.05, hint = "Additional SNR-adaptive EMA. 0 = off (default)." },
        { key = "post_look_back_snr", type = "slider", label = "Post-Step LB SNR Power",
          default = 1.0, min = 0.5, max = 3.0, step = 0.1, hint = "Falloff." },
    },
}

C.append_common_params(solver.params)

-- ── SAMPLE ──────────────────────────────────────────────────────────────────

function sample(xt, vt_buf, schedule, n, model_fn)
    local p = params or {}
    local B, NPB = C.get_batch_routing(n)

    local mom_weight    = C.num_param(p, "momentum_weight", 0.20)
    local mom_decay     = C.num_param(p, "momentum_decay", 0.85)
    local mom_alpha     = C.num_param(p, "momentum_ema_alpha", 0.3)
    local mass          = C.num_param(p, "mass", 1.0)
    local energy_tol    = C.num_param(p, "energy_tolerance", 0.05)
    local corr_str      = C.num_param(p, "correction_strength", 0.7)
    local energy_mode   = p.energy_tracking or "adaptive"
    local conf_floor    = C.num_param(p, "confidence_floor", 0.2)
    local conf_ceil     = C.num_param(p, "confidence_ceiling", 0.7)
    local f_spec_mom    = C.bool_param(p, "spectral_momentum", true)
    local spec_hi       = C.num_param(p, "spectral_hi_boost", 1.4)
    local spec_mid      = C.num_param(p, "spectral_mid_cut", 0.6)
    local post_lb_lam   = C.num_param(p, "post_look_back", 0.0)
    local post_lb_snr   = C.num_param(p, "post_look_back_snr", 1.0)
    local opts          = C.read_common_opts(p)
    local state         = C.new_state()

    -- Engine schedule has NO trailing 0 (fix ported from 46c081e): iterate all ns
    -- entries so the last iteration gets sigma_next = 0.0 and the terminal branch
    -- performs the final x0 projection. With ns - 1 that branch is dead code and
    -- the output keeps ~final-sigma noise.
    local ns, n_steps = #schedule, #schedule
    if n_steps < 1 then return end

    local sigma_max = schedule[1]
    local momentum = nil
    local H_ref = nil
    local post_lb_enabled = (post_lb_lam > 0)

    local x = C.fa_to_tbl(xt, n)

    if opts.verbose then
        print(string.format("[HAMILTONIAN V2] Schedule: %d steps | B=%d NPB=%d | weight=%.2f decay=%.2f mass=%.1f",
            n_steps, B, NPB, mom_weight, mom_decay, mass))
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

        -- 1. Euler advance
        local x_euler = {}
        for j = 0, n - 1 do x_euler[j] = x[j] + dt * v_curr[j] end

        -- 2. Momentum update (sigma-adaptive)
        local decay_power    = 1.0 + 2.0 * (1.0 - sigma_ratio)
        local effective_decay = mom_decay ^ decay_power
        local effective_alpha = mom_alpha + (1.0 - mom_alpha) * 0.5 * (1.0 - sigma_ratio)

        if momentum == nil then
            momentum = {}
            for j = 0, n - 1 do momentum[j] = v_curr[j] * mass end
        else
            for j = 0, n - 1 do momentum[j] = momentum[j] * effective_decay end
            for j = 0, n - 1 do
                momentum[j] = (1.0 - effective_alpha) * momentum[j] + effective_alpha * v_curr[j] * mass
            end
        end

        -- 3. Momentum-predicted position
        local x_mom = {}
        local inv_mass = 1.0 / mass
        for j = 0, n - 1 do x_mom[j] = x[j] + dt * momentum[j] * inv_mass end

        -- 4. Confidence gating + sigma fadeout (linear)
        local mom_norm = C.vec_norm(momentum, n)
        local v_norm   = C.vec_norm(v_curr, n)
        local alignment = 0.0
        if mom_norm > C.EPSILON and v_norm > C.EPSILON then
            alignment = C.vec_dot(momentum, v_curr, n) / (mom_norm * v_norm)
        end
        local confidence = C.smoothstep(alignment, conf_floor, conf_ceil)
        local sigma_fade = sigma_ratio
        local eff_weight = mom_weight * confidence * sigma_fade

        -- Re-alignment when fighting
        if mom_norm > C.EPSILON and v_norm > C.EPSILON and alignment < 0.3 then
            local blend = 0.3 * (1.0 - alignment)
            for j = 0, n - 1 do
                momentum[j] = (1.0 - blend) * momentum[j] + blend * v_curr[j] * mass * math.abs(dt)
            end
        end

        -- 5. Blend (per-batch spectral awareness)
        local x_new = {}
        if f_spec_mom and eff_weight > 1e-6 then
            local band_mults = { 1.0, spec_mid, spec_mid, spec_hi }
            local bsize = math.floor(NPB / 4)
            for j = 0, n - 1 do
                local local_idx = j % NPB
                local band = math.min(math.floor(local_idx / bsize), 3)
                local local_w = C.clamp(eff_weight * band_mults[band + 1], 0.0, 0.95)
                x_new[j] = (1.0 - local_w) * x_euler[j] + local_w * x_mom[j]
            end
        else
            for j = 0, n - 1 do
                x_new[j] = (1.0 - eff_weight) * x_euler[j] + eff_weight * x_mom[j]
            end
        end

        if C.has_nan_inf(x_new, n) then
            for j = 0, n - 1 do x_new[j] = x_euler[j] end
        end

        -- 6. Hamiltonian energy tracking
        local T = kinetic_energy(momentum, mass, n)
        local V = potential_energy(x_new, v_curr, sigma_ratio, n)
        local H = T + V
        local corrected = false

        if H_ref == nil then
            H_ref = H
        else
            local rel_drift = math.abs(H - H_ref) / (math.abs(H_ref) + C.EPSILON)
            if energy_mode ~= "monitor" and rel_drift > energy_tol and corr_str > 0 then
                local T_target = H_ref - V
                if T_target < 0.01 then T_target = 0.01 end
                local scale = math.sqrt(T_target / (T + C.EPSILON))
                scale = 1.0 + corr_str * (scale - 1.0)
                scale = C.clamp(scale, 0.5, 2.0)
                for j = 0, n - 1 do momentum[j] = momentum[j] * scale end
                corrected = true
            end
            if energy_mode == "adaptive" then H_ref = 0.95 * H_ref + 0.05 * H end
        end

        -- 7-8. Identity + tonal anchor (via commons)
        if opts.f_id_anchor then
            C.apply_identity_anchor(x_new, n, sigma_ratio, opts.anchor_sigma, opts.anchor_blend, state)
        end
        if opts.f_tonal then
            C.apply_tonal_anchor(x_new, n, B, NPB, sigma_ratio, opts.anchor_sigma, opts.tonal_str, state)
        end

        -- 9. Primary look-back (via commons)
        if opts.f_lookback then
            C.apply_look_back(x_new, n, sigma_ratio, opts.lb_lambda, opts.lb_snr_power, state, "lb_prev")
        end

        -- 10. RMS servo (via commons)
        if opts.f_rms then
            C.apply_rms_servo(x_new, n, B, NPB, sigma_ratio, opts.rms_tgt_min, opts.rms_tgt_max, opts.rms_gain)
        end

        -- 11. Post-step look-back (secondary, via commons)
        if post_lb_enabled then
            C.apply_look_back(x_new, n, sigma_ratio, post_lb_lam, post_lb_snr, state, "lb2_prev")
        end

        -- 12. SDE noise + safety clamp (via commons)
        C.apply_sde_noise(x_new, n, sigma_next, opts.eta, opts.seed, step_idx)
        C.apply_safety_clamp(x_new, n, opts.sclamp)

        if opts.verbose then
            print(string.format(
                "[HAMILTONIAN V2] step %02d | H=%.2f %s | align=%.3f conf=%.2f ew=%.3f | rms=%.3f",
                step_idx, H, corrected and "CORR" or "ok",
                alignment, confidence, eff_weight, C.rms(x_new, n)))
        end

        x = x_new
        C.tbl_to_fa(x, xt, n)
        C.tbl_to_fa(v_curr, vt_buf, n)
        if on_step(step_idx, sigma_curr, sigma_next) then return end
        x = C.fa_to_tbl(xt, n)
    end

    C.tbl_to_fa(x, xt, n)
end
