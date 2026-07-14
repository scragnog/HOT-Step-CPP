-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
-- ============================================================================

local C = require("md_solver_commons")

-- MD Confluence V4 -- STORM / Trajectory Anchor hybrid solver
-- MDMAchine | A&E Concepts (c) 2026
--
-- V4: Commons integration + relational velocity decomposition.
-- Tonal ramp, look-back floor, RMS default on, anchor_blend 0.12.
--
-- owns_loop = true. Forks STORM's stiffness-gated multi-order dispatch AND
-- Trajectory Anchor's full 13-stage stateful correction stack into one loop,
-- blending their two x_next candidates per step via disagreement- and
-- inertia-modulated mixing.
--
-- CANDIDATE MODEL:
--   Both candidates are x_next (post-advance latents), NOT vt.
--   v_curr is computed ONCE per step and shared by both candidates.
--
-- STATE-FEEDBACK RULE:
--   STORM's v_cache stores velocity (v_curr, shared) -- no desync possible.
--   Anchor's latent state (_anc_prev_out, _anc_history) is OVERWRITTEN with
--   x_final (the blended result) so its memory/inertia/concept-lock math
--   believes the blended trajectory is what happened. One-shot references
--   (identity anchor snapshot, tonal anchor capture) fire against x_final too
--   since they read whatever the actual trajectory is at anchor_sigma.
-- ============================================================================

solver = {
    name        = "md_confluence_v4",
    display     = "MD Confluence V4",
    description = "STORM / Trajectory Anchor hybrid with batch-aware routing. Disagreement- and inertia-modulated latent blend. Per-batch tonal anchor, spectral guard, and RMS servo.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = true,
    owns_loop   = true,
    params      = {
        -- ── Mix ──────────────────────────────────────────────────────────────
        { key = "mix_amount", type = "slider", label = "Mix Amount",
          default = 50, min = 0, max = 100, step = 1,
          hint = "Base blend: 0 = pure STORM, 100 = pure Trajectory Anchor. Modulated at runtime by disagreement mode and inertia gating -- effective mix moves around this value, not on it." },
        { key = "disagreement_mode", type = "select", label = "Disagreement Mode",
          default = "adaptive",
          options = {
            { value = "damp",     label = "Damp (consensus)" },
            { value = "amplify",  label = "Amplify (instability)" },
            { value = "adaptive", label = "Adaptive (damp early, amplify late)" },
          },
          hint = "How blend reacts when STORM and Anchor candidates disagree. Damp = pull toward consensus. Amplify = disagreement becomes controlled texture. Adaptive = damp during structure, amplify during detail." },
        { key = "damp_strength", type = "slider", label = "Damp Strength",
          default = 0.4, min = 0, max = 1, step = 0.05,
          hint = "How hard disagreement pulls mix toward consensus (damp/adaptive mode)." },
        { key = "chaos_strength", type = "slider", label = "Chaos Strength",
          default = 0.3, min = 0, max = 1, step = 0.05,
          hint = "How hard disagreement pushes mix further from center (amplify/adaptive mode)." },
        { key = "inertia_influence", type = "slider", label = "Inertia Influence",
          default = 0.7, min = 0, max = 1, step = 0.05,
          hint = "How much Anchor's inertia state gates the mix. 0 = pure user mix. 1 = full auto-gating (low inertia collapses toward STORM)." },
        { key = "inertia_gate_low", type = "slider", label = "Inertia Gate Low",
          default = 0.15, min = 0, max = 1, step = 0.01,
          hint = "Smoothstep floor: inertia magnitude below this = mix fully gated toward STORM." },
        { key = "inertia_gate_high", type = "slider", label = "Inertia Gate High",
          default = 0.6, min = 0, max = 1, step = 0.01,
          hint = "Smoothstep ceiling: inertia magnitude above this = user's stated mix takes over fully." },

        -- ── STORM params ──────────────────────────────────────────────────
        { key = "stiffness_threshold", type = "slider", label = "STORM: Detail Sensitivity",
          default = 0.15, min = 0.05, max = 0.50, step = 0.01,
          hint = "Stiffness threshold. Lower = more careful on transients." },
        { key = "rk_order", type = "select", label = "STORM: Precision Level",
          default = "auto",
          options = {
            { value = "auto", label = "Auto" },
            { value = "2", label = "RK2" }, { value = "3", label = "RK3" },
            { value = "4", label = "RK4" }, { value = "5", label = "RK5" },
          },
          hint = "STORK solver order when stiff." },
        { key = "cache_depth", type = "slider", label = "STORM: History Memory",
          default = 5, min = 2, max = 10, step = 1,
          hint = "Velocity cache depth for STORM's multi-order dispatch." },
        { key = "look_back_lambda_storm", type = "slider", label = "STORM: Look-Back Lambda",
          default = 0.15, min = 0, max = 1, step = 0.01,
          hint = "STORM's own look-back smoother weight. 0 = off." },
        { key = "look_back_snr_power_storm", type = "slider", label = "STORM: Look-Back SNR Power",
          default = 1.5, min = 0.5, max = 3, step = 0.1,
          hint = "STORM look-back falloff exponent." },

        -- ── Anchor params ──────────────────────────────────────────────────
        { key = "warmup_steps", type = "slider", label = "Anchor: Warmup Steps",
          default = 2, min = 0, max = 6, step = 1,
          hint = "Skip Anchor stateful features for first N steps. Also gates inertia toward 0 during warmup." },
        { key = "inertia_alpha", type = "slider", label = "Anchor: Inertia Alpha",
          default = 0.15, min = 0.0, max = 0.5, step = 0.01,
          hint = "Anchor velocity carry-over coefficient. Entropy-modulated at runtime." },
        { key = "memory_blend", type = "slider", label = "Anchor: Memory Blend",
          default = 0.12, min = 0.0, max = 0.5, step = 0.01,
          hint = "3-step ring buffer blend fraction." },
        { key = "concept_lock", type = "toggle", label = "Anchor: Concept Lock",
          default = true, hint = "Stability mask on settled regions." },
        { key = "concept_sigma_power", type = "slider", label = "Anchor: Concept Sigma Power",
          default = 1.0, min = 0.25, max = 3.0, step = 0.25,
          hint = "Concept lock fade curve across sigma." },
        { key = "identity_anchor", type = "toggle", label = "Anchor: Identity Anchor",
          default = true, hint = "Snapshot pull-back at anchor_sigma." },
        { key = "anchor_sigma", type = "slider", label = "Anchor: Anchor Sigma",
          default = 0.5, min = 0.1, max = 0.9, step = 0.05,
          hint = "Sigma fraction for identity/tonal anchor capture." },
        { key = "anchor_blend", type = "slider", label = "Anchor: Anchor Blend",
          default = 0.08, min = 0.01, max = 0.30, step = 0.01,
          hint = "Pull strength toward identity anchor." },
        { key = "tonal_anchor", type = "toggle", label = "Anchor: Tonal Anchor",
          default = true, hint = "Spectral centroid drift correction." },
        { key = "tonal_strength", type = "slider", label = "Anchor: Tonal Strength",
          default = 0.15, min = 0.0, max = 1.0, step = 0.05,
          hint = "Tonal correction scale (hard-capped 0.1%/step regardless)." },
        { key = "look_back_enabled_anchor", type = "toggle", label = "Anchor: Look-Back Smoother",
          default = true, hint = "SNR-adaptive latent EMA." },
        { key = "look_back_lambda_anchor", type = "slider", label = "Anchor: Look-Back Lambda",
          default = 0.55, min = 0.05, max = 1.0, step = 0.05,
          hint = "Max look-back weight at high sigma." },
        { key = "look_back_snr_power_anchor", type = "slider", label = "Anchor: Look-Back SNR Power",
          default = 1.3, min = 0.5, max = 3.0, step = 0.1,
          hint = "Look-back falloff exponent." },
        { key = "rms_servo", type = "toggle", label = "Anchor: RMS Servo",
          default = false, hint = "Downward-only RMS ceiling." },
        { key = "rms_target_min", type = "slider", label = "Anchor: RMS Target Min",
          default = 1.2, min = 0.1, max = 3.0, step = 0.05, hint = "RMS ceiling at low sigma." },
        { key = "rms_target_max", type = "slider", label = "Anchor: RMS Target Max",
          default = 2.5, min = 0.5, max = 5.0, step = 0.05, hint = "RMS ceiling at high sigma." },
        { key = "rms_servo_gain", type = "slider", label = "Anchor: RMS Servo Gain",
          default = 0.6, min = 0.1, max = 1.0, step = 0.05, hint = "Servo correction aggressiveness." },
        { key = "latent_pressure", type = "toggle", label = "Anchor: Latent Pressure",
          default = false, hint = "Entropy x RMS target correction (off by default)." },
        { key = "pressure_target_rms", type = "slider", label = "Anchor: Pressure Target RMS",
          default = 2.0, min = 0.5, max = 4.0, step = 0.1, hint = "RMS component of pressure target." },
        { key = "pressure_target_entropy", type = "slider", label = "Anchor: Pressure Target Entropy",
          default = 7.5, min = 1.0, max = 15.0, step = 0.5, hint = "Shannon entropy target." },

        -- ── Post-Blend Shearing Control ──────────────────────────────────
        { key = "post_blend_lookback", type = "slider", label = "Post-Blend Look-Back",
          default = 0.25, min = 0.0, max = 0.7, step = 0.05,
          hint = "SNR-adaptive EMA on x_final AFTER the blend. Neither sub-solver's look-back covers the blend seam -- this does. 0 = off. 0.25 = subtle anti-shear. Fades with sigma like anchor's look-back." },
        { key = "post_blend_snr_power", type = "slider", label = "Post-Blend SNR Power",
          default = 1.0, min = 0.5, max = 3.0, step = 0.1,
          hint = "Falloff exponent for post-blend look-back. 1.0 = linear fade (more late-step smoothing than anchor's 1.3 default). Lower = more smoothing persists into detail steps." },
        { key = "spectral_guard", type = "slider", label = "Spectral Blend Guard",
          default = 0.4, min = 0.0, max = 1.0, step = 0.05,
          hint = "Frequency-aware blend correction. When STORM and anchor disagree, their delta concentrates in high-freq (metallic) components. This attenuates the blend delta in the upper latent bands proportional to disagreement. 0 = off (flat blend). 0.4 = moderate HF damping. 1.0 = aggressive." },
        { key = "late_damp_override", type = "slider", label = "Late Damp Override",
          default = 0.7, min = 0.0, max = 1.0, step = 0.05,
          hint = "In adaptive mode, overrides amplify with damp for the final portion of the run. 0.7 = last 30% of steps forced to damp. 0 = no override (pure adaptive all the way). Prevents late-step disagreement amplification causing metallic ringing." },

        -- ── SDE / Safety ──────────────────────────────────────────────────
        { key = "eta", type = "slider", label = "Noise Injection (0 = ODE)",
          default = 0.0, min = 0.0, max = 1.0, step = 0.05,
          hint = "Post-blend SDE noise: scale = sigma_next * eta." },
        { key = "seed", type = "slider", label = "Seed",
          default = 42, min = 0, max = 999999, step = 1,
          hint = "RNG seed for SDE noise." },
        { key = "safety_clamp", type = "slider", label = "Safety Clamp",
          default = 2.5, min = 1.0, max = 5.0, step = 0.1,
          hint = "Max abs latent value post-blend." },
        { key = "verbose", type = "toggle", label = "Verbose Logging",
          default = false,
          hint = "Per-step blend diagnostics: agreement, inertia, effective_mix, STORM mode." },
        { key = "relational_weight", type = "slider", label = "Relational Weight",
          default = 0.0, min = 0.0, max = 1.0, step = 0.05,
          hint = "Barbour Best Matching velocity decomposition. 0 = off." },
        { key = "relational_sigma_power", type = "slider", label = "Relational Sigma Decay",
          default = 1.0, min = 0.25, max = 4.0, step = 0.25,
          hint = "How fast relational weight fades." },
    },
}

-- ─────────────────────────────────────────────────────────────────────────────
-- HELPERS (aliased from md_solver_commons)
-- ─────────────────────────────────────────────────────────────────────────────

local EPSILON      = C.EPSILON
local PRESSURE_CAP = 5e-4

local clamp            = C.clamp
local smoothstep       = C.smoothstep
local fa_to_tbl        = C.fa_to_tbl
local tbl_to_fa        = C.tbl_to_fa
local vec_norm         = C.vec_norm
local vec_sub_norm     = C.vec_sub_norm
local vec_dot          = C.vec_dot
local vec_clone        = C.vec_clone
local cosine_sim       = C.cosine_sim
local has_nan_inf      = C.has_nan_inf
local rms_range        = C.rms_range
local rms              = C.rms

local function shannon_entropy(a, n)
    local sum = 0.0
    for i = 0, n - 1 do sum = sum + math.abs(a[i]) + 1e-7 end
    local inv_sum = 1.0 / (sum + 1e-8)
    local H = 0.0
    for i = 0, n - 1 do
        local p = (math.abs(a[i]) + 1e-7) * inv_sum
        H = H - p * math.log(p + EPSILON) / math.log(2.0)
    end
    H = math.max(0.05, H)
    if H ~= H or H == math.huge or H == -math.huge then H = 5.0 end
    return H
end

local spectral_centroid = C.spectral_centroid
local band_energy      = C.band_energy
local make_rng         = C.make_rng
local normal           = C.normal
local bool_param       = C.bool_param
local num_param        = C.num_param

-- ─────────────────────────────────────────────────────────────────────────────
-- STORM INTERNALS (ported verbatim from storm_sampler_core.lua)
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
    local cos_sim_val = dot / (nc * np_ + 1e-8)

    if step_idx < n_calib then
        baseline.sum        = (baseline.sum   or 0.0) + smoothed
        baseline.count      = (baseline.count or 0)   + 1
        baseline.last_ratio = smoothed
        return true, baseline, cos_sim_val
    end

    local bmean    = baseline.sum / math.max(baseline.count, 1)
    local adap_thr = threshold * (bmean / 0.15)
    adap_thr       = clamp(adap_thr, 0.05, 0.50)

    local stiff = smoothed > adap_thr
    baseline.last_ratio     = smoothed
    baseline.last_threshold = adap_thr
    return stiff, baseline, cos_sim_val
end

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

    local dot     = vec_dot(v_curr, v_prev_0, n)
    local nc      = vec_norm(v_curr, n)
    local np_     = vec_norm(v_prev_0, n)
    local cos_sim_val = dot / (nc * np_ + 1e-8)
    local damping = clamp(cos_sim_val, 0.0, 1.0)

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

-- STORM's own look-back (operates on its x_next candidate independently)
local function storm_look_back(x_curr, x_prev, sigma_curr, sigma_max, lambda_base, snr_power, n)
    if x_prev == nil then return x_curr, 0.0 end
    local ratio = math.min(sigma_curr / math.max(sigma_max, 1e-8), 1.0)
    local lam   = lambda_base * (ratio ^ snr_power)
    local out   = {}
    for i = 0, n - 1 do out[i] = (1.0 - lam) * x_curr[i] + lam * x_prev[i] end
    return out, lam
end

-- ─────────────────────────────────────────────────────────────────────────────
-- ANCHOR STATE (module-level, reset per generation)
-- ─────────────────────────────────────────────────────────────────────────────

local _anc_sigma_max       = nil
local _anc_has_prev        = false
local _anc_has_velocity    = false
local _anc_has_anchor      = false
local _anc_tonal_ref_cent  = {}    -- per-batch
local _anc_tonal_ref_bands = {}    -- per-batch
local _anc_tonal_captured  = false
local _anc_last_entropy    = 7.5
local _anc_hist_head       = 1
local _anc_hist_count      = 0

-- Hoisted buffers (resized on n change)
local _anc_out        = {}
local _anc_fallback   = {}
local _anc_vel_old    = {}
local _anc_vel_raw    = {}
local _anc_id_buf     = {}
local _anc_prev_out   = {}
local _anc_hist_mean  = {}
local _anc_history    = { {}, {}, {} }

local function reset_anchor_state(n)
    _anc_sigma_max       = nil
    _anc_has_prev        = false
    _anc_has_velocity    = false
    _anc_has_anchor      = false
    _anc_tonal_ref_cent  = {}   -- per-batch: [b] = centroid
    _anc_tonal_ref_bands = {}   -- per-batch: [b] = {band1..4}
    _anc_tonal_captured  = false
    _anc_last_entropy    = 7.5
    _anc_hist_head       = 1
    _anc_hist_count      = 0
    for i = 0, n - 1 do
        _anc_out[i]       = 0.0
        _anc_fallback[i]  = 0.0
        _anc_vel_old[i]   = 0.0
        _anc_vel_raw[i]   = 0.0
        _anc_id_buf[i]    = 0.0
        _anc_prev_out[i]  = 0.0
        _anc_hist_mean[i] = 0.0
        _anc_history[1][i] = 0.0
        _anc_history[2][i] = 0.0
        _anc_history[3][i] = 0.0
    end
end

-- ─────────────────────────────────────────────────────────────────────────────
-- ANCHOR CANDIDATE (full 13-stage pipeline from md_trajectory_anchor.lua)
-- Input: x (Lua table, current latent), v_curr (velocity), sigma_curr, sigma_next, n
-- Reads/writes _anc_* state. Returns x_next_anchor as Lua table.
-- ─────────────────────────────────────────────────────────────────────────────

local function anchor_candidate(x, v_curr, sigma_curr, sigma_next, step_idx, n, p, B, NPB)
    local warmup        = math.floor(num_param(p, "warmup_steps", 2))
    local f_inertia     = true  -- always on in confluence (inertia_alpha=0 to disable)
    local inertia_a     = num_param(p, "inertia_alpha", 0.15)
    local f_memory      = true  -- always on (memory_blend=0 to disable)
    local mem_blend     = num_param(p, "memory_blend", 0.12)
    local f_concept     = bool_param(p, "concept_lock", true)
    local concept_power = num_param(p, "concept_sigma_power", 1.0)
    local f_anchor      = bool_param(p, "identity_anchor", true)
    local anchor_sigma  = num_param(p, "anchor_sigma", 0.5)
    local anchor_blend  = num_param(p, "anchor_blend", 0.08)
    local f_tonal       = bool_param(p, "tonal_anchor", true)
    local tonal_str     = num_param(p, "tonal_strength", 0.15)
    local f_lookback    = bool_param(p, "look_back_enabled_anchor", true)
    local lb_lambda     = num_param(p, "look_back_lambda_anchor", 0.55)
    local lb_snr_power  = num_param(p, "look_back_snr_power_anchor", 1.3)
    local f_rms         = bool_param(p, "rms_servo", false)
    local rms_tgt_min   = num_param(p, "rms_target_min", 1.2)
    local rms_tgt_max   = num_param(p, "rms_target_max", 2.5)
    local rms_gain      = num_param(p, "rms_servo_gain", 0.6)
    local f_pressure    = bool_param(p, "latent_pressure", false)
    local p_tgt_rms     = num_param(p, "pressure_target_rms", 2.0)
    local p_tgt_entropy = num_param(p, "pressure_target_entropy", 7.5)
    local sclamp        = num_param(p, "safety_clamp", 2.5)

    if _anc_sigma_max == nil then _anc_sigma_max = sigma_curr end
    local sigma_ratio = clamp(sigma_curr / math.max(_anc_sigma_max, EPSILON), 0.0, 1.0)
    local past_warmup = (step_idx >= warmup)

    -- 2. Entropy (from input x)
    _anc_last_entropy = shannon_entropy(x, n)

    -- 3. Euler advance: dt = sigma_next - sigma_curr (negative in flow-matching)
    local dt = sigma_next - sigma_curr
    for i = 0, n - 1 do
        local v = x[i] + dt * v_curr[i]
        _anc_out[i]      = v
        _anc_fallback[i] = v
    end

    -- 4. Latent Pressure
    if f_pressure then
        local cur_rms        = rms(_anc_out, n)
        local target_product = p_tgt_entropy * p_tgt_rms
        local cur_product    = _anc_last_entropy * cur_rms
        local correction     = clamp(
            (target_product - cur_product) / (target_product + EPSILON),
            -PRESSURE_CAP, PRESSURE_CAP)
        if math.abs(correction) > 1e-6 then
            for i = 0, n - 1 do _anc_out[i] = _anc_out[i] * (1.0 + correction) end
        end
    end

    -- 5. Memory Buffer
    if mem_blend > 0 and past_warmup and _anc_hist_count > 0 then
        for i = 0, n - 1 do _anc_hist_mean[i] = 0.0 end
        local hw = 1.0 / _anc_hist_count
        for h = 1, _anc_hist_count do
            for i = 0, n - 1 do _anc_hist_mean[i] = _anc_hist_mean[i] + _anc_history[h][i] end
        end
        for i = 0, n - 1 do
            _anc_out[i] = (1.0 - mem_blend) * _anc_out[i] + mem_blend * (_anc_hist_mean[i] * hw)
        end
    end

    -- 6. Inertia Engine
    if inertia_a > 0 and past_warmup and _anc_has_prev then
        for i = 0, n - 1 do _anc_vel_raw[i] = _anc_out[i] - _anc_prev_out[i] end
        if _anc_has_velocity then
            for i = 0, n - 1 do
                _anc_vel_old[i] = 0.8 * _anc_vel_old[i] + 0.2 * _anc_vel_raw[i]
            end
        else
            for i = 0, n - 1 do _anc_vel_old[i] = _anc_vel_raw[i] end
            _anc_has_velocity = true
        end
        local alpha = inertia_a * clamp(_anc_last_entropy / 7.5, 0.0, 1.5)
        for i = 0, n - 1 do _anc_out[i] = _anc_out[i] + alpha * _anc_vel_old[i] end
    end

    -- 7. Concept Lock
    if f_concept and past_warmup and _anc_has_prev then
        local sigma_mod = sigma_ratio ^ concept_power
        if sigma_mod > 1e-4 then
            for i = 0, n - 1 do
                local delta  = math.abs(_anc_out[i] - _anc_prev_out[i])
                local lock_w = (1.0 / (1.0 + math.exp(delta * 40.0 - 2.0))) * sigma_mod
                _anc_out[i]  = (1.0 - lock_w) * _anc_out[i] + lock_w * _anc_prev_out[i]
            end
        end
    end

    -- 8. Identity Anchor
    if f_anchor and past_warmup then
        if not _anc_has_anchor and sigma_ratio <= anchor_sigma then
            for i = 0, n - 1 do _anc_id_buf[i] = _anc_out[i] end
            _anc_has_anchor = true
        elseif _anc_has_anchor then
            for i = 0, n - 1 do
                _anc_out[i] = (1.0 - anchor_blend) * _anc_out[i] + anchor_blend * _anc_id_buf[i]
            end
        end
    end

    -- 9. Tonal Anchor (per-batch centroid + band correction)
    if f_tonal and past_warmup then
        if not _anc_tonal_captured and sigma_ratio <= anchor_sigma then
            for b = 0, B - 1 do
                local off = b * NPB
                _anc_tonal_ref_cent[b]  = spectral_centroid(_anc_out, off, NPB)
                _anc_tonal_ref_bands[b] = band_energy(_anc_out, off, NPB)
            end
            _anc_tonal_captured = true
        elseif _anc_tonal_captured then
            local eff_str = tonal_str * sigma_ratio
            if eff_str > 1e-6 then
                for b = 0, B - 1 do
                    local off = b * NPB
                    local curr_centroid = spectral_centroid(_anc_out, off, NPB)
                    local curr_bands    = band_energy(_anc_out, off, NPB)
                    local drift_norm_val = (curr_centroid - _anc_tonal_ref_cent[b]) /
                                       (math.abs(_anc_tonal_ref_cent[b]) + EPSILON)
                    local tilt = clamp(-drift_norm_val * eff_str, -1e-3, 1e-3)
                    local center = (NPB - 1) / 2.0
                    for i = off, off + NPB - 1 do
                        local dist_w = ((i - off) - center) / (center + EPSILON)
                        _anc_out[i] = _anc_out[i] + tilt * dist_w * math.abs(_anc_out[i])
                    end
                    local ref_total, curr_total = 0.0, 0.0
                    for bb = 1, 4 do
                        ref_total  = ref_total  + _anc_tonal_ref_bands[b][bb]
                        curr_total = curr_total + curr_bands[bb]
                    end
                    if ref_total > EPSILON and curr_total > EPSILON then
                        local bsize = math.floor(NPB / 4)
                        for bb = 0, 3 do
                            local ref_ratio  = _anc_tonal_ref_bands[b][bb + 1] / ref_total
                            local curr_ratio = curr_bands[bb + 1]               / curr_total
                            local band_corr  = clamp((ref_ratio - curr_ratio) * eff_str, -1e-3, 1e-3)
                            local blo = off + bb * bsize
                            local bhi = (bb == 3) and (off + NPB - 1) or (blo + bsize - 1)
                            for i = blo, bhi do
                                _anc_out[i] = _anc_out[i] + band_corr * math.abs(_anc_out[i])
                            end
                        end
                    end
                end
            end
        end
    end

    -- 10. Look-Back Smoother
    if f_lookback and past_warmup and _anc_has_prev then
        local lb_w = lb_lambda * (sigma_ratio ^ lb_snr_power)
        if lb_w > 1e-6 then
            for i = 0, n - 1 do
                _anc_out[i] = (1.0 - lb_w) * _anc_out[i] + lb_w * _anc_prev_out[i]
            end
        end
    end

    -- 11. RMS Servo (per-batch)
    if f_rms then
        local rms_target = rms_tgt_min + (sigma_ratio ^ 0.6) * (rms_tgt_max - rms_tgt_min)
        for b = 0, B - 1 do
            local off = b * NPB
            local cur_rms = rms_range(_anc_out, off, NPB)
            if cur_rms > rms_target then
                local servo_rms = cur_rms + rms_gain * (rms_target - cur_rms)
                local scale     = servo_rms / cur_rms
                for i = off, off + NPB - 1 do _anc_out[i] = _anc_out[i] * scale end
            end
        end
    end

    -- 12. Safety Clamp + NaN Guard
    if has_nan_inf(_anc_out, n) then
        for i = 0, n - 1 do _anc_out[i] = _anc_fallback[i] end
    end
    for i = 0, n - 1 do _anc_out[i] = clamp(_anc_out[i], -sclamp, sclamp) end

    -- Return candidate (state feedback happens in main loop AFTER blend)
    local result = {}
    for i = 0, n - 1 do result[i] = _anc_out[i] end
    return result
end

-- Feed blended x_final back into anchor's state so its memory believes
-- the blended trajectory is what happened
local function anchor_state_feedback(x_final, step_idx, past_warmup, n)
    if past_warmup then
        for i = 0, n - 1 do _anc_prev_out[i] = x_final[i] end
        _anc_has_prev = true
        -- Ring buffer push
        for i = 0, n - 1 do _anc_history[_anc_hist_head][i] = x_final[i] end
        _anc_hist_head = _anc_hist_head + 1
        if _anc_hist_head > 3 then _anc_hist_head = 1 end
        if _anc_hist_count < 3 then _anc_hist_count = _anc_hist_count + 1 end
    end
end

-- ─────────────────────────────────────────────────────────────────────────────
-- CONFLUENCE BLEND
-- ─────────────────────────────────────────────────────────────────────────────

local function compute_effective_mix(user_mix, disagreement, inertia_mag, inertia_influence,
                                      gate_low, gate_high, mode, damp_str, chaos_str, t_frac)
    local base = user_mix / 100.0
    local gate = smoothstep(inertia_mag, gate_low, gate_high)
    local gated_low  = base * 0.3
    local gated_full = base
    local gate_mixed = gated_low * (1 - gate) + gated_full * gate
    local gated = base * (1 - inertia_influence) + gate_mixed * inertia_influence

    local function damp_term()
        return gated * (1 - disagreement * damp_str)
    end
    local function amplify_term()
        local push = disagreement * chaos_str
        local sign = (gated >= 0.5) and 1.0 or -1.0
        return clamp(gated + push * sign, 0, 1)
    end

    local effective
    if mode == "damp" then
        effective = damp_term()
    elseif mode == "amplify" then
        effective = amplify_term()
    else -- adaptive
        effective = damp_term() * (1 - t_frac) + amplify_term() * t_frac
    end
    return clamp(effective, 0, 1)
end

-- ─────────────────────────────────────────────────────────────────────────────
-- SAMPLE -- full loop
-- ─────────────────────────────────────────────────────────────────────────────

function sample(xt, vt_buf, schedule, n, model_fn)
    local p = params or {}

    local mix_amount        = num_param(p, "mix_amount", 50)
    local disagreement_mode = p.disagreement_mode or "adaptive"
    local damp_str          = num_param(p, "damp_strength", 0.4)
    local chaos_str         = num_param(p, "chaos_strength", 0.3)
    local inertia_influence = num_param(p, "inertia_influence", 0.7)
    local gate_low          = num_param(p, "inertia_gate_low", 0.15)
    local gate_high         = num_param(p, "inertia_gate_high", 0.6)
    local stiffness_thr     = num_param(p, "stiffness_threshold", 0.15)
    local rk_order          = p.rk_order or "auto"
    local depth_max         = math.floor(num_param(p, "cache_depth", 5))
    local lb_lambda_storm   = num_param(p, "look_back_lambda_storm", 0.15)
    local lb_snr_storm      = num_param(p, "look_back_snr_power_storm", 1.5)
    local warmup            = math.floor(num_param(p, "warmup_steps", 2))
    local pb_lb_lambda      = num_param(p, "post_blend_lookback", 0.25)
    local pb_lb_snr         = num_param(p, "post_blend_snr_power", 1.0)
    local spec_guard        = num_param(p, "spectral_guard", 0.4)
    local late_damp_at      = num_param(p, "late_damp_override", 0.7)
    local eta               = num_param(p, "eta", 0.0)
    local seed              = math.floor(num_param(p, "seed", 42))
    local sclamp            = num_param(p, "safety_clamp", 2.5)
    local verbose           = bool_param(p, "verbose", false)
    local rw                = num_param(p, "relational_weight", 0.0)
    local rw_sig_pow        = num_param(p, "relational_sigma_power", 1.0)

    local ns      = #schedule
    -- Engine schedule has NO trailing 0 (fix ported from 46c081e): iterate all ns
    -- entries so the last iteration gets sigma_next = 0.0 and the terminal branch
    -- performs the final x0 projection. With ns - 1 that branch is dead code and
    -- the output keeps ~final-sigma noise.
    local n_steps = ns
    if n_steps < 1 then return end

    -- Batch routing: engine exposes batch_n and n_per as globals
    local B   = (batch_n and batch_n > 0) and batch_n or 1
    local NPB = (n_per and n_per > 0) and n_per or n
    if B * NPB ~= n then B = 1; NPB = n end

    -- Reset both sub-solver states
    local v_cache  = {}
    local baseline = { sum = 0.0, count = 0 }
    local hyst     = 0.05
    local ema_a    = 0.3
    local n_calib  = math.max(2, math.min(5, math.floor(n_steps * 0.12)))

    reset_anchor_state(n)

    local sigma_max = schedule[1]
    local x = fa_to_tbl(xt, n)

    -- STORM look-back state
    local storm_lb_prev = nil
    local lb_storm_enabled = (lb_lambda_storm > 0)

    -- Post-blend look-back state
    local pb_prev = nil
    local pb_enabled = (pb_lb_lambda > 0)

    if verbose then
        print(string.format("[CONFLUENCE V4] Schedule: %d steps | B=%d NPB=%d | Mix: %d | Mode: %s | RK: %s",
            n_steps, B, NPB, mix_amount, disagreement_mode, tostring(rk_order)))
    end

    for i = 1, n_steps do
        local sigma_curr = schedule[i]
        local sigma_next = (i < ns) and schedule[i + 1] or 0.0
        local step_idx   = i - 1

        -- Terminal step: plain Euler, no blend
        if sigma_next == 0.0 then
            tbl_to_fa(x, xt, n)
            model_fn(xt, sigma_curr)
            local v_final = fa_to_tbl(vt_buf, n)
            for j = 0, n - 1 do x[j] = x[j] - v_final[j] * sigma_curr end
            if verbose then print(string.format("[CONFLUENCE] Step %02d: TERMINAL (Euler)", step_idx)) end
            break
        end

        -- Single model call, shared by both candidates
        tbl_to_fa(x, xt, n)
        model_fn(xt, sigma_curr)
        local v_curr = fa_to_tbl(vt_buf, n)

        -- Relational decomposition
        if rw > 0 then
            local sr = clamp(sigma_curr / math.max(sigma_max, EPSILON), 0.0, 1.0)
            C.apply_relational(v_curr, n, B, NPB, sr, sigma_max,
                rw, rw_sig_pow, false, 0.85, x)
        end

        -- Save pre-step x for STORM look-back
        local x_before_storm = nil
        if lb_storm_enabled then x_before_storm = vec_clone(x, n) end

        -- ── CANDIDATE A: STORM ──────────────────────────────────────────
        local stiff, cos_sim_out
        if #v_cache >= 1 then
            stiff, baseline, cos_sim_out = compute_stiffness(
                v_curr, v_cache, step_idx, baseline, stiffness_thr, ema_a, n_calib, n)
        else
            stiff, cos_sim_out = true, nil
        end

        -- Hysteresis
        local prev_mode = baseline.prev_mode or "STORK"
        if prev_mode == "DPM++" and not stiff then
            if (baseline.last_ratio or 0) > (baseline.last_threshold or stiffness_thr) + hyst then
                stiff = true
            end
        end

        local x_next_storm, actual_order, storm_mode
        if stiff then
            x_next_storm, actual_order = stork_step(v_cache, x, sigma_curr, sigma_next, v_curr, rk_order, n)
            storm_mode = "STORK"
        else
            x_next_storm = dpmpp3m_step(v_cache, x, sigma_curr, sigma_next, v_curr, n)
            storm_mode   = "DPM++"
            actual_order = 3
        end

        -- STORM NaN guard
        if has_nan_inf(x_next_storm, n) then
            local dt = sigma_next - sigma_curr
            x_next_storm = {}
            for j = 0, n - 1 do x_next_storm[j] = x[j] + dt * v_curr[j] end
            v_cache = {}
            actual_order = 1
        end

        -- STORM look-back (its own, independent of anchor's)
        if lb_storm_enabled then
            x_next_storm = storm_look_back(x_next_storm, storm_lb_prev, sigma_curr, sigma_max, lb_lambda_storm, lb_snr_storm, n)
            storm_lb_prev = x_before_storm
        end

        baseline.prev_mode = storm_mode

        -- Update STORM v_cache (stores velocity, not latent -- no desync)
        table.insert(v_cache, { v = v_curr, sigma = sigma_curr })
        while #v_cache > depth_max do table.remove(v_cache, 1) end

        -- ── CANDIDATE B: ANCHOR ─────────────────────────────────────────
        local x_next_anchor = anchor_candidate(x, v_curr, sigma_curr, sigma_next, step_idx, n, p, B, NPB)

        -- ── DISAGREEMENT + INERTIA ──────────────────────────────────────
        local agreement    = cosine_sim(x_next_storm, x_next_anchor, n)
        local disagreement = 1.0 - agreement
        local mag_ratio    = vec_norm(x_next_anchor, n) / (vec_norm(x_next_storm, n) + EPSILON)

        local inertia_mag = vec_norm(_anc_vel_old, n) / (vec_norm(v_curr, n) + EPSILON)
        inertia_mag = clamp(inertia_mag, 0, 1.5)

        local t_frac = step_idx / math.max(n_steps - 1, 1)
        local past_warmup = (step_idx >= warmup)

        -- Late damp override: force damp mode past late_damp_at fraction
        local active_mode = disagreement_mode
        if active_mode == "adaptive" and late_damp_at > 0 and t_frac >= late_damp_at then
            active_mode = "damp"
        end

        local effective_mix = compute_effective_mix(
            mix_amount, disagreement, inertia_mag, inertia_influence,
            gate_low, gate_high, active_mode, damp_str, chaos_str, t_frac)

        -- ── BLEND (with spectral guard) ──────────────────────────────────
        local x_final = {}

        if spec_guard > 0 and disagreement > 0.01 then
            -- Frequency-aware blend: attenuate the blend delta in upper bands
            -- proportional to disagreement. Per-batch band assignment.
            local bsize  = math.floor(NPB / 4)
            local atten  = disagreement * spec_guard
            for j = 0, n - 1 do
                local local_idx = j % NPB
                local band = math.floor(local_idx / bsize)
                if band > 3 then band = 3 end
                -- band 0 (low) = no attenuation, band 3 (high) = full attenuation
                local band_atten = (band / 3.0) * atten
                local local_mix  = effective_mix * (1.0 - clamp(band_atten, 0.0, 0.8))
                x_final[j] = (1.0 - local_mix) * x_next_storm[j] + local_mix * x_next_anchor[j]
            end
        else
            for j = 0, n - 1 do
                x_final[j] = (1.0 - effective_mix) * x_next_storm[j] + effective_mix * x_next_anchor[j]
            end
        end

        -- Post-blend NaN guard
        if has_nan_inf(x_final, n) then
            if verbose then print(string.format("[CONFLUENCE] NaN post-blend step %d, using STORM", step_idx)) end
            for j = 0, n - 1 do x_final[j] = x_next_storm[j] end
        end
        for j = 0, n - 1 do x_final[j] = clamp(x_final[j], -sclamp, sclamp) end

        -- ── POST-BLEND LOOK-BACK ─────────────────────────────────────────
        -- SNR-adaptive EMA on x_final itself. Covers the blend seam that
        -- neither sub-solver's own look-back touches.
        if pb_enabled and pb_prev ~= nil then
            local ratio = clamp(sigma_curr / math.max(sigma_max, EPSILON), 0.0, 1.0)
            local pb_w  = pb_lb_lambda * (ratio ^ pb_lb_snr)
            if pb_w > 1e-6 then
                for j = 0, n - 1 do
                    x_final[j] = (1.0 - pb_w) * x_final[j] + pb_w * pb_prev[j]
                end
            end
        end
        if pb_enabled then pb_prev = vec_clone(x_final, n) end

        -- ── STATE FEEDBACK ───────────────────────────────────────────────
        -- Anchor gets the blended result, not its own unblended candidate
        anchor_state_feedback(x_final, step_idx, past_warmup, n)

        -- ── SDE NOISE (post-blend, same convention as anchor) ────────────
        if eta > 0.0 and sigma_next > EPSILON then
            local rng   = make_rng(seed + step_idx * 7919)
            local scale = sigma_next * eta
            for j = 0, n - 1 do
                local u1 = math.max(rng(), EPSILON)
                local u2 = rng()
                x_final[j] = x_final[j] + normal(u1, u2) * scale
            end
        end

        -- ── VERBOSE ──────────────────────────────────────────────────────
        if verbose then
            print(string.format(
                "[CONFLUENCE] step %02d %-5s RK%d | agree=%.3f mag=%.3f inertia=%.3f mix=%d->%.3f mode=%s t=%.2f",
                step_idx, storm_mode, actual_order, agreement, mag_ratio,
                inertia_mag, mix_amount, effective_mix, active_mode, t_frac))
        end

        x = x_final
        tbl_to_fa(x, xt, n)
        tbl_to_fa(v_curr, vt_buf, n)

        if on_step(step_idx, sigma_curr, sigma_next) then return end
        x = fa_to_tbl(xt, n)
    end

    tbl_to_fa(x, xt, n)
end
