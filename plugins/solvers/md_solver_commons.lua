-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
-- ============================================================================

-- md_solver_commons.lua -- Shared infrastructure for MD solver plugins
-- MDMAchine | A&E Concepts (c) 2026
--
-- Provides:
--   Level 1: Pure helpers (stateless math, array ops, param readers)
--   Level 2: Stateful stages (identity anchor, tonal anchor, look-back,
--            RMS servo, SDE noise, safety clamp) -- each takes state table
--   Level 3: post_advance() convenience -- calls all stages in order
--   Param defs: standard param definitions solvers can append
--
-- DEFAULTS REVISED (2026-07-14 listening tests, Rob/scragnog):
--   Look-back floor REMOVED -- the 0.15 floor never faded out and smeared the
--   final detail steps (main garble source). Look-back now fades to zero.
--   look_back_enabled, identity_anchor, rms_servo: default OFF (opt-in).
--   anchor_blend back to 0.08 (was 0.12). Tonal ramp kept:
--   0.3 + 0.7 * (1 - sigma_ratio)
-- ============================================================================

local C = {}

C.EPSILON = 1e-8

-- ═══════════════════════════════════════════════════════════════════════════
-- LEVEL 1: PURE HELPERS
-- ═══════════════════════════════════════════════════════════════════════════

function C.clamp(x, lo, hi) return math.max(lo, math.min(hi, x)) end

function C.smoothstep(x, lo, hi)
    if hi <= lo then return (x >= hi) and 1.0 or 0.0 end
    local t = C.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
end

function C.vec_norm(v, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + v[i] * v[i] end
    return math.sqrt(s)
end

function C.vec_dot(a, b, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + a[i] * b[i] end
    return s
end

function C.vec_sub_norm(a, b, n)
    local s = 0.0
    for i = 0, n - 1 do local d = a[i] - b[i]; s = s + d * d end
    return math.sqrt(s)
end

function C.cosine_sim(a, b, n)
    local dot, na, nb = 0.0, 0.0, 0.0
    for i = 0, n - 1 do
        dot = dot + a[i] * b[i]
        na  = na  + a[i] * a[i]
        nb  = nb  + b[i] * b[i]
    end
    return dot / (math.sqrt(na) * math.sqrt(nb) + C.EPSILON)
end

function C.shannon_entropy(a, n)
    local sum_abs = 0.0
    for i = 0, n - 1 do sum_abs = sum_abs + math.abs(a[i]) end
    if sum_abs < C.EPSILON then return 0.0 end
    local H = 0.0
    for i = 0, n - 1 do
        local p = math.abs(a[i]) / sum_abs
        if p > C.EPSILON then H = H - p * math.log(p) end
    end
    return H
end

function C.vec_clone(v, n)
    local c = {}
    for i = 0, n - 1 do c[i] = v[i] end
    return c
end

function C.fa_to_tbl(fa, n)
    local t = {}
    for i = 0, n - 1 do t[i] = fa[i] end
    return t
end

function C.tbl_to_fa(t, fa, n)
    for i = 0, n - 1 do fa[i] = t[i] end
end

function C.has_nan_inf(v, n)
    for i = 0, n - 1 do
        if v[i] ~= v[i] or math.abs(v[i]) == math.huge then return true end
    end
    return false
end

function C.rms_range(a, off, cnt)
    local s = 0.0
    for i = off, off + cnt - 1 do s = s + a[i] * a[i] end
    return math.sqrt(s / math.max(cnt, 1) + C.EPSILON)
end

function C.rms(a, n) return C.rms_range(a, 0, n) end

function C.spectral_centroid(a, off, cnt)
    local sum_mag, sum_w = 0.0, 0.0
    for i = 0, cnt - 1 do
        local m = math.abs(a[off + i])
        sum_mag = sum_mag + m
        sum_w   = sum_w + m * i
    end
    if sum_mag < C.EPSILON then return 0.0 end
    return sum_w / sum_mag
end

function C.band_energy(a, off, cnt)
    local bands = {0.0, 0.0, 0.0, 0.0}
    local bsize = math.floor(cnt / 4)
    for b = 0, 3 do
        local s  = 0.0
        local lo = off + b * bsize
        local hi = (b == 3) and (off + cnt - 1) or (lo + bsize - 1)
        for i = lo, hi do s = s + math.abs(a[i]) end
        bands[b + 1] = s / math.max(hi - lo + 1, 1)
    end
    return bands
end

function C.make_rng(seed)
    local state = math.floor(seed) % 2147483647
    if state <= 0 then state = state + 2147483646 end
    return function()
        state = (state * 1664525 + 1013904223) % 2147483648
        return state / 2147483648.0
    end
end

function C.normal(u1, u2)
    return math.sqrt(-2.0 * math.log(math.max(u1, C.EPSILON))) * math.cos(2.0 * math.pi * u2)
end

function C.num_param(p, key, default)
    if p == nil or p[key] == nil then return default end
    return tonumber(p[key]) or default
end

function C.bool_param(p, key, default)
    if p == nil or p[key] == nil then return default end
    return p[key]
end

-- Batch routing: reads engine globals, returns B, NPB with sanity fallback
function C.get_batch_routing(n)
    local B   = (batch_n and batch_n > 0) and batch_n or 1
    local NPB = (n_per and n_per > 0) and n_per or n
    if B * NPB ~= n then B = 1; NPB = n end
    return B, NPB
end

-- ═══════════════════════════════════════════════════════════════════════════
-- RELATIONAL DECOMPOSITION (from OmniRelational V3)
-- Barbour Best Matching: separates velocity into unit direction (shape)
-- and magnitude (scale). Blends shape-recomposed velocity with raw velocity
-- using sigma-adaptive weight. Prevents any single latent component from
-- dominating. Optional Gram-Schmidt drift guard.
-- ═══════════════════════════════════════════════════════════════════════════

-- Per-batch shape decomposition: returns unit direction + L2 norm
local function decompose_shape_batch(v, off, cnt)
    local s = 0.0
    for i = off, off + cnt - 1 do s = s + v[i] * v[i] end
    local norm = math.sqrt(s + C.EPSILON)
    local inv = (norm > C.EPSILON) and (1.0 / norm) or 0.0
    local shape = {}
    for i = 0, cnt - 1 do shape[i] = v[off + i] * inv end
    return shape, norm
end

-- Optional drift guard: projects shape onto orthogonal complement of x
local function drift_guard_batch(shape, x, off, cnt, threshold)
    local norm_x = 0.0
    for i = off, off + cnt - 1 do norm_x = norm_x + x[i] * x[i] end
    norm_x = math.sqrt(norm_x + C.EPSILON)
    if norm_x < C.EPSILON then return shape end

    local inv_x = 1.0 / norm_x
    local dot_sx = 0.0
    for i = 0, cnt - 1 do dot_sx = dot_sx + shape[i] * (x[off + i] * inv_x) end

    if math.abs(dot_sx) <= threshold then return shape end

    local proj = {}
    for i = 0, cnt - 1 do proj[i] = shape[i] - dot_sx * (x[off + i] * inv_x) end

    local proj_norm = 0.0
    for i = 0, cnt - 1 do proj_norm = proj_norm + proj[i] * proj[i] end
    proj_norm = math.sqrt(proj_norm + C.EPSILON)
    if proj_norm < C.EPSILON then return shape end

    local inv_proj = 1.0 / proj_norm
    for i = 0, cnt - 1 do proj[i] = proj[i] * inv_proj end
    return proj
end

-- Apply relational decomposition to velocity (in-place, per-batch).
-- v_out is modified: blends shape-recomposed velocity with raw velocity.
-- x_curr needed only when drift_guard is enabled.
function C.apply_relational(v_out, n, B, NPB, sigma_ratio, sigma_max,
                             rw, sigma_power, drift_on, drift_thr, x_curr)
    if rw < 1e-6 then return v_out end

    -- Sigma-adaptive relational weight: fades toward pure raw at low sigma
    local sr = math.min(sigma_ratio, 1.0)
    local rw_eff = rw * (sr ^ sigma_power)
    if rw_eff < 1e-6 then return v_out end

    for b = 0, B - 1 do
        local off = b * NPB

        -- Decompose into shape (unit direction) and scale (L2 norm)
        local shape, scale = decompose_shape_batch(v_out, off, NPB)

        -- Optional drift guard
        if drift_on and drift_thr < 1.0 and x_curr ~= nil then
            shape = drift_guard_batch(shape, x_curr, off, NPB, drift_thr)
        end

        -- Blend: rw_eff * (shape * scale) + (1 - rw_eff) * raw
        for i = 0, NPB - 1 do
            local rel_v = shape[i] * scale
            v_out[off + i] = rw_eff * rel_v + (1.0 - rw_eff) * v_out[off + i]
        end
    end

    return v_out
end

-- ═══════════════════════════════════════════════════════════════════════════
-- STATE MANAGEMENT
-- ═══════════════════════════════════════════════════════════════════════════

-- Creates a fresh state table for one solver run.
-- Solvers call this once at the top of sample() and pass it to stage functions.
function C.new_state()
    return {
        -- Identity anchor
        has_anchor    = false,
        id_buf        = {},
        -- Tonal anchor
        tonal_captured  = false,
        tonal_ref_cent  = {},
        tonal_ref_bands = {},
        -- Look-back (primary)
        lb_prev       = nil,
        -- Look-back (secondary, for solvers that need two)
        lb2_prev      = nil,
    }
end

-- ═══════════════════════════════════════════════════════════════════════════
-- LEVEL 2: STATEFUL STAGES
-- Each operates on x_new (table, 0-indexed), modifies in place, returns x_new.
-- ═══════════════════════════════════════════════════════════════════════════

-- Identity anchor: captures snapshot at anchor_sigma, then pulls back
function C.apply_identity_anchor(x_new, n, sigma_ratio, anchor_sigma, anchor_blend, state)
    if not state.has_anchor and sigma_ratio <= anchor_sigma then
        state.id_buf = C.vec_clone(x_new, n)
        state.has_anchor = true
    elseif state.has_anchor then
        for j = 0, n - 1 do
            x_new[j] = (1.0 - anchor_blend) * x_new[j] + anchor_blend * state.id_buf[j]
        end
    end
    return x_new
end

-- Tonal anchor: per-batch spectral centroid + 4-band energy ratio correction
-- Uses detail-phase ramp: 0.3 + 0.7 * (1 - sigma_ratio)
function C.apply_tonal_anchor(x_new, n, B, NPB, sigma_ratio, anchor_sigma, tonal_str, state)
    if not state.tonal_captured and sigma_ratio <= anchor_sigma then
        for b = 0, B - 1 do
            local off = b * NPB
            state.tonal_ref_cent[b]  = C.spectral_centroid(x_new, off, NPB)
            state.tonal_ref_bands[b] = C.band_energy(x_new, off, NPB)
        end
        state.tonal_captured = true
    elseif state.tonal_captured then
        local tonal_ramp = 0.3 + 0.7 * (1.0 - sigma_ratio)
        local eff_str = tonal_str * tonal_ramp
        if eff_str > 1e-6 then
            for b = 0, B - 1 do
                local off = b * NPB

                -- Centroid drift correction
                local curr_centroid = C.spectral_centroid(x_new, off, NPB)
                local drift_norm_val = (curr_centroid - state.tonal_ref_cent[b]) /
                                   (math.abs(state.tonal_ref_cent[b]) + C.EPSILON)
                local tilt = C.clamp(-drift_norm_val * eff_str, -1e-3, 1e-3)
                local center = (NPB - 1) / 2.0
                for j = off, off + NPB - 1 do
                    local dist_w = ((j - off) - center) / (center + C.EPSILON)
                    x_new[j] = x_new[j] + tilt * dist_w * math.abs(x_new[j])
                end

                -- Band energy ratio correction
                local curr_bands = C.band_energy(x_new, off, NPB)
                local ref_total, curr_total = 0.0, 0.0
                for bb = 1, 4 do
                    ref_total  = ref_total  + state.tonal_ref_bands[b][bb]
                    curr_total = curr_total + curr_bands[bb]
                end
                if ref_total > C.EPSILON and curr_total > C.EPSILON then
                    local bsize = math.floor(NPB / 4)
                    for bb = 0, 3 do
                        local ref_ratio  = state.tonal_ref_bands[b][bb + 1] / ref_total
                        local curr_ratio = curr_bands[bb + 1]               / curr_total
                        local band_corr  = C.clamp((ref_ratio - curr_ratio) * eff_str, -1e-3, 1e-3)
                        local blo = off + bb * bsize
                        local bhi = (bb == 3) and (off + NPB - 1) or (blo + bsize - 1)
                        for j = blo, bhi do
                            x_new[j] = x_new[j] + band_corr * math.abs(x_new[j])
                        end
                    end
                end
            end
        end
    end
    return x_new
end

-- Look-back smoother, SNR-adaptive, fades to zero at low sigma
-- slot: "lb_prev" (primary) or "lb2_prev" (secondary)
function C.apply_look_back(x_new, n, sigma_ratio, lb_lambda, lb_snr_power, state, slot)
    slot = slot or "lb_prev"
    local prev = state[slot]
    if prev ~= nil then
        local lb_w = lb_lambda * (sigma_ratio ^ lb_snr_power)
        if lb_w > 1e-6 then
            for j = 0, n - 1 do
                x_new[j] = (1.0 - lb_w) * x_new[j] + lb_w * prev[j]
            end
        end
    end
    state[slot] = C.vec_clone(x_new, n)
    return x_new
end

-- RMS servo: per-batch downward-only ceiling
function C.apply_rms_servo(x_new, n, B, NPB, sigma_ratio, rms_tgt_min, rms_tgt_max, rms_gain)
    local rms_target = rms_tgt_min + (sigma_ratio ^ 0.6) * (rms_tgt_max - rms_tgt_min)
    for b = 0, B - 1 do
        local off = b * NPB
        local cur_rms = C.rms_range(x_new, off, NPB)
        if cur_rms > rms_target then
            local servo_rms = cur_rms + rms_gain * (rms_target - cur_rms)
            local scale = servo_rms / cur_rms
            for j = off, off + NPB - 1 do x_new[j] = x_new[j] * scale end
        end
    end
    return x_new
end

-- SDE noise injection
function C.apply_sde_noise(x_new, n, sigma_next, eta, seed, step_idx)
    if eta > 0.0 and sigma_next > C.EPSILON then
        local rng   = C.make_rng(seed + step_idx * 7919)
        local scale = sigma_next * eta
        for j = 0, n - 1 do
            local u1 = math.max(rng(), C.EPSILON)
            local u2 = rng()
            x_new[j] = x_new[j] + C.normal(u1, u2) * scale
        end
    end
    return x_new
end

-- Safety clamp
function C.apply_safety_clamp(x_new, n, sclamp)
    for j = 0, n - 1 do x_new[j] = C.clamp(x_new[j], -sclamp, sclamp) end
    return x_new
end

-- ═══════════════════════════════════════════════════════════════════════════
-- LEVEL 3: POST-ADVANCE CONVENIENCE
-- Calls all stages in standard order. opts table keys:
--   f_id_anchor, anchor_sigma, anchor_blend,
--   f_tonal, tonal_str,
--   f_lookback, lb_lambda, lb_snr_power,
--   f_rms, rms_tgt_min, rms_tgt_max, rms_gain,
--   eta, seed, step_idx, sigma_next,
--   sclamp
-- ═══════════════════════════════════════════════════════════════════════════

function C.post_advance(x_new, n, B, NPB, sigma_ratio, opts, state)
    -- Identity anchor
    if opts.f_id_anchor then
        C.apply_identity_anchor(x_new, n, sigma_ratio,
            opts.anchor_sigma, opts.anchor_blend, state)
    end

    -- Tonal anchor
    if opts.f_tonal then
        C.apply_tonal_anchor(x_new, n, B, NPB, sigma_ratio,
            opts.anchor_sigma, opts.tonal_str, state)
    end

    -- Look-back
    if opts.f_lookback then
        C.apply_look_back(x_new, n, sigma_ratio,
            opts.lb_lambda, opts.lb_snr_power, state, "lb_prev")
    end

    -- RMS servo
    if opts.f_rms then
        C.apply_rms_servo(x_new, n, B, NPB, sigma_ratio,
            opts.rms_tgt_min, opts.rms_tgt_max, opts.rms_gain)
    end

    -- SDE noise
    C.apply_sde_noise(x_new, n, opts.sigma_next, opts.eta, opts.seed, opts.step_idx)

    -- Safety clamp
    C.apply_safety_clamp(x_new, n, opts.sclamp)

    return x_new
end

-- ═══════════════════════════════════════════════════════════════════════════
-- PARAM DEFINITIONS
-- Solvers call C.append_common_params(params_table) to add these.
-- Defaults reflect cross-cutting fixes (anchor_blend=0.12, rms=true).
-- ═══════════════════════════════════════════════════════════════════════════

C.RELATIONAL_PARAMS = {
    { key = "relational_weight", type = "slider", label = "Relational Weight",
      default = 0.0, min = 0.0, max = 1.0, step = 0.05,
      hint = "Barbour Best Matching: shape/scale decomposition on velocity. 0 = off (raw velocity). 0.3-0.5 = balanced. Sigma-adaptive: fades to raw at low sigma." },
    { key = "relational_sigma_power", type = "slider", label = "Relational Sigma Decay",
      default = 1.0, min = 0.25, max = 4.0, step = 0.25,
      hint = "How fast relational weight fades. 1.0 = linear. 2.0 = quadratic (faster fade)." },
    { key = "drift_guard", type = "toggle", label = "Drift Guard",
      default = false,
      hint = "Gram-Schmidt projection prevents velocity reinforcing existing latent structure." },
    { key = "drift_threshold", type = "slider", label = "Drift Threshold",
      default = 0.85, min = 0.1, max = 1.0, step = 0.05,
      hint = "Cosine similarity ceiling before drift guard fires." },
}

C.ANCHOR_PARAMS = {
    { key = "identity_anchor", type = "toggle", label = "Identity Anchor",
      default = false,
      hint = "Captures latent snapshot at anchor_sigma. Gently pulls output back." },
    { key = "anchor_sigma", type = "slider", label = "Anchor Sigma",
      default = 0.5, min = 0.1, max = 0.9, step = 0.05,
      hint = "Sigma fraction for identity/tonal anchor capture." },
    { key = "anchor_blend", type = "slider", label = "Anchor Blend",
      default = 0.08, min = 0.01, max = 0.30, step = 0.01,
      hint = "Pull strength toward identity anchor snapshot." },
    { key = "tonal_anchor", type = "toggle", label = "Tonal Anchor",
      default = true,
      hint = "Per-batch spectral centroid + 4-band energy ratio drift correction. Ramps up in detail phase." },
    { key = "tonal_strength", type = "slider", label = "Tonal Strength",
      default = 0.20, min = 0.0, max = 1.0, step = 0.05,
      hint = "Tonal correction scale. Detail-phase ramp built in. Hard-capped 0.1%/step." },
}

C.LOOKBACK_PARAMS = {
    { key = "look_back_enabled", type = "toggle", label = "Look-Back Smoother",
      default = false, hint = "SNR-adaptive latent EMA. Fades to zero at low sigma." },
    { key = "look_back_lambda", type = "slider", label = "Look-Back Lambda",
      default = 0.15, min = 0.05, max = 1.0, step = 0.05, hint = "Max smoothing at high sigma." },
    { key = "look_back_snr_power", type = "slider", label = "Look-Back SNR Power",
      default = 1.3, min = 0.5, max = 3.0, step = 0.1, hint = "Falloff exponent." },
}

C.RMS_PARAMS = {
    { key = "rms_servo", type = "toggle", label = "RMS Servo",
      default = false, hint = "Per-batch downward-only RMS ceiling. ACE-Step latents run ~2.0 RMS -- calibrate targets before enabling." },
    { key = "rms_target_min", type = "slider", label = "RMS Target Min",
      default = 1.2, min = 0.1, max = 3.0, step = 0.05, hint = "RMS ceiling at low sigma." },
    { key = "rms_target_max", type = "slider", label = "RMS Target Max",
      default = 2.5, min = 0.5, max = 5.0, step = 0.05, hint = "RMS ceiling at high sigma." },
    { key = "rms_servo_gain", type = "slider", label = "RMS Servo Gain",
      default = 0.6, min = 0.1, max = 1.0, step = 0.05, hint = "Servo correction aggressiveness." },
}

C.SDE_PARAMS = {
    { key = "eta", type = "slider", label = "Noise Injection (0 = ODE)",
      default = 0.0, min = 0.0, max = 1.0, step = 0.05, hint = "Post-step SDE noise." },
    { key = "seed", type = "slider", label = "Seed",
      default = 42, min = 0, max = 999999, step = 1, hint = "RNG seed." },
    { key = "safety_clamp", type = "slider", label = "Safety Clamp",
      default = 2.5, min = 1.0, max = 5.0, step = 0.1, hint = "Max abs latent value." },
}

C.VERBOSE_PARAM = {
    { key = "verbose", type = "toggle", label = "Verbose Logging",
      default = false, hint = "Per-step diagnostics." },
}

-- Appends param definitions to a solver's params table.
-- Usage: C.append_common_params(solver.params)
-- Adds: anchor, look-back, RMS, SDE, verbose (in that order)
function C.append_common_params(params_table)
    local sets = { C.RELATIONAL_PARAMS, C.ANCHOR_PARAMS, C.LOOKBACK_PARAMS, C.RMS_PARAMS, C.SDE_PARAMS, C.VERBOSE_PARAM }
    for _, set in ipairs(sets) do
        for _, p in ipairs(set) do
            params_table[#params_table + 1] = p
        end
    end
end

-- Reads all common params from the params global into an opts table
-- suitable for passing to post_advance().
function C.read_common_opts(p)
    return {
        -- Relational
        rw            = C.num_param(p, "relational_weight", 0.0),
        rw_sigma_pow  = C.num_param(p, "relational_sigma_power", 1.0),
        drift_on      = C.bool_param(p, "drift_guard", false),
        drift_thr     = C.num_param(p, "drift_threshold", 0.85),
        -- Anchors
        f_id_anchor   = C.bool_param(p, "identity_anchor", false),
        anchor_sigma  = C.num_param(p, "anchor_sigma", 0.5),
        anchor_blend  = C.num_param(p, "anchor_blend", 0.08),
        f_tonal       = C.bool_param(p, "tonal_anchor", true),
        tonal_str     = C.num_param(p, "tonal_strength", 0.20),
        f_lookback    = C.bool_param(p, "look_back_enabled", false),
        lb_lambda     = C.num_param(p, "look_back_lambda", 0.15),
        lb_snr_power  = C.num_param(p, "look_back_snr_power", 1.3),
        f_rms         = C.bool_param(p, "rms_servo", false),
        rms_tgt_min   = C.num_param(p, "rms_target_min", 1.2),
        rms_tgt_max   = C.num_param(p, "rms_target_max", 2.5),
        rms_gain      = C.num_param(p, "rms_servo_gain", 0.6),
        eta           = C.num_param(p, "eta", 0.0),
        seed          = math.floor(C.num_param(p, "seed", 42)),
        sclamp        = C.num_param(p, "safety_clamp", 2.5),
        verbose       = C.bool_param(p, "verbose", false),
        -- These are set per-step by the solver before calling post_advance:
        sigma_next    = 0.0,
        step_idx      = 0,
    }
end

return C
