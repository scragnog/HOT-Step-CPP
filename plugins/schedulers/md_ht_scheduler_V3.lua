-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
-- GNU General Public License for more details: https://www.gnu.org/licenses/
-- ============================================================================

-- MD HT Scheduler v5.0 — HAP + TPT Thermodynamic Timestep Schedule
-- MDMAchine | A&E Concepts © 2026
--
-- Plugin version: V3 (HOT-Step UI / filename — what users see)
-- Internal version: v5.0 (math/changelog — what developers track)
-- These are different: plugin version bumps on breaking changes or major
-- feature drops. Internal version bumps on any code change.
--
-- Sub-index CDF interpolation, parameter caching, post-CDF smoothing,
-- density floor, LINA warp, SNR-space mode, poly slope, uniformity blend,
-- verbose step-size diagnostics.
--
-- WHAT THIS DOES:
--   Standard schedulers space timesteps linearly or with a simple power curve.
--   HT uses two coupled density functions to place steps where they matter:
--
--   HAP (Hamiltonian Action-Principle):
--     density_hap = 1 / ((1 + KE * t) * exp(-DF * t))
--     High KE = front-loaded steps (aggressive early denoising)
--     High DF = fast exponential damping toward formation
--
--   TPT (Thermodynamic Phase Transition):
--     density_tpt = 1 / (|sigma - Tc| + well_width)
--     Creates a "gravity well" that clusters steps near the critical temp
--     where latent structure crystallizes.
--
--   Combined: density = density_hap + phase_intensity * density_tpt + floor
--   Result: non-uniform sigma sequence. Works at any step count (12-150+).
--
-- POST-PROCESSING CHAIN (all optional, all default off):
--   1. Shift warp (native HOT-Step sigma warp)
--   2. LINA warp (time-axis resampling from MD Causal)
--   3. Poly slope (power curve on sigma values)
--   4. Uniformity blend (blend with linear uniform schedule)
--   5. Schedule smoothing (moving average on final sigmas)
--
-- CHANGELOG:
--   v5.0: Density floor, LINA warp, SNR-space mode, poly slope, uniformity
--         blend, post-CDF smoothing, verbose diagnostics. Additive HAP+TPT
--         blending. Exposed well width. Restored descriptive header. Complete
--         rework from v4.0 baseline.
-- ============================================================================

scheduler = {
    name        = "md_ht_scheduler V3",
    display     = "MD HT Scheduler (HAP+TPT) V3",
    description = "Thermodynamic timestep schedule: HAP + TPT additive density, density floor, LINA warp, SNR-space, poly slope, uniformity blend, smoothing. 12 to 150+ steps.",
    params      = {
        -- ── HAP ─────────────────────────────────────────────────────────────
        { key = "kinetic_energy", type = "slider", label = "Kinetic Energy",
          default = 0.3, min = 0.0, max = 3.0, step = 0.05,
          hint = "HAP leading-edge sharpness. 0=uniform, 0.3=standard, 2+=aggressive front-loading." },
        { key = "damping_friction", type = "slider", label = "Damping Friction",
          default = 2.2, min = 0.0, max = 6.0, step = 0.1,
          hint = "HAP tail compression. Higher = steps cluster toward the front." },
        -- ── TPT ─────────────────────────────────────────────────────────────
        { key = "critical_temp", type = "slider", label = "Critical Temp",
          default = 0.6, min = 0.05, max = 0.95, step = 0.05,
          hint = "TPT phase transition center (sigma fraction). Steps cluster here." },
        { key = "phase_intensity", type = "slider", label = "Phase Intensity",
          default = 1.0, min = 0.0, max = 3.0, step = 0.1,
          hint = "TPT clustering strength. 0=off (pure HAP). 1=moderate. 2+=strong." },
        { key = "well_width", type = "slider", label = "Well Width",
          default = 0.25, min = 0.05, max = 0.5, step = 0.05,
          hint = "TPT softening radius. 0.1=tight. 0.25=balanced. 0.4+=broad." },
        -- ── Density Floor ───────────────────────────────────────────────────
        { key = "density_floor", type = "slider", label = "Density Floor",
          default = 0.1, min = 0.0, max = 1.0, step = 0.05,
          hint = "Minimum density everywhere. Prevents sparse gaps. 0=off. 0.1=gentle. 0.3+=uniform-leaning." },
        -- ── SNR Space ───────────────────────────────────────────────────────
        { key = "snr_space", type = "toggle", label = "SNR Space",
          default = false,
          hint = "Compute density on an SNR-uniform grid instead of sigma-uniform. Steps track perceptual importance. Better for audio at high step counts." },
        -- ── Post-Processing ─────────────────────────────────────────────────
        { key = "lina_shift", type = "slider", label = "LINA Warp",
          default = 1.0, min = 0.5, max = 2.0, step = 0.05,
          hint = "Time-axis resampling (from MD Causal). 1.0=off. <1=front-load (more high-sigma steps). >1=back-load (more low-sigma steps). Different from shift warp — this resamples WHERE on the curve, not the sigma VALUES." },
        { key = "poly_slope", type = "slider", label = "Poly Slope",
          default = 1.0, min = 0.5, max = 2.0, step = 0.05,
          hint = "Power curve on sigma values. 1.0=off. >1=compress toward zero (more detail steps, good for long runs). <1=compress toward one (more structure steps, good for 12-step turbo)." },
        { key = "uniform_blend", type = "slider", label = "Uniformity Blend",
          default = 0.0, min = 0.0, max = 1.0, step = 0.05,
          hint = "Blend with linear uniform schedule. 0=pure HT. 0.3=gentle uniformity. 1.0=pure uniform. Tames HT clustering for stabilization solvers (Trajectory Anchor)." },
        { key = "smooth_window", type = "slider", label = "Schedule Smoothing",
          default = 0, min = 0, max = 7, step = 1,
          hint = "Post-CDF moving average on final sigmas. 0=off. 3=mild. 5+=heavy. Smooths step-size transitions." },
        -- ── Engine ──────────────────────────────────────────────────────────
        { key = "dense_steps", type = "slider", label = "CDF Resolution",
          default = 1000, min = 200, max = 5000, step = 100,
          hint = "Resolution of the integration grid." },
        { key = "shift", type = "slider", label = "Shift Warp",
          default = 1.0, min = 0.5, max = 8.0, step = 0.1,
          hint = "Native HOT-Step sigma warp. Applied first in the post-processing chain." },
        { key = "verbose", type = "toggle", label = "Verbose",
          default = false,
          hint = "Print per-step sigma values, step sizes, and gap ratio to console." },
    },
}

-- ── Hoisted Buffers & Cache ──────────────────────────────────────────────────

local EPSILON = 1e-6

local _cache = { ke = -1, df = -1, tc = -1, pi = -1, ww = -1, fl = -1, snr = -1, dense_n = -1 }
local _dense = {}
local _cdf   = {}
local _sigmas = {}
local _smooth_buf = {}
local _last_num_steps = -1

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- ── SNR-Space Grid ──────────────────────────────────────────────────────────
-- Build dense grid uniform in SNR space: snr = log(sigma / (1 - sigma)).
-- Maps back to sigma via sigmoid: sigma = 1 / (1 + exp(-snr)).
-- Endpoints clamped to avoid inf at sigma=0 and sigma=1.

local function build_snr_grid(dense, dense_n, sigma_max, sigma_min)
    local s_hi = clamp(sigma_max, 0.001, 0.999)
    local s_lo = clamp(sigma_min + 0.001, 0.001, 0.999)
    local snr_hi = math.log(s_hi / (1.0 - s_hi))
    local snr_lo = math.log(s_lo / (1.0 - s_lo))

    for i = 0, dense_n - 1 do
        local snr = snr_hi + (snr_lo - snr_hi) * i / (dense_n - 1)
        dense[i] = 1.0 / (1.0 + math.exp(-snr))
    end
    dense[0] = sigma_max
    dense[dense_n - 1] = sigma_min
end

-- ── LINA Warp (from MD Causal) ──────────────────────────────────────────────
-- Time-axis resampling: t_warped = t^shift, then interpolate into the sigma
-- array at the warped position. Shift < 1 front-loads, shift > 1 back-loads.
-- Different from native shift warp which transforms sigma values directly.

local function apply_lina_warp(sigmas, n, shift)
    if shift == 1.0 then return end
    -- Read into scratch buffer first
    for i = 0, n do _smooth_buf[i] = sigmas[i] end

    for i = 0, n do
        local t = i / n
        local t_w = t ^ shift
        local raw_idx = t_w * n
        local idx_lo = clamp(math.floor(raw_idx), 0, n)
        local idx_hi = clamp(idx_lo + 1, 0, n)
        local frac = raw_idx - idx_lo
        local s_lo = _smooth_buf[idx_lo]
        local s_hi = _smooth_buf[idx_hi] or _smooth_buf[n]
        sigmas[i] = s_lo * (1.0 - frac) + s_hi * frac
    end
    sigmas[0] = _smooth_buf[0]
    sigmas[n] = _smooth_buf[n]
end

-- ── Post-CDF Schedule Smoothing ─────────────────────────────────────────────

local function smooth_schedule(sigmas, n, window)
    if window < 2 or n < window then return end
    local half = math.floor(window / 2)

    for i = 1, n - 1 do
        local sum = 0.0
        local count = 0
        for j = math.max(0, i - half), math.min(n, i + half) do
            sum = sum + sigmas[j]
            count = count + 1
        end
        _smooth_buf[i] = sum / count
    end

    for i = 1, n - 1 do
        sigmas[i] = _smooth_buf[i]
    end

    -- Enforce monotonically decreasing
    for i = 1, n do
        if sigmas[i] >= sigmas[i - 1] then
            sigmas[i] = sigmas[i - 1] - EPSILON
        end
    end
end

-- ── Core Schedule Builder ────────────────────────────────────────────────────

local function build_ht_schedule(num_steps, ke, df, tc_frac, pi, ww, fl, snr_mode, dense_n)
    local sigma_max = 1.0
    local sigma_min = 0.0
    local snr_flag = snr_mode and 1 or 0

    -- Only rebuild CDF if params changed
    if ke ~= _cache.ke or df ~= _cache.df or tc_frac ~= _cache.tc
       or pi ~= _cache.pi or ww ~= _cache.ww or fl ~= _cache.fl
       or snr_flag ~= _cache.snr or dense_n ~= _cache.dense_n then

        -- Build dense grid (sigma-uniform or SNR-uniform)
        if snr_mode then
            build_snr_grid(_dense, dense_n, sigma_max, sigma_min)
        else
            for i = 0, dense_n - 1 do
                _dense[i] = sigma_max - (sigma_max - sigma_min) * i / (dense_n - 1)
            end
        end

        local critical_temp = sigma_min + tc_frac * (sigma_max - sigma_min)
        local running = 0.0

        for i = 0, dense_n - 1 do
            local s = _dense[i]
            local t = (sigma_max - s) / (sigma_max - sigma_min + EPSILON)

            -- HAP density
            local v_hap = (1.0 + ke * t) * math.exp(-df * t)
            local d_hap = 1.0 / (v_hap + EPSILON)

            -- TPT density
            local dist_tc = math.abs(s - critical_temp)
            local d_tpt = 1.0 / (dist_tc + ww)

            -- Additive blending + density floor
            running = running + d_hap + pi * d_tpt + fl
            _cdf[i] = running
        end

        -- Normalize CDF to [0, 1]
        local cdf0 = _cdf[0]
        local cdf_n1 = _cdf[dense_n - 1]
        local range = cdf_n1 - cdf0 + EPSILON

        for i = 0, dense_n - 1 do
            _cdf[i] = (_cdf[i] - cdf0) / range
        end

        _cache.ke = ke
        _cache.df = df
        _cache.tc = tc_frac
        _cache.pi = pi
        _cache.ww = ww
        _cache.fl = fl
        _cache.snr = snr_flag
        _cache.dense_n = dense_n
    end

    -- Pre-allocate
    if num_steps > _last_num_steps then
        for i = 0, num_steps do _sigmas[i] = 0.0 end
        for i = 0, num_steps do _smooth_buf[i] = 0.0 end
        _last_num_steps = num_steps
    end

    -- Binary search
    local function searchsorted(target)
        local lo, hi = 0, dense_n - 1
        while lo < hi do
            local mid = math.floor((lo + hi) / 2)
            if _cdf[mid] < target then lo = mid + 1 else hi = mid end
        end
        return clamp(lo, 0, dense_n - 1)
    end

    -- Sub-index interpolation
    for i = 0, num_steps do
        local tgt = i / num_steps
        local idx = searchsorted(tgt)

        if idx == 0 then
            _sigmas[i] = _dense[0]
        else
            local c0 = _cdf[idx - 1]
            local c1 = _cdf[idx]
            local t_interp = (c1 > c0) and ((tgt - c0) / (c1 - c0)) or 0.0

            local d0 = _dense[idx - 1]
            local d1 = _dense[idx]
            _sigmas[i] = d0 + t_interp * (d1 - d0)
        end
    end

    -- Force exact endpoints
    _sigmas[0]         = sigma_max
    _sigmas[num_steps] = sigma_min

    return _sigmas
end

-- ── Required schedule() function ─────────────────────────────────────────────

function schedule(output, num_steps, shift_val)
    local ke       = (params and params.kinetic_energy)   or 0.3
    local df       = (params and params.damping_friction) or 2.2
    local tc_frac  = (params and params.critical_temp)    or 0.6
    local pi_      = (params and params.phase_intensity)  or 1.0
    local ww       = (params and params.well_width)       or 0.25
    local fl       = (params and params.density_floor)    or 0.1
    local snr_mode = (params and params.snr_space)        or false
    local lina     = (params and params.lina_shift)       or 1.0
    local poly     = (params and params.poly_slope)       or 1.0
    local u_blend  = (params and params.uniform_blend)    or 0.0
    local sm_win   = math.floor((params and params.smooth_window) or 0)
    local dense_n  = math.floor((params and params.dense_steps) or 1000)
    local sh       = (params and params.shift)            or shift_val
    local verbose  = (params and params.verbose)          or false

    local sigmas = build_ht_schedule(num_steps, ke, df, tc_frac, pi_, ww, fl, snr_mode, dense_n)

    -- ── POST-PROCESSING CHAIN ───────────────────────────────────────────
    -- Order: shift → LINA → poly → uniformity → smooth
    -- Each is independent and default-off. Compose cleanly at any step count.

    -- 1. Native shift warp (sigma-value transform)
    if sh ~= 1.0 then
        for i = 0, num_steps do
            local t = sigmas[i]
            sigmas[i] = sh * t / (1.0 + (sh - 1.0) * t)
        end
    end

    -- 2. LINA warp (time-axis resampling)
    if lina ~= 1.0 then
        apply_lina_warp(sigmas, num_steps, lina)
    end

    -- 3. Poly slope (power curve on sigma values)
    --    >1 = compress toward zero (more detail steps)
    --    <1 = compress toward one (more structure steps, good for 12-step turbo)
    if poly ~= 1.0 then
        for i = 1, num_steps - 1 do
            sigmas[i] = sigmas[i] ^ poly
        end
        -- Endpoints stay exact
    end

    -- 4. Uniformity blend (blend with linear schedule)
    --    Tames HT clustering for stabilization solvers
    if u_blend > 0.0 then
        local inv = 1.0 - u_blend
        for i = 0, num_steps do
            local uniform_sigma = 1.0 - (i / num_steps)
            sigmas[i] = inv * sigmas[i] + u_blend * uniform_sigma
        end
    end

    -- 5. Schedule smoothing (moving average)
    if sm_win >= 2 then
        smooth_schedule(sigmas, num_steps, sm_win)
        sigmas[0]         = 1.0
        sigmas[num_steps] = 0.0
    end

    -- ── WRITE OUTPUT (no trailing zero — engine contract) ───────────────
    for i = 0, num_steps - 1 do
        output[i] = sigmas[i]
    end

    -- ── VERBOSE ─────────────────────────────────────────────────────────
    if verbose then
        print(string.format(
            "[HT V5] %d steps | ke=%.2f df=%.1f tc=%.2f pi=%.1f ww=%.2f fl=%.2f | snr=%s lina=%.2f poly=%.2f ub=%.2f sm=%d",
            num_steps, ke, df, tc_frac, pi_, ww, fl,
            snr_mode and "on" or "off", lina, poly, u_blend, sm_win))
        local min_gap, max_gap = 1.0, 0.0
        for i = 0, num_steps - 1 do
            local sigma_next = (i < num_steps - 1) and sigmas[i + 1] or 0.0
            local gap = sigmas[i] - sigma_next
            if gap < min_gap then min_gap = gap end
            if gap > max_gap then max_gap = gap end
            print(string.format("  step %02d: sigma=%.5f  gap=%.5f", i, sigmas[i], gap))
        end
        print(string.format("[HT V5] Gap range: min=%.5f max=%.5f ratio=%.1f:1",
            min_gap, max_gap, max_gap / (min_gap + EPSILON)))
    end
end
