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

local C = require("md_solver_commons")

-- MD Trajectory Anchor V3 — Latent Path Stabilizer
-- MDMAchine | A&E Concepts © 2026
--
-- A stateful trajectory stabilization solver for HOT-Step-CPP.
-- Runs via step() — NOT owns_loop. The guider pipeline (APG, ADG, PMG, etc.)
-- remains fully active. Receives the pre-guided vt from the engine and applies
-- stateful corrections on top of the advancing latent.
--
-- WHY step() INSTEAD OF owns_loop:
--   All features here (inertia, concept lock, anchors, look-back) only need
--   xt — the latent tensor — which step() provides directly. owns_loop was
--   used in V1/V2 because guidance features needed cond/uncond, but those
--   have been removed. step() is the correct, minimal contract for this work.
--   Guiders run normally alongside this solver.
--
-- PIPELINE PER STEP:
--   xt → Euler advance (xt + dt * vt) → _out_buf
--       → [entropy measurement]         — Shannon H from xt (step 0+)
--       → [latent pressure]             — entropy×RMS correction (toggle, off)
--       → [memory buffer]               — 3-step ring buffer smoothing
--       → [inertia engine]              — EMA velocity carry-over
--       → [concept lock]                — stability mask, sigma-adaptive
--       → [identity anchor]             — mid-sigma snapshot pull-back
--       → [tonal anchor]                — spectral centroid correction, sigma-adaptive
--       → [look-back smoother]          — SNR-adaptive EMA (arXiv:2602.09449)
--       → [RMS servo]                   — descending RMS ceiling (toggle, off)
--       → [safety clamp + NaN guard]    — abs ceiling + Euler rollback on NaN
--       → write _out_buf to xt
--
-- STATE RESET:
--   All module-level state resets on step_index == 0 OR n change.
--   Same-length consecutive generations do not bleed state.
--
-- INERTIA EMA FIX (V2 regression):
--   V2 computed: vel = 0.8 * vel + 0.2 * vel (no-op, same buffer).
--   V1.0 uses two separate buffers: _vel_old_buf (EMA) and _vel_raw_buf (delta).
--   EMA: _vel_old_buf[i] = 0.8 * _vel_old_buf[i] + 0.2 * _vel_raw_buf[i]
--
-- SIGMA-ADAPTIVE FEATURES (from OmniRelational V3 pattern):
--   concept lock strength  = full * (sigma_ratio ^ concept_sigma_power)
--   tonal correction scale = tonal_strength * sigma_ratio
--   look-back weight       = lb_lambda * (sigma_ratio ^ lb_snr_power)
--   All three are heavy at high sigma (structure phase), fade to zero at sigma=0.
--
-- PARAMS:
--   warmup_steps           — skip stateful features for first N steps
--   inertia_engine         — EMA latent velocity carry-over
--   inertia_alpha          — base velocity coefficient, entropy-modulated
--   memory_buffer          — 3-step ring buffer output smoothing
--   memory_blend           — history blend fraction
--   concept_lock           — stability mask on settled regions
--   concept_sigma_power    — how fast lock fades with sigma
--   identity_anchor        — captures xt snapshot at anchor_sigma, pulls back
--   anchor_sigma           — sigma fraction at which anchors are captured
--   anchor_blend           — pull strength toward identity anchor
--   tonal_anchor           — spectral centroid drift correction
--   tonal_strength         — correction scale (per-element hard cap 0.1%)
--   look_back_enabled      — SNR-adaptive latent EMA smoother
--   look_back_lambda       — max smoothing weight at high sigma
--   look_back_snr_power    — falloff exponent
--   rms_servo              — descending RMS ceiling (off by default)
--   rms_target_min         — RMS ceiling at low sigma
--   rms_target_max         — RMS ceiling at high sigma
--   rms_servo_gain         — servo correction aggressiveness
--   latent_pressure        — entropy×RMS target correction (off by default)
--   pressure_target_rms    — RMS target for pressure correction
--   pressure_target_entropy — entropy target for pressure weighting
--   safety_clamp           — max absolute latent value
-- ============================================================================

solver = {
    name        = "md_trajectory_anchor_V3",
    display     = "MD Trajectory Anchor V3",
    description = "Latent path stabilizer. step() solver — guiders stay active. Inertia engine, concept lock, identity anchor, tonal anchor, memory buffer, look-back smoother, RMS servo. All stateful. State resets cleanly between generations.",
    nfe         = 1,
    order       = 1,
    needs_model = false,
    stateful    = true,
    stochastic  = true,
    params      = {
        -- ── Warmup ────────────────────────────────────────────────────────────
        {
            key     = "warmup_steps",
            type    = "slider",
            label   = "Warmup Steps",
            default = 2,
            min     = 0,
            max     = 6,
            step    = 1,
            hint    = "Skip stateful features (inertia, concept lock, anchors) for first N steps. Latent is mostly noise at high sigma — anchoring into chaos makes things worse. 2=recommended. 0=always active.",
        },
        -- ── Inertia Engine ────────────────────────────────────────────────────
        {
            key     = "inertia_engine",
            type    = "toggle",
            label   = "Inertia Engine",
            default = true,
            hint    = "EMA-smoothed latent velocity carry-over. Adds step-to-step momentum — reduces abrupt trajectory direction changes. Alpha is entropy-modulated: less inertia when latent is already structured (low entropy).",
        },
        {
            key     = "inertia_alpha",
            type    = "slider",
            label   = "Inertia Alpha",
            default = 0.15,
            min     = 0.0,
            max     = 0.5,
            step    = 0.01,
            hint    = "Base velocity carry-over coefficient. 0.10=subtle. 0.20=noticeable. 0.30+=strong. Scaled down at runtime when entropy is low (structured latent needs less push).",
        },
        -- ── Memory Buffer ─────────────────────────────────────────────────────
        {
            key     = "memory_buffer",
            type    = "toggle",
            label   = "Memory Buffer",
            default = true,
            hint    = "Blends last 3 step outputs into the current step output. Suppresses step-to-step jitter without redirecting the trajectory. Ring buffer, zero-alloc.",
        },
        {
            key     = "memory_blend",
            type    = "slider",
            label   = "Memory Blend",
            default = 0.12,
            min     = 0.0,
            max     = 0.5,
            step    = 0.01,
            hint    = "Fraction of 3-step history mean blended into each step output. 0.12=subtle. 0.25+=heavy smoothing (may soften transients in audio).",
        },
        -- ── Concept Lock ──────────────────────────────────────────────────────
        {
            key     = "concept_lock",
            type    = "toggle",
            label   = "Concept Lock",
            default = true,
            hint    = "Stability mask: elements with small step-to-step delta are pulled back toward their previous state. Protects settled structure from noise. Sigma-adaptive — full strength at high sigma, fades at low sigma (detail phase).",
        },
        {
            key     = "concept_sigma_power",
            type    = "slider",
            label   = "Concept Lock Sigma Power",
            default = 1.0,
            min     = 0.25,
            max     = 3.0,
            step    = 0.25,
            hint    = "Controls how fast concept lock fades as sigma decreases. 1.0=linear decay. 2.0=quadratic (lock concentrated on early structure steps only). 0.5=slow fade (lock persists into detail steps).",
        },
        -- ── Identity Anchor ───────────────────────────────────────────────────
        {
            key     = "identity_anchor",
            type    = "toggle",
            label   = "Identity Anchor",
            default = true,
            hint    = "Captures a snapshot of xt at anchor_sigma, then gently pulls toward it on all subsequent steps. Prevents late-stage structural drift. Tonal anchor fires at the same sigma.",
        },
        {
            key     = "anchor_sigma",
            type    = "slider",
            label   = "Anchor Sigma",
            default = 0.5,
            min     = 0.1,
            max     = 0.9,
            step    = 0.05,
            hint    = "Sigma level (as fraction of sigma_max) at which the identity and tonal anchors are captured. 0.5=mid-generation. Lower=locks in more detail. Higher=locks coarser structure only.",
        },
        {
            key     = "anchor_blend",
            type    = "slider",
            label   = "Anchor Blend",
            default = 0.08,
            min     = 0.01,
            max     = 0.30,
            step    = 0.01,
            hint    = "Pull strength toward identity anchor per step. 0.08=gentle (recommended). 0.15=noticeable. Setting too high constrains creative refinement after anchor capture.",
        },
        -- ── Tonal Anchor ──────────────────────────────────────────────────────
        {
            key     = "tonal_anchor",
            type    = "toggle",
            label   = "Tonal Anchor",
            default = true,
            hint    = "Captures spectral centroid and band energy ratios at anchor_sigma. Applies centroid drift correction and band ratio correction on subsequent steps. Sigma-adaptive — correction strength fades proportionally with sigma.",
        },
        {
            key     = "tonal_strength",
            type    = "slider",
            label   = "Tonal Strength",
            default = 0.15,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Tonal correction scale. Each per-element correction is hard-capped at 0.1% per step regardless of this value. 0.10-0.20=recommended for audio. Higher values widen the correction window but the cap still applies.",
        },
        -- ── Look-Back Smoother ────────────────────────────────────────────────
        {
            key     = "look_back_enabled",
            type    = "toggle",
            label   = "Look-Back Smoother",
            default = true,
            hint    = "SNR-adaptive latent EMA. Blends current output toward previous step output, weighted heavily at high sigma (structure), fading to zero at low sigma (detail). Suppresses ODE manifold shearing and harmonic hum. arXiv:2602.09449.",
        },
        {
            key     = "look_back_lambda",
            type    = "slider",
            label   = "Look-Back Lambda",
            default = 0.55,
            min     = 0.05,
            max     = 1.0,
            step    = 0.05,
            hint    = "Max smoothing weight at sigma=sigma_max. 0.55=25-step DDIM (default). 0.35=35-step simple. Always fades to zero at sigma=0 regardless of this value.",
        },
        {
            key     = "look_back_snr_power",
            type    = "slider",
            label   = "Look-Back SNR Power",
            default = 1.3,
            min     = 0.5,
            max     = 3.0,
            step    = 0.1,
            hint    = "Falloff exponent for look-back weight. 1.3=25-step DDIM. 1.5=35-step simple. Higher = smoothing concentrated on early structure steps only.",
        },
        -- ── RMS Servo ─────────────────────────────────────────────────────────
        {
            key     = "rms_servo",
            type    = "toggle",
            label   = "RMS Servo",
            default = false,
            hint    = "Downward-only RMS ceiling. Prevents latent energy runaway without hard clipping. Off by default — calibrate target_min and target_max for your domain before enabling. ACE-Step latents run ~2.0 RMS.",
        },
        {
            key     = "rms_target_min",
            type    = "slider",
            label   = "RMS Target Min",
            default = 1.2,
            min     = 0.1,
            max     = 3.0,
            step    = 0.05,
            hint    = "RMS ceiling at low sigma (late/detail steps). ACE-Step latents ~2.0 RMS at x0. Start at 1.2-1.8 and observe results.",
        },
        {
            key     = "rms_target_max",
            type    = "slider",
            label   = "RMS Target Max",
            default = 2.5,
            min     = 0.5,
            max     = 5.0,
            step    = 0.05,
            hint    = "RMS ceiling at high sigma (early/structure steps). Should be >= target_min. ACE-Step early sigma ~2.5-3.5. Servo only fires downward.",
        },
        {
            key     = "rms_servo_gain",
            type    = "slider",
            label   = "RMS Servo Gain",
            default = 0.6,
            min     = 0.1,
            max     = 1.0,
            step    = 0.05,
            hint    = "Servo correction aggressiveness. 0.6=gradual correction. 1.0=hard snap to target each step. Lower is smoother but slower to converge.",
        },
        -- ── Latent Pressure ───────────────────────────────────────────────────
        {
            key     = "latent_pressure",
            type    = "toggle",
            label   = "Latent Pressure",
            default = false,
            hint    = "Applies a small per-step RMS correction weighted by Shannon entropy. Nudges latent toward a healthy entropy×RMS product. Off by default — tune target params before enabling. Correction capped at 0.05% per step.",
        },
        {
            key     = "pressure_target_rms",
            type    = "slider",
            label   = "Pressure Target RMS",
            default = 2.0,
            min     = 0.5,
            max     = 4.0,
            step    = 0.1,
            hint    = "RMS component of pressure target. ACE-Step ~2.0. Correction direction flips if current entropy×RMS is above target.",
        },
        {
            key     = "pressure_target_entropy",
            type    = "slider",
            label   = "Pressure Target Entropy",
            default = 7.5,
            min     = 1.0,
            max     = 15.0,
            step    = 0.5,
            hint    = "Shannon entropy component of pressure target. 7.5=image-domain default. Audio domain may differ — run with verbose output and measure entropy distribution before setting this.",
        },
        -- ── SDE Noise ─────────────────────────────────────────────────────────
        {
            key     = "relational_weight",
            type    = "slider",
            label   = "Relational Weight",
            default = 0.0,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Barbour Best Matching velocity decomposition. 0 = off. 0.3-0.5 = balanced.",
        },
        {
            key     = "relational_sigma_power",
            type    = "slider",
            label   = "Relational Sigma Decay",
            default = 1.0,
            min     = 0.25,
            max     = 4.0,
            step    = 0.25,
            hint    = "How fast relational weight fades. 1.0 = linear.",
        },
        {
            key     = "eta",
            type    = "slider",
            label   = "Eta (SDE Noise)",
            default = 0.0,
            min     = 0.0,
            max     = 1.0,
            step    = 0.05,
            hint    = "Ancestral noise injection. 0=deterministic ODE (default). Scales with t_prev each step. Low values (0.05-0.15) add subtle stochasticity without overwhelming the stabilization features.",
        },
        {
            key     = "seed",
            type    = "slider",
            label   = "Seed",
            default = 42,
            min     = 0,
            max     = 999999,
            step    = 1,
            hint    = "RNG seed for SDE noise. Deterministic per-step via seed + step_index * 7919.",
        },
        -- ── Safety ────────────────────────────────────────────────────────────
        {
            key     = "safety_clamp",
            type    = "slider",
            label   = "Safety Clamp",
            default = 2.5,
            min     = 1.0,
            max     = 5.0,
            step    = 0.1,
            hint    = "Max absolute latent value after all corrections. NaN/Inf triggers a full rollback to raw Euler output before clamping. 2.5=standard. Raise to 4.0+ if clamping is audible.",
        },
    },
}

-- ── Constants ─────────────────────────────────────────────────────────────────
local EPSILON      = 1e-8
local PRESSURE_CAP = 5e-4   -- max pressure correction per step (0.05%)

local function make_rng(seed)
    local state = math.floor(seed) % 2147483647
    if state <= 0 then state = state + 2147483646 end
    return function()
        state = (state * 1664525 + 1013904223) % 2147483648
        return state / 2147483648.0
    end
end

local function normal(u1, u2)
    return math.sqrt(-2.0 * math.log(math.max(u1, EPSILON))) * math.cos(2.0 * math.pi * u2)
end

-- ── Hoisted Buffers (Zero Allocation Hot Loop) ────────────────────────────────
-- Sized on first run or n-change. Reused every step — no GC pressure.
local _last_n        = 0
local _out_buf       = {}   -- working output for this step
local _fallback_buf  = {}   -- raw Euler output (NaN rollback)
local _vel_old_buf   = {}   -- EMA velocity (carries across steps)
local _vel_raw_buf   = {}   -- raw velocity delta (computed this step)
local _anchor_buf    = {}   -- identity anchor snapshot (frozen at anchor_sigma)
local _prev_out_buf  = {}   -- previous step final output (inertia + concept lock + look-back)
local _hist_mean_buf = {}   -- history mean scratch
local _history       = { {}, {}, {} }   -- ring buffer (3 slots, 0-indexed elements)

-- ── Module State (reset on n change or step_index == 0) ───────────────────────
local _sigma_max          = nil
local _has_prev           = false   -- true after first step output is stored
local _has_velocity       = false   -- true after first EMA velocity is initialized
local _has_anchor         = false   -- true after identity anchor is captured
local _tonal_ref_centroid = nil
local _tonal_ref_bands    = nil
local _last_entropy       = 7.5
local _hist_head          = 1
local _hist_count         = 0

-- ── Helpers ───────────────────────────────────────────────────────────────────

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function bool_param(p, key, default)
    if p == nil or p[key] == nil then return default end
    return p[key]
end

local function num_param(p, key, default)
    if p == nil or p[key] == nil then return default end
    return tonumber(p[key]) or default
end

local function rms(a, n)
    local s = 0.0
    for i = 0, n - 1 do s = s + a[i] * a[i] end
    return math.sqrt(s / n + EPSILON)
end

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

local function spectral_centroid(a, n)
    local sum_mag, sum_w = 0.0, 0.0
    for i = 0, n - 1 do
        local m = math.abs(a[i])
        sum_mag = sum_mag + m
        sum_w   = sum_w   + m * i
    end
    if sum_mag < EPSILON then return 0.0 end
    return sum_w / sum_mag
end

local function band_energy(a, n)
    local bands = {0.0, 0.0, 0.0, 0.0}
    local bsize = math.floor(n / 4)
    for b = 0, 3 do
        local s  = 0.0
        local lo = b * bsize
        local hi = (b == 3) and (n - 1) or (lo + bsize - 1)
        for i = lo, hi do s = s + math.abs(a[i]) end
        bands[b + 1] = s / math.max(hi - lo + 1, 1)
    end
    return bands
end

local function is_safe(a, n)
    for i = 0, n - 1 do
        local v = a[i]
        if v ~= v or v == math.huge or v == -math.huge then return false end
    end
    return true
end

-- ── step() ────────────────────────────────────────────────────────────────────

function step(xt, vt, t_curr, t_prev, n)

    -- ── 0. Read params ────────────────────────────────────────────────────────
    local warmup        = math.floor(num_param(params, "warmup_steps", 2))
    local f_inertia     = bool_param(params, "inertia_engine", true)
    local inertia_a     = num_param(params, "inertia_alpha", 0.15)
    local f_memory      = bool_param(params, "memory_buffer", true)
    local mem_blend     = num_param(params, "memory_blend", 0.12)
    local f_concept     = bool_param(params, "concept_lock", true)
    local concept_power = num_param(params, "concept_sigma_power", 1.0)
    local f_anchor      = bool_param(params, "identity_anchor", true)
    local anchor_sigma  = num_param(params, "anchor_sigma", 0.5)
    local anchor_blend  = num_param(params, "anchor_blend", 0.08)
    local f_tonal       = bool_param(params, "tonal_anchor", true)
    local tonal_str     = num_param(params, "tonal_strength", 0.15)
    local f_lookback    = bool_param(params, "look_back_enabled", true)
    local lb_lambda     = num_param(params, "look_back_lambda", 0.55)
    local lb_snr_power  = num_param(params, "look_back_snr_power", 1.3)
    local f_rms         = bool_param(params, "rms_servo", false)
    local rms_tgt_min   = num_param(params, "rms_target_min", 1.2)
    local rms_tgt_max   = num_param(params, "rms_target_max", 2.5)
    local rms_gain      = num_param(params, "rms_servo_gain", 0.6)
    local f_pressure    = bool_param(params, "latent_pressure", false)
    local p_tgt_rms     = num_param(params, "pressure_target_rms", 2.0)
    local p_tgt_entropy = num_param(params, "pressure_target_entropy", 7.5)
    local eta           = num_param(params, "eta", 0.0)
    local seed          = math.floor(num_param(params, "seed", 42))
    local sclamp        = num_param(params, "safety_clamp", 2.5)
    local rw            = num_param(params, "relational_weight", 0.0)
    local rw_sig_pow    = num_param(params, "relational_sigma_power", 1.0)

    local step_idx = step_index or 0

    -- ── 1. State reset (generation start or n change) ─────────────────────────
    -- n change: new latent shape (different duration/channels)
    -- step_idx == 0: new generation with same shape — must reset or prev
    --                generation's final state bleeds into next run's step 1
    if n ~= _last_n or step_idx == 0 then
        _sigma_max          = nil
        _has_prev           = false
        _has_velocity       = false
        _has_anchor         = false
        _tonal_ref_centroid = nil
        _tonal_ref_bands    = nil
        _last_entropy       = 7.5
        _hist_head          = 1
        _hist_count         = 0
        -- Resize hoisted buffers
        for i = 0, n - 1 do
            _out_buf[i]       = 0.0
            _fallback_buf[i]  = 0.0
            _vel_old_buf[i]   = 0.0
            _vel_raw_buf[i]   = 0.0
            _anchor_buf[i]    = 0.0
            _prev_out_buf[i]  = 0.0
            _hist_mean_buf[i] = 0.0
            _history[1][i]    = 0.0
            _history[2][i]    = 0.0
            _history[3][i]    = 0.0
        end
        _last_n = n
    end

    -- Capture sigma_max on first step of this generation
    if _sigma_max == nil then _sigma_max = t_curr end

    -- sigma_ratio: 1.0 at high sigma (early), 0.0 at sigma=0 (final step)
    local sigma_ratio = clamp(t_curr / math.max(_sigma_max, EPSILON), 0.0, 1.0)

    -- Warmup gate: stateful features are skipped for first `warmup` steps
    local past_warmup = (step_idx >= warmup)

    -- ── 2. Entropy measurement (always, from step 0) ──────────────────────────
    -- Measured from xt (input), not the output. Represents current latent state.
    _last_entropy = shannon_entropy(xt, n)

    -- ── 2b. Relational velocity decomposition ──────────────────────────────
    -- vt is read-only FloatArray, so we create a local velocity reference
    local vel = vt  -- default: use vt directly (no copy overhead when rw=0)
    if rw > 0 and _sigma_max ~= nil then
        local v_tbl = {}
        for i = 0, n - 1 do v_tbl[i] = vt[i] end
        local x_tbl = {}
        for i = 0, n - 1 do x_tbl[i] = xt[i] end
        C.apply_relational(v_tbl, n, 1, n, sigma_ratio, _sigma_max,
            rw, rw_sig_pow, false, 0.85, x_tbl)
        vel = v_tbl
    end

    -- ── 3. Euler advance ──────────────────────────────────────────────────────
    -- dt = t_prev - t_curr. t decrements each step, so dt < 0 (standard).
    -- x_next = xt + dt * vel
    local dt = t_prev - t_curr
    for i = 0, n - 1 do
        local v = xt[i] + dt * vel[i]
        _out_buf[i]      = v
        _fallback_buf[i] = v   -- save raw Euler for NaN rollback
    end

    -- ── 4. Latent Pressure (always if enabled, from step 0) ───────────────────
    -- Nudges latent RMS toward pressure_target_rms, weighted by entropy proximity
    -- to pressure_target_entropy. Correction hard-capped at PRESSURE_CAP per step.
    if f_pressure then
        local cur_rms        = rms(_out_buf, n)
        local target_product = p_tgt_entropy * p_tgt_rms
        local cur_product    = _last_entropy  * cur_rms
        local correction     = clamp(
            (target_product - cur_product) / (target_product + EPSILON),
            -PRESSURE_CAP, PRESSURE_CAP
        )
        if math.abs(correction) > 1e-6 then
            for i = 0, n - 1 do _out_buf[i] = _out_buf[i] * (1.0 + correction) end
        end
    end

    -- ── Stateful features below: all gated on past_warmup AND _has_prev ────────

    -- ── 5. Memory Buffer ──────────────────────────────────────────────────────
    -- Blends mean of last 3 step outputs into current output.
    -- Ring buffer: _hist_head cycles 1→2→3→1. _hist_count tracks fill level.
    if f_memory and past_warmup and _hist_count > 0 then
        for i = 0, n - 1 do _hist_mean_buf[i] = 0.0 end
        local hw = 1.0 / _hist_count
        for h = 1, _hist_count do
            for i = 0, n - 1 do _hist_mean_buf[i] = _hist_mean_buf[i] + _history[h][i] end
        end
        for i = 0, n - 1 do
            _out_buf[i] = (1.0 - mem_blend) * _out_buf[i] + mem_blend * (_hist_mean_buf[i] * hw)
        end
    end

    -- ── 6. Inertia Engine ─────────────────────────────────────────────────────
    -- EMA velocity = smoothed step-to-step output delta.
    -- Velocity raw this step: _out_buf - _prev_out_buf (output delta).
    -- EMA update: vel_old = 0.8 * vel_old + 0.2 * vel_raw  (two separate buffers)
    -- Alpha entropy-modulated: less inertia when latent is structured (low H).
    if f_inertia and past_warmup and _has_prev then
        -- Compute raw velocity delta into _vel_raw_buf
        for i = 0, n - 1 do _vel_raw_buf[i] = _out_buf[i] - _prev_out_buf[i] end
        -- EMA update or initialization
        if _has_velocity then
            for i = 0, n - 1 do
                _vel_old_buf[i] = 0.8 * _vel_old_buf[i] + 0.2 * _vel_raw_buf[i]
            end
        else
            for i = 0, n - 1 do _vel_old_buf[i] = _vel_raw_buf[i] end
            _has_velocity = true
        end
        -- Alpha modulated by entropy: low entropy (structured) → less inertia
        local alpha = inertia_a * clamp(_last_entropy / 7.5, 0.0, 1.5)
        for i = 0, n - 1 do _out_buf[i] = _out_buf[i] + alpha * _vel_old_buf[i] end
    end

    -- ── 7. Concept Lock ───────────────────────────────────────────────────────
    -- Stability mask: elements with small step-to-step delta get pulled back
    -- toward their previous state. Sigmoid-shaped lock weight per element.
    -- Sigma-adaptive: lock_w scaled by (sigma_ratio ^ concept_sigma_power)
    -- → full effect at high sigma, fades to zero at sigma=0.
    if f_concept and past_warmup and _has_prev then
        local sigma_mod = sigma_ratio ^ concept_power
        if sigma_mod > 1e-4 then
            for i = 0, n - 1 do
                local delta  = math.abs(_out_buf[i] - _prev_out_buf[i])
                -- Sigmoid: regions with delta < ~0.05 get near-full lock
                local lock_w = (1.0 / (1.0 + math.exp(delta * 40.0 - 2.0))) * sigma_mod
                _out_buf[i]  = (1.0 - lock_w) * _out_buf[i] + lock_w * _prev_out_buf[i]
            end
        end
    end

    -- ── 8. Identity Anchor ────────────────────────────────────────────────────
    -- Captures _out_buf snapshot when sigma_ratio crosses anchor_sigma threshold.
    -- On subsequent steps: gentle pull back toward the captured snapshot.
    -- Anchor sigma is a ratio of sigma_max (same as OmniRelational pattern).
    if f_anchor and past_warmup then
        if not _has_anchor and sigma_ratio <= anchor_sigma then
            -- Capture snapshot
            for i = 0, n - 1 do _anchor_buf[i] = _out_buf[i] end
            _has_anchor = true
        elseif _has_anchor then
            for i = 0, n - 1 do
                _out_buf[i] = (1.0 - anchor_blend) * _out_buf[i] + anchor_blend * _anchor_buf[i]
            end
        end
    end

    -- ── 9. Tonal Anchor ───────────────────────────────────────────────────────
    -- Captures spectral centroid and 4-band energy ratios at anchor_sigma.
    -- Correction: per-element tilt for centroid drift + per-band ratio correction.
    -- Each per-element correction hard-capped at ±0.1% regardless of tonal_str.
    -- Sigma-adaptive: effective_str = tonal_str * sigma_ratio
    -- → full correction just after capture, fades to zero at sigma=0.
    if f_tonal and past_warmup then
        if _tonal_ref_centroid == nil and sigma_ratio <= anchor_sigma then
            -- Capture reference (fires same step as identity anchor)
            _tonal_ref_centroid = spectral_centroid(_out_buf, n)
            _tonal_ref_bands    = band_energy(_out_buf, n)
        elseif _tonal_ref_centroid ~= nil then
            -- Sigma-adaptive correction scale
            local eff_str = tonal_str * sigma_ratio
            if eff_str > 1e-6 then
                local curr_centroid = spectral_centroid(_out_buf, n)
                local curr_bands    = band_energy(_out_buf, n)

                -- Centroid drift: linear tilt across elements, capped at 0.1%
                local drift_norm  = (curr_centroid - _tonal_ref_centroid) /
                                    (math.abs(_tonal_ref_centroid) + EPSILON)
                local tilt        = clamp(-drift_norm * eff_str, -1e-3, 1e-3)
                local center      = (n - 1) / 2.0
                for i = 0, n - 1 do
                    local dist_w = (i - center) / (center + EPSILON)
                    _out_buf[i]  = _out_buf[i] + tilt * dist_w * math.abs(_out_buf[i])
                end

                -- Band energy ratio correction, capped at 0.1% per band
                local ref_total, curr_total = 0.0, 0.0
                for b = 1, 4 do
                    ref_total  = ref_total  + _tonal_ref_bands[b]
                    curr_total = curr_total + curr_bands[b]
                end
                if ref_total > EPSILON and curr_total > EPSILON then
                    local bsize = math.floor(n / 4)
                    for b = 0, 3 do
                        local ref_ratio  = _tonal_ref_bands[b + 1] / ref_total
                        local curr_ratio = curr_bands[b + 1]       / curr_total
                        local band_corr  = clamp((ref_ratio - curr_ratio) * eff_str, -1e-3, 1e-3)
                        local lo = b * bsize
                        local hi = (b == 3) and (n - 1) or (lo + bsize - 1)
                        for i = lo, hi do
                            _out_buf[i] = _out_buf[i] + band_corr * math.abs(_out_buf[i])
                        end
                    end
                end
            end
        end
    end

    -- ── 10. Look-Back Smoother ────────────────────────────────────────────────
    -- SNR-adaptive EMA: lb_w = lb_lambda * (sigma_ratio ^ lb_snr_power)
    -- Blends current output toward previous step output.
    -- Heavy at high sigma (structure coherence), zero at sigma=0 (preserve detail).
    -- Pattern from MD PingPong. arXiv:2602.09449.
    if f_lookback and past_warmup and _has_prev then
        local lb_w = lb_lambda * (sigma_ratio ^ lb_snr_power)
        if lb_w > 1e-6 then
            for i = 0, n - 1 do
                _out_buf[i] = (1.0 - lb_w) * _out_buf[i] + lb_w * _prev_out_buf[i]
            end
        end
    end

    -- ── 11. RMS Servo ─────────────────────────────────────────────────────────
    -- Downward-only RMS ceiling: fires only when cur_rms > rms_target.
    -- Target descends from rms_target_max (high sigma) to rms_target_min (low sigma).
    -- Curve: target = min + sigma_ratio^0.6 * (max - min)  (from PingPong).
    -- Pattern from MD PingPong.
    if f_rms then
        local rms_target = rms_tgt_min + (sigma_ratio ^ 0.6) * (rms_tgt_max - rms_tgt_min)
        local cur_rms    = rms(_out_buf, n)
        if cur_rms > rms_target then
            local servo_rms = cur_rms + rms_gain * (rms_target - cur_rms)
            local scale     = servo_rms / cur_rms
            for i = 0, n - 1 do _out_buf[i] = _out_buf[i] * scale end
        end
    end

    -- ── 12. Safety Clamp + NaN Guard ──────────────────────────────────────────
    -- NaN/Inf in output: roll back to raw Euler result before clamping.
    -- Abs ceiling applied regardless.
    if not is_safe(_out_buf, n) then
        for i = 0, n - 1 do _out_buf[i] = _fallback_buf[i] end
    end
    for i = 0, n - 1 do _out_buf[i] = clamp(_out_buf[i], -sclamp, sclamp) end

    -- ── 13. Update state ──────────────────────────────────────────────────────
    -- Store this step's output as prev_out_buf for next step.
    -- Also push to memory ring buffer.
    if past_warmup then
        for i = 0, n - 1 do _prev_out_buf[i] = _out_buf[i] end
        _has_prev = true
        -- Ring buffer push
        if f_memory then
            for i = 0, n - 1 do _history[_hist_head][i] = _out_buf[i] end
            _hist_head = _hist_head + 1
            if _hist_head > 3 then _hist_head = 1 end
            if _hist_count < 3 then _hist_count = _hist_count + 1 end
        end
    end

    -- ── 14. Write output ──────────────────────────────────────────────────────
    for i = 0, n - 1 do xt[i] = _out_buf[i] end

    -- ── 15. SDE Noise Injection ───────────────────────────────────────────────
    -- Applied after write-back, outside the safety clamp, matching OmniRelational
    -- convention. scale = t_prev * eta — noise magnitude tracks current sigma level,
    -- naturally fades to zero as generation converges.
    if eta > 0.0 and t_prev > EPSILON then
        local rng   = make_rng(seed + step_idx * 7919)
        local scale = t_prev * eta
        for i = 0, n - 1 do
            local u1 = math.max(rng(), EPSILON)
            local u2 = rng()
            xt[i]    = xt[i] + normal(u1, u2) * scale
        end
    end

end
