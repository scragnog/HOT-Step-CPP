-- ============================================================================
-- SPDX-License-Identifier: GPL-3.0-or-later
-- Copyright (C) 2026 Alexander Allan (MDMAchine) -- A&E Concepts
-- ============================================================================

-- MD Clarity V1 — Lightweight Post-CFG Cleanup Guidance
-- MDMAchine | A&E Concepts (c) 2026
--
-- Simple spectral cleanup for flow-matching audio. Tames HF harshness,
-- clamps magnitude spikes, optional orthogonal projection to keep
-- corrections perpendicular to the original signal direction.
--
-- Designed to pair with MD solvers (STORM, Confluence, Hamiltonian, etc.)
-- Drop-in guidance module. Minimal state, zero-allocation hot path.
-- ============================================================================

guidance = {
    name        = "md_clarity_v1",
    display     = "MD Clarity V1",
    description = "Lightweight post-CFG cleanup. HF smoothing, spike clamping, orthogonal projection. Pairs with MD solvers.",
    params      = {
        { key = "strength",   type = "slider", label = "Strength",
          default = 0.15, min = 0.0, max = 0.5, step = 0.01,
          hint = "Overall correction intensity. 0.10-0.20 for subtle cleanup." },
        { key = "hf_smooth",  type = "slider", label = "HF Smoothing",
          default = 0.25, min = 0.0, max = 1.0, step = 0.05,
          hint = "Laplacian HF damping. Tames harshness/metallic edge. 0=off." },
        { key = "spike_clamp", type = "slider", label = "Spike Clamp",
          default = 2.5, min = 1.0, max = 6.0, step = 0.25,
          hint = "Hard clamp on per-element magnitude relative to mean. Lower=more aggressive." },
        { key = "orthogonal", type = "toggle", label = "Orthogonal Projection",
          default = true,
          hint = "Project corrections perpendicular to original signal. Prevents reinforcing existing structure." },
        { key = "preserve_energy", type = "slider", label = "Preserve Energy",
          default = 0.0, min = 0.0, max = 0.5, step = 0.05,
          hint = "Blend output back toward original. 0=full correction, 0.5=half." },
    },
}

local EPSILON = 1e-8

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- ── HF SMOOTHING (Laplacian damping) ────────────────────────────────────────
-- Applies a simple neighbor-averaging pass weighted by `blend`.
-- Targets high-frequency oscillations without touching broadband energy.

local function smooth_hf(buf, n, blend)
    if blend <= 0.0 or n < 3 then return end

    local prev = buf[0]
    local curr = buf[0]

    for i = 0, n - 1 do
        local next_val = (i < n - 1) and buf[i + 1] or buf[i]
        curr = buf[i]
        local smoothed = (prev + curr + next_val) / 3.0
        buf[i] = curr * (1.0 - blend) + smoothed * blend
        prev = curr
    end
end

-- ── SPIKE CLAMPING ──────────────────────────────────────────────────────────
-- Clamps any element whose absolute value exceeds `threshold * mean_abs`.
-- Prevents outlier magnitudes from dominating the latent.

local function clamp_spikes(buf, n, threshold)
    if threshold <= 0.0 then return end

    local mean_abs = 0.0
    for i = 0, n - 1 do mean_abs = mean_abs + math.abs(buf[i]) end
    mean_abs = mean_abs / math.max(n, 1) + EPSILON

    local limit = mean_abs * threshold
    for i = 0, n - 1 do
        buf[i] = clamp(buf[i], -limit, limit)
    end
end

-- ── ORTHOGONAL PROJECTION ───────────────────────────────────────────────────
-- Decomposes delta into components parallel and perpendicular to the original
-- signal. Keeps only the perpendicular part (scaled to preserve magnitude).
-- Standard Gram-Schmidt, nothing exotic.

local function project_orthogonal(delta, original, n)
    local dot_do = 0.0
    local dot_oo = 0.0
    local dot_dd = 0.0

    for i = 0, n - 1 do
        dot_do = dot_do + delta[i] * original[i]
        dot_oo = dot_oo + original[i] * original[i]
        dot_dd = dot_dd + delta[i] * delta[i]
    end

    if dot_oo < EPSILON then return end

    local proj_scale = dot_do / dot_oo
    local ortho_sq = 0.0

    for i = 0, n - 1 do
        delta[i] = delta[i] - proj_scale * original[i]
        ortho_sq = ortho_sq + delta[i] * delta[i]
    end

    -- Rescale to preserve original delta magnitude
    if ortho_sq > EPSILON then
        local rescale = math.sqrt(dot_dd / ortho_sq)
        for i = 0, n - 1 do delta[i] = delta[i] * rescale end
    end
end

-- ── GUIDE ───────────────────────────────────────────────────────────────────

function guide(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)
    local n = Oc * T
    local p = params or {}

    -- HOT-Step integration fix: result is an OUTPUT buffer holding the previous
    -- step's stale velocity at entry -- guide() must produce the CFG combine
    -- itself. Route the base combine through native apg() (momentum smoothing,
    -- perpendicular projection, norm thresholding), then run the clarity
    -- cleanup on top of it -- true "post-CFG" as designed.
    apg(pred_cond, pred_uncond, guidance_scale, result, Oc, T, norm_threshold)

    local strength = clamp((p.strength or 0.15), 0.0, 0.5)
    if strength <= 0.0 then return end   -- result already holds the APG combine

    local hf_blend = clamp((p.hf_smooth or 0.25), 0.0, 1.0)
    local spike_th = clamp((p.spike_clamp or 2.5), 1.0, 6.0)
    local f_ortho  = p.orthogonal
    if f_ortho == nil then f_ortho = true end
    local preserve = clamp((p.preserve_energy or 0.0), 0.0, 0.5)

    -- 1. Snapshot original
    local original = {}
    for i = 0, n - 1 do original[i] = result[i] end

    -- 2. Compute delta (what APG/CFG added beyond unconditional)
    local delta = {}
    for i = 0, n - 1 do delta[i] = result[i] - pred_uncond[i] end

    -- 3. HF smoothing on delta
    smooth_hf(delta, n, hf_blend)

    -- 4. Spike clamping on delta
    clamp_spikes(delta, n, spike_th)

    -- 5. Orthogonal projection (keep corrections perpendicular to signal)
    if f_ortho then
        project_orthogonal(delta, original, n)
    end

    -- 6. Apply cleaned delta
    for i = 0, n - 1 do
        local cleaned = pred_uncond[i] + delta[i]
        local blended = original[i] + (cleaned - original[i]) * strength

        if preserve > 0.0 then
            result[i] = blended * (1.0 - preserve) + original[i] * preserve
        else
            result[i] = blended
        end
    end
end
