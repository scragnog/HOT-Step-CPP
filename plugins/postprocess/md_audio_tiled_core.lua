--[[
md_audio_tiled_core.lua
MD Audio VAE Tiled Decoder — Core Math Engine

© 2026 Alexander Allan (MDMAchine) | A&E Concepts
GPL v3 — Public version.

Version: 3.0.1
PARITY: Algorithmic parity with md_audio_tiled_core.py v3.0.1
        All coefficients, thresholds, and control flow identical.

Implements the host-side tiling arithmetic and DSP chain for Lua-based
VAE runtimes (HotStep, custom scripting environments).

What this file provides:
  • Tile schedule builder (fixed and BPM-synced)
  • Fade-in window generation (Hann/Cosine/Linear)
  • Trapezoidal weight maps (dual-pass merge)
  • Latent Spectral Suppressor (LSS)
  • Biquad filter engine (peaking EQ + low shelf)
  • Hum notch chain (bass shelf + surgical cuts)
  • High-pass filter (Butterworth 2nd-order biquad)
  • Soft clipper (tanh saturation)
  • Stereo width (M/S)
  • OLA write primitive with crossfade
  • RMS leveling + absolute ceiling
  • Dual-pass trapezoidal merge

What this file does NOT provide (requires your VAE runtime):
  • vae.decode() — neural network inference
  • STFT-domain ops (HPC spectral crossfade, SCE, Wiener)
  • GPU tensor ops

TENSOR CONVENTION:
  All audio/latent buffers are flat Lua tables indexed [1..N].
  Layout: row-major [B, C, L] — B outermost, then C, then sample index.
  index(b, c, i, C, L) = (b-1)*C*L + (c-1)*L + i   (1-indexed)

IMPORTANT: This is a reference/scripting port. For production use in a
           Lua JIT environment, profile biquad inner loops and cache
           filter states across tiles.
--]]

local M = {}

-- Lua 5.4 removed math.tanh — polyfill via exp identity
local tanh = math.tanh or function(x)
    if x > 20 then return 1.0 end
    if x < -20 then return -1.0 end
    local e2x = math.exp(2 * x)
    return (e2x - 1) / (e2x + 1)
end

-- =============================================================================
-- CONSTANTS
-- =============================================================================

M.EPSILON             = 1e-8
M.GAIN_CLAMP_BASE     = 0.05      -- ±5% base RMS ride
M.GAIN_CLAMP_MAX      = 0.20      -- ±20% max on entropy spikes
M.RMS_ABS_CEIL        = 0.35      -- Hard per-tile RMS ceiling
M.DUAL_PASS_TAPER     = 0.25      -- Trapezoidal edge ramp fraction
M.MIN_OLA_SAMPLES     = 8
M.VAE_CONTEXT_FRAMES  = 128       -- Oobleck causal warm-up prefix

-- =============================================================================
-- UTILITY
-- =============================================================================

local function clamp(v, lo, hi) return math.max(lo, math.min(hi, v)) end
local function sign(v) return v > 0 and 1 or (v < 0 and -1 or 0) end

-- 1-indexed flat buffer index: [B, C, L] row-major
local function idx(b, c, i, C, L) return (b-1)*C*L + (c-1)*L + i end

-- Allocate zeroed flat table of length N
local function zeros(N)
    local t = {}
    for i = 1, N do t[i] = 0.0 end
    return t
end

local function copy(src, N)
    local t = {}
    for i = 1, N do t[i] = src[i] end
    return t
end

-- =============================================================================
-- WINDOW GENERATION
-- =============================================================================

---Fade-in ramp [0 → 1], `length` samples.
---@param length integer
---@param mode string "Hann"|"Cosine"|"Linear"
---@return table
function M.make_fade_in(length, mode)
    local w = {}
    for i = 1, length do
        local t = (i - 1) / math.max(1, length - 1)
        if mode == "Hann" then
            w[i] = 0.5 * (1 - math.cos(math.pi * t))
        elseif mode == "Cosine" then
            w[i] = math.sin(math.pi / 2 * t)
        else  -- Linear
            w[i] = t
        end
    end
    return w
end

---Trapezoidal weight: flat 1.0 centre, ramps from 0.5 at both edges.
---@param length integer
---@param edge_frac number  fraction of length used for ramp (default 0.25)
---@return table
function M.make_trapezoid(length, edge_frac)
    edge_frac = edge_frac or M.DUAL_PASS_TAPER
    local taper = math.max(M.MIN_OLA_SAMPLES, math.floor(length * edge_frac))
    taper = math.min(taper, math.floor(length / 2))
    local w = {}
    for i = 1, length do
        if i <= taper then
            w[i] = 0.5 + 0.5 * ((i - 1) / math.max(1, taper - 1))
        elseif i > length - taper then
            local j = length - i
            w[i] = 0.5 + 0.5 * (j / math.max(1, taper - 1))
        else
            w[i] = 1.0
        end
    end
    return w
end

-- =============================================================================
-- TILE SCHEDULE BUILDERS
-- =============================================================================

---Build a fixed (non-adaptive) tile schedule covering [1, W] in latent frames.
---Returns list of {start, end_, overlap} tables (1-indexed start/end).
---@param W integer  total latent frames
---@param tile_size integer
---@param overlap integer
---@param start_offset integer  0-indexed start position (default 0)
---@return table[]
function M.build_fixed_schedule(W, tile_size, overlap, start_offset)
    start_offset = start_offset or 0
    local schedule = {}
    local hop = tile_size - overlap
    if hop <= 0 then hop = math.max(1, math.floor(tile_size / 2)) end
    local cursor = start_offset
    while cursor < W do
        local e = math.min(W, cursor + tile_size)
        table.insert(schedule, {start = cursor, end_ = e, overlap = overlap})
        cursor = cursor + hop
    end
    return schedule
end

---BPM-synced overlap in latent frames.
---Downgrades bar count until overlap fits within tile_size / 2.
---@param bpm integer
---@param target_bars number  e.g. 4.0 for "Max 4 Bars"
---@param tile_size integer
---@param latents_per_second number  ACE-Step = 5.0
---@return integer  overlap in latent frames
function M.bpm_sync_overlap(bpm, target_bars, tile_size, latents_per_second)
    latents_per_second = latents_per_second or 5.0
    local sec_per_bar    = (60.0 / bpm) * 4.0
    local frames_per_bar = sec_per_bar * latents_per_second
    local bars           = target_bars
    local calc = math.floor(sec_per_bar * bars * latents_per_second + 0.5)

    while calc > math.floor(tile_size / 2) and bars > 0.25 do
        bars = bars / 2.0
        calc = math.floor(sec_per_bar * bars * latents_per_second + 0.5)
    end

    -- Integer-multiple snap
    if frames_per_bar >= 1.0 then
        local n = math.floor(calc / frames_per_bar + 0.5)
        calc = n * math.floor(frames_per_bar)
        if calc < 8 then calc = 8 end
    end
    return calc
end

-- =============================================================================
-- LSS: LATENT SPECTRAL SUPPRESSOR
-- =============================================================================

---Suppress low-variance latent channels in-place.
---Modifies `latents` (flat [B, C_lat, T] table) in place.
---@param latents table  flat [B, C, T] row-major
---@param B integer  batch size
---@param C integer  latent channels
---@param T integer  latent time frames
---@param strength number  suppression strength (0–1, gold standard 0.25)
---@param var_threshold number  normalized variance threshold (gold standard 0.12)
---@param dc_remove boolean  WARNING: causes metallic distortion — keep false
function M.apply_lss(latents, B, C, T, strength, var_threshold, dc_remove)
    strength      = strength      or 0.25
    var_threshold = var_threshold or 0.12
    dc_remove     = dc_remove     or false
    if strength < 1e-4 then return end

    -- Per-channel variance averaged over batch
    local ch_var = zeros(C)
    for b = 1, B do
        for c = 1, C do
            local s, sq = 0.0, 0.0
            for t = 1, T do
                local v = latents[idx(b, c, t, C, T)]
                s  = s  + v
                sq = sq + v * v
            end
            local mean = s / T
            ch_var[c] = ch_var[c] + (sq / T - mean * mean)
        end
    end
    for c = 1, C do ch_var[c] = ch_var[c] / B end

    local var_min = math.huge
    local var_max = -math.huge
    for c = 1, C do
        if ch_var[c] < var_min then var_min = ch_var[c] end
        if ch_var[c] > var_max then var_max = ch_var[c] end
    end
    local var_rng = (var_max - var_min) + M.EPSILON

    -- Per-channel suppression gain
    local gain = zeros(C)
    for c = 1, C do
        local var_norm = (ch_var[c] - var_min) / var_rng
        if var_norm < var_threshold then
            local smooth = (1 - strength) +
                           strength * (var_norm / (var_threshold + M.EPSILON))
            smooth = clamp(smooth, 1 - strength, 1.0)
            gain[c] = smooth
        else
            gain[c] = 1.0
        end
    end

    -- Apply (with optional DC removal)
    for b = 1, B do
        for c = 1, C do
            if dc_remove then
                local s = 0.0
                for t = 1, T do s = s + latents[idx(b, c, t, C, T)] end
                local mean = s / T
                for t = 1, T do
                    latents[idx(b, c, t, C, T)] = latents[idx(b, c, t, C, T)] - mean
                end
            end
            local g = gain[c]
            for t = 1, T do
                latents[idx(b, c, t, C, T)] = latents[idx(b, c, t, C, T)] * g
            end
        end
    end
end

-- =============================================================================
-- BIQUAD FILTER ENGINE
-- =============================================================================

---Compute peaking EQ biquad coefficients (normalized, a[1]=1).
---@return table b  {b0,b1,b2}
---@return table a  {1, a1, a2}
function M.peaking_biquad_coeffs(f0, gain_db, Q, sr)
    local A      = 10 ^ (gain_db / 40)
    local w0     = 2 * math.pi * f0 / sr
    local cos_w0 = math.cos(w0)
    local sin_w0 = math.sin(w0)
    local alpha  = sin_w0 / (2 * Q)
    local denom  = 1 + alpha / A
    return
        {(1 + alpha * A) / denom, (-2 * cos_w0) / denom, (1 - alpha * A) / denom},
        {1.0,                     (-2 * cos_w0) / denom, (1 - alpha / A) / denom}
end

---Compute low shelf biquad coefficients.
---@return table b, table a
function M.low_shelf_biquad_coeffs(f0, gain_db, slope, sr)
    local A      = 10 ^ (gain_db / 40)
    local w0     = 2 * math.pi * f0 / sr
    local cos_w0 = math.cos(w0)
    local sin_w0 = math.sin(w0)
    local alpha  = sin_w0 / 2 * math.sqrt((A + 1/A) * (1/slope - 1) + 2)
    local a0     = (A+1) + (A-1)*cos_w0 + 2*math.sqrt(A)*alpha
    local b = {
        A * ((A+1) - (A-1)*cos_w0 + 2*math.sqrt(A)*alpha) / a0,
        2*A*((A-1) - (A+1)*cos_w0)                         / a0,
        A * ((A+1) - (A-1)*cos_w0 - 2*math.sqrt(A)*alpha) / a0,
    }
    local a = {
        1.0,
        -2 * ((A-1) + (A+1)*cos_w0)                        / a0,
        ((A+1) + (A-1)*cos_w0 - 2*math.sqrt(A)*alpha)      / a0,
    }
    return b, a
end

---Apply biquad IIR filter in-place to a single-channel flat buffer [1..L].
---Returns updated biquad state {x1,x2,y1,y2}.
---@param buf table  [1..L] float samples
---@param L integer
---@param b table  {b0,b1,b2}
---@param a table  {1,a1,a2}
---@param state table  {x1,x2,y1,y2} (pass {} to initialise fresh)
---@return table state
function M.biquad_filter_inplace(buf, L, b, a, state)
    local x1 = state.x1 or 0
    local x2 = state.x2 or 0
    local y1 = state.y1 or 0
    local y2 = state.y2 or 0
    for i = 1, L do
        local xn = buf[i]
        local yn = b[1]*xn + b[2]*x1 + b[3]*x2 - a[2]*y1 - a[3]*y2
        x2, x1 = x1, xn
        y2, y1 = y1, yn
        buf[i] = yn
    end
    return {x1=x1, x2=x2, y1=y1, y2=y2}
end

-- =============================================================================
-- HUM NOTCH CHAIN
-- =============================================================================

---Multi-band hum suppression. Modifies `audio` in place.
---@param audio table  flat [B, C, L] row-major
---@param B integer
---@param C integer
---@param L integer
---@param params table  TiledDecodeParams-compatible
function M.apply_hum_notch(audio, B, C, L, params)
    if not params.hum_notch_enabled then return end
    local sr = params.sample_rate or 48000

    -- Build filter chain
    local chain = {}  -- {b, a} pairs
    if params.hum_bass_shelf_enabled then
        local b, a = M.low_shelf_biquad_coeffs(
            params.hum_bass_shelf_hz or 120.0,
            params.hum_bass_shelf_db  or -2.0,
            params.hum_bass_shelf_slope or 0.7,
            sr)
        table.insert(chain, {b=b, a=a})
    end
    local notches = {
        {en="hum_74_enabled",  hz="hum_74_hz",  db="hum_74_db",  q="hum_74_q"},
        {en="hum_94_enabled",  hz="hum_94_hz",  db="hum_94_db",  q="hum_94_q"},
        {en="hum_656_enabled", hz="hum_656_hz", db="hum_656_db", q="hum_656_q"},
    }
    for _, n in ipairs(notches) do
        if params[n.en] then
            local b, a = M.peaking_biquad_coeffs(
                params[n.hz], params[n.db], params[n.q], sr)
            table.insert(chain, {b=b, a=a})
        end
    end

    if #chain == 0 then return end

    -- Apply each band to each channel independently
    for b = 1, B do
        for c = 1, C do
            -- Extract channel slice into temp buffer
            local ch = {}
            local base = idx(b, c, 1, C, L)
            for i = 1, L do ch[i] = audio[base + i - 1] end
            -- Filter chain
            for _, filt in ipairs(chain) do
                M.biquad_filter_inplace(ch, L, filt.b, filt.a, {})
            end
            -- Write back
            for i = 1, L do audio[base + i - 1] = ch[i] end
        end
    end
end

-- =============================================================================
-- HIGH-PASS FILTER
-- =============================================================================

---Butterworth 2nd-order high-pass filter (biquad). In-place.
---@param audio table  flat [B, C, L]
---@param B integer
---@param C integer
---@param L integer
---@param cutoff_hz number
---@param sample_rate number
function M.apply_highpass(audio, B, C, L, cutoff_hz, sample_rate)
    if cutoff_hz < 1.0 then return end
    local w0     = 2 * math.pi * cutoff_hz / sample_rate
    local cos_w0 = math.cos(w0)
    local sin_w0 = math.sin(w0)
    local alpha  = sin_w0 / (2 * 0.7071)  -- Q = 1/sqrt(2) Butterworth
    local denom  = 1 + alpha
    local b = {
        (1 + cos_w0) / (2 * denom),
        -(1 + cos_w0) / denom,
        (1 + cos_w0) / (2 * denom),
    }
    local a = {
        1.0,
        (-2 * cos_w0) / denom,
        (1 - alpha)   / denom,
    }

    for bi = 1, B do
        for c = 1, C do
            local base = idx(bi, c, 1, C, L)
            local ch = {}
            for i = 1, L do ch[i] = audio[base + i - 1] end
            M.biquad_filter_inplace(ch, L, b, a, {})
            for i = 1, L do audio[base + i - 1] = ch[i] end
        end
    end
end

-- =============================================================================
-- SOFT CLIPPER
-- =============================================================================

---tanh-based soft clipper. In-place.
---@param audio table  flat [B, C, L]
---@param ceiling_db number  e.g. -3.0 (must be < 0 to have effect)
function M.apply_soft_clip(audio, ceiling_db)
    if ceiling_db >= 0 then return end
    local ceiling_lin = 10 ^ (ceiling_db / 20)
    local scale       = ceiling_lin / tanh(1.0)
    for i = 1, #audio do
        audio[i] = scale * tanh(audio[i] / scale)
    end
end

-- =============================================================================
-- PEAK NORMALIZE
-- =============================================================================

---Transparent peak normalization: scale entire audio so max |sample| = target.
---Pure gain reduction — no waveform distortion, no waveshaping.
---Only attenuates (never boosts). Skipped if peak is already below target.
---@param audio table  flat buffer
---@param N integer  total samples
---@param target_db number  target peak in dBFS (e.g., -1.0)
function M.apply_peak_normalize(audio, N, target_db)
    if not target_db or target_db >= 0 then return end
    local target_lin = 10 ^ (target_db / 20)
    local peak = 0.0
    for i = 1, N do
        local v = math.abs(audio[i])
        if v > peak then peak = v end
    end
    if peak < 1e-8 then return end   -- silence
    if peak <= target_lin then return end  -- already below target
    local gain = target_lin / peak
    for i = 1, N do
        audio[i] = audio[i] * gain
    end
end

-- =============================================================================
-- STEREO WIDTH (M/S)
-- =============================================================================

---M/S stereo width. Only operates when C == 2. In-place.
---@param audio table  flat [B, 2, L]
---@param B integer
---@param L integer
---@param width number  1.0=unity, 0.0=mono, 2.0=doubled side
function M.apply_stereo_width(audio, B, L, width)
    if math.abs(width - 1.0) < 1e-4 then return end
    for b = 1, B do
        local base_l = idx(b, 1, 1, 2, L)
        local base_r = idx(b, 2, 1, 2, L)
        for i = 0, L - 1 do
            local l = audio[base_l + i]
            local r = audio[base_r + i]
            local mid  = (l + r) * 0.5
            local side = (l - r) * 0.5 * width
            audio[base_l + i] = mid + side
            audio[base_r + i] = mid - side
        end
    end
end

-- =============================================================================
-- RMS UTILITIES
-- =============================================================================

---Compute RMS of flat buffer.
function M.compute_rms(buf, N)
    local s = 0.0
    for i = 1, N do local v = buf[i]; s = s + v*v end
    return math.sqrt(s / N + M.EPSILON)
end

---Downward-only absolute RMS ceiling. In-place.
---@param audio table  flat buffer
---@param ceiling number  default M.RMS_ABS_CEIL
function M.apply_rms_ceiling(audio, N, ceiling)
    ceiling = ceiling or M.RMS_ABS_CEIL
    local rms = M.compute_rms(audio, N)
    if rms > ceiling then
        local g = ceiling / rms
        for i = 1, N do audio[i] = audio[i] * g end
    end
end

-- =============================================================================
-- OLA WRITE PRIMITIVE
-- =============================================================================

---Write decoded_chunk into output_audio at out_start with OLA crossfade.
---All buffers flat [B, C, length] row-major (1-indexed).
---@param output_audio table  flat [B, C, final_samples]
---@param B integer
---@param C integer
---@param final_samples integer
---@param decoded_chunk table  flat [B, C, decoded_len]
---@param decoded_len integer
---@param out_start integer  0-indexed write position in audio samples
---@param overlap_audio integer  audio-domain overlap samples
---@param blend_mode string  "Hann"|"Cosine"|"Linear"
function M.ola_write(output_audio, B, C, final_samples,
                      decoded_chunk, decoded_len,
                      out_start, overlap_audio, blend_mode)
    local valid_len  = math.min(decoded_len, final_samples - out_start)
    if valid_len <= 0 then return end
    local ov = math.min(overlap_audio, math.floor(valid_len / 2))

    if out_start > 0 and ov > 0 then
        local fade_in  = M.make_fade_in(ov, blend_mode)
        -- Cosine equal-power fade-out: sqrt(1 - f^2)
        -- Linear/Hann: simple 1-f
        local use_eqp = (blend_mode == "Cosine")

        for b = 1, B do
            for c = 1, C do
                local out_base = idx(b, c, 1, C, final_samples)
                local in_base  = idx(b, c, 1, C, decoded_len)

                -- Crossfade
                for i = 1, ov do
                    local fi = fade_in[i]
                    local fo = use_eqp
                               and math.sqrt(math.max(0, 1 - fi * fi))
                               or  (1 - fi)
                    local out_i = out_start + i  -- 1-indexed position in output
                    if out_i >= 1 and out_i <= final_samples then
                        output_audio[out_base + out_i - 1] =
                            output_audio[out_base + out_i - 1] * fo +
                            decoded_chunk[in_base + i - 1]    * fi
                    end
                end

                -- Tail (straight copy after crossfade)
                for i = ov + 1, valid_len do
                    local out_i = out_start + i
                    if out_i >= 1 and out_i <= final_samples then
                        output_audio[out_base + out_i - 1] =
                            decoded_chunk[in_base + i - 1]
                    end
                end
            end
        end

    else
        -- First tile — straight write
        for b = 1, B do
            for c = 1, C do
                local out_base = idx(b, c, 1, C, final_samples)
                local in_base  = idx(b, c, 1, C, decoded_len)
                for i = 1, valid_len do
                    output_audio[out_base + out_start + i - 1] =
                        decoded_chunk[in_base + i - 1]
                end
            end
        end
    end
end

-- =============================================================================
-- DUAL-PASS WEIGHT ACCUMULATION
-- =============================================================================

---Build trapezoidal weight map for one pass's tile boundaries.
---@param weight_buf table  [1..final_samples] float, modified in place
---@param final_samples integer
---@param schedule table[]  list of {start, end_, overlap}
---@param boundaries table  list of out_start integers (0-indexed audio positions)
---@param upscale_factor number
function M.fill_trapezoid_weights(weight_buf, final_samples,
                                    schedule, boundaries, upscale_factor)
    for i, bound in ipairs(boundaries) do
        local tile = schedule[i]
        if not tile then break end
        local lat_len  = tile.end_ - tile.start
        local tile_len = math.floor(lat_len * upscale_factor + 0.5)
        local out_s    = bound + 1  -- convert to 1-indexed
        local out_e    = math.min(final_samples, out_s + tile_len - 1)
        local L        = out_e - out_s + 1
        if L <= 0 then goto continue end

        local trap = M.make_trapezoid(L, M.DUAL_PASS_TAPER)
        for j = 1, L do
            local pos = out_s + j - 1
            if pos >= 1 and pos <= final_samples then
                if trap[j] > weight_buf[pos] then weight_buf[pos] = trap[j] end
            end
        end
        ::continue::
    end
end

-- =============================================================================
-- UPSCALE FACTOR SNAP
-- =============================================================================

---Snap upscale factor to nearest integer if within 0.5%.
---ACE-Step Oobleck always produces an exact integer ratio.
---Sub-sample error compounds across tiles → audible timing drift.
---@param raw number
---@return number
function M.snap_upscale_factor(raw)
    local snapped = math.floor(raw + 0.5)
    if math.abs(snapped - raw) / (raw + M.EPSILON) < 0.005 then
        return snapped
    end
    return raw
end

-- =============================================================================
-- MASTER TILED DECODE ORCHESTRATOR
-- =============================================================================

---Master tiled VAE decode engine.
---
---@param vae_decode_fn function(latent_slice, lat_len) -> audio_chunk, actual_len
---       latent_slice: flat [B, C_lat, lat_len] table
---       Returns: flat [B, C_aud, actual_len] table, integer actual_len
---
---@param latents table  flat [B, C_lat, W] latent buffer (may be modified by LSS)
---@param B integer  batch size
---@param C_lat integer  latent channels
---@param W integer  latent frame count
---@param C_aud integer  audio channels (typically 2)
---@param final_samples integer  total output audio samples
---@param upscale_factor number  audio samples per latent frame
---@param params table  TiledDecodeParams-compatible:
---   tile_size, overlap, context_prefix, dual_pass, rms_leveling,
---   lss_enabled, lss_strength, lss_var_thresh, lss_dc_remove,
---   hum_notch_enabled (+ per-band params), highpass_hz, soft_clip_db,
---   stereo_width, sample_rate
---
---@return table  flat [B, C_aud, final_samples] decoded audio
function M.execute_tiled_decode(vae_decode_fn, latents, B, C_lat, W,
                                  C_aud, final_samples, upscale_factor, params)
    params = params or {}
    local tile_size      = params.tile_size      or 1024
    local overlap        = params.overlap        or 64
    local context_prefix = params.context_prefix or 512
    local dual_pass      = params.dual_pass      ~= false  -- default true
    local rms_leveling   = params.rms_leveling   ~= false  -- default true
    local sample_rate    = params.sample_rate    or 48000

    -- ── LSS ──────────────────────────────────────────────────────────────────
    if params.lss_enabled ~= false then
        M.apply_lss(latents, B, C_lat, W,
                    params.lss_strength   or 0.25,
                    params.lss_var_thresh or 0.12,
                    params.lss_dc_remove  or false)
    end

    -- ── Tile schedules ────────────────────────────────────────────────────────
    local sched_a = M.build_fixed_schedule(W, tile_size, overlap, 0)
    local sched_b = dual_pass
                    and M.build_fixed_schedule(W, tile_size, overlap,
                                               math.floor(tile_size / 2))
                    or nil

    -- ── Run one pass ──────────────────────────────────────────────────────────
    local function run_pass(schedule)
        local audio_out = zeros(B * C_aud * final_samples)
        local boundaries = {}
        local prev_rms = -1.0

        for _, tile in ipairs(schedule) do
            local ctx_start = math.max(0, tile.start - context_prefix)
            local lat_len   = tile.end_ - ctx_start

            -- Extract latent slice
            local lat_slice = zeros(B * C_lat * lat_len)
            for b = 1, B do
                for c = 1, C_lat do
                    local src_base = (b-1)*C_lat*W + (c-1)*W + ctx_start + 1
                    local dst_base = (b-1)*C_lat*lat_len + (c-1)*lat_len + 1
                    for t = 1, lat_len do
                        lat_slice[dst_base + t - 1] = latents[src_base + t - 1]
                    end
                end
            end

            -- VAE decode
            local chunk, actual_len = vae_decode_fn(lat_slice, lat_len)
            if not chunk or actual_len <= 0 then goto next_tile end

            -- ctx_skip: absorb VAE rounding into discarded context region
            local expected_write = math.floor((tile.end_ - tile.start) * upscale_factor + 0.5)
            local ctx_skip       = math.max(0, actual_len - expected_write)
            local write_len      = actual_len - ctx_skip
            if write_len <= 0 then goto next_tile end

            -- Build write chunk (post ctx_skip)
            local write_chunk = zeros(B * C_aud * write_len)
            for b = 1, B do
                for c = 1, C_aud do
                    local src_base = (b-1)*C_aud*actual_len + (c-1)*actual_len + ctx_skip + 1
                    local dst_base = (b-1)*C_aud*write_len  + (c-1)*write_len  + 1
                    for i = 1, write_len do
                        write_chunk[dst_base + i - 1] = chunk[src_base + i - 1]
                    end
                end
            end

            -- RMS leveling
            if rms_leveling and prev_rms > 0 then
                local N = B * C_aud * write_len
                local crms = M.compute_rms(write_chunk, N)
                local gain = clamp(prev_rms / (crms + M.EPSILON),
                                   1 - M.GAIN_CLAMP_BASE, 1 + M.GAIN_CLAMP_BASE)
                for i = 1, N do write_chunk[i] = write_chunk[i] * gain end
            end

            -- Absolute RMS ceiling
            M.apply_rms_ceiling(write_chunk, B * C_aud * write_len)

            prev_rms = M.compute_rms(write_chunk, B * C_aud * write_len)

            local out_start   = math.floor(tile.start * upscale_factor + 0.5)
            local overlap_aud = math.floor(tile.overlap * upscale_factor + 0.5)
            table.insert(boundaries, out_start)

            M.ola_write(audio_out, B, C_aud, final_samples,
                        write_chunk, write_len,
                        out_start, overlap_aud, "Cosine")

            ::next_tile::
        end

        return audio_out, boundaries
    end

    local audio_a, bounds_a = run_pass(sched_a)
    local audio_b, bounds_b = nil, {}
    if sched_b then
        audio_b, bounds_b = run_pass(sched_b)
    end

    -- ── Dual-Pass Merge ───────────────────────────────────────────────────────
    local output_audio
    if dual_pass and audio_b then
        local weight_a = zeros(final_samples)
        local weight_b = zeros(final_samples)
        M.fill_trapezoid_weights(weight_a, final_samples, sched_a, bounds_a, upscale_factor)
        M.fill_trapezoid_weights(weight_b, final_samples, sched_b, bounds_b, upscale_factor)

        output_audio = zeros(B * C_aud * final_samples)
        for b = 1, B do
            for c = 1, C_aud do
                local base = (b-1)*C_aud*final_samples + (c-1)*final_samples
                for i = 1, final_samples do
                    local wa = weight_a[i]
                    local wb = weight_b[i]
                    local total = wa + wb + M.EPSILON
                    output_audio[base + i] = audio_a[base + i] * (wa / total)
                                           + audio_b[base + i] * (wb / total)
                end
            end
        end
    else
        output_audio = audio_a
    end

    -- ── Post-Decode DSP Chain ─────────────────────────────────────────────────
    -- Order: hum notch → highpass → stereo width → soft clip
    M.apply_hum_notch(output_audio, B, C_aud, final_samples, params)
    M.apply_highpass(output_audio, B, C_aud, final_samples,
                     params.highpass_hz or 20.0, sample_rate)
    if C_aud == 2 then
        M.apply_stereo_width(output_audio, B, final_samples,
                              params.stereo_width or 0.8)
    end
    M.apply_peak_normalize(output_audio, B * C_aud * final_samples,
                            params.peak_normalize_db)  -- nil = skip
    M.apply_soft_clip(output_audio, params.soft_clip_db or -3.0)

    return output_audio
end

-- =============================================================================
-- DEFAULT PARAMS
-- =============================================================================

---Gold-standard default parameters. Copy and override as needed.
M.DEFAULT_PARAMS = {
    -- Tiling
    tile_size      = 1024,
    overlap        = 64,
    context_prefix = 512,
    dual_pass      = true,
    rms_leveling   = true,
    sample_rate    = 48000,

    -- LSS
    lss_enabled    = true,
    lss_strength   = 0.25,
    lss_var_thresh = 0.12,
    lss_dc_remove  = false,  -- WARNING: metallic distortion if true

    -- DSP chain
    highpass_hz    = 20.0,
    peak_normalize_db = nil,    -- nil = disabled; e.g. -1.0 for -1dBFS peak target
    soft_clip_db   = -3.0,
    stereo_width   = 0.8,

    -- Hum notch (gold standard: cuts only)
    hum_notch_enabled        = true,
    hum_bass_shelf_enabled   = true,
    hum_bass_shelf_hz        = 120.0,
    hum_bass_shelf_db        = -2.0,
    hum_bass_shelf_slope     = 0.7,
    hum_74_enabled           = true,
    hum_74_hz                = 74.4,
    hum_74_db                = -1.43,
    hum_74_q                 = 6.27,
    hum_94_enabled           = true,
    hum_94_hz                = 94.0,
    hum_94_db                = -1.86,
    hum_94_q                 = 7.08,
    hum_656_enabled          = true,
    hum_656_hz               = 654.0,
    hum_656_db               = -15.0,
    hum_656_q                = 6.0,
}

return M
