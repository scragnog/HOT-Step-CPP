--[[
md_audio_tiled.lua
Postprocess plugin adapter for MD Audio Tiled Core

Wraps md_audio_tiled_core.lua (v3.0.1) to conform to the HOT-Step
postprocess plugin contract. The core module is loaded via require()
and exposes execute_tiled_decode() as the entry point.

© 2026 Alexander Allan (MDMAchine) | A&E Concepts
GPL v3
--]]

local core = require("md_audio_tiled_core")

postprocess = {
    name        = "md_audio_tiled",
    display     = "MD Audio Tiled Decoder",
    description = "Tiled VAE decode with OLA crossfade, dual-pass merge, LSS, and DSP chain",
    accent      = "cyan",

    params = {
        { key = "dual_pass", type = "toggle", label = "Dual Pass",
          default = false,
          hint = "Two staggered decode passes merged with trapezoidal weights. Eliminates seam artifacts but doubles VAE decode time." },
        { key = "lss_strength", type = "slider", label = "LSS Strength",
          default = 0.25, min = 0, max = 1, step = 0.01,
          hint = "Latent channel suppression. Reduces hum from low-variance VAE bias channels." },
        { key = "stereo_width", type = "slider", label = "Stereo Width",
          default = 0.8, min = 0, max = 2, step = 0.01,
          hint = "M/S stereo width. 0=mono, 1=unity, 2=doubled side." },
        { key = "hum_notch", type = "toggle", label = "Hum Notch Filter",
          default = true,
          hint = "Multi-band surgical cuts at 74/94/654Hz to remove Oobleck VAE hum." },
        { key = "peak_normalize_db", type = "slider", label = "Peak Normalize",
          default = -1, min = -12, max = 0, step = 0.5,
          hint = "Transparent peak normalization (pure gain reduction, no distortion). Scales audio so the loudest peak = target dBFS. Applied before soft clip. Set to 0 to disable." },
        { key = "soft_clip_db", type = "slider", label = "Soft Clip Ceiling",
          default = -3.0, min = -12, max = 0, step = 0.5,
          hint = "tanh saturation ceiling in dB. Acts as safety net after peak normalize. Set to 0 to disable." },
    },
}

-- Entry point called by the engine via lua_call_postprocess()
-- Args:
--   latents:       Lua table, 1-indexed [B * C_lat * W] flat row-major
--   B:             batch size (always 1 — engine calls per-batch-item)
--   C_lat:         latent channels (64)
--   W:             latent width (time frames)
--   C_aud:         audio channels (2)
--   final_samples: expected audio samples per channel
--   upscale_factor: 1920 (VAE upsample ratio)
--   vae_decode_fn: callback(latent_table, T_latent) → audio_table, T_audio
function process(latents, B, C_lat, W, C_aud, final_samples, upscale_factor, vae_decode_fn)
    -- Build params table from UI-injected globals
    local p = {}
    for k, v in pairs(core.DEFAULT_PARAMS) do
        p[k] = v
    end

    -- Override from UI params (injected by lua_inject_params)
    if params then
        if params.dual_pass ~= nil then p.dual_pass = params.dual_pass end
        if params.lss_strength then p.lss_strength = params.lss_strength end
        if params.stereo_width then p.stereo_width = params.stereo_width end
        if params.hum_notch ~= nil then p.hum_notch_enabled = params.hum_notch end
        if params.peak_normalize_db then
            -- 0 dB means disabled (peak_normalize_db must be < 0 to activate)
            if params.peak_normalize_db < 0 then
                p.peak_normalize_db = params.peak_normalize_db
            else
                p.peak_normalize_db = nil  -- disable
            end
        end
        if params.soft_clip_db then p.soft_clip_db = params.soft_clip_db end
    end

    local audio = core.execute_tiled_decode(
        vae_decode_fn, latents, B, C_lat, W,
        C_aud, final_samples, upscale_factor, p)

    -- Bridge expects two return values: audio_table, T_audio
    return audio, final_samples
end
