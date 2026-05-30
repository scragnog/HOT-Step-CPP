#pragma once
// vae-ort.h: ONNX Runtime VAE decoder wrapper (TensorRT / CUDA EP)
//
// Encapsulates ORT session lifecycle and inference for the VAE decoder
// exported as an ONNX model. Supports TensorRT EP (preferred, builds an
// optimised engine on first run) with CUDA EP fallback.
//
// The VAE decoder ONNX model takes latent input [batch, 64, T_latent]
// and produces audio output [batch, 1, T_audio] (mono 48kHz).
// T_audio = T_latent * 1920 (upsampling factor of the VAE).
//
// Thread safety: none. Caller serialises access (single GPU worker thread).
//
// Guarded by HOT_STEP_SUPERSEP because ORT headers and libraries are only
// available when the SuperSep feature is compiled in (shared ORT dependency).
//
// Part of HOT-Step CPP. MIT license.

#ifndef VAE_ORT_H
#define VAE_ORT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HOT_STEP_SUPERSEP

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>

// ── VAE ORT context ────────────────────────────────────────────────────

struct VaeOrt {
    Ort::Env            env;
    Ort::Session *      session;
    Ort::SessionOptions session_opts;
    std::string         model_path;
    bool                using_trt;   // true if TRT EP is active

    VaeOrt() : env(ORT_LOGGING_LEVEL_WARNING, "vae-ort"),
               session(nullptr), using_trt(false) {}
    ~VaeOrt() { delete session; }
};

// ── Load / Free ────────────────────────────────────────────────────────

// Initialise a VaeOrt context from an ONNX model file.
// Attempts TensorRT EP first (builds engine cache alongside the .onnx),
// falls back to CUDA EP, then CPU.
// device_id: CUDA device ordinal (0 for single-GPU systems).
// Returns true on success.
static inline bool vae_ort_load(VaeOrt * ctx, const char * onnx_path, int device_id = 0) {
    if (!ctx || !onnx_path) return false;

    ctx->model_path = onnx_path;
    ctx->session_opts.SetIntraOpNumThreads(1);
    ctx->session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ── TensorRT EP (preferred) ────────────────────────────────────
#if defined(GGML_USE_CUDA)
    // Build TRT engine cache directory alongside the model file.
    std::string trt_cache_dir;
    {
        std::string p = onnx_path;
        auto slash = p.find_last_of("/\\");
        trt_cache_dir = (slash != std::string::npos) ? p.substr(0, slash) : ".";
    }

    // Try TensorRT RTX EP first (ORT 1.21+ uses NvTensorRTRTXExecutionProvider)
    try {
        std::unordered_map<std::string, std::string> trt_opts;
        trt_opts["device_id"]                = std::to_string(device_id);
        trt_opts["nv_max_workspace_size"]    = std::to_string((size_t)2 << 30);  // 2 GiB
        trt_opts["nv_runtime_cache_path"]    = trt_cache_dir;
        trt_opts["nv_detailed_build_log"]    = "1";

        // Dynamic shape profiles for VAE: input "latents" is [1, 64, T]
        // T varies per generation length. Profile: min=64, opt=2048, max=8192 latent frames.
        trt_opts["nv_profile_min_shapes"]    = "latents:1x64x64";
        trt_opts["nv_profile_opt_shapes"]    = "latents:1x64x2048";
        trt_opts["nv_profile_max_shapes"]    = "latents:1x64x8192";

        ctx->session_opts.AppendExecutionProvider("NvTensorRTRTXExecutionProvider", trt_opts);
        ctx->using_trt = true;
        fprintf(stderr, "[VAE-ORT] TensorRT RTX EP appended (device %d, cache=%s)\n",
                device_id, trt_cache_dir.c_str());
    } catch (const std::exception & e) {
        fprintf(stderr, "[VAE-ORT] TensorRT RTX EP unavailable: %s — trying CUDA EP\n", e.what());
        ctx->using_trt = false;
    }

    // CUDA EP fallback (or as backup after TRT — ORT falls through automatically)
    try {
        OrtCUDAProviderOptions cuda_opts;
        memset(&cuda_opts, 0, sizeof(cuda_opts));
        cuda_opts.device_id             = device_id;
        cuda_opts.arena_extend_strategy = 1;  // kSameAsRequested
        ctx->session_opts.AppendExecutionProvider_CUDA(cuda_opts);
        fprintf(stderr, "[VAE-ORT] CUDA EP appended (device %d, arena=exact)\n", device_id);
    } catch (const std::exception & e) {
        fprintf(stderr, "[VAE-ORT] CUDA EP failed: %s — falling back to CPU\n", e.what());
    }
#else
    fprintf(stderr, "[VAE-ORT] No GPU EP available — using CPU\n");
#endif

    // ── Create session ─────────────────────────────────────────────
    try {
#ifdef _WIN32
        int wlen = MultiByteToWideChar(CP_UTF8, 0, onnx_path, -1, nullptr, 0);
        std::vector<wchar_t> wpath(wlen);
        MultiByteToWideChar(CP_UTF8, 0, onnx_path, -1, wpath.data(), wlen);
        ctx->session = new Ort::Session(ctx->env, wpath.data(), ctx->session_opts);
#else
        ctx->session = new Ort::Session(ctx->env, onnx_path, ctx->session_opts);
#endif
        fprintf(stderr, "[VAE-ORT] Session created: %s (TRT=%s)\n",
                onnx_path, ctx->using_trt ? "yes" : "no");
    } catch (const std::exception & e) {
        fprintf(stderr, "[VAE-ORT] FATAL: session creation failed: %s\n", e.what());
        ctx->session = nullptr;
        return false;
    }

    return true;
}

static inline void vae_ort_free(VaeOrt * ctx) {
    if (!ctx) return;
    delete ctx->session;
    ctx->session = nullptr;
}

// ── Decode ─────────────────────────────────────────────────────────────
//
// latent: [T_latent, 64] f32 time-major (from DiT output), same layout as
//         the GGML VAE decoder expects.
// audio_out: caller-allocated buffer, at least T_latent * 1920 * 2 floats
//            (planar stereo [L0..LN, R0..RN]).
// T_audio_max: max audio frames the buffer can hold.
//
// Returns the number of audio frames decoded (per channel), or -1 on error.

static inline int vae_ort_decode(VaeOrt *    ctx,
                                  const float * latent,
                                  int           T_latent,
                                  float *       audio_out,
                                  int           T_audio_max) {
    if (!ctx || !ctx->session || !latent || T_latent <= 0) return -1;

    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input: [1, 64, T_latent] — ONNX model expects channel-first.
    // Our latent is [T_latent, 64] time-major, so transpose.
    std::vector<float> input_buf((size_t)64 * T_latent);
    for (int t = 0; t < T_latent; t++) {
        for (int c = 0; c < 64; c++) {
            input_buf[(size_t)c * T_latent + t] = latent[(size_t)t * 64 + c];
        }
    }

    std::vector<int64_t> input_shape = {1, 64, (int64_t)T_latent};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem, input_buf.data(), input_buf.size(),
        input_shape.data(), input_shape.size());

    // Get I/O names from the model
    auto in_name  = ctx->session->GetInputNameAllocated(0, alloc);
    auto out_name = ctx->session->GetOutputNameAllocated(0, alloc);
    const char * in_names[]  = { in_name.get() };
    const char * out_names[] = { out_name.get() };

    // Run inference
    std::vector<Ort::Value> outputs;
    try {
        outputs = ctx->session->Run(
            Ort::RunOptions{nullptr},
            in_names, &input_tensor, 1,
            out_names, 1);
    } catch (const std::exception & e) {
        fprintf(stderr, "[VAE-ORT] Inference failed: %s\n", e.what());
        return -1;
    }

    // Parse output shape
    auto & out_tensor = outputs[0];
    auto   type_info  = out_tensor.GetTensorTypeAndShapeInfo();
    auto   out_shape  = type_info.GetShape();
    size_t out_count  = type_info.GetElementCount();
    const float * out_data = out_tensor.GetTensorData<float>();

    // Expected shapes:
    //   [1, 1, T_audio]  — mono model (duplicate to stereo)
    //   [1, 2, T_audio]  — stereo model
    int n_channels = 1;
    int T_audio    = 0;

    if (out_shape.size() == 3) {
        n_channels = (int)out_shape[1];
        T_audio    = (int)out_shape[2];
    } else if (out_shape.size() == 2) {
        T_audio = (int)out_shape[1];
    } else {
        // Fallback: treat as flat
        T_audio = (int)out_count;
    }

    if (T_audio > T_audio_max) {
        fprintf(stderr, "[VAE-ORT] WARNING: output T_audio=%d > buffer max=%d, clamping\n",
                T_audio, T_audio_max);
        T_audio = T_audio_max;
    }

    // Copy to planar stereo output [L0..LN, R0..RN]
    if (n_channels >= 2) {
        // Stereo: channel-first → planar
        memcpy(audio_out,             out_data,                      (size_t)T_audio * sizeof(float));
        memcpy(audio_out + T_audio,   out_data + (size_t)T_audio,    (size_t)T_audio * sizeof(float));
    } else {
        // Mono: duplicate to both channels
        memcpy(audio_out,           out_data, (size_t)T_audio * sizeof(float));
        memcpy(audio_out + T_audio, out_data, (size_t)T_audio * sizeof(float));
    }

    fprintf(stderr, "[VAE-ORT] Decoded: T_latent=%d → T_audio=%d (channels=%d, TRT=%s)\n",
            T_latent, T_audio, n_channels, ctx->using_trt ? "yes" : "no");

    return T_audio;
}

// ── Tiled decode ──────────────────────────────────────────────────────────
//
// Overlap-discard chunking for bounded VRAM usage, mirroring
// vae_ggml_decode_tiled(). Each tile decodes a latent window with overlap
// context; the core (non-overlap) portion is kept and concatenated.
//
// stride = chunk_size - 2*overlap
// Default chunk=256 / overlap=64 matches the GGML defaults.
// Returns T_audio (total audio frames per channel) or -1 on error.

static inline int vae_ort_decode_tiled(VaeOrt *      ctx,
                                        const float * latent,      // [T_latent, 64] time-major
                                        int           T_latent,
                                        float *       audio_out,   // [2, T_audio_max] planar stereo
                                        int           T_audio_max,
                                        int           chunk_size = 256,
                                        int           overlap    = 64) {
    if (!ctx || !ctx->session || !latent || T_latent <= 0) return -1;

    // Ensure positive stride
    while (chunk_size - 2 * overlap <= 0 && overlap > 0) {
        overlap /= 2;
    }

    // Short sequence: decode directly (no tiling needed)
    if (T_latent <= chunk_size) {
        return vae_ort_decode(ctx, latent, T_latent, audio_out, T_audio_max);
    }

    int stride    = chunk_size - 2 * overlap;
    int num_steps = (T_latent + stride - 1) / stride;

    fprintf(stderr, "[VAE-ORT] Tiled decode: %d tiles (chunk=%d, overlap=%d, stride=%d)\n",
            num_steps, chunk_size, overlap, stride);

    float upsample_factor = 0.0f;
    int   audio_write_pos = 0;

    // Temporary buffer for each tile's audio output
    int tile_audio_max = chunk_size * 1920;
    std::vector<float> tile_audio(2 * tile_audio_max);

    for (int i = 0; i < num_steps; i++) {
        // Core range in latent frames (the part we keep)
        int core_start = i * stride;
        int core_end   = core_start + stride;
        if (core_end > T_latent) core_end = T_latent;

        // Window range with overlap context
        int win_start = core_start - overlap;
        if (win_start < 0) win_start = 0;
        int win_end = core_end + overlap;
        if (win_end > T_latent) win_end = T_latent;
        int win_len = win_end - win_start;

        // Decode this tile
        int tile_T = vae_ort_decode(ctx,
                                     latent + win_start * 64,  // offset into latent
                                     win_len,
                                     tile_audio.data(),
                                     tile_audio_max);
        if (tile_T < 0) {
            fprintf(stderr, "[VAE-ORT] FATAL: tile %d decode failed\n", i);
            return -1;
        }

        // Determine upsample factor from first tile
        if (i == 0) {
            upsample_factor = (float)tile_T / (float)win_len;
            fprintf(stderr, "[VAE-ORT] Upsample factor: %.2f (expected ~1920)\n", upsample_factor);
        }

        // Compute trim in audio samples
        int added_start = core_start - win_start;
        int trim_start  = (int)roundf((float)added_start * upsample_factor);
        int added_end   = win_end - core_end;
        int trim_end    = (int)roundf((float)added_end * upsample_factor);

        int end_idx  = (trim_end > 0) ? (tile_T - trim_end) : tile_T;
        int core_len = end_idx - trim_start;
        if (core_len <= 0) continue;

        // Check output bounds
        if (audio_write_pos + core_len > T_audio_max) {
            fprintf(stderr, "[VAE-ORT] FATAL: tiled output exceeds T_audio_max\n");
            return -1;
        }

        // Copy trimmed ch0 and ch1 into final audio_out (planar: [L...][R...])
        // tile_audio layout: [L0..LN, R0..RN] where N = tile_T
        memcpy(audio_out + audio_write_pos,
               tile_audio.data() + trim_start,
               (size_t)core_len * sizeof(float));
        memcpy(audio_out + T_audio_max + audio_write_pos,
               tile_audio.data() + tile_T + trim_start,
               (size_t)core_len * sizeof(float));
        audio_write_pos += core_len;
    }

    // Compact ch1 from offset T_audio_max to offset audio_write_pos
    memmove(audio_out + audio_write_pos,
            audio_out + T_audio_max,
            (size_t)audio_write_pos * sizeof(float));

    fprintf(stderr, "[VAE-ORT] Tiled decode done: %d tiles → T_audio=%d (%.2fs @ 48kHz, TRT=%s)\n",
            num_steps, audio_write_pos, (float)audio_write_pos / 48000.0f,
            ctx->using_trt ? "yes" : "no");

    return audio_write_pos;
}

#else  // !HOT_STEP_SUPERSEP — stubs

struct VaeOrt {};

static inline bool vae_ort_load(VaeOrt *, const char *, int = 0) {
    fprintf(stderr, "[VAE-ORT] Not compiled (HOT_STEP_SUPERSEP not defined)\n");
    return false;
}

static inline void vae_ort_free(VaeOrt *) {}

static inline int vae_ort_decode(VaeOrt *, const float *, int, float *, int) {
    fprintf(stderr, "[VAE-ORT] Not compiled (HOT_STEP_SUPERSEP not defined)\n");
    return -1;
}

static inline int vae_ort_decode_tiled(VaeOrt *, const float *, int, float *, int, int = 256, int = 64) {
    fprintf(stderr, "[VAE-ORT] Not compiled (HOT_STEP_SUPERSEP not defined)\n");
    return -1;
}

#endif  // HOT_STEP_SUPERSEP

#endif  // VAE_ORT_H
