#pragma once
// vae-enc-ort.h: ONNX Runtime VAE encoder wrapper (TensorRT / CUDA EP)
//
// Encapsulates ORT session lifecycle and inference for the VAE encoder
// exported as an ONNX model. Supports TensorRT EP (preferred, builds an
// optimised engine on first run) with CUDA EP fallback.
//
// The VAE encoder ONNX model takes audio input [batch, 2, samples]
// and produces latent output [batch, 64, T_latent] (mean only, deterministic).
// T_latent = samples / 1920 (downsampling factor of the VAE).
//
// Reuses the VaeOrt struct and vae_ort_load/free from vae-ort.h since the
// session management is identical — only the I/O tensor layout differs.
//
// Thread safety: none. Caller serialises access (single GPU worker thread).
//
// Guarded by HOT_STEP_SUPERSEP because ORT headers and libraries are only
// available when the SuperSep feature is compiled in (shared ORT dependency).
//
// Part of HOT-Step CPP. MIT license.

#ifndef VAE_ENC_ORT_H
#define VAE_ENC_ORT_H

#include "vae-ort.h"  // VaeOrt, vae_ort_load, vae_ort_free

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

// We reuse VaeOrt for the encoder — it's just an ORT session.
// Type alias for clarity in the model store.
using VaeEncOrt = VaeOrt;

#ifdef HOT_STEP_SUPERSEP

// ── Encode ─────────────────────────────────────────────────────────────
//
// audio: [T_audio, 2] f32 interleaved stereo (same layout as vae_enc_encode),
//        or [T_audio * 2] flat interleaved.
// latent_out: caller-allocated buffer, at least (T_audio / 1920) * 64 floats,
//             written as [T_latent, 64] time-major (matching GGML encoder output).
// max_T_latent: max latent frames the buffer can hold.
//
// Returns T_latent (number of latent frames), or -1 on error.

static inline int vae_enc_ort_encode(VaeEncOrt *   ctx,
                                      const float * audio,         // [T_audio * 2] interleaved stereo
                                      int           T_audio,
                                      float *       latent_out,    // [T_latent, 64] time-major
                                      int           max_T_latent) {
    if (!ctx || !ctx->session || !audio || T_audio <= 0) return -1;

    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input: [1, 2, T_audio] — ONNX model expects [batch, channels, samples].
    // Our audio is [T_audio, 2] interleaved stereo, so deinterleave to channel-first.
    std::vector<float> input_buf((size_t)2 * T_audio);
    for (int t = 0; t < T_audio; t++) {
        input_buf[t]             = audio[t * 2];      // L channel
        input_buf[T_audio + t]   = audio[t * 2 + 1];  // R channel
    }

    std::vector<int64_t> input_shape = {1, 2, (int64_t)T_audio};
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
        fprintf(stderr, "[VAE-Enc-ORT] Inference failed: %s\n", e.what());
        return -1;
    }

    // Parse output shape — expected [1, 64, T_latent]
    auto & out_tensor = outputs[0];
    auto   type_info  = out_tensor.GetTensorTypeAndShapeInfo();
    auto   out_shape  = type_info.GetShape();
    const float * out_data = out_tensor.GetTensorData<float>();

    int T_latent = 0;
    int n_channels = 64;

    if (out_shape.size() == 3) {
        n_channels = (int)out_shape[1];
        T_latent   = (int)out_shape[2];
    } else if (out_shape.size() == 2) {
        n_channels = (int)out_shape[0];
        T_latent   = (int)out_shape[1];
    } else {
        fprintf(stderr, "[VAE-Enc-ORT] Unexpected output rank: %zu\n", out_shape.size());
        return -1;
    }

    if (n_channels != 64) {
        fprintf(stderr, "[VAE-Enc-ORT] WARNING: expected 64 latent channels, got %d\n", n_channels);
    }

    if (T_latent > max_T_latent) {
        fprintf(stderr, "[VAE-Enc-ORT] WARNING: output T_latent=%d > max=%d, clamping\n",
                T_latent, max_T_latent);
        T_latent = max_T_latent;
    }

    // Transpose: ONNX output [1, 64, T_latent] channel-first → [T_latent, 64] time-major
    // to match the GGML encoder output layout expected by callers.
    for (int t = 0; t < T_latent; t++) {
        for (int c = 0; c < 64; c++) {
            latent_out[t * 64 + c] = out_data[c * T_latent + t];
        }
    }

    fprintf(stderr, "[VAE-Enc-ORT] Encoded: T_audio=%d → T_latent=%d (%.2fs @ 48kHz, TRT=%s)\n",
            T_audio, T_latent, (float)T_audio / 48000.0f,
            ctx->using_trt ? "yes" : "no");

    return T_latent;
}

// ── Tiled encode ──────────────────────────────────────────────────────────
//
// Overlap-discard chunking for bounded VRAM usage, mirroring
// vae_enc_encode_tiled(). Each tile encodes an audio window with overlap
// context; the core (non-overlap) portion is kept and concatenated.
//
// chunk_size / overlap are in latent frames (same as GGML version).
// Returns T_latent (total latent frames) or -1 on error.

static inline int vae_enc_ort_encode_tiled(VaeEncOrt *   ctx,
                                            const float * audio,        // [T_audio * 2] interleaved stereo
                                            int           T_audio,
                                            float *       latent_out,   // [max_T_latent, 64] time-major
                                            int           max_T_latent,
                                            int           chunk_size = 256,
                                            int           overlap    = 64) {
    if (!ctx || !ctx->session || !audio || T_audio <= 0) return -1;

    // Work in audio-sample space. Each latent frame = 1920 audio samples.
    int audio_chunk   = chunk_size * 1920;
    int audio_overlap = overlap * 1920;

    // Shrink overlap until stride is positive
    while (audio_chunk - 2 * audio_overlap <= 0 && audio_overlap > 0) {
        audio_overlap /= 2;
    }

    // Short audio: encode directly
    if (T_audio <= audio_chunk) {
        return vae_enc_ort_encode(ctx, audio, T_audio, latent_out, max_T_latent);
    }

    int audio_stride = audio_chunk - 2 * audio_overlap;
    int num_steps    = (T_audio + audio_stride - 1) / audio_stride;

    fprintf(stderr, "[VAE-Enc-ORT] Tiled encode: %d tiles (chunk=%d, overlap=%d, stride=%d audio samples)\n",
            num_steps, audio_chunk, audio_overlap, audio_stride);

    float downsample_factor = 0.0f;
    int   latent_write_pos  = 0;

    // Temporary buffer for each tile's latent output
    int tile_latent_max = chunk_size + 64;  // generous
    std::vector<float> tile_latent((size_t)tile_latent_max * 64);

    for (int i = 0; i < num_steps; i++) {
        // Core range in audio samples (the part we keep)
        int core_start = i * audio_stride;
        int core_end   = core_start + audio_stride;
        if (core_end > T_audio) core_end = T_audio;

        // Window with overlap context
        int win_start = core_start - audio_overlap;
        if (win_start < 0) win_start = 0;
        int win_end = core_end + audio_overlap;
        if (win_end > T_audio) win_end = T_audio;
        int win_len = win_end - win_start;

        // Encode this window
        int tile_T = vae_enc_ort_encode(ctx,
                                         audio + win_start * 2,  // interleaved stereo offset
                                         win_len,
                                         tile_latent.data(),
                                         tile_latent_max);
        if (tile_T < 0) {
            fprintf(stderr, "[VAE-Enc-ORT] FATAL: tile %d encode failed\n", i);
            return -1;
        }

        // Determine downsample factor from first tile
        if (i == 0) {
            downsample_factor = (float)tile_T / (float)win_len;
            fprintf(stderr, "[VAE-Enc-ORT] Downsample factor: %.6f (expected ~1/1920)\n",
                    downsample_factor);
        }

        // Trim in latent frames (mirror of decoder trim logic)
        int added_start = core_start - win_start;
        int trim_start  = (int)roundf((float)added_start * downsample_factor);
        int added_end   = win_end - core_end;
        int trim_end    = (int)roundf((float)added_end * downsample_factor);

        int end_idx  = (trim_end > 0) ? (tile_T - trim_end) : tile_T;
        int core_len = end_idx - trim_start;
        if (core_len <= 0) continue;

        if (latent_write_pos + core_len > max_T_latent) {
            fprintf(stderr, "[VAE-Enc-ORT] FATAL: tiled output exceeds max_T_latent\n");
            return -1;
        }

        // Copy trimmed core from tile_latent (already time-major [T, 64])
        memcpy(latent_out + latent_write_pos * 64,
               tile_latent.data() + trim_start * 64,
               (size_t)core_len * 64 * sizeof(float));

        latent_write_pos += core_len;
    }

    fprintf(stderr, "[VAE-Enc-ORT] Tiled encode done: %d tiles → T_latent=%d (%.2fs @ 48kHz, TRT=%s)\n",
            num_steps, latent_write_pos, (float)T_audio / 48000.0f,
            ctx->using_trt ? "yes" : "no");

    return latent_write_pos;
}

#else  // !HOT_STEP_SUPERSEP — stubs

static inline int vae_enc_ort_encode(VaeEncOrt *, const float *, int, float *, int) {
    fprintf(stderr, "[VAE-Enc-ORT] Not compiled (HOT_STEP_SUPERSEP not defined)\n");
    return -1;
}

static inline int vae_enc_ort_encode_tiled(VaeEncOrt *, const float *, int, float *, int, int = 256, int = 64) {
    fprintf(stderr, "[VAE-Enc-ORT] Not compiled (HOT_STEP_SUPERSEP not defined)\n");
    return -1;
}

#endif  // HOT_STEP_SUPERSEP

#endif  // VAE_ENC_ORT_H
