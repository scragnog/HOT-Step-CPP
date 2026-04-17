#pragma once
// pipeline-synth.h: ACE-Step synthesis pipeline
//
// Loads lightweight modules (TextEnc + CondEnc + FSQ tok/detok + BPE) at init.
// DiT and VAE decoder are loaded on demand via explicit phase calls, so the
// caller can batch many jobs through the DiT while it is resident, then swap
// it out and run all VAE decodes with the largest possible tiles.

#include "request.h"

#include <cstdlib>

struct AceSynth;
struct AceSynthJob;

struct AceSynthParams {
    const char * text_encoder_path;  // Qwen3 text encoder GGUF (required)
    const char * dit_path;           // DiT GGUF (required)
    const char * vae_path;           // VAE GGUF (required)
    const char * adapter_path;       // adapter safetensors or directory (NULL to disable)
    float        adapter_scale;      // user scale multiplier, 1.0 by default
    bool         use_fa;             // flash attention (default: true)
    bool         clamp_fp16;         // clamp hidden states to FP16 range (default: false)
    bool         use_batch_cfg;      // batch cond+uncond in one DiT forward (default: true)
    int          vae_chunk;          // latent frames per tile (default: 256)
    int          vae_overlap;        // overlap frames per side (default: 64)
    const char * dump_dir;           // intermediate tensor dump dir (NULL = disabled)
};

// Output audio buffer. Caller must free with ace_audio_free().
struct AceAudio {
    float * samples;      // planar stereo [L0..LN, R0..RN]
    int     n_samples;    // per channel
    int     sample_rate;  // always 48000
};

void ace_synth_default_params(AceSynthParams * p);

// Load lightweight modules (TextEnc, CondEnc, FSQ, BPE) and cache DiT metadata.
// DiT weights and VAE weights are not allocated here.
// Returns NULL on failure.
AceSynth * ace_synth_load(const AceSynthParams * params);

// Bring the DiT into VRAM (with adapter merge if configured). Idempotent:
// returns true if already loaded. Required before ace_synth_job_run_dit.
bool ace_synth_dit_load(AceSynth * ctx);

// Release DiT VRAM. Idempotent. Safe to call while VAE is loaded.
void ace_synth_dit_unload(AceSynth * ctx);

// Bring the VAE decoder into VRAM. Idempotent: returns true if already loaded.
// Required before ace_synth_job_run_vae. Returns false when no vae_path was set.
bool ace_synth_vae_load(AceSynth * ctx);

// Release VAE decoder VRAM. Idempotent.
void ace_synth_vae_unload(AceSynth * ctx);

// Phase 1: encode sources, build context, run all DiT denoising steps.
// Requires the DiT loaded. VAE decoder may be unloaded at this point.
// Produces a job that carries DiT latents in RAM for later VAE decoding.
// reqs[batch_n]: each request has its own caption, lyrics, metadata, audio_codes, and seed.
//   The first request (reqs[0]) is used for shared params (mode, duration, DiT settings).
//   seed must be resolved (non-negative) before calling this function.
// src_audio / ref_audio: interleaved stereo 48kHz buffers, NULL when not applicable.
// batch_n: number of requests (1..9).
// cancel/cancel_data: abort callback, polled between DiT steps. NULL = never cancel.
// Returns NULL on error or cancellation.
AceSynthJob * ace_synth_job_run_dit(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      src_audio,
                                    int                src_len,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    int                batch_n,
                                    bool (*cancel)(void *) = nullptr,
                                    void * cancel_data     = nullptr);

// Phase 2: VAE decode and waveform splice.
// Requires the VAE decoder loaded. DiT may be unloaded at this point.
// splice_src / splice_len: interleaved stereo source reused for repaint/lego wave splicing.
//   Pass NULL when the job did not carry a source audio.
// out[batch_n] allocated by caller, filled with audio buffers.
// Returns 0 on success, -1 on error or cancellation.
int ace_synth_job_run_vae(AceSynth *    ctx,
                          AceSynthJob * job,
                          const float * splice_src,
                          int           splice_len,
                          AceAudio *    out,
                          bool (*cancel)(void *) = nullptr,
                          void * cancel_data     = nullptr);

// How many output slots the job expects. Equals the batch_n passed to run_dit.
int ace_synth_job_batch_n(const AceSynthJob * job);

void ace_synth_job_free(AceSynthJob * job);

void ace_audio_free(AceAudio * audio);

// Free the pipeline. Also releases any residual DiT or VAE still loaded.
void ace_synth_free(AceSynth * ctx);
