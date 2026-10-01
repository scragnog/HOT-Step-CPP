#pragma once
// supersep.h: Native C++ stem separation on GGML.
//
// Implements the SuperSep multi-stage pipeline:
//   Stage 1: BS-RoFormer (6 primary stems)
//   Stage 2: Mel-Band RoFormer Karaoke (vocal split)
//   Stage 3: MDX23C DrumSep (6 drum components)
//   Stage 4: HTDemucs 6s (refine "other")
//
// Audio preprocessing (STFT/iSTFT) is handled natively in C++.
// Models are GGUF files run through GGML (bs-roformer-ggml.h, mdx23c-ggml.h).
//
// VRAM policy: sequential with the GGML model store. The caller must
// ensure DiT/VAE models are evicted before calling supersep_run().
//
// Part of HOT-Step CPP. MIT license.

#ifndef HOT_STEP_SUPERSEP_H
#define HOT_STEP_SUPERSEP_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Types ───────────────────────────────────────────────────────────────

// Opaque context (holds loaded models, configuration)
typedef struct SuperSep SuperSep;

// Separation levels (which stages to run)
typedef enum {
    SUPERSEP_BASIC       = 0,  // Stage 1 only: 6 stems
    SUPERSEP_VOCAL_SPLIT = 1,  // Stages 1+2: 8 stems (lead/backing vocals)
    SUPERSEP_FULL        = 2,  // Stages 1+2+3: 14 stems (+ drum breakdown)
    SUPERSEP_MAXIMUM     = 3,  // DEPRECATED — was stage 4 ("Other" refinement
                               // via HTDemucs), removed along with the model.
                               // The value is retained so persisted UI settings
                               // and old API callers keep working; it now
                               // behaves identically to SUPERSEP_FULL.
    SUPERSEP_VOCALS_ONLY = 4,  // BS-RoFormer pass, 2-stem output: Vocals
                               // (lead + backing) + Instrumental (mix −
                               // vocals, exact complement). Used by
                               // StableStep. NOT the karaoke model — its
                               // "vocals" is lead-only and leaves backing
                               // vocals in the instrumental.
    SUPERSEP_STABLESTEP  = 5,  // Dual BS-Roformer-Leap Xe pass, 2-stem output.
                               // Runs the vocal-target checkpoint AND the
                               // instrumental-target checkpoint over the same
                               // input and keeps the *neural output* of each,
                               // discarding both arithmetic complements.
                               //
                               // Each Leap Xe checkpoint has num_stems=1 and
                               // predicts only its target; the "other" stem a
                               // normal inference wrapper reports is just
                               // mix − target, which inherits the inverse of
                               // every model error. Taking both neural outputs
                               // means neither stem is an error bucket.
                               //
                               // Trade-off: vocals + instrumental no longer
                               // reconstruct the input exactly (anything both
                               // models claim is doubled; anything neither
                               // claims is lost). Acceptable for StableStep,
                               // which re-renders the instrumental through SA3
                               // and so destroys the complement relationship
                               // downstream anyway.
} SuperSepLevel;

// A single separated stem
typedef struct {
    char    name[64];       // Human-readable: "Lead Vocals", "Kick", etc.
    char    category[32];   // "vocals", "instruments", "drums", "other"
    char    stem_type[32];  // Machine key: "04_Lead_Vocals", "06_Kick", etc.
    float * samples;        // Interleaved stereo PCM, 44100 Hz
    int     n_samples;      // Total sample count (frames * 2 for stereo)
    int     n_frames;       // Per-channel frame count
    int     stage;          // Which stage produced this stem (1-4)
    bool    hidden;         // If true, save to disk but exclude from UI
} SuperSepStem;

// Result of a separation run
typedef struct {
    SuperSepStem * stems;
    int            n_stems;
} SuperSepResult;

// Progress callback
// stage: 1-4 (which pipeline stage), 0 for collecting/finalizing
// msg: human-readable status string
// progress: 0.0 to 1.0
typedef void (*supersep_progress_fn)(int stage, const char * msg, float progress, void * user_data);

// Cancel callback: return true to abort
typedef bool (*supersep_cancel_fn)(void * user_data);

// ── API ─────────────────────────────────────────────────────────────────

// Initialize SuperSep context.
// model_dir: directory containing the SuperSep GGUF files
// device_id: CUDA device ID (-1 for CPU)
// Returns NULL on failure.
SuperSep * supersep_init(const char * model_dir, int device_id);

// Run the separation pipeline.
// audio: interleaved stereo float PCM at 44100 Hz
// n_frames: number of frames (samples per channel)
// level: which stages to run
// progress: optional progress callback (may be NULL)
// cancel: optional cancel callback (may be NULL)
// user_data: passed to both callbacks
// Returns NULL on failure or cancellation. Caller must free with supersep_result_free().
SuperSepResult * supersep_run(
    SuperSep *          ctx,
    const float *       audio,
    int                 n_frames,
    SuperSepLevel       level,
    supersep_progress_fn progress,
    supersep_cancel_fn   cancel,
    void *              user_data
);

// Free a result and all its stem buffers.
void supersep_result_free(SuperSepResult * result);

// Free the SuperSep context and all loaded models.
void supersep_free(SuperSep * ctx);

// Release all loaded models to reclaim VRAM.
// The context remains valid; models are re-loaded on next run.
void supersep_release_models(SuperSep * ctx);

// Recombine stems with per-stem volume and mute controls.
// stems: array of stem descriptors (path not used, samples pointer is used)
// volumes: per-stem volume (0.0 to 2.0), parallel array
// muted: per-stem mute flag, parallel array
// n_stems: count
// out_frames: receives the number of output frames
// Returns interleaved stereo float PCM at 44100 Hz. Caller must free().
float * supersep_recombine(
    const SuperSepStem * stems,
    const float *        volumes,
    const bool *         muted,
    int                  n_stems,
    int *                out_frames
);

#ifdef __cplusplus
}
#endif

#endif // HOT_STEP_SUPERSEP_H
