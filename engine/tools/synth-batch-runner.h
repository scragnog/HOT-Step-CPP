#pragma once
// synth-batch-runner.h: two-phase orchestration shared by the synth binaries
//
// Phase 1 (all groups) runs ace_synth_job_run_dit. Each call acquires the DiT
// from the store for the duration of its denoising loop and releases it on
// scope exit. Under EVICT_STRICT this means the VAE encoder, text encoder,
// cond encoder and DiT never coexist in VRAM; under EVICT_NEVER they all
// accumulate across calls. Phase 2 (all jobs) runs ace_synth_job_run_vae,
// which acquires the VAE decoder on entry and releases it on exit.

#include "pipeline-synth.h"

#include <cstdio>
#include <vector>

// Run a batch of request groups through the two synthesis phases.
//
// groups[g][i]: request i of group g. All requests in a group must share
//   the same T (same audio_codes or same duration), which the ops assume
//   when they stack per-batch tensors for a single DiT forward.
//   seed must be resolved (non-negative) on every request.
// src_audio / ref_audio: interleaved stereo 48kHz buffers, NULL when not applicable.
// audio_out[sum_g(groups[g].size())]: pre-allocated slots filled by phase 2.
//   On error, slots completed before the failure keep their audio; the rest
//   are left at {NULL, 0, 0}. Caller owns ace_audio_free.
// Returns 0 on success, -1 on any error or cancellation.
static int synth_batch_run(AceSynth *                             ctx,
                           std::vector<std::vector<AceRequest>> & groups,
                           const float *                          src_audio,
                           int                                    src_len,
                           const float *                          ref_audio,
                           int                                    ref_len,
                           AceAudio *                             audio_out,
                           std::string *                          lrc_out = nullptr,
                           bool (*cancel)(void *) = nullptr,
                           void * cancel_data     = nullptr) {
    const int                  n_groups = (int) groups.size();
    std::vector<AceSynthJob *> jobs(n_groups, nullptr);
    std::vector<int>           audio_off(n_groups, 0);

    // Phase 1: denoising loop for each group. The DiT is acquired and released
    // by ops_dit_generate inside ace_synth_job_run_dit.
    // src_latents / ref_latents: NULL = use audio path (VAE encode).
    // When we add /vae latent I/O, callers can pass pre-encoded latents here.
    int off = 0;
    for (int g = 0; g < n_groups; g++) {
        const int gn = (int) groups[g].size();
        jobs[g]      = ace_synth_job_run_dit(ctx, groups[g].data(), src_audio, src_len,
                                             nullptr, 0,  // src_latents
                                             ref_audio, ref_len,
                                             nullptr, 0,  // ref_latents
                                             gn, cancel, cancel_data);
        if (!jobs[g]) {
            for (int j = 0; j < g; j++) {
                ace_synth_job_free(jobs[j]);
            }
            return -1;
        }
        audio_off[g] = off;
        off += gn;
    }

    // Phase 2: VAE decode for each job. The decoder is acquired and released
    // by ops_vae_decode_and_splice inside ace_synth_job_run_vae.
    // Phase 3 (LRC) runs after VAE but before job free — SynthState carries
    // the encoder hidden states and lyric tokens needed by the alignment graph.
    for (int g = 0; g < n_groups; g++) {
        const int gn = (int) groups[g].size();
        const int rc =
            ace_synth_job_run_vae(ctx, jobs[g], audio_out + audio_off[g], cancel, cancel_data);
        if (rc != 0) {
            ace_synth_job_free(jobs[g]);
            jobs[g] = nullptr;
            for (int j = g + 1; j < n_groups; j++) {
                ace_synth_job_free(jobs[j]);
            }
            return -1;
        }

        // Phase 3: LRC alignment (optional)
        if (lrc_out && groups[g][0].get_lrc) {
            ace_synth_job_run_lrc(ctx, jobs[g], lrc_out + audio_off[g], gn);
        }

        ace_synth_job_free(jobs[g]);
        jobs[g] = nullptr;
    }

    return 0;
}
