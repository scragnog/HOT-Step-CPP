#pragma once
// synth-batch-runner.h: two-phase orchestration shared by the synth binaries
//
// Runs phase 1 (DiT) on every group while the DiT is resident, unloads the
// DiT, then runs phase 2 (VAE) on every job with the widest tiles the GPU
// can hold. DiT and VAE loads are idempotent, so calling this repeatedly
// with --keep-loaded paths simply reuses what is already in VRAM.
//
// The helper does NOT unload at the end: the caller decides that based on
// --keep-loaded and picks either ace_synth_free (drop everything) or just
// letting the context live on for the next call.

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
                           bool (*cancel)(void *) = nullptr,
                           void * cancel_data     = nullptr) {
    const int                  n_groups = (int) groups.size();
    std::vector<AceSynthJob *> jobs(n_groups, nullptr);
    std::vector<int>           audio_off(n_groups, 0);

    // Phase 1: DiT resident, iterate all groups and carry latents in RAM.
    if (!ace_synth_dit_load(ctx)) {
        return -1;
    }

    int off = 0;
    for (int g = 0; g < n_groups; g++) {
        const int gn = (int) groups[g].size();
        jobs[g]      = ace_synth_job_run_dit(ctx, groups[g].data(), src_audio, src_len, ref_audio, ref_len, gn, cancel,
                                             cancel_data);
        if (!jobs[g]) {
            for (int j = 0; j < g; j++) {
                ace_synth_job_free(jobs[j]);
            }
            return -1;
        }
        audio_off[g] = off;
        off += gn;
    }

    // DiT out, VAE in: the decoder sees the full VRAM budget.
    ace_synth_dit_unload(ctx);
    if (!ace_synth_vae_load(ctx)) {
        for (int g = 0; g < n_groups; g++) {
            ace_synth_job_free(jobs[g]);
        }
        return -1;
    }

    // Phase 2: VAE decode + splice on every job.
    for (int g = 0; g < n_groups; g++) {
        const int rc =
            ace_synth_job_run_vae(ctx, jobs[g], src_audio, src_len, audio_out + audio_off[g], cancel, cancel_data);
        ace_synth_job_free(jobs[g]);
        jobs[g] = nullptr;
        if (rc != 0) {
            for (int j = g + 1; j < n_groups; j++) {
                ace_synth_job_free(jobs[j]);
            }
            return -1;
        }
    }

    return 0;
}
