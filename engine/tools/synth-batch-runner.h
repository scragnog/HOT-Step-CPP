#pragma once
// synth-batch-runner.h: three-phase orchestration shared by the synth binaries
//
// Phase 1 (all groups) runs ace_synth_job_run_dit.  Each call acquires the DiT,
// runs the denoising loop AND LRC alignment (via ops_lrc_extract, while the DiT
// is still held), then releases it.  Phase 2 (all groups) runs
// ace_synth_job_run_vae, which acquires the VAE decoder on entry and releases it
// on exit.  Phase 3 (LRC) simply copies the pre-computed alignment from
// SynthState — no DiT acquisition needed.
// Under EVICT_STRICT, at most one GPU module is resident at a time.

#include "pipeline-synth.h"

#include <cstdio>
#include <vector>

// Run a batch of request groups through the synthesis phases.
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

    // Phase 1: denoising + inline LRC for each group. ops_dit_generate
    // acquires the DiT, runs the denoising loop, then calls ops_lrc_extract
    // while the DiT is still held — avoiding a redundant adapter merge+reload
    // under EVICT_STRICT.  Results are cached in SynthState.lrc_results[].
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
    // by ops_vae_decode inside ace_synth_job_run_vae.
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

        // Phase 3: LRC — copy pre-computed alignment (no DiT acquisition)
        if (lrc_out && groups[g][0].get_lrc) {
            ace_synth_job_run_lrc(ctx, jobs[g], lrc_out + audio_off[g], gn);
        }

        ace_synth_job_free(jobs[g]);
        jobs[g] = nullptr;
    }

    return 0;
}
