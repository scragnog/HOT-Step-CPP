#pragma once
// hot-step-params.h: sideband parameter channel for HOT-Step custom features
//
// The upstream pipeline-synth-ops.cpp calls dit_ggml_generate() with only
// upstream parameters. Our custom params (solver, guidance mode, scheduler,
// APG tuning, etc.) cannot flow through the upstream pipeline without
// modifying vanilla files.
//
// Solution: a global struct set by hot-step-server.cpp before each
// synth_batch_run() call, read by hot-step-sampler.h inside dit_ggml_generate().
// Safe because the GPU worker is single-threaded.

#include <cstring>
#include <string>

struct HotStepParams {
    // Solver / scheduler
    std::string solver_name   = "euler";
    std::string scheduler     = "";       // empty = use upstream default (shifted linear)
    std::string guidance_mode = "apg";
    float       shift         = -1.0f;    // -1 = use upstream value (auto or from request)

    // APG tuning
    float apg_momentum       = 0.75f;
    float apg_norm_threshold = 2.5f;

    // STORK solver params
    int   stork_substeps     = 10;
    float beat_stability     = 0.25f;
    float frequency_damping  = 0.4f;
    float temporal_smoothing = 0.13f;
};

// Single-worker-thread global. Set in hot-step-server.cpp before
// synth_batch_run(), read in hot-step-sampler.h during dit_ggml_generate().
inline HotStepParams g_hotstep_params;
