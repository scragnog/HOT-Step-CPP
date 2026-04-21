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

// Per-group adapter scale multipliers. Applied on top of the global adapter_scale.
struct AdapterGroupScales {
    float self_attn  = 1.0f;
    float cross_attn = 1.0f;
    float mlp        = 1.0f;
    float cond_embed = 1.0f;
};

// Classify a GGUF tensor name into its adapter group.
// Returns "self_attn", "cross_attn", "mlp", "cond_embed", or "" for unclassified.
//
// NOTE: These patterns intentionally replicate the Python hot-step-9000
// _determine_group() behaviour (.attn. / .ff. patterns) for parity with
// the settings users have been tuning against.
static inline std::string adapter_determine_group(const std::string & name) {
    if (name.find("cross_attn")      != std::string::npos) return "cross_attn";
    if (name.find(".attn.")          != std::string::npos) return "self_attn";
    if (name.find(".ff.")            != std::string::npos) return "mlp";
    if (name.find("condition_embed") != std::string::npos) return "cond_embed";
    return "";
}

// Look up the effective scale for a given group name.
// Unclassified tensors get the average of all group scales.
static inline float adapter_group_scale_for(const AdapterGroupScales & gs, const std::string & group) {
    if (group == "self_attn")   return gs.self_attn;
    if (group == "cross_attn") return gs.cross_attn;
    if (group == "mlp")        return gs.mlp;
    if (group == "cond_embed") return gs.cond_embed;
    return (gs.self_attn + gs.cross_attn + gs.mlp + gs.cond_embed) / 4.0f;
}

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

    // Per-group adapter scales
    AdapterGroupScales adapter_group_scales;
};

// Single-worker-thread global. Set in hot-step-server.cpp before
// synth_batch_run(), read in hot-step-sampler.h during dit_ggml_generate()
// and in adapter-merge.h during adapter loading.
inline HotStepParams g_hotstep_params;

