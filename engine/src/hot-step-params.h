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
// ACE-Step v1.5 GGUF tensor naming:
//   decoder.layers.N.self_attn.{q,k,v,o}_proj.weight  → self_attn
//   decoder.layers.N.cross_attn.{q,k,v,o}_proj.weight → cross_attn
//   decoder.layers.N.mlp.{gate,up,down}_proj.weight    → mlp
//   decoder.condition_embedder.weight                  → cond_embed
//
// NOTE: cross_attn must be checked BEFORE the generic .attn. pattern.
// .ff. is kept alongside .mlp. for backward compat with older model variants.
static inline std::string adapter_determine_group(const std::string & name) {
    if (name.find("cross_attn")      != std::string::npos) return "cross_attn";
    if (name.find(".attn.")          != std::string::npos) return "self_attn";
    if (name.find(".mlp.")           != std::string::npos) return "mlp";
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

    // Adapter loading mode: "merge" (default) or "runtime"
    std::string adapter_mode = "merge";

    // DCW (Differential Correction in Wavelet domain) — CVPR 2026
    // Training-free sampler-side correction that mitigates SNR-t bias.
    bool        dcw_enabled      = false;
    std::string dcw_mode         = "low";     // "pix", "low", "high", "double"
    float       dcw_scaler       = 0.1f;
    float       dcw_high_scaler  = 0.0f;      // only used in "double" mode

    // Latent post-processing (applied after DiT output, before VAE decode)
    // Formula: output[i] = output[i] * latent_rescale + latent_shift
    float       latent_shift     = 0.0f;
    float       latent_rescale   = 1.0f;

    // Custom timestep schedule — CSV of descending floats.
    // When non-empty, completely overrides both upstream schedule AND sideband scheduler.
    // N values = N-1 steps (trailing 0/endpoint is dropped by sampler).
    std::string custom_timesteps = "";
};

// Single-worker-thread global. Set in hot-step-server.cpp before
// synth_batch_run(), read in hot-step-sampler.h during dit_ggml_generate()
// and in adapter-merge.h during adapter loading.
inline HotStepParams g_hotstep_params;

