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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// Per-group adapter scale multipliers. Applied on top of the global adapter_scale.
struct AdapterGroupScales {
    float self_attn  = 1.0f;
    float cross_attn  = 1.0f;
    float mlp         = 1.0f;
    float cond_embed  = 1.0f;
    float time_embed  = 1.0f;
    float proj_in     = 1.0f;
};

// One adapter in a (possibly multi-adapter) stack. `path` is the resolved
// absolute adapter path (flat .safetensors file or PEFT directory); `scale` is
// this adapter's individual user-scale multiplier. Group scales and basin
// re-base apply globally to every adapter in the stack.
struct AdapterSpec {
    std::string path;
    float       scale = 1.0f;
};

// One lyric section for per-section adapter masking (regional LoRA). `weights`
// are the effective per-adapter scales for this section (indexed to the adapter
// stack); `size` is a relative frame-allocation hint. See
// docs/plans/per-section-adapter-masking.md.
struct AdapterSection {
    std::vector<float> weights;
    float              size = 1.0f;
};

// Classify a GGUF tensor name into its adapter group.
// Returns "self_attn", "cross_attn", "mlp", "cond_embed", "time_embed",
// "proj_in", or "" for truly unclassified.
//
// ACE-Step v1.5 GGUF tensor naming:
//   decoder.layers.N.self_attn.{q,k,v,o}_proj.weight  → self_attn
//   decoder.layers.N.cross_attn.{q,k,v,o}_proj.weight → cross_attn
//   decoder.layers.N.mlp.{gate,up,down}_proj.weight    → mlp
//   decoder.condition_embedder.weight                  → cond_embed
//   decoder.time_embed*.{linear_1,linear_2,time_proj}  → time_embed
//   decoder.proj_in.1.weight                           → proj_in
//
// NOTE: cross_attn must be checked BEFORE self_attn.
// .ff. is kept alongside .mlp. for backward compat with older model variants.
static inline std::string adapter_determine_group(const std::string & name) {
    if (name.find("cross_attn")      != std::string::npos) return "cross_attn";
    if (name.find("self_attn")       != std::string::npos) return "self_attn";
    if (name.find(".mlp.")           != std::string::npos) return "mlp";
    if (name.find(".ff.")            != std::string::npos) return "mlp";
    if (name.find("time_embed")      != std::string::npos) return "time_embed";
    if (name.find("condition_embed") != std::string::npos) return "cond_embed";
    if (name.find("proj_in")         != std::string::npos) return "proj_in";
    return "";
}

// Look up the effective scale for a given group name.
// Truly unclassified tensors get the average of all group scales.
static inline float adapter_group_scale_for(const AdapterGroupScales & gs, const std::string & group) {
    if (group == "self_attn")   return gs.self_attn;
    if (group == "cross_attn") return gs.cross_attn;
    if (group == "mlp")        return gs.mlp;
    if (group == "cond_embed") return gs.cond_embed;
    if (group == "time_embed") return gs.time_embed;
    if (group == "proj_in")    return gs.proj_in;
    return (gs.self_attn + gs.cross_attn + gs.mlp + gs.cond_embed + gs.time_embed + gs.proj_in) / 6.0f;
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

    // Adapter loading mode: "merge" (default, F32 promoted) or "runtime" for runtime LoRA.
    // merge stores merged weights as F32 to avoid catastrophic BF16 cancellation.
    std::string adapter_mode = "merge";

    // Runtime adapter delta storage precision: "bf16" (default, full quality),
    // "q8_0" (~half VRAM), or "q4_k" (~quarter VRAM). Quantizes the precomputed
    // delta tensors in VRAM at load time — nothing is written to disk. Lets many
    // stacked adapters (per-section masking) fit in VRAM; quality impact is small
    // since the base is typically already 4-bit (NVFP4). Runtime mode only.
    std::string adapter_runtime_quant = "bf16";

    // Multi-adapter stack. When non-empty, supersedes the single adapter_path
    // passed to dit_ggml_load: every entry is applied with its own scale —
    // merged sequentially into the base weights (merge mode) or summed into the
    // per-projection runtime deltas (runtime mode). Empty = single-adapter legacy
    // path. Populated by the server worker from the resolved request adapters.
    std::vector<AdapterSpec> adapters;

    // Per-section adapter masking (regional LoRA). Ordered per lyric section.
    // When non-empty (and runtime mode), each adapter's per-projection delta is
    // gated by a per-frame mask derived from these sections, so an adapter's
    // influence varies along the song timeline. Empty = feature off (adapters
    // summed as usual). See docs/plans/per-section-adapter-masking.md.
    std::vector<AdapterSection> adapter_sections;
    // Fraction of denoising steps to run before deriving the frame→section mask
    // from cross-attention alignment (P2). <= 0 disables alignment (P1 proportional
    // map only).
    float adapter_section_align_at = 0.55f;
    // P2 token→section map: for each encoder token, which section index it belongs
    // to (or -1 for non-lyric / unmapped tokens). Length == enc_S. Built by the
    // pipeline from the lyric token texts + per-section char boundaries. When
    // non-empty (and align_at > 0), the sampler runs a mid-sampling cross-attention
    // alignment pass and rebuilds the frame→section masks from the model's real
    // lyric→audio alignment instead of the proportional guess.
    std::vector<int> adapter_section_token_map;

    // Basin re-base: nudge adapted weights toward the base the adapter was trained
    // on (S) before merging, by beta*(S - T). Lets a heavy adapter trained on one
    // DiT base behave on a sibling base. rebase_source is an absolute path to S
    // (safetensors model dir or model.safetensors), resolved by the Node server.
    // Empty / 0.0f = off. merge mode only.
    std::string rebase_source = "";
    float       rebase_beta   = 0.0f;

    // DCW (Differential Correction in Wavelet domain) — CVPR 2026
    // Training-free sampler-side correction that mitigates SNR-t bias.
    bool        dcw_enabled      = false;
    std::string dcw_mode         = "double";   // "pix", "low", "high", "double"
    float       dcw_scaler       = 0.05f;      // upstream default: 0.05
    float       dcw_high_scaler  = 0.02f;      // upstream default: 0.02 (only used in "double" mode)

    // Latent post-processing (applied after DiT output, before VAE decode)
    // Formula: output[i] = output[i] * latent_rescale + latent_shift
    float       latent_shift     = 0.0f;
    float       latent_rescale   = 1.0f;

    // CFG step scheduling: ratio of steps that use full CFG (2×compute).
    // 1.0 = all steps use CFG (default, current behavior).
    // 0.5 = CFG for first 50% of steps, cond-only for the rest (~25% speedup).
    // 0.0 = no CFG at all (fastest, but ignores guidance_scale).
    float       cfg_cutoff_ratio = 1.0f;

    // Step-level velocity caching: ratio of steps to skip by reusing the
    // previous velocity prediction. 0.0 = no caching (default), 0.5 = skip
    // ~50% of forward passes. First 2 and last 2 steps always compute.
    // Stacks with cfg_cutoff_ratio for compound speedup.
    float       cache_ratio      = 0.0f;

    // Custom timestep schedule — CSV of descending floats.
    // When non-empty, completely overrides both upstream schedule AND sideband scheduler.
    // N values = N-1 steps (trailing 0/endpoint is dropped by sampler).
    std::string custom_timesteps = "";

    // Dynamic plugin parameters (Lua plugin system).
    // Key format: "pluginName:paramKey", value is string representation.
    // Populated from the JSON request's plugin_params object.
    std::unordered_map<std::string, std::string> plugin_params;

    // Structural seed for repeated sections (Song Builder). When a repaint
    // extension wants to follow an earlier section's harmonic shape, the server
    // sends that section's clean VAE latents here; the repaint region's initial
    // noise is biased toward them by seed_strength (0 = off). Raw f32 [T, 64].
    std::vector<float> seed_latents;
    float              seed_strength = 0.0f;

    // Per-request VRAM knobs (Song Builder / low-VRAM). Override the load-time
    // synth params for this request only. 0 / -1 = use the loaded default.
    int vae_chunk_override = 0;   // >0: VAE tile size (smaller = less VAE peak)
    int batch_cfg_override = -1;  // 0: split CFG into 2 forwards (half DiT mem,
                                  //    ~2x DiT time); 1: batch; -1: default
};

// Single-worker-thread global. Set in hot-step-server.cpp before
// synth_batch_run(), read in hot-step-sampler.h during dit_ggml_generate()
// and in adapter-merge.h during adapter loading.
inline HotStepParams g_hotstep_params;

// Stable signature of a multi-adapter stack, for the DiT model cache key.
// Encodes each adapter's path and the bit-pattern of its scale so distinct
// stacks (or the same stack with a different per-adapter scale) map to distinct
// cache entries. Empty stack => empty string (single-adapter legacy keying).
static inline std::string hotstep_adapter_stack_sig(const std::vector<AdapterSpec> & adapters) {
    std::string s;
    for (const auto & a : adapters) {
        uint32_t bits;
        memcpy(&bits, &a.scale, sizeof(bits));
        char buf[16];
        snprintf(buf, sizeof(buf), "@%08x|", bits);
        s += a.path;
        s += buf;
    }
    return s;
}

