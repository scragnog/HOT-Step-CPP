#pragma once
// pipeline-synth-impl.h: private state of the synthesis pipeline
//
// Holds the types that the public API exposes as opaque handles (AceSynth,
// AceSynthJob) plus SynthState, the transient state that flows between the
// primitive ops during a single job.
//
// Included by the two implementation files:
//   pipeline-synth.cpp      orchestrator: load, phases, free
//   pipeline-synth-ops.cpp  primitives:   encode, context, noise, dit, vae

#include "bpe.h"
#include "cond-enc.h"
#include "debug.h"
#include "dit.h"
#include "fsq-detok.h"
#include "fsq-tok.h"
#include "pipeline-synth.h"
#include "qwen3-enc.h"
#include "request.h"
#include "timer.h"
#include "vae.h"

#include <string>
#include <vector>

// AceSynth carries the resident modules, the CPU-side DiT metadata cached at
// load time, and the slots for the on-demand DiT and VAE decoder. have_dit
// and have_vae track VRAM residency and gate the phase entry points.
struct AceSynth {
    // Resident modules (loaded once at ace_synth_load)
    Qwen3GGML    text_enc;
    CondGGML     cond_enc;
    DetokGGML    detok;
    TokGGML      tok;
    BPETokenizer bpe;

    // On-demand modules (loaded via ace_synth_dit_load / ace_synth_vae_load)
    DiTGGML dit;
    VAEGGML vae;
    bool    have_dit;
    bool    have_vae;

    // CPU-side DiT state, populated at ace_synth_load from the GGUF metadata.
    // Lets text encoding and T resolution run without the DiT in VRAM.
    DiTGGMLConfig      dit_cfg;
    std::vector<float> silence_full;   // [15000, 64] f32, from silence_latent tensor
    std::vector<float> null_cond_cpu;  // [hidden_size] f32, from null_condition_emb (empty when the model has none)
    bool               is_turbo;

    // Config
    AceSynthParams params;
    bool           have_detok;
    bool           have_tok;

    // Derived constants
    int Oc;      // out_channels (64)
    int ctx_ch;  // in_channels - Oc (128)
};

// Transient state for a single job, shared by reference across the primitive
// ops that make up phase 1 and phase 2.
struct SynthState {
    // model dimensions (from ctx)
    int Oc;
    int ctx_ch;

    // src audio encoding
    bool               have_cover;
    std::vector<float> cover_latents;        // [T_cover * 64] time-major VAE latents
    std::vector<float> noise_blend_latents;  // clean VAE latents for cover_noise_strength blending
    int                T_cover;

    // outpainting: silence padding for repaint beyond source bounds
    float              left_pad_sec;  // seconds of silence prepended (coordinate shift)
    std::vector<float> padded_src;    // interleaved stereo buffer with silence padding

    // mode flags
    std::string task;
    bool        is_repaint;
    bool        is_lego_region;
    float       rs;
    float       re;
    bool        use_source_context;
    bool        have_codes;
    int         max_codes_len;

    // shared params (from reqs[0])
    AceRequest rr;
    float      duration;
    int        num_steps;
    float      guidance_scale;
    float      shift;

    // diffusion schedule
    std::vector<float> schedule;

    // Solver name: dispatched via solver_lookup() in solvers/solver-registry.h
    // Valid values: "euler", "heun", "rk4", "rk5", "dopri5", "dop853",
    //              "dpm2m", "dpm3m", "dpm2m_ada", "jkass_quality", "jkass_fast",
    //              "stork2", "stork4", "sde"
    // Legacy alias: "ode" -> "euler"
    std::string solver;

    // Scheduler name: dispatched via scheduler_lookup() in schedulers/scheduler-registry.h
    // Valid values: "linear", "ddim_uniform", "sgm_uniform", "bong_tangent",
    //              "linear_quadratic", "cosine", "power", "beta57"
    // Alias: "karras" -> "sgm_uniform"
    std::string scheduler;

    // Guidance mode: dispatched via guidance_lookup() in guidance/guidance-registry.h
    // Valid values: "apg", "cfg_pp", "dynamic_cfg", "rescaled_cfg"
    std::string guidance_mode;

    // solver sub-parameters (from request, 0/-1 = use defaults)
    int   stork_substeps;      // 0 = default (10)
    float beat_stability;      // -1 = default (0.25)
    float frequency_damping;   // -1 = default (0.4)
    float temporal_smoothing;  // -1 = default (0.13)

    // guidance sub-parameters (from request, 0 = use defaults)
    float apg_momentum;        // 0 = default (0.75)
    float apg_norm_threshold;  // 0 = default (2.5)

    // per-batch seeds (for reproducible SDE re-noising: seed + step offset)
    std::vector<int64_t> seeds;

    // latent dimensions
    int T;
    int S;
    int enc_S;

    // region
    int repaint_t0;
    int repaint_t1;

    // repaint quality (resolved from repaint_strength via _resolve_repaint_config)
    float repaint_injection_ratio;
    int   repaint_crossfade_frames;
    float repaint_wav_cf_sec;

    // DiT instruction
    std::string instruction_str;     // main DiT instruction (set by orchestrator)
    std::string nc_instruction_str;  // non-cover pass instruction (for cover_steps < 1.0 switching)

    // conditioning tensors
    std::vector<float> timbre_feats;
    int                S_ref_timbre;  // actual timbre sequence length (from VAE encode)

    // text encoding (per-batch)
    std::vector<std::vector<float>> per_enc;
    std::vector<int>                per_enc_S;
    std::vector<std::vector<float>> per_enc_nc;
    std::vector<int>                per_enc_S_nc;
    bool                            need_enc_switch;

    // stacked encoder hidden states
    int                max_enc_S;
    std::vector<float> enc_hidden;
    std::vector<float> enc_hidden_nc;
    std::vector<int>   per_enc_S_nc_final;
    std::vector<float> null_cond_vec;

    // DiT context
    std::vector<float> context;
    std::vector<float> context_silence;
    int                cover_steps;

    // noise + output
    std::vector<float> noise;
    std::vector<float> output;
    std::vector<float> repaint_src;
    std::vector<int>   per_S;

    // debug / timing
    DebugDumper dbg;
    Timer       timer;
};

// Job bridges phase 1 (DiT) and phase 2 (VAE). The latents live in state.output
// as planar [batch_n, Oc, T] f32 until ace_synth_job_run_vae consumes them.
struct AceSynthJob {
    SynthState state;
    int        batch_n;
};
