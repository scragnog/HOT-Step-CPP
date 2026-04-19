#pragma once
// request.h: AceStep generation request (JSON serialization)
//
// Pure data container + JSON read/write. Zero business logic.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct AceRequest {
    // text content
    std::string caption;  // ""
    std::string lyrics;   // ""

    // metadata (user-provided or LLM-enriched)
    int         bpm;             // 0 = unset
    float       duration;        // 0 = unset
    std::string keyscale;        // "" = unset
    std::string timesignature;   // "" = unset
    std::string vocal_language;  // "" = unset

    // generation
    int     lm_batch_size;     // 1 (number of LLM variations)
    int     synth_batch_size;  // 1 (synth batch: number of DiT variations per request)
    int64_t seed;              // -1 = random (DiT Philox noise)

    // LM control
    float       lm_temperature;      // 0.85
    float       lm_cfg_scale;        // 2.0
    float       lm_top_p;            // 0.9
    int         lm_top_k;            // 0 = disabled (matches Python None)
    std::string lm_negative_prompt;  // ""
    bool        use_cot_caption;     // true = LLM enriches caption via CoT

    // codes (Python-compatible string: "3101,11837,27514,...")
    // empty = text2music (silence context), non-empty = cover mode
    std::string audio_codes;  // ""

    // DiT control (0 = auto-detect from model: turbo vs base/sft)
    int   inference_steps;  // 0 = auto (turbo: 8, base/sft: 50)
    float guidance_scale;   // 0 = auto (1.0 for all models)
    float shift;            // 0 = auto (turbo: 3.0, base/sft: 1.0)

    // cover mode (active when source audio is provided)
    float audio_cover_strength;  // 1.0 (0-1, fraction of DiT steps using source context)
    float cover_noise_strength;  // 0.0 (0-1, how close to source: 0=pure noise, 1=source)

    // repaint region (requires source audio)
    // start: seconds offset. 0 = source start. Negative = outpaint before source.
    // end: seconds offset. Negative = source duration (sentinel).
    //      Values beyond source duration outpaint after source.
    float repainting_start;  // 0
    float repainting_end;    // -1

    // repaint/lego region quality (Python _resolve_repaint_config).
    // 0.0 = conservative (max source preservation).
    // 0.5 = balanced (default).
    // 1.0 = aggressive (pure diffusion, no injection).
    float repaint_strength;  // 0.5

    // task type: "" = auto-detect from data, or one of:
    // text2music, cover, cover-nofsq, repaint, lego, extract, complete
    std::string task_type;  // ""

    // track name for lego/extract/complete (e.g. "vocals", "drums", "guitar")
    std::string track;  // ""

    // inference method: "" or "ode" = ODE Euler, "sde" = SDE Stochastic
    std::string infer_method;  // ""

    // scheduler: "" = linear (default), or one of the registered schedulers
    std::string scheduler;  // ""

    // guidance mode: "" = apg (default), or one of the registered guidance modes
    std::string guidance_mode;  // ""

    // audio output: peak clip via percentile normalization.
    // 0 = peak normalization (100.0000th percentile, no clipping).
    // 10 = default (99.9990th percentile, clips top 0.001%).
    // 999 = max (99.9001th percentile, clips top 0.1%).
    int peak_clip;  // 10

    // solver sub-parameters (0 = use defaults)
    int   stork_substeps;      // 0 = default (10)
    float beat_stability;      // -1 = default (0.25)
    float frequency_damping;   // -1 = default (0.4)
    float temporal_smoothing;  // -1 = default (0.13)

    // guidance sub-parameters (0 = use defaults)
    float apg_momentum;        // 0 = default (0.75)
    float apg_norm_threshold;  // 0 = default (2.5)
};

// Initialize all fields to defaults (matches Python GenerationParams defaults)
void request_init(AceRequest * r);

// Parse JSON file into struct. Missing fields keep their defaults.
// Returns false on file error or malformed JSON.
bool request_parse(AceRequest * r, const char * path);

// Parse JSON string into struct. Missing fields keep their defaults.
// Returns false on malformed JSON.
bool request_parse_json(AceRequest * r, const char * json);

// Write struct to JSON file (overwrites). Returns false on file error.
bool request_write(const AceRequest * r, const char * path);

// Serialize struct to JSON string.
// sparse=true: omit fields at their default value (for cards and exports).
// sparse=false: serialize all fields (for /props documentation).
std::string request_to_json(const AceRequest * r, bool sparse = true);

// Parse JSON: single object {} or array [{}, ...] into a vector.
// Returns false on malformed JSON or empty result.
bool request_parse_json_array(const char * json, std::vector<AceRequest> * out);

// Dump human-readable summary to stream (debug)
void request_dump(const AceRequest * r, FILE * f);

// Resolve seed: if negative, replace with a hardware random value.
void request_resolve_seed(AceRequest * r);
