#pragma once
// request.h: AceStep generation request (JSON serialization)
//
// Pure data container + JSON read/write. Zero business logic.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// One adapter reference in a multi-adapter stack. `name` is a registry adapter
// id (or an absolute path, resolved by the server); `scale` is its individual
// user-scale multiplier. `gain_curve` (optional) is a timestep-dependent gain:
// uniform samples of g(t) over flow-matching t ∈ [0,1], multiplied into the
// adapter's per-frame mask each sampling step (interval experts / MoE mixing).
// `gain_in_steps` (wire field "gain_domain": "steps" | "t", default "t"):
// whether gain_curve's x-axis is remaining-steps fraction or flow-matching t.
// UI windows use "steps"; trained-expert / router curves use "t".
struct AceAdapterRef {
    std::string        name;
    float              scale = 1.0f;
    std::vector<float> gain_curve;
    bool               gain_in_steps = false;
};

// One lyric section for per-section adapter masking (regional LoRA). `weights`
// gives the effective per-adapter scale for this section, indexed to `adapters`;
// `size` is a relative size hint (e.g. lyric char count) for proportional
// frame allocation. The server has already applied Sum/Blend and filled
// uniform-blend for directive-less sections, so the engine just maps it.
struct AceAdapterSection {
    std::vector<float> weights;
    float              size = 1.0f;
};

struct AceRequest {
    // text content
    std::string caption;         // ""
    std::string negative_prompt;   // "" -- encoded as uncond when non-empty
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
    int64_t seed;              // -1 = random. DiT Philox noise consumes the low
                               // 32 bits. Stored in int64_t to land positive
                               // after rd(). Range [0, UINT32_MAX], matches the
                               // Python ACE-Step reference (random.randint(0, 2**32 - 1)).

    // LM control
    float       lm_temperature;      // 0.85
    float       lm_cfg_scale;        // 2.0
    float       lm_cfg_cutoff_ratio; // 1.0 (1.0 = full CFG, 0.5 = CFG for first 50% of tokens)
    float       lm_top_p;            // 0.9
    int         lm_top_k;            // 0 = disabled (matches Python None)
    float       lm_rep_penalty;      // 1.0 = off. Windowed repetition penalty
                                     // over recently emitted audio codes; breaks
                                     // degenerate code loops (adapters sharpen
                                     // the code distribution). ~1.05-1.15.
    int         lm_rep_window;       // 64 codes (~13 s at 5 Hz) penalty lookback
    // How lm_rep_penalty is turned into a per-code logit adjustment. All three
    // modes are inert at lm_rep_penalty == 1.0.
    //   "presence"  (default, legacy) — every DISTINCT code in the window is
    //               penalised once, however often it recurred. It therefore
    //               cannot tell a stuck loop from ordinary musical restatement,
    //               and at strengths that break loops it also flattens the
    //               structural reuse (bars, phrase restatement, sustains) that
    //               the 5 Hz code stream encodes.
    //   "frequency" — penalty^occurrences, occurrences capped at
    //               LM_REP_FREQ_MAX_COUNT. Identical to "presence" for codes
    //               seen once, so the loop's few codes are hit far harder than
    //               the section's one-off codes and a much gentler penalty
    //               suffices.
    //   "dry"       — Don't Repeat Yourself. Penalises ONLY codes that would
    //               extend a verbatim recent cycle, scaled exponentially in the
    //               matched suffix length. Non-verbatim restatement is left
    //               completely alone; verbatim loops are annihilated.
    std::string lm_rep_mode;         // "presence"
    float       lm_dry_base;         // 1.75 ("dry" only) growth per extra matched code
    int         lm_dry_min_len;      // 3    ("dry" only) matched codes before any
                                     // penalty at all (~0.6 s at 5 Hz). Raise it if
                                     // legitimate sustained textures get chewed up.
                                     // Do not raise it far: this is a licence to
                                     // repeat verbatim, and the LM will use all of
                                     // it and then restart the cycle with a fresh
                                     // code. Measured on one 151 s track, min_len 6
                                     // left 4% of frames in a >=2 s loop and produced
                                     // 94 runs of >=6 verbatim frames (vs 36 with the
                                     // penalty off) -- short stitched repeats that
                                     // still sound like one continuous loop. min_len
                                     // 3 gave 0% looped frames and 4 such runs.
    std::string lm_negative_prompt;  // ""
    int64_t     lm_seed;             // -1 = random. mt19937 consumes the low 32
                                     // bits. Same int64_t storage trick as seed
                                     // above: rd() lands positive, no -1
                                     // collision, no mask needed.
    bool        use_cot_caption;     // true = LLM enriches caption via CoT

    // codes (Python-compatible string: "3101,11837,27514,...")
    // empty = text2music (silence context), non-empty = cover mode
    std::string audio_codes;  // ""

    // DiT control (0 = auto-detect from model: turbo vs base/sft)
    int   inference_steps;  // 0 = auto (turbo: 8, base/sft: 50)
    float guidance_scale;   // 0 = auto (1.0 for all models)
    float shift;            // 0 = auto (turbo: 3.0, base/sft: 1.0)

    // Self-attention sliding-window override for the DiT's layer_type=0 layers.
    // The model alternates 16 windowed / 16 full-attention layers; the windowed
    // half sees only +/- sliding_window TOKENS, and tokens run at 12.5 Hz
    // (25 Hz latents / patch_size 2). At the stock 128 that is +/-10.2 s, so a
    // repeated chorus a minute later is invisible to half the network's depth.
    //   -1 = use the model's own value (default, bit-identical to before)
    //    0 = no window at all: every layer runs full attention
    //   >0 = custom window in tokens (256 = +/-20.5 s, 1024 = +/-82 s)
    // Widening is OUT OF DISTRIBUTION — those layers were trained windowed —
    // but it is compute-free: the window is a dense S*S mask handed to
    // flash_attn_ext, not a sparse kernel, so the O(S^2) cost is already paid.
    int dit_sliding_window;  // -1 = model default

    // Differential Correction in Wavelet domain (CVPR 2026, arXiv:2604.16044).
    // Sampler-side correction for SNR-t bias in flow matching.
    // dcw_mode = "low"|"high"|"double"|"pix". dcw_scaler applies to the low
    // band in "low" and "double", to all bands in "high" and "pix". In
    // "double", dcw_high_scaler is the independent scaler for the high
    // band. Both scalers are modulated by t_curr. 0.0 disables (bit-perfect
    // master). Paper-recommended starting value: 0.1.
    float       dcw_scaler;       // 0.0 (disabled)
    float       dcw_high_scaler;  // 0.0 (only read in mode "double")
    std::string dcw_mode;         // "low"

    // cover mode (active when source audio is provided)
    float audio_cover_strength;  // 1.0 (0-1, fraction of DiT steps using source context)
    float cover_noise_strength;  // 0.0 (0-1, how close to source: 0=pure noise, 1=source)
    std::string cover_noise_method;  // "" = truncate (default), "rescale" = rebuild schedule

    // repaint region (requires source audio)
    // start: seconds offset. 0 = source start. Negative = outpaint before source.
    // end: seconds offset. Negative = source duration (sentinel).
    //      Values beyond source duration outpaint after source.
    float repainting_start;  // 0
    float repainting_end;    // -1

    // Latent post-processing applied after DiT sampling, before VAE decode:
    //   pred = pred * latent_rescale + latent_shift
    // Defaults are no-op.
    float latent_shift;    // 0.0
    float latent_rescale;  // 1.0

    // LSS: Latent Spectral Suppressor (MDMAchine / A&E Concepts).
    // Attenuates low-variance latent channels before VAE decode; channels whose
    // relative variance is below lss_var_thresh are scaled toward 1-lss_strength.
    // lss_strength 0 = off (default).
    float lss_strength;    // 0.0 = off
    float lss_var_thresh;  // 0.15
    bool  lss_dc_remove;   // true (per-channel DC removal, only when LSS active)

    // Custom flow matching schedule: comma-separated floats,
    // e.g. "0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0". When non-empty,
    // overrides inference_steps and shift. The trailing endpoint is the x0
    // target dropped by the sampler, so a CSV of N values yields N-1 steps.
    std::string custom_timesteps;  // ""

    // task type: one of text2music, cover, cover-nofsq, repaint, lego, extract, complete.
    // Default: text2music.
    std::string task_type;  // "text2music"

    // track name for lego/extract/complete (e.g. "vocals", "drums", "guitar")
    std::string track;  // ""

    // Solver name resolved by solver_lookup() (see src/solvers).
    // Accepted values: "euler", "sde", "dpm3m", "stork4".
    std::string solver;          // "euler"
    int         stork_substeps;  // 10, only used by the "stork4" solver

    // LM mode: "generate" (full: metadata + lyrics + codes),
    // "inspire" (short query -> metadata + lyrics, no codes),
    // "format" (caption + lyrics -> metadata + lyrics, no codes). Default: generate.
    std::string lm_mode;  // "generate"

    // Audio output format: "mp3", "wav16", "wav24", "wav32". Default: mp3.
    std::string output_format;  // "mp3"

    // model selection. synth_model, lm_model and vae are resolved through
    // the registry scanned from --models <dir>, by both the HTTP server and
    // the CLI binaries. An empty value falls to the first matching entry of
    // the registry. adapter and adapter_scale are read by server and
    // ace-synth and resolved against --adapters <dir> when set.
    std::string synth_model;    // ""
    std::string lm_model;       // ""
    std::string adapter;        // ""
    float       adapter_scale;  // 1.0

    // Planner-LM runtime LoRA (local HOT-Step feature): name resolved
    // against --adapters <dir>/lm (or an absolute/relative path), applied
    // to the 5Hz LM at runtime (never merged, base quant untouched).
    std::string lm_adapter;        // ""
    float       lm_adapter_scale;  // 1.0
    // Multi-adapter stack. When non-empty, supersedes `adapter`/`adapter_scale`:
    // every entry is applied with its own scale (merged or summed depending on
    // adapter_mode). Each name is resolved against --adapters by the server.
    std::vector<AceAdapterRef> adapters;
    // Per-section adapter masking (regional LoRA). Ordered per lyric section.
    // When non-empty, each adapter's influence varies along the song timeline
    // per these sections. Runtime mode only. Empty = feature off.
    std::vector<AceAdapterSection> adapter_sections;
    std::string vae;            // ""

    // PP-VAE re-encode: round-trip audio through the post-processing VAE
    // for spectral cleanup. Only applies when a PP-VAE GGUF is available.
    bool pp_vae_reencode;  // false

    // Postprocess plugin: name of the Lua postprocess plugin to use for VAE
    // decode. Empty = disabled (use built-in tiled decoder).
    std::string postprocess_plugin;  // ""

    // LRC timestamp generation toggle. When true, the synth pipeline runs an
    // additional DiT forward pass post-generation to extract cross-attention
    // alignment and produce LRC-format synchronized lyrics.
    bool get_lrc;  // false

    // audio output: peak clip via percentile normalization.
    // 0 = peak normalization (100.0000th percentile, no clipping).
    // 10 = default (99.9990th percentile, clips top 0.001%).
    // 999 = max (99.9001th percentile, clips top 0.1%).
    int peak_clip;  // 10

    // audio output: MP3 encoder bitrate in kbps. Applies only when
    // output_format is "mp3". WAV outputs ignore this field.
    int mp3_bitrate;  // 128
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

// Resolve LM seed: if negative, replace with a hardware random value.
void request_resolve_lm_seed(AceRequest * r);
