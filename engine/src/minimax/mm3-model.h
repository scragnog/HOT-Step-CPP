#pragma once
// minimax/mm3-model.h — MiniMax-Music3 GGUF loader, config parse and VRAM residency.
//
// HOT-Step file (does not exist upstream). Nothing in engine/src/*.h outside
// engine/src/minimax/ may include this: the only wiring into upstream-derived
// sources is the single hook include in tools/hot-step-server.cpp
// (`#include "minimax/mm3-server.h"`), per the fork's hook doctrine.
//
// SCOPE OF THIS FILE (increment 1): open the two GGUFs, parse every mm3.* KV
// into typed config structs, validate every tensor's name AND config-derived
// shape, put the weights in a backend buffer, and account the VRAM. There are
// NO compute graphs and NO inference here — those are later increments. The
// tensor pointers collected below are the handles those increments will build
// graphs from.
//
// The contract this consumes is docs/plans/mm3-gguf-layout.md, produced by
// engine/tools/convert-mm3.py (+ engine/tools/split-mm3.py). Five files:
//
//   mm3-lm-<quant>.gguf     arch "qwen3"      399 tensors  llama.cpp tensor names
//   mm3-depth-<quant>.gguf  arch "mm3-depth"   47 tensors  depth.*
//   mm3-cond-<quant>.gguf   arch "mm3-cond"     4 tensors  cond.*
//   mm3-dit-<quant>.gguf    arch "mm3-dit"    370 tensors  dit.*  (+ mm3.flow.* KVs)
//   mm3-voc-<quant>.gguf    arch "mm3-voc"     91 tensors  voc.*
//
// LEGACY: the original two-file layout bundled depth/cond/dit/voc into one
//   mm3-synth-<quant>.gguf  arch "mm3"        512 tensors
// and it still loads: a bundle enters every non-LM role's variant list, and in
// bundle mode all four role files simply point at the same path. Split files
// win over a bundle offering the same quant token. The point of the split
// (ported from ServeurpersoCom/minimaxmusic.cpp, which validated the policy):
// per-role quant mixing — LM Q5_K_M + DiT Q4_K_M with depth/cond/voc near
// native — and a DiT quant/adapter change that reloads ~300 MB of cond/voc
// alongside the DiT instead of re-reading a 6 GB bundle.
//
// Discovery scans <models>/mm3/*.gguf first (the canonical home — keeps 24 GB
// of Music-3 weights out of the flat ACE model scan, which classifies by
// general.architecture and would just warn "unknown architecture"), then falls
// back to <models>/mm3-*.gguf for a flat layout.
//
// Residency stays THREE parts (lm | depth | rest = cond+dit+voc), not five:
// stage 1 needs the LM *and* depth resident at once, stage 2 needs the rest,
// and cond/voc are too small (~300 MB) for finer eviction to buy anything.
// The file split is orthogonal to the residency split — each part just loads
// its tensors from whichever file(s) its roles resolved to.

#include "backend.h"
#include "gguf-weights.h"
#include "minimax/mm3-adapter.h"
#include "weight-ctx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

// For mm3_list_dir (quant-variant enumeration).
#ifdef _WIN32
#    include <windows.h>
#else
#    include <dirent.h>
#endif

// ── Config structs ──────────────────────────────────────────────────────────

// mm3-lm-*.gguf: standard llama.cpp qwen3.* geometry + the mm3.* audio surface.
struct MM3LmConfig {
    // qwen3.* (llama.cpp standard keys)
    uint32_t block_count          = 0;
    uint32_t context_length       = 0;
    uint32_t embedding_length     = 0;
    uint32_t feed_forward_length  = 0;
    uint32_t head_count           = 0;
    uint32_t head_count_kv        = 0;
    uint32_t key_length           = 0;
    uint32_t value_length         = 0;
    float    rms_eps              = 0.0f;
    float    rope_freq_base       = 0.0f;
    uint32_t vocab_size           = 0;

    // mm3.* audio vocabulary / AR loop surface
    uint32_t semantic_vocab_offset = 0;  // token id of semantic code 0 (151675)
    uint32_t semantic_vocab_size   = 0;  // codebook 0 size (16384)
    uint32_t acoustic_vocab_size   = 0;  // codebooks 1..7 (1024)
    uint32_t num_codebooks         = 0;  // 8 = 1 semantic + 7 acoustic
    uint32_t eos_audio             = 0;  // real AR stop token (151670), NOT tokenizer eos
    uint32_t frame_rate            = 0;  // AR frames/second (25)
    uint32_t max_audio_frames      = 0;  // 9000 = 6 min
    uint32_t max_prompt_tokens     = 0;  // 5000
    float    ar_cfg_scale          = 0.0f;   // checkpoint-fixed 1.5, not a user knob
    uint32_t ar_top_k              = 0;      // checkpoint-fixed 50
    float    ar_embedding_scale    = 0.0f;   // 8^-0.5, on the summed frame embedding

    // mm3.token.*
    uint32_t tok_im_start      = 0;
    uint32_t tok_im_end        = 0;
    uint32_t tok_audio_cfg     = 0;
    uint32_t tok_audio_start   = 0;
    uint32_t tok_audio_end     = 0;
    uint32_t tok_caption_start = 0;
    uint32_t tok_caption_end   = 0;
    uint32_t tok_lyrics_start  = 0;
    uint32_t tok_lyrics_end    = 0;
};

// ── AR guidance rows ────────────────────────────────────────────────────────
//
// The AR stage (LM + depth decoder) normally evaluates a conditional and an
// unconditional row and blends them at mm3.ar.cfg_scale. A guidance-DISTILLED
// composer — the depth-pruned 5.7B students are trained against the teacher's
// CFG-guided distributions — declares a scale of 1.0, at which
//
//     guided = u + (c - u) * 1.0 == c
//
// identically, for every logit. The unconditional row is then computed, read
// back, and algebraically cancelled: pure waste. Both stages therefore build
// single-row graphs at 1.0 and the full pair at anything else.
//
// Keyed on the arithmetic, never on a model name, so it is correct for any
// checkpoint that ships this scale — and bit-identical to the old code path at
// the stock 1.5. Lives here rather than in mm3-lm-graph.h because the depth
// decoder needs it too and only sees mm3-model.h.
#define MM3_LM_CFG_ROWS 2

// The scale to guide with, with the pre-KV-metadata fallback applied once.
static inline float mm3_ar_cfg_scale(const MM3LmConfig & c) {
    return c.ar_cfg_scale > 0.0f ? c.ar_cfg_scale : 1.5f;
}

static inline int mm3_cfg_rows(const MM3LmConfig & c) {
    // MM3_AR_CFG_ROWS=2 forces the pair back on for a guidance-baked checkpoint.
    // This is a CORRECTNESS GATE, not a tuning knob: at scale 1.0 the blend is
    // the identity, so a 2-row render MUST be bit-identical to the 1-row one.
    // If it is not, the single-row path is broken and the difference is a bug
    // here — not the checkpoint's fault. Run it before blaming a model for
    // incoherence. (=1 forces single-row; anything else is ignored.)
    static const int forced = [] {
        const char * e = std::getenv("MM3_AR_CFG_ROWS");
        if (!e || !e[0]) {
            return 0;
        }
        const int v = atoi(e);
        return (v == 1 || v == 2) ? v : 0;
    }();
    if (forced) {
        return forced;
    }
    return std::fabs(mm3_ar_cfg_scale(c) - 1.0f) <= 1e-6f ? 1 : MM3_LM_CFG_ROWS;
}

// RVQ depth decoder — 4-layer llama-shaped causal stack, no RoPE.
struct MM3DepthConfig {
    uint32_t block_count         = 0;
    uint32_t embedding_length    = 0;
    uint32_t feed_forward_length = 0;
    uint32_t head_count          = 0;
    uint32_t head_dim            = 0;
    uint32_t max_position        = 0;  // learned absolute pos_embd rows (16)
    float    rms_eps             = 0.0f;
    uint32_t num_codebooks       = 0;
    uint32_t audio_vocab_size    = 0;  // per acoustic head (1024)
    uint32_t audio_embd_rows     = 0;  // 7168 = 7 * 1024
    bool     causal              = false;
    bool     rope                = false;  // documented FALSE
};

// Condition encoder — ~25 M, consumes 8 LM hidden layers.
struct MM3CondConfig {
    uint32_t    num_layers           = 0;  // 8 LM layers mixed by softmax(layer_logits)
    uint32_t    hidden_dim           = 0;
    uint32_t    out_dim              = 0;
    uint32_t    kernel_size          = 0;
    uint32_t    padding              = 0;
    uint32_t    input_sampling_rate  = 0;
    uint32_t    input_hop_length     = 0;
    uint32_t    output_sampling_rate = 0;
    uint32_t    output_hop_length    = 0;
    std::string interpolation;  // "nearest"
    std::string layer_mix;      // "softmax"
};

// Flow-matching transformer.
struct MM3DitConfig {
    uint32_t block_count      = 0;
    uint32_t embedding_length = 0;
    uint32_t head_count       = 0;
    uint32_t head_dim         = 0;
    uint32_t ff_inner         = 0;
    uint32_t in_channels      = 0;
    uint32_t condition_dim    = 0;
    uint32_t concat_channels  = 0;
    float    layer_norm_eps   = 0.0f;
    uint32_t rope_dim         = 0;
    float    rope_theta       = 0.0f;
    std::string rope_type;   // "neox"
    uint32_t fourier_dim      = 0;
    std::string glu_order;   // "value_gate" — getting this backwards is silent
    bool     output_negated           = false;
    bool     timestep_token_prepended = false;
    bool     pre_post_conv_residual   = false;
    bool     attn_bias                = false;
    uint32_t window_frames  = 0;
    uint32_t hop_frames     = 0;
    uint32_t window_latents = 0;
    uint32_t hop_latents    = 0;
};

struct MM3FlowConfig {
    std::string scheduler;  // "FlowMatchEulerDiscrete"
    uint32_t    steps               = 0;
    float       cfg_scale           = 0.0f;
    bool        invert_sigmas       = false;
    float       shift               = 0.0f;
    uint32_t    num_train_timesteps = 0;
};

// DAC-style vocoder decoder, weight-norm folded at conversion.
struct MM3VocConfig {
    uint32_t              latent_channels   = 0;
    uint32_t              fold_channels     = 0;  // stereo fold: [B,128,T] -> [B*2,64,T]
    uint32_t              dec_in_dim        = 0;
    uint32_t              hidden_dim        = 0;
    std::vector<int32_t>  upsample_rates;         // [8,8,4,2]
    std::vector<int32_t>  res_dilations;          // [1,3,9]
    uint32_t              total_upsample    = 0;  // 512
    uint32_t              sampling_rate     = 0;  // 44100 per the code (model card says 32k)
    uint32_t              channels          = 0;
    float                 snake_eps         = 0.0f;
    bool                  final_tanh        = false;
    bool                  weight_norm_folded = false;
    std::string           snake;  // formula string, documentation only
};

struct MM3SynthConfig {
    std::vector<std::string> components;  // ["depth","cond","dit","vocoder"]
    MM3DepthConfig depth;
    MM3CondConfig  cond;
    MM3DitConfig   dit;
    MM3FlowConfig  flow;
    MM3VocConfig   voc;
};

// ── Weight structs — one ggml_tensor* per GGUF tensor ───────────────────────

struct MM3LmLayer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_q      = nullptr;
    ggml_tensor * attn_k      = nullptr;
    ggml_tensor * attn_v      = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_gate    = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_down    = nullptr;
};

struct MM3LmWeights {
    ggml_tensor *           token_embd  = nullptr;
    ggml_tensor *           output_norm = nullptr;
    ggml_tensor *           output      = nullptr;  // tie_word_embeddings is false
    std::vector<MM3LmLayer> blk;
};

struct MM3DepthLayer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_q      = nullptr;
    ggml_tensor * attn_k      = nullptr;
    ggml_tensor * attn_v      = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_gate    = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_down    = nullptr;
};

struct MM3DepthWeights {
    ggml_tensor *              proj        = nullptr;
    ggml_tensor *              pos_embd    = nullptr;
    ggml_tensor *              audio_embd  = nullptr;  // also read by the AR feedback path
    ggml_tensor *              output_norm = nullptr;
    std::vector<ggml_tensor *> head;                   // one per acoustic codebook
    std::vector<MM3DepthLayer> blk;
};

struct MM3CondWeights {
    ggml_tensor * layer_logits = nullptr;
    ggml_tensor * layer_scale  = nullptr;
    ggml_tensor * proj_w       = nullptr;
    ggml_tensor * proj_b       = nullptr;
};

struct MM3DitBlock {
    ggml_tensor * attn_norm_w   = nullptr;  // LayerNorm gamma
    ggml_tensor * attn_norm_b   = nullptr;  // LayerNorm beta  (NOT RMSNorm)
    ggml_tensor * attn_qkv      = nullptr;  // fused q||k||v
    ggml_tensor * attn_output   = nullptr;
    ggml_tensor * ffn_norm_w    = nullptr;
    ggml_tensor * ffn_norm_b    = nullptr;
    ggml_tensor * ffn_in_w      = nullptr;  // GLU, value first
    ggml_tensor * ffn_in_b      = nullptr;
    ggml_tensor * ffn_out_w     = nullptr;
    ggml_tensor * ffn_out_b     = nullptr;
};

struct MM3DitWeights {
    ggml_tensor *             preprocess_conv  = nullptr;
    ggml_tensor *             postprocess_conv = nullptr;
    ggml_tensor *             time_fourier     = nullptr;
    ggml_tensor *             time_embd_w[2]   = { nullptr, nullptr };
    ggml_tensor *             time_embd_b[2]   = { nullptr, nullptr };
    ggml_tensor *             proj_in          = nullptr;
    ggml_tensor *             proj_out         = nullptr;
    ggml_tensor *             rope_inv_freq    = nullptr;  // synthesised F32 by the converter
    std::vector<MM3DitBlock>  blk;
};

struct MM3VocResUnit {
    ggml_tensor * snake1_alpha = nullptr;
    ggml_tensor * conv1_w      = nullptr;
    ggml_tensor * conv1_b      = nullptr;
    ggml_tensor * snake2_alpha = nullptr;
    ggml_tensor * conv2_w      = nullptr;
    ggml_tensor * conv2_b      = nullptr;
};

struct MM3VocBlock {
    ggml_tensor *              snake_alpha = nullptr;
    ggml_tensor *              convt_w     = nullptr;
    ggml_tensor *              convt_b     = nullptr;
    std::vector<MM3VocResUnit> res;
};

struct MM3VocWeights {
    ggml_tensor *             dec_in_w    = nullptr;
    ggml_tensor *             dec_in_b    = nullptr;
    ggml_tensor *             conv_in_w   = nullptr;
    ggml_tensor *             conv_in_b   = nullptr;
    std::vector<MM3VocBlock>  blk;
    ggml_tensor *             snake_out_alpha = nullptr;
    ggml_tensor *             conv_out_w      = nullptr;
    ggml_tensor *             conv_out_b      = nullptr;
};

struct MM3SynthWeights {
    MM3DepthWeights depth;
    MM3CondWeights  cond;
    MM3DitWeights   dit;
    MM3VocWeights   voc;
};

// ── File-level metadata, filled by a header-only probe (no weights loaded) ───

struct MM3FileInfo {
    bool        found = false;
    std::string path;
    std::string name;         // basename
    std::string arch;         // general.architecture
    std::string general_name;
    std::string license;
    std::string source_layout;      // comfy | diffusers | unknown
    uint32_t    converter_version = 0;
    uint32_t    file_type         = 0;  // 1 = MOSTLY_F16, 7 = MOSTLY_Q8_0
    uint64_t    file_bytes        = 0;
    uint64_t    tensor_bytes      = 0;  // sum of ggml_nbytes over the header
    int         n_tensors         = 0;
    bool        probe_ok          = false;
    std::string probe_error;
};

// ── Roles ───────────────────────────────────────────────────────────────────

// The four non-LM components, each its own file in the split layout. The LM
// stays a named member: different arch, different config struct, and the
// tokenizer is coupled to its path.
enum MM3SynthRole { MM3_R_DEPTH = 0, MM3_R_COND, MM3_R_DIT, MM3_R_VOC, MM3_R_COUNT };

static const char * const MM3_SYNTH_ROLE[MM3_R_COUNT] = { "depth", "cond", "dit", "voc" };

// ── Quant variants ──────────────────────────────────────────────────────────

// One mm3-<role>-<quant>.gguf found on disk. The quant token is taken from
// the filename verbatim -- it is the authoritative label (general.file_type is
// a display-only u32, and has no assigned value for some quants).
struct MM3Variant {
    std::string quant;   // "f16" | "q8_0" | "Q4_K_M" | ...
    std::string path;
    std::string name;    // basename
    uint64_t    bytes  = 0;
    bool        bundle = false;  // true when this is a legacy mm3-synth bundle
};

// ── The model ───────────────────────────────────────────────────────────────

struct MM3Model {
    // discovery + probe (populated at server start, cheap, header-only)
    std::string              models_dir;
    std::vector<std::string> search_dirs;
    MM3FileInfo              lm_file;
    // One resolved file per non-LM role. In legacy bundle mode several (or all
    // four) entries share one path; loading dedupes by path, but the byte
    // accounting must too — use mm3_total_tensor_bytes(), never a plain sum.
    MM3FileInfo              role_file[MM3_R_COUNT];
    // Every quant found on disk, best-first. The UI's model dropdown is built
    // from these; lm_file/role_file point at whichever is currently selected.
    std::vector<MM3Variant>  lm_variants;
    std::vector<MM3Variant>  role_variants[MM3_R_COUNT];
    // User's explicit quant choice, "" = auto (best-first). Kept separate from
    // lm_file so a re-scan that no longer sees the chosen file falls back
    // gracefully instead of leaving MM3 unavailable.
    std::string              want_lm_quant;
    std::string              want_role_quant[MM3_R_COUNT];
    MM3LmConfig              lm_cfg;
    MM3SynthConfig           synth_cfg;
    std::vector<std::string> meta_errors;

    // residency
    //
    // Three independently-loadable parts, because the pipeline needs them at
    // different times: stage 1 (AR) uses lm + depth, stage 2 (cond/flow/vocode)
    // uses rest. `loaded` means ALL THREE are resident — the precondition the
    // bring-up endpoints and a keep-loaded warm still expect.
    bool           loaded         = false;
    bool           lm_resident    = false;
    bool           depth_resident = false;
    bool           rest_resident  = false;
    // Identity of the LM adapter merged INTO the resident LM weights
    // (mm3-lm-merge.h), or "" = pristine base. Encodes (path, mtime, scales),
    // so any change forces an LM reload before the next merge or base render.
    // Cleared wherever the LM weights are freed — a fresh load is pristine.
    std::string    lm_merge_tag;
    bool           backend_ref    = false;
    ggml_backend_t backend        = nullptr;
    ggml_backend_t cpu_backend    = nullptr;
    WeightCtx      wctx_lm        = {};
    WeightCtx      wctx_depth     = {};
    WeightCtx      wctx_synth     = {};   // cond + dit + voc
    size_t         vram_lm        = 0;
    size_t         vram_depth     = 0;
    size_t         vram_synth     = 0;
    double         load_ms        = 0.0;

    MM3LmWeights    lm;
    MM3SynthWeights synth;

    // Adapters merged into wctx_synth's DiT weights at load. A merge is
    // irreversible in-place, so this is the record of what the resident weights
    // ACTUALLY contain — /mm3/props reports it, and a later increment compares
    // it against the requested set to decide whether `rest` must be reloaded.
    // Empty means the DiT is pristine.
    std::string     rest_adapter_desc;

    // name -> tensor, per file. Introspection only; graphs use the typed structs.
    std::map<std::string, ggml_tensor *> tmap_lm;
    std::map<std::string, ggml_tensor *> tmap_synth;
};

// ── Small helpers ───────────────────────────────────────────────────────────

#ifdef _WIN32
#    define MM3_SEP "\\"
#else
#    define MM3_SEP "/"
#endif

static bool mm3_file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

static std::string mm3_basename(const std::string & path) {
    size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? path : path.substr(p + 1);
}

static uint64_t mm3_file_size(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return 0;
    }
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    const uint64_t n = (uint64_t) _ftelli64(f);
#else
    fseeko(f, 0, SEEK_END);
    const uint64_t n = (uint64_t) ftello(f);
#endif
    fclose(f);
    return n;
}

// List plain files in a directory. Deliberately local rather than reusing
// model-registry.h's registry_list_dir: this module is self-contained by
// design (see the file header) and the whole helper is a dozen lines.
static void mm3_list_dir(const std::string & dir, std::vector<std::string> * names) {
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE           h = FindFirstFileA((dir + "\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            names->push_back(fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return;
    }
    while (struct dirent * e = readdir(d)) {
        if (e->d_type != DT_DIR) {
            names->push_back(e->d_name);
        }
    }
    closedir(d);
#endif
}

// Read a GGUF int array as int32. Accepts INT32/UINT32 element types.
static std::vector<int32_t> mm3_get_i32_arr(const GGUFModel & gf, const char * key) {
    std::vector<int32_t> out;
    int64_t              idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_ARRAY) {
        return out;
    }
    enum gguf_type et = gguf_get_arr_type(gf.gguf, idx);
    if (et != GGUF_TYPE_INT32 && et != GGUF_TYPE_UINT32) {
        return out;
    }
    size_t          n = gguf_get_arr_n(gf.gguf, idx);
    const int32_t * d = (const int32_t *) gguf_get_arr_data(gf.gguf, idx);
    out.assign(d, d + n);
    return out;
}

static std::vector<std::string> mm3_get_str_arr(const GGUFModel & gf, const char * key) {
    std::vector<std::string> out;
    int64_t                  idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_ARRAY) {
        return out;
    }
    if (gguf_get_arr_type(gf.gguf, idx) != GGUF_TYPE_STRING) {
        return out;
    }
    size_t n = gguf_get_arr_n(gf.gguf, idx);
    for (size_t i = 0; i < n; i++) {
        out.push_back(gguf_get_arr_str(gf.gguf, idx, i));
    }
    return out;
}

// Loader with config-derived shape validation. Every tensor is fetched by exact
// name and checked against the shape the KV metadata implies — a converter that
// silently changed a dim, or a half-written file, fails here with both shapes
// named instead of producing garbage audio ten increments later.
//
// Never exits the process: a bad MM3 file must not take the ACE-Step server
// down with it, so errors accumulate and the caller aborts the load cleanly.
struct MM3Loader {
    WeightCtx *                            wctx   = nullptr;
    const GGUFModel *                      gf     = nullptr;
    std::map<std::string, ggml_tensor *> * tmap   = nullptr;
    std::vector<std::string> *             errors = nullptr;

    void fail(const std::string & msg) const {
        if (errors && errors->size() < 24) {
            errors->push_back(msg);
        }
    }

    ggml_tensor * req(const std::string & name, int64_t e0, int64_t e1 = 1, int64_t e2 = 1, int64_t e3 = 1) {
        if (gguf_find_tensor(gf->gguf, name.c_str()) < 0) {
            fail("missing tensor '" + name + "'");
            return nullptr;
        }
        ggml_tensor * src = ggml_get_tensor(gf->meta, name.c_str());
        if (!src) {
            fail("tensor '" + name + "' not in meta context");
            return nullptr;
        }
        const int64_t want[4] = { e0, e1, e2, e3 };
        for (int i = 0; i < 4; i++) {
            if (src->ne[i] != want[i]) {
                char buf[320];
                snprintf(buf, sizeof(buf),
                         "tensor '%s' shape mismatch: file [%lld,%lld,%lld,%lld] expected [%lld,%lld,%lld,%lld]",
                         name.c_str(), (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2],
                         (long long) src->ne[3], (long long) want[0], (long long) want[1], (long long) want[2],
                         (long long) want[3]);
                fail(buf);
                return nullptr;
            }
        }
        ggml_tensor * t = gf_load_tensor(wctx, *gf, name);  // safe: presence verified above
        if (t && tmap) {
            (*tmap)[name] = t;
        }
        return t;
    }
};

static std::string mm3_fmt(const char * fmt, int a) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, a);
    return buf;
}

static std::string mm3_fmt2(const char * fmt, int a, int b) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, a, b);
    return buf;
}

// ── Config parsing ──────────────────────────────────────────────────────────

static void mm3_parse_lm_config(const GGUFModel & gf, MM3LmConfig * c) {
    c->block_count         = gf_get_u32(gf, "qwen3.block_count");
    c->context_length      = gf_get_u32(gf, "qwen3.context_length");
    c->embedding_length    = gf_get_u32(gf, "qwen3.embedding_length");
    c->feed_forward_length = gf_get_u32(gf, "qwen3.feed_forward_length");
    c->head_count          = gf_get_u32(gf, "qwen3.attention.head_count");
    c->head_count_kv       = gf_get_u32(gf, "qwen3.attention.head_count_kv");
    c->key_length          = gf_get_u32(gf, "qwen3.attention.key_length");
    c->value_length        = gf_get_u32(gf, "qwen3.attention.value_length");
    c->rms_eps             = gf_get_f32(gf, "qwen3.attention.layer_norm_rms_epsilon");
    c->rope_freq_base      = gf_get_f32(gf, "qwen3.rope.freq_base");
    c->vocab_size          = gf_get_u32(gf, "qwen3.vocab_size");

    c->semantic_vocab_offset = gf_get_u32(gf, "mm3.semantic_vocab_offset");
    c->semantic_vocab_size   = gf_get_u32(gf, "mm3.semantic_vocab_size");
    c->acoustic_vocab_size   = gf_get_u32(gf, "mm3.acoustic_vocab_size");
    c->num_codebooks         = gf_get_u32(gf, "mm3.num_codebooks");
    c->eos_audio             = gf_get_u32(gf, "mm3.eos_audio");
    c->frame_rate            = gf_get_u32(gf, "mm3.frame_rate");
    c->max_audio_frames      = gf_get_u32(gf, "mm3.max_audio_frames");
    c->max_prompt_tokens     = gf_get_u32(gf, "mm3.max_prompt_tokens");
    c->ar_cfg_scale          = gf_get_f32(gf, "mm3.ar.cfg_scale");
    c->ar_top_k              = gf_get_u32(gf, "mm3.ar.top_k");
    c->ar_embedding_scale    = gf_get_f32(gf, "mm3.ar.embedding_scale");

    c->tok_im_start      = gf_get_u32(gf, "mm3.token.im_start");
    c->tok_im_end        = gf_get_u32(gf, "mm3.token.im_end");
    c->tok_audio_cfg     = gf_get_u32(gf, "mm3.token.audio_cfg");
    c->tok_audio_start   = gf_get_u32(gf, "mm3.token.audio_start");
    c->tok_audio_end     = gf_get_u32(gf, "mm3.token.audio_end");
    c->tok_caption_start = gf_get_u32(gf, "mm3.token.caption_start");
    c->tok_caption_end   = gf_get_u32(gf, "mm3.token.caption_end");
    c->tok_lyrics_start  = gf_get_u32(gf, "mm3.token.lyrics_start");
    c->tok_lyrics_end    = gf_get_u32(gf, "mm3.token.lyrics_end");
}

// Per-section parsers: a split file carries exactly one section's KVs, a
// legacy bundle carries all four. The probe applies whichever sections the
// file is expected to provide.
static void mm3_parse_depth_config(const GGUFModel & gf, MM3SynthConfig * c) {
    MM3DepthConfig & d     = c->depth;
    d.block_count          = gf_get_u32(gf, "mm3.depth.block_count");
    d.embedding_length     = gf_get_u32(gf, "mm3.depth.embedding_length");
    d.feed_forward_length  = gf_get_u32(gf, "mm3.depth.feed_forward_length");
    d.head_count           = gf_get_u32(gf, "mm3.depth.head_count");
    d.head_dim             = gf_get_u32(gf, "mm3.depth.head_dim");
    d.max_position         = gf_get_u32(gf, "mm3.depth.max_position");
    d.rms_eps              = gf_get_f32(gf, "mm3.depth.rms_eps");
    d.num_codebooks        = gf_get_u32(gf, "mm3.depth.num_codebooks");
    d.audio_vocab_size     = gf_get_u32(gf, "mm3.depth.audio_vocab_size");
    d.audio_embd_rows      = gf_get_u32(gf, "mm3.depth.audio_embd_rows");
    d.causal               = gf_get_bool(gf, "mm3.depth.causal");
    d.rope                 = gf_get_bool(gf, "mm3.depth.rope");
}

static void mm3_parse_cond_config(const GGUFModel & gf, MM3SynthConfig * c) {
    MM3CondConfig & e        = c->cond;
    e.num_layers             = gf_get_u32(gf, "mm3.cond.num_layers");
    e.hidden_dim             = gf_get_u32(gf, "mm3.cond.hidden_dim");
    e.out_dim                = gf_get_u32(gf, "mm3.cond.out_dim");
    e.kernel_size            = gf_get_u32(gf, "mm3.cond.kernel_size");
    e.padding                = gf_get_u32(gf, "mm3.cond.padding");
    e.input_sampling_rate    = gf_get_u32(gf, "mm3.cond.input_sampling_rate");
    e.input_hop_length       = gf_get_u32(gf, "mm3.cond.input_hop_length");
    e.output_sampling_rate   = gf_get_u32(gf, "mm3.cond.output_sampling_rate");
    e.output_hop_length      = gf_get_u32(gf, "mm3.cond.output_hop_length");
    e.interpolation          = gf_get_str(gf, "mm3.cond.interpolation");
    e.layer_mix              = gf_get_str(gf, "mm3.cond.layer_mix");
}

// The flow schedule KVs ride with the DiT file: they drive its Euler loop.
static void mm3_parse_dit_config(const GGUFModel & gf, MM3SynthConfig * c) {
    MM3DitConfig & t             = c->dit;
    t.block_count                = gf_get_u32(gf, "mm3.dit.block_count");
    t.embedding_length           = gf_get_u32(gf, "mm3.dit.embedding_length");
    t.head_count                 = gf_get_u32(gf, "mm3.dit.head_count");
    t.head_dim                   = gf_get_u32(gf, "mm3.dit.head_dim");
    t.ff_inner                   = gf_get_u32(gf, "mm3.dit.ff_inner");
    t.in_channels                = gf_get_u32(gf, "mm3.dit.in_channels");
    t.condition_dim              = gf_get_u32(gf, "mm3.dit.condition_dim");
    t.concat_channels            = gf_get_u32(gf, "mm3.dit.concat_channels");
    t.layer_norm_eps             = gf_get_f32(gf, "mm3.dit.layer_norm_eps");
    t.rope_dim                   = gf_get_u32(gf, "mm3.dit.rope_dim");
    t.rope_theta                 = gf_get_f32(gf, "mm3.dit.rope_theta");
    t.rope_type                  = gf_get_str(gf, "mm3.dit.rope_type");
    t.fourier_dim                = gf_get_u32(gf, "mm3.dit.fourier_dim");
    t.glu_order                  = gf_get_str(gf, "mm3.dit.glu_order");
    t.output_negated             = gf_get_bool(gf, "mm3.dit.output_negated");
    t.timestep_token_prepended   = gf_get_bool(gf, "mm3.dit.timestep_token_prepended");
    t.pre_post_conv_residual     = gf_get_bool(gf, "mm3.dit.pre_post_conv_residual");
    t.attn_bias                  = gf_get_bool(gf, "mm3.dit.attn_bias");
    t.window_frames              = gf_get_u32(gf, "mm3.dit.window_frames");
    t.hop_frames                 = gf_get_u32(gf, "mm3.dit.hop_frames");
    t.window_latents             = gf_get_u32(gf, "mm3.dit.window_latents");
    t.hop_latents                = gf_get_u32(gf, "mm3.dit.hop_latents");

    MM3FlowConfig & f      = c->flow;
    f.scheduler            = gf_get_str(gf, "mm3.flow.scheduler");
    f.steps                = gf_get_u32(gf, "mm3.flow.steps");
    f.cfg_scale            = gf_get_f32(gf, "mm3.flow.cfg_scale");
    f.invert_sigmas        = gf_get_bool(gf, "mm3.flow.invert_sigmas");
    f.shift                = gf_get_f32(gf, "mm3.flow.shift");
    f.num_train_timesteps  = gf_get_u32(gf, "mm3.flow.num_train_timesteps");
}

static void mm3_parse_voc_config(const GGUFModel & gf, MM3SynthConfig * c) {
    MM3VocConfig & v      = c->voc;
    v.latent_channels     = gf_get_u32(gf, "mm3.voc.latent_channels");
    v.fold_channels       = gf_get_u32(gf, "mm3.voc.fold_channels");
    v.dec_in_dim          = gf_get_u32(gf, "mm3.voc.dec_in_dim");
    v.hidden_dim          = gf_get_u32(gf, "mm3.voc.hidden_dim");
    v.upsample_rates      = mm3_get_i32_arr(gf, "mm3.voc.upsample_rates");
    v.res_dilations       = mm3_get_i32_arr(gf, "mm3.voc.res_dilations");
    v.total_upsample      = gf_get_u32(gf, "mm3.voc.total_upsample");
    v.sampling_rate       = gf_get_u32(gf, "mm3.voc.sampling_rate");
    v.channels            = gf_get_u32(gf, "mm3.voc.channels");
    v.snake_eps           = gf_get_f32(gf, "mm3.voc.snake_eps");
    v.final_tanh          = gf_get_bool(gf, "mm3.voc.final_tanh");
    v.weight_norm_folded  = gf_get_bool(gf, "mm3.voc.weight_norm_folded");
    v.snake               = gf_get_str(gf, "mm3.voc.snake");
}

// Sanity checks that catch a structurally wrong file before any VRAM is touched.
// Only invariants the layout doc pins down — not a re-derivation of the model.
static void mm3_validate_lm_config(const MM3LmConfig & c, std::vector<std::string> * errs) {
    auto need = [&](bool ok, const char * what) {
        if (!ok && errs->size() < 24) {
            errs->push_back(std::string("LM config: ") + what);
        }
    };
    need(c.block_count > 0, "qwen3.block_count is 0 or missing");
    need(c.embedding_length > 0, "qwen3.embedding_length is 0 or missing");
    need(c.vocab_size > 0, "qwen3.vocab_size is 0 or missing");
    need(c.head_count > 0 && c.head_count_kv > 0, "head counts are 0 or missing");
    need(c.key_length > 0 && c.value_length > 0, "key/value length is 0 or missing");
    need(c.num_codebooks > 0, "mm3.num_codebooks is 0 or missing (not an MM3 LM?)");
    need(c.semantic_vocab_offset > 0, "mm3.semantic_vocab_offset is 0 or missing");
    need(c.vocab_size >= c.semantic_vocab_offset + c.semantic_vocab_size,
         "vocab_size does not cover semantic_vocab_offset + semantic_vocab_size");
    need(c.eos_audio > 0, "mm3.eos_audio is 0 or missing");
    need(c.frame_rate > 0, "mm3.frame_rate is 0 or missing");
    need(c.head_count * c.key_length == c.embedding_length,
         "head_count * key_length != embedding_length");
}

// Per-section validators, applied only to the sections a file provides.
static void mm3_validate_section(const MM3SynthConfig & c, MM3SynthRole role, std::vector<std::string> * errs) {
    auto need = [&](bool ok, const char * what) {
        if (!ok && errs->size() < 24) {
            errs->push_back(std::string(MM3_SYNTH_ROLE[role]) + " config: " + what);
        }
    };
    switch (role) {
        case MM3_R_DEPTH:
            need(c.depth.block_count > 0, "mm3.depth.block_count is 0 or missing");
            need(c.depth.embedding_length > 0, "mm3.depth.embedding_length is 0 or missing");
            need(c.depth.num_codebooks > 1, "mm3.depth.num_codebooks must be > 1");
            need(c.depth.head_count * c.depth.head_dim == c.depth.embedding_length,
                 "depth head_count * head_dim != embedding_length");
            need(c.depth.audio_embd_rows == (c.depth.num_codebooks - 1) * c.depth.audio_vocab_size,
                 "depth audio_embd_rows != (num_codebooks-1) * audio_vocab_size");
            break;
        case MM3_R_COND:
            need(c.cond.num_layers > 0 && c.cond.hidden_dim > 0, "cond dims are 0 or missing");
            break;
        case MM3_R_DIT:
            need(c.dit.block_count > 0 && c.dit.embedding_length > 0, "dit dims are 0 or missing");
            need(c.dit.head_count * c.dit.head_dim == c.dit.embedding_length,
                 "dit head_count * head_dim != embedding_length");
            need(c.dit.concat_channels == c.dit.in_channels * 2 + c.dit.condition_dim,
                 "dit concat_channels != 2*in_channels + condition_dim");
            break;
        case MM3_R_VOC:
            need(c.voc.upsample_rates.size() == 4, "mm3.voc.upsample_rates is not 4 entries");
            need(c.voc.res_dilations.size() == 3, "mm3.voc.res_dilations is not 3 entries");
            need(c.voc.hidden_dim > 0 && c.voc.dec_in_dim > 0, "voc dims are 0 or missing");
            need(c.voc.latent_channels == c.voc.fold_channels * 2,
                 "voc latent_channels != 2 * fold_channels (stereo fold)");
            break;
        default:
            break;
    }
}

// ── Probe: read a file's header only, no weights ─────────────────────────────

// Probe one file's header. `lm_cfg` non-null = this is the LM file. Otherwise
// `role_mask` says which sections (bit per MM3SynthRole) this file provides —
// one bit for a split file, all four for a legacy bundle — and those sections
// are parsed into `synth_cfg` and validated.
static void mm3_probe_file(const std::string & path, MM3FileInfo * fi, MM3LmConfig * lm_cfg,
                           MM3SynthConfig * synth_cfg, uint32_t role_mask, std::vector<std::string> * errs) {
    fi->found = true;
    fi->path  = path;
    fi->name  = mm3_basename(path);

    GGUFModel gf = {};
    if (!gf_load(&gf, path.c_str())) {
        fi->probe_error = "gguf header parse failed";
        if (errs && errs->size() < 24) {
            errs->push_back(fi->name + ": " + fi->probe_error);
        }
        return;
    }

    fi->file_bytes        = (uint64_t) gf.file_size;
    fi->arch              = gf_get_str(gf, "general.architecture");
    fi->general_name      = gf_get_str(gf, "general.name");
    fi->license           = gf_get_str(gf, "general.license");
    fi->source_layout     = gf_get_str(gf, "mm3.source_layout");
    fi->converter_version = gf_get_u32(gf, "mm3.converter_version");
    fi->file_type         = gf_get_u32(gf, "general.file_type");
    fi->n_tensors         = (int) gguf_get_n_tensors(gf.gguf);

    uint64_t bytes = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(gf.gguf); i++) {
        ggml_tensor * t = ggml_get_tensor(gf.meta, gguf_get_tensor_name(gf.gguf, i));
        if (t) {
            bytes += (uint64_t) ggml_nbytes(t);
        }
    }
    fi->tensor_bytes = bytes;

    const std::string model_kv = gf_get_str(gf, "mm3.model");
    if (model_kv != "MiniMax-Music3") {
        fi->probe_error = "mm3.model KV is '" + model_kv + "', expected 'MiniMax-Music3'";
        if (errs && errs->size() < 24) {
            errs->push_back(fi->name + ": " + fi->probe_error);
        }
        gf_close(&gf);
        return;
    }

    if (lm_cfg) {
        mm3_parse_lm_config(gf, lm_cfg);
        mm3_validate_lm_config(*lm_cfg, errs);
    }
    if (synth_cfg && role_mask) {
        if (role_mask & (1u << MM3_R_DEPTH)) {
            mm3_parse_depth_config(gf, synth_cfg);
            mm3_validate_section(*synth_cfg, MM3_R_DEPTH, errs);
        }
        if (role_mask & (1u << MM3_R_COND)) {
            mm3_parse_cond_config(gf, synth_cfg);
            mm3_validate_section(*synth_cfg, MM3_R_COND, errs);
        }
        if (role_mask & (1u << MM3_R_DIT)) {
            mm3_parse_dit_config(gf, synth_cfg);
            mm3_validate_section(*synth_cfg, MM3_R_DIT, errs);
        }
        if (role_mask & (1u << MM3_R_VOC)) {
            mm3_parse_voc_config(gf, synth_cfg);
            mm3_validate_section(*synth_cfg, MM3_R_VOC, errs);
        }
    }

    fi->probe_ok = true;
    gf_close(&gf);
}

// ── Discovery ───────────────────────────────────────────────────────────────

// Rank a quant token for "best first" ordering. Lower is better. This is a
// fidelity ordering, not a size ordering: f16 stays first so that adding a
// quantized file next to an existing f16 install never silently changes which
// weights a user's generations run on. Explicit selection overrides it.
static int mm3_quant_rank(const std::string & q) {
    static const char * order[] = { "f16",   "F16",   "bf16",  "BF16",  "q8_0",  "Q8_0",
                                    "Q6_K",  "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
                                    "NVFP4", "MXFP4", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K" };
    for (int i = 0; i < (int) (sizeof(order) / sizeof(order[0])); i++) {
        if (q == order[i]) {
            return i;
        }
    }
    return 1000;   // unknown quant: offered, but never auto-selected over a known one
}

// Enumerate every mm3-<role>-<quant>.gguf across the search dirs, best-first.
// The mm3/ subdir wins over the models root for the same quant token.
static void mm3_enumerate(const MM3Model & m, const char * role, std::vector<MM3Variant> * out) {
    out->clear();
    const std::string prefix = std::string("mm3-") + role + "-";
    for (const auto & dir : m.search_dirs) {
        std::vector<std::string> names;
        mm3_list_dir(dir, &names);
        for (const auto & n : names) {
            if (n.size() <= prefix.size() + 5 || n.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            if (n.compare(n.size() - 5, 5, ".gguf") != 0) {
                continue;
            }
            MM3Variant v;
            v.quant = n.substr(prefix.size(), n.size() - prefix.size() - 5);
            bool dup = false;
            for (const auto & e : *out) {
                if (e.quant == v.quant) {
                    dup = true;   // earlier search dir already supplied this quant
                    break;
                }
            }
            if (dup) {
                continue;
            }
            v.path  = dir + MM3_SEP + n;
            v.name  = n;
            v.bytes = mm3_file_size(v.path);
            out->push_back(v);
        }
    }
    std::sort(out->begin(), out->end(), [](const MM3Variant & a, const MM3Variant & b) {
        const int ra = mm3_quant_rank(a.quant), rb = mm3_quant_rank(b.quant);
        return ra != rb ? ra < rb : a.quant < b.quant;
    });
}

// Boot-time probe: does <models_dir> (or its mm3/ subdir — the same two
// directories mm3_discover searches) hold ANY mm3-*.gguf? Filename check only,
// no header reads. The server's startup gates use this to stay alive on an
// MM3-only install instead of exiting over an empty ACE registry — a dead
// process can't serve /mm3/*, and /mm3/props is where a partial or broken MM3
// install gets reported accurately.
static bool mm3_weights_present(const char * models_dir) {
    if (!models_dir || !models_dir[0]) {
        return false;
    }
    const std::string root    = models_dir;
    const std::string dirs[2] = { root + MM3_SEP "mm3", root };
    for (const auto & dir : dirs) {
        std::vector<std::string> names;
        mm3_list_dir(dir, &names);
        for (const auto & n : names) {
            // mm3-<role>-<quant>.gguf: at least one char between the prefix
            // and the extension, mirroring mm3_enumerate's bounds.
            if (n.size() > 9 && n.compare(0, 4, "mm3-") == 0 &&
                n.compare(n.size() - 5, 5, ".gguf") == 0) {
                return true;
            }
        }
    }
    return false;
}

// Locate the GGUFs and probe them. Called once at server start; cheap
// (mmap + header parse, no weight reads). Safe to call when nothing is there —
// MM3 is simply reported unavailable.
//
// `want_lm` / `want_roles[]` name a quant token to prefer per role (from the
// UI's model dropdown). Empty, or not present on disk, falls back to
// best-first. Legacy mm3-synth bundles enter every role's variant list (marked
// `bundle`); a split file wins over a bundle at the same quant token.
static void mm3_discover(MM3Model * m, const char * models_dir, const std::string & want_lm = std::string(),
                         const std::string * want_roles = nullptr) {
    m->models_dir = models_dir ? models_dir : "";
    m->search_dirs.clear();
    m->meta_errors.clear();
    m->lm_file = MM3FileInfo{};
    for (int r = 0; r < MM3_R_COUNT; r++) {
        m->role_file[r] = MM3FileInfo{};
    }

    if (m->models_dir.empty()) {
        return;
    }

    const std::string sub = m->models_dir + MM3_SEP "mm3";
    m->search_dirs.push_back(sub);
    m->search_dirs.push_back(m->models_dir);

    mm3_enumerate(*m, "lm", &m->lm_variants);

    std::vector<MM3Variant> bundles;
    mm3_enumerate(*m, "synth", &bundles);
    for (auto & b : bundles) {
        b.bundle = true;
    }

    for (int r = 0; r < MM3_R_COUNT; r++) {
        mm3_enumerate(*m, MM3_SYNTH_ROLE[r], &m->role_variants[r]);
        // A bundle can stand in for a role — but ONLY when that role has no
        // split files at all (a pure pre-split install). Merging bundles into
        // a role that IS split would put every bundle quant in the UI's
        // dropdown, inviting a Q2_K condition encoder — the near-native-only
        // policy the split exists to enforce. All-or-nothing per role keeps
        // legacy installs working and split installs clean.
        if (m->role_variants[r].empty()) {
            m->role_variants[r] = bundles;
        }
        std::sort(m->role_variants[r].begin(), m->role_variants[r].end(),
                  [](const MM3Variant & a, const MM3Variant & b) {
                      const int ra = mm3_quant_rank(a.quant), rb = mm3_quant_rank(b.quant);
                      return ra != rb ? ra < rb : a.quant < b.quant;
                  });
    }

    auto pick = [](const std::vector<MM3Variant> & vars, const std::string & want, MM3FileInfo * fi) {
        if (vars.empty()) {
            return;
        }
        const MM3Variant * chosen = &vars[0];
        if (!want.empty()) {
            for (const auto & v : vars) {
                if (v.quant == want) {
                    chosen = &v;
                    break;
                }
            }
        }
        fi->found = true;
        fi->path  = chosen->path;
        fi->name  = chosen->name;
    };

    pick(m->lm_variants, want_lm, &m->lm_file);
    for (int r = 0; r < MM3_R_COUNT; r++) {
        pick(m->role_variants[r], want_roles ? want_roles[r] : std::string(), &m->role_file[r]);
    }

    if (m->lm_file.found) {
        mm3_probe_file(m->lm_file.path, &m->lm_file, &m->lm_cfg, nullptr, 0, &m->meta_errors);
        if (m->lm_file.probe_ok && m->lm_file.arch != "qwen3") {
            m->meta_errors.push_back(m->lm_file.name + ": general.architecture is '" + m->lm_file.arch +
                                     "', expected 'qwen3'");
        }
    }

    // Probe each DISTINCT non-LM path once, with the union of the roles it
    // serves, then stamp the result onto every role that resolved to it.
    m->synth_cfg = MM3SynthConfig{};
    for (int r = 0; r < MM3_R_COUNT; r++) {
        if (!m->role_file[r].found || m->role_file[r].probe_ok || !m->role_file[r].probe_error.empty()) {
            continue;  // absent, or already stamped by an earlier role's probe
        }
        uint32_t mask = 0;
        for (int r2 = 0; r2 < MM3_R_COUNT; r2++) {
            if (m->role_file[r2].found && m->role_file[r2].path == m->role_file[r].path) {
                mask |= 1u << r2;
            }
        }
        MM3FileInfo probed = m->role_file[r];
        mm3_probe_file(probed.path, &probed, nullptr, &m->synth_cfg, mask, &m->meta_errors);
        if (probed.probe_ok) {
            const bool is_bundle = probed.arch == "mm3";
            for (int r2 = 0; r2 < MM3_R_COUNT; r2++) {
                if (mask & (1u << r2)) {
                    const std::string want_arch = std::string("mm3-") + MM3_SYNTH_ROLE[r2];
                    if (!is_bundle && probed.arch != want_arch) {
                        m->meta_errors.push_back(probed.name + ": general.architecture is '" + probed.arch +
                                                 "', expected '" + want_arch + "' or 'mm3'");
                    }
                }
            }
        }
        for (int r2 = 0; r2 < MM3_R_COUNT; r2++) {
            if (mask & (1u << r2)) {
                m->role_file[r2] = probed;
            }
        }
    }
    // Informational: which components resolved (props reports this).
    m->synth_cfg.components.clear();
    for (int r = 0; r < MM3_R_COUNT; r++) {
        if (m->role_file[r].probe_ok) {
            m->synth_cfg.components.push_back(r == MM3_R_VOC ? "vocoder" : MM3_SYNTH_ROLE[r]);
        }
    }

    bool any_role = false;
    for (int r = 0; r < MM3_R_COUNT; r++) {
        any_role = any_role || m->role_file[r].found;
    }
    if (!m->lm_file.found && !any_role) {
        fprintf(stderr, "[MM3] No MiniMax-Music3 GGUFs found (looked in %s and %s)\n", sub.c_str(),
                m->models_dir.c_str());
        return;
    }
    fprintf(stderr, "[MM3] LM:    %s\n", m->lm_file.found ? m->lm_file.path.c_str() : "(not found)");
    for (int r = 0; r < MM3_R_COUNT; r++) {
        fprintf(stderr, "[MM3] %-5s: %s\n", MM3_SYNTH_ROLE[r],
                m->role_file[r].found ? m->role_file[r].name.c_str() : "(not found)");
    }
    {
        auto log_variants = [](const char * role, const std::vector<MM3Variant> & vars) {
            if (vars.size() < 2) {
                return;
            }
            std::string line;
            for (const auto & v : vars) {
                line += (line.empty() ? "" : ", ") + v.quant + (v.bundle ? "(bundle)" : "");
            }
            fprintf(stderr, "[MM3] %s quants available: %s\n", role, line.c_str());
        };
        log_variants("LM", m->lm_variants);
        for (int r = 0; r < MM3_R_COUNT; r++) {
            log_variants(MM3_SYNTH_ROLE[r], m->role_variants[r]);
        }
    }
    if (m->lm_file.probe_ok) {
        fprintf(stderr, "[MM3] LM cfg: %uL H=%u V=%u heads=%u/%u codebooks=%u sem@%u+%u fps=%u\n",
                m->lm_cfg.block_count, m->lm_cfg.embedding_length, m->lm_cfg.vocab_size, m->lm_cfg.head_count,
                m->lm_cfg.head_count_kv, m->lm_cfg.num_codebooks, m->lm_cfg.semantic_vocab_offset,
                m->lm_cfg.semantic_vocab_size, m->lm_cfg.frame_rate);
    }
    if (m->role_file[MM3_R_DEPTH].probe_ok && m->role_file[MM3_R_COND].probe_ok &&
        m->role_file[MM3_R_DIT].probe_ok && m->role_file[MM3_R_VOC].probe_ok) {
        fprintf(stderr, "[MM3] Synth cfg: depth %uL/%u  cond %u->%u  dit %uL/%u  voc %u->%u Hz x%u\n",
                m->synth_cfg.depth.block_count, m->synth_cfg.depth.embedding_length, m->synth_cfg.cond.hidden_dim,
                m->synth_cfg.cond.out_dim, m->synth_cfg.dit.block_count, m->synth_cfg.dit.embedding_length,
                m->synth_cfg.voc.latent_channels, m->synth_cfg.voc.sampling_rate, m->synth_cfg.voc.total_upsample);
    }
    for (const auto & e : m->meta_errors) {
        fprintf(stderr, "[MM3] WARNING: %s\n", e.c_str());
    }
}

static bool mm3_available(const MM3Model & m) {
    bool ok = m.lm_file.probe_ok && m.meta_errors.empty();
    for (int r = 0; r < MM3_R_COUNT; r++) {
        ok = ok && m.role_file[r].probe_ok;
    }
    return ok;
}

// Weight bytes on disk across all resolved files, counting each DISTINCT path
// once — in bundle mode all four role files point at the same GGUF.
static uint64_t mm3_total_tensor_bytes(const MM3Model & m) {
    uint64_t total = m.lm_file.found ? m.lm_file.tensor_bytes : 0;
    for (int r = 0; r < MM3_R_COUNT; r++) {
        if (!m.role_file[r].found) {
            continue;
        }
        bool seen = false;
        for (int r2 = 0; r2 < r; r2++) {
            if (m.role_file[r2].found && m.role_file[r2].path == m.role_file[r].path) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            total += m.role_file[r].tensor_bytes;
        }
    }
    return total;
}

// ── Tensor loading ──────────────────────────────────────────────────────────

static bool mm3_load_lm_tensors(MM3Model * m, const GGUFModel & gf, std::vector<std::string> * errs) {
    const MM3LmConfig & c = m->lm_cfg;
    const int64_t       H = c.embedding_length;
    const int64_t       V = c.vocab_size;
    const int64_t       D = c.key_length;
    const int64_t       Q = (int64_t) c.head_count * D;
    const int64_t       K = (int64_t) c.head_count_kv * D;
    const int64_t       F = c.feed_forward_length;
    const int           L = (int) c.block_count;

    wctx_init(&m->wctx_lm, 3 + L * 11);
    MM3Loader ld{ &m->wctx_lm, &gf, &m->tmap_lm, errs };

    m->lm.token_embd  = ld.req("token_embd.weight", H, V);
    m->lm.output_norm = ld.req("output_norm.weight", H);
    m->lm.output      = ld.req("output.weight", H, V);

    m->lm.blk.assign((size_t) L, MM3LmLayer{});
    for (int i = 0; i < L; i++) {
        MM3LmLayer & b = m->lm.blk[(size_t) i];
        b.attn_norm    = ld.req(mm3_fmt("blk.%d.attn_norm.weight", i), H);
        b.attn_q       = ld.req(mm3_fmt("blk.%d.attn_q.weight", i), H, Q);
        b.attn_k       = ld.req(mm3_fmt("blk.%d.attn_k.weight", i), H, K);
        b.attn_v       = ld.req(mm3_fmt("blk.%d.attn_v.weight", i), H, K);
        b.attn_output  = ld.req(mm3_fmt("blk.%d.attn_output.weight", i), Q, H);
        b.attn_q_norm  = ld.req(mm3_fmt("blk.%d.attn_q_norm.weight", i), D);
        b.attn_k_norm  = ld.req(mm3_fmt("blk.%d.attn_k_norm.weight", i), D);
        b.ffn_norm     = ld.req(mm3_fmt("blk.%d.ffn_norm.weight", i), H);
        b.ffn_gate     = ld.req(mm3_fmt("blk.%d.ffn_gate.weight", i), H, F);
        b.ffn_up       = ld.req(mm3_fmt("blk.%d.ffn_up.weight", i), H, F);
        b.ffn_down     = ld.req(mm3_fmt("blk.%d.ffn_down.weight", i), F, H);
        if (!errs->empty()) {
            break;  // one bad layer means the file is wrong; don't spam 36 copies
        }
    }
    return errs->empty();
}

// The depth decoder gets its OWN backend buffer, separate from the rest of the
// synth stack, even though both come from the same GGUF.
//
// Why: depth runs INSIDE the AR loop (it consumes the LM's hidden state every
// frame), so stage 1 needs LM + depth. The condition encoder, flow DiT and
// vocoder are not touched until stage 2, by which time the LM is finished. One
// combined synth buffer forced all of it resident for the whole run, making the
// peak LM + everything. Splitting here lets stage 1 hold LM + depth (~0.65 GB)
// and stage 2 hold only cond/dit/voc — the peak drops by roughly the size of
// the flow stack. This is a load-time split only: no change to the GGUF, which
// is why it needs no re-conversion (see docs/plans/mm3-gguf-layout.md, which
// calls this out as the intended seam).
static bool mm3_load_depth_tensors(MM3Model * m, const GGUFModel & gf, std::vector<std::string> * errs) {
    const MM3SynthConfig & c = m->synth_cfg;

    // Tensor budget from the layout doc: depth 47.
    const int n_depth = 4 + (int) (c.depth.num_codebooks - 1) + (int) c.depth.block_count * 9;
    wctx_init(&m->wctx_depth, n_depth);

    MM3Loader ld{ &m->wctx_depth, &gf, &m->tmap_synth, errs };

    // ── depth ──
    {
        const MM3DepthConfig & d  = c.depth;
        const int64_t          H  = d.embedding_length;
        const int64_t          F  = d.feed_forward_length;
        const int              L  = (int) d.block_count;
        const int              NC = (int) d.num_codebooks - 1;

        m->synth.depth.proj        = ld.req("depth.proj.weight", H, H);
        m->synth.depth.pos_embd    = ld.req("depth.pos_embd.weight", H, d.max_position);
        m->synth.depth.audio_embd  = ld.req("depth.audio_embd.weight", H, d.audio_embd_rows);
        m->synth.depth.output_norm = ld.req("depth.output_norm.weight", H);

        m->synth.depth.head.assign((size_t) NC, nullptr);
        for (int i = 0; i < NC; i++) {
            m->synth.depth.head[(size_t) i] = ld.req(mm3_fmt("depth.head.%d.weight", i), H, d.audio_vocab_size);
        }

        m->synth.depth.blk.assign((size_t) L, MM3DepthLayer{});
        for (int i = 0; i < L && errs->empty(); i++) {
            MM3DepthLayer & b = m->synth.depth.blk[(size_t) i];
            b.attn_norm       = ld.req(mm3_fmt("depth.blk.%d.attn_norm.weight", i), H);
            b.attn_q          = ld.req(mm3_fmt("depth.blk.%d.attn_q.weight", i), H, H);
            b.attn_k          = ld.req(mm3_fmt("depth.blk.%d.attn_k.weight", i), H, H);
            b.attn_v          = ld.req(mm3_fmt("depth.blk.%d.attn_v.weight", i), H, H);
            b.attn_output     = ld.req(mm3_fmt("depth.blk.%d.attn_output.weight", i), H, H);
            b.ffn_norm        = ld.req(mm3_fmt("depth.blk.%d.ffn_norm.weight", i), H);
            b.ffn_gate        = ld.req(mm3_fmt("depth.blk.%d.ffn_gate.weight", i), H, F);
            b.ffn_up          = ld.req(mm3_fmt("depth.blk.%d.ffn_up.weight", i), H, F);
            b.ffn_down        = ld.req(mm3_fmt("depth.blk.%d.ffn_down.weight", i), F, H);
        }
    }

    return errs->empty();
}

// The stage-2 parts: condition encoder + flow DiT + vocoder, one shared
// backend buffer (they load and evict together), each read from its own
// resolved file. In legacy bundle mode all three GGUFModel refs alias the same
// open handle. Loaded separately from depth so it can be absent while the AR
// loop runs.
static bool mm3_load_rest_tensors(MM3Model * m, const GGUFModel & gf_cond, const GGUFModel & gf_dit,
                                  const GGUFModel & gf_voc, std::vector<std::string> * errs) {
    const MM3SynthConfig & c = m->synth_cfg;

    // Tensor budget from the layout doc: cond 4 + dit 370 + voc 91.
    const int n_cond = 4;
    const int n_dit  = 10 + (int) c.dit.block_count * 10;
    const int n_voc  = 7 + (int) c.voc.upsample_rates.size() * (3 + (int) c.voc.res_dilations.size() * 6);
    wctx_init(&m->wctx_synth, n_cond + n_dit + n_voc);

    MM3Loader ld{ &m->wctx_synth, &gf_cond, &m->tmap_synth, errs };

    // ── cond ──
    {
        const MM3CondConfig & e = c.cond;
        m->synth.cond.layer_logits = ld.req("cond.layer_logits", e.num_layers);
        m->synth.cond.layer_scale  = ld.req("cond.layer_scale", 1);
        m->synth.cond.proj_w       = ld.req("cond.proj.weight", e.kernel_size, e.hidden_dim, e.out_dim);
        m->synth.cond.proj_b       = ld.req("cond.proj.bias", e.out_dim);
    }

    // ── dit ──
    ld.gf = &gf_dit;
    {
        const MM3DitConfig & t  = c.dit;
        const int64_t        E  = t.embedding_length;
        const int64_t        CC = t.concat_channels;
        const int64_t        IC = t.in_channels;
        const int64_t        FI = t.ff_inner;
        const int            L  = (int) t.block_count;

        m->synth.dit.preprocess_conv  = ld.req("dit.preprocess_conv.weight", 1, CC, CC);
        m->synth.dit.postprocess_conv = ld.req("dit.postprocess_conv.weight", 1, IC, IC);
        m->synth.dit.time_fourier     = ld.req("dit.time_fourier.weight", 1, t.fourier_dim / 2);
        m->synth.dit.time_embd_w[0]   = ld.req("dit.time_embd.0.weight", t.fourier_dim, E);
        m->synth.dit.time_embd_b[0]   = ld.req("dit.time_embd.0.bias", E);
        m->synth.dit.time_embd_w[1]   = ld.req("dit.time_embd.1.weight", E, E);
        m->synth.dit.time_embd_b[1]   = ld.req("dit.time_embd.1.bias", E);
        m->synth.dit.proj_in          = ld.req("dit.proj_in.weight", CC, E);
        m->synth.dit.proj_out         = ld.req("dit.proj_out.weight", E, IC);
        m->synth.dit.rope_inv_freq    = ld.req("dit.rope_inv_freq", t.rope_dim / 2);

        m->synth.dit.blk.assign((size_t) L, MM3DitBlock{});
        for (int i = 0; i < L && errs->empty(); i++) {
            MM3DitBlock & b = m->synth.dit.blk[(size_t) i];
            b.attn_norm_w   = ld.req(mm3_fmt("dit.blk.%d.attn_norm.weight", i), E);
            b.attn_norm_b   = ld.req(mm3_fmt("dit.blk.%d.attn_norm.bias", i), E);
            b.attn_qkv      = ld.req(mm3_fmt("dit.blk.%d.attn_qkv.weight", i), E, E * 3);
            b.attn_output   = ld.req(mm3_fmt("dit.blk.%d.attn_output.weight", i), E, E);
            b.ffn_norm_w    = ld.req(mm3_fmt("dit.blk.%d.ffn_norm.weight", i), E);
            b.ffn_norm_b    = ld.req(mm3_fmt("dit.blk.%d.ffn_norm.bias", i), E);
            b.ffn_in_w      = ld.req(mm3_fmt("dit.blk.%d.ffn_in.weight", i), E, FI * 2);
            b.ffn_in_b      = ld.req(mm3_fmt("dit.blk.%d.ffn_in.bias", i), FI * 2);
            b.ffn_out_w     = ld.req(mm3_fmt("dit.blk.%d.ffn_out.weight", i), FI, E);
            b.ffn_out_b     = ld.req(mm3_fmt("dit.blk.%d.ffn_out.bias", i), E);
        }
    }

    // ── voc ── channel ladder hidden_dim >> B, strides from upsample_rates
    ld.gf = &gf_voc;
    {
        const MM3VocConfig & v  = c.voc;
        const int            NB = (int) v.upsample_rates.size();
        const int            NR = (int) v.res_dilations.size();

        m->synth.voc.dec_in_w  = ld.req("voc.dec_in.weight", 1, v.fold_channels, v.dec_in_dim);
        m->synth.voc.dec_in_b  = ld.req("voc.dec_in.bias", v.dec_in_dim);
        m->synth.voc.conv_in_w = ld.req("voc.conv_in.weight", 7, v.dec_in_dim, v.hidden_dim);
        m->synth.voc.conv_in_b = ld.req("voc.conv_in.bias", v.hidden_dim);

        m->synth.voc.blk.assign((size_t) NB, MM3VocBlock{});
        int64_t cin = (int64_t) v.hidden_dim;
        for (int b = 0; b < NB && errs->empty(); b++) {
            const int64_t cout   = cin / 2;
            const int64_t stride = v.upsample_rates[(size_t) b];
            MM3VocBlock & vb     = m->synth.voc.blk[(size_t) b];

            vb.snake_alpha = ld.req(mm3_fmt("voc.blk.%d.snake.alpha", b), 1, cin, 1);
            vb.convt_w     = ld.req(mm3_fmt("voc.blk.%d.convt.weight", b), stride * 2, cout, cin);
            vb.convt_b     = ld.req(mm3_fmt("voc.blk.%d.convt.bias", b), cout);

            vb.res.assign((size_t) NR, MM3VocResUnit{});
            for (int r = 0; r < NR && errs->empty(); r++) {
                MM3VocResUnit & ru = vb.res[(size_t) r];
                ru.snake1_alpha    = ld.req(mm3_fmt2("voc.blk.%d.res.%d.snake1.alpha", b, r), 1, cout, 1);
                ru.conv1_w         = ld.req(mm3_fmt2("voc.blk.%d.res.%d.conv1.weight", b, r), 7, cout, cout);
                ru.conv1_b         = ld.req(mm3_fmt2("voc.blk.%d.res.%d.conv1.bias", b, r), cout);
                ru.snake2_alpha    = ld.req(mm3_fmt2("voc.blk.%d.res.%d.snake2.alpha", b, r), 1, cout, 1);
                ru.conv2_w         = ld.req(mm3_fmt2("voc.blk.%d.res.%d.conv2.weight", b, r), 1, cout, cout);
                ru.conv2_b         = ld.req(mm3_fmt2("voc.blk.%d.res.%d.conv2.bias", b, r), cout);
            }
            cin = cout;
        }

        m->synth.voc.snake_out_alpha = ld.req("voc.snake_out.alpha", 1, cin, 1);
        m->synth.voc.conv_out_w      = ld.req("voc.conv_out.weight", 7, cin, 1);
        m->synth.voc.conv_out_b      = ld.req("voc.conv_out.bias", 1);
    }

    return errs->empty();
}

// ── Load / unload ───────────────────────────────────────────────────────────

// Weight bytes currently resident across all three parts. Excludes the AR KV
// cache, which is a separate allocation owned by the LM graph state.
static size_t mm3_vram_bytes(const MM3Model & m) {
    return m.vram_lm + m.vram_depth + m.vram_synth;
}

static void mm3_unload(MM3Model * m) {
    if (!m->loaded && !m->wctx_lm.ctx && !m->wctx_depth.ctx && !m->wctx_synth.ctx) {
        return;
    }
    wctx_free(&m->wctx_lm);
    wctx_free(&m->wctx_depth);
    wctx_free(&m->wctx_synth);
    m->lm    = MM3LmWeights{};
    m->synth = MM3SynthWeights{};
    m->tmap_lm.clear();
    m->tmap_synth.clear();
    m->vram_lm        = 0;
    m->vram_depth     = 0;
    m->vram_synth     = 0;
    m->load_ms        = 0.0;
    m->loaded         = false;
    m->lm_resident    = false;
    m->depth_resident = false;
    m->rest_resident  = false;
    m->lm_merge_tag.clear();
    if (m->backend_ref) {
        backend_release(m->backend, m->cpu_backend);
        m->backend     = nullptr;
        m->cpu_backend = nullptr;
        m->backend_ref = false;
    }
    fprintf(stderr, "[MM3] Unloaded\n");
}

// Forward decls for the targeted residency drops select uses.
static size_t mm3_free_lm(MM3Model * m);
static void   mm3_free_depth(MM3Model * m);
static void   mm3_free_rest(MM3Model * m);

// Choose which quant of each role to run. Empty string = auto (best-first);
// unchanged values are a no-op so the UI can post the full selection every time.
// `role_quants` is indexed by MM3SynthRole; null means "leave all four alone".
//
// Selecting a different file must drop the residency it feeds — but ONLY that:
// swapping the DiT quant frees cond+dit+voc (~300 MB of company) and leaves the
// 17 GB LM warm. The tokenizer caches by LM path (mm3-tokenizer.h reloads when
// the path changes). Callers must hold the MM3 mutex — this unloads.
static bool mm3_select_variant(MM3Model * m, const std::string & lm_quant, const std::string * role_quants,
                               std::string * err_out) {
    auto known = [](const std::vector<MM3Variant> & vars, const std::string & q) {
        if (q.empty()) {
            return true;   // auto
        }
        for (const auto & v : vars) {
            if (v.quant == q) {
                return true;
            }
        }
        return false;
    };
    if (!known(m->lm_variants, lm_quant)) {
        if (err_out) {
            *err_out = "no mm3-lm-" + lm_quant + ".gguf on disk";
        }
        return false;
    }
    for (int r = 0; r < MM3_R_COUNT; r++) {
        if (role_quants && !known(m->role_variants[r], role_quants[r])) {
            if (err_out) {
                *err_out = "no mm3-" + std::string(MM3_SYNTH_ROLE[r]) + "-" + role_quants[r] +
                           ".gguf (or matching bundle) on disk";
            }
            return false;
        }
    }

    const bool lm_changed = lm_quant != m->want_lm_quant;
    bool       role_changed[MM3_R_COUNT] = { false, false, false, false };
    bool       any_role_changed          = false;
    for (int r = 0; r < MM3_R_COUNT; r++) {
        role_changed[r] = role_quants && role_quants[r] != m->want_role_quant[r];
        any_role_changed = any_role_changed || role_changed[r];
    }
    if (!lm_changed && !any_role_changed) {
        return true;
    }

    m->want_lm_quant = lm_quant;
    if (role_quants) {
        for (int r = 0; r < MM3_R_COUNT; r++) {
            m->want_role_quant[r] = role_quants[r];
        }
    }

    // Drop only the parts whose weights are the outgoing files. In bundle mode
    // a change to any bundled role swaps the file under every role it serves,
    // so the depth/rest distinction still holds: depth rides wctx_depth, the
    // other three ride wctx_synth.
    if (lm_changed) {
        mm3_free_lm(m);
    }
    if (role_changed[MM3_R_DEPTH]) {
        mm3_free_depth(m);
    }
    if (role_changed[MM3_R_COND] || role_changed[MM3_R_DIT] || role_changed[MM3_R_VOC]) {
        mm3_free_rest(m);
    }

    const std::string dir = m->models_dir;
    mm3_discover(m, dir.c_str(), m->want_lm_quant, m->want_role_quant);
    fprintf(stderr, "[MM3] Selected quants: lm=%s depth=%s cond=%s dit=%s voc=%s\n",
            m->lm_file.found ? m->lm_file.name.c_str() : "(none)",
            m->role_file[MM3_R_DEPTH].found ? m->role_file[MM3_R_DEPTH].name.c_str() : "(none)",
            m->role_file[MM3_R_COND].found ? m->role_file[MM3_R_COND].name.c_str() : "(none)",
            m->role_file[MM3_R_DIT].found ? m->role_file[MM3_R_DIT].name.c_str() : "(none)",
            m->role_file[MM3_R_VOC].found ? m->role_file[MM3_R_VOC].name.c_str() : "(none)");
    return true;
}

// Merge any requested LoRAs into the freshly-staged DiT weights.
//
// INCREMENT 1a — the adapter set comes from the environment, not the request:
//
//   MM3_ADAPTER        one path, or several separated by ';' (Windows) or ':'
//   MM3_ADAPTER_SCALE  strength, default 1.0, applied to every adapter
//
// This is deliberately a bring-up surface, not the final one: an env var is
// fixed for the process lifetime, so it sidesteps the reload-on-change question
// entirely while the merge maths is being validated. Increment 1b moves this to
// a request field and reloads `rest` when the set changes (rest_adapter_desc is
// the comparison key it will use).
// `gf_dit` is the file carrying the dit.* tensors (the split DiT file, or the
// legacy bundle) — the only file the adapter merge needs.
static void mm3_apply_adapters(MM3Model * m, const GGUFModel & gf_sy) {
    m->rest_adapter_desc.clear();

    const char * spec = getenv("MM3_ADAPTER");
    if (!spec || !*spec) {
        return;
    }
    float        scale     = 1.0f;
    const char * scale_env = getenv("MM3_ADAPTER_SCALE");
    if (scale_env && *scale_env) {
        scale = (float) atof(scale_env);
    }

    std::vector<std::string> paths;
    {
        std::string s = spec;
        size_t      start = 0;
        while (start <= s.size()) {
            size_t sep = s.find_first_of(";:", start);
            // Don't split a Windows drive letter ("D:\...").
            if (sep != std::string::npos && s[sep] == ':' && sep == start + 1 && isalpha((unsigned char) s[start])) {
                sep = s.find_first_of(";:", sep + 1);
            }
            std::string one = s.substr(start, sep == std::string::npos ? std::string::npos : sep - start);
            if (!one.empty()) {
                paths.push_back(one);
            }
            if (sep == std::string::npos) {
                break;
            }
            start = sep + 1;
        }
    }

    int total = 0;
    for (const std::string & p : paths) {
        const int n = mm3_adapter_merge(&m->wctx_synth, gf_sy, p.c_str(), scale, m->backend);
        if (n > 0) {
            total += n;
            if (!m->rest_adapter_desc.empty()) {
                m->rest_adapter_desc += "; ";
            }
            m->rest_adapter_desc += p + "@" + std::to_string(scale);
        } else {
            fprintf(stderr, "[MM3-Adapter] WARNING: %s merged nothing — the DiT is unchanged by it\n", p.c_str());
        }
    }
    if (total) {
        fprintf(stderr, "[MM3-Adapter] %d tensor(s) patched across %zu adapter(s)\n", total, paths.size());
    }
}

// Load a chosen subset of the three parts into backend buffers. Parts already
// resident are left alone, so this is idempotent and safe to call repeatedly.
// On any failure NOTHING stays resident — a half-loaded model is never a state
// the pipeline has to reason about.
//
// Files are opened only when a part that lives in them is actually wanted, so
// a stage-2 top-up never re-reads the 17 GB LM header. Distinct paths are
// opened once: in legacy bundle mode depth/cond/dit/voc all resolve to the
// same GGUF and share one handle.
static bool mm3_load_parts(MM3Model * m, bool want_lm, bool want_depth, bool want_rest, std::string * err_out) {
    const bool need_lm    = want_lm && !m->lm_resident;
    const bool need_depth = want_depth && !m->depth_resident;
    const bool need_rest  = want_rest && !m->rest_resident;
    if (!need_lm && !need_depth && !need_rest) {
        return true;
    }
    {
        bool all_found = m->lm_file.found;
        for (int r = 0; r < MM3_R_COUNT; r++) {
            all_found = all_found && m->role_file[r].found;
        }
        if (!all_found) {
            if (err_out) {
                *err_out = "MiniMax-Music3 GGUFs not found (need mm3-lm plus depth/cond/dit/voc files or an "
                           "mm3-synth bundle)";
            }
            return false;
        }
    }
    if (!mm3_available(*m)) {
        if (err_out) {
            *err_out = m->meta_errors.empty() ? "MiniMax-Music3 GGUF metadata probe failed; see /mm3/props errors"
                                              : m->meta_errors[0];
        }
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();

    // The backend reference is shared by all three parts and taken once.
    if (!m->backend_ref) {
        BackendPair bp = backend_init("MM3");
        m->backend     = bp.backend;
        m->cpu_backend = bp.cpu_backend;
        m->backend_ref = true;
    }

    std::vector<std::string> errs;
    bool                     ok = true;

    // One open handle per distinct path (map nodes are pointer-stable).
    std::map<std::string, GGUFModel> handles;
    auto open = [&](const std::string & path) -> const GGUFModel * {
        auto it = handles.find(path);
        if (it != handles.end()) {
            return &it->second;
        }
        GGUFModel gf = {};
        if (!gf_load(&gf, path.c_str())) {
            errs.push_back("cannot open " + path);
            return nullptr;
        }
        return &handles.emplace(path, gf).first->second;
    };
    auto close_all = [&]() {
        for (auto & kv : handles) {
            gf_close(&kv.second);
        }
        handles.clear();
    };

    if (need_lm) {
        const GGUFModel * gf_lm = open(m->lm_file.path);
        ok                      = gf_lm != nullptr;
        if (ok) {
            ok = mm3_load_lm_tensors(m, *gf_lm, &errs);
        }
        if (ok) {
            ok = wctx_alloc(&m->wctx_lm, m->backend);
            if (!ok) {
                errs.push_back("backend buffer allocation failed for the LM (out of VRAM?)");
            }
        }
    }

    if (ok && need_depth) {
        const GGUFModel * gf_depth = open(m->role_file[MM3_R_DEPTH].path);
        ok                         = gf_depth != nullptr;
        if (ok) {
            ok = mm3_load_depth_tensors(m, *gf_depth, &errs);
        }
        if (ok) {
            ok = wctx_alloc(&m->wctx_depth, m->backend);
            if (!ok) {
                errs.push_back("backend buffer allocation failed for the depth decoder (out of VRAM?)");
            }
        }
    }

    if (ok && need_rest) {
        const GGUFModel * gf_cond = open(m->role_file[MM3_R_COND].path);
        const GGUFModel * gf_dit  = gf_cond ? open(m->role_file[MM3_R_DIT].path) : nullptr;
        const GGUFModel * gf_voc  = gf_dit ? open(m->role_file[MM3_R_VOC].path) : nullptr;
        ok                        = gf_voc != nullptr;
        if (ok) {
            ok = mm3_load_rest_tensors(m, *gf_cond, *gf_dit, *gf_voc, &errs);
        }
        if (ok) {
            // Adapter merge goes HERE — after the tensors are staged in
            // wctx_synth but before they are uploaded, the same seam ACE
            // uses in dit.h. Merging patches the PendingCopy sources, so
            // wctx_alloc then uploads already-adapted weights and no
            // extra VRAM or second pass is needed. It reads the DiT's file.
            mm3_apply_adapters(m, *gf_dit);
            ok = wctx_alloc(&m->wctx_synth, m->backend);
            if (!ok) {
                errs.push_back("backend buffer allocation failed for the synth stack (out of VRAM?)");
            }
        }
    }
    close_all();

    if (!ok) {
        std::string msg = errs.empty() ? "MM3 load failed" : errs[0];
        for (size_t i = 1; i < errs.size() && i < 6; i++) {
            msg += "; " + errs[i];
        }
        fprintf(stderr, "[MM3] LOAD FAILED: %s\n", msg.c_str());
        if (err_out) {
            *err_out = msg;
        }
        mm3_unload(m);
        return false;
    }

    if (need_lm) {
        m->vram_lm     = m->wctx_lm.buffer ? ggml_backend_buffer_get_size(m->wctx_lm.buffer) : 0;
        m->lm_resident = true;
    }
    if (need_depth) {
        m->vram_depth     = m->wctx_depth.buffer ? ggml_backend_buffer_get_size(m->wctx_depth.buffer) : 0;
        m->depth_resident = true;
    }
    if (need_rest) {
        m->vram_synth    = m->wctx_synth.buffer ? ggml_backend_buffer_get_size(m->wctx_synth.buffer) : 0;
        m->rest_resident = true;
    }
    m->loaded  = m->lm_resident && m->depth_resident && m->rest_resident;
    m->load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    fprintf(stderr, "[MM3] Loaded%s%s%s: LM %.2f + depth %.2f + synth %.2f = %.2f GB in %.0f ms\n",
            need_lm ? " lm" : "", need_depth ? " depth" : "", need_rest ? " cond/dit/voc" : "",
            (double) m->vram_lm / 1073741824.0, (double) m->vram_depth / 1073741824.0,
            (double) m->vram_synth / 1073741824.0, (double) mm3_vram_bytes(*m) / 1073741824.0, m->load_ms);
    return true;
}

// Load everything. The contract every existing caller (POST /mm3/warm, the
// bring-up endpoints) was written against.
static bool mm3_load(MM3Model * m, std::string * err_out) {
    return mm3_load_parts(m, true, true, true, err_out);
}

// Drop ONLY the global LM (the 8.59B half). Valid the moment the AR loop is
// done: its output is a CPU-side hidden-state block, and nothing downstream
// reads LM weights again. Callers must have already torn down the LM graph
// state (mm3_lm_free), which holds pointers into this buffer.
static size_t mm3_free_lm(MM3Model * m) {
    if (!m->lm_resident) {
        return 0;
    }
    const size_t freed = m->vram_lm;
    wctx_free(&m->wctx_lm);
    m->lm = MM3LmWeights{};
    m->tmap_lm.clear();
    m->vram_lm     = 0;
    m->lm_resident = false;
    m->loaded      = false;
    m->lm_merge_tag.clear();
    return freed;
}

// Drop ONLY the depth decoder. Used by a per-role quant swap; the LM graph
// reads depth.audio_embd out of this buffer, but its prepare keys on the
// buffer identity and rebuilds on the next call. tmap_synth entries pointing
// into this buffer die with it — the map is introspection-only and rebuilt on
// the next load.
static void mm3_free_depth(MM3Model * m) {
    if (!m->depth_resident) {
        return;
    }
    wctx_free(&m->wctx_depth);
    m->synth.depth = MM3DepthWeights{};
    m->tmap_synth.clear();
    m->vram_depth     = 0;
    m->depth_resident = false;
    m->loaded         = false;
}

// Drop ONLY cond+dit+voc (the stage-2 buffer). The cheap reload path behind a
// DiT quant or adapter change: LM and depth stay warm.
static void mm3_free_rest(MM3Model * m) {
    if (!m->rest_resident) {
        return;
    }
    wctx_free(&m->wctx_synth);
    m->synth.cond = MM3CondWeights{};
    m->synth.dit  = MM3DitWeights{};
    m->synth.voc  = MM3VocWeights{};
    m->tmap_synth.clear();
    m->rest_adapter_desc.clear();
    m->vram_synth    = 0;
    m->rest_resident = false;
    m->loaded        = false;
}
