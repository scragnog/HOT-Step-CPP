#pragma once
// yue2/yue2-model.h — YuE2 GGUF loader, config parse and VRAM residency.
//
// HOT-Step file (does not exist upstream, no acestep.cpp analog). Nothing
// outside engine/src/yue2/ may include this in this milestone — hot-step-server.cpp
// is NOT touched (see docs/plans/yue2/06-engine-port-plan.md milestone M0).
//
// SCOPE OF THIS FILE (milestone M0): open the LM GGUF (arch "yue2") and a VAE
// GGUF (arch "yue2-vae", "standard" or "legacy" variant), parse every yue2.*/
// yue2vae.* KV into typed config structs, validate every tensor's name AND
// config-derived shape, put the weights in a backend buffer, and account the
// VRAM. There are NO compute graphs and NO inference here.
//
// Contract consumed: docs/plans/yue2/05-gguf-layout.md (tensor names/shapes),
// docs/plans/yue2/06-engine-port-plan.md §1 (module map), produced by
// engine/tools/convert-yue2.py. All tensor names/shapes/KV keys below were
// additionally cross-checked directly against the real converted files
// (models/yue2/yue2-lm-bf16.gguf, yue2-vae-{standard,legacy}-f32.gguf) with
// the `gguf` python package, since 05-gguf-layout.md's own prose tensor-count
// claims (658 LM / 435 VAE) do not match what the converter actually wrote
// (628 LM / 347 VAE per file — the real files are ground truth here).
//
// Two arches, per 05-gguf-layout.md §2:
//   yue2-lm-<type>.gguf                  arch "yue2"       628 tensors
//   yue2-vae-{standard,legacy}-<type>.gguf arch "yue2-vae" 347 tensors
//     (347 = 217 decoder + 130 encoder after weight-norm folding; both trees
//      are present in the file, but the loader only reads dec.* by default —
//      decoder_only, mirroring the reference pipeline and MM3's own vocoder)
//
// Residency: TWO parts, not three — per docs/plans/yue2/06-engine-port-plan.md
// §0, the AR and NAR halves must be resident together for the whole synthesis
// stage (no MM3-shaped "stage 1 frees this, stage 2 frees that" split), so
// `lm_resident` covers AR+NAR+flow heads all-or-nothing. `vae_resident` covers
// whichever variant (standard/legacy) is currently loaded — the two variants
// are alternatives, never both resident at once.

// <string> MUST precede backend.h: backend.h's Windows CUDA-DLL diagnostic
// path uses std::wstring but does not include <string> itself (it normally
// arrives transitively from whatever this TU's other headers pulled in
// first — not guaranteed here, since yue2-probe.cpp's first include is this
// file).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"
#include "yue2-adapter.h"  // NAR LoRA merge-at-load (Yue2AdapterSpec lives there)

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dirent.h>
#endif

// ── Config structs ──────────────────────────────────────────────────────────

// yue2-lm-<type>.gguf: AR+NAR backbone + flow-matching heads.
struct Yue2LmConfig {
    uint32_t context_length      = 0;  // yue2.context_length (24576)
    uint32_t embedding_length    = 0;  // yue2.embedding_length (2048)
    uint32_t block_count         = 0;  // yue2.block_count (28)
    uint32_t feed_forward_length = 0;  // yue2.feed_forward_length (6144)
    uint32_t head_count          = 0;  // yue2.attention.head_count (16)
    uint32_t head_count_kv       = 0;  // yue2.attention.head_count_kv (8)
    uint32_t key_length          = 0;  // yue2.attention.key_length (128)
    uint32_t value_length        = 0;  // yue2.attention.value_length (128)
    float    rms_eps             = 0.0f;  // yue2.attention.layer_norm_rms_epsilon (1e-6)
    float    rope_freq_base      = 0.0f;  // yue2.rope.freq_base (1e6)
    uint32_t vocab_size          = 0;  // yue2.vocab_size (184704)
    uint32_t latent_dim          = 0;  // yue2.latent_dim (64)
    uint32_t max_latent_frames   = 0;  // yue2.max_latent_frames (24576)
    float    timestep_shift      = 0.0f;  // yue2.timestep_shift (general formula, don't hardcode sigmoid)
    float    softmax_scale       = 0.0f;  // yue2.attention.softmax_scale (128^-0.5)

    // yue2.token.* — special/structural ids, spliced as raw ints, never
    // round-tripped through the BPE encoder/decoder for music_start/music_end.
    uint32_t tok_eod          = 0;  // 151643 — also BOS and PAD
    uint32_t tok_abc_start    = 0;  // 151847
    uint32_t tok_abc_end      = 0;  // 151848 — ABC stage EOS
    uint32_t tok_music_start  = 0;  // 151851
    uint32_t tok_music_end    = 0;  // 151852 — semantic stage EOS
    uint32_t tok_codec_offset = 0;  // 151853
    uint32_t tok_codec_size   = 0;  // 32768
    uint32_t tok_latent_start = 0;  // 184621 — training-time only, never in live stream
    uint32_t tok_latent_end   = 0;  // 184622
    uint32_t tok_latent_pad   = 0;  // 184623

    // yue2.sampling.{abc,semantic}.* — checkpoint-fixed per-stage defaults.
    struct Stage {
        float    temperature        = 0.0f;
        float    top_p              = 0.0f;
        uint32_t top_k               = 0;
        float    repetition_penalty = 0.0f;
        uint32_t penalty_window      = 0;
        uint32_t min_tokens          = 0;
        uint32_t max_tokens          = 0;
    };
    Stage abc;
    Stage semantic;

    uint32_t    ode_steps = 0;  // yue2.ode.steps (32)
    std::string ode_method;     // yue2.ode.method ("midpoint")
};

// yue2-vae-<variant>-<type>.gguf: Oobleck decoder (+ optional encoder).
struct Yue2VaeConfig {
    std::string variant;  // yue2vae.variant: "standard" | "legacy"
    uint32_t    sample_rate         = 0;  // 48000
    uint32_t    downsampling_ratio  = 0;  // 1920 (= product of strides)
    uint32_t    audio_channels      = 0;  // 2
    uint32_t    latent_dim          = 0;  // 64 (decoder input width)
    uint32_t    encoder_latent_dim  = 0;  // 128 (encoder output width, pre mean||scale split)
    uint32_t    channels            = 0;  // 64 (base channel count both sides)
    std::vector<int32_t> strides;         // [2,2,4,4,5,6], encoder forward order
    std::vector<int32_t> res_dilations;   // [1,3,9]
    bool        use_snake           = false;
    std::string snake_type;               // "vanilla" (SnakeBeta)
    float       snake_eps            = 0.0f;  // 1e-9
    std::string snake_formula;
    bool        final_tanh           = false;  // false — decoder's last module is Identity
    float       output_clamp_min     = 0.0f;   // -1.0, pipeline-level, informational
    float       output_clamp_max     = 0.0f;   // 1.0
    bool        weight_norm_folded   = false;  // true — GGUF carries no weight_g/weight_v
    bool        has_encoder          = false;  // true for both shipped VAE checkpoints
    uint32_t    decode_core_frames    = 0;  // 1024
    uint32_t    decode_halo_frames    = 0;  // 16
    uint32_t    required_halo         = 0;  // 12 (16 has 4 frames of margin)
};

// ── Weight structs — one ggml_tensor* per GGUF tensor ───────────────────────

// AR half of a decoder layer (blk.N.*).
struct Yue2LmLayer {
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

// NAR half of the SAME decoder layer (blk.N.nar_*) — full-rank, independently
// trained, zero weight sharing with the AR half (Mixture-of-Transformers).
struct Yue2NarLayer {
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

struct Yue2LmWeights {
    ggml_tensor * token_embd  = nullptr;  // [H,V]
    ggml_tensor * output_norm = nullptr;  // [H] — shared AR/NAR final norm ("model.norm")
    ggml_tensor * output      = nullptr;  // [H,V] — tie_word_embeddings=false, distinct from token_embd

    std::vector<Yue2LmLayer>  blk;      // AR path, block_count entries
    std::vector<Yue2NarLayer> nar_blk;  // NAR path, block_count entries

    // Flow-matching heads (flat names, no blk. prefix, one instance each).
    ggml_tensor * vae2llm_w = nullptr;  // [latent_dim,H]
    ggml_tensor * vae2llm_b = nullptr;  // [H]
    ggml_tensor * llm2vae_w = nullptr;  // [H,latent_dim]
    ggml_tensor * llm2vae_b = nullptr;  // [latent_dim]

    ggml_tensor * time_embd_w[2] = { nullptr, nullptr };  // mlp.0: [256,H], mlp.2: [H,H]
    ggml_tensor * time_embd_b[2] = { nullptr, nullptr };  // [H], [H]

    // AudioPositionEmbedding.pe, [H, max_latent_frames] in GGUF ne order.
    // TRAP (05-gguf-layout.md §3.3): loaded VERBATIM, never recomputed from
    // the textbook sinusoid formula — recompute-then-round mismatches the
    // checkpoint's actual stored values in 62,153/50,331,648 entries. Must
    // stay whatever native (non-F32-promoted) type the file stores it as;
    // yue2_load_lm_tensors asserts this against token_embd's own type.
    ggml_tensor * latent_pos_embed = nullptr;
};

// One SnakeBeta + Conv1d residual unit (dilations 1/3/9 across R=0..2).
struct Yue2VaeResUnit {
    ggml_tensor * snake1_alpha = nullptr;
    ggml_tensor * snake1_beta  = nullptr;
    ggml_tensor * conv1_w      = nullptr;
    ggml_tensor * conv1_b      = nullptr;
    ggml_tensor * snake2_alpha = nullptr;
    ggml_tensor * snake2_beta  = nullptr;
    ggml_tensor * conv2_w      = nullptr;
    ggml_tensor * conv2_b      = nullptr;
};

// Decoder upsample block: SnakeBeta -> ConvTranspose1d -> 3 residual units.
struct Yue2VaeDecBlock {
    ggml_tensor * snake_pre_alpha = nullptr;
    ggml_tensor * snake_pre_beta  = nullptr;
    ggml_tensor * upsample_w      = nullptr;  // [k=2*stride, Cout, Cin]
    ggml_tensor * upsample_b      = nullptr;  // [Cout]
    std::vector<Yue2VaeResUnit> res;          // 3 entries, dilations 1/3/9
};

// Encoder downsample block: 3 residual units -> SnakeBeta -> Conv1d (strided).
struct Yue2VaeEncBlock {
    std::vector<Yue2VaeResUnit> res;  // 3 entries, dilations 1/3/9
    ggml_tensor * snake_post_alpha = nullptr;
    ggml_tensor * snake_post_beta  = nullptr;
    ggml_tensor * downsample_w     = nullptr;  // [k=2*stride, Cin, Cout]
    ggml_tensor * downsample_b     = nullptr;  // [Cout]
};

struct Yue2VaeWeights {
    // Decoder — always loaded, this is what generation actually runs.
    ggml_tensor * dec_conv_in_w = nullptr;  // [7, latent_dim, C0]
    ggml_tensor * dec_conv_in_b = nullptr;  // [C0]
    std::vector<Yue2VaeDecBlock> dec_blk;   // 6 entries, strides.size()
    ggml_tensor * dec_snake_out_alpha = nullptr;
    ggml_tensor * dec_snake_out_beta  = nullptr;
    ggml_tensor * dec_conv_out_w      = nullptr;  // [7, C_last, audio_channels] — NO BIAS

    // Encoder — present in every shipped checkpoint (has_encoder=true) but
    // decoder_only by default, per the reference pipeline and MM3's own
    // vocoder-has-no-encoder posture. Only populated if yue2_load_parts is
    // called with want_encoder=true.
    bool          enc_loaded         = false;
    ggml_tensor * enc_conv_in_w      = nullptr;
    ggml_tensor * enc_conv_in_b      = nullptr;
    std::vector<Yue2VaeEncBlock> enc_blk;
    ggml_tensor * enc_snake_out_alpha = nullptr;
    ggml_tensor * enc_snake_out_beta  = nullptr;
    ggml_tensor * enc_conv_out_w      = nullptr;
    ggml_tensor * enc_conv_out_b      = nullptr;
};

// ── File-level metadata, filled by a header-only probe (no weights loaded) ──

struct Yue2FileInfo {
    bool        found = false;
    std::string path;
    std::string name;    // basename
    std::string arch;    // general.architecture ("yue2" | "yue2-vae")
    std::string license;
    uint32_t    file_type    = 0;  // general.file_type
    uint64_t    file_bytes   = 0;
    uint64_t    tensor_bytes = 0;  // sum of ggml_nbytes over the header
    int         n_tensors    = 0;
    bool        probe_ok     = false;
    std::string probe_error;
};

// One yue2-lm-<type>.gguf found on disk, for the props quant catalogue.
// Mirrors MM3Variant (engine/src/minimax/mm3-model.h) exactly — the type
// token is taken from the filename verbatim, same as MM3's `quant`.
struct Yue2Variant {
    std::string type;   // "bf16" | "Q8_0" | "Q4_K_M-imat" | ...
    std::string path;
    std::string name;   // basename
    uint64_t    bytes = 0;
};

enum Yue2VaeVariant { YUE2_VAE_STANDARD = 0, YUE2_VAE_LEGACY = 1, YUE2_VAE_VARIANT_COUNT };

static const char * const YUE2_VAE_VARIANT_NAME[YUE2_VAE_VARIANT_COUNT] = { "standard", "legacy" };

// ── The model ────────────────────────────────────────────────────────────────

struct Yue2Model {
    // discovery + probe (cheap, header-only)
    std::string               models_dir;
    std::vector<std::string>  search_dirs;
    Yue2FileInfo               lm_file;
    Yue2FileInfo               vae_file[YUE2_VAE_VARIANT_COUNT];
    Yue2LmConfig               lm_cfg;
    Yue2VaeConfig              vae_cfg;  // config of whichever variant is currently loaded/probed
    std::vector<std::string>  meta_errors;
    // Every yue2-lm-*.gguf found across search_dirs (filename-only scan, no
    // header probe) — reported at GET /yue2/props under variants.lm.available
    // so the UI's quant picker can list installed LM types. Filled by
    // yue2_discover(), same lifecycle as lm_file.
    std::vector<Yue2Variant>  lm_variants;

    // residency — two parts only, per file-header note above.
    bool           lm_resident  = false;
    bool           vae_resident = false;
    Yue2VaeVariant vae_loaded_variant = YUE2_VAE_STANDARD;
    // "" = auto/best-first (yue2_quant_rank order); otherwise pins discovery
    // to yue2-lm-<lm_type_want>.gguf. Set by POST /yue2/select-model; a
    // change forces yue2_unload() + re-discover so the next warm/synth picks
    // it up (mirrors mm3-server.h's want_lm_quant contract).
    std::string    lm_type_want;

    // NAR LoRA adapters to merge into the LM's weights at load time
    // (yue2-adapter.h). Empty = pristine base, which is what every caller that
    // never touches this field gets — and there are several: yue2-probe.cpp
    // and train/yue2-nar-train-run.h call yue2_load_parts on their OWN local
    // Yue2Model, not on g_yue2, so the selection has to live on the model
    // rather than in a global or an env var (MM3 reads getenv("MM3_ADAPTER")
    // and gets away with it only because its model is effectively the one
    // global). Set by POST /yue2/select-model.
    //
    // COMPOUNDING: a merge bakes the delta into the resident weights, so
    // merging twice would double it. The only thing that prevents that is that
    // the merge runs exclusively inside yue2_load_parts's `need_lm` branch,
    // and `need_lm` is false whenever the LM is already resident — so a
    // changed adapter set MUST go through yue2_unload() first. That is the
    // same full-teardown contract lm_type_want already relies on, and
    // yue2_handle_select_model enforces it for both.
    std::vector<Yue2AdapterSpec> lm_adapter_want;
    // What is actually merged into the RESIDENT weights, "path@scale; ..." as
    // rendered by yue2_adapter_key(), plus how many tensors it patched. Echoed
    // by /yue2/props: without it there is no way to tell from outside whether
    // the loaded model is adapted, which is precisely the state a debug
    // surface must not hide (MM3 keeps rest_adapter_desc for the same reason).
    std::string    lm_adapter_desc;
    int            lm_adapter_tensors = 0;
    // Which half the merged adapters landed on: "ar", "nar", or "ar+nar" when a
    // stack covers both (AR and NAR are a Mixture of Transformers with zero
    // weight sharing, so that stack is legal and disjoint). A bare tensor count
    // reads identically whichever half it patched, and which half it patched is
    // precisely what loading the wrong file gets wrong — so /yue2/props reports
    // this next to the count (contract 14 §7.3).
    std::string    lm_adapter_family;

    bool           backend_ref = false;
    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;

    WeightCtx wctx_lm  = {};
    WeightCtx wctx_vae = {};

    size_t vram_lm  = 0;
    size_t vram_vae = 0;
    double load_ms  = 0.0;

    Yue2LmWeights  lm;
    Yue2VaeWeights vae;

    // name -> tensor, per file. Introspection only.
    std::map<std::string, ggml_tensor *> tmap_lm;
    std::map<std::string, ggml_tensor *> tmap_vae;
};

// ── Small helpers ────────────────────────────────────────────────────────────

#ifdef _WIN32
#    define YUE2_SEP "\\"
#else
#    define YUE2_SEP "/"
#endif

static bool yue2_file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

static std::string yue2_basename(const std::string & path) {
    size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? path : path.substr(p + 1);
}

static uint64_t yue2_file_size(const std::string & path) {
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

// List plain files in a directory. Deliberately local, mirrors mm3_list_dir.
static void yue2_list_dir(const std::string & dir, std::vector<std::string> * names) {
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

static std::vector<int32_t> yue2_get_i32_arr(const GGUFModel & gf, const char * key) {
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

static std::string yue2_fmt(const char * fmt, int a) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, a);
    return buf;
}

static std::string yue2_fmt2(const char * fmt, int a, int b) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, a, b);
    return buf;
}

// Loader with config-derived shape validation — same contract as MM3Loader
// (engine/src/minimax/mm3-model.h): every tensor fetched by exact name,
// checked against the shape the KV metadata implies. Never exits the
// process — errors accumulate and the caller aborts the load cleanly.
struct Yue2Loader {
    WeightCtx *                            wctx   = nullptr;
    const GGUFModel *                      gf     = nullptr;
    std::map<std::string, ggml_tensor *> * tmap   = nullptr;
    std::vector<std::string> *             errors = nullptr;
    // Promote every loaded tensor to F32 at load time (gf_load_tensor_f32)
    // instead of keeping the file's stored type. Off for the LM and the VAE,
    // which want the quantized/F16 weights they were converted to. ON for
    // yue2-tok's head (yue2-tok-head.h): 42.8 M parameters is 171 MB at F32,
    // and an F32 src0 is the one ggml_mul_mat path that lands on a plain
    // cublasSgemm with no F16 round trip — which is what the FP32 stage
    // fixtures were captured against. It does NOT undo the F16 STORAGE
    // rounding already baked into a --outtype f16 file; it only stops a
    // second narrowing happening inside the matmul.
    bool                                   force_f32 = false;

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
        // safe: presence verified above
        ggml_tensor * t = force_f32 ? gf_load_tensor_f32(wctx, *gf, name) : gf_load_tensor(wctx, *gf, name);
        if (t && tmap) {
            (*tmap)[name] = t;
        }
        return t;
    }
};

// ── Config parsing ───────────────────────────────────────────────────────────

static void yue2_parse_lm_config(const GGUFModel & gf, Yue2LmConfig * c) {
    c->context_length      = gf_get_u32(gf, "yue2.context_length");
    c->embedding_length    = gf_get_u32(gf, "yue2.embedding_length");
    c->block_count         = gf_get_u32(gf, "yue2.block_count");
    c->feed_forward_length = gf_get_u32(gf, "yue2.feed_forward_length");
    c->head_count          = gf_get_u32(gf, "yue2.attention.head_count");
    c->head_count_kv       = gf_get_u32(gf, "yue2.attention.head_count_kv");
    c->key_length          = gf_get_u32(gf, "yue2.attention.key_length");
    c->value_length        = gf_get_u32(gf, "yue2.attention.value_length");
    c->rms_eps             = gf_get_f32(gf, "yue2.attention.layer_norm_rms_epsilon");
    c->rope_freq_base      = gf_get_f32(gf, "yue2.rope.freq_base");
    c->vocab_size          = gf_get_u32(gf, "yue2.vocab_size");
    c->latent_dim          = gf_get_u32(gf, "yue2.latent_dim");
    c->max_latent_frames   = gf_get_u32(gf, "yue2.max_latent_frames");
    c->timestep_shift      = gf_get_f32(gf, "yue2.timestep_shift");
    c->softmax_scale       = gf_get_f32(gf, "yue2.attention.softmax_scale");

    c->tok_eod          = gf_get_u32(gf, "yue2.token.eod");
    c->tok_abc_start    = gf_get_u32(gf, "yue2.token.abc_start");
    c->tok_abc_end      = gf_get_u32(gf, "yue2.token.abc_end");
    c->tok_music_start  = gf_get_u32(gf, "yue2.token.music_start");
    c->tok_music_end    = gf_get_u32(gf, "yue2.token.music_end");
    c->tok_codec_offset = gf_get_u32(gf, "yue2.token.codec_offset");
    c->tok_codec_size   = gf_get_u32(gf, "yue2.token.codec_size");
    c->tok_latent_start = gf_get_u32(gf, "yue2.token.latent_start");
    c->tok_latent_end   = gf_get_u32(gf, "yue2.token.latent_end");
    c->tok_latent_pad   = gf_get_u32(gf, "yue2.token.latent_pad");

    c->abc.temperature        = gf_get_f32(gf, "yue2.sampling.abc.temperature");
    c->abc.top_p              = gf_get_f32(gf, "yue2.sampling.abc.top_p");
    c->abc.top_k              = gf_get_u32(gf, "yue2.sampling.abc.top_k");
    c->abc.repetition_penalty = gf_get_f32(gf, "yue2.sampling.abc.repetition_penalty");
    c->abc.penalty_window     = gf_get_u32(gf, "yue2.sampling.abc.penalty_window");
    c->abc.min_tokens         = gf_get_u32(gf, "yue2.sampling.abc.min_tokens");
    c->abc.max_tokens         = gf_get_u32(gf, "yue2.sampling.abc.max_tokens");

    c->semantic.temperature        = gf_get_f32(gf, "yue2.sampling.semantic.temperature");
    c->semantic.top_p              = gf_get_f32(gf, "yue2.sampling.semantic.top_p");
    c->semantic.top_k              = gf_get_u32(gf, "yue2.sampling.semantic.top_k");
    c->semantic.repetition_penalty = gf_get_f32(gf, "yue2.sampling.semantic.repetition_penalty");
    c->semantic.penalty_window     = gf_get_u32(gf, "yue2.sampling.semantic.penalty_window");
    c->semantic.min_tokens         = gf_get_u32(gf, "yue2.sampling.semantic.min_tokens");
    c->semantic.max_tokens         = gf_get_u32(gf, "yue2.sampling.semantic.max_tokens");

    c->ode_steps  = gf_get_u32(gf, "yue2.ode.steps");
    c->ode_method = gf_get_str(gf, "yue2.ode.method");
}

static void yue2_parse_vae_config(const GGUFModel & gf, Yue2VaeConfig * c) {
    c->variant             = gf_get_str(gf, "yue2vae.variant");
    c->sample_rate          = gf_get_u32(gf, "yue2vae.sample_rate");
    c->downsampling_ratio   = gf_get_u32(gf, "yue2vae.downsampling_ratio");
    c->audio_channels       = gf_get_u32(gf, "yue2vae.audio_channels");
    c->latent_dim           = gf_get_u32(gf, "yue2vae.latent_dim");
    c->encoder_latent_dim   = gf_get_u32(gf, "yue2vae.encoder_latent_dim");
    c->channels             = gf_get_u32(gf, "yue2vae.channels");
    c->strides              = yue2_get_i32_arr(gf, "yue2vae.strides");
    c->res_dilations        = yue2_get_i32_arr(gf, "yue2vae.res_dilations");
    c->use_snake            = gf_get_bool(gf, "yue2vae.use_snake");
    c->snake_type           = gf_get_str(gf, "yue2vae.snake_type");
    c->snake_eps            = gf_get_f32(gf, "yue2vae.snake_eps");
    c->snake_formula        = gf_get_str(gf, "yue2vae.snake_formula");
    c->final_tanh           = gf_get_bool(gf, "yue2vae.final_tanh");
    c->output_clamp_min     = gf_get_f32(gf, "yue2vae.output_clamp_min");
    c->output_clamp_max     = gf_get_f32(gf, "yue2vae.output_clamp_max");
    c->weight_norm_folded   = gf_get_bool(gf, "yue2vae.weight_norm_folded");
    c->has_encoder          = gf_get_bool(gf, "yue2vae.has_encoder");
    c->decode_core_frames   = gf_get_u32(gf, "yue2vae.decode_core_frames");
    c->decode_halo_frames   = gf_get_u32(gf, "yue2vae.decode_halo_frames");
    c->required_halo        = gf_get_u32(gf, "yue2vae.required_halo");
}

// Sanity checks that catch a structurally wrong file before any VRAM is touched.
static void yue2_validate_lm_config(const Yue2LmConfig & c, std::vector<std::string> * errs) {
    auto need = [&](bool ok, const char * what) {
        if (!ok && errs->size() < 24) {
            errs->push_back(std::string("LM config: ") + what);
        }
    };
    need(c.block_count > 0, "yue2.block_count is 0 or missing");
    need(c.embedding_length > 0, "yue2.embedding_length is 0 or missing");
    need(c.vocab_size > 0, "yue2.vocab_size is 0 or missing");
    need(c.head_count > 0 && c.head_count_kv > 0, "head counts are 0 or missing");
    need(c.key_length > 0 && c.value_length > 0, "key/value length is 0 or missing");
    need(c.head_count * c.key_length == c.embedding_length, "head_count * key_length != embedding_length");
    need(c.latent_dim > 0, "yue2.latent_dim is 0 or missing");
    need(c.tok_eod > 0, "yue2.token.eod is 0 or missing");
    need(c.tok_music_end > c.tok_music_start, "yue2.token.music_end <= music_start");
    need(c.tok_codec_offset >= c.tok_music_end, "yue2.token.codec_offset < music_end");
}

static void yue2_validate_vae_config(const Yue2VaeConfig & c, std::vector<std::string> * errs) {
    auto need = [&](bool ok, const char * what) {
        if (!ok && errs->size() < 24) {
            errs->push_back(std::string("VAE config: ") + what);
        }
    };
    need(!c.strides.empty(), "yue2vae.strides is empty or missing");
    need(c.res_dilations.size() == 3, "yue2vae.res_dilations is not 3 entries");
    need(c.channels > 0, "yue2vae.channels is 0 or missing");
    need(c.latent_dim > 0, "yue2vae.latent_dim is 0 or missing");
    need(c.audio_channels > 0, "yue2vae.audio_channels is 0 or missing");
    if (!c.strides.empty()) {
        uint32_t prod = 1;
        for (int32_t s : c.strides) {
            prod *= (uint32_t) s;
        }
        need(prod == c.downsampling_ratio, "product(strides) != downsampling_ratio");
    }
}

// ── Probe: read a file's header only, no weights ─────────────────────────────

static void yue2_probe_file(const std::string & path, Yue2FileInfo * fi, bool is_lm, Yue2LmConfig * lm_cfg,
                            Yue2VaeConfig * vae_cfg, std::vector<std::string> * errs) {
    fi->found = true;
    fi->path  = path;
    fi->name  = yue2_basename(path);

    GGUFModel gf = {};
    if (!gf_load(&gf, path.c_str())) {
        fi->probe_error = "gguf header parse failed";
        if (errs && errs->size() < 24) {
            errs->push_back(fi->name + ": " + fi->probe_error);
        }
        return;
    }

    fi->file_bytes = (uint64_t) gf.file_size;
    fi->arch       = gf_get_str(gf, "general.architecture");
    fi->license    = gf_get_str(gf, "general.license");
    fi->file_type  = gf_get_u32(gf, "general.file_type");
    fi->n_tensors  = (int) gguf_get_n_tensors(gf.gguf);

    uint64_t bytes = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(gf.gguf); i++) {
        ggml_tensor * t = ggml_get_tensor(gf.meta, gguf_get_tensor_name(gf.gguf, i));
        if (t) {
            bytes += (uint64_t) ggml_nbytes(t);
        }
    }
    fi->tensor_bytes = bytes;

    const std::string want_arch = is_lm ? "yue2" : "yue2-vae";
    if (fi->arch != want_arch) {
        fi->probe_error = "general.architecture is '" + fi->arch + "', expected '" + want_arch + "'";
        if (errs && errs->size() < 24) {
            errs->push_back(fi->name + ": " + fi->probe_error);
        }
        gf_close(&gf);
        return;
    }

    if (is_lm && lm_cfg) {
        yue2_parse_lm_config(gf, lm_cfg);
        yue2_validate_lm_config(*lm_cfg, errs);
    }
    if (!is_lm && vae_cfg) {
        yue2_parse_vae_config(gf, vae_cfg);
        yue2_validate_vae_config(*vae_cfg, errs);
    }

    fi->probe_ok = true;
    gf_close(&gf);
}

// ── Discovery ─────────────────────────────────────────────────────────────

// Plain ASCII case-insensitive compare -- avoids pulling in <cstring>'s
// strcasecmp/_stricmp split (quantize.cpp's #ifdef _WIN32 macro trick isn't
// worth repeating in a header included from six different .h files here).
static bool yue2_ieq(const std::string & a, const char * b) {
    size_t i = 0;
    for (; a[i] && b[i]; i++) {
        char ca = a[i], cb = b[i];
        if (ca >= 'A' && ca <= 'Z') {
            ca = (char) (ca - 'A' + 'a');
        }
        if (cb >= 'A' && cb <= 'Z') {
            cb = (char) (cb - 'A' + 'a');
        }
        if (ca != cb) {
            return false;
        }
    }
    return a[i] == '\0' && b[i] == '\0';
}

// Rank a quant token best-first. Mirrors mm3_quant_rank (engine/src/minimax/
// mm3-model.h) exactly: same ladder order, same "-imat suffix ranks one notch
// better than its plain twin" rule, same unknown-token fallback. Quantized
// files come from engine/tools/quantize.cpp's arch "yue2" rules (§2 of the
// quant-ladder doc); f32 has no quantize.cpp path (it is only ever the
// converter's own --type f32 output) but stays in the ladder as a rank
// between the converter's native types and the quantized ones.
static int yue2_quant_rank(const std::string & q) {
    static const char * order[] = { "bf16",  "f16",    "f32",    "q8_0",   "Q6_K",   "Q5_K_M",
                                    "Q5_K_S", "Q4_K_M", "Q4_K_S", "NVFP4",  "MXFP4",  "IQ4_XS",
                                    "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_XXS", "Q2_K",  "IQ2_XS",
                                    "IQ2_XXS" };
    // Case-insensitive on the base token: quantize.cpp writes filenames from
    // its uppercase VARIANTS[].name (Q6_K, IQ4_XS, ...), the converter writes
    // lowercase (bf16, f16, f32) -- accept whichever casing shows up on disk
    // rather than keeping two literal entries per token like mm3_quant_rank
    // does for its handful of dual-case tokens.
    std::string       base   = q;
    int               bonus  = 1;
    const std::string suffix = "-imat";
    if (base.size() > suffix.size() && base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
        base  = base.substr(0, base.size() - suffix.size());
        bonus = 0;
    }
    for (int i = 0; i < (int) (sizeof(order) / sizeof(order[0])); i++) {
        if (yue2_ieq(base, order[i])) {
            return i * 2 + bonus;
        }
    }
    return 1000;  // unknown quant: offered, but never auto-selected over a known one
}

// Every yue2-lm-<type>.gguf across the search dirs (filename-only, no header
// probe), de-duplicated by type (earlier search dir wins, same "yue2/ before
// root" precedence as yue2_find_variant) and sorted best-first by
// yue2_quant_rank. Mirrors mm3_enumerate (engine/src/minimax/mm3-model.h).
static void yue2_enumerate_lm(const Yue2Model & m, std::vector<Yue2Variant> * out) {
    out->clear();
    const std::string prefix = "yue2-lm-";
    for (const auto & dir : m.search_dirs) {
        std::vector<std::string> names;
        yue2_list_dir(dir, &names);
        for (const auto & n : names) {
            if (n.size() <= prefix.size() + 5 || n.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            if (n.compare(n.size() - 5, 5, ".gguf") != 0) {
                continue;
            }
            Yue2Variant v;
            v.type = n.substr(prefix.size(), n.size() - prefix.size() - 5);
            bool dup = false;
            for (const auto & e : *out) {
                if (e.type == v.type) {
                    dup = true;  // earlier search dir already supplied this type
                    break;
                }
            }
            if (dup) {
                continue;
            }
            v.path  = dir + YUE2_SEP + n;
            v.name  = n;
            v.bytes = yue2_file_size(v.path);
            out->push_back(v);
        }
    }
    std::sort(out->begin(), out->end(), [](const Yue2Variant & a, const Yue2Variant & b) {
        const int ra = yue2_quant_rank(a.type), rb = yue2_quant_rank(b.type);
        return ra != rb ? ra < rb : a.type < b.type;
    });
}

// Find the best-first yue2-<stem>-<quant>.gguf across the search dirs.
static bool yue2_find_variant(const std::vector<std::string> & search_dirs, const std::string & stem,
                              std::string * out_path) {
    const std::string prefix = std::string("yue2-") + stem + "-";
    std::string       best_path;
    int                best_rank = 1 << 30;
    for (const auto & dir : search_dirs) {
        std::vector<std::string> names;
        yue2_list_dir(dir, &names);
        for (const auto & n : names) {
            if (n.size() <= prefix.size() + 5 || n.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            if (n.compare(n.size() - 5, 5, ".gguf") != 0) {
                continue;
            }
            const std::string quant = n.substr(prefix.size(), n.size() - prefix.size() - 5);
            const int         rank  = yue2_quant_rank(quant);
            if (rank < best_rank) {
                best_rank = rank;
                best_path = dir + YUE2_SEP + n;
            }
        }
        if (!best_path.empty()) {
            break;  // this dir supplied a match; mirrors mm3's "subdir wins" ordering
        }
    }
    if (best_path.empty()) {
        return false;
    }
    *out_path = best_path;
    return true;
}

// Cheap filename-only probe: does <models_dir>/yue2/ (or its root) hold ANY
// yue2-lm-*.gguf? No header read. Mirrors mm3_weights_present() — this is
// the function hot-step-families.h's commented-out row already names.
static bool yue2_weights_present(const char * models_dir) {
    if (!models_dir || !models_dir[0]) {
        return false;
    }
    const std::string root    = models_dir;
    const std::string dirs[2] = { root + YUE2_SEP "yue2", root };
    for (const auto & dir : dirs) {
        std::vector<std::string> names;
        yue2_list_dir(dir, &names);
        for (const auto & n : names) {
            if (n.size() > 12 && n.compare(0, 8, "yue2-lm-") == 0 && n.compare(n.size() - 5, 5, ".gguf") == 0) {
                return true;
            }
        }
    }
    return false;
}

// Locate the LM GGUF and both VAE variant GGUFs, and probe their headers.
// Cheap (mmap + header parse, no weight reads). Safe to call when nothing is
// there — YuE2 is simply reported unavailable via Yue2FileInfo::found.
//
// `lm_type_override`, when non-null/non-empty, pins the LM file to the exact
// yue2-lm-<lm_type_override>.gguf (first search dir that has it) instead of
// yue2_find_variant's best-first pick — quant-ladder measurement needs every
// type probed in turn (bf16 as reference, then each quantize.cpp output),
// not always the best one on disk. nullptr/empty preserves the normal
// best-first behavior for every other caller.
static void yue2_discover(Yue2Model * m, const char * models_dir, const char * lm_type_override = nullptr) {
    m->models_dir = models_dir ? models_dir : "";
    m->search_dirs.clear();
    m->meta_errors.clear();
    m->lm_file = Yue2FileInfo{};
    for (int v = 0; v < YUE2_VAE_VARIANT_COUNT; v++) {
        m->vae_file[v] = Yue2FileInfo{};
    }

    if (m->models_dir.empty()) {
        return;
    }

    m->search_dirs.push_back(m->models_dir + YUE2_SEP "yue2");
    m->search_dirs.push_back(m->models_dir);

    std::string lm_path;
    bool        have_lm_path = false;
    if (lm_type_override && lm_type_override[0]) {
        const std::string want = std::string("yue2-lm-") + lm_type_override + ".gguf";
        for (const auto & dir : m->search_dirs) {
            const std::string candidate = dir + YUE2_SEP + want;
            if (yue2_file_exists(candidate)) {
                lm_path      = candidate;
                have_lm_path = true;
                break;
            }
        }
        if (!have_lm_path && m->meta_errors.size() < 24) {
            m->meta_errors.push_back("--lm-type '" + std::string(lm_type_override) + "': no " + want +
                                      " under " + m->search_dirs[0] + " or " + m->search_dirs[1]);
        }
    } else {
        have_lm_path = yue2_find_variant(m->search_dirs, "lm", &lm_path);
    }
    if (have_lm_path) {
        yue2_probe_file(lm_path, &m->lm_file, /*is_lm=*/true, &m->lm_cfg, nullptr, &m->meta_errors);
    }
    yue2_enumerate_lm(*m, &m->lm_variants);
    for (int v = 0; v < YUE2_VAE_VARIANT_COUNT; v++) {
        std::string path;
        const std::string stem = std::string("vae-") + YUE2_VAE_VARIANT_NAME[v];
        if (yue2_find_variant(m->search_dirs, stem, &path)) {
            Yue2VaeConfig cfg;  // scratch — the model's own vae_cfg is filled at load time
            yue2_probe_file(path, &m->vae_file[v], /*is_lm=*/false, nullptr, &cfg, &m->meta_errors);
        }
    }
}

static bool yue2_available(const Yue2Model & m) {
    return m.lm_file.found && m.lm_file.probe_ok;
}

// ── Weight loading ────────────────────────────────────────────────────────

static bool yue2_load_lm_tensors(Yue2Model * m, const GGUFModel & gf, std::vector<std::string> * errs) {
    const Yue2LmConfig & c = m->lm_cfg;
    const int64_t        H = c.embedding_length;
    const int64_t        V = c.vocab_size;
    const int64_t        D = c.key_length;
    const int64_t        Q = (int64_t) c.head_count * D;
    const int64_t        K = (int64_t) c.head_count_kv * D;
    const int64_t        F = c.feed_forward_length;
    const int64_t         LD = c.latent_dim;
    const int64_t         MF = c.max_latent_frames;
    const int             L  = (int) c.block_count;

    // Budget: 3 (embed/norm/output) + 4 (flow2, vae2llm+llm2vae) + 4 (time_embd)
    // + 1 (latent_pos_embed) + L * (11 AR + 11 NAR).
    wctx_init(&m->wctx_lm, 12 + L * 22);
    Yue2Loader ld{ &m->wctx_lm, &gf, &m->tmap_lm, errs };

    m->lm.token_embd  = ld.req("token_embd.weight", H, V);
    m->lm.output_norm = ld.req("output_norm.weight", H);
    m->lm.output      = ld.req("output.weight", H, V);  // distinct tensor: not tied to token_embd

    m->lm.vae2llm_w      = ld.req("vae2llm.weight", LD, H);
    m->lm.vae2llm_b      = ld.req("vae2llm.bias", H);
    m->lm.llm2vae_w      = ld.req("llm2vae.weight", H, LD);
    m->lm.llm2vae_b      = ld.req("llm2vae.bias", LD);
    m->lm.time_embd_w[0] = ld.req("time_embd.0.weight", 256, H);
    m->lm.time_embd_b[0] = ld.req("time_embd.0.bias", H);
    m->lm.time_embd_w[1] = ld.req("time_embd.1.weight", H, H);
    m->lm.time_embd_b[1] = ld.req("time_embd.1.bias", H);
    m->lm.latent_pos_embed = ld.req("latent_pos_embed.weight", H, MF);

    // TRAP (05-gguf-layout.md §3.3): latent_pos_embed must never be
    // block-quantized -- it is read by a plain ggml_get_rows index gather
    // (yue2-nar-graph.h), not a matmul, and recompute-then-round-trip from
    // the textbook sinusoid formula already mismatches the checkpoint's own
    // stored values in 62,153/50,331,648 entries, so a lossy quant on top
    // would only compound that. quantize.cpp's should_quantize() excludes
    // this tensor by name for exactly this reason (see its YUE2 POLICY
    // comment) -- this assert is the loader's own backstop in case a future
    // quantizer or hand-edited GGUF skips that exclusion.
    //
    // On an UNQUANTIZED file (plain converter output, every big matmul weight
    // sharing one NATIVE dtype) it must additionally match token_embd's own
    // type exactly -- that is the stronger invariant the comment above used
    // to assert unconditionally, before quantize.cpp made token_embd's type
    // independently choosable per variant (Q8_0/Q6_K/... via VARIANTS[].embed)
    // while latent_pos_embed stays untouched. Once token_embd itself is a
    // quantized type, exact equality is no longer the right test; "still a
    // genuine float type" is.
    if (m->lm.latent_pos_embed) {
        if (ggml_is_quantized(m->lm.latent_pos_embed->type)) {
            char buf[192];
            snprintf(buf, sizeof(buf), "latent_pos_embed.weight was block-quantized (type %d) -- must stay F32/F16/BF16",
                     (int) m->lm.latent_pos_embed->type);
            ld.fail(buf);
        } else if (m->lm.token_embd && !ggml_is_quantized(m->lm.token_embd->type) &&
                   m->lm.latent_pos_embed->type != m->lm.token_embd->type) {
            char buf[192];
            snprintf(buf, sizeof(buf),
                     "latent_pos_embed.weight type (%d) != token_embd.weight type (%d) — NATIVE policy violated",
                     (int) m->lm.latent_pos_embed->type, (int) m->lm.token_embd->type);
            ld.fail(buf);
        }
    }

    m->lm.blk.assign((size_t) L, Yue2LmLayer{});
    m->lm.nar_blk.assign((size_t) L, Yue2NarLayer{});
    for (int i = 0; i < L; i++) {
        Yue2LmLayer & b = m->lm.blk[(size_t) i];
        b.attn_norm    = ld.req(yue2_fmt("blk.%d.attn_norm.weight", i), H);
        b.attn_q       = ld.req(yue2_fmt("blk.%d.attn_q.weight", i), H, Q);
        b.attn_k       = ld.req(yue2_fmt("blk.%d.attn_k.weight", i), H, K);
        b.attn_v       = ld.req(yue2_fmt("blk.%d.attn_v.weight", i), H, K);
        b.attn_output  = ld.req(yue2_fmt("blk.%d.attn_output.weight", i), Q, H);
        b.attn_q_norm  = ld.req(yue2_fmt("blk.%d.attn_q_norm.weight", i), D);
        b.attn_k_norm  = ld.req(yue2_fmt("blk.%d.attn_k_norm.weight", i), D);
        b.ffn_norm     = ld.req(yue2_fmt("blk.%d.ffn_norm.weight", i), H);
        b.ffn_gate     = ld.req(yue2_fmt("blk.%d.ffn_gate.weight", i), H, F);
        b.ffn_up       = ld.req(yue2_fmt("blk.%d.ffn_up.weight", i), H, F);
        b.ffn_down     = ld.req(yue2_fmt("blk.%d.ffn_down.weight", i), F, H);

        Yue2NarLayer & nb = m->lm.nar_blk[(size_t) i];
        nb.attn_norm    = ld.req(yue2_fmt("blk.%d.nar_attn_norm.weight", i), H);
        nb.attn_q       = ld.req(yue2_fmt("blk.%d.nar_attn_q.weight", i), H, Q);
        nb.attn_k       = ld.req(yue2_fmt("blk.%d.nar_attn_k.weight", i), H, K);
        nb.attn_v       = ld.req(yue2_fmt("blk.%d.nar_attn_v.weight", i), H, K);
        nb.attn_output  = ld.req(yue2_fmt("blk.%d.nar_attn_output.weight", i), Q, H);
        nb.attn_q_norm  = ld.req(yue2_fmt("blk.%d.nar_attn_q_norm.weight", i), D);
        nb.attn_k_norm  = ld.req(yue2_fmt("blk.%d.nar_attn_k_norm.weight", i), D);
        nb.ffn_norm     = ld.req(yue2_fmt("blk.%d.nar_ffn_norm.weight", i), H);
        nb.ffn_gate     = ld.req(yue2_fmt("blk.%d.nar_ffn_gate.weight", i), H, F);
        nb.ffn_up       = ld.req(yue2_fmt("blk.%d.nar_ffn_up.weight", i), H, F);
        nb.ffn_down     = ld.req(yue2_fmt("blk.%d.nar_ffn_down.weight", i), F, H);

        if (!errs->empty()) {
            break;  // one bad layer means the file is wrong; don't spam 28 copies
        }
    }
    return errs->empty();
}

// Decoder channel widths per block, index 0..NB (NB=strides.size(), 6 for
// both shipped checkpoints): encoder forward order is
// [channels, channels, 2*channels, 4*channels, 8*channels, 16*channels, 32*channels]
// (c_mults=[1,2,4,8,16,32] with a leading 1 duplicated, per
// 04-weights-inventory.md §3.2); the decoder runs the exact reverse. Neither
// c_mults nor these per-block widths are their own GGUF KV — only `channels`
// and `strides` are — so this formula is the loader's one hardcoded piece of
// architecture knowledge, same status as MM3's own fixed vocoder ladder.
static std::vector<int64_t> yue2_vae_encoder_widths(const Yue2VaeConfig & c) {
    const int64_t        ch = c.channels;
    const size_t          nb = c.strides.size();
    std::vector<int64_t> w(nb + 1, ch);
    for (size_t i = 1; i <= nb; i++) {
        w[i] = ch * (int64_t) (1ull << (i - 1));
    }
    return w;
}

static bool yue2_load_vae_tensors(Yue2Model * m, const GGUFModel & gf, bool want_encoder,
                                  std::vector<std::string> * errs) {
    const Yue2VaeConfig & c  = m->vae_cfg;
    const size_t           NB = c.strides.size();
    const size_t           NR = c.res_dilations.size();  // always 3

    const std::vector<int64_t> ew = yue2_vae_encoder_widths(c);  // size NB+1, index 0..NB
    // Decoder widths are the exact reverse: dw[i] = ew[NB-i].
    auto dw = [&](size_t i) { return ew[NB - i]; };

    // Budget: dec (2 conv_in + NB*(2 snake_pre + 2 upsample + NR*8) + 2 snake_out + 1 conv_out)
    // + optionally enc (mirror shape, minus the dec asymmetries).
    const int n_dec = 2 + (int) NB * (2 + 2 + (int) NR * 8) + 2 + 1;
    const int n_enc = want_encoder ? 2 + (int) NB * ((int) NR * 8 + 2 + 2) + 2 + 2 : 0;
    wctx_init(&m->wctx_vae, n_dec + n_enc);
    Yue2Loader ld{ &m->wctx_vae, &gf, &m->tmap_vae, errs };

    // ── decoder ──
    m->vae.dec_conv_in_w = ld.req("dec.conv_in.weight", 7, c.latent_dim, dw(0));
    m->vae.dec_conv_in_b = ld.req("dec.conv_in.bias", dw(0));

    m->vae.dec_blk.assign(NB, Yue2VaeDecBlock{});
    for (size_t bi = 0; bi < NB && errs->empty(); bi++) {
        const int           b    = (int) bi + 1;  // blocks are 1-indexed in the GGUF
        const int64_t        cin  = dw(bi);
        const int64_t        cout = dw(bi + 1);
        const int64_t        stride = c.strides[NB - 1 - bi];  // decoder runs strides reversed
        Yue2VaeDecBlock &    db   = m->vae.dec_blk[bi];

        db.snake_pre_alpha = ld.req(yue2_fmt("dec.blk.%d.snake_pre.alpha", b), cin);
        db.snake_pre_beta  = ld.req(yue2_fmt("dec.blk.%d.snake_pre.beta", b), cin);
        db.upsample_w      = ld.req(yue2_fmt("dec.blk.%d.upsample.weight", b), stride * 2, cout, cin);
        db.upsample_b      = ld.req(yue2_fmt("dec.blk.%d.upsample.bias", b), cout);

        // Dilation (1/3/9 across r=0..2) affects conv1's receptive field at
        // graph-build time, not its stored tensor shape — nothing to check here.
        db.res.assign(NR, Yue2VaeResUnit{});
        for (size_t r = 0; r < NR && errs->empty(); r++) {
            Yue2VaeResUnit & ru = db.res[r];
            ru.snake1_alpha = ld.req(yue2_fmt2("dec.blk.%d.res.%d.snake1.alpha", b, (int) r), cout);
            ru.snake1_beta  = ld.req(yue2_fmt2("dec.blk.%d.res.%d.snake1.beta", b, (int) r), cout);
            ru.conv1_w      = ld.req(yue2_fmt2("dec.blk.%d.res.%d.conv1.weight", b, (int) r), 7, cout, cout);
            ru.conv1_b      = ld.req(yue2_fmt2("dec.blk.%d.res.%d.conv1.bias", b, (int) r), cout);
            ru.snake2_alpha = ld.req(yue2_fmt2("dec.blk.%d.res.%d.snake2.alpha", b, (int) r), cout);
            ru.snake2_beta  = ld.req(yue2_fmt2("dec.blk.%d.res.%d.snake2.beta", b, (int) r), cout);
            ru.conv2_w      = ld.req(yue2_fmt2("dec.blk.%d.res.%d.conv2.weight", b, (int) r), 1, cout, cout);
            ru.conv2_b      = ld.req(yue2_fmt2("dec.blk.%d.res.%d.conv2.bias", b, (int) r), cout);
        }
    }

    m->vae.dec_snake_out_alpha = ld.req("dec.snake_out.alpha", dw(NB));
    m->vae.dec_snake_out_beta  = ld.req("dec.snake_out.beta", dw(NB));
    // dec.conv_out has NO bias — decoder.layers.8 is the one bias=False WNConv1d
    // in the whole stack (04-weights-inventory.md §3.1). Don't look for one.
    m->vae.dec_conv_out_w = ld.req("dec.conv_out.weight", 7, dw(NB), c.audio_channels);

    // ── encoder (optional, decoder_only by default) ──
    if (want_encoder && errs->empty()) {
        m->vae.enc_conv_in_w = ld.req("enc.conv_in.weight", 7, c.audio_channels, ew[0]);
        m->vae.enc_conv_in_b = ld.req("enc.conv_in.bias", ew[0]);

        m->vae.enc_blk.assign(NB, Yue2VaeEncBlock{});
        for (size_t bi = 0; bi < NB && errs->empty(); bi++) {
            const int      b    = (int) bi + 1;
            const int64_t  cin  = ew[bi];
            const int64_t  cout = ew[bi + 1];
            const int64_t  stride = c.strides[bi];  // encoder runs strides forward
            Yue2VaeEncBlock & eb = m->vae.enc_blk[bi];

            eb.res.assign(NR, Yue2VaeResUnit{});
            for (size_t r = 0; r < NR && errs->empty(); r++) {
                Yue2VaeResUnit & ru = eb.res[r];
                ru.snake1_alpha = ld.req(yue2_fmt2("enc.blk.%d.res.%d.snake1.alpha", b, (int) r), cin);
                ru.snake1_beta  = ld.req(yue2_fmt2("enc.blk.%d.res.%d.snake1.beta", b, (int) r), cin);
                ru.conv1_w      = ld.req(yue2_fmt2("enc.blk.%d.res.%d.conv1.weight", b, (int) r), 7, cin, cin);
                ru.conv1_b      = ld.req(yue2_fmt2("enc.blk.%d.res.%d.conv1.bias", b, (int) r), cin);
                ru.snake2_alpha = ld.req(yue2_fmt2("enc.blk.%d.res.%d.snake2.alpha", b, (int) r), cin);
                ru.snake2_beta  = ld.req(yue2_fmt2("enc.blk.%d.res.%d.snake2.beta", b, (int) r), cin);
                ru.conv2_w      = ld.req(yue2_fmt2("enc.blk.%d.res.%d.conv2.weight", b, (int) r), 1, cin, cin);
                ru.conv2_b      = ld.req(yue2_fmt2("enc.blk.%d.res.%d.conv2.bias", b, (int) r), cin);
            }
            eb.snake_post_alpha = ld.req(yue2_fmt("enc.blk.%d.snake_post.alpha", b), cin);
            eb.snake_post_beta  = ld.req(yue2_fmt("enc.blk.%d.snake_post.beta", b), cin);
            eb.downsample_w     = ld.req(yue2_fmt("enc.blk.%d.downsample.weight", b), stride * 2, cin, cout);
            eb.downsample_b     = ld.req(yue2_fmt("enc.blk.%d.downsample.bias", b), cout);
        }

        m->vae.enc_snake_out_alpha = ld.req("enc.snake_out.alpha", ew[NB]);
        m->vae.enc_snake_out_beta  = ld.req("enc.snake_out.beta", ew[NB]);
        m->vae.enc_conv_out_w      = ld.req("enc.conv_out.weight", 3, ew[NB], c.encoder_latent_dim);
        m->vae.enc_conv_out_b      = ld.req("enc.conv_out.bias", c.encoder_latent_dim);
        m->vae.enc_loaded          = errs->empty();
    }

    return errs->empty();
}

// ── Load / unload ────────────────────────────────────────────────────────

static size_t yue2_vram_bytes(const Yue2Model & m) {
    return m.vram_lm + m.vram_vae;
}

static void yue2_unload(Yue2Model * m) {
    if (!m->backend_ref && !m->lm_resident && !m->vae_resident && !m->wctx_lm.ctx && !m->wctx_vae.ctx) {
        return;
    }
    wctx_free(&m->wctx_lm);
    wctx_free(&m->wctx_vae);
    m->lm    = Yue2LmWeights{};
    m->vae   = Yue2VaeWeights{};
    m->tmap_lm.clear();
    m->tmap_vae.clear();
    m->vram_lm     = 0;
    m->vram_vae    = 0;
    m->load_ms     = 0.0;
    m->lm_resident  = false;
    m->vae_resident = false;
    // The merged-adapter description belongs to the RESIDENT weights, which
    // have just gone away. lm_adapter_want (the request) deliberately
    // survives, so the next warm/synth re-merges the same set.
    m->lm_adapter_desc.clear();
    m->lm_adapter_tensors = 0;
    m->lm_adapter_family.clear();
    if (m->backend_ref) {
        backend_release(m->backend, m->cpu_backend);
        m->backend     = nullptr;
        m->cpu_backend = nullptr;
        m->backend_ref = false;
    }
    fprintf(stderr, "[YuE2] Unloaded\n");
}

// Merge every requested NAR LoRA into the LM's staged weights. Call site is
// the seam inside yue2_load_parts: after yue2_load_lm_tensors() has staged
// every PendingCopy with `src` pointing straight into the GGUF mmap
// (gf_load_tensor, gguf-weights.h:159-200) and BEFORE wctx_alloc() walks
// `pending` and uploads (weight-ctx.h:53-71). Patching pc->src in that window
// costs no extra VRAM and needs no second pass — and it has to happen while
// `gf` is still open, since those pointers live in its mmap.
//
// Returns false (and fills `errs`) on a refusal or a failed merge. That is
// deliberately FATAL to the whole load: a user who asked for an adapter and
// silently got the base model has no way to tell, and the whole point of the
// quantized-base guard in yue2-adapter.h is not to ship a wrong-but-quiet
// model. yue2_load_parts's own all-or-nothing contract then unloads.
static bool yue2_apply_adapters(Yue2Model * m, const GGUFModel & gf, std::vector<std::string> * errs) {
    m->lm_adapter_desc.clear();
    m->lm_adapter_tensors = 0;
    m->lm_adapter_family.clear();
    if (m->lm_adapter_want.empty()) {
        return true;
    }

    int  total  = 0;
    bool has_ar = false, has_nar = false;
    for (const Yue2AdapterSpec & spec : m->lm_adapter_want) {
        std::string err;
        std::string fam;
        const int   n =
            yue2_adapter_merge(&m->wctx_lm, gf, spec.path.c_str(), spec.scale, m->backend, &err, &fam);
        if (n < 0) {
            errs->push_back("adapter " + spec.path + ": " + (err.empty() ? "merge failed" : err));
            return false;
        }
        if (n == 0) {
            // Zero matched tensors is not a partial merge, it is a no-op — and
            // an adapter that changes nothing is far more likely to be the
            // wrong file than a deliberate choice. Refuse rather than load a
            // model the caller will believe is adapted.
            errs->push_back("adapter " + spec.path +
                            " matched no YuE2 tensors — wrong file, or exported with keys this loader does "
                            "not recognise (expected yue2.blk.N.nar_*.lora_A.weight for a NAR adapter, or "
                            "yue2.blk.N.attn_q/ffn_*.lora_A.weight for an AR one)");
            return false;
        }
        if (fam == "ar") {
            has_ar = true;
        } else if (fam == "nar") {
            has_nar = true;
        }
        total += n;
    }
    m->lm_adapter_tensors = total;
    m->lm_adapter_desc    = yue2_adapter_key(m->lm_adapter_want);
    // A stack covering both halves is legal and disjoint, so say so rather than
    // picking one — the AR and NAR blocks share no weights (this file's own
    // Mixture-of-Transformers note), which is what makes an AR + NAR pair merge
    // into two non-overlapping sets of tensors.
    m->lm_adapter_family = (has_ar && has_nar) ? "ar+nar" : has_ar ? "ar" : has_nar ? "nar" : "";
    fprintf(stderr, "[YuE2-Adapter] %d tensor(s) patched across %zu adapter(s) [%s]\n", total,
            m->lm_adapter_want.size(),
            m->lm_adapter_family.empty() ? "?" : m->lm_adapter_family.c_str());
    return true;
}

// Load a chosen subset of the two parts into backend buffers. Parts already
// resident are left alone (idempotent). On any failure NOTHING stays
// resident — mirrors mm3_load_parts's all-or-nothing contract.
//
// `variant` selects which VAE file to load when want_vae is set; loading a
// different variant while one is already resident frees the old one first
// (only one VAE variant is ever resident at a time, per the file header note).
static bool yue2_load_parts(Yue2Model * m, bool want_lm, bool want_vae, Yue2VaeVariant variant, bool want_encoder,
                            std::string * err_out) {
    if (want_vae && m->vae_resident && m->vae_loaded_variant != variant) {
        wctx_free(&m->wctx_vae);
        m->vae             = Yue2VaeWeights{};
        m->tmap_vae.clear();
        m->vram_vae        = 0;
        m->vae_resident    = false;
    }

    const bool need_lm  = want_lm && !m->lm_resident;
    const bool need_vae = want_vae && !m->vae_resident;
    if (!need_lm && !need_vae) {
        return true;
    }

    if (need_lm && !yue2_available(*m)) {
        if (err_out) {
            *err_out = m->meta_errors.empty() ? "YuE2 LM GGUF not found or metadata probe failed"
                                              : m->meta_errors[0];
        }
        return false;
    }
    if (need_vae && (!m->vae_file[variant].found || !m->vae_file[variant].probe_ok)) {
        if (err_out) {
            *err_out = std::string("YuE2 VAE (") + YUE2_VAE_VARIANT_NAME[variant] + ") GGUF not found or metadata probe failed";
        }
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();

    if (!m->backend_ref) {
        BackendPair bp = backend_init("YuE2");
        m->backend     = bp.backend;
        m->cpu_backend = bp.cpu_backend;
        m->backend_ref = true;
    }

    std::vector<std::string> errs;
    bool                     ok = true;

    if (need_lm) {
        GGUFModel gf = {};
        ok           = gf_load(&gf, m->lm_file.path.c_str());
        if (!ok) {
            errs.push_back("cannot open " + m->lm_file.path);
        } else {
            ok = yue2_load_lm_tensors(m, gf, &errs);
            if (ok) {
                // Adapter merge goes HERE — between staging and upload, the
                // same seam MM3 uses (mm3-model.h:1766) and ACE uses in dit.h,
                // and inside the `gf` lifetime because the staged pointers are
                // into its mmap. VAE-only loads never reach this branch, which
                // is the whole of "adapters apply to the LM part only": the
                // VAE part has no adaptable tensors and a vae_variant switch
                // (the early-out at the top of this function) frees only
                // wctx_vae, leaving an already-merged wctx_lm untouched.
                ok = yue2_apply_adapters(m, gf, &errs);
            }
            if (ok) {
                ok = wctx_alloc(&m->wctx_lm, m->backend);
                if (!ok) {
                    errs.push_back("backend buffer allocation failed for the LM (out of VRAM?)");
                }
            }
            gf_close(&gf);
        }
    }

    if (ok && need_vae) {
        // Re-parse this variant's own config into m->vae_cfg (probe used a
        // scratch struct so switching variants at discovery time never
        // clobbered a different variant's already-loaded config).
        GGUFModel gf = {};
        ok           = gf_load(&gf, m->vae_file[variant].path.c_str());
        if (!ok) {
            errs.push_back("cannot open " + m->vae_file[variant].path);
        } else {
            yue2_parse_vae_config(gf, &m->vae_cfg);
            yue2_validate_vae_config(m->vae_cfg, &errs);
            if (errs.empty()) {
                ok = yue2_load_vae_tensors(m, gf, want_encoder, &errs);
            } else {
                ok = false;
            }
            if (ok) {
                ok = wctx_alloc(&m->wctx_vae, m->backend);
                if (!ok) {
                    errs.push_back("backend buffer allocation failed for the VAE (out of VRAM?)");
                }
            }
            gf_close(&gf);
        }
    }

    if (!ok) {
        std::string msg = errs.empty() ? "YuE2 load failed" : errs[0];
        for (size_t i = 1; i < errs.size() && i < 6; i++) {
            msg += "; " + errs[i];
        }
        fprintf(stderr, "[YuE2] LOAD FAILED: %s\n", msg.c_str());
        if (err_out) {
            *err_out = msg;
        }
        yue2_unload(m);
        return false;
    }

    if (need_lm) {
        m->vram_lm    = m->wctx_lm.buffer ? ggml_backend_buffer_get_size(m->wctx_lm.buffer) : 0;
        m->lm_resident = true;
    }
    if (need_vae) {
        m->vram_vae         = m->wctx_vae.buffer ? ggml_backend_buffer_get_size(m->wctx_vae.buffer) : 0;
        m->vae_resident      = true;
        m->vae_loaded_variant = variant;
    }
    m->load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    fprintf(stderr, "[YuE2] Loaded%s%s: LM %.2f + VAE(%s) %.2f = %.2f GB in %.0f ms\n", need_lm ? " lm" : "",
            need_vae ? " vae" : "", (double) m->vram_lm / (1024.0 * 1024.0 * 1024.0),
            YUE2_VAE_VARIANT_NAME[m->vae_loaded_variant], (double) m->vram_vae / (1024.0 * 1024.0 * 1024.0),
            (double) yue2_vram_bytes(*m) / (1024.0 * 1024.0 * 1024.0), m->load_ms);
    return true;
}
