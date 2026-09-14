#pragma once
// yue2/yue2-mmsfa.h — MMS_FA forced-alignment acoustic model: GGUF loader,
// conv frontend, wav2vec2-large encoder, CTC head.
//
// HOT-Step file (no acestep.cpp analog). This is the acoustic half of the
// native forced aligner: 16 kHz mono waveform in, 50 Hz log-probabilities over
// a 28-symbol romanised character set out. The CTC Viterbi that turns those
// log-probs into word spans is NOT here (yue2-ctc-align.h).
//
//   pcm 16 kHz -> [waveform layer_norm] -> [7x conv frontend, stride 320] ->
//                 [feature projection] -> [+ positional conv] ->
//                 [24 PRE-LN layers] -> [final LayerNorm] ->
//                 [Linear 1024->28] -> log_softmax -> emissions [T, 28]
//
// AUTHORITY: docs/plans/yue2/17-mms-fa-port.md, and
// K:/yue2/fixtures/mms-fa/pin.json which that document summarises. Both were
// produced by RUNNING torchaudio's own module, not by reading it. Where this
// file and that document disagree, the document wins. Section references (§1,
// §3, §4) point into it.
//
// Weights come from engine/tools/convert-mms-fa.py: one GGUF, arch "mmsfa",
// 422 tensors, every one F32 (1.26 GB). Every tensor name and shape below was
// checked against the real file (K:/yue2/models/mms-fa/mms-fa-f32.gguf) with
// the `gguf` python package, not taken on trust from the converter source.
//
// Gated by `yue2-probe mmsfa-stages` (the 10 s excerpt, stage by stage) and
// `yue2-probe mmsfa-emit` (the full track's argmax agreement). Both modes'
// headers in engine/tools/yue2-probe.cpp carry the bars and the discussion;
// the scoreboard as it stands is 13/14 stage gates plus the full-track gate at
// 11148/11148 frames (100.000% argmax agreement, log-probs max-abs 5.0e-04
// against a 2e-3 bar). The one stage that does not pass is layer23, where the
// doc's ABSOLUTE 2e-3 bar is below the fp32 fixture's own noise floor — a
// float64 recomputation from these same weights scores 2.44e-03 there. See the
// probe's layer23 note.
//
// ── THREE PLACES THE PLAN DOC IS WRONG, AND WHAT THE FIXTURES SAY ──────────
//
// Found while gating, each one localised by the stage it first broke, each one
// settled by an independent float64 recomputation from the same GGUF weights
// (the scratch oracle is reproduced in the report; anyone can rebuild it from
// numpy + the gguf package in a dozen lines). The plan doc says its §1 came
// from "the loaded module, not the docs" — but pin.json records the module's
// CONFIG, and on all three of these the config and the captured activations
// disagree. The activations win: they are what the reference actually
// computed, and they are what the emissions gate is scored against.
//
// A. THE WAVEFORM IS LAYER-NORMED BEFORE THE FRONTEND, and neither §1 nor
//    pin.json mentions it at all. torchaudio's Wav2Vec2 bundle runs
//    `F.layer_norm(waveform, waveform.shape)` — one mean and one variance over
//    the WHOLE utterance, eps 1e-5 — for every "layer_norm"-flavoured
//    wav2vec2-large, which MMS_FA is. Feeding conv0 the raw waveform scores
//    max-abs 9.80 against ex_conv0.npy; feeding it the normalised waveform
//    scores 3.4e-06. This is a per-utterance statistic, so it couples every
//    frame: a track normalised in pieces is NOT the same model input as the
//    same track normalised whole.
//
// B. THE ENCODER IS PRE-LN, not post-LN. pin.json says
//    `layer_norm_first: false` and §1 spells out `a = ln1(x + attn(x))`, but
//    the fixture says the opposite:
//
//        x = x + attn(ln1(x));   x = x + ffn(ln2(x))
//
//    Measured against ex_layer0.npy from the same input: pre-LN 1.6e-05,
//    post-LN 19.5 (rel-L2 1.00, i.e. uncorrelated). This is wav2vec2's
//    do_stable_layer_norm / layer_norm_first=True variant, which is what every
//    wav2vec2-LARGE checkpoint uses. The tensor names are the same either way,
//    which is exactly why this is worth stating: `blk.N.ln1` is the
//    attention's INPUT norm and `blk.N.ln2` is the FFN's INPUT norm, whatever
//    convert-mms-fa.py's own comment ("applied AFTER the attention residual")
//    says.
//
// C. `mmsfa.enc.ln` IS THE FINAL NORM, applied AFTER all 24 layers, not to
//    the transformer's input. §1 says "Residual: z = LayerNorm(x + pos(x))"
//    and §3 labels the fixture `ex_pre_layers.npy` "the transformer input" —
//    both wrong. The transformer's input is plain `x + pos(x)`, no norm; the
//    24 layers run on that; then enc.ln, then the head. Proof, all from the
//    shipped fixtures with no model involved: LayerNorm(ex_layer23) matches
//    ex_pre_layers to 2.2e-07, and head(ex_pre_layers) matches ex_logits to
//    2.4e-06, while head(ex_layer23) misses ex_logits by 324. So the file the
//    doc calls "pre_layers" is the HEAD'S INPUT — the last thing before the
//    aux Linear, not the first thing after the pos-conv. This file taps it
//    under the fixture's name so the gate keeps lining up, and the probe gates
//    it at the deep bar rather than the doc's pre-transformer 1e-4.
//
// ── FIVE THINGS THAT ARE EASY TO GET WRONG HERE ─────────────────────────────
//
// 1. THE CONV-LAYER LayerNorm IS OVER THE CHANNEL AXIS, PER TIME STEP
//    (§1: torchaudio's `LayerNormConvLayer`, which transposes, norms the last
//    axis, and transposes back). It is not a GroupNorm and it is not over
//    time. Norming over time instead still runs and still produces
//    plausible-looking features.
//
// 2. THE POSITIONAL CONV IS GROUPED (groups=16) AND DROPS ITS LAST FRAME.
//    k=128 with pad=k/2 yields T+1 frames; torchaudio's
//    ConvolutionalPositionalEmbedding removes num_remove=1 from the END. Keep
//    the extra frame and every downstream stage is shifted by 20 ms.
//    ggml has no grouped conv: this file runs 16 independent 64-channel convs
//    over channel slices and concatenates them (see mmsfa_pos_conv below).
//
// 3. THE POS-CONV WEIGHT IS ALREADY WEIGHT-NORM BAKED. The checkpoint stores
//    `parametrizations.weight.original{0,1}`; convert-mms-fa.py computes
//    `torch._weight_norm(v, g, dim=2)` at conversion time and cross-checks it
//    against the live module. mmsfa.pos_conv.weight is a plain conv weight —
//    do not re-normalise it here.
//
// 4. THE ENCODER IS PRE-LN and the post-LN spelling uses the same tensor
//    names — see note B above. The KV `mmsfa.layer_norm_first` claims
//    otherwise and this file deliberately does not obey it; the loader prints
//    a one-line warning when it disagrees rather than building a graph the
//    weights do not want.
//
// 5. THE GELU FLAVOUR IS MEASURED, NOT ASSUMED. The plan doc says exact erf
//    everywhere and explicitly asks for it to be measured at conv0 and at
//    layer0 (the MERT port found tanh in one head and erf in the other —
//    yue2-mert.h's "GELU: DETERMINED, NOT ASSUMED"). So the flavour is a
//    runtime switch (Yue2MmsfaOptions::gelu_tanh) and the probe runs both.
//    MEASURED on the 10 s excerpt, max-abs against the fixtures:
//        conv0    erf 5.72e-06   tanh 3.91e-03     683x
//        layer0   erf 2.05e-05   tanh 4.05e-02    1976x
//    Erf wins at both depths by three orders of magnitude, and the tanh error
//    GROWS with depth (4.9 at layer23) while the erf error does not — the
//    signature of a systematic per-layer bias rather than rounding.
//
// ── LAYOUT CONVENTION ───────────────────────────────────────────────────────
//
// The canonical activation layout is ne = [C, T]: channel fastest, time
// slowest. That is byte-for-byte torch's row-major [B, T, C], and it is also
// byte-for-byte the fixtures' own row-major [T, C] .npy bodies — ex_conv0.npy
// is [31999, 512] and reads back with NO transpose. Every stage tap in this
// file is handed out in that layout, so the probe's comparison is a straight
// memcmp-shaped diff and not a permutation.
//
// The convs want the opposite order (ggml_im2col reads time along ne0), so
// every conv is bracketed by a transpose, exactly mirroring the reference's
// own `transpose(-2, -1)` calls.
//
// ── PRECISION ───────────────────────────────────────────────────────────────
//
// The reference is fp32 on the CPU with no autocast anywhere, and the GGUF is
// F32 throughout, so this is an fp32-vs-fp32 comparison with no storage
// rounding on either side. Everything below therefore uses the F32 im2col path
// (never ggml_conv_1d, which forces an F16 im2col — same call yue2-mert.h and
// yue2_vae_conv1d already made) and ggml_gelu_erf (exact erff in f32 on the
// CPU backend, no F16 lookup table — unlike ggml_gelu, which routes through
// ggml_table_gelu_f16 when ggml is built with GGML_GELU_FP16).
//
// ── MEMORY, AND WHY ATTENTION IS CHUNKED ────────────────────────────────────
//
// A full track is ~3.7 minutes = 3.57 M samples = 11148 frames at 50 Hz. A
// naive [T_k, T_q, head] score tensor is 11148 * 11148 * 16 * 4 = 7.95 GB in
// F32 — per layer, and the graph allocator must size for the largest live set.
// So attention is built one HEAD at a time, and within a head in QUERY CHUNKS
// sized to a byte budget (Yue2MmsfaOptions::attn_score_budget, 256 MB by
// default). This is arithmetically identical to the unchunked form — softmax
// normalises over ne0 = keys, so every query row is independent — and it is
// NOT ggml_flash_attn_ext: the CPU flash kernel is an F16 path
// (ggml_compute_forward_flash_attn_ext_f16), and this port is gated at 2e-3
// against an fp32 reference with no F16 anywhere else in it.
//
// Measured on a 3.7-minute track (11148 frames, 16 CPU threads on a Ryzen
// 9950X3D): 2 query chunks per head, 5963 graph nodes, 4878 MB compute buffer,
// 6343 MB peak working set including the 1.2 GB of weights, 95.2 s wall —
// 2.34x realtime. The conv frontend's own im2col, not attention, is what sets
// the peak: conv1 alone materialises a [1536, 356763] F32 column matrix.
// Chunking is arithmetically free, not a trade: `mmsfa-stages
// --attn-budget 128` forces 8 chunks per head on the 10 s clip and reproduces
// the 1-chunk numbers to the last printed digit at every stage.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "yue2-model.h"  // WeightCtx/Yue2Loader/backend machinery, gguf-weights.h

// ── Config ──────────────────────────────────────────────────────────────────

// Every field is read from the GGUF's mmsfa.* KV block; nothing here is
// hardcoded architecture knowledge. convert-mms-fa.py wrote these from
// pin.json after expect-checking every tensor against them, so a config drift
// shows up as a load error rather than as silently wrong emissions.
struct Yue2MmsfaConfig {
    uint32_t sample_rate = 0;  // 16000

    // Feature extractor: 7 x (Conv1d -> LayerNorm(C) -> GELU), all with bias.
    uint32_t             n_conv = 0;  // 7
    std::vector<int32_t> conv_in;     // [1, 512, 512, 512, 512, 512, 512]
    std::vector<int32_t> conv_out;    // [512] * 7
    std::vector<int32_t> conv_kernel; // [10, 3, 3, 3, 3, 2, 2]
    std::vector<int32_t> conv_stride; // [5, 2, 2, 2, 2, 2, 2]  product = 320
    uint32_t             feat_dim = 0;  // 512

    // Encoder.
    uint32_t embed     = 0;  // 1024
    uint32_t n_layers  = 0;  // 24
    uint32_t n_heads   = 0;  // 16
    uint32_t head_dim  = 0;  // 64
    uint32_t ffn       = 0;  // 4096
    float    attn_scale = 0.0f;  // 64^-0.5 = 0.125
    float    ln_eps     = 0.0f;  // 1e-5
    bool     layer_norm_first = true;  // MUST be false — trap 4

    // Positional conv.
    uint32_t pos_conv_kernel     = 0;  // 128
    uint32_t pos_conv_groups     = 0;  // 16
    uint32_t pos_conv_padding    = 0;  // 64
    uint32_t pos_conv_num_remove = 0;  // 1 — trap 2

    // CTC head.
    uint32_t                 n_labels = 0;  // 28
    std::vector<std::string> labels;        // index = class id, 0 = blank
    std::string              blank_label;   // "-"
};

// ── Weights ─────────────────────────────────────────────────────────────────

// One feature-extractor layer: Conv1d -> LayerNorm(out) -> GELU.
struct Yue2MmsfaConvLayer {
    ggml_tensor * w    = nullptr;  // [K, IC, OC]  (ne order; torch [OC, IC, K])
    ggml_tensor * b    = nullptr;  // [OC]
    ggml_tensor * ln_w = nullptr;  // [OC]  LayerNorm over CHANNELS — trap 1
    ggml_tensor * ln_b = nullptr;  // [OC]
};

// One PRE-LN encoder layer. Residual structure, spelled out because the
// post-LN variant shares every tensor name (note B):
//     x = x + attn(ln1(x));   x = x + ffn(ln2(x))
struct Yue2MmsfaBlock {
    ggml_tensor * attn_q_w = nullptr, * attn_q_b = nullptr;  // [H,H] / [H]
    ggml_tensor * attn_k_w = nullptr, * attn_k_b = nullptr;
    ggml_tensor * attn_v_w = nullptr, * attn_v_b = nullptr;
    ggml_tensor * attn_o_w = nullptr, * attn_o_b = nullptr;
    ggml_tensor * ln1_w    = nullptr, * ln1_b    = nullptr;  // the ATTENTION's input norm
    ggml_tensor * ffn_up_w = nullptr, * ffn_up_b = nullptr;  // [H,F] / [F]
    ggml_tensor * ffn_dn_w = nullptr, * ffn_dn_b = nullptr;  // [F,H] / [H]
    ggml_tensor * ln2_w    = nullptr, * ln2_b    = nullptr;  // the FFN's input norm
};

struct Yue2MmsfaWeights {
    std::vector<Yue2MmsfaConvLayer> conv;  // n_conv entries

    ggml_tensor * fp_ln_w = nullptr, * fp_ln_b = nullptr;  // [512] LayerNorm
    ggml_tensor * fp_w    = nullptr, * fp_b    = nullptr;  // [512,1024] / [1024]

    ggml_tensor * pos_w = nullptr;  // [128, 64, 1024] weight-norm BAKED — trap 3
    ggml_tensor * pos_b = nullptr;  // [1024]

    ggml_tensor * enc_ln_w = nullptr, * enc_ln_b = nullptr;  // the FINAL norm — note C

    std::vector<Yue2MmsfaBlock> blk;  // n_layers entries

    ggml_tensor * head_w = nullptr;  // [1024, 28] — raw logits, NOT log_softmax
    ggml_tensor * head_b = nullptr;  // [28]
};

struct Yue2MmsfaModel {
    std::string      path;
    std::string      arch;  // must be "mmsfa"
    std::string      license;
    Yue2MmsfaConfig  cfg;
    Yue2MmsfaWeights w;

    bool           backend_ref = false;
    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    WeightCtx      wctx        = {};
    size_t         vram        = 0;
    double         load_ms     = 0.0;

    std::map<std::string, ggml_tensor *> tmap;  // introspection only
};

// ── Config parsing / validation ─────────────────────────────────────────────

// gguf string arrays (mmsfa.labels). yue2-model.h only has the i32 flavour.
static std::vector<std::string> yue2_mmsfa_get_str_arr(const GGUFModel & gf, const char * key) {
    std::vector<std::string> out;
    const int64_t            idx = gguf_find_key(gf.gguf, key);
    if (idx < 0 || gguf_get_kv_type(gf.gguf, idx) != GGUF_TYPE_ARRAY) {
        return out;
    }
    if (gguf_get_arr_type(gf.gguf, idx) != GGUF_TYPE_STRING) {
        return out;
    }
    const size_t n = gguf_get_arr_n(gf.gguf, idx);
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        out.push_back(gguf_get_arr_str(gf.gguf, idx, (int) i));
    }
    return out;
}

static void yue2_mmsfa_parse_config(const GGUFModel & gf, Yue2MmsfaConfig * c) {
    c->sample_rate = gf_get_u32(gf, "mmsfa.sample_rate");

    c->n_conv      = gf_get_u32(gf, "mmsfa.n_conv");
    c->conv_in     = yue2_get_i32_arr(gf, "mmsfa.conv_in");
    c->conv_out    = yue2_get_i32_arr(gf, "mmsfa.conv_out");
    c->conv_kernel = yue2_get_i32_arr(gf, "mmsfa.conv_kernel");
    c->conv_stride = yue2_get_i32_arr(gf, "mmsfa.conv_stride");
    c->feat_dim    = gf_get_u32(gf, "mmsfa.feat_dim");

    c->embed            = gf_get_u32(gf, "mmsfa.embed");
    c->n_layers         = gf_get_u32(gf, "mmsfa.n_layers");
    c->n_heads          = gf_get_u32(gf, "mmsfa.n_heads");
    c->head_dim         = gf_get_u32(gf, "mmsfa.head_dim");
    c->ffn              = gf_get_u32(gf, "mmsfa.ffn");
    c->attn_scale       = gf_get_f32(gf, "mmsfa.attention_scale");
    c->ln_eps           = gf_get_f32(gf, "mmsfa.ln_eps");
    c->layer_norm_first = gf_get_bool(gf, "mmsfa.layer_norm_first");

    c->pos_conv_kernel     = gf_get_u32(gf, "mmsfa.pos_conv_kernel");
    c->pos_conv_groups     = gf_get_u32(gf, "mmsfa.pos_conv_groups");
    c->pos_conv_padding    = gf_get_u32(gf, "mmsfa.pos_conv_padding");
    c->pos_conv_num_remove = gf_get_u32(gf, "mmsfa.pos_conv_num_remove");

    c->n_labels    = gf_get_u32(gf, "mmsfa.n_labels");
    c->labels      = yue2_mmsfa_get_str_arr(gf, "mmsfa.labels");
    c->blank_label = gf_get_str(gf, "mmsfa.ctc_blank_label");
}

// Structural sanity, not taste. Every check corresponds to something pin.json
// fixed at runtime; a file that fails one of these would produce emissions
// that look like emissions and align at chance.
static void yue2_mmsfa_validate_config(const Yue2MmsfaConfig & c, std::vector<std::string> * errs) {
    auto bad = [&](const std::string & m) {
        if (errs->size() < 24) {
            errs->push_back(m);
        }
    };
    if (c.sample_rate != 16000) {
        bad("mmsfa.sample_rate is not 16000 (MMS_FA is a 16 kHz model)");
    }
    if (c.n_conv == 0 || c.conv_in.size() != c.n_conv || c.conv_out.size() != c.n_conv ||
        c.conv_kernel.size() != c.n_conv || c.conv_stride.size() != c.n_conv) {
        bad("mmsfa.conv_* arrays disagree with mmsfa.n_conv");
        return;  // everything below indexes them
    }
    if (c.conv_in[0] != 1) {
        bad("mmsfa.conv_in[0] != 1 — the frontend eats a mono waveform, not features");
    }
    if ((uint32_t) c.conv_out.back() != c.feat_dim) {
        bad("the last conv's output width != mmsfa.feat_dim");
    }
    int64_t total_stride = 1;
    for (size_t i = 0; i < c.conv_stride.size(); i++) {
        total_stride *= (int64_t) c.conv_stride[i];
        if (i > 0 && c.conv_in[i] != c.conv_out[i - 1]) {
            bad("conv layer " + std::to_string(i) + " input width != previous output width");
        }
    }
    if (total_stride != 320) {
        bad("product(mmsfa.conv_stride) is " + std::to_string(total_stride) +
            ", not 320 — the frame rate would not be 50 Hz");
    }
    if (c.embed == 0 || c.n_heads == 0 || c.head_dim == 0 || c.n_heads * c.head_dim != c.embed) {
        bad("mmsfa.n_heads * mmsfa.head_dim != mmsfa.embed");
    }
    if (c.n_layers == 0 || c.ffn == 0) {
        bad("mmsfa.n_layers / mmsfa.ffn missing");
    }
    // NOT an error: see note B. mmsfa.layer_norm_first records what pin.json
    // read off the module's config; the captured activations say pre-LN, and
    // this port builds what the activations say. The disagreement is reported
    // at load time so nobody has to rediscover it from a 19.5 max-abs.
    if (c.ln_eps <= 0.0f) {
        bad("mmsfa.ln_eps is 0 or missing");
    }
    if (c.attn_scale <= 0.0f) {
        bad("mmsfa.attention_scale is 0 or missing");
    }
    if (c.pos_conv_groups == 0 || c.embed % c.pos_conv_groups != 0) {
        bad("mmsfa.pos_conv_groups does not divide mmsfa.embed");
    }
    if (c.pos_conv_kernel == 0 || c.pos_conv_padding != c.pos_conv_kernel / 2) {
        bad("mmsfa.pos_conv_padding != pos_conv_kernel/2 — the num_remove rule below assumes that");
    }
    if (c.pos_conv_kernel % 2 == 0 && c.pos_conv_num_remove != 1) {
        // Trap 2: k even + pad k/2 gives T+1 frames, and torchaudio removes
        // exactly one from the end.
        bad("mmsfa.pos_conv_num_remove != 1 for an even kernel — the pos-conv would be one frame long");
    }
    if (c.n_labels == 0 || c.labels.size() != c.n_labels) {
        bad("mmsfa.labels does not have mmsfa.n_labels entries");
    }
    if (!c.labels.empty() && !c.blank_label.empty() && c.labels[0] != c.blank_label) {
        bad("mmsfa.labels[0] is not the blank label — CTC blank MUST be index 0");
    }
}

// ── Loader ──────────────────────────────────────────────────────────────────

static void yue2_mmsfa_free(Yue2MmsfaModel * m) {
    wctx_free(&m->wctx);
    m->w = Yue2MmsfaWeights{};
    m->tmap.clear();
    m->vram = 0;
    if (m->backend_ref) {
        backend_release(m->backend, m->cpu_backend);
        m->backend     = nullptr;
        m->cpu_backend = nullptr;
        m->backend_ref = false;
    }
}

// Load every mmsfa.* tensor out of one GGUF. 422 tensors, 1.26 GB at F32 —
// there is no partial-load mode here (unlike MERT's want_blocks): the stage
// gates run the whole chain, and so does the aligner.
static bool yue2_mmsfa_load(Yue2MmsfaModel * m, const std::string & path, std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    yue2_mmsfa_free(m);
    m->path = path;

    GGUFModel gf;
    if (!gf_load(&gf, path.c_str())) {
        if (err) {
            *err = "cannot open " + path;
        }
        return false;
    }
    m->arch    = gf_get_str(gf, "general.architecture");
    m->license = gf_get_str(gf, "mmsfa.license_attribution");
    if (m->arch != "mmsfa") {
        if (err) {
            *err = "architecture is '" + m->arch + "', expected 'mmsfa' (" + path + ")";
        }
        gf_close(&gf);
        return false;
    }

    yue2_mmsfa_parse_config(gf, &m->cfg);
    std::vector<std::string> errs;
    yue2_mmsfa_validate_config(m->cfg, &errs);
    if (!errs.empty()) {
        if (err) {
            *err = "mmsfa config rejected: " + errs[0] +
                   (errs.size() > 1 ? " (+" + std::to_string(errs.size() - 1) + " more)" : "");
        }
        gf_close(&gf);
        return false;
    }

    const Yue2MmsfaConfig & c  = m->cfg;
    const int64_t           H  = (int64_t) c.embed;
    const int64_t           F  = (int64_t) c.ffn;
    const int64_t           NL = (int64_t) c.n_layers;

    BackendPair bp = backend_init("YuE2-MMSFA");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->backend_ref = true;

    // 4 per conv layer + 4 feat_proj + 2 pos_conv + 2 enc.ln + 16 per block + 2 head.
    wctx_init(&m->wctx, (int) (4 * c.n_conv + 4 + 2 + 2 + 16 * NL + 2) + 16);
    Yue2Loader ld{ &m->wctx, &gf, &m->tmap, &errs };

    m->w.conv.assign((size_t) c.n_conv, Yue2MmsfaConvLayer{});
    for (uint32_t i = 0; i < c.n_conv && errs.empty(); i++) {
        const int64_t ic = (int64_t) c.conv_in[i];
        const int64_t oc = (int64_t) c.conv_out[i];
        const int64_t k  = (int64_t) c.conv_kernel[i];
        Yue2MmsfaConvLayer & cl = m->w.conv[i];
        char                 nm[64];
        snprintf(nm, sizeof(nm), "mmsfa.conv.%u.weight", i);
        cl.w = ld.req(nm, k, ic, oc);  // torch [oc, ic, k] -> ne (k, ic, oc)
        snprintf(nm, sizeof(nm), "mmsfa.conv.%u.bias", i);
        cl.b = ld.req(nm, oc);
        snprintf(nm, sizeof(nm), "mmsfa.conv.%u.ln.weight", i);
        cl.ln_w = ld.req(nm, oc);
        snprintf(nm, sizeof(nm), "mmsfa.conv.%u.ln.bias", i);
        cl.ln_b = ld.req(nm, oc);
    }

    const int64_t FD = (int64_t) c.feat_dim;
    m->w.fp_ln_w = ld.req("mmsfa.feat_proj.ln.weight", FD);
    m->w.fp_ln_b = ld.req("mmsfa.feat_proj.ln.bias", FD);
    m->w.fp_w    = ld.req("mmsfa.feat_proj.weight", FD, H);  // torch [H, FD] -> ne (FD, H)
    m->w.fp_b    = ld.req("mmsfa.feat_proj.bias", H);

    const int64_t PK  = (int64_t) c.pos_conv_kernel;
    const int64_t PIC = H / (int64_t) c.pos_conv_groups;
    m->w.pos_w = ld.req("mmsfa.pos_conv.weight", PK, PIC, H);  // torch [H, H/g, K]
    m->w.pos_b = ld.req("mmsfa.pos_conv.bias", H);

    m->w.enc_ln_w = ld.req("mmsfa.enc.ln.weight", H);
    m->w.enc_ln_b = ld.req("mmsfa.enc.ln.bias", H);

    m->w.blk.assign((size_t) NL, Yue2MmsfaBlock{});
    for (int64_t n = 0; n < NL && errs.empty(); n++) {
        Yue2MmsfaBlock & b = m->w.blk[(size_t) n];
        char             nm[64];
#define YUE2_MMSFA_T(field, fmtstr, ...)                 \
    do {                                                 \
        snprintf(nm, sizeof(nm), fmtstr, (int) n);       \
        b.field = ld.req(nm, __VA_ARGS__);               \
    } while (0)
        YUE2_MMSFA_T(attn_q_w, "mmsfa.blk.%d.attn_q.weight", H, H);
        YUE2_MMSFA_T(attn_q_b, "mmsfa.blk.%d.attn_q.bias", H);
        YUE2_MMSFA_T(attn_k_w, "mmsfa.blk.%d.attn_k.weight", H, H);
        YUE2_MMSFA_T(attn_k_b, "mmsfa.blk.%d.attn_k.bias", H);
        YUE2_MMSFA_T(attn_v_w, "mmsfa.blk.%d.attn_v.weight", H, H);
        YUE2_MMSFA_T(attn_v_b, "mmsfa.blk.%d.attn_v.bias", H);
        YUE2_MMSFA_T(attn_o_w, "mmsfa.blk.%d.attn_o.weight", H, H);
        YUE2_MMSFA_T(attn_o_b, "mmsfa.blk.%d.attn_o.bias", H);
        YUE2_MMSFA_T(ln1_w, "mmsfa.blk.%d.ln1.weight", H);
        YUE2_MMSFA_T(ln1_b, "mmsfa.blk.%d.ln1.bias", H);
        YUE2_MMSFA_T(ffn_up_w, "mmsfa.blk.%d.ffn_up.weight", H, F);
        YUE2_MMSFA_T(ffn_up_b, "mmsfa.blk.%d.ffn_up.bias", F);
        YUE2_MMSFA_T(ffn_dn_w, "mmsfa.blk.%d.ffn_down.weight", F, H);
        YUE2_MMSFA_T(ffn_dn_b, "mmsfa.blk.%d.ffn_down.bias", H);
        YUE2_MMSFA_T(ln2_w, "mmsfa.blk.%d.ln2.weight", H);
        YUE2_MMSFA_T(ln2_b, "mmsfa.blk.%d.ln2.bias", H);
#undef YUE2_MMSFA_T
    }

    const int64_t NLAB = (int64_t) c.n_labels;
    m->w.head_w = ld.req("mmsfa.head.weight", H, NLAB);
    m->w.head_b = ld.req("mmsfa.head.bias", NLAB);

    if (!errs.empty()) {
        if (err) {
            *err = errs[0] + (errs.size() > 1 ? " (+" + std::to_string(errs.size() - 1) + " more)" : "");
        }
        gf_close(&gf);
        yue2_mmsfa_free(m);
        return false;
    }

    if (!wctx_alloc(&m->wctx, m->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the MMS_FA weights";
        }
        gf_close(&gf);
        yue2_mmsfa_free(m);
        return false;
    }
    gf_close(&gf);  // safe: wctx_alloc has copied everything to the backend

    m->vram    = m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;
    m->load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!m->cfg.layer_norm_first) {
        // Note B. Say it out loud once per load: the KV and the fixtures
        // disagree, this port follows the fixtures, and the difference is not
        // subtle (19.5 max-abs at layer0 if you follow the KV instead).
        fprintf(stderr,
                "[YuE2-MMSFA] NOTE: mmsfa.layer_norm_first is false, but the captured activations are PRE-LN "
                "(x + attn(ln1(x))); building pre-LN. See yue2-mmsfa.h note B.\n");
    }
    fprintf(stderr, "[YuE2-MMSFA] Loaded %s: %zu tensors, %.1f MB, %.0f ms\n", yue2_basename(path).c_str(),
            m->tmap.size(), (double) m->vram / (1024.0 * 1024.0), m->load_ms);
    return true;
}

// True when every weight is stored F32 (what convert-mms-fa.py writes today).
// A narrowed file would still load; callers print which regime they are in
// rather than quietly picking a tolerance.
static bool yue2_mmsfa_weights_are_f32(const Yue2MmsfaModel & m) {
    for (const auto & kv : m.tmap) {
        if (kv.second && kv.second->type != GGML_TYPE_F32) {
            return false;
        }
    }
    return !m.tmap.empty();
}

// How many 50 Hz frames a clip of `n_samples` produces. Every conv is
// unpadded, so this is floor((L - k)/s) + 1 per layer — NOT L/320, which is
// off by a frame or two on almost every real length.
static int64_t yue2_mmsfa_frames(const Yue2MmsfaConfig & c, int64_t n_samples) {
    int64_t L = n_samples;
    for (uint32_t i = 0; i < c.n_conv; i++) {
        const int64_t k = (int64_t) c.conv_kernel[i];
        const int64_t s = (int64_t) c.conv_stride[i];
        if (L < k) {
            return 0;
        }
        L = (L - k) / s + 1;
    }
    return L;
}

// ── Graph ───────────────────────────────────────────────────────────────────

#define YUE2_MMSFA_MAX_NODES 65536

struct Yue2MmsfaOptions {
    // Per-utterance waveform normalisation (note A): one mean and one
    // variance over the WHOLE clip handed in, then (x - mean)/sqrt(var + eps).
    // ON by default, because it is part of the model's input contract — the
    // bundle does it, not the caller's audio loader — and conv0 is off by 9.8
    // without it. The eps is F.layer_norm's own default (1e-5), which is NOT
    // the same constant as mmsfa.ln_eps even though both happen to be 1e-5.
    bool  normalize_waveform = true;
    float waveform_norm_eps  = 1e-5f;

    // GELU flavour. false = exact erf, which is what the pin says and what the
    // fixtures confirm at conv0 AND at layer0. See trap 5 before flipping it.
    bool gelu_tanh = false;

    // Attention score-tensor byte budget. A head's [T_k, T_q] score block is
    // split into query chunks so that no single one exceeds this. 0 = no
    // chunking (fine below a few thousand frames, catastrophic above).
    size_t attn_score_budget = 256u << 20;

    // Tap the named stages (conv0..conv6, feat_proj, pos_conv, and the final
    // norm) and hand them back. Off in production: each tap is a live tensor
    // the allocator cannot reuse, and conv0 alone is 65 MB for a 10 s clip
    // (1.5 GB for a full track).
    bool want_stages = false;

    // Encoder layer indices whose OUTPUT to also hand back, for bisection.
    // Empty in production.
    std::vector<int> tap_layers;
};

// Everything the stage gate asks for. All [T, C] row-major — i.e. the ggml
// ne = [C, T] bytes verbatim, and the fixtures' own .npy body layout.
struct Yue2MmsfaStages {
    std::vector<std::vector<float>> conv;        // n_conv entries, [T_i, C_i]
    std::vector<float>              feat_proj;   // [T, 1024]
    std::vector<float>              pos_conv;    // [T, 1024]  the pos term ALONE
    // The FINAL norm's output, i.e. the head's input. Named for the fixture it
    // is scored against (ex_pre_layers.npy), which the plan doc mislabels —
    // note C. It is 24 layers deep, not one conv deep.
    std::vector<float>              pre_layers;  // [T, 1024]
    std::vector<std::vector<float>> layers;      // parallel to tap_layers
    std::vector<float>              logits;      // [T, 28]  pre-log_softmax
};

struct Yue2MmsfaGraph {
    ggml_backend_t       backend       = nullptr;
    ggml_backend_t       cpu_backend   = nullptr;
    bool                 backend_ref   = false;
    ggml_backend_sched_t sched         = nullptr;
    const void *         weights_token = nullptr;

    ggml_context * gctx  = nullptr;
    uint8_t *      gbuf  = nullptr;
    ggml_cgraph *  graph = nullptr;

    ggml_tensor * input  = nullptr;  // [n_samples, 1]  F32 waveform
    ggml_tensor * output = nullptr;  // [28, T]         raw logits

    std::vector<ggml_tensor *> conv_taps;  // n_conv, or empty
    ggml_tensor *              fp_tap   = nullptr;
    ggml_tensor *              pos_tap  = nullptr;
    ggml_tensor *              pre_tap  = nullptr;
    std::vector<ggml_tensor *> layer_taps;

    // What the cached graph was built for.
    int64_t          graph_samples = 0;
    bool             graph_gelu_tanh = false;
    bool             graph_want_stages = false;
    size_t           graph_budget = 0;
    std::vector<int> graph_tap_layers;

    size_t compute_bytes = 0;  // sched buffer size of the last build
    int    n_nodes       = 0;
    int    attn_chunks   = 0;  // query chunks per head in the last build
};

namespace yue2_mmsfa_detail {

// LayerNorm over ne0 (channels), then affine. w and b are 1-D [C], which
// ggml_mul/ggml_add broadcast across ne1 without a reshape.
static ggml_tensor * ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

// MMS_FA's GELU. Default (tanh_approx=false) is the exact erf form, which is
// what F.gelu / nn.GELU do with approximate='none'. ggml_gelu_erf is erff in
// f32 on the CPU backend; ggml_gelu can route through an F16 lookup table.
static ggml_tensor * gelu(ggml_context * ctx, ggml_tensor * x, bool tanh_approx) {
    return tanh_approx ? ggml_gelu(ctx, x) : ggml_gelu_erf(ctx, x);
}

// Conv1d over TIME. w [K, IC, OC], x [T, IC] (ne0 = time!), -> [T_out, OC].
// Explicit F32 im2col, never ggml_conv_1d — identical to yue2_vae_conv1d
// (yue2-vae-graph.h) and yue2-mert.h's conv1d_t, and for the same reason:
// ggml_conv_1d forces an F16 im2col and this stage is gated at 1e-4 against
// an fp32 reference.
static ggml_tensor * conv1d_t(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int pad,
                              int stride) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, /*s0*/ stride, /*s1*/ 0, pad, 0, /*d0*/ 1, 0, /*is_2D*/ false,
                                    GGML_TYPE_F32);  // [IC*K, OL, 1, 1]
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                   ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));  // [OL, OC]
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

// The 7-layer feature extractor (§1). `x` is the waveform as [n_samples, 1];
// returns the last layer's output as [C, T] (channel fastest).
//
// Per layer: Conv1d(no padding) -> LayerNorm over CHANNELS -> GELU. The
// transposes are the reference's own (LayerNormConvLayer transposes to norm
// the channel axis and transposes back); this file keeps the post-LN value in
// [C, T] and transposes back only to feed the next conv.
static ggml_tensor * conv_frontend(ggml_context * ctx, const Yue2MmsfaConfig & c, const Yue2MmsfaWeights & w,
                                   ggml_tensor * x, bool tanh_gelu, std::vector<ggml_tensor *> * taps) {
    ggml_tensor * xc = nullptr;
    for (uint32_t i = 0; i < c.n_conv; i++) {
        if (i > 0) {
            x = ggml_cont(ctx, ggml_transpose(ctx, xc));  // [C, T] -> [T, C]
        }
        x  = conv1d_t(ctx, w.conv[i].w, w.conv[i].b, x, /*pad*/ 0, (int) c.conv_stride[i]);  // [T', OC]
        xc = ggml_cont(ctx, ggml_transpose(ctx, x));                                        // [OC, T']
        xc = ln(ctx, xc, w.conv[i].ln_w, w.conv[i].ln_b, c.ln_eps);  // over CHANNELS — trap 1
        xc = gelu(ctx, xc, tanh_gelu);
        if (taps) {
            char nm[32];
            snprintf(nm, sizeof(nm), "mmsfa_conv%u", i);
            ggml_set_name(xc, nm);
            ggml_set_output(xc);
            taps->push_back(xc);
        }
    }
    return xc;
}

// The grouped positional conv (§1, trap 2 + trap 3). `x` is [H, T]; returns
// the pos term alone, post-GELU and post-trim, as [H, T].
//
// GROUPS: ggml has no grouped convolution — neither ggml_conv_1d nor
// ggml_im2col takes a group count, and im2col's own assert (b->ne[1] ==
// a->ne[1]) demands the kernel's input width equal the data's. So this runs
// `groups` independent convs over contiguous 64-channel slices of both the
// data and the weight, and concatenates. Both slices are plain views with no
// copy: in [T, C] layout a channel range is a contiguous byte range, and the
// weight's ne2 (output channel) slice is likewise contiguous.
static ggml_tensor * pos_conv(ggml_context * ctx, const Yue2MmsfaConfig & c, const Yue2MmsfaWeights & w,
                              ggml_tensor * x, bool tanh_gelu) {
    const int64_t H  = (int64_t) c.embed;
    const int64_t T  = x->ne[1];
    const int64_t NG = (int64_t) c.pos_conv_groups;
    const int64_t CG = H / NG;  // channels per group, in AND out

    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));  // [T, H]

    ggml_tensor * acc = nullptr;
    for (int64_t g = 0; g < NG; g++) {
        ggml_tensor * xg = ggml_view_2d(ctx, xt, T, CG, xt->nb[1], (size_t) (g * CG) * xt->nb[1]);
        ggml_tensor * wg = ggml_view_3d(ctx, w.pos_w, w.pos_w->ne[0], CG, CG, w.pos_w->nb[1], w.pos_w->nb[2],
                                        (size_t) (g * CG) * w.pos_w->nb[2]);
        // No bias here: the conv bias is [H] across all groups, added once below.
        ggml_tensor * yg = conv1d_t(ctx, wg, nullptr, xg, (int) c.pos_conv_padding, /*stride*/ 1);  // [T+1, CG]
        acc = acc ? ggml_concat(ctx, acc, yg, 1) : yg;
    }
    acc = ggml_add(ctx, acc, ggml_reshape_2d(ctx, w.pos_b, 1, w.pos_b->ne[0]));  // [T+1, H]

    // Trap 2: k even + pad k/2 yields T+1 frames; drop num_remove from the END.
    const int64_t keep = acc->ne[0] - (int64_t) c.pos_conv_num_remove;
    acc = ggml_cont(ctx, ggml_view_2d(ctx, acc, keep, H, acc->nb[1], 0));  // [T, H]
    acc = gelu(ctx, acc, tanh_gelu);
    return ggml_cont(ctx, ggml_transpose(ctx, acc));  // [H, T]
}

// Self-attention, 16 heads x 64, four separate projections all with bias, no
// positional term and no mask. `x` is [H, T]; returns out_proj's result,
// [H, T]; the caller owns the residual and the LayerNorm after it.
//
// Built ONE HEAD AT A TIME, and within a head in query chunks of `q_chunk`
// frames — see the file header's memory note. Arithmetically identical to the
// unchunked form: ggml_soft_max_ext normalises over ne0, which here is the
// key axis, so each query row is independent of every other.
static ggml_tensor * attn(ggml_context * ctx, const Yue2MmsfaConfig & c, const Yue2MmsfaBlock & b, ggml_tensor * x,
                          int64_t q_chunk) {
    const int64_t H  = (int64_t) c.embed;
    const int64_t HD = (int64_t) c.head_dim;
    const int64_t NH = (int64_t) c.n_heads;
    const int64_t T  = x->ne[1];

    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, b.attn_q_w, x), b.attn_q_b);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, b.attn_k_w, x), b.attn_k_b);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, b.attn_v_w, x), b.attn_v_b);

    // [H, T] -> [HD, T, NH]: per-head slices become contiguous views.
    ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, HD, NH, T), 0, 2, 1, 3));
    ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, HD, NH, T), 0, 2, 1, 3));
    // v wants [T, HD, NH] so that mul_mat(v_head, scores) contracts over time.
    ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, HD, NH, T), 1, 2, 0, 3));

    if (q_chunk <= 0 || q_chunk > T) {
        q_chunk = T;
    }

    ggml_tensor * heads = nullptr;  // [HD, T, n_done]
    for (int64_t h = 0; h < NH; h++) {
        ggml_tensor * k_h = ggml_view_2d(ctx, kp, HD, T, kp->nb[1], (size_t) h * kp->nb[2]);
        ggml_tensor * v_h = ggml_view_2d(ctx, vp, T, HD, vp->nb[1], (size_t) h * vp->nb[2]);

        ggml_tensor * out_h = nullptr;
        for (int64_t c0 = 0; c0 < T; c0 += q_chunk) {
            const int64_t tc  = std::min(q_chunk, T - c0);
            ggml_tensor * q_c = ggml_view_2d(ctx, qp, HD, tc, qp->nb[1],
                                             (size_t) h * qp->nb[2] + (size_t) c0 * qp->nb[1]);
            ggml_tensor * kq  = ggml_mul_mat(ctx, k_h, q_c);  // [T_k, tc]
            kq = ggml_soft_max_ext(ctx, kq, /*mask*/ nullptr, c.attn_scale, /*max_bias*/ 0.0f);
            ggml_tensor * oc = ggml_mul_mat(ctx, v_h, kq);  // [HD, tc]
            out_h = out_h ? ggml_concat(ctx, out_h, oc, 1) : oc;
        }
        ggml_tensor * h3 = ggml_reshape_3d(ctx, out_h, HD, T, 1);
        heads            = heads ? ggml_concat(ctx, heads, h3, 2) : h3;
    }

    ggml_tensor * att = ggml_cont_2d(ctx, ggml_permute(ctx, heads, 0, 2, 1, 3), H, T);  // [HD,NH,T] -> [H, T]
    return ggml_add(ctx, ggml_mul_mat(ctx, b.attn_o_w, att), b.attn_o_b);
}

// FeedForward: down(gelu(up(x))), 1024 -> 4096 -> 1024, both with bias.
static ggml_tensor * ffn(ggml_context * ctx, const Yue2MmsfaBlock & b, ggml_tensor * x, bool tanh_gelu) {
    ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, b.ffn_up_w, x), b.ffn_up_b);
    h               = gelu(ctx, h, tanh_gelu);
    return ggml_add(ctx, ggml_mul_mat(ctx, b.ffn_dn_w, h), b.ffn_dn_b);
}

// One PRE-LN encoder layer (note B):
//     x = x + attn(ln1(x));   x = x + ffn(ln2(x))
// Both norms are INPUT norms; neither sits on a residual sum. Post-LN with
// these same tensors scores 19.5 max-abs at layer0 against a 2e-4 bar.
static ggml_tensor * block(ggml_context * ctx, const Yue2MmsfaConfig & c, const Yue2MmsfaBlock & b, ggml_tensor * x,
                           bool tanh_gelu, int64_t q_chunk) {
    ggml_tensor * a = ggml_add(ctx, x, attn(ctx, c, b, ln(ctx, x, b.ln1_w, b.ln1_b, c.ln_eps), q_chunk));
    return ggml_add(ctx, a, ffn(ctx, b, ln(ctx, a, b.ln2_w, b.ln2_b, c.ln_eps), tanh_gelu));
}

}  // namespace yue2_mmsfa_detail

static void yue2_mmsfa_free_graph(Yue2MmsfaGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx   = nullptr;
    g->gbuf   = nullptr;
    g->graph  = nullptr;
    g->input  = nullptr;
    g->output = nullptr;
    g->conv_taps.clear();
    g->layer_taps.clear();
    g->fp_tap        = nullptr;
    g->pos_tap       = nullptr;
    g->pre_tap       = nullptr;
    g->graph_samples = 0;
    g->graph_tap_layers.clear();
}

static void yue2_mmsfa_graph_free(Yue2MmsfaGraph * g) {
    yue2_mmsfa_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static bool yue2_mmsfa_prepare(const Yue2MmsfaModel & m, Yue2MmsfaGraph * g, std::string * err) {
    const void * token = m.wctx.buffer;
    if (g->sched && g->weights_token == token) {
        return true;
    }
    yue2_mmsfa_graph_free(g);

    BackendPair bp = backend_init("YuE2-MMSFA-graph");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;
    g->sched       = backend_sched_new(bp, YUE2_MMSFA_MAX_NODES * 2);
    if (!g->sched) {
        if (err) {
            *err = "ggml_backend_sched_new failed for the MMS_FA graph";
        }
        yue2_mmsfa_graph_free(g);
        return false;
    }
    g->weights_token = token;
    return true;
}

// Query-chunk size for a given frame count: the largest chunk whose
// [T_k, T_q] F32 score block fits the byte budget, floored at 64 frames so a
// pathological budget cannot explode the node count.
static int64_t yue2_mmsfa_q_chunk(int64_t T, size_t budget) {
    if (budget == 0) {
        return T;
    }
    const int64_t per_q = T * (int64_t) sizeof(float);
    int64_t       chunk = per_q > 0 ? (int64_t) (budget / (size_t) per_q) : T;
    chunk               = std::max<int64_t>(chunk, 64);
    return std::min<int64_t>(chunk, T);
}

static bool yue2_mmsfa_ensure_graph(const Yue2MmsfaModel & m, Yue2MmsfaGraph * g, int64_t n_samples,
                                    const Yue2MmsfaOptions & opt, std::string * err) {
    if (g->gctx && g->graph_samples == n_samples && g->graph_gelu_tanh == opt.gelu_tanh &&
        g->graph_want_stages == opt.want_stages && g->graph_budget == opt.attn_score_budget &&
        g->graph_tap_layers == opt.tap_layers) {
        return true;
    }
    yue2_mmsfa_free_graph(g);

    const Yue2MmsfaConfig & c = m.cfg;
    const int64_t           T = yue2_mmsfa_frames(c, n_samples);
    if (T <= 0) {
        if (err) {
            *err = "clip is " + std::to_string(n_samples) +
                   " samples; the feature extractor needs at least 400 to produce one frame";
        }
        return false;
    }

    const size_t ctx_bytes = ggml_tensor_overhead() * (YUE2_MMSFA_MAX_NODES + 256) +
                             ggml_graph_overhead_custom(YUE2_MMSFA_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the MMS_FA graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the MMS_FA graph context";
        }
        return false;
    }

    using namespace yue2_mmsfa_detail;

    // The waveform as [n_samples, 1]: ne0 = time, ne1 = the single input
    // channel, which is what conv1d_t/im2col want.
    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(g->input, "mmsfa_pcm16k");
    ggml_set_input(g->input);

    std::vector<ggml_tensor *> conv_taps;
    ggml_tensor *              x =
        conv_frontend(ctx, c, m.w, g->input, opt.gelu_tanh, opt.want_stages ? &conv_taps : nullptr);  // [512, T]
    g->conv_taps = conv_taps;

    // Feature projection: LayerNorm(512) -> Linear 512->1024 (+bias).
    ggml_tensor * h = ln(ctx, x, m.w.fp_ln_w, m.w.fp_ln_b, c.ln_eps);
    h               = ggml_add(ctx, ggml_mul_mat(ctx, m.w.fp_w, h), m.w.fp_b);  // [1024, T]
    if (opt.want_stages) {
        ggml_set_name(h, "mmsfa_feat_proj");
        ggml_set_output(h);
        g->fp_tap = h;
    }

    ggml_tensor * pos = pos_conv(ctx, c, m.w, h, opt.gelu_tanh);  // [1024, T]
    if (opt.want_stages) {
        ggml_set_name(pos, "mmsfa_pos_conv");
        ggml_set_output(pos);
        g->pos_tap = pos;
    }

    // The transformer's input is the bare residual sum — NO norm here (note C).
    ggml_tensor * e = ggml_add(ctx, h, pos);

    const int64_t q_chunk = yue2_mmsfa_q_chunk(T, opt.attn_score_budget);
    g->attn_chunks        = (int) ((T + q_chunk - 1) / q_chunk);

    g->layer_taps.assign(opt.tap_layers.size(), nullptr);
    for (size_t l = 0; l < m.w.blk.size(); l++) {
        e = block(ctx, c, m.w.blk[l], e, opt.gelu_tanh, q_chunk);
        for (size_t ti = 0; ti < opt.tap_layers.size(); ti++) {
            if (opt.tap_layers[ti] >= 0 && (size_t) opt.tap_layers[ti] == l) {
                char nm[40];
                snprintf(nm, sizeof(nm), "mmsfa_layer%02d", (int) l);
                ggml_set_name(e, nm);
                ggml_set_output(e);
                g->layer_taps[ti] = e;
            }
        }
    }

    // The FINAL norm, after all 24 layers (note C). This is the tensor the
    // fixture set calls ex_pre_layers.npy — the head's input, not the
    // transformer's.
    e = ln(ctx, e, m.w.enc_ln_w, m.w.enc_ln_b, c.ln_eps);
    if (opt.want_stages) {
        ggml_set_name(e, "mmsfa_final_ln");
        ggml_set_output(e);
        g->pre_tap = e;
    }

    // The CTC head: raw logits. log_softmax is applied on the host in double
    // (yue2_mmsfa_emissions) — the head's own KV note says the weights do not
    // carry it, and a graph-side exp/log round trip in F32 buys nothing.
    g->output = ggml_add(ctx, ggml_mul_mat(ctx, m.w.head_w, e), m.w.head_b);  // [28, T]
    ggml_set_name(g->output, "mmsfa_logits");
    ggml_set_output(g->output);

    g->graph = ggml_new_graph_custom(ctx, YUE2_MMSFA_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->output);
    for (ggml_tensor * t : g->conv_taps) {
        ggml_build_forward_expand(g->graph, t);
    }
    for (ggml_tensor * t : { g->fp_tap, g->pos_tap, g->pre_tap }) {
        if (t) {
            ggml_build_forward_expand(g->graph, t);
        }
    }
    for (ggml_tensor * t : g->layer_taps) {
        if (t) {
            ggml_build_forward_expand(g->graph, t);
        }
    }

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf   = nullptr;
        g->graph  = nullptr;
        g->output = nullptr;
        if (err) {
            *err = "MMS_FA graph allocation failed (out of memory?) for " + std::to_string(n_samples) +
                   " samples / " + std::to_string(T) + " frames";
        }
        return false;
    }

    g->gctx              = ctx;
    g->graph_samples     = n_samples;
    g->graph_gelu_tanh   = opt.gelu_tanh;
    g->graph_want_stages = opt.want_stages;
    g->graph_budget      = opt.attn_score_budget;
    g->graph_tap_layers  = opt.tap_layers;
    g->n_nodes           = ggml_graph_n_nodes(g->graph);
    g->compute_bytes     = ggml_backend_sched_get_buffer_size(g->sched, g->backend);
    fprintf(stderr,
            "[YuE2-MMSFA] Graph: %lld samples -> %lld frames, %zu layers, %d nodes, %d splits, "
            "attn %d chunk(s)/head, compute %.0f MB\n",
            (long long) n_samples, (long long) T, m.w.blk.size(), g->n_nodes,
            ggml_backend_sched_get_n_splits(g->sched), g->attn_chunks, (double) g->compute_bytes / (1024.0 * 1024.0));
    return true;
}

namespace yue2_mmsfa_detail {

// Pull a graph tensor back as row-major [ne1, ne0] floats — which, for the
// [C, T] activations in this file, IS the fixtures' [T, C] layout.
static void fetch(ggml_tensor * t, std::vector<float> * out) {
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out->data(), 0, out->size() * sizeof(float));
}

}  // namespace yue2_mmsfa_detail

// ── The whole chain: waveform -> emissions ──────────────────────────────────
//
// `pcm16k` is mono 16 kHz, `n_samples` long. `out` is resized to T * n_labels,
// row-major [T, 28] log-probabilities at 50 Hz — exactly the layout of
// K:/yue2/fixtures/mms-fa/01_emissions.npy, and exactly what
// yue2_ctc_forced_align() reads.
//
// log_softmax runs on the host in double and subtracts the row max first, so
// it is strictly more accurate than the reference's own F32
// `torch.log_softmax`, not differently accurate.
//
// Not thread-safe: one cached graph per Yue2MmsfaGraph (same contract as
// yue2_mert_encode_chunk).
static bool yue2_mmsfa_emissions(const Yue2MmsfaModel & m, Yue2MmsfaGraph * g, const float * pcm16k, int64_t n_samples,
                                 std::vector<float> * out, const Yue2MmsfaOptions & opt = Yue2MmsfaOptions{},
                                 Yue2MmsfaStages * stages = nullptr, std::string * err = nullptr) {
    if (!pcm16k || n_samples <= 0) {
        if (err) {
            *err = "no audio handed to yue2_mmsfa_emissions";
        }
        return false;
    }
    if (!yue2_mmsfa_prepare(m, g, err)) {
        return false;
    }
    if (!yue2_mmsfa_ensure_graph(m, g, n_samples, opt, err)) {
        return false;
    }

    // Note A: F.layer_norm over the whole utterance, before anything else.
    // Mean and variance in double (the reference is fp32, so this is strictly
    // more accurate, not differently accurate); population variance, matching
    // torch. A track normalised in pieces is a different model input, so this
    // happens here — once, over exactly the samples handed in — and not in
    // whatever loaded the audio.
    std::vector<float> norm;
    const float *      feed = pcm16k;
    if (opt.normalize_waveform) {
        double sum = 0.0;
        for (int64_t i = 0; i < n_samples; i++) {
            sum += (double) pcm16k[i];
        }
        const double mean = sum / (double) n_samples;
        double       var  = 0.0;
        for (int64_t i = 0; i < n_samples; i++) {
            const double d = (double) pcm16k[i] - mean;
            var += d * d;
        }
        var /= (double) n_samples;
        const double inv = 1.0 / std::sqrt(var + (double) opt.waveform_norm_eps);
        norm.resize((size_t) n_samples);
        for (int64_t i = 0; i < n_samples; i++) {
            norm[(size_t) i] = (float) (((double) pcm16k[i] - mean) * inv);
        }
        feed = norm.data();
    }

    ggml_backend_tensor_set(g->input, feed, 0, ggml_nbytes(g->input));
    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "MMS_FA graph compute failed";
        }
        return false;
    }

    const int64_t NL = g->output->ne[0];  // n_labels
    const int64_t T  = g->output->ne[1];

    std::vector<float> logits;
    yue2_mmsfa_detail::fetch(g->output, &logits);

    if (stages) {
        *stages = Yue2MmsfaStages{};
        stages->conv.assign(g->conv_taps.size(), {});
        for (size_t i = 0; i < g->conv_taps.size(); i++) {
            yue2_mmsfa_detail::fetch(g->conv_taps[i], &stages->conv[i]);
        }
        if (g->fp_tap) {
            yue2_mmsfa_detail::fetch(g->fp_tap, &stages->feat_proj);
        }
        if (g->pos_tap) {
            yue2_mmsfa_detail::fetch(g->pos_tap, &stages->pos_conv);
        }
        if (g->pre_tap) {
            yue2_mmsfa_detail::fetch(g->pre_tap, &stages->pre_layers);
        }
        stages->layers.assign(g->layer_taps.size(), {});
        for (size_t i = 0; i < g->layer_taps.size(); i++) {
            if (g->layer_taps[i]) {
                yue2_mmsfa_detail::fetch(g->layer_taps[i], &stages->layers[i]);
            }
        }
        stages->logits = logits;
    }

    out->assign((size_t) (T * NL), 0.0f);
    for (int64_t t = 0; t < T; t++) {
        const float * row = logits.data() + t * NL;
        double        mx  = (double) row[0];
        for (int64_t i = 1; i < NL; i++) {
            mx = std::max(mx, (double) row[i]);
        }
        double sum = 0.0;
        for (int64_t i = 0; i < NL; i++) {
            sum += std::exp((double) row[i] - mx);
        }
        const double lse = mx + std::log(sum);
        float *      dst = out->data() + t * NL;
        for (int64_t i = 0; i < NL; i++) {
            dst[i] = (float) ((double) row[i] - lse);
        }
    }
    return true;
}
