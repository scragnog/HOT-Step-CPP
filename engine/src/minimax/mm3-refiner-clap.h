#pragma once
// minimax/mm3-refiner-clap.h — CLAP audio tower for the MM3 latent refiner.
//
// HOT-Step file. The refiner (terminusresearch minimax-music3-latent-refiner
// v0.10) conditions every window on one pooled laion/larger_clap_music audio
// vector. This file is that style source end to end, the chain in the package's
// MiniMaxMusic3RefinerPipeline._clap_style:
//
//   44.1 kHz stereo window -> channel mean -> centre crop of at most 10 s
//   -> torchaudio sinc resample to 48 kHz -> ClapFeatureExtractor
//   (rand_trunc, repeatpad, slaney log-mel, 64 bins, 1001 frames) -> HTSAT
//   (batch norm, bicubic time stretch to 1024, 256x256 image, 4x4 patches,
//   4 Swin stages of 2/2/12/2 blocks, window 8, patch merging) -> LayerNorm
//   -> mean over tokens -> projection 1024->512 (ReLU) -> L2 normalise
//
// Weights: mm3-refiner-clap-f32.gguf (arch "mm3-refiner-clap") from
// tools/mm3-refiner/convert.py: the audio tower and projection only, torch
// names and shapes unchanged. No feature fusion (this checkpoint has
// enable_fusion false, the stock path only ever feeds input_features), no text
// tower, no random crop: a clip never exceeds 10 s here, so rand_trunc never
// draws a random number.
//
// Traps, each checked against the slice-1 fixtures:
// - torchaudio resamples in float32, so its 48 kHz audio differs from an exact
//   one by 1-2 ulp (1e-7). The mel stage turns that into up to 4e-3 dB in the
//   top, near-silent bins of band-limited material; the stage fed the reference
//   audio itself is exact to 8e-6 dB, and the final vector moves by 1e-7.
// - The processor works in float64 and takes a 1024-point periodic Hann
//   window, reflect padding of 512 each side and the slaney mel bank; its
//   dB floor is 1e-10 on the mel power.
// - Short clips are tiled floor(480000 / n) times and then zero padded (repeatpad).
// - The 1001 x 64 mel is stretched to 1024 frames with a bicubic
//   (A = -0.75, align_corners) pass over time only, then folded into 4 stacked
//   64 x 256 strips: row = (t / 256) * 64 + bin, column = t % 256.
// - Swin blocks of odd index shift by 4 where the stage resolution exceeds the
//   window (stages 0-2); the last stage is one 8x8 window with no shift.
// - GELU is the exact erf form. The pooled vector is L2 normalised after the
//   projection (ClapModel.get_audio_features).
//
// Gated by mm3-refiner-clap-probe / tools/mm3-refiner/check-clap.ps1.

// backend.h needs <string> first.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"
#include "yyjson.h"

static constexpr int     MM3_CLAP_SR_IN     = 44100;
static constexpr int     MM3_CLAP_SR        = 48000;
static constexpr int     MM3_CLAP_MAX_SAMP  = 480000;  // nb_max_samples: 10 s at 48 kHz
static constexpr int     MM3_CLAP_FFT       = 1024;
static constexpr int     MM3_CLAP_HOP       = 480;
static constexpr int     MM3_CLAP_BINS      = 513;
static constexpr int     MM3_CLAP_MELS      = 64;
static constexpr int     MM3_CLAP_FRAMES    = 1001;  // 1 + 480000 / 480
static constexpr int     MM3_CLAP_STRETCH   = 1024;  // spec_size * (spec_size / num_mel_bins)
static constexpr int     MM3_CLAP_IMG       = 256;   // spec_size
static constexpr int     MM3_CLAP_PATCH     = 4;
static constexpr int     MM3_CLAP_GRID      = 64;    // IMG / PATCH
static constexpr int     MM3_CLAP_STAGES    = 4;
static constexpr int     MM3_CLAP_WIN       = 8;
static constexpr int     MM3_CLAP_WTOK      = 64;    // WIN * WIN
static constexpr int64_t MM3_CLAP_DIM0      = 128;   // patch_embeds_hidden_size
static constexpr int64_t MM3_CLAP_DIM_OUT   = 1024;  // hidden_size
static constexpr int64_t MM3_CLAP_PROJ      = 512;
static constexpr float   MM3_CLAP_EPS       = 1e-5f;
static constexpr int     MM3_CLAP_MAX_NODES = 4096;
static constexpr float   MM3_CLAP_PI        = 3.14159265358979323846f;

static const int mm3_clap_depths[MM3_CLAP_STAGES] = { 2, 2, 12, 2 };
static const int mm3_clap_heads[MM3_CLAP_STAGES]  = { 4, 8, 16, 32 };

struct Mm3ClapBlock {
    ggml_tensor *ln1_w, *ln1_b, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    ggml_tensor *ln2_w, *ln2_b, *up_w, *up_b, *down_w, *down_b;
    ggml_tensor * rel_bias;  // [64 keys, 64 queries, heads], gathered from the position table at load
};

struct Mm3ClapStage {
    std::vector<Mm3ClapBlock> block;
    ggml_tensor *             merge_ln_w = nullptr, *merge_ln_b = nullptr, *merge_w = nullptr;
};

struct Mm3Clap {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            wctx        = {};

    ggml_tensor * patch_w    = nullptr;  // ne [4, 4, 1, 128]
    ggml_tensor * patch_b    = nullptr;
    ggml_tensor * patch_ln_w = nullptr;
    ggml_tensor * patch_ln_b = nullptr;
    ggml_tensor * norm_w     = nullptr;
    ggml_tensor * norm_b     = nullptr;
    ggml_tensor * proj1_w    = nullptr;
    ggml_tensor * proj1_b    = nullptr;
    ggml_tensor * proj2_w    = nullptr;
    ggml_tensor * proj2_b    = nullptr;
    Mm3ClapStage  stage[MM3_CLAP_STAGES];

    std::vector<double> bn_scale;  // eval batch norm folded per mel bin: gamma / sqrt(var + eps)
    std::vector<double> bn_shift;  // beta - mean * scale
};

static void mm3_clap_free(Mm3Clap * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    wctx_free(&m->wctx);
    if (m->backend) {
        backend_release(m->backend, m->cpu_backend);
    }
    *m = Mm3Clap{};
}

// The pinned config must be the HTSAT audio tower this graph implements.
static bool mm3_clap_check_config(const char * json, std::string * err) {
    yyjson_doc * doc = json ? yyjson_read(json, strlen(json), 0) : nullptr;
    if (!doc) {
        *err = "mm3.refiner.config missing or not JSON";
        return false;
    }
    yyjson_val * root  = yyjson_doc_get_root(doc);
    yyjson_val * audio = yyjson_obj_get(root, "audio_config");
    auto         num   = [&](yyjson_val * o, const char * k, double dflt) {
        yyjson_val * v = yyjson_obj_get(o, k);
        return v ? yyjson_get_num(v) : dflt;
    };
    const char * act = yyjson_get_str(yyjson_obj_get(root, "projection_hidden_act"));
    bool ok = audio && num(root, "projection_dim", 0) == MM3_CLAP_PROJ && act && std::string(act) == "relu" &&
              num(audio, "hidden_size", 0) == MM3_CLAP_DIM_OUT &&
              num(audio, "patch_embeds_hidden_size", 0) == MM3_CLAP_DIM0 &&
              !yyjson_get_bool(yyjson_obj_get(audio, "enable_fusion")) &&
              num(audio, "window_size", MM3_CLAP_WIN) == MM3_CLAP_WIN &&
              num(audio, "num_mel_bins", MM3_CLAP_MELS) == MM3_CLAP_MELS &&
              num(audio, "spec_size", MM3_CLAP_IMG) == MM3_CLAP_IMG &&
              num(audio, "patch_size", MM3_CLAP_PATCH) == MM3_CLAP_PATCH &&
              num(audio, "mlp_ratio", 4.0) == 4.0 && num(audio, "layer_norm_eps", 1e-5) == 1e-5;
    yyjson_val * depths = ok ? yyjson_obj_get(audio, "depths") : nullptr;
    ok                  = ok && depths && yyjson_arr_size(depths) == MM3_CLAP_STAGES;
    for (int i = 0; ok && i < MM3_CLAP_STAGES; i++) {
        ok = yyjson_get_num(yyjson_arr_get(depths, i)) == mm3_clap_depths[i];
    }
    yyjson_val * heads = ok ? yyjson_obj_get(audio, "num_attention_heads") : nullptr;
    if (heads) {  // absent means the transformers default [4, 8, 16, 32]
        ok = yyjson_arr_size(heads) == MM3_CLAP_STAGES;
        for (int i = 0; ok && i < MM3_CLAP_STAGES; i++) {
            ok = yyjson_get_num(yyjson_arr_get(heads, i)) == mm3_clap_heads[i];
        }
    }
    yyjson_doc_free(doc);
    if (!ok) {
        *err = "CLAP config differs from the larger_clap_music audio tower this port implements";
    }
    return ok;
}

// path: mm3-refiner-clap-f32.gguf.
static bool mm3_clap_load(Mm3Clap * m, const std::string & path, std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    mm3_clap_free(m);
    GGUFModel gf;
    if (!gf_load(&gf, path.c_str())) {
        *err = "cannot open " + path;
        return false;
    }
    struct Close {
        GGUFModel * g;
        ~Close() { gf_close(g); }
    } close_gf{ &gf };
    auto str = [&](const char * k) {
        int64_t i = gguf_find_key(gf.gguf, k);
        return i >= 0 && gguf_get_kv_type(gf.gguf, i) == GGUF_TYPE_STRING ? gguf_get_val_str(gf.gguf, i) : nullptr;
    };
    const char * arch = str("general.architecture");
    if (!arch || std::string(arch) != "mm3-refiner-clap") {
        *err = path + ": architecture is not mm3-refiner-clap";
        return false;
    }
    if (!mm3_clap_check_config(str("mm3.refiner.config"), err)) {
        return false;
    }

    BackendPair bp = backend_init("MM3-CLAP");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    wctx_init(&m->wctx, 400);

    // ne is the reversed torch shape; every weight must be F32 with exactly it.
    auto find = [&](const std::string & name) -> ggml_tensor * {
        return gguf_find_tensor(gf.gguf, name.c_str()) < 0 ? nullptr : ggml_get_tensor(gf.meta, name.c_str());
    };
    auto req = [&](const std::string & name, std::initializer_list<int64_t> ne) -> ggml_tensor * {
        if (!err->empty()) {
            return nullptr;
        }
        ggml_tensor * src = find(name);
        bool          ok  = src && src->type == GGML_TYPE_F32 && ggml_n_dims(src) <= (int) ne.size();
        int           d   = 0;
        for (int64_t n : ne) {
            ok = ok && src->ne[d++] == n;
        }
        if (!ok) {
            *err = "tensor " + name + " missing or wrong shape/type in " + path;
            return nullptr;
        }
        return gf_load_tensor(&m->wctx, gf, name);
    };
    auto host = [&](const std::string & name) {  // F32 payload, valid while gf is open
        return (const float *) (gf.mapping + gf.data_offset +
                                gguf_get_tensor_offset(gf.gguf, gguf_find_tensor(gf.gguf, name.c_str())));
    };
    err->clear();
    const std::string enc = "audio_model.audio_encoder.";

    // Eval batch norm over the 64 mel bins, folded on the host in double.
    for (const char * n : { "weight", "bias", "running_mean", "running_var" }) {
        ggml_tensor * t = err->empty() ? find(enc + "batch_norm." + n) : nullptr;
        if (err->empty() && (!t || t->type != GGML_TYPE_F32 || t->ne[0] != MM3_CLAP_MELS || ggml_n_dims(t) != 1)) {
            *err = std::string("tensor ") + enc + "batch_norm." + n + " missing or wrong shape/type in " + path;
        }
    }
    if (err->empty()) {
        const float *w = host(enc + "batch_norm.weight"), *b = host(enc + "batch_norm.bias");
        const float *mu = host(enc + "batch_norm.running_mean"), *var = host(enc + "batch_norm.running_var");
        m->bn_scale.resize(MM3_CLAP_MELS);
        m->bn_shift.resize(MM3_CLAP_MELS);
        for (int i = 0; i < MM3_CLAP_MELS; i++) {
            m->bn_scale[i] = (double) w[i] / std::sqrt((double) var[i] + 1e-5);
            m->bn_shift[i] = (double) b[i] - (double) mu[i] * m->bn_scale[i];
        }
    }

    m->patch_w    = req(enc + "patch_embed.proj.weight", { MM3_CLAP_PATCH, MM3_CLAP_PATCH, 1, MM3_CLAP_DIM0 });
    m->patch_b    = req(enc + "patch_embed.proj.bias", { MM3_CLAP_DIM0 });
    m->patch_ln_w = req(enc + "patch_embed.norm.weight", { MM3_CLAP_DIM0 });
    m->patch_ln_b = req(enc + "patch_embed.norm.bias", { MM3_CLAP_DIM0 });
    m->norm_w     = req(enc + "norm.weight", { MM3_CLAP_DIM_OUT });
    m->norm_b     = req(enc + "norm.bias", { MM3_CLAP_DIM_OUT });
    m->proj1_w    = req("audio_projection.linear1.weight", { MM3_CLAP_DIM_OUT, MM3_CLAP_PROJ });
    m->proj1_b    = req("audio_projection.linear1.bias", { MM3_CLAP_PROJ });
    m->proj2_w    = req("audio_projection.linear2.weight", { MM3_CLAP_PROJ, MM3_CLAP_PROJ });
    m->proj2_b    = req("audio_projection.linear2.bias", { MM3_CLAP_PROJ });
    for (int s = 0; s < MM3_CLAP_STAGES; s++) {
        const int64_t C = MM3_CLAP_DIM0 << s, H = mm3_clap_heads[s];
        Mm3ClapStage & S = m->stage[s];
        S.block.resize((size_t) mm3_clap_depths[s]);
        for (int b = 0; b < mm3_clap_depths[s]; b++) {
            const std::string p = enc + "layers." + std::to_string(s) + ".blocks." + std::to_string(b) + ".";
            Mm3ClapBlock &    B = S.block[(size_t) b];
            B.ln1_w  = req(p + "layernorm_before.weight", { C });
            B.ln1_b  = req(p + "layernorm_before.bias", { C });
            B.q_w    = req(p + "attention.self.query.weight", { C, C });
            B.q_b    = req(p + "attention.self.query.bias", { C });
            B.k_w    = req(p + "attention.self.key.weight", { C, C });
            B.k_b    = req(p + "attention.self.key.bias", { C });
            B.v_w    = req(p + "attention.self.value.weight", { C, C });
            B.v_b    = req(p + "attention.self.value.bias", { C });
            B.o_w    = req(p + "attention.output.dense.weight", { C, C });
            B.o_b    = req(p + "attention.output.dense.bias", { C });
            B.ln2_w  = req(p + "layernorm_after.weight", { C });
            B.ln2_b  = req(p + "layernorm_after.bias", { C });
            B.up_w   = req(p + "intermediate.dense.weight", { C, 4 * C });
            B.up_b   = req(p + "intermediate.dense.bias", { 4 * C });
            B.down_w = req(p + "output.dense.weight", { 4 * C, C });
            B.down_b = req(p + "output.dense.bias", { C });

            // Relative position bias: table [(2w-1)^2, heads] indexed per (query, key) pair of one
            // 8x8 window, stored as [key, query, head] so it adds straight onto the scores.
            const std::string tn = p + "attention.self.relative_position_bias_table";
            ggml_tensor *     tt = err->empty() ? find(tn) : nullptr;
            if (err->empty() && (!tt || tt->type != GGML_TYPE_F32 || tt->ne[0] != H || tt->ne[1] != 15 * 15)) {
                *err = "tensor " + tn + " missing or wrong shape/type in " + path;
            }
            if (err->empty()) {
                const float * table  = host(tn);
                auto          folded = std::make_unique<float[]>((size_t) (MM3_CLAP_WTOK * MM3_CLAP_WTOK * H));
                for (int64_t h = 0; h < H; h++) {
                    for (int i = 0; i < MM3_CLAP_WTOK; i++) {
                        for (int j = 0; j < MM3_CLAP_WTOK; j++) {
                            const int dh = i / MM3_CLAP_WIN - j / MM3_CLAP_WIN + MM3_CLAP_WIN - 1;
                            const int dw = i % MM3_CLAP_WIN - j % MM3_CLAP_WIN + MM3_CLAP_WIN - 1;
                            folded[(size_t) ((h * MM3_CLAP_WTOK + i) * MM3_CLAP_WTOK + j)] =
                                table[(dh * (2 * MM3_CLAP_WIN - 1) + dw) * H + h];
                        }
                    }
                }
                B.rel_bias = ggml_new_tensor_3d(m->wctx.ctx, GGML_TYPE_F32, MM3_CLAP_WTOK, MM3_CLAP_WTOK, H);
                ggml_set_name(B.rel_bias, (p + "attention.self.relative_bias").c_str());
                m->wctx.pending.push_back({ B.rel_bias, folded.get(), ggml_nbytes(B.rel_bias), 0 });
                m->wctx.staging.push_back(std::move(folded));
            }
        }
        if (s + 1 < MM3_CLAP_STAGES) {
            const std::string p = enc + "layers." + std::to_string(s) + ".downsample.";
            S.merge_ln_w = req(p + "norm.weight", { 4 * C });
            S.merge_ln_b = req(p + "norm.bias", { 4 * C });
            S.merge_w    = req(p + "reduction.weight", { 4 * C, 2 * C });
        }
    }
    if (!err->empty()) {
        mm3_clap_free(m);
        return false;
    }
    if (!wctx_alloc(&m->wctx, m->backend)) {
        *err = "backend buffer allocation failed for the CLAP weights";
        mm3_clap_free(m);
        return false;
    }
    m->sched = backend_sched_new(bp, MM3_CLAP_MAX_NODES);
    fprintf(stderr, "[MM3-CLAP] Loaded %s: %.1f MB, %.0f ms\n", path.c_str(),
            (double) ggml_backend_buffer_get_size(m->wctx.buffer) / (1024.0 * 1024.0),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// ---- Host front end -------------------------------------------------------

// torchaudio.functional.resample(x, 44100, 48000) with its defaults
// (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99), gcd 300: 147 -> 160.
// The kernel is built in float32 like torchaudio's for a float32 waveform; the
// dot products accumulate in double.
static std::vector<float> mm3_clap_resample(const float * x, int64_t n) {
    const int     orig = MM3_CLAP_SR_IN / 300, next = MM3_CLAP_SR / 300;
    const int     lpw  = 6;
    const double  base = std::min(orig, next) * 0.99;
    const int     width = (int) std::ceil(lpw * (double) orig / base);
    const int     taps  = 2 * width + orig;
    std::vector<float> kernel((size_t) (next * taps));
    for (int p = 0; p < next; p++) {
        for (int j = 0; j < taps; j++) {
            float t = (float) (-p) / (float) next + (float) (j - width) / (float) orig;
            t       = std::max((float) -lpw, std::min((float) lpw, t * (float) base));
            float w = std::cos(t * MM3_CLAP_PI / (float) lpw / 2.0f);
            t       = t * MM3_CLAP_PI;
            kernel[(size_t) (p * taps + j)] = (t == 0.0f ? 1.0f : std::sin(t) / t) * (w * w * (float) (base / orig));
        }
    }
    const int64_t      out_n = (int64_t) std::ceil((double) next * n / orig);
    std::vector<float> out((size_t) out_n);
    for (int64_t i = 0; i < out_n; i++) {
        const int64_t frame = i / next, p = i % next;
        const int64_t start = frame * orig - width;  // padded index frame*orig maps to x[frame*orig - width]
        double        acc   = 0.0;
        for (int j = 0; j < taps; j++) {
            const int64_t s = start + j;
            if (s >= 0 && s < n) {
                acc += (double) kernel[(size_t) (p * taps + j)] * x[s];
            }
        }
        out[(size_t) i] = (float) acc;
    }
    return out;
}

// The pipeline's centre crop: channel mean, then mono[c - h, c + h) with
// c = n / 2 and h = min(5 s, c). planar is [2][n] at 44.1 kHz.
static std::vector<float> mm3_clap_centre_crop(const float * planar, int64_t n) {
    const int64_t center = n / 2, half = std::min<int64_t>(5 * MM3_CLAP_SR_IN, center);
    std::vector<float> clip((size_t) (2 * half));
    for (int64_t i = 0; i < 2 * half; i++) {
        const int64_t s = center - half + i;
        clip[(size_t) i] = (planar[s] + planar[n + s]) / 2.0f;
    }
    return clip;
}

namespace mm3_clap_detail {

// np.linspace(a, b, n)[i]
static double linspace(double a, double b, int n, int i) {
    return i == n - 1 ? b : a + (b - a) / (n - 1) * i;
}

static double hz_to_mel(double f) {  // slaney
    return f >= 1000.0 ? 15.0 + std::log(f / 1000.0) * (27.0 / std::log(6.4)) : 3.0 * f / 200.0;
}

static double mel_to_hz(double m) {
    return m >= 15.0 ? 1000.0 * std::exp(std::log(6.4) / 27.0 * (m - 15.0)) : 200.0 * m / 3.0;
}

// transformers mel_filter_bank(513, 64, 50, 14000, 48000, norm="slaney", mel_scale="slaney"),
// returned as [bin][mel].
static const std::vector<double> & mel_bank() {
    static const std::vector<double> bank = [] {
        const int          M = MM3_CLAP_MELS, K = MM3_CLAP_BINS;
        std::vector<double> edge((size_t) (M + 2));
        for (int i = 0; i < M + 2; i++) {
            edge[(size_t) i] = mel_to_hz(linspace(hz_to_mel(50.0), hz_to_mel(14000.0), M + 2, i));
        }
        std::vector<double> fb((size_t) (K * M));
        for (int k = 0; k < K; k++) {
            const double f = linspace(0.0, MM3_CLAP_SR / 2, K, k);
            for (int m = 0; m < M; m++) {
                const double down = -(edge[(size_t) m] - f) / (edge[(size_t) m + 1] - edge[(size_t) m]);
                const double up   = (edge[(size_t) m + 2] - f) / (edge[(size_t) m + 2] - edge[(size_t) m + 1]);
                const double enorm = 2.0 / (edge[(size_t) m + 2] - edge[(size_t) m]);
                fb[(size_t) (k * M + m)] = std::max(0.0, std::min(down, up)) * enorm;
            }
        }
        return fb;
    }();
    return bank;
}

// In-place radix-2 FFT of 1024 points.
static void fft1024(std::complex<double> * a) {
    static const std::vector<std::complex<double>> tw = [] {
        std::vector<std::complex<double>> t(MM3_CLAP_FFT / 2);
        for (int i = 0; i < MM3_CLAP_FFT / 2; i++) {
            t[(size_t) i] = std::polar(1.0, -2.0 * 3.14159265358979323846 * i / MM3_CLAP_FFT);
        }
        return t;
    }();
    for (int i = 1, j = 0; i < MM3_CLAP_FFT; i++) {
        int bit = MM3_CLAP_FFT >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (int len = 2; len <= MM3_CLAP_FFT; len <<= 1) {
        const int step = MM3_CLAP_FFT / len;
        for (int i = 0; i < MM3_CLAP_FFT; i += len) {
            for (int k = 0; k < len / 2; k++) {
                const std::complex<double> u = a[i + k], v = a[i + k + len / 2] * tw[(size_t) (k * step)];
                a[i + k]           = u + v;
                a[i + k + len / 2] = u - v;
            }
        }
    }
}

static double cubic1(double x) {
    const double A = -0.75;
    return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
}

static double cubic2(double x) {
    const double A = -0.75;
    return ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
}

}  // namespace mm3_clap_detail

// ClapFeatureExtractor(truncation="rand_trunc", padding="repeatpad") on one
// mono 48 kHz clip: the stock processor's input_features, [1001][64] time-major.
static bool mm3_clap_features(const std::vector<float> & audio, std::vector<float> * out, std::string * err) {
    using namespace mm3_clap_detail;
    const size_t n = audio.size();
    if (n == 0 || n > (size_t) MM3_CLAP_MAX_SAMP) {
        *err = "CLAP clip of " + std::to_string(n) + " samples is outside 1.." + std::to_string(MM3_CLAP_MAX_SAMP);
        return false;
    }
    // repeatpad: tile floor(max / n) times, then zero pad.
    std::vector<double> wav((size_t) MM3_CLAP_MAX_SAMP, 0.0);
    const size_t        reps = (size_t) MM3_CLAP_MAX_SAMP / n;
    for (size_t i = 0; i < reps * n; i++) {
        wav[i] = audio[i % n];
    }
    // center=True, reflect padding of fft/2 each side, periodic Hann.
    const int           P = MM3_CLAP_FFT / 2, T = MM3_CLAP_MAX_SAMP;
    std::vector<double> pad((size_t) (T + 2 * P));
    for (int i = 0; i < T + 2 * P; i++) {
        int s = i - P;
        s     = s < 0 ? -s : (s >= T ? 2 * (T - 1) - s : s);
        pad[(size_t) i] = wav[(size_t) s];
    }
    std::vector<double> hann((size_t) MM3_CLAP_FFT);
    for (int k = 0; k < MM3_CLAP_FFT; k++) {
        hann[(size_t) k] = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * k / MM3_CLAP_FFT);
    }
    const std::vector<double> & fb = mel_bank();
    out->assign((size_t) MM3_CLAP_FRAMES * MM3_CLAP_MELS, 0.0f);
    std::vector<std::complex<double>> buf((size_t) MM3_CLAP_FFT);
    std::vector<double>               power((size_t) MM3_CLAP_BINS);
    for (int f = 0; f < MM3_CLAP_FRAMES; f++) {
        for (int k = 0; k < MM3_CLAP_FFT; k++) {
            buf[(size_t) k] = pad[(size_t) (f * MM3_CLAP_HOP + k)] * hann[(size_t) k];
        }
        fft1024(buf.data());
        for (int k = 0; k < MM3_CLAP_BINS; k++) {
            power[(size_t) k] = std::norm(buf[(size_t) k]);
        }
        for (int m = 0; m < MM3_CLAP_MELS; m++) {
            double acc = 0.0;
            for (int k = 0; k < MM3_CLAP_BINS; k++) {
                acc += fb[(size_t) (k * MM3_CLAP_MELS + m)] * power[(size_t) k];
            }
            (*out)[(size_t) (f * MM3_CLAP_MELS + m)] = (float) (10.0 * std::log10(std::max(acc, 1e-10)));
        }
    }
    return true;
}

// HTSAT front end on input_features [1001][64]: eval batch norm per mel bin, bicubic
// stretch of the time axis to 1024 (align_corners), fold into the 256x256 image,
// then cut into 4x4 patches. Returns [patch = ph * 64 + pw][kh * 4 + kw].
static std::vector<float> mm3_clap_patches(const Mm3Clap & m, const std::vector<float> & feat) {
    using namespace mm3_clap_detail;
    const int           T = MM3_CLAP_FRAMES, S = MM3_CLAP_STRETCH, M = MM3_CLAP_MELS;
    std::vector<double> bn((size_t) (T * M));
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < M; k++) {
            bn[(size_t) (t * M + k)] = (double) feat[(size_t) (t * M + k)] * m.bn_scale[(size_t) k] + m.bn_shift[(size_t) k];
        }
    }
    std::vector<float> img((size_t) MM3_CLAP_IMG * MM3_CLAP_IMG);
    const double       scale = (double) (T - 1) / (double) (S - 1);
    for (int t = 0; t < S; t++) {
        const double real = scale * t, fl = std::floor(real), u = real - fl;
        const double w[4] = { cubic2(u + 1.0), cubic1(u), cubic1(1.0 - u), cubic2(2.0 - u) };
        int          idx[4];
        for (int i = 0; i < 4; i++) {
            idx[i] = std::max(0, std::min(T - 1, (int) fl - 1 + i));
        }
        for (int k = 0; k < M; k++) {
            double v = 0.0;
            for (int i = 0; i < 4; i++) {
                v += w[i] * bn[(size_t) (idx[i] * M + k)];
            }
            img[(size_t) (((t / MM3_CLAP_IMG) * M + k) * MM3_CLAP_IMG + t % MM3_CLAP_IMG)] = (float) v;
        }
    }
    std::vector<float> patches((size_t) MM3_CLAP_GRID * MM3_CLAP_GRID * 16);
    for (int ph = 0; ph < MM3_CLAP_GRID; ph++) {
        for (int pw = 0; pw < MM3_CLAP_GRID; pw++) {
            for (int kh = 0; kh < MM3_CLAP_PATCH; kh++) {
                for (int kw = 0; kw < MM3_CLAP_PATCH; kw++) {
                    patches[(size_t) ((ph * MM3_CLAP_GRID + pw) * 16 + kh * MM3_CLAP_PATCH + kw)] =
                        img[(size_t) ((ph * MM3_CLAP_PATCH + kh) * MM3_CLAP_IMG + pw * MM3_CLAP_PATCH + kw)];
                }
            }
        }
    }
    return patches;
}

// ---- Graph ---------------------------------------------------------------

namespace mm3_clap_detail {

// Every CLAP GEMM runs in full FP32 (see mm3-refiner-mert.h: TF32 tensor cores miss the gate).
static ggml_tensor * mul_mat_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_hint(y, GGML_HINT_STRICT_F32);
    return y;
}

static ggml_tensor * ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, MM3_CLAP_EPS), w), b);
}

static ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    return ggml_add(ctx, mul_mat_f32(ctx, w, x), b);
}

// Window partition and cyclic shift for one (resolution, shift) pair, as gather
// maps: fwd[window position] = token to read, inv[token] = window position to
// read after attention (the reverse roll and window merge). Windows run in
// torch's order, row-major over the window grid, then row-major inside it.
struct Plan {
    int                  res = 0, shift = 0;
    std::vector<int32_t> fwd, inv;
    std::vector<float>   mask;  // [window][query][key], torch's SW-MSA mask; empty when shift is 0
    ggml_tensor *        t_fwd = nullptr, *t_inv = nullptr, *t_mask = nullptr;
};

static void build_plan(Plan * p, int res, int shift) {
    const int W = MM3_CLAP_WIN, nw = res / W;
    p->res = res;
    p->shift = shift;
    p->fwd.resize((size_t) res * res);
    p->inv.resize((size_t) res * res);
    auto pos = [&](int h, int w) { return ((h / W * nw + w / W) * W + h % W) * W + w % W; };
    for (int h = 0; h < res; h++) {
        for (int w = 0; w < res; w++) {
            p->fwd[(size_t) pos(h, w)] = ((h + shift) % res) * res + (w + shift) % res;
            p->inv[(size_t) (h * res + w)] = pos((h - shift + res) % res, (w - shift + res) % res);
        }
    }
    if (shift > 0) {
        auto region = [&](int c) { return c < res - W ? 0 : (c < res - shift ? 1 : 2); };
        p->mask.assign((size_t) nw * nw * W * W * W * W, 0.0f);
        for (int win = 0; win < nw * nw; win++) {
            for (int i = 0; i < W * W; i++) {
                for (int j = 0; j < W * W; j++) {
                    auto label = [&](int q) {
                        return region((win / nw) * W + q / W) * 3 + region((win % nw) * W + q % W);
                    };
                    p->mask[(size_t) ((win * W * W + i) * W * W + j)] = label(i) != label(j) ? -100.0f : 0.0f;
                }
            }
        }
    }
}

// One Swin block on x [C, res*res] (tokens row-major over the grid).
static ggml_tensor * block(ggml_context * ctx, const Mm3ClapBlock & B, Plan & P, int heads, ggml_tensor * x) {
    const int64_t C = x->ne[0], N = x->ne[1], HD = C / heads, NW = N / MM3_CLAP_WTOK;
    ggml_tensor * h = ln(ctx, x, B.ln1_w, B.ln1_b);
    h = ggml_reshape_3d(ctx, ggml_get_rows(ctx, h, P.t_fwd), C, MM3_CLAP_WTOK, NW);  // [C, 64, windows]
    ggml_tensor * q = linear(ctx, B.q_w, B.q_b, h);
    ggml_tensor * k = linear(ctx, B.k_w, B.k_b, h);
    ggml_tensor * v = linear(ctx, B.v_w, B.v_b, h);
    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, HD, heads, MM3_CLAP_WTOK, NW), 0, 2, 1, 3));  // [hd, 64, head, win]
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, HD, heads, MM3_CLAP_WTOK, NW), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, HD, heads, MM3_CLAP_WTOK, NW), 1, 2, 0, 3));  // [64, hd, head, win]
    ggml_tensor * s = ggml_add(ctx, ggml_scale(ctx, mul_mat_f32(ctx, k, q), 1.0f / std::sqrt((float) HD)), B.rel_bias);
    if (P.t_mask) {
        s = ggml_add(ctx, s, P.t_mask);  // [64, 64, 1, win] broadcast over heads
    }
    ggml_tensor * a = mul_mat_f32(ctx, v, ggml_soft_max(ctx, s));  // [hd, 64, head, win]
    a = ggml_cont_2d(ctx, ggml_permute(ctx, a, 0, 2, 1, 3), C, N);
    a = ggml_get_rows(ctx, linear(ctx, B.o_w, B.o_b, a), P.t_inv);
    x = ggml_add(ctx, x, a);
    ggml_tensor * f = ggml_gelu_erf(ctx, linear(ctx, B.up_w, B.up_b, ln(ctx, x, B.ln2_w, B.ln2_b)));
    return ggml_add(ctx, x, linear(ctx, B.down_w, B.down_b, f));
}

// Patch merging on x [C, res*res]: 2x2 neighbours concatenated in torch's
// order (even row/even col, odd row/even col, even row/odd col, odd/odd).
static ggml_tensor * merge(ggml_context * ctx, const Mm3ClapStage & S, ggml_tensor * x, int res) {
    const int64_t C = x->ne[0];
    ggml_tensor * g = ggml_reshape_3d(ctx, x, C, res, res);
    ggml_tensor * cat = nullptr;
    for (int piece = 0; piece < 4; piece++) {
        const int ph = piece & 1, pw = piece >> 1;
        ggml_tensor * v = ggml_cont(ctx, ggml_view_3d(ctx, g, C, res / 2, res / 2, g->nb[1] * 2, g->nb[2] * 2,
                                                      (size_t) pw * g->nb[1] + (size_t) ph * g->nb[2]));
        cat = cat ? ggml_concat(ctx, cat, v, 0) : v;
    }
    cat = ggml_reshape_2d(ctx, cat, 4 * C, (int64_t) res / 2 * (res / 2));
    return mul_mat_f32(ctx, S.merge_w, ln(ctx, cat, S.merge_ln_w, S.merge_ln_b));
}

}  // namespace mm3_clap_detail

// The pooled, L2-normalised 512-d vector for processor input_features [1001][64].
static bool mm3_clap_embed(Mm3Clap * m, const std::vector<float> & feat, std::vector<float> * out, std::string * err) {
    using namespace mm3_clap_detail;
    if (feat.size() != (size_t) MM3_CLAP_FRAMES * MM3_CLAP_MELS) {
        *err = "CLAP input_features must be 1001 x 64";
        return false;
    }
    const std::vector<float> patches = mm3_clap_patches(*m, feat);

    const size_t     bytes = ggml_tensor_overhead() * (MM3_CLAP_MAX_NODES + 64) +
                             ggml_graph_overhead_custom(MM3_CLAP_MAX_NODES, false);
    std::vector<uint8_t> buf(bytes);
    ggml_init_params ip  = { bytes, buf.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    struct Free {
        ggml_context * c;
        ~Free() { ggml_free(c); }
    } free_ctx{ ctx };

    ggml_tensor * p_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, MM3_CLAP_GRID * MM3_CLAP_GRID);
    ggml_set_input(p_in);
    // Patch embedding: the 4x4 stride-4 conv is one GEMM over the host-cut patches.
    ggml_tensor * x = ggml_add(ctx, mul_mat_f32(ctx, ggml_reshape_2d(ctx, m->patch_w, 16, MM3_CLAP_DIM0), p_in), m->patch_b);
    x               = ln(ctx, x, m->patch_ln_w, m->patch_ln_b);  // [128, 4096]

    std::vector<std::unique_ptr<Plan>> plans;
    auto plan_for = [&](int res, int shift) -> Plan & {
        for (auto & p : plans) {
            if (p->res == res && p->shift == shift) {
                return *p;
            }
        }
        plans.push_back(std::make_unique<Plan>());
        Plan & p = *plans.back();
        build_plan(&p, res, shift);
        p.t_fwd = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) res * res);
        p.t_inv = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) res * res);
        ggml_set_input(p.t_fwd);
        ggml_set_input(p.t_inv);
        if (shift > 0) {
            const int64_t nw = (int64_t) (res / MM3_CLAP_WIN) * (res / MM3_CLAP_WIN);
            p.t_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, MM3_CLAP_WTOK, MM3_CLAP_WTOK, 1, nw);
            ggml_set_input(p.t_mask);
        }
        return p;
    };
    int res = MM3_CLAP_GRID;
    for (int s = 0; s < MM3_CLAP_STAGES; s++) {
        for (int b = 0; b < mm3_clap_depths[s]; b++) {
            // Odd blocks shift by half a window, unless the stage is a single window.
            const int shift = (b & 1) && res > MM3_CLAP_WIN ? MM3_CLAP_WIN / 2 : 0;
            x               = block(ctx, m->stage[s].block[(size_t) b], plan_for(res, shift), mm3_clap_heads[s], x);
        }
        if (s + 1 < MM3_CLAP_STAGES) {
            x = merge(ctx, m->stage[s], x, res);
            res /= 2;
        }
    }
    // Final LayerNorm, mean over the 64 tokens (the freq/time regrouping before the
    // adaptive average pool does not change a global mean), then the projection.
    x = ln(ctx, x, m->norm_w, m->norm_b);                                    // [1024, 64]
    ggml_tensor * pooled = ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)));  // [1, 1024]
    pooled               = ggml_reshape_1d(ctx, pooled, MM3_CLAP_DIM_OUT);
    ggml_tensor * y = ggml_relu(ctx, linear(ctx, m->proj1_w, m->proj1_b, pooled));
    y               = linear(ctx, m->proj2_w, m->proj2_b, y);
    ggml_set_output(y);

    ggml_cgraph * gr = ggml_new_graph_custom(ctx, MM3_CLAP_MAX_NODES, false);
    ggml_build_forward_expand(gr, y);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gr)) {
        *err = "CLAP graph allocation failed (out of memory?)";
        return false;
    }
    ggml_backend_tensor_set(p_in, patches.data(), 0, ggml_nbytes(p_in));
    for (auto & p : plans) {
        ggml_backend_tensor_set(p->t_fwd, p->fwd.data(), 0, ggml_nbytes(p->t_fwd));
        ggml_backend_tensor_set(p->t_inv, p->inv.data(), 0, ggml_nbytes(p->t_inv));
        if (p->t_mask) {
            ggml_backend_tensor_set(p->t_mask, p->mask.data(), 0, ggml_nbytes(p->t_mask));
        }
    }
    if (ggml_backend_sched_graph_compute(m->sched, gr) != GGML_STATUS_SUCCESS) {
        *err = "CLAP graph compute failed";
        return false;
    }
    fprintf(stderr, "[MM3-CLAP] %d nodes, %d splits, compute %.0f MB\n", ggml_graph_n_nodes(gr),
            ggml_backend_sched_get_n_splits(m->sched),
            (double) ggml_backend_sched_get_buffer_size(m->sched, m->backend) / (1024.0 * 1024.0));
    out->resize(MM3_CLAP_PROJ);
    ggml_backend_tensor_get(y, out->data(), 0, MM3_CLAP_PROJ * sizeof(float));
    ggml_backend_sched_reset(m->sched);
    // F.normalize(dim=-1, eps=1e-12)
    double sq = 0.0;
    for (float v : *out) {
        sq += (double) v * v;
    }
    const double inv = 1.0 / std::max(std::sqrt(sq), 1e-12);
    for (float & v : *out) {
        v = (float) (v * inv);
    }
    return true;
}

struct Mm3ClapResult {
    std::vector<float> audio_48k;  // resampled centre crop, before the processor's padding
    std::vector<float> features;   // processor input_features [1001][64]
    std::vector<float> vector;     // [512], unit norm
};

// The full style source for one window. planar: 44.1 kHz stereo, [2][n]
// (channel 0 then channel 1).
static bool mm3_clap_encode(Mm3Clap * m, const float * planar, int64_t n, Mm3ClapResult * r, std::string * err) {
    if (n < 2) {
        *err = "CLAP needs a window of at least 2 samples";
        return false;
    }
    const std::vector<float> clip = mm3_clap_centre_crop(planar, n);
    r->audio_48k                  = mm3_clap_resample(clip.data(), (int64_t) clip.size());
    return mm3_clap_features(r->audio_48k, &r->features, err) && mm3_clap_embed(m, r->features, &r->vector, err);
}
