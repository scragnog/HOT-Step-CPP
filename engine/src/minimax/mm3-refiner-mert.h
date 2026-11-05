#pragma once
// minimax/mm3-refiner-mert.h — MERT-v1-95M conditioning for the MM3 latent refiner.
//
// HOT-Step file. The refiner (terminusresearch minimax-music3-latent-refiner
// v0.10) conditions every window on all 13 MERT-v1-95M hidden states. This file
// is that conditioner end to end, the chain in the package's
// MiniMaxMusic3RefinerPipeline._mert_layers plus the normalisation in
// _refine_chunk:
//
//   44.1 kHz stereo window -> channel mean -> torchaudio sinc resample to 24 kHz
//   -> Wav2Vec2FeatureExtractor zero-mean/unit-variance -> 7 conv feature layers
//   (GroupNorm on the first) -> LayerNorm + projection 512->768 -> weight-normed
//   grouped positional conv -> LayerNorm -> 12 post-LN transformer layers
//   -> hidden_states[0..12] -> linear interpolation (align_corners=True) to the
//   DAV latent frame count -> (x - layer_mean) / layer_std
//
// Weights: mm3-refiner-mert-f32.gguf (arch "mm3-refiner-mert") from
// tools/mm3-refiner/convert.py, torch names and shapes unchanged. The layer
// statistics are normalization.layer_{mean,std} in the refiner GGUF.
//
// Not yue2-mert.h: that is MERT-v2 (mel + ConvNext + Conformer). Only its
// loader conventions are followed here; no code is shared.
//
// Traps, each measured against the slice-1 fixtures:
// - torchaudio builds the resample kernel in float32 (the waveform dtype), and
//   its output length is ceil(n * 80 / 147). The engine's audio-resample.h is a
//   different (Kaiser) filter.
// - The positional conv has kernel 128 and padding 64, so it yields T+1 frames;
//   HubertSamePadLayer drops the LAST one. weight_norm is over dim 2: one norm
//   per kernel tap across all [out, in/groups].
// - hidden_states[0] is the state AFTER pos-conv add and encoder.layer_norm,
//   before layer 0. The layers are post-LN (do_stable_layer_norm=false).
// - GELU is the exact erf form everywhere.
//
// Gated by mm3-refiner-mert-probe / tools/mm3-refiner/check-mert.ps1.

// backend.h needs <string> first.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"
#include "yyjson.h"

static constexpr int     MM3_MERT_SR_IN    = 44100;
static constexpr int     MM3_MERT_SR       = 24000;
static constexpr int     MM3_MERT_LAYERS   = 12;
static constexpr int     MM3_MERT_STATES   = 13;
static constexpr int64_t MM3_MERT_DIM      = 768;
static constexpr int64_t MM3_MERT_FFN      = 3072;
static constexpr int64_t MM3_MERT_CONV_DIM = 512;
static constexpr int     MM3_MERT_HEADS    = 12;
static constexpr int     MM3_MERT_POS_K    = 128;
static constexpr int     MM3_MERT_POS_G    = 16;
static constexpr float   MM3_MERT_EPS      = 1e-5f;
static constexpr int     MM3_MERT_MAX_NODES = 4096;
static constexpr float   MM3_MERT_PI        = 3.14159265358979323846f;

static const int mm3_mert_conv_kernel[7] = { 10, 3, 3, 3, 3, 2, 2 };
static const int mm3_mert_conv_stride[7] = { 5, 2, 2, 2, 2, 2, 2 };

struct Mm3MertLayer {
    ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    ggml_tensor *ln_w, *ln_b, *up_w, *up_b, *down_w, *down_b, *fln_w, *fln_b;
};

struct Mm3Mert {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            wctx        = {};

    ggml_tensor * conv_w[7]  = {};
    ggml_tensor * gn_w       = nullptr;
    ggml_tensor * gn_b       = nullptr;
    ggml_tensor * fp_ln_w    = nullptr;
    ggml_tensor * fp_ln_b    = nullptr;
    ggml_tensor * fp_w       = nullptr;
    ggml_tensor * fp_b       = nullptr;
    ggml_tensor * pos_w      = nullptr;  // folded g * v / ||v||, ne [128, 48, 768]
    ggml_tensor * pos_b      = nullptr;
    ggml_tensor * enc_ln_w   = nullptr;
    ggml_tensor * enc_ln_b   = nullptr;
    Mm3MertLayer  layer[MM3_MERT_LAYERS] = {};

    std::vector<float> layer_mean;  // [13 * 768]
    std::vector<float> layer_std;   // [13 * 768]
};

static void mm3_mert_free(Mm3Mert * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    wctx_free(&m->wctx);
    if (m->backend) {
        backend_release(m->backend, m->cpu_backend);
    }
    *m = Mm3Mert{};
}

// Reads normalization.layer_{mean,std} [13, 768] from the refiner GGUF.
static bool mm3_mert_load_stats(Mm3Mert * m, const std::string & refiner_path, std::string * err) {
    GGUFModel gf;
    if (!gf_load(&gf, refiner_path.c_str())) {
        *err = "cannot open " + refiner_path;
        return false;
    }
    bool ok = true;
    for (auto [name, out] : { std::pair{ "normalization.layer_mean", &m->layer_mean },
                              std::pair{ "normalization.layer_std", &m->layer_std } }) {
        int64_t       idx = gguf_find_tensor(gf.gguf, name);
        ggml_tensor * t   = idx < 0 ? nullptr : ggml_get_tensor(gf.meta, name);
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != MM3_MERT_DIM || t->ne[1] != MM3_MERT_STATES ||
            ggml_n_dims(t) != 2) {
            *err = std::string(name) + " missing or not F32 [13,768] in " + refiner_path;
            ok   = false;
            break;
        }
        out->resize((size_t) (MM3_MERT_STATES * MM3_MERT_DIM));
        std::memcpy(out->data(), gf.mapping + gf.data_offset + gguf_get_tensor_offset(gf.gguf, idx),
                    out->size() * sizeof(float));
    }
    for (float s : m->layer_std) {
        if (ok && !(s > 0.0f && std::isfinite(s))) {
            *err = "normalization.layer_std has a non-positive entry";
            ok   = false;
        }
    }
    gf_close(&gf);
    return ok;
}

// The pinned config must be the HuBERT-style post-LN MERT this graph implements.
static bool mm3_mert_check_config(const char * json, std::string * err) {
    yyjson_doc * doc = json ? yyjson_read(json, strlen(json), 0) : nullptr;
    if (!doc) {
        *err = "mm3.refiner.config missing or not JSON";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    auto         num  = [&](const char * k) { return yyjson_get_num(yyjson_obj_get(root, k)); };
    auto         flag = [&](const char * k) { return yyjson_get_bool(yyjson_obj_get(root, k)); };
    auto         text = [&](const char * k) {
        const char * v = yyjson_get_str(yyjson_obj_get(root, k));
        return std::string(v ? v : "");
    };
    bool         ok   = num("hidden_size") == MM3_MERT_DIM && num("num_hidden_layers") == MM3_MERT_LAYERS &&
              num("num_attention_heads") == MM3_MERT_HEADS && num("intermediate_size") == MM3_MERT_FFN &&
              std::abs(num("layer_norm_eps") - MM3_MERT_EPS) < 1e-9 && num("sample_rate") == MM3_MERT_SR &&
              num("num_conv_pos_embeddings") == MM3_MERT_POS_K &&
              num("num_conv_pos_embedding_groups") == MM3_MERT_POS_G && !flag("do_stable_layer_norm") &&
              !flag("feature_extractor_cqt") && !flag("deepnorm") && !flag("conv_bias") &&
              flag("feat_proj_layer_norm") && num("attention_relax") <= 0 &&
              text("feat_extract_norm") == "group" && text("hidden_act") == "gelu";
    yyjson_val * ks = yyjson_obj_get(root, "conv_kernel");
    yyjson_val * ss = yyjson_obj_get(root, "conv_stride");
    ok              = ok && yyjson_arr_size(ks) == 7 && yyjson_arr_size(ss) == 7;
    for (int i = 0; ok && i < 7; i++) {
        ok = yyjson_get_num(yyjson_arr_get(ks, i)) == mm3_mert_conv_kernel[i] &&
             yyjson_get_num(yyjson_arr_get(ss, i)) == mm3_mert_conv_stride[i];
    }
    yyjson_doc_free(doc);
    if (!ok) {
        *err = "MERT config differs from the MERT-v1-95M layout this port implements";
    }
    return ok;
}

// mert_path: mm3-refiner-mert-f32.gguf. refiner_path: the refiner GGUF holding
// the per-layer statistics.
static bool mm3_mert_load(Mm3Mert * m, const std::string & mert_path, const std::string & refiner_path,
                          std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    mm3_mert_free(m);
    if (!mm3_mert_load_stats(m, refiner_path, err)) {
        return false;
    }
    GGUFModel gf;
    if (!gf_load(&gf, mert_path.c_str())) {
        *err = "cannot open " + mert_path;
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
    if (!arch || std::string(arch) != "mm3-refiner-mert") {
        *err = mert_path + ": architecture is not mm3-refiner-mert";
        return false;
    }
    if (!mm3_mert_check_config(str("mm3.refiner.config"), err)) {
        return false;
    }

    BackendPair bp = backend_init("MM3-MERT");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    wctx_init(&m->wctx, 16 + 16 * MM3_MERT_LAYERS);

    // ne is the reversed torch shape; every weight must be F32 with exactly it.
    auto req = [&](const std::string & name, std::initializer_list<int64_t> ne) -> ggml_tensor * {
        if (!err->empty()) {
            return nullptr;
        }
        ggml_tensor * src = gguf_find_tensor(gf.gguf, name.c_str()) < 0 ? nullptr : ggml_get_tensor(gf.meta, name.c_str());
        bool          ok  = src && src->type == GGML_TYPE_F32 && ggml_n_dims(src) <= (int) ne.size();
        int           d   = 0;
        for (int64_t n : ne) {
            ok = ok && src->ne[d++] == n;
        }
        if (!ok) {
            *err = "tensor " + name + " missing or wrong shape/type in " + mert_path;
            return nullptr;
        }
        return gf_load_tensor(&m->wctx, gf, name);
    };
    err->clear();
    for (int i = 0; i < 7; i++) {
        m->conv_w[i] = req("feature_extractor.conv_layers." + std::to_string(i) + ".conv.weight",
                           { mm3_mert_conv_kernel[i], i ? MM3_MERT_CONV_DIM : 1, MM3_MERT_CONV_DIM });
    }
    m->gn_w     = req("feature_extractor.conv_layers.0.layer_norm.weight", { MM3_MERT_CONV_DIM });
    m->gn_b     = req("feature_extractor.conv_layers.0.layer_norm.bias", { MM3_MERT_CONV_DIM });
    m->fp_ln_w  = req("feature_projection.layer_norm.weight", { MM3_MERT_CONV_DIM });
    m->fp_ln_b  = req("feature_projection.layer_norm.bias", { MM3_MERT_CONV_DIM });
    m->fp_w     = req("feature_projection.projection.weight", { MM3_MERT_CONV_DIM, MM3_MERT_DIM });
    m->fp_b     = req("feature_projection.projection.bias", { MM3_MERT_DIM });
    m->pos_b    = req("encoder.pos_conv_embed.conv.bias", { MM3_MERT_DIM });
    m->enc_ln_w = req("encoder.layer_norm.weight", { MM3_MERT_DIM });
    m->enc_ln_b = req("encoder.layer_norm.bias", { MM3_MERT_DIM });
    for (int l = 0; l < MM3_MERT_LAYERS; l++) {
        const std::string p = "encoder.layers." + std::to_string(l) + ".";
        Mm3MertLayer &    L = m->layer[l];
        const int64_t     D = MM3_MERT_DIM, F = MM3_MERT_FFN;
        L.q_w    = req(p + "attention.q_proj.weight", { D, D });
        L.q_b    = req(p + "attention.q_proj.bias", { D });
        L.k_w    = req(p + "attention.k_proj.weight", { D, D });
        L.k_b    = req(p + "attention.k_proj.bias", { D });
        L.v_w    = req(p + "attention.v_proj.weight", { D, D });
        L.v_b    = req(p + "attention.v_proj.bias", { D });
        L.o_w    = req(p + "attention.out_proj.weight", { D, D });
        L.o_b    = req(p + "attention.out_proj.bias", { D });
        L.ln_w   = req(p + "layer_norm.weight", { D });
        L.ln_b   = req(p + "layer_norm.bias", { D });
        L.up_w   = req(p + "feed_forward.intermediate_dense.weight", { D, F });
        L.up_b   = req(p + "feed_forward.intermediate_dense.bias", { F });
        L.down_w = req(p + "feed_forward.output_dense.weight", { F, D });
        L.down_b = req(p + "feed_forward.output_dense.bias", { D });
        L.fln_w  = req(p + "final_layer_norm.weight", { D });
        L.fln_b  = req(p + "final_layer_norm.bias", { D });
    }

    // Positional conv weight_norm(dim=2): w[o,i,k] = g[k] * v[o,i,k] / ||v[:,:,k]||.
    // Folded once on the host, in double, into one staged F32 tensor.
    const int64_t K = MM3_MERT_POS_K, IG = MM3_MERT_DIM / MM3_MERT_POS_G, O = MM3_MERT_DIM;
    ggml_tensor * g = nullptr, *v = nullptr;
    if (err->empty()) {
        g = ggml_get_tensor(gf.meta, "encoder.pos_conv_embed.conv.weight_g");
        v = ggml_get_tensor(gf.meta, "encoder.pos_conv_embed.conv.weight_v");
        if (!g || !v || g->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 || g->ne[0] != K || g->ne[1] != 1 ||
            g->ne[2] != 1 || v->ne[0] != K || v->ne[1] != IG || v->ne[2] != O) {
            *err = "positional conv weight_g/weight_v missing or wrong shape in " + mert_path;
        }
    }
    if (!err->empty()) {
        mm3_mert_free(m);
        return false;
    }
    auto at = [&](const char * name) {
        return (const float *) (gf.mapping + gf.data_offset +
                                gguf_get_tensor_offset(gf.gguf, gguf_find_tensor(gf.gguf, name)));
    };
    const float * gv = at("encoder.pos_conv_embed.conv.weight_g");
    const float * vv = at("encoder.pos_conv_embed.conv.weight_v");  // torch [O, IG, K] C order
    std::vector<double> norm((size_t) K, 0.0);
    for (int64_t j = 0; j < O * IG; j++) {
        for (int64_t k = 0; k < K; k++) {
            norm[(size_t) k] += (double) vv[j * K + k] * vv[j * K + k];
        }
    }
    auto folded = std::make_unique<float[]>((size_t) (O * IG * K));
    for (int64_t j = 0; j < O * IG; j++) {
        for (int64_t k = 0; k < K; k++) {
            folded[(size_t) (j * K + k)] = (float) (gv[k] * (double) vv[j * K + k] / std::sqrt(norm[(size_t) k]));
        }
    }
    m->pos_w = ggml_new_tensor_3d(m->wctx.ctx, GGML_TYPE_F32, K, IG, O);
    ggml_set_name(m->pos_w, "encoder.pos_conv_embed.conv.weight");
    m->wctx.pending.push_back({ m->pos_w, folded.get(), ggml_nbytes(m->pos_w), 0 });
    m->wctx.staging.push_back(std::move(folded));

    if (!wctx_alloc(&m->wctx, m->backend)) {
        *err = "backend buffer allocation failed for the MERT weights";
        mm3_mert_free(m);
        return false;
    }
    m->sched = backend_sched_new(bp, MM3_MERT_MAX_NODES);
    fprintf(stderr, "[MM3-MERT] Loaded %s: %.1f MB, %.0f ms\n", mert_path.c_str(),
            (double) ggml_backend_buffer_get_size(m->wctx.buffer) / (1024.0 * 1024.0),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// torchaudio.functional.resample(x, 44100, 24000) with its defaults
// (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99). The kernel is
// built in float32 like torchaudio's for a float32 waveform; the dot products
// accumulate in double.
static std::vector<float> mm3_mert_resample(const float * x, int64_t n) {
    const int     orig = MM3_MERT_SR_IN / 300, next = MM3_MERT_SR / 300;  // gcd 300: 147 -> 80
    const int     lpw  = 6;
    const double  base = std::min(orig, next) * 0.99;
    const int     width = (int) std::ceil(lpw * (double) orig / base);
    const int     taps  = 2 * width + orig;
    std::vector<float> kernel((size_t) (next * taps));
    for (int p = 0; p < next; p++) {
        for (int j = 0; j < taps; j++) {
            float t = (float) (-p) / (float) next + (float) (j - width) / (float) orig;
            t       = std::max((float) -lpw, std::min((float) lpw, t * (float) base));
            float w = std::cos(t * MM3_MERT_PI / (float) lpw / 2.0f);
            t       = t * MM3_MERT_PI;
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

// Wav2Vec2FeatureExtractor(do_normalize=True) on one unpadded clip.
static void mm3_mert_normalize(std::vector<float> & x) {
    double mean = 0.0, var = 0.0;
    for (float v : x) {
        mean += v;
    }
    mean /= (double) x.size();
    for (float v : x) {
        var += ((double) v - mean) * ((double) v - mean);
    }
    var /= (double) x.size();
    const double inv = 1.0 / std::sqrt(var + 1e-7);
    for (float & v : x) {
        v = (float) (((double) v - mean) * inv);
    }
}

static int64_t mm3_mert_frames(int64_t samples) {
    int64_t t = samples;
    for (int i = 0; i < 7; i++) {
        t = t < mm3_mert_conv_kernel[i] ? 0 : (t - mm3_mert_conv_kernel[i]) / mm3_mert_conv_stride[i] + 1;
    }
    return t;
}

namespace mm3_mert_detail {

// Every MERT GEMM runs in full FP32: on TF32 tensor cores (ggml-cuda's default) the
// 13 states miss the slice-1 gate. GGML_HINT_STRICT_F32 is a HOT-ggml fork hint.
static ggml_tensor * mul_mat_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_hint(y, GGML_HINT_STRICT_F32);
    return y;
}

static ggml_tensor * ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, MM3_MERT_EPS), w), b);
}

// Conv1d over time with an F32 im2col (ggml_conv_1d forces F16). x [T, IC], w [K, IC, OC] -> [T_out, OC].
static ggml_tensor * conv(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, int stride, int pad) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);
    return mul_mat_f32(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
}

// One post-LN HuBERT encoder layer on h [D, T].
static ggml_tensor * layer(ggml_context * ctx, const Mm3MertLayer & L, ggml_tensor * h) {
    const int64_t D = MM3_MERT_DIM, HD = D / MM3_MERT_HEADS, NH = MM3_MERT_HEADS, T = h->ne[1];
    ggml_tensor * q = ggml_add(ctx, mul_mat_f32(ctx, L.q_w, h), L.q_b);
    ggml_tensor * k = ggml_add(ctx, mul_mat_f32(ctx, L.k_w, h), L.k_b);
    ggml_tensor * v = ggml_add(ctx, mul_mat_f32(ctx, L.v_w, h), L.v_b);
    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, HD, NH, T), 0, 2, 1, 3));  // [hd, T, head]
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, HD, NH, T), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, HD, NH, T), 1, 2, 0, 3));  // [T, hd, head]
    // ponytail: dense [T, T, head] F32 scores (0.24 GB for a 30 s window, quadratic);
    // query-blocked or flash attention once direct mode feeds whole songs.
    ggml_tensor * kq = ggml_soft_max_ext(ctx, mul_mat_f32(ctx, k, q), nullptr, 1.0f / std::sqrt((float) HD), 0.0f);
    ggml_tensor * a  = ggml_cont_2d(ctx, ggml_permute(ctx, mul_mat_f32(ctx, v, kq), 0, 2, 1, 3), D, T);
    h = ln(ctx, ggml_add(ctx, h, ggml_add(ctx, mul_mat_f32(ctx, L.o_w, a), L.o_b)), L.ln_w, L.ln_b);
    ggml_tensor * f = ggml_gelu_erf(ctx, ggml_add(ctx, mul_mat_f32(ctx, L.up_w, h), L.up_b));
    f               = ggml_add(ctx, mul_mat_f32(ctx, L.down_w, f), L.down_b);
    return ln(ctx, ggml_add(ctx, h, f), L.fln_w, L.fln_b);
}

}  // namespace mm3_mert_detail

// Raw hidden states for one normalised 24 kHz clip. out: 13 * T * 768, [state][frame][dim].
static bool mm3_mert_hidden(Mm3Mert * m, const std::vector<float> & input, std::vector<float> * out, int64_t * frames,
                            std::string * err) {
    using namespace mm3_mert_detail;
    const int64_t T = mm3_mert_frames((int64_t) input.size());
    if (T < 1) {
        *err = "MERT input of " + std::to_string(input.size()) + " samples is too short";
        return false;
    }
    const size_t     bytes = ggml_tensor_overhead() * (MM3_MERT_MAX_NODES + 64) +
                             ggml_graph_overhead_custom(MM3_MERT_MAX_NODES, false);
    std::vector<uint8_t> buf(bytes);
    ggml_init_params ip  = { bytes, buf.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    struct Free {
        ggml_context * c;
        ~Free() { ggml_free(c); }
    } free_ctx{ ctx };

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) input.size(), 1);
    ggml_set_input(x_in);
    ggml_tensor * x = conv(ctx, m->conv_w[0], x_in, mm3_mert_conv_stride[0], 0);  // [T0, 512]
    // GroupNorm(512 groups, 512 channels): each channel over time, which is ne0 here.
    x = ggml_norm(ctx, x, MM3_MERT_EPS);
    x = ggml_add(ctx, ggml_mul(ctx, x, ggml_reshape_2d(ctx, m->gn_w, 1, MM3_MERT_CONV_DIM)),
                 ggml_reshape_2d(ctx, m->gn_b, 1, MM3_MERT_CONV_DIM));
    x = ggml_gelu_erf(ctx, x);
    for (int i = 1; i < 7; i++) {
        x = ggml_gelu_erf(ctx, conv(ctx, m->conv_w[i], x, mm3_mert_conv_stride[i], 0));
    }
    x = ggml_cont(ctx, ggml_transpose(ctx, x));  // [512, T]
    ggml_tensor * h = ggml_add(ctx, mul_mat_f32(ctx, m->fp_w, ln(ctx, x, m->fp_ln_w, m->fp_ln_b)), m->fp_b);

    // Grouped positional conv: 16 groups of 48 channels, k 128, pad 64, drop the last frame.
    ggml_tensor * ht  = ggml_cont(ctx, ggml_transpose(ctx, h));  // [T, 768]
    const int64_t IG  = MM3_MERT_DIM / MM3_MERT_POS_G;
    ggml_tensor * pos = nullptr;
    for (int gi = 0; gi < MM3_MERT_POS_G; gi++) {
        ggml_tensor * xg = ggml_view_2d(ctx, ht, T, IG, ht->nb[1], (size_t) gi * IG * ht->nb[1]);
        ggml_tensor * wg = ggml_view_3d(ctx, m->pos_w, MM3_MERT_POS_K, IG, IG, m->pos_w->nb[1], m->pos_w->nb[2],
                                        (size_t) gi * IG * m->pos_w->nb[2]);
        ggml_tensor * y  = conv(ctx, wg, ggml_cont(ctx, xg), 1, MM3_MERT_POS_K / 2);  // [T+1, 48]
        y                = ggml_cont(ctx, ggml_view_2d(ctx, y, T, IG, y->nb[1], 0));
        pos              = pos ? ggml_concat(ctx, pos, y, 1) : y;
    }
    pos = ggml_add(ctx, pos, ggml_reshape_2d(ctx, m->pos_b, 1, MM3_MERT_DIM));
    pos = ggml_cont(ctx, ggml_transpose(ctx, ggml_gelu_erf(ctx, pos)));  // [768, T]
    h   = ln(ctx, ggml_add(ctx, h, pos), m->enc_ln_w, m->enc_ln_b);

    ggml_tensor * states[MM3_MERT_STATES];
    states[0] = h;
    for (int l = 0; l < MM3_MERT_LAYERS; l++) {
        states[l + 1] = h = layer(ctx, m->layer[l], h);
    }
    ggml_cgraph * gr = ggml_new_graph_custom(ctx, MM3_MERT_MAX_NODES, false);
    for (ggml_tensor * s : states) {
        ggml_set_output(s);
        ggml_build_forward_expand(gr, s);
    }
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gr)) {
        *err = "MERT graph allocation failed (out of memory?) for " + std::to_string(T) + " frames";
        return false;
    }
    ggml_backend_tensor_set(x_in, input.data(), 0, ggml_nbytes(x_in));
    if (ggml_backend_sched_graph_compute(m->sched, gr) != GGML_STATUS_SUCCESS) {
        *err = "MERT graph compute failed";
        return false;
    }
    fprintf(stderr, "[MM3-MERT] %lld samples -> %lld frames, %d nodes, %d splits, compute %.0f MB\n",
            (long long) input.size(), (long long) T, ggml_graph_n_nodes(gr), ggml_backend_sched_get_n_splits(m->sched),
            (double) ggml_backend_sched_get_buffer_size(m->sched, m->backend) / (1024.0 * 1024.0));
    const size_t per = (size_t) (T * MM3_MERT_DIM);
    out->resize(per * MM3_MERT_STATES);
    for (int s = 0; s < MM3_MERT_STATES; s++) {
        ggml_backend_tensor_get(states[s], out->data() + s * per, 0, per * sizeof(float));
    }
    ggml_backend_sched_reset(m->sched);
    *frames = T;
    return true;
}

// F.interpolate(mode="linear", align_corners=True) along time, per state.
// src [S][T][D] -> [S][L][D].
static std::vector<float> mm3_mert_interpolate(const std::vector<float> & src, int64_t T, int64_t L) {
    const int64_t      D = MM3_MERT_DIM;
    std::vector<float> dst((size_t) (MM3_MERT_STATES * L * D));
    const float        scale = L > 1 ? (float) (T - 1) / (float) (L - 1) : 0.0f;
    for (int64_t j = 0; j < L; j++) {
        const float   p  = scale * (float) j;
        const int64_t i0 = std::min<int64_t>((int64_t) p, T - 1);
        const int64_t i1 = std::min<int64_t>(i0 + 1, T - 1);
        const float   l1 = std::min(std::max(p - (float) i0, 0.0f), 1.0f), l0 = 1.0f - l1;
        for (int s = 0; s < MM3_MERT_STATES; s++) {
            const float * a = src.data() + (s * T + i0) * D;
            const float * b = src.data() + (s * T + i1) * D;
            float *       o = dst.data() + (s * L + j) * D;
            for (int64_t d = 0; d < D; d++) {
                o[d] = l0 * a[d] + l1 * b[d];
            }
        }
    }
    return dst;
}

struct Mm3MertResult {
    std::vector<float> audio_24k;     // resampled mono, before normalisation
    std::vector<float> input_values;  // normalised processor output
    std::vector<float> interpolated;  // [13][L][768]
    std::vector<float> normalized;    // [13][L][768], what the refiner reads
    int64_t            mert_frames = 0;
};

// The full conditioner for one window. planar: 44.1 kHz stereo, [2][n]
// (channel 0 then channel 1). latent_frames: the window's DAV latent count L.
static bool mm3_mert_encode(Mm3Mert * m, const float * planar, int64_t n, int64_t latent_frames, Mm3MertResult * r,
                            std::string * err) {
    if (n < 1 || latent_frames < 1) {
        *err = "MERT needs a non-empty window and latent frame count";
        return false;
    }
    std::vector<float> mono((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        mono[(size_t) i] = (planar[i] + planar[n + i]) / 2.0f;
    }
    r->audio_24k    = mm3_mert_resample(mono.data(), n);
    r->input_values = r->audio_24k;
    mm3_mert_normalize(r->input_values);
    std::vector<float> hidden;
    if (!mm3_mert_hidden(m, r->input_values, &hidden, &r->mert_frames, err)) {
        return false;
    }
    r->interpolated = mm3_mert_interpolate(hidden, r->mert_frames, latent_frames);
    r->normalized   = r->interpolated;
    const int64_t D = MM3_MERT_DIM;
    for (int s = 0; s < MM3_MERT_STATES; s++) {
        for (int64_t j = 0; j < latent_frames; j++) {
            float * o = r->normalized.data() + (s * latent_frames + j) * D;
            for (int64_t d = 0; d < D; d++) {
                o[d] = (o[d] - m->layer_mean[(size_t) (s * D + d)]) / m->layer_std[(size_t) (s * D + d)];
            }
        }
    }
    return true;
}
