#pragma once
// minimax/mm3-dav-encode.h — MiniMax-Music3 DAV encoder (audio -> flow latents).
//
// HOT-Step file (does not exist upstream). TRAINING-SIDE: this is loaded by
// ace-train's preprocess, never by ace-server. It has its own small GGUF
// (mm3-enc-<quant>.gguf, arch "mm3enc", 91 tensors / 0.09 GB) written by
// convert-mm3.py's `enc` component from the original dav.pth.
//
// SCOPE (training path step 2): produce the flow DiT's TARGET latents.
//
//   stereo 44.1 kHz [2, N]  ->  latents [128, L],  L = N/512  (86.13 Hz)
//
// ── Architecture ─────────────────────────────────────────────────────────────
//
// The mirror of mm3-vocoder-graph.h, and deliberately written against it — the
// Snake composite, the precomputed inv_alpha, the explicit-F32 im2col and the
// [T, C] activation layout are all the same, so read that file's design notes
// first; only the differences are restated here.
//
//   conv_in   Conv1d(1 -> 64, k=7, p=3)
//   4 x encoder block, strides (2,4,8,8), ladder 64->128->256->512->1024:
//       3 x ResidualUnit(d in {1,3,9}):
//           skip -> Snake -> Conv1d k=7 dilation d pad=3d -> Snake -> Conv1d k=1 -> +skip
//       Snake(Cin)
//       Conv1d(Cin -> Cout, k=2r, stride=r, pad=ceil(r/2))
//   Snake(1024) -> conv_out Conv1d(1024 -> 1024, k=3, p=1)
//   mean_proj Conv1d(1024 -> 64, k=1)
//
//   Total downsample x512. Stereo is a FOLD, exactly as in the vocoder: L and R
//   each run through the same weights as a mono stream, and the two 64-channel
//   results stack into 128. Channels 0..63 are audio ch 0, 64..127 are ch 1.
//
// ── Differences from the vocoder that matter ─────────────────────────────────
//
// 1. NO ConvTranspose1d, so none of the vocoder's GEMM+col2im emulation or its
//    symmetric-crop length arithmetic is needed. Downsampling is just a STRIDE
//    on im2col, which ggml supports directly. This file is correspondingly
//    simpler; do not port the convt machinery across looking for symmetry.
//
//    The stride/padding arithmetic still has to land exactly on the reference:
//    with k = 2s and p = ceil(s/2) = s/2 (every rate is even),
//        OL = (T + 2p - k)/s + 1 = (T + s - 2s)/s + 1 = T/s
//    which is PyTorch's Conv1d output length for the same parameters. So each
//    block divides the length exactly and the four together divide by 512.
//
// 2. POSTERIOR MEAN ONLY. `logs_proj` exists in dav.pth and is deliberately
//    neither converted nor used: encode() returns the mean, so a training
//    target carries no sampling noise. If a reparameterised sample is ever
//    wanted, that is a new decision, not a missing feature.
//
// 3. NO final tanh (that belongs to the decoder's waveform head).
//
// ── Precision ────────────────────────────────────────────────────────────────
//
// F32 throughout, for the same reason the vocoder is: Snake is a periodic
// non-linearity applied 29 times, and half precision visibly damages it. The
// GGUF stores every tensor F32 already (0.09 GB, so there is nothing to save).
// ggml_conv_1d() is again avoided because it forces its im2col to F16.

// Standard headers FIRST: backend.h uses std::wstring without including
// <string>, so it only compiles when something upstream has already pulled it
// in. This header is included early in ace-train.cpp, so it must not rely on
// that happening by accident.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"

// Largest L encoded in one graph (~11.9 s of audio). A training crop is one
// 200-frame DiT window = 689 latents, so the normal path is always single-shot.
#define MM3_ENC_CHUNK 1024
// Latent frames of context carried into EACH side of a window, then discarded.
// Sized above the encoder's receptive field (~50-100 latents once the dilated
// k7 stacks are unrolled through all four rates), and MEASURED: a 12 s clip
// (L=1034, so it crosses the chunk boundary and windows) parity-checks at
// rel-RMSE 7.576e-07 against the reference, versus 7.212e-07 for a single-shot
// L=1024 run of the same audio. The seam therefore contributes nothing above
// the F32 noise floor. Shrink this only with that comparison re-run.
#define MM3_ENC_OVERLAP 96
#define MM3_ENC_MAX_NODES 2048

// ── Config + weights ────────────────────────────────────────────────────────

struct MM3EncConfig {
    uint32_t             in_dim           = 64;
    std::vector<int32_t> rates;                    // (2,4,8,8)
    std::vector<int32_t> res_dilations;            // (1,3,9)
    uint32_t             total_downsample = 512;
    uint32_t             latent_dim       = 1024;
    uint32_t             fold_channels    = 64;
    uint32_t             latent_channels  = 128;
    uint32_t             sampling_rate    = 44100;
    float                snake_eps        = 1e-9f;
};

struct MM3EncResUnit {
    ggml_tensor * snake1_alpha = nullptr;
    ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;
    ggml_tensor * snake2_alpha = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;
    // derived
    ggml_tensor * inv1 = nullptr, * inv2 = nullptr;
};

struct MM3EncBlock {
    std::vector<MM3EncResUnit> res;
    ggml_tensor *              snake_alpha = nullptr;
    ggml_tensor *              conv_w = nullptr, * conv_b = nullptr;
    ggml_tensor *              inv = nullptr;      // derived
};

struct MM3EncWeights {
    ggml_tensor *            conv_in_w = nullptr, * conv_in_b = nullptr;
    std::vector<MM3EncBlock> blk;
    ggml_tensor *            snake_out_alpha = nullptr;
    ggml_tensor *            inv_out         = nullptr;   // derived
    ggml_tensor *            conv_out_w = nullptr, * conv_out_b = nullptr;
    ggml_tensor *            mean_w = nullptr, * mean_b = nullptr;
};

struct MM3Enc {
    bool                  loaded  = false;
    std::string           path;
    MM3EncConfig          cfg;
    MM3EncWeights         w;
    ggml_backend_t        backend     = nullptr;
    ggml_backend_t        cpu_backend = nullptr;
    bool                  backend_ref = false;
    WeightCtx             wctx        = {};
    size_t                vram        = 0;
    // Full-FP32 GEMMs (see mm3_enc_conv1d). The refiner sets it: on TF32 the DAV latents miss its
    // parity gate. Training leaves it off, so its speed and numbers are unchanged.
    bool                  strict_f32  = false;

    // cached graph, rebuilt when the chunk length changes
    ggml_context *        gctx  = nullptr;
    ggml_cgraph *         graph = nullptr;
    ggml_backend_buffer_t gbuf  = nullptr;
    ggml_gallocr_t        alloc = nullptr;
    ggml_tensor *         g_in  = nullptr;
    ggml_tensor *         g_out = nullptr;
    int64_t               g_L   = 0;
};

// ── Load ────────────────────────────────────────────────────────────────────

static void mm3_enc_free_graph(MM3Enc * e) {
    if (e->alloc) { ggml_gallocr_free(e->alloc); e->alloc = nullptr; }
    if (e->gbuf)  { ggml_backend_buffer_free(e->gbuf); e->gbuf = nullptr; }
    if (e->gctx)  { ggml_free(e->gctx); e->gctx = nullptr; }
    e->graph = nullptr; e->g_in = nullptr; e->g_out = nullptr; e->g_L = 0;
}

static void mm3_enc_free(MM3Enc * e) {
    mm3_enc_free_graph(e);
    if (e->wctx.buffer) { ggml_backend_buffer_free(e->wctx.buffer); e->wctx.buffer = nullptr; }
    if (e->wctx.ctx)    { ggml_free(e->wctx.ctx); e->wctx.ctx = nullptr; }
    e->wctx = {};
    if (e->backend_ref) {
        backend_release(e->backend, e->cpu_backend);
        e->backend = e->cpu_backend = nullptr;
        e->backend_ref = false;
    }
    e->w = {};
    e->loaded = false;
    e->vram = 0;
}

// 1/(alpha + eps), precomputed into a staged F32 tensor. The fused Snake kernel
// multiplies by this; an in-graph ggml_div would both cost a divide per sample
// and break the fusion pattern (vocoder design note 2).
static ggml_tensor * mm3_enc_make_inv(WeightCtx * wctx, const GGUFModel & gf, const std::string & name,
                                      const ggml_tensor * alpha, float eps) {
    if (!alpha) {
        return nullptr;
    }
    const int64_t n = ggml_nelements(alpha);
    const float * a = (const float *) gf_get_data(gf, name.c_str());
    if (!a) {
        return nullptr;
    }
    auto buf = std::make_unique<float[]>((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        buf[i] = 1.0f / (a[i] + eps);
    }
    ggml_tensor * t = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, 1, n);
    ggml_set_name(t, (name + ".inv").c_str());
    wctx->pending.push_back({ t, buf.get(), (size_t) n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return t;
}

static bool mm3_enc_load(MM3Enc * e, const char * path, std::string * err) {
    if (e->loaded) {
        return true;
    }
    GGUFModel gf = {};
    if (!gf_load(&gf, path)) {
        if (err) *err = std::string("cannot open ") + path;
        return false;
    }

    std::vector<std::string> errs;
    MM3EncConfig &           c = e->cfg;
    c.in_dim           = gf_get_u32(gf, "mm3.enc.in_dim");
    c.total_downsample = gf_get_u32(gf, "mm3.enc.total_downsample");
    c.latent_dim       = gf_get_u32(gf, "mm3.enc.latent_dim");
    c.fold_channels    = gf_get_u32(gf, "mm3.enc.fold_channels");
    c.latent_channels  = gf_get_u32(gf, "mm3.enc.latent_channels");
    c.sampling_rate    = gf_get_u32(gf, "mm3.enc.sampling_rate");
    c.snake_eps        = gf_get_f32(gf, "mm3.enc.snake_eps");
    c.rates            = mm3_get_i32_arr(gf, "mm3.enc.rates");
    c.res_dilations    = mm3_get_i32_arr(gf, "mm3.enc.res_dilations");

    if (c.rates.empty() || c.res_dilations.empty()) {
        gf_close(&gf);
        if (err) *err = "mm3-enc GGUF is missing mm3.enc.rates / mm3.enc.res_dilations";
        return false;
    }
    {
        int64_t prod = 1;
        for (int32_t r : c.rates) prod *= r;
        if (prod != (int64_t) c.total_downsample) {
            gf_close(&gf);
            if (err) *err = "mm3-enc rates do not multiply to total_downsample";
            return false;
        }
    }
    if (c.fold_channels * 2 != c.latent_channels) {
        gf_close(&gf);
        if (err) *err = "mm3-enc fold_channels*2 != latent_channels";
        return false;
    }

    const int n_blk = (int) c.rates.size();
    const int n_res = (int) c.res_dilations.size();
    // 2 (conv_in) + per block [n_res*(1+2+1+2) + 1 + 2] + 1 + 2 + 2, plus one
    // derived inv per alpha.
    const int n_alpha   = n_blk * (n_res * 2 + 1) + 1;
    const int n_tensors = 2 + n_blk * (n_res * 6 + 3) + 1 + 2 + 2 + n_alpha + 8;
    wctx_init(&e->wctx, n_tensors);

    MM3Loader ld{ &e->wctx, &gf, nullptr, &errs };
    MM3EncWeights & w = e->w;

    const int64_t IN_CH = c.in_dim;
    w.conv_in_w = ld.req("enc.conv_in.weight", 7, 1, IN_CH);
    w.conv_in_b = ld.req("enc.conv_in.bias", IN_CH);

    int64_t ch = IN_CH;
    w.blk.assign((size_t) n_blk, MM3EncBlock{});
    for (int bi = 0; bi < n_blk && errs.empty(); bi++) {
        MM3EncBlock & b   = w.blk[(size_t) bi];
        const int64_t out = ch * 2;
        const int     s   = c.rates[(size_t) bi];
        b.res.assign((size_t) n_res, MM3EncResUnit{});
        for (int ri = 0; ri < n_res; ri++) {
            MM3EncResUnit & r = b.res[(size_t) ri];
            r.snake1_alpha = ld.req(mm3_fmt2("enc.blk.%d.res.%d.snake1.alpha", bi, ri), 1, ch);
            r.conv1_w      = ld.req(mm3_fmt2("enc.blk.%d.res.%d.conv1.weight", bi, ri), 7, ch, ch);
            r.conv1_b      = ld.req(mm3_fmt2("enc.blk.%d.res.%d.conv1.bias", bi, ri), ch);
            r.snake2_alpha = ld.req(mm3_fmt2("enc.blk.%d.res.%d.snake2.alpha", bi, ri), 1, ch);
            r.conv2_w      = ld.req(mm3_fmt2("enc.blk.%d.res.%d.conv2.weight", bi, ri), 1, ch, ch);
            r.conv2_b      = ld.req(mm3_fmt2("enc.blk.%d.res.%d.conv2.bias", bi, ri), ch);
        }
        b.snake_alpha = ld.req(mm3_fmt("enc.blk.%d.snake.alpha", bi), 1, ch);
        b.conv_w      = ld.req(mm3_fmt("enc.blk.%d.conv.weight", bi), 2 * s, ch, out);
        b.conv_b      = ld.req(mm3_fmt("enc.blk.%d.conv.bias", bi), out);
        ch = out;
    }

    w.snake_out_alpha = ld.req("enc.snake_out.alpha", 1, ch);
    w.conv_out_w      = ld.req("enc.conv_out.weight", 3, ch, (int64_t) c.latent_dim);
    w.conv_out_b      = ld.req("enc.conv_out.bias", (int64_t) c.latent_dim);
    w.mean_w          = ld.req("enc.mean_proj.weight", 1, (int64_t) c.latent_dim, (int64_t) c.fold_channels);
    w.mean_b          = ld.req("enc.mean_proj.bias", (int64_t) c.fold_channels);

    if (!errs.empty()) {
        gf_close(&gf);
        mm3_enc_free(e);
        if (err) *err = errs[0];
        return false;
    }

    // Derived reciprocals, read from the GGUF's own bytes before upload.
    for (int bi = 0; bi < n_blk; bi++) {
        MM3EncBlock & b = w.blk[(size_t) bi];
        for (int ri = 0; ri < n_res; ri++) {
            MM3EncResUnit & r = b.res[(size_t) ri];
            r.inv1 = mm3_enc_make_inv(&e->wctx, gf, mm3_fmt2("enc.blk.%d.res.%d.snake1.alpha", bi, ri),
                                      r.snake1_alpha, c.snake_eps);
            r.inv2 = mm3_enc_make_inv(&e->wctx, gf, mm3_fmt2("enc.blk.%d.res.%d.snake2.alpha", bi, ri),
                                      r.snake2_alpha, c.snake_eps);
        }
        b.inv = mm3_enc_make_inv(&e->wctx, gf, mm3_fmt("enc.blk.%d.snake.alpha", bi), b.snake_alpha, c.snake_eps);
    }
    w.inv_out = mm3_enc_make_inv(&e->wctx, gf, "enc.snake_out.alpha", w.snake_out_alpha, c.snake_eps);

    if (!e->backend_ref) {
        BackendPair bp  = backend_init("MM3Enc");
        e->backend      = bp.backend;
        e->cpu_backend  = bp.cpu_backend;
        e->backend_ref  = true;
    }
    for (const auto & pc : e->wctx.pending) {
        e->vram += pc.nbytes;
    }
    if (!wctx_alloc(&e->wctx, e->backend)) {
        gf_close(&gf);
        mm3_enc_free(e);
        if (err) *err = "backend buffer allocation failed for the DAV encoder";
        return false;
    }

    gf_close(&gf);
    e->path   = path;
    e->loaded = true;
    fprintf(stderr, "[MM3Enc] loaded %s (%d blocks, x%u, %.1f MB)\n", path, n_blk,
            c.total_downsample, (double) e->vram / (1024.0 * 1024.0));
    return true;
}

// ── Graph pieces ────────────────────────────────────────────────────────────

// Snake1d: y = x + inv_alpha * sin(alpha*x)^2. Op order is the CUDA/Vulkan
// fusion pattern — do not "simplify" it (vocoder design note 1).
static ggml_tensor * mm3_enc_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha,
                                   ggml_tensor * inv_alpha) {
    ggml_tensor * ax = ggml_mul(ctx, x, alpha);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_sqr(ctx, s);
    ggml_tensor * d  = ggml_mul(ctx, s2, inv_alpha);
    return ggml_add(ctx, x, d);
}

// Conv1d (+bias) with stride and dilation. w [K, IC, OC], x [T, IC] -> [OL, OC].
// Explicit F32 im2col: ggml_conv_1d forces F16 (vocoder design note 5).
// strict: run the GEMM in full FP32 (HOT-ggml GGML_HINT_STRICT_F32) instead of ggml-cuda's TF32 default.
static ggml_tensor * mm3_enc_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x,
                                    int stride, int pad, int dilation, bool strict = false) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, stride, /*s1*/ 0, pad, 0, dilation, 0, /*is_2D*/ false,
                                    GGML_TYPE_F32);  // [IC*K, OL, 1, 1]
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                   ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));  // [OL, OC]
    if (strict) {
        ggml_mul_mat_set_hint(y, GGML_HINT_STRICT_F32);
    }
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

static ggml_tensor * mm3_enc_res_unit(ggml_context * ctx, const MM3EncResUnit & r, ggml_tensor * x,
                                      int dilation, bool strict) {
    ggml_tensor * skip = x;
    ggml_tensor * y    = mm3_enc_snake(ctx, x, r.snake1_alpha, r.inv1);
    y                  = mm3_enc_conv1d(ctx, r.conv1_w, r.conv1_b, y, 1, 3 * dilation, dilation, strict);
    y                  = mm3_enc_snake(ctx, y, r.snake2_alpha, r.inv2);
    y                  = mm3_enc_conv1d(ctx, r.conv2_w, r.conv2_b, y, 1, 0, 1, strict);
    // Padding keeps the length, so this never fires — kept because the
    // reference has it and a config change that broke the invariant should
    // crop rather than assert.
    if (y->ne[0] != skip->ne[0]) {
        const int64_t off = (skip->ne[0] - y->ne[0]) / 2;
        skip = ggml_cont(ctx, ggml_view_2d(ctx, skip, y->ne[0], skip->ne[1], skip->nb[1],
                                           (size_t) off * sizeof(float)));
    }
    return ggml_add(ctx, skip, y);
}

// One mono stream: audio [N, 1] -> latents [N/512, 64].
static ggml_tensor * mm3_enc_build(ggml_context * ctx, const MM3Enc & e, ggml_tensor * audio) {
    const MM3EncConfig &  c = e.cfg;
    const MM3EncWeights & w = e.w;

    ggml_tensor * h = mm3_enc_conv1d(ctx, w.conv_in_w, w.conv_in_b, audio, 1, 3, 1, e.strict_f32);

    for (size_t bi = 0; bi < w.blk.size(); bi++) {
        const MM3EncBlock & b = w.blk[bi];
        for (size_t ri = 0; ri < b.res.size(); ri++) {
            h = mm3_enc_res_unit(ctx, b.res[ri], h, c.res_dilations[ri], e.strict_f32);
        }
        h = mm3_enc_snake(ctx, h, b.snake_alpha, b.inv);
        const int s = c.rates[bi];
        // k = 2s, pad = ceil(s/2); every rate is even so this is s/2 and the
        // output length is exactly T/s (see the header's length note).
        h = mm3_enc_conv1d(ctx, b.conv_w, b.conv_b, h, s, (s + 1) / 2, 1, e.strict_f32);
    }

    h = mm3_enc_snake(ctx, h, w.snake_out_alpha, w.inv_out);
    h = mm3_enc_conv1d(ctx, w.conv_out_w, w.conv_out_b, h, 1, 1, 1, e.strict_f32);
    h = mm3_enc_conv1d(ctx, w.mean_w, w.mean_b, h, 1, 0, 1, e.strict_f32);   // posterior MEAN
    return h;
}

static bool mm3_enc_ensure_graph(MM3Enc * e, int64_t L, std::string * err) {
    if (e->graph && e->g_L == L) {
        return true;
    }
    mm3_enc_free_graph(e);

    const int64_t N = L * (int64_t) e->cfg.total_downsample;

    const size_t          meta = ggml_tensor_overhead() * MM3_ENC_MAX_NODES + ggml_graph_overhead_custom(MM3_ENC_MAX_NODES, false);
    ggml_init_params      ip   = { meta, nullptr, true };
    e->gctx = ggml_init(ip);
    if (!e->gctx) {
        if (err) *err = "MM3Enc: ggml_init failed";
        return false;
    }

    e->g_in = ggml_new_tensor_2d(e->gctx, GGML_TYPE_F32, N, 1);
    ggml_set_name(e->g_in, "audio");
    ggml_set_input(e->g_in);

    e->g_out = mm3_enc_build(e->gctx, *e, e->g_in);
    ggml_set_name(e->g_out, "latents");
    ggml_set_output(e->g_out);

    e->graph = ggml_new_graph_custom(e->gctx, MM3_ENC_MAX_NODES, false);
    ggml_build_forward_expand(e->graph, e->g_out);

    if (e->g_out->ne[0] != L || e->g_out->ne[1] != (int64_t) e->cfg.fold_channels) {
        char buf[192];
        snprintf(buf, sizeof(buf), "MM3Enc: graph produced [%lld,%lld], expected [%lld,%u]",
                 (long long) e->g_out->ne[0], (long long) e->g_out->ne[1], (long long) L, e->cfg.fold_channels);
        if (err) *err = buf;
        mm3_enc_free_graph(e);
        return false;
    }

    e->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(e->backend));
    if (!e->alloc || !ggml_gallocr_reserve(e->alloc, e->graph)) {
        if (err) *err = "MM3Enc: graph allocation failed (out of VRAM?)";
        mm3_enc_free_graph(e);
        return false;
    }
    e->g_L = L;
    return true;
}

// Encode ONE mono stream of exactly L*512 samples into dst [L*64] laid out
// channel-major (64 rows of L contiguous frames).
static bool mm3_enc_run_stream(MM3Enc * e, const float * src, int64_t L, float * dst, std::string * err) {
    if (!mm3_enc_ensure_graph(e, L, err)) {
        return false;
    }
    if (!ggml_gallocr_alloc_graph(e->alloc, e->graph)) {
        if (err) *err = "MM3Enc: gallocr_alloc_graph failed";
        return false;
    }
    const int64_t N = L * (int64_t) e->cfg.total_downsample;
    ggml_backend_tensor_set(e->g_in, src, 0, (size_t) N * sizeof(float));
    if (ggml_backend_graph_compute(e->backend, e->graph) != GGML_STATUS_SUCCESS) {
        if (err) *err = "MM3Enc: graph compute failed";
        return false;
    }
    // ne0 = L is the fastest-varying axis, so the download is already
    // [64][L] C-order — exactly the reference's [128, L] per-channel layout.
    ggml_backend_tensor_get(e->g_out, dst, 0, (size_t) L * (int64_t) e->cfg.fold_channels * sizeof(float));
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

// audio: [2][n_samples] planar, 44.1 kHz, float. Mono is the caller's job to
// duplicate (the reference repeats it), matching _prepare_waveform.
// out: resized to latent_channels * L, laid out [128][L].
//
// Input is zero-padded up to a multiple of the hop, as the reference does.
static bool mm3_enc_encode(MM3Enc * e, const float * const * audio, int64_t n_samples,
                           std::vector<float> * out, int64_t * L_out, std::string * err) {
    if (!e->loaded) {
        if (err) *err = "MM3Enc: not loaded";
        return false;
    }
    const int64_t hop = (int64_t) e->cfg.total_downsample;
    const int64_t FC  = (int64_t) e->cfg.fold_channels;
    const int64_t L   = (n_samples + hop - 1) / hop;
    if (L <= 0) {
        if (err) *err = "MM3Enc: empty input";
        return false;
    }
    out->assign((size_t) (2 * FC * L), 0.0f);
    if (L_out) {
        *L_out = L;
    }

    // Zero-padded, hop-aligned copy of each channel.
    std::vector<float> pad((size_t) (L * hop));

    for (int ch = 0; ch < 2; ch++) {
        std::fill(pad.begin(), pad.end(), 0.0f);
        memcpy(pad.data(), audio[ch], (size_t) n_samples * sizeof(float));

        float * dst_ch = out->data() + (int64_t) ch * FC * L;

        if (L <= MM3_ENC_CHUNK) {
            if (!mm3_enc_run_stream(e, pad.data(), L, dst_ch, err)) {
                return false;
            }
            continue;
        }

        // Windowed: each window carries MM3_ENC_OVERLAP latents of real context
        // on each side (except at the signal ends, where the reference's own
        // zero padding is the correct context) and keeps only its interior.
        const int64_t core = MM3_ENC_CHUNK - 2 * MM3_ENC_OVERLAP;
        std::vector<float> tmp;
        for (int64_t start = 0; start < L; start += core) {
            const int64_t lead  = std::min<int64_t>(MM3_ENC_OVERLAP, start);
            const int64_t begin = start - lead;
            const int64_t want  = std::min<int64_t>(MM3_ENC_CHUNK, L - begin);
            tmp.resize((size_t) (want * FC));
            if (!mm3_enc_run_stream(e, pad.data() + begin * hop, want, tmp.data(), err)) {
                return false;
            }
            const int64_t keep = std::min<int64_t>(core, L - start);
            for (int64_t c = 0; c < FC; c++) {
                memcpy(dst_ch + c * L + start, tmp.data() + c * want + lead, (size_t) keep * sizeof(float));
            }
        }
    }
    return true;
}
