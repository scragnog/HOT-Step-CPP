#pragma once
// yue2/yue2-vae-graph.h — YuE2 Oobleck VAE decoder graph (M6).
//
// HOT-Step file (does not exist upstream). Included only by bring-up tooling
// (engine/tools/yue2-probe.cpp) in this milestone — hot-step-server.cpp is
// NOT touched yet.
//
// Contract: docs/plans/yue2/06-engine-port-plan.md §6, docs/plans/yue2/
// 03-reference-numerics.md §4 (Oobleck block order, SnakeBeta, tiling),
// docs/plans/yue2/05-gguf-layout.md §4 (dec.* tensor names/shapes, already
// bound into Yue2VaeWeights by yue2-model.h). Weight structs/loading live in
// yue2-model.h; this file only builds compute graphs over them.
//
// ── Architecture (decoder-only; enc.* is never touched here) ───────────────
//
//   latent [64, T]   channel-major (memory index = c*T + t — the SAME
//                    convention mm3-vocoder-graph.h's `x [T, IC]` already
//                    uses: ggml ne0=T fastest, ne1=C). The fixture's own
//                    input_latents.bin is [T,64] ROW-MAJOR (index = t*64+c,
//                    i.e. channel-contiguous per frame) — the CALLER
//                    transposes when reading the fixture, this graph never
//                    sees the fixture's own on-disk layout.
//
//   dec.conv_in   Conv1d 64 -> 2048, k=7, p=3
//   6x DecoderBlock, strides [6,5,4,4,2,2], widths 2048->1024->512->256->128->64->64:
//       SnakeBeta(Cin)                              -- BEFORE the upsample
//       ConvTranspose1d(Cin->Cout, k=2*stride, stride, pad=ceil(stride/2))
//       3x ResidualUnit(dilation in {1,3,9}):
//           skip -> SnakeBeta -> Conv1d k=7 dilation d pad=3d -> SnakeBeta
//                -> Conv1d k=1 -> +skip
//   SnakeBeta(64) -> Conv1d 64 -> 2 (STEREO, no bias, no tanh: final_tanh=false)
//
//   Output is NATIVELY stereo (dec.conv_out emits 2 channels directly) — do
//   NOT port mm3-vocoder-graph.h's fold_channels/two-pass-per-channel
//   machinery, there is nothing to fold here (06-plan §6, "Must be
//   rewritten" list).
//
// ── SnakeBeta vs MM3's Snake — the one place copying MM3 verbatim is wrong ──
//
//   MM3 vocoder: y = x + 1/(alpha+eps) * sin(alpha*x)^2        (ONE param)
//   YuE2 VAE:    y = x + 1/(exp(beta)+eps) * sin(x*exp(alpha))^2  (TWO params,
//                BOTH stored in log-space, both need exp() before use —
//                05-gguf-layout.md §4.4, 03-reference-numerics.md §4.3).
//   The five-op fused chain (mul->sin->sqr->mul->add) is structurally
//   unchanged from MM3's — just two independently-precomputed derived
//   tensors (exp_alpha, inv_beta) feeding it instead of one (alpha, inv_alpha
//   reusing the raw activation). Precomputed once per snake instance into
//   this file's own derived-weight buffer, exactly like MM3's inv_alpha.
//
// ── Tiled decode (03-reference-numerics.md §4.1) ────────────────────────────
//
//   core=1024, halo=16 (schema-fixed, 02-fixture-schema.md v1.4 §7). Each
//   tile is an INDEPENDENT FULL decode of [left,right) (no incremental
//   reuse across tiles, unlike MM3's overlap-discard windowing which is the
//   same idea but MM3 has no "core vs halo" distinction) — cropped back to
//   the tile's own core range with `crop_start=(start-left)*ratio`, placed
//   with a direct copy (no crossfade/blend — reference's own contract).
//   Tiled-vs-untiled is "mathematically equivalent, bitwise unpromised" per
//   the reference's own docstring — never treat a small tiled-vs-full delta
//   as a port bug without checking what the reference's own delta is first.

#include "yue2-model.h"

#include "backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Largest T decoded in one graph before yue2_vae_decode_tiled must be used
// instead of yue2_vae_decode (untiled). Purely a node/VRAM-budget guard for
// the untiled entry point — the tiled path has its own, schema-fixed
// core/halo geometry and never consults this.
#define YUE2_VAE_UNTILED_MAX_T 8192
// Node budget: 6 blocks * (1 snake + 1 convt + 3 res units * ~8 ops) + a
// handful of top/tail ops. Measured shape is a few hundred nodes; loose cap.
#define YUE2_VAE_MAX_NODES 2048

// ── Derived weights (computed once from the loaded GGUF tensors) ───────────

struct Yue2VaeSnakePrep {
    ggml_tensor * exp_alpha = nullptr;  // exp(alpha), [1,C]
    ggml_tensor * inv_beta  = nullptr;  // 1/(exp(beta)+eps), [1,C]
};

struct Yue2VaePrepRes {
    Yue2VaeSnakePrep snake1;
    Yue2VaeSnakePrep snake2;
};

struct Yue2VaePrepBlk {
    Yue2VaeSnakePrep            snake_pre;
    ggml_tensor *                upsample_w = nullptr;  // repacked [Cin, K*Cout], GEMM layout
    std::vector<Yue2VaePrepRes> res;                    // 3 entries, dilations 1/3/9
};

struct Yue2VaeGraph {
    // backend + residency
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            prep        = {};
    // Identity of the weights this prep was derived from (the VAE weight
    // buffer pointer) — changes whenever yue2_load_parts swaps VAE variants,
    // which frees and reallocates wctx_vae. Rebuild whenever this changes.
    const void * weights_token = nullptr;

    std::vector<Yue2VaePrepBlk> blk;
    Yue2VaeSnakePrep             snake_out;

    // cached graph (rebuilt when T changes)
    ggml_context * gctx    = nullptr;
    uint8_t *      gbuf    = nullptr;
    ggml_cgraph *  graph   = nullptr;
    ggml_tensor *  input   = nullptr;  // [T, 64] F32, channel-major (ne0=T,ne1=64)
    ggml_tensor *  output  = nullptr;  // [S, 2] F32, planar stereo (ne0=S,ne1=2)
    int64_t        graph_T = 0;
};

// ── Small helpers ────────────────────────────────────────────────────────────

// Read an F32 backend tensor back to host. The whole VAE is F32 on disk
// (03-reference-numerics.md §6) — anything else means a converter change
// this loader must notice loudly rather than silently mis-read.
static bool yue2_vae_readback(const ggml_tensor * t, std::vector<float> * out, std::string * err,
                              const char * what) {
    if (!t) {
        if (err) {
            *err = std::string("VAE tensor missing: ") + what;
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("VAE tensor '") + what + "' is not F32 (type " + std::to_string((int) t->type) +
                   "); the layout contract pins the whole VAE to F32";
        }
        return false;
    }
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get((ggml_tensor *) t, out->data(), 0, ggml_nbytes(t));
    return true;
}

// Create a [1, C] F32 tensor in wctx and queue `data` for upload. The [1,C]
// shape (not a bare 1D [C]) is what broadcasts correctly against an
// activation x with ne=[T,C] in ggml_mul below.
static ggml_tensor * yue2_vae_stage(WeightCtx * wctx, int64_t C, std::unique_ptr<float[]> data, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, 1, C);
    ggml_set_name(t, name);
    const size_t nbytes = (size_t) C * sizeof(float);
    wctx->pending.push_back({ t, data.get(), nbytes, 0 });
    wctx->staging.push_back(std::move(data));
    return t;
}

// Create a general [ne0, ne1] F32 tensor in wctx and queue `data` for
// upload — the plain 2D GEMM-operand form (mirrors mm3_voc_stage exactly),
// distinct from yue2_vae_stage's [1,C] broadcast-vector shape above.
static ggml_tensor * yue2_vae_stage2d(WeightCtx * wctx, int64_t ne0, int64_t ne1, std::unique_ptr<float[]> data,
                                      const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    const size_t nbytes = (size_t) ne0 * (size_t) ne1 * sizeof(float);
    wctx->pending.push_back({ t, data.get(), nbytes, 0 });
    wctx->staging.push_back(std::move(data));
    return t;
}

// exp(alpha) and 1/(exp(beta)+eps) for a raw log-space [C] alpha/beta pair.
// TRAP (05-gguf-layout.md §4.4): both go through exp() — this is NOT MM3's
// single-alpha Snake (mm3_voc_make_inv only ever exponentiates nothing and
// reciprocates the raw stored value; here BOTH stored values are logs).
static bool yue2_vae_make_snake_prep(WeightCtx * wctx, const ggml_tensor * alpha, const ggml_tensor * beta, float eps,
                                     const std::string & name_prefix, Yue2VaeSnakePrep * out, std::string * err) {
    std::vector<float> a, b;
    if (!yue2_vae_readback(alpha, &a, err, (name_prefix + ".alpha").c_str())) {
        return false;
    }
    if (!yue2_vae_readback(beta, &b, err, (name_prefix + ".beta").c_str())) {
        return false;
    }
    if (a.size() != b.size()) {
        if (err) {
            *err = "VAE snake '" + name_prefix + "': alpha/beta size mismatch (" + std::to_string(a.size()) + " vs " +
                   std::to_string(b.size()) + ")";
        }
        return false;
    }
    const int64_t C          = (int64_t) a.size();
    auto          exp_alpha  = std::make_unique<float[]>((size_t) C);
    auto          inv_beta   = std::make_unique<float[]>((size_t) C);
    for (int64_t i = 0; i < C; i++) {
        exp_alpha[(size_t) i] = std::exp(a[(size_t) i]);
        inv_beta[(size_t) i]  = 1.0f / (std::exp(b[(size_t) i]) + eps);
    }
    out->exp_alpha = yue2_vae_stage(wctx, C, std::move(exp_alpha), (name_prefix + ".exp_alpha").c_str());
    out->inv_beta  = yue2_vae_stage(wctx, C, std::move(inv_beta), (name_prefix + ".inv_beta").c_str());
    return out->exp_alpha != nullptr && out->inv_beta != nullptr;
}

// Repack a ConvTranspose1d kernel from the GGUF layout to the GEMM layout —
// IDENTICAL formula/shape convention to mm3_voc_repack_convt (05-gguf-layout.md
// §4.1's dec.blk.B.upsample is the same [K,OC,IC] convention MM3 already
// documents and implements; 06-plan §6 calls this "the single cleanest
// verbatim-shape reuse in this whole plan").
//   src: [K, OC, IC]   idx = k + oc*K + ic*K*OC
//   dst: [IC, K*OC]    idx = ic + (k + oc*K)*IC
static ggml_tensor * yue2_vae_repack_convt(WeightCtx * wctx, const ggml_tensor * w, const char * name,
                                           std::string * err) {
    std::vector<float> src;
    if (!yue2_vae_readback(w, &src, err, name)) {
        return nullptr;
    }
    const int64_t K  = w->ne[0];
    const int64_t OC = w->ne[1];
    const int64_t IC = w->ne[2];

    auto dst = std::make_unique<float[]>((size_t) (IC * K * OC));
    for (int64_t ic = 0; ic < IC; ic++) {
        const float * s = src.data() + ic * K * OC;
        for (int64_t oc = 0; oc < OC; oc++) {
            for (int64_t k = 0; k < K; k++) {
                dst[(size_t) (ic + (k + oc * K) * IC)] = s[(size_t) (k + oc * K)];
            }
        }
    }
    return yue2_vae_stage2d(wctx, IC, K * OC, std::move(dst), name);
}

// ── Prep / free ─────────────────────────────────────────────────────────────

static void yue2_vae_free_graph(Yue2VaeGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx    = nullptr;
    g->gbuf    = nullptr;
    g->graph   = nullptr;
    g->input   = nullptr;
    g->output  = nullptr;
    g->graph_T = 0;
}

static void yue2_vae_graph_free(Yue2VaeGraph * g) {
    yue2_vae_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    wctx_free(&g->prep);
    g->blk.clear();
    g->snake_out     = Yue2VaeSnakePrep{};
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

// Build (or rebuild) the derived-weight buffer and the scheduler for
// whichever VAE variant is currently resident in `m` (m.vae_loaded_variant).
// Cheap after the first call: returns immediately when the model's VAE
// weight buffer is the same one this prep was derived from.
static bool yue2_vae_prepare(const Yue2Model & m, Yue2VaeGraph * g, std::string * err) {
    if (!m.vae_resident) {
        if (err) {
            *err = "YuE2 VAE is not resident (load it first)";
        }
        return false;
    }
    const void * token = (const void *) m.wctx_vae.buffer;
    if (g->weights_token == token && g->sched) {
        return true;
    }
    yue2_vae_graph_free(g);

    const Yue2VaeConfig &  vc  = m.vae_cfg;
    const Yue2VaeWeights & vw  = m.vae;
    const int              NB  = (int) vc.strides.size();
    const int              NR  = (int) vc.res_dilations.size();
    const float            eps = vc.snake_eps > 0.0f ? vc.snake_eps : 1e-9f;

    BackendPair bp = backend_init("YuE2-Vae");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    // Per block: 1 snake_pre (2 tensors) + 1 repacked convt + NR*(2 snakes,
    // 4 tensors). Plus 1 snake_out (2 tensors).
    const int n_prep = NB * (2 + 1 + NR * 4) + 2;
    wctx_init(&g->prep, n_prep);

    std::string e;
    bool        ok = true;
    g->blk.assign((size_t) NB, Yue2VaePrepBlk{});
    for (int b = 0; b < NB && ok; b++) {
        char nm[96];
        Yue2VaePrepBlk &        pb = g->blk[(size_t) b];
        const Yue2VaeDecBlock & wb = vw.dec_blk[(size_t) b];

        snprintf(nm, sizeof(nm), "vae.dec.blk.%d.snake_pre", b);
        ok = yue2_vae_make_snake_prep(&g->prep, wb.snake_pre_alpha, wb.snake_pre_beta, eps, nm, &pb.snake_pre, &e);

        if (ok) {
            snprintf(nm, sizeof(nm), "vae.dec.blk.%d.upsample.gemm", b);
            pb.upsample_w = yue2_vae_repack_convt(&g->prep, wb.upsample_w, nm, &e);
            ok            = pb.upsample_w != nullptr;
        }

        pb.res.assign((size_t) NR, Yue2VaePrepRes{});
        for (int r = 0; r < NR && ok; r++) {
            const Yue2VaeResUnit & wr = wb.res[(size_t) r];
            snprintf(nm, sizeof(nm), "vae.dec.blk.%d.res.%d.snake1", b, r);
            ok = yue2_vae_make_snake_prep(&g->prep, wr.snake1_alpha, wr.snake1_beta, eps, nm,
                                          &pb.res[(size_t) r].snake1, &e);
            if (!ok) {
                break;
            }
            snprintf(nm, sizeof(nm), "vae.dec.blk.%d.res.%d.snake2", b, r);
            ok = yue2_vae_make_snake_prep(&g->prep, wr.snake2_alpha, wr.snake2_beta, eps, nm,
                                          &pb.res[(size_t) r].snake2, &e);
        }
    }
    if (ok) {
        ok = yue2_vae_make_snake_prep(&g->prep, vw.dec_snake_out_alpha, vw.dec_snake_out_beta, eps,
                                      "vae.dec.snake_out", &g->snake_out, &e);
    }
    if (ok) {
        ok = wctx_alloc(&g->prep, g->backend);
        if (!ok) {
            e = "backend buffer allocation failed for the VAE derived weights";
        }
    }
    if (!ok) {
        if (err) {
            *err = e.empty() ? "VAE prepare failed" : e;
        }
        yue2_vae_graph_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, YUE2_VAE_MAX_NODES * 2);
    g->weights_token = token;

    const size_t prep_bytes = g->prep.buffer ? ggml_backend_buffer_get_size(g->prep.buffer) : 0;
    fprintf(stderr, "[YuE2-Vae] Prepared (%s): %d derived tensors, %.1f MB (exp_alpha/inv_beta + repacked convT)\n",
            vc.variant.c_str(), n_prep, (double) prep_bytes / (1024.0 * 1024.0));
    return true;
}

// ── Graph pieces ────────────────────────────────────────────────────────────

// SnakeBeta: y = x + inv_beta * sin(x * exp_alpha)^2
// x [T, C]; exp_alpha and inv_beta [1, C] F32 contiguous. Op order is
// load-bearing (matches the CUDA/Vulkan snake-fusion pattern) — see file
// header note on SnakeBeta vs MM3's Snake. Do not "simplify" it.
static ggml_tensor * yue2_vae_snake_beta(ggml_context * ctx, ggml_tensor * x, const Yue2VaeSnakePrep & p) {
    ggml_tensor * ax = ggml_mul(ctx, x, p.exp_alpha);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_sqr(ctx, s);
    ggml_tensor * d  = ggml_mul(ctx, s2, p.inv_beta);
    return ggml_add(ctx, x, d);
}

// Conv1d (+bias). w [K, IC, OC], x [T, IC] -> [T_out, OC]. `b` may be
// nullptr (dec.conv_out has none — the one bias=False WNConv1d in the whole
// decoder). Explicit F32 im2col (never ggml_conv_1d's forced F16 path) — the
// VAE precision contract (03-reference-numerics.md §6: autocast explicitly
// disabled, TF32 explicitly disabled) is identical to MM3's own vocoder
// rationale, verbatim-copied.
//
// `stride` defaults to 1: every conv in the DECODER is stride 1 (its only
// resampling is the ConvTranspose1d below). The ENCODER's six downsamples
// are strided, so yue2-vae-encode.h passes a real stride here, which
// ggml_im2col's own s0 argument already supports — output length follows
// PyTorch's floor((L + 2p - d(k-1) - 1)/s) + 1 either way.
static ggml_tensor * yue2_vae_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int pad,
                                     int dilation, int stride = 1) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, /*s0*/ stride, /*s1*/ 0, pad, 0, dilation, 0, /*is_2D*/ false,
                                    GGML_TYPE_F32);  // [IC*K, OL, 1, 1]
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                   ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));  // [OL, OC]
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

// ConvTranspose1d (+bias) with k = 2*stride and pad = ceil(stride/2), via
// GEMM+col2im — IDENTICAL formula to mm3_voc_convt (both models share the
// same k=2*stride convention, 06-plan §6). wg is the repacked [IC, K*OC]
// GEMM operand from yue2_vae_repack_convt.
static ggml_tensor * yue2_vae_convt(ggml_context * ctx, ggml_tensor * wg, ggml_tensor * b, ggml_tensor * x,
                                    int stride, int64_t oc) {
    const int64_t T_in   = x->ne[0];
    const int     S      = stride;
    const int     K      = 2 * S;
    const int64_t T_full = (T_in - 1) * S + K;  // == (T_in + 1) * S
    const int     p      = (S + 1) / 2;         // ceil(s/2)

    ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));         // [IC, T_in]
    ggml_tensor * col = ggml_mul_mat(ctx, wg, xt);                     // [K*OC, T_in]
    ggml_tensor * c3  = ggml_reshape_3d(ctx, col, K, oc, T_in);        // [K, OC, T_in]

    ggml_tensor * lo = ggml_view_3d(ctx, c3, S, oc, T_in, c3->nb[1], c3->nb[2], 0);
    lo               = ggml_cont(ctx, ggml_permute(ctx, lo, 0, 2, 1, 3));  // [S, T_in, OC]
    lo               = ggml_reshape_2d(ctx, lo, (int64_t) S * T_in, oc);
    ggml_tensor * y  = ggml_pad(ctx, lo, S, 0, 0, 0);                      // [T_full, OC]

    ggml_tensor * hi = ggml_view_3d(ctx, c3, S, oc, T_in, c3->nb[1], c3->nb[2], (size_t) S * c3->nb[0]);
    hi               = ggml_cont(ctx, ggml_permute(ctx, hi, 0, 2, 1, 3));
    hi               = ggml_reshape_2d(ctx, hi, (int64_t) S * T_in, oc);
    y = ggml_acc(ctx, y, hi, y->nb[1], y->nb[2], y->nb[3], (size_t) S * sizeof(float));

    // Symmetric crop = the ConvTranspose1d padding.
    y = ggml_cont(ctx, ggml_view_2d(ctx, y, T_full - 2 * p, oc, y->nb[1], (size_t) p * sizeof(float)));

    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

// ResidualUnit: skip + conv2(snake2(conv1(snake1(x)))), pad=3*dilation on the
// k=7 conv (dilation in {1,3,9}), pad=0 on the k=1 conv. Centre-crop guard
// kept (mirrors mm3_voc_res_unit) though it should always be a no-op here.
static ggml_tensor * yue2_vae_res_unit(ggml_context * ctx, const Yue2VaeResUnit & w, const Yue2VaePrepRes & p,
                                       ggml_tensor * x, int dilation) {
    ggml_tensor * skip = x;
    ggml_tensor * y    = yue2_vae_snake_beta(ctx, x, p.snake1);
    y                  = yue2_vae_conv1d(ctx, w.conv1_w, w.conv1_b, y, 3 * dilation, dilation);
    y                  = yue2_vae_snake_beta(ctx, y, p.snake2);
    y                  = yue2_vae_conv1d(ctx, w.conv2_w, w.conv2_b, y, 0, 1);
    if (y->ne[0] != skip->ne[0]) {
        const int64_t off = (skip->ne[0] - y->ne[0]) / 2;
        skip = ggml_cont(ctx, ggml_view_2d(ctx, skip, y->ne[0], skip->ne[1], skip->nb[1],
                                           (size_t) off * sizeof(float)));
    }
    return ggml_add(ctx, skip, y);
}

// Full decoder. latent [T, 64] -> stereo audio [S, 2] (planar: ne0=S fastest,
// ne1=2 — memory index = s + ch*S, which IS the fixture's [2,S] planar
// layout, no transpose needed on the way out).
static ggml_tensor * yue2_vae_build(ggml_context * ctx, const Yue2Model & m, const Yue2VaeGraph & g,
                                    ggml_tensor * latent) {
    const Yue2VaeConfig &  vc = m.vae_cfg;
    const Yue2VaeWeights & vw = m.vae;
    const int              NB = (int) vc.strides.size();

    ggml_tensor * x = yue2_vae_conv1d(ctx, vw.dec_conv_in_w, vw.dec_conv_in_b, latent, 3, 1);  // 64 -> C0, k=7 p=3

    for (int b = 0; b < NB; b++) {
        const Yue2VaeDecBlock & wb     = vw.dec_blk[(size_t) b];
        const Yue2VaePrepBlk &  pb     = g.blk[(size_t) b];
        const int               stride = vc.strides[(size_t) (NB - 1 - b)];  // decoder runs strides reversed
        const int64_t           oc     = wb.upsample_b->ne[0];

        x = yue2_vae_snake_beta(ctx, x, pb.snake_pre);  // activation BEFORE upsample
        x = yue2_vae_convt(ctx, pb.upsample_w, wb.upsample_b, x, stride, oc);
        for (size_t r = 0; r < vc.res_dilations.size(); r++) {
            x = yue2_vae_res_unit(ctx, wb.res[r], pb.res[r], x, (int) vc.res_dilations[r]);
        }
    }

    x = yue2_vae_snake_beta(ctx, x, g.snake_out);
    x = yue2_vae_conv1d(ctx, vw.dec_conv_out_w, /*bias=*/nullptr, x, 3, 1);  // C_last -> 2, k=7 p=3, NO bias, NO tanh
    return x;
}

// Build the graph for a given input length, or reuse the cached one.
static bool yue2_vae_ensure_graph(const Yue2Model & m, Yue2VaeGraph * g, int64_t T, std::string * err) {
    if (g->graph && g->graph_T == T) {
        return true;
    }
    yue2_vae_free_graph(g);

    const size_t ctx_bytes = ggml_tensor_overhead() * (YUE2_VAE_MAX_NODES + 64) +
                              ggml_graph_overhead_custom(YUE2_VAE_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the VAE graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the VAE graph context";
        }
        return false;
    }

    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, (int64_t) m.vae_cfg.latent_dim);
    ggml_set_name(g->input, "yue2_vae_in");
    ggml_set_input(g->input);

    g->output = yue2_vae_build(ctx, m, *g, g->input);
    ggml_set_name(g->output, "yue2_vae_out");
    ggml_set_output(g->output);

    g->graph = ggml_new_graph_custom(ctx, YUE2_VAE_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->output);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "VAE graph allocation failed (out of VRAM?) for T=" + std::to_string(T);
        }
        return false;
    }

    g->gctx    = ctx;
    g->graph_T = T;
    const size_t compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);
    fprintf(stderr, "[YuE2-Vae] Graph: T=%lld -> %lld samples/ch, %d nodes, %d splits, compute buffer %.0f MB\n",
            (long long) T, (long long) g->output->ne[0], ggml_graph_n_nodes(g->graph),
            ggml_backend_sched_get_n_splits(g->sched), (double) compute_bytes / (1024.0 * 1024.0));
    return true;
}

// Run one full decode. `src` points at 64*T contiguous F32, channel-major
// (memory index = c*T + t — see file header). `dst` receives S*2 floats
// planar (ch0 then ch1), S read back from the graph's own output shape.
// YUE2_VAE_PROFILE=1 splits each call into ensure_graph() (graph build/
// alloc when the tile width changes, ~free on a cache hit) vs the actual
// upload/compute/readback, so a slow tile can be attributed to one or the
// other instead of guessed at. Added to settle whether the VAE stage's
// poor measured GFLOPS/GB-s utilization is compute-bound or just paying
// for a fresh multi-GB compute-buffer allocation on every tile-width
// change (yue2_vae_ensure_graph logs a fresh build separately already).
static bool yue2_vae_step_profile_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("YUE2_VAE_PROFILE");
        return e && e[0] && e[0] != '0';
    }();
    return enabled;
}

static bool yue2_vae_run(const Yue2Model & m, Yue2VaeGraph * g, const float * src, int64_t T,
                         std::vector<float> * dst, int64_t * out_samples, std::string * err) {
    const bool profile = yue2_vae_step_profile_enabled();
    const bool was_cached = profile && g->graph && g->graph_T == T;
    auto t0 = std::chrono::steady_clock::now();

    if (!yue2_vae_ensure_graph(m, g, T, err)) {
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();

    ggml_backend_tensor_set(g->input, src, 0, ggml_nbytes(g->input));
    const auto t2 = std::chrono::steady_clock::now();
    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "VAE graph compute failed";
        }
        return false;
    }
    const auto t3 = std::chrono::steady_clock::now();
    const int64_t S = g->output->ne[0];
    dst->resize((size_t) (2 * S));
    ggml_backend_tensor_get(g->output, dst->data(), 0, (size_t) (2 * S) * sizeof(float));
    const auto t4 = std::chrono::steady_clock::now();
    if (out_samples) {
        *out_samples = S;
    }
    if (profile) {
        const auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr,
                "[YuE2-Vae-Profile] T=%lld cached=%s ensure_ms=%.1f upload_ms=%.1f compute_ms=%.1f readback_ms=%.1f total_ms=%.1f\n",
                (long long) T, was_cached ? "true" : "false", ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4),
                ms(t0, t4));
    }
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

static Yue2VaeGraph g_yue2_vae;

// Untiled ("full=True") decode: one graph over the whole song. `latent_ct`
// is 64*T floats, channel-major (see file header — this is a TRANSPOSE of
// the fixture's own on-disk [T,64] row-major layout; the caller transposes
// when reading the fixture, never this function). `out_planar` is resized to
// 2*S and filled [ch0 S floats][ch1 S floats], S = natural_output_length(T)
// as actually computed by the decoder graph (never hardcoded — 02-fixture-
// schema.md v1.4 §7's own point about querying the model, not a literal).
//
// Not thread-safe: caller serializes (mirrors mm3_vocoder_decode's contract).
static bool yue2_vae_decode(Yue2Model & m, const float * latent_ct, int64_t T, std::vector<float> * out_planar,
                            int64_t * out_samples, std::string * err = nullptr) {
    if (T <= 0) {
        if (err) {
            *err = "frames must be > 0";
        }
        return false;
    }
    if (!yue2_vae_prepare(m, &g_yue2_vae, err)) {
        return false;
    }
    return yue2_vae_run(m, &g_yue2_vae, latent_ct, T, out_planar, out_samples, err);
}

struct Yue2VaeTileBoundary {
    int64_t tile_index = 0;
    int64_t start = 0, end = 0, left = 0, right = 0;
    int64_t out_start = 0, out_end = 0, crop_start = 0;
};

// Tiled decode, exactly per 03-reference-numerics.md §4.1's pseudocode: each
// tile [left,right) (core +/- halo, clipped at the song's own boundaries) is
// an INDEPENDENT full decode through the whole stack, then cropped back to
// the core's own output range and placed with a direct copy — no crossfade,
// no zero-padding at the ends. `ratio` is read from the loaded model's own
// config (m.vae_cfg.downsampling_ratio), never hardcoded.
static bool yue2_vae_decode_tiled(Yue2Model & m, const float * latent_ct, int64_t T, int64_t core_frames,
                                  int64_t halo_frames, std::vector<float> * out_planar, int64_t * out_samples,
                                  std::vector<Yue2VaeTileBoundary> * boundaries, std::string * err = nullptr) {
    if (T <= 0 || core_frames <= 0) {
        if (err) {
            *err = "frames and core_frames must be > 0";
        }
        return false;
    }
    if (!yue2_vae_prepare(m, &g_yue2_vae, err)) {
        return false;
    }
    if (halo_frames < (int64_t) m.vae_cfg.required_halo) {
        if (err) {
            *err = "halo_frames (" + std::to_string(halo_frames) + ") < required_halo (" +
                   std::to_string(m.vae_cfg.required_halo) + ")";
        }
        return false;
    }

    const int64_t ratio = (int64_t) m.vae_cfg.downsampling_ratio;

    // First pass: decode tile 0 to learn the true per-sample output length
    // relation, then size the assembled buffer. Every tile after the first
    // reuses whatever graph shape recurs; T varies per tile (interior tiles
    // have a full core+2*halo, edge tiles are clipped by the song boundary).
    if (boundaries) {
        boundaries->clear();
    }
    std::vector<float> tile_audio;
    std::vector<float> assembled;  // filled lazily once `total` is known
    int64_t             total = -1;

    Yue2VaeGraph * g    = &g_yue2_vae;
    int64_t        tidx = 0;
    for (int64_t start = 0; start < T; start += core_frames, tidx++) {
        const int64_t end   = std::min(T, start + core_frames);
        const int64_t left  = std::max((int64_t) 0, start - halo_frames);
        const int64_t right = std::min(T, end + halo_frames);
        const int64_t wl    = right - left;

        std::vector<float> tile_in((size_t) (64 * wl));
        for (int64_t c = 0; c < (int64_t) m.vae_cfg.latent_dim; c++) {
            memcpy(tile_in.data() + (size_t) (c * wl), latent_ct + (size_t) (c * T + left),
                   (size_t) wl * sizeof(float));
        }

        int64_t tile_samples = 0;
        if (!yue2_vae_run(m, g, tile_in.data(), wl, &tile_audio, &tile_samples, err)) {
            return false;
        }

        if (total < 0) {
            // natural_output_length is a property of the WHOLE song's frame
            // count, not this tile's — but every tile's decoder graph is the
            // exact same architecture, so the affine relation
            // total = ratio*T - (tile_samples - ratio*wl) holds from any one
            // tile's own (wl, tile_samples) pair. Compute it once here.
            const int64_t offset = ratio * wl - tile_samples;  // >= 0, the decoder's fixed length deficit
            total                = ratio * T - offset;
            assembled.assign((size_t) (2 * total), 0.0f);
        }

        const int64_t out_start  = start * ratio;
        const int64_t out_end    = std::min(end * ratio, total);
        const int64_t crop_start = (start - left) * ratio;
        const int64_t n          = out_end - out_start;

        for (int ch = 0; ch < 2; ch++) {
            memcpy(assembled.data() + (size_t) (ch * total + out_start),
                   tile_audio.data() + (size_t) (ch * tile_samples + crop_start), (size_t) n * sizeof(float));
        }

        if (boundaries) {
            Yue2VaeTileBoundary tb;
            tb.tile_index  = tidx;
            tb.start       = start;
            tb.end         = end;
            tb.left        = left;
            tb.right       = right;
            tb.out_start   = out_start;
            tb.out_end     = out_end;
            tb.crop_start  = crop_start;
            boundaries->push_back(tb);
        }
    }

    *out_planar = std::move(assembled);
    if (out_samples) {
        *out_samples = total;
    }
    return true;
}
