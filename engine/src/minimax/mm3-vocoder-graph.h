#pragma once
// minimax/mm3-vocoder-graph.h — MiniMax-Music3 vocoder (DAC-style decoder) graph.
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-server.h,
// which is itself the single hook include into tools/hot-step-server.cpp.
//
// SCOPE (increment 2): the first real inference code of the MM3 port. Takes flow
// latents [128, L] and produces stereo PCM at mm3.voc.sampling_rate. Nothing else
// in the pipeline (LM / depth / cond / DiT) exists yet — this is validated
// standalone through POST /mm3/voc-decode.
//
// ── Architecture (contract: docs/plans/mm3-gguf-layout.md §3.5 + §10) ──────────
//
//   latents [128, L]                          (channel-major, time contiguous)
//     └─ stereo fold: channels 0..63 -> audio ch 0, 64..127 -> audio ch 1.
//        Both run through the SAME weights. This is a batch fold in the
//        reference ([B,128,T] -> [B*2,64,T]); here it is two sequential passes
//        over one cached graph, because ggml_conv_transpose_1d's GEMM
//        emulation below is written for a 2D [T, C] activation.
//
//   dec_in   Conv1d 64 -> 1024, k=1, p=0
//   conv_in  Conv1d 1024 -> 1536, k=7, p=3
//   4 x upsample block, rates (8,8,4,2), ladder 1536->768->384->192->96:
//       Snake1d(alpha[Cin])
//       ConvTranspose1d(Cin->Cout, k=2r, stride=r, pad=ceil(r/2))
//       3 x ResidualUnit(d in {1,3,9}):
//           skip -> Snake -> Conv1d k=7 dilation d pad=3d -> Snake -> Conv1d k=1 -> +skip
//   Snake1d(alpha[96]) -> Conv1d 96 -> 1, k=7, p=3 -> tanh
//
//   Total upsampling x512. conv_out has ONE output channel — stereo comes from
//   the fold, never from conv_out (layout doc §10 trap note).
//
// ── Design decisions, and why ─────────────────────────────────────────────────
//
// 1. SNAKE = COMPOSITE, NOT A CUSTOM OP.
//    MM3's Snake is `x + 1/(alpha+eps) * sin(alpha*x)^2` (layout doc §3.5,
//    mm3.voc.snake). Emitted as the naive 5-op chain mul -> sin -> sqr -> mul ->
//    add, which is exactly the pattern this repo's CUDA and Vulkan backends
//    already fuse into a single snake kernel (ggml-cuda.cu "Snake activation"
//    fusion at the mul/sin/sqr/mul/add subgraph; ggml-vulkan.cpp snake_pattern).
//    A custom op would be slower on CUDA (no fusion) and would need a kernel per
//    backend. The fusion has hard requirements we satisfy deliberately:
//      - `a` and `inv_b` are both F32, contiguous, shape [1, C, 1, 1];
//      - x and the trailing add are 2D and contiguous;
//      - the add reads the same x as the leading mul.
//    ACE-Step's own VAE (engine/src/vae.h) uses the identical decomposition, so
//    this is house style, not invention. NOTE the formula differs from ACE's
//    VAE: ACE stores log-alpha/log-beta and uses exp(); MM3 stores raw alpha and
//    the same alpha appears in both places. Do not copy vae.h's exp() handling.
//
// 2. inv_alpha IS PRECOMPUTED, NOT DIVIDED IN-GRAPH.
//    The fused kernel wants a multiply by inv_b, and 1/(alpha+eps) is constant.
//    All 29 reciprocals (11,616 floats total) are computed once into a small
//    derived-weight buffer. A ggml_div would both cost a per-sample divide and
//    break the fusion pattern.
//
// 3. CONVTRANSPOSE1D IS EMULATED BY GEMM + col2im, NOT ggml_conv_transpose_1d.
//    Two independent reasons:
//    (a) PADDING. ggml_conv_transpose_1d hard-asserts p0 == 0 (ggml.c). PyTorch
//        output length is (T-1)*s + k - 2p; ggml gives (T-1)*s + k. With
//        k = 2s and p = ceil(s/2) = s/2 (all four rates are even), the ggml
//        output is (T+1)*s and the wanted output is T*s, so the fix is a
//        symmetric crop of exactly s/2 = p samples per side. That crop is the
//        documented workaround (multi-backend-architecture.md §1.5 item 3) and
//        it applies to BOTH implementations.
//    (b) SPEED. The CUDA kernel (ggml-cuda/conv-transpose-1d.cu) is one thread
//        per output element with an inner loop over ALL T_in inputs and a
//        `continue` filter — O(out * IC * T_in), not O(out * IC * k/s). For
//        block 1 (out 6.3e6, IC 768, T_in 2048) that is ~1e13 iterations. It is
//        not usable at these sizes. The GEMM formulation (transpose -> mul_mat
//        -> reshape/permute/pad/acc col2im) is the same trick vae.h already uses
//        for ACE's VAE, and it runs on cuBLAS. The weights must be repacked from
//        the GGUF layout [k, OC, IC] to [IC, k*OC] for the mul_mat; that is done
//        once into the derived-weight buffer, not per decode.
//
// 4. CPU/CUDA PLACEMENT: everything is on the default backend (CUDA when
//    present). No op in this graph falls back — im2col/mul_mat/mul/sin/sqr/add/
//    transpose/cont/permute/pad/acc/view/tanh all have CUDA implementations, and
//    ggml_conv_transpose_1d (the one op that would have been a concern) is not
//    used. The scheduler is still built with the CPU backend as a second device
//    so a Vulkan/CPU-only build degrades rather than fails.
//
// 5. F32 ACTIVATIONS THROUGHOUT. ggml_conv_1d() forces its im2col to F16, which
//    would push both operands of every conv through half precision. The layout
//    doc pins the entire vocoder to F32 precisely because Snake is a periodic
//    non-linearity applied 29 times across 512x upsampling (§7). So this file
//    builds im2col explicitly with GGML_TYPE_F32. Cost: the im2col scratch is
//    2x larger (~350 MB at the largest conv for L=256). Bounded by chunking.
//
// 6. CHUNKING. A single graph handles L <= MM3_VOC_CHUNK (1024 latent frames,
//    ~11.9 s of audio) — comfortably above the reference's 689-latent DiT
//    window, so the normal path is always single-shot. Longer inputs are decoded
//    in overlapping windows with the overlap discarded in the audio domain. The
//    upsample factor is exactly 512, so the trim is exact and needs no rounding.

#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Largest L decoded in one graph. Above this, mm3_vocoder_decode windows.
#define MM3_VOC_CHUNK   1024
// Latent frames of context carried into each side of a window (discarded after).
#define MM3_VOC_OVERLAP 32
// Node budget. Measured shape is ~400 nodes; the cap is deliberately loose.
#define MM3_VOC_MAX_NODES 2048

// ── Derived weights (computed once from the loaded GGUF tensors) ─────────────

struct MM3VocPrepRes {
    ggml_tensor * inv1 = nullptr;  // 1/(snake1.alpha + eps)
    ggml_tensor * inv2 = nullptr;  // 1/(snake2.alpha + eps)
};

struct MM3VocPrepBlk {
    ggml_tensor *              inv     = nullptr;  // 1/(blk snake.alpha + eps)
    ggml_tensor *              convt_w = nullptr;  // repacked [IC, k*OC]
    std::vector<MM3VocPrepRes> res;
};

struct MM3VocGraph {
    // backend + residency
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            prep        = {};
    // Identity of the weights this prep was derived from. If the model is
    // unloaded and reloaded the buffer pointer changes and prep is rebuilt.
    const void * weights_token = nullptr;

    std::vector<MM3VocPrepBlk> blk;
    ggml_tensor *              inv_out = nullptr;

    // cached graph (rebuilt when L changes)
    ggml_context * gctx    = nullptr;
    uint8_t *      gbuf    = nullptr;
    ggml_cgraph *  graph   = nullptr;
    ggml_tensor *  input   = nullptr;  // [L, 64] F32
    ggml_tensor *  output  = nullptr;  // [L*512, 1] F32
    int64_t        graph_L = 0;
};

// ── Small helpers ───────────────────────────────────────────────────────────

// Read an F32 backend tensor back to host. Returns false if it is not F32
// (the layout doc pins the whole vocoder F32; anything else means a converter
// change we must notice loudly rather than silently mis-read).
static bool mm3_voc_readback(const ggml_tensor * t, std::vector<float> * out, std::string * err,
                             const char * what) {
    if (!t) {
        if (err) {
            *err = std::string("vocoder tensor missing: ") + what;
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("vocoder tensor '") + what + "' is not F32 (type " + std::to_string((int) t->type) +
                   "); the layout contract pins the vocoder to F32";
        }
        return false;
    }
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get((ggml_tensor *) t, out->data(), 0, ggml_nbytes(t));
    return true;
}

// Create a [ne0, ne1] F32 tensor in wctx and queue `data` for upload.
// Ownership of `data` moves into wctx->staging so the pointer stays valid
// until wctx_alloc runs.
static ggml_tensor * mm3_voc_stage(WeightCtx * wctx, int64_t ne0, int64_t ne1,
                                   std::unique_ptr<float[]> data, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    const size_t nbytes = (size_t) ne0 * (size_t) ne1 * sizeof(float);
    wctx->pending.push_back({ t, data.get(), nbytes, 0 });
    wctx->staging.push_back(std::move(data));
    return t;
}

// 1/(alpha + eps) for a [1, C] alpha tensor.
static ggml_tensor * mm3_voc_make_inv(WeightCtx * wctx, const ggml_tensor * alpha, float eps,
                                      const char * name, std::string * err) {
    std::vector<float> a;
    if (!mm3_voc_readback(alpha, &a, err, name)) {
        return nullptr;
    }
    const int64_t C   = (int64_t) a.size();
    auto          inv = std::make_unique<float[]>((size_t) C);
    for (int64_t i = 0; i < C; i++) {
        inv[(size_t) i] = 1.0f / (a[(size_t) i] + eps);
    }
    return mm3_voc_stage(wctx, 1, C, std::move(inv), name);
}

// Repack a ConvTranspose1d kernel from the GGUF layout to the GEMM layout.
//   src: [K, OC, IC]   idx = k + oc*K + ic*K*OC     (ggml/torch (in,out,k))
//   dst: [IC, K*OC]    idx = ic + (k + oc*K)*IC
// mul_mat(dst, x^T) then contracts over IC and yields [K*OC, T_in], which the
// col2im below slices as [K, OC, T_in].
static ggml_tensor * mm3_voc_repack_convt(WeightCtx * wctx, const ggml_tensor * w, const char * name,
                                          std::string * err) {
    std::vector<float> src;
    if (!mm3_voc_readback(w, &src, err, name)) {
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
    return mm3_voc_stage(wctx, IC, K * OC, std::move(dst), name);
}

// ── Prep / free ─────────────────────────────────────────────────────────────

static void mm3_vocoder_free_graph(MM3VocGraph * g) {
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
    g->graph_L = 0;
}

static void mm3_vocoder_free(MM3VocGraph * g) {
    mm3_vocoder_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    wctx_free(&g->prep);
    g->blk.clear();
    g->inv_out       = nullptr;
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

// Build (or rebuild) the derived-weight buffer and the scheduler. Cheap after
// the first call: returns immediately when the model's synth buffer is the same
// one the current prep was derived from.
static bool mm3_vocoder_prepare(const MM3Model & m, MM3VocGraph * g, std::string * err) {
    // Staged residency: a stage-2 graph needs only the cond/dit/voc buffer.
    if (!m.rest_resident) {
        if (err) {
            *err = "MiniMax-Music3 is not warm (POST /mm3/warm first)";
        }
        return false;
    }
    const void * token = (const void *) m.wctx_synth.buffer;
    if (g->weights_token == token && g->sched) {
        return true;
    }
    mm3_vocoder_free(g);

    const MM3VocConfig &  vc = m.synth_cfg.voc;
    const MM3VocWeights & vw = m.synth.voc;
    const int             NB = (int) vc.upsample_rates.size();
    const int             NR = (int) vc.res_dilations.size();
    const float           eps = vc.snake_eps > 0.0f ? vc.snake_eps : 1e-9f;

    BackendPair bp = backend_init("MM3-Voc");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    // 1 inv per block snake + 2 per residual unit + 1 output snake, plus one
    // repacked convt kernel per block.
    const int n_prep = NB * (1 + 1 + NR * 2) + 1;
    wctx_init(&g->prep, n_prep);

    std::string e;
    bool        ok = true;
    g->blk.assign((size_t) NB, MM3VocPrepBlk{});
    for (int b = 0; b < NB && ok; b++) {
        char nm[96];
        MM3VocPrepBlk & pb = g->blk[(size_t) b];

        snprintf(nm, sizeof(nm), "voc.blk.%d.snake.inv_alpha", b);
        pb.inv = mm3_voc_make_inv(&g->prep, vw.blk[(size_t) b].snake_alpha, eps, nm, &e);
        ok     = ok && pb.inv != nullptr;

        snprintf(nm, sizeof(nm), "voc.blk.%d.convt.gemm", b);
        if (ok) {
            pb.convt_w = mm3_voc_repack_convt(&g->prep, vw.blk[(size_t) b].convt_w, nm, &e);
            ok         = pb.convt_w != nullptr;
        }

        pb.res.assign((size_t) NR, MM3VocPrepRes{});
        for (int r = 0; r < NR && ok; r++) {
            snprintf(nm, sizeof(nm), "voc.blk.%d.res.%d.snake1.inv_alpha", b, r);
            pb.res[(size_t) r].inv1 =
                mm3_voc_make_inv(&g->prep, vw.blk[(size_t) b].res[(size_t) r].snake1_alpha, eps, nm, &e);
            ok = pb.res[(size_t) r].inv1 != nullptr;
            if (!ok) {
                break;
            }
            snprintf(nm, sizeof(nm), "voc.blk.%d.res.%d.snake2.inv_alpha", b, r);
            pb.res[(size_t) r].inv2 =
                mm3_voc_make_inv(&g->prep, vw.blk[(size_t) b].res[(size_t) r].snake2_alpha, eps, nm, &e);
            ok = pb.res[(size_t) r].inv2 != nullptr;
        }
    }
    if (ok) {
        g->inv_out = mm3_voc_make_inv(&g->prep, vw.snake_out_alpha, eps, "voc.snake_out.inv_alpha", &e);
        ok         = g->inv_out != nullptr;
    }
    if (ok) {
        ok = wctx_alloc(&g->prep, g->backend);
        if (!ok) {
            e = "backend buffer allocation failed for the vocoder derived weights";
        }
    }
    if (!ok) {
        if (err) {
            *err = e.empty() ? "vocoder prepare failed" : e;
        }
        mm3_vocoder_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, MM3_VOC_MAX_NODES * 2);
    g->weights_token = token;

    const size_t prep_bytes = g->prep.buffer ? ggml_backend_buffer_get_size(g->prep.buffer) : 0;
    fprintf(stderr, "[MM3-Voc] Prepared: %d derived tensors, %.1f MB (inv-alpha + repacked convT)\n", n_prep,
            (double) prep_bytes / (1024.0 * 1024.0));
    return true;
}

// ── Graph pieces ────────────────────────────────────────────────────────────

// Snake1d: y = x + inv_alpha * sin(alpha * x)^2
// x [T, C]; alpha and inv_alpha [1, C] F32 contiguous.
// The op order below is load-bearing — it is the pattern the CUDA/Vulkan
// backends fuse. Do not "simplify" it.
static ggml_tensor * mm3_voc_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha,
                                   ggml_tensor * inv_alpha) {
    ggml_tensor * ax = ggml_mul(ctx, x, alpha);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_sqr(ctx, s);
    ggml_tensor * d  = ggml_mul(ctx, s2, inv_alpha);
    return ggml_add(ctx, x, d);
}

// Conv1d (+bias), stride 1. w [K, IC, OC], x [T, IC] -> [T_out, OC].
// Explicit F32 im2col (see header note 5) instead of ggml_conv_1d's forced F16.
static ggml_tensor * mm3_voc_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x,
                                    int pad, int dilation) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, /*s0*/ 1, /*s1*/ 0, pad, 0, dilation, 0, /*is_2D*/ false,
                                    GGML_TYPE_F32);  // [IC*K, OL, 1, 1]
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                   ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));  // [OL, OC]
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

// ConvTranspose1d (+bias) with k = 2*stride and pad = stride/2, via GEMM+col2im.
//   wg  [IC, K*OC] repacked (index ic + (k + oc*K)*IC)
//   x   [T_in, IC]
//   ->  [T_in*stride, OC]
//
// Length math, spelled out because the crop is the whole trick:
//   PyTorch:  T_torch = (T_in-1)*s + k - 2p            with k=2s, p=s/2  ->  T_in*s
//   ggml/full: T_full = (T_in-1)*s + k                                   ->  (T_in+1)*s
//   T_full - T_torch = s = 2p, split symmetrically -> drop p samples per side.
static ggml_tensor * mm3_voc_convt(ggml_context * ctx, ggml_tensor * wg, ggml_tensor * b, ggml_tensor * x,
                                   int stride, int64_t oc) {
    const int64_t T_in   = x->ne[0];
    const int     S      = stride;
    const int     K      = 2 * S;
    const int64_t T_full = (T_in - 1) * S + K;  // == (T_in + 1) * S
    const int     p      = (S + 1) / 2;         // ceil(s/2); s is even for all four rates

    // [T_in, IC] -> [IC, T_in], then GEMM contracting over IC.
    ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor * col = ggml_mul_mat(ctx, wg, xt);                     // [K*OC, T_in]
    ggml_tensor * c3  = ggml_reshape_3d(ctx, col, K, oc, T_in);        // [K, OC, T_in]

    // col2im. Output position of tap k for input i is i*S + k, and k spans
    // [0, 2S). Split at S: the low half tiles output positions [0, S*T_in)
    // exactly once, the high half tiles [S, S*T_in + S) exactly once, and the
    // two overlap-add.
    ggml_tensor * lo = ggml_view_3d(ctx, c3, S, oc, T_in, c3->nb[1], c3->nb[2], 0);
    lo               = ggml_cont(ctx, ggml_permute(ctx, lo, 0, 2, 1, 3));  // [S, T_in, OC]
    lo               = ggml_reshape_2d(ctx, lo, (int64_t) S * T_in, oc);
    ggml_tensor * y  = ggml_pad(ctx, lo, S, 0, 0, 0);                      // [S*T_in + S, OC] = [T_full, OC]

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

// ResidualUnit: skip + conv2(snake2(conv1(snake1(x))))
// With pad = 3*dilation on the k=7 conv and pad 0 on the k=1 conv, the residual
// length always equals the skip length, so the reference's centre-crop is a
// no-op here. Guarded anyway.
static ggml_tensor * mm3_voc_res_unit(ggml_context * ctx, const MM3VocResUnit & w, const MM3VocPrepRes & p,
                                      ggml_tensor * x, int dilation) {
    ggml_tensor * skip = x;
    ggml_tensor * y    = mm3_voc_snake(ctx, x, w.snake1_alpha, p.inv1);
    y                  = mm3_voc_conv1d(ctx, w.conv1_w, w.conv1_b, y, 3 * dilation, dilation);
    y                  = mm3_voc_snake(ctx, y, w.snake2_alpha, p.inv2);
    y                  = mm3_voc_conv1d(ctx, w.conv2_w, w.conv2_b, y, 0, 1);
    if (y->ne[0] != skip->ne[0]) {
        const int64_t off = (skip->ne[0] - y->ne[0]) / 2;
        skip = ggml_cont(ctx, ggml_view_2d(ctx, skip, y->ne[0], skip->ne[1], skip->nb[1],
                                           (size_t) off * sizeof(float)));
    }
    return ggml_add(ctx, skip, y);
}

// Full decoder for ONE 64-channel stream. latent [L, 64] -> audio [L*512, 1].
static ggml_tensor * mm3_voc_build(ggml_context * ctx, const MM3Model & m, const MM3VocGraph & g,
                                   ggml_tensor * latent) {
    const MM3VocConfig &  vc = m.synth_cfg.voc;
    const MM3VocWeights & vw = m.synth.voc;

    ggml_tensor * x = mm3_voc_conv1d(ctx, vw.dec_in_w, vw.dec_in_b, latent, 0, 1);   // 64 -> 1024, k=1
    x               = mm3_voc_conv1d(ctx, vw.conv_in_w, vw.conv_in_b, x, 3, 1);      // 1024 -> 1536, k=7 p=3

    for (size_t b = 0; b < vc.upsample_rates.size(); b++) {
        const MM3VocBlock &   wb = vw.blk[b];
        const MM3VocPrepBlk & pb = g.blk[b];
        const int             s  = (int) vc.upsample_rates[b];
        const int64_t         oc = wb.convt_w->ne[1];

        x = mm3_voc_snake(ctx, x, wb.snake_alpha, pb.inv);
        x = mm3_voc_convt(ctx, pb.convt_w, wb.convt_b, x, s, oc);
        for (size_t r = 0; r < vc.res_dilations.size(); r++) {
            x = mm3_voc_res_unit(ctx, wb.res[r], pb.res[r], x, (int) vc.res_dilations[r]);
        }
    }

    x = mm3_voc_snake(ctx, x, vw.snake_out_alpha, g.inv_out);
    x = mm3_voc_conv1d(ctx, vw.conv_out_w, vw.conv_out_b, x, 3, 1);  // 96 -> 1, k=7 p=3
    if (vc.final_tanh) {
        x = ggml_tanh(ctx, x);
    }
    return x;
}

// Build the graph for a given window length, or reuse the cached one.
static bool mm3_voc_ensure_graph(const MM3Model & m, MM3VocGraph * g, int64_t L, std::string * err) {
    if (g->graph && g->graph_L == L) {
        return true;
    }
    mm3_vocoder_free_graph(g);

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (MM3_VOC_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_VOC_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the vocoder graph context";
        }
        return false;
    }
    ggml_init_params ip = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the vocoder graph context";
        }
        return false;
    }

    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, (int64_t) m.synth_cfg.voc.fold_channels);
    ggml_set_name(g->input, "mm3_voc_in");
    ggml_set_input(g->input);

    g->output = mm3_voc_build(ctx, m, *g, g->input);
    ggml_set_name(g->output, "mm3_voc_out");
    ggml_set_output(g->output);

    g->graph = ggml_new_graph_custom(ctx, MM3_VOC_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->output);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "vocoder graph allocation failed (out of VRAM?) for L=" + std::to_string(L);
        }
        return false;
    }

    g->gctx    = ctx;
    g->graph_L = L;
    // The compute buffer is the memory story of this graph: the k=7 convs'
    // F32 im2col dominates it (7*C*T*4 bytes at the widest point). Logged so a
    // future VRAM budget for the full MM3 pipeline has a real number, not a
    // guess, and so a chunk-size change is visible in its effect.
    const size_t compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);
    fprintf(stderr, "[MM3-Voc] Graph: L=%lld -> %lld samples, %d nodes, %d splits, compute buffer %.0f MB\n",
            (long long) L, (long long) g->output->ne[0], ggml_graph_n_nodes(g->graph),
            ggml_backend_sched_get_n_splits(g->sched), (double) compute_bytes / (1024.0 * 1024.0));
    return true;
}

// Run one 64-channel stream. `src` points at 64*L contiguous F32 (channel-major,
// time contiguous — exactly the memory order of a torch [1,64,L] slice).
static bool mm3_voc_run(const MM3Model & m, MM3VocGraph * g, const float * src, int64_t L, float * dst,
                        std::string * err) {
    if (!mm3_voc_ensure_graph(m, g, L, err)) {
        return false;
    }
    (void) L;
    ggml_backend_tensor_set(g->input, src, 0, ggml_nbytes(g->input));
    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "vocoder graph compute failed";
        }
        return false;
    }
    ggml_backend_tensor_get(g->output, dst, 0, (size_t) g->output->ne[0] * sizeof(float));
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

static MM3VocGraph g_mm3_voc;

// Decode flow latents to stereo PCM.
//
//   latents      128*L floats, channel-major (channel c at latents + c*L),
//                i.e. the memory order of a torch [1, 128, L] tensor.
//   out_stereo   resized to 2*L*512 and filled PLANAR: [ch0 T floats][ch1 T].
//                That is the layout audio_encode_wav_* expects.
//
// Not thread-safe: the caller serialises (mm3-server.h holds g_mm3_mutex).
// _with takes the graph state explicitly (the refiner owns its own); the plain
// mm3_vocoder_decode below is the MM3 backend's, on the process-wide g_mm3_voc.
static bool mm3_vocoder_decode_with(const MM3Model & m, MM3VocGraph * g, const float * latents, int64_t L,
                                   std::vector<float> & out_stereo, std::string * err = nullptr) {
    if (L <= 0) {
        if (err) {
            *err = "frames must be > 0";
        }
        return false;
    }
    if (!mm3_vocoder_prepare(m, g, err)) {
        return false;
    }

    const MM3VocConfig & vc  = m.synth_cfg.voc;
    const int64_t        up  = (int64_t) vc.total_upsample;
    const int64_t        FC  = (int64_t) vc.fold_channels;
    const int64_t        T   = L * up;
    out_stereo.assign((size_t) (2 * T), 0.0f);

    // Single-shot path. Channels 0..63 and 64..127 are each a contiguous run of
    // FC*L floats, so the graph input can be fed straight from `latents`.
    if (L <= MM3_VOC_CHUNK) {
        for (int ch = 0; ch < 2; ch++) {
            if (!mm3_voc_run(m, g, latents + (size_t) (ch * FC * L), L, out_stereo.data() + (size_t) (ch * T),
                             err)) {
                return false;
            }
        }
        return true;
    }

    // Windowed path: overlap-discard. Upsampling is exactly `up`, so the audio
    // trim is exact.
    // Window loop is OUTER and the channel loop INNER, so each distinct window
    // length builds its graph once and both channels reuse it. The other order
    // rebuilds on every alternation (992, 272, 992, 272, ...).
    const int64_t      ov   = MM3_VOC_OVERLAP;
    const int64_t      core = MM3_VOC_CHUNK - 2 * ov;
    std::vector<float> win;
    std::vector<float> tile;
    for (int64_t cs = 0; cs < L; cs += core) {
        const int64_t ce = cs + core < L ? cs + core : L;
        const int64_t ws = cs - ov > 0 ? cs - ov : 0;
        const int64_t we = ce + ov < L ? ce + ov : L;
        const int64_t wl = we - ws;

        win.resize((size_t) (FC * wl));
        tile.resize((size_t) (wl * up));
        for (int ch = 0; ch < 2; ch++) {
            for (int64_t c = 0; c < FC; c++) {
                memcpy(win.data() + (size_t) (c * wl), latents + (size_t) ((ch * FC + c) * L + ws),
                       (size_t) wl * sizeof(float));
            }
            if (!mm3_voc_run(m, g, win.data(), wl, tile.data(), err)) {
                return false;
            }
            // Upsampling is exactly `up`, so the trim is exact — no rounding.
            memcpy(out_stereo.data() + (size_t) (ch * T + cs * up), tile.data() + (size_t) ((cs - ws) * up),
                   (size_t) ((ce - cs) * up) * sizeof(float));
        }
    }
    return true;
}

static bool mm3_vocoder_decode(const MM3Model & m, const float * latents, int64_t L,
                               std::vector<float> & out_stereo, std::string * err = nullptr) {
    return mm3_vocoder_decode_with(m, &g_mm3_voc, latents, L, out_stereo, err);
}

// ── Vocoder-only holder (the MM3 latent refiner's decoder) ──────────────────
//
// The refiner decodes with the same DAV decoder as the MM3 backend but must not
// need the 12-23 GB MM3 model warm, so it loads just the voc tensors from the
// mm3-voc GGUF (or a legacy bundle) into an MM3Model that carries nothing else,
// with its own graph state. ModelStore owns one of these (MODEL_MM3_REFINER_VOC).

struct Mm3RefVoc {
    MM3Model    model;
    MM3VocGraph graph;
    size_t      vram = 0;
};

static void mm3_refvoc_free(Mm3RefVoc * v) {
    mm3_vocoder_free(&v->graph);  // graph state points into the weights: first
    wctx_free(&v->model.wctx_synth);
    if (v->model.backend_ref) {
        backend_release(v->model.backend, v->model.cpu_backend);
    }
    v->model = MM3Model{};
    v->vram  = 0;
}

static bool mm3_refvoc_load(Mm3RefVoc * v, const std::string & path, std::string * err) {
    mm3_refvoc_free(v);
    MM3Model &               m = v->model;
    std::vector<std::string> errs;
    MM3FileInfo              fi;
    mm3_probe_file(path, &fi, nullptr, &m.synth_cfg, 1u << MM3_R_VOC, &errs);
    if (!fi.probe_ok || !errs.empty()) {
        *err = errs.empty() ? path + ": not a readable MiniMax-Music3 vocoder GGUF" : errs[0];
        return false;
    }
    GGUFModel gf = {};
    if (!gf_load(&gf, path.c_str())) {
        *err = "cannot open " + path;
        return false;
    }
    const MM3VocConfig & c = m.synth_cfg.voc;
    wctx_init(&m.wctx_synth, 7 + (int) c.upsample_rates.size() * (3 + (int) c.res_dilations.size() * 6));
    MM3Loader ld{ &m.wctx_synth, &gf, nullptr, &errs };
    mm3_load_voc_tensors(&m, ld);
    if (!errs.empty()) {
        gf_close(&gf);
        mm3_refvoc_free(v);
        *err = errs[0];
        return false;
    }
    BackendPair bp = backend_init("MM3-RefVoc");
    m.backend      = bp.backend;
    m.cpu_backend  = bp.cpu_backend;
    m.backend_ref  = true;
    const bool ok  = wctx_alloc(&m.wctx_synth, m.backend);
    gf_close(&gf);
    if (!ok) {
        mm3_refvoc_free(v);
        *err = "backend buffer allocation failed for the refiner vocoder";
        return false;
    }
    v->vram         = m.wctx_synth.buffer ? ggml_backend_buffer_get_size(m.wctx_synth.buffer) : 0;
    m.vram_synth    = v->vram;
    m.rest_resident = true;  // what mm3_vocoder_prepare checks
    m.loaded        = true;
    return true;
}
