#pragma once
// yue2/yue2-vae-encode.h — YuE2 Oobleck VAE *encoder* graph (NAR-LoRA phase 1).
//
// HOT-Step file (does not exist upstream). Sibling of yue2-vae-graph.h, which
// owns the decoder; this file adds the forward direction so the NAR trainer
// can turn audio into the `[T, 64]` posterior-mean latents it trains on
// (docs/plans/yue2/08-nar-lora-trainer.md §2 item 1 / §6 row 1).
//
// Included only by bring-up tooling (engine/tools/yue2-probe.cpp) and, later,
// the trainer's preprocess stage. hot-step-server.cpp is NOT touched.
//
// Ground truth: the reference `modeling_vae.py` (OobleckEncoder /
// EncoderBlock / ResidualUnit / SnakeBeta, and YuE2VAE.encode). Tensor names
// and shapes: docs/plans/yue2/04-weights-inventory.md §3, bound into
// Yue2VaeEncBlock by yue2-model.h's yue2_load_vae_tensors(want_encoder=true).
// Weight-norm is already folded at conversion time, so every conv here is a
// plain Conv1d over the stored weight.
//
// ── Architecture (encoder-only; dec.* is never touched here) ───────────────
//
//   audio [S, 2]     planar stereo (memory index = ch*S + s), which is the
//                    EXACT layout yue2_vae_build EMITS. encode(decode(z)) and
//                    decode(encode(x)) need no host-side shuffling.
//
//   enc.conv_in   Conv1d 2 -> 64, k=7, p=3, WITH bias
//   6x EncoderBlock, strides FORWARD [2,2,4,4,5,6],
//                    widths 64 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048:
//       3x ResidualUnit(dilation in {1,3,9}), all at Cin:
//           skip -> SnakeBeta -> Conv1d k=7 dilation d pad=3d -> SnakeBeta
//                -> Conv1d k=1 -> +skip
//       SnakeBeta(Cin)                              -- AFTER the res units
//       Conv1d(Cin->Cout, k=2*stride, stride=stride, pad=ceil(stride/2))
//   enc.snake_out SnakeBeta(2048)
//   enc.conv_out  Conv1d 2048 -> 128, k=3, p=1, WITH bias
//
//   This is the exact MIRROR of the decoder: the decoder's block is
//   SnakeBeta -> upsample -> 3x ResidualUnit (activation FIRST), the
//   encoder's is 3x ResidualUnit -> SnakeBeta -> downsample (activation
//   LAST). Getting that order backwards compiles and runs and is silently
//   wrong, so it is stated here and checked against modeling_vae.py, not
//   inferred from the decoder.
//
//   The 128 output channels are `[mean || scale]` (torch chunk(2, dim=1)).
//   YuE2VAE.encode() with sample=False returns the posterior MEAN and never
//   touches `scale`, so this file drops channels 64..127 outright. There is
//   no sampling path here on purpose — the trainer's contract is the mean.
//
// ── Length relation ────────────────────────────────────────────────────────
//
//   Every conv except the 6 strided downsamples is length-preserving
//   (k=7/p=3, k=7/p=3d dilated, k=1/p=0, k=3/p=1). Each downsample is
//   PyTorch's Conv1d formula with k=2s, p=ceil(s/2):
//
//       L <- floor( (L + 2*ceil(s/2) - (2s-1) - 1) / s ) + 1
//
//   yue2_vae_enc_frames() walks that over the loaded config's own strides,
//   so it follows whatever the GGUF says rather than a literal. For the
//   SHIPPED strides [2,2,4,4,5,6] (product 1920) it closes to
//
//       T(S) = floor( ( floor(S / 64) + 1 ) / 30 )
//
//   which equals floor(S / 1920) for every S with (S mod 1920) < 1856, and
//   is floor(S / 1920) + 1 for the last 64 samples of each frame period.
//   So "T = floor(S/1920)" is exact on frame-aligned input (the only case
//   the trainer feeds it) but is NOT a general identity — the probe checks
//   the exact formula, not the convenient one.
//
// ── Tiling ─────────────────────────────────────────────────────────────────
//
//   The composite map is shift-equivariant with period 1920: output frame f
//   depends on input samples [f*1920 - 18991, f*1920 + 20846] (computed by
//   walking the reference's own _dependency_interval rule backwards through
//   conv_in, 6x(3 res units + downsample), conv_out). Hence, for a tile that
//   starts at a sample offset that is a MULTIPLE of 1920:
//
//       left halo  >= 18991 samples   (9.89 frames)
//       right halo >= 18927 samples   (9.86 frames)
//
//   YUE2_VAE_ENC_REQUIRED_HALO_SAMPLES rounds that up to 10 frames = 19200
//   samples; the default is double it (20 frames) for the same kind of
//   margin the reference's decoder ships (required 12, ships 16). With a
//   halo at or above the requirement each tile is an INDEPENDENT full encode
//   cropped back to its own core — no crossfade, no blending, no overlap
//   reuse — mirroring yue2_vae_decode_tiled's posture exactly.
//
//   VRAM note: this graph is im2col-based, and the first block runs at 64
//   channels over the FULL sample rate, so a k=7 conv there materialises
//   [64*7, S] F32 — roughly 1.3 GB per 10 s of audio. That, not
//   YUE2_VAE_ENC_UNTILED_MAX_T, is the practical ceiling on the untiled
//   path; tile anything long.
//
// ── Precision — TF32 MUST BE OFF, and nothing in ggml does that for you ────
//
//   Same contract as the decoder (03-reference-numerics.md §6: autocast off,
//   TF32 off, FP32 weights): explicit F32 im2col, no F16 anywhere, never
//   ggml_conv_1d (which forces an F16 kernel path).
//
//   Keeping every tensor F32 is NOT enough on CUDA. ggml creates its cuBLAS
//   handle with CUBLAS_TF32_TENSOR_OP_MATH (ggml/src/ggml-cuda/common.cuh),
//   so every F32 ggml_mul_mat — i.e. every conv in this file — runs on TF32
//   tensor cores with an 11-bit mantissa. There is no per-op opt-out:
//   GGML_PREC_F32 only steers the F16 path, and the F32 branch of
//   ggml_cuda_mul_mat_cublas is a plain cublasSgemm on that handle.
//
//   This encoder AMPLIFIES that rounding hard. Measured against the
//   encode-v1/first-song oracle (30 s, T=750) on an RTX 5090:
//
//       TF32 on (ggml default)   rel-L2 5.0e-2   max|d| 4.7e-1   <- 50x the gate
//       TF32 off                 rel-L2 1.3e-5   max|d| 3.6e-4
//
//   It is a per-layer snowball, not a flat offset: conv_in already differs in
//   the 4th digit and each of the 6 blocks roughly doubles it (a Python
//   reference run with torch TF32 deliberately ENABLED lands at rel-L2 1.3e-2,
//   the same failure from the other side). The bias-dominated first samples of
//   every stage still match to 1e-7, which is exactly why this reads as "the
//   graph is basically right, but 5% off" rather than as a precision bug.
//
//   So the process must disable TF32 before it creates a CUDA context, which
//   is what yue2_vae_enc_disable_tf32() below is for. Call it from main();
//   yue2_vae_enc_prepare() warns loudly if nobody did.

#include "yue2-vae-graph.h"

#include "backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Node budget. Measured shape, per graph piece:
//   conv1d w/ bias   6 nodes (im2col, reshape col, reshape w, mul_mat,
//                             reshape bias, add)
//   SnakeBeta        5 nodes (mul, sin, sqr, mul, add)
//   ResidualUnit    23 nodes (snake + k7 conv + snake + k1 conv + add)
//   EncoderBlock    80 nodes (3*23 + snake 5 + strided conv 6)
// Total: conv_in 6 + 6*80 + snake_out 5 + conv_out 6 + mean view/cont 2
//        = 499. Cap at 1024 (>2x margin, same slack the decoder's 2048 has).
#define YUE2_VAE_ENC_MAX_NODES 1024

// Sanity guard on the untiled entry point, in FRAMES (mirrors the decoder's
// YUE2_VAE_UNTILED_MAX_T). VRAM bites long before this — see the header's
// VRAM note — this only stops an absurd request from trying to allocate.
#define YUE2_VAE_ENC_UNTILED_MAX_T 8192

// Minimum tile halo, in SAMPLES, for tiled == untiled. Derived above:
// max(18991 left, 18927 right) rounded up to a whole 1920-sample frame.
#define YUE2_VAE_ENC_REQUIRED_HALO_SAMPLES 19200
// Defaults: 30 s core (750 frames) and a 20-frame halo.
#define YUE2_VAE_ENC_DEFAULT_CORE_SAMPLES (1920 * 750)
#define YUE2_VAE_ENC_DEFAULT_HALO_SAMPLES (1920 * 20)

// ── TF32 opt-out ───────────────────────────────────────────────────────────

// Turn TF32 off for this whole process, the only lever that reaches ggml's
// own cuBLAS handle: the CUDA driver reads NVIDIA_TF32_OVERRIDE when the
// context is created, so this MUST run before the first ggml_backend_cuda_init
// (in practice: the first statement of main()). Setting it later is a silent
// no-op — the handle already exists — which is why this is a named function
// with a warning on the other side rather than a line buried in the graph
// builder. Harmless on non-CUDA backends and on pre-Ampere cards, neither of
// which has a TF32 path to disable.
//
// The repo already relies on this variable for the same reason in the DiT
// trainer's finite-difference gate (engine/src/train/dit-selftest.h); the
// difference is that gate spawns a child process to get it set early enough,
// and here we are early enough already.
static void yue2_vae_enc_disable_tf32() {
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", /*overwrite*/ 1);
#endif
}

// True when TF32 is off for this process — i.e. whoever owns main() called
// yue2_vae_enc_disable_tf32(), or the operator exported the variable.
static bool yue2_vae_enc_tf32_disabled() {
    const char * v = getenv("NVIDIA_TF32_OVERRIDE");
    return v && v[0] == '0' && v[1] == '\0';
}

// ── Derived weights + graph state ──────────────────────────────────────────

struct Yue2VaeEncPrepBlk {
    std::vector<Yue2VaePrepRes> res;         // 3 entries, dilations 1/3/9
    Yue2VaeSnakePrep            snake_post;  // AFTER the res units (mirror of the decoder)
};

struct Yue2VaeEncGraph {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            prep        = {};
    // Same identity trick yue2_vae_prepare uses: the VAE weight buffer
    // pointer, which changes whenever yue2_load_parts swaps variants (or
    // reloads with want_encoder flipped on).
    const void * weights_token = nullptr;

    std::vector<Yue2VaeEncPrepBlk> blk;
    Yue2VaeSnakePrep               snake_out;

    ggml_context * gctx    = nullptr;
    uint8_t *      gbuf    = nullptr;
    ggml_cgraph *  graph   = nullptr;
    ggml_tensor *  input   = nullptr;  // [S, 2]  F32 planar stereo (ne0=S, ne1=ch)
    ggml_tensor *  output  = nullptr;  // [T, 64] F32 channel-major latent MEAN
    int64_t        graph_S = 0;
};

// ── Length arithmetic ──────────────────────────────────────────────────────

// Frames produced for `S` input samples, by walking the config's own strides
// through PyTorch's Conv1d output-length rule. Returns 0 when the input is
// too short for a single frame. See the header's "Length relation" note for
// the closed form on the shipped strides.
static int64_t yue2_vae_enc_frames(const Yue2VaeConfig & c, int64_t S) {
    int64_t L = S;
    if (L <= 0) {
        return 0;
    }
    for (size_t i = 0; i < c.strides.size(); i++) {
        const int64_t s   = (int64_t) c.strides[i];
        if (s <= 0) {
            return 0;
        }
        const int64_t p   = (s + 1) / 2;  // ceil(s/2)
        const int64_t k   = 2 * s;
        const int64_t num = L + 2 * p - (k - 1) - 1;
        if (num < 0) {
            return 0;
        }
        L = num / s + 1;
    }
    return L;
}

// ── Prep / free ────────────────────────────────────────────────────────────

static void yue2_vae_enc_free_graph(Yue2VaeEncGraph * g) {
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
    g->graph_S = 0;
}

static void yue2_vae_enc_graph_free(Yue2VaeEncGraph * g) {
    yue2_vae_enc_free_graph(g);
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

// Build (or rebuild) the encoder's derived-weight buffer (exp_alpha /
// inv_beta per SnakeBeta — there is no ConvTranspose1d to repack on this
// side) and its scheduler. Cheap after the first call.
static bool yue2_vae_enc_prepare(const Yue2Model & m, Yue2VaeEncGraph * g, std::string * err) {
    if (!m.vae_resident) {
        if (err) {
            *err = "YuE2 VAE is not resident (load it first)";
        }
        return false;
    }
    if (!m.vae.enc_loaded) {
        if (err) {
            *err = "YuE2 VAE encoder weights are not loaded — call yue2_load_parts(..., want_encoder=true); "
                   "the VAE loads decoder-only by default";
        }
        return false;
    }
    const void * token = (const void *) m.wctx_vae.buffer;
    if (g->weights_token == token && g->sched) {
        return true;
    }
    yue2_vae_enc_graph_free(g);

    const Yue2VaeConfig &  vc  = m.vae_cfg;
    const Yue2VaeWeights & vw  = m.vae;
    const int              NB  = (int) vc.strides.size();
    const int              NR  = (int) vc.res_dilations.size();
    const float            eps = vc.snake_eps > 0.0f ? vc.snake_eps : 1e-9f;

    if ((int) vw.enc_blk.size() != NB) {
        if (err) {
            *err = "VAE encoder block count (" + std::to_string(vw.enc_blk.size()) + ") != strides (" +
                   std::to_string(NB) + ")";
        }
        return false;
    }

    BackendPair bp = backend_init("YuE2-VaeEnc");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    // Per block: NR*(2 snakes, 4 tensors) + 1 snake_post (2 tensors).
    // Plus 1 snake_out (2 tensors).
    const int n_prep = NB * (NR * 4 + 2) + 2;
    wctx_init(&g->prep, n_prep);

    std::string e;
    bool        ok = true;
    g->blk.assign((size_t) NB, Yue2VaeEncPrepBlk{});
    for (int b = 0; b < NB && ok; b++) {
        char                    nm[96];
        Yue2VaeEncPrepBlk &     pb = g->blk[(size_t) b];
        const Yue2VaeEncBlock & wb = vw.enc_blk[(size_t) b];

        pb.res.assign((size_t) NR, Yue2VaePrepRes{});
        for (int r = 0; r < NR && ok; r++) {
            const Yue2VaeResUnit & wr = wb.res[(size_t) r];
            snprintf(nm, sizeof(nm), "vae.enc.blk.%d.res.%d.snake1", b, r);
            ok = yue2_vae_make_snake_prep(&g->prep, wr.snake1_alpha, wr.snake1_beta, eps, nm,
                                          &pb.res[(size_t) r].snake1, &e);
            if (!ok) {
                break;
            }
            snprintf(nm, sizeof(nm), "vae.enc.blk.%d.res.%d.snake2", b, r);
            ok = yue2_vae_make_snake_prep(&g->prep, wr.snake2_alpha, wr.snake2_beta, eps, nm,
                                          &pb.res[(size_t) r].snake2, &e);
        }
        if (ok) {
            snprintf(nm, sizeof(nm), "vae.enc.blk.%d.snake_post", b);
            ok = yue2_vae_make_snake_prep(&g->prep, wb.snake_post_alpha, wb.snake_post_beta, eps, nm, &pb.snake_post,
                                          &e);
        }
    }
    if (ok) {
        ok = yue2_vae_make_snake_prep(&g->prep, vw.enc_snake_out_alpha, vw.enc_snake_out_beta, eps,
                                      "vae.enc.snake_out", &g->snake_out, &e);
    }
    if (ok) {
        ok = wctx_alloc(&g->prep, g->backend);
        if (!ok) {
            e = "backend buffer allocation failed for the VAE encoder derived weights";
        }
    }
    if (!ok) {
        if (err) {
            *err = e.empty() ? "VAE encoder prepare failed" : e;
        }
        yue2_vae_enc_graph_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, YUE2_VAE_ENC_MAX_NODES * 2);
    g->weights_token = token;

    const size_t prep_bytes = g->prep.buffer ? ggml_backend_buffer_get_size(g->prep.buffer) : 0;
    fprintf(stderr, "[YuE2-VaeEnc] Prepared (%s): %d derived tensors, %.1f MB (exp_alpha/inv_beta)\n",
            vc.variant.c_str(), n_prep, (double) prep_bytes / (1024.0 * 1024.0));
    if (!yue2_vae_enc_tf32_disabled()) {
        fprintf(stderr,
                "[YuE2-VaeEnc] WARNING: NVIDIA_TF32_OVERRIDE is not \"0\". On CUDA every conv in this\n"
                "              encoder is an F32 ggml_mul_mat, and ggml's cuBLAS handle runs those on\n"
                "              TF32 tensor cores — measured at rel-L2 5e-2 vs the oracle instead of\n"
                "              1.3e-5. Call yue2_vae_enc_disable_tf32() at the top of main().\n");
    }
    return true;
}

// ── Graph ──────────────────────────────────────────────────────────────────

// Full encoder. audio [S, 2] planar -> latent MEAN [T, 64] channel-major
// (memory index = c*T + t) — the same layout yue2_vae_decode CONSUMES, so an
// encode result can be handed straight back to the decoder.
static ggml_tensor * yue2_vae_enc_build(ggml_context * ctx, const Yue2Model & m, const Yue2VaeEncGraph & g,
                                        ggml_tensor * audio) {
    const Yue2VaeConfig &  vc = m.vae_cfg;
    const Yue2VaeWeights & vw = m.vae;
    const int              NB = (int) vc.strides.size();

    // enc.conv_in: 2 -> C0, k=7, p=3, WITH bias.
    ggml_tensor * x = yue2_vae_conv1d(ctx, vw.enc_conv_in_w, vw.enc_conv_in_b, audio, /*pad*/ 3, /*dilation*/ 1);

    for (int b = 0; b < NB; b++) {
        const Yue2VaeEncBlock &   wb     = vw.enc_blk[(size_t) b];
        const Yue2VaeEncPrepBlk & pb     = g.blk[(size_t) b];
        const int                 stride = vc.strides[(size_t) b];  // encoder runs strides FORWARD

        // Res units FIRST, activation LAST — the mirror of the decoder block.
        for (size_t r = 0; r < vc.res_dilations.size(); r++) {
            x = yue2_vae_res_unit(ctx, wb.res[r], pb.res[r], x, (int) vc.res_dilations[r]);
        }
        x = yue2_vae_snake_beta(ctx, x, pb.snake_post);
        // Strided Conv1d, k = 2*stride, pad = ceil(stride/2).
        x = yue2_vae_conv1d(ctx, wb.downsample_w, wb.downsample_b, x, /*pad*/ (stride + 1) / 2, /*dilation*/ 1,
                            /*stride*/ stride);
    }

    x = yue2_vae_snake_beta(ctx, x, g.snake_out);
    // enc.conv_out: C_last -> 128, k=3, p=1, WITH bias. 128 = [mean || scale].
    x = yue2_vae_conv1d(ctx, vw.enc_conv_out_w, vw.enc_conv_out_b, x, /*pad*/ 1, /*dilation*/ 1);

    // Posterior MEAN = chunk(2, dim=1)[0] = channels 0..latent_dim-1. In this
    // channel-major layout (ne0=T fastest, ne1=C) that is the contiguous
    // leading prefix, so the view is free; ggml_cont only makes the shape
    // honest for the readback.
    const int64_t T  = x->ne[0];
    const int64_t LD = (int64_t) vc.latent_dim;
    x                = ggml_cont(ctx, ggml_view_2d(ctx, x, T, LD, x->nb[1], 0));
    return x;
}

static bool yue2_vae_enc_ensure_graph(const Yue2Model & m, Yue2VaeEncGraph * g, int64_t S, std::string * err) {
    if (g->graph && g->graph_S == S) {
        return true;
    }
    yue2_vae_enc_free_graph(g);

    const size_t ctx_bytes = ggml_tensor_overhead() * (YUE2_VAE_ENC_MAX_NODES + 64) +
                             ggml_graph_overhead_custom(YUE2_VAE_ENC_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the VAE encoder graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the VAE encoder graph context";
        }
        return false;
    }

    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, (int64_t) m.vae_cfg.audio_channels);
    ggml_set_name(g->input, "yue2_vae_enc_in");
    ggml_set_input(g->input);

    g->output = yue2_vae_enc_build(ctx, m, *g, g->input);
    ggml_set_name(g->output, "yue2_vae_enc_out");
    ggml_set_output(g->output);

    g->graph = ggml_new_graph_custom(ctx, YUE2_VAE_ENC_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->output);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "VAE encoder graph allocation failed (out of VRAM?) for S=" + std::to_string(S);
        }
        return false;
    }

    g->gctx    = ctx;
    g->graph_S = S;
    const size_t compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);
    fprintf(stderr, "[YuE2-VaeEnc] Graph: S=%lld samples/ch -> T=%lld, %d nodes, %d splits, compute buffer %.0f MB\n",
            (long long) S, (long long) g->output->ne[0], ggml_graph_n_nodes(g->graph),
            ggml_backend_sched_get_n_splits(g->sched), (double) compute_bytes / (1024.0 * 1024.0));
    return true;
}

// Run one full encode. `src` points at audio_channels*S contiguous F32,
// planar (ch0's S samples, then ch1's). `dst` receives latent_dim*T floats,
// channel-major, T read back from the graph's own output shape.
static bool yue2_vae_enc_run(const Yue2Model & m, Yue2VaeEncGraph * g, const float * src, int64_t S,
                             std::vector<float> * dst, int64_t * out_frames, std::string * err) {
    if (!yue2_vae_enc_ensure_graph(m, g, S, err)) {
        return false;
    }
    ggml_backend_tensor_set(g->input, src, 0, ggml_nbytes(g->input));
    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "VAE encoder graph compute failed";
        }
        return false;
    }
    const int64_t T  = g->output->ne[0];
    const int64_t LD = g->output->ne[1];
    dst->resize((size_t) (LD * T));
    ggml_backend_tensor_get(g->output, dst->data(), 0, (size_t) (LD * T) * sizeof(float));
    if (out_frames) {
        *out_frames = T;
    }
    return true;
}

// ── Public API ─────────────────────────────────────────────────────────────

static Yue2VaeEncGraph g_yue2_vae_enc;

// Untiled encode: one graph over the whole clip.
//
//   audio_planar : audio_channels*S floats, planar (index = ch*S + s) — the
//                  exact layout yue2_vae_decode EMITS.
//   latent_out   : resized to latent_dim*T and filled channel-major
//                  (index = c*T + t) — the exact layout yue2_vae_decode
//                  CONSUMES as its `latent_ct` argument.
//   T_out        : frames actually produced (== yue2_vae_enc_frames(cfg, S)).
//
// Posterior MEAN only; the 64 `scale` channels are dropped inside the graph.
// Requires the VAE to have been loaded with want_encoder=true.
//
// Not thread-safe: caller serializes (mirrors yue2_vae_decode's contract).
static bool yue2_vae_encode(Yue2Model & m, const float * audio_planar, int64_t S, std::vector<float> * latent_out,
                            int64_t * T_out, std::string * err = nullptr) {
    if (S <= 0) {
        if (err) {
            *err = "samples must be > 0";
        }
        return false;
    }
    if (!yue2_vae_enc_prepare(m, &g_yue2_vae_enc, err)) {
        return false;
    }
    const int64_t T = yue2_vae_enc_frames(m.vae_cfg, S);
    if (T < 1) {
        if (err) {
            *err = "audio is shorter than one latent frame (" + std::to_string(S) + " samples < " +
                   std::to_string(m.vae_cfg.downsampling_ratio) + ")";
        }
        return false;
    }
    if (T > YUE2_VAE_ENC_UNTILED_MAX_T) {
        if (err) {
            *err = "untiled encode would produce " + std::to_string(T) + " frames (cap " +
                   std::to_string((int64_t) YUE2_VAE_ENC_UNTILED_MAX_T) + ") — use yue2_vae_encode_tiled";
        }
        return false;
    }
    int64_t    got = 0;
    const bool ok  = yue2_vae_enc_run(m, &g_yue2_vae_enc, audio_planar, S, latent_out, &got, err);
    if (!ok) {
        return false;
    }
    if (got != T) {
        if (err) {
            *err = "VAE encoder length model disagrees with the graph: predicted " + std::to_string(T) + " frames, "
                   "graph produced " + std::to_string(got);
        }
        return false;
    }
    if (T_out) {
        *T_out = T;
    }
    return true;
}

struct Yue2VaeEncTile {
    int64_t tile_index = 0;
    // Sample-space geometry of this tile.
    int64_t start = 0, end = 0, left = 0, right = 0;
    // Frame-space placement of the tile's core.
    int64_t out_start = 0, out_end = 0, crop_start = 0;
};

// Tiled encode. `core_samples` and `halo_samples` are in SAMPLES and must
// both be positive multiples of downsampling_ratio (1920) — that alignment
// is what makes each tile's frame grid coincide with the whole clip's.
//
// Each tile [left, right) is an INDEPENDENT full encode through the whole
// stack, cropped back to its own core's frame range and placed with a direct
// copy: no crossfade, no blending, no overlap reuse. Same posture as
// yue2_vae_decode_tiled, for the same reason — the halo already covers the
// full receptive field, so there is nothing to smooth.
static bool yue2_vae_encode_tiled(Yue2Model & m, const float * audio_planar, int64_t S, int64_t core_samples,
                                  int64_t halo_samples, std::vector<float> * latent_out, int64_t * T_out,
                                  std::vector<Yue2VaeEncTile> * boundaries, std::string * err = nullptr) {
    if (S <= 0) {
        if (err) {
            *err = "samples must be > 0";
        }
        return false;
    }
    if (!yue2_vae_enc_prepare(m, &g_yue2_vae_enc, err)) {
        return false;
    }

    const int64_t ratio = (int64_t) m.vae_cfg.downsampling_ratio;
    const int64_t LD    = (int64_t) m.vae_cfg.latent_dim;
    const int64_t AC    = (int64_t) m.vae_cfg.audio_channels;
    if (ratio <= 0) {
        if (err) {
            *err = "downsampling_ratio is 0 — VAE config was not parsed";
        }
        return false;
    }
    if (core_samples <= 0 || core_samples % ratio != 0) {
        if (err) {
            *err = "core_samples (" + std::to_string(core_samples) + ") must be a positive multiple of " +
                   std::to_string(ratio);
        }
        return false;
    }
    if (halo_samples % ratio != 0) {
        if (err) {
            *err = "halo_samples (" + std::to_string(halo_samples) + ") must be a multiple of " +
                   std::to_string(ratio);
        }
        return false;
    }
    if (halo_samples < (int64_t) YUE2_VAE_ENC_REQUIRED_HALO_SAMPLES) {
        if (err) {
            *err = "halo_samples (" + std::to_string(halo_samples) + ") < the encoder's receptive field (" +
                   std::to_string((int64_t) YUE2_VAE_ENC_REQUIRED_HALO_SAMPLES) + " samples)";
        }
        return false;
    }

    const int64_t T = yue2_vae_enc_frames(m.vae_cfg, S);
    if (T < 1) {
        if (err) {
            *err = "audio is shorter than one latent frame (" + std::to_string(S) + " samples < " +
                   std::to_string(ratio) + ")";
        }
        return false;
    }

    if (boundaries) {
        boundaries->clear();
    }
    std::vector<float> assembled((size_t) (LD * T), 0.0f);
    std::vector<float> tile_in;
    std::vector<float> tile_out;

    Yue2VaeEncGraph * g    = &g_yue2_vae_enc;
    int64_t           tidx = 0;
    for (int64_t start = 0; start < S; start += core_samples, tidx++) {
        const int64_t end   = std::min(S, start + core_samples);
        const int64_t left  = std::max((int64_t) 0, start - halo_samples);
        const int64_t right = std::min(S, end + halo_samples);
        const int64_t wl    = right - left;

        const int64_t f_start = start / ratio;
        if (f_start >= T) {
            break;  // tail shorter than a frame; the previous tile already owns everything
        }
        // The last core runs to the end of the clip, so it owns every
        // remaining frame — including the extra frame the exact length
        // formula can produce in the last 64 samples of a frame period.
        const int64_t f_end = (start + core_samples >= S) ? T : std::min(T, (start + core_samples) / ratio);

        tile_in.resize((size_t) (AC * wl));
        for (int64_t ch = 0; ch < AC; ch++) {
            memcpy(tile_in.data() + (size_t) (ch * wl), audio_planar + (size_t) (ch * S + left),
                   (size_t) wl * sizeof(float));
        }

        int64_t tile_frames = 0;
        if (!yue2_vae_enc_run(m, g, tile_in.data(), wl, &tile_out, &tile_frames, err)) {
            return false;
        }

        const int64_t crop_start = (start - left) / ratio;
        const int64_t n          = f_end - f_start;
        if (n <= 0) {
            continue;
        }
        if (crop_start + n > tile_frames) {
            if (err) {
                *err = "VAE encoder tile " + std::to_string(tidx) + " did not cover its core (" +
                       std::to_string(tile_frames) + " frames, needed " + std::to_string(crop_start + n) + ")";
            }
            return false;
        }

        for (int64_t c = 0; c < LD; c++) {
            memcpy(assembled.data() + (size_t) (c * T + f_start),
                   tile_out.data() + (size_t) (c * tile_frames + crop_start), (size_t) n * sizeof(float));
        }

        if (boundaries) {
            Yue2VaeEncTile tb;
            tb.tile_index = tidx;
            tb.start      = start;
            tb.end        = end;
            tb.left       = left;
            tb.right      = right;
            tb.out_start  = f_start;
            tb.out_end    = f_end;
            tb.crop_start = crop_start;
            boundaries->push_back(tb);
        }

        if (f_end >= T) {
            break;
        }
    }

    *latent_out = std::move(assembled);
    if (T_out) {
        *T_out = T;
    }
    return true;
}
