#pragma once
// bs-roformer-ggml.h: native GGML implementation of BS-RoFormer.
//
// Replaces the ONNX Runtime path for BS-Roformer-Leap "Xe" (used by
// SUPERSEP_STABLESTEP). Weights come from a GGUF produced by
// scripts/convert-bs-roformer-gguf.py.
//
// WHY GGML AND NOT ONNX
// ---------------------
// Two reasons, one of which is the whole ballgame:
//
//  1. VRAM. The time transformer attends over T (=1722 at the trained 20 s
//     chunk) with the 90 bands folded into the batch. A materialised score
//     tensor is 90 * 8 heads * 1722^2 * 4 B = 8.5 GB, and softmax needs the
//     probs alive alongside it. That is what the old ONNX Runtime path paid (see the
//     "~3GB per MatMul" note in supersep.cpp) and why stem separation is both
//     slow and VRAM-hungry. ggml_flash_attn_ext never materialises it.
//
//  2. Portability. No ONNX Runtime + cuDNN dependency (~1.3 GB of DLLs), and
//     the CUDA / Vulkan / Metal / CPU backends all work.
//
// The reference is ZFTurbo's Music-Source-Separation-Training BSRoformer
// (models_without_stft/bs_roformer_no_stft.py) — the STFT/iSTFT-amputated
// variant, because SuperSep does both natively in supersep-stft.h and applies
// the returned mask itself.
//
// CONTRACT (identical to the ONNX graph it replaces)
// --------------------------------------------------
//     input   [in_dim, T]           in_dim = (n_fft/2+1) * n_ch * 2 = 4100
//                                   element order (f * C + ch) * 2 + {re,im}
//     output  [2, T, F*C] per stem  read linearly = [fs][t][{re,im}]
//                                   i.e. supersep.cpp's (fs_idx * T + t) * 2
//
// TWO GOTCHAS WORTH THE INK
// -------------------------
//  * RoPE is INTERLEAVED (GGML_ROPE_TYPE_NORMAL), not NeoX. Upstream uses
//    rotary_embedding_torch, whose rotate_half pairs ADJACENT elements
//    ('... (d r) -> ... d r', r=2). Using NEOX here compiles, runs, and
//    produces confident garbage.
//  * RMSNorm here is F.normalize(x, dim=-1) * sqrt(dim) * gamma. That is
//    algebraically x / sqrt(mean(x^2)) * gamma, i.e. exactly ggml_rms_norm
//    followed by a gamma multiply — despite not looking like it.
//
// Part of HOT-Step CPP. MIT license.

#ifndef HOT_STEP_BS_ROFORMER_GGML_H
#define HOT_STEP_BS_ROFORMER_GGML_H

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

// F.normalize clamps the norm at 1e-12; mirror that as the rms_norm epsilon.
#define BSR_NORM_EPS 1e-12f
#define BSR_ROPE_THETA 10000.0f

// ── config ──────────────────────────────────────────────────────────────────

struct BsRoformerConfig {
    int dim         = 0;
    int depth       = 0;
    int heads       = 0;
    int dim_head    = 0;
    int dim_inner   = 0;  // heads * dim_head
    int ff_inner    = 0;  // dim * mlp_expansion_factor
    int n_bands     = 0;
    int n_stems     = 0;
    int n_channels  = 0;
    int n_fft       = 0;
    int hop_length  = 0;
    int win_length  = 0;
    int chunk_size  = 0;
    int in_dim      = 0;  // sum(band_widths)

    int mask_layers = 2;  // Linears in each per-band mask MLP
    int has_out_norm = 0; // per-Transformer trailing RMSNorm present
    int has_final_norm = 1;

    std::vector<int> band_widths;   // feature width per band (bins * ch * 2)
    std::vector<int> band_offsets;  // prefix sum of band_widths
    std::string      target;        // "vocals" | "other"
    std::string      arch;          // "bs" | "mel"
};

// ── weights ─────────────────────────────────────────────────────────────────

struct BsrAttn {
    struct ggml_tensor * norm    = nullptr;
    struct ggml_tensor * qkv     = nullptr;
    struct ggml_tensor * gates_w = nullptr;
    struct ggml_tensor * gates_b = nullptr;
    struct ggml_tensor * out     = nullptr;
};

struct BsrFF {
    struct ggml_tensor * norm = nullptr;
    struct ggml_tensor * w1   = nullptr;
    struct ggml_tensor * b1   = nullptr;
    struct ggml_tensor * w2   = nullptr;
    struct ggml_tensor * b2   = nullptr;
};

// One layer = a time transformer then a freq transformer, each depth 1
// (one Attention + one FeedForward). *_out_norm is the Transformer's trailing
// RMSNorm and is null when the checkpoint was built with norm_output=False —
// true for BS-RoFormer SW and Leap Xe, false for Mel-Band Karaoke.
struct BsrBlock {
    BsrAttn time_attn, freq_attn;
    BsrFF   time_ff, freq_ff;
    struct ggml_tensor * time_out_norm = nullptr;
    struct ggml_tensor * freq_out_norm = nullptr;
};

struct BsrBand {
    struct ggml_tensor * norm = nullptr;  // RMSNorm gamma [width]
    struct ggml_tensor * w    = nullptr;  // Linear [width, dim]
    struct ggml_tensor * b    = nullptr;  // [dim]
};

// Per-band mask MLP: N Linears with Tanh between, the last emitting width*2
// for the GLU. N is 2 for BS-RoFormer, 3 for Mel-Band Karaoke.
struct BsrMaskBand {
    std::vector<struct ggml_tensor *> w;
    std::vector<struct ggml_tensor *> b;
};

struct BsRoformer {
    BsRoformerConfig cfg;

    ggml_backend_t              backend     = nullptr;
    ggml_backend_t              cpu_backend = nullptr;
    ggml_backend_sched_t        sched       = nullptr;
    WeightCtx                   wctx        = {};

    std::vector<BsrBand>                   bands;
    std::vector<BsrBlock>                  blocks;
    struct ggml_tensor *                   final_norm = nullptr;
    std::vector<std::vector<BsrMaskBand>>  mask;  // [stem][band]

    bool use_flash_attn = true;

    // Validation hooks (see scripts/dump_bs_roformer_goldens.py):
    //   -1 off, 0 after band_split, 1..depth after layer i-1,
    //   depth+1 after final_norm. Captured tensor is [dim, n_bands, T].
    int                debug_stage = -1;
    std::vector<float> debug_out;
};

// ── loading ─────────────────────────────────────────────────────────────────

static bool bsr_load(BsRoformer * m, const char * gguf_path) {
    BackendPair bp = backend_init("BS-RoFormer");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 16384);

    GGUFModel gf = {};
    if (!gf_load(&gf, gguf_path)) {
        fprintf(stderr, "[BSR] FATAL: cannot load %s\n", gguf_path);
        return false;
    }

    BsRoformerConfig & c = m->cfg;
    c.dim        = (int) gf_get_u32(gf, "bs_roformer.dim");
    c.depth      = (int) gf_get_u32(gf, "bs_roformer.depth");
    c.heads      = (int) gf_get_u32(gf, "bs_roformer.heads");
    c.dim_head   = (int) gf_get_u32(gf, "bs_roformer.dim_head");
    c.dim_inner  = (int) gf_get_u32(gf, "bs_roformer.dim_inner");
    c.ff_inner   = (int) gf_get_u32(gf, "bs_roformer.ff_inner");
    c.n_bands    = (int) gf_get_u32(gf, "bs_roformer.n_bands");
    c.n_stems    = (int) gf_get_u32(gf, "bs_roformer.n_stems");
    c.n_channels = (int) gf_get_u32(gf, "bs_roformer.n_channels");
    c.n_fft      = (int) gf_get_u32(gf, "bs_roformer.n_fft");
    c.hop_length = (int) gf_get_u32(gf, "bs_roformer.hop_length");
    c.win_length = (int) gf_get_u32(gf, "bs_roformer.win_length");
    c.chunk_size = (int) gf_get_u32(gf, "bs_roformer.chunk_size");
    c.target     = gf_get_str(gf, "bs_roformer.target_instrument");
    c.arch       = gf_get_str(gf, "bs_roformer.arch");
    if (c.arch.empty()) c.arch = "bs";
    c.mask_layers    = (int) gf_get_u32(gf, "bs_roformer.mask_layers");
    if (c.mask_layers <= 0) c.mask_layers = 2;
    c.has_out_norm   = gf_get_bool(gf, "bs_roformer.has_out_norm") ? 1 : 0;
    c.has_final_norm = gf_get_bool(gf, "bs_roformer.has_final_norm") ? 1 : 0;

    if (c.dim <= 0 || c.depth <= 0 || c.n_bands <= 0) {
        fprintf(stderr, "[BSR] FATAL: %s has no bs_roformer.* metadata — "
                        "was it produced by scripts/convert-bs-roformer-gguf.py?\n",
                gguf_path);
        gf_close(&gf);
        return false;
    }

    // Band widths live in a KV array.
    {
        int64_t idx = gguf_find_key(gf.gguf, "bs_roformer.band_widths");
        if (idx < 0 || (int) gguf_get_arr_n(gf.gguf, idx) != c.n_bands) {
            fprintf(stderr, "[BSR] FATAL: bs_roformer.band_widths missing or "
                            "not %d entries\n", c.n_bands);
            gf_close(&gf);
            return false;
        }
        const int32_t * arr = (const int32_t *) gguf_get_arr_data(gf.gguf, idx);
        c.band_widths.assign(arr, arr + c.n_bands);
    }
    c.band_offsets.resize((size_t) c.n_bands);
    c.in_dim = 0;
    for (int b = 0; b < c.n_bands; b++) {
        c.band_offsets[(size_t) b] = c.in_dim;
        c.in_dim += c.band_widths[(size_t) b];
    }

    // For the contiguous band split (BS-RoFormer) the widths must tile the full
    // spectrum. Mel-Band bands OVERLAP and the caller gathers freq_indices
    // before the graph, so its in_dim is the gathered size instead — see
    // mel_band_tables.inc / mel_band_process_chunk in supersep.cpp.
    if (c.arch == "bs") {
        const int expect_in = (c.n_fft / 2 + 1) * c.n_channels * 2;
        if (c.in_dim != expect_in) {
            fprintf(stderr, "[BSR] FATAL: band widths sum to %d, expected %d\n",
                    c.in_dim, expect_in);
            gf_close(&gf);
            return false;
        }
    }

    // Exact tensor budget (undersizing this trips a GGML_ASSERT deep in
    // ggml_new_object rather than anything readable):
    //   band split      3 per band
    //   transformer     (11 + out_norm) per (layer, axis), 2 axes
    //   final_norm      0 or 1
    //   mask estimator  2 per MLP layer, per band, per stem
    const int n_weights = 3 * c.n_bands
                        + 2 * c.depth * (11 + (c.has_out_norm ? 1 : 0))
                        + (c.has_final_norm ? 1 : 0)
                        + 2 * c.mask_layers * c.n_stems * c.n_bands;
    wctx_init(&m->wctx, n_weights + 32);

    char nm[128];

    m->bands.resize((size_t) c.n_bands);
    for (int b = 0; b < c.n_bands; b++) {
        BsrBand & bd = m->bands[(size_t) b];
        snprintf(nm, sizeof(nm), "band_split.%d.norm", b);
        bd.norm = gf_load_tensor_f32(&m->wctx, gf, nm);
        snprintf(nm, sizeof(nm), "band_split.%d.w", b);
        bd.w = gf_load_tensor_f32(&m->wctx, gf, nm);
        snprintf(nm, sizeof(nm), "band_split.%d.b", b);
        bd.b = gf_load_tensor_f32(&m->wctx, gf, nm);
    }

    // The converter exports rope_freqs so a checkpoint trained with
    // learned_freq=True can be detected. We compute RoPE from theta=10000
    // internally, so verify the stored table matches before trusting it.
    auto check_rope = [&](const char * name) {
        const float * f = (const float *) gf_get_data(gf, name);
        if (!f) return;
        const int n = c.dim_head / 2;
        for (int i = 0; i < n; i++) {
            float want = 1.0f / powf(BSR_ROPE_THETA, (float) (2 * i) / (float) c.dim_head);
            if (fabsf(f[i] - want) > 1e-4f * fmaxf(1.0f, fabsf(want))) {
                fprintf(stderr,
                        "[BSR] FATAL: %s deviates from the theta=%.0f progression "
                        "at index %d (%.6g vs %.6g).\n"
                        "      This checkpoint has LEARNED rotary frequencies, which "
                        "this implementation does not support.\n",
                        name, (double) BSR_ROPE_THETA, i, (double) f[i], (double) want);
                exit(1);
            }
        }
    };

    m->blocks.resize((size_t) c.depth);
    for (int i = 0; i < c.depth; i++) {
        BsrBlock & blk = m->blocks[(size_t) i];
        for (int axis = 0; axis < 2; axis++) {
            const char * an = axis == 0 ? "time" : "freq";
            BsrAttn & at = axis == 0 ? blk.time_attn : blk.freq_attn;
            BsrFF   & ff = axis == 0 ? blk.time_ff   : blk.freq_ff;

            snprintf(nm, sizeof(nm), "blk.%d.%s.rope_freqs", i, an);
            check_rope(nm);

            snprintf(nm, sizeof(nm), "blk.%d.%s.attn_norm", i, an);
            at.norm = gf_load_tensor_f32(&m->wctx, gf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.%s.qkv", i, an);
            at.qkv = gf_load_tensor_f32(&m->wctx, gf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.%s.gates_w", i, an);
            at.gates_w = gf_load_tensor_f32(&m->wctx, gf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.%s.gates_b", i, an);
            at.gates_b = gf_load_tensor_f32(&m->wctx, gf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.%s.out", i, an);
            at.out = gf_load_tensor_f32(&m->wctx, gf, nm);

            snprintf(nm, sizeof(nm), "blk.%d.%s.ff_norm", i, an);
            ff.norm = gf_load_tensor_f32(&m->wctx, gf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.%s.ff1_w", i, an);
            ff.w1 = gf_load_tensor_f32(&m->wctx, gf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.%s.ff1_b", i, an);
            ff.b1 = gf_load_tensor_f32(&m->wctx, gf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.%s.ff2_w", i, an);
            ff.w2 = gf_load_tensor_f32(&m->wctx, gf, nm);
            snprintf(nm, sizeof(nm), "blk.%d.%s.ff2_b", i, an);
            ff.b2 = gf_load_tensor_f32(&m->wctx, gf, nm);

            if (c.has_out_norm) {
                snprintf(nm, sizeof(nm), "blk.%d.%s.out_norm", i, an);
                struct ggml_tensor * t = gf_load_tensor_f32(&m->wctx, gf, nm);
                if (axis == 0) blk.time_out_norm = t; else blk.freq_out_norm = t;
            }
        }
    }

    if (c.has_final_norm) {
        m->final_norm = gf_load_tensor_f32(&m->wctx, gf, "final_norm");
    }

    m->mask.resize((size_t) c.n_stems);
    for (int s = 0; s < c.n_stems; s++) {
        m->mask[(size_t) s].resize((size_t) c.n_bands);
        for (int b = 0; b < c.n_bands; b++) {
            BsrMaskBand & mb = m->mask[(size_t) s][(size_t) b];
            for (int l = 0; l < c.mask_layers; l++) {
                snprintf(nm, sizeof(nm), "mask.%d.%d.w%d", s, b, l + 1);
                mb.w.push_back(gf_load_tensor_f32(&m->wctx, gf, nm));
                snprintf(nm, sizeof(nm), "mask.%d.%d.b%d", s, b, l + 1);
                mb.b.push_back(gf_load_tensor_f32(&m->wctx, gf, nm));
            }
        }
    }

    if (!wctx_alloc(&m->wctx, m->backend)) {
        fprintf(stderr, "[BSR] FATAL: weight allocation failed\n");
        gf_close(&gf);
        return false;
    }
    gf_close(&gf);

    fprintf(stderr,
            "[BSR] Loaded %s — dim=%d depth=%d heads=%d bands=%d stems=%d "
            "target=%s chunk=%d\n",
            gguf_path, c.dim, c.depth, c.heads, c.n_bands, c.n_stems,
            c.target.c_str(), c.chunk_size);
    return true;
}

// ── graph helpers ───────────────────────────────────────────────────────────

// RMSNorm: F.normalize(x, dim=-1) * sqrt(dim) * gamma  ==  rms_norm(x) * gamma
static struct ggml_tensor * bsr_rms(struct ggml_context * ctx, struct ggml_tensor * x,
                                    struct ggml_tensor * gamma) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, BSR_NORM_EPS), gamma);
}

// Attention over axis ne1 of x [dim, S, B], batching over ne2.
// positions must be [S] i32.
static struct ggml_tensor * bsr_attn(struct ggml_context * ctx, const BsRoformerConfig & c,
                                     const BsrAttn * a, struct ggml_tensor * x,
                                     struct ggml_tensor * positions, bool flash) {
    const int64_t D = c.dim_head;
    const int64_t H = c.heads;
    const int64_t DI = c.dim_inner;
    const int64_t S = x->ne[1];
    const int64_t B = x->ne[2];

    struct ggml_tensor * xn = bsr_rms(ctx, x, a->norm);

    // Fused qkv: torch packs (qkv, heads, dim_head) with qkv outermost.
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, a->qkv, xn);  // [3*DI, S, B]

    auto part = [&](int idx) {
        struct ggml_tensor * v = ggml_view_3d(ctx, qkv, DI, S, B, qkv->nb[1], qkv->nb[2],
                                              (size_t) idx * DI * sizeof(float));
        v = ggml_cont(ctx, v);
        v = ggml_reshape_4d(ctx, v, D, H, S, B);
        // RoPE wants the sequence on ne2 — it already is.
        return v;
    };

    struct ggml_tensor * q = part(0);
    struct ggml_tensor * k = part(1);
    struct ggml_tensor * v = part(2);

    // INTERLEAVED rotary (rotary_embedding_torch), NOT NeoX. Full head width.
    auto rope = [&](struct ggml_tensor * t) {
        return ggml_rope_ext(ctx, t, positions, NULL, (int) D, GGML_ROPE_TYPE_NORMAL, 0,
                             BSR_ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };
    q = rope(q);
    k = rope(k);

    // [D, H, S, B] -> [D, S, H, B]
    auto to_attn = [&](struct ggml_tensor * t) {
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
    };
    q = to_attn(q);
    k = to_attn(k);
    v = to_attn(v);

    const float scale = 1.0f / sqrtf((float) D);

    struct ggml_tensor * attn;
    if (flash) {
        // Flash attention is the reason this port exists: at T=1722 the
        // materialised score tensor would be 8.5 GB.
        struct ggml_tensor * kf = ggml_cast(ctx, k, GGML_TYPE_F16);
        struct ggml_tensor * vf = ggml_cast(ctx, v, GGML_TYPE_F16);
        attn = ggml_flash_attn_ext(ctx, q, kf, vf, NULL, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
        struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // [S_kv, S_q, H, B]
        struct ggml_tensor * probs  = ggml_soft_max_ext(ctx, scores, NULL, scale, 0.0f);
        attn = ggml_mul_mat(ctx, vt, probs);                    // [D, S_q, H, B]
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    }
    // Both paths now yield [D, H, S, B].

    // Per-head sigmoid gate (out * sigmoid(to_gates(x)), broadcast over D).
    struct ggml_tensor * g = ggml_mul_mat(ctx, a->gates_w, xn);  // [H, S, B]
    g = ggml_add(ctx, g, a->gates_b);
    g = ggml_sigmoid(ctx, g);
    // [H, S, B] and [1, H, S, B] have identical memory order, so this is free.
    g = ggml_reshape_4d(ctx, g, 1, H, S, B);
    attn = ggml_mul(ctx, attn, g);

    attn = ggml_reshape_3d(ctx, attn, DI, S, B);
    struct ggml_tensor * out = ggml_mul_mat(ctx, a->out, attn);  // [dim, S, B]
    return ggml_add(ctx, x, out);
}

static struct ggml_tensor * bsr_ff(struct ggml_context * ctx, const BsrFF * f,
                                   struct ggml_tensor * x) {
    struct ggml_tensor * h = bsr_rms(ctx, x, f->norm);
    h = ggml_add(ctx, ggml_mul_mat(ctx, f->w1, h), f->b1);
    h = ggml_gelu(ctx, h);
    h = ggml_add(ctx, ggml_mul_mat(ctx, f->w2, h), f->b2);
    return ggml_add(ctx, x, h);
}

// ── forward ─────────────────────────────────────────────────────────────────
//
// input:  [in_dim, T] f32, contiguous, element order (f*C+ch)*2 + {re,im}
// output: per stem, [2, T, fs] f32 — read linearly this is [fs][t][{re,im}],
//         matching supersep.cpp's (s*fs*T + fs_idx*T + t)*2 indexing.
// out must hold n_stems * in_dim * T floats.
static void bsr_forward(BsRoformer * m, const float * input, int T, float * out) {
    const BsRoformerConfig & c = m->cfg;
    const int64_t dim = c.dim;
    const int64_t NB  = c.n_bands;

    const size_t ctx_size = (size_t) 65536 * ggml_tensor_overhead() + ggml_graph_overhead_custom(65536, false);
    struct ggml_init_params gp = { ctx_size, NULL, true };
    struct ggml_context *   ctx   = ggml_init(gp);
    struct ggml_cgraph *    graph = ggml_new_graph_custom(ctx, 65536, false);

    struct ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.in_dim, T);
    ggml_set_name(x_in, "stft_in");
    ggml_set_input(x_in);

    struct ggml_tensor * pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos_t, "pos_time");
    ggml_set_input(pos_t);

    struct ggml_tensor * pos_f = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NB);
    ggml_set_name(pos_f, "pos_freq");
    ggml_set_input(pos_f);

    // ── band split ──────────────────────────────────────────────────────
    // Per band: RMSNorm(width) -> Linear(width, dim). Stacked to [dim, NB, T].
    struct ggml_tensor * h = nullptr;
    for (int b = 0; b < c.n_bands; b++) {
        const BsrBand & bd = m->bands[(size_t) b];
        const int64_t   w  = c.band_widths[(size_t) b];

        struct ggml_tensor * slice =
            ggml_view_2d(ctx, x_in, w, T, x_in->nb[1],
                         (size_t) c.band_offsets[(size_t) b] * sizeof(float));
        slice = ggml_cont(ctx, slice);
        slice = bsr_rms(ctx, slice, bd.norm);
        struct ggml_tensor * e = ggml_mul_mat(ctx, bd.w, slice);  // [dim, T]
        e = ggml_add(ctx, e, bd.b);
        e = ggml_reshape_3d(ctx, e, dim, 1, T);
        h = h ? ggml_concat(ctx, h, e, 1) : e;
    }
    // h: [dim, NB, T]

    // Debug capture must be materialised and expanded into the graph AT THE
    // POINT IT IS TAKEN. Deferring the ggml_cont to the end lets the scheduler
    // reuse the captured tensor's buffer in-place for a later op, and the cont
    // then reads whatever overwrote it — which looks exactly like a numerically
    // wrong layer while the model output stays perfect.
    // ggml_cont alone is not enough: the allocator may satisfy it in place, so
    // the "copy" aliases the very buffer a later op then reuses. Copy into a
    // tensor we own instead, and expand it immediately so its liveness starts
    // at the capture point rather than at the end of the graph.
    struct ggml_tensor * dbg = nullptr;
    auto capture = [&](int stage, struct ggml_tensor * t) {
        if (m->debug_stage != stage) return;
        struct ggml_tensor * dst =
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, t->ne[0], t->ne[1], t->ne[2]);
        dbg = ggml_cpy(ctx, ggml_cont(ctx, t), dst);
        ggml_set_name(dbg, "debug_stage");
        ggml_set_output(dbg);
        ggml_build_forward_expand(graph, dbg);
    };
    capture(0, h);

    // ── axial transformer ───────────────────────────────────────────────
    for (int i = 0; i < c.depth; i++) {
        const BsrBlock & blk = m->blocks[(size_t) i];

        // time: [dim, NB, T] -> [dim, T, NB], attend over T, back again
        h = ggml_cont(ctx, ggml_permute(ctx, h, 0, 2, 1, 3));
        h = bsr_attn(ctx, c, &blk.time_attn, h, pos_t, m->use_flash_attn);
        h = bsr_ff(ctx, &blk.time_ff, h);
        if (blk.time_out_norm) h = bsr_rms(ctx, h, blk.time_out_norm);
        h = ggml_cont(ctx, ggml_permute(ctx, h, 0, 2, 1, 3));

        // freq: attend over NB directly (already ne1)
        h = bsr_attn(ctx, c, &blk.freq_attn, h, pos_f, m->use_flash_attn);
        h = bsr_ff(ctx, &blk.freq_ff, h);
        if (blk.freq_out_norm) h = bsr_rms(ctx, h, blk.freq_out_norm);

        capture(i + 1, h);
    }

    if (m->final_norm) h = bsr_rms(ctx, h, m->final_norm);  // [dim, NB, T]
    capture(c.depth + 1, h);

    // ── mask estimators ─────────────────────────────────────────────────
    // Per stem, per band: Linear -> Tanh -> Linear -> GLU, then concat bands.
    std::vector<struct ggml_tensor *> stem_out((size_t) c.n_stems, nullptr);
    for (int s = 0; s < c.n_stems; s++) {
        struct ggml_tensor * acc = nullptr;
        for (int b = 0; b < c.n_bands; b++) {
            const BsrMaskBand & mb = m->mask[(size_t) s][(size_t) b];
            const int64_t       w  = c.band_widths[(size_t) b];

            // band slice of [dim, NB, T] along ne1 -> [dim, T]
            struct ggml_tensor * xb =
                ggml_view_2d(ctx, h, dim, T, h->nb[2], (size_t) b * h->nb[1]);
            xb = ggml_cont(ctx, xb);

            // Linear -> Tanh -> ... -> Linear, ending at [2w, T] for the GLU.
            struct ggml_tensor * y = xb;
            for (size_t l = 0; l < mb.w.size(); l++) {
                y = ggml_add(ctx, ggml_mul_mat(ctx, mb.w[l], y), mb.b[l]);
                if (l + 1 < mb.w.size()) y = ggml_tanh(ctx, y);
            }

            // nn.GLU(dim=-1): first half * sigmoid(second half)
            struct ggml_tensor * lin =
                ggml_cont(ctx, ggml_view_2d(ctx, y, w, T, y->nb[1], 0));
            struct ggml_tensor * gate =
                ggml_cont(ctx, ggml_view_2d(ctx, y, w, T, y->nb[1], (size_t) w * sizeof(float)));
            struct ggml_tensor * band = ggml_mul(ctx, lin, ggml_sigmoid(ctx, gate));

            acc = acc ? ggml_concat(ctx, acc, band, 0) : band;
        }
        // acc: [in_dim, T] with ne0 index = fs_idx*2 + c
        // -> [2, fs, T] -> permute to [2, T, fs] so linear memory is
        //    fs-major, then t, then re/im.
        struct ggml_tensor * mk = ggml_reshape_3d(ctx, acc, 2, c.in_dim / 2, T);
        mk = ggml_cont(ctx, ggml_permute(ctx, mk, 0, 2, 1, 3));
        char nm[32];
        snprintf(nm, sizeof(nm), "mask%d", s);
        ggml_set_name(mk, nm);
        ggml_set_output(mk);
        stem_out[(size_t) s] = mk;
        ggml_build_forward_expand(graph, mk);
    }

    if (!ggml_backend_sched_alloc_graph(m->sched, graph)) {
        fprintf(stderr, "[BSR] FATAL: failed to allocate graph (T=%d)\n", T);
        exit(1);
    }

    ggml_backend_tensor_set(x_in, input, 0, (size_t) c.in_dim * T * sizeof(float));
    {
        std::vector<int32_t> p((size_t) (T > NB ? T : NB));
        for (int i = 0; i < (int) p.size(); i++) p[(size_t) i] = i;
        ggml_backend_tensor_set(pos_t, p.data(), 0, (size_t) T * sizeof(int32_t));
        ggml_backend_tensor_set(pos_f, p.data(), 0, (size_t) NB * sizeof(int32_t));
    }

    ggml_backend_sched_graph_compute(m->sched, graph);

    for (int s = 0; s < c.n_stems; s++) {
        ggml_backend_tensor_get(stem_out[(size_t) s],
                                out + (size_t) s * c.in_dim * T, 0,
                                (size_t) c.in_dim * T * sizeof(float));
    }

    if (dbg) {
        m->debug_out.resize((size_t) dim * NB * T);
        ggml_backend_tensor_get(dbg, m->debug_out.data(), 0,
                                m->debug_out.size() * sizeof(float));
    }

    ggml_backend_sched_reset(m->sched);
    ggml_free(ctx);
}

static void bsr_free(BsRoformer * m) {
    if (!m) return;
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
        m->sched = nullptr;
    }
    wctx_free(&m->wctx);
    // MUST be backend_release, never ggml_backend_free: backend_init hands out
    // a refcounted process-wide singleton (backend.h g_backend_cache). Freeing
    // it directly destroys the backend for every other model in the process,
    // and the next backend_init hands back the dangling pointer because the
    // refcount was never decremented — an access violation on the SECOND model
    // loaded, with the first one having worked perfectly.
    backend_release(m->backend, m->cpu_backend);
    m->backend     = nullptr;
    m->cpu_backend = nullptr;
}

#endif  // HOT_STEP_BS_ROFORMER_GGML_H
