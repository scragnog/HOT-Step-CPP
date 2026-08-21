#pragma once
// minimax/mm3-dit-graph.h — MiniMax-Music3 flow-matching DiT (2.4B) + Euler sampler.
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-server.h,
// which is itself the single hook include into tools/hot-step-server.cpp.
//
// SCOPE (increment 3): one DiT forward pass, and the 30-step flow-matching Euler
// loop that drives it. The condition tensor is supplied by the caller — the
// condition encoder and the AR/LM stage that feed it are later increments, and
// the chunk-overlap orchestration (window 1 onwards) is a later increment too.
// Validated standalone through POST /mm3/dit-forward and POST /mm3/flow-sample
// against the diffusers reference dumps in mm3-weights/fixtures/.
//
// ── Architecture (contract: docs/plans/mm3-gguf-layout.md §3.4 + §10, and the
//    diffusers reference MiniMaxMusic3Transformer1DModel at commit dafe3733) ───
//
//   latents [128, L] , condition [2048, L] , t in [0, 1]
//     full = cat(latents, ZEROS(128), condition)            -> [2304, L]
//     full = preprocess_conv(full) + full                    (Conv1d k=1, no bias)
//     h    = proj_in(full)                                   (Linear 2304->2048, no bias)
//     temb = time_embd.1(silu(time_embd.0(fourier(t))))      -> [2048]
//     h    = cat(temb, h)  along the SEQUENCE axis           -> [2048, L+1]
//     36 x block:
//        h += attn(LN(h))          full bidirectional, 32 heads x 64, no bias,
//                                  partial NeoX RoPE on the first 32 of 64 dims
//        v,g = ff_in(LN(h)).chunk(2);  h += ff_out(v * silu(g))
//     h    = proj_out(h[1:])                                 (Linear 2048->128)
//     out  = postprocess_conv(h) + h                         (Conv1d k=1, no bias)
//
// ── Conventions that are easy to get wrong, and where each is pinned ──────────
//
// 1. FOURIER ORDER IS COS-FIRST. `torch.cat((angles.cos(), angles.sin()), -1)`
//    in MiniMaxMusic3FourierEmbedding. The layout doc does not state the order;
//    the reference source does. Getting it backwards is silent.
//
// 2. THE TIME TOKEN SHIFTS ROPE. The embedding is prepended as sequence token 0
//    and the rotary table is built for seq_len = L+1, so the latent frames sit at
//    positions 1..L, not 0..L-1. The token is stripped with `h[:, 1:]` after the
//    stack.
//
// 3. GLU: VALUE FIRST, GATE SECOND. `gate_states, gate = ff_in(x).chunk(2, -1)`
//    then `ff_out(gate_states * silu(gate))` — despite the misleading local name,
//    the SECOND half is the one that goes through silu. Layout doc §3.4 agrees
//    (`mm3.dit.glu_order = value_gate`).
//
// 4. NORMS ARE LAYERNORM WITH BETA, eps = mm3.dit.layer_norm_eps (1e-5). Not
//    RMSNorm, and the bias is not optional.
//
// 5. **THE OUTPUT IS NOT NEGATED HERE.** `mm3.dit.output_negated = true` in the
//    GGUF records ComfyUI's wrapper behaviour (`return -dit(...)`), which exists
//    to match ComfyUI's own sampler sign convention. The diffusers reference —
//    which produced every fixture we validate against, and whose Euler step is
//    `x + (sigma_next - sigma) * v` with sigma RISING 0->1 — returns the raw
//    transformer output. This port follows diffusers. If a future increment ever
//    routes MM3 through a ComfyUI-derived sampler, the negation belongs there,
//    not in this file. Verified in the reference `denoise.py`: the tapped
//    `flow.pred` is `transformer(...)[0]` with nothing applied to it.
//
// ── Design decisions, and why ─────────────────────────────────────────────────
//
// A. TENSOR LAYOUT: CHANNEL-FASTEST INSIDE THE GRAPH, TIME-FASTEST AT THE EDGE.
//    ggml_mul_mat wants the contracted dimension in ne0, so every activation in
//    the block stack is [C, S] (channel fastest). But a torch [1, 128, L] latent
//    is time-fastest in memory, which is ggml ne = [L, 128]. So the graph takes
//    latents as [L, 128] and transposes once on entry, and transposes once on
//    exit — 88 k elements each, free. The CONDITION needs no transpose at all:
//    torch [1, L, 2048] is already ne = [2048, L]. That asymmetry is real, not a
//    mistake: the reference itself does `encoder_hidden_states.transpose(1, 2)`.
//
// B. CFG RUNS AS TWO SEQUENTIAL PASSES OVER ONE CACHED GRAPH, NOT A BATCH OF 2
//    — and that is now MEASURED, not just reasoned. A full batched variant
//    exists (the graph generalises to [.., S, B] with the batch riding
//    ne[2]/ne[3]; RoPE broadcasts positions over ne[3], flash_attn_ext batches
//    over ne[3] natively) and was A/B'd on 2026-08-21 (RTX 5090, f16 DiT,
//    L=689, same seed): batched 81 ms/step vs two-pass 66 ms/step — 22 %
//    SLOWER, with output corr 0.999991 (numerically equivalent). The forward
//    is COMPUTE-bound at this L (~4.8 GB streamed in 33 ms ≈ 145 GB/s, far
//    under bandwidth), so batching's theoretical ceiling was only the ~3 ms of
//    weight re-streaming, and ggml's batched matmul/flash dispatch costs more
//    than that. `MM3_DIT_CFG_BATCH=1` re-enables the batched path if a future
//    ggml/backend change moves the trade-off; re-measure before believing it.
//    The single-branch B=1 graph remains what POST /mm3/dit-forward and
//    POST /mm3/flow-sample run, so the parity fixtures cover the production
//    path. In both variants the unconditional pass reuses the SAME uploaded
//    condition multiplied by a `cond_gate` input (1.0 = conditional, 0.0 =
//    unconditional; [1,1,B], so the batched graph gates per batch row). That
//    is exactly `torch.zeros_like(condition)` — the manifest pins
//    `flow_uncond_conditioning: "zeros_like(condition), not a re-encoded
//    empty prompt"` — and it avoids re-uploading 5.6 MB per pass, 60 times per
//    window.
//
// C. THE FOURIER EMBEDDING IS COMPUTED ON THE HOST. `t` is a scalar and the
//    Fourier basis is 128 fixed floats, so the 256-value embedding is a trivial
//    host loop uploaded as a graph input. This keeps `t` out of the graph
//    entirely (no per-step graph edits, no scalar-tensor plumbing) and computes
//    the cos/sin in double precision, which is strictly better than the
//    reference's bf16.
//
// D. GRAPH CACHED PER L, like the vocoder. Rebuilt only when the sequence length
//    changes or the model is reloaded. The 30-step loop therefore builds one
//    graph and computes it 60 times.
//
// E. EULER UPDATE AND CFG COMBINE HAPPEN ON THE HOST. Both are elementwise over
//    128*L floats (~350 kB); doing them in-graph would need a second graph and
//    device-resident latents for a saving of well under a millisecond per step
//    against ~2.4 B parameters of matmul. Host-side also makes the loop readable
//    and lets the debug endpoints return the exact same numbers the sampler uses.
//
// F. F32 ACTIVATIONS, F16 K/V FOR FLASH ATTENTION. Same trade the ACE DiT makes
//    (dit-graph.h): flash_attn_ext wants F16 K/V and accumulates in F32 under
//    GGML_PREC_F32. The reference ran the whole stack in bf16, which has FEWER
//    mantissa bits than F16, so this is strictly more precise than the thing we
//    are validating against. Non-GPU (or -DHOT_STEP_DISABLE_FA) builds take the
//    manual soft_max path, which can also be forced at runtime with
//    MM3_DIT_NO_FLASH=1 — the knob for deciding whether a precision gap is the
//    F16 K/V cast or something real. Measured: it is NOT the K/V cast (see the
//    parity block below).
//
// ── Parity, measured 2026-08-13 (L=689, RTX 5090, f16 GGUF) ───────────────────
//
// The fixtures were captured from a CPU **bf16** reference run, so the dumps are
// not ground truth — they carry their own ~1.7 % error. To separate "our port is
// wrong" from "the fixtures are bf16", the reference transformer was re-run on
// CPU in **float32** on the same inputs and all three compared:
//
//   step 0, conditional          corr        rel RMSE
//     ours   vs bf16 dump        0.9998656   1.64e-2
//     ours   vs fp32 reference   0.9999735   7.30e-3   <- we are CLOSER to truth
//     dump   vs fp32 reference   0.9998487   1.74e-2      than the capture is
//
//   30-step loop, final latents
//     ours   vs bf16 dump        0.9995528   3.01e-2
//     ours   vs fp32 reference   0.9999561   9.38e-3
//     dump   vs fp32 reference   0.9995930   2.89e-2
//
// So every conventions-level decision above is confirmed: this port sits inside
// the reference implementation's own numerical noise, on the correct side of it.
// The residual ~7e-3 against fp32 is precision, not structure — the error is
// FLAT across sequence position (per-frame rel RMSE 0.0049..0.0089, no trend),
// which is exactly what a RoPE offset or GLU-order bug could not look like.
// Neither the F16 K/V cast (MM3_DIT_NO_FLASH=1 changes it by 0.01 %) nor ggml's
// F16 cuBLAS accumulation (GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F improves it by
// only 8 %) explains it; the remainder is the F16 weight/activation path, and it
// is a third of the error the reference itself ships with. Not worth buying back.

#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"
#include "hot-step-build-flags.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ~1400 nodes measured at 36 blocks; the cap is deliberately loose.
#define MM3_DIT_MAX_NODES 4096
// Sequence-length ceiling. The reference window is 689 latents; anything much
// larger is a caller bug, and the compute buffer grows linearly with it.
#define MM3_DIT_MAX_FRAMES 4096

struct MM3DitGraph {
    // backend + residency
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    bool                 use_flash_attn = false;
    // Identity of the weights this prep was derived from. If the model is
    // unloaded and reloaded the buffer pointer changes and prep is rebuilt.
    const void *         weights_token = nullptr;
    // Host copy of dit.time_fourier.weight — [fourier_dim/2] F32.
    std::vector<float>   fourier_w;

    // cached graph (rebuilt when L changes)
    ggml_context * gctx    = nullptr;
    uint8_t *      gbuf    = nullptr;
    ggml_cgraph *  graph   = nullptr;
    ggml_tensor *  in_lat  = nullptr;  // [L, 128]   F32, channel-major (torch order)
    ggml_tensor *  in_cond = nullptr;  // [2048, L]  F32, torch [1, L, 2048] order
    ggml_tensor *  in_temb = nullptr;  // [256, 1]   F32, host-computed Fourier embedding
    ggml_tensor *  in_gate = nullptr;  // [1, 1]     F32, 1.0 = cond, 0.0 = uncond
    ggml_tensor *  in_pos  = nullptr;  // [L+1]      I32, RoPE positions
    ggml_tensor *  output  = nullptr;  // [L, 128]   F32, channel-major
    int64_t        graph_L = 0;
    // Batch size the cached graph was built for: 1 = the classic single-branch
    // forward, 2 = both CFG branches in one compute (batch 0 = conditional,
    // batch 1 = unconditional via the gate). Part of the cache key.
    int64_t        graph_B = 1;
    size_t         compute_bytes = 0;
    int            n_nodes       = 0;
    // Cleared on every graph rebuild; see design note B.
    bool           cond_uploaded = false;
};

// Timing/diagnostic surface for the debug endpoints.
struct MM3FlowStats {
    int    steps         = 0;
    int    forwards      = 0;
    double total_ms      = 0.0;
    double forward_ms    = 0.0;  // sum of the forward passes only
    double first_ms      = 0.0;  // includes graph build
    double last_ms       = 0.0;
    size_t compute_bytes = 0;
};

// ── Small helpers ───────────────────────────────────────────────────────────

// Read an F32 backend tensor back to host. The layout doc pins time_fourier to
// F32 (it is a random-Fourier frequency basis; rounding it perturbs the timestep
// embedding non-linearly across every step), so anything else is a converter
// change we must notice loudly rather than silently mis-read.
static bool mm3_dit_readback_f32(const ggml_tensor * t, std::vector<float> * out, std::string * err,
                                 const char * what) {
    if (!t) {
        if (err) {
            *err = std::string("DiT tensor missing: ") + what;
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("DiT tensor '") + what + "' is not F32 (type " + std::to_string((int) t->type) +
                   "); the layout contract pins it to F32";
        }
        return false;
    }
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get((ggml_tensor *) t, out->data(), 0, ggml_nbytes(t));
    return true;
}

// ── Prep / free ─────────────────────────────────────────────────────────────

static void mm3_dit_free_graph(MM3DitGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx          = nullptr;
    g->gbuf          = nullptr;
    g->graph         = nullptr;
    g->in_lat        = nullptr;
    g->in_cond       = nullptr;
    g->in_temb       = nullptr;
    g->in_gate       = nullptr;
    g->in_pos        = nullptr;
    g->output        = nullptr;
    g->graph_L       = 0;
    g->cond_uploaded = false;
}

static void mm3_dit_free(MM3DitGraph * g) {
    mm3_dit_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    g->fourier_w.clear();
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

// Acquire a backend + scheduler and pull the Fourier basis to host. Cheap after
// the first call: returns immediately when the model's synth buffer is the same
// one the current prep was derived from.
static bool mm3_dit_prepare(const MM3Model & m, MM3DitGraph * g, std::string * err) {
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
    mm3_dit_free(g);

    const MM3DitConfig & c = m.synth_cfg.dit;
    if (c.block_count == 0 || c.embedding_length == 0 || c.head_count == 0 || c.head_dim == 0) {
        if (err) {
            *err = "DiT config is empty — mm3.dit.* KVs missing from the synth GGUF";
        }
        return false;
    }
    if (c.rope_type != "neox") {
        if (err) {
            *err = "mm3.dit.rope_type is '" + c.rope_type + "', this port only implements 'neox'";
        }
        return false;
    }
    if (c.glu_order != "value_gate") {
        if (err) {
            *err = "mm3.dit.glu_order is '" + c.glu_order + "', this port only implements 'value_gate'";
        }
        return false;
    }

    BackendPair bp = backend_init("MM3-DiT");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;
    // MM3_DIT_NO_FLASH=1 forces the manual F32 soft_max path — a parity-debug
    // escape hatch, not a production knob.
    const char * no_fa = std::getenv("MM3_DIT_NO_FLASH");
    g->use_flash_attn  = bp.has_gpu && !HOT_STEP_FA_DISABLED && !(no_fa && no_fa[0] && no_fa[0] != '0');

    std::string e;
    if (!mm3_dit_readback_f32(m.synth.dit.time_fourier, &g->fourier_w, &e, "dit.time_fourier.weight")) {
        if (err) {
            *err = e;
        }
        mm3_dit_free(g);
        return false;
    }
    if ((int64_t) g->fourier_w.size() * 2 != (int64_t) c.fourier_dim) {
        if (err) {
            *err = "dit.time_fourier.weight has " + std::to_string(g->fourier_w.size()) +
                   " elements, expected fourier_dim/2 = " + std::to_string(c.fourier_dim / 2);
        }
        mm3_dit_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, MM3_DIT_MAX_NODES * 2);
    g->weights_token = token;

    fprintf(stderr, "[MM3-DiT] Prepared: %u blocks, %u heads x %u, rope %u/%.0f neox, flash_attn=%s\n",
            c.block_count, c.head_count, c.head_dim, c.rope_dim, (double) c.rope_theta,
            g->use_flash_attn ? "yes" : "no");
    return true;
}

// ── Graph pieces ────────────────────────────────────────────────────────────

// LayerNorm with learned gamma AND beta over ne0. x [E, S], w/b [E].
static ggml_tensor * mm3_dit_ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * n = ggml_norm(ctx, x, eps);
    n               = ggml_mul(ctx, n, w);
    return ggml_add(ctx, n, b);
}

// Manual F32 attention, for CPU / -DHOT_STEP_DISABLE_FA builds. Same helper the
// ACE DiT uses (dit-graph.h::dit_attn_f32). q/k/v [D, S, Nh, B] -> [D, Nh, S, B]
// — every op here broadcasts/batches over the trailing dims, so the CFG batch
// rides through untouched.
static ggml_tensor * mm3_dit_attn_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                      float scale) {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores               = ggml_soft_max_ext(ctx, scores, NULL, scale, 0.0f);
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// One transformer block. h [E, S, B] -> [E, S, B]. B is the CFG batch (1 or
// 2); with B == 1 every op below is shape-for-shape the original 2D/3D build.
static ggml_tensor * mm3_dit_block(ggml_context * ctx, const MM3DitGraph & g, const MM3DitConfig & c,
                                   const MM3DitBlock & w, ggml_tensor * h, ggml_tensor * positions) {
    const int64_t E  = (int64_t) c.embedding_length;
    const int64_t D  = (int64_t) c.head_dim;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t FI = (int64_t) c.ff_inner;
    const int64_t S  = h->ne[1];
    const int64_t B  = h->ne[2];

    // ── self-attention ──
    ggml_tensor * n   = mm3_dit_ln(ctx, h, w.attn_norm_w, w.attn_norm_b, c.layer_norm_eps);
    ggml_tensor * qkv = ggml_mul_mat(ctx, w.attn_qkv, n);  // [3E, S, B], fused q||k||v

    ggml_tensor * q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, E, S, B, qkv->nb[1], qkv->nb[2], 0));
    ggml_tensor * k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, E, S, B, qkv->nb[1], qkv->nb[2], (size_t) E * qkv->nb[0]));
    ggml_tensor * v =
        ggml_cont(ctx, ggml_view_3d(ctx, qkv, E, S, B, qkv->nb[1], qkv->nb[2], (size_t) (2 * E) * qkv->nb[0]));

    q = ggml_reshape_4d(ctx, q, D, Nh, S, B);
    k = ggml_reshape_4d(ctx, k, D, Nh, S, B);
    v = ggml_reshape_4d(ctx, v, D, Nh, S, B);

    // Partial RoPE: n_dims = rope_dim (32) of head_dim (64). ggml's NEOX mode
    // pairs element i with i + n_dims/2 and passes dims >= n_dims through
    // untouched — exactly the reference's chunk-2 rotate-half over hs[..., :32].
    // The position vector indexes ne[2] (= S) and broadcasts over the ne[3]
    // batch, so both CFG branches see identical rotations.
    q = ggml_rope_ext(ctx, q, positions, NULL, (int) c.rope_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, (int) c.rope_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D, S, Nh, B]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn;
    if (g.use_flash_attn) {
        // Full bidirectional attention — no mask, no causality, no GQA. The
        // ne[3] batch keeps the two CFG branches strictly separate.
        attn = ggml_flash_attn_ext(ctx, q, ggml_cast(ctx, k, GGML_TYPE_F16), ggml_cast(ctx, v, GGML_TYPE_F16), NULL,
                                   scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = mm3_dit_attn_f32(ctx, q, k, v, scale);
    }
    attn = ggml_reshape_3d(ctx, attn, Nh * D, S, B);        // [E, S, B]
    h    = ggml_add(ctx, h, ggml_mul_mat(ctx, w.attn_output, attn));

    // ── feed-forward: GLU with the VALUE half first ──
    ggml_tensor * n2 = mm3_dit_ln(ctx, h, w.ffn_norm_w, w.ffn_norm_b, c.layer_norm_eps);
    ggml_tensor * f  = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_in_w, n2), w.ffn_in_b);  // [2*FI, S, B]

    ggml_tensor * val  = ggml_cont(ctx, ggml_view_3d(ctx, f, FI, S, B, f->nb[1], f->nb[2], 0));
    ggml_tensor * gate = ggml_cont(ctx, ggml_view_3d(ctx, f, FI, S, B, f->nb[1], f->nb[2], (size_t) FI * f->nb[0]));
    ggml_tensor * y    = ggml_mul(ctx, val, ggml_silu(ctx, gate));

    y = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_out_w, y), w.ffn_out_b);
    return ggml_add(ctx, h, y);
}

// Full forward. Returns velocity [L, 128, B] (channel-major, torch memory
// order; batch outermost). B == 1 reproduces the original build node for node.
static ggml_tensor * mm3_dit_build(ggml_context * ctx, const MM3Model & m, const MM3DitGraph & g) {
    const MM3DitConfig &  c = m.synth_cfg.dit;
    const MM3DitWeights & w = m.synth.dit;

    const int64_t E  = (int64_t) c.embedding_length;
    const int64_t IC = (int64_t) c.in_channels;
    const int64_t CC = (int64_t) c.concat_channels;
    const int64_t L  = g.in_lat->ne[0];
    const int64_t B  = g.graph_B;

    // [L, 128] (time fastest, as uploaded) -> [128, L] (channel fastest, as the
    // matmuls want). See design note A. The latents are uploaded ONCE and
    // repeated across the CFG batch in-graph (both branches denoise the same x).
    ggml_tensor * x = ggml_cont(ctx, ggml_transpose(ctx, g.in_lat));
    if (B > 1) {
        x = ggml_repeat_4d(ctx, x, IC, L, B, 1);
    }

    // The permanent zero block. The open release has no cover-audio slot, so the
    // middle 128 channels are always zero (reference: `torch.zeros_like`).
    ggml_tensor * zeros = ggml_scale(ctx, x, 0.0f);

    // cond_gate is 1.0 for the conditional branch and 0.0 for the unconditional
    // one — CFG's uncond branch is zeros_like(condition), not an empty prompt.
    // In the batched graph the gate is [1, 1, B] ({1, 0}), so one mul produces
    // the conditional AND the zeroed condition in their respective batch rows.
    ggml_tensor * cond = g.in_cond;
    if (B > 1) {
        cond = ggml_repeat_4d(ctx, cond, cond->ne[0], L, B, 1);
    }
    cond = ggml_mul(ctx, cond, g.in_gate);

    ggml_tensor * full = ggml_concat(ctx, ggml_concat(ctx, x, zeros, 0), cond, 0);  // [2304, L, B]

    // preprocess_conv is Conv1d k=1 no bias == a plain matmul over channels, and
    // it is RESIDUAL.
    ggml_tensor * pre_w = ggml_reshape_2d(ctx, w.preprocess_conv, CC, CC);
    full                = ggml_add(ctx, ggml_mul_mat(ctx, pre_w, full), full);

    ggml_tensor * h = ggml_mul_mat(ctx, w.proj_in, full);  // [2048, L, B]

    // Timestep embedding: the Fourier features arrive precomputed from the host
    // (design note C); this is just the two-layer MLP with a SiLU between. Both
    // CFG branches share t, so the MLP runs once and the result is repeated.
    ggml_tensor * temb = ggml_add(ctx, ggml_mul_mat(ctx, w.time_embd_w[0], g.in_temb), w.time_embd_b[0]);
    temb               = ggml_silu(ctx, temb);
    temb               = ggml_add(ctx, ggml_mul_mat(ctx, w.time_embd_w[1], temb), w.time_embd_b[1]);  // [2048, 1]
    if (B > 1) {
        temb = ggml_repeat_4d(ctx, temb, temb->ne[0], 1, B, 1);
    }

    // Prepended as sequence token 0 — this is what shifts every latent frame's
    // RoPE position by one.
    h = ggml_concat(ctx, temb, h, 1);  // [2048, L+1, B]

    for (size_t i = 0; i < w.blk.size(); i++) {
        h = mm3_dit_block(ctx, g, c, w.blk[i], h, g.in_pos);
    }

    // Strip the time token, then project back to latent channels.
    h = ggml_cont(ctx, ggml_view_3d(ctx, h, E, L, B, h->nb[1], h->nb[2], h->nb[1]));  // [2048, L, B]
    ggml_tensor * out = ggml_mul_mat(ctx, w.proj_out, h);                              // [128, L, B]

    ggml_tensor * post_w = ggml_reshape_2d(ctx, w.postprocess_conv, IC, IC);
    out                  = ggml_add(ctx, ggml_mul_mat(ctx, post_w, out), out);

    // Back to torch memory order for the caller. ggml_transpose swaps ne0/ne1
    // only, so the batch stays outermost: [L, 128, B], one contiguous
    // 128*L-float velocity per branch.
    return ggml_cont(ctx, ggml_transpose(ctx, out));
}

// Build the graph for a given sequence length and CFG batch (1 or 2), or reuse
// the cached one. Switching B rebuilds — the two callers never interleave
// within a window, so in practice this fires once per residency, not per step.
static bool mm3_dit_ensure_graph(const MM3Model & m, MM3DitGraph * g, int64_t L, int64_t B, std::string * err) {
    if (g->graph && g->graph_L == L && g->graph_B == B) {
        return true;
    }
    mm3_dit_free_graph(g);
    g->graph_B = B;

    const MM3DitConfig & c = m.synth_cfg.dit;

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (MM3_DIT_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_DIT_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the DiT graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the DiT graph context";
        }
        return false;
    }

    g->in_lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, (int64_t) c.in_channels);
    ggml_set_name(g->in_lat, "mm3_dit_latents");
    ggml_set_input(g->in_lat);

    g->in_cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) c.condition_dim, L);
    ggml_set_name(g->in_cond, "mm3_dit_cond");
    ggml_set_input(g->in_cond);

    g->in_temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) c.fourier_dim, 1);
    ggml_set_name(g->in_temb, "mm3_dit_fourier");
    ggml_set_input(g->in_temb);

    // [1, 1, B]: one gate value per CFG batch row ({1} single, {1, 0} batched).
    g->in_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, B);
    ggml_set_name(g->in_gate, "mm3_dit_cond_gate");
    ggml_set_input(g->in_gate);

    // One position per token INCLUDING the prepended time token.
    g->in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L + 1);
    ggml_set_name(g->in_pos, "mm3_dit_positions");
    ggml_set_input(g->in_pos);

    g->output = mm3_dit_build(ctx, m, *g);
    ggml_set_name(g->output, "mm3_dit_velocity");
    ggml_set_output(g->output);

    g->graph = ggml_new_graph_custom(ctx, MM3_DIT_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->output);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "DiT graph allocation failed (out of VRAM?) for L=" + std::to_string(L);
        }
        return false;
    }

    g->gctx          = ctx;
    g->graph_L       = L;
    g->cond_uploaded = false;
    g->n_nodes       = ggml_graph_n_nodes(g->graph);
    g->compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);

    fprintf(stderr, "[MM3-DiT] Graph: L=%lld (S=%lld), B=%lld%s, %d nodes, %d splits, compute buffer %.0f MB\n",
            (long long) L, (long long) (L + 1), (long long) B, B > 1 ? " (batched CFG)" : "", g->n_nodes,
            ggml_backend_sched_get_n_splits(g->sched), (double) g->compute_bytes / (1024.0 * 1024.0));
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

static MM3DitGraph g_mm3_dit;

// One forward pass.
//
//   latents  128*L floats, channel-major (channel c at latents + c*L) — the
//            memory order of a torch [1, 128, L] tensor.
//   cond     2048*L floats in torch [1, L, 2048] order (feature fastest), or
//            nullptr to reuse the condition already resident on the device.
//   gate     1.0 = conditional pass, 0.0 = unconditional (zeros conditioning).
//   t        flow-matching time in [0, 1]; 0 = noise, 1 = data. Fed raw.
//   out      128*L floats, same layout as `latents`.
//
// Not thread-safe: the caller serialises (mm3-server.h holds g_mm3_mutex).
static bool mm3_dit_run(const MM3Model & m, MM3DitGraph * g, const float * latents, const float * cond, float gate,
                        float t, int64_t L, float * out, std::string * err) {
    if (!mm3_dit_ensure_graph(m, g, L, 1, err)) {
        return false;
    }

    // Fourier features on the host, in double, cos-first (design note C + §1).
    const size_t       H = g->fourier_w.size();
    std::vector<float> temb(2 * H);
    for (size_t i = 0; i < H; i++) {
        const double a = 2.0 * 3.14159265358979323846 * (double) t * (double) g->fourier_w[i];
        temb[i]        = (float) std::cos(a);
        temb[H + i]    = (float) std::sin(a);
    }
    ggml_backend_tensor_set(g->in_temb, temb.data(), 0, temb.size() * sizeof(float));

    // Positions 0..L: the time token is 0, the latent frames are 1..L. Re-sent
    // every run rather than once at build time — the scheduler re-allocates the
    // graph on every compute, and paying 2.7 kB to not depend on the allocator
    // handing back identical addresses is the right trade.
    std::vector<int32_t> pos((size_t) (L + 1));
    for (int64_t i = 0; i <= L; i++) {
        pos[(size_t) i] = (int32_t) i;
    }
    ggml_backend_tensor_set(g->in_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

    ggml_backend_tensor_set(g->in_gate, &gate, 0, sizeof(float));
    ggml_backend_tensor_set(g->in_lat, latents, 0, ggml_nbytes(g->in_lat));
    if (cond) {
        ggml_backend_tensor_set(g->in_cond, cond, 0, ggml_nbytes(g->in_cond));
        g->cond_uploaded = true;
    } else if (!g->cond_uploaded) {
        if (err) {
            *err = "no condition uploaded yet (internal: mm3_dit_run called with cond=nullptr first)";
        }
        return false;
    }

    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "DiT graph compute failed";
        }
        return false;
    }
    ggml_backend_tensor_get(g->output, out, 0, ggml_nbytes(g->output));
    return true;
}

// Both CFG branches in ONE graph compute (design note B). Same contract as
// mm3_dit_run, but the gate is fixed to {1, 0} across the batch and both
// velocities come back from the one forward: batch row 0 -> out_c (conditional),
// row 1 -> out_u (unconditional). The 2.4B weights stream once instead of twice.
static bool mm3_dit_run_cfg2(const MM3Model & m, MM3DitGraph * g, const float * latents, const float * cond, float t,
                             int64_t L, float * out_c, float * out_u, std::string * err) {
    if (!mm3_dit_ensure_graph(m, g, L, 2, err)) {
        return false;
    }

    const size_t       H = g->fourier_w.size();
    std::vector<float> temb(2 * H);
    for (size_t i = 0; i < H; i++) {
        const double a = 2.0 * 3.14159265358979323846 * (double) t * (double) g->fourier_w[i];
        temb[i]        = (float) std::cos(a);
        temb[H + i]    = (float) std::sin(a);
    }
    ggml_backend_tensor_set(g->in_temb, temb.data(), 0, temb.size() * sizeof(float));

    std::vector<int32_t> pos((size_t) (L + 1));
    for (int64_t i = 0; i <= L; i++) {
        pos[(size_t) i] = (int32_t) i;
    }
    ggml_backend_tensor_set(g->in_pos, pos.data(), 0, pos.size() * sizeof(int32_t));

    const float gate2[2] = { 1.0f, 0.0f };
    ggml_backend_tensor_set(g->in_gate, gate2, 0, sizeof(gate2));
    ggml_backend_tensor_set(g->in_lat, latents, 0, ggml_nbytes(g->in_lat));
    if (cond) {
        ggml_backend_tensor_set(g->in_cond, cond, 0, ggml_nbytes(g->in_cond));
        g->cond_uploaded = true;
    } else if (!g->cond_uploaded) {
        if (err) {
            *err = "no condition uploaded yet (internal: mm3_dit_run_cfg2 called with cond=nullptr first)";
        }
        return false;
    }

    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "DiT graph compute failed";
        }
        return false;
    }
    // Output is [L, 128, 2]: one contiguous 128*L velocity per branch.
    const size_t half = ggml_nbytes(g->output) / 2;
    ggml_backend_tensor_get(g->output, out_c, 0, half);
    ggml_backend_tensor_get(g->output, out_u, half, half);
    return true;
}

// MM3_DIT_CFG_BATCH=1 enables the batched-CFG graph. OFF by default: measured
// 22 % slower than the two-pass loop on the hardware that matters (see design
// note B) — this is a re-testing knob for future ggml versions, not a
// production switch. Read once; flipping it needs an engine restart.
static bool mm3_dit_cfg_batched(void) {
    static const bool batched = [] {
        const char * e = std::getenv("MM3_DIT_CFG_BATCH");
        return e && e[0] && e[0] != '0';
    }();
    return batched;
}

// Single conditional forward, the parity workhorse behind POST /mm3/dit-forward.
// `cond` may itself be all zeros to reproduce the unconditional branch.
static bool mm3_dit_forward(const MM3Model & m, const float * latents, const float * cond, int64_t L, float t,
                            std::vector<float> & out_velocity, std::string * err = nullptr) {
    if (L <= 0 || L > MM3_DIT_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_DIT_MAX_FRAMES);
        }
        return false;
    }
    if (!mm3_dit_prepare(m, &g_mm3_dit, err)) {
        return false;
    }
    out_velocity.assign((size_t) ((int64_t) m.synth_cfg.dit.in_channels * L), 0.0f);
    return mm3_dit_run(m, &g_mm3_dit, latents, cond, 1.0f, t, L, out_velocity.data(), err);
}

// FlowMatchEulerDiscreteScheduler with invert_sigmas=true, shift=1,
// num_train_timesteps=1, and the pipeline's own sigma schedule.
//
// The reference builds `np.linspace(1, 1/steps, steps)`, casts to float32,
// inverts with `1 - sigma` and appends a trailing 1.0. Reproduced here EXACTLY,
// float32 rounding included: `1 - float32(29/30)` is 0.03333336, not 1/30, and
// the fixture timesteps carry that noise. Deriving the schedule as `i/steps`
// would look right and be wrong in the 7th digit at every step.
static void mm3_flow_sigmas(int steps, std::vector<float> * sigmas, std::vector<float> * timesteps) {
    sigmas->assign((size_t) steps + 1, 0.0f);
    timesteps->assign((size_t) steps, 0.0f);
    const double start = 1.0;
    const double stop  = 1.0 / (double) steps;
    const double step  = steps > 1 ? (stop - start) / (double) (steps - 1) : 0.0;
    for (int i = 0; i < steps; i++) {
        // numpy linspace: y[i] = i*step + start, with the last element forced to
        // `stop` exactly.
        const double lin = (i == steps - 1) ? stop : (double) i * step + start;
        const float  s   = (float) lin;      // .astype(np.float32)
        const float  inv = 1.0f - s;         // torch float32 invert
        (*sigmas)[(size_t) i]    = inv;
        (*timesteps)[(size_t) i] = inv;
    }
    (*sigmas)[(size_t) steps] = 1.0f;
}

// The 30-step flow-matching Euler loop for ONE window.
//
//   noise    128*L floats, channel-major — the window's initial latents.
//   cond     2048*L floats, torch [1, L, 2048] order.
//   out      final latents, 128*L floats, channel-major.
//
// Per step: one conditional forward, one unconditional forward (same condition
// tensor, gate 0), CFG combine `u + scale*(c - u)`, then the Euler update
// `x += (sigma[i+1] - sigma[i]) * v`. No overlap blending — window 0 has
// overlap 0, and the chunk-overlap machinery belongs to the orchestration
// increment.
static bool mm3_flow_sample(const MM3Model & m, const float * noise, const float * cond, int64_t L, int steps,
                            float cfg_scale, std::vector<float> & out_latents, MM3FlowStats * stats = nullptr,
                            std::string * err = nullptr) {
    if (L <= 0 || L > MM3_DIT_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_DIT_MAX_FRAMES);
        }
        return false;
    }
    if (steps <= 0 || steps > 1000) {
        if (err) {
            *err = "steps must be in 1..1000";
        }
        return false;
    }
    if (!mm3_dit_prepare(m, &g_mm3_dit, err)) {
        return false;
    }

    const int64_t N = (int64_t) m.synth_cfg.dit.in_channels * L;
    out_latents.assign((size_t) N, 0.0f);
    memcpy(out_latents.data(), noise, (size_t) N * sizeof(float));

    std::vector<float> sigmas, timesteps;
    mm3_flow_sigmas(steps, &sigmas, &timesteps);

    std::vector<float> pred_c((size_t) N);
    std::vector<float> pred_u((size_t) N);

    const auto t_all = std::chrono::steady_clock::now();
    double     fwd_ms = 0.0, first_ms = 0.0, last_ms = 0.0;

    for (int i = 0; i < steps; i++) {
        const float t = timesteps[(size_t) i];
        const auto  t0 = std::chrono::steady_clock::now();

        // Conditional. The condition is uploaded on the first pass only; every
        // later pass reuses the resident copy and toggles the gate.
        if (!mm3_dit_run(m, &g_mm3_dit, out_latents.data(), i == 0 ? cond : nullptr, 1.0f, t, L, pred_c.data(), err)) {
            return false;
        }
        // Unconditional: same tensor, gate 0 == zeros_like(condition).
        if (!mm3_dit_run(m, &g_mm3_dit, out_latents.data(), nullptr, 0.0f, t, L, pred_u.data(), err)) {
            return false;
        }

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fwd_ms += ms;
        if (i == 0) {
            first_ms = ms;
        }
        last_ms = ms;

        const float dsigma = sigmas[(size_t) i + 1] - sigmas[(size_t) i];
        for (int64_t j = 0; j < N; j++) {
            const float u = pred_u[(size_t) j];
            const float v = u + cfg_scale * (pred_c[(size_t) j] - u);
            out_latents[(size_t) j] += dsigma * v;
        }
    }

    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all).count();
    if (stats) {
        stats->steps         = steps;
        stats->forwards      = steps * 2;
        stats->total_ms      = total_ms;
        stats->forward_ms    = fwd_ms;
        stats->first_ms      = first_ms;
        stats->last_ms       = last_ms;
        stats->compute_bytes = g_mm3_dit.compute_bytes;
    }
    fprintf(stderr,
            "[MM3-DiT] Flow sample: L=%lld, %d steps, cfg %.2f, t %.6f..%.6f -> %.0f ms "
            "(%.0f ms/step, %.0f ms/forward)\n",
            (long long) L, steps, (double) cfg_scale, (double) timesteps.front(), (double) timesteps.back(), total_ms,
            total_ms / (double) steps, fwd_ms / (double) (steps * 2));
    return true;
}
