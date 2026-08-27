#pragma once
// minimax/mm3-lm-graph.h — MiniMax-Music3 global LM (Qwen3-8B) as an AR engine.
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-ar-loop.h
// and minimax/mm3-server.h, the latter being the single hook include into
// tools/hot-step-server.cpp.
//
// SCOPE (increment 5a): the Qwen3-8B forward graph in the exact shape the MM3 AR
// stage needs — a PERSISTENT 2-ROW CFG BATCH over one KV cache, a prefill that
// takes token ids, a decode step that takes a SOFT INPUT EMBEDDING it builds
// itself from the previous frame's eight RVQ codes, and both hidden-state and
// head-logit taps on every step. The loop that drives it is mm3-ar-loop.h.
//
// ── Architecture ─────────────────────────────────────────────────────────────
//
// Stock Qwen3-8B (layout doc §3.1): 36 blocks, H=4096, 32 query heads / 8 KV
// heads x 128, SwiGLU 12288, RMSNorm eps 1e-6, per-head q/k RMSNorm BEFORE RoPE,
// NeoX RoPE over all 128 head dims at theta 1e6, untied 200000-row head. The only
// non-stock thing about it is the vocabulary: ids 151675..168058 are semantic
// audio codes and 151670 is the AR stop token.
//
// ── Design decisions, and why ────────────────────────────────────────────────
//
// A. THE 2-ROW CFG BATCH IS THE WHOLE POINT, AND IT IS FREE. AR CFG is normally
//    1.5 (`mm3.ar.cfg_scale`), so every step must evaluate a conditional and an
//    unconditional row. At one token per row a decode step is pure BANDWIDTH:
//    17.2 GB of f16 weights get streamed whatever the batch is. Two rows through
//    one graph therefore costs almost exactly what one row costs, and two
//    sequential passes would cost double. This is the opposite trade from the
//    flow DiT (mm3-dit-graph.h note B), where each pass is already 690 tokens
//    wide and batching buys nothing — same model, different regime.
//
//    A1. ...UNLESS THE CHECKPOINT BAKED THE GUIDANCE IN. A guidance-distilled
//    composer (the depth-pruned 5.7B students distil against the teacher's
//    CFG-GUIDED distributions) declares `mm3.ar.cfg_scale = 1.0`, at which the
//    blend `u + (c - u) * 1.0` is identically `c` — the unconditional row is
//    computed, read back, and then algebraically cancelled. So `cfg_rows` is a
//    RUNTIME 1-or-2 derived from the scale, not a compile-time 2:
//
//        |cfg - 1| <= 1e-6  ->  cfg_rows = 1
//
//    Because it is keyed on the arithmetic and not on a model name, it is safe
//    for any checkpoint: at any other scale the pair still runs, bit-identical
//    to before. What this buys is mostly NOT decode latency — note A's
//    bandwidth argument cuts both ways, and dropping a row from a
//    memory-bound step saves little — but:
//      * the KV cache HALVES (288 -> 144 kB/position; 1.1 GB on a 5-minute
//        render), which is the binding constraint on a 22.5 GB f16 stack, and
//      * prefill, which IS compute-bound at T = prompt length, halves.
//    The student's real speed win is its 21 blocks instead of 36; this change
//    is what stops us paying for a row whose only effect is to be subtracted
//    from itself.
//
//    Batch lives in ne3 throughout: activations are [H, T, B], attention is
//    [D, T, Nh, B] against a [D, n_kv, Nkv, B] cache. GQA falls out of ggml's
//    mul_mat/flash_attn broadcast (query head h reads KV head h/4) and needs no
//    repeat.
//
// B. ONE KV CACHE ALLOCATION, ne3 = 2. Per layer a [D, n_ctx, Nkv, 2] F16 tensor,
//    all layers in one backend buffer, zeroed once (a padded attention window
//    must read finite values, never uninitialised F16 bit patterns that decode to
//    NaN). Writes go through `ggml_set_rows` with the destination row indices
//    supplied AS DATA, so the graph topology is identical at every step and only
//    an upload changes between frames.
//
//    SIZE IT FROM THE REQUEST. n_ctx = prompt + max_frames + slack, rounded to the
//    bucket. The cache costs
//        2 rows x 36 layers x 2 (K+V) x 8 heads x 128 dims x 2 bytes = 288 kB
//    per position (halved at cfg_rows = 1, note A1) — 106 MB for a 12-second render, 2.2 GB for the 5-minute
//    ceiling, 2.9 GB at the model's declared 10240-token context. Allocating the
//    ceiling unconditionally would cost more VRAM than the depth decoder. The
//    cache therefore GROWS on demand and never shrinks, so a repeat request at
//    the same length is free.
//
// C. THE ATTENTION WINDOW IS BUCKETED TO 256, NOT EXACT. A graph is cached per
//    (T, n_kv_pad); an exact window would rebuild all 36 blocks every frame. At a
//    256 bucket a 300-frame render rebuilds twice and a 9000-frame render 36
//    times, and the wasted attention is at most 255 masked positions — a rounding
//    error against 8.6 B parameters of matmul. Same constant the ACE LM uses
//    (qwen3-lm.h), for the same reason.
//
// D. PREFILL AND DECODE ARE THE SAME BUILDER, TWO CACHED SLOTS, TWO SCHEDULERS.
//    They differ only in T and in how the input embedding is produced. Separate
//    schedulers because `ggml_backend_sched_alloc_graph` owns the allocation: a
//    second graph on a shared scheduler would invalidate the first. Same reason
//    the depth decoder keeps seven (mm3-depth-graph.h note A).
//
// E. THE FEEDBACK EMBEDDING IS BUILT INSIDE THE DECODE GRAPH. The reference's
//    `_embed_audio_frame` is
//        (token_embd[semantic + 151675] + SUM_c audio_embd[code_c + (c-1)*1024])
//        * num_codebooks^-0.5
//    Both tables are already resident (token_embd in the LM file, audio_embd in
//    the synth file — this graph reads BOTH, like the depth decoder does), so
//    doing it in-graph replaces a 16 kB readback + 16 kB upload per frame with
//    eight row gathers. It is still exposed as a graph OUTPUT, because
//    `lm_i*_feedback_embed` is a parity target and an unobservable intermediate
//    cannot be validated.
//
//    Note the asymmetry with the depth decoder: THERE every token is pushed
//    through `depth.proj` first; HERE nothing is projected. Same two tables, two
//    different consumers.
//
// F. THE HEAD IS APPLIED TO THE LAST POSITION ONLY, AND OVER THE FULL 200000.
//    Restricting it to the semantic slice + EOS would be mathematically identical
//    (everything else is masked out downstream) and 12x cheaper — but it is 1.6 GB
//    of the 17.2 GB streamed per step, under 10 %, and computing the whole thing
//    keeps the masking honest and the logits directly comparable to the reference
//    dump. Revisit if the AR stage ever becomes the bottleneck it is not today.
//
// G. F32 ACTIVATIONS, F16 K/V. Same trade as the DiT and the ACE LM. The
//    reference ran bf16 end to end, which has FEWER mantissa bits than F16, so
//    this is strictly more precise than the thing it is validated against.
//    MM3_LM_NO_FLASH=1 forces the manual soft_max path for parity debugging.
//
// ── Parity + cost, measured 2026-08-13 (RTX 5090, f16 GGUF, 70-token prompt) ──
//
// Forced replay against mm3-weights/fixtures/ (corr / rel RMSE vs the bf16 dumps):
//
//                          prefill      iter 0      iter 1      iter 2..4
//   last_hidden cond      .99953/3.5e-2  (same)    .99960/2.9e-2  .99943..99964 / 2.7..3.4e-2
//   last_hidden uncond    .99995/1.1e-2  (same)    .99995/1.1e-2  .99989..99995 / 1.0..1.6e-2
//   semantic_logits cond        —      .99982/2.4e-2 .99980/2.1e-2 .99941..99971 / 2.3..3.5e-2
//   semantic_logits uncond      —      .99998/9.1e-3 .99994/1.2e-2 .99989..99993 / 1.2..1.4e-2
//   feedback_embed              —      .999996/2.8e-3 ................ 2.77e-3 every iteration
//
// THE FEEDBACK EMBEDDING IS THE CALIBRATION. It is one row gather plus six adds
// plus a scale — there is no depth in which a port bug could hide — and it lands
// at 2.78e-3, invariant across iterations. That is the CAPTURE's bf16 floor, not
// ours, and it is confirmed independently: every value in lm_prefill_last_hidden,
// lm_i*_feedback_embed, lm_i*_depth_hidden and lm_i*_semantic_logits is EXACTLY
// bf16-representable (zero values with a non-zero low mantissa half). Read the
// 1..3.5e-2 on the hidden states as that 2.8e-3 accumulated through 36 residual
// blocks; it matches the flow DiT's independently measured 1.74e-2 dump floor at
// the same depth.
//
// The `guided_logits` -inf mask differs from the dump by 1..7 positions per
// iteration, ALWAYS in the same direction: the reference keeps 50..55 candidates
// where we keep exactly 50. That is not a filter-rule difference. The top-k
// threshold is compared with STRICT less-than, so exact ties survive — and the
// reference's 50th-largest conditional logit is a coarse bf16 grid point
// (6.40625, 9.0, 7.4375, 8.4375, 10.5625) with 2..6 exact duplicates, while our
// F32 logits essentially never tie. The bf16 capture manufactures the ties.
//
// Speed: 15.3 ms per decode step (2-row batch, steady state over 1200 frames,
// 5 attention-window rebuilds, no drift). Prefill 41 ms for 70 tokens x 2 rows
// after the graph is cached, 140 ms including the build. That is 17.2 GB of f16
// weights streamed in 15.3 ms = ~1.12 TB/s effective — the step is bandwidth
// bound, as design note A assumed, which is exactly why the second CFG row is
// nearly free.

#include "mm3-align.h"
#include "mm3-imatrix.h"
#include "mm3-lm-adapter.h"
#include "../qwen3-lora.h"
#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"
#include "hot-step-build-flags.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 36 blocks x ~45 nodes + embedding/head plumbing measures ~1700; loose on purpose.
//
// A LoKr adapter is the reason this is not 4096 any more. Its apply costs ~11
// nodes per module against a LoRA's 3, and there are 7 modules x 36 layers, so
// it adds ~2,000 nodes and lands around 3,700 — inside 4096, but with under 10%
// headroom, which is not a margin worth defending. The cost of a bigger cap is
// one pointer per slot in the node array.
#define MM3_LM_MAX_NODES 8192
// Attention-window quantum (design note C).
#define MM3_LM_KV_BUCKET 256
// MM3_MAX_BATCH_ROWS (mm3-model.h) is the MAXIMUM — the size every host-side
// staging buffer is cut to, so a caller's [8, H] hidden / [8, V] logit buffers
// are always big enough. The number of rows actually computed and read back is
// MM3LmGraph::rows() = cfg_rows * n_takes (1..8); use that for graph shapes and
// transfer sizes, never the ceiling. cfg_rows alone is the CFG pair size and is
// NOT the batch once ensemble takes are in play.

// One cached graph. Prefill and decode differ only in T and the input mode.
struct MM3LmSlot {
    ggml_backend_sched_t sched = nullptr;
    ggml_context *       gctx  = nullptr;
    uint8_t *            gbuf  = nullptr;
    ggml_cgraph *        graph = nullptr;

    // inputs
    ggml_tensor * in_ids  = nullptr;  // prefill: I32 [T*B] (row-major: cond row then uncond row)
                                      // decode:  I32 [num_codebooks] feedback row indices
                                      // replay:  I32 [T] token_embd row per position
    ggml_tensor * in_pos  = nullptr;  // I32 [T]
    ggml_tensor * in_rows = nullptr;  // I64 [T] KV destination rows
    ggml_tensor * in_mask = nullptr;  // F16 [n_kv_pad, T]
    // Replay-slot extras (LRC replay, see mm3_lm_lrc_replay): the per-position
    // acoustic embedding rows, the 0/1 gate that zeroes them on prompt
    // positions, and the per-position embedding scale (1.0 on prompt positions,
    // ar_embedding_scale on frame positions).
    ggml_tensor * in_ac    = nullptr;  // I32 [T * (num_codebooks-1)]
    ggml_tensor * in_gate  = nullptr;  // F32 [1, T]
    ggml_tensor * in_scale = nullptr;  // F32 [1, T]

    // outputs
    ggml_tensor * out_hidden   = nullptr;  // F32 [H, 1, B] last position, pre-head
    ggml_tensor * out_logits   = nullptr;  // F32 [logits_n, 1, B]
    ggml_tensor * out_feedback = nullptr;  // F32 [H, 1]     decode only
    // The head rows this slot computes: [logits_lo, logits_lo + logits_n) of
    // the vocabulary. Full head = (0, V); sliced (MM3LmGraph::head_slice) =
    // the contiguous EOS+semantic span the AR sampler actually reads.
    int64_t       logits_lo = 0;
    int64_t       logits_n  = 0;

    // Alignment probe (MM3_ALIGN_DUMP=1): post-softmax attention weights per
    // layer, [n_kv, T, Nh, B]. Only populated on the manual F32 path — flash
    // attention never materialises them. Empty in normal runs.
    std::vector<ggml_tensor *> attn_scores;

    int64_t T             = 0;
    int64_t n_kv_pad      = 0;
    size_t  compute_bytes = 0;
    int     n_nodes       = 0;
};

struct MM3LmGraph {
    BackendPair    bp             = {};
    ggml_backend_t backend        = nullptr;
    ggml_backend_t cpu_backend    = nullptr;
    bool           backend_ref    = false;
    bool           use_flash_attn = false;
    // Runtime LM LoRA (mm3-lm-adapter.h). Scales are baked into cached graphs
    // as constants, so changing (adapter, scales) must invalidate the slots —
    // callers go through mm3_lm_set_adapter, never assign these directly.
    const MM3LmAdapter * adapter        = nullptr;
    MM3LmAdapterScales   adapter_scales = {};
    // MM3_ALIGN_DUMP=1 — capture EVERY layer's attention for lyric alignment
    // discovery. Forces the manual F32 attention path on all 36 layers (flash
    // fuses the softmax and never produces a score tensor to read).
    bool           dump_attn      = false;
    // Production alignment: capture only the layers in MM3_ALIGN_HEADS. The
    // other 33 keep flash attention, so this costs a fraction of dump_attn.
    bool           align_capture  = false;
    // Compute the output head over ONLY the contiguous EOS+semantic span
    // instead of the full 200k vocabulary (design note F's "revisit"). The AR
    // sampler never reads anything else — every other row is masked to -inf
    // downstream — so the arithmetic is identical while ~0.8 GB/step of head
    // weights stop streaming and the logit readback shrinks 12×. Opt-in like
    // align_capture (the graphs bake the choice): mm3_ar_plan sets it; the
    // mm3-lm-probe parity tool keeps the full head it compares against.
    bool           head_slice     = false;

    // Identity of the weights this prep was derived from. BOTH buffers matter:
    // the feedback embedding reads token_embd from the LM file and audio_embd
    // from the synth file, so either being reloaded invalidates the graphs.
    const void * lm_token    = nullptr;
    const void * synth_token = nullptr;

    // KV cache: one buffer, one [D, n_ctx, Nkv, B] F16 tensor per layer.
    ggml_context *             kv_ctx = nullptr;
    ggml_backend_buffer_t      kv_buf = nullptr;
    std::vector<ggml_tensor *> kv_k;
    std::vector<ggml_tensor *> kv_v;
    int64_t                    n_ctx    = 0;
    size_t                     kv_bytes = 0;
    int64_t                    kv_pos   = 0;  // shared: all CFG rows advance together
    // 1 or 2 (note A1), derived from mm3.ar.cfg_scale in mm3_lm_prepare. This is
    // the CFG PAIR size, not the batch: the batch is rows() = cfg_rows * takes.
    // A change invalidates the KV cache and all slots.
    int                        cfg_rows = MM3_LM_CFG_ROWS;
    // Ensemble takes K (mm3-model.h): how many independent songs decode in
    // lockstep. 1 = today's single-song path, and every shape below then reduces
    // to exactly what it was. Also invalidates the KV cache and all slots.
    int                        n_takes  = 1;
    // Fold the decode batch onto ne1 so ggml's CUDA backend picks the kernel
    // that shares the weight read across rows (see mm3_lm_mm). Baked into the
    // cached slots, so mm3_lm_set_takes owns it and a change tears them down.
    bool                       fold_rows = false;

    // The real batch. EVERY graph shape and every host<->device transfer size
    // keys off this, never off cfg_rows and never off a compile-time ceiling.
    int rows() const { return cfg_rows * n_takes; }

    MM3LmSlot prefill;
    MM3LmSlot decode;
    // Post-hoc LRC replay (teacher-forced chunk over blocks 0..max align layer,
    // manual attention, single CFG row). Built lazily by mm3_lm_lrc_replay.
    MM3LmSlot replay;

    // host staging, reused every step
    std::vector<int32_t>  ids_host;
    std::vector<int32_t>  pos_host;
    std::vector<int64_t>  rows_host;
    std::vector<uint16_t> mask_host;
};

// ── Free ────────────────────────────────────────────────────────────────────

static void mm3_lm_free_slot(MM3LmSlot * s) {
    if (s->gctx) {
        if (s->sched) {
            ggml_backend_sched_reset(s->sched);
        }
        ggml_free(s->gctx);
        free(s->gbuf);
    }
    ggml_backend_sched_t keep = s->sched;
    *s                        = MM3LmSlot{};
    s->sched                  = keep;
}

static void mm3_lm_free_slot_all(MM3LmSlot * s) {
    mm3_lm_free_slot(s);
    if (s->sched) {
        ggml_backend_sched_free(s->sched);
        s->sched = nullptr;
    }
}

static void mm3_lm_free_kv(MM3LmGraph * g) {
    if (g->kv_buf) {
        ggml_backend_buffer_free(g->kv_buf);
    }
    if (g->kv_ctx) {
        ggml_free(g->kv_ctx);
    }
    g->kv_buf = nullptr;
    g->kv_ctx = nullptr;
    g->kv_k.clear();
    g->kv_v.clear();
    g->n_ctx    = 0;
    g->kv_bytes = 0;
    g->kv_pos   = 0;
}

static void mm3_lm_free(MM3LmGraph * g) {
    mm3_lm_free_slot_all(&g->prefill);
    mm3_lm_free_slot_all(&g->decode);
    mm3_lm_free_slot_all(&g->replay);
    mm3_lm_free_kv(g);
    g->lm_token    = nullptr;
    g->synth_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

// Set (or clear) the runtime LM adapter + scales for subsequent graphs. The
// scales are baked into the cached prefill/decode graphs as constants, so any
// change frees the slots — they rebuild lazily on the next step (graph
// construction only; weights and the KV cache are untouched). Only call
// BETWEEN generations: an adapter swap mid-sequence would splice two different
// models' hidden-state streams through one KV cache.
static void mm3_lm_set_adapter(MM3LmGraph * g, const MM3LmAdapter * ad, const MM3LmAdapterScales & sc) {
    if (g->adapter == ad && g->adapter_scales == sc) {
        return;
    }
    g->adapter        = ad;
    g->adapter_scales = sc;
    mm3_lm_free_slot_all(&g->prefill);
    mm3_lm_free_slot_all(&g->decode);
    mm3_lm_free_slot_all(&g->replay);
}

// ── Graph pieces ────────────────────────────────────────────────────────────

static ggml_tensor * mm3_lm_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// The contiguous head-row span covering EOS and the semantic codes — the only
// vocabulary rows the AR sampler ever reads. For the shipped checkpoint that
// is rows 151670..168059: EOS 151670, then 4 dead rows, then the 16384
// semantic codes at 151675. Returns false (caller keeps the full head) if the
// config doesn't describe such a span or EOS sits far from the codes.
static bool mm3_lm_head_slice_span(const MM3LmConfig & c, int64_t * lo, int64_t * n) {
    const int64_t EOS = (int64_t) c.eos_audio;
    const int64_t OFF = (int64_t) c.semantic_vocab_offset;
    const int64_t SV  = (int64_t) c.semantic_vocab_size;
    const int64_t V   = (int64_t) c.vocab_size;
    if (SV <= 0 || OFF < 0 || EOS < 0 || OFF + SV > V || EOS >= V) {
        return false;
    }
    const int64_t a = EOS < OFF ? EOS : OFF;
    const int64_t b = (EOS + 1) > (OFF + SV) ? (EOS + 1) : (OFF + SV);
    if (b - a > SV + 4096) {
        return false;  // EOS far outside the code block: slicing buys nothing
    }
    *lo = a;
    *n  = b - a;
    return true;
}

// Manual F32 attention, for CPU / -DHOT_STEP_DISABLE_FA / MM3_LM_NO_FLASH builds.
// q [D, T, Nh, B], k/v [D, n_kv, Nkv, B] (strided cache views) -> [D, Nh, T, B].
static ggml_tensor * mm3_lm_attn_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                     ggml_tensor * mask, float scale, ggml_tensor ** out_scores = nullptr) {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);                            // [n_kv, T, Nh, B]
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    if (out_scores) {
        *out_scores = scores;   // the lyric->frame alignment signal, pre-V
    }
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));             // [n_kv, D, Nkv, B]
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);                      // [D, T, Nh, B]
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));                 // [D, Nh, T, B]
}

// Base matmul + optional runtime LoRA delta: y = W x + s * B (A x). A/B are
// f16, x is f32 — ggml's mixed mul_mat handles both. The scale is baked as a
// graph constant; MM3LmGraph invalidates cached slots when it changes.
// One weight matmul, with the batch rows folded onto ne1 for the decode step.
//
// WHY THIS RESHAPE EARNS ITS KEEP. ggml's CUDA backend picks the kernel that
// amortises a quantised weight read across columns by looking at `src1->ne[1]`
// (ggml-cuda.cu, `use_mul_mat_vec_q`, capped at MMVQ_MAX_BATCH_SIZE = 8).
// A decode activation is [H, T=1, B]: ne1 is 1 and the batch rides ne2, so ggml
// issues B INDEPENDENT matrix-vector products and streams the whole 9 GB of
// weights once per row — the exact cost the ensemble exists to share.
//
// At T == 1, [H, 1, B] and [H, B, 1] are the SAME BYTES (ne0 fastest, then a
// unit dimension), so folding is a free relabel that puts the rows where the
// kernel looks for them. Measured on an RTX 5090, q8_0, 300 frames — LM decode
// step, which is the whole story:
//
//     takes         1      2      3      4
//     ne2 (before)  8.13  11.56  15.18  18.65 ms   +43 % per extra take
//     ne1 (after)   6.99   7.14   7.81   8.24 ms    +6 % per extra take
//
// IT MAY NOT BE BIT-IDENTICAL, AND THAT IS FINE. A different kernel sums the
// same products in a different order, so a logit can move by an ulp, which the
// top-k multinomial can amplify into a different — equally valid — song from
// the same seed. It measured identical over every draw tried, but the argument
// for shipping it does not rest on that: a render nobody heard has no claim on
// being reproduced, and any take a listener DID hear is already a wav on disk.
//
// What actually has to hold is CORRECTNESS, not reproducibility: the folded
// kernel must compute the right forward, not merely a different one. That is
// what check-mm3-ensemble.mjs pins, by replaying the fixture's own codes
// through the folded decode path and comparing the hidden states and logits
// against the reference bf16 dumps.
//
// On by default at every take count. MM3_LM_FOLD_ROWS=0 restores the old
// kernels, which is how the two paths are A/B'd.
//
// The prefill path is untouched on both settings: there ne1 is already T (the
// whole prompt), so it is a real GEMM and belongs on the MMQ/cuBLAS path it
// already takes.
static ggml_tensor * mm3_lm_mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, const MM3LmAdapter * ad,
                               const MM3LmAdapterScales & sc, int layer, int module, bool want_fold) {
    const bool fold = want_fold && x->ne[1] == 1 && x->ne[2] > 1 && x->ne[3] == 1 && ggml_is_contiguous(x);
    if (fold) {
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[2]);  // [H, 1, B] -> [H, B]
    }
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (ad) {
        const MM3LmAdapterPair & p = ad->mods[layer][module];
        if (p.has_lora()) {
            const float s = ad->effective(layer, module, sc);
            if (s != 0.0f) {
                ggml_tensor * d = ggml_mul_mat(ctx, p.b, ggml_mul_mat(ctx, p.a, x));
                y               = ggml_add(ctx, y, ggml_scale(ctx, d, s));
            }
        } else if (p.has_lokr()) {
            const float s = ad->effective(layer, module, sc);
            if (s != 0.0f) {
                // Hand the shared apply a pair whose scale already carries this
                // request's dial. qwen3_lokr_delta scales the kron OUTPUT, which
                // is the same point the LoRA branch above scales, so the
                // attn/mlp/early-mid-late sliders mean the same thing for both.
                QwLoraPair kp;
                kp.w1         = p.w1;
                kp.w2         = p.w2;
                kp.w2_a       = p.w2_a;
                kp.w2_b       = p.w2_b;
                kp.in_m       = p.in_m;
                kp.in_n       = p.in_n;
                kp.out_l      = p.out_l;
                kp.out_k      = p.out_k;
                kp.lokr_scale = p.lokr_scale * s;
                y             = qwen3_lokr_delta(ctx, &kp, x, y);
            }
        }
    }
    if (fold) {
        y = ggml_reshape_3d(ctx, y, y->ne[0], 1, y->ne[1]);  // [N, B] -> [N, 1, B]
    }
    return y;
}

// One transformer block. h [H, T, B] -> [H, T, B]. The KV writes are expanded
// into `gf` eagerly so they are ordered before this layer's cache read (and,
// transitively, before every later layer's) — ggml executes nodes in list order,
// and a cache view carries no data dependency on the write that filled it.
static ggml_tensor * mm3_lm_block(ggml_context * ctx, ggml_cgraph * gf, const MM3LmConfig & c, const MM3LmLayer & w,
                                  ggml_tensor * h, ggml_tensor * positions, ggml_tensor * mask, ggml_tensor * rows,
                                  ggml_tensor * kcache, ggml_tensor * vcache, int64_t n_kv_pad, bool use_flash,
                                  ggml_tensor ** out_scores = nullptr, const MM3LmAdapter * ad = nullptr,
                                  const MM3LmAdapterScales * ad_sc = nullptr, int layer_idx = 0,
                                  bool fold_rows = false) {
    const int64_t H   = (int64_t) c.embedding_length;
    const int64_t D   = (int64_t) c.key_length;
    const int64_t Nh  = (int64_t) c.head_count;
    const int64_t Nkv = (int64_t) c.head_count_kv;
    const int64_t T   = h->ne[1];
    const int64_t B   = h->ne[2];

    const MM3LmAdapterScales sc_local = ad_sc ? *ad_sc : MM3LmAdapterScales{};

    ggml_tensor * n = mm3_lm_rms(ctx, h, w.attn_norm, c.rms_eps);

    ggml_tensor * q = mm3_lm_mm(ctx, w.attn_q, n, ad, sc_local, layer_idx, MM3_LM_ADAPTER_Q, fold_rows);  // [Nh*D, T, B]
    ggml_tensor * k = mm3_lm_mm(ctx, w.attn_k, n, ad, sc_local, layer_idx, MM3_LM_ADAPTER_K, fold_rows);  // [Nkv*D, T, B]
    ggml_tensor * v = mm3_lm_mm(ctx, w.attn_v, n, ad, sc_local, layer_idx, MM3_LM_ADAPTER_V, fold_rows);

    q = ggml_reshape_4d(ctx, q, D, Nh, T, B);
    k = ggml_reshape_4d(ctx, k, D, Nkv, T, B);
    v = ggml_reshape_4d(ctx, v, D, Nkv, T, B);

    // Qwen3's per-head q/k RMSNorm, over the head dim, BEFORE RoPE.
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_eps), w.attn_q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_eps), w.attn_k_norm);

    // NeoX RoPE over all head dims. Positions map to ne2 (= T) and are shared by
    // both CFG rows in ne3 — they are the same sequence position by construction.
    q = ggml_rope_ext(ctx, q, positions, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                      0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                      0.0f, 0.0f);

    // Cache layout is [D, position, head, row]; the projections come out
    // [D, head, position, row]. One permute each, then set_rows.
    ggml_tensor * k_w = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [D, T, Nkv, B]
    ggml_tensor * v_w = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kcache, k_w, rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, vcache, v_w, rows));

    ggml_tensor * k_win = ggml_view_4d(ctx, kcache, D, n_kv_pad, Nkv, B, kcache->nb[1], kcache->nb[2], kcache->nb[3], 0);
    ggml_tensor * v_win = ggml_view_4d(ctx, vcache, D, n_kv_pad, Nkv, B, vcache->nb[1], vcache->nb[2], vcache->nb[3], 0);

    ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D, T, Nh, B]

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn;
    if (use_flash) {
        attn = ggml_flash_attn_ext(ctx, q4, k_win, v_win, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = mm3_lm_attn_f32(ctx, q4, k_win, v_win, mask, scale, out_scores);
    }
    attn = ggml_reshape_3d(ctx, attn, H, T, B);  // [D, Nh, T, B] -> [H, T, B]

    h = ggml_add(ctx, h, mm3_lm_mm(ctx, w.attn_output, attn, ad, sc_local, layer_idx, MM3_LM_ADAPTER_O, fold_rows));

    // SwiGLU: down(silu(gate) * up).
    ggml_tensor * n2   = mm3_lm_rms(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gate = ggml_silu(ctx, mm3_lm_mm(ctx, w.ffn_gate, n2, ad, sc_local, layer_idx, MM3_LM_ADAPTER_GATE, fold_rows));
    ggml_tensor * up   = mm3_lm_mm(ctx, w.ffn_up, n2, ad, sc_local, layer_idx, MM3_LM_ADAPTER_UP, fold_rows);
    return ggml_add(ctx, h,
                    mm3_lm_mm(ctx, w.ffn_down, ggml_mul(ctx, gate, up), ad, sc_local, layer_idx, MM3_LM_ADAPTER_DOWN, fold_rows));
}

// Build one slot. `decode` selects the input mode: false = token ids for T
// positions per CFG row, true = a single soft feedback embedding built in-graph
// from the previous frame's eight codes.
static bool mm3_lm_build_slot(const MM3Model & m, MM3LmGraph * g, MM3LmSlot * s, int64_t T, int64_t n_kv_pad,
                              bool decode, std::string * err) {
    const MM3LmConfig & c  = m.lm_cfg;
    const int64_t       H  = (int64_t) c.embedding_length;
    const int64_t       B  = (int64_t) g->rows();
    const int64_t       K  = (int64_t) g->n_takes;
    const int64_t       P  = (int64_t) g->cfg_rows;
    const int64_t       NC = (int64_t) c.num_codebooks - 1;

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (MM3_LM_MAX_NODES + 256) + ggml_graph_overhead_custom(MM3_LM_MAX_NODES, false);
    s->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!s->gbuf) {
        if (err) {
            *err = "out of host memory allocating the MM3 LM graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, s->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(s->gbuf);
        s->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the MM3 LM graph context";
        }
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MM3_LM_MAX_NODES, false);

    s->in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(s->in_pos, "mm3_lm_positions");
    ggml_set_input(s->in_pos);

    s->in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    ggml_set_name(s->in_rows, "mm3_lm_kv_rows");
    ggml_set_input(s->in_rows);

    s->in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, T);
    ggml_set_name(s->in_mask, "mm3_lm_mask");
    ggml_set_input(s->in_mask);

    ggml_tensor * h = nullptr;
    if (!decode) {
        // Prefill: [T*B] ids, cond row first. get_rows demands a 1-D index
        // tensor (a->ne[2] == b->ne[1]), so the two rows are flattened and the
        // result reshaped — identical memory order either way.
        s->in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T * B);
        ggml_set_name(s->in_ids, "mm3_lm_token_ids");
        ggml_set_input(s->in_ids);
        h = ggml_get_rows(ctx, m.lm.token_embd, s->in_ids);  // [H, T*B]
        h = ggml_reshape_3d(ctx, h, H, T, B);
    } else {
        // Decode: the soft feedback embedding (design note E). in_ids is
        // TAKE-MAJOR, K blocks of (NC + 1): within take t, entry 0 is the LM
        // vocab id of that take's semantic code and entries 1..NC are FLAT row
        // indices into depth.audio_embd (code + (i-1)*audio_vocab_size),
        // computed by the caller because the offset arithmetic is shared with
        // the depth decoder. At K = 1 this is the old [NC + 1] layout exactly.
        s->in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K * (NC + 1));
        ggml_set_name(s->in_ids, "mm3_lm_feedback_rows");
        ggml_set_input(s->in_ids);

        // Gather all K takes at once, then fold the per-take blocks apart with
        // reshapes. get_rows demands a 1-D index tensor, so the two strided
        // views below pick the semantic and acoustic entries out of the
        // interleaved take-major array. ggml_view_1d cannot stride, so the
        // semantic ids are gathered with a stride via view_2d instead: the
        // block is [1 sem, NC acoustic] per take, so sem sits every (NC+1).
        ggml_tensor * ids2 = ggml_reshape_2d(ctx, s->in_ids, NC + 1, K);   // [NC+1, K]
        ggml_tensor * sem_idx =
            ggml_cont(ctx, ggml_view_2d(ctx, ids2, 1, K, ids2->nb[1], 0));                     // [1, K]
        ggml_tensor * ac_idx =
            ggml_cont(ctx, ggml_view_2d(ctx, ids2, NC, K, ids2->nb[1], ggml_type_size(GGML_TYPE_I32)));  // [NC, K]
        sem_idx = ggml_reshape_1d(ctx, sem_idx, K);
        ac_idx  = ggml_reshape_1d(ctx, ac_idx, NC * K);

        ggml_tensor * e_sem = ggml_get_rows(ctx, m.lm.token_embd, sem_idx);          // [H, K]
        ggml_tensor * e_ac  = ggml_get_rows(ctx, m.synth.depth.audio_embd, ac_idx);  // [H, NC*K]

        // Sum the NC acoustic embeddings WITHIN each take. Reshaped to
        // [H, NC, K], a ne1-sliced view picks codebook i for every take at once,
        // so this stays NC adds regardless of K.
        e_sem              = ggml_reshape_3d(ctx, e_sem, H, 1, K);
        e_ac               = ggml_reshape_3d(ctx, e_ac, H, NC, K);
        ggml_tensor * acc  = ggml_view_3d(ctx, e_ac, H, 1, K, e_ac->nb[1], e_ac->nb[2], 0);
        for (int64_t i = 1; i < NC; i++) {
            acc = ggml_add(ctx, acc,
                           ggml_view_3d(ctx, e_ac, H, 1, K, e_ac->nb[1], e_ac->nb[2], (size_t) i * e_ac->nb[1]));
        }
        ggml_tensor * fb = ggml_scale(ctx, ggml_add(ctx, e_sem, acc), c.ar_embedding_scale);  // [H, 1, K]

        s->out_feedback = fb;
        ggml_set_name(s->out_feedback, "mm3_lm_feedback");
        ggml_set_output(s->out_feedback);

        // Both CFG rows of a take are fed the SAME embedding: within a take the
        // sampled codes are shared (the reference `.repeat(2)`s them) and only
        // the KV history differs. Takes, in contrast, are fully independent.
        //
        // repeat over ne1 then flatten IS repeat_interleave, and that is why the
        // row layout is take-major: the result indexes as r + P*t, i.e. exactly
        // t*cfg_rows + r. At K = 1, P = 2 this produces the same two rows the
        // old ggml_concat(fb, fb, 2) did, in the same order.
        fb = ggml_cont(ctx, fb);
        h  = ggml_reshape_3d(ctx, ggml_repeat_4d(ctx, fb, H, P, K, 1), H, 1, B);  // [H, 1, B]
    }

    // Alignment probe: keep a handle on every layer's post-softmax attention so
    // the AR loop can slice the lyric columns out of it. Costs nothing when off,
    // and cannot work under flash attention (which never materialises them) —
    // mm3_lm_prepare forces the manual path when the probe is enabled.
    const bool any_scores = g->dump_attn || g->align_capture;
    s->attn_scores.assign(any_scores ? m.lm.blk.size() : 0, nullptr);
    for (size_t i = 0; i < m.lm.blk.size(); i++) {
        ggml_tensor * sc = nullptr;
        // Which layers must materialise their softmax. The discovery dump wants
        // all of them; production alignment wants only the three that carry the
        // signal, so the other 33 keep flash attention and its speed.
        const bool want = g->dump_attn || (g->align_capture && mm3_align_layer_needed((int) i));
        h = mm3_lm_block(ctx, gf, c, m.lm.blk[i], h, s->in_pos, s->in_mask, s->in_rows, g->kv_k[i], g->kv_v[i],
                         n_kv_pad, g->use_flash_attn && !want, want ? &sc : nullptr, g->adapter, &g->adapter_scales,
                         (int) i, g->fold_rows);
        if (want && sc) {
            ggml_set_name(sc, ("mm3_lm_attn_" + std::to_string(i)).c_str());
            ggml_set_output(sc);
            ggml_build_forward_expand(gf, sc);
            s->attn_scores[i] = sc;
        }
    }
    h = mm3_lm_rms(ctx, h, m.lm.output_norm, c.rms_eps);  // [H, T, B]

    // Last position only — the head never reads anything else, and materialising
    // [200000, T, B] for a 70-token prefill would cost 112 MB for nothing.
    ggml_tensor * last =
        ggml_cont(ctx, ggml_view_3d(ctx, h, H, 1, B, h->nb[1], h->nb[2], (size_t) (T - 1) * h->nb[1]));

    s->out_hidden = last;
    ggml_set_name(s->out_hidden, "mm3_lm_last_hidden");
    ggml_set_output(s->out_hidden);

    // Full head, or just the EOS+semantic row span (a row-aligned view of the
    // possibly-quantized weight — rows are whole quant blocks, so any row
    // offset is representable). Identical arithmetic for every row that
    // survives: the sampler masks everything outside the span to -inf anyway.
    s->logits_lo = 0;
    s->logits_n  = (int64_t) c.vocab_size;
    ggml_tensor * head = m.lm.output;
    if (g->head_slice && mm3_lm_head_slice_span(c, &s->logits_lo, &s->logits_n)) {
        head = ggml_view_2d(ctx, m.lm.output, m.lm.output->ne[0], s->logits_n, m.lm.output->nb[1],
                            (size_t) s->logits_lo * m.lm.output->nb[1]);
    }
    s->out_logits = ggml_mul_mat(ctx, head, last);  // [logits_n, 1, B]
    ggml_set_name(s->out_logits, "mm3_lm_logits");
    ggml_set_output(s->out_logits);

    ggml_build_forward_expand(gf, s->out_hidden);
    ggml_build_forward_expand(gf, s->out_logits);
    if (s->out_feedback) {
        ggml_build_forward_expand(gf, s->out_feedback);
    }
    s->graph = gf;

    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, s->graph)) {
        ggml_free(ctx);
        free(s->gbuf);
        s->gbuf  = nullptr;
        s->graph = nullptr;
        if (err) {
            *err = std::string("MM3 LM graph allocation failed (out of VRAM?) for the ") +
                   (decode ? "decode" : "prefill") + " slot at T=" + std::to_string(T) +
                   ", kv=" + std::to_string(n_kv_pad);
        }
        return false;
    }

    s->gctx          = ctx;
    s->T             = T;
    s->n_kv_pad      = n_kv_pad;
    s->n_nodes       = ggml_graph_n_nodes(s->graph);
    s->compute_bytes = ggml_backend_sched_get_buffer_size(s->sched, g->backend);
    return true;
}

// ── Prep ────────────────────────────────────────────────────────────────────

static int64_t mm3_lm_bucket(int64_t n) {
    const int64_t b = MM3_LM_KV_BUCKET;
    return ((n + b - 1) / b) * b;
}

// Acquire the backend and make sure the KV cache covers `n_ctx_needed` positions.
// Set the ensemble take count. Must be called BEFORE mm3_lm_prepare, because
// the KV cache and every cached slot bake rows() into their shapes — so a
// change tears them down exactly like a weight swap does. Clamped by the
// caller (mm3_clamp_takes); this only enforces the hard kernel ceiling so a
// bad call site can never silently push ggml off the matrix-vector path.
static void mm3_lm_set_takes(MM3LmGraph * g, int takes) {
    if (takes < 1) {
        takes = 1;
    }
    if (takes > MM3_MAX_BATCH_ROWS) {
        takes = MM3_MAX_BATCH_ROWS;
    }
    // On everywhere, including single-track renders — see the note on
    // mm3_lm_mm for why reproducing a counterfactual seed is not a thing worth
    // paying 15 %/step for. MM3_LM_FOLD_ROWS=0 restores the old kernels.
    static const int forced = [] {
        const char * e = std::getenv("MM3_LM_FOLD_ROWS");
        if (!e || !e[0]) {
            return -1;
        }
        return e[0] == '0' ? 0 : 1;
    }();
    const bool fold = forced >= 0 ? forced == 1 : true;
    if (g->n_takes == takes && g->fold_rows == fold) {
        return;
    }
    mm3_lm_free(g);
    g->n_takes   = takes;
    g->fold_rows = fold;
}

// Grows only; a shrink would throw away a valid allocation for nothing.
static bool mm3_lm_prepare(const MM3Model & m, MM3LmGraph * g, int64_t n_ctx_needed, std::string * err) {
    // Residency is staged (mm3-model.h): check the parts this graph actually
    // reads, not the all-three `loaded` flag. During stage 2 the LM is gone by
    // design and `loaded` is false, so the old check would have refused a
    // perfectly valid model.
    if (!m.lm_resident || !m.depth_resident) {
        if (err) {
            *err = "MiniMax-Music3 is not warm (POST /mm3/warm first)";
        }
        return false;
    }
    const MM3LmConfig & c = m.lm_cfg;
    if (c.block_count == 0 || c.embedding_length == 0 || c.head_count == 0 || c.key_length == 0) {
        if (err) {
            *err = "LM config is empty — qwen3.* KVs missing from the LM GGUF";
        }
        return false;
    }
    if (!m.lm.token_embd || !m.lm.output || !m.lm.output_norm) {
        if (err) {
            *err = "the LM weights are not resident";
        }
        return false;
    }
    if (!m.synth.depth.audio_embd) {
        if (err) {
            *err = "depth.audio_embd is not resident; the AR feedback embedding needs it";
        }
        return false;
    }

    const void * lt = (const void *) m.wctx_lm.buffer;
    // The only non-LM weight this graph touches is depth.audio_embd (the AR
    // feedback embedding), which lives in the DEPTH buffer — so that is the
    // buffer whose identity must invalidate the cached graph.
    const void * st = (const void *) m.wctx_depth.buffer;
    // A guidance-distilled LM (cfg_scale 1.0) needs single-row graphs and a
    // half-size KV cache, so a checkpoint swap that changes the scale must tear
    // everything down exactly like a weight swap does — the cached slots bake
    // the row count into their shapes.
    const int rows = mm3_cfg_rows(c);
    // Ensemble takes tear everything down for the same reason: the KV cache and
    // every slot bake rows() = cfg_rows * n_takes into their shapes. `n_takes`
    // is set by mm3_lm_set_takes BEFORE this call.
    if (g->lm_token != lt || g->synth_token != st || g->cfg_rows != rows) {
        mm3_lm_free(g);
    }
    g->cfg_rows = rows;

    if (!g->backend_ref) {
        BackendPair bp = backend_init("MM3-LM");
        g->bp          = bp;
        g->backend     = bp.backend;
        g->cpu_backend = bp.cpu_backend;
        g->backend_ref = true;
        // MM3_LM_NO_FLASH=1 forces the manual F32 soft_max path — a parity-debug
        // escape hatch, not a production knob.
        const char * no_fa = std::getenv("MM3_LM_NO_FLASH");
        const char * dump  = std::getenv("MM3_ALIGN_DUMP");
        g->dump_attn       = dump && dump[0] && dump[0] != '0';
        // Alignment capture forces the manual path on EVERY layer, not just the
        // three whose scores are read.
        //
        // That is not the obvious choice and it was not the first one: capturing
        // only layers 12/19/24 and leaving the other 33 on flash costs far less
        // (9.8 vs 11.2 ms/step) and produces the right audio — but it produces
        // WRONG alignment. Measured on the same seed, selective capture stamped
        // the four lyric lines at 0.9/3.6/7.5/10.3 s where the vocal actually
        // sits at 0.1/10.2/15.1/19.4; all-manual gives 2.2/9.9/16.5/19.5. The
        // captured layers' attention evidently depends on whether the layers
        // FEEDING them ran flash or manual, by more than the numerical noise
        // that difference is supposed to be. Until that is understood, the mixed
        // graph is not a safe optimisation — a silently mistimed LRC is worse
        // than a slower one.
        g->use_flash_attn  = bp.has_gpu && !HOT_STEP_FA_DISABLED && !g->dump_attn && !g->align_capture &&
                             !(no_fa && no_fa[0] && no_fa[0] != '0');
        if (g->dump_attn) {
            fprintf(stderr, "[MM3-Align] Attention DUMP on — manual F32 attention for all %d layers "
                            "(discovery only, slow)\n", (int) m.lm.blk.size());
        }
        g->lm_token        = lt;
        g->synth_token     = st;
    }

    const int64_t want = mm3_lm_bucket(n_ctx_needed);
    if (g->n_ctx >= want) {
        return true;
    }

    // A bigger cache means new tensors, so every graph that views them dies too.
    mm3_lm_free_slot(&g->prefill);
    mm3_lm_free_slot(&g->decode);
    mm3_lm_free_slot(&g->replay);
    mm3_lm_free_kv(g);

    const int64_t D   = (int64_t) c.key_length;
    const int64_t Nkv = (int64_t) c.head_count_kv;
    const int     L   = (int) c.block_count;

    ggml_init_params ip = { (size_t) (L * 2) * ggml_tensor_overhead() + 1024, NULL, /*no_alloc*/ true };
    g->kv_ctx           = ggml_init(ip);
    if (!g->kv_ctx) {
        if (err) {
            *err = "ggml_init failed for the MM3 LM KV cache context";
        }
        return false;
    }
    g->kv_k.assign((size_t) L, nullptr);
    g->kv_v.assign((size_t) L, nullptr);
    for (int i = 0; i < L; i++) {
        char nm[64];
        g->kv_k[(size_t) i] = ggml_new_tensor_4d(g->kv_ctx, GGML_TYPE_F16, D, want, Nkv, g->rows());
        snprintf(nm, sizeof(nm), "mm3.lm.kv_k.%d", i);
        ggml_set_name(g->kv_k[(size_t) i], nm);
        g->kv_v[(size_t) i] = ggml_new_tensor_4d(g->kv_ctx, GGML_TYPE_F16, D, want, Nkv, g->rows());
        snprintf(nm, sizeof(nm), "mm3.lm.kv_v.%d", i);
        ggml_set_name(g->kv_v[(size_t) i], nm);
    }
    g->kv_buf = ggml_backend_alloc_ctx_tensors(g->kv_ctx, g->backend);
    if (!g->kv_buf) {
        ggml_free(g->kv_ctx);
        g->kv_ctx = nullptr;
        g->kv_k.clear();
        g->kv_v.clear();
        if (err) {
            *err = "backend buffer allocation failed for the MM3 LM KV cache (" +
                   std::to_string((long long) want) + " positions, out of VRAM?)";
        }
        return false;
    }
    // The attention window is padded past kv_pos; the masked tail must read
    // finite values, never uninitialised F16 bit patterns that decode to NaN.
    ggml_backend_buffer_clear(g->kv_buf, 0);

    g->n_ctx    = want;
    g->kv_bytes = ggml_backend_buffer_get_size(g->kv_buf);
    g->kv_pos   = 0;

    const std::string takes_note = g->n_takes > 1 ? " (" + std::to_string(g->n_takes) + " takes)" : std::string();
    fprintf(stderr,
            "[MM3-LM] KV cache: %lld positions x %d layers x %d row%s%s = %.2f GB (%.0f kB/position), flash=%s\n",
            (long long) want, L, g->rows(), g->rows() == 1 ? "" : "s", takes_note.c_str(),
            (double) g->kv_bytes / (1024.0 * 1024.0 * 1024.0),
            (double) g->kv_bytes / (double) want / 1024.0, g->use_flash_attn ? "yes" : "no");
    return true;
}

static void mm3_lm_reset(MM3LmGraph * g) {
    g->kv_pos = 0;
}

// ── Step plumbing ───────────────────────────────────────────────────────────

// Fill positions / KV rows / causal mask for a step of T tokens starting at
// kv_pos, and upload them. The mask covers the whole padded window on every path:
// row i (absolute position kv_pos+i) attends columns 0..kv_pos+i and nothing else.
static void mm3_lm_upload_step(MM3LmGraph * g, MM3LmSlot * s, int64_t T, int64_t n_kv_pad) {
    g->pos_host.resize((size_t) T);
    g->rows_host.resize((size_t) T);
    g->mask_host.resize((size_t) (n_kv_pad * T));
    for (int64_t i = 0; i < T; i++) {
        const int64_t abs      = g->kv_pos + i;
        g->pos_host[(size_t) i]  = (int32_t) abs;
        g->rows_host[(size_t) i] = abs;
        for (int64_t j = 0; j < n_kv_pad; j++) {
            g->mask_host[(size_t) (i * n_kv_pad + j)] = ggml_fp32_to_fp16(j <= abs ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(s->in_pos, g->pos_host.data(), 0, (size_t) T * sizeof(int32_t));
    ggml_backend_tensor_set(s->in_rows, g->rows_host.data(), 0, (size_t) T * sizeof(int64_t));
    ggml_backend_tensor_set(s->in_mask, g->mask_host.data(), 0, (size_t) (n_kv_pad * T) * sizeof(uint16_t));
}

static void mm3_lm_read_outputs(const MM3LmConfig & c, const MM3LmSlot & s, float * out_hidden, float * out_logits,
                                float * out_feedback) {
    const size_t H    = (size_t) c.embedding_length;
    // Rows this slot actually computed. Callers size their buffers to the
    // MM3_LM_CFG_ROWS ceiling, so reading fewer rows is always safe; reading
    // the ceiling from a 1-row output tensor would overrun it.
    const size_t rows = (size_t) s.out_hidden->ne[2];
    if (out_hidden) {
        ggml_backend_tensor_get(s.out_hidden, out_hidden, 0, H * rows * sizeof(float));
    }
    if (out_logits) {
        // The caller's buffer is sized [logits_n, B] — full V or the sliced
        // span, whichever this slot was built for.
        ggml_backend_tensor_get(s.out_logits, out_logits, 0, (size_t) s.logits_n * rows * sizeof(float));
    }
    if (out_feedback && s.out_feedback) {
        ggml_backend_tensor_get(s.out_feedback, out_feedback, 0, H * sizeof(float));
    }
}

// Copy one layer's post-softmax attention for the CONDITIONAL CFG row (row 0)
// into `dst`, restricted to KV columns [col0, col1). Decode graphs have T = 1,
// so the result is [Nh, n_cols]: for each head, its distribution over those
// prompt positions at this frame. That is exactly the per-frame column of a
// lyric-token x frame alignment matrix.
//
// Returns false when the probe is off or the layer has no score tensor.
static bool mm3_lm_read_attn_slice(const MM3LmConfig & c, const MM3LmSlot & s, size_t layer,
                                   int64_t col0, int64_t col1, std::vector<float> & dst) {
    if (layer >= s.attn_scores.size() || !s.attn_scores[layer]) {
        return false;
    }
    ggml_tensor * sc = s.attn_scores[layer];      // [n_kv, T, Nh, B]
    const int64_t n_kv = sc->ne[0];
    const int64_t T    = sc->ne[1];
    const int64_t Nh   = sc->ne[2];
    if (col0 < 0 || col1 > n_kv || col1 <= col0) {
        return false;
    }
    const int64_t n_cols = col1 - col0;
    dst.assign((size_t) (Nh * n_cols), 0.0f);

    // Row 0 of the CFG batch is the conditional one — the only row whose context
    // contains the lyrics. ne3 is the batch axis.
    std::vector<float> row((size_t) n_kv);
    for (int64_t hh = 0; hh < Nh; hh++) {
        const size_t off = (size_t) ((T - 1) * sc->nb[1] + hh * sc->nb[2] + 0 * sc->nb[3]);
        ggml_backend_tensor_get(sc, row.data(), off, (size_t) n_kv * sizeof(float));
        for (int64_t j = 0; j < n_cols; j++) {
            dst[(size_t) (hh * n_cols + j)] = row[(size_t) (col0 + j)];
        }
    }
    (void) c;
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

// Prefill both CFG rows of the prompt in one pass and return the last position's
// hidden state and head logits.
//
//   ids_cond / ids_uncond   n_prompt token ids each; row 0 is conditional
//   out_hidden              [2, H] F32, row 0 conditional
//   out_logits              [2, V] F32, row 0 conditional
//
// Resets the KV cache: a prefill always starts a new sequence.
static bool mm3_lm_prefill(const MM3Model & m, MM3LmGraph * g, const int32_t * ids_cond, const int32_t * ids_uncond,
                           int64_t n_prompt, float * out_hidden, float * out_logits, std::string * err) {
    const MM3LmConfig & c = m.lm_cfg;
    if (n_prompt <= 0) {
        if (err) {
            *err = "the prompt is empty";
        }
        return false;
    }
    if (n_prompt > g->n_ctx) {
        if (err) {
            *err = "prompt of " + std::to_string((long long) n_prompt) + " tokens exceeds the KV cache (" +
                   std::to_string((long long) g->n_ctx) + ")";
        }
        return false;
    }

    const int64_t n_kv_pad = std::min<int64_t>(mm3_lm_bucket(n_prompt), g->n_ctx);
    if (!g->prefill.graph || g->prefill.T != n_prompt || g->prefill.n_kv_pad != n_kv_pad) {
        mm3_lm_free_slot(&g->prefill);
        if (!g->prefill.sched) {
            g->prefill.sched = backend_sched_new(g->bp, MM3_LM_MAX_NODES * 2);
        }
        if (!mm3_lm_build_slot(m, g, &g->prefill, n_prompt, n_kv_pad, /*decode*/ false, err)) {
            return false;
        }
        fprintf(stderr, "[MM3-LM] Prefill graph: T=%lld kv=%lld, %d nodes, %.1f MB compute\n", (long long) n_prompt,
                (long long) n_kv_pad, g->prefill.n_nodes, (double) g->prefill.compute_bytes / (1024.0 * 1024.0));
    }

    g->kv_pos = 0;

    // The prefill index array is B contiguous blocks of n_prompt (get_rows +
    // reshape to [H, T, B] means block b is row b). Every take starts from the
    // SAME prompt — takes diverge only through sampling — so this is the CFG
    // pair written K times, take-major.
    //
    // At cfg_rows = 1 the unconditional row is never built, so ids_uncond is
    // accepted and ignored rather than made optional at every call site.
    const int64_t B = (int64_t) g->rows();
    g->ids_host.resize((size_t) (n_prompt * B));
    for (int t = 0; t < g->n_takes; t++) {
        int32_t * blk = g->ids_host.data() + (size_t) (t * g->cfg_rows) * (size_t) n_prompt;
        memcpy(blk, ids_cond, (size_t) n_prompt * sizeof(int32_t));
        if (g->cfg_rows > 1) {
            memcpy(blk + n_prompt, ids_uncond, (size_t) n_prompt * sizeof(int32_t));
        }
    }
    ggml_backend_tensor_set(g->prefill.in_ids, g->ids_host.data(), 0, (size_t) (n_prompt * B) * sizeof(int32_t));
    mm3_lm_upload_step(g, &g->prefill, n_prompt, n_kv_pad);

    mm3_imatrix_hook(g->prefill.sched);
    if (ggml_backend_sched_graph_compute(g->prefill.sched, g->prefill.graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "MM3 LM prefill graph compute failed";
        }
        return false;
    }
    mm3_lm_read_outputs(c, g->prefill, out_hidden, out_logits, nullptr);
    g->kv_pos = n_prompt;
    return true;
}

// One decode step. Builds the soft feedback embedding from the previous frame's
// codes in-graph, advances both CFG rows by one position, and returns the new
// last hidden state, head logits, and (for parity) the feedback embedding itself.
//
//   sem_token_id   [takes] LM vocab ids of each take's semantic code (+ offset)
//   acoustic_rows  [takes, num_codebooks-1] FLAT depth.audio_embd row indices
//   out_feedback   [H] F32 (take 0), may be null
//
// Both inputs are arrays of `g->n_takes` entries; at K = 1 that is a pointer to
// a single value and the old one-song call site is unchanged.
static bool mm3_lm_decode(const MM3Model & m, MM3LmGraph * g, const int32_t * sem_token_id,
                          const int32_t * acoustic_rows, float * out_hidden, float * out_logits, float * out_feedback,
                          std::string * err) {
    const MM3LmConfig & c  = m.lm_cfg;
    const int64_t       NC = (int64_t) c.num_codebooks - 1;

    if (g->kv_pos + 1 > g->n_ctx) {
        if (err) {
            *err = "the KV cache is full at " + std::to_string((long long) g->n_ctx) + " positions";
        }
        return false;
    }

    const int64_t n_kv_pad = std::min<int64_t>(mm3_lm_bucket(g->kv_pos + 1), g->n_ctx);
    if (!g->decode.graph || g->decode.n_kv_pad != n_kv_pad) {
        mm3_lm_free_slot(&g->decode);
        if (!g->decode.sched) {
            g->decode.sched = backend_sched_new(g->bp, MM3_LM_MAX_NODES * 2);
        }
        if (!mm3_lm_build_slot(m, g, &g->decode, 1, n_kv_pad, /*decode*/ true, err)) {
            return false;
        }
        fprintf(stderr, "[MM3-LM] Decode graph: kv=%lld, %d nodes, %.1f MB compute\n", (long long) n_kv_pad,
                g->decode.n_nodes, (double) g->decode.compute_bytes / (1024.0 * 1024.0));
    }

    // Take-major blocks of [sem, ac0..ac_{NC-1}], matching the slot's reshape.
    const int64_t K = (int64_t) g->n_takes;
    g->ids_host.resize((size_t) (K * (NC + 1)));
    for (int64_t t = 0; t < K; t++) {
        int32_t * blk = g->ids_host.data() + (size_t) (t * (NC + 1));
        blk[0]        = sem_token_id[t];
        memcpy(blk + 1, acoustic_rows + t * NC, (size_t) NC * sizeof(int32_t));
    }
    ggml_backend_tensor_set(g->decode.in_ids, g->ids_host.data(), 0, (size_t) (K * (NC + 1)) * sizeof(int32_t));
    mm3_lm_upload_step(g, &g->decode, 1, n_kv_pad);

    mm3_imatrix_hook(g->decode.sched);
    if (ggml_backend_sched_graph_compute(g->decode.sched, g->decode.graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "MM3 LM decode graph compute failed";
        }
        return false;
    }
    mm3_lm_read_outputs(c, g->decode, out_hidden, out_logits, out_feedback);
    g->kv_pos++;
    return true;
}

// ── LRC replay: post-hoc alignment capture ──────────────────────────────────
//
// The live capture path (MM3_LRC_LIVE=1) pays ~+41 % on EVERY decode step,
// because reading attention requires the manual F32 path and — measured, not
// assumed — it must run on ALL 36 layers or the alignment comes out wrong
// (see the trap note in mm3_lm_prepare). This replay pass gets the same
// attention for a fraction of the cost by exploiting causality: for a GIVEN
// token sequence, a teacher-forced prefill computes exactly the attention the
// step-by-step decode computed. So the decode runs pure flash (zero capture
// overhead, audio bit-identical to a no-LRC render), and afterwards — while
// the LM is still resident, before the staged handover — the full sequence
// [prompt + frame feedback embeddings] is re-run in chunks with manual
// attention everywhere, and the three alignment heads' lyric columns are read
// out of it.
//
// Three properties keep this in the VALIDATED regime rather than the broken
// mixed one:
//   1. Every replayed layer is manual — no flash/manual mixing inside a
//      forward, which is what produced the wrong-alignment trap.
//   2. Only blocks 0..mm3_align_max_layer() run. Layers above the deepest
//      capture layer feed nothing the replay reads (causally downstream), so
//      truncating them cannot change the captured scores.
//   3. One CFG row. The conditional row's KV never depends on the uncond row,
//      so the replay reproduces row 0 of the decode exactly (via ne3=1 views
//      onto row 0 of the shared KV cache — whose contents are clobbered, which
//      is fine: the AR stage is done with them and the handover frees them
//      next).
//
// The per-position input embedding reproduces both prefill and decode inputs
// in one formula: h[t] = (token_embd[id] + gate * SUM_k audio_embd[ac_k]) *
// scale, with gate=0/scale=1 on prompt positions (plain token embedding) and
// gate=1/scale=ar_embedding_scale on frame positions (the decode graph's
// feedback embedding, same summation order, so the arithmetic matches).

// Queries per replay chunk. The manual-attention score tensor is
// [n_kv_pad, T, Nh] F32 per layer transiently, so T trades VRAM for launch
// count: 256 keeps the peak a few hundred MB at 5-minute song lengths.
#define MM3_LRC_REPLAY_CHUNK 256

static bool mm3_lm_build_replay_slot(const MM3Model & m, MM3LmGraph * g, MM3LmSlot * s, int64_t T, int64_t n_kv_pad,
                                     std::string * err) {
    const MM3LmConfig & c    = m.lm_cfg;
    const int64_t       H    = (int64_t) c.embedding_length;
    const int64_t       D    = (int64_t) c.key_length;
    const int64_t       Nkv  = (int64_t) c.head_count_kv;
    const int64_t       NC   = (int64_t) c.num_codebooks - 1;
    const int           Lmax = mm3_align_max_layer();

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (MM3_LM_MAX_NODES + 256) + ggml_graph_overhead_custom(MM3_LM_MAX_NODES, false);
    s->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!s->gbuf) {
        if (err) {
            *err = "out of host memory allocating the MM3 LM replay graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, s->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(s->gbuf);
        s->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the MM3 LM replay graph context";
        }
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MM3_LM_MAX_NODES, false);

    s->in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(s->in_pos, "mm3_lm_replay_positions");
    ggml_set_input(s->in_pos);
    s->in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    ggml_set_name(s->in_rows, "mm3_lm_replay_kv_rows");
    ggml_set_input(s->in_rows);
    s->in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, T);
    ggml_set_name(s->in_mask, "mm3_lm_replay_mask");
    ggml_set_input(s->in_mask);
    s->in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(s->in_ids, "mm3_lm_replay_ids");
    ggml_set_input(s->in_ids);
    s->in_ac = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T * NC);
    ggml_set_name(s->in_ac, "mm3_lm_replay_ac_rows");
    ggml_set_input(s->in_ac);
    s->in_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T);
    ggml_set_name(s->in_gate, "mm3_lm_replay_gate");
    ggml_set_input(s->in_gate);
    s->in_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T);
    ggml_set_name(s->in_scale, "mm3_lm_replay_scale");
    ggml_set_input(s->in_scale);

    // The hybrid input embedding. Acoustic rows are position-major (position t's
    // NC entries at t*NC..t*NC+NC-1), so position t's k-th code sits at column
    // t*NC + k of the gather — a stride-NC*nb1 view per k, summed in the same
    // left-to-right order the decode graph uses.
    ggml_tensor * h_tok = ggml_get_rows(ctx, m.lm.token_embd, s->in_ids);          // [H, T]
    ggml_tensor * e_ac  = ggml_get_rows(ctx, m.synth.depth.audio_embd, s->in_ac);  // [H, T*NC]
    ggml_tensor * acc   = ggml_view_2d(ctx, e_ac, H, T, (size_t) NC * e_ac->nb[1], 0);
    for (int64_t k = 1; k < NC; k++) {
        acc = ggml_add(ctx, acc,
                       ggml_view_2d(ctx, e_ac, H, T, (size_t) NC * e_ac->nb[1], (size_t) k * e_ac->nb[1]));
    }
    ggml_tensor * h = ggml_add(ctx, h_tok, ggml_mul(ctx, acc, s->in_gate));
    h               = ggml_mul(ctx, h, s->in_scale);
    h               = ggml_reshape_3d(ctx, h, H, T, 1);

    s->attn_scores.assign(m.lm.blk.size(), nullptr);
    for (int i = 0; i <= Lmax && i < (int) m.lm.blk.size(); i++) {
        // ne3=1 views onto CFG row 0 of the shared cache — offset 0, contiguous.
        ggml_tensor * kc0 = ggml_view_4d(ctx, g->kv_k[(size_t) i], D, g->n_ctx, Nkv, 1, g->kv_k[(size_t) i]->nb[1],
                                         g->kv_k[(size_t) i]->nb[2], g->kv_k[(size_t) i]->nb[3], 0);
        ggml_tensor * vc0 = ggml_view_4d(ctx, g->kv_v[(size_t) i], D, g->n_ctx, Nkv, 1, g->kv_v[(size_t) i]->nb[1],
                                         g->kv_v[(size_t) i]->nb[2], g->kv_v[(size_t) i]->nb[3], 0);
        ggml_tensor * sc   = nullptr;
        const bool    want = mm3_align_layer_needed(i);
        h = mm3_lm_block(ctx, gf, c, m.lm.blk[(size_t) i], h, s->in_pos, s->in_mask, s->in_rows, kc0, vc0, n_kv_pad,
                         /*use_flash*/ false, want ? &sc : nullptr, g->adapter, &g->adapter_scales, i);
        if (want && sc) {
            ggml_set_name(sc, ("mm3_lm_replay_attn_" + std::to_string(i)).c_str());
            ggml_set_output(sc);
            ggml_build_forward_expand(gf, sc);
            s->attn_scores[(size_t) i] = sc;
        }
    }
    s->graph = gf;

    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, s->graph)) {
        ggml_free(ctx);
        free(s->gbuf);
        s->gbuf  = nullptr;
        s->graph = nullptr;
        if (err) {
            *err = "MM3 LM replay graph allocation failed (out of VRAM?) at T=" + std::to_string((long long) T) +
                   ", kv=" + std::to_string((long long) n_kv_pad);
        }
        return false;
    }
    s->gctx          = ctx;
    s->T             = T;
    s->n_kv_pad      = n_kv_pad;
    s->n_nodes       = ggml_graph_n_nodes(s->graph);
    s->compute_bytes = ggml_backend_sched_get_buffer_size(s->sched, g->backend);
    return true;
}

// Run the replay over [prompt + n_steps frame feedbacks] and fill `flat` in
// mm3_align_build_lrc()'s [head][token][frame] layout. `sem_all` / `ac_all`
// are the AR result's code arrays (raw codes, no vocab offset). Clobbers the
// KV cache — call only after the AR stage is completely done with it.
static bool mm3_lm_lrc_replay(const MM3Model & m, MM3LmGraph * g, const int32_t * cond_ids, int64_t n_prompt,
                              const int32_t * sem_all, const int32_t * ac_all, int64_t n_steps, int64_t lyr0,
                              int64_t lyr1, std::vector<float> * flat, std::string * err) {
    const MM3LmConfig & c     = m.lm_cfg;
    const int64_t       NC    = (int64_t) c.num_codebooks - 1;
    const int64_t       AV    = (int64_t) c.acoustic_vocab_size;
    const int64_t       OFF   = (int64_t) c.semantic_vocab_offset;
    const int64_t       total = n_prompt + n_steps;
    const int64_t       n_tok = lyr1 - lyr0;
    if (n_steps <= 0 || n_tok <= 0) {
        if (err) {
            *err = "nothing to replay";
        }
        return false;
    }
    if (total > g->n_ctx) {
        if (err) {
            *err = "replay of " + std::to_string((long long) total) + " positions exceeds the KV cache (" +
                   std::to_string((long long) g->n_ctx) + ")";
        }
        return false;
    }
    const int64_t n_kv_pad = std::min<int64_t>(mm3_lm_bucket(total), g->n_ctx);

    flat->assign((size_t) (MM3_ALIGN_N_HEADS * n_tok * n_steps), 0.0f);

    const auto t0 = std::chrono::steady_clock::now();
    g->kv_pos     = 0;

    std::vector<int32_t> ids, ac;
    std::vector<float>   gate, scl, row((size_t) n_tok);
    int                  n_chunks = 0;
    while (g->kv_pos < total) {
        const int64_t T = std::min<int64_t>(MM3_LRC_REPLAY_CHUNK, total - g->kv_pos);
        if (!g->replay.graph || g->replay.T != T || g->replay.n_kv_pad != n_kv_pad) {
            mm3_lm_free_slot(&g->replay);
            if (!g->replay.sched) {
                g->replay.sched = backend_sched_new(g->bp, MM3_LM_MAX_NODES * 2);
            }
            if (!mm3_lm_build_replay_slot(m, g, &g->replay, T, n_kv_pad, err)) {
                return false;
            }
        }
        MM3LmSlot & s = g->replay;

        ids.assign((size_t) T, 0);
        ac.assign((size_t) (T * NC), 0);
        gate.assign((size_t) T, 0.0f);
        scl.assign((size_t) T, 1.0f);
        for (int64_t i = 0; i < T; i++) {
            const int64_t abs = g->kv_pos + i;
            if (abs < n_prompt) {
                ids[(size_t) i] = cond_ids[abs];
            } else {
                const int64_t j = abs - n_prompt;
                ids[(size_t) i] = (int32_t) OFF + sem_all[j];
                for (int64_t k = 0; k < NC; k++) {
                    ac[(size_t) (i * NC + k)] = ac_all[j * NC + k] + (int32_t) (k * AV);
                }
                gate[(size_t) i] = 1.0f;
                scl[(size_t) i]  = c.ar_embedding_scale;
            }
        }
        ggml_backend_tensor_set(s.in_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(s.in_ac, ac.data(), 0, ac.size() * sizeof(int32_t));
        ggml_backend_tensor_set(s.in_gate, gate.data(), 0, gate.size() * sizeof(float));
        ggml_backend_tensor_set(s.in_scale, scl.data(), 0, scl.size() * sizeof(float));
        mm3_lm_upload_step(g, &s, T, n_kv_pad);

        if (ggml_backend_sched_graph_compute(s.sched, s.graph) != GGML_STATUS_SUCCESS) {
            if (err) {
                *err = "MM3 LM replay graph compute failed";
            }
            return false;
        }

        for (int hh = 0; hh < MM3_ALIGN_N_HEADS; hh++) {
            const MM3AlignHead & ah = MM3_ALIGN_HEADS[hh];
            ggml_tensor *        sc = s.attn_scores[(size_t) ah.layer];
            if (!sc) {
                continue;
            }
            for (int64_t i = 0; i < T; i++) {
                const int64_t abs = g->kv_pos + i;
                if (abs < n_prompt) {
                    continue;
                }
                const int64_t j   = abs - n_prompt;
                const size_t  off = (size_t) lyr0 * sc->nb[0] + (size_t) i * sc->nb[1] + (size_t) ah.head * sc->nb[2];
                ggml_backend_tensor_get(sc, row.data(), off, (size_t) n_tok * sizeof(float));
                for (int64_t t = 0; t < n_tok; t++) {
                    (*flat)[(size_t) ((hh * n_tok + t) * n_steps + j)] = row[(size_t) t];
                }
            }
        }
        g->kv_pos += T;
        n_chunks++;
    }

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[MM3-LRC] Replay: %lld positions (%lld frames) in %d chunks, blocks 0..%d, %.0f ms\n",
            (long long) total, (long long) n_steps, n_chunks, mm3_align_max_layer(), ms);
    return true;
}
