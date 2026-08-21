#pragma once
// minimax/mm3-depth-graph.h — MiniMax-Music3 RVQ depth decoder (0.6B).
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-server.h,
// which is itself the single hook include into tools/hot-step-server.cpp.
//
// SCOPE (increment 4a): one FRAME of depth decoding — given the global LM's
// last_hidden_state (both CFG rows) and the frame's semantic code, sample the
// seven residual RVQ codes c1..c7 and collect the seven hidden states that
// become 7/8 of that frame's flow conditioning. The AR loop that calls this
// 25 times per second, and the LM that produces `lm_hidden`, are later
// increments. Validated standalone through POST /mm3/depth-frame against the
// diffusers reference dumps in mm3-weights/fixtures/.
//
// ── Architecture (contract: docs/plans/mm3-gguf-layout.md §3.2 + §10, and the
//    diffusers reference MiniMaxMusic3RVQDepthDecoder at commit dafe3733) ──────
//
//   Per frame, for codebook c = 1..7:
//     seq = [ proj(lm_hidden), proj(embed_sem), proj(embed_ac_1..ac_{c-1}) ]
//     h   = seq + pos_embd[0 : c+1]            (learned ABSOLUTE positions)
//     4 x block:
//        h += attn(RMSNorm(h))                 16 heads x 256, CAUSAL, NO RoPE
//        h += down(silu(gate(RMSNorm(h))) * up(RMSNorm(h)))
//     h   = RMSNorm(h)[last position]
//     logits_c = head_{c-1}(h)                 Linear 4096 -> 1024, no bias
//     code_c   = sample(uncond + (cond - uncond) * ar_cfg_scale)
//
// ── Conventions that are easy to get wrong, and where each is pinned ──────────
//
// 1. THE SEMANTIC CODE IS EMBEDDED BY THE **LM's** token table, NOT by
//    depth.audio_embd. The reference is explicit:
//      `code_embed = language_model.model.embed_tokens(semantic_code + 151675)`
//    Only the six acoustic feedback embeddings come from `depth.audio_embd`.
//    That is why this graph reads tensors out of BOTH GGUF files, and why the
//    depth decoder cannot run without the LM file resident (layout doc §2.1).
//
// 2. THE ACOUSTIC EMBEDDING TABLE IS ONE FLAT 7168-ROW TABLE. Codebook i (1-based)
//    reads row `code + (i-1) * audio_vocab_size`. Getting the offset wrong is
//    silent — every row is a plausible embedding.
//
// 3. EVERY TOKEN GOES THROUGH `depth.proj` (4096->4096, no bias) BEFORE ENTRY,
//    including the LM hidden state and including both embedding tables.
//
// 4. POSITIONS ARE A LEARNED ABSOLUTE EMBEDDING ADDED AFTER THE PROJECTION, and
//    there is NO RoPE (`mm3.depth.rope = false`). The table has 16 rows and the
//    sequence never exceeds 8, so rows 8..15 are dead weight.
//
// 5. CFG IS A REAL 2-ROW BATCH HERE, unlike the flow stage. The AR stage runs a
//    persistent [conditional, unconditional] pair through the LM, and the depth
//    decoder inherits it: both rows are decoded together, the CFG combine happens
//    on the sampled logits, and the SAMPLED CODE IS SHARED by both rows (the
//    reference `.repeat(2)`s it). Row 0 is conditional — pinned by the fixture
//    manifest's `cfg_row_order`.
//
// 6. ONLY THE CONDITIONAL ROW'S HIDDEN STATES ARE KEPT. `hidden_parts.append(
//    hidden[:1])` in the reference. The unconditional row exists purely to guide
//    sampling.
//
// ── Design decisions, and why ─────────────────────────────────────────────────
//
// A. SEVEN CACHED GRAPHS, ONE PER CODEBOOK STEP, EACH AT ITS EXACT SEQUENCE
//    LENGTH. The sequence grows by one token per codebook (2, 3, ... 8) and the
//    output head changes every step, so a single graph cannot serve all seven
//    without either dynamic indexing (ggml has none) or wasted compute. Seven
//    graphs cost ~35 MB of compute buffer in total and are built once.
//
//    Each step also gets its OWN scheduler. One shared scheduler would need
//    reset + re-split + possible gallocr buffer churn on every step — 7 times
//    per frame, 25 frames per second of audio. Seven schedulers make each step
//    a pure `ggml_backend_sched_graph_compute` with no allocator involvement.
//
// B. NO KV CACHE — FULL RECOMPUTE PER CODEBOOK. The reference does the same
//    (layout doc §9 Q6). At ≤8 tokens the cache bookkeeping would cost more than
//    the ~2.3x of redundant attention it saves, and a cache would have to be
//    invalidated per frame anyway.
//
// C. MANUAL F32 ATTENTION WITH AN EXPLICIT CAUSAL MASK, NOT flash_attn_ext.
//    The sequences are 2..8 tokens; flash attention's win is asymptotic and its
//    F16 K/V cast is pure downside at this size. The mask is a per-step F32
//    constant built once into the derived-weight buffer.
//
// D. SAMPLING IS SEEDED TOP-K MULTINOMIAL, GREEDY, OR FORCED. The reference
//    samples multinomially from the top-50 of the CFG-guided distribution with a
//    torch Generator, which is not reproducible outside torch. Parity is therefore
//    validated on LOGITS, with `forced_codes` feeding the fixture's own sampled
//    codes so the sequence the graph sees is bit-for-bit the reference's.
//    UPDATED (increment 5): the AR loop passes its own std::mt19937_64 and the
//    checkpoint's top-k, and the same `mm3_sample_top_k` the LM uses runs here —
//    the seam design note A anticipated. With no generator the behaviour is
//    unchanged: greedy argmax over the CFG-guided logits, which is what
//    POST /mm3/depth-frame still does by default.
//
// ── Parity, measured 2026-08-13 (AR iteration 0, RTX 5090, f16 GGUF) ──────────
//
// Fed `lm_i0_last_hidden` (both rows), semantic = `depth_i0_codes[0]` = 13095,
// forced = `depth_i0_acoustic_codes`. The fixtures come from a CPU **bf16**
// reference run, so they are not ground truth; the reference module was re-run in
// **float32** on the same input sequence to separate "our port is wrong" from
// "the capture is bf16":
//
//                          ours vs bf16 dump     ours vs fp32 ref    dump vs fp32 ref
//   logits, conditional    0.9999910 / 4.2e-3    0.9999998 / 6.2e-4  0.9999910 / 4.2e-3
//   logits, uncondition'l  0.9999872 / 5.1e-3    0.9999998 / 6.4e-4  0.9999872 / 5.0e-3
//   hidden states          0.9999863 / 5.2e-3    0.9999999 / 4.1e-4  0.9999864 / 5.2e-3
//                          (corr / rel RMSE, over all 7 codebooks at once)
//
// Read the columns together: our error against the DUMP is, to three digits, the
// dump's OWN error against float32 — i.e. we are reproducing the reference to
// ~6e-4 and the remaining 4-5e-3 is the capture's bf16. Per codebook the spread
// is 3.0e-3..7.4e-3 against the dump with no trend across c1..c7, and the
// CFG-guided argmax agrees with the reference on all 7 codebooks.
//
// Speed: 9.2 ms median per frame (40 runs, warm), = 1.3 ms per codebook step.
// At the model's 25 fps that is 231 ms of wall clock per second of audio for
// this stage alone. The 7 passes each stream the full 0.6 B f16 weights
// (~4.2 GB/frame => ~456 GB/s effective on a card with ~1.8 TB/s), so the gap is
// launch/occupancy overhead on tiny (<=8-token, 2-row) matmuls, not bandwidth.
// The obvious lever if the AR loop needs it is fewer, larger kernels — not a KV
// cache, which would cut positions but not the seven weight sweeps.

#include "mm3-model.h"
#include "mm3-sample.h"

#include "backend.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

// 4 blocks x ~30 nodes + embedding/head plumbing measures ~150; loose on purpose.
#define MM3_DEPTH_MAX_NODES 512
// Hard cap on acoustic codebooks, so the per-step arrays can be fixed size.
#define MM3_DEPTH_MAX_STEPS 16

// One cached graph: codebook step c (1-based), sequence length c+1, head c-1.
struct MM3DepthStep {
    ggml_backend_sched_t sched = nullptr;
    ggml_context *       gctx  = nullptr;
    uint8_t *            gbuf  = nullptr;
    ggml_cgraph *        graph = nullptr;

    ggml_tensor * in_hidden = nullptr;  // [H, 1, 2] F32 — LM last_hidden, row 0 = cond
    ggml_tensor * in_sem    = nullptr;  // [1]       I32 — LM vocab id of the semantic code
    ggml_tensor * in_ac     = nullptr;  // [c-1]     I32 — depth.audio_embd row indices (null at c=1)
    ggml_tensor * out_logit = nullptr;  // [V, 1, 2] F32 — pre-CFG logits, row 0 = cond
    ggml_tensor * out_hid   = nullptr;  // [H, 1, 2] F32 — last-position hidden, row 0 = cond

    ggml_tensor * mask = nullptr;  // [S, S] F32 causal mask, lives in the prep buffer
    int64_t       S    = 0;
    size_t        compute_bytes = 0;
    int           n_nodes       = 0;
};

struct MM3DepthGraph {
    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    bool           backend_ref = false;
    WeightCtx      prep        = {};  // the seven causal masks

    // Identity of the weights this prep was derived from. BOTH buffers matter:
    // the semantic embedding is read from the LM file, everything else from the
    // synth file, so either being reloaded invalidates the graphs.
    const void * synth_token = nullptr;
    const void * lm_token    = nullptr;

    MM3DepthStep step[MM3_DEPTH_MAX_STEPS];
    int          n_steps = 0;
};

// One frame's worth of depth-decoder output.
struct MM3DepthFrame {
    int32_t            codes[MM3_DEPTH_MAX_STEPS] = { 0 };  // sampled c1..c7
    std::vector<float> hiddens;                             // [NC, H], CONDITIONAL row only
    std::vector<float> logits_cond;                         // [NC, V] pre-CFG
    std::vector<float> logits_uncond;                       // [NC, V] pre-CFG
    int                n_codes = 0;
    double             ms      = 0.0;
};

// ── Free ────────────────────────────────────────────────────────────────────

static void mm3_depth_free_step(MM3DepthStep * s) {
    if (s->gctx) {
        if (s->sched) {
            ggml_backend_sched_reset(s->sched);
        }
        ggml_free(s->gctx);
        free(s->gbuf);
    }
    if (s->sched) {
        ggml_backend_sched_free(s->sched);
    }
    *s = MM3DepthStep{};
}

static void mm3_depth_free(MM3DepthGraph * g) {
    for (int i = 0; i < MM3_DEPTH_MAX_STEPS; i++) {
        mm3_depth_free_step(&g->step[i]);
    }
    wctx_free(&g->prep);
    g->n_steps     = 0;
    g->synth_token = nullptr;
    g->lm_token    = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

// ── Graph pieces ────────────────────────────────────────────────────────────

// RMSNorm with a learned gain over ne0.
static ggml_tensor * mm3_depth_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// One decoder block. h [H, S, 2] -> [H, S, 2]. Causal, no RoPE.
static ggml_tensor * mm3_depth_block(ggml_context * ctx, const MM3DepthConfig & c, const MM3DepthLayer & w,
                                     ggml_tensor * h, ggml_tensor * mask) {
    const int64_t H  = (int64_t) c.embedding_length;
    const int64_t D  = (int64_t) c.head_dim;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t S  = h->ne[1];
    const int64_t B  = h->ne[2];

    ggml_tensor * n = mm3_depth_norm(ctx, h, w.attn_norm, c.rms_eps);

    ggml_tensor * q = ggml_mul_mat(ctx, w.attn_q, n);  // [H, S, B]
    ggml_tensor * k = ggml_mul_mat(ctx, w.attn_k, n);
    ggml_tensor * v = ggml_mul_mat(ctx, w.attn_v, n);

    // [H, S, B] -> [D, Nh, S, B] -> [D, S, Nh, B]
    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, D, Nh, S, B), 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, D, Nh, S, B), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, D, Nh, S, B), 0, 2, 1, 3));

    const float   scale  = 1.0f / sqrtf((float) D);
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);                          // [S_k, S_q, Nh, B]
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);  // causal

    ggml_tensor * vt   = ggml_cont(ctx, ggml_transpose(ctx, v));  // [S, D, Nh, B]
    ggml_tensor * attn = ggml_mul_mat(ctx, vt, scores);           // [D, S_q, Nh, B]
    attn               = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));  // [D, Nh, S, B]
    attn               = ggml_reshape_3d(ctx, attn, H, S, B);

    h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.attn_output, attn));

    // SwiGLU, llama shape: down(silu(gate) * up).
    ggml_tensor * n2   = mm3_depth_norm(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, w.ffn_gate, n2));
    ggml_tensor * up   = ggml_mul_mat(ctx, w.ffn_up, n2);
    ggml_tensor * y    = ggml_mul_mat(ctx, w.ffn_down, ggml_mul(ctx, gate, up));
    return ggml_add(ctx, h, y);
}

// Build one step graph. `cb` is the 1-based codebook index (1..NC).
static bool mm3_depth_build_step(const MM3Model & m, MM3DepthGraph * g, int cb, std::string * err) {
    const MM3DepthConfig &  c  = m.synth_cfg.depth;
    const MM3DepthWeights & w  = m.synth.depth;
    const int64_t           H  = (int64_t) c.embedding_length;
    const int64_t           S  = cb + 1;  // hidden + semantic + (cb-1) acoustic
    MM3DepthStep *          s  = &g->step[cb - 1];

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (MM3_DEPTH_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_DEPTH_MAX_NODES, false);
    s->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!s->gbuf) {
        if (err) {
            *err = "out of host memory allocating the depth graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, s->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(s->gbuf);
        s->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the depth graph context";
        }
        return false;
    }

    s->in_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, 2);
    ggml_set_name(s->in_hidden, "mm3_depth_lm_hidden");
    ggml_set_input(s->in_hidden);

    s->in_sem = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(s->in_sem, "mm3_depth_semantic_id");
    ggml_set_input(s->in_sem);

    if (cb > 1) {
        s->in_ac = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cb - 1);
        ggml_set_name(s->in_ac, "mm3_depth_acoustic_rows");
        ggml_set_input(s->in_ac);
    }

    // Token 0: the LM hidden, per CFG row. Tokens 1..: the shared embeddings —
    // identical in both rows because the sampled codes are shared (note 5).
    ggml_tensor * tok0 = ggml_mul_mat(ctx, w.proj, s->in_hidden);  // [H, 1, 2]

    ggml_tensor * shared = ggml_get_rows(ctx, m.lm.token_embd, s->in_sem);  // [H, 1] — LM table (note 1)
    if (cb > 1) {
        ggml_tensor * ac = ggml_get_rows(ctx, w.audio_embd, s->in_ac);  // [H, cb-1]
        shared           = ggml_concat(ctx, shared, ac, 1);             // [H, cb]
    }
    shared = ggml_mul_mat(ctx, w.proj, shared);          // [H, cb]
    shared = ggml_concat(ctx, shared, shared, 2);        // [H, cb, 2] — same for both rows
    ggml_tensor * seq = ggml_concat(ctx, tok0, shared, 1);  // [H, S, 2]

    // Learned absolute positions 0..S-1, added after the projection (note 4).
    ggml_tensor * pos = ggml_view_2d(ctx, w.pos_embd, H, S, w.pos_embd->nb[1], 0);
    ggml_tensor * h   = ggml_add(ctx, seq, pos);

    for (size_t i = 0; i < w.blk.size(); i++) {
        h = mm3_depth_block(ctx, c, w.blk[i], h, s->mask);
    }
    h = mm3_depth_norm(ctx, h, w.output_norm, c.rms_eps);  // [H, S, 2]

    // Last position only — that is what the head reads and what the flow stage
    // collects.
    ggml_tensor * last = ggml_cont(
        ctx, ggml_view_3d(ctx, h, H, 1, 2, h->nb[1], h->nb[2], (size_t) (S - 1) * h->nb[1]));  // [H, 1, 2]

    ggml_tensor * logits = ggml_mul_mat(ctx, w.head[(size_t) (cb - 1)], last);  // [V, 1, 2]

    s->out_hid   = last;
    s->out_logit = logits;
    ggml_set_name(s->out_hid, "mm3_depth_hidden");
    ggml_set_name(s->out_logit, "mm3_depth_logits");
    ggml_set_output(s->out_hid);
    ggml_set_output(s->out_logit);

    s->graph = ggml_new_graph_custom(ctx, MM3_DEPTH_MAX_NODES, false);
    ggml_build_forward_expand(s->graph, s->out_hid);
    ggml_build_forward_expand(s->graph, s->out_logit);

    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, s->graph)) {
        ggml_free(ctx);
        free(s->gbuf);
        s->gbuf  = nullptr;
        s->graph = nullptr;
        if (err) {
            *err = "depth graph allocation failed (out of VRAM?) for codebook " + std::to_string(cb);
        }
        return false;
    }

    s->gctx          = ctx;
    s->S             = S;
    s->n_nodes       = ggml_graph_n_nodes(s->graph);
    s->compute_bytes = ggml_backend_sched_get_buffer_size(s->sched, g->backend);
    return true;
}

// ── Prep ────────────────────────────────────────────────────────────────────

// Acquire a backend, build the causal masks, and build all seven step graphs.
// Cheap after the first call: returns immediately when both model buffers are
// the ones the current graphs were derived from.
static bool mm3_depth_prepare(const MM3Model & m, MM3DepthGraph * g, std::string * err) {
    // Staged residency: depth runs inside the AR loop, so it needs the LM and
    // depth buffers — never the flow stack, which is not resident yet.
    if (!m.lm_resident || !m.depth_resident) {
        if (err) {
            *err = "MiniMax-Music3 is not warm (POST /mm3/warm first)";
        }
        return false;
    }
    const void * st = (const void *) m.wctx_depth.buffer;
    const void * lt = (const void *) m.wctx_lm.buffer;
    if (g->synth_token == st && g->lm_token == lt && g->n_steps > 0) {
        return true;
    }
    mm3_depth_free(g);

    const MM3DepthConfig & c = m.synth_cfg.depth;
    if (c.block_count == 0 || c.embedding_length == 0 || c.head_count == 0 || c.head_dim == 0) {
        if (err) {
            *err = "depth config is empty — mm3.depth.* KVs missing from the synth GGUF";
        }
        return false;
    }
    if (c.rope) {
        if (err) {
            *err = "mm3.depth.rope is true; this port implements the documented learned-absolute-position variant only";
        }
        return false;
    }
    if (!c.causal) {
        if (err) {
            *err = "mm3.depth.causal is false; this port implements the documented causal variant only";
        }
        return false;
    }
    const int NC = (int) c.num_codebooks - 1;
    if (NC < 1 || NC >= MM3_DEPTH_MAX_STEPS) {
        if (err) {
            *err = "mm3.depth.num_codebooks - 1 = " + std::to_string(NC) + " is outside 1.." +
                   std::to_string(MM3_DEPTH_MAX_STEPS - 1);
        }
        return false;
    }
    if ((int64_t) c.max_position < (int64_t) NC + 1) {
        if (err) {
            *err = "depth.pos_embd has " + std::to_string(c.max_position) + " rows, need " + std::to_string(NC + 1);
        }
        return false;
    }
    if (!m.lm.token_embd) {
        if (err) {
            *err = "the LM token embedding is not resident; the depth decoder needs it for the semantic code";
        }
        return false;
    }
    // pos_embd is added straight onto F32 activations, so it must be F32 — the
    // layout doc pins it there (§7) precisely because it lands in a 4-layer stack
    // with no normalisation before the first residual. A converter change that
    // quantised it must fail here, not silently take a different ggml_add path.
    if (!m.synth.depth.pos_embd || m.synth.depth.pos_embd->type != GGML_TYPE_F32) {
        if (err) {
            *err = "depth.pos_embd.weight is not F32; the layout contract pins it to F32";
        }
        return false;
    }

    BackendPair bp = backend_init("MM3-Depth");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    // One causal mask per step. F32 so soft_max_ext takes it directly; tiny
    // (2+3+...+8 = 35 rows total at the reference geometry).
    wctx_init(&g->prep, NC);
    for (int cb = 1; cb <= NC; cb++) {
        const int64_t S = cb + 1;
        auto          d = std::make_unique<float[]>((size_t) (S * S));
        for (int64_t q = 0; q < S; q++) {
            for (int64_t k = 0; k < S; k++) {
                d[(size_t) (k + q * S)] = k <= q ? 0.0f : -INFINITY;
            }
        }
        ggml_tensor * t = ggml_new_tensor_2d(g->prep.ctx, GGML_TYPE_F32, S, S);
        char          nm[64];
        snprintf(nm, sizeof(nm), "mm3.depth.causal_mask.%d", cb);
        ggml_set_name(t, nm);
        g->prep.pending.push_back({ t, d.get(), (size_t) (S * S) * sizeof(float), 0 });
        g->prep.staging.push_back(std::move(d));
        g->step[cb - 1].mask = t;
    }
    if (!wctx_alloc(&g->prep, g->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the depth causal masks";
        }
        mm3_depth_free(g);
        return false;
    }

    for (int cb = 1; cb <= NC; cb++) {
        g->step[cb - 1].sched = backend_sched_new(bp, MM3_DEPTH_MAX_NODES * 2);
        if (!mm3_depth_build_step(m, g, cb, err)) {
            mm3_depth_free(g);
            return false;
        }
    }
    g->n_steps     = NC;
    g->synth_token = st;
    g->lm_token    = lt;

    size_t total = 0;
    for (int i = 0; i < NC; i++) {
        total += g->step[i].compute_bytes;
    }
    fprintf(stderr,
            "[MM3-Depth] Prepared: %u blocks, %u heads x %u, %d step graphs (S=2..%d, %d nodes each), "
            "compute buffers %.1f MB\n",
            c.block_count, c.head_count, c.head_dim, NC, NC + 1, g->step[NC - 1].n_nodes,
            (double) total / (1024.0 * 1024.0));
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────────

static MM3DepthGraph g_mm3_depth;

// Decode one frame's seven residual codes.
//
//   lm_hidden_cond / lm_hidden_uncond
//                   4096 floats each — the global LM's last_hidden_state for the
//                   conditional and unconditional CFG rows of THIS AR iteration.
//   semantic_code   the frame's codebook-0 code, in [0, semantic_vocab_size).
//                   NOT the raw token id: the LM vocab offset is added here.
//   forced_codes    optional [7]; when non-null the sampler is bypassed and these
//                   codes are fed back into the sequence instead. This is the
//                   parity mode — it makes the graph see exactly the token
//                   sequence the reference saw, so the logits are comparable even
//                   though the reference's multinomial draw is not reproducible.
//   out             codes, the 7 CONDITIONAL-row hidden states (the flow stage's
//                   conditioning), and both pre-CFG logit rows per codebook.
//   rng             optional; when non-null each code is drawn by the reference's
//                   top-k multinomial recipe (mm3-sample.h) instead of argmax.
//                   Ignored under `forced_codes`.
//   top_k           the checkpoint's `mm3.ar.top_k`; <= 0 means "no cap".
//
// Not thread-safe: the caller serialises (mm3-server.h holds g_mm3_mutex).
static bool mm3_depth_decode_frame(const MM3Model & m, const float * lm_hidden_cond, const float * lm_hidden_uncond,
                                   int32_t semantic_code, const int32_t * forced_codes, MM3DepthFrame * out,
                                   std::string * err = nullptr, std::mt19937_64 * rng = nullptr, int top_k = 0) {
    if (!mm3_depth_prepare(m, &g_mm3_depth, err)) {
        return false;
    }
    const MM3DepthConfig & c  = m.synth_cfg.depth;
    const MM3LmConfig &    lc = m.lm_cfg;
    const int64_t          H  = (int64_t) c.embedding_length;
    const int64_t          V  = (int64_t) c.audio_vocab_size;
    const int              NC = g_mm3_depth.n_steps;

    if (semantic_code < 0 || (uint32_t) semantic_code >= lc.semantic_vocab_size) {
        if (err) {
            *err = "semantic code " + std::to_string(semantic_code) + " is outside [0, " +
                   std::to_string(lc.semantic_vocab_size) + ")";
        }
        return false;
    }
    if (forced_codes) {
        for (int i = 0; i < NC; i++) {
            if (forced_codes[i] < 0 || (int64_t) forced_codes[i] >= V) {
                if (err) {
                    *err = "forced code " + std::to_string(forced_codes[i]) + " for codebook " + std::to_string(i + 1) +
                           " is outside [0, " + std::to_string(V) + ")";
                }
                return false;
            }
        }
    }

    out->n_codes = NC;
    out->hiddens.assign((size_t) (NC * H), 0.0f);
    out->logits_cond.assign((size_t) (NC * V), 0.0f);
    out->logits_uncond.assign((size_t) (NC * V), 0.0f);

    const int32_t      sem_id  = semantic_code + (int32_t) lc.semantic_vocab_offset;
    const float        cfg     = lc.ar_cfg_scale > 0.0f ? lc.ar_cfg_scale : 1.5f;
    std::vector<int32_t> ac_rows((size_t) NC, 0);
    std::vector<float>   logit_buf((size_t) (V * 2));
    std::vector<float>   hid_buf((size_t) (H * 2));
    std::vector<float>   samp_scratch;

    // MM3_DEPTH_PROF=1 — phase-level timing across frames, printed every 100
    // frames. Diagnostic for the "9.3 ms/frame is launch-bound" question: it
    // separates upload / graph compute / readback / host sampling so the fix
    // (CUDA graphs? fewer syncs? fused readback?) targets the real cost.
    static const bool depth_prof = [] {
        const char * e = std::getenv("MM3_DEPTH_PROF");
        return e && e[0] && e[0] != '0';
    }();
    static double prof_up = 0.0, prof_comp = 0.0, prof_down = 0.0, prof_host = 0.0;
    static int    prof_frames = 0;
    const auto    now         = [] { return std::chrono::steady_clock::now(); };
    const auto    ms_since    = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };

    const auto t0 = std::chrono::steady_clock::now();
    for (int cb = 1; cb <= NC; cb++) {
        MM3DepthStep * s = &g_mm3_depth.step[cb - 1];

        const auto tu = now();
        ggml_backend_tensor_set(s->in_hidden, lm_hidden_cond, 0, (size_t) H * sizeof(float));
        ggml_backend_tensor_set(s->in_hidden, lm_hidden_uncond, (size_t) H * sizeof(float),
                                (size_t) H * sizeof(float));
        ggml_backend_tensor_set(s->in_sem, &sem_id, 0, sizeof(int32_t));
        if (s->in_ac) {
            ggml_backend_tensor_set(s->in_ac, ac_rows.data(), 0, (size_t) (cb - 1) * sizeof(int32_t));
        }
        if (depth_prof) {
            prof_up += ms_since(tu);
        }

        const auto tc = now();
        if (ggml_backend_sched_graph_compute(s->sched, s->graph) != GGML_STATUS_SUCCESS) {
            if (err) {
                *err = "depth graph compute failed at codebook " + std::to_string(cb);
            }
            return false;
        }
        if (depth_prof) {
            prof_comp += ms_since(tc);
        }

        const auto td = now();
        ggml_backend_tensor_get(s->out_logit, logit_buf.data(), 0, (size_t) (V * 2) * sizeof(float));
        ggml_backend_tensor_get(s->out_hid, hid_buf.data(), 0, (size_t) (H * 2) * sizeof(float));
        if (depth_prof) {
            prof_down += ms_since(td);
        }
        const auto th = now();

        float * lc_row = out->logits_cond.data() + (size_t) ((cb - 1) * V);
        float * lu_row = out->logits_uncond.data() + (size_t) ((cb - 1) * V);
        memcpy(lc_row, logit_buf.data(), (size_t) V * sizeof(float));
        memcpy(lu_row, logit_buf.data() + V, (size_t) V * sizeof(float));
        // Conditional row only (note 6).
        memcpy(out->hiddens.data() + (size_t) ((cb - 1) * H), hid_buf.data(), (size_t) H * sizeof(float));

        int32_t code;
        if (forced_codes) {
            code = forced_codes[cb - 1];
        } else {
            // CFG-guided distribution (design note D), reusing the logit buffer's
            // second half — the uncond row is not needed once combined.
            float * guided = logit_buf.data() + V;
            for (int64_t i = 0; i < V; i++) {
                const float u = lu_row[i];
                guided[i]     = u + cfg * (lc_row[i] - u);
            }
            if (rng) {
                code = (int32_t) mm3_sample_top_k(guided, V, top_k, *rng, &samp_scratch);
            } else {
                int32_t best_i = 0;
                float   best_v = -INFINITY;
                for (int64_t i = 0; i < V; i++) {
                    if (guided[i] > best_v) {
                        best_v = guided[i];
                        best_i = (int32_t) i;
                    }
                }
                code = best_i;
            }
        }
        out->codes[cb - 1] = code;
        // Codebook cb's feedback embedding lives at row code + (cb-1)*V (note 2).
        ac_rows[(size_t) (cb - 1)] = code + (int32_t) ((cb - 1) * V);
        if (depth_prof) {
            prof_host += ms_since(th);
        }
    }
    out->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (depth_prof && ++prof_frames % 100 == 0) {
        fprintf(stderr,
                "[MM3-Depth] prof over %d frames: upload %.2f + compute %.2f + readback %.2f + host %.2f "
                "= %.2f ms/frame\n",
                prof_frames, prof_up / prof_frames, prof_comp / prof_frames, prof_down / prof_frames,
                prof_host / prof_frames,
                (prof_up + prof_comp + prof_down + prof_host) / prof_frames);
    }
    return true;
}
