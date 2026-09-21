#pragma once
// yue2/yue2-nar-graph.h — YuE2 NAR (Mixture-of-Transformers) flow-matching
// block + the 32-step midpoint ODE solver. Milestone M5, per
// docs/plans/yue2/06-engine-port-plan.md §5 and
// docs/plans/yue2/03-reference-numerics.md §3.
//
// HOT-Step file (does not exist upstream, no acestep.cpp analog). Builds on
// yue2-model.h (weights) and yue2-lm-graph.h (yue2_ar_block's math skeleton,
// yue2_lm_rms, yue2_lm_attn_f32, yue2_lm_use_flash, and — the single biggest
// reuse win here — Yue2ArKvCache/yue2_ar_prefill for the per-chunk AR-prefix
// KV, per the engine-port-plan's own explicit recommendation).
//
// ── What this file is ────────────────────────────────────────────────────
//
// One `Yue2NarChunk` = one `CachedNAR` instance (nar.py) = one fresh causal
// AR-prefix prefill (this chunk's prompt+codec-ids+MUSIC_END, via
// yue2_ar_prefill — same math as AR decode, NAR never reuses the semantic
// stage's own KV cache) plus 32 midpoint-solver steps (64 velocity() network
// evaluations) over that chunk's own noise slice. `yue2_nar_velocity()` is
// the per-call forward (input embedding -> 28 NAR blocks, each attending over
// [cached AR prefix ++ fresh NAR K/V] -> shared final norm -> llm2vae); the
// host-side ODE loop (`yue2_nar_solve_midpoint`) calls it exactly twice per
// step, matching mm3-dit-graph.h's own "Euler update happens on the host,
// only the network forward is a compiled graph" precedent (mm3-dit-graph.h
// note E) — the update arithmetic itself (`state - v*dt`) is a few thousand
// floats, cheap to do outside ggml.
//
// ── Precision policy — same as yue2-lm-graph.h, not the reference's ────────
//
// The reference keeps the ODE `state` in native BF16 for the entire 32-step
// solve (03-reference-numerics.md §3.8) and only upcasts the FINAL result to
// F32. This port instead runs F32 activations throughout (weights stay BF16,
// NATIVE-policy per yue2-model.h) — identical posture to yue2-lm-graph.h's
// own file header ("F32 activations ... strictly MORE precise than the thing
// this is validated against, never less"). §9's already-calibrated gates
// (YUE2_LOGIT_GATE-style k-layer-accumulation formulas, reused verbatim in
// yue2-probe.cpp's --nar-parity) already account for exactly this BF16-vs-F32
// gap for the AR stage; the same reasoning is assumed to carry over to the
// NAR stage without re-deriving it (rough edge, noted in the milestone
// report — a from-scratch reference-vs-reference recalibration for the NAR
// stage specifically was not run this pass).
//
// ── The two position schemes (03-reference-numerics.md §3.6, easy to
//    conflate) ───────────────────────────────────────────────────────────
//
// Every NAR frame carries TWO independent position numbers: a GLOBAL RoPE
// position (`ar_length + local_index`, continuing the chunk's own AR-prefix
// length — used only for `ggml_rope_ext`'s `positions` input) and a LOCAL
// AudioPositionEmbedding index (`local_index` alone, restarting at 0 every
// chunk — used only for the `ggml_get_rows` gather into `latent_pos_embed`).
// Both are built and fed as SEPARATE input tensors below; nothing here
// aliases one for the other.
//
// ── AR-prefix KV: cached, F16, never re-rotated at the concat ────────────
//
// `Yue2ArKvCache` (yue2-lm-graph.h) already stores K/V post-project_qkv (i.e.
// post-per-head-RMSNorm, post-RoPE) at absolute positions [0, ar_length) —
// exactly the "AR-prefix cache" shape 03-reference-numerics.md §3.1 demands.
// Each NAR block below concatenates that persistent, already-rotated F16
// cache with THIS call's freshly-computed, freshly-rotated NAR K/V along the
// sequence axis (`ggml_concat`, dim=1) — no RoPE is re-applied to either
// side at the concat point, matching the reference's own
// `torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))` (nar.py:163) exactly.
//
// ── Attention pattern: full bidirectional ─────────────────────────────────
//
// Every NAR query attends over the ENTIRE concatenated [AR-prefix ++ fresh
// NAR] key/value span with no restriction whatsoever (03-reference-numerics.md
// §3.3: NAR sees all of AR, and NAR sees all of NAR, bidirectionally; AR
// never sees NAR at all, which is automatically true here since the AR cache
// was written before any NAR computation exists to poison it). This resolves
// cleanly to `ggml_flash_attn_ext`/`yue2_lm_attn_f32` with a NULL mask by
// default. The opt-in padded GQA path masks only alignment rows, never real
// tokens.
//
// ── What's NOT here (rough edges, listed rather than polished — task rule) ─
//
// - Chunking (`chunk_ranges`, 03 §3.7) and the whole-song noise draw-then-
//   slice (03 §3.10) are exercised by the probe CLI, not implemented as a
//   standalone public function in this file — every v1.4 fixture (off-a,
//   full-a, full-cfg) captures exactly ONE chunk, so the multi-chunk driving
//   loop lives in yue2-probe.cpp's --nar-parity handler (still generic over
//   N chunks, just not yet exercised against a multi-chunk fixture).
// - The NAR-side `nar_cond_end>0` "codec dropout" attention-visibility
//   restriction (03 §3.1/§3.3) is unreachable from the shipped pipeline and
//   is not implemented — `visible_length == ar_length` unconditionally here,
//   matching every real fixture and the reference's own inference-time
//   behavior.

#include "yue2-imatrix.h"
#include "yue2-lm-graph.h"
#include "yue2-model.h"

#include "backend.h"
#include "ggml.h"
#include "hot-step-build-flags.h"
#include "lua-plugin-registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

// 28 layers x ~25 nodes (concat/cast/rope/norm/matmul per layer, no
// adapter/CFG/cache-write branching) + input-embedding plumbing + output
// head. Same loose budget posture as YUE2_LM_MAX_NODES.
#define YUE2_NAR_MAX_NODES 4096

// ── Per-chunk state: the AR-prefix KV cache, reused across all 64 velocity
//    evaluations of that chunk's ODE solve ──────────────────────────────────

struct Yue2NarVelocityGraph {
    uint8_t * gbuf = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_tensor * in_x_nar = nullptr;
    ggml_tensor * in_time_feat = nullptr;
    ggml_tensor * in_local_idx = nullptr;
    ggml_tensor * in_rope_pos = nullptr;
    ggml_tensor * in_content_idx = nullptr;
    ggml_tensor * in_attn_mask = nullptr;
    ggml_tensor * in_attn_zeros = nullptr;
    bool attn_constants_uploaded = false;
    ggml_tensor * x_embed_tap = nullptr;
    ggml_tensor * velocity_out = nullptr;
};

static void yue2_nar_velocity_graph_free(Yue2NarVelocityGraph * g) {
    if (g->sched) ggml_backend_sched_free(g->sched);
    if (g->ctx) ggml_free(g->ctx);
    free(g->gbuf);
    *g = {};
}

struct Yue2NarChunk {
    int64_t       ar_length  = 0;  // this chunk's AR-prefix length (prefix+codec+MUSIC_END)
    int64_t       chunk_len  = 0;  // content latent frames in this chunk
    int64_t       nar_length = 0;  // chunk_len + 2 (the two zero-padded boundary rows)
    Yue2ArKvCache ar_cache;        // OWNED prefix cache (yue2_nar_chunk_init: trainer/probe path)
    // The cache the velocity graph actually reads and which set of it holds
    // this chunk's AR prefix: &ar_cache after yue2_nar_chunk_init, or a
    // pipeline-owned cache after yue2_nar_chunk_bind (doc 30 #2 — the
    // semantic stage's rows are reused, only the missing tail is forwarded).
    Yue2ArKvCache * cache = nullptr;
    int64_t         set   = 0;
    // Noise variations solved in one graph (doc 30 #6): the latent block
    // carries M on the batch axis, the AR prefix K/V is repeated M times
    // before the concat. state/noise/velocity buffers are [M][chunk_len][64].
    int64_t         n_var = 1;
    mutable Yue2NarVelocityGraph velocity_graph; // serial ODE calls share one shape
};

// Builds this chunk's AR-prefix KV via a fresh causal prefill (yue2_ar_prefill,
// yue2-lm-graph.h — same math as the semantic-stage AR decode, deliberately
// NOT sharing that stage's own KV cache, per 03-reference-numerics.md §3.1).
// No logits/hidden taps are requested (cheap: skips the lm_head matmul and
// the 30-way hidden gather entirely, since NAR needs neither).
static bool yue2_nar_chunk_init(const Yue2Model & m, const std::vector<int32_t> & ar_prefix_ids, int64_t chunk_len,
                                Yue2NarChunk * out, std::string * err) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    if (ar_prefix_ids.empty() || chunk_len <= 0) {
        if (err) {
            *err = "yue2_nar_chunk_init: ar_prefix_ids must be non-empty and chunk_len > 0";
        }
        return false;
    }
    out->ar_length  = (int64_t) ar_prefix_ids.size();
    out->chunk_len  = chunk_len;
    out->nar_length = chunk_len + 2;
    out->cache      = &out->ar_cache;
    out->set        = 0;
    out->n_var      = 1;

    if (!yue2_ar_kv_cache_alloc(m, out->ar_length, &out->ar_cache, err)) {
        return false;
    }
    Yue2ArForwardResult dummy;
    if (!yue2_ar_prefill(m, out->ar_cache, ar_prefix_ids, {}, {}, &dummy, err)) {
        yue2_ar_kv_cache_free(&out->ar_cache);
        return false;
    }
    if (out->ar_cache.filled[0] != out->ar_length) {
        if (err) {
            *err = "yue2_nar_chunk_init: prefill did not fill the whole AR-prefix cache (internal inconsistency)";
        }
        yue2_ar_kv_cache_free(&out->ar_cache);
        return false;
    }
    return true;
}

// Binds a chunk to set `set` of a caller-owned cache whose rows [0,filled)
// already hold this chunk's AR prefix (prompt + codec slice + MUSIC_END, put
// there by yue2_ar_prefill against that set). Nothing is forwarded here.
static bool yue2_nar_chunk_bind(Yue2ArKvCache & cache, int64_t set, int64_t chunk_len, int64_t n_var,
                                Yue2NarChunk * out, std::string * err) {
    if (set < 0 || set >= cache.n_sets || chunk_len <= 0 || n_var <= 0 || cache.filled[(size_t) set] <= 0) {
        if (err) *err = "yue2_nar_chunk_bind: bad set, chunk_len, n_var or an empty set";
        return false;
    }
    yue2_nar_velocity_graph_free(&out->velocity_graph);
    out->ar_length  = cache.filled[(size_t) set];
    out->chunk_len  = chunk_len;
    out->nar_length = chunk_len + 2;
    out->cache      = &cache;
    out->set        = set;
    out->n_var      = n_var;
    return true;
}

static void yue2_nar_chunk_free(Yue2NarChunk * c) {
    yue2_nar_velocity_graph_free(&c->velocity_graph);
    yue2_ar_kv_cache_free(&c->ar_cache);  // no-op for a bound chunk (nothing allocated)
    c->cache = nullptr;
    c->set   = 0;
    c->n_var = 1;
    c->ar_length = c->chunk_len = c->nar_length = 0;
}

// ── Scalar time math (03-reference-numerics.md §3.5) ────────────────────────

// torch.logit(t).clamp(-20,20), computed at double precision on the host —
// the ONE place the reference deliberately runs higher than its own BF16
// default, specifically to avoid catastrophic cancellation/inf at t->{0,1}.
static double yue2_nar_logit_clamped(double t) {
    const double x = std::log(t / (1.0 - t));  // may be +-inf at t=1/t=0; IEEE-safe, clamp handles it
    if (x < -20.0) {
        return -20.0;
    }
    if (x > 20.0) {
        return 20.0;
    }
    return x;
}

// _shift_t_value: shift*sigmoid(t)/(1+(shift-1)*sigmoid(t)) — general formula,
// never hardcode the shift=1 sigmoid collapse (03 §3.5). Computed in double
// here rather than simulating the reference's BF16-sigmoid rounding step;
// the numerics doc measured 64/64 bitwise agreement between the general
// formula and plain sigmoid at this checkpoint's shift=1.0, so the gap this
// simplification introduces is expected to be at or below the "single BF16
// read" rounding bound already budgeted elsewhere — not independently
// re-measured this pass (rough edge).
static double yue2_nar_shift_t(double raw_t, double shift) {
    const double t_sig = 1.0 / (1.0 + std::exp(-raw_t));
    return shift * t_sig / (1.0 + (shift - 1.0) * t_sig);
}

// TimestepEmbedder's sinusoid features (03 §3.4): freqs computed in fp32-
// equivalent double precision, cos-block-then-sin-block layout (NOT
// interleaved — contrast AudioPositionEmbedding's interleaved layout below).
static void yue2_nar_time_features(double shifted_t, std::vector<float> * out /* size 256 */) {
    out->assign(256, 0.0f);
    const double ln10000 = std::log(10000.0);
    for (int i = 0; i < 128; i++) {
        const double freq = std::exp(-ln10000 * (double) i / 128.0);
        const double arg  = shifted_t * freq;
        (*out)[(size_t) i]       = (float) std::cos(arg);
        (*out)[(size_t) (128 + i)] = (float) std::sin(arg);
    }
}

// Host-computed RoPE angle table at this chunk's GLOBAL NAR positions
// [ar_length, ar_length+nar_len) — standalone verification helper for
// --nar-parity's nar_cos.bin/nar_sin.bin, independent of the ggml_rope_ext
// call inside yue2_nar_velocity (which computes the identical formula
// in-graph from the same position ids; this lets the parity tool check the
// formula/positions assumption before trusting the full forward).
static void yue2_nar_rope_table(const Yue2LmConfig & c, int64_t ar_length, int64_t nar_len,
                                std::vector<float> * cos_out, std::vector<float> * sin_out) {
    const int64_t half = (int64_t) c.key_length / 2;  // 64 freqs at head_dim=128
    cos_out->assign((size_t) (nar_len * half), 0.0f);
    sin_out->assign((size_t) (nar_len * half), 0.0f);
    for (int64_t t = 0; t < nar_len; t++) {
        const double pos = (double) (ar_length + t);
        for (int64_t i = 0; i < half; i++) {
            const double freq  = 1.0 / std::pow((double) c.rope_freq_base, (2.0 * (double) i) / (double) c.key_length);
            const double angle = pos * freq;
            (*cos_out)[(size_t) (t * half + i)] = (float) std::cos(angle);
            (*sin_out)[(size_t) (t * half + i)] = (float) std::sin(angle);
        }
    }
}

// Standalone AudioPositionEmbedding gather (`latent_pos_embed.pe[local_idx]`)
// — a tiny dedicated graph, reused both as a --nar-parity check against
// nar_pos_emb.bin and (conceptually) as the same op yue2_nar_velocity builds
// inline. `local_idx` is clamped to max_latent_frames-1 defensively (never
// actually reached at realistic chunk sizes).
static bool yue2_nar_pos_emb_lookup(const Yue2Model & m, int64_t nar_len, std::vector<float> * out,
                                    std::string * err) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    const Yue2LmConfig & c = m.lm_cfg;
    const int64_t         H = (int64_t) c.embedding_length;

    const size_t ctx_bytes = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(64, false);
    uint8_t *     gbuf     = (uint8_t *) malloc(ctx_bytes);
    if (!gbuf) {
        if (err) {
            *err = "out of host memory (nar_pos_emb_lookup)";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(gbuf);
        if (err) {
            *err = "ggml_init failed (nar_pos_emb_lookup)";
        }
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);

    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nar_len);
    ggml_set_input(idx);
    ggml_tensor * g = ggml_get_rows(ctx, m.lm.latent_pos_embed, idx);  // [H, nar_len], F32 (get_rows always widens)
    ggml_set_output(g);
    ggml_build_forward_expand(gf, g);

    BackendPair           bp    = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
    ggml_backend_sched_t sched = backend_sched_new(bp, 64);
    bool                 ok    = sched && ggml_backend_sched_alloc_graph(sched, gf);
    if (!ok) {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        ggml_free(ctx);
        free(gbuf);
        if (err) {
            *err = "nar_pos_emb_lookup graph allocation failed";
        }
        return false;
    }

    std::vector<int32_t> idx_host((size_t) nar_len);
    const int64_t         max_idx = (int64_t) c.max_latent_frames - 1;
    for (int64_t i = 0; i < nar_len; i++) {
        idx_host[(size_t) i] = (int32_t) std::min<int64_t>(i, max_idx);
    }
    ggml_backend_tensor_set(idx, idx_host.data(), 0, (size_t) nar_len * sizeof(int32_t));

    ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        out->assign((size_t) (nar_len * H), 0.0f);
        ggml_backend_tensor_get(g, out->data(), 0, (size_t) (nar_len * H) * sizeof(float));
    } else if (err) {
        *err = "nar_pos_emb_lookup graph compute failed";
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    free(gbuf);
    return ok;
}

// One NAR transformer block (blk.N.nar_* — the independently-trained NAR
// half, never blk.N.*'s AR weights). Attends over [cached AR-prefix K/V ++
// this call's fresh NAR K/V], fully bidirectional (see file header).
// h [H,nar_len] -> [H,nar_len].
static ggml_tensor * yue2_nar_block(ggml_context * ctx, const Yue2LmConfig & c, const Yue2NarLayer & w,
                                    ggml_tensor * h, ggml_tensor * rope_pos, ggml_tensor * ar_k, ggml_tensor * ar_v,
                                    bool use_flash, ggml_tensor * attn_mask = nullptr,
                                    ggml_tensor * attn_zeros = nullptr,
                                    const Yue2Model * cm = nullptr, int layer = -1) {
    const Yue2AitkLayerWeights * cw = cm && cm->convrot && layer >= 0 ?
        &cm->convrot->nar().layers[(size_t) layer] : nullptr;
    const int64_t H    = (int64_t) c.embedding_length;
    const int64_t D    = (int64_t) c.key_length;
    const int64_t Nh   = (int64_t) c.head_count;
    const int64_t Nkv  = (int64_t) c.head_count_kv;
    const int64_t T    = h->ne[1];  // nar_len
    const int64_t M    = h->ne[2];  // noise variations on the batch axis (1 outside a batched render)

    ggml_tensor * n = yue2_lm_rms(ctx, h, w.attn_norm, c.rms_eps);

    ggml_tensor * qkv = cw ? yue2_convrot_linear(ctx, cw->qkv, n) : nullptr;
    ggml_tensor * q0 = cw ? yue2_convrot_rows(ctx, qkv, 0, H) : ggml_mul_mat(ctx, w.attn_q, n);
    ggml_tensor * k0 = cw ? yue2_convrot_rows(ctx, qkv, H, D*Nkv) : ggml_mul_mat(ctx, w.attn_k, n);
    ggml_tensor * v0 = cw ? yue2_convrot_rows(ctx, qkv, H+D*Nkv, D*Nkv) : ggml_mul_mat(ctx, w.attn_v, n);
    if (cw) {
        q0 = yue2_convrot_adapt(ctx, cm, layer, true, YUE2_CR_Q, n, q0);
        k0 = yue2_convrot_adapt(ctx, cm, layer, true, YUE2_CR_K, n, k0);
        v0 = yue2_convrot_adapt(ctx, cm, layer, true, YUE2_CR_V, n, v0);
    }
    ggml_tensor * q = ggml_reshape_4d(ctx, q0, D, Nh, T, M);
    ggml_tensor * k = ggml_reshape_4d(ctx, k0, D, Nkv, T, M);
    ggml_tensor * v = ggml_reshape_4d(ctx, v0, D, Nkv, T, M);

    // Per-head QK RMSNorm before RoPE, no norm on V — identical convention to
    // the AR path's own project_qkv (same Attention class, separate weights).
    q = yue2_lm_rms(ctx, q, w.attn_q_norm, c.rms_eps);
    k = yue2_lm_rms(ctx, k, w.attn_k_norm, c.rms_eps);

    // GLOBAL RoPE positions (ar_length + local_index) — see file header on
    // why this must not be conflated with the pos-embedding's LOCAL index.
    q = ggml_rope_ext(ctx, q, rope_pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                       0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, rope_pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                       0.0f, 0.0f);

    // [D,Nkv,T,1] -> [D,T,Nkv,1], cast to F16 to match the persistent
    // AR-prefix cache's own dtype (ggml_concat requires identical types).
    ggml_tensor * k_w = ggml_cast(ctx, ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)), GGML_TYPE_F16);
    ggml_tensor * v_w = ggml_cast(ctx, ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)), GGML_TYPE_F16);

    // Concat along the sequence axis: AR-prefix K/V is ALREADY post-QK-norm,
    // post-RoPE (Yue2ArKvCache's own contract, yue2-lm-graph.h) — no
    // re-rotation happens here, matching nar.py:163 exactly (file header trap).
    // ar_k/ar_v are [D, ar_length, Nkv, 1]; with M variations the prefix is
    // repeated on the batch axis first (upstream 2d21090f does the same).
    if (M > 1) {
        ar_k = ggml_repeat_4d(ctx, ar_k, D, ar_k->ne[1], Nkv, M);
        ar_v = ggml_repeat_4d(ctx, ar_v, D, ar_v->ne[1], Nkv, M);
    }
    ggml_tensor * k_cat = ggml_concat(ctx, ar_k, k_w, 1);  // [D, ar_length+T, Nkv, M]
    ggml_tensor * v_cat = ggml_concat(ctx, ar_v, v_w, 1);

    if (attn_mask) {
        const int64_t pad = attn_mask->ne[0] - k_cat->ne[1];
        GGML_ASSERT(pad >= 0);
        if (pad > 0) {
            GGML_ASSERT(attn_zeros && attn_zeros->ne[1] == pad);
            ggml_tensor * zeros = M > 1 ? ggml_repeat_4d(ctx, attn_zeros, D, pad, Nkv, M) : attn_zeros;
            k_cat = ggml_concat(ctx, k_cat, zeros, 1);
            v_cat = ggml_concat(ctx, v_cat, zeros, 1);
        }
    }

    ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D,T,Nh,M]

    const float   scale = c.softmax_scale;
    ggml_tensor * attn;
    if (use_flash) {
        // The optional mask hides only alignment padding; real tokens remain bidirectional.
        attn = ggml_flash_attn_ext(ctx, q4, k_cat, v_cat, attn_mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = yue2_lm_attn_f32(ctx, q4, k_cat, v_cat, nullptr, scale);
    }
    attn = ggml_reshape_3d(ctx, attn, H, T, M);

    ggml_tensor * projected = cw ? yue2_convrot_linear(ctx, cw->output, attn) :
                                   ggml_mul_mat(ctx, w.attn_output, attn);
    if (cw) projected = yue2_convrot_adapt(ctx, cm, layer, true, YUE2_CR_O, attn, projected);
    h = ggml_add(ctx, h, projected);

    ggml_tensor * n2   = yue2_lm_rms(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gate_up = cw ? yue2_convrot_linear(ctx, cw->gate_up, n2) : nullptr;
    ggml_tensor * gate0 = cw ? yue2_convrot_rows(ctx, gate_up, 0, c.feed_forward_length) :
                              ggml_mul_mat(ctx, w.ffn_gate, n2);
    ggml_tensor * up = cw ? yue2_convrot_rows(ctx, gate_up, c.feed_forward_length, c.feed_forward_length) :
                            ggml_mul_mat(ctx, w.ffn_up, n2);
    if (cw) {
        gate0 = yue2_convrot_adapt(ctx, cm, layer, true, YUE2_CR_GATE, n2, gate0);
        up = yue2_convrot_adapt(ctx, cm, layer, true, YUE2_CR_UP, n2, up);
    }
    ggml_tensor * gate = ggml_silu(ctx, gate0);
    ggml_tensor * activated = ggml_mul(ctx, gate, up);
    ggml_tensor * down = cw ? yue2_convrot_linear(ctx, cw->down, activated) :
                              ggml_mul_mat(ctx, w.ffn_down, activated);
    if (cw) down = yue2_convrot_adapt(ctx, cm, layer, true, YUE2_CR_DOWN, activated, down);
    return ggml_add(ctx, h, down);
}

// ── velocity(): one full NAR forward, one ODE network evaluation ───────────

struct Yue2NarVelocityResult {
    std::vector<float> velocity;        // [chunk_len, 64], position-major (t*64+d)
    std::vector<float> input_embedding; // [nar_len, H], only filled if requested — the `x` tap before layer 0
};

// `state` is [chunk_len, 64] F32, position-major, UNPADDED content only (the
// two zero boundary rows are added on the host before upload — see file
// header on why this is done host-side rather than with an in-graph pad op).
// `raw_t` is the ALREADY logit-transformed, [-20,20]-clamped scalar time
// value (yue2_nar_logit_clamped's output) — this function applies
// _shift_t_value and the TimestepEmbedder's sinusoid features itself.
static bool yue2_nar_velocity(const Yue2Model & m, const Yue2NarChunk & chunk, const std::vector<float> & state,
                              double raw_t, bool want_input_embedding, Yue2NarVelocityResult * out,
                              std::string * err) {
    if (!m.nar_resident) {
        if (err) {
            *err = "YuE2 NAR half is not resident (yue2_load_parts(want_nar=true) first)";
        }
        return false;
    }
    const Yue2LmConfig & c         = m.lm_cfg;
    const int64_t         H         = (int64_t) c.embedding_length;
    const int64_t         LD        = (int64_t) c.latent_dim;
    const int              L         = (int) c.block_count;
    const int64_t         ar_length = chunk.ar_length;
    const int64_t         nar_len   = chunk.nar_length;
    const int64_t         chunk_len = chunk.chunk_len;
    const bool pad_gqa = yue2_lm_use_flash(m.backend) &&
        std::getenv("YUE2_NAR_PAD_GQA") != nullptr &&
        std::strcmp(std::getenv("YUE2_NAR_PAD_GQA"), "1") == 0;
    const int64_t kv_len = ar_length + nar_len;
    const int64_t kv_pad = (kv_len + 255) / 256 * 256;
    const int64_t M      = chunk.n_var;
    if ((int64_t) state.size() != M * chunk_len * LD) {
        if (err) {
            *err = "yue2_nar_velocity: state size mismatch (expected n_var*chunk_len*latent_dim)";
        }
        return false;
    }
    if (!chunk.cache || chunk.set < 0 || chunk.set >= chunk.cache->n_sets ||
        chunk.cache->filled[(size_t) chunk.set] < ar_length) {
        if (err) {
            *err = "yue2_nar_velocity: the chunk's AR prefix cache is not bound or not filled";
        }
        return false;
    }

    // ── host-side scalar/vector precompute ──
    const double shifted = yue2_nar_shift_t(raw_t, (double) c.timestep_shift);
    std::vector<float> time_feat;
    yue2_nar_time_features(shifted, &time_feat);  // 256 floats

    std::vector<float> x_nar_host((size_t) (M * nar_len * LD), 0.0f);  // zero-padded front+back rows, per variation
    for (int64_t j = 0; j < M; j++) {
        memcpy(x_nar_host.data() + (size_t) (j * nar_len * LD + LD), state.data() + (size_t) (j * chunk_len * LD),
               (size_t) (chunk_len * LD) * sizeof(float));
    }

    std::vector<int32_t> local_idx_host((size_t) nar_len);
    std::vector<int32_t> rope_pos_host((size_t) nar_len);
    const int64_t         max_idx = (int64_t) c.max_latent_frames - 1;
    for (int64_t t = 0; t < nar_len; t++) {
        local_idx_host[(size_t) t] = (int32_t) std::min<int64_t>(t, max_idx);
        rope_pos_host[(size_t) t]  = (int32_t) (ar_length + t);
    }
    std::vector<int32_t> content_idx_host((size_t) (M * chunk_len));
    for (int64_t j = 0; j < M; j++) {
        for (int64_t t = 0; t < chunk_len; t++) {
            content_idx_host[(size_t) (j * chunk_len + t)] = (int32_t) (t + 1);  // skip the front boundary row
        }
    }

    // ── compute graph: its shape and model inputs are fixed for this chunk ──
    Yue2NarVelocityGraph & g = chunk.velocity_graph;
    if (!g.gf) {
    const size_t ctx_bytes =
        ggml_tensor_overhead() * (YUE2_NAR_MAX_NODES + 256) + ggml_graph_overhead_custom(YUE2_NAR_MAX_NODES, false);
    g.gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g.gbuf) {
        if (err) *err = "out of host memory allocating the YuE2 NAR compute graph context";
        return false;
    }
    ggml_init_params ip = { ctx_bytes, g.gbuf, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) {
        yue2_nar_velocity_graph_free(&g);
        if (err) *err = "ggml_init failed for the YuE2 NAR compute graph context";
        return false;
    }
    ggml_context * ctx = g.ctx;
    ggml_cgraph * gf = g.gf = ggml_new_graph_custom(ctx, YUE2_NAR_MAX_NODES, false);

    ggml_tensor * in_x_nar = g.in_x_nar = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, LD, nar_len, M);
    ggml_set_name(in_x_nar, "yue2_nar_x_nar");
    ggml_set_input(in_x_nar);
    ggml_tensor * in_time_feat = g.in_time_feat = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_set_name(in_time_feat, "yue2_nar_time_feat");
    ggml_set_input(in_time_feat);
    ggml_tensor * in_local_idx = g.in_local_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nar_len);
    ggml_set_name(in_local_idx, "yue2_nar_local_idx");
    ggml_set_input(in_local_idx);
    ggml_tensor * in_rope_pos = g.in_rope_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nar_len);
    ggml_set_name(in_rope_pos, "yue2_nar_rope_pos");
    ggml_set_input(in_rope_pos);
    ggml_tensor * in_content_idx = g.in_content_idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, chunk_len, M);
    ggml_set_name(in_content_idx, "yue2_nar_content_idx");
    ggml_set_input(in_content_idx);
    if (pad_gqa) {
        g.in_attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_pad, nar_len);
        ggml_set_name(g.in_attn_mask, "yue2_nar_padding_mask");
        ggml_set_input(g.in_attn_mask);
        if (kv_pad > kv_len) {
            g.in_attn_zeros = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.key_length,
                                                   kv_pad - kv_len, c.head_count_kv, 1);
            ggml_set_name(g.in_attn_zeros, "yue2_nar_padding_zeros");
            ggml_set_input(g.in_attn_zeros);
        }
    }

    // Input embedding: vae2llm(padded state) + time_embedder(shifted t) + pos_emb[local_idx].
    ggml_tensor * ve_linear = ggml_mul_mat(ctx, m.lm.vae2llm_w, in_x_nar);
    ggml_tensor * ve = ggml_add(ctx, ve_linear, yue2_lm_match_type(ctx, ve_linear, m.lm.vae2llm_b));

    ggml_tensor * t0 = ggml_reshape_2d(ctx, in_time_feat, 256, 1);
    ggml_tensor * t1_linear = m.convrot ?
        yue2_convrot_linear(ctx, m.convrot->time0(), t0) :
        ggml_mul_mat(ctx, m.lm.time_embd_w[0], t0);
    ggml_tensor * t1 = ggml_silu(ctx, ggml_add(ctx, t1_linear,
        yue2_lm_match_type(ctx, t1_linear, m.lm.time_embd_b[0])));
    ggml_tensor * t2_linear = m.convrot ?
        yue2_convrot_linear(ctx, m.convrot->time2(), t1) :
        ggml_mul_mat(ctx, m.lm.time_embd_w[1], t1);
    ggml_tensor * t2 = ggml_add(ctx, t2_linear,
        yue2_lm_match_type(ctx, t2_linear, m.lm.time_embd_b[1]));

    ggml_tensor * pos_emb = ggml_get_rows(ctx, m.lm.latent_pos_embed, in_local_idx);  // [H, nar_len]

    ggml_tensor * x = ggml_add(ctx, ggml_add(ctx, ve, t2),
                               yue2_lm_match_type(ctx, ve, pos_emb));

    g.x_embed_tap = x;
    ggml_set_output(g.x_embed_tap);
    ggml_build_forward_expand(gf, g.x_embed_tap);

    const bool use_flash = yue2_lm_use_flash(m.backend);
    // This chunk's AR prefix: rows [0, ar_length) of set `chunk.set`. A chunk
    // bound to a pipeline cache reads a strict sub-view (more capacity, maybe
    // more sets) and is made contiguous first so every backend's concat and
    // repeat take it; the trainer/probe path owns an exact-size cache and
    // passes the tensor itself, as before.
    const Yue2ArKvCache & kc = *chunk.cache;
    const bool whole = kc.n_sets == 1 && kc.capacity == ar_length;
    for (int i = 0; i < L; i++) {
        ggml_tensor * ar_k = kc.k[(size_t) i];
        ggml_tensor * ar_v = kc.v[(size_t) i];
        if (!whole) {
            ar_k = ggml_cont(ctx, ggml_view_4d(ctx, ar_k, ar_k->ne[0], ar_length, ar_k->ne[2], 1, ar_k->nb[1],
                                                ar_k->nb[2], ar_k->nb[3], (size_t) chunk.set * ar_k->nb[3]));
            ar_v = ggml_cont(ctx, ggml_view_4d(ctx, ar_v, ar_v->ne[0], ar_length, ar_v->ne[2], 1, ar_v->nb[1],
                                                ar_v->nb[2], ar_v->nb[3], (size_t) chunk.set * ar_v->nb[3]));
        }
        x = yue2_nar_block(ctx, c, m.lm.nar_blk[(size_t) i], x, in_rope_pos, ar_k, ar_v, use_flash,
                           g.in_attn_mask, g.in_attn_zeros, &m, i);
    }

    ggml_tensor * h_final = yue2_lm_rms(ctx, x, m.convrot ? m.convrot->nar().final_norm :
                                                     m.lm.nar_output_norm, c.rms_eps);
    ggml_tensor * out_linear = m.convrot ?
        yue2_convrot_linear(ctx, m.convrot->llm2vae(), h_final) :
        ggml_mul_mat(ctx, m.lm.llm2vae_w, h_final);
    ggml_tensor * out_full = ggml_add(ctx, out_linear,
        yue2_lm_match_type(ctx, out_linear, m.lm.llm2vae_b));
    ggml_tensor * velocity_out = g.velocity_out = ggml_get_rows(ctx, out_full, in_content_idx);  // [LD, chunk_len, M] — drops both boundary rows
    ggml_set_name(velocity_out, "yue2_nar_velocity");
    ggml_set_output(velocity_out);
    ggml_build_forward_expand(gf, velocity_out);

    BackendPair bp = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
    g.sched = backend_sched_new(bp, YUE2_NAR_MAX_NODES);
    if (!g.sched || !ggml_backend_sched_alloc_graph(g.sched, g.gf)) {
        yue2_nar_velocity_graph_free(&g);
        if (err) {
            *err = "YuE2 NAR velocity graph allocation failed (out of VRAM?) at nar_len=" +
                   std::to_string((long long) nar_len);
        }
        return false;
    }
    }

    ggml_backend_tensor_set(g.in_x_nar, x_nar_host.data(), 0, x_nar_host.size() * sizeof(float));
    ggml_backend_tensor_set(g.in_time_feat, time_feat.data(), 0, time_feat.size() * sizeof(float));
    ggml_backend_tensor_set(g.in_local_idx, local_idx_host.data(), 0, local_idx_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.in_rope_pos, rope_pos_host.data(), 0, rope_pos_host.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.in_content_idx, content_idx_host.data(), 0, content_idx_host.size() * sizeof(int32_t));
    if (g.in_attn_mask && !g.attn_constants_uploaded) {
        std::vector<uint16_t> mask_host((size_t) (kv_pad * nar_len), ggml_fp32_to_fp16(-INFINITY));
        for (int64_t row = 0; row < nar_len; ++row) {
            std::fill(mask_host.begin() + (size_t) (row * kv_pad),
                      mask_host.begin() + (size_t) (row * kv_pad + kv_len), ggml_fp32_to_fp16(0.0f));
        }
        ggml_backend_tensor_set(g.in_attn_mask, mask_host.data(), 0, mask_host.size() * sizeof(uint16_t));
        if (g.in_attn_zeros) {
            std::vector<uint16_t> zeros(ggml_nelements(g.in_attn_zeros), 0);
            ggml_backend_tensor_set(g.in_attn_zeros, zeros.data(), 0, zeros.size() * sizeof(uint16_t));
        }
        g.attn_constants_uploaded = true;
    }

    yue2_imatrix_hook(g.sched);
    bool ok = ggml_backend_sched_graph_compute(g.sched, g.gf) == GGML_STATUS_SUCCESS;
    if (!ok) {
        if (err) {
            *err = "YuE2 NAR velocity graph compute failed";
        }
    } else {
        out->velocity.assign((size_t) (M * chunk_len * LD), 0.0f);
        ggml_backend_tensor_get(g.velocity_out, out->velocity.data(), 0, (size_t) (M * chunk_len * LD) * sizeof(float));
        if (want_input_embedding) {
            out->input_embedding.assign((size_t) (M * nar_len * H), 0.0f);
            ggml_backend_tensor_get(g.x_embed_tap, out->input_embedding.data(), 0,
                                    (size_t) (M * nar_len * H) * sizeof(float));
        }
    }
    return ok;
}

// ── The 32-step midpoint (RK2) solver (03-reference-numerics.md §3.8) ──────

struct Yue2NarPinnedStep {
    int64_t             step = 0;
    std::vector<float>  velocity_first;  // v1 = velocity(state, t)          [chunk_len,64]
    std::vector<float>  velocity_mid;    // v2 = velocity(mid, t-dt/2)       [chunk_len,64]
    std::vector<float>  state_after;     // state after this step's full update
};

struct Yue2NarSolveResult {
    std::vector<float>              final_latents;  // [chunk_len, 64]
    std::vector<Yue2NarPinnedStep>  pinned;
    std::vector<float>              input_embedding_step0;  // [nar_len, H], only if requested
    int                              velocity_calls = 0;
    double                           total_velocity_ms = 0.0;
};

// `initial_noise` is this chunk's own noise slice (from the whole-song draw,
// per 03 §3.10/§3.7 — draw once, slice per chunk, never reseed per chunk),
// [chunk_len, 64] F32. `pinned_steps` selects which steps' velocity/state get
// captured for parity reporting (typically {0,1,ode_steps/2-1,ode_steps-1}
// per 02-fixture-schema.md §2.6). `steps` is `ode_steps` (32 for this
// checkpoint, but read from config, never hardcoded — 03 §3.9's own caution).
static bool yue2_nar_solve_midpoint(const Yue2Model & m, const Yue2NarChunk & chunk,
                                    const std::vector<float> & initial_noise, int steps,
                                    const std::vector<int64_t> & pinned_steps, bool want_input_embedding_step0,
                                    Yue2NarSolveResult * out, std::string * err, float cache_ratio = 0.0f) {
    if (steps <= 0) {
        if (err) {
            *err = "yue2_nar_solve_midpoint: steps must be > 0";
        }
        return false;
    }
    const int64_t n = (int64_t) initial_noise.size();
    if (n != chunk.n_var * chunk.chunk_len * (int64_t) m.lm_cfg.latent_dim) {
        if (err) {
            *err = "yue2_nar_solve_midpoint: initial_noise size mismatch";
        }
        return false;
    }
    auto is_pinned = [&](int64_t s) {
        return std::find(pinned_steps.begin(), pinned_steps.end(), s) != pinned_steps.end();
    };

    // ── Step-level velocity caching (ported from hot-step-sampler.h's own
    // cache_ratio for ACE-Step's DiT solver -- same idea: skip the full
    // predictor+corrector network evaluation for a fraction of the middle
    // ODE steps and just re-use the last real step's corrector velocity to
    // advance the state instead. First/last 2 steps and any pinned step
    // (parity capture needs a real compute) always run for real.
    // cache_ratio=0 (default) reproduces the original behaviour exactly --
    // every step computed for real, no caller change required. Unvalidated
    // quality tradeoff, off by default. ─────────────────────────────────
    std::vector<bool> step_computes((size_t) steps, true);
    if (cache_ratio > 0.0f && steps > 4) {
        const int protect      = 2;
        const int middle_start = protect;
        const int middle_end   = steps - protect;
        const int middle_len   = middle_end - middle_start;
        if (middle_len > 1) {
            const int target_cached  = std::min(middle_len - 1, (int) roundf(cache_ratio * (float) middle_len));
            const int target_compute = middle_len - target_cached;
            if (target_compute > 0 && target_cached > 0) {
                for (int s = middle_start; s < middle_end; s++) {
                    step_computes[(size_t) s] = false;
                }
                for (int ci = 0; ci < target_compute; ci++) {
                    int idx = middle_start + (int) roundf((float) ci * (float) middle_len / (float) target_compute);
                    if (idx < middle_end) {
                        step_computes[(size_t) idx] = true;
                    }
                }
            }
        }
        for (int64_t p : pinned_steps) {
            if (p >= 0 && p < steps) {
                step_computes[(size_t) p] = true;
            }
        }
        int cached_count = 0;
        for (int s = 0; s < steps; s++) {
            if (!step_computes[(size_t) s]) cached_count++;
        }
        fprintf(stderr, "[YuE2-NAR] Velocity cache: ratio=%.2f, %d/%d steps cached, %d computed\n",
                cache_ratio, cached_count, steps, steps - cached_count);
    }

    const double dt = 1.0 / (double) steps;
    std::vector<float> state = initial_noise;
    std::vector<float> v_cached;
    bool have_cached_v = false;

    auto eval = [&](const std::vector<float> & s, double raw_t, bool want_emb, Yue2NarVelocityResult * r) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = yue2_nar_velocity(m, chunk, s, raw_t, want_emb, r, err);
        out->total_velocity_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        out->velocity_calls++;
        return ok;
    };

    for (int step = 0; step < steps; step++) {
        const double t   = 1.0 - (double) step * dt;
        const double raw = yue2_nar_logit_clamped(t);

        if (!step_computes[(size_t) step] && have_cached_v) {
            // Cached step: no network call at all, just advance the state
            // with the last real step's corrector velocity.
            std::vector<float> new_state((size_t) n);
            for (int64_t i = 0; i < n; i++) {
                new_state[(size_t) i] = state[(size_t) i] - v_cached[(size_t) i] * (float) dt;
            }
            state = std::move(new_state);
            continue;
        }

        Yue2NarVelocityResult first;
        const bool want_emb0 = want_input_embedding_step0 && step == 0;
        if (!eval(state, raw, want_emb0, &first)) {
            return false;
        }
        if (want_emb0) {
            out->input_embedding_step0 = first.input_embedding;
        }

        std::vector<float> mid((size_t) n);
        for (int64_t i = 0; i < n; i++) {
            mid[(size_t) i] = state[(size_t) i] - first.velocity[(size_t) i] * (float) (dt / 2.0);
        }

        const double raw_mid = yue2_nar_logit_clamped(t - dt / 2.0);
        Yue2NarVelocityResult second;
        if (!eval(mid, raw_mid, false, &second)) {
            return false;
        }

        std::vector<float> new_state((size_t) n);
        for (int64_t i = 0; i < n; i++) {
            new_state[(size_t) i] = state[(size_t) i] - second.velocity[(size_t) i] * (float) dt;
        }

        if (is_pinned((int64_t) step)) {
            Yue2NarPinnedStep p;
            p.step           = step;
            p.velocity_first = first.velocity;
            p.velocity_mid   = second.velocity;
            p.state_after    = new_state;
            out->pinned.push_back(std::move(p));
        }

        v_cached      = second.velocity;
        have_cached_v = true;
        state         = std::move(new_state);
    }

    out->final_latents = std::move(state);
    return true;
}

// Optional YuE2 NAR Lua path. The native midpoint function above remains the
// untouched default and continues to serve the forced-replay parity harness.
// A scheduler may be selected on its own (variable-step midpoint), or with
// the single-evaluation Wasserstein solver. The plugin scheduler returns N
// source timesteps; this loop owns the final destination t=0, so N means N
// actual updates and the denoise trajectory always reaches zero.
static bool yue2_nar_solve_plugins(
    const Yue2Model & m, const Yue2NarChunk & chunk,
    const std::vector<float> & initial_noise, int steps,
    const std::string & solver_name, const std::string & scheduler_name,
    const std::unordered_map<std::string, std::string> & plugin_params,
    Yue2NarSolveResult * out, std::string * err) {
    if (steps <= 0 || chunk.n_var != 1 ||
        initial_noise.size() != (size_t) (chunk.chunk_len * m.lm_cfg.latent_dim)) {
        if (err) *err = "YuE2 NAR plugin solve: invalid steps or noise shape (plugins solve one variation at a time)";
        return false;
    }
    auto & registry = PluginRegistry::instance();
    LuaPlugin * solver = solver_name.empty() ? nullptr : registry.solver_lookup(solver_name.c_str());
    LuaPlugin * scheduler = scheduler_name.empty() ? nullptr : registry.scheduler_lookup(scheduler_name.c_str());
    if ((!solver_name.empty() && !solver) || (!scheduler_name.empty() && !scheduler)) {
        if (err) *err = "YuE2 NAR plugin was not loaded; check the plugin startup log";
        return false;
    }
    if (solver && (solver->owns_loop || solver->needs_model)) {
        if (err) *err = "YuE2 NAR supports only single-evaluation step() solvers";
        return false;
    }

    std::vector<float> times((size_t) steps + 1, 0.0f);
    if (scheduler) {
        std::vector<float> emitted((size_t) steps, std::numeric_limits<float>::quiet_NaN());
        LuaModelContext ctx;
        ctx.model_id = "yue2";
        lua_call_scheduler(*scheduler, emitted.data(), steps, 1.0f, plugin_params, ctx);
        for (int i = 0; i < steps; i++) times[(size_t) i] = emitted[(size_t) i];
    } else {
        for (int i = 0; i < steps; i++) {
            times[(size_t) i] = 1.0f - (float) i / (float) steps;
        }
    }
    times[(size_t) steps] = 0.0f;
    if (!std::isfinite(times[0]) || std::abs(times[0] - 1.0f) > 1e-5f) {
        if (err) *err = "YuE2 NAR scheduler must start at t=1";
        return false;
    }
    for (int i = 0; i < steps; i++) {
        if (!std::isfinite(times[(size_t) i]) ||
            !(times[(size_t) i] > times[(size_t) i + 1]) ||
            times[(size_t) i] > 1.0f || times[(size_t) i + 1] < 0.0f) {
            if (err) *err = "YuE2 NAR scheduler emitted an invalid or non-descending timestep";
            return false;
        }
    }

    fprintf(stderr, "[YuE2-NAR-Plugins] solver=%s scheduler=%s steps=%d\n",
            solver ? solver->name.c_str() : "midpoint",
            scheduler ? scheduler->name.c_str() : "uniform", steps);
    std::vector<float> state = initial_noise;
    const int n = (int) state.size();
    SolverState solver_state;
    solver_state.batch_n = 1;
    solver_state.n_per = n;
    LuaModelContext ctx;
    ctx.model_id = "yue2";
    auto eval = [&](const std::vector<float> & s, float t, Yue2NarVelocityResult * result) -> bool {
        const auto start = std::chrono::steady_clock::now();
        const bool ok = yue2_nar_velocity(m, chunk, s, yue2_nar_logit_clamped(t), false, result, err);
        out->total_velocity_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        out->velocity_calls++;
        return ok;
    };

    for (int step = 0; step < steps; step++) {
        const float t = times[(size_t) step];
        const float next_t = times[(size_t) step + 1];
        const float dt = t - next_t;
        Yue2NarVelocityResult first;
        if (!eval(state, t, &first)) return false;
        if (solver) {
            solver_state.step_index = step;
            lua_call_solver_step(*solver, state.data(), first.velocity.data(), t, next_t, n,
                                 solver_state, SolverModelFn{}, nullptr, plugin_params, ctx);
        } else {
            std::vector<float> mid((size_t) n);
            for (int i = 0; i < n; i++) mid[(size_t) i] = state[(size_t) i] - first.velocity[(size_t) i] * (dt * 0.5f);
            Yue2NarVelocityResult second;
            if (!eval(mid, t - dt * 0.5f, &second)) return false;
            for (int i = 0; i < n; i++) state[(size_t) i] -= second.velocity[(size_t) i] * dt;
        }
        for (float x : state) {
            if (!std::isfinite(x)) {
                if (err) *err = "YuE2 NAR plugin produced non-finite latents";
                return false;
            }
        }
    }
    out->final_latents = std::move(state);
    return true;
}
