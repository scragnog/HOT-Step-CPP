#pragma once
// lm-optim.h — the hand-rolled optimizer (L1).
//
// ggml-opt cannot be used: it structurally cannot do grad-norm clipping and
// cannot expose gradients in dynamic-graph mode (ggml_opt_eval nulls gb_opt,
// ggml_opt_grad_acc then derefs null — access violation reproduced in the
// Phase-0 spike). So we drive ggml_build_backward_expand ourselves with our own
// persistent accumulators and emit our own ggml_opt_step_adamw nodes.
//
// Buffer split (mandatory, plan §3.5.1):
//   buf_grad — accumulators; ggml_backend_buffer_clear() at the start of every
//              optimizer window.
//   buf_mom  — AdamW m/v; cleared exactly once at init and never again.
//   (scalars live in the caller's static buffer)
//
// ggml_graph_reset() zeroes AdamW momenta whenever the graph contains an
// OPT_STEP_ADAMW node, so we never call it.
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §3.5

#include "train/lm-common.h"
#include "train/lm-graph.h"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

// Side-Step's lr_lambda (train_lm.py:268-272) with L6(b) applied:
// `step`, NOT `step + 1` — lr == 0 at optimizer step 0.
//
// `floor` is the fraction of base_lr the cosine bottoms out at. 0.1 is
// Side-Step's and stays the default for every existing caller; the MM3 LM
// trainer lowers it to match SimpleTuner's `lr_end`, which runs the tail of the
// schedule an order of magnitude colder than we did.
static inline float lm_lr_lambda(int step, int total, int warmup, float floor = 0.1f) {
    if (warmup > 0 && step < warmup) {
        return (float) step / (float) warmup;
    }
    const float p = (float) (step - warmup) / (float) std::max(1, total - warmup);
    return floor + (1.0f - floor) * 0.5f * (1.0f + cosf(3.14159265358979f * p));
}

// ─── per-parameter update rule (2026-07-30) ─────────────────────────────────
//
// The optimizer is chosen PER PARAMETER, not globally, because Muon only means
// anything on a matrix: it orthogonalizes the update, and "orthogonalize" is
// vacuous on a LoKR w1 of [4,5]. Every published Muon puts the non-matrix
// parameters on AdamW for exactly this reason, so the hybrid is the design, not
// a compromise.
//
// This split is also the seam a Lua optimizer plugin would slot into later: a
// rule is (state buffers it needs) + (nodes it emits per parameter). Adding
// LM_RULE_LUA means implementing those two things against an interface that
// already has two real users, rather than guessing from AdamW alone.
enum LmOptRule {
    LM_RULE_ADAMW = 0,
    LM_RULE_MUON  = 1,
};

struct LmMuonCfg {
    // Muon's update is normalized by construction, so its LR does not mean what
    // AdamW's means. Kept as a multiplier on base_lr so the shared cosine
    // schedule (lm_lr_lambda) still drives both classes coherently.
    float lr_scale = 1.0f;
    float momentum = 0.95f;
    int   ns_steps = 5;
    bool  nesterov = true;
    // Below this on the SHORT side a matrix is not worth orthogonalizing: the
    // Newton-Schulz iteration would be shaping a subspace of that many
    // dimensions. LoKR w1 ([4,5] at factor 6) sits here and falls to AdamW.
    int   min_dim = 16;
    // < 0 inherits LmOptim::weight_decay. Decoupled either way.
    float wd = -1.0f;
    // Max parameters per shape bucket. 16 is MEASURED, not chosen: the optimizer
    // step on a LoKR dim512+MLP run goes 298.6 ms (bucket 1, i.e. unbatched) ->
    // 136.1 (8) -> 138.0 (16) -> 187.3 (64) -> 201.2 (128).
    //
    // It gets WORSE above ~16 because the gather is a left-folded ggml_concat,
    // which re-copies everything accumulated so far at each step — quadratic in
    // bucket size. So this knob trades launch overhead (small buckets) against
    // concat traffic (large ones), and the optimum sits early. Removing the
    // concat entirely — allocating the accumulators and momenta as slabs of one
    // per-bucket tensor, so the bucket IS the storage — is what would push this
    // toward the ~55 ms of irreducible Newton-Schulz FLOPs and make big buckets
    // the right answer again.
    int bucket = 16;
};

// A run of Muon parameters that share a shape exactly, so one batched
// Newton-Schulz serves all of them.
struct LmMuonBucket {
    int64_t          ne0 = 0, ne1 = 0;
    std::vector<int> idx;  // parameter slots
};

// Quintic Newton-Schulz coefficients (Jordan 2024). Tuned so the iteration
// converges on the singular values from the [0,1] range the pre-normalization
// guarantees; they are NOT the coefficients of an exact matrix sign iteration.
static const float LM_MUON_NS_A = 3.4445f;
static const float LM_MUON_NS_B = -4.7750f;
static const float LM_MUON_NS_C = 2.0315f;

struct LmOptim {
    ggml_context *        ctx_grad = nullptr;
    ggml_backend_buffer_t buf_grad = nullptr;
    ggml_context *        ctx_mom  = nullptr;
    ggml_backend_buffer_t buf_mom  = nullptr;

    std::vector<ggml_tensor *> params;  // == LmLora::params (already ggml_set_param'd)
    std::vector<ggml_tensor *> acc, mom_m, mom_v;
    std::unordered_map<ggml_tensor *, int> param_slot;

    // Per-parameter rule, and the config that produced it. `mom_v[j]` is
    // nullptr for a Muon parameter — Muon carries ONE momentum buffer where
    // AdamW carries two, which is worth ~900 MB at LoKR dim512 (228.6M params).
    std::string            optimizer = "adamw";
    LmMuonCfg              muon;
    std::vector<uint8_t>      rule;
    std::vector<LmMuonBucket> muon_buckets;
    int                    n_muon = 0;   // parameters actually on Muon
    int                    est_nodes = 0;  // sized in init, drives the graph cap

    // ── Prodigy (optimizer == "prodigy") ───────────────────────────────────
    //
    // Estimates its own step size, so base_lr is gamma and stays at 1.0. Two
    // extra parameter-sized buffers over AdamW: s, and x0 (the INITIAL weights,
    // which the <g, x0 - x> numerator needs). d and r are scalars carried on
    // the HOST between steps: computing max() and a scalar divide in-graph
    // would need ops ggml does not have, and a 4-byte readback per step is
    // already the pattern t_gnorm2 uses.
    std::vector<ggml_tensor *> pg_s, pg_x0;
    ggml_tensor * t_pnum = nullptr;  // [1] readback: sum over params of <g, x0-x>
    ggml_tensor * t_ps1  = nullptr;  // [1] readback: sum over params of |s|
    ggml_tensor * t_pdeps = nullptr; // [1] d*eps for the step's denominator
    double        prodigy_d = 1e-6;  // the step-size estimate; never decreases
    double        prodigy_r = 0.0;   // the running numerator
    float         prodigy_d0 = 1e-6f;

    // ── Per-parameter learning-rate groups (P1b, 2026-09-03) ───────────────
    //
    // ggml_opt_step_adamw reads alpha from a [7] tensor per call, so one shared
    // t_adamw means one LR for everything. Two groups need two tensors. `lr_mul`
    // is a per-parameter multiplier on base_lr, 1.0 for every parameter by
    // default; any parameter whose multiplier is not 1.0 is stepped with
    // t_adamw_alt, whose alpha is base * mul. ONE distinct alternate multiplier
    // is supported — enough for "the soft-prompt parameters run hot while the
    // LoRA runs at its usual rate", which is what joint token+LoRA training
    // needs (TI wants ~50x the LoRA's LR). More groups = more tensors; not yet.
    //
    // `adamw_only` pins a parameter to AdamW regardless of Muon eligibility.
    // A [H, k] token or an [Nkv*D, n] prefix is a 2-D matrix and would
    // otherwise be orthogonalised as if it were a weight — which it is not.
    std::vector<float>         lr_mul;
    std::vector<ggml_tensor *> adamw_only;
    ggml_tensor *              t_adamw_alt = nullptr;  // [7], caller-allocated when used

    // scalars, allocated by the caller in its static buffer
    ggml_tensor * t_adamw    = nullptr;  // [7] {alpha,beta1,beta2,eps,wd,beta1h,beta2h}
    ggml_tensor * t_lossgrad = nullptr;  // [1] 1/grad_accum
    ggml_tensor * t_clip     = nullptr;  // [1] grad_clip
    ggml_tensor * t_eps      = nullptr;  // [1] 1e-6
    ggml_tensor * t_gnorm2   = nullptr;  // [1] readback only

    float base_lr      = 1e-4f;
    /** Cosine floor as a fraction of base_lr. 0.1 is Side-Step's; the MM3 LM
     *  trainer sets it from --lr-end-frac to match SimpleTuner's lr_end. */
    float lr_floor     = 0.1f;
    float weight_decay = 0.01f;
    float grad_clip    = 1.0f;
    int   total_steps  = 1;
    int   warmup_steps = 1;
    int   opt_step     = 0;  // 0-based; drives the LR schedule
    int   opt_iter     = 0;  // 1-based; drives AdamW bias correction

    std::vector<uint8_t> arena;  // reused every optimizer step
};

// A parameter is Muon-eligible when it is a genuine 2-D matrix whose SHORT side
// is wide enough for orthogonalization to mean something.
static inline bool lm_muon_eligible(const ggml_tensor * p, const LmMuonCfg & c) {
    if (p->ne[2] != 1 || p->ne[3] != 1) {
        return false;
    }
    const int64_t lo = p->ne[0] < p->ne[1] ? p->ne[0] : p->ne[1];
    return lo >= (int64_t) c.min_dim;
}

// ─── Newton-Schulz orthogonalization, as ggml nodes ─────────────────────────
//
// Returns an approximately semi-orthogonal matrix with G's shape, computed
// entirely on the backend — the host never sees an element. Reference:
//
//   X = G / (||G||_F + eps)                     (puts the spectrum in [0,1])
//   repeat ns_steps:  A = X X^T
//                     B = b*A + c*A*A
//                     X = a*X + B X
//
// GGML INDEXING, because it is the whole difficulty here. A ggml tensor
// [ne0=C, ne1=R] is an R-row, C-column matrix with rows contiguous, and
// ggml_mul_mat(u, v) contracts over ne0: out(m,n) = sum_k u(k,m)*v(k,n).
// Therefore:
//   X X^T   -> mul_mat(X, X)                     [R,R], symmetric
//   A A     -> mul_mat(A, A)                     (equals A*A^T; A is symmetric)
//   B X     -> mul_mat(cont(transpose(X)), B)    [C,R], back in X's layout
//
// The reference implementation transposes when rows > cols so the Gram matrix
// is the smaller one; NS(G^T) == NS(G)^T, so this is free accuracy AND a large
// cost saving (a LoRA B of [out 2560, rank 128] would otherwise build a
// 2560x2560 Gram instead of 128x128).
static ggml_tensor * lm_muon_newton_schulz(ggml_context * ctx, ggml_tensor * G, int steps, ggml_tensor * t_eps) {
    const bool tall = G->ne[1] > G->ne[0];  // more rows than cols in matrix terms
    ggml_tensor * X = tall ? ggml_cont(ctx, ggml_transpose(ctx, G)) : G;

    // Frobenius normalization. [1] broadcasts against any shape (ggml_can_repeat).
    ggml_tensor * nrm = ggml_sqrt(ctx, ggml_sum(ctx, ggml_sqr(ctx, X)));
    X                 = ggml_div(ctx, X, ggml_add(ctx, nrm, t_eps));

    for (int i = 0; i < steps; i++) {
        ggml_tensor * A  = ggml_mul_mat(ctx, X, X);                       // [R,R]
        ggml_tensor * A2 = ggml_mul_mat(ctx, A, A);                       // [R,R]
        ggml_tensor * B  = ggml_add(ctx, ggml_scale(ctx, A, LM_MUON_NS_B),
                                    ggml_scale(ctx, A2, LM_MUON_NS_C));   // [R,R]
        ggml_tensor * BX = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, X)), B);
        X                = ggml_add(ctx, ggml_scale(ctx, X, LM_MUON_NS_A), BX);
    }
    return tall ? ggml_cont(ctx, ggml_transpose(ctx, X)) : X;
}

// ─── batched Newton-Schulz over a shape bucket ──────────────────────────────
//
// Identical maths to lm_muon_newton_schulz, run on [C, R, N] instead of [C, R]:
// every ggml op below already broadcasts over ne2, and mul_mat with equal ne2
// on both operands is one strided-batched cuBLAS call. This is the whole point
// of bucketing — a LoKR dim512+MLP run has 448 Muon parameters in THREE
// distinct shapes, so the ~31k tiny kernels the per-parameter form emits become
// a few hundred.
//
// The one thing that does NOT come free is the Frobenius normalization: a plain
// ggml_sum would collapse the whole bucket to a single scalar and normalize
// every parameter by its neighbours' magnitudes. It has to be PER SLAB, which
// is what the two sum_rows do — [C,R,N] -> [1,R,N] -> (transpose) -> [1,1,N].
static ggml_tensor * lm_muon_ns_batched(ggml_context * ctx, ggml_tensor * G, int steps, ggml_tensor * t_eps) {
    const bool tall = G->ne[1] > G->ne[0];  // transpose keeps ne2, so this is still per-slab
    ggml_tensor * X = tall ? ggml_cont(ctx, ggml_transpose(ctx, G)) : G;

    ggml_tensor * s2 = ggml_sum_rows(ctx, ggml_sqr(ctx, X));                  // [1, R, N]
    s2               = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, s2)));  // [1, 1, N]
    ggml_tensor * nrm = ggml_sqrt(ctx, s2);
    X                 = ggml_div(ctx, X, ggml_add(ctx, nrm, t_eps));  // [1,1,N] repeats into [C,R,N]

    for (int i = 0; i < steps; i++) {
        ggml_tensor * A  = ggml_mul_mat(ctx, X, X);  // [R,R,N], one batched GEMM
        ggml_tensor * A2 = ggml_mul_mat(ctx, A, A);
        ggml_tensor * B  = ggml_add(ctx, ggml_scale(ctx, A, LM_MUON_NS_B), ggml_scale(ctx, A2, LM_MUON_NS_C));
        ggml_tensor * BX = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, X)), B);
        X                = ggml_add(ctx, ggml_scale(ctx, X, LM_MUON_NS_A), BX);
    }
    return tall ? ggml_cont(ctx, ggml_transpose(ctx, X)) : X;
}

// Nodes emitted per Muon parameter, for sizing the optimizer graph. COUNTED
// FROM THE EMITTED GRAPH, not estimated — an earlier guess of 7 per iteration
// undersized the cap and ggml aborts on overflow (ggml.c:7001) partway through
// the first optimizer step, i.e. minutes into a run.
//
// Per NS iteration: mul_mat(X,X), mul_mat(A,A), 2 scale, 1 add, transpose+cont,
// mul_mat, scale, add = 10. Fixed: momentum (2), Nesterov (2), the Frobenius
// normalization sqr/sum/sqrt/add/div (5), shape scale (1), decoupled weight
// decay (2), two write-backs (2), and up to 4 for the tall-matrix transpose in
// and out.
static inline int lm_muon_nodes_per_param(int ns_steps) {
    return 20 + ns_steps * 10;
}

static void lm_optim_free(LmOptim * o) {
    if (o->buf_grad) {
        ggml_backend_buffer_free(o->buf_grad);
    }
    if (o->ctx_grad) {
        ggml_free(o->ctx_grad);
    }
    if (o->buf_mom) {
        ggml_backend_buffer_free(o->buf_mom);
    }
    if (o->ctx_mom) {
        ggml_free(o->ctx_mom);
    }
    o->buf_grad = nullptr;
    o->ctx_grad = nullptr;
    o->buf_mom  = nullptr;
    o->ctx_mom  = nullptr;
    o->acc.clear();
    o->mom_m.clear();
    o->mom_v.clear();
    o->params.clear();
    o->param_slot.clear();
    o->arena.clear();
    o->arena.shrink_to_fit();
}

// `params` must already be ggml_set_param'd and backend-allocated in a buffer
// that nothing clears (LmLora::buf, or the self-test's toy buffer).
static bool lm_optim_init(LmOptim * o, const std::vector<ggml_tensor *> & params, ggml_backend_t backend,
                          std::string * err) {
    o->params      = params;
    const size_t n = o->params.size();

    {
        ggml_init_params p = { (n + 8) * ggml_tensor_overhead(), nullptr, true };
        o->ctx_grad        = ggml_init(p);
    }
    const bool want_prodigy = (o->optimizer == "prodigy");
    {
        // AdamW/Muon: m (+v). Prodigy: m, v, s, x0 — four per parameter, plus
        // the two readback scalars.
        const size_t per = want_prodigy ? 4 : 2;
        ggml_init_params p = { (per * n + 16) * ggml_tensor_overhead(), nullptr, true };
        o->ctx_mom         = ggml_init(p);
    }
    if (!o->ctx_grad || !o->ctx_mom) {
        *err = "cannot create the optimizer contexts";
        return false;
    }

    o->acc.resize(n);
    o->mom_m.resize(n);
    o->mom_v.resize(n);
    o->rule.assign(n, (uint8_t) LM_RULE_ADAMW);
    o->n_muon = 0;
    const bool want_muon = (o->optimizer == "muon");
    if (want_prodigy) {
        o->pg_s.resize(n);
        o->pg_x0.resize(n);
    }
    if (o->lr_mul.size() != n) {
        o->lr_mul.assign(n, 1.0f);
    }
    for (size_t j = 0; j < n; j++) {
        ggml_tensor * pj = o->params[j];
        bool pinned = false;
        for (size_t q = 0; q < o->adamw_only.size(); q++) {
            if (o->adamw_only[q] == pj) {
                pinned = true;
                break;
            }
        }
        if (want_muon && !pinned && lm_muon_eligible(pj, o->muon)) {
            o->rule[j] = (uint8_t) LM_RULE_MUON;
            o->n_muon++;
        }
        o->acc[j]   = ggml_new_tensor_2d(o->ctx_grad, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        o->mom_m[j] = ggml_new_tensor_2d(o->ctx_mom, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        // Muon carries one momentum buffer, AdamW two. Skipping v where it is
        // unused is the difference between 2x and 1x the parameter set in
        // optimizer state — ~900 MB at LoKR dim512. dit-vram.h still charges for
        // both, so the auto-fit stays CONSERVATIVE rather than wrong; teaching
        // it the rule split is a follow-up, not a correctness fix.
        o->mom_v[j] = (o->rule[j] == LM_RULE_MUON)
                          ? nullptr
                          : ggml_new_tensor_2d(o->ctx_mom, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        if (want_prodigy) {
            o->pg_s[j]  = ggml_new_tensor_2d(o->ctx_mom, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
            o->pg_x0[j] = ggml_new_tensor_2d(o->ctx_mom, GGML_TYPE_F32, pj->ne[0], pj->ne[1]);
        }
        char nm[96];
        snprintf(nm, sizeof(nm), "acc.%s", pj->name);
        ggml_set_name(o->acc[j], nm);
        snprintf(nm, sizeof(nm), "m.%s", pj->name);
        ggml_set_name(o->mom_m[j], nm);
        if (o->mom_v[j]) {
            snprintf(nm, sizeof(nm), "v.%s", pj->name);
            ggml_set_name(o->mom_v[j], nm);
        }
        // NAMES MATTER: the resume file and the x0 file key tensors BY NAME, so
        // unnamed state all collapses onto one map entry and restores as "tensor
        // \"\" is the wrong size".
        if (want_prodigy) {
            snprintf(nm, sizeof(nm), "pg_s.%s", pj->name);
            ggml_set_name(o->pg_s[j], nm);
            snprintf(nm, sizeof(nm), "pg_x0.%s", pj->name);
            ggml_set_name(o->pg_x0[j], nm);
        }
        GGML_ASSERT(ggml_are_same_shape(o->acc[j], pj));
        o->param_slot[pj] = (int) j;
    }
    // Group the Muon parameters by EXACT shape, then chunk each group so only a
    // bounded slice of a bucket is live at once. A LoKR dim512+MLP run collapses
    // to three shapes (352 x [512,512], 64 x [512,2432], 32 x [2432,512]) — the
    // difference between ~31k kernel launches per optimizer step and a few
    // hundred.
    if (want_prodigy) {
        // In ctx_mom so the existing alloc/clear covers them. ctx_mom was sized
        // with +16 tensors of slack for exactly this.
        o->t_pnum  = ggml_new_tensor_1d(o->ctx_mom, GGML_TYPE_F32, 1);
        o->t_ps1   = ggml_new_tensor_1d(o->ctx_mom, GGML_TYPE_F32, 1);
        o->t_pdeps = ggml_new_tensor_1d(o->ctx_mom, GGML_TYPE_F32, 1);
        ggml_set_name(o->t_pnum, "prodigy.num");
        ggml_set_name(o->t_ps1, "prodigy.s_l1");
        ggml_set_name(o->t_pdeps, "prodigy.d_eps");
    }
    o->muon_buckets.clear();
    if (want_muon) {
        const int cap = o->muon.bucket > 0 ? o->muon.bucket : 1;
        for (size_t j = 0; j < n; j++) {
            if (o->rule[j] != LM_RULE_MUON) {
                continue;
            }
            const ggml_tensor * pj = o->params[j];
            LmMuonBucket *      b  = nullptr;
            for (size_t k = 0; k < o->muon_buckets.size(); k++) {
                LmMuonBucket & cand = o->muon_buckets[k];
                if (cand.ne0 == pj->ne[0] && cand.ne1 == pj->ne[1] && (int) cand.idx.size() < cap) {
                    b = &cand;
                    break;
                }
            }
            if (!b) {
                LmMuonBucket nb;
                nb.ne0 = pj->ne[0];
                nb.ne1 = pj->ne[1];
                o->muon_buckets.push_back(nb);
                b = &o->muon_buckets.back();
            }
            b->idx.push_back((int) j);
        }
        fprintf(stderr, "[optim] muon: %d of %zu parameters on Muon (min_dim %d), %zu on AdamW; %zu shape buckets\n",
                o->n_muon, n, o->muon.min_dim, n - (size_t) o->n_muon, o->muon_buckets.size());
    }

    o->buf_grad = ggml_backend_alloc_ctx_tensors(o->ctx_grad, backend);
    o->buf_mom  = ggml_backend_alloc_ctx_tensors(o->ctx_mom, backend);
    if (!o->buf_grad || !o->buf_mom) {
        *err = "optimizer state allocation failed";
        return false;
    }
    ggml_backend_buffer_clear(o->buf_grad, 0);
    ggml_backend_buffer_clear(o->buf_mom, 0);  // exactly once, never again

    if (want_prodigy) {
        // x0 := the weights as they are RIGHT NOW. This runs after the clear,
        // which would otherwise zero it, and before the first step. For a LoRA
        // that means x0 holds A's kaiming draw and B's zeros — the numerator
        // <g, x0 - x> is measured from wherever training actually began.
        std::vector<float> tmp;
        for (size_t j = 0; j < n; j++) {
            const size_t cnt = (size_t) ggml_nelements(o->params[j]);
            tmp.resize(cnt);
            ggml_backend_tensor_get(o->params[j], tmp.data(), 0, cnt * sizeof(float));
            ggml_backend_tensor_set(o->pg_x0[j], tmp.data(), 0, cnt * sizeof(float));
        }
        o->prodigy_d = (double) o->prodigy_d0;
        o->prodigy_r = 0.0;
        fprintf(stderr,
                "[optim] prodigy: d0 %.3g, %zu parameters, +2 state buffers over AdamW; "
                "lr is gamma (schedule only) and d sets the real step size%s",
                (double) o->prodigy_d0, n, "\n");
    }

    // Graph size. AdamW is ~6 nodes/param (392 params -> ~2400), for which 16 MB
    // was generous. Muon is ~50, and a LoKR dim512+MLP run has ~1000 parameters,
    // so both the node cap and the arena have to be derived rather than fixed —
    // an 8192 cap would abort partway through the first optimizer step.
    // n*8 covers the global-norm reduction (sqr + sum per parameter plus the
    // balanced-tree adds) and the AdamW nodes; +25% because overflowing the cap
    // is a hard abort mid-run, and the headroom costs only address space.
    // Bucketed Muon pays ~(20 + 10*ns) per BUCKET, plus a small fixed cost per
    // parameter for the concat in and the view/scale/add/cpy out.
    o->est_nodes = (int) (((double) n * (want_prodigy ? 24.0 : 8.0)
                           + (double) o->muon_buckets.size() * lm_muon_nodes_per_param(o->muon.ns_steps)
                           + (double) o->n_muon * 10.0)
                          * 1.25)
                   + 1024;
    const size_t cap   = (size_t) o->est_nodes;
    const size_t bytes = ggml_graph_overhead_custom(cap, false) + cap * (ggml_tensor_overhead() + 64) + (4u << 20);
    o->arena.resize(bytes > (16u << 20) ? bytes : (16u << 20));
    return true;
}

// Clear the accumulators for the next optimizer window.
static inline void lm_optim_zero_grad(LmOptim * o) {
    ggml_backend_buffer_clear(o->buf_grad, 0);
}

// Fill the gacc[] array for one forward graph: PARAM nodes -> our persistent
// accumulator, the LOSS node -> the persistent 1/grad_accum constant.
//
// ggml_build_backward_expand indexes grad_accs[] by FORWARD-GRAPH NODE INDEX
// (ggml.c: `if (grad_accs && grad_accs[i]) cgraph->grad_accs[ihash] = ...`),
// and ggml_set_param tensors ARE nodes, not leafs (ggml_visit_parents_graph).
static void lm_optim_fill_gacc(const LmOptim * o, ggml_cgraph * gf, std::vector<ggml_tensor *> * gacc) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    gacc->assign((size_t) n_nodes, nullptr);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * nd = ggml_graph_node(gf, i);
        if (nd->flags & GGML_TENSOR_FLAG_PARAM) {
            auto it = o->param_slot.find(nd);
            GGML_ASSERT(it != o->param_slot.end());
            (*gacc)[(size_t) i] = o->acc[(size_t) it->second];
        } else if (nd->flags & GGML_TENSOR_FLAG_LOSS) {
            (*gacc)[(size_t) i] = o->t_lossgrad;
        }
    }
}

// Per-parameter LR multiplier, set AFTER lm_optim_init (which sizes lr_mul).
// Returns false when `t` is not an optimizer parameter.
static bool lm_optim_set_lr_mul(LmOptim * o, ggml_tensor * t, float mul) {
    auto it = o->param_slot.find(t);
    if (it == o->param_slot.end()) {
        return false;
    }
    o->lr_mul[(size_t) it->second] = mul;
    return true;
}

struct LmStepStats {
    float lr        = 0.0f;
    float grad_norm = 0.0f;  // pre-clip global L2
    float clip      = 1.0f;  // factor actually applied
};

// One optimizer step: global norm + clip + AdamW in a single graph, with no
// host round-trip in the critical path. Zeroes the accumulators afterwards.
// ── Prodigy ─────────────────────────────────────────────────────────────────
//
// `lr_now` is gamma_k: the SCHEDULE ONLY. Prodigy's own d carries the
// magnitude, which is the entire point of using it.
//
// Built as GROUPED graphs rather than one big one. Prodigy needs ~10
// intermediates per parameter where AdamW has a single fused node, and 504
// parameters in one graph exhausts ggml-alloc's free-block list
// (MAX_FREE_BLOCKS) long before the node cap matters.
static bool lm_optim_step_prodigy(LmOptim * o, ggml_backend_sched_t sched, float lr_now, LmStepStats * out) {
    const float  b1  = 0.9f;
    const float  b2  = 0.999f;
    const float  sb2 = sqrtf(b2);
    const float  eps = 1e-8f;
    const double d_k = o->prodigy_d;
    const size_t n   = o->acc.size();
    const size_t GROUP = 32;

    auto tree = [](ggml_context * ctx, std::vector<ggml_tensor *> & v) -> ggml_tensor * {
        while (v.size() > 1) {
            std::vector<ggml_tensor *> nx;
            nx.reserve((v.size() + 1) / 2);
            for (size_t j = 0; j + 1 < v.size(); j += 2) {
                nx.push_back(ggml_add(ctx, v[j], v[j + 1]));
            }
            if (v.size() % 2) {
                nx.push_back(v.back());
            }
            v.swap(nx);
        }
        return v[0];
    };

    // ── pass 1: global ||g||^2, so the clip factor is a host constant ──────
    double gn2 = 0.0;
    for (size_t lo = 0; lo < n; lo += GROUP) {
        const size_t hi = std::min(n, lo + GROUP);
        ggml_init_params ip  = { o->arena.size(), o->arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) {
            return false;
        }
        ggml_cgraph *              go = ggml_new_graph_custom(ctx, 4096, false);
        std::vector<ggml_tensor *> sq;
        sq.reserve(hi - lo);
        for (size_t j = lo; j < hi; j++) {
            sq.push_back(ggml_sum(ctx, ggml_sqr(ctx, o->acc[j])));
        }
        ggml_build_forward_expand(go, ggml_cpy(ctx, tree(ctx, sq), o->t_gnorm2));
        ggml_backend_sched_reset(sched);
        const bool ok = ggml_backend_sched_graph_compute(sched, go) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
        float part = 0.0f;
        ggml_backend_tensor_get(o->t_gnorm2, &part, 0, sizeof(float));
        gn2 += (double) part;
    }
    const float gnorm = (gn2 > 0.0) ? (float) sqrt(gn2) : 0.0f;
    const float clipf = (o->grad_clip > 0.0f) ? std::min(1.0f, o->grad_clip / (gnorm + 1e-6f)) : 1.0f;

    // ── pass 2: m, v, s at d_k + the two reductions ────────────────────────
    const float dk  = (float) d_k;
    const float dk2 = (float) (d_k * d_k);
    double      num = 0.0, s1 = 0.0;
    for (size_t lo = 0; lo < n; lo += GROUP) {
        const size_t hi = std::min(n, lo + GROUP);
        ggml_init_params ip  = { o->arena.size(), o->arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) {
            return false;
        }
        ggml_cgraph *              go = ggml_new_graph_custom(ctx, 4096, false);
        std::vector<ggml_tensor *> nums, l1s;
        nums.reserve(hi - lo);
        l1s.reserve(hi - lo);
        for (size_t j = lo; j < hi; j++) {
            ggml_tensor * g = (clipf != 1.0f) ? ggml_scale(ctx, o->acc[j], clipf) : o->acc[j];

            // WRITE-AFTER-READ: each _new is an ancestor of its own cpy, so no
            // write can be scheduled ahead of a node still reading the old value.
            ggml_tensor * m_new =
                ggml_add(ctx, ggml_scale(ctx, o->mom_m[j], b1), ggml_scale(ctx, g, (1.0f - b1) * dk));
            ggml_tensor * v_new =
                ggml_add(ctx, ggml_scale(ctx, o->mom_v[j], b2), ggml_scale(ctx, ggml_sqr(ctx, g), (1.0f - b2) * dk2));
            ggml_tensor * s_new = ggml_add(ctx, ggml_scale(ctx, o->pg_s[j], sb2),
                                           ggml_scale(ctx, g, (1.0f - sb2) * lr_now * dk2));

            nums.push_back(ggml_sum(ctx, ggml_mul(ctx, g, ggml_sub(ctx, o->pg_x0[j], o->params[j]))));
            l1s.push_back(ggml_sum(ctx, ggml_abs(ctx, s_new)));

            ggml_build_forward_expand(go, ggml_cpy(ctx, m_new, o->mom_m[j]));
            ggml_build_forward_expand(go, ggml_cpy(ctx, v_new, o->mom_v[j]));
            ggml_build_forward_expand(go, ggml_cpy(ctx, s_new, o->pg_s[j]));
        }
        ggml_build_forward_expand(go, ggml_cpy(ctx, tree(ctx, nums), o->t_pnum));
        ggml_build_forward_expand(go, ggml_cpy(ctx, tree(ctx, l1s), o->t_ps1));
        ggml_backend_sched_reset(sched);
        const bool ok = ggml_backend_sched_graph_compute(sched, go) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
        float pn = 0.0f, ps = 0.0f;
        ggml_backend_tensor_get(o->t_pnum, &pn, 0, sizeof(float));
        ggml_backend_tensor_get(o->t_ps1, &ps, 0, sizeof(float));
        num += (double) pn;
        s1  += (double) ps;
    }

    // ── d, on the host. It NEVER decreases: d is a lower bound on 1/L. ─────
    o->prodigy_r = (double) sb2 * o->prodigy_r + (1.0 - (double) sb2) * (double) lr_now * d_k * d_k * num;
    double d_new = d_k;
    if (s1 > 0.0 && std::isfinite(o->prodigy_r)) {
        const double d_hat = o->prodigy_r / s1;
        if (std::isfinite(d_hat) && d_hat > d_new) {
            d_new = d_hat;
        }
    }
    o->prodigy_d = d_new;

    // ── pass 3: the weight update at d_{k+1} ───────────────────────────────
    {
        const float deps = (float) d_new * eps;
        ggml_backend_tensor_set(o->t_pdeps, &deps, 0, sizeof(float));
    }
    const float step = lr_now * (float) d_new;
    for (size_t lo = 0; lo < n; lo += GROUP) {
        const size_t hi = std::min(n, lo + GROUP);
        ggml_init_params ip  = { o->arena.size(), o->arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) {
            return false;
        }
        ggml_cgraph * go = ggml_new_graph_custom(ctx, 4096, false);
        for (size_t j = lo; j < hi; j++) {
            // t_pdeps is [1] and broadcasts (ggml_add1 is deprecated in favour
            // of exactly this).
            ggml_tensor * den = ggml_add(ctx, ggml_sqrt(ctx, o->mom_v[j]), o->t_pdeps);
            ggml_tensor * upd = ggml_div(ctx, o->mom_m[j], den);
            ggml_tensor * cur = (o->weight_decay > 0.0f)
                                    ? ggml_scale(ctx, o->params[j], 1.0f - step * o->weight_decay)
                                    : o->params[j];
            ggml_build_forward_expand(go, ggml_cpy(ctx, ggml_sub(ctx, cur, ggml_scale(ctx, upd, step)), o->params[j]));
        }
        ggml_backend_sched_reset(sched);
        const bool ok = ggml_backend_sched_graph_compute(sched, go) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }

    // Report the EFFECTIVE step, gamma*d — reporting gamma alone would show a
    // flat 1.0 for the whole run and tell you nothing.
    out->lr        = step;
    out->grad_norm = gnorm;
    out->clip      = clipf;
    lm_optim_zero_grad(o);
    o->opt_step++;
    o->opt_iter++;
    return true;
}

static bool lm_optim_step(LmOptim * o, ggml_backend_sched_t sched, LmStepStats * out) {
    if (o->optimizer == "prodigy") {
        if (o->acc.empty()) {
            return false;
        }
        const float g_k = o->base_lr * lm_lr_lambda(o->opt_step, o->total_steps, o->warmup_steps, o->lr_floor);
        return lm_optim_step_prodigy(o, sched, g_k, out);
    }
    ggml_init_params ip  = { o->arena.size(), o->arena.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    ggml_cgraph * go = ggml_new_graph_custom(ctx, (size_t) (o->est_nodes > 0 ? o->est_nodes : 8192), /*grads=*/false);

    // (a) global squared L2 norm of every accumulator.
    //
    // Reduced as a BALANCED TREE, not the plan's left-fold: a left-fold is a
    // 392-deep serial chain of single-element adds with no available parallelism
    // in its tail, i.e. 392 dependent kernel launches per optimizer step (twice
    // that on 1.7B). Same node count (n-1 adds), same result, depth log2(n).
    if (o->acc.empty()) {
        ggml_free(ctx);
        return false;
    }
    std::vector<ggml_tensor *> sums;
    sums.reserve(o->acc.size());
    for (size_t j = 0; j < o->acc.size(); j++) {
        sums.push_back(ggml_sum(ctx, ggml_sqr(ctx, o->acc[j])));  // scalar
    }
    while (sums.size() > 1) {
        std::vector<ggml_tensor *> next;
        next.reserve((sums.size() + 1) / 2);
        for (size_t j = 0; j + 1 < sums.size(); j += 2) {
            next.push_back(ggml_add(ctx, sums[j], sums[j + 1]));
        }
        if (sums.size() % 2) {
            next.push_back(sums.back());
        }
        sums.swap(next);
    }
    ggml_tensor * gn2 = sums[0];
    ggml_build_forward_expand(go, ggml_cpy(ctx, gn2, o->t_gnorm2));  // 4-byte readback, LOGGING ONLY

    // (b) clip factor, entirely in-graph: c = clamp(clip / (sqrt(gn2) + eps), 0, 1)
    ggml_tensor * c = nullptr;
    if (o->grad_clip > 0.0f) {
        c = ggml_clamp(ctx, ggml_div(ctx, o->t_clip, ggml_add(ctx, ggml_sqrt(ctx, gn2), o->t_eps)), 0.0f, 1.0f);
    }

    // (c) the update, per parameter, by rule
    const float lr_now = o->base_lr * lm_lr_lambda(o->opt_step, o->total_steps, o->warmup_steps, o->lr_floor);
    const float mu_wd  = (o->muon.wd >= 0.0f) ? o->muon.wd : o->weight_decay;
    for (size_t j = 0; j < o->acc.size(); j++) {
        ggml_tensor * g = c ? ggml_mul(ctx, o->acc[j], c) : o->acc[j];  // [1] broadcasts

        if (o->rule[j] != LM_RULE_MUON) {
            const bool    alt = (o->lr_mul[j] != 1.0f);
            ggml_tensor * pa  = alt ? o->t_adamw_alt : o->t_adamw;
            GGML_ASSERT(pa && "a parameter has lr_mul != 1 but no t_adamw_alt was allocated");
            ggml_build_forward_expand(go, ggml_opt_step_adamw(ctx, o->params[j], g, o->mom_m[j], o->mom_v[j], pa));
        }
        // Muon parameters are handled below, a whole shape bucket at a time.
    }

    // ── Muon, bucketed ───────────────────────────────────────────────────────
    //
    // Per bucket: concat the accumulators and momenta into [ne0, ne1, N], run
    // ONE batched Newton-Schulz, then scatter back. The concat is what makes the
    // gather correct by construction — writing N slabs of a scratch tensor with
    // N independent cpy nodes would have no dependency edges, and nothing would
    // stop the scheduler running the Newton-Schulz before the last of them.
    for (size_t b = 0; b < o->muon_buckets.size(); b++) {
        const LmMuonBucket & bk = o->muon_buckets[b];
        if (bk.idx.empty()) {
            continue;
        }
        ggml_tensor * G = nullptr;
        ggml_tensor * M = nullptr;
        for (size_t i = 0; i < bk.idx.size(); i++) {
            const int     j  = bk.idx[i];
            ggml_tensor * gj = c ? ggml_mul(ctx, o->acc[(size_t) j], c) : o->acc[(size_t) j];
            G                = G ? ggml_concat(ctx, G, gj, 2) : gj;
            M                = M ? ggml_concat(ctx, M, o->mom_m[(size_t) j], 2) : o->mom_m[(size_t) j];
        }

        // WRITE-AFTER-READ ORDER IS LOAD-BEARING: `m_new` is computed from the
        // OLD momenta and is an ancestor of every write-back below, so no cpy
        // into mom_m can be scheduled ahead of a node that still needs the old
        // value. ggml_opt_step_adamw sidesteps this by fusing; there is no fused
        // op here, so the dependency is built deliberately.
        ggml_tensor * m_new = ggml_add(ctx, ggml_scale(ctx, M, o->muon.momentum), G);
        ggml_tensor * u     = o->muon.nesterov ? ggml_add(ctx, G, ggml_scale(ctx, m_new, o->muon.momentum)) : m_new;
        ggml_tensor * ortho = lm_muon_ns_batched(ctx, u, o->muon.ns_steps, o->t_eps);

        // Reference scale sqrt(max(1, rows/cols)) — identical for every member of
        // the bucket, since a bucket is one exact shape, so it folds into a
        // single scale over the whole batch.
        const float rows  = (float) bk.ne1;
        const float cols  = (float) bk.ne0;
        const float shape = sqrtf(rows > cols ? rows / cols : 1.0f);
        ortho             = ggml_scale(ctx, ortho, -lr_now * o->muon.lr_scale * shape);

        const size_t slab = (size_t) bk.ne0 * (size_t) bk.ne1 * sizeof(float);
        for (size_t i = 0; i < bk.idx.size(); i++) {
            const int     j  = bk.idx[i];
            ggml_tensor * Ov = ggml_view_2d(ctx, ortho, bk.ne0, bk.ne1, (size_t) bk.ne0 * sizeof(float), i * slab);
            ggml_tensor * Mv = ggml_view_2d(ctx, m_new, bk.ne0, bk.ne1, (size_t) bk.ne0 * sizeof(float), i * slab);
            ggml_tensor * decayed =
                (mu_wd > 0.0f) ? ggml_scale(ctx, o->params[(size_t) j], 1.0f - lr_now * o->muon.lr_scale * mu_wd)
                               : o->params[(size_t) j];
            ggml_build_forward_expand(go, ggml_cpy(ctx, ggml_add(ctx, decayed, Ov), o->params[(size_t) j]));
            ggml_build_forward_expand(go, ggml_cpy(ctx, Mv, o->mom_m[(size_t) j]));
        }
    }

    // host side, immediately BEFORE running `go`. `lr` is the same value the
    // Muon branch above already baked into its ggml_scale constants — one
    // schedule drives both rules, so a mixed run cannot drift between classes.
    o->opt_iter++;  // 1-based, matches ggml-opt's opt_ctx->iter
    const float lr = lr_now;
    const float p7[7] = { lr,
                          0.9f,
                          0.999f,
                          1e-8f,
                          o->weight_decay,
                          1.0f / (1.0f - powf(0.9f, (float) o->opt_iter)),
                          1.0f / (1.0f - powf(0.999f, (float) o->opt_iter)) };
    ggml_backend_tensor_set(o->t_adamw, p7, 0, sizeof(p7));
    if (o->t_adamw_alt) {
        // One alternate multiplier. If two parameters disagree that is a wiring
        // bug, not a configuration — say so instead of silently using one.
        float mul = 1.0f;
        for (size_t j = 0; j < o->lr_mul.size(); j++) {
            if (o->lr_mul[j] != 1.0f) {
                GGML_ASSERT((mul == 1.0f || mul == o->lr_mul[j]) && "more than one distinct lr_mul; only one alt group");
                mul = o->lr_mul[j];
            }
        }
        float p7a[7];
        memcpy(p7a, p7, sizeof(p7a));
        p7a[0] = lr * mul;
        ggml_backend_tensor_set(o->t_adamw_alt, p7a, 0, sizeof(p7a));
    }

    ggml_backend_sched_reset(sched);
    const bool ok = ggml_backend_sched_graph_compute(sched, go) == GGML_STATUS_SUCCESS;

    float gn2v = 0.0f;
    ggml_backend_tensor_get(o->t_gnorm2, &gn2v, 0, sizeof(float));
    const float gnorm = (gn2v > 0.0f) ? sqrtf(gn2v) : 0.0f;

    out->lr        = lr;
    out->grad_norm = gnorm;
    out->clip      = (o->grad_clip > 0.0f) ? std::min(1.0f, o->grad_clip / (gnorm + 1e-6f)) : 1.0f;

    lm_optim_zero_grad(o);
    o->opt_step++;
    ggml_free(ctx);
    return ok;
}
