#pragma once
// lokr-apply.h — the LoKr forward delta, contracted factor by factor.
//
// Extracted from train/dit-adapter-lokr.h (2026-09-22) when the YuE2 joint
// trainer needed the same maths. lokr-common.h owns the SHAPE rules (how a
// dimension splits, whether w2 factorizes); this header owns the GRAPH: given
// x and a site's factors, build delta = kron(w1, w2) x without ever
// materializing the kron. Every LoKr trainer must call this, never re-derive
// it — a trainer/loader disagreement here loads fine and computes something
// else.
//
// Conventions (LyCORIS parity, see dit-adapter-lokr.h's header):
//   torch w1 [out_l, in_m], w2 [out_k, in_n]     dW = kron(w1, w2) * scale
//   dW[l*out_k + k, m*in_n + n] = scale * w1[l,m] * w2[k,n]
//   ggml stores the transpose: w1 [ne0=in_m, ne1=out_l], w2 [ne0=in_n, ne1=out_k],
//   factorized w2 = w2_a @ w2_b with w2_a [ne0=dim, ne1=out_k], w2_b [ne0=in_n, ne1=dim].
//
// THE TOKEN AXIS MUST NOT REACH ne2 OF A MUL_MAT HERE (measured 2026-07-30).
// Both contractions keep a 2-D trainable factor (ne2 == 1) as src0 and fold
// the token count into src1's ne1. If src1 carried S in ne2, ggml's backward
// would emit the weight gradient as out_prod with dst->ne[2] == S, and
// ggml-cuda's out_prod takes its per-token fallback there: one cublasSgemm per
// token plus a repeat_back, which made LoKr training 16.4x slower than LoRA
// (docs/plans/2026-07-30-dit-trainer-step-profile.md). [in_n, in_m, S] and
// [in_n, in_m*S] are the same bytes, so the fold is a pure reshape.
//
// The two axis swaps cannot be deleted, only placed (2026-09-02): ggml_mul_mat
// contracts ne0 of both operands and writes src0->ne1 to the result's ne0, and
// the operand carrying the tokens can only be src1. delta must come out as
// [out_k, out_l*S] (k fastest, since the kron row index is l*out_k + k), so the
// LAST contraction has w2 as src0 and needs its index at src1's ne0 while the
// w1 leg already put out_l there — one swap. x itself arrives with the w2
// index at ne0 (row-major m*in_n + n), so whichever leg runs first but is not
// the w2 leg needs it transposed — the other swap. LokrOrder picks the
// cheapest place to pay them.

#include "../ggml/include/ggml.h"

#include <cstdint>

enum LokrOrder {
    LOKR_ORDER_W2  = 0,  // w2 (or w2_b,w2_a) then w1        — cut at out_k
    LOKR_ORDER_W1  = 1,  // w1 then w2 (or w2_b,w2_a)        — cut at in_n
    LOKR_ORDER_MID = 2,  // w2_b then w1 then w2_a           — cut at dim
};

static inline const char * lokr_order_name(int o) {
    return o == LOKR_ORDER_W1 ? "w1-first" : (o == LOKR_ORDER_MID ? "mid" : "w2-first");
}

// Matmul MACs per token for one order, used only as the tie-break below.
static inline double lokr_order_flops(int order, bool mono, int64_t in_m, int64_t in_n, int64_t out_l,
                                      int64_t out_k, int64_t dim) {
    const double M = (double) in_m, N = (double) in_n, L = (double) out_l, K = (double) out_k, D = (double) dim;
    if (order == LOKR_ORDER_W1) {
        return mono ? (M * L * N + L * K * N) : (M * L * N + L * D * N + L * K * D);
    }
    if (order == LOKR_ORDER_MID) {
        return M * D * N + M * L * D + L * K * D;
    }
    return mono ? (M * K * N + M * L * K) : (M * D * N + M * K * D + M * L * K);
}

// Cut channel count: the width of BOTH transposes, so the copy cost is
// (in_m + out_l) * cut.
static inline int64_t lokr_order_cut(int order, int64_t in_n, int64_t out_k, int64_t dim) {
    return order == LOKR_ORDER_W1 ? in_n : (order == LOKR_ORDER_MID ? dim : out_k);
}

// Smallest cut wins; equal cuts go to the lower flop count.
static inline int lokr_pick_order(bool mono, int64_t in_m, int64_t in_n, int64_t out_l, int64_t out_k,
                                  int64_t dim) {
    const int cands[3]   = { LOKR_ORDER_W2, LOKR_ORDER_W1, LOKR_ORDER_MID };
    int       best       = LOKR_ORDER_W2;
    int64_t   best_cut   = out_k;
    double    best_flops = lokr_order_flops(LOKR_ORDER_W2, mono, in_m, in_n, out_l, out_k, dim);
    for (int i = 1; i < 3; i++) {
        if (cands[i] == LOKR_ORDER_MID && mono) {
            continue;  // no w2_a/w2_b to cut between
        }
        const int64_t cut = lokr_order_cut(cands[i], in_n, out_k, dim);
        const double  fl  = lokr_order_flops(cands[i], mono, in_m, in_n, out_l, out_k, dim);
        if (cut < best_cut || (cut == best_cut && fl < best_flops)) {
            best       = cands[i];
            best_cut   = cut;
            best_flops = fl;
        }
    }
    return best;
}

// Everything lokr_apply_delta needs to know about one site. Trainers embed or
// derive this from their own site records; `mono` is `w2 != nullptr`.
struct LokrApplySite {
    ggml_tensor * w1   = nullptr;  // ggml [in_m, out_l]
    ggml_tensor * w2   = nullptr;  // ggml [in_n, out_k]   monolithic
    ggml_tensor * w2_a = nullptr;  // ggml [dim,  out_k]   factorized
    ggml_tensor * w2_b = nullptr;  // ggml [in_n, dim]
    int64_t       out_l = 0, out_k = 0, in_m = 0, in_n = 0;
    int64_t       dim   = 0;
    bool          mono  = true;
    int           order = LOKR_ORDER_W2;  // decide once with lokr_pick_order
    float         scale = 1.0f;

    bool valid() const { return w1 && (mono ? w2 != nullptr : (w2_a && w2_b)) && out_l > 0 && out_k > 0 && in_m > 0 && in_n > 0; }
};

// Exchange the two leading axes of `t`, read as [a, b, S], and hand back the
// [b, a*S] fold the next mul_mat wants. The 3-D form exists only so
// ggml_permute(1, 0, 2, 3) — src axis0 to position 1, axis1 to position 0 —
// is expressible. ggml_cont_2d rather than cont + reshape: CONT's backward
// already reshapes the gradient when the shapes differ.
static inline ggml_tensor * lokr_swap(ggml_context * ctx, ggml_tensor * t, int64_t a, int64_t b, int64_t S) {
    ggml_tensor * t3 = ggml_reshape_3d(ctx, t, a, b, S);
    return ggml_cont_2d(ctx, ggml_permute(ctx, t3, 1, 0, 2, 3), b, a * S);
}

// x [in_m*in_n, ...] with every trailing axis a token axis. Returns the delta
// as [out_k, out_l*S] — the same bytes as the row-major [out, S] delta, so the
// caller reshapes it onto its base output and adds. Gradients come free from
// ggml_build_backward_expand; nothing here is DiT- or YuE2-shaped.
static inline ggml_tensor * lokr_apply_delta(ggml_context * ctx, ggml_tensor * x, const LokrApplySite & k) {
    ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    const int64_t S  = ggml_nelements(xc) / xc->ne[0];
    // x * 1.0f is exact in IEEE-754, so skipping the node at scale 1 is a
    // bit-identical graph shortening (and it is 1 on every production DiT run).
    ggml_tensor * w1s = (k.scale == 1.0f) ? k.w1 : ggml_scale(ctx, k.w1, k.scale);
    ggml_tensor * d   = nullptr;  // [out_k, out_l*S] when done

    if (k.order == LOKR_ORDER_W1) {
        // x --swap--> [in_m, in_n*S] --w1--> [out_l, in_n*S] --swap-->
        // [in_n, out_l*S] --w2--> [out_k, out_l*S].
        ggml_tensor * Xf = lokr_swap(ctx, xc, k.in_n, k.in_m, S);   // [in_m, in_n*S]
        ggml_tensor * U  = ggml_mul_mat(ctx, w1s, Xf);              // [out_l, in_n*S]
        ggml_tensor * Uf = lokr_swap(ctx, U, k.out_l, k.in_n, S);   // [in_n, out_l*S]
        d = k.mono ? ggml_mul_mat(ctx, k.w2, Uf)
                   : ggml_mul_mat(ctx, k.w2_a, ggml_mul_mat(ctx, k.w2_b, Uf));
    } else if (k.order == LOKR_ORDER_MID) {
        // x --w2_b--> [dim, in_m*S] --swap--> [in_m, dim*S] --w1-->
        // [out_l, dim*S] --swap--> [dim, out_l*S] --w2_a--> [out_k, out_l*S].
        ggml_tensor * X2 = ggml_reshape_2d(ctx, xc, k.in_n, k.in_m * S);
        ggml_tensor * Z  = ggml_mul_mat(ctx, k.w2_b, X2);           // [dim, in_m*S]
        ggml_tensor * Zf = lokr_swap(ctx, Z, k.dim, k.in_m, S);     // [in_m, dim*S]
        ggml_tensor * V  = ggml_mul_mat(ctx, w1s, Zf);              // [out_l, dim*S]
        ggml_tensor * Vf = lokr_swap(ctx, V, k.out_l, k.dim, S);    // [dim, out_l*S]
        d                = ggml_mul_mat(ctx, k.w2_a, Vf);           // [out_k, out_l*S]
    } else {
        // x --w2--> [out_k, in_m*S] --swap--> [in_m, out_k*S] --w1-->
        // [out_l, out_k*S] --swap--> [out_k, out_l*S].
        ggml_tensor * X2 = ggml_reshape_2d(ctx, xc, k.in_n, k.in_m * S);  // same bytes as [in_n, in_m, S]
        ggml_tensor * T1 = k.mono ? ggml_mul_mat(ctx, k.w2, X2)
                                  : ggml_mul_mat(ctx, k.w2_a, ggml_mul_mat(ctx, k.w2_b, X2));
        ggml_tensor * T1f = lokr_swap(ctx, T1, k.out_k, k.in_m, S);  // [in_m, out_k*S]
        ggml_tensor * T2  = ggml_mul_mat(ctx, w1s, T1f);             // [out_l, out_k*S]
        d                 = lokr_swap(ctx, T2, k.out_l, k.out_k, S); // [out_k, out_l*S]
    }
    return d;
}
