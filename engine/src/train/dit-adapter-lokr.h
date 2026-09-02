#pragma once
// dit-adapter-lokr.h — the LoKR parameterization behind the D22 seam
// (docs/plans/2026-07-28-lokr-dit-training.md §2).
//
// LyCORIS parity, exactly:
//   (out_l, out_k) = factorization(out, factor)   (a, c) in LyCORIS terms
//   (in_m, in_n)   = factorization(in,  factor)   (b, d)
//   torch w1 [out_l, in_m], w2 [out_k, in_n]      dW = kron(w1, w2) * scale
//   dW[l*out_k + k, m*in_n + n] = scale * w1[l,m] * w2[k,n]
//
// ggml stores the transpose of the torch row-major view, so w1 lives as
// [ne0=in_m, ne1=out_l] and w2 as [ne0=in_n, ne1=out_k] — the same convention
// adapter-merge.h's LoKr path builds its kron from.
//
// dW is NEVER materialized in the training graph (K7): a per-site 6144x2048 F32
// delta is 50 MB that the backward pass would have to retain. apply() contracts
// against x directly (§2.3) and its retained intermediates scale with the
// activations instead.

#include "lokr-common.h"
#include "train/dit-adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ─── LyCORIS factorization (lycoris/functional/general.py) ──────────────────
//
// The `while (m < n)` loop guard is load-bearing: without it the sweep walks
// past the crossover to m = dimension, n = 1 and then spins forever looking for
// a divisor above `dimension`. The plan's §1 pseudocode drops it; LyCORIS has
// it, and (2048, -1) -> (32, 64) only comes out with it.
// Moved to engine/src/lokr-common.h (2026-07-30) so the LM trainer uses the
// SAME rules — a trainer/loader disagreement here is not a crash, it is an
// adapter that loads and quietly computes something else.
static inline void dit_lokr_factorization(int64_t dimension, int factor, int64_t * out_m, int64_t * out_n) {
    lokr_factorization(dimension, factor, out_m, out_n);
}

// LyCORIS keeps w2 monolithic unless the factorized form is strictly cheaper.
static inline bool dit_lokr_w2_mono(int dim, int64_t out_k, int64_t in_n) {
    return lokr_w2_mono(dim, out_k, in_n);
}

// K3: HOT-Step's loaders read only `lokr_w1`, so w1 is always monolithic. This
// reports whether LyCORIS WOULD have factorized it, so the override can warn.
static inline bool dit_lokr_w1_would_factor(int dim, bool decompose_both, int64_t out_l, int64_t in_m) {
    return decompose_both && ((double) dim < (double) std::max(out_l, in_m) / 2.0);
}

// ─── contraction order (2026-09-02) ─────────────────────────────────────────
//
// apply() contracts kron(w1,w2)x factor by factor. ggml_mul_mat contracts ne0
// and puts the freshly produced index BACK at ne0, so every time the chain
// switches from the w2 leg to the w1 leg (or back) the running tensor has to be
// transposed — one ggml_cont(ggml_permute(...)) full-activation copy each, and
// its mirror in the backward (ggml's RESHAPE backward conts the permuted grad).
//
// The chain has exactly one leg switch, so the transpose count is fixed at TWO
// (see the "why not fewer" note on apply()). What is NOT fixed is WHERE the
// switch happens, and that decides the SIZE of both copies. Cutting between
// stages c and c+1 makes both copies (in_m + out_l) * C wide, where C is the
// channel count at the cut:
//
//   W2 first (out_k):  x --w2--> [out_k] --swap--> --w1--> --swap--> delta
//   MID     (dim):     x --w2_b--> [dim] --swap--> --w1--> --swap--> --w2_a-->
//   W1 first (in_n):   x --swap--> --w1--> --swap--> --w2--> delta
//
// so the cheapest order is simply the one with the SMALLEST channel count at
// the cut: min(out_k, in_n) monolithic, min(out_k, dim, in_n) factorized. Ties
// go to the lower flop count. On dit-xl-thirds at dim 512 / factor 6 that flips
// every H-input site whose output factorizes with out_l < in_m — self/cross
// q_proj (out_k 1024 -> in_n 512) and gate/up (out_k 2432 -> in_n 512) — and
// leaves o_proj / down_proj / k_proj / v_proj where they were. Measured effect
// on the adapter branch: copy elements per token per layer 80640 -> 36864
// (-54 %) and its matmul MACs -15 %, because the same reorder that shrinks the
// copies also shrinks the intermediate the big GEMM writes.
//
// EVERY order keeps a 2-D trainable factor (ne2 == 1) as mul_mat's src0 and the
// token axis folded into ne1 — the two invariants apply()'s note below explains
// are load-bearing. Reordering changes float summation order only, which LK3's
// materialised-kron reference tolerates.
enum DitLokrOrder {
    DIT_LOKR_ORDER_W2 = 0,  // w2 (or w2_b,w2_a) then w1        — cut at out_k
    DIT_LOKR_ORDER_W1 = 1,  // w1 then w2 (or w2_b,w2_a)        — cut at in_n
    DIT_LOKR_ORDER_MID = 2, // w2_b then w1 then w2_a           — cut at dim
};

static inline const char * dit_lokr_order_name(int o) {
    return o == DIT_LOKR_ORDER_W1 ? "w1-first" : (o == DIT_LOKR_ORDER_MID ? "mid" : "w2-first");
}

// Matmul MACs per token for one order, used only as the tie-break below.
static inline double dit_lokr_order_flops(int order, bool mono, int64_t in_m, int64_t in_n, int64_t out_l,
                                          int64_t out_k, int64_t dim) {
    const double M = (double) in_m, N = (double) in_n, L = (double) out_l, K = (double) out_k, D = (double) dim;
    if (order == DIT_LOKR_ORDER_W1) {
        return mono ? (M * L * N + L * K * N) : (M * L * N + L * D * N + L * K * D);
    }
    if (order == DIT_LOKR_ORDER_MID) {
        return M * D * N + M * L * D + L * K * D;
    }
    return mono ? (M * K * N + M * L * K) : (M * D * N + M * K * D + M * L * K);
}

// Cut channel count: the width of BOTH transposes, so the copy cost is
// (in_m + out_l) * cut.
static inline int64_t dit_lokr_order_cut(int order, int64_t in_n, int64_t out_k, int64_t dim) {
    return order == DIT_LOKR_ORDER_W1 ? in_n : (order == DIT_LOKR_ORDER_MID ? dim : out_k);
}

static inline int dit_lokr_pick_order(bool mono, int64_t in_m, int64_t in_n, int64_t out_l, int64_t out_k,
                                      int64_t dim) {
    const int cands[3]   = { DIT_LOKR_ORDER_W2, DIT_LOKR_ORDER_W1, DIT_LOKR_ORDER_MID };
    int       best       = DIT_LOKR_ORDER_W2;
    int64_t   best_cut   = out_k;
    double    best_flops = dit_lokr_order_flops(DIT_LOKR_ORDER_W2, mono, in_m, in_n, out_l, out_k, dim);
    for (int i = 1; i < 3; i++) {
        if (cands[i] == DIT_LOKR_ORDER_MID && mono) {
            continue;  // no w2_a/w2_b to cut between
        }
        const int64_t cut = dit_lokr_order_cut(cands[i], in_n, out_k, dim);
        const double  fl  = dit_lokr_order_flops(cands[i], mono, in_m, in_n, out_l, out_k, dim);
        if (cut < best_cut || (cut == best_cut && fl < best_flops)) {
            best       = cands[i];
            best_cut   = cut;
            best_flops = fl;
        }
    }
    return best;
}

// Retained apply() intermediates for one site, ELEMENTS PER TOKEN — every
// tensor the chain allocates, counted once. Shared by the two arena estimates
// below so the VRAM model can never disagree with the graph about which order
// ran. For DIT_LOKR_ORDER_W2 this reproduces the pre-2026-09-02 arithmetic
// (2*(out_k*in_m + out_l*out_k), plus dim*in_m when w2 is factorized) exactly.
static inline double dit_lokr_site_retained_elems(int order, bool mono, int64_t in_m, int64_t in_n, int64_t out_l,
                                                  int64_t out_k, int64_t dim) {
    const double M = (double) in_m, N = (double) in_n, L = (double) out_l, K = (double) out_k, D = (double) dim;
    if (order == DIT_LOKR_ORDER_W1) {
        // Xp (cont) + U + Up (cont) + delta, plus the factorized inner [dim, out_l*S].
        return M * N + 2.0 * L * N + L * K + (mono ? 0.0 : L * D);
    }
    if (order == DIT_LOKR_ORDER_MID) {
        // Z + Zp (cont) + V + Vp (cont) + delta.
        return 2.0 * D * M + 2.0 * D * L + L * K;
    }
    // W2: T1 + T1p (cont) + T2 + T2p (cont), plus the factorized inner [dim, in_m*S].
    return 2.0 * (K * M + L * K) + (mono ? 0.0 : D * M);
}

// The same geometry counted the way dit_vram_lokr_apply_bytes_flash() needs it:
// ONE of each cont/source pair, because ggml_gallocr reuses the source block
// once the cont has consumed it. That is the convention the fitted
// DIT_FLASH_LOKR_RETENTION was measured against, so the W2 arm below is
// deliberately byte-identical to the expression it replaces.
static inline double dit_lokr_site_live_elems(int order, bool mono, int64_t in_m, int64_t in_n, int64_t out_l,
                                              int64_t out_k, int64_t dim) {
    const double M = (double) in_m, N = (double) in_n, L = (double) out_l, K = (double) out_k, D = (double) dim;
    if (order == DIT_LOKR_ORDER_W1) {
        return M * N + L * N + L * K + (mono ? 0.0 : L * D);
    }
    if (order == DIT_LOKR_ORDER_MID) {
        return D * M + D * L + L * K;
    }
    return K * M + L * K + (mono ? 0.0 : D * M);
}

// ─── per-site state ─────────────────────────────────────────────────────────

struct DitLokrSite {
    ggml_tensor * w1   = nullptr;  // ggml [in_m, out_l]
    ggml_tensor * w2   = nullptr;  // ggml [in_n, out_k]   monolithic
    ggml_tensor * w2_a = nullptr;  // ggml [dim,  out_k]   factorized
    ggml_tensor * w2_b = nullptr;  // ggml [in_n, dim]
    int64_t       out_l = 0, out_k = 0, in_m = 0, in_n = 0;
    bool          mono  = true;
    int           order = DIT_LOKR_ORDER_W2;  // decided once in init(), see dit_lokr_pick_order
    float         scale = 1.0f, alpha_eff = 0.0f;
};

// ─── LoKR ───────────────────────────────────────────────────────────────────

struct DitAdapterLoKr final : DitAdapter {
    ggml_context *                                   ctx = nullptr;
    ggml_backend_buffer_t                            buf = nullptr;
    std::vector<std::array<DitLokrSite, DIT_NSITES>> layers;  // indexed by (layer - lo)
    std::vector<ggml_tensor *>                       par;
    int    lo = 0, hi = 0, dim = 0, factor = 0, n_sites = DIT_NSITES_ATTN;
    float  alpha           = 0.0f;
    bool   decompose_both  = true;
    size_t n_params        = 0;

    const std::vector<ggml_tensor *> & params() const override { return par; }
    size_t                             nParams() const override { return n_params; }
    int                                nSites() const override { return n_sites; }
    const char *                       typeName() const override { return "lokr"; }

    void free() override {
        if (buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        layers.clear();
        par.clear();
        n_params = 0;
    }

    const DitLokrSite * site(int layer, int s) const {
        if (layer < lo || layer >= hi || s < 0 || s >= n_sites) {
            return nullptr;
        }
        const DitLokrSite & k = layers[(size_t) (layer - lo)][(size_t) s];
        return (k.w1 && (k.w2 || (k.w2_a && k.w2_b))) ? &k : nullptr;
    }

    bool init(DiTGGML * m, ggml_backend_t backend, int lo_, int hi_, const DitAdapterCfg & cfg,
              std::string * err) override {
        lo             = lo_;
        hi             = hi_;
        dim            = cfg.lokr_dim;
        factor         = cfg.lokr_factor;
        alpha          = cfg.lokr_alpha;
        decompose_both = cfg.lokr_decompose_both;
        n_sites        = cfg.target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
        layers.assign((size_t) (hi - lo), std::array<DitLokrSite, DIT_NSITES>{});

        const int n_ten = (hi - lo) * n_sites * 3;
        {
            ggml_init_params p = { (size_t) (n_ten + 8) * ggml_tensor_overhead(), nullptr, true };
            ctx                = ggml_init(p);
        }
        if (!ctx) {
            *err = "cannot create the LoKR context";
            return false;
        }

        int w1_forced = 0, alpha_forced = 0, n_mono = 0, n_fact = 0;
        int n_order[3] = { 0, 0, 0 };
        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                ggml_tensor * w = dit_site_weight(&m->layers[l], s);
                if (!w) {
                    char b[128];
                    snprintf(b, sizeof(b), "layer %d site %s has no weight", l, dit_site_peft(s));
                    *err = b;
                    return false;
                }
                DitLokrSite & k = layers[(size_t) (l - lo)][(size_t) s];
                dit_lokr_factorization(w->ne[1], factor, &k.out_l, &k.out_k);
                dit_lokr_factorization(w->ne[0], factor, &k.in_m, &k.in_n);
                k.mono = dit_lokr_w2_mono(dim, k.out_k, k.in_n);
                if (dit_lokr_w1_would_factor(dim, decompose_both, k.out_l, k.in_m)) {
                    w1_forced++;  // K3: monolithic anyway, the loaders read no w1_a/w1_b
                }
                // K6: LyCORIS forces alpha = dim whenever BOTH factors are
                // monolithic, which makes the in-graph scale exactly 1.
                float a_eff = (alpha == 0.0f) ? (float) dim : alpha;
                if (k.mono) {
                    if (a_eff != (float) dim) {
                        alpha_forced++;
                    }
                    a_eff = (float) dim;
                }
                k.alpha_eff = a_eff;
                k.scale     = a_eff / (float) dim;
                k.order     = dit_lokr_pick_order(k.mono, k.in_m, k.in_n, k.out_l, k.out_k, dim);
                n_order[k.order]++;
                if (k.mono) {
                    n_mono++;
                } else {
                    n_fact++;
                }

                char nm[128];
                k.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_m, k.out_l);
                snprintf(nm, sizeof(nm), "L%d.%s.lokr_w1", l, dit_site_peft(s));
                ggml_set_name(k.w1, nm);
                ggml_set_param(k.w1);
                par.push_back(k.w1);
                if (k.mono) {
                    k.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_n, k.out_k);
                    snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2", l, dit_site_peft(s));
                    ggml_set_name(k.w2, nm);
                    ggml_set_param(k.w2);
                    par.push_back(k.w2);
                } else {
                    k.w2_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, k.out_k);
                    k.w2_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k.in_n, dim);
                    snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2_a", l, dit_site_peft(s));
                    ggml_set_name(k.w2_a, nm);
                    snprintf(nm, sizeof(nm), "L%d.%s.lokr_w2_b", l, dit_site_peft(s));
                    ggml_set_name(k.w2_b, nm);
                    ggml_set_param(k.w2_a);
                    ggml_set_param(k.w2_b);
                    par.push_back(k.w2_a);
                    par.push_back(k.w2_b);
                }

                // Exportability is CHECKED, not assumed: kron(w1, w2) has to land
                // exactly on the base weight, or the adapter is unloadable.
                if (k.out_l * k.out_k != w->ne[1] || k.in_m * k.in_n != w->ne[0]) {
                    char b[240];
                    snprintf(b, sizeof(b),
                             "layer %d site %s does not factorize: base[%lld,%lld] but kron(w1[%lld,%lld], "
                             "w2[%lld,%lld]) = [%lld,%lld]",
                             l, dit_site_peft(s), (long long) w->ne[0], (long long) w->ne[1], (long long) k.out_l,
                             (long long) k.in_m, (long long) k.out_k, (long long) k.in_n,
                             (long long) (k.in_m * k.in_n), (long long) (k.out_l * k.out_k));
                    *err = b;
                    return false;
                }
            }
        }

        // LoKR tensors live in their OWN backend buffer, never the mirror's:
        // ggml_opt_step_adamw writes into them and the mirror is USAGE_WEIGHTS.
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            *err = "LoKR parameter buffer allocation failed";
            return false;
        }

        // Init (§1): monolithic w2 -> 0, factorized w2_a kaiming / w2_b -> 0, w1
        // kaiming_uniform(a=sqrt(5)) == U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        // Drawn in param order so the stream is reproducible from the seed alone.
        LmRng rng;
        lm_rng_seed(&rng, cfg.seed);
        std::vector<float> vals;
        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                DitLokrSite & k = layers[(size_t) (l - lo)][(size_t) s];
                fill_kaiming(&rng, k.w1, &vals);
                if (k.mono) {
                    fill_zero(&rng, k.w2, cfg.b_sigma, &vals);
                } else {
                    fill_kaiming(&rng, k.w2_a, &vals);
                    fill_zero(&rng, k.w2_b, cfg.b_sigma, &vals);
                }
            }
        }
        for (size_t i = 0; i < par.size(); i++) {
            n_params += (size_t) ggml_nelements(par[i]);
        }

        if (w1_forced > 0) {
            char b[224];
            snprintf(b, sizeof(b),
                     "LoKR dim %d would factorize w1 on %d site(s); forced monolithic — the engine's adapter "
                     "loaders read lokr_w1 only (K3)",
                     dim, w1_forced);
            lm_log("warn", b);
        }
        if (alpha_forced > 0) {
            char b[224];
            snprintf(b, sizeof(b),
                     "LoKR alpha %.4g overridden to dim %d on %d monolithic site(s) — LyCORIS forces scale 1 when "
                     "both factors are monolithic (K6)",
                     (double) alpha, dim, alpha_forced);
            lm_log("info", b);
        }
        fprintf(stderr, "[train-dit] LoKR: dim %d factor %d, %d monolithic / %d factorized w2, %zu params\n", dim,
                factor, n_mono, n_fact, n_params);
        fprintf(stderr, "[train-dit] LoKR contraction order: %d w2-first / %d w1-first / %d mid\n",
                n_order[DIT_LOKR_ORDER_W2], n_order[DIT_LOKR_ORDER_W1], n_order[DIT_LOKR_ORDER_MID]);
        return true;
    }

    // §2.3 kron-matvec. x [in, S]; the delta is contracted factor-by-factor:
    //   X3  = reshape(x)                     [in_n, in_m, S]     (n fastest, per
    //                                        row-major col index m*in_n + n)
    //   T1  = w2 . X3                        [out_k, in_m, S]
    //   T2  = (scale*w1) . swap(T1)          [out_l, out_k, S]
    //   d   = reshape(swap(T2))              [out_l*out_k, S]    (k fastest, per
    //                                        row index l*out_k + k)
    // ggml_permute(t, 1, 0, 2, 3) sends src axis0 to position 1 and axis1 to
    // position 0 — the ne0/ne1 swap both steps need. LK3 is the arbiter.
    // THE TOKEN AXIS MUST NOT REACH ne2 OF A MUL_MAT HERE (measured 2026-07-30).
    //
    // Both contractions below have a 2-D trainable factor (ne2 == 1) as src0. If
    // src1 carries the token count in ne2, ggml's backward emits the weight
    // gradient as out_prod(src1, grad) with dst->ne[2] == S — and ggml-cuda's
    // out_prod takes its `dps2 > 1` FALLBACK there: one cublasSgemm per token
    // (out-prod.cu:96-108), i.e. 344 launches of a 512x5 GEMM for ONE node, plus
    // a repeat_back to reduce the per-token gradient slabs afterwards. With 352
    // sites that was ~80% out_prod + ~8% repeat_back of the whole step and made
    // LoKR training 16.4x slower than the same run with a LoRA (1902 vs 116
    // ms/step at S=344, 32 layers). See docs/plans/2026-07-30-dit-trainer-step-profile.md.
    //
    // The cure is free: [in_n, in_m, S] and [in_n, in_m*S] are the SAME BYTES —
    // element (a,b,c) sits at a + in_n*b + in_m*in_n*c either way — so folding
    // the token axis into the column count is a pure reshape, not a maths
    // change. It leaves dst->ne[2] == 1, which takes out_prod's strided-batched
    // fast path in ONE call and removes the repeat_back entirely. The 3-D form
    // is restored around each swap, where the ne0/ne1 exchange actually needs it.
    //
    // WHY THE SWAPS CANNOT BE DELETED, only moved (2026-09-02). ggml_mul_mat
    // contracts ne0 of both operands and writes src0->ne1 to the RESULT's ne0,
    // with the batch axes taken from src1 (dst = [a->ne1, b->ne1, b->ne2,
    // b->ne3], and can_mul_mat requires b->ne2 % a->ne2 == 0, so the operand
    // carrying the tokens can only ever be src1). delta must come out as
    // [out_k, out_l*S] — k fastest, since the kron row index is l*out_k + k —
    // so the LAST contraction has to produce out_k at ne0, i.e. its src0 is the
    // w2 factor. Its src1 must then carry the w2 contraction index (in_n or
    // dim) at ne0 while the earlier w1 contraction already put out_l there:
    // one swap, unavoidable. The other swap is x itself, which arrives with the
    // w2 index at ne0 (row-major m*in_n + n), so whichever leg runs FIRST but
    // isn't the w2 leg needs it transposed. Only two escapes exist and both are
    // worse: making an activation src0 (then --bwd mm emits
    // ggml_cont(ggml_transpose(src0)) for the weight gradient — an
    // activation-sized copy moved into the backward), or broadcasting a factor
    // over S (then dst->ne[2] == S and out_prod takes the per-token fallback
    // this note's first half exists to avoid). So the win here is choosing the
    // CHEAPEST place to pay them — see DitLokrOrder above.
    ggml_tensor * apply(ggml_context * ctx_g, ggml_tensor * w, int layer, int s, ggml_tensor * x) const override {
        ggml_tensor *       y = ggml_mul_mat(ctx_g, w, x);
        const DitLokrSite * k = site(layer, s);
        if (!k) {
            return y;
        }
        ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx_g, x);
        const int64_t S  = ggml_nelements(xc) / xc->ne[0];
        // alpha/dim on the tiny side. K6 forces alpha = dim wherever both kron
        // factors are monolithic, and the shipped preset passes alpha == dim
        // anyway, so scale is 1.0f on every site of a production run: skip the
        // node rather than launch 352 forward + 352 backward kernels per
        // micro-step to multiply a 20-element tensor by exactly one. x * 1.0f is
        // exact in IEEE-754, so this is a bit-identical graph shortening.
        ggml_tensor * w1s = (k->scale == 1.0f) ? k->w1 : ggml_scale(ctx_g, k->w1, k->scale);
        ggml_tensor * d   = nullptr;                             // [out_k, out_l*S] when done

        if (k->order == DIT_LOKR_ORDER_W1) {
            // x --swap--> [in_m, in_n*S] --w1--> [out_l, in_n*S] --swap-->
            // [in_n, out_l*S] --w2--> [out_k, out_l*S].
            ggml_tensor * Xf = lokr_swap(ctx_g, xc, k->in_n, k->in_m, S);  // [in_m, in_n*S]
            ggml_tensor * U  = ggml_mul_mat(ctx_g, w1s, Xf);               // [out_l, in_n*S]
            ggml_tensor * Uf = lokr_swap(ctx_g, U, k->out_l, k->in_n, S);  // [in_n, out_l*S]
            d = k->mono ? ggml_mul_mat(ctx_g, k->w2, Uf)
                        : ggml_mul_mat(ctx_g, k->w2_a, ggml_mul_mat(ctx_g, k->w2_b, Uf));
        } else if (k->order == DIT_LOKR_ORDER_MID) {
            // x --w2_b--> [dim, in_m*S] --swap--> [in_m, dim*S] --w1-->
            // [out_l, dim*S] --swap--> [dim, out_l*S] --w2_a--> [out_k, out_l*S].
            ggml_tensor * X2 = ggml_reshape_2d(ctx_g, xc, k->in_n, k->in_m * S);
            ggml_tensor * Z  = ggml_mul_mat(ctx_g, k->w2_b, X2);            // [dim, in_m*S]
            ggml_tensor * Zf = lokr_swap(ctx_g, Z, (int64_t) dim, k->in_m, S);   // [in_m, dim*S]
            ggml_tensor * V  = ggml_mul_mat(ctx_g, w1s, Zf);                     // [out_l, dim*S]
            ggml_tensor * Vf = lokr_swap(ctx_g, V, k->out_l, (int64_t) dim, S);  // [dim, out_l*S]
            d                = ggml_mul_mat(ctx_g, k->w2_a, Vf);                 // [out_k, out_l*S]
        } else {
            // x --w2--> [out_k, in_m*S] --swap--> [in_m, out_k*S] --w1-->
            // [out_l, out_k*S] --swap--> [out_k, out_l*S].
            // [in_n, in_m*S] — same bytes as [in_n, in_m, S], ne2 == 1.
            ggml_tensor * X2 = ggml_reshape_2d(ctx_g, xc, k->in_n, k->in_m * S);
            ggml_tensor * T1 = k->mono ? ggml_mul_mat(ctx_g, k->w2, X2)
                                       : ggml_mul_mat(ctx_g, k->w2_a, ggml_mul_mat(ctx_g, k->w2_b, X2));
            ggml_tensor * T1f = lokr_swap(ctx_g, T1, k->out_k, k->in_m, S);   // [in_m, out_k*S]
            ggml_tensor * T2  = ggml_mul_mat(ctx_g, w1s, T1f);                // [out_l, out_k*S]
            d                 = lokr_swap(ctx_g, T2, k->out_l, k->out_k, S);  // [out_k, out_l*S]
        }
        ggml_tensor * delta = ggml_reshape_4d(ctx_g, d, y->ne[0], y->ne[1], y->ne[2], y->ne[3]);
        return ggml_add(ctx_g, y, delta);
    }

    // Exchange the two leading axes of `t`, read as [a, b, S], and hand back the
    // [b, a*S] fold the next mul_mat wants. The 3-D form exists only so
    // ggml_permute(1, 0, 2, 3) — src axis0 to position 1, axis1 to position 0 —
    // is expressible. ggml_cont_2d rather than cont + a separate reshape_2d: the
    // CONT backward already reshapes the gradient when the shapes differ, so the
    // extra view node bought nothing (measured: 17645 -> 17389 graph nodes at 32
    // layers / 11 sites, no measurable step-time change — kept because it is
    // strictly less graph for the same maths).
    static ggml_tensor * lokr_swap(ggml_context * ctx_g, ggml_tensor * t, int64_t a, int64_t b, int64_t S) {
        ggml_tensor * t3 = ggml_reshape_3d(ctx_g, t, a, b, S);
        return ggml_cont_2d(ctx_g, ggml_permute(ctx_g, t3, 1, 0, 2, 3), b, a * S);
    }

    // §2.4: <dir>/lokr_weights.safetensors, LyCORIS key layout. No
    // adapter_config.json — that is PEFT-only and the LoKr loader never reads it.
    bool exportDir(const char * dir, const DitExportMeta & meta, DitExportResult * res,
                   std::string * err) const override {
        const std::string d(dir);
        if (!pm_mkdir_p(d)) {
            *err = "cannot create " + d;
            return false;
        }

        std::vector<STWTensor>          tensors;
        std::vector<std::vector<float>> store;
        tensors.reserve((size_t) (hi - lo) * (size_t) n_sites * 4);
        store.reserve(tensors.capacity());

        for (int l = lo; l < hi; l++) {
            for (int s = 0; s < n_sites; s++) {
                const DitLokrSite & k    = layers[(size_t) (l - lo)][(size_t) s];
                const std::string   stem = lokr_key_stem(l, s);

                store.push_back(std::vector<float>(1, k.alpha_eff));
                STWTensor sa;
                sa.name  = stem + ".alpha";
                sa.shape = { 1 };
                sa.data  = nullptr;
                tensors.push_back(sa);

                ggml_tensor * ts[3]  = { k.w1, k.mono ? k.w2 : k.w2_a, k.mono ? nullptr : k.w2_b };
                const char *  sfx[3] = { "lokr_w1", k.mono ? "lokr_w2" : "lokr_w2_a", "lokr_w2_b" };
                for (int i = 0; i < 3 && ts[i]; i++) {
                    store.push_back(std::vector<float>((size_t) ggml_nelements(ts[i])));
                    std::vector<float> & v = store.back();
                    ggml_backend_tensor_get(ts[i], v.data(), 0, v.size() * sizeof(float));
                    STWTensor st;
                    st.name  = stem + "." + sfx[i];
                    st.shape = { ts[i]->ne[1], ts[i]->ne[0] };  // row-major (rows, cols)
                    st.data  = nullptr;
                    tensors.push_back(st);
                }
            }
        }
        // `store` reallocates as it grows; re-point once it has stopped moving.
        GGML_ASSERT(store.size() == tensors.size());
        for (size_t i = 0; i < tensors.size(); i++) {
            tensors[i].data = store[i].data();
        }

        std::vector<std::pair<std::string, std::string>> md;
        md.push_back({ "format", "lycoris" });
        md.push_back({ "algo", "lokr" });
        md.push_back({ "producer", meta.producer });
        md.push_back({ "hot_step_dit_trainer", "v1" });
        // linear_dim is NOT optional: adapter-merge.h skips every monolithic
        // module whose lokr_config lacks it, with only a warning.
        md.push_back({ "lokr_config", lokr_config_json() });
        if (!meta.trigger.empty()) {
            md.push_back({ "hot_step_trigger", meta.trigger });
            md.push_back({ "hot_step_trigger_position",
                           meta.trigger_position.empty() ? std::string("prepend") : meta.trigger_position });
            md.push_back({ "modelspec.trigger_phrase", meta.trigger });
        }

        const std::string sf = lm_join(d, "lokr_weights.safetensors");
        // BF16, not F32: LyCORIS writes its lokr_weights.safetensors in BF16 and
        // Side-Step's are byte-comparable at half our old size (872 MB vs 444 MB
        // for the same dim-512/factor-6 census). Every consumer reads it —
        // adapter-merge.h:60-64 handles "F32"/"BF16"/"F16", ComfyUI and LyCORIS
        // are BF16-native — and nothing round-trips this file back into training
        // (no resume path), so the export dtype is purely a storage decision.
        if (!st_write_file(sf.c_str(), tensors, md, STW_BF16)) {
            *err = "cannot write " + sf;
            return false;
        }
        if (res) {
            long long bytes = 0;
            pm_stat_file(sf, &bytes, NULL);
            res->tensors = (int) tensors.size();
            res->bytes   = bytes;
        }
        return true;
    }

    // "lycoris_layers_<L>_<peft site with dots flattened>" — the reverse of
    // lokr_build_reverse_map() in adapter-merge.h.
    static std::string lokr_key_stem(int l, int s) {
        std::string site_key(dit_site_peft(s));
        for (size_t i = 0; i < site_key.size(); i++) {
            if (site_key[i] == '.') {
                site_key[i] = '_';
            }
        }
        return "lycoris_layers_" + std::to_string(l) + "_" + site_key;
    }

    std::string lokr_config_json() const {
        // §2.4: linear_alpha is alpha_eff — the value that was actually TRAINED,
        // after K6's "both factors monolithic forces alpha = dim" override — not
        // the configured alpha. Read off a real site so the two can never drift;
        // the per-module `.alpha` tensors stay authoritative for the loaders.
        float a_eff = (alpha == 0.0f) ? (float) dim : alpha;
        if (!layers.empty()) {
            a_eff = layers[0][0].alpha_eff;
        }
        char b[224];
        snprintf(b, sizeof(b),
                 "{\"linear_dim\":%d,\"linear_alpha\":%.9g,\"factor\":%d,\"decompose_both\":%s,\"target_mlp\":%s}",
                 dim, (double) a_eff, factor,
                 decompose_both ? "true" : "false", n_sites == DIT_NSITES ? "true" : "false");
        return std::string(b);
    }

    static void fill_kaiming(LmRng * rng, ggml_tensor * t, std::vector<float> * v) {
        const size_t n = (size_t) ggml_nelements(t);
        const float  b = 1.0f / sqrtf((float) t->ne[0]);  // ggml ne0 == torch fan_in
        v->assign(n, 0.0f);
        for (size_t i = 0; i < n; i++) {
            (*v)[i] = (2.0f * lm_rng_uniform(rng) - 1.0f) * b;
        }
        ggml_backend_tensor_set(t, v->data(), 0, n * sizeof(float));
    }

    // dW = 0 at step 0. `sigma` > 0 only for the finite-difference gate, which
    // needs a non-zero w2 or every dL/dw1 is trivially zero.
    static void fill_zero(LmRng * rng, ggml_tensor * t, float sigma, std::vector<float> * v) {
        const size_t n = (size_t) ggml_nelements(t);
        v->assign(n, 0.0f);
        if (sigma > 0.0f) {
            lm_rng_fill_normal(rng, *v, sigma);
        }
        ggml_backend_tensor_set(t, v->data(), 0, n * sizeof(float));
    }
};

// {in, out} for every adapted site, in DIT_NSITES order (the attention sites
// first, so the leading DIT_NSITES_ATTN entries are the target_mlp=false set).
// Shared by the parameter count and the arena estimate below so the two can
// never disagree about the geometry they are counting.
static void dit_lokr_site_dims(const DiTGGMLConfig & c, int64_t sites[DIT_NSITES][2]) {
    const int64_t H = c.hidden_size, Q = (int64_t) c.n_heads * c.head_dim, KV = (int64_t) c.n_kv_heads * c.head_dim,
                  F = c.intermediate_size;
    const int64_t src[DIT_NSITES][2] = {
        { H, Q },  { H, KV }, { H, KV }, { Q, H },  // self q/k/v/o   {in, out}
        { H, Q },  { H, KV }, { H, KV }, { Q, H },  // cross q/k/v/o
        { H, F },  { H, F },  { F, H },             // gate / up / down
    };
    for (int s = 0; s < DIT_NSITES; s++) {
        sites[s][0] = src[s][0];
        sites[s][1] = src[s][1];
    }
}

// Expected parameter count for the cross-check in dit-train-run.h.
static size_t dit_lokr_expected_params(const DiTGGMLConfig & c, int lo, int hi, int lokr_dim, int lokr_factor,
                                       bool target_mlp) {
    int64_t sites[DIT_NSITES][2];
    dit_lokr_site_dims(c, sites);
    const int n_sites   = target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
    int64_t   per_layer = 0;
    for (int s = 0; s < n_sites; s++) {
        int64_t out_l, out_k, in_m, in_n;
        dit_lokr_factorization(sites[s][1], lokr_factor, &out_l, &out_k);
        dit_lokr_factorization(sites[s][0], lokr_factor, &in_m, &in_n);
        per_layer += out_l * in_m;
        per_layer += dit_lokr_w2_mono(lokr_dim, out_k, in_n) ? out_k * in_n
                                                             : out_k * (int64_t) lokr_dim + (int64_t) lokr_dim * in_n;
    }
    return (size_t) (per_layer * (int64_t) (hi - lo));
}

// Retained kron-matvec intermediates in apply(), summed over every trained site,
// bytes. S = crop / patch (the token count apply() sees).
//
// This is the arena cost dit-vram.h's fitted polynomial does NOT model: that fit
// was taken on LoRA runs, whose adapter branch retains only the [rank, S]
// bottleneck, while §2.3's LoKR contraction retains two ACTIVATION-shaped F32
// tensors per site, each immediately followed by a ggml_cont of its permuted
// view. Hence the pairs in dit_lokr_site_retained_elems(), and F32 throughout.
//
// Deliberately conservative: after ggml_gallocr's block reuse the true
// simultaneous retention is lower than this, but the backward pass adds gradient
// buffers of the same shapes, so until it is measured against a real high-water
// probe we err high.
//
// A factorized w2 splits the w2 contraction in two and inserts one further
// [lokr_dim, ...] intermediate; it is counted as well. It is a small term —
// factorization only fires when lokr_dim < max(out_k, in_n)/2 — but it is real,
// so leaving it out would make this an under-estimate on exactly the MLP sites
// that dominate the total.
//
// 2026-09-02, DELIBERATELY NOT UPDATED FOR THE CONTRACTION REORDER. The graph
// now retains less on the sites dit_lokr_pick_order() moves off
// DIT_LOKR_ORDER_W2 — dit_lokr_site_retained_elems() prices that, and
// dit_vram_lokr_apply_bytes_flash() uses it — but this is the EXACT-mode term,
// and dit-vram.h's own note (see DIT_FLASH_LOKR_RETENTION) records that exact
// mode's arena polynomial under-predicts by 13-18 % and that this LoKR
// over-count is silently covering for it. Pricing the reorder here would cut
// the exact estimate by ~40 % against a true arena that falls by ~15 %, i.e. it
// would spend a safety margin that is holding up a different, unrefitted fit.
// So this keeps the pre-reorder all-W2 arithmetic on purpose: it stays an
// over-estimate of a graph that got cheaper, which is the safe direction.
// Refitting the exact polynomial and then teaching this the order is one
// follow-up, not two changes at once — the same reasoning the flash-mode fix
// was gated under.
static double dit_lokr_apply_arena_bytes(const DiTGGMLConfig & c, int lo, int hi, int lokr_dim, int lokr_factor,
                                         bool target_mlp, int S) {
    if (hi <= lo || S <= 0) {
        return 0.0;
    }
    int64_t sites[DIT_NSITES][2];
    dit_lokr_site_dims(c, sites);
    const int n_sites   = target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
    double    per_layer = 0.0;  // elements per token
    for (int s = 0; s < n_sites; s++) {
        int64_t out_l, out_k, in_m, in_n;
        dit_lokr_factorization(sites[s][1], lokr_factor, &out_l, &out_k);
        dit_lokr_factorization(sites[s][0], lokr_factor, &in_m, &in_n);
        per_layer += dit_lokr_site_retained_elems(DIT_LOKR_ORDER_W2,
                                                  dit_lokr_w2_mono(lokr_dim, out_k, in_n), in_m, in_n, out_l, out_k,
                                                  lokr_dim);
    }
    return per_layer * (double) S * 4.0 * (double) (hi - lo);
}
