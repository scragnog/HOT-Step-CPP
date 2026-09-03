#pragma once
// dit-vram.h — the footprint model, the crop/depth auto-fit ladder and the
// card-support gate (plan §3.7, D9/D10/D11).
//
// Model (S = crop / patch, K = trained decoder layers):
//
//   arena_MB(S,K) = K*(0.3274*S + 1.139e-4*S^2) + (-0.125*S + 1.863e-4*S^2)
//   fixed_bytes(K) = mirror_bytes(K) + 16*adapter_params(K) + static input buffers
//   trainer_MB(S,K) = fixed + arena [+ lokr apply arena, LoKR only]
//
// The arena coefficients are the verifier's refit over all 12 measured points
// (the (K/32)*f(S) form has no K-independent term and under-predicts by 10-25%
// exactly in the low-K / long-crop corner the small-card map depends on).
//
// K10 ("reuse the polynomial as-is for LoKR") turned out to hold for the LoRA
// runs it was measured on and NOT for LoKR: every one of those 12 points is a
// LoRA run, whose adapter branch retains a single [rank, S] bottleneck, whereas
// the LoKR kron-matvec retains two activation-shaped tensors per site. The
// difference is thousands of MB at a long crop, which is how a LoKR run at the
// "fitted" crop ends up spilling into Windows' shared GPU memory. LoKR therefore
// adds dit_lokr_apply_arena_bytes() on top; the LoRA path is untouched.
//
// DEVIATION (E2, documented in the handoff): §3.7's fixed term is the fitted
// constant `7786 + (243.0 + state_MB_per_layer)*K`. We instead compute the mirror
// byte count EXACTLY by walking the same slot list the mirror builder walks
// (dit_mirror_bytes_for), and the optimizer state exactly from the adapter's own
// parameter count. That reproduces the fit's intent without its +-3 MB residual
// and, unlike the constant, stays correct for a base with different geometry.
// The arena term — the part that genuinely needs a fit — is used verbatim.
//
// docs/plans/2026-07-28-dit-trainer-implementation.md §3.7

#include "train/dit-adapter-lokr.h"

// HiRA's apply() materialises delta = A.B and wd = W (.) delta, both [in,out]
// F32, per trained site, and both are retained for the backward. That is a
// weight-sized pair per site the fitted arena polynomial never saw — exactly
// the class of term dit_lokr_apply_arena_bytes exists for. Independent of S.
// n_full = weight-sized F32 tensors retained per site: HiRA 2 (delta, W (.) delta),
// LoHa 3 (two products and their Hadamard).
static double dit_hira_apply_arena_bytes(const DiTGGMLConfig & c, int lo, int hi, bool target_mlp,
                                         double n_full = 2.0) {
    if (hi <= lo) {
        return 0.0;
    }
    int64_t sites[DIT_NSITES][2];
    dit_lokr_site_dims(c, sites);
    const int n_sites   = target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
    double    per_layer = 0.0;
    for (int s = 0; s < n_sites; s++) {
        per_layer += n_full * (double) sites[s][0] * (double) sites[s][1] * 4.0;
    }
    return per_layer * (double) (hi - lo);
}
#include "train/dit-adapter.h"
#include "train/dit-mirror.h"
#include "train/gpu-mem.h"
#include "train/lm-common.h"
#include "train/lm-vram.h"  // LmVramTracker (leak counter), reused unchanged

#include <algorithm>
#include <string>
#include <vector>

struct DitVramModel {
    DiTGGML * m       = nullptr;
    int       n_layers = 0;
    int       patch    = 2;
    int       rank     = 16;
    bool      target_mlp = false;
    // Adapter parameterization: the params term is the only footprint that
    // differs, and the arena polynomial is reused as-is (K10).
    bool is_lokr     = false;
    bool hira        = false;  // HiRA: [in,out] delta + Hadamard per site retained for backward
    bool loha        = false;  // LoHa: two [in,out] products + Hadamard per site
    // A trained cond leaf (artist token) forces the checkpoint plan over the
    // WHOLE stack with the same layers-per-segment: more boundaries, same
    // widest segment. See dit-train-run.h at the plan.
    bool full_stack_ckpt = false;
    int  lokr_dim    = 512;
    int  lokr_factor = 6;
    // Mirror precision: the fixed term is the mirror, so this is the single
    // largest lever in the whole model (--mirror bf16 roughly halves it).
    // DIT_MIRROR_BF16_F32 shares bf16's fixed term exactly (same stored bytes)
    // and adds one transient arena term — see dit_vram_total_bytes.
    DitMirrorMode mirror = DIT_MIRROR_F32;
    // Micro-batch size (design C4). Every activation/arena/static term scales
    // with it; mirror and optimizer state do not.
    int batch = 1;
    // Gradient-checkpoint segments (design C1/C4). 1 = off (the monolithic
    // graph). The retained activation set is ONE segment, so the per-layer arena
    // and the A1 expanded-KV term divide by it; the boundary buffers are the
    // price paid back. Mirror and optimizer state are segment-invariant.
    int segments = 1;
    // static input geometry
    int in_ch = 192, out_ch = 64, hidden = 2560, enc_H = 2048, enc_S = 0;
    // attention geometry — only the A1 expanded-KV term needs it
    int n_heads = 0, n_kv_heads = 0, head_dim = 0;
    // Attention formulation the graph will be built with (--attn). false = the
    // shipped `dit_attn_f32` chain, priced by dit_vram_arena_bytes(); true = the
    // fused GGML_OP_FLASH_ATTN_TRAIN pair on BOTH attention sites, priced by
    // dit_vram_arena_bytes_flash(). Default false so every existing caller
    // (dit-selftest.h included) keeps the exact model to the byte.
    bool flash_attn = false;
};

// arena (scheduler compute buffer), bytes.
//
// BATCHING VALIDATED (S-C2, 2026-07-29). The straight xB on the activation
// terms below was carried over UNMEASURED from chunk 1/2; chunk 3's S-C2 grid
// measured it against `ace-train`'s own trainer-owned high-water figure at
// {B 1,5} x {ckpt 0,4} x {crop 375,750} (snapped to 374/750, patch 2), rank-16
// LoRA, --layers 8, bf16 mirror, on a 32-layer turbo-XL base (Nh 32/Nkv 8):
//
//   B  ckpt  crop  estMb  trainerMb  deviceWideMb  est-vs-trainer
//   1    0    374   8719    8641       13843          +0.9%
//   1    0    750   9304    9124       14316          +2.0%
//   1    4    374   8339    8329       13530          +0.1%
//   1    4    750   8494    8460       13673          +0.4%
//   5    0    374  11046   10673       15863          +3.5%
//   5    0    750  14149   13234       18425          +6.9%
//   5    4    374   8926    8888       14135          +0.4%
//   5    4    750   9744    9570       14871          +1.8%
//
// Every cell is within the +-10% bar and, notably, on the SAME side of it
// (over-, never under-predicting) — the safe direction for a budget the
// reserve/safety margins and the NVML tripwire back up. Decomposing the
// residual (isolating this polynomial's own contribution from the exactly-
// computed mirror/optimizer/static terms, which are arithmetic, not fit) shows
// the straight xB does NOT introduce or amplify error with batch size: the
// arena-term ratio at K=8 (ckpt off) is ~0.84-0.85 at BOTH B=1 and B=5, and at
// Kr=2 (ckpt on, the widest segment at --ckpt 4) it is ~0.91-0.94 at both B
// values too. That spread is the pre-existing per-K residual the file header
// already documents ("the fit ... under-predicts by 10-25% exactly in the
// low-K / long-crop corner") — batching inherits it unchanged rather than
// making it worse, so no numeric constant changes for the B term. NO CHANGE
// was needed here: this comment records the validation, not a refit.
// `K` is the RETAINED layer count — the trained depth when checkpointing is off,
// the widest single segment when it is on (design C4: the retained activation set
// is one segment, and the head/loss term, which is K-independent, is not divided).
// UNMEASURED past this grid's range (K > 8, B > 5, crop > 750, ckpt other than
// 0/4) — the backstop for those stays the high-water probe + safety margins.
static double dit_vram_arena_bytes(int S, int K, int B = 1) {
    const double dS = (double) S;
    const double mb = (double) K * (0.3274 * dS + 1.139e-4 * dS * dS) + (-0.125 * dS + 1.863e-4 * dS * dS);
    return (mb > 0.0 ? mb : 0.0) * 1048576.0 * (double) std::max(1, B);
}

// ─── flash-mode arena (--attn flash) ────────────────────────────────────────
//
// The exact polynomial above cannot be reused with a fudge factor: the term
// `--attn flash` removes IS its per-layer quadratic, and the moment that is gone
// the linear coefficient is the whole model, with an undeclared enc_S dependence
// buried inside it (fattn-train-spec.md §11.1). So the flash branch is built term
// by term, arithmetic wherever the arithmetic is knowable, with ONE fitted
// headroom coefficient on the outside — never a refitted polynomial.
//
// Read the exact model's coefficients against the tensors they stand in for
// (dit-xl-thirds: Nh 32, Nkv 8, D 128; 4 B per f32, 2^20 B per MB):
//
//   per-layer quadratic  1.139e-4 MB/S^2  vs  Nh*4/2^20 = 1.2207e-4  ->  that term
//                        IS the retained self-attention softmax (93 % of the
//                        arithmetic; ggml-alloc reuse is the rest).
//   per-layer linear     0.3274 MB/token  ->  the retained CROSS-attention softmax
//                        (enc_S*Nh*4/2^20 per token) plus every non-attention
//                        activation, with no way to tell which is which — the
//                        §11.1 gap.
//
// `--attn flash` flashes BOTH attention sites (dit-train-graph.h), so BOTH
// retained softmaxes go. What the flash branch carries instead:
//
//   per layer, per batch element
//     DIT_FLASH_ACT_MB_PER_TOKEN * S   the non-attention retained activations.
//                        MEASURED, not arithmetic — the counterpart of the exact
//                        model's 0.3274, and the only structural constant here.
//     fused packed O+LSE * S           spec §11.2 term 2: [D,Nh,S,B] O plus
//                        [Nh,S,B] LSE, per site, retained across the backward
//                        because it is src[4] of the back op. ARITHMETIC.
//     cross-attention K/V * enc_S      three [D,Nkv,enc_S,B] widths (ck, cv and
//                        one live gradient). These scale with enc_S and not with
//                        S, so they appear in NO term of the exact polynomial —
//                        which has no K-only term at all — and were inside that
//                        fit's residual. ARITHMETIC.
//   K-independent, per batch element
//     SCALE + ACC pair                 spec §11.2 term 3: ggml_acc_or_set builds
//                        the packed tensor's gradient as ACC(SCALE(packed, 0), dO)
//                        and neither can run in place, so two packed-size f32
//                        buffers are live at once. ARITHMETIC.
//     back op's dQ|dK|dV               spec §11.2 term 4, the larger of the two
//                        sites (only one is live at a time). ARITHMETIC.
//
// The exact model's K-independent head/loss term (-0.125*S + 1.863e-4*S^2) is
// NOT carried: the measured flash-mode K-independent residual is the gradient
// machinery above and essentially nothing else (13.6 / 22.4 / 41.2 / 82.4 MB of
// arithmetic against 15 / 25 / 41 / 77 MB measured at S = 187/375/750/1500),
// while that polynomial would ask for 232 MB at S = 1500.
//
// ─── REFIT 2026-09-02 (overnight-sweep defect 1) ────────────────────────────
//
// The original fit is superseded. It was taken mid-development on a binary that
// predates the TF32 fused kernels and the final graph, and on today's `ace-train`
// it UNDER-predicts on every cell — the opposite of what this model promises.
// The refit below is measured against the shipped binary; the numbers the old
// coefficients produce are in the `est_old` column so the regression is on the
// record rather than in a commit message.
//
// What the residuals said (48 rank-16 LoRA cells, {enc_S 434, 609, 640, 1188,
// 1877} x {S 187, 375, 425, 750, 1250, 1500} x {K 4, 8, 16, 32}, --mirror bf16,
// B = 1, both --bwd formulations, RTX 5090, dit-xl-thirds BF16):
//
//   * the cross-attention enc_S term is RIGHT. Measured d(arena)/d(enc_S) is
//     0.376-0.389 MB per encoder token against the term's 0.383 — so the
//     "cross-attention priced at self-attention shapes" hypothesis is dead, and
//     the enc_S half of the model needs no change at all.
//   * the per-token activation coefficient was simply too small: the residual is
//     linear in K*S with no enc_S content whatsoever (the implied coefficient is
//     0.2447-0.2549 at every one of the 48 cells, flat across enc_S). 0.19779 ->
//     0.2463, i.e. +24.5 %.
//   * headroom does NOT need to scale with anything. It comes down, because the
//     refitted arithmetic is tight enough not to need 12 %.
//   * one term was missing: `--bwd mm` costs a flat ~52 MB the outprod
//     formulation does not pay (measured as a K- and S-independent offset
//     between matched op/mm cells). Charged unconditionally, so an outprod run
//     is over-priced by 52 MB — the safe direction, and negligible next to the
//     headroom.
//
// Raw (pre-headroom) the refit reproduces all 48 cells at +0.0 % to +9.7 %, none
// under; with the headroom, +5.0 % to +15.2 %, and every `--bwd mm` cell (what
// the server and the trainer actually run) lands +5.0 % to +5.8 %.
//
//   dataset  enc_S     S   K   meas  est_old        est_new
//   joyce      434   187  32   1880     1733 -7.8%     1984 +5.5%
//   joyce      434  1250  32  11345    10531 -7.2%    11965 +5.5%
//   fight      640   187  32   1960     1821 -7.1%     2067 +5.4%
//   fight      640  1250  32  11454    10618 -7.3%    12046 +5.2%
//   ladis     1188   425  32   4289     4024 -6.2%     4520 +5.4%
//   ladis     1188  1250  32  11618    10848 -6.6%    12262 +5.5%
//   nwa       1877   187  32   2441     2351 -3.7%     2564 +5.0%
//   nwa       1877   425  32   4558     4320 -5.2%     4797 +5.2%
//   nwa       1877  1250  32  11900    11143 -6.4%    12538 +5.4%
//   pres/op    609  1500  32  13570    12674 -6.6%    14382 +6.0%
//   fight      640   187   4    302      241 -20.1%     319 +5.6%
//   fight      640  1250   8   2947     2712 -8.0%     3107 +5.4%
//   nwa       1877   750  16   3775     3532 -6.4%     3977 +5.3%
//
// (full 48-cell table and the raw logs: _experiments/fattn-arena-refit/)
//
// UNMEASURED past this grid: B > 1, rank > 16, and any base with a different
// Nh/Nkv/D. The backstop for those is unchanged — the mandatory high-water probe
// and the NVML tripwire.
//
// docs/plans/2026-09-01-flash-attn-backward.md (MEASUREMENT 2), spec §11.

// Non-attention retained activations, MB per token per layer. The flash-mode
// counterpart of the exact model's 0.3274, measured the same way. (Sanity: 0.2463
// MB is 25.2 hidden-widths at H = 2560, f32 — the qkv/o/mlp/norm/residual set of
// one layer's forward plus the gradients ggml keeps live, which is the right
// order of magnitude.)
#define DIT_FLASH_ACT_MB_PER_TOKEN 0.2463

// `--bwd mm`'s flat surcharge, MB. Measured as the offset between matched
// `--bwd mm` and `--bwd outprod` cells: 47-51 MB at every K and S, so it is a
// scratch buffer of the MUL_MAT activation-gradient formulation and not an
// activation. `--bwd` is not part of DitVramModel, so it is charged always.
#define DIT_FLASH_BWD_MM_MB 52.0

// The ONE fitted safety coefficient in this branch. It is not a shape: the terms
// above already reproduce every measured cell to +0.0/+9.7 %, and this keeps the
// estimate on the over-predicting side of all of them with room to spare, which
// is the side the reserve, the safety margin and the NVML tripwire back up.
#define DIT_FLASH_HEADROOM 1.05

static double dit_vram_arena_bytes_flash(int S, int K, int B, int enc_S, int Nh, int Nkv, int D) {
    const double dS  = (double) S;
    const double dNh = (double) std::max(1, Nh);
    const double dKv = (double) std::max(1, Nkv);
    const double dD  = (double) std::max(1, D);
    const double dE  = (double) std::max(1, enc_S);
    const double MB  = 1048576.0;

    // the fused op's packed output: [D,Nh,S,B] O + [Nh,S,B] LSE, per site.
    const double packed_mb_per_token = (dD * dNh + dNh) * 4.0 / MB;

    // per layer, per batch element.
    const double act_mb   = DIT_FLASH_ACT_MB_PER_TOKEN * dS;
    const double fused_mb = 2.0 /* self + cross */ * packed_mb_per_token * dS;
    const double ckv_mb   = 3.0 * dD * dKv * 4.0 / MB * dE;
    const double per_layer_mb = act_mb + fused_mb + ckv_mb;

    // K-independent, per batch element: the gradient machinery of ONE site (the
    // larger of the two — they are not live at the same time).
    const double scale_acc_mb  = 2.0 * packed_mb_per_token * dS;
    const double back_self_mb  = (dD * dNh * dS + 2.0 * dD * dKv * dS) * 4.0 / MB;
    const double back_cross_mb = (dD * dNh * dS + 2.0 * dD * dKv * dE) * 4.0 / MB;
    const double machinery_mb  = scale_acc_mb + std::max(back_self_mb, back_cross_mb) + DIT_FLASH_BWD_MM_MB;

    const double mb = (double) std::max(0, K) * per_layer_mb + machinery_mb;
    return (mb > 0.0 ? mb : 0.0) * DIT_FLASH_HEADROOM * MB * (double) std::max(1, B);
}

// ─── flash-mode LoKR apply arena ────────────────────────────────────────────
//
// dit_lokr_apply_arena_bytes() (dit-adapter-lokr.h) is the term that actually
// caused the reported defect, and it is not the flash model: on
// `nwa_straightoutta` at crop 850 the flash arena is 4319 MB of a 13,536 MB
// estimate and the LoKR apply term is the other 9217 MB, against 7824 MB
// measured for the whole arena. The diagnostic line only printed the flash half,
// which is what made a 73 % OVER-prediction read as a 45 % under-prediction.
//
// That function says of itself "deliberately conservative ... we err high", and
// the measurements say how high: it over-prices the arena by 46-93 % across 21
// LoKR cells. It is wrong in two ways.
//
//   1. Retention. It charges 2x (T1 + T2) per site on the assumption that each
//      kron intermediate is followed by a live ggml_cont and that the backward
//      adds gradient buffers of the same shapes. Measured, the simultaneously
//      live set is 0.61x of ONE copy of (T1 + T2 + the factorized-w2
//      intermediate) — ggml_gallocr reuses far more aggressively than the
//      comment assumed. That is a 3.3x difference.
//   2. Token count. It bills EVERY site at S tokens. The cross-attention K and V
//      sites contract against the encoder states, so they see enc_S tokens, not
//      S. That is why the measured LoKR surcharge rises with enc_S (1998 ->
//      2186 MB from enc_S 434 to 1877 at K 8 / S 1250) while the old term is
//      completely flat in enc_S.
//
// So this is the same site walk, with the cross-K/V sites moved onto enc_S and
// ONE fitted retention factor replacing the factor 2. Measured against
// `measured LoKR arena - measured LoRA arena` at matched (K, S, enc_S) — 21
// cells, {enc_S 434, 640, 1188, 1877} x {S 187, 425, 750, 1250} x {K 4, 8, 16,
// 32}, dim 512 / factor 6 / target-mlp plus one dim-128 cell — it lands +0.0 %
// to +6.0 % over, none under, where the old term was +46 % to +93 % over.
//
// WHY THE EXACT PATH KEEPS THE OLD TERM. Exact mode's own arena polynomial
// currently under-predicts too (measured on the same binary: 2533 MB vs 2069 est
// at K 32 / S 187, 5838 vs 5091 at S 425 — 13-18 % under), and its LoKR
// over-count is silently covering for that. Refitting the exact polynomial is
// explicitly out of this task's scope, so removing the compensation on that path
// would turn exact-mode LoKR runs into OOMs. Flash mode does not need the
// compensation: its arena term above is now measured and over-predicting on its
// own. One follow-up, not two changes at once.
//
// Grid, flash arena TOTAL (flash term + this one) vs measured arena:
//
//   enc_S     S   K  dim   meas   est_old         est_new
//     640   187   4  512    477      748 +56.8%      511 +7.2%
//     640  1250   4  512   2567     4783 +86.3%     2736 +6.6%
//     640  1250   8  512   4997     9489 +89.9%     5346 +7.0%
//     640   425  32  512   7023    13006 +85.2%     7500 +6.8%
//     434  1250   8  512   4917     9468 +92.6%     5307 +7.9%
//    1188  1250   8  512   5089     9547 +87.6%     5450 +7.1%
//    1877   187   8  512   1109     1621 +46.2%     1198 +8.1%
//    1877  1250   8  512   5259     9625 +83.0%     5586 +6.2%
//    1877   425  32  512   7824    13537 +73.0%     8450 +8.0%   <- the defect
//     640  1250   8  128   4937     9470 +91.8%     5317 +7.7%
//
// UNMEASURED: lokr_dim other than 512 (one dim-128 cell only), lokr_factor other
// than 6, decompose_both off, B > 1. High-water probe and tripwire as ever.
#define DIT_FLASH_LOKR_RETENTION 0.62

// `S_tok` and `encS_tok` are token counts that already carry the batch factor,
// matching dit_lokr_apply_arena_bytes' `S` argument.
static double dit_vram_lokr_apply_bytes_flash(const DiTGGMLConfig & c, int lo, int hi, int lokr_dim, int lokr_factor,
                                              bool target_mlp, double S_tok, double encS_tok) {
    if (hi <= lo || S_tok <= 0.0) {
        return 0.0;
    }
    int64_t sites[DIT_NSITES][2];
    dit_lokr_site_dims(c, sites);
    const int n_sites = target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
    double    per_S   = 0.0;  // elements per activation token, per layer
    double    per_E   = 0.0;  // elements per ENCODER token, per layer
    for (int s = 0; s < n_sites; s++) {
        int64_t out_l, out_k, in_m, in_n;
        dit_lokr_factorization(sites[s][1], lokr_factor, &out_l, &out_k);
        dit_lokr_factorization(sites[s][0], lokr_factor, &in_m, &in_n);
        // 2026-09-02: the per-site live set follows whichever contraction order
        // dit_lokr_pick_order() gives this site, so this estimate cannot drift
        // from the graph apply() builds. Sites still on DIT_LOKR_ORDER_W2 keep
        // the expression this replaced, element for element.
        const bool mono = dit_lokr_w2_mono(lokr_dim, out_k, in_n);
        const int  ord  = dit_lokr_pick_order(mono, in_m, in_n, out_l, out_k, lokr_dim);
        double     e    = dit_lokr_site_live_elems(ord, mono, in_m, in_n, out_l, out_k, lokr_dim);
        // dit_lokr_site_dims' order is self q/k/v/o, cross q/k/v/o, gate/up/down:
        // sites 5 and 6 are the cross-attention K and V projections, and they
        // contract against the encoder states.
        ((s == 5 || s == 6) ? per_E : per_S) += e;
    }
    const double mb = DIT_FLASH_LOKR_RETENTION * (per_S * S_tok + per_E * encS_tok) * 4.0 / 1048576.0 *
                      (double) (hi - lo);
    return mb * DIT_FLASH_HEADROOM * 1048576.0;
}

// Layers retained at once: the widest segment. Checkpointing off (segments <= 1)
// is the whole trained depth, i.e. the pre-batching term unchanged.
static int dit_vram_seg_layers(const DitVramModel & vm, int K) {
    const int seg = std::max(1, std::min(vm.segments, std::max(1, K)));
    return (K + seg - 1) / seg;
}

// The C1 boundary buffers: one [H,S,B] activation per segment (segment 0's is the
// pre-stack output) plus two ping-ponging boundary gradients. Zero when off.
// EXACT, not fitted — a deterministic byte count matching dit_ckpt_alloc's own
// allocation size. S-C2 (2026-07-29) confirms it: every grid cell's reported
// boundaryMb equalled this formula to the rounded MB (e.g. B=5/crop=750/segments=4
// -> 2560*375*5*4 * 6B = 109 MB, exactly what the run logged).
static double dit_vram_boundary_bytes(const DitVramModel & vm, int S, int K) {
    if (vm.segments <= 1 && !vm.full_stack_ckpt) {
        return 0.0;
    }
    const double one   = (double) vm.hidden * (double) S * (double) std::max(1, vm.batch) * 4.0;
    int          n_bnd = vm.segments;
    if (vm.full_stack_ckpt) {
        const int seg_len = std::max(1, dit_vram_seg_layers(vm, K));
        n_bnd             = (vm.n_layers + seg_len - 1) / seg_len;
    }
    return one * (double) (n_bnd + 2);
}

// A1's expanded-KV surcharge, bytes. At B > 1 the K/V activations are tiled from
// Nkv to Nh heads BEFORE the attention mul_mats (dit_expand_heads), because ggml's
// GQA mul_mat backward aborts on a batch axis. Charged per trained layer for
// self-attention (S tokens) and cross-attention (enc_S tokens), K and V each.
// A shape-correct (arithmetic, not fitted) estimate of the retained set. S-C2
// (2026-07-29) backed this term out of the grid's total-vs-estimate residual
// (Nh 32/Nkv 8 on the test base) and found it consistent with the measured
// totals to the same margin as the arena term above — not independently
// isolated, since it and the arena term only separate cleanly by algebra, but
// nothing in the grid suggests it is the source of any cell's error.
// HOT-Step patch: flash-attn-train (investigation B2) — flash mode does not pay
// this term at all. dit_attn_needs_kv_expand() skips the expansion when the graph
// is built with the fused ops, so charging for it would over-price every B > 1
// flash run and hand the crop walk a smaller crop than the card can hold. The
// flash arena polynomial was already written in terms of Nkv (dKv), i.e. the
// native-GQA K/V widths, so the two halves agree.
static double dit_vram_kv_expand_bytes(const DitVramModel & vm, int S, int K) {
    if (vm.flash_attn) {
        return 0.0;
    }
    if (vm.batch <= 1 || vm.n_heads <= 0 || vm.n_kv_heads <= 0 || vm.n_heads == vm.n_kv_heads) {
        return 0.0;
    }
    const double extra_heads = (double) (vm.n_heads - vm.n_kv_heads);
    const double per_token   = (double) vm.head_dim * extra_heads * 4.0;
    const double tokens      = ((double) S + (double) std::max(1, vm.enc_S)) * (double) vm.batch;
    return 2.0 /*K and V*/ * per_token * tokens * (double) std::max(1, K);
}

// persistent input tensors we allocate ourselves, bytes.
static double dit_vram_static_bytes(const DitVramModel & vm, int crop) {
    const int    S  = crop / vm.patch;
    const double B  = (double) std::max(1, vm.batch);
    const double b  = (double) vm.in_ch * crop * 4.0 * B                    // b_input
                    + (double) vm.enc_H * std::max(1, vm.enc_S) * 4.0 * B   // b_enc
                    + (double) S * 4.0 * B                                  // b_pos
                    + (double) vm.hidden * 7.0 * 4.0 * B                    // b_temb + b_tproj
                    // b_sa (+ b_sa_pad, allocated only at B > 1): f16, per element
                    // when a mixed-length batch needs a padded-KV mask.
                    + (double) S * S * 2.0 * B * (vm.batch > 1 ? 2.0 : 1.0)
                    + (double) std::max(1, vm.enc_S) * S * 2.0 * B          // b_ca  f16
                    + (double) vm.out_ch * crop * 4.0 * B                   // b_vtgt
                    + (double) crop * 4.0 * 2.0 * B                         // b_lw + b_lwu
                    + (double) vm.out_ch * 4.0                              // t_cw
                    + 64.0;                                                 // scalars
    return b;
}

static size_t dit_vram_adapter_params(const DitVramModel & vm, int K) {
    return vm.is_lokr ? dit_lokr_expected_params(vm.m->cfg, vm.n_layers - K, vm.n_layers, vm.lokr_dim, vm.lokr_factor,
                                                 vm.target_mlp)
                      : dit_lora_expected_params(vm.m->cfg, vm.n_layers - K, vm.n_layers, vm.rank, vm.target_mlp);
}

// mirror + optimizer state + adapter params, bytes. K = trained depth (top-K).
static double dit_vram_fixed_bytes(const DitVramModel & vm, int K, int crop) {
    const size_t mirror = dit_mirror_bytes_for(vm.m, vm.n_layers - K, vm.mirror, vm.batch > 1);
    const size_t np     = dit_vram_adapter_params(vm, K);
    return (double) mirror + 16.0 * (double) np + dit_vram_static_bytes(vm, crop);
}

static double dit_vram_total_bytes(const DitVramModel & vm, int crop, int K) {
    const int S   = crop / vm.patch;
    const int Kr  = dit_vram_seg_layers(vm, K);  // retained layers (C4)
    // Which arena model prices this run. `flash_attn` defaults false, so every
    // caller that has not opted in gets dit_vram_arena_bytes() to the byte.
    const double arena = vm.flash_attn ? dit_vram_arena_bytes_flash(S, Kr, vm.batch, vm.enc_S, vm.n_heads,
                                                                    vm.n_kv_heads, vm.head_dim)
                                       : dit_vram_arena_bytes(S, Kr, vm.batch);
    // --mirror bf16-f32's transient F32 window (2026-09-02). The mirror term
    // inside dit_vram_fixed_bytes already prices this mode as bf16, because
    // dit_mirror_bytes_for walks the same slot list the builder walks and
    // bf16-f32 stores exactly what bf16 stores. What it cannot see is the
    // in-graph ggml_cast at each trainable-layer mul_mat site, whose F32 output
    // lives in the ARENA. Arithmetic, not fitted: the largest single trainable
    // layer's eleven projections at 4 B/element (~0.5 GB on a 32-layer XL base,
    // against the ~8 GB this mode saves versus a full F32 mirror). Added on both
    // the exact and the flash path, since the cast sites are the same in each and
    // neither arena polynomial was fitted with a run that emitted them. Zero for
    // every other mirror mode, so no existing estimate moves by a byte.
    const double cast_window = (double) dit_mirror_cast_window_bytes(vm.m, vm.n_layers - K, vm.mirror);
    double    b   = dit_vram_fixed_bytes(vm, K, crop) + arena + cast_window +
               dit_vram_kv_expand_bytes(vm, S, Kr) + dit_vram_boundary_bytes(vm, S, K);
    if (vm.is_lokr) {
        // The intermediates the fitted polynomial never saw. Added here rather
        // than at any one call site so every consumer — the crop walk, the depth
        // ladder, the `vram` JSONL est and dit_train_log.json — sees them too.
        // C4: its `S` is already "tokens seen by apply()", which batching multiplies,
        // and its layer range is the RETAINED one, which checkpointing divides.
        const double Bf = (double) std::max(1, vm.batch);
        b += vm.flash_attn ? dit_vram_lokr_apply_bytes_flash(vm.m->cfg, vm.n_layers - Kr, vm.n_layers, vm.lokr_dim,
                                                             vm.lokr_factor, vm.target_mlp, (double) S * Bf,
                                                             (double) std::max(1, vm.enc_S) * Bf)
                           : dit_lokr_apply_arena_bytes(vm.m->cfg, vm.n_layers - Kr, vm.n_layers, vm.lokr_dim,
                                                        vm.lokr_factor, vm.target_mlp, S * std::max(1, vm.batch));
    }
    if (vm.hira || vm.loha) {
        b += dit_hira_apply_arena_bytes(vm.m->cfg, vm.n_layers - Kr, vm.n_layers, vm.target_mlp, vm.hira ? 2.0 : 3.0);
    }
    return b;
}

// ─── the A1 hard cap on B ───────────────────────────────────────────────────
//
// dit_expand_heads' backward is ggml_repeat_back, and the CUDA kernel refuses
// grad->ne[2]*grad->ne[3] above 32768 (ggml-cuda.cu:5305). This is a CORRECTNESS
// cap, not a footprint one: exceeding it aborts the run mid-backward.
//
// BOTH attentions expand. Self-attention's expanded K/V is [D,G,Nkv,S*B], so its
// product is Nkv*S*B — but cross-attention's is [D,G,Nkv,enc_S*B], i.e. Nkv *
// enc_S * B, and enc_S is the DATASET's padded encoder length, which has nothing
// to do with the crop. A long-lyrics variant (enc_S in the high hundreds) trips
// the cross-attention side while the self-attention side is comfortable, so the
// constraint is Nkv * max(S, enc_S) * B. (Fixed 2026-07-29: the cap was
// originally applied to S alone, which left enc_S entirely unguarded.)
//
// Returns the largest batch that satisfies it; `want` when unconstrained
// (Nh == Nkv means no expansion happens at all, and neither does B == 1 —
// dit_train_layer only calls dit_expand_heads when B > 1).
//
// HOT-Step patch: flash-attn-train (investigation B2) — `flash` lifts it
// entirely. The cap is a property of ggml's REPEAT_BACK CUDA kernel, and in
// flash mode there is no repeat and therefore no repeat_back: the fused ops
// consume native GQA. Passing the mode in rather than reading a global keeps
// this a pure function of its arguments, which is what dit_vram_fit's callers
// and the selftest both assume.
#define DIT_REPEAT_BACK_MAX 32768

static int dit_vram_max_batch(int n_heads, int n_kv_heads, int S, int enc_S, int want, bool flash) {
    const int tok = std::max(S, enc_S);
    if (flash || want <= 1 || n_kv_heads <= 0 || n_heads == n_kv_heads || tok <= 0) {
        return want;
    }
    const long long per = (long long) n_kv_heads * (long long) tok;
    if (per <= 0) {
        return want;
    }
    const int cap = (int) ((long long) DIT_REPEAT_BACK_MAX / per);
    return (cap < 1) ? 1 : std::min(want, cap);
}

// ─── auto-fit ───────────────────────────────────────────────────────────────

struct DitVramFit {
    int    crop = 0, layers = 0;
    int    segments     = 1;  // resolved checkpoint segments (1 = off)
    double est_bytes    = 0.0;
    size_t free_mb      = 0, total_mb = 0;
    const char * free_source = "cuda";  // which probe produced free_mb (gpu-mem.h)
    bool   ok           = false;
    bool   crop_user    = false;
    bool   layers_user  = false;
    bool   lowered      = false;  // the ladder had to reduce depth
    // The refusal number: the cheapest configuration this walk can express, and
    // the configuration itself so a refusal can name it. See the block that
    // fills these in inside dit_vram_fit.
    double floor_bytes    = 0.0;
    int    floor_crop     = 0;
    int    floor_layers   = 0;
    int    floor_segments = 1;
    // What the walk was actually allowed to spend: free - reserve, less safety.
    // Reported so a refusal can show the subtraction rather than just its result.
    double budget_bytes   = 0.0;
};

// The widest segment count the walk would try at depth K — the last element of
// dit_vram_fit's seg_candidates(K). Hoisted out so the floor below and the walk
// cannot drift apart: a floor priced at a segment count the walk would never
// reach is either a refusal that lies or a gate that lets an OOM through.
static int dit_vram_widest_segments(int ckpt_mode, int K) {
    const int kmax = std::max(1, K);
    if (ckpt_mode == 0) {
        return 1;  // --ckpt 0 pins the monolithic graph; segments are not an axis
    }
    if (ckpt_mode >= 2) {
        return std::min(ckpt_mode, kmax);  // a pinned N is never changed behind the user's back
    }
    return std::min(32, kmax);  // auto: the { 1, 2, 4, 8, 16, 32 } ladder's last rung
}

// Largest patch-aligned crop in [crop_min, cap] whose footprint fits `budget`.
// 0 when even crop_min does not fit.
static int dit_vram_best_crop(const DitVramModel & vm, int K, double budget, int crop_min, int cap) {
    int best = 0;
    for (int crop = crop_min; crop <= cap; crop += vm.patch) {
        if (dit_vram_total_bytes(vm, crop, K) <= budget) {
            best = crop;
        } else {
            break;  // monotonically increasing in crop
        }
    }
    return best;
}

// `user_layers` / `user_crop` are 0 for auto. `max_T` is the longest song, which
// caps the useful crop (songs shorter than the crop use their whole length).
//
// `mem` is the honest device figure (gpu-mem.h). Passing nullptr keeps the old
// cudaMemGetInfo behaviour; the trainer always passes one, because under WDDM
// that probe over-reports free VRAM by ~9 GB and the budget below is only as
// good as the number it starts from.
//
// `ckpt_mode` is the --ckpt knob (C3): 0 = off, 1 = auto (this function picks),
// N >= 2 = exactly N segments. `depth_ladder` == false stops the search at full
// depth, which is how the caller honours the C4 order — full depth, shrink crop,
// raise segments, THEN reduce B with a warn, and only after all of that walk the
// depth ladder (depth is the quality axis). With ckpt_mode 0 and depth_ladder
// true this is the pre-checkpointing function, decision for decision.
static DitVramFit dit_vram_fit(const DitVramModel & vm, ggml_backend_t backend, int reserve_mb, float safety,
                               int user_crop, int user_layers, int crop_min, int crop_max, int max_T,
                               const DitGpuMem * mem = nullptr, int ckpt_mode = 0, bool depth_ladder = true) {
    DitVramFit   r;
    DitVramModel v  = vm;  // the segment count is what this search moves
    size_t       fb = 0, tb = 0;
    if (mem) {
        fb            = mem->free;
        tb            = mem->total;
        r.free_source = mem->source();
    } else {
        lm_vram_query(backend, &fb, &tb);
    }
    r.free_mb     = fb / (1024 * 1024);
    r.total_mb    = tb / (1024 * 1024);
    r.crop_user   = user_crop > 0;
    r.layers_user = user_layers > 0;

    const double budget = ((double) fb - (double) reserve_mb * 1048576.0) * (double) (1.0f - safety);

    int cap = std::min(crop_max, max_T);
    cap -= cap % vm.patch;
    int cmin = std::max(vm.patch, crop_min);
    cmin -= cmin % vm.patch;
    if (cmin > cap) {
        cmin = cap;  // a dataset shorter than crop_min still trains on whole songs
    }

    r.budget_bytes = budget;

    // ── the floor: the cheapest configuration this walk can express ─────────
    //
    // This is the refusal number, and since 2026-09-02 it is also the GATE (the
    // flat 16 GB card check it replaced is retired at the bottom of this file).
    // It has to be the walk's own last rung or it is worthless in both
    // directions — priced too high and it refuses a card the walk could have
    // fitted, too low and it waves through a run that OOMs three ladder rungs
    // later. So every axis here mirrors the search below, decision for decision:
    //
    //   depth     K = 2 (the ladder's bottom rung) when --layers is auto; the
    //             PINNED depth when it is not, because the ladder is not allowed
    //             to go under a user's number.
    //   crop      crop_min when --crop is auto; the pinned crop otherwise, with
    //             the same max_T clamp the crop_user path applies.
    //   segments  the widest rung seg_candidates() would reach at that depth.
    //   batch     1, because ordered_fit() walks B down to 1 before it gives up.
    //
    // Adapter type/rank/dim, mirror precision and --attn are NOT floored: they
    // are the run's own choices, not axes the fit is allowed to move. That is
    // exactly why they are the levers a refusal should name.
    {
        DitVramModel f  = vm;
        f.batch         = 1;
        const int Kf    = r.layers_user ? std::max(1, std::min(user_layers, vm.n_layers))
                                        : std::min(2, std::max(1, vm.n_layers));
        int       cf    = cmin;
        if (r.crop_user) {
            cf = user_crop - (user_crop % vm.patch);
            if (cf > max_T) {
                cf = max_T - (max_T % vm.patch);
            }
        }
        if (cf < vm.patch) {
            cf = vm.patch;
        }
        f.segments       = dit_vram_widest_segments(ckpt_mode, Kf);
        r.floor_crop     = cf;
        r.floor_layers   = Kf;
        r.floor_segments = f.segments;
        r.floor_bytes    = dit_vram_total_bytes(f, cf, Kf);
    }

    v.segments = (ckpt_mode == 0) ? 1 : std::max(1, ckpt_mode);

    const int ladder[] = { vm.n_layers, 16, 8, 4, 2 };
    const int n_rungs  = (int) (sizeof(ladder) / sizeof(ladder[0]));

    // Segment counts to try, cheapest FIRST (C4: raising segments buys VRAM with
    // one extra no-grad forward, so it is only worth doing when 1 does not fit).
    // A pinned --ckpt N is a single-element list: the user's number is never
    // changed behind their back.
    auto seg_candidates = [&](int K) {
        std::vector<int> s;
        const int        kmax = std::max(1, K);
        if (ckpt_mode == 0) {
            s.push_back(1);
            return s;
        }
        if (ckpt_mode >= 2) {
            s.push_back(std::min(ckpt_mode, kmax));
            return s;
        }
        const int steps[] = { 1, 2, 4, 8, 16, 32 };
        for (int i = 0; i < (int) (sizeof(steps) / sizeof(steps[0])); i++) {
            const int q = std::min(steps[i], kmax);
            if (s.empty() || q > s.back()) {
                s.push_back(q);
            }
        }
        return s;
    };

    // Fixed axes: honour them, but still refuse an over-budget combination.
    if (r.crop_user || r.layers_user) {
        const int        K0   = r.layers_user ? std::min(user_layers, vm.n_layers) : vm.n_layers;
        int              K    = K0;
        int              crop = 0;
        bool             hit  = false;
        // Depth rungs to try: the requested depth first; lower ones only when the
        // depth is AUTO and the ladder is enabled (see the header comment).
        std::vector<int> depths;
        depths.push_back(K0);
        if (depth_ladder && !r.layers_user) {
            for (int i = 1; i < n_rungs; i++) {
                if (ladder[i] < K0) {
                    depths.push_back(ladder[i]);
                }
            }
        }
        for (size_t di = 0; di < depths.size() && !hit; di++) {
            const int Kd = depths[di];
            const std::vector<int> segs = seg_candidates(Kd);
            for (size_t si = 0; si < segs.size() && !hit; si++) {
                v.segments = segs[si];
                int c2     = user_crop;
                if (r.crop_user) {
                    c2 -= c2 % vm.patch;
                    if (c2 > max_T) {
                        c2 = max_T - (max_T % vm.patch);
                    }
                } else {
                    c2 = dit_vram_best_crop(v, Kd, budget, cmin, cap);
                }
                if (c2 > 0 && dit_vram_total_bytes(v, c2, Kd) <= budget) {
                    K    = Kd;
                    crop = c2;
                    hit  = true;
                }
            }
        }
        if (!hit) {
            // Nothing fitted: report the requested depth at the widest segment
            // count tried, so the fatal quotes the number that was actually short.
            const std::vector<int> segs = seg_candidates(K0);
            v.segments                  = segs.back();
            crop                        = r.crop_user ? std::min(user_crop - (user_crop % vm.patch),
                                                                 max_T - (max_T % vm.patch))
                                                      : dit_vram_best_crop(v, K0, budget, cmin, cap);
        }
        r.layers    = K;
        r.crop      = crop;
        r.segments  = v.segments;
        r.est_bytes = crop > 0 ? dit_vram_total_bytes(v, crop, K) : r.floor_bytes;
        r.ok        = hit && crop > 0;
        r.lowered   = K < vm.n_layers;
        return r;
    }

    // Both auto: full depth first (shrink crop, then raise segments), and only
    // then — when the caller allows it — the depth ladder, re-raising the crop
    // and re-trying the segment counts at every rung.
    for (int i = 0; i < n_rungs; i++) {
        const int K = ladder[i];
        if (K > vm.n_layers || (i > 0 && K >= ladder[i - 1])) {
            continue;
        }
        const std::vector<int> segs = seg_candidates(K);
        for (size_t si = 0; si < segs.size(); si++) {
            v.segments     = segs[si];
            const int crop = dit_vram_best_crop(v, K, budget, cmin, cap);
            if (crop > 0) {
                r.layers    = K;
                r.crop      = crop;
                r.segments  = v.segments;
                r.est_bytes = dit_vram_total_bytes(v, crop, K);
                r.ok        = true;
                r.lowered   = K < vm.n_layers;
                return r;
            }
        }
        if (!depth_ladder) {
            break;
        }
    }
    r.layers    = 2;
    r.crop      = 0;
    r.segments  = v.segments;
    r.est_bytes = r.floor_bytes;
    r.ok        = false;
    return r;
}

// ─── D9 RETIRED (2026-09-02) ────────────────────────────────────────────────
//
// What used to live here:
//
//   #define DIT_MIN_VRAM_MB 16384
//   static bool dit_vram_card_supported(size_t total_mb) { ... }
//
// — a flat refusal of any card reporting under 16 GB TOTAL, on the grounds that
// "the K=2 floor alone (~8.3 GB of F32 mirror) exceeds a 12 GB budget". That
// sentence was true of the trainer D9 was written for. It has not been true for
// some time: `--mirror bf16` halves the term the number was derived from,
// `--ckpt` splits the activation set, the --layers ladder trims the mirror and
// the optimizer state with the depth, `--attn flash` deletes both retained
// softmaxes, and rank/dim move the adapter and its Adam state. The gate could
// not see any of that, because it ran on `total_mb` alone, BEFORE the footprint
// model had been asked what this particular run costs.
//
// The replacement is `DitVramFit::floor_bytes` versus `DitVramFit::budget_bytes`
// — the walk's own bottom rung against what the card actually has free. It is
// strictly better informed than a constant: it moves with the run's settings,
// it is the same arithmetic that sizes the run, and by construction it refuses
// exactly when the walk finds nothing (floor > budget <=> !fit.ok), so there is
// one refusal path instead of two that could disagree.
//
// WHAT THIS DOES NOT TOUCH. The floor is an ESTIMATE, and the estimate is not
// the safety net. The mandatory high-water probe (dit-train-run.h §"high-water
// probe") still builds and runs the widest graph before epoch 1, and the NVML
// tripwire still compares device-wide usage against the device total afterwards.
// Both are unchanged. Lowering an estimate-based gate is only defensible
// BECAUSE those two are there to catch the cells the polynomial gets wrong.

// Advisory only, and not consulted by any refusal: the number the server hands
// the UI for its "this build wants a card at least this big" banner. Not a
// floor — the engine's floor is per-run and computed above.
#define DIT_ADVISORY_VRAM_MB 8192

// The two or three settings that would lower this run's floor most, as text:
//   "--mirror bf16 (-6104 MB), --attn flash (-2411 MB), --lokr-dim 64 (-822 MB)"
//
// Every candidate is PRICED, never ranked by folklore — the floor configuration
// is re-costed with that one lever flipped and the difference is what gets
// printed. Levers the run has already taken are not offered, and neither is one
// that saves nothing. `allow_bf16` is the caller's business because the BF16
// mirror is CUDA-only (dit-train-run.h falls back to f32 elsewhere, so
// suggesting it on a Vulkan build would be advice that does nothing).
//
// `all_bytes` is the floor with EVERY offered lever pulled at once, priced the
// same way rather than by adding the individual savings (they interact — the
// arena term is not separable). It exists because "-47 MB" against a 3645 MB
// shortfall is advice that wastes somebody's afternoon: the caller compares
// `all_bytes` to the budget and, when even that does not fit, says so instead of
// implying the flags are worth trying. Measured on this base (2026-09-02), an
// emulated 8 GB card is exactly that case — the bf16 mirror of the frozen base
// is 7956 MB of an 8351 MB floor, so no combination of run settings reaches it.
struct DitVramAdvice {
    std::string text;
    double      all_bytes = 0.0;  // the floor with every offered lever applied
    int         n         = 0;    // how many levers were offered at all
};

static DitVramAdvice dit_vram_floor_advice(const DitVramModel & vm, const DitVramFit & fit, int ckpt_mode,
                                           bool allow_bf16, int top_n = 3) {
    struct Cand {
        std::string label;
        double      saves = 0.0;
    };
    const int    K0   = std::max(1, fit.floor_layers);
    const int    c0   = std::max(vm.patch, fit.floor_crop);
    const int    cmin = std::max(vm.patch, 128 - (128 % std::max(1, vm.patch)));
    DitVramModel base = vm;
    base.batch        = 1;
    base.segments     = fit.floor_segments;
    const double b0   = fit.floor_bytes;

    // Which levers apply. Decided once, so the individual pricing below and the
    // all-at-once pricing cannot disagree about what was offered.
    const bool lv_bf16  = allow_bf16 && base.mirror != DIT_MIRROR_BF16;
    const bool lv_flash = !base.flash_attn;
    const bool lv_ckpt  = ckpt_mode == 0 && K0 > 1;
    const bool lv_dim   = base.is_lokr && base.lokr_dim > 64;
    const bool lv_rank  = !base.is_lokr && base.rank > 8;
    const bool lv_mlp   = base.target_mlp;
    const bool lv_depth = K0 > 2;  // only reachable when --layers was PINNED above the bottom rung
    const bool lv_crop  = c0 > cmin;

    // `all` accumulates every applicable lever; each individual candidate is a
    // fresh copy of `base` with exactly one flipped.
    DitVramModel all    = base;
    int          all_K  = K0;
    int          all_c  = c0;

    std::vector<Cand> cands;
    auto price = [&](const char * label, DitVramModel m, int crop, int K) {
        m.batch        = 1;
        const double b = dit_vram_total_bytes(m, crop, K);
        if (b0 - b > 8.0 * 1048576.0) {  // below ~8 MB it is noise, not advice
            Cand cd;
            cd.label = label;
            cd.saves = b0 - b;
            cands.push_back(cd);
        }
    };

    if (lv_bf16) {
        DitVramModel m = base;
        m.mirror = all.mirror = DIT_MIRROR_BF16;
        price("--mirror bf16", m, c0, K0);
    }
    if (lv_flash) {
        DitVramModel m = base;
        m.flash_attn = all.flash_attn = true;
        price("--attn flash", m, c0, K0);
    }
    if (lv_ckpt) {
        DitVramModel m = base;
        m.segments = all.segments = dit_vram_widest_segments(1, K0);
        price("--ckpt 1 (auto gradient checkpointing)", m, c0, K0);
    }
    if (lv_dim) {
        DitVramModel m = base;
        m.lokr_dim = all.lokr_dim = 64;
        price("--lokr-dim 64", m, c0, K0);
    } else if (lv_rank) {
        DitVramModel m = base;
        m.rank = all.rank = 8;
        price("--rank 8", m, c0, K0);
    }
    if (lv_mlp) {
        DitVramModel m = base;
        m.target_mlp = all.target_mlp = false;
        price("--no-target-mlp", m, c0, K0);
    }
    if (lv_depth) {
        DitVramModel m = base;
        m.segments = dit_vram_widest_segments(ckpt_mode, 2);
        all_K      = 2;
        all.segments = m.segments;
        price("--layers 2", m, c0, 2);
    }
    if (lv_crop) {
        all_c = cmin;
        price(fit.crop_user ? "--crop 128" : "--crop-min 128", base, cmin, K0);
    }

    DitVramAdvice out;
    out.all_bytes = dit_vram_total_bytes(all, all_c, all_K);
    out.n         = (int) cands.size();
    std::sort(cands.begin(), cands.end(), [](const Cand & a, const Cand & b) { return a.saves > b.saves; });
    const int n = std::min((int) cands.size(), std::max(1, top_n));
    for (int i = 0; i < n; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s%s (-%lld MB)", i ? ", " : "", cands[(size_t) i].label.c_str(),
                 (long long) (cands[(size_t) i].saves / 1048576.0));
        out.text += buf;
    }
    return out;
}
