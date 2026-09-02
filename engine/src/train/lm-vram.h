#pragma once
// lm-vram.h — the footprint model, the auto-fit binary search (L8), the
// high-water probe and the leak counter (L10).
//
// Footprint model (plan §3.7), everything in F32 bytes:
//
//   bytes(S, s_tr) = mirror
//                  + 3 * V * s_tr * 4                 // logits + logits-grad + labels
//                  + 4 * n_lora * 4                   // params + accs + m + v
//                  + L * ( c1*Nh*S*S + (c2f*F + c2h*H)*S ) * 4
//                  + static                           // causal mask + token/pos inputs
//   with c1 = 1.135, c2f = 2.4, c2h = 1.0
//
// c1/c2 are calibrated on the two measured 0.6B points (S=1536 -> 6,351 MB of
// activations; S=2113 -> 11,174 MB) and reproduce both by construction.
//
// DEVIATION (documented): the plan's formula omits the persistent causal-mask
// buffer. It is real (S*S*4 bytes: 17.8 MB at S=2113, 67 MB at S=4096) and is
// added here as an explicit `static` term. It is ~0.1 % of the total at the
// calibration points, so the c1/c2 fit is unaffected.
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §3.7

#include "train/lm-common.h"

#include <algorithm>
#include <string>

struct LmVramModel {
    // model geometry
    int    n_layers = 0;
    int    hidden   = 0;
    int    ffn      = 0;
    int    n_heads  = 0;
    int    vocab    = 0;
    size_t mirror_bytes = 0;
    size_t lora_params  = 0;

    // ── --attn flash (D9) ────────────────────────────────────────────────
    //
    // PURELY ADDITIVE. `attn_flash` defaults false and every expression below
    // branches on it, so an exact-mode run produces the byte-identical estMb /
    // maxLen integers it produced before this field existed (gate G0).
    //
    // head_dim / n_kv_heads are needed ONLY by the flash branch: the fused op's
    // retained state is per-head-dim, and its packed gradient is GQA-shaped
    // (dK/dV are Nkv-wide, not Nh-wide). The shipped exact polynomial never
    // needed either, which is why they were not here.
    bool   attn_flash = false;
    int    head_dim   = 0;
    int    n_kv_heads = 0;

    static constexpr double c1  = 1.135;
    static constexpr double c2f = 2.4;
    static constexpr double c2h = 1.0;
};

// ─── flash-attention footprint constants (--attn flash, D9) ─────────────────
//
// DERIVED FROM THE OP, NOT FITTED — and that is a statement about their status,
// not a claim about their accuracy. They are counted straight off
// engine/ggml/src/ggml.c's packing helpers and ggml's autodiff, and they are the
// same three terms the DiT branch charges (dit-vram.h:246-272 — `fused_mb`,
// `scale_acc_mb`, `back_*_mb`), where they were refitted against 48 measured
// cells. Splitting them the same way here is deliberate: it is the only version
// of this arithmetic that has ever been held against a measurement.
//
//   PACKED — the fused FORWARD's single output tensor, retained until the
//   matching backward node runs (it is that node's src[4] `fwd` source, so
//   gallocr cannot free it earlier):
//       O    [D, S, Nh]  ->  D*Nh floats per token
//       LSE  [Nh, S]     ->  Nh   floats per token
//       PACKED = D*Nh + Nh floats per token
//
//   the gradient OF THAT FORWARD PACKED TENSOR, materialised TWICE. dO arrives
//   through the O view, whose backward goes through ggml_acc_or_set
//   (ggml.c:6829-6834): it builds ggml_scale(packed, 0) and then
//   ggml_acc_impl(that, dO, ..., inplace=false), so two PACKED-SHAPED buffers
//   are live at once (flash-attn-training SKILL.md trap 3):
//       2 * PACKED floats per token
//
//   BWD — the packed dQ|dK|dV that ggml_flash_attn_train_back (ggml.c:7434)
//   produces. ONE tensor, ONE copy, GQA-shaped (dK/dV are Nkv-wide):
//       D*(Nh + 2*Nkv) floats per token
//
// So an attention site whose backward is being built holds 3*PACKED + 1*BWD
// floats per token; a forward-only site holds 1*PACKED.
//
// WHY THE SPLIT IS NOT BOOKKEEPING. The first version of this header put the x2
// on BWD instead of on PACKED and omitted the scale/acc pair entirely. That errs
// by (2*PACKED - BWD) = D*Nh + 2*Nh - 2*D*Nkv floats per token, which is
// POSITIVE — an UNDER-prediction, the direction D9 and skill §5 forbid — for
// every base with Nh > 2*Nkv. Qwen3 4B is exactly that shape (D 128, Nh 32,
// Nkv 8): 4128 + 2*6144 = 16416 floats/token against a true 3*4128 + 6144 =
// 18528, so even the whole 12 % margin (18386) landed 0.8 % UNDER, before any
// allocator overhead. At Nh == 2*Nkv the D terms cancel and only -2*Nh survives
// (32 floats/token at 0.6B/1.7B, 16/8 — 0.3 %, invisible inside the margin),
// which is why the naive path never showed it and the low-VRAM path, the one
// that exists FOR the 4B, did.
//
// SANITY, one backward segment at the 4B low-VRAM headline geometry
// (D 128, Nh 32, Nkv 8, S 3500, head-block off):
//       1.12 * (3*4128 +   6144) * 3500 * 4 = 290 MB   <- this header
//              (3*4128 +   6144) * 3500 * 4 = 259 MB   <- unmargined truth
//       1.12 * (  4128 + 2*6144) * 3500 * 4 = 257 MB   <- the old arithmetic, under
//
// THE RULE FOR THESE NUMBERS (skill §5): the estimate must OVER-predict, never
// under, by +5-15 %. The high-water probe and the NVML tripwire are the
// backstop, never the plan. `margin` carries that headroom explicitly so the
// structural terms above stay readable as arithmetic.
//
// TO BE RE-FITTED IN THE BUILD PHASE. `bwd_live_layers` is the one genuinely
// uncertain coefficient on the naive path: the gradient machinery is transient
// per layer, but how many layers' worth ggml_gallocr keeps alive simultaneously
// depends on its reuse decisions, not on arithmetic. 2.0 is "the layer being
// differentiated plus one in flight"; the fully-conservative bound is n_layers,
// and the honest answer comes from G4 — flash est vs the high-water probe at
// S in {1024, 3500} (4B lowvram) and {2048, 3800} (0.6B naive). If G4 measures
// UNDER-prediction, raise this first; if it measures more than +15 %, lower it.
struct LmVramFlash {
    // PACKED: floats per token per attention site that the fused FORWARD leaves
    // live (O then LSE, one tensor).
    static double fwd_floats_per_token(int D, int Nh) {
        return (double) D * (double) Nh   // O
               + (double) Nh;             // LSE
    }
    // BWD: floats per token for the ONE packed dQ|dK|dV tensor the fused
    // backward produces. GQA-shaped — dK/dV are Nkv-wide, not Nh-wide.
    static double bwd_floats_per_token(int D, int Nh, int Nkv) {
        return (double) D * ((double) Nh + 2.0 * (double) Nkv);
    }

    // Copies of the FORWARD packed tensor's gradient that autodiff materialises:
    // ggml_scale(packed, 0) and the non-inplace ggml_acc(dO) on top of it. It
    // multiplies fwd_floats_per_token, NOT the backward term — they are
    // different shapes and the 4B geometry is where that difference bites.
    static constexpr double grad_fwd_copies = 2.0;   // scale(packed,0) + acc(dO)

    // ── FITTED AT G4, 2026-09-03 (see the table below) ───────────────────
    //
    // bwd_live_layers 2.0 -> 5.0. The pre-fit guess was "the layer being
    // differentiated plus one in flight"; gallocr keeps about five. Measured on
    // the 0.6B naive path at S 1824 and 3915 by differencing the flash-mode
    // per-token deficit against the exact-mode one at the same S (the shared
    // non-attention term cancels): flash carries 0.103 MB/token of attention
    // state the old coefficient did not charge, and 3.0 extra layers' worth of
    // (2*PACKED + BWD) is 0.105 MB/token. Still well under the fully-
    // conservative n_layers = 28 bound.
    static constexpr double bwd_live_layers = 5.0;

    // The SHARED non-attention activation polynomial ((c2f*ffn + c2h*hidden)*S
    // per layer) is 2.2x UNDER-counted on the naive path, and has been since
    // before this flag existed. G4, per-token deficit (measured high-water minus
    // estimate, divided by S):
    //
    //     0.6B naive  S 1708  exact  1.135 MB/token under
    //     0.6B naive  S 1824  exact  1.109 MB/token under
    //     0.6B naive  S 1824  flash  1.213 MB/token under
    //     0.6B naive  S 3915  flash  1.213 MB/token under
    //
    // Flat in S across a 2.15x span, and present in EXACT mode too — so it is
    // not an attention term and not something --attn introduced. The shipped
    // polynomial charges 0.897 MB/token at 0.6B against a real ~2.0, i.e. c2f
    // and c2h are both about 2.2x light.
    //
    // WHY THE CORRECTION IS GATED ON attn_flash. Fixing c2f/c2h themselves would
    // move exact-mode estMb and maxLen, which D2/G0 forbid — exact mode must
    // stay byte-identical to the pre-flag binary. So the corrected polynomial
    // lives on the flash branch only, exactly as the DiT did it (skill S5:
    // "exact-mode arena polynomial under-predicts 13-18% ... fix gated to
    // flash"). The exact-mode refit is OWED and needs its own gate, because it
    // changes what every shipped naive run auto-fits to.
    //
    // 2.67 rather than the bare 2.24 the deficit implies: the extra 0.43 carries
    // the mandated +5-15 % over-prediction on the naive TOTAL, which `margin`
    // cannot do on its own (margin multiplies only the attention term, and on
    // this path that term is a fifth of the estimate).
    //
    // FITTED ACROSS BOTH NAIVE BASES, not just one. It scales the polynomial
    // rather than adding a per-token constant so it can travel between widths,
    // and it had to: the four cells each admit their own window of s, and only
    // their intersection [2.58, 2.77] satisfies all of them. 2.67 is the middle
    // of it. Measured est-vs-high-water after the refit:
    //
    //     0.6B naive  S 1824   est 12044  measured 11332   +6.3 %
    //     0.6B naive  S 3915   est 21725  measured 20197   +7.6 %
    //     1.7B naive  S 1278   est 16158  measured 14424   +12.0 %
    //     1.7B naive  S 1824   est 21068  measured 18596   +13.3 %
    //
    // The 1.7B cells sit at the top of the band because the non-attention term
    // is a larger share of that base's estimate, so the same scale buys more
    // headroom there. A per-token constant ALONGSIDE the scale would flatten
    // that, but four cells do not justify a second free parameter.
    static constexpr double naive_nonattn_scale = 2.67;

    // Applies to the attention term only. Left at the derived value: G4 measured
    // the 4B low-VRAM flash estimate over-predicting by a FLAT 0.052 MB/token at
    // S 1004 / 1708 / 2418 / 3915 (+0.5 % to +1.6 % of the total), so the
    // structural terms are right and there is nothing to fit. See the note on
    // lm_vram_flash_segment_bytes for why the +5-15 %-of-total bar is not the
    // right bar on that path.
    static constexpr double margin          = 1.12;
};

// The whole-trunk (naive autodiff) flash attention term, in bytes. Charged as
// L retained forward packed states — every layer's O/LSE really is held across
// the backward — plus `bwd_live_layers` layers' worth of gradient machinery,
// which is 2 forward-shaped copies plus one packed dQ|dK|dV each.
static double lm_vram_flash_activation_bytes(const LmVramModel & m, int S) {
    const double dS   = (double) S;
    const double pack = LmVramFlash::fwd_floats_per_token(m.head_dim, m.n_heads);
    const double bwd  = LmVramFlash::bwd_floats_per_token(m.head_dim, m.n_heads, m.n_kv_heads);
    const double retained = pack * (double) m.n_layers;
    const double machinery =
        (LmVramFlash::grad_fwd_copies * pack + bwd) * LmVramFlash::bwd_live_layers;
    return LmVramFlash::margin * (retained + machinery) * dS * 4.0;
}

static double lm_vram_activation_bytes(const LmVramModel & m, int S) {
    const double dS = (double) S;
    // D9: under flash the c1*Nh*S^2 retained-softmax term does not exist — the
    // fused op never materialises it — and is replaced by a per-token linear
    // term. The non-attention term (c2f*F + c2h*H)*S is untouched: nothing
    // outside the attention span changed.
    if (m.attn_flash) {
        // naive_nonattn_scale: the shared polynomial is 2.2x light and always
        // was (G4 measured it in exact mode too). Corrected here, on the branch
        // D2 lets us move. See LmVramFlash for the measurements.
        return LmVramFlash::naive_nonattn_scale * (double) m.n_layers *
                   ((LmVramModel::c2f * (double) m.ffn + LmVramModel::c2h * (double) m.hidden) * dS) * 4.0 +
               lm_vram_flash_activation_bytes(m, S);
    }
    const double per_layer =
        LmVramModel::c1 * (double) m.n_heads * dS * dS +
        (LmVramModel::c2f * (double) m.ffn + LmVramModel::c2h * (double) m.hidden) * dS;
    return (double) m.n_layers * per_layer * 4.0;
}

// `flash` halves the mask term: the persistent [S*S] causal buffer is F16 under
// --attn flash (D4, lm_mask_alloc), because the fused op will not take an F32
// mask. Default false, so every shipped caller keeps its exact byte count.
static double lm_vram_static_bytes(int S, bool flash = false) {
    return (double) S * (double) S * (flash ? 2.0 : 4.0)   // causal mask [S*S] f16/f32
           + (double) S * 4.0 * 2.0;                       // tokens + positions (i32)
}

static double lm_vram_bytes(const LmVramModel & m, int S, int s_tr) {
    return (double) m.mirror_bytes                                    //
           + 3.0 * (double) m.vocab * (double) s_tr * 4.0             // logits + grad + labels
           + 4.0 * (double) m.lora_params * 4.0                       // params + acc + m + v
           + lm_vram_activation_bytes(m, S)                           //
           + lm_vram_static_bytes(S, m.attn_flash);
}

// 0.62 = 1272/2113, the measured trained fraction of a real full-song sequence.
// It only sizes the label buffer, and the graph views a prefix of it, so a low
// estimate is self-correcting while a high one merely wastes VRAM.
static inline int lm_vram_str_est(int S) {
    return (int) ((double) S * 0.62 + 0.5);
}

struct LmVramFit {
    int    max_len   = 0;
    double est_bytes = 0.0;
    size_t free_mb   = 0;
    size_t total_mb  = 0;
    bool   ok        = false;
};

// Largest S in [1024, 8192] (step 64) whose predicted footprint fits the
// budget. `user_len > 0` skips the search but still checks the budget.
static LmVramFit lm_vram_fit(const LmVramModel & m, ggml_backend_t backend, int reserve_mb, int user_len) {
    LmVramFit r;
    size_t    fb = 0, tb = 0;
    lm_vram_query(backend, &fb, &tb);
    r.free_mb  = fb / (1024 * 1024);
    r.total_mb = tb / (1024 * 1024);

    // `fb` is queried AFTER lm_build_f32_mirror() allocated the mirror and freed
    // the BF16 buffer, so the mirror is ALREADY subtracted from it. lm_vram_bytes()
    // also includes mirror_bytes, so the budget must add it back — otherwise the
    // mirror is charged twice and maxLen collapses (0.6B 2944 -> ~3200 once fixed;
    // 1.7B ~1891 -> ~2515, i.e. most of a real dataset would be skipped as
    // over-length). The plan's §3.7 pseudocode carries the same defect.
    const double budget = (double) fb + (double) m.mirror_bytes - (double) reserve_mb * 1048576.0;

    if (user_len > 0) {
        r.max_len   = user_len;
        r.est_bytes = lm_vram_bytes(m, user_len, lm_vram_str_est(user_len));
        r.ok        = r.est_bytes <= budget;
        return r;
    }

    int best = 0;
    for (int S = 1024; S <= 8192; S += 64) {
        if (lm_vram_bytes(m, S, lm_vram_str_est(S)) <= budget) {
            best = S;
        } else {
            break;  // monotonically increasing in S
        }
    }
    r.max_len   = best > 0 ? best : 1024;
    r.est_bytes = lm_vram_bytes(m, r.max_len, lm_vram_str_est(r.max_len));
    r.ok        = best > 0;
    return r;
}

// ─── low-VRAM footprint model (4B plan §3.8) ────────────────────────────────
//
// PURELY ADDITIVE. Everything above this line is byte-identical to the shipped
// header (dit-vram.h reuses LmVramTracker, and gate (e) compares the naive
// maxLen integer bit-for-bit).
//
//   resident(S) = base_bytes                       // BF16 base, never mirrored
//               + emb_t_bytes                      // [V,H] transpose for dL/dh
//               + L * H * S * 4                    // checkpoints C[0..L-1]
//               + 3 * H * S * 4                    // t_H, t_G(==Gh0), Gh1
//               + Nh * D * S * 4                   // t_zero_attn (0 when hb off)
//               + S*S*4 + 2*S*4                    // causal mask + tok/pos
//               + V * chunk * 4                    // per-chunk one-hot labels
//               + 4 * n_lora * 4                   // params + acc + m + v
//
//   seg_bwd(S)  = w_layer_f32 + 3*hb*S*S*4 + 2*(c2f*F + c2h*H)*S*4 + 2*Nh*D*S*4
//   seg_fwd(S)  = w_layer_f32 + 2*hb*S*S*4 +   (c2f*F + c2h*H)*S*4 + 2*Nh*D*S*4
//   head        = 3 * V * chunk * 4
//   bytes       = resident + max(seg_bwd, seg_fwd, head)
//
// c2f/c2h are the [M]-calibrated constants of the naive model, reused verbatim.
// The 3x/2x multipliers and the hb*S^2 term are DERIVED from the op-level
// analysis in D6 (soft_max_ext_back holds `sm`, `grad_sm` and `d_scores` live
// simultaneously), NOT fitted. Honest uncertainty band: +-15 %.
//
// UNDER --attn flash (D9) the two hb*S^2 terms are ZERO — the fused op never
// materialises a score or softmax array — and are replaced by
// lm_vram_flash_segment_bytes(), which is linear in S. That is the whole reason
// the flag exists on the low-VRAM path: at 4B / S 3500 / hb 8 the 3*hb*S^2 term
// alone is ~1.18 GB per segment.
//
// DEVIATION vs plan §1.1: LmVramLowCfg carries `head_dim` and `layer_w_bytes`
// in addition to the four fields the plan lists. That was originally because
// LmVramModel had no head_dim; it now has one (the flash branch needs it there
// too), but lc keeps its copy so no shipped expression moves.

enum LmVramMode { LM_VRAM_NAIVE, LM_VRAM_LOWVRAM };

struct LmVramLowCfg {
    int    attn_head_block = 0;
    int    chunk           = 128;
    int    head_dim        = 0;
    size_t base_bytes      = 0;  // ggml_backend_buffer_get_size(lm.wctx.buffer)
    size_t emb_t_bytes     = 0;  // V * H * ggml_type_size(embed dtype)
    size_t layer_w_bytes   = 0;  // lm_layer_weight_bytes(cfg) — the F32 window

    // ── Lever A (2026-07-28 speed-levers plan §3.5) ──────────────────────
    // Default OFF, so lm_vram_lowvram_transient() below selects layer_w_bytes
    // exactly as it ships and the §6.0 estMb integer is unchanged.
    size_t layer_wt_bytes  = 0;      // lm_layer_proj_bytes(cfg, BF16) — the BF16
                                     // transposed window that replaces the F32 one
    bool   weights_bf16    = false;

    // ── --attn flash (D9) ────────────────────────────────────────────────
    // Additive and default-off, exactly like the lever above: with attn_flash
    // false every number below is the shipped one. n_kv_heads is needed only by
    // the flash branch (the packed gradient is GQA-shaped); head_dim is already
    // here for t_zero_attn.
    bool   attn_flash      = false;
    int    n_kv_heads      = 0;
};

static double lm_vram_lowvram_resident(const LmVramModel & m, const LmVramLowCfg & lc, int S) {
    const double dS = (double) S;
    const double hbz =
        (lc.attn_head_block > 0) ? (double) m.n_heads * (double) lc.head_dim * dS * 4.0 : 0.0;  // t_zero_attn
    return (double) lc.base_bytes + (double) lc.emb_t_bytes                                //
           + (double) m.n_layers * (double) m.hidden * dS * 4.0                            // C[0..L-1]
           + 3.0 * (double) m.hidden * dS * 4.0                                            // t_H, t_G, Gh1
           + hbz                                                                           //
           + lm_vram_static_bytes(S, lc.attn_flash)                                        // mask + tok/pos
           + (double) m.vocab * (double) lc.chunk * 4.0                                    // t_labc
           + 4.0 * (double) m.lora_params * 4.0;                                           // param/acc/m/v
}

// Per-SEGMENT flash attention bytes. A low-VRAM segment builds exactly ONE
// layer, so unlike the naive whole-trunk term there is one attention site live,
// not L of them — the `bwd_live_layers` uncertainty does not apply here, which
// makes this the better-conditioned of the two flash branches.
//
// A backward segment holds 3*PACKED + 1*BWD floats per token (the retained
// forward, the scale(packed,0)/acc(dO) pair on its gradient, and the fused
// backward's single dQ|dK|dV); a forward-only segment holds the retained
// forward alone. See the derivation above for why the x2 rides on PACKED.
//
// G4, 2026-09-03 — MEASURED, AND WHY THIS PATH KEEPS ITS COEFFICIENTS. 4B
// low-VRAM, --attn flash, est vs the run's own high-water probe:
//
//     S 1004   est 10740   measured 10688   +0.49 %   over by 0.0518 MB/token
//     S 1708   est 11229   measured 11140   +0.80 %   over by 0.0521 MB/token
//     S 2418   est 11723   measured 11599   +1.07 %   over by 0.0513 MB/token
//     S 3915   est 12773   measured 12571   +1.61 %   over by 0.0516 MB/token
//
// The per-token error is FLAT to three digits across a 3.9x span of S: the
// structural terms are right and there is no coefficient to fit. What is not
// right is the gate's phrasing. G4 asks for +5-15 % on the TOTAL, but ~70 % of
// this total is the resident BF16 base (7991 MB at 4B) — a number the loader
// measured, not one this header estimated. Over-predicting the total by 5 %
// would mean deliberately mispricing 8 GB of known allocation by 17 % and
// charging the user ~1.5 GB of crop for it. The bar belongs on the estimated
// part, and on that reading this branch passes at every S.
//
// For scale, the SHIPPED exact branch at the same three geometries UNDER-
// predicts: -0.90 % (S 1004), -1.74 % (S 1708), -4.94 % (S 3915). Flash is on
// the safe side of the measurement everywhere exact is on the wrong side of it.
// That comparison, not the 5 % figure, is the claim worth making here.
static double lm_vram_flash_segment_bytes(const LmVramModel & m, const LmVramLowCfg & lc, int S, bool with_grad) {
    const double pack = LmVramFlash::fwd_floats_per_token(lc.head_dim, m.n_heads);
    const double f =
        pack + (with_grad ? LmVramFlash::grad_fwd_copies * pack +
                                LmVramFlash::bwd_floats_per_token(lc.head_dim, m.n_heads, lc.n_kv_heads)
                          : 0.0);
    return LmVramFlash::margin * f * (double) S * 4.0;
}

static double lm_vram_lowvram_transient(const LmVramModel & m, const LmVramLowCfg & lc, int S) {
    const double dS  = (double) S;
    const double hb  = (lc.attn_head_block > 0) ? (double) lc.attn_head_block : (double) m.n_heads;
    // D9: `s2` is the [S,S] score/softmax state soft_max_ext_back holds three
    // copies of. Under --attn flash the fused op materialises none of it, so the
    // 3x/2x hb*S^2 terms go to ZERO and are replaced by the linear per-token
    // attention state below. Exact mode is untouched.
    const double s2  = lc.attn_flash ? 0.0 : hb * dS * dS * 4.0;
    const double fa_bwd = lc.attn_flash ? lm_vram_flash_segment_bytes(m, lc, S, /*with_grad=*/true) : 0.0;
    const double fa_fwd = lc.attn_flash ? lm_vram_flash_segment_bytes(m, lc, S, /*with_grad=*/false) : 0.0;
    const double non = (LmVramModel::c2f * (double) m.ffn + LmVramModel::c2h * (double) m.hidden) * dS * 4.0;
    const double qkv = 2.0 * (double) m.n_heads * (double) lc.head_dim * dS * 4.0;  // acc + cont churn
    // §3.5: Lever A replaces the per-segment F32 weight window with a BF16
    // TRANSPOSED window over the 7 projections only — 192.5 MiB instead of
    // 385.0 MiB at 4B, i.e. the lever is VRAM-POSITIVE.
    //
    // MEASURED, not just modelled — and the two do not agree, so read the
    // measurement. LmVramTracker::peak_mb over the §6.5 runs: 4B 12,466 ->
    // 12,363 MiB, i.e. -103 MiB against the -192.5 MiB this line predicts
    // (0.6B: 3,267 -> 3,242 = -25 vs -30). Checking the estimate against itself
    // would of course reproduce -192.5 exactly; it is circular and means nothing.
    // Second-order effect that matters if this ever feeds the auto-fit: the
    // model already under-predicts the real 4B peak by 340 MiB on the f32
    // window (est 12,126 vs peak 12,466), and enabling the lever widens that to
    // 429 MiB (est 11,934 vs peak 12,363) — ~26 % less conservative. Still
    // inside the 1,024 MiB reserve, but the headroom is smaller than the
    // arithmetic suggests.
    const double w_window = (lc.weights_bf16 && lc.layer_wt_bytes > 0) ? (double) lc.layer_wt_bytes
                                                                      : (double) lc.layer_w_bytes;
    const double seg_bwd = w_window + 3.0 * s2 + fa_bwd + 2.0 * non + qkv;
    const double seg_fwd = w_window + 2.0 * s2 + fa_fwd + non + qkv;
    const double head    = 3.0 * (double) m.vocab * (double) lc.chunk * 4.0;
    return std::max(seg_bwd, std::max(seg_fwd, head));
}

// `s_tr` is accepted for signature symmetry with lm_vram_bytes(); the low-VRAM
// label buffer is `chunk`-sized, so it genuinely does not depend on s_tr.
static double lm_vram_bytes_lowvram(const LmVramModel & m, const LmVramLowCfg & lc, int S, int s_tr) {
    (void) s_tr;
    return lm_vram_lowvram_resident(m, lc, S) + lm_vram_lowvram_transient(m, lc, S);
}

// The ATTENTION share of the estimate above, so the est line can print both
// terms instead of one total.
//
// This exists because of a specific, expensive mistake on the DiT side: an
// arena line that printed only the total ("4319 est vs 7824 measured") omitted
// the LoKR-apply term and sent a whole refit chasing an under-prediction that
// did not exist — the total was over-predicting by 73 %. Skill §5: "Read the
// arena log line as the TOTAL... The line now prints both terms — keep it that
// way in every trainer."
//
// Low-VRAM returns the WORST segment's attention state (the same max() the
// transient model takes); naive returns the whole-trunk term.
static double lm_vram_attn_term_bytes(const LmVramModel & m, const LmVramLowCfg & lc, int S, bool low) {
    if (low) {
        if (lc.attn_flash) {
            return lm_vram_flash_segment_bytes(m, lc, S, /*with_grad=*/true);
        }
        const double hb = (lc.attn_head_block > 0) ? (double) lc.attn_head_block : (double) m.n_heads;
        return 3.0 * hb * (double) S * (double) S * 4.0;
    }
    if (m.attn_flash) {
        return lm_vram_flash_activation_bytes(m, S);
    }
    return (double) m.n_layers * LmVramModel::c1 * (double) m.n_heads * (double) S * (double) S * 4.0;
}

// Same binary search as lm_vram_fit(). `fb` is queried BEFORE any mirror exists
// but the base buffer is already allocated and therefore already subtracted from
// it, while lm_vram_bytes_lowvram() charges base_bytes — so, exactly as in the
// naive fit, the budget has to add it back.
static LmVramFit lm_vram_fit_lowvram(const LmVramModel & m, const LmVramLowCfg & lc, ggml_backend_t backend,
                                     int reserve_mb, int user_len) {
    LmVramFit r;
    size_t    fb = 0, tb = 0;
    lm_vram_query(backend, &fb, &tb);
    r.free_mb  = fb / (1024 * 1024);
    r.total_mb = tb / (1024 * 1024);

    const double budget = (double) fb + (double) lc.base_bytes - (double) reserve_mb * 1048576.0;

    if (user_len > 0) {
        r.max_len   = user_len;
        r.est_bytes = lm_vram_bytes_lowvram(m, lc, user_len, lm_vram_str_est(user_len));
        r.ok        = r.est_bytes <= budget;
        return r;
    }

    int best = 0;
    for (int S = 1024; S <= 8192; S += 64) {
        if (lm_vram_bytes_lowvram(m, lc, S, lm_vram_str_est(S)) <= budget) {
            best = S;
        } else {
            break;
        }
    }
    r.max_len   = best > 0 ? best : 1024;
    r.est_bytes = lm_vram_bytes_lowvram(m, lc, r.max_len, lm_vram_str_est(r.max_len));
    r.ok        = best > 0;
    return r;
}

// §3.2 step 6. `naive_max_len` is the integer the SHIPPED fit produces; passing
// 0 means "not evaluated" (the caller already forced a mode).
static inline LmVramMode lm_vram_pick_mode(const std::string & flag, const std::string & size_label,
                                           int naive_max_len) {
    if (flag == "on") {
        return LM_VRAM_LOWVRAM;
    }
    if (flag == "off") {
        return LM_VRAM_NAIVE;
    }
    if (size_label == "4B") {
        return LM_VRAM_LOWVRAM;
    }
    return (naive_max_len > 0 && naive_max_len < 2048) ? LM_VRAM_LOWVRAM : LM_VRAM_NAIVE;
}

// ─── leak counter (L10 / G3) ────────────────────────────────────────────────

// TRAINER-OWNED accounting, not device-wide.
//
// lm_vram_used_mb() is (total - free) for the whole device, so it carries every
// other process on the card (~3.2 GB of desktop apps on this machine). That
// makes `step.vramMb` incomparable with the trainer-only `estMb` printed beside
// it (§2.2's own example has them equal), and it lets an unrelated process move
// the L10 leak delta in either direction. Instead we sum the buffers we own plus
// the scheduler's own compute arena, which is exactly what a leak would grow.
struct LmVramTracker {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    size_t               fixed_bytes = 0;  // mirror + static inputs + LoRA + optimizer state
    size_t               base_mb     = 0;  // sampled right after the high-water probe
    size_t               peak_mb     = 0;
    size_t               last_mb     = 0;
    long long            max_delta   = 0;
    bool                 warned      = false;

    size_t used_mb() const {
        size_t b = fixed_bytes;
        if (sched && backend) {
            b += ggml_backend_sched_get_buffer_size(sched, backend);
        }
        return b / (1024 * 1024);
    }

    void probe_baseline(ggml_backend_t b, ggml_backend_sched_t s, size_t fixed) {
        backend     = b;
        sched       = s;
        fixed_bytes = fixed;
        base_mb     = used_mb();
        peak_mb     = base_mb;
        last_mb     = base_mb;
    }

    size_t sample() {
        last_mb = used_mb();
        if (last_mb > peak_mb) {
            peak_mb = last_mb;
        }
        const long long d = (long long) last_mb - (long long) base_mb;
        if (d > max_delta) {
            max_delta = d;
        }
        if (!warned && max_delta > 512) {
            warned = true;
            char b[192];
            snprintf(b, sizeof(b), "trainer VRAM grew %lld MB since the high-water probe — possible leak", max_delta);
            lm_log("warn", b);
        }
        return last_mb;
    }
};
