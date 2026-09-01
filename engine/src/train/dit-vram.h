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
    int  lokr_dim    = 512;
    int  lokr_factor = 6;
    // Mirror precision: the fixed term is the mirror, so this is the single
    // largest lever in the whole model (--mirror bf16 roughly halves it).
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
// MEASURED 2026-09-01 on an RTX 5090 (32 GB), dit-xl-thirds BF16, rank-16 LoRA,
// --layers 32, --mirror bf16, B = 1, against `ace-train`'s own high-water arena
// figure — 20 cells: {crop 375, 750, 1500, 3000} x {segments 1, 8, 32} on
// `president` (enc_S 609), plus {crop 750, 1500} x {segments 1, 8} on
// `joycemanor_neverhungoveragain` (enc_S 434) and `nwa_straightoutta`
// (enc_S 1877), which is what separates the enc_S term from the rest. The raw
// (pre-headroom) model reproduces all 20 within -2.0 % / +1.3 %; with the
// headroom below every cell over-predicts, by +9.5 % to +13.4 %, none under.
// UNMEASURED past this grid: B > 1, LoKR, rank > 16, --target-mlp, and any base
// with a different Nh/Nkv/D. The backstop for those is unchanged — the mandatory
// high-water probe and the NVML tripwire.
//
// docs/plans/2026-09-01-flash-attn-backward.md (MEASUREMENT 2), spec §11.

// Non-attention retained activations, MB per token per layer. The flash-mode
// counterpart of the exact model's 0.3274, measured the same way. (Sanity: 0.1978
// MB is 20.3 hidden-widths at H = 2560, f32 — the qkv/o/mlp/norm/residual set of
// one layer's forward plus the gradients ggml keeps live, which is the right
// order of magnitude.)
#define DIT_FLASH_ACT_MB_PER_TOKEN 0.19779

// The ONE fitted number in this branch. It is a safety coefficient, not a shape:
// the terms above already reproduce every measured cell to ±2 %, and this makes
// the estimate land on the over-predicting side of all of them, which is the side
// the reserve, the safety margin and the NVML tripwire are built to back up.
#define DIT_FLASH_HEADROOM 1.12

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
    const double machinery_mb  = scale_acc_mb + std::max(back_self_mb, back_cross_mb);

    const double mb = (double) std::max(0, K) * per_layer_mb + machinery_mb;
    return (mb > 0.0 ? mb : 0.0) * DIT_FLASH_HEADROOM * MB * (double) std::max(1, B);
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
static double dit_vram_boundary_bytes(const DitVramModel & vm, int S) {
    if (vm.segments <= 1) {
        return 0.0;
    }
    const double one = (double) vm.hidden * (double) S * (double) std::max(1, vm.batch) * 4.0;
    return one * (double) (vm.segments + 2);
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
static double dit_vram_kv_expand_bytes(const DitVramModel & vm, int S, int K) {
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
    double    b   = dit_vram_fixed_bytes(vm, K, crop) + arena +
               dit_vram_kv_expand_bytes(vm, S, Kr) + dit_vram_boundary_bytes(vm, S);
    if (vm.is_lokr) {
        // The intermediates the fitted polynomial never saw. Added here rather
        // than at any one call site so every consumer — the crop walk, the depth
        // ladder, the `vram` JSONL est and dit_train_log.json — sees them too.
        // C4: its `S` is already "tokens seen by apply()", which batching multiplies,
        // and its layer range is the RETAINED one, which checkpointing divides.
        b += dit_lokr_apply_arena_bytes(vm.m->cfg, vm.n_layers - Kr, vm.n_layers, vm.lokr_dim, vm.lokr_factor,
                                        vm.target_mlp, S * std::max(1, vm.batch));
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
#define DIT_REPEAT_BACK_MAX 32768

static int dit_vram_max_batch(int n_heads, int n_kv_heads, int S, int enc_S, int want) {
    const int tok = std::max(S, enc_S);
    if (want <= 1 || n_kv_heads <= 0 || n_heads == n_kv_heads || tok <= 0) {
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
    double floor_bytes  = 0.0;    // fixed footprint at K=2, crop_min — the refusal number
};

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

    v.segments    = (ckpt_mode == 0) ? 1 : std::max(1, ckpt_mode);
    r.floor_bytes = dit_vram_total_bytes(v, cmin, 2);

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

// D9: 12 GB does not fit at ANY depth (the K=2 floor alone exceeds the budget).
// Refuse it by name rather than letting the ladder produce a cryptic vram fatal.
#define DIT_MIN_VRAM_MB 16384

static bool dit_vram_card_supported(size_t total_mb) {
    return total_mb + 512 >= (size_t) DIT_MIN_VRAM_MB;  // 512 MB slack for reporting rounding
}
