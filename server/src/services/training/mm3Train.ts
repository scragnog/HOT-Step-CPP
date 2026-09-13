// training/mm3Train.ts — MiniMax-Music3 training: paths, defaults, arg building.
//
// The MM3 analogue of aceTrain.ts. Kept apart from it deliberately: the two
// share no CLI surface (`mm3-codes` / `mm3-lm-train` take a manifest, a
// captions directory and a codes directory; `train-lm` takes a tensor cache),
// and folding them together would mean a file where half the fields are always
// unused.
//
// Spec: docs/plans/2026-08-20-mm3-training-server-design.md §2, §3.

import fs from 'fs';
import path from 'path';

import { config } from '../../config.js';
import { datasetDir } from './paths.js';
import type { Mm3PreviewOptions } from './types.js';

/** The MM3 model files training needs, resolved under <models>/mm3/.
 *
 *  CORRECTION (2026-08-21): this comment used to say a quantized base "cannot
 *  be trained" because out_prod is F32-only. That was true of the code, not of
 *  the maths. The default lm_linear path emits `mul_mat(qwen3_f32(w), x)` — an
 *  in-graph cast that gallocr frees with the segment — so the backward never
 *  sees the quantized tensor, only the cast's F32 output. That is QLoRA's
 *  dequantize-per-matmul, and it now works.
 *
 *  The base is a CHOICE. It used to be a real trade; since the
 *  cpy-q-occupancy patch (engine/patches/) it is not (5090, production recipe):
 *
 *                     base    VRAM used        free      s/step
 *      f16       16.0 GB   31.0/32.6 GB    1.5 GB    3.7   (idle card)
 *      q8_0       8.5 GB   22.6/32.6 GB   10.0 GB    3.75
 *
 *  q8_0 was 11.1 s/step before that patch, which is the ONLY reason f16 used to
 *  be the default. Now they are the same speed, so the headroom decides it:
 *  f16's 1.5 GB does not survive a desktop with a browser open. Measured on a
 *  card holding ~3.2 GB of other work, the SAME 12 steps ran 3.75 s/step on
 *  q8_0 and 12-14 s/step on f16, because f16 was paging over WDDM.
 *
 *  Accuracy is not the trade-off it sounds like: the trained thing is the F32
 *  LoRA, the base is frozen, and a same-seed 12-step A/B agreed to ~4 s.f. at
 *  every step (max relative loss deviation 5.7e-4). */
export interface Mm3TrainModels {
  lm: string;
  depth: string;
  /** Audio -> RVQ codes encoder. Any mm3-rvq-*.gguf; newest wins. */
  rvq: string;
  /** DAV encoder (audio -> latents), the mm3-codes input stage. */
  enc: string;
  /** rec7 state encoder (audio -> LM frame hiddens), the launder stage. Any
   *  mm3-rec7-*.gguf; newest wins. Empty string when none is installed. */
  rec7: string;
}

function mm3ModelDir(): string {
  return path.join(config.aceServer.models, 'mm3');
}

/** Newest file matching a prefix, or '' — used for the RVQ encoder, whose
 *  filename carries the checkpoint name rather than a fixed quant token. */
function newestMatching(dir: string, prefix: string): string {
  try {
    const hits = fs.readdirSync(dir)
      .filter(f => f.startsWith(prefix) && f.endsWith('.gguf'))
      .map(f => ({ f, m: fs.statSync(path.join(dir, f)).mtimeMs }))
      .sort((a, b) => b.m - a.m);
    return hits.length ? path.join(dir, hits[0].f) : '';
  } catch {
    return '';
  }
}

/** The quant token from `mm3-lm-<token>.gguf` — 'f16', 'q8_0', 'Q4_K_M', ...
 *
 *  Deliberately an open string rather than a union. The set of trainable bases
 *  is now "whatever the user has installed that CUDA can dequantize", which the
 *  quant-cpy-kquant patch made large (every K-quant, IQ type, MXFP4, NVFP4) and
 *  which the Model Manager can grow at any time. A union here would need
 *  editing every time someone downloads a new quant. Validation happens against
 *  what is ON DISK instead — see availableMm3Bases(). */
export type Mm3BasePrecision = string;

export function resolveMm3TrainModels(base: Mm3BasePrecision = 'f16'): Mm3TrainModels {
  const dir = mm3ModelDir();
  return {
    lm:    path.join(dir, 'mm3-lm-' + base + '.gguf'),
    depth: path.join(dir, 'mm3-depth-f16.gguf'),
    rvq:   newestMatching(dir, 'mm3-rvq-'),
    enc:   newestMatching(dir, 'mm3-enc-'),
    rec7:  newestMatching(dir, 'mm3-rec7-'),
  };
}

/** Which of the required files are missing, as user-facing names. Empty = ready.
 *  `need` narrows the check: the codes job does not need the LM. */
export function missingMm3TrainModels(need: 'codes' | 'train' | 'launder',
                                      base: Mm3BasePrecision = 'f16'): string[] {
  const m = resolveMm3TrainModels(base);
  const wanted: Array<[string, string]> = need === 'codes'
    ? [[m.rvq, 'an RVQ encoder (mm3-rvq-*.gguf)'], [m.enc, 'the DAV encoder (mm3-enc-*.gguf)']]
    : need === 'launder'
    // The launder runs the whole render stack: DAV in, rec7 states, the
    // released depth chain (which needs the LM's token embedding resident),
    // the flow DiT, and the champion back out.
    ? [[m.rvq, 'an RVQ encoder (mm3-rvq-*.gguf)'], [m.enc, 'the DAV encoder (mm3-enc-*.gguf)'],
       [m.rec7, 'the rec7 state encoder (mm3-rec7-*.gguf)'],
       [path.join(mm3ModelDir(), 'mm3-lm-q8_0.gguf'), 'mm3-lm-q8_0.gguf'],
       [m.depth, 'mm3-depth-f16.gguf']]
    : [[m.lm, path.basename(m.lm)], [m.depth, 'mm3-depth-f16.gguf']];
  return wanted.filter(([p]) => !p || !fs.existsSync(p)).map(([, label]) => label);
}

// ── Base catalogue ──────────────────────────────────────────────────────────
//
// FIDELITY IS MEASURED, NOT ASSUMED. Each `lossDelta` is the first-step
// training loss against the f16 reference on an identical seed and crop, which
// is the only comparison that isolates the base: same data, same LoRA init,
// same everything else. Numbers from a 12-step A/B on a 5090.
//
// The point of publishing them is Q2_K. It is the smallest base, it looks like
// the obvious choice on a small card, and it is the one base here that should
// not be used: +14.3% on the loss and a gradient norm twice everything else's,
// i.e. the quantizer injects more error than the LoRA is being asked to learn.
// A picker that offered it without saying so would be a trap.
interface Mm3BaseFacts {
  /** Relative first-step loss vs the f16 base. Null = not measured here. */
  lossDelta: number | null;
  quality: 'reference' | 'excellent' | 'good' | 'fair' | 'poor';
  /** Measured MB this base costs ABOVE the fitted model. Only f16 needs one:
   *  the fit is over the quantized bases, where the in-graph cast is freed with
   *  its segment, and f16 holds its weights resident instead. Without this the
   *  recommender offers f16 on a 32 GB card — which is precisely the
   *  configuration measured at 1250 MB free and 12-14 s/step. */
  extraMb?: number;
}

const MM3_BASE_FACTS: Record<string, Mm3BaseFacts> = {
  /** TRAINING ONLY, and the only base that reaches the tensor cores.
   *
   *  Every other base here trains through the F32 window: ggml_out_prod is
   *  F32-only, so lm_linear dequantizes each weight to F32 in-graph and the
   *  GEMMs run as TF32. A BF16 base lets `--weights bf16` (Lever A) feed the raw
   *  weight to mul_mat and rewrite the backward's out_prod nodes into mul_mat,
   *  so both directions use BF16 kernels.
   *
   *  MM3's published weights ARE BF16, so this is the source dtype rather than a
   *  conversion — but it is not better for INFERENCE than f16, which keeps all
   *  7 of BF16's mantissa bits and adds 3 more. Render on q8_0 as always.
   *
   *  MEASURED, and it is a bad trade on a 32 GB card. At crop 2496, matched in
   *  every other respect: 7.5 s/step against q8_0's 10.5 (1.4x) for 29.6 GB
   *  against 20.6 (+9.0). The base itself is 7.7 GB of that and Lever A's
   *  in-graph transposes are the other 1.3. It ceilings at ~3100 frames with
   *  1.2 GB to spare, where q8_0 reaches 4272 — so the speed costs 23 points of
   *  track coverage, and coverage is the thing that decides whether a render
   *  sounds like a song. Per supervised frame it is only 1.15x ahead there.
   *
   *  Pick it when coverage is NOT the binding constraint: a bigger card, a
   *  shorter corpus, or a deliberate speed run. Its gradients are also cleaner —
   *  identical step-1 loss to q8_0 (3.5930 vs 3.5932) at a 29% lower gradient
   *  norm (5.561 vs 7.870), which is the quantizer's error showing up as noise —
   *  but that has never been shown to matter by ear. */
  'bf16':   { lossDelta: 0,      quality: 'reference', extraMb: 1336 },
  'f16':    { lossDelta: 0,      quality: 'reference', extraMb: 1047 },
  'q8_0':   { lossDelta: 0.0002, quality: 'excellent' },
  'Q6_K':   { lossDelta: 0.003,  quality: 'excellent' },
  'Q5_K_M': { lossDelta: null,   quality: 'good' },
  'Q5_K_S': { lossDelta: null,   quality: 'good' },
  'Q4_K_M': { lossDelta: 0.008,  quality: 'good' },
  'Q4_K_S': { lossDelta: null,   quality: 'good' },
  'MXFP4':  { lossDelta: 0.027,  quality: 'fair' },
  'NVFP4':  { lossDelta: null,   quality: 'fair' },
  'Q3_K_L': { lossDelta: null,   quality: 'fair' },
  'Q3_K_M': { lossDelta: null,   quality: 'fair' },
  'Q3_K_S': { lossDelta: null,   quality: 'poor' },
  'Q2_K':   { lossDelta: 0.143,  quality: 'poor' },
};

export interface Mm3BaseInfo {
  id: string;
  file: string;
  bytes: number;
  lossDelta: number | null;
  quality: Mm3BaseFacts['quality'];
  /** Measured excess over the fitted model; the UI adds it to its own estimate. */
  extraMb: number;
  /** Estimated peak VRAM in MB at the rank/max-frames it was asked about. */
  peakMb: number;
}

/** Peak VRAM for a configuration, in MB.
 *
 *      peak = loaded + (31.2 + adamw ? 10.4 : 0)*rank
 *                    + 0.2679*S + 0.00044765*S^2 + 441      S = 1142 + frames
 *
 *  `loaded` is the LM file plus ~1672 MB of fixed company (depth model, audio
 *  embeddings, tokenizer) and is measured to within 7 MB. 31.2 MB per rank is
 *  12 bytes per LoRA parameter — weights, gradients and one momentum buffer,
 *  F32 each; AdamW carries a SECOND momentum buffer, which is the extra 10.4.
 *
 *  THE S TERM IS QUADRATIC, and it did not used to be (2026-08-23). The old
 *  model was `1.4515*S`, fitted over configurations that all sat near S ~ 2642
 *  (a 1500-frame crop plus the ~1142-token prompt), where it was exact: it
 *  predicts 3835 MB of activations against 3833 measured. It simply does not
 *  extrapolate — at S = 5238 it predicts 7603 MB against 13686 measured, a
 *  6 GB miss, because the non-flash attention branch the trainer backpropagates
 *  through holds [S, S, heads] scores. That range stopped being hypothetical
 *  the moment the recipe moved to 4096-frame windows, so it is now fitted
 *  through both anchors instead:
 *
 *      rank 64 / q8_0 / AdamW,  1500 frames (S=2642)  ->  17307 MB measured
 *      rank 64 / q8_0 / AdamW,  4096 frames (S=5238)  ->  27160 MB measured
 *
 *  Both anchors share a rank and a base, so any error in the per-rank or
 *  constant terms is absorbed into the two S coefficients. Treat the curve as
 *  calibrated for the shipped recipe and as an estimate elsewhere — and note
 *  the quadratic term means a longer crop is a far more expensive knob than the
 *  old linear model made it look. */
export const MM3_VRAM_MODEL = {
  /** Depth model + audio embeddings + tokenizer, on top of the LM file. */
  loadedOverheadMb: 1672,
  /** 12 bytes per LoRA parameter (weights + grads + one momentum, F32 each),
   *  times 2.6 M parameters per unit of rank. */
  perRankMb: 31.2,
  /** AdamW's SECOND momentum buffer, one more F32 per parameter. Only 0.67 GB
   *  at rank 64 — which is the whole reason the recipe could move off Muon. */
  adamwPerRankMb: 10.4,
  /** Linear part of the activation cost: per-layer checkpoints and the arena. */
  perTokenMb: 0.2679,
  /** Quadratic part: the [S, S, heads] attention scores the backward retains
   *  for the live checkpoint segment. This is the term that makes crop length
   *  expensive. */
  perTokenSqMb: 0.00044765,
  /** Typical MM3 prompt. Added to maxFrames to get the sequence length. */
  promptTokens: 1142,
  constMb: 441,
  /** Frozen KV prefix (engine train/lm-kvprefix.h). NOT fitted — this one is
   *  arithmetic: 36 layers x K and V x n_kv_heads*head_dim (1024) x 4 bytes is
   *  exactly 0.28125 MB per stored column, and the store spans the prefill
   *  stream plus the window (the window splices its own K/V onto the end).
   *  Predicted 848 MB at 750 prefix frames against 856 measured. */
  prefixMbPerColumn: 0.28125,
  /** Flash-attention training's own coefficient set (2026-09-05 flag-contract
   *  work) — a NAMED PLACE for the two-anchor fit the flash-attn-training
   *  skill's methodology calls for, once mm3-lm-train-run.h gains --attn flash
   *  and someone has actually measured it. null = not measured: `estimateMm3PeakMb`
   *  falls back to the exact-mode quadratic term rather than guessing a
   *  halving, and callers should show "flash: estimate pending measurement"
   *  instead of a number they cannot stand behind. Shape mirrors the fields
   *  flash mode would plausibly change; extend it once real anchors exist. */
  /** MEASURED 2026-09-11 (RTX 5090, mm3-lm-q8_0, rank 128, AdamW, acoustic
   *  loss on, no prefix, --attn flash), three anchors on one album:
   *
   *      S =  3050 (2000 frames)  ->  25425 MB
   *      S =  5050 (4000 frames)  ->  26768 MB
   *      S = 10425 (9000 frames)  ->  29125 MB
   *
   *  Linear in S (0.50 MB per token: the 36 f32 layer checkpoints plus the
   *  arena; no [S, S] scores are retained under flash), on a floor that is
   *  7.8 GB above what loaded + rank + constMb account for — the fused
   *  attention's own workspace and the checkpoint segments at their fixed
   *  size. The fit reproduces the outer anchors to 1 MB and the middle one to
   *  340 MB. Calibrated at rank 128 / q8_0; an estimate elsewhere. */
  flash: { perTokenSqMb: 0, perTokenMb: 0.5016, constMb: 7759 } as null | {
    perTokenSqMb: number; perTokenMb: number; constMb: number;
  },
} as const;

/** Whether MM3_VRAM_MODEL.flash carries a real measurement. False today. */
export function mm3FlashVramCalibrated(m: typeof MM3_VRAM_MODEL = MM3_VRAM_MODEL): boolean {
  return m.flash !== null;
}

/** Extra peak VRAM from a frozen KV prefix, in MB. 0 when it is off.
 *
 *  Linear in the prefix, which is the whole point: the quadratic term above is
 *  the backward retaining [S, S, heads] attention scores, and a prefix has no
 *  backward. 750 frames of history cost about what 60 frames of crop do. */
export function estimateMm3PrefixMb(prefixFrames: number, maxFrames: number,
                                    chunk = 256,
                                    promptTokens = MM3_VRAM_MODEL.promptTokens): number {
  if (!(prefixFrames > 0)) return 0;
  const qMax = promptTokens + prefixFrames;
  const sMax = promptTokens + maxFrames + 1;   // +1: the window's lead frame
  const w    = Math.max(sMax, chunk);
  const mask = (qMax + w) * w * 4 / 1048576;
  return Math.round(MM3_VRAM_MODEL.prefixMbPerColumn * (qMax + sMax + chunk) + mask);
}

/** Peak VRAM for a configuration, in MB.
 *
 *  SHIPPED TO THE UI AS COEFFICIENTS, not just as an answer: the form re-runs
 *  this as the user drags rank and crop length, and a second copy of these
 *  numbers over there would drift from the measurements that produced them. */
export function estimateMm3PeakMb(baseBytes: number, rank: number, maxFrames: number,
                                  extraMb = 0,
                                  optimizer: 'muon' | 'adamw' | 'prodigy' = MM3_LM_DEFAULTS.optimizer,
                                  prefixFrames = 0,
                                  attn: 'exact' | 'flash' = 'exact'): number {
  const M      = MM3_VRAM_MODEL;
  const loaded = baseBytes / 1048576 + M.loadedOverheadMb;
  const S      = M.promptTokens + Math.max(0, maxFrames);
  // Muon carries one momentum buffer, AdamW two, Prodigy four (m, v, s and x0 —
  // the initial weights, which the <g, x0-x> numerator needs). Each extra buffer
  // is one adamwPerRankMb, so Prodigy is 3x AdamW's surcharge. Measured at rank
  // 128: Muon 14.2 GB, AdamW 17.5, Prodigy 20.2 — a 2.7 GB Prodigy-over-AdamW
  // gap against 2.66 predicted.
  const extraBuffers = optimizer === 'adamw' ? 1 : optimizer === 'prodigy' ? 3 : 0;
  const perRank = M.perRankMb + extraBuffers * M.adamwPerRankMb;
  // Flash mode is the whole reason this term exists to be replaced — non-flash
  // attention backpropagates through retained [S, S, heads] scores, and flash
  // does not. But M.flash is null until a real two-anchor fit exists (see the
  // field's comment), so asking for 'flash' with no measurement falls back to
  // the exact-mode coefficient rather than reporting a saving nobody has
  // proven. mm3FlashVramCalibrated() tells a caller whether the number below
  // actually reflects flash mode or is standing in for it.
  if (attn === 'flash' && M.flash) {
    // Calibrated flash curve (see M.flash): linear in S on its own floor.
    return Math.round(loaded + perRank * rank + M.flash.perTokenMb * S + M.flash.perTokenSqMb * S * S
                      + M.constMb + M.flash.constMb + extraMb + estimateMm3PrefixMb(prefixFrames, maxFrames));
  }
  return Math.round(loaded + perRank * rank + M.perTokenMb * S + M.perTokenSqMb * S * S
                    + M.constMb + extraMb + estimateMm3PrefixMb(prefixFrames, maxFrames));
}

/** Every installed base, best quality first.
 *
 *  Scans rather than enumerates: the trainable set is whatever CUDA can
 *  dequantize, and the Model Manager can add to it without a code change. A
 *  base with no catalogue entry still appears — unknown quality beats
 *  invisible, since hiding a file the user deliberately downloaded is the more
 *  confusing failure. */
export function availableMm3Bases(rank: number = MM3_LM_DEFAULTS.rank,
                                  maxFrames: number = MM3_LM_DEFAULTS.maxFrames): Mm3BaseInfo[] {
  const dir = mm3ModelDir();
  let files: string[] = [];
  try {
    files = fs.readdirSync(dir).filter(f => f.startsWith('mm3-lm-') && f.endsWith('.gguf'));
  } catch {
    return [];
  }
  const QORDER = { reference: 0, excellent: 1, good: 2, fair: 3, poor: 4 } as const;
  // A base whose MEASURED loss delta is negligible is not meaningfully worse
  // than f16, whatever band its name suggests, so it collapses into the top
  // tier and the tie-break on size decides. This is what stops the recommender
  // offering f16 on a 32 GB card: q8_0 measures +0.02% — 1/40th of the gap to
  // the next base — for HALF the VRAM, so trading rank 256 down to 128 to buy
  // that 0.02% back is a straight loss. Threshold is 0.1%, comfortably above
  // the two negligible measurements and comfortably below Q6_K's 0.30%.
  const NEGLIGIBLE = 0.001;
  const tier = (b: { lossDelta: number | null; quality: keyof typeof QORDER }) =>
    (b.lossDelta !== null && b.lossDelta <= NEGLIGIBLE) ? 0 : QORDER[b.quality];
  return files
    .map(f => {
      const id = f.slice('mm3-lm-'.length, -'.gguf'.length);
      let bytes = 0;
      try { bytes = fs.statSync(path.join(dir, f)).size; } catch { /* raced */ }
      const facts = MM3_BASE_FACTS[id] ?? { lossDelta: null, quality: 'fair' as const };
      const extraMb = facts.extraMb ?? 0;
      return {
        id, file: f, bytes,
        lossDelta: facts.lossDelta,
        quality:   facts.quality,
        extraMb,
        peakMb:    estimateMm3PeakMb(bytes, rank, maxFrames, extraMb),
      };
    })
    .filter(b => b.bytes > 0)
    // Quality band first, then MEASURED loss, then size. The middle term is not
    // decoration: q8_0 and Q6_K are both 'excellent', but q8_0 measures +0.02%
    // against f16 and Q6_K +0.30% — 15x apart. Sorting the band by size instead
    // put the bigger-error base first on any card with room for both, which is
    // the opposite of what a fidelity-ordered list is for. An unmeasured base
    // sorts after every measured one in its band rather than being assumed
    // good.
    .sort((a, b) => {
      const ta = tier(a), tb = tier(b);
      if (ta !== tb) return ta - tb;
      // Inside the collapsed top tier the measured deltas ARE noise, so size is
      // the real decider. Everywhere else, measured fidelity leads and an
      // unmeasured base sorts behind every measured one in its band.
      if (ta === 0) return a.bytes - b.bytes;
      return ((a.lossDelta ?? Infinity) - (b.lossDelta ?? Infinity)) || (a.bytes - b.bytes);
    });
}

/** RANK LADDER for the recommender. The defaults were validated at 256 on a
 *  32 GB card; below that, rank is the term that has to give. 31.2 MB per unit
 *  of rank means 256 -> 64 frees 6.0 GB, which is more than any base swap. */
// Ranks the recommender may fall back to, high to low. It starts at the RECIPE
// rank and only ever goes down: rank is a recipe choice with published evidence
// behind it, not a "more is better" dial the card size should win. Before this
// was capped, a 32 GB card was offered r256 — overriding the default and
// quietly reinstating the configuration the recipe change exists to leave.
const MM3_RANK_LADDER_FULL = [256, 128, 64, 32, 16];

export interface Mm3Recommendation {
  base: string;
  rank: number;
  /** True when nothing in the catalogue fits this card at any ladder rank, so
   *  the values below are a best effort rather than a promise. */
  overBudget: boolean;
}

/** The best CONFIGURATION that fits the card, not just the best base.
 *
 *  A base picker alone is not enough for a small card: at the default rank 256
 *  nothing in the catalogue fits in 16 GB, so a recommender that only chose a
 *  base would hand a 16 GB user a red warning and no way out. Rank is the
 *  bigger lever anyway.
 *
 *  ORDER OF SACRIFICE: fidelity first, rank second. The search walks the bases
 *  in fidelity order and gives each one the highest ladder rank it can afford,
 *  taking the first that fits at all. Ranking it the other way round — highest
 *  rank first, best base that fits at that rank — was tried and produced
 *  visibly worse advice: on a 20 GB card it offered Q3_K_M ('fair', unmeasured)
 *  at rank 256 in preference to q8_0 (+0.02% against f16) at rank 128, and on
 *  16 GB it reached for Q4_K_S over Q6_K. A base error floors what the adapter
 *  can learn no matter how much rank sits on top of it, whereas a smaller rank
 *  simply fits less detail. `quality: 'poor'` is never recommended at any rank.
 *
 *  `headroomMb` is not padding for the estimate (good to <0.3%) but for the
 *  desktop session sharing the GPU, which is what pushed f16 from 3.7 to
 *  12-14 s/step in the A/B that set the default. */
/** The ladder, capped at the recipe's rank. Declared as a function because
 *  MM3_LM_DEFAULTS is defined further down the file. */
function rankLadder(): number[] {
  const capped = MM3_RANK_LADDER_FULL.filter(r => r <= MM3_LM_DEFAULTS.rank);
  return capped.length ? capped : [MM3_LM_DEFAULTS.rank];
}

export function recommendMm3Config(gpuTotalMb: number,
                                   maxFrames: number = MM3_LM_DEFAULTS.maxFrames,
                                   headroomMb?: number): Mm3Recommendation {
  const fallback: Mm3Recommendation = {
    base: MM3_LM_DEFAULTS.basePrecision, rank: MM3_LM_DEFAULTS.rank, overBudget: false,
  };
  if (!availableMm3Bases().length) {
    return fallback;
  }
  if (gpuTotalMb <= 0) {
    // Card unknown: do not guess downward. The defaults are what the recipe was
    // validated on, and a wrong small guess is worse than no guess.
    return fallback;
  }
  // Headroom scales with the card. A flat 1.5 GB is right on a 24 GB card and
  // absurd on a 12 GB one, where it is an eighth of the whole budget and pushes
  // configurations that genuinely run into "over budget".
  const headroom = headroomMb ?? Math.min(1536, Math.round(gpuTotalMb * 0.08));
  // availableMm3Bases() returns fidelity order, so the OUTER loop is the one
  // that must be over bases.
  for (const base of availableMm3Bases(MM3_LM_DEFAULTS.rank, maxFrames)) {
    if (base.quality === 'poor') {
      continue;
    }
    for (const rank of rankLadder()) {
      const peak = estimateMm3PeakMb(base.bytes, rank, maxFrames, base.extraMb);
      if (peak + headroom <= gpuTotalMb) {
        return { base: base.id, rank, overBudget: false };
      }
    }
  }
  // Nothing fits at any rank. Offer the SMALLEST usable base at the LOWEST rank
  // — the configuration with the best chance — and say so, rather than silently
  // landing on something that cannot run.
  const ladder = rankLadder();
  const lowest = ladder[ladder.length - 1];
  const usable = availableMm3Bases(lowest, maxFrames).filter(b => b.quality !== 'poor');
  const smallest = usable.reduce<Mm3BaseInfo | null>(
    (best, b) => (!best || b.peakMb < best.peakMb ? b : best), null);
  return { base: smallest ? smallest.id : MM3_LM_DEFAULTS.basePrecision,
           rank: lowest, overBudget: true };
}

// ── Layout ──────────────────────────────────────────────────────────────────

/** Codes live beside the dataset, not in a shared pool.
 *
 *  The design doc floated a shared `<training>/mm3-codes/<encoder>/<dataset>/`
 *  so one corpus could be re-encoded by several encoders for a shoot-out. That
 *  is a research workflow; the product one is "this dataset's codes", and
 *  keeping them under the dataset means deleting a dataset takes its derived
 *  data with it. Re-encoding with a different encoder overwrites, and
 *  `codes.json` records which encoder produced what. */
export function mm3CodesDir(slug: string, laundered = false): string {
  // Two caches, never one dir with two producers: flipping the toggle must
  // not silently reuse the other kind's codes.
  return path.join(datasetDir(slug), laundered ? 'mm3-codes-laundered' : 'mm3-codes');
}

/** Adapters go straight where the picker looks. Writing anywhere else would
 *  add an install step for no reason — the shipped lister scans two directory
 *  levels under this root and reads `<file>.json` as the sidecar, which is
 *  exactly the layout `ace-train mm3-lm-train` writes. */
export function mm3AdapterRunDir(runName: string): string {
  return path.join(config.aceServer.adapters, 'mm3-lm-adapters', runName);
}

/** `<dataset>-YYYY-MM-DD_HH-MM-SS`, the logs/ convention: name-sorted is
 *  time-sorted, and retraining never overwrites an earlier run. */
export function mm3RunName(slug: string): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `${slug}-${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_`
       + `${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

// ── Defaults: the validated recipe, in one place ────────────────────────────
//
// Every number here is from docs/plans/2026-08-20-mm3-training-studio.md's
// recipe table or from a measurement recorded there. The Jobs form is a VIEW
// of this object; it must never carry a second copy of these numbers.
export const MM3_LM_DEFAULTS = {
  // ── bghira's published SimpleTuner recipe (adopted 2026-08-23) ───────────
  //
  // Sources, both with full configs: the album C tournament
  // (RareConcepts/<his public MM3 dataset>, simpletuner_config.json in every
  // checkpoint) and terminusresearch/minimax-music3-lm-lora-fiona-crapple.
  //
  // What we ran before: r256/alpha256, lr 8e-5, Muon @ lr_scale 64, random
  // 1500-frame crops, 800 steps, no warmup. That produced adapters with a trace
  // of the artist over a backing track that had gone simpler and cheaper —
  // weak identity AND a damaged base. The arithmetic says why: he runs 42
  // epochs over 45 tracks, we ran 61 over 13, at four times the rank.
  //
  // The full reasoning, and the two of his settings we deliberately do NOT
  // copy, are on MM3LmTrainArgs in engine/src/train/mm3-lm-train-run.h.
  rank: 128,
  alpha: 128,
  lr: 8e-5,     // the default (Balanced) recipe; Fast's 1.6e-4 is experimental — see MM3_LM_PRESETS
  /** 1200, down from 2500, because a step is no longer the same size. At the
   *  128-frame crop a step supervised 5 seconds; at 4272 it supervises 171, so
   *  2500 steps went from 320k supervised frames to 10.7M — 312 epochs over 8
   *  songs, each now covering 85% of every track instead of 2.5%.
   *
   *  It does NOT drop as far as the 33x that arithmetic suggests, and the reason
   *  is worth keeping: AdamW travels roughly `lr` in parameter space per step
   *  regardless of how much data backs the gradient, so the adapter still needs
   *  updates to move. What DID change is that the gradients are far less noisy,
   *  so there is less noise regularising the fit — expect the ear-optimum
   *  earlier in step count and harder overfitting past it.
   *
   *  1200 steps x ~15.5 s is ~5.2 hours. NOT tuned by ear yet; it is reasoning
   *  from the step-size change, not a measured optimum.
   *
   *  250 as of 2026-08-26 (Rob), alongside a 2250-frame history prefix. A step
   *  that carries 90 s of real context is a bigger step in learning terms and a
   *  slower one in wall-clock, so the same hour buys fewer, richer steps. The
   *  ear-optimum under this regime is UNKNOWN - the 750-2000 range was measured
   *  on adapters trained with no history at all. Ladder the checkpoints. */
  /** 1000 — and under the default stopping strategy this is the CAP, not the
   *  plan: the run ends here only if the target loss never arrives. Raised from
   *  250 on 2026-08-26 to give a target-loss run room to reach one; a 250-step
   *  cap would have ended almost every run before the target could bind, which
   *  is the same as not having a target. */
  steps: 500,   // the default (Balanced) recipe; Fast's 300 is experimental — see MM3_LM_PRESETS
  /** ── Stopping strategy ──────────────────────────────────────────────────
   *
   *  'steps', 500, since 2026-09-06 (Rob): the default method is HOT-PiZZA
   *  (below), whose training loss reads high by construction — a random
   *  share of the base's principal subspace is deleted every micro-step, so
   *  the trailing mean never reaches a LoRA-style target and a loss stop
   *  would run to the cap anyway. The adapter Rob rated best twice was that
   *  method's step-500 checkpoint; the 250/350 ladder is under test. Switch
   *  the method to plain LoRA and 'loss' with targetLoss below is the
   *  measured rule again.
   *
   *  'steps' is the default and the one every measured recipe was run under.
   *  'loss' trains until the loss reaches `targetLoss` instead, with `steps`
   *  demoted to a CAP so a target that never arrives still ends the run.
   *
   *  There is no validated target number for MM3 — loss here is CE over RVQ
   *  codes and its absolute value depends on the crop length, the prefix, the
   *  acoustic-loss weight and the dataset, so a target that suits one album is
   *  meaningless on another. The default 0 is honest about that: pick the
   *  target by reading the curve of a run you have already done.
   *
   *  A NOTE THE UI REPEATS: at 1.0 held-out these adapters were already good,
   *  and a training loss under ~0.05 was pure sequence memorisation on the
   *  runs that got there. Down is not automatically better. */
  stopMode: 'steps' as 'steps' | 'loss',
  /** 0.1 on the trailing training mean — Rob, 2026-08-26.
   *
   *  Measured trajectories at the shipped recipe (trailing-25 mean):
   *
   *      step        25     100     250     500
   *      album D   3.14   2.50    1.43    0.59
   *      albumE  3.28   2.23    0.92    0.31
   *      limbizkit   3.62   2.72    2.03      -
   *
   *  So 0.1 binds on the albums that converge fast, somewhere past 500, and
   *  never arrives on the slow ones — which is what the 1000-step cap is for.
   *  It sits just above the ~0.05 mark where runs had demonstrably memorised
   *  their songs, and for an ALBUM CLONE that is the intended end of the range.
   *
   *  1.0 since 2026-09-05 (Rob), from a blind depth ladder on albumA:
   *  the step-250 checkpoint (5-epoch mean ~1.06) beat the step-500 one
   *  (0.25 at the save, 0.08 at the best step) on every one of three songs,
   *  and the step-500 checkpoint emitted an empty plan (EOS at once) on one
   *  of them — the same likeness-up-then-coherence-collapse curve the AS1.5
   *  planner shows below ~1. 0.1 trained straight past the crossover. The
   *  every-50 checkpoints still give the ear ladder; this just stops the run
   *  where the ladder has been landing. */
  targetLoss: 1.0,
  /** 'train' is available on every run. 'eval' is the number that means the
   *  adapter GENERALISES rather than memorised, but it only lands on eval
   *  steps, so with the default evalEvery of 250 it fires at most once in a
   *  250-step run — lower evalEvery before targeting it. */
  targetLossMetric: 'train' as 'train' | 'eval',
  /** Whole PASSES over the album, averaged — the DiT trainer's ma5 by another
   *  name, and 5 for the same reason.
   *
   *  One step's loss is one crop of one song and swings by more than a whole
   *  run's improvement, so a target has to be checked against an average. It
   *  has to be an average over COMPLETE passes: a 25-step window on an 11-song
   *  album covers two passes plus three songs, so three tracks weigh triple and
   *  eight weigh double, and the window then rises and falls with which songs
   *  it caught rather than with the model. A whole number of passes counts
   *  every song identically. */
  targetLossEpochs: 5,
  /** 100 (Rob, 2026-09-09; was 50). Five checkpoints on a 500-step run. The
   *  depth ladder showed 500/750/1000 inside the noise, so the mid-run grid
   *  was costing 2.8 GB a checkpoint for auditions nobody did; the final is
   *  what ships. */
  saveEvery: 100,
  /** One preview per checkpoint (Rob, 2026-08-25). The clock cadence made
   *  sense at 15 s steps and a 250-step save interval; at saveEvery 100 the
   *  checkpoint grid IS the cadence you want the ear to track. */
  previewEverySteps: 100,
  /** Keep resume-state.bin (~4.2 GB of optimizer state) after a run that
   *  reaches its end. Off (Rob, 2026-09-09): a finished adapter is not
   *  continued in practice, and the state outweighed the adapter itself. A
   *  run that stops short (cancel, crash, pause) keeps it regardless. */
  keepResumeState: false,
  /** Cadence follows checkpoints instead of the clock (Rob, 2026-08-25): a
   *  preview per checkpoint means every audition candidate on disk has an ear
   *  sample attached, and none are rendered twice. */
  previewEveryMinutes: 0,
  /** His lr_warmup_steps. Note step 1 still runs at lr 0 — the shared schedule
   *  is 0-based, which costs one step out of a thousand. */
  /** 25, scaled to the shorter run: 50 would be 10% of a 500-step run spent at
   *  near-zero lr. bghira's 50 belongs to his 1000+-step schedules. */
  warmup: 25,
  gradAccum: 1,
  seed: 42,
  /** 4272 frames = 171 s, which is 84% of a 204 s track — the most the card
   *  holds at q8_0 / rank 128 / AdamW inside a 30 GB ceiling (est. 29.1 GB,
   *  against a 27.16 GB MEASURED anchor at rank 64 / 4096 frames).
   *
   *  It was 128 — 5.1 SECONDS — because f16 + rank 128 + Prodigy costs 26.9 GB
   *  before a single frame of crop, and the S term is quadratic. At that window
   *  the adapter could not learn any structure longer than five seconds, and the
   *  two positions that decide whether a render sounds like a song were left to
   *  chance: EOS supervised on 2.6% of steps, frame 0 on 0.02%. Renders began
   *  mid-flow and never resolved.
   *
   *  Whole-track training is NOT reachable here: 9 of 10 tracks in the reference
   *  dataset exceed 4096 frames, and the median (5099) would need ~17 GB of
   *  attention scores alone. cropMode `structured` is what covers the gap. */
  /** 750 frames = 30 s, set by ear (Rob, 2026-08-25) over the 4272-frame
   *  window this default previously held. The long window bought track
   *  coverage; the listening said shorter crops train better here, and the
   *  song-structure duties the long window carried are now covered elsewhere
   *  — structured start/end crops teach openings and EOS, and the acoustic
   *  loss holds timbre. ~5.4 GB of activations at q8/r128 instead of ~19. */
  /** 500 since 2026-09-07 (Rob): the speed trial's crop-500 arm tied crop 750
   *  blind (67.5 vs 68 of 90) and the combined safe stack (crop 500 + prefix
   *  1024 + prefill chunk 1024) tied the crop-750 recipe again (67 vs 70.5,
   *  noise ~6) at 3.1 s/step against 4.5: 26 min per 500 steps instead of 37. */
  /** 9000 = WHOLE SONG since 2026-09-11 (Rob). The window-recipe adapters
   *  above (crop 750, history 2048) lost the song's arc — one sung passage,
   *  minutes of looped instrumental, no ending — because nothing in training
   *  ever scored more than 30 s at once. At 9000 frames every track under
   *  six minutes trains as one sequence; the engine clamps its buffers to the
   *  album's longest track, so a short album costs what it needs and not
   *  what was asked. Tracks longer than the window are dropped (longTracks). */
  maxFrames: 9000,
  /** Tracks longer than maxFrames: `exclude` leaves them out of the run (the
   *  engine's --drop-over-frames, named in the log); `crop` trains them in
   *  structured crops of maxFrames, which is the pre-2026-09-11 behaviour and
   *  puts their tail at positions past the base's 10240-token range. About
   *  6.5% of the catalogue's tracks exceed 360 s; the route refuses a run that
   *  exclusion would empty by more than half.
   *
   *  `excise` instead cuts one repeated section out of each over-length track
   *  so the intro and the real ending both survive, and trains the result
   *  whole (ace-train mm3-retarget). It still passes --drop-over-frames,
   *  because a track the excision refuses -- no clean repeat to skip, or the
   *  removal would have to cross a vocal -- must stay excluded rather than be
   *  cropped behind the user's back. */
  longTracks: 'exclude' as 'exclude' | 'crop' | 'excise',
  /** `structured`: a fixed share of steps pinned to frame 0, a fixed share
   *  flush to the track's end, the rest random.
   *
   *  Neither of the other two modes can work here, and for opposite reasons.
   *  `random` leaves the opening and the ending to chance — 0.02% and 2.6% of
   *  steps on a 204 s track. `beginning` always starts at 0, so it teaches the
   *  opening perfectly and an ending NEVER, because EOS is only supervised when
   *  a crop reaches n_frames and no track here fits in the window. That is the
   *  intros-only trainer the lm2 run became.
   *
   *  `structured` costs nothing over `random` — same crop length, same memory —
   *  and it is what the EVAL plan in mm3-lm-train-run.h has always done ("evenly
   *  spaced starts, with the last crop flush to the end so the set always
   *  includes a real ending"). Training simply never got the same treatment. */
  cropMode: 'structured' as 'random' | 'beginning' | 'structured',
  /** 40% anchored at frame 0, 15% flush to the track's end, 45% random.
   *
   *  The random share is NOT optional, and 85/15/0 proved it the hard way:
   *  frame-0 and flush-to-end are SINGLE positions per song, so a split with no
   *  random remainder collapses an 8-song dataset to 16 distinct samples, and
   *  the first such run recited (train loss 0.0003 by step 800, audible quality
   *  loss from ~ck550). Random crops are what put ~20k distinct samples back.
   *
   *  The anchored shares keep what the structured mode exists for: frame-0
   *  crops give supervised positions the song's real history (the condition at
   *  inference), and end crops are the only place EOS is supervised. */
  /** 20/15 (random = the remainder, 65%) — Rob, 2026-08-26.
   *
   *  This REVERSES the 2026-08-25 raise to 55, and the reasoning for that raise
   *  is kept below rather than deleted, because it was a real observation and
   *  this is a re-weighting of it rather than a refutation. That day's
   *  argument: at crop 750 the random+end shares made ~60% of supervised audio
   *  mid-flow material with no arc, the adapter's content prior followed it,
   *  and renders jumped in "like a cut".
   *
   *  What 0.55 costs is DISTINCTNESS. Frame-0 and flush-to-end are single
   *  positions per song, so more than half of every run was repeated exposure
   *  to the same few seconds of each track — the regime that recited on an
   *  8-song dataset. 0.2 puts the random share back where the sample count
   *  lives, while still supervising openings and endings far above what
   *  `random` alone reaches (0.02% and 2.6%). */
  cropStartFrac: 0.2,
  cropEndFrac: 0.15,
  /** 1 = the start share is frame 0 and nothing else — Rob, 2026-08-26.
   *
   *  Tiling spends part of the start share on aligned tiles at K, 2K … so short
   *  crops teach the intro→build→verse ARC in order. That was worth having when
   *  the start share was 0.55 and half of it went to one repeated position. At
   *  a 0.2 start share there is little left to spread, and the arc is bought
   *  more cheaply from the 65% random share, which covers those same positions
   *  with far more distinct crops. Raise this again if the start share does. */
  cropStartTiles: 1,
  /** The acoustic loss: teacher-forced CE through the FROZEN depth decoder,
   *  gradient into the adapter via last_hidden — the 2026-08-25 fix for
   *  adapters shifting vocal timbre ("chipmunk"/"goblin" renders). The depth
   *  decoder generates every acoustic codebook from the LM's hidden state at
   *  render, and the old semantic-only objective left that state completely
   *  unconstrained; the ear-validated "MLP 0.5" render dial was this fault
   *  being managed by hand. 0 restores the old objective, for A/B only.
   *
   *  Costs the full depth decoder resident during training (+1.2 GB f16) plus
   *  a small per-step pass over depthLossFrames sampled frames. */
  depthLossWeight: 1.0,
  depthLossFrames: 128,
  /** AdamW, matching `adamw_bf16`. Muon was only ever chosen because it FIT at
   *  rank 256 (AdamW's second momentum buffer costs +2.66 GB there, +0.67 GB at
   *  rank 64) and it made `lr` meaningless — its update is normalised, so it
   *  needed a --muon-lr-scale that was never tuned past "best of four values". */
  /** AdamW, because Prodigy's three extra parameter-sized buffers cost ~2.6 GB
   *  and that is 800 frames of crop — and crop length is the axis that decides
   *  whether the adapter learns song structure at all, while Prodigy has never
   *  once been validated by ear. Prodigy remains selectable and still resumes.
   *
   *  What Prodigy buys, when the budget is not the binding constraint: it sets
   *  its own step size, so `lr` becomes a schedule multiplier only and the
   *  trainer forces it to 1.0. On album B it converged to an effective 8.19e-5
   *  against the 8e-5 tuned by hand — within 2.4%, from d0 = 1e-6 and no
   *  guidance.
   *
   *  It resumes as of resume format v2, so mid-training previews work: s, d and
   *  r ride in the pause state and x0 lives once in the run directory (it never
   *  changes after init, so copying it into every pause would be a parameter-
   *  sized buffer per preview for nothing).
   *
   *  NOT YET VALIDATED BY EAR — its only comparison render was made on the f16
   *  base and is void. */
  /** Prodigy again (Rob, 2026-08-25). It was demoted to AdamW only to buy
   *  crop VRAM at 4272 frames; at crop 750 there is >12 GB of headroom and its
   *  ~2.7 GB of extra buffers stop mattering. It sets its own step size (on
   *  album B it converged within 2.4% of the hand-tuned 8e-5), resumes as of
   *  format v2 so checkpoint-cadence previews work, and `lr` becomes a
   *  schedule multiplier the trainer forces to 1.0. */
  /** adamw since 2026-09-06 (Rob): in the blind HOT-PiZZA recipe test on
   *  albumA, AdamW at lr 8e-5 tied Prodigy by ear (69.5 vs 72 of 90,
   *  inside the noise floor) and saved 2.3 GB of optimizer state. The stacked
   *  recipe (prefix 2048 + flash + AdamW + f16 factors) was rated "fantastic"
   *  sighted and is the shipped default. Prodigy stays selectable. */
  optimizer: 'adamw' as 'muon' | 'adamw' | 'prodigy',
  muonLrScale: 64,
  /** q8_0, not f16 — see the note on Mm3TrainModels. Same step time since the
   *  cpy-q-occupancy patch, ~8.5 GB less resident, and therefore the only one
   *  of the two that survives a GPU shared with a desktop session. Pick f16
   *  only to reproduce a pre-patch run exactly. */
  /** q8_0, and this REVERSES an ear result — deliberately, with the reasoning
   *  on the record so it can be re-tested.
   *
   *  The result: a 750-step f16-trained adapter beat a 2000-step q8_0-trained
   *  one, rendered on q8_0 both times. The reason it does not decide this: it
   *  was collected at maxFrames 128, where both candidates had been trained on
   *  5-second fragments and neither had learned how a song starts or ends. It
   *  compared two structurally broken adapters, so it says nothing about which
   *  base is better once the crop is fixed.
   *
   *  What f16 costs HERE is 8.0 GB, and 8.0 GB is 2600 frames of crop — the
   *  difference between covering a third of a track and covering 84% of one.
   *  Re-run the comparison at this crop before spending it again.
   *
   *  RENDER on q8_0 regardless: adapters are garbled on an f16 base. */
  basePrecision: 'q8_0' as Mm3BasePrecision,
  holdout: 0.15,
  evalEvery: 250,
  /** MUST be <= maxFrames. The engine default is 400 and is pinned independently
   *  of the crop, so at a 128-frame crop EVERY eval crop exceeds the sequence
   *  limit, all of them are skipped, and run_eval returns "no result" silently —
   *  the run reports an eval plan at startup and then never evaluates.
   *
   *  Held at 1024 rather than tracking maxFrames: eval is forward-only and reuses
   *  the training arena, so a long eval crop is affordable, but held-out CE is
   *  only comparable ACROSS runs while the crop it is measured at stays put. */
  /** = maxFrames (the route clamps it there anyway; stating it avoids the
   *  silently-skipped-eval trap this comment block documents). */
  evalCrop: 750,
  /** LyCORIS-style rank masking, part of bghira's published config. */
  rankDropout: 0.1,
  /** LoKr: dW = kron(w1, w2) instead of a low-rank pair.
   *
   *  FACTOR IS THE SIZE KNOB and 6 is chosen for a ~528 MB file, matching the
   *  AS1.5 DiT LoKr budget. The engine's own default of 16 gives 27.2M params
   *  (109 MB) which is FEWER than rank 64 — and rank 64 was the setting that
   *  turned lyrics to gibberish. Factor 6 / dim 512 gives 264M, between rank 64
   *  and rank 128.
   *
   *  LoKr was the default until 2026-09-06. In the blind method test on
   *  albumA (nine arms, three songs, everything locked but the method)
   *  it came last of the arms that rendered all three songs, with one drone
   *  failure, at 1.5x LoRA's step time. Plain LoRA is next in line after
   *  HOT-PiZZA below; LoKr stays selectable. */
  adapterType: 'lora' as 'lora' | 'lokr',
  lokrFactor: 6,
  lokrDim: 512,
  lokrAlpha: 512,
  /** Each crop is presented at its TRUE position in the track.
   *
   *  This is a RECIPE CHANGE as of 2026-08-23 and runs before it used `zero`,
   *  where every crop was labelled as if it were the song's opening. That was a
   *  straight train/inference mismatch — generation always begins at frame 0,
   *  so the positions a mid-song crop occupied during training are the ones
   *  that mean "the first two seconds" at render time. bghira's album C campaign
   *  independently reports the two symptoms this predicts (sound arriving
   *  instantly at 0:00, tempo drifting mid-track) and that position-labelled
   *  windowed crops fix the pacing. `zero` is kept only to reproduce an older
   *  run; the two are not comparable. */
  cropAnchor: 'song' as 'song' | 'zero',
  /** Frames of REAL, no-grad history placed in front of every crop
   *  (engine train/lm-kvprefix.h). OFF at 0.
   *
   *  DEFAULT 4096 (164 s) as of 2026-08-26. Chosen off the measured corpus (202
   *  tracks, 14 datasets, median 203 s): it is the point where the coverage
   *  curve flattens. The share of supervised steps that see as much history as
   *  they will at render runs 48.9% at 750, 73.5% at 2250, 87.7% at 4096, then
   *  94.4% at 5000 for 13% more compute and 98.0% at 6000 for 6% more again --
   *  quadratic spend chasing the tail of a length distribution whose top end is
   *  one 438 s track in 202.
   *
   *  This is a CEILING, not a fixed cost: a crop near the song's start has
   *  little history to load. Mean prefix actually used at 4096 is ~1450 frames.
   *
   *  NOTHING TRAINED WITH IT HAS BEEN HEARD YET - the number optimises
   *  train/render context match, which is the mechanism believed to be broken.
   *  It is not a measured coherence optimum.
   *
   *  What it is for: without it a crop is presented at its true position in
   *  the track with an EMPTY context, so the middle third of the planner — the
   *  layers doing long-range aggregation — is trained to produce late-song
   *  behaviour from a 30-second view. That is the band that renders better
   *  with the Middle Third dial at 0.
   *
   *  A prefix has no backward, so it escapes the quadratic VRAM term that
   *  makes crop length expensive: 0.28125 MB per stored column, measured at
   *  +856 MB and about +60% step time for 750 frames. Matching prefixFrames to
   *  maxFrames doubles the history the model sees for a fraction of what
   *  doubling the crop would cost. */
  /** 2048 since 2026-09-06 (Rob): tied 4096 blind (71 vs 72 of 90) and takes
   *  14% off the step, 0.8 GB off the peak. The per-step cost of a prefix is
   *  the prefill through every layer, not the attention over it. */
  /** 1024 since 2026-09-07 (Rob): tied 2048 blind (69 vs 68 of 90) and takes
   *  another 12% off the step; part of the safe stack with crop 500. */
  /** 0 since 2026-09-11: under the whole-song window the track IS its own
   *  history, and a frozen prefix would only add prefill cost. */
  prefixFrames: 0,
  /** Prefill positions per graph. Trades host graph-build overhead against the
   *  transient attention scores of one chunk; 256 is a middle setting and has
   *  no effect on the result, only on speed and peak. 1024 since 2026-09-07:
   *  3.72 vs 4.46 s/step at 256 with the same step-1 loss and peak, then
   *  heard inside the safe stack. */
  prefixChunk: 256,
  /** Prove the prefix before training on it. Attention over [prefix ; window]
   *  is mathematically identical to one long crop covering both, so the
   *  supervised CE must not care which way it was produced. It caught two real
   *  bugs during bring-up; it costs one forward pass and it refuses to train if
   *  it fails. Leave it on. */
  prefixSelftest: true,
  /** ON, and it did not used to exist. A trigger that is only in the sidecar is
   *  not a trigger — it is an unseen token sequence you then paste in front of
   *  your prompts at render time, which is worse than not having one. */
  triggerPrepend: true,
  /** SimpleTuner's `lr_end: 4e-7` over a 5e-5 base. Our shared cosine bottomed
   *  at 0.1 of base, running the tail of the schedule twelve times hotter. */
  lrEndFrac: 0.005,
  /** Prior preservation, OFF until a regularisation dataset is chosen — it
   *  needs a corpus the user has to supply. `regEvery: 3` is bghira's 1:2 ratio
   *  and is what the UI offers the moment one is picked. */
  regEvery: 3,
  regTopK: 64,

  // ── Flag-contract parity with train-dit / train-lm (2026-09-05) ──────────
  //
  // ENGINE STATUS, updated as the concurrent engine port landed DURING this
  // change: the mm3-lm-train parser (engine/tools/ace-train.cpp) already
  // accepted --artist-token[-k/-lr] and --lora-plus-ratio before this work
  // started (they existed, they were just never wired past the arg builder);
  // --attn exact|flash|flash-f32 landed on mm3-lm-train mid-session (R3 of the
  // flash-attn roadmap) and is real — see the note on attnBackend below.
  // --rslora, --dora, --hira and --loha LANDED on mm3-lm-train in the same
  // session and are real: the four are FD-gated on the MM3 trainer, refuse the
  // same combinations train-dit refuses, and reach the runtime and merge
  // loaders (rsLoRA as use_rslora, DoRA as lora_magnitude_vector, HiRA/LoHa as
  // peft_type + merge mode only).
  // --pissa and --hra landed on mm3-lm-train (and on train-lm) the same day and
  // are also real. Both are FD-gated on the MM3 trainer and both export as
  // ORDINARY PEFT LoRAs — PiSSA at rank 2r, HRA at rank r — so unlike HiRA and
  // LoHa they need nothing from the runtime or merge loaders and render as they
  // trained. --pissa is refused together with --resume, which is why the arg
  // builder below drops it when resumeFrom is set rather than letting the run
  // exit 2. A trainable --prefix-n landed on that parser on 2026-09-05 and is
  // FD-gated (train/lm-prefix.h; still a different thing from --prefix-frames,
  // which is the FROZEN history prefix). It is checkpointed-path only, and the
  // engine refuses it together with --prefix-frames, --attn flash, or
  // --reg-every — this builder emits prefixFrames unconditionally, so a
  // prefixN run needs prefixFrames set to 0 or the engine exits 2 naming the
  // pair. Emitting a flag here ahead of the engine is deliberate where it
  // happens: an ace-train that predates a flag exits 2 loudly on the unknown
  // option, which is a far better failure than a UI control that silently does
  // nothing.
  /** 'exact' is the byte-identical graph; 'flash' routes through the fused
   *  FLASH_ATTN_TRAIN/_BACK ops mm3-lm-train-run.h gained mid-session
   *  (2026-09-05) — real and engine-verified, NOT the ahead-of-the-engine
   *  case the rest of this block is. The engine REFUSES `--attn flash`
   *  together with a nonzero `--prefix-frames` (the frozen history prefix
   *  below, default 4096): a KV prefix makes the attention mask rectangular
   *  (S_kv = n_pfx + S), which the fused ops do not accept, and the run exits
   *  fatally rather than silently falling back. The mm3-train-lm and
   *  mm3-resume-lm routes coerce this pair to 'exact' and log it — see
   *  attnBackendResolved in routes/training.ts — so a bare "turn on flash"
   *  click is never refused for a setting the user never touched. Only the
   *  non-default value is emitted, so a build that predates --attn on this
   *  subcommand stays compatible for every existing caller. */
  /** flash since 2026-09-06 (Rob): the engine composes it with the KV prefix
   *  (7070238e; probe at S_kv = n_pfx + S, F16 prefill mask), gates passed
   *  (self-test, FD, step-1 loss within tf32 rounding), tied exact blind
   *  (68 vs 72 of 90) and takes 14% off the step, 1 GB off the peak. */
  attnBackend: 'flash' as 'exact' | 'flash',
  /** LoRA-family parameterizations, mutually exclusive with each other and
   *  with LoKr (adapterType). Same semantics as train-dit's dora/hira/loha/
   *  pissa/hra — see DitMethod in TrainDitForm.tsx. */
  rslora: false,
  dora: false,
  hira: false,
  loha: false,
  /** HOT-PiZZA — the default MM3 method since 2026-09-06 (Rob).
   *
   *  PiSSA (A/B start on each weight's top-r singular directions, the
   *  residual frozen as -B0A0) with the rank-dropout mask applied to the
   *  principal component itself rather than to the delta: every micro-step a
   *  random `rankDropout` share of the base's own top-128 subspace is deleted
   *  and the rest scaled 1/keep while the album is fitted. Found as a masking
   *  bug on 2026-09-05; kept on purpose because its adapter beat every
   *  correctly-masked method by ear on albumA, twice (66.5-68/90 vs
   *  LoRA 60-64, DoRA 63, LoKr 41.5; corrected PiSSA 39.5 with drone plans).
   *  One album so far. The export is an ordinary rank-2r LoRA; loaders need
   *  nothing. Train loss reads high under it — stop on steps. `hotPizza`
   *  implies `pissa`; the engine flag is --hot-pizza. */
  pissa: true,
  hotPizza: true,
  /** SVD init cache (engine --pissa-cache-dir, lm-pissa.h): the PiSSA init is
   *  a pure function of the base file, rank, oversample, iters and layer
   *  range, so the factors are stored once per base under
   *  <adapters>/mm3-lm-adapters/_pissa-init-cache/ and every later run at the
   *  same rank uploads the identical bytes instead of recomputing 252 SVDs.
   *  Bit-exact by construction; off only for a deliberate recompute. */
  pissaCache: true,
  /** Hold the frozen A0/B0 pair in F16 (engine --pissa-frozen-f16): halves the
   *  ~1.3 GB it costs at r128. The init then cancels to f16 precision rather
   *  than exactly. Heard as part of the stacked recipe (2026-09-06). */
  pissaFrozenF16: true,   // on since 2026-09-06: part of the stacked recipe Rob rated; 0.6 GB back, same speed
  hra: false,
  /** LoRA+'s B-side learning-rate multiplier. 1 = off (paper default 16). */
  loraPlusRatio: 1,
  /** Soft prompt (train/artist-token-io.h via the shared LM trainer core).
   *  '' = no token. Name defaults to the run's adapter name in the UI, not
   *  here — this module has no adapter name to default to. */
  artistToken: '',
  artistTokenK: 32,
  artistTokenLr: 0.005,
  /** Trainable per-layer K/V prefix (train/lm-prefix.h) — NOT the same thing
   *  as prefixFrames/prefixChunk above, which is a FROZEN prefix baked from
   *  real audio history (train/mm3-lm-kvprefix.h). 0 = off. */
  prefixN: 0,
} as const;

/** The three MM3 training presets. Each is a set of overrides on
 *  MM3_LM_DEFAULTS; everything not listed is shared.
 *
 *   balanced  THE DEFAULT: crop 750, a 2048-frame history prefilled in
 *             256-token chunks, 500 steps at 8e-5, per-track captions, no
 *             prior. ~36 min per album. On 2026-09-09 this exact recipe
 *             (GOODCAPS) gave album B 4/6 natural endings with likeness,
 *             intelligibility and style Rob rated perfect; the same recipe
 *             under the old dataset-wide caption ended 0/6.
 *   fast      Balanced's geometry (crop 750, history 2048) over 300 steps,
 *             with the history prefilled in 1024-token chunks: ~22 min. The
 *             2026-09-07 Fast (crop 500, history 1024, 2x LR) broke vocals on
 *             albumB; each of those three ingredients is gone here
 *             and the chunk size was cleared as a lever in that bisect. The
 *             300-step depth is the one thing not re-heard since.
 *   thorough  crop 750 and a 4096-frame history over 1000 steps, prefilled
 *             in the verified 256-token chunks: ~85 min. The window and
 *             history behind the highest scores recorded here (72 and 70.5
 *             of 90); the gain over Balanced sat inside the listening noise.
 *
 *  MM3_LM_DEFAULTS carries the Balanced values, so an empty request and the
 *  form's initial state are the same recipe. The route applies a named
 *  preset UNDER the request's own fields (applyMm3Preset). */
export type Mm3PresetName = 'fast' | 'balanced' | 'thorough';
export const MM3_LM_DEFAULT_PRESET: Mm3PresetName = 'balanced';
export const MM3_LM_PRESETS: Record<Mm3PresetName, {
  steps: number; lr: number; maxFrames: number; prefixFrames: number; prefixChunk: number;
}> = {
  // WHOLE-SONG since 2026-09-11 (Rob): every track trains as ONE sequence
  // (maxFrames = the engine's 9000-frame ceiling, no history prefix — the
  // song is its own history), and the three recipes differ only in depth.
  // The 2026-09-10 overnight: crop-750 adapters ended 12 of 72 renders across
  // six albums while the base ended 36 of 36 on the same prompts; whole-song
  // adapters ended 7/12 (album S), 7/12 (album B) and 6/12 (album P at the
  // 360 s ceiling) and brought album P's vocal share from 0.16 to 0.42 (album
  // 0.68). Cost on an RTX 5090: 6-12 s/step, 29 GB peak at rank 128.
  fast:     { steps: 300, lr: 8e-5, maxFrames: 9000, prefixFrames: 0, prefixChunk: 256 },
  balanced: { steps: 600, lr: 8e-5, maxFrames: 9000, prefixFrames: 0, prefixChunk: 256 },
  thorough: { steps: 900, lr: 8e-5, maxFrames: 9000, prefixFrames: 0, prefixChunk: 256 },
};
export function isMm3PresetName(v: unknown): v is Mm3PresetName {
  return v === 'fast' || v === 'balanced' || v === 'thorough';
}
/** Defaults with a named preset laid over them; an unknown or absent name
 *  returns the defaults untouched (which are the Balanced preset). */
type Mm3PresetFields = (typeof MM3_LM_PRESETS)[Mm3PresetName];
/** The defaults with the preset-governed fields widened to plain numbers. */
export type Mm3EffectiveDefaults = Omit<typeof MM3_LM_DEFAULTS, keyof Mm3PresetFields> & Mm3PresetFields;
export function applyMm3Preset(defaults: typeof MM3_LM_DEFAULTS, preset: unknown): Mm3EffectiveDefaults {
  return isMm3PresetName(preset) ? { ...defaults, ...MM3_LM_PRESETS[preset] } : defaults;
}

/** Where a regularisation corpus's captured base distributions live.
 *
 *  Beside the dataset's codes rather than under the training run, because the
 *  capture depends only on (base model, song, crop) — so a second run over the
 *  same corpus reuses it, and re-running a whole training sweep does not pay
 *  for the capture every time. */
export function mm3PriorDir(slug: string): string {
  return path.join(mm3CodesDir(slug), 'prior');
}

// ── Arg building ────────────────────────────────────────────────────────────

export interface Mm3CodesArgs {
  datasetJson: string;
  outDir: string;
  maxDuration?: number;
  /** Cover-launder: real audio -> rec7 states -> flow DiT latents -> champion
   *  codes (ace-train mm3-launder). Ear-validated for dense-mix artists
   *  (deftones A/B, 2026-08-31). Off = plain mm3-codes, byte-identical to the
   *  pipeline before this option existed. */
  launder?: boolean;
}

export function buildMm3CodesArgs(a: Mm3CodesArgs): string[] {
  const m = resolveMm3TrainModels();
  const args = a.launder
    ? [
        'mm3-launder', '--jsonl',
        '--dataset', a.datasetJson,
        '--rvq', m.rvq,
        '--enc', m.enc,
        '--rec7', m.rec7,
        '--models', mm3ModelDir(),
        '--out', a.outDir,
      ]
    : [
        'mm3-codes', '--jsonl',
        '--dataset', a.datasetJson,
        '--rvq', m.rvq,
        '--enc', m.enc,
        '--out', a.outDir,
      ];
  // The launder renders through the DiT, so its 359 s ceiling is the engine's
  // 9000-frame cap, not a preference — clamp rather than refuse.
  const cap = a.launder ? 359 : Infinity;
  const md = a.maxDuration && a.maxDuration > 0 ? Math.min(Math.round(a.maxDuration), cap) : (a.launder ? 359 : 0);
  if (md > 0) args.push('--max-duration', String(md));
  return args;
}

export interface ResolvedMm3TrainLmOptions {
  manifest: string;
  captionsDir: string;
  codesDir: string;
  outDir: string;
  rank: number;
  alpha: number;
  lr: number;
  steps: number;
  saveEvery: number;
  warmup: number;
  gradAccum: number;
  seed: number;
  maxFrames: number;
  cropMode: 'random' | 'beginning' | 'structured';
  cropStartFrac: number;
  cropEndFrac: number;
  cropStartTiles: number;
  /** Lever 4a (2026-09-08): end crops draw their length in [endCropMin, K] and
   *  their frozen-prefix span in [0, prefixFrames]. Off unless asked. */
  endCropVary?: boolean;
  endCropMin?: number;
  /** Rev-7 ending-targeted prior (2026-09-08): score a reg step's loss on the
   *  last N supervised rows only. 0/absent = every row. */
  regScoreLast?: number;
  /** Style-step counterpart: score only the last N supervised rows of every
   *  style crop (SimpleTuner continuation objective, 2026-09-08). 0 = all. */
  scoreLast?: number;
  /** Apply scoreLast to END crops only (interior crops keep every row scored). */
  scoreLastEndOnly?: boolean;
  /** Server-side only (no engine flag): keep resume-state.bin after completion. */
  keepResumeState?: boolean;
  /** Tracks longer than maxFrames: exclude (engine --drop-over-frames), crop,
   *  or excise (cut a repeated section out and train the result whole).
   *  See MM3_LM_DEFAULTS.longTracks. Absent = exclude. */
  longTracks?: 'exclude' | 'crop' | 'excise';
  /** Engine --verify-export: after every checkpoint, load it back through the
   *  RUNTIME loader and compare against the live trainer (mm3-lm-verify-export.h).
   *  Opt-in: it briefly holds a second copy of the adapter on the card. */
  verifyExport?: boolean;
  /** Lyrics dropout (2026-09-08): share of style steps trained on a prompt
   *  without lyrics (instrumental marker). 0/absent = never. */
  lyricsDropout?: number;
  /** Stage A (2026-09-07): drop each style track's trailing digital silence
   *  before the EOS target, from <codes>/trim.json (tools/mm3-trim-silence).
   *  Off unless the request asks; the file must exist when it does. */
  trimTrailingSilence?: boolean;
  depthLossWeight: number;
  depthLossFrames: number;
  optimizer: 'muon' | 'adamw' | 'prodigy';
  muonLrScale: number;
  holdout: number;
  evalEvery: number;
  evalCrop: number;
  rankDropout: number;
  adapterType: 'lora' | 'lokr';
  lokrFactor: number;
  lokrDim: number;
  lokrAlpha: number;
  /** Attention backend. See MM3_LM_DEFAULTS.attnBackend. */
  attnBackend: 'exact' | 'flash';
  /** LoRA-family parameterizations. All five are LoKr-incompatible and
   *  mutually exclusive with each other (dora > rslora > hira > loha > pissa
   *  > hra precedence in buildMm3TrainLmArgs, textually identical to
   *  buildTrainDitArgs / buildTrainLmArgs). See MM3_LM_DEFAULTS. */
  rslora: boolean;
  dora: boolean;
  hira: boolean;
  loha: boolean;
  pissa: boolean;
  /** PiSSA with the mask on the principal component (implies pissa). */
  hotPizza: boolean;
  /** See MM3_LM_DEFAULTS.pissaCache / pissaFrozenF16. */
  pissaCache: boolean;
  pissaFrozenF16: boolean;
  hra: boolean;
  loraPlusRatio: number;
  /** Soft prompt. '' = no token. See MM3_LM_DEFAULTS.artistToken. */
  artistToken: string;
  artistTokenK: number;
  artistTokenLr: number;
  /** Trainable per-layer K/V prefix. 0 = off. Distinct from prefixFrames/
   *  prefixChunk (the frozen history prefix) — see MM3_LM_DEFAULTS.prefixN. */
  prefixN: number;
  /** One caption for EVERY track. The mechanism that binds a style to the
   *  prompt: with the caption constant across rows the adapter has nowhere
   *  to put the style except into itself. Empty = per-song captions. */
  captionFile?: string;
  trigger: string;
  /** Whether the trigger is injected into the training captions (and therefore
   *  learned at all). See Mm3TrainLmRequest.triggerPrepend. */
  triggerPrepend: boolean;
  datasetName: string;
  basePrecision: Mm3BasePrecision;
  cropAnchor: 'song' | 'zero';
  /** Frozen KV prefix. 0 = off. See MM3_LM_DEFAULTS.prefixFrames. */
  prefixFrames: number;
  prefixChunk: number;
  prefixSelftest: boolean;
  /** Cosine floor as a fraction of lr (SimpleTuner's lr_end / lr). */
  lrEndFrac: number;
  /** Prior preservation. All four move together or none of them do. */
  regManifest?: string;
  regCaptionsDir?: string;
  regCodesDir?: string;
  regPriorDir?: string;
  regEvery?: number;
  regTopK?: number;
  /** Mid-run audio previews. Undefined = off; the runner resolves the plan. */
  preview?: Mm3PreviewOptions;
  /** Which question the run is answering: "train for N steps" or "train until
   *  the loss reaches X". `steps` is the cap in BOTH modes — a target that is
   *  never met has to end somewhere. */
  stopMode: 'steps' | 'loss';
  /** Only read when stopMode is 'loss'. 0 = off. */
  targetLoss: number;
  /** 'train' averages the last `targetLossEpochs` completed passes over the
   *  dataset; 'eval' waits for a fresh held-out loss and needs holdout +
   *  evalEvery on. */
  targetLossMetric: 'train' | 'eval';
  targetLossEpochs: number;
  /** Set by the runner when relaunching a paused run, and by the RESUME route
   *  when continuing a finished or halted one — never by the start route. */
  resumeFrom?: string;
  /** Where a resumed run is picking up from. Only used for the segment loop's
   *  preview cadence and its timeout budget; the engine reads the step out of
   *  the state file itself. */
  resumeStep?: number;
}

export function buildMm3TrainLmArgs(o: ResolvedMm3TrainLmOptions): string[] {
  const m = resolveMm3TrainModels(o.basePrecision);
  // DERIVED, not a separate knob. Lever A needs a BF16-native base and a BF16
  // base has no other reason to be selected — it is the same size as f16 and
  // worse for inference — so the two are one decision. Offering them separately
  // would only create two ways to ask for a run the engine then falls back out
  // of, with a warning the user has to notice.
  const weights = o.basePrecision === 'bf16' ? 'bf16' : 'f32-window';
  const args = [
    'mm3-lm-train', '--jsonl',
    '--lm', m.lm,
    '--depth', m.depth,
    '--manifest', o.manifest,
    '--captions', o.captionsDir,
    '--codes', o.codesDir,
    '--out', o.outDir,
    '--rank', String(o.rank),
    '--alpha', String(o.alpha),
    '--lr', String(o.lr),
    '--lr-end-frac', String(o.lrEndFrac),
    '--steps', String(o.steps),
    '--save-every', String(o.saveEvery),
    '--warmup', String(o.warmup),
    '--grad-accum', String(o.gradAccum),
    '--seed', String(o.seed),
    '--max-frames', String(o.maxFrames),
    '--crop-mode', o.cropMode,
    '--optimizer', o.optimizer,
    '--weights', weights,
  ];
  if (o.cropMode === 'structured') {
    args.push('--crop-start-frac', String(o.cropStartFrac));
    args.push('--crop-end-frac', String(o.cropEndFrac));
    if (o.endCropVary) {
      args.push('--end-crop-vary');
      if (o.endCropMin) args.push('--end-crop-min', String(o.endCropMin));
    }
    args.push('--crop-start-tiles', String(o.cropStartTiles));
  }
  if (o.trimTrailingSilence) args.push('--trim-trailing-silence');
  // The second stopping strategy. --steps is still passed above and is still
  // the cap: a target the run never reaches has to end somewhere, and "runs
  // forever" is not an acceptable answer to "train until the loss is 0.2".
  if (o.stopMode === 'loss' && o.targetLoss > 0) {
    args.push('--target-loss', String(o.targetLoss));
    args.push('--target-loss-metric', o.targetLossMetric);
    args.push('--target-loss-epochs', String(o.targetLossEpochs));
  }
  args.push('--depth-loss-weight', String(o.depthLossWeight));
  args.push('--depth-loss-frames', String(o.depthLossFrames));
  if (o.optimizer === 'muon') args.push('--muon-lr-scale', String(o.muonLrScale));
  args.push('--holdout', String(o.holdout));
  args.push('--eval-every', String(o.evalEvery));
  args.push('--eval-crop', String(o.evalCrop));
  if (o.rankDropout > 0) args.push('--rank-dropout', String(o.rankDropout));
  if (o.adapterType === 'lokr') {
    args.push('--adapter-type', 'lokr');
    args.push('--lokr-factor', String(o.lokrFactor));
    args.push('--lokr-dim', String(o.lokrDim));
    args.push('--lokr-alpha', String(o.lokrAlpha));
  } else {
    // LoRA-family parameterizations. Precedence order is textually identical
    // to buildTrainDitArgs / buildTrainLmArgs (aceTrain.ts) so the same set of
    // booleans never resolves to a different method on a different trainer:
    // dora, then rslora (independent of dora — a rank-scaling modifier, not a
    // competing method), then hira (excludes dora), loha (excludes dora/hira),
    // pissa (excludes dora/hira/loha, and not resumable), hra (excludes
    // dora/hira/loha/pissa/rslora).
    if (o.dora) args.push('--dora');
    if (o.rslora) args.push('--rslora');
    if (o.hira && !o.dora) args.push('--hira');
    if (o.loha && !o.dora && !o.hira) args.push('--loha');
    // NOT suppressed on a resume. --pissa changes what the frozen base of the
    // delta IS (y = Wx + s*(BA - B0A0)x), and the A/B tensor set is identical
    // with and without it — so dropping the flag to make `--pissa --resume`
    // legal produced a run that looked like a resume and trained against a
    // different function from step 1. The engine refuses the pair outright
    // (ace-train.cpp) and the resume route refuses it before spawning; leaving
    // it on here means the illegal state fails loudly instead of quietly.
    if (o.pissa && !o.dora && !o.hira && !o.loha) {
      args.push(o.hotPizza ? '--hot-pizza' : '--pissa');
      if (o.pissaCache) args.push('--pissa-cache-dir', path.join(config.aceServer.adapters, 'mm3-lm-adapters', '_pissa-init-cache'));
      if (o.pissaFrozenF16) args.push('--pissa-frozen-f16');
    }
    // rslora is in the guard because the engine refuses --hra --rslora, but the
    // routes refuse that pair with a 400 first: HRA has no B for a rank-scaling
    // rule to apply to, so silently dropping it here would train a plain rsLoRA
    // LoRA and label the checkpoint HRA. This line is the backstop, not the
    // rule.
    if (o.hra && !o.dora && !o.hira && !o.loha && !o.pissa && !o.rslora) args.push('--hra');
  }
  if (o.loraPlusRatio && o.loraPlusRatio !== 1) args.push('--lora-plus-ratio', String(o.loraPlusRatio));
  // Attention backend — REAL on mm3-lm-train (landed 2026-09-05, mid-session,
  // alongside this change). Only the non-default value is emitted (same
  // "older exe stays compatible" rule as train-lm's --attn), so a build that
  // predates it never sees the flag for the 'exact' default every existing
  // caller still asks for. The route resolves the flash+frozen-prefix
  // collision (engine exit) before this ever runs — see attnBackendResolved
  // in routes/training.ts.
  if (o.attnBackend && o.attnBackend !== 'exact') args.push('--attn', o.attnBackend);
  // Whole-song recipe (2026-09-11): tracks that do not fit the window are left
  // out rather than trained in pieces. Emitted only for 'exclude' so an older
  // engine never sees the flag for the 'crop' behaviour it already has.
  // 'excise' keeps the flag too: the retarget pass shortens what it can, and whatever it refused must still be
  // left out rather than silently falling back to crops.
  if ((o.longTracks ?? 'exclude') !== 'crop') args.push('--drop-over-frames', String(o.maxFrames));
  // Soft prompt. Uses the SAME flag names the parser already accepts
  // (--artist-token/-k/-lr) — this is not a new engine surface, just a
  // previously-unwired one.
  if (o.artistToken) {
    args.push('--artist-token', o.artistToken, '--artist-token-k', String(o.artistTokenK),
              '--artist-token-lr', String(o.artistTokenLr));
  }
  // Trainable per-layer K/V prefix (lm-prefix.h). No cropAnchor restriction —
  // it has no position, unlike the frozen history prefix below. The engine
  // refuses it alongside --prefix-frames, so the emit below is suppressed
  // whenever this is on rather than letting the run exit 2.
  if (o.prefixN > 0) args.push('--prefix-n', String(o.prefixN));
  if (o.captionFile) args.push('--caption-file', o.captionFile);
  if (o.trigger) {
    args.push('--trigger', o.trigger);
    // Without this the word is recorded and never trained — the failure the
    // first album C run shipped with.
    if (o.triggerPrepend) args.push('--trigger-prepend');
  }
  if (o.datasetName) args.push('--dataset-name', o.datasetName);
  args.push('--crop-anchor', o.cropAnchor);
  // The engine refuses a prefix under `zero` anchoring (a history at positions
  // the window then reuses is a contradiction), so never emit that pair.
  if (o.prefixFrames > 0 && o.cropAnchor === 'song' && !(o.prefixN > 0)) {
    args.push('--prefix-frames', String(o.prefixFrames));
    args.push('--prefix-chunk', String(o.prefixChunk));
    if (o.prefixSelftest) args.push('--prefix-selftest');
  }
  // Prior preservation. Guarded on the whole set, not on `regEvery` alone: the
  // engine refuses a partial set, and it should never see one from here.
  if (o.regEvery && o.regEvery > 0 && o.regManifest && o.regCaptionsDir && o.regCodesDir) {
    args.push('--reg-manifest', o.regManifest);
    args.push('--reg-captions', o.regCaptionsDir);
    args.push('--reg-codes', o.regCodesDir);
    args.push('--reg-every', String(o.regEvery));
    if (o.regScoreLast && o.regScoreLast > 0) args.push('--reg-score-last', String(o.regScoreLast));
    args.push('--reg-topk', String(o.regTopK ?? MM3_LM_DEFAULTS.regTopK));
    if (o.regPriorDir) args.push('--reg-prior', o.regPriorDir);
  }
  // Style-step knobs, independent of the prior. Until 2026-09-09 01:20 these
  // two sat inside the regularisation block above, so every no-prior run that
  // asked for them (CONT, FAITHFUL, FAITHLYD) silently trained without them.
  if (o.scoreLast && o.scoreLast > 0) {
    args.push('--score-last', String(o.scoreLast));
    if (o.scoreLastEndOnly) args.push('--score-last-end-only');
  }
  if (o.verifyExport) args.push('--verify-export');
  if (o.lyricsDropout && o.lyricsDropout > 0) args.push('--lyrics-dropout', String(o.lyricsDropout));
  // Previews pause the trainer through a sentinel file. When they are off, say
  // so explicitly: a stray PAUSE left behind by a killed run would otherwise
  // stop the next run at its first step.
  if (!o.preview) args.push('--no-pause');
  // NOT gated on previews. It was, back when the only thing that produced a
  // state file was a preview pause — but continuing a finished run resumes from
  // a state written at the end of it, and that run may well have had previews
  // off. Gating here silently ignored the resume and retrained from step 1.
  if (o.resumeFrom) args.push('--resume', o.resumeFrom);
  return args;
}
