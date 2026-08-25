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
  };
}

/** Which of the required files are missing, as user-facing names. Empty = ready.
 *  `need` narrows the check: the codes job does not need the LM. */
export function missingMm3TrainModels(need: 'codes' | 'train',
                                      base: Mm3BasePrecision = 'f16'): string[] {
  const m = resolveMm3TrainModels(base);
  const wanted: Array<[string, string]> = need === 'codes'
    ? [[m.rvq, 'an RVQ encoder (mm3-rvq-*.gguf)'], [m.enc, 'the DAV encoder (mm3-enc-*.gguf)']]
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
} as const;

/** Peak VRAM for a configuration, in MB.
 *
 *  SHIPPED TO THE UI AS COEFFICIENTS, not just as an answer: the form re-runs
 *  this as the user drags rank and crop length, and a second copy of these
 *  numbers over there would drift from the measurements that produced them. */
export function estimateMm3PeakMb(baseBytes: number, rank: number, maxFrames: number,
                                  extraMb = 0,
                                  optimizer: 'muon' | 'adamw' | 'prodigy' = MM3_LM_DEFAULTS.optimizer): number {
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
  return Math.round(loaded + perRank * rank + M.perTokenMb * S + M.perTokenSqMb * S * S
                    + M.constMb + extraMb);
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
export function mm3CodesDir(slug: string): string {
  return path.join(datasetDir(slug), 'mm3-codes');
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
  // Sources, both with full configs: the SOAD tournament
  // (RareConcepts/soad-mm3-vanilla-20260822, simpletuner_config.json in every
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
  lr: 8e-5,
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
   *  from the step-size change, not a measured optimum. */
  steps: 500,
  /** 50, so a 500-step run yields 10 checkpoints to audition — the memorised
   *  run put the plausible ear-optimum somewhere in 150-350, and checkpoints
   *  every 250 would straddle it blind. */
  saveEvery: 50,
  /** One preview per checkpoint (Rob, 2026-08-25). The clock cadence made
   *  sense at 15 s steps and a 250-step save interval; at saveEvery 50 the
   *  checkpoint grid IS the cadence you want the ear to track. */
  previewEverySteps: 50,
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
  maxFrames: 750,
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
  /** 55/15 (random = the remainder, 30%). Raised from 40/15/45 on 2026-08-25:
   *  at crop 750 the random+end shares made ~60% of supervised audio mid-flow
   *  material with no arc, and the adapter's content prior followed it —
   *  renders jumped in "like a cut", and lowering render-MLP brought intros
   *  back at the cost of identity (Rob's A/B). Adapter deltas are position-
   *  independent, so the CONTENT MIX is the dial that decides whether songs
   *  open like songs; the position anchor alone cannot carry it. */
  cropStartFrac: 0.55,
  cropEndFrac: 0.15,
  /** Tiled starts: half the start share at frame 0, half across aligned tiles
   *  at K, 2K — so short crops still teach the intro→build→verse ARC in order,
   *  at true positions. 1 = frame-0 only (the old behaviour). 3 × crop 750
   *  covers what crop 2496's start share used to. */
  cropStartTiles: 3,
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
   *  trainer forces it to 1.0. On Green Day it converged to an effective 8.19e-5
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
   *  Green Day it converged within 2.4% of the hand-tuned 8e-5), resumes as of
   *  format v2 so checkpoint-cadence previews work, and `lr` becomes a
   *  schedule multiplier the trainer forces to 1.0. */
  optimizer: 'prodigy' as 'muon' | 'adamw' | 'prodigy',
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
   *  UNVALIDATED BY EAR: no LoKr adapter has been auditioned. Default on at
   *  Rob's request so it can be tested in-app. */
  adapterType: 'lokr' as 'lora' | 'lokr',
  lokrFactor: 6,
  lokrDim: 512,
  lokrAlpha: 512,
  /** Each crop is presented at its TRUE position in the track.
   *
   *  This is a RECIPE CHANGE as of 2026-08-23 and runs before it used `zero`,
   *  where every crop was labelled as if it were the song's opening. That was a
   *  straight train/inference mismatch — generation always begins at frame 0,
   *  so the positions a mid-song crop occupied during training are the ones
   *  that mean "the first two seconds" at render time. bghira's SOAD campaign
   *  independently reports the two symptoms this predicts (sound arriving
   *  instantly at 0:00, tempo drifting mid-track) and that position-labelled
   *  windowed crops fix the pacing. `zero` is kept only to reproduce an older
   *  run; the two are not comparable. */
  cropAnchor: 'song' as 'song' | 'zero',
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
} as const;

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
}

export function buildMm3CodesArgs(a: Mm3CodesArgs): string[] {
  const m = resolveMm3TrainModels();
  const args = [
    'mm3-codes', '--jsonl',
    '--dataset', a.datasetJson,
    '--rvq', m.rvq,
    '--enc', m.enc,
    '--out', a.outDir,
  ];
  if (a.maxDuration && a.maxDuration > 0) args.push('--max-duration', String(Math.round(a.maxDuration)));
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
  /** Set by the runner when relaunching a paused run — never by a route. */
  resumeFrom?: string;
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
    args.push('--crop-start-tiles', String(o.cropStartTiles));
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
  }
  if (o.captionFile) args.push('--caption-file', o.captionFile);
  if (o.trigger) {
    args.push('--trigger', o.trigger);
    // Without this the word is recorded and never trained — the failure the
    // first SOAD run shipped with.
    if (o.triggerPrepend) args.push('--trigger-prepend');
  }
  if (o.datasetName) args.push('--dataset-name', o.datasetName);
  args.push('--crop-anchor', o.cropAnchor);
  // Prior preservation. Guarded on the whole set, not on `regEvery` alone: the
  // engine refuses a partial set, and it should never see one from here.
  if (o.regEvery && o.regEvery > 0 && o.regManifest && o.regCaptionsDir && o.regCodesDir) {
    args.push('--reg-manifest', o.regManifest);
    args.push('--reg-captions', o.regCaptionsDir);
    args.push('--reg-codes', o.regCodesDir);
    args.push('--reg-every', String(o.regEvery));
    args.push('--reg-topk', String(o.regTopK ?? MM3_LM_DEFAULTS.regTopK));
    if (o.regPriorDir) args.push('--reg-prior', o.regPriorDir);
  }
  // Previews pause the trainer through a sentinel file. When they are off, say
  // so explicitly: a stray PAUSE left behind by a killed run would otherwise
  // stop the next run at its first step.
  if (o.preview) {
    if (o.resumeFrom) args.push('--resume', o.resumeFrom);
  } else {
    args.push('--no-pause');
  }
  return args;
}
