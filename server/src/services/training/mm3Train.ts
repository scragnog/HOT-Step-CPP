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
 *  Fitted to six measured points on a 5090 and accurate to <0.5% across the
 *  whole range it covers (10.2 GB to 22.6 GB), which is why the UI can show a
 *  number instead of a shrug:
 *
 *      peak = loaded + 31.2*rank + 1.4515*S + 441          S = 1142 + frames
 *
 *  `loaded` is the LM file plus ~1672 MB of fixed company (depth model, audio
 *  embeddings, tokenizer). 31.2 MB per rank is 12 bytes per LoRA parameter —
 *  weights, gradients and Muon momentum, one F32 copy each. The S terms are the
 *  36 per-layer checkpoints plus the graph arena, both linear in sequence
 *  length. 1142 is the typical MM3 prompt: it dominates short crops, which is
 *  why shrinking maxFrames is a weaker lever than it looks.
 *
 *  MEASURED f16 comes out ~1 GB ABOVE this (31.4 vs 30.3 predicted). The fit is
 *  over the quantized bases, where the in-graph cast is freed per segment; f16
 *  holds its weights differently. Treat an f16 estimate as a floor. */
export const MM3_VRAM_MODEL = {
  /** Depth model + audio embeddings + tokenizer, on top of the LM file. */
  loadedOverheadMb: 1672,
  /** 12 bytes per LoRA parameter (weights + grads + Muon momentum, F32 each),
   *  times 2.728 M parameters per unit of rank. */
  perRankMb: 31.2,
  /** 36 per-layer checkpoints plus the graph arena, both linear in sequence. */
  perTokenMb: 1.4515,
  /** Typical MM3 prompt. Added to maxFrames to get the sequence length, and the
   *  reason a short crop saves less than it looks like it should. */
  promptTokens: 1142,
  constMb: 441,
} as const;

/** Peak VRAM for a configuration, in MB.
 *
 *  SHIPPED TO THE UI AS COEFFICIENTS, not just as an answer: the form re-runs
 *  this as the user drags rank and crop length, and a second copy of these
 *  numbers over there would drift from the measurements that produced them. */
export function estimateMm3PeakMb(baseBytes: number, rank: number, maxFrames: number,
                                  extraMb = 0): number {
  const M      = MM3_VRAM_MODEL;
  const loaded = baseBytes / 1048576 + M.loadedOverheadMb;
  const S      = M.promptTokens + Math.max(0, maxFrames);
  return Math.round(loaded + M.perRankMb * rank + M.perTokenMb * S + M.constMb + extraMb);
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
const MM3_RANK_LADDER = [256, 128, 64, 32, 16];

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
    for (const rank of MM3_RANK_LADDER) {
      const peak = estimateMm3PeakMb(base.bytes, rank, maxFrames, base.extraMb);
      if (peak + headroom <= gpuTotalMb) {
        return { base: base.id, rank, overBudget: false };
      }
    }
  }
  // Nothing fits at any rank. Offer the SMALLEST usable base at the LOWEST rank
  // — the configuration with the best chance — and say so, rather than silently
  // landing on something that cannot run.
  const lowest = MM3_RANK_LADDER[MM3_RANK_LADDER.length - 1];
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
  rank: 256,
  alpha: 256,
  lr: 8e-5,
  steps: 800,
  saveEvery: 100,
  warmup: 0,
  gradAccum: 1,
  seed: 42,
  /** Requires random crops — with `beginning` this trains intros only, which
   *  is exactly what went wrong in the lm2 run. */
  maxFrames: 1500,
  cropMode: 'random' as 'random' | 'beginning',
  /** Muon, not AdamW, for two measured reasons: AdamW's second momentum buffer
   *  does not FIT at rank 256 on a 32 GB card, and Muon's normalised update
   *  makes an AdamW learning rate meaningless. 64 is the best of {1,4,16,64}
   *  measured over 50 steps — not a tuned optimum. */
  optimizer: 'muon' as 'muon' | 'adamw',
  muonLrScale: 64,
  /** q8_0, not f16 — see the note on Mm3TrainModels. Same step time since the
   *  cpy-q-occupancy patch, ~8.5 GB less resident, and therefore the only one
   *  of the two that survives a GPU shared with a desktop session. Pick f16
   *  only to reproduce a pre-patch run exactly. */
  basePrecision: 'q8_0' as Mm3BasePrecision,
  holdout: 0.15,
  evalEvery: 50,
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
} as const;

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
  cropMode: 'random' | 'beginning';
  optimizer: 'muon' | 'adamw';
  muonLrScale: number;
  holdout: number;
  evalEvery: number;
  trigger: string;
  datasetName: string;
  basePrecision: Mm3BasePrecision;
  cropAnchor: 'song' | 'zero';
  /** Mid-run audio previews. Undefined = off; the runner resolves the plan. */
  preview?: Mm3PreviewOptions;
  /** Set by the runner when relaunching a paused run — never by a route. */
  resumeFrom?: string;
}

export function buildMm3TrainLmArgs(o: ResolvedMm3TrainLmOptions): string[] {
  const m = resolveMm3TrainModels(o.basePrecision);
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
    '--steps', String(o.steps),
    '--save-every', String(o.saveEvery),
    '--warmup', String(o.warmup),
    '--grad-accum', String(o.gradAccum),
    '--seed', String(o.seed),
    '--max-frames', String(o.maxFrames),
    '--crop-mode', o.cropMode,
    '--optimizer', o.optimizer,
  ];
  if (o.optimizer === 'muon') args.push('--muon-lr-scale', String(o.muonLrScale));
  args.push('--holdout', String(o.holdout));
  args.push('--eval-every', String(o.evalEvery));
  if (o.trigger) args.push('--trigger', o.trigger);
  if (o.datasetName) args.push('--dataset-name', o.datasetName);
  args.push('--crop-anchor', o.cropAnchor);
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
