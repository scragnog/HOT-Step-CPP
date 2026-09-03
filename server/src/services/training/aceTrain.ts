// training/aceTrain.ts — ace-train binary discovery, tensor-cache paths and
// the cached engine model snapshot.
//
// The preprocess job stops ace-server to free VRAM, so the model picker cannot
// depend on the engine being reachable at the moment it is rendered. Every
// successful /props read is cached here and served while the engine is down
// (P28).
//
// Spec: docs/plans/2026-07-27-preprocess-implementation.md §4.2

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import { aceClient } from '../aceClient.js';
import { slugify, trainingBaseDir } from './paths.js';
import type {
  DitAdapterType, LmSize, PreprocessCompat, PreprocessDtype, PreprocessNormalize,
  PreprocessOptions, TrainDitStage, TrainLmStage,
} from './types.js';

/** Model-file extensions stripped when deriving a variant key. */
const MODEL_EXTENSIONS = ['.gguf', '.safetensors', '.bin', '.pt', '.pth', '.onnx'];

/** Absolute path to ace-train, or null. Sibling of ace-server in both the
 *  CMake and portable layouts — same pattern as aceMidiExe(). */
export function aceTrainExe(): string | null {
  const dir = path.dirname(config.aceServer.exe);
  const exe = path.join(dir, process.platform === 'win32' ? 'ace-train.exe' : 'ace-train');
  return fs.existsSync(exe) ? exe : null;
}

/** DiT model name → filesystem-safe variant key (extension stripped). */
export function variantKeyFor(ditModel: string): string {
  const raw = String(ditModel ?? '').replace(/\\/g, '/');
  let base = raw.slice(raw.lastIndexOf('/') + 1);
  // Only KNOWN model extensions are stripped: names like
  // `acestep-v15-merge-base-turbo-xl-ta-0.5` must not lose their `.5`.
  const lower = base.toLowerCase();
  for (const ext of MODEL_EXTENSIONS) {
    if (lower.endsWith(ext)) { base = base.slice(0, base.length - ext.length); break; }
  }
  const safe = base.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 96);
  return safe || 'default';
}

/** data/training/tensors/<slug> */
export function tensorsRoot(slug: string): string {
  return path.join(trainingBaseDir, 'tensors', slugify(slug));
}

/** data/training/tensors/<slug>/<variantKey> */
export function tensorsDir(slug: string, variantKey: string): string {
  return path.join(tensorsRoot(slug), variantKeyFor(variantKey));
}

// ── Prior-preservation corpus discovery (2026-09-02) ─────────────────────

/** One other artist's regularisation corpus: the newest 600 s-cap variant
 *  that has an extracted lm_codes.jsonl. */
export interface RegCorpusCandidate {
  slug: string;
  variantKey: string;
  codesPath: string;
}

/** FNV-1a-ish string hash, used only to pick a deterministic-but-varying
 *  rotation offset per dataset — not for anything security-sensitive. */
function stableHash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Up to `max` OTHER artists' lm_codes.jsonl for the --reg-codes corpus.
 *
 * Only variants whose preprocess_meta.json records max_duration 600 qualify
 * (§A-1: the 53 files extracted under the old 240 s cap taught truncated
 * endings — a regularisation corpus is exactly the wrong place to reintroduce
 * that). One variant per artist (the newest qualifying one). The excluded
 * slug is the dataset being trained — regularising against your own songs
 * cancels the objective (same rule as resolveMm3Regularisation in training.ts).
 *
 * Order is deterministic per call (sorted slugs) but the SELECTION rotates by
 * a hash of `excludeSlug`, so two different artists' auto-picked reg sets
 * differ from each other rather than every artist regularising against the
 * same fixed six.
 */
export function findRegCorpora(excludeSlug: string, max = 6): RegCorpusCandidate[] {
  const root = path.join(trainingBaseDir, 'tensors');
  const excluded = slugify(excludeSlug);
  let slugs: string[] = [];
  try {
    slugs = fs.readdirSync(root, { withFileTypes: true })
      .filter(e => e.isDirectory() && e.name !== excluded)
      .map(e => e.name)
      .sort();
  } catch {
    return [];
  }

  const candidates: RegCorpusCandidate[] = [];
  for (const slug of slugs) {
    const slugRoot = path.join(root, slug);
    let variants: string[] = [];
    try {
      variants = fs.readdirSync(slugRoot, { withFileTypes: true })
        .filter(e => e.isDirectory())
        .map(e => e.name)
        .sort();
    } catch {
      continue;
    }
    // Newest-first: the last sorted entry is usually newest (timestamp-ish
    // variant keys), but sort by preprocess_meta.json created_at properly.
    let best: { variantKey: string; codesPath: string; createdAt: string } | null = null;
    for (const variantKey of variants) {
      const vdir = path.join(slugRoot, variantKey);
      const metaPath = path.join(vdir, 'preprocess_meta.json');
      const codesPath = path.join(vdir, 'lm_codes.jsonl');
      if (!fs.existsSync(metaPath) || !fs.existsSync(codesPath)) continue;
      let meta: { max_duration?: unknown; created_at?: unknown };
      try {
        meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
      } catch {
        continue;
      }
      if (Number(meta.max_duration) !== 600) continue;
      const createdAt = typeof meta.created_at === 'string' ? meta.created_at : '';
      if (!best || createdAt > best.createdAt) best = { variantKey, codesPath, createdAt };
    }
    if (best) candidates.push({ slug, variantKey: best.variantKey, codesPath: best.codesPath });
  }
  if (candidates.length <= max) return candidates;

  const start = stableHash(excludeSlug) % candidates.length;
  const rotated = [...candidates.slice(start), ...candidates.slice(0, start)];
  return rotated.slice(0, max);
}

// ── Cached /props model snapshot (P28) ───────────────────────────────────

export interface ModelSnapshot {
  dit: string[]; vae: string[]; textEnc: string[]; lm: string[]; cachedAt: number;
}

let snapshot: ModelSnapshot = { dit: [], vae: [], textEnc: [], lm: [], cachedAt: 0 };

/** Last successful /props read. Survives the engine being stopped mid-job. */
export function getModelSnapshot(): ModelSnapshot {
  return snapshot;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : [];
}

/** Re-probe /props. Never throws — returns the previous snapshot on any failure. */
export async function refreshModelSnapshot(): Promise<ModelSnapshot> {
  try {
    const props = await aceClient.props();
    const dit = stringList(props?.models?.dit);
    const vae = stringList(props?.models?.vae);
    // `models.embedding` IS the text-encoder bucket (Qwen3-Embedding).
    const textEnc = stringList(props?.models?.embedding);
    const lm = stringList(props?.models?.lm);
    // §4.2 says the previous snapshot survives a FAILURE, not emptiness. A
    // successful /props that honestly reports three empty buckets (models
    // deleted, ACESTEPCPP_MODELS repointed) must replace the cache, or the
    // picker offers names that no longer exist and the POST validation accepts
    // them — the user only finds out as an ace-train exit 2, after the engine
    // has already been stopped. A malformed response (no `models` object at
    // all) is still treated as a failure.
    if (props && typeof props === 'object' && props.models && typeof props.models === 'object') {
      snapshot = { dit, vae, textEnc, lm, cachedAt: Date.now() };
    }
  } catch {
    // Engine down or stopped for a job — the cache is exactly what we want.
  }
  return snapshot;
}

/** First name matching /bf16/i, else ''. */
export function pickBf16(names: string[]): string {
  return names.find(n => /bf16/i.test(n)) ?? '';
}

/**
 * Preferred BF16 LM for a size, e.g. '0.6B' -> 'acestep-5Hz-lm-0.6B-BF16.gguf'.
 *
 * Match rule (§4.2): the name contains `-<size>-` (case-insensitive) AND /bf16/i.
 * Falls back to the first name containing `-<size>-` at all; '' when none.
 * The fallback is deliberately kept: the trainer refuses a genuinely quantized
 * base at mirror time with a clear message, which beats offering nothing.
 */
export function pickLmFor(size: LmSize, names: string[]): string {
  // '0.6B' carries a literal '.', which must not become a regex wildcard.
  const token = new RegExp(`-${size.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-`, 'i');
  const matches = names.filter(n => token.test(n));
  return matches.find(n => /bf16/i.test(n)) ?? matches[0] ?? '';
}

// ── Argv construction ────────────────────────────────────────────────────

/** Every PreprocessOptions field the runner needs, with defaults already applied
 *  by the route. `job.opts` carries exactly this shape for a preprocess job. */
export interface ResolvedPreprocessOptions {
  ditModel: string;
  vaeModel: string;
  textEncoder: string;
  maxDuration: number;
  normalize: PreprocessNormalize;
  targetDb: number;
  dtype: PreprocessDtype;
  compat: PreprocessCompat;
  maxCaptionTokens: number;
  maxLyricTokens: number;
  vaeChunk: number;
  vaeOverlap: number;
  overwrite: boolean;
  stopEngine: boolean;
  /** Absolute; data/training/tensors/<slug>/<variantKey> unless overridden. */
  outputDir: string;
  variantKey: string;
}

type PreprocessArgOpts = Required<Pick<PreprocessOptions,
  'maxDuration' | 'normalize' | 'targetDb' | 'dtype' | 'compat' |
  'maxCaptionTokens' | 'maxLyricTokens' | 'vaeChunk' | 'vaeOverlap' | 'overwrite'>>;

/** Build the full argv for `ace-train preprocess` (§2.1 order). */
export function buildPreprocessArgs(input: {
  manifestPath: string; outDir: string; modelsDir: string;
  dit: string; vae: string; textEnc: string;
  opts: PreprocessArgOpts;
  ffmpeg: string | null;
}): string[] {
  const o = input.opts;
  const args = [
    'preprocess',
    '--manifest', input.manifestPath,
    '--out', input.outDir,
    '--models', input.modelsDir,
    '--dit', input.dit,
    '--vae', input.vae,
    '--text-enc', input.textEnc,
    '--max-duration', String(o.maxDuration),
    '--normalize', o.normalize,
    '--target-db', String(o.targetDb),
    '--dtype', o.dtype,
    '--compat', o.compat,
    '--max-caption-tokens', String(o.maxCaptionTokens),
    '--max-lyric-tokens', String(o.maxLyricTokens),
    '--vae-chunk', String(o.vaeChunk),
    '--vae-overlap', String(o.vaeOverlap),
  ];
  if (input.ffmpeg) args.push('--ffmpeg', input.ffmpeg);
  if (o.overwrite) args.push('--overwrite');
  args.push('--jsonl');
  return args;
}

// ── LM LoRA training (phase 3) ───────────────────────────────────────────

/** Every TrainLmOptions field the runner needs, with defaults and clamps
 *  already applied by the route. `job.opts` carries exactly this shape for a
 *  train-lm job. Spec §4.2. */
export interface ResolvedTrainLmOptions {
  lmSize: LmSize;
  lmModel: string;
  ditModel: string;
  variantKey: string;
  /** Absolute preprocess variant dir the codes are extracted from. */
  tensorsDir: string;
  /** Absolute <tensorsDir>/lm_codes.jsonl. */
  codesPath: string;
  adapterName: string;
  /** Absolute <adapters>/lm/<adapterName>-<lmSize>. */
  adapterDir: string;
  targetLoss: number;
  /** Staged target-loss chain (2026-08-29). Each entry is one full ace-train
   *  leg; legs after the first --init-adapter from the previous leg's export,
   *  which resets the optimizer state and the LR schedule between stages.
   *  Validated to prevent the straight-dive loop attractor AND to beat the
   *  straight run on songwriting (nirvana E3, gojira chainfix — see the
   *  2026-08-29 listening folders). The LAST entry equals `targetLoss`; a
   *  single-entry array reproduces the legacy one-shot run. */
  targetLossStages: number[];
  epochs: number;
  /** Adapter parameterization. 'lokr' emits the LyCORIS kron factors and makes
   *  rank/alpha inert; the exporter writes lokr_weights.safetensors. */
  adapterType: 'lora' | 'lokr';
  /** Optimizer rule set. 'muon' orthogonalizes 2-D parameters (short side >=
   *  16, which for a LoRA is the RANK) and leaves the rest on AdamW. */
  optimizer: 'adamw' | 'muon' | 'prodigy';
  muonLrScale: number;
  muonNsSteps: number;
  rank: number;
  alpha: number;
  lokrDim: number;
  lokrAlpha: number;
  lokrFactor: number;
  lokrDecomposeBoth: boolean;
  learningRate: number;
  gradAccum: number;
  gradClip: number;
  warmupRatio: number;
  weightDecay: number;
  maxLen: number;
  seed: number;
  lossOnCot: boolean;
  order: 'shuffle' | 'fixed';
  milestoneStep: number;
  milestoneKeep: number;
  stages: TrainLmStage[];
  overwrite: boolean;
  stopEngine: boolean;
  /** Resume source run dir ('' = scratch). When set, the identity flags
   *  (adapter type, rank/alpha, lokr dims, --weights) are NOT emitted — the
   *  engine adopts them from the source run's log and refuses contradictions,
   *  so emitting the form's defaults would fail every resume of a run whose
   *  identity differs from the current form state. */
  initAdapter: string;
  calibrate: boolean;
  calibrateRepoint: boolean;
  /** 'auto' = the engine's own default (ON for 4B, and for smaller bases only
   *  when the naive fit would drop full-song samples). */
  lowVram: 'auto' | 'on' | 'off';
  /** 0 = engine picks (n_heads <= 16 -> off, else 8). */
  attnHeadBlock: number;
  /** 0 = engine default (128 trained positions per CE chunk). */
  chunk: number;
  /** 'f32-window' = the shipped per-segment F32 weight cast (still the CLI's
   *  own default, ace-train.cpp). 'bf16' = BF16 projections + backward
   *  surgery; needs CUDA + a BF16-native base + low-VRAM and changes the
   *  trained weights. A non-CUDA backend or non-BF16 base each warn and fall
   *  back to 'f32-window' (lm-train-run.h) rather than failing the run. The
   *  SERVER default is 'bf16' (2026-07-29, training.ts train-lm handler). */
  weights: 'f32-window' | 'bf16';
  /** Micro-batch size 1..8, or 'auto'. 1 is the CLI default; >1 implies low-VRAM. */
  batch: number | 'auto';
  /** MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch).
   *  'outprod' = upstream ggml out_prod, F32-only on CUDA. 'mm' =
   *  mul_mat(cont(transpose(W)), grad) — identical maths, dtype-agnostic, so a
   *  BF16 weight uses BF16 tensor cores (~1.7-1.8x per layer on an RTX 5090).
   *  ace-train's own default is 'outprod', and so is the LM SERVER default —
   *  `weights: 'bf16'` already reaches the same backward by rewriting ggml's
   *  out_prod nodes in place (lm-bf16.h) and aborts if --bwd mm leaves it none,
   *  so the route refuses that pair. train-dit, which has no such surgery,
   *  defaults to 'mm'. */
  bwd: 'outprod' | 'mm';
  /** Caption dropout fraction, 0..1. 0 = off, no flag emitted. Emitted on
   *  every leg of a staged chain, including resumes — this is a recipe knob
   *  like --lr, not an adapter-identity flag. */
  captionDropout: number;
  rslora: boolean;
  loraPlusRatio: number;
  /** Soft prompt: '' = no token. Emitted on every leg; the engine adopts the
   *  trained token/prefix from --init-adapter on a resume. */
  artistToken: string;
  artistTokenK: number;
  artistTokenLr: number;
  prefixN: number;
  /** Regularisation cadence. 0 = off, no --reg-* flags emitted at all. */
  regEvery: number;
  regTopk: number;
  regSongs: number;
  /** Comma-separated absolute paths to other artists' lm_codes.jsonl, already
   *  resolved by the route (findRegCorpora for 'auto', or the caller's
   *  explicit list). '' when regEvery is 0. */
  regCodes: string;
  /** Where the captured base-model prior distributions are cached. Fixed at
   *  the TOP-LEVEL adapter dir (not per-stage-leg) so every leg of a staged
   *  chain shares the same cache and it is captured only once. '' when
   *  regEvery is 0. */
  regPriorDir: string;
  /** Prior teacher (docs/plans/lm-attr-probe/OVERNIGHT.md "Live teacher spec").
   *  'cached' is today's byte-identical top-K prior; 'live' scores every reg
   *  step against the frozen base's full live distribution instead. Only
   *  emitted (as `--reg-teacher live`) when non-default AND regEvery > 0. */
  regTeacher: 'cached' | 'live';
  /** Attention backend (2026-09-02 lm-flash-attn plan, Stream B) — the DiT's
   *  attnBackend ported to train-lm. 'exact' is the byte-identical graph;
   *  'flash' routes through the fused ggml_flash_attn_train/_back kernels.
   *  See ResolvedTrainDitOptions.attnBackend for the shared background;
   *  unlike the DiT's, this one is emitted ONLY when non-default (see
   *  buildTrainLmArgs), matching `weights`'s older-exe compatibility rule —
   *  train-lm does not parse --attn on every ace-train build yet. */
  attnBackend: 'exact' | 'flash' | 'flash-f32';
}

/** Full argv for `ace-train train-lm` (§2.1 order). */
export function buildTrainLmArgs(input: {
  opts: ResolvedTrainLmOptions; modelsDir: string;
}): string[] {
  const o = input.opts;
  const args = [
    'train-lm',
    '--stages', o.stages.join(','),
    '--tensors', o.tensorsDir,
    '--codes', o.codesPath,
    '--out', o.adapterDir,
    '--models', input.modelsDir,
    // `--dit` DEFAULTS to preprocess_meta.json's dit_path (§2.1). Omitting the
    // pair is not the same as passing an empty value: resolve_model('') fails and
    // the sibling cmd_preprocess exits 2 on it — which here would happen only
    // AFTER the runner stopped ace-server, so the user would pay a full engine
    // stop/restart cycle for a resolvable-by-default condition. ditModel is ''
    // whenever the variant's meta is missing/unparseable or lacks model_variant.
    ...(o.ditModel ? ['--dit', o.ditModel] : []),
    '--lm', o.lmModel,
    '--lm-size', o.lmSize,
    // Resume: the engine adopts identity hyperparams (adapter type, rank/alpha,
    // lokr dims, --weights) from the source run's log and REFUSES explicit
    // contradictions — so with --init-adapter the identity flags are omitted
    // entirely; emitting the form's defaults would fail every resume of a run
    // whose identity differs from the current form state.
    ...(o.initAdapter ? ['--init-adapter', o.initAdapter] : []),
    // Always emitted (scratch runs) so an ace-train that predates
    // --adapter-type rejects it loudly rather than silently training a LoRA
    // when a LoKr was asked for.
    ...(o.initAdapter ? [] : ['--adapter-type', o.adapterType]),
    // Always emitted so an ace-train that predates --optimizer rejects it loudly
    // rather than silently training on AdamW when Muon was asked for.
    '--optimizer', o.optimizer,
    ...(o.optimizer === 'muon'
      ? ['--muon-lr-scale', String(o.muonLrScale), '--muon-ns-steps', String(o.muonNsSteps)]
      : []),
    ...(o.initAdapter ? []
      : o.adapterType === 'lokr'
        ? ['--lokr-dim', String(o.lokrDim), '--lokr-alpha', String(o.lokrAlpha), '--lokr-factor', String(o.lokrFactor)]
        : ['--rank', String(o.rank), '--alpha', String(o.alpha)]),
    '--lr', String(o.learningRate),
    '--epochs', String(o.epochs),
    '--grad-accum', String(o.gradAccum),
    '--warmup-ratio', String(o.warmupRatio),
    '--grad-clip', String(o.gradClip),
    '--weight-decay', String(o.weightDecay),
    '--seed', String(o.seed),
    '--target-loss', String(o.targetLoss),
    '--order', o.order,
    '--max-len', String(o.maxLen),
    '--milestone-step', String(o.milestoneStep),
    '--milestone-keep', String(o.milestoneKeep),
  ];
  // Low-VRAM knobs: every default here IS the CLI default, so a normal run
  // emits none of them and the argv stays byte-identical to the pre-4B build.
  // That keeps an older ace-train.exe (no --low-vram flag) working for the
  // untouched 0.6B/1.7B path.
  if (o.lowVram && o.lowVram !== 'auto') args.push('--low-vram', o.lowVram);
  if (o.attnHeadBlock > 0) args.push('--attn-head-block', String(o.attnHeadBlock));
  if (o.chunk > 0) args.push('--lm-chunk', String(o.chunk));
  // Speed levers (2026-07-28 plan §1.3). `batch` still defaults to the CLI's
  // own default (1), so a normal run emits no --batch flag. `weights` no
  // longer matches the CLI default: the server defaults it to 'bf16'
  // (2026-07-29), so a normal run now DOES emit --weights bf16 explicitly.
  // An older ace-train.exe without --weights/--batch only stays compatible
  // if the caller explicitly requests 'f32-window'.
  // Suppressed on resume like the other identity flags — the S6 rule makes a
  // --weights that contradicts the source run a hard refuse, and the server
  // default ('bf16') contradicts the whole f32-window-trained corpus.
  if (!o.initAdapter && o.weights && o.weights !== 'f32-window') args.push('--weights', o.weights);
  if (o.batch !== undefined && o.batch !== 1) args.push('--batch', String(o.batch));
  // Always emitted, both sides: an ace-train that predates --bwd rejects it
  // loudly rather than silently running the slow out_prod backward.
  args.push('--bwd', o.bwd);
  // Caption dropout + prior preservation (2026-09-02). Both are recipe knobs,
  // not adapter-identity flags, so — unlike rank/alpha/adapter-type/weights —
  // they are emitted on EVERY leg of a staged chain, resumes included: the
  // point is that every leg trains under the same regularised objective, not
  // just the first one. Only-non-default emission (same rule as --weights)
  // keeps an older ace-train.exe that predates these flags working.
  if (o.captionDropout > 0) args.push('--caption-dropout', String(o.captionDropout));
  if (o.rslora && o.adapterType === 'lora') args.push('--rslora');
  if (o.loraPlusRatio && o.loraPlusRatio !== 1) args.push('--lora-plus-ratio', String(o.loraPlusRatio));
  // Soft prompt (token + prefix). These ARE adapter-identity flags, but they
  // are emitted on every leg on purpose: the engine reads the trained vectors
  // out of --init-adapter and only uses these to know the token is wanted.
  if (o.artistToken) {
    args.push('--artist-token', o.artistToken, '--artist-token-k', String(o.artistTokenK),
              '--artist-token-lr', String(o.artistTokenLr));
  }
  if (o.prefixN > 0) args.push('--prefix-n', String(o.prefixN));
  if (o.regEvery > 0 && o.regCodes) {
    args.push('--reg-codes', o.regCodes);
    args.push('--reg-songs', String(o.regSongs));
    args.push('--reg-every', String(o.regEvery));
    args.push('--reg-topk', String(o.regTopk));
    args.push('--reg-prior-dir', o.regPriorDir);
    if (o.regTeacher === 'live') args.push('--reg-teacher', 'live');
  }
  // Attention backend (2026-09-02 lm-flash-attn plan, Stream B). Only the
  // non-default value is emitted — unlike train-dit's --attn, which is always
  // sent — so an ace-train build that predates the LM's --attn parsing (every
  // build so far) stays compatible for the default 'exact' run every caller
  // still makes. Emitted on every leg of a staged chain, resumes included:
  // this is a recipe knob (which attention formulation ran), not an
  // adapter-identity flag the engine would need to adopt from a resumed run.
  if (o.attnBackend && o.attnBackend !== 'exact') args.push('--attn', o.attnBackend);
  // `--loss-on-cot` is the CLI default; only the negation needs emitting.
  if (!o.lossOnCot) args.push('--no-loss-on-cot');
  if (o.overwrite) args.push('--overwrite');
  args.push('--jsonl');
  return args;
}

// ── DiT LoRA training (phase 4) ──────────────────────────────────────────

/** Every TrainDitOptions field the runner needs, with defaults and clamps
 *  already applied by the route. `job.opts` carries exactly this shape for a
 *  train-dit job. Spec §4.2. */
export interface ResolvedTrainDitOptions {
  variantKey: string; tensorsDir: string;
  ditModel: string; ditPath: string;
  adapterName: string; adapterDir: string;
  adapterType: DitAdapterType; rank: number; alpha: number; targetMlp: boolean; dora?: boolean;
  rslora?: boolean; loraPlusRatio?: number; hira?: boolean; loha?: boolean;
  // LyCORIS LoKR factors (K2 / plan §2.1). Always resolved regardless of
  // adapterType — buildTrainDitArgs only emits them when adapterType==='lokr'.
  lokrDim: number; lokrAlpha: number; lokrFactor: number; lokrDecomposeBoth: boolean;
  layers: number; crop: number; cropMin: number; cropMax: number;
  /** Crop regime (2026-08-29): song-anchored RoPE positions + structured
   *  start/end-weighted crop draws are the fixed defaults; 'zero'/'random'
   *  reproduce the legacy behaviour. Emitted always so an older ace-train
   *  rejects them loudly rather than silently training the old way. */
  cropAnchor: 'song' | 'zero'; cropMode: 'structured' | 'random';
  cropStartFrac: number; cropEndFrac: number;
  /** --crop-jitter (2026-09-03, experimental): per-draw crop length uniform
   *  over [cropMin, crop]. Off = the engine's two-draw sampler, byte-identical. */
  cropJitter: boolean;
  targetLoss: number; epochs: number; learningRate: number;
  gradAccum: number; gradClip: number; warmupRatio: number; weightDecay: number;
  lossWeighting: 'none' | 'flow_snr'; snrGamma: number; tBias: number;
  channelBalance: boolean; timestepMu: number; timestepSigma: number;
  tMin: number; tMax: number; cfgRatio: number; genreRatio: number;
  seed: number; order: 'shuffle' | 'fixed';
  milestoneStep: number; milestoneKeep: number; vramReserveMb: number;
  /** Frozen-weight mirror precision, passed straight to `--mirror`. 'bf16-f32'
   *  (2026-09-02) is bf16 storage with f32 arithmetic — see
   *  StartDitTrainingRequest.mirror in types.ts for the full comparison. */
  mirror: 'f32' | 'bf16' | 'bf16-f32';
  /** MUL_MAT activation-gradient formulation — see ResolvedTrainLmOptions.bwd.
   *  ace-train defaults to 'outprod'; the SERVER default is 'mm'. */
  bwd: 'outprod' | 'mm';
  /** Optimizer rule set (2026-07-30). 'adamw' is the shipped path; 'muon' is
   *  per-parameter — 2-D parameters with a short side >= muonMinDim get
   *  orthogonalized-momentum updates, the rest stay on AdamW. */
  optimizer: 'adamw' | 'muon' | 'prodigy';
  muonLrScale: number;
  muonMomentum: number;
  muonNsSteps: number;
  muonMinDim: number;
  batch: number; ckptSegments: number;
  stages: TrainDitStage[]; overwrite: boolean; stopEngine: boolean;
  /** Resume source run dir ('' = scratch). When set, identity flags (adapter
   *  type, rank/alpha, lokr dims, target-mlp, layers) are NOT emitted — the
   *  engine adopts them from the source run's log and refuses contradictions.
   *  Same rule as ResolvedTrainLmOptions.initAdapter. */
  initAdapter: string;
  calibrate: boolean;
  calibrateRepoint: boolean;
  /** Attention backend (2026-09-01, docs/plans/2026-09-01-flash-attn-backward.md).
   *  'exact' is the byte-identical dit_attn_f32 graph; 'flash' routes through
   *  the fused ggml_flash_attn_train/_back kernels, enabling much longer
   *  training crops at some per-step speed cost. Always emitted so an
   *  ace-train that predates --attn rejects it loudly rather than silently
   *  running the other backend.
   *
   *  'flash-f32' is the same fused ops pinned to strict f32 — the scalar
   *  kernels rather than the TF32 tensor-core ones 'flash' selects. API-only:
   *  the Training Studio checkbox never produces it. */
  attnBackend: 'exact' | 'flash' | 'flash-f32';
}

/**
 * The variant's own base DiT, as an ABSOLUTE path, '' when it is no longer on
 * disk (§4.2 base-match guard).
 *
 * The encoder states and context latents in the cache are that exact model's
 * outputs, so training against anything else is silently wrong — which is why
 * this reads the variant's record and never user input. Three sources, in
 * descending order of trust:
 *   1. `dit_path`      — the absolute path the preprocess run actually loaded
 *   2. `model_variant` — the file name, resolved against the models dir
 *   3. `variantKey`    — the same name with its extension stripped (§ variantKeyFor)
 * Returning '' is the caller's cue to answer 400 rather than stop the engine
 * for a run that cannot load its base.
 */
export function pickDitBaseFor(variantKey: string, tensorsDirPath: string): string {
  const exists = (p: string): boolean => {
    try { return !!p && fs.existsSync(p); } catch { return false; }
  };

  let ditPath = '';
  let modelVariant = '';
  try {
    const meta = JSON.parse(
      fs.readFileSync(path.join(tensorsDirPath, 'preprocess_meta.json'), 'utf-8'),
    ) as { dit_path?: unknown; model_variant?: unknown };
    if (typeof meta.dit_path === 'string') ditPath = meta.dit_path.trim();
    if (typeof meta.model_variant === 'string') modelVariant = meta.model_variant.trim();
  } catch {
    return '';   // no meta = not a real variant
  }

  if (exists(ditPath)) return ditPath;

  // The models dir may have moved since the preprocess run (portable release,
  // ACESTEPCPP_MODELS repointed) while the same file is still installed.
  const modelsDir = config.aceServer.models;
  if (modelVariant) {
    const byName = path.join(modelsDir, modelVariant);
    if (exists(byName)) return byName;
  }
  if (variantKey) {
    for (const ext of ['.gguf', '']) {
      const byKey = path.join(modelsDir, `${variantKey}${ext}`);
      if (exists(byKey)) return byKey;
    }
  }
  return '';
}

/** Full argv for `ace-train train-dit` (§2.1 order). */
export function buildTrainDitArgs(input: {
  opts: ResolvedTrainDitOptions; modelsDir: string;
}): string[] {
  const o = input.opts;
  const args = [
    'train-dit',
    '--stages', o.stages.join(','),
    '--tensors', o.tensorsDir,
    '--out', o.adapterDir,
    '--models', input.modelsDir,
    // Always present in practice — the route refuses the request when
    // pickDitBaseFor() came back empty, so the engine never has to fall back to
    // preprocess_meta.json's own default. Guarded anyway: passing an empty value
    // would make resolve_model('') exit 2, AFTER the runner already stopped the
    // engine.
    ...(o.ditPath ? ['--dit', o.ditPath] : []),
    // Resume: identity flags (adapter type, rank/alpha, lokr dims, layers,
    // target-mlp below) are omitted — the engine adopts them from the source
    // run's log and refuses explicit contradictions, so emitting the form's
    // defaults would fail every resume of a differently-shaped run. Same rule
    // as buildTrainLmArgs.
    ...(o.initAdapter ? ['--init-adapter', o.initAdapter] : []),
    ...(o.initAdapter ? [] : ['--adapter-type', o.adapterType]),
    ...(o.dora && o.adapterType === 'lora' ? ['--dora'] : []),
    ...(o.rslora && o.adapterType === 'lora' ? ['--rslora'] : []),
    ...(o.hira && o.adapterType === 'lora' && !o.dora ? ['--hira'] : []),
    ...(o.loha && o.adapterType === 'lora' && !o.dora && !o.hira ? ['--loha'] : []),
    ...(o.loraPlusRatio && o.loraPlusRatio !== 1 ? ['--lora-plus-ratio', String(o.loraPlusRatio)] : []),
    // §2.1: lora trains via --rank/--alpha; lokr via the four --lokr-* flags.
    // The two are mutually exclusive on the CLI side, so only one set is ever
    // emitted — sending both would be harmless (ace-train ignores the unused
    // side) but would misreport the run in logs/JSONL relays that echo argv.
    ...(o.initAdapter ? []
      : o.adapterType === 'lokr'
        ? [
            '--lokr-dim', String(o.lokrDim),
            '--lokr-alpha', String(o.lokrAlpha),
            '--lokr-factor', String(o.lokrFactor),
            // Flag-shaped, default on (§2.1) — same "only the non-default side
            // is emitted" convention as --channel-balance below.
            ...(o.lokrDecomposeBoth ? [] : ['--no-lokr-decompose-both']),
          ]
        : ['--rank', String(o.rank), '--alpha', String(o.alpha)]),
    ...(o.initAdapter ? [] : ['--layers', String(o.layers)]),
    '--crop', String(o.crop),
    '--crop-min', String(o.cropMin),
    // 0 = don't pass the flag. The engine treats an EXPLICIT --crop-max as a
    // user pin that must never be moved, and only lifts the default cap to the
    // dataset's longest track in flash mode — so always emitting it would kill
    // the lift for every server-launched run (the day-one-dead-feature trap,
    // one level up).
    ...(o.cropMax > 0 ? ['--crop-max', String(o.cropMax)] : []),
    '--crop-anchor', o.cropAnchor,
    '--crop-mode', o.cropMode,
    '--crop-start-frac', String(o.cropStartFrac),
    '--crop-end-frac', String(o.cropEndFrac),
    ...(o.cropJitter ? ['--crop-jitter'] : []),
    '--loss-weighting', o.lossWeighting,
    '--snr-gamma', String(o.snrGamma),
    '--t-bias', String(o.tBias),
    '--timestep-mu', String(o.timestepMu),
    '--timestep-sigma', String(o.timestepSigma),
    '--t-min', String(o.tMin),
    '--t-max', String(o.tMax),
    '--cfg-ratio', String(o.cfgRatio),
    '--genre-ratio', String(o.genreRatio),
    '--lr', String(o.learningRate),
    '--epochs', String(o.epochs),
    '--grad-accum', String(o.gradAccum),
    '--warmup-ratio', String(o.warmupRatio),
    '--grad-clip', String(o.gradClip),
    '--weight-decay', String(o.weightDecay),
    '--seed', String(o.seed),
    '--target-loss', String(o.targetLoss),
    '--order', o.order,
    '--vram-reserve-mb', String(o.vramReserveMb),
    // Always emitted, both sides: an ace-train that predates the flag rejects it
    // loudly rather than silently running the other precision.
    '--mirror', o.mirror,
    // Ditto: always emitted, so an ace-train that predates --bwd rejects it
    // loudly rather than silently running the slow out_prod backward.
    '--bwd', o.bwd,
    // Always emitted so an ace-train that predates --optimizer rejects it
    // loudly rather than silently training on AdamW when Muon was asked for.
    '--optimizer', o.optimizer,
    // Batching/checkpointing (design §2.2): always emitted on both sides —
    // an ace-train that predates the flags rejects them loudly rather than
    // silently training at batch 1 / no checkpointing.
    '--batch', String(o.batch),
    '--ckpt', String(o.ckptSegments),
    '--milestone-step', String(o.milestoneStep),
    '--milestone-keep', String(o.milestoneKeep),
    // Always emitted, both sides: an ace-train that predates --attn rejects it
    // loudly rather than silently running the other attention backend
    // (2026-09-01 flash-attn-backward plan §11).
    '--attn', o.attnBackend,
  ];
  // Flag-shaped options: only the non-default side is emitted (§2.1).
  // target-mlp is the exception — it is emitted on BOTH sides. Its default
  // flipped to ON, so "omit when false" would silently train the MLP anyway,
  // and the checkbox in TrainDitForm would be dead. Needs an ace-train that
  // knows --no-target-mlp (added alongside the default flip); an older binary
  // rejects the unknown option loudly rather than doing the wrong thing.
  // On resume it is an IDENTITY flag (changes which tensors exist) and is
  // suppressed like the others — the engine adopts it from the source log.
  if (!o.initAdapter) args.push(o.targetMlp ? '--target-mlp' : '--no-target-mlp');
  // Muon knobs only when Muon is actually selected — they are inert on the
  // AdamW path, and emitting them there would put noise in the recorded argv
  // of every run that never used them.
  if (o.optimizer === 'muon') {
    args.push('--muon-lr-scale', String(o.muonLrScale));
    args.push('--muon-momentum', String(o.muonMomentum));
    args.push('--muon-ns-steps', String(o.muonNsSteps));
    args.push('--muon-min-dim', String(o.muonMinDim));
  }
  if (!o.channelBalance) args.push('--no-channel-balance');
  if (o.overwrite) args.push('--overwrite');
  args.push('--jsonl');
  return args;
}
