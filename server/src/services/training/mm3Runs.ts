// training/mm3Runs.ts — what is on disk from previous MM3 LM training runs, and
// what it would take to continue one.
//
// WHY THIS EXISTS
//
// A run directory has always held everything needed to carry on: the exported
// checkpoints, the optimizer state (`resume-state.bin`) and the full JSONL log.
// The engine has taken `--resume` since the preview loop was built. What was
// missing was any way to ASK for it: the state file was written only when a
// preview paused the run, nothing recorded what the run had been configured
// with, and no route listed the runs. So a run that stopped — finished, killed,
// crashed, or abandoned — was over, and "another 250 steps on that one" meant
// starting again from step 1 with a different adapter.
//
// Two files close that gap, and neither is the source of truth:
//
//   hotstep-run.json    written by the runner when a run starts: the resolved
//                       options, verbatim, plus which dataset they came from.
//                       This is what a resume replays.
//   resume-state.json   written by the ENGINE beside resume-state.bin every
//                       time it saves state, so the server can read "step 250
//                       of 250, saved on a clean exit" without opening a 4 GB
//                       binary or hard-coding its layout in TypeScript.
//
// Runs made before those files existed are still resumable: everything the
// engine FINGERPRINTS (rank, alpha, tensor count, optimizer, and the training/
// held-out split) is recoverable from the `init` line of train-log.jsonl, and
// anything that is not fingerprinted falls back to today's defaults. The
// listing says which of the two happened, because "resumed with the recipe it
// was trained with" and "resumed with today's defaults bolted on" are not the
// same claim.
//
// Nothing here trusts its own reconstruction over the engine: a mismatch it
// misses is refused by the fingerprint check in mm3-lm-resume.h, loudly, before
// a single step is trained.

import fs from 'fs';
import path from 'path';

import { config } from '../../config.js';
import { MM3_LM_DEFAULTS, type ResolvedMm3TrainLmOptions } from './mm3Train.js';
import type { TrainingAdapterHit } from './types.js';

/** Where every MM3 LM run lives. One level above a run directory. */
export function mm3AdapterRoot(): string {
  return path.join(config.aceServer.adapters, 'mm3-lm-adapters');
}

export function mm3RunManifestPath(outDir: string): string {
  return path.join(outDir, 'hotstep-run.json');
}

export interface Mm3RunManifest {
  version: 1;
  runName: string;
  datasetId: string;
  datasetSlug: string;
  datasetName: string;
  createdAt: number;
  /** Every time the run is (re)started, including each resume. */
  updatedAt: number;
  /** How many times this directory has been trained into. 1 = the original. */
  launches: number;
  options: ResolvedMm3TrainLmOptions;
  /** Which manifest row the artist preview caption came from on the most
   *  recent launch, and how it was picked. Absent when previews are off or the
   *  plan had nothing to render an artist take from. See mm3Preview.ts. */
  previewSong?: { id: string; filename: string; source: 'held' | 'train' | 'explicit' };
}

/** Record what a run was started with, so it can be continued with the same
 *  recipe. Called on every launch INCLUDING a resume, because a resume may have
 *  raised the step cap or switched the stopping strategy, and the next resume
 *  should pick up from the newer of the two. Never fails a run. */
export function writeMm3RunManifest(opts: ResolvedMm3TrainLmOptions,
                                    meta: { datasetId: string; datasetSlug: string;
                                            datasetName: string;
                                            previewSong?: Mm3RunManifest['previewSong'] }): void {
  try {
    const file = mm3RunManifestPath(opts.outDir);
    const prev = readJson<Mm3RunManifest>(file);
    // resumeFrom/resumeStep describe ONE launch, not the run. Persisting them
    // would make the next resume replay the last resume's starting point.
    const { resumeFrom: _rf, resumeStep: _rs, ...clean } = opts;
    const out: Mm3RunManifest = {
      version: 1,
      runName: path.basename(opts.outDir),
      datasetId: meta.datasetId,
      datasetSlug: meta.datasetSlug,
      datasetName: meta.datasetName,
      createdAt: prev?.createdAt ?? Date.now(),
      updatedAt: Date.now(),
      launches: (prev?.launches ?? 0) + 1,
      options: clean as ResolvedMm3TrainLmOptions,
      // Previews were computed AFTER this write on earlier launches too — a
      // resume that renders no new artist take (previews off this time)
      // should not erase what the last launch recorded.
      previewSong: meta.previewSong ?? prev?.previewSong,
    };
    fs.mkdirSync(opts.outDir, { recursive: true });
    fs.writeFileSync(file, JSON.stringify(out, null, 2), 'utf-8');
  } catch { /* a run must not fail because its description could not be saved */ }
}

// ── the engine's own state sidecar ──────────────────────────────────────────

export interface Mm3ResumeMeta {
  reason: 'pause' | 'final';
  state: string;
  step: number;
  totalSteps: number;
  epoch: number;
  lastLoss: number;
  meanLoss: number;
  bestEval: number;
  bestEvalStep: number;
  rank: number;
  alpha: number;
  seed: number;
  samples: number;
  holdout: number;
  optimizer: string;
  adapterType: string;
  savedAt: number;
}

function readJson<T>(file: string): T | null {
  try {
    if (!fs.existsSync(file)) return null;
    return JSON.parse(fs.readFileSync(file, 'utf-8')) as T;
  } catch {
    return null;
  }
}

// ── the run log, read for what the manifest does not have ───────────────────

interface LogFacts {
  init?: Record<string, unknown>;
  adapter?: Record<string, unknown>;
  /** Every `kind` the launch's adapter events reported, not just the first.
   *  rsLoRA is emitted as its OWN event alongside dora/hira/loha/hra, so the
   *  single first-wins `adapter` above can only ever describe one of the two —
   *  and a run whose parameterization is not reconstructed here comes back as a
   *  plain LoRA, which is a different adapter. */
  adapterKinds: Set<string>;
  cropAnchor?: string;
  cropPolicy?: Record<string, unknown>;
  kvPrefix?: Record<string, unknown>;
  /** 'exact' | 'flash' | 'flash-f32', from the engine's own `attn` event. */
  attn?: string;
  depthLoss?: Record<string, unknown>;
  targetLoss?: Record<string, unknown>;
  lastStep: number;
  lastLoss?: number;
  /** The step of the LAST pause, which is the step a pre-sidecar run's
   *  resume-state.bin actually holds. */
  pausedAt?: number;
  best?: { step: number; loss: number };
  milestones: Map<number, number>;   // step -> loss
  /** The last terminal-ish event the log carries. */
  ending: 'done' | 'target_stop' | 'paused' | 'fatal' | 'none';
  fatalMessage?: string;
}

/** Read a run's JSONL. Whole-file: a 4000-step run's log is about a megabyte,
 *  and the facts wanted live at both ends of it. */
function readLog(dir: string): LogFacts {
  const out: LogFacts = { lastStep: 0, milestones: new Map(), ending: 'none', adapterKinds: new Set() };
  let text: string;
  try {
    text = fs.readFileSync(path.join(dir, 'train-log.jsonl'), 'utf-8');
  } catch {
    return out;
  }
  for (const line of text.split(/\r?\n/)) {
    if (!line) continue;
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(line) as Record<string, unknown>;
    } catch {
      continue;
    }
    const num = (k: string): number | undefined => {
      const v = Number(ev[k]);
      return Number.isFinite(v) ? v : undefined;
    };
    // CONFIGURATION events are FIRST-WINS; progress events below are last-wins.
    //
    // The log is append-only across launches, so a resume writes a second
    // `init`. Taking the newest would mean reading the shape of the most recent
    // ATTEMPT — including one that was refused before it trained a step — when
    // what a resume needs is the shape the saved state was written under, which
    // is the original launch's. A failed resume that misread the split would
    // otherwise poison every later attempt with its own bad numbers.
    switch (ev.type) {
      case 'init':          out.init ??= ev; break;
      case 'adapter':
        out.adapter ??= ev;
        // Kinds accumulate rather than first-win: one launch can emit `rslora`
        // AND `dora`. Later launches append to the same file, but a resume runs
        // the same parameterization by construction (the engine now refuses a
        // mismatch), so the union is that one recipe.
        if (typeof ev.kind === 'string') out.adapterKinds.add(ev.kind);
        break;
      case 'cropAnchor':
        if (out.cropAnchor === undefined && typeof ev.mode === 'string') out.cropAnchor = ev.mode;
        break;
      case 'cropPolicy':    out.cropPolicy ??= ev; break;
      case 'kvPrefix':      out.kvPrefix ??= ev; break;
      // Written by mm3-lm-train-run.h once --attn is resolved (landed
      // 2026-09-05, same day this reader was taught to read it). First-wins,
      // same as cropAnchor/kvPrefix — a resume's second `attn` line describes
      // the shape a LATER attempt asked for, not the one the saved state was
      // written under.
      case 'attn':
        if (out.attn === undefined && typeof ev.mode === 'string') out.attn = ev.mode;
        break;
      case 'depthLossCfg':  out.depthLoss ??= ev; break;
      case 'targetLoss':    out.targetLoss ??= ev; break;
      case 'step':
        out.lastStep = Math.max(out.lastStep, num('step') ?? 0);
        out.lastLoss = num('loss') ?? out.lastLoss;
        break;
      case 'milestone': {
        const st = num('step'), ls = num('loss');
        if (st !== undefined) out.milestones.set(st, ls ?? 0);
        break;
      }
      case 'best': {
        const st = num('step'), ls = num('loss');
        if (st !== undefined && ls !== undefined) out.best = { step: st, loss: ls };
        break;
      }
      // A run can pause many times and finish once; the LAST of these wins,
      // which is why this is a straight assignment rather than a first-seen.
      case 'paused':
        out.ending = 'paused';
        out.pausedAt = num('step') ?? out.pausedAt;
        break;
      case 'target_stop':   out.ending = 'target_stop'; break;
      case 'done':          out.ending = 'done'; break;
      case 'fatal':
        out.ending = 'fatal';
        out.fatalMessage = typeof ev.message === 'string' ? ev.message : undefined;
        break;
      default: break;
    }
  }
  return out;
}

// ── the summary a UI can act on ─────────────────────────────────────────────

export interface Mm3RunCheckpoint {
  step: number;
  name: string;
  dir: string;
  loss?: number;
}

export type Mm3RunOutcome = 'completed' | 'target-reached' | 'halted' | 'failed' | 'unknown';

export interface Mm3RunSummary {
  runName: string;
  dir: string;
  datasetId?: string;
  datasetName?: string;
  startedAt?: number;
  updatedAt: number;
  launches: number;
  /** What the run was told to do — the cap, in either stopping mode. */
  configuredSteps: number;
  /** The furthest step the log saw. */
  lastStep: number;
  lastLoss?: number;
  outcome: Mm3RunOutcome;
  failure?: string;
  checkpoints: Mm3RunCheckpoint[];
  best?: { step: number; loss: number };
  targetLoss?: number;
  targetLossMetric?: string;
  /** Which manifest row the most recent artist preview was rendered from.
   *  Manifest-sourced only — a pre-manifest run does not carry this. */
  previewSong?: { id: string; filename: string; source: 'held' | 'train' | 'explicit' };
  /** Present only when this run can actually be continued. */
  resume?: {
    step: number;
    reason: 'pause' | 'final';
    savedAt: number;
    statePath: string;
    /** The state is BEHIND the checkpoints: it was written at the last preview
     *  pause and the run then ran on. Continuing repeats those steps. Only ever
     *  true for runs trained before the engine learned to save state on a clean
     *  exit, or for runs that were killed. */
    behindBy: number;
    rank: number;
    alpha: number;
    optimizer: string;
    adapterType: string;
    samples: number;
    holdout: number;
    bestEval?: number;
  };
  /** 'manifest' = the run's own recorded options. 'log' = reconstructed from
   *  train-log.jsonl with today's defaults filling the gaps. */
  optionsSource: 'manifest' | 'log' | 'none';
  sizeBytes: number;
  /** Set by the route when this run's job is the one running right now. */
  running?: boolean;
}

function dirSize(dir: string): number {
  let total = 0;
  try {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, e.name);
      try {
        if (e.isDirectory()) total += dirSize(full);
        else total += fs.statSync(full).size;
      } catch { /* vanished mid-walk */ }
    }
  } catch { /* unreadable */ }
  return total;
}

function checkpointsIn(dir: string, milestones: Map<number, number>): Mm3RunCheckpoint[] {
  const out: Mm3RunCheckpoint[] = [];
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const m = /^ckpt-(\d+)$/.exec(e.name);
    if (!m) continue;
    const full = path.join(dir, e.name);
    // Both parameterizations, because looking only for the PEFT name once
    // reported a perfectly good LoKr run as having no checkpoints.
    const real = fs.existsSync(path.join(full, 'adapter_model.safetensors'))
              || fs.existsSync(path.join(full, 'lokr_weights.safetensors'));
    if (!real) continue;
    const step = Number(m[1]);
    out.push({ step, name: e.name, dir: full, loss: milestones.get(step) });
  }
  return out.sort((a, b) => a.step - b.step);
}

/** The hold-out FRACTION that reproduces a known split of `train` + `held`.
 *
 *  The engine splits with `ceil(holdout * total)`, so any fraction inside
 *  ((h-1)/total, h/total] gives the same two counts — and those counts are what
 *  the resume fingerprint checks.
 *
 *  Take the MIDDLE of that interval, never its top edge. h/total is exactly the
 *  value ceil() sits on the boundary of, and the arithmetic does not stay on
 *  the boundary: the engine holds the fraction in a float, so 2/13 becomes
 *  0.15384615957, times 13 is 2.0000000745, and ceil gives 3. A run that held
 *  out 2 songs was then resumed as one holding out 3 and the engine refused it
 *  — correctly, over a difference this file had invented. The midpoint
 *  reproduces every split from 6 to 60 songs exactly; the top edge got 541 of
 *  them wrong. */
function holdoutFractionFor(trainSongs: number, heldOut: number): number {
  const total = trainSongs + heldOut;
  return total > 0 && heldOut > 0 ? (heldOut - 0.5) / total : 0;
}

/** Rebuild the options of a run that predates hotstep-run.json.
 *
 *  Only the FINGERPRINTED fields matter for the resume to be accepted — rank,
 *  alpha, optimizer, adapter parameterization (they decide the tensor count)
 *  and the training/held-out split. Those all come from the log. The rest falls
 *  back to today's defaults, which is why the listing reports this as a
 *  reconstruction rather than as the run's recipe. */
function optionsFromLog(dir: string, facts: LogFacts): ResolvedMm3TrainLmOptions | null {
  const init = facts.init;
  if (!init) return null;
  const D = MM3_LM_DEFAULTS;
  const n = (v: unknown, d: number): number => {
    const x = Number(v);
    return Number.isFinite(x) ? x : d;
  };
  const trainSongs = n(init.samples, 0);
  const heldOut    = n(init.holdout, 0);
  const holdout = holdoutFractionFor(trainSongs, heldOut);
  const adapterType = facts.adapter?.kind === 'lokr' ? 'lokr' : 'lora';
  const optimizer = init.optimizer === 'muon' || init.optimizer === 'prodigy'
    || init.optimizer === 'adamw' ? init.optimizer : D.optimizer;
  return {
    manifest: '', captionsDir: '', codesDir: '',   // filled in by the caller
    outDir: dir,
    rank: n(init.rank, D.rank),
    alpha: n(init.alpha, D.alpha),
    lr: D.lr,
    steps: n(init.totalSteps, D.steps),
    saveEvery: D.saveEvery,
    warmup: D.warmup,
    gradAccum: D.gradAccum,
    seed: D.seed,
    maxFrames: n(init.maxFrames, D.maxFrames),
    cropMode: (facts.cropPolicy?.mode as ResolvedMm3TrainLmOptions['cropMode']) ?? D.cropMode,
    cropStartFrac: n(facts.cropPolicy?.startFrac, D.cropStartFrac),
    cropEndFrac: n(facts.cropPolicy?.endFrac, D.cropEndFrac),
    cropStartTiles: D.cropStartTiles,
    depthLossWeight: n(facts.depthLoss?.weight, D.depthLossWeight),
    depthLossFrames: n(facts.depthLoss?.frames, D.depthLossFrames),
    optimizer,
    muonLrScale: n(init.lrScale, D.muonLrScale),
    holdout,
    evalEvery: D.evalEvery,
    evalCrop: D.evalCrop,
    rankDropout: D.rankDropout,
    adapterType,
    lokrFactor: n(facts.adapter?.factor, D.lokrFactor),
    lokrDim: n(facts.adapter?.dim, D.lokrDim),
    lokrAlpha: n(facts.adapter?.alpha, D.lokrAlpha),
    // attn and the parameterization are both fingerprinted by the engine's own
    // events (`attn`, and `adapter` with a per-method `kind`), so a run with no
    // manifest still resumes as the adapter it is. Reconstructing these as "off"
    // was not a smaller lie than reconstructing rank wrong: a DoRA run continued
    // without --dora has no magnitudes to restore, and an rsLoRA run continued
    // without --rslora simply gets quieter by sqrt(r).
    attnBackend: facts.attn === 'flash' || facts.attn === 'flash-f32' ? 'flash' : D.attnBackend,
    rslora: facts.adapterKinds.has('rslora') || D.rslora,
    dora: facts.adapterKinds.has('dora'),
    hira: facts.adapterKinds.has('hira'),
    loha: facts.adapterKinds.has('loha'),
    pissa: facts.adapterKinds.has('pissa') || facts.adapterKinds.has('hot-pizza') || facts.adapterKinds.has('hot-pissa'),
    hotPizza: facts.adapterKinds.has('hot-pizza') || facts.adapterKinds.has('hot-pissa'),
    pissaCache: D.pissaCache,
    pissaFrozenF16: false,
    hra: facts.adapterKinds.has('hra'),
    loraPlusRatio: D.loraPlusRatio,
    artistToken: '',
    artistTokenK: D.artistTokenK,
    artistTokenLr: D.artistTokenLr,
    prefixN: D.prefixN,
    trigger: '',
    triggerPrepend: D.triggerPrepend,
    datasetName: '',
    basePrecision: D.basePrecision,
    cropAnchor: facts.cropAnchor === 'zero' ? 'zero' : 'song',
    prefixFrames: n(facts.kvPrefix?.frames, 0),
    prefixChunk: n(facts.kvPrefix?.chunk, D.prefixChunk),
    prefixSelftest: false,
    lrEndFrac: D.lrEndFrac,
    stopMode: facts.targetLoss ? 'loss' : 'steps',
    targetLoss: n(facts.targetLoss?.target, 0),
    targetLossMetric: facts.targetLoss?.metric === 'eval' ? 'eval' : 'train',
    targetLossEpochs: n(facts.targetLoss?.epochs, D.targetLossEpochs),
  };
}

/** Everything about one run directory. `dir` need not belong to any dataset. */
export function readMm3Run(dir: string): Mm3RunSummary | null {
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return null;
  const manifest = readJson<Mm3RunManifest>(mm3RunManifestPath(dir));
  const facts    = readLog(dir);
  const state    = readJson<Mm3ResumeMeta>(path.join(dir, 'resume-state.json'));
  const ckpts    = checkpointsIn(dir, facts.milestones);
  const statePath = path.join(dir, 'resume-state.bin');
  const hasState  = fs.existsSync(statePath);

  const configuredSteps = manifest?.options.steps
    ?? (facts.init ? Number(facts.init.totalSteps) || 0 : 0);
  const lastStep = Math.max(facts.lastStep, state?.step ?? 0,
                            ckpts.length ? ckpts[ckpts.length - 1].step : 0);

  let outcome: Mm3RunOutcome;
  switch (facts.ending) {
    case 'done':        outcome = 'completed'; break;
    case 'target_stop': outcome = 'target-reached'; break;
    case 'fatal':       outcome = 'failed'; break;
    case 'paused':      outcome = 'halted'; break;
    default:            outcome = facts.lastStep > 0 ? 'halted' : 'unknown'; break;
  }

  let stat: fs.Stats | null = null;
  try { stat = fs.statSync(dir); } catch { /* ignore */ }

  const summary: Mm3RunSummary = {
    runName: path.basename(dir),
    dir,
    datasetId: manifest?.datasetId,
    datasetName: manifest?.datasetName,
    startedAt: manifest?.createdAt ?? stat?.birthtimeMs ?? undefined,
    updatedAt: manifest?.updatedAt ?? stat?.mtimeMs ?? 0,
    launches: manifest?.launches ?? 1,
    configuredSteps,
    lastStep,
    lastLoss: facts.lastLoss,
    outcome,
    failure: facts.fatalMessage,
    checkpoints: ckpts,
    best: facts.best,
    targetLoss: manifest?.options.stopMode === 'loss'
      ? manifest.options.targetLoss
      : (facts.targetLoss ? Number(facts.targetLoss.target) : undefined),
    targetLossMetric: manifest?.options.stopMode === 'loss'
      ? manifest.options.targetLossMetric
      : (facts.targetLoss?.metric as string | undefined),
    previewSong: manifest?.previewSong,
    optionsSource: manifest ? 'manifest' : (facts.init ? 'log' : 'none'),
    sizeBytes: dirSize(dir),
  };

  if (hasState && state && state.step > 0) {
    summary.resume = {
      step: state.step,
      reason: state.reason === 'final' ? 'final' : 'pause',
      savedAt: state.savedAt,
      statePath,
      behindBy: Math.max(0, lastStep - state.step),
      rank: state.rank,
      alpha: state.alpha,
      optimizer: state.optimizer,
      adapterType: state.adapterType,
      samples: state.samples,
      holdout: state.holdout,
      bestEval: state.bestEval >= 0 ? state.bestEval : undefined,
    };
  } else if (hasState && facts.init) {
    // A run from before the engine wrote the readable sidecar. Its state file
    // was written by a PREVIEW PAUSE, so the log's last `paused` event names the
    // step it holds — which is very often behind the run's last step, and saying
    // so is the whole point of `behindBy`. Without a pause event in the log there
    // is nothing honest to quote, so the run is reported as unresumable rather
    // than resumable from a number that was guessed.
    const known = facts.pausedAt ?? 0;
    if (known <= 0) return summary;
    summary.resume = {
      step: known,
      reason: 'pause',
      savedAt: (() => { try { return fs.statSync(statePath).mtimeMs; } catch { return 0; } })(),
      statePath,
      behindBy: Math.max(0, lastStep - known),
      rank: Number(facts.init.rank) || 0,
      alpha: Number(facts.init.alpha) || 0,
      optimizer: String(facts.init.optimizer || ''),
      adapterType: facts.adapter?.kind === 'lokr' ? 'lokr' : 'lora',
      samples: Number(facts.init.samples) || 0,
      holdout: Number(facts.init.holdout) || 0,
    };
  }
  return summary;
}

/** Every run belonging to a dataset, newest first.
 *
 *  Attribution is by MANIFEST first and by the `<slug>-<timestamp>` directory
 *  name second. The name alone is not enough: two datasets whose slugs share a
 *  prefix ("albumB" and "albumB-live") would collect each other's runs, and
 *  a run started before the manifest existed has nothing else to go on. */
export function listMm3Runs(datasetId: string, slug: string): Mm3RunSummary[] {
  const root = mm3AdapterRoot();
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const out: Mm3RunSummary[] = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const dir = path.join(root, e.name);
    const manifest = readJson<Mm3RunManifest>(mm3RunManifestPath(dir));
    const mine = manifest
      ? manifest.datasetId === datasetId
      : new RegExp(`^${slug.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-\\d{4}-\\d{2}-\\d{2}_`).test(e.name);
    if (!mine) continue;
    const run = readMm3Run(dir);
    if (run) out.push(run);
  }
  return out.sort((a, b) => b.updatedAt - a.updatedAt);
}

/** The run's own recorded recipe, if it has one. */
export function readMm3RunManifest(dir: string): Mm3RunManifest | null {
  return readJson<Mm3RunManifest>(mm3RunManifestPath(dir));
}

// ── cheap per-dataset lookup, for the dataset list ──────────────────────────
//
// readMm3Run() above is right for one run's detail page — it parses the whole
// train-log.jsonl and walks the run's directory size. Asking it once per
// dataset per list request does not scale. This section answers a narrower
// question ("has this dataset got a trained LM adapter, and when") from ONE
// readdir of the adapter root plus, per candidate run, a manifest read and a
// stat — nothing that opens train-log.jsonl or sums bytes.

/** The newest exported checkpoint's weights file in a run directory: the
 *  highest `ckpt-<N>` that actually holds weights. One readdir of the run
 *  dir, one stat — never a log parse. Mirrors what mm3TrainRunner.ts's
 *  `finalCkpt` resolves to for the "last checkpoint any segment exported". */
function newestMm3CheckpointWeights(dir: string): { path: string; mtime: number } | null {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return null;
  }
  let bestStep = -1;
  let bestName = '';
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const m = /^ckpt-(\d+)$/.exec(e.name);
    if (!m) continue;
    const step = Number(m[1]);
    if (step > bestStep) { bestStep = step; bestName = e.name; }
  }
  if (!bestName) return null;
  const ckptDir = path.join(dir, bestName);
  const mtime = weightsMtimeInDir(ckptDir);
  return mtime ? { path: ckptDir, mtime } : null;
}

function weightsMtimeInDir(dir: string): number {
  for (const f of ['adapter_model.safetensors', 'lokr_weights.safetensors']) {
    try { return fs.statSync(path.join(dir, f)).mtimeMs; } catch { /* next */ }
  }
  return 0;
}

/** Every dataset's newest trained MM3 LM adapter, in ONE pass over the
 *  adapter root. Attribution mirrors listMm3Runs(): the run's own manifest
 *  first, then the `<slug>-YYYY-MM-DD_HH-MM-SS` directory-name fallback for
 *  runs trained before hotstep-run.json existed — replicated here so a
 *  pre-manifest run does not silently read back as "never trained". */
export function findMm3LmAdaptersFor(
  datasets: Array<{ id: string; slug: string }>,
): Map<string, TrainingAdapterHit> {
  const out = new Map<string, TrainingAdapterHit>();
  const root = mm3AdapterRoot();
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return out;
  }
  const ids = new Set(datasets.map(d => d.id));
  const bySlug = datasets.map(ds => ({
    id: ds.id,
    re: new RegExp(`^${ds.slug.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-\\d{4}-\\d{2}-\\d{2}_`),
  }));
  const bestMtime = new Map<string, number>();
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const dir = path.join(root, e.name);
    const manifest = readMm3RunManifest(dir);
    let datasetId = manifest?.datasetId && ids.has(manifest.datasetId) ? manifest.datasetId : '';
    if (!datasetId) {
      const match = bySlug.find(b => b.re.test(e.name));
      if (match) datasetId = match.id;
    }
    if (!datasetId) continue;
    const weights = newestMm3CheckpointWeights(dir);
    if (!weights) continue;
    const prior = bestMtime.get(datasetId);
    if (prior !== undefined && prior >= weights.mtime) continue;
    bestMtime.set(datasetId, weights.mtime);
    out.set(datasetId, {
      path: dir,
      kind: 'mm3-lm',
      detail: '',
      trainedAt: new Date(weights.mtime).toISOString(),
    });
  }
  return out;
}

/** Single-dataset convenience over findMm3LmAdaptersFor(), for a caller (the
 *  dataset detail endpoint) that already has exactly one dataset in hand and
 *  would otherwise have to build a one-item array itself. */
export function findMm3LmAdapter(ds: { id: string; slug: string }): TrainingAdapterHit | null {
  return findMm3LmAdaptersFor([ds]).get(ds.id) ?? null;
}

/** What a resume should replay: the manifest when the run recorded one, and
 *  otherwise a reconstruction from its log. Null when the directory holds
 *  neither — there is then no honest way to continue it. */
export function resumeOptionsFor(dir: string): ResolvedMm3TrainLmOptions | null {
  const facts = readLog(dir);
  const manifest = readMm3RunManifest(dir);
  const opts = manifest?.options ? { ...manifest.options } : optionsFromLog(dir, facts);
  if (!opts) return null;

  // The hold-out fraction is re-derived from the counts the run ACTUALLY
  // trained under, even when a manifest states one, because the manifest can be
  // wrong about it in a way nothing else is: a resume writes a manifest before
  // it launches, so a resume that computed the fraction badly persisted that
  // mistake and every later attempt inherited it. The log's `init` line is the
  // engine's own report of the split the state file was written against, so it
  // is the better authority here.
  //
  // This does NOT paper over a dataset that has genuinely changed: the fraction
  // reproduces the OLD counts, applying it to a new song total yields a
  // different split, and the fingerprint refuses the resume — which is the
  // correct outcome.
  //
  // Authority order: the state's own sidecar first (it reports the split stored
  // IN the state file), then the log's first `init`.
  const meta = readJson<Mm3ResumeMeta>(path.join(dir, 'resume-state.json'));
  const trainSongs = Number(meta?.samples ?? facts.init?.samples);
  const heldOut    = Number(meta?.holdout ?? facts.init?.holdout);
  if (Number.isFinite(trainSongs) && Number.isFinite(heldOut) && trainSongs + heldOut > 0) {
    opts.holdout = holdoutFractionFor(trainSongs, heldOut);
  }
  return opts;
}

/** Resolve one run by name, refusing anything that is not a direct child of the
 *  adapter root — the name arrives from a request body. */
export function resolveMm3RunDir(runName: string): string | null {
  if (!runName || /[\\/]/.test(runName) || runName === '.' || runName === '..') return null;
  const dir = path.join(mm3AdapterRoot(), runName);
  if (path.dirname(dir) !== mm3AdapterRoot()) return null;
  return fs.existsSync(dir) ? dir : null;
}
