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
}

/** Record what a run was started with, so it can be continued with the same
 *  recipe. Called on every launch INCLUDING a resume, because a resume may have
 *  raised the step cap or switched the stopping strategy, and the next resume
 *  should pick up from the newer of the two. Never fails a run. */
export function writeMm3RunManifest(opts: ResolvedMm3TrainLmOptions,
                                    meta: { datasetId: string; datasetSlug: string;
                                            datasetName: string }): void {
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
  cropAnchor?: string;
  cropPolicy?: Record<string, unknown>;
  kvPrefix?: Record<string, unknown>;
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
  const out: LogFacts = { lastStep: 0, milestones: new Map(), ending: 'none' };
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
      case 'adapter':       out.adapter ??= ev; break;
      case 'cropAnchor':
        if (out.cropAnchor === undefined && typeof ev.mode === 'string') out.cropAnchor = ev.mode;
        break;
      case 'cropPolicy':    out.cropPolicy ??= ev; break;
      case 'kvPrefix':      out.kvPrefix ??= ev; break;
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
 *  prefix ("greenday" and "greenday-live") would collect each other's runs, and
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
