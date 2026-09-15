// training/yue2ArRuns.ts — reading a finished (or half-finished) YuE2 AR run
// off disk. The AR twin of yue2Runs.ts, and the three sources are the same
// three: hotstep-run.json written here on every launch, train-log.jsonl written
// by the runner from the trainer's stderr, and the `__metadata__` block of each
// exported safetensors. There is still no database.
//
// It is a separate file rather than a `kind` parameter on yue2Runs because the
// two halves share a directory layout and nothing else. The run manifests carry
// different option types, the adapters carry a different `format` — and that
// difference is load-bearing, not cosmetic: the engine's loader gates the
// tensor-name family on `format`, so an AR adapter offered where a NAR one is
// wanted is refused rather than half-applied (yue2-ar-train-run.h §"THE LOADER
// SIDE"). Two scanners keyed on two stems is how that stays true here too.
//
// FOUR THINGS THAT DIFFER FROM THE NAR RUN:
//
//   * The stem is YUE2_AR_ADAPTER_STEM and the resume state is
//     `yue2_ar_ckpt.bin`. Both halves may be trained from the same dataset, so
//     a scanner that matched either stem would list the other's adapters as
//     loadable and the loader would reject them at generation time.
//   * The ladder starts at `--ckpt-from`, not at zero: the lowest rung on disk
//     is the first multiple of --save-every at or after it. Nothing here
//     assumes a rung exists at any particular step.
//   * The export carries `style_template` and, when the run used it,
//     `caption_dropout`. Generation has to compose the prompt the same way the
//     training rows were composed, so both are promoted out of `raw`.
//   * `song_frames`, where the NAR exporter writes `clip_frames` — an AR row is
//     a whole song, not a 10 s crop.

import fs from 'fs';
import path from 'path';

import {
  YUE2_AR_ADAPTER_STEM, yue2ArAdapterRoot,
  type ResolvedYue2ArTrainOptions,
} from './yue2ArTrain.js';
import { readSafetensorsMeta, type Yue2AdapterMeta } from './yue2Runs.js';
import { type Yue2CacheSummary } from './yue2Train.js';
import type { TrainingAdapterHit } from './types.js';

export { yue2ArAdapterRoot };

export function yue2ArRunManifestPath(outDir: string): string {
  return path.join(outDir, 'hotstep-run.json');
}

/** The engine's resume state. Opaque to us; only its presence and mtime mean
 *  anything here. Deleted on a clean finish — it is the run's middle, not its
 *  result — so finding one means the run stopped early. */
export function yue2ArResumeStatePath(outDir: string): string {
  return path.join(outDir, 'yue2_ar_ckpt.bin');
}

function readJson<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf-8')) as T;
  } catch {
    return null;
  }
}

// ── the run's own manifest ──────────────────────────────────────────────────

export interface Yue2ArRunManifest {
  version: 1;
  runName: string;
  datasetId: string;
  datasetSlug: string;
  datasetName: string;
  createdAt: number;
  /** Every (re)start of this directory, including each resume. */
  updatedAt: number;
  launches: number;
  options: ResolvedYue2ArTrainOptions;
  /** The preprocess manifest's own summary at the time of the launch. The AR
   *  trainer reads the SAME `yue2_preprocess.json` the NAR one does, after
   *  yue2-tokenize and yue2-align have added their stages to it, so this is the
   *  latents summary and says nothing about whether codes/ and cursor/ were
   *  present — the run's own log does. */
  clips?: Yue2CacheSummary;
}

/** Record what a run was started with, so it can be continued with the same
 *  recipe. Called on EVERY launch including a resume. Never fails a run. */
export function writeYue2ArRunManifest(opts: ResolvedYue2ArTrainOptions,
                                       meta: { datasetId: string; datasetSlug: string; datasetName: string;
                                               clips?: Yue2ArRunManifest['clips'] }): void {
  try {
    const file = yue2ArRunManifestPath(opts.outDir);
    const prev = readJson<Yue2ArRunManifest>(file);
    // `resume` describes ONE launch, not the run. Persisting it would make
    // every later read think the run had always been a continuation.
    const { resume: _r, ...clean } = opts;
    const out: Yue2ArRunManifest = {
      version: 1,
      runName: path.basename(opts.outDir),
      datasetId: meta.datasetId,
      datasetSlug: meta.datasetSlug,
      datasetName: meta.datasetName,
      createdAt: prev?.createdAt ?? Date.now(),
      updatedAt: Date.now(),
      launches: (prev?.launches ?? 0) + 1,
      options: clean as ResolvedYue2ArTrainOptions,
      clips: meta.clips ?? prev?.clips,
    };
    fs.mkdirSync(opts.outDir, { recursive: true });
    fs.writeFileSync(file, JSON.stringify(out, null, 2), 'utf-8');
  } catch { /* a run must not fail because its description could not be saved */ }
}

export function readYue2ArRunManifest(dir: string): Yue2ArRunManifest | null {
  return readJson<Yue2ArRunManifest>(yue2ArRunManifestPath(dir));
}

// ── safetensors headers ─────────────────────────────────────────────────────

/** An AR adapter's `__metadata__`, as strings, with the AR-only keys promoted.
 *
 *  `format` is "yue2-ar-lora-v1" here where a NAR adapter says
 *  "yue2-nar-lora-v1"; everything the exporter writes and this interface does
 *  not name (`cursor`, `minted`, `cot`, `adam_betas`, `overtrain`) is still in
 *  `raw`. */
export interface Yue2ArAdapterMeta extends Yue2AdapterMeta {
  /** How the training rows' style string was built. Generation has to compose
   *  the prompt the same way or the trigger lands in a context the adapter
   *  never saw. */
  styleTemplate?: string;
  /** Present only when the run trained with dropout, because that is the only
   *  case in which the exporter writes it — and it is what says the adapter can
   *  be addressed by its trigger phrase alone. */
  captionDropout?: number;
  /** The AR analogue of the NAR header's `clip_frames`: mean frames per
   *  SONG, not per crop. The inherited `clipFrames` is always undefined here. */
  songFrames?: number;
}

/** Read an AR adapter's header. readSafetensorsMeta() does the file format —
 *  it is format-agnostic and shared, not re-implemented — and this adds the
 *  three fields the AR exporter writes on top of it.
 *
 *  Returns null on ANY failure, for its reason: a listing must not throw
 *  because one file in it was half-written. */
export function readYue2ArAdapterMeta(file: string): Yue2ArAdapterMeta | null {
  const base = readSafetensorsMeta(file);
  if (!base) return null;
  const dropout = Number(base.raw.caption_dropout);
  const frames  = Number(base.raw.song_frames);
  return {
    ...base,
    styleTemplate: base.raw.style_template,
    captionDropout: Number.isFinite(dropout) ? dropout : undefined,
    songFrames: Number.isFinite(frames) ? frames : undefined,
  };
}

// ── the run's log ───────────────────────────────────────────────────────────

interface LogFacts {
  init?: Record<string, unknown>;
  lastStep: number;
  lastLoss?: number;
  totalSteps: number;
  /** step -> loss, for the checkpoint ladder. */
  milestones: Map<number, number>;
  best?: { step: number; loss: number };
  ending?: 'done' | 'fatal';
  fatalMessage?: string;
  /** The final line's mean loss, when the run reached one. */
  meanLoss?: number;
}

/** Whole-file parse of train-log.jsonl, one JSON object per line. The runner
 *  emits the same vocabulary for both YuE2 trainers, and an unknown `type` is
 *  ignored so it can grow. */
function readLog(dir: string): LogFacts {
  const facts: LogFacts = { lastStep: 0, totalSteps: 0, milestones: new Map() };
  let text = '';
  try {
    text = fs.readFileSync(path.join(dir, 'train-log.jsonl'), 'utf-8');
  } catch {
    return facts;
  }
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    let ev: Record<string, unknown>;
    try { ev = JSON.parse(line) as Record<string, unknown>; } catch { continue; }
    const num = (k: string): number | undefined => {
      const v = Number(ev[k]);
      return Number.isFinite(v) ? v : undefined;
    };
    switch (ev.type) {
      case 'init':
        facts.init = ev;
        facts.totalSteps = num('totalSteps') ?? facts.totalSteps;
        break;
      case 'step': {
        const s = num('step');
        if (s !== undefined) facts.lastStep = Math.max(facts.lastStep, s);
        const loss = num('loss');
        if (loss !== undefined) {
          facts.lastLoss = loss;
          if (!facts.best || loss < facts.best.loss) facts.best = { step: s ?? facts.lastStep, loss };
        }
        facts.totalSteps = num('totalSteps') ?? facts.totalSteps;
        break;
      }
      case 'milestone': {
        const s = num('step');
        if (s !== undefined) {
          facts.milestones.set(s, num('loss') ?? NaN);
          facts.lastStep = Math.max(facts.lastStep, s);
        }
        break;
      }
      case 'fatal':
        facts.ending = 'fatal';
        facts.fatalMessage = typeof ev.message === 'string' ? ev.message : 'ace-train reported a fatal error';
        break;
      case 'done':
        facts.ending = 'done';
        facts.meanLoss = num('meanLoss');
        facts.lastStep = Math.max(facts.lastStep, num('steps') ?? 0);
        break;
      default:
        break;
    }
  }
  return facts;
}

// ── run summaries ───────────────────────────────────────────────────────────

export interface Yue2ArRunCheckpoint {
  /** The step the file was exported at. The FINAL export carries the run's
   *  configured step count, which is how it sorts to the end of the ladder. */
  step: number;
  /** File name, not a directory — `yue2_ar_lora_step300.safetensors`. */
  name: string;
  /** Absolute path, which is what the engine's adapter field takes: it opens
   *  the path AS GIVEN and resolves it against no adapter root. */
  path: string;
  bytes: number;
  /** True for `<stem>.safetensors` — the end of the run, not a snapshot. */
  final: boolean;
  loss?: number;
  meta?: Yue2ArAdapterMeta;
}

export type Yue2ArRunOutcome = 'completed' | 'halted' | 'failed' | 'unknown';

export interface Yue2ArRunSummary {
  runName: string;
  dir: string;
  datasetId?: string;
  datasetName?: string;
  startedAt?: number;
  updatedAt: number;
  launches: number;
  /** What the run was told to do. */
  configuredSteps: number;
  /** The furthest step anything on disk saw. */
  lastStep: number;
  lastLoss?: number;
  outcome: Yue2ArRunOutcome;
  failure?: string;
  checkpoints: Yue2ArRunCheckpoint[];
  best?: { step: number; loss: number };
  /** The word the adapter is addressed by, from the manifest or from the
   *  newest checkpoint's own header. */
  trigger?: string;
  rank?: number;
  alpha?: number;
  target?: string;
  /** Read off the newest checkpoint rather than the manifest, because the
   *  exporter writes both into every file: an adapter copied out of its run
   *  still says how its prompt was built, and a run with no checkpoints yet has
   *  nothing to generate from anyway. */
  styleTemplate?: string;
  captionDropout?: number;
  /** Present only when `yue2_ar_ckpt.bin` is still there, which means the run
   *  stopped BEFORE its clean finish.
   *
   *  `behindBy` is the gap the state cannot close: it is rewritten only at each
   *  --save-every checkpoint, so a run killed between two of them comes back at
   *  the earlier one and repeats the steps since. */
  resume?: {
    step: number;
    savedAt: number;
    statePath: string;
    behindBy: number;
    /** The engine REFUSES a resume whose rank/alpha/target/grad-accum/seed
     *  differ, and equally one whose conditioning moved — manifest, minted
     *  pack, trigger, style template, lyrics, sidecars, artist-frac, max-len or
     *  attention kernel are hashed together. A changed schedule is allowed with
     *  a NOTE, and even then the run is no longer bit-identical to the
     *  uninterrupted one. Exact only within one machine and build. */
    exactWithinBuildOnly: true;
  };
  /** 'manifest' = the run's own recorded recipe. 'checkpoint' = read back off
   *  an adapter's safetensors header. 'none' = neither was available. */
  optionsSource: 'manifest' | 'checkpoint' | 'none';
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

/** Every AR adapter file in a run directory, oldest step first.
 *
 *  Matches `<stem>_step<N>.safetensors` and `<stem>.safetensors` against
 *  YUE2_AR_ADAPTER_STEM only. A NAR run trained from the same dataset can share
 *  this root, and listing its files here would offer the picker adapters the
 *  loader refuses on `format`. */
export function yue2ArCheckpointsIn(dir: string, milestones: Map<number, number>,
                                    configuredSteps = 0): Yue2ArRunCheckpoint[] {
  const out: Yue2ArRunCheckpoint[] = [];
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  const snap = new RegExp(`^${YUE2_AR_ADAPTER_STEM}_step(\\d+)\\.safetensors$`);
  for (const e of entries) {
    if (!e.isFile()) continue;
    const full = path.join(dir, e.name);
    let bytes = 0;
    try { bytes = fs.statSync(full).size; } catch { continue; }
    const m = snap.exec(e.name);
    if (m) {
      const step = Number(m[1]);
      const loss = milestones.get(step);
      out.push({
        step, name: e.name, path: full, bytes, final: false,
        loss: Number.isFinite(loss as number) ? loss : undefined,
        meta: readYue2ArAdapterMeta(full) ?? undefined,
      });
      continue;
    }
    if (e.name === `${YUE2_AR_ADAPTER_STEM}.safetensors`) {
      const meta = readYue2ArAdapterMeta(full) ?? undefined;
      // The final export records its own step count in its header; fall back to
      // what the run was configured for, and only then to a plain large number
      // so it still sorts to the end of the ladder.
      const step = meta?.steps ?? (configuredSteps > 0 ? configuredSteps : Number.MAX_SAFE_INTEGER);
      out.push({ step, name: e.name, path: full, bytes, final: true, meta });
    }
  }
  return out.sort((a, b) => (a.step - b.step) || (Number(a.final) - Number(b.final)));
}

/** Everything about one AR run directory. `dir` need not belong to any dataset. */
export function readYue2ArRun(dir: string): Yue2ArRunSummary | null {
  try {
    if (!fs.statSync(dir).isDirectory()) return null;
  } catch {
    return null;
  }
  const manifest = readYue2ArRunManifest(dir);
  const facts    = readLog(dir);
  const configuredSteps = manifest?.options.steps ?? facts.totalSteps ?? 0;
  const ckpts    = yue2ArCheckpointsIn(dir, facts.milestones, configuredSteps);
  const statePath = yue2ArResumeStatePath(dir);
  let stateStat: fs.Stats | null = null;
  try { stateStat = fs.statSync(statePath); } catch { /* no state: finished cleanly, or never started */ }

  const lastCkptStep = ckpts.reduce((m, c) => (c.final ? m : Math.max(m, c.step)), 0);
  const lastStep = Math.max(facts.lastStep, lastCkptStep);

  let outcome: Yue2ArRunOutcome;
  switch (facts.ending) {
    case 'done':  outcome = 'completed'; break;
    case 'fatal': outcome = 'failed'; break;
    // No log ending and a final export on disk still means the run finished —
    // the log can be missing entirely for a run trained from the CLI.
    default:      outcome = ckpts.some(c => c.final) ? 'completed'
                          : (lastStep > 0 ? 'halted' : 'unknown'); break;
  }

  let stat: fs.Stats | null = null;
  try { stat = fs.statSync(dir); } catch { /* ignore */ }

  const newest = ckpts.length ? ckpts[ckpts.length - 1] : undefined;

  const summary: Yue2ArRunSummary = {
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
    trigger: manifest?.options.trigger || newest?.meta?.trigger || undefined,
    rank: manifest?.options.rank ?? newest?.meta?.rank,
    alpha: manifest?.options.alpha ?? newest?.meta?.alpha,
    target: manifest?.options.target ?? newest?.meta?.targets,
    styleTemplate: newest?.meta?.styleTemplate,
    captionDropout: newest?.meta?.captionDropout,
    optionsSource: manifest ? 'manifest' : (newest?.meta ? 'checkpoint' : 'none'),
    sizeBytes: dirSize(dir),
  };

  if (stateStat) {
    // The state is rewritten at every --save-every checkpoint, so the step it
    // holds is the last SNAPSHOT step, not the last step the log saw.
    const known = lastCkptStep;
    summary.resume = {
      step: known,
      savedAt: stateStat.mtimeMs,
      statePath,
      behindBy: Math.max(0, lastStep - known),
      exactWithinBuildOnly: true,
    };
  }
  return summary;
}

/** Every AR run belonging to a dataset, newest first.
 *
 *  Attribution is by MANIFEST first and by the `<slug>-<timestamp>` directory
 *  name second: the name alone would let two datasets whose slugs share a
 *  prefix collect each other's runs, and a directory trained from the CLI has
 *  nothing else to go on. */
export function listYue2ArRuns(datasetId: string, slug: string): Yue2ArRunSummary[] {
  const root = yue2ArAdapterRoot();
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const byName = new RegExp(`^${slug.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-\\d{4}-\\d{2}-\\d{2}_`);
  const out: Yue2ArRunSummary[] = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const dir = path.join(root, e.name);
    const manifest = readYue2ArRunManifest(dir);
    const mine = manifest ? manifest.datasetId === datasetId : byName.test(e.name);
    if (!mine) continue;
    const run = readYue2ArRun(dir);
    if (run) out.push(run);
  }
  return out.sort((a, b) => b.updatedAt - a.updatedAt);
}

/** Every AR run under the adapter root, newest first, whatever dataset it came
 *  from — or none, for a directory trained from the command line.
 *
 *  listYue2ArRuns() answers the Training Studio's question ("what has THIS
 *  dataset produced"); this answers the generation picker's ("what could I
 *  load"). Both go through readYue2ArRun so there is one scanner and one
 *  definition of what a checkpoint is. Never throws. */
export function listAllYue2ArRuns(): Yue2ArRunSummary[] {
  const root = yue2ArAdapterRoot();
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const out: Yue2ArRunSummary[] = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const run = readYue2ArRun(path.join(root, e.name));
    if (run) out.push(run);
  }
  return out.sort((a, b) => b.updatedAt - a.updatedAt);
}

// ── cheap per-dataset lookup, for the dataset list ──────────────────────────
//
// readYue2ArRun() above is right for a run's detail page — full log parse,
// header reads, directory-size walk. This answers a narrower question ("has
// this dataset got a trained AR adapter, and when") from ONE readdir of the
// adapter root plus, per candidate run, a manifest read and a single stat on
// the final export.

/** Every dataset's newest trained YuE2 AR adapter, in ONE pass over the
 *  adapter root. Attribution mirrors listYue2ArRuns(): the run's own manifest
 *  first, then the `<slug>-YYYY-MM-DD_HH-MM-SS` directory-name fallback for
 *  runs trained before hotstep-run.json existed — replicated here so a
 *  pre-manifest run does not silently read back as "never trained". */
export function findYue2ArAdaptersFor(
  datasets: Array<{ id: string; slug: string }>,
): Map<string, TrainingAdapterHit> {
  const out = new Map<string, TrainingAdapterHit>();
  const root = yue2ArAdapterRoot();
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
    const manifest = readYue2ArRunManifest(dir);
    let datasetId = manifest?.datasetId && ids.has(manifest.datasetId) ? manifest.datasetId : '';
    if (!datasetId) {
      const match = bySlug.find(b => b.re.test(e.name));
      if (match) datasetId = match.id;
    }
    if (!datasetId) continue;
    let mtime: number;
    try {
      mtime = fs.statSync(path.join(dir, `${YUE2_AR_ADAPTER_STEM}.safetensors`)).mtimeMs;
    } catch {
      continue;   // no final export in this run dir — not trained-to-completion
    }
    const prior = bestMtime.get(datasetId);
    if (prior !== undefined && prior >= mtime) continue;
    bestMtime.set(datasetId, mtime);
    out.set(datasetId, { path: dir, kind: 'yue2-ar', detail: '', trainedAt: new Date(mtime).toISOString() });
  }
  return out;
}

/** Single-dataset convenience over findYue2ArAdaptersFor(), for a caller that
 *  already has exactly one dataset in hand. */
export function findYue2ArAdapter(ds: { id: string; slug: string }): TrainingAdapterHit | null {
  return findYue2ArAdaptersFor([ds]).get(ds.id) ?? null;
}

/** Resolve one AR run by name, refusing anything that is not a direct child of
 *  the adapter root — the name arrives from a request body. */
export function resolveYue2ArRunDir(runName: string): string | null {
  if (!runName || /[\\/]/.test(runName) || runName === '.' || runName === '..') return null;
  const root = yue2ArAdapterRoot();
  const dir = path.join(root, runName);
  if (path.dirname(dir) !== root) return null;
  return fs.existsSync(dir) ? dir : null;
}

/** What a resume should replay. Null without a manifest: the engine hashes the
 *  conditioning (manifest, minted pack, trigger, style template, lyrics,
 *  sidecars, artist-frac, max-len, attention) into the state and refuses a
 *  mismatch, and none of that is recoverable from a checkpoint header — a
 *  reconstruction that guessed one of them would be refused at spawn time with
 *  nothing useful to say. */
export function yue2ArResumeOptionsFor(dir: string): ResolvedYue2ArTrainOptions | null {
  const manifest = readYue2ArRunManifest(dir);
  return manifest?.options ? { ...manifest.options, outDir: dir } : null;
}
