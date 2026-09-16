// training/yue2Runs.ts — reading a finished (or half-finished) YuE2 NAR run
// off disk: what it was told to do, how far it got, and which adapters it left.
//
// The YuE2 analogue of mm3Runs.ts, and its header's claim holds here too: the
// run directory has always held everything needed to say what happened; what
// was missing was a way to ask. Three sources, none of them a database:
//
//   hotstep-run.json               written HERE, on every launch including a
//                                  resume. The recipe, the dataset it belongs
//                                  to, and how many times it has been trained.
//   train-log.jsonl                written by the runner from the trainer's
//                                  stderr (the YuE2 trainers emit no JSONL of
//                                  their own — see yue2TrainRunner.ts).
//   *.safetensors + their headers  the adapters themselves. Their
//                                  `__metadata__` is the engine's own record
//                                  of rank/alpha/target/steps/trigger, so a
//                                  file copied out of a run still describes
//                                  itself.
//
// THREE WAYS THIS DIFFERS FROM MM3's LADDER, all of them exporter shape rather
// than concept (11 §3.3):
//
//   * A checkpoint is a FILE, not a directory: `<stem>_step<N>.safetensors`
//     for the snapshots and `<stem>.safetensors` for the final export.
//   * There is no `resume-state.json` sidecar. The state is one opaque
//     `yue2_nar_ckpt.bin` (adapter + AdamW moments + step), so "how far is the
//     state behind the snapshots" is answered from the log, not from the file.
//   * That state file is DELETED on a clean finish — it is the run's middle,
//     not its result. Its presence therefore means "this run stopped early",
//     which is exactly when someone wants to continue it.

import fs from 'fs';
import path from 'path';

import {
  YUE2_ADAPTER_STEM, yue2AdapterRoot,
  type ResolvedYue2TrainOptions, type Yue2CacheSummary,
} from './yue2Train.js';
import type { TrainingAdapterHit } from './types.js';

export { yue2AdapterRoot };

export function yue2RunManifestPath(outDir: string): string {
  return path.join(outDir, 'hotstep-run.json');
}

/** The engine's resume state. Opaque to us; only its presence and mtime mean
 *  anything here. */
export function yue2ResumeStatePath(outDir: string): string {
  return path.join(outDir, 'yue2_nar_ckpt.bin');
}

function readJson<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf-8')) as T;
  } catch {
    return null;
  }
}

// ── the run's own manifest ──────────────────────────────────────────────────

export interface Yue2RunManifest {
  version: 1;
  runName: string;
  datasetId: string;
  datasetSlug: string;
  datasetName: string;
  createdAt: number;
  /** Every (re)start of this directory, including each resume. */
  updatedAt: number;
  launches: number;
  options: ResolvedYue2TrainOptions;
  /** The preprocess manifest's own summary at the time of the launch, so a
   *  run says how many clips it trained on without re-reading a cache that may
   *  since have been re-cut. */
  clips?: Yue2CacheSummary;
}

/** Record what a run was started with, so it can be continued with the same
 *  recipe. Called on EVERY launch including a resume, because a resume may
 *  have raised the step cap or reshaped the schedule and the next resume
 *  should pick up from the newer of the two. Never fails a run. */
export function writeYue2RunManifest(opts: ResolvedYue2TrainOptions,
                                     meta: { datasetId: string; datasetSlug: string; datasetName: string;
                                             clips?: Yue2RunManifest['clips'] }): void {
  try {
    const file = yue2RunManifestPath(opts.outDir);
    const prev = readJson<Yue2RunManifest>(file);
    // `resume` describes ONE launch, not the run. Persisting it would make
    // every later read think the run had always been a continuation.
    const { resume: _r, ...clean } = opts;
    const out: Yue2RunManifest = {
      version: 1,
      runName: path.basename(opts.outDir),
      datasetId: meta.datasetId,
      datasetSlug: meta.datasetSlug,
      datasetName: meta.datasetName,
      createdAt: prev?.createdAt ?? Date.now(),
      updatedAt: Date.now(),
      launches: (prev?.launches ?? 0) + 1,
      options: clean as ResolvedYue2TrainOptions,
      clips: meta.clips ?? prev?.clips,
    };
    fs.mkdirSync(opts.outDir, { recursive: true });
    fs.writeFileSync(file, JSON.stringify(out, null, 2), 'utf-8');
  } catch { /* a run must not fail because its description could not be saved */ }
}

export function readYue2RunManifest(dir: string): Yue2RunManifest | null {
  return readJson<Yue2RunManifest>(yue2RunManifestPath(dir));
}

// ── safetensors headers ─────────────────────────────────────────────────────

/** The `__metadata__` block of a safetensors file, as strings.
 *
 *  st_write_file writes every metadata VALUE as a string, including the
 *  numbers (yue2-nar-train-run.h's export note says so explicitly, and
 *  yue2_adapter_read_meta parses both spellings), so nothing here is coerced
 *  on the way in — the numeric views below are separate fields. */
export interface Yue2AdapterMeta {
  format?: string;
  rank?: number;
  alpha?: number;
  targets?: string;
  steps?: number;
  clipFrames?: number;
  trigger?: string;
  baseSha?: string;
  /** `upstream` or `bare`: which shape the training rows' style string had.
   *  Generation has to compose the prompt the same way or the trigger lands in
   *  a context the adapter never saw (backends/yue2/style.ts). Both exporters
   *  write it, so ABSENT means a file from before the flag existed — which is
   *  the `bare` behaviour, not the `upstream` default. */
  styleTemplate?: string;
  /** The share of artist rows trained with the caption dropped, leaving the
   *  trigger to stand alone — which is what says the adapter can be addressed
   *  by its trigger alone at generation time. Only the AR exporter writes it,
   *  and only when the run used it; a NAR run records it in its manifest
   *  instead. */
  captionDropout?: number;
  /** doc 19 decision 4: which chain-of-thought mode(s) this adapter trained,
   *  as the exporter wrote it — "off" (no --abc-dropout run, or every source
   *  lacked a lead sheet) or "off,full" (at least one cot=full draw was
   *  possible). Absent means a file exported before the flag existed, which
   *  is the "off" behaviour, not a default to fill in with "off,full". Both
   *  trainers' exporters write it identically (yue2-ar-train-run.h's
   *  yue2_at_export, yue2-nar-train-run.h's yue2_nt_export). */
  cot?: string;
  /** Everything the header carried, unparsed, for anything added later. */
  raw: Record<string, string>;
}

/** Read a safetensors header without loading the tensors.
 *
 *  Layout: 8 bytes little-endian u64 header length, then that many bytes of
 *  UTF-8 JSON. The JSON's keys are tensor names plus the optional
 *  `__metadata__` object. A 1 MiB ceiling on the header is a sanity guard, not
 *  a format rule: these adapters carry a couple of hundred entries, and a
 *  claimed length beyond that means the file is not what it says it is.
 *
 *  Returns null on ANY failure. A run listing must never throw because one
 *  file in it was half-written. */
export function readSafetensorsMeta(file: string): Yue2AdapterMeta | null {
  let fd: number | undefined;
  try {
    fd = fs.openSync(file, 'r');
    const len = Buffer.alloc(8);
    if (fs.readSync(fd, len, 0, 8, 0) !== 8) return null;
    const n = Number(len.readBigUInt64LE(0));
    if (!Number.isFinite(n) || n <= 0 || n > 1024 * 1024) return null;
    const buf = Buffer.alloc(n);
    if (fs.readSync(fd, buf, 0, n, 8) !== n) return null;
    const header = JSON.parse(buf.toString('utf-8')) as Record<string, unknown>;
    const md = (header.__metadata__ ?? {}) as Record<string, unknown>;
    const raw: Record<string, string> = {};
    for (const [k, v] of Object.entries(md)) raw[k] = typeof v === 'string' ? v : JSON.stringify(v);
    const num = (k: string): number | undefined => {
      const v = Number(raw[k]);
      return Number.isFinite(v) ? v : undefined;
    };
    return {
      format: raw.format,
      rank: num('rank'),
      alpha: num('alpha'),
      targets: raw.targets,
      steps: num('steps'),
      clipFrames: num('clip_frames'),
      trigger: raw.trigger,
      baseSha: raw.base_sha,
      styleTemplate: raw.style_template,
      captionDropout: num('caption_dropout'),
      cot: raw.cot,
      raw,
    };
  } catch {
    return null;
  } finally {
    if (fd !== undefined) { try { fs.closeSync(fd); } catch { /* ignore */ } }
  }
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

/** Whole-file parse of train-log.jsonl. Written by the runner, one JSON object
 *  per line, in the same shape mm3's engine emits — unknown `type` values are
 *  ignored so the vocabulary can grow without breaking old runs. */
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

export interface Yue2RunCheckpoint {
  /** The step the file was exported at. The FINAL export carries the run's
   *  configured step count, which is how it sorts to the end of the ladder. */
  step: number;
  /** File name, not a directory — `yue2_nar_lora_step4000.safetensors`. */
  name: string;
  /** Absolute path. This is what `/yue2/select-model`'s `lm_adapter` field
   *  takes: the engine opens the path AS GIVEN and does not resolve it against
   *  any adapter root (yue2-adapter.h), so an absolute path is required. */
  path: string;
  bytes: number;
  /** True for `<stem>.safetensors` — the end of the run, not a snapshot. */
  final: boolean;
  loss?: number;
  /** The file's own `__metadata__`, when it could be read. */
  meta?: Yue2AdapterMeta;
}

export type Yue2RunOutcome = 'completed' | 'halted' | 'failed' | 'unknown';

export interface Yue2RunSummary {
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
  outcome: Yue2RunOutcome;
  failure?: string;
  checkpoints: Yue2RunCheckpoint[];
  best?: { step: number; loss: number };
  /** The word the adapter is addressed by, from the manifest or from the
   *  newest checkpoint's own header. */
  trigger?: string;
  rank?: number;
  alpha?: number;
  target?: string;
  /** How this run's prompts were built, for the generation path and for the
   *  Studio to show. Off the newest checkpoint rather than the manifest,
   *  because the exporter writes it into every file: an adapter copied out of
   *  its run still says how its prompt was built. The dropout has no header to
   *  come from on this half (the NAR exporter does not write it), so the
   *  manifest's own recipe answers for it. */
  styleTemplate?: string;
  captionDropout?: number;
  /** Present only when `yue2_nar_ckpt.bin` is still there, which means the run
   *  stopped BEFORE its clean finish (the engine deletes it on export).
   *
   *  `behindBy` is honest about a gap the MM3 version also surfaces: the state
   *  is rewritten at every --save-every checkpoint, so a run killed between
   *  two of them comes back at the earlier one and repeats the steps since. */
  resume?: {
    step: number;
    savedAt: number;
    statePath: string;
    behindBy: number;
    /** The engine REFUSES a resume whose rank/alpha/target/grad-accum/frames/
     *  seed or whose trigger/lyrics/t-sampling/caption-dropout differ, and it
     *  is exact only within one machine and build — a different GPU, driver,
     *  ggml build, base GGUF or manifest breaks it undetectably. Carried so
     *  the UI can say so rather than promising more than the engine does. */
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

/** Every adapter file in a run directory, oldest step first.
 *
 *  Matches `<stem>_step<N>.safetensors` for the snapshots and `<stem>.safetensors`
 *  for the final export, where `<stem>` is fixed at YUE2_ADAPTER_STEM. A run
 *  trained by hand with a different `--name` will show no checkpoints rather
 *  than the wrong ones, which is the safer failure: the alternative is
 *  matching any `*.safetensors` and listing files the picker cannot load. */
export function yue2CheckpointsIn(dir: string, milestones: Map<number, number>,
                                  configuredSteps = 0): Yue2RunCheckpoint[] {
  const out: Yue2RunCheckpoint[] = [];
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  const snap = new RegExp(`^${YUE2_ADAPTER_STEM}_step(\\d+)\\.safetensors$`);
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
        meta: readSafetensorsMeta(full) ?? undefined,
      });
      continue;
    }
    if (e.name === `${YUE2_ADAPTER_STEM}.safetensors`) {
      const meta = readSafetensorsMeta(full) ?? undefined;
      // The final export records its own step count in its header; fall back
      // to what the run was configured for, and only then to Infinity-as-last
      // (a plain large number) so it still sorts to the end of the ladder.
      const step = meta?.steps ?? (configuredSteps > 0 ? configuredSteps : Number.MAX_SAFE_INTEGER);
      out.push({ step, name: e.name, path: full, bytes, final: true, meta });
    }
  }
  return out.sort((a, b) => (a.step - b.step) || (Number(a.final) - Number(b.final)));
}

/** Everything about one run directory. `dir` need not belong to any dataset. */
export function readYue2Run(dir: string): Yue2RunSummary | null {
  try {
    if (!fs.statSync(dir).isDirectory()) return null;
  } catch {
    return null;
  }
  const manifest = readYue2RunManifest(dir);
  const facts    = readLog(dir);
  const configuredSteps = manifest?.options.steps ?? facts.totalSteps ?? 0;
  const ckpts    = yue2CheckpointsIn(dir, facts.milestones, configuredSteps);
  const statePath = yue2ResumeStatePath(dir);
  let stateStat: fs.Stats | null = null;
  try { stateStat = fs.statSync(statePath); } catch { /* no state: the run finished cleanly, or never started */ }

  const lastCkptStep = ckpts.reduce((m, c) => (c.final ? m : Math.max(m, c.step)), 0);
  const lastStep = Math.max(facts.lastStep, lastCkptStep);

  let outcome: Yue2RunOutcome;
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

  // Newest checkpoint header is the fallback identity for a run with no
  // manifest — a directory trained from the command line still describes
  // itself, because the engine writes rank/alpha/target/trigger into every
  // file it exports.
  const newest = ckpts.length ? ckpts[ckpts.length - 1] : undefined;

  const summary: Yue2RunSummary = {
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
    captionDropout: manifest?.options.captionDropout ?? newest?.meta?.captionDropout,
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

/** Every run belonging to a dataset, newest first.
 *
 *  Attribution is by MANIFEST first and by the `<slug>-<timestamp>` directory
 *  name second, for mm3Runs' reason: the name alone would let two datasets
 *  whose slugs share a prefix collect each other's runs, and a directory
 *  trained from the CLI has nothing else to go on. */
export function listYue2Runs(datasetId: string, slug: string): Yue2RunSummary[] {
  const root = yue2AdapterRoot();
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const byName = new RegExp(`^${slug.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}-\\d{4}-\\d{2}-\\d{2}_`);
  const out: Yue2RunSummary[] = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const dir = path.join(root, e.name);
    const manifest = readYue2RunManifest(dir);
    const mine = manifest ? manifest.datasetId === datasetId : byName.test(e.name);
    if (!mine) continue;
    const run = readYue2Run(dir);
    if (run) out.push(run);
  }
  return out.sort((a, b) => b.updatedAt - a.updatedAt);
}

/** Every run under the adapter root, newest first, whatever dataset it came
 *  from — or none, for a directory trained from the command line.
 *
 *  listYue2Runs() answers the Training Studio's question ("what has THIS
 *  dataset produced"); this answers the generation picker's ("what could I
 *  load"). Both go through readYue2Run so there is exactly one scanner and one
 *  definition of what a checkpoint is. Never throws: a missing root is an
 *  empty list, which renders as "no adapters trained yet". */
export function listAllYue2Runs(): Yue2RunSummary[] {
  const root = yue2AdapterRoot();
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const out: Yue2RunSummary[] = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const run = readYue2Run(path.join(root, e.name));
    if (run) out.push(run);
  }
  return out.sort((a, b) => b.updatedAt - a.updatedAt);
}

// ── cheap per-dataset lookup, for the dataset list ──────────────────────────
//
// readYue2Run() above is right for a run's detail page — it parses the whole
// train-log.jsonl and walks the run's directory size, and yue2CheckpointsIn()
// reads every checkpoint's safetensors header on top of that. Asking for it
// once per dataset per list request does not scale. This answers a narrower
// question ("has this dataset got a trained NAR adapter, and when") from ONE
// readdir of the adapter root plus, per candidate run, a manifest read and a
// single stat on the final export — never the log, the header parse or the
// byte walk.

/** Every dataset's newest trained YuE2 NAR adapter, in ONE pass over the
 *  adapter root. Attribution mirrors listYue2Runs(): the run's own manifest
 *  first, then the `<slug>-YYYY-MM-DD_HH-MM-SS` directory-name fallback for
 *  runs trained before hotstep-run.json existed — replicated here so a
 *  pre-manifest run does not silently read back as "never trained". */
export function findYue2NarAdaptersFor(
  datasets: Array<{ id: string; slug: string }>,
): Map<string, TrainingAdapterHit> {
  const out = new Map<string, TrainingAdapterHit>();
  const root = yue2AdapterRoot();
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
    const manifest = readYue2RunManifest(dir);
    let datasetId = manifest?.datasetId && ids.has(manifest.datasetId) ? manifest.datasetId : '';
    if (!datasetId) {
      const match = bySlug.find(b => b.re.test(e.name));
      if (match) datasetId = match.id;
    }
    if (!datasetId) continue;
    let mtime: number;
    try {
      mtime = fs.statSync(path.join(dir, `${YUE2_ADAPTER_STEM}.safetensors`)).mtimeMs;
    } catch {
      continue;   // no final export in this run dir — not trained-to-completion
    }
    const prior = bestMtime.get(datasetId);
    if (prior !== undefined && prior >= mtime) continue;
    bestMtime.set(datasetId, mtime);
    out.set(datasetId, { path: dir, kind: 'yue2-nar', detail: '', trainedAt: new Date(mtime).toISOString() });
  }
  return out;
}

/** Single-dataset convenience over findYue2NarAdaptersFor(), for a caller
 *  that already has exactly one dataset in hand. */
export function findYue2NarAdapter(ds: { id: string; slug: string }): TrainingAdapterHit | null {
  return findYue2NarAdaptersFor([ds]).get(ds.id) ?? null;
}

/** Resolve one run by name, refusing anything that is not a direct child of
 *  the adapter root — the name arrives from a request body. */
export function resolveYue2RunDir(runName: string): string | null {
  if (!runName || /[\\/]/.test(runName) || runName === '.' || runName === '..') return null;
  const root = yue2AdapterRoot();
  const dir = path.join(root, runName);
  if (path.dirname(dir) !== root) return null;
  return fs.existsSync(dir) ? dir : null;
}

/** What a resume should replay. Null when the directory holds no manifest —
 *  unlike MM3 there is no log-based reconstruction, because the fields the
 *  engine fingerprints (rank, alpha, target, grad-accum, frames, seed,
 *  trigger, t-sampling, caption dropout) are not all recoverable from a
 *  checkpoint header, and a reconstruction that guessed one of them would be
 *  refused at spawn time with nothing useful to say. */
export function yue2ResumeOptionsFor(dir: string): ResolvedYue2TrainOptions | null {
  const manifest = readYue2RunManifest(dir);
  return manifest?.options ? { ...manifest.options, outDir: dir } : null;
}
