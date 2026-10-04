// Durable catalogue for native AITK YuE2 joint runs. The native trainer owns
// its output directory; this index only records paths after that directory has
// appeared, so a failed launch cannot reserve or overwrite a user directory.

import fs from 'fs';
import path from 'path';
import { createHash } from 'crypto';
import { trainingBaseDir } from './paths.js';
import { archiveYue2TrainLogs } from './datasetProfile.js';
import { runStamp } from './adapterLayout.js';
import { config } from '../../config.js';
import { listDatasets, updateDataset } from './datasetsRepo.js';
import { uniqueDatasetTrigger } from './datasetTrigger.js';

export function yue2JointOutputDirectory(adaptersRoot: string, trigger: string, when = new Date()): string {
  const name = trigger.trim().replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^[. ]+|[. ]+$/g, '').slice(0, 120) || 'dataset';
  return path.join(adaptersRoot, 'yue2-joint-adapters', `${name}_${runStamp(when)}`);
}

/** One chain of work per run, so a pull, a checkpoint fetch and a folder
 *  move of the same run never interleave: a move mid-pull would let the pull
 *  recreate the old folder and point the index back at it. In-process only,
 *  which is why the remote-folder migration runs inside the app. The map
 *  entry is dropped once nothing is waiting. */
const runLocks = new Map<string, Promise<unknown>>();
export function withYue2RunLock<T>(jobId: string, fn: () => Promise<T>): Promise<T> {
  const prior = runLocks.get(jobId) ?? Promise.resolve();
  const next = prior.then(fn, fn);
  const settled = next.then(() => {}, () => {});
  runLocks.set(jobId, settled);
  void settled.finally(() => { if (runLocks.get(jobId) === settled) runLocks.delete(jobId); });
  return next;
}

/** Pulled ladders wait here, out of the local run list's way, until a rung is
 *  chosen: `yue2-joint-adapters/_remote/<worker>/<trigger>_<stamp>`. */
export const YUE2_REMOTE_STAGING = '_remote';

/** Where a run pulled from `worker` belongs: the staging folder while it is
 *  under review, then the same `<trigger>_<stamp>` name a local run gets once
 *  finished. The stamp is the run's start time on the worker. */
export function yue2RemoteRunDirectory(adaptersRoot: string, trigger: string, worker: string, createdAt: number, finished: boolean): string {
  const local = yue2JointOutputDirectory(adaptersRoot, trigger, new Date(createdAt));
  if (finished) return local;
  const folder = worker.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '_').replace(/^[._]+|[._]+$/g, '') || 'worker';
  return path.join(path.dirname(local), YUE2_REMOTE_STAGING, folder, path.basename(local));
}

/** `wanted`, or `wanted-2`, `wanted-3`... when another run already owns it
 *  (on disk, or in `taken`, which a dry run uses to see its own claims). */
export function freeYue2RunDirectory(wanted: string, taken?: Set<string>): string {
  for (let n = 1; ; n++) {
    const candidate = n === 1 ? wanted : `${wanted}-${n}`;
    if (!fs.existsSync(candidate) && !taken?.has(path.resolve(candidate).toLowerCase())) return candidate;
  }
}

const INDEX = path.join(trainingBaseDir, 'yue2-aitk-runs.json');
const MAX_RECORDS = 256;
const MAX_INDEX_BYTES = 4 * 1024 * 1024;
const warnedRunFiles = new Set<string>();

function warnRunFile(output: string, err: unknown): void {
  if (warnedRunFiles.has(output)) return;
  warnedRunFiles.add(output);
  console.warn(`[YuE2] Could not write run.json in ${output}: ${String(err)}`);
}

export interface Yue2AitkCheckpointRecord {
  step: number;
  dir: string;
  loss?: number;
  /** From the checkpoint's meters.json: the planner's KL stop reading (or
   *  its 20-step mean), the decoder's reconstruction meter, frozen state. */
  kl?: number;
  recon?: number;
  drift?: number;
  frozen?: boolean;
  /** Written at a KL rung (--kl-checkpoint-every), not a routine save. */
  rung?: boolean;
  adapterPath?: string;
  optimizerPath?: string;
  arPath?: string;
  narPath?: string;
  /** Which `segments/segment-NNNNNN` folder this checkpoint lives under, for
   *  a parallel/sharded run. Absent for a checkpoint directly under `output`. */
  segment?: string;
  /** 'remote': this step's meters came from a training worker's ladder pull
   *  (yue2LadderPull.ts) and no weight file is on this disk yet — it can be
   *  listed, previewed and scored, but not linked or used for NAR follow-up
   *  until its checkpoint is fetched (hydrateYue2LadderCheckpoint). Absent
   *  (the default) means a real local checkpoint: `checkpointRecords` found
   *  it on disk. */
  availability?: 'remote';
  /** sha256 of every file the worker's manifest declared for this remote-
   *  origin checkpoint, written by hydrateYue2LadderCheckpoint only once ALL
   *  of them are verified on this disk — a partial transfer writes nothing,
   *  so presence means complete. Once the worker's own copy is gone (deleted
   *  after a successful link) this is what a later use revalidates against.
   *  Keyed by filename (LADDER_CHECKPOINT_FILES), never derived from a disk scan. */
  manifestSha256?: Record<string, string>;
  /** Optional files of that manifest this machine deleted on purpose
   *  (cleanup's resume prune), so revalidation stops demanding them and a
   *  repair never fetches them back. Never includes the ar/nar weights. */
  prunedFiles?: string[];
}

export interface Yue2AitkRunRecord {
  version: 1;
  jobId: string;
  datasetId: string;
  datasetSlug: string;
  method: 'aitk';
  output: string;
  options: Record<string, unknown>;
  /** 'interrupted': the index said running when a fresh server started, so
   *  the process that ran it is gone (an app restart or crash mid-run). */
  status: 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';
  createdAt: number;
  updatedAt: number;
  error?: string;
  checkpoints: Yue2AitkCheckpointRecord[];
  /** Stable, server-assigned labels keyed by canonical checkpoint step. */
  blindLabels?: Record<string, string>;
  /** Set once by a ladder pull: this run was trained on `worker`, not here.
   *  `remoteJobId` is the worker's own jobId for the same run, needed to ask
   *  it for anything not yet local (a chosen rung's weights, in a later
   *  slice). Never set for a run trained on this machine. */
  origin?: { worker: string; remoteJobId: string };
}

function readIndex(): Yue2AitkRunRecord[] {
  try {
    const stat = fs.statSync(INDEX);
    if (!stat.isFile() || stat.size > MAX_INDEX_BYTES) return [];
    const value = JSON.parse(fs.readFileSync(INDEX, 'utf8')) as unknown;
    if (!Array.isArray(value)) return [];
    const records = value.filter(isRunRecord);
    // A run folder moved by hand into refined/ (the cleanup's own destination)
    // is found again here, so every consumer of the index (the adapter
    // catalogue, the caption-source picker, jointRunForAdapter) keeps working
    // without the user re-registering anything. The fix is written back once.
    let relocated = false;
    const healed = records.map(r => {
      const moved = relocatedOutput(r.output);
      if (!moved) return r;
      relocated = true;
      return { ...r, output: moved, updatedAt: Date.now() };
    });
    if (relocated) { try { writeIndex(healed, healed.filter((r, i) => r !== records[i])); } catch { /* next read heals again */ } }
    return healed;
  } catch { return []; }
}

/** Where a recorded run folder went if it is gone from its recorded path:
 *  `<parent>/refined/<name>`, the only place the app itself moves runs to. */
function relocatedOutput(output: string): string | null {
  try {
    if (fs.existsSync(output)) return null;
    const candidate = path.join(path.dirname(output), 'refined', path.basename(output));
    return fs.existsSync(candidate) && fs.statSync(candidate).isDirectory() ? candidate : null;
  } catch { return null; }
}

function isRunRecord(value: unknown): value is Yue2AitkRunRecord {
  if (!value || typeof value !== 'object') return false;
  const r = value as Yue2AitkRunRecord;
  return r.version === 1 && typeof r.jobId === 'string' && r.jobId.length <= 128
    && typeof r.datasetId === 'string' && typeof r.datasetSlug === 'string'
    && r.method === 'aitk' && typeof r.output === 'string' && r.output.length <= 32768
    && ['running', 'done', 'failed', 'cancelled', 'interrupted'].includes(r.status)
    && Number.isFinite(r.createdAt) && Number.isFinite(r.updatedAt)
    && (r.blindLabels === undefined || (r.blindLabels !== null && typeof r.blindLabels === 'object'
      && Object.entries(r.blindLabels).every(([step, label]) => /^\d+$/.test(step) && typeof label === 'string' && /^[A-Z]+$/.test(label))))
    && (r.origin === undefined || (!!r.origin && typeof r.origin === 'object'
      && typeof r.origin.worker === 'string' && r.origin.worker.length <= 128
      && typeof r.origin.remoteJobId === 'string' && r.origin.remoteJobId.length <= 128))
    && Array.isArray(r.checkpoints) && r.checkpoints.length <= 1024
    && r.checkpoints.every(c => !!c && Number.isInteger(c.step) && c.step >= 0
      && typeof c.dir === 'string' && c.dir.length <= 32768
      && (c.loss === undefined || (typeof c.loss === 'number' && Number.isFinite(c.loss)))
      && (c.segment === undefined || (typeof c.segment === 'string' && c.segment.length <= 64))
      && (c.availability === undefined || c.availability === 'remote')
      && ['adapterPath', 'optimizerPath', 'arPath', 'narPath'].every(k => {
        const v = c[k as keyof Yue2AitkCheckpointRecord];
        return v === undefined || (typeof v === 'string' && v.length <= 32768);
      }));
}

/** A checkpoint step known only from a worker's ladder pull (no local weight
 *  file) is kept alongside whatever `checkpointRecords` actually finds on
 *  this disk, never in place of it — a step this machine has for real (local
 *  training, or a later-slice hydration) always wins over its remote-only
 *  shadow. */
function mergeCheckpoints(local: Yue2AitkCheckpointRecord[], incoming: Yue2AitkCheckpointRecord[]): Yue2AitkCheckpointRecord[] {
  if (!incoming.length) return local;
  const incomingByStep = new Map(incoming.map(c => [c.step, c]));
  // manifestSha256/prunedFiles are never derivable from the disk scan that
  // produced `local` — carry them over from whatever was already persisted
  // for this step unless `local` brings its own, or they would vanish on
  // every subsequent pull/list.
  const merged = local.map(c => {
    const prior = incomingByStep.get(c.step);
    const manifestSha256 = c.manifestSha256 ?? prior?.manifestSha256, prunedFiles = c.prunedFiles ?? prior?.prunedFiles;
    return { ...c, ...(manifestSha256 && { manifestSha256 }), ...(prunedFiles && { prunedFiles }) };
  });
  const localSteps = new Set(local.map(c => c.step));
  const remoteOnly = incoming.filter(c => c.availability === 'remote' && !localSteps.has(c.step));
  return [...merged, ...remoteOnly].sort((a, b) => b.step - a.step);
}

function writeIndex(records: Yue2AitkRunRecord[], changedRecords: Yue2AitkRunRecord[]): void {
  fs.mkdirSync(path.dirname(INDEX), { recursive: true });
  const tmp = `${INDEX}.${process.pid}.${Date.now()}.tmp`;
  const persisted = records.slice(-MAX_RECORDS);
  fs.writeFileSync(tmp, JSON.stringify(persisted, null, 2), 'utf8');
  fs.renameSync(tmp, INDEX);
  const changed = new Set(changedRecords);
  for (const record of persisted.filter(r => changed.has(r))) {
    try { fs.writeFileSync(path.join(record.output, 'run.json'), JSON.stringify(record, null, 2), 'utf8'); }
    catch (err) { warnRunFile(record.output, err); }
  }
}

export function aitkRunIndexPath(): string { return INDEX; }

export function checkpointRecords(output: string): Yue2AitkCheckpointRecord[] {
  let entries: fs.Dirent[];
  try { entries = fs.readdirSync(output, { withFileTypes: true }); } catch { return []; }
  const rows: Yue2AitkCheckpointRecord[] = [];
  const dirs = [output];
  const segments = entries.find(e => e.isDirectory() && e.name === 'segments');
  if (segments) {
    try { for (const e of fs.readdirSync(path.join(output, 'segments'), { withFileTypes: true })) {
      if (e.isDirectory() && /^segment-\d{6}$/.test(e.name)) dirs.push(path.join(output, 'segments', e.name));
    } } catch { /* incomplete catalogue is handled by the caller */ }
  }
  for (const base of dirs) {
   const losses = new Map<number, number>();
   try {
     const log = path.join(base, 'train.jsonl');
     if (fs.statSync(log).size <= 8 * 1024 * 1024) {
       for (const line of fs.readFileSync(log, 'utf8').split(/\r?\n/)) {
         try {
           const event = JSON.parse(line) as Record<string, unknown>;
           if (event.stage !== 'joint' || !Number.isInteger(event.step)) continue;
           const { ar_ce, ar_kl, nar_mse, cursor_ce, cursor_weight } = event;
           if (![ar_ce, ar_kl, nar_mse].every(v => typeof v === 'number' && Number.isFinite(v))) continue;
           const cursor = typeof cursor_ce === 'number' && Number.isFinite(cursor_ce)
             && typeof cursor_weight === 'number' && Number.isFinite(cursor_weight)
             ? cursor_ce * cursor_weight : 0;
           losses.set(event.step as number, (ar_ce as number) + 0.2 * (ar_kl as number) + (nar_mse as number) + cursor);
         } catch { /* an incomplete log line does not invalidate other steps */ }
       }
     }
   } catch { /* an unfinished segment may not have its JSONL log yet */ }
   let local: fs.Dirent[];
   try { local = fs.readdirSync(base, { withFileTypes: true }); } catch { continue; }
   for (const e of local) {
    const match = e.isDirectory() ? /^checkpoint-step(\d+)$/.exec(e.name) : null;
    if (!match) continue;
    const dir = path.join(base, e.name);
    const file = (name: string): string | undefined => {
      const candidate = path.join(dir, name);
      return fs.existsSync(candidate) && fs.statSync(candidate).isFile() ? candidate : undefined;
    };
    let meters: Record<string, unknown> = {};
    try { meters = JSON.parse(fs.readFileSync(path.join(dir, 'meters.json'), 'utf8')) as Record<string, unknown>; } catch { /* older checkpoints have none */ }
    const num = (v: unknown) => typeof v === 'number' && Number.isFinite(v) ? v : undefined;
    const kl = num(meters.kl_reading) ?? num(meters.ar_kl_mean20);
    const step = Number(match[1]);
    const recent = [...losses].filter(([s]) => s <= step)
      .sort(([a], [b]) => b - a).slice(0, 20);
    const meanLoss = recent.length
      ? recent.reduce((sum, [, loss]) => sum + loss, 0) / recent.length : undefined;
    rows.push({
      step, dir,
      ...(meanLoss !== undefined ? { loss: meanLoss } : {}),
      ...(kl !== undefined ? { kl } : {}),
      ...(num(meters.nar_recon) !== undefined ? { recon: num(meters.nar_recon) } : {}),
      ...(num(meters.nar_drift) !== undefined ? { drift: num(meters.nar_drift) } : {}),
      ...(meters.planner_frozen === true ? { frozen: true } : {}),
      ...(meters.kl_rung === true ? { rung: true } : {}),
      adapterPath: file('adapter.safetensors'), optimizerPath: file('optimizer.resume'),
      arPath: file('native-ar.safetensors'), narPath: file('native-nar.safetensors'),
    });
   }
  }
  return rows.sort((a, b) => b.step - a.step);
}

export function recordYue2AitkRun(record: Yue2AitkRunRecord): void {
  try {
    const index = readIndex();
    const prior = index.find(r => r.jobId === record.jobId);
    // A repull must never let the worker's facts overwrite labels this
    // machine already assigned (or inherited on first pull) — prior wins
    // whenever it has any, remote/incoming only seeds a brand new record.
    // manifestSha256/prunedFiles are likewise carried from whatever this
    // machine already persisted for a step, since a caller recording fresh
    // pull data has no reason to know about them.
    const incoming = prior ? mergeCheckpoints(record.checkpoints, prior.checkpoints) : record.checkpoints;
    const updated = { ...record,
      blindLabels: prior?.blindLabels ?? record.blindLabels,
      checkpoints: mergeCheckpoints(checkpointRecords(record.output), incoming) };
    writeIndex([...index.filter(r => r.jobId !== record.jobId), updated], [updated]);
  } catch { /* a catalogue failure must never change the training result */ }
}

/** Remove a run from the catalogue and its output directory from disk. The
 *  caller has checked it is not live. Refuses paths outside the joint adapters
 *  tree as a guard against a corrupted index entry. */
export function deleteYue2AitkRun(jobId: string): { output: string } {
  const run = readIndex().find(r => r.jobId === jobId);
  if (!run) throw new Error('Unknown run');
  const output = path.resolve(run.output);
  if (!/yue2-joint-adapters/i.test(output)) throw new Error(`Refusing to delete outside the joint adapters folder: ${output}`);
  try { archiveYue2TrainLogs(run.datasetSlug, run.jobId, output); }
  catch (err: any) { throw new Error(`Could not archive the loss log of run ${jobId}, refusing to delete: ${err?.message || err}`); }
  fs.rmSync(output, { recursive: true, force: true });
  writeIndex(readIndex().filter(r => r.jobId !== jobId), []);
  return { output };
}

/** Rename a run's output directory and repoint the durable index at the new
 *  path, checkpoint paths included. Refuses to land on an existing path so a
 *  move never silently merges two runs; if the index cannot be written the
 *  folder is renamed back, so a failed move leaves everything where it was. */
export function moveYue2AitkRun(jobId: string, newOutput: string): void {
  const index = readIndex();
  const run = index.find(r => r.jobId === jobId);
  if (!run) throw new Error('Unknown run');
  const oldOutput = path.resolve(run.output);
  const target = path.resolve(newOutput);
  if (fs.existsSync(target)) throw new Error(`Refusing to move onto an existing path: ${target}`);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.renameSync(oldOutput, target);
  const prefix = oldOutput.toLowerCase() + path.sep;
  const rebase = <T extends string | undefined>(p: T): T =>
    (p && path.resolve(p).toLowerCase().startsWith(prefix) ? path.join(target, path.resolve(p).slice(prefix.length)) : p) as T;
  const moved: Yue2AitkRunRecord = { ...run, output: target, updatedAt: Date.now(),
    checkpoints: run.checkpoints.map(c => ({ ...c, dir: rebase(c.dir), adapterPath: rebase(c.adapterPath),
      optimizerPath: rebase(c.optimizerPath), arPath: rebase(c.arPath), narPath: rebase(c.narPath) })) };
  try { writeIndex(index.map(r => r.jobId === jobId ? moved : r), [moved]); }
  catch (err) { fs.renameSync(target, oldOutput); throw err; }
}

const skippedFolders = new Set<string>();
let lastFolderSet: string | null = null;
let lastDatasetSet: string | null = null;

function migrateSharedTriggers(): void {
  let datasets: ReturnType<typeof listDatasets>;
  try { datasets = listDatasets(); }
  catch (err) {
    if (err instanceof Error && err.message === 'Database not initialized. Call initDb() first.') return;
    throw err;
  }
  const groups = new Map<string, typeof datasets>();
  const taken = new Set(datasets.map(ds => ds.customTag.toLowerCase()));
  for (const ds of datasets) {
    if (!ds.customTag) continue;
    const key = ds.customTag.toLowerCase();
    groups.set(key, [...(groups.get(key) ?? []), ds]);
  }
  for (const group of groups.values()) {
    if (group.length < 2) continue;
    const ordered = group.sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id));
    const owner = ordered.find(ds => ds.slug === ds.customTag) ?? ordered[0];
    for (const ds of ordered) {
      if (ds.id === owner.id) continue;
      const trigger = uniqueDatasetTrigger(ds.customTag, ds.slug, taken);
      updateDataset(ds.id, { customTag: trigger });
      taken.add(trigger.toLowerCase());
    }
  }
}

function datasetSetForSkipped(): string | null {
  if (!skippedFolders.size) return null;
  try { return listDatasets().map(ds => `${ds.id}:${ds.slug}:${ds.customTag}`).join('\n'); }
  catch { return null; }
}

function jointFolders(): string[] | null {
  const root = path.join(config.aceServer.adapters, 'yue2-joint-adapters');
  try {
    const direct = fs.readdirSync(root, { withFileTypes: true });
    const refined = direct.some(e => e.isDirectory() && e.name === 'refined')
      ? fs.readdirSync(path.join(root, 'refined'), { withFileTypes: true }) : [];
    return [
      // _remote holds pulled ladders, each found through the index, never imported as a run itself.
      ...direct.filter(e => e.isDirectory() && e.name !== 'refined' && e.name !== YUE2_REMOTE_STAGING).map(e => path.join(root, e.name)),
      ...refined.filter(e => e.isDirectory()).map(e => path.join(root, 'refined', e.name)),
    ].sort();
  } catch { return null; } // An unavailable adapter drive must not erase the index.
}

function inferredDataset(output: string): { id: string; slug: string } | null {
  const match = /^(.*)_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d$/.exec(path.basename(output));
  if (!match) return null;
  const datasets = listDatasets();
  const exact = datasets.find(ds => ds.slug === match[1]);
  if (exact) return { id: exact.id, slug: exact.slug };
  const matches = datasets.filter(ds => {
    const trigger = ds.customTag || ds.slug;
    const folder = path.basename(yue2JointOutputDirectory(config.aceServer.adapters, trigger));
    return folder.slice(0, -20) === match[1];
  });
  const owner = matches.length === 1 ? matches[0] : null;
  return owner ? { id: owner.id, slug: owner.slug } : null;
}

function importedRun(output: string): Yue2AitkRunRecord | null {
  const file = path.join(output, 'run.json');
  if (fs.existsSync(file)) {
    try {
      const saved = JSON.parse(fs.readFileSync(file, 'utf8')) as unknown;
      if (!isRunRecord(saved)) throw new Error('invalid run record');
      return { ...saved, output, status: saved.status === 'running' || saved.status === 'interrupted' ? 'done' : saved.status,
        checkpoints: checkpointRecords(output) };
    } catch (err) {
      if (!skippedFolders.has(output)) console.warn(`[YuE2] Skipping ${output}: ${String(err)}`);
      skippedFolders.add(output);
      return null;
    }
  }
  const dataset = inferredDataset(output);
  if (!dataset) {
    if (!skippedFolders.has(output)) console.warn(`[YuE2] Skipping unrecognised joint run folder ${output}`);
    skippedFolders.add(output);
    return null;
  }
  const checkpoints = checkpointRecords(output);
  if (!checkpoints.some(c => c.arPath && c.narPath)) return null;
  const stamp = fs.statSync(output).mtimeMs;
  return { version: 1, jobId: `import-${createHash('sha256').update(path.resolve(output).toLowerCase()).digest('hex').slice(0, 24)}`,
    datasetId: dataset.id, datasetSlug: dataset.slug, method: 'aitk', output,
    options: checkpoints.some(c => c.rung) ? {} : { method: 'base-matched' }, status: 'done',
    createdAt: stamp, updatedAt: stamp, checkpoints };
}

function reconcileFromDisk(force = false): void {
  migrateSharedTriggers();
  const folders = jointFolders();
  if (!folders) return;
  for (const skipped of skippedFolders) if (!folders.includes(skipped)) skippedFolders.delete(skipped);
  const folderSet = folders.join('\n');
  const datasetSet = datasetSetForSkipped();
  if (!force && folderSet === lastFolderSet && datasetSet === lastDatasetSet) return;
  const index = readIndex();
  const onDisk = new Set(folders.map(p => path.resolve(p).toLowerCase()));
  const retained = index.filter(r => r.status === 'running' || r.status === 'interrupted'
    || onDisk.has(path.resolve(r.output).toLowerCase()) || fs.existsSync(r.output));
  const known = new Set(retained.map(r => path.resolve(r.output).toLowerCase()));
  const jobs = new Set(retained.map(r => r.jobId));
  const imported: Yue2AitkRunRecord[] = [];
  for (const folder of folders) {
    if (known.has(path.resolve(folder).toLowerCase())) continue;
    try {
      const run = importedRun(folder);
      if (!run || jobs.has(run.jobId)) continue;
      skippedFolders.delete(folder);
      retained.push(run);
      imported.push(run);
      jobs.add(run.jobId);
    } catch (err) {
      if (!skippedFolders.has(folder)) console.warn(`[YuE2] Skipping ${folder}: ${String(err)}`);
      skippedFolders.add(folder);
    }
  }
  if (retained.length !== index.length || retained.some((r, i) => r !== index[i])) writeIndex(retained, imported);
  lastFolderSet = folderSet;
  lastDatasetSet = datasetSetForSkipped();
}

/** At server start nothing is training, so every 'running' entry was killed
 *  with the previous process: mark it interrupted rather than lie forever. */
export function reconcileYue2AitkRunsAtStartup(): number {
  try {
    const index = readIndex();
    // A pulled ladder's `running` means "still training on its worker", not
    // on this process — this machine restarting says nothing about that.
    const stale = index.filter(r => r.status === 'running' && !r.origin);
    if (stale.length) {
      const updated = index.map(r => r.status === 'running' && !r.origin ? { ...r, status: 'interrupted' as const, updatedAt: Date.now() } : r);
      writeIndex(updated, updated.filter(r => r.status === 'interrupted' && stale.some(s => s.jobId === r.jobId)));
    }
    reconcileFromDisk(true);
    return stale.length;
  } catch { return 0; }
}

/** Per dataset: the newest joint run holding a checkpoint with both native
 *  AR and NAR weights. One index read for the whole list, and only two stats
 *  per checkpoint dir (no train.jsonl or meters parse) so the dataset grid can
 *  call it on every list request. */
export function findYue2JointAdaptersFor(rows: Array<{ id: string; slug: string }>): Map<string, { dir: string; trainedAt: string }> {
  reconcileFromDisk();
  const out = new Map<string, { dir: string; trainedAt: string }>();
  const bySlug = new Map(rows.map(r => [r.slug, r.id]));
  const ids = new Set(rows.map(r => r.id));
  for (const r of readIndex().sort((a, b) => b.updatedAt - a.updatedAt)) {
    const id = ids.has(r.datasetId) ? r.datasetId : bySlug.get(r.datasetSlug);
    if (!id || out.has(id)) continue;
    const ls = (dir: string) => { try { return fs.readdirSync(dir).map(n => path.join(dir, n)); } catch { return []; } };
    const bases = [r.output, ...ls(path.join(r.output, 'segments'))];
    const dir = bases.flatMap(ls).filter(d => /[\\/]checkpoint-step\d+$/.test(d)).find(d =>
      fs.existsSync(path.join(d, 'native-ar.safetensors')) && fs.existsSync(path.join(d, 'native-nar.safetensors')));
    if (dir) out.set(id, { dir, trainedAt: new Date(r.updatedAt).toISOString() });
  }
  return out;
}

/** "Reviewing complete": the listener has found the ladder's winner and will
 *  not score the other rungs. A marker file in the run folder, so it moves
 *  with the run when cleanup relocates it. */
const REVIEW_MARKER = 'review-complete';
export function yue2ReviewComplete(output: string): boolean {
  return fs.existsSync(path.join(output, REVIEW_MARKER));
}
export function setYue2ReviewComplete(output: string, complete: boolean): void {
  const marker = path.join(output, REVIEW_MARKER);
  if (complete) fs.writeFileSync(marker, `${new Date().toISOString()}
`, 'utf8');
  else fs.rmSync(marker, { force: true });
}

/** "Finished": a rung was linked and the run cleaned up. A marker file in the
 *  run folder; runs parked under `refined/` by the old cleanup count too. */
const FINISHED_MARKER = 'finished';
export function yue2RunFinished(output: string): boolean {
  return /[\\/]refined[\\/]/i.test(path.resolve(output)) || fs.existsSync(path.join(output, FINISHED_MARKER));
}
export interface Yue2FinishedPick { pickedStep: number; pickedBlind: boolean; pickedLabel: string; at: string }
export function setYue2RunFinished(output: string, pick: Omit<Yue2FinishedPick, 'at'>): Yue2FinishedPick {
  const marker = { ...pick, at: new Date().toISOString() };
  fs.writeFileSync(path.join(output, FINISHED_MARKER), JSON.stringify(marker) + '\n', 'utf8');
  return marker;
}

export function listYue2AitkRuns(datasetId: string, datasetSlug?: string): Yue2AitkRunRecord[] {
  reconcileFromDisk();
  const index = readIndex();
  const changed: Yue2AitkRunRecord[] = [];
  const runs = index.filter(r => r.datasetId === datasetId || (!!datasetSlug && r.datasetSlug === datasetSlug))
    .map(r => {
      const checkpoints = rungsOf(r, mergeCheckpoints(checkpointRecords(r.output), r.checkpoints));
      if (assignBlindLabels(r, checkpoints)) changed.push(r);
      return { ...r, checkpoints };
    })
    .sort((a, b) => b.updatedAt - a.updatedAt);
  if (changed.length) writeIndex(index, changed);
  return runs;
}

function blindLetter(index: number): string {
  let value = index + 1;
  let label = '';
  while (value > 0) { value--; label = String.fromCharCode(65 + value % 26) + label; value = Math.floor(value / 26); }
  return label;
}

/** Assign new checkpoints together so directory order and reloads cannot move a label. */
function assignBlindLabels(run: Yue2AitkRunRecord, checkpoints: Yue2AitkCheckpointRecord[]): boolean {
  const labels = run.blindLabels ?? {};
  const missing = checkpoints.filter(c => !labels[c.step]);
  if (!missing.length) return false;
  const firstAssignment = Object.keys(labels).length === 0;
  missing.sort((a, b) => {
    const hash = (step: number) => createHash('sha256').update(`${run.jobId}:${step}`).digest('hex');
    return hash(a.step).localeCompare(hash(b.step));
  });
  const used = new Set(Object.values(labels));
  let next = 0;
  for (const checkpoint of missing) {
    while (used.has(blindLetter(next))) next++;
    labels[checkpoint.step] = blindLetter(next);
    used.add(labels[checkpoint.step]);
    next++;
  }
  if (firstAssignment && missing.length >= 3) {
    const byStep = [...missing].sort((a, b) => a.step - b.step);
    if (byStep.every((checkpoint, i) => labels[checkpoint.step] === blindLetter(i))) {
      const first = labels[byStep[0].step];
      for (let i = 0; i < byStep.length - 1; i++) labels[byStep[i].step] = labels[byStep[i + 1].step];
      labels[byStep[byStep.length - 1].step] = first;
    }
  }
  run.blindLabels = labels;
  return true;
}

/** A base-matched run (2026-09-27) has no KL rungs: its ladder is every
 *  complete checkpoint, so each one is flagged as a rung for the Review page,
 *  the best-rung pick and the ladder cards. Other runs keep meters.json's flag. */
function rungsOf(run: Yue2AitkRunRecord, checkpoints: Yue2AitkCheckpointRecord[]): Yue2AitkCheckpointRecord[] {
  if ((run.options as Record<string, unknown> | undefined)?.method !== 'base-matched') return checkpoints;
  return checkpoints.map(c => c.arPath && c.narPath ? { ...c, rung: true } : c);
}

/** Every indexed joint run, newest first, with checkpoint files re-scanned.
 * The generation picker has no dataset filter, so it must use the same durable
 * index and checkpoint scanner as Training Studio rather than walking a second
 * directory tree. */
export function listAllYue2AitkRuns(): Yue2AitkRunRecord[] {
  reconcileFromDisk();
  return readIndex()
    .map(r => ({ ...r, checkpoints: mergeCheckpoints(checkpointRecords(r.output), r.checkpoints) }))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

/** Resolve only paths present in the durable joint catalogue. This is shared by
 * the model picker, prompt builder and caption-source route. */
export function jointRunForAdapter(adapterPath: string): Yue2AitkRunRecord | undefined {
  if (!adapterPath) return undefined;
  const wanted = path.resolve(adapterPath).toLowerCase();
  return listAllYue2AitkRuns().find(run => run.checkpoints.some(checkpoint =>
    [checkpoint.arPath, checkpoint.narPath].some(ref => ref && path.resolve(ref).toLowerCase() === wanted)));
}
