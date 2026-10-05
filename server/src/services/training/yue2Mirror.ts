// yue2Mirror.ts — copy each training worker's YuE2 runs down to this machine.
//
// A worker trains and renders previews, nothing else. As each checkpoint and
// preview lands there, a background pass per worker copies it into this
// machine's own yue2-joint-adapters/<folder>, after which the run is an
// ordinary local run (tagged `origin`): Review, scoring, Use this rung,
// cleanup and linking read local files only. Once a run is terminal on the
// worker and fully copied, the worker's copy is deleted.
//
// The worker's side is trainingWorkers.ts (workerMirrorManifest,
// workerMirrorFile, deleteWorkerMirrorRun) behind /api/training/worker/yue2-mirror*.
import fs from 'fs';
import path from 'path';
import { createHash, randomUUID } from 'crypto';
import { Readable, Transform } from 'stream';
import { pipeline } from 'stream/promises';
import { config } from '../../config.js';
import * as repo from './datasetsRepo.js';
import { isInside, trainingBaseDir } from './paths.js';
import { getWorker, listWorkers, workerFetch, workerJson, type MirrorRun, type WorkerInfo } from './trainingWorkers.js';
import { deleteYue2AitkRun, freeYue2RunDirectory, listAllYue2AitkRuns, moveYue2AitkRun, recordYue2AitkRun, withYue2RunLock, yue2RunFinished,
  type Yue2AitkRunRecord } from './yue2AitkRuns.js';
import { listYue2JointPreviews, recordYue2JointPreview } from './yue2JointPreview.js';
import { rebaseYue2JointLinks } from './lyricStudioExport.js';
import { noteYue2TrainLog, trainLogArchiveDir, trainLogName } from './datasetProfile.js';

export interface MirrorStatus {
  worker: string; running: boolean; lastPassAt: number | null; lastError: string | null; online: boolean;
  pendingFiles: number; pendingBytes: number; runs: number;
}

const FAST_MS = 15_000;
const SLOW_MS = 120_000;
const TERMINAL = new Set<Yue2AitkRunRecord['status']>(['done', 'failed', 'cancelled']);
const SAFE_NAME = /^[A-Za-z0-9._-]{1,128}$/;
/** Every path a worker may hand over, relative to the run folder. Anything
 *  else in a manifest is refused, never joined onto a local path. */
const MIRROR_REL = /^(?:train\.jsonl|style-norms\.json|segments\/segment-\d{6}\/train\.jsonl|(?:segments\/segment-\d{6}\/)?checkpoint-step\d+\/(?:native-ar\.safetensors|native-nar\.safetensors|meters\.json)|previews\/[A-Za-z0-9._-]+\.(?:wav|score\.abc))$/;
const SAFE_PREVIEW = /^[A-Za-z0-9._-]+\.wav$/i;
const errorText = (err: unknown) => (err as any)?.cause?.code || (err as any)?.message || String(err);

// ── Tombstones: runs discarded here that the next pass must not recreate ────

type Tombstone = { worker: string; remoteJobId: string };
const tombstonesPath = () => path.join(trainingBaseDir, 'yue2-mirror-tombstones.json');

function readTombstones(): Tombstone[] {
  try {
    const value = JSON.parse(fs.readFileSync(tombstonesPath(), 'utf8')) as unknown;
    return Array.isArray(value) ? value.filter((t): t is Tombstone => typeof t?.worker === 'string' && typeof t?.remoteJobId === 'string') : [];
  } catch { return []; }
}
const isTombstoned = (worker: string, remoteJobId: string) => readTombstones().some(t => t.worker === worker && t.remoteJobId === remoteJobId);

function addTombstone(worker: string, remoteJobId: string): void {
  const list = readTombstones();
  if (list.some(t => t.worker === worker && t.remoteJobId === remoteJobId)) return;
  const file = tombstonesPath();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.${randomUUID()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify([...list, { worker, remoteJobId }], null, 1), 'utf8');
  fs.renameSync(tmp, file);
}

/** Ask a worker to drop its copy of a run. 'busy' (409: still training, or
 *  something on the worker may read it) is retried by the next pass. */
async function deleteOnWorker(w: WorkerInfo, remoteJobId: string): Promise<'deleted' | 'busy'> {
  try {
    await workerJson(w, `/api/training/worker/yue2-mirror/${encodeURIComponent(remoteJobId)}`, { method: 'DELETE', signal: AbortSignal.timeout(30_000) });
    return 'deleted';
  } catch (err: any) {
    if (err?.status === 404) return 'deleted';
    if (err?.status === 409) return 'busy';
    throw err;
  }
}

/** Discard a run on this machine. A mirrored run is tombstoned first, so no
 *  later pass brings it back, and its worker copy is dropped best-effort (the
 *  passes keep asking until the worker agrees). Holds the run's lock, so a
 *  pass never writes into a folder being deleted. */
export function deleteYue2Run(jobId: string): Promise<{ output: string }> {
  return withYue2RunLock(jobId, async () => {
    const origin = listAllYue2AitkRuns().find(r => r.jobId === jobId)?.origin;
    if (origin) addTombstone(origin.worker, origin.remoteJobId);
    const out = deleteYue2AitkRun(jobId);
    const w = origin && getWorker(origin.worker);
    if (w) void deleteOnWorker(w, origin!.remoteJobId).catch(err => console.warn(`[Mirror] ${w.name}: could not drop ${origin!.remoteJobId}: ${errorText(err)}`));
    return out;
  });
}

// ── The per-run ledger: what this machine holds and was verified ────────────

type LedgerEntry = { size: number; mtimeMs: number; sha256: string };
type Ledger = Record<string, LedgerEntry>;
const ledgerPath = (output: string) => path.join(output, '.mirror.json');

function readLedger(output: string): Ledger {
  try { const v = JSON.parse(fs.readFileSync(ledgerPath(output), 'utf8')); return v && typeof v === 'object' ? v as Ledger : {}; }
  catch { return {}; }
}
function writeLedger(output: string, ledger: Ledger): void {
  const tmp = `${ledgerPath(output)}.${randomUUID()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(ledger, null, 1), 'utf8');
  fs.renameSync(tmp, ledgerPath(output));
}

const localSize = (file: string) => { try { return fs.statSync(file).size; } catch { return -1; } };

const mirrorFileUrl = (remoteJobId: string, rel: string) =>
  `/api/training/worker/yue2-mirror-file?jobId=${encodeURIComponent(remoteJobId)}&rel=${encodeURIComponent(rel)}`;

async function hashLocal(file: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk as Buffer);
  return hash.digest('hex');
}

/** A file already here at the worker's size but missing from the ledger (a
 *  run the migration moved here, or a pass cut off between rename and ledger
 *  write): compare it with the worker's hash, asked by HEAD so nothing is
 *  downloaded. The ledger entry when they match, else null. Hashed once: the
 *  entry is recorded. */
async function adoptLocal(w: WorkerInfo, remoteJobId: string, rel: string, file: string): Promise<LedgerEntry | null> {
  const r = await workerFetch(w, mirrorFileUrl(remoteJobId, rel), { method: 'HEAD', signal: AbortSignal.timeout(30 * 60_000) });
  const want = r.headers.get('x-sha256');
  if (!r.ok || !want) return null;
  const sha256 = await hashLocal(file);
  return sha256 === want ? { size: localSize(file), mtimeMs: Number(r.headers.get('x-mtime')), sha256 } : null;
}

// A download or ledger write cut off by a restart leaves its uniquely named
// temp file behind. Swept once per run folder per process, under the run's
// lock, so no fetch of this pass is writing one.
const PARTIAL = /(?:\.part-|^\.mirror\.json\.)[0-9a-f-]{36}(?:\.tmp)?$/;
const swept = new Set<string>();

function sweepPartials(output: string): void {
  if (swept.has(output)) return;
  swept.add(output);
  let names: string[] = [];
  try { names = fs.readdirSync(output, { recursive: true }) as string[]; } catch { return; }
  for (const rel of names) if (PARTIAL.test(path.basename(rel))) fs.rmSync(path.join(output, rel), { force: true });
}

/** Download one file to a unique .part, hashing as it streams, and rename it
 *  into place only when it matches the worker's x-sha256. One retry. */
async function fetchVerified(w: WorkerInfo, remoteJobId: string, rel: string, dest: string): Promise<LedgerEntry> {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  let lastError = '';
  for (let attempt = 1; attempt <= 2; attempt++) {
    const part = `${dest}.part-${randomUUID()}`;
    try {
      const r = await workerFetch(w, mirrorFileUrl(remoteJobId, rel));
      if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
      const want = r.headers.get('x-sha256');
      const mtimeMs = Number(r.headers.get('x-mtime'));
      const hash = createHash('sha256');
      let size = 0;
      const tap = new Transform({ transform(chunk: Buffer, _enc, done) { hash.update(chunk); size += chunk.length; done(null, chunk); } });
      await pipeline(Readable.fromWeb(r.body as any), tap, fs.createWriteStream(part));
      const sha256 = hash.digest('hex');
      if (!want || sha256 !== want) throw new Error('checksum mismatch');
      fs.renameSync(part, dest);
      return { size, mtimeMs, sha256 };
    } catch (err) {
      lastError = errorText(err);
    } finally {
      fs.rmSync(part, { force: true });
    }
  }
  throw new Error(lastError);
}

// ── Passes ──────────────────────────────────────────────────────────────────

interface WorkerState { status: MirrorStatus; pass: Promise<MirrorStatus> | null; offline: boolean; active: boolean; warned: Set<string> }
const states = new Map<string, WorkerState>();

function stateFor(worker: string): WorkerState {
  let s = states.get(worker);
  if (!s) {
    s = { status: { worker, running: false, lastPassAt: null, lastError: null, online: false, pendingFiles: 0, pendingBytes: 0, runs: 0 },
      pass: null, offline: false, active: false, warned: new Set() };
    states.set(worker, s);
  }
  return s;
}

export function yue2MirrorStatus(worker: string): MirrorStatus | null {
  const s = states.get(worker);
  return s ? { ...s.status } : null;
}

/** Run one pass for `w` now, or join the one already running. */
export function syncYue2Mirror(w: WorkerInfo): Promise<MirrorStatus> {
  const st = stateFor(w.name);
  if (!st.pass) st.pass = runPass(w, st).finally(() => { st.pass = null; });
  return st.pass;
}

async function runPass(w: WorkerInfo, st: WorkerState): Promise<MirrorStatus> {
  const s = st.status;
  s.running = true; s.pendingFiles = 0; s.pendingBytes = 0;
  const errors: string[] = [];
  let active = false, runs = 0;
  try {
    const reply = await workerJson<{ runs?: MirrorRun[] }>(w, '/api/training/worker/yue2-mirror', { signal: AbortSignal.timeout(30_000) });
    // A worker on older code answers an unknown route with the app's HTML page and a 200.
    if (!Array.isArray(reply.runs)) throw new Error(`${w.name} has no mirror route; update it to this version`);
    const manifest = reply.runs;
    if (st.offline) console.log(`[Mirror] ${w.name} is reachable again`);
    st.offline = false; s.online = true;
    for (const run of manifest) {
      try {
        const r = await mirrorRun(w, run, st, errors);
        if (r !== 'skipped') runs++;
        if (r === 'active') active = true;
      } catch (err) { errors.push(`${run.folder}: ${errorText(err)}`); }
    }
    s.runs = runs;
  } catch (err) {
    s.online = false;
    errors.push(`${w.name} is unreachable: ${errorText(err)}`);
    if (!st.offline) console.warn(`[Mirror] ${errors[errors.length - 1]}`);
    st.offline = true;
  }
  st.active = active || s.pendingFiles > 0;
  Object.assign(s, { running: false, lastPassAt: Date.now(), lastError: errors.length ? errors.join('; ') : null });
  return { ...s };
}

/** One manifest run. 'active': it is still changing on the worker or not yet
 *  fully copied, so the loop polls fast. */
async function mirrorRun(w: WorkerInfo, run: MirrorRun, st: WorkerState, errors: string[]): Promise<'active' | 'idle' | 'skipped'> {
  if (!SAFE_NAME.test(run.jobId) || !SAFE_NAME.test(run.folder) || /^\.+$/.test(run.folder)) {
    errors.push(`${w.name} sent an unsafe run name: ${run.jobId} / ${run.folder}`); return 'skipped';
  }
  const retire = async () => {
    try { await deleteOnWorker(w, run.jobId); } catch (err) { errors.push(`${run.folder}: could not drop the worker copy: ${errorText(err)}`); }
  };
  if (isTombstoned(w.name, run.jobId)) { await retire(); return 'skipped'; }
  const ds = repo.getDataset(run.datasetId) ?? repo.listDatasets().find(d => d.slug === run.datasetSlug);
  if (!ds) {
    if (!st.warned.has(run.jobId)) console.warn(`[Mirror] ${w.name}: no dataset here for ${run.datasetSlug}; ${run.folder} is not mirrored`);
    st.warned.add(run.jobId);
    return 'skipped';
  }
  const ours = (r: Yue2AitkRunRecord) => r.origin?.worker === w.name && r.origin.remoteJobId === run.jobId;
  const jobId = listAllYue2AitkRuns().find(ours)?.jobId ?? run.jobId;
  return withYue2RunLock(jobId, async () => {
    // Re-read under the lock: a delete or cleanup may have run while we waited.
    if (isTombstoned(w.name, run.jobId)) { await retire(); return 'skipped'; }
    const local = listAllYue2AitkRuns().find(r => r.jobId === jobId);
    if (local && !ours(local)) throw new Error(`a run trained here already has the id ${jobId}`);
    if (local && yue2RunFinished(local.output)) { await retire(); return 'idle'; }
    const output = local?.output ?? freeYue2RunDirectory(path.join(config.aceServer.adapters, 'yue2-joint-adapters', run.folder));
    const record = (): Yue2AitkRunRecord => ({
      version: 1, jobId, datasetId: ds.id, datasetSlug: ds.slug, method: 'aitk', output, options: run.options ?? {},
      status: run.status, createdAt: run.createdAt, updatedAt: run.updatedAt, checkpoints: [], blindLabels: run.blindLabels,
      origin: { worker: w.name, remoteJobId: run.jobId },
    });
    // Claim the folder in the index before any file lands, so the disk
    // reconcile never imports it as a run of its own.
    if (!local) { fs.mkdirSync(output, { recursive: true }); recordYue2AitkRun(record()); }
    sweepPartials(output);

    const ledger = readLedger(output);
    const failedBefore = errors.length;
    const files = run.files.filter(f => {
      if (MIRROR_REL.test(f.rel)) return true;
      errors.push(`${run.folder}: refused ${f.rel}`); return false;
    });
    for (const f of files) {
      const file = path.join(output, ...f.rel.split('/'));
      if (ledger[f.rel] || localSize(file) !== f.size) continue;
      try {
        const entry = await adoptLocal(w, run.jobId, f.rel, file);
        if (entry) { ledger[f.rel] = entry; writeLedger(output, ledger); }
      } catch (err) { errors.push(`${run.folder}/${f.rel}: ${errorText(err)}`); }
    }
    const current = (f: MirrorRun['files'][number]) => ledger[f.rel]?.mtimeMs === f.mtimeMs
      && localSize(path.join(output, ...f.rel.split('/'))) === f.size;
    // Small files first: meters before weights, and a ladder's previews
    // become playable before its checkpoints finish copying.
    const todo = files.filter(f => !current(f)).sort((a, b) => a.size - b.size);
    st.status.pendingFiles += todo.length;
    st.status.pendingBytes += todo.reduce((n, f) => n + f.size, 0);
    for (const f of todo) {
      const dest = path.join(output, ...f.rel.split('/'));
      if (!isInside(output, dest)) continue;
      try {
        ledger[f.rel] = await fetchVerified(w, run.jobId, f.rel, dest);
        writeLedger(output, ledger);
        st.status.pendingFiles--; st.status.pendingBytes -= f.size;
      } catch (err) { errors.push(`${run.folder}/${f.rel}: ${errorText(err)}`); }
    }

    // A done preview whose audio is not here yet shows as still rendering,
    // never as failed; previews rendered on this machine are left alone.
    const prior = new Map(listYue2JointPreviews(output).map(p => [p.id, p]));
    for (const p of run.previews ?? []) {
      if (p.file && !SAFE_PREVIEW.test(p.file)) continue;
      const pending = p.status === 'done' && !!p.file && !ledger[`previews/${p.file}`];
      const rec = pending ? { ...p, status: 'rendering' as const } : p;
      if (JSON.stringify(prior.get(p.id)) !== JSON.stringify(rec)) recordYue2JointPreview(output, rec);
    }

    if (!local || todo.length || local.status !== run.status || local.updatedAt !== run.updatedAt) recordYue2AitkRun(record());

    const complete = errors.length === failedBefore && files.every(current);
    if (TERMINAL.has(run.status) && !run.previewsBusy && complete) await retire();
    return run.status === 'running' || run.previewsBusy || !complete ? 'active' : 'idle';
  });
}

let started = false;

/** One background loop per configured worker: every 15 s while a worker run
 *  is changing or not fully copied, every 120 s otherwise. The migration of
 *  old pulled runs runs first, so no pass writes into a `_remote` folder that
 *  is about to move.
 *  ponytail: the worker list is read once; a worker added in Settings starts
 *  mirroring at the next server start. */
export async function startYue2Mirror(): Promise<void> {
  if (started) return;
  started = true;
  await migrateYue2OriginRuns();
  for (const { name } of listWorkers()) {
    const tick = async () => {
      const w = getWorker(name);
      if (w) await syncYue2Mirror(w).catch(err => console.warn(`[Mirror] ${name}: ${errorText(err)}`));
      setTimeout(() => { void tick(); }, stateFor(name).active ? FAST_MS : SLOW_MS).unref();
    };
    void tick();
  }
}

// ── One-off migration of runs pulled by the old ladder pull ─────────────────

/** Move a run folder and repoint the index, yue2-linked.json, album presets,
 *  rung scores and the loss-log note. Any failure puts the folder and every
 *  record back and rethrows. Caller holds the run's lock. */
async function relocateRun(run: Yue2AitkRunRecord, target: string): Promise<void> {
  const from = path.resolve(run.output);
  // Windows refuses the rename (EPERM/EBUSY) while a scanner still holds a
  // handle inside the folder, for a second or so. A failed rename changed nothing.
  for (let wait = 250; ; wait *= 2) {
    try { moveYue2AitkRun(run.jobId, target); break; }
    catch (err: any) {
      if (!['EPERM', 'EBUSY', 'EACCES'].includes(err?.code) || wait > 8000) throw err;
      await new Promise(r => setTimeout(r, wait));
    }
  }
  try { rebaseYue2JointLinks(from, target); }
  catch (err) {
    try { moveYue2AitkRun(run.jobId, from); }
    catch (back: any) { console.error(`[Mirror] could not move ${target} back to ${from}: ${back?.message || back}`); }
    throw err;
  }
  // Informational only: the archived loss log notes where its run lives.
  try {
    if (fs.existsSync(path.join(trainLogArchiveDir(run.datasetSlug), `${trainLogName(run.jobId)}.json`))) noteYue2TrainLog(run.datasetSlug, run.jobId, { output: target });
  } catch (err: any) { console.warn(`[Mirror] could not note the new folder of ${run.jobId}: ${err?.message || err}`); }
}

/** Runs pulled by the old ladder pull become ordinary mirrored runs: a folder
 *  still under `yue2-joint-adapters/_remote` moves to the plain
 *  `yue2-joint-adapters/<name>`, and each record is rewritten from the disk
 *  scan, which drops the pull's per-checkpoint fields. The `remote:<worker>:<id>`
 *  jobId stays (rung scores are keyed by it); the mirror finds the run through
 *  `origin.remoteJobId` and fills in what the worker still holds. Idempotent. */
export async function migrateYue2OriginRuns(): Promise<{ moved: number; errors: string[] }> {
  const joint = path.join(config.aceServer.adapters, 'yue2-joint-adapters');
  const staging = path.join(joint, '_remote');
  let moved = 0;
  const errors: string[] = [];
  for (const { jobId } of listAllYue2AitkRuns().filter(r => r.origin)) {
    try {
      await withYue2RunLock(jobId, async () => {
        let run = listAllYue2AitkRuns().find(r => r.jobId === jobId);
        if (!run) return;
        if (isInside(staging, path.resolve(run.output)) && fs.existsSync(run.output)) {
          await relocateRun(run, freeYue2RunDirectory(path.join(joint, path.basename(run.output))));
          moved++;
          run = listAllYue2AitkRuns().find(r => r.jobId === jobId)!;
        }
        recordYue2AitkRun(run);
      });
    } catch (err) { errors.push(`${jobId}: ${errorText(err)}`); }
  }
  // Drop the staging tree (`_remote/<worker>/`) once nothing is left in it.
  const rmdir = (dir: string) => { try { fs.rmdirSync(dir); } catch { /* not empty, or gone */ } };
  try { for (const e of fs.readdirSync(staging)) rmdir(path.join(staging, e)); } catch { /* gone */ }
  rmdir(staging);
  for (const e of errors) console.warn(`[Mirror] migration: ${e}`);
  return { moved, errors };
}
