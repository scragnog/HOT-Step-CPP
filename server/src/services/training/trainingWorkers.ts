// training/trainingWorkers.ts — YuE2 training on another PC's GPU
//
// A worker is an ordinary HOT-Step server on another machine, started with
// WORKER_TOKEN set. The controlling machine (TRAINING_WORKERS) captions each
// dataset here, pushes its files to the worker under the same dataset id and
// slug, and queues it on the worker's own YuE2 batch. Every path the pipeline
// passes to ace-train, the engine and the run catalogue stays local to the
// machine that trains; nothing is path-mapped.
//
// The worker only trains and renders previews. Everything else — review,
// scoring, blind labels, linking — runs on this machine against its own index
// (ui/src/services/trainingApi.ts's LOCAL_BASE): yue2Mirror.ts copies each
// worker run here as its checkpoints and previews land, after which it is an
// ordinary local run tagged `origin`. pullLinked brings back a pair a ladder
// finish linked on the worker itself.
import fs from 'fs';
import path from 'path';
import { timingSafeEqual, createHash } from 'crypto';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import type { NextFunction, Request, Response } from 'express';
import { APP_VERSION, config } from '../../config.js';
import * as repo from './datasetsRepo.js';
import * as queue from './labelingQueue.js';
import { isInside, labelsDir, slugify, trainingBaseDir } from './paths.js';
import { detailFor, syncCounters } from './datasetDetail.js';
import { readYue2Linked, refreshYue2PresetsForJointCheckpoint } from './lyricStudioExport.js';
import { latestGenerationLyrics, PUSHED_PREVIEW_LYRICS } from './yue2JointTrainRunner.js';
import { samplesMissingYue2Caption } from './yue2CaptionJob.js';
import type { TrainingDatasetRow } from './types.js';
import { deleteYue2AitkRun, jointRunForAdapter, listAllYue2AitkRuns, yue2RunFinished, type Yue2AitkRunRecord } from './yue2AitkRuns.js';
import { listYue2JointPreviews, resolveYue2JointPreview, type Yue2JointPreviewRecord } from './yue2JointPreview.js';
import { listYue2TrainLogs, noteYue2TrainLog, trainLogArchiveDir } from './datasetProfile.js';
import { classifyCommit, currentCommit } from './workerUpdate.js';
import { hasActiveBatch } from './yue2BatchRunner.js';
import { gpuLaneBusy, gpuLaneDepth } from '../generation/gpuLane.js';

export const TOKEN_HEADER = 'x-hotstep-worker-token';
const LABELS_PREFIX = '__labels/';

type FileStamp = { size: number; mtimeMs: number };

// ── Shared: a dataset's files, keyed by forward-slash relative path ─────────

/** Audio and sidecars under the dataset folder (the scanner's own skip rules:
 *  dotted folders and node_modules), plus the label files under __labels/.
 *  Backups (.bak, .prev) never train, so they never travel. */
function datasetFiles(root: string, recursive: boolean, slug: string): Record<string, FileStamp> {
  const out: Record<string, FileStamp> = {};
  const walk = (dir: string, prefix: string, deep: boolean) => {
    let entries: fs.Dirent[];
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const rel = prefix + e.name;
      if (e.isDirectory()) {
        if (deep && !e.name.startsWith('.') && e.name !== 'node_modules') walk(path.join(dir, e.name), `${rel}/`, true);
        continue;
      }
      if (!e.isFile() || /\.(bak|prev|part)$/i.test(e.name) || e.name.startsWith('.hotstep-')) continue;
      const st = fs.statSync(path.join(dir, e.name));
      out[rel] = { size: st.size, mtimeMs: Math.round(st.mtimeMs) };
    }
  };
  walk(root, '', recursive);
  walk(labelsDir(slug), LABELS_PREFIX, false);
  return out;
}

/** Where a relative dataset path lives on this machine. Refuses anything
 *  that climbs out of the dataset folder or its labels folder. */
export function resolveDatasetFile(root: string, slug: string, rel: string): string {
  const labels = rel.startsWith(LABELS_PREFIX);
  const base = labels ? labelsDir(slug) : root;
  const out = path.resolve(base, labels ? rel.slice(LABELS_PREFIX.length) : rel);
  if (out === path.resolve(base) || !isInside(base, out)) throw new Error(`Refused path: ${rel}`);
  return out;
}

// ── Worker side ─────────────────────────────────────────────────────────────

const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

function tokenMatches(given: string | undefined, expected: string): boolean {
  const a = Buffer.from(given ?? ''); const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

/** WORKER_TOKEN set: API calls from other machines must carry it. The
 *  worker's own batch runner calls itself over loopback, as does a browser
 *  on the worker itself. */
export function workerTokenGate(req: Request, res: Response, next: NextFunction): void {
  const token = config.workers.acceptToken;
  if (!token || LOOPBACK.has(req.socket.remoteAddress ?? '') || tokenMatches(req.get(TOKEN_HEADER), token)) { next(); return; }
  res.status(401).json({ error: 'This HOT-Step is a training worker: the request needs its WORKER_TOKEN' });
}

export const workerDatasetsDir = () => path.join(trainingBaseDir, 'worker-datasets');
const workerDatasetRoot = (slug: string) => path.join(workerDatasetsDir(), slugify(slug));

export function workerDatasetFiles(slug: string): Record<string, FileStamp> {
  return datasetFiles(workerDatasetRoot(slug), true, slugify(slug));
}

export async function receiveDatasetFile(datasetId: string, slug: string, rel: string, mtimeMs: number, body: NodeJS.ReadableStream): Promise<void> {
  if (queue.activeJobForDataset(datasetId)) throw Object.assign(new Error('A job is running for this dataset on the worker'), { status: 409 });
  const dest = resolveDatasetFile(workerDatasetRoot(slug), slugify(slug), rel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const part = `${dest}.part`;
  await pipeline(body, fs.createWriteStream(part));
  fs.renameSync(part, dest);
  if (Number.isFinite(mtimeMs) && mtimeMs > 0) { const t = new Date(mtimeMs); fs.utimesSync(dest, t, t); }
}

/** The dataset row, under the controller's id and slug, pointing at the
 *  pushed folder. A new row, or the settings of an existing one refreshed. */
export async function upsertPushedDataset(id: string, row: TrainingDatasetRow): Promise<void> {
  const slug = slugify(row.slug);
  const sourceDir = workerDatasetRoot(slug);
  if (!fs.existsSync(sourceDir)) throw Object.assign(new Error('Push the dataset files first'), { status: 400 });
  const datasetJson = path.join(sourceDir, 'dataset.json');
  const settings = {
    name: row.name, recursive: true, customTag: row.customTag, tagPosition: row.tagPosition, genreRatio: row.genreRatio,
    defaultArtist: row.defaultArtist, defaultAlbum: row.defaultAlbum, defaultGenre: row.defaultGenre, defaultLanguage: row.defaultLanguage,
    status: row.status, builtAt: row.builtAt, datasetJsonPath: fs.existsSync(datasetJson) ? datasetJson : '', albumName: row.albumName,
  };
  const existing = repo.getDataset(id);
  if (!existing) {
    const clash = repo.listDatasets().find(d => d.slug === slug || path.resolve(d.sourceDir) === path.resolve(sourceDir));
    if (clash) throw Object.assign(new Error(`The worker already has a different dataset "${clash.name}" at ${slug}`), { status: 409 });
    const now = new Date().toISOString();
    repo.insertDataset({ ...settings, id, slug, sourceDir, sampleCount: 0, labeledCount: 0, excludedCount: 0, lyricsSetId: 0, createdAt: now, updatedAt: now });
  } else {
    repo.updateDataset(id, settings);
  }
  const ds = repo.getDataset(id)!;
  syncCounters(ds, (await detailFor(ds)).samples);
}

export interface WorkerLinkedLog { seg: string; rel: string; size: number; mtimeMs: number }
export interface WorkerLinkedPair {
  slug: string; at: string; ar: FileStamp & { rel: string }; nar: FileStamp & { rel: string };
  jobId?: string; keptStep?: number; logs?: WorkerLinkedLog[];
}

/** Linked AR/NAR pairs under the adapters folder, relative to it, plus (when
 *  the run that produced the pair is still in this worker's durable index)
 *  the jobId, the kept checkpoint's own step, and its train.jsonl file(s) —
 *  so the controller can pull the loss curve back through the same route
 *  that serves the two safetensors. */
export function workerLinkedPairs(): WorkerLinkedPair[] {
  const root = config.aceServer.adapters;
  const stamp = (p: string) => {
    if (!isInside(root, p)) return null;
    try { const st = fs.statSync(p); return { rel: path.relative(root, p).split(path.sep).join('/'), size: st.size, mtimeMs: Math.round(st.mtimeMs) }; } catch { return null; }
  };
  const out: WorkerLinkedPair[] = [];
  for (const [slug, pair] of Object.entries(readYue2Linked())) {
    const ar = stamp(pair.arPath); const nar = stamp(pair.narPath);
    if (!ar || !nar) continue;
    const entry: WorkerLinkedPair = { slug, at: pair.at, ar, nar };
    const run = jointRunForAdapter(pair.arPath);
    if (run) {
      entry.jobId = run.jobId;
      // The checkpoint directory itself, not a substring match anywhere in the
      // path — a run folder's own name can legally contain "checkpoint-stepN"
      // too (yue2JointOutputDirectory sanitizes a trigger into the folder name
      // without forbidding that), which would misreport an ancestor's step.
      const step = /^checkpoint-step(\d+)$/.exec(path.basename(path.dirname(pair.arPath)))?.[1];
      if (step) entry.keptStep = Number(step);
      const logs = listYue2TrainLogs(run.output)
        .map(({ seg, file }) => { try { const st = fs.statSync(file); return { seg, rel: path.relative(root, file).split(path.sep).join('/'), size: st.size, mtimeMs: Math.round(st.mtimeMs) }; } catch { return null; } })
        .filter((l): l is WorkerLinkedLog => !!l);
      if (logs.length) entry.logs = logs;
    }
    out.push(entry);
  }
  return out;
}

export function workerAdapterFile(rel: string): string {
  const root = config.aceServer.adapters;
  const abs = path.resolve(root, rel);
  if (!isInside(root, abs) || !/\.(safetensors|jsonl)$/i.test(abs)) throw Object.assign(new Error('Refused path'), { status: 400 });
  return abs;
}

// The mirror (yue2Mirror.ts on the controller) copies each run down here file
// by file as it lands: the manifest says what exists, the file route serves
// one entry of it, the delete route drops the worker's copy once mirrored.

export interface MirrorFile { rel: string; size: number; mtimeMs: number }
export interface MirrorRun {
  jobId: string; datasetId: string; datasetSlug: string;
  /** Basename of the run's output folder on the worker. */
  folder: string;
  status: Yue2AitkRunRecord['status'];
  createdAt: number; updatedAt: number;
  options: Record<string, unknown>; blindLabels?: Record<string, string>;
  /** A preview of this run is still rendering on the worker right now. */
  previewsBusy: boolean;
  /** Forward-slash paths relative to the run folder. */
  files: MirrorFile[];
  previews: Yue2JointPreviewRecord[];
}

/** What the controller needs of a run: its loss logs, style norms, every
 *  complete checkpoint's weights and meters, and every rendered preview. Never
 *  optimizer.resume or adapter.safetensors: nothing on the controller reads
 *  them. ace-train writes a checkpoint into a temp folder and renames it into
 *  place (yue2-aitk-runtime.cpp, save_checkpoint), so a checkpoint folder
 *  holding both weight files is complete. */
function mirrorFiles(run: Yue2AitkRunRecord, previews: Yue2JointPreviewRecord[]): MirrorFile[] {
  const out: MirrorFile[] = [];
  const add = (abs: string) => {
    try {
      const st = fs.statSync(abs);
      if (st.isFile()) out.push({ rel: path.relative(run.output, abs).split(path.sep).join('/'), size: st.size, mtimeMs: Math.round(st.mtimeMs) });
    } catch { /* not there (yet) */ }
  };
  for (const log of listYue2TrainLogs(run.output)) add(log.file);
  add(path.join(run.output, 'style-norms.json'));
  for (const c of run.checkpoints) {
    if (!c.arPath || !c.narPath) continue;
    for (const name of ['native-ar.safetensors', 'native-nar.safetensors', 'meters.json']) add(path.join(c.dir, name));
  }
  for (const p of previews) {
    const wav = p.status === 'done' && p.file ? resolveYue2JointPreview(run.output, p.file) : null;
    if (!wav) continue;
    add(wav);
    add(wav.replace(/\.wav$/i, '.score.abc'));
  }
  return out;
}

/** Every run still in play on this worker. A run already finished here (a
 *  `finished` marker, or a NAR-further run under refined/) is left out: the
 *  controller would land it as an ordinary unfinished ladder, which its next
 *  cleanup sweeps away, and the worker would drop its own copy. Such a pair
 *  reaches the controller through pullLinked instead. */
export function workerMirrorManifest(): MirrorRun[] {
  return listAllYue2AitkRuns().filter(run => !yue2RunFinished(run.output)).map(run => {
    // Previews render inside the run's training job or on the GPU lane. A
    // 'rendering' record with neither alive was cut off by a restart or crash
    // and will never finish: report it failed, so it neither shows as
    // rendering forever on the controller nor keeps the run from retiring.
    const live = !!queue.activeJobForDataset(run.datasetId) || gpuLaneBusy() || gpuLaneDepth() > 0;
    const previews = listYue2JointPreviews(run.output).map(p => p.status === 'rendering' && !live
      ? { ...p, status: 'failed' as const, error: p.error ?? 'Render interrupted on the worker' } : p);
    return {
      jobId: run.jobId, datasetId: run.datasetId, datasetSlug: run.datasetSlug, folder: path.basename(run.output),
      status: run.status, createdAt: run.createdAt, updatedAt: run.updatedAt, options: run.options, blindLabels: run.blindLabels,
      previewsBusy: previews.some(p => p.status === 'rendering'), files: mirrorFiles(run, previews), previews,
    };
  });
}

const mirrorHashes = new Map<string, string>();

/** sha256 of the first `size` bytes, streamed so a large file never blocks
 *  the event loop, and cached per file state: hashed once per worker process. A log
 *  still growing is served (and hashed) only up to the size it had when asked. */
async function mirrorHash(abs: string, size: number, mtimeMs: number): Promise<string> {
  const key = `${abs}|${size}|${mtimeMs}`;
  const known = mirrorHashes.get(key);
  if (known) return known;
  const hash = createHash('sha256');
  if (size > 0) for await (const chunk of fs.createReadStream(abs, { start: 0, end: size - 1 })) hash.update(chunk as Buffer);
  const hex = hash.digest('hex');
  mirrorHashes.set(key, hex);
  return hex;
}

/** One manifest entry of one run, re-derived from the worker's own index:
 *  `rel` is matched against the list, never joined onto a path. */
export async function workerMirrorFile(jobId: string, rel: string): Promise<{ abs: string; size: number; mtimeMs: number; sha256: string }> {
  const run = listAllYue2AitkRuns().find(r => r.jobId === jobId);
  if (!run) throw Object.assign(new Error('Run not found'), { status: 404 });
  const entry = mirrorFiles(run, listYue2JointPreviews(run.output)).find(f => f.rel === rel);
  if (!entry) throw Object.assign(new Error('Not a mirrored file of this run'), { status: 404 });
  const abs = path.join(run.output, ...entry.rel.split('/'));
  const st = fs.statSync(abs);
  const mtimeMs = Math.round(st.mtimeMs);
  return { abs, size: st.size, mtimeMs, sha256: await mirrorHash(abs, st.size, mtimeMs) };
}

/** A run that just ended may be about to seed a chained stage: a batch's
 *  refine stage, or the training runner's automatic refinement (posted 1.5 s
 *  after the job ends). Both resume from its optimizer.resume. */
const CHAIN_GRACE_MS = 2 * 60_000;

/** Drop a run's folder and index entry on this worker once the controller
 *  has mirrored it (or discarded it). Refuses while the run is still
 *  training, while a batch is active or the run ended moments ago (the next
 *  chained stage may resume from it), or while any other job for its dataset
 *  is queued or running —
 *  a NAR follow-up or GPU preview job can read this run's checkpoint/output
 *  while the run record itself already shows 'done'; dataset-scoped because
 *  the reading job's own record rarely points back at this folder. Also
 *  refuses a run whose `output` resolves outside the worker's own ladder
 *  tree, guarding against a corrupted index entry. */
export function deleteWorkerMirrorRun(jobId: string): void {
  const run = listAllYue2AitkRuns().find(r => r.jobId === jobId);
  if (!run) throw Object.assign(new Error('Run not found'), { status: 404 });
  if (run.status === 'running') throw Object.assign(new Error('Run is still training'), { status: 409 });
  if (hasActiveBatch() || Date.now() - run.updatedAt < CHAIN_GRACE_MS) throw Object.assign(new Error('A chained training stage may still resume from this run'), { status: 409 });
  if (queue.activeJobForDataset(run.datasetId)) throw Object.assign(new Error('A job is running for this dataset on the worker'), { status: 409 });
  // A manual preview render (training.ts's yue2-joint-previews/render) runs on
  // the GPU lane directly, not through the labeling queue, so
  // activeJobForDataset above never sees it. The lane has no per-dataset
  // tag, so any current or queued reader blocks any delete.
  if (gpuLaneBusy() || gpuLaneDepth() > 0) throw Object.assign(new Error('The GPU is busy rendering on this worker'), { status: 409 });
  const ladderRoot = path.resolve(path.join(config.aceServer.adapters, 'yue2-joint-adapters'));
  const output = path.resolve(run.output);
  // Strict descendant only — isInside treats the root itself as "inside", which
  // would let a corrupted record whose output IS the root delete every ladder.
  if (output === ladderRoot || !isInside(ladderRoot, output)) {
    throw Object.assign(new Error(`Refusing to delete outside the joint adapters folder: ${output}`), { status: 400 });
  }
  deleteYue2AitkRun(jobId);
}

// ── Controller side ─────────────────────────────────────────────────────────

export interface WorkerInfo { name: string; url: string }

/** TRAINING_WORKERS: `LivingRoom=http://192.168.50.50:3001, Other=http://...`. */
export function listWorkers(): WorkerInfo[] {
  const out: WorkerInfo[] = [];
  for (const entry of config.workers.list.split(/[,;\n]/)) {
    const i = entry.indexOf('=');
    if (i <= 0) continue;
    const name = entry.slice(0, i).trim(); const url = entry.slice(i + 1).trim().replace(/\/+$/, '');
    if (name && /^https?:\/\//i.test(url)) out.push({ name, url });
  }
  return out;
}

export const getWorker = (name: string) => listWorkers().find(w => w.name === name);

export function workerFetch(w: WorkerInfo, pathAndQuery: string, init: RequestInit & { duplex?: 'half' } = {}): Promise<globalThis.Response> {
  const headers = new Headers(init.headers);
  if (config.workers.token) headers.set(TOKEN_HEADER, config.workers.token);
  return fetch(`${w.url}${pathAndQuery}`, { ...init, headers });
}

export async function workerJson<T>(w: WorkerInfo, pathAndQuery: string, init?: RequestInit): Promise<T> {
  const r = await workerFetch(w, pathAndQuery, init);
  const body = await r.json().catch(() => ({})) as T & { error?: string };
  if (!r.ok) throw Object.assign(new Error(`${w.name}: ${body.error || `HTTP ${r.status}`}`), { status: r.status });
  return body;
}

const jsonInit = (method: string, body: unknown): RequestInit => ({ method, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });

export async function workerStatus(w: WorkerInfo): Promise<{ name: string; url: string; online: boolean; version?: string; versionMatch?: boolean; engine?: string; error?: string; commit?: string; dirty?: boolean; relation?: string; behind?: number; engineVersion?: string; engineBuiltAt?: string | null; gpu?: { memoryUsedMiB: number; utilization: number } | null; job?: { kind: string; dataset: string; done: number; total: number; status: string } | null; idle?: boolean }> {
  try {
    const h = await workerJson<{ version?: string; aceServer?: { status?: string; version?: string }; commit?: string; dirty?: boolean; engineBuiltAt?: string | null; gpu?: { memoryUsedMiB: number; utilization: number } | null; job?: { kind: string; dataset: string; done: number; total: number; status: string } | null; idle?: boolean }>(w, '/api/training/worker/status', { signal: AbortSignal.timeout(6000) });
    const relation = h.commit ? classifyCommit(h.commit, currentCommit()) : undefined;
    return { name: w.name, url: w.url, online: true, version: h.version, versionMatch: h.version === APP_VERSION, engine: h.aceServer?.status, engineVersion: h.aceServer?.version, engineBuiltAt: h.engineBuiltAt, commit: h.commit, dirty: h.dirty, relation: relation?.relation, behind: relation?.commits, gpu: h.gpu, job: h.job, idle: h.idle };
  } catch (err: any) {
    if (err?.status === 404) {
      try {
        const old = await workerJson<{ version?: string; aceServer?: { status?: string; version?: string } }>(w, '/api/health', { signal: AbortSignal.timeout(4000) });
        return { name: w.name, url: w.url, online: true, version: old.version, versionMatch: old.version === APP_VERSION, engine: old.aceServer?.status, engineVersion: old.aceServer?.version, error: 'Worker update API unavailable; this worker needs a one-time bootstrap' };
      } catch { /* report the original status failure below */ }
    }
    return { name: w.name, url: w.url, online: false, error: err?.cause?.code || err?.message || String(err) };
  }
}

/** Send whatever the worker's copy lacks (by size and mtime), the preview
 *  lyrics this machine's Lyric Studio would have picked, then the row. */
export async function pushDataset(w: WorkerInfo, ds: TrainingDatasetRow): Promise<{ sent: number; skipped: number; bytes: number }> {
  const base = `/api/training/worker/datasets/${encodeURIComponent(ds.id)}`;
  const q = `slug=${encodeURIComponent(ds.slug)}`;
  const { files: theirs } = await workerJson<{ files: Record<string, FileStamp> }>(w, `${base}/files?${q}`);
  const put = async (rel: string, body: BodyInit, mtimeMs: number, size?: number) => {
    const r = await workerFetch(w, `${base}/file?${q}&rel=${encodeURIComponent(rel)}&mtime=${mtimeMs}`, {
      method: 'PUT', body, duplex: 'half',
      headers: { 'content-type': 'application/octet-stream', ...(size !== undefined ? { 'content-length': String(size) } : {}) },
    });
    if (!r.ok) throw new Error(`${w.name}: sending ${rel} failed: ${((await r.json().catch(() => ({}))) as { error?: string }).error || `HTTP ${r.status}`}`);
  };
  let sent = 0, skipped = 0, bytes = 0;
  for (const [rel, st] of Object.entries(datasetFiles(ds.sourceDir, ds.recursive, ds.slug))) {
    const t = theirs[rel];
    if (t && t.size === st.size && Math.abs(t.mtimeMs - st.mtimeMs) < 2000) { skipped++; continue; }
    const abs = resolveDatasetFile(ds.sourceDir, ds.slug, rel);
    await put(rel, Readable.toWeb(fs.createReadStream(abs)) as ReadableStream, st.mtimeMs, st.size);
    sent++; bytes += st.size;
  }
  const gen = ds.lyricsSetId ? latestGenerationLyrics(ds.lyricsSetId) : undefined;
  if (gen) await put(PUSHED_PREVIEW_LYRICS, JSON.stringify(gen), Date.now());
  await workerJson(w, base, jsonInit('POST', { row: ds }));
  return { sent, skipped, bytes };
}

// Dispatch: caption here → push → queue on the worker's batch, per dataset,
// so the worker starts training the first while the rest are still captioning.

export type DispatchItemStatus = 'pending' | 'captioning' | 'pushing' | 'queued' | 'failed';
export interface DispatchItem { datasetId: string; name: string; status: DispatchItemStatus; error: string | null; sent?: number; bytes?: number }
export interface DispatchState { worker: string; batchId: string | null; items: DispatchItem[]; running: boolean; startedAt: number }
export interface DispatchInput { datasetIds: string[]; lyricTiming: boolean; clearCache?: boolean; recipe: Record<string, unknown> }

// ponytail: in memory; a restart mid-dispatch loses the queue line (not the
// worker's batch) and the user dispatches the rest again, which re-sends nothing.
const dispatches = new Map<string, DispatchState & { input: DispatchInput }>();
const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));
const CAPTION_ATTEMPTS = 3;
const CAPTION_RETRY_WAIT_MS = 5 * 60_000;

export function getDispatch(worker: string): DispatchState | null {
  const d = dispatches.get(worker);
  if (!d) return null;
  const { input, ...state } = d;
  return state;
}

export function startDispatch(w: WorkerInfo, input: DispatchInput): DispatchState | { error: string } {
  const items: DispatchItem[] = [];
  for (const id of input.datasetIds) {
    const ds = repo.getDataset(id);
    if (!ds) return { error: `Dataset not found: ${id}` };
    items.push({ datasetId: id, name: ds.name || ds.slug, status: 'pending', error: null });
  }
  if (!items.length) return { error: 'Select at least one dataset' };
  const live = dispatches.get(w.name);
  if (live?.running) {
    for (const item of items) if (!live.items.some(i => i.datasetId === item.datasetId && i.status !== 'failed')) live.items.push(item);
    return getDispatch(w.name)!;
  }
  const state = { worker: w.name, batchId: null, items, running: true, startedAt: Date.now(), input };
  dispatches.set(w.name, state);
  void runDispatch(w, state);
  return getDispatch(w.name)!;
}

async function runDispatch(w: WorkerInfo, state: DispatchState & { input: DispatchInput }): Promise<void> {
  for (let i = 0; i < state.items.length; i++) {
    const item = state.items[i];
    if (item.status !== 'pending') continue;
    try {
      const ds = repo.getDataset(item.datasetId);
      if (!ds) throw new Error('Dataset not found');
      if (state.input.recipe.autoCaption) { item.status = 'captioning'; await captionHere(ds, state.input.recipe.autoCaption as { provider?: string; model?: string }); }
      item.status = 'pushing';
      const pushed = await pushDataset(w, ds);
      item.sent = pushed.sent; item.bytes = pushed.bytes;
      await queueOnWorker(w, state, item.datasetId);
      item.status = 'queued';
    } catch (err: any) {
      item.status = 'failed'; item.error = err?.message || String(err);
      console.warn(`[Workers] ${w.name}: ${item.name} not dispatched: ${item.error}`);
    }
  }
  state.running = false;
}

/** YuE2 captions for the tracks that have none, written here with Gemini (never
 *  MOSS) so the worker needs no API keys. Retried like the batch's own stage,
 *  since one network blip fails every call at once. */
async function captionHere(ds: TrainingDatasetRow, want: { provider?: string; model?: string }): Promise<void> {
  const body = { provider: 'gemini', ...(want.provider === 'gemini' && want.model ? { model: want.model } : {}) };
  for (let attempt = 1; attempt <= CAPTION_ATTEMPTS; attempt++) {
    if (!(await samplesMissingYue2Caption(ds)).length) return;
    if (attempt > 1) await sleep(CAPTION_RETRY_WAIT_MS);
    const r = await fetch(`http://127.0.0.1:${config.server.port}/api/training/datasets/${encodeURIComponent(ds.id)}/yue2-captions-missing`, jsonInit('POST', body));
    const payload = await r.json().catch(() => ({})) as { jobId?: string; error?: string };
    if (!r.ok) throw new Error(`Captions: ${payload.error || `HTTP ${r.status}`}`);
    for (let job = payload.jobId ? queue.getJob(payload.jobId) : undefined; job && (job.status === 'queued' || job.status === 'running'); job = queue.getJob(job.id)) await sleep(1500);
  }
  const left = (await samplesMissingYue2Caption(ds)).length;
  if (left) throw new Error(`${left} track(s) still have no YuE2 caption after ${CAPTION_ATTEMPTS} Gemini passes; not sent to train on the long ACE captions`);
}

async function queueOnWorker(w: WorkerInfo, state: DispatchState & { input: DispatchInput }, datasetId: string): Promise<void> {
  if (state.batchId) {
    const r = await workerFetch(w, `/api/training/yue2-batch/${encodeURIComponent(state.batchId)}/items`, jsonInit('POST', { datasetIds: [datasetId] }));
    if (r.ok) return;
    // 409: that batch finished before this dataset was ready; start another.
  }
  const { lyricTiming, clearCache, recipe } = state.input;
  const start = await workerFetch(w, '/api/training/yue2-batch', jsonInit('POST', { datasetIds: [datasetId], lyricTiming, clearCache, recipe: { ...recipe, autoCaption: false } }));
  if (start.ok) { state.batchId = ((await start.json()) as { batch: { id: string } }).batch.id; return; }
  if (start.status !== 409) throw new Error(`${w.name}: ${((await start.json().catch(() => ({}))) as { error?: string }).error || `HTTP ${start.status}`}`);
  // The worker is already running a batch someone else started: join it.
  const { batches } = await workerJson<{ batches: Array<{ id: string; status: string }> }>(w, '/api/training/yue2-batch');
  const active = batches.find(b => b.status === 'running' || b.status === 'paused');
  if (!active) throw new Error(`${w.name}: batch start refused and no active batch to join`);
  await workerJson(w, `/api/training/yue2-batch/${encodeURIComponent(active.id)}/items`, jsonInit('POST', { datasetIds: [datasetId] }));
  state.batchId = active.id;
}

/** Fetch every linked pair whose dataset exists here and link it to this
 *  machine's presets. Files already here at the same size are not re-sent. */
export async function pullLinked(w: WorkerInfo): Promise<Array<{ slug: string; status: 'fetched' | 'current' | 'no-dataset'; bytes: number }>> {
  const { linked } = await workerJson<{ linked: ReturnType<typeof workerLinkedPairs> }>(w, '/api/training/worker/linked');
  const root = config.aceServer.adapters;
  const out: Array<{ slug: string; status: 'fetched' | 'current' | 'no-dataset'; bytes: number }> = [];
  for (const pair of linked) {
    const ds = repo.listDatasets().find(d => d.slug.toLowerCase() === pair.slug.toLowerCase());
    if (!ds) { out.push({ slug: pair.slug, status: 'no-dataset', bytes: 0 }); continue; }
    let bytes = 0;
    const local: string[] = [];
    for (const f of [pair.ar, pair.nar]) {
      const dest = path.resolve(root, f.rel);
      if (!isInside(root, dest)) throw new Error(`Refused path from ${w.name}: ${f.rel}`);
      local.push(dest);
      try { if (fs.statSync(dest).size === f.size) continue; } catch { /* not here yet */ }
      const r = await workerFetch(w, `/api/training/worker/adapter-file?rel=${encodeURIComponent(f.rel)}`);
      if (!r.ok || !r.body) throw new Error(`${w.name}: fetching ${f.rel} failed (HTTP ${r.status})`);
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      await pipeline(Readable.fromWeb(r.body as any), fs.createWriteStream(`${dest}.part`));
      fs.renameSync(`${dest}.part`, dest);
      bytes += f.size;
    }
    refreshYue2PresetsForJointCheckpoint(ds, local[0], local[1]);
    bytes += await pullLinkedLogs(w, ds.slug, pair);
    out.push({ slug: pair.slug, status: bytes ? 'fetched' : 'current', bytes });
  }
  return out;
}

const SAFE_NAME = /^[A-Za-z0-9._-]+$/;

/** Fetch a linked pair's train.jsonl file(s) through the same adapter-file
 *  route as the two safetensors, and note which rung was kept. An old
 *  worker sends no jobId/logs, so this is a no-op for it. jobId is validated
 *  once up front: noteYue2TrainLog builds its sidecar path from it too, so an
 *  unsafe jobId must skip the whole pair, not just the per-log fetch loop. */
async function pullLinkedLogs(w: WorkerInfo, slug: string, pair: ReturnType<typeof workerLinkedPairs>[number]): Promise<number> {
  if (!pair.jobId) return 0;
  if (!SAFE_NAME.test(pair.jobId) || pair.jobId === '.' || pair.jobId === '..') {
    console.warn(`[Workers] ${w.name}: skipping train log with an unsafe jobId (${pair.jobId})`); return 0;
  }
  const archiveDir = path.resolve(trainLogArchiveDir(slug));
  let bytes = 0;
  for (const log of pair.logs ?? []) {
    if (!SAFE_NAME.test(log.seg) || log.seg === '.' || log.seg === '..') {
      console.warn(`[Workers] ${w.name}: skipping train log with an unsafe seg (${log.seg})`); continue;
    }
    const dest = path.resolve(archiveDir, `${pair.jobId}-${log.seg}.jsonl`);
    if (!isInside(archiveDir, dest)) { console.warn(`[Workers] ${w.name}: refused train log path ${log.rel}`); continue; }
    try { if (fs.statSync(dest).size === log.size) continue; } catch { /* not here yet */ }
    const r = await workerFetch(w, `/api/training/worker/adapter-file?rel=${encodeURIComponent(log.rel)}`);
    if (!r.ok || !r.body) { console.warn(`[Workers] ${w.name}: fetching ${log.rel} failed (HTTP ${r.status})`); continue; }
    fs.mkdirSync(archiveDir, { recursive: true });
    await pipeline(Readable.fromWeb(r.body as any), fs.createWriteStream(`${dest}.part`));
    fs.renameSync(`${dest}.part`, dest);
    bytes += log.size;
  }
  try { noteYue2TrainLog(slug, pair.jobId, { ...(pair.keptStep !== undefined ? { keptStep: pair.keptStep } : {}), pulledFrom: w.name, pulledAt: Date.now() }); }
  catch (err: any) { console.warn(`[Workers] ${w.name}: could not note the loss log of run ${pair.jobId}: ${err?.message || err}`); }
  return bytes;
}

// Proxy: the Training Studio's /api/training calls, sent to a worker.

const DROP_REQUEST = new Set(['host', 'connection', 'content-length', 'transfer-encoding', 'keep-alive', 'upgrade', 'accept-encoding', TOKEN_HEADER]);
const DROP_RESPONSE = new Set(['connection', 'transfer-encoding', 'keep-alive', 'content-encoding']);

/** Mounted at /api/workers/:name/api. Training routes only. Server-built
 *  /api/training/ URLs in JSON (preview audio) are rewritten to come back
 *  through the proxy. */
export async function proxyToWorker(req: Request, res: Response): Promise<void> {
  const w = getWorker(req.params.name as string);
  if (!w) { res.status(404).json({ error: `No training worker named ${req.params.name}` }); return; }
  if (!req.url.startsWith('/training/')) { res.status(403).json({ error: 'Only training routes are forwarded to a worker' }); return; }
  const ac = new AbortController();
  res.on('close', () => ac.abort());
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(req.headers)) if (typeof v === 'string' && !DROP_REQUEST.has(k)) headers[k] = v;
  if (config.workers.token) headers[TOKEN_HEADER] = config.workers.token;
  const hasBody = req.method !== 'GET' && req.method !== 'HEAD';
  let r: globalThis.Response;
  try {
    r = await fetch(`${w.url}/api${req.url}`, {
      method: req.method, headers, signal: ac.signal, redirect: 'manual',
      ...(hasBody ? { body: Readable.toWeb(req) as ReadableStream, duplex: 'half' } : {}),
    } as RequestInit);
  } catch (err: any) {
    if (!res.headersSent && !ac.signal.aborted) res.status(502).json({ error: `Training worker ${w.name} is unreachable: ${err?.cause?.code || err?.message || err}` });
    return;
  }
  res.status(r.status);
  const json = (r.headers.get('content-type') ?? '').includes('application/json');
  const encoded = r.headers.has('content-encoding');
  r.headers.forEach((v, k) => { if (!DROP_RESPONSE.has(k) && !(k === 'content-length' && (json || encoded))) res.setHeader(k, v); });
  if (json) {
    const text = await r.text();
    res.send(text.replaceAll('"/api/training/', `"/api/workers/${encodeURIComponent(w.name)}/api/training/`));
    return;
  }
  if (!r.body) { res.end(); return; }
  Readable.fromWeb(r.body as any).on('error', () => res.destroy()).pipe(res);
}

