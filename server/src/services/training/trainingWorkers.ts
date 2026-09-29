// training/trainingWorkers.ts — YuE2 training on another PC's GPU
//
// A worker is an ordinary HOT-Step server on another machine, started with
// WORKER_TOKEN set. The controlling machine (TRAINING_WORKERS) captions each
// dataset here, pushes its files to the worker under the same dataset id and
// slug, and queues it on the worker's own YuE2 batch. Every path the pipeline
// passes to ace-train, the engine and the run catalogue stays local to the
// machine that trains; nothing is path-mapped. The Training Studio reaches the
// worker through proxyToWorker, so its batches, ladders, previews and rung
// scores are the worker's. pullLinked brings each linked AR/NAR pair back and
// points this machine's Lyric Studio presets at it.

import fs from 'fs';
import path from 'path';
import { timingSafeEqual } from 'crypto';
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
function resolveDatasetFile(root: string, slug: string, rel: string): string {
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

/** Linked AR/NAR pairs under the adapters folder, relative to it. */
export function workerLinkedPairs(): Array<{ slug: string; at: string; ar: FileStamp & { rel: string }; nar: FileStamp & { rel: string } }> {
  const root = config.aceServer.adapters;
  const stamp = (p: string) => {
    if (!isInside(root, p)) return null;
    try { const st = fs.statSync(p); return { rel: path.relative(root, p).split(path.sep).join('/'), size: st.size, mtimeMs: Math.round(st.mtimeMs) }; } catch { return null; }
  };
  const out = [];
  for (const [slug, pair] of Object.entries(readYue2Linked())) {
    const ar = stamp(pair.arPath); const nar = stamp(pair.narPath);
    if (ar && nar) out.push({ slug, at: pair.at, ar, nar });
  }
  return out;
}

export function workerAdapterFile(rel: string): string {
  const root = config.aceServer.adapters;
  const abs = path.resolve(root, rel);
  if (!isInside(root, abs) || !/\.safetensors$/i.test(abs)) throw Object.assign(new Error('Refused path'), { status: 400 });
  return abs;
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

function workerFetch(w: WorkerInfo, pathAndQuery: string, init: RequestInit & { duplex?: 'half' } = {}): Promise<globalThis.Response> {
  const headers = new Headers(init.headers);
  if (config.workers.token) headers.set(TOKEN_HEADER, config.workers.token);
  return fetch(`${w.url}${pathAndQuery}`, { ...init, headers });
}

async function workerJson<T>(w: WorkerInfo, pathAndQuery: string, init?: RequestInit): Promise<T> {
  const r = await workerFetch(w, pathAndQuery, init);
  const body = await r.json().catch(() => ({})) as T & { error?: string };
  if (!r.ok) throw Object.assign(new Error(`${w.name}: ${body.error || `HTTP ${r.status}`}`), { status: r.status });
  return body;
}

const jsonInit = (method: string, body: unknown): RequestInit => ({ method, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });

export async function workerStatus(w: WorkerInfo): Promise<{ name: string; url: string; online: boolean; version?: string; versionMatch?: boolean; engine?: string; error?: string }> {
  try {
    const h = await workerJson<{ version?: string; aceServer?: { status?: string } }>(w, '/api/health', { signal: AbortSignal.timeout(4000) });
    return { name: w.name, url: w.url, online: true, version: h.version, versionMatch: h.version === APP_VERSION, engine: h.aceServer?.status };
  } catch (err: any) {
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
    out.push({ slug: pair.slug, status: bytes ? 'fetched' : 'current', bytes });
  }
  return out;
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
