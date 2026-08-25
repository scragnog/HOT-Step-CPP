// training/labelingQueue.ts — the Dataset Studio job engine
//
// Structurally a sibling of routes/midiStudio.ts (job Map, replayable event
// buffer, SSE listener set, _meta.json persistence) with three differences:
//   * no child process — all work is in-process (HTTP to ace-server, execFile
//     for Essentia, fetch for Genius/LLM), so cancellation is an AbortController
//   * TWO promise-chain queues — a GPU lane and a network/CPU lane (laneFor);
//     originally one global chain for all kinds (D10), split 2026-07-30 when
//     the trainers joined it and started blocking labeling for hours
//   * one active job per dataset, enforced at the route layer with a 409
//
// Within a label job, Essentia + tag reading (Pass A) and the network steps
// (Pass B) run as concurrent lanes, so the review grid populates within seconds
// while Genius/LLM work is still going. Per-service throughput is governed by
// the limiters in rateLimit.ts, not by hardcoded sleeps.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.7, §4.8, §7.1, §7.9

import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { execSync, type ChildProcess } from 'child_process';
import type { Response } from 'express';
import { pushLog } from '../../routes/logs.js';
import { config } from '../../config.js';
import { jobsDir } from './paths.js';
import { buildSamples, loadSidecarMetadata, sampleFromParts } from './datasetScan.js';
import { mergeFields, writeSidecar } from './sidecarIO.js';
import {
  clearTransientStatus, clearTransientStatuses, patchLabel, readLabel, setTransientStatus,
} from './labelStore.js';
import { analyzeWithEssentia, essentiaAvailable, essentiaKeyString } from './essentiaClient.js';
import { captionLimiter, essentiaLimiter, geniusLimiter } from './rateLimit.js';
import { AbortError, runUnderstand } from './understandClient.js';
import {
  enhanceCaption, enhanceGenius, GeniusNoArtistError, resolveArtistTitle,
} from './enhanceService.js';
import { buildDataset } from './datasetBuilder.js';
import * as audioMeta from './audioMeta.js';
import { getDataset, updateCounters, updateDataset } from './datasetsRepo.js';
import type { AceRequest } from '../aceClient.js';
import { arbitrateSidecarGenre, prepareMossBatch, clearMossBatch } from './mossCaption.js';
import type {
  BuildOptions, CaptionOptions, FieldSource, GeniusOptions, LabelOptions, MergePolicy,
  TrainingDatasetRow, TrainingJobKind, TrainingJobStatus, TrainingJobSummary,
  TrainingSample, TrainingStreamEvent,
} from './types.js';

const MAX_EVENTS = 2000;
const ENGINE_FAILURE_LIMIT = 3; // consecutive connection failures before we give up

/**
 * Run `body` over every target with at most `pool` in flight, stopping the
 * moment the job is cancelled.
 *
 * Replaces the `for (const sample of targets) { ...; await sleep(DELAY) }`
 * shape the Genius and caption jobs both used. The per-call spacing that sleep
 * provided now lives in the service limiter, which paces call STARTS instead of
 * idling between completions — with concurrency 1 the two are equivalent, and
 * above 1 only the limiter is correct.
 *
 * `body` is expected to handle its own per-sample errors (both callers mark the
 * row and continue); anything that escapes stops that worker only.
 */
async function runPooled(
  job: TrainingJob,
  targets: TrainingSample[],
  pool: number,
  body: (sample: TrainingSample) => Promise<void>,
): Promise<void> {
  let cursor = 0;
  const worker = async (): Promise<void> => {
    for (;;) {
      if (isCancelled(job)) return;
      const index = cursor++;
      if (index >= targets.length) return;
      await body(targets[index]);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, pool) }, () => worker()));
}

export interface TrainingJob {
  id: string;
  datasetId: string;
  kind: TrainingJobKind;
  status: TrainingJobStatus;
  total: number;
  done: number;
  failed: number;
  currentSampleId: string | null;
  phase: string;
  engineQueueDepth: number;
  error: string | null;
  createdAt: number;
  startedAt: number | null;
  finishedAt: number | null;
  sampleIds: string[];
  opts: unknown;
  events: TrainingStreamEvent[];
  listeners: Set<Response>;
  controller: AbortController;
  /** Live child process for job kinds that spawn one (preprocess → ace-train).
   *  An AbortController does nothing to a spawned process — cancelJob() kills
   *  this directly. */
  child?: ChildProcess;
}

const jobs = new Map<string, TrainingJob>();
// Two serial chains, not one — see laneFor(). GPU work (trainers, preprocess,
// audition) must not block CPU/network work (labeling, Genius, captions).
let gpuTail: Promise<void> = Promise.resolve();
let netTail: Promise<void> = Promise.resolve();

// ── Job model ────────────────────────────────────────────────────────────

export function toSummary(job: TrainingJob): TrainingJobSummary {
  return {
    id: job.id,
    datasetId: job.datasetId,
    kind: job.kind,
    status: job.status,
    total: job.total,
    done: job.done,
    failed: job.failed,
    currentSampleId: job.currentSampleId,
    phase: job.phase,
    engineQueueDepth: job.engineQueueDepth,
    error: job.error,
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
  };
}

export function createJob(
  kind: TrainingJobKind,
  datasetId: string,
  sampleIds: string[],
  opts: unknown,
): TrainingJob {
  const job: TrainingJob = {
    id: randomUUID(),
    datasetId,
    kind,
    status: 'queued',
    total: sampleIds.length,
    done: 0,
    failed: 0,
    currentSampleId: null,
    phase: 'queued',
    engineQueueDepth: 0,
    error: null,
    createdAt: Date.now(),
    startedAt: null,
    finishedAt: null,
    sampleIds,
    opts,
    events: [],
    listeners: new Set(),
    controller: new AbortController(),
  };
  jobs.set(job.id, job);
  return job;
}

export function getJob(id: string): TrainingJob | undefined {
  return jobs.get(id);
}

export function activeJobForDataset(datasetId: string): TrainingJob | undefined {
  for (const job of jobs.values()) {
    if (job.datasetId === datasetId && (job.status === 'queued' || job.status === 'running')) return job;
  }
  return undefined;
}

/** Active jobs (memory) + finished jobs (disk `_meta.json`), newest first. */
export function listJobs(datasetId?: string): TrainingJobSummary[] {
  const out: TrainingJobSummary[] = [];
  const seen = new Set<string>();

  for (const job of jobs.values()) {
    if (datasetId && job.datasetId !== datasetId) continue;
    out.push(toSummary(job));
    seen.add(job.id);
  }

  try {
    const base = jobsDir();
    if (fs.existsSync(base)) {
      for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
        if (!entry.isDirectory() || seen.has(entry.name)) continue;
        const metaPath = path.join(base, entry.name, '_meta.json');
        if (!fs.existsSync(metaPath)) continue;
        try {
          const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8')) as TrainingJobSummary;
          if (datasetId && meta.datasetId !== datasetId) continue;
          out.push(meta);
        } catch { /* skip corrupted meta */ }
      }
    }
  } catch { /* jobs dir unreadable — memory list is still valid */ }

  out.sort((a, b) => b.createdAt - a.createdAt);
  return out;
}

// ── Events / SSE ─────────────────────────────────────────────────────────

function broadcast(job: TrainingJob, ev: TrainingStreamEvent): void {
  const line = `data: ${JSON.stringify(ev)}\n\n`;
  for (const res of job.listeners) {
    try { res.write(line); } catch { /* client gone; cleanup runs on close */ }
  }
}

/**
 * Append to the replay buffer and push.
 *
 * §2.9: the buffer is capped at 2000, dropping the oldest `log` events first and
 * **never** `sample`/`status`. A clean label run emits almost no logs, so the
 * cap is also kept in reach by superseding a sample's own older event — replay
 * is keyed by sampleId, so only the latest state for each row matters.
 */
export function pushEvent(job: TrainingJob, ev: TrainingStreamEvent): void {
  if (ev.type === 'sample') {
    const prior = job.events.findIndex(e => e.type === 'sample' && e.sampleId === ev.sampleId);
    if (prior >= 0) job.events.splice(prior, 1);
  }
  job.events.push(ev);
  if (job.events.length > MAX_EVENTS) {
    let victim = job.events.findIndex(e => e.type === 'log');
    if (victim < 0) victim = job.events.findIndex(e => e.type === 'progress');
    if (victim < 0) victim = job.events.findIndex(e => e.type === 'job');
    // Terminal fallback. `train-lm` is the first kind that emits a high-volume
    // metric class — one `metric:'step'` per optimizer step, ~25 000 for a
    // 500-song run — so once log/progress/job are all evicted the buffer would
    // otherwise grow without bound and attachStream() would replay every frame on
    // each SSE reconnect. Metrics still outrank logs; `sample`/`status` remain
    // untouchable.
    if (victim < 0) victim = job.events.findIndex(e => e.type === 'metric');
    if (victim >= 0) job.events.splice(victim, 1);   // never `sample`/`status`
  }
  broadcast(job, ev);
}

export function emitJob(job: TrainingJob): void {
  pushEvent(job, { type: 'job', job: toSummary(job) });
}

export function emitProgress(job: TrainingJob): void {
  pushEvent(job, {
    type: 'progress',
    done: job.done,
    total: job.total,
    failed: job.failed,
    phase: job.phase,
    currentSampleId: job.currentSampleId,
    engineQueueDepth: job.engineQueueDepth,
  });
}

function emitLog(job: TrainingJob, level: 'info' | 'warn' | 'error', message: string): void {
  pushEvent(job, { type: 'log', level, message, ts: Date.now() });
}

function endStream(job: TrainingJob): void {
  for (const res of job.listeners) {
    try { res.end(); } catch { /* already closed */ }
  }
  job.listeners.clear();
}

/** SSE: the full replay buffer, then live or closed (§2.9). */
export function attachStream(job: TrainingJob, res: Response): void {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  for (const ev of job.events) {
    res.write(`data: ${JSON.stringify(ev)}\n\n`);
  }
  // The buffer already opens with a `job` frame once the job has started; only
  // synthesise one when it has not, so a replay can never clobber fresh state
  // with the stale summary recorded at job start.
  if (!job.events.some(e => e.type === 'job')) {
    res.write(`data: ${JSON.stringify({ type: 'job', job: toSummary(job) })}\n\n`);
  }

  if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') {
    res.end();
    return;
  }
  job.listeners.add(res);
  res.on('close', () => job.listeners.delete(res));
}

// ── Persistence (§7.9) ───────────────────────────────────────────────────

function persistMeta(job: TrainingJob): void {
  try {
    const dir = path.join(jobsDir(), job.id);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, '_meta.json'), JSON.stringify(toSummary(job), null, 2), 'utf-8');
  } catch (err: any) {
    console.warn(`[Training] Could not persist job ${job.id}: ${err.message}`);
  }
}

/** A server restart loses running jobs — rewrite their meta so the list is honest. */
function recoverStaleJobs(): void {
  try {
    const base = jobsDir();
    if (!fs.existsSync(base)) return;
    for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const metaPath = path.join(base, entry.name, '_meta.json');
      if (!fs.existsSync(metaPath)) continue;
      try {
        const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8')) as TrainingJobSummary;
        if (meta.status !== 'running' && meta.status !== 'queued') continue;
        meta.status = 'failed';
        meta.error = 'Server restarted';
        meta.finishedAt = Date.now();
        fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf-8');
      } catch { /* skip corrupted meta */ }
    }
  } catch { /* jobs dir unreadable */ }
}
recoverStaleJobs();

// ── Queue + cancellation ─────────────────────────────────────────────────

/**
 * Which serial chain a job belongs to.
 *
 * Until 2026-07-30 there was ONE chain for every kind, so a multi-hour DiT
 * training run blocked a label job that never touches the GPU. The original
 * rationale ("the engine has a single FIFO GPU worker") only ever applied to
 * the legacy /understand step, which has been default-off since the 2026-07-27
 * pivot — a modern label job is Essentia (CPU) plus Genius/LLM (network).
 *
 * The GPU lane is still strictly serial: preprocess and both trainers spawn
 * ace-train, which owns the whole card, and they default to stopEngine:true.
 * That is exactly why a label job with useUnderstand belongs in the GPU lane
 * too — /understand needs ace-server UP, and a trainer will have stopped it.
 */
function laneFor(job: TrainingJob): 'gpu' | 'net' {
  switch (job.kind) {
    case 'preprocess':
    case 'train-lm':
    case 'train-dit':
    case 'audition':
    case 'lm-calibrate':   // drives /lm generations — engine must be up, GPU-serial
    case 'dit-calibrate':  // drives /synth + /vae — same rule
    case 'mm3-codes':      // DAV + RVQ encode, whole-card
    case 'mm3-train-lm':   // peaks at 31.7 GB of 32 — nothing else may run
      return 'gpu';
    case 'label':
      return (job.opts as LabelOptions | undefined)?.useUnderstand === true ? 'gpu' : 'net';
    default:
      return 'net';   // enhance-genius, enhance-caption, build
  }
}

/**
 * Append to this job's lane. The two lanes run concurrently; each is serial
 * internally. Per-dataset exclusivity is unaffected — the route layer still
 * rejects a second active job for the same dataset with a 409.
 */
export function enqueue(job: TrainingJob, run: (j: TrainingJob) => Promise<void>): void {
  const swallow = () => { /* runners handle their own errors */ };
  if (laneFor(job) === 'gpu') {
    gpuTail = gpuTail.then(() => run(job)).catch(swallow);
  } else {
    netTail = netTail.then(() => run(job)).catch(swallow);
  }
}

/**
 * Kill a job's spawned child and everything it started.
 *
 * `child.kill()` is a plain TerminateProcess on the ace-train PID only, but
 * ace-train shells out to ffmpeg per song (P5). A cancel mid-transcode would
 * otherwise leave ffmpeg running and writing its temp WAV into `<out>/.tmp/`
 * long after the UI says "cancelled" — nothing reaps it until a later run's
 * startup purge. Same reasoning as the engine stop's `taskkill /T`.
 */
export function killJobChild(job: TrainingJob): void {
  const child = job.child;
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  try {
    if (process.platform === 'win32' && child.pid) {
      execSync(`taskkill /PID ${child.pid} /T /F`, { stdio: 'ignore' });
    } else if (child.pid) {
      // Not detached, so there is no process group to signal — kill the leader
      // and let ffmpeg die on its broken stdio pipe.
      child.kill('SIGKILL');
    }
  } catch {
    try { child.kill(); } catch { /* already gone */ }
  }
}

/**
 * Kill every live job child. Called from index.ts's shutdown() — Node exiting
 * mid-preprocess otherwise orphans a GPU-resident ace-train (~3.2 GB), which
 * then competes with the engine the relaunched server immediately starts. The
 * routine tsx-watch SIGTERM in this repo's dev loop hits exactly this.
 */
export function killActiveChildren(): void {
  for (const job of jobs.values()) {
    if (job.child) {
      console.log(`[Training] Shutdown — killing child of job ${job.id} (pid ${job.child.pid})`);
      killJobChild(job);
    }
  }
}

export function cancelJob(id: string): boolean {
  const job = jobs.get(id);
  if (!job) return false;
  if (job.status === 'queued' || job.status === 'running') {
    job.status = 'cancelled';
    job.finishedAt = Date.now();
    job.controller.abort();
    // A spawned child (ace-train) ignores the AbortController entirely — kill
    // the whole tree, or a cancelled preprocess job keeps the GPU for hours and
    // leaves an ffmpeg grandchild transcoding into <out>/.tmp/.
    killJobChild(job);
    // Rows this job had queued/in flight go back to their on-disk state at once,
    // or they would render "Queued…"/"Working…" (and read-only) forever.
    clearTransientStatuses(job.datasetId, job.sampleIds);
    emitJob(job);
    pushEvent(job, { type: 'status', status: 'cancelled' });
    persistMeta(job);
    endStream(job);
    console.log(`[Training] Job ${id} cancelled`);
  }
  // §2.8 — DELETE always removes the job from the map. The finished `_meta.json`
  // keeps it visible in GET /jobs, and dropping it here bounds memory: every job
  // holds up to MAX_EVENTS events, some carrying a full sample with lyrics.
  jobs.delete(id);
  return true;
}

export function isCancelled(job: TrainingJob): boolean {
  return job.status === 'cancelled' || job.controller.signal.aborted;
}

export function finishJob(job: TrainingJob, status: TrainingJobStatus, error?: string): void {
  if (job.status === 'cancelled') return;   // a cancel already closed the stream
  job.status = status;
  job.error = error ?? null;
  job.finishedAt = Date.now();
  job.currentSampleId = null;
  // Nothing is pending or processing once the job is over — including the rows
  // a failure never reached.
  clearTransientStatuses(job.datasetId, job.sampleIds);
  emitJob(job);
  pushEvent(job, { type: 'status', status, ...(error ? { error } : {}) });
  persistMeta(job);
  endStream(job);
}

// ── Shared helpers ───────────────────────────────────────────────────────

function isEngineConnectionError(err: any): boolean {
  const msg = String(err?.message || err || '');
  return /fetch failed|ECONNREFUSED|ECONNRESET|ENOTFOUND|socket hang up|Engine is not running/i.test(msg);
}

/** Re-read one sample from disk after a sidecar/label write. */
function refreshSample(ds: TrainingDatasetRow, sample: TrainingSample): TrainingSample {
  let sizeBytes = sample.sizeBytes;
  let mtimeMs = 0;
  try {
    const st = fs.statSync(sample.audioPath);
    sizeBytes = st.size;
    mtimeMs = st.mtimeMs;
  } catch { /* file vanished mid-job — the refreshed row keeps the old size */ }

  const label = readLabel(ds.slug, sample.sampleId) ?? undefined;
  const meta = loadSidecarMetadata(sample.audioPath);
  const duration = label?.durationCache?.seconds ?? sample.duration;
  return sampleFromParts(
    ds.id,
    { relPath: sample.relPath, absPath: sample.audioPath, sizeBytes, mtimeMs },
    meta, label, duration, false,
  );
}

/**
 * Lyrics that SAY the track is instrumental, rather than being lyrics.
 *
 * Two shapes, both seen in the wild (americanfootball, 2026-08-18):
 *  - Genius's stock sentence — "This song is an instrumental" — sometimes
 *    embedded in scraped page junk ("[Original song bio:]… Read More This
 *    song is an instrumental").
 *  - Nothing but section tags, at least one of them [Instrumental].
 * Neither contains sung words, so treating them as lyrics both misses the
 * `is_instrumental` flag AND trains the model to sing the Genius boilerplate.
 */
function lyricsSayInstrumental(lyrics: string): boolean {
  const text = lyrics.trim();
  if (!text) return false;
  if (/\bthis (?:song|track) is an instrumental\b/i.test(text)) return true;
  const untagged = text
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line && !/^\[[^\]]*\]$/.test(line));
  return untagged.length === 0 && /\[[^\]]*instrumental[^\]]*\]/i.test(text);
}

/**
 * Merge `incoming` into the sample's sidecar under `policy` and rewrite it.
 *
 * With `deriveInstrumental`, §4.8's rule is applied to the **winning** lyrics —
 * the merge result, not just what /understand returned. Otherwise a track whose
 * existing lyrics survive the merge would still be flagged instrumental
 * whenever the local transcription came back empty.
 */
async function mergeIntoSidecar(
  sample: TrainingSample,
  incoming: Record<string, string>,
  policy: MergePolicy,
  deriveInstrumental = false,
  forced?: Record<string, string>,
): Promise<void> {
  // `existing` comes from the 3-convention detect, but the write is always
  // Option A — that is how a split-convention folder gets normalised (§6.6).
  const existing = loadSidecarMetadata(sample.audioPath);
  let merged = mergeFields(existing, incoming, policy);
  if (deriveInstrumental) {
    // Lyrics PRESENT is evidence the track is not instrumental — unless what
    // they say is "this IS an instrumental" (Genius stock sentence, or a lone
    // [Instrumental] tag), which is evidence of exactly the opposite. Lyrics
    // ABSENT is not evidence of anything — a failed Genius lookup must not
    // brand a vocal track instrumental. Truly instrumental tracks are marked
    // by this rule, the user, or an is_instrumental already in their sidecar.
    // Direct set, NOT policy-mediated: the flag is derived from the winning
    // lyrics, and it must track them whatever the merge policy decided —
    // under `overwrite_caption` an is_instrumental left behind by an earlier
    // run is "occupied" and a policy-mediated write silently loses.
    const winning = (merged.lyrics ?? '').trim();
    if (winning) {
      if (lyricsSayInstrumental(winning)) {
        // The boilerplate is not lyrics — normalise to the canonical tag so
        // the model is never trained to sing "This song is an instrumental".
        merged = { ...merged, is_instrumental: 'true', lyrics: '[Instrumental]' };
      } else {
        merged = { ...merged, is_instrumental: 'false' };
      }
    }
  }
  // `forced` bypasses the merge policy — for values whose precedence was
  // already decided elsewhere: fields the user DECLARED (dataset language)
  // and the arbitrated genre (arbitrateSidecarGenre consumed the existing
  // value as input, so the occupancy check has nothing left to protect).
  if (forced) merged = { ...merged, ...forced };
  await writeSidecar(sample.sidecarPath, merged);
}

function refreshCounters(ds: TrainingDatasetRow, samples: TrainingSample[], status?: string): void {
  const labeled = samples.filter(s => !!s.caption.trim()).length;
  const excluded = samples.filter(s => s.excluded).length;
  // Always leave 'labeling' behind when a job ends — otherwise a partially
  // labeled dataset would sit in the running state forever. A dataset that has
  // already been built stays 'built': the route's syncCounters guards on that
  // status and could never restore it once we dropped it.
  const alreadyBuilt = !!ds.builtAt && !!ds.datasetJsonPath;
  const next = status ?? (alreadyBuilt ? 'built' : (labeled > 0 ? 'labeled' : 'draft'));
  try {
    updateCounters(ds.id, {
      sampleCount: samples.length,
      labeledCount: labeled,
      excludedCount: excluded,
      status: next,
    });
  } catch (err: any) {
    console.warn(`[Training] Counter update failed for ${ds.id}: ${err.message}`);
  }
}

/** Resolve the job's target samples, in scan order. */
async function loadTargets(
  job: TrainingJob,
  ds: TrainingDatasetRow,
): Promise<{ all: TrainingSample[]; targets: TrainingSample[] }> {
  const all = await buildSamples(ds, { withDuration: false });
  const wanted = new Set(job.sampleIds);
  const targets = all.filter(s => wanted.has(s.sampleId));
  return { all, targets };
}

// 'pending'/'processing' are live-job state and stay in memory (§2.0); only the
// durable outcome ('labeled'/'error') is written to the label record.
function markPending(job: TrainingJob, ds: TrainingDatasetRow, targets: TrainingSample[]): void {
  for (const s of targets) {
    setTransientStatus(job.datasetId, s.sampleId, 'pending');
    patchLabel(ds.slug, s.sampleId, { relPath: s.relPath, error: null });
    pushEvent(job, { type: 'sample', sampleId: s.sampleId, status: 'pending' });
  }
}

function markProcessing(job: TrainingJob, ds: TrainingDatasetRow, sample: TrainingSample): void {
  setTransientStatus(job.datasetId, sample.sampleId, 'processing');
  patchLabel(ds.slug, sample.sampleId, { relPath: sample.relPath, error: null });
  pushEvent(job, { type: 'sample', sampleId: sample.sampleId, status: 'processing' });
}

function markError(job: TrainingJob, ds: TrainingDatasetRow, sample: TrainingSample, message: string): void {
  clearTransientStatus(job.datasetId, sample.sampleId);
  patchLabel(ds.slug, sample.sampleId, { relPath: sample.relPath, labelStatus: 'error', error: message });
  job.failed++;
  pushEvent(job, { type: 'sample', sampleId: sample.sampleId, status: 'error', error: message });
  emitLog(job, 'error', `${sample.filename}: ${message}`);
}

function markLabeled(
  job: TrainingJob, ds: TrainingDatasetRow, sample: TrainingSample,
  sources: Record<string, FieldSource>,
  lyricsError: string | null = null,
): void {
  clearTransientStatus(job.datasetId, sample.sampleId);
  patchLabel(ds.slug, sample.sampleId, {
    relPath: sample.relPath,
    labelStatus: 'labeled',
    error: null,
    labeledAt: new Date().toISOString(),
    sources,
    lyricsError,
  });
  const fresh = refreshSample(ds, sample);
  pushEvent(job, { type: 'sample', sampleId: sample.sampleId, status: fresh.labelStatus, sample: fresh });
}

// ── Label job (§4.7) ─────────────────────────────────────────────────────

/** Does this label job have a per-sample cloud/GPU lane after the CPU pass? */
function hasCloudLane(opts: LabelOptions): boolean {
  return opts.useUnderstand === true || opts.useGenius === true || opts.useCaption === true;
}

interface PassAResult {
  duration: number;
  tags: { artist: string; title: string; album: string };
  tagGenre: string;
  tagBpm: number | null;
  essentiaBpm: number | null;
  essentiaKey: string;
}

async function runPassA(
  job: TrainingJob,
  ds: TrainingDatasetRow,
  sample: TrainingSample,
  opts: LabelOptions,
  policy: MergePolicy,
  results: Map<string, PassAResult>,
): Promise<void> {
  // When a cloud lane runs concurrently it owns job.phase/currentSampleId
  // (that progress is what the user watches); the CPU lane reports per-sample only.
  if (!hasCloudLane(opts)) {
    job.currentSampleId = sample.sampleId;
    job.phase = 'essentia';
  }
  markProcessing(job, ds, sample);

  const meta = await audioMeta.read(sample.audioPath);

  // Essentia is deterministic — if we already analyzed this exact file
  // (size+mtime unchanged since the cached run), reuse the raw result from the
  // label store instead of re-running ffmpeg + Essentia on a re-label.
  const cached = readLabel(ds.slug, sample.sampleId);
  const cacheFresh = (() => {
    if (!cached?.durationCache) return false;
    try {
      const st = fs.statSync(sample.audioPath);
      return cached.durationCache.sizeBytes === st.size && cached.durationCache.mtimeMs === st.mtimeMs;
    } catch {
      return false;
    }
  })();
  const essentia = (opts.useEssentia !== false && essentiaAvailable())
    ? (cacheFresh && cached?.essentia
        ? cached.essentia
        : await analyzeWithEssentia(sample.audioPath, job.controller.signal))
    : null;

  const essentiaKey = essentia ? essentiaKeyString(essentia.key, essentia.scale) : '';
  const result: PassAResult = {
    duration: meta.duration,
    tags: { artist: meta.artist, title: meta.title, album: meta.album },
    tagGenre: meta.genre,
    tagBpm: meta.bpm,
    essentiaBpm: essentia?.bpm ?? null,
    essentiaKey,
  };
  results.set(sample.sampleId, result);

  let sizeBytes = sample.sizeBytes;
  let mtimeMs = 0;
  try {
    const st = fs.statSync(sample.audioPath);
    sizeBytes = st.size;
    mtimeMs = st.mtimeMs;
  } catch { /* stat failure just costs us the duration cache */ }

  patchLabel(ds.slug, sample.sampleId, {
    relPath: sample.relPath,
    tags: result.tags,
    essentia: essentia ?? null,
    durationCache: meta.duration > 0 ? { seconds: meta.duration, sizeBytes, mtimeMs } : null,
  });

  // Interim sidecar write — a cancelled job still leaves usable BPM/key data.
  const incoming: Record<string, string> = {};
  const sources: Record<string, FieldSource> = {};
  if (result.essentiaBpm !== null) { incoming.bpm = String(result.essentiaBpm); sources.bpm = 'essentia'; }
  else if (result.tagBpm !== null) { incoming.bpm = String(result.tagBpm); sources.bpm = 'tags'; }
  if (essentiaKey) { incoming.key = essentiaKey; sources.key = 'essentia'; }
  if (meta.genre) { incoming.genre = meta.genre; sources.genre = 'tags'; }
  if (ds.defaultLanguage) { sources.language = 'user'; }
  if (meta.duration > 0) incoming.duration = meta.duration.toFixed(2);

  try {
    if (Object.keys(incoming).length > 0 || ds.defaultLanguage) {
      await mergeIntoSidecar(sample, incoming, policy, false,
        ds.defaultLanguage ? { language: ds.defaultLanguage } : undefined);
    }
  } catch (err: any) {
    markError(job, ds, sample, `Could not write sidecar: ${err.message}`);
    return;
  }

  // With no cloud lane this file is finished; otherwise keep it queued.
  if (!hasCloudLane(opts)) {
    markLabeled(job, ds, sample, sources);
    job.done++;
  } else {
    patchLabel(ds.slug, sample.sampleId, { sources });
    // The cloud lane still owes this file — hand it back to the queue rather
    // than leaving every row showing "Working…" (and read-only) until then.
    setTransientStatus(job.datasetId, sample.sampleId, 'pending');
    const fresh = refreshSample(ds, sample);
    pushEvent(job, { type: 'sample', sampleId: sample.sampleId, status: 'pending', sample: fresh });
  }
  emitProgress(job);
}

function understandParams(opts: LabelOptions): Partial<AceRequest> {
  const u = opts.understand || {};
  const params: Partial<AceRequest> = {};
  if (u.lmModel) params.lm_model = u.lmModel;
  if (u.synthModel) params.synth_model = u.synthModel;
  if (typeof u.lmTemperature === 'number') params.lm_temperature = u.lmTemperature;
  if (typeof u.lmTopP === 'number') params.lm_top_p = u.lmTopP;
  if (typeof u.lmTopK === 'number') params.lm_top_k = u.lmTopK;
  if (typeof u.seed === 'number') params.seed = u.seed;
  return params;
}

async function runLabelJob(job: TrainingJob): Promise<void> {
  if (isCancelled(job)) return;
  const ds = getDataset(job.datasetId);
  if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

  const opts = (job.opts || {}) as LabelOptions;
  const policy: MergePolicy = opts.mergePolicy || 'fill_missing';

  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'essentia';
  emitJob(job);

  try {
    const { targets } = await loadTargets(job, ds);
    job.total = targets.length;
    markPending(job, ds, targets);
    pushLog(`[Training] Label job ${job.id} started — ${job.total} files`);
    emitLog(job, 'info', `Labeling ${job.total} file(s)`);
    emitProgress(job);

    // ── Pass A: tags + Essentia (local CPU)  ──
    // ── Pass B: Genius + LLM caption (network) ──
    // The two lanes run CONCURRENTLY — Essentia/ffmpeg never touch the network,
    // so Genius/caption start on file 1 as soon as its Pass A is done instead of
    // after the whole folder is analyzed. Per-sample ordering is preserved (B
    // awaits A per file), which keeps the §4.8 merge precedence and makes the
    // two lanes' sidecar writes race-free.
    // Both lanes are now worker pools sized from rateLimit.ts rather than the
    // old fixed 2-and-1.
    const passA = new Map<string, PassAResult>();
    const passAResolvers = new Map<string, () => void>();
    const passADone = new Map<string, Promise<void>>();
    for (const t of targets) {
      passADone.set(t.sampleId, new Promise<void>(r => passAResolvers.set(t.sampleId, r)));
    }
    let cursor = 0;
    const worker = async () => {
      for (;;) {
        if (isCancelled(job)) return;
        const index = cursor++;
        if (index >= targets.length) return;
        const sample = targets[index];
        try {
          await runPassA(job, ds, sample, opts, policy, passA);
        } catch (err: any) {
          if (!isCancelled(job)) {
            markError(job, ds, sample, err?.message || 'Analysis failed');
            emitProgress(job);
          }
        } finally {
          passAResolvers.get(sample.sampleId)?.();
        }
      }
    };
    const passAPool = Math.max(1, essentiaLimiter.spec.concurrency);
    const passALane = Promise.all(
      Array.from({ length: passAPool }, () => worker()),
    ).then(() => {
      // Release anything the workers never reached (cancellation) so the
      // understand lane observes cancellation instead of awaiting forever.
      for (const release of passAResolvers.values()) release();
    });

    // The cloud lane runs any of three per-sample steps in sequence:
    //   understand (LEGACY, default off — 2026-07-27 pivot: its captions/lyrics
    //   are poor; audio_codes extraction moves to the preprocess phase)
    //   → Genius lyrics → LLM caption+genre (audio-grounded on gemini).
    // All results accumulate into ONE sidecar write per sample.
    const cloudLane = hasCloudLane(opts);
    const passBLane = (async () => {
      if (!cloudLane) return;
      let engineFailures = 0;

      // Sample-level pool. Each worker drives ONE sample through the whole
      // Genius -> caption chain (the caption prompt embeds the freshest lyrics,
      // so those two stay ordered per sample); the per-service limiters cap how
      // many of each call are actually in flight across all workers.
      //
      // useUnderstand pins the pool to 1: that path talks to the engine's single
      // FIFO GPU worker and keeps a consecutive-failure counter that only means
      // anything when nothing else is racing it.
      const passBPool = opts.useUnderstand === true
        ? 1
        : Math.max(1, geniusLimiter.spec.concurrency, captionLimiter.spec.concurrency);
      let passBCursor = 0;

      const passBWorker = async (): Promise<void> => {
       for (;;) {
        if (isCancelled(job)) return;
        const passBIndex = passBCursor++;
        if (passBIndex >= targets.length) return;
        const sample = targets[passBIndex];
        await passADone.get(sample.sampleId);
        if (isCancelled(job)) return;
        job.currentSampleId = sample.sampleId;
        markProcessing(job, ds, sample);
        emitProgress(job);

        const a = passA.get(sample.sampleId);
        // `targets` was snapshotted BEFORE Pass A read the file tags — on a
        // fresh dataset sample.tagArtist/tagTitle are empty, which sent Genius
        // hunting with mangled filename guesses. Use Pass A's fresh tags.
        const enriched: TrainingSample = a ? {
          ...sample,
          tagArtist: a.tags.artist || sample.tagArtist,
          tagTitle: a.tags.title || sample.tagTitle,
          tagAlbum: a.tags.album || sample.tagAlbum,
        } : sample;
        const incoming: Record<string, string> = {};
        const sources: Record<string, FieldSource> = {};
        const failures: string[] = [];
        // Persisted onto the label record below — see SampleLabelRecord.lyricsError.
        let lyricsError: string | null = null;

        // Local results first — Essentia wins bpm/key (§4.8), tags win genre.
        if (a?.essentiaBpm != null) { incoming.bpm = String(a.essentiaBpm); sources.bpm = 'essentia'; }
        else if (a?.tagBpm != null) { incoming.bpm = String(a.tagBpm); sources.bpm = 'tags'; }
        if (a?.essentiaKey) { incoming.key = a.essentiaKey; sources.key = 'essentia'; }
        if (a?.tagGenre) { incoming.genre = a.tagGenre; sources.genre = 'tags'; }
        if (ds.defaultLanguage) sources.language = 'user';
        if (a && a.duration > 0) incoming.duration = a.duration.toFixed(2);

        try {
          // ── Step 1 (legacy): /understand ──
          if (opts.useUnderstand === true) {
            job.phase = 'understand';
            emitProgress(job);
            try {
              const result = await runUnderstand(sample.audioPath, understandParams(opts), {
                signal: job.controller.signal,
                onQueue: (depth) => { job.engineQueueDepth = depth; emitProgress(job); },
                onPhase: (p) => { job.phase = p; emitProgress(job); },
              });
              engineFailures = 0;
              if (!incoming.bpm && result.bpm != null) { incoming.bpm = String(Math.trunc(result.bpm)); sources.bpm = 'understand'; }
              if (!incoming.key && result.keyscale) { incoming.key = result.keyscale; sources.key = 'understand'; }
              if (result.caption) { incoming.caption = result.caption; sources.caption = 'understand'; }
              if (result.lyrics) { incoming.lyrics = result.lyrics; sources.lyrics = 'understand'; }
              if (result.timesignature) { incoming.signature = result.timesignature; sources.signature = 'understand'; }
              if (!ds.defaultLanguage && result.vocalLanguage) { incoming.language = result.vocalLanguage; sources.language = 'understand'; }
              // audio_codes NEVER enters the sidecar — it goes here (D7).
              patchLabel(ds.slug, sample.sampleId, {
                relPath: sample.relPath,
                audioCodes: result.audioCodes,
                audioCodesModel: opts.understand?.lmModel || null,
                understand: {
                  caption: result.caption,
                  lyrics: result.lyrics,
                  bpm: result.bpm,
                  keyscale: result.keyscale,
                  timesignature: result.timesignature,
                  vocalLanguage: result.vocalLanguage,
                  duration: result.duration,
                  seed: result.seed,
                },
              });
            } catch (err: any) {
              if (err instanceof AbortError || isCancelled(job)) return;
              failures.push(`understand: ${err?.message || 'failed'}`);
              emitLog(job, 'warn', `${sample.filename}: understand failed — ${err?.message || err}`);
              if (isEngineConnectionError(err)) {
                engineFailures++;
                if (engineFailures >= ENGINE_FAILURE_LIMIT) {
                  // Stop the CPU lane too — the job is over.
                  job.controller.abort();
                  finishJob(job, 'failed', 'Engine became unreachable');
                  return;
                }
              } else {
                engineFailures = 0;
              }
            }
          }

          // ── Step 2: Genius lyrics ──
          if (opts.useGenius === true) {
            job.phase = 'genius';
            emitProgress(job);
            try {
              // The limiter supplies both the courtesy spacing and the in-flight
              // cap, so there is no sleep after the call any more — pacing is on
              // call STARTS, which is what a quota bucket actually counts.
              const hit = await geniusLimiter.run(
                () => enhanceGenius(enriched, ds, { sanitizeHeaders: true }));
              if (hit) {
                incoming.lyrics = hit.lyrics;
                sources.lyrics = 'genius';
                lyricsError = null;
              } else {
                const { artist, title } = resolveArtistTitle(enriched, ds, {});
                lyricsError = `no Genius match ("${artist} — ${title}")`;
                failures.push(lyricsError);
                emitLog(job, 'warn', `${sample.filename}: ${lyricsError}`);
              }
            } catch (err: any) {
              if (err instanceof AbortError || isCancelled(job)) return;
              lyricsError = err instanceof GeniusNoArtistError
                ? `genius skipped: ${err.message}`
                : `genius: ${err?.message || 'failed'}`;
              failures.push(lyricsError);
              emitLog(job, 'warn', `${sample.filename}: Genius failed — ${err?.message || err}`);
            }
          }

          // ── Step 3: LLM caption + genre (hears the audio on gemini) ──
          // Genre is arbitrated, not merged: Pass A seeds the file's rip tag
          // into the sidecar before this step runs, so under every policy but
          // overwrite_all the field is "occupied" and a policy-mediated write
          // from the provider silently loses. The arbitrated winner goes in
          // via `forced` — arbitration already weighed the existing value.
          let arbitratedGenre: string | undefined;
          if (opts.useCaption === true) {
            job.phase = 'llm';
            emitProgress(job);
            try {
              // Feed the freshest local results into the caption step, not just
              // the job-start snapshot. On a FIRST labeling run the snapshot has
              // no bpm/key — Pass A computes them DURING this job — so building
              // facts from `enriched` alone emitted MM3 Basic Attributes lines
              // with no `bpm is N. key is K` clause on every fresh dataset
              // (measured: 0/24 across americanfootball + chasestatus, while
              // the identical pipeline emits them once the sidecar is warm).
              // `incoming` already holds Pass A's essentia-first precedence.
              const promptSample: TrainingSample = {
                ...enriched,
                ...(incoming.lyrics ? { lyrics: incoming.lyrics } : {}),
                ...(incoming.bpm ? { bpm: Number(incoming.bpm) } : {}),
                ...(incoming.key ? { key: incoming.key } : {}),
                ...(incoming.signature ? { signature: incoming.signature } : {}),
              };
              const fields = await captionLimiter.run(() => enhanceCaption(promptSample, ds, {
                provider: opts.caption?.provider || config.lireek.defaultProvider,
                // Emit the MM3 Structured Caption alongside the AS1.5 block —
                // MOSS does it as a second decode, Gemini as a second audio
                // call. It is what `ace-train mm3-condition` reads.
                wantMm3: true,
                model: opts.caption?.model,
                includeLyricsExcerpt: true,
                temperature: 0.45,
                signal: job.controller.signal,
                log: (level, message) => emitLog(job, level, `${sample.filename}: ${message}`),
              }));
              if (fields.caption) { incoming.caption = fields.caption; sources.caption = 'llm'; }
              else failures.push('LLM returned no caption');
              // Never fight an explicit user edit — provenance says who wrote it.
              if (readLabel(ds.slug, sample.sampleId)?.sources?.genre !== 'user') {
                const existingGenre = (loadSidecarMetadata(sample.audioPath).genre ?? '').trim();
                const genre = arbitrateSidecarGenre({
                  providerGenre: fields.genre, providerCaption: fields.caption, existing: existingGenre,
                });
                if (genre && genre !== existingGenre) { arbitratedGenre = genre; sources.genre = 'llm'; }
              }
              if (!incoming.bpm && fields.bpm) { incoming.bpm = fields.bpm; sources.bpm = 'llm'; }
              if (!incoming.key && fields.key) { incoming.key = fields.key; sources.key = 'llm'; }
              if (!incoming.signature && fields.signature) { incoming.signature = fields.signature; sources.signature = 'llm'; }
            } catch (err: any) {
              if (err instanceof AbortError || isCancelled(job)) return;
              failures.push(`caption: ${err?.message || 'failed'}`);
              emitLog(job, 'warn', `${sample.filename}: caption failed — ${err?.message || err}`);
            }
          }

          job.phase = 'writing';
          // is_instrumental follows the WINNING lyrics (§4.8).
          const forced: Record<string, string> = {};
          if (ds.defaultLanguage) forced.language = ds.defaultLanguage;
          if (arbitratedGenre) forced.genre = arbitratedGenre;
          await mergeIntoSidecar(sample, incoming, policy, true,
            Object.keys(forced).length > 0 ? forced : undefined);

          // Error only when every cloud step came back empty-handed; partial
          // success is labeled with its warnings already in the job log.
          if (failures.length > 0 && !incoming.caption && !incoming.lyrics) {
            markError(job, ds, sample, failures.join('; '));
          } else {
            markLabeled(job, ds, sample, sources, incoming.lyrics ? null : lyricsError);
            job.done++;
          }
        } catch (err: any) {
          if (err instanceof AbortError || isCancelled(job)) return;
          markError(job, ds, sample, err?.message || 'Labeling failed');
        }
        emitProgress(job);
       }
      };

      // MOSS: caption the whole dataset in ONE process so the ~10 GB model is
      // loaded once instead of per track. Resolves as soon as the batch is
      // underway — Pass B consumes each track's caption the moment it lands
      // (per-track promises in mossCaption.ts), so the review grid fills
      // progressively rather than all at once when the process exits.
      // Placed here, after passALane has already started, so Essentia (CPU) runs
      // concurrently with captioning (GPU) rather than queueing behind it.
      if (opts.useCaption !== false && opts.caption?.provider === 'moss') {
        await prepareMossBatch(targets.map(t => t.audioPath), {
          wantMm3: true,
          signal: job.controller.signal,
          log: (level, message) => emitLog(job, level, message),
        });
      }

      await Promise.all(Array.from({ length: passBPool }, () => passBWorker()));
    })();

    await Promise.all([passALane, passBLane]);

    if (isCancelled(job)) return;
    if (job.status !== 'running') return;   // engine-unreachable already finished the job

    const finalSamples = await buildSamples(ds, { withDuration: false });
    refreshCounters(ds, finalSamples);
    pushLog(`[Training] Label job ${job.id} finished — ${job.done} labeled, ${job.failed} failed`);
    finishJob(job, 'done');
  } catch (err: any) {
    if (isCancelled(job)) return;
    console.error(`[Training] Label job ${job.id} FAILED — ${err?.message || err}`);
    finishJob(job, 'failed', err?.message || 'Label job failed');
  } finally {
    // Prefetched MOSS captions are scoped to this job — see runCaptionJob.
    clearMossBatch();
  }
}

// ── Genius job ───────────────────────────────────────────────────────────

async function runGeniusJob(job: TrainingJob): Promise<void> {
  if (isCancelled(job)) return;
  const ds = getDataset(job.datasetId);
  if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

  const opts = (job.opts || {}) as GeniusOptions;
  const policy: MergePolicy = opts.mergePolicy || 'overwrite_lyrics';

  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'genius';
  emitJob(job);

  try {
    const { targets } = await loadTargets(job, ds);
    job.total = targets.length;
    markPending(job, ds, targets);
    pushLog(`[Training] Genius job ${job.id} started — ${job.total} files`);
    emitProgress(job);

    // Pooled over samples; geniusLimiter enforces both the in-flight cap and the
    // spacing that used to be a fixed sleep at the end of each iteration.
    await runPooled(job, targets, geniusLimiter.spec.concurrency, async (sample) => {
      job.currentSampleId = sample.sampleId;
      markProcessing(job, ds, sample);

      try {
        const hit = await geniusLimiter.run(() => enhanceGenius(sample, ds, {
          artist: opts.artist,
          album: opts.album,
          sanitizeHeaders: opts.sanitizeHeaders !== false,
        }));
        if (!hit) {
          const { artist, title } = resolveArtistTitle(sample, ds, { artist: opts.artist });
          markError(job, ds, sample,
            artist && title ? `No Genius match (searched "${artist} — ${title}")` : 'No Genius match (no artist/title found)');
        } else {
          // deriveInstrumental, not a hardcoded 'false': Genius answers with
          // "This song is an instrumental" for instrumentals, and that hit
          // must set the flag rather than clear it.
          await mergeIntoSidecar(sample, { lyrics: hit.lyrics }, policy, true);
          markLabeled(job, ds, sample, { lyrics: 'genius' });
          job.done++;
        }
      } catch (err: any) {
        markError(job, ds, sample, err?.message || 'Genius lookup failed');
      }

      emitProgress(job);
    });

    if (isCancelled(job)) return;
    refreshCounters(ds, await buildSamples(ds, { withDuration: false }));
    finishJob(job, 'done');
  } catch (err: any) {
    if (isCancelled(job)) return;
    finishJob(job, 'failed', err?.message || 'Genius job failed');
  }
}

// ── LLM caption job ──────────────────────────────────────────────────────

async function runCaptionJob(job: TrainingJob): Promise<void> {
  if (isCancelled(job)) return;
  const ds = getDataset(job.datasetId);
  if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

  const opts = (job.opts || {}) as CaptionOptions & { provider: string };
  const policy: MergePolicy = opts.mergePolicy || 'overwrite_caption';

  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'llm';
  emitJob(job);

  try {
    const { targets } = await loadTargets(job, ds);
    job.total = targets.length;
    markPending(job, ds, targets);
    pushLog(`[Training] Caption job ${job.id} started — ${job.total} files (${opts.provider})`);
    emitProgress(job);

    // MOSS only: caption the whole dataset in ONE process, holding the ~10 GB
    // model across tracks instead of reloading it per sample. The pooled loop
    // below is unchanged — it consumes the prefetched results and keeps all its
    // per-sample merge policy, error marking, progress and cancellation.
    //
    // Scoped to this job, i.e. one dataset. A bulk run interleaves captioning
    // with LM/DiT training per dataset, and holding the captioner resident while
    // a trainer wants the card is the wrong trade — the process exits here.
    if (opts.provider === 'moss') {
      await prepareMossBatch(targets.map(s => s.audioPath), {
        wantMm3: opts.wantMm3 !== false,
        signal: job.controller.signal,
        log: (level, message) => emitLog(job, level, message),
      });
    }

    await runPooled(job, targets, captionLimiter.spec.concurrency, async (sample) => {
      job.currentSampleId = sample.sampleId;
      markProcessing(job, ds, sample);

      try {
        const fields = await captionLimiter.run(() => enhanceCaption(sample, ds, {
          provider: opts.provider,
          model: opts.model,
          includeLyricsExcerpt: opts.includeLyricsExcerpt !== false,
          temperature: typeof opts.temperature === 'number' ? opts.temperature : 0.45,
          // MOSS only — costs one extra decode off the same encode and writes
          // <stem>.mm3.txt, which is what `ace-train mm3-condition` reads.
          wantMm3: opts.wantMm3 !== false,
          signal: job.controller.signal,
          log: (level, message) => emitLog(job, level, `${sample.filename}: ${message}`),
        }));
        if (Object.keys(fields).length === 0) {
          markError(job, ds, sample, 'LLM returned nothing usable');
        } else {
          // Genre is arbitrated against the sidecar's current value (usually
          // the file's rip tag) and lands via `forced` — same reasoning as the
          // label job's Step 3: an occupied field beats a policy-mediated
          // write under every policy but overwrite_all. Both caption paths
          // must do this; wiring one does not wire the other.
          const { genre: providerGenre, ...rest } = fields;
          const sources: Record<string, FieldSource> = {};
          for (const key of Object.keys(rest)) sources[key] = 'llm';
          let forced: Record<string, string> | undefined;
          if (readLabel(ds.slug, sample.sampleId)?.sources?.genre !== 'user') {
            const existingGenre = (loadSidecarMetadata(sample.audioPath).genre ?? '').trim();
            const genre = arbitrateSidecarGenre({
              providerGenre, providerCaption: fields.caption, existing: existingGenre,
            });
            if (genre && genre !== existingGenre) { forced = { genre }; sources.genre = 'llm'; }
          }
          await mergeIntoSidecar(sample, rest, policy, false, forced);
          markLabeled(job, ds, sample, sources);
          job.done++;
        }
      } catch (err: any) {
        markError(job, ds, sample, err?.message || 'LLM call failed');
      }

      emitProgress(job);
    });

    if (isCancelled(job)) return;
    refreshCounters(ds, await buildSamples(ds, { withDuration: false }));
    finishJob(job, 'done');
  } catch (err: any) {
    if (isCancelled(job)) return;
    finishJob(job, 'failed', err?.message || 'Caption job failed');
  } finally {
    // However this ended — done, failed, cancelled — the prefetched captions are
    // scoped to this job. Leaving them would let a later job for the same file
    // replay a stale result instead of re-running the model.
    clearMossBatch();
  }
}

// ── Build job (D18) ──────────────────────────────────────────────────────

async function runBuildJob(job: TrainingJob): Promise<void> {
  if (isCancelled(job)) return;
  const ds = getDataset(job.datasetId);
  if (!ds) { finishJob(job, 'failed', 'Dataset not found'); return; }

  const opts = (job.opts || {}) as BuildOptions & { outputPath: string };

  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'build';
  emitJob(job);

  try {
    // Durations are measured for the build — this is the one path that pays.
    const samples = await buildSamples(ds, { withDuration: true });
    const included = samples.filter(s => !s.excluded);
    job.total = included.length;
    emitProgress(job);

    const missing = included.filter(s => s.fileMissing);
    if (missing.length > 0) {
      finishJob(job, 'failed', `${missing.length} included file(s) are missing from disk`);
      return;
    }

    const outputPath = opts.outputPath || path.join(ds.sourceDir, 'dataset.json');
    const result = await buildDataset(ds, included, outputPath, (done, total) => {
      job.done = done;
      job.total = total;
      emitProgress(job);
    });

    const builtAt = new Date().toISOString();
    updateDataset(ds.id, { builtAt, datasetJsonPath: result.path, status: 'built' });
    refreshCounters(ds, samples, 'built');

    pushLog(`[Training] Built ${result.path} — ${result.total} samples`);
    emitLog(job, 'info', `Wrote ${result.path} (${result.total} samples)`);
    finishJob(job, 'done');
  } catch (err: any) {
    if (isCancelled(job)) return;
    console.error(`[Training] Build job ${job.id} FAILED — ${err?.message || err}`);
    finishJob(job, 'failed', err?.message || 'Build failed');
  }
}

// ── Public starters ──────────────────────────────────────────────────────

export function startLabelJob(datasetId: string, sampleIds: string[], opts: LabelOptions): TrainingJob {
  const job = createJob('label', datasetId, sampleIds, opts);
  enqueue(job, runLabelJob);
  return job;
}

export function startGeniusJob(datasetId: string, sampleIds: string[], opts: GeniusOptions): TrainingJob {
  const job = createJob('enhance-genius', datasetId, sampleIds, opts);
  enqueue(job, runGeniusJob);
  return job;
}

export function startCaptionJob(
  datasetId: string,
  sampleIds: string[],
  opts: CaptionOptions & { provider: string },
): TrainingJob {
  const job = createJob('enhance-caption', datasetId, sampleIds, opts);
  enqueue(job, runCaptionJob);
  return job;
}

export function startBuildJob(datasetId: string, opts: BuildOptions & { outputPath: string }): TrainingJob {
  const job = createJob('build', datasetId, [], opts);
  enqueue(job, runBuildJob);
  return job;
}

/**
 * Preprocess (Training Studio phase 2) — spawns ace-train, which owns the GPU
 * for the whole run. The runner is imported lazily: preprocessRunner imports
 * emitProgress/finishJob/isCancelled from this module, and a top-level import
 * both ways is a cycle.
 */
export function startPreprocessJob(datasetId: string, sampleIds: string[], opts: unknown): TrainingJob {
  const job = createJob('preprocess', datasetId, sampleIds, opts);
  enqueue(job, async (j) => {
    const { runPreprocessJob } = await import('./preprocessRunner.js');
    await runPreprocessJob(j);
  });
  return job;
}

/**
 * LM LoRA training (Training Studio phase 3) — spawns ace-train, which owns the
 * whole card for the run. Same lazy import as preprocess: trainLmRunner imports
 * emitProgress/finishJob/isCancelled/killJobChild from this module, and a
 * top-level import both ways is a cycle.
 *
 * `sampleIds` is empty — the run's inputs are the preprocess variant's tensor
 * files, not dataset rows, so there is no per-sample row state to mark.
 */
export function startTrainLmJob(datasetId: string, opts: unknown): TrainingJob {
  const job = createJob('train-lm', datasetId, [], opts);
  enqueue(job, async (j) => {
    const { runTrainLmJob } = await import('./trainLmRunner.js');
    await runTrainLmJob(j);
  });
  return job;
}

/**
 * MiniMax-Music3 code export (S4) — spawns `ace-train mm3-codes`, which owns
 * the GPU for the run. Same lazy import as the other trainers: the runner
 * imports emitProgress/finishJob/isCancelled from this module.
 */
export function startMm3CodesJob(datasetId: string, opts: unknown): TrainingJob {
  const job = createJob('mm3-codes', datasetId, [], opts);
  enqueue(job, async (j) => {
    const { runMm3CodesJob } = await import('./mm3TrainRunner.js');
    await runMm3CodesJob(j);
  });
  return job;
}

/**
 * MiniMax-Music3 LM LoRA training (S4). `sampleIds` is empty: the inputs are
 * the codes cache and the caption sidecars, not dataset rows, so there is no
 * per-sample row state to mark.
 */
export function startMm3TrainLmJob(datasetId: string, opts: unknown): TrainingJob {
  const job = createJob('mm3-train-lm', datasetId, [], opts);
  enqueue(job, async (j) => {
    const { runMm3TrainLmJob } = await import('./mm3TrainRunner.js');
    await runMm3TrainLmJob(j);
  });
  return job;
}

/**
 * Post-training LM calibration (2026-08-10) — evals candidate adapters x
 * scales against the dataset's ground-truth codes, picks under guards, bakes
 * the winner and repoints presets. Needs ace-server UP (it drives /lm), so it
 * lives in the GPU lane strictly AFTER the train job whose `finally` restarts
 * the engine. Enqueued by trainLmRunner on success when opts.calibrate.
 */
export interface LmCalibrateOptions {
  /** The just-trained run dir the calibration is centered on. */
  newRunDir: string;
  /** Explicit rival ('' = auto: newest other non-calibrated run). */
  oldRunDir: string;
  variantKey: string;
  repoint: boolean;
}

export function startLmCalibrateJob(datasetId: string, opts: LmCalibrateOptions): TrainingJob {
  const job = createJob('lm-calibrate', datasetId, [], opts);
  enqueue(job, async (j) => {
    const { runLmCalibrateJob } = await import('./calibrateRunner.js');
    await runLmCalibrateJob(j);
  });
  return job;
}

/** DiT twin: latent-Frechet calibration after train-dit (2026-08-11).
 *  Same options shape; the runner picks the DiT script + preset column. */
export function startDitCalibrateJob(datasetId: string, opts: LmCalibrateOptions): TrainingJob {
  const job = createJob('dit-calibrate', datasetId, [], opts);
  enqueue(job, async (j) => {
    const { runDitCalibrateJob } = await import('./calibrateRunner.js');
    await runDitCalibrateJob(j);
  });
  return job;
}

/**
 * DiT LoRA training (Training Studio phase 4) — spawns ace-train, which owns
 * the whole card for the run (a ~15 GB F32 weight mirror plus the activation
 * arena). Same lazy import as train-lm: trainDitRunner imports
 * emitProgress/finishJob/isCancelled/killJobChild from this module, and a
 * top-level import both ways is a cycle.
 *
 * `sampleIds` is empty — the run's inputs are the preprocess variant's tensor
 * files, not dataset rows, so there is no per-sample row state to mark.
 */
export function startTrainDitJob(datasetId: string, opts: unknown): TrainingJob {
  const job = createJob('train-dit', datasetId, [], opts);
  enqueue(job, async (j) => {
    const { runTrainDitJob } = await import('./trainDitRunner.js');
    await runTrainDitJob(j);
  });
  return job;
}

/**
 * Codes audition (pure-LM preview) — the ONE training-studio job that needs
 * ace-server UP: /lm is the only adapter-capable LM host and /codes-decode lives
 * in the same process, so this runner never stops the engine. Same lazy import
 * as the trainers: auditionRunner imports emitProgress/finishJob/isCancelled
 * from this module, and a top-level import both ways is a cycle.
 *
 * `sampleIds` is empty — an audition's input is a caption/lyrics pair plus an
 * adapter, not a set of dataset rows.
 */
export function startAuditionJob(datasetId: string, opts: unknown): TrainingJob {
  const job = createJob('audition', datasetId, [], opts);
  enqueue(job, async (j) => {
    const { runAuditionJob } = await import('./auditionRunner.js');
    await runAuditionJob(j);
  });
  return job;
}
