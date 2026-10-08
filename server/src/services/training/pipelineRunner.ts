// training/pipelineRunner.ts — batch import orchestrator
//
// Runs N audio folders through create → label → build → preprocess → train-dit
// → train-lm with no user intervention, using the stored stage defaults.
//
// Two rules shape the whole file:
//   * stages are started by an INTERNAL HTTP POST to this server's own
//     /api/training routes — the route layer stays the single source of
//     validation and of the stage default literals (§3.2). Progress is then
//     read straight from the job map (no HTTP).
//   * the orchestrator never touches the engine (§3.6). Every runner already
//     owns its own ace-server stop/restart.
//
// One pipeline at a time; items run sequentially, which costs nothing — the job
// queue is one global serial chain anyway. A failed folder stops that folder
// only; the batch carries on.
//
// Spec: docs/plans/2026-07-28-training-batch-pipeline.md §3

import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { config } from '../../config.js';
import { createDatasetFromFolder } from './datasetCreate.js';
import { detailFor } from './datasetDetail.js';
import * as repo from './datasetsRepo.js';
import * as queue from './labelingQueue.js';
import { trainingBaseDir } from './paths.js';
import { getTrainingDefaults } from './trainingDefaults.js';
import type {
  PipelineLabelOptions,
  PipelineFolderSpec, PipelineItem, PipelineItemStatus, PipelineStage, PipelineStageResult,
  PipelineSummary, TrainingDefaults,
} from './types.js';

/** Canonical stage order. Any subset a caller asks for is filtered through it. */
// 'lyric-studio' runs LAST, after both trainers, because that is what makes it
// useful: the export links the run's freshly trained adapters onto the album
// preset (linkAdapters defaults true), so running it earlier would publish an
// artist/album page pointing at nothing. Added 2026-07-31 — a bulk run used to
// finish with trained adapters that no album preset referenced.
export const PIPELINE_STAGES: readonly PipelineStage[] =
  ['label', 'build', 'preprocess', 'train-dit', 'train-lm', 'lyric-studio'];

const STAGE_PATH: Record<PipelineStage, string> = {
  'label': 'label',
  'build': 'build',
  'preprocess': 'preprocess',
  'train-dit': 'train-dit',
  'train-lm': 'train-lm',
  'lyric-studio': 'lyric-studio',
};

/** Stages that complete SYNCHRONOUSLY — they return their result, not a jobId,
 *  so there is no queue job to poll. */
const SYNC_STAGES: ReadonlySet<PipelineStage> = new Set<PipelineStage>(['lyric-studio']);

/** Which stored-defaults section feeds each stage's POST body. Build takes no
 *  options beyond outputPath, which must stay the route's own default. */
const STAGE_DEFAULTS_KEY: Record<PipelineStage, keyof TrainingDefaults | ''> = {
  'label': 'label',
  'build': '',
  'preprocess': 'preprocess',
  'train-dit': 'trainDit',
  'train-lm': 'trainLm',
  // No stored section: artist/album are derived from the dataset's own tags and
  // linkAdapters already defaults true on the route.
  'lyric-studio': '',
};

const POLL_MS = 1500;
/** How long to wait for a job the USER started on the same dataset (§3.5). */
const IDLE_WAIT_MS = 10 * 60 * 1000;
const MAX_LISTED = 20;

interface PipelineState extends PipelineSummary {
  /** Set by cancelPipeline; every loop checks it between awaits. */
  cancelRequested: boolean;
  /** Per-item creation inputs — not part of the wire contract. */
  specs: PipelineFolderSpec[];
}

/** A pipeline the runner still owns: its loop is (or will be) making progress.
 *  'paused' counts — the loop is parked, not gone. */
function isActive(status: PipelineState['status']): boolean {
  return status === 'running' || status === 'paused';
}

const pipelines = new Map<string, PipelineState>();

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function toSummary(state: PipelineState): PipelineSummary {
  const { cancelRequested, specs, ...summary } = state;
  return summary;
}

// ── Persistence (§2.3) ───────────────────────────────────────────────────

function pipelinesDir(): string {
  return path.join(trainingBaseDir, 'pipelines');
}

/** What actually lives in a snapshot file: the wire summary plus the private
 *  per-item creation specs. Resume needs the specs (customTag is not derivable
 *  from the items); files written before specs were persisted still resume via
 *  the item-based fallback in revivePipeline. */
type SnapshotFile = PipelineSummary & { specs?: PipelineFolderSpec[] };

function writeSnapshotFile(snap: SnapshotFile): void {
  fs.mkdirSync(pipelinesDir(), { recursive: true });
  fs.writeFileSync(
    path.join(pipelinesDir(), `${snap.id}.json`),
    JSON.stringify(snap, null, 2),
    'utf-8',
  );
}

/** Snapshot on every item/stage transition — same spirit as jobs' _meta.json. */
function persist(state: PipelineState): void {
  try {
    writeSnapshotFile({ ...toSummary(state), specs: state.specs });
  } catch (err: any) {
    console.warn(`[Training] Could not persist pipeline ${state.id}: ${err.message}`);
  }
}

function readSnapshotFiles(): SnapshotFile[] {
  const out: SnapshotFile[] = [];
  try {
    const base = pipelinesDir();
    if (!fs.existsSync(base)) return out;
    for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
      try {
        out.push(JSON.parse(fs.readFileSync(path.join(base, entry.name), 'utf-8')) as SnapshotFile);
      } catch { /* skip corrupted snapshot */ }
    }
  } catch { /* pipelines dir unreadable — memory list is still valid */ }
  return out;
}

/** The wire view of the snapshots — specs stay private to the runner. */
function readSnapshots(): PipelineSummary[] {
  return readSnapshotFiles().map(({ specs: _specs, ...summary }) => summary);
}

/**
 * A server restart loses the in-memory loop, but the snapshot IS the work list
 * — every item, every stage's status, and the creation specs. Recover any
 * running/paused pipeline as 'paused' so Resume can rebuild the loop and carry
 * on from the last completed stage. The stage that was mid-flight when the
 * process died is reset to pending (its job died with the process) and re-runs
 * from scratch on resume.
 */
function recoverStalePipelines(): void {
  try {
    const base = pipelinesDir();
    if (!fs.existsSync(base)) return;
    for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
      const file = path.join(base, entry.name);
      try {
        const snap = JSON.parse(fs.readFileSync(file, 'utf-8')) as SnapshotFile;
        if (snap.status !== 'running' && snap.status !== 'paused') continue;
        snap.status = 'paused';
        snap.pauseRequested = false;
        snap.finishedAt = null;
        for (const item of snap.items ?? []) {
          for (const stage of item.stages ?? []) {
            if (stage.status === 'running' || stage.status === 'creating') {
              stage.status = 'pending';
              stage.jobId = '';
              stage.error = null;
              stage.startedAt = null;
              stage.finishedAt = null;
            }
          }
          if (item.status === 'creating' || item.status === 'running') {
            item.status = 'pending';
            item.currentStage = null;
          }
        }
        fs.writeFileSync(file, JSON.stringify(snap, null, 2), 'utf-8');
        console.log(`[Training] Pipeline ${snap.id} recovered as paused after restart — resume to continue`);
      } catch { /* skip corrupted snapshot */ }
    }
  } catch { /* pipelines dir unreadable */ }
}
recoverStalePipelines();

// ── Public API ───────────────────────────────────────────────────────────

let hasActiveYue2Preparation: () => boolean = () => false;

/** The YuE2 operation registers its durable activity without making the
 *  legacy batch runner depend on its implementation. Both share admission. */
export function registerYue2PreparationActivity(check: () => boolean): void {
  hasActiveYue2Preparation = check;
}

export function hasActivePipeline(): boolean {
  if (hasActiveYue2Preparation()) return true;
  for (const state of pipelines.values()) {
    if (isActive(state.status)) return true;
  }
  return false;
}

export function getPipeline(id: string): PipelineSummary | null {
  const state = pipelines.get(id);
  if (state) return toSummary(state);
  return readSnapshots().find(p => p.id === id) ?? null;
}

/** Active first, then the most recent snapshots from disk, max 20. */
export function listPipelines(): PipelineSummary[] {
  const live = [...pipelines.values()].map(toSummary);
  const seen = new Set(live.map(p => p.id));
  const disk = readSnapshots().filter(p => !seen.has(p.id));

  const byNewest = (a: PipelineSummary, b: PipelineSummary): number => b.createdAt - a.createdAt;
  const running = live.filter(p => isActive(p.status)).sort(byNewest);
  const rest = [...live.filter(p => !isActive(p.status)), ...disk].sort(byNewest);
  return [...running, ...rest].slice(0, MAX_LISTED);
}

export function startPipeline(
  folders: PipelineFolderSpec[], stages: PipelineStage[], labelOptions?: PipelineLabelOptions,
): PipelineSummary {
  const state: PipelineState = {
    id: randomUUID(),
    status: 'running',
    stages: [...stages],
    ...(labelOptions && Object.keys(labelOptions).length > 0 ? { labelOptions } : {}),
    items: folders.map(spec => ({
      sourceDir: spec.sourceDir,
      name: (spec.name || '').trim() || path.basename(path.resolve(spec.sourceDir)) || spec.sourceDir,
      datasetId: '',
      reusedExisting: false,
      status: 'pending',
      currentStage: null,
      stages: stages.map(stage => ({
        stage, jobId: '', status: 'pending', error: null, startedAt: null, finishedAt: null,
      })),
      error: null,
    })),
    createdAt: Date.now(),
    finishedAt: null,
    pauseRequested: false,
    cancelRequested: false,
    specs: folders.map(spec => ({ ...spec })),
  };

  pipelines.set(state.id, state);
  persist(state);
  console.log(
    `[Training] Pipeline ${state.id} started — ${state.items.length} folder(s), stages: ${stages.join(' → ') || 'none'}`);
  // Fire and forget, one tick later: the first folder's create is a synchronous
  // folder scan, and running it inside the handler would hold up the 202.
  setImmediate(() => { void runPipeline(state); });
  return toSummary(state);
}

/** Cancels the running job, then marks everything still pending as cancelled. */
export function cancelPipeline(id: string): boolean {
  const state = pipelines.get(id);
  if (!state) {
    // Not owned by this process. A finished snapshot has nothing to cancel —
    // still a 200, since the id is real. But one recovered as 'paused' after a
    // restart must stay dismissable, so rewrite the file directly.
    const snap = readSnapshotFiles().find(p => p.id === id);
    if (!snap) return false;
    if (isActive(snap.status)) {
      snap.status = 'cancelled';
      snap.pauseRequested = false;
      snap.finishedAt = Date.now();
      for (const item of snap.items ?? []) {
        for (const stage of item.stages ?? []) {
          if (stage.status !== 'done' && stage.status !== 'failed') stage.status = 'cancelled';
        }
        if (item.status !== 'done' && item.status !== 'failed') item.status = 'cancelled';
        item.currentStage = null;
      }
      try {
        writeSnapshotFile(snap);
      } catch (err: any) {
        console.warn(`[Training] Could not cancel snapshot pipeline ${id}: ${err.message}`);
      }
    }
    return true;
  }
  // 'paused' must stay cancellable — the parked loop exits on cancelRequested.
  if (!isActive(state.status)) return true;

  state.cancelRequested = true;
  state.pauseRequested = false;
  for (const item of state.items) {
    for (const stage of item.stages) {
      if (stage.status === 'running') {
        // The runner's poll loop maps the cancelled job onto this stage; killing
        // the job here is what actually stops the work.
        if (stage.jobId) queue.cancelJob(stage.jobId);
      } else if (stage.status === 'pending') {
        stage.status = 'cancelled';
      }
    }
    if (item.status === 'pending') item.status = 'cancelled';
  }
  persist(state);
  console.log(`[Training] Pipeline ${id} cancel requested`);
  return true;
}

// ── Pause / resume ───────────────────────────────────────────────────────
//
// Pause is a STAGE-BOUNDARY hold, never a kill: the in-flight stage (which can
// be a multi-hour training job) runs to completion, then the loop parks before
// starting the next stage or item. Nothing about the current run is disturbed.
// `pauseRequested` is the user's intent; status 'paused' is the loop actually
// parked — the UI shows the gap as "pausing".

export type PipelinePauseResult = 'ok' | 'not_found' | 'not_active';
export type PipelineResumeResult = 'ok' | 'not_found' | 'busy';

export function pausePipeline(id: string): PipelinePauseResult {
  const state = pipelines.get(id);
  if (!state) return readSnapshots().some(p => p.id === id) ? 'not_active' : 'not_found';
  if (!isActive(state.status)) return 'not_active';
  if (!state.pauseRequested) {
    state.pauseRequested = true;
    persist(state);
    console.log(`[Training] Pipeline ${id} pause requested — holding at the next stage boundary`);
  }
  return 'ok';
}

export function resumePipeline(id: string): PipelineResumeResult {
  const state = pipelines.get(id);

  // Live and active: just lift the pause.
  if (state && isActive(state.status)) {
    if (state.pauseRequested || state.status === 'paused') {
      state.pauseRequested = false;
      // Flip status here rather than waiting for the parked loop's next poll,
      // so the UI's immediate refetch already sees 'running'. The loop's own
      // flip on wake-up is idempotent.
      if (state.status === 'paused') state.status = 'running';
      persist(state);
      console.log(`[Training] Pipeline ${id} resumed`);
    }
    return 'ok';
  }

  // Terminal in this session, or snapshot-only from a previous one (possibly
  // recovered as 'paused' after a restart): rebuild the loop from the record.
  const snap: SnapshotFile | undefined = state
    ? { ...toSummary(state), specs: state.specs }
    : readSnapshotFiles().find(p => p.id === id);
  if (!snap) return 'not_found';
  if (hasActivePipeline()) return 'busy';
  revivePipeline(snap);
  return 'ok';
}

/**
 * Resume-from-record: a pipeline is just a list of datasets × stages plus the
 * stored stage params, so any ended/cancelled/restart-killed pipeline can pick
 * up where it stopped. Everything 'done' stays done; every other item and
 * stage resets to pending and re-runs — failures retry too, since the most
 * common cause of a dead batch is a server restart, not a bad dataset. A stage
 * that was mid-flight re-runs from its own start; only that stage's progress
 * is lost.
 */
function revivePipeline(snap: SnapshotFile): void {
  const items: PipelineItem[] = snap.items.map(item => {
    if (item.status === 'done') return { ...item, stages: item.stages.map(s => ({ ...s })) };
    return {
      ...item,
      status: 'pending',
      error: null,
      currentStage: null,
      stages: item.stages.map(s => s.status === 'done'
        ? { ...s }
        : { stage: s.stage, jobId: '', status: 'pending', error: null, startedAt: null, finishedAt: null }),
    };
  });

  const state: PipelineState = {
    id: snap.id,
    status: 'running',
    stages: [...snap.stages],
    items,
    createdAt: snap.createdAt,
    finishedAt: null,
    pauseRequested: false,
    cancelRequested: false,
    // Snapshots from before specs were persisted rebuild them from the items:
    // only customTag is lost, and it only mattered at dataset creation — any
    // item that got that far already has its dataset.
    specs: snap.specs && snap.specs.length === snap.items.length
      ? snap.specs.map(s => ({ ...s }))
      : snap.items.map(i => ({ sourceDir: i.sourceDir, name: i.name })),
  };

  pipelines.set(state.id, state);
  persist(state);
  const remaining = items.filter(i => i.status !== 'done').length;
  console.log(
    `[Training] Pipeline ${state.id} resumed from record — ${remaining} of ${items.length} folder(s) left to run`);
  setImmediate(() => { void runPipeline(state); });
}

/** Parks the loop while a pause is requested. Called only at stage/item
 *  boundaries. Returns on resume or cancel; the caller re-checks cancel. */
async function waitWhilePaused(state: PipelineState): Promise<void> {
  if (!state.pauseRequested || state.cancelRequested) return;
  state.status = 'paused';
  persist(state);
  console.log(`[Training] Pipeline ${state.id} paused`);
  while (state.pauseRequested && !state.cancelRequested) {
    await sleep(POLL_MS);
  }
  if (!state.cancelRequested && state.status === 'paused') {
    state.status = 'running';
    persist(state);
  }
}

// ── Orchestration (§3.1) ─────────────────────────────────────────────────

async function runPipeline(state: PipelineState): Promise<void> {
  try {
    for (let i = 0; i < state.items.length; i++) {
      const item = state.items[i];
      // Resumed pipeline: items completed in an earlier run stay done.
      if (item.status === 'done') continue;
      await waitWhilePaused(state);
      if (state.cancelRequested) {
        cancelItem(item);
        continue;
      }
      await runItem(state, item, state.specs[i]);
    }
  } catch (err: any) {
    // A throw here would leave the pipeline 'running' forever — the finaliser
    // below has to run whatever happened.
    console.error(`[Training] Pipeline ${state.id} crashed: ${err?.message || err}`);
  }

  state.status = state.cancelRequested
    ? 'cancelled'
    : (state.items.some(i => i.status === 'failed') ? 'failed' : 'done');
  state.finishedAt = Date.now();
  persist(state);
  console.log(`[Training] Pipeline ${state.id} ${state.status}`);
}

function cancelItem(item: PipelineItem): void {
  for (const stage of item.stages) {
    if (stage.status === 'pending' || stage.status === 'running') stage.status = 'cancelled';
  }
  if (item.status === 'pending' || item.status === 'creating' || item.status === 'running') {
    item.status = 'cancelled';
  }
  item.currentStage = null;
}

function failItem(state: PipelineState, item: PipelineItem, error: string): void {
  item.status = 'failed';
  item.error = error;
  item.currentStage = null;
  for (const stage of item.stages) {
    if (stage.status === 'pending') stage.status = 'cancelled';
  }
  persist(state);
  console.warn(`[Training] Pipeline ${state.id}: ${item.name} failed — ${error}`);
}

async function runItem(state: PipelineState, item: PipelineItem, spec: PipelineFolderSpec): Promise<void> {
  // ── 1. dataset create/reuse (§3.3) — direct service calls, not HTTP ────
  const sourceDir = path.resolve(item.sourceDir);
  if (!fs.existsSync(sourceDir) || !fs.statSync(sourceDir).isDirectory()) {
    failItem(state, item, `Folder does not exist: ${item.sourceDir}`);
    return;
  }

  const existing = repo.getDatasetBySourceDir(sourceDir);
  if (existing) {
    item.datasetId = existing.id;
    item.reusedExisting = true;
    item.name = existing.name;
  } else {
    item.status = 'creating';
    persist(state);
    try {
      const detail = await createDatasetFromFolder({
        name: item.name,
        sourceDir,
        ...(spec.customTag ? { customTag: spec.customTag } : {}),
      });
      item.datasetId = detail.id;
    } catch (err: any) {
      failItem(state, item, err?.message || String(err));
      return;
    }
  }

  // ── 2. stages ─────────────────────────────────────────────────────────
  item.status = 'running';
  persist(state);

  for (const result of item.stages) {
    // Resumed pipeline: stages completed in an earlier run stay done.
    if (result.status === 'done') continue;
    await waitWhilePaused(state);
    if (state.cancelRequested) {
      cancelItem(item);
      persist(state);
      return;
    }
    await runStage(state, item, result);

    // Fresh read: TS still holds the `!== 'done'` narrowing from the skip
    // above and can't see runStage's mutation of result.status.
    const outcome = result.status as PipelineItemStatus;
    if (outcome !== 'done') {
      if (outcome === 'cancelled') {
        cancelItem(item);
        persist(state);
      } else {
        failItem(state, item, `${result.stage}: ${result.error || 'failed'}`);
      }
      return;
    }

    // ── Quality gate (§3.4): build and train would happily run on empty
    // captions, and the result is silent garbage nobody notices until it is
    // trained. Only a fresh scan can answer this.
    if (result.stage === 'label') {
      const gate = await labelGateError(item.datasetId);
      if (gate) {
        failItem(state, item, gate);
        return;
      }
    }
  }

  item.status = 'done';
  item.currentStage = null;
  persist(state);
}

/** `null` when the dataset has at least one caption after labeling. */
async function labelGateError(datasetId: string): Promise<string | null> {
  const ds = repo.getDataset(datasetId);
  if (!ds) return 'Dataset disappeared';
  try {
    const detail = await detailFor(ds);
    if (detail.labeledCount > 0) return null;
  } catch (err: any) {
    return `Could not re-scan after labeling: ${err?.message || err}`;
  }
  return 'Labeling produced no labeled samples — check labeling capabilities';
}

// ── One stage (§3.2) ─────────────────────────────────────────────────────

async function runStage(
  state: PipelineState,
  item: PipelineItem,
  result: PipelineStageResult,
): Promise<void> {
  result.status = 'running';
  result.startedAt = Date.now();
  item.currentStage = result.stage;
  persist(state);

  // §3.5 — a job the USER started on this dataset would make the route answer
  // 409. Wait it out instead of consuming that as a stage failure.
  await waitForDatasetIdle(state, item.datasetId);
  if (state.cancelRequested) {
    finishStage(state, result, 'cancelled', null);
    return;
  }

  const section = STAGE_DEFAULTS_KEY[result.stage];
  const body: Record<string, unknown> = section ? { ...getTrainingDefaults()[section] } : {};
  // A dataset that arrives already labelled is a DONE label stage, not a failed
  // one. Without this the route answers 400 "Nothing to label" and the item is
  // abandoned before preprocess/train ever run (2026-07-31).
  if (result.stage === 'label') {
    body.allowEmpty = true;
    // Per-run overrides ride ON TOP of the stored defaults — a bulk re-caption
    // skips Genius/Essentia and forces overwrite_caption without touching the
    // defaults every other run reads. Absent field = default decides.
    const lo = state.labelOptions;
    if (lo) {
      if (lo.scope) body.scope = lo.scope;
      if (typeof lo.useEssentia === 'boolean') body.useEssentia = lo.useEssentia;
      if (typeof lo.useGenius === 'boolean') body.useGenius = lo.useGenius;
      if (typeof lo.useCaption === 'boolean') body.useCaption = lo.useCaption;
      if (lo.mergePolicy) body.mergePolicy = lo.mergePolicy;
    }
  }
  const url = `http://127.0.0.1:${config.server.port}`
    + `/api/training/datasets/${encodeURIComponent(item.datasetId)}/${STAGE_PATH[result.stage]}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err: any) {
    finishStage(state, result, 'failed', err?.message || String(err));
    return;
  }

  if (!response.ok) {
    // The route's own messages are already user-legible — pass them through.
    finishStage(state, result, 'failed', await errorTextOf(response));
    return;
  }

  let jobId = '';
  let skipped = '';
  try {
    const payload = await response.json() as { jobId?: unknown; skipped?: unknown };
    if (typeof payload?.jobId === 'string') jobId = payload.jobId;
    if (typeof payload?.skipped === 'string') skipped = payload.skipped;
  } catch { /* handled by the empty check below */ }
  // A 200 with no jobId and an explicit `skipped` is a stage that had nothing to
  // do — complete, not failed, and the item carries on to the next stage.
  if (!jobId && skipped) {
    finishStage(state, result, 'done', null);
    return;
  }
  // Synchronous stages have already finished by the time they answer 200.
  if (SYNC_STAGES.has(result.stage)) {
    finishStage(state, result, 'done', null);
    return;
  }
  if (!jobId) {
    finishStage(state, result, 'failed', 'Stage returned no jobId');
    return;
  }
  result.jobId = jobId;
  persist(state);

  await pollStageJob(state, result, jobId);
}

async function waitForDatasetIdle(state: PipelineState, datasetId: string): Promise<void> {
  const deadline = Date.now() + IDLE_WAIT_MS;
  while (queue.activeJobForDataset(datasetId) && !state.cancelRequested && Date.now() < deadline) {
    await sleep(POLL_MS);
  }
}

async function pollStageJob(state: PipelineState, result: PipelineStageResult, jobId: string): Promise<void> {
  for (;;) {
    if (state.cancelRequested) queue.cancelJob(jobId);

    const job = queue.getJob(jobId);
    // cancelJob DELETES the job from the map (ours above, or a user hitting
    // DELETE /jobs/:id) — a job that was there a moment ago and is now gone was
    // cancelled, not lost.
    if (!job) {
      finishStage(state, result, 'cancelled', null);
      return;
    }
    if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') {
      finishStage(state, result, job.status, job.error);
      return;
    }
    await sleep(POLL_MS);
  }
}

function finishStage(
  state: PipelineState,
  result: PipelineStageResult,
  status: 'done' | 'failed' | 'cancelled',
  error: string | null,
): void {
  result.status = status;
  result.error = status === 'done' ? null : error;
  result.finishedAt = Date.now();
  persist(state);
}

async function errorTextOf(response: Response): Promise<string> {
  try {
    const payload = await response.json() as { error?: unknown };
    if (typeof payload?.error === 'string' && payload.error) return payload.error;
  } catch { /* not a JSON body */ }
  return `Stage request failed (HTTP ${response.status})`;
}
