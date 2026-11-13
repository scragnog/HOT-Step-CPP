// yue2BatchRunner.ts — durable YuE2 joint and HeartMuLa training queues.
//
// The AS1.5 batch pipeline (pipelineRunner.ts) is the model: the batch is a
// record on disk under <training>/yue2-batches/<id>.json, re-persisted on
// every item/stage transition, recovered as `paused` when the server comes
// back, and resumed from the record with everything already done kept done.
// The old Yue2AitkBatchWizard was a React loop that died with its tab.
//
// One recipe for the whole batch: the same Yue2JointTrainRequest the single
// training card sends, minus the per-dataset paths, applied to every dataset.
// Stages start by POSTing this server's own routes so route validation stays
// the single source of truth, and each stage is skipped when the dataset's
// own status route already reports it done — the same rule the single-dataset
// "perform all stages" chain applies client-side.

import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { config } from '../../config.js';
import * as repo from './datasetsRepo.js';
import * as queue from './labelingQueue.js';
import { trainingBaseDir } from './paths.js';
import { isCheckpointLadder, listYue2AitkRuns, type Yue2AitkRunRecord } from './yue2AitkRuns.js';
import { autoRefineRequest } from './yue2JointTrainRunner.js';
import { YUE2_CACHE_LUFS } from './yue2Train.js';
import { yue2CodesReady, yue2TokenizerSetting } from './yue2Tokenize.js';
import { listPreparedCaches } from './preparedDataReset.js';
import { samplesMissingYue2Caption } from './yue2CaptionJob.js';
import { bestScoredRung } from './yue2BestRung.js';
import { listYue2RungScores } from './yue2RungScores.js';
import { runYue2Cleanup } from './yue2Cleanup.js';
import { refreshYue2PresetsForJointCheckpoint } from './lyricStudioExport.js';
import { hasActivePipeline } from './pipelineRunner.js';
import { describeOwner, isMine, owner, ownerAlive, type Owner } from '../instance.js';
import type { HeartmulaStatusResponse } from '../../contracts/heartmulaTraining.js';
import { mm3CodesDir } from './mm3Train.js';

export type Yue2BatchStage = 'prepare' | 'captions' | 'cache' | 'codes' | 'sheet' | 'stems' | 'align' | 'train' | 'refine' | 'nar' | 'finish';
export type BatchFamily = 'yue2' | 'heartmula' | 'mm3';
export type Yue2BatchStatus = 'running' | 'paused' | 'done' | 'failed' | 'cancelled';
export type Yue2BatchItemStatus = 'pending' | 'running' | 'done' | 'failed' | 'cancelled';

export interface Yue2BatchStageResult {
  stage: Yue2BatchStage;
  jobId: string;
  status: Yue2BatchItemStatus;
  error: string | null;
  startedAt: number | null;
  finishedAt: number | null;
  /** The training job a crash or failure interrupted. On resume the train
   *  stage continues from that run's last optimizer checkpoint instead of
   *  starting over. */
  resumeJobId?: string;
  /** HeartMuLa's immutable run identity, retained across job retries. */
  runId?: string;
}

export interface Yue2BatchItem {
  datasetId: string;
  name: string;
  status: Yue2BatchItemStatus;
  currentStage: Yue2BatchStage | null;
  stages: Yue2BatchStageResult[];
  error: string | null;
  /** A "finish" item: the scored refinement ladder to finish (NAR further
   *  training from its best rung, then link + cleanup), and the rung picked. */
  refineRun?: string;
  pickStep?: number;
  /** false: NAR further training ignores the plateau stop (budget or recon
   *  target only). */
  narKnee?: boolean;
}

export interface Yue2BatchSummary {
  /** Absent in older snapshots, which are all YuE2. */
  family?: BatchFamily;
  id: string;
  status: Yue2BatchStatus;
  items: Yue2BatchItem[];
  /** The dataset whose stage is running right now, for the UI to follow. */
  currentDatasetId: string | null;
  lyricTiming: boolean;
  /** Delete each dataset's YuE2 caches (latents, codes, lead sheets, stems,
   *  timing, prepared datasets) before its first stage, so it rebuilds from
   *  the audio. Once per dataset: a resumed batch keeps what it rebuilt. */
  clearCache?: boolean;
  /** The joint-train request applied to every dataset, paths stripped. */
  recipe: Record<string, unknown>;
  createdAt: number;
  finishedAt: number | null;
  pauseRequested?: boolean;
}

interface BatchState extends Yue2BatchSummary {
  cancelRequested: boolean;
}

const POLL_MS = 1500;
const IDLE_WAIT_MS = 10 * 60_000;
/** Captions stage re-runs while tracks are still uncaptioned (network blips), 10 min apart. */
const CAPTION_RETRIES = 3;
const CAPTION_RETRY_WAIT_MS = 10 * 60_000;
const MAX_LISTED = 20;
/** Body fields that belong to one dataset, never to a recipe. */
/** The KL-rung refinement belongs to the tuned recipe; a base-matched run
 *  ends at its update count and is picked by ear from its own ladder. */
const wantsRefine = (recipe: Record<string, unknown>) => recipe.autoRefine !== false && recipe.method !== 'base-matched' && recipe.method !== 'fixed-recipe';
const RECIPE_STRIP = new Set(['dataset', 'output', 'resume', 'resumeRunId', 'resumeStep', 'preparation', 'checkpoint', 'autoPrepare']);

const batches = new Map<string, BatchState>();
const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));
const isActive = (s: Yue2BatchStatus) => s === 'running' || s === 'paused';

function toSummary(state: BatchState): Yue2BatchSummary {
  const { cancelRequested, ...summary } = state;
  return summary;
}

// ── Persistence ──────────────────────────────────────────────────────────

function batchesDir(): string { return path.join(trainingBaseDir, 'yue2-batches'); }

/** On disk the summary also names the process running its loop (absent in
 *  legacy files). Only that process persists a live batch. */
type SnapshotFile = Yue2BatchSummary & { owner?: Owner };

function persist(state: BatchState): void {
  try {
    fs.mkdirSync(batchesDir(), { recursive: true });
    const file: SnapshotFile = { ...toSummary(state), owner: owner() };
    fs.writeFileSync(path.join(batchesDir(), `${state.id}.json`), JSON.stringify(file, null, 2), 'utf-8');
  } catch (err: any) {
    console.warn(`[Training] yue2 batch ${state.id} snapshot failed: ${err?.message || err}`);
  }
}

function readSnapshotFiles(): SnapshotFile[] {
  let names: string[];
  try { names = fs.readdirSync(batchesDir()).filter(n => n.endsWith('.json')); } catch { return []; }
  const out: SnapshotFile[] = [];
  for (const name of names) {
    try {
      const snap = JSON.parse(fs.readFileSync(path.join(batchesDir(), name), 'utf-8')) as SnapshotFile;
      if (snap && typeof snap.id === 'string' && Array.isArray(snap.items)) out.push(snap);
    } catch { /* a corrupt snapshot is skipped, not fatal */ }
  }
  return out;
}

function readSnapshots(): Yue2BatchSummary[] {
  return readSnapshotFiles().map(({ owner: _owner, ...summary }) => summary);
}

/** A batch that was running when the server died is paused, with its
 *  in-flight stage reset so resume re-runs it. Nothing restarts on its own.
 *  A batch another live process is running (the other copy of the app,
 *  services/instance.ts) is left alone. Called once at startup, after the
 *  instance registry opens (index.ts). */
export function recoverStaleBatches(): void {
  for (const snap of readSnapshotFiles()) {
    if (!isActive(snap.status)) continue;
    if (!isMine(snap.owner) && ownerAlive(snap.owner)) continue;
    for (const item of snap.items) {
      for (const s of item.stages) if (s.status === 'running') { if (s.jobId) s.resumeJobId = s.jobId; s.status = 'pending'; s.jobId = ''; s.startedAt = null; }
      if (item.status === 'running') { item.status = 'pending'; item.currentStage = null; }
    }
    snap.status = 'paused'; snap.pauseRequested = false; snap.currentDatasetId = null;
    try { fs.writeFileSync(path.join(batchesDir(), `${snap.id}.json`), JSON.stringify(snap, null, 2), 'utf-8'); } catch { /* best effort */ }
    console.log(`[Training] yue2 batch ${snap.id} recovered as paused after restart — resume to continue`);
  }
}

/** Why this process must not cancel, pause, resume or extend batch `id`:
 *  another live process runs its loop. null when it may. */
export function batchRunsElsewhere(id: string): string | null {
  if (batches.has(id)) return null;
  const snap = readSnapshotFiles().find(s => s.id === id);
  if (!snap || !isActive(snap.status) || isMine(snap.owner) || !ownerAlive(snap.owner)) return null;
  return `This batch is running in ${describeOwner(snap.owner)}; control it there`;
}

// ── Public API ───────────────────────────────────────────────────────────

export function hasActiveBatch(): boolean {
  for (const b of batches.values()) if (isActive(b.status)) return true;
  return false;
}

export function listBatches(): Yue2BatchSummary[] {
  const byId = new Map<string, Yue2BatchSummary>();
  for (const snap of readSnapshots()) byId.set(snap.id, snap);
  for (const b of batches.values()) byId.set(b.id, toSummary(b));
  return [...byId.values()].sort((a, b) => b.createdAt - a.createdAt).slice(0, MAX_LISTED);
}

export function getBatch(id: string): Yue2BatchSummary | undefined {
  const live = batches.get(id);
  return live ? toSummary(live) : readSnapshots().find(s => s.id === id);
}

export function startBatch(input: { datasetIds: string[]; family?: BatchFamily; lyricTiming: boolean; clearCache?: boolean; recipe: Record<string, unknown> }): Yue2BatchSummary | { error: string } {
  if (hasActiveBatch()) return { error: 'A training batch is already running' };
  // A captured YuE2 preparation may not have queued a job yet, so check it too.
  if (hasActivePipeline()) return { error: 'A training pipeline or YuE2 preparation is already running' };
  const items: Yue2BatchItem[] = [];
  for (const id of new Set(input.datasetIds)) {
    const ds = repo.getDataset(id);
    if (!ds) return { error: `Dataset not found: ${id}` };
    items.push({ datasetId: ds.id, name: ds.name || ds.slug, status: 'pending', currentStage: null, error: null,
      stages: stagesOf(input.family, input.lyricTiming, input.recipe).map(stage => ({ stage, jobId: '', status: 'pending', error: null, startedAt: null, finishedAt: null })) });
  }
  if (!items.length) return { error: 'Select at least one dataset' };
  const recipe: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(input.recipe)) if (!RECIPE_STRIP.has(k) && v !== undefined) recipe[k] = v;
  const state: BatchState = { id: randomUUID(), family: input.family ?? 'yue2', status: 'running', items, currentDatasetId: null, lyricTiming: input.lyricTiming,
    ...(input.clearCache ? { clearCache: true } : {}),
    recipe, createdAt: Date.now(), finishedAt: null, pauseRequested: false, cancelRequested: false };
  batches.set(state.id, state);
  persist(state);
  setImmediate(() => void runBatch(state));
  return toSummary(state);
}

/** Queue more datasets on an active batch. The loop iterates the live items
 *  array, so appended items run after the current ones, with the batch's recipe. */
export function appendToBatch(id: string, datasetIds: string[]): Yue2BatchSummary | { error: string } {
  const state = batches.get(id);
  if (!state || !isActive(state.status)) return { error: 'Batch is not running' };
  if (state.family === 'heartmula' && state.recipe.resumeRun !== undefined) return { error: 'A resumed HeartMuLa run belongs to one dataset; start a new batch for other datasets' };
  for (const dsId of datasetIds) {
    if (state.items.some(i => i.datasetId === dsId)) continue;
    const ds = repo.getDataset(dsId);
    if (!ds) return { error: `Dataset not found: ${dsId}` };
    state.items.push({ datasetId: ds.id, name: ds.name || ds.slug, status: 'pending', currentStage: null, error: null,
      stages: stagesOf(state.family, state.lyricTiming, state.recipe).map(stage => ({ stage, jobId: '', status: 'pending', error: null, startedAt: null, finishedAt: null })) });
  }
  persist(state);
  return toSummary(state);
}

/** Why `step` of `runId` cannot be finished from, or null when it can: the
 *  run must be one of this dataset's and hold a complete (AR + NAR)
 *  checkpoint at that step. Exported for a direct unit test. */
export function pickedCheckpointError(runs: Array<Pick<Yue2AitkRunRecord, 'jobId' | 'checkpoints'>>, runId: string, step: number): string | null {
  const run = runs.find(r => r.jobId === runId);
  if (!run) return `Run ${runId} is not one of this dataset's runs`;
  const ckpt = run.checkpoints.find(c => c.step === step);
  if (!ckpt) return `No checkpoint at step ${step} of run ${runId}`;
  if (!ckpt.arPath || !ckpt.narPath) return `No complete checkpoint at step ${step} of run ${runId}`;
  return null;
}

/** Finish scored ladders the way "Use this rung" does on the Refine tab:
 *  NAR further training from the chosen rung, then link the result to the
 *  album preset and clean up with the default choices. Queued on the end of
 *  the running batch when there is one, else run as a batch of its own.
 *  `pickStep` is the rung the user accepted, captured now and used as is: a
 *  score edited later cannot move it. Without it (legacy callers) the
 *  best-scored rung is chosen when the NAR stage starts. */
export function finishScoredLadders(entries: Array<{ datasetId: string; refineRun: string; pickStep?: number }>, opts: { knee?: boolean } = {}): Yue2BatchSummary | { error: string } {
  const items: Yue2BatchItem[] = [];
  for (const e of entries) {
    const ds = repo.getDataset(e.datasetId);
    if (!ds) return { error: `Dataset not found: ${e.datasetId}` };
    const runs = listYue2AitkRuns(ds.id, ds.slug);
    if (e.pickStep !== undefined) {
      const bad = pickedCheckpointError(runs, e.refineRun, e.pickStep);
      if (bad) return { error: bad };
    }
    // A checkpoint ladder is picked as it is: no decoder follow-up.
    const baseMatched = isCheckpointLadder(runs.find(r => r.jobId === e.refineRun)?.options);
    items.push({ datasetId: ds.id, name: ds.name || ds.slug, refineRun: e.refineRun, ...(e.pickStep !== undefined ? { pickStep: e.pickStep } : {}),
      narKnee: opts.knee !== false, status: 'pending', currentStage: null, error: null,
      stages: (baseMatched ? ['finish'] as const : ['nar', 'finish'] as const).map(stage => ({ stage, jobId: '', status: 'pending' as const, error: null, startedAt: null, finishedAt: null })) });
  }
  if (!items.length) return { error: 'Nothing to finish' };
  const live = [...batches.values()].find(b => isActive(b.status));
  if (live && live.family !== undefined && live.family !== 'yue2') return { error: `A ${live.family === 'mm3' ? 'MM3' : 'HeartMuLa'} batch is already running` };
  if (live) {
    for (const item of items) if (!live.items.some(i => i.refineRun === item.refineRun)) live.items.push(item);
    persist(live);
    return toSummary(live);
  }
  if (hasActivePipeline()) return { error: 'A training pipeline or YuE2 preparation is already running' };
  const state: BatchState = { id: randomUUID(), status: 'running', items, currentDatasetId: null, lyricTiming: true, recipe: {},
    createdAt: Date.now(), finishedAt: null, pauseRequested: false, cancelRequested: false };
  batches.set(state.id, state);
  persist(state);
  setImmediate(() => void runBatch(state));
  return toSummary(state);
}

/** A decoder-only follow-up gets no second one, as on the Refine tab — a
 *  legitimate skip. (A remote-origin run is a separate, rejected case,
 *  checked by the caller before this: it must stop the whole batch item
 *  rather than silently fall through to 'finish', see stageRequest's 'nar'
 *  branch and RefinePanel.tsx's equivalent check.) Exported for a direct
 *  unit test. */
export function skipNarFurther(run: Pick<Yue2AitkRunRecord, 'options'> | undefined): boolean {
  return (run?.options as Record<string, unknown> | undefined)?.freezePlannerNow === true;
}

/** The Refine tab's "Further training for NAR" request, at its defaults. */
function narFurtherRequest(runId: string, step: number, knee = true): Record<string, unknown> {
  const budget = 250;
  return { trainingMethod: 'aitk', refine: true, resumeRunId: runId, resumeStep: step,
    steps: step + budget, saveEvery: 10, stopMode: 'kl', narExtraSteps: step + budget, freezePlannerNow: true, narLrScale: 0.2,
    reconStop: knee ? 0.005 : 0, reconStopWindow: 10, reconKeepDelta: 0.003, reconTarget: 0.25, stopEngine: false,
    lyricTiming: true, alignmentEnabled: true, autoPrepare: false, checkpoint: '', output: '',
    preview: { enabled: false, everySteps: 0, seconds: 90, seed: 424242, previewMaxFrames: 2250, baseline: false, control: false } };
}

/** Link the finished checkpoint (the NAR run's last, or the picked rung when
 *  NAR was skipped) to the album preset, then clean up around it. */
async function finishLadder(item: Yue2BatchItem): Promise<void> {
  const ds = repo.getDataset(item.datasetId);
  if (!ds) throw new Error('Dataset not found');
  const runs = listYue2AitkRuns(ds.id, ds.slug);
  const narJob = item.stages.find(s => s.stage === 'nar')?.jobId;
  let runId = item.refineRun ?? '';
  let step = item.pickStep;
  if (step === undefined && !narJob && runId) {
    // No decoder stage chose a step (a base-matched ladder): the best-scored rung.
    const best = bestScoredRung(ds.id, runId, ds.slug);
    if (!best) throw new Error('No rung of this ladder has both a likeness and a corruption score');
    step = best.step;
  }
  if (narJob) {
    const nar = runs.find(r => r.jobId === narJob);
    const last = nar?.checkpoints.filter(c => c.arPath && c.narPath).sort((a, b) => b.step - a.step)[0];
    if (!nar || !last) throw new Error('The NAR further-training run left no complete checkpoint');
    runId = nar.jobId; step = last.step;
  }
  const ckpt = runs.find(r => r.jobId === runId)?.checkpoints.find(c => c.step === step);
  if (!ckpt || step === undefined) throw new Error(`No checkpoint at step ${step} of run ${runId}`);
  if (!ckpt.arPath || !ckpt.narPath) throw new Error(`No complete checkpoint at step ${step} of run ${runId}`);
  const known = runs.flatMap(r => r.checkpoints).flatMap(c => [c.arPath, c.narPath].filter((v): v is string => !!v));
  const refreshed = refreshYue2PresetsForJointCheckpoint({ slug: ds.slug, lyricsSetId: ds.lyricsSetId }, ckpt.arPath, ckpt.narPath, known);
  // The durable link record, not the preset count, is proof the checkpoint
  // is now the record of truth — cleanup below must never run off an
  // unlinked checkpoint (Reviewer, blocker #1).
  if (!refreshed.linked) throw new Error(`Could not record this checkpoint as linked: ${refreshed.error || 'unknown error'}`);
  const score = item.refineRun && !narJob ? listYue2RungScores(ds.id, item.refineRun).find(r => r.step === step) : undefined;
  const result = await runYue2Cleanup({ id: ds.id, slug: ds.slug, sourceDir: ds.sourceDir, lyricsSetId: ds.lyricsSetId }, runId, step,
    { caches: true, otherCheckpoints: true, otherRuns: true, resume: true, otherPreviews: true },
    { blind: score?.blind ?? false, blindLabel: score?.blindLabel ?? '' });
  if (result.finishError) throw new Error(result.finishError);
  console.log(`[Training] yue2 batch finish ${ds.slug}: linked step ${step} of ${runId}; removed ${result.done.join(', ') || 'nothing'}`);
}

export function pauseBatch(id: string): 'ok' | 'not_found' | 'not_active' {
  const state = batches.get(id);
  if (!state) return getBatch(id) ? 'not_active' : 'not_found';
  if (!isActive(state.status)) return 'not_active';
  state.pauseRequested = true; persist(state);
  return 'ok';
}

/** Resume keeps every done stage done and re-runs the rest, failures included. */
export function resumeBatch(id: string): 'ok' | 'not_found' | 'busy' {
  const live = batches.get(id);
  if (live && live.status === 'paused' && !live.pauseRequested) { live.status = 'running'; persist(live); return 'ok'; }
  if (live && live.status === 'paused') { live.pauseRequested = false; live.status = 'running'; persist(live); return 'ok'; }
  const snap = getBatch(id);
  if (!snap) return 'not_found';
  if (hasActiveBatch() || hasActivePipeline() || batchRunsElsewhere(id)) return 'busy';
  for (const item of snap.items) {
    if (item.status === 'done') continue;
    item.status = 'pending'; item.currentStage = null; item.error = null;
    for (const s of item.stages) if (s.status !== 'done') { if (s.jobId && s.stage === 'train') s.resumeJobId = s.jobId; s.status = 'pending'; s.jobId = ''; s.error = null; s.startedAt = null; s.finishedAt = null; }
  }
  const state: BatchState = { ...snap, status: 'running', finishedAt: null, pauseRequested: false, currentDatasetId: null, cancelRequested: false };
  batches.set(state.id, state);
  persist(state);
  setImmediate(() => void runBatch(state));
  return 'ok';
}

export function cancelBatch(id: string): boolean {
  const state = batches.get(id);
  if (!state) {
    const snap = getBatch(id);
    if (!snap) return false;
    if (batchRunsElsewhere(id)) return true;   // its own loop owns the file; the route answers 409
    if (isActive(snap.status)) { snap.status = 'cancelled'; snap.finishedAt = Date.now(); const s = { ...snap, cancelRequested: false }; batches.set(id, s); persist(s); }
    return true;
  }
  state.cancelRequested = true;
  for (const item of state.items) for (const s of item.stages) if (s.status === 'running' && s.jobId) queue.cancelJob(s.jobId);
  persist(state);
  return true;
}

// ── The loop ─────────────────────────────────────────────────────────────

/** The shared MM3 recipe minus `trigger`, which belongs to each dataset: without it the train route
 *  uses the dataset's own custom tag. A batch that carried it trained every album on the first album's
 *  trigger (HOTSTEP-1092), so it is dropped here, where every batch passes, whatever the client sent. */
export function mm3BatchTrainBody(recipe: Record<string, unknown>): Record<string, unknown> {
  const { trigger: _trigger, ...shared } = recipe;
  return shared;
}

/** True when every manifest row has its `<id>.codes` file (the trainer silently skips a row without one).
 *  An unreadable or empty manifest is "not complete", so the codes stage runs and reports the real error. */
export function mm3CodesComplete(manifest: string | null | undefined, codesInner: string): boolean {
  try {
    const parsed = JSON.parse(fs.readFileSync(manifest ?? '', 'utf-8'));
    const rows: unknown[] = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.samples) ? parsed.samples : [];
    const ids = rows.map(r => (r as { id?: unknown })?.id);
    return ids.length > 0 && ids.every(id => typeof id === 'string' && id !== '' && fs.existsSync(path.join(codesInner, `${id}.codes`)));
  } catch { return false; }
}

function stagesFor(lyricTiming: boolean, refine = true, captions = false): Yue2BatchStage[] {
  // The batch owns the chain: train, then the planner refinement with its
  // rung previews, then the next dataset. The run-level auto-refine hook is
  // switched off for batch runs so nothing fires twice.
  const base: Yue2BatchStage[] = lyricTiming ? ['cache', 'codes', 'sheet', 'stems', 'align', 'train'] : ['cache', 'codes', 'sheet', 'train'];
  // recipe.autoCaption: tracks with no .yue2.txt are re-captioned from the
  // audio first, so nothing trains on the long ACE caption.
  return [...(captions ? ['captions' as const] : []), ...base, ...(refine ? ['refine' as const] : [])];
}

/** The stage chain per family. YuE2 builds its own (stagesFor); the others are fixed. */
const FIXED_STAGES: Partial<Record<BatchFamily, readonly Yue2BatchStage[]>> = {
  heartmula: ['prepare', 'train'],
  mm3: ['codes', 'train'],
};
function stagesOf(family: BatchFamily | undefined, lyricTiming: boolean, recipe: Record<string, unknown>): readonly Yue2BatchStage[] {
  return (family && FIXED_STAGES[family]) || stagesFor(lyricTiming, wantsRefine(recipe), !!recipe.autoCaption);
}

/** Per-family start route for a stage where it differs from STAGE_PATH. */
const FAMILY_ROUTE: Partial<Record<BatchFamily, Partial<Record<Yue2BatchStage, string>>>> = {
  heartmula: { train: 'heartmula-train' },
  mm3: { codes: 'mm3-codes', train: 'mm3-train-lm' },
};

/** Which caches `clearCache` removes. Each family clears only its own: the
 *  others belong to other backends. */
function cacheScope(family: BatchFamily | undefined): { scope: { target?: 'heartmula' }; owns: (name: string) => boolean } {
  if (family === 'heartmula') return { scope: { target: 'heartmula' }, owns: () => true };
  if (family === 'mm3') return { scope: {}, owns: name => name === 'mm3-codes' || name === 'mm3-codes-laundered' };
  return { scope: {}, owns: name => name.startsWith('yue2-') };
}

const STAGE_PATH: Record<Yue2BatchStage, string> = {
  prepare: 'heartmula-prepare',
  captions: 'yue2-captions-missing',
  nar: 'yue2-joint-train',
  finish: '',
  cache: 'yue2-preprocess', codes: 'yue2-tokenize', sheet: 'yue2-sheet', stems: 'yue2-stems', align: 'yue2-align', train: 'yue2-joint-train',
  refine: 'yue2-joint-train',
};

async function runBatch(state: BatchState): Promise<void> {
  for (const item of state.items) {
    await waitWhilePaused(state);
    if (state.cancelRequested) break;
    if (item.status === 'done') continue;
    await runItem(state, item);
  }
  state.currentDatasetId = null;
  state.status = state.cancelRequested ? 'cancelled'
    : state.items.some(i => i.status === 'failed') ? 'failed' : 'done';
  for (const item of state.items) if (item.status === 'pending') item.status = state.cancelRequested ? 'cancelled' : item.status;
  state.finishedAt = Date.now();
  persist(state);
}

async function waitWhilePaused(state: BatchState): Promise<void> {
  if (!state.pauseRequested) return;
  state.status = 'paused'; persist(state);
  while (state.pauseRequested && !state.cancelRequested) await sleep(POLL_MS);
  if (!state.cancelRequested) { state.status = 'running'; persist(state); }
}

async function runItem(state: BatchState, item: Yue2BatchItem): Promise<void> {
  item.status = 'running'; item.error = null; state.currentDatasetId = item.datasetId; persist(state);
  if (state.clearCache && !item.refineRun && item.stages.every(s => s.status === 'pending' && !s.jobId)) {
    try {
      const ds = repo.getDataset(item.datasetId);
      if (!ds) throw new Error('Dataset not found');
      // YuE2 caches only: this dataset's ACE and MM3 caches belong to other backends.
      // The core (latents, codes, lead sheets, prepared set) is kept: stale
      // parts are rebuilt by the stages' own checks, so this clears the stems.
      const { scope, owns } = cacheScope(state.family);
      for (const cache of listPreparedCaches(ds.slug, ds.sourceDir, scope)) {
        if (owns(cache.name)) fs.rmSync(cache.path, { recursive: true, force: false });
      }
    } catch (err) {
      item.status = 'failed'; item.error = `Clearing cached data failed: ${err instanceof Error ? err.message : String(err)}`;
      for (const rest of item.stages) rest.status = 'cancelled';
      persist(state);
      return;
    }
  }
  for (const result of item.stages) {
    if (result.status === 'done') continue;
    await waitWhilePaused(state);
    if (state.cancelRequested) { finishStage(state, result, 'cancelled', null); break; }
    await runStage(state, item, result);
    // A network blip fails every caption call at once (2026-09-29: 03:15-03:19,
    // "fetch failed" on 25 tracks, and two albums lost the night). The stage
    // only captions what is missing, so wait and run it again before the
    // cache stage gives up on the item.
    for (let retry = 1; result.stage === 'captions' && (result as { status: Yue2BatchItemStatus }).status === 'done' && retry <= CAPTION_RETRIES; retry++) {
      const ds = repo.getDataset(item.datasetId);
      if (!ds || !(await samplesMissingYue2Caption(ds)).length) break;
      const until = Date.now() + CAPTION_RETRY_WAIT_MS;
      while (Date.now() < until && !state.cancelRequested) await sleep(POLL_MS);
      if (state.cancelRequested) break;
      result.jobId = ''; result.error = null;
      await runStage(state, item, result);
    }
    // TS narrowed result.status at the loop head; the stage ran since then.
    const outcome = (result as { status: Yue2BatchItemStatus }).status;
    if (outcome !== 'done') {
      item.status = outcome === 'cancelled' ? 'cancelled' : 'failed';
      item.error = result.error;
      for (const rest of item.stages) if (rest.status === 'pending') rest.status = 'cancelled';
      item.currentStage = null; persist(state);
      return;
    }
  }
  item.status = state.cancelRequested ? 'cancelled' : 'done'; item.currentStage = null; persist(state);
}

interface ArStatus {
  stages: {
    preprocess: { done: boolean; captionModeOk: boolean; captionsStale?: boolean; loudnessLufs?: number | null };
    tokenize: { done: boolean; status?: { tokenizer?: string } | null };
    sheet: { done: boolean };
    align: { done: boolean; stemsReady: number; stemsNeeded: number };
  };
}

async function readStatus(datasetId: string): Promise<ArStatus> {
  const r = await fetch(`${self()}/api/training/datasets/${encodeURIComponent(datasetId)}/yue2-ar`);
  if (!r.ok) throw new Error(`status ${r.status}: ${await errorTextOf(r)}`);
  return await r.json() as ArStatus;
}

const self = () => `http://127.0.0.1:${config.server.port}`;

async function errorTextOf(r: Response): Promise<string> {
  try { const j = await r.json() as { error?: unknown }; if (typeof j?.error === 'string') return j.error; } catch { /* not JSON */ }
  return `HTTP ${r.status}`;
}

/** What each stage needs already true to be skipped, and the body it posts.
 *  `null` body = the stage is already done for this dataset. */
async function stageRequest(state: BatchState, item: Yue2BatchItem, result: Yue2BatchStageResult): Promise<Record<string, unknown> | null> {
  const stage = result.stage;
  if (state.family === 'heartmula') {
    const response = await fetch(`${self()}/api/training/datasets/${encodeURIComponent(item.datasetId)}/heartmula-status`);
    if (!response.ok) throw new Error(await errorTextOf(response));
    const status = await response.json() as HeartmulaStatusResponse;
    if (stage === 'prepare') return status.prepare.cacheValid && status.prepare.preparedSamples > 0 && status.prepare.staleSamples === 0 ? null : {};
    if (stage !== 'train') throw new Error(`Unsupported HeartMuLa batch stage: ${stage}`);
    if (result.runId) {
      const run = status.runs.find(r => r.run === result.runId);
      if (!run) throw new Error('The interrupted HeartMuLa run is missing');
      if (run.state === 'done') return null;
      if (run.resumable) return { resumeRun: run.run };
      throw new Error('The interrupted HeartMuLa run has no checkpoint to resume');
    }
    return state.recipe;
  }
  if (state.family === 'mm3') {
    const ds = repo.getDataset(item.datasetId);
    if (!ds) throw new Error('Dataset not found');
    const launder = state.recipe.launder === true;
    // Codes are done when every current manifest row has its .codes file; a
    // dataset that grew or swapped tracks since gets re-encoded. One rank/base for the whole batch: the recipe
    // is pinned, not re-recommended per dataset.
    if (stage === 'codes') return mm3CodesComplete(ds.datasetJsonPath, path.join(mm3CodesDir(ds.slug, launder), 'codes')) ? null : { launder };
    if (stage !== 'train') throw new Error(`Unsupported MM3 batch stage: ${stage}`);
    return mm3BatchTrainBody(state.recipe);
  }
  if (stage === 'train' && result.resumeJobId) {
    const resumed = resumeTrainingBody(state, item, result.resumeJobId);
    if (resumed !== undefined) return resumed;
  }
  if (stage === 'nar') {
    const ds = repo.getDataset(item.datasetId);
    const runs = listYue2AitkRuns(item.datasetId, ds?.slug);
    let step = item.pickStep;
    if (step === undefined) {
      // Legacy item, no captured pick: the best-scored rung, recorded once.
      const best = item.refineRun ? bestScoredRung(item.datasetId, item.refineRun, ds?.slug) : null;
      if (!best) throw new Error('No rung of this ladder has both a likeness and a corruption score');
      step = best.step;
      item.pickStep = step; persist(state);
    } else {
      // The captured pick, checked again: the run may have changed since.
      const bad = pickedCheckpointError(runs, item.refineRun ?? '', step);
      if (bad) throw new Error(bad);
    }
    const run = runs.find(r => r.jobId === item.refineRun);
    // A decoder-only follow-up legitimately gets no second one (nothing to
    // reject — this stage is correctly "done"). A mirrored run is a different
    // case: NAR further training resumes from the rung's optimizer.resume and
    // the run's prepared dataset, and the mirror copies neither, so this stops
    // the item with a clear reason rather than skip to 'finish' and mark NAR
    // done when it never ran.
    if (run?.origin) throw new Error(`NAR further training needs the optimizer state, which stays on ${run.origin.worker}; this batch item needs manual handling.`);
    if (skipNarFurther(run)) return null;
    return narFurtherRequest(item.refineRun!, step, item.narKnee !== false);
  }
  if (stage === 'finish') { await finishLadder(item); return null; }
  if (stage === 'captions') {
    const c = state.recipe.autoCaption as { provider?: string; model?: string } | undefined;
    return c ? { provider: c.provider, ...(c.model ? { model: c.model } : {}) } : null;
  }
  if (stage === 'cache' && state.recipe.autoCaption) {
    const ds = repo.getDataset(item.datasetId);
    const left = ds ? (await samplesMissingYue2Caption(ds)).length : 0;
    if (left) throw new Error(`${left} track(s) still have no YuE2 caption (.yue2.txt) after the captions stage; see its job log. Training would fall back to the long ACE captions.`);
  }
  const st = await readStatus(item.datasetId);
  switch (stage) {
    case 'prepare': throw new Error('HeartMuLa preparation requires a HeartMuLa batch');
    case 'cache': {
      // Every method trains on a YUE2_CACHE_LUFS cache. A cache at another level is
      // re-cut: the key changes, so codes, sheet and alignment are dropped and
      // their stages run again.
      const lufs = YUE2_CACHE_LUFS;
      const p = st.stages.preprocess;
      const cut = p.done && p.captionModeOk !== false && !p.captionsStale && p.loudnessLufs === lufs;
      return cut ? null : { captionMode: 'yue2', loudnessLufs: lufs };
    }
    case 'codes': {
      const tokenizer = yue2TokenizerSetting().training;
      return yue2CodesReady(st.stages.tokenize, tokenizer) ? null : { tokenizer };
    }
    case 'sheet': return st.stages.sheet.done ? null : {};
    case 'stems': {
      if (st.stages.align.stemsNeeded < 1) throw new Error('The latent cache contains no songs to align.');
      return st.stages.align.stemsReady >= st.stages.align.stemsNeeded ? null : {};
    }
    case 'align': {
      if (st.stages.align.stemsReady < st.stages.align.stemsNeeded) {
        throw new Error(`Vocal stems are incomplete (${st.stages.align.stemsReady}/${st.stages.align.stemsNeeded}); retry separation before alignment.`);
      }
      return st.stages.align.done ? null : {};
    }
    case 'train': return { ...state.recipe, autoCaption: undefined, trainingMethod: 'aitk', autoPrepare: true, checkpoint: '', dataset: '', output: '',
      lyricTiming: state.lyricTiming, alignmentEnabled: state.lyricTiming, autoRefine: false,
      ...(state.lyricTiming ? {} : { cursorWeight: 0 }) };
    case 'refine': {
      const trainJob = item.stages.find(s => s.stage === 'train')?.jobId;
      const run = trainJob ? listYue2AitkRuns(item.datasetId).find(r => r.jobId === trainJob) : undefined;
      const last = run?.checkpoints.filter(c => !!c.optimizerPath && c.arPath && c.narPath).sort((a, b) => b.step - a.step)[0];
      if (!run || !last) throw new Error('No resumable checkpoint from the training stage to refine');
      return autoRefineRequest(run.jobId, last.step);
    }
  }
}

/** Continue an interrupted joint run from its last optimizer checkpoint.
 *  `undefined` = nothing to resume (no indexed run, no optimizer checkpoint),
 *  so the stage starts from scratch; `null` = the run already reached the
 *  step cap, so the stage is done. */
function resumeTrainingBody(state: BatchState, item: Yue2BatchItem, jobId: string): Record<string, unknown> | null | undefined {
  // The interrupted job's own run, or — when that job died before it was
  // indexed (a resume that itself crashed) — the newest run this batch made
  // for the dataset that still has an optimizer checkpoint.
  const runs = listYue2AitkRuns(item.datasetId);
  const withCheckpoint = (r: typeof runs[number]) => r.checkpoints.filter(c => !!c.optimizerPath).sort((a, b) => b.step - a.step)[0];
  const run = runs.find(r => r.jobId === jobId && withCheckpoint(r))
    ?? runs.filter(r => r.createdAt >= state.createdAt && withCheckpoint(r)).sort((a, b) => b.createdAt - a.createdAt)[0];
  if (!run) return undefined;
  const last = withCheckpoint(run)!;
  const cap = Number(state.recipe.steps ?? run.options.steps);
  if (Number.isFinite(cap) && last.step >= cap) return null;
  console.log(`[Training] yue2 batch ${state.id}: resuming ${item.name} from step ${last.step} of run ${run.jobId}`);
  // The route rebuilds the recipe from the indexed run; only the cap, stop
  // policy and previews are read from the body.
  return { trainingMethod: 'aitk', resumeRunId: run.jobId, resumeStep: last.step,
    steps: cap, stopMode: state.recipe.stopMode, targetLoss: state.recipe.targetLoss, targetKl: state.recipe.targetKl,
    preview: state.recipe.preview };
}

async function runStage(state: BatchState, item: Yue2BatchItem, result: Yue2BatchStageResult): Promise<void> {
  result.status = 'running'; result.startedAt = Date.now(); item.currentStage = result.stage; persist(state);
  const deadline = Date.now() + IDLE_WAIT_MS;
  while (queue.activeJobForDataset(item.datasetId) && !state.cancelRequested && Date.now() < deadline) await sleep(POLL_MS);
  if (state.cancelRequested) { finishStage(state, result, 'cancelled', null); return; }

  let body: Record<string, unknown> | null;
  try { body = await stageRequest(state, item, result); }
  catch (err: any) { finishStage(state, result, 'failed', err?.message || String(err)); return; }
  if (body === null) { finishStage(state, result, 'done', null); return; }

  let response: Response;
  try {
    const route = FAMILY_ROUTE[state.family ?? 'yue2']?.[result.stage] ?? STAGE_PATH[result.stage];
    response = await fetch(`${self()}/api/training/datasets/${encodeURIComponent(item.datasetId)}/${route}`,
      { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  } catch (err: any) { finishStage(state, result, 'failed', err?.message || String(err)); return; }
  if (!response.ok) { finishStage(state, result, 'failed', await errorTextOf(response)); return; }

  let jobId = '', skipped = '';
  try {
    const payload = await response.json() as { jobId?: unknown; skipped?: unknown; run?: unknown };
    if (typeof payload?.jobId === 'string') jobId = payload.jobId;
    if (state.family === 'heartmula' && typeof payload?.run === 'string') result.runId = payload.run;
    if (typeof payload?.skipped === 'string') skipped = payload.skipped;
  } catch { /* handled below */ }
  if (!jobId && skipped) { finishStage(state, result, 'done', null); return; }
  if (!jobId) { finishStage(state, result, 'failed', 'Stage returned no jobId'); return; }
  result.jobId = jobId; result.resumeJobId = undefined; persist(state);

  for (;;) {
    if (state.cancelRequested) queue.cancelJob(jobId);
    const job = queue.getJob(jobId);
    if (!job) { finishStage(state, result, 'cancelled', null); return; }
    if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') { finishStage(state, result, job.status, job.error ?? null); return; }
    await sleep(POLL_MS);
  }
}

function finishStage(state: BatchState, result: Yue2BatchStageResult, status: 'done' | 'failed' | 'cancelled', error: string | null): void {
  result.status = status; result.error = error; result.finishedAt = Date.now();
  persist(state);
}
