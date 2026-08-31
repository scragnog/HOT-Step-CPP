// trainingStore.ts — Zustand store for the Training Studio (Dataset phase)
//
// Disk is the source of truth for sample data; this store mirrors what the
// server reports. `samplesById` is the authoritative row map — SSE `sample`
// events REPLACE entries by sampleId (never append), so an EventSource
// reconnect that replays the whole buffer is idempotent.

import { create } from 'zustand';
import * as trainingApi from '../services/trainingApi';
import { computeTrainingEta, type TrainingEta } from '../utils/trainingEta';
import type {
  AuditionOptions,
  AuditionPreview,
  BulkSetInput,
  CaptionOptions,
  CreateDatasetInput,
  GeniusOptions,
  LabelOptions,
  PatchDatasetInput,
  PatchSampleInput,
  LmSize,
  PipelineSummary,
  PreprocessOptions,
  PreprocessStatus,
  StartPipelineInput,
  TrainDitEpoch,
  TrainDitOptions,
  TrainDitStatus,
  TrainLmEpoch,
  TrainLmOptions,
  TrainLmStatus,
  TrainingCapabilities,
  TrainingDatasetDetail,
  TrainingDatasetSummary,
  TrainingJobSummary,
  TrainingPreview,
  TrainingSample,
  TrainingStreamEvent,
} from '../services/trainingApi';

/** A preview plus the run it belongs to — the run name is what the audio
 *  URL needs, and it lives on the event rather than on the preview. */
export type TrainingPreviewRow = TrainingPreview & { run: string };

const JOB_LOG_CAP = 200;
const EDIT_DEBOUNCE_MS = 500;
/** Points kept for the chart's per-step layer. A 500-song LM run emits ~25 000
 *  `metric:'step'` frames, so the series is decimated rather than truncated. */
const TRAIN_STEP_CAP = 2000;

/** One `metric:'step'` frame, reduced to what TrainingChart plots. */
export interface TrainStepPoint {
  step: number;
  loss: number;
  /** Fractional epoch position — the x-domain the epoch series also lives in. */
  ep: number;
}

/** One `metric:'milestone'` frame — a tick on the chart's x-axis. */
export interface TrainMilestonePoint {
  epoch: number;
  loss: number;
  path: string;
}

/** Fields the grid/drawer may edit inline. */
export type EditableField = keyof PatchSampleInput;

// Debounce timers live outside React state — one per (sampleId, field).
const editTimers = new Map<string, number>();
// Jobs whose terminal `status` event already triggered the authoritative
// refresh. The SSE buffer is replayed on reconnect, so this must be guarded.
const refreshedJobs = new Set<string>();
// The last EXPLICIT train-lm status query. §5.5 spells the terminal refresh as
// `loadTrainLmStatus()` with no argument, but the panel carries its own
// lmSize/adapterName: a no-arg call defaults server-side to 0.6B + dataset slug,
// i.e. a DIFFERENT adapter directory, and it races the panel's own correctly
// parameterised refresh. Remembering the last query makes the no-arg form mean
// "refresh what is on screen".
let lastTrainLmQuery: { variantKey?: string; adapterName?: string; lmSize?: LmSize } | undefined;
// Monotonic request id — the older of two in-flight status reads must not win.
let trainLmSeq = 0;
// Same two devices for train-dit.
let lastTrainDitQuery: { variantKey?: string; adapterName?: string } | undefined;
let trainDitSeq = 0;
// The `vram` metric carries crop/layers exactly ONCE, before the first step, and
// `trainDitLast` does not exist yet at that point — creating it there would put a
// fabricated 0.0000 loss on the metric strip. Park them here instead and fold them
// into every later trainDitLast write.
let ditVram = { crop: 0, layers: 0 };
// The per-step series is decimated, so "do I already have this step?" cannot be
// answered by scanning it — a replayed frame that was thinned away would be
// re-inserted, out of order and double-counted. Steps are monotonic within a
// run, so the high-water mark is the idempotency key instead.
let lastStepSeen = -1;

/** The chart slice, blanked. Every place that starts or swaps a job spreads
 *  this, and it is the only thing that may reset `lastStepSeen`.
 *
 *  Exported because the bulk Monitor page swaps jobs too (PipelineStageProgress
 *  adopts whichever stage is running). Skipping it there left the previous
 *  stage's step points — trainStepSeries is SHARED by both training kinds — in
 *  the series, so an LM run at epoch 15 drew into an axis still scaled to the
 *  DiT's 500 (2026-07-31). */
export function blankTrainSeries(): {
  trainStepSeries: TrainStepPoint[];
  trainMilestones: TrainMilestonePoint[];
  trainTargetLoss: number;
  trainMaxEpochs: number;
  trainStepsPerEpoch: number;
  trainPreviews: TrainingPreviewRow[];
} {
  lastStepSeen = -1;
  return {
    trainStepSeries: [],
    trainMilestones: [],
    trainTargetLoss: 0,
    trainMaxEpochs: 0,
    trainStepsPerEpoch: 0,
    trainPreviews: [],
  };
}

/**
 * Append one step point, halving the series once it outgrows the cap by
 * dropping every other retained point. The curve then keeps spanning the whole
 * run and simply gets coarser — a trailing window would throw away the shape of
 * everything before it, which is the part of a loss curve people read.
 */
function appendStepPoint(prev: TrainStepPoint[], point: TrainStepPoint): TrainStepPoint[] {
  const next = [...prev, point];
  if (next.length <= TRAIN_STEP_CAP) return next;
  const thinned = next.filter((_, i) => i % 2 === 0);
  // The newest point must survive the thin or the line stops tracking "now".
  if (thinned[thinned.length - 1]?.step !== point.step) thinned.push(point);
  return thinned;
}

function timerKey(sampleId: string, field: string): string {
  return `${sampleId}:${field}`;
}

function clearTimer(key: string): void {
  const t = editTimers.get(key);
  if (t !== undefined) { window.clearTimeout(t); editTimers.delete(key); }
}

interface TrainingState {
  // navigation
  phase: 'dataset' | 'preprocess' | 'train' | 'monitor';   // 'dataset' and 'preprocess' selectable
  step: 'label' | 'review' | 'build';
  selectedDatasetId: string | null;

  // data
  capabilities: TrainingCapabilities | null;
  datasets: TrainingDatasetSummary[];
  detail: TrainingDatasetDetail | null;
  samplesById: Record<string, TrainingSample>;   // authoritative row map, merged from SSE
  sampleOrder: string[];                          // display order (relPath asc)

  // job
  activeJob: TrainingJobSummary | null;
  jobLog: Array<{ level: string; message: string; ts: number }>;   // capped at 200

  // preprocess
  preprocessStatus: PreprocessStatus | null;
  preprocessLoading: boolean;

  // train (LM)
  trainLmStatus: TrainLmStatus | null;
  trainLmLoading: boolean;
  trainLmEpochs: TrainLmEpoch[];          // live, from `metric` events
  trainLmLast: { loss: number; lr: number; gradNorm: number; etaMs: number } | null;
  /** MiniMax-Music3 live run stats. Its own field rather than reusing
   *  trainLmLast: MM3 reports STEPS (with a per-step wall time) and no epochs,
   *  so the two shapes genuinely differ, and sharing one would leave a field
   *  that means different things depending on which backend ran. */
  mm3Live: {
    step: number; totalSteps: number; loss: number; lr: number; gradNorm: number;
    stepMs: number; usedMb: number; totalMb: number;
  } | null;
  /** Held-out loss, in the same fractional-epoch x domain as the other series. */
  trainEvalSeries: Array<{ step: number; loss: number; ep: number }>;
  /** From the one `data` metric — songs skipped for exceeding max sequence
   *  length. Surfaced in the panel; without it the only trace is a warn line
   *  in the scrolling log tail (§5.6 mandates the string). */
  trainLmSkippedLong: number;
  /** From the one `vram` metric (4B plan §1.3/§2.2). Drives the per-size VRAM
   *  hint line in TrainLmForm; null until the engine has reported. */
  trainLmVram: {
    mode: string; maxLen: number; estMb: number; freeMb: number;
    baseMb: number; ckptMb: number; segPeakMb: number;
  } | null;

  // train (DiT)
  trainDitStatus: TrainDitStatus | null;
  trainDitLoading: boolean;
  trainDitEpochs: TrainDitEpoch[];        // live, from `metric` events
  trainDitLast: {
    loss: number; ma5: number; lr: number; gradNorm: number;
    crop: number; layers: number; etaMs: number;
  } | null;

  // train (chart + ETA, shared by both kinds — only one job owns a dataset)
  /** Per-step loss for the chart's noise layer. Capped + decimated, replay-safe. */
  trainStepSeries: TrainStepPoint[];
  /** Milestone ticks, deduped by path (the SSE buffer is replayed on reconnect). */
  trainMilestones: TrainMilestonePoint[];
  /** Target loss of the RUNNING job, captured at start; 0 = auto-stop disabled.
   *  Not recoverable after a page reload mid-run — the ETA then falls back to
   *  the epoch cap, which is still true, just less useful. */
  trainTargetLoss: number;
  /** Epoch cap of the running job: from the start options, then confirmed by the
   *  `epochs` field the engine stamps on every epoch/step metric. */
  trainMaxEpochs: number;
  /** From the one `data` metric — turns a global step number into an epoch
   *  position so the step and epoch layers share one x-axis. */
  trainStepsPerEpoch: number;
  /** Mid-run audio previews of an MM3 LM run, oldest first. Deduped by id: the
   *  SSE buffer is replayed on every reconnect. */
  trainPreviews: TrainingPreviewRow[];

  // audition (codes preview) — §6.6
  /** Preview history for the open dataset, newest first. */
  auditions: AuditionPreview[];
  /** True from the POST until the job is adopted; the job's own status owns the
   *  rest of the run, so this is deliberately short-lived. */
  auditionRunning: boolean;
  auditionError: string | null;
  /** Sample-drawer previews, keyed by sampleId (the sync path). */
  samplePreviews: Record<string, AuditionPreview>;
  /** Which stored-codes source each sample preview came from (C12 precedence). */
  samplePreviewSources: Record<string, 'lm_codes' | 'label'>;
  samplePreviewPending: Record<string, boolean>;
  samplePreviewErrors: Record<string, string>;

  // batch pipeline (monitor phase)
  pipelines: PipelineSummary[];
  pipelinesLoading: boolean;

  // grid
  selectedSampleIds: Set<string>;
  openSampleId: string | null;
  pendingEdits: Record<string, PatchSampleInput>;  // optimistic, keyed by sampleId
  savingSampleIds: Set<string>;
  saveError: { sampleId: string; message: string } | null;
  loading: boolean;
  error: string | null;

  // actions
  setPhase(phase: TrainingState['phase']): void;
  setStep(step: TrainingState['step']): void;
  closeDataset(): void;
  loadCapabilities(): Promise<void>;
  loadDatasets(): Promise<void>;
  openDataset(id: string): Promise<void>;
  createDataset(input: CreateDatasetInput): Promise<string>;
  patchDataset(patch: PatchDatasetInput): Promise<void>;
  deleteDataset(id: string): Promise<void>;
  rescan(): Promise<void>;
  editSample(sampleId: string, patch: PatchSampleInput): Promise<void>;   // debounced PATCH
  toggleExcluded(sampleId: string, excluded: boolean): Promise<void>;      // works on missing files too
  flushSample(sampleId: string): Promise<void>;                           // blur / Enter
  revertSampleField(sampleId: string, field: EditableField): void;         // Escape
  clearSaveError(): void;
  bulkSet(set: BulkSetInput): Promise<void>;
  toggleSampleSelected(sampleId: string): void;
  setSelectedSampleIds(ids: string[]): void;
  clearSelection(): void;
  setOpenSampleId(sampleId: string | null): void;
  startLabel(opts: LabelOptions): Promise<void>;
  startGenius(opts: GeniusOptions): Promise<void>;
  startCaption(opts: CaptionOptions): Promise<void>;
  startBuild(outputPath?: string): Promise<void>;
  loadPreprocessStatus(): Promise<void>;
  startPreprocess(opts: PreprocessOptions): Promise<void>;
  deletePreprocessVariant(variantKey: string): Promise<void>;
  loadTrainLmStatus(q?: { variantKey?: string; adapterName?: string; lmSize?: LmSize }): Promise<void>;
  startTrainLm(opts: TrainLmOptions): Promise<void>;
  /** MiniMax-Music3: audio -> RVQ codes. */
  startMm3Codes(opts?: { launder?: boolean }): Promise<void>;
  /** MiniMax-Music3: codes + captions -> an LM LoRA. */
  startMm3TrainLm(opts: trainingApi.Mm3TrainLmRequest): Promise<void>;
  /** MiniMax-Music3: carry on training into an existing run directory. Same
   *  adapter, same recipe, more steps — the optimizer state comes back off
   *  disk, so it continues rather than restarts. */
  resumeMm3TrainLm(opts: trainingApi.Mm3ResumeRequest): Promise<void>;
  loadTrainDitStatus(q?: { variantKey?: string; adapterName?: string }): Promise<void>;
  startTrainDit(opts: TrainDitOptions): Promise<void>;
  loadAuditions(): Promise<void>;
  startAudition(opts: AuditionOptions): Promise<void>;
  auditionSample(sampleId: string): Promise<void>;
  cancelJob(): Promise<void>;
  applyStreamEvent(ev: TrainingStreamEvent): void;   // called by useTrainingStream
  loadPipelines(): Promise<void>;
  startPipeline(input: StartPipelineInput): Promise<PipelineSummary>;
  cancelPipeline(id: string): Promise<void>;
  pausePipeline(id: string): Promise<void>;
  resumePipeline(id: string): Promise<void>;
}

/** Split a detail payload into the row map + display order. */
function indexSamples(samples: TrainingSample[]): { byId: Record<string, TrainingSample>; order: string[] } {
  const byId: Record<string, TrainingSample> = {};
  for (const s of samples) byId[s.sampleId] = s;
  const order = [...samples]
    .sort((a, b) => a.relPath.localeCompare(b.relPath))
    .map(s => s.sampleId);
  return { byId, order };
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  phase: 'dataset',
  step: 'label',
  selectedDatasetId: null,

  capabilities: null,
  datasets: [],
  detail: null,
  samplesById: {},
  sampleOrder: [],

  activeJob: null,
  jobLog: [],

  preprocessStatus: null,
  preprocessLoading: false,

  trainLmStatus: null,
  trainLmLoading: false,
  trainLmEpochs: [],
  trainLmLast: null,
  mm3Live: null,
  trainEvalSeries: [],
  trainLmSkippedLong: 0,
  trainLmVram: null,

  trainDitStatus: null,
  trainDitLoading: false,
  trainDitEpochs: [],
  trainDitLast: null,

  trainStepSeries: [],
  trainMilestones: [],
  trainTargetLoss: 0,
  trainMaxEpochs: 0,
  trainStepsPerEpoch: 0,
  trainPreviews: [],

  auditions: [],
  auditionRunning: false,
  auditionError: null,
  samplePreviews: {},
  samplePreviewSources: {},
  samplePreviewPending: {},
  samplePreviewErrors: {},

  pipelines: [],
  pipelinesLoading: false,

  selectedSampleIds: new Set<string>(),
  openSampleId: null,
  pendingEdits: {},
  savingSampleIds: new Set<string>(),
  saveError: null,
  loading: false,
  error: null,

  setPhase: (phase) => set({ phase }),
  setStep: (step) => set({ step }),

  closeDataset: () => set({
    selectedDatasetId: null,
    detail: null,
    samplesById: {},
    sampleOrder: [],
    activeJob: null,
    jobLog: [],
    // Variant cards are per-dataset; keeping them would attribute one dataset's
    // caches to the next one opened.
    preprocessStatus: null,
    preprocessLoading: false,
    // Same reasoning: the adapter dir is keyed on the dataset slug, so B would
    // otherwise render A's trained-adapter card.
    trainLmStatus: null,
    trainLmLoading: false,
    trainLmEpochs: [],
    trainLmLast: null,
    mm3Live: null,
    trainEvalSeries: [],
    trainLmSkippedLong: 0,
    trainLmVram: null,
    trainDitStatus: null,
    trainDitLoading: false,
    trainDitEpochs: [],
    trainDitLast: null,
    ...blankTrainSeries(),
    // Previews are per-dataset (and their URLs point at that dataset's files),
    // so carrying them over would attribute A's auditions to B.
    auditions: [],
    auditionRunning: false,
    auditionError: null,
    samplePreviews: {},
    samplePreviewSources: {},
    samplePreviewPending: {},
    samplePreviewErrors: {},
    selectedSampleIds: new Set<string>(),
    openSampleId: null,
    pendingEdits: {},
    savingSampleIds: new Set<string>(),
    saveError: null,
    // Leaving a dataset must not carry its error back to the list, where
    // TrainingStudio would read it as a fatal load failure.
    error: null,
    step: 'label',
  }),

  loadCapabilities: async () => {
    try {
      const capabilities = await trainingApi.getCapabilities();
      set({ capabilities });
    } catch (err) {
      // Capabilities are advisory — a failure must never blank the studio.
      console.warn('[Training] capabilities failed:', errMessage(err));
      set({ capabilities: null });
    }
  },

  loadDatasets: async () => {
    set({ loading: true, error: null });
    try {
      const datasets = await trainingApi.listDatasets();
      set({ datasets, loading: false });
    } catch (err) {
      set({ error: errMessage(err), loading: false });
    }
  },

  openDataset: async (id) => {
    const isSwitch = get().selectedDatasetId !== id;
    set({ loading: true, error: null });
    try {
      const detail = await trainingApi.getDataset(id);
      const { byId, order } = indexSamples(detail.samples);
      set({
        detail,
        selectedDatasetId: id,
        samplesById: byId,
        sampleOrder: order,
        loading: false,
        ...(isSwitch
          ? {
              step: detail.labeledCount > 0 ? ('review' as const) : ('label' as const),
              selectedSampleIds: new Set<string>(),
              openSampleId: null,
              pendingEdits: {},
              jobLog: [],
              // Per-dataset, exactly like closeDataset. The Preprocess panel's
              // own picker calls openDataset(), so without this A's variant
              // cards stay rendered under B — and Delete would then post B's id
              // with A's variantKey (identical keys across datasets, since the
              // key is just the DiT base name) and rm -rf the wrong cache.
              preprocessStatus: null,
              preprocessLoading: false,
              trainLmStatus: null,
              trainLmLoading: false,
              trainLmEpochs: [],
              trainLmLast: null,
              mm3Live: null,
              trainEvalSeries: [],
              trainLmSkippedLong: 0,
              trainLmVram: null,
              trainDitStatus: null,
              trainDitLoading: false,
              trainDitEpochs: [],
              trainDitLast: null,
              ...blankTrainSeries(),
              auditions: [],
              auditionRunning: false,
              auditionError: null,
              samplePreviews: {},
              samplePreviewSources: {},
              samplePreviewPending: {},
              samplePreviewErrors: {},
            }
          : {}),
      });
      if (detail.activeJobId) {
        try { set({ activeJob: await trainingApi.getJob(detail.activeJobId) }); } catch { /* job may have just ended */ }
      } else if (isSwitch) {
        set({ activeJob: null });
      }
    } catch (err) {
      set({ error: errMessage(err), loading: false });
    }
  },

  createDataset: async (input) => {
    const detail = await trainingApi.createDataset(input);
    const { byId, order } = indexSamples(detail.samples);
    set({
      detail,
      selectedDatasetId: detail.id,
      samplesById: byId,
      sampleOrder: order,
      step: 'label',
      selectedSampleIds: new Set<string>(),
      openSampleId: null,
      pendingEdits: {},
      jobLog: [],
      activeJob: null,
    });
    await get().loadDatasets();
    return detail.id;
  },

  patchDataset: async (patch) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const summary = await trainingApi.patchDataset(id, patch);
      const detail = get().detail;
      set({
        detail: detail ? { ...detail, ...summary } : detail,
        datasets: get().datasets.map(d => (d.id === id ? { ...d, ...summary } : d)),
      });
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  deleteDataset: async (id) => {
    await trainingApi.deleteDataset(id);
    if (get().selectedDatasetId === id) get().closeDataset();
    await get().loadDatasets();
  },

  rescan: async () => {
    const id = get().selectedDatasetId;
    if (!id) return;
    set({ loading: true, error: null });
    try {
      const detail = await trainingApi.rescanDataset(id);
      const { byId, order } = indexSamples(detail.samples);
      set({ detail, samplesById: byId, sampleOrder: order, loading: false });
    } catch (err) {
      set({ error: errMessage(err), loading: false });
    }
  },

  editSample: async (sampleId, patch) => {
    const sample = get().samplesById[sampleId];
    // Rows the job is currently rewriting are read-only.
    if (sample && sample.labelStatus === 'processing') return;

    // 1. Optimistic — the cell renders from samplesById merged with pendingEdits.
    set({
      pendingEdits: { ...get().pendingEdits, [sampleId]: { ...get().pendingEdits[sampleId], ...patch } },
      saveError: null,
    });

    // 2. Debounce 500 ms per (sampleId, field).
    for (const field of Object.keys(patch) as EditableField[]) {
      const key = timerKey(sampleId, field);
      clearTimer(key);
      const handle = window.setTimeout(() => {
        editTimers.delete(key);
        void flushFields(set, get, sampleId, [field]);
      }, EDIT_DEBOUNCE_MS);
      editTimers.set(key, handle);
    }
  },

  toggleExcluded: async (sampleId, excluded) => {
    const id = get().selectedDatasetId;
    const sample = get().samplesById[sampleId];
    if (!id || !sample) return;

    // PATCH 409s on a sample whose audio has vanished (§2.4), but excluding it
    // is the documented way to unblock Build (§5.5) — and the bulk endpoint
    // allows an exclude-only change on a missing file. Route it there.
    if (sample.fileMissing) {
      try {
        const result = await trainingApi.bulkSetSamples(id, [sampleId], { excluded });
        if (result.failed.length > 0) throw new Error(result.failed[0].error);
        set({ samplesById: { ...get().samplesById, [sampleId]: { ...sample, excluded } }, saveError: null });
      } catch (err) {
        set({ saveError: { sampleId, message: errMessage(err) } });
      }
      return;
    }

    await get().editSample(sampleId, { excluded });
    await get().flushSample(sampleId);
  },

  flushSample: async (sampleId) => {
    const pending = get().pendingEdits[sampleId];
    const fields = pending ? (Object.keys(pending) as EditableField[]) : [];
    for (const f of fields) clearTimer(timerKey(sampleId, f));
    if (fields.length === 0) return;
    await flushFields(set, get, sampleId, fields);
  },

  revertSampleField: (sampleId, field) => {
    clearTimer(timerKey(sampleId, field));
    const pending = get().pendingEdits[sampleId];
    if (!pending) return;
    const next = { ...pending };
    delete next[field];
    const edits = { ...get().pendingEdits };
    if (Object.keys(next).length === 0) delete edits[sampleId];
    else edits[sampleId] = next;
    set({ pendingEdits: edits, saveError: null });
  },

  clearSaveError: () => set({ saveError: null }),

  bulkSet: async (bulk) => {
    const id = get().selectedDatasetId;
    const ids = Array.from(get().selectedSampleIds);
    if (!id || ids.length === 0) return;
    try {
      await trainingApi.bulkSetSamples(id, ids, bulk);
      await get().openDataset(id);
    } catch (err) {
      set({ saveError: { sampleId: '', message: errMessage(err) } });
    }
  },

  toggleSampleSelected: (sampleId) => {
    const next = new Set(get().selectedSampleIds);
    if (next.has(sampleId)) next.delete(sampleId);
    else next.add(sampleId);
    set({ selectedSampleIds: next });
  },

  setSelectedSampleIds: (ids) => set({ selectedSampleIds: new Set(ids) }),
  clearSelection: () => set({ selectedSampleIds: new Set<string>() }),
  setOpenSampleId: (sampleId) => set({ openSampleId: sampleId }),

  startLabel: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startLabel(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  startGenius: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startGenius(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  startCaption: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startCaption(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  startBuild: async (outputPath) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startBuild(id, outputPath);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  loadPreprocessStatus: async () => {
    const id = get().selectedDatasetId;
    if (!id) { set({ preprocessStatus: null }); return; }
    set({ preprocessLoading: true });
    try {
      const preprocessStatus = await trainingApi.getPreprocessStatus(id);
      // A slow request must not overwrite a newer dataset's status — but it must
      // still clear the spinner, or a stale response landing last leaves it
      // spinning forever next to the "Tensor caches" heading.
      if (get().selectedDatasetId !== id) { set({ preprocessLoading: false }); return; }
      set({ preprocessStatus, preprocessLoading: false });
    } catch (err) {
      // Advisory — a failed status read must never blank the panel's controls,
      // but it must not leave ANOTHER dataset's cards on screen either.
      console.warn('[Training] preprocess status failed:', errMessage(err));
      if (get().selectedDatasetId === id) set({ preprocessStatus: null, preprocessLoading: false });
      else set({ preprocessLoading: false });
    }
  },

  startPreprocess: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startPreprocess(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  deletePreprocessVariant: async (variantKey) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      await trainingApi.deletePreprocessVariant(id, variantKey);
      set({ error: null });
      await get().loadPreprocessStatus();
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  loadTrainLmStatus: async (q) => {
    const id = get().selectedDatasetId;
    if (!id) { set({ trainLmStatus: null }); return; }
    // No argument = "refresh whatever the panel last asked for" (see
    // lastTrainLmQuery), not "read the 0.6B/<slug> directory".
    const query = q ?? lastTrainLmQuery;
    if (q) lastTrainLmQuery = q;
    const seq = ++trainLmSeq;
    set({ trainLmLoading: true });
    try {
      const trainLmStatus = await trainingApi.getTrainLmStatus(id, query);
      // A superseded request must not win: a newer one is in flight and owns the
      // spinner. A slow request must not overwrite a newer dataset's status, but
      // it must still clear the spinner — same rule as loadPreprocessStatus.
      if (seq !== trainLmSeq) return;
      if (get().selectedDatasetId !== id) { set({ trainLmLoading: false }); return; }
      set({ trainLmStatus, trainLmLoading: false });
    } catch (err) {
      // Advisory: a failed status read must never blank the panel's controls.
      console.warn('[Training] train-lm status failed:', errMessage(err));
      if (seq !== trainLmSeq) return;
      if (get().selectedDatasetId === id) set({ trainLmStatus: null, trainLmLoading: false });
      else set({ trainLmLoading: false });
    }
  },

  startMm3Codes: async (opts?: { launder?: boolean }) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startMm3Codes(id, opts?.launder ? { launder: true } : {});
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  startMm3TrainLm: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startMm3TrainLm(id, opts);
      // MM3 reports STEPS, not epochs, so the epoch series stays empty and the
      // chart draws the step layer alone. blankTrainSeries() clears whatever a
      // previous ACE run left behind.
      set({ jobLog: [], error: null, ...blankTrainSeries(), trainMaxEpochs: 0,
        // Drawn immediately rather than waiting for the engine's own
        // announcement, which arrives a model load later.
        trainTargetLoss: opts.stopMode === 'loss' ? (opts.targetLoss ?? 0) : 0 });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  resumeMm3TrainLm: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.resumeMm3TrainLm(id, opts);
      // The chart starts empty and refills from the resumed run's own step
      // events, which carry on from where it left off — so the curve picks up
      // at step N rather than redrawing the whole history.
      set({ jobLog: [], error: null, ...blankTrainSeries(), trainMaxEpochs: 0,
        trainTargetLoss: opts.stopMode === 'loss' ? (opts.targetLoss ?? 0) : 0 });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  startTrainLm: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startTrainLm(id, opts);
      set({ jobLog: [], error: null, trainLmEpochs: [], trainLmLast: null, trainLmSkippedLong: 0,
        trainLmVram: null,
        // Seeded from the options the run actually started with; the engine's
        // own `epochs` field confirms the cap on the first epoch metric.
        ...blankTrainSeries(),
        trainTargetLoss: opts.targetLoss ?? 0,
        trainMaxEpochs: opts.epochs ?? 0 });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  loadTrainDitStatus: async (q) => {
    const id = get().selectedDatasetId;
    if (!id) { set({ trainDitStatus: null }); return; }
    // No argument = "refresh whatever the panel last asked for", not "read the
    // <slug> directory" — same rule as loadTrainLmStatus.
    const query = q ?? lastTrainDitQuery;
    if (q) lastTrainDitQuery = q;
    const seq = ++trainDitSeq;
    set({ trainDitLoading: true });
    try {
      const trainDitStatus = await trainingApi.getTrainDitStatus(id, query);
      if (seq !== trainDitSeq) return;
      if (get().selectedDatasetId !== id) { set({ trainDitLoading: false }); return; }
      set({ trainDitStatus, trainDitLoading: false });
    } catch (err) {
      // Advisory: a failed status read must never blank the panel's controls.
      console.warn('[Training] train-dit status failed:', errMessage(err));
      if (seq !== trainDitSeq) return;
      if (get().selectedDatasetId === id) set({ trainDitStatus: null, trainDitLoading: false });
      else set({ trainDitLoading: false });
    }
  },

  startTrainDit: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    try {
      const { jobId } = await trainingApi.startTrainDit(id, opts);
      ditVram = { crop: 0, layers: 0 };
      set({ jobLog: [], error: null, trainDitEpochs: [], trainDitLast: null,
        ...blankTrainSeries(),
        trainTargetLoss: opts.targetLoss ?? 0,
        trainMaxEpochs: opts.epochs ?? 0 });
      await adoptJob(set, get, jobId);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  loadAuditions: async () => {
    const id = get().selectedDatasetId;
    if (!id) { set({ auditions: [] }); return; }
    try {
      const res = await trainingApi.getAuditions(id);
      // A slow read must not attribute one dataset's previews to the next.
      if (get().selectedDatasetId !== id) return;
      set({ auditions: res.previews });
    } catch (err) {
      // Advisory — the history list is not worth blanking the card for. The
      // route does not exist until the server workstream lands, and a 404 here
      // must leave the Run button perfectly usable.
      console.warn('[Training] audition list failed:', errMessage(err));
      if (get().selectedDatasetId === id) set({ auditions: [] });
    }
  },

  startAudition: async (opts) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    set({ auditionRunning: true, auditionError: null });
    try {
      const { jobId } = await trainingApi.startAudition(id, opts);
      set({ jobLog: [], error: null });
      await adoptJob(set, get, jobId);
    } catch (err) {
      // The audition's own errors belong on the audition card — `error` is the
      // panel-wide banner and would outlive the card the user is looking at.
      set({ auditionError: errMessage(err) });
    } finally {
      set({ auditionRunning: false });
    }
  },

  auditionSample: async (sampleId) => {
    const id = get().selectedDatasetId;
    if (!id) return;
    set({
      samplePreviewPending: { ...get().samplePreviewPending, [sampleId]: true },
      samplePreviewErrors: { ...get().samplePreviewErrors, [sampleId]: '' },
    });
    try {
      const res = await trainingApi.auditionSample(id, sampleId);
      if (get().selectedDatasetId !== id) return;
      set({
        samplePreviews: { ...get().samplePreviews, [sampleId]: res.preview },
        samplePreviewSources: { ...get().samplePreviewSources, [sampleId]: res.source },
      });
    } catch (err) {
      if (get().selectedDatasetId !== id) return;
      set({ samplePreviewErrors: { ...get().samplePreviewErrors, [sampleId]: errMessage(err) } });
    } finally {
      const pending = { ...get().samplePreviewPending };
      delete pending[sampleId];
      set({ samplePreviewPending: pending });
    }
  },

  cancelJob: async () => {
    const job = get().activeJob;
    if (!job) return;
    try {
      await trainingApi.cancelJob(job.id);
      set({ activeJob: { ...job, status: 'cancelled' } });
      // That status change unmounts the EventSource, so the server's terminal
      // `status` frame may never arrive — run the authoritative refresh here.
      refreshAfterJob(get, job.id);
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  loadPipelines: async () => {
    set({ pipelinesLoading: true });
    try {
      const pipelines = await trainingApi.listPipelines();
      set({ pipelines, pipelinesLoading: false });
    } catch (err) {
      // Advisory — a failed poll must never blank the monitor panel mid-run.
      console.warn('[Training] pipelines failed:', errMessage(err));
      set({ pipelinesLoading: false });
    }
  },

  startPipeline: async (input) => {
    const pipeline = await trainingApi.startPipeline(input);
    await get().loadPipelines();
    return pipeline;
  },

  cancelPipeline: async (id) => {
    try {
      await trainingApi.cancelPipeline(id);
      await get().loadPipelines();
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  pausePipeline: async (id) => {
    try {
      await trainingApi.pausePipeline(id);
      await get().loadPipelines();
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  resumePipeline: async (id) => {
    try {
      await trainingApi.resumePipeline(id);
      await get().loadPipelines();
    } catch (err) {
      set({ error: errMessage(err) });
    }
  },

  applyStreamEvent: (ev) => {
    switch (ev.type) {
      case 'job':
        set({ activeJob: ev.job });
        // finishJob() emits this terminal 'job' frame BEFORE the 'status'
        // frame — and setting a terminal activeJob closes the EventSource
        // (useTrainingStream keys on jobActive), racing the status frame's
        // delivery. When the close won, refreshAfterJob never ran and a
        // finished audition's previews (or a trained adapter's status) only
        // appeared after a manual page refresh. Terminal is terminal whichever
        // frame says it first; refreshedJobs dedupes the double-fire.
        if (ev.job.status !== 'queued' && ev.job.status !== 'running') {
          refreshAfterJob(get, ev.job.id);
        }
        break;

      case 'progress': {
        const job = get().activeJob;
        if (!job) break;
        set({
          activeJob: {
            ...job,
            done: ev.done,
            total: ev.total,
            failed: ev.failed,
            phase: ev.phase,
            currentSampleId: ev.currentSampleId,
            engineQueueDepth: ev.engineQueueDepth,
          },
        });
        break;
      }

      case 'sample': {
        // Idempotent: REPLACE the row keyed by sampleId. The SSE buffer is
        // replayed on every reconnect, so appending would duplicate rows.
        const prev = get().samplesById[ev.sampleId];
        const merged: TrainingSample | undefined = ev.sample
          ? ev.sample
          : prev
            ? { ...prev, labelStatus: ev.status, error: ev.error ?? null }
            : undefined;
        if (!merged) break;
        const order = get().sampleOrder.includes(ev.sampleId)
          ? get().sampleOrder
          : [...get().sampleOrder, ev.sampleId];
        // A finished row is authoritative — drop any optimistic edit for it.
        const edits = { ...get().pendingEdits };
        if (ev.sample) delete edits[ev.sampleId];
        set({
          samplesById: { ...get().samplesById, [ev.sampleId]: merged },
          sampleOrder: order,
          pendingEdits: edits,
        });
        break;
      }

      case 'log': {
        // The whole buffer is replayed on every reconnect, so an append-only
        // handler would duplicate the log each time. Identity is (ts, level, message).
        const cur = get().jobLog;
        if (cur.some(l => l.ts === ev.ts && l.level === ev.level && l.message === ev.message)) break;
        const log = [...cur, { level: ev.level, message: ev.message, ts: ev.ts }];
        set({ jobLog: log.length > JOB_LOG_CAP ? log.slice(log.length - JOB_LOG_CAP) : log });
        break;
      }

      case 'preview': {
        // Append-and-dedupe: the whole SSE buffer replays on every reconnect,
        // and the id (step + kind, or base + kind) is stable per render.
        const cur = get().trainPreviews;
        if (cur.some(p => p.id === ev.preview.id && p.run === ev.run)) break;
        const next = [...cur, { ...ev.preview, run: ev.run }];
        // Oldest first, base pinned to the front — the reference belongs at the
        // start of the strip, and it is the only row with step 0.
        next.sort((a, b) => (a.step - b.step) || a.kind.localeCompare(b.kind));
        set({ trainPreviews: next });
        break;
      }

      case 'metric': {
        // ── Chart series, shared by both training kinds ────────────────────
        // Only one job can own a dataset, so a single set of series is enough
        // and neither kind needs its own copy. Everything below is idempotent:
        // the SSE buffer is replayed on every reconnect.
        const chartKind = get().activeJob?.kind;
        // mm3-train-lm emits the SAME step/milestone metrics (the binary speaks
        // train-lm's JSONL vocabulary), so it draws on the same chart with no
        // other change.
        if (chartKind === 'train-lm' || chartKind === 'train-dit' || chartKind === 'mm3-train-lm') {
          if (ev.metric === 'data' && typeof ev.stepsPerEpoch === 'number' && ev.stepsPerEpoch > 0) {
            set({ trainStepsPerEpoch: ev.stepsPerEpoch });
          }
          // MM3 needs no x-axis special case any more: it trains in shuffled
          // passes, so it publishes a real stepsPerEpoch and real epoch events,
          // and the generic handling above puts the step layer and the epoch
          // line on one fractional-epoch axis exactly as ACE does. What remains
          // here is MM3-only DATA — the live tiles and the held-out series.
          if (chartKind === 'mm3-train-lm') {
            const total = typeof ev.totalSteps === 'number' ? ev.totalSteps : 0;
            if (ev.metric === 'eval' && typeof ev.loss === 'number' && typeof ev.step === 'number') {
              const perEpoch = get().trainStepsPerEpoch;
              const ep = perEpoch > 0 ? ev.step / perEpoch : ev.step;
              // Replace by step, never append: the SSE buffer replays on
              // reconnect and the curve has to be idempotent.
              const next = [...get().trainEvalSeries.filter(e => e.step !== ev.step),
                { step: ev.step, loss: ev.loss, ep }].sort((x, y) => x.step - y.step);
              set({ trainEvalSeries: next });
            }
            const prev = get().mm3Live;
            const base = {
              step: prev?.step ?? 0, totalSteps: prev?.totalSteps ?? 0, loss: prev?.loss ?? 0,
              lr: prev?.lr ?? 0, gradNorm: prev?.gradNorm ?? 0, stepMs: prev?.stepMs ?? 0,
              usedMb: prev?.usedMb ?? 0, totalMb: prev?.totalMb ?? 0,
            };
            if (ev.metric === 'step') {
              set({ mm3Live: { ...base,
                step: typeof ev.step === 'number' ? ev.step : base.step,
                totalSteps: total || base.totalSteps,
                loss: typeof ev.loss === 'number' ? ev.loss : base.loss,
                lr: typeof ev.lr === 'number' ? ev.lr : base.lr,
                gradNorm: typeof ev.gradNorm === 'number' ? ev.gradNorm : base.gradNorm,
                stepMs: typeof ev.stepMs === 'number' ? ev.stepMs : base.stepMs,
              } });
            }
            if (ev.metric === 'vram' && typeof ev.usedMb === 'number') {
              set({ mm3Live: { ...base,
                usedMb: ev.usedMb,
                totalMb: typeof ev.totalMb === 'number' ? ev.totalMb : base.totalMb,
              } });
            }
          }
          if ((ev.metric === 'epoch' || ev.metric === 'step')
            && typeof ev.epochs === 'number' && ev.epochs > 0
            && ev.epochs !== get().trainMaxEpochs) {
            set({ trainMaxEpochs: ev.epochs });
          }
          if (ev.metric === 'step' && typeof ev.step === 'number' && typeof ev.loss === 'number'
            && ev.step > lastStepSeen) {
            lastStepSeen = ev.step;
            const perEpoch = get().trainStepsPerEpoch;
            // Without stepsPerEpoch the integer epoch is the best x available;
            // the layer then draws as short flat runs rather than a smooth line.
            const epPos = perEpoch > 0 ? ev.step / perEpoch : (ev.epoch ?? 0);
            set({
              trainStepSeries: appendStepPoint(
                get().trainStepSeries, { step: ev.step, loss: ev.loss, ep: epPos },
              ),
            });
          }
          // The stopping line of a target-loss run, announced once at the top
          // of it. Set from the EVENT rather than from the form so a run
          // adopted after a page reload still draws its target.
          if (ev.metric === 'target' && typeof ev.loss === 'number' && ev.loss > 0) {
            if (get().trainTargetLoss !== ev.loss) set({ trainTargetLoss: ev.loss });
          }
          if (ev.metric === 'milestone') {
            const path = typeof ev.path === 'string' ? ev.path : '';
            const known = get().trainMilestones;
            if (path && !known.some(m => m.path === path)) {
              set({ trainMilestones: [...known, { epoch: ev.epoch ?? 0, loss: ev.loss ?? 0, path }] });
            }
          }
        }

        // train-dit writes its own slices. Same event shape, same omit-vs-zero
        // discipline as the LM path below; only the destination differs.
        if (get().activeJob?.kind === 'train-dit') {
          if (ev.metric === 'vram') {
            // One-shot, before the first step. Parked in `ditVram` rather than
            // written into trainDitLast — see the note at the top of this file.
            ditVram = {
              crop: typeof ev.crop === 'number' ? ev.crop : ditVram.crop,
              layers: typeof ev.layers === 'number' ? ev.layers : ditVram.layers,
            };
            break;
          }
          if (ev.metric !== 'epoch' && ev.metric !== 'step') break;
          const prevDit = get().trainDitLast;
          const crop = typeof ev.crop === 'number' ? ev.crop : (prevDit?.crop ?? ditVram.crop);
          const layers = prevDit?.layers ?? ditVram.layers;
          if (ev.metric === 'epoch' && typeof ev.epoch === 'number' && typeof ev.loss === 'number') {
            // REPLACE by epoch number, never append — the SSE buffer is replayed
            // on reconnect and the curve must be idempotent.
            const epoch = ev.epoch;
            const next = [...get().trainDitEpochs.filter(e => e.epoch !== epoch), {
              epoch, loss: ev.loss, lr: ev.lr ?? 0,
              gradNorm: ev.gradNorm ?? 0, ms: ev.ms ?? 0,
            }].sort((a, b) => a.epoch - b.epoch);
            set({
              trainDitEpochs: next,
              trainDitLast: {
                loss: ev.loss,
                ma5: ev.ma5 ?? prevDit?.ma5 ?? 0,
                lr: ev.lr ?? prevDit?.lr ?? 0,
                gradNorm: ev.gradNorm ?? prevDit?.gradNorm ?? 0,
                crop, layers,
                etaMs: ev.etaMs ?? prevDit?.etaMs ?? 0,
              },
            });
          } else if (
            typeof ev.loss === 'number' || typeof ev.lr === 'number'
            || typeof ev.gradNorm === 'number' || typeof ev.crop === 'number'
          ) {
            set({
              trainDitLast: {
                loss: ev.loss ?? prevDit?.loss ?? 0,
                ma5: ev.ma5 ?? prevDit?.ma5 ?? 0,
                lr: ev.lr ?? prevDit?.lr ?? 0,
                gradNorm: ev.gradNorm ?? prevDit?.gradNorm ?? 0,
                crop, layers,
                etaMs: ev.etaMs ?? prevDit?.etaMs ?? 0,
              },
            });
          }
          break;
        }

        // §5.5 spells these as `ev.loss ?? 0`, but the runner deliberately OMITS
        // a metric field the engine did not send (trainLmRunner optNum: "a metric
        // field defaulted to 0 is indistinguishable from a real 0 in the loss
        // curve"). Defaulting here re-introduces exactly that: a loss-less epoch
        // frame would plot at 0.0000, drag TrainingChart's y-range and the target
        // line to the floor, and read as "converged". So an epoch with no loss is
        // not plotted, and trainLmLast keeps its previous value per field.
        const prev = get().trainLmLast;
        if (ev.metric === 'epoch' && typeof ev.epoch === 'number' && typeof ev.loss === 'number') {
          // REPLACE by epoch number, never append — the SSE buffer is replayed
          // on reconnect and the curve must be idempotent (same rule as
          // samplesById).
          const epoch = ev.epoch;
          const next = [...get().trainLmEpochs.filter(e => e.epoch !== epoch), {
            epoch, loss: ev.loss, lr: ev.lr ?? 0,
            gradNorm: ev.gradNorm ?? 0, ms: ev.ms ?? 0,
          }].sort((a, b) => a.epoch - b.epoch);
          set({
            trainLmEpochs: next,
            trainLmLast: {
              loss: ev.loss, lr: ev.lr ?? prev?.lr ?? 0,
              gradNorm: ev.gradNorm ?? prev?.gradNorm ?? 0, etaMs: ev.etaMs ?? prev?.etaMs ?? 0,
            },
          });
        } else if (ev.metric === 'epoch' || ev.metric === 'step') {
          if (typeof ev.loss === 'number' || typeof ev.lr === 'number' || typeof ev.gradNorm === 'number') {
            set({
              trainLmLast: {
                loss: ev.loss ?? prev?.loss ?? 0, lr: ev.lr ?? prev?.lr ?? 0,
                gradNorm: ev.gradNorm ?? prev?.gradNorm ?? 0, etaMs: ev.etaMs ?? prev?.etaMs ?? 0,
              },
            });
          }
        } else if (ev.metric === 'data' && typeof ev.skippedLong === 'number') {
          set({ trainLmSkippedLong: ev.skippedLong });
        } else if (ev.metric === 'vram') {
          // 4B plan §1.3: the per-size VRAM hint is driven by this one event.
          // Fires once, before the first step. `mode`/`baseMb`/`ckptMb`/
          // `segPeakMb` only exist for low-VRAM runs, so they default to 0/''.
          set({
            trainLmVram: {
              mode: typeof ev.mode === 'string' ? ev.mode : '',
              maxLen: ev.maxLen ?? 0, estMb: ev.estMb ?? 0, freeMb: ev.freeMb ?? 0,
              baseMb: ev.baseMb ?? 0, ckptMb: ev.ckptMb ?? 0, segPeakMb: ev.segPeakMb ?? 0,
            },
          });
        }
        // 'milestone' carries its own log line — nothing to store.
        break;
      }

      case 'status': {
        const job = get().activeJob;
        if (job) set({ activeJob: { ...job, status: ev.status, error: ev.error ?? job.error } });
        // Authoritative refresh — exactly once per job, replay-safe.
        if (job) refreshAfterJob(get, job.id);
        break;
      }
    }
  },
}));

/** Re-read the dataset once a job reaches a terminal state. Guarded: the SSE
 *  buffer is replayed on reconnect and cancel calls this too. */
function refreshAfterJob(get: () => TrainingState, jobId: string): void {
  const id = get().selectedDatasetId;
  if (!id || refreshedJobs.has(jobId)) return;
  refreshedJobs.add(jobId);
  void get().openDataset(id);
  void get().loadDatasets();
  // A finished train-lm job wrote (or failed to write) the adapter dir — the
  // done-state card reads that from disk, so re-read it here.
  if (get().activeJob?.kind === 'train-lm') void get().loadTrainLmStatus();
  // Same for the DiT adapter dir — the per-epoch export means even a cancelled
  // run usually leaves one on disk, so this refresh matters on every terminal
  // status, not just 'done'.
  if (get().activeJob?.kind === 'train-dit') void get().loadTrainDitStatus();
  // A finished audition wrote data/training/previews/<id>/ — the card renders
  // its players from the list, so re-read it. Placed here rather than inline in
  // applyStreamEvent's `status` case so a cancelled job refreshes too (§6.6).
  if (get().activeJob?.kind === 'audition') void get().loadAuditions();
}

/**
 * Adopt a freshly started job. A quick label over a handful of files (or an
 * immediate failure) can finish before this GET resolves — the stream is then
 * never opened, so the terminal refresh has to happen here instead.
 */
async function adoptJob(
  set: (partial: Partial<TrainingState>) => void,
  get: () => TrainingState,
  jobId: string,
): Promise<void> {
  const job = await trainingApi.getJob(jobId);
  set({ activeJob: job });
  if (job.status !== 'queued' && job.status !== 'running') refreshAfterJob(get, job.id);
}

/**
 * PATCH the given fields of one sample. Optimistic values are kept in
 * `pendingEdits` until the server confirms; on failure they are reverted so
 * the cell snaps back to the on-disk value.
 */
async function flushFields(
  set: (partial: Partial<TrainingState>) => void,
  get: () => TrainingState,
  sampleId: string,
  fields: EditableField[],
): Promise<void> {
  const state = get();
  const datasetId = state.selectedDatasetId;
  const pending = state.pendingEdits[sampleId];
  if (!datasetId || !pending) return;

  const patch: PatchSampleInput = {};
  let any = false;
  for (const f of fields) {
    if (f in pending) { (patch as Record<string, unknown>)[f] = pending[f]; any = true; }
  }
  if (!any) return;

  const saving = new Set(state.savingSampleIds);
  saving.add(sampleId);
  set({ savingSampleIds: saving });

  // Drop only what this request actually sent. If the user kept typing while it
  // was in flight, the newer value is still pending and must survive — clearing
  // it blindly loses those keystrokes and no-ops the queued flush.
  const dropFields = () => {
    const edits = { ...get().pendingEdits };
    const cur = edits[sampleId];
    if (cur) {
      const next = { ...cur } as Record<string, unknown>;
      for (const f of fields) {
        if (Object.is(next[f], (patch as Record<string, unknown>)[f])) delete next[f];
      }
      if (Object.keys(next).length === 0) delete edits[sampleId];
      else edits[sampleId] = next as PatchSampleInput;
    }
    return edits;
  };

  try {
    const sample = await trainingApi.patchSample(datasetId, sampleId, patch);
    const done = new Set(get().savingSampleIds);
    done.delete(sampleId);
    set({
      samplesById: { ...get().samplesById, [sampleId]: sample },
      pendingEdits: dropFields(),
      savingSampleIds: done,
    });
  } catch (err) {
    const done = new Set(get().savingSampleIds);
    done.delete(sampleId);
    set({
      pendingEdits: dropFields(),      // revert the optimistic value
      savingSampleIds: done,
      saveError: { sampleId, message: errMessage(err) },
    });
  }
}

// ── Training ETA: ONE computation, ONE renderer ────────────────────────────
// Every training-run time estimate in the studio comes from here. Two used to
// exist — JobProgress's loss-curve estimate ("6m to target") and the engine's
// own `etaMs` on the metric tile ("ETA 19m", which is time to the epoch CAP,
// not to the target the run stops on) — and they disagreed by design. The
// engine field is still carried in trainLmLast/trainDitLast (it costs nothing
// and the log lines use it), but nothing renders it any more.
//
// Memoised on the identity of the inputs: zustand re-runs a selector on every
// store change and compares the result with Object.is, so returning a fresh
// object each time would re-render the stats strip on every log line.
let etaMemo: { key: readonly unknown[]; value: TrainingEta } | null = null;

/** The live run's ETA, or `{kind:'none'}` when no training job is active. */
export function selectTrainingEta(s: TrainingState): TrainingEta {
  const kind = s.activeJob?.kind;
  if (kind !== 'train-lm' && kind !== 'train-dit') return NO_ETA;
  const series = kind === 'train-dit' ? s.trainDitEpochs : s.trainLmEpochs;
  const key = [kind, series, s.trainTargetLoss, s.trainMaxEpochs] as const;
  if (etaMemo && etaMemo.key.length === key.length && etaMemo.key.every((v, i) => v === key[i])) {
    return etaMemo.value;
  }
  const value = computeTrainingEta(series, s.trainTargetLoss, s.trainMaxEpochs, kind);
  etaMemo = { key, value };
  return value;
}

/** Stable identity so the "nothing to estimate" case never re-renders either. */
const NO_ETA: TrainingEta = { kind: 'none' };

/** The live run's epoch series, whichever kind owns the dataset. */
export function selectTrainingEpochs(s: TrainingState): Array<TrainLmEpoch | TrainDitEpoch> {
  const kind = s.activeJob?.kind;
  if (kind === 'train-dit') return s.trainDitEpochs;
  if (kind === 'train-lm') return s.trainLmEpochs;
  return EMPTY_EPOCHS;
}

const EMPTY_EPOCHS: Array<TrainLmEpoch | TrainDitEpoch> = [];

/** Row as the UI should render it: server truth + any un-saved optimistic edit. */
export function mergedSample(
  sample: TrainingSample | undefined,
  pending: PatchSampleInput | undefined,
): TrainingSample | undefined {
  if (!sample) return undefined;
  return pending ? { ...sample, ...pending } : sample;
}
