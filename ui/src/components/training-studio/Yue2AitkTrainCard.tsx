import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Check, Loader2, Play, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Yue2OptimizerFields } from './Yue2OptimizerFields';
import { YUE2_JOINT_PRESETS_KEY, type Yue2JointPreset } from './yue2JointPresets';
import { TrainingChart } from './TrainingChart';
import {
  cancelJob,
  clearPreparedData,
  getPreparedData,
  getJob,
  getYue2AitkPrepare,
  listYue2AitkRuns,
  linkYue2JointCheckpointPreset,
  listYue2JointPreviews,
  listJobs,
  jobStreamUrl,
  startYue2AitkPrepare,
  startYue2JointTrain,
  type Yue2AitkPrepareRequest,
  type TrainingJobSummary,
  type TrainingMetricEvent,
  type TrainingStreamEvent,
  type Yue2AitkCheckpointRecord,
  type Yue2AitkRunRecord,
  type Yue2JointTrainRequest,
  type Yue2JointPreviewOptions,
  type Yue2JointPreviewRecord,
  type Yue2OptimOptions,
  type PreparedCache,
} from '../../services/trainingApi';
import { useBackendStore } from '../../stores/backendStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { descentRate, formatDurationMs } from '../../utils/trainingEta';

const JOB_KEY = 'hs-yue2-aitk-job:';
const FORM_KEY = 'hs-yue2-aitk-form:';
const PREP_KEY = 'hs-yue2-aitk-prepare:';
const METRIC_CAP = 2000;
type JointStepPoint = { step: number; loss: number; ep: number; arKl?: number; gradNorm?: number; stepMs?: number; elapsedMs?: number; ma5?: number; ma20?: number };
type JointMilestone = { epoch: number; loss: number; path: string };
function jointLossRate(points: JointStepPoint[]): number | null {
  const means = points.map(p => p.ma20).filter((v): v is number => typeof v === 'number');
  return means.length >= 9 ? descentRate(means) : null;
}
function jointEta(points: JointStepPoint[], form: Yue2JointTrainRequest): string {
  const last = points[points.length - 1];
  const durations = points.map(p => p.stepMs).filter((ms): ms is number => typeof ms === 'number' && ms > 0).slice(-20);
  if (!last || !durations.length) return '';
  const pace = durations.reduce((sum, ms) => sum + ms, 0) / durations.length;
  const remaining = Math.max(0, form.steps - last.step);
  if (form.stopMode === 'kl' && form.targetKl && form.targetKl > 0) {
    const kls = points.slice(-20).map(p => p.arKl).filter((v): v is number => typeof v === 'number');
    if (kls.length < 20) return `AR KL warming up · cap ${formatDurationMs(remaining * pace)}`;
    const mean = kls.reduce((s, v) => s + v, 0) / kls.length;
    if (mean >= form.targetKl) return `KL target reached · cap ${formatDurationMs(remaining * pace)}`;
    const older = points.slice(-40, -20).map(p => p.arKl).filter((v): v is number => typeof v === 'number');
    const rate = older.length === 20 ? (mean - older.reduce((s, v) => s + v, 0) / 20) / 20 : 0;
    if (!(rate > 0)) return `AR KL ${mean.toFixed(2)} of ${form.targetKl} · cap ${formatDurationMs(remaining * pace)}`;
    const stepsToTarget = (form.targetKl - mean) / rate;
    return stepsToTarget > remaining
      ? `KL ${mean.toFixed(2)} · target unlikely before cap · cap ${formatDurationMs(remaining * pace)}`
      : `KL ${mean.toFixed(2)} · target ETA ${formatDurationMs(Math.max(1, stepsToTarget) * pace)}`;
  }
  if (form.stopMode !== 'loss' || !(form.targetLoss && form.targetLoss > 0))
    return `cap ETA ${formatDurationMs(remaining * pace)}`;
  const current = last.ma20;
  const rate = jointLossRate(points);
  if (rate === null || current === undefined) return `target ETA estimating · cap ${formatDurationMs(remaining * pace)}`;
  if (current <= form.targetLoss) return `target reached · cap ${formatDurationMs(remaining * pace)}`;
  if (!(rate > 0)) return `target trend stalled · cap ${formatDurationMs(remaining * pace)}`;
  const stepsToTarget = (current - form.targetLoss) / rate;
  if (stepsToTarget > remaining) return `target unlikely before cap · cap ${formatDurationMs(remaining * pace)}`;
  return `target ETA ${formatDurationMs(Math.max(1, stepsToTarget) * pace)}`;
}
/** A named snapshot of the training settings. Per-run and per-machine values
 *  (dataset/checkpoint/output paths, resume record) are deliberately not part
 *  of a preset: a preset answers "how do I train", never "against which run". */
const PRESET_EXCLUDED_KEYS: ReadonlySet<keyof Yue2JointTrainRequest> = new Set([
  'trainingMethod', 'autoPrepare', 'preparation', 'checkpoint', 'dataset', 'output', 'resume', 'alignmentEnabled',
]);
function snapshotPresetSettings(form: Yue2JointTrainRequest, lyricTiming: boolean): Partial<Yue2JointTrainRequest> {
  const settings: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(form)) {
    if (PRESET_EXCLUDED_KEYS.has(key as keyof Yue2JointTrainRequest) || value === undefined) continue;
    settings[key] = value;
  }
  settings.lyricTiming = lyricTiming;
  return settings as Partial<Yue2JointTrainRequest>;
}
const DEFAULT_FORM: Yue2JointTrainRequest = {
  trainingMethod: 'aitk', checkpoint: '', dataset: '', output: '',
  // NAR budget recipe (2026-09-21): the run still ends on the planner's KL to
  // base, but the decoder gets twice the learning rate on the way there.
  // lr 2e-4 with the planner at 0.3x leaves the planner's ABSOLUTE rate at
  // 6e-5 — the same 1e-4 x 0.6 Recipe A used — so the AR walks an unchanged
  // path to KL 1.4 while the decoder, which is where likeness lives, moves
  // twice as far per step. The two halves are independent: the NAR's
  // conditioning prefix is a detached recompute and NAR backward only ever
  // uploads to the NAR adapters (yue2-aitk-joint-step.h), so the decoder's
  // rate cannot perturb the planner's KL. Gen-time NAR strength 2.0 was
  // beating 1.0 by ear; this is that, trained in rather than dialled in.
  // 750 is a cap, not a target — an artist that has not reached KL 1.4 by
  // then is not going to.
  //
  // Prodigy by default (2026-09-21, later the same day): once its weight
  // update was bias-corrected (lm-optim.h) it beat this AdamW recipe by ear
  // AND by clock on the same album — KL 1.4 in 214 steps against AdamW's
  // 308, d settling at ~1e-3 where AdamW had been told 2e-4. `lr` is ignored
  // under Prodigy (base_lr is gamma = 1.0); the planner scale still applies,
  // now on top of d. The pre-fix Prodigy history — "learns too fast,
  // corrupts the AR" — was the missing bias correction, not the optimizer.
  steps: 750, saveEvery: 50, seed: 42, device: 'CUDA0', lyricTiming: true, cursorWeight: 0.08,
  optimizer: 'prodigy', prodigyD0: 1e-6, muonLrScale: 1, muonNsSteps: 5,
  rank: 64, alpha: 64, stopMode: 'kl', targetKl: 1.4, lr: 2e-4, plannerLrScale: 0.3,
};
type PrepareForm = Yue2AitkPrepareRequest;

function defaultPreview(everySteps: number): Yue2JointPreviewOptions {
  return { enabled: false, everySteps, seconds: 40, seed: 424242,
    previewMaxFrames: 1000, baseline: false, control: false };
}

function readStored<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  try {
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) as T : fallback;
  } catch { return fallback; }
}

/** A form saved before a field existed is missing that key, so loading merges
 *  the stored values over the defaults: the user's saved values win, new
 *  fields fill in with their defaults. The persistence effect writes the
 *  merged form back, so this self-heals on the first load after an upgrade. */
function readStoredForm(datasetId: string): Yue2JointTrainRequest {
  const stored = readStored<Partial<Yue2JointTrainRequest>>(`${FORM_KEY}${datasetId}`, {});
  const migration = `${FORM_KEY}${datasetId}:defaults-64`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(migration)) {
    // Move values that match the former defaults; retain deliberate custom values.
    if (stored.steps === 3000) stored.steps = 400;
    if (stored.saveEvery === 250) stored.saveEvery = 50;
    if (stored.rank === 32) stored.rank = 64;
    if (stored.alpha === 32) stored.alpha = 64;
    window.localStorage.setItem(migration, '1');
  }
  // Recipe A (2026-09-20): values still sitting on the old defaults move to
  // the new ones; anything the user changed on purpose stays.
  const recipeA = `${FORM_KEY}${datasetId}:defaults-recipe-a`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(recipeA)) {
    if (stored.rank === 64) stored.rank = 32;
    if (stored.alpha === 64) stored.alpha = 32;
    if (stored.steps === 400) stored.steps = 500;
    if (stored.optimizer === 'prodigy') stored.optimizer = 'adamw';
    if (stored.stopMode === undefined || stored.stopMode === 'steps') { stored.stopMode = 'kl'; stored.targetKl = 1.4; }
    if (stored.lr === undefined) stored.lr = 1e-4;
    if (stored.plannerLrScale === undefined) stored.plannerLrScale = 0.6;
    if (stored.preview?.enabled) stored.preview = { ...stored.preview, enabled: false };
    window.localStorage.setItem(recipeA, '1');
  }
  // NAR budget (2026-09-21): forms still sitting on Recipe A's values move to
  // the new defaults; anything the user set deliberately stays put.
  const narBudget = `${FORM_KEY}${datasetId}:defaults-nar-budget`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(narBudget)) {
    if (stored.rank === 32) stored.rank = 64;
    if (stored.alpha === 32) stored.alpha = 64;
    if (stored.steps === 500) stored.steps = 750;
    if (stored.lr === 1e-4) stored.lr = 2e-4;
    if (stored.plannerLrScale === 0.6) stored.plannerLrScale = 0.3;
    window.localStorage.setItem(narBudget, '1');
  }
  // Prodigy default (2026-09-21): Recipe A had moved everyone to adamw, so a
  // stored adamw is the old default, not a choice, and moves with it.
  const prodigy = `${FORM_KEY}${datasetId}:defaults-prodigy`;
  if (typeof window !== 'undefined' && !window.localStorage.getItem(prodigy)) {
    if (stored.optimizer === 'adamw') stored.optimizer = 'prodigy';
    window.localStorage.setItem(prodigy, '1');
  }
  return { ...DEFAULT_FORM, ...stored };
}
function writeStored(key: string, value: unknown): void {
  if (typeof window === 'undefined') return;
  try { window.localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage may be full or unavailable */ }
}

function isJointJob(job: TrainingJobSummary, datasetId: string): boolean {
  return job.datasetId === datasetId && job.kind === 'yue2-joint-train';
}
function isPrepareJob(job: TrainingJobSummary, datasetId: string): boolean {
  return job.datasetId === datasetId && job.kind === 'yue2-prepare-aitk';
}

export const Yue2AitkTrainCard: React.FC<{ datasetId: string; legacyManifest?: string; cursorReady?: boolean; lyricTiming: boolean; onLyricTimingChange: (value: boolean) => void; exposeStart?: (fn: () => Promise<string | null>) => void }> = ({ datasetId, legacyManifest, cursorReady = false, lyricTiming, onLyricTimingChange, exposeStart }) => {
  const { t } = useTranslation();
  const [form, setForm] = useState<Yue2JointTrainRequest>(() => readStoredForm(datasetId));
  const [job, setJob] = useState<TrainingJobSummary | null>(null);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const [presets, setPresets] = useState<Yue2JointPreset[]>(() => readStored<Yue2JointPreset[]>(YUE2_JOINT_PRESETS_KEY, []));
  const [presetName, setPresetName] = useState('');
  const [presetError, setPresetError] = useState('');
  const [prepare, setPrepare] = useState<PrepareForm>(() => readStored(`${PREP_KEY}${datasetId}`, {
    legacyManifest: legacyManifest ?? '', checkpoint: '', tokenizer: '', output: '',
    models: { vae: '', semantic: '', sheetsage: '' },
  }));
  const [prepareJob, setPrepareJob] = useState<TrainingJobSummary | null>(null);
  const [prepareManifest, setPrepareManifest] = useState('');
  const [appliedPrepareJobId, setAppliedPrepareJobId] = useState(() => readStored<string>(`${PREP_KEY}${datasetId}:applied`, ''));
  const [defaultsAvailable, setDefaultsAvailable] = useState<boolean | null>(null);
  const [missingDefaults, setMissingDefaults] = useState<string[]>([]);
  const [defaultsRevision, setDefaultsRevision] = useState(0);
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const selectModels = useBackendStore(s => s.selectModels);
  const yue2RunAllActive = useTrainingStore(s => s.yue2RunAllActive);
  const batchDraft = useTrainingStore(s => s.yue2BatchDraft);
  const datasets = useTrainingStore(s => s.datasets);
  const setBatchDraft = useTrainingStore(s => s.setYue2BatchDraft);
  const startBatch = useTrainingStore(s => s.startYue2Batch);
  const setPhase = useTrainingStore(s => s.setPhase);
  const [batchStarting, setBatchStarting] = useState(false);
  // A batch draft turns this card into the recipe editor for N datasets: the
  // form is the same, Start sends it to the server-side batch instead of one
  // run, and the per-dataset paths are resolved per item by the runner.
  const runBatch = async () => {
    if (!batchDraft?.length) return;
    setBatchStarting(true); setError('');
    try {
      const timingWeight = lyricTiming
        ? (typeof form.cursorWeight === 'number' && Number.isFinite(form.cursorWeight) ? form.cursorWeight : 0.08) : 0;
      const recipe = { ...form, cursorWeight: timingWeight, dataset: '', output: '', resume: '',
        ...(form.preview ? { preview: { ...defaultPreview(form.saveEvery), ...form.preview,
          everySteps: form.saveEvery, previewMaxFrames: Math.max(8, Math.min(120, form.preview.seconds || 40)) * 25 } } : {}) };
      await startBatch({ datasetIds: batchDraft, lyricTiming, recipe });
      setPhase('train');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { setBatchStarting(false); }
  };
  const [aitkRuns, setAitkRuns] = useState<Yue2AitkRunRecord[]>([]);
  const [resumeChoice, setResumeChoice] = useState('');
  const [clearing, setClearing] = useState(false);
  const [cacheInfo, setCacheInfo] = useState<{ slug: string; caches: PreparedCache[]; busy: boolean } | null>(null);
  const [clearNote, setClearNote] = useState('');
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [applyingCheckpoint, setApplyingCheckpoint] = useState(false);
  const [applyNote, setApplyNote] = useState('');
  const [linkingPreset, setLinkingPreset] = useState(false);
  const [presetLinkNote, setPresetLinkNote] = useState('');
  const [runsError, setRunsError] = useState('');
  const [liveMetric, setLiveMetric] = useState<TrainingMetricEvent | null>(null);
  // AITK's joint runner emits step metrics without an epoch stream. Keep the
  // history here in the same step-domain shape used by the legacy YuE2 chart.
  // The job SSE endpoint replays its buffer, so this also reconstructs the
  // curve after a reload or EventSource reconnect.
  const [stepHistory, setStepHistory] = useState<JointStepPoint[]>([]);
  const [milestones, setMilestones] = useState<JointMilestone[]>([]);
  const [jobLogs, setJobLogs] = useState<string[]>([]);
  const [showJobLogs, setShowJobLogs] = useState(false);
  const [jointPreviews, setJointPreviews] = useState<Yue2JointPreviewRecord[]>([]);

  useEffect(() => {
    if (legacyManifest && !prepare.legacyManifest) {
      setPrepare(previous => ({ ...previous, legacyManifest }));
    }
  }, [legacyManifest, prepare.legacyManifest]);

  useEffect(() => {
    let cancelled = false;
    void getPreparedData(datasetId).then(value => { if (!cancelled) setCacheInfo(value); })
      .catch(() => { if (!cancelled) setCacheInfo(null); });
    return () => { cancelled = true; };
  }, [datasetId]);

  useEffect(() => {
    let cancelled = false;
    void getYue2AitkPrepare(datasetId).then(result => {
      const defaults = result.defaults;
      if (cancelled) return;
      setDefaultsAvailable(!!defaults);
      setMissingDefaults(result.missing ?? []);
      if (!defaults) return;
      setPrepare(previous => ({
        ...previous,
        ...Object.fromEntries(Object.entries(defaults).filter(([key, value]) => key !== 'models' && !previous[key as keyof PrepareForm] && typeof value === 'string')),
        models: { ...previous.models, ...Object.fromEntries(Object.entries(defaults.models ?? {}).filter(([key, value]) => !previous.models[key as keyof PrepareForm['models']] && typeof value === 'string')) },
      }));
    }).catch(() => { /* Defaults are advisory; manual paths remain available. */ });
    return () => { cancelled = true; };
  }, [datasetId, defaultsRevision]);

  useEffect(() => {
    let cancelled = false;
    setRunsError('');
    const refresh = () => listYue2AitkRuns(datasetId).then(result => {
      if (cancelled) return;
      setRunsError('');
      setAitkRuns(result.runs);
      const available = result.runs.flatMap(run => run.checkpoints)
        .filter(checkpoint => checkpoint.arPath && checkpoint.narPath);
      setSelectedCheckpoint(previous => previous && available.some(checkpoint => checkpoint.dir === previous)
        ? previous : (available[0]?.dir ?? ''));
    }).catch(err => {
      if (!cancelled) setRunsError(err instanceof Error ? err.message : String(err));
    });
    void refresh();
    const running = job?.status === 'queued' || job?.status === 'running';
    const timer = running ? window.setInterval(refresh, 5000) : undefined;
    return () => { cancelled = true; if (timer !== undefined) window.clearInterval(timer); };
  }, [datasetId, job?.id, job?.status]);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => listYue2JointPreviews(datasetId, job?.id)
      .then(result => { if (!cancelled) setJointPreviews(result.previews); })
      .catch(() => { if (!cancelled) setJointPreviews([]); });
    void refresh();
    const timer = job?.status === 'queued' || job?.status === 'running'
      ? window.setInterval(refresh, 5000) : undefined;
    return () => { cancelled = true; if (timer !== undefined) window.clearInterval(timer); };
  }, [datasetId, job?.id, job?.status]);

  useEffect(() => {
    setLiveMetric(null);
    setStepHistory([]);
    setMilestones([]);
    setJobLogs([]);
    setShowJobLogs(false);
    if (!job?.id) return;
    const metricKey = `${JOB_KEY}${datasetId}:metrics:${job.id}`;
    const saved = readStored<{ steps?: JointStepPoint[]; milestones?: JointMilestone[] }>(metricKey, {});
    if (saved.steps?.length) setStepHistory(saved.steps);
    const savedMilestones = readStored<JointMilestone[]>(`${metricKey}:milestones`, saved.milestones ?? []);
    if (savedMilestones.length) setMilestones(savedMilestones);
    const stream = new EventSource(jobStreamUrl(job.id));
    stream.onmessage = event => {
      try {
        const item = JSON.parse(event.data) as TrainingStreamEvent;
        if (item.type === 'metric' && item.metric === 'step') {
          setLiveMetric(item);
          if (typeof item.step === 'number' && typeof item.loss === 'number'
            && Number.isFinite(item.step) && Number.isFinite(item.loss)) {
            setStepHistory(previous => {
              const prior = previous.filter(point => point.step !== item.step);
              const stepMs = typeof item.stepMs === 'number' ? item.stepMs : undefined;
              const next = [...prior, { step: item.step!, loss: item.loss!, ep: item.step!,
                ...(typeof item.arKl === 'number' ? { arKl: item.arKl } : {}),
                ...(typeof item.gradNorm === 'number' ? { gradNorm: item.gradNorm } : {}),
                ...(typeof job.startedAt === 'number' && typeof item.ts === 'number'
                  ? { elapsedMs: Math.max(0, item.ts - job.startedAt) } : {}),
                ...(stepMs !== undefined ? { stepMs } : {}) }];
              next.sort((a, b) => a.step - b.step);
              const capped = next.slice(-METRIC_CAP);
              const withMean = capped.map((point, index) => ({ ...point,
                ma5: capped.slice(Math.max(0, index - 4), index + 1).reduce((sum, p) => sum + p.loss, 0)
                  / Math.min(5, index + 1),
                ...(index >= 19 ? { ma20: capped.slice(index - 19, index + 1).reduce((sum, p) => sum + p.loss, 0) / 20 } : {}) }));
              writeStored(metricKey, { steps: withMean });
              return withMean;
            });
          }
        } else if (item.type === 'metric' && item.metric === 'milestone'
          && typeof item.step === 'number' && typeof item.path === 'string') {
          setMilestones(previous => {
            const next = [...previous.filter(point => point.path !== item.path), { epoch: item.step!, loss: item.loss ?? 0, path: item.path! }];
            writeStored(`${metricKey}:milestones`, next);
            return next;
          });
        } else if (item.type === 'log') {
          const stamp = new Date(item.ts).toLocaleTimeString();
          setJobLogs(previous => [...previous, `${stamp} ${item.level}: ${item.message}`].slice(-100));
        } else if (item.type === 'status' && !['queued', 'running'].includes(item.status)) {
          stream.close();
        }
      } catch { /* Ignore malformed replay frames; polling remains authoritative. */ }
    };
    stream.onerror = () => { /* EventSource reconnects; job polling handles terminal state. */ };
    return () => stream.close();
  }, [job?.id, job?.startedAt]);

  useEffect(() => {
    setForm(readStoredForm(datasetId));
    setResumeChoice('');
    setPrepare(readStored<PrepareForm>(`${PREP_KEY}${datasetId}`, {
      legacyManifest: legacyManifest ?? '', checkpoint: '', tokenizer: '', output: '',
      models: { vae: '', semantic: '', sheetsage: '' },
    }));
    setPrepareManifest(readStored<string>(`${PREP_KEY}${datasetId}:manifest`, ''));
    setJob(null);
    setPrepareJob(null);
    setAppliedPrepareJobId(readStored<string>(`${PREP_KEY}${datasetId}:applied`, ''));
    setError('');
    let cancelled = false;
    const storedId = readStored<string>(`${JOB_KEY}${datasetId}`, '');
    const storedPrepareId = readStored<string>(`${PREP_KEY}${datasetId}:job`, '');
    const restore = async () => {
      if (storedPrepareId) {
        try {
          const restored = await getJob(storedPrepareId);
          if (!cancelled && isPrepareJob(restored, datasetId)) setPrepareJob(restored);
        } catch { /* Fall back to the job list below. */ }
      }
      if (storedId) {
        try {
          const restored = await getJob(storedId);
          if (!cancelled && isJointJob(restored, datasetId)) {
            setJob(restored);
            return;
          }
        } catch { /* Fall back to the dataset job list. */ }
      }
      try {
        const result = await listJobs(datasetId);
        const latestPrepare = result
          .filter(item => isPrepareJob(item, datasetId))
          .sort((a, b) => b.createdAt - a.createdAt)[0];
        if (!cancelled && latestPrepare) {
          setPrepareJob(latestPrepare);
          window.localStorage.setItem(`${PREP_KEY}${datasetId}:job`, JSON.stringify(latestPrepare.id));
        }
        const latest = result
          .filter(item => isJointJob(item, datasetId))
          .sort((a, b) => b.createdAt - a.createdAt)[0];
        if (!cancelled && latest) {
          setJob(latest);
          window.localStorage.setItem(`${JOB_KEY}${datasetId}`, JSON.stringify(latest.id));
        }
      } catch { /* The form remains usable when history is unavailable. */ }
    };
    void restore();
    return () => { cancelled = true; };
  }, [datasetId]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(`${FORM_KEY}${datasetId}`, JSON.stringify(form));
    }
  }, [datasetId, form]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(`${PREP_KEY}${datasetId}`, JSON.stringify(prepare));
      if (prepareJob) window.localStorage.setItem(`${PREP_KEY}${datasetId}:job`, JSON.stringify(prepareJob.id));
      if (prepareManifest) window.localStorage.setItem(`${PREP_KEY}${datasetId}:manifest`, JSON.stringify(prepareManifest));
      window.localStorage.setItem(`${PREP_KEY}${datasetId}:applied`, JSON.stringify(appliedPrepareJobId));
    }
  }, [datasetId, prepare, prepareJob, prepareManifest, appliedPrepareJobId]);

  useEffect(() => {
    if (typeof window !== 'undefined') window.localStorage.setItem(YUE2_JOINT_PRESETS_KEY, JSON.stringify(presets));
  }, [presets]);

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    const id = job.id;
    const timer = window.setInterval(() => {
      void getJob(id).then(next => {
        if (next.datasetId === datasetId) setJob(next);
      }).catch(err => setError(err instanceof Error ? err.message : String(err)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [datasetId, job?.id, job?.status]);

  useEffect(() => {
    if (!prepareJob || !['queued', 'running'].includes(prepareJob.status)) return;
    const id = prepareJob.id;
    const timer = window.setInterval(() => {
      void getJob(id).then(next => {
        if (next.datasetId === datasetId) {
          setPrepareJob(next);
          if (next.status === 'done' && prepareManifest && appliedPrepareJobId !== next.id) {
            setForm(previous => ({ ...previous, checkpoint: prepare.checkpoint, dataset: prepareManifest }));
            setAppliedPrepareJobId(next.id);
          }
        }
      }).catch(err => setError(err instanceof Error ? err.message : String(err)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [datasetId, prepareJob?.id, prepareJob?.status, prepareManifest, prepare.checkpoint, appliedPrepareJobId]);

  useEffect(() => {
    if (prepareJob?.status === 'done' && prepareManifest && appliedPrepareJobId !== prepareJob.id) {
      setForm(previous => ({ ...previous, checkpoint: prepare.checkpoint, dataset: prepareManifest }));
      setAppliedPrepareJobId(prepareJob.id);
    }
  }, [prepareJob?.id, prepareJob?.status, prepareManifest, prepare.checkpoint, appliedPrepareJobId]);

  const set = <K extends keyof Yue2JointTrainRequest>(key: K, value: Yue2JointTrainRequest[K]) =>
    setForm(previous => ({ ...previous, [key]: value }));
  const optimValue: Yue2OptimOptions = {
    optimizer: form.optimizer ?? 'adamw',
    prodigyD0: form.prodigyD0 ?? 1e-6,
    muonLrScale: form.muonLrScale ?? 1,
    muonNsSteps: form.muonNsSteps ?? 5,
  };
  const savePreset = () => {
    const name = presetName.trim();
    if (!name) { setPresetError(t('trainingStudio.yue2.method.presetNameRequired', 'Give the preset a name first.')); return; }
    if (presets.some(preset => preset.name.toLowerCase() === name.toLowerCase())) {
      setPresetError(t('trainingStudio.yue2.method.presetExists', 'A preset with that name already exists.'));
      return;
    }
    setPresetError('');
    setPresets(previous => [...previous, { version: 2, name, settings: snapshotPresetSettings(form, lyricTiming) }]);
    setPresetName('');
  };
  const loadPreset = (preset: Yue2JointPreset) => {
    setForm(previous => ({ ...previous, ...preset.settings }));
    if (preset.version === 2 && typeof preset.settings.lyricTiming === 'boolean') onLyricTimingChange(preset.settings.lyricTiming);
  };
  const removePreset = (name: string) => {
    setPresets(previous => previous.filter(preset => preset.name !== name));
  };
  const run = async (): Promise<string | null> => {
    setStarting(true); setError('');
    if ((form.stopMode ?? 'steps') === 'kl' && !(typeof form.targetKl === 'number' && form.targetKl > 0)) {
      setError(t('trainingStudio.yue2.method.targetKlRequired', 'Enter an AR KL target above 0 to train until KL.'));
      setStarting(false);
      return null;
    }
    if ((form.stopMode ?? 'steps') === 'loss' && !(typeof form.targetLoss === 'number' && form.targetLoss > 0)) {
      setError(t('trainingStudio.yue2.method.targetLossRequired', 'Enter a target loss above 0 to train until loss.'));
      setStarting(false);
      return null;
    }
    try {
      const timingWeight = lyricTiming
        ? (typeof form.cursorWeight === 'number' && Number.isFinite(form.cursorWeight) ? form.cursorWeight : 0.08)
        : 0;
      const [resumeRunId, resumeStepText] = resumeChoice.split('|');
      const selectedResume = resumeRunId && resumeStepText ? { resumeRunId, resumeStep: Number(resumeStepText) } : {};
      const request = { ...form, lyricTiming, alignmentEnabled: lyricTiming, cursorWeight: timingWeight,
        autoPrepare: !resumeChoice && !form.resume?.trim(), preparation: prepare,
        checkpoint: '', output: '',
        ...(form.preview ? { preview: { ...defaultPreview(form.saveEvery), ...form.preview,
          everySteps: form.saveEvery, previewMaxFrames: Math.max(8, Math.min(120, form.preview.seconds || 40)) * 25 } } : {}),
        ...(form.resume?.trim() && !resumeChoice ? { resume: form.resume.trim() } : {}),
        ...selectedResume };
      const result = await startYue2JointTrain(datasetId, request);
      if (typeof window !== 'undefined') window.localStorage.setItem(`${JOB_KEY}${datasetId}`, JSON.stringify(result.jobId));
      setJob(await getJob(result.jobId));
      return result.jobId;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally { setStarting(false); }
  };
  // "Perform all stages" (Yue2TrainStages) drives its final training stage
  // through this card's own start so it always trains with the form the user
  // sees. The card hands out the freshest `run` after every render.
  const startRef = useRef<(() => Promise<string | null>) | null>(null);
  useEffect(() => {
    startRef.current = run;
    if (exposeStart) exposeStart(() => startRef.current ? startRef.current() : Promise.resolve(null));
  });
  const prepareDataset = async () => {
    setStarting(true); setError('');
    try {
      const result = await startYue2AitkPrepare(datasetId, { ...prepare, lyricTiming });
      setPrepareManifest(result.manifest);
      setAppliedPrepareJobId('');
      setPrepareJob(await getJob(result.jobId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally { setStarting(false); }
  };
  const stop = async () => {
    if (!job) return;
    setError('');
    try {
      await cancelJob(job.id);
      setJob(await getJob(job.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const stopPrepare = async () => {
    if (!prepareJob) return;
    setError('');
    try {
      await cancelJob(prepareJob.id);
      setPrepareJob(await getJob(prepareJob.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const availableCheckpoints: Yue2AitkCheckpointRecord[] = aitkRuns.flatMap(run => run.checkpoints)
    .filter(checkpoint => checkpoint.arPath && checkpoint.narPath);
  const applyCheckpoint = async () => {
    const checkpoint = availableCheckpoints.find(item => item.dir === selectedCheckpoint);
    if (!checkpoint?.arPath || !checkpoint.narPath || activeBackendId !== 'yue2') return;
    setApplyingCheckpoint(true);
    setApplyNote('');
    const unity = {
      lmAdapterArScale: '1', lmAdapterArScaleAttn: '1', lmAdapterArScaleMlp: '1',
      lmAdapterArScaleEarly: '1', lmAdapterArScaleMid: '1', lmAdapterArScaleLate: '1',
      lmAdapterNarScale: '1', lmAdapterNarScaleAttn: '1', lmAdapterNarScaleMlp: '1',
      lmAdapterNarScaleEarly: '1', lmAdapterNarScaleMid: '1', lmAdapterNarScaleLate: '1',
    };
    const ok = await selectModels({ lmAdapterAr: checkpoint.arPath, lmAdapterNar: checkpoint.narPath, ...unity }, activeBackendId);
    setApplyingCheckpoint(false);
    setApplyNote(ok
      ? t('trainingStudio.yue2.method.checkpointApplied', 'AR and NAR adapters applied for the next generation.')
      : t('trainingStudio.yue2.method.checkpointApplyFailed', 'Could not apply this checkpoint. The previous model selection is still active.'));
  };
  const selectResume = (value: string) => {
    setResumeChoice(value);
    if (!value) { setForm(previous => ({ ...previous, resume: '', dataset: '' })); return; }
    const [jobId, stepText] = value.split('|');
    const source = aitkRuns.find(item => item.jobId === jobId);
    const step = Number(stepText);
    if (!source) return;
    const saved = source.options as Partial<Yue2JointTrainRequest> & { alignment?: { enabled?: boolean; cursorWeight?: number } };
    setForm(previous => ({ ...previous, ...saved, trainingMethod: 'aitk',
      checkpoint: '', output: '', resume: '',
      steps: Math.max(previous.steps, step + 400), saveEvery: saved.saveEvery ?? previous.saveEvery,
      dataset: typeof saved.dataset === 'string' ? saved.dataset : '',
      lyricTiming: saved.alignment?.enabled ?? previous.lyricTiming,
      cursorWeight: saved.alignment?.cursorWeight ?? previous.cursorWeight }));
    if (saved.alignment) onLyricTimingChange(saved.alignment.enabled === true);
  };
  const clearCaches = async () => {
    if (!cacheInfo || clearing) return;
    const summary = cacheInfo.caches.map(item => `${item.name}: ${item.files} files, ${(item.bytes / 1048576).toFixed(1)} MiB`).join('\n');
    if (!window.confirm(`Clear all prepared data for ${cacheInfo.slug}?\n\n${summary || 'No generated caches found.'}\n\nSource tracks, sidecars, labels and adapters will remain.`)) return;
    setClearing(true); setClearNote('');
    try {
      await clearPreparedData(datasetId, cacheInfo.slug);
      setResumeChoice('');
      setForm(previous => ({ ...previous, resume: '', dataset: '' }));
      window.localStorage.setItem(`${FORM_KEY}${datasetId}`, JSON.stringify({ ...form, resume: '', dataset: '' }));
      window.localStorage.removeItem(`${PREP_KEY}${datasetId}:manifest`);
      window.localStorage.removeItem(`${PREP_KEY}${datasetId}:applied`);
      window.location.reload();
    } catch (err) {
      setClearNote(err instanceof Error ? err.message : String(err));
      void getPreparedData(datasetId).then(setCacheInfo).catch(() => {});
    } finally { setClearing(false); }
  };
  const linkCheckpointPreset = async () => {
    if (!selectedCheckpoint) return;
    setLinkingPreset(true);
    setPresetLinkNote('');
    try {
      const result = await linkYue2JointCheckpointPreset(datasetId, selectedCheckpoint);
      setPresetLinkNote(result.updated > 0
        ? `Linked this AR/NAR pair to ${result.updated} Lyric Studio album preset${result.updated === 1 ? '' : 's'}.`
        : 'No Lyric Studio album preset is linked to this dataset yet. Export the dataset to Lyric Studio first.');
    } catch (err) {
      setPresetLinkNote(err instanceof Error ? err.message : String(err));
    } finally {
      setLinkingPreset(false);
    }
  };
  const active = job?.status === 'queued' || job?.status === 'running';
  const preparing = prepareJob?.status === 'queued' || prepareJob?.status === 'running';
  const input = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-xs text-zinc-800 dark:text-zinc-200 outline-none focus:border-amber-500/50';
  const field = (label: string, key: string, type = 'text', source: unknown = form, update?: (value: string) => void) => (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{label}</span>
      <input className={input} type={type} value={String((source as Record<string, unknown>)[key] ?? '')} disabled={(!!resumeChoice && ['seed', 'device', 'rank', 'alpha', 'saveEvery', 'cursorWeight'].includes(key)) || active || starting || preparing || yue2RunAllActive}
        onChange={event => update ? update(event.target.value) : set(key as keyof Yue2JointTrainRequest, type === 'number' ? Number(event.target.value) : event.target.value as never)} />
    </label>
  );
  const progress = job && job.total > 0 ? ` · ${job.done}/${job.total}` : '';
  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
      <h3 className="text-sm font-semibold text-amber-700 dark:text-amber-300">
        {t('trainingStudio.yue2.method.aitkTitle', 'Joint Training is selected')}
      </h3>
      <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-2 leading-relaxed">
        {t('trainingStudio.yue2.method.autoTrainHint', 'Start training prepares the dataset automatically, then trains AR and NAR together. Unchanged prepared data is reused.')}
      </p>
      <label className="mt-3 flex items-start gap-2 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer select-none">
        <input type="checkbox" className="mt-0.5 accent-amber-500" checked={lyricTiming} disabled={!!resumeChoice || active || preparing || starting || yue2RunAllActive}
          onChange={event => onLyricTimingChange(event.target.checked)} />
        <span>
          <span className="font-semibold">{t('trainingStudio.yue2.method.lyricTiming', 'Lyric timing supervision')}</span>
          <span className="block text-[11px] text-zinc-500">{lyricTiming
            ? t('trainingStudio.yue2.method.lyricTimingOn', 'Uses vocal stems and forced alignment before training.')
            : t('trainingStudio.yue2.method.lyricTimingOff', 'Skips stems, alignment, and the optional cursor objective.')}</span>
        </span>
      </label>
      {lyricTiming && !cursorReady && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.lyricTimingNeedsAlignment', 'Run vocal stems and lyric alignment below before starting with timing supervision enabled.')}</p>}
      <label className="mt-4 flex flex-col gap-1">
        <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">Resume a previous run</span>
        <select className={input} value={resumeChoice} disabled={active || preparing || starting || yue2RunAllActive}
          onChange={event => selectResume(event.target.value)}>
          <option value="">Start a new run</option>
          {aitkRuns.map(run => run.checkpoints.filter(checkpoint => !!checkpoint.optimizerPath).map(checkpoint => (
            <option key={`${run.jobId}|${checkpoint.step}`} value={`${run.jobId}|${checkpoint.step}`}
              disabled={!!run.resumeError || run.live}>
              {new Date(run.createdAt).toLocaleString()} · step {checkpoint.step} · {run.status}
              {run.resumeError ? ` — ${run.resumeError}` : ''}{run.live ? ' — running' : ''}
            </option>
          )))}
          {aitkRuns.filter(run => !run.checkpoints.some(checkpoint => !!checkpoint.optimizerPath)).map(run => (
            <option key={run.jobId} disabled value={`unavailable:${run.jobId}`}>
              {new Date(run.createdAt).toLocaleString()} · {run.resumeError || 'No saved optimizer checkpoint'}
            </option>
          ))}
        </select>
      </label>
      {resumeChoice && <p className="mt-1 text-[11px] text-zinc-500">The server restores the original dataset, base, optimizer and adapter settings. Set Steps to the total step you want to reach.</p>}
      <details className="mt-4 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 p-3">
        <summary className="cursor-pointer text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.autoPrepareAdvanced', 'Advanced: dataset preparation and model paths')}</summary>
        <p className="text-[11px] text-zinc-500 mt-1">{t('trainingStudio.yue2.method.autoPrepareHint', 'Preparation runs automatically at the start of training. These controls are only needed for custom paths, manual preparation or resuming a run.')}</p>
        {(defaultsAvailable === false || missingDefaults.length > 0) && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.defaultsMissing', 'Model paths were not found automatically. Install the YuE2 training assets in Model Manager or enter their paths here.')} {missingDefaults.join('; ')}</p>}
        <details className="mt-3">
          <summary className="cursor-pointer text-[11px] font-medium text-zinc-600 dark:text-zinc-400">{t('trainingStudio.yue2.method.advancedPaths', 'Advanced paths and provenance')}</summary>
        <button type="button" disabled={active || preparing || starting || yue2RunAllActive} onClick={() => setDefaultsRevision(value => value + 1)} className="mt-2 text-xs text-amber-700 dark:text-amber-300 hover:underline">{t('trainingStudio.yue2.method.refreshPaths', 'Check installed assets again')}</button>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          {field(t('trainingStudio.yue2.method.legacyManifest', 'Existing YuE2 manifest'), 'legacyManifest', 'text', prepare, value => setPrepare(previous => ({ ...previous, legacyManifest: value })))}
          {field(t('trainingStudio.yue2.method.tokenizer', 'Tokenizer path'), 'tokenizer', 'text', prepare, value => setPrepare(previous => ({ ...previous, tokenizer: value })))}
          {field(t('trainingStudio.yue2.method.prepareOutput', 'New prepared output directory'), 'output', 'text', prepare, value => setPrepare(previous => ({ ...previous, output: value })))}
          {field(t('trainingStudio.yue2.method.vae', 'VAE model'), 'vae', 'text', prepare.models, value => setPrepare(previous => ({ ...previous, models: { ...previous.models, vae: value } })))}
          {field(t('trainingStudio.yue2.method.semantic', 'Semantic tokenizer'), 'semantic', 'text', prepare.models, value => setPrepare(previous => ({ ...previous, models: { ...previous.models, semantic: value } })))}
          {field(t('trainingStudio.yue2.method.sheetsage', 'SheetSage model'), 'sheetsage', 'text', prepare.models, value => setPrepare(previous => ({ ...previous, models: { ...previous.models, sheetsage: value } })))}
        </div>
        </details>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          {form.resume?.trim() && field(t('trainingStudio.yue2.method.resumeDataset', 'Prepared manifest for resume'), 'dataset')}
        </div>
        <button type="button" onClick={() => void prepareDataset()} disabled={preparing || active || starting || yue2RunAllActive || !prepare.legacyManifest || !prepare.checkpoint || !prepare.tokenizer || !prepare.output || !prepare.models.vae || !prepare.models.semantic || !prepare.models.sheetsage}
          className="mt-3 px-3 py-1.5 rounded-lg text-xs font-semibold border border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 disabled:opacity-40">
          {preparing ? t('trainingStudio.yue2.method.preparing', 'Preparing dataset…') : t('trainingStudio.yue2.method.prepare', 'Prepare native dataset')}
        </button>
        {prepareJob && <span className="ml-3 text-[11px] text-zinc-600 dark:text-zinc-400">{prepareJob.status} · {prepareJob.phase || 'waiting'}</span>}
        {preparing && <button type="button" onClick={() => void stopPrepare()} className="ml-3 text-xs text-red-600 dark:text-red-400 hover:underline">{t('trainingStudio.yue2.method.cancel', 'Stop')}</button>}
        {prepareJob?.error && <div className="mt-2 text-xs text-red-600 dark:text-red-400">{prepareJob.error}</div>}
      </details>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
        {!resumeChoice && <details className="md:col-span-2"><summary className="cursor-pointer text-[11px] text-zinc-500">Manual resume path</summary>{field(t('trainingStudio.yue2.method.resume', 'Resume record (optional)'), 'resume')}</details>}
        <label className="flex flex-col gap-1">
          <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.stopMode', 'Train until')}</span>
          <select className={input} value={form.stopMode ?? 'steps'} disabled={active || starting || preparing || yue2RunAllActive}
            onChange={event => set('stopMode', event.target.value as 'steps' | 'loss' | 'kl')}>
            <option value="kl">{t('trainingStudio.yue2.method.stopKl', 'AR KL target')}</option>
            <option value="steps">{t('trainingStudio.yue2.method.stopSteps', 'Step count')}</option>
            <option value="loss">{t('trainingStudio.yue2.method.stopLoss', 'Target loss')}</option>
          </select>
        </label>
        {field((form.stopMode ?? 'steps') !== 'steps'
          ? t('trainingStudio.yue2.method.maxSteps', 'Max steps')
          : t('trainingStudio.yue2.method.steps', 'Steps'), 'steps', 'number')}
        {(form.stopMode ?? 'steps') === 'loss' && field(t('trainingStudio.yue2.method.targetLoss', 'Target loss (composite, trailing mean)'), 'targetLoss', 'number')}
        {(form.stopMode ?? 'steps') === 'kl' && field(t('trainingStudio.yue2.method.targetKl', 'AR KL target (trailing mean)'), 'targetKl', 'number')}
        {field(t('trainingStudio.yue2.method.saveEvery', 'Save every'), 'saveEvery', 'number')}
        {field(t('trainingStudio.yue2.method.seed', 'Seed'), 'seed', 'number')}
        {field(t('trainingStudio.yue2.method.device', 'CUDA device'), 'device')}
        {field(t('trainingStudio.yue2.method.rank', 'LoRA rank'), 'rank', 'number')}
        {field(t('trainingStudio.yue2.method.alpha', 'LoRA alpha'), 'alpha', 'number')}
        {lyricTiming && field(t('trainingStudio.yue2.method.cursorWeight', 'Timing loss weight'), 'cursorWeight', 'number')}
      </div>
      {(form.stopMode ?? 'steps') === 'kl' && <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.targetKlHint', 'AR KL is how far the planner has moved from the base model, so it means the same for every artist. Likeness starts near 1.25; planner damage (looping outros) near 1.9. Training stops once the trailing 20-step mean reaches the target; steps is the cap.')}</p>}
      {(form.stopMode ?? 'steps') === 'loss' && <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.targetLossHint', 'Composite = AR CE + 0.2 × AR KL + NAR flow MSE + timing CE × weight. Training stops once the trailing 20-step mean is at or below this.')}</p>}
      <div className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/40 dark:bg-black/5 p-3">
        {resumeChoice ? <p className="text-xs text-zinc-500">Optimizer: {form.optimizer ?? 'adamw'} (restored from the selected run)</p>
          : <Yue2OptimizerFields value={optimValue} onChange={patch => setForm(previous => ({ ...previous, ...patch }))} />}
      </div>
      <div className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/40 dark:bg-black/5 p-3">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.presets', 'Training presets')}</span>
          <span className="text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.presetHint', 'A preset captures the training settings (steps, save cadence, seed, device, optimizer, rank/alpha and stop target) — never the dataset, checkpoint or output paths. Presets are stored in this browser.')}</span>
        </div>
        <div className="mt-2 flex items-center gap-2 flex-wrap">
          <input className={`${input} min-w-40 flex-1`} placeholder={t('trainingStudio.yue2.method.presetNamePlaceholder', 'New preset name')}
            value={presetName} disabled={active || preparing || starting || yue2RunAllActive}
            onChange={event => { setPresetName(event.target.value); setPresetError(''); }}
            onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); void savePreset(); } }} />
          <button type="button" onClick={() => void savePreset()} disabled={active || preparing || starting || yue2RunAllActive}
            className="px-2.5 py-1 rounded-lg text-[11px] font-semibold border border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 disabled:opacity-40">
            {t('trainingStudio.yue2.method.presetSave', 'Save current settings')}
          </button>
        </div>
        {presetError && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{presetError}</p>}
        {presets.length > 0 && <div className="mt-2 flex flex-wrap gap-2">
          {presets.map(preset => (
            <span key={preset.name} className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-300 dark:border-white/10 bg-white/60 dark:bg-zinc-900/40 pl-2.5 pr-1 py-1 text-[11px] text-zinc-700 dark:text-zinc-300">
              <button type="button" onClick={() => loadPreset(preset)} disabled={active || preparing || starting || yue2RunAllActive}
                title={t('trainingStudio.yue2.method.presetLoad', 'Load this preset into the form')}
                className="font-medium hover:underline disabled:no-underline disabled:opacity-40">
                {preset.name}{preset.settings.steps !== undefined && preset.settings.saveEvery !== undefined
                  ? ` · ${preset.settings.steps} steps, save every ${preset.settings.saveEvery}` : ''}
              </button>
              <button type="button" onClick={() => removePreset(preset.name)}
                title={t('trainingStudio.yue2.method.presetRemove', 'Remove this preset')}
                className="rounded p-0.5 text-zinc-400 hover:text-red-600 dark:hover:text-red-400">
                <X size={12} />
              </button>
            </span>
          ))}
        </div>}
      </div>
      <details className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/30 dark:bg-black/10 p-3">
        <summary className="cursor-pointer text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">
          {t('trainingStudio.yue2.method.advancedTitle', 'Advanced: planner and optimizer')}
        </summary>
        <p className="mt-2 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.advancedHint', 'Blank = the engine default. These are the knobs the reference AITK recipe exposes; the defaults are what every run so far has used.')}</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
          {([
            ['lr', t('trainingStudio.yue2.method.lr', 'Learning rate'), 'default 1e-4 · AdamW only (Prodigy learns its own; Muon uses its scale)'],
            ['weightDecay', t('trainingStudio.yue2.method.weightDecay', 'Weight decay'), 'default 1e-4'],
            ['plannerLrScale', t('trainingStudio.yue2.method.plannerLrScale', 'Planner learning-rate scale'), 'default 1.0 · the AR half trains at lr × this. This card ships 0.3, which holds the planner at 6e-5 while the decoder trains at 2e-4'],
            ['klWeight', t('trainingStudio.yue2.method.klWeight', 'KL anchor to base (planner)'), 'default 0.2 · higher keeps the planner closer to the base model'],
            ['abcDropout', t('trainingStudio.yue2.method.abcDropout', 'ABC dropout'), 'default 0.5 · share of lead-sheet examples trained without their sheet, so one adapter serves cot on and off'],
            ['captionDropout', t('trainingStudio.yue2.method.captionDropout', 'Caption dropout'), 'default 0 · share of steps trained on the trigger alone instead of the song\'s caption. 0.5 is the measured recipe: it stops the adapter binding to each track\'s caption, so a NEW caption generalises. Needs a dataset prepared after 2026-09-20.'],
          ] as const).map(([key, label, hint]) => (
            <label key={key} className="flex flex-col gap-1">
              <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{label}</span>
              <input className={input} type="number" step="any" placeholder="engine default"
                value={form[key] ?? ''} disabled={active || starting || preparing || yue2RunAllActive}
                onChange={event => set(key, event.target.value === '' ? undefined : Number(event.target.value))} />
              <span className="text-[10px] text-zinc-500">{hint}</span>
            </label>
          ))}
        </div>
      </details>
      <details className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/30 dark:bg-black/10 p-3">
        <summary className="cursor-pointer text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">
          {t('trainingStudio.yue2.method.previewTitle', 'Checkpoint previews (optional)')}
        </summary>
        <label className="mt-2 flex items-start gap-2 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer select-none">
          <input type="checkbox" className="mt-0.5 accent-amber-500" checked={form.preview?.enabled ?? false}
            disabled={active || preparing || starting || yue2RunAllActive}
            onChange={event => setForm(previous => ({ ...previous, preview: { ...(previous.preview ?? defaultPreview(form.saveEvery)), enabled: event.target.checked } }))} />
          <span>
            <span className="font-semibold">{t('trainingStudio.yue2.method.previewEnable', 'Render one artist sample at saved checkpoints')}</span>
            <span className="block text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.previewManual', 'Off by default. Samples are saved for manual listening; they do not start automatically in the player.')}</span>
          </span>
        </label>
        {form.preview?.enabled && <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          {field(t('trainingStudio.yue2.method.previewSeconds', 'Preview seconds'), 'seconds', 'number', form.preview, value => setForm(previous => {
            const seconds = Math.max(8, Math.min(120, Number(value) || 40));
            return { ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, seconds, previewMaxFrames: seconds * 25 } };
          }))}
          {field(t('trainingStudio.yue2.method.previewSeed', 'Preview seed'), 'seed', 'number', form.preview, value => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, seed: Number(value) } })))}
          <p className="text-[11px] text-zinc-500 md:col-span-2">{t('trainingStudio.yue2.method.previewSongHint', 'The first track in this dataset is used for the preview. Caption and lyrics overrides below are optional.')}</p>
          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300"><input type="checkbox" checked={form.preview.baseline} disabled={active || preparing || starting || yue2RunAllActive} onChange={event => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, baseline: event.target.checked } }))} />{t('trainingStudio.yue2.method.previewBaseline', 'Include baseline')}</label>
          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300"><input type="checkbox" checked={form.preview.control} disabled={active || preparing || starting || yue2RunAllActive} onChange={event => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, control: event.target.checked } }))} />{t('trainingStudio.yue2.method.previewControl', 'Include control')}</label>
          <label className="md:col-span-2 flex flex-col gap-1"><span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.previewCaption', 'Caption override (optional)')}</span><textarea className={`${input} min-h-16 resize-y`} value={form.preview.caption ?? ''} disabled={active || preparing || starting || yue2RunAllActive} onChange={event => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, caption: event.target.value } }))} /></label>
          <label className="md:col-span-2 flex flex-col gap-1"><span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.previewLyrics', 'Lyrics override (optional)')}</span><textarea className={`${input} min-h-20 resize-y`} value={form.preview.lyrics ?? ''} disabled={active || preparing || starting || yue2RunAllActive} onChange={event => setForm(previous => ({ ...previous, preview: { ...defaultPreview(previous.saveEvery), ...previous.preview, lyrics: event.target.value } }))} /></label>
        </div>}
      </details>
      <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.hardware', 'Joint training requires a CUDA build and an NVIDIA GPU with BF16 support (Ampere or newer).')}</p>
      <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.autoOutput', 'Adapters are saved in your global adapters folder under yue2-joint-adapters/triggerword_date_time.')}</p>
      <div className="mt-4 rounded-lg border border-red-500/20 p-3">
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">Prepared data</p>
        <p className="mt-1 text-[11px] text-zinc-500">Clear generated latents, codes, lead sheets, alignment, stems, MM3 caches and ACE tensors for this dataset. Source files, labels and adapters are kept. Older runs may then be unavailable to resume.</p>
        <button type="button" onClick={() => void clearCaches()}
          disabled={!cacheInfo?.caches.length || cacheInfo.busy || active || preparing || starting || clearing || yue2RunAllActive}
          className="mt-2 px-3 py-1.5 rounded-lg text-xs font-semibold border border-red-500/50 text-red-600 dark:text-red-400 hover:bg-red-500/10 disabled:opacity-40">
          {clearing ? 'Clearing…' : 'Clear all prepared data'}
        </button>
        {cacheInfo && <span className="ml-2 text-[11px] text-zinc-500">{cacheInfo.caches.length} cache folders · {(cacheInfo.caches.reduce((sum, item) => sum + item.bytes, 0) / 1048576).toFixed(1)} MiB</span>}
        {clearNote && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{clearNote}</p>}
      </div>
      {error && <div className="mt-3 flex items-start gap-2 text-xs text-red-600 dark:text-red-400"><AlertTriangle size={14} className="mt-0.5 shrink-0" />{error}</div>}
      {job?.error && <div className="mt-2 text-xs text-red-600 dark:text-red-400">{job.error}</div>}
      {batchDraft && batchDraft.length > 0 && <div className="mt-4 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
        <p className="text-xs font-semibold text-amber-700 dark:text-amber-300">
          {t('trainingStudio.yue2.aitkBatch.draftTitle', 'Batch: these settings apply to {{count}} dataset(s)', { count: batchDraft.length })}
        </p>
        <p className="mt-1 text-[11px] text-zinc-600 dark:text-zinc-400 break-words">
          {batchDraft.map(id => datasets.find(d => d.id === id)?.name ?? id).join(' · ')}
        </p>
        <p className="mt-1 text-[11px] text-zinc-500">
          {t('trainingStudio.yue2.aitkBatch.draftHint', 'Each dataset runs its caches, lead sheets, timing (when enabled), preparation and joint training in order with this recipe. The batch runs on the server; the page follows the dataset being trained.')}
        </p>
        <div className="mt-2 flex items-center gap-3">
          <button type="button" onClick={() => void runBatch()} disabled={batchStarting || yue2RunAllActive}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2">
            {batchStarting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {t('trainingStudio.yue2.aitkBatch.start', 'Start batch ({{count}})', { count: batchDraft.length })}
          </button>
          <button type="button" onClick={() => setBatchDraft(null)} className="text-xs text-zinc-500 hover:underline">{t('trainingStudio.yue2.aitkBatch.discard', 'Discard batch')}</button>
        </div>
      </div>}
      <div className="mt-4 flex items-center gap-3 flex-wrap">
        <button type="button" onClick={() => void run()} disabled={!!batchDraft?.length || active || preparing || starting || yue2RunAllActive || (resumeChoice || form.resume?.trim() ? !form.dataset : (!prepare.legacyManifest || !prepare.tokenizer)) || (!resumeChoice && lyricTiming && !cursorReady)}
          className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2">
          {starting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
          {active ? (job?.phase === 'preparing' ? t('trainingStudio.yue2.method.preparing', 'Preparing dataset…') : t('trainingStudio.yue2.method.running', 'Joint training is running')) : t('trainingStudio.yue2.method.start', 'Start joint training')}
        </button>
        {active && <button type="button" onClick={() => void stop()} className="text-xs text-red-600 dark:text-red-400 hover:underline">{t('trainingStudio.yue2.method.cancel', 'Stop')}</button>}
        {job && <span className="text-[11px] text-zinc-600 dark:text-zinc-400">{job.status} · {job.phase || 'waiting'}{progress}
          {liveMetric?.step !== undefined && ` · step ${liveMetric.step}${liveMetric.loss !== undefined ? ` · loss ${liveMetric.loss.toFixed(4)}` : ''}`}
        </span>}
      </div>
      {stepHistory.length > 1 && (
        <div className="mt-3">
          <TrainingChart
            epochs={[]}
            steps={stepHistory}
            milestones={milestones}
            target={form.stopMode === 'loss' ? (form.targetLoss ?? 0) : 0}
            maxEpochs={form.steps}
          />
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] tabular-nums text-zinc-500">
            <span>MA5 {stepHistory[stepHistory.length - 1].ma5?.toFixed(4) ?? '—'}</span>
            <span>20-step stop mean {stepHistory[stepHistory.length - 1].ma20?.toFixed(4) ?? '—'}</span>
            {jointLossRate(stepHistory) !== null && <span>loss rate {jointLossRate(stepHistory)!.toFixed(5)}/step</span>}
            <span>{stepHistory[stepHistory.length - 1].step} / {form.steps} steps</span>
            {stepHistory[stepHistory.length - 1].elapsedMs !== undefined && <span>elapsed {formatDurationMs(stepHistory[stepHistory.length - 1].elapsedMs!)}</span>}
            {stepHistory[stepHistory.length - 1].stepMs !== undefined && <span>pace {(stepHistory.slice(-20).reduce((sum, point) => sum + (point.stepMs ?? 0), 0) / Math.max(1, stepHistory.slice(-20).filter(point => point.stepMs !== undefined).length) / 1000).toFixed(2)}s/step</span>}
            {job?.status === 'running' && <span>{jointEta(stepHistory, form)}</span>}
          </div>
        </div>
      )}
      {jobLogs.length > 0 && <details className="mt-2" open={showJobLogs} onToggle={event => setShowJobLogs(event.currentTarget.open)}>
        <summary className="cursor-pointer text-[11px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.yue2.method.showLogs', 'Show training log (last 100 lines)')}</summary>
        <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-zinc-950 p-2 text-[10px] leading-4 text-zinc-300 whitespace-pre-wrap">{jobLogs.join('\n')}</pre>
      </details>}
      {job?.status === 'done' && <p className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-400">{t('trainingStudio.yue2.method.checkpointWritten', 'Joint checkpoints are in the selected output directory.')}</p>}
      {(aitkRuns.length > 0 || runsError) && <div className="mt-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.auditionTitle', 'Audition a joint checkpoint')}</p>
        {activeBackendId !== 'yue2' && <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.selectYue2', 'Select the YuE2 backend to use these adapters for generation.')}</p>}
        {runsError && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{runsError}</p>}
        {availableCheckpoints.length > 0 && <div className="mt-2 flex items-center gap-2 flex-wrap">
          <select className={input} value={selectedCheckpoint} onChange={event => { setSelectedCheckpoint(event.target.value); setApplyNote(''); setPresetLinkNote(''); }}>
            {availableCheckpoints.map(checkpoint => <option key={checkpoint.dir} value={checkpoint.dir}>step {checkpoint.step}{checkpoint.dir === availableCheckpoints[0]?.dir ? ' (latest)' : ''}</option>)}
          </select>
          <button type="button" onClick={() => void applyCheckpoint()} disabled={activeBackendId !== 'yue2' || active || preparing || applyingCheckpoint || yue2RunAllActive || !selectedCheckpoint}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-emerald-500/50 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-40 flex items-center gap-1.5">
            {applyingCheckpoint ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
            {t('trainingStudio.yue2.method.useCheckpoint', 'Use for generation')}
          </button>
          <button type="button" onClick={() => void linkCheckpointPreset()} disabled={active || preparing || linkingPreset || yue2RunAllActive || !selectedCheckpoint}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-emerald-500/50 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-40 flex items-center gap-1.5">
            {linkingPreset ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
            {t('trainingStudio.yue2.method.linkAlbumPreset', 'Use in Lyric Studio album preset')}
          </button>
        </div>}
        {!runsError && availableCheckpoints.length === 0 && <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.noAuditionCheckpoint', 'No complete AR/NAR checkpoint is available yet.')}</p>}
        {applyNote && <p className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-400">{applyNote}</p>}
        {presetLinkNote && <p className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-400">{presetLinkNote}</p>}
      </div>}
      {jointPreviews.length > 0 && <div className="mt-3 rounded-lg border border-sky-500/30 bg-sky-500/5 p-3">
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.previewStrip', 'Checkpoint previews')}</p>
        <div className="mt-2 flex flex-col gap-2">
          {jointPreviews.map(preview => <div key={preview.id} className="flex items-center gap-2 text-[11px] text-zinc-600 dark:text-zinc-400">
            <span className="w-20 shrink-0">step {preview.step} · {preview.kind}</span>
            <span className="flex-1">{preview.status === 'failed' ? preview.error || 'render failed' : preview.endReason === 'preview_limit' ? t('trainingStudio.yue2.method.previewCapped', 'Preview length reached') : preview.status}</span>
            {preview.score && <span title={preview.score.reason} className={`shrink-0 font-semibold uppercase tracking-wider ${preview.score.verdict === 'healthy' ? 'text-emerald-500' : preview.score.verdict === 'long' ? 'text-amber-500' : 'text-red-500'}`}>
              {preview.score.verdict} · {preview.score.bars} bars · {Math.round(preview.score.vocalShare * 100)}% vocal
            </span>}
            {preview.audioUrl && preview.status === 'done' && <audio controls preload="none" src={preview.audioUrl} className="h-7 max-w-[240px]" />}
          </div>)}
        </div>
      </div>}
    </div>
  );
};

