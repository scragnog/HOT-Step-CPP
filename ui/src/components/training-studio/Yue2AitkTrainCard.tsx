import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Check, Loader2, Play, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Yue2OptimizerFields } from './Yue2OptimizerFields';
import { TrainingChart } from './TrainingChart';
import {
  cancelJob,
  getJob,
  getYue2AitkPrepare,
  listYue2AitkRuns,
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
} from '../../services/trainingApi';
import { useBackendStore } from '../../stores/backendStore';
import { useTrainingStore } from '../../stores/trainingStore';

const JOB_KEY = 'hs-yue2-aitk-job:';
const FORM_KEY = 'hs-yue2-aitk-form:';
const PREP_KEY = 'hs-yue2-aitk-prepare:';
const PRESETS_KEY = 'hs-yue2-joint-presets';
/** A named snapshot of the training settings. Per-run and per-machine values
 *  (dataset/checkpoint/output paths, resume record) are deliberately not part
 *  of a preset: a preset answers "how do I train", never "against which run". */
type Yue2JointPreset = { name: string; settings: Partial<Yue2JointTrainRequest> };
const PRESET_EXCLUDED_KEYS: ReadonlySet<keyof Yue2JointTrainRequest> = new Set([
  'trainingMethod', 'autoPrepare', 'preparation', 'checkpoint', 'dataset', 'output', 'resume', 'alignmentEnabled',
]);
function snapshotPresetSettings(form: Yue2JointTrainRequest): Partial<Yue2JointTrainRequest> {
  const settings: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(form)) {
    if (PRESET_EXCLUDED_KEYS.has(key as keyof Yue2JointTrainRequest) || value === undefined) continue;
    settings[key] = value;
  }
  return settings as Partial<Yue2JointTrainRequest>;
}
const DEFAULT_FORM: Yue2JointTrainRequest = {
  trainingMethod: 'aitk', checkpoint: '', dataset: '', output: '',
  steps: 400, saveEvery: 50, seed: 42, device: 'CUDA0', lyricTiming: true, cursorWeight: 0.08,
  optimizer: 'prodigy', prodigyD0: 1e-6, muonLrScale: 1, muonNsSteps: 5,
  rank: 64, alpha: 64, stopMode: 'steps',
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
  return { ...DEFAULT_FORM, ...stored };
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
  const [presets, setPresets] = useState<Yue2JointPreset[]>(() => readStored<Yue2JointPreset[]>(PRESETS_KEY, []));
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
  const [aitkRuns, setAitkRuns] = useState<Yue2AitkRunRecord[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [applyingCheckpoint, setApplyingCheckpoint] = useState(false);
  const [applyNote, setApplyNote] = useState('');
  const [runsError, setRunsError] = useState('');
  const [liveMetric, setLiveMetric] = useState<TrainingMetricEvent | null>(null);
  // AITK's joint runner emits step metrics without an epoch stream. Keep the
  // history here in the same step-domain shape used by the legacy YuE2 chart.
  // The job SSE endpoint replays its buffer, so this also reconstructs the
  // curve after a reload or EventSource reconnect.
  const [stepHistory, setStepHistory] = useState<Array<{ step: number; loss: number; ep: number }>>([]);
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
    setJobLogs([]);
    setShowJobLogs(false);
    if (!job?.id) return;
    const stream = new EventSource(jobStreamUrl(job.id));
    stream.onmessage = event => {
      try {
        const item = JSON.parse(event.data) as TrainingStreamEvent;
        if (item.type === 'metric' && item.metric === 'step') {
          setLiveMetric(item);
          if (typeof item.step === 'number' && typeof item.loss === 'number'
            && Number.isFinite(item.step) && Number.isFinite(item.loss)) {
            setStepHistory(previous => {
              const next = [...previous.filter(point => point.step !== item.step),
                { step: item.step!, loss: item.loss!, ep: item.step! }];
              next.sort((a, b) => a.step - b.step);
              return next.slice(-2000);
            });
          }
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
  }, [job?.id]);

  useEffect(() => {
    setForm(readStoredForm(datasetId));
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
    if (typeof window !== 'undefined') window.localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
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
    setPresets(previous => [...previous, { name, settings: snapshotPresetSettings(form) }]);
    setPresetName('');
  };
  const loadPreset = (preset: Yue2JointPreset) => {
    setForm(previous => ({ ...previous, ...preset.settings }));
  };
  const removePreset = (name: string) => {
    setPresets(previous => previous.filter(preset => preset.name !== name));
  };
  const run = async (): Promise<string | null> => {
    setStarting(true); setError('');
    if ((form.stopMode ?? 'steps') === 'loss' && !(typeof form.targetLoss === 'number' && form.targetLoss > 0)) {
      setError(t('trainingStudio.yue2.method.targetLossRequired', 'Enter a target loss above 0 to train until loss.'));
      setStarting(false);
      return null;
    }
    try {
      const timingWeight = lyricTiming
        ? (typeof form.cursorWeight === 'number' && Number.isFinite(form.cursorWeight) ? form.cursorWeight : 0.08)
        : 0;
      const request = { ...form, lyricTiming, alignmentEnabled: lyricTiming, cursorWeight: timingWeight,
        autoPrepare: !form.resume?.trim(), preparation: prepare,
        checkpoint: '', output: '',
        ...(form.preview ? { preview: { ...defaultPreview(form.saveEvery), ...form.preview,
          everySteps: form.saveEvery, previewMaxFrames: Math.max(8, Math.min(120, form.preview.seconds || 40)) * 25 } } : {}),
        ...(form.resume?.trim() ? { resume: form.resume.trim() } : {}) };
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
  const active = job?.status === 'queued' || job?.status === 'running';
  const preparing = prepareJob?.status === 'queued' || prepareJob?.status === 'running';
  const input = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-xs text-zinc-800 dark:text-zinc-200 outline-none focus:border-amber-500/50';
  const field = (label: string, key: string, type = 'text', source: unknown = form, update?: (value: string) => void) => (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{label}</span>
      <input className={input} type={type} value={String((source as Record<string, unknown>)[key] ?? '')} disabled={active || starting || preparing || yue2RunAllActive}
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
        <input type="checkbox" className="mt-0.5 accent-amber-500" checked={lyricTiming} disabled={active || preparing || starting || yue2RunAllActive}
          onChange={event => onLyricTimingChange(event.target.checked)} />
        <span>
          <span className="font-semibold">{t('trainingStudio.yue2.method.lyricTiming', 'Lyric timing supervision')}</span>
          <span className="block text-[11px] text-zinc-500">{lyricTiming
            ? t('trainingStudio.yue2.method.lyricTimingOn', 'Uses vocal stems and forced alignment before training.')
            : t('trainingStudio.yue2.method.lyricTimingOff', 'Skips stems, alignment, and the optional cursor objective.')}</span>
        </span>
      </label>
      {lyricTiming && !cursorReady && <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">{t('trainingStudio.yue2.method.lyricTimingNeedsAlignment', 'Run vocal stems and lyric alignment below before starting with timing supervision enabled.')}</p>}
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
        {field(t('trainingStudio.yue2.method.resume', 'Resume record (optional)'), 'resume')}
        <label className="flex flex-col gap-1">
          <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.stopMode', 'Train until')}</span>
          <select className={input} value={form.stopMode ?? 'steps'} disabled={active || starting || preparing || yue2RunAllActive}
            onChange={event => set('stopMode', event.target.value as 'steps' | 'loss')}>
            <option value="steps">{t('trainingStudio.yue2.method.stopSteps', 'Step count')}</option>
            <option value="loss">{t('trainingStudio.yue2.method.stopLoss', 'Target loss')}</option>
          </select>
        </label>
        {field((form.stopMode ?? 'steps') === 'loss'
          ? t('trainingStudio.yue2.method.maxSteps', 'Max steps')
          : t('trainingStudio.yue2.method.steps', 'Steps'), 'steps', 'number')}
        {(form.stopMode ?? 'steps') === 'loss' && field(t('trainingStudio.yue2.method.targetLoss', 'Target loss (composite, trailing mean)'), 'targetLoss', 'number')}
        {field(t('trainingStudio.yue2.method.saveEvery', 'Save every'), 'saveEvery', 'number')}
        {field(t('trainingStudio.yue2.method.seed', 'Seed'), 'seed', 'number')}
        {field(t('trainingStudio.yue2.method.device', 'CUDA device'), 'device')}
        {field(t('trainingStudio.yue2.method.rank', 'LoRA rank'), 'rank', 'number')}
        {field(t('trainingStudio.yue2.method.alpha', 'LoRA alpha'), 'alpha', 'number')}
        {lyricTiming && field(t('trainingStudio.yue2.method.cursorWeight', 'Timing loss weight'), 'cursorWeight', 'number')}
      </div>
      {(form.stopMode ?? 'steps') === 'loss' && <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.yue2.method.targetLossHint', 'Composite = AR CE + 0.2 × AR KL + NAR flow MSE + timing CE × weight. Training stops once the trailing 20-step mean is at or below this.')}</p>}
      <div className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/40 dark:bg-black/5 p-3">
        <Yue2OptimizerFields value={optimValue} onChange={patch => setForm(previous => ({ ...previous, ...patch }))} />
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
      {error && <div className="mt-3 flex items-start gap-2 text-xs text-red-600 dark:text-red-400"><AlertTriangle size={14} className="mt-0.5 shrink-0" />{error}</div>}
      {job?.error && <div className="mt-2 text-xs text-red-600 dark:text-red-400">{job.error}</div>}
      <div className="mt-4 flex items-center gap-3 flex-wrap">
        <button type="button" onClick={() => void run()} disabled={active || preparing || starting || yue2RunAllActive || (form.resume?.trim() ? !form.dataset : (!prepare.legacyManifest || !prepare.tokenizer)) || (lyricTiming && !cursorReady)}
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
            target={0}
          />
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
          <select className={input} value={selectedCheckpoint} onChange={event => { setSelectedCheckpoint(event.target.value); setApplyNote(''); }}>
            {availableCheckpoints.map(checkpoint => <option key={checkpoint.dir} value={checkpoint.dir}>step {checkpoint.step}{checkpoint.dir === availableCheckpoints[0]?.dir ? ' (latest)' : ''}</option>)}
          </select>
          <button type="button" onClick={() => void applyCheckpoint()} disabled={activeBackendId !== 'yue2' || active || preparing || applyingCheckpoint || yue2RunAllActive || !selectedCheckpoint}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-emerald-500/50 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-40 flex items-center gap-1.5">
            {applyingCheckpoint ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
            {t('trainingStudio.yue2.method.useCheckpoint', 'Use for generation')}
          </button>
        </div>}
        {!runsError && availableCheckpoints.length === 0 && <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.noAuditionCheckpoint', 'No complete AR/NAR checkpoint is available yet.')}</p>}
        {applyNote && <p className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-400">{applyNote}</p>}
      </div>}
      {jointPreviews.length > 0 && <div className="mt-3 rounded-lg border border-sky-500/30 bg-sky-500/5 p-3">
        <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.previewStrip', 'Checkpoint previews')}</p>
        <div className="mt-2 flex flex-col gap-2">
          {jointPreviews.map(preview => <div key={preview.id} className="flex items-center gap-2 text-[11px] text-zinc-600 dark:text-zinc-400">
            <span className="w-20 shrink-0">step {preview.step} · {preview.kind}</span>
            <span className="flex-1">{preview.status === 'failed' ? preview.error || 'render failed' : preview.endReason === 'preview_limit' ? t('trainingStudio.yue2.method.previewCapped', 'Preview length reached') : preview.status}</span>
            {preview.audioUrl && preview.status === 'done' && <audio controls preload="none" src={preview.audioUrl} className="h-7 max-w-[240px]" />}
          </div>)}
        </div>
      </div>}
    </div>
  );
};

