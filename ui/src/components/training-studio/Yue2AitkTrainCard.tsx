import React, { useEffect, useState } from 'react';
import { AlertTriangle, Loader2, Play } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import {
  cancelJob,
  getJob,
  listJobs,
  startYue2JointTrain,
  type TrainingJobSummary,
  type Yue2JointTrainRequest,
} from '../../services/trainingApi';

const JOB_KEY = 'hs-yue2-aitk-job:';
const FORM_KEY = 'hs-yue2-aitk-form:';
const DEFAULT_FORM: Yue2JointTrainRequest = {
  trainingMethod: 'aitk', checkpoint: '', dataset: '', output: '',
  steps: 3000, saveEvery: 250, seed: 0, device: 'CUDA0',
};

function readStored<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  try {
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) as T : fallback;
  } catch { return fallback; }
}

function isJointJob(job: TrainingJobSummary, datasetId: string): boolean {
  return job.datasetId === datasetId && job.kind === 'yue2-joint-train';
}

export const Yue2AitkTrainCard: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const { t } = useTranslation();
  const [form, setForm] = useState<Yue2JointTrainRequest>(() =>
    readStored(`${FORM_KEY}${datasetId}`, DEFAULT_FORM));
  const [job, setJob] = useState<TrainingJobSummary | null>(null);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    setForm(readStored(`${FORM_KEY}${datasetId}`, DEFAULT_FORM));
    setJob(null);
    setError('');
    let cancelled = false;
    const storedId = readStored<string>(`${JOB_KEY}${datasetId}`, '');
    const restore = async () => {
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
    if (!job || !['queued', 'running'].includes(job.status)) return;
    const id = job.id;
    const timer = window.setInterval(() => {
      void getJob(id).then(next => {
        if (next.datasetId === datasetId) setJob(next);
      }).catch(err => setError(err instanceof Error ? err.message : String(err)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [datasetId, job?.id, job?.status]);

  const set = <K extends keyof Yue2JointTrainRequest>(key: K, value: Yue2JointTrainRequest[K]) =>
    setForm(previous => ({ ...previous, [key]: value }));
  const run = async () => {
    setStarting(true); setError('');
    try {
      const request = form.resume?.trim() ? { ...form, resume: form.resume.trim() } : form;
      const result = await startYue2JointTrain(datasetId, request);
      if (typeof window !== 'undefined') window.localStorage.setItem(`${JOB_KEY}${datasetId}`, JSON.stringify(result.jobId));
      setJob(await getJob(result.jobId));
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
  const active = job?.status === 'queued' || job?.status === 'running';
  const input = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-xs text-zinc-800 dark:text-zinc-200 outline-none focus:border-amber-500/50';
  const field = (label: string, key: keyof Yue2JointTrainRequest, type = 'text') => (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{label}</span>
      <input className={input} type={type} value={String(form[key] ?? '')} disabled={active || starting}
        onChange={event => set(key, type === 'number' ? Number(event.target.value) : event.target.value as Yue2JointTrainRequest[typeof key])} />
    </label>
  );
  const progress = job && job.total > 0 ? ` · ${job.done}/${job.total}` : '';
  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
      <h3 className="text-sm font-semibold text-amber-700 dark:text-amber-300">
        {t('trainingStudio.yue2.method.aitkTitle', 'AI Toolkit-compatible training is selected')}
      </h3>
      <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-2 leading-relaxed">
        {t('trainingStudio.yue2.method.aitkUnavailable', 'Provide the prepared manifest and raw ConvRot checkpoint paths to start the joint trainer. Legacy caches are rejected.')}
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
        {field(t('trainingStudio.yue2.method.checkpoint', 'Raw ConvRot checkpoint'), 'checkpoint')}
        {field(t('trainingStudio.yue2.method.dataset', 'Prepared AITK manifest'), 'dataset')}
        {field(t('trainingStudio.yue2.method.output', 'New output directory'), 'output')}
        {field(t('trainingStudio.yue2.method.resume', 'Resume record (optional)'), 'resume')}
        {field(t('trainingStudio.yue2.method.steps', 'Steps'), 'steps', 'number')}
        {field(t('trainingStudio.yue2.method.saveEvery', 'Save every'), 'saveEvery', 'number')}
        {field(t('trainingStudio.yue2.method.seed', 'Seed'), 'seed', 'number')}
        {field(t('trainingStudio.yue2.method.device', 'CUDA device'), 'device')}
      </div>
      <div className="mt-3 text-[11px] text-zinc-600 dark:text-zinc-400">
        <p className="font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.aitkNeeds', 'Before it can start, the dataset needs:')}</p>
        <ul className="list-disc pl-5 mt-1 space-y-0.5">
          <li>{t('trainingStudio.yue2.method.aitkManifest', 'A versioned full-song manifest with latents, semantic tokens and ABC provenance.')}</li>
          <li>{t('trainingStudio.yue2.method.aitkAssets', 'The pinned YuE2 base model and tokenizer/runtime assets.')}</li>
        </ul>
      </div>
      <p className="text-[11px] text-amber-700 dark:text-amber-300 mt-3">{t('trainingStudio.yue2.method.aitkNext', 'The server validates these paths and refuses missing or Legacy-formatted assets.')}</p>
      {error && <div className="mt-3 flex items-start gap-2 text-xs text-red-600 dark:text-red-400"><AlertTriangle size={14} className="mt-0.5 shrink-0" />{error}</div>}
      {job?.error && <div className="mt-2 text-xs text-red-600 dark:text-red-400">{job.error}</div>}
      <div className="mt-4 flex items-center gap-3 flex-wrap">
        <button type="button" onClick={() => void run()} disabled={active || starting || !form.checkpoint || !form.dataset || !form.output}
          className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2">
          {starting ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
          {active ? t('trainingStudio.yue2.method.running', 'Joint training is running') : t('trainingStudio.yue2.method.start', 'Start joint training')}
        </button>
        {active && <button type="button" onClick={() => void stop()} className="text-xs text-red-600 dark:text-red-400 hover:underline">{t('trainingStudio.yue2.method.cancel', 'Stop')}</button>}
        {job && <span className="text-[11px] text-zinc-600 dark:text-zinc-400">{job.status} · {job.phase || 'waiting'}{progress}</span>}
      </div>
      {job?.status === 'done' && <p className="mt-2 text-[11px] text-emerald-600 dark:text-emerald-400">{t('trainingStudio.yue2.method.checkpointWritten', 'Joint checkpoints are in the selected output directory.')}</p>}
    </div>
  );
};
