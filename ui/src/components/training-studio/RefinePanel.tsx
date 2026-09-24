// RefinePanel.tsx — YuE2 "Refine" phase: push a finished adapter's planner up
// the KL rungs and listen.
//
// The safe presets stop the planner at KL 1.2 because some albums decay late
// in the song past ~1.7 while others are clean to 1.8. Nothing programmatic
// tells those apart (five judges tried, 2026-09-24), so refinement is a
// listening ladder: resume the finished run with the planner live (the
// decoder rides along; it never decays), save a checkpoint every 0.1 KL up to
// a ceiling, render a preview per rung, pick the last good one.
import React, { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, Play, Sparkles } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore';
import {
  cancelJob, getJob, linkYue2JointCheckpointPreset, listYue2AitkRuns, listYue2JointPreviews,
  renderYue2JointPreviews, startYue2JointTrain,
  type TrainingJobSummary, type Yue2AitkRunRecord, type Yue2JointPreviewRecord, type Yue2JointTrainRequest,
} from '../../services/trainingApi';

const input = 'px-2 py-1.5 rounded-lg text-xs bg-white/70 dark:bg-black/20 border border-zinc-300/70 dark:border-white/10 text-zinc-800 dark:text-zinc-100';

export const RefinePanel: React.FC = () => {
  const { t } = useTranslation();
  const detail = useTrainingStore(s => s.detail);
  const datasetId = useTrainingStore(s => s.selectedDatasetId);
  const [runs, setRuns] = useState<Yue2AitkRunRecord[]>([]);
  const [source, setSource] = useState('');
  const [ceiling, setCeiling] = useState(2.0);
  const [rung, setRung] = useState(0.1);
  const [seconds, setSeconds] = useState(180);
  const [takes, setTakes] = useState(2);
  const [job, setJob] = useState<TrainingJobSummary | null>(null);
  const [ladderRun, setLadderRun] = useState('');
  const [previews, setPreviews] = useState<Yue2JointPreviewRecord[]>([]);
  const [rendering, setRendering] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [picked, setPicked] = useState('');

  const refreshRuns = async () => {
    if (!datasetId) return;
    try {
      const r = await listYue2AitkRuns(datasetId);
      setRuns(r.runs);
      if (r.activeJob?.kind === 'yue2-joint-train') setJob(r.activeJob);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };
  useEffect(() => { void refreshRuns(); }, [datasetId]);
  // Poll the refine job while it runs; refresh the ladder as rungs land.
  useEffect(() => {
    if (!job || (job.status !== 'running' && job.status !== 'queued')) return;
    const id = window.setInterval(() => {
      void getJob(job.id).then(next => { setJob(next); void refreshRuns(); }).catch(() => {});
    }, 5000);
    return () => window.clearInterval(id);
  }, [job?.id, job?.status]);
  useEffect(() => {
    if (!datasetId || !ladderRun) { setPreviews([]); return; }
    void listYue2JointPreviews(datasetId, ladderRun).then(r => setPreviews(r.previews)).catch(() => setPreviews([]));
  }, [datasetId, ladderRun, job?.status, rendering]);

  const finished = useMemo(() => runs.filter(r => !r.live && !r.resumeError && r.checkpoints.some(c => !!c.optimizerPath)), [runs]);
  const lastOf = (r: Yue2AitkRunRecord) => r.checkpoints.filter(c => !!c.optimizerPath).sort((a, b) => b.step - a.step)[0];
  const ladder = useMemo(() => {
    const r = runs.find(x => x.jobId === ladderRun);
    return r ? [...r.checkpoints].filter(c => c.arPath && c.narPath).sort((a, b) => a.step - b.step) : [];
  }, [runs, ladderRun]);
  const active = !!job && (job.status === 'running' || job.status === 'queued');

  const start = async () => {
    const run = finished.find(r => r.jobId === source);
    const last = run ? lastOf(run) : undefined;
    if (!datasetId || !run || !last) return;
    setBusy(true); setError('');
    try {
      const opts = run.options as Record<string, unknown>;
      const request = { trainingMethod: 'aitk', refinePlanner: true, resumeRunId: run.jobId, resumeStep: last.step,
        steps: last.step + 1000, stopMode: 'kl', targetKl: ceiling, klCheckpointEvery: rung,
        stopEngine: false, spikeFactor: 5, spikeStop: 3, spikeStopWindow: 20,
        lyricTiming: (opts.alignment as { enabled?: boolean } | undefined)?.enabled === true, autoPrepare: false, checkpoint: '', output: '',
        preview: { enabled: false, everySteps: 0, seconds: 90, seed: 424242, previewMaxFrames: 2250, baseline: false, control: false } } as unknown as Yue2JointTrainRequest;
      const result = await startYue2JointTrain(datasetId, request);
      setJob(await getJob(result.jobId));
      setLadderRun(result.jobId);
      await refreshRuns();
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  };
  const render = async (step: number) => {
    if (!datasetId || !ladderRun) return;
    setRendering(step); setError('');
    try { await renderYue2JointPreviews(datasetId, { run: ladderRun, step, seconds, takes }); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setRendering(null); }
  };
  const use = async (dir: string) => {
    if (!datasetId) return;
    setError('');
    try { await linkYue2JointCheckpointPreset(datasetId, dir); setPicked(dir); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };

  if (!datasetId) return <p className="text-sm text-zinc-500">{t('trainingStudio.refine.noDataset', 'Pick a dataset first.')}</p>;
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 p-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-800 dark:text-zinc-100"><Sparkles size={16} className="text-amber-500" />{t('trainingStudio.refine.title', 'Refine the planner')}{detail?.name ? ` · ${detail.name}` : ''}</div>
        <p className="mt-1 text-[12px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.refine.intro', 'The safe presets stop the planner at a conservative KL. Some artists take more. This continues a finished run with the planner live and the decoder along for the ride, saves a checkpoint at every KL rung up to the ceiling, and renders a preview per rung so you can hear where it starts to fall apart late in the song. Pick the last good rung.')}</p>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-4 gap-3">
          <label className="flex flex-col gap-1 md:col-span-2">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.source', 'Finished run')}</span>
            <select className={input} value={source} disabled={active || busy} onChange={e => setSource(e.target.value)}>
              <option value="">{t('trainingStudio.refine.pick', 'Pick a run')}</option>
              {finished.map(r => { const last = lastOf(r); return <option key={r.jobId} value={r.jobId}>{new Date(r.createdAt).toLocaleString()} · step {last.step}{last.kl !== undefined ? ` · KL ${last.kl.toFixed(2)}` : ''} · {r.status}</option>; })}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.ceiling', 'KL ceiling')}</span>
            <input className={input} type="number" step="0.1" min={0.5} max={4} value={ceiling} disabled={active || busy} onChange={e => setCeiling(Number(e.target.value) || 2)} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.rung', 'Rung (KL)')}</span>
            <input className={input} type="number" step="0.05" min={0.05} max={1} value={rung} disabled={active || busy} onChange={e => setRung(Number(e.target.value) || 0.1)} />
          </label>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <button type="button" onClick={() => void start()} disabled={!source || active || busy}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2">
            {busy || active ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {active ? t('trainingStudio.refine.running', 'Refining…') : t('trainingStudio.refine.start', 'Start refinement')}
          </button>
          {active && <button type="button" onClick={() => { if (job) void cancelJob(job.id).then(() => getJob(job.id)).then(setJob); }} className="text-xs text-red-600 dark:text-red-400 hover:underline">{t('trainingStudio.refine.stop', 'Stop')}</button>}
          {job && <span className="text-[11px] text-zinc-600 dark:text-zinc-400">{job.status} · {job.phase || 'waiting'}{job.total ? ` · step ${job.done} / ${job.total}` : ''}</span>}
        </div>
        {error && <div className="mt-2 text-xs text-red-600 dark:text-red-400">{error}</div>}
      </div>

      <div className="rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 min-w-[280px] flex-1">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.ladderRun', 'Ladder')}</span>
            <select className={input} value={ladderRun} onChange={e => setLadderRun(e.target.value)}>
              <option value="">{t('trainingStudio.refine.pickLadder', 'Pick a refinement run')}</option>
              {runs.filter(r => r.checkpoints.some(c => c.kl !== undefined)).map(r => <option key={r.jobId} value={r.jobId}>{new Date(r.createdAt).toLocaleString()} · {r.checkpoints.length} checkpoints · {r.status}{r.live ? ' (running)' : ''}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 w-28">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.seconds', 'Preview (s)')}</span>
            <input className={input} type="number" min={30} max={360} step={30} value={seconds} onChange={e => setSeconds(Math.max(30, Math.min(360, Number(e.target.value) || 180)))} />
          </label>
          <label className="flex flex-col gap-1 w-20">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.takes', 'Takes')}</span>
            <input className={input} type="number" min={1} max={4} value={takes} onChange={e => setTakes(Math.max(1, Math.min(4, Number(e.target.value) || 1)))} />
          </label>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.refine.ladderHint', 'Late-song decay shows after two minutes, so keep previews at 180 s or more. Renders are not deterministic: two takes per rung is the minimum to trust a rung.')}</p>
        {ladder.length > 0 && <table className="mt-3 w-full text-xs">
          <thead><tr className="text-[10px] uppercase tracking-wider text-zinc-500 text-left"><th className="py-1">Step</th><th>KL</th><th>Decoder recon</th><th>Previews</th><th></th></tr></thead>
          <tbody>
            {ladder.map(c => {
              const mine = previews.filter(p => p.step === c.step);
              return <tr key={c.step} className={`border-t border-zinc-200/70 dark:border-white/5 ${picked === c.dir ? 'bg-emerald-500/10' : ''}`}>
                <td className="py-1.5 font-mono">{c.step}</td>
                <td className="font-mono">{c.kl !== undefined ? c.kl.toFixed(2) : '—'}{c.frozen ? ' (frozen)' : ''}</td>
                <td className="font-mono">{c.recon !== undefined ? c.recon.toFixed(3) : '—'}</td>
                <td>
                  <div className="flex flex-wrap items-center gap-2">
                    {mine.map(p => p.audioUrl && p.status === 'done'
                      ? <audio key={p.id} controls preload="none" src={p.audioUrl} className="h-7 max-w-[220px]" />
                      : <span key={p.id} className="text-[11px] text-zinc-500">{p.status}{p.error ? `: ${p.error}` : ''}</span>)}
                    <button type="button" onClick={() => void render(c.step)} disabled={rendering !== null}
                      className="px-2 py-1 rounded-lg text-[11px] border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 disabled:opacity-40">
                      {rendering === c.step ? t('trainingStudio.refine.rendering', 'Rendering…') : t('trainingStudio.refine.render', 'Render')}
                    </button>
                  </div>
                </td>
                <td className="text-right">
                  <button type="button" onClick={() => void use(c.dir)}
                    className={`px-2 py-1 rounded-lg text-[11px] font-semibold border ${picked === c.dir ? 'border-emerald-500 text-emerald-700 dark:text-emerald-300' : 'border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10'}`}>
                    {picked === c.dir ? t('trainingStudio.refine.picked', 'In use') : t('trainingStudio.refine.use', 'Use this rung')}
                  </button>
                </td>
              </tr>;
            })}
          </tbody>
        </table>}
      </div>
    </div>
  );
};

export default RefinePanel;
