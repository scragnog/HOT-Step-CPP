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
import { Loader2, Play, Sparkles, Trash2 } from 'lucide-react';
import { ModelSelect } from '../global-bar/ModelSelect';
import { Toggle } from '../settings/SettingsPrimitives';
import { useTrainingStore } from '../../stores/trainingStore';
import { Yue2JointRunChart } from './Yue2JointRunChart';
import { PreviewPlayer } from './PreviewPlayer';
import {
  cancelJob, getJob, linkYue2JointCheckpointPreset, listYue2AitkRuns, listYue2JointPreviews,
  renderYue2JointPreviews, startYue2JointTrain, listYue2RungScores, scoreYue2Rung, yue2RungScoresExportUrl, deleteYue2AitkRun,
  type TrainingJobSummary, type Yue2AitkRunRecord, type Yue2JointPreviewRecord, type Yue2JointTrainRequest, type Yue2RungScore,
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
  const [autoPreview, setAutoPreview] = useState(true);
  const [parallel, setParallel] = useState(true);
  const [lrScale, setLrScale] = useState(0.1);
  const [job, setJob] = useState<TrainingJobSummary | null>(null);
  const [ladderRun, setLadderRun] = useState('');
  const [previews, setPreviews] = useState<Yue2JointPreviewRecord[]>([]);
  const [rendering, setRendering] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [picked, setPicked] = useState('');
  // Scores per rung (by step) for the ladder run; notes are saved on blur.
  const [scores, setScores] = useState<Record<number, Yue2RungScore>>({});
  const [noteDraft, setNoteDraft] = useState<Record<number, string>>({});
  const loadScores = () => { if (datasetId && ladderRun) void listYue2RungScores(datasetId, ladderRun).then(r => setScores(Object.fromEntries(r.scores.map(s => [s.step, s])))).catch(() => {}); };
  useEffect(() => { setScores({}); setNoteDraft({}); loadScores(); }, [datasetId, ladderRun]);
  const score = async (step: number, patch: { likeness?: number | null; corruption?: number | null; notes?: string }) => {
    if (!datasetId || !ladderRun) return;
    try { const r = await scoreYue2Rung(datasetId, { refineRun: ladderRun, step, ...patch }); setScores(prev => ({ ...prev, [step]: r.score })); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };

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
      // Rung previews land while the job runs: keep the ladder's players current.
      if (datasetId && ladderRun) void listYue2JointPreviews(datasetId, ladderRun).then(r => setPreviews(r.previews)).catch(() => {});
    }, 5000);
    return () => window.clearInterval(id);
  }, [job?.id, job?.status, datasetId, ladderRun]);
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
        steps: last.step + 1000, stopMode: 'kl', targetKl: ceiling, klCheckpointEvery: rung, refineLrScale: lrScale,
        stopEngine: false, spikeFactor: 5, spikeStop: 3, spikeStopWindow: 20,
        lyricTiming: (opts.alignment as { enabled?: boolean } | undefined)?.enabled === true, autoPrepare: false, checkpoint: '', output: '',
        // Rung previews: the engine pauses after each rung checkpoint and the
        // server renders `takes` previews there before resuming.
        preview: { enabled: autoPreview, everySteps: 0, takes, seconds, seed: 424242, previewMaxFrames: seconds * 25, baseline: false, control: false, parallel } } as unknown as Yue2JointTrainRequest;
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
  const remove = async (jobId: string) => {
    const run = runs.find(r => r.jobId === jobId);
    if (!datasetId || !run) return;
    if (!window.confirm(t('trainingStudio.refine.deleteConfirm', 'Delete this run and its {{n}} checkpoint(s) from disk? Scores you entered are kept.', { n: run.checkpoints.length }))) return;
    setError('');
    try {
      await deleteYue2AitkRun(datasetId, jobId);
      if (source === jobId) setSource('');
      if (ladderRun === jobId) setLadderRun('');
      await refreshRuns();
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };
  const runLabel = (r: Yue2AitkRunRecord) => { const last = lastOf(r); return `${new Date(r.createdAt).toLocaleString()} · step ${last?.step ?? '—'}${last?.kl !== undefined ? ` · KL ${last.kl.toFixed(2)}` : ''} · ${r.checkpoints.length} ckpt · ${r.status}${r.live ? ' (running)' : ''}`; };
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
        <div className="mt-3 grid grid-cols-1 md:grid-cols-5 gap-3">
          <div className="flex flex-col gap-1 md:col-span-2">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.source', 'Finished run')}</span>
            <div className="flex items-center gap-2">
              <div className="flex-1 min-w-0"><ModelSelect id="refine-source" value={source} onChange={setSource} options={finished.map(r => r.jobId)} formatLabel={id => { const r = finished.find(x => x.jobId === id); return r ? runLabel(r) : id; }} formatOf={null} filterable={false} disabled={active || busy} placeholder={t('trainingStudio.refine.pick', 'Pick a run')} /></div>
              <button type="button" onClick={() => void remove(source)} disabled={!source || active || busy} title={t('trainingStudio.refine.deleteRun', 'Delete run')}
                className="p-2 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:text-red-500 hover:border-red-500/40 disabled:opacity-40"><Trash2 size={14} /></button>
            </div>
          </div>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.ceiling', 'KL ceiling')}</span>
            <input className={input} type="number" step="0.1" min={0.5} max={4} value={ceiling} disabled={active || busy} onChange={e => setCeiling(Number(e.target.value) || 2)} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.rung', 'Rung (KL)')}</span>
            <input className={input} type="number" step="0.05" min={0.05} max={1} value={rung} disabled={active || busy} onChange={e => setRung(Number(e.target.value) || 0.1)} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.lrScale', 'Learning rate (× run)')}</span>
            <input className={input} type="number" step="0.05" min={0.05} max={1} value={lrScale} disabled={active || busy} onChange={e => setLrScale(Math.max(0.05, Math.min(1, Number(e.target.value) || 0.1)))} title={t('trainingStudio.refine.lrHint', 'Ramps up over the first 30 steps and halves itself whenever the KL jumps more than one rung between checkpoints.')} />
          </label>
        </div>
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <div className={`flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300 self-center ${active || busy ? 'opacity-50 pointer-events-none' : ''}`}>
            <Toggle id="refine-auto-preview" checked={autoPreview} onChange={setAutoPreview} />
            <span>{t('trainingStudio.refine.autoPreview', 'Render previews at each rung')}</span>
          </div>
          <div className={`flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300 self-center ${active || busy || !autoPreview ? 'opacity-50 pointer-events-none' : ''}`} title={t('trainingStudio.refine.parallelHint', 'Renders while training continues instead of pausing it. Needs about 22 GB of VRAM for both; renders run at about half speed.')}>
            <Toggle id="refine-parallel" checked={parallel} onChange={setParallel} />
            <span>{t('trainingStudio.refine.parallel', 'In parallel with training')}</span>
          </div>
          <label className="flex flex-col gap-1 w-24">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.takes', 'Tracks')}</span>
            <input className={input} type="number" min={1} max={4} value={takes} disabled={active || busy} onChange={e => setTakes(Math.max(1, Math.min(4, Number(e.target.value) || 1)))} />
          </label>
          <label className="flex flex-col gap-1 w-28">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.seconds', 'Length (s)')}</span>
            <input className={input} type="number" min={30} max={360} step={30} value={seconds} disabled={active || busy} onChange={e => setSeconds(Math.max(30, Math.min(360, Number(e.target.value) || 180)))} />
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
        {job && <div className="mt-3"><Yue2JointRunChart job={job} totalSteps={job.total || 0} klTarget={ceiling} /></div>}
      </div>

      <div className="rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col gap-1 min-w-[280px] flex-1">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.ladderRun', 'Ladder')}</span>
            <div className="flex items-center gap-2">
              <div className="flex-1 min-w-0"><ModelSelect id="refine-ladder" value={ladderRun} onChange={setLadderRun} options={runs.filter(r => r.checkpoints.some(c => c.kl !== undefined)).map(r => r.jobId)} formatLabel={id => { const r = runs.find(x => x.jobId === id); return r ? runLabel(r) : id; }} formatOf={null} filterable={false} placeholder={t('trainingStudio.refine.pickLadder', 'Pick a refinement run')} /></div>
              <button type="button" onClick={() => void remove(ladderRun)} disabled={!ladderRun || !!runs.find(r => r.jobId === ladderRun)?.live} title={t('trainingStudio.refine.deleteRun', 'Delete run')}
                className="p-2 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:text-red-500 hover:border-red-500/40 disabled:opacity-40"><Trash2 size={14} /></button>
            </div>
          </div>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.refine.ladderHint', 'Late-song decay shows after two minutes, so keep previews at 180 s or more. Renders are not deterministic: two tracks per rung is the minimum to trust a rung. Render adds more tracks to a rung with the count and length above.')}
          {' '}<a className="underline hover:text-zinc-700 dark:hover:text-zinc-300" href={yue2RungScoresExportUrl('csv')}>{t('trainingStudio.refine.exportCsv', 'Export all scores (CSV)')}</a>
          {' · '}<a className="underline hover:text-zinc-700 dark:hover:text-zinc-300" href={yue2RungScoresExportUrl('json')} target="_blank" rel="noreferrer">JSON</a></p>
        {ladder.length > 0 && <div className="mt-3 flex flex-col gap-3">
          {ladder.map(c => {
            const mine = previews.filter(p => p.step === c.step).sort((a, b) => a.seed - b.seed);
            return <div key={c.step} className={`rounded-lg border p-3 ${picked === c.dir ? 'border-emerald-500/60 bg-emerald-500/5' : c.rung ? 'border-amber-500/40' : 'border-zinc-300/70 dark:border-white/10'}`}>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                <span className="font-semibold text-zinc-800 dark:text-zinc-100">{t('trainingStudio.refine.rungStep', 'Step {{step}}', { step: c.step })}</span>
                {c.rung && <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[10px] font-semibold">{t('trainingStudio.refine.rungBadge', 'rung')}</span>}
                {!c.rung && <span className="text-[10px] text-zinc-500">{t('trainingStudio.refine.routineSave', 'routine save')}</span>}
                <span className="font-mono text-zinc-600 dark:text-zinc-300">KL {c.kl !== undefined ? c.kl.toFixed(2) : '—'}{c.frozen ? ' (frozen)' : ''}</span>
                <span className="font-mono text-zinc-600 dark:text-zinc-300">recon {c.recon !== undefined ? c.recon.toFixed(3) : '—'}</span>
                <span className="flex-1" />
                <button type="button" onClick={() => void render(c.step)} disabled={rendering !== null}
                  className="px-2 py-1 rounded-lg text-[11px] border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 disabled:opacity-40">
                  {rendering === c.step ? t('trainingStudio.refine.rendering', 'Rendering…') : t('trainingStudio.refine.render', 'Render more')}
                </button>
                <button type="button" onClick={() => void use(c.dir)}
                  className={`px-2 py-1 rounded-lg text-[11px] font-semibold border ${picked === c.dir ? 'border-emerald-500 text-emerald-700 dark:text-emerald-300' : 'border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10'}`}>
                  {picked === c.dir ? t('trainingStudio.refine.picked', 'In use') : t('trainingStudio.refine.use', 'Use this rung')}
                </button>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px]">
                {(['likeness', 'corruption'] as const).map(key => <div key={key} className="flex items-center gap-1">
                  <span className="text-zinc-500 w-16">{key === 'likeness' ? t('trainingStudio.refine.scoreLikeness', 'Likeness') : t('trainingStudio.refine.scoreCorruption', 'Corruption')}</span>
                  {[1, 2, 3, 4, 5].map(n => <button key={n} type="button" onClick={() => void score(c.step, { [key]: scores[c.step]?.[key] === n ? null : n })}
                    className={`w-6 h-6 rounded border text-[11px] font-semibold ${scores[c.step]?.[key] === n ? (key === 'corruption' ? 'bg-red-500/20 border-red-500 text-red-700 dark:text-red-300' : 'bg-amber-500/20 border-amber-500 text-amber-700 dark:text-amber-300') : 'border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10'}`}>{n}</button>)}
                </div>)}
                <input className={`${input} flex-1 min-w-[240px]`} placeholder={t('trainingStudio.refine.notes', 'Notes: what you heard, and where (m:ss)')}
                  value={noteDraft[c.step] ?? scores[c.step]?.notes ?? ''} onChange={e => setNoteDraft(prev => ({ ...prev, [c.step]: e.target.value }))}
                  onBlur={e => { if (e.target.value !== (scores[c.step]?.notes ?? '')) void score(c.step, { notes: e.target.value }); }} />
              </div>
              {mine.length > 0 && <div className="mt-2 flex flex-col gap-2">
                {mine.map((p, i) => p.audioUrl && p.status === 'done'
                  ? <PreviewPlayer key={p.id} src={p.audioUrl} label={t('trainingStudio.refine.take', 'Take {{n}}', { n: i + 1 })} sublabel={`${p.seconds} s · seed ${p.seed}${p.endReason && p.endReason !== 'completed' ? ` · ${p.endReason}` : ''}${p.score?.verdict ? ` · plan ${p.score.verdict}` : ''}`} />
                  : <div key={p.id} className="text-[11px] text-zinc-500">{t('trainingStudio.refine.take', 'Take {{n}}', { n: i + 1 })}: {p.status}{p.error ? ` — ${p.error}` : ''}</div>)}
              </div>}
            </div>;
          })}
        </div>}
      </div>
    </div>
  );
};

export default RefinePanel;
