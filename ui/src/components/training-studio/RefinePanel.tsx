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
  getYue2CleanupPlan, runYue2Cleanup, type Yue2CleanupPlan, type Yue2CleanupChoice,
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
  // Further training for the decoder from the picked rung (planner frozen):
  // to a reconstruction target, a step budget, or the knee, whichever first.
  const [narFurther, setNarFurther] = useState(true);
  const [narTarget, setNarTarget] = useState<number | ''>(0.25);
  const [narBudget, setNarBudget] = useState(500);
  const [narJob, setNarJob] = useState<{ jobId: string; step: number } | null>(null);
  const [job, setJob] = useState<TrainingJobSummary | null>(null);
  const wantedLadder = useTrainingStore(s => s.refineLadderRun);
  const setWantedLadder = useTrainingStore(s => s.setRefineLadderRun);
  const [ladderRun, setLadderRun] = useState('');
  // The review page hands over a ladder to show; take it once.
  useEffect(() => { if (wantedLadder) { setLadderRun(wantedLadder); setWantedLadder(''); } }, [wantedLadder]);
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
  // Short enough to survive the picker's middle-ellipsis (about 40 chars).
  const runLabel = (r: Yue2AitkRunRecord) => {
    const last = lastOf(r); const d = new Date(r.createdAt);
    const when = `${d.toLocaleDateString(undefined, { day: '2-digit', month: '2-digit' })} ${d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`;
    return `${when} · s${last?.step ?? '?'}${last?.kl !== undefined ? ` KL${last.kl.toFixed(2)}` : ''} · ${r.checkpoints.length}ck · ${r.live ? 'running' : r.status}`;
  };
  // Cleanup modal after a rung is chosen: what else can go, with sizes.
  const [cleanup, setCleanup] = useState<{ step: number; plan: Yue2CleanupPlan } | null>(null);
  const [choice, setChoice] = useState<Yue2CleanupChoice>({ caches: true, otherCheckpoints: true, otherRuns: true, resume: true, otherPreviews: true });
  const [cleaning, setCleaning] = useState(false);
  const [cleanupNote, setCleanupNote] = useState('');
  const mib = (b: number) => b >= 1073741824 ? `${(b / 1073741824).toFixed(2)} GiB` : `${(b / 1048576).toFixed(0)} MiB`;
  const finishPick = async (run: string, dir: string, step: number) => {
    await linkYue2JointCheckpointPreset(datasetId!, dir); setPicked(dir);
    const plan = await getYue2CleanupPlan(datasetId!, run, step);
    setLadderRun(run);
    setCleanup({ step, plan });
  };
  const use = async (dir: string, step: number) => {
    if (!datasetId || !ladderRun) return;
    setError(''); setCleanupNote('');
    try {
      if (!narFurther) { await finishPick(ladderRun, dir, step); return; }
      // Decoder on from this rung, planner frozen; the result becomes the adapter.
      const result = await startYue2JointTrain(datasetId, { trainingMethod: 'aitk', refine: true, resumeRunId: ladderRun, resumeStep: step,
        steps: step + narBudget, saveEvery: 10, stopMode: 'kl', narExtraSteps: step + narBudget, freezePlannerNow: true,
        reconStop: 0.005, reconStopWindow: 5, ...(narTarget !== '' ? { reconTarget: narTarget } : {}), stopEngine: false,
        lyricTiming: true, alignmentEnabled: true, autoPrepare: false, checkpoint: '', output: '',
        preview: { enabled: false, everySteps: 0, seconds: 90, seed: 424242, previewMaxFrames: 2250, baseline: false, control: false } } as unknown as Yue2JointTrainRequest);
      setNarJob({ jobId: result.jobId, step });
      setJob(await getJob(result.jobId));
      setCleanupNote(t('trainingStudio.refine.narStarted', 'Decoder training on from step {{step}}; the adapter is linked when it stops.', { step }));
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };
  // When the decoder follow-up ends, its last checkpoint is the adapter.
  useEffect(() => {
    if (!narJob || !job || job.id !== narJob.jobId || job.status !== 'done' || !datasetId) return;
    const pending = narJob; setNarJob(null);
    void (async () => {
      try {
        const r = await listYue2AitkRuns(datasetId);
        const run = r.runs.find(x => x.jobId === pending.jobId);
        const last = run?.checkpoints.filter(c => c.arPath && c.narPath).sort((a, b) => b.step - a.step)[0];
        if (!run || !last) { setError(t('trainingStudio.refine.narNoCheckpoint', 'The decoder run left no complete checkpoint.')); return; }
        setRuns(r.runs);
        await finishPick(run.jobId, last.dir, last.step);
      } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    })();
  }, [job?.id, job?.status]);
  const doCleanup = async () => {
    if (!datasetId || !ladderRun || !cleanup) return;
    setCleaning(true); setError('');
    try {
      const r = await runYue2Cleanup(datasetId, { run: ladderRun, step: cleanup.step, ...choice });
      setCleanupNote(t('trainingStudio.refine.cleanupDone', 'Removed {{what}}; about {{size}} freed.', { what: r.done.join(', ') || 'nothing', size: mib(r.freedBytes) }));
      setCleanup(null);
      await refreshRuns();
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setCleaning(false); }
  };

  if (!datasetId) return <p className="text-sm text-zinc-500">{t('trainingStudio.refine.noDataset', 'Pick a dataset first.')}</p>;
  const items: Array<{ key: keyof Yue2CleanupChoice; label: string; item?: { count: number; bytes: number; detail?: string[] } }> = cleanup ? [
    { key: 'caches', label: t('trainingStudio.refine.cleanCaches', 'Prepared data and caches for this dataset (latents, codes, lead sheets, alignment, stems, MM3 and ACE caches)'), item: cleanup.plan.caches },
    { key: 'otherCheckpoints', label: t('trainingStudio.refine.cleanCheckpoints', 'The other checkpoints in this run'), item: cleanup.plan.otherCheckpoints },
    { key: 'otherRuns', label: t('trainingStudio.refine.cleanRuns', 'All other joint runs for this dataset'), item: cleanup.plan.otherRuns },
    { key: 'resume', label: t('trainingStudio.refine.cleanResume', 'This rung\'s resume file (the adapter files stay)'), item: cleanup.plan.resume },
    { key: 'otherPreviews', label: t('trainingStudio.refine.cleanPreviews', 'Previews from the other rungs'), item: cleanup.plan.otherPreviews },
  ] : [];
  const totalBytes = items.reduce((s, i) => s + (choice[i.key] && i.item ? i.item.bytes : 0), 0);
  return (
    <div className="flex flex-col gap-4">
      {cleanup && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => !cleaning && setCleanup(null)}>
        <div className="w-full max-w-xl rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white dark:bg-zinc-900 p-5 shadow-xl" onClick={e => e.stopPropagation()}>
          <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">{t('trainingStudio.refine.cleanupTitle', 'Step {{step}} is now the adapter. Clean up around it?', { step: cleanup.step })}</div>
          <p className="mt-1 text-[12px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.refine.cleanupIntro', 'Everything below is app-generated and can be rebuilt. Source audio, sidecars, captions, labels and this rung\'s adapter files are never touched. Your rung scores are kept.')}</p>
          <div className="mt-3 flex flex-col gap-2">
            {items.map(i => <label key={i.key} className="flex items-start gap-3 text-xs text-zinc-700 dark:text-zinc-300">
              <Toggle id={`cleanup-${i.key}`} checked={!!choice[i.key] && !!i.item?.count} onChange={v => setChoice(prev => ({ ...prev, [i.key]: v }))} />
              <span className={`flex-1 ${!i.item?.count ? 'opacity-50' : ''}`}>{i.label}<span className="ml-2 text-zinc-500 tabular-nums">{i.item ? `${i.item.count} · ${mib(i.item.bytes)}` : ''}</span>
                {i.item?.detail?.length ? <span className="block text-[10px] text-zinc-500 truncate">{i.item.detail.join(', ')}</span> : null}</span>
            </label>)}
          </div>
          <div className="mt-4 flex items-center justify-between gap-3">
            <span className="text-[11px] text-zinc-500 tabular-nums">{t('trainingStudio.refine.cleanupTotal', 'About {{size}} to free', { size: mib(totalBytes) })}</span>
            <div className="flex gap-2">
              <button type="button" disabled={cleaning} onClick={() => setCleanup(null)} className="px-3 py-1.5 rounded-lg text-xs border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10">{t('trainingStudio.refine.cleanupSkip', 'Keep everything')}</button>
              <button type="button" disabled={cleaning || totalBytes === 0} onClick={() => void doCleanup()} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-500 text-white hover:bg-red-400 disabled:opacity-40">{cleaning ? t('trainingStudio.refine.cleaning', 'Cleaning…') : t('trainingStudio.refine.cleanupGo', 'Delete selected')}</button>
            </div>
          </div>
        </div>
      </div>}
      {cleanupNote && <div className="text-[12px] text-emerald-700 dark:text-emerald-300">{cleanupNote}</div>}
      <div className="rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 p-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-800 dark:text-zinc-100"><Sparkles size={16} className="text-amber-500" />{t('trainingStudio.refine.title', 'Refine the planner')}{detail?.name ? ` · ${detail.name}` : ''}</div>
        <p className="mt-1 text-[12px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.refine.intro', 'The safe presets stop the planner at a conservative KL. Some artists take more. This continues a finished run with the planner live and the decoder along for the ride, saves a checkpoint at every KL rung up to the ceiling, and renders a preview per rung so you can hear where it starts to fall apart late in the song. Pick the last good rung.')}</p>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-4 gap-3">
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
          <div className={`flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300 self-center ${active || busy ? 'opacity-50 pointer-events-none' : ''}`}
            title={t('trainingStudio.refine.narFurtherHint', 'When you pick a rung, the decoder trains on from it with the planner frozen until the reconstruction target, the step budget, or the knee. The result becomes the adapter.')}>
            <Toggle id="refine-nar-further" checked={narFurther} onChange={setNarFurther} />
            <span>{t('trainingStudio.refine.narFurther', 'Further training for NAR')}</span>
          </div>
          <label className="flex flex-col gap-1 w-28">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.narTarget', 'Recon target')}</span>
            <input className={input} type="number" step="0.005" min={0} max={10} value={narTarget} placeholder={t('trainingStudio.refine.narKnee', 'knee')} disabled={active || busy || !narFurther}
              onChange={e => setNarTarget(e.target.value === '' ? '' : Math.max(0, Number(e.target.value) || 0))} />
          </label>
          <label className="flex flex-col gap-1 w-28">
            <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.narBudget', 'NAR max steps')}</span>
            <input className={input} type="number" min={10} step={10} value={narBudget} disabled={active || busy || !narFurther} onChange={e => setNarBudget(Math.max(10, Math.round(Number(e.target.value) || 0)))} />
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
                <button type="button" onClick={() => void use(c.dir, c.step)}
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
