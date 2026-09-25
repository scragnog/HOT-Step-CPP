// RefinePanel.tsx — YuE2 "Refine" phase: push a finished adapter's planner up
// the KL rungs and listen.
//
// The safe presets stop the planner at KL 1.2 because some albums decay late
// in the song past ~1.7 while others are clean to 1.8. Nothing programmatic
// tells those apart (five judges tried, 2026-09-24), so refinement is a
// listening ladder: resume the finished run with the planner live (the
// decoder rides along; it never decays), save a checkpoint every 0.1 KL up to
// a ceiling, render a preview per rung, pick the last good one.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronUp, Loader2, Play, Sparkles, Trash2 } from 'lucide-react';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
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

// A rung's overall score for the scoreboard: base is likeness and inverted
// corruption averaged onto the same 1..5 scale as ((6 − corruption) mirrors
// likeness's direction), then a soft penalty for replan load — the app
// auto-replans on its own, so a high count is a smell, not a verdict — capped
// at 1 point so it can never flip the ranking on its own.
export function rungOverall(args: {
  likeness: number | null | undefined;
  corruption: number | null | undefined;
  plannerReplans: number;
  composerReplans: number;
  takes: number;
}): { overall: number; replansPerTake: number } | null {
  const { likeness, corruption, plannerReplans, composerReplans, takes } = args;
  if (typeof likeness !== 'number' || typeof corruption !== 'number') return null;
  const base = (likeness + (6 - corruption)) / 2;
  const replansPerTake = (plannerReplans + composerReplans) / Math.max(1, takes);
  const penalty = Math.min(1, 0.25 * replansPerTake);
  return { overall: Math.round((base - penalty) * 100) / 100, replansPerTake };
}

export const RefinePanel: React.FC = () => {
  const { t } = useTranslation();
  const detail = useTrainingStore(s => s.detail);
  const datasetId = useTrainingStore(s => s.selectedDatasetId);
  const datasetName = useTrainingStore(s => s.datasets.find(d => d.id === s.selectedDatasetId)?.name);
  const [runs, setRuns] = useState<Yue2AitkRunRecord[]>([]);
  const [source, setSource] = useState('');
  const [ceiling, setCeiling] = useState(2.0);
  const [rung, setRung] = useState(0.1);
  const [seconds, setSeconds] = useState(300);
  const [takes, setTakes] = useState(2);
  const [autoPreview, setAutoPreview] = useState(true);
  const [parallel, setParallel] = useState(true);
  // Draft previews: decoder at 12 ODE steps, velocity cache OFF (Rob's
  // tests 2026-09-23: a raised cache ratio blunts artist likeness). The
  // planner stage, which is what a preview judges, is untouched.
  const [draft, setDraft] = useState(true);
  const draftOpts = draft ? { odeSteps: 12, narCacheRatio: 0 } : {};
  const [lrScale, setLrScale] = useState(0.1);
  // Further training for the decoder from the picked rung (planner frozen):
  // to a reconstruction target, a step budget, or the knee, whichever first.
  const [narFurther, setNarFurther] = useState(true);
  const [narTarget, setNarTarget] = useState<number | ''>(0.25);
  const [narBudget, setNarBudget] = useState(500);
  const [narKeepDelta, setNarKeepDelta] = useState(0.003);
  // Persisted in the store (not local state) so the running decoder follow-up
  // keeps being tracked, and the running ladder keeps being shown, across a
  // navigation away from this tab and back.
  const narJob = useTrainingStore(s => s.refineNarJob);
  const setNarJob = useTrainingStore(s => s.setRefineNarJob);
  const [job, setJob] = useState<TrainingJobSummary | null>(null);
  // The review page hands over a ladder by calling setRefineLadderRun; this IS
  // the tab's selection, so it survives navigating away and back.
  const ladderRun = useTrainingStore(s => s.refineLadderRun);
  const setLadderRun = useTrainingStore(s => s.setRefineLadderRun);
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

  // The last live run this panel auto-switched the ladder to — so a server-
  // started follow-up (planner refinement after a primary run, or a decoder
  // follow-up) steals the selection once, even while some other run is
  // currently picked, without fighting the user's own picks afterwards.
  const lastAutoSelectedLive = useRef('');
  const refreshRuns = async () => {
    if (!datasetId) return;
    try {
      const r = await listYue2AitkRuns(datasetId);
      setRuns(r.runs);
      if (r.activeJob?.kind === 'yue2-joint-train') setJob(r.activeJob);
      const live = r.runs.find(x => x.live);
      if (live && live.jobId !== lastAutoSelectedLive.current) {
        lastAutoSelectedLive.current = live.jobId;
        setLadderRun(live.jobId);
      } else if (!ladderRun || !r.runs.some(x => x.jobId === ladderRun)) {
        // Keep the ladder selection live: default it to the running ladder,
        // or else the newest one with KL checkpoints, whenever it's empty or
        // names a run that's gone (deleted, or never set on this dataset
        // before).
        const newestKl = [...r.runs].filter(x => x.checkpoints.some(c => c.kl !== undefined)).sort((a, b) => b.createdAt - a.createdAt)[0];
        const next = live?.jobId ?? newestKl?.jobId;
        if (next) setLadderRun(next);
        if (live && next === live.jobId) lastAutoSelectedLive.current = live.jobId;
      }
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  };
  useEffect(() => { void refreshRuns(); }, [datasetId]);
  // A decoder follow-up may have finished while this panel was unmounted; pick
  // its job back up so the "when it ends" effect below still fires.
  useEffect(() => {
    if (!narJob) return;
    if (job && job.id === narJob.jobId) return;
    void getJob(narJob.jobId).then(setJob).catch(() => {});
  }, [narJob?.jobId]);
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
        preview: { enabled: autoPreview, everySteps: 0, takes, seconds, seed: 424242, previewMaxFrames: seconds * 25, baseline: false, control: false, parallel, ...draftOpts } } as unknown as Yue2JointTrainRequest;
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
    try { await renderYue2JointPreviews(datasetId, { run: ladderRun, step, seconds, takes, ...draftOpts }); }
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
  const [scoreboardCollapsed, setScoreboardCollapsed] = useState(false);
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
      // A decoder-only run (its own "further training" follow-up) never gets
      // another decoder follow-up chained onto it — that rung finishes here.
      const run = runs.find(r => r.jobId === ladderRun);
      const isDecoderOnly = (run?.options as Record<string, unknown> | undefined)?.freezePlannerNow === true;
      if (!narFurther || isDecoderOnly) { await finishPick(ladderRun, dir, step); return; }
      // Decoder on from this rung, planner frozen; the result becomes the adapter.
      const result = await startYue2JointTrain(datasetId, { trainingMethod: 'aitk', refine: true, resumeRunId: ladderRun, resumeStep: step,
        steps: step + narBudget, saveEvery: 10, stopMode: 'kl', narExtraSteps: step + narBudget, freezePlannerNow: true,
        reconStop: 0.005, reconStopWindow: 5, reconKeepDelta: narKeepDelta, ...(narTarget !== '' ? { reconTarget: narTarget } : {}), stopEngine: false,
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
  const noCleanupChoice: Yue2CleanupChoice = { caches: false, otherCheckpoints: false, otherRuns: false, resume: false, otherPreviews: false };
  const doCleanup = async (over?: Yue2CleanupChoice) => {
    if (!datasetId || !ladderRun || !cleanup) return;
    setCleaning(true); setError('');
    try {
      const r = await runYue2Cleanup(datasetId, { run: ladderRun, step: cleanup.step, ...(over ?? choice) });
      const moved = r.movedTo ? ` ${t('trainingStudio.refine.cleanupMoved', 'Moved to {{dir}}.', { dir: r.movedTo })}` : '';
      setCleanupNote(t('trainingStudio.refine.cleanupDone', 'Removed {{what}}; about {{size}} freed.', { what: r.done.join(', ') || 'nothing', size: mib(r.freedBytes) }) + moved);
      setCleanup(null);
      await refreshRuns();
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setCleaning(false); }
  };

  if (!datasetId) return <p className="text-sm text-zinc-500">{t('trainingStudio.refine.noDataset', 'Pick a dataset first.')}</p>;
  // Once a run has stopped and none of its rungs is picked yet, say plainly
  // that the ladder isn't done until "Use this rung" is pressed.
  const ladderRunRec = runs.find(r => r.jobId === ladderRun);
  const ladderIsDecoderOnly = (ladderRunRec?.options as Record<string, unknown> | undefined)?.freezePlannerNow === true;
  const ladderIsRefinement = (ladderRunRec?.options as Record<string, unknown> | undefined)?.refinePlanner === true;
  // A primary run finishing is not itself a thing to "pick a rung" for — that
  // notice is only for a planner refinement or a decoder-only follow-up.
  const finishedUnpickedNotice = ladderRunRec && !ladderRunRec.live && ladderRunRec.status === 'done' && !ladder.some(c => c.dir === picked) && (ladderIsRefinement || ladderIsDecoderOnly)
    ? (ladderIsDecoderOnly
      ? t('trainingStudio.refine.noticeDecoderDone', 'Decoder training finished. Press Use this rung on its last checkpoint to link the adapter and clean up.')
      : t('trainingStudio.refine.noticePlannerDone', 'Refinement finished. Listen to the rungs, then press Use this rung on the one you choose. With Further training for NAR on, the decoder then trains on from that rung and the adapter is linked when it stops; off, the rung is linked as it is and you can clean up.'))
    : '';
  // Per-rung facts shared by the ladder cards and the scoreboard: this
  // rung's previews, its replan load, and its overall score (null unless
  // it's been scored on both axes).
  const rungStats = (step: number) => {
    const mine = previews.filter(p => p.step === step).sort((a, b) => a.seed - b.seed);
    const doneTakes = mine.filter(p => p.status === 'done');
    const plannerReplans = doneTakes.reduce((sum, p) => sum + (p.plan ? p.plan.attempts.length - 1 : 0), 0);
    const composerReplans = doneTakes.reduce((sum, p) => sum + (typeof p.composerReplans === 'number' ? p.composerReplans : 0), 0);
    const hasReplanData = doneTakes.some(p => p.plan || typeof p.composerReplans === 'number');
    // Legibility flags on the rendered plans: a plan the verdict passed that
    // still loops a riff, sings on two pitches or skips lyric sections. A
    // count and its reasons, not part of the score, until ear scores have
    // been checked against them.
    const flaggedTakes = doneTakes.filter(p => p.score?.flags?.length);
    const flagReasons = flaggedTakes.flatMap(p => (p.score?.flags ?? []).map(f => `take ${mine.indexOf(p) + 1}: ${f}`)).join('\n');
    const sc = scores[step];
    const overall = rungOverall({ likeness: sc?.likeness, corruption: sc?.corruption, plannerReplans, composerReplans, takes: doneTakes.length });
    return { mine, doneTakes, plannerReplans, composerReplans, hasReplanData, flaggedTakes, flagReasons, overall };
  };
  // Best rung by overall score, ties going to the lower (less-trained, so
  // less likely overcooked) step. Ladder is sorted ascending, so keeping the
  // first strictly-greater score already resolves ties that way.
  let bestStep: number | undefined;
  let bestOverall = -Infinity;
  for (const c of ladder) {
    const o = rungStats(c.step).overall;
    if (o && o.overall > bestOverall) { bestOverall = o.overall; bestStep = c.step; }
  }
  const scoreboardRows = ladder.map(c => {
    const stats = rungStats(c.step);
    const sc = scores[c.step];
    return { step: c.step, kl: c.kl, likeness: sc?.likeness ?? null, corruption: sc?.corruption ?? null, overall: stats.overall };
  }).sort((a, b) => {
    if (a.overall && b.overall) return b.overall.overall - a.overall.overall || a.step - b.step;
    if (a.overall) return -1;
    if (b.overall) return 1;
    return a.step - b.step;
  });
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
          <p className="mt-1 text-[12px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.refine.cleanupIntro', 'Everything below is app-generated and can be rebuilt. Source audio, sidecars, captions, labels and this rung\'s adapter files are never touched. Your rung scores are kept.')} {t('trainingStudio.refine.cleanupMovesAnyway', 'Either way the adapter\'s run folder moves to yue2-joint-adapters\\refined.')}</p>
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
              <button type="button" disabled={cleaning} onClick={() => void doCleanup(noCleanupChoice)} className="px-3 py-1.5 rounded-lg text-xs border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10">{t('trainingStudio.refine.cleanupSkip', 'Keep everything')}</button>
              <button type="button" disabled={cleaning || totalBytes === 0} onClick={() => void doCleanup()} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-500 text-white hover:bg-red-400 disabled:opacity-40">{cleaning ? t('trainingStudio.refine.cleaning', 'Cleaning…') : t('trainingStudio.refine.cleanupGo', 'Delete selected')}</button>
            </div>
          </div>
        </div>
      </div>}
      {cleanupNote && <div className="text-[12px] text-emerald-700 dark:text-emerald-300">{cleanupNote}</div>}
      <div className="rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 p-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-800 dark:text-zinc-100"><Sparkles size={16} className="text-amber-500" />{t('trainingStudio.refine.title', 'Refine the planner')}{detail?.name ? ` · ${detail.name}` : ''}</div>
        <p className="mt-1 text-[12px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.refine.intro', 'The safe presets stop the planner at a conservative KL. Some artists take more. This continues a finished run with the planner live and the decoder along for the ride, saves a checkpoint at every KL rung up to the ceiling, and renders a preview per rung so you can hear where it starts to fall apart late in the song. Pick the last good rung.')}</p>
        <div className="mt-4 flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <ParamLabel label={t('trainingStudio.refine.source', 'Finished run')} info={t('trainingStudio.refine.sourceInfo', 'The run to continue from. Its last checkpoint is the starting point: the planner is unfrozen there and both halves train on. Refinements of refinements are fine; the run list shows what each one reached.')} />
            <div className="flex items-center gap-2">
              <div className="flex-1 min-w-0">
                <StyledSelect accent="amber" className="w-full" value={source} onChange={v => setSource(String(v))} disabled={active || busy} searchable={false}
                  placeholder={t('trainingStudio.refine.pick', 'Pick a run')}
                  options={finished.map(r => ({ value: r.jobId, label: runLabel(r), hint: `${r.checkpoints.length} checkpoints · ${r.status}` }))} />
              </div>
              <button type="button" onClick={() => void remove(source)} disabled={!source || active || busy} title={t('trainingStudio.refine.deleteRun', 'Delete run')}
                className="p-2 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:text-red-500 hover:border-red-500/40 disabled:opacity-40"><Trash2 size={14} /></button>
            </div>
          </div>

          <div className="rounded-lg border border-zinc-300/60 dark:border-white/5 p-3">
            <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.secPlanner', 'Planner ladder')}</div>
            <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-3">
              <label className="flex flex-col gap-1">
                <ParamLabel label={t('trainingStudio.refine.ceiling', 'KL ceiling')} meta={t('trainingStudio.refine.ceilingMeta', '0.5–4 · default 2.0')} info={t('trainingStudio.refine.ceilingInfo', 'The planner stops when its KL to the base model reaches this. Late-song decay has shown from about 1.7 on some albums and not at all by 1.8 on others, so the ceiling is a search range, not a target: pick the last good rung by ear.')} />
                <input className={input} type="number" step="0.1" min={0.5} max={4} value={ceiling} disabled={active || busy} onChange={e => setCeiling(Number(e.target.value) || 2)} />
              </label>
              <label className="flex flex-col gap-1">
                <ParamLabel label={t('trainingStudio.refine.rung', 'Rung size (KL)')} meta={t('trainingStudio.refine.rungMeta', '0.05–1 · default 0.1')} info={t('trainingStudio.refine.rungInfo', 'A checkpoint is saved, and previews rendered, each time the KL reading crosses the next multiple of this. Smaller rungs give a finer ladder and more previews to listen to.')} />
                <input className={input} type="number" step="0.05" min={0.05} max={1} value={rung} disabled={active || busy} onChange={e => setRung(Number(e.target.value) || 0.1)} />
              </label>
              <label className="flex flex-col gap-1">
                <ParamLabel label={t('trainingStudio.refine.lrScale', 'Learning rate (× run)')} meta={t('trainingStudio.refine.lrMeta', '0.05–1 · default 0.1')} info={t('trainingStudio.refine.lrInfo', 'Fraction of the source run\'s rate. A converged adapter restarted at full rate diverged in three steps, so refinement walks: the rate ramps up over the first 30 steps and halves itself whenever the KL jumps more than one rung between checkpoints. Lower if rungs are still being skipped.')} />
                <input className={input} type="number" step="0.05" min={0.05} max={1} value={lrScale} disabled={active || busy} onChange={e => setLrScale(Math.max(0.05, Math.min(1, Number(e.target.value) || 0.1)))} />
              </label>
            </div>
          </div>

          <div className="rounded-lg border border-zinc-300/60 dark:border-white/5 p-3">
            <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.secDecoder', 'Decoder, after you pick a rung')}</div>
            <div className="mt-2 flex flex-wrap items-end gap-x-6 gap-y-3">
              <div className={`flex items-center gap-2 self-center ${active || busy ? 'opacity-50 pointer-events-none' : ''}`}>
                <Toggle id="refine-nar-further" checked={narFurther} onChange={setNarFurther} />
                <ParamLabel underline={false} className="text-xs normal-case tracking-normal font-normal text-zinc-700 dark:text-zinc-300" label={t('trainingStudio.refine.narFurther', 'Further training for NAR')} info={t('trainingStudio.refine.narFurtherInfo', 'When you press "Use this rung", the decoder trains on from that rung with the planner frozen, until the reconstruction target, the step budget, or the point where the reconstruction meter stops improving, whichever comes first. That final checkpoint becomes the adapter. Off: the rung is used as it is.')} />
              </div>
              <label className="flex flex-col gap-1 w-32">
                <ParamLabel label={t('trainingStudio.refine.narTarget', 'Recon target')} meta={t('trainingStudio.refine.narTargetMeta', 'default 0.25 · blank = knee')} info={t('trainingStudio.refine.narTargetInfo', 'Stop the decoder once its reconstruction error (how closely it reproduces the album\'s own latents on fixed probes; lower is better, typically 0.29–0.35 and falling) reaches this value. Some albums plateau above it; the knee rule and the step budget then stop the run instead.')} />
                <input className={input} type="number" step="0.005" min={0} max={10} value={narTarget} placeholder={t('trainingStudio.refine.narKnee', 'knee')} disabled={active || busy || !narFurther}
                  onChange={e => setNarTarget(e.target.value === '' ? '' : Math.max(0, Number(e.target.value) || 0))} />
              </label>
              <label className="flex flex-col gap-1 w-32">
                <ParamLabel label={t('trainingStudio.refine.narBudget', 'NAR max steps')} meta={t('trainingStudio.refine.narBudgetMeta', 'default 500')} info={t('trainingStudio.refine.narBudgetInfo', 'The most decoder steps to train after the pick. At about 1.5 s a step, 500 is roughly 13 minutes.')} />
                <input className={input} type="number" min={10} step={10} value={narBudget} disabled={active || busy || !narFurther} onChange={e => setNarBudget(Math.max(10, Math.round(Number(e.target.value) || 0)))} />
              </label>
              <label className="flex flex-col gap-1 w-40">
                <ParamLabel label={t('trainingStudio.refine.narKeepDelta', 'Keep a checkpoint per recon drop')} meta={t('trainingStudio.refine.narKeepDeltaMeta', 'default 0.003')} info={t('trainingStudio.refine.narKeepDeltaInfo', 'During further decoder training a checkpoint is kept only when the reconstruction meter has dropped by at least this since the last kept one; the rest are deleted so the ladder shows real progress, not every 10 steps. The newest checkpoint is always kept.')} />
                <input className={input} type="number" min={0} step={0.001} value={narKeepDelta} disabled={active || busy || !narFurther} onChange={e => setNarKeepDelta(Math.max(0, Number(e.target.value) || 0))} />
              </label>
            </div>
          </div>

          <div className="rounded-lg border border-zinc-300/60 dark:border-white/5 p-3">
            <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">{t('trainingStudio.refine.secPreviews', 'Rung previews')}</div>
            <div className="mt-2 flex flex-wrap items-end gap-x-6 gap-y-3">
              <div className={`flex items-center gap-2 self-center ${active || busy ? 'opacity-50 pointer-events-none' : ''}`}>
                <Toggle id="refine-auto-preview" checked={autoPreview} onChange={setAutoPreview} />
                <ParamLabel underline={false} className="text-xs normal-case tracking-normal font-normal text-zinc-700 dark:text-zinc-300" label={t('trainingStudio.refine.autoPreview', 'Render previews at each rung')} info={t('trainingStudio.refine.autoPreviewInfo', 'Render the tracks for every rung as it lands, so the ladder is ready to listen to when the run ends. Off: render rungs by hand from the ladder.')} />
              </div>
              <div className={`flex items-center gap-2 self-center ${active || busy || !autoPreview ? 'opacity-50 pointer-events-none' : ''}`}>
                <Toggle id="refine-parallel" checked={parallel} onChange={setParallel} />
                <ParamLabel underline={false} className="text-xs normal-case tracking-normal font-normal text-zinc-700 dark:text-zinc-300" label={t('trainingStudio.refine.parallel', 'In parallel with training')} info={t('trainingStudio.refine.parallelInfo', 'Render while training continues instead of pausing it at each rung. Needs the VRAM for both (about 22 GB measured); renders run at about half speed and training a little slower. Untick on a smaller card.')} />
              </div>
              <div className={`flex items-center gap-2 self-center ${active || busy ? 'opacity-50 pointer-events-none' : ''}`}>
                <Toggle id="refine-draft" checked={draft} onChange={setDraft} />
                <ParamLabel underline={false} className="text-xs normal-case tracking-normal font-normal text-zinc-700 dark:text-zinc-300" label={t('trainingStudio.refine.draft', 'Draft quality')} info={t('trainingStudio.refine.draftInfo', 'Previews only: the decoder runs 12 ODE steps instead of 32, with the velocity cache off (a raised cache ratio blunts artist likeness), about a third of the decoder time. Timbre is a little softer; structure, diction and late-song behaviour, which the planner sets, are unchanged. Off renders previews at production quality.')} />
              </div>
              <label className="flex flex-col gap-1 w-24">
                <ParamLabel label={t('trainingStudio.refine.takes', 'Tracks')} meta={t('trainingStudio.refine.takesMeta', '1–4 · default 2')} info={t('trainingStudio.refine.takesInfo', 'Tracks per rung, different seeds. Renders are not deterministic: one take can decay late while another from the same checkpoint is clean, so two is the minimum to trust a rung.')} />
                <input className={input} type="number" min={1} max={4} value={takes} disabled={active || busy} onChange={e => setTakes(Math.max(1, Math.min(4, Number(e.target.value) || 1)))} />
              </label>
              <label className="flex flex-col gap-1 w-28">
                <ParamLabel label={t('trainingStudio.refine.seconds', 'Length (s)')} meta={t('trainingStudio.refine.secondsMeta', '30–360 · default 300')} info={t('trainingStudio.refine.secondsInfo', 'Preview length. A take shorter than the planned song is cut mid-song at this limit and has no ending, which can read as corruption in its last seconds; the take then shows "preview_limit". Keep it long enough for a whole song (most albums fit in 300 s; 360 is the most).')} />
                <input className={input} type="number" min={30} max={360} step={30} value={seconds} disabled={active || busy} onChange={e => setSeconds(Math.max(30, Math.min(360, Number(e.target.value) || 300)))} />
              </label>
            </div>
          </div>
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
            <ParamLabel label={t('trainingStudio.refine.ladderRun', 'Ladder')} info={t('trainingStudio.refine.ladderInfo', 'A refinement run and its rungs. Rung cards carry the planner\'s KL and the decoder\'s reconstruction at that checkpoint, the rendered tracks, and your scores.')} />
            <div className="flex items-center gap-2">
              <div className="flex-1 min-w-0">
                <StyledSelect accent="amber" className="w-full" value={ladderRun} onChange={v => setLadderRun(String(v))} searchable={false}
                  placeholder={t('trainingStudio.refine.pickLadder', 'Pick a refinement run')}
                  options={runs.filter(r => r.checkpoints.some(c => c.kl !== undefined)).map(r => ({ value: r.jobId, label: runLabel(r), hint: `${r.checkpoints.filter(c => c.rung).length} rungs · ${r.checkpoints.length} checkpoints` }))} />
              </div>
              <button type="button" onClick={() => void remove(ladderRun)} disabled={!ladderRun || !!runs.find(r => r.jobId === ladderRun)?.live} title={t('trainingStudio.refine.deleteRun', 'Delete run')}
                className="p-2 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:text-red-500 hover:border-red-500/40 disabled:opacity-40"><Trash2 size={14} /></button>
            </div>
          </div>
        </div>
        <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.refine.ladderHint', 'A take marked preview_limit was cut at the length limit mid-song, so its abrupt end is the limit, not the planner. Renders are not deterministic: two tracks per rung is the minimum to trust a rung. Render adds more tracks to a rung with the count and length above.')}
          {' '}<a className="underline hover:text-zinc-700 dark:hover:text-zinc-300" href={yue2RungScoresExportUrl('csv')}>{t('trainingStudio.refine.exportCsv', 'Export all scores (CSV)')}</a>
          {' · '}<a className="underline hover:text-zinc-700 dark:hover:text-zinc-300" href={yue2RungScoresExportUrl('json')} target="_blank" rel="noreferrer">JSON</a></p>
        {finishedUnpickedNotice && <div className="mt-3 rounded-lg border border-amber-500/60 bg-amber-500/10 p-3 text-xs text-amber-800 dark:text-amber-200">
          {finishedUnpickedNotice}
        </div>}
        {ladder.length > 0 && <div className="mt-3 flex flex-col gap-3">
          {ladder.map(c => {
            const stats = rungStats(c.step);
            const { mine, plannerReplans, composerReplans, hasReplanData, flaggedTakes, flagReasons, overall } = stats;
            return <div key={c.step} id={`refine-rung-${c.step}`}
              className={`rounded-lg border p-3 ${picked === c.dir ? 'border-emerald-500/60 bg-emerald-500/5' : c.step === bestStep ? 'border-sky-500/70' : c.rung ? 'border-amber-500/40' : 'border-zinc-300/70 dark:border-white/10'}`}>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                <span className="font-semibold text-zinc-800 dark:text-zinc-100">{t('trainingStudio.refine.rungStep', 'Step {{step}}', { step: c.step })}</span>
                {c.rung && <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[10px] font-semibold">{t('trainingStudio.refine.rungBadge', 'rung')}</span>}
                {!c.rung && <span className="text-[10px] text-zinc-500">{t('trainingStudio.refine.routineSave', 'routine save')}</span>}
                {c.step === bestStep && <span className="px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-700 dark:text-sky-300 text-[10px] font-semibold">{t('trainingStudio.refine.bestBadge', 'best by score')}</span>}
                <span className="font-mono text-zinc-600 dark:text-zinc-300">KL {c.kl !== undefined ? c.kl.toFixed(2) : '—'}{c.frozen ? ' (frozen)' : ''}</span>
                <span className="font-mono text-zinc-600 dark:text-zinc-300">recon {c.recon !== undefined ? c.recon.toFixed(3) : '—'}</span>
                {hasReplanData && <span className="font-mono text-zinc-500" title={t('trainingStudio.refine.replansInfo', 'planner / composer re-plans across this rung\'s takes; rising counts are a sign of over-training')}>{t('trainingStudio.refine.replans', 'replans {{p}}/{{c}}', { p: plannerReplans, c: composerReplans })}</span>}
                {flaggedTakes.length > 0 && <span className="font-mono text-red-600 dark:text-red-400" title={flagReasons}>{t('trainingStudio.refine.planFlags', 'plan flags {{n}}/{{takes}}', { n: flaggedTakes.length, takes: stats.doneTakes.length })}</span>}
                {overall && <span className="font-mono text-sky-700 dark:text-sky-300" title={t('trainingStudio.refine.overallInfo', 'overall = (likeness + (6 − corruption)) / 2, minus a soft penalty for replan load')}>{t('trainingStudio.refine.overall', 'overall {{n}}', { n: overall.overall.toFixed(2) })}</span>}
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
                  ? <PreviewPlayer key={p.id} src={p.audioUrl} downloadName={`${datasetName || 'preview'}_step${c.step}_take${i + 1}_seed${p.seed}.wav`} label={t('trainingStudio.refine.take', 'Take {{n}}', { n: i + 1 })} sublabel={`${p.seconds} s · seed ${p.seed}${p.endReason && p.endReason !== 'completed' ? ` · ${p.endReason}` : ''}${p.score?.verdict ? ` · plan ${p.score.verdict}` : ''}${p.score?.flags?.length ? ` · ${p.score.flags[0]}` : ''}${p.plan ? ` · planner replans ${p.plan.attempts.length - 1}` : ''}${typeof p.composerReplans === 'number' ? ` · composer replans ${p.composerReplans}` : ''}`} />
                  : <div key={p.id} className="text-[11px] text-zinc-500">{t('trainingStudio.refine.take', 'Take {{n}}', { n: i + 1 })}: {p.status === 'done' && !p.file ? t('trainingStudio.refine.audioPruned', 'audio removed by cleanup') : p.status}{p.error ? ` — ${p.error}` : ''}{p.score?.verdict ? ` · plan ${p.score.verdict}` : ''}{p.score?.flags?.length ? ` · ${p.score.flags[0]}` : ''}</div>)}
              </div>}
            </div>;
          })}
        </div>}
      </div>

      {bestStep !== undefined && <div className="hidden md:block fixed right-4 bottom-4 z-40 max-w-xs rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white/90 dark:bg-zinc-900/90 shadow-xl p-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-100">{t('trainingStudio.refine.scoreboardTitle', 'Scoreboard')}</span>
          <button type="button" onClick={() => setScoreboardCollapsed(v => !v)} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200">
            {scoreboardCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
          </button>
        </div>
        {!scoreboardCollapsed && <>
          <p className="mt-1 text-[10px] text-zinc-500">{t('trainingStudio.refine.scoreboardFormula', 'Overall = (likeness + (6 − corruption)) / 2, minus 0.25 per replan per take, capped at 1.')}</p>
          <div className="mt-2 flex flex-col gap-1 max-h-64 overflow-y-auto">
            {scoreboardRows.map(row => (
              <button key={row.step} type="button"
                onClick={() => document.getElementById(`refine-rung-${row.step}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
                className={`w-full text-left px-2 py-1 rounded-lg text-[11px] ${row.overall ? (row.step === bestStep ? 'bg-sky-500/15 text-sky-800 dark:text-sky-200' : 'hover:bg-zinc-500/10 text-zinc-700 dark:text-zinc-300') : 'text-zinc-500'}`}>
                {row.overall ? <>
                  <div className="flex items-center justify-between gap-2">
                    <span>{t('trainingStudio.refine.scoreboardStep', 'step {{step}}', { step: row.step })}
                      {row.step === bestStep && <span className="ml-1 px-1 py-0.5 rounded bg-sky-500/20 text-[9px] font-semibold">{t('trainingStudio.refine.bestChip', 'best')}</span>}</span>
                    <span className="font-bold font-mono">{row.overall.overall.toFixed(2)}</span>
                  </div>
                  <div className="text-[10px] text-zinc-500">KL {row.kl !== undefined ? row.kl.toFixed(2) : '—'} · L {row.likeness} · C {row.corruption} · {t('trainingStudio.refine.scoreboardReplans', 'replans {{n}}/take', { n: row.overall.replansPerTake.toFixed(2) })}</div>
                </> : t('trainingStudio.refine.scoreboardUnscored', 'step {{step}} · unscored', { step: row.step })}
              </button>
            ))}
          </div>
        </>}
      </div>}
    </div>
  );
};

export default RefinePanel;
