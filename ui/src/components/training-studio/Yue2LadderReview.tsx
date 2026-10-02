// Yue2LadderReview.tsx — a joint run's checkpoint ladder as a listening test:
// one card per checkpoint with its previews, likeness/corruption scores and
// notes, a floating scoreboard, "Render more", and "Use this rung", which
// links the checkpoint as the dataset's adapter and opens the cleanup modal.
//
// Shared by the training card (every base-matched run's own ladder, 2026-09-27)
// and the Refine tab (KL-rung ladders of the earlier recipe). Extracted from
// RefinePanel; the scoring maths mirrors server/src/services/training/
// yue2BestRung.ts, so change both together.
import { forwardRef, useEffect, useImperativeHandle, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Toggle } from '../settings/SettingsPrimitives';
import { PreviewPlayer } from './PreviewPlayer';
import { ParamLabel } from '../shared/ParamLabel';
import { usePersistedState, writePersistedState } from '../../hooks/usePersistedState';
import {
  getYue2AlbumScore, getYue2CleanupPlan, linkYue2JointCheckpointPreset, listYue2RungScores, renderYue2JointPreviews, runYue2Cleanup, scoreYue2Album, scoreYue2Rung,
  type Yue2AitkRunRecord, type Yue2AlbumScore, type Yue2TrainedDirection, type Yue2CleanupChoice, type Yue2CleanupPlan, type Yue2JointPreviewRecord, type Yue2RungScore,
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
  /** Plan legibility flags on takes this rung planned itself (not a shared
   *  sheet), and how many such takes there were. Weighted under replans. */
  planFlags?: number;
  ownTakes?: number;
}): { overall: number; replansPerTake: number } | null {
  const { likeness, corruption, plannerReplans, composerReplans, takes, planFlags = 0, ownTakes = takes } = args;
  if (typeof likeness !== 'number' || typeof corruption !== 'number') return null;
  const base = (likeness + (6 - corruption)) / 2;
  const replansPerTake = (plannerReplans + composerReplans) / Math.max(1, takes);
  const penalty = Math.min(1, 0.25 * replansPerTake + 0.1 * planFlags / Math.max(1, ownTakes));
  return { overall: Math.round((base - penalty) * 100) / 100, replansPerTake };
}

export interface Yue2LadderReviewHandle {
  /** Link `dir` (step `step` of `run`) as the adapter and open the cleanup modal. */
  finishPick: (run: string, dir: string, step: number) => Promise<void>;
}

export const Yue2LadderReview = forwardRef<Yue2LadderReviewHandle, {
  datasetId: string;
  datasetName?: string;
  /** The ladder's run; its complete checkpoints are the rungs. */
  run: Yue2AitkRunRecord | undefined;
  previews: Yue2JointPreviewRecord[];
  /** "Render more" settings. */
  renderOpts: { seconds: number; takes: number; draft: boolean };
  /** Called after a cleanup or a render, so the owner refreshes runs and previews. */
  onChanged: () => void | Promise<void>;
  /** Owner's take on "Use this rung". Resolve true when handled (nothing
   *  more happens here); false or absent = link the rung and open cleanup. */
  onUse?: (dir: string, step: number) => Promise<boolean>;
  onPicked?: (dir: string) => void;
  onError?: (message: string) => void;
  /** Element id prefix for scoreboard scroll targets; unique per instance. */
  idPrefix?: string;
}>(({ datasetId, datasetName, run, previews, renderOpts, onChanged, onUse, onPicked, onError, idPrefix = 'ladder-rung' }, ref) => {
  const { t } = useTranslation();
  const runId = run?.jobId ?? '';
  const [blindRungs] = usePersistedState('hs-yue2-blind-rungs', true);
  const jointLadder = run?.options.method === 'base-matched';
  const blind = jointLadder && blindRungs;
  const labelFor = (step: number) => run?.blindLabels?.[step] ?? '';
  const rungName = (step: number) => blind
    ? t('trainingStudio.refine.blindRung', 'Rung {{label}}', { label: labelFor(step) || '?' })
    : t('trainingStudio.refine.rungStep', 'Step {{step}}', { step });
  const ladder = run ? [...run.checkpoints].filter(c => c.arPath && c.narPath).sort((a, b) => a.step - b.step) : [];
  const visibleLadder = blind ? [...ladder].sort((a, b) => labelFor(a.step).localeCompare(labelFor(b.step))) : ladder;
  const fail = (err: unknown) => onError?.(err instanceof Error ? err.message : String(err));
  const draftOpts = renderOpts.draft ? { odeSteps: 12, narCacheRatio: 0 } : {};

  // Scores per rung (by step); notes are saved on blur.
  const [scores, setScores] = useState<Record<number, Yue2RungScore>>({});
  const [noteDraft, setNoteDraft] = useState<Record<number, string>>({});
  useEffect(() => {
    setScores({}); setNoteDraft({});
    if (datasetId && runId) void listYue2RungScores(datasetId, runId).then(r => setScores(Object.fromEntries(r.scores.map(s => [s.step, s])))).catch(() => {});
  }, [datasetId, runId]);
  const score = async (step: number, patch: { likeness?: number | null; corruption?: number | null; notes?: string }) => {
    if (!datasetId || !runId) return;
    try { const r = await scoreYue2Rung(datasetId, { refineRun: runId, step, ...patch, blind: !!blind, blindLabel: blind ? labelFor(step) : '' }); setScores(prev => ({ ...prev, [step]: r.score })); }
    catch (err) { fail(err); }
  };

  // The album verdict: one score per run, beside the per-rung ones.
  const [album, setAlbum] = useState<Yue2AlbumScore | null>(null);
  const [albumNote, setAlbumNote] = useState<string | undefined>(undefined);
  useEffect(() => {
    setAlbum(null); setAlbumNote(undefined);
    if (datasetId && runId) void getYue2AlbumScore(datasetId, runId).then(r => setAlbum(r.score)).catch(() => {});
  }, [datasetId, runId]);
  const scoreAlbum = async (patch: { score?: number | null; instruments?: Yue2TrainedDirection | null; vocals?: Yue2TrainedDirection | null; notes?: string }) => {
    if (!datasetId || !runId) return;
    try { setAlbum((await scoreYue2Album(datasetId, { refineRun: runId, ...patch })).score); }
    catch (err) { fail(err); }
  };

  const [rendering, setRendering] = useState<number | null>(null);
  const render = async (step: number) => {
    if (!datasetId || !runId) return;
    setRendering(step);
    try { await renderYue2JointPreviews(datasetId, { run: runId, step, seconds: renderOpts.seconds, takes: renderOpts.takes, ...draftOpts }); await onChanged(); }
    catch (err) { fail(err); }
    finally { setRendering(null); }
  };

  // Cleanup modal after a rung is chosen: what else can go, with sizes.
  const [picked, setPicked] = useState('');
  const [cleanup, setCleanup] = useState<{ run: string; step: number; plan: Yue2CleanupPlan; blind: boolean; blindLabel: string } | null>(null);
  const [choice, setChoice] = useState<Yue2CleanupChoice>({ caches: true, otherCheckpoints: true, otherRuns: true, resume: true, otherPreviews: true });
  const [cleaning, setCleaning] = useState(false);
  const [cleanupNote, setCleanupNote] = useState('');
  const [scoreboardCollapsed, setScoreboardCollapsed] = useState(false);
  const mib = (b: number) => b >= 1073741824 ? `${(b / 1073741824).toFixed(2)} GiB` : `${(b / 1048576).toFixed(0)} MiB`;
  const finishPick = async (pickRun: string, dir: string, step: number) => {
    try {
      await linkYue2JointCheckpointPreset(datasetId, dir); setPicked(dir); onPicked?.(dir);
      const plan = await getYue2CleanupPlan(datasetId, pickRun, step);
      setCleanup({ run: pickRun, step, plan, blind: !!blind, blindLabel: blind ? labelFor(step) : '' });
    } catch (err) { fail(err); }
  };
  useImperativeHandle(ref, () => ({ finishPick }), [datasetId]);
  const use = async (dir: string, step: number) => {
    if (!runId) return;
    setCleanupNote('');
    try {
      if (onUse && await onUse(dir, step)) return;
      await finishPick(runId, dir, step);
    } catch (err) { fail(err); }
  };
  const noCleanupChoice: Yue2CleanupChoice = { caches: false, otherCheckpoints: false, otherRuns: false, resume: false, otherPreviews: false };
  const doCleanup = async (over?: Yue2CleanupChoice) => {
    if (!cleanup) return;
    setCleaning(true);
    try {
      const r = await runYue2Cleanup(datasetId, { run: cleanup.run, step: cleanup.step, blind: cleanup.blind, blindLabel: cleanup.blindLabel, ...(over ?? choice) });
      if (r.finishError) { fail(r.finishError); return; }
      setCleanupNote(`${t('trainingStudio.refine.cleanupDone', 'Removed {{what}}; about {{size}} freed.', { what: r.done.join(', ') || 'nothing', size: mib(r.freedBytes) })}${cleanup.blind ? ` ${t('trainingStudio.refine.blindReveal', 'Rung {{label}} was step {{step}}.', { label: cleanup.blindLabel, step: cleanup.step })}` : ''}`);
      setCleanup(null);
      await onChanged();
    } catch (err) { fail(err); }
    finally { setCleaning(false); }
  };

  // Per-rung facts shared by the ladder cards and the scoreboard: this
  // rung's previews, its replan load, and its overall score (null unless
  // it's been scored on both axes).
  const rungStats = (step: number) => {
    // Fixed-seed take first, then random-seed takes in render order.
    const mine = previews.filter(p => p.step === step).sort((a, b) => Number(a.seedKind === 'random') - Number(b.seedKind === 'random') || a.createdAt - b.createdAt || a.seed - b.seed);
    const doneTakes = mine.filter(p => p.status === 'done');
    const plannerReplans = doneTakes.reduce((sum, p) => sum + (p.plan ? p.plan.attempts.length - 1 : 0), 0);
    const composerReplans = doneTakes.reduce((sum, p) => sum + (typeof p.composerReplans === 'number' ? p.composerReplans : 0), 0);
    const hasReplanData = doneTakes.some(p => p.plan || typeof p.composerReplans === 'number');
    // Legibility flags on the rendered plans: a plan the verdict passed that
    // still loops a riff, sings on two pitches or skips lyric sections. A
    // count and its reasons, not part of the score, until ear scores have
    // been checked against them.
    const flaggedTakes = doneTakes.filter(p => p.sheet !== 'shared' && p.score?.flags?.length);
    const flagReasons = flaggedTakes.flatMap(p => (p.score?.flags ?? []).map(f => `take ${mine.indexOf(p) + 1}: ${f}`)).join('\n');
    const sc = scores[step];
    // A shared-sheet take's plan is not this rung's, so its flags do not count.
    const ownPlanned = doneTakes.filter(p => p.sheet !== 'shared');
    const planFlags = ownPlanned.reduce((sum, p) => sum + (p.score?.flags?.length ?? 0), 0);
    const overall = rungOverall({ likeness: sc?.likeness, corruption: sc?.corruption, plannerReplans, composerReplans, takes: doneTakes.length, planFlags, ownTakes: ownPlanned.length });
    // Takes whose re-plan loop ran out without a clean plan (plan flags or a
    // broken verdict on every attempt): the checkpoint itself is suspect.
    const unclean = doneTakes.filter(p => p.plan && p.plan.clean === false);
    return { mine, doneTakes, plannerReplans, composerReplans, hasReplanData, flaggedTakes, flagReasons, overall, unclean };
  };
  // Best rung by overall score, ties going to the higher (more-trained)
  // step. Ladder is sorted ascending, so keeping the last score that is
  // greater or equal resolves ties that way.
  let bestStep: number | undefined;
  let bestOverall = -Infinity;
  for (const c of ladder) {
    const o = rungStats(c.step).overall;
    if (o && o.overall >= bestOverall) { bestOverall = o.overall; bestStep = c.step; }
  }
  const scoreboardRows = ladder.map(c => {
    const stats = rungStats(c.step);
    const sc = scores[c.step];
    return { step: c.step, kl: c.kl, likeness: sc?.likeness ?? null, corruption: sc?.corruption ?? null, overall: stats.overall };
  }).sort((a, b) => {
    if (blind) return labelFor(a.step).localeCompare(labelFor(b.step));
    if (a.overall && b.overall) return b.overall.overall - a.overall.overall || b.step - a.step;
    if (a.overall) return -1;
    if (b.overall) return 1;
    return a.step - b.step;
  });
  const items: Array<{ key: keyof Yue2CleanupChoice; label: string; item?: { count: number; bytes: number; detail?: string[] } }> = cleanup ? [
    { key: 'caches', label: t('trainingStudio.refine.cleanCaches', 'Vocal stems and other backends\' caches (MM3, ACE) for this dataset. The YuE2 latents, codes, lead sheets and prepared set are kept: small, and slow to rebuild'), item: cleanup.plan.caches },
    { key: 'otherCheckpoints', label: t('trainingStudio.refine.cleanCheckpoints', 'The other checkpoints in this run'), item: cleanup.plan.otherCheckpoints },
    { key: 'otherRuns', label: t('trainingStudio.refine.cleanRuns', 'All other joint runs for this dataset'), item: cleanup.plan.otherRuns },
    { key: 'resume', label: t('trainingStudio.refine.cleanResume', 'This rung\'s resume file (the adapter files stay)'), item: cleanup.plan.resume },
    { key: 'otherPreviews', label: t('trainingStudio.refine.cleanPreviews', 'Previews from the other rungs'), item: cleanup.plan.otherPreviews },
  ] : [];
  const totalBytes = items.reduce((s, i) => s + (choice[i.key] && i.item ? i.item.bytes : 0), 0);
  return (
    <>
      {cleanup && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => !cleaning && setCleanup(null)}>
        <div className="w-full max-w-xl rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white dark:bg-zinc-900 p-5 shadow-xl" onClick={e => e.stopPropagation()}>
          <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">{cleanup.blind
            ? t('trainingStudio.refine.blindCleanupTitle', 'Rung {{label}} is now the adapter. Clean up around it?', { label: cleanup.blindLabel })
            : t('trainingStudio.refine.cleanupTitle', 'Step {{step}} is now the adapter. Clean up around it?', { step: cleanup.step })}</div>
          <p className="mt-1 text-[12px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.refine.cleanupIntro', 'Everything below is app-generated and can be rebuilt. Source audio, sidecars, captions, labels and this rung\'s adapter files are never touched. Your rung scores are kept.')} {!cleanup.blind && t('trainingStudio.refine.cleanupMovesAnyway', 'Either way the adapter\'s run folder moves to yue2-joint-adapters\\refined.')}</p>
          <div className="mt-3 flex flex-col gap-2">
            {items.map(i => <label key={i.key} className="flex items-start gap-3 text-xs text-zinc-700 dark:text-zinc-300">
              <Toggle id={`cleanup-${idPrefix}-${i.key}`} checked={!!choice[i.key] && !!i.item?.count} onChange={v => setChoice(prev => ({ ...prev, [i.key]: v }))} />
              <span className={`flex-1 ${!i.item?.count ? 'opacity-50' : ''}`}>{i.label}<span className="ml-2 text-zinc-500 tabular-nums">{i.item ? `${i.item.count} · ${mib(i.item.bytes)}` : ''}</span>
                {!cleanup.blind && i.item?.detail?.length ? <span className="block text-[10px] text-zinc-500 truncate">{i.item.detail.join(', ')}</span> : null}</span>
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
      {ladder.length > 0 && <div className="mt-3 flex flex-col gap-3">
        {jointLadder && <div className="flex items-center gap-2 text-xs">
          <Toggle id={`blind-${idPrefix}`} checked={blindRungs} onChange={value => writePersistedState('hs-yue2-blind-rungs', value)} />
          <ParamLabel label={t('trainingStudio.refine.blindToggle', 'Blind rungs')}
            info={t('trainingStudio.refine.blindToggleInfo', 'Show lettered rungs while listening and scoring. The chosen step is revealed after cleanup finishes.')} />
        </div>}
        <div className="rounded-lg border border-violet-500/40 bg-violet-500/5 p-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px]">
          <ParamLabel label={t('trainingStudio.refine.albumScore', 'How well did this album train?')}
            info={t('trainingStudio.refine.albumScoreInfo', 'One score for the whole run, once you have heard enough of its ladder: 1 = the adapter never really caught the album, 5 = as good as the best albums you have trained. Score it against your other albums, not against this run\'s own rungs. The rung scores show how fast an album trains; this is the only score that shows how well, and Dataset-Calibrated Training learns from it.')} />
          <div className="flex items-center gap-1">
            {[1, 2, 3, 4, 5].map(n => <button key={n} type="button" onClick={() => void scoreAlbum({ score: album?.score === n ? null : n })}
              className={`w-6 h-6 rounded border text-[11px] font-semibold ${album?.score === n ? 'bg-violet-500/20 border-violet-500 text-violet-700 dark:text-violet-300' : 'border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10'}`}>{n}</button>)}
          </div>
          {(['instruments', 'vocals'] as const).map(half => <div key={half} className="flex items-center gap-1">
            <ParamLabel label={half === 'instruments' ? t('trainingStudio.refine.albumInstruments', 'Instruments') : t('trainingStudio.refine.albumVocals', 'Vocals')}
              className="text-[11px] text-zinc-500 w-20"
              info={half === 'instruments'
                ? t('trainingStudio.refine.albumInstrumentsInfo', 'Which way the instruments missed at the rung you would pick. Under: the band\'s sound, playing and arrangement never really arrived. Right: nothing to change. Over: overcooked, memorised riffs or the same parts in every song, a smeared or brittle mix. The score says how well; this says whether the next run should train more or less.')
                : t('trainingStudio.refine.albumVocalsInfo', 'Which way the vocals missed at the rung you would pick. Under: the singer\'s voice and delivery never really arrived. Right: nothing to change. Over: overcooked, garbled words, one phrase or melody in every song, a strained or breaking voice. The score says how well; this says whether the next run should train more or less.')} />
            {(['under', 'right', 'over'] as const).map(d => <button key={d} type="button" onClick={() => void scoreAlbum({ [half]: album?.[half] === d ? null : d })}
              className={`px-2 h-6 rounded border text-[11px] font-semibold ${album?.[half] === d ? (d === 'right' ? 'bg-emerald-500/20 border-emerald-500 text-emerald-700 dark:text-emerald-300' : 'bg-violet-500/20 border-violet-500 text-violet-700 dark:text-violet-300') : 'border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10'}`}>
              {d === 'under' ? t('trainingStudio.refine.dirUnder', 'Under') : d === 'right' ? t('trainingStudio.refine.dirRight', 'Right') : t('trainingStudio.refine.dirOver', 'Over')}</button>)}
          </div>)}
          <input className={`${input} flex-1 min-w-[240px]`} placeholder={t('trainingStudio.refine.albumNotes', 'Notes: what this album got right or never got')}
            value={albumNote ?? album?.notes ?? ''} onChange={e => setAlbumNote(e.target.value)}
            onBlur={e => { if (e.target.value !== (album?.notes ?? '')) void scoreAlbum({ notes: e.target.value }); }} />
        </div>
        {visibleLadder.map(c => {
          const stats = rungStats(c.step);
          const { mine, plannerReplans, composerReplans, hasReplanData, flaggedTakes, flagReasons, overall } = stats;
          return <div key={c.step} id={`${idPrefix}-${c.step}`}
            className={`rounded-lg border p-3 ${picked === c.dir ? 'border-emerald-500/60 bg-emerald-500/5' : c.step === bestStep ? 'border-sky-500/70' : c.rung ? 'border-amber-500/40' : 'border-zinc-300/70 dark:border-white/10'}`}>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
              <span className="font-semibold text-zinc-800 dark:text-zinc-100">{rungName(c.step)}</span>
              {!blind && c.rung && c.kl !== undefined && <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[10px] font-semibold">{t('trainingStudio.refine.rungBadge', 'rung')}</span>}
              {!blind && !c.rung && <span className="text-[10px] text-zinc-500">{t('trainingStudio.refine.routineSave', 'routine save')}</span>}
              {c.step === bestStep && <span className="px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-700 dark:text-sky-300 text-[10px] font-semibold">{t('trainingStudio.refine.bestBadge', 'best by score')}</span>}
              {!blind && c.kl !== undefined && <span className="font-mono text-zinc-600 dark:text-zinc-300">KL {c.kl.toFixed(2)}{c.frozen ? ' (frozen)' : ''}</span>}
              {!blind && c.recon !== undefined && <span className="font-mono text-zinc-600 dark:text-zinc-300">recon {c.recon.toFixed(3)}</span>}
              {hasReplanData && <span className="font-mono text-zinc-500" title={blind ? undefined : t('trainingStudio.refine.replansInfo', 'planner / composer re-plans across this rung\'s takes; rising counts are a sign of over-training')}>{t('trainingStudio.refine.replans', 'replans {{p}}/{{c}}', { p: plannerReplans, c: composerReplans })}</span>}
              {flaggedTakes.length > 0 && <span className="font-mono text-red-600 dark:text-red-400" title={flagReasons}>{t('trainingStudio.refine.planFlags', 'plan flags {{n}}/{{takes}}', { n: flaggedTakes.length, takes: stats.doneTakes.length })}</span>}
              {overall && <span className="font-mono text-sky-700 dark:text-sky-300" title={t('trainingStudio.refine.overallInfo', 'overall = (likeness + (6 − corruption)) / 2, minus a soft penalty for replan load')}>{t('trainingStudio.refine.overall', 'overall {{n}}', { n: overall.overall.toFixed(2) })}</span>}
              <span className="flex-1" />
              <button type="button" onClick={() => void render(c.step)} disabled={rendering !== null}
                className="px-2 py-1 rounded-lg text-[11px] border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 disabled:opacity-40">
                {rendering === c.step ? t('trainingStudio.refine.rendering', 'Rendering…') : mine.length ? t('trainingStudio.refine.render', 'Render more') : t('trainingStudio.refine.renderFirst', 'Render')}
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
            {stats.unclean.length > 0 && <div className="mt-2 rounded-lg border-2 border-red-500 bg-red-500/10 px-3 py-2 text-[12px] text-red-800 dark:text-red-200">
              <div className="font-semibold">{t('trainingStudio.refine.noCleanPlan', 'This checkpoint is not writing good plans: no clean lead sheet in {{n}} tries.', { n: stats.unclean[0].plan!.attempts.length })}</div>
              <div className="mt-0.5">{t('trainingStudio.refine.noCleanPlanDetail', 'The take below was rendered from the least-flagged plan anyway. Flags on it:')} {(stats.unclean[0].score?.flags ?? stats.unclean[0].plan!.attempts[stats.unclean[0].plan!.attempts.length - 1].flags ?? []).join('; ') || t('trainingStudio.refine.noCleanPlanBroken', 'the judge rejected every plan')}</div>
            </div>}
            {mine.length > 0 && <div className="mt-2 flex flex-col gap-2">
              {mine.map((p, i) => p.audioUrl && p.status === 'done'
                ? <PreviewPlayer key={p.id} src={p.audioUrl} downloadName={`${datasetName || 'preview'}_${blind ? `rung${labelFor(c.step)}` : `step${c.step}`}_take${i + 1}_seed${p.seed}.wav`} label={`${t('trainingStudio.refine.take', 'Take {{n}}', { n: i + 1 })}${p.sheet === 'own' ? ` · ${t('trainingStudio.refine.sheetOwn', "this rung's plan")}` : p.sheet === 'shared' ? ` · ${blind ? t('trainingStudio.refine.blindSharedSheet', 'shared sheet from Rung {{label}}', { label: labelFor(p.sheetStep ?? 0) || '?' }) : t('trainingStudio.refine.sheetShared', 'shared sheet from step {{s}}', { s: p.sheetStep })}` : ''}${p.seedKind === 'fixed' ? ` · ${t('trainingStudio.refine.seedFixed', 'same seed on every rung: compare rungs on this one')}` : p.seedKind === 'random' ? ` · ${t('trainingStudio.refine.seedRandom', 'random seed: a new song, not comparable with other rungs')}` : ''}`} sublabel={`${p.seconds} s · seed ${p.seed}${p.endReason && p.endReason !== 'completed' ? ` · ${p.endReason}` : ''}${p.score?.verdict ? ` · plan ${p.score.verdict}` : ''}${p.score?.flags?.length ? ` · ⚠ ${p.score.flags.join('; ')}` : ''}${p.plan ? ` · planner replans ${p.plan.attempts.length - 1}` : ''}${typeof p.composerReplans === 'number' ? ` · composer replans ${p.composerReplans}` : ''}`} />
                : <div key={p.id} className="text-[11px] text-zinc-500">{t('trainingStudio.refine.take', 'Take {{n}}', { n: i + 1 })}: {p.status === 'done' && !p.file ? t('trainingStudio.refine.audioPruned', 'audio removed by cleanup') : p.status}{p.error ? ` — ${p.error}` : ''}{p.score?.verdict ? ` · plan ${p.score.verdict}` : ''}{p.score?.flags?.length ? ` · ${p.score.flags[0]}` : ''}</div>)}
            </div>}
          </div>;
        })}
      </div>}
      {bestStep !== undefined && <div className="hidden md:block fixed right-4 bottom-4 z-40 max-w-xs rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white/90 dark:bg-zinc-900/90 shadow-xl p-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-100">{t('trainingStudio.refine.scoreboardTitle', 'Scoreboard')}</span>
          <button type="button" onClick={() => setScoreboardCollapsed(v => !v)} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200">
            {scoreboardCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
          </button>
        </div>
        {!scoreboardCollapsed && <>
          <p className="mt-1 text-[10px] text-zinc-500">{t('trainingStudio.refine.scoreboardFormula', 'Overall = (likeness + (6 − corruption)) / 2, minus 0.25 per replan per take and 0.1 per plan flag per take this rung planned, capped at 1 together.')}</p>
          <div className="mt-2 flex flex-col gap-1 max-h-64 overflow-y-auto">
            {scoreboardRows.map(row => (
              <button key={row.step} type="button"
                onClick={() => document.getElementById(`${idPrefix}-${row.step}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
                className={`w-full text-left px-2 py-1 rounded-lg text-[11px] ${row.overall ? (row.step === bestStep ? 'bg-sky-500/15 text-sky-800 dark:text-sky-200' : 'hover:bg-zinc-500/10 text-zinc-700 dark:text-zinc-300') : 'text-zinc-500'}`}>
                {row.overall ? <>
                  <div className="flex items-center justify-between gap-2">
                    <span>{blind ? rungName(row.step) : t('trainingStudio.refine.scoreboardStep', 'step {{step}}', { step: row.step })}
                      {row.step === bestStep && <span className="ml-1 px-1 py-0.5 rounded bg-sky-500/20 text-[9px] font-semibold">{t('trainingStudio.refine.bestChip', 'best')}</span>}</span>
                    <span className="font-bold font-mono">{row.overall.overall.toFixed(2)}</span>
                  </div>
                  <div className="text-[10px] text-zinc-500">{!blind && row.kl !== undefined ? `KL ${row.kl.toFixed(2)} · ` : ''}L {row.likeness} · C {row.corruption} · {t('trainingStudio.refine.scoreboardReplans', 'replans {{n}}/take', { n: row.overall.replansPerTake.toFixed(2) })}</div>
                </> : blind ? `${rungName(row.step)} · ${t('trainingStudio.refine.unscored', 'unscored')}` : t('trainingStudio.refine.scoreboardUnscored', 'step {{step}} · unscored', { step: row.step })}
              </button>
            ))}
          </div>
        </>}
      </div>}
    </>
  );
});
Yue2LadderReview.displayName = 'Yue2LadderReview';

export default Yue2LadderReview;
