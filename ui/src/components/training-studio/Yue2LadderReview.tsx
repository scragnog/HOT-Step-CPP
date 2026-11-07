// Yue2LadderReview.tsx — a joint run's checkpoint ladder as a listening test:
// one card per checkpoint with its previews, likeness/corruption scores and
// notes, a floating scoreboard, "Render more", and "Use this rung", which
// links the checkpoint as the dataset's adapter and opens the cleanup modal.
//
// The rungs draw through CheckpointLadder and the review parts in
// LadderReview.tsx, both shared with MM3; this file owns YuE2's data: scores,
// blind labels, the album score, replans and plan flags, and cleanup.
// Shared by the training card (every base-matched run's own ladder, 2026-09-27)
// and the Refine tab (KL-rung ladders of the earlier recipe). Extracted from
// RefinePanel; the scoring maths mirrors server/src/services/training/
// yue2BestRung.ts, so change both together.
import { yue2SamplerSnapshot } from '../../stores/globalParamsStore';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { resourceKey } from '../../queries/keys';
import { runQueries, trainingWritePlan } from '../../queries/training-runs/queries';
import {
  renderYue2JointPreviews, scoreYue2Album, scoreYue2Rung,
  type Yue2AitkRunRecord, type Yue2AlbumScore,
  type Yue2CleanupChoice, type Yue2CleanupPlan, type Yue2JointPreviewRecord, type Yue2RungScore,
  type Yue2TrainedDirection
} from '../../services/trainingApi';
import { CheckpointLadder, type CheckpointLadderRung } from './CheckpointLadder';
import { AlbumScoreBar, BlindRungsToggle, LadderCleanupModal, LadderScoreboard, mib, RungScoreRow, useBlindRungs, type CleanupItemRow } from './LadderReview';
import type { PreviewTake } from './TrainingPreviewPanel';

import { cleanupReviewRung, pickFromLadder, selectReviewRung, type ReviewPick } from '../../services/trainingReviewApi';

/** Blind-ladder visibility for a run: labels pending (known locally, no blind
 *  letters yet) must never fall back to listing rungs in their real, revealing
 *  step order just because every label reads empty — no ladder renders, no
 *  scoring, until the labels land. Pure so it's unit-testable without React. */
export function ladderVisibility(run: Yue2AitkRunRecord | undefined, blindRungs: boolean) {
  const jointLadder = run?.options.method === 'base-matched' || run?.options.method === 'fixed-recipe';
  const labelsPending = !!jointLadder && blindRungs && run != null && !run.blindLabels;
  const blind = !!jointLadder && blindRungs && !labelsPending;
  const labelFor = (step: number) => run?.blindLabels?.[step] ?? '';
  // A remote-only rung (pulled from a worker, no local weights yet) is still
  // listenable and scoreable from its previews and meters.
  const ladder = run && !labelsPending ? [...run.checkpoints].filter(c => c.arPath && c.narPath).sort((a, b) => a.step - b.step) : [];
  const visibleLadder = blind ? [...ladder].sort((a, b) => labelFor(a.step).localeCompare(labelFor(b.step))) : ladder;
  return { jointLadder, labelsPending, blind, labelFor, ladder, visibleLadder };
}

export interface Yue2LadderReviewHandle {
  /** Link `dir` (step `step` of `run`) as the adapter and open the cleanup modal. */
  finishPick: (run: string, dir: string, step: number) => void;
}

export const Yue2LadderReview = forwardRef<Yue2LadderReviewHandle, {
  datasetId: string;
  datasetName?: string;
  /** The ladder's run; its complete checkpoints are the rungs. */
  run: Yue2AitkRunRecord | undefined;
  previews: Yue2JointPreviewRecord[];
  /** "Render more" settings. */
  renderOpts: { seconds: number; takes: number; arScale?: number; draft: boolean };
  /** Owner's take on "Use this rung". Resolve true when handled (nothing
   *  more happens here); false or absent = link the rung and open cleanup. */
  onUse?: (dir: string, step: number) => Promise<boolean>;
  onPicked?: (dir: string) => void;
  onError?: (message: string) => void;
  /** Element id prefix for scoreboard scroll targets; unique per instance. */
  idPrefix?: string;
}>(({ datasetId, datasetName, run, previews, renderOpts, onUse, onPicked, onError, idPrefix = 'ladder-rung' }, ref) => {
  const { t } = useTranslation();
  const runId = run?.jobId ?? '';
  const client = useQueryClient();
  const reviewQuery = useQuery({ ...runQueries.ladder(datasetId, runId), enabled: !!datasetId && !!runId });
  const review = reviewQuery.data ?? null;
  const blindRungs = useBlindRungs();
  const { jointLadder, labelsPending, blind, labelFor, ladder, visibleLadder } = ladderVisibility(run, blindRungs);
  const rungName = (step: number) => blind
    ? t('trainingStudio.refine.blindRung', 'Rung {{label}}', { label: labelFor(step) || '?' })
    : t('trainingStudio.refine.rungStep', 'Step {{step}}', { step });
  // This ladder's records as the shared take rows; the sheet, seed and plan
  // wording stays here because only YuE2 has lead sheets.
  const takeRows = (takes: Yue2JointPreviewRecord[], step: number): PreviewTake[] => takes.map((p, i) => {
    const take = t('trainingStudio.refine.take', 'Take {{n}}', { n: i + 1 });
    if (!(p.audioUrl && p.status === 'done')) return { id: p.id, text: `${take}: ${p.status === 'done' && !p.file ? t('trainingStudio.refine.audioPruned', 'audio removed by cleanup') : p.status}${p.error ? ` — ${p.error}` : ''}${p.score?.verdict ? ` · plan ${p.score.verdict}` : ''}${p.score?.flags?.length ? ` · ${p.score.flags[0]}` : ''}`, label: take };
    return {
      id: p.id, src: `${p.audioUrl}&format=mp3`, downloadSrc: p.audioUrl,
      downloadName: `${datasetName || 'preview'}_${blind ? `rung${labelFor(step)}` : `step${step}`}_take${i + 1}_seed${p.seed}.wav`,
      label: `${take}${p.sheet === 'own' ? ` · ${t('trainingStudio.refine.sheetOwn', "this rung's plan")}` : p.sheet === 'shared' ? ` · ${blind ? t('trainingStudio.refine.blindSharedSheet', 'shared sheet from Rung {{label}}', { label: labelFor(p.sheetStep ?? 0) || '?' }) : t('trainingStudio.refine.sheetShared', 'shared sheet from step {{s}}', { s: p.sheetStep })}` : ''}${p.seedKind === 'fixed' ? ` · ${t('trainingStudio.refine.seedFixed', 'same seed on every rung: compare rungs on this one')}` : p.seedKind === 'random' ? ` · ${t('trainingStudio.refine.seedRandom', 'random seed: a new song, not comparable with other rungs')}` : ''}`,
      sublabel: `${p.seconds} s · seed ${p.seed}${p.endReason && p.endReason !== 'completed' ? ` · ${p.endReason}` : ''}${p.score?.verdict ? ` · plan ${p.score.verdict}` : ''}${p.score?.flags?.length ? ` · ⚠ ${p.score.flags.join('; ')}` : ''}${p.plan ? ` · planner replans ${p.plan.attempts.length - 1}` : ''}${typeof p.composerReplans === 'number' ? ` · composer replans ${p.composerReplans}` : ''}`,
    };
  });
  const fail = (err: unknown) => onError?.(err instanceof Error ? err.message : String(err));
  const draftOpts = renderOpts.draft ? { odeSteps: 12, narCacheRatio: 0 } : {};

  // Scores per rung (by step); notes are saved on blur.
  const scoresQuery = useQuery({ ...runQueries.scores(datasetId, runId), enabled: !!datasetId && !!runId });
  const scores = Object.fromEntries((scoresQuery.data?.scores ?? []).map(score => [score.step, score]));
  const setScores = (update: Record<number, Yue2RungScore> | ((old: Record<number, Yue2RungScore>) => Record<number, Yue2RungScore>)) => {
    client.setQueryData(runQueries.scores(datasetId, runId).queryKey, old => {
      const previous = Object.fromEntries((old?.scores ?? []).map(score => [score.step, score]));
      return { scores: Object.values(typeof update === 'function' ? update(previous) : update) };
    });
  };
  const scoreSeq = useRef<Record<number, number>>({});
  const [noteDraft, setNoteDraft] = useState<Record<number, string>>({});
  useEffect(() => { setNoteDraft({}); }, [datasetId, runId]);
  // The save can queue behind other requests for seconds, so the click is
  // shown at once (optimistic patch in the handler). Only the newest save for
  // a rung writes back, so an earlier reply cannot undo a later click; a
  // failure reloads the stored scores. `seq` travels as a variable: it must
  // be the value captured at click time when the reply lands.
  const scoreMut = useMutation({
    mutationFn: (v: { step: number; seq: number; patch: { likeness?: number | null; corruption?: number | null; notes?: string } }) =>
      scoreYue2Rung(datasetId, { refineRun: runId, step: v.step, ...v.patch, blind: !!blind, blindLabel: blind ? labelFor(v.step) : '' }),
    onSuccess: (r, v) => {
      if (scoreSeq.current[v.step] === v.seq) setScores(prev => ({ ...prev, [v.step]: r.score }));
      void trainingWritePlan(client, null);
    },
    onError: err => { fail(err); void scoresQuery.refetch(); },
  });
  const score = (step: number, patch: { likeness?: number | null; corruption?: number | null; notes?: string }) => {
    if (!datasetId || !runId) return;
    const seq = (scoreSeq.current[step] ?? 0) + 1;
    scoreSeq.current[step] = seq;
    setScores(prev => ({ ...prev, [step]: { ...prev[step], ...patch } as Yue2RungScore }));
    scoreMut.mutate({ step, patch, seq });
  };

  // The album verdict: one score per run, beside the per-rung ones.
  const albumQuery = useQuery({ ...runQueries.album(datasetId, runId), enabled: !!datasetId && !!runId });
  const album = albumQuery.data?.score ?? null;
  const setAlbum = (value: Yue2AlbumScore | null) => client.setQueryData(runQueries.album(datasetId, runId).queryKey, { score: value });
  const [albumNote, setAlbumNote] = useState<string | undefined>(undefined);
  useEffect(() => { setAlbumNote(undefined); }, [datasetId, runId]);
  const scoreAlbumMut = useMutation({
    mutationFn: (patch: { score?: number | null; instruments?: Yue2TrainedDirection | null; vocals?: Yue2TrainedDirection | null; notes?: string }) =>
      scoreYue2Album(datasetId, { refineRun: runId, ...patch }),
    onSuccess: r => { setAlbum(r.score); void trainingWritePlan(client, null); },
    onError: err => fail(err),
  });
  const scoreAlbum = (patch: { score?: number | null; instruments?: Yue2TrainedDirection | null; vocals?: Yue2TrainedDirection | null; notes?: string }) => {
    if (!datasetId || !runId) return;
    scoreAlbumMut.mutate(patch);
  };

  const renderMut = useMutation({
    mutationFn: (step: number) =>
      renderYue2JointPreviews(datasetId, { run: runId, step, seconds: renderOpts.seconds, takes: renderOpts.takes, arScale: renderOpts.arScale, sampler: yue2SamplerSnapshot(), ...draftOpts }),
    onSuccess: () => void trainingWritePlan(client, null),
    onError: err => fail(err),
  });
  const rendering = renderMut.isPending ? (renderMut.variables ?? null) : null;
  const render = (step: number) => {
    if (!datasetId || !runId) return;
    renderMut.mutate(step);
  };

  // Cleanup modal after a rung is chosen: what else can go, with sizes.
  const [picked, setPicked] = useState('');
  const [cleanup, setCleanup] = useState<{ pick: ReviewPick; plan: Yue2CleanupPlan; blind: boolean; blindLabel: string } | null>(null);
  const [choice, setChoice] = useState<Yue2CleanupChoice>({ caches: true, otherCheckpoints: true, otherRuns: true, resume: true, otherPreviews: true });
  const [cleanupNote, setCleanupNote] = useState('');
  // The ladder is re-read hard (staleTime 0) so a rung added or removed since
  // render cannot be linked; the blind reveal values are captured at click
  // time and travel as variables.
  const selectMut = useMutation({
    mutationFn: async ({ pickRun, dir, step }: { pickRun: string; dir: string; step: number; blind: boolean; blindLabel: string }) => {
      const fresh = await client.query({ ...runQueries.ladder(datasetId, pickRun), staleTime: 0 });
      const pick = pickFromLadder(fresh, step);
      if (pick.checkpointDir !== dir) throw new Error('The selected checkpoint changed. Reload the ladder.');
      return { pick, ...(await selectReviewRung(pick)) };
    },
    onSuccess: (r, v) => {
      void trainingWritePlan(client, null, resourceKey('training-data/dataset', undefined, r.pick.datasetId));
      setPicked(v.dir); onPicked?.(v.dir);
      setCleanup({ pick: r.pick, plan: r.plan, blind: v.blind, blindLabel: v.blindLabel });
    },
    onError: err => fail(err),
  });
  const finishPick = (pickRun: string, dir: string, step: number) => {
    selectMut.mutate({ pickRun, dir, step, blind: !!blind, blindLabel: blind ? labelFor(step) : '' });
  };
  useImperativeHandle(ref, () => ({ finishPick }), [datasetId]);
  const [usingStep, setUsingStep] = useState<number | null>(null);
  const pickRung = async (dir: string, step: number) => {
    if (!runId) return;
    setCleanupNote(''); setUsingStep(step);
    try {
      if (onUse && await onUse(dir, step)) return;
      finishPick(runId, dir, step);
    } catch (err) { fail(err); }
    finally { setUsingStep(null); }
  };
  const cleanupMut = useMutation({
    mutationFn: ({ pick, over }: { pick: ReviewPick; over: Yue2CleanupChoice; blind: boolean; blindLabel: string }) => cleanupReviewRung(pick, over),
    onSuccess: (r, v) => {
      void trainingWritePlan(client, null, resourceKey('training-data/dataset', undefined, v.pick.datasetId));
      if (r.finishError) { fail(r.finishError); return; }
      setCleanupNote(`${t('trainingStudio.refine.cleanupDone', 'Removed {{what}}; about {{size}} freed.', { what: r.done.join(', ') || 'nothing', size: mib(r.freedBytes) })}${v.blind ? ` ${t('trainingStudio.refine.blindReveal', 'Rung {{label}} was step {{step}}.', { label: v.blindLabel, step: v.pick.step })}` : ''}`);
      setCleanup(null);
    },
    onError: err => fail(err),
  });
  const cleaning = cleanupMut.isPending;
  const doCleanup = (over: Yue2CleanupChoice) => {
    if (!cleanup) return;
    cleanupMut.mutate({ pick: cleanup.pick, over, blind: cleanup.blind, blindLabel: cleanup.blindLabel });
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
    // A shared-sheet take's plan is not this rung's, so its flags do not count.
    const fact = review?.facts.find(f => f.step === step);
    const overall = fact?.overall === null || fact?.overall === undefined ? null : { overall: fact.overall, replansPerTake: fact.replansPerTake };
    // Takes whose re-plan loop ran out without a clean plan (plan flags or a
    // broken verdict on every attempt): the checkpoint itself is suspect.
    const unclean = doneTakes.filter(p => p.plan && p.plan.clean === false);
    return { mine, doneTakes, plannerReplans, composerReplans, hasReplanData, flaggedTakes, flagReasons, overall, unclean };
  };
  const bestStep = review?.best?.step;
  const scoreboardRows = ladder.map(c => {
    const stats = rungStats(c.step);
    const sc = scores[c.step];
    return { step: c.step, kl: c.kl, likeness: sc?.likeness ?? null, corruption: sc?.corruption ?? null, overall: stats.overall };
  }).sort((a, b) => {
    if (blind) return labelFor(a.step).localeCompare(labelFor(b.step));
    if (a.overall && b.overall) return b.overall.overall - a.overall.overall || a.step - b.step;
    if (a.overall) return -1;
    if (b.overall) return 1;
    return a.step - b.step;
  });
  const items: CleanupItemRow[] = cleanup ? [
    { key: 'caches', label: t('trainingStudio.refine.cleanCaches', 'Vocal stems and other backends\' caches (MM3, ACE) for this dataset. The YuE2 latents, codes, lead sheets and prepared set are kept: small, and slow to rebuild'), item: cleanup.plan.caches },
    { key: 'otherCheckpoints', label: t('trainingStudio.refine.cleanCheckpoints', 'The other checkpoints in this run'), item: cleanup.plan.otherCheckpoints },
    { key: 'otherRuns', label: t('trainingStudio.refine.cleanRuns', 'All other joint runs for this dataset'), item: cleanup.plan.otherRuns },
    { key: 'resume', label: t('trainingStudio.refine.cleanResume', 'This rung\'s resume file (the adapter files stay)'), item: cleanup.plan.resume },
    { key: 'otherPreviews', label: t('trainingStudio.refine.cleanPreviews', 'Previews from the other rungs'), item: cleanup.plan.otherPreviews },
  ] : [];
  return (
    <>
      {run && datasetName && <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100 break-words">{datasetName}</h3>}
      {cleanup && <LadderCleanupModal idPrefix={idPrefix} items={items} blind={cleanup.blind} choice={choice} onChoice={setChoice}
        cleaning={cleaning} onClose={() => setCleanup(null)} onCleanup={doCleanup}
        title={cleanup.blind
          ? t('trainingStudio.refine.blindCleanupTitle', 'Rung {{label}} is now the adapter. Clean up around it?', { label: cleanup.blindLabel })
          : t('trainingStudio.refine.cleanupTitle', 'Step {{step}} is now the adapter. Clean up around it?', { step: cleanup.pick.step })}
        intro={`${t('trainingStudio.refine.cleanupIntro', 'Everything below is app-generated and can be rebuilt. Source audio, sidecars, captions, labels and this rung\'s adapter files are never touched. Your rung scores are kept.')} ${!cleanup.blind ? t('trainingStudio.refine.cleanupMovesAnyway', 'Either way the adapter\'s run folder moves to yue2-joint-adapters\\refined.') : ''}`} />}
      {cleanupNote && <div className="text-[12px] text-emerald-700 dark:text-emerald-300">{cleanupNote}</div>}
      {labelsPending && <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[12px] text-amber-800 dark:text-amber-200">
        {t('trainingStudio.refine.labelsPending', 'Blind labels for this run are not in yet — scoring opens once they are.')}
      </div>}
      <CheckpointLadder rendering={rendering} onRender={render} picking={usingStep !== null || selectMut.isPending}
        useLabel={t('trainingStudio.refine.use', 'Use this rung')} header={<>
        {jointLadder && <BlindRungsToggle id={`blind-${idPrefix}`} checked={blindRungs} />}
        <AlbumScoreBar album={album} pending={scoreAlbumMut.isPending} onScore={scoreAlbum} note={albumNote} onNote={setAlbumNote} />
        </>} rungs={visibleLadder.map((c): CheckpointLadderRung => {
          const stats = rungStats(c.step);
          const { mine, plannerReplans, composerReplans, hasReplanData, flaggedTakes, flagReasons, overall } = stats;
          return { key: c.step, step: c.step, id: `${idPrefix}-${c.step}`, title: rungName(c.step), takes: takeRows(mine, c.step),
            inUse: picked === c.dir, canRender: true, onUse: () => void pickRung(c.dir, c.step),
            className: c.step === bestStep ? 'border-sky-500/70' : c.rung ? 'border-amber-500/40' : 'border-zinc-300/70 dark:border-white/10',
            badges: <>
              {!blind && c.rung && c.kl !== undefined && <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[10px] font-semibold">{t('trainingStudio.refine.rungBadge', 'rung')}</span>}
              {!blind && !c.rung && <span className="text-[10px] text-zinc-500">{t('trainingStudio.refine.routineSave', 'routine save')}</span>}
              {c.step === bestStep && <span className="px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-700 dark:text-sky-300 text-[10px] font-semibold">{t('trainingStudio.refine.bestBadge', 'best by score')}</span>}
              {!blind && c.kl !== undefined && <span className="font-mono text-zinc-600 dark:text-zinc-300">KL {c.kl.toFixed(2)}{c.frozen ? ' (frozen)' : ''}</span>}
              {!blind && c.recon !== undefined && <span className="font-mono text-zinc-600 dark:text-zinc-300">recon {c.recon.toFixed(3)}</span>}
              {hasReplanData && <span className="font-mono text-zinc-500" title={blind ? undefined : t('trainingStudio.refine.replansInfo', 'planner / composer re-plans across this rung\'s takes; rising counts are a sign of over-training')}>{t('trainingStudio.refine.replans', 'replans {{p}}/{{c}}', { p: plannerReplans, c: composerReplans })}</span>}
              {flaggedTakes.length > 0 && <span className="font-mono text-red-600 dark:text-red-400" title={flagReasons}>{t('trainingStudio.refine.planFlags', 'plan flags {{n}}/{{takes}}', { n: flaggedTakes.length, takes: stats.doneTakes.length })}</span>}
              {overall && <span className="font-mono text-sky-700 dark:text-sky-300" title={t('trainingStudio.refine.overallInfo', 'overall = (likeness + (6 − corruption)) / 2, minus a soft penalty for replan load')}>{t('trainingStudio.refine.overall', 'overall {{n}}', { n: overall.overall.toFixed(2) })}</span>}
            </>,
            children: <>
            <RungScoreRow score={scores[c.step]} note={noteDraft[c.step]} onNote={v => setNoteDraft(prev => ({ ...prev, [c.step]: v }))}
              onScore={patch => score(c.step, patch)} />
            {stats.unclean.length > 0 && <div className="mt-2 rounded-lg border-2 border-red-500 bg-red-500/10 px-3 py-2 text-[12px] text-red-800 dark:text-red-200">
              <div className="font-semibold">{t('trainingStudio.refine.noCleanPlan', 'This checkpoint is not writing good plans: no clean lead sheet in {{n}} tries.', { n: stats.unclean[0].plan!.attempts.length })}</div>
              <div className="mt-0.5">{t('trainingStudio.refine.noCleanPlanDetail', 'The take below was rendered from the least-flagged plan anyway. Flags on it:')} {(stats.unclean[0].score?.flags ?? stats.unclean[0].plan!.attempts[stats.unclean[0].plan!.attempts.length - 1].flags ?? []).join('; ') || t('trainingStudio.refine.noCleanPlanBroken', 'the judge rejected every plan')}</div>
            </div>}
            </>,
          };
        })} />
      {bestStep !== undefined && <LadderScoreboard rows={scoreboardRows} bestStep={bestStep} blind={blind} rungName={rungName} idPrefix={idPrefix}
        formula={t('trainingStudio.refine.scoreboardFormula', 'Overall = (likeness + (6 − corruption)) / 2, minus 0.25 per replan per take and 0.1 per plan flag per take this rung planned, capped at 1 together.')} />}
    </>
  );
});
Yue2LadderReview.displayName = 'Yue2LadderReview';

export default Yue2LadderReview;
