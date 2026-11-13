// Mm3CheckpointLadder.tsx — an MM3 run's checkpoint ladder as the same
// listening test YuE2's is (Yue2LadderReview): one rung per saved checkpoint
// with its takes, blind letters, likeness/corruption and notes, the album
// score, a floating scoreboard, "Render more", and "Use this checkpoint", which
// points the dataset's album presets and the global LM adapter at that step and
// opens the cleanup modal. Read from the run folder, so it survives a reload.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Trash2 } from 'lucide-react';
import React, { useEffect, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { useReload } from '../../queries/training-runs/hooks';
import { runQueries, trainingWritePlan } from '../../queries/training-runs/queries';
import { queryManager } from '../../queries/invalidation';
import { deleteMm3Run, getMm3Ladder, mm3RungScoresExportUrl, renderMm3LadderTake, setMm3ReviewComplete,
  type Mm3CleanupPlan, type Mm3Ladder, type Yue2CleanupChoice } from '../../services/trainingApi';
import { cleanupMm3Checkpoint, scoreMm3Album, scoreMm3Rung, selectMm3Checkpoint } from '../../services/trainingReviewApi';
import { setBackendParamFor } from '../../stores/globalParamsStore';
import { AlbumScoreBar, BlindRungsToggle, LadderCleanupModal, LadderScoreboard, mib, useBlindRungs, type AlbumPatch, type CleanupItemRow, type RungPatch } from './LadderReview';
import { mm3Overall, Mm3LadderRows } from './Mm3LadderRows';
import { MM3_DELETE_CONFIRM } from './ReviewPanel';
import { TrainingCheckpointLadder } from './TrainingPreviewPanel';

export const Mm3CheckpointLadder: React.FC<{
  datasetId: string;
  run: string;
  /** Changes when the live run reports a new preview, to re-read the folder. */
  reloadKey?: unknown;
  /** Render with the preview length and MLP scale the form shows now, as
   *  YuE2's Render does; absent = the run's own. */
  renderOpts?: { seconds?: number; scaleMlp?: number };
  /** The run picker, drawn under the heading as YuE2's is. */
  picker?: ReactNode;
}> = ({ datasetId, run, reloadKey, renderOpts, picker }) => {
  const { t } = useTranslation();
  const client = useQueryClient();
  const options = runQueries.mm3Ladder(datasetId, run);
  const query = useQuery({ ...options, enabled: !!datasetId && !!run });
  useReload(options.queryKey, reloadKey);
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const fail = (err: unknown) => setError(err instanceof Error ? err.message : String(err));
  const refresh = () => void queryManager.invalidate(client, options.queryKey);
  const patchLadder = (fn: (l: Mm3Ladder) => Mm3Ladder) => client.setQueryData(options.queryKey, (old: Mm3Ladder | undefined) => old && fn(old));
  const ladder = query.data;
  const blindRungs = useBlindRungs();
  // A checkpoint with no letter yet must not fall back to listing the rungs in
  // their real, revealing step order: no rungs and no scoring until it has one.
  const labelsPending = blindRungs && !!ladder && ladder.checkpoints.some(c => !ladder.blindLabels[c.step]);
  const blind = blindRungs && !!ladder && !labelsPending;
  const label = (step: number) => ladder?.blindLabels[step] ?? '';
  const rungName = (step: number) => blind
    ? t('trainingStudio.refine.blindRung', 'Rung {{label}}', { label: label(step) || '?' })
    : t('trainingStudio.refine.rungStep', 'Step {{step}}', { step });
  const idPrefix = `mm3-rung-${run}`;

  const [renderError, setRenderError] = useState<{ step: number; message: string } | null>(null);
  const [pickError, setPickError] = useState<{ step: number; message: string } | null>(null);
  useEffect(() => { setPickError(null); }, [datasetId, run]);
  const renderMut = useMutation({
    mutationFn: (step: number) => renderMm3LadderTake(datasetId, run, step, renderOpts),
    onMutate: () => { setError(''); setRenderError(null); },
    onSuccess: refresh,
    onError: (err, step) => setRenderError({ step, message: err instanceof Error ? err.message : String(err) }),
  });
  const reviewedMut = useMutation({
    mutationFn: (complete: boolean) => setMm3ReviewComplete(datasetId, run, complete),
    onSuccess: r => { patchLadder(l => ({ ...l, reviewComplete: r.reviewComplete })); void trainingWritePlan(client, null); },
    onError: fail,
  });
  const deleteMut = useMutation({
    mutationFn: () => deleteMm3Run(datasetId, run),
    onSuccess: () => void trainingWritePlan(client, null),
    onError: fail,
  });
  const remove = () => {
    if (!window.confirm(t('trainingStudio.mm3.deleteRunConfirm', MM3_DELETE_CONFIRM))) return;
    setError('');
    deleteMut.mutate();
  };

  // Rung scores: the click shows at once, and only the newest save for a
  // rung writes back, so an earlier reply cannot undo a later click (as YuE2).
  const scoreSeq = useRef<Record<number, number>>({});
  const [noteDraft, setNoteDraft] = useState<Record<number, string>>({});
  const [albumNote, setAlbumNote] = useState<string | undefined>(undefined);
  useEffect(() => { setNoteDraft({}); setAlbumNote(undefined); }, [datasetId, run]);
  const patchScore = (step: number, patch: object) => patchLadder(l => {
    const rest = l.scores.filter(s => s.step !== step);
    const was = l.scores.find(s => s.step === step);
    return { ...l, scores: [...rest, { ...(was ?? { run, step, checkpointDir: '', loss: null, likeness: null, corruption: null, notes: '', blind: false, blindLabel: '', updatedAt: '' }), ...patch }] };
  });
  const scoreMut = useMutation({
    mutationFn: (v: { step: number; seq: number; patch: RungPatch }) =>
      scoreMm3Rung({ datasetId, runName: run, step: v.step, ...v.patch, blind, blindLabel: blind ? label(v.step) : '' }),
    onSuccess: (r, v) => { if (scoreSeq.current[v.step] === v.seq) patchScore(v.step, r.score); },
    onError: err => { fail(err); refresh(); },
  });
  const score = (step: number, patch: RungPatch) => {
    const seq = (scoreSeq.current[step] ?? 0) + 1;
    scoreSeq.current[step] = seq;
    patchScore(step, patch);
    scoreMut.mutate({ step, patch, seq });
  };
  const albumMut = useMutation({
    mutationFn: (patch: AlbumPatch) => scoreMm3Album({ datasetId, runName: run, ...patch }),
    onSuccess: r => patchLadder(l => ({ ...l, album: r.score })),
    onError: fail,
  });

  // "Use this checkpoint", then the cleanup modal while the step is linked.
  // The ladder is re-read first so the dataset revision the command carries
  // is current and a checkpoint deleted since can't be picked.
  const [cleanup, setCleanup] = useState<{ dataset: { id: string; revision: string }; plan: Mm3CleanupPlan; blind: boolean; blindLabel: string } | null>(null);
  const [choice, setChoice] = useState<Yue2CleanupChoice>({ caches: true, otherCheckpoints: true, otherRuns: true, resume: true, otherPreviews: true });
  const selectMut = useMutation({
    mutationFn: async (step: number) => {
      const fresh = await getMm3Ladder(datasetId, run);
      if (!fresh.checkpoints.some(c => c.step === step)) throw new Error('That checkpoint is no longer on disk. Reload the ladder.');
      return { dataset: fresh.dataset, ...(await selectMm3Checkpoint(fresh.dataset, run, step)) };
    },
    onMutate: () => { setError(''); setNote(''); setPickError(null); },
    onSuccess: (r, step) => {
      setBackendParamFor('minimax-m3', 'mm3LmAdapter', r.ref);
      const name = blind ? rungName(step) : t('trainingStudio.refine.rungStep', 'Step {{step}}', { step });
      setNote(r.presets
        ? t('trainingStudio.mm3.usePresetsRung', '{{name}} is now the LM adapter, and {{count}} album preset(s) use it.', { name, count: r.presets })
        : t('trainingStudio.mm3.useDatasetRung', '{{name}} is now this dataset\'s LM adapter.', { name }));
      if (r.plan) setCleanup({ dataset: r.dataset, plan: r.plan, blind, blindLabel: blind ? label(step) : '' });
      refresh();
    },
    onError: (err, step) => setPickError({ step, message: err instanceof Error ? err.message : String(err) }),
  });
  const cleanupMut = useMutation({
    mutationFn: (over: Yue2CleanupChoice) => cleanupMm3Checkpoint(cleanup!.dataset, run, cleanup!.plan.step, over, cleanup!.blindLabel || undefined),
    onSuccess: r => {
      setNote(`${t('trainingStudio.refine.cleanupDone', 'Removed {{what}}; about {{size}} freed.', { what: r.done.join(', ') || 'nothing', size: mib(r.freedBytes) })}${cleanup?.blind ? ` ${t('trainingStudio.refine.blindReveal', 'Rung {{label}} was step {{step}}.', { label: cleanup.blindLabel, step: cleanup.plan.step })}` : ''}`);
      setCleanup(null);
      refresh();
    },
    onError: fail,
  });

  if (deleteMut.isSuccess) return null;
  if (!ladder) return query.error ? <div className="mt-2 text-xs text-rose-500">{query.error.message}</div> : null;
  if (!ladder.checkpoints.length && !ladder.previews.length) return null;
  const { overall, best } = mm3Overall(ladder);
  const scores = Object.fromEntries(ladder.scores.map(s => [s.step, s]));
  const scoreboardRows = ladder.checkpoints.map(c => {
    const o = overall.get(c.step) ?? null;
    return { step: c.step, likeness: scores[c.step]?.likeness ?? null, corruption: scores[c.step]?.corruption ?? null, overall: o === null ? null : { overall: o } };
  }).sort((a, b) => {
    if (blind) return label(a.step).localeCompare(label(b.step));
    if (a.overall && b.overall) return b.overall.overall - a.overall.overall || a.step - b.step;
    if (a.overall) return -1;
    if (b.overall) return 1;
    return a.step - b.step;
  });
  const items: CleanupItemRow[] = cleanup ? [
    { key: 'caches', label: t('trainingStudio.mm3.cleanCaches', 'Vocal stems and other backends\' caches (YuE2, ACE) for this dataset. The MM3 codes and the YuE2 latents are kept: slow to rebuild'), item: cleanup.plan.caches },
    { key: 'otherCheckpoints', label: t('trainingStudio.refine.cleanCheckpoints', 'The other checkpoints in this run'), item: cleanup.plan.otherCheckpoints },
    { key: 'otherRuns', label: t('trainingStudio.mm3.cleanRuns', 'All other MM3 runs for this dataset (not one training now, one already finished, or one an album preset uses)'), item: cleanup.plan.otherRuns },
    { key: 'resume', label: t('trainingStudio.mm3.cleanResume', 'This run\'s resume state, so it can no longer be continued (the adapter files stay)'), item: cleanup.plan.resume },
    { key: 'otherPreviews', label: t('trainingStudio.mm3.cleanPreviews', 'Previews from the other checkpoints (the base takes stay)'), item: cleanup.plan.otherPreviews },
  ] : [];
  // Heading, picker, hint and dataset name in YuE2's order (Yue2AitkTrainCard
  // and Yue2LadderReview); reviewed, delete and the score export as Refine has them.
  return <TrainingCheckpointLadder>
    <div className="flex flex-wrap items-center gap-2">
      <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300 flex items-center gap-1.5">{t('trainingStudio.yue2.method.ladderTitle', 'Checkpoint ladder')} · {new Date(ladder.createdAt).toLocaleString()}{ladder.running ? ` · ${t('trainingStudio.yue2.method.ladderLive', 'training')}` : ''}
        {ladder.origin && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-violet-500/15 text-violet-700 dark:text-violet-300"
          title={t('trainingStudio.review.originInfo', 'Trained on this worker and mirrored here; previews and scoring run here.') as string}>{ladder.origin}</span>}</p>
      <span className="flex-1" />
      {ladder.checkpoints.length > 0 && <button type="button" disabled={reviewedMut.isPending} onClick={() => reviewedMut.mutate(!ladder.reviewComplete)}
        title={t('trainingStudio.refine.reviewCompleteInfo', 'You have found the winner and will not score the other rungs. The Review tab then counts this ladder as scored and offers it to Finish scored, which uses the best-scored rung.')}
        className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border ${ladder.reviewComplete ? 'border-emerald-500 text-emerald-700 dark:text-emerald-300 bg-emerald-500/10' : 'border-zinc-300/70 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-500/10'}`}>
        {ladder.reviewComplete ? t('trainingStudio.refine.reviewCompleteOn', 'Review complete ✓') : t('trainingStudio.refine.reviewComplete', 'Reviewing complete')}
      </button>}
      {!ladder.running && <button type="button" onClick={remove} disabled={deleteMut.isPending}
        title={t('trainingStudio.refine.deleteRun', 'Delete run')} aria-label={t('trainingStudio.refine.deleteRun', 'Delete run') as string}
        className="p-1.5 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:text-red-500 hover:border-red-500/40 disabled:opacity-40"><Trash2 size={13} /></button>}
    </div>
    {picker}
    <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.mm3.ladderHintScored', "Every saved checkpoint is a rung. Its first take uses the run's preview seed and caption, so rungs compare like with like; Render more adds takes on a new random seed. Score each rung as you listen. Use this checkpoint makes that step the LM adapter in the global bar and in this dataset's album presets, then offers to clean up around it; pick another step to move it. Rendering and picking open once the run has finished.")}
      {' '}<a className="underline hover:text-zinc-700 dark:hover:text-zinc-300" href={mm3RungScoresExportUrl('csv')}>{t('trainingStudio.refine.exportCsv', 'Export all scores (CSV)')}</a>
      {' · '}<a className="underline hover:text-zinc-700 dark:hover:text-zinc-300" href={mm3RungScoresExportUrl('json')} target="_blank" rel="noreferrer">JSON</a></p>
    {ladder.datasetName && <h3 className="mt-2 text-sm font-semibold text-zinc-800 dark:text-zinc-100 break-words">{ladder.datasetName}</h3>}
    {cleanup && <LadderCleanupModal idPrefix={idPrefix} items={items} blind={cleanup.blind} choice={choice} onChoice={setChoice}
      cleaning={cleanupMut.isPending} onClose={() => setCleanup(null)} onCleanup={over => cleanupMut.mutate(over)}
      title={cleanup.blind
        ? t('trainingStudio.refine.blindCleanupTitle', 'Rung {{label}} is now the adapter. Clean up around it?', { label: cleanup.blindLabel })
        : t('trainingStudio.refine.cleanupTitle', 'Step {{step}} is now the adapter. Clean up around it?', { step: cleanup.plan.step })}
      intro={t('trainingStudio.mm3.cleanupIntroRuns', 'Source audio, captions, labels and this checkpoint\'s adapter files are never touched, and your rung scores are kept. Everything else below can be rebuilt except the other runs: their checkpoints are gone for good.')} />}
    {note && <div className="mt-2 text-[12px] text-emerald-700 dark:text-emerald-300">{note}</div>}
    {error && <div className="mt-2 text-[12px] text-rose-500">{error}</div>}
    {labelsPending ? <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[12px] text-amber-800 dark:text-amber-200">
      {t('trainingStudio.refine.labelsPending', 'Blind labels for this run are not in yet — scoring opens once they are.')}
    </div> : <Mm3LadderRows ladder={ladder} rendering={renderMut.isPending ? renderMut.variables ?? null : null} renderError={renderError}
      picking={selectMut.isPending} pickError={pickError} onRender={step => renderMut.mutate(step)} onUse={step => selectMut.mutate(step)}
      review={{ blind, scores, notes: noteDraft, idPrefix, onScore: score, onNote: (step, v) => setNoteDraft(prev => ({ ...prev, [step]: v })) }}
      header={ladder.checkpoints.length > 0 && <>
        <BlindRungsToggle id={`blind-${idPrefix}`} checked={blindRungs} />
        <AlbumScoreBar album={ladder.album} pending={albumMut.isPending} onScore={patch => albumMut.mutate(patch)} note={albumNote} onNote={setAlbumNote} />
      </>} />}
    {best !== undefined && !labelsPending && <LadderScoreboard rows={scoreboardRows} bestStep={best} blind={blind} rungName={rungName} idPrefix={idPrefix}
      formula={t('trainingStudio.mm3.scoreboardFormula', 'Overall = (likeness + (6 − corruption)) / 2.')} />}
  </TrainingCheckpointLadder>;
};

export default Mm3CheckpointLadder;
