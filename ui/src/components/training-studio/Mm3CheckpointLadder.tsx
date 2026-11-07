// Mm3CheckpointLadder.tsx — an MM3 run's checkpoint ladder as the same
// listening test YuE2's is (Yue2LadderReview): one rung per saved checkpoint
// with its takes, blind letters, likeness/corruption and notes, the album
// score, a floating scoreboard, "Render more", and "Use this checkpoint", which
// points the dataset's album presets and the global LM adapter at that step and
// opens the cleanup modal. Read from the run folder, so it survives a reload.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useReload } from '../../queries/training-runs/hooks';
import { runQueries } from '../../queries/training-runs/queries';
import { queryManager } from '../../queries/invalidation';
import { getMm3Ladder, renderMm3LadderTake, type Mm3CleanupPlan, type Mm3Ladder, type Yue2CleanupChoice } from '../../services/trainingApi';
import { cleanupMm3Checkpoint, scoreMm3Album, scoreMm3Rung, selectMm3Checkpoint } from '../../services/trainingReviewApi';
import { setBackendParamFor } from '../../stores/globalParamsStore';
import { AlbumScoreBar, BlindRungsToggle, LadderCleanupModal, LadderScoreboard, mib, useBlindRungs, type AlbumPatch, type CleanupItemRow, type RungPatch } from './LadderReview';
import { mm3Overall, Mm3LadderRows } from './Mm3LadderRows';
import { TrainingCheckpointLadder } from './TrainingPreviewPanel';

export const Mm3CheckpointLadder: React.FC<{
  datasetId: string;
  run: string;
  /** Changes when the live run reports a new preview, to re-read the folder. */
  reloadKey?: unknown;
}> = ({ datasetId, run, reloadKey }) => {
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
  const blind = blindRungs && !!ladder && Object.keys(ladder.blindLabels).length > 0;
  const label = (step: number) => ladder?.blindLabels[step] ?? '';
  const rungName = (step: number) => blind
    ? t('trainingStudio.refine.blindRung', 'Rung {{label}}', { label: label(step) || '?' })
    : t('trainingStudio.refine.rungStep', 'Step {{step}}', { step });
  const idPrefix = `mm3-rung-${run}`;

  const renderMut = useMutation({
    mutationFn: (step: number) => renderMm3LadderTake(datasetId, run, step),
    onMutate: () => setError(''),
    onSuccess: refresh,
    onError: fail,
  });

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
  const [choice, setChoice] = useState<Yue2CleanupChoice>({ caches: true, otherCheckpoints: true, resume: true, otherPreviews: true });
  const selectMut = useMutation({
    mutationFn: async (step: number) => {
      const fresh = await getMm3Ladder(datasetId, run);
      if (!fresh.checkpoints.some(c => c.step === step)) throw new Error('That checkpoint is no longer on disk. Reload the ladder.');
      return { dataset: fresh.dataset, ...(await selectMm3Checkpoint(fresh.dataset, run, step)) };
    },
    onMutate: () => { setError(''); setNote(''); },
    onSuccess: (r, step) => {
      setBackendParamFor('minimax-m3', 'mm3LmAdapter', r.ref);
      const name = blind ? rungName(step) : t('trainingStudio.refine.rungStep', 'Step {{step}}', { step });
      setNote(r.presets
        ? t('trainingStudio.mm3.usePresetsRung', '{{name}} is now the LM adapter, and {{count}} album preset(s) use it.', { name, count: r.presets })
        : t('trainingStudio.mm3.useNoPresetRung', '{{name}} is now the LM adapter. No album preset was linked to this dataset to update, so there is nothing to clean up around.', { name }));
      if (r.plan) setCleanup({ dataset: r.dataset, plan: r.plan, blind, blindLabel: blind ? label(step) : '' });
      refresh();
    },
    onError: fail,
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
    { key: 'resume', label: t('trainingStudio.mm3.cleanResume', 'This run\'s resume state, so it can no longer be continued (the adapter files stay)'), item: cleanup.plan.resume },
    { key: 'otherPreviews', label: t('trainingStudio.mm3.cleanPreviews', 'Previews from the other checkpoints (the base takes stay)'), item: cleanup.plan.otherPreviews },
  ] : [];
  return <TrainingCheckpointLadder>
    <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.ladderTitle', 'Checkpoint ladder')} · {run}{ladder.running ? ` · ${t('trainingStudio.yue2.method.ladderLive', 'training')}` : ''}</p>
    <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.mm3.ladderHintScored', "Every saved checkpoint is a rung. Its first take uses the run's preview seed and caption, so rungs compare like with like; Render more adds takes on a new random seed. Score each rung as you listen. Use this checkpoint makes that step the LM adapter in the global bar and in this dataset's album presets, then offers to clean up around it; pick another step to move it. Rendering and picking open once the run has finished.")}</p>
    {cleanup && <LadderCleanupModal idPrefix={idPrefix} items={items} blind={cleanup.blind} choice={choice} onChoice={setChoice}
      cleaning={cleanupMut.isPending} onClose={() => setCleanup(null)} onCleanup={over => cleanupMut.mutate(over)}
      title={cleanup.blind
        ? t('trainingStudio.refine.blindCleanupTitle', 'Rung {{label}} is now the adapter. Clean up around it?', { label: cleanup.blindLabel })
        : t('trainingStudio.refine.cleanupTitle', 'Step {{step}} is now the adapter. Clean up around it?', { step: cleanup.plan.step })}
      intro={t('trainingStudio.mm3.cleanupIntro', 'Everything below is app-generated and can be rebuilt. Source audio, captions, labels, other runs and this checkpoint\'s adapter files are never touched. Your rung scores are kept.')} />}
    {note && <div className="mt-2 text-[12px] text-emerald-700 dark:text-emerald-300">{note}</div>}
    {error && <div className="mt-2 text-[12px] text-rose-500">{error}</div>}
    <Mm3LadderRows ladder={ladder} rendering={renderMut.isPending ? renderMut.variables ?? null : null}
      picking={selectMut.isPending} onRender={step => renderMut.mutate(step)} onUse={step => selectMut.mutate(step)}
      review={{ blind, scores, notes: noteDraft, idPrefix, onScore: score, onNote: (step, v) => setNoteDraft(prev => ({ ...prev, [step]: v })) }}
      header={ladder.checkpoints.length > 0 && <>
        <BlindRungsToggle id={`blind-${idPrefix}`} checked={blindRungs} />
        <AlbumScoreBar album={ladder.album} pending={albumMut.isPending} onScore={patch => albumMut.mutate(patch)} note={albumNote} onNote={setAlbumNote} />
      </>} />
    {best !== undefined && <LadderScoreboard rows={scoreboardRows} bestStep={best} blind={blind} rungName={rungName} idPrefix={idPrefix}
      formula={t('trainingStudio.mm3.scoreboardFormula', 'Overall = (likeness + (6 − corruption)) / 2.')} />}
  </TrainingCheckpointLadder>;
};

export default Mm3CheckpointLadder;
