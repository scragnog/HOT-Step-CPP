// ReviewPanel.tsx — "Awaiting review": every YuE2 refinement ladder with
// rung previews, across datasets, with how many rungs still have no score.
// A batch trains, refines and renders overnight; this is the morning list.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ListChecks, RefreshCw } from 'lucide-react';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { usePersistedState } from '../../hooks/usePersistedState';
import { resourceKey } from '../../queries/keys';
import { runQueries, trainingWritePlan } from '../../queries/training-runs/queries';
import { formatBytes } from '../../services/stemStudioApi';
import {
  deleteYue2AitkRun,
  syncWorkerMirror,
  type Yue2ReviewRow
} from '../../services/trainingApi';
import { finishReviewBatch, pickFromLadder, type ReviewPick } from '../../services/trainingReviewApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { Toggle } from '../shared/Toggle';

export function TrainingReviewHeader({ title, description, actions }: {
  title: React.ReactNode; description: React.ReactNode; actions?: React.ReactNode;
}) {
  return <div className="flex items-center gap-2">
    <ListChecks size={16} className="text-amber-500" />
    <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">{title}</span>
    <span className="text-[11px] text-zinc-500">{description}</span>
    <span className="flex-1" />
    {actions}
  </div>;
}

export function TrainingReviewRunRow({ onOpen, children }: { onOpen: () => void; children: React.ReactNode }) {
  return <button type="button" onClick={onOpen} className="w-full text-left rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 hover:bg-amber-500/5 px-4 py-3 flex flex-wrap items-center gap-x-4 gap-y-1">{children}</button>;
}

/** Scored, not-yet-finished, not-already-queued rows eligible for "Finish
 *  scored". A worker-origin row is eligible too: the mirror has already
 *  copied its rungs here, so it finishes like any local run. Exported for a
 *  direct unit test since this logic has no UI to click through in CI. */
export function selectFinishable(rest: readonly Yue2ReviewRow[], queued: ReadonlySet<string | undefined>): Yue2ReviewRow[] {
  return rest.filter(r => r.best && !r.finished && !r.live && r.status !== 'running' && !queued.has(r.refineRun));
}

export function awaitingReview(row: Yue2ReviewRow): boolean {
  return row.unscored > 0 && !row.reviewed && !row.finished;
}

export const ReviewPanel: React.FC = () => {
  const { t } = useTranslation();
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const setRefineLadderRun = useTrainingStore(s => s.setRefineLadderRun);
  const trainingWorker = useTrainingStore(s => s.trainingWorker);
  const client = useQueryClient();
  const reviewQuery = useQuery(runQueries.review());
  const rows = reviewQuery.data?.rows ?? [];
  const [commandError, setError] = useState('');
  const loading = reviewQuery.isFetching;
  const load = () => reviewQuery.refetch();
  const removeMut = useMutation({
    mutationFn: async (target: { datasetId: string; jobId: string }) => {
      await client.cancelQueries({ queryKey: runQueries.review().queryKey });
      await deleteYue2AitkRun(target.datasetId, target.jobId);
      return target;
    },
    onSuccess: async target => {
      await client.cancelQueries({ queryKey: runQueries.review().queryKey });
      client.setQueryData(runQueries.review().queryKey, previous => previous && ({
        ...previous, rows: previous.rows.filter(r => r.datasetId !== target.datasetId || r.refineRun !== target.jobId),
      }));
      await trainingWritePlan(client, null);
    },
    onError: err => setError(err instanceof Error ? err.message : String(err)),
  });
  const remove = (r: Yue2ReviewRow) => {
    if (!window.confirm(t('trainingStudio.review.deleteRunConfirm', 'Delete this run? The run folder, including its checkpoints, previews and MP3 cache, will be removed. Archived logs and any checkpoint saved outside the run folder stay.'))) return;
    setError('');
    removeMut.mutate({ datasetId: r.datasetId, jobId: r.refineRun });
  };
  const mirrorQuery = useQuery({ ...runQueries.mirror(trainingWorker ?? ''), enabled: !!trainingWorker });
  const mirror = mirrorQuery.data ?? null;
  const syncMut = useMutation({
    mutationFn: (worker: string) => syncWorkerMirror(worker),
    onSuccess: () => { setError(''); void trainingWritePlan(client, trainingWorker); },
    onError: err => setError(err instanceof Error ? err.message : String(err)),
  });
  const syncing = syncMut.isPending;
  const syncNow = () => { if (!trainingWorker) return; syncMut.mutate(trainingWorker); };
  // "Reviewing complete" on the Refine tab counts as scored; a finished
  // (linked and cleaned up) ladder has nothing left to review.
  const pending = rows.filter(awaitingReview);
  const rest = rows.filter(r => !r.finished && !awaitingReview(r));
  // Scored ladders can be finished in one go on the server: NAR further
  // training from the best-scored rung, then link + cleanup (yue2BatchRunner).
  // Finish always runs on this machine, so its batch is read from the local
  // base — the store's `yue2Batches` tracks job control on "Train on" worker
  // and would never see this one.
  const batchesQuery = useQuery(runQueries.localBatches());
  const batches = batchesQuery.data ?? [];
  const error = commandError || reviewQuery.error?.message || mirrorQuery.error?.message || batchesQuery.error?.message || '';
  const queued = new Set(batches.filter(b => b.status === 'running' || b.status === 'paused')
    .flatMap(b => b.items.filter(i => i.refineRun && (i.status === 'pending' || i.status === 'running')).map(i => i.refineRun)));
  const finishable = selectFinishable(rest, queued);
  const [finishOpen, setFinishOpen] = useState(false);
  const [approved, setApproved] = useState<Array<{ pick: ReviewPick; name: string; score: number; decoderOnly: boolean; baseMatched: boolean }>>([]);
  const [preparing, setPreparing] = useState(false);
  const [skip, setSkip] = useState<Record<string, boolean>>({});
  const [knee, setKnee] = useState(true);
  const [blindRungs] = usePersistedState('hs-yue2-blind-rungs', true);
  const prepareFinish = async () => {
    if (finishOpen) { setFinishOpen(false); setApproved([]); return; }
    setPreparing(true); setError('');
    try {
      const ladders = await Promise.all(finishable.map(r => client.query({ ...runQueries.ladder(r.datasetId, r.refineRun), staleTime: 0 })));
      setApproved(ladders.map((ladder, index) => {
        if (!ladder.best || ladder.best.overall === null) throw new Error('A ladder has no fully scored rung. Reload Review before finishing.');
        return { pick: pickFromLadder(ladder, ladder.best.step), name: finishable[index].datasetName,
          score: ladder.best.overall, decoderOnly: finishable[index].decoderOnly,
          baseMatched: finishable[index].baseMatched === true };
      }));
      setFinishOpen(true);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setPreparing(false); }
  };
  const finishMut = useMutation({
    mutationFn: async (v: { picks: ReviewPick[]; knee: boolean }) => {
      await finishReviewBatch(v.picks, v.knee);
      return v.picks;
    },
    onSuccess: picks => { setFinishOpen(false); setApproved([]); setSkip({});
      void trainingWritePlan(client, null, ...picks.map(p => resourceKey('training-data/dataset', undefined, p.datasetId))); },
    onError: err => setError(err instanceof Error ? err.message : String(err)),
  });
  const finishing = finishMut.isPending;
  const finish = () => {
    setError('');
    finishMut.mutate({ picks: approved.filter(item => !skip[item.pick.runId]).map(item => item.pick), knee });
  };
  // A base-matched run's ladder lives on the Train page; the earlier recipe's
  // KL ladders still open on the (hidden) Refine tab.
  const open = (r: Yue2ReviewRow) => { setRefineLadderRun(r.refineRun); void openDataset(r.datasetId).then(() => setPhase(r.baseMatched ? 'train' : 'refine')); };
  const Row: React.FC<{ r: Yue2ReviewRow }> = ({ r }) => (
    <div className="flex items-center gap-2">
      <div className="flex-1 min-w-0">
        <TrainingReviewRunRow onOpen={() => open(r)}>
          <span className="font-semibold text-sm text-zinc-800 dark:text-zinc-100 max-[1023px]:min-w-0 min-[1024px]:min-w-[180px]">{r.datasetName}</span>
          {r.origin && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-violet-500/15 text-violet-700 dark:text-violet-300" title={t('trainingStudio.review.originInfo', 'Trained on this worker and mirrored here; previews and scoring run here.') as string}>{r.origin}</span>}
          <span className="text-[11px] text-zinc-500">{new Date(r.createdAt).toLocaleString()}</span>
          <span className="text-[11px] text-zinc-600 dark:text-zinc-300 tabular-nums">{t('trainingStudio.review.rungs', '{{n}} rungs', { n: r.rungs })}{!(r.baseMatched && blindRungs) && r.klMin !== null && r.klMax !== null ? ` · KL ${r.klMin.toFixed(2)}–${r.klMax.toFixed(2)}` : ''} · {t('trainingStudio.review.previews', '{{n}} previews', { n: r.previews })}</span>
          <span className="flex-1" />
          {r.live ? <span className="text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.review.stillRunning', 'still refining')}</span>
            : r.unscored > 0 && !r.reviewed ? <span className="px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[11px] font-semibold">{t('trainingStudio.review.toScore', '{{n}} to score', { n: r.unscored })}</span>
            : <span className="text-[11px] text-emerald-600 dark:text-emerald-400">{r.reviewed && r.unscored > 0 ? t('trainingStudio.review.reviewed', 'reviewed') : t('trainingStudio.review.scored', 'scored')}</span>}
        </TrainingReviewRunRow>
      </div>
      <button type="button" onClick={() => remove(r)} disabled={removeMut.isPending}
        className="shrink-0 px-2 py-1.5 rounded-lg border border-red-500/30 text-xs text-red-600 dark:text-red-400 hover:bg-red-500/10 disabled:opacity-40">
        {t('trainingStudio.review.deleteRun', 'Delete run')}
      </button>
    </div>
  );
  return (
    <div className="flex flex-col gap-4">
      <TrainingReviewHeader title={t('trainingStudio.review.title', 'Awaiting review')}
        description={t('trainingStudio.review.intro', 'Runs with rungs you have not scored yet. Click one to render previews, listen and score.')}
        actions={<button type="button" onClick={() => void load()} disabled={loading} className="p-1.5 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10"><RefreshCw size={13} className={loading ? 'animate-spin' : ''} /></button>} />
      {trainingWorker && <div className="flex items-center gap-2 text-[11px] text-zinc-500">
        <span>
          {!mirror ? t('trainingStudio.review.mirrorNone', 'Mirror from {{worker}}: no pass yet', { worker: trainingWorker })
            : t('trainingStudio.review.mirror', 'Mirror from {{worker}}: {{state}} · last sync {{when}} · {{files}} file(s) pending ({{size}})', {
              worker: trainingWorker,
              state: mirror.running ? t('trainingStudio.review.mirrorRunning', 'syncing') : mirror.online ? t('trainingStudio.review.mirrorOnline', 'online') : t('trainingStudio.review.mirrorOffline', 'offline'),
              when: mirror.lastPassAt ? new Date(mirror.lastPassAt).toLocaleTimeString() : t('trainingStudio.review.mirrorNever', 'never'),
              files: mirror.pendingFiles, size: formatBytes(mirror.pendingBytes),
            })}
        </span>
        {mirror?.lastError && <span className="text-amber-600 dark:text-amber-400 truncate" title={mirror.lastError}>{mirror.lastError}</span>}
        <button type="button" onClick={syncNow} disabled={syncing || mirror?.running}
          className="px-2 py-0.5 rounded border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 disabled:opacity-40">
          {syncing ? t('trainingStudio.review.mirrorSyncing', 'Syncing…') : t('trainingStudio.review.mirrorSync', 'Sync now')}
        </button>
      </div>}
      {error && <div className="text-xs text-red-600 dark:text-red-400">{error}</div>}
      {pending.length === 0 && !loading && <p className="text-xs text-zinc-500">{t('trainingStudio.review.empty', 'Nothing waiting. Runs appear here once they have a saved checkpoint pair.')}</p>}
      <div className="flex flex-col gap-2">{pending.map(r => <Row key={r.refineRun} r={r} />)}</div>
      {finishable.length > 0 && <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-700 dark:text-zinc-300 flex-1">{t('trainingStudio.review.finishIntro', '{{n}} scored ladder(s) ready to finish: link the best-scored rung to the album preset and clean up (all cleanup options, caches included). Ladders of the earlier recipe get NAR further training from that rung first.', { n: finishable.length })}</span>
          <button type="button" onClick={() => void prepareFinish()} disabled={preparing || finishing} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40">
            {preparing ? t('trainingStudio.review.finishPreparing', 'Loading picks…') : t('trainingStudio.review.finishCta', 'Finish scored ({{n}})', { n: finishable.length })}
          </button>
        </div>
        {finishOpen && <>
          {approved.map(({ pick, name, score, decoderOnly, baseMatched }) => <div key={pick.runId} className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <Toggle size="sm" accent="amber" checked={!skip[pick.runId]} onChange={v => setSkip(prev => ({ ...prev, [pick.runId]: !v }))} aria-label={t('trainingStudio.review.finishInclude', 'Include {{name}} in this finish batch', { name }) as string} />
            <span className="font-semibold max-[1023px]:min-w-0 min-[1024px]:min-w-[180px]">{name}</span>
            <span className="text-zinc-500">{baseMatched && blindRungs && pick.blindLabel
              ? t('trainingStudio.review.finishBlindPick', 'Rung {{label}}, overall {{score}}', { label: pick.blindLabel, score: score.toFixed(2) })
              : t('trainingStudio.review.finishPick', 'step {{step}}, overall {{score}}', { step: pick.step, score: score.toFixed(2) })}{decoderOnly || baseMatched ? ` · ${t('trainingStudio.review.finishNoNar', 'linked as it is, no NAR step')}` : ''}</span>
          </div>)}
          <div className="flex items-center justify-end gap-3">
            <Toggle
              size="sm"
              accent="amber"
              checked={knee}
              onChange={setKnee}
              label={t('trainingStudio.review.finishKnee', 'Stop NAR at the plateau')}
              info={t('trainingStudio.review.finishKneeInfo', 'Stop NAR further training once a line fitted through its last 10 checkpoints gains under 0.5%. Off: train to the 250-step budget or the recon target.')}
            />
            <button type="button" disabled={finishing || approved.every(item => skip[item.pick.runId])} onClick={finish}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40">
              {finishing ? t('trainingStudio.review.finishStarting', 'Starting…') : t('trainingStudio.review.finishGo', 'Start')}
            </button>
          </div>
        </>}
      </div>}
      {rest.length > 0 && <details className="mt-2"><summary className="cursor-pointer text-[11px] text-zinc-500">{t('trainingStudio.review.done', 'Scored or finished ({{n}})', { n: rest.length })}</summary>
        <div className="mt-2 flex flex-col gap-2 opacity-80">{rest.map(r => <Row key={r.refineRun} r={r} />)}</div></details>}
    </div>
  );
};

export default ReviewPanel;
