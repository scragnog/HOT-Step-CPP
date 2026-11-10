// ReviewPanel.tsx — "Awaiting review": every YuE2 refinement ladder with
// rung previews, across datasets, with how many rungs still have no score.
// A batch trains, refines and renders overnight; this is the morning list.
// MM3 mode lists MM3 runs through the same page: the server sends both in one
// list, each row naming its family.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ListChecks, RefreshCw } from 'lucide-react';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { usePersistedState } from '../../hooks/usePersistedState';
import { resourceKey } from '../../queries/keys';
import { runQueries, trainingWritePlan } from '../../queries/training-runs/queries';
import { formatBytes } from '../../services/stemStudioApi';
import {
  deleteMm3Run,
  deleteYue2AitkRun,
  syncWorkerMirror,
  type Yue2ReviewRow
} from '../../services/trainingApi';
import { cleanupMm3Checkpoint, finishReviewBatch, pickFromLadder, selectMm3Checkpoint, type ReviewPick } from '../../services/trainingReviewApi';
import { mm3Overall } from './Mm3LadderRows';
import { useTrainingStore } from '../../stores/trainingStore';
import { ConfirmDialog } from '../shared/ConfirmDialog';
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

/** The worker mirror's last pass, pending files and Sync now, for the
 *  "Train on" worker. Renders nothing when training runs here. */
export const WorkerMirrorStatus: React.FC = () => {
  const { t } = useTranslation();
  const client = useQueryClient();
  const trainingWorker = useTrainingStore(s => s.trainingWorker);
  const mirrorQuery = useQuery({ ...runQueries.mirror(trainingWorker ?? ''), enabled: !!trainingWorker });
  const mirror = mirrorQuery.data ?? null;
  const syncMut = useMutation({
    mutationFn: (worker: string) => syncWorkerMirror(worker),
    onSuccess: () => void trainingWritePlan(client, trainingWorker),
  });
  const syncing = syncMut.isPending;
  if (!trainingWorker) return null;
  const error = syncMut.error?.message || mirrorQuery.error?.message || mirror?.lastError || '';
  return <div className="flex items-center gap-2 text-[11px] text-zinc-500">
    <span>
      {!mirror ? t('trainingStudio.review.mirrorNone', 'Mirror from {{worker}}: no pass yet', { worker: trainingWorker })
        : t('trainingStudio.review.mirror', 'Mirror from {{worker}}: {{state}} · last sync {{when}} · {{files}} file(s) pending ({{size}})', {
          worker: trainingWorker,
          state: mirror.running ? t('trainingStudio.review.mirrorRunning', 'syncing') : mirror.online ? t('trainingStudio.review.mirrorOnline', 'online') : t('trainingStudio.review.mirrorOffline', 'offline'),
          when: mirror.lastPassAt ? new Date(mirror.lastPassAt).toLocaleTimeString() : t('trainingStudio.review.mirrorNever', 'never'),
          files: mirror.pendingFiles, size: formatBytes(mirror.pendingBytes),
        })}
    </span>
    {error && <span className="text-amber-600 dark:text-amber-400 truncate" title={error}>{error}</span>}
    <button type="button" onClick={() => syncMut.mutate(trainingWorker)} disabled={syncing || mirror?.running}
      className="px-2 py-0.5 rounded border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 disabled:opacity-40">
      {syncing ? t('trainingStudio.review.mirrorSyncing', 'Syncing…') : t('trainingStudio.review.mirrorSync', 'Sync now')}
    </button>
  </div>;
};

export const MM3_DELETE_CONFIRM = 'Delete this run? The run folder, including its checkpoints, previews and resume state, will be removed. Your rung scores stay.';

/** One "Finish scored" pick: a YuE2 review pick, or an MM3 run and step. */
type FinishItem = { key: string; name: string; score: number; step: number; blindLabel?: string; noNar: boolean;
  pick?: ReviewPick; mm3?: { dataset: { id: string; revision: string }; run: string } };

export const ReviewPanel: React.FC<{ family?: 'yue2' | 'mm3' }> = ({ family = 'yue2' }) => {
  const { t } = useTranslation();
  const mm3 = family === 'mm3';
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const setRefineLadderRun = useTrainingStore(s => s.setRefineLadderRun);
  const client = useQueryClient();
  const reviewQuery = useQuery(runQueries.review());
  const rows = (reviewQuery.data?.rows ?? []).filter(r => (r.family ?? 'yue2') === family);
  const [commandError, setError] = useState('');
  const loading = reviewQuery.isFetching;
  const load = () => reviewQuery.refetch();
  const removeMut = useMutation({
    mutationFn: async (target: { datasetId: string; jobId: string }) => {
      await client.cancelQueries({ queryKey: runQueries.review().queryKey });
      await (mm3 ? deleteMm3Run(target.datasetId, target.jobId, true) : deleteYue2AitkRun(target.datasetId, target.jobId));
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
  // MM3 deletes everything from here, Lyric Studio links included, so it asks in a modal that names them.
  const [confirmMm3, setConfirmMm3] = useState<Yue2ReviewRow | null>(null);
  const remove = (r: Yue2ReviewRow) => {
    if (mm3) { setConfirmMm3(r); return; }
    if (!window.confirm(t('trainingStudio.review.deleteRunConfirm', 'Delete this run? The run folder, including its checkpoints, previews and MP3 cache, will be removed. Archived logs and any checkpoint saved outside the run folder stay.'))) return;
    setError('');
    removeMut.mutate({ datasetId: r.datasetId, jobId: r.refineRun });
  };
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
  const error = commandError || reviewQuery.error?.message || batchesQuery.error?.message || '';
  const queued = new Set(batches.filter(b => b.status === 'running' || b.status === 'paused')
    .flatMap(b => b.items.filter(i => i.refineRun && (i.status === 'pending' || i.status === 'running')).map(i => i.refineRun)));
  const finishable = selectFinishable(rest, queued);
  const [finishOpen, setFinishOpen] = useState(false);
  const [approved, setApproved] = useState<FinishItem[]>([]);
  const [preparing, setPreparing] = useState(false);
  const [skip, setSkip] = useState<Record<string, boolean>>({});
  const [knee, setKnee] = useState(true);
  const [blindRungs] = usePersistedState('hs-yue2-blind-rungs', true);
  const prepareFinish = async () => {
    if (finishOpen) { setFinishOpen(false); setApproved([]); return; }
    setPreparing(true); setError('');
    try {
      if (mm3) {
        const ladders = await Promise.all(finishable.map(r => client.query({ ...runQueries.mm3Ladder(r.datasetId, r.refineRun), staleTime: 0 })));
        const items = ladders.map((ladder, index): FinishItem => {
          const { overall, best } = mm3Overall(ladder);
          if (best === undefined) throw new Error('A ladder has no fully scored rung. Reload Review before finishing.');
          return { key: ladder.run, name: finishable[index].datasetName, score: overall.get(best)!, step: best,
            blindLabel: ladder.blindLabels[best] || undefined, noNar: true, mm3: { dataset: ladder.dataset, run: ladder.run } };
        });
        // One run per dataset, the best-scored (the first on a tie): finishing
        // deletes the dataset's other runs, so a second pick there would fail.
        const top = new Map<string, FinishItem>();
        for (const item of items) {
          const was = top.get(item.mm3!.dataset.id);
          if (!was || item.score > was.score) top.set(item.mm3!.dataset.id, item);
        }
        setApproved([...top.values()]);
      } else {
        const ladders = await Promise.all(finishable.map(r => client.query({ ...runQueries.ladder(r.datasetId, r.refineRun), staleTime: 0 })));
        setApproved(ladders.map((ladder, index): FinishItem => {
          if (!ladder.best || ladder.best.overall === null) throw new Error('A ladder has no fully scored rung. Reload Review before finishing.');
          const pick = pickFromLadder(ladder, ladder.best.step);
          return { key: pick.runId, pick, name: finishable[index].datasetName, score: ladder.best.overall, step: pick.step,
            blindLabel: finishable[index].baseMatched ? pick.blindLabel : undefined,
            noNar: finishable[index].decoderOnly || finishable[index].baseMatched === true };
        }));
      }
      setFinishOpen(true);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setPreparing(false); }
  };
  const finishMut = useMutation({
    mutationFn: async (v: { items: FinishItem[]; knee: boolean }) => {
      if (!mm3) { await finishReviewBatch(v.items.map(item => item.pick!), v.knee); return v.items; }
      // MM3: link and clean up one album at a time from here, every cleanup
      // option on, as YuE2's finish does. One failure does not stop the rest.
      const failed: string[] = [];
      for (const item of v.items) {
        try {
          const { plan } = await selectMm3Checkpoint(item.mm3!.dataset, item.mm3!.run, item.step);
          if (!plan) throw new Error(t('trainingStudio.mm3.finishNoPreset', 'no album preset uses this dataset, so there was nothing to link'));
          await cleanupMm3Checkpoint(item.mm3!.dataset, item.mm3!.run, item.step,
            { caches: true, otherCheckpoints: true, otherRuns: true, resume: true, otherPreviews: true }, blindRungs ? item.blindLabel : undefined);
        } catch (err) { failed.push(`${item.name}: ${err instanceof Error ? err.message : String(err)}`); }
      }
      if (failed.length) throw new Error(failed.join('; '));
      return v.items;
    },
    onSuccess: items => { setFinishOpen(false); setApproved([]); setSkip({});
      void trainingWritePlan(client, null, ...items.map(item => resourceKey('training-data/dataset', undefined, item.pick?.datasetId ?? item.mm3!.dataset.id))); },
    onError: err => { setError(err instanceof Error ? err.message : String(err)); void trainingWritePlan(client, null); },
  });
  const finishing = finishMut.isPending;
  const finish = () => {
    setError('');
    finishMut.mutate({ items: approved.filter(item => !skip[item.key]), knee });
  };
  // A base-matched run's ladder lives on the Train page; the earlier recipe's
  // KL ladders still open on the (hidden) Refine tab.
  // An MM3 run's ladder is the one under Previous runs on the Train page.
  const open = (r: Yue2ReviewRow) => { setRefineLadderRun(r.refineRun); void openDataset(r.datasetId).then(() => setPhase(r.baseMatched || mm3 ? 'train' : 'refine')); };
  const Row: React.FC<{ r: Yue2ReviewRow }> = ({ r }) => (
    <div className="flex items-center gap-2">
      <div className="flex-1 min-w-0">
        <TrainingReviewRunRow onOpen={() => open(r)}>
          <span className="font-semibold text-sm text-zinc-800 dark:text-zinc-100 max-[1023px]:min-w-0 min-[1024px]:min-w-[180px]">{r.datasetName}</span>
          {r.origin && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-violet-500/15 text-violet-700 dark:text-violet-300" title={t('trainingStudio.review.originInfo', 'Trained on this worker and mirrored here; previews and scoring run here.') as string}>{r.origin}</span>}
          <span className="text-[11px] text-zinc-500">{new Date(r.createdAt).toLocaleString()}</span>
          <span className="text-[11px] text-zinc-600 dark:text-zinc-300 tabular-nums">{t('trainingStudio.review.rungs', '{{n}} rungs', { n: r.rungs })}{!(r.baseMatched && blindRungs) && r.klMin !== null && r.klMax !== null ? ` · KL ${r.klMin.toFixed(2)}–${r.klMax.toFixed(2)}` : ''} · {t('trainingStudio.review.previews', '{{n}} previews', { n: r.previews })}</span>
          <span className="flex-1" />
          {r.live ? <span className="text-[11px] text-amber-600 dark:text-amber-400">{mm3 ? t('trainingStudio.mm3.runsRunning', 'Running now') : t('trainingStudio.review.stillRunning', 'still refining')}</span>
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
      <ConfirmDialog isOpen={!!confirmMm3} danger
        title={t('trainingStudio.mm3.deleteRunTitle', 'Delete run')}
        message={confirmMm3?.linkedAlbums?.length
          ? t('trainingStudio.mm3.deleteLinkedRunConfirm', 'This will unlink the adapter in Lyric Studio from {{albums}}. The run folder (checkpoints, previews and resume state) and its scores will be removed.', { albums: confirmMm3.linkedAlbums.join(', ') })
          : t('trainingStudio.mm3.deleteRunPurgeConfirm', 'Delete this run? The run folder (checkpoints, previews and resume state) and its scores will be removed.')}
        confirmLabel={confirmMm3?.linkedAlbums?.length ? t('trainingStudio.mm3.unlinkAndDelete', 'Unlink and delete') : t('trainingStudio.review.deleteRun', 'Delete run')}
        onConfirm={() => { const r = confirmMm3!; setConfirmMm3(null); setError(''); removeMut.mutate({ datasetId: r.datasetId, jobId: r.refineRun }); }}
        onCancel={() => setConfirmMm3(null)} />
      <TrainingReviewHeader title={t('trainingStudio.review.title', 'Awaiting review')}
        description={t('trainingStudio.review.intro', 'Runs with rungs you have not scored yet. Click one to render previews, listen and score.')}
        actions={<button type="button" onClick={() => void load()} disabled={loading} className="p-1.5 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10"><RefreshCw size={13} className={loading ? 'animate-spin' : ''} /></button>} />
      <WorkerMirrorStatus />
      {error && <div className="text-xs text-red-600 dark:text-red-400">{error}</div>}
      {pending.length === 0 && !loading && <p className="text-xs text-zinc-500">{t('trainingStudio.review.empty', 'Nothing waiting. Runs appear here once they have a saved checkpoint pair.')}</p>}
      <div className="flex flex-col gap-2">{pending.map(r => <Row key={r.refineRun} r={r} />)}</div>
      {finishable.length > 0 && <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-700 dark:text-zinc-300 flex-1">{mm3 ? t('trainingStudio.mm3.finishIntro', "{{n}} scored ladder(s) ready to finish: link the best-scored checkpoint to the album preset and clean up (all cleanup options, caches and the dataset's other runs included). One run per dataset, the best-scored.", { n: finishable.length }) : t('trainingStudio.review.finishIntro', '{{n}} scored ladder(s) ready to finish: link the best-scored rung to the album preset and clean up (all cleanup options, caches included). Ladders of the earlier recipe get NAR further training from that rung first.', { n: finishable.length })}</span>
          <button type="button" onClick={() => void prepareFinish()} disabled={preparing || finishing} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40">
            {preparing ? t('trainingStudio.review.finishPreparing', 'Loading picks…') : t('trainingStudio.review.finishCta', 'Finish scored ({{n}})', { n: finishable.length })}
          </button>
        </div>
        {finishOpen && <>
          {approved.map(({ key, name, score, step, blindLabel, noNar }) => <div key={key} className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <Toggle size="sm" accent="amber" checked={!skip[key]} onChange={v => setSkip(prev => ({ ...prev, [key]: !v }))} aria-label={t('trainingStudio.review.finishInclude', 'Include {{name}} in this finish batch', { name }) as string} />
            <span className="font-semibold max-[1023px]:min-w-0 min-[1024px]:min-w-[180px]">{name}</span>
            <span className="text-zinc-500">{blindRungs && blindLabel
              ? t('trainingStudio.review.finishBlindPick', 'Rung {{label}}, overall {{score}}', { label: blindLabel, score: score.toFixed(2) })
              : t('trainingStudio.review.finishPick', 'step {{step}}, overall {{score}}', { step, score: score.toFixed(2) })}{noNar && !mm3 ? ` · ${t('trainingStudio.review.finishNoNar', 'linked as it is, no NAR step')}` : ''}</span>
          </div>)}
          <div className="flex items-center justify-end gap-3">
            {!mm3 && <Toggle
              size="sm"
              accent="amber"
              checked={knee}
              onChange={setKnee}
              label={t('trainingStudio.review.finishKnee', 'Stop NAR at the plateau')}
              info={t('trainingStudio.review.finishKneeInfo', 'Stop NAR further training once a line fitted through its last 10 checkpoints gains under 0.5%. Off: train to the 250-step budget or the recon target.')}
            />}
            <button type="button" disabled={finishing || approved.every(item => skip[item.key])} onClick={finish}
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
