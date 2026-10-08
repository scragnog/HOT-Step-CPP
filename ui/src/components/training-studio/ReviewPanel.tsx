// ReviewPanel.tsx — "Awaiting review": every YuE2 refinement ladder with
// rung previews, across datasets, with how many rungs still have no score.
// A batch trains, refines and renders overnight; this is the morning list.
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ListChecks, RefreshCw } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore';
import {
  getWorkerMirror, listYue2BatchesLocal, listYue2Review, syncWorkerMirror,
  type MirrorStatus, type Yue2BatchSummary, type Yue2ReviewRow,
} from '../../services/trainingApi';
import { formatBytes } from '../../services/stemStudioApi';
import { finishReviewBatch, getReviewLadder, pickFromLadder, type ReviewPick } from '../../services/trainingReviewApi';
import { Toggle } from '../shared/Toggle';
import { usePersistedState } from '../../hooks/usePersistedState';

/** Scored, not-yet-finished, not-already-queued rows eligible for "Finish
 *  scored". A worker-origin row is eligible too: the mirror has already
 *  copied its rungs here, so it finishes like any local run. Exported for a
 *  direct unit test since this logic has no UI to click through in CI. */
export function selectFinishable(rest: readonly Yue2ReviewRow[], queued: ReadonlySet<string | undefined>): Yue2ReviewRow[] {
  return rest.filter(r => r.best && !r.finished && !r.live && r.status !== 'running' && !queued.has(r.refineRun));
}

export const ReviewPanel: React.FC = () => {
  const { t } = useTranslation();
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const setRefineLadderRun = useTrainingStore(s => s.setRefineLadderRun);
  const trainingWorker = useTrainingStore(s => s.trainingWorker);
  const [rows, setRows] = useState<Yue2ReviewRow[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const load = async () => {
    setLoading(true); setError('');
    try { setRows((await listYue2Review()).rows); } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setLoading(false); }
  };
  useEffect(() => { void load(); const id = window.setInterval(() => void load(), 30_000); return () => window.clearInterval(id); }, []);
  // Worker runs reach this machine's index through the server's background
  // mirror; this line only shows how far behind it is and can nudge it.
  const [mirror, setMirror] = useState<MirrorStatus | null>(null);
  const [syncing, setSyncing] = useState(false);
  useEffect(() => {
    setMirror(null);
    if (!trainingWorker) return;
    const poll = () => getWorkerMirror(trainingWorker).then(setMirror).catch(() => { /* keep the last status */ });
    void poll(); const id = window.setInterval(poll, 15_000); return () => window.clearInterval(id);
  }, [trainingWorker]);
  const syncNow = async () => {
    if (!trainingWorker) return;
    setSyncing(true);
    try { setMirror(await syncWorkerMirror(trainingWorker)); await load(); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setSyncing(false); }
  };
  // "Reviewing complete" on the Refine tab counts as scored.
  const awaiting = (r: Yue2ReviewRow) => r.unscored > 0 && r.previews > 0 && !r.reviewed;
  const pending = rows.filter(awaiting);
  const rest = rows.filter(r => !awaiting(r));
  // Scored ladders can be finished in one go on the server: NAR further
  // training from the best-scored rung, then link + cleanup (yue2BatchRunner).
  // Finish always runs on this machine, so its batch is read from the local
  // base — the store's `yue2Batches` tracks job control on "Train on" worker
  // and would never see this one.
  const [batches, setBatches] = useState<Yue2BatchSummary[]>([]);
  const loadBatches = async () => { try { setBatches(await listYue2BatchesLocal()); } catch { /* keep the last list */ } };
  useEffect(() => { void loadBatches(); const id = window.setInterval(() => void loadBatches(), 10_000); return () => window.clearInterval(id); }, []);
  const queued = new Set(batches.filter(b => b.status === 'running' || b.status === 'paused')
    .flatMap(b => b.items.filter(i => i.refineRun && (i.status === 'pending' || i.status === 'running')).map(i => i.refineRun)));
  const finishable = selectFinishable(rest, queued);
  const [finishOpen, setFinishOpen] = useState(false);
  const [approved, setApproved] = useState<Array<{ pick: ReviewPick; name: string; score: number; decoderOnly: boolean; baseMatched: boolean }>>([]);
  const [preparing, setPreparing] = useState(false);
  const [skip, setSkip] = useState<Record<string, boolean>>({});
  const [finishing, setFinishing] = useState(false);
  const [knee, setKnee] = useState(true);
  const [blindRungs] = usePersistedState('hs-yue2-blind-rungs', true);
  const prepareFinish = async () => {
    if (finishOpen) { setFinishOpen(false); setApproved([]); return; }
    setPreparing(true); setError('');
    try {
      const ladders = await Promise.all(finishable.map(r => getReviewLadder(r.datasetId, r.refineRun)));
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
  const finish = async () => {
    setFinishing(true); setError('');
    try {
      const picks = approved.filter(item => !skip[item.pick.runId]).map(item => item.pick);
      await finishReviewBatch(picks, knee);
      setFinishOpen(false); setApproved([]); setSkip({});
      await loadBatches();
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setFinishing(false); }
  };
  // A base-matched run's ladder lives on the Train page; the earlier recipe's
  // KL ladders still open on the (hidden) Refine tab.
  const open = (r: Yue2ReviewRow) => { setRefineLadderRun(r.refineRun); void openDataset(r.datasetId).then(() => setPhase(r.baseMatched ? 'train' : 'refine')); };
  const Row: React.FC<{ r: Yue2ReviewRow }> = ({ r }) => (
    <button type="button" onClick={() => open(r)} className="w-full text-left rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 hover:bg-amber-500/5 px-4 py-3 flex flex-wrap items-center gap-x-4 gap-y-1">
      <span className="font-semibold text-sm text-zinc-800 dark:text-zinc-100 min-w-[180px]">{r.datasetName}</span>
      {r.origin && <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-violet-500/15 text-violet-700 dark:text-violet-300" title={t('trainingStudio.review.originInfo', 'Trained on this worker and mirrored here; previews and scoring run here.') as string}>{r.origin}</span>}
      <span className="text-[11px] text-zinc-500">{new Date(r.createdAt).toLocaleString()}</span>
      <span className="text-[11px] text-zinc-600 dark:text-zinc-300 tabular-nums">{t('trainingStudio.review.rungs', '{{n}} rungs', { n: r.rungs })}{!(r.baseMatched && blindRungs) && r.klMin !== null && r.klMax !== null ? ` · KL ${r.klMin.toFixed(2)}–${r.klMax.toFixed(2)}` : ''} · {t('trainingStudio.review.previews', '{{n}} previews', { n: r.previews })}</span>
      <span className="flex-1" />
      {r.live ? <span className="text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.review.stillRunning', 'still refining')}</span>
        : r.unscored > 0 && !r.reviewed ? <span className="px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[11px] font-semibold">{t('trainingStudio.review.toScore', '{{n}} to score', { n: r.unscored })}</span>
        : <span className="text-[11px] text-emerald-600 dark:text-emerald-400">{r.reviewed && r.unscored > 0 ? t('trainingStudio.review.reviewed', 'reviewed') : t('trainingStudio.review.scored', 'scored')}</span>}
    </button>
  );
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <ListChecks size={16} className="text-amber-500" />
        <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">{t('trainingStudio.review.title', 'Awaiting review')}</span>
        <span className="text-[11px] text-zinc-500">{t('trainingStudio.review.intro', 'Runs with previews and rungs you have not scored yet. Click one to listen and score.')}</span>
        <span className="flex-1" />
        <button type="button" onClick={() => void load()} disabled={loading} className="p-1.5 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10"><RefreshCw size={13} className={loading ? 'animate-spin' : ''} /></button>
      </div>
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
        <button type="button" onClick={() => void syncNow()} disabled={syncing || mirror?.running}
          className="px-2 py-0.5 rounded border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 disabled:opacity-40">
          {syncing ? t('trainingStudio.review.mirrorSyncing', 'Syncing…') : t('trainingStudio.review.mirrorSync', 'Sync now')}
        </button>
      </div>}
      {error && <div className="text-xs text-red-600 dark:text-red-400">{error}</div>}
      {pending.length === 0 && !loading && <p className="text-xs text-zinc-500">{t('trainingStudio.review.empty', 'Nothing waiting. Ladders appear here once their rung previews exist.')}</p>}
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
            <span className="font-semibold min-w-[180px]">{name}</span>
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
            <button type="button" disabled={finishing || approved.every(item => skip[item.pick.runId])} onClick={() => void finish()}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40">
              {finishing ? t('trainingStudio.review.finishStarting', 'Starting…') : t('trainingStudio.review.finishGo', 'Start')}
            </button>
          </div>
        </>}
      </div>}
      {rest.length > 0 && <details className="mt-2"><summary className="cursor-pointer text-[11px] text-zinc-500">{t('trainingStudio.review.done', 'Scored or without previews ({{n}})', { n: rest.length })}</summary>
        <div className="mt-2 flex flex-col gap-2 opacity-80">{rest.map(r => <Row key={r.refineRun} r={r} />)}</div></details>}
    </div>
  );
};

export default ReviewPanel;
