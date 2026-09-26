// ReviewPanel.tsx — "Awaiting review": every YuE2 refinement ladder with
// rung previews, across datasets, with how many rungs still have no score.
// A batch trains, refines and renders overnight; this is the morning list.
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ListChecks, RefreshCw } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore';
import { finishYue2Ladders, listYue2Review, type Yue2ReviewRow } from '../../services/trainingApi';
import { Toggle } from '../shared/Toggle';

export const ReviewPanel: React.FC = () => {
  const { t } = useTranslation();
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const setRefineLadderRun = useTrainingStore(s => s.setRefineLadderRun);
  const [rows, setRows] = useState<Yue2ReviewRow[]>([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const load = async () => {
    setLoading(true); setError('');
    try { setRows((await listYue2Review()).rows); } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setLoading(false); }
  };
  useEffect(() => { void load(); const id = window.setInterval(() => void load(), 30_000); return () => window.clearInterval(id); }, []);
  // "Reviewing complete" on the Refine tab counts as scored.
  const awaiting = (r: Yue2ReviewRow) => r.unscored > 0 && r.previews > 0 && !r.reviewed;
  const pending = rows.filter(awaiting);
  const rest = rows.filter(r => !awaiting(r));
  // Scored ladders can be finished in one go on the server: NAR further
  // training from the best-scored rung, then link + cleanup (yue2BatchRunner).
  const batches = useTrainingStore(s => s.yue2Batches);
  const loadBatches = useTrainingStore(s => s.loadYue2Batches);
  const queued = new Set(batches.filter(b => b.status === 'running' || b.status === 'paused')
    .flatMap(b => b.items.filter(i => i.refineRun && (i.status === 'pending' || i.status === 'running')).map(i => i.refineRun)));
  const finishable = rest.filter(r => r.best && !r.live && r.status !== 'running' && !queued.has(r.refineRun));
  const [finishOpen, setFinishOpen] = useState(false);
  const [skip, setSkip] = useState<Record<string, boolean>>({});
  const [finishing, setFinishing] = useState(false);
  const [knee, setKnee] = useState(true);
  const finish = async () => {
    setFinishing(true); setError('');
    try {
      await finishYue2Ladders(finishable.filter(r => !skip[r.refineRun]).map(r => ({ datasetId: r.datasetId, refineRun: r.refineRun })), knee);
      setFinishOpen(false); setSkip({});
      await loadBatches();
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setFinishing(false); }
  };
  const open = (r: Yue2ReviewRow) => { setRefineLadderRun(r.refineRun); void openDataset(r.datasetId).then(() => setPhase('refine')); };
  const Row: React.FC<{ r: Yue2ReviewRow }> = ({ r }) => (
    <button type="button" onClick={() => open(r)} className="w-full text-left rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 hover:bg-amber-500/5 px-4 py-3 flex flex-wrap items-center gap-x-4 gap-y-1">
      <span className="font-semibold text-sm text-zinc-800 dark:text-zinc-100 min-w-[180px]">{r.datasetName}</span>
      <span className="text-[11px] text-zinc-500">{new Date(r.createdAt).toLocaleString()}</span>
      <span className="text-[11px] text-zinc-600 dark:text-zinc-300 tabular-nums">{t('trainingStudio.review.rungs', '{{n}} rungs', { n: r.rungs })}{r.klMin !== null && r.klMax !== null ? ` · KL ${r.klMin.toFixed(2)}–${r.klMax.toFixed(2)}` : ''} · {t('trainingStudio.review.previews', '{{n}} previews', { n: r.previews })}</span>
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
        <span className="text-[11px] text-zinc-500">{t('trainingStudio.review.intro', 'Refinement ladders with previews and rungs you have not scored yet. Click one to listen and score on the Refine tab.')}</span>
        <span className="flex-1" />
        <button type="button" onClick={() => void load()} disabled={loading} className="p-1.5 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10"><RefreshCw size={13} className={loading ? 'animate-spin' : ''} /></button>
      </div>
      {error && <div className="text-xs text-red-600 dark:text-red-400">{error}</div>}
      {pending.length === 0 && !loading && <p className="text-xs text-zinc-500">{t('trainingStudio.review.empty', 'Nothing waiting. Ladders appear here once their rung previews exist.')}</p>}
      <div className="flex flex-col gap-2">{pending.map(r => <Row key={r.refineRun} r={r} />)}</div>
      {finishable.length > 0 && <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-700 dark:text-zinc-300 flex-1">{t('trainingStudio.review.finishIntro', '{{n}} scored ladder(s) ready to finish: NAR further training from the best-scored rung, then link to the album preset and clean up (all cleanup options, caches included).', { n: finishable.length })}</span>
          <button type="button" onClick={() => setFinishOpen(v => !v)} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-500">
            {t('trainingStudio.review.finishCta', 'Finish scored ({{n}})', { n: finishable.length })}
          </button>
        </div>
        {finishOpen && <>
          {finishable.map(r => <div key={r.refineRun} className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <Toggle size="sm" accent="amber" checked={!skip[r.refineRun]} onChange={v => setSkip(prev => ({ ...prev, [r.refineRun]: !v }))} aria-label={t('trainingStudio.review.finishInclude', 'Include {{name}} in this finish batch', { name: r.datasetName }) as string} />
            <span className="font-semibold min-w-[180px]">{r.datasetName}</span>
            <span className="text-zinc-500">{t('trainingStudio.review.finishPick', 'step {{step}}, overall {{score}}', { step: r.best!.step, score: r.best!.overall.toFixed(2) })}{r.decoderOnly ? ` · ${t('trainingStudio.review.finishNoNar', 'decoder run already, no NAR step')}` : ''}</span>
          </div>)}
          <div className="flex items-center justify-end gap-3">
            <Toggle
              size="sm"
              accent="amber"
              checked={knee}
              onChange={setKnee}
              label={t('trainingStudio.review.finishKnee', 'Stop NAR at the plateau')}
              info={t('trainingStudio.review.finishKneeInfo', 'Stop NAR further training once a line fitted through its last 10 checkpoints gains under 0.5%. Off: train to the 500-step budget or the recon target.')}
            />
            <button type="button" disabled={finishing || finishable.every(r => skip[r.refineRun])} onClick={() => void finish()}
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
