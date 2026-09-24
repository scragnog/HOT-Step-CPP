// ReviewPanel.tsx — "Awaiting review": every YuE2 refinement ladder with
// rung previews, across datasets, with how many rungs still have no score.
// A batch trains, refines and renders overnight; this is the morning list.
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ListChecks, RefreshCw } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore';
import { listYue2Review, type Yue2ReviewRow } from '../../services/trainingApi';

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
  const pending = rows.filter(r => r.unscored > 0 && r.previews > 0);
  const rest = rows.filter(r => !(r.unscored > 0 && r.previews > 0));
  const open = (r: Yue2ReviewRow) => { setRefineLadderRun(r.refineRun); void openDataset(r.datasetId).then(() => setPhase('refine')); };
  const Row: React.FC<{ r: Yue2ReviewRow }> = ({ r }) => (
    <button type="button" onClick={() => open(r)} className="w-full text-left rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 hover:bg-amber-500/5 px-4 py-3 flex flex-wrap items-center gap-x-4 gap-y-1">
      <span className="font-semibold text-sm text-zinc-800 dark:text-zinc-100 min-w-[180px]">{r.datasetName}</span>
      <span className="text-[11px] text-zinc-500">{new Date(r.createdAt).toLocaleString()}</span>
      <span className="text-[11px] text-zinc-600 dark:text-zinc-300 tabular-nums">{t('trainingStudio.review.rungs', '{{n}} rungs', { n: r.rungs })}{r.klMin !== null && r.klMax !== null ? ` · KL ${r.klMin.toFixed(2)}–${r.klMax.toFixed(2)}` : ''} · {t('trainingStudio.review.previews', '{{n}} previews', { n: r.previews })}</span>
      <span className="flex-1" />
      {r.live ? <span className="text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.review.stillRunning', 'still refining')}</span>
        : r.unscored > 0 ? <span className="px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[11px] font-semibold">{t('trainingStudio.review.toScore', '{{n}} to score', { n: r.unscored })}</span>
        : <span className="text-[11px] text-emerald-600 dark:text-emerald-400">{t('trainingStudio.review.scored', 'scored')}</span>}
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
      {rest.length > 0 && <details className="mt-2"><summary className="cursor-pointer text-[11px] text-zinc-500">{t('trainingStudio.review.done', 'Scored or without previews ({{n}})', { n: rest.length })}</summary>
        <div className="mt-2 flex flex-col gap-2 opacity-80">{rest.map(r => <Row key={r.refineRun} r={r} />)}</div></details>}
    </div>
  );
};

export default ReviewPanel;
