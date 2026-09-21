// Yue2AitkBatchWizard.tsx — pick the datasets for a server-side YuE2 batch.
//
// Selection only. The recipe is configured on the ordinary training card,
// which shows "applies to N datasets" while a draft exists and starts the
// batch instead of one run. The batch itself lives on the server
// (yue2BatchRunner.ts): it survives reloads and restarts, and Yue2BatchPanel
// shows it wherever the user is.

import React, { useMemo, useState } from 'react';
import { ArrowRight, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '../../stores/trainingStore';

interface Props { open: boolean; onClose: () => void }

export const Yue2AitkBatchWizard: React.FC<Props> = ({ open, onClose }) => {
  const { t } = useTranslation();
  const datasets = useTrainingStore(s => s.datasets);
  const batches = useTrainingStore(s => s.yue2Batches);
  const setDraft = useTrainingStore(s => s.setYue2BatchDraft);
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [filter, setFilter] = useState('');

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return q ? datasets.filter(d => d.name.toLowerCase().includes(q) || d.customTag.toLowerCase().includes(q)) : datasets;
  }, [datasets, filter]);
  const selected = datasets.filter(d => checked[d.id]);
  const busy = batches.some(b => b.status === 'running' || b.status === 'paused');

  const configure = async () => {
    const ids = selected.map(d => d.id);
    setDraft(ids);
    onClose();
    await openDataset(ids[0]);
    setPhase('train');
  };

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl border border-zinc-200 dark:border-white/10 bg-white dark:bg-suno-card p-5 shadow-2xl">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{t('trainingStudio.yue2.aitkBatch.title', 'Train several datasets')}</h3>
          <button type="button" onClick={onClose}><X size={16} /></button>
        </div>
        <p className="text-[11px] text-zinc-500 mb-3">
          {t('trainingStudio.yue2.aitkBatch.selectHelp', 'Tick the datasets, then configure the recipe on the training page exactly as for one dataset. Start there applies it to every dataset in order. The batch runs on the server and survives reloads and restarts.')}
        </p>
        {busy && <p className="text-xs text-amber-600 mb-3">{t('trainingStudio.yue2.aitkBatch.busy', 'A batch is already running. Finish or cancel it before starting another.')}</p>}
        <div className="flex items-center gap-2 mb-2">
          <input value={filter} onChange={e => setFilter(e.target.value)} placeholder={t('trainingStudio.batch.filter', 'Filter…')}
            className="flex-1 rounded-lg bg-zinc-100 dark:bg-black/20 p-2 text-xs" />
          <button type="button" className="text-[11px] text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
            onClick={() => setChecked(prev => { const next = { ...prev }; const all = visible.every(d => next[d.id]); for (const d of visible) next[d.id] = !all; return next; })}>
            {visible.every(d => checked[d.id]) ? t('trainingStudio.batch.selectNone', 'Select none') : t('trainingStudio.batch.selectAll', 'Select all')}
          </button>
        </div>
        <div className="rounded-lg border border-zinc-200 dark:border-white/10 divide-y divide-zinc-200 dark:divide-white/10 mb-4 max-h-[50vh] overflow-y-auto">
          {visible.map(ds => <label key={ds.id} className="flex items-center gap-2 p-2 text-xs">
            <input type="checkbox" checked={!!checked[ds.id]} onChange={e => setChecked(previous => ({ ...previous, [ds.id]: e.target.checked }))} className="accent-amber-500" />
            <span className="flex-1 truncate">{ds.name}</span>
            {ds.customTag && <span className="text-zinc-500">{ds.customTag}</span>}
          </label>)}
          {visible.length === 0 && <div className="p-3 text-xs text-zinc-500">{t('trainingStudio.batch.noDatasets', 'No datasets available.')}</div>}
        </div>
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-2 text-xs">{t('common.close', 'Close')}</button>
          <button type="button" onClick={() => void configure()} disabled={busy || selected.length === 0}
            className="px-3 py-2 rounded-lg bg-amber-500 text-black text-xs font-semibold disabled:opacity-40">
            {t('trainingStudio.yue2.aitkBatch.configure', 'Configure training for {{count}} dataset(s)', { count: selected.length })}
            <ArrowRight size={13} className="inline ml-1" />
          </button>
        </div>
      </div>
    </div>
  );
};
