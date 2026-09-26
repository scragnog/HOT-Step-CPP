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
import { Toggle } from '../shared/Toggle';

interface Props { open: boolean; onClose: () => void }

export const Yue2AitkBatchWizard: React.FC<Props> = ({ open, onClose }) => {
  const { t } = useTranslation();
  const datasets = useTrainingStore(s => s.datasets);
  const batches = useTrainingStore(s => s.yue2Batches);
  const setDraft = useTrainingStore(s => s.setYue2BatchDraft);
  const openDataset = useTrainingStore(s => s.openDataset);
  const addToBatch = useTrainingStore(s => s.addToYue2Batch);
  const [addError, setAddError] = useState<string | null>(null);
  const setPhase = useTrainingStore(s => s.setPhase);
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [filter, setFilter] = useState('');
  const [showTrained, setShowTrained] = useState(false);

  // A dataset with a linked YuE2 pair is trained (the grid's "done" chip).
  const isTrained = (d: typeof datasets[number]) => !!(d.assets?.yue2?.arAdapter && d.assets?.yue2?.narAdapter);
  const trainedCount = datasets.filter(isTrained).length;
  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return datasets.filter(d => (showTrained || !isTrained(d))
      && (!q || d.name.toLowerCase().includes(q) || d.customTag.toLowerCase().includes(q) || (d.artistName ?? '').toLowerCase().includes(q)));
  }, [datasets, filter, showTrained]);
  const selected = datasets.filter(d => checked[d.id]);
  const active = batches.find(b => b.status === 'running' || b.status === 'paused');
  const busy = !!active;
  const queued = new Set(active?.items.map(i => i.datasetId) ?? []);
  const toAdd = selected.filter(d => !queued.has(d.id));

  // A running batch takes more datasets on the end of its queue, with its own recipe.
  const add = async () => {
    if (!active) return;
    setAddError(null);
    try { await addToBatch(active.id, toAdd.map(d => d.id)); setChecked({}); onClose(); }
    catch (err) { setAddError(err instanceof Error ? err.message : String(err)); }
  };

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
        {busy && <p className="text-xs text-amber-600 mb-3">{t('trainingStudio.yue2.aitkBatch.busyAdd', 'A batch is already running. Ticked datasets are added to the end of its queue and trained with its recipe.')}</p>}
        <div className="flex items-center gap-2 mb-2">
          <input value={filter} onChange={e => setFilter(e.target.value)} placeholder={t('trainingStudio.batch.filter', 'Filter…')}
            className="flex-1 rounded-lg bg-zinc-100 dark:bg-black/20 p-2 text-xs" />
          <label className="flex items-center gap-1.5 text-[11px] text-zinc-500 whitespace-nowrap">
            <Toggle size="sm" accent="amber" checked={showTrained} onChange={setShowTrained} aria-label={t('trainingStudio.yue2.aitkBatch.showTrained', 'Show trained')} />
            <span title={t('trainingStudio.yue2.aitkBatch.showTrainedInfo', 'Datasets that already have a linked YuE2 adapter pair are hidden so the list shows only what is still to train. Turn this on to retrain one.')}>
              {t('trainingStudio.yue2.aitkBatch.showTrainedCount', 'Show trained ({{count}})', { count: trainedCount })}
            </span>
          </label>
          <button type="button" className="text-[11px] text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
            onClick={() => setChecked(prev => { const next = { ...prev }; const all = visible.every(d => next[d.id]); for (const d of visible) next[d.id] = !all; return next; })}>
            {visible.every(d => checked[d.id]) ? t('trainingStudio.batch.selectNone', 'Select none') : t('trainingStudio.batch.selectAll', 'Select all')}
          </button>
        </div>
        <div className="rounded-lg border border-zinc-200 dark:border-white/10 divide-y divide-zinc-200 dark:divide-white/10 mb-4 max-h-[50vh] overflow-y-auto">
          {visible.map(ds => <label key={ds.id} className="flex items-center gap-2 p-2 text-xs">
            <Toggle size="sm" accent="amber" disabled={queued.has(ds.id)} checked={queued.has(ds.id) || !!checked[ds.id]} onChange={v => setChecked(previous => ({ ...previous, [ds.id]: v }))} aria-label={t('trainingStudio.yue2.aitkBatch.selectDataset', 'Select {{name}}', { name: ds.name })} />
            <span className="flex-1 truncate">{ds.name}</span>
            {queued.has(ds.id) ? <span className="text-amber-600">{t('trainingStudio.yue2.aitkBatch.inBatch', 'in batch')}</span>
              : isTrained(ds) ? <span className="text-emerald-600">{t('trainingStudio.yue2.aitkBatch.trained', 'trained')}</span>
              : ds.customTag && <span className="text-zinc-500">{ds.customTag}</span>}
          </label>)}
          {visible.length === 0 && <div className="p-3 text-xs text-zinc-500">{t('trainingStudio.batch.noDatasets', 'No datasets available.')}</div>}
        </div>
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-2 text-xs">{t('common.close', 'Close')}</button>
          {addError && <span className="self-center text-xs text-red-500">{addError}</span>}
          {busy
            ? <button type="button" onClick={() => void add()} disabled={toAdd.length === 0}
                className="px-3 py-2 rounded-lg bg-amber-500 text-black text-xs font-semibold disabled:opacity-40">
                {t('trainingStudio.yue2.aitkBatch.add', 'Add {{count}} to running batch', { count: toAdd.length })}
              </button>
            : <button type="button" onClick={() => void configure()} disabled={selected.length === 0}
                className="px-3 py-2 rounded-lg bg-amber-500 text-black text-xs font-semibold disabled:opacity-40">
                {t('trainingStudio.yue2.aitkBatch.configure', 'Configure training for {{count}} dataset(s)', { count: selected.length })}
                <ArrowRight size={13} className="inline ml-1" />
              </button>}
        </div>
      </div>
    </div>
  );
};
