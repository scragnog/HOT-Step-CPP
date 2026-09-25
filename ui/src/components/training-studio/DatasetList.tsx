// DatasetList.tsx — dataset cards + "New dataset"
//
// The empty state is the studio's front door: no datasets means a single
// centred card explaining what the Dataset phase does.

import React, { useState } from 'react';
import { Database, Disc3, FolderOpen, FolderPlus, GraduationCap, ListChecks, Loader2, Plus, Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useBackendStore } from '../../stores/backendStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { YUE2_BACKEND_ID } from '../../utils/yue2CaptionSource';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { BatchImportWizard } from './BatchImportWizard';
import { DatasetAssetChips } from './DatasetAssetChips';
import { NewDatasetWizard } from './NewDatasetWizard';
import { Yue2AitkBatchWizard } from './Yue2AitkBatchWizard';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

function formatWhen(iso: string): string {
  if (!iso) return '';
  const d = new Date(iso.includes('T') || iso.includes('Z') ? iso : iso.replace(' ', 'T') + 'Z');
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

const STATUS_STYLE: Record<string, string> = {
  draft:    'text-zinc-500 bg-zinc-500/10 border-zinc-500/20',
  labeling: 'text-sky-500 bg-sky-500/10 border-sky-500/20',
  labeled:  'text-emerald-500 bg-emerald-500/10 border-emerald-500/20',
  built:    'text-amber-500 bg-amber-500/10 border-amber-500/20',
  error:    'text-red-500 bg-red-500/10 border-red-500/20',
};

export const DatasetList: React.FC = () => {
  const { t } = useTranslation();
  const datasets = useTrainingStore(s => s.datasets);
  const loading = useTrainingStore(s => s.loading);
  const openDataset = useTrainingStore(s => s.openDataset);
  const deleteDataset = useTrainingStore(s => s.deleteDataset);

  // "Train multiple…" is YuE2's server-owned batch (yue2BatchRunner.ts:
  // caches, preparation, joint training, refinement per dataset). ACE's bulk
  // path is the pipeline behind "Import multiple…" and MM3 has none, so the
  // button appears only when YuE2 is the active backend.
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const isYue2 = activeBackendId === YUE2_BACKEND_ID;

  const [wizardOpen, setWizardOpen] = useState(false);
  const [batchWizardOpen, setBatchWizardOpen] = useState(false);
  const [yue2BatchOpen, setYue2BatchOpen] = useState(false);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const confirmTarget = datasets.find(d => d.id === confirmId) || null;

  if (loading && datasets.length === 0) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  if (datasets.length === 0) {
    return (
      <>
        <div className={`${CARD} flex flex-col items-center text-center gap-3 py-14`}>
          <GraduationCap size={40} className="text-amber-500/70" />
          <div className="text-lg font-bold text-zinc-900 dark:text-white">{t('trainingStudio.empty.title')}</div>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 max-w-md">{t('trainingStudio.empty.body')}</p>
          <div className="mt-2 flex items-center gap-2">
            <button
              onClick={() => setWizardOpen(true)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-amber-500 text-black hover:bg-amber-400 transition-colors"
            >
              <Plus size={15} /> {t('trainingStudio.empty.cta')}
            </button>
            <button
              onClick={() => setBatchWizardOpen(true)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
            >
              <FolderPlus size={15} /> {t('trainingStudio.batch.cta')}
            </button>
          </div>
        </div>
        <NewDatasetWizard open={wizardOpen} onClose={() => setWizardOpen(false)} />
        <BatchImportWizard open={batchWizardOpen} onClose={() => setBatchWizardOpen(false)} />
      </>
    );
  }

  return (
    <>
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.list.title')}</h2>
        <div className="flex items-center gap-2">
          {isYue2 && (
            <button
              onClick={() => setYue2BatchOpen(true)}
              title={t('trainingStudio.yue2.batch.ctaHint',
                'Train several of these datasets one after another on the server: preparation, joint training and refinement each. Survives reloads and restarts.') as string}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
            >
              <ListChecks size={14} /> {t('trainingStudio.yue2.batch.cta', 'Train multiple…')}
            </button>
          )}
          <button
            onClick={() => setBatchWizardOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
          >
            <FolderPlus size={14} /> {t('trainingStudio.batch.cta')}
          </button>
          <button
            onClick={() => setWizardOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 transition-colors"
          >
            <Plus size={14} /> {t('trainingStudio.list.new')}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">
        {datasets.map(ds => (
          <div
            key={ds.id}
            onClick={() => void openDataset(ds.id)}
            className={`${CARD} group cursor-pointer hover:border-amber-500/40 transition-colors flex flex-col gap-2`}
          >
            <div className="flex items-start gap-2">
              <Database size={16} className="text-amber-500 mt-0.5 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold text-zinc-900 dark:text-white truncate">{ds.name}</div>
                <div className="text-[11px] font-mono text-zinc-500 truncate" title={ds.sourceDir}>{ds.sourceDir}</div>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); setConfirmId(ds.id); }}
                title={t('trainingStudio.list.delete')}
                className="p-1 rounded text-zinc-500 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0"
              >
                <Trash2 size={13} />
              </button>
            </div>

            {ds.albumName && (
              <div
                className="flex items-center gap-1.5 text-[11px] text-zinc-600 dark:text-zinc-400 min-w-0"
                title={t('trainingStudio.list.albumTitle')}
              >
                <Disc3 size={11} className="flex-shrink-0 text-zinc-400 dark:text-zinc-500" />
                <span className="truncate">{ds.albumName}</span>
              </div>
            )}

            <div className="flex items-center gap-2 flex-wrap text-[11px] text-zinc-600 dark:text-zinc-400">
              <span>{t('trainingStudio.list.samples', { count: ds.sampleCount })}</span>
              <span className="text-zinc-400 dark:text-zinc-600">·</span>
              <span>{t('trainingStudio.list.labeled', { done: ds.labeledCount, total: ds.sampleCount })}</span>
            </div>

            <div className="flex items-center gap-2">
              <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border ${STATUS_STYLE[ds.status] || STATUS_STYLE.draft}`}>
                {ds.status}
              </span>
              <span className="text-[11px] text-zinc-500">
                {ds.builtAt ? t('trainingStudio.list.built', { when: formatWhen(ds.builtAt) }) : t('trainingStudio.list.notBuilt')}
              </span>
            </div>

            {/* What the pipeline has actually produced for this dataset. */}
            <DatasetAssetChips
              assets={ds.assets}
              labeledCount={ds.labeledCount}
              sampleCount={ds.sampleCount}
            />

            {ds.customTag && (
              <div className="flex items-center gap-1.5 text-[11px] text-amber-600 dark:text-amber-400">
                <FolderOpen size={11} /> {ds.customTag}
              </div>
            )}
          </div>
        ))}
      </div>

      <NewDatasetWizard open={wizardOpen} onClose={() => setWizardOpen(false)} />
      <BatchImportWizard open={batchWizardOpen} onClose={() => setBatchWizardOpen(false)} />
      <Yue2AitkBatchWizard open={yue2BatchOpen} onClose={() => setYue2BatchOpen(false)} />

      <ConfirmDialog
        isOpen={!!confirmTarget}
        title={t('trainingStudio.list.delete')}
        message={t('trainingStudio.list.deleteConfirm', { name: confirmTarget?.name ?? '' })}
        danger
        onConfirm={() => { const id = confirmId; setConfirmId(null); if (id) void deleteDataset(id); }}
        onCancel={() => setConfirmId(null)}
      />
    </>
  );
};

export default DatasetList;
