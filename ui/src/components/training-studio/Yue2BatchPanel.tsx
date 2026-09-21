// Yue2BatchPanel.tsx — the server-side YuE2 batch, wherever the user is.
//
// Mounted by TrainingStudio above the phase content like Yue2QueuePanel. It
// polls GET /yue2-batch while a batch is active, and with "follow" on it opens
// the dataset the batch is training so the ordinary training card (chart,
// logs, previews) shows the live run. Rows link to their datasets.

import React, { useEffect } from 'react';
import { Check, CircleDashed, Loader2, ListChecks, Pause, Play, StopCircle, X, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '../../stores/trainingStore';
import type { Yue2BatchItem, Yue2BatchSummary } from '../../services/trainingApi';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const STAGE_LABEL: Record<string, string> = { cache: 'latent cache', codes: 'codes', sheet: 'lead sheets', stems: 'vocal stems', align: 'lyric timing', train: 'joint training' };

const StatusIcon: React.FC<{ status: Yue2BatchItem['status'] }> = ({ status }) => {
  switch (status) {
    case 'running': return <Loader2 size={14} className="animate-spin text-amber-500 flex-shrink-0" />;
    case 'done': return <Check size={14} className="text-emerald-500 flex-shrink-0" />;
    case 'failed': return <XCircle size={14} className="text-red-500 flex-shrink-0" />;
    case 'cancelled': return <X size={14} className="text-zinc-500 flex-shrink-0" />;
    default: return <CircleDashed size={14} className="text-zinc-400 flex-shrink-0" />;
  }
};

const isActive = (b: Yue2BatchSummary) => b.status === 'running' || b.status === 'paused';

export const Yue2BatchPanel: React.FC = () => {
  const { t } = useTranslation();
  const batches = useTrainingStore(s => s.yue2Batches);
  const follow = useTrainingStore(s => s.yue2BatchFollow);
  const setFollow = useTrainingStore(s => s.setYue2BatchFollow);
  const load = useTrainingStore(s => s.loadYue2Batches);
  const pause = useTrainingStore(s => s.pauseYue2Batch);
  const resume = useTrainingStore(s => s.resumeYue2Batch);
  const cancel = useTrainingStore(s => s.cancelYue2Batch);
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);

  const batch = batches.find(isActive) ?? batches[0];
  const active = !!batch && isActive(batch);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(id);
  }, [active, load]);
  // Follow the batch: open whichever dataset it is on so the training card
  // (job adoption + SSE chart) shows the live run.
  const current = batch?.status === 'running' ? batch.currentDatasetId : null;
  useEffect(() => {
    if (!follow || !current || current === selectedDatasetId) return;
    void openDataset(current).then(() => setPhase('train'));
  }, [follow, current, selectedDatasetId, openDataset, setPhase]);

  if (!batch || (!active && Date.now() - (batch.finishedAt ?? 0) > 6 * 3600_000)) return null;

  const done = batch.items.filter(i => i.status === 'done').length;
  const failed = batch.items.filter(i => i.status === 'failed').length;
  const recipe = batch.recipe;
  return (
    <div className={`${CARD} flex flex-col gap-3`}>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <ListChecks size={15} className="text-amber-500 flex-shrink-0" />
          <span className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t('trainingStudio.yue2.batch.title', 'YuE2 batch')} · {batch.status}
          </span>
          <span className="text-[11px] text-zinc-500">{done}/{batch.items.length} done{failed ? ` · ${failed} failed` : ''}</span>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <label className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-400">
            <input type="checkbox" className="accent-amber-500" checked={follow} onChange={e => setFollow(e.target.checked)} />
            {t('trainingStudio.yue2.batch.follow', 'Follow the dataset being trained')}
          </label>
          {batch.status === 'running' && <button type="button" onClick={() => void pause(batch.id)} className="flex items-center gap-1 text-zinc-600 dark:text-zinc-300 hover:underline"><Pause size={12} />{batch.pauseRequested ? t('trainingStudio.yue2.batch.pausing', 'Pausing after this stage…') : t('common.pause', 'Pause')}</button>}
          {batch.status !== 'running' && <button type="button" onClick={() => void resume(batch.id)} className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 hover:underline"><Play size={12} />{t('common.resume', 'Resume')}</button>}
          {active && <button type="button" onClick={() => void cancel(batch.id)} className="flex items-center gap-1 text-red-600 dark:text-red-400 hover:underline"><StopCircle size={12} />{t('common.cancel', 'Cancel')}</button>}
        </div>
      </div>
      <p className="text-[11px] text-zinc-500">
        {t('trainingStudio.yue2.batch.recipe', 'Recipe')}: {String(recipe.optimizer ?? 'adamw')} · rank {String(recipe.rank ?? 32)}/{String(recipe.alpha ?? 32)}
        {recipe.lr !== undefined ? ` · lr ${recipe.lr}` : ''}{recipe.plannerLrScale !== undefined ? ` · planner ×${recipe.plannerLrScale}` : ''}
        {recipe.stopMode === 'kl' ? ` · until AR KL ${recipe.targetKl} (cap ${recipe.steps})` : recipe.stopMode === 'loss' ? ` · until loss ${recipe.targetLoss} (cap ${recipe.steps})` : ` · ${recipe.steps} steps`}
        {batch.lyricTiming ? ' · lyric timing' : ''}
      </p>
      <div className="flex flex-col gap-1.5">
        {batch.items.map(item => (
          <div key={item.datasetId} className="flex flex-col gap-0.5">
            <div className="flex items-center gap-2 min-w-0">
              <StatusIcon status={item.status} />
              <button type="button" className="text-xs text-zinc-800 dark:text-zinc-200 hover:underline truncate text-left"
                onClick={() => void openDataset(item.datasetId).then(() => setPhase('train'))}>{item.name}</button>
              <span className="text-[11px] text-zinc-500 flex-shrink-0">
                {item.status === 'running' && item.currentStage ? (STAGE_LABEL[item.currentStage] ?? item.currentStage) : item.status}
              </span>
            </div>
            {item.error && <div className="pl-[26px] text-[11px] text-red-500 break-words">{item.error}</div>}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Yue2BatchPanel;
