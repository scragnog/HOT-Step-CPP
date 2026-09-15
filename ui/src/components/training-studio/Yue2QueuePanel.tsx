// Yue2QueuePanel.tsx — live progress for the bulk YuE2 training queue
//
// Rendered by TrainingStudio ABOVE the phase content rather than inside any
// one phase, because the queue outlives the page the user started it from:
// Yue2BatchTrainWizard opens from the dataset grid, the job it starts belongs
// to a dataset that may never be opened, and the user is free to wander into
// Preprocess or Monitor while it runs. A panel that lived on the Train page
// would vanish exactly when someone went looking for it.
//
// It shows rows, not logs. The RUNNING dataset's live job — the log tail, the
// loss chart, Cancel — is still JobProgress on that dataset's own page, and
// the row here links there by opening the dataset.

import React from 'react';
import { Check, CircleDashed, Loader2, ListChecks, StopCircle, X, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useTrainingStore, type Yue2QueueItem } from '../../stores/trainingStore';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

const STAGE_NAMES: Record<number, [key: string, fallback: string]> = {
  1: ['trainingStudio.yue2.runAllStageName1', 'latent cache'],
  2: ['trainingStudio.yue2.runAllStageName2', 'codes'],
  3: ['trainingStudio.yue2.runAllStageName3', 'vocal stems'],
  4: ['trainingStudio.yue2.runAllStageName4', 'lyric cursor spans'],
  5: ['trainingStudio.yue2.runAllStageName5', 'NAR LoRA training'],
  6: ['trainingStudio.yue2.runAllStageName6', 'AR LoRA training'],
};

const StatusIcon: React.FC<{ status: Yue2QueueItem['status'] }> = ({ status }) => {
  switch (status) {
    case 'running': return <Loader2 size={14} className="animate-spin text-amber-500 flex-shrink-0" />;
    case 'done':    return <Check size={14} className="text-emerald-500 flex-shrink-0" />;
    case 'failed':  return <XCircle size={14} className="text-red-500 flex-shrink-0" />;
    case 'cancelled': return <X size={14} className="text-zinc-500 flex-shrink-0" />;
    default:        return <CircleDashed size={14} className="text-zinc-400 flex-shrink-0" />;
  }
};

export const Yue2QueuePanel: React.FC = () => {
  const { t } = useTranslation();
  const queue = useTrainingStore(s => s.yue2Queue);
  const active = useTrainingStore(s => s.yue2QueueActive);
  const stopping = useTrainingStore(s => s.yue2QueueStopping);
  const stopQueue = useTrainingStore(s => s.stopYue2Queue);
  const clearQueue = useTrainingStore(s => s.clearYue2Queue);
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);

  if (queue.length === 0) return null;

  const done = queue.filter(q => q.status === 'done').length;
  const failed = queue.filter(q => q.status === 'failed').length;

  return (
    <div className={`${CARD} flex flex-col gap-3`}>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <ListChecks size={15} className="text-amber-500 flex-shrink-0" />
          <span className="text-sm font-bold text-zinc-900 dark:text-white">
            {t('trainingStudio.yue2.queue.title', 'YuE2 training queue')}
          </span>
          <span className="text-[11px] text-zinc-500 tabular-nums">
            {t('trainingStudio.yue2.queue.counts', '{{done}} of {{total}} done', { done, total: queue.length })}
            {failed > 0 && ` · ${t('trainingStudio.yue2.queue.failedCount', '{{count}} failed', { count: failed })}`}
          </span>
        </div>
        {active ? (
          <button
            onClick={stopQueue}
            disabled={stopping}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 disabled:opacity-50 transition-colors"
          >
            <StopCircle size={13} />
            {stopping
              ? t('trainingStudio.yue2.queue.stopping', 'Stopping after this dataset…')
              : t('trainingStudio.yue2.queue.stop', 'Stop after this dataset')}
          </button>
        ) : (
          <button
            onClick={clearQueue}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
          >
            <X size={13} /> {t('trainingStudio.yue2.queue.clear', 'Clear')}
          </button>
        )}
      </div>

      <div className="rounded-lg border border-zinc-200 dark:border-white/5 divide-y divide-zinc-200 dark:divide-white/5 max-h-56 overflow-y-auto">
        {queue.map((item, i) => (
          <div key={`${item.datasetId}:${i}`} className="flex flex-col gap-0.5 px-3 py-2">
            <div className="flex items-center gap-2.5 text-xs">
              <StatusIcon status={item.status} />
              <button
                onClick={() => { void openDataset(item.datasetId); setPhase('train'); }}
                className="flex-1 min-w-0 truncate text-left text-zinc-700 dark:text-zinc-300 font-medium hover:text-amber-600 dark:hover:text-amber-400 transition-colors"
                title={t('trainingStudio.yue2.queue.openHint', 'Open this dataset on the Train page') as string}
              >
                {item.name}
              </button>
              {item.trigger && (
                <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20 flex-shrink-0">
                  {item.trigger}
                </span>
              )}
              <span className="text-[11px] text-zinc-500 flex-shrink-0">
                {item.status === 'running' && item.stage
                  ? t('trainingStudio.yue2.queue.stage', 'stage {{n}}/6 · {{name}}',
                      { n: item.stage, name: t(STAGE_NAMES[item.stage][0], STAGE_NAMES[item.stage][1]) })
                  : t(`trainingStudio.yue2.queue.status.${item.status}`, item.status)}
              </span>
            </div>
            {item.error && (
              <div className="pl-[26px] text-[11px] text-red-500 break-words">{item.error}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Yue2QueuePanel;
