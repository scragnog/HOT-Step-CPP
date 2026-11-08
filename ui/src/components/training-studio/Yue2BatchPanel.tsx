// Yue2BatchPanel.tsx — the server-side YuE2 batch, wherever the user is.
//
// Mounted by TrainingStudio above the phase content. It polls GET /yue2-batch
// while a batch is active, and with "follow" on it opens the dataset the batch
// has just moved to, so the ordinary training card (chart, logs, previews)
// shows the live run. Follow fires once per dataset change, never because the
// user opened something else: scoring another dataset's refine previews while
// the batch trains must not drag them back. Rows link to their datasets, and
// the running stage shows its job progress.

import { useQuery } from '@tanstack/react-query';
import { ArrowRight, Check, CircleDashed, ListChecks, Loader2, Pause, Play, StopCircle, X, XCircle } from 'lucide-react';
import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { pollWhile } from '../../queries/contract';
import { trainingQueries } from '../../queries/training-data/queries';
import { useRunJob } from '../../queries/training-runs/hooks';
import { type Yue2BatchItem, type Yue2BatchSummary } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { Toggle } from '../shared/Toggle';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const STAGE_LABEL: Record<string, string> = { captions: 'captions', cache: 'latent cache', codes: 'codes', sheet: 'lead sheets', stems: 'vocal stems', align: 'lyric timing', train: 'joint training', refine: 'refinement', nar: 'NAR further training', finish: 'link + cleanup' };

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
/** The dataset follow last jumped to. Module-level so re-entering the studio
 *  does not count as a change and yank the user off whatever they opened. */
let lastFollowed: string | null = null;

export const Yue2BatchPanel: React.FC = () => {
  const { t } = useTranslation();
  const worker = useTrainingStore(s => s.trainingWorker);
  const batchesQuery = useQuery({ ...trainingQueries.yue2Batches(worker),
    refetchInterval: pollWhile<Yue2BatchSummary[]>(data => !data || data.some(isActive), 3000) });
  const batches = batchesQuery.data ?? [];
  const follow = useTrainingStore(s => s.yue2BatchFollow);
  const setFollow = useTrainingStore(s => s.setYue2BatchFollow);
  const pause = useTrainingStore(s => s.pauseYue2Batch);
  const resume = useTrainingStore(s => s.resumeYue2Batch);
  const cancel = useTrainingStore(s => s.cancelYue2Batch);
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);

  // Only a running or paused batch has anything to act on; a finished one
  // (done, failed, cancelled) is history and its datasets say how it went.
  const batch = batches.find(isActive);
  const runningJobId = batch?.items.find(i => i.status === 'running')?.stages.find(s => s.status === 'running')?.jobId || '';
  const jobQuery = useRunJob(worker, runningJobId);
  const job = jobQuery.data ?? null;
  // Follow the batch: when it moves to a new dataset, open it so the training
  // card (job adoption + SSE chart) shows the live run. Once per move.
  const current = batch?.status === 'running' ? batch.currentDatasetId : null;
  const goToCurrent = () => { if (current) void openDataset(current).then(() => setPhase('train')); };
  useEffect(() => {
    if (!follow || !current || current === lastFollowed) return;
    lastFollowed = current;
    if (current !== selectedDatasetId) goToCurrent();
  }, [follow, current]);

  if (!batch) return null;

  const done = batch.items.filter(i => i.status === 'done').length;
  const failed = batch.items.filter(i => i.status === 'failed').length;
  const recipe = batch.recipe;
  return (
    <div className={`${CARD} flex flex-col gap-3`}>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <ListChecks size={15} className="text-amber-500 flex-shrink-0" />
          <span className="text-sm font-semibold text-zinc-900 dark:text-white">
            {batch.family === 'heartmula' ? t('trainingStudio.heartmula.batchTitle', 'HeartMuLa batch') : batch.family === 'mm3' ? t('trainingStudio.mm3.batchTitle', 'MiniMax-Music3 batch') : t('trainingStudio.yue2.batch.title', 'YuE2 batch')} · {batch.status}
          </span>
          <span className="text-[11px] text-zinc-500">{done}/{batch.items.length} done{failed ? ` · ${failed} failed` : ''}</span>
          {current && current !== selectedDatasetId && (
            <button type="button" onClick={goToCurrent} className="flex items-center gap-1 text-[11px] font-semibold text-amber-600 dark:text-amber-400 hover:underline">
              {t('trainingStudio.yue2.batch.goTo', 'Go to the running training')}<ArrowRight size={12} />
            </button>
          )}
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <Toggle
            size="sm"
            accent="amber"
            checked={follow}
            onChange={setFollow}
            label={t('trainingStudio.yue2.batch.follow', 'Follow the dataset being trained')}
            info={t('trainingStudio.yue2.batch.followInfo', 'When the batch moves on to its next dataset, switches this panel to that dataset automatically so its chart, logs and previews show the run in progress. Off: stays on whatever dataset you have open, even as the batch trains others in the background.')}
          />
          {batch.status === 'running' && <button type="button" onClick={() => void pause(batch.id)} className="flex items-center gap-1 text-zinc-600 dark:text-zinc-300 hover:underline"><Pause size={12} />{batch.pauseRequested ? t('trainingStudio.yue2.batch.pausing', 'Pausing after this stage…') : t('common.pause', 'Pause')}</button>}
          {batch.status !== 'running' && <button type="button" onClick={() => void resume(batch.id)} className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 hover:underline"><Play size={12} />{t('common.resume', 'Resume')}</button>}
          <button type="button" onClick={() => void cancel(batch.id)} className="flex items-center gap-1 text-red-600 dark:text-red-400 hover:underline"><StopCircle size={12} />{t('common.cancel', 'Cancel')}</button>
        </div>
      </div>
      {batch.family === 'mm3' && Object.keys(recipe).length > 0 && <p className="text-[11px] text-zinc-500">
        {t('trainingStudio.yue2.batch.recipe', 'Recipe')}: {String(recipe.optimizer ?? 'prodigy')} · {String(recipe.adapterType ?? 'lora')} rank {String(recipe.rank ?? '?')}/{String(recipe.alpha ?? '?')}
        {recipe.lr !== undefined ? ` · lr ${recipe.lr}` : ''}{recipe.basePrecision ? ` · base ${recipe.basePrecision}` : ''}{recipe.maxFrames !== undefined ? ` · crop ${recipe.maxFrames}` : ''}
        {recipe.stopMode === 'loss' ? ` · until ${recipe.targetLossMetric === 'eval' ? 'held-out' : 'training'} loss ${recipe.targetLoss} (cap ${recipe.steps})` : ` · ${recipe.steps} steps`}
        {recipe.saveEvery ? ` · checkpoint every ${recipe.saveEvery}` : ''}
      </p>}
      {(!batch.family || batch.family === 'yue2') && Object.keys(recipe).length > 0 && <p className="text-[11px] text-zinc-500">
        {t('trainingStudio.yue2.batch.recipe', 'Recipe')}: {String(recipe.optimizer ?? 'adamw')} · rank {String(recipe.rank ?? 32)}/{String(recipe.alpha ?? 32)}
        {recipe.lr !== undefined ? ` · lr ${recipe.lr}` : ''}{recipe.plannerLrScale !== undefined ? ` · planner ×${recipe.plannerLrScale}` : ''}
        {recipe.stopMode === 'kl' ? ` · until AR KL ${recipe.targetKl}${Number(recipe.narExtraSteps) > 0 ? `, then decoder +${recipe.narExtraSteps}` : ''} (cap ${recipe.steps})` : recipe.stopMode === 'loss' ? ` · until loss ${recipe.targetLoss} (cap ${recipe.steps})` : ` · ${recipe.steps} steps`}
        {batch.lyricTiming ? ' · lyric timing' : ''}
      </p>}
      <div className="flex flex-col gap-1.5">
        {batch.items.map(item => (
          <div key={`${item.datasetId}-${item.refineRun ?? ''}`} className="flex flex-col gap-0.5">
            <div className="flex items-center gap-2 min-w-0">
              <StatusIcon status={item.status} />
              <button type="button" className="text-xs text-zinc-800 dark:text-zinc-200 hover:underline truncate text-left"
                onClick={() => void openDataset(item.datasetId).then(() => setPhase(item.refineRun ? 'refine' : 'train'))}>{item.name}{item.refineRun ? ` (finish${item.pickStep !== undefined ? `, step ${item.pickStep}` : ''})` : ''}</button>
              <span className="text-[11px] text-zinc-500 flex-shrink-0">
                {item.status === 'running' && item.currentStage ? (batch.family === 'heartmula'
                  ? item.currentStage === 'prepare' ? t('trainingStudio.heartmula.prepare', 'Prepare HeartMuLa audio') : t('trainingStudio.heartmula.start', 'Start training')
                  : batch.family === 'mm3' && item.currentStage === 'train' ? 'LoRA training'
                  : STAGE_LABEL[item.currentStage] ?? item.currentStage) : item.status}
                {item.status === 'running' && job && job.total > 0 && ` · ${job.done}/${job.total}${job.phase && job.phase !== 'training' && job.phase !== 'train' ? ` (${job.phase})` : ''}`}
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
