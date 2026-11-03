// MonitorPanel.tsx — batch pipeline queue view (PhaseStepper's "monitor" tab)
//
// The LIST is polling-only: while a pipeline is running this re-fetches every
// 2s, and on completion it also refreshes the dataset list so newly-created
// datasets appear without a manual reload.
//
// DEVIATION from plan §5.2, which kept the SSE hook per-dataset. The stage
// CHIPS say which stage an item is on and nothing about how far through it is,
// so a multi-hour train-dit inside a bulk run had no visible progress anywhere
// in the app (Rob, 2026-07-31). PipelineStageProgress now points the studio's
// single job projection at the running stage and restores the previous
// selection when it changes. The per-dataset invariant §5.2 was protecting
// still holds: there is never more than one stream opener.

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, CheckCircle2, ChevronDown, ChevronRight, Clock, Loader2, Pause, PauseCircle, Play, XCircle } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { pollWhile } from '../../queries/contract';
import { trainingQueries } from '../../queries/training-data/queries';
import { queryManager } from '../../queries/invalidation';
import { PIPELINE_STAGES, type PipelineItem, type PipelineStage, type PipelineSummary } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { formatDayTime } from '../../utils/dateUtils';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { PipelineStageProgress } from './PipelineStageProgress';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

const STAGE_LABEL_KEYS: Record<PipelineStage, string> = {
  label: 'trainingStudio.batch.stageLabel',
  build: 'trainingStudio.batch.stageBuild',
  preprocess: 'trainingStudio.batch.stagePreprocess',
  'train-dit': 'trainingStudio.batch.stageTrainDit',
  'train-lm': 'trainingStudio.batch.stageTrainLm',
  'lyric-studio': 'trainingStudio.batch.stageLyricStudio',
};

const CHIP_STYLE: Record<string, string> = {
  pending:   'text-zinc-500 bg-zinc-500/10 border-zinc-500/20',
  creating:  'text-sky-500 bg-sky-500/10 border-sky-500/20 animate-pulse',
  running:   'text-amber-500 bg-amber-500/10 border-amber-500/20 animate-pulse',
  done:      'text-emerald-500 bg-emerald-500/10 border-emerald-500/20',
  failed:    'text-rose-500 bg-rose-500/10 border-rose-500/20',
  cancelled: 'text-zinc-500 bg-zinc-500/10 border-zinc-500/20 line-through',
};

function folderBasename(dir: string): string {
  return dir.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? dir;
}

const PipelineItemRow: React.FC<{ item: PipelineItem; stages: PipelineStage[] }> = ({ item, stages }) => {
  const { t } = useTranslation();
  const openDataset = useTrainingStore(s => s.openDataset);
  const setPhase = useTrainingStore(s => s.setPhase);
  const stageByName = new Map(item.stages.map(s => [s.stage, s]));

  return (
    <div className="flex flex-col gap-1.5 py-2">
      <div className="flex items-center gap-2 flex-wrap">
        {item.datasetId ? (
          <button
            onClick={() => { void openDataset(item.datasetId); setPhase('dataset'); }}
            className="text-xs font-semibold text-zinc-800 dark:text-zinc-200 hover:text-amber-500 transition-colors truncate max-w-[240px]"
            title={item.sourceDir}
          >
            {folderBasename(item.sourceDir)} → {item.name}
          </button>
        ) : (
          <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200 truncate max-w-[240px]" title={item.sourceDir}>
            {folderBasename(item.sourceDir)}
          </span>
        )}
        {item.reusedExisting && (
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-sky-500 bg-sky-500/10 border-sky-500/20">
            {t('trainingStudio.monitor.reused')}
          </span>
        )}
        <div className="flex items-center gap-1 flex-wrap">
          {stages.map(stage => {
            const result = stageByName.get(stage);
            const status = result?.status ?? 'pending';
            return (
              <span
                key={stage}
                title={result?.error || undefined}
                className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border ${CHIP_STYLE[status] || CHIP_STYLE.pending}`}
              >
                {t(STAGE_LABEL_KEYS[stage])}
              </span>
            );
          })}
        </div>
      </div>
      {item.error && <div className="text-[11px] text-rose-500 dark:text-rose-400">{item.error}</div>}
    </div>
  );
};

const StatusIcon: React.FC<{ status: PipelineSummary['status'] }> = ({ status }) => {
  if (status === 'running') return <Loader2 size={15} className="animate-spin text-amber-500 flex-shrink-0" />;
  if (status === 'paused') return <PauseCircle size={15} className="text-sky-500 flex-shrink-0" />;
  if (status === 'done') return <CheckCircle2 size={15} className="text-emerald-500 flex-shrink-0" />;
  if (status === 'failed') return <XCircle size={15} className="text-rose-500 flex-shrink-0" />;
  return <AlertTriangle size={15} className="text-amber-500 flex-shrink-0" />;
};

/** Recent (non-active) pipeline: one collapsed title line; click to expand the
 *  per-folder rows. Anything with work left can be resumed — the server keeps
 *  done items done and re-runs the rest, so a cancelled/failed/restart-killed
 *  batch continues from where it stopped. */
const RecentPipelineRow: React.FC<{ pipeline: PipelineSummary; resumeBlocked: boolean }> = ({ pipeline, resumeBlocked }) => {
  const { t } = useTranslation();
  const resumePipeline = useTrainingStore(s => s.resumePipeline);
  const [open, setOpen] = useState(false);
  const [resuming, setResuming] = useState(false);

  const doneItems = pipeline.items.filter(i => i.status === 'done').length;
  const failedItems = pipeline.items.filter(i => i.status === 'failed').length;
  const resumable = pipeline.items.some(i => i.status !== 'done');

  return (
    <div className={`${CARD} flex flex-col gap-2 !p-3`}>
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={() => setOpen(o => !o)}
          className="flex items-center gap-2 flex-1 min-w-0 text-left"
        >
          {open ? <ChevronDown size={13} className="text-zinc-500 flex-shrink-0" /> : <ChevronRight size={13} className="text-zinc-500 flex-shrink-0" />}
          <StatusIcon status={pipeline.status} />
          <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-200 truncate">
            {t('trainingStudio.monitor.progress', { done: doneItems, total: pipeline.items.length })}
            {failedItems > 0 && (
              <span className="text-rose-500 ml-2">{t('trainingStudio.monitor.failedCount', { count: failedItems })}</span>
            )}
          </span>
        </button>
        <span className="text-[11px] text-zinc-500 flex items-center gap-1 flex-shrink-0">
          <Clock size={11} /> {formatDayTime(pipeline.createdAt)}
        </span>
        {resumable && (
          <button
            onClick={() => { setResuming(true); void resumePipeline(pipeline.id).finally(() => setResuming(false)); }}
            disabled={resumeBlocked || resuming}
            title={resumeBlocked ? t('trainingStudio.monitor.resumeBlocked') : t('trainingStudio.monitor.resumeHint')}
            className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-emerald-500/10 border border-emerald-500/20 text-emerald-500 hover:bg-emerald-500/20 transition-colors flex-shrink-0 flex items-center gap-1 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {resuming ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />} {t('trainingStudio.monitor.resume')}
          </button>
        )}
      </div>

      {open && (
        <div className="divide-y divide-zinc-200 dark:divide-white/5">
          {pipeline.items.map(item => (
            <PipelineItemRow key={item.sourceDir} item={item} stages={pipeline.stages.length > 0 ? pipeline.stages : PIPELINE_STAGES.slice()} />
          ))}
        </div>
      )}
    </div>
  );
};

const PipelineCard: React.FC<{ pipeline: PipelineSummary; active: boolean }> = ({ pipeline, active }) => {
  const { t } = useTranslation();
  const cancelPipeline = useTrainingStore(s => s.cancelPipeline);
  const pausePipeline = useTrainingStore(s => s.pausePipeline);
  const resumePipeline = useTrainingStore(s => s.resumePipeline);
  const [confirmCancel, setConfirmCancel] = useState(false);

  const doneItems = pipeline.items.filter(i => i.status === 'done').length;
  const failedItems = pipeline.items.filter(i => i.status === 'failed').length;
  // Requested but not yet parked — the in-flight stage is still finishing.
  const pausePending = pipeline.status === 'running' && pipeline.pauseRequested === true;

  return (
    <div className={`${CARD} flex flex-col gap-3 ${pipeline.status === 'failed' ? 'border-rose-500/30' : ''}`}>
      <div className="flex items-center gap-2 flex-wrap">
        <StatusIcon status={pipeline.status} />
        <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-200 flex-1 min-w-0">
          {t('trainingStudio.monitor.progress', { done: doneItems, total: pipeline.items.length })}
          {failedItems > 0 && (
            <span className="text-rose-500 ml-2">{t('trainingStudio.monitor.failedCount', { count: failedItems })}</span>
          )}
          {pausePending && (
            <span className="text-sky-500 ml-2 font-normal">{t('trainingStudio.monitor.pausing')}</span>
          )}
          {pipeline.status === 'paused' && (
            <span className="text-sky-500 ml-2 font-normal">{t('trainingStudio.monitor.paused')}</span>
          )}
        </div>
        <span className="text-[11px] text-zinc-500 flex items-center gap-1 flex-shrink-0">
          <Clock size={11} /> {formatDayTime(pipeline.createdAt)}
        </span>
        {active && (pausePending || pipeline.status === 'paused' ? (
          <button
            onClick={() => void resumePipeline(pipeline.id)}
            className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-emerald-500/10 border border-emerald-500/20 text-emerald-500 hover:bg-emerald-500/20 transition-colors flex-shrink-0 flex items-center gap-1"
          >
            <Play size={11} /> {t('trainingStudio.monitor.resume')}
          </button>
        ) : (
          <button
            onClick={() => void pausePipeline(pipeline.id)}
            title={t('trainingStudio.monitor.pauseHint')}
            className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-sky-500/10 border border-sky-500/20 text-sky-500 hover:bg-sky-500/20 transition-colors flex-shrink-0 flex items-center gap-1"
          >
            <Pause size={11} /> {t('trainingStudio.monitor.pause')}
          </button>
        ))}
        {active && (
          <button
            onClick={() => setConfirmCancel(true)}
            className="px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-red-500/10 border border-red-500/20 text-red-500 hover:bg-red-500/20 transition-colors flex-shrink-0"
          >
            {t('trainingStudio.monitor.cancel')}
          </button>
        )}
      </div>

      {/* The stage actually executing. The chips below say WHICH stage each item
          is on; this is the only place that shows how far through it is. */}
      <PipelineStageProgress item={pipeline.items.find(i => i.status === 'running') ?? null} />

      <div className="divide-y divide-zinc-200 dark:divide-white/5">
        {pipeline.items.map(item => (
          <PipelineItemRow key={item.sourceDir} item={item} stages={pipeline.stages.length > 0 ? pipeline.stages : PIPELINE_STAGES.slice()} />
        ))}
      </div>

      <ConfirmDialog
        isOpen={confirmCancel}
        title={t('trainingStudio.monitor.cancel')}
        message={t('trainingStudio.monitor.cancelConfirm')}
        danger
        onConfirm={() => { setConfirmCancel(false); void cancelPipeline(pipeline.id); }}
        onCancel={() => setConfirmCancel(false)}
      />
    </div>
  );
};

export const MonitorPanel: React.FC = () => {
  const { t } = useTranslation();
  const client = useQueryClient();
  const worker = useTrainingStore(s => s.trainingWorker);
  const pipelineQuery = useQuery({ ...trainingQueries.pipelines(worker),
    refetchInterval: pollWhile<PipelineSummary[]>(data => !data || data.some(p => p.status === 'running' || p.status === 'paused'), 2000) });
  const pipelines = pipelineQuery.data ?? [];
  const pipelinesLoading = pipelineQuery.isPending;
  const loadDatasets = useTrainingStore(s => s.loadDatasets);
  const [recentOpen, setRecentOpen] = useState(false);

  const activePipeline = pipelines.find(p => p.status === 'running' || p.status === 'paused') ?? null;
  const recent = pipelines.filter(p => p.id !== activePipeline?.id);

  // Poll only while something is running; once it drops out, catch up the
  // dataset list once so freshly-created datasets appear without a reload.
  const wasRunningRef = useRef(false);
  useEffect(() => {
    if (!activePipeline) {
      if (wasRunningRef.current) { wasRunningRef.current = false; void queryManager.invalidate(client, trainingQueries.datasets(worker).queryKey).then(() => loadDatasets()); }
      return;
    }
    wasRunningRef.current = true;
  }, [activePipeline, client, worker, loadDatasets]);

  if (pipelineQuery.error && pipelines.length === 0) return <p role="alert" className="text-sm text-red-500">{pipelineQuery.error.message}</p>;

  if (pipelinesLoading && pipelines.length === 0) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  if (pipelines.length === 0) {
    return (
      <div className={`${CARD} flex flex-col items-center text-center gap-2 py-14`}>
        <div className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.monitor.empty')}</div>
        <p className="text-xs text-zinc-500 max-w-md">{t('trainingStudio.monitor.emptyHint')}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {activePipeline ? (
        <PipelineCard pipeline={activePipeline} active />
      ) : (
        <div className={`${CARD} text-xs text-zinc-500`}>{t('trainingStudio.monitor.noActive')}</div>
      )}

      {recent.length > 0 && (
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setRecentOpen(o => !o)}
            className="flex items-center gap-1.5 text-xs font-semibold text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200 transition-colors self-start"
          >
            {recentOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            {t('trainingStudio.monitor.recent', { count: recent.length })}
          </button>
          {recentOpen && (
            <div className="flex flex-col gap-2">
              {recent.map(p => <RecentPipelineRow key={p.id} pipeline={p} resumeBlocked={!!activePipeline} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MonitorPanel;
