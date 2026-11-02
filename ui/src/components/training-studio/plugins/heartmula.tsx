import type { HeartmulaStatusResponse } from '../../../../../server/src/contracts/heartmulaTraining';
import { HeartmulaPrepareCard, type HeartmulaPrepareCardProps } from '../HeartmulaPrepareCard';
import { HeartmulaPreviewCard, type HeartmulaPreviewCardProps } from '../HeartmulaPreviewCard';
import { HeartmulaTrainCard, type HeartmulaTrainCardProps } from '../HeartmulaTrainCard';
import type { TrainingPlugin, TrainingRenderContext } from '../trainingPluginTypes';
import { useTranslation } from 'react-i18next';
import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { trainingQueries } from '../../../queries/training-data/queries';
import { queryManager } from '../../../queries/invalidation';
import { useTrainingStore } from '../../../stores/trainingStore';
import { Cpu, Loader2, RefreshCw } from 'lucide-react';
import { TrainingReviewHeader, TrainingReviewRunRow } from '../ReviewPanel';
import { TrainingAllStagesCard } from '../Yue2StageCard';
import { Yue2ClearPreparedCard } from '../Yue2ClearPreparedCard';

/** The integration owns Query snapshots, captured targets, commands and errors.
 * No leaf starts a second reader or job lifecycle beside that owner. */
export interface HeartmulaTrainingView {
  /** Dataset and worker identity; changing it discards the previous form. */
  scopeKey: string;
  status: HeartmulaStatusResponse | null;
  loading: boolean;
  error: string | null;
  busy: boolean;
  preparing: boolean;
  trainStarted: boolean;
  prepareBlockedReason?: string;
  trainBlockedReason?: string;
  onPrepare: HeartmulaPrepareCardProps['onPrepare'];
  onTrain: HeartmulaTrainCardProps['onTrain'];
  onChain?: HeartmulaTrainCardProps['onChain'];
  onCheckpoint: HeartmulaTrainCardProps['onCheckpoint'];
  onCancel: HeartmulaTrainCardProps['onCancel'];
  previews: HeartmulaPreviewCardProps['previews'];
  previewError?: string | null;
  /** A rejected render command, scoped by the controller to this dataset/worker. */
  previewCommandError?: { run: string; message: string } | null;
  /** A manual preview render is in flight (the server refuses a second one). */
  previewBusy?: boolean;
  onRenderPreview?: HeartmulaPreviewCardProps['onRender'];
}

export interface HeartmulaPluginIntegration {
  target: Pick<TrainingPlugin, 'id' | 'phases' | 'reachablePhases' | 'phaseLabels' | 'prepareStages' | 'trainStages' | 'recipeFields' | 'previewSource' | 'chips' | 'readiness'>;
  useController: (context: TrainingRenderContext) => HeartmulaTrainingView;
}

// eslint-disable-next-line react-refresh/only-export-components -- The registry exports a factory; leaf components have their own refresh boundaries.
function HeartmulaPage({ context, useController, review = false }: {
  context: TrainingRenderContext;
  useController: HeartmulaPluginIntegration['useController'];
  review?: boolean;
}) {
  const { t } = useTranslation();
  const view = useController(context);
  const client = useQueryClient();
  const worker = useTrainingStore(s => s.trainingWorker);
  const [selection, setSelection] = useState({ scope: '', run: '' });
  const selectedRun = selection.scope === view.scopeKey && view.status?.runs.some(run => run.run === selection.run)
    ? selection.run : view.status?.run?.run ?? '';
  const olderRun = selectedRun !== (view.status?.run?.run ?? '');
  const previews = useQuery({ ...trainingQueries.heartmulaPreviews(worker, context.datasetId, selectedRun, view.busy), enabled: olderRun && !!selectedRun });
  const previewCard = <HeartmulaPreviewCard runs={view.status?.runs} selectedRun={selectedRun}
    onSelectRun={run => setSelection({ scope: view.scopeKey, run })}
    previews={olderRun ? previews.data ?? null : view.previews} busy={view.busy || !!view.previewBusy}
    error={olderRun ? previews.error?.message : view.previewError}
    commandError={view.previewCommandError?.run === selectedRun ? view.previewCommandError.message : null}
    onRender={view.onRenderPreview} />;
  return <div className="flex flex-col gap-4">
    {review ? <>
      <TrainingReviewHeader title={t('trainingStudio.review.title', 'Awaiting review')}
        description={t('trainingStudio.heartmula.reviewIntro', 'Choose a run to listen to its saved checkpoint previews.')}
        actions={<button type="button" disabled={view.loading} onClick={() => {
          void queryManager.invalidate(client,
            trainingQueries.heartmulaStatus(worker, context.datasetId, false).queryKey,
            trainingQueries.heartmulaPreviews(worker, context.datasetId, selectedRun, false).queryKey);
        }} className="p-1.5 rounded-lg border border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10"><RefreshCw size={13} className={view.loading ? 'animate-spin' : ''} /></button>} />
      <div className="flex flex-col gap-2">{view.status?.runs.map(run => <TrainingReviewRunRow key={run.run}
        onOpen={() => setSelection({ scope: view.scopeKey, run: run.run })}>
        <span className="font-semibold text-sm text-zinc-800 dark:text-zinc-100 max-[1023px]:min-w-0 min-[1024px]:min-w-[180px]">{run.adapterName}</span>
        <span className="text-[11px] text-zinc-500">{run.run}</span>
        <span className="text-[11px] text-zinc-600 dark:text-zinc-300 tabular-nums">{t('trainingStudio.heartmula.step', 'Step {{step}}', { step: run.step })}</span>
        <span className="flex-1" /><span className="text-[11px] text-emerald-600 dark:text-emerald-400">{t(`trainingStudio.heartmula.runState.${run.state}`, { defaultValue: run.state })}</span>
      </TrainingReviewRunRow>)}</div>
    </> : <div className="flex items-start gap-3">
      <Cpu size={18} className="text-amber-500 mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <h2 className="text-sm font-bold text-zinc-900 dark:text-white">{context.section === 'prepare' ? t('trainingStudio.phase.prepare', 'Prepare') : t('trainingStudio.train.title')}</h2>
        <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{context.section === 'prepare' ? t('trainingStudio.heartmula.prepareSubtitle', 'Turn the dataset audio into HeartCodec codes using its existing tags and lyrics.')
            : t('trainingStudio.train.subtitle')}</p>
      </div>
    </div>}
    {view.error && <p role="alert" className="text-xs text-red-500">{view.error}</p>}
    {view.loading && !view.status && <div role="status" className="flex items-center justify-center py-20 text-zinc-500 text-sm"><Loader2 size={18} className="animate-spin mr-2" />{t('trainingStudio.heartmula.loading', 'Loading HeartMuLa training status…')}</div>}
    {review ? previewCard
      : view.status && (context.section === 'prepare'
        ? <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <TrainingAllStagesCard description={t('trainingStudio.heartmula.allPrepare', 'Runs the preparation stage below. Training is on the Train page.')}
              disabled={view.busy || !!view.prepareBlockedReason} onRun={() => view.onPrepare({})} />
            <Yue2ClearPreparedCard key={view.scopeKey} datasetId={context.datasetId} disabled={view.busy} target="heartmula" />
          </div>
          <HeartmulaPrepareCard key={view.scopeKey} datasetId={context.datasetId} status={view.status.prepare} busy={view.busy} preparing={view.preparing}
            blockedReason={view.prepareBlockedReason} onPrepare={view.onPrepare} />
        </>
        : <>
          <HeartmulaTrainCard key={view.scopeKey} datasetId={context.datasetId} status={view.status} slug={context.slug} busy={view.busy}
            previews={view.previews}
            trainStarted={view.trainStarted} blockedReason={view.trainBlockedReason}
            onTrain={view.onTrain} onChain={view.onChain} onCheckpoint={view.onCheckpoint} onCancel={view.onCancel} />
          {/* Previews land under the run as they render, as on the YuE2 Train page. */}
          {view.status.runs.length > 0 && previewCard}
        </>)}
  </div>;
}

/** The integration supplies its registry facts and existing state owner here; TrainPanel hosts the Review page. */
export function createHeartmulaPlugin({ target, useController }: HeartmulaPluginIntegration): TrainingPlugin {
  return {
    ...target,
    renderPrepare: context => <HeartmulaPage context={context} useController={useController} />,
    renderTrain: context => <HeartmulaPage context={context} useController={useController} />,
    renderReview: context => <HeartmulaPage context={context} useController={useController} review />,
  };
}
