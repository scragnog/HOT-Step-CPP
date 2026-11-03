// useHeartmulaController.ts — the one state owner behind the HeartMuLa cards.
//
// The cards (Heartmula{Prepare,Train,Preview}Card) are pure: this hook reads
// the status route, starts and cancels jobs through the training store, and
// hands them props. Checkpoint previews are read from the run being shown; the
// server renders them between trainer segments, a retry is a mutation here.
// The keys live with the resource (`queries/training-data/queries.ts`) and
// every invalidation goes through `queryManager` (P1–P4, docs/dev/tanstack-query.md).

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { HeartmulaPreviewRenderRequest } from '../../../../server/src/contracts/heartmulaTraining';
import { pollWhile } from '../../queries/contract';
import { queryManager } from '../../queries/invalidation';
import { renderHeartmulaPreview, startHeartmulaBatch, cancelYue2Batch, type TrainingJobSummary, type Yue2BatchSummary } from '../../services/trainingApi';
import { trainingQueries } from '../../queries/training-data/queries';
import { useRunJob, useActiveTrainingJob } from '../../queries/training-runs/hooks';
import { useTrainingStore } from '../../stores/trainingStore';
import type { HeartmulaTrainingView } from './plugins/heartmula';
import type { TrainingRenderContext } from './trainingPluginTypes';

const live = (job: TrainingJobSummary | null | undefined) => job?.status === 'queued' || job?.status === 'running';

export function useHeartmulaController({ datasetId }: TrainingRenderContext): HeartmulaTrainingView {
  const { t } = useTranslation();
  const client = useQueryClient();
  const worker = useTrainingStore(s => s.trainingWorker);
  const job = useActiveTrainingJob();
  const storeError = useTrainingStore(s => s.error);
  const startPrepare = useTrainingStore(s => s.startHeartmulaPrepare);
  const startTrain = useTrainingStore(s => s.startHeartmulaTrain);
  const checkpoint = useTrainingStore(s => s.requestHeartmulaCheckpoint);
  const cancel = useTrainingStore(s => s.cancelJob);
  const mine = job?.datasetId === datasetId ? job : null;
  const batchesQuery = useQuery({ ...trainingQueries.yue2Batches(worker),
    refetchInterval: pollWhile<Yue2BatchSummary[]>(data => !data || data.some(b => b.status === 'running' || b.status === 'paused'), 3000) });
  const batch = batchesQuery.data?.find(b => b.family === 'heartmula' && (b.status === 'running' || b.status === 'paused'));
  const batchJobId = batch?.items.find(item => item.datasetId === datasetId)?.stages.find(stage => stage.status === 'running')?.jobId ?? '';
  const batchJob = useRunJob(worker, batchJobId).data;
  useEffect(() => {
    const state = useTrainingStore.getState();
    if (batchJob && state.trainingWorker === worker && state.selectedDatasetId === datasetId
      && state.activeJobId !== batchJob.id) {
      useTrainingStore.setState({ activeJobId: batchJob.id, jobLog: [] });
    }
  }, [batchJob, datasetId, worker, client]);
  const [submitting, setSubmitting] = useState(false);
  const [chainError, setChainError] = useState<{ scope: string; message: string } | null>(null);
  const scope = `${worker ?? ''}:${datasetId}`;
  const busy = live(mine) || !!batch || submitting;
  const query = useQuery(trainingQueries.heartmulaStatus(worker, datasetId, busy));
  const status = query.data ?? null;
  const runId = status?.run?.run ?? null;
  const previewQuery = useQuery({ ...trainingQueries.heartmulaPreviews(worker, datasetId, runId ?? '', busy), enabled: runId !== null });

  // One render/retry command: `isPending` is the card's busy flag, a failure
  // stays on its own run as inline text (so the request is silent — the
  // transport must not toast a second copy), and the run's take list plus the
  // status snapshot are invalidated through the one owner. The variables carry
  // the scope the command ran in, so a late answer cannot report onto another
  // dataset or worker.
  const renderMut = useMutation({
    mutationFn: (v: { request: HeartmulaPreviewRenderRequest; worker: string | null; datasetId: string }) =>
      renderHeartmulaPreview(v.datasetId, v.request, { worker: v.worker, silent: true }),
    // Declare the effect: whether the take rendered or was rejected, the run's
    // take list and the status snapshot are stale (a rejection still records
    // the attempt). queryManager resolves it — the only invalidation owner.
    onSettled: (_list, _err, v) => {
      void queryManager.invalidate(client,
        trainingQueries.heartmulaPreviews(v.worker, v.datasetId, v.request.run, false).queryKey,
        trainingQueries.heartmulaStatus(v.worker, v.datasetId, false).queryKey);
    },
  });

  // The last poll can predate the job's final state: read once more when it ends.
  const wasBusy = useRef(busy);
  useEffect(() => {
    if (wasBusy.current && !busy) void queryManager.invalidate(client, trainingQueries.heartmulaStatus(worker, datasetId, false).queryKey);
    wasBusy.current = busy;
  }, [busy, client, worker, datasetId]);

  const refresh = () => queryManager.invalidate(client, trainingQueries.heartmulaStatus(worker, datasetId, false).queryKey);
  const ready = status?.ready;
  const blocked = ready && !ready.available
    ? ready.backend !== 'cuda'
      ? t('trainingStudio.heartmula.needsCuda', 'HeartMuLa training needs a CUDA engine build.')
      : ready.deviceProblem ? ready.deviceProblem
      : ready.missingModels.length
        ? t('trainingStudio.heartmula.missingModels', 'HeartMuLa models are missing: {{files}}.', { files: ready.missingModels.join(', ') })
        : t('trainingStudio.heartmula.noTrainer', 'ace-train was not found next to the engine. Rebuild the engine.')
    : undefined;
  const prepareBlockedReason = blocked ?? (ready?.encoderMissing ? ready.encoderHelp ?? undefined : undefined);

  return {
    scopeKey: `${worker ?? ''}:${datasetId}`,
    status,
    loading: query.isPending,
    error: query.error?.message ?? (chainError?.scope === scope ? chainError.message : null) ?? storeError,
    busy,
    preparing: busy && mine?.kind === 'heartmula-prepare',
    trainStarted: busy && !batch && mine?.kind === 'heartmula-train' && status?.run?.state === 'running',
    prepareBlockedReason,
    trainBlockedReason: blocked,
    onPrepare: request => { void startPrepare(request).then(refresh); },
    onTrain: request => { void startTrain(request).then(refresh); },
    onChain: (request, datasetIds) => {
      setSubmitting(true); setChainError(null);
      // A batch start is a job-orchestration step (stays imperative, like
      // QueuePanel); it renders its failure inline, so the request is silent.
      void startHeartmulaBatch(datasetIds ?? [datasetId], request, { worker, silent: true })
        .catch(err => setChainError({ scope, message: err instanceof Error ? err.message : String(err) }))
        .finally(() => {
          setSubmitting(false);
          void queryManager.invalidate(client, trainingQueries.yue2Batches(worker).queryKey);
        });
    },
    onCheckpoint: run => { void checkpoint(run); },
    onCancel: () => { void (batch ? cancelYue2Batch(batch.id, { worker }) : cancel()).then(refresh); },
    previews: runId !== null ? previewQuery.data ?? null : null,
    previewError: previewQuery.error?.message ?? null,
    previewCommandError: renderMut.isError && renderMut.variables?.worker === worker && renderMut.variables.datasetId === datasetId
      ? { run: renderMut.variables.request.run, message: renderMut.error instanceof Error ? renderMut.error.message : String(renderMut.error) }
      : null,
    previewBusy: renderMut.isPending,
    onRenderPreview: request => renderMut.mutate({ request, worker, datasetId }),
  };
}
