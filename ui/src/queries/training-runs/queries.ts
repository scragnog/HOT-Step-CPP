import { queryOptions, type QueryClient, type QueryKey } from '@tanstack/react-query';
import * as api from '../../services/trainingApi';
import * as preparation from '../../services/trainingPreparationApi';
import { getReviewLadder } from '../../services/trainingReviewApi';
import { trainingOperation } from '../../services/trainingOperations';
import type { TrainingRecipe } from '../../services/trainingRecipesApi';
import type { TrainingRecipeFamily } from '../../../../server/src/contracts/trainingRecipes';
import { scopeOf, serverKey, resourceKey, SHARED, type Resource, type Worker } from '../keys';
import { gpusOptions } from '../shell-queue/settings';
import { trainingQueries } from '../training-data/queries';
import { pollWhile } from '../contract';
import { queryManager } from '../invalidation';

export type { Worker };
const TR = {
  recipe: 'training-runs/recipe',
  mm3Status: 'training-runs/mm3-status',
  yue2Status: 'training-runs/yue2-status',
  arStatus: 'training-runs/ar-status',
  mm3Runs: 'training-runs/mm3-runs',
  mm3Ladder: 'training-runs/mm3-ladder',
  narRuns: 'training-runs/nar-runs',
  arRuns: 'training-runs/ar-runs',
  jointRuns: 'training-runs/joint-runs',
  holdout: 'training-runs/holdout',
  previews: 'training-runs/previews',
  review: 'training-runs/review',
  ladder: 'training-runs/ladder',
  scores: 'training-runs/scores',
  album: 'training-runs/album',
  localBatches: 'training-runs/local-batches',
  mirror: 'training-runs/mirror',
  workerUpdate: 'training-runs/worker-update',
  optimise: 'training-runs/optimise',
  prepare: 'training-runs/prepare',
  jobs: 'training-runs/jobs',
  sources: 'training-runs/sources',
  source: 'training-runs/source',
  preparations: 'training-runs/preparations',
} as const satisfies Record<string, Resource>;
export const activeJob = (job: { status: string } | null | undefined) =>
  !job || job.status === 'queued' || job.status === 'running';

// Local indexes deliberately omit worker scope. A remote job's mirrored run
// and review records belong to this PC, regardless of the Train on selection.
export const runQueries = {
  recipe: <T>(worker: Worker, family: TrainingRecipeFamily) => queryOptions({
    queryKey: serverKey(TR.recipe, { worker, id: family }),
    queryFn: ({ queryKey, signal }) => {
      const target = scopeOf(queryKey)!.worker;
      return trainingOperation<TrainingRecipe<T>>('recipes', `/${family}${target ? `?worker=${encodeURIComponent(target)}` : ''}`, { signal });
    },
  }),
  /** One owner: the settings GPUs factory. */
  gpus: () => gpusOptions(),
  mm3Status: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TR.mm3Status, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getMm3Status(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  yue2Status: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TR.yue2Status, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getYue2Status(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  arStatus: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TR.arStatus, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getYue2ArStatus(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  mm3Runs: (id: string) => queryOptions({
    queryKey: serverKey(TR.mm3Runs, { id }),
    queryFn: ({ signal }) => api.listMm3Runs(id, { signal }),
  }),
  mm3Ladder: (id: string, run: string) => queryOptions({
    queryKey: serverKey(TR.mm3Ladder, { id, filters: { run } }),
    queryFn: ({ signal }) => api.getMm3Ladder(id, run, { signal }),
  }),
  narRuns: (id: string) => queryOptions({
    queryKey: serverKey(TR.narRuns, { id }),
    queryFn: ({ signal }) => api.listYue2Runs(id, { signal }),
  }),
  arRuns: (id: string) => queryOptions({
    queryKey: serverKey(TR.arRuns, { id }),
    queryFn: ({ signal }) => api.listYue2ArRuns(id, { signal }),
  }),
  jointRuns: (id: string) => queryOptions({
    queryKey: serverKey(TR.jointRuns, { id }),
    queryFn: ({ signal }) => api.listYue2AitkRuns(id, { signal }),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof api.listYue2AitkRuns>>>(data =>
      !data || !!data.activeJob && activeJob(data.activeJob) || data.runs.some(run => run.live), 5000),
  }),
  holdout: (worker: Worker, id: string, seed: number) => queryOptions({
    queryKey: serverKey(TR.holdout, { worker, id, filters: { seed } }),
    queryFn: ({ queryKey, signal }) => api.getYue2JointHoldout(id, { seed },{ worker: scopeOf(queryKey)!.worker, signal }),
  }),
  previews: (id: string, run: string) => queryOptions({
    queryKey: serverKey(TR.previews, { id, filters: { run } }),
    queryFn: ({ signal }) => api.listYue2JointPreviewsLocal(id, run, { signal }),
  }),
  review: () => queryOptions({
    queryKey: serverKey(TR.review),
    queryFn: ({ signal }) => api.listYue2Review({ signal }),
    refetchInterval: pollWhile(() => true, 30_000),
  }),
  ladder: (id: string, run: string) => queryOptions({
    queryKey: serverKey(TR.ladder, { id, filters: { run } }),
    queryFn: ({ signal }) => getReviewLadder(id, run, { signal }),
  }),
  scores: (id: string, run: string) => queryOptions({
    queryKey: serverKey(TR.scores, { id, filters: { run } }),
    queryFn: ({ signal }) => api.listYue2RungScores(id, run, { signal }),
  }),
  album: (id: string, run: string) => queryOptions({
    queryKey: serverKey(TR.album, { id, filters: { run } }),
    queryFn: ({ signal }) => api.getYue2AlbumScore(id, run, { signal }),
  }),
  localBatches: () => queryOptions({
    queryKey: serverKey(TR.localBatches),
    queryFn: ({ signal }) => api.listYue2BatchesLocal({ signal }),
    refetchInterval: pollWhile(() => true, 10_000),
  }),
  mirror: (name: string) => queryOptions({
    queryKey: serverKey(TR.mirror, { id: name }),
    queryFn: ({ signal }) => api.getWorkerMirror(name, { signal }),
    refetchInterval: pollWhile(() => true, 15_000),
  }),
  workerUpdate: (name: string) => queryOptions({
    queryKey: serverKey(TR.workerUpdate, { id: name }),
    queryFn: ({ signal }) => api.getWorkerUpdate(name, { signal }),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof api.getWorkerUpdate>>>(data =>
      !!data && !api.isTerminalStatus(data.status), 5000),
  }),
  optimise: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TR.optimise, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getYue2Optimise(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  prepare: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TR.prepare, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.getYue2AitkPrepare(id, undefined, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  jobs: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TR.jobs, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.listJobs(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  sources: (worker: Worker, id: string) => queryOptions({
    queryKey: serverKey(TR.sources, { worker, id }),
    queryFn: ({ queryKey, signal }) => api.listYue2SheetSources(id, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  source: (worker: Worker, id: string, name: string) => queryOptions({
    queryKey: serverKey(TR.source, { worker, id, filters: { name } }),
    queryFn: ({ queryKey, signal }) => api.getYue2SheetSource(id, name, { worker: scopeOf(queryKey)!.worker, signal }),
  }),
  preparations: (id: string) => queryOptions({
    queryKey: serverKey(TR.preparations, { id }),
    queryFn: ({ signal }) => preparation.listYue2Preparations(id, { signal }),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof preparation.listYue2Preparations>>>(data =>
      !data || data.some(p => !['done', 'failed', 'cancelled', 'interrupted'].includes(p.status))),
  }),
  /** One owner: training-data's factories; this view adds the polling cadence. */
  preparation: (id: string) => queryOptions({
    ...trainingQueries.preparation(id),
    refetchInterval: pollWhile<Awaited<ReturnType<typeof preparation.getYue2Preparation>>>(data =>
      !data || !['done', 'failed', 'cancelled', 'interrupted'].includes(data.status)),
  }),
  preparationJob: (id: string) => trainingQueries.preparationJob(id),
};

/** What a training command dirtied (replaces runCommand's success-only plan):
 * the preparation pair, this worker's training-runs reads (unscoped ones
 * included), and its training-data reads. A write that dirties keys outside
 * those filters (an unscoped training-data entry, another worker's scope)
 * names them explicitly in its own plan. Capture the worker before starting
 * the command, including across awaits. */
export function trainingWritePlan(client: QueryClient, worker: Worker, ...explicit: QueryKey[]): Promise<unknown> {
  return Promise.all([
    queryManager.invalidate(client, ...[SHARED.preparation, SHARED.preparationJob].map(r => resourceKey(r)), ...explicit),
    queryManager.filter(client, { queryKey: ['hs'], predicate: q =>
      String(q.queryKey[1]).startsWith('training-runs/') &&
      (scopeOf(q.queryKey)?.worker === undefined || scopeOf(q.queryKey)?.worker === worker) }),
    queryManager.filter(client, { queryKey: ['hs'], predicate: q =>
      String(q.queryKey[1]).startsWith('training-data/') && scopeOf(q.queryKey)?.worker === worker }),
  ]);
}

