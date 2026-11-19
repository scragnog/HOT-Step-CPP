import { queryOptions, type QueryClient } from '@tanstack/react-query';
import { listSa3Previews, renderSa3Preview, type Sa3PreviewListResponse, type Sa3PreviewRenderRequest } from '@/services/sa3PreviewApi';
import { scopeOf } from '@/queries/keys';
import { pollWhile } from '@/queries/contract';
import { queryManager } from '@/queries/invalidation';
import { sa3PreviewKeys } from './keys';

/** Rows still moving: poll until every cell is settled. */
export const sa3PreviewsPending = (data: Sa3PreviewListResponse | undefined) =>
  !!data?.previews.some(p => p.status === 'queued' || p.status === 'rendering');

/** `jobActive` keeps the list polling while a training job may add rows. */
export const sa3PreviewOptions = (worker: string | null, datasetId: string, run: string, jobActive = false) => queryOptions({
  queryKey: sa3PreviewKeys.list(worker, datasetId, run),
  queryFn: ({ queryKey, signal }) => listSa3Previews(datasetId, run, { worker: scopeOf(queryKey)!.worker!, signal }),
  refetchInterval: pollWhile<Sa3PreviewListResponse>(data => jobActive || sa3PreviewsPending(data)),
});

export const invalidateSa3Previews = (client: QueryClient, worker: string | null, datasetId: string) =>
  queryManager.invalidate(client, sa3PreviewKeys.dataset(worker, datasetId));

/** A manual render or retry never carries browser form values: the server renders with the selected
 *  run's own saved settings, so a reload, a resumed run or previews switched off cannot override them. */
export const manualRenderRequest = (run: string, step: number): Sa3PreviewRenderRequest => ({ run, step });

/** Render or retry once; only a success refreshes this dataset's previews on the captured worker. */
export async function renderSa3PreviewCommand(client: QueryClient, worker: string | null, datasetId: string, body: Sa3PreviewRenderRequest) {
  const result = await renderSa3Preview(datasetId, body, worker);
  await invalidateSa3Previews(client, worker, datasetId);
  return result;
}
