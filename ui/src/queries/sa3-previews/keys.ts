import { resourceKey, serverKey } from '@/queries/keys';

// Lives in the training-runs area: previews are run records read per worker.
const RESOURCE = 'training-runs/sa3-previews';

export const sa3PreviewKeys = {
  /** One run's preview list on one worker. */
  list: (worker: string | null, datasetId: string, run: string) => serverKey(RESOURCE, { worker, id: datasetId, filters: { run } }),
  /** Every run of one dataset on one worker, the unit a render or job event refreshes. */
  dataset: (worker: string | null, datasetId: string) => resourceKey(RESOURCE, { worker }, datasetId),
};
