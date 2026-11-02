import type { QueryClient } from '@tanstack/react-query';
import * as api from '../../services/trainingApi';
import { bridgeEvents, type BridgeOptions } from '../sseBridge';
import { resourceKey, scopeOf, SHARED } from '../keys';
import type { Worker } from './queries';
import { trainingQueries } from '../training-data/queries';
import { jobFrameUpdate } from './jobFrame';
import { useTrainingStore } from '../../stores/trainingStore';
import { queryManager } from '../invalidation';
import { ApiError, HttpStatus } from '../../services/apiError';

export function invalidateRunReads(client: QueryClient, worker: Worker) {
  return queryManager.filter(client, { queryKey: ['hs'], predicate: q =>
    (q.queryKey[1] === SHARED.preparation || q.queryKey[1] === SHARED.preparationJob ||
      String(q.queryKey[1]).startsWith('training-runs/') || String(q.queryKey[1]).startsWith('training-data/')) &&
    (scopeOf(q.queryKey)?.worker === undefined || scopeOf(q.queryKey)?.worker === worker) });
}

/** One terminal rule for a job status, shared with the hook and the cards. */
const terminal = (event: api.TrainingStreamEvent): boolean => {
  const status = event.type === 'status' ? event.status
    : event.type === 'job' ? event.job.status : undefined;
  return api.isTerminalStatus(status);
};

export interface RunBridgeOptions {
  /** Apply store-owned frames (samples, logs, previews, metrics) and probe a
   *  dropped stream. Only the one active-job projection sets this: every
   *  other bridge only updates invalidation state. */
  projectStore?: boolean;
}

/** Metrics and logs keep their existing reducers and shared connection.
 * This bridge refreshes snapshots, updates the job cache and, for the active
 * job projection, feeds the store's sample/log/preview/metric reducers. It
 * cannot submit a job or audition. */
export function bridgeRunEvents(client: QueryClient, jobId: string, worker: Worker,
  subscribe?: BridgeOptions<api.TrainingStreamEvent>['subscribe'],
  options: RunBridgeOptions = {}) {
  const jobKey = trainingQueries.job(worker, jobId).queryKey;
  const projection = options.projectStore === true;
  return bridgeEvents<api.TrainingStreamEvent>(client, {
    url: api.jobStreamUrl(jobId, { worker }),
    subscribe,
    onMessage: event => {
      if (!projection) return;
      const store = useTrainingStore.getState();
      if (store.trainingWorker === worker && store.activeJobId === jobId) {
        store.applyStreamEvent(event);
      }
    },
    onError: () => {
      if (!projection) return;
      // EventSource auto-reconnects and the buffer replay makes that safe. But
      // if the server restarted, the in-memory job is gone (the stream 404s
      // forever) and no terminal event will ever arrive — the UI would show
      // "running" and keep Start disabled indefinitely. Query once: a live job
      // is a no-op; a vanished job lands its final status in the same cache the
      // projection observes. Only a 404 means the job is truly gone; other
      // errors (network blip, 5xx) leave the cached state untouched so the
      // poll can recover naturally. Never overwrite an already-terminal job.
      void client.query(trainingQueries.job(worker, jobId)).catch((err: unknown) => {
        // ApiError carries the status; a bare Error cannot say the job is gone.
        // Anything but a definite 404 is transient, and the poll can recover.
        if (!(err instanceof ApiError) || err.status !== HttpStatus.notFound) return;
        client.setQueryData(jobKey, old => {
          if (!old) return old;
          const prev = old as api.TrainingJobSummary;
          if (api.isTerminalStatus(prev.status)) return old; // already terminal
          return { ...prev, status: 'failed' as const,
            error: 'Job lost — the server restarted' };
        });
      });
    },
    reconnectInvalidate: [
      resourceKey(SHARED.preparation), resourceKey(SHARED.preparationJob),
      resourceKey('training-data/job', { worker }, jobId),
      ...['mm3-status', 'yue2-status', 'ar-status', 'optimise', 'prepare', 'sources', 'source', 'jobs']
        .map(name => resourceKey(`training-runs/${name}`, { worker })),
      ...['joint-runs', 'mm3-runs', 'nar-runs', 'ar-runs', 'previews', 'review', 'ladder', 'scores', 'album']
        .map(name => resourceKey(`training-runs/${name}`)),
    ],
    map: event => {
      if (!event) return undefined;
      const update = jobFrameUpdate(event);
      const done = terminal(event);
      if (!update && !done) return undefined;
      return {
        ...(update ? { update: [{ key: jobKey, apply: update }] } : {}),
        ...(done ? { once: { id: `training-runs:${worker ?? 'local'}:${jobId}:terminal`,
          run: () => { void invalidateRunReads(client, worker); } } } : {}),
      };
    },
  });
}
