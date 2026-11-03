import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import * as api from '../../services/trainingApi';
import { bridgeRunEvents, type RunBridgeOptions } from './events';
import { pollWhile } from '../contract';
import { trainingQueries } from '../training-data/queries';
import { activeJob, type Worker } from './queries';
import { useTrainingStore } from '../../stores/trainingStore';
import { queryManager } from '../invalidation';
import { subscribeEventSource } from '../../services/sharedEventSource';

/** The stream is shared with the chart/log reducers. Replayed terminal
 * frames invalidate once per job; they never start a command or audition.
 * Once the cached job reaches a terminal status the bridge tears down so the
 * EventSource does not reconnect and replay forever. */
export function useRunEvents(jobId: string | null | undefined, worker: Worker,
  options?: RunBridgeOptions) {
  const client = useQueryClient();
  const projectStore = options?.projectStore === true;
  // Disabled observer: reads whatever the projection (or a fetch) has already
  // written into the cache without triggering its own request.
  const status = useQuery({
    ...trainingQueries.job(worker, jobId ?? ''), enabled: false,
    select: (j: api.TrainingJobSummary | undefined) => j?.status,
  }).data;
  const terminal = api.isTerminalStatus(status);
  useEffect(() => {
    if (!jobId || terminal) return;
    return bridgeRunEvents(client, jobId, worker, (url, subscriber, opts) =>
      subscribeEventSource(url, subscriber, { ...opts, isTerminal: data => {
        try {
          const event = JSON.parse(data) as api.TrainingStreamEvent;
          return event.type === 'status' && !activeJob({ status: event.status });
        } catch { return false; }
      } }), { projectStore });
  }, [client, jobId, worker, projectStore, terminal]);
}

export function useRunJob(worker: Worker, id: string | null | undefined,
  options?: RunBridgeOptions) {
  const query = useQuery({ ...trainingQueries.job(worker, id ?? ''), enabled: !!id,
    refetchInterval: pollWhile<api.TrainingJobSummary>(activeJob) });
  // Resolve status before subscribing, including a restored terminal job.
  useRunEvents(query.data && activeJob(query.data) ? id : null, worker, options);
  return query;
}

/** The job the studio is following, read from the one Query cache entry. The
 *  store keeps only the selected id; the projection below is the single
 *  stream/poll owner that fills this entry. */
export function useActiveTrainingJob(): api.TrainingJobSummary | null {
  const worker = useTrainingStore(s => s.trainingWorker);
  const id = useTrainingStore(s => s.activeJobId);
  // A disabled observer: the projection writes this entry, and every card
  // reads it without opening a second stream or poll.
  const cached = useQuery({
    ...trainingQueries.job(worker, id ?? ''),
    enabled: false,
    staleTime: Infinity,
  });
  return cached.data ?? null;
}

/** One mount, beside the studio shell. It opens the job stream, polls the
 *  selected job and feeds the store's sample/log/preview/metric reducers.
 *  Terminal status runs the authoritative post-job refresh once. */
export function useTrainingJobProjection(): void {
  const worker = useTrainingStore(s => s.trainingWorker);
  const id = useTrainingStore(s => s.activeJobId);
  const jobTerminal = useTrainingStore(s => s.jobTerminal);
  const job = useRunJob(worker, id, { projectStore: true }).data;
  const jobId = job?.id;
  const status = job?.status;
  useEffect(() => {
    if (!jobId || !status || status === 'queued' || status === 'running') return;
    jobTerminal(jobId);
  }, [jobId, status, jobTerminal]);
}

/** Only the selected id is client state. The job itself has one cache owner.
 * A callback from a previous dataset/worker cannot adopt a job on this one. */
export function useJobSelection(worker: Worker, datasetId: string | null) {
  const client = useQueryClient();
  const scope = JSON.stringify([worker, datasetId]);
  const current = useRef({ scope, generation: 0 });
  if (current.current.scope !== scope) current.current = { scope, generation: current.current.generation + 1 };
  const generation = current.current.generation;
  const [selection, select] = useState<{ generation: number; id: string } | null>(null);
  const query = useRunJob(worker, selection?.generation === generation ? selection.id : null);
  const setJob = (job: api.TrainingJobSummary | null) => {
    if (current.current.generation !== generation) return;
    if (job) client.setQueryData(trainingQueries.job(worker, job.id).queryKey, job);
    select(job ? { generation, id: job.id } : null);
  };
  return [query.data ?? null, setJob] as const;
}

export function useScopedId(worker: Worker, datasetId: string) {
  const scope = JSON.stringify([worker, datasetId]);
  const current = useRef({ scope, generation: 0 });
  if (current.current.scope !== scope) current.current = { scope, generation: current.current.generation + 1 };
  const generation = current.current.generation;
  const [selection, select] = useState<{ generation: number; id: string } | null>(null);
  return [selection?.generation === generation ? selection.id : '', (id: string) => {
    if (current.current.generation === generation) select({ generation, id });
  }] as const;
}

/** A reload signal invalidates an existing snapshot, rather than adding
 * another cache entry for every completion or refresh click. Mounts reuse it. */
export function useReload(key: readonly unknown[], revision: unknown) {
  const client = useQueryClient();
  const encoded = JSON.stringify(key);
  const previous = useRef({ encoded, revision });
  useEffect(() => {
    const before = previous.current;
    previous.current = { encoded, revision };
    if (revision === undefined || before.encoded !== encoded || Object.is(before.revision, revision)) return;
    void queryManager.invalidate(client, key);
  // The resource key is structural, not the allocation made each render.
  }, [client, encoded, revision]);
}
