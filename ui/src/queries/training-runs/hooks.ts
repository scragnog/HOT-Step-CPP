import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import * as api from '../../services/trainingApi';
import { bridgeRunEvents } from './events';
import { pollWhile } from '../contract';
import { trainingQueries } from '../training-data/queries';
import { activeJob, type Worker } from './queries';
import { queryManager } from '../invalidation';

/** The stream is shared with the chart/log reducers. Replayed terminal
 * frames invalidate once per job; they never start a command or audition. */
export function useRunEvents(jobId: string | null | undefined, worker: Worker) {
  const client = useQueryClient();
  useEffect(() => {
    if (!jobId) return;
    return bridgeRunEvents(client, jobId, worker);
  }, [client, jobId, worker]);
}

export function useRunJob(worker: Worker, id: string | null | undefined) {
  const query = useQuery({ ...trainingQueries.job(worker, id ?? ''), enabled: !!id,
    refetchInterval: pollWhile<api.TrainingJobSummary>(activeJob) });
  // Resolve status before subscribing, including a restored terminal job.
  useRunEvents(query.data && activeJob(query.data) ? id : null, worker);
  return query;
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
