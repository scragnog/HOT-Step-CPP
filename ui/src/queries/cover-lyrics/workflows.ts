import { queryOptions } from '@tanstack/react-query';
import { apiClient, type ReadOptions } from '@/services/httpClient';
import { followJob, workflowApi } from '@/services/workflowApi';
import type { WorkflowDocument, WorkflowReplay } from '../../../../server/src/contracts/workflow';
import { queryClient } from '@/queries/client';
import { resourceKey, sharedKeys } from '@/queries/keys';
import { queryManager } from '@/queries/invalidation';
import { followWorkflowJob } from '@/queries/workflowFollow';

export const workflowOptions = (id: string, after = 0, options: ReadOptions = {}) => {
  return queryOptions({
  queryKey: sharedKeys.workflowJob(id, after),
  queryFn: ({ signal }) => apiClient.get<WorkflowReplay>(`/workflows/jobs/${encodeURIComponent(id)}?after=${after}`, { signal: options.signal ? AbortSignal.any([signal, options.signal]) : signal }),
  staleTime: 0,
  });
};
export const documentOptions = (id: string, options: ReadOptions = {}) => {
  return queryOptions({
  queryKey: sharedKeys.workflowDocument(id),
  queryFn: ({ signal }) => apiClient.get<{ document: WorkflowDocument }>(`/workflows/documents/${encodeURIComponent(id)}`, { signal: options.signal ? AbortSignal.any([signal, options.signal]) : signal }),
  staleTime: 0,
  });
};

export const coverLyricWorkflowApi = {
  ...workflowApi,
  get: (id: string, after = 0, options: ReadOptions = {}) => queryClient.query(workflowOptions(id, after, options)),
  getDocument: (id: string, options: ReadOptions = {}) => queryClient.query(documentOptions(id, options)),
  updateDocument: async (...args: Parameters<typeof workflowApi.updateDocument>) => {
    const key = documentOptions(args[0]).queryKey;
    await queryClient.cancelQueries({ queryKey: key });
    const result = await workflowApi.updateDocument(...args);
    queryClient.setQueryData(key, result);
    await queryManager.invalidate(queryClient, key);
    return result;
  },
};

/** Streaming reducers retain sequencing; events invalidate captured snapshots.
 *  Reconnects always reconcile, including when the replay history has a gap.
 *  A cover job also dirties every lireek read and the drafts of its scope. */
export const followCoverLyricJob: typeof followJob = (id, handlers, opts) =>
  followWorkflowJob(id, handlers, opts, scope => [
    resourceKey('cover-lyrics/lireek'),
    resourceKey('cover-lyrics/drafts', scope),
  ]);
