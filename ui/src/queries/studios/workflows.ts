import { followJob, workflowApi } from '@/services/workflowApi';
import { queryClient } from '@/queries/client';
import { queryManager } from '@/queries/invalidation';
import { documentOptions, workflowOptions } from '@/queries/cover-lyrics/workflows';
import { followWorkflowJob } from '@/queries/workflowFollow';

// One owner per resource: cover-lyrics' workflow factories carry the optional
// ReadOptions signal and are consumed by both workflow APIs.
export { documentOptions, workflowOptions } from '@/queries/cover-lyrics/workflows';

export const studioWorkflowApi = {
  ...workflowApi,
  get: (id: string, after = 0) => queryClient.query(workflowOptions(id, after)),
  getDocument: (id: string) => queryClient.query(documentOptions(id)),
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
 *  Reconnects always reconcile, including when the replay history has a gap. */
export const followStudioJob: typeof followJob = (id, handlers, opts) =>
  followWorkflowJob(id, handlers, opts);
