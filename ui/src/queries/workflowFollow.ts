// workflowFollow.ts — the one implementation of "follow a workflow job and
// invalidate what it can change".
//
// Studio workflows and cover/lyric workflows used to carry near-identical
// copies of this loop; the only difference was which extra resources a job
// dirties. The auth-epoch guard, abort plumbing and 50 ms coalescing live here
// so a fix in one path cannot miss the other (same reasoning as queries/wait.ts).

import type { QueryKey } from '@tanstack/react-query';
import { followJob } from '@/services/workflowApi';
import { queryClient } from './client';
import { resourceKey, getSessionEpoch, sharedKeys, SHARED, type KeyScope } from './keys';
import { queryManager } from './invalidation';

type FollowJob = typeof followJob;
type Handlers = Parameters<FollowJob>[1];
type Opts = Parameters<FollowJob>[2];

/** Follow `id` while revalidating its replay, its document and every song view.
 *  `extraKeys` names resources this job's domain also changes (e.g. cover
 *  drafts); it receives the job's scope so keys stay in the same auth epoch. */
export function followWorkflowJob(
  id: string,
  handlers: Handlers,
  opts: Opts | undefined,
  extraKeys: (scope: KeyScope) => QueryKey[] = () => [],
): ReturnType<FollowJob> {
  const jobScope = sharedKeys.workflowJob(id)[2];
  const jobKey = resourceKey(SHARED.workflowJob, jobScope, id);
  const documentsKey = resourceKey(SHARED.workflowDocument, jobScope);
  const extras = extraKeys(jobScope);

  const epoch = getSessionEpoch();
  const controller = new AbortController();
  const abort = () => controller.abort();
  opts?.signal?.addEventListener('abort', abort, { once: true });
  if (opts?.signal?.aborted) controller.abort();

  const current = () => {
    if (getSessionEpoch() === epoch) return true;
    controller.abort();
    return false;
  };

  let timer: ReturnType<typeof setTimeout> | undefined;
  const flush = () => {
    clearTimeout(timer);
    timer = undefined;
    if (getSessionEpoch() !== epoch) return;
    void queryManager.invalidate(queryClient, jobKey);
    void queryManager.invalidate(queryClient, documentsKey);
    void queryManager.songs.views(queryClient);
    for (const key of extras) void queryManager.invalidate(queryClient, key);
  };
  const invalidate = () => { timer ??= setTimeout(flush, 50); };

  return followJob(id, {
    ...handlers,
    onSnapshot: (job, gap) => { if (current()) { invalidate(); handlers.onSnapshot?.(job, gap); } },
    onEvent: event => { if (current()) { invalidate(); handlers.onEvent?.(event); } },
  }, { ...opts, signal: controller.signal }).then(status => {
    if (!current()) throw new DOMException('Auth session changed', 'AbortError');
    return status;
  }).finally(() => {
    opts?.signal?.removeEventListener('abort', abort);
    flush();
  });
}
