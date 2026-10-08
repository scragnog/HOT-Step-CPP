// followSection.ts — follow one Song Builder section job and keep the open
// project's view current. The project id is passed in, not read from React
// state, so a follow started while a project is being opened reloads that
// project; once the user has moved to another project (isCurrent false) its
// results are dropped.
import { followJob } from '../../services/workflowApi';
import type { WorkflowJobStatus } from '../../../../server/src/contracts/workflow';

export interface FollowSectionHandlers<V> {
  /** Load a project's current view. */
  load(projectId: string): Promise<V>;
  /** True while `projectId` is still the open project. */
  isCurrent(projectId: string): boolean;
  onView(view: V): void;
  onProgress(p: { finished: number; landed: number; total: number }): void;
  /** The job ended; the view was reloaded first. */
  onEnd(status: WorkflowJobStatus | null, landed: number): void;
  onError(error: Error): void;
  /** Injected in tests. */
  follow?: typeof followJob;
}

export function followSection<V>(token: string, projectId: string, jobId: string, h: FollowSectionHandlers<V>, signal: AbortSignal): Promise<void> {
  let total = 1, finished = 0, landed = 0;
  const live = () => !signal.aborted && h.isCurrent(projectId);
  const reload = async () => {
    try { const view = await h.load(projectId); if (live()) h.onView(view); } catch { /* the next event or the end reloads */ }
  };
  // followJob starts at event 0 and resumes after the last one it delivered,
  // so each variant event is counted once across reconnects.
  return (h.follow ?? followJob)(token, jobId, {
    onSnapshot: job => { total = Number((job.input as { variants?: number }).variants) || 1; },
    onEvent: e => {
      if (e.type !== 'variant' || !live()) return;
      finished++;
      if (((e.data as { songIds?: string[] } | null)?.songIds?.length ?? 0) > 0) landed++;
      h.onProgress({ finished, landed, total });
      void reload();
    },
  }, { signal }).then(async status => {
    if (!live()) return;
    await reload();
    if (live()) h.onEnd(status, landed);
  }, (err: Error) => { if (live()) h.onError(err); });
}
