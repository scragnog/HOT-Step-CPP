// workflowApi.ts — client for /api/workflows: durable workflow jobs and
// revisioned documents (server/src/routes/workflows.ts).
//
// followJob reads the job's event stream with fetch rather than EventSource,
// so it can send the bearer token. It resumes from the last event it saw
// after a drop, and hands back a fresh snapshot (gap: true) when the server
// no longer holds the events in between.
import type {
  WorkflowDocument, WorkflowEvent, WorkflowJob, WorkflowJobStatus, WorkflowReplay,
} from '../../../server/src/contracts/workflow';

export class WorkflowRequestError extends Error {
  readonly status: number;
  /** Set on a 409 caused by a stale revision. */
  readonly currentRevision?: number;
  constructor(status: number, message: string, currentRevision?: number) {
    super(message);
    this.status = status;
    this.currentRevision = currentRevision;
  }
}

async function call<T>(token: string, path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`/api/workflows${path}`, {
    method,
    headers: { Authorization: `Bearer ${token}`, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new WorkflowRequestError(response.status, value.error || `Workflow request failed (${response.status})`, value.currentRevision);
  }
  return value as T;
}

const id = (v: string) => encodeURIComponent(v);
const FINAL = new Set<WorkflowJobStatus>(['succeeded', 'failed', 'cancelled', 'interrupted']);

export const workflowApi = {
  submit: (token: string, kind: string, idempotencyKey: string, input: Record<string, unknown>) =>
    call<{ job: WorkflowJob; created: boolean }>(token, '/jobs', 'POST', { kind, idempotencyKey, input }),
  list: (token: string, filter: { kind?: string; status?: WorkflowJobStatus } = {}) =>
    call<{ jobs: WorkflowJob[] }>(token, `/jobs?${new URLSearchParams(filter as Record<string, string>)}`),
  get: (token: string, jobId: string, after = 0) => call<WorkflowReplay>(token, `/jobs/${id(jobId)}?after=${after}`),
  cancel: (token: string, jobId: string) => call<{ job: WorkflowJob }>(token, `/jobs/${id(jobId)}/cancel`, 'POST', {}),
  retry: (token: string, jobId: string) => call<{ job: WorkflowJob }>(token, `/jobs/${id(jobId)}/retry`, 'POST', {}),

  createDocument: (token: string, kind: string, data: Record<string, unknown>) =>
    call<{ document: WorkflowDocument }>(token, '/documents', 'POST', { kind, data }),
  listDocuments: (token: string, kind: string) => call<{ documents: WorkflowDocument[] }>(token, `/documents?kind=${id(kind)}`),
  getDocument: (token: string, docId: string) => call<{ document: WorkflowDocument }>(token, `/documents/${id(docId)}`),
  /** Throws WorkflowRequestError with status 409 and currentRevision when stale. */
  updateDocument: (token: string, docId: string, expectedRevision: number, data: Record<string, unknown>) =>
    call<{ document: WorkflowDocument }>(token, `/documents/${id(docId)}`, 'PUT', { expectedRevision, data }),
  removeDocument: (token: string, docId: string, expectedRevision: number) =>
    call<{ removed: true }>(token, `/documents/${id(docId)}?expectedRevision=${expectedRevision}`, 'DELETE'),
};

export interface FollowHandlers {
  /** The job's state on every (re)connect. `gap` means events since the last
   *  one seen were dropped: rebuild the view from `job`. */
  onSnapshot?: (job: WorkflowJob, gap: boolean) => void;
  /** Each event, once, in order. */
  onEvent: (event: WorkflowEvent) => void;
  /** A connection dropped; it will be retried. */
  onError?: (error: unknown) => void;
}

/** Follow a job until it finishes or `signal` aborts. Resolves with the
 *  final status, or null when aborted. `after` resumes from a known event. */
export async function followJob(token: string, jobId: string, handlers: FollowHandlers,
  opts: { after?: number; signal?: AbortSignal; retryMs?: number } = {}): Promise<WorkflowJobStatus | null> {
  let last = opts.after ?? 0;
  let delay = opts.retryMs ?? 1000;
  while (!opts.signal?.aborted) {
    try {
      const response = await fetch(`/api/workflows/jobs/${id(jobId)}/events?after=${last}`, {
        headers: { Authorization: `Bearer ${token}` }, signal: opts.signal,
      });
      if (!response.ok || !response.body) {
        const value = await response.json().catch(() => ({}));
        // A missing job or a bad request will not fix itself by retrying.
        if (response.status === 404 || response.status === 401 || response.status === 400) {
          throw new WorkflowRequestError(response.status, value.error || `Cannot follow job (${response.status})`);
        }
        throw new Error(value.error || `Cannot follow job (${response.status})`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalStatus: WorkflowJobStatus | null = null;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let at;
        while ((at = buffer.indexOf('\n\n')) >= 0) {
          const line = buffer.slice(0, at).split('\n').find(l => l.startsWith('data: '));
          buffer = buffer.slice(at + 2);
          if (!line) continue;
          const frame = JSON.parse(line.slice(6));
          if (frame.type === 'snapshot') {
            const job = frame.job as WorkflowJob;
            // On a gap the kept events still follow; seq checks skip none of them.
            handlers.onSnapshot?.(job, frame.gap);
            if (FINAL.has(job.status) && job.lastSeq <= last) finalStatus = job.status;
          } else if (frame.type === 'event') {
            const event = frame.event as WorkflowEvent;
            if (event.seq <= last) continue;
            last = event.seq;
            handlers.onEvent(event);
            const status = (event.data as { status?: WorkflowJobStatus } | null)?.status;
            if (event.type === 'status' && status && FINAL.has(status)) finalStatus = status;
          }
        }
      }
      if (finalStatus) return finalStatus;
      delay = opts.retryMs ?? 1000;  // a clean end without a final status: reconnect promptly
    } catch (err) {
      if (opts.signal?.aborted) return null;
      if (err instanceof WorkflowRequestError) throw err;
      handlers.onError?.(err);
      delay = Math.min(delay * 2, 10_000);
    }
    await new Promise(resolve => setTimeout(resolve, delay));
  }
  return null;
}
