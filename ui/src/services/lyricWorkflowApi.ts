import { apiClient, type ClientOptions, type ReadOptions } from './httpClient';
import { readSnapshot } from '@/queries/cover-lyrics/reads';
import { coverLyricWorkflowApi as workflowApi, followCoverLyricJob as followJob } from '@/queries/cover-lyrics/workflows';
import type { WorkflowJob } from '../../../server/src/contracts/workflow';
import type { WrittenSongIntent } from '../../../server/src/contracts/resolution';
import { newId } from '@/utils/id';

export type LyricBatchRequest = {
  type: 'profile' | 'generate' | 'refine' | 'fetch';
  targetId?: number; provider?: string; model?: string; count?: number;
  extraInstructions?: string; userSubject?: string; noThink?: boolean;
  artist?: string; album?: string; maxSongs?: number;
};

export type LyricItemResult = { index: number; status: 'done' | 'error'; value?: any; error?: string };
export type LyricBatchResult = { results: LyricItemResult[] };

async function submit(path: string, body: unknown, opts?: ClientOptions): Promise<WorkflowJob> {
  const value = await apiClient.post<{ job: WorkflowJob }>(`/lireek/${path}`, body, opts);
  return value.job;
}

export const lyricWorkflowApi = {
  submitBatch: (items: LyricBatchRequest[], options: { concurrency?: number; silent?: boolean } = {}, idempotencyKey = newId()) =>
    submit('workflow-batches', { items, idempotencyKey, concurrency: options.concurrency }, options.silent ? { silent: true } : undefined),
  submitRenders: (intents: WrittenSongIntent[], idempotencyKey = newId()) =>
    submit('workflow-renders', { intents, idempotencyKey }),
  cancel: (id: string) => workflowApi.cancel(id),
  list: (options: ReadOptions = {}) => readSnapshot<{ jobs: WorkflowJob[] }>('workflow', '/workflows/jobs?kind=lyric-batch', options, 0),
  get: (id: string, options: ReadOptions = {}) => workflowApi.get(id, 0, options),
  follow: followJob,
};

export async function runLyricOperation(request: LyricBatchRequest, handlers: {
  onChunk?: (text: string) => void; onPhase?: (phase: string) => void;
  onResult?: (value: unknown) => void; onError?: (message: string) => void;
} = {}): Promise<void> {
  // Silent submit: this function owns failure notification — it throws with the
  // job's own error message (often from the SSE follow, which the transport
  // never sees), and every caller surfaces that. A transport toast on the POST
  // leg would be a second copy of the same failure.
  const job = await lyricWorkflowApi.submitBatch([request], { silent: true });
  let failure: string | undefined;
  const seen = new Set<number>();
  const status = await lyricWorkflowApi.follow(job.id, {
    onEvent: event => {
      const data = event.data as { text?: string; phase?: string; value?: unknown; error?: string } | null;
      if (event.type === 'chunk' && data?.text) handlers.onChunk?.(data.text);
      if (event.type === 'phase' && data?.phase) handlers.onPhase?.(data.phase);
      const index = (data as { index?: number } | null)?.index;
      if (event.type === 'item-result') { if (index !== undefined) seen.add(index); handlers.onResult?.(data?.value); }
      if (event.type === 'item-error') { if (index !== undefined) seen.add(index); failure = data?.error || 'Item failed'; handlers.onError?.(failure); }
    },
  });
  if (status === 'succeeded') {
    const final = await lyricWorkflowApi.get(job.id);
    const results = (final.job.result as LyricBatchResult | null)?.results || [];
    for (const result of results) {
      if (seen.has(result.index)) continue;
      if (result.status === 'done') handlers.onResult?.(result.value);
      else { failure = result.error || 'Item failed'; handlers.onError?.(failure); }
    }
  }
  if (status !== 'succeeded' || failure) throw new Error(failure || `Workflow ${status}`);
}
