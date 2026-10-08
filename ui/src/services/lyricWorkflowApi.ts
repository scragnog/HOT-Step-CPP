import { followJob, workflowApi } from './workflowApi';
import type { WorkflowJob } from '../../../server/src/contracts/workflow';
import type { WrittenSongIntent } from '../../../server/src/contracts/resolution';

export type LyricBatchRequest = {
  type: 'profile' | 'generate' | 'refine' | 'fetch';
  targetId?: number; provider?: string; model?: string; count?: number;
  extraInstructions?: string; userSubject?: string; noThink?: boolean;
  artist?: string; album?: string; maxSongs?: number;
};

export type LyricItemResult = { index: number; status: 'done' | 'error'; value?: any; error?: string };
export type LyricBatchResult = { results: LyricItemResult[] };

async function submit(token: string, path: string, body: unknown): Promise<WorkflowJob> {
  const response = await fetch(`/api/lireek/${path}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `Lyric workflow failed (${response.status})`);
  return value.job as WorkflowJob;
}

export const lyricWorkflowApi = {
  submitBatch: (token: string, items: LyricBatchRequest[], idempotencyKey = crypto.randomUUID()) =>
    submit(token, 'workflow-batches', { items, idempotencyKey }),
  submitRenders: (token: string, intents: WrittenSongIntent[], idempotencyKey = crypto.randomUUID()) =>
    submit(token, 'workflow-renders', { intents, idempotencyKey }),
  cancel: (token: string, id: string) => workflowApi.cancel(token, id),
  list: (token: string) => workflowApi.list(token, { kind: 'lyric-batch' }),
  get: (token: string, id: string) => workflowApi.get(token, id),
  follow: followJob,
};

export async function runLyricOperation(token: string, request: LyricBatchRequest, handlers: {
  onChunk?: (text: string) => void; onPhase?: (phase: string) => void;
  onResult?: (value: unknown) => void; onError?: (message: string) => void;
} = {}): Promise<void> {
  const job = await lyricWorkflowApi.submitBatch(token, [request]);
  let failure: string | undefined;
  const seen = new Set<number>();
  const status = await lyricWorkflowApi.follow(token, job.id, {
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
    const final = await lyricWorkflowApi.get(token, job.id);
    const results = (final.job.result as LyricBatchResult | null)?.results || [];
    for (const result of results) {
      if (seen.has(result.index)) continue;
      if (result.status === 'done') handlers.onResult?.(result.value);
      else { failure = result.error || 'Item failed'; handlers.onError?.(failure); }
    }
  }
  if (status !== 'succeeded' || failure) throw new Error(failure || `Workflow ${status}`);
}
