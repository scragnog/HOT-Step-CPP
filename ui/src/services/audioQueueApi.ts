import { apiClient, type ClientOptions } from './httpClient';
import type { AudioIntentItem, AudioQueueState, ImportAudioQueue } from '../../../server/src/contracts/audioQueue';

async function call<T>(path: string, body?: unknown, opts: ClientOptions = {}): Promise<T> {
  return body === undefined
    ? apiClient.get<T>(`/audio-queue${path}`, opts)
    : apiClient.post<T>(`/audio-queue${path}`, body, opts);
}

export const audioQueueApi = {
  list: (opts?: ClientOptions) => call<{ items: AudioIntentItem[] }>('/items', undefined, opts),
  state: (opts?: ClientOptions) => call<AudioQueueState>('/state', undefined, opts),
  enqueue: (idempotencyKey: string, request: Record<string, unknown>, meta?: Record<string, unknown>) =>
    call<{ item: AudioIntentItem; created: boolean }>('/items', { idempotencyKey, request, meta }),
  cancel: (id: string) => call<{ item: AudioIntentItem }>(`/items/${encodeURIComponent(id)}/cancel`, {}),
  retry: (id: string) => call<{ item: AudioIntentItem }>(`/items/${encodeURIComponent(id)}/retry`, {}),
  dismiss: async (id: string): Promise<void> => {
    await apiClient.delete(`/audio-queue/items/${encodeURIComponent(id)}`);
  },
  // Silent: a failed migration surfaces through the queue banner (or the
  // store's owner-decision path) as the only sign anything happened — a toast
  // would be a second copy of the same failure.
  importLegacy: (body: ImportAudioQueue) => call<{
    backupId: string; imported: number; existing: number;
    items: { legacyId: string; itemId: string; status: string }[];
  }>('/migration/import', body, { silent: true }),
  resumeHeld: (ids: string[]) => call('/migration/resume-held', { ids }),
  pause: () => call<AudioQueueState>('/pause', {}),
  resume: () => call<AudioQueueState>('/resume', {}),
};
