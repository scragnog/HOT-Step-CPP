import type { AudioIntentItem, AudioQueueState, ImportAudioQueue } from '../../../server/src/contracts/audioQueue';

async function call<T>(path: string, token: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/audio-queue${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { Authorization: `Bearer ${token}`, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `Audio queue request failed (${response.status})`);
  return value as T;
}

export const audioQueueApi = {
  list: (token: string) => call<{ items: AudioIntentItem[] }>('/items', token),
  state: (token: string) => call<AudioQueueState>('/state', token),
  enqueue: (token: string, idempotencyKey: string, request: Record<string, unknown>, meta?: Record<string, unknown>) =>
    call<{ item: AudioIntentItem; created: boolean }>('/items', token, { idempotencyKey, request, meta }),
  cancel: (token: string, id: string) => call<{ item: AudioIntentItem }>(`/items/${encodeURIComponent(id)}/cancel`, token, {}),
  retry: (token: string, id: string) => call<{ item: AudioIntentItem }>(`/items/${encodeURIComponent(id)}/retry`, token, {}),
  dismiss: async (token: string, id: string): Promise<void> => {
    const response = await fetch(`/api/audio-queue/items/${encodeURIComponent(id)}`, {
      method: 'DELETE', headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'Cannot remove queue item');
  },
  importLegacy: (token: string, body: ImportAudioQueue) => call<{
    backupId: string; imported: number; existing: number;
    items: { legacyId: string; itemId: string; status: string }[];
  }>('/migration/import', token, body),
  resumeHeld: (token: string, ids: string[]) => call('/migration/resume-held', token, { ids }),
  rollbackExport: (token: string) => call<{ version: number; exportedAt: number; state: AudioQueueState; items: AudioIntentItem[] }>(
    '/migration/rollback-export', token, {}),
  pause: (token: string) => call<AudioQueueState>('/pause', token, {}),
  resume: (token: string) => call<AudioQueueState>('/resume', token, {}),
};
