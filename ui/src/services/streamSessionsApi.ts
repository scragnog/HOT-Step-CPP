// streamSessionsApi.ts — client for /api/stream-sessions (Batch 5 row 8a).
//
// The STORM and MM3 stream responses name their Node session in the
// X-Stream-Session header (useStreamAudio's sessionId, mm3StreamSessionId).
// Node analyses every chunk and, between start and stop, records the engine's
// own samples; export returns that canonical recording. Browser device capture
// (useStreamAudio's startRecording) is separate and unchanged.
import type { StreamExportFormat, StreamSessionSummary } from '../../../server/src/contracts/streamSessions';

export type { StreamExportFormat, StreamSessionSummary } from '../../../server/src/contracts/streamSessions';

async function readJson<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((body as { error?: string }).error || `Request failed (${res.status})`);
  return body as T;
}

const base = (id: string) => `/api/stream-sessions/${encodeURIComponent(id)}`;

export async function getStreamSession(id: string): Promise<StreamSessionSummary> {
  return (await readJson<{ session: StreamSessionSummary }>(await fetch(base(id)))).session;
}

export async function streamRecording(id: string, action: 'start' | 'stop' | 'discard'): Promise<StreamSessionSummary> {
  return (await readJson<{ session: StreamSessionSummary }>(await fetch(`${base(id)}/recording`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }),
  }))).session;
}

/** URL of a stopped recording as a download; open it in an <a download>. */
export function streamRecordingExportUrl(id: string, format: StreamExportFormat = 'wav', bitrate?: number): string {
  const query = new URLSearchParams({ format, ...(bitrate ? { bitrate: String(bitrate) } : {}) });
  return `${base(id)}/recording/export?${query}`;
}
