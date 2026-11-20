export function openStormStream(params: Record<string, unknown>, streamId: string, signal: AbortSignal): Promise<Response> {
  return fetch('/api/generate/storm/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...params, streamId }),
    signal,
  });
}
