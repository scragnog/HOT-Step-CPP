// Media and sidecars use server-returned URLs rather than the JSON API base.
// Keep the response intact for optional-sidecar and streaming consumers.
export function readMedia(url: string, signal?: AbortSignal): Promise<Response> {
  return fetch(url, { signal });
}

export function readAudioPeaks(url: string, signal?: AbortSignal): Promise<Response> {
  return fetch(`/api/audio/peaks?url=${encodeURIComponent(url)}`, { signal, cache: 'no-cache' });
}
