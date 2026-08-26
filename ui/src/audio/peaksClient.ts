// peaksClient.ts — waveform envelopes, fetched from the server and cached.
//
// The server computes these once per file (see server/src/services/audio/peaks.ts)
// so the browser never downloads or decodes audio to draw a waveform. A payload
// is ~25 KB against the 48–96 MB the file itself weighs.

export interface Peaks {
  version: number;
  duration: number;
  sampleRate: number;
  channels: number;
  min: number[];
  max: number[];
  unsupported?: string;
}

/** Keyed by audio URL. Small enough to keep a decent number of them — 25 KB
 *  each, so 60 entries is under 2 MB and covers any realistic listening
 *  session without the map becoming the thing that eats the tab. */
const cache = new Map<string, Peaks>();
const inFlight = new Map<string, Promise<Peaks | null>>();
const MAX_CACHED = 60;

/** Already-fetched peaks for this URL, or null. Synchronous, for the paint
 *  that happens before the fetch resolves. */
export function peaksCached(url: string): Peaks | null {
  return cache.get(url) ?? null;
}

function remember(url: string, peaks: Peaks): void {
  // Map preserves insertion order, so the first key is the oldest.
  if (cache.size >= MAX_CACHED) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  cache.set(url, peaks);
}

/**
 * Peaks for one audio URL. Returns null when the server could not produce
 * them, which callers must treat as "draw nothing" rather than as a failure to
 * play — a missing waveform is cosmetic.
 *
 * Concurrent calls for the same URL share one request.
 */
export function fetchPeaks(url: string): Promise<Peaks | null> {
  const hit = cache.get(url);
  if (hit) return Promise.resolve(hit);

  const pending = inFlight.get(url);
  if (pending) return pending;

  const req = (async (): Promise<Peaks | null> => {
    try {
      const res = await fetch(`/api/audio/peaks?url=${encodeURIComponent(url)}`);
      if (!res.ok) {
        console.warn(`[peaks] ${res.status} for ${url}`);
        return null;
      }
      const data = await res.json() as Peaks;
      if (data.unsupported) {
        console.warn(`[peaks] ${url}: ${data.unsupported}`);
        return null;
      }
      remember(url, data);
      return data;
    } catch (err) {
      console.warn('[peaks] fetch failed for', url, err);
      return null;
    } finally {
      inFlight.delete(url);
    }
  })();

  inFlight.set(url, req);
  return req;
}

/** Drop everything cached. For the reload-after-edit path, where a file keeps
 *  its URL but its contents changed. */
export function forgetPeaks(url?: string): void {
  if (url) cache.delete(url);
  else cache.clear();
}
