// peaksClient.ts — waveform envelopes, fetched from the server and cached.
//
// The server computes these once per file (see server/src/services/audio/peaks.ts)
// so the browser never downloads or decodes audio to draw a waveform. A payload
// is ~25 KB against the 48–96 MB the file itself weighs.

import { queryOptions } from '@tanstack/react-query';
import { queryClient } from '../queries/client';
import { resourceKey, sharedKeys, SHARED } from '../queries/keys';
import { ApiError } from '../services/httpClient';

export interface Peaks {
  version: number;
  duration: number;
  sampleRate: number;
  channels: number;
  min: number[];
  max: number[];
  unsupported?: string;
}

// Query owns the cache (shared key `peaks`, one entry per audio URL), so
// concurrent calls for a URL share one request and failures are never cached.
// Crop, fade and post-process passes rewrite a file under the same URL, so
// every load asks the server again, past the browser's HTTP cache too. The
// server keeps its own copy per file version, so an unchanged file is a 304.
// The cached entry only paints the first frame while that answer is in flight.
const peaksOptions = (url: string) => queryOptions({
  queryKey: sharedKeys.peaks(url),
  queryFn: async ({ signal }) => {
    const res = await fetch(`/api/audio/peaks?url=${encodeURIComponent(url)}`, { signal, cache: 'no-cache' });
    if (!res.ok) throw new ApiError(res.status, `peaks ${res.status} for ${url}`, null);
    return await res.json() as Peaks;
  },
  staleTime: 0,
  gcTime: 30 * 60_000,
});

const usable = (peaks: Peaks | undefined): Peaks | null => (peaks && !peaks.unsupported ? peaks : null);

/** Already-fetched peaks for this URL, or null. Synchronous, for the paint
 *  that happens before the fetch resolves. */
export function peaksCached(url: string): Peaks | null {
  return usable(queryClient.getQueryData<Peaks>(sharedKeys.peaks(url)));
}

/**
 * Peaks for one audio URL. Returns null when the server could not produce
 * them, which callers must treat as "draw nothing" rather than as a failure to
 * play — a missing waveform is cosmetic.
 */
export async function fetchPeaks(url: string): Promise<Peaks | null> {
  try {
    const data = await queryClient.query(peaksOptions(url));
    if (data.unsupported) console.warn(`[peaks] ${url}: ${data.unsupported}`);
    return usable(data);
  } catch (err) {
    console.warn('[peaks] fetch failed for', url, err);
    return null;
  }
}

/** Drop cached peaks, for one URL or all. For the reload-after-edit path,
 *  where a file keeps its URL but its contents changed. */
export function forgetPeaks(url?: string): void {
  queryClient.removeQueries({ queryKey: url ? sharedKeys.peaks(url) : resourceKey(SHARED.peaks) });
}
