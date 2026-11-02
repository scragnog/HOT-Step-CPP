// lyrics.ts — a track's lyric sidecars (.lrc, .lyrics.json) for the player,
// and YuE2's .score.abc lead sheet for the song actions menu.
//
// Shared keys: every variant of a track (`_mastered`, `_noadapter`, `?_t=`)
// maps to one sidecar URL, so the lyrics bar and section markers share one
// read per track. A missing sidecar caches null; failures are not cached.
// A crop or retranscription rewrites them: queryManager.sidecars.all.

import { queryOptions } from '@tanstack/react-query';
import { fetchLrc } from '../../utils/lrcUtils';
import { fetchLyricsJson, sidecarUrl } from '../../utils/wordLrcUtils';
import { ApiError, HttpStatus } from '../../services/httpClient';
import { sharedKeys } from '../keys';

export const lrcOptions = (audioUrl: string) => queryOptions({
  queryKey: sharedKeys.lrc(sidecarUrl(audioUrl, '.lrc')),
  queryFn: ({ signal }) => fetchLrc(audioUrl, { signal }),
  enabled: !!audioUrl,
});

export const lyricsJsonOptions = (audioUrl: string) => queryOptions({
  queryKey: sharedKeys.lyricsJson(sidecarUrl(audioUrl, '.lyrics.json')),
  queryFn: ({ signal }) => fetchLyricsJson(audioUrl, { signal }),
  enabled: !!audioUrl,
});


/** The ABC lead sheet YuE2 saved beside the render, or null when the track has
 *  none. A built UI answers a missing /audio file with index.html, so HTML
 *  counts as missing too. */
export async function fetchScoreAbc(audioUrl: string, opts: { signal?: AbortSignal } = {}): Promise<string | null> {
  const res = await fetch(sidecarUrl(audioUrl, '.score.abc'), { signal: opts.signal });
  if (res.status === HttpStatus.notFound) return null;
  if (!res.ok) throw new ApiError(res.status, `Lead sheet fetch failed (${res.status})`, null);
  if (res.headers.get('content-type')?.includes('text/html')) return null;
  return (await res.text()).trim() || null;
}

export const scoreAbcOptions = (audioUrl: string) => queryOptions({
  queryKey: sharedKeys.scoreAbc(sidecarUrl(audioUrl, '.score.abc')),
  queryFn: ({ signal }) => fetchScoreAbc(audioUrl, { signal }),
  enabled: !!audioUrl,
});
