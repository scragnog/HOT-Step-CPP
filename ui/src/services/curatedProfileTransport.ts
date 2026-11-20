// The existing endpoint may return JSON or SSE. Preserve both response paths.
// The server stub is under /artists/:id/curated-profile(-stream); the old UI
// path below differs. Keep that existing behavior until the endpoint lands.
export function openCuratedProfile(
  artistId: number,
  selections: { lyrics_set_id: number; song_indices: number[] }[],
  provider: string,
  model?: string,
): Promise<Response> {
  return fetch('/api/lireek/curated-profile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ artist_id: artistId, selections, provider, model }),
  });
}
