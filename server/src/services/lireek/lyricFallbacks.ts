export type LyricSource = 'genius' | 'hf-genius' | 'lrclib';

export interface SourcedLyrics {
  title: string;
  lyrics: string;
  url: string;
  source: LyricSource;
}

const DATASET = 'theelderemo/genius-lyrics-cleaned';

function normalize(value: string): string {
  return value.toLowerCase().normalize('NFD').replace(/\p{M}/gu, '')
    .replace(/\s*&\s*/g, ' and ').replace(/[^\p{L}\p{N}]/gu, '');
}

// The dataset also contains names whose accented letters were deleted entirely.
function agrees(requested: string, actual: unknown, hf = false): boolean {
  if (typeof actual !== 'string' || !normalize(requested)) return false;
  return normalize(requested) === normalize(actual)
    || (hf && !!normalize(requested.replace(/[^\x00-\x7f]/g, ''))
      && normalize(requested.replace(/[^\x00-\x7f]/g, '')) === normalize(actual));
}

/** Bounded hosted lookups; a failing HF index gets a one-minute cooldown. */
export function createLyricFallbacks(request: typeof fetch = (...args) => fetch(...args)) {
  let hfRetryAt = 0;
  return async (artist: string, title: string): Promise<SourcedLyrics | null> => {
    if (!artist.trim() || !title.trim()) return null;
    if (Date.now() >= hfRetryAt) {
      try {
        const query = new URLSearchParams({ dataset: DATASET, config: 'default', split: 'train', query: `${artist} ${title}`, length: '30' });
        const response = await request(`https://datasets-server.huggingface.co/search?${query}`, {
          signal: AbortSignal.timeout(4000),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json() as { rows?: { row_idx: number; row: Record<string, unknown> }[] };
        for (const { row, row_idx } of data.rows ?? []) {
          if (agrees(artist, row.artist, true) && agrees(title, row.title, true)
            && typeof row.lyrics === 'string' && row.lyrics.trim()) {
            return { title: row.title as string, lyrics: row.lyrics.trim(), source: 'hf-genius',
              url: `https://huggingface.co/datasets/${DATASET}/viewer/default/train?row=${row_idx}` };
          }
        }
      } catch (error) {
        hfRetryAt = Date.now() + 60_000;
        console.warn('[Lyrics] HF index unavailable; trying LRCLIB:', (error as Error).message);
      }
    }
    try {
      const query = new URLSearchParams({ artist_name: artist, track_name: title });
      const response = await request(`https://lrclib.net/api/get?${query}`, {
        signal: AbortSignal.timeout(8000), headers: { 'User-Agent': 'HOT-Step lyrics lookup' },
      });
      if (!response.ok) return null;
      const row = await response.json() as Record<string, unknown>;
      if (!row.instrumental && agrees(artist, row.artistName) && agrees(title, row.trackName)
        && typeof row.plainLyrics === 'string' && row.plainLyrics.trim() && Number.isSafeInteger(row.id)) {
        return { title: row.trackName as string, lyrics: row.plainLyrics.trim(), source: 'lrclib',
          url: `https://lrclib.net/api/get/${row.id}` };
      }
    } catch (error) {
      console.warn('[Lyrics] LRCLIB unavailable:', (error as Error).message);
    }
    return null;
  };
}

export const lookupFallbackLyrics = createLyricFallbacks();
