// wordLrcUtils.ts — Parse .lyrics.json and find active word/line by time

export interface LyricsWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
  source: 'matched' | 'whisper' | 'ad-lib';
}

export interface LyricsLine {
  start: number;
  end: number;
  text: string;
  words: LyricsWord[];
  section?: string;
}

export interface LyricsJson {
  version: number;
  method: string;
  whisperModel: string;
  vocalsIsolated: boolean;
  lines: LyricsLine[];
}

/**
 * The URL of a sidecar (`.lyrics.json`, `.lrc`) for a track.
 *
 * Sidecars are written once, beside the BASE render, and the derived files
 * share its timeline: mastering and the no-adapter reference re-render the
 * same performance, they do not re-sing it. The player, though, is playing
 * whichever variant is selected — `<uuid>_mastered.wav` whenever
 * post-processing produced one — and a naive extension swap then asks for
 * `<uuid>_mastered.lyrics.json`, which nothing ever writes.
 *
 * That failure is silent rather than loud: the server answers a missing file
 * under /audio with the SPA's index.html and a 200, so `res.ok` is true and
 * the caller parses HTML as JSON. Strip the variant suffix instead, and the
 * lyrics bar works on every variant of every backend.
 *
 * The query string goes too — playbackStore cache-busts audio URLs with
 * `?_t=`, which would otherwise defeat the extension match entirely.
 */
export function sidecarUrl(audioUrl: string, suffix: string): string {
  return audioUrl.split('?')[0].replace(/(?:_mastered|_noadapter)?\.\w+$/, suffix);
}

/** Fetch .lyrics.json for the given audio URL. Returns null if not found. */
export async function fetchLyricsJson(audioUrl: string): Promise<LyricsJson | null> {
  if (!audioUrl) return null;
  try {
    const res = await fetch(sidecarUrl(audioUrl, '.lyrics.json'));
    if (!res.ok) return null;
    const data = await res.json();
    return data?.version && data?.lines ? data : null;
  } catch {
    // Includes the HTML-instead-of-JSON case above: a missing sidecar answers
    // 200 with the SPA shell, and .json() throws on it.
    return null;
  }
}

/** Find the index of the current line at the given time. */
export function findCurrentLineIndex(lines: LyricsLine[], time: number): number {
  if (lines.length === 0) return -1;
  let result = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].start <= time) result = i;
    else break;
  }
  if (result >= 0 && time <= lines[result].end + 2.0) return result;
  return result;
}

/** Find the index of the active word within a line at the given time. */
export function findActiveWordIndex(words: LyricsWord[], time: number): number {
  if (words.length === 0) return -1;
  for (let i = words.length - 1; i >= 0; i--) {
    if (time >= words[i].start) return i;
  }
  return -1;
}
