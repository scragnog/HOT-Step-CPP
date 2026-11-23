/**
 * lyricTimings.ts — persist hand-placed word timings.
 *
 * Whisper's alignment is a good starting point and a bad ending point: a line
 * that covers two phrases, or a syllable it placed half a second early, needs a
 * person with the audio in front of them. The lyric editor drags those words on
 * a zoomed timeline and saves the result here.
 *
 * The editor sends the whole line array, so this is a replace rather than a
 * patch: lines and words are sorted, malformed entries are dropped, and each
 * line's start/end is recomputed from its words. That last part matters — a line
 * whose start sits before its first syllable is exactly the "it lights up too
 * early" complaint, and it cannot be fixed while the old start survives.
 *
 * The `.lrc` is rebuilt from the new timings so the export, the strip and the
 * waveform ticks all read the same numbers.
 */

import fs from 'fs';
import { buildLrc, sidecarPath } from './lrcExport.js';
import { bumpRevision, revisionOf } from './lyricsRevision.js';
import type { VocalLayer } from './lyricsReconcile.js';

/** Guard rails: a whole song's worth, with room to spare. */
const MAX_LINES = 2000;
const MAX_WORDS = 20_000;
/** Shortest word span we will write, so nothing becomes zero-width. */
const MIN_WORD_SEC = 0.05;
/** Only these layers are written; an unknown string is dropped like a bad time. */
const LAYERS: VocalLayer[] = ['lead', 'backing', 'adlib'];
const asLayer = (value: unknown): VocalLayer | undefined =>
  (LAYERS.includes(value as VocalLayer) ? value as VocalLayer : undefined);

export interface SavedTimingWord {
  word: string;
  start: number;
  end: number;
  confidence?: number;
  source?: string;
  layer?: VocalLayer;
}

export interface SavedTimingLine {
  start: number;
  end: number;
  text: string;
  section?: string;
  /** Placed by ear, so the fixture store may keep it. See `lyricsFixture`. */
  checked?: boolean;
  /** The stream this line belongs to; a backing line may overlap the lead. */
  layer?: VocalLayer;
  words: SavedTimingWord[];
}

export type SaveTimingsResult =
  | { ok: true; lines: number; words: number; revision: number }
  | { ok: false; reason: 'no-sidecar' | 'invalid' }
  | { ok: false; reason: 'stale'; revision: number };

const round3 = (n: number) => Math.round(n * 1000) / 1000;

function cleanWord(raw: any): SavedTimingWord | null {
  const word = typeof raw?.word === 'string' ? raw.word.trim() : '';
  const start = Number(raw?.start);
  if (!word || !Number.isFinite(start) || start < 0) return null;
  const end = Number(raw?.end);
  const confidence = Number(raw?.confidence);
  const layer = asLayer(raw?.layer);
  return {
    word,
    start: round3(start),
    end: round3(Number.isFinite(end) && end > start ? end : start + 0.2),
    ...(Number.isFinite(confidence) ? { confidence } : {}),
    ...(typeof raw?.source === 'string' ? { source: raw.source } : {}),
    ...(layer ? { layer } : {}),
  };
}

/**
 * Coerce whatever arrived into lines we are willing to write. Null means the
 * payload as a whole is unusable; individual broken entries are dropped.
 */
export function normalizeTimingLines(input: unknown): SavedTimingLine[] | null {
  if (!Array.isArray(input) || input.length === 0 || input.length > MAX_LINES) return null;

  const out: SavedTimingLine[] = [];
  let wordCount = 0;

  for (const raw of input) {
    const words: SavedTimingWord[] = [];
    if (Array.isArray(raw?.words)) {
      for (const w of raw.words) {
        if (wordCount >= MAX_WORDS) break;
        const clean = cleanWord(w);
        if (!clean) continue;
        words.push(clean);
        wordCount++;
      }
    }
    // A line that carries text but no word times is real: an aligner can time
    // lines without placing words. Dropping it here used to erase it on the
    // next save, so it is kept verbatim and left untimed.
    if (words.length === 0) {
      const text = typeof raw?.text === 'string' ? raw.text.trim() : '';
      const rawStart = Number(raw?.start);
      if (!text || !Number.isFinite(rawStart) || rawStart < 0) continue;
      const rawEnd = Number(raw?.end);
      const section = typeof raw?.section === 'string' ? raw.section.trim() : '';
      const layer = asLayer(raw?.layer);
      out.push({
        start: round3(rawStart),
        end: round3(Number.isFinite(rawEnd) && rawEnd > rawStart ? rawEnd : rawStart + 0.2),
        text,
        ...(section ? { section } : {}),
        ...(raw?.checked === true ? { checked: true } : {}),
        ...(layer ? { layer } : {}),
        words: [],
      });
      continue;
    }
    // Sorted before the overlap repair, so an out-of-order word is moved into
    // place rather than deleted.
    words.sort((a, b) => a.start - b.start);
    // No overlaps, ever: each word starts no earlier than the previous one ended
    // and keeps a minimum width. A drag that landed on top of a neighbour is
    // stored as the nearest legal position, so nobody can produce two words
    // fighting for the same instant — the editor or another client.
    let cursor = 0;
    for (const w of words) {
      if (w.start < cursor) w.start = round3(cursor);
      if (w.end < w.start + MIN_WORD_SEC) w.end = round3(w.start + MIN_WORD_SEC);
      cursor = w.end;
    }

    const text = words.map(w => w.word).join(' ').trim();
    const start = words[0].start;
    const end = Math.max(words[words.length - 1].end, start + MIN_WORD_SEC);
    const section = typeof raw?.section === 'string' ? raw.section.trim() : '';
    // `checked` rides along untouched: it is a note about the line, not about
    // its times, so sorting and the overlap repair must not clear it.
    const layer = asLayer(raw?.layer);
    out.push({
      start, end, text,
      ...(section ? { section } : {}),
      ...(raw?.checked === true ? { checked: true } : {}),
      ...(layer ? { layer } : {}),
      words,
    });
  }

  if (out.length === 0) return null;
  out.sort((a, b) => a.start - b.start);
  return out;
}

/**
 * Replace this track's lyric lines with `input`.
 *
 * Creates the `.lyrics.json` when the track had none but has a line-level
 * `.lrc`: that is the editor's "use the lyrics I already have" path, where the
 * line starts in the `.lrc` are what the words were placed against. `no-sidecar`
 * now means neither file exists — a timeline invented from a request alone, with
 * nothing on disk to place it against, is still refused.
 */
export function saveLyricTimings(
  audioDir: string,
  audioUrl: string,
  input: unknown,
  meta: { title?: string; artist?: string } = {},
  expectedRevision?: number,
): SaveTimingsResult {
  const jsonPath = sidecarPath(audioDir, audioUrl, '.lyrics.json');
  if (!audioUrl) return { ok: false, reason: 'no-sidecar' };

  const exists = fs.existsSync(jsonPath);
  if (!exists && !fs.existsSync(sidecarPath(audioDir, audioUrl, '.lrc'))) {
    return { ok: false, reason: 'no-sidecar' };
  }

  const data = exists
    ? JSON.parse(fs.readFileSync(jsonPath, 'utf-8'))
    // A new file starts at revision 0, which is what the editor sends when it
    // opened without a sidecar: the first save lands, the second is checked.
    : { version: 1, method: 'lrc-lines', vocalsIsolated: false };
  const revision = revisionOf(data);
  // The editor sends the revision it loaded. A mismatch means a crop, another
  // window or a retranscription moved the file on, and writing now would undo it.
  if (expectedRevision !== undefined && Number(expectedRevision) !== revision) {
    return { ok: false, reason: 'stale', revision };
  }

  const lines = normalizeTimingLines(input);
  if (!lines) return { ok: false, reason: 'invalid' };

  data.version = typeof data.version === 'number' ? data.version : 1;
  data.lines = lines;
  data.editedByHand = true;
  bumpRevision(data);
  fs.writeFileSync(jsonPath, JSON.stringify(data, null, 2));
  fs.writeFileSync(sidecarPath(audioDir, audioUrl, '.lrc'), buildLrc(lines, meta));

  return {
    ok: true,
    lines: lines.length,
    words: lines.reduce((n, l) => n + l.words.length, 0),
    revision: revisionOf(data),
  };
}
