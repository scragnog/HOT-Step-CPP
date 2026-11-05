/**
 * lyricsSplit.ts — give one lyric line a boundary at a point in time.
 *
 * Whisper times a phrase as a unit, so on a slow or rubato passage one line can
 * cover several seconds of singing: the strip shows those words long before they
 * are heard, and inside the line a single syllable may hold the highlight for
 * most of it. There is nothing between "the line started" and "the line ended"
 * that can light up at the right moment.
 *
 * Splitting at the playhead adds the missing boundary. Word times are left
 * exactly as they are — they are the only thing that knows where the syllables
 * really land — and the `.lrc` is rebuilt from them, so the strip, the waveform
 * ticks and an external player all agree on where the second half begins.
 *
 * A track with only an engine `.lrc` has no word times, so its text is divided
 * in proportion to how far into the line the playhead sits.
 */

import fs from 'fs';
import { buildLrc, formatLrcTimestamp, sidecarPath, splitLrcLine } from './lrcExport.js';
import { bumpRevision } from './lyricsRevision.js';

export type SplitFailure = 'no-sidecar' | 'no-line' | 'too-short';

export interface SplitSuccess {
  ok: true;
  source: 'lyrics.json' | 'lrc';
  /** The boundary the caller asked for, in seconds. */
  atSec: number;
  left: string;
  right: string;
  /** Line count after the split. */
  lines: number;
}

export type SplitResult = SplitSuccess | { ok: false; reason: SplitFailure };

interface TimedWord {
  word?: string;
  start?: number;
  end?: number;
}

interface TimedLine {
  start?: number;
  end?: number;
  text?: string;
  section?: string;
  words?: TimedWord[];
}

/** The two halves of a line, whichever way the cut was found. */
interface LineHalves {
  leftText: string;
  rightText: string;
  leftWords?: TimedWord[];
  rightWords?: TimedWord[];
}

const NO_SIDECAR: SplitResult = { ok: false, reason: 'no-sidecar' };
const NO_LINE: SplitResult = { ok: false, reason: 'no-line' };
const TOO_SHORT: SplitResult = { ok: false, reason: 'too-short' };

/** A boundary closer than this to the line's own start is not a split. */
const MIN_OFFSET_SEC = 0.02;
/** Assumed line length when neither an end nor the next start is known. */
const FALLBACK_SPAN_SEC = 4;

const wordText = (w: TimedWord) => (w?.word || '').trim();

/** Index of the last row whose time has started, or -1. Scans back, returns. */
function findLastStarted<T>(rows: T[], startOf: (row: T) => number, atSec: number): number {
  for (let i = rows.length - 1; i >= 0; i--) {
    if (startOf(rows[i]) <= atSec) return i;
  }
  return -1;
}

/** Index of the line the time falls in: the last one that has started. */
function findLineIndex(lines: TimedLine[], atSec: number): number {
  return findLastStarted(lines, line => Number(line?.start) || 0, atSec);
}

/**
 * Divide a line's words in two at `atSec`: everything whose start is before the
 * playhead stays, the rest moves to the new line. Null when one side would be
 * empty, which is a sign the playhead is not inside the line's lyrics.
 */
function splitWords(words: TimedWord[], atSec: number): { left: TimedWord[]; right: TimedWord[] } | null {
  const cut = words.findIndex(w => (Number(w.start) || 0) >= atSec);
  const at = cut < 0 ? words.length : cut;
  if (at <= 0 || at >= words.length) return null;
  return { left: words.slice(0, at), right: words.slice(at) };
}

/** Divide untimed text by where the playhead sits within the line's span. */
function splitWordsByRatio(parts: string[], ratio: number): { left: string[]; right: string[] } | null {
  if (parts.length < 2) return null;
  const at = Math.min(parts.length - 1, Math.max(1, Math.round(ratio * parts.length)));
  return { left: parts.slice(0, at), right: parts.slice(at) };
}

/**
 * With real word times, only a word boundary is a split. Falling back to a
 * proportional cut here would put the new line after its own syllable started,
 * which is the drift this feature exists to remove.
 */
function halvesByWords(words: TimedWord[], atSec: number): LineHalves | null {
  const halves = splitWords(words, atSec);
  if (!halves) return null;
  return {
    leftWords: halves.left,
    rightWords: halves.right,
    leftText: halves.left.map(wordText).filter(Boolean).join(' '),
    rightText: halves.right.map(wordText).filter(Boolean).join(' '),
  };
}

/** No usable word times: divide the text by how far into the line we are. */
function halvesByRatio(line: TimedLine, next: TimedLine | undefined, atSec: number): LineHalves | null {
  const start = Number(line?.start) || 0;
  const end = Number(line.end);
  const nextStart = Number(next?.start);
  const span = Number.isFinite(end) && end > start
    ? end - start
    : (Number.isFinite(nextStart) && nextStart > start ? nextStart - start : FALLBACK_SPAN_SEC);

  const parts = (line.text || '').trim().split(/\s+/).filter(Boolean);
  const byRatio = splitWordsByRatio(parts, Math.max(0, Math.min(1, (atSec - start) / span)));
  if (!byRatio) return null;
  return { leftText: byRatio.left.join(' '), rightText: byRatio.right.join(' ') };
}

function splitJson(data: any, atSec: number): SplitResult {
  if (!Array.isArray(data?.lines)) return NO_LINE;
  const lines: TimedLine[] = data.lines;
  const idx = findLineIndex(lines, atSec);
  if (idx < 0) return NO_LINE;

  const line = lines[idx];
  const start = Number(line?.start) || 0;
  if (atSec - start < MIN_OFFSET_SEC) return TOO_SHORT;

  const words = (line.words || []).filter(w => wordText(w) || Number.isFinite(w?.start));
  const halves = words.length >= 2
    ? halvesByWords(words, atSec)
    : halvesByRatio(line, lines[idx + 1], atSec);
  if (!halves) return TOO_SHORT;
  if (!halves.leftText || !halves.rightText) return TOO_SHORT;

  const left: TimedLine = {
    ...line,
    end: atSec,
    text: halves.leftText,
    ...(halves.leftWords ? { words: halves.leftWords } : {}),
  };
  const right: TimedLine = {
    ...line,
    start: atSec,
    text: halves.rightText,
    ...(halves.rightWords ? { words: halves.rightWords } : {}),
  };
  // The second half belongs to the same section; buildLrc drops the repeat.
  data.lines.splice(idx, 1, left, right);
  return {
    ok: true,
    source: 'lyrics.json',
    atSec,
    left: halves.leftText,
    right: halves.rightText,
    lines: data.lines.length,
  };
}

/**
 * Split an LRC line in place. A repeated line keeps its other stamps on the
 * first half, which is where they already were; there is no word data to move.
 */
function splitLrcFile(lrcPath: string, atSec: number): SplitResult {
  const raw = fs.readFileSync(lrcPath, 'utf-8').replace(/\r/g, '');
  const rows = raw.split('\n');
  const parsed = rows.map(line => splitLrcLine(line));
  const lyricRows = parsed
    .map((p, i) => ({ p, i }))
    .filter(r => r.p && r.p.text && !/^\[.*\]$/.test(r.p.text))
    .map(r => ({ index: r.i, times: r.p!.times, text: r.p!.text }));

  const pick = findLastStarted(lyricRows, row => row.times[0], atSec);
  if (pick < 0) return NO_LINE;

  const row = lyricRows[pick];
  const rowTime = row.times[0];
  if (atSec - rowTime < MIN_OFFSET_SEC) return TOO_SHORT;

  const next = lyricRows[pick + 1]?.times[0];
  const span = next !== undefined && next > rowTime ? next - rowTime : FALLBACK_SPAN_SEC;
  const parts = row.text.split(/\s+/).filter(Boolean);
  const byRatio = splitWordsByRatio(parts, Math.max(0, Math.min(1, (atSec - rowTime) / span)));
  if (!byRatio) return TOO_SHORT;

  const left = byRatio.left.join(' ');
  const right = byRatio.right.join(' ');
  const prefix = rows[row.index].match(/^(?:\[\d+:\d+(?:\.\d+)?\])+/)?.[0]
    || formatLrcTimestamp(rowTime);

  rows[row.index] = `${prefix}${left}`;
  rows.splice(row.index + 1, 0, `${formatLrcTimestamp(atSec)}${right}`);
  fs.writeFileSync(lrcPath, rows.join('\n'));

  return { ok: true, source: 'lrc', atSec, left, right, lines: lyricRows.length + 1 };
}

/**
 * Split this track's line at `atSec` (seconds, positive). Word timings move to
 * the half they belong to and the `.lrc` is rebuilt, so the export matches.
 */
export function splitLyricLine(
  audioDir: string,
  audioUrl: string,
  atSec: number,
  meta: { title?: string; artist?: string } = {},
): SplitResult {
  if (!audioUrl || !Number.isFinite(atSec) || atSec <= 0) return NO_LINE;

  const jsonPath = sidecarPath(audioDir, audioUrl, '.lyrics.json');
  if (fs.existsSync(jsonPath)) {
    const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    const result = splitJson(data, atSec);
    if (!result.ok) return result;
    bumpRevision(data);
    fs.writeFileSync(jsonPath, JSON.stringify(data, null, 2));
    fs.writeFileSync(sidecarPath(audioDir, audioUrl, '.lrc'), buildLrc(data.lines, meta));
    return result;
  }

  const lrcPath = sidecarPath(audioDir, audioUrl, '.lrc');
  if (fs.existsSync(lrcPath)) return splitLrcFile(lrcPath, atSec);

  return NO_SIDECAR;
}
