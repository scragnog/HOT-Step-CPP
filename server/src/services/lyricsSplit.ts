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

const NO_SIDECAR: SplitResult = { ok: false, reason: 'no-sidecar' };
const NO_LINE: SplitResult = { ok: false, reason: 'no-line' };
const TOO_SHORT: SplitResult = { ok: false, reason: 'too-short' };

/** A boundary closer than this to the line's own start is not a split. */
const MIN_OFFSET_SEC = 0.02;

const wordText = (w: TimedWord) => (w?.word || '').trim();

/** Index of the line the time falls in: the last one that has started. */
function findLineIndex(lines: TimedLine[], atSec: number): number {
  let idx = -1;
  for (let i = 0; i < lines.length; i++) {
    if ((Number(lines[i]?.start) || 0) <= atSec) idx = i;
    else break;
  }
  return idx;
}

/**
 * Divide a line's words in two at `atSec`: everything whose start is before the
 * playhead stays, the rest moves to the new line. Returns null when one side
 * would be empty, which is a sign the playhead is not inside the line's lyrics.
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

function splitJson(data: any, atSec: number): SplitResult {
  if (!Array.isArray(data?.lines)) return NO_LINE;
  const lines: TimedLine[] = data.lines;
  const idx = findLineIndex(lines, atSec);
  if (idx < 0) return NO_LINE;

  const line = lines[idx];
  const start = Number(line?.start) || 0;
  if (atSec - start < MIN_OFFSET_SEC) return TOO_SHORT;

  const words = (line.words || []).filter(w => wordText(w) || Number.isFinite(w?.start));
  let leftText: string;
  let rightText: string;
  let leftWords: TimedWord[] | undefined;
  let rightWords: TimedWord[] | undefined;

  // With real word times, only a word boundary is a split. Falling back to a
  // proportional cut here would put the new line after its own syllable started,
  // which is the drift this feature exists to remove.
  const hasWords = words.length >= 2;
  const byWord = hasWords ? splitWords(words, atSec) : null;
  if (hasWords) {
    if (!byWord) return TOO_SHORT;
    leftWords = byWord.left;
    rightWords = byWord.right;
    leftText = byWord.left.map(wordText).filter(Boolean).join(' ');
    rightText = byWord.right.map(wordText).filter(Boolean).join(' ');
  } else {
    // No usable word times: divide the text by how far into the line we are.
    const parts = (line.text || '').trim().split(/\s+/).filter(Boolean);
    const nextStart = Number(lines[idx + 1]?.start);
    const end = Number(line.end);
    const span = Number.isFinite(end) && end > start
      ? end - start
      : (Number.isFinite(nextStart) && nextStart > start ? nextStart - start : 4);
    const byRatio = splitWordsByRatio(parts, Math.max(0, Math.min(1, (atSec - start) / span)));
    if (!byRatio) return TOO_SHORT;
    leftText = byRatio.left.join(' ');
    rightText = byRatio.right.join(' ');
  }

  if (!leftText || !rightText) return TOO_SHORT;

  const left: TimedLine = {
    ...line,
    end: atSec,
    text: leftText,
    ...(leftWords ? { words: leftWords } : {}),
  };
  const right: TimedLine = {
    ...line,
    start: atSec,
    text: rightText,
    ...(rightWords ? { words: rightWords } : {}),
  };
  // The second half belongs to the same section; buildLrc drops the repeat.
  data.lines.splice(idx, 1, left, right);
  return { ok: true, source: 'lyrics.json', atSec, left: leftText, right: rightText, lines: data.lines.length };
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

  let pick = -1;
  for (let k = 0; k < lyricRows.length; k++) {
    if (lyricRows[k].times[0] <= atSec) pick = k;
    else break;
  }
  if (pick < 0) return NO_LINE;

  const row = lyricRows[pick];
  const rowTime = row.times[0];
  if (atSec - rowTime < MIN_OFFSET_SEC) return TOO_SHORT;

  const next = lyricRows[pick + 1]?.times[0];
  const span = next !== undefined && next > rowTime ? next - rowTime : 4;
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
