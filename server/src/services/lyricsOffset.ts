/**
 * lyricsOffset.ts — shift a track's lyric timing by hand.
 *
 * Whisper lands close but not exact, and the error is usually one constant
 * shift: every line a little early or a little late. The UI's nudge buttons
 * move the whole track by a step and persist it here, so the player, the `.lrc`
 * export and the waveform agree afterwards.
 *
 * The `.lyrics.json` word timings are what the player highlights, so when they
 * exist the `.lrc` is rebuilt from them instead of being shifted as a second,
 * possibly stale copy. A track with only an engine `.lrc` has its timestamps
 * moved in place.
 */

import fs from 'fs';
import { buildLrc, shiftLineStamps, sidecarPath } from './lrcExport.js';
import { bumpRevision } from './lyricsRevision.js';

export interface LyricsShiftResult {
  source: 'lyrics.json' | 'lrc';
  lines: number;
  /** The shift applied, in seconds, as the caller gave it. */
  appliedSec: number;
}

/** Times cannot go before the start of the audio. */
function shiftTime(t: number, deltaSec: number): number {
  return Math.max(0, (Number.isFinite(t) ? t : 0) + deltaSec);
}

/** Shift every line and word of a parsed `.lyrics.json`. Returns the line count. */
function shiftJson(data: any, deltaSec: number): number {
  if (!Array.isArray(data?.lines)) return 0;
  data.lines = data.lines.map((line: any) => ({
    ...line,
    start: shiftTime(line?.start, deltaSec),
    ...(typeof line?.end === 'number' ? { end: shiftTime(line.end, deltaSec) } : {}),
    ...(Array.isArray(line?.words)
      ? {
          words: line.words.map((w: any) => ({
            ...w,
            start: shiftTime(w?.start, deltaSec),
            ...(typeof w?.end === 'number' ? { end: shiftTime(w.end, deltaSec) } : {}),
          })),
        }
      : {}),
  }));
  return data.lines.length;
}

/**
 * Shift every timestamp in raw LRC text, in place.
 *
 * Per line rather than through `splitLrcLine`, which strips the enhanced
 * `<mm:ss.xx>` stamps: those have to move with the line, not vanish. Metadata
 * lines (`[ti:]`, `[ar:]`) and anything else without a leading stamp pass
 * through untouched, and the stamp scanner is `lrcExport`'s, so the parsing
 * lives in one place.
 */
function shiftLrcText(text: string, deltaSec: number): string {
  const shifted = text.replace(/\r/g, '').split('\n').map(line => shiftLineStamps(line, deltaSec));
  return `${shifted.join('\n')}\n`;
}

/**
 * Move this track's lyrics by `deltaSec` (positive is later). Null when the
 * track has neither sidecar, so the caller can answer 404 instead of pretending.
 */
export function shiftLyrics(
  audioDir: string,
  audioUrl: string,
  deltaSec: number,
  meta: { title?: string; artist?: string } = {},
): LyricsShiftResult | null {
  if (!audioUrl || !Number.isFinite(deltaSec)) return null;

  const jsonPath = sidecarPath(audioDir, audioUrl, '.lyrics.json');
  if (fs.existsSync(jsonPath)) {
    const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    const lines = shiftJson(data, deltaSec);
    if (!lines) return null;
    bumpRevision(data);
    fs.writeFileSync(jsonPath, JSON.stringify(data, null, 2));
    // The `.lrc` is derived from these timings: rebuild it, do not shift it.
    fs.writeFileSync(sidecarPath(audioDir, audioUrl, '.lrc'), buildLrc(data.lines, meta));
    return { source: 'lyrics.json', lines, appliedSec: deltaSec };
  }

  const lrcPath = sidecarPath(audioDir, audioUrl, '.lrc');
  if (fs.existsSync(lrcPath)) {
    const shifted = shiftLrcText(fs.readFileSync(lrcPath, 'utf-8'), deltaSec);
    fs.writeFileSync(lrcPath, shifted);
    const lines = (shifted.match(/^\[\d/gm) || []).length;
    return { source: 'lrc', lines, appliedSec: deltaSec };
  }

  return null;
}
