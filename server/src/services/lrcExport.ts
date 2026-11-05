// lrcExport.ts — render synced lyrics as a `.lrc` sidecar
//
// The engine writes a `.lrc` beside a render when it produced its own
// alignment, but Whisper retranscription and the YuE2/MM3 aligners write only
// `.lyrics.json`. The timings then exist in the player and nowhere else, which
// is exactly the gap when someone wants the lyrics in an external player, a DAW
// or a dataset. This serializer is shared by the retranscribe route (refresh
// the sidecar it just recomputed) and the download route (export it).
//
// Line format matches what the engine emits and the in-app parser expects:
// `[mm:ss.xx]text`, with section headers as `[mm:ss.xx] [Verse 1]`. The
// enhanced variant writes the A2 word stamps as well:
// `[00:12.00]<00:12.00>Hello <00:12.40>world`.

import fs from 'fs';
import path from 'path';

export interface LrcWordSource {
  word?: string;
  /** Seconds. Word stamps need this; a line without them stays plain. */
  start?: number;
}

export interface LrcLineSource {
  start: number;
  end?: number;
  text?: string;
  section?: string;
  words?: LrcWordSource[];
}

export interface LrcMeta {
  title?: string;
  artist?: string;
}

/** `[mm:ss.xx]`; the minute field grows past 99 instead of wrapping. */
export function formatLrcTimestamp(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const centis = Math.round(safe * 100);
  const mins = Math.floor(centis / 6000);
  const secs = Math.floor((centis % 6000) / 100);
  return `[${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(centis % 100).padStart(2, '0')}]`;
}

const LINE_STAMP = /\[(\d+):(\d+)(?:\.(\d+))?\]/g;
const WORD_STAMP = /<(\d+):(\d+)(?:\.(\d+))?>/g;

function stampSeconds(mins: string, secs: string, frac?: string): number {
  const cs = frac ? parseInt(frac.padEnd(2, '0').slice(0, 2), 10) : 0;
  return parseInt(mins, 10) * 60 + parseInt(secs, 10) + cs / 100;
}

/**
 * Split one raw LRC line into every timestamp it carries and the text after
 * them, or null when the line does not start with a timestamp.
 *
 * The format allows a repeated line to carry several stamps
 * (`[00:12.00][01:20.00]Chorus`), which is how hand-made files usually spell a
 * chorus that comes back. Readers and the crop path have to see every stamp,
 * not just the first, or the rest is rendered as part of the lyric.
 */
export function splitLrcLine(line: string): { times: number[]; text: string } | null {
  LINE_STAMP.lastIndex = 0;
  const times: number[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = LINE_STAMP.exec(line)) !== null) {
    if (match.index !== cursor) break; // stamps must be one contiguous prefix
    times.push(stampSeconds(match[1], match[2], match[3]));
    cursor = match.index + match[0].length;
  }
  if (times.length === 0) return null;
  return { times, text: stripWordStamps(line.slice(cursor)).trim() };
}

/** Enhanced LRC word stamps are timing, not text; readers want the lyric only. */
export function stripWordStamps(text: string): string {
  return text.replace(WORD_STAMP, '');
}

/**
 * Shift the leading `[mm:ss.xx]` stamps of one line, leaving metadata lines and
 * anything without a stamp untouched. Enhanced `<...>` stamps in the text move
 * with it, because they are the same moment written twice.
 */
export function shiftLineStamps(line: string, deltaSec: number): string {
  LINE_STAMP.lastIndex = 0;
  const stamps: string[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = LINE_STAMP.exec(line)) !== null) {
    if (match.index !== cursor) break; // stamps must be one contiguous prefix
    stamps.push(formatLrcTimestamp(stampSeconds(match[1], match[2], match[3]) + deltaSec));
    cursor = match.index + match[0].length;
  }
  if (stamps.length === 0) return line;
  return stamps.join('') + shiftWordStamps(line.slice(cursor), deltaSec);
}

/** Shift the `<mm:ss.xx>` stamps inside a line, leaving everything else alone. */
export function shiftWordStamps(text: string, deltaSec: number): string {
  return text.replace(WORD_STAMP, (_all, mins: string, secs: string, frac?: string) => {
    const stamp = formatLrcTimestamp(Math.max(0, stampSeconds(mins, secs, frac) + deltaSec));
    return `<${stamp.slice(1, -1)}>`;
  });
}

/** A header value cannot carry newlines or brackets without breaking the tag. */
function headerValue(value?: string): string {
  return (value || '').replace(/[\r\n]+/g, ' ').replace(/[\[\]]/g, '').trim();
}

/** Line text: `text` when present, else the words joined back together. */
function lineText(line: LrcLineSource): string {
  const direct = (line.text || '').trim();
  if (direct) return direct;
  return (line.words || [])
    .map(w => (w.word || '').trim())
    .filter(Boolean)
    .join(' ')
    .trim();
}

/** One line's worth of enhanced LRC, or null when it has no usable word times. */
function enhancedLineText(line: LrcLineSource): string | null {
  const words = (line.words || []).filter(w => (w.word || '').trim() && Number.isFinite(w.start));
  if (words.length === 0) return null;
  return words
    .map(w => `<${formatLrcTimestamp(w.start as number).slice(1, -1)}>${(w.word || '').trim()}`)
    .join(' ');
}

/**
 * Build LRC text from timed lines. Untimed/empty lines are dropped rather than
 * emitted as bare timestamps, and a repeated section label is not repeated.
 * `enhanced` adds a `<mm:ss.xx>` stamp before each word, which players that
 * support the extension use for word-level highlighting.
 */
function renderLrc(lines: LrcLineSource[], meta: LrcMeta, enhanced: boolean): string {
  const out: string[] = [];
  const title = headerValue(meta.title);
  const artist = headerValue(meta.artist);
  if (title) out.push(`[ti:${title}]`);
  if (artist) out.push(`[ar:${artist}]`);

  const ordered = [...(lines || [])].sort((a, b) => (a?.start || 0) - (b?.start || 0));
  let lastSection = '';
  for (const line of ordered) {
    if (!line) continue;
    const text = lineText(line);
    const words = enhanced ? enhancedLineText(line) : null;
    if (!text && !words) continue;
    const stamp = formatLrcTimestamp(line.start || 0);
    const section = (line.section || '').trim();
    if (section && section !== lastSection) {
      out.push(`${stamp} [${section}]`);
      lastSection = section;
    }
    out.push(`${stamp}${words ?? text}`);
  }
  return out.length > 0 ? `${out.join('\n')}\n` : '';
}

export function buildLrc(lines: LrcLineSource[], meta: LrcMeta = {}): string {
  return renderLrc(lines, meta, false);
}

/** Enhanced (A2) LRC: word stamps from `.lyrics.json`. Falls back per line when
 *  a line carries no word starts, so a partially timed track still exports. */
export function buildEnhancedLrc(lines: LrcLineSource[], meta: LrcMeta = {}): string {
  return renderLrc(lines, meta, true);
}

/** Build LRC from a parsed `.lyrics.json` payload. */
export function buildLrcFromLyricsJson(json: unknown, meta: LrcMeta = {}): string {
  const lines = (json as { lines?: LrcLineSource[] } | null)?.lines;
  return buildLrc(Array.isArray(lines) ? lines : [], meta);
}

/**
 * Path of `<stem><ext>` for a track.
 *
 * Sidecars are written once, beside the BASE render: mastering and the
 * no-adapter reference re-render the same performance, they do not re-sing it.
 * A naive extension swap on `_mastered.wav` looks for a file nothing writes.
 */
export function sidecarPath(audioDir: string, audioUrl: string, ext: string): string {
  const base = path.basename(String(audioUrl || '').split('?')[0]);
  const stem = base.replace(/(?:_mastered|_noadapter)?\.[^.]+$/, '') || base;
  return path.join(audioDir, stem + ext);
}

/** True when at least one line has a usable word start, i.e. stamps can be written. */
function hasWordStarts(lines: LrcLineSource[]): boolean {
  return lines.some(line =>
    (line.words || []).some(w => (w.word || '').trim() && Number.isFinite(w.start)));
}

/** The parsed `.lyrics.json` beside a track, or null when there is none. */
function readLyricsJson(audioDir: string, audioUrl: string): { lines: LrcLineSource[] } | null {
  try {
    const json = JSON.parse(fs.readFileSync(sidecarPath(audioDir, audioUrl, '.lyrics.json'), 'utf-8'));
    return Array.isArray(json?.lines) ? json : null;
  } catch {
    return null;
  }
}

/**
 * The track's synced lyrics as LRC text: the `.lrc` beside the render when it
 * exists, else rebuilt from the word-level `.lyrics.json`. Null when the track
 * has neither.
 *
 * `enhanced` asks for word stamps. Those only exist in `.lyrics.json`, so an
 * enhanced export is rebuilt even when a plain `.lrc` is on disk, but only when
 * the word timings are actually there: a track whose JSON carries line starts
 * alone keeps the engine's `.lrc`, headers and all.
 */
export function readTrackLrc(
  audioDir: string,
  audioUrl: string,
  meta: LrcMeta = {},
  opts: { enhanced?: boolean } = {},
): { text: string; source: 'lrc' | 'lyrics.json' } | null {
  if (!audioUrl) return null;
  const json = readLyricsJson(audioDir, audioUrl);
  if (opts.enhanced && json && hasWordStarts(json.lines)) {
    const enhanced = buildEnhancedLrc(json.lines, meta);
    if (enhanced.trim()) return { text: enhanced, source: 'lyrics.json' };
  }
  try {
    const text = fs.readFileSync(sidecarPath(audioDir, audioUrl, '.lrc'), 'utf-8');
    if (text.trim()) return { text, source: 'lrc' };
  } catch { /* no LRC beside this render */ }
  if (json) {
    const text = buildLrc(json.lines, meta);
    if (text.trim()) return { text, source: 'lyrics.json' };
  }
  return null;
}
