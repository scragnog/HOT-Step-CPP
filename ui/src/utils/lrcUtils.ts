// lrcUtils.ts — Shared LRC parsing and manipulation utilities
//
// Used by: LyricsBar, SectionMarkers, RepaintStudio (RegionLyricsEditor)

import { ApiError, HttpStatus } from '../services/httpClient';
import { sidecarUrl } from './wordLrcUtils';

/** A single timestamped lyric line from an LRC file. */
export interface LrcLine {
  time: number;   // seconds
  text: string;
}

/** A section marker (e.g. [Verse 1], [Chorus]) with resolved start time. */
export interface SectionMarker {
  time: number;
  label: string;
}

/** One timestamped entry from an LRC file. */
export interface RawLrcEntry {
  time: number;
  text: string;
  isSection: boolean;
}

// ── LRC Parsing ──────────────────────────────────────────────────────────────

/**
 * Split one raw LRC line into every timestamp it carries and the text after
 * them, or null when the line does not start with a timestamp.
 *
 * A repeated line may carry several stamps (`[00:12.00][01:20.00]Chorus`), and
 * reading only the first both loses the repeat and leaves the other stamps in
 * the displayed text. Enhanced LRC word stamps (`<mm:ss.xx>`) are timing too,
 * so they are dropped from the text here.
 */
export function splitLrcLine(line: string): { times: number[]; text: string } | null {
  const stamp = /\[(\d+):(\d+)(?:\.(\d+))?\]/g;
  const times: number[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = stamp.exec(line)) !== null) {
    if (match.index !== cursor) break; // stamps must be one contiguous prefix
    const cs = match[3] ? parseInt(match[3].padEnd(2, '0').slice(0, 2), 10) : 0;
    times.push(parseInt(match[1], 10) * 60 + parseInt(match[2], 10) + cs / 100);
    cursor = match.index + match[0].length;
  }
  if (times.length === 0) return null;
  const text = line.slice(cursor).replace(/<(\d+):(\d+)(?:\.(\d+))?>/g, '').trim();
  return { times, text };
}

/** Every timestamped entry in an LRC file, one per timestamp, sorted by time. */
export function parseLrcEntries(raw: string): RawLrcEntry[] {
  const entries: RawLrcEntry[] = [];
  for (const line of raw.replace(/\r/g, '').split('\n')) {
    const parsed = splitLrcLine(line);
    if (!parsed || !parsed.text) continue;
    const isSection = /^\[.*\]$/.test(parsed.text);
    for (const time of parsed.times) entries.push({ time, text: parsed.text, isSection });
  }
  return entries.sort((a, b) => a.time - b.time);
}

/**
 * Parse raw LRC text into timestamped lyric lines.
 * Skips section markers (lines like `[Chorus]`) — use parseSectionMarkers for those.
 */
export function parseLrc(raw: string): LrcLine[] {
  return parseLrcEntries(raw)
    .filter(entry => !entry.isSection)
    .map(entry => ({ time: entry.time, text: entry.text }));
}

/** `[Verse 1 - whispered]` -> `Verse 1`: brackets off, style hint dropped. */
function cleanSectionLabel(text: string): string {
  const inner = text.slice(1, -1);
  const dashIdx = inner.indexOf(' - ');
  const base = dashIdx === -1 ? inner : inner.slice(0, dashIdx);
  return base.charAt(0).toUpperCase() + base.slice(1);
}

/** Time of the first lyric line in `[from, to)`, or null when there is none. */
function firstLyricTimeBetween(entries: RawLrcEntry[], from: number, to: number): number | null {
  for (let i = from; i < to; i++) {
    if (!entries[i].isSection && entries[i].text.length > 0) return entries[i].time;
  }
  return null;
}

/**
 * Parse raw LRC into section markers with corrected start times.
 * Section markers are lines like `[00:45.10] [Chorus]`.
 * The real section start is adjusted to the first lyric line after the marker.
 */
export function parseSectionMarkers(raw: string): SectionMarker[] {
  // Step 1: every timestamped entry — section markers and lyric lines alike
  const allEntries = parseLrcEntries(raw);

  // Step 2: Extract section markers with adjusted times
  const sectionIndices: number[] = [];
  for (let i = 0; i < allEntries.length; i++) {
    if (allEntries[i].isSection) sectionIndices.push(i);
  }

  const markers: SectionMarker[] = [];
  for (let si = 0; si < sectionIndices.length; si++) {
    const idx = sectionIndices[si];
    const entry = allEntries[idx];
    const nextSectionIdx = si + 1 < sectionIndices.length
      ? sectionIndices[si + 1]
      : allEntries.length;
    const firstLyricTime = firstLyricTimeBetween(allEntries, idx + 1, nextSectionIdx);

    // A marker well before its first line is an instrumental lead-in: start the
    // section where the singing does (>1 s gap).
    const realTime = firstLyricTime !== null && firstLyricTime - entry.time > 1
      ? firstLyricTime
      : entry.time;

    markers.push({ time: realTime, label: cleanSectionLabel(entry.text) });
  }

  // Step 3: Insert synthetic Intro if the first marker is well after 0:00
  if (markers.length > 0 && markers[0].time > 2) {
    const firstLabel = markers[0].label.toLowerCase();
    if (firstLabel !== 'intro' && firstLabel !== 'introduction') {
      markers.unshift({ time: 0, label: 'Intro' });
    }
  }

  return markers.sort((a, b) => a.time - b.time);
}

// ── LRC Region Extraction ────────────────────────────────────────────────────

/** Parsed LRC line extended with section header info for the region editor. */
export interface RegionLrcLine extends LrcLine {
  /** If this line is a section header (e.g. "Verse 1"), the label text. Null for normal lyric lines. */
  sectionLabel: string | null;
  /** Whether this line falls inside the selected region. */
  inRegion: boolean;
}

/**
 * Parse ALL lines (lyrics + section markers) from raw LRC,
 * marking which ones fall inside the given time region.
 * Used by the RegionLyricsEditor to show context + editable lines.
 */
export function parseAllLrcLines(raw: string): RegionLrcLine[] {
  return parseLrcEntries(raw).map(entry => ({
    time: entry.time,
    text: entry.text,
    sectionLabel: entry.isSection ? entry.text.slice(1, -1) : null,
    inRegion: false, // computed later
  }));
}

/**
 * Mark lines as in/out of region and return all lines with context.
 * Lines inside [startSec, endSec) are marked inRegion=true.
 */
export function markRegion(
  allLines: RegionLrcLine[],
  startSec: number,
  endSec: number,
): RegionLrcLine[] {
  return allLines.map(line => ({
    ...line,
    inRegion: line.time >= startSec && line.time < endSec,
  }));
}

/**
 * Get just the lyric lines (no section markers) within the region.
 */
export function getLyricsInRange(
  lines: LrcLine[],
  startSec: number,
  endSec: number,
): LrcLine[] {
  return lines.filter(l => l.time >= startSec && l.time < endSec);
}

/**
 * Reconstruct full lyrics by splicing edited region lyrics back into the original.
 * Lines outside the region keep their original text.
 * Lines inside the region are replaced with the edited texts (matched by index order).
 */
export function reconstructLyrics(
  allLines: RegionLrcLine[],
  editedTexts: string[],
  startSec: number,
  endSec: number,
): string {
  let editIdx = 0;
  const result: string[] = [];
  for (const line of allLines) {
    if (line.sectionLabel !== null) {
      // Section headers: wrap back in brackets
      result.push(`[${line.sectionLabel}]`);
    } else if (line.time >= startSec && line.time < endSec && editIdx < editedTexts.length) {
      // In-region lyric: use edited text
      result.push(editedTexts[editIdx]);
      editIdx++;
    } else {
      // Out-of-region lyric: keep original
      result.push(line.text);
    }
  }
  return result.join('\n');
}

/**
 * Build a plain lyrics string from LRC lines (no timestamps, with section markers).
 */
export function lrcToPlainLyrics(allLines: RegionLrcLine[]): string {
  return allLines.map(l => l.sectionLabel ? `[${l.sectionLabel}]` : l.text).join('\n');
}

// ── LRC Fetch ────────────────────────────────────────────────────────────────

/**
 * Fetch a companion .lrc file for the given audio URL: the raw LRC text, or
 * null when the track has none. Other failures throw, so Query never caches
 * them as "no lyrics". Read it through the shared `lrc` key
 * (queries/shell-queue/lyrics.ts), not directly.
 */
export async function fetchLrc(audioUrl: string, opts: { signal?: AbortSignal } = {}): Promise<string | null> {
  if (!audioUrl) return null;
  const res = await fetch(sidecarUrl(audioUrl, '.lrc'), { signal: opts.signal });
  if (res.status === HttpStatus.notFound) return null;
  if (!res.ok) throw new ApiError(res.status, `Lyrics fetch failed (${res.status})`, null);
  // A built UI answers a missing /audio file with the SPA's index.html and a
  // 200, so check the type before trusting the body as LRC.
  if (res.headers.get('content-type')?.includes('text/html')) return null;
  const text = await res.text();
  return text.includes('[') ? text : null;
}

/**
 * Format seconds as MM:SS for display.
 */
export function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
