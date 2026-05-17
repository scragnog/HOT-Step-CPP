// lrcUtils.ts — Shared LRC parsing and manipulation utilities
//
// Used by: LyricsBar, SectionMarkers, RepaintStudio (RegionLyricsEditor)

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

// ── LRC Parsing ──────────────────────────────────────────────────────────────

/**
 * Parse raw LRC text into timestamped lyric lines.
 * Skips section markers (lines like `[Chorus]`) — use parseSectionMarkers for those.
 */
export function parseLrc(raw: string): LrcLine[] {
  const lines: LrcLine[] = [];
  for (const line of raw.replace(/\r/g, '').split('\n')) {
    const match = line.match(/^\[(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)$/);
    if (match) {
      const mins = parseInt(match[1], 10);
      const secs = parseInt(match[2], 10);
      const cs = match[3] ? parseInt(match[3].padEnd(2, '0').slice(0, 2), 10) : 0;
      const text = match[4].trim();
      if (text && !/^\[.*\]$/.test(text)) {
        lines.push({ time: mins * 60 + secs + cs / 100, text });
      }
    }
  }
  return lines.sort((a, b) => a.time - b.time);
}

/**
 * Parse raw LRC into section markers with corrected start times.
 * Section markers are lines like `[00:45.10] [Chorus]`.
 * The real section start is adjusted to the first lyric line after the marker.
 */
export function parseSectionMarkers(raw: string): SectionMarker[] {
  // Step 1: Parse ALL lines — both section markers and lyric lines
  const allEntries: { time: number; text: string; isSection: boolean }[] = [];
  for (const line of raw.replace(/\r/g, '').split('\n')) {
    const match = line.match(/^\[(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)$/);
    if (match) {
      const mins = parseInt(match[1], 10);
      const secs = parseInt(match[2], 10);
      const cs = match[3] ? parseInt(match[3].padEnd(2, '0').slice(0, 2), 10) : 0;
      const text = match[4].trim();
      const time = mins * 60 + secs + cs / 100;
      const isSection = /^\[.*\]$/.test(text);
      allEntries.push({ time, text, isSection });
    }
  }
  allEntries.sort((a, b) => a.time - b.time);

  // Step 2: Extract section markers with adjusted times
  const sectionIndices: number[] = [];
  for (let i = 0; i < allEntries.length; i++) {
    if (allEntries[i].isSection) sectionIndices.push(i);
  }

  const markers: SectionMarker[] = [];
  for (let si = 0; si < sectionIndices.length; si++) {
    const idx = sectionIndices[si];
    const entry = allEntries[idx];
    // Clean the label: strip brackets, remove style hints after " - "
    let label = entry.text.slice(1, -1);
    const dashIdx = label.indexOf(' - ');
    if (dashIdx !== -1) label = label.slice(0, dashIdx);
    label = label.charAt(0).toUpperCase() + label.slice(1);

    // Find the first actual lyric line between this section and the next section
    const nextSectionIdx = si + 1 < sectionIndices.length
      ? sectionIndices[si + 1]
      : allEntries.length;
    let firstLyricTime: number | null = null;
    for (let j = idx + 1; j < nextSectionIdx; j++) {
      if (!allEntries[j].isSection && allEntries[j].text.length > 0) {
        firstLyricTime = allEntries[j].time;
        break;
      }
    }

    // Use first lyric timestamp if it's significantly later (>1s gap = instrumental lead-in)
    const realTime = (firstLyricTime !== null && firstLyricTime - entry.time > 1)
      ? firstLyricTime
      : entry.time;

    markers.push({ time: realTime, label });
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
  const lines: RegionLrcLine[] = [];
  for (const line of raw.replace(/\r/g, '').split('\n')) {
    const match = line.match(/^\[(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)$/);
    if (match) {
      const mins = parseInt(match[1], 10);
      const secs = parseInt(match[2], 10);
      const cs = match[3] ? parseInt(match[3].padEnd(2, '0').slice(0, 2), 10) : 0;
      const text = match[4].trim();
      if (!text) continue;
      const time = mins * 60 + secs + cs / 100;
      const isSection = /^\[.*\]$/.test(text);
      lines.push({
        time,
        text: isSection ? text : text,
        sectionLabel: isSection ? text.slice(1, -1) : null,
        inRegion: false, // computed later
      });
    }
  }
  return lines.sort((a, b) => a.time - b.time);
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
 * Fetch a companion .lrc file for the given audio URL.
 * Returns the raw LRC text, or null if not found.
 */
export async function fetchLrc(audioUrl: string): Promise<string | null> {
  if (!audioUrl) return null;
  try {
    const lrcUrl = audioUrl.replace(/\.\w+$/, '.lrc');
    const res = await fetch(lrcUrl);
    if (!res.ok) return null;
    const text = await res.text();
    return text.includes('[') ? text : null;
  } catch {
    return null;
  }
}

/**
 * Format seconds as MM:SS for display.
 */
export function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
