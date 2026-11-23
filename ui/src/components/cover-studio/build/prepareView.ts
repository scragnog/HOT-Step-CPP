// What the Prepare views share: the section spans of a score, its bar clock
// and its short bars, all computed from the ABC alone (no DOM), so they are
// tested in node. PrepareStructure and PrepareLyrics draw them.

import { barMeters } from '@/utils/abc/abcRebar';
import { isVocal, parseScore } from '@/utils/abc/abcSectionMove';
import type { useYue2Prepare } from './useYue2Prepare';
import { formatClock } from '@/utils/format/format';

export type Prepare = ReturnType<typeof useYue2Prepare>;
/** What Hear score plays: bars `from` to `to` (1-based, inclusive). */
export interface ScoreRange { from: number; to: number; loop: boolean }

export interface SectionSpan { id: string; label: string; occurrence: number; repeats: number; start: number; end: number }

/** Each section's id, label, occurrence among sections of the same label and
 *  bar range (1-based, Vocal numbering), or why the score cannot be read. */
export function sectionSpans(abc: string, ids: string[]): SectionSpan[] | string {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return parsed;
  const total = parsed.groups.reduce((n, g) => n + g.bars, 0);
  const marks: Array<{ label: string; start: number | null }> = [];
  for (const item of parsed.items) {
    if (item.kind === 'marker') marks.push({ label: item.text.trim().slice(1).trim(), start: null });
    else if (item.kind === 'group' && marks.length && marks[marks.length - 1].start === null) marks[marks.length - 1].start = item.group.firstBar;
  }
  if (marks.length !== ids.length) return 'A section marker sits above the K: line; fix it in the ABC text.';
  const seen = new Map<string, number>(), repeats = new Map<string, number>();
  for (const m of marks) repeats.set(m.label.toLowerCase(), (repeats.get(m.label.toLowerCase()) ?? 0) + 1);
  return marks.map((m, i) => {
    const key = m.label.toLowerCase(), occurrence = (seen.get(key) ?? 0) + 1;
    seen.set(key, occurrence);
    const start = m.start ?? total + 1, next = marks[i + 1]?.start ?? total + 1;
    return { id: ids[i], label: m.label, occurrence, repeats: repeats.get(key)!, start, end: next - 1 };
  });
}

/** "Chorus" for a one-off, "Chorus (2/3)" for a repeated label. */
export const spanName = (s: SectionSpan) => s.repeats > 1 ? `${s.label} (${s.occurrence}/${s.repeats})` : s.label;
const barsText = (s: SectionSpan) => s.start === s.end ? `${s.start}` : `${s.start}–${s.end}`;
export const spanBars = barsText;
/** m:ss */
export const clock = (seconds: number) => formatClock(seconds);

/** Each bar's start and end in seconds at the score's own M:/Q:, by the Vocal
 *  voice (index 0 is bar 1). This is score time, not recording time. Null when
 *  the score has no readable meter and tempo. */
export function barTimes(abc: string): { start: number[]; end: number[] } | null {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return null;
  let whole = 0, bpm = 0, beat = 0.25;
  const field = (text: string) => {
    const line = text.trim();
    const m = /^M:\s*(?:(\d+)\s*\/\s*(\d+)|(C\|?))\s*$/.exec(line);
    if (m) { whole = m[3] ? 1 : Number(m[1]) / Number(m[2]); return; }
    const q = /^Q:\s*(?:(\d+)\s*\/\s*(\d+)\s*=\s*)?(\d+(?:\.\d+)?)/.exec(line);
    if (q) { beat = q[1] ? Number(q[1]) / Number(q[2]) : 0.25; bpm = Number(q[3]); }
  };
  const start: number[] = [], end: number[] = [];
  let at = 0;
  for (const item of parsed.items) {
    if (item.kind === 'line') { field(item.text); continue; }
    if (item.kind !== 'group') continue;
    const part = item.group.parts.find(p => p.header && isVocal(p.header)) ?? item.group.parts[0];
    part.pre.forEach(field);
    for (const bar of part.bars) for (let i = 0; i < bar.count; i++) {
      if (!whole || !bpm || !beat) return null;
      const length = whole / beat * 60 / bpm;
      start.push(at); end.push(at += length);
    }
  }
  return start.length ? { start, end } : null;
}

/** Seconds from the start of bar `start` to the end of bar `end`, or null when the clock does not cover them. */
export function spanSeconds(map: { start: number[]; end: number[] } | null, start: number, end: number): number | null {
  const from = map?.start[start - 1], to = map?.end[end - 1];
  return from === undefined || to === undefined ? null : to - from;
}

/** Interior bars shorter than the score's usual bar (never bar 1, a pickup,
 *  or the last bar), as 1-based bar numbers. A warning, not an error. */
export function shortBars(abc: string): number[] {
  const meters = barMeters(abc);
  if (!Array.isArray(meters) || meters.length < 3) return [];
  const length = (m: string) => { const [n, d] = m.split('/').map(Number); return n / d; };
  const usual = usualMeter(meters);
  return meters.flatMap((m, i) => i > 0 && i < meters.length - 1 && length(m) < length(usual) - 1e-9 ? [i + 1] : []);
}

/** The meter most bars are in ("2/4"), or null when the score cannot be read. */
export function mainMeter(abc: string): string | null {
  const meters = barMeters(abc);
  return Array.isArray(meters) && meters.length ? usualMeter(meters) : null;
}
function usualMeter(meters: string[]): string {
  const count = new Map<string, number>();
  meters.forEach(m => count.set(m, (count.get(m) ?? 0) + 1));
  return [...count].sort((a, b) => b[1] - a[1])[0][0];
}

// ── Playback ──

/** The slice of abcjs's SynthController the score view drives. At runtime
 *  `pause()` leaves `isStarted` set (only `play()` flips it) and the real
 *  "sounding" flag is the timer's `isRunning`; none of these are in abcjs's typings. */
export interface AbcPlayer {
  isStarted: boolean;
  timer?: { isRunning: boolean };
  play(): Promise<unknown>;
  seek(position: number, units: 'seconds'): void;
}

export const isSounding = (player: AbcPlayer) => player.isStarted && !!player.timer?.isRunning;

/** Stops a sounding player by toggling it off, which leaves it ready for a fresh start. */
export async function stopPlayer(player: AbcPlayer) {
  if (isSounding(player)) await player.play();
}

/** Starts the player, or resumes one that was paused (still `isStarted`, timer idle: toggle off, then on). */
export async function startPlayer(player: AbcPlayer) {
  if (isSounding(player)) return;
  if (player.isStarted) await player.play();
  await player.play();
}

/** One thing sounds at a time, the source recording or the score. `start`
 *  silences the other; `ended` clears the claim when the owner stops by itself. */
export type PlaybackSource = 'recording' | 'score';
export function createPlaybackOwner(stop: Record<PlaybackSource, () => void>) {
  let owner: PlaybackSource | null = null;
  return {
    get current() { return owner; },
    start(next: PlaybackSource) {
      if (owner && owner !== next) stop[owner]();
      owner = next;
    },
    ended(source: PlaybackSource) { if (owner === source) owner = null; },
  };
}
