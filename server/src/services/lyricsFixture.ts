// lyricsFixture.ts — hand-checked timings, and the error a run has against them.
//
// The scope document's evaluation plan is a fixture: place about thirty words a
// track by ear in the editor, save the ticked lines as ground truth, then score
// every later run against them. This file is the second half of that loop. It is pure,
// so the numbers it prints can be trusted without a browser, a model or an
// engine, and so the publishing checks can call it directly.
//
// What it deliberately does NOT do is call Needleman-Wunsch. Both sides are the
// same lyric text: the fixture is the sidecar a person placed, and the candidate
// is what a run produced from the same words. The only difference to absorb is a
// word a run dropped or invented, so the pairing looks a few words ahead and
// resyncs. A general sequence alignment would hide exactly the failure this is
// meant to measure — a whole line timed somewhere else.

import type { LyricsLine } from './lyricsReconcile.js';
import { pairByText } from './textPairing.js';

export interface LyricsFixture {
  version: 1;
  audioUrl: string;
  /** ISO time the fixture was taken, so a later reader knows how old it is. */
  createdAt: string;
  /**
   * The lines exactly as the editor had them, and only the ones a person
   * checked by ear. `handCheckedLines` is the filter that guarantees it.
   */
  lines: LyricsLine[];
}

/**
 * The only lines a fixture may contain: the ones a person listened to.
 *
 * A sidecar the machine produced and a person skimmed is not ground truth —
 * every later score would be mostly agreement with the previous run, and a
 * machine error would become the reference. The editor marks a line `checked`
 * when someone places it by ear; everything else stays out of the store.
 */
export function handCheckedLines(lines: LyricsLine[]): LyricsLine[] {
  return lines.filter(line => line?.checked === true);
}

export interface FixtureSectionOffset {
  section: string;
  /** 1-based occurrence of this section in the track. */
  occurrence: number;
  words: number;
  medianMs: number;
}

export interface FixtureScore {
  /** Words the fixture has, and how many of them the candidate matched. */
  words: number;
  matched: number;
  /** Fixture words left with no candidate word to pair with. */
  missing: number;
  /** Candidate onset error against the fixture, in milliseconds. */
  medianMs: number;
  p90Ms: number;
  maxMs: number;
  /** Share of matched words inside each tolerance, 0 to 1. */
  within100: number;
  within250: number;
  /** Candidate words that start before the previous one ended. */
  monotonicViolations: number;
  /** |offset of a section's second occurrence - its first|. A global-alignment
   *  failure shows up here long before the median moves. Null when no section
   *  repeats. */
  secondChorusOffsetMs: number | null;
  sections: FixtureSectionOffset[];
}

const MONOTONIC_EPS_SEC = 1e-6;

interface FlatWord {
  word: string;
  start: number;
  end: number;
  /** Index into the lines array the word came from. */
  line: number;
}

/** Every word in reading order, with the line it belongs to. */
function flatten(lines: LyricsLine[]): FlatWord[] {
  const out: FlatWord[] = [];
  lines.forEach((line, li) => (line.words ?? []).forEach(word => {
    out.push({ word: word.word, start: word.start, end: word.end, line: li });
  }));
  return out;
}

/** Nearest-rank percentile of an ascending array. */
function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

/**
 * A section's lines, one entry per occurrence.
 *
 * A label is written only on the line where a section STARTS, and it is written
 * again when the section comes back — that is exactly how a second chorus
 * announces itself. So every labelled line opens a new run, and the unlabelled
 * lines below it belong to that run until the next label.
 */
function sectionRuns(lines: LyricsLine[]): Array<{ section: string; from: number; to: number }> {
  const runs: Array<{ section: string; from: number; to: number }> = [];
  lines.forEach((line, li) => {
    const label = (line.section || '').trim();
    if (label) {
      runs.push({ section: label, from: li, to: li });
      return;
    }
    const last = runs.at(-1);
    if (last) last.to = li;
  });
  return runs;
}

export function scoreFixture(fixture: LyricsFixture, candidate: LyricsLine[]): FixtureScore {
  const ideal = flatten(fixture.lines);
  const got = flatten(candidate);
  const pairs = pairByText(ideal, got);

  const absolute = pairs
    .map(([a, b]) => Math.abs(b.start - a.start) * 1000)
    .sort((x, y) => x - y);
  const matched = pairs.length;

  let monotonicViolations = 0;
  let cursor = -Infinity;
  for (const word of got) {
    if (word.start + MONOTONIC_EPS_SEC < cursor) monotonicViolations++;
    cursor = Math.max(cursor, word.end);
  }

  const runs = sectionRuns(fixture.lines);
  const sections: FixtureSectionOffset[] = runs.map(run => {
    const offsets = pairs
      .filter(([a]) => a.line >= run.from && a.line <= run.to)
      .map(([a, b]) => (b.start - a.start) * 1000)
      .sort((x, y) => x - y);
    const occurrence = runs.filter(other => other.section === run.section && other.from <= run.from).length;
    return {
      section: run.section,
      occurrence,
      words: offsets.length,
      medianMs: Math.round(percentile(offsets, 50)),
    };
  });

  // The first section that appears twice: its two occurrences' medians are the
  // cheapest read on whether the run followed the track or drifted.
  const repeated = sections.filter(entry => entry.words > 0).reduce<FixtureSectionOffset[][]>((groups, entry) => {
    const group = groups.find(list => list[0].section === entry.section);
    if (group) group.push(entry);
    else groups.push([entry]);
    return groups;
  }, []).find(group => group.length >= 2) ?? null;
  const secondChorusOffsetMs = repeated
    ? Math.abs(repeated[1].medianMs - repeated[0].medianMs)
    : null;

  return {
    words: ideal.length,
    matched,
    missing: Math.max(0, ideal.length - matched),
    medianMs: Math.round(percentile(absolute, 50)),
    p90Ms: Math.round(percentile(absolute, 90)),
    maxMs: Math.round(absolute.at(-1) ?? 0),
    within100: matched ? absolute.filter(ms => ms <= 100).length / matched : 0,
    within250: matched ? absolute.filter(ms => ms <= 250).length / matched : 0,
    monotonicViolations,
    secondChorusOffsetMs,
    sections,
  };
}
