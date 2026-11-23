// Moving a score section's start to another bar by rewriting the ABC.
//
// A transcribed score's body is a run of groups: one `V:` line and one line
// of bars per voice, all covering the same bars, with each `% label` section
// marker sitting between two groups. Moving a section to bar N means splitting
// the group that holds bar N at that bar, in every voice, and putting the
// marker in the gap. Bars are numbered as the server's scoreSections.ts does:
// Vocal bars from 1, with `Z4` counting as four.

import { abcSectionMarks, hasLyricTags, mergeLyricBlocks, relabelLyricBlock, sectionPairingError, splitLyricBlock } from './abcSections';
import { byNumber } from '@/utils/sort';

export interface Bar { text: string; start: number; count: number }
// pre: the lines before this part's music since the last part (its V: line,
// if it has one); header: the V: line governing it, own or inherited.
export interface Part { pre: string[]; line: string; header: string | null; voice: string; bars: Bar[] }
export interface Group { parts: Part[]; firstBar: number; bars: number }
export type Item = { kind: 'line'; text: string } | { kind: 'marker'; text: string } | { kind: 'group'; group: Group };

const isMarker = (line: string) => line.trim().startsWith('%') && !line.trim().startsWith('%%') && !!line.trim().slice(1).trim();
const voiceOf = (line: string) => line.trim().slice(2).trim().split(/\s+/)[0] ?? '';
export const isVocal = (line: string) => /vocal/i.test(voiceOf(line)) || /name="Vocal/i.test(line);

/** A music line's bars, splitting on `|` outside chord quotes. */
function barsOf(line: string, offset: number): Bar[] {
  const bars: Bar[] = [];
  let start = 0, quoted = false;
  for (let i = 0; i <= line.length; i++) {
    const c = line[i];
    if (c === '"') quoted = !quoted;
    if (i < line.length && (c !== '|' || quoted)) continue;
    const text = line.slice(start, i);
    if (text.trim()) {
      const multi = text.trim().replace(/"[^"]*"/g, '').match(/^Z(\d*)$/);
      bars.push({ text, start: offset + start, count: multi ? Math.max(1, Number(multi[1] || 1)) : 1 });
    }
    start = i + 1;
  }
  return bars;
}

export function parseScore(abc: string): { items: Item[]; groups: Group[] } | string {
  const lines = abc.replace(/\r\n?/g, '\n').split('\n');
  const keyLine = lines.findIndex(line => /^K:/.test(line.trim()));
  if (keyLine < 0) return 'The score has no K: line.';
  if (lines.some(line => /\|:|:\||\|\||\[\|/.test(line) && !line.trim().startsWith('%')))
    return 'Repeats and double bar lines are not supported here; move the section in the ABC text.';
  const voiced = lines.some((line, i) => i > keyLine && /^V:/.test(line.trim()));
  const items: Item[] = [], groups: Group[] = [];
  const parts: Part[] = [];
  let header: string | null = null, group: Group | null = null, vocalBars = 0, offset = 0;
  let pending: string[] = [];
  const flush = () => { pending.forEach(text => items.push({ kind: 'line', text })); pending = []; };
  lines.forEach((line, i) => {
    const lineStart = offset;
    offset += line.length + 1;
    const text = line.trim();
    if (i <= keyLine) { items.push({ kind: 'line', text: line }); return; }
    if (isMarker(line)) { flush(); group = null; items.push({ kind: 'marker', text: line }); return; }
    if (!text || text.startsWith('%') || /^[A-Za-z]:/.test(text)) {
      if (/^V:/.test(text)) header = line;
      pending.push(line);
      return;
    }
    const voice = header ? voiceOf(header) : '';
    if (!group || group.parts.some(part => part.voice === voice)) {
      group = { parts: [], firstBar: vocalBars + 1, bars: 0 };
      groups.push(group);
      items.push({ kind: 'group', group });
    }
    const part: Part = { pre: pending, line, header, voice, bars: barsOf(line, lineStart) };
    pending = [];
    group.parts.push(part);
    parts.push(part);
    const count = part.bars.reduce((sum, bar) => sum + bar.count, 0);
    if (!voiced || (header && isVocal(header))) { group.bars = count; vocalBars += count; }
  });
  flush();
  if (voiced && !parts.some(part => part.header && isVocal(part.header))) return 'The score has no Vocal voice.';
  return { items, groups };
}

/** The score bar (1-based, Vocal numbering) that the element at `char`
 *  belongs to, or null when it is not in a bar. A multi-bar rest gives its
 *  first bar. */
export function barAtChar(abc: string, char: number): number | null {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return null;
  for (const group of parsed.groups)
    for (const part of group.parts) {
      let bar = group.firstBar;
      for (const b of part.bars) {
        if (char >= b.start && char <= b.start + b.text.length) return bar;
        bar += b.count;
      }
    }
  return null;
}

const rest = (chords: string, count: number) => `${chords}Z${count > 1 ? count : ''}`;

/** Splits a part's bars before its `offset`-th bar, cutting a multi-bar rest. */
function splitBars(bars: Bar[], offset: number): [string[], string[]] | null {
  const before: string[] = [], after: string[] = [];
  let at = 0;
  for (const bar of bars) {
    if (at + bar.count <= offset) before.push(bar.text);
    else if (at >= offset) after.push(bar.text);
    else {
      const chords = (bar.text.trim().match(/^(?:"[^"]*")*/) ?? [''])[0];
      before.push(rest(chords, offset - at));
      after.push(rest('', at + bar.count - offset));
    }
    at += bar.count;
  }
  return at === bars.reduce((sum, bar) => sum + bar.count, 0) ? [before, after] : null;
}

const barLine = (bars: string[]) => bars.map(bar => `${bar}|`).join('');

/** The ABC with section `index` (its `% label` marker, in order) starting at
 *  score bar `bar`, or an error message. The bar must lie after the previous
 *  section's start and before the next one's. */
export function moveSectionStart(abc: string, index: number, bar: number): { abc: string } | { error: string } {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return { error: parsed };
  const { items, groups } = parsed;
  const markers = items.filter(item => item.kind === 'marker');
  const moving = markers[index];
  if (!moving) return { error: 'That section is not in the score.' };
  // Each marker's current start bar: the first group after it.
  const startOf = (marker: Item) => {
    const after = items.slice(items.indexOf(marker) + 1).find(item => item.kind === 'group');
    return after?.kind === 'group' ? after.group.firstBar : Infinity;
  };
  const prev = index > 0 ? startOf(markers[index - 1]) : 0;
  const next = index + 1 < markers.length ? startOf(markers[index + 1]) : Infinity;
  const total = groups.reduce((sum, group) => sum + group.bars, 0);
  if (!Number.isInteger(bar) || bar <= prev || bar >= next || bar < 1 || bar > total)
    return { error: `${moving.text.trim().slice(1).trim()} can only move between bar ${prev + 1} and bar ${Math.min(next, total + 1) - 1}.` };
  if (bar === startOf(moving)) return { abc };

  const target = groups.find(group => bar >= group.firstBar && bar < group.firstBar + group.bars);
  if (!target) return { error: `Bar ${bar} is not in the score.` };
  const out: string[] = [];
  for (const item of items) {
    if (item === moving) continue;
    if (item.kind !== 'group') { out.push(item.text); continue; }
    // Untouched parts keep their text; a split's second half repeats the V: line.
    const whole = (part: Part) => [...part.pre, part.line];
    if (item.group !== target) { item.group.parts.forEach(part => out.push(...whole(part))); continue; }
    if (bar === target.firstBar) {
      out.push(moving.text);
      target.parts.forEach(part => out.push(...whole(part)));
      continue;
    }
    const splits = target.parts.map(part => splitBars(part.bars, bar - target.firstBar));
    const counts = target.parts.map(part => part.bars.reduce((sum, b) => sum + b.count, 0));
    if (splits.some(split => !split) || counts.some(count => count !== target.bars))
      return { error: `The voices do not line up bar for bar around bar ${bar}; move the section in the ABC text.` };
    target.parts.forEach((part, i) => out.push(...part.pre, barLine(splits[i]![0])));
    out.push(moving.text);
    target.parts.forEach((part, i) => out.push(...(part.header ? [part.header] : []), barLine(splits[i]![1])));
  }
  return { abc: out.join('\n') };
}

/** The ABC without section `index`: its `% label` marker and every group of
 *  voice lines up to the next marker, or an error message. A tie from the
 *  bar before the cut into it is dropped, and the first group after the cut
 *  gets the V: and field lines it inherited from the cut section. Lines between the
 *  cut section's last group and the next marker go with it. */
export function cutSection(abc: string, index: number): { abc: string } | { error: string } {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return { error: parsed };
  const { items } = parsed;
  const markers = items.filter(item => item.kind === 'marker');
  if (!markers[index]) return { error: 'That section is not in the score.' };
  if (markers.length < 2) return { error: 'The score has only this section.' };
  const from = items.indexOf(markers[index]);
  const nextMarker = markers[index + 1];
  const to = nextMarker ? items.indexOf(nextMarker) : items.length;
  const kept = [...items.slice(0, from), ...items.slice(to)];
  const before = items.slice(0, from).reverse().find(item => item.kind === 'group');
  const after = items.slice(to).find(item => item.kind === 'group');
  // The last M:, L:, K: or Q: line each voice met in the cut section still
  // governs the bars after it, unless they set their own.
  const isField = (line: string) => /^[A-Za-z]:/.test(line.trim()) && !/^V:/.test(line.trim());
  const fields = new Map<string, Map<string, string>>();
  for (const item of items.slice(from, to))
    for (const [voice, lines] of item.kind === 'group' ? item.group.parts.map(part => [part.voice, part.pre] as const)
      : item.kind === 'line' ? [['', [item.text]] as const] : [])
      for (const line of lines.filter(isField)) {
        if (!fields.has(voice)) fields.set(voice, new Map());
        fields.get(voice)!.set(line.trim()[0], line);
      }
  const out: string[] = [];
  for (const item of kept) {
    if (item.kind !== 'group') { out.push(item.text); continue; }
    for (const part of item.group.parts) {
      let pre = part.pre;
      if (item === after) {
        const own = new Set(pre.filter(isField).map(line => line.trim()[0]));
        const carried = [...new Map([...(fields.get('') ?? []), ...(fields.get(part.voice) ?? [])])]
          .filter(([field]) => !own.has(field)).map(([, line]) => line);
        const v = pre.findIndex(line => /^V:/.test(line.trim()));
        pre = v >= 0 ? [...pre.slice(0, v + 1), ...carried, ...pre.slice(v + 1)]
          : [...(part.header ? [part.header] : []), ...carried, ...pre];
      }
      out.push(...pre, item === before ? part.line.replace(/-(\s*\|?\s*)$/, '$1') : part.line);
    }
  }
  return { abc: out.join('\n') };
}

/** Every bar's character range in every voice, with its score bar number (a
 *  multi-bar rest gives its first bar), in text order. */
export function barRanges(abc: string): Array<{ start: number; end: number; bar: number }> {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return [];
  const out: Array<{ start: number; end: number; bar: number }> = [];
  for (const group of parsed.groups)
    for (const part of group.parts) {
      let bar = group.firstBar;
      for (const b of part.bars) { out.push({ start: b.start, end: b.start + b.text.length, bar }); bar += b.count; }
    }
  return out.sort(byNumber(x => x.start));
}

/** A part's bars cut before each of `offsets` (ascending, from the group's
 *  start): one list of bar texts per piece, the first empty for offset 0. */
function splitMany(bars: Bar[], offsets: number[]): string[][] | null {
  const pieces: string[][] = [];
  let remaining = bars, done = 0;
  for (const offset of offsets) {
    const split = splitBars(remaining, offset - done);
    if (!split) return null;
    pieces.push(split[0]);
    remaining = barsOf(barLine(split[1]), 0);
    done = offset;
  }
  pieces.push(remaining.map(bar => bar.text));
  return pieces;
}

/** The ABC with every `% label` marker replaced by `sections`: one marker at
 *  the start of each given bar (1-based, Vocal numbering), splitting the
 *  groups those bars fall inside in every voice. Groups with no new marker
 *  inside keep their text. */
export function resectionScore(abc: string, sections: Array<{ label: string; bar: number }>): { abc: string } | { error: string } {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return { error: parsed };
  const total = parsed.groups.reduce((sum, group) => sum + group.bars, 0);
  const wanted = [...sections].sort(byNumber(x => x.bar));
  const bad = wanted.find((section, i) => section.bar < 1 || section.bar > total || section.bar === wanted[i - 1]?.bar);
  if (bad) return { error: `Bar ${bad.bar} cannot start a section here (the score has ${total} bars).` };
  const out: string[] = [];
  let next = 0;
  for (const item of parsed.items) {
    if (item.kind === 'marker') continue;
    if (item.kind === 'line') { out.push(item.text); continue; }
    const { group } = item;
    const cuts: typeof wanted = [];
    while (next < wanted.length && wanted[next].bar < group.firstBar + group.bars) cuts.push(wanted[next++]);
    if (!cuts.length) { group.parts.forEach(part => out.push(...part.pre, part.line)); continue; }
    const offsets = cuts.map(cut => cut.bar - group.firstBar);
    const pieces = group.parts.map(part => splitMany(part.bars, offsets));
    const counts = group.parts.map(part => part.bars.reduce((sum, b) => sum + b.count, 0));
    if (pieces.some(piece => !piece) || counts.some(count => count !== group.bars))
      return { error: `The voices do not line up bar for bar around bar ${cuts[0].bar}; set the sections in the ABC text.` };
    // A part's own lines (its V: line, field lines) go before its first
    // piece; later pieces repeat its V: line.
    const started = group.parts.map(() => false);
    pieces[0]!.forEach((_, k) => {
      if (k > 0) out.push(`% ${cuts[k - 1].label}`);
      group.parts.forEach((part, p) => {
        const bars = pieces[p]![k];
        if (!bars.length) return;
        out.push(...(started[p] ? (part.header ? [part.header] : []) : part.pre), barLine(bars));
        started[p] = true;
      });
    });
  }
  return { abc: out.join('\n') };
}

// ── Merge, relabel and split: section edits that keep every bar ──
// The score-only edits (mergeScoreSections and the rest) change the ABC
// alone: Prepare's structure step uses them, and they never read lyrics. The
// paired wrappers (mergeSections and the rest) edit lyric block i with score
// section i, after checking that every section already pairs with its tag;
// otherwise they refuse rather than guess which block to change. Lyrics with
// no tags at all have nothing to pair, and stay as they are. None of these
// edits removes music, so a score's alignment to its audio still holds.

export type SectionEdit = { abc: string; lyrics: string; removedMusic: false } | { error: string };
export type ScoreEdit = { abc: string } | { error: string };

/** The score's lines and the line of each section marker, or why section
 *  `index` (and the `span - 1` after it) cannot be edited. */
function scoreEditable(abc: string, index: number, span: number): { lines: string[]; markers: number[]; labels: string[] } | { error: string } {
  const text = abc.replace(/\r\n?/g, '\n');
  const parsed = parseScore(text);
  if (typeof parsed === 'string') return { error: parsed };
  const labels = abcSectionMarks(text).map(mark => mark.label);
  // The server counts a marker above K: too, and a `%%` directive line as a
  // section; this file only edits markers in the body.
  if (labels.length !== parsed.items.filter(item => item.kind === 'marker').length)
    return { error: 'The score has a section marker above its K: line; edit the sections in the ABC text.' };
  if (text.split('\n').some(line => line.trim().startsWith('%%') && line.trim().slice(1).trim()))
    return { error: 'The score has a %% directive line, which the section check counts as a section; remove it in the ABC text first.' };
  if (!Number.isInteger(index) || index < 0 || index + span > labels.length) return { error: 'That section is not in the score.' };
  const lines = text.split('\n');
  return { lines, markers: lines.flatMap((line, i) => isMarker(line) ? [i] : []), labels };
}

/** scoreEditable, plus whether the lyrics pair with the score (or why not). */
function editable(abc: string, lyrics: string, index: number, span: number)
  : { lines: string[]; markers: number[]; paired: boolean } | { error: string } {
  const at = scoreEditable(abc, index, span);
  if ('error' in at) return at;
  const paired = hasLyricTags(lyrics);
  const mismatch = paired ? sectionPairingError(at.labels, lyrics) : null;
  return mismatch ? { error: mismatch } : { lines: at.lines, markers: at.markers, paired };
}

/** Why `label` cannot name a section, or null. */
const badLabel = (label: string) => !label.trim() || /[[\]\r\n]/.test(label) || label.trim().startsWith('%')
  ? 'A section name needs some text, and no brackets, line breaks or leading %.' : null;

/** A marker line with its label replaced, keeping its `%` and spacing. */
const relabelMarker = (line: string, label: string) => line.match(/^\s*%\s*/)![0] + label.trim();

/** Sections `index` and `index + 1` as one, named after the first or the second. */
export function mergeScoreSections(abc: string, index: number, keep: 'first' | 'second'): ScoreEdit {
  const at = scoreEditable(abc, index, 2);
  if ('error' in at) return at;
  const { lines, markers } = at;
  if (keep === 'second') lines[markers[index]] = lines[markers[index + 1]];
  lines.splice(markers[index + 1], 1);
  return { abc: lines.join('\n') };
}

/** Section `index` renamed to `label`. */
export function relabelScoreSection(abc: string, index: number, label: string): ScoreEdit {
  const bad = badLabel(label);
  if (bad) return { error: bad };
  const at = scoreEditable(abc, index, 1);
  if ('error' in at) return at;
  at.lines[at.markers[index]] = relabelMarker(at.lines[at.markers[index]], label);
  return { abc: at.lines.join('\n') };
}

/** Section `index` split in two: a new section `label` from score bar `bar`. */
export function splitScoreSection(abc: string, index: number, bar: number, label: string): ScoreEdit {
  const bad = badLabel(label);
  if (bad) return { error: bad };
  const at = scoreEditable(abc, index, 1);
  if ('error' in at) return at;
  // The new marker starts beside the old one, then moves to its bar.
  at.lines.splice(at.markers[index] + 1, 0, `% ${label.trim()}`);
  return moveSectionStart(at.lines.join('\n'), index + 1, bar);
}

/** Sections `index` and `index + 1` as one, named after the first or the
 *  second; the lyric blocks join the same way. */
export function mergeSections(abc: string, lyrics: string, index: number, keep: 'first' | 'second'): SectionEdit {
  const at = editable(abc, lyrics, index, 2);
  if ('error' in at) return at;
  const merged = mergeScoreSections(abc, index, keep);
  if ('error' in merged) return merged;
  return { abc: merged.abc, lyrics: at.paired ? mergeLyricBlocks(lyrics, index, keep === 'second') : lyrics, removedMusic: false };
}

/** Section `index` and its lyric tag renamed to `label`. */
export function relabelSection(abc: string, lyrics: string, index: number, label: string): SectionEdit {
  const bad = badLabel(label);
  if (bad) return { error: bad };
  const at = editable(abc, lyrics, index, 1);
  if ('error' in at) return at;
  const renamed = relabelScoreSection(abc, index, label);
  if ('error' in renamed) return renamed;
  return { abc: renamed.abc, lyrics: at.paired ? relabelLyricBlock(lyrics, index, label.trim()) : lyrics, removedMusic: false };
}

/** Section `index` split in two: a new section `label` from score bar `bar`,
 *  holding block i's lyric lines from its `lyricLine`-th non-blank line on
 *  (see lyricBlockLines). */
export function splitSection(abc: string, lyrics: string, index: number, bar: number, lyricLine: number, label: string): SectionEdit {
  const bad = badLabel(label);
  if (bad) return { error: bad };
  const at = editable(abc, lyrics, index, 1);
  if ('error' in at) return at;
  const newLyrics = at.paired ? splitLyricBlock(lyrics, index, lyricLine, label.trim()) : lyrics;
  if (newLyrics === null) return { error: 'That lyric line is not in the section being split.' };
  const split = splitScoreSection(abc, index, bar, label);
  return 'error' in split ? split : { abc: split.abc, lyrics: newLyrics, removedMusic: false };
}
