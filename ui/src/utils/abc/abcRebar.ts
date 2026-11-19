// Regrouping a run of bars into bars of one meter: 3/4 + 4/4 + 1/4 becomes
// two 4/4 bars. Every voice is cut at the same new bar lines. A note that
// crosses one is split and tied, a rest is split, a whole-bar `Z` is written
// out at its real length. Chord symbols keep their positions. A note whose
// accidental came from a bar line that moved, or that its new bar would read
// another way, gets its accidental written in, so abcjs (per letter and
// octave) and YuE2 (per letter) read the same pitch. Everything outside the
// run is left as written: the bar after it gets its meter back.
//
// Nothing is applied here. The result is checked with abcjs (each note's
// pitch, start and length, chord positions, rests, total length) and anything
// this cannot do exactly is refused with a reason, the score untouched.

import abcjs from 'abcjs';
import { isVocal, parseScore, type Bar, type Group, type Item, type Part } from './abcSectionMove';

export type RegroupResult = { abc: string; bars: [number, number] } | { error: string };

interface Row { bar: number; group: Group; text: string; z: string | null; meter: string; sig: number[] }
interface PartMeters { before: string; meter: string }
interface Note { kind: 'note'; acc: string; name: string; len: number; lenText?: string; tie: boolean; index: number; step: number; alt: number; cont: boolean }
interface Rest { kind: 'rest'; char: string; len: number; lenText?: string }
interface Mark { kind: 'mark'; text: string; space: boolean }
type Ev = Note | Rest | Mark;

const STEPS = 'CDEFGAB';
const MARK: Record<string, number> = { '=': 0, '^': 1, '^^': 2, _: -1, __: -2 };
// The lengths the transcriber writes; a longer new piece is tied from these.
const LENGTHS = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1];
const TOKEN = /(\s+)|("[^"]*")|(\^\^|\^|__|_|=)?([A-Ga-g][,']*)([\d/]*)(-?)|([zx])([\d/]*)|(.)/g;
const MODES: Record<string, number> = {
  '': 0, maj: 0, major: 0, ion: 0, ionian: 0, mix: -1, mixolydian: -1, dor: -2, dorian: -2,
  m: -3, min: -3, minor: -3, aeo: -3, aeolian: -3, phr: -4, phrygian: -4, loc: -5, locrian: -5, lyd: 1, lydian: 1,
};
const FIFTHS: Record<string, number> = { C: 0, G: 1, D: 2, A: 3, E: 4, B: 5, F: -1 };

function meterOf(text: string): string | null {
  const t = text.trim();
  if (t === 'C') return '4/4';
  if (t === 'C|') return '2/2';
  const m = /^(\d+)\/(\d+)$/.exec(t);
  return m && Number(m[1]) && Number(m[2]) ? `${Number(m[1])}/${Number(m[2])}` : null;
}
const wholes = (meter: string) => { const [n, d] = meter.split('/').map(Number); return n / d; };

/** Per-letter alterations (C..B) of a K: field, or null for one this cannot read. */
function keySignature(field: string): number[] | null {
  const words = field.trim().split(/\s+/);
  const m = /^([A-G])([#b]?)(.*)$/.exec(words[0] ?? '');
  if (!m) return null;
  let mode = m[3].toLowerCase();
  const rest = words.slice(1);
  if (!mode && rest[0] && Object.hasOwn(MODES, rest[0].toLowerCase())) mode = rest.shift()!.toLowerCase();
  if (!Object.hasOwn(MODES, mode) || rest.some(word => !word.includes('='))) return null;
  const count = FIFTHS[m[1]] + (m[2] === '#' ? 7 : m[2] === 'b' ? -7 : 0) + MODES[mode];
  if (Math.abs(count) > 7) return null;
  const sig = [0, 0, 0, 0, 0, 0, 0], order = count > 0 ? 'FCGDAEB' : 'BEADGCF';
  for (let i = 0; i < Math.abs(count); i++) sig[STEPS.indexOf(order[i])] = Math.sign(count);
  return sig;
}

/** An ABC length (`2`, `/`, `3/2`, `//`) in L: units; null unless its denominator is a power of two. */
function parseLen(text: string): number | null {
  const m = /^(\d*)(\/*)(\d*)$/.exec(text);
  if (!m) return null;
  const slashes = m[2].length;
  const den = !slashes ? (m[3] ? NaN : 1) : m[3] ? (slashes === 1 ? Number(m[3]) : NaN) : 2 ** slashes;
  const num = Number(m[1] || 1);
  return num > 0 && den > 0 && Number.isInteger(Math.log2(den)) ? num / den : null;
}

function lenText(x: number): string {
  let d = 1;
  while (!Number.isInteger(x * d)) d *= 2;
  const n = x * d;
  return d === 1 ? (n === 1 ? '' : `${n}`) : `${n === 1 ? '' : n}/${d}`;
}

/** A new piece's lengths: whole units the transcriber would not write are tied from ones it would. */
function chunks(x: number): number[] {
  if (!Number.isInteger(x) || LENGTHS.includes(x)) return [x];
  const out: number[] = [];
  while (x > 0) { const c = LENGTHS.find(v => v <= x)!; out.push(c); x -= c; }
  return out;
}

const written = (alt: number) => alt > 0 ? '^'.repeat(alt) : alt < 0 ? '_'.repeat(-alt) : '=';
const barLine = (bars: string[]) => bars.map(bar => `${bar}|`).join('');
const unitCount = (part: Part) => part.bars.reduce((sum, bar) => sum + bar.count, 0);
const label = (marker: string) => marker.trim().slice(1).trim();

/** Bars `from` to `to` (unit offsets, a multi-bar rest counting each bar) of a part, cutting a `Zn` where needed. */
function sliceBars(bars: Bar[], from: number, to: number): string[] {
  const out: string[] = [];
  let at = 0;
  for (const bar of bars) {
    const lo = Math.max(from, at), hi = Math.min(to, at + bar.count);
    if (lo < hi) {
      out.push(hi - lo === bar.count ? bar.text
        : `${lo === at ? bar.text.slice(0, bar.text.lastIndexOf('Z')) : ''}Z${hi - lo > 1 ? hi - lo : ''}`);
    }
    at += bar.count;
  }
  return out;
}

/** The score's parse, its L: unit and each voice's bars with their meter and key. */
function scan(abc: string) {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return parsed;
  const lines = abc.replace(/\r\n?/g, '\n').split('\n');
  const keyLine = lines.findIndex(line => /^K:/.test(line.trim()));
  const header = lines.slice(0, keyLine + 1).map(line => line.trim());
  const field = (tag: string) => header.find(line => line.startsWith(`${tag}:`))?.slice(2);
  const headerMeter = meterOf(field('M') ?? '');
  if (!headerMeter) return 'The score needs an M: line with a meter such as 4/4.';
  const length = field('L');
  const unit = length === undefined ? (wholes(headerMeter) < 0.75 ? 1 / 16 : 1 / 8) : parseLen(length.trim().replace(/^1(?=\/)/, ''));
  if (!unit) return `Unsupported note length "L:${length}".`;
  const headerSig = keySignature(field('K') ?? '');
  if (!headerSig) return `Unsupported key "K:${field('K')}".`;
  if (lines.slice(keyLine + 1).some(line => /^L:/.test(line.trim()) || (!line.trim().startsWith('%') && /\[[LM]:/.test(line))))
    return 'Length changes and inline meter changes are not supported here.';

  const meters = new Map<string, string>(), keys = new Map<string, number[]>();
  const rows = new Map<string, Row[]>(), partMeters = new Map<Part, PartMeters>();
  let voice = '';
  const apply = (line: string): string | null => {
    const text = line.trim();
    if (/^V:/.test(text)) voice = text.slice(2).trim().split(/\s+/)[0] ?? '';
    else if (/^M:/.test(text)) { const m = meterOf(text.slice(2)); if (!m) return `Unsupported meter "${text}".`; meters.set(voice, m); }
    else if (/^K:/.test(text)) { const k = keySignature(text.slice(2)); if (!k) return `Unsupported key "${text}".`; keys.set(voice, k); }
    return null;
  };
  for (const item of parsed.items.slice(keyLine + 1)) {
    const problem = item.kind === 'line' ? apply(item.text) : null;
    if (problem) return problem;
    if (item.kind !== 'group') continue;
    for (const part of item.group.parts) {
      const before = meters.get(part.voice) ?? headerMeter;
      for (const line of part.pre) { const p = apply(line); if (p) return p; }
      const meter = meters.get(part.voice) ?? headerMeter;
      partMeters.set(part, { before, meter });
      const list = rows.get(part.voice) ?? [];
      rows.set(part.voice, list);
      let bar = item.group.firstBar;
      for (const b of part.bars) {
        const sig = keys.get(part.voice) ?? headerSig;
        const multi = b.text.replace(/"[^"]*"/g, '').trim().match(/^Z(\d*)$/);
        const prefix = b.text.slice(0, b.text.lastIndexOf('Z'));
        for (let i = 0; i < b.count; i++)
          list.push({ bar: bar++, group: item.group, text: multi ? '' : b.text, z: multi ? (i ? '' : prefix) : null, meter, sig });
        // A key change inside a bar governs the bars after it.
        for (const [, k] of b.text.matchAll(/\[K:([^\]]*)\]/g)) { const s = keySignature(k); if (s) keys.set(part.voice, s); }
      }
    }
  }
  const parts = parsed.groups.flatMap(g => g.parts);
  const lead = parts.find(p => !p.header || isVocal(p.header))?.voice ?? '';
  return { ...parsed, unit, rows, partMeters, lead };
}

/** A bar's events, or the bit of notation this cannot read. */
function readBar(text: string): Ev[] | string {
  const evs: Ev[] = [];
  for (const m of text.matchAll(TOKEN)) {
    const [whole, space, chord, acc, name, len, tie, rest, restLen, other] = m;
    if (space || chord) { evs.push({ kind: 'mark', text: whole, space: !!space }); continue; }
    if (other) return other;
    const l = parseLen(name ? len : restLen);
    if (l === null) return whole;
    if (!name) { evs.push({ kind: 'rest', char: rest, len: l, lenText: restLen }); continue; }
    const step = STEPS.indexOf(name[0].toUpperCase());
    const octave = (name[0] === name[0].toUpperCase() ? 4 : 5) + (name.match(/'/g)?.length ?? 0) - (name.match(/,/g)?.length ?? 0);
    evs.push({ kind: 'note', acc: acc ?? '', name, len: l, lenText: len, tie: !!tie, index: 7 * octave + step, step, alt: 0, cont: false });
  }
  return evs;
}

/** The voice's events for `rows`, each note with the alteration it sounds at
 *  and whether a tie carries it on from the note before. */
function sourceEvents(rows: Row[], unit: number, voice: string, tieIn: { index: number; alt: number } | null, full = true): Ev[] | string {
  const evs: Ev[] = [];
  let tie = tieIn;
  for (const row of rows) {
    const size = wholes(row.meter) / unit;
    const read = readBar(row.z ?? row.text);
    if (typeof read === 'string') return `Bar ${row.bar} has notation this cannot regroup ("${read}").`;
    if (row.z !== null) read.push({ kind: 'rest', char: 'z', len: size });
    const units = read.reduce((sum, ev) => sum + (ev.kind === 'mark' ? 0 : ev.len), 0);
    if (full && Math.abs(units - size) > 1e-9) {
      return `Bar ${row.bar}${voice ? ` in ${voice}` : ''} does not fill its ${row.meter} meter, so the voices cannot be regrouped together.`;
    }
    const exact = new Map<number, number>();
    for (const ev of read) {
      if (ev.kind === 'rest') tie = null;
      if (ev.kind !== 'note') continue;
      ev.cont = tie?.index === ev.index;
      ev.alt = ev.cont ? tie!.alt : ev.acc ? MARK[ev.acc] : exact.get(ev.index) ?? row.sig[ev.step];
      if (!ev.cont && ev.acc) exact.set(ev.index, ev.alt);
      tie = ev.tie ? { index: ev.index, alt: ev.alt } : null;
    }
    evs.push(...read);
  }
  return evs;
}

/** The events cut into bars of `size` units, with ties and accidentals for the new bar lines. */
function cut(evs: Ev[], size: number, sig: number[]): string[] {
  const bars: string[] = [];
  let cur = '', t = 0, exact = new Map<number, number>(), letter = new Map<number, number>();
  const close = () => { bars.push(cur); cur = ''; t = 0; exact = new Map(); letter = new Map(); };
  // abcjs reads a tied-on note at the tied pitch and ignores its accidental,
  // so only the letter view (YuE2's) follows one; any other note sets both.
  const prefix = (ev: Note, cont: boolean, acc: string) => {
    if (cont) {
      if (!acc && (letter.get(ev.step) ?? sig[ev.step]) === ev.alt) return '';
      letter.set(ev.step, ev.alt);
      return acc || written(ev.alt);
    }
    if (!acc && (exact.get(ev.index) ?? sig[ev.step]) === ev.alt && (letter.get(ev.step) ?? sig[ev.step]) === ev.alt) return '';
    exact.set(ev.index, ev.alt);
    letter.set(ev.step, ev.alt);
    return acc || written(ev.alt);
  };
  // Rests next to each other are one rest, cut where the new bar lines fall.
  const joined: Ev[] = [];
  for (const ev of evs) {
    const prev = joined.at(-1);
    if (ev.kind === 'rest' && prev?.kind === 'rest' && prev.char === ev.char) joined[joined.length - 1] = { ...prev, len: prev.len + ev.len, lenText: undefined };
    else joined.push(ev);
  }
  for (const ev of joined) {
    if (t === size) close();
    if (ev.kind === 'mark') { if (!ev.space || cur) cur += ev.text; continue; }
    let left = ev.len, first = true;
    while (left > 1e-9) {
      if (t === size) close();
      const piece = Math.min(left, size - t);
      const lens = piece === ev.len && ev.lenText !== undefined ? [ev.len] : chunks(piece);
      for (const l of lens) {
        const text = piece === ev.len && ev.lenText !== undefined ? ev.lenText : lenText(l);
        left -= l;
        t += l;
        if (ev.kind === 'rest') cur += ev.char + text;
        else cur += prefix(ev, ev.cont || !first, first ? ev.acc : '') + ev.name + text + (left > 1e-9 || ev.tie ? '-' : '');
        first = false;
      }
    }
  }
  if (cur) close();
  return bars;
}

/** The score with every `Z` bar written out as plain rests, timed by its own
 *  meter: abcjs times a `Z` by the header meter. */
function explicitRests(abc: string): string {
  let meter = '', length = '', voice = '', body = false;
  const meters = new Map<string, string>();
  return abc.split(/(\r?\n)/).map(line => {
    if (/^M:/.test(line)) { if (body) meters.set(voice, meterOf(line.slice(2)) ?? ''); else meter = meterOf(line.slice(2)) ?? ''; return line; }
    if (/^L:/.test(line)) { length = line.slice(2).trim(); return line; }
    if (/^K:/.test(line)) { body = true; return line; }
    if (/^V:/.test(line)) { voice = line.slice(2).trim().split(/\s+/)[0]; return line; }
    if (!body || !meter || !line.includes('Z') || /^([A-Za-z]:|%)/.test(line)) return line;
    const unit = length ? parseLen(length.replace(/^1(?=\/)/, '')) ?? 1 / 8 : wholes(meter) < 0.75 ? 1 / 16 : 1 / 8;
    const units = lenText(wholes(meters.get(voice) || meter) / unit);
    return line.replace(/(^|\|)((?:"[^"]*")*)Z(\d*)(?=\|)/g, (_, bar: string, chords: string, count: string) =>
      bar + Array.from({ length: Number(count || 1) }, (_, i) => `${i ? '' : chords}z${units}`).join('|'));
  }).join('');
}

interface AbcElement { el_type: string; duration?: number; rest?: unknown; chord?: Array<{ name: string }> }
interface AbcTune {
  lines: Array<{ staff?: Array<{ voices: AbcElement[][] }> }>;
  setUpAudio(): { totalDuration: number; tracks: Array<Array<{ cmd: string; pitch: number; start: number; duration: number; startChar?: number }>> };
}

/** What the score sounds like, as abcjs reads it: each track's notes
 *  [pitch, start, length], each staff's chord symbols [name, start] and
 *  rests [start, end] (adjacent rests joined), and the total length. */
export function scoreTimeline(abc: string) {
  const tune = abcjs.parseOnly(explicitRests(abc))[0] as unknown as AbcTune;
  const audio = tune.setUpAudio();
  const staves = new Map<string, { at: number; chords: Array<[string, number]>; rests: Array<[number, number]> }>();
  for (const line of tune.lines) line.staff?.forEach((staff, s) => staff.voices.forEach((elements, v) => {
    const key = `${s}.${v}`;
    const staffLine = staves.get(key) ?? { at: 0, chords: [], rests: [] };
    staves.set(key, staffLine);
    for (const el of elements) {
      if (el.el_type !== 'note') continue;
      const d = el.duration ?? 0;
      for (const chord of el.chord ?? []) staffLine.chords.push([chord.name, staffLine.at]);
      const last = staffLine.rests.at(-1);
      if (el.rest && last && Math.abs(last[1] - staffLine.at) < 1e-9) last[1] += d;
      else if (el.rest) staffLine.rests.push([staffLine.at, staffLine.at + d]);
      staffLine.at += d;
    }
  }));
  return {
    total: audio.totalDuration,
    notes: audio.tracks.map(track => track.filter(e => e.cmd === 'note' && e.startChar !== undefined).map(e => [e.pitch, e.start, e.duration])),
    staves: [...staves.values()].map(({ chords, rests }) => ({ chords, rests })),
  };
}

const rounded = (value: unknown) => JSON.stringify(value, (_, x) => typeof x === 'number' ? Math.round(x * 1e9) / 1e9 : x);

/** Each score bar's meter, Vocal numbering (a `Z4` is four bars), or why the score cannot be read. */
export function barMeters(abc: string): string[] | { error: string } {
  const s = scan(abc);
  return typeof s === 'string' ? { error: s } : (s.rows.get(s.lead) ?? []).map(row => row.meter);
}

/** Bars `range` (first and last, Vocal numbering, inclusive) rewritten as
 *  bars of `targetMeter` in every voice, with the new bar numbers of the run,
 *  or why not. Section boundaries inside the run must fall on a new bar line;
 *  a key or tempo change inside it, voices that do not line up, or notation
 *  this cannot read are refused. */
export function previewRegroup(abc: string, range: [number, number], targetMeter: string): RegroupResult {
  const s = scan(abc);
  if (typeof s === 'string') return { error: s };
  const target = meterOf(targetMeter);
  if (!target) return { error: `"${targetMeter}" is not a meter such as 4/4.` };
  const [first, last] = range;
  const total = s.groups.reduce((sum, g) => sum + g.bars, 0);
  if (!Number.isInteger(first) || !Number.isInteger(last) || first < 1 || last < first || last > total)
    return { error: `Choose bars between 1 and ${total}.` };
  const span = s.groups.filter(g => g.bars && g.firstBar <= last && g.firstBar + g.bars > first);
  const gFirst = span[0], gLast = span.at(-1)!;
  const voices = gFirst.parts.map(p => p.voice);
  const where = first === last ? `bar ${first}` : `bars ${first}–${last}`;
  if (span.some(g => g.parts.length !== voices.length || g.parts.some((p, i) => p.voice !== voices[i] || unitCount(p) !== g.bars)))
    return { error: `The voices do not line up in ${where}, so they cannot be regrouped together.` };

  // What sits between the run's groups: section markers, V: and M: lines only.
  const index = (g: Group) => s.items.findIndex(item => item.kind === 'group' && item.group === g);
  const iFirst = index(gFirst), iLast = index(gLast);
  const markers: Array<{ text: string; bar: number }> = [];
  for (let i = iFirst + 1; i <= iLast; i++) {
    const item = s.items[i];
    if (item.kind === 'marker') {
      const next = s.items.slice(i + 1).find(it => it.kind === 'group') as Extract<Item, { kind: 'group' }>;
      markers.push({ text: item.text, bar: next.group.firstBar });
    }
    for (const line of item.kind === 'line' ? [item.text] : item.kind === 'group' ? item.group.parts.flatMap(p => p.pre) : []) {
      const text = line.trim();
      if (/^K:/.test(text)) return { error: `The key changes inside ${where}; regroup the bars on each side of the change instead.` };
      if (/^Q:/.test(text)) return { error: `The tempo changes inside ${where}; regroup the bars on each side of the change instead.` };
      if (text && !/^[VM]:/.test(text)) return { error: `"${text}" inside ${where} is not supported here.` };
    }
  }

  const runs = voices.map(v => {
    const all = s.rows.get(v)!;
    const rows = all.filter(r => span.includes(r.group) && r.bar >= first && r.bar <= last);
    return { voice: v, all, rows, at: all.indexOf(rows[0]) };
  });
  for (const run of runs) {
    for (const [k, row] of run.rows.entries()) {
      if (row.meter !== runs[0].rows[k].meter)
        return { error: `The voices disagree on the meter of bar ${row.bar}, so they cannot be regrouped together.` };
      if (/\[K:/.test(row.text)) return { error: `The key changes inside ${where}; regroup the bars on each side of the change instead.` };
      if (/\[Q:/.test(row.text)) return { error: `The tempo changes inside ${where}; regroup the bars on each side of the change instead.` };
    }
  }
  const lead = runs[0].rows;
  if (lead.every(r => r.meter === target)) return { error: `${where[0].toUpperCase()}${where.slice(1)} ${first === last ? 'is' : 'are'} already ${target}.` };

  const size = wholes(target) / s.unit;
  const units = (rows: Row[]) => rows.reduce((sum, r) => sum + wholes(r.meter) / s.unit, 0);
  const count = units(lead) / size;
  if (Math.abs(count - Math.round(count)) > 1e-9) {
    const beats = Math.round(units(lead) * s.unit * Number(target.split('/')[1]) * 1e6) / 1e6;
    return { error: `${where[0].toUpperCase()}${where.slice(1)} hold ${beats} beats, which is not a whole number of ${target} bars.` };
  }
  const cuts = [0];
  for (const marker of markers) {
    const at = units(lead.filter(r => r.bar < marker.bar)) / size;
    if (Math.abs(at - Math.round(at)) > 1e-9)
      return { error: `"${label(marker.text)}" would start inside a new ${target} bar (it starts at bar ${marker.bar}). Move or merge that section boundary first.` };
    cuts.push(Math.round(at));
  }
  cuts.push(Math.round(count));

  const newBars: string[][] = [];
  for (const run of runs) {
    // A note tied into the run: walk back to where its tie chain starts.
    let from = run.at;
    while (from > 0 && /-\s*$/.test(run.all[from - 1].text)) from--;
    const chain = sourceEvents(run.all.slice(from, run.at), s.unit, run.voice, null, false);
    if (typeof chain === 'string') return { error: `The note tied into bar ${first} uses notation this cannot read.` };
    const lastNote = chain.filter(ev => ev.kind !== 'mark').at(-1);
    const tieIn = lastNote?.kind === 'note' && lastNote.tie ? { index: lastNote.index, alt: lastNote.alt } : null;
    const evs = sourceEvents(run.rows, s.unit, run.voice, tieIn);
    if (typeof evs === 'string') return { error: evs };
    newBars.push(cut(evs, size, run.rows[0].sig));
    if (newBars.at(-1)!.length !== Math.round(count)) return { error: `${where[0].toUpperCase()}${where.slice(1)} could not be regrouped exactly.` };
  }

  // The run's groups are rewritten; the bars around it stay as written.
  const meterOfPart = (p: Part) => s.partMeters.get(p)!;
  const lead0 = first - gFirst.firstBar, tail = gLast.firstBar + gLast.bars - 1 - last;
  const span0: string[] = [];
  if (lead0) for (const p of gFirst.parts) span0.push(...p.pre, barLine(sliceBars(p.bars, 0, lead0)));
  for (let k = 0; k + 1 < cuts.length; k++) {
    if (k) span0.push(markers[k - 1].text);
    gFirst.parts.forEach((p, v) => {
      if (!k && !lead0) span0.push(...p.pre.filter(line => !/^M:/.test(line.trim())));
      else if (p.header) span0.push(p.header);
      const before = k ? target : lead0 ? meterOfPart(p).meter : meterOfPart(p).before;
      if (before !== target) span0.push(`M:${target}`);
      span0.push(barLine(newBars[v].slice(cuts[k], cuts[k + 1])));
    });
  }
  const pre = new Map<Part, string[]>();
  if (tail) {
    for (const p of gLast.parts) {
      if (p.header) span0.push(p.header);
      if (meterOfPart(p).meter !== target) span0.push(`M:${meterOfPart(p).meter}`);
      span0.push(barLine(sliceBars(p.bars, unitCount(p) - tail, unitCount(p))));
    }
  } else {
    // Each voice's next part gets its meter back.
    const after = s.items.slice(iLast + 1).flatMap(it => it.kind === 'group' ? it.group.parts : []);
    for (const p of voices.map(v => after.find(part => part.voice === v)).filter(p => p !== undefined)) {
      if (p.pre.some(line => /^M:/.test(line.trim())) || meterOfPart(p).meter === target) continue;
      const v = p.pre.map(line => /^V:/.test(line.trim())).lastIndexOf(true);
      pre.set(p, [...p.pre.slice(0, v + 1), `M:${meterOfPart(p).meter}`, ...p.pre.slice(v + 1)]);
    }
  }
  const emit = (item: Item) => item.kind === 'group' ? item.group.parts.flatMap(p => [...(pre.get(p) ?? p.pre), p.line]) : [item.text];
  const out = [...s.items.slice(0, iFirst).flatMap(emit), ...span0, ...s.items.slice(iLast + 1).flatMap(emit)].join('\n');
  const result = abc.includes('\r\n') ? out.replace(/\n/g, '\r\n') : out;

  let same: boolean;
  try { same = rounded(scoreTimeline(abc)) === rounded(scoreTimeline(result)); } catch { same = false; }
  if (!same) return { error: 'Regrouping these bars would change the music, so the score was left as it was.' };
  return { abc: result, bars: [first, first + Math.round(count) - 1] };
}
