import { unzipSync, strFromU8 } from 'fflate';
import { byNumber } from '@/utils/sort';
import { isPresent } from '@/utils/guards';

export interface MusicXmlPart { id: string; name: string; staves: number; voices: number; hasLyrics: boolean; verses: string[] }
/** `chords`: where the score's chord symbols came from: its own <harmony>, inferred
 *  from the accompaniment, or none. */
export interface MusicXmlResult { abc: string; title: string; lyrics: string; chords: 'score' | 'inferred' | null; parts: MusicXmlPart[]; partId: string | null; warnings: string[] }
export class MusicXmlImportError extends Error {}

const MAX_FILE = 8 * 1024 * 1024;
const MAX_XML = 16 * 1024 * 1024;
const MAX_ENTRIES = 256;
const child = (node: Element, name: string) => Array.from(node.children).find(e => e.localName === name);
const children = (node: Element, name: string) => Array.from(node.children).filter(e => e.localName === name);
const value = (node: Element, name: string) => child(node, name)?.textContent?.trim() ?? '';
const descendants = (node: Element, name: string) => Array.from(node.getElementsByTagName(name));
const clean = (s: string) => s.replace(/[\r\n]+/g, ' ').replace(/[:\\]/g, ' ').trim();

function xml(text: string): Document {
  if (text.length > MAX_XML) throw new MusicXmlImportError('MusicXML is too large');
  if (/<!ENTITY/i.test(text)) throw new MusicXmlImportError('DTD entities are not supported');
  const doctype = /<!DOCTYPE\s+score-(?:partwise|timewise)\s+PUBLIC\s+"-\/\/Recordare\/\/DTD MusicXML [\d.]+ (?:Partwise|Timewise)\/\/EN"\s+"https?:\/\/www\.musicxml\.org\/dtds\/(?:partwise|timewise)\.dtd"\s*(?:\[\s*\])?\s*>/i;
  const withoutKnownDtd = text.replace(doctype, '');
  if (/<!DOCTYPE/i.test(withoutKnownDtd)) throw new MusicXmlImportError('Unsupported DTD');
  const doc = new DOMParser().parseFromString(withoutKnownDtd, 'application/xml');
  if (doc.querySelector('parsererror')) throw new MusicXmlImportError('Invalid MusicXML');
  return doc;
}

// YuE2's native score (upstream skills/yue2-music/instrumental/references/abc-repair.md):
// L:1/32, so a quarter note is 8 ticks; only these lengths, longer ones split with ties.
const TICKS_PER_QUARTER = 8;
const LENGTHS = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1];
function split(ticks: number): number[] {
  const out: number[] = [];
  while (ticks > 0) { const n = LENGTHS.find(l => l <= ticks)!; out.push(n); ticks -= n; }
  return out;
}
const len = (n: number) => n === 1 ? '' : String(n);

const MAJOR_SHARP = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#'];
const MAJOR_FLAT = ['C', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb'];
const MINOR_SHARP = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#'];
const MINOR_FLAT = ['A', 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab'];
function keyAlters(fifths: number): Record<string, number> {
  const order = fifths >= 0 ? 'FCGDAEB' : 'BEADGCF';
  return Object.fromEntries(Array.from(order.slice(0, Math.abs(fifths)), l => [l, Math.sign(fifths)]));
}
function soundingKey(fifths: number, mode: string, chromatic: number): { name: string; fifths: number } {
  const names = mode === 'minor' ? [MINOR_FLAT, MINOR_SHARP] : [MAJOR_FLAT, MAJOR_SHARP];
  const root = (name: string) => (NATURAL[name[0]] + (name.includes('b') ? -1 : name.includes('#') ? 1 : 0) + 12) % 12;
  const written = fifths < 0 ? names[0][-fifths] : names[1][fifths];
  const target = (root(written) + chromatic + 120) % 12;
  const candidates = Array.from({ length: 15 }, (_, i) => i - 7).filter(f => root(f < 0 ? names[0][-f] : names[1][f]) === target);
  const chosen = candidates.sort((a, b) => Math.abs(a - fifths) - Math.abs(b - fifths))[0] ?? fifths;
  return { name: (chosen < 0 ? names[0][-chosen] : names[1][chosen]) + (mode === 'minor' ? 'm' : ''), fifths: chosen };
}

interface Pitch { step: string; octave: number; alter: number }
interface Transpose { diatonic: number; chromatic: number; octave: number }
const STEPS = 'CDEFGAB';
function transposePitch(p: Pitch, t: Transpose): Pitch {
  const old = p.octave * 7 + STEPS.indexOf(p.step);
  const next = old + t.diatonic + 7 * t.octave;
  const step = STEPS[((next % 7) + 7) % 7];
  const octave = Math.floor(next / 7);
  const midi = 12 * (p.octave + 1) + NATURAL[p.step] + p.alter + t.chromatic + 12 * t.octave;
  const alter = midi - (12 * (octave + 1) + NATURAL[step]);
  if (Math.abs(alter) > 2) throw new MusicXmlImportError('Unsupported transposed pitch');
  return { step, octave, alter };
}
function pitch(note: Element): Pitch {
  const p = child(note, 'pitch');
  if (!p) throw new MusicXmlImportError('Unpitched notes are unsupported');
  const step = value(p, 'step');
  const octave = Number(value(p, 'octave'));
  const alter = Number(value(p, 'alter') || 0);
  if (!/^[A-G]$/.test(step) || !Number.isInteger(octave) || ![-2, -1, 0, 1, 2].includes(alter)) throw new MusicXmlImportError('Unsupported pitch');
  return { step, octave, alter };
}

/** Writes accidentals so the pitch reads the same whether an accidental carries
 *  to every octave of its letter for the rest of the bar (YuE2's training
 *  convention) or only to its own octave (standard ABC, abcjs). */
function spell(p: Pitch, key: Record<string, number>, letter: Map<string, number>, exact: Map<string, number>): string {
  const inKey = key[p.step] ?? 0;
  const id = p.step + p.octave;
  const mark = (letter.get(p.step) ?? inKey) === p.alter && (exact.get(id) ?? inKey) === p.alter ? ''
    : p.alter < 0 ? '_'.repeat(-p.alter) : p.alter > 0 ? '^'.repeat(p.alter) : '=';
  if (mark) { letter.set(p.step, p.alter); exact.set(id, p.alter); }
  return mark + (p.octave >= 5 ? p.step.toLowerCase() + "'".repeat(p.octave - 5) : p.step + ','.repeat(Math.max(0, 4 - p.octave)));
}

interface Bar { body: string; rest: boolean }

/** Convert the timewise container into the partwise shape used by the parser. */
function partwise(doc: Document): Element[] {
  const root = doc.documentElement;
  if (root.localName === 'score-partwise') return children(root, 'part');
  if (root.localName !== 'score-timewise') throw new MusicXmlImportError('Only score-partwise or score-timewise MusicXML is supported');
  const ids = children(child(root, 'part-list') ?? root, 'score-part').map(p => p.getAttribute('id') ?? '');
  const parts = ids.map(id => {
    const part = doc.createElement('part');
    part.setAttribute('id', id);
    for (const measure of children(root, 'measure')) {
      const source = children(measure, 'part').find(p => p.getAttribute('id') === id);
      const copy = doc.createElement('measure');
      for (const attr of Array.from(measure.attributes)) copy.setAttribute(attr.name, attr.value);
      for (const item of Array.from(source?.children ?? [])) copy.appendChild(item.cloneNode(true));
      part.appendChild(copy);
    }
    return part;
  });
  return parts;
}

/** Resolve written navigation before assigning score time. A finite play count and
 * cycle guard keep malformed repeat instructions from looping indefinitely. */
function playOrder(measures: Element[], warnings: Set<string>): number[] {
  const left = (m: Element) => children(m, 'barline').filter(b => b.getAttribute('location') !== 'right');
  const right = (m: Element) => children(m, 'barline').filter(b => b.getAttribute('location') !== 'left');
  const hasRepeat = (bars: Element[], direction: string) => bars.some(b => children(b, 'repeat').some(r => r.getAttribute('direction') === direction));
  const ending = (bars: Element[]) => bars.flatMap(b => children(b, 'ending'));
  const start = measures.map(m => hasRepeat(left(m), 'forward'));
  const back = measures.map(m => right(m).flatMap(b => children(b, 'repeat')).find(r => r.getAttribute('direction') === 'backward'));
  const allowed: (Set<number> | null)[] = [];
  let current: Set<number> | null = null;
  for (let i = 0; i < measures.length; i++) {
    for (const e of ending(left(measures[i]))) if (e.getAttribute('type') === 'start') current = new Set((e.getAttribute('number') ?? '').split(/[,\s-]+/).map(Number));
    allowed.push(current);
    for (const e of ending(right(measures[i]))) if (e.getAttribute('type') === 'stop' || e.getAttribute('type') === 'discontinue') current = null;
  }
  const directions = measures.map(m => descendants(m, 'direction'));
  const sound = (i: number, attr: string) => directions[i].some(d => descendants(d, 'sound').some(s => s.hasAttribute(attr)));
  const words = (i: number) => directions[i].flatMap(d => descendants(d, 'words')).map(w => w.textContent ?? '').join(' ').toLowerCase();
  const segno = measures.findIndex((_, i) => sound(i, 'segno') || directions[i].some(d => descendants(d, 'segno').length));
  const coda = measures.findIndex((_, i) => sound(i, 'coda') || directions[i].some(d => descendants(d, 'coda').length));
  const order: number[] = [];
  const visits = new Map<number, number>();
  let repeatStart = 0, pass = 1, jumped = false, codaJumped = false, i = 0;
  while (i < measures.length && order.length < 2048) {
    if (start[i] && (!visits.has(i) || pass === 1)) repeatStart = i;
    if (!allowed[i] || allowed[i]!.has(pass)) order.push(i);
    visits.set(i, (visits.get(i) ?? 0) + 1);
    const backward = back[i];
    if (backward && pass < Math.max(2, Number(backward.getAttribute('times')) || 2)) { pass++; i = repeatStart; continue; }
    if (backward && !allowed[i + 1]?.has(pass)) pass = 1;
    const cue = words(i);
    const ds = sound(i, 'dalsegno') || /d\.?\s?s\.?\s?(?:al|to|$)/i.test(cue);
    const dc = sound(i, 'dacapo') || /d\.?\s?c\.?\s?(?:al|to|$)/i.test(cue);
    if (!jumped && (ds || dc)) { jumped = true; i = ds && segno >= 0 ? segno : 0; continue; }
    if (jumped && !codaJumped && coda >= 0 && (sound(i, 'tocoda') || /to\s*coda/i.test(cue))) { codaJumped = true; i = coda; continue; }
    if (jumped && (sound(i, 'fine') || /\bfine\b/i.test(cue))) break;
    if (ending(right(measures[i])).some(e => e.getAttribute('number')?.split(/[,\s-]+/).includes(String(pass)))) pass = 1;
    i++;
  }
  if (order.length >= 2048) warnings.add('Navigation exceeds 2,048 played measures; the score is cut there');
  return order;
}

// Chord symbols use YuE2's chord set (upstream abc_tools.QUALITIES); MusicXML
// kinds outside it map to the nearest member and are flagged as simplified.
const KINDS: Record<string, [string, boolean]> = {
  major: ['', true], minor: ['m', true], augmented: ['aug', true], diminished: ['dim', true],
  dominant: ['7', true], 'major-seventh': ['maj7', true], 'minor-seventh': ['m7', true],
  'diminished-seventh': ['dim7', true], 'half-diminished': ['m7b5', true], 'major-minor': ['m(maj7)', true],
  'major-sixth': ['6', true], 'minor-sixth': ['m6', true], 'suspended-fourth': ['sus4', true], 'suspended-second': ['sus2', true],
  'augmented-seventh': ['aug', false], 'dominant-ninth': ['7', false], 'dominant-11th': ['7', false], 'dominant-13th': ['7', false],
  'major-ninth': ['maj7', false], 'major-11th': ['maj7', false], 'major-13th': ['maj7', false],
  'minor-ninth': ['m7', false], 'minor-11th': ['m7', false], 'minor-13th': ['m7', false], power: ['', false], pedal: ['', false],
};
const NATURAL: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const accidental = (alter: number) => alter < 0 ? 'b'.repeat(-alter) : '#'.repeat(alter);

interface Chord { root: number; quality: string; bass: string }
// Each quality's tones as steps on the line of fifths from the root (dim7's 6th is a bb7).
const FIFTHS_FROM_ROOT: Record<string, number[]> = {
  '': [0, 4, 1], m: [0, -3, 1], aug: [0, 4, 8], dim: [0, -3, -6], sus4: [0, -1, 1], '7sus4': [0, -1, 1, -2], sus2: [0, 2, 1],
  '7': [0, 4, 1, -2], maj7: [0, 4, 1, 5], m7: [0, -3, 1, -2], 'm(maj7)': [0, -3, 1, 5], '6': [0, 4, 1, 3], m6: [0, -3, 1, 3],
  dim7: [0, -3, -6, -9], m7b5: [0, -3, -6, -2],
};
const LINE_OF_FIFTHS: Record<string, number> = { F: 0, C: 1, G: 2, D: 3, A: 4, E: 5, B: 6 };

/** The root's name as SheetSage2 spells chords (sheetsage-events.h correct_chord,
 *  after upstream correct_chord_spelling): the spelling whose tones sit closest to
 *  the key signature on the line of fifths, the root counting twice; ties go to C-B order. */
function rootName(chord: Chord, fifths: number): string {
  const window = fifths + 1 + 2;
  let best = { score: Infinity, name: '' };
  for (const step of 'CDEFGAB') {
    const alter = ((chord.root - NATURAL[step] + 6) % 12 + 12) % 12 - 6;
    const pos = LINE_OF_FIFTHS[step] + 7 * alter;
    const far = (off: number) => Math.max(Math.abs(pos + off - window) - 3, 0);
    const score = (FIFTHS_FROM_ROOT[chord.quality] ?? [0]).reduce((s, off) => s + far(off), 0) + far(0);
    if (score < best.score) best = { score, name: step + accidental(alter) };
  }
  return best.name;
}
const label = (chord: Chord, fifths: number) => rootName(chord, fifths) + chord.quality + chord.bass;

/** A <harmony> as a chord; null for N.C., undefined when unsupported. */
function harmony(item: Element, warnings: Set<string>): Chord | null | undefined {
  const kind = value(item, 'kind');
  if (kind === 'none') return null;
  const root = child(item, 'root'), bass = child(item, 'bass');
  const mapped = KINDS[kind];
  if (!root || !mapped || !(value(root, 'root-step') in NATURAL)) { warnings.add('Chord symbols of an unsupported type are omitted'); return undefined; }
  let [quality, exact] = mapped;
  const degrees = children(item, 'degree').map(d => value(d, 'degree-value'));
  if (quality === 'sus4' && degrees.length === 1 && degrees[0] === '7') quality = '7sus4';
  else if (degrees.length) exact = false;
  if (!exact) warnings.add("Chord symbols outside YuE2's chord set are simplified");
  const name = (node: Element, prefix: string) => value(node, `${prefix}-step`) + accidental(Math.round(Number(value(node, `${prefix}-alter`) || 0)));
  const alter = Math.round(Number(value(root, 'root-alter') || 0));
  return { root: ((NATURAL[value(root, 'root-step')] ?? NaN) + alter + 12) % 12, quality, bass: bass ? '/' + name(bass, 'bass') : '' };
}

interface Sounding { start: number; end: number; pc: number; midi: number; weight: number }

/** Every sounding note and chord symbol of a part per measure, in 32nd ticks, for
 *  chords from other parts and chord inference. With `melodyVoice`, notes of the
 *  imported melody line weigh half. */
function partHarmony(part: Element, melodyVoice: string | null, melodyStaff: number, warnings: Set<string>): { notes: Sounding[]; chords: { at: number; chord: Chord | null }[] }[] {
  let divisions = 1, transposition: Transpose = { diatonic: 0, chromatic: 0, octave: 0 };
  return children(part, 'measure').map(measure => {
    let cursor = 0, last = 0;
    const notes: Sounding[] = [];
    const chords: { at: number; chord: Chord | null }[] = [];
    for (const item of Array.from(measure.children)) {
      if (item.localName === 'attributes') {
        divisions = Number(value(item, 'divisions')) || divisions;
        const trans = child(item, 'transpose');
        if (trans) transposition = { diatonic: Number(value(trans, 'diatonic')) || 0,
          chromatic: Number(value(trans, 'chromatic')) || 0, octave: Number(value(trans, 'octave-change')) || 0 };
      }
      if (item.localName === 'harmony') {
        const chord = harmony(item, warnings);
        if (chord !== undefined) chords.push({ at: Math.max(0, Math.round((cursor + (Number(value(item, 'offset')) || 0)) * TICKS_PER_QUARTER / divisions)), chord });
      }
      if (item.localName === 'backup' || item.localName === 'forward') {
        const n = Number(value(item, 'duration')) || 0;
        cursor += item.localName === 'backup' ? -n : n;
      }
      if (item.localName !== 'note' || child(item, 'grace')) continue;
      const n = Number(value(item, 'duration')) || 0;
      const chord = !!child(item, 'chord');
      const start = chord ? last : cursor;
      if (!chord) { last = cursor; cursor += n; }
      const p = child(item, 'pitch');
      if (!p || child(item, 'rest')) continue;
      const step = value(p, 'step'), alter = Math.round(Number(value(p, 'alter') || 0)), octave = Number(value(p, 'octave'));
      if (!(step in NATURAL) || !Number.isFinite(octave)) continue;
      const melody = melodyVoice !== null && !chord && (Number(value(item, 'staff')) || 1) === melodyStaff && (value(item, 'voice') || melodyVoice) === melodyVoice;
      const midi = 12 * (octave + 1) + NATURAL[step] + alter + transposition.chromatic + 12 * transposition.octave;
      if (Math.abs(start * TICKS_PER_QUARTER / divisions - Math.round(start * TICKS_PER_QUARTER / divisions)) > 1e-6
        || Math.abs((start + n) * TICKS_PER_QUARTER / divisions - Math.round((start + n) * TICKS_PER_QUARTER / divisions)) > 1e-6)
        warnings.add('Tuplets and notes shorter than a 32nd are rounded to the 32nd-note grid');
      notes.push({ start: start * TICKS_PER_QUARTER / divisions, end: (start + n) * TICKS_PER_QUARTER / divisions,
        pc: ((midi % 12) + 12) % 12, midi, weight: melody ? 0.5 : 1 });
    }
    return { notes, chords };
  });
}

const TEMPLATES: [string, number[]][] = [['', [0, 4, 7]], ['m', [0, 3, 7]], ['7', [0, 4, 7, 10]], ['m7', [0, 3, 7, 10]], ['maj7', [0, 4, 7, 11]], ['dim', [0, 3, 6]]];

/** The best-matching chord for the notes sounding in [from, to), or null if none sound.
 *  ponytail: duration-weighted pitch-class template match over six chord types,
 *  with a bonus for the lowest note as root; no voice leading, key context or
 *  slash chords. A real chord recogniser (or SheetSage on a render) if this misreads. */
function inferChord(notes: Sounding[], from: number, to: number): Chord | null {
  const chroma = new Array<number>(12).fill(0);
  let bass: Sounding | null = null;
  for (const n of notes) {
    const overlap = Math.min(n.end, to) - Math.max(n.start, from);
    if (overlap <= 0) continue;
    chroma[n.pc] += overlap * n.weight;
    if (n.weight === 1 && (!bass || n.midi < bass.midi)) bass = n;
  }
  const total = chroma.reduce((a, b) => a + b, 0);
  if (!total) return null;
  let best = { score: -Infinity, root: 0, quality: '' };
  for (let root = 0; root < 12; root++) {
    for (const [quality, intervals] of TEMPLATES) {
      const inside = intervals.reduce((s, i) => s + chroma[(root + i) % 12], 0);
      const score = 2 * inside - total - (intervals.length > 3 ? 0.15 * total : 0) + (bass?.pc === root ? 0.1 * total : 0);
      if (score > best.score) best = { score, root, quality };
    }
  }
  return { root: best.root, quality: best.quality, bass: '' };
}

function convert(part: Element, others: Element[], warnings: Set<string>, staff: number, verse: string): { abc: string; lyrics: string; chords: 'score' | 'inferred' | null } {
  let divisions = 1, keyName = 'C', key: Record<string, number> = {}, keyFifths = 0, meter = '4/4', tempo = 120;
  let firstKey = 'C', firstMeter = '4/4';
  const selectedVoice = children(part, 'measure').flatMap(m => descendants(m, 'note'))
    .find(n => (Number(value(n, 'staff')) || 1) === staff && !!child(n, 'voice'))?.querySelector('voice')?.textContent?.trim() ?? '1';
  let keySeen = false, meterSeen = false, tempoSeen = false, quantised = false;
  const measures: { implicit: boolean; events: { start: number; end: number; pitch: Pitch | null; tie: boolean; key: Record<string, number> }[];
    chords: { at: number; chord: Chord | null }[]; changes: { at: number; key?: string; meter?: string }[];
    key: Record<string, number>; keyName: string; keyFifths: number; meter: string; barTicks: number; words: string[] }[] = [];
  let transposition: Transpose = { diatonic: 0, chromatic: 0, octave: 0 };
  for (const measure of children(part, 'measure')) {
    let cursor = 0;
    const events: (typeof measures)[number]['events'] = [];
    const chords: (typeof measures)[number]['chords'] = [];
    const changes: (typeof measures)[number]['changes'] = [];
    const words: string[] = [];
    // Onsets and ends snap to the 32nd grid; a tuplet or 64th lands on its nearest tick.
    const tick = (n: number) => {
      const exact = n * TICKS_PER_QUARTER / divisions;
      if (Math.abs(exact - Math.round(exact)) > 1e-6) quantised = true;
      return Math.round(exact);
    };
    for (const item of Array.from(measure.children)) {
      if (item.localName === 'attributes') {
        const d = Number(value(item, 'divisions'));
        if (d) divisions = d;
        const trans = child(item, 'transpose');
        if (trans) transposition = { diatonic: Number(value(trans, 'diatonic')) || 0,
          chromatic: Number(value(trans, 'chromatic')) || 0, octave: Number(value(trans, 'octave-change')) || 0 };
        const k = child(item, 'key');
        if (k) {
          const fifths = Number(value(k, 'fifths'));
          const mode = value(k, 'mode') || 'major';
          const sounding = Number.isInteger(fifths) && Math.abs(fifths) <= 7
            ? soundingKey(fifths, mode, transposition.chromatic) : undefined;
          const name = sounding?.name;
          if (!name) warnings.add('Unsupported key signature: written in C with accidentals');
          else {
            if (mode !== 'major' && mode !== 'minor') warnings.add(`Key mode ${mode} is written as its major signature`);
            if (!keySeen) firstKey = name;
            else if (name !== keyName) changes.push({ at: tick(cursor), key: name });
            keyName = name; key = keyAlters(sounding!.fifths); keyFifths = sounding!.fifths; keySeen = true;
          }
        }
        const t = child(item, 'time');
        if (t) {
          const next = `${value(t, 'beats')}/${value(t, 'beat-type')}`;
          if (!/^\d+\/\d+$/.test(next)) warnings.add('Unsupported meter');
          else { if (!meterSeen) firstMeter = next; else if (next !== meter) changes.push({ at: tick(cursor), meter: next }); meter = next; meterSeen = true; }
        }
      }
      if (item.localName === 'direction') {
        const sound = child(item, 'sound');
        const q = Number(sound?.getAttribute('tempo') ?? descendants(item, 'per-minute')[0]?.textContent?.trim());
        if (q > 0) { if (tempoSeen && q !== tempo) warnings.add('Later tempo changes are ignored; the first tempo is kept'); else if (!tempoSeen) { tempo = q; tempoSeen = true; } }
      }
      if (item.localName === 'harmony') {
        const chord = harmony(item, warnings);
        if (chord !== undefined) chords.push({ at: Math.max(0, tick(cursor + (Number(value(item, 'offset')) || 0))), chord });
      }
      if (item.localName === 'backup' || item.localName === 'forward') {
        const n = Number(value(item, 'duration'));
        if (!Number.isFinite(n)) throw new MusicXmlImportError('Invalid cursor duration');
        cursor += item.localName === 'backup' ? -n : n;
      }
      if (item.localName !== 'note') continue;
      if (child(item, 'grace')) { warnings.add('Grace notes are omitted'); continue; }
      if (child(item, 'chord')) { warnings.add('Chords are omitted'); continue; }
      const n = Number(value(item, 'duration'));
      if (!Number.isFinite(n) || n <= 0) throw new MusicXmlImportError('Invalid note duration');
      const start = cursor;
      cursor = start + n;
      if (cursor < 0 || start < 0) throw new MusicXmlImportError('Invalid cursor movement');
      if (value(item, 'staff') && Number(value(item, 'staff')) !== staff) continue;
      if (value(item, 'voice') && value(item, 'voice') !== selectedVoice) continue;
      const lyric = children(item, 'lyric').find(l => (l.getAttribute('number') ?? '') === verse);
      const word = lyric ? value(lyric, 'text').replace(/\s+/g, ' ').trim() : '';
      if (word) {
        const joined = words.length && /-$/.test(words[words.length - 1]);
        const syllabic = value(lyric!, 'syllabic');
        const piece = word + (syllabic === 'begin' || syllabic === 'middle' ? '-' : '');
        if (joined) words[words.length - 1] = words[words.length - 1].slice(0, -1) + piece; else words.push(piece);
      }
      events.push({ start: tick(start), end: tick(start + n), pitch: child(item, 'rest') ? null : transposePitch(pitch(item), transposition), key: { ...key },
        tie: children(item, 'tie').some(t => t.getAttribute('type') === 'start') });
    }
    const [beats, beatType] = meter.split('/').map(Number);
    const barTicks = 32 * beats / beatType;
    if (!Number.isInteger(barTicks)) throw new MusicXmlImportError(`Meter ${meter} cannot be written in 32nd notes`);
    measures.push({ implicit: measure.getAttribute('implicit') === 'yes', events, chords, changes,
      key: { ...key }, keyName, keyFifths, meter, barTicks, words });
  }
  if (quantised) warnings.add('Tuplets and notes shorter than a 32nd are rounded to the 32nd-note grid');

  const order = playOrder(children(part, 'measure'), warnings);
  const played = order.map(i => ({ ...measures[i], events: measures[i].events.map(e => ({ ...e })), chords: [...measures[i].chords] }));
  // Chords: the score's own <harmony>, else inferred from the other staves and parts.
  // A measure whose melody part has no chord symbols takes all of its chords
  // from the first other part, in score order, that has some there; parts never mix.
  const sounding = [partHarmony(part, selectedVoice, staff, warnings), ...others.map(o => partHarmony(o, null, 0, warnings))];
  played.forEach((m, playedIndex) => {
    const i = order[playedIndex];
    if (!m.chords.length) m.chords.push(...(sounding.slice(1).find(p => p[i]?.chords.length)?.[i].chords ?? []));
  });
  let chordSource: 'score' | 'inferred' | null = played.some(m => m.chords.length) ? 'score' : null;
  if (!chordSource) {
    if (sounding.some(p => p.some(m => m.notes.some(n => n.weight === 1)))) {
      played.forEach((m, playedIndex) => {
        const notes = sounding.flatMap(p => p[order[playedIndex]]?.notes ?? []);
        // A pickup bar is read whole; any other bar by halves when the meter has an even beat count.
        const whole = m.implicit ? Math.max(0, ...m.events.map(e => e.end), ...notes.map(n => n.end)) : m.barTicks;
        const [beats] = m.meter.split('/').map(Number);
        const half = !m.implicit && beats % 2 === 0 ? whole / 2 : whole;
        const first = inferChord(notes, 0, half);
        const second = half < whole ? inferChord(notes, half, whole) : null;
        if (first ?? second) m.chords.push({ at: 0, chord: (first ?? second)! });
        if (first && second && label(second, m.keyFifths) !== label(first, m.keyFifths)) m.chords.push({ at: half, chord: second });
      });
      if (played.some(m => m.chords.length)) {
        chordSource = 'inferred';
        warnings.add('Chords are inferred from the accompaniment (the score has no chord symbols): check them before approving');
      }
    }
  }

  // A tie holds only into the melody's next note at the same pitch; a tie into a
  // skipped chord tone, another voice or a rest would be invalid in the native score.
  let offset = 0;
  const timeline = played.flatMap(m => {
    const at = offset;
    offset += m.implicit ? m.events.reduce((t, e) => Math.max(t, e.end), 0) : m.barTicks;
    return [...m.events].sort(byNumber(x => x.start)).map(e => ({ e, start: at + e.start, end: at + e.end }));
  });
  timeline.forEach((x, i) => {
    const next = timeline[i + 1];
    const p = x.e.pitch, q = next?.e.pitch;
    if (x.e.tie && !(p && q && next.start === x.end && p.step === q.step && p.octave === q.octave && p.alter === q.alter)) x.e.tie = false;
  });

  let active: string | null = null;
  const playedWords: string[] = [];
  const bars: Bar[] = played.map(({ implicit, events, chords, changes: scoreChanges, keyFifths: measureFifths, barTicks, words }) => {
    for (const word of words) { const joined = playedWords.length && /-$/.test(playedWords[playedWords.length - 1]);
      if (joined) playedWords[playedWords.length - 1] = playedWords[playedWords.length - 1].slice(0, -1) + word; else playedWords.push(word); }
    events.sort(byNumber(x => x.start));
    const letter = new Map<string, number>(), exact = new Map<string, number>();
    const used = events.reduce((t, e) => Math.max(t, e.end), 0);
    // A pickup bar is padded before its notes, any other short bar after them.
    const lead = implicit && used < barTicks ? barTicks - used : 0;
    const notes: { start: number; end: number; name: string; tie: boolean }[] = [];
    let end = 0;
    let changeIndex = 0;
    for (const event of events) {
      while (changeIndex < scoreChanges.length && scoreChanges[changeIndex].at <= event.start) {
        if (scoreChanges[changeIndex].key) { letter.clear(); exact.clear(); }
        changeIndex++;
      }
      if (event.end <= event.start) continue;
      if (event.start < end) { warnings.add('Overlapping notes are omitted'); continue; }
      if (event.pitch) notes.push({ start: lead + event.start, end: lead + event.end, name: spell(event.pitch, event.key, letter, exact), tie: event.tie });
      end = event.end;
    }
    if (lead + end > barTicks) warnings.add('Some bars are longer than the meter');
    // Only notes stretch a bar past the meter; trailing rests are cut to it.
    const barEnd = Math.max(barTicks, ...notes.map(n => n.end));
    // Chord changes in written ticks. As SheetSage writes it, the chord still
    // sounding at a bar line is repeated there, and a note a change falls inside
    // is split and tied.
    const chordChanges = new Map<number, string | null>();
    if (active) chordChanges.set(0, active);
    for (const c of [...chords].sort(byNumber(x => x.at))) chordChanges.set(Math.min(lead + c.at, barEnd - 1), c.chord && label(c.chord, measureFifths));
    const points = [...new Set([0, barEnd, ...chordChanges.keys(), ...scoreChanges.map(c => lead + c.at), ...notes.flatMap(n => [n.start, n.end])])].sort((a, b) => a - b);
    const tokens: string[] = [];
    // Adjacent rests merge into one before they are split into written lengths.
    let pending = 0;
    const flush = () => { for (const p of split(pending)) tokens.push('z' + len(p)); pending = 0; };
    for (let i = 0; i + 1 < points.length; i++) {
      const a = points[i], b = points[i + 1];
      for (const change of scoreChanges.filter(c => lead + c.at === a)) {
        flush();
        if (change.key) tokens.push(`[K:${change.key}]`);
        if (change.meter) tokens.push(`[M:${change.meter}]`);
      }
      if (chordChanges.has(a)) {
        flush();
        active = chordChanges.get(a)!;
        if (active) tokens.push(`"${active}"`);
      }
      const note = notes.find(n => n.start <= a && a < n.end);
      if (!note) { pending += b - a; continue; }
      flush();
      const parts = split(b - a);
      parts.forEach((p, k) => tokens.push(note.name.replace(/^[_^=]+/, a === note.start && k === 0 ? '$&' : '') + len(p)
        + (k < parts.length - 1 || b < note.end || note.tie ? '-' : '')));
    }
    flush();
    return { body: tokens.join(''), rest: !notes.length && !tokens.some(t => t.startsWith('"')) };
  });

  // Up to four bars per group, Vocal then Ins; whole-bar rests are Z, runs of them Z2-Z4.
  const line = (group: Bar[]) => {
    const out: string[] = [];
    for (const bar of group) {
      const last = out[out.length - 1];
      if (!bar.rest) out.push(bar.body);
      else if (last?.startsWith('Z')) out[out.length - 1] = `Z${Number(last.slice(1) || 1) + 1}`;
      else out.push('Z');
    }
    return out.join('|') + '|';
  };
  const insBars: Bar[] = played.map((m, playedIndex) => {
    const notes = sounding.flatMap(p => p[order[playedIndex]]?.notes ?? []).filter(n => n.weight === 1);
    if (!notes.length && !m.changes.length) return { body: '', rest: true };
    const limit = m.barTicks;
    const points = [...new Set([0, limit, ...m.changes.map(c => c.at), ...notes.flatMap(n => [Math.max(0, Math.min(limit, Math.round(n.start))), Math.max(0, Math.min(limit, Math.round(n.end)))])])].sort((a, b) => a - b);
    const letter = new Map<string, number>(), exact = new Map<string, number>();
    const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const tokens: string[] = [];
    for (let i = 0; i + 1 < points.length; i++) {
      const a = points[i], b = points[i + 1];
      if (b <= a) continue;
      for (const change of m.changes.filter(c => c.at === a)) {
        if (change.key) tokens.push(`[K:${change.key}]`);
        if (change.meter) tokens.push(`[M:${change.meter}]`);
      }
      const note = notes.filter(n => n.start < b && n.end > a).sort((x, y) => y.midi - x.midi)[0];
      if (!note) { for (const length of split(b - a)) tokens.push('z' + len(length)); continue; }
      const name = names[((note.midi % 12) + 12) % 12];
      const p: Pitch = { step: name[0], alter: name.length > 1 ? 1 : 0, octave: Math.floor(note.midi / 12) - 1 };
      const spelled = spell(p, m.key, letter, exact);
      const parts = split(b - a);
      parts.forEach((length, j) => tokens.push(spelled.replace(/^[_^=]+/, j === 0 ? '$&' : '') + len(length) + (j < parts.length - 1 ? '-' : '')));
    }
    return { body: tokens.join(''), rest: false };
  });
  const lines = ['X:1', 'T:', `M:${firstMeter}`, 'L:1/32', `Q:1/4=${Math.round(tempo)}`,
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"', 'V: Ins clef=treble name="Ins Melody" snm="Inst."',
    `K:${firstKey}`, '% verse'];
  for (let i = 0; i < bars.length; i += 4) {
    const group = bars.slice(i, i + 4);
    lines.push('V: Vocal', line(group), 'V: Ins', line(insBars.slice(i, i + 4)));
  }
  return { abc: lines.join('\n') + '\n', lyrics: playedWords.length ? `[Verse]\n${playedWords.join(' ')}\n` : '', chords: chordSource };
}

export async function importMusicXml(file: Blob, fileName: string, opts: { partId?: string; staff?: number; verse?: string } = {}): Promise<MusicXmlResult> {
  if (!/\.(musicxml|xml|mxl)$/i.test(fileName)) throw new MusicXmlImportError('Choose a .musicxml, .xml or .mxl file');
  if (file.size > MAX_FILE) throw new MusicXmlImportError('MusicXML file is too large');
  let source: string;
  if (/\.mxl$/i.test(fileName)) {
    let entries: Record<string, Uint8Array>;
    let count = 0, totalSize = 0;
    try { entries = unzipSync(new Uint8Array(await file.arrayBuffer()), { filter: e => {
      totalSize += e.originalSize;
      if (++count > MAX_ENTRIES || e.originalSize > MAX_XML || totalSize > 32 * 1024 * 1024) throw new MusicXmlImportError('MXL archive exceeds limits');
      return e.name === 'META-INF/container.xml' || e.name.endsWith('.xml') || e.name.endsWith('.musicxml');
    } }); }
    catch { throw new MusicXmlImportError('Invalid MXL archive'); }
    if (Object.keys(entries).length > MAX_ENTRIES || Object.values(entries).some(e => e.length > MAX_XML)) throw new MusicXmlImportError('MXL archive exceeds limits');
    const container = entries['META-INF/container.xml'];
    if (!container) throw new MusicXmlImportError('MXL container.xml is missing');
    const root = xml(strFromU8(container)).querySelector('rootfile')?.getAttribute('full-path');
    if (!root || root.includes('..') || root.startsWith('/') || !entries[root]) throw new MusicXmlImportError('MXL rootfile is missing');
    source = strFromU8(entries[root]);
  } else source = await file.text();
  const doc = xml(source);
  const nodes = partwise(doc);
  const parts = nodes.map(p => {
    const id = p.getAttribute('id') ?? '';
    const info = descendants(doc.documentElement, 'score-part').find(x => x.getAttribute('id') === id);
    return { id, name: info ? value(info, 'part-name') : id, staves: Math.max(1, ...descendants(p, 'staves').map(x => Number(x.textContent) || 1), ...descendants(p, 'staff').map(x => Number(x.textContent) || 1)), voices: new Set(descendants(p, 'voice').map(x => x.textContent?.trim())).size || 1, hasLyrics: descendants(p, 'lyric').length > 0,
      verses: [...new Set(descendants(p, 'lyric').map(x => x.getAttribute('number') ?? ''))] };
  });
  if (!parts.length) throw new MusicXmlImportError('No parts found');
  const title = clean(doc.querySelector('work-title, movement-title')?.textContent ?? '');
  const selected = opts.partId ? nodes.find(p => p.getAttribute('id') === opts.partId) : nodes.length === 1 ? nodes[0] : undefined;
  if (opts.partId && !selected) throw new MusicXmlImportError('Selected part was not found');
  if (!selected) return { abc: '', title, lyrics: '', chords: null, parts, partId: null, warnings: [] };
  const warnings = new Set<string>();
  const info = parts.find(p => p.id === selected.getAttribute('id'))!;
  const staff = opts.staff ?? 1;
  if (staff < 1 || staff > info.staves) throw new MusicXmlImportError('Selected staff was not found');
  const verse = opts.verse !== undefined && info.verses.includes(opts.verse)
    ? opts.verse : info.verses.includes('1') ? '1' : info.verses[0] ?? '';
  if (info.voices > 1) warnings.add('Multiple voices: the first voice on the chosen staff is imported');
  const result = convert(selected, nodes.filter(p => p !== selected), warnings, staff, verse);
  if (result.abc.length > 5120) {
    const playedBars = [...result.abc.matchAll(/^V: Vocal\n([^\n]+)/gm)].flatMap(m => m[1].split('|').filter(isPresent)).length;
    const near = Math.max(1, Math.floor(5120 / result.abc.length * playedBars));
    warnings.add(`This score may pass YuE2's 5,120-token limit and be cut near played bar ${near} (estimated from ABC length); check the final bars`);
  }
  return { ...result, title, parts, partId: info.id, warnings: [...warnings] };
}
