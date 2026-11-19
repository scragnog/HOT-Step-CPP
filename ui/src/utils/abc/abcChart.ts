// A song-chart view of a transcribed score: its header (tempo, meter, key)
// and, per section, one card per bar with its chords and the Vocal and Ins
// melodies as note names. Edits rewrite only the characters they touch, at
// offsets into the score with its line endings turned into LF (as the
// section move does), so the rest of the text is unchanged.

import { isVocal, parseScore, type Bar } from './abcSectionMove';

/** `held`: the continuation of a note tied over from the previous bar.
 *  `tie`: the note is tied into the next bar. */
export interface ChartNote { name: string; midi: number | null; length: number; held?: boolean; tie?: boolean }
/** A chord symbol: its text and the offsets of the whole `"…"` token. */
export interface ChartChord { text: string; start: number; end: number }
export interface ChartBar {
  number: number;
  chords: ChartChord[];
  /** Where a new chord goes, or null on the later bars of a multi-bar rest. */
  chordAt: number | null;
  vocal: ChartNote[];
  ins: ChartNote[] | null;
}
export interface ChartSection { label: string; index: number; bars: ChartBar[] }
export interface Chart {
  tempo: number | null; meter: string; unit: string; key: string;
  sections: ChartSection[];
}

// Sharps (positive) or flats (negative) in each key signature.
const MAJOR: Record<string, number> = { C: 0, G: 1, D: 2, A: 3, E: 4, B: 5, 'F#': 6, 'C#': 7,
  F: -1, Bb: -2, Eb: -3, Ab: -4, Db: -5, Gb: -6, Cb: -7 };
const MINOR: Record<string, number> = { A: 0, E: 1, B: 2, 'F#': 3, 'C#': 4, 'G#': 5, 'D#': 6, 'A#': 7,
  D: -1, G: -2, C: -3, F: -4, Bb: -5, Eb: -6, Ab: -7 };
const SEMITONE: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

/** The accidental (in semitones) the key signature gives each letter. */
export function keyAccidentals(key: string): Record<string, number> {
  const m = key.trim().match(/^([A-G])([#b]?)\s*(m(?:in(?:or)?)?)?(?![a-z])/i);
  const out: Record<string, number> = { C: 0, D: 0, E: 0, F: 0, G: 0, A: 0, B: 0 };
  if (!m) return out;
  const tonic = m[1].toUpperCase() + m[2];
  const count = (m[3] ? MINOR : MAJOR)[tonic] ?? 0;
  const order = count > 0 ? 'FCGDAEB' : 'BEADGCF';
  for (const letter of order.slice(0, Math.abs(count))) out[letter] = Math.sign(count);
  return out;
}

function length(text: string): number {
  if (!text) return 1;
  const m = text.match(/^(\d*)(\/*)(\d*)$/);
  if (!m) return 1;
  const num = m[1] ? Number(m[1]) : 1;
  const den = m[3] ? Number(m[3]) : 2 ** m[2].length;
  return m[2] ? num / den : num;
}

const NOTE = /(\^\^|\^|__|_|=)?([A-Ga-g])([',]*)/;
const TOKEN = new RegExp(String.raw`"[^"]*"|![^!]*!|\{[^}]*\}|\[[A-Za-z]:[^\]]*\]|\[((?:${NOTE.source})+)\]([\d/]*)|${NOTE.source}([\d/]*)|([zx])([\d/]*)|Z(\d*)`, 'g');

/** A bar's notes and rests, with pitches spelled as they sound: an explicit
 *  accidental holds for that pitch to the end of the bar, as in ABC. Tied
 *  notes within the bar are merged into one; `heldIn` marks the first note
 *  as tied over from the previous bar. */
export function barNotes(text: string, key: Record<string, number>, heldIn = false): ChartNote[] {
  const notes: ChartNote[] = [];
  let tied = heldIn;
  const push = (note: ChartNote, end: number) => {
    const last = notes[notes.length - 1];
    if (tied && last && last.midi === note.midi && note.midi !== null) last.length += note.length;
    else notes.push(tied && !last && note.midi !== null ? { ...note, held: true } : note);
    tied = /^\s*-/.test(text.slice(end));
    if (tied) notes[notes.length - 1].tie = true;
    else delete notes[notes.length - 1].tie;
  };
  const held = new Map<string, number>();
  const pitch = (acc: string | undefined, letter: string, octave: string) => {
    const upper = letter.toUpperCase();
    const where = `${letter}${octave}`;
    if (acc) held.set(where, acc === '=' ? 0 : (acc.startsWith('^') ? 1 : -1) * acc.length);
    const shift = held.get(where) ?? key[upper];
    let midi = 60 + SEMITONE[upper] + shift + (letter === upper ? 0 : 12);
    for (const mark of octave) midi += mark === "'" ? 12 : -12;
    return { name: upper + (shift > 0 ? '#'.repeat(shift) : 'b'.repeat(-shift)), midi };
  };
  for (const m of text.matchAll(TOKEN)) {
    const t = m[0];
    if (t.startsWith('"') || t.startsWith('!') || t.startsWith('{') || /^\[[A-Za-z]:/.test(t)) continue;
    if (m[1] !== undefined) {
      // A chord of notes: shown by its top note.
      const inner = [...m[1].matchAll(new RegExp(NOTE.source, 'g'))].map(n => pitch(n[1], n[2], n[3]));
      const top = inner.reduce((a, b) => (b.midi > a.midi ? b : a));
      push({ ...top, length: length(m[5]) }, m.index! + t.length);
    } else if (m[7] !== undefined) {
      push({ ...pitch(m[6], m[7], m[8]), length: length(m[9]) }, m.index! + t.length);
    } else if (m[10] !== undefined) {
      push({ name: 'rest', midi: null, length: length(m[11]) }, m.index! + t.length);
    }
  }
  return notes;
}

/** Chord symbols in a bar, by offset into the score. */
function barChords(bar: Bar): ChartChord[] {
  return [...bar.text.matchAll(/"([^"]*)"/g)]
    .filter(m => !/^[_^<>@]/.test(m[1])) // annotations, not chords
    .map(m => ({ text: m[1], start: bar.start + m.index!, end: bar.start + m.index! + m[0].length }));
}

const header = (abc: string, field: string) =>
  abc.replace(/\r\n?/g, '\n').match(new RegExp(`^${field}:\\s*([^\\n%]*)`, 'm'))?.[1].trim() ?? '';

/** The score as a chart, or an error message when it cannot be read. */
export function abcChart(abc: string): Chart | string {
  const parsed = parseScore(abc);
  if (typeof parsed === 'string') return parsed;
  const key = keyAccidentals(header(abc, 'K'));
  const tempo = Number(header(abc, 'Q').match(/=\s*(\d+(?:\.\d+)?)/)?.[1]) || null;
  const sections: ChartSection[] = [{ label: 'Start', index: -1, bars: [] }];
  let markers = 0;
  // Whether the last note so far in each voice is tied into the next bar.
  let vocalTie: boolean | undefined, insTie: boolean | undefined;
  for (const item of parsed.items) {
    if (item.kind === 'marker') {
      sections.push({ label: item.text.trim().slice(1).trim(), index: markers++, bars: [] });
      continue;
    }
    if (item.kind !== 'group') continue;
    const { group } = item;
    const vocal = group.parts.find(part => part.header && isVocal(part.header)) ?? group.parts[0];
    const ins = group.parts.find(part => part !== vocal);
    // Ins bars by position in the group, a multi-bar rest filling its bars.
    const insBars: ChartNote[][] = [];
    for (const bar of ins?.bars ?? [])
      for (let i = 0; i < bar.count; i++) {
        insBars.push(i ? [] : barNotes(bar.text, key, !!insTie));
        insTie = insBars[insBars.length - 1].at(-1)?.tie;
      }
    const target = sections[sections.length - 1];
    let at = 0;
    for (const bar of vocal.bars) {
      for (let i = 0; i < bar.count; i++) {
        target.bars.push({
          number: group.firstBar + at, chords: i ? [] : barChords(bar),
          chordAt: i ? null : bar.start + (bar.text.length - bar.text.trimStart().length),
          vocal: i ? [] : barNotes(bar.text, key, !!vocalTie), ins: ins ? insBars[at] ?? [] : null,
        });
        vocalTie = target.bars[target.bars.length - 1].vocal.at(-1)?.tie;
        at++;
      }
    }
  }
  return { tempo, meter: header(abc, 'M'), unit: header(abc, 'L'), key: header(abc, 'K'),
    sections: sections.filter(section => section.index >= 0 || section.bars.length) };
}

const lf = (abc: string) => abc.replace(/\r\n?/g, '\n');
const splice = (text: string, start: number, end: number, insert: string) => text.slice(0, start) + insert + text.slice(end);

/** Sets a chord's text; an empty text removes the chord. */
export function setChord(abc: string, chord: ChartChord, text: string): string {
  const value = text.trim().replace(/"/g, '');
  return splice(lf(abc), chord.start, chord.end, value ? `"${value}"` : '');
}

/** Adds a chord at the start of a bar. */
export function addChord(abc: string, at: number, text: string): string {
  const value = text.trim().replace(/"/g, '');
  return value ? splice(lf(abc), at, at, `"${value}"`) : lf(abc);
}

/** Sets the tempo in the header's Q: line, keeping its beat unit, or adds
 *  one before the K: line. */
export function setTempo(abc: string, bpm: number): string {
  const text = lf(abc);
  const lines = text.split('\n');
  const keyLine = lines.findIndex(line => /^K:/.test(line.trim()));
  const q = lines.findIndex((line, i) => i < keyLine && /^Q:/.test(line.trim()));
  const value = Math.round(bpm);
  if (q >= 0) {
    const beat = lines[q].match(/^Q:\s*([^=]*)=/)?.[1].trim() || '1/4';
    lines[q] = `Q:${beat}=${value}`;
  } else if (keyLine >= 0) {
    lines.splice(keyLine, 0, `Q:1/4=${value}`);
  }
  return lines.join('\n');
}
