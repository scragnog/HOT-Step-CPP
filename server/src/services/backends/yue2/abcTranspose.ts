import abcjs from 'abcjs';

// Transposes a native YuE2 score (MusicXML import or SheetSage2) by a signed
// number of semitones, in place of abcjs.strTranspose, which mis-spells a note
// whose accidental cancels a key-signature or bar-carried accidental.
// Notes are read the way abcjs reads them: per voice, an accidental carries to
// its own letter and octave until the bar line, and a tie continuation keeps
// the tied pitch. Each note moves by its key's letter interval and is respelled
// with the MusicXML importer's rule: mark it unless both YuE2's letter-wide
// carry and abcjs's octave carry already give its pitch.

const STEPS = 'CDEFGAB';
const NATURAL = [0, 2, 4, 5, 7, 9, 11];
const ALTER: Record<string, number> = { '': 0, '=': 0, '^': 1, '^^': 2, _: -1, __: -2 };
// Index - 7 = fifths.
const MAJOR = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#'];
const MINOR = ['Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#'];
const TOKEN = /%.*$|"[^"]*"|\[[A-Za-z]:[^\]]*\]|!.*?!|(\^\^|\^|__|_|=)?([A-Ga-g])([,']*)|./g;

interface Key {
  /** Signature by letter index, before and after. */
  source: number[];
  target: number[];
  /** Diatonic steps every note in this key moves by. */
  steps: number;
}

interface Voice {
  key: Key;
  /** Bar-carried accidentals: source by diatonic index, output by letter and by index. */
  source: Map<number, number>;
  letter: Map<number, number>;
  exact: Map<number, number>;
  last: { index: number; pitch: number } | null;
  tie: { index: number; pitch: number } | null;
}

const mod = (n: number, m: number) => ((n % m) + m) % m;
const pc = (name: string) => mod(NATURAL[STEPS.indexOf(name[0])] + (name[1] === '#' ? 1 : name[1] === 'b' ? -1 : 0), 12);
const midi = (index: number, alter: number) => 12 * (Math.floor(index / 7) + 1) + NATURAL[mod(index, 7)] + alter;

function signature(fifths: number): number[] {
  const sig = [0, 0, 0, 0, 0, 0, 0];
  for (const l of (fifths > 0 ? 'FCGDAEB' : 'BEADGCF').slice(0, Math.abs(fifths))) sig[STEPS.indexOf(l)] = Math.sign(fifths);
  return sig;
}

/** Fifths of a mode's signature below its tonic's major signature, by the
 *  ABC mode name (first three letters; "m" alone is minor). SheetSage2-Refined
 *  writes dor/phr/lyd/mix/loc keys. */
const MODE_FIFTHS: Record<string, number> = {
  '': 0, maj: 0, ion: 0, m: -3, min: -3, aeo: -3, dor: -2, phr: -4, lyd: 1, mix: -1, loc: -5,
};

/** The 15 standard tonic names of a mode, by signature fifths - (-7). */
function modeNames(delta: number): string[] {
  if (delta === 0) return MAJOR;
  if (delta === -3) return MINOR;
  return Array.from({ length: 15 }, (_, i) => {
    const x = i - 7 - delta + 1; // the tonic's fifths from F
    const acc = Math.floor(x / 7);
    return 'FCGDAEB'[mod(x, 7)] + (acc > 0 ? '#' : acc < 0 ? 'b' : '');
  });
}

/** A K: value moved by `semitones`: the standard name with the fewest
 *  accidentals, ties broken toward the source key's side, then the shift's. */
export function transposeKeyField(field: string, semitones: number): { text: string; key: Key } {
  const m = field.match(/^(\s*)([A-G][#b]?)(.*)$/);
  const rest = m?.[3] ?? '';
  const word = rest.trim().split(/\s+/)[0].toLowerCase();
  const mode = word.includes('=') ? '' : word; // clef=, middle= and the like are not a mode
  const delta = MODE_FIFTHS[mode === 'm' ? 'm' : mode.slice(0, 3)];
  if (!m || delta === undefined) {
    throw new Error(`Cannot transpose key "${field.trim()}": unsupported key mode.`);
  }
  const names = modeNames(delta);
  const from = names.indexOf(m[2]) - 7;
  if (from < -7) throw new Error(`Cannot transpose key "${field.trim()}": not a standard key name.`);
  const want = mod(pc(m[2]) + semitones, 12);
  const to = names.map((n, i) => ({ n, f: i - 7 })).filter(c => pc(c.n) === want)
    .sort((a, b) => Math.abs(a.f) - Math.abs(b.f) || Math.abs(a.f - from) - Math.abs(b.f - from)
      || (semitones >= 0 ? b.f - a.f : a.f - b.f))[0];
  const letters = mod(STEPS.indexOf(to.n[0]) - STEPS.indexOf(m[2][0]), 7);
  const steps = letters + 7 * Math.round((semitones * 7 / 12 - letters) / 7);
  return { text: m[1] + to.n + rest, key: { source: signature(from), target: signature(to.f), steps } };
}

/** `pitch` spelled `steps` letters from `index`, or on a neighbouring letter
 *  when that would take a double sharp or flat: YuE2's own speller
 *  (compile_score.spelling) never writes one. */
function respell(index: number, pitch: number, steps: number): { index: number; alter: number } {
  const at = (i: number) => ({ index: i, alter: pitch - midi(i, 0) });
  return [at(index + steps), at(index + steps - 1), at(index + steps + 1)].find(s => Math.abs(s.alter) <= 1)!;
}

function chordSymbol(text: string, key: Key, semitones: number): string {
  const m = text.match(/^"([A-G])([#b]?)([^"/]*)(?:\/([A-G])([#b]?))?"$/);
  if (!m) return text; // an annotation, not a chord symbol
  const move = (letter: string, acc: string) => {
    const index = 28 + STEPS.indexOf(letter);
    const to = respell(index, midi(index, ALTER[acc === '#' ? '^' : acc === 'b' ? '_' : '']) + semitones, key.steps);
    return STEPS[mod(to.index, 7)] + (to.alter > 0 ? '#' : to.alter < 0 ? 'b' : '');
  };
  return `"${move(m[1], m[2])}${m[3]}${m[4] ? `/${move(m[4], m[5])}` : ''}"`;
}

function note(voice: Voice, acc: string | undefined, name: string, octaves: string, semitones: number): string {
  const step = STEPS.indexOf(name.toUpperCase());
  const index = 7 * ((name === name.toUpperCase() ? 4 : 5)
    + (octaves.match(/'/g)?.length ?? 0) - (octaves.match(/,/g)?.length ?? 0)) + step;
  const { key } = voice;
  const tie = voice.tie?.index === index ? voice.tie : null;
  voice.tie = null;
  let pitch: number;
  if (tie) pitch = tie.pitch; // abcjs ignores a tie continuation's mark, for pitch and for the bar
  else {
    if (acc !== undefined) voice.source.set(index, ALTER[acc]);
    pitch = midi(index, voice.source.get(index) ?? key.source[step]);
  }
  voice.last = { index, pitch };
  const to = respell(index, pitch + semitones, key.steps);
  const toStep = mod(to.index, 7);
  let mark = '';
  if (!tie && ((voice.letter.get(toStep) ?? key.target[toStep]) !== to.alter
    || (voice.exact.get(to.index) ?? key.target[toStep]) !== to.alter)) {
    mark = to.alter > 0 ? '^'.repeat(to.alter) : to.alter < 0 ? '_'.repeat(-to.alter) : '=';
    voice.letter.set(toStep, to.alter);
    voice.exact.set(to.index, to.alter);
  }
  const octave = Math.floor(to.index / 7);
  return mark + (octave >= 5 ? STEPS[toStep].toLowerCase() + "'".repeat(octave - 5) : STEPS[toStep] + ','.repeat(4 - octave));
}

/** MIDI pitch of every written note, per track, as abcjs plays the text.
 *  abcjs's chord-symbol accompaniment (events with no source position) is
 *  left out: its bass voicing depends on the chord's name, not just its pitch. */
export function soundingPitches(abc: string): number[][] {
  const tune = abcjs.parseOnly(abc)[0] as any;
  return tune.setUpAudio().tracks.map((track: any[]) => track
    .filter(e => e.cmd === 'note' && e.startChar !== undefined).map(e => e.pitch));
}

/** Moves every note, key and chord symbol by `semitones`, then checks with
 *  abcjs that each note sounds exactly that far from where it was. */
export function transposeAbc(abc: string, semitones: number): string {
  if (!Number.isInteger(semitones)) throw new Error('Transposition needs a whole number of semitones.');
  if (semitones === 0) return abc;
  if (!/^K:/m.test(abc)) throw new Error('Cannot transpose a score without K:.');
  const voices = new Map<string, Voice>();
  let fallback: Key | null = null;
  let current: Voice | null = null;
  const voice = (name: string) => {
    if (!voices.has(name)) {
      voices.set(name, { key: fallback!, source: new Map(), letter: new Map(), exact: new Map(), last: null, tie: null });
    }
    return voices.get(name)!;
  };
  const setKey = (field: string) => {
    const moved = transposeKeyField(field, semitones);
    if (current) current.key = moved.key;
    else { fallback = moved.key; for (const v of voices.values()) v.key = moved.key; }
    return moved.text;
  };
  const lines = abc.split(/(\r?\n)/);
  for (let i = 0; i < lines.length; i += 2) {
    const line = lines[i];
    const field = line.match(/^K:(.*)$/);
    if (field) { lines[i] = `K:${setKey(field[1])}`; continue; }
    if (!fallback) continue; // header
    const v = line.match(/^V:\s*(\S+)/);
    if (v) { current = voice(v[1]); continue; }
    if (/^([A-Za-z]:|%)/.test(line)) continue;
    current ??= voice('');
    lines[i] = line.replace(TOKEN, (t, acc?: string, name?: string, octaves?: string) => {
      const cv = current!;
      if (name) return note(cv, acc, name, octaves!, semitones);
      if (t.startsWith('"')) return chordSymbol(t, cv.key, semitones);
      const inline = t.match(/^\[([A-Za-z]):(.*)\]$/);
      if (inline?.[1] === 'K') return `[K:${setKey(inline[2])}]`;
      if (inline?.[1] === 'V') { current = voice(inline[2].trim().split(/\s/)[0]); return t; }
      if (t === '-') { cv.tie = cv.last; return t; }
      if (t === '|') { cv.source.clear(); cv.letter.clear(); cv.exact.clear(); }
      if (/[zxZ]/.test(t)) { cv.tie = null; cv.last = null; }
      return t;
    });
  }
  const result = lines.join('');
  const before = soundingPitches(abc), after = soundingPitches(result);
  const track = before.length === after.length ? before.findIndex((t, i) => t.length !== after[i].length
    || t.some((p, j) => after[i][j] - p !== semitones)) : 0;
  if (track >= 0) {
    const j = before[track]?.findIndex((p, n) => after[track]?.[n] - p !== semitones);
    throw new Error(`Transposition by ${semitones} semitones did not move every note by exactly that much `
      + `(track ${track + 1}, note ${j + 1}: ${before[track]?.[j]} became ${after[track]?.[j]}).`);
  }
  return result;
}
