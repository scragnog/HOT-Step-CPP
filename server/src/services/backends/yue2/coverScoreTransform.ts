import abcjs from 'abcjs';
import { transposeAbc, transposeKeyField } from './abcTranspose.js';
import { coverScoreSnapshot } from './coverScore.js';

export type CoverVoices = 'vocal' | 'both';
export type CoverTempo = 'source' | 'free' | number;
export type CoverKey = 'source' | string;

export interface CoverScoreChoices {
  voices: CoverVoices;
  keepChords: boolean;
  tempo: CoverTempo;
  key: CoverKey;
  /** The exact signed semitone shift that produced `key`, when the caller has
   *  one. Key names alone only carry a pitch class (mod 12), so without this
   *  a +7 or +12 request folds to the nearest route (e.g. -5) and loses the
   *  requested direction/octave; see coverTranspose. */
  keyShift?: number;
  cfgScale: number;
}

/** Submit-time defaults; every captured cover writes all five choices. */
export const COVER_SCORE_DEFAULTS: CoverScoreChoices = {
  voices: 'both', keepChords: false, tempo: 'free', key: 'source', cfgScale: 1,
};

export function coverScoreChoices(raw: Partial<CoverScoreChoices>): CoverScoreChoices {
  const choices = { ...COVER_SCORE_DEFAULTS, ...raw };
  if (choices.voices !== 'vocal' && choices.voices !== 'both') throw new Error('yue2Cover.voices must be vocal or both.');
  if (typeof choices.keepChords !== 'boolean') throw new Error('yue2Cover.keepChords must be a boolean.');
  if (choices.tempo !== 'source' && choices.tempo !== 'free' &&
      (typeof choices.tempo !== 'number' || !Number.isFinite(choices.tempo) || choices.tempo < 20 || choices.tempo > 300)) {
    throw new Error('yue2Cover.tempo must be source, free or 20–300 BPM.');
  }
  if (typeof choices.key !== 'string' || !choices.key.trim()) throw new Error('yue2Cover.key must be source or a target key.');
  if (choices.keyShift !== undefined && (!Number.isInteger(choices.keyShift) || choices.keyShift < -12 || choices.keyShift > 12)) {
    throw new Error('yue2Cover.keyShift must be an integer between -12 and 12.');
  }
  if (typeof choices.cfgScale !== 'number' || !Number.isFinite(choices.cfgScale) || choices.cfgScale <= 0 || choices.cfgScale > 2) {
    throw new Error('yue2Cover.cfgScale must be greater than 0 and at most 2.');
  }
  return choices;
}

function eachLine(abc: string, map: (line: string, newline: string) => string): string {
  const parts = abc.split(/(\r?\n)/);
  let result = '';
  for (let i = 0; i < parts.length; i += 2) result += map(parts[i], parts[i + 1] ?? '');
  return result;
}

/** SheetSage declares both voices before the first K:, then switches voice for each body block. */
export function coverVocalOnly(abc: string): string {
  let inHeader = true;
  let inIns = false;
  return eachLine(abc, (line, newline) => {
    if (/^K:/.test(line)) inHeader = false;
    if (/^V:\s*Ins\b/i.test(line)) { if (!inHeader) inIns = true; return ''; }
    if (/^V:\s*/.test(line) || /^\s*%/.test(line)) inIns = false;
    return inIns ? '' : line + newline;
  });
}

export function coverTempo(abc: string, tempo: CoverTempo): string {
  if (tempo === 'source') return abc;
  let found = false;
  let result = eachLine(abc, (line, newline) => {
    if (!/^Q:/.test(line)) return line + newline;
    found = true;
    if (tempo === 'free') return '';
    const match = line.match(/^(Q:\s*[^=\r\n]+?=\s*)(\d+(?:\.\d+)?)(.*)$/);
    if (!match) throw new Error('Cannot set tempo: unsupported Q: field.');
    return `${match[1]}${tempo}${match[3]}${newline}`;
  });
  if (tempo !== 'free' && !found) {
    const nl = abc.includes('\r\n') ? '\r\n' : '\n';
    result = result.replace(/^K:/m, `Q:1/4=${tempo}${nl}K:`);
    if (result === abc) throw new Error('Cannot set tempo: score has no K: field.');
  }
  return result;
}

const PITCH_CLASS: Record<string, number> = {
  C: 0, 'B#': 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3, E: 4, Fb: 4,
  'E#': 5, F: 5, 'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9,
  'A#': 10, Bb: 10, B: 11, Cb: 11,
};

function parsedKey(value: string): { pitch: number; minor: boolean; mode: string; suffix: string; wordMode: boolean } {
  // ABC modes as SheetSage2-Refined writes them (Ddor, Gmix, ...) besides major and minor.
  const match = value.match(/^\s*([A-Ga-g])([#b]?)(mix|dor|phr|lyd|loc|m)?(?:(\s+)(minor|major|maj))?(\s+.*)?$/i);
  if (!match) throw new Error(`Unsupported cover key: ${value}`);
  const root = match[1].toUpperCase() + match[2];
  const short = match[3]?.toLowerCase();
  const minor = short === 'm' || match[5]?.toLowerCase() === 'minor';
  return { pitch: PITCH_CLASS[root], minor, mode: minor ? 'minor' : short ?? 'major',
    suffix: `${match[4] ?? ''}${match[5] ?? ''}${match[6] ?? ''}`, wordMode: !!match[5] };
}

/** transposeAbc moves keys, inline keys, notes and chord roots/bass together.
 *  `targetKey` only carries a pitch class (mod 12): C and a request for +7 or
 *  -5 both name "G", so without the caller's exact signed shift this always
 *  picks the nearest route (<= 6 semitones), silently reinterpreting a wider
 *  or opposite-direction request. Pass `keyShift` whenever the caller knows
 *  the real signed distance (e.g. a UI pitch-shift slider) so it is used
 *  as-is instead of re-derived. */
export function coverTranspose(abc: string, targetKey: CoverKey, keyShift?: number): string {
  if (targetKey === 'source') return abc;
  const sourceField = abc.match(/^K:\s*([^\r\n]+)/m);
  if (!sourceField) throw new Error('Cannot transpose a cover score without K:.');
  const source = parsedKey(sourceField[1]);
  const target = parsedKey(targetKey);
  if (source.mode !== target.mode) throw new Error('Cover transposition keeps the source key mode.');
  let steps: number;
  if (typeof keyShift === 'number' && keyShift !== 0) {
    if (((source.pitch + keyShift) % 12 + 12) % 12 !== target.pitch) {
      throw new Error('yue2Cover.keyShift does not land on the requested key.');
    }
    steps = keyShift;
  } else {
    steps = (target.pitch - source.pitch + 12) % 12;
    if (steps > 6) steps -= 12;
  }
  if (steps === 0) return abc;
  const result = transposeAbc(abc, steps);
  if (parsedKey(result.match(/^K:\s*([^\r\n]+)/m)![1]).pitch !== target.pitch) {
    throw new Error('Could not transpose the cover score to the requested key.');
  }
  return result;
}

/** The order is part of the cover contract: voices, chords, tempo, key. */
export function transformCoverScore(fullScore: string, choices: CoverScoreChoices) {
  const voices = choices.voices === 'vocal' ? coverVocalOnly(fullScore) : fullScore;
  const chords = coverScoreSnapshot(voices, choices.keepChords).renderedAbc;
  const tempo = coverTempo(chords, choices.tempo);
  return { fullScore, renderedAbc: coverTranspose(tempo, choices.key, choices.keyShift) };
}

// ── 2/4 to 4/4 re-metering ──
//
// Pairs consecutive 2/4 bars into 4/4 bars in every voice at once. Section
// markers stay where they are, so a section always starts on a bar line: a
// 2/4 bar left without a partner before a marker, a meter change or the end
// stays a 2/4 bar under its own M: line. With `pickup`, the score's first bar
// stays 2/4 too and pairing starts at bar 2. Notes are copied unchanged except
// where a removed bar line would carry an accidental into the second half;
// there the note gets the accidental it had. abcjs then checks that every
// note keeps its pitch, start and length, as transposeAbc does.

const STEP_LETTERS = 'CDEFGAB';
const MARK: Record<string, number> = { '=': 0, '^': 1, '^^': 2, _: -1, __: -2 };
const NOTE_TOKEN = /"[^"]*"|\[[A-Za-z]:[^\]]*\]|!.*?!|(\^\^|\^|__|_|=)?([A-Ga-g])([,']*)|./g;
const UNSUPPORTED_BAR = /\|\||\|:|:\||\[\||\|\]|::|\[\d|\|\d|&|\\|%|\[[LMQVIPw]:/;

/** `keys`: the K: lines just before this bar. */
interface RemeterBar { text: string; meter: string; rest: boolean; keys: string[] }
interface NoteState { sig: number[]; tie: number | null; last: number | null }
interface Carry { exact: Map<number, number>; letter: Map<number, number> }

function fraction(text: string) {
  const m = /^(\d+)\/(\d+)$/.exec(text.trim());
  if (!m || !Number(m[1]) || !Number(m[2])) throw new Error(`Unsupported meter or length "${text.trim()}".`);
  return { n: Number(m[1]), d: Number(m[2]), text: `${Number(m[1])}/${Number(m[2])}` };
}
const keySignature = (field: string) => transposeKeyField(field, 0).key.source;
const unquoted = (text: string) => text.replace(/"[^"]*"/g, '');

/** A music line's bars; a multi-bar rest `Z4` becomes four one-bar rests. */
function remeterBars(line: string, meter: string, unit: { n: number; d: number }): RemeterBar[] {
  if (UNSUPPORTED_BAR.test(unquoted(line))) {
    throw new Error('Repeats, endings, double bars, voice overlays, comments and inline meter or length changes are not supported.');
  }
  const parts: string[] = [];
  let start = 0, quoted = false;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === '"') quoted = !quoted;
    if (line[i] === '|' && !quoted) { parts.push(line.slice(start, i)); start = i + 1; }
  }
  if (line.slice(start).trim()) throw new Error('Every music line must end with a bar line.');
  return parts.filter(text => text.trim()).flatMap(text => {
    const multi = unquoted(text).trim().match(/^Z(\d*)$/);
    if (!multi) return [{ text, meter, rest: !!meter && restUnits(text) === barUnits(meter, unit), keys: [] }];
    const prefix = text.slice(0, text.lastIndexOf('Z'));
    return Array.from({ length: Math.max(1, Number(multi[1] || 1)) }, (_, i) =>
      ({ text: `${i ? '' : prefix}Z`, meter, rest: i > 0 || !prefix.trim(), keys: [] }));
  });
}

/** L: units in a bar of plain rests, or null for anything else. Only a full
 *  bar of rests may become a `Z`: a short one would gain the missing time. */
function restUnits(text: string): number | null {
  if (!/^\s*(?:[zx]\d*(?:\/\d*)?\s*)+$/.test(text)) return null;
  let units = 0;
  for (const [, n, slash, d] of text.matchAll(/[zx](\d*)(\/?)(\d*)/g)) units += Number(n || 1) / (slash ? Number(d || 2) : 1);
  return units;
}

/** L: units in one bar of `meter`. */
function barUnits(meter: string, unit: { n: number; d: number }): number {
  const m = fraction(meter), units = (m.n * unit.d) / (m.d * unit.n);
  if (!Number.isInteger(units)) throw new Error(`A ${m.text} bar is not a whole number of L: units.`);
  return units;
}

/** The meter's default L: under 3/4 it is 1/16, otherwise 1/8. */
const impliedLength = (meter: string) => meter && fraction(meter).n / fraction(meter).d < 0.75 ? '1/16' : '1/8';

/** The score with every `Z` bar written out as plain rests, timed by its own
 *  meter. abcjs times a `Z` by the header meter instead, which is wrong for a
 *  `Z` under a body M: line, so the check reads the source this way. */
function explicitRests(abc: string): string {
  let meter = '', length = '', voice = '', body = false;
  const meters = new Map<string, string>();
  return abc.split(/(\r?\n)/).map(line => {
    if (/^M:/.test(line)) { if (body) meters.set(voice, line.slice(2)); else meter = line.slice(2); return line; }
    if (/^L:/.test(line)) { length = line.slice(2); return line; }
    if (/^K:/.test(line)) { body = true; return line; }
    if (/^V:/.test(line)) { voice = line.slice(2).trim().split(/\s+/)[0]; return line; }
    if (!body || !line.includes('Z') || /^([A-Za-z]:|%)/.test(line)) return line;
    const units = barUnits(meters.get(voice) ?? meter, fraction(length || impliedLength(meter)));
    return line.replace(/(^|\|)((?:"[^"]*")*)Z(\d*)(?=\|)/g, (_, bar: string, prefix: string, count: string) =>
      bar + Array.from({ length: Number(count || 1) }, (_, i) => `${i ? '' : prefix}z${units}`).join('|'));
  }).join('');
}

/** Walks one bar's notes as abcjs reads them: an accidental carries to its own
 *  letter and octave until the bar line, and a tie continuation keeps the tied
 *  pitch and sets nothing. YuE2 carries by letter, so both are tracked. Given
 *  `before` (what the first half of a merged bar carries), a note whose pitch
 *  either carry would change gets its own accidental written in. */
function walkBar(text: string, state: NoteState, before?: Carry): { text: string; carry: Carry } {
  const own: Carry = { exact: new Map(), letter: new Map() };
  const merged: Carry = before ? { exact: new Map(before.exact), letter: new Map(before.letter) } : own;
  const out = text.replace(NOTE_TOKEN, (token, mark?: string, name?: string, octaves?: string) => {
    if (!name) {
      const inline = token.match(/^\[K:(.*)\]$/);
      if (inline) state.sig = keySignature(inline[1]);
      else if (token === '-') state.tie = state.last;
      else if (/[zxZ]/.test(token)) { state.tie = null; state.last = null; }
      return token;
    }
    const step = STEP_LETTERS.indexOf(name.toUpperCase());
    const index = 7 * ((name === name.toUpperCase() ? 4 : 5)
      + (octaves!.match(/'/g)?.length ?? 0) - (octaves!.match(/,/g)?.length ?? 0)) + step;
    const tied = state.tie === index;
    state.tie = null;
    state.last = index;
    if (tied) return token;
    if (mark !== undefined) {
      for (const carry of new Set([own, merged])) { carry.exact.set(index, MARK[mark]); carry.letter.set(step, MARK[mark]); }
      return token;
    }
    if (!before) return token;
    const alter = own.exact.get(index) ?? state.sig[step];
    if ((merged.exact.get(index) ?? state.sig[step]) === alter
      && (merged.letter.get(step) ?? state.sig[step]) === (own.letter.get(step) ?? state.sig[step])) return token;
    merged.exact.set(index, alter);
    merged.letter.set(step, alter);
    return (alter > 0 ? '^'.repeat(alter) : alter < 0 ? '_'.repeat(-alter) : '=') + token;
  });
  return { text: out, carry: merged };
}

/** Each note's pitch, start and length per track, and the whole length with
 *  trailing rests, as abcjs plays the text. */
function soundingNotes(abc: string): { tracks: number[][][]; total: number } {
  const audio = (abcjs.parseOnly(abc)[0] as any).setUpAudio();
  return { total: audio.totalDuration, tracks: audio.tracks.map((track: any[]) => track
    .filter(e => e.cmd === 'note' && e.startChar !== undefined).map(e => [e.pitch, e.start, e.duration])) };
}

export interface CoverRemeterResult {
  abc: string;
  /** 2/4 bars left in the result: a pickup, an odd bar before a section or meter change, or the last bar. */
  twoFourBars: number;
}

export function remeterTwoFourToFourFour(abc: string, pickup: boolean): CoverRemeterResult {
  const nl = abc.includes('\r\n') ? '\r\n' : '\n';
  const lines = abc.split(/\r?\n/);
  let end = lines.length;
  while (end > 0 && !lines[end - 1].trim()) end--;
  const keyLine = lines.findIndex(line => /^K:/.test(line));
  if (keyLine < 0) throw new Error('The score has no K: line.');
  const header = lines.slice(0, keyLine + 1);
  const meterLine = header.findIndex(line => /^M:/.test(line));
  const headerMeter = meterLine >= 0 ? fraction(header[meterLine].slice(2)) : null;
  // An implicit length would double every note once the header says 4/4.
  if (!header.some(line => /^L:/.test(line))) {
    header.splice(meterLine >= 0 ? meterLine + 1 : keyLine, 0, `L:${impliedLength(headerMeter?.text ?? '')}`);
  }
  const unit = fraction(header.find(line => /^L:/.test(line))!.slice(2));
  const sig = keySignature(header.at(-1)!.slice(2));

  // Segments run between section markers; each holds every voice's bars.
  type Segment = { marker: string | null; order: string[]; voices: Map<string, RemeterBar[]>; heads: Map<string, string> };
  const segment = (marker: string | null): Segment => ({ marker, order: [], voices: new Map(), heads: new Map() });
  const segments = [segment(null)];
  const meters = new Map<string, string>();
  const keys = new Map<string, string[]>();
  let voice = '', head = '';
  for (const line of lines.slice(keyLine + 1, end)) {
    const text = line.trim();
    if (!text) throw new Error('A blank line inside the score ends the tune.');
    if (text.startsWith('%') && !text.startsWith('%%') && text.slice(1).trim()) { segments.push(segment(line)); continue; }
    if (text.startsWith('%')) throw new Error('Comments and directives inside the score are not supported.');
    if (/^V:/.test(text)) { voice = text.slice(2).trim().split(/\s+/)[0]; head = line; continue; }
    if (/^M:/.test(text)) { meters.set(voice, fraction(text.slice(2)).text); continue; }
    if (/^K:/.test(text)) { keySignature(text.slice(2)); keys.set(voice, [...keys.get(voice) ?? [], text]); continue; }
    if (/^[A-Za-z]:/.test(text)) throw new Error(`A ${text.slice(0, 2)} line inside the score is not supported.`);
    const current = segments.at(-1)!;
    if (!current.voices.has(voice)) { current.order.push(voice); current.voices.set(voice, []); current.heads.set(voice, head); }
    const bars = remeterBars(line, meters.get(voice) ?? headerMeter?.text ?? '', unit);
    if (bars[0]) { bars[0].keys = keys.get(voice) ?? []; keys.delete(voice); }
    current.voices.get(voice)!.push(...bars);
  }
  if (keys.size) throw new Error('A K: line after the last bar is not supported.');
  const ids = new Set(segments.flatMap(s => s.order));
  if (ids.has('') && ids.size > 1) throw new Error('Music before the first V: line is not supported in a score with voices.');

  // One plan for every voice: [first bar, bar count] per output bar.
  let first = true, paired = 0, twoFourBars = 0;
  const plans = segments.map(s => {
    const [lead, ...rest] = s.order.map(id => s.voices.get(id)!);
    if (!lead) return [];
    if (rest.some(bars => bars.length !== lead.length || bars.some((bar, i) => bar.meter !== lead[i].meter))) {
      throw new Error(`The voices disagree on bar count or meter${s.marker ? ` in section "${s.marker.trim().slice(1).trim()}"` : ''}.`);
    }
    if (first && pickup && lead[0].meter !== '2/4') throw new Error('A pickup needs the score to start with a 2/4 bar.');
    const plan: Array<[number, number]> = [];
    let pending = -1;
    const flush = () => { if (pending >= 0) { plan.push([pending, 1]); twoFourBars++; pending = -1; } };
    lead.forEach((bar, i) => {
      if (bar.meter !== '2/4') { flush(); plan.push([i, 1]); }
      else if (pending >= 0) { plan.push([pending, 2]); paired++; pending = -1; }
      else if (first && pickup) { plan.push([i, 1]); twoFourBars++; }
      else pending = i;
      first = false;
    });
    flush();
    return plan;
  });
  if (!paired) throw new Error(twoFourBars ? 'No two 2/4 bars can be paired.' : 'The score has no 2/4 bars.');

  // The header says 4/4 where it said 2/4; body M: lines follow each output bar's meter.
  if (headerMeter?.text === '2/4') header[meterLine] = 'M:4/4';
  const startMeter = headerMeter?.text === '2/4' ? '4/4' : headerMeter?.text ?? '';
  const running = new Map<string, string>();
  const states = new Map<string, NoteState>();
  const rest = (text: string, meter: string) => text.replace(/Z$/, `z${barUnits(meter, unit)}`);
  const body: string[] = [];
  segments.forEach((s, n) => {
    if (s.marker !== null) body.push(s.marker);
    const out = s.order.map(id => {
      const bars = s.voices.get(id)!;
      if (!states.has(id)) states.set(id, { sig: [...sig], tie: null, last: null });
      const state = states.get(id)!;
      return plans[n].map(([from, count]) => {
        const a = bars[from], b = bars[from + 1];
        for (const key of a.keys) state.sig = keySignature(key.slice(2));
        if (count === 1) { walkBar(a.text, state); return { text: a.text, meter: a.meter, keys: a.keys }; }
        // A key change before the second half moves inside the bar.
        const left = walkBar(rest(a.text, '2/4'), state);
        const right = walkBar(b.keys.map(key => `[${key}]`).join('') + rest(b.text, '2/4'), state, left.carry);
        return { text: a.rest && b.rest ? 'Z' : left.text + right.text, meter: '4/4', keys: a.keys };
      });
    });
    // Lines of up to four bars, broken where the meter or key changes.
    const lead = out[0] ?? [];
    for (let i = 0; i < lead.length;) {
      let j = i + 1;
      while (j < lead.length && j - i < 4 && lead[j].meter === lead[i].meter && out.every(bars => !bars[j].keys.length)) j++;
      s.order.forEach((id, v) => {
        if (id) body.push(s.heads.get(id)!);
        if ((running.get(id) ?? startMeter) !== lead[i].meter) { body.push(`M:${lead[i].meter}`); running.set(id, lead[i].meter); }
        body.push(...out[v][i].keys);
        // abcjs times a `Z` by the header meter, or by an M: line straight after
        // K: in a score without voices, so other bars write their rests out.
        const tokens: string[] = [];
        for (const bar of out[v].slice(i, j)) {
          const run = bar.text === 'Z' ? tokens.at(-1)?.match(/^Z(\d*)$/) : null;
          if (bar.meter !== startMeter || (!id && pickup)) tokens.push(rest(bar.text, bar.meter));
          else if (run) tokens[tokens.length - 1] = `Z${Number(run[1] || 1) + 1}`;
          else tokens.push(bar.text);
        }
        body.push(tokens.map(t => `${t}|`).join(''));
      });
      i = j;
    }
  });
  const result = [...header, ...body, ...lines.slice(end)].join(nl);

  checkSameMusic(abc, result, 'Re-metering');
  return { abc: result, twoFourBars };
}

/** Throws unless `result` plays every note of `abc` at the same pitch, start
 *  and length. The source's own `Z` bars may be mis-timed; the result must
 *  play right as written. */
function checkSameMusic(abc: string, result: string, change: string) {
  const before = soundingNotes(explicitRests(abc)), after = soundingNotes(result);
  const same = (x: number[], y?: number[]) => !!y && x[0] === y[0] && Math.abs(x[1] - y[1]) < 1e-9 && Math.abs(x[2] - y[2]) < 1e-9;
  if (Math.abs(before.total - after.total) > 1e-9) {
    throw new Error(`${change} would change the length of the score, so the score was left as it was.`);
  }
  const [from, to] = [before.tracks, after.tracks];
  const track = from.length === to.length
    ? from.findIndex((t, i) => t.length !== to[i].length || t.some((note, j) => !same(note, to[i][j]))) : 0;
  if (track >= 0) {
    const j = from[track]?.findIndex((note, k) => !same(note, to[track]?.[k])) ?? -1;
    throw new Error(`${change} would change the music (track ${track + 1}, note ${j + 1}), so the score was left as it was.`);
  }
}

// ── Bar-line shift ──
//
// Moves every bar line `beats` beats later (earlier when negative) in every
// voice, for a transcription whose downbeats landed a beat off. The notes keep
// their time, so later turns the first beats into a pickup bar and earlier
// shortens bar 1; the last bar takes what is left. A note a new bar line cuts
// becomes tied notes, and accidentals are written wherever the new bars would
// carry them differently, for abcjs (by letter and octave) and YuE2 (by letter)
// alike. Section markers move with their bar lines; one at the very start
// stays. abcjs then checks every note keeps its pitch, start and length.

/** Ticks per L: unit: exact for lengths down to 1/64 and for thirds, fifths and sevenths. */
const TICKS = 20160;
const SHIFT_TOKEN = /"[^"]*"|![^!]*!|\[K:([^\]]*)\]|\[[A-Za-z]:[^\]]*\]|\[([^\]"]*)\]([\d/]*)|(\^\^|\^|__|_|=)?([A-Ga-g])([,']*)([\d/]*)|([zx])([\d/]*)|Z(\d*)|\(\d|\s+|./g;
const CHORD_NOTE = /(\^\^|\^|__|_|=)?([A-Ga-g])([,']*)/g;
const gcd = (a: number, b: number): number => b ? gcd(b, a % b) : a;

interface ShiftPitch { step: number; index: number; letter: string; octaves: string; alter: number }
/** One note, chord or rest; `prefix` holds its chord symbols, decorations, slur and [K:] fields. */
interface ShiftEvent { prefix: string; key: string | null; rest: string | null; pitches: ShiftPitch[]; ticks: number; tie: boolean; suffix: string }
interface ShiftVoice {
  events: ShiftEvent[]; ends: number[]; meters: string[]; time: number; head: string; meter: string;
  sig: number[]; carry: Map<number, number>; tied: Map<number, number>; prefix: string; key: string | null;
}

function shiftTicks(length: string): number {
  const m = /^(\d*)(\/*)(\d*)$/.exec(length);
  if (!m || (m[2].length > 1 && m[3])) throw new Error(`Unsupported note length "${length}".`);
  const ticks = Number(m[1] || 1) * TICKS / (m[2] ? Number(m[3] || 2 ** m[2].length) : 1);
  if (!Number.isInteger(ticks) || ticks <= 0) throw new Error(`Unsupported note length "${length}".`);
  return ticks;
}

function lengthText(ticks: number): string {
  const g = gcd(ticks, TICKS), n = ticks / g, d = TICKS / g;
  return d === 1 ? (n === 1 ? '' : String(n)) : `${n === 1 ? '' : n}/${d}`;
}

/** Reads one music line into the voice's events and bar ends, pitching each note as abcjs does. */
function readShiftLine(line: string, v: ShiftVoice, unit: { n: number; d: number }) {
  if (UNSUPPORTED_BAR.test(unquoted(line))) {
    throw new Error('Repeats, endings, double bars, voice overlays, comments and inline meter or length changes are not supported.');
  }
  const pitch = (mark: string | undefined, letter: string, octaves: string): ShiftPitch => {
    const step = STEP_LETTERS.indexOf(letter.toUpperCase());
    const index = 7 * ((letter === letter.toUpperCase() ? 4 : 5)
      + (octaves.match(/'/g)?.length ?? 0) - (octaves.match(/,/g)?.length ?? 0)) + step;
    // A tie continuation keeps the tied pitch and carries nothing.
    let alter = v.tied.get(index);
    if (alter === undefined) {
      alter = mark === undefined ? v.carry.get(index) ?? v.sig[step] : MARK[mark];
      if (mark !== undefined) v.carry.set(index, alter);
    }
    return { step, index, letter, octaves, alter };
  };
  const push = (rest: string | null, pitches: ShiftPitch[], ticks: number) => {
    v.events.push({ prefix: v.prefix, key: v.key, rest, pitches, ticks, tie: false, suffix: '' });
    v.time += ticks;
    v.prefix = ''; v.key = null;
    v.tied = new Map();
  };
  const barEnd = () => {
    if (v.time === (v.ends.at(-1) ?? 0)) throw new Error('An empty bar is not supported.');
    v.ends.push(v.time); v.meters.push(v.meter); v.carry = new Map();
  };
  for (const m of line.matchAll(SHIFT_TOKEN)) {
    const [token, key, chord, chordLength, mark, letter, octaves, length, rest, restLength, bars] = m;
    const last = v.events.at(-1);
    if (key !== undefined) { v.sig = keySignature(key); v.key = key; v.prefix += token; }
    else if (/^["!]|^[(.~]$/.test(token)) v.prefix += token;
    else if (chord !== undefined) {
      const notes = [...chord.matchAll(CHORD_NOTE)];
      if (!notes.length || notes.map(n => n[0]).join('') !== chord) throw new Error(`Unsupported chord "${token}".`);
      push(null, notes.map(n => pitch(n[1], n[2], n[3])), shiftTicks(chordLength));
    } else if (letter) push(null, [pitch(mark, letter, octaves)], shiftTicks(length));
    else if (rest) push(rest, [], shiftTicks(restLength));
    else if (bars !== undefined) {
      if (v.time !== (v.ends.at(-1) ?? 0)) throw new Error('A Z rest must fill its bars.');
      const count = Math.max(1, Number(bars || 1));
      for (let i = 0; i < count; i++) { if (i) barEnd(); push('z', [], barUnits(v.meter, unit) * TICKS); }
    } else if (token === '-') {
      if (!last?.pitches.length) throw new Error('A tie must follow a note.');
      last.tie = true;
      v.tied = new Map(last.pitches.map(p => [p.index, p.alter]));
    } else if (token === ')') { if (last) last.suffix += token; }
    else if (token === '|') barEnd();
    else if (!/^\s+$/.test(token)) throw new Error(`"${token}" is not supported (nor are tuplets, grace notes or broken rhythms).`);
  }
  if (!/\|\s*$/.test(line)) throw new Error('Every music line must end with a bar line.');
}

export function shiftBarLines(abc: string, beats: number): { abc: string } {
  const nl = abc.includes('\r\n') ? '\r\n' : '\n';
  const lines = abc.split(/\r?\n/);
  let end = lines.length;
  while (end > 0 && !lines[end - 1].trim()) end--;
  const keyLine = lines.findIndex(line => /^K:/.test(line));
  if (keyLine < 0) throw new Error('The score has no K: line.');
  const header = lines.slice(0, keyLine + 1);
  const meterField = header.find(line => /^M:/.test(line));
  if (!meterField) throw new Error('The score has no M: line.');
  const startMeter = fraction(meterField.slice(2)).text;
  const unit = fraction(header.find(line => /^L:/.test(line))?.slice(2) ?? impliedLength(startMeter));
  const beatTicks = (meter: string) => TICKS * unit.d / (fraction(meter).d * unit.n);
  if (!Number.isInteger(beats) || !beats || Math.abs(beats) >= fraction(startMeter).n) {
    throw new Error(`Shift by a whole number of beats, less than a ${startMeter} bar.`);
  }
  const shift = beats * beatTicks(startMeter);
  const headerSig = keySignature(header.at(-1)!.slice(2));

  const voices = new Map<string, ShiftVoice>();
  const markers: Array<{ line: string; time: number }> = [];
  let voice: ShiftVoice | null = null;
  const open = (id: string, head: string) => {
    if (!voices.has(id)) voices.set(id, { events: [], ends: [], meters: [], time: 0, head, meter: startMeter,
      sig: [...headerSig], carry: new Map(), tied: new Map(), prefix: '', key: null });
    return voices.get(id)!;
  };
  for (const line of lines.slice(keyLine + 1, end)) {
    const text = line.trim();
    if (!text) throw new Error('A blank line inside the score ends the tune.');
    if (text.startsWith('%') && !text.startsWith('%%') && text.slice(1).trim()) {
      const times = new Set([...voices.values()].map(v => v.time));
      if (times.size > 1) throw new Error(`The voices disagree on where section "${text.slice(1).trim()}" starts.`);
      markers.push({ line, time: [...times][0] ?? 0 });
      continue;
    }
    if (text.startsWith('%')) throw new Error('Comments and directives inside the score are not supported.');
    if (/^V:/.test(text)) { voice = open(text.slice(2).trim().split(/\s+/)[0], line); continue; }
    voice ??= open('', '');
    if (/^M:/.test(text)) { voice.meter = fraction(text.slice(2)).text; continue; }
    if (/^K:/.test(text)) { voice.sig = keySignature(text.slice(2)); voice.key = text.slice(2); voice.prefix += `[${text}]`; continue; }
    if (/^[A-Za-z]:/.test(text)) throw new Error(`A ${text.slice(0, 2)} line inside the score is not supported.`);
    readShiftLine(line, voice, unit);
  }
  if (voices.has('') && voices.size > 1) throw new Error('Music before the first V: line is not supported in a score with voices.');
  const all = [...voices.entries()];
  const lead = all[0]?.[1];
  if (!lead?.ends.length) throw new Error('The score has no bars.');
  if (all.some(([, v]) => v.prefix || v.time !== lead.time || v.ends.join() !== lead.ends.join())) {
    throw new Error('The voices disagree on bar lengths.');
  }

  // Every old bar line moved by the shift, inside the score.
  const total = lead.time;
  const ends = [...new Set([0, ...lead.ends].map(t => t + shift).filter(t => t > 0 && t < total))].sort((a, b) => a - b);
  ends.push(total);
  // A new bar keeps the meter of the old bar under its middle when it is that long, or is written as so many beats.
  const meters = ends.map((stop, i) => {
    const start = i ? ends[i - 1] : 0;
    const meter = lead.meters[lead.ends.findIndex(t => t > (start + stop) / 2)];
    const length = stop - start, beat = beatTicks(meter);
    if (length === barUnits(meter, unit) * TICKS) return meter;
    if (length % beat) throw new Error('The bars are not whole beats long.');
    return `${length / beat}/${fraction(meter).d}`;
  });
  // The new bar each section starts on.
  const starts = markers.map(marker => {
    const time = marker.time && marker.time + shift;
    const bar = time ? ends.indexOf(time) + 1 : 0;
    if (time >= total || (time && !bar)) throw new Error(`Section "${marker.line.trim().slice(1).trim()}" would move off the score.`);
    return bar;
  });

  // Each voice's events cut at the new bar lines.
  const markOf = (alter: number) => alter > 0 ? '^'.repeat(alter) : alter < 0 ? '_'.repeat(-alter) : '=';
  const out = all.map(([id, v]) => {
    const bars: Array<{ text: string; rest: boolean }> = [];
    let text = '', rest = true, t = 0, bar = 0, sig = [...headerSig], tied = new Set<number>();
    let exact = new Map<number, number>(), letters = new Map<number, number>();
    for (const e of v.events) {
      let left = e.ticks, first = true;
      while (left > 0) {
        const take = Math.min(left, ends[bar] - t), lastPart = take === left;
        if (first && e.key !== null) sig = keySignature(e.key);
        const tie = lastPart ? e.tie : !e.rest;
        const notes = e.pitches.map(p => {
          const held = tied.has(p.index);
          const abcjsAlter = held ? p.alter : exact.get(p.index) ?? sig[p.step];
          const yue2Alter = letters.get(p.step) ?? sig[p.step];
          if (!held) exact.set(p.index, p.alter);
          letters.set(p.step, p.alter);
          return (abcjsAlter === p.alter && yue2Alter === p.alter ? '' : markOf(p.alter)) + p.letter + p.octaves;
        });
        const prefix = first ? e.prefix : '', suffix = lastPart ? e.suffix : '';
        text += prefix + (e.rest ?? (notes.length > 1 ? `[${notes.join('')}]` : notes[0])) + lengthText(take) + (tie ? '-' : '') + suffix;
        rest &&= !!e.rest && !prefix && !suffix;
        tied = tie ? new Set(e.pitches.map(p => p.index)) : new Set();
        first = false; t += take; left -= take;
        if (t === ends[bar]) {
          bars.push({ text, rest });
          text = ''; rest = true; bar++;
          exact = new Map(); letters = new Map();
        }
      }
    }
    return { id, head: v.head, bars };
  });

  // Lines of up to four bars, broken at a section or a meter change.
  const body: string[] = [];
  const running = new Map<string, string>();
  for (let i = 0; i < ends.length;) {
    starts.forEach((bar, k) => { if (bar === i) body.push(markers[k].line); });
    let j = i + 1;
    while (j < ends.length && j - i < 4 && meters[j] === meters[i] && !starts.includes(j)) j++;
    for (const v of out) {
      if (v.id) body.push(v.head);
      if ((running.get(v.id) ?? startMeter) !== meters[i]) { body.push(`M:${meters[i]}`); running.set(v.id, meters[i]); }
      // abcjs times a `Z` by the header meter, so only those bars, in a score with voices, use one.
      const tokens: string[] = [];
      for (const bar of v.bars.slice(i, j)) {
        const run = tokens.at(-1)?.match(/^Z(\d*)$/);
        if (!bar.rest || !v.id || meters[i] !== startMeter) tokens.push(bar.text);
        else if (run) tokens[tokens.length - 1] = `Z${Number(run[1] || 1) + 1}`;
        else tokens.push('Z');
      }
      body.push(tokens.map(token => `${token}|`).join(''));
    }
    i = j;
  }
  const result = [...header, ...body, ...lines.slice(end)].join(nl);
  checkSameMusic(abc, result, 'Shifting the bar lines');
  return { abc: result };
}
