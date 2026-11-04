// Instrumental YuE2 renders, after upstream's yue2-music instrumental skill
// (multimodal-art-projection/YuE 72272f9, skills/yue2-music/instrumental):
// every Vocal note moves to Ins, Vocal keeps only rests and chord symbols, and
// the style says plainly that nothing is sung. Empty lyrics alone leave the
// melody on the Vocal voice, and YuE2 sings it as wordless oohs.
//
// Two modes. 'overlap' is the default above. 'resting' follows the ComfyUI node
// nappaniconico/yue2_instrumental_node: a sung bar moves to Ins only when the
// Ins bar is wholly resting, an Ins bar with any note is kept as it is, and
// Vocal is muted either way.

const LENGTHS = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1];
const TOKEN = /"[^"]*"|\[[^\]]*\]|[_^=]*[A-Ga-g][,']*\d*(?:\/\d*)?-?|[zx]\d*(?:\/\d*)?|Z\d*|\S/g;
const NOTE = /^[_^=]*[A-Ga-g]/;
const NATURAL: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const ALTER: Record<string, number> = { '': 0, '=': 0, '^': 1, '^^': 2, _: -1, __: -2 };
// Sharps (+) or flats (-) per key, as upstream's abc_tools.KEYS.
const MAJOR = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#'];
const MINOR = ['Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#'];

function length(token: string): number {
  const m = token.match(/(\d*)(?:\/(\d*))?-?$/)!;
  const num = m[1] ? Number(m[1]) : 1;
  return m[0].includes('/') ? num / (m[2] ? Number(m[2]) : 2) : num;
}

function split(ticks: number): number[] {
  const out: number[] = [];
  while (ticks > 0) {
    const n = LENGTHS.find(l => l <= ticks) ?? ticks;
    out.push(n);
    ticks -= n;
  }
  return out;
}

function rests(ticks: number): string {
  return split(ticks).map(n => (n === 1 ? 'z' : `z${n}`)).join('');
}

/** A key's signature by letter; an unrecognised key reads as C. */
function signature(key: string): Record<string, number> {
  const m = key.trim().match(/^([A-G][b#]?)(m(?:in)?)?(?![a-z])/);
  const count = m ? (m[2] ? MINOR : MAJOR).indexOf(m[1]) - 7 : 0;
  const sig: Record<string, number> = { C: 0, D: 0, E: 0, F: 0, G: 0, A: 0, B: 0 };
  if (count >= -7) for (const l of (count > 0 ? 'FCGDAEB' : 'BEADGCF').slice(0, Math.abs(count))) sig[l] = Math.sign(count);
  return sig;
}

/** upstream compile_score.spelling: the plainest name for a MIDI pitch, with an
 *  accidental only where the bar's running state needs one. */
function spell(pitch: number, key: string, active: Record<string, number>, tied = false): string {
  const sig = signature(key);
  const sharp = Object.values(sig).reduce((a, b) => a + b, 0) >= 0;
  let best: { score: number; letter: string; alt: number; octave: number } | null = null;
  for (const letter of 'ABCDEFG') {
    for (const alt of [-1, 0, 1]) {
      const base = pitch - alt - 60 - NATURAL[letter];
      if (((base % 12) + 12) % 12) continue;
      const score = 4 * Number(alt !== sig[letter]) + 2 * Math.abs(alt) + Number(sharp ? alt < 0 : alt > 0);
      if (!best || score < best.score) best = { score, letter, alt, octave: base / 12 };
    }
  }
  const { letter, alt, octave } = best!;
  // YuE2 carries an accidental by letter, abcjs by letter and octave (the
  // MusicXML converter's spell() does the same): mark the note unless both agree.
  // abcjs also ignores the mark on a tie target, so it sets no state there.
  const exact = letter + octave;
  let acc = '';
  if (alt !== (active[letter] ?? sig[letter]) || alt !== (active[exact] ?? sig[letter])) {
    acc = alt < 0 ? '_' : alt > 0 ? '^' : '=';
    active[letter] = alt;
    if (!tied) active[exact] = alt;
  }
  return acc + (octave < 0 ? letter + ','.repeat(-octave) : octave === 0 ? letter : letter.toLowerCase() + "'".repeat(octave - 1));
}

interface Voice { key: string; pending: { pitch: number; written: number } | null }
interface Event { start: number; end: number; pitch: number; tie: boolean; continued: boolean }

/** One bar's sounding notes as MIDI pitches (upstream abc_tools.parse_bar:
 *  accidentals carry by letter to the bar's end; an unmarked tie continuation
 *  keeps the tied pitch). Updates the voice's key and tie state. */
function parseBar(bar: string, voice: Voice): { events: Event[]; ticks: number; fields: string[] } {
  const events: Event[] = [], fields: string[] = [];
  const local: Record<string, number> = {};
  let at = 0;
  for (const t of bar.match(TOKEN) ?? []) {
    if (t.startsWith('[')) {
      fields.push(t);
      const key = t.match(/^\[K:\s*([^\]]*)\]$/);
      if (key) voice.key = key[1];
      continue;
    }
    const note = t.match(/^([_^=]*)([A-Ga-g])([,']*)(\d*(?:\/\d*)?)(-?)$/);
    if (note) {
      const [, acc, name, octave, , tie] = note;
      const letter = name.toUpperCase();
      const written = 60 + NATURAL[letter] + (name === letter ? 0 : 12) + 12 * ((octave.match(/'/g)?.length ?? 0) - (octave.match(/,/g)?.length ?? 0));
      if (acc) local[letter] = ALTER[acc] ?? 0;
      let pitch = written + (local[letter] ?? signature(voice.key)[letter]);
      const continued = voice.pending !== null;
      if (voice.pending && !acc && written === voice.pending.written) pitch = voice.pending.pitch;
      const len = length(t);
      events.push({ start: at, end: at + len, pitch, tie: tie === '-', continued });
      voice.pending = tie ? { pitch, written } : null;
      at += len;
    } else if (/^[zx]/.test(t)) {
      at += length(t);
      voice.pending = null;
    } else if (/^Z/.test(t)) voice.pending = null;
  }
  return { events, ticks: at, fields };
}

/** Ins pieces left after cutting out every interval a Vocal note sounds
 *  (upstream instrumentalize.subtract_intervals). */
function subtract(e: Event, vocal: Event[]): Event[] {
  let pieces: Event[] = [e];
  for (const v of vocal) {
    pieces = pieces.flatMap(p => {
      if (v.end <= p.start || v.start >= p.end) return [p];
      const keep: Event[] = [];
      if (p.start < v.start) keep.push({ ...p, end: v.start, tie: false });
      if (v.end < p.end) keep.push({ ...p, start: v.end, continued: false });
      return keep;
    });
  }
  return pieces;
}

/** Notes -> one bar of ABC, as upstream compile_score writes a bar. */
function writeBar(fields: string[], events: Event[], ticks: number, key: string): string {
  const active: Record<string, number> = {};
  let out = fields.join(''), cursor = 0;
  for (const e of [...events].sort((a, b) => a.start - b.start)) {
    out += rests(e.start - cursor);
    const parts = split(e.end - e.start);
    parts.forEach((n, i) => {
      out += spell(e.pitch, key, active, i > 0 || (e.continued && e.start === 0)) + (n === 1 ? '' : n) + (i < parts.length - 1 || e.tie ? '-' : '');
    });
    cursor = e.end;
  }
  return out + rests(ticks - cursor);
}

/** One line's bars, with Z2-Z4 expanded to single whole-bar rests. */
function bars(line: string): string[] {
  return line.replace(/\|\s*$/, '').split('|').flatMap(bar => {
    const multi = bar.trim().match(/^Z(\d+)$/);
    return multi ? Array<string>(Number(multi[1])).fill('Z') : [bar.trim()];
  });
}

/** Runs of bare whole-bar rests fold back into Z2-Z4 within the group. */
function line(barList: string[]): string {
  const out: string[] = [];
  for (const bar of barList) {
    const last = out[out.length - 1];
    const run = last?.match(/^Z(\d*)$/);
    if (bar === 'Z' && run && Number(run[1] || 1) < 4) out[out.length - 1] = `Z${Number(run[1] || 1) + 1}`;
    else out.push(bar);
  }
  return out.join('|') + '|';
}

/** What Vocal keeps of a sung bar: inline fields and chord symbols on rests,
 *  or Z when it has no chords. Null when nothing sounds. */
function vocalRemainder(bar: string): string | null {
  const tokens = bar.match(TOKEN) ?? [];
  if (!tokens.some(t => NOTE.test(t))) return null;
  let at = 0, kept = '', last = 0;
  const fields = tokens.filter(t => t.startsWith('['));
  for (const t of tokens) {
    if (t.startsWith('"')) { kept += rests(at - last) + t; last = at; }
    else if (NOTE.test(t) || /^[zx]/.test(t)) at += length(t);
  }
  return kept ? fields.join('') + kept + rests(at - last) : 'Z';
}

export type InstrumentalMode = 'overlap' | 'resting';

/** `keptInsBars` counts sung bars whose melody was dropped because Ins already
 *  had notes ('resting' mode only). */
export interface InstrumentalScore { abc: string; movedBars: number; replacedInsBars: number; keptInsBars: number }

interface Group { vocal: string[]; vocalFields: string[]; ins: string[]; insFields: string[] }

/** Moves the Vocal melody of a native YuE2 score onto Ins. 'overlap' (default):
 *  Vocal notes win wherever they sound; Ins keeps whatever of its own notes
 *  they don't cover (upstream's default --overlap vocal). `replacedInsBars`
 *  counts bars where Ins notes were cut. 'resting': only wholly resting Ins
 *  bars take the melody. Output keeps the native layout: Vocal/Ins line pairs
 *  of at most four bars. */
export function instrumentalScore(abc: string, mode: InstrumentalMode = 'overlap'): InstrumentalScore {
  const nl = abc.includes('\r\n') ? '\r\n' : '\n';
  const lines = abc.split(/\r?\n/);
  const out: (string | Group)[] = [];
  let header = true, voice = '';
  let group: Group | null = null;
  let movedBars = 0, replacedInsBars = 0, keptInsBars = 0;
  const keyLine = lines.findIndex(l => /^K:/.test(l));
  const hasInsDeclaration = lines.some((l, i) => /^V:\s*Ins\b/i.test(l) && i < keyLine);
  const vocalVoice: Voice = { key: '', pending: null }, insVoice: Voice = { key: '', pending: null };
  // The last Ins bar written and whether it ends on a tied Ins note, so the
  // tie can be dropped when Vocal cuts the note it ties into.
  let lastIns: { bars: string[]; i: number; insTie: boolean } | null = null;
  let insTieBroken = false;
  // 'resting': the last moved Ins bar when it ends on a tie, which is dropped
  // unless the next bar moves too.
  let movedTie: { bars: string[]; i: number } | null = null;
  const flush = () => {
    if (!group) return;
    const g = group;
    group = null;
    for (const f of [...g.vocalFields, ...g.insFields]) {
      const k = f.trim().match(/^K:\s*(.*)$/);
      if (k) vocalVoice.key = insVoice.key = k[1];
    }
    const count = Math.max(g.vocal.length, g.ins.length);
    while (g.vocal.length < count) g.vocal.push('Z');
    while (g.ins.length < count) g.ins.push('Z');
    g.vocal.forEach((bar, i) => {
      const v = parseBar(bar, vocalVoice), ins = parseBar(g.ins[i], insVoice);
      const ticks = Math.max(v.ticks, ins.ticks);
      const kept = vocalRemainder(bar);
      if (mode === 'resting') {
        if (kept !== null) g.vocal[i] = kept;
        if (kept === null || ins.events.length) {
          if (kept !== null) keptInsBars++;
          if (movedTie) movedTie.bars[movedTie.i] = movedTie.bars[movedTie.i].replace(/-$/, '');
          movedTie = null;
          return;
        }
        movedBars++;
        // A tie into this bar from an unmoved one has nothing to tie to.
        if (!movedTie) v.events[0].continued = false;
        const last = v.events[v.events.length - 1];
        g.ins[i] = writeBar(ins.fields.length ? ins.fields : v.fields, v.events, ticks, ins.fields.length ? insVoice.key : vocalVoice.key);
        movedTie = last.tie && last.end === ticks ? { bars: g.ins, i } : null;
        return;
      }
      if (kept !== null) {
        movedBars++;
        g.vocal[i] = kept;
      } else if (!insTieBroken) {
        const last = ins.events[ins.events.length - 1];
        lastIns = { bars: g.ins, i, insTie: !!last?.tie && last.end === ticks };
        return;
      }
      // The tie into this bar was cut, so its first note now starts fresh.
      if (insTieBroken && ins.events[0]) ins.events[0].continued = false;
      const pieces = ins.events.flatMap(e => subtract(e, v.events));
      if (pieces.length !== ins.events.length
        || pieces.some(p => !ins.events.some(e => e.start === p.start && e.end === p.end))) replacedInsBars++;
      if (ins.events[0]?.continued && !pieces.some(p => p.continued) && lastIns?.insTie) {
        const prev = lastIns as { bars: string[]; i: number };
        prev.bars[prev.i] = prev.bars[prev.i].replace(/-$/, '');
      }
      const last = ins.events[ins.events.length - 1];
      insTieBroken = !!last?.tie && !pieces.some(p => p.end === last.end && p.tie);
      const events = [...v.events, ...pieces];
      const fields = ins.fields.length ? ins.fields : v.fields;
      g.ins[i] = events.length ? writeBar(fields, events, ticks, ins.fields.length ? insVoice.key : vocalVoice.key) : 'Z';
      lastIns = { bars: g.ins, i, insTie: pieces.some(p => p.tie && p.end === ticks) };
    });
    out.push(g);
  };
  for (const raw of lines) {
    if (header) {
      out.push(raw);
      if (!hasInsDeclaration && /^V:\s*Vocal\b/i.test(raw)) out.push('V: Ins clef=treble name="Ins Melody" snm="Inst."');
      const k = raw.match(/^K:\s*(.*)$/);
      if (k) { vocalVoice.key = insVoice.key = k[1]; header = false; }
      continue;
    }
    const text = raw.trim();
    if (/^V:/.test(text)) {
      voice = /^V:\s*Vocal\b/i.test(text) ? 'vocal' : /^V:\s*Ins\b/i.test(text) ? 'ins' : 'other';
      if (voice === 'vocal' && group?.vocal.length) flush();
      if (voice === 'other') { flush(); out.push(raw); }
      continue;
    }
    if (voice === 'vocal') {
      if (/^[A-Za-z]:/.test(text)) { (group ??= { vocal: [], vocalFields: [], ins: [], insFields: [] }).vocalFields.push(raw); continue; }
      if (text && !text.startsWith('%')) { (group ??= { vocal: [], vocalFields: [], ins: [], insFields: [] }).vocal.push(...bars(text)); continue; }
    }
    if (voice === 'ins' && group) {
      if (/^[A-Za-z]:/.test(text)) { group.insFields.push(raw); continue; }
      if (text && !text.startsWith('%')) { group.ins.push(...bars(text)); continue; }
    }
    // Blank lines, section comments, other voices, or Ins with no Vocal before it.
    if (text && !text.startsWith('%') && !/^[A-Za-z]:/.test(text) && voice === 'ins') { out.push('V: Ins', raw); continue; }
    flush();
    out.push(raw);
  }
  flush();
  // Groups render last: a later bar may still drop a tie in an earlier group.
  const text = out.flatMap(o => {
    if (typeof o === 'string') return [o];
    const pairs: string[] = [];
    for (let i = 0; i < o.vocal.length; i += 4) {
      pairs.push('V: Vocal', ...(i ? [] : o.vocalFields), line(o.vocal.slice(i, i + 4)),
        'V: Ins', ...(i ? [] : o.insFields), line(o.ins.slice(i, i + 4)));
    }
    return pairs;
  });
  return { abc: text.join(nl), movedBars, replacedInsBars, keptInsBars };
}

/** Upstream's instrumental style wording: lead with "Instrumental", and say
 *  no vocals, singing, choir or speech. Idempotent. An empty style stays
 *  empty, so the missing-caption check still fires: the wording alone gives
 *  YuE2 no genre or instrument, and it answers with a speaking voice. */
export function instrumentalStyle(style: string): string {
  let s = style.trim().replace(/[.,]+$/, '');
  if (!s) return '';
  if (!/^instrumental\b/i.test(s)) s = s ? `Instrumental, ${s}` : 'Instrumental';
  for (const condition of ['no vocals', 'no singing', 'no choir', 'no spoken words']) {
    if (!s.toLowerCase().includes(condition)) s += `, ${condition}`;
  }
  return `${s}.`;
}
