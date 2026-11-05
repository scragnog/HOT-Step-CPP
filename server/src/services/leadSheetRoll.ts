import abcjs from 'abcjs';

export interface LeadSheetNote {
  onset: number; offset: number; pitch: number;
  /** Original PPQ coordinates avoid float drift at exact half-sixteenths. */
  midiTicks?: { onset: number; offset: number; ppq: number };
}
export interface LeadSheetEvents {
  bpm: number;
  melody: LeadSheetNote[];
  chords: LeadSheetNote[];
  drums?: LeadSheetNote[];
  /** MIDI interchange already carries sixteenth positions; raw chord labels use whole-beat snapping. */
  chordTiming?: 'seconds' | 'midi';
}
export interface LeadSheetRoll {
  bpm: number; length: number;
  pianoroll: Float32Array; chord: Float32Array;
  drum_pianoroll: Float32Array; context_mask: Float32Array;
}
export const MAX_LEAD_SHEET_STEPS = 5000;
const halfUp = (n: number) => Math.floor(n + 0.5);
function nearestEven(n: number): number {
  const floor = Math.floor(n), fraction = n - floor;
  return fraction === 0.5 ? floor + floor % 2 : halfUp(n);
}

/** Quantise raw seconds once, on the source's constant quarter-note BPM grid. */
export function leadSheetRoll(events: LeadSheetEvents): LeadSheetRoll {
  const { bpm } = events;
  if (!Number.isFinite(bpm) || bpm <= 0) throw new Error('Lead sheet BPM must be positive');
  const quantise = (notes: LeadSheetNote[], chord = false) => notes.flatMap(note => {
    if (!Number.isFinite(note.onset) || !Number.isFinite(note.offset) || note.onset < 0 || note.offset < note.onset
      || !Number.isInteger(note.pitch) || note.pitch < 0 || note.pitch > 127) throw new Error('Invalid lead sheet note');
    const ticks = note.midiTicks;
    if (ticks && (events.chordTiming !== 'midi' || !Number.isInteger(ticks.ppq) || ticks.ppq <= 0
      || !Number.isSafeInteger(ticks.onset) || !Number.isSafeInteger(ticks.offset)
      || ticks.onset < 0 || ticks.offset < ticks.onset || !Number.isSafeInteger(ticks.offset * 8 + ticks.ppq))) {
      throw new Error('Invalid lead sheet MIDI coordinates');
    }
    const snap = (seconds: number, tick?: number) => ticks && tick !== undefined
      ? Math.floor((tick * 8 + ticks.ppq) / (2 * ticks.ppq))
      : chord && events.chordTiming !== 'midi' ? nearestEven(seconds * bpm / 60) * 4 : halfUp(seconds * bpm / 15);
    const onset = snap(note.onset, ticks?.onset), end = snap(note.offset, ticks?.offset);
    if (chord && end === onset) return [];
    const duration = chord ? end - onset : Math.max(1, end - onset);
    if (!Number.isSafeInteger(onset) || onset + duration > MAX_LEAD_SHEET_STEPS) {
      throw new Error(`Lead sheet exceeds ${MAX_LEAD_SHEET_STEPS} sixteenth notes; trim the score`);
    }
    return [{ onset, duration, pitch: chord ? note.pitch % 12 : note.pitch }];
  });
  const melody = quantise(events.melody), chords = quantise(events.chords, true), drums = quantise(events.drums ?? []);
  if (!melody.length) throw new Error('Lead sheet melody must contain at least one note');
  const length = [...melody, ...chords, ...drums].reduce((end, n) => Math.max(end, n.onset + n.duration), 0);
  const roll: LeadSheetRoll = { bpm, length, pianoroll: new Float32Array(length * 256),
    chord: new Float32Array(length * 12), drum_pianoroll: new Float32Array(length * 128), context_mask: new Float32Array(length) };
  for (const n of melody) {
    roll.pianoroll[n.onset * 256 + n.pitch * 2] = 1;
    for (let t = n.onset + 1; t < n.onset + n.duration; t++) roll.pianoroll[t * 256 + n.pitch * 2 + 1] = 1;
  }
  for (const [notes, data, width] of [[chords, roll.chord, 12], [drums, roll.drum_pianoroll, 128]] as const) {
    for (const n of notes) for (let t = n.onset; t < n.onset + n.duration; t++) data[t * width + n.pitch] = 1;
  }
  return roll;
}

const roots: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const intervals: Record<string, number[]> = { '': [0, 4, 7], maj: [0, 4, 7], m: [0, 3, 7], min: [0, 3, 7],
  dim: [0, 3, 6], aug: [0, 4, 8], '7': [0, 4, 7, 10], maj7: [0, 4, 7, 11], m7: [0, 3, 7, 10],
  min7: [0, 3, 7, 10], sus2: [0, 2, 7], sus4: [0, 5, 7], '5': [0, 7] };
function chordPitches(name: string): number[] {
  if (/^(N\.?C\.?|none)$/i.test(name)) return [];
  const normalized = name.replace(/♭/g, 'b').replace(/♯/g, '#').replace(':', '');
  const match = /^([A-G])([#b]?)([^/]*)(?:\/[A-G][#b]?)?$/.exec(normalized);
  if (!match || !intervals[match[3]]) throw new Error(`Unsupported lead sheet chord: ${name}`);
  const root = roots[match[1]] + (match[2] === '#' ? 1 : match[2] === 'b' ? -1 : 0);
  return intervals[match[3]].map(pitch => (root + pitch + 12) % 12);
}

/** abcjs resolves ties, tuplets, accidentals, pickups, meters and tempo changes.
 * Its audio positions are whole notes scaled to the initial tempo, not MIDI ticks.
 * Chord annotations are held to the next annotation, rather than using its comping track. */
export function abcLeadSheetEvents(abc: string): LeadSheetEvents {
  if (!/^Q:/m.test(abc)) throw new Error('Lead sheet needs an explicit Q: tempo');
  // abcjs silently substitutes playback defaults for zero or malformed Q:.
  // Reject those before parsing instead of accidentally changing the source BPM.
  const tempos = [...abc.matchAll(/^Q:([^\r\n]*)/gm), ...abc.matchAll(/\[Q:([^\]]*)\]/g)];
  for (const [, value] of tempos) {
    const match = /^(?:(\d+)\s*\/\s*(\d+)\s*=\s*)?(\d+(?:\.\d+)?)$/.exec(value.trim());
    if (!match || Number(match[3]) <= 0 || match[1] !== undefined && (Number(match[1]) <= 0 || Number(match[2]) <= 0)) {
      throw new Error('Lead sheet needs a readable positive Q: tempo');
    }
  }
  const tune = abcjs.parseOnly(abc)[0];
  if (!tune) throw new Error('Cannot parse lead sheet ABC');
  const audio = (tune as any).setUpAudio();
  const bpm = Number(audio.tempo), secondsPerWhole = 240 / bpm;
  if (!Number.isFinite(bpm) || bpm <= 0) throw new Error('Lead sheet needs a readable tempo');
  const voices = [...abc.matchAll(/^V:\s*(\S+)/gm)].map(match => match[1]);
  const ids = [...new Set(voices)];
  const vocal = ids.findIndex(id => /^(vocal|vocals)$/i.test(id));
  const selected = vocal >= 0 ? vocal : 0;
  const melody = (audio.tracks[selected] ?? []).filter((n: any) => n.cmd === 'note').map((n: any) => ({
    onset: n.start * secondsPerWhole, offset: (n.start + n.duration) * secondsPerWhole, pitch: n.pitch,
  }));
  const annotations: Array<{ onset: number; pitches: number[] }> = [];
  // Read only one voice's chord symbols: duplicate symbols on an accompaniment
  // staff must not replace or extend the lead's harmony.
  const byVoice = new Map<number, typeof annotations>();
  for (const line of tune.lines) {
    let track = 0;
    for (const staff of line.staff ?? []) for (const voice of staff.voices ?? []) {
      const list = byVoice.get(track) ?? [];
      byVoice.set(track++, list);
      for (const event of voice as any[]) {
        const chord = event.chord?.find((c: any) => c.position === 'default');
        if (chord) list.push({ onset: event.currentTrackMilliseconds / 1000, pitches: chordPitches(chord.name) });
      }
    }
  }
  annotations.push(...(byVoice.get(selected)?.length ? byVoice.get(selected)! : [...byVoice.values()].find(v => v.length) ?? []));
  annotations.sort((a, b) => a.onset - b.onset);
  const end = audio.totalDuration * secondsPerWhole;
  const chords = annotations.flatMap((item, index) => item.pitches.map(pitch => ({
    pitch, onset: item.onset, offset: annotations[index + 1]?.onset ?? end,
  })));
  return { bpm, melody, chords, drums: [], chordTiming: 'seconds' };
}
export const abcLeadSheetRoll = (abc: string): LeadSheetRoll => leadSheetRoll(abcLeadSheetEvents(abc));

/** Lossless SMF note events for the upstream interchange path. Tempo is metadata:
 * ticks map to raw seconds at the supplied constant BPM, as upstream reads beat space.
 * Unlike the preview parser, no millisecond rounding or minimum duration is applied. */
export function midiLeadSheetNotes(buffer: Buffer, bpm: number, includeDrums = false): LeadSheetNote[] {
  if (!Number.isFinite(bpm) || bpm <= 0) throw new Error('MIDI BPM must be positive');
  let pos = 0;
  const need = (n: number) => { if (pos + n > buffer.length) throw new Error('Truncated lead sheet MIDI'); };
  const u8 = () => { need(1); return buffer[pos++]; };
  const u16 = () => { need(2); const n = buffer.readUInt16BE(pos); pos += 2; return n; };
  const u32 = () => { need(4); const n = buffer.readUInt32BE(pos); pos += 4; return n; };
  const text = () => { need(4); const s = buffer.toString('ascii', pos, pos + 4); pos += 4; return s; };
  const variable = () => { let n = 0; for (let i = 0; i < 4; i++) { const b = u8(); n = n * 128 + (b & 127); if (b < 128) return n; } throw new Error('Invalid MIDI variable length'); };
  if (text() !== 'MThd') throw new Error('Missing MIDI header');
  const header = u32(), format = u16(), tracks = u16(), ppq = u16();
  if (header < 6 || format > 1 || ppq === 0 || ppq & 0x8000) throw new Error('Lead sheet MIDI requires format 0/1 and positive PPQ');
  pos += header - 6;
  const notes: LeadSheetNote[] = [];
  for (let track = 0; track < tracks; track++) {
    if (text() !== 'MTrk') throw new Error('Missing MIDI track');
    const size = u32(), end = pos + size;
    need(size);
    let tick = 0, running = 0;
    const active = new Map<number, number[]>();
    while (pos < end) {
      tick += variable();
      let status = u8();
      if (status < 128) { pos--; status = running; }
      else if (status < 0xf0) running = status;
      if (!status) throw new Error('MIDI running status missing');
      if (status === 0xff) { u8(); const size = variable(); pos += size; }
      else if (status === 0xf0 || status === 0xf7) { const size = variable(); pos += size; }
      else {
        const kind = status & 0xf0, channel = status & 15, pitch = u8();
        const velocity = kind === 0xc0 || kind === 0xd0 ? 0 : u8();
        if ((kind === 0x90 || kind === 0x80) && (includeDrums || channel !== 9)) {
          const key = channel * 128 + pitch, queue = active.get(key) ?? [];
          active.set(key, queue);
          if (kind === 0x90 && velocity > 0) queue.push(tick);
          else {
            const start = queue.shift();
            if (start === undefined) throw new Error('Unmatched MIDI note-off');
            notes.push({ pitch, onset: start / ppq * 60 / bpm, offset: tick / ppq * 60 / bpm,
              midiTicks: { onset: start, offset: tick, ppq } });
          }
        }
      }
      if (pos > end) throw new Error('Truncated MIDI event');
    }
    if ([...active.values()].some(queue => queue.length)) throw new Error('Unterminated MIDI note');
  }
  return notes;
}
