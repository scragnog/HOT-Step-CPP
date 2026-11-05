import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { abcLeadSheetEvents, abcLeadSheetRoll, leadSheetRoll, midiLeadSheetNotes } from './leadSheetRoll.js';

const abc = (body: string, bpm = 90) => `X:1\nM:4/4\nL:1/4\nQ:1/4=${bpm}\nK:C\nV:Vocal\n${body}`;
const note = (onset: number, offset: number, pitch = 60) => ({ onset, offset, pitch });
const bytes = (data: Float32Array) => {
  const buffer = Buffer.alloc(data.length * 4);
  data.forEach((value, index) => buffer.writeFloatLE(value, index * 4));
  return buffer;
};

test('constant-BPM sixteenths, interleaved onsets/sustains, overlaps and same-pitch retriggers', () => {
  const r = leadSheetRoll({ bpm: 90, melody: [note(0, 1), note(1 / 3, 2 / 3), note(0, 1 / 3, 64)], chords: [] });
  assert.equal(r.length, 6);
  assert.equal(r.pianoroll[120], 1);
  assert.equal(r.pianoroll[256 + 121], 1);
  assert.equal(r.pianoroll[2 * 256 + 120], 1);
  assert.equal(r.pianoroll[2 * 256 + 121], 1); // The older overlapping note sustains here.
  assert.ok(r.drum_pianoroll.every(n => n === 0));
  assert.ok(r.context_mask.every(n => n === 0));
});

test('melody rounds half up; raw chords round nearest even to whole beats first', () => {
  const r = leadSheetRoll({ bpm: 60, melody: [note(0.125, 0.375)],
    chords: [note(0.5, 1.5, 64), note(1.5, 2.5, 67)] });
  assert.equal(r.pianoroll[256 + 120], 1);
  assert.equal(r.length, 8); // First chord spans beats 0..2; second collapses at beat 2.
  assert.equal(r.chord[4], 1);
  assert.equal(r.chord[7 * 12 + 4], 1);
  assert.equal(r.chord[7], 0);
});

test('MIDI half-grid boundaries use original integer ticks, independent of BPM float drift', () => {
  for (const bpm of [59, 73.5, 90, 123.456, 137, 180]) {
    for (let tick = 60; tick < 24000; tick += 120) {
      const onset = tick / 480 * 60 / bpm, offset = (tick + 120) / 480 * 60 / bpm;
      const r = leadSheetRoll({ bpm, chordTiming: 'midi', chords: [], melody: [{ onset, offset, pitch: 60,
        midiTicks: { onset: tick, offset: tick + 120, ppq: 480 } }] });
      const start = Math.floor((tick * 8 + 480) / 960);
      assert.equal(r.pianoroll[start * 256 + 120], 1);
      assert.equal(r.length, start + 1);
    }
  }
});

test('one-step minimum, end-exclusive duration, and 5000-step limit without silent trim', () => {
  const r = leadSheetRoll({ bpm: 60, melody: [note(0, 0), note(1249.75, 1250)], chords: [] });
  assert.equal(r.length, 5000);
  assert.equal(r.pianoroll[4999 * 256 + 120], 1);
  assert.throws(() => leadSheetRoll({ bpm: 60, melody: [note(1250, 1250)], chords: [] }), /5000/);
  for (const invalid of [note(-1, 1), note(2, 1), note(0, Infinity), note(0, 1, 128)]) {
    assert.throws(() => leadSheetRoll({ bpm: 90, melody: [invalid], chords: [] }), /Invalid/);
  }
  assert.throws(() => leadSheetRoll({ bpm: 0, melody: [note(0, 1)], chords: [] }), /BPM/);
  assert.throws(() => leadSheetRoll({ bpm: 90, melody: [], chords: [] }), /melody/);
});

test('ABC preserves non-120 tempo, pickups, ties, polyphony and repeat-pitch onsets', () => {
  const events = abcLeadSheetEvents(abc('C | [CE]2 C-C |'));
  assert.equal(events.bpm, 90);
  assert.equal(events.melody[0].onset, 0);
  assert.equal(events.melody[0].offset, 2 / 3);
  assert.equal(events.melody.filter(n => n.pitch === 60).length, 3);
  assert.ok(Math.abs(events.melody.at(-1)!.offset - 10 / 3) < 1e-6);
  const r = abcLeadSheetRoll(abc('C C C C|'));
  assert.equal(r.length, 16);
  for (const t of [0, 4, 8, 12]) assert.equal(r.pianoroll[t * 256 + 120], 1);
});

test('ABC tempo and meter changes produce raw seconds before the constant-BPM quantiser', () => {
  const events = abcLeadSheetEvents(abc('C4 |\nM:3/4\nQ:1/4=60\nD3 |'));
  assert.equal(events.melody[1].onset, 8 / 3);
  assert.ok(Math.abs(events.melody[1].offset - 17 / 3) < 1e-6);
  const r = leadSheetRoll(events);
  assert.equal(r.length, 34);
  assert.equal(r.pianoroll[16 * 256 + 62 * 2], 1);
  assert.equal(abcLeadSheetRoll(abc('C4|').replace('Q:1/4=90', 'Q:1/8=90')).bpm, 45);
  for (const tempo of ['0', 'bad', '1/4=0', '1/0=90']) {
    assert.throws(() => abcLeadSheetRoll(abc('C4|').replace('Q:1/4=90', `Q:${tempo}`)), /tempo/);
  }
});

test('ABC holds chord symbols including inversions and no-chord boundaries, with no comping gaps', () => {
  const r = abcLeadSheetRoll(abc('"C/E"C2 "G/B"D "N.C."z|'));
  assert.equal(r.length, 12); // Trailing rests do not pad upstream's condition.
  for (let t = 0; t < 8; t++) for (const pc of [0, 4, 7]) assert.equal(r.chord[t * 12 + pc], 1);
  for (const pc of [7, 11, 2]) assert.equal(r.chord[8 * 12 + pc], 1);
  assert.throws(() => abcLeadSheetRoll(abc('"Cunknown"C4|')), /Unsupported.*chord/);
  assert.throws(() => abcLeadSheetRoll('X:1\nK:C\nC|'), /tempo/);
});

const captures = fileURLToPath(new URL('../../../_experiments/lead-sheet-reference/captures/', import.meta.url));
test('ABC chord accidentals from abcjs preserve flats, sharps and slash chords', () => {
  for (const [symbol, pitches] of [
    ['Db', [1, 5, 8]], ['D♭', [1, 5, 8]],
    ['F#', [1, 6, 10]], ['F♯', [1, 6, 10]],
    ['Bbm7/Db', [1, 5, 8, 10]], ['B♭m7/D♭', [1, 5, 8, 10]],
    ['C#/G#', [1, 5, 8]], ['C♯/G♯', [1, 5, 8]],
  ] as const) {
    const events = abcLeadSheetEvents(abc(`"${symbol}"C4|`));
    assert.deepEqual(events.chords.map(n => n.pitch).sort((a, b) => a - b), pitches, symbol);
    const roll = abcLeadSheetRoll(abc(`"${symbol}"C4|`));
    for (let t = 0; t < 16; t++) {
      assert.deepEqual(Array.from(roll.chord.slice(t * 12, (t + 1) * 12)),
        Array.from({ length: 12 }, (_, pc) => Number((pitches as readonly number[]).includes(pc))), symbol);
    }
  }
  assert.throws(() => abcLeadSheetRoll(abc('"D♭unknown"C4|')), /Unsupported.*chord/);
});

test('upstream MIDI-to-roll fixtures match every byte through raw note events', async t => {
  if (!fs.existsSync(captures)) return t.skip('Local 6a captures absent; gitignored reference data is required for parity');
  let count = 0;
  for (const entry of fs.readdirSync(captures, { withFileTypes: true }).filter(e => e.isDirectory())) {
    for (const variant of ['', 'without-drums']) {
      const dir = path.join(captures, entry.name, variant);
      if (!fs.existsSync(path.join(dir, 'roll.pianoroll.f32'))) continue;
      count++;
      await t.test(`${entry.name}${variant ? '/without-drums' : ''}`, () => {
        const bpm = Number(fs.readFileSync(path.join(dir, 'midi/bpm.txt'), 'utf8'));
        const read = (name: string, drums = false) => midiLeadSheetNotes(fs.readFileSync(path.join(dir, `midi/${name}.mid`)), bpm, drums);
        const r = leadSheetRoll({ bpm, melody: read('melody'), chords: read('chord'), drums: read('drums', true), chordTiming: 'midi' });
        const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'tensors.json'), 'utf8'));
        for (const key of ['pianoroll', 'chord', 'drum_pianoroll', 'context_mask'] as const) {
          assert.equal(r.length, manifest[`roll.${key}`].shape[0]);
          assert.deepEqual(bytes(r[key]), fs.readFileSync(path.join(dir, `roll.${key}.f32`)), key);
        }
      });
    }
  }
  assert.ok(count > 0, 'Capture directory exists but contains no valid roll fixtures');
});
