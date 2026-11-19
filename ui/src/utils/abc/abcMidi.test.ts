import assert from 'node:assert/strict';
import test from 'node:test';
import { abcToMidi } from './abcMidi';
import { parseMidiFile } from '../../../../server/src/services/midiParser';

const SCORE = ['X:1', 'T:', 'M:4/4', 'L:1/8', 'Q:1/4=96', 'V: Vocal name="Vocal"', 'V: Ins name="Ins"', 'K:C',
  'V: Vocal', '"C"C2 E2 G2 c2 | "G"B4 G4 |', 'V: Ins', 'C,4 G,4 | G,,8 |'].join('\n');

test('abcToMidi writes one track per voice at the score tempo', () => {
  const midi = parseMidiFile(Buffer.from(abcToMidi(SCORE)!));
  const pitches = new Set(midi.notes.map(n => n.pitch));
  for (const p of [60, 64, 67, 72, 71, 36, 43]) assert.ok(pitches.has(p), `pitch ${p}`);
  // two bars of 4/4 at 96 BPM = 5 s
  const vocalEnd = Math.max(...midi.notes.filter(n => n.pitch === 67).map(n => n.start + n.duration));
  assert.ok(Math.abs(vocalEnd - 5) < 0.05, `ends at ${vocalEnd}`);
});

test('abcToMidi is null for a score with no notes', () => {
  assert.equal(abcToMidi('X:1\nK:C\n'), null);
});
