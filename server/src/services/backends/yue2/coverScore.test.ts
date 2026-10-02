import assert from 'node:assert/strict';
import test from 'node:test';
import { coverScoreSnapshot, stripYue2CoverChords } from './coverScore.js';

test('strips SheetSage chord symbols from tune notes', () => {
  const full = 'X:1\nK:C\nV: Vocal\n"Am"C2 "F#dim7"D2 | "Bbmaj7/F"E2 |\n';
  assert.equal(stripYue2CoverChords(full), 'X:1\nK:C\nV: Vocal\nC2 D2 | E2 |\n');
});

test('preserves headers, lyrics, other fields, comments and non-chord annotations', () => {
  const full = 'T:"Am"\r\nV: Vocal name="C"\r\nK:C\r\nw:"Am" words\r\n+:"Dm" continuation\r\n% "Em" note\r\n"Am"C "verse"D | % "Em" note\r\n';
  assert.equal(stripYue2CoverChords(full), 'T:"Am"\r\nV: Vocal name="C"\r\nK:C\r\nw:"Am" words\r\n+:"Dm" continuation\r\n% "Em" note\r\nC "verse"D | % "Em" note\r\n');
});

test('melody-only ABC is unchanged and stripping is idempotent', () => {
  const melody = 'X:1\nT:Song\nK:C\nV: Vocal\nC2 D2 | E2 |\n';
  assert.equal(stripYue2CoverChords(melody), melody);
  assert.equal(stripYue2CoverChords(stripYue2CoverChords(melody)), melody);
});

test('submit snapshot retains full ABC and derives only the rendered copy', () => {
  const full = 'X:1\nK:C\n"Am"C2 |';
  assert.deepEqual(coverScoreSnapshot(full, true), { fullScore: full, renderedAbc: full });
  assert.deepEqual(coverScoreSnapshot(full, false), { fullScore: full, renderedAbc: 'X:1\nK:C\nC2 |' });
});
