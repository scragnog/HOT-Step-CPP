import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { buildC4Sidecar, readWords5, vocalNotes, writeC4Sidecar } from './yue2-c4-sidecar.js';

const score = `X:1
M:5/4
L:1/4
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody"
V: Ins clef=treble name="Ins Melody"
K:C
V: Vocal
z C-C D E | Z2 |
V: Ins
C4 | Z2 |
`;

test('maps words to Vocal notes, retains ties and marks multiple notes per word', () => {
  const result = buildC4Sidecar(score, 'one two', [
    { start: 0.5, end: 1.45, score: 0.1, char0: 0, char1: 3 },
    { start: 1.5, end: 2.45, score: 0.2, char0: 4, char1: 7 },
  ]);
  assert.equal(result.notes.length, 4);
  assert.deepEqual(result.words.map(w => w.noteIndices), [[0, 1], [2, 3]]);
  assert.equal(result.words[0].hasTie, true);
  assert.equal(result.words[0].possibleMelisma, false);
  assert.equal(result.words[1].possibleMelisma, true);
  assert.equal(result.summary.clean, 2);
});

test('shared notes and rests are uncertain rather than invented matches', () => {
  const result = buildC4Sidecar(score, 'a b c', [
    { start: 0.1, end: 0.3, score: 0, char0: 0, char1: 1 },
    { start: 0.55, end: 0.8, score: 0, char0: 2, char1: 3 },
    { start: 0.85, end: 0.95, score: 0, char0: 4, char1: 5 },
  ]);
  assert.equal(result.words[0].reason, 'no-overlapping-vocal-note');
  assert.equal(result.words[1].reason, 'note-shared-by-words');
  assert.equal(result.words[2].reason, 'note-shared-by-words');
  assert.equal(result.summary.uncertain, 3);
});

test('words5 codepoint offsets and malformed rows', () => {
  const bytes = Buffer.alloc(20);
  [0, 0.5, 0.1, 2, 6].forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  assert.deepEqual(readWords5(bytes, '😀 word')[0], { start: 0, end: 0.5,
    score: bytes.readFloatLE(8), char0: 2, char1: 6 });
  assert.throws(() => readWords5(bytes, 'word'), /Malformed words5/);
  assert.throws(() => readWords5(bytes.subarray(0, 19), '😀 word'), /complete f32 rows/);
});

test('rejects a missing Vocal voice', () => {
  assert.throws(() => vocalNotes(score.replaceAll('V: Vocal', 'V: Melody')), /Vocal voice/);
});

test('note seconds follow meter and tempo changes in the Vocal bar grid', () => {
  const changing = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'V: Vocal', 'K:C',
    'V: Vocal', 'C4|', 'M:1/4', 'D|', 'Q:1/4=120', 'E|', ''].join('\n');
  const { notes } = vocalNotes(changing);
  assert.deepEqual(notes.map(note => [note.start, note.end, note.bar]),
    [[0, 4, 1], [4, 5, 2], [5, 5.5, 3]]);
});

test('writes a JSON sidecar from a manifest and its relative cursor file', t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-c4-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const bytes = Buffer.alloc(20);
  [0.5, 1.45, 0.1, 0, 3].forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  fs.writeFileSync(path.join(dir, 'words.f32'), bytes);
  const manifest = path.join(dir, 'manifest.json');
  fs.writeFileSync(manifest, JSON.stringify({ sources: [{ abc: score, lyrics: 'one', cursor_words: 'words.f32' }] }));
  const output = path.join(dir, 'out', 'sidecar.json');
  const result = writeC4Sidecar(manifest, 0, output);
  assert.equal(result.summary.total, 1);
  assert.deepEqual(JSON.parse(fs.readFileSync(output, 'utf8')).words[0].noteIndices, [0, 1]);
  fs.writeFileSync(manifest, JSON.stringify({ sources: [{ abc: score, lyrics: 'one', cursor_words: '../words.f32' }] }));
  assert.throws(() => writeC4Sidecar(manifest, 0, output), /escapes cache/);
});
