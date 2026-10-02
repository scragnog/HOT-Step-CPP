import assert from 'node:assert/strict';
import test from 'node:test';
import {
  coverScoreChoices, coverTempo, coverTranspose, coverVocalOnly, transformCoverScore,
} from './coverScoreTransform.js';

// The two-voice layout, fields, rests and section markers follow a transcribed
// excerpt from _experiments/yue2-cover-qual-2026-10-02/approved.abc.
const score = [
  'X:1', 'M:4/4', 'L:1/16', 'Q:1/4=122',
  'V: Vocal clef=treble name="Vocal Melody"',
  'V: Ins clef=treble name="Ins Melody"',
  'K:Em', '% intro', 'V: Vocal', '"Em/B"E2 ^F2 [K:G]"G/D"G2 _B2|',
  'V: Ins', 'z12E,2E,2|E,2E,2E2E,2|',
  '% verse', 'V: Vocal', 'B4A2G2A4A2G2|',
  'V: Ins', 'Z4|', '',
].join('\n');

test('vocal-only removes every Ins declaration and body block, preserving the rest', () => {
  const expected = score.replace('V: Ins clef=treble name="Ins Melody"\n', '')
    .replace('V: Ins\nz12E,2E,2|E,2E,2E2E,2|\n', '')
    .replace('V: Ins\nZ4|\n', '');
  assert.equal(coverVocalOnly(score), expected);
  assert.equal(coverVocalOnly(expected), expected);
});

test('tempo free removes Q:; set keeps the beat unit and is idempotent', () => {
  const free = coverTempo(score, 'free');
  assert.equal(free.includes('Q:'), false);
  assert.equal(coverTempo(free, 'free'), free);
  const set = coverTempo(score, 138);
  assert.match(set, /^Q:1\/4=138$/m);
  assert.equal(coverTempo(set, 138), set);
  assert.equal(coverTempo(score, 'source'), score);
});

test('transpose moves header and inline keys, notes, and chord root/bass; +2 and -5 round-trip', () => {
  for (const [target, header, inline, chord] of [
    ['F#m', 'K:F#m', '[K:A]', '"F#m/C#"'],
    ['Bm', 'K:Bm', '[K:D]', '"Bm/F#"'],
  ] as const) {
    const moved = coverTranspose(score, target);
    assert.match(moved, new RegExp(`^${header}$`, 'm'));
    assert.ok(moved.includes(inline));
    assert.ok(moved.includes(chord));
    assert.notEqual(moved, score);
    assert.equal(coverTranspose(moved, target), moved);
    assert.equal(coverTranspose(moved, 'Em'), score);
  }
});

test('one submit-time default set composes voices, chords, tempo, key in that order', () => {
  const choices = coverScoreChoices({});
  assert.deepEqual(choices, { voices: 'vocal', keepChords: false, tempo: 'free', key: 'source', cfgScale: 1 });
  const result = transformCoverScore(score, choices);
  assert.equal(result.fullScore, score);
  assert.equal(result.renderedAbc.includes('V: Ins'), false);
  assert.equal(result.renderedAbc.includes('"Em/B"'), false);
  assert.equal(result.renderedAbc.includes('Q:'), false);
  assert.match(result.renderedAbc, /^K:Em$/m);
  assert.equal(transformCoverScore(result.renderedAbc, choices).renderedAbc, result.renderedAbc);
  const changed = coverScoreChoices({ voices: 'vocal', keepChords: false, tempo: 140, key: 'F#m' });
  const rendered = transformCoverScore(score, changed).renderedAbc;
  assert.equal(transformCoverScore(rendered, changed).renderedAbc, rendered);
  assert.throws(() => coverScoreChoices({ tempo: 301 }), /tempo must/);
  assert.throws(() => coverScoreChoices({ cfgScale: 0 }), /cfgScale must/);
});
