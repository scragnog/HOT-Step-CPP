import assert from 'node:assert/strict';
import test from 'node:test';
import { measureCoverDrift } from './coverDrift.js';
import type { Yue2AlignWord } from './align.js';

const score = [
  'X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C',
  '% verse', 'V: Vocal', 'C|D|', 'V: Ins', 'G|G|',
  '% chorus', 'V: Vocal', 'E|F|', 'V: Ins', 'G|G|', '',
].join('\n');
const lyrics = '[Verse]\nfirst line\n[Chorus]\nsecond line';
function word(text: string, start: number, end: number): Yue2AlignWord {
  const char0 = Array.from(lyrics.slice(0, lyrics.indexOf(text))).length;
  return { char0, char1: char0 + Array.from(text).length, start, end, score: 0.9 };
}

test('perfect alignment has zero offsets on the score meter and beat grid', () => {
  const result = measureCoverDrift(score, score, lyrics, [word('first', 0, 0.5), word('second', 6, 6.5)]);
  assert.equal(result.secondsPerBar, 3); // 3/4 is six eighth notes at 120/min.
  assert.deepEqual(result.sections.map(s => [s.startBar, s.endBar]), [[1, 2], [3, 4]]);
  assert.deepEqual(result.sections.map(s => s.offsetBars), [0, 0]);
  assert.equal(result.meanAbsoluteOffsetBars, 0);
  assert.equal(result.firstOverOneBar, null);
});

test('a section one bar late reports one bar and contributes to the mean', () => {
  const result = measureCoverDrift(score, score, lyrics, [word('first', 0, 0.5), word('second', 9, 9.5)]);
  assert.equal(result.sections[1].offsetBars, 1);
  assert.equal(result.meanAbsoluteOffsetBars, 0.5);
  assert.equal(result.firstOverOneBar, null);
  const later = measureCoverDrift(score, score, lyrics, [word('first', 0, 0.5), word('second', 12, 12.5)]);
  assert.deepEqual(later.firstOverOneBar, { index: 2, label: 'chorus', offsetBars: 2 });
});

test('free tempo falls back to the reviewed score and says so', () => {
  const result = measureCoverDrift(score.replace('Q:1/8=120\n', ''), score, lyrics, [word('first', 0, 0.5)]);
  assert.equal(result.tempoSource, 'source-score-fallback');
  assert.equal(result.tempoBpm, 120);
  assert.equal(result.secondsPerBar, 3);
});

test('missing words and a mismatched tag are reported without inventing a time', () => {
  const wrong = lyrics.replace('[Chorus]', '[Verse 2]');
  const result = measureCoverDrift(score, score, wrong, [word('first', 0, 0.5)]);
  assert.equal(result.sections[1].sung, null);
  assert.equal(result.sections[1].offsetBars, null);
  assert.match(result.sectionWarning!, /score section 2 is chorus, lyric tag 2 is Verse 2/);
});

test('qualification-style multi-bar rests put verse after the intro', () => {
  const fixture = [
    'X:1', 'M:4/4', 'L:1/16', 'Q:1/4=122', 'K:Em', '% intro',
    'V: Vocal', 'Z4|', 'V: Ins', 'V: Vocal', 'Z4|', 'V: Ins',
    'V: Vocal', 'Z4|', 'V: Ins', 'V: Vocal', 'Z4|', 'V: Ins',
    'V: Vocal', 'Z|', 'V: Ins', '% verse', 'V: Vocal', 'B4A2G2A4A2G2|', '',
  ].join('\n');
  const result = measureCoverDrift(fixture, fixture, '[Intro]\nword\n[Verse]\nword', []);
  assert.equal(result.sections[1].startBar, 18);
  assert.ok(Math.abs(result.sections[1].expected.start - 17 * 4 * 60 / 122) < 1e-8);
});
