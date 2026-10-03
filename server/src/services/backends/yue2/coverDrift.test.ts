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
  assert.equal(result.sections[1].unscoredReason, 'no_matching_lyric_tag');
  assert.match(result.sectionWarning!, /1 score sections lack a matching sung tag/);
});

test('matches sung labels in order while skipping instrumental tags and score gaps', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C',
    '% verse', 'V: Vocal', 'C4|', '% interlude', 'V: Vocal', 'z4|',
    '% chorus', 'V: Vocal', 'D4|', '% verse', 'V: Vocal', 'E4|', ''].join('\n');
  const text = '[Verse 1]\none\n[Instrumental Break]\n[Chorus]\ntwo\n[Guitar Solo]\n[Verse 2]\nthree';
  const at = (value: string, start: number): Yue2AlignWord => {
    const char0 = Array.from(text.slice(0, text.indexOf(value))).length;
    return { char0, char1: char0 + value.length, start, end: start + 0.5, score: 0.9 };
  };
  const result = measureCoverDrift(abc, abc, text, [at('one', 0), at('two', 8), at('three', 12)]);
  assert.deepEqual(result.sections.map(row => row.lyricTag), ['Verse 1', null, 'Chorus', 'Verse 2']);
  assert.deepEqual(result.sections.map(row => row.unscoredReason),
    [null, 'no_matching_lyric_tag', null, null]);
  assert.deepEqual(result.sections.map(row => row.offsetBars), [0, null, 0, 0]);
  assert.equal(result.sungLyricBlocks, 3);
  assert.deepEqual(result.unmatchedLyricBlocks, []);
});

test('lists sung lyric blocks that cannot match a score label or order', () => {
  const text = '[Verse]\nfirst\n[Bridge]\nextra\n[Instrumental Break]\nstray words\n[Chorus]\nsecond\n[Outro - Instrumental]';
  const result = measureCoverDrift(score, score, text, []);
  assert.deepEqual(result.sections.map(row => row.lyricTag), ['Verse', 'Chorus']);
  assert.equal(result.sungLyricBlocks, 4);
  assert.deepEqual(result.unmatchedLyricBlocks,
    [{ index: 2, label: 'Bridge' }, { index: 3, label: 'Instrumental Break' }]);
  assert.match(result.sectionWarning!, /2 sung lyric tags unused/);
});

test('low-confidence force-fit and mix/stem disagreement leave rows unscored', () => {
  const good = word('first', 0, 0.5);
  const low = { ...word('second', 6, 6.5), score: 0.01 };
  const lowResult = measureCoverDrift(score, score, lyrics, [good, low]);
  assert.equal(lowResult.sections[1].unscoredReason, 'low_word_confidence');
  assert.equal(lowResult.sections[1].sung, null);
  assert.equal(lowResult.meanAbsoluteOffsetBars, 0);
  const disputed = measureCoverDrift(score, score, lyrics, [good, word('second', 6, 6.5)],
    [good, word('second', 9, 9.5)]);
  assert.equal(disputed.sections[1].disagreementBars, 1);
  assert.equal(disputed.sections[1].unscoredReason, 'mix_stem_disagreement');
  assert.equal(disputed.sections[1].offsetBars, null);
  const boundary = measureCoverDrift(score, score, lyrics, [good, word('second', 6, 6.5)],
    [good, word('second', 7.5, 8)]);
  assert.equal(boundary.sections[1].disagreementBars, 0.5);
  assert.equal(boundary.sections[1].unscoredReason, null);
});

test('low confidence anywhere in a section rejects a plausible opening with a force-fitted tail', () => {
  const second = word('second', 6, 6.5);
  const tail = Array.from({ length: 8 }, (_, index) =>
    ({ ...second, start: 10 + index, end: 10.5 + index, score: 0.01 }));
  const result = measureCoverDrift(score, score, lyrics, [word('first', 0, 0.5), second, ...tail]);
  assert.equal(result.sections[1].unscoredReason, 'low_word_confidence');
  assert.equal(result.sections[1].sung, null);
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

test('meter change before a section puts its start at the sum of the bars', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C', '% verse',
    'V: Vocal', 'C4|', 'M:1/4', 'D|', 'M:4/4', '% chorus', 'V: Vocal', 'E4|', ''].join('\n');
  const result = measureCoverDrift(abc, abc, '[Verse]\nfirst\n[Chorus]\nsecond', [
    { char0: 8, char1: 13, start: 0, end: 0.5, score: 1 },
    { char0: 23, char1: 29, start: 5, end: 5.5, score: 1 },
  ]);
  assert.equal(result.sections[1].expected.start, 5);
  assert.equal(result.sections[1].offsetBars, 0);
});

test('tempo change within a section changes later bar and section times', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C', '% verse',
    'V: Vocal', 'C4|', 'Q:1/4=120', 'D4|', '% chorus', 'V: Vocal', 'E4|', ''].join('\n');
  const result = measureCoverDrift(abc, abc, '[Verse]\nfirst\n[Chorus]\nsecond', [
    { char0: 8, char1: 13, start: 0, end: 0.5, score: 1 },
    { char0: 23, char1: 29, start: 6, end: 6.5, score: 1 },
  ]);
  assert.equal(result.sections[0].expected.end, 6);
  assert.equal(result.sections[1].expected.start, 6);
  assert.equal(result.sections[1].offsetBars, 0);
});

test('meter change inside a section changes its end and following start', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C', '% verse',
    'V: Vocal', 'C4|', 'M:1/4', 'D|', 'E|', 'M:4/4', '% chorus', 'V: Vocal', 'F4|', ''].join('\n');
  const result = measureCoverDrift(abc, abc, '[Verse]\nfirst\n[Chorus]\nsecond', [
    { char0: 8, char1: 13, start: 0, end: 0.5, score: 1 },
    { char0: 23, char1: 29, start: 6, end: 6.5, score: 1 },
  ]);
  assert.equal(result.sections[0].expected.end, 6);
  assert.equal(result.sections[1].expected.start, 6);
});
