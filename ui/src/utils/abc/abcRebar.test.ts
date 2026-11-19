import assert from 'node:assert/strict';
import test from 'node:test';
import { barMeters, previewRegroup, scoreTimeline, type RegroupResult } from './abcRebar';
import { mergeScoreSections } from './abcSectionMove';
import { prepareSectionLabels } from '../../../../server/src/contracts/yue2Prepare';

const HEADER = ['X:1', 'T:', 'M:4/4', 'L:1/32', 'Q:1/4=120',
  'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"', 'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:C#m'];

// The verse/chorus excerpt from the original report, as the transcriber wrote it.
const REPORT = [...HEADER,
  '% verse', 'V: Vocal', '"C#m"z4c4c4c4^B4c4d4c4-|"C#m"c4z8c4^B4c4d4A4-|', 'V: Ins', 'Z2|',
  'V: Vocal', 'M:3/4', '"A"A4z12e4e4|', 'V: Ins', 'M:3/4', 'Z|',
  '% pre-chorus', 'V: Vocal', 'M:4/4', '"G#"z4A4G4F8E4D4C4|', 'V: Ins', 'M:4/4', 'Z|',
  '% chorus', 'V: Vocal', 'M:1/4', '"C#m"z4c4|', 'V: Ins', 'M:1/4', 'Z|',
  'V: Vocal', 'M:4/4', '"C#m"c4c4c4z4B4B4c4c4-|"C#m"c4z8c2c2c4d4e4e4-|', 'V: Ins', 'M:4/4', 'Z2|',
  '% chorus', 'V: Vocal', '"A"e2d2c4z8e4e4f4f2e2|', 'V: Ins', 'Z|',
  'V: Vocal', 'M:3/4', '"G#"c4d6c2d8z4|', 'V: Ins', 'M:3/4', 'Z|',
].join('\n');

const ok = (result: RegroupResult) => {
  if ('error' in result) assert.fail(result.error);
  return result;
};
const refused = (result: RegroupResult, pattern: RegExp) => {
  assert.ok('error' in result, 'expected a refusal');
  assert.match(result.error, pattern);
};
// Same sound in every voice: notes (pitch, start, length), chord symbols, rests, total.
const sameMusic = (before: string, after: string) => {
  const round = (x: unknown) => JSON.parse(JSON.stringify(x, (_, v) => typeof v === 'number' ? Math.round(v * 1e9) / 1e9 : v));
  const a = round(scoreTimeline(before)), b = round(scoreTimeline(after));
  assert.deepEqual(b.notes, a.notes);
  assert.deepEqual(b.staves, a.staves);
  assert.equal(b.total, a.total);
  assert.ok(a.notes.flat().length > 0);
};

test('the report: a section boundary inside a new bar is refused', () => {
  // 3/4 + 4/4 + 1/4 across verse, pre-chorus and chorus: the pre-chorus would start mid-bar.
  refused(previewRegroup(REPORT, [3, 5], '4/4'), /"pre-chorus" would start inside a new 4\/4 bar.*Move or merge/);
});

test('the report: after merging, 3/4 + 4/4 + 1/4 becomes two 4/4 bars in both voices', () => {
  let abc = REPORT;
  for (let i = 0; i < 2; i++) abc = (mergeScoreSections(abc, 0, 'first') as { abc: string }).abc;
  assert.deepEqual(prepareSectionLabels(abc), ['verse', 'chorus']);
  assert.deepEqual((barMeters(abc) as string[]).slice(0, 7), ['4/4', '4/4', '3/4', '4/4', '1/4', '4/4', '4/4']);
  const result = ok(previewRegroup(abc, [3, 5], '4/4'));
  assert.deepEqual(result.bars, [3, 4]);
  assert.equal(result.abc, [...HEADER,
    '% verse', 'V: Vocal', '"C#m"z4c4c4c4^B4c4d4c4-|"C#m"c4z8c4^B4c4d4A4-|', 'V: Ins', 'Z2|',
    'V: Vocal', '"A"A4z12e4e4"G#"z4A4|G4F8E4D4C4"C#m"z4c4|', 'V: Ins', 'z32|z32|',
    'V: Vocal', 'M:4/4', '"C#m"c4c4c4z4B4B4c4c4-|"C#m"c4z8c2c2c4d4e4e4-|', 'V: Ins', 'M:4/4', 'Z2|',
    '% chorus', 'V: Vocal', '"A"e2d2c4z8e4e4f4f2e2|', 'V: Ins', 'Z|',
    'V: Vocal', 'M:3/4', '"G#"c4d6c2d8z4|', 'V: Ins', 'M:3/4', 'Z|',
  ].join('\n'));
  sameMusic(abc, result.abc);
  assert.deepEqual(prepareSectionLabels(result.abc), ['verse', 'chorus']);
  // The final 3/4 bar is outside the run and stays.
  assert.deepEqual(barMeters(result.abc), ['4/4', '4/4', '4/4', '4/4', '4/4', '4/4', '4/4', '3/4']);
});

const SCORE = (bar3Vocal = 'c4"D"d4-', bar3Ins = 'F8', extra: string[] = []) => [
  'X:1', 'M:4/4', 'L:1/8', 'V: Vocal', 'V: Ins', 'K:G',
  '% verse', 'V: Vocal', '"G"G8|', 'V: Ins', 'G,8|',
  'V: Vocal', 'M:3/4', '"C"c2d2^c2|', 'V: Ins', 'M:3/4', '"C"C6|',
  'V: Vocal', 'M:4/4', ...extra, `${bar3Vocal}|`, 'V: Ins', 'M:4/4', `${bar3Ins}|`,
  'V: Vocal', 'M:1/4', 'd2|', 'V: Ins', 'M:1/4', 'Z|',
  'V: Vocal', 'M:4/4', '"G"g8|', 'V: Ins', 'M:4/4', 'G8|',
].join('\n');

test('3/4 + 4/4 + 1/4: notes split and tied, accidentals rewritten, chords kept, in both voices', () => {
  const abc = SCORE();
  const result = ok(previewRegroup(abc, [2, 4], '4/4'));
  assert.equal(result.abc, [
    'X:1', 'M:4/4', 'L:1/8', 'V: Vocal', 'V: Ins', 'K:G',
    '% verse', 'V: Vocal', '"G"G8|', 'V: Ins', 'G,8|',
    // The c after ^c lost its bar line, so it is marked natural; the tie into d2 stays.
    'V: Vocal', '"C"c2d2^c2=c2-|c2"D"d4-d2|',
    // F is sharp in G; the 1/4 Z becomes a real rest.
    'V: Ins', '"C"C6F2-|F6z2|',
    'V: Vocal', 'M:4/4', '"G"g8|', 'V: Ins', 'M:4/4', 'G8|',
  ].join('\n'));
  sameMusic(abc, result.abc);
});

test('a tied-on note that lands at a new bar line gets its accidental written', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/8', 'K:C', 'M:3/4', 'z6|', 'M:4/4', '^c4z4|', 'M:1/4', 'z2|', 'M:4/4', 'c8|'].join('\n');
  const result = ok(previewRegroup(abc, [1, 3], '4/4'));
  assert.equal(result.abc, ['X:1', 'M:4/4', 'L:1/8', 'K:C', 'z6^c2-|^c2z6|', 'M:4/4', 'c8|'].join('\n'));
  sameMusic(abc, result.abc);
});

test('a simple join, 3/4 + 1/4, keeps an accidental that used to reset at the removed bar line', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/8', 'K:C', 'C8|', 'M:3/4', '^F2G2A2|', 'M:1/4', 'F2|', 'M:4/4', 'G8|'].join('\n');
  const result = ok(previewRegroup(abc, [2, 3], '4/4'));
  assert.equal(result.abc, ['X:1', 'M:4/4', 'L:1/8', 'K:C', 'C8|', '^F2G2A2=F2|', 'M:4/4', 'G8|'].join('\n'));
  assert.deepEqual(result.bars, [2, 2]);
  sameMusic(abc, result.abc);
});

test('a run inside a line: the bars around it keep their text and meter', () => {
  // Without voices; the pickup and the last bar are outside the run.
  const abc = ['X:1', 'M:4/4', 'L:1/8', 'K:C', 'M:1/4', 'G2|', 'M:4/4', 'c6d2|e2f6|g8|', 'M:3/4', 'a6|'].join('\n');
  const result = ok(previewRegroup(abc, [3, 3], '2/4'));
  assert.equal(result.abc, ['X:1', 'M:4/4', 'L:1/8', 'K:C', 'M:1/4', 'G2|', 'M:4/4', 'c6d2|', 'M:2/4', 'e2f2-|f4|', 'M:4/4', 'g8|', 'M:3/4', 'a6|'].join('\n'));
  assert.deepEqual(result.bars, [3, 4]);
  sameMusic(abc, result.abc);
  assert.deepEqual(barMeters(result.abc), ['1/4', '4/4', '2/4', '2/4', '4/4', '3/4']);
});

test('CRLF scores keep CRLF', () => {
  const abc = ['X:1', 'M:4/4', 'L:1/8', 'K:C', 'M:3/4', 'c6|', 'M:1/4', 'd2|', 'M:4/4', 'e8|'].join('\r\n');
  assert.equal(ok(previewRegroup(abc, [1, 2], '4/4')).abc, ['X:1', 'M:4/4', 'L:1/8', 'K:C', 'c6d2|', 'M:4/4', 'e8|'].join('\r\n'));
});

test('refusals leave the reason, and nothing else', () => {
  refused(previewRegroup(SCORE(), [2, 3], '4/4'), /hold 7 beats, which is not a whole number of 4\/4 bars/);
  refused(previewRegroup(SCORE(undefined, undefined, ['K:D']), [2, 4], '4/4'), /key changes inside bars 2–4/);
  refused(previewRegroup(SCORE(undefined, undefined, ['Q:1/4=90']), [2, 4], '4/4'), /tempo changes inside bars 2–4/);
  refused(previewRegroup(SCORE('[K:D]c4"D"d4-'), [2, 4], '4/4'), /key changes inside/);
  // Unequal voice durations: the Ins bar is a beat short.
  refused(previewRegroup(SCORE(undefined, 'F6'), [2, 4], '4/4'), /Bar 3 in Ins does not fill its 4\/4 meter/);
  refused(previewRegroup(SCORE('(3c2c2c2"D"d2-'), [2, 4], '4/4'), /Bar 3 has notation this cannot regroup \("\("\)/);
  refused(previewRegroup(SCORE(), [1, 1], '4/4'), /Bar 1 is already 4\/4/);
  refused(previewRegroup(SCORE(), [0, 2], '4/4'), /between 1 and 5/);
  refused(previewRegroup(SCORE(), [2, 4], 'four'), /not a meter/);
});
