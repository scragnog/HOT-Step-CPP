import assert from 'node:assert/strict';
import test from 'node:test';
import { lintScoreSections, scoreSections } from './scoreSections.js';
import { classifyYue2Score } from './scoreHealth.js';

// Vocal lines and section markers copied from the first two groups of
// _experiments/yue2-cover-qual-2026-10-02/approved.abc.
const transcribedExcerpt = [
  'X:1', 'K:C', '% intro',
  'V: Vocal', 'Z4|', 'V: Ins', 'V: Vocal', 'Z4|', 'V: Ins',
  'V: Vocal', 'Z4|', 'V: Ins', 'V: Vocal', 'Z4|', 'V: Ins',
  'V: Vocal', 'Z|', 'V: Ins', '% verse', 'V: Vocal',
  'B4A2G2A4A2G2|A2B4z8z2|B4A4A4A2G2|A2B4z8z2|',
].join('\n');

test('reads ordered sections and Vocal bar numbers from a transcribed score excerpt', () => {
  assert.deepEqual(scoreSections(transcribedExcerpt), [
    { label: 'intro', startBar: 1 }, { label: 'verse', startBar: 18 },
  ]);
  const health = classifyYue2Score(transcribedExcerpt);
  assert.equal(health.bars, 21);
  assert.deepEqual(health.sections, ['intro', 'verse']);
});

test('empty score has no sections', () => {
  assert.deepEqual(scoreSections(''), []);
});

test('lint reports count and order mismatches, and passes matching tags', () => {
  const sections = scoreSections(transcribedExcerpt);
  assert.equal(lintScoreSections(sections, '[Intro]\nhello\n[Verse]\nworld').ok, true);
  const count = lintScoreSections(sections, '[Intro]\nhello');
  assert.equal(count.ok, false);
  assert.match(count.message, /score has 2 sections, lyrics have 1 tag/);
  assert.match(count.message, /score section 2 is verse, lyric tag 2 is missing/);
  const order = lintScoreSections(sections, '[Verse]\nhello\n[Intro]\nworld');
  assert.equal(order.ok, false);
  assert.match(order.message, /score section 1 is intro, lyric tag 1 is Verse/);
});
