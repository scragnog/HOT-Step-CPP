// textPairing.test.ts — what a comparison does when one side loses words.
//
// Both callers score timings by pairing the same lyric text across two
// sequences, so the only difference they must absorb is a word one side
// dropped or invented. A dropped line is the case that matters: every word
// after it has to keep its right partner, or the score reports an error that
// belongs to the comparison rather than to the alignment.

import test from 'node:test';
import assert from 'node:assert/strict';
import { pairByText } from './textPairing.js';

const words = (text: string) => text.split(' ').map(word => ({ word }));

test('a dropped block of six words still resyncs the rest of the line', () => {
  // The near window is four words, so a six-word line lost in the middle used
  // to step both sides once and pair every remaining word with its neighbour.
  const ideal = words('one two three four five six seven eight nine ten eleven twelve');
  const got = words('one two three ten eleven twelve');
  const pairs = pairByText(ideal, got);

  assert.deepEqual(pairs.map(([a, b]) => [a.word, b.word]), [
    ['one', 'one'], ['two', 'two'], ['three', 'three'],
    ['ten', 'ten'], ['eleven', 'eleven'], ['twelve', 'twelve'],
  ]);
});

test('a block one side invented is skipped the same way', () => {
  const ideal = words('one two three ten');
  const got = words('one two three four five six seven eight nine ten');
  const pairs = pairByText(ideal, got);

  assert.deepEqual(pairs.map(([a, b]) => [a.word, b.word]), [
    ['one', 'one'], ['two', 'two'], ['three', 'three'], ['ten', 'ten'],
  ]);
});

test('a tail with no shared words is left unpaired instead of guessed', () => {
  // Nothing ahead in common means the rest cannot be paired without inventing
  // a comparison; the honest answer is to stop.
  const pairs = pairByText(words('one two x y z'), words('one two q r s'));

  assert.deepEqual(pairs.map(([a, b]) => [a.word, b.word]), [['one', 'one'], ['two', 'two']]);
});
