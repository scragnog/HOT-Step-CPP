// InstaGenPanel.test.ts — the Insta-Gen duration regression from Batch 2,
// Tester's report (intent ff15b35f rendered 130s with Custom-Gen set to 10s).
// Run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/insta-gen/InstaGenPanel.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { resolveInstaDuration } from './instaDuration.ts';

test('a user-set Custom-Gen duration is captured, not dropped', () => {
  assert.equal(resolveInstaDuration(false, () => '10'), 10);
});

test('absent storage resolves to the auto sentinel', () => {
  assert.equal(resolveInstaDuration(false, () => null), -1);
});

test("CreatePanel's own -1 auto sentinel passes through unchanged", () => {
  assert.equal(resolveInstaDuration(false, () => '-1'), -1);
});

test('MM3 forces auto regardless of what is stored', () => {
  assert.equal(resolveInstaDuration(true, () => '10'), -1);
});

test('corrupt storage falls back to auto instead of throwing', () => {
  assert.equal(resolveInstaDuration(false, () => 'not json'), -1);
});
