import assert from 'node:assert/strict';
import test from 'node:test';
import { rungOverall } from './yue2BestRung.js';

test('rungOverall matches the Refine scoreboard formula', () => {
  assert.equal(rungOverall(5, 1, 0, 2), 5);           // (5 + 5) / 2, no replans
  assert.equal(rungOverall(4, 2, 2, 2), 3.75);        // 4 − 0.25 × (2 replans / 2 takes)
  assert.equal(rungOverall(4, 2, 40, 2), 3);          // penalty capped at 1
  assert.equal(rungOverall(4, null, 0, 2), null);     // needs both scores
});
