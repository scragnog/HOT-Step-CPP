// Yue2LadderReview.test.ts — a record without blindLabels must never render
// a step-ordered ladder or allow scoring. No UI test runner is wired up for
// this project; run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/Yue2LadderReview.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { ladderVisibility } from './Yue2LadderReview.tsx';
import type { Yue2AitkRunRecord } from '../../services/trainingApi.js';

const checkpoints = [
  { step: 10, arPath: 'a10', narPath: 'n10' },
  { step: 20, arPath: 'a20', narPath: 'n20' },
] as Yue2AitkRunRecord['checkpoints'];

function run(blindLabels?: Record<number, string>): Yue2AitkRunRecord {
  return { version: 1, jobId: 'r1', datasetId: 'd1', datasetSlug: 'd', method: 'aitk', output: 'out',
    options: { method: 'base-matched' }, status: 'done', createdAt: 1, updatedAt: 1, checkpoints,
    ...(blindLabels ? { blindLabels } : {}) };
}

test('a base-matched run with no blindLabels yet renders no ladder and disables scoring', () => {
  const v = ladderVisibility(run(undefined), true);
  assert.equal(v.labelsPending, true);
  assert.equal(v.blind, false);
  assert.deepEqual(v.ladder, []);
  assert.deepEqual(v.visibleLadder, []);
});

test('once blindLabels land, the ladder renders sorted by label, never by step order', () => {
  const v = ladderVisibility(run({ 10: 'B', 20: 'A' }), true);
  assert.equal(v.labelsPending, false);
  assert.equal(v.blind, true);
  assert.deepEqual(v.visibleLadder.map(c => c.step), [20, 10]);
});

test('blind mode off reads the real step order regardless of labels', () => {
  const v = ladderVisibility(run(undefined), false);
  assert.equal(v.labelsPending, false);
  assert.equal(v.blind, false);
  assert.deepEqual(v.visibleLadder.map(c => c.step), [10, 20]);
});
