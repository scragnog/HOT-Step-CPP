// ReviewPanel.test.ts — a ladder mirrored from a worker must stay reachable
// through "Finish scored" (an earlier filter excluded every row with an
// `origin`). No UI test runner is wired up for this project;
// run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/ReviewPanel.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { awaitingReview, selectFinishable } from './ReviewPanel.tsx';
import type { Yue2ReviewRow } from '@/services/trainingApi.js';

function row(over: Partial<Yue2ReviewRow>): Yue2ReviewRow {
  return {
    datasetId: 'd1', datasetSlug: 'd', datasetName: 'd', refineRun: 'r1', status: 'done',
    createdAt: 1, live: false, rungs: 2, previews: 4, scored: 2, unscored: 0,
    klMin: null, klMax: null, reviewed: true, best: { step: 10, overall: 4, blindLabel: 'A' },
    decoderOnly: false, ...over,
  };
}

test('a completed joint run without previews stays visible awaiting review', () => {
  const run = row({ baseMatched: true, previews: 0, scored: 0, unscored: 2, reviewed: false, best: null });
  assert.equal(awaitingReview(run), true);
  assert.deepEqual(selectFinishable([run], new Set()), []);
});

test('previews arriving do not remove the run; review completion does', () => {
  const run = row({ unscored: 2, reviewed: false });
  assert.equal(awaitingReview(run), true);
  assert.equal(awaitingReview({ ...run, reviewed: true }), false);
  assert.equal(awaitingReview({ ...run, finished: true }), false);
  assert.equal(awaitingReview({ ...run, unscored: 0 }), false);
});

test('a scored, finished, worker-origin row is still finishable', () => {
  const out = selectFinishable([row({ origin: 'LivingRoom' })], new Set());
  assert.equal(out.length, 1);
});

test('a finished row never comes back, origin or not', () => {
  const out = selectFinishable([row({ origin: 'LivingRoom', finished: true }), row({ finished: true })], new Set());
  assert.equal(out.length, 0);
});

test('a row still running, or already queued in a batch, is excluded regardless of origin', () => {
  const out = selectFinishable(
    [row({ origin: 'LivingRoom', status: 'running' }), row({ origin: 'LivingRoom', refineRun: 'r2' })],
    new Set(['r2']),
  );
  assert.equal(out.length, 0);
});

test('a row with no best rung yet is excluded', () => {
  const out = selectFinishable([row({ origin: 'LivingRoom', best: null })], new Set());
  assert.equal(out.length, 0);
});
