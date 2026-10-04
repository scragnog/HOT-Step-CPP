// ReviewPanel.test.ts — a remote-origin ladder must stay reachable through
// "Finish scored" (Reviewer, slice 3 blocker #5: the old filter excluded
// every row with an `origin`, making a finished, scored remote ladder
// unreachable from this UI). No UI test runner is wired up for this project;
// run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/ReviewPanel.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { selectFinishable } from './ReviewPanel.tsx';
import type { Yue2ReviewRow } from '../../services/trainingApi.js';

function row(over: Partial<Yue2ReviewRow>): Yue2ReviewRow {
  return {
    datasetId: 'd1', datasetSlug: 'd', datasetName: 'd', refineRun: 'r1', status: 'done',
    createdAt: 1, live: false, rungs: 2, previews: 4, scored: 2, unscored: 0,
    klMin: null, klMax: null, reviewed: true, best: { step: 10, overall: 4, blindLabel: 'A' },
    decoderOnly: false, ...over,
  };
}

test('a scored, finished, remote-origin row is still finishable', () => {
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
