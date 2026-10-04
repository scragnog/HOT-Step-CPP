// RefinePanel.skipNarFurther.test.ts — "Use this rung" with Further
// training for NAR on must reject a remote-origin ladder outright, never
// silently finish/link/delete instead (Reviewer/Lead, round 3 #4: its
// prepared dataset lives on its own worker, a path this machine cannot
// resolve — no local resume support yet, and no silent substitute for one
// either). No UI test runner is wired up for this project; run with the
// server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/RefinePanel.skipNarFurther.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { narOnUseOutcome, skipNarFurther } from './RefinePanel.tsx';
import type { Yue2AitkRunRecord } from '../../services/trainingApi.js';

function run(over: Partial<Yue2AitkRunRecord>): Pick<Yue2AitkRunRecord, 'options' | 'origin'> {
  return { options: {}, ...over };
}

test('a local run with no decoder freeze gets the NAR follow-up', () => {
  assert.equal(skipNarFurther(run({})), false);
});

test('a decoder-only follow-up never gets a second one', () => {
  assert.equal(skipNarFurther(run({ options: { freezePlannerNow: true } })), true);
});

test('NAR further off: always skip, origin or not', () => {
  assert.equal(narOnUseOutcome(run({}), false), 'skip');
  assert.equal(narOnUseOutcome(run({ origin: { worker: 'LivingRoom', remoteJobId: 'job1' } }), false), 'skip');
});

test('NAR further on, remote-origin: reject, never chain or silently finish', () => {
  assert.equal(narOnUseOutcome(run({ origin: { worker: 'LivingRoom', remoteJobId: 'job1' } }), true), 'reject');
});

test('NAR further on, decoder-only: skip (legitimate no-op, not a rejection)', () => {
  assert.equal(narOnUseOutcome(run({ options: { freezePlannerNow: true } }), true), 'skip');
});

test('NAR further on, local and not decoder-only: chain', () => {
  assert.equal(narOnUseOutcome(run({}), true), 'chain');
});
