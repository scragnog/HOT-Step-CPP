// RefinePanel.skipNarFurther.test.ts — "Use this rung" with Further
// training for NAR on must never chain a resume onto a remote-origin ladder
// (Reviewer, slice 3 blocker #6: its prepared dataset lives on its own
// worker, a path this machine cannot resolve). No UI test runner is wired
// up for this project; run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/RefinePanel.skipNarFurther.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { skipNarFurther } from './RefinePanel.tsx';
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

test('a remote-origin run skips NAR further training', () => {
  assert.equal(skipNarFurther(run({ origin: { worker: 'LivingRoom', remoteJobId: 'job1' } })), true);
});
