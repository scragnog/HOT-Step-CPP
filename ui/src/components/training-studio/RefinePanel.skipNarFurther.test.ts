import assert from 'node:assert/strict';
import test from 'node:test';
import { reviewUseDecision } from '../../../../server/src/services/training/review/reviewPolicy.js';
import type { Yue2AitkRunRecord } from '../../services/trainingApi.js';

function run(over: Partial<Yue2AitkRunRecord>): Pick<Yue2AitkRunRecord, 'options' | 'origin'> {
  return { options: {}, ...over };
}

test('the Node review decision skips NAR when the user turns it off', () => {
  assert.equal(reviewUseDecision(run({}), false), 'skip');
  assert.equal(reviewUseDecision(run({ origin: { worker: 'worker', remoteJobId: 'job1' } }), false), 'skip');
});

test('a remote-origin ladder is rejected for NAR further training', () => {
  assert.equal(reviewUseDecision(run({ origin: { worker: 'worker', remoteJobId: 'job1' } }), true), 'reject');
});

test('a decoder-only run skips a second NAR pass', () => {
  assert.equal(reviewUseDecision(run({ options: { freezePlannerNow: true } }), true), 'skip');
});

test('local unfinished decoder can chain from the selected rung', () => {
  assert.equal(reviewUseDecision(run({}), true), 'chain');
});
