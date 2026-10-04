import assert from 'node:assert/strict';
import test from 'node:test';
import { skipNarFurther } from './yue2BatchRunner.js';
import type { Yue2AitkRunRecord } from './yue2AitkRuns.js';

function run(over: Partial<Yue2AitkRunRecord>): Pick<Yue2AitkRunRecord, 'options' | 'origin'> {
  return { options: {}, ...over };
}

test('a local run with no decoder freeze gets the NAR follow-up', () => {
  assert.equal(skipNarFurther(run({})), false);
});

test('a decoder-only follow-up never gets a second one', () => {
  assert.equal(skipNarFurther(run({ options: { freezePlannerNow: true } })), true);
});

test('a remote-origin run skips NAR further training (blocker #6: no local resume path yet)', () => {
  assert.equal(skipNarFurther(run({ origin: { worker: 'LivingRoom', remoteJobId: 'job1' } })), true);
});

test('undefined (no run found) skips rather than throwing', () => {
  assert.equal(skipNarFurther(undefined), false);
});
