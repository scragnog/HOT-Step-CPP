import assert from 'node:assert/strict';
import test from 'node:test';
import { skipNarFurther } from './yue2BatchRunner.js';
import type { Yue2AitkRunRecord } from './yue2AitkRuns.js';

function run(over: Partial<Yue2AitkRunRecord>): Pick<Yue2AitkRunRecord, 'options'> {
  return { options: {}, ...over };
}

test('a local run with no decoder freeze gets the NAR follow-up', () => {
  assert.equal(skipNarFurther(run({})), false);
});

test('a decoder-only follow-up never gets a second one', () => {
  assert.equal(skipNarFurther(run({ options: { freezePlannerNow: true } })), true);
});

test('undefined (no run found) skips rather than throwing', () => {
  assert.equal(skipNarFurther(undefined), false);
});

// A remote-origin run is rejected before skipNarFurther is even consulted —
// see stageRequest's 'nar' branch, which throws directly (Reviewer/Lead,
// round 3 #4: no silent substitute). Not unit-testable here since
// stageRequest is not exported; RefinePanel.tsx's equivalent decision
// (narOnUseOutcome) is directly tested in RefinePanel.skipNarFurther.test.ts.
