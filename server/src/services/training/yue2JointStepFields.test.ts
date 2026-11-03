import assert from 'node:assert/strict';
import test from 'node:test';
import { parseYue2JointEvent } from './yue2JointTrainRunner.js';

test('joint step keeps planner readings separate from composite loss and omits missing frozen fields', () => {
  const joint = parseYue2JointEvent(JSON.stringify({
    stage: 'joint', step: 2, ar_ce: 1, ar_kl: 2, nar_mse: 3,
    cursor_ce: 4, cursor_weight: 0.08, lr: 0.0001,
  }), 10);
  assert.ok(joint);
  assert.ok(Math.abs((joint.loss ?? 0) - 4.72) < 1e-12);
  const { loss: _loss, ...readings } = joint;
  assert.deepEqual(readings, {
    stage: 'joint', step: 2, totalSteps: 10,
    arCe: 1, cursorCe: 4, arKl: 2, narMse: 3, lr: 0.0001,
  });

  // A frozen step has no planner terms, but the decoder is still training, so
  // it still has a loss: the decoder's own, flagged so it is never mistaken for
  // the composite (the stale-label bug this replaced).
  const frozen = parseYue2JointEvent(JSON.stringify({
    stage: 'joint', step: 3, planner_frozen: true, nar_mse: 2.5,
  }), 10);
  assert.deepEqual(frozen, {
    stage: 'joint', step: 3, totalSteps: 10, plannerFrozen: true, narMse: 2.5,
    loss: 2.5, decoderOnly: true,
  });
});
