// mm3CkptOffloadPlan: when a run moves its activation checkpoints to host RAM.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mm3CkptOffloadPlan } from './mm3Train.js';

const bases = [{ id: 'q8_0', bytes: 9129105696, extraMb: 0 }];
const run = { basePrecision: 'q8_0', rank: 128, optimizer: 'adamw' as const, attn: 'flash' as const,
  prefixFrames: 0, trainPrecision: 'fastest', frames: 7242 };

test('a whole-song rank-128 run on a 24 GB card offloads; the same run on a 32 GB card does not', () => {
  const small = mm3CkptOffloadPlan(run, 24563, bases);
  assert.equal(small.offload, true);
  assert.ok(small.peakMb > small.budgetMb, `${small.peakMb} vs ${small.budgetMb}`);
  // 34 of 36 checkpoints of [4096, 1142 + 7242] f32.
  assert.equal(small.offloadedMb, Math.round(34 * 4096 * (1142 + 7242) * 4 / 1048576));
  assert.equal(mm3CkptOffloadPlan(run, 32607, bases).offload, false);
});

test('an unknown card or base never offloads', () => {
  assert.equal(mm3CkptOffloadPlan(run, 0, bases).offload, false);
  assert.equal(mm3CkptOffloadPlan({ ...run, basePrecision: 'Q4_K_M' }, 24563, bases).offload, false);
});
