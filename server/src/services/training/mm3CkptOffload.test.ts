// mm3CkptOffloadPlan: when a run moves its activation checkpoints to host RAM.
import assert from 'node:assert/strict';
import test from 'node:test';
import { fitMm3WorkerCrop, mm3CkptOffloadPlan } from './mm3Train.js';

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

test('worker caps the measured 24 GiB recipe at 3000 frames', () => {
  assert.equal(fitMm3WorkerCrop({ ...run, frames: 9000 }, 24564, true, bases), 3000);
  assert.equal(fitMm3WorkerCrop({ ...run, frames: 2000 }, 24564, true, bases), 2000);
  assert.equal(fitMm3WorkerCrop({ ...run, frames: 9000 }, 32607, true, bases), 9000);
  assert.equal(fitMm3WorkerCrop({ ...run, frames: 9000 }, 0, true, bases), 9000);
});

test('worker finds the largest fitting window for the other recipes', () => {
  const recipe = { ...run, rank: 64 };
  const frames = fitMm3WorkerCrop(recipe, 24564, false, bases);
  assert.ok(frames >= 64 && frames < run.frames);
  assert.equal(mm3CkptOffloadPlan({ ...recipe, frames }, 24564, bases).offload, false);
  assert.equal(mm3CkptOffloadPlan({ ...recipe, frames: frames + 1 }, 24564, bases).offload, true);
});
