import test from 'node:test';
import assert from 'node:assert/strict';
import { lmSongsPerPass, planLmPasses, trackFromPlan } from './lmBatch.js';
import type { AceRequest } from '../../aceClient.js';

const base = { caption: 'c', seed: 100 } as AceRequest;

test('a batch within the cap is one pass of N', () => {
  const p = planLmPasses(base, 3, 4);
  assert.equal(p.length, 1);
  assert.equal(p[0].lm_batch_size, 3);
  assert.equal(p[0].seed, 100);
});

test('a batch above the cap splits, shifting the tied seed by the songs before it', () => {
  const p = planLmPasses(base, 9, 4);
  assert.deepEqual(p.map(x => x.lm_batch_size), [4, 4, 1]);
  assert.deepEqual(p.map(x => x.seed), [100, 104, 108]);
  assert.ok(p.every(x => x.lm_seed === undefined));
});

test('an explicit lm_seed shifts with the DiT seed, so no two songs share a pair', () => {
  const p = planLmPasses({ ...base, lm_seed: 7 }, 6, 4);
  assert.deepEqual(p.map(x => x.lm_seed), [7, 11]);
  assert.deepEqual(p.map(x => x.seed), [100, 104]);
});

test('an engine-resolved seed is left for the engine', () => {
  const p = planLmPasses({ caption: 'c', seed: -1 } as AceRequest, 5, 4);
  assert.deepEqual(p.map(x => x.seed), [-1, -1]);
});

test('batch 1 and junk sizes give one pass of one', () => {
  assert.equal(planLmPasses(base, 1, 4)[0].lm_batch_size, 1);
  assert.equal(planLmPasses(base, 0, 4).length, 1);
  assert.equal(planLmPasses(base, 2, 0).length, 2);
});

test('songs per pass follow the card: 24 and 32 GB take 4, 16 GB 2, 12 GB and unknown 1', () => {
  assert.equal(lmSongsPerPass(24563, false, 4), 4);
  assert.equal(lmSongsPerPass(32607, false, 4), 4);
  assert.equal(lmSongsPerPass(16376, false, 4), 2);
  assert.equal(lmSongsPerPass(12282, false, 4), 1);
  assert.equal(lmSongsPerPass(null, false, 4), 1);
  assert.equal(lmSongsPerPass(81920, false, 4), 4, 'never above --max-batch');
});

test('keeping the synth models resident leaves a 24 GB card one song per pass, a 32 GB card four', () => {
  assert.equal(lmSongsPerPass(24563, true, 4), 1);
  assert.equal(lmSongsPerPass(32607, true, 4), 4);
});

test('each track keeps the seed pair the engine planned it with, across split passes', () => {
  // Fixed seed 100, batch 5 in passes of 4: the engine answers seed + b and lm_seed + b per pass.
  const req = { ...base, adapter: 'dit.safetensors' } as AceRequest;
  const out = planLmPasses(req, 5, 4)
    .flatMap(p => Array.from({ length: p.lm_batch_size! }, (_, b) =>
      ({ audio_codes: 'c', seed: p.seed! + b, lm_seed: (p.lm_seed ?? p.seed!) + b })));
  const tracks = out.map((o, i) => trackFromPlan(req, o, i));
  assert.deepEqual(tracks.map(t => [t.seed, t.lm_seed]), [[100, 100], [101, 101], [102, 102], [103, 103], [104, 104]]);
  assert.ok(tracks.every(t => t.adapter === 'dit.safetensors'), 'job fields carried');
  // A plan-guard resample keeps its slot's DiT seed.
  assert.equal(trackFromPlan(base, { audio_codes: 'c', seed: 777, lm_seed: 1110 }, 1, 101).seed, 101);
});

test('a cache hit keeps the plans but renders with the current DiT seed', () => {
  // Independent LM seed 700, batch 2: planned under DiT seed 100, cached, then asked again with 200.
  const cached = [{ audio_codes: 'a', seed: 100, lm_seed: 700 }, { audio_codes: 'b', seed: 101, lm_seed: 701 }];
  const again = cached.map((p, i) => trackFromPlan({ caption: 'c', seed: 200, lm_seed: 700 } as AceRequest, p, i));
  assert.deepEqual(again.map(t => [t.audio_codes, t.seed, t.lm_seed]), [['a', 200, 700], ['b', 201, 701]]);
  // A job whose seed the engine drew keeps the engine's.
  assert.equal(trackFromPlan({ caption: 'c', seed: -1 } as AceRequest, { audio_codes: 'a', seed: 5 }, 0).seed, 5);
});
