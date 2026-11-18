import test from 'node:test';
import assert from 'node:assert/strict';
import { lmSongsPerPass, planLmPasses } from './lmBatch.js';
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

test('an explicit lm_seed shifts instead, and the DiT seed stays put', () => {
  const p = planLmPasses({ ...base, lm_seed: 7 }, 6, 4);
  assert.deepEqual(p.map(x => x.lm_seed), [7, 11]);
  assert.deepEqual(p.map(x => x.seed), [100, 100]);
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
