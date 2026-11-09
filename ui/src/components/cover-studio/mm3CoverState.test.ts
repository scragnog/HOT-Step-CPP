import assert from 'node:assert/strict';
import test from 'node:test';
import { DEFAULT_MM3_COVER, mm3CoverInput, mm3CoverReady, restoreMm3Cover } from './mm3CoverState';

test('old saved controls are ignored and cannot change semantic hard pinning', () => {
  const restored = restoreMm3Cover({ plank: 'a.mm3plank', plan: false, sound: true,
    semanticOnly: false, forceBias: '16', patternMode: 'prefix', launder: false });
  assert.deepEqual(restored, { ...DEFAULT_MM3_COVER, plank: 'a.mm3plank' });
  assert.deepEqual(mm3CoverInput(restored), { plank: 'a.mm3plank', semanticOnly: true, trimSeconds: null, arSeed: -1 });
  assert.equal(mm3CoverReady(DEFAULT_MM3_COVER), false);
  assert.equal(mm3CoverReady(restored), true);
  assert.deepEqual(restoreMm3Cover(null), DEFAULT_MM3_COVER);
});

test('trim, planner seed and launder settings survive restore', () => {
  const restored = restoreMm3Cover({ launderSteps: 40, launderSeed: -1, launderCaption: 'acoustic guitar', trimSeconds: '12', arSeed: 9 });
  assert.equal(restored.launderSteps, 40);
  assert.equal(restored.launderSeed, -1);
  assert.equal(restored.launderCaption, 'acoustic guitar');
  assert.equal(mm3CoverInput(restored).trimSeconds, 12);
  assert.equal(mm3CoverInput(restored).arSeed, 9);
  assert.equal(mm3CoverInput({ ...restored, trimSeconds: 'bad' }).trimSeconds, null);
});
