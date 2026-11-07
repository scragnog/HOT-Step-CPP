import assert from 'node:assert/strict';
import test from 'node:test';
import { DEFAULT_MM3_COVER, mm3CoverInput, mm3CoverReady } from './mm3CoverState';

test('MM3 cover: Plan and Sound each switch their half of the request off', () => {
  const both = { ...DEFAULT_MM3_COVER, plank: 'a.mm3plank', sound: true, soundStrength: 0.5, soundHold: 0.2 };
  assert.deepEqual([mm3CoverInput(both).plank, mm3CoverInput(both).soundStrength, mm3CoverInput(both).soundHold], ['a.mm3plank', 0.5, 0.2]);
  assert.equal(mm3CoverInput({ ...both, plan: false }).plank, null);
  assert.equal(mm3CoverInput({ ...both, sound: false }).soundStrength, null);
  assert.equal(mm3CoverInput({ ...both, soundStrength: 0 }).soundStrength, null);
  assert.equal(mm3CoverReady(DEFAULT_MM3_COVER), false);
  assert.equal(mm3CoverReady({ ...DEFAULT_MM3_COVER, plan: false, sound: true }), true);
  assert.equal(mm3CoverReady({ ...both, plan: false, sound: false }), false);
});
