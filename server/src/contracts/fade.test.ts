import { test } from 'node:test';
import assert from 'node:assert/strict';
import { endEditGainAt, type EndEdit } from './fade.js';

const edit = (e: Partial<EndEdit>): EndEdit => ({
  cropIn: 0, cropOut: null, fadeIn: 0, fadeOut: 0, curveIn: 'linear', curveOut: 'linear', ...e,
});

test('endEditGainAt is silent where the edit cuts and fades inwards from each cut', () => {
  const e = edit({ cropIn: 10, cropOut: 90, fadeIn: 4, fadeOut: 10 });
  assert.equal(endEditGainAt(e, 100, 5), 0, 'before the start cut');
  assert.equal(endEditGainAt(e, 100, 95), 0, 'after the end cut');
  assert.equal(endEditGainAt(e, 100, 10), 0, 'fade-in starts at the cut');
  assert.equal(endEditGainAt(e, 100, 12), 0.5);
  assert.equal(endEditGainAt(e, 100, 50), 1, 'untouched middle');
  assert.equal(endEditGainAt(e, 100, 85), 0.5);
  assert.equal(endEditGainAt(e, 100, 90), 0, 'fade-out ends at the cut');
});

test('endEditGainAt with no edit is unity and a null end cut keeps the track end', () => {
  assert.equal(endEditGainAt(edit({}), 100, 0), 1);
  assert.equal(endEditGainAt(edit({}), 100, 100), 1);
  assert.equal(endEditGainAt(edit({ fadeOut: 20, curveOut: 'exponential' }), 100, 90), 0.25);
});
