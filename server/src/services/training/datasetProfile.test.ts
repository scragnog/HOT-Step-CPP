import test from 'node:test';
import assert from 'node:assert/strict';
import { calibrateYue2Length } from './datasetProfile.js';

test('calibrateYue2Length keeps the preset at the reference album length', () => {
  const c = calibrateYue2Length(45, 100, 10);
  assert.equal(c.steps, 100);
  assert.equal(c.saveEvery, 10);
});

test('calibrateYue2Length scales steps and saves together, keeping the rung count', () => {
  const long = calibrateYue2Length(53, 100, 10);
  assert.deepEqual([long.steps, long.saveEvery], [120, 12]);
  const short = calibrateYue2Length(32, 100, 10);
  assert.deepEqual([short.steps, short.saveEvery], [70, 7]);
});

test('calibrateYue2Length clamps the factor to 0.6..2', () => {
  assert.equal(calibrateYue2Length(5, 100, 10).factor, 0.6);
  assert.equal(calibrateYue2Length(500, 100, 10).factor, 2);
  assert.equal(calibrateYue2Length(500, 100, 10).steps, 200);
});
