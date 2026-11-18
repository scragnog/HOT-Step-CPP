import assert from 'node:assert/strict';
import test from 'node:test';
import { measuredMetadata } from './measuredMetadata';

test('valid measurements save only the measured metadata fields', () => {
  assert.deepEqual(measuredMetadata({ bpm: 127.7, key: 'F#', scale: 'minor' }), { bpm: 128, key_scale: 'F# Minor' });
  assert.deepEqual(measuredMetadata({ bpm: 0, key: 'Bb', scale: 'major' }), { key_scale: 'Bb Major' });
  assert.deepEqual(measuredMetadata({ bpm: 95, key: '', scale: '' }), { bpm: 95 });
});

test('unknown, invalid and non-finite measurements cannot clear saved metadata', () => {
  for (const bpm of [undefined, 0, -30, NaN, Infinity]) {
    assert.equal(measuredMetadata({ bpm, key: 'unknown', scale: 'major' }), null);
  }
  assert.equal(measuredMetadata({ key: 'C', scale: '' }), null);
  assert.equal(measuredMetadata({}), null);
});
