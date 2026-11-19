import assert from 'node:assert/strict';
import test from 'node:test';
import { mergeProfileField } from './profileMerge';

const live = { yue2NarSolver: 'euler', modelPath: 'keep.gguf' };

test('ACE-Step 1.5 / SA3 profiles: non-backendParams fields pass through unchanged', () => {
  // A profile without backendParams never calls the merge for it; every field it
  // does carry is returned as given, pluginParams replaced unless _mergeMaps.
  assert.deepEqual(mergeProfileField('pluginParams', { 'b:y': 2 }, { 'a:x': 1 }, false), { 'b:y': 2 });
  assert.equal(mergeProfileField('inferenceSteps', 8, 50, true), 8);
  assert.deepEqual(mergeProfileField('backendParams', undefined, live, false), undefined);
});

test('backendParams merges over the live map; _mergeMaps merges pluginParams', () => {
  assert.deepEqual(mergeProfileField('backendParams', { yue2NarSolver: 'ancestral' }, live, false),
    { yue2NarSolver: 'ancestral', modelPath: 'keep.gguf' });
  assert.deepEqual(mergeProfileField('pluginParams', { 'b:y': 2 }, { 'a:x': 1 }, true), { 'a:x': 1, 'b:y': 2 });
});
