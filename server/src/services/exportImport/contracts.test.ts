import assert from 'node:assert/strict';
import { test } from 'node:test';
import { exportRequest, importRequest, validateProfileImport } from './contracts.js';

test('export rejects malformed formats, variants, paths and oversized batches', () => {
  for (const body of [
    { items: [] },
    { items: [{ songId: '../audio/x.wav' }], format: 'aac' },
    { items: [{ songId: 'a', variant: 'bogus' }] },
    { items: Array.from({ length: 101 }, () => ({ songId: 'a' })) },
    { items: [{ songId: 'a' }], bitrate: 0 },
    { items: [{ songId: 'a' }], path: 'C:\\audio.wav' },
  ]) assert.equal(exportRequest.safeParse(body).success, false);
  assert.deepEqual(exportRequest.parse({ items: [{ songId: 'a' }] }).format, 'flac');
});

test('import accepts asset ids only and bounds per-item descriptions', () => {
  const assetId = '72a9d55e-0929-45b4-9d49-c15f1052aeca';
  assert.equal(importRequest.safeParse({ items: [{ assetId }] }).success, true);
  for (const body of [
    { items: [{ assetId: 'C:\\audio.wav' }] },
    { items: [{ assetId, sourcePath: '/tmp/audio.wav' }] },
    { items: [{ assetId, description: 'x'.repeat(1001) }] },
    { items: Array.from({ length: 51 }, () => ({ assetId })) },
  ]) assert.equal(importRequest.safeParse(body).success, false);
});

test('profile import unwraps a saved preset and rejects empty or malformed input', () => {
  assert.deepEqual(validateProfileImport({
    filename: 'set.json', profile: { data: { _format: 'hot-step-preset', tempo: 120 } },
  }), { name: 'set', data: { _format: 'hot-step-preset', tempo: 120 } });
  assert.throws(() => validateProfileImport({ filename: 'empty.json', profile: {} }));
  assert.throws(() => validateProfileImport({ filename: 'bad.json', profile: [] }));
});
