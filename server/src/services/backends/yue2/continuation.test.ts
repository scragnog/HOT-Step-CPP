import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyContinuation, continuationCodes, jobContinuation, yue2Continuation } from './continuation.js';
import type { Yue2SynthRequest } from './client.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

test('a queued continuation reads its captured source and range, not later raw params', () => {
  const captured = { sourceAudioUrl: '/references/source.wav', start: 1, end: 3 };
  const job = { envelope: { options: { yue2: { yue2Continuation: captured } } },
    params: { yue2Continuation: { sourceAudioUrl: '/audio/stale.wav', start: 4, end: 5 } } } as unknown as GenerationJob;
  assert.deepEqual(jobContinuation(job), captured);
});

test('continuation captures a bounded uploaded section and refuses paths and invalid ranges', () => {
  const s = { sourceAudioUrl: '/references/abc-123.wav', start: 0.04, end: 0.12 };
  assert.deepEqual(yue2Continuation(s), s);
  for (const patch of [{ sourceAudioUrl: '/references/../secret.wav' }, { start: NaN }, { end: Infinity }, { start: -1 }, { end: 40 }, { end: 0.04 }]) {
    assert.throws(() => yue2Continuation({ ...s, ...patch }));
  }
});

test('continuation slices top-1 raw IDs on the 25 Hz clock and detects corrupt files', () => {
  const b = Buffer.alloc(16 + 4 * 2 * 8);
  b.write('Y2GD'); b.writeUInt32LE(1, 4); b.writeUInt32LE(4, 8); b.writeUInt32LE(2, 12);
  [5, 99, 6, 98, 7, 97, 8, 96].forEach((v, i) => b.writeInt32LE(v, 16 + i * 4));
  assert.deepEqual(continuationCodes(b, { start: 0.04, end: 0.12 }), [6, 7]);
  assert.throws(() => continuationCodes(b.subarray(0, -1), { start: 0, end: 0.12 }));
  assert.throws(() => continuationCodes(b, { start: 0, end: 1 }));
  b.writeInt32LE(32768, 16);
  assert.throws(() => continuationCodes(b, { start: 0, end: 0.04 }));
});

test('continuation disables planning, retries and song batching while preserving seeds and variations', () => {
  const req = { style: '', cot: 'full', abc: 'old score', semantic_retries: 3, lm_batch_size: 4,
    synth_batch_size: 2, seed: '15213961153666092475', noise_seed: '15213961153666092476', songs: [{}] } as Yue2SynthRequest;
  applyContinuation(req, [2, 3]);
  assert.deepEqual(req, { style: '', cot: 'off', codec_ids: [2, 3], continue_codec_ids: true, lm_batch_size: 1,
    synth_batch_size: 2, seed: '15213961153666092475', noise_seed: '15213961153666092476' });
});
