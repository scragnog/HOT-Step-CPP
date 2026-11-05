import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizePpParams } from './postProcessing.js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { config } from '../../config.js';
import { aceClient } from '../aceClient.js';
import { encodeWav } from './audioLevel.js';
import { runPostProcessingChain } from './postProcessing.js';
import { requestedPpStages, startRePostProcess } from './rePostProcess.js';

// The ONNX StableStep backend is retired: a saved 'onnx' must load as 'auto'
// (GGML), so no request built from an old profile can ask the engine for ONNX.
test('normalizePpParams: stableStepBackend keeps only gguf, everything else is auto', () => {
  const backend = (v: unknown) => normalizePpParams({ stableStepBackend: v }, []).stableStepBackend;
  assert.equal(backend('onnx'), 'auto');
  assert.equal(backend('gguf'), 'gguf');
  assert.equal(backend('auto'), 'auto');
  assert.equal(backend(undefined), 'auto');
  assert.equal(backend('trt'), 'auto');
});

test('refiner settings normalize and re-process identifies the GPU stage', () => {
  const defaults = normalizePpParams({}, []);
  assert.equal(defaults.mm3RefinerEnabled, false);
  assert.equal(defaults.mm3RefinerSteps, 32);
  const saved = normalizePpParams({ mm3RefinerEnabled: true, mm3RefinerWindowSeconds: 0,
    mm3RefinerOverlapSeconds: 0, mm3RefinerMix: 0.5 }, []);
  assert.equal(saved.mm3RefinerWindowSeconds, 0);
  assert.equal(saved.mm3RefinerOverlapSeconds, 0);
  assert.equal(requestedPpStages(saved)[0], 'MM3 refiner');
  assert.ok(!requestedPpStages({ ...saved, mm3RefinerMix: 0 }).includes('MM3 refiner'));
});

test('refiner processes only a copy, ahead of the existing chain; bypasses are byte-exact', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-refiner-chain-'));
  t.mock.getter(config.data, 'audioDir', () => dir);
  t.mock.getter(config.vst, 'chainFile', () => path.join(dir, 'empty-vst-chain.json'));
  t.after(() => { fs.rmSync(dir, { recursive: true, force: true }); });
  const source = encodeWav({ channels: [new Float32Array(129).fill(0.1)], sampleRate: 44100,
    numChannels: 1, sourceFormat: 's16' }, 's16');
  fs.writeFileSync(path.join(dir, 'raw.wav'), source);
  let called = 0;
  const refined = encodeWav({ channels: [new Float32Array(129).fill(0.2)], sampleRate: 44100,
    numChannels: 1, sourceFormat: 'f32' }, 'f32');
  t.mock.method(aceClient, 'submitMm3Refine', async () => { called++; return 'refine-job'; });
  t.mock.method(aceClient, 'pollJob', async () => ({ status: 'done' }));
  t.mock.method(aceClient, 'getJobResult', async () => new Response(refined));
  for (const params of [{ postProcessingEnabled: false, mm3RefinerEnabled: true },
    { mm3RefinerEnabled: false }, { mm3RefinerEnabled: true, mm3RefinerMix: 0 }]) {
    const result = await runPostProcessingChain(['/audio/raw.wav'], params, 1, 'test', () => {}, () => {});
    assert.deepEqual(result.masteredUrls, ['']);
    assert.deepEqual(fs.readFileSync(path.join(dir, 'raw.wav')), source);
  }
  assert.equal(called, 0);
  const result = await runPostProcessingChain(['/audio/raw.wav'], { mm3RefinerEnabled: true },
    1, 'test', () => {}, () => {});
  assert.deepEqual(result.masteredUrls, ['/audio/raw_mastered.wav']);
  assert.deepEqual(fs.readFileSync(path.join(dir, 'raw_mastered.wav')), refined);
  assert.deepEqual(fs.readFileSync(path.join(dir, 'raw.wav')), source);
  assert.equal(called, 1);
  const lifter = t.mock.method(aceClient, 'submitSpectralLifter', async (input: Buffer) => {
    assert.deepEqual(input, refined, 'the next stage must consume the refined copy');
    return input;
  });
  await runPostProcessingChain(['/audio/raw.wav'], { mm3RefinerEnabled: true, spectralLifterEnabled: true },
    1, 'test', () => {}, () => {});
  assert.equal(lifter.mock.callCount(), 1);
  t.mock.method(aceClient, 'submitMm3Refine', async () => { throw new Error('501: no style source'); });
  const skipped = await runPostProcessingChain(['/audio/raw.wav'], { mm3RefinerEnabled: true },
    1, 'test', () => {}, () => {});
  assert.deepEqual(skipped.masteredUrls, ['']);
  assert.match(skipped.warnings?.[0] ?? '', /501: no style source/);
  assert.deepEqual(fs.readFileSync(path.join(dir, 'raw.wav')), source);
  const reprocess = startRePostProcess({ id: 'refiner-fixture', audio_url: '/audio/raw.wav',
    generation_params: '{}' }, { mm3RefinerEnabled: true });
  const deadline = Date.now() + 2000;
  while ((reprocess.status === 'pending' || reprocess.status === 'running') && Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  assert.equal(reprocess.status, 'succeeded', reprocess.error);
  assert.match(reprocess.warnings?.[0] ?? '', /501: no style source/);
  assert.equal(reprocess.masteredAudioUrl, undefined);
  assert.deepEqual(fs.readFileSync(path.join(dir, 'raw.wav')), source);
});
