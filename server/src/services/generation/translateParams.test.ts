import { test } from 'node:test';
import assert from 'node:assert/strict';
import { translateParams } from './translateParams.js';
import { resolveGenerationIntent } from './intent.js';

test('ACE parity controls preserve zeros and survive compact intent resolution', () => {
  const params = { guidanceMode: 'adg', cfgIntervalStart: 0.2, cfgIntervalEnd: 0,
    retakeVariance: 0.5, retakeSeed: 4294967295 };
  for (const input of [params, resolveGenerationIntent({
    contract: 'generation-intent/1', params,
  }, 'ace', [])]) {
    const req = translateParams(input);
    assert.equal(req.guidance_mode, 'adg');
    assert.equal(req.cfg_interval_start, 0.2);
    assert.equal(req.cfg_interval_end, 0);
    assert.equal(req.retake_variance, 0.5);
    assert.equal(req.retake_seed, 4294967295);
  }
  const req = translateParams({ cfgIntervalStart: 0.2, cfgIntervalEnd: 0.8, retakeVariance: 0.5, retakeSeed: 99 });
  assert.equal(req.cfg_interval_start, 0.2);
  assert.equal(req.cfg_interval_end, 0.8);
  assert.equal(req.retake_variance, 0.5);
  assert.equal(req.retake_seed, 99);
  const zero = translateParams({ cfgIntervalStart: 0, retakeVariance: 0 });
  assert.equal(zero.cfg_interval_start, 0);
  assert.equal(zero.retake_variance, 0);
});

test('old ONNX VAE selections use the engine default', () => {
  assert.equal(translateParams({ vaeModel: 'vae-DreamVAE.onnx' }).vae_model, undefined);
  assert.equal(translateParams({ vaeModel: '' }).vae_model, undefined);
  assert.equal(translateParams({ vaeModel: 'vae-f16.gguf' }).vae_model, 'vae-f16.gguf');
});

test('runtime DoRA is opt-in and survives runtime-forcing adapter windows', () => {
  for (const adapterRuntimeDora of [undefined, false, 'true', 1]) {
    assert.equal(translateParams({ adapterRuntimeDora }).adapter_runtime_dora, undefined);
  }
  const req = translateParams({
    adapterRuntimeDora: true,
    adapterMode: 'merge',
    loraStack: [{ path: 'example.safetensors', scale: 1, stepStart: 0.2, stepEnd: 0.8 }],
  });
  assert.equal(req.adapter_mode, 'runtime');
  assert.equal(req.adapter_runtime_dora, true);
});

test('compact generation intents preserve the shared runtime DoRA opt-in', () => {
  for (const backend of ['ace', 'minimax-m3']) {
    for (const enabled of [false, true]) {
      const params = resolveGenerationIntent({
        contract: 'generation-intent/1', params: { adapterRuntimeDora: enabled },
      }, backend, []);
      assert.equal(params.adapterRuntimeDora, enabled ? true : undefined);
    }
  }
});
