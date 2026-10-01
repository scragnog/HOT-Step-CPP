import { test } from 'node:test';
import assert from 'node:assert/strict';
import { translateParams } from './translateParams.js';

test('old ONNX VAE selections use the engine default', () => {
  assert.equal(translateParams({ vaeModel: 'vae-DreamVAE.onnx' }).vae_model, undefined);
  assert.equal(translateParams({ vaeModel: '' }).vae_model, undefined);
  assert.equal(translateParams({ vaeModel: 'vae-f16.gguf' }).vae_model, 'vae-f16.gguf');
});
