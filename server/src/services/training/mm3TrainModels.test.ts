import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { config } from '../../config.js';
import { MM3_LM_DEFAULTS, missingMm3TrainModels } from './mm3Train.js';

test('only Q8_0 rank-128 PiSSA training asks for an absent residual', () => {
  const root = fs.mkdtempSync(path.join(process.cwd(), '.test-mm3-models-'));
  const previous = config.aceServer.models;
  config.aceServer.models = root;
  try {
    fs.mkdirSync(path.join(root, 'mm3'));
    for (const name of ['mm3-lm-q8_0.gguf', 'mm3-lm-f16.gguf', 'mm3-depth-f16.gguf']) {
      fs.writeFileSync(path.join(root, 'mm3', name), 'fixture');
    }
    const residual = 'mm3-lm-q8_0.pissa-r128.safetensors';
    const options = { ...MM3_LM_DEFAULTS };
    assert.deepEqual(missingMm3TrainModels('train', 'q8_0', options), [residual]);
    assert.deepEqual(missingMm3TrainModels('train', 'q8_0', { ...options, hotPizza: false }), [residual]);
    for (const over of [{ pissa: false, hotPizza: false }, { rank: 64 },
      { adapterType: 'lokr' as const }, { dora: true }, { hira: true }, { loha: true }]) {
      assert.deepEqual(missingMm3TrainModels('train', 'q8_0', { ...options, ...over }), []);
    }
    assert.deepEqual(missingMm3TrainModels('train', 'f16', options), []);
    fs.writeFileSync(path.join(root, 'mm3', residual), 'locally generated residual');
    assert.deepEqual(missingMm3TrainModels('train', 'q8_0', options), []);
  } finally {
    config.aceServer.models = previous;
    fs.rmSync(root, { recursive: true, force: true });
  }
});
