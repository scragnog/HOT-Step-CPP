import assert from 'node:assert/strict';
import test from 'node:test';
import { buildYue2JointTrainArgs, parseYue2JointEvent } from './yue2JointTrainRunner.js';

test('AITK native joint JSON event maps to the server metric contract', () => {
  assert.deepEqual(parseYue2JointEvent(
    '{"stage":"joint","step":7,"ar_ce":1.2,"ar_kl":0.3,"nar_mse":0.004,"gradient_norm":2.5}', 100),
    { stage: 'joint', step: 7, loss: 1.264, gradNorm: 2.5, totalSteps: 100 });
});

test('AITK CLI contract preserves checkpoint, dataset, output and resume order', () => {
  assert.deepEqual(buildYue2JointTrainArgs({
    checkpoint: 'base.gguf', dataset: 'dataset.json', outDir: 'run-new',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0', resume: 'optimizer.resume',
  }), [
    'yue2-joint-train', '--checkpoint', 'base.gguf', '--dataset', 'dataset.json',
    '--output', 'run-new', '--steps', '100', '--save-every', '10', '--seed', '3',
    '--device', 'CUDA0', '--resume', 'optimizer.resume',
  ]);
});
