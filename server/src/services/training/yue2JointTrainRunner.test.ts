import assert from 'node:assert/strict';
import test from 'node:test';
import { buildYue2JointTrainArgs, parseYue2JointEvent } from './yue2JointTrainRunner.js';

test('AITK native joint JSON event maps to the server metric contract', () => {
  assert.deepEqual(parseYue2JointEvent(
    '{"stage":"joint","step":7,"ar_ce":1.2,"ar_kl":0.3,"nar_mse":0.004,"gradient_norm":2.5}', 100),
    { stage: 'joint', step: 7, loss: 1.264, gradNorm: 2.5, totalSteps: 100 });
});

test('AITK joint metric includes weighted cursor loss when present', () => {
  const event = parseYue2JointEvent('{"stage":"joint","step":2,"ar_ce":1,"ar_kl":2,"nar_mse":3,"cursor_ce":4,"cursor_weight":0.08}', 10);
  assert.ok(Math.abs((event?.loss ?? 0) - 4.72) < 1e-12);
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

test('pause resume CLI stays on the absolute step axis and preview cap is synth-only', () => {
  const args = buildYue2JointTrainArgs({ checkpoint: 'b', dataset: 'd', outDir: 'segment-2',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0', resume: 'segment-1/checkpoint-step50/optimizer.resume',
    pauseAt: 75, preview: { enabled: true, everySteps: 25, seconds: 40, seed: 424242,
      previewMaxFrames: 1000, baseline: false, control: false } });
  assert.deepEqual(args.slice(-4), ['--resume', 'segment-1/checkpoint-step50/optimizer.resume', '--pause-at', '75']);
  assert.equal(args.includes('--preview-max-frames'), false);
});

test('paused JSON event is recognized without becoming terminal done', () => {
  assert.deepEqual(parseYue2JointEvent('{"stage":"paused","step":50,"resume":"optimizer.resume"}', 100),
    { stage: 'paused', step: 50, totalSteps: 100 });
});
