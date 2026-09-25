import assert from 'node:assert/strict';
import test from 'node:test';
import { buildYue2JointTrainArgs, parseYue2JointEvent, parseYue2JointStopMode, reconKeepDecision } from './yue2JointTrainRunner.js';

test('joint stop mode accepts the explicit step count sent by the UI', () => {
  assert.equal(parseYue2JointStopMode('steps'), 'steps');
  assert.equal(parseYue2JointStopMode(undefined), 'steps');
  assert.equal(parseYue2JointStopMode('loss'), 'loss');
  assert.equal(parseYue2JointStopMode('other'), null);
});

test('AITK native joint JSON event maps to the server metric contract', () => {
  assert.deepEqual(parseYue2JointEvent(
    '{"stage":"joint","step":7,"ar_ce":1.2,"ar_kl":0.3,"nar_mse":0.004,"gradient_norm":2.5}', 100),
    { stage: 'joint', step: 7, loss: 1.264, arKl: 0.3, narMse: 0.004, gradNorm: 2.5, totalSteps: 100 });
});

test('AITK native joint JSON forwards the measured step duration', () => {
  assert.equal(parseYue2JointEvent(
    '{"stage":"joint","step":3,"ar_ce":1,"ar_kl":0,"nar_mse":0,"gradient_norm":1,"step_ms":42.5}', 10)?.stepMs, 42.5);
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
    '--device', 'CUDA0', '--nar-drift', '--resume', 'optimizer.resume',
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

test('decoder steps after KL ride only on a KL stop', () => {
  const base = { checkpoint: 'b', dataset: 'd', outDir: 'o', steps: 750, saveEvery: 25, seed: 42, device: 'CUDA0', narExtraSteps: 100 };
  const kl = buildYue2JointTrainArgs({ ...base, stopMode: 'kl', targetKl: 1.1 });
  assert.deepEqual(kl.slice(kl.indexOf('--nar-extra-steps'), kl.indexOf('--nar-extra-steps') + 2), ['--nar-extra-steps', '100']);
  assert.equal(buildYue2JointTrainArgs({ ...base, stopMode: 'steps' }).includes('--nar-extra-steps'), false);
  assert.equal(buildYue2JointTrainArgs({ ...base, stopMode: 'kl', targetKl: 1.1, narExtraSteps: 0 }).includes('--nar-extra-steps'), false);
});

test('spike guard flags ride only when the factor is on', () => {
  const base = { checkpoint: 'b', dataset: 'd', outDir: 'o', steps: 700, saveEvery: 50, seed: 42, device: 'CUDA0' };
  const on = buildYue2JointTrainArgs({ ...base, spikeFactor: 5, spikeStop: 3, spikeStopWindow: 20 });
  assert.deepEqual(on.slice(on.indexOf('--spike-factor'), on.indexOf('--spike-factor') + 6),
    ['--spike-factor', '5', '--spike-stop', '3', '--spike-stop-window', '20']);
  assert.equal(buildYue2JointTrainArgs({ ...base, spikeFactor: 0, spikeStop: 3 }).includes('--spike-stop'), false);
});

test('paused JSON event is recognized without becoming terminal done', () => {
  assert.deepEqual(parseYue2JointEvent('{"stage":"paused","step":50,"resume":"optimizer.resume"}', 100),
    { stage: 'paused', step: 50, totalSteps: 100 });
});

test('target-loss event carries its step for the post-exit validator', () => {
  const ev = parseYue2JointEvent('{"stage":"target","step":212,"ar_ce":0.4,"ar_kl":0.1,"nar_mse":0.01}', 400);
  assert.equal(ev?.stage, 'target');
  assert.equal(ev?.step, 212);
  assert.equal(ev?.totalSteps, 400);
  assert.ok(ev && Math.abs((ev.loss ?? 0) - 0.43) < 1e-12);
});

test('prodigy run carries optimizer, rank/alpha and target-loss flags', () => {
  assert.deepEqual(buildYue2JointTrainArgs({
    checkpoint: 'base.gguf', dataset: 'dataset.json', outDir: 'run-new',
    steps: 400, saveEvery: 50, seed: 3, device: 'CUDA0',
    optimizer: 'prodigy', prodigyD0: 1e-6, rank: 16, alpha: 32,
    stopMode: 'loss', targetLoss: 0.9,
  }), [
    'yue2-joint-train', '--checkpoint', 'base.gguf', '--dataset', 'dataset.json',
    '--output', 'run-new', '--steps', '400', '--save-every', '50', '--seed', '3',
    '--device', 'CUDA0', '--nar-drift', '--rank', '16', '--alpha', '32',
    '--optimizer', 'prodigy', '--prodigy-d0', '0.000001', '--target-loss', '0.9',
  ]);
});

test('recon-keep decision keeps the first checkpoint and every real improvement, drops a flat one', () => {
  const first = reconKeepDecision(undefined, 0.05, 0.01);
  assert.deepEqual(first, { keep: true, bestRecon: 0.05 });
  const improved = reconKeepDecision(0.05, 0.03, 0.01);
  assert.deepEqual(improved, { keep: true, bestRecon: 0.03 });
  const flat = reconKeepDecision(0.03, 0.029, 0.01);
  assert.deepEqual(flat, { keep: false, bestRecon: 0.03 });
  const wellPastDelta = reconKeepDecision(0.03, 0.018, 0.01);
  assert.deepEqual(wellPastDelta, { keep: true, bestRecon: 0.018 });
});

test('muon run carries lr-scale and Newton-Schulz step count, adamw adds no optimizer flag', () => {
  assert.deepEqual(buildYue2JointTrainArgs({
    checkpoint: 'b', dataset: 'd', outDir: 'run-new',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0',
    optimizer: 'muon', muonLrScale: 1.2, muonNsSteps: 7,
  }), [
    'yue2-joint-train', '--checkpoint', 'b', '--dataset', 'd', '--output', 'run-new',
    '--steps', '100', '--save-every', '10', '--seed', '3', '--device', 'CUDA0', '--nar-drift',
    '--optimizer', 'muon', '--muon-lr-scale', '1.2', '--muon-ns-steps', '7',
  ]);
  assert.deepEqual(buildYue2JointTrainArgs({
    checkpoint: 'b', dataset: 'd', outDir: 'run-new',
    steps: 100, saveEvery: 10, seed: 3, device: 'CUDA0', optimizer: 'adamw',
  }), [
    'yue2-joint-train', '--checkpoint', 'b', '--dataset', 'd', '--output', 'run-new',
    '--steps', '100', '--save-every', '10', '--seed', '3', '--device', 'CUDA0', '--nar-drift',
  ]);
});
