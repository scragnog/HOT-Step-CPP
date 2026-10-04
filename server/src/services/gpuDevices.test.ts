import assert from 'node:assert/strict';
import test from 'node:test';
import { buildGpuEnv, resolveJobGpuSelection, selectedGpuMemoryMB, type NvidiaGpu } from './gpuDevices.js';

const gpus: NvidiaGpu[] = [
  { index: 0, uuid: 'GPU-small', name: 'Small card', memoryMB: 12288 },
  { index: 1, uuid: 'GPU-large', name: 'Large card', memoryMB: 24576 },
];

test('a training GPU UUID overrides Settings with one CUDA-visible card', () => {
  const selection = resolveJobGpuSelection('GPU-small', 'gpu-LARGE', gpus);
  assert.equal(selection.visibleDevices, 'GPU-large');
  assert.equal(selection.forcePciOrder, false);
  const { env } = buildGpuEnv({ Cuda_Visible_Devices: 'GPU-small' }, 'GPU-large', gpus);
  assert.equal(env.CUDA_VISIBLE_DEVICES, 'GPU-large');
  assert.equal(env.Cuda_Visible_Devices, undefined);
  assert.equal(selectedGpuMemoryMB('GPU-large', gpus), 24576);
});

test('a job without a GPU UUID follows the Settings selection', () => {
  assert.equal(resolveJobGpuSelection('GPU-small', undefined, gpus).visibleDevices, 'GPU-small');
  assert.equal(resolveJobGpuSelection('', undefined, gpus).visibleDevices, 'GPU-large');
});

test('an unknown training GPU UUID is rejected', () => {
  assert.throws(() => resolveJobGpuSelection('', 'GPU-missing', gpus), /Unknown training GPU UUID: GPU-missing/);
});
