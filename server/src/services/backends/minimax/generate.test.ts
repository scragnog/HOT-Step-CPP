import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  MM3_DEFAULT_FLOW_HOP, MM3_DEFAULT_STEPS, MM3_FP8_CALIB_PATH, mapMinimaxParams, mm3LowStepShift,
} from './generate.js';

test('MM3 sends the shared runtime DoRA opt-in without changing legacy requests', () => {
  const params = { prompt: 'Instrumental ambient', seed: 42 };
  const legacy = mapMinimaxParams(params).req;
  assert.equal(Object.hasOwn(legacy, 'adapter_runtime_dora'), false);
  for (const adapterRuntimeDora of [false, 'true', 1]) {
    assert.deepEqual(mapMinimaxParams({ ...params, adapterRuntimeDora }).req, legacy);
  }
  const { adapter_runtime_dora, ...rest } = mapMinimaxParams({ ...params, adapterRuntimeDora: true }).req;
  assert.equal(adapter_runtime_dora, true);
  assert.deepEqual(rest, legacy);
});

test('MM3 defaults to 20 flow steps with low-step compensation', () => {
  const req = mapMinimaxParams({ prompt: 'Instrumental ambient', seed: 42 }).req;
  assert.equal(req.steps, MM3_DEFAULT_STEPS);
  assert.equal(MM3_DEFAULT_STEPS, 20);
  assert.equal(req.scheduler, 'linear');
  assert.equal(req.flow_shift, mm3LowStepShift(20));
  for (const mm3Steps of [1, 61, 'x', undefined]) {
    assert.equal(mapMinimaxParams({ prompt: 'x', seed: 42, mm3Steps }).req.steps, 20);
  }
  const at30 = mapMinimaxParams({ prompt: 'x', seed: 42, mm3Steps: 30 }).req;
  assert.equal(at30.steps, 30);
  assert.equal(Object.hasOwn(at30, 'flow_shift'), false);
});

test('MM3 sends flow_hop_frames 150 by default and an explicit 100 as itself', () => {
  const params = { prompt: 'Instrumental ambient', seed: 42 };
  assert.equal(mapMinimaxParams(params).req.flow_hop_frames, 150);
  assert.equal(MM3_DEFAULT_FLOW_HOP, 150);
  for (const mm3FlowHop of [99, 161, 'x', undefined]) {
    assert.equal(mapMinimaxParams({ ...params, mm3FlowHop }).req.flow_hop_frames, 150);
  }
  assert.equal(mapMinimaxParams({ ...params, mm3FlowHop: 100 }).req.flow_hop_frames, 100);
  assert.equal(mapMinimaxParams({ ...params, mm3FlowHop: '130' }).req.flow_hop_frames, 130);
});

test('MM3 defaults to TensorRT FP8 with the shipped calibration; BF16 stays selectable', () => {
  const params = { prompt: 'Instrumental ambient', seed: 42 };
  const def = mapMinimaxParams(params).req;
  assert.equal(def.dit_precision, 'fp8');
  assert.equal(def.dit_fp8_calib, MM3_FP8_CALIB_PATH);
  assert.match(MM3_FP8_CALIB_PATH, /server[\\/]src[\\/]data[\\/]mm3-dit-trt\.fp8\.json$/);
  assert.deepEqual(mapMinimaxParams({ ...params, mm3DitPrecision: 'fp8' }).req, def);
  const bf16 = mapMinimaxParams({ ...params, mm3DitPrecision: 'bf16' }).req;
  assert.equal(Object.hasOwn(bf16, 'dit_precision'), false);
  assert.equal(Object.hasOwn(bf16, 'dit_fp8_calib'), false);
  assert.throws(() => mapMinimaxParams({ ...params, mm3DitPrecision: 'int4' }), /Invalid MM3 DiT precision/);
});
