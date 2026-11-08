import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  MM3_DEFAULT_FLOW_HOP, MM3_DEFAULT_STEPS, MM3_FP8_CALIB_PATH, mapMinimaxParams, mm3LowStepShift,
  mm3TailFadeStartSec,
} from './generate.js';
import { listMm3LmAdapters } from './lmAdapter.js';

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

test('MM3 audio to audio: source and dials travel only with a strength, and the source sets the length', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-a2a-'));
  const src = path.join(dir, 'source.mp3');
  fs.writeFileSync(src, '');
  try {
    const params = { prompt: 'Instrumental ambient', seed: 42, sourceAudioUrl: src };
    const plain = mapMinimaxParams(params).req;
    assert.equal(Object.keys(plain).some(k => k.startsWith('a2a_')), false);
    assert.deepEqual(mapMinimaxParams({ ...params, mm3A2aStrength: 0, mm3A2aHold: 0.5 }).req, plain);
    const on = mapMinimaxParams({ ...params, mm3A2aStrength: '0.5', mm3A2aHold: 0.25, mm3A2aSourceSeconds: 20 }).req;
    assert.equal(on.a2a_source, src);
    assert.equal(on.a2a_strength, 0.5);
    assert.equal(on.a2a_hold, 0.25);
    assert.equal(on.a2a_source_seconds, 20);
    assert.equal(Object.hasOwn(on, 'require_eos'), false);
    assert.equal(Object.hasOwn(on, 'min_frames'), false);
    assert.equal(Object.hasOwn(mapMinimaxParams({ ...params, mm3A2aStrength: 1 }).req, 'a2a_hold'), false);
    assert.throws(() => mapMinimaxParams({ ...params, sourceAudioUrl: path.join(dir, 'gone.mp3'), mm3A2aStrength: 0.5 }),
      /Source audio file is missing/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('MM3 sends the LM adapter fade only when set, and a fade forces runtime mode', (t) => {
  const installed = listMm3LmAdapters()[0];
  if (!installed) return t.skip('no MM3 LM adapter installed');
  const params = { prompt: 'x', seed: 42, mm3LmAdapter: installed.file, mm3LmAdapterMode: 'merge' };
  const plain = mapMinimaxParams(params).req as unknown as Record<string, unknown>;
  assert.equal(plain.lm_adapter_mode, 'merge');
  assert.equal(Object.keys(plain).some((k) => k.startsWith('lm_adapter_fade')), false);
  const faded = mapMinimaxParams({ ...params, mm3LmAdapterFadeOut: 150, mm3LmAdapterFadeLen: 60, mm3LmAdapterFadeFloor: 7 })
    .req as unknown as Record<string, unknown>;
  assert.equal(faded.lm_adapter_mode, 'runtime');
  assert.equal(faded.lm_adapter_fade_out, 150);
  assert.equal(faded.lm_adapter_fade_len, 60);
  assert.equal(faded.lm_adapter_fade_floor, undefined);  // out of range is dropped
});

test('MM3 tail fade toggle times the fade from the lyrics', (t) => {
  assert.equal(mm3TailFadeStartSec(''), null);
  assert.equal(mm3TailFadeStartSec('[Verse] ' + 'word '.repeat(336)), 212);  // the 2026-11-08 test sheet: ~210 s
  assert.equal(mm3TailFadeStartSec('word '.repeat(10)), 60);
  assert.equal(mm3TailFadeStartSec('word '.repeat(2000)), 240);
  const installed = listMm3LmAdapters()[0];
  if (!installed) return t.skip('no MM3 LM adapter installed');
  const base = { prompt: 'x', seed: 42, mm3LmAdapter: installed.file, mm3AdapterTailFade: true };
  const sung = mapMinimaxParams({ ...base, lyrics: 'word '.repeat(336) }).req as unknown as Record<string, unknown>;
  assert.equal(sung.lm_adapter_fade_out, 212);
  assert.equal(sung.lm_adapter_fade_len, 60);
  assert.equal(sung.lm_adapter_mode, 'runtime');
  const inst = mapMinimaxParams({ ...base, lyrics: '' }).req as unknown as Record<string, unknown>;
  assert.equal(inst.lm_adapter_fade_out, undefined);
});
