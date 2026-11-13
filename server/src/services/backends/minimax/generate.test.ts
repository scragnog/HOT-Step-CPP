import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  MM3_DEFAULT_FLOW_HOP, MM3_DEFAULT_STEPS, MM3_FP8_CALIB_PATH, mapMinimaxParams, mm3LowStepShift,
  mm3TailFadeStartSec,
} from './generate.js';
import { mm3PlankDir } from './plank.js';
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

test('MM3 sends the plan dial only below 1, and it forces runtime mode', (t) => {
  const installed = listMm3LmAdapters()[0];
  if (!installed) return t.skip('no MM3 LM adapter installed');
  const params = { prompt: 'x', seed: 42, mm3LmAdapter: installed.file, mm3LmAdapterMode: 'merge' };
  const req = (p: Record<string, unknown>) => mapMinimaxParams({ ...params, ...p }).req as unknown as Record<string, unknown>;
  assert.equal(req({ mm3LmAdapterScalePlan: 1 }).lm_adapter_plan, undefined);
  assert.equal(req({ mm3LmAdapterScalePlan: 1 }).lm_adapter_mode, 'merge');
  assert.equal(req({ mm3LmAdapterScalePlan: 0.5 }).lm_adapter_plan, 0.5);
  assert.equal(req({ mm3LmAdapterScalePlan: 0 }).lm_adapter_plan, 0);
  assert.equal(req({ mm3LmAdapterScalePlan: 0 }).lm_adapter_mode, 'runtime');
  assert.equal(req({ mm3LmAdapterScalePlan: -1 }).lm_adapter_plan, undefined);  // out of range is dropped
  const noAdapter = mapMinimaxParams({ prompt: 'x', seed: 42, mm3LmAdapterScalePlan: 0 }).req as unknown as Record<string, unknown>;
  assert.equal(noAdapter.lm_adapter_plan, undefined);
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

test('MM3 source guide: sent only from the plank folder, with its dials, and endings stand down', () => {
  const params = { prompt: 'x', seed: 42 };
  const req = (p: Record<string, unknown>) => mapMinimaxParams({ ...params, ...p }).req as unknown as Record<string, unknown>;
  const plain = req({});
  assert.equal(Object.keys(plain).some(k => k.startsWith('lm_guide')), false);
  const on = req({ mm3GuidePath: 'song.guide', mm3GuideScale: '0.9', mm3GuideK: 32, mm3GuideOutside: -100,
    mm3GuideGatePath: 'song.gate', mm3Takes: 4 });
  assert.equal(on.lm_guide_file, path.join(mm3PlankDir(), 'song.guide'));
  assert.equal(on.lm_guide_gate_file, path.join(mm3PlankDir(), 'song.gate'));
  assert.equal(on.lm_guide_scale, 0.9);
  assert.equal(on.lm_guide_k, 32);
  assert.equal(on.lm_guide_outside, -100);
  assert.equal(Object.hasOwn(on, 'require_eos'), false);
  assert.equal(Object.hasOwn(on, 'min_frames'), false);
  assert.equal(on.takes ?? 1, 1);
  const bare = req({ mm3GuidePath: 'song.guide', mm3GuideScale: '' });
  assert.equal(Object.hasOwn(bare, 'lm_guide_scale'), false);
  assert.equal(Object.hasOwn(req({ mm3GuidePath: '../escape.guide' }), 'lm_guide_file'), false);
});

test('MM3 cover modes: state fix, product rule, guide trim and the state blend reach the engine', () => {
  const req = (p: Record<string, unknown>) => mapMinimaxParams({ prompt: 'x', seed: 42, ...p }).req as unknown as Record<string, unknown>;
  const guided = req({ mm3GuidePath: 'a.guide', mm3GuideStateFix: 2, mm3GuideProduct: '', mm3GuideSeconds: 30 });
  assert.equal(guided.lm_state_fix, 2);
  assert.equal(Object.hasOwn(guided, 'lm_guide_product'), false);
  assert.equal(guided.lm_guide_seconds, 30);
  assert.equal(Object.hasOwn(req({ mm3GuidePath: 'a.guide', mm3GuideStateFix: 0 }), 'lm_state_fix'), false);
  assert.equal(req({ mm3GuidePath: 'a.guide', mm3GuideProduct: 1 }).lm_guide_product, 1);
  const blend = req({ mm3GuidePath: 'a.guide', mm3BlendPath: 'a.mm3hiddens', mm3BlendAmount: 0.25 });
  assert.equal(blend.lm_blend_file, path.join(mm3PlankDir(), 'a.mm3hiddens'));
  assert.equal(blend.lm_blend_amount, 0.25);
  assert.equal(Object.hasOwn(req({ mm3BlendPath: 'a.mm3hiddens', mm3BlendAmount: 0 }), 'lm_blend_file'), false);
  assert.equal(Object.hasOwn(req({ mm3BlendPath: '../x.mm3hiddens', mm3BlendAmount: 0.5 }), 'lm_blend_file'), false);
});
