import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  MM3_DEFAULT_FLOW_HOP, MM3_DEFAULT_STEPS, MM3_FP8_CALIB_PATH, mapMinimaxParams, minimaxStageText, mm3LowStepShift,
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

test('MM3 ignores the retired adapter fade params from saved presets', (t) => {
  const installed = listMm3LmAdapters()[0];
  if (!installed) return t.skip('no MM3 LM adapter installed');
  const params = { prompt: 'x', seed: 42, lyrics: 'word '.repeat(336), mm3LmAdapter: installed.file, mm3LmAdapterMode: 'merge' };
  const plain = mapMinimaxParams(params).req as unknown as Record<string, unknown>;
  const stale = mapMinimaxParams({ ...params, mm3AdapterTailFade: true, mm3LmAdapterFadeIn: 10,
    mm3LmAdapterFadeOut: 150, mm3LmAdapterFadeLen: 60, mm3LmAdapterFadeFloor: 0.5 }).req as unknown as Record<string, unknown>;
  assert.deepEqual(stale, plain);
  assert.equal(stale.lm_adapter_mode, 'merge');
  assert.equal(Object.keys(stale).some((k) => k.startsWith('lm_adapter_fade')), false);
});

test('MM3 base handoff rides Require Natural Ending, only with an LM adapter', (t) => {
  const plain = mapMinimaxParams({ prompt: 'x', seed: 42 }).req as unknown as Record<string, unknown>;
  assert.equal(Object.hasOwn(plain, 'eos_handoff'), false);  // a base render is untouched
  const installed = listMm3LmAdapters()[0];
  if (!installed) return t.skip('no MM3 LM adapter installed');
  const params = { prompt: 'x', seed: 42, mm3LmAdapter: installed.file };
  const req = (p: Record<string, unknown>) => mapMinimaxParams({ ...params, ...p }).req as unknown as Record<string, unknown>;
  assert.equal(req({}).eos_handoff, true);
  assert.equal(req({}).require_eos, true);
  const off = req({ mm3BaseHandoff: false });
  assert.equal(Object.hasOwn(off, 'eos_handoff'), false);
  assert.deepEqual({ ...req({}), eos_handoff: undefined }, { ...off, eos_handoff: undefined });
  assert.equal(Object.hasOwn(req({ mm3RequireEnding: false }), 'eos_handoff'), false);
  assert.equal(Object.hasOwn(req({ mm3LmAdapter: 'no-such-run/adapter_model.safetensors' }), 'eos_handoff'), false);
});

test('MM3 base handoff progress is labelled as the base ending, not another planning round', () => {
  const d = { stage: 'handoff', window: 2, n_windows: 4, step: 100, n_steps: 7500, require_eos: true, takes: 4 } as any;
  const t = minimaxStageText(d, 0).stage;
  assert.match(t, /base model ending take 3 of 4/);
  assert.doesNotMatch(t, /planning/);
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

test('MM3 plan dial: more than 2 takes ride the q8 planner cache, otherwise 2 as before', (t) => {
  const installed = listMm3LmAdapters()[0];
  if (!installed) return t.skip('no MM3 LM adapter installed');
  const params = { prompt: 'x', seed: 42, mm3LmAdapter: installed.file };
  const req = (p: Record<string, unknown>) => mapMinimaxParams({ ...params, ...p }).req as unknown as Record<string, unknown>;
  const four = req({ mm3LmAdapterScalePlan: 0.25, mm3Takes: 4 });
  assert.equal(four.takes, 4);
  assert.equal(four.lm_kv_q8, true);
  const off = req({ mm3LmAdapterScalePlan: 0.25, mm3Takes: 4, mm3KvQ8: false });
  assert.equal(off.takes, 2);
  assert.equal(Object.hasOwn(off, 'lm_kv_q8'), false);
  // The ending check's own 3 candidates stay at 2 and f16 when one song was asked.
  const one = req({ mm3LmAdapterScalePlan: 0.25 });
  assert.equal(one.takes, 2);
  assert.equal(Object.hasOwn(one, 'lm_kv_q8'), false);
  // Plan 1: no base rows, nothing changes.
  const plain = req({ mm3LmAdapterScalePlan: 1, mm3Takes: 4 });
  assert.equal(plain.takes, 4);
  assert.equal(Object.hasOwn(plain, 'lm_kv_q8'), false);
  assert.equal(req({ mm3LmAdapterScalePlan: 1 }).takes, 3);
  // A missing adapter renders the base model: no base rows, so no q8 cache.
  const missing = req({ mm3LmAdapter: 'no-such-run/adapter_model.safetensors', mm3LmAdapterScalePlan: 0.25, mm3Takes: 4 });
  assert.equal(missing.takes, 4);
  assert.equal(Object.hasOwn(missing, 'lm_kv_q8'), false);
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
