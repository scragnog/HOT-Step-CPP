import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { yue2ResolvePick, yue2PickFromModels, type Yue2PersistedSelection } from './index.js';
import { yue2CoalesceKey } from './generate.js';
import type { GenerationJob } from '../../generation/jobTypes.js';
import type { Yue2SynthRequest } from './client.js';

// #204: a job renders with the pick captured when it was submitted.

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-pick-'));
const fileA = path.join(dir, 'album-a-nar.safetensors');
const fileB = path.join(dir, 'album-b-nar.safetensors');
fs.writeFileSync(fileA, '');
fs.writeFileSync(fileB, '');

const dials = { global: 0.8, attn: 1, mlp: 0.5, early: 1, mid: 1, late: 1 };
const defaults: Yue2PersistedSelection = {
  lm: 'q8_0',
  vae_variant: 'standard',
  adapters: {
    ar: { path: '', scales: { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 } },
    nar: { path: fileA, scales: dials },
  },
};

test('a job with no pick of its own takes the default at submit', () => {
  const pick = yue2ResolvePick({}, defaults);
  assert.equal(pick.lm, 'q8_0');
  assert.equal(pick.adapters.nar.path, fileA);
  assert.deepEqual(pick.adapters.nar.scales, dials);
  assert.equal(pick.adapters.ar.path, '');
});

test('explicit fields are kept, omitted ones come from the default', () => {
  const pick = yue2ResolvePick({ yue2Pick: { lmAdapterNar: fileB, lmAdapterNarScale: 0.3 } }, defaults);
  assert.equal(pick.adapters.nar.path, fileB);
  assert.equal(pick.adapters.nar.scales.global, 0.3);
  assert.equal(pick.adapters.nar.scales.mlp, 0.5);  // dial not given: the default's
  assert.equal(pick.lm, 'q8_0');
});

test("an explicit '' clears the half: an album without that half renders the base model", () => {
  const pick = yue2ResolvePick({ yue2Pick: { lmAdapterNar: '' } }, defaults);
  assert.equal(pick.adapters.nar.path, '');
});

test('an unusable explicit adapter fails the submit', () => {
  assert.throws(() => yue2ResolvePick({ yue2Pick: { lmAdapterNar: path.join(dir, 'missing.safetensors') } }, defaults),
    /not usable/);
});

test('the pick survives the trip through envelope.models', () => {
  const pick = yue2ResolvePick({ yue2Pick: { lmAdapterNar: fileB, lmAdapterNarScaleAttn: 1.25 } }, defaults);
  const models: Record<string, string> = { lm: pick.lm, vae_variant: pick.vae_variant };
  for (const kind of ['ar', 'nar'] as const) {
    const slot = pick.adapters[kind];
    if (!slot.path) continue;
    models[`lm_adapter_${kind}`] = slot.path;
    models[`lm_adapter_${kind}_scale`] = String(slot.scales.global);
    models[`lm_adapter_${kind}_scale_attn`] = String(slot.scales.attn);
    models[`lm_adapter_${kind}_scale_mlp`] = String(slot.scales.mlp);
    models[`lm_adapter_${kind}_scale_early`] = String(slot.scales.early);
    models[`lm_adapter_${kind}_scale_mid`] = String(slot.scales.mid);
    models[`lm_adapter_${kind}_scale_late`] = String(slot.scales.late);
  }
  assert.deepEqual(yue2PickFromModels(models), pick);
});

test('only jobs with the same pick coalesce', () => {
  const req: Yue2SynthRequest = { style: 's', cot: 'full', vae_variant: 'standard' } as Yue2SynthRequest;
  const job = (models: Record<string, string>) =>
    ({ userId: 'u', envelope: { models } } as unknown as GenerationJob);
  const a = { lm: 'q8_0', lm_adapter_nar: fileA, lm_adapter_nar_scale: '1' };
  const b = { lm: 'q8_0', lm_adapter_nar: fileB, lm_adapter_nar_scale: '1' };
  const aScaled = { ...a, lm_adapter_nar_scale: '0.5' };
  assert.equal(yue2CoalesceKey(job(a), req), yue2CoalesceKey(job({ ...a }), req));
  assert.notEqual(yue2CoalesceKey(job(a), req), yue2CoalesceKey(job(b), req));
  assert.notEqual(yue2CoalesceKey(job(a), req), yue2CoalesceKey(job(aScaled), req));
  assert.notEqual(yue2CoalesceKey(job(a), req), yue2CoalesceKey(job({ lm: 'q8_0' }), req));
});

test('the picker reports the saved LM and VAE, not what is resident', async () => {
  const { yue2PickerLmVae } = await import('./index.js');
  const props = { variants: { lm: { selected: 'bf16' } }, files: { vae_standard: { found: true } } } as any;
  assert.deepEqual(yue2PickerLmVae({ lm: 'q8_0', vae_variant: 'legacy' }, props), { lm: 'q8_0', vae: 'legacy' });
  // Nothing saved: what the engine resolves on its own.
  assert.deepEqual(yue2PickerLmVae({ lm: '', vae_variant: '' }, props), { lm: 'bf16', vae: 'standard' });
});
