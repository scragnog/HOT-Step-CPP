import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  yue2ResolvePick, yue2PickFromModels, yue2CoverFromSubmission, yue2CoverPickBase,
  type Yue2PersistedSelection,
} from './index.js';
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

test('scheduled jobs coalesce only with the identical prompt', () => {
  const job = { userId: 'u', envelope: { models: { lm: 'q8_0' } } } as unknown as GenerationJob;
  const lyric_schedule = { mode: 'bias' as const, bias: -4, sections: [{ start_sec: 1, lyric: [0, 8] as [number, number] }] };
  const base = { style: 's', cot: 'full', vae_variant: 'standard', abc: 'X:1', lyric_schedule } as Yue2SynthRequest;
  const key = (r: Partial<Yue2SynthRequest>) => yue2CoalesceKey(job, { ...base, ...r });
  // Same length, different words: equal spans, different token rows.
  assert.notEqual(key({ lyrics: '[Verse]\nfirst line\n\n[Chorus]\nsecond line' }),
    key({ lyrics: '[Verse]\nother text\n\n[Chorus]\nthird words' }));
  assert.notEqual(key({ lyrics: 'x', style: 's' }), key({ lyrics: 'x', style: 't' }));
  assert.notEqual(key({ lyrics: 'x', abc: 'X:1' }), key({ lyrics: 'x', abc: 'X:2' }));
  assert.equal(key({ lyrics: 'x', seed: 1 }), key({ lyrics: 'x', seed: 2 }));
  // Unscheduled jobs still share a batch across prompts.
  const plain = { ...base, lyric_schedule: undefined };
  assert.equal(yue2CoalesceKey(job, { ...plain, lyrics: 'a' }), yue2CoalesceKey(job, { ...plain, lyrics: 'b', style: 't' }));
});

test('the picker reports the saved LM and VAE, not what is resident', async () => {
  const { yue2PickerLmVae } = await import('./index.js');
  const props = { variants: { lm: { selected: 'bf16' } }, files: { vae_standard: { found: true } } } as any;
  assert.deepEqual(yue2PickerLmVae({ lm: 'q8_0', vae_variant: 'legacy' }, props), { lm: 'q8_0', vae: 'legacy' });
  // Nothing saved: what the engine resolves on its own.
  assert.deepEqual(yue2PickerLmVae({ lm: '', vae_variant: '' }, props), { lm: 'bf16', vae: 'standard' });
});

// Every pick field reaches the engine request from the captured pick, never
// from a top-level param or the live picker.
test('the captured VAE reaches the request', async () => {
  const { mapYue2Params } = await import('./generate.js');
  const pick = { ...defaults, vae_variant: 'legacy' };
  assert.equal(mapYue2Params({ caption: 'folk' }, pick).req.vae_variant, 'legacy');
  // The per-request VAE select is folded into the pick at submit.
  assert.equal(yue2ResolvePick({ yue2VaeVariant: 'legacy' }, defaults).vae_variant, 'legacy');
  assert.equal(yue2ResolvePick({ yue2VaeVariant: 'legacy', yue2Pick: { vae: 'standard' } }, defaults).vae_variant, 'standard');
});

test('the captured lm_type reaches the request', async () => {
  const { mapYue2Params } = await import('./generate.js');
  assert.equal(mapYue2Params({ caption: 'folk' }, { ...defaults, lm: 'Q6_K' }).req.lm_type, 'Q6_K');
});

test('the captured adapters reach the request', async () => {
  const { mapYue2Params } = await import('./generate.js');
  const pick = yue2ResolvePick({ yue2Pick: { lmAdapterNar: fileB, lmAdapterNarScale: 0.3 } }, defaults);
  const wire = mapYue2Params({ caption: 'folk' }, pick).req.lm_adapter;
  assert.deepEqual(wire?.map(a => [a.path, a.scale]), [[fileB, 0.3]]);
  const none = yue2ResolvePick({ yue2Pick: { lmAdapterAr: '', lmAdapterNar: '' } }, defaults);
  assert.equal(mapYue2Params({ caption: 'folk' }, none).req.lm_adapter, null);
});

// ── Cover submission (S3): yue2Cover marker, rejected at submit, provenance
// captured into the envelope — never a silent plain generation. ──

test('yue2CoverFromSubmission requires a sourceId', () => {
  assert.equal(yue2CoverFromSubmission({}), undefined);
  assert.throws(() => yue2CoverFromSubmission({ yue2Cover: {} }), /sourceId/);
  assert.deepEqual(yue2CoverFromSubmission({ yue2Cover: { sourceId: 'song-1' } }), { sourceId: 'song-1' });
  assert.deepEqual(
    yue2CoverFromSubmission({ yue2Cover: { sourceId: 'song-1', sourceLabel: 'My Take' } }),
    { sourceId: 'song-1', sourceLabel: 'My Take' },
  );
});

test('yue2CoverFromSubmission keeps a render chord choice and rejects invalid values', () => {
  assert.deepEqual(yue2CoverFromSubmission({ yue2Cover: { sourceId: 'song-1', keepChords: false } }),
    { sourceId: 'song-1', keepChords: false });
  assert.throws(() => yue2CoverFromSubmission({ yue2Cover: { sourceId: 'song-1', keepChords: 'no' } }),
    /keepChords must be a boolean/);
});

test('an invalid keepChords value fails the generation submit', async () => {
  const { yue2Backend } = await import('./index.js');
  assert.throws(() => yue2Backend.resolveRequest({
    caption: 'c', yue2Cover: { sourceId: 'song-1', keepChords: 'no' }, yue2Abc: 'X:1\nK:C\nC|',
  }), /keepChords must be a boolean/);
});

// A malformed truthy yue2Cover must fail the submit, never be treated as
// "no marker" — that would let a malformed cover slip through as a plain
// generation with its score silently dropped.
test('yue2CoverFromSubmission rejects a truthy marker of the wrong shape instead of ignoring it', () => {
  assert.throws(() => yue2CoverFromSubmission({ yue2Cover: true }), /object/);
  assert.throws(() => yue2CoverFromSubmission({ yue2Cover: 'song-1' }), /object/);
  assert.throws(() => yue2CoverFromSubmission({ yue2Cover: [] }), /object/);
  assert.throws(() => yue2CoverFromSubmission({ yue2Cover: 0 }), /object/);
  // null and absent both mean "not a cover" — no marker to validate.
  assert.equal(yue2CoverFromSubmission({ yue2Cover: null }), undefined);
  assert.equal(yue2CoverFromSubmission({}), undefined);
});

test('a malformed yue2Cover is rejected at submit, not silently rendered as a plain generation', async () => {
  const { yue2Backend } = await import('./index.js');
  assert.throws(
    () => yue2Backend.resolveRequest({ caption: 'c', yue2Cover: true, yue2Abc: '', yue2Cot: 'off' }),
    /object/,
  );
});

test('a cover submission with cot=off is rejected at submit, not silently rendered plain', async () => {
  const { yue2Backend } = await import('./index.js');
  assert.throws(
    () => yue2Backend.resolveRequest({ caption: 'c', yue2Cover: { sourceId: 's' }, yue2Abc: 'X:1\n', yue2Cot: 'off' }),
    /Chain of Thought/,
  );
});

test('a cover submission with a blank ABC is rejected at submit', async () => {
  const { yue2Backend } = await import('./index.js');
  assert.throws(
    () => yue2Backend.resolveRequest({ caption: 'c', yue2Cover: { sourceId: 's' }, yue2Abc: '  ' }),
    /approved lead sheet/,
  );
  assert.throws(
    () => yue2Backend.resolveRequest({ caption: 'c', yue2Cover: { sourceId: 's' } }),
    /approved lead sheet/,
  );
});

test('a cover submission rejects a silent Vocal score', async () => {
  const { yue2Backend } = await import('./index.js');
  assert.throws(() => yue2Backend.resolveRequest({ caption: 'c',
    yue2Cover: { sourceId: 's' }, yue2Abc: 'X:1\nV: Vocal\nZ4|z4|\nV: Ins\nC4|' }),
  /The transcriber heard no melody in this source/);
});

test('cover submission captures both voices by default and preserves explicit vocal-only', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-cover-submit-'));
  try {
    const script = [
      "import assert from 'node:assert/strict';",
      "import { initDb } from './src/db/database.js'; import { yue2Backend } from './src/services/backends/yue2/index.js';",
      "initDb(); const resolved = yue2Backend.resolveRequest({ caption: 'c', yue2Cover: { sourceId: 's' }, yue2Abc: 'X:1\\nV: Vocal\\nz4|C4|\\nV: Ins\\nC4|' });",
      "assert.ok(resolved.options.yue2Abc.includes('C4|'));",
      "assert.equal(resolved.options.yue2Cover.voices, 'both');",
      "assert.ok(resolved.options.yue2Abc.includes('V: Ins'));",
      "const vocal = yue2Backend.resolveRequest({ caption: 'c', yue2Cover: { sourceId: 's', voices: 'vocal' }, yue2Abc: 'X:1\\nV: Vocal\\nz4|C4|\\nV: Ins\\nC4|' });",
      "assert.equal(vocal.options.yue2Cover.voices, 'vocal');",
      "assert.equal(vocal.options.yue2Abc.includes('V: Ins'), false);",
    ].join('\n');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../../', import.meta.url)),
      env: { ...process.env, DATA_DIR: path.join(root, 'data'), TRAINING_DIR: path.join(root, 'training') }, stdio: 'pipe',
    });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('invalid cover score choices are rejected at the submit boundary', () => {
  for (const choice of [{ voices: 'ins' }, { keepChords: 'no' }, { tempo: 301 },
    { key: '' }, { cfgScale: 0 }]) {
    assert.throws(() => yue2CoverFromSubmission({ yue2Cover: { sourceId: 's', ...choice } }));
  }
});

// resolveRequest itself also calls yue2PersistedSelection() (reads real
// settings), which needs an initialized DB this file deliberately avoids —
// same reason every other test here injects `defaults` instead of touching
// settings. So the base-mode-clearing contract is tested through the same
// pure pieces resolveRequest is built from: yue2CoverPickBase (DB-free) feeds
// yue2ResolvePick exactly as resolveRequest wires them.

test('yue2CoverPickBase clears both slots regardless of the persisted pick', () => {
  const base = yue2CoverPickBase(defaults);
  assert.equal(base.adapters.ar.path, '');
  assert.equal(base.adapters.nar.path, '');
  assert.equal(base.lm, defaults.lm);           // LM/VAE defaults still carry through
  assert.equal(base.vae_variant, defaults.vae_variant);
});

test('base-mode cover (no explicit pair) clears both AR and NAR slots, unlike an ordinary job', () => {
  const pick = yue2ResolvePick({}, yue2CoverPickBase(defaults));
  assert.equal(pick.adapters.ar.path, '');
  assert.equal(pick.adapters.nar.path, '');
  // The same empty submission against the ordinary (non-cover) base inherits
  // the persisted pair — this is what cover mode deliberately does not do.
  assert.equal(yue2ResolvePick({}, defaults).adapters.nar.path, fileA);
});

test('a pair explicitly chosen on a cover submission still renders, base-mode clearing does not override it', () => {
  const pick = yue2ResolvePick({ yue2Pick: { lmAdapterNar: fileA } }, yue2CoverPickBase(defaults));
  assert.equal(pick.adapters.nar.path, fileA);
});
