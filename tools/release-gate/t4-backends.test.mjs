// Tier 4 — the other generation backends: MiniMax-Music3 and YuE2.
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { MIN, api, generate, assertAudible, setBackend, activeBackend, backendModels, LYRICS, MM3_CAPTION } from './lib.mjs';

let original = 'ace';
before(async () => { original = await activeBackend(); });
after(async () => { await setBackend(original); });

const SEED = 99;

test('MiniMax-Music3: text2music with lyrics', { timeout: 25 * MIN }, async (t) => {
  const m = await backendModels('minimax-m3', ['lm', 'dit']);
  if (!m.ok) return t.skip(`MM3 models not installed (missing: ${m.missing.join(', ')})`);
  await setBackend('minimax-m3');
  const g = await generate('t4-mm3', {
    prompt: MM3_CAPTION, lyrics: LYRICS, instrumental: false,
    duration: 30, mm3Steps: 8, seed: SEED, randomSeed: false, batchSize: 1,
  }, { timeoutMs: 25 * MIN });
  assertAudible(assert, g.file, 'mm3');
  t.diagnostic(`${g.seconds} s`);
});

test('MiniMax-Music3: with a planner-LM adapter', { timeout: 25 * MIN }, async (t) => {
  const m = await backendModels('minimax-m3', ['lm', 'dit']);
  if (!m.ok) return t.skip(`MM3 models not installed (missing: ${m.missing.join(', ')})`);
  const r = await api('GET', '/api/mm3/lm-adapters');
  const first = (r.adapters ?? [])[0];
  const ref = typeof first === 'string' ? first : first?.path ?? first?.name ?? first?.id;
  if (!ref) return t.skip('no MM3 LM adapter installed');
  await setBackend('minimax-m3');
  const g = await generate('t4-mm3-adapter', {
    prompt: MM3_CAPTION, lyrics: LYRICS, instrumental: false,
    duration: 30, mm3Steps: 8, seed: SEED, randomSeed: false, batchSize: 1, mm3LmAdapter: ref,
  }, { timeoutMs: 25 * MIN });
  assertAudible(assert, g.file, 'mm3 adapter');
  t.diagnostic(`adapter ${ref}: ${g.seconds} s`);
});

test('YuE2: text2music with lyrics', { timeout: 30 * MIN }, async (t) => {
  const m = await backendModels('yue2');
  const buckets = m.buckets ?? {};
  const any = Object.values(buckets).some((b) => Array.isArray(b) && b.length);
  if (!any) return t.skip('YuE2 models not installed');
  await setBackend('yue2');
  const g = await generate('t4-yue2', {
    prompt: 'warm indie pop, acoustic guitar, soft brushed drums, female vocal', lyrics: LYRICS, instrumental: false,
    duration: 30, yue2OdeSteps: 8, seed: SEED, randomSeed: false, batchSize: 1,
  }, { timeoutMs: 30 * MIN });
  assertAudible(assert, g.file, 'yue2');
  t.diagnostic(`${g.seconds} s`);
});
