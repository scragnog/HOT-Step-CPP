// Tier 3 — plugins and adapters actually change the output.
//
// The failure this exists for is silent: if hot-step-sampler.h loses its hook,
// every solver, scheduler and guidance mode still "works" and every render is
// the upstream default. So each probe renders the same seed twice, default vs
// non-default, and asserts the audio differs. Same trick for adapters.
import test, { before } from 'node:test';
import assert from 'node:assert/strict';
import { MIN, api, generate, assertAudible, wavFile, correlation, setBackend, state, LYRICS } from './lib.mjs';

const base = {
  prompt: 'minimal ambient techno, warm analog pads, slow',
  instrumental: true,
  duration: 6,
  inferenceSteps: 6,
  seed: 777,
  randomSeed: false,
  batchSize: 1,
  skipLm: true,
};
const DIFFERENT = 0.99; // same seed, changed knob: renders must correlate below this

let baseline = null;
let plugins = null;
let defaults = {};

before(async () => {
  await setBackend('ace');
  plugins = await api('GET', '/api/plugins');
  const models = await api('GET', '/api/backends/models?backend=ace');
  defaults = models.defaults ?? {};
});

function pick(list, current) {
  const usable = (list ?? []).filter((p) => p.name !== current && !p.needs_model);
  return (usable.find((p) => !p.stochastic) ?? usable[0])?.name ?? null;
}

async function renderMono(label, body) {
  const g = await generate(label, body);
  assertAudible(assert, g.file, label);
  return wavFile(g.file).mono;
}

test('baseline render, default solver and scheduler', { timeout: 10 * MIN }, async () => {
  baseline = await renderMono('t3-baseline', base);
});

test('same seed reproduces the baseline', { timeout: 10 * MIN }, async (t) => {
  if (!baseline) return t.skip('no baseline');
  const again = await renderMono('t3-baseline-again', base);
  const c = correlation(baseline, again);
  state.set('deterministic', c > 0.999);
  t.diagnostic(`repeat correlation ${c.toFixed(5)} (${c > 0.999 ? 'bit-stable' : 'not bit-stable'})`);
  assert.ok(c > 0.95, `same seed gave a different render (correlation ${c.toFixed(4)}); the seed is being ignored`);
});

test('a non-default solver changes the output', { timeout: 10 * MIN }, async (t) => {
  if (!baseline) return t.skip('no baseline');
  const name = pick(plugins.solvers, defaults.infer_method);
  if (!name) return t.skip('no alternative solver registered');
  const other = await renderMono(`t3-solver-${name}`, { ...base, inferMethod: name });
  const c = correlation(baseline, other);
  t.diagnostic(`solver ${name}: correlation to baseline ${c.toFixed(4)}`);
  assert.ok(c < DIFFERENT, `solver ${name} produced the default render (correlation ${c.toFixed(4)}); solver plugins are dead`);
});

test('a non-default scheduler changes the output', { timeout: 10 * MIN }, async (t) => {
  if (!baseline) return t.skip('no baseline');
  const name = pick(plugins.schedulers, defaults.scheduler);
  if (!name) return t.skip('no alternative scheduler registered');
  const other = await renderMono(`t3-scheduler-${name}`, { ...base, scheduler: name });
  const c = correlation(baseline, other);
  t.diagnostic(`scheduler ${name}: correlation to baseline ${c.toFixed(4)}`);
  assert.ok(c < DIFFERENT, `scheduler ${name} produced the default render (correlation ${c.toFixed(4)}); scheduler plugins are dead`);
});

test('a non-default guidance mode changes the output (CFG on)', { timeout: 15 * MIN }, async (t) => {
  const name = pick(plugins.guidance, defaults.guidance_mode);
  if (!name) return t.skip('no alternative guidance mode registered');
  const cfg = { ...base, guidanceScale: 4 };
  const a = await renderMono('t3-guidance-default', cfg);
  const b = await renderMono(`t3-guidance-${name}`, { ...cfg, guidanceMode: name });
  const c = correlation(a, b);
  t.diagnostic(`guidance ${name}: correlation to default ${c.toFixed(4)}`);
  assert.ok(c < DIFFERENT, `guidance ${name} produced the default render (correlation ${c.toFixed(4)}); guidance plugins are dead`);
});

test('a DiT adapter changes the output', { timeout: 10 * MIN }, async (t) => {
  if (!baseline) return t.skip('no baseline');
  const m = await api('GET', '/api/models');
  const name = (m.adapters ?? [])[0];
  if (!name) return t.skip('no DiT adapter installed under adapters/');
  const other = await renderMono('t3-dit-adapter', { ...base, loraPath: name });
  const c = correlation(baseline, other);
  t.diagnostic(`adapter ${name}: correlation to baseline ${c.toFixed(4)}`);
  assert.ok(c < DIFFERENT, `adapter ${name} produced the base render (correlation ${c.toFixed(4)}); adapter merge is dead`);
});

test('an LM adapter loads and the generation completes', { timeout: 15 * MIN }, async (t) => {
  const r = await api('GET', '/api/adapters/lm');
  const list = Array.isArray(r) ? r : (r.adapters ?? r.lmAdapters ?? []);
  const first = list[0];
  const ref = typeof first === 'string' ? first : first?.path ?? first?.name;
  if (!ref) return t.skip('no LM adapter installed under adapters/lm/');
  const g = await generate('t3-lm-adapter', { ...base, instrumental: false, lyrics: LYRICS, skipLm: false, duration: 10, lmAdapter: ref });
  assertAudible(assert, g.file, 'lm adapter');
  t.diagnostic(`LM adapter ${ref}: ${g.seconds} s`);
});
