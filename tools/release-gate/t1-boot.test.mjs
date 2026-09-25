// Tier 1 — the app is up, the engine is ready, and every read-only surface answers.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { MIN, REPO, api, apiRaw, health, waitEngineReady, state } from './lib.mjs';

test('health: server ok, engine ready', { timeout: 15 * MIN }, async () => {
  const h = await waitEngineReady();
  assert.equal(h.status, 'ok');
  assert.equal(h.engine?.ready, true, `engine.ready=false (${h.engine?.bootStatus})`);
});

test('backends registered: ace, minimax-m3, yue2; one active', async () => {
  const r = await api('GET', '/api/backends');
  const ids = (r.backends ?? []).map((b) => b.id);
  for (const want of ['ace', 'minimax-m3', 'yue2']) assert.ok(ids.includes(want), `backend ${want} missing from ${ids.join(', ')}`);
  assert.ok(ids.includes(r.activeId), `activeId ${r.activeId} is not a registered backend`);
  state.set('originalBackend', r.activeId);
});

test('ACE model catalogue has a DiT, a VAE and an LM', async () => {
  const m = await api('GET', '/api/models');
  const b = m.buckets ?? m.models ?? {};
  for (const k of ['dit', 'vae', 'lm']) assert.ok(Array.isArray(b[k]) && b[k].length, `no ${k} models listed`);
});

test('plugin registry lists every solver, scheduler and guidance file in engine/plugins', async () => {
  const reg = await api('GET', '/api/plugins');
  const dirs = { solvers: 'solvers', schedulers: 'schedulers', guidance: 'guidance' };
  const missing = [];
  for (const [key, dir] of Object.entries(dirs)) {
    const names = new Set((reg[key] ?? []).map((p) => p.name));
    const folder = path.join(REPO, 'engine', 'plugins', dir);
    if (!fs.existsSync(folder)) continue;
    for (const f of fs.readdirSync(folder)) {
      if (!f.endsWith('.lua') || f.startsWith('_')) continue;
      const stem = f.slice(0, -4);
      if (!names.has(stem)) missing.push(`${dir}/${f}`);
    }
    assert.ok((reg[key] ?? []).length > 0, `registry has no ${key}`);
  }
  assert.deepEqual(missing, [], `plugins on disk but not registered by the engine: ${missing.join(', ')}`);
});

test('backend capabilities answer', async () => {
  const c = await api('GET', '/api/capabilities');
  assert.ok(c && typeof c === 'object');
});

test('training capabilities answer', async () => {
  const c = await api('GET', '/api/training/capabilities');
  for (const k of ['engine', 'preprocess', 'trainLm', 'trainDit']) assert.ok(c[k], `capabilities.${k} missing`);
  state.set('trainingCaps', c);
});

const READ_ONLY = [
  '/api/health',
  '/api/models',
  '/api/models/pp-vae',
  '/api/models/stablestep',
  '/api/plugins',
  '/api/backends',
  '/api/backends/models',
  '/api/settings',
  '/api/songs',
  '/api/songs/recent',
  '/api/seeds',
  '/api/profiles',
  '/api/adapters/lm',
  '/api/mastering/references',
  '/api/midi-studio/status',
  '/api/midi-studio/jobs',
  '/api/stem-studio/jobs',
  '/api/stem-studio/stats',
  '/api/assistant/providers',
  '/api/inspire/llm/providers',
  '/api/inspire/llm/prompt',
  '/api/lireek/prompts',
  '/api/lireek/recent-songs',
  '/api/training/datasets',
  '/api/training/defaults',
  '/api/training/active-models',
  '/api/training/jobs',
  '/api/mm3/lm-adapters',
  '/api/mm3/planks',
  '/api/mm3/plans',
  '/api/generate/queue',
  '/api/logs?after=-1',
  '/api/logs/vram',
  '/api/logs/models-loaded',
];

test('read-only endpoints answer 200', async (t) => {
  for (const route of READ_ONLY) {
    await t.test(route, async () => {
      const r = await apiRaw('GET', route, undefined, { auth: true, timeoutMs: 30_000 });
      assert.equal(r.status, 200, `${route} -> ${r.status}: ${typeof r.body === 'object' ? JSON.stringify(r.body).slice(0, 200) : String(r.body).slice(0, 200)}`);
    });
  }
});

test('LLM providers report configured state (informational)', async (t) => {
  const p = await api('GET', '/api/assistant/providers');
  const list = Array.isArray(p) ? p : (p.providers ?? []);
  const configured = list.filter((x) => x.configured ?? x.available).map((x) => x.id ?? x.name);
  t.diagnostic(configured.length ? `configured: ${configured.join(', ')}` : 'no LLM provider configured; Lyric Studio, assistant and cover art are not exercised');
  const h = await health();
  assert.equal(h.engine?.ready, true, 'engine went away during tier 1');
});
