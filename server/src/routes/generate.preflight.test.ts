// What this guards: POST /api/generate/preflight (preflightGeneration). One engine
// fixture per verdict, the backend without an admission model, engine failure, and a
// body generate would refuse. Engine calls go through a fetch stub; nothing reaches
// a live engine and nothing is queued.
// Run: npx tsx --test src/routes/generate.preflight.test.ts
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

process.env.DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'route-preflight-'));
const database = await import('../db/database.js');
database.initDb();
const { setEngineReady } = await import('../engineState.js');
const { preflightGeneration } = await import('./generate.js');
const { gpuLaneBusy } = await import('../services/generation/gpuLane.js');

const calls: { route: string; body: any }[] = [];
let reply: (route: string) => Response = () => { throw new Error('test: no engine'); };
const realFetch = globalThis.fetch;
globalThis.fetch = (async (input: any, init?: RequestInit) => {
  const url = String(input instanceof Request ? input.url : input);
  const m = /^https?:\/\/(?:127\.0\.0\.1|localhost):8085(\/[^?]*(?:\?.*)?)$/.exec(url);
  if (!m) return realFetch(input, init);
  calls.push({ route: m[1], body: init?.body ? JSON.parse(String(init.body)) : undefined });
  return reply(m[1]);
}) as typeof fetch;
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const GiB = 2 ** 30;

setEngineReady(true, 'test');
const heartmula = { backendOverride: 'heartmula', tags: 'pop, piano', lyrics: '[verse]\nhello', duration: 60 };
const mm3 = { backendOverride: 'minimax-m3', caption: 'synth pop, female vocal', lyrics: '', mm3DitBackend: 'ggml' };

test.beforeEach(() => { calls.length = 0; });

test('HeartMuLa fits: the mapped wire goes to /heartmula/preflight', async () => {
  reply = () => json({ verdict: 'fits', need_bytes: 10.5 * GiB, free_bytes: 20 * GiB, reclaimable_bytes: 0, phase: 'lm', detail: '' });
  const r = await preflightGeneration(heartmula, 'user-1');
  assert.deepEqual(r, { ok: true, preflight: { verdict: 'fits', estimateGb: 10.5, freeGb: 20 } });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].route, '/heartmula/preflight');
  assert.equal(calls[0].body.tags, 'pop, piano');
  assert.equal(calls[0].body.duration_s, 60);
  assert.equal(gpuLaneBusy(), false, 'nothing queued');
});

test('HeartMuLa passes: detail kept', async () => {
  reply = () => json({ verdict: 'passes', need_bytes: 12 * GiB, free_bytes: 8 * GiB, reclaimable_bytes: 9 * GiB, phase: 'lm',
    detail: 'idle models of other backends are unloaded first' });
  const r = await preflightGeneration(heartmula, 'user-1');
  assert.deepEqual(r, { ok: true, preflight: { verdict: 'passes', estimateGb: 12, freeGb: 8,
    detail: 'idle models of other backends are unloaded first' } });
});

test('MM3 refused: the /mm3/synth body goes to /mm3/preflight', async () => {
  reply = () => json({ verdict: 'refused', need_bytes: 30 * GiB, free_bytes: 20 * GiB, budget_bytes: 22 * GiB,
    detail: 'planning does not fit even with the planner cache at Q8_0' });
  const r = await preflightGeneration(mm3, 'user-1');
  assert.deepEqual(r, { ok: true, preflight: { verdict: 'refused', estimateGb: 30, freeGb: 20,
    detail: 'planning does not fit even with the planner cache at Q8_0' } });
  assert.deepEqual(calls.map(c => c.route), ['/mm3/preflight']);
  assert.equal(calls[0].body.caption, 'synth pop, female vocal');
  assert.equal(calls[0].body.dit_backend, 'ggml');
});

const { mm3DitBackend: _, ...mm3Auto } = mm3;

test('MM3 default renderer, cold props cache: unknown, and no engine call', async () => {
  // Runs before anything fills the cache. Guessing ggml would under-budget a
  // TensorRT render by 1 GiB under keep-loaded.
  const r = await preflightGeneration(mm3Auto, 'user-1');
  assert.deepEqual(r, { ok: true, preflight: { verdict: 'unknown', detail: 'the MM3 renderer is not known yet; ask again in a moment' } });
  assert.equal(calls.length, 0);
});

test('MM3 unknown (GPU busy), default renderer: resolves from the cache, never GET /mm3/props', async () => {
  const { mm3Props } = await import('../services/backends/minimax/client.js');
  reply = () => json({ dit_runtime: { available: true } });
  await mm3Props();  // fills the last-good cache: TensorRT available
  calls.length = 0;
  // /mm3/props waits on a running render's mutex; auto must resolve without it.
  reply = route => route === '/mm3/preflight'
    ? json({ verdict: 'unknown', need_bytes: 0, free_bytes: 0, budget_bytes: 0, detail: 'the GPU is busy with another job' })
    : new Promise<Response>(() => {}) as unknown as Response;
  const r = await preflightGeneration(mm3Auto, 'user-1');
  assert.deepEqual(r, { ok: true, preflight: { verdict: 'unknown', detail: 'the GPU is busy with another job' } });
  assert.deepEqual(calls.map(c => c.route), ['/mm3/preflight']);
  assert.equal(calls[0].body.dit_backend, 'tensorrt');
});

test('a backend without an admission model is unknown and never asks the engine', async () => {
  const r = await preflightGeneration({ backendOverride: 'ace', caption: 'rock' }, 'user-1');
  assert.equal(r.ok, true);
  assert.equal(r.ok && r.preflight.verdict, 'unknown');
  assert.match(r.ok ? r.preflight.detail ?? '' : '', /no VRAM admission model/);
  assert.equal(calls.length, 0);
});

test('an engine error is unknown, not a failed request', async () => {
  reply = () => json({ error: 'HeartMuLa is not ready', code: 'unavailable' }, 503);
  const r = await preflightGeneration(heartmula, 'user-1');
  assert.deepEqual(r, { ok: true, preflight: { verdict: 'unknown', detail: 'HeartMuLa is not ready' } });
});

test('a body generate would refuse gets the same status, and no engine call', async () => {
  const bad = await preflightGeneration({ backendOverride: 'heartmula', tags: '' }, 'user-1');
  assert.equal(bad.ok, false);
  assert.equal(!bad.ok && bad.status, 400);
  const anon = await preflightGeneration(heartmula, null);
  assert.equal(!anon.ok && anon.status, 401);
  assert.equal(calls.length, 0);
});
