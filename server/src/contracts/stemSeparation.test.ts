import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// stemStudio.ts resolves its stems dir from config.data.dir at import time.
const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'stem-separation-'));
process.env.DATA_DIR = DATA_DIR;

const express = (await import('express')).default;
const stemStudioRoutes = (await import('../routes/stemStudio.js')).default;
const supersepRoutes = (await import('../routes/supersep.js')).default;
const { STEM_TRACK_NAMES, stemExtractRequestSchema, stemSupersepRequestSchema, supersepSeparateRequestSchema, supersepRecombineRequestSchema } = await import('./stemSeparation.js');

const app = express();
app.use(express.json());
app.use('/api/stem-studio', stemStudioRoutes);
app.use('/api/supersep', supersepRoutes);
const server = app.listen(0);
await new Promise(resolve => server.once('listening', resolve));
const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
test.after(() => new Promise(resolve => server.close(() => { fs.rmSync(DATA_DIR, { recursive: true, force: true }); resolve(undefined); })));

const post = async (p: string, body: unknown) => {
  const res = await fetch(`${base}${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return { status: res.status, body: await res.json() as Record<string, any> };
};
const get = async (p: string) => {
  const res = await fetch(`${base}${p}`);
  return { status: res.status, body: await res.json() as Record<string, any> };
};

// ── Fixture schema parity: the route rejects exactly what the schema rejects ──

test('stemExtractRequestSchema: sourceAudioUrl and a non-empty tracks array are required; everything else is pass-through, including null', () => {
  assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav', tracks: ['vocals'] }).success, true);
  assert.equal(stemExtractRequestSchema.safeParse({ tracks: ['vocals'] }).success, false);
  assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav', tracks: [] }).success, false);
  assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav' }).success, false);
  // Old behavior: sourceFileName/style/lyrics/ditSettings are only ever used
  // behind `|| <default>`, so null (and any other falsy value) must pass.
  assert.equal(stemExtractRequestSchema.safeParse({
    sourceAudioUrl: '/references/a.wav', tracks: ['vocals'],
    sourceFileName: null, style: null, lyrics: null, ditSettings: null,
  }).success, true);
  // A non-string track entry is not a schema rejection — it reaches the
  // route's own VALID_TRACKS filter, same as it always did.
  assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav', tracks: ['vocals', 5] }).success, true);
  // Old guard was a bare `if (!sourceAudioUrl)` — any truthy value passes,
  // not just a non-empty string.
  for (const truthy of [1, true, {}, [1]]) {
    assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: truthy, tracks: ['vocals'] }).success, true, JSON.stringify(truthy));
  }
  for (const falsy of [0, false, '', null, undefined]) {
    assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: falsy, tracks: ['vocals'] }).success, false, JSON.stringify(falsy));
  }
});

test('stemSupersepRequestSchema: sourceAudioUrl required (truthy, not string), sourceFileName/level pass through including null', () => {
  assert.equal(stemSupersepRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav' }).success, true);
  assert.equal(stemSupersepRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav', level: 2 }).success, true);
  assert.equal(stemSupersepRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav', sourceFileName: null, level: null }).success, true);
  assert.equal(stemSupersepRequestSchema.safeParse({}).success, false);
  assert.equal(stemSupersepRequestSchema.safeParse({ sourceAudioUrl: 1 }).success, true, 'old guard was truthiness, not a string type check');
  assert.equal(stemSupersepRequestSchema.safeParse({ sourceAudioUrl: 0 }).success, false);
});

test('supersepSeparateRequestSchema: audioUrl required', () => {
  assert.equal(supersepSeparateRequestSchema.safeParse({ audioUrl: '/references/a.wav' }).success, true);
  assert.equal(supersepSeparateRequestSchema.safeParse({}).success, false);
});

test('supersepRecombineRequestSchema: id and stems[].index required; volume/muted optional', () => {
  assert.equal(supersepRecombineRequestSchema.safeParse({ id: 'job-1', stems: [{ index: 0, volume: 0.8, muted: false }] }).success, true);
  assert.equal(supersepRecombineRequestSchema.safeParse({ id: 'job-1' }).success, true, 'stems is optional, matching the engine silently defaulting it');
  assert.equal(supersepRecombineRequestSchema.safeParse({ stems: [{ index: 0 }] }).success, false, 'id is required, matching the engine 400');
  assert.equal(supersepRecombineRequestSchema.safeParse({ id: 'job-1', stems: [{}] }).success, false);
});

// ── HTTP: stemStudio.ts input validation (synchronous; no engine call reached) ─

test('POST /extract: missing source, missing/empty tracks, and unknown track names are all refused before any job is created', async () => {
  assert.deepEqual((await post('/api/stem-studio/extract', { tracks: ['vocals'] })).body, { error: 'sourceAudioUrl is required' });
  assert.deepEqual((await post('/api/stem-studio/extract', { sourceAudioUrl: '/references/a.wav' })).body, { error: 'tracks must be a non-empty array' });
  assert.deepEqual((await post('/api/stem-studio/extract', { sourceAudioUrl: '/references/a.wav', tracks: [] })).body, { error: 'tracks must be a non-empty array' });
  const bad = await post('/api/stem-studio/extract', { sourceAudioUrl: '/references/a.wav', tracks: ['vocals', 'kazoo'] });
  assert.deepEqual(bad.body, { error: 'Invalid track names: kazoo' });
  assert.equal(bad.status, 400);
});

test('POST /extract: null optional fields and a non-string track entry behave exactly as before (old guards, not new rejections)', async () => {
  // null sourceFileName/style/lyrics/ditSettings must reach the pipeline
  // (not 400 here) — accepted via the route's own `|| <default>` fallback.
  // 202-style jobs run async, so just confirm the request itself is admitted.
  const ok = await post('/api/stem-studio/extract', {
    sourceAudioUrl: '/references/a.wav', tracks: ['vocals'],
    sourceFileName: null, style: null, lyrics: null, ditSettings: null,
  });
  assert.equal(ok.status, 200);
  assert.match(ok.body.id, /^[0-9a-f-]{36}$/);
  // A truthy non-string sourceAudioUrl is accepted by the old guard; with an
  // empty tracks array this must still report the tracks error, matching
  // the old code's check order (sourceAudioUrl first, then tracks).
  const truthySource = await post('/api/stem-studio/extract', { sourceAudioUrl: 1, tracks: [] });
  assert.deepEqual(truthySource.body, { error: 'tracks must be a non-empty array' });
  // A non-string entry fails the VALID_TRACKS filter (old error), not the schema.
  const mixed = await post('/api/stem-studio/extract', { sourceAudioUrl: '/references/a.wav', tracks: ['vocals', 5] });
  assert.deepEqual(mixed.body, { error: 'Invalid track names: 5' });
});

test('POST /supersep: missing sourceAudioUrl is refused; null sourceFileName/level are accepted like before', async () => {
  const bad = await post('/api/stem-studio/supersep', {});
  assert.deepEqual(bad.body, { error: 'sourceAudioUrl is required' });
  assert.equal(bad.status, 400);
  const ok = await post('/api/stem-studio/supersep', { sourceAudioUrl: '/references/a.wav', sourceFileName: null, level: null });
  assert.equal(ok.status, 200);
  assert.match(ok.body.id, /^[0-9a-f-]{36}$/);
});

test('POST /api/supersep/separate: missing audioUrl is refused before any file read', async () => {
  const bad = await post('/api/supersep/separate', {});
  assert.deepEqual(bad.body, { error: 'audioUrl required in request body' });
  assert.equal(bad.status, 400);
});

// ── HTTP: stemStudio.ts reads from disk, independent of a running job ─────────

function writeFixtureJob(id: string, meta: Record<string, unknown>, stems: Record<string, Buffer> = {}) {
  const dir = path.join(DATA_DIR, 'stems', id);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, '_meta.json'), JSON.stringify(meta, null, 2));
  for (const [name, buf] of Object.entries(stems)) fs.writeFileSync(path.join(dir, `${name}.wav`), buf);
}

test('GET /:jobId/progress and /:jobId/result read a completed job back from disk, not the in-memory map', async () => {
  const id = 'fixture-extract-1';
  writeFixtureJob(id, {
    id, type: 'extract', sourceAudioUrl: '/references/a.wav', sourceFileName: 'a.wav',
    tracks: ['vocals', 'drums'], completedStems: ['vocals', 'drums'], createdAt: new Date().toISOString(),
  }, { vocals: Buffer.from([1, 2, 3]), drums: Buffer.from([4, 5]) });

  const progress = await get(`/api/stem-studio/${id}/progress`);
  assert.equal(progress.status, 200);
  assert.equal(progress.body.status, 'done');
  assert.deepEqual(progress.body.completedStems, ['vocals', 'drums']);

  const result = await get(`/api/stem-studio/${id}/result`);
  assert.equal(result.status, 200);
  assert.equal(result.body.type, 'extract');
  assert.deepEqual(result.body.stems.map((s: any) => s.trackName), ['vocals', 'drums']);
  assert.equal(result.body.stems[0].audioUrl, `/api/stem-studio/${id}/stem/vocals`);
  assert.equal(result.body.stems[0].sizeBytes, 3);
});

test('a SuperSep fixture carries category/stage on its stems; an Extract fixture does not', async () => {
  const id = 'fixture-supersep-1';
  writeFixtureJob(id, {
    id, type: 'supersep', sourceAudioUrl: '/references/a.wav', sourceFileName: 'a.wav', sepLevel: 2,
    tracks: ['vocals'], completedStems: ['vocals'],
    stemMeta: [{ originalName: 'Vocals', safeName: 'vocals', category: 'vocal', index: 0, stage: 1 }],
    createdAt: new Date().toISOString(),
  }, { vocals: Buffer.from([1]) });
  const result = await get(`/api/stem-studio/${id}/result`);
  assert.equal(result.body.stems[0].trackName, 'Vocals');
  assert.equal(result.body.stems[0].category, 'vocal');
  assert.equal(result.body.stems[0].stage, 1);
});

test('GET /:jobId/progress and /:jobId/result 404 for an unknown job', async () => {
  assert.equal((await get('/api/stem-studio/does-not-exist/progress')).status, 404);
  assert.equal((await get('/api/stem-studio/does-not-exist/result')).status, 404);
});

test('GET /jobs lists fixtures from disk, newest first; GET /stats sums their bytes', async () => {
  const jobs = await get('/api/stem-studio/jobs');
  assert.equal(jobs.status, 200);
  assert.ok(Array.isArray(jobs.body));
  assert.ok(jobs.body.some((j: any) => j.id === 'fixture-extract-1'));

  const stats = await get('/api/stem-studio/stats');
  assert.equal(stats.status, 200);
  assert.ok(stats.body.jobCount >= 2);
  assert.ok(stats.body.stemCount >= 3);
  assert.ok(stats.body.totalBytes > 0);
});

test('DELETE /:jobId removes the fixture directory; a second delete 404s', async () => {
  writeFixtureJob('fixture-to-delete', { id: 'fixture-to-delete', type: 'extract', completedStems: [], tracks: [], createdAt: new Date().toISOString() });
  const res = await fetch(`${base}/api/stem-studio/fixture-to-delete`, { method: 'DELETE' });
  assert.equal(res.status, 200);
  assert.equal(fs.existsSync(path.join(DATA_DIR, 'stems', 'fixture-to-delete')), false);
  const again = await fetch(`${base}/api/stem-studio/fixture-to-delete`, { method: 'DELETE' });
  assert.equal(again.status, 404);
});

test('DELETE /all clears every job directory', async () => {
  writeFixtureJob('fixture-clear-all', { id: 'fixture-clear-all', type: 'extract', completedStems: [], tracks: [], createdAt: new Date().toISOString() });
  const res = await fetch(`${base}/api/stem-studio/all`, { method: 'DELETE' });
  assert.equal(res.status, 200);
  assert.deepEqual((await get('/api/stem-studio/jobs')).body, []);
});

// ── HTTP: supersep.ts proxy — fake ace-server, no network ──────────────────

function fakeAceServer(responses: Record<string, Response | ((init?: any) => Response)>) {
  const seen: Array<{ method: string; url: string; body?: string }> = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: any) => {
    if (url.startsWith(base)) return realFetch(url, init); // the test's own HTTP calls into our app
    seen.push({ method: init?.method || 'GET', url, body: init?.body });
    for (const [match, respond] of Object.entries(responses)) {
      if (url.includes(match)) return typeof respond === 'function' ? respond(init) : respond;
    }
    throw new Error(`unexpected fetch ${url}`);
  }) as typeof fetch;
  return { seen, restore: () => { globalThis.fetch = realFetch; } };
}

test('GET /:jobId/progress always replies 200 with the decoded body, even when ace-server itself errored', async (t) => {
  const fake = fakeAceServer({ '/supersep/progress': () => new Response(JSON.stringify({ error: 'worker crashed' }), { status: 500 }) });
  t.after(fake.restore);
  const progress = await get('/api/supersep/job-1/progress');
  assert.equal(progress.status, 200, 'Node does not check ace-server\'s status for this route');
  assert.deepEqual(progress.body, { error: 'worker crashed' });
});

test('POST /:jobId/release always replies with Node\'s own { ok }, discarding ace-server\'s body', async (t) => {
  const okFake = fakeAceServer({ '/supersep/release': () => new Response('{"released":true}', { status: 200 }) });
  const ok = await fetch(`${base}/api/supersep/job-1/release`, { method: 'POST' });
  assert.equal(ok.status, 200);
  assert.deepEqual(await ok.json(), { ok: true });
  okFake.restore();

  const failFake = fakeAceServer({ '/supersep/release': () => new Response(JSON.stringify({ error: 'not found' }), { status: 404 }) });
  t.after(failFake.restore);
  const failed = await fetch(`${base}/api/supersep/job-1/release`, { method: 'POST' });
  assert.equal(failed.status, 404);
  assert.deepEqual(await failed.json(), { ok: false });
});

test('GET /:jobId/result forwards ace-server\'s status and body unchanged, success or failure', async (t) => {
  const okFake = fakeAceServer({ '/supersep/result': () => new Response(JSON.stringify({ stems: [{ name: 'vocals', category: 'vocal', index: 0 }] })) });
  const ok = await get('/api/supersep/job-1/result');
  assert.equal(ok.status, 200);
  assert.deepEqual(ok.body, { stems: [{ name: 'vocals', category: 'vocal', index: 0 }] });
  okFake.restore();

  const failFake = fakeAceServer({ '/supersep/result': () => new Response(JSON.stringify({ error: 'Job not found' }), { status: 404 }) });
  t.after(failFake.restore);
  const failed = await get('/api/supersep/job-1/result');
  assert.equal(failed.status, 404);
  assert.deepEqual(failed.body, { error: 'Job not found' });
});

test('GET /:jobId/stem/:index streams the WAV on success, but replaces ace-server\'s error body on failure', async (t) => {
  const okFake = fakeAceServer({ '/supersep/serve': () => new Response(Buffer.from([9, 9, 9]), { headers: { 'Content-Type': 'audio/wav' } }) });
  const stem = await fetch(`${base}/api/supersep/job-1/stem/0`);
  assert.equal(stem.status, 200);
  assert.equal(stem.headers.get('content-type'), 'audio/wav');
  assert.deepEqual([...new Uint8Array(await stem.arrayBuffer())], [9, 9, 9]);
  okFake.restore();

  const failFake = fakeAceServer({ '/supersep/serve': () => new Response(JSON.stringify({ error: 'stem index out of range' }), { status: 400 }) });
  t.after(failFake.restore);
  const failed = await fetch(`${base}/api/supersep/job-1/stem/99`);
  assert.equal(failed.status, 400);
  assert.deepEqual(await failed.json(), { error: 'Failed to fetch stem' }, 'the real ace-server message is discarded, not forwarded');
});

test('POST /recombine forwards the real { id, stems } payload and relays success/failure unchanged', async (t) => {
  const okFake = fakeAceServer({ '/supersep/recombine': () => new Response(Buffer.from([7, 7]), { headers: { 'Content-Type': 'audio/wav' } }) });
  const payload = { id: 'job-1', stems: [{ index: 0, volume: 1, muted: false }] };
  assert.equal(supersepRecombineRequestSchema.safeParse(payload).success, true);
  const ok = await fetch(`${base}/api/supersep/recombine`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  assert.equal(ok.status, 200);
  assert.deepEqual([...new Uint8Array(await ok.arrayBuffer())], [7, 7]);
  assert.deepEqual(JSON.parse(okFake.seen.find(s => s.url.includes('/supersep/recombine'))!.body!), payload, 'the real id+stems payload reaches ace-server, not a partial one');
  okFake.restore();

  const failFake = fakeAceServer({ '/supersep/recombine': () => new Response(JSON.stringify({ error: 'Job not complete' }), { status: 409 }) });
  t.after(failFake.restore);
  const failed = await fetch(`${base}/api/supersep/recombine`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  assert.equal(failed.status, 409);
  assert.deepEqual(await failed.json(), { error: 'Job not complete' });
});

test('STEM_TRACK_NAMES matches the twelve tracks the UI and layer-render both rely on', () => {
  assert.equal(STEM_TRACK_NAMES.length, 12);
  assert.ok(STEM_TRACK_NAMES.includes('vocals'));
  assert.ok(STEM_TRACK_NAMES.includes('woodwinds'));
});
