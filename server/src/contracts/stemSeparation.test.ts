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
const { STEM_TRACK_NAMES, stemExtractRequestSchema, stemSupersepRequestSchema, supersepSeparateRequestSchema } = await import('./stemSeparation.js');

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

test('stemExtractRequestSchema: sourceAudioUrl and a non-empty tracks array are required', () => {
  assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav', tracks: ['vocals'] }).success, true);
  assert.equal(stemExtractRequestSchema.safeParse({ tracks: ['vocals'] }).success, false);
  assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav', tracks: [] }).success, false);
  assert.equal(stemExtractRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav' }).success, false);
});

test('stemSupersepRequestSchema: sourceAudioUrl required, level optional', () => {
  assert.equal(stemSupersepRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav' }).success, true);
  assert.equal(stemSupersepRequestSchema.safeParse({ sourceAudioUrl: '/references/a.wav', level: 2 }).success, true);
  assert.equal(stemSupersepRequestSchema.safeParse({}).success, false);
});

test('supersepSeparateRequestSchema: audioUrl required', () => {
  assert.equal(supersepSeparateRequestSchema.safeParse({ audioUrl: '/references/a.wav' }).success, true);
  assert.equal(supersepSeparateRequestSchema.safeParse({}).success, false);
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

test('POST /supersep: missing sourceAudioUrl is refused', async () => {
  const bad = await post('/api/stem-studio/supersep', {});
  assert.deepEqual(bad.body, { error: 'sourceAudioUrl is required' });
  assert.equal(bad.status, 400);
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

test('supersep.ts proxies progress/result/stem/release/recombine to ace-server and relays the response unchanged', async (t) => {
  const seen: string[] = [];
  const realFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = realFetch; });
  globalThis.fetch = (async (url: string, init?: any) => {
    if (url.startsWith(base)) return realFetch(url, init); // the test's own HTTP calls into our app
    seen.push(`${init?.method || 'GET'} ${url}`);
    if (url.includes('/supersep/progress')) return new Response(JSON.stringify({ status: 'running', progress: 42, message: 'Separating' }));
    if (url.includes('/supersep/release')) return new Response(JSON.stringify({}), { status: 200 });
    if (url.includes('/supersep/result')) return new Response(JSON.stringify({ stems: [{ name: 'vocals', category: 'vocal', index: 0 }] }));
    if (url.includes('/supersep/serve')) return new Response(Buffer.from([9, 9, 9]), { headers: { 'Content-Type': 'audio/wav' } });
    if (url.includes('/supersep/recombine')) return new Response(Buffer.from([7, 7]), { headers: { 'Content-Type': 'audio/wav' } });
    throw new Error(`unexpected fetch ${url}`);
  }) as typeof fetch;

  const progress = await get('/api/supersep/job-1/progress');
  assert.deepEqual(progress.body, { status: 'running', progress: 42, message: 'Separating' });

  const release = await fetch(`${base}/api/supersep/job-1/release`, { method: 'POST' });
  assert.equal(release.status, 200);

  const result = await get('/api/supersep/job-1/result');
  assert.deepEqual(result.body, { stems: [{ name: 'vocals', category: 'vocal', index: 0 }] });

  const stem = await fetch(`${base}/api/supersep/job-1/stem/0`);
  assert.equal(stem.status, 200);
  assert.equal(stem.headers.get('content-type'), 'audio/wav');
  assert.deepEqual([...new Uint8Array(await stem.arrayBuffer())], [9, 9, 9]);

  const recombine = await fetch(`${base}/api/supersep/recombine`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ stems: [{ index: 0, volume: 1, muted: false }] }) });
  assert.equal(recombine.status, 200);
  assert.deepEqual([...new Uint8Array(await recombine.arrayBuffer())], [7, 7]);
  assert.ok(seen.some(s => s.startsWith('POST') && s.includes('/supersep/recombine')));
});

test('STEM_TRACK_NAMES matches the twelve tracks the UI and layer-render both rely on', () => {
  assert.equal(STEM_TRACK_NAMES.length, 12);
  assert.ok(STEM_TRACK_NAMES.includes('vocals'));
  assert.ok(STEM_TRACK_NAMES.includes('woodwinds'));
});
