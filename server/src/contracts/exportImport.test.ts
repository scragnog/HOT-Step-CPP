import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// The routes write uploads under DATA_DIR and use its database: a fresh one,
// always, before anything reads config.
const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'export-import-'));
process.env.DATA_DIR = DATA_DIR;

const express = (await import('express')).default;
const database = await import('../db/database.js');
const authRoutes = (await import('../routes/auth.js')).default;
const exportImportRoutes = (await import('../routes/exportImport.js')).default;
const contracts = await import('./exportImport.js');
const legacy = await import('../services/exportImport/contracts.js');
const { IMPORT_EXTENSIONS } = await import('../services/library/importTrack.js');
type ExportResolveResponse = import('./exportImport.js').ExportResolveResponse;
type ImportResponse = import('./exportImport.js').ImportResponse;

database.initDb();
const app = express();
app.use(express.json());
app.use('/api/auth', authRoutes);
app.use('/api/export-import', exportImportRoutes);
const server = app.listen(0);
await new Promise(resolve => server.once('listening', resolve));
const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
test.after(() => new Promise(resolve => server.close(() => { database.closeDb(); fs.rmSync(DATA_DIR, { recursive: true, force: true }); resolve(undefined); })));

const { token, user } = await (await fetch(`${base}/api/auth/auto`)).json() as { token: string; user: { id: string } };
const json = async (p: string, body: unknown, auth = true) => {
  const res = await fetch(`${base}/api/export-import${p}`, { method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(auth ? { Authorization: `Bearer ${token}` } : {}) }, body: JSON.stringify(body) });
  return { status: res.status, type: res.headers.get('content-type') ?? '', body: await res.json() as Record<string, any> };
};
const upload = async (name: string, bytes: Uint8Array, auth = true) => {
  const form = new FormData();
  form.append('audio', new Blob([bytes]), name);
  const res = await fetch(`${base}/api/export-import/assets`, { method: 'POST', body: form,
    headers: auth ? { Authorization: `Bearer ${token}` } : {} });
  return { status: res.status, body: await res.json() as Record<string, any> };
};

test('the old service path re-exports the published contract unchanged', () => {
  for (const name of ['audioFormat', 'audioVariant', 'exportRequest', 'importRequest', 'profileImportRequest', 'validateProfileImport'] as const)
    assert.equal(legacy[name], contracts[name], name);
  assert.deepEqual([...contracts.IMPORT_AUDIO_EXTENSIONS], IMPORT_EXTENSIONS);
});

test('upload: multipart `audio` becomes an asset id; auth, missing file and extension are refused', async () => {
  const ok = await upload('take.wav', new Uint8Array([1, 2, 3, 4]));
  assert.equal(ok.status, 200);
  assert.match(ok.body.assetId, /^[0-9a-f-]{36}$/);
  assert.equal((await upload('take.wav', new Uint8Array([1]), false)).status, 401);
  const bad = await upload('notes.txt', new Uint8Array([1]));
  assert.deepEqual([bad.status, bad.body.error], [400, 'Unsupported or missing audio file']);
  // Stored under DATA_DIR/references with a new name; never the client's path.
  const stored = fs.readdirSync(path.join(DATA_DIR, 'references'));
  assert.equal(stored.length, 1);
  assert.match(stored[0], /^[0-9a-f-]{36}\.wav$/);
});

test('import: each item fails or succeeds on its own, and malformed bodies are 400', async () => {
  const unknown = '72a9d55e-0929-45b4-9d49-c15f1052aeca';
  const res = await json('/imports', { items: [{ assetId: unknown }, { assetId: unknown, description: 'x' }] });
  assert.equal(res.status, 200);
  const items = (res.body as ImportResponse).items;
  assert.deepEqual(items.map(i => [i.index, i.assetId, typeof i.error, i.song]), [[0, unknown, 'string', undefined], [1, unknown, 'string', undefined]]);
  const malformed = await json('/imports', { items: [{ assetId: 'C:\\audio.wav' }] });
  assert.equal(malformed.status, 400);
  assert.equal(malformed.body.error, 'Invalid import request');
  assert.ok(Array.isArray(malformed.body.issues));
  assert.equal((await json('/imports', { items: [{ assetId: unknown }] }, false)).status, 401);
});

test('export resolve: relative URLs and filenames per variant, per-item errors, JSON only', async () => {
  const db = database.getDb();
  fs.mkdirSync(path.join(DATA_DIR, 'audio'), { recursive: true });
  for (const f of ['a.wav', 'a_mastered.wav']) fs.writeFileSync(path.join(DATA_DIR, 'audio', f), 'x');
  db.prepare(`INSERT INTO songs (id, user_id, title, audio_url, mastered_audio_url) VALUES (?, ?, ?, ?, ?)`)
    .run('song-a', user.id, 'Fixture Song', '/audio/a.wav', '/audio/a_mastered.wav');
  db.prepare(`INSERT INTO songs (id, user_id, title, audio_url) VALUES (?, ?, ?, ?)`)
    .run('song-b', user.id, 'No Master', '/audio/missing.wav');
  const res = await json('/exports/resolve', { items: [{ songId: 'song-a' }, { songId: 'song-b' }, { songId: 'nope' }],
    format: 'mp3', bitrate: 192, downloadVersion: 'both', prepend: '01' });
  assert.equal(res.status, 200);
  assert.match(res.type, /application\/json/);
  const items = (res.body as ExportResolveResponse).items;
  const a = items.filter(i => i.index === 0);
  assert.deepEqual(a.map(i => i.variant), ['original', 'mastered']);
  for (const item of a) {
    assert.ok(item.url!.startsWith('/api/download/song-a?'), item.url);
    assert.equal(new URL(item.url!, base).searchParams.get('format'), 'mp3');
    assert.equal(item.error, undefined);
  }
  assert.deepEqual(a.map(i => i.filename), ['01 - Fixture Song - Unmastered.mp3', '01 - Fixture Song.mp3']);
  const b = items.filter(i => i.index === 1);
  assert.equal(b.length, 1);
  assert.match(b[0].error!, /file missing/);
  assert.equal(b[0].url, undefined);
  assert.deepEqual(items.filter(i => i.index === 2).map(i => i.error), ['Song not found']);
  assert.equal((await json('/exports/resolve', { items: [{ songId: '../x' }] })).status, 400);
  assert.equal((await json('/exports/resolve', { items: [{ songId: 'song-a' }] }, false)).status, 401);
});

test('profile validation unwraps a saved profile, names it from the file, and refuses empty input', async () => {
  const ok = await json('/profiles/validate', { filename: 'My:Set.json', profile: { name: 'x', data: { _format: 'hot-step-preset', bpm: 120 } } });
  assert.deepEqual([ok.status, ok.body], [200, { name: 'My_Set', data: { _format: 'hot-step-preset', bpm: 120 } }]);
  const empty = await json('/profiles/validate', { filename: 'e.json', profile: {} });
  assert.deepEqual([empty.status, empty.body.error, empty.body.details], [400, 'Invalid profile', 'Profile data is empty']);
  assert.equal((await json('/profiles/validate', { filename: 'e.json', profile: [] })).status, 400);
  assert.equal((await json('/profiles/validate', { filename: 'e.json', profile: { a: 1 } }, false)).status, 401);
});
