import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// A fresh DATA_DIR and database, always, before anything reads config.
const DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'songs-contract-'));
process.env.DATA_DIR = DATA_DIR;

const express = (await import('express')).default;
const database = await import('../db/database.js');
database.initDb();
const authRoutes = (await import('../routes/auth.js')).default;
const songRoutes = (await import('../routes/songs.js')).default;
const { RECENT_SONG_FIELDS, SONG_EDITABLE_FIELDS, SONG_FIELDS } = await import('./songs.js');
type Song = import('./songs.js').Song;

const app = express();
app.use(express.json());
app.use('/api/auth', authRoutes);
app.use('/api/songs', songRoutes);
const server = app.listen(0);
await new Promise(resolve => server.once('listening', resolve));
const base = `http://127.0.0.1:${(server.address() as { port: number }).port}/api`;
test.after(() => new Promise(resolve => server.close(() => { database.closeDb(); fs.rmSync(DATA_DIR, { recursive: true, force: true }); resolve(undefined); })));

const { token } = await (await fetch(`${base}/auth/auto`)).json() as { token: string };
const call = async (method: string, p: string, body?: unknown) => {
  const res = await fetch(`${base}/songs${p}`, { method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: res.status, body: await res.json() as any };
};

test('SONG_FIELDS is the songs table, in column order', () => {
  const columns = (database.getDb().prepare('PRAGMA table_info(songs)').all() as Array<{ name: string }>).map(c => c.name);
  assert.deepEqual(columns, [...SONG_FIELDS]);
});

test('create, detail and list send the row with tags and is_public decoded', async () => {
  const created = await call('POST', '', { id: 'song-1', title: 'One', tags: ['a', 'b'], generation_params: { source: 'create', seed: 7 } });
  assert.equal(created.status, 200);
  for (const song of [created.body.song as Song, (await call('GET', '/song-1')).body.song as Song]) {
    assert.deepEqual(Object.keys(song), [...SONG_FIELDS]);
    assert.deepEqual(song.tags, ['a', 'b']);
    assert.equal(song.is_public, false);
    assert.equal(song.generation_params, '{"source":"create","seed":7}', 'generation_params stays a JSON string');
    assert.equal(song.backend, 'ace');
  }
  await call('POST', '', { id: 'song-2', title: 'Two', generation_params: { source: 'cover-studio' } });
  database.getDb().prepare("UPDATE songs SET created_at = '2020-01-01 00:00:00' WHERE id = 'song-1'").run();
  const list = await call('GET', '');
  assert.deepEqual(list.body.songs.map((s: Song) => s.id), ['song-2', 'song-1'], 'newest first');
  assert.deepEqual((await call('GET', '?source=cover-studio')).body.songs.map((s: Song) => s.id), ['song-2']);
  assert.deepEqual((await call('GET', '/ids')).body.ids.sort(), ['song-1', 'song-2']);
  const recent = await call('GET', '/recent?limit=1');
  assert.equal(recent.body.songs.length, 1);
  assert.deepEqual(Object.keys(recent.body.songs[0]), [...RECENT_SONG_FIELDS]);
  assert.equal(recent.body.songs[0].source, 'cover-studio');
  assert.equal((await call('GET', '/nope')).status, 404);
});

test('PATCH writes only the editable fields and decodes the same way', async () => {
  const before = (await call('GET', '/song-1')).body.song as Song;
  const res = await call('PATCH', '/song-1', { title: 'Renamed', is_public: true, tags: ['z'], metadata_overrides: { artist: 'X' },
    audio_url: '/audio/other.wav', user_id: 'someone', backend: 'yue2', bogus: 1 });
  assert.equal(res.status, 200);
  const song = res.body.song as Song;
  assert.deepEqual([song.title, song.is_public, song.tags, song.metadata_overrides], ['Renamed', true, ['z'], '{"artist":"X"}']);
  assert.deepEqual([song.audio_url, song.user_id, song.backend], [before.audio_url, before.user_id, before.backend]);
  assert.equal((await call('PATCH', '/song-1', { metadata_overrides: null })).body.song.metadata_overrides, '');
  assert.ok(SONG_EDITABLE_FIELDS.every(f => (SONG_FIELDS as readonly string[]).includes(f)));
  assert.equal((await call('PATCH', '/missing', { title: 'x' })).status, 404);
});

test('bulk delete reports its count; deleting an unknown song is 404', async () => {
  assert.deepEqual((await call('POST', '/bulk-delete', { ids: [] })).status, 400);
  const res = await call('POST', '/bulk-delete', { ids: ['song-2', 'never'] });
  assert.deepEqual([res.status, res.body], [200, { success: true, deletedCount: 1 }]);
  assert.equal((await call('DELETE', '/never')).status, 404);
});
