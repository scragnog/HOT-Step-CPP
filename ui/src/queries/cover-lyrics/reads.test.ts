import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { QueryObserver } from '@tanstack/react-query';
import { queryClient, createQueryClient } from '../client';
import { sharedKeys, startAuthSession, serverKey } from '../keys';
import { pollWhile } from '../contract';
import { snapshotOptions, songsChanged } from './reads';
import { lireekApi } from '../../services/lireekApi';
import { studioDraftsApi } from '../../services/studioDraftsApi';
import { yue2CoverApi } from '../../services/yue2CoverApi';
import { waitForSnapshot } from './wait';
import { coverSongsOptions, deleteCovers } from './songs';
import { waitForCoverJob } from '../../services/coverWorkflowApi';
import { setAuthToken } from '../../services/axiosClient';

const originalFetch = globalThis.fetch;
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
const tick = () => new Promise(resolve => setImmediate(resolve));
afterEach(() => { globalThis.fetch = originalFetch; queryClient.clear(); });

test('Cover writes, deletes and render completion refresh every cached song view', async () => {
  startAuthSession(queryClient, 'cover-writes');
  let revision = 0;
  let deleted = false;
  const options = [
    { queryKey: sharedKeys.songs(), path: '/api/songs' },
    { queryKey: sharedKeys.song('keep'), path: '/api/songs/keep' },
    { queryKey: serverKey('shell-queue/recent-songs', { auth: true }), path: '/api/songs/recent' },
    { queryKey: serverKey('studios/recent-songs', { auth: true }), path: '/api/songs/recent?source=studio' },
    { queryKey: coverSongsOptions('cover-writes').queryKey, path: '/api/songs?source=cover-studio' },
  ].map(({ queryKey, path }) => ({ queryKey, queryFn: async () => (await fetch(path)).json() }));
  const reads = new Map<string, number>();
  globalThis.fetch = async (url, opts) => {
    const path = String(url);
    if (opts?.method === 'POST') { deleted = true; revision++; return json({ success: true }); }
    if (path.includes('/workflows/jobs/')) return json({ job: { id: 'render', kind: 'cover-render', status: 'succeeded' }, events: [] });
    reads.set(path, (reads.get(path) ?? 0) + 1);
    return json({ songs: deleted ? [{ id: 'keep' }] : [{ id: 'keep' }, { id: 'remove' }], revision });
  };
  const observers = options.map(option => new QueryObserver(queryClient, option));
  const off = observers.map(observer => observer.subscribe(() => {}));
  try {
    await Promise.all(options.map(option => queryClient.query(option)));
    revision++;
    await songsChanged();
    for (const option of options) assert.equal(queryClient.getQueryData(option.queryKey).revision, 1);
    await deleteCovers(['remove']);
    for (const option of options) assert.equal(queryClient.getQueryData(option.queryKey).revision, 2);
    revision++;
    await waitForCoverJob('render', () => false);
    for (const option of options) {
      assert.equal(queryClient.getQueryData(option.queryKey).revision, 3);
      assert.deepEqual(queryClient.getQueryData(option.queryKey).songs.map((song: { id: string }) => song.id), ['keep']);
    }
    for (const count of reads.values()) assert.ok(count >= 4, 'each view refetched after all three changes');
  } finally { off.forEach(unsubscribe => unsubscribe()); }
});

test('690 simultaneous catalogue consumers and a fresh return use one list and one bulk preset read', async () => {
  const calls: string[] = [];
  globalThis.fetch = async url => {
    calls.push(String(url));
    return json(String(url).endsWith('/presets') ? { presets: [{ lyrics_set_id: 7 }] } : [{ id: 7, artist_id: 1 }]);
  };
  const result = await Promise.all(Array.from({ length: 690 }, () => Promise.all([lireekApi.listLyricsSets(), lireekApi.listAllPresets()])));
  assert.equal(calls.length, 2);
  assert.equal(result[0][0].lyrics_sets[0].id, 7);
  await lireekApi.listLyricsSets(); await lireekApi.listAllPresets();
  assert.deepEqual(calls.sort(), ['/api/lireek/lyrics-sets', '/api/lireek/presets']);
});

test('late source A is aborted and cannot populate source B or its cancelled cache', async () => {
  startAuthSession(queryClient, 'source-test');
  let release!: (response: Response) => void;
  let signal: AbortSignal | undefined;
  globalThis.fetch = async (_url, opts) => {
    if (JSON.parse(String(opts?.body)).songId === 'A') {
      signal = opts?.signal ?? undefined;
      return new Promise(resolve => { release = resolve; });
    }
    return json({ matched: true, lyrics: 'B' });
  };
  const a = yue2CoverApi.sourceMetadata({ songId: 'A' }).catch(error => error);
  await tick();
  await queryClient.cancelQueries({ queryKey: ['hs', 'cover-lyrics/cover'] });
  const b = await yue2CoverApi.sourceMetadata({ songId: 'B' });
  release(json({ matched: true, lyrics: 'A' }));
  await a;
  assert.equal(signal?.aborted, true);
  assert.equal(b.lyrics, 'B');
  assert.equal(queryClient.getQueriesData({ queryKey: ['hs', 'cover-lyrics/cover'] }).filter(([, data]) => data !== undefined).length, 1);
});

test('a queryable cover POST is keyed only by the body fields that change its answer', async () => {
  startAuthSession(queryClient, 'source-key');
  let posts = 0;
  globalThis.fetch = async () => { posts++; return json({ matched: false }); };
  // Different labels/force flags describe the same source; the lookup ignores
  // them, so they must not create another cache entry.
  await yue2CoverApi.sourceMetadata({ songId: 'S', sourceLabel: 'One', force: true });
  await yue2CoverApi.sourceMetadata({ songId: 'S', sourceLabel: 'Two', force: false });
  const entries = queryClient.getQueriesData({ queryKey: ['hs', 'cover-lyrics/cover'] })
    .filter(([, data]) => data !== undefined);
  assert.equal(entries.length, 1, 'one song has one metadata entry');
  assert.ok(posts >= 1);
});

test('profile and draft writes refresh their snapshots and submit only once', async () => {
  startAuthSession(queryClient, 'write-test');
  // The key's auth scope now derives from the session accessor (P2), so pair
  // it like AuthProvider does — otherwise reads are unscoped and the
  // session-scoped invalidation prefix never matches.
  setAuthToken('write-test');
  let profile = 'before'; let revision = 1; let writes = 0;
  globalThis.fetch = async (url, opts) => {
    const write = opts?.method && opts.method !== 'GET';
    if (write) { writes++; profile = 'after'; revision++; }
    if (String(url).includes('/studio-drafts')) return json({ document: { id: 'draft', revision } });
    return json({ id: 1, profile_data: { style_caption: profile } });
  };
  try {
    assert.equal((await lireekApi.getProfile(1)).profile_data.style_caption, 'before');
    await lireekApi.buildProfile(1, { provider: 'test' });
    assert.equal((await lireekApi.getProfile(1)).profile_data.style_caption, 'after');
    assert.equal((await studioDraftsApi.get('draft')).document.revision, 2);
    await studioDraftsApi.update('draft', 2, { studio: 'cover', fields: {} });
    assert.equal((await studioDraftsApi.get('draft')).document.revision, 3);
    assert.equal(writes, 2);
  } finally { setAuthToken(null); }
});

test('read errors retain HTTP status, failed submits never retry and failures are not cached as empty lists', async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls++; return json({ error: 'Denied' }, 403); };
  await assert.rejects(lireekApi.listArtists(), error => (error as { status: number }).status === 403);
  assert.equal(calls, 1);
  globalThis.fetch = async () => { calls++; return json({ error: 'Failed' }, 500); };
  await assert.rejects(lireekApi.createArtist({ name: 'Test' }), /Failed/);
  assert.equal(calls, 2);
  assert.equal(queryClient.getQueryData(snapshotOptions('lireek', '/lireek/artists').queryKey), undefined);
});

test('build snapshots share one request for simultaneous observers and terminal data stops polling', async () => {
  const client = createQueryClient();
  let calls = 0;
  globalThis.fetch = async () => { calls++; await tick(); return json({ id: 'session', pending: [] }); };
  const options = snapshotOptions<{ id: string; pending: unknown[] }>('cover', '/yue2-cover/build/session', 'test');
  const first = new QueryObserver(client, options); const second = new QueryObserver(client, options);
  const off = [first.subscribe(() => {}), second.subscribe(() => {})];
  try {
    await client.query(options);
    assert.equal(calls, 1);
    const interval = pollWhile<{ pending: unknown[] }>(data => !data || data.pending.length > 0);
    assert.equal(interval({ state: client.getQueryState(options.queryKey)! }), false);
  } finally { off.forEach(fn => fn()); client.clear(); }
});

test('workflow waiter detaches after terminal delivery and releases its observer', async () => {
  const key = snapshotOptions('cover', '/test-wait').queryKey;
  let deliveries = 0;
  const result = await waitForSnapshot(key, async () => ({ status: 'done' }), value => value.status !== 'done', () => { deliveries++; });
  await tick();
  assert.equal(result.status, 'done');
  assert.equal(deliveries, 1);
  assert.equal(queryClient.getQueryCache().find({ queryKey: key })?.getObserversCount(), 0);
});

test('deleting a cover cancels a late list and only removes data after success', async () => {
  const key = coverSongsOptions('test').queryKey;
  queryClient.setQueryData(key, { songs: [{ id: 'remove' }, { id: 'keep' }] });
  let release!: (response: Response) => void;
  let signal: AbortSignal | undefined;
  globalThis.fetch = async (_url, opts) => {
    if (opts?.method === 'POST') return json({ success: true });
    signal = opts?.signal ?? undefined;
    return new Promise(resolve => { release = resolve; });
  };
  const old = queryClient.query({ ...coverSongsOptions('test'), staleTime: 0 }).catch(error => error);
  await tick();
  await deleteCovers(['remove']);
  release(json({ songs: [{ id: 'remove' }, { id: 'keep' }] })); await old;
  assert.equal(signal?.aborted, true);
  assert.deepEqual(queryClient.getQueryData(key)?.songs.map(s => s.id), ['keep']);
});


test('cover catalogue restoration has no command path and entry has no per-id preset scan', async () => {
  const { readFile } = await import('node:fs/promises');
  const source = await readFile(new URL('../../components/cover-studio/CoverStudio.tsx', import.meta.url), 'utf8');
  const restore = source.slice(source.indexOf('  const restoredPresetLoaded'), source.indexOf('  const openUploadedAsset'));
  assert.doesNotMatch(restore, /loadArtistPresets|generateCaption|resolveCaptionOnNode/);
  assert.doesNotMatch(source, /lireekApi\.getPreset\(/);
  assert.doesNotMatch(source, /listLyricsSets\(artist\.id\)/);
});
