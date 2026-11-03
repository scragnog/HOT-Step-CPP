import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { QueryObserver } from '@tanstack/react-query';
import { createQueryClient, queryClient } from '../client';
import { resourceKey, serverKey, sharedKeys, startAuthSession } from '../keys';
import {
  invalidateSongs, patchSong, postProcessStatusOptions, recentSongsOptions, removeSongs, songsOptions,
} from './songs';
import { readAudioQueue, readGenerationStatus } from './queue';
import { queriedPresetsApi, queriedSettingsApi } from './preferences';
import { ENGINE_WORKERS_POLL_MS, engineWorkersOptions, gpusOptions } from './settings';
import { lrcOptions, lyricsJsonOptions, scoreAbcOptions } from './lyrics';
import { fetchPeaks, forgetPeaks, peaksCached } from '../../audio/peaksClient';
import { queryManager } from '../invalidation';

const clients: ReturnType<typeof createQueryClient>[] = [];
const client = () => { const c = createQueryClient(); clients.push(c); return c; };
const originalFetch = globalThis.fetch;
afterEach(() => { clients.splice(0).forEach(c => c.clear()); queryClient.clear(); globalThis.fetch = originalFetch; });
const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const song = (id: string, extra: Record<string, unknown> = {}) => ({ id, title: id, audio_url: `/audio/${id}.wav`, ...extra });

test('library and recent consumers share one request per key; sources and sessions stay apart', async () => {
  const c = client();
  const requests: string[] = [];
  globalThis.fetch = async input => { requests.push(String(input)); await tick(); return json({ songs: [song('a')] }); };
  await Promise.all([c.query(songsOptions()), c.query(songsOptions()), c.query(songsOptions())]);
  await Promise.all([c.query(recentSongsOptions('create')), c.query(recentSongsOptions('create'))]);
  await c.query(recentSongsOptions());
  assert.deepEqual(requests, ['/api/songs', '/api/songs/recent?limit=50&source=create', '/api/songs/recent?limit=50']);
  // The token never enters a key (it is not part of one any more), and a new
  // session gets its own entries.
  assert.ok(!JSON.stringify(songsOptions().queryKey).includes('secret-token'));
  const before = sharedKeys.songs();
  startAuthSession(c, 'another-token');
  assert.notDeepEqual(sharedKeys.songs(), before);
});

test('a song write refetches every song view, studios recent feed included', async () => {
  const c = client();
  let version = 0;
  globalThis.fetch = async input => {
    await tick();
    return String(input).includes('/recent') ? json({ songs: [song(`r${version}`)] }) : json({ songs: [song(`s${version}`)] });
  };
  const library = new QueryObserver(c, songsOptions());
  const recent = new QueryObserver(c, recentSongsOptions());
  const stops = [library.subscribe(() => {}), recent.subscribe(() => {})];
  const studios = serverKey('studios/recent-songs', { auth: true, filters: { limit: 50 } });
  c.setQueryData(studios, { songs: [] });
  try {
    await c.query(songsOptions());
    await c.query(recentSongsOptions());
    version = 1;
    await invalidateSongs(c);
    assert.equal(library.getCurrentResult().data?.songs[0].id, 's1');
    assert.equal(recent.getCurrentResult().data?.songs[0].id, 'r1');
    assert.equal(c.getQueryState(studios)?.isInvalidated, true, 'studios feed is invalidated, not edited');
  } finally { stops.forEach(stop => stop()); }
});

test('a deleted row is gone at once and a list read already in flight cannot bring it back', async () => {
  const c = client();
  let release: (() => void) | undefined;
  let deleted = false;
  globalThis.fetch = async () => {
    // The first read is the slow one, started before the delete succeeds.
    if (!deleted) { await new Promise<void>(resolve => { release = resolve; }); return json({ songs: [song('a'), song('b')] }); }
    return json({ songs: [song('b')] });
  };
  c.setQueryData(songsOptions().queryKey, { songs: [song('a'), song('b')] as never[] });
  const observer = new QueryObserver(c, { ...songsOptions(), staleTime: 0 });
  const stop = observer.subscribe(() => {});
  try {
    while (!release) await tick();
    deleted = true;
    const refetched = removeSongs(c, ['a']);
    assert.deepEqual(observer.getCurrentResult().data?.songs.map(s => s.id), ['b']);
    release!();
    await refetched;
    await tick();
    assert.deepEqual(observer.getCurrentResult().data?.songs.map(s => s.id), ['b']);
  } finally { stop(); }
});

test('patchSong edits the library, the detail and the recent feed in place', () => {
  const c = client();
  c.setQueryData(songsOptions().queryKey, { songs: [song('a'), song('b')] as never[] });
  c.setQueryData(sharedKeys.song('a'), { song: song('a') });
  c.setQueryData(recentSongsOptions('create').queryKey, { songs: [song('a')] as never[] });
  void patchSong(c, 'a', { mastered_audio_url: '/audio/a_m.wav' });
  const lib = c.getQueryData<{ songs: Record<string, unknown>[] }>(songsOptions().queryKey)!;
  assert.equal(lib.songs[0].mastered_audio_url, '/audio/a_m.wav');
  assert.equal(lib.songs[1].mastered_audio_url, undefined);
  assert.equal(c.getQueryData<{ song: Record<string, unknown> }>(sharedKeys.song('a'))!.song.mastered_audio_url, '/audio/a_m.wav');
  assert.equal(c.getQueryData<{ songs: Record<string, unknown>[] }>(recentSongsOptions('create').queryKey)!.songs[0].mastered_audio_url, '/audio/a_m.wav');
});

test('a completion patch reconciles every song view; an older read in flight cannot undo it', async () => {
  const c = client();
  let release: (() => void) | undefined;
  let served = 0;
  globalThis.fetch = async () => {
    // The first library read predates the post-processing pass and is slow.
    if (served++ === 0) { await new Promise<void>(resolve => { release = resolve; }); return json({ songs: [song('a')] }); }
    return json({ songs: [song('a', { mastered_audio_url: '/audio/a_m.wav' })] });
  };
  c.setQueryData(songsOptions().queryKey, { songs: [song('a')] as never[] });
  const studios = serverKey('studios/recent-songs', { auth: true, filters: { limit: 50 } });
  c.setQueryData(studios, { songs: [song('a')] });
  const library = new QueryObserver(c, { ...songsOptions(), staleTime: 0 });
  const stop = library.subscribe(() => {});
  try {
    while (!release) await tick();
    const reconciled = patchSong(c, 'a', { mastered_audio_url: '/audio/a_m.wav' });
    assert.equal(c.getQueryState(studios)?.isInvalidated, true, 'studios feed is reconciled too');
    release!();
    await reconciled;
    await tick();
    const row = library.getCurrentResult().data!.songs[0] as unknown as Record<string, unknown>;
    assert.equal(row.mastered_audio_url, '/audio/a_m.wav');
  } finally { stop(); }
});

test('queue status reads: concurrent loops share one request, and failures are not retried', async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls++; await tick(); return json({ jobId: 'j', status: 'running', progress: 0.4 }); };
  const [a, b, c] = await Promise.all([readGenerationStatus('j'), readGenerationStatus('j'), readGenerationStatus('j')]);
  assert.equal(calls, 1);
  assert.equal(a.status, 'running'); assert.equal(b, a); assert.equal(c, a);

  calls = 0;
  globalThis.fetch = async () => { calls++; return json({ error: 'down' }, 503); };
  await assert.rejects(readGenerationStatus('other'));
  await assert.rejects(readAudioQueue());
  assert.equal(calls, 2, 'one request per read: the executor owns what a failure means');
});

test('a waiter that accepts a recent list rides the projection read instead of asking again', async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls++; return json({ items: [{ id: 'x', status: 'pending' }] }); };
  await readAudioQueue();              // the projection: always asks
  await readAudioQueue(1500);          // the enqueue waiter: a list under one poll old is enough
  assert.equal(calls, 1);
  await readAudioQueue();
  assert.equal(calls, 2);
});

test('post-processing polls while running and stops for good at a terminal status', () => {
  const interval = postProcessStatusOptions('s', 'j').refetchInterval as (q: { state: Record<string, unknown> }) => number | false;
  const state = (data: unknown) => ({ state: { data, status: 'success', errorUpdateCount: 0 } });
  assert.equal(interval(state(undefined)), 1500);
  assert.equal(interval(state(null)), 1500, 'an aged-out job keeps the last state and keeps polling');
  assert.equal(interval(state({ status: 'running' })), 1500);
  assert.equal(interval(state({ status: 'succeeded' })), false);
  assert.equal(interval(state({ status: 'failed' })), false);
});

test('preference reads are fresh and shared; a write runs once and the next read sees it', async () => {
  const requests: string[] = [];
  let revision = 1;
  globalThis.fetch = async (input, init) => {
    const method = init?.method ?? 'GET';
    requests.push(`${method} ${String(input)}`);
    await tick();
    if (method === 'PUT') { revision++; return json({ error: 'conflict', currentRevision: revision }, 409); }
    return json({ documents: [{ id: 'p', revision }] });
  };
  await Promise.all([queriedPresetsApi.list('vst-chain'), queriedPresetsApi.list('vst-chain')]);
  assert.equal(requests.length, 1);
  await assert.rejects(queriedPresetsApi.update('vst-chain', 'p', 1, {}));
  const after = await queriedPresetsApi.list<{ id: string; revision: number }>('vst-chain');
  assert.deepEqual(requests, [
    'GET /api/preferences/presets/vst-chain',
    'PUT /api/preferences/presets/vst-chain/p',
    'GET /api/preferences/presets/vst-chain',
  ]);
  assert.equal(after.documents[0].revision, 2, 'conflict read-back sees the current revision');

  requests.length = 0;
  globalThis.fetch = async (input, init) => { requests.push(`${init?.method ?? 'GET'} ${String(input)}`); return json({ document: null }); };
  await queriedSettingsApi.get('storm-tuning');
  assert.deepEqual(requests, ['GET /api/preferences/settings/storm-tuning']);
  assert.deepEqual(resourceKey('shell-queue/presets').slice(0, 2), ['hs', 'shell-queue/presets']);
});

test("a shell write or delete invalidates Cover's cached feed as well as the library", async () => {
  const c = client();
  globalThis.fetch = async () => json({ songs: [song('b')] });
  const cover = serverKey('cover-lyrics/cover-songs', { auth: true });
  c.setQueryData(cover, { songs: [song('a'), song('b')] });
  await patchSong(c, 'a', { title: 'renamed' });
  assert.equal(c.getQueryState(cover)?.isInvalidated, true, 'write: Cover feed refetches on next use');
  c.setQueryData(cover, { songs: [song('a'), song('b')] });
  await removeSongs(c, ['a']);
  assert.equal(c.getQueryState(cover)?.isInvalidated, true, 'delete: Cover feed refetches on next use');
});

test('engine worker status refreshes under one observer: idle, busy, idle', async () => {
  const c = client();
  const states = [false, true, false];
  let calls = 0;
  globalThis.fetch = async () => {
    const busy = states[Math.min(calls++, states.length - 1)];
    return json({ workers: [{ id: 'w0', busy: busy ? 1 : 0 }], busy });
  };
  const options = engineWorkersOptions();
  const observer = new QueryObserver(c, options);
  const seen: boolean[] = [];
  const unsubscribe = observer.subscribe(r => { if (r.data && seen[seen.length - 1] !== r.data.busy) seen.push(r.data.busy); });
  // Timers are off without a window (tanstack treats Node as a server), so each
  // refetch below stands in for one interval tick on the same observer.
  for (let i = 0; i < 2; i++) { await tick(); await observer.refetch(); }
  await tick();
  unsubscribe();
  assert.deepEqual(seen, [false, true, false]);
  const interval = options.refetchInterval as (q: { state: { data: unknown; status: string; errorUpdateCount: number } }) => number | false;
  assert.equal(interval({ state: { data: { busy: false }, status: 'success', errorUpdateCount: 0 } }), ENGINE_WORKERS_POLL_MS, 'keeps polling while idle');
});

test('duplicated snapshots share one canonical key across areas', () => {
  assert.deepEqual(gpusOptions().queryKey, sharedKeys.gpus());
  assert.notDeepEqual(sharedKeys.workflowJob('j', 0), sharedKeys.workflowJob('j', 5), 'after filter stays in the key');
  assert.equal(typeof (sharedKeys.workflowJob('j')[2] as { session?: number }).session, 'number');
  // Every variant of a track reads one sidecar entry.
  assert.deepEqual(lrcOptions('/audio/x_mastered.wav?_t=1').queryKey, lrcOptions('/audio/x.wav').queryKey);
});

test('lyric sidecars: missing is cached as none, failures throw, crop/retranscribe invalidate', async () => {
  const c = client();
  globalThis.fetch = async input => {
    const url = String(input);
    if (url.includes('missing')) return new Response('', { status: 404 });
    if (url.includes('shell')) return new Response('<html>spa</html>', { status: 200 });
    if (url.includes('broken')) return new Response('', { status: 500 });
    return url.endsWith('.lrc') ? new Response('[00:01.00] Words', { status: 200 }) : json({ version: 1, lines: [] });
  };
  assert.equal(await c.query(lrcOptions('/audio/missing.wav')), null);
  assert.equal(await c.query(lyricsJsonOptions('/audio/shell.wav')), null);
  assert.equal(await c.query(lrcOptions('/audio/ok.wav')), '[00:01.00] Words');
  assert.deepEqual(await c.query(lyricsJsonOptions('/audio/ok.wav')), { version: 1, lines: [] });
  await assert.rejects(c.query({ ...lrcOptions('/audio/broken.wav'), retry: false }), /500/);
  await queryManager.sidecars.all(c);
  assert.equal(c.getQueryState(lrcOptions('/audio/ok.wav').queryKey)?.isInvalidated, true);
  assert.equal(c.getQueryState(lyricsJsonOptions('/audio/ok.wav').queryKey)?.isInvalidated, true);
});

test('ABC lead sheet: read from the base render, missing or SPA shell is none', async () => {
  const c = client();
  const requests: string[] = [];
  globalThis.fetch = async input => {
    const url = String(input);
    requests.push(url);
    if (url.includes('missing')) return new Response('', { status: 404 });
    if (url.includes('shell')) return new Response('<html>spa</html>', { status: 200, headers: { 'Content-Type': 'text/html' } });
    return new Response('X:1\nK:C\nCDEF|\n', { status: 200 });
  };
  assert.equal(await c.query(scoreAbcOptions('/audio/ok_mastered.wav?_t=1')), 'X:1\nK:C\nCDEF|');
  assert.equal(requests[0], '/audio/ok.score.abc');
  assert.equal(await c.query(scoreAbcOptions('/audio/missing.wav')), null);
  assert.equal(await c.query(scoreAbcOptions('/audio/shell.wav')), null);
});

test('peaks: concurrent reads share a request, failures are not cached, forget drops the entry', async () => {
  let calls = 0;
  let fail = true;
  globalThis.fetch = async () => {
    calls++;
    await tick();
    return fail ? json({ error: 'nope' }, 404) : json({ version: 1, duration: 1, sampleRate: 1, channels: 1, min: [0], max: [1] });
  };
  assert.equal(await fetchPeaks('/audio/p.wav'), null);
  assert.equal(peaksCached('/audio/p.wav'), null);
  fail = false;
  calls = 0;
  const [a, b] = await Promise.all([fetchPeaks('/audio/p.wav'), fetchPeaks('/audio/p.wav')]);
  assert.equal(calls, 1);
  assert.equal(a, b);
  assert.equal(peaksCached('/audio/p.wav')?.duration, 1);
  forgetPeaks('/audio/p.wav');
  assert.equal(peaksCached('/audio/p.wav'), null);
  await fetchPeaks('/audio/p.wav');
  assert.equal(calls, 2, 'the rewritten file is read again');
});
