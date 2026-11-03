// audioGenQueueStore.delivery.test.ts — a server-owned Create render reaches
// the library once. The server projection and the enqueue's own waiter both
// see it finish; between the `song-created` event and onSongCreated (both end
// in App's handleSongCreated) the song must be delivered exactly once, from
// one submission. See audioGenQueueStore.migration.test.ts for why the store
// is imported this way.
import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { createFakeIndexedDB, MemoryStorage, startProjectionForTests } from './audioQueueStoreTestFake.js';

register('./cssStubLoader.mjs', import.meta.url);
const songCreatedEvents: string[] = [];
Object.assign(globalThis, {
  window: {
    addEventListener() {}, removeEventListener() {},
    dispatchEvent(event: Event) {
      if (event.type === 'song-created') songCreatedEvents.push((event as CustomEvent).detail.song.id);
      return true;
    },
  },
  CustomEvent: class extends Event {
    detail: unknown;
    constructor(type: string, init?: { detail?: unknown }) { super(type); this.detail = init?.detail; }
  },
  localStorage: new MemoryStorage(),
  indexedDB: createFakeIndexedDB().indexedDB,
});

const {
  OWNER_KEY, enqueueSimpleGen, resumeQueue, refreshServerProjection,
  _queueItemsForTests, _resetAudioQueueForTests,
} = await import('./audioGenQueueStore.js');

const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });

test('server-owned render: one submission, and the finished song is delivered once', async () => {
  localStorage.setItem(OWNER_KEY, 'server');
  const server: Record<string, unknown>[] = [];
  const calls: string[] = [];
  const saved = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const path = new URL(url, 'http://local').pathname;
    calls.push(`${method} ${path}`);
    if (path === '/api/audio-queue/items' && method === 'POST') {
      const { idempotencyKey, request, meta } = JSON.parse(String(init!.body));
      const item = { id: 'srv1', idempotencyKey, status: 'submitted', jobId: 'job1', request, meta, createdAt: 1 };
      server.push(item);
      return ok({ item, created: true });
    }
    if (path === '/api/audio-queue/items') return ok({ items: server });
    if (path === '/api/generate/status/job1') return ok({ jobId: 'job1', status: 'running', progress: 50 });
    if (path === '/api/songs/song1') return ok({ song: { id: 'song1', title: 'Done', audio_url: '/audio/s.wav' } });
    return new Response(JSON.stringify({ error: 'no route' }), { status: 404 });
  }) as typeof fetch;
  _resetAudioQueueForTests([]);
  const delivered: string[] = [];
  try {
    await resumeQueue('t');
    await enqueueSimpleGen({ title: 'Done' } as never, 't', song => delivered.push(song.id));
    Object.assign(server[0], { status: 'succeeded', result: { songIds: ['song1'], audioUrls: ['/audio/s.wav'] } });
    await new Promise(resolve => setTimeout(resolve, 4000));
    assert.equal(calls.filter(c => c === 'POST /api/audio-queue/items').length, 1, 'submitted once');
    assert.deepEqual([...delivered, ...songCreatedEvents], ['song1'], 'delivered once, by either path');
    // The enqueue raced the projection's first read; the projection must still
    // pick the row up at the active cadence, not wait out the idle one.
    assert.equal(_queueItemsForTests().find(item => item.serverQueueId === 'srv1')?.status, 'succeeded');
  } finally {
    _resetAudioQueueForTests();
    globalThis.fetch = saved;
  }
});

test('a failed song read by the enqueue waiter does not stop the projection delivering it', async () => {
  localStorage.setItem(OWNER_KEY, 'server');
  songCreatedEvents.length = 0;
  const server: Record<string, unknown>[] = [];
  let holdStatus = false;
  let releaseStatus!: () => void;
  const statusHeld = new Promise<void>(resolve => { releaseStatus = resolve; });
  let songReads = 0;
  const saved = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const path = new URL(url, 'http://local').pathname;
    if (path === '/api/audio-queue/items' && method === 'POST') {
      const { idempotencyKey, request, meta } = JSON.parse(String(init!.body));
      const item = { id: 'srv1', idempotencyKey, status: 'submitted', jobId: 'job1', request, meta, createdAt: 1 };
      server.push(item);
      return ok({ item, created: true });
    }
    if (path === '/api/audio-queue/items') return ok({ items: server });
    if (path === '/api/generate/status/job1') {
      // Holding the status read keeps the projection busy mid-refresh, so
      // the enqueue waiter is the first to see the render finish.
      if (holdStatus) await statusHeld;
      return ok({ jobId: 'job1', status: 'running', progress: 50 });
    }
    if (path === '/api/songs/song1') {
      // The first read fails (a 4xx is never retried); later reads work.
      if (songReads++ === 0) return new Response(JSON.stringify({ error: 'busy' }), { status: 404 });
      return ok({ song: { id: 'song1', title: 'Done', audio_url: '/audio/s.wav' } });
    }
    return new Response(JSON.stringify({ error: 'no route' }), { status: 404 });
  }) as typeof fetch;
  _resetAudioQueueForTests([]);
  const delivered: string[] = [];
  const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
  const stop = startProjectionForTests(refreshServerProjection);
  try {
    await resumeQueue('t');
    await enqueueSimpleGen({ title: 'Done' } as never, 't', song => delivered.push(song.id));
    holdStatus = true;
    await wait(2000);                       // the projection's next tick is now stuck on status
    Object.assign(server[0], { status: 'succeeded', result: { songIds: ['song1'], audioUrls: ['/audio/s.wav'] } });
    await wait(3500);                       // the waiter sees it and its song read fails
    assert.equal(songReads, 1, 'the waiter read first and failed');
    assert.deepEqual([...delivered, ...songCreatedEvents], []);
    releaseStatus();                        // the projection catches up and delivers
    await wait(2500);
    assert.deepEqual([...delivered, ...songCreatedEvents], ['song1'], 'delivered once after recovery');
  } finally {
    releaseStatus();
    stop();
    _resetAudioQueueForTests();
    globalThis.fetch = saved;
  }
});
