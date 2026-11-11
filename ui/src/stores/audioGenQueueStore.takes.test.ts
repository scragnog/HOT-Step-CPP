// audioGenQueueStore.takes.test.ts — a server-owned multi-song render (MM3
// Variations Per Render) shows one queue row per song, boxed together like a
// coalesced YuE2 batch, and keeps them across refreshes and a reload. See
// audioGenQueueStore.migration.test.ts for why the store is imported this way.
import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { createFakeIndexedDB, MemoryStorage } from './audioQueueStoreTestFake.js';

register('./cssStubLoader.mjs', import.meta.url);
Object.assign(globalThis, {
  window: { addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; } },
  CustomEvent: class extends Event {
    detail: unknown;
    constructor(type: string, init?: { detail?: unknown }) { super(type); this.detail = init?.detail; }
  },
  localStorage: new MemoryStorage(),
  indexedDB: createFakeIndexedDB().indexedDB,
});

const {
  OWNER_KEY, applyServerItems, clearFinishedFromAudioQueue, resumeQueue, _queueItemsForTests, _resetAudioQueueForTests,
} = await import('./audioGenQueueStore.js');

const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });

test('MM3 variations: one row per take, boxed, kept across refreshes and a reload', async () => {
  localStorage.setItem(OWNER_KEY, 'server');
  const saved = globalThis.fetch;
  const deletes: string[] = [];
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://local').pathname;
    if (init?.method === 'DELETE') { deletes.push(path); return new Response(null, { status: 204 }); }
    if (path === '/api/audio-queue/items') return ok({ items: [server] });
    if (path === '/api/generate/status/job1') {
      return ok({ jobId: 'job1', status: 'running', progress: 40, stage: 'Planning', mm3_takes: 3, mm3_take_seeds: ['7', '8', '9'] });
    }
    if (path.startsWith('/api/songs/')) return ok({ song: { id: path.split('/').pop() } });
    return new Response('{}', { status: 404 });
  }) as typeof fetch;
  const view = { id: 'aq-1', generation: { id: 0, title: 'Song' }, globalParams: {}, status: 'pending' };
  const server = { id: 'srv1', idempotencyKey: 'aq-1', status: 'submitted', jobId: 'job1', request: {}, meta: { view }, createdAt: 1 } as Record<string, any>;
  _resetAudioQueueForTests([]);
  try {
    const rows = () => _queueItemsForTests();
    for (let tick = 0; tick < 3; tick++) {
      await applyServerItems([server as never]);
      assert.deepEqual(rows().map(r => r.id), ['aq-1', 'aq-1:take1', 'aq-1:take2'], `three rows on refresh ${tick}`);
    }
    assert.ok(rows().every(r => r.batchId === 'job1' && r.status === 'generating'), 'one box, all rendering');
    assert.deepEqual(rows().map(r => r.mm3TakeSeed), ['7', '8', '9']);

    Object.assign(server, { status: 'succeeded', result: { songIds: ['s0', 's1', 's2'], audioUrls: ['/a0', '/a1', '/a2'], durations: [60, 61, 62] } });
    await applyServerItems([server as never]);
    assert.deepEqual(rows().map(r => [r.status, r.songId, r.audioUrl, r.audioDuration]),
      [['succeeded', 's0', '/a0', 60], ['succeeded', 's1', '/a1', 61], ['succeeded', 's2', '/a2', 62]]);

    _resetAudioQueueForTests([]);               // a reload: no previous rows
    await applyServerItems([server as never]);
    assert.deepEqual(rows().map(r => r.songId), ['s0', 's1', 's2'], 'reload keeps one row per song');
    assert.ok(rows().every(r => r.batchId === 'job1'));

    // Clear finished: the render's one server item is dismissed once, not
    // once per take row (the rest 404'd and each raised a toast).
    await resumeQueue('t');
    clearFinishedFromAudioQueue();
    await new Promise(resolve => setTimeout(resolve, 100));
    assert.deepEqual(deletes, ['/api/audio-queue/items/srv1']);
  } finally {
    _resetAudioQueueForTests();
    globalThis.fetch = saved;
  }
});
