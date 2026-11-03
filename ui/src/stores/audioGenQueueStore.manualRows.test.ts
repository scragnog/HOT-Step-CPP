// audioGenQueueStore.manualRows.test.ts — a studio's own queue row survives a
// server-owned queue's refresh until the server has its render.
//
// Cover Studio adds its row in the browser (addManualQueueItem). A cover
// workflow queued behind another has no server audio item until the first
// finishes, and refreshServerProjection replaced the whole list with the
// server's every 1.5 s, so the queued row vanished and only came back as the
// server's item once it started.
//
// See audioGenQueueStore.progress.test.ts for why the store is imported this
// way and the shared test fixture shape.
import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { createFakeIndexedDB, AudioQueueApiFake, MemoryStorage, startProjectionForTests } from './audioQueueStoreTestFake.js';

register('./cssStubLoader.mjs', import.meta.url);
Object.assign(globalThis, {
  window: { addEventListener() {}, removeEventListener() {} },
  localStorage: new MemoryStorage(),
  indexedDB: createFakeIndexedDB().indexedDB,
});

const {
  OWNER_KEY, addManualQueueItem, updateManualQueueItem, resumeQueue, refreshServerProjection,
  _queueItemsForTests, _resetAudioQueueForTests,
} = await import('./audioGenQueueStore.js');

const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
const settle = (ms = 300) => new Promise(resolve => setTimeout(resolve, ms));

test('server-owned queue: a queued studio row stays until the server item for its workflow replaces it', async () => {
  localStorage.setItem(OWNER_KEY, 'server');
  const api = new AudioQueueApiFake();
  // The first cover is rendering; the second's workflow has no server item yet.
  api.serverItems = [{ id: 'srv1', status: 'submitted', jobId: 'job1', request: { title: 'First' },
    meta: { source: 'cover-studio', workflowJobId: 'wf1', workflowKey: 'render' }, createdAt: 1 }];
  const uninstall = api.install();
  const queue = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    if (String(url).includes('/generate/status/')) return ok({ status: 'running', progress: 40 });
    return queue(url, init);
  }) as typeof fetch;
  _resetAudioQueueForTests([]);
  const stop = startProjectionForTests(refreshServerProjection);
  try {
    await resumeQueue('t');
    const id = addManualQueueItem({ title: 'Second' });
    updateManualQueueItem(id, { workflowJob: { id: 'wf2', token: 't' }, stage: 'Queued…', status: 'pending' });
    await settle(1700);
    const second = _queueItemsForTests().find(item => item.id === id);
    assert.equal(second?.status, 'pending', 'the queued row is still there after a refresh');

    api.serverItems.push({ id: 'srv2', status: 'pending', jobId: null, request: { title: 'Second' },
      meta: { source: 'cover-studio', workflowJobId: 'wf2', workflowKey: 'render' }, createdAt: 2 });
    await settle(1700);
    const rows = _queueItemsForTests();
    assert.equal(rows.some(item => item.id === id), false, 'the server item replaces the studio row');
    assert.equal(rows.filter(item => item.serverQueueId === 'srv2').length, 1);
  } finally {
    stop();
    _resetAudioQueueForTests([]);
    globalThis.fetch = queue;
    uninstall();
  }
});
