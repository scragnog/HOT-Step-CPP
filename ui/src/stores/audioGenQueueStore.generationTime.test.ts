// audioGenQueueStore.generationTime.test.ts — a completed song keeps showing
// real generation time next to the track duration, in both queue modes.
//
// Moving the queue onto the server (a1997989) added
// refreshServerProjection/projectServerItem but never taught them to set
// `elapsed`, so every server-owned row lost its generation-time badge
// (InlineAudioQueue.tsx only renders it when elapsed > 0). Cover's Node
// workflow render (89ccc119) had the same gap in its own browser-owned
// follower, _followWorkflowRender.
//
// The first fix for this ran its own client-side timer against whatever
// job.status a poll happened to observe. Review caught the real bug that left
// in: a job that starts and finishes between two polls is never observed as
// "running" at all, so a client-side clock can never start for it — the
// badge stays missing exactly as before. The actual fix moved the clock onto
// the server (intentQueue's follow() loop, server/src/services/audioQueue/
// intentQueue.elapsed.test.ts), which runs on its own interval regardless of
// whether any browser is polling and so cannot miss the transition. These
// tests check the client side of that: it must display whatever
// elapsedSeconds the server reports, verbatim, including when that number
// appears already-frozen on the very first poll that ever sees the item.
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

test('server-owned queue: the row shows the server\'s elapsedSeconds live, and keeps it when the job finishes between polls', async () => {
  localStorage.setItem(OWNER_KEY, 'server');
  const api = new AudioQueueApiFake();
  api.serverItems = [{ id: 'srv1', status: 'submitted', jobId: 'job1', request: { title: 'Cover' },
    meta: { source: 'cover-studio', workflowJobId: 'wf1' }, createdAt: 1, elapsedSeconds: null }];
  const uninstall = api.install();
  const queue = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    if (String(url).includes('/generate/status/job1')) return ok({ jobId: 'job1', status: 'running', progress: 40 });
    return queue(url, init);
  }) as typeof fetch;
  _resetAudioQueueForTests([]);
  const stop = startProjectionForTests(refreshServerProjection);
  try {
    await resumeQueue('t');
    await settle();
    const row = () => _queueItemsForTests().find(item => item.serverQueueId === 'srv1')!;
    assert.ok(!row().elapsed, 'still queued server-side (elapsedSeconds: null); nothing to show yet');

    // The server is now counting real generation time; the client just mirrors it.
    api.serverItems[0].elapsedSeconds = 5;
    await settle(1700);
    assert.equal(row().elapsed, 5);

    // The job finished entirely between two client polls: status jumps
    // straight to 'succeeded' with an already-frozen elapsedSeconds, no
    // intermediate 'running' observation ever happened client-side. This is
    // exactly the case a client-side timer could not handle.
    api.serverItems[0].status = 'succeeded';
    api.serverItems[0].elapsedSeconds = 8;
    (api.serverItems[0] as any).result = { audioUrls: ['/audio/a.wav'], durations: [12] };
    await settle(1700);
    assert.equal(row().elapsed, 8, 'the badge survives a job that finished between polls');

    // Further polls must not grow or recompute it on the client.
    await settle(1700);
    assert.equal(row().elapsed, 8);
  } finally { stop(); _resetAudioQueueForTests(); uninstall(); }
});

test('browser-owned queue: a workflow-backed Cover row reads the server\'s elapsedSeconds, not a client-side clock', async () => {
  localStorage.setItem(OWNER_KEY, 'browser');
  const api = new AudioQueueApiFake();
  const uninstall = api.install();
  const queue = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    if (String(url).includes('/generate/status/job1')) return ok({ jobId: 'job1', status: 'running', progress: 40 });
    if (String(url).includes('/workflows/jobs/wf1')) return ok({ job: { id: 'wf1', status: 'running', audioIntentIds: ['srv1'] }, events: [], gap: false });
    return queue(url, init);
  }) as typeof fetch;
  // Already shows 7s the very first time the UI ever asks — there was no
  // client-side observation of the job running at all.
  api.serverItems = [{ id: 'srv1', status: 'submitted', jobId: 'job1', request: {}, createdAt: 1, elapsedSeconds: 7 }];
  _resetAudioQueueForTests([]);
  try {
    const id = addManualQueueItem({ title: 'Cover' });
    updateManualQueueItem(id, { workflowJob: { id: 'wf1', token: 't' }, stage: 'Generating…' });
    await settle(1700);
    const row = _queueItemsForTests().find(item => item.id === id)!;
    assert.equal(row.elapsed, 7);
  } finally { _resetAudioQueueForTests(); uninstall(); }
});
