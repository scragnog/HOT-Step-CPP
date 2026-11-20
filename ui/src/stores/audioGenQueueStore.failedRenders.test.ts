// audioGenQueueStore.failedRenders.test.ts — a Lyric Studio render that failed
// before reaching the audio queue shows as a failed row with its error, read
// from its lyric-render job, so it survives a reload; Retry submits its intent
// again (the server recaptures it) and Clear Done dismisses it for good. See
// audioGenQueueStore.migration.test.ts for why the store is imported this way.
import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { createFakeIndexedDB, MemoryStorage } from './audioQueueStoreTestFake.js';

register('./cssStubLoader.mjs', import.meta.url);
Object.assign(globalThis, {
  window: { addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; }, location: { href: 'http://localhost' } },
  CustomEvent: class extends Event {},
  localStorage: new MemoryStorage(),
  indexedDB: createFakeIndexedDB().indexedDB,
});

const {
  OWNER_KEY, resumeQueue, refreshFailedRenders, retryFailedInAudioQueue, clearFinishedFromAudioQueue,
  _queueItemsForTests, _resetAudioQueueForTests,
} = await import('./audioGenQueueStore.js');

const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
const intent = (generationId: number) => ({ kind: 'written-song', engine: 'ace', generationId, lyricsSetId: 7, params: { seed: generationId }, settings: {} });
const view = (title: string) => ({ generation: { title }, artistName: 'Fixture Artist', lyricsSetId: 7 });
const job = (id: string, extra: Record<string, unknown>) => ({
  id, kind: 'lyric-render', idempotencyKey: id, attempt: 1, cancelRequested: false, lastSeq: 0,
  createdAt: 1, startedAt: 1, finishedAt: 2, updatedAt: 2, error: null, audioIntentIds: [], ...extra,
});

test('a render that never reached the queue is a failed row that survives reload; Retry recaptures it', async () => {
  localStorage.setItem(OWNER_KEY, 'server');
  const jobs = [
    // One single render whose preset changed before its job ran.
    job('wf-stale', { status: 'succeeded',
      input: { items: [{ type: 'render', intent: intent(1), sourceRevision: 'r', view: view('Stale Song') }] },
      result: { results: [{ index: 0, status: 'error', error: 'Written song or preset changed since submission', unqueued: true }] } }),
    // A render that reached the queue and failed there: its audio row shows it, not this.
    job('wf-audio', { status: 'succeeded', audioIntentIds: ['srv1'],
      input: { items: [{ type: 'render', intent: intent(2), sourceRevision: 'r', view: view('Audio Song') }] },
      result: { results: [{ index: 0, status: 'error', error: 'Audio failed' }] } }),
    // A job interrupted by a restart before it queued anything.
    job('wf-cut', { status: 'interrupted', error: 'The server stopped while this job was running.',
      input: { items: [{ type: 'render', intent: intent(3), sourceRevision: 'r', view: view('Cut Song') }] }, result: null }),
  ];
  const resubmitted: unknown[] = [];
  const saved = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const u = new URL(url, 'http://local');
    if (u.pathname === '/api/audio-queue/items') return ok({ items: [] });
    if (u.pathname === '/api/workflows/jobs' && u.searchParams.get('kind') === 'lyric-render') return ok({ jobs });
    if (u.pathname === '/api/lireek/workflow-renders' && method === 'POST') {
      resubmitted.push(...JSON.parse(String(init!.body)).intents);
      return ok({ job: job('wf-new', { status: 'pending', input: {}, result: null }), created: true });
    }
    return new Response(JSON.stringify({ error: 'no route' }), { status: 404 });
  }) as typeof fetch;
  _resetAudioQueueForTests([]);
  const failedRows = () => _queueItemsForTests().filter(row => row.failedRender);
  try {
    await resumeQueue('t');
    await refreshFailedRenders();
    assert.deepEqual(failedRows().map(r => [r.generation.title, r.status, r.error]), [
      ['Stale Song', 'failed', 'Written song or preset changed since submission'],
      ['Cut Song', 'failed', 'The server stopped while this job was running.'],
    ]);
    assert.equal(failedRows()[0].artistName, 'Fixture Artist');

    // Reload: the rows come back from the jobs.
    _resetAudioQueueForTests([]);
    await resumeQueue('t');
    await refreshFailedRenders();
    assert.equal(failedRows().length, 2);

    // Retry submits the captured intents again and drops the old rows.
    assert.equal(retryFailedInAudioQueue(), 2);
    await new Promise(resolve => setTimeout(resolve, 50));
    assert.deepEqual(resubmitted.map(i => (i as { generationId: number }).generationId), [1, 3]);
    await refreshFailedRenders();
    assert.equal(failedRows().length, 0, 'retried rows stay dismissed after a refresh');

    // A new failure appears; Clear Done dismisses it for good.
    jobs.push(job('wf-later', { status: 'succeeded',
      input: { items: [{ type: 'preflight-error', error: 'Generation 9 not found', intent: intent(9) }] },
      result: { results: [{ index: 0, status: 'error', error: 'Generation 9 not found', unqueued: true }] } }));
    await refreshFailedRenders();
    assert.deepEqual(failedRows().map(r => r.generation.title), ['Written song 9']);
    clearFinishedFromAudioQueue();
    _resetAudioQueueForTests([]);
    await refreshFailedRenders();
    assert.equal(failedRows().length, 0);
  } finally {
    _resetAudioQueueForTests();
    globalThis.fetch = saved;
  }
});

test('a refused Retry keeps the rows, a double click submits once, and more than 200 rows go up in batches', async () => {
  localStorage.setItem(OWNER_KEY, 'server');
  const many = Array.from({ length: 201 }, (_, i) => i + 100);
  const jobs = [job('wf-many', { status: 'succeeded',
    input: { items: many.map(g => ({ type: 'render', intent: intent(g), sourceRevision: 'r', view: view(`Song ${g}`) })) },
    result: { results: many.map((_, index) => ({ index, status: 'error', error: 'Stale', unqueued: true })) } })];
  let refuse = true;
  let release!: () => void;
  let held: Promise<void> = Promise.resolve();
  const posts: number[] = [];
  const saved = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    const u = new URL(url, 'http://local');
    if (u.pathname === '/api/audio-queue/items') return ok({ items: [] });
    if (u.pathname === '/api/workflows/jobs' && u.searchParams.get('kind') === 'lyric-render') return ok({ jobs });
    if (u.pathname === '/api/lireek/workflow-renders' && method === 'POST') {
      posts.push(JSON.parse(String(init!.body)).intents.length);
      await held;
      if (refuse) return new Response(JSON.stringify({ error: 'Batch must contain 1 to 200 renders' }), { status: 400 });
      return ok({ job: job(`wf-retry-${posts.length}`, { status: 'pending', input: {}, result: null }), created: true });
    }
    return new Response(JSON.stringify({ error: 'no route' }), { status: 404 });
  }) as typeof fetch;
  _resetAudioQueueForTests([]);
  const failedRows = () => _queueItemsForTests().filter(row => row.failedRender);
  const settle = () => new Promise(resolve => setTimeout(resolve, 50));
  try {
    await resumeQueue('t');
    await refreshFailedRenders();
    assert.equal(failedRows().length, 201);

    // Refused: nothing was queued, so every row stays, after a reload too.
    retryFailedInAudioQueue();
    await settle();
    assert.deepEqual(posts, [200, 1], 'the route limit is respected');
    assert.equal(failedRows().length, 201);
    _resetAudioQueueForTests([]);
    await resumeQueue('t');
    await refreshFailedRenders();
    assert.equal(failedRows().length, 201, 'a refused Retry dismissed nothing');

    // Accepted, with a second click while the first is in flight.
    refuse = false;
    posts.length = 0;
    held = new Promise(resolve => { release = resolve; });
    assert.equal(retryFailedInAudioQueue(), 201);
    assert.equal(retryFailedInAudioQueue(), 0, 'rows already being resubmitted are skipped');
    release();
    await settle();
    assert.deepEqual(posts, [200, 1]);
    await refreshFailedRenders();
    assert.equal(failedRows().length, 0);
  } finally {
    _resetAudioQueueForTests();
    globalThis.fetch = saved;
  }
});
