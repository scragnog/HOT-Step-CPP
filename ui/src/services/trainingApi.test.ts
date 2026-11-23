// trainingApi.test.ts — the worker/local base split, as pure assertions.
// No UI test runner is wired up for this project; run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/services/trainingApi.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  cancelJob, finishYue2Ladders, getDataset, getJob, getWorkerMirror, jobStreamUrl, listDatasets,
  listYue2AitkRuns, listYue2BatchesLocal, listYue2JointPreviewsLocal, pickLadderRun, scoreYue2Rung,
  setTrainingWorker, syncWorkerMirror,
} from './trainingApi.js';
import { ApiError } from './httpClient.js';

function mockFetch(urls: string[]) {
  const real = globalThis.fetch;
  globalThis.fetch = (async (url: string) => {
    urls.push(String(url));
    return new Response(JSON.stringify({}), { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
  return () => { globalThis.fetch = real; };
}

test('with a worker selected, run records/scores/finish/previews hit the local base; job control hits the worker base', async () => {
  const urls: string[] = [];
  const restore = mockFetch(urls);
  try {
    setTrainingWorker('LivingRoom');
    await listYue2AitkRuns('ds1');
    await scoreYue2Rung('ds1', { refineRun: 'run1', step: 10 });
    await finishYue2Ladders([{ datasetId: 'ds1', refineRun: 'run1' }]);
    await listYue2BatchesLocal();
    await listYue2JointPreviewsLocal('ds1', 'run1');
    const localCalls = urls.length;
    assert.ok(urls.every(u => u.startsWith('/api/training/') && !u.includes('/api/workers/')), `expected only local-base calls, got ${JSON.stringify(urls)}`);

    await getJob('job1');
    assert.equal(urls.length, localCalls + 1);
    for (const u of urls.slice(localCalls)) assert.match(u, /^\/api\/workers\/LivingRoom\/api\/training\//);
  } finally { restore(); setTrainingWorker(null); }
});

test('mirror status and sync go to the local workers route, never through the worker proxy', async () => {
  const urls: string[] = [];
  const real = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    urls.push(`${init?.method ?? 'GET'} ${url}`);
    return new Response(JSON.stringify({ mirror: null }), { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
  try {
    setTrainingWorker('LivingRoom');
    await getWorkerMirror('LivingRoom');
    await syncWorkerMirror('LivingRoom');
    assert.deepEqual(urls, ['GET /api/workers/LivingRoom/mirror', 'POST /api/workers/LivingRoom/mirror']);
  } finally { globalThis.fetch = real; setTrainingWorker(null); }
});

test('pickLadderRun: an explicit pick always wins', () => {
  const runs = [{ jobId: 'old', live: false, createdAt: 1 }, { jobId: 'new', live: true, createdAt: 2 }];
  assert.equal(pickLadderRun(runs, 'old', 'new')?.jobId, 'old');
});

test('pickLadderRun: an unindexed active job shows live-only instead of falling back to older history', () => {
  const runs = [{ jobId: 'old', live: false, createdAt: 1 }];
  assert.equal(pickLadderRun(runs, undefined, 'unindexed-worker-job'), undefined);
});

test('pickLadderRun: preparation waits for the new run, while an explicit history pick remains available', () => {
  const runs = [{ jobId: 'old', live: false, createdAt: 1 }];
  assert.equal(pickLadderRun(runs, undefined, undefined, true), undefined);
  assert.equal(pickLadderRun(runs, 'old', undefined, true)?.jobId, 'old');
  const newRun = { jobId: 'new', live: true, createdAt: 2, checkpoints: [] };
  assert.equal(pickLadderRun([...runs, newRun], undefined, 'new', true), newRun);
});

test('pickLadderRun: an indexed active job, or else the live run, or else the newest, when nothing is picked', () => {
  const runs = [{ jobId: 'old', live: false, createdAt: 1 }, { jobId: 'active', live: false, createdAt: 2 }];
  assert.equal(pickLadderRun(runs, undefined, 'active')?.jobId, 'active');
  // No active job and no pick: falls back to the newest run with history.
  assert.equal(pickLadderRun(runs, undefined, undefined)?.jobId, 'active');
  const withLive = [{ jobId: 'old', live: false, createdAt: 1 }, { jobId: 'live', live: true, createdAt: 2 }];
  assert.equal(pickLadderRun(withLive, undefined, undefined)?.jobId, 'live');
});

test('a captured target overrides the current selection; reads pass their signal', async () => {
  const calls: Array<{ url: string; signal?: AbortSignal | null; method?: string }> = [];
  const real = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    calls.push({ url: String(url), signal: init?.signal, method: init?.method });
    return new Response(JSON.stringify({ datasets: [], jobs: [] }), { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
  try {
    setTrainingWorker('B');
    const ctl = new AbortController();
    await listDatasets({ worker: 'A', signal: ctl.signal });
    await listDatasets({ worker: null });
    await listDatasets();
    await cancelJob('j1', { worker: 'A' });
    await listYue2AitkRuns('ds1', { signal: ctl.signal });
    assert.deepEqual(calls.map(c => c.url), [
      '/api/workers/A/api/training/datasets',
      '/api/training/datasets',
      '/api/workers/B/api/training/datasets',
      '/api/workers/A/api/training/jobs/j1',
      '/api/training/datasets/ds1/yue2-joint-runs',
    ]);
    // P2: the transport composes the caller's signal with the instance timeout,
    // so fetch sees a composite — aborting the controller must abort it.
    ctl.abort();
    assert.equal(calls[0].signal?.aborted, true);
    assert.equal(calls[3].method, 'DELETE');
    assert.equal(calls[4].signal?.aborted, true);
    assert.equal(jobStreamUrl('j1', { worker: null }), '/api/training/jobs/j1/stream');
    assert.equal(jobStreamUrl('j1'), '/api/workers/B/api/training/jobs/j1/stream');
  } finally { globalThis.fetch = real; setTrainingWorker(null); }
});

test('a failed read throws ApiError with its status and the server message', async () => {
  const real = globalThis.fetch;
  globalThis.fetch = (async () => new Response(JSON.stringify({ error: 'no such dataset' }), { status: 404 })) as typeof fetch;
  try {
    await assert.rejects(getDataset('nope'), (e: unknown) =>
      e instanceof ApiError && e.status === 404 && e.message === 'no such dataset');
  } finally { globalThis.fetch = real; }
});
