import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { QueryObserver } from '@tanstack/react-query';
import { createQueryClient } from '../client';
import { trainingQueries } from '../training-data/queries';
import { setTrainingWorker } from '../../services/trainingApi';
import { pollWhile } from '../contract';
import { runQueries, trainingWritePlan, activeJob } from './queries';
import { gpusOptions } from '../shell-queue/settings';
import { invalidateRunReads } from './events';

const originalFetch = globalThis.fetch;
const clients: ReturnType<typeof createQueryClient>[] = [];
const client = () => { const value = createQueryClient(); clients.push(value); return value; };
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
afterEach(() => {
  globalThis.fetch = originalFetch;
  setTrainingWorker(null);
  clients.splice(0).forEach(value => value.clear());
});
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json' },
});

test('run GPU and preparation reads join other areas and refresh after commands and events', async () => {
  const cache = client();
  let calls = 0;
  globalThis.fetch = async () => {
    calls++; await tick();
    return json({ gpus: [], pipeline: { status: 'running' }, job: { status: 'running' } });
  };
  const pairs = [
    [runQueries.gpus(), gpusOptions()],
    [runQueries.preparation('chain'), trainingQueries.preparation('chain')],
    [runQueries.preparationJob('chain'), trainingQueries.preparationJob('chain')],
  ] as const;
  for (const [one, two] of pairs) {
    await Promise.all([cache.query(one), cache.query(two)]);
  }
  assert.equal(calls, 3);
  await trainingWritePlan(cache, 'remote-worker');
  for (const [one] of pairs.slice(1)) {
    assert.equal(cache.getQueryState(one.queryKey)?.isInvalidated, true);
    cache.setQueryData(one.queryKey, { status: 'running' });
  }
  await invalidateRunReads(cache, 'remote-worker');
  for (const [one] of pairs.slice(1)) assert.equal(cache.getQueryState(one.queryKey)?.isInvalidated, true);
  assert.equal(cache.getQueryState(runQueries.gpus().queryKey)?.isInvalidated, false);
});

test('two status consumers share one read, including a fresh remount', async () => {
  const cache = client();
  let count = 0;
  globalThis.fetch = async () => { count++; await tick(); return json({ ready: true }); };
  const options = runQueries.mm3Status(null, 'set');
  const [one, two] = await Promise.all([cache.query(options), cache.query(options)]);
  assert.equal(one, two);
  assert.equal(count, 1);
  await cache.query(runQueries.mm3Status(null, 'set'));
  assert.equal(count, 1);
});

test('a delayed worker A read stays in A cache after worker B is selected', async () => {
  const cache = client();
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  const urls: string[] = [];
  globalThis.fetch = async input => {
    const url = String(input); urls.push(url);
    if (url.includes('/A/')) await held;
    return json({ file: url });
  };
  const a = cache.query(runQueries.optimise('A', 'set'));
  setTrainingWorker('B');
  const b = await cache.query(runQueries.optimise('B', 'set'));
  release(); await a;
  assert.match(b.file, /\/B\//);
  assert.equal(cache.getQueryData(runQueries.optimise('B', 'set').queryKey), b);
  assert.match(urls[0], /\/A\//);
});

test('unsubscribing a panel consumes cancellation without cancelling the server job', async () => {
  const cache = client();
  const calls: string[] = [];
  let aborted = false;
  globalThis.fetch = async (input, init) => {
    calls.push(`${init?.method ?? 'GET'} ${input}`);
    await new Promise((_, reject) => init?.signal?.addEventListener('abort', () => {
      aborted = true; reject(new DOMException('aborted', 'AbortError'));
    }));
    return json({});
  };
  const observer = new QueryObserver(cache, runQueries.arStatus('A', 'set'));
  const stop = observer.subscribe(() => {});
  await tick(); stop(); await tick();
  assert.equal(aborted, true);
  assert.deepEqual(calls, ['GET /api/workers/A/api/training/datasets/set/yue2-ar']);
  assert.equal(cache.getQueryState(runQueries.arStatus('A', 'set').queryKey)?.status, 'pending');
});

test('local run/review indexes stay local while a remote worker is selected', async () => {
  const cache = client(); setTrainingWorker('A');
  const urls: string[] = [];
  globalThis.fetch = async input => { urls.push(String(input)); return json({ runs: [], rows: [], scores: [], score: null, batches: [] }); };
  await Promise.all([
    cache.query(runQueries.mm3Runs('set')), cache.query(runQueries.narRuns('set')),
    cache.query(runQueries.arRuns('set')), cache.query(runQueries.jointRuns('set')),
    cache.query(runQueries.previews('set', 'run')), cache.query(runQueries.review()),
    cache.query(runQueries.ladder('set', 'run')), cache.query(runQueries.scores('set', 'run')),
    cache.query(runQueries.album('set', 'run')), cache.query(runQueries.localBatches()),
  ]);
  assert.ok(urls.every(url => url.startsWith('/api/training/')));
  assert.ok(urls.includes('/api/training/ops/review/ladder/set/run'));
});

test('terminal jobs stop polling; failing active reads use capped backoff', () => {
  const interval = pollWhile(activeJob);
  const query = { state: { data: { status: 'running' }, status: 'success', errorUpdateCount: 0 } };
  assert.equal(interval(query as never), 1500);
  query.state.status = 'error'; query.state.errorUpdateCount = 1;
  assert.equal(interval(query as never), 3000);
  query.state.errorUpdateCount = 20;
  assert.equal(interval(query as never), 30_000);
  for (const status of ['done', 'failed', 'cancelled']) {
    query.state.data.status = status;
    assert.equal(interval(query as never), false);
  }
});

test('a write plan invalidates its captured worker scope and shared reads', async () => {
  const cache = client();
  const a = runQueries.arStatus('A', 'set');
  const b = runQueries.arStatus('B', 'set');
  cache.setQueryData(a.queryKey, {} as never);
  cache.setQueryData(b.queryKey, {} as never);
  cache.setQueryData(trainingQueries.datasets('A').queryKey, []);
  cache.setQueryData(runQueries.review().queryKey, { rows: [] });
  await trainingWritePlan(cache, 'A');
  assert.equal(cache.getQueryState(a.queryKey)?.isInvalidated, true);
  assert.equal(cache.getQueryState(b.queryKey)?.isInvalidated, false);
  assert.equal(cache.getQueryState(trainingQueries.datasets('A').queryKey)?.isInvalidated, true);
  assert.equal(cache.getQueryState(runQueries.review().queryKey)?.isInvalidated, true);
});

test('review ladder cancellation reaches the local operation transport', async () => {
  const cache = client(); let aborted = false;
  globalThis.fetch = async (_input, init) => {
    await new Promise((_, reject) => init?.signal?.addEventListener('abort', () => {
      aborted = true; reject(new DOMException('aborted', 'AbortError'));
    }));
    return json({});
  };
  const options = runQueries.ladder('set', 'run');
  const pending = cache.query(options).catch(() => undefined);
  await tick(); await cache.cancelQueries({ queryKey: options.queryKey }); await pending;
  assert.equal(aborted, true);
});

test('a write plan refreshes mounted consumers', async () => {
  const cache = client();
  let reads = 0;
  globalThis.fetch = async () => json({ ready: ++reads });
  const options = runQueries.mm3Status('A', 'set');
  const observer = new QueryObserver(cache, options);
  const stop = observer.subscribe(() => {});
  try {
    await tick();
    assert.equal(reads, 1);
    await trainingWritePlan(cache, 'A');
    assert.equal(reads, 2);
    assert.equal((cache.getQueryData(options.queryKey) as unknown as { ready: number }).ready, 2);
  } finally { stop(); }
});


test('terminal SSE replay invalidates once, reconnect refreshes scope, teardown ignores late frames', async () => {
  const { bridgeRunEvents } = await import('./events');
  const cache = client();
  const a = runQueries.arStatus('A', 'set');
  const b = runQueries.arStatus('B', 'set');
  cache.setQueryData(a.queryKey, {} as never); cache.setQueryData(b.queryKey, {} as never);
  let handlers: any; let stopped = 0;
  const stop = bridgeRunEvents(cache, 'job', 'A', (_url, callbacks) => {
    handlers = callbacks; return () => { stopped++; };
  });
  const terminal = JSON.stringify({ type: 'status', status: 'done' });
  handlers.message(terminal);
  assert.equal(cache.getQueryState(a.queryKey)?.isInvalidated, true);
  assert.equal(cache.getQueryState(b.queryKey)?.isInvalidated, false);
  cache.setQueryData(a.queryKey, {} as never);
  handlers.message(terminal);
  assert.equal(cache.getQueryState(a.queryKey)?.isInvalidated, false);
  handlers.error(); handlers.open();
  await new Promise(resolve => setTimeout(resolve, 70));
  assert.equal(cache.getQueryState(a.queryKey)?.isInvalidated, true);
  assert.equal(cache.getQueryState(b.queryKey)?.isInvalidated, false);
  cache.setQueryData(a.queryKey, {} as never);
  stop(); handlers.message(terminal); handlers.error(); handlers.open();
  await new Promise(resolve => setTimeout(resolve, 70));
  assert.equal(stopped, 1);
  assert.equal(cache.getQueryState(a.queryKey)?.isInvalidated, false);
});

test('run stream frames merge into the job cache and never blank omitted fields', async () => {
  const { bridgeRunEvents } = await import('./events');
  const cache = client();
  const jobKey = trainingQueries.job('A', 'job1').queryKey;
  cache.setQueryData(jobKey, {
    id: 'job1', status: 'queued', done: 0, total: 10, failed: 0,
    phase: '', currentSampleId: null, engineQueueDepth: 0, error: null,
  } as never);
  let handlers: { message?: (data: string) => void } = {};
  const stop = bridgeRunEvents(cache, 'job1', 'A', (_url, callbacks) => {
    handlers = callbacks; return () => {};
  });
  try {
    handlers.message?.(JSON.stringify({
      type: 'progress', done: 3, total: 10, failed: 0,
      phase: 'preprocess', currentSampleId: 's1', engineQueueDepth: 2,
    }));
    let job = cache.getQueryData(jobKey) as Record<string, unknown>;
    assert.equal(job.status, 'running', 'progress implies running');
    assert.equal(job.done, 3);
    assert.equal(job.phase, 'preprocess');

    handlers.message?.(JSON.stringify({ type: 'job', job: { id: 'job1', status: 'running', done: 4 } }));
    job = cache.getQueryData(jobKey) as Record<string, unknown>;
    assert.equal(job.done, 4);
    assert.equal(job.phase, 'preprocess', 'a job frame leaves fields it omits intact');

    handlers.message?.(JSON.stringify({ type: 'status', status: 'failed', error: 'boom' }));
    job = cache.getQueryData(jobKey) as Record<string, unknown>;
    assert.equal(job.status, 'failed');
    assert.equal(job.error, 'boom');
  } finally { stop(); }
});
