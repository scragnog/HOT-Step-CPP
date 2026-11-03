import assert from 'node:assert/strict';
import { register } from 'node:module';
import test from 'node:test';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { parseHTML } from 'linkedom';
import { QueryClientProvider } from '@tanstack/react-query';
import { createQueryClient } from '../../queries/client';
import { trainingQueries } from '../../queries/training-data/queries';
import type { TrainingJobSummary } from '../../services/trainingApi';

// The card imports CSS, which the node loader cannot. Stub it for this file.
register('data:text/javascript,' + encodeURIComponent(
  'export async function load(url, ctx, next) { if (url.endsWith(".css")) return { format: "module", source: "export default {}", shortCircuit: true }; return next(url, ctx); }'));
Object.assign(globalThis, { React: await import('react') });
const { createPresetLoadGate, baselineFor, Yue2AitkTrainCard } = await import('./Yue2AitkTrainCard');

async function mountChart(status: TrainingJobSummary['status'], cache?: unknown, replayStatus = 200) {
  const saved = { window: globalThis.window, document: globalThis.document,
    EventSource: globalThis.EventSource, fetch: globalThis.fetch, localStorage: globalThis.localStorage };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  const stored = new Map<string, string>([['hs-yue2-aitk-job:chart-set', JSON.stringify('chart-job')]]);
  if (cache) stored.set('hs-yue2-aitk-job:chart-set:metrics:chart-job', JSON.stringify(cache));
  const storage = { getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => { stored.set(key, value); },
    removeItem: (key: string) => { stored.delete(key); } };
  Object.assign(window, { localStorage: storage });
  Object.assign(globalThis, { window, document, localStorage: storage, IS_REACT_ACT_ENVIRONMENT: true });
  class Stream {
    static made: Stream[] = [];
    static CLOSED = 2; static OPEN = 1;
    readyState = 1;
    onmessage?: (event: { data: string }) => void;
    onopen?: () => void;
    onerror?: () => void;
    constructor() { Stream.made.push(this); }
    close() { this.readyState = 2; }
    send(event: unknown) { this.onmessage?.({ data: JSON.stringify(event) }); }
  }
  globalThis.EventSource = Stream as unknown as typeof EventSource;
  const job = { id: 'chart-job', datasetId: 'chart-set', kind: 'yue2-joint-train', status,
    createdAt: 1, startedAt: 1, finishedAt: null, total: 10, done: 0, failed: 0, phase: 'train' };
  let replayRequests = 0;
  globalThis.fetch = async input => {
    if (String(input).endsWith('/stream')) {
      replayRequests++;
      if (replayStatus !== 200) return new Response(null, { status: replayStatus });
      const stream = new Stream();
      return new Response(new ReadableStream({
        start(controller) {
          stream.onmessage = event => controller.enqueue(new TextEncoder().encode(`data: ${event.data}\n\n`));
        },
        cancel() { stream.close(); },
      }));
    }
    return new Response(JSON.stringify(String(input).includes('/jobs/chart-job') ? job
    : { runs: [], previews: [], bases: [], defaults: null, jobs: [job], activeJob: null }), {
    headers: { 'Content-Type': 'application/json' },
  });
  };
  const client = createQueryClient();
  const root = createRoot(document.getElementById('root')!);
  const render = async (key: string) => {
    await act(async () => {
      root.render(createElement(QueryClientProvider, { client }, createElement(Yue2AitkTrainCard, {
        key, datasetId: 'chart-set', lyricTiming: false, onLyricTimingChange: () => {},
      })));
      await new Promise(resolve => setTimeout(resolve, 10));
    });
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
  };
  return { Stream, client, stored, document, render, replayRequests: () => replayRequests,
    cleanup: async () => {
      await act(async () => root.unmount()); client.clear();
      Object.assign(globalThis, saved);
    } };
}

test('a finished job without a chart cache replays once and the next visit uses its saved chart', async () => {
  const view = await mountChart('done');
  try {
    await view.render('first');
    assert.equal(view.Stream.made.length, 1, 'uncached terminal history needs one replay');
    const stream = view.Stream.made[0];
    await act(async () => {
      stream.send({ type: 'metric', metric: 'step', step: 10, loss: 0.25, ts: 10 });
      stream.send({ type: 'status', status: 'done' });
    });
    assert.equal(stream.readyState, view.Stream.CLOSED);
    assert.match(view.document.body.textContent ?? '', /step 10/);
    const cached = JSON.parse(view.stored.get('hs-yue2-aitk-job:chart-set:metrics:chart-job')!);
    assert.equal(cached.status, 'done');
    assert.equal(cached.steps[0].step, 10);
    await view.render('revisit');
    assert.equal(view.Stream.made.length, 1, 'complete cache must suppress replay');
    assert.match(view.document.body.textContent ?? '', /Step 10 \/ 10/, 'cached chart remains visible');
  } finally { await view.cleanup(); }
});

test('leaving a partial terminal-job replay keeps its cache incomplete', async () => {
  const view = await mountChart('done');
  try {
    await view.render('first');
    await act(async () => {
      view.Stream.made[0].send({ type: 'metric', metric: 'step', step: 3, loss: 0.5, ts: 10 });
    });
    await view.render('revisit');
    const cached = JSON.parse(view.stored.get('hs-yue2-aitk-job:chart-set:metrics:chart-job')!);
    assert.equal(cached.status, 'running');
    assert.equal(view.Stream.made.length, 2, 'partial replay must resume on the next visit');
  } finally { await view.cleanup(); }
});

test('a terminal stream 404 is final and never persists a complete chart', async () => {
  const view = await mountChart('done', undefined, 404);
  try {
    await view.render('first');
    assert.equal(view.replayRequests(), 1);
    assert.equal(view.Stream.made.length, 0, 'terminal replay must not create a reconnecting EventSource');
    assert.equal(view.stored.has('hs-yue2-aitk-job:chart-set:metrics:chart-job'), false);
    assert.match(view.document.body.textContent ?? '', /Training history for this run is no longer available/);
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 20)); });
    assert.equal(view.replayRequests(), 1, '404 must not retry');
  } finally { await view.cleanup(); }
});

test('terminal and legacy complete chart caches skip replay; a partial cache still replays', async () => {
  for (const status of [undefined, 'done', 'running']) {
    const view = await mountChart('done', { status, steps: [{ step: 3, ep: 3, loss: 0.5 }] });
    try {
      await view.render('cached');
      assert.equal(view.Stream.made.length, status === 'running' ? 1 : 0, String(status));
      assert.match(view.document.body.textContent ?? '', /Step 3 \/ 10/);
    } finally { await view.cleanup(); }
  }
});

test('a live job keeps its final metric and failure log when polling sees completion', async () => {
  const view = await mountChart('running');
  try {
    await view.render('live');
    const stream = view.Stream.made[0];
    assert.ok(stream);
    await act(async () => {
      stream.send({ type: 'metric', metric: 'step', step: 7, loss: 0.4, ts: 10 });
      stream.send({ type: 'log', level: 'error', message: 'training failure detail', ts: 10 });
      stream.send({ type: 'status', status: 'failed' });
      await view.client.cancelQueries({ queryKey: trainingQueries.job(null, 'chart-job').queryKey });
      view.client.setQueryData(trainingQueries.job(null, 'chart-job').queryKey,
        (previous: TrainingJobSummary) => ({ ...previous, status: 'failed' }));
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    assert.match(view.document.body.textContent ?? '', /training failure detail/);
    assert.match(view.document.body.textContent ?? '', /step 7.*loss 0\.4000/);
    assert.equal(view.Stream.made.length, 1);
  } finally { await view.cleanup(); }
});

test('a delayed preset response cannot submit the old form through batch or exposed Start', async () => {
  const gate = createPresetLoadGate();
  let resolveResponse!: () => void;
  const response = new Promise<void>(resolve => { resolveResponse = resolve; });
  let commitForm!: () => void;
  const committed = new Promise<void>(resolve => { commitForm = resolve; });
  let form = { steps: 50 };
  const submitted: number[] = [];
  const startBatch = () => { if (!gate.pending) submitted.push(form.steps); };
  const exposedStart = async () => gate.capture(() => { submitted.push(form.steps); });

  const load = (async () => {
    assert.equal(gate.begin(), true);
    await response;
    form = { steps: 200 };
    await committed;
    gate.release();
  })();

  startBatch();
  await assert.rejects(exposedStart(), /Wait for the preset/);
  assert.deepEqual(submitted, []);

  resolveResponse();
  await response;
  startBatch();
  await assert.rejects(exposedStart(), /Wait for the preset/);
  assert.deepEqual(submitted, []);

  commitForm();
  await load;
  startBatch();
  await exposedStart();
  assert.deepEqual(submitted, [200, 200]);
});

test('DoRA and LoRA baselines carry their own alpha and KL stop, LoKr keeps the recipe defaults (#318)', () => {
  for (const adapterType of ['dora', 'lora'] as const) {
    const base = baselineFor({ adapterType, rank: 64 });
    assert.equal(base.alpha, 64);
    assert.equal(base.targetKl, 1.4);
    assert.equal(base.plannerLrScale, 0.3);
  }
  const lokr = baselineFor({ adapterType: 'lokr', rank: 64 });
  assert.equal(lokr.targetKl, 1.0);
  assert.equal(lokr.plannerLrScale, 0.6);
});
