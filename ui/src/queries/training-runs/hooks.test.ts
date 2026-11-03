import assert from 'node:assert/strict';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClientProvider } from '@tanstack/react-query';
import { createQueryClient } from '../client';
import { useJobSelection, useReload, useRunEvents } from './hooks';
import { runQueries } from './queries';
import type { TrainingJobSummary } from '../../services/trainingApi';
import { trainingQueries } from '../training-data/queries';

test('a standalone job bridge closes its shared source on a terminal frame without a chart', async () => {
  const saved = { window: globalThis.window, document: globalThis.document, EventSource: globalThis.EventSource };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  class Stream {
    static made: Stream[] = [];
    static CLOSED = 2; static OPEN = 1;
    readyState = 1;
    onmessage?: (event: { data: string }) => void;
    constructor() { Stream.made.push(this); }
    close() { this.readyState = 2; }
  }
  globalThis.EventSource = Stream as unknown as typeof EventSource;
  const client = createQueryClient();
  function Panel() { useRunEvents('bridge-only', null); return null; }
  const root = createRoot(document.getElementById('root')!);
  try {
    await act(async () => root.render(createElement(QueryClientProvider, { client }, createElement(Panel))));
    const stream = Stream.made[0];
    stream.onmessage?.({ data: JSON.stringify({ type: 'metric', metric: 'step', step: 1 }) });
    assert.equal(stream.readyState, Stream.OPEN);
    stream.onmessage?.({ data: JSON.stringify({ type: 'status', status: 'done' }) });
    assert.equal(stream.readyState, Stream.CLOSED);
  } finally {
    await act(async () => root.unmount()); client.clear();
    Object.assign(globalThis, saved);
  }
});

test('restored terminal jobs never stream; live jobs release their bridge on completion', async () => {
  const saved = { window: globalThis.window, document: globalThis.document,
    EventSource: globalThis.EventSource, fetch: globalThis.fetch };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  class Stream {
    static made: Stream[] = [];
    static CLOSED = 2; static OPEN = 1;
    readyState = 1;
    constructor() { Stream.made.push(this); }
    close() { this.readyState = 2; }
  }
  globalThis.EventSource = Stream as unknown as typeof EventSource;
  // Keep background revalidation pending so each transition below comes
  // from the canonical job cache, as it does after a server status refresh.
  globalThis.fetch = () => new Promise<Response>(() => {});
  const client = createQueryClient();
  let current: ReturnType<typeof useJobSelection>;
  function Panel() { current = useJobSelection(null, 'set'); return null; }
  const root = createRoot(document.getElementById('root')!);
  try {
    await act(async () => root.render(createElement(QueryClientProvider, { client }, createElement(Panel))));
    for (const status of ['done', 'failed', 'cancelled', 'interrupted'] as const) {
      await act(async () => current![1]({ id: status, status } as TrainingJobSummary));
      assert.equal(Stream.made.length, 0, status);
    }
    await act(async () => current![1]({ id: 'live', status: 'running' } as TrainingJobSummary));
    assert.equal(Stream.made.length, 1);
    await act(async () => {
      client.setQueryData(trainingQueries.job(null, 'live').queryKey, { id: 'live', status: 'done' });
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    assert.equal(Stream.made[0].readyState, Stream.CLOSED);
    await act(async () => current![1](null));
    await act(async () => current![1]({ id: 'live', status: 'done' } as TrainingJobSummary));
    assert.equal(Stream.made.length, 1, 'revisiting the finished job must not reconnect');
  } finally {
    await act(async () => root.unmount()); client.clear();
    Object.assign(globalThis, saved);
  }
});

test('a mounted panel rejects job adoption from a previous worker or dataset', async () => {
  const savedWindow = globalThis.window;
  const savedDocument = globalThis.document;
  const savedEventSource = globalThis.EventSource;
  const savedFetch = globalThis.fetch;
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  class Stream {
    static CLOSED = 2; static OPEN = 1;
    readyState = 1;
    close() { this.readyState = 2; }
  }
  globalThis.EventSource = Stream as unknown as typeof EventSource;
  const client = createQueryClient();
  let current: ReturnType<typeof useJobSelection>;
  function Panel({ worker, dataset }: { worker: string; dataset: string }) {
    current = useJobSelection(worker, dataset);
    return createElement('span', {}, current[0]?.id ?? 'none');
  }
  // The real /jobs/:id returns the summary itself, and the job query has
  // staleTime 0, so adopting a job immediately starts a refetch. A mock with
  // the wrong shape would replace the seeded summary with an envelope whose
  // `id` is undefined, and whether that landed before the assertion was a
  // timing race — the source of this test's intermittent failure.
  globalThis.fetch = async () => new Response(JSON.stringify({ id: 'B-job', status: 'done' }), {
    headers: { 'Content-Type': 'application/json' },
  });
  const root = createRoot(document.getElementById('root')!);
  const render = async (worker: string, dataset: string) => {
    await act(async () => { root.render(createElement(QueryClientProvider, { client }, createElement(Panel, { worker, dataset }))); });
  };
  try {
    await render('A', 'one');
    const adoptA = current![1];
    await render('B', 'one');
    await act(async () => { adoptA({ id: 'A-job', status: 'running' } as TrainingJobSummary); });
    assert.equal(document.querySelector('span')?.textContent, 'none');
    const adoptB = current![1];
    await render('B', 'two');
    await act(async () => { adoptB({ id: 'old-dataset', status: 'done' } as TrainingJobSummary); });
    assert.equal(document.querySelector('span')?.textContent, 'none');
    await render('A', 'one');
    await act(async () => { adoptA({ id: 'before-round-trip', status: 'running' } as TrainingJobSummary); });
    assert.equal(document.querySelector('span')?.textContent, 'none');
    await render('B', 'two');
    await act(async () => { current![1]({ id: 'B-job', status: 'done' } as TrainingJobSummary); });
    assert.equal(document.querySelector('span')?.textContent, 'B-job');
    assert.equal(client.getQueryCache().findAll().some(q => q.queryKey[3] === 'A-job'), false);
  } finally {
    await act(async () => root.unmount()); client.clear();
    Object.assign(globalThis, { window: savedWindow, document: savedDocument, EventSource: savedEventSource, fetch: savedFetch });
  }
});

test('reload revisions invalidate on change and reuse fresh data on remount', async () => {
  const savedWindow = globalThis.window;
  const savedDocument = globalThis.document;
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  const client = createQueryClient();
  const key = runQueries.mm3Runs('set').queryKey;
  client.setQueryData(key, { runs: [], busy: false });
  function Panel({ revision }: { revision: number }) { useReload(key, revision); return null; }
  const root = createRoot(document.getElementById('root')!);
  try {
    await act(async () => root.render(createElement(QueryClientProvider, { client }, createElement(Panel, { revision: 1 }))));
    assert.equal(client.getQueryState(key)?.isInvalidated, false);
    await act(async () => root.render(createElement(QueryClientProvider, { client }, createElement(Panel, { revision: 2 }))));
    assert.equal(client.getQueryState(key)?.isInvalidated, true);
  } finally {
    await act(async () => root.unmount()); client.clear();
    Object.assign(globalThis, { window: savedWindow, document: savedDocument });
  }
});
