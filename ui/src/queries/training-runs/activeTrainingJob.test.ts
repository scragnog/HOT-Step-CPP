// activeTrainingJob.test.ts — stage (c): components read the job from the one
// Query cache entry, and the store keeps only the selected id.

import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClientProvider } from '@tanstack/react-query';
import { createQueryClient } from '../client';
import { trainingQueries } from '../training-data/queries';
import { useTrainingStore } from '../../stores/trainingStore';
import { useActiveTrainingJob } from './hooks';

// The store module reads localStorage at import time in the browser; the Node
// test runner has none.
Object.assign(globalThis, { localStorage: { getItem: () => null, setItem() {}, removeItem() {} } });

const Probe = () => createElement('span', null, useActiveTrainingJob()?.status ?? 'none');

// SSR reads Zustand's getInitialState snapshot, so seed that object the same
// way the component tests do and restore it after each case.
function withSelectedJob(id: string | null, run: () => void) {
  const initial = useTrainingStore.getInitialState();
  const saved = { activeJobId: initial.activeJobId, trainingWorker: initial.trainingWorker };
  Object.assign(initial, { trainingWorker: null, activeJobId: id });
  try { run(); } finally { Object.assign(initial, saved); }
}

afterEach(() => {
  useTrainingStore.setState({ trainingWorker: null, activeJobId: null });
});

test('the hook reads the selected id from the cache entry', () => {
  const client = createQueryClient();
  client.setQueryData(trainingQueries.job(null, 'j1').queryKey, { id: 'j1', status: 'running' } as never);
  let html = '';
  withSelectedJob('j1', () => {
    html = renderToStaticMarkup(
      createElement(QueryClientProvider, { client }, createElement(Probe)),
    );
  });
  assert.match(html, /running/, 'the cache is the only job summary');
  client.clear();
});

test('no cache entry means no job, even with an id', () => {
  const client = createQueryClient();
  let html = '';
  withSelectedJob('j2', () => {
    html = renderToStaticMarkup(
      createElement(QueryClientProvider, { client }, createElement(Probe)),
    );
  });
  assert.match(html, /none/, 'the store no longer falls back to a copy');
  client.clear();
});
