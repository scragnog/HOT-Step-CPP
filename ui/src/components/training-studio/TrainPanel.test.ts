import assert from 'node:assert/strict';
import test from 'node:test';
import { register } from 'node:module';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { parseHTML } from 'linkedom';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { HEARTMULA_PREVIEW_DEFAULTS, HEARTMULA_RECIPE_DEFAULTS } from '../../../../server/src/contracts/heartmulaTraining';
import type { HeartmulaStatusResponse } from '../../../../server/src/contracts/heartmulaTraining';
import type { TrainingJobSummary } from '../../services/trainingApi';

register('data:text/javascript,' + encodeURIComponent(
  'export async function load(url, ctx, next) { if (url.endsWith(".css")) return { format: "module", source: "export default {}", shortCircuit: true }; return next(url, ctx); }'));
Object.assign(globalThis, {
  React: await import('react'),
  localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
});
const { useTrainingStore } = await import('../../stores/trainingStore');
const { heartmulaStatusOptions } = await import('./useHeartmulaController');
const { TrainPanel } = await import('./TrainPanel');

await i18n.use(initReactI18next).init({ lng: 'en', resources: {}, interpolation: { escapeValue: false } });

test('the phase host only streams active jobs and releases both subscribers when training finishes', async () => {
  const saved = { window: globalThis.window, document: globalThis.document, EventSource: globalThis.EventSource };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(window, { localStorage: globalThis.localStorage });
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  class Stream {
    static made: Stream[] = [];
    static OPEN = 1; static CLOSED = 2;
    readyState = 1;
    constructor() { Stream.made.push(this); }
    close() { this.readyState = 2; }
  }
  globalThis.EventSource = Stream as unknown as typeof EventSource;
  const original = useTrainingStore.getState();
  const client = new QueryClient();
  const root = createRoot(document.getElementById('root')!);
  try {
    useTrainingStore.setState({ detail: null, datasets: [], trainingWorker: null,
      activeJob: { id: 'finished', status: 'done' } as TrainingJobSummary });
    await act(async () => root.render(createElement(QueryClientProvider, { client },
      createElement(TrainPanel, { target: 'heartmula' }))));
    assert.equal(Stream.made.length, 0);
    await act(async () => useTrainingStore.setState({ activeJob: { id: 'live-panel', status: 'running' } as TrainingJobSummary }));
    assert.equal(Stream.made.length, 1, 'the live reducer and bridge share a socket');
    await act(async () => useTrainingStore.setState({ activeJob: { id: 'live-panel', status: 'done' } as TrainingJobSummary }));
    assert.equal(Stream.made[0].readyState, Stream.CLOSED);
    await act(async () => root.render(createElement(QueryClientProvider, { client },
      createElement(TrainPanel, { target: 'heartmula', key: 'revisit' }))));
    assert.equal(Stream.made.length, 1);
  } finally {
    await act(async () => root.unmount()); client.clear();
    useTrainingStore.setState(original);
    Object.assign(globalThis, saved);
  }
});

test('HeartMuLa Prepare and Train reach their cards independently of ACE capabilities and manifests', () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  const initial = useTrainingStore.getInitialState();
  const original = { ...initial };
  const status = {
    ready: { available: true, backend: 'cuda', missingModels: [], encoderMissing: false },
    prepare: { cacheValid: true, cacheVersion: 1, preparedSamples: 4, staleSamples: 0, skipped: [], contextLimit: 8192, preparedAt: null },
    run: null, runs: [], adapter: null,
    defaults: HEARTMULA_RECIPE_DEFAULTS, previewDefaults: HEARTMULA_PREVIEW_DEFAULTS,
  } as HeartmulaStatusResponse;
  client.setQueryData(heartmulaStatusOptions(null, 'ds-1', false).queryKey, status);
  // Match the existing shell tests: SSR reads Zustand's initial snapshot.
  Object.assign(initial, { selectedDatasetId: 'ds-1', trainingWorker: null, activeJob: null, error: null });
  const render = (section: 'prepare' | 'train') => renderToStaticMarkup(createElement(QueryClientProvider, { client },
    createElement(TrainPanel, { target: 'heartmula', section })));
  try {
    for (const capabilities of [null, { trainLm: { available: false } }, { trainLm: { available: true } }]) {
      for (const built of [false, true]) {
        Object.assign(initial, { capabilities, detail: {
          id: 'ds-1', slug: 'example-set', customTag: '',
          builtAt: built ? '2026-10-26' : '', datasetJsonPath: built ? 'dataset.json' : '',
        } });
        const prepare = render('prepare');
        assert.match(prepare, /Prepare HeartMuLa audio/);
        assert.match(prepare, /4 prepared/);
        assert.doesNotMatch(prepare, /Backbone LoRA training is selected/);
        const train = render('train');
        assert.match(train, /Backbone LoRA training is selected/);
        assert.match(train, /Start training/);
        assert.doesNotMatch(train, /Prepare HeartMuLa audio/);
      }
    }
  } finally {
    Object.assign(initial, original);
    client.clear();
  }
});
