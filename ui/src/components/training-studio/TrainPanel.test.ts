import assert from 'node:assert/strict';
import test from 'node:test';
import { register } from 'node:module';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { HEARTMULA_PREVIEW_DEFAULTS, HEARTMULA_RECIPE_DEFAULTS } from '../../../../server/src/contracts/heartmulaTraining';
import type { HeartmulaStatusResponse } from '../../../../server/src/contracts/heartmulaTraining';

register('data:text/javascript,' + encodeURIComponent(
  'export async function load(url, ctx, next) { if (url.endsWith(".css")) return { format: "module", source: "export default {}", shortCircuit: true }; return next(url, ctx); }'));
Object.assign(globalThis, {
  React: await import('react'),
  localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
});
const { useTrainingStore } = await import('../../stores/trainingStore');
const { trainingQueries } = await import('../../queries/training-data/queries');
const { TrainPanel } = await import('./TrainPanel');

await i18n.use(initReactI18next).init({ lng: 'en', resources: {}, interpolation: { escapeValue: false } });

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
  client.setQueryData(trainingQueries.heartmulaStatus(null, 'ds-1', false).queryKey, status);
  // Match the existing shell tests: SSR reads Zustand's initial snapshot.
  Object.assign(initial, { selectedDatasetId: 'ds-1', trainingWorker: null, activeJobId: null, error: null });
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
