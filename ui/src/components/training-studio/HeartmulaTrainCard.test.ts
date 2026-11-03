import assert from 'node:assert/strict';
import test from 'node:test';
import { register } from 'node:module';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderToStaticMarkup } from 'react-dom/server';
import { parseHTML } from 'linkedom';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { HEARTMULA_PREVIEW_DEFAULTS, HEARTMULA_RECIPE_DEFAULTS, type HeartmulaStatusResponse } from '../../../../server/src/contracts/heartmulaTraining';

register('data:text/javascript,' + encodeURIComponent(
  'export async function load(url, ctx, next) { if (url.endsWith(".css")) return { format: "module", source: "export default {}", shortCircuit: true }; return next(url, ctx); }'));
Object.assign(globalThis, { React: await import('react'), localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} } });
const { HeartmulaTrainCard } = await import('./HeartmulaTrainCard');
const { HeartmulaPrepareCard } = await import('./HeartmulaPrepareCard');
const { useTrainingStore } = await import('../../stores/trainingStore');
const { createHeartmulaPlugin } = await import('./plugins/heartmula');
const { useHeartmulaController } = await import('./useHeartmulaController');
const { trainingQueries } = await import('../../queries/training-data/queries');
const { TRAINING_TARGETS } = await import('./trainingTargets');
const { heartmulaChips, heartmulaReadiness } = await import('./plugins/chips');
const testClient = () => new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
type Element = ReturnType<typeof createElement>;
const renderWithClient = (element: Element, client = testClient()) =>
  renderToStaticMarkup(createElement(QueryClientProvider, { client }, element));
await i18n.use(initReactI18next).init({ lng: 'en', resources: { en: { translation: {
  'trainingStudio.job.running': '{{done}}/{{total}} · {{phase}}',
  'trainingStudio.job.phaseEncoding': 'Encoding audio',
} } }, interpolation: { escapeValue: false } });

const status: HeartmulaStatusResponse = {
  ready: { available: true, backend: 'cuda', missingModels: [], deviceProblem: null, encoderMissing: false, encoderHelp: null },
  prepare: { cacheValid: true, cacheVersion: 1, preparedSamples: 4, staleSamples: 0, skipped: [], contextLimit: 8192, preparedAt: null },
  base: '3B', runs: [], run: null, adapter: null, defaults: HEARTMULA_RECIPE_DEFAULTS, previewDefaults: HEARTMULA_PREVIEW_DEFAULTS,
};

test('HeartMuLa recipe and start stay visible, with only optional settings folded', () => {
  const { document } = parseHTML(renderWithClient(createElement(HeartmulaTrainCard, {
    status, slug: 'example', busy: false, trainStarted: false, onTrain() {}, onCheckpoint() {}, onCancel() {},
  })));
  const fields = [...document.querySelectorAll('input[type="number"]')];
  assert.equal(fields.filter(field => !field.closest('details')).length, 6);
  // Preview cadence follows 'Save every', so the previews section has no step field of its own.
  assert.equal(fields.filter(field => field.closest('details')).length, 4);
  // the preview section matches YuE2's: dataset-sourced prompt, a full-song length, no fixed fixture
  const text0 = document.documentElement.textContent;
  for (const label of ['Preview maximum seconds', 'Preview seed', 'Preview lyrics', 'Tags override', 'Lyrics override', 'Lyric Studio generation']) assert.ok(text0.includes(label), label);
  assert.doesNotMatch(text0, /Fixed fixture|30-second/);
  const text = document.documentElement.textContent;
  for (const label of ['HOT-Approved', 'Fast', 'Balanced', 'Thorough', 'Save current settings', 'Changed from defaults']) assert.ok(text.includes(label), label);
  assert.doesNotMatch(text, /Checkpoint ladder|Train HeartMuLa adapter/);
  const start = [...document.querySelectorAll('button')].find(button => button.textContent.includes('Start training'))!;
  assert.ok(start);
  assert.equal(start.closest('details'), null);
  assert.equal(start.hasAttribute('disabled'), false);
});

test('HeartMuLa Train follows the preset and settings order and hides old run metrics', () => {
  const render = (state: 'running' | 'done', updatedAt: number) => renderWithClient(createElement(HeartmulaTrainCard, {
    status: { ...status, run: { run: 'example/one', state, updatedAt, step: 1, totalSteps: 480, loss: 2,
      lossHistory: [{ step: 1, loss: 2 }], msPerStep: 1000, peakVramMiB: 10000, checkpointStep: 1,
      adapterFile: null, error: null, startedAt: 0 } },
    slug: 'example', busy: false, trainStarted: false, onTrain() {}, onCheckpoint() {}, onCancel() {},
  }));
  const old = render('done', 0);
  const labels = ['Resume a previous run', 'HOT-Approved', 'Your presets', 'Changed from defaults', 'Base model',
    'Training GPU', 'Adapter type', 'Updates', 'Advanced', 'Checkpoint previews (optional)', 'heartmula-lm-adapters/', 'Start training'];
  let position = -1;
  for (const label of labels) {
    const next = old.indexOf(label, position + 1);
    assert.ok(next > position, `Section out of order or missing: ${label}`); position = next;
  }
  assert.doesNotMatch(old, /Peak VRAM|Checkpoint ladder/);
  assert.match(render('running', 0), /Peak VRAM/);
  // ETA on a live run: 479 steps at 1 s plus 5 previews (100..400 and the final) at the base take's 60 s.
  const live = renderWithClient(createElement(HeartmulaTrainCard, {
    status: { ...status, run: { run: 'example/one', state: 'running', updatedAt: 0, step: 1, totalSteps: 480, loss: 2,
      lossHistory: [{ step: 1, loss: 2 }], msPerStep: 1000, peakVramMiB: 10000, checkpointStep: 1, adapterFile: null, error: null, startedAt: 0 } },
    previews: { run: 'example/one', settings: { ...HEARTMULA_PREVIEW_DEFAULTS, everySteps: 100 }, source: null,
      previews: [{ id: 'b', run: 'example/one', kind: 'base', step: 0, settings: HEARTMULA_PREVIEW_DEFAULTS, status: 'done', audioUrl: null,
        adapter: null, adapterSha256: null, renderMs: 60_000, pauseMs: null, error: null, createdAt: 0 }] },
    slug: 'example', busy: true, trainStarted: true, onTrain() {}, onCheckpoint() {}, onCancel() {},
  }));
  assert.match(live, /ETA: 13 min \(incl\. 5 preview\(s\)\)/);
  assert.match(render('done', Date.now()), /Peak VRAM/);
});

test('HeartMuLa Prepare shows live per-track progress outside the settings fold and rejects another dataset job', () => {
  const initial = useTrainingStore.getInitialState();
  const saved = { ...initial };
  try {
    Object.assign(initial, {
      activeJobId: 'prepare',
      jobLog: [{ ts: 1, level: 'info', message: 'Encoding track three' }], samplesById: {},
    });
    const client = testClient();
    client.setQueryData(trainingQueries.job(null, 'prepare').queryKey, {
      id: 'prepare', kind: 'heartmula-prepare', datasetId: 'one', status: 'running', phase: 'encoding',
      done: 2, total: 4, failed: 0, currentSampleId: 'track-three.wav', engineQueueDepth: 0,
    });
    const render = (datasetId: string) => renderWithClient(createElement(HeartmulaPrepareCard, {
      datasetId, status: status.prepare, busy: true, preparing: true, onPrepare() {},
    }), client);
    const html = render('one');
    assert.match(html, /track-three.wav/);
    assert.match(html, /2 \/ 4/);
    assert.match(html, /Encoding track three/);
    assert.match(html, /Encoding audio/);
    const { document } = parseHTML(html);
    const log = [...document.querySelectorAll('div')].find(node => node.textContent === 'Encoding track three')!;
    assert.equal(log.closest('details'), null);
    assert.doesNotMatch(render('two'), /track-three.wav|Encoding track three/);
  } finally { Object.assign(initial, saved); }
});

test('a rejected older-run preview retry stays visible after a successful list refetch and belongs to that run', async () => {
  const saved = { window: globalThis.window, document: globalThis.document, fetch: globalThis.fetch };
  const savedStore = useTrainingStore.getState();
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  useTrainingStore.setState({ trainingWorker: null, activeJobId: null, error: null });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const runs = ['older', 'latest'].map(run => ({ run, adapterName: `${run} run`, recipe: HEARTMULA_RECIPE_DEFAULTS,
    state: 'failed' as const, step: 20, resumable: true, published: null }));
  const snapshot = { ...status, runs, run: { run: 'latest', state: 'failed', step: 20, totalSteps: 480,
    loss: null, lossHistory: [], msPerStep: null, peakVramMiB: null, checkpointStep: 20, adapterFile: null,
    error: null, startedAt: null, updatedAt: 0 } };
  let olderReads = 0;
  let rejectedRun = '';
  let failList = false;
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    const respond = (body: unknown, code = 200) => new Response(JSON.stringify(body), { status: code, headers: { 'Content-Type': 'application/json' } });
    if (init?.method === 'POST') {
      rejectedRun = JSON.parse(String(init.body)).run;
      return respond({ error: 'Preview retry rejected' }, 409);
    }
    if (url.includes('heartmula-status')) return respond(snapshot);
    if (url.endsWith('/yue2-batch')) return respond({ batches: [] });
    if (url.includes('heartmula-previews')) {
      const run = new URL(url, 'http://localhost').searchParams.get('run');
      if (run === 'older') {
        olderReads++;
        if (failList) return respond({ error: 'Preview list unavailable' }, 400);
      }
      return respond({ run, settings: HEARTMULA_PREVIEW_DEFAULTS, previews: run === 'older' ? [{
        run, kind: 'step', step: 20, settings: HEARTMULA_PREVIEW_DEFAULTS, status: 'failed', audioUrl: null,
        adapter: null, adapterSha256: null, renderMs: null, pauseMs: null, error: 'Original take failed', createdAt: 1,
      }] : [] });
    }
    throw new Error(`Unexpected request: ${url}`);
  };
  const root = createRoot(document.getElementById('root')!);
  const plugin = createHeartmulaPlugin({ target: { ...TRAINING_TARGETS.heartmula, chips: heartmulaChips, readiness: heartmulaReadiness }, useController: useHeartmulaController });
  const click = async (text: string) => {
    const button = [...document.querySelectorAll('button')].find(node => node.textContent.includes(text));
    assert.ok(button, `Missing button: ${text}`);
    await act(async () => { button.click(); });
  };
  const until = async (predicate: () => boolean) => {
    for (let attempt = 0; attempt < 50 && !predicate(); attempt++) await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
    assert.ok(predicate(), document.body.textContent ?? 'No rendered content');
  };
  try {
    await act(async () => { root.render(createElement(QueryClientProvider, { client },
      plugin.renderReview!({ datasetId: 'one', slug: 'example', trigger: '', section: 'review' }))); });
    await until(() => document.body.textContent!.includes('older run'));
    await click('older run');
    await until(() => document.body.textContent!.includes('Retry preview'));
    await click('Retry preview');
    await until(() => olderReads >= 2 && rejectedRun === 'older' && !client.isFetching());
    assert.match(document.querySelector('[role="alert"]')?.textContent ?? '', /Preview retry rejected/);
    await click('latest run');
    assert.doesNotMatch(document.body.textContent!, /Preview retry rejected/);
    await click('older run');
    assert.match(document.body.textContent!, /Preview retry rejected/);
    failList = true;
    await act(async () => { await client.invalidateQueries({ queryKey: trainingQueries.heartmulaPreviews(null, 'one', 'older', false).queryKey }); });
    await until(() => document.body.textContent!.includes('Preview list unavailable'));
    assert.match(document.body.textContent!, /Preview retry rejected/);
    // the Train page lists the run's previews under the card, as YuE2's does
    await act(async () => { root.render(createElement(QueryClientProvider, { client },
      plugin.renderTrain!({ datasetId: 'one', slug: 'example', trigger: '', section: 'train' }))); });
    await until(() => document.body.textContent!.includes('Checkpoint ladder'));
    assert.ok(document.body.textContent!.indexOf('Start training') < document.body.textContent!.indexOf('Checkpoint ladder'));
  } finally {
    await act(async () => { root.unmount(); });
    client.clear(); useTrainingStore.setState(savedStore); Object.assign(globalThis, saved);
  }
});

test('Perform all stages accepts stale preparation and a pasted resume path names its run', async () => {
  const saved = { window: globalThis.window, document: globalThis.document };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  const root = createRoot(document.getElementById('root')!);
  let request: unknown;
  const run = 'example/2026-10-26_12-00-00';
  try {
    await act(async () => root.render(createElement(QueryClientProvider, { client: testClient() }, createElement(HeartmulaTrainCard, {
      status: { ...status, prepare: { ...status.prepare, cacheValid: false, staleSamples: 1 },
        runs: [{ run, adapterName: 'example', state: 'failed', step: 1, recipe: HEARTMULA_RECIPE_DEFAULTS, resumable: true, published: null }] },
      slug: 'example', busy: false, trainStarted: false, onTrain() {}, onChain: value => { request = value; }, onCheckpoint() {}, onCancel() {},
    }))));
    const chain = [...document.querySelectorAll('button')].find(button => button.textContent === 'Perform all stages')!;
    assert.equal(chain.hasAttribute('disabled'), false);
    await act(async () => chain.click());
    const { steps: _steps, ...automaticRecipe } = HEARTMULA_RECIPE_DEFAULTS;
    assert.deepEqual((request as { recipe: unknown }).recipe, automaticRecipe, 'each prepared dataset resolves its own default updates');
    const input = document.querySelector('input[placeholder="Or paste a run folder (optional)"]')!;
    const props = (input as any)[Object.keys(input).find(key => key.startsWith('__reactProps'))!];
    await act(async () => props.onChange({ target: { value: `D:/saved/${run}` } }));
    assert.match(document.body.textContent!, new RegExp(`Resume ${run} with its saved recipe`));
    await act(async () => chain.click());
    assert.deepEqual(request, { resumeRun: run });
  } finally { await act(async () => root.unmount()); Object.assign(globalThis, saved); }
});

test('default updates follow accumulation, while an explicit update edit survives submission', async () => {
  const saved = { window: globalThis.window, document: globalThis.document };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  const root = createRoot(document.getElementById('root')!);
  let request: any;
  const numeric = () => [...document.querySelectorAll('input[type="number"]')];
  const change = async (input: Element, value: number) => {
    const props = (input as any)[Object.keys(input).find(key => key.startsWith('__reactProps'))!];
    await act(async () => props.onChange({ target: { value: String(value), valueAsNumber: value } }));
  };
  const start = async () => {
    const button = [...document.querySelectorAll('button')].find(button => button.textContent!.includes('Start training'))!;
    await act(async () => button.click());
  };
  try {
    await act(async () => root.render(createElement(QueryClientProvider, { client: testClient() }, createElement(HeartmulaTrainCard, {
      status: { ...status, prepare: { ...status.prepare, preparedSamples: 5 } }, slug: 'example', busy: false,
      trainStarted: false, onTrain: value => { request = value; }, onCheckpoint() {}, onCancel() {},
    }))));
    assert.equal(numeric()[0].getAttribute('value'), '60');
    await start();
    assert.equal(request.recipe.steps, undefined);
    const accum = numeric().find(input => input.getAttribute('value') === '4')!;
    await change(accum, 1);
    assert.equal(numeric()[0].getAttribute('value'), '150');
    await change(numeric()[0], 777);
    await change(accum, 8);
    assert.equal(numeric()[0].getAttribute('value'), '777');
    await start();
    assert.equal(request.recipe.steps, 777);
    assert.equal(request.recipe.gradAccum, 8);
    await act(async () => [...document.querySelectorAll('button')].find(button => button.textContent === 'Reset all to defaults')!.click());
    assert.equal(numeric()[0].getAttribute('value'), '60');
    await start();
    assert.equal(request.recipe.steps, undefined);
  } finally { await act(async () => root.unmount()); Object.assign(globalThis, saved); }
});
