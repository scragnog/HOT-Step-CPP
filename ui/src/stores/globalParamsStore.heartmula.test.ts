import assert from 'node:assert/strict';
import test from 'node:test';
import { register } from 'node:module';

register('data:text/javascript,' + encodeURIComponent(
  'export async function load(url, ctx, next) { if (url.endsWith(".css")) return { format: "module", source: "export default {}", shortCircuit: true }; return next(url, ctx); }'));
const storage = new Map<string, string>();
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
  getItem: (key: string) => storage.get(key) ?? null,
  setItem: (key: string, value: string) => storage.set(key, value),
} });
const { useGlobalParamsStore } = await import('./globalParamsStore');
const { useBackendStore } = await import('./backendStore');
const { createContentParams, createContentFromDraft } = await import('../components/create/createContent');
const { generateApi } = await import('../services/api');
const { queryClient } = await import('../queries/client');

test('HeartMuLa requests freeze components, preserve lyrics, cap duration/batch and cancel the submitted job after switching', async () => {
  const fetchBefore = globalThis.fetch;
  const calls: { url: string; body: Record<string, unknown> }[] = [];
  globalThis.fetch = async (input, init) => {
    calls.push({ url: String(input), body: JSON.parse(String(init?.body ?? '{}')) });
    return Response.json(String(input).endsWith('/generate') ? { jobId: 'heartmula-job', status: 'pending' } : { success: true });
  };
  try {
    useBackendStore.setState({ activeBackendId: 'heartmula', capabilities: {
      heartmula: { core: { duration: { max: 240 }, batch: { max: 1 } } } as never,
    }, models: { heartmula: {
      backend: 'heartmula', selectable: true, buckets: {},
      defaults: { variant: '3B', global: 'Q8_0', local: 'F16', flow: 'Q8_0', scalar_decoder: 'F16', lmAdapter: '' },
    } } });
    const gp = useGlobalParamsStore.getState();
    gp.setBackendParam('heartmulaAdapterStrength', 0);
    gp.setBackendParam('heartmulaTemperature', 0.75);
    gp.setSeed(73);
    gp.setRandomSeed(false);
    useGlobalParamsStore.setState({ batchSize: 4 });
    const content = createContentFromDraft({ 'hs-caption': 'folk, acoustic', 'hs-lyrics': '[verse]\nA quiet morning', 'hs-duration': 600 });
    const request = { ...gp.getGlobalParams(), ...createContentParams(content, { mm3Mode: false, durationMax: 240 }) };
    assert.equal(request.heartmulaGlobalModel, 'Q8_0');
    assert.equal(request.heartmulaAdapter, '');
    assert.equal(request.heartmulaAdapterStrength, 0);
    assert.equal(request.duration, 240);
    assert.equal(request.batchSize, 1);
    assert.equal(request.seed, 73);
    // GH #337: a uint64 seed (here one an older build had already rounded to a
    // double) goes to HeartMuLa as its exact decimal string, never as a number.
    gp.setSeed(16302813008879570944);
    assert.equal(useGlobalParamsStore.getState().getGlobalParams().seed, '16302813008879570944');
    gp.setSeed(73);
    assert.equal(request.randomSeed, false);
    assert.equal(request.lyrics, content.lyrics);
    assert.equal(request.caption, content.caption);
    assert.equal(createContentParams({ ...content, duration: -1 }, { mm3Mode: false, durationMax: 240 }).duration, -1);
    useBackendStore.setState({ activeBackendId: 'ace' });
    useGlobalParamsStore.getState().setBackendParam('heartmulaTemperature', 2);
    assert.equal(request.backend, 'heartmula');
    assert.equal(request.heartmulaTemperature, 0.75);
    const submitted = await generateApi.submit(request as never);
    await generateApi.cancel(submitted.jobId);
    assert.equal(calls[0].body.heartmulaAdapter, '');
    assert.equal(calls[0].body.backend, 'heartmula');
    assert.equal(calls[0].body.backendOverride, 'heartmula');
    assert.equal(calls[1].url, '/api/generate/cancel/heartmula-job');
    useBackendStore.setState({ activeBackendId: 'heartmula' });
    assert.equal(useGlobalParamsStore.getState().backendParams.heartmulaTemperature, 0.75);
    assert.equal(useGlobalParamsStore.getState().backendParams.heartmulaAdapterStrength, 0);
  } finally {
    globalThis.fetch = fetchBefore;
    queryClient.clear();
  }
});

test('an adapter selection finishing after a backend switch updates only HeartMuLa', async () => {
  const fetchBefore = globalThis.fetch;
  let finish!: () => void;
  const pending = new Promise<void>(resolve => { finish = resolve; });
  globalThis.fetch = async (input, init) => {
    if (init?.method === 'POST') {
      assert.deepEqual(JSON.parse(String(init.body)), { backend: 'heartmula', selection: { lmAdapter: '' } });
      await pending;
      return Response.json({ changed: true });
    }
    assert.equal(String(input), '/api/backends/models?backend=heartmula');
    return Response.json({ backend: 'heartmula', selectable: true, buckets: {}, defaults: { lmAdapter: '' } });
  };
  try {
    const ace = { backend: 'ace', selectable: false, buckets: {} };
    useBackendStore.setState({ activeBackendId: 'heartmula', models: { ace } });
    const pick = useBackendStore.getState().selectModels({ lmAdapter: '' }, 'heartmula');
    useBackendStore.setState({ activeBackendId: 'ace' });
    finish();
    assert.equal(await pick, true);
    assert.equal(useBackendStore.getState().activeBackendId, 'ace');
    assert.equal(useBackendStore.getState().models.ace, ace);
    assert.equal(useBackendStore.getState().models.heartmula.defaults?.lmAdapter, '');
  } finally {
    finish();
    globalThis.fetch = fetchBefore;
    queryClient.clear();
  }
});
