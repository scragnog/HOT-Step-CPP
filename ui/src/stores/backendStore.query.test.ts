import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { pendingSelection, useBackendStore } from './backendStore';
import { yue2PickAtEnqueue } from '../utils/yue2CaptionSource';
import { queryClient } from '../queries/client';
import { capabilitiesOptions } from '../queries/global-models/options';

const originalFetch = globalThis.fetch;
afterEach(() => { queryClient.clear(); globalThis.fetch = originalFetch; });
const json = (data: unknown) => new Response(JSON.stringify(data));

test('backend switch keeps B selected when A capabilities finish late', async () => {
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  globalThis.fetch = async input => {
    const path = String(input);
    if (path === '/api/backends/active') return json({ activeId: 'B' });
    if (path.includes('capabilities?backend=A')) { await held; return json({ up: true, displayName: 'A' }); }
    if (path.includes('capabilities?backend=B')) return json({ up: true, displayName: 'B' });
    if (path.includes('/backends/models?backend=B')) return json({ selectable: false, buckets: {} });
    throw new Error(`Unexpected request ${path}`);
  };
  const a = useBackendStore.getState().fetchCapabilities('A');
  await useBackendStore.getState().switchBackend('B');
  release(); await a;
  const state = useBackendStore.getState();
  assert.equal(state.activeBackendId, 'B');
  assert.equal(state.capabilities.B.displayName, 'B');
  assert.equal(state.capabilities.A.displayName, 'A');
});

test('down recovery polls the raw manifest while compatibility readers retain the last up manifest', async () => {
  const options = capabilitiesOptions('recovery');
  let up = false;
  globalThis.fetch = async () => json({ up, core: { modelsMissing: !up } });
  await useBackendStore.getState().fetchCapabilities('recovery');
  assert.equal(useBackendStore.getState().capabilities.recovery.up, false);
  up = true;
  await queryClient.invalidateQueries({ queryKey: options.queryKey });
  await useBackendStore.getState().fetchCapabilities('recovery');
  const usable = useBackendStore.getState().capabilities.recovery;
  assert.equal(usable.up, true);
  up = false;
  await queryClient.invalidateQueries({ queryKey: options.queryKey });
  await useBackendStore.getState().fetchCapabilities('recovery');
  assert.equal(queryClient.getQueryData(options.queryKey)?.up, false);
  assert.equal(useBackendStore.getState().capabilities.recovery, usable);
  const query = queryClient.getQueryCache().find({ queryKey: options.queryKey })!;
  assert.equal(options.refetchInterval(query as never), 10_000);
});

test('a YuE2 pick taken while a picker save is in flight carries the new values, not the stale defaults', async () => {
  let saved: Record<string, unknown> = { lmAdapterAr: 'C:/a/old.safetensors', lmAdapterArScale: 1 };
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  let failPost = false;
  globalThis.fetch = async (input, init) => {
    const path = String(input);
    if (init?.method === 'POST' && path.endsWith('/backends/models')) {
      await held;
      if (failPost) return new Response('{"error":"nope"}', { status: 400 });
      saved = { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: 0.5 };
      return json({ changed: true });
    }
    if (path.includes('/backends/models?backend=yue2')) return json({ buckets: {}, defaults: saved });
    throw new Error(`Unexpected request ${path}`);
  };
  await useBackendStore.getState().fetchModels('yue2');
  assert.deepEqual(yue2PickAtEnqueue(null), { lmAdapterAr: 'C:/a/old.safetensors', lmAdapterArScale: 1 });

  const save = useBackendStore.getState().selectModels({ lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: '0.5' }, 'yue2');
  // The store still holds the old defaults; the pick must not.
  assert.equal(useBackendStore.getState().models.yue2.defaults?.lmAdapterAr, 'C:/a/old.safetensors');
  assert.deepEqual(yue2PickAtEnqueue(null), { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: '0.5' });
  release();
  assert.equal(await save, true);
  assert.deepEqual(pendingSelection('yue2'), {});
  assert.deepEqual(yue2PickAtEnqueue(null), { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: 0.5 });

  // A failed save stops overlaying once it settles.
  failPost = true;
  assert.equal(await useBackendStore.getState().selectModels({ lmAdapterArScale: '0.2' }, 'yue2'), false);
  assert.deepEqual(pendingSelection('yue2'), {});
  assert.equal(yue2PickAtEnqueue(null).lmAdapterArScale, 0.5);
});

test('a failed save does not drop an overlapping save that carries the same value', async () => {
  const saved: Record<string, unknown> = { lmAdapterAr: 'C:/a/old.safetensors', lmAdapterArScale: 1 };
  const gates = new Map<string, { release: (ok: boolean) => void; done: Promise<boolean> }>();
  const gate = (scale: string) => {
    let release!: (ok: boolean) => void;
    const done = new Promise<boolean>(resolve => { release = resolve; });
    gates.set(scale, { release, done });
  };
  gate('0.5'); gate('0.6');
  globalThis.fetch = async (input, init) => {
    const path = String(input);
    if (init?.method === 'POST' && path.endsWith('/backends/models')) {
      const scale = JSON.parse(String(init.body)).selection.lmAdapterArScale as string;
      if (!(await gates.get(scale)!.done)) return new Response('{"error":"nope"}', { status: 400 });
      Object.assign(saved, { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: Number(scale) });
      return json({ changed: true });
    }
    if (path.includes('/backends/models?backend=yue2')) return json({ buckets: {}, defaults: { ...saved } });
    throw new Error(`Unexpected request ${path}`);
  };
  await useBackendStore.getState().fetchModels('yue2');
  const store = useBackendStore.getState();
  const first = store.selectModels({ lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: '0.5' }, 'yue2');
  const second = store.selectModels({ lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: '0.6' }, 'yue2');
  gates.get('0.5')!.release(false);
  assert.equal(await first, false);
  // The second save is still in flight and owns the same path.
  assert.deepEqual(yue2PickAtEnqueue(null), { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: '0.6' });
  gates.get('0.6')!.release(true);
  assert.equal(await second, true);
  assert.deepEqual(pendingSelection('yue2'), {});
  assert.deepEqual(yue2PickAtEnqueue(null), { lmAdapterAr: 'C:/a/new.safetensors', lmAdapterArScale: 0.6 });
});
