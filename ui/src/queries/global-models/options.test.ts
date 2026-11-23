import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { QueryObserver } from '@tanstack/react-query';
import { createQueryClient } from '@/queries/client';
import { sharedKeys } from '@/queries/keys';
import { ApiError } from '@/services/httpClient';
import { pluginsOptions, capabilitiesOptions, backendModelsOptions, registryOptions,
  vramOptions, invalidateModels, usableCapabilities } from './options';
import { bridgeDownloads, downloadsKey, downloadFrame } from './downloads';
import { adapterScanOptions } from './options';
import { adapterScanOptions as shellScanOptions } from '@/queries/shell-queue/settings';
import { projectQuery } from './projection';
import type { BackendCapabilities } from '@/stores/backendStore';

const clients: ReturnType<typeof createQueryClient>[] = [];
const client = () => { const c = createQueryClient(); clients.push(c); return c; };
const originalFetch = globalThis.fetch;
afterEach(() => { clients.splice(0).forEach(c => c.clear()); globalThis.fetch = originalFetch; });
const json = (data: unknown) => new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

test('global and shell adapter scans share a request and retain folder isolation', async () => {
  const cache = client();
  let calls = 0;
  globalThis.fetch = async () => { calls++; await tick(); return json({ files: [] }); };
  await Promise.all([cache.query(adapterScanOptions('folder-A')), cache.query(shellScanOptions('folder-A'))]);
  assert.equal(calls, 1);
  await cache.query(adapterScanOptions('folder-B'));
  assert.equal(calls, 2);
});

test('simultaneous plugin, capability and VRAM consumers share one request per key', async () => {
  const c = client();
  const requests: string[] = [];
  globalThis.fetch = async input => { requests.push(String(input)); await tick(); return json({ up: true }); };
  for (const options of [pluginsOptions(), capabilitiesOptions('a'), vramOptions()]) {
    await Promise.all([c.query(options), c.query(options), c.query(options)]);
  }
  assert.deepEqual(requests, ['/api/plugins', '/api/capabilities?backend=a', '/api/logs/vram']);
});

test('plugin reload refetches once and notifies every subscribed consumer', async () => {
  const c = client();
  let calls = 0;
  globalThis.fetch = async () => { calls++; await tick(); return json({ solvers: [{ name: `solver-${calls}` }], schedulers: [], guidance: [] }); };
  const a = new QueryObserver(c, pluginsOptions()), b = new QueryObserver(c, pluginsOptions());
  const stopA = a.subscribe(() => {}), stopB = b.subscribe(() => {});
  try {
    await c.query(pluginsOptions());
    await c.invalidateQueries({ queryKey: sharedKeys.plugins() });
    assert.equal(calls, 2);
    assert.equal(a.getCurrentResult().data?.solvers[0].name, 'solver-2');
    assert.equal(b.getCurrentResult().data?.solvers[0].name, 'solver-2');
  } finally { stopA(); stopB(); }
});

test('a late backend A response cannot overwrite B, and recovery preserves the usable manifest', async () => {
  const c = client();
  let release!: () => void;
  const pendingA = new Promise<void>(resolve => { release = resolve; });
  globalThis.fetch = async input => {
    if (String(input).endsWith('backend=A')) { await pendingA; return json({ up: false, core: { modelsMissing: true } }); }
    return json({ up: true, core: { modelsMissing: false } });
  };
  const a = c.query(capabilitiesOptions('A'));
  const b = await c.query(capabilitiesOptions('B'));
  release();
  const down = await a;
  assert.equal(c.getQueryData(capabilitiesOptions('B').queryKey), b);
  assert.equal(usableCapabilities(undefined, down), down);
  assert.equal(usableCapabilities(b, down), b);
  const query = c.getQueryCache().find({ queryKey: capabilitiesOptions('A').queryKey })!;
  assert.equal(capabilitiesOptions('A').refetchInterval(query as never), 10_000);
  globalThis.fetch = async () => json({ up: true });
  await c.invalidateQueries({ queryKey: query.queryKey });
  const recovered = await c.query(capabilitiesOptions('A'));
  assert.equal(recovered.up, true);
  assert.equal(capabilitiesOptions('A').refetchInterval(query as never), false);
});

test('unmount/remount reuses model registry and consumed signals abort cancelled reads', async () => {
  const c = client();
  let calls = 0;
  globalThis.fetch = async () => { calls++; return json({ files: [], packs: [] }); };
  const observer = new QueryObserver(c, registryOptions());
  const stop = observer.subscribe(() => {});
  await c.query(registryOptions()); stop();
  await c.query(registryOptions());
  assert.equal(calls, 1);
  let signal!: AbortSignal;
  globalThis.fetch = async (_input, opts) => {
    signal = opts!.signal!;
    return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError'))));
  };
  const read = c.query(backendModelsOptions('A')).catch(error => error);
  // The axios chain reaches the transport a microtask or two after the query
  // starts; let it dispatch before cancelling so the abort lands mid-flight.
  await tick();
  assert.ok(signal, 'the request reached the transport');
  await c.cancelQueries({ queryKey: backendModelsOptions('A').queryKey });
  await read;
  assert.equal(signal.aborted, true);
  assert.equal(c.getQueryData(backendModelsOptions('A').queryKey), undefined);
});

test('failed reads throw their HTTP status and do not cache an empty snapshot', async () => {
  const c = client();
  globalThis.fetch = async () => new Response('{"error":"missing"}', { status: 404 });
  await assert.rejects(c.query(registryOptions()), (error: unknown) => error instanceof ApiError && error.status === 404);
  assert.equal(c.getQueryData(registryOptions().queryKey), undefined);
});

type Sub = { message?: (data: string) => void; error?: () => void; open?: () => void };
test('download completion, replay, reconnect and teardown refresh only the required snapshots', async () => {
  const c = client();
  c.setQueryData(downloadsKey(), []);
  c.setQueryData(registryOptions().queryKey, { files: [], packs: [] });
  c.setQueryData(backendModelsOptions('A').queryKey, {});
  c.setQueryData(backendModelsOptions('B').queryKey, {});
  c.setQueryData(capabilitiesOptions('B').queryKey, {});
  c.setQueryData(pluginsOptions().queryKey, {});
  const subscribers = new Set<Sub>();
  let sideEffects = 0;
  const subscribe = (_url: string, sub: Sub) => { subscribers.add(sub); return () => { subscribers.delete(sub); }; };
  const attach = () => bridgeDownloads(c, '/downloads', () => sideEffects++, subscribe);
  let stop = attach();
  const job = { jobId: '1', fileId: 'f1', filename: 'model.gguf', status: 'completed', bytesDownloaded: 10, totalBytes: 10, speed: 0 };
  const frame = { jobs: [job] };
  const send = (data = frame) => subscribers.forEach(s => s.message?.(JSON.stringify(data)));
  send(); send();
  await tick();
  assert.equal(sideEffects, 1);
  assert.equal(c.getQueryState(registryOptions().queryKey)?.isInvalidated, true);
  assert.equal(c.getQueryState(backendModelsOptions('B').queryKey)?.isInvalidated, true);
  assert.equal(c.getQueryState(capabilitiesOptions('B').queryKey)?.isInvalidated, true);
  assert.equal(c.getQueryState(pluginsOptions().queryKey)?.isInvalidated, false);
  stop(); stop = attach(); send();
  assert.equal(sideEffects, 1);
  send({ jobs: [...frame.jobs, { ...job, jobId: '2', fileId: 'f2' }] });
  assert.equal(sideEffects, 2);
  c.setQueryData(registryOptions().queryKey, { files: ['fresh'], packs: [] });
  subscribers.forEach(s => s.error?.()); subscribers.forEach(s => s.open?.());
  await new Promise(resolve => setTimeout(resolve, 70));
  assert.equal(c.getQueryState(registryOptions().queryKey)?.isInvalidated, true);
  assert.equal(downloadFrame('{"jobs":[null]}'), undefined);
  const late = [...subscribers][0];
  stop();
  assert.equal(subscribers.size, 0);
  const before = c.getQueryData(downloadsKey());
  late.message?.(JSON.stringify({ jobs: [] }));
  assert.equal(c.getQueryData(downloadsKey()), before);
});

test('model deletion invalidates inactive backend catalogues as well as registry and availability', async () => {
  const c = client();
  const key = backendModelsOptions('inactive').queryKey;
  c.setQueryData(key, {});
  await invalidateModels(c);
  assert.equal(c.getQueryState(key)?.isInvalidated, true);
});

test('compatibility projection hydrates only its captured key and unsubscribes on teardown', () => {
  const c = client();
  const a = backendModelsOptions('A').queryKey;
  const b = backendModelsOptions('B').queryKey;
  c.setQueryData(a, { selectable: true });
  const received: unknown[] = [];
  const stop = projectQuery(c, a, data => received.push(data));
  c.setQueryData(b, { selectable: false });
  c.setQueryData(a, { selectable: false });
  stop();
  c.setQueryData(a, { selectable: true });
  assert.deepEqual(received, [{ selectable: true }, { selectable: false }]);
});
