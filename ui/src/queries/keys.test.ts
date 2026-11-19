// keys.test.ts — key separation, session epochs, dedup and cancellation.
// Run with the ui suite: node --import ../server/node_modules/tsx/dist/loader.mjs --test "src/**/*.test.ts"
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { hashKey } from '@tanstack/react-query';
import { createQueryClient as makeClient } from './client';
import {
  getSessionEpoch, normalizeFilters, resourceKey, scopeOf, serverKey, sharedKeys, startAuthSession,
} from './keys';
import { listDatasets, setTrainingWorker } from '@/services/trainingApi';

// gcTime keeps a 5 min timer per query; clear every client so node can exit.
// A query removed mid-fetch re-arms its timer when the fetch settles, out of
// clear()'s reach, so these clients (whose defaults aren't under test) never gc.
const made: ReturnType<typeof makeClient>[] = [];
const createQueryClient = () => {
  const c = makeClient();
  c.setDefaultOptions({ ...c.getDefaultOptions(), queries: { ...c.getDefaultOptions().queries, gcTime: Infinity } });
  made.push(c);
  return c;
};
after(() => made.forEach(c => c.clear()));

const json = (body: unknown) =>
  new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });

test('worker, local and shared scopes produce distinct keys', () => {
  const a = serverKey('training-data/datasets', { worker: 'A' });
  const b = serverKey('training-data/datasets', { worker: 'B' });
  const local = serverKey('training-data/datasets', { worker: null });
  const unscoped = serverKey('training-data/datasets');
  assert.equal(new Set([a, b, local, unscoped].map(hashKey)).size, 4);
  assert.equal(scopeOf(a)?.worker, 'A');
  assert.equal(scopeOf(local)?.worker, null);
  assert.equal('worker' in scopeOf(sharedKeys.vram())!, false);
  assert.equal(scopeOf(sharedKeys.capabilities('ace'))?.backend, 'ace');
});

test('keys carry the API base and never a credential', () => {
  const key = sharedKeys.songs();
  assert.equal(scopeOf(key)?.api, '/api');
  assert.equal(typeof scopeOf(key)?.session, 'number');
  assert.equal('session' in scopeOf(sharedKeys.plugins())!, false);
  startAuthSession(createQueryClient(), 'secret-token');
  assert.ok(!hashKey(sharedKeys.songs()).includes('secret-token'));
});

test('filters normalize: undefined and empty dropped, order irrelevant', () => {
  assert.deepEqual(normalizeFilters({ b: 2, a: undefined, c: '', d: null }), { b: 2, d: null });
  assert.equal(
    hashKey(serverKey('studios/jobs', { filters: { x: 1, y: 2 } })),
    hashKey(serverKey('studios/jobs', { filters: { y: 2, x: 1, z: undefined } })),
  );
});

test('resourceKey prefixes match by resource, scope and id', async () => {
  const client = createQueryClient();
  const keys = [
    serverKey('training-data/datasets', { worker: 'A' }),
    serverKey('training-data/datasets', { worker: 'B' }),
    serverKey('training-data/dataset', { worker: 'A', id: 'd1' }),
  ];
  for (const k of keys) client.setQueryData(k, 1);
  const hits = (key: readonly unknown[]) => client.getQueryCache().findAll({ queryKey: key }).length;
  assert.equal(hits(resourceKey('training-data/datasets')), 2);
  assert.equal(hits(resourceKey('training-data/datasets', { worker: 'A' })), 1);
  assert.equal(hits(resourceKey('training-data/dataset', undefined, 'd1')), 1);
  client.clear();
});

test('a new auth session cancels and drops old auth queries, keeps public ones', async () => {
  const client = createQueryClient();
  startAuthSession(client, 'token-1');
  const before = getSessionEpoch();
  const oldKey = sharedKeys.songs();
  let aborted = false;
  const pending = client.query({
    queryKey: oldKey,
    queryFn: ({ signal }) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => { aborted = true; reject(new DOMException('Aborted', 'AbortError')); });
    }),
  }).catch(() => 'cancelled');
  client.setQueryData(sharedKeys.vram(), { used: 1 });

  assert.equal(startAuthSession(client, 'token-1'), before, 'same token keeps the epoch');
  const next = startAuthSession(client, 'token-2');
  assert.equal(next, before + 1);
  assert.equal(await pending, 'cancelled');
  assert.equal(aborted, true);
  assert.equal(client.getQueryCache().find({ queryKey: oldKey, exact: true }), undefined);
  assert.deepEqual(client.getQueryData(sharedKeys.vram()), { used: 1 });
  assert.notEqual(hashKey(sharedKeys.songs()), hashKey(oldKey));
  client.clear();
});

test('simultaneous reads of one key share one request; another worker gets its own', async () => {
  const client = createQueryClient();
  const urls: string[] = [];
  const real = globalThis.fetch;
  globalThis.fetch = (async (url: string) => {
    urls.push(String(url));
    await new Promise(r => setTimeout(r, 5));
    return json({ datasets: [] });
  }) as typeof fetch;
  try {
    const options = (worker: string | null) => ({
      queryKey: serverKey('training-data/datasets', { worker }),
      queryFn: ({ queryKey, signal }: { queryKey: readonly unknown[]; signal: AbortSignal }) =>
        listDatasets({ worker: scopeOf(queryKey)!.worker, signal }),
    });
    await Promise.all([
      client.query(options('A')), client.query(options('A')), client.query(options('A')),
      client.query(options(null)),
    ]);
    assert.deepEqual(urls.sort(), ['/api/training/datasets', '/api/workers/A/api/training/datasets']);
  } finally { globalThis.fetch = real; client.clear(); }
});

test('the request goes where its key says, whatever the current selection', async () => {
  const client = createQueryClient();
  const urls: string[] = [];
  const real = globalThis.fetch;
  globalThis.fetch = (async (url: string) => { urls.push(String(url)); return json({ datasets: [] }); }) as typeof fetch;
  try {
    setTrainingWorker('B');
    const queryKey = serverKey('training-data/datasets', { worker: 'A' });
    await client.query({ queryKey, queryFn: ({ signal }) => listDatasets({ worker: scopeOf(queryKey)!.worker, signal }) });
    assert.deepEqual(urls, ['/api/workers/A/api/training/datasets']);
  } finally { globalThis.fetch = real; setTrainingWorker(null); client.clear(); }
});

test('cancelling a query aborts the fetch it started', async () => {
  const client = createQueryClient();
  const real = globalThis.fetch;
  let seen: AbortSignal | undefined;
  globalThis.fetch = ((_url: string, init?: RequestInit) => new Promise((_, reject) => {
    seen = init?.signal ?? undefined;
    seen?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
  })) as typeof fetch;
  try {
    const queryKey = serverKey('training-data/datasets', { worker: null });
    const run = client.query({ queryKey, queryFn: ({ signal }) => listDatasets({ worker: null, signal }) }).catch(e => e);
    await new Promise(r => setTimeout(r, 0));
    await client.cancelQueries({ queryKey });
    await run;
    assert.equal(seen?.aborted, true);
    assert.equal(client.getQueryData(queryKey), undefined);
  } finally { globalThis.fetch = real; client.clear(); }
});
