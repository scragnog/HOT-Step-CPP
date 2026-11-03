import test from 'node:test';
import assert from 'node:assert/strict';
import { createQueryClient, queryClient } from '../client';
import { projectOptions, inspirePromptOptions } from './resources';
import { lyricsOptions } from './lyrics';
import { lrcOptions } from '../shell-queue/lyrics';
import { waitForSnapshot } from './wait';
import { serverKey } from '../keys';
import { saveInstagenPrompt } from '../../services/inspireApi';
import { QueryObserver } from '@tanstack/react-query';
import { studioPluginsOptions } from './shared';
import { sharedKeys } from '../keys';
import { apiClient } from '../../services/httpClient';
import type { PluginRegistry } from '../../types/pluginTypes';

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status });
const tick = () => new Promise(resolve => setImmediate(resolve));

test('Repaint and player variants share the LRC request and sidecar invalidation', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  let calls = 0;
  globalThis.fetch = async () => { calls++; await tick(); return new Response('[00:01] Words'); };
  try {
    const studio = lyricsOptions('/audio/track_mastered.wav');
    const player = lrcOptions('/audio/track.wav');
    const results = await Promise.all([client.query(studio), client.query(player)]);
    assert.deepEqual(results, ['[00:01] Words', '[00:01] Words']);
    assert.equal(calls, 1);
    const { queryManager } = await import('../invalidation');
    await queryManager.sidecars.all(client);
    assert.equal(client.getQueryState(studio.queryKey)?.isInvalidated, true);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('studio plugin selection shares the canonical request and preserves the full cached registry', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  const registry: PluginRegistry = {
    solvers: [{ name: 'test-solver', display: 'Test solver', description: 'Metadata stays cached', params: [] }],
    schedulers: [], guidance: [], postprocess: [],
  };
  let calls = 0;
  globalThis.fetch = async url => {
    assert.equal(String(url), '/api/plugins');
    calls++;
    return json(registry);
  };
  const observer = new QueryObserver(client, {
    ...studioPluginsOptions(), refetchInterval: false,
    select: data => data.solvers.map(plugin => ({ name: plugin.name, display: plugin.display })),
  });
  const unsubscribe = observer.subscribe(() => {});
  try {
    const result = await client.query({
      queryKey: sharedKeys.plugins(),
      queryFn: ({ signal }) => apiClient.get<PluginRegistry>('/plugins', { signal }),
    });
    assert.equal(calls, 1);
    assert.deepEqual(result, registry);
    assert.deepEqual(client.getQueryData(sharedKeys.plugins()), registry);
    assert.deepEqual(observer.getCurrentResult().data, [{ name: 'test-solver', display: 'Test solver' }]);
  } finally { unsubscribe(); globalThis.fetch = original; client.clear(); }
});

test('late project A response cannot replace project B, and project reads consume abort', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  let resolveA!: (value: Response) => void;
  const heldA = new Promise<Response>(resolve => { resolveA = resolve; });
  let signal: AbortSignal | undefined;
  globalThis.fetch = async (url, opts) => {
    if (String(url).endsWith('/A')) { signal = opts?.signal ?? undefined; return heldA; }
    return json({ project: { id: 'B' }, sections: [] });
  };
  try {
    const first = client.query(projectOptions('A')).catch(error => error);
    await tick();
    await client.cancelQueries({ queryKey: projectOptions('A').queryKey });
    assert.equal(signal?.aborted, true);
    await client.query(projectOptions('B'));
    resolveA(json({ project: { id: 'A' }, sections: [] }));
    await first;
    assert.equal(client.getQueryData(projectOptions('B').queryKey)?.project.id, 'B');
    assert.equal(client.getQueryData(projectOptions('A').queryKey), undefined);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('prompt mutation invalidation refreshes the cached server value', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  let content = 'Before';
  let writes = 0;
  globalThis.fetch = async (_url, opts) => {
    if (opts?.method === 'PUT') { content = JSON.parse(String(opts.body)).value; writes++; return json({}); }
    return json({ name: 'test', custom: content, default_content: 'Default' });
  };
  try {
    const options = inspirePromptOptions();
    assert.equal((await client.query(options)).custom, 'Before');
    await saveInstagenPrompt('After');
    await client.invalidateQueries({ queryKey: options.queryKey });
    assert.equal((await client.query(options)).custom, 'After');
    assert.equal(writes, 1);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('lyrics follow the selected source and distinguish missing sidecars from errors', async () => {
  const original = globalThis.fetch;
  const client = createQueryClient();
  globalThis.fetch = async url => String(url).includes('missing') ? json({}, 404) : new Response('[00:01] Words');
  try {
    assert.equal(await client.query(lyricsOptions('/audio/missing.wav')), null);
    assert.equal(await client.query(lyricsOptions('/audio/present.wav')), '[00:01] Words');
    globalThis.fetch = async () => json({}, 403);
    await assert.rejects(client.query(lyricsOptions('/audio/forbidden.wav')), /403/);
  } finally { globalThis.fetch = original; client.clear(); }
});

test('repeated identical snapshots reach onData only with deliverRepeats', async () => {
  queryClient.clear();
  const key = serverKey('studios/test-repeats', { id: 'repeat-job' });
  let value = { status: 'running' };
  let deliveries = 0;
  const read = async () => value;
  const active = (snapshot: { status: string }) => snapshot.status === 'running';
  try {
    const done = waitForSnapshot(key, read, active, () => { deliveries++; }, { deliverRepeats: true });
    await tick();
    await queryClient.refetchQueries({ queryKey: key });
    await tick();
    // The second poll returned the same object; a cancellation-checking
    // caller still needs its onData tick.
    assert.equal(deliveries, 2);
    value = { status: 'done' };
    await queryClient.refetchQueries({ queryKey: key });
    assert.deepEqual(await done, { status: 'done' });
  } finally { queryClient.clear(); }
});

test('imperative workflow wait uses one shared query and detaches observers at terminal status', async () => {
  queryClient.clear();
  const key = serverKey('studios/test-wait', { id: 'test-job' });
  let calls = 0;
  let status = 'running';
  const read = async (_signal: AbortSignal) => { calls++; return { status }; };
  const active = (value: { status: string }) => value.status === 'running';
  try {
    const first = waitForSnapshot(key, read, active);
    const second = waitForSnapshot(key, read, active);
    await tick();
    assert.equal(calls, 1);
    assert.equal(queryClient.getQueryCache().find({ queryKey: key })?.getObserversCount(), 2);
    status = 'done';
    await queryClient.refetchQueries({ queryKey: key });
    assert.deepEqual(await first, { status: 'done' });
    assert.deepEqual(await second, { status: 'done' });
    await tick();
    assert.equal(calls, 2);
    assert.equal(queryClient.getQueryCache().find({ queryKey: key })?.getObserversCount(), 0);
  } finally { queryClient.clear(); }
});
