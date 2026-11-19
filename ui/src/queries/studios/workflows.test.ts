import test from 'node:test';
import assert from 'node:assert/strict';
import { queryClient } from '@/queries/client';
import { sharedKeys, startAuthSession } from '@/queries/keys';
import { followStudioJob, workflowOptions, documentOptions } from './workflows';

const frame = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;
const finished = frame({ type: 'snapshot', job: { id: 'job', status: 'succeeded', lastSeq: 0 } });

test('workflow bursts and replay invalidate captured snapshots once while reducers see each event once', async () => {
  const original = globalThis.fetch;
  queryClient.clear();
  startAuthSession(queryClient, 'test-A');
  const keys = [sharedKeys.workflowJob('job'), sharedKeys.workflowJob('job', 7),
    sharedKeys.workflowDocument('doc'), sharedKeys.songs()];
  assert.deepEqual(workflowOptions('job', 7).queryKey, keys[1]);
  assert.deepEqual(documentOptions('doc').queryKey, keys[2]);
  for (const key of keys) queryClient.setQueryData(key, {});
  let invalidations = 0;
  const unsubscribe = queryClient.getQueryCache().subscribe(event => {
    if (event.type === 'updated' && event.action.type === 'invalidate') invalidations++;
  });
  const status = frame({ type: 'event', event: { seq: 1, type: 'status', data: { status: 'succeeded' } } });
  globalThis.fetch = async () => new Response(finished + status + status, { headers: { 'Content-Type': 'text/event-stream' } });
  let effects = 0;
  try {
    assert.equal(await followStudioJob('job', { onEvent: () => { effects++; } }), 'succeeded');
    assert.equal(effects, 1);
    assert.equal(invalidations, 4);
    for (const key of keys) assert.equal(queryClient.getQueryState(key)?.isInvalidated, true);
    await new Promise(resolve => setTimeout(resolve, 60));
    assert.equal(invalidations, 4, 'terminal cleanup removed the pending coalescing timer');
  } finally { unsubscribe(); globalThis.fetch = original; queryClient.clear(); }
});

test('a delayed old-session stream cannot deliver effects or invalidate the new session', async () => {
  const original = globalThis.fetch;
  queryClient.clear();
  startAuthSession(queryClient, 'test-old');
  let release!: (response: Response) => void;
  const held = new Promise<Response>(resolve => { release = resolve; });
  let signal: AbortSignal | undefined;
  globalThis.fetch = async (_url, opts) => { signal = opts?.signal ?? undefined; return held; };
  let effects = 0;
  try {
    const following = followStudioJob('job', { onEvent: () => { effects++; } });
    startAuthSession(queryClient, 'test-new');
    const key = sharedKeys.songs();
    queryClient.setQueryData(key, { songs: [] });
    release(new Response(finished + frame({ type: 'event', event: { seq: 1, type: 'status', data: { status: 'succeeded' } } })));
    await assert.rejects(following, { name: 'AbortError' });
    assert.equal(effects, 0);
    assert.equal(signal?.aborted, true);
    assert.equal(queryClient.getQueryState(key)?.isInvalidated, false);
  } finally { globalThis.fetch = original; queryClient.clear(); }
});
