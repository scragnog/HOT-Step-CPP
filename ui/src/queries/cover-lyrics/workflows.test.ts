import test from 'node:test';
import assert from 'node:assert/strict';
import { queryClient } from '@/queries/client';
import { sharedKeys, startAuthSession, resourceKey } from '@/queries/keys';
import { followCoverLyricJob, workflowOptions, documentOptions } from './workflows';

const frame = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;
const finished = frame({ type: 'snapshot', job: { id: 'job', status: 'succeeded', lastSeq: 0 } });

test('workflow bursts and replay invalidate captured snapshots once while reducers see each event once', async () => {
  const original = globalThis.fetch;
  queryClient.clear();
  startAuthSession(queryClient, 'test-A');
  const keys = [workflowOptions('job').queryKey, sharedKeys.workflowJob('job', 7), documentOptions('doc').queryKey, sharedKeys.songs()];
  const otherJob = sharedKeys.workflowJob('other', 7);
  queryClient.setQueryData(otherJob, {});
  for (const key of keys) queryClient.setQueryData(key, {});
  let invalidations = 0;
  const unsubscribe = queryClient.getQueryCache().subscribe(event => {
    if (event.type === 'updated' && event.action.type === 'invalidate') invalidations++;
  });
  const status = frame({ type: 'event', event: { seq: 1, type: 'status', data: { status: 'succeeded' } } });
  globalThis.fetch = async () => new Response(finished + status + status, { headers: { 'Content-Type': 'text/event-stream' } });
  let effects = 0;
  try {
    assert.equal(await followCoverLyricJob('job', { onEvent: () => { effects++; } }), 'succeeded');
    assert.equal(effects, 1);
    assert.equal(invalidations, 4);
    for (const key of keys) assert.equal(queryClient.getQueryState(key)?.isInvalidated, true);
    await new Promise(resolve => setTimeout(resolve, 60));
    assert.equal(queryClient.getQueryState(otherJob)?.isInvalidated, false);
    assert.equal(invalidations, 4, 'terminal cleanup removed the pending coalescing timer');
  } finally { unsubscribe(); globalThis.fetch = original; queryClient.clear(); }
});

test('Cover and canonical workflow consumers deduplicate raw job and document reads', async () => {
  const original = globalThis.fetch;
  queryClient.clear();
  startAuthSession(queryClient, 'shared-workflow');
  const replay = { job: { id: 'job', status: 'running' }, events: [{ seq: 1 }] };
  const document = { document: { id: 'doc', revision: 2 } };
  let calls = 0;
  globalThis.fetch = async url => {
    calls++;
    await new Promise(resolve => setImmediate(resolve));
    return new Response(JSON.stringify(String(url).includes('/documents/') ? document : replay));
  };
  try {
    const job = workflowOptions('job');
    const doc = documentOptions('doc');
    assert.deepEqual(job.queryKey, sharedKeys.workflowJob('job', 0));
    assert.deepEqual(doc.queryKey, sharedKeys.workflowDocument('doc'));
    const [cover, canonical, coverDoc, canonicalDoc] = await Promise.all([
      queryClient.query(job),
      queryClient.query({ ...job, queryKey: sharedKeys.workflowJob('job', 0) }),
      queryClient.query(doc),
      queryClient.query({ ...doc, queryKey: sharedKeys.workflowDocument('doc') }),
    ]);
    assert.equal(calls, 2);
    assert.deepEqual(cover, replay);
    assert.deepEqual(canonical, replay);
    assert.deepEqual(coverDoc, document);
    assert.deepEqual(canonicalDoc, document);
    assert.equal(queryClient.getQueriesData({ queryKey: resourceKey('cover-lyrics/workflow-job') }).length, 0);
  } finally { globalThis.fetch = original; queryClient.clear(); }
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
    const following = followCoverLyricJob('job', { onEvent: () => { effects++; } });
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
