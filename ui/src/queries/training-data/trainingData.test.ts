// trainingData.test.ts — worker scoping, cancellation and stale-response
// guards for the training-data store and its query options.
// Run with the ui suite: node --import ../server/node_modules/tsx/dist/loader.mjs --test "src/**/*.test.ts"
import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { queryClient } from '../client';
import { scopeOf, sharedKeys } from '../keys';
import { trainingQueries } from './queries';
import { useTrainingStore } from '../../stores/trainingStore';
import { setTrainingWorker } from '../../services/trainingApi';

const initial = useTrainingStore.getState();
const realFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
  setTrainingWorker(null);
  useTrainingStore.setState(initial, true);
  queryClient.clear();
});

type Reply = { body: unknown; status?: number };
/** A fake server. `routes` maps a URL to its reply; a reply held with
 *  `hold(url)` waits until `release(url)`. Aborts reject like fetch does. */
function fakeServer(routes: Record<string, unknown>) {
  const calls: string[] = [];
  const aborted: string[] = [];
  const held = new Map<string, () => void>();
  const holds = new Set<string>();
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    // A route or hold may name the method ("PATCH /api/..."); a bare URL
    // matches any method.
    const methodUrl = `${init?.method ?? 'GET'} ${url}`;
    calls.push(url);
    const key = methodUrl in routes || holds.has(methodUrl) ? methodUrl : url;
    if (holds.has(key)) {
      await new Promise<void>((resolve, reject) => {
        held.set(key, resolve);
        init?.signal?.addEventListener('abort', () => {
          aborted.push(key);
          reject(new DOMException('aborted', 'AbortError'));
        });
      });
    }
    if (!(key in routes)) return new Response(JSON.stringify({ error: 'nope' }), { status: 404 });
    const reply = routes[key] as Reply;
    return new Response(JSON.stringify(reply.body), {
      status: reply.status ?? 200, headers: { 'content-type': 'application/json' },
    });
  }) as typeof fetch;
  return {
    calls, aborted,
    hold: (url: string) => { holds.add(url); },
    release: (url: string) => { holds.delete(url); held.get(url)?.(); },
  };
}

const A = '/api/workers/A/api/training';
const B = '/api/workers/B/api/training';
const LOCAL = '/api/training';
const ds = (id: string) => ({ id, name: id, labeledCount: 0, samples: [], activeJobId: null });
const tick = () => new Promise(r => setTimeout(r, 0));

test('local and worker reads route by the worker in their key', async () => {
  const srv = fakeServer({
    [`${LOCAL}/datasets`]: { body: { datasets: [ds('here')] } },
    [`${A}/datasets`]: { body: { datasets: [ds('there')] } },
  });
  const local = await queryClient.query(trainingQueries.datasets(null));
  const remote = await queryClient.query(trainingQueries.datasets('A'));
  assert.deepEqual(srv.calls, [`${LOCAL}/datasets`, `${A}/datasets`]);
  assert.equal(local[0].id, 'here');
  assert.equal(remote[0].id, 'there');
  assert.equal(scopeOf(trainingQueries.datasets('A').queryKey)?.worker, 'A');
  assert.equal(scopeOf(trainingQueries.datasets(null).queryKey)?.worker, null);
});

test('a read captures its worker: switching the global after scheduling does not move it', async () => {
  const srv = fakeServer({ [`${A}/datasets`]: { body: { datasets: [] } } });
  srv.hold(`${A}/datasets`);
  const read = queryClient.query(trainingQueries.datasets('A'));
  await tick();
  setTrainingWorker('B');
  srv.release(`${A}/datasets`);
  await read;
  assert.deepEqual(srv.calls, [`${A}/datasets`]);
});

test('a slow dataset list from worker A never lands after switching to B', async () => {
  const srv = fakeServer({
    [`${A}/datasets`]: { body: { datasets: [ds('a1')] } },
    [`${B}/datasets`]: { body: { datasets: [ds('b1')] } },
    [`${B}/capabilities`]: { body: {} },
    [`${B}/yue2-batch`]: { body: { batches: [] } },
  });
  setTrainingWorker('A');
  useTrainingStore.setState({ trainingWorker: 'A' });
  srv.hold(`${A}/datasets`);
  const loadA = useTrainingStore.getState().loadDatasets();
  await tick();
  await useTrainingStore.getState().setTrainingWorker('B');
  srv.release(`${A}/datasets`);
  await loadA;
  const s = useTrainingStore.getState();
  assert.equal(s.trainingWorker, 'B');
  assert.deepEqual(s.datasets.map(d => d.id), ['b1']);
  assert.equal(s.error, null, 'the cancelled read must not surface as an error');
  // The switch cancelled the read it no longer wanted.
  assert.deepEqual(srv.aborted, [`${A}/datasets`]);
});

test('a slow dataset open from worker A does not open on B', async () => {
  const srv = fakeServer({
    [`${A}/datasets/x`]: { body: ds('x') },
    [`${B}/datasets`]: { body: { datasets: [] } },
    [`${B}/capabilities`]: { body: {} },
  });
  setTrainingWorker('A');
  useTrainingStore.setState({ trainingWorker: 'A' });
  srv.hold(`${A}/datasets/x`);
  const open = useTrainingStore.getState().openDataset('x');
  await tick();
  await useTrainingStore.getState().setTrainingWorker('B');
  srv.release(`${A}/datasets/x`);
  await open;
  const s = useTrainingStore.getState();
  assert.equal(s.selectedDatasetId, null);
  assert.equal(s.detail, null);
});

test('the newest open wins, and a close beats an open still in flight', async () => {
  const srv = fakeServer({
    [`${LOCAL}/datasets/slow`]: { body: ds('slow') },
    [`${LOCAL}/datasets/fast`]: { body: ds('fast') },
  });
  srv.hold(`${LOCAL}/datasets/slow`);
  const slow = useTrainingStore.getState().openDataset('slow');
  await tick();
  await useTrainingStore.getState().openDataset('fast');
  srv.release(`${LOCAL}/datasets/slow`);
  await slow;
  assert.equal(useTrainingStore.getState().selectedDatasetId, 'fast');

  srv.hold(`${LOCAL}/datasets/slow`);
  const again = useTrainingStore.getState().openDataset('slow');
  await tick();
  useTrainingStore.getState().closeDataset();
  srv.release(`${LOCAL}/datasets/slow`);
  await again;
  assert.equal(useTrainingStore.getState().selectedDatasetId, null);
});

test('concurrent loads of one list share a request', async () => {
  const srv = fakeServer({ [`${LOCAL}/datasets`]: { body: { datasets: [] } } });
  srv.hold(`${LOCAL}/datasets`);
  const a = useTrainingStore.getState().loadDatasets();
  const b = useTrainingStore.getState().loadDatasets();
  await tick();
  srv.release(`${LOCAL}/datasets`);
  await Promise.all([a, b]);
  assert.equal(srv.calls.filter(u => u === `${LOCAL}/datasets`).length, 1);
});

test('a status read for the previous dataset does not land on the next', async () => {
  const srv = fakeServer({
    [`${LOCAL}/datasets/one/preprocess`]: { body: { variants: [{ key: 'one' }] } },
  });
  useTrainingStore.setState({ selectedDatasetId: 'one' });
  srv.hold(`${LOCAL}/datasets/one/preprocess`);
  const read = useTrainingStore.getState().loadPreprocessStatus();
  await tick();
  useTrainingStore.setState({ selectedDatasetId: 'two' });
  srv.release(`${LOCAL}/datasets/one/preprocess`);
  await read;
  assert.deepEqual(srv.calls, [`${LOCAL}/datasets/one/preprocess`]);
  const s = useTrainingStore.getState();
  assert.equal(s.preprocessStatus, null);
  assert.equal(s.preprocessLoading, false);
});

test('a terminal stream event invalidates the worker scope once, replay included', async () => {
  const srv = fakeServer({
    [`${A}/datasets/d`]: { body: ds('d') },
    [`${A}/datasets`]: { body: { datasets: [ds('d')] } },
    [`${A}/datasets/d/ls-generations`]: { body: { lyricsSetId: 0, artist: '', album: '', generations: [] } },
    [`${B}/datasets/d/ls-generations`]: { body: { lyricsSetId: 0, artist: '', album: '', generations: [] } },
  });
  setTrainingWorker('A');
  useTrainingStore.setState({
    trainingWorker: 'A', selectedDatasetId: 'd', activeJobId: 'job-term',
  });
  queryClient.setQueryData(trainingQueries.job('A', 'job-term').queryKey,
    { id: 'job-term', kind: 'label', status: 'done' } as never);
  await queryClient.query(trainingQueries.lsGenerations('A', 'd'));
  await queryClient.query(trainingQueries.lsGenerations('B', 'd'));
  const terminal = useTrainingStore.getState().jobTerminal;
  terminal('job-term');
  terminal('job-term'); // replayed on reconnect
  await tick(); await tick();
  const stateOf = (w: string) => queryClient.getQueryState(trainingQueries.lsGenerations(w, 'd').queryKey);
  assert.equal(stateOf('A')?.isInvalidated, true);
  assert.equal(stateOf('B')?.isInvalidated, false, 'another worker is untouched');
  assert.equal(srv.calls.filter(u => u === `${A}/datasets/d`).length, 1, 'one refresh per job');
});

test('a job started on A is not adopted once the studio is on B', async () => {
  const srv = fakeServer({
    [`${A}/datasets/d/label`]: { body: { jobId: 'j1' } },
    [`${A}/jobs/j1`]: { body: { id: 'j1', kind: 'label', status: 'running' } },
  });
  setTrainingWorker('A');
  useTrainingStore.setState({ trainingWorker: 'A', selectedDatasetId: 'd' });
  srv.hold(`${A}/datasets/d/label`);
  const start = useTrainingStore.getState().startLabel({} as never);
  await tick();
  setTrainingWorker('B');
  useTrainingStore.setState({ trainingWorker: 'B', selectedDatasetId: null });
  srv.release(`${A}/datasets/d/label`);
  await start;
  assert.equal(useTrainingStore.getState().activeJobId, null);
  assert.ok(!srv.calls.includes(`${A}/jobs/j1`), 'no adoption read for a scope that is gone');
});

const sample = (sampleId: string, caption: string) => ({
  sampleId, relPath: `${sampleId}.wav`, caption, labelStatus: 'done', excluded: false, fileMissing: false,
});

test('a save that finishes after a dataset switch leaves the new dataset alone', async () => {
  const srv = fakeServer({
    [`${LOCAL}/datasets/A`]: { body: { ...ds('A'), samples: [sample('s1', 'old')] } },
    [`${LOCAL}/datasets/B`]: { body: { ...ds('B'), samples: [sample('s1', 'old')] } },
    [`PATCH ${LOCAL}/datasets/A/samples/s1`]: { body: { sample: sample('s1', 'same') } },
  });
  await useTrainingStore.getState().openDataset('A');
  useTrainingStore.setState({ pendingEdits: { s1: { caption: 'same' } } });
  srv.hold(`PATCH ${LOCAL}/datasets/A/samples/s1`);
  const save = useTrainingStore.getState().flushSample('s1');
  await tick();
  assert.ok(useTrainingStore.getState().savingSampleIds.has('s1'));
  await useTrainingStore.getState().openDataset('B');
  assert.equal(useTrainingStore.getState().savingSampleIds.size, 0, "the switch leaves A's save behind");
  // B has its own pending edit, same sample id and same value as A's save.
  useTrainingStore.setState({ pendingEdits: { s1: { caption: 'same' } } });
  srv.release(`PATCH ${LOCAL}/datasets/A/samples/s1`);
  await save;
  const s = useTrainingStore.getState();
  assert.deepEqual(s.pendingEdits, { s1: { caption: 'same' } }, 'B keeps its pending edit');
  assert.equal(s.samplesById.s1.caption, 'old', 'B shows its own row');
  assert.equal(s.saveError, null);
  // A's cached detail took the saved row, for when A is opened again.
  const cachedA = queryClient.getQueryData(trainingQueries.dataset(null, 'A').queryKey);
  assert.equal(cachedA?.samples[0].caption, 'same');
});

test("a failed save after a dataset switch does not clear the new dataset's state", async () => {
  const srv = fakeServer({
    [`${LOCAL}/datasets/A`]: { body: { ...ds('A'), samples: [sample('s1', 'old')] } },
    [`${LOCAL}/datasets/B`]: { body: { ...ds('B'), samples: [sample('s1', 'old')] } },
    [`PATCH ${LOCAL}/datasets/A/samples/s1`]: { body: { error: 'locked' }, status: 409 },
  });
  await useTrainingStore.getState().openDataset('A');
  useTrainingStore.setState({ pendingEdits: { s1: { caption: 'x' } } });
  srv.hold(`PATCH ${LOCAL}/datasets/A/samples/s1`);
  const save = useTrainingStore.getState().flushSample('s1');
  await tick();
  await useTrainingStore.getState().openDataset('B');
  useTrainingStore.setState({ pendingEdits: { s1: { caption: 'x' } }, savingSampleIds: new Set(['s1']) });
  srv.release(`PATCH ${LOCAL}/datasets/A/samples/s1`);
  await save;
  const s = useTrainingStore.getState();
  assert.deepEqual(s.pendingEdits, { s1: { caption: 'x' } });
  assert.ok(s.savingSampleIds.has('s1'), "B's own save is still marked");
  assert.equal(s.saveError, null);
});

test('loads use the cache while fresh; a mutation makes the next load read again', async () => {
  const srv = fakeServer({
    [`GET ${LOCAL}/datasets`]: { body: { datasets: [ds('a')] } },
    [`POST ${LOCAL}/datasets`]: { body: { dataset: ds('new') } },
  });
  await useTrainingStore.getState().loadDatasets();
  await useTrainingStore.getState().loadDatasets(); // tab return
  assert.equal(srv.calls.length, 1, 'one read for two loads');
  await useTrainingStore.getState().createDataset({} as never);
  assert.equal(srv.calls.filter(u => u === `${LOCAL}/datasets`).length, 3, 'POST, then one fresh list read');
  // The created detail is cached for its scope, so opening it reads nothing.
  await useTrainingStore.getState().openDataset('new');
  assert.equal(srv.calls.length, 3);
});

test('a rename updates the cache of the scope it was made in, even after the selection moved', async () => {
  const srv = fakeServer({
    [`GET ${A}/datasets`]: { body: { datasets: [{ ...ds('d'), name: 'old' }] } },
    [`GET ${A}/datasets/d`]: { body: { ...ds('d'), name: 'old' } },
    [`PATCH ${A}/datasets/d`]: { body: { dataset: { ...ds('d'), name: 'new' } } },
    [`GET ${B}/datasets`]: { body: { datasets: [] } },
    [`GET ${B}/capabilities`]: { body: {} },
  });
  setTrainingWorker('A');
  useTrainingStore.setState({ trainingWorker: 'A' });
  await useTrainingStore.getState().loadDatasets();
  await useTrainingStore.getState().openDataset('d');
  srv.hold(`PATCH ${A}/datasets/d`);
  const rename = useTrainingStore.getState().patchDataset({ name: 'new' } as never);
  await tick();
  await useTrainingStore.getState().setTrainingWorker('B');
  srv.release(`PATCH ${A}/datasets/d`);
  await rename;
  assert.equal(queryClient.getQueryData(trainingQueries.dataset('A', 'd').queryKey)?.name, 'new');
  assert.equal(queryClient.getQueryData(trainingQueries.datasets('A').queryKey)?.[0].name, 'new');
  assert.deepEqual(useTrainingStore.getState().datasets, [], "B's list is untouched");
});

test('a rename on the open dataset updates store and cache together', async () => {
  fakeServer({
    [`GET ${LOCAL}/datasets`]: { body: { datasets: [{ ...ds('d'), name: 'old' }] } },
    [`GET ${LOCAL}/datasets/d`]: { body: { ...ds('d'), name: 'old' } },
    [`PATCH ${LOCAL}/datasets/d`]: { body: { dataset: { ...ds('d'), name: 'new' } } },
  });
  await useTrainingStore.getState().loadDatasets();
  await useTrainingStore.getState().openDataset('d');
  await useTrainingStore.getState().patchDataset({ name: 'new' } as never);
  const s = useTrainingStore.getState();
  assert.equal(s.detail?.name, 'new');
  assert.equal(s.datasets[0].name, 'new');
  assert.equal(queryClient.getQueryData(trainingQueries.dataset(null, 'd').queryKey)?.name, 'new');
});

test('deleting a dataset drops its cached detail and refreshes the list', async () => {
  const srv = fakeServer({
    [`GET ${LOCAL}/datasets`]: { body: { datasets: [ds('d')] } },
    [`GET ${LOCAL}/datasets/d`]: { body: ds('d') },
    [`DELETE ${LOCAL}/datasets/d`]: { body: { ok: true } },
  });
  await useTrainingStore.getState().loadDatasets();
  await useTrainingStore.getState().openDataset('d');
  await useTrainingStore.getState().deleteDataset('d');
  assert.equal(queryClient.getQueryData(trainingQueries.dataset(null, 'd').queryKey), undefined);
  assert.equal(srv.calls.filter(u => u === `${LOCAL}/datasets`).length, 2, 'list read again after the delete');
  assert.equal(useTrainingStore.getState().selectedDatasetId, null);
});

test('preparation reads use the local shared keys: concurrent polls join, each poll asks again', async () => {
  let calls = 0;
  globalThis.fetch = (async () => {
    calls++;
    await tick();
    return new Response(JSON.stringify({ pipeline: { id: 'p', status: 'running', stages: [] } }), { status: 200 });
  }) as typeof fetch;
  setTrainingWorker('A');
  const options = trainingQueries.preparation('p');
  assert.deepEqual(options.queryKey, sharedKeys.preparation('p'));
  assert.equal('worker' in scopeOf(options.queryKey)!, false);
  await Promise.all([queryClient.query(options), queryClient.query(trainingQueries.preparation('p'))]);
  assert.equal(calls, 1);
  await queryClient.query(options);
  assert.equal(calls, 2);
});

test('a scan preview goes to the worker captured when it was asked for', async () => {
  const server = fakeServer({ [`${A}/scan-preview?path=x&recursive=1`]: { body: { audioFiles: 3 } } });
  const read = queryClient.query(trainingQueries.scanPreview('A', 'x'));
  setTrainingWorker('B');
  assert.equal((await read).audioFiles, 3);
  assert.deepEqual(server.calls, [`${A}/scan-preview?path=x&recursive=1`]);
});
