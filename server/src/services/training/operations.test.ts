import test from 'node:test';
import assert from 'node:assert/strict';
import express, { Router } from 'express';
import type { Server } from 'node:http';
import { z } from 'zod/v4';
import {
  acceptTrainingSnapshot, assertDatasetCurrent, registerTrainingOperations, resolveTrainingWorker, trainingOp, trainingOperationsRouter,
  TrainingOperationFailure, type TrainingOperationDeps,
} from './operations.js';
import { proxyToWorker } from './trainingWorkers.js';
import { config } from '../../config.js';

const worker = { name: 'Den', url: 'http://127.0.0.1:1' };
function fakeDeps(over: Partial<TrainingOperationDeps> = {}): TrainingOperationDeps {
  return {
    getWorker: name => (name === 'Den' ? worker : undefined),
    workerStatus: async () => ({ online: true, version: 'v1' }),
    datasetRevision: id => (id === 'ds' ? 'rev-2' : null),
    appVersion: 'v1',
    ...over,
  };
}
const payload = z.object({ steps: z.number().int().min(1) });
const envelope = (over: Record<string, unknown> = {}) => ({
  version: 1, operation: { kind: 'test-op', idempotencyKey: 'k1' }, worker: { kind: 'local' },
  dataset: { id: 'ds', revision: 'rev-2' }, payload: { steps: 5 }, ...over,
});
const failsWith = (status: number, reason?: string) => (e: unknown) =>
  e instanceof TrainingOperationFailure && e.status === status && (reason === undefined || e.body.reason === reason);

test('accept: a valid envelope comes back as a captured copy with defaults filled', async () => {
  const body = envelope();
  const { snapshot, worker: w } = await acceptTrainingSnapshot(body, payload, 'train', fakeDeps());
  assert.deepEqual(snapshot.sources, []);
  assert.equal(w.remote, null);
  (body.worker as { kind: string }).kind = 'remote';  // the client's object changing later...
  assert.equal(snapshot.worker.kind, 'local');         // ...does not move the accepted command
});

test('accept: bad envelopes and payloads are 400s with issues', async () => {
  await assert.rejects(acceptTrainingSnapshot(envelope({ version: 2 }), payload, 'train', fakeDeps()), failsWith(400));
  await assert.rejects(acceptTrainingSnapshot(envelope({ payload: { steps: 0 } }), payload, 'train', fakeDeps()),
    (e: unknown) => failsWith(400)(e) && (e as TrainingOperationFailure).body.issues!.some(i => i.path === 'payload.steps'));
  await assert.rejects(acceptTrainingSnapshot(envelope({ worker: { kind: 'browser' } }), payload, 'train', fakeDeps()), failsWith(400));
  await assert.rejects(acceptTrainingSnapshot(envelope({ operation: { kind: 'x' } }), payload, 'train', fakeDeps()), failsWith(400));
});

test('accept: a missing dataset is 404, a changed one is 409 with its current revision', async () => {
  await assert.rejects(acceptTrainingSnapshot(envelope({ dataset: { id: 'gone', revision: 'r' } }), payload, 'train', fakeDeps()), failsWith(404, 'missing-dataset'));
  await assert.rejects(acceptTrainingSnapshot(envelope({ dataset: { id: 'ds', revision: 'rev-1' } }), payload, 'train', fakeDeps()),
    (e: unknown) => failsWith(409, 'stale-dataset')(e) && (e as TrainingOperationFailure).body.currentRevision === 'rev-2');
  const { snapshot } = await acceptTrainingSnapshot(envelope({ dataset: undefined }), payload, 'train', fakeDeps());
  assert.equal(snapshot.dataset, undefined);
});

test('workers: local does everything; a remote worker must be configured, online, on this version, and able to', async () => {
  for (const cap of ['prepare', 'train', 'preview', 'review'] as const) {
    assert.equal((await resolveTrainingWorker({ kind: 'local' }, cap, fakeDeps())).remote, null);
  }
  const ok = await resolveTrainingWorker({ kind: 'remote', name: 'Den' }, 'train', fakeDeps());
  assert.deepEqual(ok.remote, worker);
  let asked = false;
  const watch = fakeDeps({ workerStatus: async () => { asked = true; return { online: true, version: 'v1' }; } });
  await assert.rejects(resolveTrainingWorker({ kind: 'remote', name: 'Den' }, 'review', watch), failsWith(409, 'unsupported-capability'));
  assert.equal(asked, false);  // refused before contacting the worker
  await assert.rejects(resolveTrainingWorker({ kind: 'remote', name: 'Gone' }, 'train', fakeDeps()), failsWith(409, 'unknown-worker'));
  await assert.rejects(resolveTrainingWorker({ kind: 'remote', name: 'Den' }, 'train', fakeDeps({ workerStatus: async () => ({ online: false, error: 'ECONNREFUSED' }) })), failsWith(503, 'worker-offline'));
  await assert.rejects(resolveTrainingWorker({ kind: 'remote', name: 'Den' }, 'prepare', fakeDeps({ workerStatus: async () => ({ online: true, version: 'v0' }) })), failsWith(409, 'worker-version'));
});

async function serve(app: express.Express): Promise<{ origin: string; close(): Promise<void> }> {
  const server: Server = await new Promise(r => { const s = app.listen(0, () => r(s)); });
  const port = (server.address() as { port: number }).port;
  return { origin: `http://127.0.0.1:${port}`, close: () => new Promise<void>(r => server.close(() => r())) };
}

test('routing: a domain is served under /ops/<domain>, legacy training routes still answer, unknown domains 404', async () => {
  const training = Router();
  training.use('/ops', trainingOperationsRouter);  // as routes/training.ts mounts it, before the legacy routes
  training.get('/datasets', (_req, res) => { res.json({ legacy: true }); });
  registerTrainingOperations('demo', (router, deps) => {
    router.post('/start', trainingOp(async req => {
      const { snapshot } = await acceptTrainingSnapshot(req.body, payload, 'train', deps);
      return { accepted: snapshot.operation.idempotencyKey, worker: snapshot.worker };
    }));
  }, fakeDeps());
  assert.throws(() => registerTrainingOperations('demo', () => {}, fakeDeps()), /already registered/);
  assert.throws(() => registerTrainingOperations('Bad/Name', () => {}, fakeDeps()), /Bad training operation domain/);
  const app = express();
  app.use(express.json());
  app.use('/api/training', training);
  const s = await serve(app);
  try {
    const post = (path: string, body: unknown) => fetch(`${s.origin}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    assert.deepEqual(await (await fetch(`${s.origin}/api/training/datasets`)).json(), { legacy: true });
    const ok = await post('/api/training/ops/demo/start', envelope());
    assert.equal(ok.status, 200);
    assert.deepEqual(await ok.json(), { accepted: 'k1', worker: { kind: 'local' } });
    const stale = await post('/api/training/ops/demo/start', envelope({ dataset: { id: 'ds', revision: 'old' } }));
    assert.equal(stale.status, 409);
    assert.deepEqual(await stale.json(), { reason: 'stale-dataset', currentRevision: 'rev-2', error: 'The dataset changed since this was prepared; reload it and try again' });
    const unsupported = await post('/api/training/ops/demo/start', envelope({ worker: { kind: 'remote', name: 'Gone' } }));
    assert.equal(unsupported.status, 409);
    assert.equal((await unsupported.json()).reason, 'unknown-worker');
    assert.equal((await post('/api/training/ops/nope/start', envelope())).status, 404);
  } finally { await s.close(); }
});

test('the worker proxy refuses operations, so "Train on" cannot redirect one', async () => {
  const saved = config.workers.list;
  config.workers.list = `Den=${worker.url}`;
  try {
    for (const url of ['/training/ops/demo/start', '/training/ops', '/training/ops?x=1']) {
      let status = 0; let body: unknown;
      const res = { status(c: number) { status = c; return this; }, json(b: unknown) { body = b; return this; } };
      await proxyToWorker({ params: { name: 'Den' }, url } as never, res as never);
      assert.equal(status, 403, url);  // never forwarded (an unreachable worker would give 502)
      assert.match((body as { error: string }).error, /explicit worker/);
    }
  } finally { config.workers.list = saved; }
});

test('accept: a dataset changed or deleted while the worker is being checked is refused', async () => {
  let rev: string | null = 'r1';
  const deps = fakeDeps({
    datasetRevision: () => rev,
    workerStatus: async () => { rev = next; return { online: true, version: 'v1' }; },
  });
  let next: string | null = 'r2';  // another client edits it during the health check
  const remote = envelope({ worker: { kind: 'remote', name: 'Den' }, dataset: { id: 'ds', revision: 'r1' } });
  await assert.rejects(acceptTrainingSnapshot(remote, payload, 'train', deps),
    (e: unknown) => failsWith(409, 'stale-dataset')(e) && (e as TrainingOperationFailure).body.currentRevision === 'r2');
  rev = 'r1'; next = null;  // ...or deletes it
  await assert.rejects(acceptTrainingSnapshot(remote, payload, 'train', deps), failsWith(404, 'missing-dataset'));
  rev = 'r1'; next = 'r1';  // unchanged: accepted
  assert.equal((await acceptTrainingSnapshot(remote, payload, 'train', deps)).snapshot.dataset?.revision, 'r1');
});

test('assertDatasetCurrent re-checks at commit time and passes without a dataset', () => {
  assert.doesNotThrow(() => assertDatasetCurrent({}, fakeDeps()));
  assert.doesNotThrow(() => assertDatasetCurrent({ dataset: { id: 'ds', revision: 'rev-2' } }, fakeDeps()));
  assert.throws(() => assertDatasetCurrent({ dataset: { id: 'ds', revision: 'rev-1' } }, fakeDeps()), failsWith(409, 'stale-dataset'));
});
