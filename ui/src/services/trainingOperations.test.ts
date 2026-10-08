import test from 'node:test';
import assert from 'node:assert/strict';
import { z } from '../../../server/node_modules/zod/v4/index.js';
import { trainingSnapshotSchema } from '../../../server/src/contracts/trainingOperation';
import { setTrainingWorker } from './trainingApi';
import { currentWorkerRef, snapshotFor, trainingOperation, TrainingOperationRequestError } from './trainingOperations';

test('the client envelope is what the server contract accepts, and the worker is captured when it is built', () => {
  setTrainingWorker(null);
  assert.deepEqual(currentWorkerRef(), { kind: 'local' });
  setTrainingWorker('Den');
  const snap = snapshotFor({ kind: 'demo', idempotencyKey: 'k', payload: { steps: 3 }, dataset: { id: 'ds', revision: 'r1' } });
  assert.deepEqual(snap.worker, { kind: 'remote', name: 'Den' });
  setTrainingWorker(null);  // "Train on" changes after the command was built
  assert.deepEqual(snap.worker, { kind: 'remote', name: 'Den' });
  const parsed = trainingSnapshotSchema(z.object({ steps: z.number() })).parse(snap);
  assert.deepEqual(parsed, snap);
  assert.deepEqual(snapshotFor({ kind: 'demo', idempotencyKey: 'k', payload: {}, worker: { kind: 'local' } }).worker, { kind: 'local' });
});

test('operations always go to this machine, and refusals keep their status, reason and revision', async () => {
  const calls: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    calls.push(`${init.method} ${url} ${(init.headers as Record<string, string>).Authorization ?? '-'}`);
    if (url.endsWith('/stale')) {
      return new Response(JSON.stringify({ error: 'changed', reason: 'stale-dataset', currentRevision: 'r2' }), { status: 409 });
    }
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  }) as typeof fetch;
  try {
    setTrainingWorker('Den');  // "Train on" a worker: operations still go here
    assert.deepEqual(await trainingOperation('prep', '/start', { body: {}, token: 't' }), { ok: true });
    await assert.rejects(trainingOperation('prep', '/stale', { body: {} }), (e: unknown) =>
      e instanceof TrainingOperationRequestError && e.status === 409 && e.reason === 'stale-dataset' && e.currentRevision === 'r2');
    assert.deepEqual(calls, ['POST /api/training/ops/prep/start Bearer t', 'POST /api/training/ops/prep/stale -']);
  } finally {
    globalThis.fetch = realFetch;
    setTrainingWorker(null);
  }
});
