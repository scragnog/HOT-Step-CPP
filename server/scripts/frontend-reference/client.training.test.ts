// client.training.test.ts — the ACE, MM3 and YuE2 training audit groups
// through the documented sequence (docs/dev/frontend-training.md): create a
// dataset, read its revision, list and resolve the family's recipe with the
// operation envelope, then post the resolved `execution` to the family's
// start route (YuE2 joint: the preparation operation). Variants: stale
// dataset revision, rejected input, an unconfigured worker, unknown jobs and
// pipelines (the cancellation routes), and auth where it applies.
//
// Every route here is real. What the harness cannot do is finish a start:
// the isolated root has no trainer binary or models, so each start route
// gives its documented refusal (503: no trainer binary; joint 409), and
// that is what is asserted. A start that got past those checks would queue a
// job whose runner spawns the trainer process, which this suite must never
// do; there is no injectable runner on that path today (see the 7f-3
// report). Restart interruption is covered by the services' own tests: the
// preparation pipeline and training job stores reconcile at construction,
// which happens once per process here.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient } from './client.js';
import { makeFixtureWav } from './fixtures.js';
import {
  TRAINING_JOB_ROUTES, TRAINING_STARTS, jointStartRequest, recipeResolveRequest, trainingStartPath,
  type RecipeResolveResponse,
} from '../../src/contracts/trainingStarts.js';
import type { TrainingRecipeFamily } from '../../src/contracts/trainingRecipes.js';
import type { Yue2PreparationContext } from '../../src/contracts/trainingPreparation.js';

let server: FakeServer;
let client: ReferenceClient;
let datasetId = '';
const sourceDirs: string[] = [];

const api = async (method: string, p: string, body?: unknown, auth = true) => {
  const res = await fetch(`${server.origin}${p}`, { method,
    headers: { 'Content-Type': 'application/json', ...(auth ? { Authorization: `Bearer ${client.token}` } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body) });
  const text = await res.text();
  let parsed: any = text;
  try { parsed = text ? JSON.parse(text) : undefined; } catch { /* not JSON */ }
  return { status: res.status, body: parsed };
};
const revision = async () => String((await api('GET', `/api/training/datasets/${datasetId}`)).body.updatedAt);

test.before(async () => {
  server = await startFakeServer();
  client = new ReferenceClient(server.origin);
  await client.login();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'training-source-'));
  sourceDirs.push(dir);
  fs.writeFileSync(path.join(dir, 'track-01.wav'), makeFixtureWav(1));
  const created = await api('POST', '/api/training/datasets', { name: 'Fixture Set', sourceDir: dir });
  assert.equal(created.status, 201, JSON.stringify(created.body));
  datasetId = created.body.dataset.id;
});
test.after(async () => {
  await server.close();
  for (const dir of sourceDirs) fs.rmSync(dir, { recursive: true, force: true });
});

test('dataset: create, read its revision, refuse a duplicate folder and bad input', async () => {
  const detail = await api('GET', `/api/training/datasets/${datasetId}`);
  assert.equal(detail.status, 200);
  assert.equal(typeof detail.body.updatedAt, 'string');
  assert.equal((await api('POST', '/api/training/datasets', { name: 'Again', sourceDir: sourceDirs[0] })).status, 409);
  assert.equal((await api('POST', '/api/training/datasets', { name: '', sourceDir: sourceDirs[0] })).status, 400);
  assert.equal((await api('GET', '/api/training/datasets/no-such-dataset')).status, 404);
});

test('recipes: the published families, each readable without a worker', async () => {
  const list = await api('GET', '/api/training/ops/recipes');
  assert.deepEqual(list.body.families, Object.keys(TRAINING_STARTS));
  for (const family of list.body.families as TrainingRecipeFamily[]) {
    const recipe = await api('GET', `/api/training/ops/recipes/${family}`);
    assert.equal(recipe.status, 200, family);
    assert.equal(recipe.body.family, family);
  }
  assert.equal((await api('GET', '/api/training/ops/recipes/not-a-family')).status, 400);
});

// Every start route checks for the trainer binary before anything else, and
// the isolated root has none: the documented 503, with no job queued.
const START_REFUSAL: Record<Exclude<TrainingRecipeFamily, 'yue2-joint'>, { status: number; error: RegExp }> = {
  'ace-lm': { status: 503, error: /ace-train was not found/ },
  'ace-dit': { status: 503, error: /ace-train was not found/ },
  'mm3-lm': { status: 503, error: /ace-train was not found/ },
  'yue2-nar': { status: 503, error: /ace-train was not found/ },
  'yue2-ar': { status: 503, error: /ace-train was not found/ },
};

for (const family of Object.keys(START_REFUSAL) as Array<keyof typeof START_REFUSAL>) {
  test(`${family}: resolve with the envelope, then post execution to the start route`, async () => {
    const dataset = { id: datasetId, revision: await revision() };
    const resolved = await api('POST', '/api/training/ops/recipes/resolve',
      recipeResolveRequest({ family, idempotencyKey: `${family}-1`, worker: { kind: 'local' }, dataset }));
    assert.equal(resolved.status, 200, JSON.stringify(resolved.body));
    const recipe = resolved.body as RecipeResolveResponse;
    assert.deepEqual([recipe.family, recipe.worker, recipe.operation.kind], [family, { kind: 'local' }, `recipe:${family}`]);
    assert.equal(typeof recipe.execution, 'object');

    const start = await api(TRAINING_STARTS[family].start.method, `/api/training${trainingStartPath(family, datasetId)}`, recipe.execution);
    assert.equal(start.status, START_REFUSAL[family].status, JSON.stringify(start.body));
    assert.match(start.body.error, START_REFUSAL[family].error);
    assert.equal(start.body.jobId, undefined, 'no job was queued');
    const runs = await api('GET', `/api/training${TRAINING_STARTS[family].runs.path.replace(':id', datasetId)}`);
    assert.ok(runs.status === 200 || runs.status === 400, `${family} run list answers (${runs.status})`);
  });
}

test('resolve variants: stale revision, missing dataset, unconfigured worker, malformed envelope', async () => {
  const current = await revision();
  const stale = await api('POST', '/api/training/ops/recipes/resolve', recipeResolveRequest({ family: 'ace-dit', idempotencyKey: 's',
    worker: { kind: 'local' }, dataset: { id: datasetId, revision: 'an-older-revision' } }));
  assert.deepEqual([stale.status, stale.body.reason, stale.body.currentRevision], [409, 'stale-dataset', current]);
  const missing = await api('POST', '/api/training/ops/recipes/resolve', recipeResolveRequest({ family: 'ace-dit', idempotencyKey: 'm',
    worker: { kind: 'local' }, dataset: { id: 'gone', revision: 'r' } }));
  assert.deepEqual([missing.status, missing.body.reason], [404, 'missing-dataset']);
  const worker = await api('POST', '/api/training/ops/recipes/resolve', recipeResolveRequest({ family: 'yue2-nar', idempotencyKey: 'w',
    worker: { kind: 'remote', name: 'nowhere' } }));
  assert.deepEqual([worker.status, worker.body.reason], [409, 'unknown-worker']);
  const bad = await api('POST', '/api/training/ops/recipes/resolve', { version: 2, payload: {} });
  assert.equal(bad.status, 400);
  assert.ok(Array.isArray(bad.body.issues));
  const ace = await api('POST', `/api/training/datasets/no-such-dataset/train-dit`, {});
  assert.equal(ace.status, 404);
});

test('yue2-joint: context, then the preparation start refuses unprepared inputs without starting anything', async () => {
  const context = await api('GET', `/api/training/ops/preparation/context/${datasetId}`);
  assert.equal(context.status, 200);
  const ctx = context.body as Yue2PreparationContext;
  assert.equal(ctx.dataset.id, datasetId);
  const body = jointStartRequest({ idempotencyKey: 'joint-1', worker: { kind: 'local' }, dataset: ctx.dataset, source: ctx.source, overrides: {} });
  const start = await api('POST', '/api/training/ops/preparation', body);
  assert.equal(start.status, 409, JSON.stringify(start.body));
  assert.match(start.body.error, /Prepare this dataset before starting joint training/);
  assert.deepEqual((await api('GET', `/api/training/ops/preparation?datasetId=${datasetId}`)).body.pipelines, []);
  const stale = await api('POST', '/api/training/ops/preparation', { ...body, dataset: { id: datasetId, revision: 'older' } });
  assert.deepEqual([stale.status, stale.body.reason], [409, 'stale-dataset']);
  const shape = await api('POST', '/api/training/ops/preparation', { ...body, operation: { ...body.operation, idempotencyKey: 'joint-2' },
    payload: { ...body.payload, stages: ['joint', 'nar'] } });
  assert.equal(shape.status, 400);
  assert.equal((await api('GET', '/api/training/ops/preparation/context/no-such-dataset')).status, 404);
});

test('jobs and pipelines: list, and the read/cancel routes answer 404 for unknown ids', async () => {
  const jobs = await api(TRAINING_JOB_ROUTES.list.method, `/api/training${TRAINING_JOB_ROUTES.list.path}?datasetId=${datasetId}`);
  assert.deepEqual([jobs.status, jobs.body.jobs], [200, []]);
  assert.equal((await api('GET', '/api/training/jobs/no-such-job')).status, 404);
  assert.equal((await api('DELETE', '/api/training/jobs/no-such-job')).status, 404);
  for (const action of ['pause', 'resume', 'retry', 'cancel']) {
    assert.equal((await api('POST', `/api/training/ops/preparation/no-such-pipeline/${action}`)).status, 404, action);
  }
});

test('auth: training routes are installation-scoped; the audition-draft routes refuse without a token', async () => {
  assert.equal((await api('GET', '/api/training/ops/recipes', undefined, false)).status, 200);
  const draft = await api('GET', '/api/training/ops/review/draft/anything', undefined, false);
  assert.equal(draft.status, 400);
  assert.match(draft.body.error, /Sign in/);
});

test('Safety: no real subprocess exec or off-origin network call occurred', () => {
  assert.deepEqual(server.violations, []);
});
