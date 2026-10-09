// client.trainingRuns.test.ts — a training start that succeeds, per family
// (ACE LM, ACE DiT, MM3 LM, YuE2 NAR, YuE2 AR), through the documented
// sequence: create the dataset, resolve its recipe, post the resolved
// `execution` to the start route, and follow the job to its terminal state
// over the job routes. Plus cancel, and a second start refused while one runs.
//
// The start routes run their real checks against fixture files
// (trainingFixtures.ts); only the runner that would spawn the trainer is
// replaced (fakeTrainingRunner.ts, through labelingQueue.ts's
// overrideTrainingRunner).

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient } from './client.js';
import { makeFixtureWav } from './fixtures.js';
import { completingRunner, heldRunner } from './fakeTrainingRunner.js';
import { installDatasetFixtures, installTrainerFixtures } from './trainingFixtures.js';
import { TRAINING_STARTS, recipeResolveRequest, trainingStartPath, type TrainingJobStatus } from '../../src/contracts/trainingStarts.js';
import type { TrainingRecipeFamily } from '../../src/contracts/trainingRecipes.js';

let server: FakeServer;
let client: ReferenceClient;
let queue: typeof import('../../src/services/training/labelingQueue.js');
const dirs: string[] = [];

const KIND: Record<Exclude<TrainingRecipeFamily, 'yue2-joint'>, string> = {
  'ace-lm': 'train-lm', 'ace-dit': 'train-dit', 'mm3-lm': 'mm3-train-lm', 'yue2-nar': 'yue2-nar-train', 'yue2-ar': 'yue2-ar-train',
};

const api = async (method: string, p: string, body?: unknown) => {
  const res = await fetch(`${server.origin}${p}`, { method, headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${client.token}` },
    body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: res.status, body: await res.json() as any };
};

async function preparedDataset(name: string) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'training-run-'));
  dirs.push(dir);
  fs.writeFileSync(path.join(dir, 'track-01.wav'), makeFixtureWav(1));
  const created = await api('POST', '/api/training/datasets', { name, sourceDir: dir });
  assert.equal(created.status, 201, JSON.stringify(created.body));
  installDatasetFixtures(created.body.dataset.slug, dir);
  return created.body.dataset as { id: string; slug: string };
}

async function resolveAndStart(family: keyof typeof KIND, datasetId: string) {
  const detail = await api('GET', `/api/training/datasets/${datasetId}`);
  const overrides = family === 'ace-lm' ? { lmModel: 'fake-lm' } : {};
  const resolved = await api('POST', '/api/training/ops/recipes/resolve', recipeResolveRequest({ family,
    idempotencyKey: `${family}-${datasetId}`, worker: { kind: 'local' }, dataset: { id: datasetId, revision: String(detail.body.updatedAt) }, overrides }));
  assert.equal(resolved.status, 200, JSON.stringify(resolved.body));
  return api('POST', `/api/training${trainingStartPath(family, datasetId)}`, resolved.body.execution);
}

async function waitJob(id: string, until: (s: TrainingJobStatus) => boolean): Promise<TrainingJobStatus> {
  for (let i = 0; i < 100; i++) {
    const res = await api('GET', `/api/training/jobs/${id}`);
    if (res.status === 200 && until(res.body)) return res.body;
    await new Promise(r => setTimeout(r, 50));
  }
  throw new Error(`training job ${id} never reached the expected state`);
}

test.before(async () => {
  server = await startFakeServer();
  client = new ReferenceClient(server.origin);
  await client.login();
  installTrainerFixtures();
  queue = await import('../../src/services/training/labelingQueue.js');
  for (const kind of Object.values(KIND)) queue.overrideTrainingRunner(kind as never, completingRunner(queue));
});
test.after(async () => {
  for (const kind of Object.values(KIND)) queue?.overrideTrainingRunner(kind as never, null);
  await server.close();
  for (const dir of dirs) fs.rmSync(dir, { recursive: true, force: true });
});

for (const family of Object.keys(KIND) as Array<keyof typeof KIND>) {
  test(`${family}: resolve, start, follow the job to done`, async () => {
    const ds = await preparedDataset(`Run ${family}`);
    const start = await resolveAndStart(family, ds.id);
    assert.equal(start.status, TRAINING_STARTS[family].start.successStatus, JSON.stringify(start.body));
    assert.equal(typeof start.body.jobId, 'string');
    const done = await waitJob(start.body.jobId, s => s.status === 'done');
    assert.deepEqual([done.kind, done.datasetId, done.done, done.total, done.error], [KIND[family], ds.id, 3, 3, null]);
    const listed = (await api('GET', `/api/training/jobs?datasetId=${ds.id}`)).body.jobs as TrainingJobStatus[];
    assert.deepEqual(listed.map(j => [j.id, j.status]), [[start.body.jobId, 'done']]);
  });
}

test('cancel: a running job ends cancelled, and the dataset takes one job at a time', async () => {
  queue.overrideTrainingRunner('train-dit', heldRunner(queue));
  try {
    const ds = await preparedDataset('Cancel me');
    const start = await resolveAndStart('ace-dit', ds.id);
    assert.equal(start.status, 202, JSON.stringify(start.body));
    await waitJob(start.body.jobId, s => s.status === 'running');
    const again = await resolveAndStart('ace-dit', ds.id);
    assert.deepEqual([again.status, again.body.error], [409, 'A job is already running for this dataset']);
    assert.deepEqual((await api('DELETE', `/api/training/jobs/${start.body.jobId}`)).body, { ok: true });
    const listed = (await api('GET', `/api/training/jobs?datasetId=${ds.id}`)).body.jobs as TrainingJobStatus[];
    assert.deepEqual(listed.map(j => [j.id, j.status]), [[start.body.jobId, 'cancelled']]);
    assert.equal((await resolveAndStart('ace-dit', ds.id)).status, 202, 'the dataset is free again');
  } finally { queue.overrideTrainingRunner('train-dit', completingRunner(queue)); }
});

test('Safety: no real subprocess exec or off-origin network call occurred', () => {
  assert.deepEqual(server.violations, []);
});
