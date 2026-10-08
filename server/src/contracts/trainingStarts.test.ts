import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import express, { type Router } from 'express';

// Always a fresh training dir, even when the caller set one: importing the
// preparation domain opens its pipeline, which marks active records interrupted.
process.env.TRAINING_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'training-starts-'));

const { TRAINING_JOB_ROUTES, TRAINING_STARTS, jointStartRequest, recipeResolveRequest, trainingStartBase, trainingStartPath } =
  await import('./trainingStarts.js');
const { trainingSnapshotSchema } = await import('./trainingOperation.js');
const { trainingRecipeFamilySchema, trainingRecipePayloadSchema } = await import('./trainingRecipes.js');
const { yue2PreparationPayloadSchema } = await import('./trainingPreparation.js');
const { mountTrainingRecipes, resolveRecipe } = await import('../services/training/recipes/operations.js');
const { mountYue2Preparation } = await import('../services/training/yue2PreparationOperations.js');
const { mountTrainingReview } = await import('../services/training/review/operations.js');
type Deps = import('../services/training/operations.js').TrainingOperationDeps;

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const DOMAINS = { recipes: mountTrainingRecipes, preparation: mountYue2Preparation, review: mountTrainingReview };

function fakeDeps(over: Partial<Deps> = {}): Deps {
  return {
    getWorker: name => (name === 'gone' ? undefined : { name, url: 'http://127.0.0.1:9' }),
    workerStatus: async () => ({ online: false, error: 'connect ECONNREFUSED' }),
    datasetRevision: id => (id === 'ds' ? 'rev-2' : null),
    appVersion: '9.9.9',
    ...over,
  };
}

test('the route index lists exactly the routes the training domains register', async () => {
  const registered: string[] = [];
  for (const [domain, mount] of Object.entries(DOMAINS)) {
    const record = (verb: string) => (p: string) => { registered.push(`${verb} /api/training/ops/${domain}${p === '/' ? '' : p}`); };
    const router = { get: record('GET'), post: record('POST'), put: record('PUT'), delete: record('DELETE'), patch: record('PATCH') };
    mount(router as unknown as Router, fakeDeps());
  }
  const { trainingOperationRoutes } = await import(pathToFileURL(path.join(ROOT, 'tools/docs/build-docs.mjs')).href) as
    { trainingOperationRoutes: (root: string) => Array<{ verb: string; path: string }> };
  const indexed = trainingOperationRoutes(ROOT).map(r => `${r.verb} ${r.path}`);
  assert.equal(new Set(indexed).size, indexed.length, 'no duplicates');
  assert.deepEqual([...indexed].sort(), [...registered].sort());
});

test('every family names a start and run route that routes/training.ts defines', () => {
  const source = fs.readFileSync(path.join(ROOT, 'server/src/routes/training.ts'), 'utf8');
  const defined = (method: string, p: string) => source.includes(`router.${method.toLowerCase()}('${p}',`);
  assert.deepEqual(Object.keys(TRAINING_STARTS).sort(), [...trainingRecipeFamilySchema.options].sort());
  for (const [family, route] of Object.entries(TRAINING_STARTS)) {
    assert.ok(defined(route.start.method, route.start.path), `${family} start ${route.start.path}`);
    assert.ok(defined(route.runs.method, route.runs.path), `${family} runs ${route.runs.path}`);
  }
  for (const route of Object.values(TRAINING_JOB_ROUTES)) assert.ok(defined(route.method, route.path), route.path);
  assert.equal(trainingStartPath('ace-dit', 'a b'), '/datasets/a%20b/train-dit');
  assert.equal(trainingStartBase({ kind: 'local' }), '/api/training');
  assert.equal(trainingStartBase({ kind: 'remote', name: 'Living Room' }), '/api/workers/Living%20Room/api/training');
});

test('the request builders produce bodies the operation schemas accept unchanged', () => {
  const resolve = recipeResolveRequest({ family: 'mm3-lm', idempotencyKey: 'k1', worker: { kind: 'local' },
    dataset: { id: 'ds', revision: 'rev-2' }, overrides: { rank: 32 }, preset: 'fast' });
  assert.deepEqual(trainingSnapshotSchema(trainingRecipePayloadSchema).parse(resolve), resolve);
  assert.equal(resolve.operation.kind, 'recipe:mm3-lm');
  const joint = jointStartRequest({ idempotencyKey: 'k2', worker: { kind: 'remote', name: 'w' }, dataset: { id: 'ds', revision: 'rev-2' },
    source: { kind: 'dataset-sources', id: 'ds', revision: 'src-1' }, overrides: { steps: 600 } });
  assert.deepEqual(trainingSnapshotSchema(yue2PreparationPayloadSchema).parse(joint), joint);
  assert.deepEqual([joint.operation.kind, joint.payload.mode, joint.payload.stages], ['yue2-preparation', 'train-after-preparation', ['joint']]);
});

async function serve(mount: (router: Router, deps: Deps) => void, deps: Deps) {
  const app = express();
  app.use(express.json());
  const router = express.Router();
  mount(router, deps);
  app.use('/', router);
  const server = app.listen(0);
  await new Promise(resolve => server.once('listening', resolve));
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const post = async (p: string, body: unknown) => {
    const res = await fetch(`${base}${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    return { status: res.status, body: await res.json() as Record<string, unknown> };
  };
  return { post, close: () => new Promise(resolve => server.close(resolve)) };
}

test('resolve refuses an unknown, offline or mismatched worker and a stale dataset before anything starts', async () => {
  const remote = (name: string) => recipeResolveRequest({ family: 'yue2-nar', idempotencyKey: name, worker: { kind: 'remote', name } });
  const cases: Array<[Deps, unknown, number, string]> = [
    [fakeDeps(), remote('gone'), 409, 'unknown-worker'],
    [fakeDeps(), remote('w'), 503, 'worker-offline'],
    [fakeDeps({ workerStatus: async () => ({ online: true, version: '1.0.0' }) }), remote('w'), 409, 'worker-version'],
    [fakeDeps(), recipeResolveRequest({ family: 'ace-dit', idempotencyKey: 's', worker: { kind: 'local' },
      dataset: { id: 'ds', revision: 'rev-1' } }), 409, 'stale-dataset'],
    [fakeDeps(), recipeResolveRequest({ family: 'ace-dit', idempotencyKey: 'm', worker: { kind: 'local' },
      dataset: { id: 'nope', revision: 'rev-1' } }), 404, 'missing-dataset'],
  ];
  for (const [deps, body, status, reason] of cases) {
    const server = await serve(mountTrainingRecipes, deps);
    try {
      const res = await server.post('/resolve', body);
      assert.equal(res.status, status, reason);
      assert.equal(res.body.reason, reason);
      if (reason === 'stale-dataset') assert.equal(res.body.currentRevision, 'rev-2');
    } finally { await server.close(); }
  }
  const server = await serve(mountTrainingRecipes, fakeDeps());
  try {
    const bad = await server.post('/resolve', { ...remote('w'), payload: { family: 'ace-dit', recipeVersion: 2, overrides: {} } });
    assert.equal(bad.status, 400);
    assert.ok(Array.isArray(bad.body.issues));
  } finally { await server.close(); }
});

test('a local resolve returns the start body; a remote one takes the recipe from the worker', async () => {
  const local = await serve(mountTrainingRecipes, fakeDeps());
  try {
    const res = await local.post('/resolve', recipeResolveRequest({ family: 'ace-dit', idempotencyKey: 'l', worker: { kind: 'local' },
      dataset: { id: 'ds', revision: 'rev-2' } }));
    assert.equal(res.status, 200);
    for (const key of ['recipeVersion', 'family', 'resolved', 'execution', 'worker', 'operation']) assert.ok(key in res.body, key);
    assert.equal(typeof res.body.execution, 'object');
    assert.deepEqual(res.body.worker, { kind: 'local' });
  } finally { await local.close(); }

  // A fake worker serving its own recipe defaults (here, this machine's).
  const requested: string[] = [];
  const workerApp = express();
  workerApp.get('/api/training/ops/recipes/:family', (req, res) => {
    requested.push(req.originalUrl);
    res.json(resolveRecipe('yue2-nar', {}, undefined));
  });
  const workerServer = workerApp.listen(0);
  await new Promise(resolve => workerServer.once('listening', resolve));
  const url = `http://127.0.0.1:${(workerServer.address() as { port: number }).port}`;
  const server = await serve(mountTrainingRecipes, fakeDeps({ getWorker: name => ({ name, url }),
    workerStatus: async () => ({ online: true, version: '9.9.9' }) }));
  try {
    const res = await server.post('/resolve', recipeResolveRequest({ family: 'yue2-nar', idempotencyKey: 'r',
      worker: { kind: 'remote', name: 'w' }, preset: 'p' }));
    assert.equal(res.status, 200, JSON.stringify(res.body));
    assert.deepEqual(requested, ['/api/training/ops/recipes/yue2-nar?preset=p']);
    assert.deepEqual(res.body.worker, { kind: 'remote', name: 'w' });
  } finally { await server.close(); await new Promise(resolve => workerServer.close(resolve)); }
});

test('review is local-only, and the joint start refuses an offline worker and a wrong kind', async () => {
  const review = await serve(mountTrainingReview, fakeDeps({ workerStatus: async () => ({ online: true, version: '9.9.9' }) }));
  try {
    const res = await review.post('/select', { version: 1, operation: { kind: 'review-select', idempotencyKey: 'x' },
      worker: { kind: 'remote', name: 'w' }, dataset: { id: 'ds', revision: 'rev-2' }, sources: [],
      payload: { datasetId: 'ds', runId: 'run', runRevision: 1, reviewRevision: 'a'.repeat(64), step: 100, checkpointDir: 'c' } });
    assert.deepEqual([res.status, res.body.reason], [409, 'unsupported-capability']);
  } finally { await review.close(); }

  const prep = await serve(mountYue2Preparation, fakeDeps());
  try {
    const body = jointStartRequest({ idempotencyKey: `joint-${Date.now()}`, worker: { kind: 'remote', name: 'w' },
      dataset: { id: 'ds', revision: 'rev-2' }, source: { kind: 'dataset-sources', id: 'ds', revision: 's' }, overrides: {} });
    const offline = await prep.post('/', body);
    assert.deepEqual([offline.status, offline.body.reason], [503, 'worker-offline']);
    const wrongKind = await prep.post('/', { ...body, operation: { ...body.operation, kind: 'recipe:yue2-joint' } });
    assert.deepEqual([wrongKind.status, wrongKind.body.error], [400, 'Wrong operation kind']);
  } finally { await prep.close(); }
});

// Compile-time: the published response shapes still fit the services that
// produce them. A field renamed or retyped in a service fails `tsc`, not a client.
type Fits<Published, Actual extends Published> = Actual;
type _Fact = Fits<import('./trainingReview.js').ReviewRungFact, import('../services/training/yue2BestRung.js').ReviewRungFact>;
type _Plan = Fits<import('./trainingReview.js').ReviewCleanupPlan, import('../services/training/yue2Cleanup.js').Yue2CleanupPlan>;
type _Cleanup = Fits<import('./trainingReview.js').ReviewCleanupResponse,
  Awaited<ReturnType<typeof import('../services/training/yue2Cleanup.js').runYue2Cleanup>>>;
type _Decision = Fits<'skip' | 'reject' | 'chain', ReturnType<typeof import('../services/training/review/reviewPolicy.js').reviewUseDecision>>;
type _Job = Fits<import('./trainingStarts.js').TrainingJobStatus, import('../services/training/types.js').TrainingJobSummary>;
type _Finish = Fits<import('./trainingReview.js').ReviewFinishResponse['batch'],
  Exclude<ReturnType<typeof import('../services/training/yue2BatchRunner.js').finishScoredLadders>, { error: string }>>;
export type { _Fact, _Plan, _Cleanup, _Decision, _Job, _Finish };
