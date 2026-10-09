// CPU-only step 8 cleanup checks. Production queue/reset and re-post-process
// orchestration are extracted from TypeScript; engine, DB and PP work are mocked.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import ts from 'typescript';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const routeFile = 'server/src/routes/generate.ts';
const base = { structuredClone, Reflect, Object, WeakSet, Number, Error, console: { log() {}, error() {}, warn() {} } };
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const opts = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS };
const transpile = (source, file) => ts.transpileModule(source, { fileName: file, compilerOptions: opts }).outputText;

function loadCommonJs(file, requireMock = () => { throw new Error(`unexpected require in ${file}`); }) {
  const module = { exports: {} };
  vm.runInNewContext(transpile(read(file), file), { ...base, module, exports: module.exports, require: requireMock }, { filename: file });
  return module.exports;
}

function extractFunction(file, name, context = {}) {
  const source = read(file);
  const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const node = ast.statements.find(s => ts.isFunctionDeclaration(s) && s.name?.text === name);
  assert.ok(node, `${name} missing from ${file}`);
  return vm.runInNewContext(transpile(`(${source.slice(node.getStart(ast), node.end)})`, file),
    { ...base, ...context }, { filename: file });
}

function extractResetHandler(context) {
  const source = read(routeFile);
  const ast = ts.createSourceFile(routeFile, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const statement = ast.statements.find(s => ts.isExpressionStatement(s) && ts.isCallExpression(s.expression)
    && s.expression.expression.getText(ast) === 'router.post'
    && s.expression.arguments[0]?.text === '/reset-queue');
  assert.ok(statement, 'production reset route missing');
  const callback = statement.expression.arguments[1];
  return vm.runInNewContext(transpile(`(${callback.getText(ast)})`, routeFile), { ...base, ...context }, { filename: routeFile });
}

function makeEnvelope(id, family = 'ace') {
  return Object.freeze({
    version: 1, jobId: id, userId: 'user', backendId: family, operation: 'text2music',
    submission: Object.freeze({ caption: id }), common: Object.freeze({ caption: id }),
    models: Object.freeze({ lm: 'q8_0' }), options: Object.freeze({ [family]: Object.freeze({}) }),
    policy: Object.freeze({ retry: Object.freeze({ maxAttempts: 1, reseedOnRetry: false }) }), enqueuedAt: 1,
  });
}

function flush() { return new Promise(resolve => setImmediate(resolve)); }
async function waitFor(predicate) {
  for (let i = 0; i < 100; i++) { if (predicate()) return; await flush(); }
  throw new Error('cleanup state did not settle');
}

function generationFixture(lane, residency, jobs, backend, setTimeoutImpl = callback => { queueMicrotask(callback); return 0; }) {
  const effectiveSeed = extractFunction(routeFile, 'effectiveSeed');
  const finalizeAttempt = extractFunction(routeFile, 'finalizeAttempt', { effectiveSeed });
  const getBackend = () => backend;
  const emptyOutcome = extractFunction(routeFile, 'emptyOutcome');
  const runGeneration = extractFunction(routeFile, 'runGeneration', {
    getBackend, emptyOutcome, pollUntilDone: () => 'poller',
    // The guard and lane helpers runGeneration references are supplied too:
    // without them the VM throws a ReferenceError and the fixtures never reach
    // the mocked backend.
    awaitBackendRelease: loadCommonJs('server/src/services/generation/backendRelease.ts').awaitBackendRelease,
    jobs, releaseGpuLane: lane.releaseGpuLane,
    gpuLaneNextFamily: lane.gpuLaneNextFamily, runOnGpuLane: lane.runOnGpuLane,
  });
  const { isActiveJob } = loadCommonJs('server/src/services/generation/jobTypes.ts');
  const context = {
    runGeneration, finalizeAttempt, runOnGpuLane: lane.runOnGpuLane,
    gpuLaneBusy: lane.gpuLaneBusy, gpuLaneDepth: lane.gpuLaneDepth,
    noteEnqueued: residency.noteEnqueued, noteFinished: residency.noteFinished,
    logGeneration() {}, failGenerationLog() {}, AbortController,
    setTimeout: setTimeoutImpl,
    Math: { random: () => 0.5, floor: Math.floor }, structuredClone,
    // The job ledger (services/generation/jobLedger.ts) is best-effort
    // persistence outside what these tests check.
    withLedger() {},
  };
  return {
    enqueue: extractFunction(routeFile, 'enqueueGeneration', context),
    reset: extractResetHandler({ jobs, isActiveJob,
      aceClient: { cancelJob: () => Promise.resolve() }, resetGpuLane: lane.resetGpuLane, console: base.console,
      withLedger() {} }),
  };
}

test('generation reset drains pending work and releases family demand after active work settles', async () => {
  const lane = loadCommonJs('server/src/services/generation/gpuLane.ts');
  const residency = loadCommonJs('server/src/services/generation/residency.ts');
  lane.resetGpuLane();
  const jobs = new Map(), calls = [];
  let release;
  const backend = { stageProfile: () => ({ stallMs: 1 }), async generate(job) {
    calls.push(job.id); job.status = 'running';
    await new Promise(resolve => { release = resolve; });
    job.status = 'failed'; job.error = 'reset requested';
    return { endReason: 'failed', stages: [], artifacts: [], songIds: [], error: job.error };
  } };
  const fixture = generationFixture(lane, residency, jobs, backend);
  const jobA = { id: 'A', status: 'pending', params: {}, attempts: [], envelope: makeEnvelope('A') };
  const jobB = { id: 'B', status: 'pending', params: {}, attempts: [], envelope: makeEnvelope('B') };
  jobs.set(jobA.id, jobA); jobs.set(jobB.id, jobB);
  fixture.enqueue(jobA);
  await waitFor(() => calls.length === 1 && lane.gpuLaneBusy());
  assert.equal(lane.gpuLaneOwner().label, 'generate:A');
  assert.equal(lane.gpuLaneOwner().family, 'ace');
  fixture.enqueue(jobB);
  assert.equal(residency.familyDemand('ace'), 2);
  const response = { json(value) { this.value = value; } };
  fixture.reset({}, response);
  assert.equal(JSON.stringify(response.value), JSON.stringify({ success: true, cancelled: 2, drained: 1 }));
  assert.equal(lane.gpuLaneBusy(), true, 'reset must not release active callback');
  await waitFor(() => residency.familyDemand('ace') === 1);
  assert.equal(jobB.status, 'failed');
  release();
  await waitFor(() => residency.familyDemand('ace') === 0 && !lane.gpuLaneBusy() && lane.gpuLaneDepth() === 0);
  assert.deepEqual(calls, ['A']);
  assert.equal(jobA.status, 'failed');
});

test('reset during retry delay invalidates the lease before a second backend entry', async () => {
  const lane = loadCommonJs('server/src/services/generation/gpuLane.ts');
  const residency = loadCommonJs('server/src/services/generation/residency.ts');
  lane.resetGpuLane();
  const jobs = new Map();
  let backendCalls = 0;
  let wakeRetry;
  let leaseSeen;
  const backend = { async generate(_job, deps) { backendCalls++; leaseSeen = deps.lease; throw new Error('transient mapping failure'); } };
  const fixture = generationFixture(lane, residency, jobs, backend, callback => { wakeRetry = callback; return 0; });
  const job = { id: 'retry', status: 'pending', params: { seed: 3 }, attempts: [], envelope: Object.freeze({
    ...makeEnvelope('retry'), policy: Object.freeze({ retry: Object.freeze({ maxAttempts: 2, reseedOnRetry: true }) }),
  }) };
  jobs.set(job.id, job);
  fixture.enqueue(job);
  await waitFor(() => backendCalls === 1 && wakeRetry);
  const response = { json(value) { this.value = value; } };
  fixture.reset({}, response);
  assert.equal(lane.gpuLaneBusy(), true);
  assert.equal(leaseSeen.isCurrent(), false);
  wakeRetry();
  await waitFor(() => !lane.gpuLaneBusy() && residency.familyDemand('ace') === 0);
  assert.equal(backendCalls, 1, 'invalidated retry lease must not re-enter the backend');
  assert.equal(job.attempts.length, 2);
  assert.equal(job.attempts[0].endReason, 'failed');
  assert.equal(job.attempts[1].endReason, 'reset');
});

function loadRePostProcess(lane) {
  const db = { prepare: () => ({ run() {} }) };
  const ppCalls = [];
  let nextJobId = 1;
  const fsMock = { existsSync: () => true, unlinkSync() {} };
  const required = spec => {
    if (spec === 'fs') return { __esModule: true, default: fsMock };
    if (spec === 'path') return { __esModule: true, default: path };
    if (spec === 'uuid') return { v4: () => `pp-${nextJobId++}` };
    if (spec.endsWith('/config.js')) return { config: { data: { audioDir: 'audio' } } };
    if (spec.endsWith('/database.js')) return { getDb: () => db };
    if (spec.endsWith('/vst.js')) return { vstChainActive: () => false };
    if (spec.endsWith('/logger.js')) return Object.fromEntries([
      'startGenerationLog', 'logGeneration', 'logGenerationParams', 'finishGenerationLog', 'failGenerationLog',
    ].map(name => [name, () => {}]));
    if (spec.endsWith('/gpuLane.js')) return lane;
    if (spec.endsWith('/postProcessing.js')) return {
      normalizePpParams: params => ({ ...params }),
      runPostProcessingChain: async () => { ppCalls.push(true); return { masteredUrls: ['mastered.wav'], qualityScores: [{}] }; },
    };
    throw new Error(`unexpected require ${spec}`);
  };
  const module = { exports: {} };
  vm.runInNewContext(transpile(read('server/src/services/generation/rePostProcess.ts'), 'rePostProcess.ts'),
    { ...base, setInterval: () => ({ unref() {} }), module, exports: module.exports, require: required }, { filename: 'rePostProcess.ts' });
  return { ...module.exports, ppCalls };
}

test('reset of pending re-post-processing clears inFlight after lane rejection', async () => {
  const lane = loadCommonJs('server/src/services/generation/gpuLane.ts');
  lane.resetGpuLane();
  const repro = loadRePostProcess(lane);
  const song = { id: 'song-1', audio_url: 'raw.wav', generation_params: JSON.stringify({ caption: 'song', seed: 7 }), style: 'style' };
  let release;
  lane.runOnGpuLane(() => new Promise(resolve => { release = resolve; }), { label: 'active-generation' });
  await flush();
  const first = repro.startRePostProcess(song, { stableStepOn: true });
  await flush();
  assert.equal(lane.gpuLaneBusy(), true);
  assert.equal(lane.resetGpuLane(), 1);
  assert.equal(lane.gpuLaneBusy(), true, 'the active lane callback still owns the GPU');
  await waitFor(() => first.status === 'failed');
  const second = repro.startRePostProcess(song, { stableStepOn: true });
  assert.equal(second.status, 'pending');
  release();
  await waitFor(() => second.status === 'succeeded');
  assert.equal(repro.ppCalls.length, 1);
  assert.equal(lane.gpuLaneBusy(), false);
});
