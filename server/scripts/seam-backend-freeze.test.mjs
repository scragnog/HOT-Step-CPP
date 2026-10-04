// CPU-only regression fixture for the immutable generation envelope and GPU lane.
// Production TypeScript is transpiled/extracted here; only backend runners are mocked.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import ts from 'typescript';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const tsOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS };
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const cloneContext = { structuredClone, Reflect, Object, WeakSet, Number, Error, console };

function transpile(source, fileName) {
  return ts.transpileModule(source, { fileName, compilerOptions: tsOptions }).outputText;
}

function loadCommonJs(file, requireMock = () => { throw new Error('unexpected require'); }) {
  const module = { exports: {} };
  const context = { ...cloneContext, module, exports: module.exports, require: requireMock };
  vm.runInNewContext(transpile(read(file), file), context, { filename: file });
  return module.exports;
}

function extractFunction(file, name, context = {}) {
  const source = read(file);
  const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const declaration = ast.statements.find(statement =>
    ts.isFunctionDeclaration(statement) && statement.name?.text === name);
  assert.ok(declaration, `${name} was not found in ${file}`);
  // `export` would make the transpiled slice an assignment statement, which is
  // not a valid expression for the `(fn)` wrapper.
  const code = transpile(`(${source.slice(declaration.getStart(ast), declaration.end).replace(/^export\s+/, '')})`, file);
  return vm.runInNewContext(code, { ...cloneContext, ...context }, { filename: file });
}

function makeEnvelopeFixture() {
  let active = 'ace';
  const calls = [];
  const shared = { nested: { value: 7 } };
  const modelSelection = { lm: 'q8_0' };
  const backends = {};
  const registry = {
    getActiveBackendId: () => active,
    getBackend: id => backends[id],
  };
  const envelope = loadCommonJs('server/src/services/generation/envelope.ts', spec => {
    if (spec.endsWith('/backends/registry.js')) return registry;
    if (spec.endsWith('/backends/types.js')) return { GENERATION_ENVELOPE_VERSION: 1 };
    throw new Error(`unexpected require ${spec}`);
  });
  const resolver = input => {
    calls.push(input);
    return {
      operation: 'text2music', common: { caption: input.caption ?? '', lyrics: '' },
      models: modelSelection, options: { shared, keepUndefined: undefined },
      policy: { retry: { maxAttempts: 2, reseedOnRetry: true } },
    };
  };
  backends.ace = { operations: ['text2music'], resolveRequest: resolver };
  backends['minimax-m3'] = { operations: ['text2music'], resolveRequest: resolver };
  return {
    ...envelope, backends, calls, setActive: id => { active = id; },
    getActiveBackendId: registry.getActiveBackendId, shared, modelSelection,
  };
}

test('envelope captures a deep immutable snapshot and routes by active backend', () => {
  const fixture = makeEnvelopeFixture();
  const raw = { backend: 'minimax-m3', caption: 'before', nested: { value: 3 }, params: { seed: 12 } };
  fixture.setActive('ace');
  const result = fixture.buildEnvelope(raw, 'user-1', 'job-a', 100);

  assert.equal(result.backendId, 'ace');
  assert.equal(result.submittedBackendMismatch, 'minimax-m3');
  assert.equal(fixture.calls.length, 1);
  assert(Object.isFrozen(fixture.calls[0]) && Object.isFrozen(fixture.calls[0].nested));
  raw.caption = 'after'; raw.nested.value = 99; raw.params.seed = 999;
  fixture.shared.nested.value = 88;
  fixture.modelSelection.lm = 'f16';
  assert.equal(result.submission.caption, 'before');
  assert.equal(result.submission.nested.value, 3);
  assert.equal(result.models.lm, 'q8_0');
  assert.equal(result.options.ace.shared.nested.value, 7);
  assert(Object.hasOwn(result.options.ace, 'keepUndefined'));
  assert.equal(result.options.ace.keepUndefined, undefined);
  assert.equal(result.submission.params.seed, 12);
  for (const value of [result, result.submission, result.common, result.models, result.options.ace.shared.nested, result.policy.retry]) {
    assert(Object.isFrozen(value));
  }
});

test('POST keeps mutable working parameters separate from the immutable submission', () => {
  const fixture = makeEnvelopeFixture();
  const source = read('server/src/routes/generate.ts');
  const ast = ts.createSourceFile('generate.ts', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const post = ast.statements.find(s => ts.isExpressionStatement(s) && ts.isCallExpression(s.expression)
    && s.expression.expression.getText(ast) === 'router.post' && s.expression.arguments[0]?.text === '/');
  assert(post);
  const callback = post.expression.arguments[1];
  let queued;
  const handler = vm.runInNewContext(transpile(`(${callback.getText(ast)})`, 'generate.ts'), {
    ...cloneContext, isEngineSuspended: () => false, engineReady: true, getUserId: () => 'u',
    uuidv4: () => 'post-job', buildEnvelope: fixture.buildEnvelope,
    GenerationEnvelopeError: fixture.GenerationEnvelopeError, jobs: new Map(),
    // The handler's guards are production code too — the real
    // expectedBackendMismatch is extracted; the timbre check is mocked because
    // it walks the data directory.
    getActiveBackendId: fixture.getActiveBackendId,
    expectedBackendMismatch: extractFunction('server/src/routes/generate.ts', 'expectedBackendMismatch'),
    timbreReferenceMissing: () => false,
    enqueueGeneration: job => { queued = job; },
  });
  const raw = { caption: 'input', seed: 12, nested: { value: 3 } };
  const original = JSON.stringify(raw);
  handler({ body: raw }, { json() {}, status() { throw Error('Unexpected rejection'); } });
  assert.equal(JSON.stringify(queued.params), original);
  assert.notEqual(queued.params, raw);
  assert.notEqual(queued.params.nested, raw.nested);
  queued.params.seed = 999;
  queued.params.nested.value = 88;
  assert.equal(JSON.stringify(raw), original);
  assert.equal(JSON.stringify(queued.envelope.submission), original);
  assert.equal(Object.isFrozen(queued.params), false);
});

test('unsupported bodies and unknown captured backends fail closed', () => {
  const fixture = makeEnvelopeFixture();
  fixture.setActive('ace');
  for (const body of [null, [], { backend: 123 }]) {
    assert.throws(() => fixture.buildEnvelope(body, 'u', 'j', 1), error => error.code === 'invalid_body');
  }
  fixture.backends.ace = {
    operations: ['text2music'],
    resolveRequest: () => ({ operation: 'cover', common: {}, models: {}, options: {}, policy: {} }),
  };
  assert.throws(() => fixture.buildEnvelope({}, 'u', 'j', 1), error => error.code === 'unsupported_operation');
  fixture.setActive('missing');
  assert.throws(() => fixture.buildEnvelope({}, 'u', 'j', 1), error => error.code === 'unknown_backend');
});

test('queued jobs retain captured ACE/MM3 backends across selector changes', async () => {
  const fixture = makeEnvelopeFixture();
  const envA = fixture.buildEnvelope({ caption: 'A' }, 'u', 'A', 1);
  fixture.setActive('minimax-m3');
  const envB = fixture.buildEnvelope({ caption: 'B' }, 'u', 'B', 2);
  fixture.setActive('ace');

  const lane = loadCommonJs('server/src/services/generation/gpuLane.ts');
  lane.resetGpuLane();
  const calls = [], enqueued = [], finished = [];
  let finishResolve;
  const finishedAll = new Promise(resolve => { finishResolve = resolve; });
  const backends = {
    ace: { stageProfile: () => ({ stallMs: 1 }), async generate(job, deps) {
      calls.push({ id: job.id, backend: 'ace', envelope: deps.envelope, signal: deps.signal, poller: deps.pollUntilDone });
      job.status = 'succeeded'; job.result = { audioUrls: [`${job.id}.wav`] };
    } },
    'minimax-m3': { stageProfile: () => ({ stallMs: 1 }), async generate(job, deps) {
      calls.push({ id: job.id, backend: 'minimax-m3', envelope: deps.envelope, signal: deps.signal, poller: deps.pollUntilDone });
      job.status = 'succeeded'; job.result = { audioUrls: [`${job.id}.wav`] };
    } },
  };
  const effectiveSeed = extractFunction('server/src/routes/generate.ts', 'effectiveSeed');
  const finalizeAttempt = extractFunction('server/src/routes/generate.ts', 'finalizeAttempt', { effectiveSeed });
  const emptyOutcome = extractFunction('server/src/routes/generate.ts', 'emptyOutcome');
  const routeContext = {
    getBackend: id => backends[id], finalizeAttempt, runGeneration: extractFunction('server/src/routes/generate.ts', 'runGeneration', {
      getBackend: id => backends[id], emptyOutcome, pollUntilDone: () => 'poller',
      awaitBackendRelease: loadCommonJs('server/src/services/generation/backendRelease.ts').awaitBackendRelease,
      jobs: new Map(), releaseGpuLane: lane.releaseGpuLane,
      gpuLaneNextFamily: lane.gpuLaneNextFamily, runOnGpuLane: lane.runOnGpuLane,
    }), runOnGpuLane: lane.runOnGpuLane, gpuLaneBusy: lane.gpuLaneBusy, gpuLaneDepth: lane.gpuLaneDepth,
    noteEnqueued: family => enqueued.push(family), noteFinished: family => {
      finished.push(family); if (finished.length === 2) finishResolve();
    }, MAX_RETRIES: 1, logGeneration: () => {}, failGenerationLog: () => {},
    AbortController, setTimeout, console,
  };
  const enqueue = extractFunction('server/src/routes/generate.ts', 'enqueueGeneration', routeContext);
  let release;
  lane.runOnGpuLane(() => new Promise(resolve => { release = resolve; }));
  await Promise.resolve(); // The lease starts callbacks on the next microtask.
  const jobA = { id: 'A', status: 'pending', params: {}, attempts: [], envelope: envA };
  const jobB = { id: 'B', status: 'pending', params: {}, attempts: [], envelope: envB };
  enqueue(jobA); enqueue(jobB);
  fixture.setActive('missing');
  release();
  await Promise.race([finishedAll, new Promise((_, reject) => setTimeout(() => reject(new Error('queue timeout')), 3000))]);
  assert.deepEqual(enqueued, ['ace', 'minimax-m3']);
  assert.deepEqual(calls.map(call => [call.id, call.backend]), [['A', 'ace'], ['B', 'minimax-m3']]);
  assert.equal(calls[0].poller(), 'poller');
  assert.equal(jobA.status, 'succeeded'); assert.equal(jobB.status, 'succeeded');
  assert.deepEqual(finished, ['ace', 'minimax-m3']);
  await lane.runOnGpuLane(async () => {});
  lane.resetGpuLane();
});

test('execution never falls back when a captured backend is missing', async () => {
  const backend = extractFunction('server/src/routes/generate.ts', 'runGeneration', {
    getBackend: () => undefined, emptyOutcome: extractFunction('server/src/routes/generate.ts', 'emptyOutcome'), pollUntilDone: () => {},
  });
  await assert.rejects(backend({ id: 'missing', status: 'pending', envelope: { backendId: 'gone' } }, {}),
    error => error.message.includes("Captured generation backend 'gone'"));
  await assert.rejects(backend({ id: 'missing', status: 'pending' }, {}),
    error => error.message.includes("Captured generation backend '(missing)'"));
});
