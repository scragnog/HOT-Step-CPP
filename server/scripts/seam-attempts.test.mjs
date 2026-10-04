// CPU-only step 7 regression test. Queue and attempt bookkeeping come from
// production generate.ts; backend generation itself is deliberately mocked.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import ts from 'typescript';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const routeFile = 'server/src/routes/generate.ts';
const quietConsole = { log() {}, error() {}, warn() {} };
const baseContext = { structuredClone, Reflect, Object, WeakSet, Number, Error, console: quietConsole };
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const options = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS };
const transpile = (source, file) => ts.transpileModule(source, { fileName: file, compilerOptions: options }).outputText;

function extract(name, context = {}) {
  const source = read(routeFile);
  const ast = ts.createSourceFile(routeFile, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const node = ast.statements.find(s => ts.isFunctionDeclaration(s) && s.name?.text === name);
  assert.ok(node, `${name} missing from production route`);
  return vm.runInNewContext(transpile(`(${source.slice(node.getStart(ast), node.end)})`, routeFile),
    { ...baseContext, ...context }, { filename: routeFile });
}

function freeze(value, seen = new WeakSet()) {
  if (!value || typeof value !== 'object' || seen.has(value)) return value;
  seen.add(value);
  for (const child of Object.values(value)) freeze(child, seen);
  return Object.freeze(value);
}

function envelopeFor(backendId, maxAttempts = 2) {
  return freeze({
    version: 1, jobId: 'attempt-job', userId: 'user', backendId, operation: 'text2music',
    submission: { caption: 'caption', lyrics: '[Verse] words', triggerWords: ['voice'], sideband: { keep: true } },
    common: { caption: 'caption' }, models: { lm: 'q8_0' }, options: { [backendId]: { guidance: 1 } },
    policy: { retry: { maxAttempts, reseedOnRetry: true } }, enqueuedAt: 1,
  });
}

function loadModule(file) {
  const module = { exports: {} };
  vm.runInNewContext(transpile(read(file), file), { ...baseContext, module, exports: module.exports }, { filename: file });
  return module.exports;
}

async function execute({ backendId = 'ace', actions, maxAttempts = 2, params = { seed: 123, randomSeed: false }, mm3TakeSeeds, exactSeed }) {
  const lane = loadLane();
  lane.resetGpuLane();
  const calls = [];
  let done;
  const finished = new Promise(resolve => { done = resolve; });
  const backend = {
    stageProfile: () => ({ stallMs: 1 }),
    async generate(job, deps) {
      const attempt = deps.attempt;
      calls.push({ attempt: attempt.attempt, seed: job.params.seed, submitted: JSON.stringify(job.envelope.submission), deps });
      // Attempts receive working mutable state; changing it must not alter the envelope.
      attempt.effective.models.lm = `attempt-${attempt.attempt}`;
      if (exactSeed !== undefined) attempt.effective.seed = exactSeed;
      const action = actions[attempt.attempt - 1];
      if (action?.throw) throw new Error(action.throw);
      if (action?.failed) {
        job.status = 'failed'; job.error = action.failed;
        return { endReason: 'engine_failed', stages: [], artifacts: [], songIds: [], error: action.failed };
      }
      job.status = 'succeeded'; job.result = { audioUrls: ['done.wav'], songIds: ['song'] };
      return { endReason: 'completed', stages: [], artifacts: [], songIds: ['song'], result: job.result };
    },
  };
  const getBackend = id => id === backendId ? backend : undefined;
  const effectiveSeed = extract('effectiveSeed');
  const finalizeAttempt = extract('finalizeAttempt', { effectiveSeed });
  const emptyOutcome = extract('emptyOutcome');
  // runGeneration's free identifiers must all be in the VM context, including
  // the backend-release guard and the lane helpers the mocked backend never
  // calls. Missing them makes the retry loop fail on a ReferenceError.
  const run = extract('runGeneration', {
    getBackend, emptyOutcome, pollUntilDone: () => 'poller',
    awaitBackendRelease: loadModule('server/src/services/generation/backendRelease.ts').awaitBackendRelease,
    jobs: new Map(), releaseGpuLane: lane.releaseGpuLane,
    gpuLaneNextFamily: lane.gpuLaneNextFamily, runOnGpuLane: lane.runOnGpuLane,
  });
  const routeContext = {
    runGeneration: run, finalizeAttempt, runOnGpuLane: lane.runOnGpuLane,
    gpuLaneBusy: lane.gpuLaneBusy, gpuLaneDepth: lane.gpuLaneDepth,
    noteEnqueued() {}, noteFinished: () => done(),
    logGeneration() {}, failGenerationLog() {}, AbortController,
    setTimeout: callback => { queueMicrotask(callback); return 0; },
    Math: { random: () => 0.5, floor: Math.floor }, structuredClone,
  };
  const enqueue = extract('enqueueGeneration', routeContext);
  const job = {
    id: 'attempt-job', status: 'pending', stage: 'Queued', progress: 0,
    params, mm3TakeSeeds, attempts: [], envelope: envelopeFor(backendId, maxAttempts),
  };
  const submittedBefore = JSON.stringify(job.envelope.submission);
  enqueue(job);
  let deadline;
  try {
    await Promise.race([finished, new Promise((_, reject) => {
      deadline = setTimeout(() => reject(new Error('attempt timeout')), 3000);
    })]);
  } finally { clearTimeout(deadline); }
  return { job, calls, submittedBefore };
}

function loadLane() {
  const source = read('server/src/services/generation/gpuLane.ts');
  const module = { exports: {} };
  vm.runInNewContext(transpile(source, 'gpuLane.ts'), { ...baseContext, module, exports: module.exports }, { filename: 'gpuLane.ts' });
  return module.exports;
}

test('thrown mapping failure retries, preserving first seed before reseeding', async () => {
  const run = await execute({ actions: [{ throw: 'mapping failed' }, {}] });
  assert.equal(run.job.status, 'succeeded');
  assert.equal(run.calls.length, 2);
  assert.deepEqual(run.calls.map(call => call.attempt), [1, 2]);
  assert.equal(run.job.attempts[0].effective.seed, 123);
  assert.notEqual(run.job.attempts[1].effective.seed, 123);
  assert.equal(run.job.attempts[0].reseeded, false);
  assert.equal(run.job.attempts[1].reseeded, true);
  assert.equal(run.job.attempts[0].error, 'mapping failed');
});

test('a consumed engine failure and a success each use one attempt', async () => {
  const failed = await execute({ actions: [{ failed: 'engine failed' }], maxAttempts: 2 });
  assert.equal(failed.job.status, 'failed');
  assert.equal(failed.job.attempts.length, 1);
  assert.equal(failed.job.attempts[0].endReason, 'engine_failed');
  const success = await execute({ actions: [{}], maxAttempts: 2 });
  assert.equal(success.job.status, 'succeeded');
  assert.equal(success.job.attempts.length, 1);
  assert.equal(success.job.attempts[0].endReason, 'completed');
});

test('MM3 uint64 seed strings and immutable submitted data survive attempts', async () => {
  const exact = '18226392072674864222';
  const run = await execute({ backendId: 'minimax-m3', actions: [{}], params: { seed: Number(exact) }, exactSeed: exact });
  assert.equal(run.job.attempts[0].effective.seed, exact);
  assert.equal(run.calls[0].submitted, run.submittedBefore);
  assert.equal(JSON.stringify(run.job.envelope.submission), run.submittedBefore);
  assert.equal(run.job.envelope.models.lm, 'q8_0');
  assert(Object.isFrozen(run.job.envelope) && Object.isFrozen(run.job.envelope.submission));
});
