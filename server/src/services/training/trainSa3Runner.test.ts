import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import type { ChildProcess, spawn } from 'node:child_process';

// Import the queue only after isolating its startup recovery and persistence.
const root = fs.mkdtempSync(path.join(process.cwd(), '.test-sa3-runner-'));
process.env.TRAINING_DIR = path.join(root, 'training');
process.env.DATA_DIR = path.join(root, 'library');
const { config } = await import('../../config.js');
config.aceServer.adapters = path.join(root, 'adapters');
config.aceServer.models = path.join(root, 'models');
config.aceServer.exe = path.join(root, 'bin', process.platform === 'win32' ? 'ace-server.exe' : 'ace-server');
fs.mkdirSync(path.dirname(config.aceServer.exe), { recursive: true });
fs.writeFileSync(config.aceServer.exe, 'mock');
const trainExe = path.join(path.dirname(config.aceServer.exe), process.platform === 'win32' ? 'ace-train.exe' : 'ace-train');
fs.writeFileSync(trainExe, 'mock');
for (const file of ['sa3-dit-base-BF16.gguf', 'sa3-dit-BF16.gguf', 'sa3-same-enc-F16.gguf', 'sa3-text-enc-BF16.gguf',
  'onnx/sa3/tokenizer.json', 'onnx/sa3/tokenizer_config.json', 'supersep/bs_leap_xe_voc-F32.gguf', 'supersep/bs_leap_xe_inst-F32.gguf']) {
  const full = path.join(config.aceServer.models, file);
  fs.mkdirSync(path.dirname(full), { recursive: true }); fs.writeFileSync(full, 'mock');
}
const layout = await import('./adapterLayout.js');
const cli = await import('./aceTrain.js');
const runner = await import('./trainSa3Runner.js');
const queue = await import('./labelingQueue.js');
const snapshots = await import('./sa3Snapshots.js');
const previews = await import('./sa3Preview.js');
after(() => fs.rmSync(root, { recursive: true, force: true }));
const hash = 'a'.repeat(64), identity = `stabilityai/stable-audio-3-medium sha256:${hash}`;

function gguf(file: string, base = identity) {
  const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
  const u64 = (n: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
  const str = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)]);
  const kv = (k: string, v: string) => Buffer.concat([str(k), u32(8), str(v)]);
  const tensor = (name: string, dims: number[], off: number) => Buffer.concat([str(`model.preprocess_conv.weight.${name}`),
    u32(dims.length), ...dims.map(u64), u32(0), u64(off)]);
  const header = Buffer.concat([Buffer.from('GGUF'), u32(3), u64(3), u64(3),
    kv('general.architecture', 'sa3-adapter'), kv('sa3.adapter_format', 'dora-rows-v1'), kv('sa3.inference_base_id', base),
    tensor('lora_a', [2, 1], 0), tensor('lora_b', [1, 2], 32), tensor('row_scale', [2], 64)]);
  const data = Buffer.alloc(72); data.writeFloatLE(1, 64); data.writeFloatLE(1, 68);
  fs.writeFileSync(file, Buffer.concat([header, Buffer.alloc((32 - header.length % 32) % 32), data]));
}

function fixture(name: string) {
  const run = runner.resolveSa3Run(`dataset-${name}`, name, { adapterName: 'adapter', steps: 2 });
  const job = queue.createJob('sa3-train', run.datasetId, [], run);
  fs.writeFileSync(run.manifest, '{"samples":[]}');
  const result = { done: true, stopped: false, error: '', baseSha256: hash, inferenceSha256: hash };
  const artifacts = () => {
    gguf(path.join(run.runDir, 'adapter.gguf'));
    for (const file of ['state.gguf', 'adapter-raw.gguf']) gguf(path.join(run.runDir, file));
  };
  return { job, run, result, artifacts };
}

test('SA3 options reject invalid values and unsupported backends before reserving runs', () => {
  for (const input of [{ steps: 0 }, { seed: -1 }, { rank: 1.5 }, { duration: Infinity }, { duration: 385 },
    { learningRate: '0.1' }, { checkpointEvery: 0 }]) assert.throws(() => cli.resolveSa3Recipe(input));
  assert.throws(() => runner.resolveSa3Run('dataset', 'invalid', { trainSeconds: true }), /Unsupported/);
  assert.throws(() => runner.resolveSa3Run('dataset', 'invalid', { adapterName: '../escape' }), /adapterName/);
  assert.equal(runner.sa3Readiness(config.aceServer.models, 'metal').available, false);
  assert.equal(runner.sa3Readiness(config.aceServer.models, 'vulkan').available, false);
  assert.equal(runner.sa3Readiness(config.aceServer.models, 'cpu').available, true);
  const a = runner.resolveSa3Run('dataset', 'versioned', {}), b = runner.resolveSa3Run('dataset', 'versioned', {});
  assert.notEqual(a.runDir, b.runDir);
  const args = cli.buildSa3TrainArgs(a);
  assert.equal(args[0], 'sa3-train'); assert.equal(args.includes('--train-seconds'), false);
  assert.equal(args.includes('--resume'), false);
});

test('SA3 publication validates finite tensors/provenance, hides partials and never overwrites', () => {
  const file = path.join(root, 'export.gguf'); gguf(file);
  assert.equal(layout.validateSa3Export(file, identity), identity);
  const name = layout.publishSa3Export(file, 'unit', identity);
  const second = layout.publishSa3Export(file, 'unit', identity);
  assert.equal(name, 'unit'); assert.equal(second, 'unit-2');
  const catalogue = path.join(config.aceServer.models, 'sa3-adapters');
  fs.writeFileSync(path.join(catalogue, '.incomplete.part'), 'partial');
  assert.deepEqual(layout.listSa3Adapters().map(a => a.name), ['unit', 'unit-2']);
  assert.throws(() => layout.publishSa3Export(file, 'bad-base', identity.replace(hash, 'b'.repeat(64))), /provenance/);
  const data = fs.readFileSync(file); data.writeFloatLE(NaN, data.length - 4); fs.writeFileSync(file, data);
  assert.throws(() => layout.publishSa3Export(file, 'bad', identity), /Non-finite/);
  fs.writeFileSync(file, data.subarray(0, -8));
  assert.throws(() => layout.validateSa3Export(file), /Truncated/);
  assert.deepEqual(layout.listSa3Adapters().map(a => a.name), ['unit', 'unit-2']);
});

test('SA3 success publishes, persists provenance and waits for engine restart', async () => {
  const f = fixture('success'); let restarted = false;
  await runner.runSa3TrainJob(f.job, { prepare: async () => {}, stop: async () => true,
    execute: async () => { f.artifacts(); return f.result; }, restart: async () => { restarted = true; return true; } });
  assert.equal(restarted, true); assert.equal(f.job.status, 'done');
  assert.equal(f.job.cleanupPending, false);
  assert.ok(layout.listSa3Adapters().some(a => a.name === f.run.published));
  const runs = runner.listSa3Runs(f.run.datasetId, 'success');
  assert.equal(runs[0].inferenceSha256, hash); assert.equal(runs[0].resumable, true);
  assert.equal(runs[0].published, f.run.published);
  const resumed = runner.resolveSa3Run(f.run.datasetId, 'success', { resumeRun: runs[0].run, steps: 4 });
  assert.equal(resumed.runDir, f.run.runDir); assert.equal(resumed.resume, true);
  assert.ok(cli.buildSa3TrainArgs(resumed).includes('--resume'));
  assert.throws(() => runner.resolveSa3Run(f.run.datasetId, 'success', { resumeRun: runs[0].run, rank: 32 }), /cannot change rank/);
  assert.throws(() => runner.resolveSa3Run('other-dataset', 'success', { resumeRun: runs[0].run }), /another dataset/);
  assert.throws(() => runner.resolveSa3Run(f.run.datasetId, 'success', { resumeRun: '../escape' }), /resumeRun/);
});

test('SA3 failures and malformed success restore the engine without publishing', async () => {
  for (const mode of ['prepare', 'stop', 'spawn', 'missing-done', 'artifact', 'restart']) {
    const f = fixture(`failure-${mode}`); let restarts = 0, publications = 0;
    await runner.runSa3TrainJob(f.job, {
      prepare: async () => { if (mode === 'prepare') throw new Error('prep failed'); },
      stop: async () => { if (mode === 'stop') throw new Error('stop failed'); return true; },
      execute: async () => { if (mode === 'spawn') throw new Error('spawn failed');
        if (mode !== 'artifact') f.artifacts(); return { ...f.result, done: mode !== 'missing-done' }; },
      restart: async () => { restarts++; return mode !== 'restart'; },
      publish: () => { publications++; return 'mock'; },
    });
    assert.equal(f.job.status, 'failed', mode);
    assert.equal(restarts, mode === 'prepare' ? 0 : 1, mode);
    assert.equal(publications, mode === 'restart' ? 1 : 0, mode);
  }
});

test('SA3 cancellation holds dataset admission and GPU queue through restart', async () => {
  const f = fixture('cancel'); let release!: () => void, entered!: () => void, next = false;
  const started = new Promise<void>(r => { entered = r; });
  const gate = new Promise<void>(r => { release = r; });
  let completed!: () => void;
  const finished = new Promise<void>(r => { completed = r; });
  queue.enqueue(f.job, async job => {
    await runner.runSa3TrainJob(job, { prepare: async () => {}, stop: async () => true,
      execute: async () => { queue.cancelJob(job.id); return { ...f.result, done: false, stopped: true }; },
      restart: async () => { entered(); await gate; return true; } });
    completed();
  });
  const second = queue.createJob('preprocess', 'next-dataset', [], {});
  queue.enqueue(second, async job => { next = true; queue.finishJob(job, 'done'); });
  await started;
  assert.equal(queue.activeJobForDataset(f.run.datasetId)?.id, f.job.id);
  assert.equal(next, false); assert.equal(f.run.published, undefined);
  release(); await finished;
  await new Promise(r => setImmediate(r));
  assert.equal(queue.activeJobForDataset(f.run.datasetId), undefined);
  assert.equal(queue.getJob(f.job.id), undefined);
  assert.equal(next, true); assert.equal(f.job.status, 'cancelled');
});

test('SA3 mocked child relays step metrics and refuses error/partial completion', async () => {
  for (const mode of ['ok', 'error', 'exit', 'spawn']) {
    const f = fixture(`child-${mode}`);
    const child = Object.assign(new EventEmitter(), { stdout: new PassThrough(), stderr: new PassThrough(),
      exitCode: null, signalCode: null });
    const launch = (() => {
      setImmediate(() => {
        if (mode === 'spawn') { child.emit('error', new Error('mock spawn failure')); return; }
        child.stdout.write(JSON.stringify({ type: 'start', backend: 'CPU', trainSeconds: false, baseSha256: hash, inferenceSha256: hash }) + '\n');
        child.stdout.write('{"type":"step","step":1,"loss":0.5,"ms":10}\n');
        child.stdout.write(mode === 'error' ? '{"type":"error","message":"mock trainer failure"}\n' : '{"type":"done"}\n');
        child.stdout.end(); child.emit('close', mode === 'exit' ? 1 : 0);
      });
      return child as unknown as ChildProcess;
    }) as typeof spawn;
    if (mode === 'ok') {
      const result = await runner.executeSa3Train(f.job, f.run, launch);
      assert.equal(result.done, true); assert.equal(f.job.done, 1);
      assert.ok(f.job.events.some(e => e.type === 'metric' && e.metric === 'step' && e.loss === 0.5));
    } else await assert.rejects(runner.executeSa3Train(f.job, f.run, launch), /mock|exited/);
    assert.equal(f.job.child, undefined);
  }
});

test('SA3 real cancellation hook writes STOP and waits for child close', async () => {
  const f = fixture('child-cancel');
  const child = Object.assign(new EventEmitter(), { stdout: new PassThrough(), stderr: new PassThrough(),
    exitCode: null, signalCode: null });
  const running = runner.executeSa3Train(f.job, f.run, (() => child as unknown as ChildProcess) as typeof spawn);
  assert.equal(queue.cancelJob(f.job.id), true);
  assert.equal(fs.readFileSync(path.join(f.run.runDir, 'STOP'), 'utf8'), 'stop\n');
  let finished = false; void running.then(() => { finished = true; });
  await new Promise(r => setImmediate(r)); assert.equal(finished, false);
  child.stdout.end(); child.emit('close', 0); await running;
  assert.equal(f.job.child, undefined); assert.equal(f.job.cancelChild, undefined);
});

test('SA3 HTTP admission, status, cancellation and picker discovery use the shared queue', async t => {
  const { initDb, closeDb } = await import('../../db/database.js');
  assert.ok(config.data.dbPath.startsWith(root + path.sep));
  fs.mkdirSync(config.data.dir, { recursive: true }); initDb();
  t.after(() => closeDb());
  const { insertDataset } = await import('./datasetsRepo.js');
  insertDataset({ id: 'http-dataset', slug: 'http-dataset', name: 'Fixture', sourceDir: root, recursive: false,
    customTag: '', tagPosition: 'prepend', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '',
    defaultLanguage: '', sampleCount: 1, labeledCount: 1, excludedCount: 0, status: 'labeled', builtAt: '',
    datasetJsonPath: '', albumName: '', createdAt: '', updatedAt: '' });
  const { default: express } = await import('express');
  const { default: trainingRoutes } = await import('../../routes/training.js');
  const { default: modelsRoutes } = await import('../../routes/models.js');
  const app = express(); app.use(express.json()); app.use('/api/training', trainingRoutes); app.use('/api/models', modelsRoutes);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(r => server.once('listening', r));
  t.after(() => new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve())));
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const url = `${base}/api/training/datasets/http-dataset/sa3-train`;
  let release!: () => void, started!: () => void, done!: () => void;
  const gate = new Promise<void>(r => { release = r; });
  const entered = new Promise<void>(r => { started = r; });
  const finished = new Promise<void>(r => { done = r; });
  queue.overrideTrainingRunner('sa3-train', async job => {
    job.cleanupPending = true; job.status = 'running'; started(); await gate;
    job.cleanupPending = false; queue.finishJob(job, 'done'); done();
  });
  t.after(() => queue.overrideTrainingRunner('sa3-train', null));
  const post = (body: object) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  assert.equal((await post({ steps: -1 })).status, 400);
  assert.equal((await fetch(`${base}/api/training/datasets/unknown/sa3-train`)).status, 404);
  const accepted = await post({ steps: 2 }); assert.equal(accepted.status, 202);
  const { jobId } = await accepted.json() as { jobId: string }; await entered;
  assert.equal((await post({ steps: 2 })).status, 409);
  const status = await (await fetch(url)).json() as { activeJob: { id: string }; runs: unknown[] };
  assert.equal(status.activeJob.id, jobId); assert.equal(status.runs.length, 1);
  assert.equal(queue.cancelJob(jobId), true);
  assert.equal((await post({ steps: 2 })).status, 409);
  release(); await finished;
  const picker = await (await fetch(`${base}/api/models/stablestep/adapters`)).json() as { adapters: Array<{ name: string }> };
  assert.ok(picker.adapters.some(a => a.name === 'unit'));
  assert.ok(picker.adapters.every(a => !a.name.includes('.part')));
});

// ── Checkpoint previews ────────────────────────────────────────────────────

function stateGguf(file: string, step: number) {
  const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
  const u64 = (n: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
  const str = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)]);
  fs.writeFileSync(file, Buffer.concat([Buffer.from('GGUF'), u32(3), u64(1), u64(3),
    str('general.architecture'), u32(8), str('sa3-train-state'), str('sa3t.fingerprint'), u32(8), str('fp'),
    str('sa3t.step'), u32(10), u64(step), Buffer.alloc(64)]));
}

const PREVIEW = { everySteps: 4, prompts: ['warm pads'], seconds: 5, seeds: [1, 2] };

function previewFixture(name: string, steps = 10) {
  const run = runner.resolveSa3Run(`dataset-${name}`, name, { adapterName: 'adapter', steps, preview: PREVIEW });
  const job = queue.createJob('sa3-train', run.datasetId, [], run);
  const result = { done: true, stopped: false, error: '', baseSha256: hash, inferenceSha256: hash };
  const calls: Array<{ steps: number; goalSteps?: number; exportEvery: number; resume: boolean }> = [];
  const order: string[] = [];
  const renders: Array<{ step: number; final: boolean }> = [];
  const deps = {
    prepare: async () => {},
    stop: async () => { order.push('stop'); return true; },
    restart: async () => { order.push('restart'); return true; },
    execute: async (_job: unknown, seg: typeof run) => {
      calls.push({ steps: seg.steps, goalSteps: seg.goalSteps, exportEvery: seg.exportEvery, resume: seg.resume });
      order.push(`train:${seg.steps}`);
      stateGguf(path.join(run.runDir, 'state.gguf'), seg.steps);
      gguf(path.join(run.runDir, 'adapter.gguf')); gguf(path.join(run.runDir, 'adapter-raw.gguf'));
      return result;
    },
    render: async (input: { step: number; final?: boolean }) => {
      renders.push({ step: input.step, final: !!input.final }); order.push(`render:${input.step}`);
      return { done: 2, failed: 0, skipped: 0 };
    },
  };
  return { run, job, result, calls, order, renders, deps };
}

test('SA3 previews segment on cadence multiples, capture once and publish only the final step', async () => {
  const f = previewFixture('cadence');
  assert.deepEqual(f.run.preview, PREVIEW);
  let published = 0;
  await runner.runSa3TrainJob(f.job, { ...f.deps, publish: (file, name, base) => { published++; return layout.publishSa3Export(file, name, base); } });
  assert.equal(f.job.status, 'done', f.job.error ?? '');
  assert.deepEqual(f.calls, [
    { steps: 4, goalSteps: 10, exportEvery: 0, resume: false },
    { steps: 8, goalSteps: 10, exportEvery: 0, resume: true },
    { steps: 10, goalSteps: 10, exportEvery: 0, resume: true }]);
  // Inference only after the trainer exits; the engine stops again before it resumes.
  assert.deepEqual(f.order, ['stop', 'train:4', 'restart', 'render:4', 'stop', 'train:8', 'restart', 'render:8',
    'stop', 'train:10', 'restart', 'render:10']);
  assert.deepEqual(f.renders.map(r => r.final), [false, false, true]);
  assert.deepEqual(snapshots.sa3SnapshotSteps(f.run.runDir), [4, 8, 10]);
  assert.equal(published, 1);
  assert.ok(fs.existsSync(path.join(f.run.runDir, 'adapter.gguf')));
  // One logical goal across segments.
  assert.ok(f.job.events.every(e => e.type !== 'progress' || e.phase !== 'loading-models' || e.total === 10));
});

test('SA3 snapshot capture is write-once and validated', () => {
  const dir = fs.mkdtempSync(path.join(root, 'capture-'));
  gguf(path.join(dir, 'adapter.gguf'));
  assert.equal(snapshots.captureSa3Snapshot(dir, 4, identity), true);
  const first = fs.readFileSync(snapshots.sa3SnapshotPath(dir, 4));
  gguf(path.join(dir, 'adapter.gguf'), identity.replace(hash, 'b'.repeat(64)));
  assert.equal(snapshots.captureSa3Snapshot(dir, 4, identity), false);
  assert.deepEqual(fs.readFileSync(snapshots.sa3SnapshotPath(dir, 4)), first);
  assert.throws(() => snapshots.captureSa3Snapshot(dir, 8, identity), /provenance/);
  assert.deepEqual(fs.readdirSync(path.join(dir, 'snapshots')), ['step-4.gguf']);
  stateGguf(path.join(dir, 'state.gguf'), 12);
  assert.equal(snapshots.sa3StateStep(dir), 12);
  assert.equal(snapshots.sa3StateStep(path.join(dir, 'none')), null);
  assert.deepEqual([0, 3, 4, 5, 9].map(d => snapshots.sa3NextBound(d, 4, 10)), [4, 4, 8, 8, 10]);
});

test('SA3 resume recovers a lost capture and renders owed previews first', async () => {
  const f = previewFixture('recover');
  // The server stopped after the segment to 8 exited: step 8 saved but never
  // captured; step 4 captured, its previews never rendered.
  gguf(path.join(f.run.runDir, 'adapter.gguf'));
  snapshots.captureSa3Snapshot(f.run.runDir, 4, identity);
  stateGguf(path.join(f.run.runDir, 'state.gguf'), 8);
  fs.writeFileSync(f.run.manifest, '{"samples":[]}');
  const id = `adapter/${path.basename(f.run.runDir)}`;
  const resumed = runner.resolveSa3Run(f.run.datasetId, 'recover', { resumeRun: id });
  assert.deepEqual(resumed.preview, PREVIEW);
  assert.throws(() => runner.resolveSa3Run(f.run.datasetId, 'recover', { resumeRun: id, preview: PREVIEW }), /keeps the preview/);
  const job = queue.createJob('sa3-train', resumed.datasetId, [], resumed);
  await runner.runSa3TrainJob(job, { ...f.deps, publish: () => 'mock' });
  assert.equal(job.status, 'done', job.error ?? '');
  assert.deepEqual(f.order, ['render:4', 'stop', 'train:8', 'restart', 'render:8', 'stop', 'train:10', 'restart', 'render:10']);
  assert.deepEqual(f.calls.map(c => c.resume), [true, true]);
});

test('SA3 preview failure continues only after cleanup; failed restoration aborts', async () => {
  let stops = 0;
  const cases: Array<[string, Record<string, unknown>, string, number]> = [
    ['render-throws', { render: async () => { throw new Error('disk full'); } }, 'done', 1],
    ['stop-after-preview', { stop: async () => ++stops < 2 }, 'failed', 0],
    ['restart-between', { restart: async () => false }, 'failed', 0],
  ];
  for (const [name, override, status, publications] of cases) {
    const f = previewFixture(`rules-${name}`);
    let published = 0;
    await runner.runSa3TrainJob(f.job, { ...f.deps, ...override, publish: () => { published++; return 'mock'; } });
    assert.equal(f.job.status, status, `${name}: ${f.job.error}`);
    assert.equal(published, publications, name);
  }
  const wrong = previewFixture('wrong-step');
  await runner.runSa3TrainJob(wrong.job, { ...wrong.deps, publish: () => 'mock', execute: async (_j, seg) => {
    stateGguf(path.join(wrong.run.runDir, 'state.gguf'), seg.steps - 1); return wrong.result; } });
  assert.match(wrong.job.error ?? '', /saved step 3, expected 4/);
});

test('SA3 cancel during a preview stops before publishing and restores the engine', async () => {
  const f = previewFixture('preview-cancel');
  let published = 0;
  await runner.runSa3TrainJob(f.job, { ...f.deps, publish: () => { published++; return 'mock'; },
    render: async () => { queue.cancelJob(f.job.id); return { done: 0, failed: 0, skipped: 2 }; } });
  assert.equal(f.job.status, 'cancelled'); assert.equal(published, 0);
  assert.deepEqual(f.order, ['stop', 'train:4', 'restart']);
});

test('SA3 preview renders record each cell once, keep going past failures and retry only what is owed', async () => {
  const dir = fs.mkdtempSync(path.join(root, 'render-'));
  const queries: URLSearchParams[] = [];
  let fail = true;
  const deps = { tokenize: async () => ({ ids: [1, 2], nTokens: 2 }),
    synth: async (q: URLSearchParams) => {
      queries.push(q);
      if (fail && q.get('seed') === '1') throw new Error('engine 400');
      return Buffer.concat([Buffer.from('RIFF'), Buffer.alloc(60)]);
    } };
  const input = { runDir: dir, runId: 'adapter/2026-01-01_00-00-00', snapshotRun: 'slug/adapter/2026-01-01_00-00-00', step: 4,
    settings: PREVIEW, inferenceDit: 'sa3-dit-BF16.gguf', deps };
  assert.deepEqual(await previews.renderSa3Previews(input), { done: 1, failed: 1, skipped: 0 });
  assert.equal(queries[0].get('snapshot_run'), 'slug/adapter/2026-01-01_00-00-00');
  assert.equal(queries[0].get('snapshot_step'), '4');
  assert.equal(queries[0].get('synth_model'), 'sa3-dit-BF16.gguf');
  assert.equal(queries[0].get('duration'), '5');
  assert.equal(previews.sa3PreviewPending(dir, input.runId, 4, PREVIEW), false);
  fail = false;
  assert.deepEqual(await previews.renderSa3Previews(input), { done: 1, failed: 0, skipped: 0 });
  assert.deepEqual(await previews.renderSa3Previews(input), { done: 0, failed: 0, skipped: 0 });
  assert.equal(queries.length, 3);
  const listed = previews.listSa3PreviewRecords(dir);
  assert.deepEqual(listed.map(r => [r.seed, r.status]), [[1, 'done'], [2, 'done']]);
  assert.ok(listed.every(r => r.file && previews.sa3PreviewAudioPath(dir, r.file)));
  assert.equal(previews.sa3PreviewAudioPath(dir, '../index.json'), null);
  // A render cut off by a crash reads as failed and is owed again.
  const index = path.join(dir, 'previews', 'index.json');
  const stale = JSON.parse(fs.readFileSync(index, 'utf8'));
  stale.push({ ...stale[0], step: 8, status: 'rendering', file: undefined });
  fs.writeFileSync(index, JSON.stringify(stale));
  assert.equal(previews.listSa3PreviewRecords(dir).find(r => r.step === 8)?.status, 'failed');
  assert.equal(previews.sa3PreviewPending(dir, input.runId, 8, PREVIEW), true);
  // Every cell of a step cut off: still owed (raw status, not the list view's failed).
  const cut = JSON.parse(fs.readFileSync(index, 'utf8')).filter((r: { step: number }) => r.step !== 8);
  cut.push(...[1, 2].map((seed, i) => ({ ...cut[0], step: 12, seed, status: i ? 'queued' : 'rendering', file: undefined })));
  fs.writeFileSync(index, JSON.stringify(cut));
  assert.deepEqual(previews.listSa3PreviewRecords(dir).filter(r => r.step === 12).map(r => r.status), ['failed', 'failed']);
  assert.equal(previews.sa3PreviewPending(dir, input.runId, 12, PREVIEW), true);
  assert.deepEqual(await previews.renderSa3Previews({ ...input, step: 12 }), { done: 2, failed: 0, skipped: 0 });
  assert.equal(previews.sa3PreviewPending(dir, input.runId, 12, PREVIEW), false);
  // A recorded name only, and never through a previews folder that leaves the run.
  const done = previews.listSa3PreviewRecords(dir).find(r => r.status === 'done')!.file!;
  fs.copyFileSync(path.join(dir, 'previews', done), path.join(dir, 'previews', 'step-4-p0-s9-00000000.wav'));
  assert.equal(previews.sa3PreviewAudioPath(dir, 'step-4-p0-s9-00000000.wav'), null);
  const outside = fs.mkdtempSync(path.join(root, 'outside-'));
  fs.cpSync(path.join(dir, 'previews'), outside, { recursive: true });
  fs.rmSync(path.join(dir, 'previews'), { recursive: true });
  fs.symlinkSync(outside, path.join(dir, 'previews'), 'junction');
  assert.equal(previews.sa3PreviewAudioPath(dir, done), null);
  // unlink, not rmdir: the previews entry is a link, and rmdir on a POSIX
  // symlink fails with ENOTDIR. unlink removes the link, never its target.
  fs.unlinkSync(path.join(dir, 'previews'));
  fs.renameSync(outside, path.join(dir, 'previews'));
  assert.ok(previews.sa3PreviewAudioPath(dir, done));
  const ac = new AbortController(); ac.abort();
  assert.deepEqual(await previews.renderSa3Previews({ ...input, step: 8, signal: ac.signal }), { done: 0, failed: 0, skipped: 2 });
  assert.throws(() => previews.parseSa3PreviewSettings({ everySteps: 4, prompts: [] }), /prompt/);
  assert.throws(() => previews.parseSa3PreviewSettings({ ...PREVIEW, seeds: [1, 2, 3, 4, 5] }), /seeds/);
  assert.throws(() => previews.parseSa3PreviewSettings({ ...PREVIEW, seconds: 0 }), /seconds/);
  assert.throws(() => previews.parseSa3PreviewSettings({ ...PREVIEW, path: 'x' }), /Unsupported/);
});

test('SA3 preview routes resolve opaque run ids and serve only recorded audio', async t => {
    const { initDb, closeDb } = await import('../../db/database.js');
  fs.mkdirSync(config.data.dir, { recursive: true }); initDb();
    t.after(() => closeDb());
  const { insertDataset } = await import('./datasetsRepo.js');
  insertDataset({ id: 'preview-http', slug: 'preview-http', name: 'Preview fixture', sourceDir: path.join(root, 'preview-src'), recursive: false,
    customTag: '', tagPosition: 'prepend', genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '',
    defaultLanguage: '', sampleCount: 1, labeledCount: 1, excludedCount: 0, status: 'labeled', builtAt: '',
    datasetJsonPath: '', albumName: '', createdAt: '', updatedAt: '' });
  const { default: express } = await import('express');
  const { default: trainingRoutes } = await import('../../routes/training.js');
  const app = express(); app.use(express.json()); app.use('/api/training', trainingRoutes);
    const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(r => server.once('listening', r));
  t.after(() => new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve())));
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}/api/training/datasets`;
  const empty = await (await fetch(`${base}/preview-http/sa3-previews`)).json() as { run: null; previews: unknown[] };
  assert.equal(empty.run, null); assert.deepEqual(empty.previews, []);
  const run = runner.resolveSa3Run('preview-http', 'preview-http', { adapterName: 'adapter', steps: 8, preview: PREVIEW });
  const id = `adapter/${path.basename(run.runDir)}`;
  gguf(path.join(run.runDir, 'adapter.gguf')); snapshots.captureSa3Snapshot(run.runDir, 4, identity);
  await previews.renderSa3Previews({ runDir: run.runDir, runId: id, snapshotRun: `preview-http/${id}`, step: 4, settings: PREVIEW,
    inferenceDit: 'sa3-dit-BF16.gguf', deps: { tokenize: async () => ({ ids: [1], nTokens: 1 }),
      synth: async q => { if (q.get('seed') === '2') throw new Error('engine 400'); return Buffer.concat([Buffer.from('RIFF'), Buffer.alloc(60)]); } } });
  const list = await (await fetch(`${base}/preview-http/sa3-previews?run=${encodeURIComponent(id)}`)).json() as
    { run: string; settings: unknown; previews: Array<{ seed: number; status: string; audioUrl: string | null; file?: string }> };
  assert.equal(list.run, id); assert.deepEqual(list.settings, PREVIEW);
  assert.deepEqual(list.previews.map(p => [p.seed, p.status, !!p.audioUrl]), [[1, 'done', true], [2, 'failed', false]]);
  assert.ok(list.previews.every(p => p.file === undefined));
  const audio = await fetch(`http://127.0.0.1:${(server.address() as { port: number }).port}${list.previews[0].audioUrl}`);
  assert.equal(audio.status, 200); assert.equal((await audio.arrayBuffer()).byteLength, 64);
  const q = (run: string, file: string) => `${base}/preview-http/sa3-previews/audio?run=${encodeURIComponent(run)}&file=${encodeURIComponent(file)}`;
  assert.equal((await fetch(q(id, '../run.json'))).status, 404);
  assert.equal((await fetch(q('../../escape', 'x.wav'))).status, 400);
  assert.equal((await fetch(q('adapter/1999-01-01_00-00-00', 'x.wav'))).status, 404);
  assert.equal((await fetch(`${base}/preview-http/sa3-previews?run=${encodeURIComponent('C:/abs')}`)).status, 400);
  assert.equal((await fetch(`${base}/other/sa3-previews`)).status, 404);
  const render = (body: object) => fetch(`${base}/preview-http/sa3-previews/render`, { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  assert.equal((await render({ run: id, step: 8 })).status, 400);
  assert.equal((await render({ run: id, step: 4, settings: { seeds: [] } })).status, 400);
  // The engine is not up in this harness: a valid request stops at the readiness check.
  assert.equal((await render({ run: id, step: 4 })).status, 503);
});
