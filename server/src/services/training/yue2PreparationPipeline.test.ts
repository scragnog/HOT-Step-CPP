import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import type { TrainingSnapshot } from '../../contracts/trainingOperation.js';
import type { Yue2PreparationPayload } from '../../contracts/trainingPreparation.js';
import { Yue2PreparationPipeline, type Yue2PreparationAdapter,
  type Yue2PreparationArtifacts } from './yue2PreparationPipeline.js';

const artifacts = (): Yue2PreparationArtifacts => ({
  preprocess: { done: false, captionModeOk: false }, tokenize: { done: false },
  sheet: { done: false }, align: { done: false, stemsReady: 0 }, minted: { present: false },
  narCompleted: false, arCompleted: false,
});

function snapshot(stages: Yue2PreparationPayload['stages'], mode: Yue2PreparationPayload['mode'] = 'prepare-only', key = 'one'):
  TrainingSnapshot<Yue2PreparationPayload> {
  return { version: 1, operation: { kind: 'yue2-preparation', idempotencyKey: key },
    worker: { kind: 'local' }, dataset: { id: 'dataset', revision: 'row-1' },
    sources: [{ kind: 'dataset-sources', id: 'dataset', revision: 'sources-1' }],
    payload: { mode, stages, trigger: '', lyricTiming: true, recipes: {} } };
}

function fixture(opts: { artifacts?: Yue2PreparationArtifacts; job?: 'done' | 'failed' | 'queued'; schedule?: (fn: () => void) => void } = {}) {
  const dir = mkdtempSync(path.join(tmpdir(), 'yue2-prep-'));
  const started: Array<{ stage: string; worker: string; workerUrl: string | null; body: Record<string, unknown> }> = [];
  let job = opts.job ?? 'done';
  let cancelled = false;
  let dataset = 'row-1';
  let source = 'sources-1';
  const adapter: Yue2PreparationAdapter = {
    datasetRevision: async () => dataset,
    sourceRevision: async () => source,
    admission: async () => {},
    artifacts: async () => opts.artifacts ?? artifacts(),
    start: async (s, stage, body, workerUrl) => { started.push({ stage, worker: s.worker.kind, workerUrl, body }); return `job-${started.length}`; },
    job: async () => ({ status: cancelled ? 'cancelled' : job }),
    cancel: async () => { cancelled = true; },
  };
  const runner = new Yue2PreparationPipeline(dir, adapter, opts.schedule,
    ms => new Promise(resolve => setTimeout(resolve, Math.min(ms, 5))));
  return { dir, runner, started, adapter,
    setJob: (v: typeof job) => { job = v; }, setSource: (v: string) => { source = v; },
    setDataset: (v: string) => { dataset = v; },
    cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

async function until(fn: () => boolean): Promise<void> {
  for (let i = 0; i < 100; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 5)); }
  throw new Error('Timed out waiting for fake stage');
}

test('prerequisite skips use fresh artifacts and prepare-only does not start a trainer', async () => {
  const ready = artifacts();
  ready.preprocess = { done: true, captionModeOk: true };
  ready.tokenize.done = true; ready.sheet.done = true; ready.align = { done: true, stemsReady: 1 };
  const f = fixture({ artifacts: ready });
  try {
    await assert.rejects(f.runner.start(snapshot(['nar']), {}), /Preparation only cannot start/);
    const s = await f.runner.start(snapshot(['latents', 'codes', 'sheet', 'stems', 'align']), {});
    await until(() => f.runner.get(s.id)?.status === 'done');
    assert.deepEqual(f.runner.get(s.id)?.stages.map(x => x.status), Array(5).fill('skipped'));
    assert.equal(f.started.length, 0);
  } finally { f.cleanup(); }
});

test('captured worker, form and source survive another client; duplicate key returns one operation', async () => {
  const f = fixture({ job: 'queued' });
  try {
    const s = snapshot(['nar'], 'train-after-preparation');
    s.worker = { kind: 'remote', name: 'worker-a' };
    const body = { steps: 5, rank: 16 };
    const first = await f.runner.start(s, { nar: body }, 'https://worker.test');
    body.steps = 99;
    const same = await f.runner.start(s, { nar: body }, 'https://worker.test');
    assert.equal(same.id, first.id);
    await assert.rejects(f.runner.start(snapshot(['nar'], 'train-after-preparation', 'other'), {}), /active/);
    await until(() => f.started.length === 1);
    assert.equal(f.started[0].worker, 'remote');
    assert.equal(f.started[0].workerUrl, 'https://worker.test');
    assert.equal(f.started[0].body.steps, 5);
    await f.runner.cancel(first.id);
    await until(() => f.runner.get(first.id)?.status === 'cancelled');
  } finally { f.cleanup(); }
});

test('failure and explicit retry resume from the failed stage only', async () => {
  const f = fixture({ job: 'failed' });
  try {
    const s = await f.runner.start(snapshot(['latents', 'codes']), {});
    await until(() => f.runner.get(s.id)?.status === 'failed');
    assert.deepEqual(f.started.map(x => x.stage), ['latents']);
    f.adapter.job = async (_snapshot, id) => ({ status: id === 'job-1' ? 'failed' : 'done' });
    await f.runner.retry(s.id);
    await until(() => f.runner.get(s.id)?.status === 'done');
    assert.deepEqual(f.started.map(x => x.stage), ['latents', 'latents', 'codes']);
  } finally { f.cleanup(); }
});

test('retry reconciles a submitted stage that finished while its status read failed', async () => {
  const f = fixture({ job: 'failed' });
  try {
    const s = await f.runner.start(snapshot(['latents', 'codes']), {});
    await until(() => f.runner.get(s.id)?.status === 'failed');
    f.setJob('done');
    await f.runner.retry(s.id);
    await until(() => f.runner.get(s.id)?.status === 'done');
    assert.deepEqual(f.started.map(x => x.stage), ['latents', 'codes']);
  } finally { f.cleanup(); }
});

test('retry refuses a competing preparation accepted during worker admission', async () => {
  const f = fixture({ job: 'failed' });
  try {
    const failed = await f.runner.start(snapshot(['latents']), {});
    await until(() => f.runner.get(failed.id)?.status === 'failed');
    let release!: () => void;
    let entered!: () => void;
    const waiting = new Promise<void>(resolve => { release = resolve; });
    const admissionEntered = new Promise<void>(resolve => { entered = resolve; });
    f.adapter.admission = async s => {
      if (s.operation.idempotencyKey === 'one') { entered(); await waiting; }
    };
    const retry = f.runner.retry(failed.id);
    await admissionEntered;
    f.setJob('queued');
    const winner = await f.runner.start(snapshot(['latents'], 'prepare-only', 'two'), {});
    release();
    await assert.rejects(retry, /Another pipeline or job is active/);
    assert.equal(f.runner.get(failed.id)?.status, 'failed');
    await until(() => f.started.length === 2);
    await f.runner.cancel(winner.id);
    await until(() => f.runner.get(winner.id)?.status === 'cancelled');
    assert.equal(f.started.length, 2);
  } finally { f.cleanup(); }
});

test('restart marks an accepted operation interrupted and never auto-submits it', async () => {
  const scheduled: Array<() => void> = [];
  const f = fixture({ schedule: fn => { scheduled.push(fn); } });
  try {
    const s = await f.runner.start(snapshot(['latents']), {});
    const restarted = new Yue2PreparationPipeline(f.dir, f.adapter, fn => { scheduled.push(fn); });
    assert.equal(restarted.get(s.id)?.status, 'interrupted');
    assert.equal(f.started.length, 0);
    await restarted.retry(s.id);
    assert.equal(scheduled.length, 2);
    scheduled[1]();
    await until(() => restarted.get(s.id)?.status === 'done');
    assert.deepEqual(f.started.map(x => x.stage), ['latents']);
  } finally { f.cleanup(); }
});

test('a source revision change fails before another stage starts', async () => {
  const f = fixture({ job: 'queued' });
  try {
    const s = await f.runner.start(snapshot(['latents', 'codes']), {});
    await until(() => f.started.length === 1);
    f.setSource('sources-2');
    f.setJob('done');
    await until(() => f.runner.get(s.id)?.status === 'failed');
    assert.deepEqual(f.started.map(x => x.stage), ['latents']);
    assert.match(f.runner.get(s.id)?.error ?? '', /sources changed/);
  } finally { f.cleanup(); }
});

test('a dataset row edit fails before another captured stage starts', async () => {
  const f = fixture({ job: 'queued' });
  try {
    const s = await f.runner.start(snapshot(['latents', 'codes']), {});
    await until(() => f.started.length === 1);
    f.setDataset('row-2');
    f.setJob('done');
    await until(() => f.runner.get(s.id)?.status === 'failed');
    assert.deepEqual(f.started.map(x => x.stage), ['latents']);
    assert.match(f.runner.get(s.id)?.error ?? '', /Dataset changed/);
  } finally { f.cleanup(); }
});

test('pause holds at the stage boundary and cancel acknowledges the running job', async () => {
  const f = fixture({ job: 'queued' });
  try {
    const s = await f.runner.start(snapshot(['latents', 'codes']), {});
    await until(() => f.started.length === 1);
    f.runner.pause(s.id);
    f.setJob('done');
    await until(() => f.runner.get(s.id)?.status === 'paused');
    assert.deepEqual(f.started.map(x => x.stage), ['latents']);
    f.runner.resume(s.id);
    await until(() => f.runner.get(s.id)?.status === 'done');
    assert.deepEqual(f.started.map(x => x.stage), ['latents', 'codes']);
  } finally { f.cleanup(); }

  const g = fixture({ job: 'queued' });
  try {
    const s = await g.runner.start(snapshot(['latents', 'codes']), {});
    await until(() => g.started.length === 1);
    await g.runner.cancel(s.id);
    await until(() => g.runner.get(s.id)?.status === 'cancelled');
    assert.deepEqual(g.started.map(x => x.stage), ['latents']);
    assert.equal(g.runner.get(s.id)?.stages[1].status, 'cancelled');
  } finally { g.cleanup(); }
});

test('offline admission refuses a command before any stage or record is created', async () => {
  const f = fixture();
  try {
    f.adapter.admission = async () => { throw new Error('worker offline'); };
    await assert.rejects(f.runner.start(snapshot(['latents']), {}), /worker offline/);
    assert.deepEqual(f.runner.list(), []);
    assert.equal(f.started.length, 0);
  } finally { f.cleanup(); }
});

test('an unreadable durable record fails closed on the next start', async () => {
  const f = fixture();
  try {
    writeFileSync(path.join(f.dir, 'damaged.json'), '{');
    const restarted = new Yue2PreparationPipeline(f.dir, f.adapter);
    await assert.rejects(restarted.start(snapshot(['latents']), {}), /unreadable/);
  } finally { f.cleanup(); }
});
