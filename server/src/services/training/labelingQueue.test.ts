import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Job records live under the training dir: a fresh one, always, before config loads.
process.env.TRAINING_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'labeling-queue-'));
const q = await import('./labelingQueue.js');
/** A second copy of the module, as a restarted process would load it, then
 *  its startup reconcile (index.ts) over the same training dir. */
const restart = async () => {
  const m = await import(`./labelingQueue.js?restart=${Math.random()}`) as typeof q;
  m.recoverStaleJobs();
  return m;
};

test.after(() => fs.rmSync(process.env.TRAINING_DIR!, { recursive: true, force: true }));

test('finished joint history recovers segmented metrics, meters and resumed step timing', async () => {
  const output = path.join(process.env.TRAINING_DIR!, 'replay');
  const a = path.join(output, 'segments', 'segment-000001');
  const b = path.join(output, 'segments', 'segment-000002');
  fs.mkdirSync(a, { recursive: true }); fs.mkdirSync(b, { recursive: true });
  const metric = (step: number, ms: number) => JSON.stringify({ stage: 'joint', step, ar_ce: 1, ar_kl: 0.5, nar_mse: 2, step_ms: ms });
  fs.writeFileSync(path.join(a, 'train.jsonl'), [metric(1, 10), metric(2, 99), '{broken'].join('\n'));
  fs.writeFileSync(path.join(b, 'train.jsonl'), [metric(2, 20), JSON.stringify({ stage: 'meters', step: 2, nar_recon: 0.7, nar_drift: 0.2 })].join('\n'));
  const job = q.createJob('yue2-joint-train', 'replay-set', [], {});
  q.finishJob(job, 'done');
  const frames = await q.finishedJointFrames(output, q.toSummary(job));
  assert.equal(frames.length, 2);
  assert.deepEqual(frames.map(frame => frame.type === 'metric' && [frame.step, frame.loss, frame.trainMs]), [[1, 3.1, 10], [2, 3.1, 30]]);
  assert.equal(frames[1].type === 'metric' && frames[1].narRecon, 0.7);
  assert.deepEqual(await q.finishedJointFrames(path.join(output, 'missing'), q.toSummary(job)), []);
});

test('a job is on disk from creation and its running state is kept current', () => {
  const job = q.createJob('build', 'ds-a', [], {});
  const metaOf = () => q.listJobs('ds-a').find(j => j.id === job.id);
  assert.equal(metaOf()?.status, 'queued');
  job.status = 'running';
  job.startedAt = Date.now();
  job.phase = 'train';
  q.emitJob(job);
  // listJobs prefers the live copy; read the file the way a restart would.
  const onDisk = JSON.parse(fs.readFileSync(path.join(process.env.TRAINING_DIR!, 'jobs', job.id, '_meta.json'), 'utf8'));
  assert.deepEqual([onDisk.status, onDisk.phase, onDisk.startedAt], ['running', 'train', job.startedAt]);
  assert.equal(q.listJobs('ds-a').filter(j => j.id === job.id).length, 1, 'listed once, not once per copy');
});

test('after a restart, queued and running jobs read back failed "Server restarted"; finished ones are untouched', async () => {
  const queued = q.createJob('build', 'ds-b', [], {});
  const running = q.createJob('train-dit', 'ds-b', [], {});
  running.status = 'running';
  running.startedAt = Date.now();
  q.emitJob(running);
  const finished = q.createJob('train-lm', 'ds-b', [], {});
  q.finishJob(finished, 'done');

  const after = await restart();
  const byId = new Map(after.listJobs('ds-b').map(j => [j.id, j]));
  for (const job of [queued, running]) {
    const meta = byId.get(job.id);
    assert.deepEqual([meta?.status, meta?.error], ['failed', 'Server restarted'], job.kind);
    assert.ok(typeof meta?.finishedAt === 'number');
  }
  assert.deepEqual([byId.get(finished.id)?.status, byId.get(finished.id)?.error], ['done', null]);
  assert.equal(after.getJob(running.id), undefined, 'nothing is resumed');
  assert.equal(after.activeJobForDataset('ds-b'), undefined, 'the dataset is free for a new job');
});

test('a meta write that fails part-way leaves the previous record whole, and a restart still finds the job', async (t) => {
  const job = q.createJob('train-lm', 'ds-c', [], {});
  const dir = path.join(process.env.TRAINING_DIR!, 'jobs', job.id);
  const before = fs.readFileSync(path.join(dir, '_meta.json'), 'utf8');
  // The next write leaves a truncated file behind, then fails.
  const real = fs.writeFileSync;
  const write = t.mock.method(fs, 'writeFileSync', (file: fs.PathOrFileDescriptor, ...rest: unknown[]) => {
    if (String(file).includes(job.id)) { real(file, '{'); throw new Error('disk full'); }
    return (real as (...a: unknown[]) => void)(file, ...rest);
  });
  job.status = 'running';
  q.emitJob(job);
  write.mock.restore();
  assert.equal(fs.readFileSync(path.join(dir, '_meta.json'), 'utf8'), before, 'the queued record survives');
  assert.deepEqual(fs.readdirSync(dir), ['_meta.json'], 'no temp file left behind');
  const after = await restart();
  const meta = after.listJobs('ds-c').find(j => j.id === job.id);
  assert.deepEqual([meta?.status, meta?.error], ['failed', 'Server restarted']);
});

test('a preview and a GPU job exclude each other across datasets', async () => {
  const fresh = await restart();
  const gate = (): { wait: Promise<void>; open: () => void } => { let open!: () => void; return { wait: new Promise<void>(r => { open = r; }), open }; };

  // A preview on dataset A is reserved: no second preview, no GPU job is admitted to run beside it.
  assert.equal(fresh.reserveGpuForPreview(), true);
  assert.equal(fresh.reserveGpuForPreview(), false);
  assert.equal(fresh.gpuPreviewReserved(), true);
  const render = gate();
  const events: string[] = [];
  const preview = fresh.runReservedPreview(async () => { events.push('preview start'); await render.wait; events.push('preview end'); });
  // A trainer on dataset B queues behind it on the same chain: it must not start while the render runs.
  const trainer = fresh.createJob('heartmula-train', 'ds-B', [], {});
  const started = gate();
  fresh.enqueue(trainer, async () => { events.push('train start'); started.open(); });
  await new Promise(r => setTimeout(r, 30));
  assert.deepEqual(events, ['preview start'], 'the trainer waits for the render');
  render.open();
  await preview;
  await started.wait;
  assert.deepEqual(events, ['preview start', 'preview end', 'train start']);
  assert.equal(fresh.gpuPreviewReserved(), false, 'the reservation ends with the render');

  // The other way round: a GPU job queued or running anywhere refuses a preview.
  const busy = fresh.createJob('heartmula-train', 'ds-C', [], {});
  assert.equal(fresh.gpuJobActive(), true);
  assert.equal(fresh.reserveGpuForPreview(), false);
  fresh.finishJob(busy, 'done');
  fresh.finishJob(trainer, 'done');
  assert.equal(fresh.gpuJobActive(), false);
  assert.equal(fresh.reserveGpuForPreview(), true);
  await fresh.runReservedPreview(async () => {});
});
