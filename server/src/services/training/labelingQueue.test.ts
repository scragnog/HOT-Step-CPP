import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Job records live under the training dir: a fresh one, always, before config loads.
process.env.TRAINING_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'labeling-queue-'));
const q = await import('./labelingQueue.js');
/** A second copy of the module, as a restarted process would load it: its
 *  startup reconcile runs again over the same training dir. */
const restart = () => import(`./labelingQueue.js?restart=${Math.random()}`) as Promise<typeof q>;

test.after(() => fs.rmSync(process.env.TRAINING_DIR!, { recursive: true, force: true }));

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
