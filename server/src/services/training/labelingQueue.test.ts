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
