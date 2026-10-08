import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { WorkflowJobs } from '../workflows/workflowJobs.js';
import { lyricBatchInput, runLyricBatch } from './lyricWorkflow.js';

const audio = {
  enqueue: () => { throw new Error('unexpected audio'); },
  get: () => { throw new Error('unexpected audio'); },
  cancel: () => { throw new Error('unexpected audio'); },
};
const input = { items: [
  { type: 'preflight-error', error: 'source missing' },
  { type: 'preflight-error', error: 'provider unavailable' },
] };

test('lyric batch records each failure in both the result and replay stream; duplicate submit reuses the job', async () => {
  const jobs = new WorkflowJobs({ db: new Database(':memory:'), audio });
  jobs.register({ kind: 'lyric-test', input: lyricBatchInput, run: runLyricBatch });
  const first = jobs.submit({ kind: 'lyric-test', idempotencyKey: 'same', input }, 'user');
  const duplicate = jobs.submit({ kind: 'lyric-test', idempotencyKey: 'same', input }, 'user');
  assert.equal(duplicate.created, false);
  assert.equal(duplicate.job.id, first.job.id);
  await jobs.settled();
  const replay = jobs.replay(first.job.id, 'user', 0);
  assert.deepEqual((replay.job.result as any).results.map((r: any) => r.error), ['source missing', 'provider unavailable']);
  assert.deepEqual(replay.events.filter(e => e.type === 'item-error').map(e => (e.data as any).error), ['source missing', 'provider unavailable']);
  const cursor = replay.events[2].seq;
  assert.deepEqual(jobs.replay(first.job.id, 'user', cursor).events.map(e => e.seq), replay.events.filter(e => e.seq > cursor).map(e => e.seq));
});

test('lyric batch cancel is acknowledged and a restarted run remains interrupted until explicit retry', async () => {
  const db = new Database(':memory:');
  let release!: () => void;
  const hold = new Promise<void>(resolve => { release = resolve; });
  const before = new WorkflowJobs({ db, audio });
  before.register({ kind: 'lyric-test', input: lyricBatchInput, run: async ctx => { await hold; return runLyricBatch(ctx); } });
  const cancelled = before.submit({ kind: 'lyric-test', idempotencyKey: 'cancel', input }, 'user').job;
  assert.equal(before.cancel(cancelled.id, 'user').cancelRequested, true);
  release();
  await before.settled();
  assert.equal(before.get(cancelled.id).status, 'cancelled');
  let releaseSecond!: () => void;
  const secondHold = new Promise<void>(resolve => { releaseSecond = resolve; });
  const original = new WorkflowJobs({ db, audio });
  original.register({ kind: 'lyric-test', input: lyricBatchInput, run: async ctx => { await secondHold; return runLyricBatch(ctx); } });
  const caught = original.submit({ kind: 'lyric-test', idempotencyKey: 'restart', input }, 'user').job;
  const after = new WorkflowJobs({ db, audio });
  after.reconcileAfterRestart();
  assert.equal(after.get(caught.id).status, 'interrupted');
  after.register({ kind: 'lyric-test', input: lyricBatchInput, run: runLyricBatch });
  await after.settled();
  assert.equal(after.get(caught.id).status, 'interrupted');
  after.retry(caught.id, 'user');
  await after.settled();
  assert.equal(after.get(caught.id).status, 'succeeded');
  assert.equal(after.replay(caught.id, 'user', 0).events.at(-1)?.type, 'status');
  releaseSecond();
  await original.settled();
});
