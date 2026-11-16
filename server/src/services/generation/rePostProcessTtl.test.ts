// The re-post-process job TTL counts from when the job finished, not when it
// was created, so a job that queued for a long time is still readable.
// Run: npx tsx --test src/services/generation/rePostProcessTtl.test.ts

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pruneJobs, JOB_TTL_MS, type RePostProcessJob } from './rePostProcess.js';

test('TTL runs from finishedAt; running jobs and fresh finishes survive', () => {
  const now = 10 * JOB_TTL_MS;
  const job = (id: string, finishedAt?: number): RePostProcessJob =>
    ({ id, songId: 's', status: finishedAt ? 'succeeded' : 'running', stage: '', createdAt: 0, finishedAt });
  const map = new Map([
    ['old', job('old', now - JOB_TTL_MS - 1)],
    ['fresh', job('fresh', now - 1000)], // created ages ago, finished just now
    ['running', job('running')],
  ]);
  pruneJobs(map, now);
  assert.deepEqual([...map.keys()].sort(), ['fresh', 'running']);
});
