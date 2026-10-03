// trainingApi.test.ts — the worker/local base split, as pure assertions.
// No UI test runner is wired up for this project; run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/services/trainingApi.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  finishYue2Ladders, getJob, listYue2AitkRuns, listYue2BatchesLocal, listYue2JointPreviews,
  listYue2JointPreviewsLocal, pickLadderRun, scoreYue2Rung, setTrainingWorker, yue2JointPreviewsReader,
} from './trainingApi.js';

function mockFetch(urls: string[]) {
  const real = globalThis.fetch;
  globalThis.fetch = (async (url: string) => {
    urls.push(String(url));
    return new Response(JSON.stringify({}), { status: 200, headers: { 'content-type': 'application/json' } });
  }) as typeof fetch;
  return () => { globalThis.fetch = real; };
}

test('with a worker selected, run records/scores/finish/local previews hit the local base; job control hits the worker base', async () => {
  const urls: string[] = [];
  const restore = mockFetch(urls);
  try {
    setTrainingWorker('LivingRoom');
    await listYue2AitkRuns('ds1');
    await scoreYue2Rung('ds1', { refineRun: 'run1', step: 10 });
    await finishYue2Ladders([{ datasetId: 'ds1', refineRun: 'run1' }]);
    await listYue2BatchesLocal();
    await listYue2JointPreviewsLocal('ds1', 'run1');
    const localCalls = urls.length;
    assert.ok(urls.every(u => u.startsWith('/api/training/') && !u.includes('/api/workers/')), `expected only local-base calls, got ${JSON.stringify(urls)}`);

    await getJob('job1');
    await listYue2JointPreviews('ds1', 'run1');
    assert.equal(urls.length, localCalls + 2);
    for (const u of urls.slice(localCalls)) assert.match(u, /^\/api\/workers\/LivingRoom\/api\/training\//);
  } finally { restore(); setTrainingWorker(null); }
});

test('yue2JointPreviewsReader: local once a run is indexed, worker reader only while unindexed', () => {
  assert.equal(yue2JointPreviewsReader(true), listYue2JointPreviewsLocal);
  assert.equal(yue2JointPreviewsReader(false), listYue2JointPreviews);
});

test('pickLadderRun: an explicit pick always wins', () => {
  const runs = [{ jobId: 'old', live: false, createdAt: 1 }, { jobId: 'new', live: true, createdAt: 2 }];
  assert.equal(pickLadderRun(runs, 'old', 'new')?.jobId, 'old');
});

test('pickLadderRun: an unindexed active job shows live-only instead of falling back to older history', () => {
  const runs = [{ jobId: 'old', live: false, createdAt: 1 }];
  assert.equal(pickLadderRun(runs, undefined, 'unindexed-worker-job'), undefined);
});

test('pickLadderRun: an indexed active job, or else the live run, or else the newest, when nothing is picked', () => {
  const runs = [{ jobId: 'old', live: false, createdAt: 1 }, { jobId: 'active', live: false, createdAt: 2 }];
  assert.equal(pickLadderRun(runs, undefined, 'active')?.jobId, 'active');
  // No active job and no pick: falls back to the newest run with history.
  assert.equal(pickLadderRun(runs, undefined, undefined)?.jobId, 'active');
  const withLive = [{ jobId: 'old', live: false, createdAt: 1 }, { jobId: 'live', live: true, createdAt: 2 }];
  assert.equal(pickLadderRun(withLive, undefined, undefined)?.jobId, 'live');
});
