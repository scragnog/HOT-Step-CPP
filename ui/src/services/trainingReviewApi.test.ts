import assert from 'node:assert/strict';
import test from 'node:test';
import { finishReviewBatch, pickFromLadder, selectReviewRung, type ReviewLadder } from './trainingReviewApi.js';

const ladder: ReviewLadder = {
  dataset: { id: 'dataset', revision: 'rev-1' }, run: 'run-1', runRevision: 7,
  reviewRevision: 'a'.repeat(64), status: 'done', finished: false, origin: null,
  freezePlannerNow: false, facts: [
    { step: 10, dir: 'ckpt-10', blindLabel: 'B', likeness: 4, corruption: 2,
      overall: 4, replansPerTake: 0, previewCount: 1, planFlags: 0 },
    { step: 20, dir: 'ckpt-20', blindLabel: 'A', likeness: null, corruption: 1,
      overall: null, replansPerTake: 0, previewCount: 1, planFlags: 0 },
  ], best: null,
};

test('review commands capture the selected run, score revision, checkpoint and local worker', async () => {
  const original = globalThis.fetch;
  const sent: Array<{ url: string; body: any }> = [];
  globalThis.fetch = async (input, init) => {
    sent.push({ url: String(input), body: JSON.parse(String(init?.body)) });
    return new Response(JSON.stringify({ batch: { id: 'batch' }, linked: true, plan: {} }), { status: 200 });
  };
  try {
    const pick = pickFromLadder(ladder, 10);
    await finishReviewBatch([pick], true);
    await selectReviewRung(pick);
    assert.equal(sent[0].url, '/api/training/ops/review/finish-batch');
    assert.equal(sent[0].body.worker.kind, 'local');
    assert.deepEqual(sent[0].body.dataset, { id: 'dataset', revision: 'rev-1' });
    assert.deepEqual(sent[0].body.sources, [{ kind: 'yue2-run', id: 'run-1', revision: '7' }]);
    assert.deepEqual(sent[0].body.payload.entries, [pick]);
    assert.equal(sent[1].body.payload.reviewRevision, ladder.reviewRevision);
    assert.equal(sent[1].body.payload.checkpointDir, 'ckpt-10');
    assert.equal('datasetRevision' in sent[1].body.payload, false);
  } finally { globalThis.fetch = original; }
});

test('a disappeared checkpoint cannot be submitted from a stale ladder', () => {
  assert.throws(() => pickFromLadder(ladder, 30), /no longer on this ladder/);
});
