import assert from 'node:assert/strict';
import test from 'node:test';
import { admitYue2JointTrain, jointAdmissionError, takeJointAdmission, type JointAdmissionResult } from './yue2JointAdmission.js';

// The route's side as routes/training.ts runs it: take the token, do the
// route's own (async) work, then the guard synchronously before enqueue.
function fakeRoute(state: { cleaned: boolean; jobs: string[] }, between: () => void = () => {}) {
  return async (_url: string, body: Record<string, unknown>): Promise<JointAdmissionResult> => {
    const guard = takeJointAdmission(body.admission);
    if (!guard) return { status: 409, body: { error: 'not admitted' } };
    await Promise.resolve();
    between();
    const refused = jointAdmissionError(guard);
    if (refused) return { status: 409, body: { error: refused } };
    state.jobs.push(`job-${state.jobs.length + 1}`);
    return { status: 200, body: { jobId: state.jobs.at(-1) } };
  };
}

test('a cleanup landing between decision and enqueue is refused and queues nothing', async () => {
  const state = { cleaned: false, jobs: [] as string[] };
  const result = await admitYue2JointTrain({ datasetId: 'd', body: {}, idempotencyKey: 'stale-1',
    revalidate: () => { if (state.cleaned) throw Object.assign(new Error('x'), { body: { error: 'The selected checkpoint changed' } }); },
    post: fakeRoute(state, () => { state.cleaned = true; }) });
  assert.equal(result.status, 409);
  assert.equal(result.body.error, 'The selected checkpoint changed');
  assert.deepEqual(state.jobs, []);
});

test('the same idempotency key starts one job; a token is single use and never guessable', async () => {
  const state = { cleaned: false, jobs: [] as string[] };
  let seen: unknown;
  const post = fakeRoute(state);
  const once = (key: string) => admitYue2JointTrain({ datasetId: 'd', body: {}, idempotencyKey: key, revalidate: () => {},
    post: async (url, body) => { seen = body.admission; return post(url, body); } });
  const [a, b] = await Promise.all([once('k'), once('k')]);
  assert.equal(a.body.jobId, 'job-1');
  assert.equal(b.body.jobId, 'job-1');
  assert.deepEqual(state.jobs, ['job-1']);
  assert.equal(takeJointAdmission(seen), undefined);
  assert.equal(takeJointAdmission('made-up'), undefined);
  assert.equal(takeJointAdmission(undefined), undefined);
});

test('a refused attempt is not remembered, so a retry with the same key is checked again', async () => {
  const state = { cleaned: true, jobs: [] as string[] };
  const attempt = () => admitYue2JointTrain({ datasetId: 'd', body: {}, idempotencyKey: 'retry',
    revalidate: () => { if (state.cleaned) throw new Error('gone'); }, post: fakeRoute(state) });
  assert.equal((await attempt()).status, 409);
  state.cleaned = false;
  assert.equal((await attempt()).status, 200);
  assert.deepEqual(state.jobs, ['job-1']);
});
