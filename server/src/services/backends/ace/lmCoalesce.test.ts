import test from 'node:test';
import assert from 'node:assert/strict';
import { aceLmKey, claimLmFollowers, runSharedLmPass } from './lmCoalesce.js';
import type { AceRequest } from '../../aceClient.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

const job = (id: string, params: Record<string, unknown> = {}, status = 'pending') =>
  ({ id, status, params: { caption: id, ...params } }) as unknown as GenerationJob;
const reqOf = (j: GenerationJob): AceRequest =>
  ({ caption: j.params.caption, lm_model: j.params.lmModel, lm_temperature: j.params.temp,
     audio_codes: j.params.codes, task_type: j.params.task }) as AceRequest;
const opts = (pending: GenerationJob[], cap = 4, cachedIds: string[] = []) => ({
  pending, cap, translate: reqOf, prepare: reqOf,
  cached: (j: GenerationJob) => cachedIds.includes(j.id),
});
const plan = (tag: string) => ({ caption: tag, audio_codes: `codes-${tag}` }) as AceRequest;

test('the key holds the LM settings and the co-resident flag, not the prompt', () => {
  const a = { caption: 'x', lm_model: 'm', lm_cfg_scale: 2 } as AceRequest;
  assert.equal(aceLmKey(a, false), aceLmKey({ ...a, caption: 'y', lyrics: 'z', seed: 9 } as AceRequest, false));
  assert.notEqual(aceLmKey(a, false), aceLmKey({ ...a, lm_cfg_scale: 3 }, false));
  assert.notEqual(aceLmKey(a, false), aceLmKey(a, true));
});

test('claims compatible waiting jobs up to the cap, oldest first', () => {
  const lead = job('lead', { batchSize: 2 });
  const f = [job('a'), job('b'), job('c')];
  const got = claimLmFollowers(lead, reqOf(lead), opts(f));
  assert.deepEqual(got.map(x => x.job.id), ['a', 'b']);
  assert.ok(f[0].aceLmShare && f[1].aceLmShare && !f[2].aceLmShare);
  assert.deepEqual(lead.aceLmBatch, ['lead', 'a', 'b']);
  assert.deepEqual(f[1].aceLmBatch, ['lead', 'a', 'b']);
});

test('skips jobs that cannot share the pass', () => {
  const lead = job('lead', { lmModel: 'm' });
  const pending = [
    job('other-lm', { lmModel: 'n' }),
    job('lm-off', { lmModel: 'm', skipLm: true }),
    job('has-codes', { lmModel: 'm', codes: '1,2' }),
    job('cover', { lmModel: 'm', task: 'cover' }),
    job('cancelled', { lmModel: 'm' }, 'cancelled'),
    job('big', { lmModel: 'm', batchSize: 4 }),
    job('cached', { lmModel: 'm' }),
    job('ok', { lmModel: 'm' }),
  ];
  const got = claimLmFollowers(lead, reqOf(lead), opts(pending, 4, ['cached']));
  assert.deepEqual(got.map(x => x.job.id), ['ok']);
});

test('a job planned by another lead is not claimed twice, and a full lead claims none', () => {
  const lead = job('lead');
  const taken = job('taken');
  claimLmFollowers(job('first'), reqOf(lead), opts([taken]));
  assert.equal(claimLmFollowers(lead, reqOf(lead), opts([taken])).length, 0);
  assert.equal(claimLmFollowers(job('full', { batchSize: 4 }), reqOf(lead), opts([job('x')])).length, 0);
  assert.equal(lead.aceLmBatch, undefined);
});

test('one pass hands each job its own plans, in order', async () => {
  const lead = job('lead', { batchSize: 2 });
  const [a, b] = [job('a'), job('b', { batchSize: 2 })];
  const followers = claimLmFollowers(lead, reqOf(lead), opts([a, b], 5));
  let sent: AceRequest[] = [];
  const mine = await runSharedLmPass(reqOf(lead), 2, followers, async body => {
    sent = body;
    return ['l0', 'l1', 'a0', 'b0', 'b1'].map(plan);
  });
  assert.deepEqual(sent.map(r => [r.caption, r.lm_batch_size]), [['lead', 2], ['a', 1], ['b', 2]]);
  assert.deepEqual(mine.map(r => r.caption), ['l0', 'l1']);
  assert.deepEqual((await a.aceLmShare!.plans)!.map(r => r.caption), ['a0']);
  assert.deepEqual((await b.aceLmShare!.plans)!.map(r => r.caption), ['b0', 'b1']);
});

test('a failed or cancelled pass hands followers null, so they plan alone', async () => {
  for (const failure of [new Error('engine 400'), new Error('Cancelled')]) {
    const lead = job('lead');
    const a = job('a');
    const followers = claimLmFollowers(lead, reqOf(lead), opts([a]));
    await assert.rejects(runSharedLmPass(reqOf(lead), 1, followers, async () => { throw failure; }), failure);
    assert.equal(await a.aceLmShare!.plans, null);
  }
});

test('a short or long answer is a failure, not a misaligned hand-out', async () => {
  const lead = job('lead');
  const a = job('a');
  const followers = claimLmFollowers(lead, reqOf(lead), opts([a]));
  await assert.rejects(runSharedLmPass(reqOf(lead), 1, followers, async () => [plan('only one')]), /1 plans for 2 songs/);
  assert.equal(await a.aceLmShare!.plans, null);
});
