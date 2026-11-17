import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runYue2Generation, yue2RequestForJob, type Yue2GenerationDeps } from './generate.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

test('a lead cancel must retain the completed survivor score through the generation caller', async t => {
  const makeJob = (id: string, status: string) => ({ id, userId: 'review-fixture', status,
    params: { caption: 'folk', lyrics: 'la', yue2ReplanAttempts: 1 },
    envelope: { backendId: 'yue2', models: {}, options: {} },
  } as GenerationJob);
  const lead = makeJob('review-lead', 'running');
  const survivor = makeJob('review-survivor', 'pending');
  const ac = new AbortController();
  const score = ['X:1', 'T:', 'M:4/4', 'L:1/32', 'Q:1/4=90',
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
    'V: Ins clef=treble name="Ins Melody" snm="Inst."', 'K:C', '% verse',
    'V: Vocal', '"C"c16"G"d16|', 'V: Ins', 'Z|', ''].join('\n');
  let polls = 0, plans = 0;
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.pathname === '/yue2/props') return Response.json({ max_lm_batch: 2, max_plan_batch: 2 });
    if (url.pathname === '/yue2/synth') {
      const req = JSON.parse(String(init?.body));
      assert.equal(req.plan_only, true);
      plans++;
      return Response.json({ id: 'review-plan' });
    }
    assert.equal(url.pathname, '/job');
    if (++polls === 1) {
      lead.status = 'cancelled'; ac.abort();
      return Response.json({ status: 'running', phase: 'plan', phase_step: 1 });
    }
    return Response.json({ status: 'done', tracks: [0, 1].map(song => ({ song, seed: 1, abc: score, end_reason: 'completed' })) });
  });
  await runYue2Generation(lead, {
    signal: ac.signal, pendingJobs: () => [survivor],
    pollUntilDone: async () => { throw new Error('unexpected render'); },
  } as unknown as Yue2GenerationDeps);
  assert.equal(plans, 1);
  assert.equal(survivor.status, 'pending', 'caller has requeued the surviving song');
  assert.ok(yue2RequestForJob(survivor).abc, 'requeued survivor must retain its already completed score');
});
