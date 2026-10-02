import { test } from 'node:test';
import assert from 'node:assert/strict';
import { jobCover } from './generate.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

// #S3: a cover job's source identity and approved ABC are captured into the
// envelope at submit (index.ts's resolveRequest) and read ONLY from there —
// never from job.params, the raw request body. These tests build a job whose
// envelope and params deliberately disagree, so a regression that reads the
// old params path instead shows up immediately.

function fakeJob(envelopeOptions: Record<string, unknown> | undefined, params: Record<string, unknown>): GenerationJob {
  return {
    envelope: { backendId: 'yue2', options: envelopeOptions ? { yue2: envelopeOptions } : {} },
    params,
  } as unknown as GenerationJob;
}

test('jobCover reads the ABC from the envelope, not from job.params', () => {
  const job = fakeJob(
    { yue2Cover: { sourceId: 'song-1' }, yue2Abc: 'X:1\nK:C\nENVELOPE\n' },
    { yue2Abc: 'X:1\nK:C\nSTALE PARAMS COPY\n' },
  );
  const captured = jobCover(job);
  assert.equal(captured?.abc, 'X:1\nK:C\nENVELOPE\n');
});

test('jobCover reads source identity from the envelope, not from job.params', () => {
  const job = fakeJob(
    { yue2Cover: { sourceId: 'envelope-id', sourceLabel: 'Envelope Label' }, yue2Abc: 'X:1\n' },
    { yue2Cover: { sourceId: 'stale-params-id', sourceLabel: 'Stale Params Label' }, yue2Abc: 'X:1\n' },
  );
  const captured = jobCover(job);
  assert.deepEqual(captured?.cover, { sourceId: 'envelope-id', sourceLabel: 'Envelope Label' });
});

test('an ordinary (non-cover) job has no captured cover, regardless of job.params', () => {
  const job = fakeJob({ yue2Abc: 'X:1\n' }, { yue2Cover: { sourceId: 'ignored' } });
  assert.equal(jobCover(job), undefined);
});

test('a job with no envelope options for this backend has no captured cover', () => {
  const job = fakeJob(undefined, {});
  assert.equal(jobCover(job), undefined);
});
