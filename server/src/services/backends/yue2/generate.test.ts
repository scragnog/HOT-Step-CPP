import { test } from 'node:test';
import assert from 'node:assert/strict';
import { jobCover, mapYue2Params } from './generate.js';
import type { GenerationJob } from '../../generation/jobTypes.js';
import type { Yue2PersistedSelection } from './index.js';

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

test('jobCover reads full score, chord choice and rendered ABC from the envelope only', () => {
  const job = fakeJob(
    { yue2Cover: { sourceId: 'source', keepChords: false, fullScore: 'X:1\n"Am"C|' }, yue2Abc: 'X:1\nC|' },
    { yue2Cover: { sourceId: 'stale', keepChords: true, fullScore: 'wrong' }, yue2Abc: 'wrong' },
  );
  assert.deepEqual(jobCover(job), {
    cover: { sourceId: 'source', keepChords: false, fullScore: 'X:1\n"Am"C|' },
    abc: 'X:1\nC|',
  });
});

test('jobCover reads all five score choices from the captured envelope', () => {
  const cover = { sourceId: 'source', voices: 'vocal', keepChords: false,
    tempo: 'free', key: 'F#m', cfgScale: 1.25, fullScore: 'X:1\nK:Em\n"Em"E|' };
  const job = fakeJob(
    { yue2Cover: cover, yue2Abc: 'X:1\nK:F#m\nF|' },
    { yue2Cover: { ...cover, key: 'Bm', cfgScale: 2 }, yue2Abc: 'wrong' },
  );
  assert.deepEqual(jobCover(job), { cover, abc: 'X:1\nK:F#m\nF|' });
});

test('an ordinary (non-cover) job has no captured cover, regardless of job.params', () => {
  const job = fakeJob({ yue2Abc: 'X:1\n' }, { yue2Cover: { sourceId: 'ignored' } });
  assert.equal(jobCover(job), undefined);
});

test('a job with no envelope options for this backend has no captured cover', () => {
  const job = fakeJob(undefined, {});
  assert.equal(jobCover(job), undefined);
});

// Lead approved 2026-10-03: soft bias on by default for a render with a
// supplied score and lyrics (RESEARCH/YUE2_ALIGNMENT_GATE_SET.md:1064). Hard
// mask stays selectable but never the default, an explicit off always wins,
// and a schedule that cannot be built must never fail the render.

const scales = { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 };
const pick: Yue2PersistedSelection = {
  lm: '', vae_variant: '',
  adapters: { ar: { path: '', scales }, nar: { path: '', scales } },
};
const abc = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C', '% verse', 'V: Vocal', 'C4|', ''].join('\n');
const lyrics = '[Verse]\nfirst line';

test('yue2LyricSchedule defaults to soft bias when a score and lyrics are supplied', () => {
  const { req, notes } = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: abc }, pick);
  assert.equal(req.lyric_schedule?.mode, 'bias');
  assert.equal(req.lyric_schedule?.bias, -4);
  assert.ok(notes.some(n => n.startsWith('Lyric schedule: bias')));
});

test('an explicit "off" is honoured even with a usable score', () => {
  const { req, notes } = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: abc, yue2LyricSchedule: 'off' }, pick);
  assert.equal(req.lyric_schedule, undefined);
  assert.ok(!notes.some(n => n.startsWith('Lyric schedule')));
});

test('hard mask stays selectable but is never the default', () => {
  const { req } = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: abc, yue2LyricSchedule: 'mask' }, pick);
  assert.equal(req.lyric_schedule?.mode, 'mask');
});

test('a schedule that cannot be built falls back instead of failing the render', () => {
  // No "% section" label line: buildYue2LyricSchedule throws "needs a score
  // with labelled sections" — the render must still go ahead without C6.
  const unlabelled = ['X:1', 'M:4/4', 'L:1/4', 'Q:1/4=60', 'K:C', 'C4|', ''].join('\n');
  const { req, notes } = mapYue2Params({ caption: 'folk', lyrics, yue2Abc: unlabelled }, pick);
  assert.equal(req.lyric_schedule, undefined);
  assert.ok(notes.some(n => n.includes('not built') && n.includes('rendering without it')));
});

test('no score or lyrics: the default never fires and never fails', () => {
  const { req } = mapYue2Params({ caption: 'folk' }, pick);
  assert.equal(req.lyric_schedule, undefined);
});
