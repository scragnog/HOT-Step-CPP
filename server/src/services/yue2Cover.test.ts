import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import test from 'node:test';
import express from 'express';
import type { Server } from 'node:http';
import * as realQueue from './training/labelingQueue.js';
import type { TrainingJob } from './training/labelingQueue.js';
import { createYue2CoverService } from './yue2Cover.js';
import { createYue2CoverRouter } from '../routes/yue2Cover.js';

function fixture(missing = false, defer = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-cover-api-'));
  const referenceDir = path.join(root, 'references');
  const libraryDir = path.join(root, 'audio');
  const jobRoot = path.join(root, 'jobs');
  fs.mkdirSync(referenceDir);
  fs.mkdirSync(libraryDir);
  const upload = `${randomUUID()}.wav`;
  fs.writeFileSync(path.join(referenceDir, upload), 'audio');
  const jobs = new Map<string, TrainingJob>();
  let started = 0;
  let cancelled = 0;
  let runnerCalls = 0;
  let duration = 120;
  let datasetAudio: string | null = null;
  let pending: (() => Promise<void>) | undefined;
  const fakeQueue = {
    createJob: (_kind: string, datasetId: string) => {
      started++;
      const job = { id: randomUUID(), datasetId, kind: 'yue2-sheet', status: 'queued', phase: 'queued', controller: new AbortController() } as TrainingJob;
      jobs.set(job.id, job);
      return job;
    },
    enqueue: (job: TrainingJob, run: (job: TrainingJob) => Promise<void>) => {
      if (defer) pending = () => run(job);
      else void run(job);
    },
    getJob: (id: string) => jobs.get(id),
    listJobs: () => [...jobs.values()].map(j => ({ id: j.id, status: j.status, phase: j.phase })),
    toSummary: (job: TrainingJob) => ({ id: job.id, status: job.status, phase: job.phase }),
    cancelJob: (id: string) => {
      const job = jobs.get(id);
      if (!job) return false;
      cancelled++;
      job.status = 'cancelled';
      job.controller.abort();
      return true;
    },
  } as unknown as typeof realQueue;
  const service = createYue2CoverService({
    queue: fakeQueue, referenceDir, libraryDir, jobRoot,
    duration: async () => duration,
    missingModels: () => missing ? ['SheetSage2'] : [],
    model: () => missing ? '' : 'sheetsage2-f16.gguf',
    datasetMatch: () => datasetAudio ? { datasetId: 'dataset-1', sampleId: 'sample-1', audioPath: datasetAudio } : null,
    song: (id, userId) => id === 'song-1' && userId === 'owner' ? { audio_url: '/audio/library.wav', title: 'Library track' } : undefined,
    run: async (job, _audioPath, dir) => {
      runnerCalls++;
      job.status = 'running';
      job.phase = 'transcribing';
      await new Promise<void>(resolve => { job.controller.signal.addEventListener('abort', () => resolve(), { once: true }); setTimeout(resolve, 15); });
      if (job.controller.signal.aborted) return '';
      fs.writeFileSync(path.join(dir, 'cover-sheet.json'), JSON.stringify({ sources: [{ name: upload, abc: 'X:1\nK:C\nC' }] }));
      job.status = 'done';
      return 'X:1\nK:C\nC';
    },
  });
  return { root, referenceDir, libraryDir, upload, service, setDuration: (n: number) => { duration = n; },
    setDatasetAudio: (audio: string | null) => { datasetAudio = audio; },
    counts: () => ({ started, cancelled, runnerCalls }), drain: async () => { await pending?.(); },
    close: () => fs.rmSync(root, { recursive: true, force: true }) };
}

test('dataset sidecar supplies lyrics, tempo, key and a cached score without a job', async () => {
  const f = fixture(true);
  try {
    const audio = path.join(f.referenceDir, f.upload);
    f.setDatasetAudio(audio);
    fs.writeFileSync(path.join(f.referenceDir, path.parse(f.upload).name + '.txt'),
      'caption: test\nbpm: 132\nkey: D minor\nis_instrumental: false\nlyrics:\nVerse line\n');
    fs.writeFileSync(path.join(f.referenceDir, path.parse(f.upload).name + '.abc'), 'X:1\nK:Dm\nD\n');
    const input = { sourceAudioUrl: `/references/${f.upload}` };
    const metadata = await f.service.lookup(input, 'owner');
    assert.deepEqual(metadata, { matched: true, datasetId: 'dataset-1', sampleId: 'sample-1',
      metadataAvailable: true, lyrics: 'Verse line', bpm: 132, key: 'D minor', isInstrumental: false,
      abc: 'X:1\nK:Dm\nD', sections: [], lyricsSource: 'dataset-sidecar' });
    const result = await f.service.start(input, 'owner');
    assert.equal(result.status, 'done');
    assert.equal(result.abc, 'X:1\nK:Dm\nD');
    assert.deepEqual(result.sections, []);
    assert.equal(result.scoreSource, 'dataset');
    assert.equal(f.counts().started, 0);
  } finally { f.close(); }
});

test('instrumental metadata and missing or malformed sidecars preserve fallback', async () => {
  const f = fixture();
  try {
    const input = { sourceAudioUrl: `/references/${f.upload}` };
    assert.deepEqual(await f.service.lookup(input, 'owner'), { matched: false });
    const audio = path.join(f.referenceDir, f.upload);
    f.setDatasetAudio(audio);
    assert.deepEqual(await f.service.lookup(input, 'owner'), { matched: false });
    const sidecar = path.join(f.referenceDir, path.parse(f.upload).name + '.txt');
    fs.writeFileSync(sidecar, 'bpm: nonsense\nkey: C major\nlyrics: line');
    assert.deepEqual(await f.service.lookup(input, 'owner'), { matched: false });
    fs.writeFileSync(sidecar, 'bpm: 88\nkey: C major\nis_instrumental: true\nlyrics:');
    const instrumental = await f.service.lookup(input, 'owner');
    assert.equal(instrumental.matched, true);
    if (!instrumental.matched || !instrumental.metadataAvailable) throw new Error('Expected dataset metadata');
    assert.equal(instrumental.isInstrumental, true);
    assert.equal(instrumental.lyrics, '');
  } finally { f.close(); }
});

test('force transcribes again even when the dataset has a cached score', async () => {
  const f = fixture();
  try {
    const audio = path.join(f.referenceDir, f.upload);
    f.setDatasetAudio(audio);
    fs.writeFileSync(path.join(f.referenceDir, path.parse(f.upload).name + '.abc'), 'X:1\nK:C\nC');
    const result = await f.service.start({ sourceAudioUrl: `/references/${f.upload}`, force: true }, 'owner');
    assert.equal(result.status, 'queued');
    assert.equal(f.counts().started, 1);
  } finally { f.close(); }
});

test('missing SheetSage gives Model Manager guidance and creates no job', async () => {
  const f = fixture(true);
  try {
    assert.match(f.service.readiness().message!, /yue2-sheetsage2-f16.*Model Manager/);
    await assert.rejects(f.service.start({ sourceAudioUrl: `/references/${f.upload}` }, 'owner'), /Download.*Model Manager/);
    assert.equal(f.counts().started, 0);
  } finally { f.close(); }
});

test('invalid, oversized, and unreadable sources are refused before enqueue', async () => {
  const f = fixture();
  try {
    await assert.rejects(f.service.start({ sourceAudioUrl: '/references/../escape.wav' }, 'owner'), /Invalid audio source/);
    await assert.rejects(f.service.start({ songId: 'song-1' }, 'other-user'), /Library song not found/);
    f.setDuration(601);
    await assert.rejects(f.service.start({ sourceAudioUrl: `/references/${f.upload}` }, 'owner'), /10 minutes/);
    f.setDuration(120);
    fs.truncateSync(path.join(f.referenceDir, f.upload), 100 * 1024 * 1024 + 1);
    await assert.rejects(f.service.start({ sourceAudioUrl: `/references/${f.upload}` }, 'owner'), /100 MB/);
    assert.equal(f.counts().started, 0);
  } finally { f.close(); }
});

test('supplied ABC bypasses the model and runner, retaining the source identity', async () => {
  const f = fixture(true);
  try {
    const result = await f.service.start({ sourceAudioUrl: `/references/${f.upload}`, abc: ' X:1\nK:C\nC ' }, 'owner');
    assert.deepEqual(result, { status: 'done', abc: 'X:1\nK:C\nC', sections: [],
      sourceId: `/references/${f.upload}`, sourceLabel: f.upload });
    assert.deepEqual(f.counts(), { started: 0, cancelled: 0, runnerCalls: 0 });
  } finally { f.close(); }
});

test('queued job reports progress and cancellation uses the existing queue', async () => {
  const f = fixture();
  try {
    fs.writeFileSync(path.join(f.libraryDir, 'library.wav'), 'audio');
    const started = await f.service.start({ songId: 'song-1' }, 'owner');
    assert.equal(started.status, 'queued');
    assert.equal(started.sourceId, 'song-1');
    const progress = f.service.find(started.jobId!, 'owner');
    assert.equal(progress.job.phase, 'transcribing');
    assert.equal(progress.sourceLabel, 'Library track');
    assert.throws(() => f.service.find(started.jobId!, 'someone-else'), /not found/);
    const stopped = f.service.cancel(started.jobId!, 'owner');
    assert.equal(stopped.job.status, 'cancelled');
    assert.equal(f.counts().cancelled, 1);
    assert.equal(f.counts().runnerCalls, 1);
  } finally { f.close(); }
});

test('completed job returns approved ABC and its source identity', async () => {
  const f = fixture();
  try {
    const started = await f.service.start({ sourceAudioUrl: `/references/${f.upload}` }, 'owner');
    for (let attempt = 0; attempt < 50 && f.service.find(started.jobId!, 'owner').job.status !== 'done'; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    const result = f.service.find(started.jobId!, 'owner');
    assert.equal(result.job.status, 'done');
    assert.equal(result.abc, 'X:1\nK:C\nC');
    assert.deepEqual(result.sections, []);
    assert.equal(result.sourceId, `/references/${f.upload}`);
  } finally { f.close(); }
});

test('cancellation before dequeue never starts transcription', async () => {
  const f = fixture(false, true);
  try {
    const started = await f.service.start({ sourceAudioUrl: `/references/${f.upload}` }, 'owner');
    f.service.cancel(started.jobId!, 'owner');
    await f.drain();
    assert.equal(f.counts().runnerCalls, 0);
  } finally { f.close(); }
});

test('unauthenticated cover request is refused before reaching the service', async () => {
  const f = fixture();
  const app = express();
  app.use(express.json());
  app.use('/api/yue2-cover', createYue2CoverRouter(f.service, () => null));
  const server: Server = app.listen(0);
  try {
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('Expected TCP address');
    const response = await fetch(`http://127.0.0.1:${address.port}/api/yue2-cover/transcriptions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sourceAudioUrl: `/references/${f.upload}` }),
    });
    assert.equal(response.status, 401);
    assert.deepEqual(await response.json(), { error: 'Unauthorized' });
    assert.equal(f.counts().started, 0);
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); f.close(); }
});

test('section review uses the score and lyrics without starting transcription', () => {
  const f = fixture(true);
  try {
    const result = f.service.reviewScore({ abc: 'X:1\n% verse\nV: Vocal\nC|', lyrics: '[Chorus]\nA line' });
    assert.deepEqual(result.sections, [{ label: 'verse', startBar: 1 }]);
    assert.match(result.lint.message, /score section 1 is verse, lyric tag 1 is Chorus/);
    assert.equal(result.insertedLyrics, '[verse]\nA line');
    assert.deepEqual(f.counts(), { started: 0, cancelled: 0, runnerCalls: 0 });
    assert.throws(() => f.service.reviewScore({ abc: 5, lyrics: '' }), /ABC and lyrics must be strings/);
  } finally { f.close(); }
});

test('section review route returns sections and rejects invalid input', async () => {
  const f = fixture();
  const app = express();
  app.use(express.json());
  app.use('/api/yue2-cover', createYue2CoverRouter(f.service, () => 'owner'));
  const server: Server = app.listen(0);
  try {
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('Expected TCP address');
    const url = `http://127.0.0.1:${address.port}/api/yue2-cover/sections/review`;
    const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ abc: 'X:1\n% chorus\nV: Vocal\nC|', lyrics: '[Chorus]\nline' }) });
    assert.equal(response.status, 200);
    const result = await response.json() as { sections: Array<{ label: string; startBar: number }>; lint: { ok: boolean } };
    assert.deepEqual(result.sections, [{ label: 'chorus', startBar: 1 }]);
    assert.equal(result.lint.ok, true);
    const invalid = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ abc: 5, lyrics: '' }) });
    assert.equal(invalid.status, 400);
    assert.equal(f.counts().started, 0);
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); f.close(); }
});

test('drift route authenticates, aligns once, and persists the cached result', async () => {
  const f = fixture();
  const abc = 'X:1\nM:4/4\nQ:1/4=120\nK:C\n% verse\nV: Vocal\nC|D|';
  const lyrics = '[Verse]\nsing';
  const row = { id: 'cover-1', audio_url: '/audio/cover.wav', lyrics,
    generation_params: JSON.stringify({ backend: 'yue2', yue2Cover: { fullScore: abc },
      yue2Abc: abc, yue2Request: { lyrics } }) };
  let alignCalls = 0;
  let audioModified = 1;
  const deps = {
    song: (id: string, userId: string) => id === row.id && userId === 'owner' ? row : undefined,
    save: (_id: string, _userId: string, result: object) => {
      row.generation_params = JSON.stringify({ ...JSON.parse(row.generation_params), yue2CoverDrift: result });
    },
    align: async () => { alignCalls++; return { words: [{ start: 0, end: 0.5, char0: 8, char1: 12, score: 1 }] }; },
    readAudio: () => Buffer.from('audio'),
    statAudio: () => ({ size: 5, mtimeMs: audioModified } as fs.Stats),
    audioDir: '/unused',
  };
  const app = express();
  app.use('/api/yue2-cover', createYue2CoverRouter(f.service, req =>
    req.headers.authorization === 'Bearer owner' ? 'owner' : null, deps));
  const server: Server = app.listen(0);
  try {
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('Expected TCP address');
    const url = `http://127.0.0.1:${address.port}/api/yue2-cover/drift/cover-1`;
    assert.equal((await fetch(url, { method: 'POST' })).status, 401);
    const first = await fetch(url, { method: 'POST', headers: { Authorization: 'Bearer owner' } });
    assert.equal(first.status, 200);
    const body = await first.json() as { meanAbsoluteOffsetBars: number; metricVersion: number; inputHash: string };
    assert.equal(body.meanAbsoluteOffsetBars, 0);
    assert.equal(body.metricVersion, 3);
    assert.equal((await fetch(url, { method: 'POST', headers: { Authorization: 'Bearer owner' } })).status, 200);
    assert.equal(alignCalls, 1);
    row.generation_params = JSON.stringify({ ...JSON.parse(row.generation_params),
      yue2CoverDrift: { ...body, metricVersion: 2, meanAbsoluteOffsetBars: 99 } });
    const refreshed = await fetch(url, { method: 'POST', headers: { Authorization: 'Bearer owner' } });
    assert.equal((await refreshed.json() as { meanAbsoluteOffsetBars: number }).meanAbsoluteOffsetBars, 0);
    assert.equal(alignCalls, 2);
    audioModified++;
    assert.equal((await fetch(url, { method: 'POST', headers: { Authorization: 'Bearer owner' } })).status, 200);
    assert.equal(alignCalls, 3);
    assert.equal(f.counts().started, 0);
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); f.close(); }
});
