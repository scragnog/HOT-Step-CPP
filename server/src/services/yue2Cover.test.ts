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
import { readSidecar, writeSidecar } from './training/sidecarIO.js';
import { sidecarPathFor } from './training/paths.js';
import { abcSidecarPath, writeAbcSidecar } from './training/abcSidecar.js';

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
  let ranOn = '';
  let duration = 120;
  let datasetAudio: string | null = null;
  let failAbc = false;
  let pending: (() => Promise<void>) | undefined;
  let alignTimes: Array<[string, number]> = [];
  let refinedInstalled = false;
  let stored = { refined: false, keyMode: false };
  let ranWith: unknown;
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
    refinedModel: () => refinedInstalled ? 'sheetsage2-refined-f16.gguf' : '',
    transcriberSetting: () => stored,
    vocalStem: async (audioPath: string) => audioPath,
    align: async (_audio: Buffer, lyrics: string) => ({ words: alignTimes.map(([text, start]) => {
      const char0 = lyrics.indexOf(text);
      return { char0, char1: char0 + text.length, start, end: start + 0.4, score: 0.9 };
    }) }),
    saveDetails: async ({ audioPath }: { audioPath: string }, fields: Record<string, string>) => {
      const file = sidecarPathFor(audioPath);
      await writeSidecar(file, { ...readSidecar(file), ...fields }, { keepBackups: true });
    },
    saveAbc: ({ audioPath }: { audioPath: string }, abc: string) => {
      if (failAbc) throw new Error('disk full');
      return writeAbcSidecar(audioPath, abc, { keepBackups: true, anySounding: true });
    },
    toMp3: async (file: string) => { const out = path.join(root, 'cover-mp3', path.basename(file) + '.mp3'); fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, 'mp3'); return out; },
    datasetMatch: () => datasetAudio ? { datasetId: 'dataset-1', sampleId: 'sample-1', audioPath: datasetAudio } : null,
    song: (id, userId) => id === 'song-1' && userId === 'owner' ? { audio_url: '/audio/library.wav', title: 'Library track' } : undefined,
    run: async (job, audioPath, dir, transcriber) => {
      runnerCalls++;
      ranWith = transcriber;
      ranOn = audioPath;
      job.status = 'running';
      job.phase = 'transcribing';
      await new Promise<void>(resolve => { job.controller.signal.addEventListener('abort', () => resolve(), { once: true }); setTimeout(resolve, 15); });
      if (job.controller.signal.aborted) return '';
      fs.writeFileSync(path.join(dir, 'cover-sheet.json'), JSON.stringify({ sources: [{ name: path.basename(audioPath), abc: 'X:1\nK:C\nC' }] }));
      job.status = 'done';
      return 'X:1\nK:C\nC';
    },
  });
  return { root, referenceDir, libraryDir, upload, service, setDuration: (n: number) => { duration = n; },
    setDatasetAudio: (audio: string | null) => { datasetAudio = audio; },
    failAbcWrites: (fail: boolean) => { failAbc = fail; },
    setAlign: (times: Array<[string, number]>) => { alignTimes = times; },
    setRefined: (installed: boolean) => { refinedInstalled = installed; }, ranWith: () => ranWith,
    setStored: (next: { refined: boolean; keyMode: boolean }) => { stored = next; },
    counts: () => ({ started, cancelled, runnerCalls }), ranOn: () => ranOn, drain: async () => { await pending?.(); },
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
    // A blank flag is read from the lyrics.
    fs.writeFileSync(sidecar, 'bpm: 123\nkey: C# Minor\nis_instrumental:\nlyrics:\n[Intro]\nSung line\n');
    const blank = await f.service.lookup(input, 'owner');
    if (!blank.matched || !blank.metadataAvailable) throw new Error('Expected dataset metadata');
    assert.equal(blank.isInstrumental, false);
    assert.equal(blank.lyrics, '[Intro]\nSung line');
    fs.writeFileSync(sidecar, 'bpm: 123\nkey: C# Minor\nis_instrumental:\nlyrics:');
    const blankEmpty = await f.service.lookup(input, 'owner');
    assert.equal(blankEmpty.matched && blankEmpty.metadataAvailable && blankEmpty.isInstrumental, true);
    fs.writeFileSync(sidecar, 'bpm: 123\nkey: C# Minor\nis_instrumental: maybe\nlyrics: line');
    assert.deepEqual(await f.service.lookup(input, 'owner'), { matched: false });
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

test('SheetSage2-Refined is chosen per transcription and needs its overlay installed', async () => {
  const f = fixture(false, true);
  try {
    const input = { sourceAudioUrl: `/references/${f.upload}`, transcriber: 'sheetsage2-refined', keyMode: true };
    assert.equal(f.service.readiness().refined, false);
    await assert.rejects(f.service.start(input, 'owner'), /yue2-sheetsage2-refined-f16.*Model Manager/);
    await assert.rejects(f.service.start({ ...input, transcriber: 'other' }, 'owner'), /transcriber must be/);
    assert.equal(f.counts().started, 0);
    f.setRefined(true);
    assert.equal(f.service.readiness().refined, true);
    assert.equal((await f.service.start(input, 'owner')).status, 'queued');
    await f.drain();
    assert.deepEqual(f.ranWith(), { refined: true, keyMode: true });
    // key_mode only means something with the refined decoder
    await f.service.start({ sourceAudioUrl: `/references/${f.upload}`, keyMode: true }, 'owner');
    await f.drain();
    assert.deepEqual(f.ranWith(), { refined: false, keyMode: false });
  } finally { f.close(); }
});

test('the Models dropdown choice applies when a transcription names none', async () => {
  const f = fixture(false, true);
  try {
    const input = { sourceAudioUrl: `/references/${f.upload}` };
    f.setStored({ refined: true, keyMode: true });
    await assert.rejects(f.service.start(input, 'owner'), /chosen but not installed/);
    f.setRefined(true);
    await f.service.start(input, 'owner');
    await f.drain();
    assert.deepEqual(f.ranWith(), { refined: true, keyMode: true });
    // an explicit SheetSage2 request still wins
    await f.service.start({ ...input, transcriber: 'sheetsage2' }, 'owner');
    await f.drain();
    assert.deepEqual(f.ranWith(), { refined: false, keyMode: false });
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
    fs.truncateSync(path.join(f.referenceDir, f.upload), 0);
    await assert.rejects(f.service.start({ sourceAudioUrl: `/references/${f.upload}` }, 'owner'), /empty/);
    assert.equal(f.counts().started, 0);
  } finally { f.close(); }
});

test('an oversized source runs on a cached MP3 copy, leaving the original alone', async () => {
  const f = fixture();
  try {
    const original = path.join(f.referenceDir, f.upload);
    fs.truncateSync(original, 100 * 1024 * 1024 + 1);
    const result = await f.service.start({ sourceAudioUrl: `/references/${f.upload}` }, 'owner');
    assert.equal(result.status, 'queued');
    await new Promise(resolve => setTimeout(resolve, 40));
    assert.equal(f.ranOn(), path.join(f.root, 'cover-mp3', f.upload + '.mp3'));
    assert.equal(fs.statSync(original).size, 100 * 1024 * 1024 + 1);
    assert.deepEqual(fs.readdirSync(f.referenceDir), [f.upload]);
    assert.equal(f.service.find(result.jobId!, 'owner').abc, 'X:1\nK:C\nC');
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
    assert.equal(body.metricVersion, 5);
    assert.equal((await fetch(url, { method: 'POST', headers: { Authorization: 'Bearer owner' } })).status, 200);
    assert.equal(alignCalls, 1);
    row.generation_params = JSON.stringify({ ...JSON.parse(row.generation_params),
      yue2CoverDrift: { ...body, metricVersion: 4, meanAbsoluteOffsetBars: 99 } });
    const refreshed = await fetch(url, { method: 'POST', headers: { Authorization: 'Bearer owner' } });
    assert.equal((await refreshed.json() as { meanAbsoluteOffsetBars: number }).meanAbsoluteOffsetBars, 0);
    assert.equal(alignCalls, 2);
    audioModified++;
    assert.equal((await fetch(url, { method: 'POST', headers: { Authorization: 'Bearer owner' } })).status, 200);
    assert.equal(alignCalls, 3);
    assert.equal(f.counts().started, 0);
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); f.close(); }
});

test('match sections retags from aligned times; save writes the dataset sidecar with a .bak', async () => {
  const f = fixture();
  try {
    const input = { sourceAudioUrl: `/references/${f.upload}` };
    const abc = ['X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C', '% verse', 'V: Vocal', 'C6|D6|',
      '% chorus', 'V: Vocal', 'E6|F6|', ''].join('\n');
    const lyrics = '[Verse 1]\nfirst line\n[Hook]\nsecond line';
    f.setAlign([['first', 0.2], ['second', 6.1]]);
    const matched = await f.service.matchSections({ ...input, abc, lyrics }, 'owner');
    assert.equal(matched.lyrics, '[verse]\nfirst line\n\n[chorus]\nsecond line');
    assert.equal(matched.datasetSong, false);
    // The other direction returns the score's new sections; nothing is written.
    const plan = await f.service.planScoreSections({ ...input, abc, lyrics }, 'owner');
    // The lyrics win, so the score takes the lyric's own kind ([Hook] → hook).
    assert.deepEqual(plan.sections.map(s => `${s.label}@${s.bar}`), ['verse@1', 'hook@3']);
    assert.equal(plan.lyrics, matched.lyrics.replace('[chorus]', '[hook]'));
    await assert.rejects(f.service.saveDatasetDetails({ ...input, lyrics: matched.lyrics }, 'owner'), /not a dataset song/);
    f.setDatasetAudio(path.join(f.referenceDir, f.upload));
    const sidecar = path.join(f.referenceDir, path.parse(f.upload).name + '.txt');
    fs.writeFileSync(sidecar, `caption: test\nbpm: 120\nkey: E Major\nlyrics:\n${lyrics}\n`);
    const onDataset = await f.service.matchSections({ ...input, abc, lyrics }, 'owner');
    assert.equal(onDataset.datasetSong, true);
    await assert.rejects(f.service.saveDatasetDetails({ ...input, lyrics: matched.lyrics }, 'owner'), /changed since matching/);
    const saved = await f.service.saveDatasetDetails({ ...input, lyrics: matched.lyrics, revision: onDataset.revision }, 'owner');
    assert.equal(saved.saved, true);
    assert.equal(readSidecar(sidecar).lyrics, matched.lyrics);
    assert.equal(readSidecar(sidecar).caption, 'test');
    assert.equal(readSidecar(`${sidecar}.bak`).lyrics, lyrics);

    // The score's tempo and key replace the sidecar's; lyrics stay.
    await assert.rejects(f.service.saveDatasetDetails({ ...input, bpm: 125, key: 'Bm' }, 'owner'), /key like/);
    await assert.rejects(f.service.saveDatasetDetails({ ...input, bpm: 125 }, 'owner'), /key like/);
    await assert.rejects(f.service.saveDatasetDetails(input, 'owner'), /Nothing to save/);
    await f.service.saveDatasetDetails({ ...input, bpm: 125.4, key: 'B Minor' }, 'owner');
    assert.equal(readSidecar(sidecar).bpm, '125');
    assert.equal(readSidecar(sidecar).key, 'B Minor');
    assert.equal(readSidecar(sidecar).lyrics, matched.lyrics);
  } finally { f.close(); }
});


test('saving with keepBackups never replaces an existing .bak', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sidecar-bak-'));
  try {
    const file = path.join(dir, 'song.txt');
    fs.writeFileSync(file, 'caption: now\nlyrics:\ncurrent words\n');
    fs.writeFileSync(`${file}.bak`, 'caption: old\nlyrics:\nolder words\n');
    await writeSidecar(file, { ...readSidecar(file), lyrics: 'new words' }, { keepBackups: true });
    assert.equal(readSidecar(file).lyrics, 'new words');
    assert.equal(readSidecar(`${file}.bak`).lyrics, 'older words');
    const stamped = fs.readdirSync(dir).filter(name => /^song\.txt\.\d{8}-\d{6}\.bak$/.test(name));
    assert.equal(stamped.length, 1);
    assert.equal(readSidecar(path.join(dir, stamped[0])).lyrics, 'current words');
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('an accepted score plan saves lyrics and the revised score with backups, and a reload reuses them', async () => {
  const f = fixture();
  try {
    const input = { sourceAudioUrl: `/references/${f.upload}` };
    const abc = ['X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C', '% verse', 'V: Vocal', 'C6|D6|',
      '% chorus', 'V: Vocal', 'E6|F6|', ''].join('\n');
    const revised = abc.replace('% chorus', '% hook');
    const lyrics = '[Verse 1]\nfirst line\n[Hook]\nsecond line';
    f.setAlign([['first', 0.2], ['second', 6.1]]);
    f.setDatasetAudio(path.join(f.referenceDir, f.upload));
    const sidecar = path.join(f.referenceDir, path.parse(f.upload).name + '.txt');
    const abcFile = abcSidecarPath(path.join(f.referenceDir, f.upload));
    fs.writeFileSync(sidecar, `caption: test\nbpm: 120\nkey: E Major\nis_instrumental: false\nlyrics:\n${lyrics}\n`);
    fs.writeFileSync(abcFile, `${abc}\n`);
    const plan = await f.service.planScoreSections({ ...input, abc, lyrics }, 'owner');
    assert.equal(plan.datasetSong, true);

    // Nothing is written by planning, and a score with no vocal is refused before any write.
    assert.equal(readSidecar(sidecar).lyrics, lyrics);
    await assert.rejects(f.service.saveDatasetDetails({ ...input, lyrics: plan.lyrics, abc: 'X:1\nK:C\n', revision: plan.revision }, 'owner'), /sounding vocal/);
    assert.equal(readSidecar(sidecar).lyrics, lyrics);
    // A song edited since matching is refused whole.
    fs.appendFileSync(sidecar, 'extra\n');
    await assert.rejects(f.service.saveDatasetDetails({ ...input, lyrics: plan.lyrics, abc: revised, revision: plan.revision }, 'owner'), /changed since matching/);
    const fresh = await f.service.planScoreSections({ ...input, abc, lyrics }, 'owner');
    assert.notEqual(fresh.revision, plan.revision);

    // An .abc failure after the .txt is written is reported, not hidden.
    f.failAbcWrites(true);
    const partial = await f.service.saveDatasetDetails({ ...input, lyrics: fresh.lyrics, abc: revised, revision: fresh.revision }, 'owner') as { saved: boolean; abcSaved?: boolean; abcError?: string };
    assert.equal(partial.saved, true);
    assert.equal(partial.abcSaved, false);
    assert.match(partial.abcError || '', /disk full/);
    assert.equal(readSidecar(sidecar).lyrics, fresh.lyrics);
    assert.equal(fs.readFileSync(abcFile, 'utf8').trim(), abc.trim());
    f.failAbcWrites(false);

    const again = await f.service.planScoreSections({ ...input, abc, lyrics }, 'owner');
    const done = await f.service.saveDatasetDetails({ ...input, lyrics: again.lyrics, abc: revised, revision: again.revision }, 'owner') as { abcSaved?: boolean };
    assert.equal(done.abcSaved, true);
    assert.equal(fs.readFileSync(abcFile, 'utf8').trim(), revised.trim());
    assert.equal(readSidecar(sidecar).caption, 'test');
    assert.equal(fs.readFileSync(`${abcFile}.bak`, 'utf8').trim(), abc.trim());
    assert.equal(readSidecar(`${sidecar}.bak`).lyrics, lyrics + '\nextra');

    // Reload: the dataset supplies the saved score and lyrics, with no job.
    const looked = await f.service.lookup(input, 'owner');
    assert.equal(looked.matched && looked.metadataAvailable && looked.lyrics, again.lyrics);
    const started = await f.service.start(input, 'owner');
    assert.equal(started.status, 'done');
    assert.equal(started.abc, revised.trim());
    assert.equal(f.counts().started, 0);

    // An instrumental score (silent Vocal, sounding Ins) saves and reloads.
    const instrumental = revised.replace(/^C6\|D6\|$/m, 'Z2|').replace(/^E6\|F6\|$/m, 'Z2|') + ['V: Ins', 'C6|D6|', ''].join(String.fromCharCode(10));
    const insPlan = await f.service.planScoreSections({ ...input, abc: instrumental, lyrics }, 'owner');
    const saved = await f.service.saveDatasetDetails({ ...input, lyrics: insPlan.lyrics, abc: instrumental, revision: insPlan.revision }, 'owner') as { abcSaved?: boolean };
    assert.equal(saved.abcSaved, true);
    assert.equal(fs.readFileSync(abcFile, 'utf8').trim(), instrumental.trim());
  } finally { f.close(); }
});

test('suggestions stamp what they were made for, stay instant unless asked, and fall back when the recording fails', async () => {
  const f = fixture();
  try {
    const abc = ['X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C', '% verse', 'V: Vocal', 'C6|D6|',
      '% chorus', 'V: Vocal', 'E6|F6|', '% chorus', 'V: Vocal', 'E6|F6|', ''].join('\n');
    const prepare = {
      version: 2, sourceKey: 'src-1', abc, sectionIds: ['s1', 's2', 's3'], paste: '', excluded: [], flagged: [], nextId: 4,
      lyricRevision: 7, assignments: {}, confirmed: { structure: null, lyrics: null },
      blocks: [{ id: 'b1', text: 'first line', tag: 'Verse' }, { id: 'b2', text: 'second line', tag: 'Chorus' }],
    };
    const input = { sourceAudioUrl: `/references/${f.upload}`, prepare };
    const instant = await f.service.suggestAssignments(input, 'owner');
    assert.deepEqual([instant.sourceKey, instant.lyricRevision, instant.structureHash.startsWith('v1:')], ['src-1', 7, true]);
    assert.deepEqual(instant.assignments, { s1: ['b1'], s2: ['b2'] });
    assert.deepEqual(instant.reuse, [{ sectionId: 's3', blockId: 'b2' }]);
    assert.equal(instant.recording, false);
    // Opt in, and the recording places the blocks.
    f.setAlign([['first', 0.2], ['second', 6.1]]);
    const aligned = await f.service.suggestAssignments({ ...input, useRecording: true }, 'owner');
    assert.equal(aligned.recording, true);
    assert.deepEqual(aligned.assignments, { s1: ['b1'], s2: ['b2'] });
    // A source that cannot be read falls back to the instant proposals, with the reason.
    const fallback = await f.service.suggestAssignments({ ...input, sourceAudioUrl: '/references/missing.wav', useRecording: true }, 'owner');
    assert.equal(fallback.recording, false);
    assert.deepEqual(fallback.assignments, instant.assignments);
    assert.equal(typeof fallback.recordingError, 'string');
    // A malformed document is a 400; replacement score or lyrics in the body are ignored.
    await assert.rejects(f.service.suggestAssignments({ ...input, prepare: { ...prepare, version: 1 } }, 'owner'), /version/);
    const sneaky = await f.service.suggestAssignments({ ...input, abc: 'X:1\nK:C\nC', lyrics: 'nope' } as never, 'owner');
    assert.equal(sneaky.structureHash, instant.structureHash);
  } finally { f.close(); }
});
