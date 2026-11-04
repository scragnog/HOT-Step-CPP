import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { WorkflowJobs, type WorkflowDeps } from './workflowJobs.js';
import { WorkflowDocuments } from './revisions.js';
import { coverScoreDetails, createCoverKinds, effectiveCoverRequest, type CoverRenderInput } from './coverWorkflow.js';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';
import { structureHash } from '../../contracts/yue2Prepare.js';
import { coverRenderSchema } from '../../contracts/studioWorkflows.js';

const input: CoverRenderInput = {
  documentId: '11111111-1111-4111-8111-111111111111', revision: 1,
  expectedBackend: 'ace', engineParams: { seed: 42, loraScale: 0.7, style: 'global style' },
  title: 'Song', artistName: 'Artist', targetArtistName: 'Target', lyrics: '[Verse]\nLine', caption: 'Cover style',
  instrumental: false, lyricsSource: null, scoreSource: null, analysis: { bpm: 100, key: 'C minor' },
  settings: { coResident: false, cacheLmCodes: true },
  controls: {
    sa3Transform: 0.3, sa3Adapters: '',
    bpmOverride: null, bpmCorrection: 2, keyOverride: null, tempoScale: 0.5,
    pitchShift: 2, noFsq: false, audioCoverStrength: 0.5, coverNoiseStrength: 0,
    coverNoiseMethod: '', vocalLanguage: 'en', sourceLatentUrl: '', timbreOverridePath: '',
    presetAdapterPath: 'adapter.safetensors', presetReferencePath: '',
    triggerUseFilename: true, triggerPlacement: 'prepend', voices: 'both', keepChords: false,
    tempoMode: 'free', coverBpm: 120, keyShift: 0, cfgScale: 1, lyricSchedule: 'off',
    lmAdapterAr: '', lmAdapterNar: '', pairMode: 'base', yue2Pick: {},
    captionMode: 'custom', captionTracks: [],
  },
};

function setup(opts: { transcribe?: (signal: AbortSignal) => Promise<{ abc: string }>;
  audioStatus?: 'pending' | 'succeeded'; capability?: (backend: string) => Promise<boolean>;
  datasetAnalysis?: () => { bpm: number; key: string; scale: string } | null;
  analyze?: () => Promise<{ bpm: number | null; key: string; scale: string } | null> } = {}) {
  const db = new Database(':memory:');
  const documents = new WorkflowDocuments(db);
  const items = new Map<string, AudioIntentItem>();
  const audio: WorkflowDeps['audio'] = {
    enqueue: ({ idempotencyKey, request, meta }) => {
      const prior = [...items.values()].find(i => i.idempotencyKey === idempotencyKey);
      if (prior) return { item: prior, created: false };
      const item = { id: `audio-${items.size + 1}`, idempotencyKey, request, meta: meta || null,
        status: opts.audioStatus || 'succeeded', result: { audioUrls: ['/audio/test.wav'], songIds: ['song-1'] },
        error: null, engine: 'ace', jobId: 'job-1', attempt: 1, previousJobIds: [],
        waiting: null, cancelRequested: false, createdAt: 1, updatedAt: 1, elapsedSeconds: null } as AudioIntentItem;
      items.set(item.id, item);
      return { item, created: true };
    },
    get: id => items.get(id)!,
    cancel: id => { const item = items.get(id)!; item.status = 'cancelled'; return item; },
  };
  const jobs = new WorkflowJobs({ db, audio, audioPollMs: 1 });
  for (const kind of createCoverKinds({ documents,
    asset: (id, user) => {
      if (user !== 'owner' || id !== '22222222-2222-4222-8222-222222222222') throw new Error('asset owner mismatch');
      return { url: '/references/source.wav', path: '/fake/source.wav', filename: 'source.wav', sha256: 'sha' };
    },
    metadata: async () => ({ artist: '', title: '', album: '', duration: null }),
    datasetAnalysis: opts.datasetAnalysis || (() => null),
    analyze: opts.analyze || (async () => ({ bpm: 100, key: 'C', scale: 'minor' })),
    transcribe: async (_url, _label, _user, signal) => opts.transcribe?.(signal) || { abc: 'X:1\nK:C\nC|' },
    capability: opts.capability || (async backend => backend === 'ace' || backend === 'yue2'),
    caption: async () => ({ text: 'Resolved style' }),
  })) jobs.register(kind);
  const submit = (kind: string, key: string, body: Record<string, unknown>) =>
    jobs.submit({ kind, idempotencyKey: key, input: body }, 'owner');
  const open = async () => {
    const job = submit('cover-open', 'open', { assetId: '22222222-2222-4222-8222-222222222222' }).job;
    await jobs.settled();
    return jobs.get(job.id).result as { documentId: string; revision: number };
  };
  return { db, documents, jobs, items, submit, open };
}

test('dataset sidecar analysis skips Essentia; a non-match still runs it', async () => {
  let calls = 0;
  const analyze = async () => { calls++; return { bpm: 100, key: 'C', scale: 'minor' }; };
  const matched = setup({ datasetAnalysis: () => ({ bpm: 132, key: 'D minor', scale: 'minor' }), analyze });
  const found = await matched.open();
  assert.deepEqual(matched.documents.get(found.documentId, 'owner').data.analysis,
    { bpm: 132, key: 'D minor', scale: 'minor' });
  assert.equal(calls, 0);
  const unmatched = setup({ datasetAnalysis: () => null, analyze });
  const other = await unmatched.open();
  assert.deepEqual(unmatched.documents.get(other.documentId, 'owner').data.analysis,
    { bpm: 100, key: 'C minor', scale: 'minor' });
  assert.equal(calls, 1);
});

test('ACE transforms retain source tempo/key defaults and preset precedence', () => {
  const request = effectiveCoverRequest(input, { assetId: 'asset', sha256: 'sha', analysis: { bpm: 90, key: 'G major' } }, '/references/source.wav');
  assert.equal(request.bpm, 100, 'current UI analysis overrides the persisted detection');
  assert.equal(request.keyScale, 'D minor');
  assert.equal(request.tempoScale, 0.5);
  assert.equal(request.loraPath, 'adapter.safetensors');
  assert.equal(request.loraScale, 0.7, 'manual adapter scale survives the preset');
  assert.equal(request.triggerWord, 'adapter');
  assert.equal(request.sourceAudioUrl, '/references/source.wav');
  assert.equal(request.artistName, 'Target');
  const fallback = effectiveCoverRequest({ ...input, analysis: null,
    controls: { ...input.controls, bpmCorrection: 1, tempoScale: 1, pitchShift: 0,
      presetAdapterPath: '', triggerUseFilename: false } }, { assetId: 'asset', sha256: 'sha' }, '/references/source.wav');
  assert.equal(fallback.bpm, 120);
  assert.equal(fallback.keyScale, 'C major');
  assert.equal(fallback.tempoScale, undefined);
  assert.equal(fallback.pitchShift, undefined);
});

test('YuE2 requires score approval and preserves caption/tempo/key fallbacks', () => {
  const yue = { ...input, expectedBackend: 'yue2' as const,
    controls: { ...input.controls, captionMode: 'track:gone',
      captionTracks: [{ name: 'other', styled: 'Dataset style', bpm: 100 }],
      yue2Pick: { lmAdapterArScale: 0.6 }, keyShift: 2, keepChords: true } };
  assert.throws(() => effectiveCoverRequest(yue, { assetId: 'asset', sha256: 'sha' }, '/references/source.wav'), /approve/);
  const request = effectiveCoverRequest(yue, { assetId: 'asset', sha256: 'sha',
    approvedAbc: 'X:1\nK:C\nC|' }, '/references/source.wav');
  assert.equal(request.caption, 'Dataset style', 'missing track degrades to nearest BPM');
  assert.equal((request.yue2Cover as any).key, 'D');
  assert.equal((request.yue2Cover as any).tempo, 'free');
  assert.equal((request.yue2Pick as any).lmAdapterArScale, 0.6);
  assert.equal(request.yue2Cot, 'full');
  assert.equal(request.sourceAudioUrl, '/references/source.wav');
  assert.equal(request.artistName, 'Target', 'covering band names the saved file');
  assert.equal(request.sourceArtist, 'Artist');
});

test('SA3 dispatch forwards its own controls without score approval or global caption leaks', () => {
  const sa3 = { ...input, expectedBackend: 'sa3' as const,
    engineParams: { seed: 73, randomSeed: false, caption: 'stale', masteringEnabled: true,
      stableStepSteps: 12, yue2Cot: 'full', duration: 25 },
    controls: { ...input.controls, sa3Transform: 0.7, sa3Adapters: 'texture:0.6,space:1' } };
  const request = effectiveCoverRequest(sa3, { assetId: 'asset', sha256: 'sha' }, '/references/source.mp3');
  assert.equal(request.backend, 'sa3');
  assert.equal(request.expectedBackend, 'sa3');
  assert.equal(request.taskType, 'cover');
  assert.equal(request.style, input.caption);
  assert.equal(request.caption, input.caption);
  assert.equal(request.sourceAudioUrl, '/references/source.mp3');
  assert.equal(request.seed, 73);
  assert.equal(request.randomSeed, false);
  assert.equal(request.masteringEnabled, true);
  assert.equal(request.stableStepSteps, 12);
  assert.deepEqual(request.sa3Cover, { transform: 0.7, adapters: 'texture:0.6,space:1' });
  assert.equal(request.duration, 0);
  assert.equal(request.instrumental, true);
  assert.equal(request.lyrics, '');
  assert.equal(request.artistName, 'Target');
  assert.equal(request.sourceArtist, 'Artist');
  assert.equal(request.yue2Cot, undefined);
});

test('cover schema accepts SA3 defaults and rejects invalid transforms and unknown backends', () => {
  const { sa3Transform, sa3Adapters, ...legacyControls } = input.controls;
  const parsed = coverRenderSchema.parse({ ...input, expectedBackend: 'sa3', controls: legacyControls });
  assert.equal(parsed.controls.sa3Transform, 0.3);
  assert.equal(parsed.controls.sa3Adapters, '');
  for (const transform of [0, 1, NaN, Infinity]) {
    assert.equal(coverRenderSchema.safeParse({ ...parsed,
      controls: { ...parsed.controls, sa3Transform: transform } }).success, false);
  }
  assert.equal(coverRenderSchema.safeParse({ ...input, expectedBackend: 'unknown' }).success, false);
  assert.throws(() => effectiveCoverRequest({ ...input, expectedBackend: 'unknown' } as unknown as CoverRenderInput,
    { assetId: 'asset', sha256: 'sha', approvedAbc: 'X:1\nK:C\nC|' }, '/references/source.wav'),
  { status: 400 });
});

test('HeartMuLa is recognized but rejects covers clearly before lyrics, score or capability checks', async () => {
  const body = coverRenderSchema.parse({ ...input, expectedBackend: 'heartmula', lyrics: '' });
  assert.throws(() => effectiveCoverRequest(body, { assetId: 'asset', sha256: 'sha' }, '/references/source.wav'),
    { status: 400, message: 'HeartMuLa covers are not available yet' });
  const s = setup({ capability: async () => { throw new Error('capability lookup must not run'); } });
  const opened = await s.open();
  const job = s.submit('cover-render', 'heartmula-pending', { ...body, ...opened }).job;
  await s.jobs.settled();
  assert.equal(s.jobs.get(job.id).status, 'failed');
  assert.match(s.jobs.get(job.id).error ?? '', /HeartMuLa covers are not available yet/);
  assert.equal(s.items.size, 0);
});

test('SA3 queues the captured full source without lyrics or an approved score', async () => {
  const s = setup({ capability: async backend => backend === 'sa3' });
  const opened = await s.open();
  const job = s.submit('cover-render', 'sa3-render', { ...input, ...opened,
    expectedBackend: 'sa3', lyrics: '', instrumental: false }).job;
  await s.jobs.settled();
  assert.equal(s.jobs.get(job.id).status, 'succeeded');
  const request = [...s.items.values()][0].request as Record<string, unknown>;
  assert.equal(request.sourceAudioUrl, '/references/source.wav');
  assert.equal(request.instrumental, true);
  assert.equal(request.style, input.caption);
});

test('SA3 top-level overrides win over compatibility controls and use cover step defaults', () => {
  const parsed = coverRenderSchema.parse({ ...input, expectedBackend: 'sa3',
    strength: 0.9, seed: 0, adapter: 'texture:0.25', steps: 12,
    engineParams: { seed: 99, randomSeed: true, stableStepSteps: 32 } });
  const request = effectiveCoverRequest(parsed, { assetId: 'asset', sha256: 'sha' }, '/references/source.wav');
  assert.equal(request.seed, 0);
  assert.equal(request.randomSeed, false);
  assert.equal(request.sa3Steps, 12);
  assert.deepEqual(request.sa3Cover, { transform: 0.9, adapters: 'texture:0.25' });
  const defaults = effectiveCoverRequest({ ...input, expectedBackend: 'sa3' },
    { assetId: 'asset', sha256: 'sha' }, '/references/source.wav');
  assert.equal(defaults.sa3Steps, 8);
  assert.deepEqual(defaults.sa3Cover, { transform: 0.3, adapters: '' });
  for (const bad of [{ strength: 1 }, { strength: 0 }, { steps: 0 }, { steps: 65 },
    { seed: 0.5 }, { adapter: '../texture:1' }, { adapter: 'texture:NaN' }]) {
    assert.equal(coverRenderSchema.safeParse({ ...parsed, ...bad }).success, false);
  }
});

test('Cover\'s own Lyric Timing choice wins over a saved global yue2LyricSchedule', () => {
  const asset = { assetId: 'asset', sha256: 'sha', approvedAbc: 'X:1\nK:C\nC|' };
  const biasedGlobal = { ...input, expectedBackend: 'yue2' as const,
    engineParams: { ...input.engineParams, yue2LyricSchedule: 'bias' },
    controls: { ...input.controls, lyricSchedule: 'off' as const } };
  assert.equal(effectiveCoverRequest(biasedGlobal, asset, '/references/source.wav').yue2LyricSchedule, 'off');
  const offGlobal = { ...input, expectedBackend: 'yue2' as const,
    engineParams: { ...input.engineParams, yue2LyricSchedule: 'off' },
    controls: { ...input.controls, lyricSchedule: 'bias' as const } };
  assert.equal(effectiveCoverRequest(offGlobal, asset, '/references/source.wav').yue2LyricSchedule, 'bias');
  const maskedCover = { ...input, expectedBackend: 'yue2' as const,
    engineParams: { ...input.engineParams, yue2LyricSchedule: 'off' },
    controls: { ...input.controls, lyricSchedule: 'mask' as const } };
  assert.equal(effectiveCoverRequest(maskedCover, asset, '/references/source.wav').yue2LyricSchedule, 'mask');
});

test('score tempo correction and key use the existing ABC transforms', () => {
  assert.deepEqual(coverScoreDetails('X:1\nQ:1/4=120\nK:Am\nC|', 0.5), { bpm: 60, key: 'A Minor' });
  assert.deepEqual(coverScoreDetails('X:1\nQ:1/4=120\nK:C\nC|', 2), { bpm: 240, key: 'C Major' });
  assert.throws(() => coverScoreDetails('X:1\nK:C\nC|', 1), /tempo/);
});

test('headless asset open, idempotency, stale transcript and replay', async () => {
  const s = setup();
  const body = { assetId: '22222222-2222-4222-8222-222222222222' };
  const first = s.submit('cover-open', 'same', body);
  assert.equal(s.submit('cover-open', 'same', body).job.id, first.job.id);
  assert.throws(() => s.submit('cover-open', 'same', { assetId: '33333333-3333-4333-8333-333333333333' }));
  await s.jobs.settled();
  const opened = s.jobs.get(first.job.id).result as { documentId: string; revision: number };
  assert.equal(s.documents.get(opened.documentId, 'owner').data.assetId, body.assetId);
  const job = s.submit('cover-transcribe', 'sheet', { ...opened, force: false }).job;
  await s.jobs.settled();
  assert.equal(s.jobs.get(job.id).status, 'succeeded');
  assert.equal(s.documents.get(opened.documentId, 'owner').revision, 2);
  const stale = s.submit('cover-transcribe', 'stale', { ...opened, force: false }).job;
  await s.jobs.settled();
  assert.equal(s.jobs.get(stale.id).status, 'failed');
  const replay = s.jobs.replay(job.id, 'owner', 0);
  assert.ok(replay.events.some(e => (e.data as any)?.status === 'succeeded'));
});

test('caption resolution advances the draft and rejects an old render snapshot', async () => {
  const s = setup();
  const opened = await s.open();
  const caption = s.submit('cover-caption', 'caption', { ...opened,
    artistId: 1, provider: '', model: '', force: false }).job;
  await s.jobs.settled();
  assert.equal((s.jobs.get(caption.id).result as any).caption, 'Resolved style');
  assert.equal(s.documents.get(opened.documentId, 'owner').revision, 2);
  const stale = s.submit('cover-render', 'stale-caption', { ...input, ...opened }).job;
  await s.jobs.settled();
  assert.equal(s.jobs.get(stale.id).status, 'failed');
  assert.equal(s.items.size, 0);
});

test('cancel rejects a late transcription result and restart never reruns it', async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const s = setup({ transcribe: async () => { await gate; return { abc: 'X:1\nK:C\nC|' }; } });
  const opened = await s.open();
  const job = s.submit('cover-transcribe', 'cancel', { ...opened, force: false }).job;
  assert.equal(s.jobs.cancel(job.id).cancelRequested, true);
  release();
  await s.jobs.settled();
  assert.equal(s.jobs.get(job.id).status, 'cancelled');
  assert.equal(s.documents.get(opened.documentId, 'owner').revision, 1);
  const hanging = setup({ transcribe: async () => new Promise(() => {}) });
  const other = await hanging.open();
  const running = hanging.submit('cover-transcribe', 'restart', { ...other, force: false }).job;
  const replacement = new WorkflowJobs({ db: hanging.db, audio: {
    enqueue: () => { throw new Error('must not rerun'); },
    get: () => { throw new Error('must not rerun'); },
    cancel: () => { throw new Error('must not rerun'); },
  } });
  replacement.reconcileAfterRestart();
  assert.equal(replacement.get(running.id).status, 'interrupted');
  assert.ok(replacement.replay(running.id, 'owner', 0).events.some(e => (e.data as any)?.status === 'interrupted'));
});

test('source replacement makes a late transcription fail its revision guard', async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const s = setup({ transcribe: async () => { await gate; return { abc: 'X:1\nK:C\nC|' }; } });
  const opened = await s.open();
  const job = s.submit('cover-transcribe', 'replace', { ...opened, force: false }).job;
  const current = s.documents.get(opened.documentId, 'owner');
  s.documents.update(opened.documentId, 'owner', opened.revision,
    { ...current.data, assetId: '33333333-3333-4333-8333-333333333333' });
  release();
  await s.jobs.settled();
  assert.equal(s.jobs.get(job.id).status, 'failed');
  assert.equal(s.documents.get(opened.documentId, 'owner').data.abc, '');
});

test('approved ACE render uses the captured asset and returns the audio result', async () => {
  const s = setup();
  const opened = await s.open();
  const job = s.submit('cover-render', 'render-success', { ...input, ...opened }).job;
  await s.jobs.settled();
  const result = s.jobs.get(job.id).result as any;
  assert.equal(result.audioIntentId, 'audio-1');
  assert.deepEqual(result.audio.songIds, ['song-1']);
  assert.equal(result.request.sourceAudioUrl, '/references/source.wav');
  assert.equal(result.request.expectedBackend, 'ace');
});

test('render rejects edit and deletion during capability lookup before audio enqueue', async () => {
  for (const change of ['edit', 'delete'] as const) {
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    const s = setup({ capability: async () => { await gate; return true; } });
    const opened = await s.open();
    const job = s.submit('cover-render', change, { ...input, ...opened }).job;
    const current = s.documents.get(opened.documentId, 'owner');
    if (change === 'edit') s.documents.update(opened.documentId, 'owner', opened.revision,
      { ...current.data, caption: 'New caption' });
    else s.db.prepare('DELETE FROM workflow_documents WHERE id = ?').run(opened.documentId);
    release();
    await s.jobs.settled();
    assert.equal(s.jobs.get(job.id).status, 'failed');
    assert.equal(s.items.size, 0);
  }
});

test('two cover submissions survive a browser disconnect and server restart', async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const s = setup({ capability: async () => { await gate; return true; } });
  const opened = await s.open();
  const first = s.submit('cover-render', 'first', { ...input, ...opened }).job;
  const second = s.submit('cover-render', 'second', { ...input, ...opened,
    title: 'Second cover' }).job;
  assert.equal(s.items.size, 0);
  assert.equal(s.jobs.get(first.id).status, 'running');
  assert.equal(s.jobs.get(second.id).status, 'pending');
  const restarted = new WorkflowJobs({ db: s.db, audio: {
    enqueue: () => { throw new Error('must not resubmit'); },
    get: () => { throw new Error('must not poll'); },
    cancel: () => { throw new Error('must not cancel'); },
  } });
  restarted.reconcileAfterRestart();
  assert.equal(restarted.get(first.id).status, 'interrupted');
  assert.equal(restarted.get(second.id).status, 'pending');
  assert.equal((restarted.get(second.id).input as CoverRenderInput).title, 'Second cover');
  release();
  await s.jobs.settled();
});

test('render cancellation cancels the audio intent', async () => {
  const s = setup({ audioStatus: 'pending' });
  const opened = await s.open();
  const job = s.submit('cover-render', 'render', { ...input, ...opened }).job;
  for (let n = 0; n < 50 && s.items.size === 0; n++) await new Promise(resolve => setTimeout(resolve, 2));
  assert.equal(s.items.size, 1);
  s.jobs.cancel(job.id);
  await s.jobs.settled();
  assert.equal([...s.items.values()][0].status, 'cancelled');
  assert.equal(s.jobs.get(job.id).status, 'cancelled');
});

test('a YuE2 render with a Prepare document renders its exported lyrics, and refuses a stale one', async () => {
  const abc = 'X:1\nK:C\n% intro\nV: Vocal\nZ|\n% verse\nV: Vocal\nC4|';
  const confirmedPrepare = (score = abc) => ({ version: 2, sourceKey: 'src', abc: score, sectionIds: ['s1', 's2'], paste: '',
    blocks: [{ id: 'b1', text: 'hello', tag: 'Verse' }],
    assignments: { s1: { kind: 'instrumental' }, s2: { kind: 'words', blockIds: ['b1'] } },
    excluded: [], lyricRevision: 3, flagged: [], nextId: 4,
    confirmed: { structure: { structureHash: structureHash(score) }, lyrics: { structureHash: structureHash(score), lyricRevision: 3 } } });
  const prepare = confirmedPrepare();
  const s = setup();
  const opened = await s.open();
  const doc = s.documents.get(opened.documentId, 'owner');
  const approved = s.documents.update(opened.documentId, 'owner', opened.revision, { ...doc.data, approvedAbc: prepare.abc });
  const yue = { ...input, ...opened, revision: approved.revision, expectedBackend: 'yue2', lyrics: '[stale]\nold words' };
  const ok = s.submit('cover-render', 'prepared', { ...yue, prepare }).job;
  await s.jobs.settled();
  assert.equal(s.jobs.get(ok.id).status, 'succeeded');
  assert.equal((s.jobs.get(ok.id).result as any).request.lyrics, '[intro]\n\n[verse]\nhello');
  const stale = s.submit('cover-render', 'stale-lyrics', { ...yue, prepare: { ...prepare, lyricRevision: 9 } }).job;
  const moved = s.submit('cover-render', 'other-score', { ...yue,
    prepare: confirmedPrepare(abc.replace('C4|', 'E4|')) }).job;
  // Instrumental mode cannot drop confirmed words, nor sing an all-instrumental map.
  const muted = s.submit('cover-render', 'muted', { ...yue, instrumental: true, prepare }).job;
  const allInstrumental = { ...prepare, blocks: [], assignments: { s1: { kind: 'instrumental' }, s2: { kind: 'instrumental' } } };
  const sung = s.submit('cover-render', 'sung', { ...yue, prepare: allInstrumental }).job;
  const quiet = s.submit('cover-render', 'quiet', { ...yue, instrumental: true, prepare: allInstrumental }).job;
  await s.jobs.settled();
  assert.match(s.jobs.get(stale.id).error ?? '', /Confirm the lyrics/);
  assert.match(s.jobs.get(moved.id).error ?? '', /score changed/);
  assert.match(s.jobs.get(muted.id).error ?? '', /have words; turn off Instrumental/);
  assert.match(s.jobs.get(sung.id).error ?? '', /turn on Instrumental/);
  assert.equal(s.jobs.get(quiet.id).status, 'succeeded');
  assert.equal(s.items.size, 2);
});

test('the Melody to Ins choice in engineParams reaches the YuE2 render request', () => {
  const asset = { assetId: 'asset', sha256: 'sha', approvedAbc: 'X:1\nK:C\nC|' };
  const resting = { ...input, expectedBackend: 'yue2' as const, instrumental: true,
    engineParams: { ...input.engineParams, yue2InstrumentalMode: 'resting' } };
  assert.equal(effectiveCoverRequest(resting, asset, '/references/source.wav').yue2InstrumentalMode, 'resting');
  const plain = { ...input, expectedBackend: 'yue2' as const, instrumental: true };
  assert.equal(effectiveCoverRequest(plain, asset, '/references/source.wav').yue2InstrumentalMode, undefined);
});
