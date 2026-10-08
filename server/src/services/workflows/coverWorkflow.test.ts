import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { WorkflowJobs, type WorkflowDeps } from './workflowJobs.js';
import { WorkflowDocuments } from './revisions.js';
import { coverScoreDetails, createCoverKinds, effectiveCoverRequest, type CoverRenderInput } from './coverWorkflow.js';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';

const input: CoverRenderInput = {
  documentId: '11111111-1111-4111-8111-111111111111', revision: 1,
  expectedBackend: 'ace', engineParams: { seed: 42, loraScale: 0.7, style: 'global style' },
  title: 'Song', artistName: 'Artist', targetArtistName: 'Target', lyrics: '[Verse]\nLine', caption: 'Cover style',
  instrumental: false, lyricsSource: null, scoreSource: null, analysis: { bpm: 100, key: 'C minor' },
  settings: { coResident: false, cacheLmCodes: true },
  controls: {
    bpmOverride: null, bpmCorrection: 2, keyOverride: null, tempoScale: 0.5,
    pitchShift: 2, noFsq: false, audioCoverStrength: 0.5, coverNoiseStrength: 0,
    coverNoiseMethod: '', vocalLanguage: 'en', sourceLatentUrl: '', timbreOverridePath: '',
    presetAdapterPath: 'adapter.safetensors', presetReferencePath: '',
    triggerUseFilename: true, triggerPlacement: 'prepend', voices: 'both', keepChords: false,
    tempoMode: 'free', coverBpm: 120, keyShift: 0, cfgScale: 1,
    lmAdapterAr: '', lmAdapterNar: '', pairMode: 'base', yue2Pick: {},
    captionMode: 'custom', captionTracks: [],
  },
};

function setup(opts: { transcribe?: (signal: AbortSignal) => Promise<{ abc: string }>;
  audioStatus?: 'pending' | 'succeeded'; capability?: (backend: string) => Promise<boolean> } = {}) {
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
        waiting: null, cancelRequested: false, createdAt: 1, updatedAt: 1 } as AudioIntentItem;
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
    analyze: async () => ({ bpm: 100, key: 'C', scale: 'minor' }),
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
