import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import Database from 'better-sqlite3';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';
import { WorkflowJobs, type WorkflowDeps } from './workflowJobs.js';
import { createRepaintLayerKinds, effectiveLayerRequest, effectiveRepaintRequest,
  resolveRepaintLayerSource, type LayerInput, type RepaintInput } from './repaintLayerWorkflow.js';
import { recordAudioAsset } from '../assets/audioAssets.js';

const assetId = 'aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa';
const common = { source: { kind: 'asset' as const, id: assetId, expectedUrl: '/references/source.wav' },
  expectedBackend: 'ace', engineParams: { seed: 42, guidanceScale: 9, style: 'global',
    source: 'wrong', postProcessingEnabled: true, adapterGroupScales: [1] } };
const repaint: RepaintInput = { ...common, regionStart: 0, regionEnd: 42,
  lyrics: '', styleCaption: '', sourceName: 'Source', repaintMode: 'balanced', crossfadeFrames: 10 };
const layer: LayerInput = { ...common, trackName: 'backing_vocals', buildModel: 'acestep-v15-base-test', caption: '' };
const src = { url: '/references/source.wav', path: '/source.wav' };

test('full effective requests preserve captured globals and every mode override', () => {
  assert.deepEqual(effectiveRepaintRequest(repaint, src), {
    ...common.engineParams, customMode: true, taskType: 'repaint', sourceAudioUrl: src.url,
    repaintingStart: 0, repaintingEnd: 42, lyrics: '[Instrumental]', style: 'global',
    title: 'Source (Repaint)', duration: 0, source: 'repaint', repaintCrossfadeFrames: 10,
    repaintInjectionRatio: 0.5, expectedBackend: 'ace',
  });
  assert.deepEqual(effectiveLayerRequest(layer, src), {
    ...common.engineParams, customMode: true, taskType: 'lego', trackName: 'backing_vocals',
    sourceAudioUrl: src.url, caption: '', lyrics: '[Instrumental]', duration: 0,
    instrumental: true, source: 'stem-builder', title: 'backing vocals layer',
    ditModel: 'acestep-v15-base-test', loraPath: '', loraScale: 0,
    adapterGroupScales: undefined, adapterMode: undefined, masteringEnabled: false,
    masteringReference: undefined, timbreReference: undefined, guidanceScale: 1,
    guidanceMode: 'apg', shift: 1, useCotCaption: false, inferMethod: 'euler',
    scheduler: 'linear', postProcessingEnabled: false, vocalNaturalizerEnabled: false,
    spectralLifterEnabled: false, ppVaeReencode: false, stableStepOn: false,
    denoiseStrength: 0, dcwEnabled: false, dcwMode: undefined, dcwLowScaler: undefined,
    dcwHighScaler: undefined, latentShift: 0, latentRescale: 1, customTimesteps: '',
    cfgCutoffRatio: 1, lmCfgCutoffRatio: 1, cacheRatio: 0, audioCoverStrength: 1,
    bpm: 0, keyScale: '', timeSignature: '', expectedBackend: 'ace',
  });
  assert.equal(effectiveRepaintRequest({ ...repaint, repaintMode: 'conservative' }, src).repaintInjectionRatio, 0.7);
  assert.equal(effectiveRepaintRequest({ ...repaint, repaintMode: 'aggressive' }, src).repaintInjectionRatio, 0.3);
  assert.equal(effectiveLayerRequest({ ...layer, trackName: 'vocals' }, src).lyrics, '');
});

test('source identity enforces ownership, current URL and file existence', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'repaint-source-'));
  const db = new Database(':memory:');
  try {
    fs.mkdirSync(path.join(dir, 'references'));
    fs.mkdirSync(path.join(dir, 'audio'));
    const file = path.join(dir, 'references', 'source.wav');
    fs.writeFileSync(file, 'audio');
    const asset = recordAudioAsset(db, { userId: 'owner', url: '/references/source.wav', filename: 'source.wav', size: 5, sha256: 'abc' });
    const dirs = { dataDir: dir, audioDir: path.join(dir, 'audio') };
    assert.equal(resolveRepaintLayerSource(db, { kind: 'asset', id: asset.id, expectedUrl: asset.url }, 'owner', dirs).url, asset.url);
    assert.throws(() => resolveRepaintLayerSource(db, { kind: 'asset', id: asset.id, expectedUrl: asset.url }, 'other', dirs));
    assert.throws(() => resolveRepaintLayerSource(db, { kind: 'asset', id: asset.id, expectedUrl: '/references/replaced.wav' }, 'owner', dirs));
    fs.unlinkSync(file);
    assert.throws(() => resolveRepaintLayerSource(db, { kind: 'asset', id: asset.id, expectedUrl: asset.url }, 'owner', dirs));
    db.exec('CREATE TABLE songs (id TEXT, user_id TEXT, audio_url TEXT, latent_url TEXT)');
    db.prepare('INSERT INTO songs VALUES (?, ?, ?, ?)').run('song-1', 'owner', '/audio/song.wav', '/audio/song.latent');
    const songFile = path.join(dir, 'audio', 'song.wav');
    fs.writeFileSync(songFile, 'audio');
    const ref = { kind: 'song' as const, id: 'song-1', expectedUrl: '/audio/song.wav' };
    assert.equal(resolveRepaintLayerSource(db, ref, 'owner', dirs).latentUrl, '/audio/song.latent');
    assert.throws(() => resolveRepaintLayerSource(db, ref, 'other', dirs));
    db.prepare('UPDATE songs SET audio_url = ? WHERE id = ?').run('/audio/new.wav', 'song-1');
    assert.throws(() => resolveRepaintLayerSource(db, ref, 'owner', dirs));
    db.prepare('DELETE FROM songs WHERE id = ?').run('song-1');
    assert.throws(() => resolveRepaintLayerSource(db, ref, 'owner', dirs));
  } finally { db.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});

function setup(audioStatus: 'succeeded' | 'pending' = 'succeeded', duration = 42, supported = true) {
  const db = new Database(':memory:');
  const items = new Map<string, AudioIntentItem>();
  let submitted = 0;
  const audio: WorkflowDeps['audio'] = {
    enqueue: ({ idempotencyKey, request, meta }) => {
      const old = [...items.values()].find(item => item.idempotencyKey === idempotencyKey);
      if (old) return { item: old, created: false };
      const item = { id: `audio-${++submitted}`, idempotencyKey, request, meta: meta || null,
        status: audioStatus, result: { audioUrls: ['/audio/output.wav'], songIds: ['song-2'] }, error: null,
        engine: 'ace', jobId: 'render', attempt: 1, previousJobIds: [], waiting: null,
        cancelRequested: false, createdAt: 1, updatedAt: 1 } as AudioIntentItem;
      items.set(item.id, item);
      return { item, created: true };
    },
    get: id => items.get(id)!,
    cancel: id => { const item = items.get(id)!; item.status = 'cancelled'; return item; },
  };
  const jobs = new WorkflowJobs({ db, audio, audioPollMs: 1 });
  for (const kind of createRepaintLayerKinds({
    resolveSource: () => src, duration: async () => duration,
    checkCapability: async (_backend, feature) => {
      if (!supported || !['repaint', 'lego'].includes(feature)) throw new Error('Unsupported capability');
    },
  })) jobs.register(kind);
  const submit = (kind: string, key: string, input: Record<string, unknown>) =>
    jobs.submit({ kind, idempotencyKey: key, input }, 'owner');
  return { db, jobs, items, submit, get submitted() { return submitted; } };
}

test('zero and edge bounds enqueue once; bad bounds and capabilities enqueue nothing', async () => {
  const s = setup();
  const first = s.submit('repaint-render', 'same', repaint as unknown as Record<string, unknown>);
  const duplicate = s.submit('repaint-render', 'same', repaint as unknown as Record<string, unknown>);
  assert.equal(duplicate.created, false);
  assert.equal(duplicate.job.id, first.job.id);
  await s.jobs.settled();
  assert.equal(s.submitted, 1);
  assert.equal(s.jobs.get(first.job.id).status, 'succeeded');
  const replay = s.jobs.replay(first.job.id, 'owner', 0);
  assert.equal(replay.events.at(-1)?.type, 'status');
  const cursor = replay.events[1].seq;
  assert.deepEqual(s.jobs.replay(first.job.id, 'owner', cursor).events.map(e => e.seq),
    replay.events.filter(e => e.seq > cursor).map(e => e.seq));
  const bad = s.submit('repaint-render', 'bad', { ...repaint, regionEnd: 43 });
  await s.jobs.settled();
  assert.equal(s.jobs.get(bad.job.id).status, 'failed');
  assert.equal(s.submitted, 1);
  const unsupported = setup('succeeded', 42, false);
  const job = unsupported.submit('layer-render', 'no', layer as unknown as Record<string, unknown>);
  await unsupported.jobs.settled();
  assert.equal(unsupported.jobs.get(job.job.id).status, 'failed');
  assert.equal(unsupported.submitted, 0);
  assert.throws(() => s.submit('repaint-render', 'negative', { ...repaint, regionStart: -1 }));
  assert.throws(() => s.submit('repaint-render', 'zero-end', { ...repaint, regionEnd: 0 }));
  assert.throws(() => s.submit('layer-render', 'not-base', { ...layer, buildModel: 'acestep-v15-turbo' }));
});

test('cancel and restart interrupt without rerun; replay retains completion status', async () => {
  const s = setup('pending');
  const job = s.submit('layer-render', 'cancel', layer as unknown as Record<string, unknown>).job;
  await new Promise(resolve => setTimeout(resolve, 10));
  s.jobs.cancel(job.id, 'owner');
  await s.jobs.settled();
  assert.equal(s.jobs.get(job.id).status, 'cancelled');
  assert.equal([...s.items.values()][0].status, 'cancelled');
  const second = s.submit('layer-render', 'restart', layer as unknown as Record<string, unknown>).job;
  await new Promise(resolve => setTimeout(resolve, 10));
  const after = new WorkflowJobs({ db: s.db, audio: {
    enqueue: () => { throw new Error('unexpected enqueue'); }, get: () => { throw new Error('unexpected get'); },
    cancel: () => { throw new Error('unexpected cancel'); },
  } });
  after.reconcileAfterRestart();
  assert.equal(after.get(second.id).status, 'interrupted');
  assert.equal(after.replay(second.id, 'owner', 0).events.at(-1)?.type, 'status');
  assert.equal(s.submitted, 2);
  for (const item of s.items.values()) item.status = 'interrupted';
  await s.jobs.settled();
});
