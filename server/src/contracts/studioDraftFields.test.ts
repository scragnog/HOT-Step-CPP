// Studio drafts and the playlist (contracts/studioDraftFields.ts,
// contracts/studioDrafts.ts) through the real /api/studio-drafts router.
import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import express from 'express';
import { STUDIO_DRAFT_FIELDS, draftFieldType, studioForDraftKey, validDraftField } from './studioDraftFields.js';
import { StudioDrafts } from '../services/studioDrafts/index.js';
import { createStudioDraftsRouter } from '../routes/studioDrafts.js';

// Recorded from the service before these rules moved into the contract
// (b92bb7fd): "<studio>|<accepted value kinds>" per key. s=string b=boolean
// n=finite number inf=Infinity null o=object a=array; '-' = no studio.
const RECORDED: Record<string, string[]> = {
  'cover|null,o,': ['cover-studio-analysis', 'cover-studio-metadata', 'cover-studio-selectedPreset'],
  'cover|s,': ['cover-studio-artistCaption', 'cover-studio-coverNoiseMethod', 'cover-studio-coverTimbreOverride', 'cover-studio-coverVocalLanguage',
    'cover-studio-lyrics', 'cover-studio-songArtist', 'cover-studio-songTitle', 'cover-studio-sourceAssetId', 'cover-studio-sourceAudioUrl',
    'cover-studio-sourceFileName', 'cover-studio-sourceLatentUrl', 'cover-studio-sourceSongId'],
  'cover|n,': ['cover-studio-audioCoverStrength', 'cover-studio-bpmCorrection', 'cover-studio-coverNoiseStrength', 'cover-studio-pitchShift',
    'cover-studio-sepLevel', 'cover-studio-tempoScale'],
  'cover|n,null,': ['cover-studio-bpmOverride', 'cover-studio-selectedArtistId'],
  'cover|b,': ['cover-studio-coverInstrumental', 'cover-studio-datasetAnalysis', 'cover-studio-noFsq'],
  'cover|s,null,': ['cover-studio-keyOverride', 'cover-studio-lyricsSource'],
  '-|': ['cover-studio-nope', 'hs-unknown', 'hs-yue2CaptionSource:ds:a:b', 'lireek-playQueue'],
  'create|s,': ['hs-artist', 'hs-caption', 'hs-keyScale', 'hs-lora-trigger', 'hs-lyrics', 'hs-negative-prompt', 'hs-sourceLatentUrl', 'hs-subject',
    'hs-timeSignature', 'hs-title', 'hs-vocalGender', 'hs-vocalLanguage', 'hs-yue2CaptionDataset'],
  'create|b,': ['hs-beat-intro', 'hs-instrumental'],
  'create|n,': ['hs-bpm', 'hs-duration', 'hs-intro-bars'],
  'create|null,o,': ['hs-mm3CaptionSource:x', 'hs-mm3CaptionSources', 'hs-yue2CaptionSource:ds:d1', 'hs-yue2CaptionSource:ds:song:d1:s1',
    'hs-yue2CaptionSources'],
  'repaint|n,': ['hs-repaint-crossfadeFrames', 'hs-repaint-regionEnd', 'hs-repaint-regionStart'],
  'repaint|s,': ['hs-repaint-lyrics', 'hs-repaint-repaintMode', 'hs-repaint-sourceAssetId', 'hs-repaint-sourceAudioUrl', 'hs-repaint-sourceName',
    'hs-repaint-styleCaption'],
  'repaint|null,o,': ['hs-repaint-sourceSong'],
  'stem-builder|s,': ['hs-sb-model', 'hs-sb-sourceFile', 'hs-sb-sourceUrl'],
  'stem-builder|null,o,': ['hs-sb-sourceRef'],
  'stem-studio|s,': ['hs-stem-extractModel', 'hs-stem-sepLevel', 'hs-stem-sourceFile', 'hs-stem-sourceUrl'],
  'storm|b,': ['hs-storm-beat-intro', 'hs-storm-instrumental'],
  'storm|n,': ['hs-storm-bpm', 'hs-storm-deck-a-bpm', 'hs-storm-deck-b-bpm', 'hs-storm-duration-v2', 'hs-storm-intro-bars'],
  'storm|s,': ['hs-storm-caption', 'hs-storm-deck-a-caption', 'hs-storm-deck-a-lyrics', 'hs-storm-deck-b-caption', 'hs-storm-deck-b-lyrics',
    'hs-storm-lora', 'hs-storm-lyrics', 'hs-storm-neg'],
};
const VALUES: Array<[string, unknown]> = [['s', 'x'], ['b', true], ['n', 1], ['inf', Infinity], ['null', null], ['o', {}], ['a', []]];

test('every draft key keeps its recorded studio and accepted value kinds', () => {
  for (const [expected, keys] of Object.entries(RECORDED)) for (const key of keys) {
    const studio = studioForDraftKey(key);
    const accepted = studio ? VALUES.filter(([, v]) => validDraftField(key, v)).map(([tag]) => `${tag},`).join('') : '';
    assert.equal(`${studio ?? '-'}|${accepted}`, expected, key);
  }
  const catalogued = Object.values(STUDIO_DRAFT_FIELDS).flatMap(fields => Object.keys(fields));
  const recordedFixed = Object.entries(RECORDED).filter(([k]) => !k.startsWith('-')).flatMap(([, keys]) => keys)
    .filter(key => !/CaptionSource|CaptionDataset/.test(key));
  assert.deepEqual([...catalogued].sort(), [...recordedFixed].sort());
  for (const [studio, fields] of Object.entries(STUDIO_DRAFT_FIELDS)) for (const [key, type] of Object.entries(fields)) {
    assert.equal(studioForDraftKey(key), studio);
    assert.equal(draftFieldType(key), type);
  }
});

function serve() {
  const db = new Database(':memory:');
  const store = new StudioDrafts(db);
  const app = express();
  app.use(express.json());
  app.use('/', createStudioDraftsRouter(store, req => (req.headers['x-user'] as string) || null));
  const server = app.listen(0);
  const ready = new Promise(resolve => server.once('listening', resolve));
  const call = async (method: string, p: string, body?: unknown, user = 'u') => {
    await ready;
    const res = await fetch(`http://127.0.0.1:${(server.address() as { port: number }).port}${p}`, { method,
      headers: { 'Content-Type': 'application/json', ...(user ? { 'x-user': user } : {}) }, body: body === undefined ? undefined : JSON.stringify(body) });
    return { status: res.status, body: await res.json() as any };
  };
  return { db, call, close: () => new Promise(resolve => server.close(resolve)) };
}

test('playlist commands: revisions, reorder rules, ignored unknown ids and 409 with currentRevision', async () => {
  const { call, close } = serve();
  try {
    const item = (id: string) => ({ id, title: id, audioUrl: `/audio/${id}.wav`, extra: { kept: true } });
    const send = (expectedRevision: number, command: unknown) => call('POST', '/playlist/commands', { expectedRevision, command });
    assert.deepEqual((await call('GET', '/playlist')).body, { document: null });
    let r = await send(0, { operation: 'add', item: item('a') });
    assert.equal(r.body.document.revision, 1);
    r = await send(1, { operation: 'add', item: item('b') });
    r = await send(2, { operation: 'add', item: item('a') });
    assert.deepEqual([r.body.document.revision, r.body.document.body.items.map((i: any) => i.id)], [3, ['a', 'b']], 'a duplicate add changes nothing but the revision');
    assert.deepEqual(r.body.document.body.items[0].extra, { kept: true }, 'unknown item fields are stored as given');
    for (const ids of [['b'], ['b', 'b'], ['b', 'a', 'c']]) {
      const bad = await send(3, { operation: 'reorder', ids });
      assert.deepEqual([bad.status, bad.body.error], [400, 'Reorder must name every current item exactly once'], ids.join());
    }
    r = await send(3, { operation: 'reorder', ids: ['b', 'a'] });
    assert.deepEqual(r.body.document.body.items.map((i: any) => i.id), ['b', 'a']);
    r = await send(4, { operation: 'update', id: 'gone', patch: { title: 'x' } });
    r = await send(5, { operation: 'remove', id: 'gone' });
    assert.deepEqual([r.body.document.revision, r.body.document.body.items.length], [6, 2]);
    const stale = await send(2, { operation: 'clear' });
    assert.deepEqual([stale.status, stale.body.currentRevision], [409, 6]);
    assert.equal((await send(6, { operation: 'shuffle' })).status, 400);
    assert.equal((await call('POST', '/playlist/commands', { command: { operation: 'clear' } })).status, 400, 'expectedRevision is required');
    assert.equal((await call('GET', '/playlist', undefined, '')).status, 401);
    r = await send(6, { operation: 'clear' });
    assert.deepEqual(r.body.document.body.items, []);
  } finally { await close(); }
});

test('drafts: field checks, stale revisions, studio lock, deleted source and source-switch cleanup', async () => {
  const { call, close } = serve();
  try {
    const body = { studio: 'cover', sourceAssetId: 'asset-1', fields: { 'cover-studio-sourceAssetId': 'asset-1', 'cover-studio-analysis': { bpm: 120 },
      'cover-studio-lyrics': 'mine', 'cover-studio-bpmOverride': null }, anyExtra: 'kept' };
    assert.equal((await call('POST', '/drafts', { body, expectedRevision: 1 })).status, 400);
    assert.equal((await call('POST', '/drafts', { body: { ...body, fields: { 'hs-caption': 'x' } } })).status, 400);
    assert.equal((await call('POST', '/drafts', { body: { ...body, fields: { 'cover-studio-noFsq': 'yes' } } })).status, 400);
    assert.equal((await call('POST', '/drafts', { body: { ...body, sourceAssetId: 'other' } })).status, 400, 'identity must agree with the fields');
    const created = await call('POST', '/drafts', { body });
    assert.equal(created.status, 201);
    const doc = created.body.document;
    assert.deepEqual([doc.revision, doc.body.anyExtra], [1, 'kept']);
    const read = await call('GET', `/drafts/${doc.id}`);
    assert.match(read.body.sourceError, /Source asset asset-1 is unavailable/, 'a deleted source asset is reported, not fatal');
    assert.equal((await call('PUT', `/drafts/${doc.id}`, { body })).status, 400, 'expectedRevision is required');
    assert.equal((await call('PUT', `/drafts/${doc.id}`, { body: { ...body, studio: 'repaint', fields: {} }, expectedRevision: 1 })).status, 400);
    const stale = await call('PUT', `/drafts/${doc.id}`, { body, expectedRevision: 7 });
    assert.deepEqual([stale.status, stale.body.currentRevision], [409, 1]);
    const switched = await call('PUT', `/drafts/${doc.id}`, { expectedRevision: 1, body: { ...body, sourceAssetId: 'asset-2',
      fields: { ...body.fields, 'cover-studio-sourceAssetId': 'asset-2' } } });
    assert.equal(switched.body.document.revision, 2);
    assert.equal(switched.body.document.body.fields['cover-studio-analysis'], undefined, 'old-source analysis dropped');
    assert.equal(switched.body.document.body.fields['cover-studio-lyrics'], 'mine', 'user text kept');
    assert.equal((await call('DELETE', `/drafts/${doc.id}`)).status, 400);
    assert.equal((await call('DELETE', `/drafts/${doc.id}?expectedRevision=1`)).status, 409);
    assert.deepEqual((await call('DELETE', `/drafts/${doc.id}?expectedRevision=2`)).body, { removed: true });
    assert.equal((await call('GET', `/drafts/${doc.id}`)).status, 404);
  } finally { await close(); }
});
