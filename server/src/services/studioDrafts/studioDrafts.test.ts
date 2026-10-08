import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import Database from 'better-sqlite3';
import express from 'express';
import { StudioDrafts, validateDraft } from './index.js';
import { withoutStaleSourceResults } from './index.js';
import { WorkflowError } from '../workflows/workflowJobs.js';
import { WorkflowDocuments } from '../workflows/revisions.js';
import { createStudioDraftsRouter } from '../../routes/studioDrafts.js';
import type { StudioDraftBody } from '../../contracts/studioDrafts.js';

const hash = (raw: string) => `sha256:${createHash('sha256').update(raw).digest('hex')}`;
const item = (id: string) => ({ id, title: id, audioUrl: `/audio/${id}.flac`, generationParams: { seed: 0, adapter: false } });
const fresh = () => new StudioDrafts(new Database(':memory:'));

test('playlist preserves orphaned ids, order, variants and generation parameters', () => {
  const store = fresh();
  const first = store.command('u', 0, { operation: 'add', item: item('song') });
  const second = store.command('u', first.revision, { operation: 'add', item: { ...item('queue-id'), masteredAudioUrl: '', noAdapterAudioUrl: '/audio/variant.flac' } });
  const reordered = store.command('u', second.revision, { operation: 'reorder', ids: ['queue-id', 'song'] });
  assert.deepEqual(reordered.body.items.map(i => i.id), ['queue-id', 'song']);
  assert.equal(reordered.body.items[0].noAdapterAudioUrl, '/audio/variant.flac');
  assert.deepEqual(reordered.body.items[1].generationParams, { seed: 0, adapter: false });
  assert.equal(store.playlist('u')?.body.items.length, 2);
});

test('playlist rejects stale and invalid changes without a partial write', () => {
  const store = fresh();
  const first = store.command('u', 0, { operation: 'add', item: item('a') });
  assert.throws(() => store.command('u', 0, { operation: 'clear' }), (e: unknown) => e instanceof WorkflowError && e.status === 409);
  assert.throws(() => store.command('u', first.revision, { operation: 'reorder', ids: ['missing'] }), (e: unknown) => e instanceof WorkflowError && e.status === 400);
  assert.deepEqual(store.playlist('u')?.body.items.map(i => i.id), ['a']);
  const duplicate = store.command('u', first.revision, { operation: 'add', item: item('a') });
  assert.equal(duplicate.body.items.length, 1);
});

test('browser import retries return a receipt and never consume invalid or credential keys', () => {
  const store = fresh();
  const raw = JSON.stringify([item('queue-id')]);
  const request = { storageKey: 'lireek-playQueue', raw, sourceHash: hash(raw), schemaVersion: 1 as const, expectedRevision: 0 };
  const first = store.importValue('u', request);
  assert.equal(first.created, true);
  const repeat = store.importValue('u', request);
  assert.equal(repeat.created, false);
  assert.equal(repeat.receipt.documentId, first.receipt.documentId);
  assert.equal(store.playlists.list('u').length, 1);
  const bad = { ...request, storageKey: 'hs-hfToken' };
  assert.throws(() => store.importValue('u', bad), (e: unknown) => e instanceof WorkflowError && e.status === 400);
  assert.equal(store.drafts.list('u').length, 0);
});

test('raw stem keys keep literal strings and a changed import needs an explicit choice', () => {
  const store = fresh();
  const raw = 'source.wav';
  const req = { storageKey: 'hs-stem-sourceFile', raw, sourceHash: hash(raw), schemaVersion: 1 as const, expectedRevision: 0 };
  const first = store.importValue('u', req);
  assert.deepEqual(first.document?.body, { studio: 'stem-studio', fields: { 'hs-stem-sourceFile': raw } });
  const changed = { ...req, raw: 'other.wav', sourceHash: hash('other.wav') };
  assert.throws(() => store.importValue('u', changed), (e: unknown) => e instanceof WorkflowError && e.status === 409);
  const replaced = store.importValue('u', { ...changed, expectedRevision: 1, resolution: 'replace' });
  assert.equal((replaced.document?.body as StudioDraftBody).fields['hs-stem-sourceFile'], 'other.wav');
  assert.equal(replaced.document?.revision, 2);
  assert.equal(store.drafts.list('u').length, 1);
  assert.equal(store.importValue('u', changed).receipt.documentId, first.receipt.documentId);
  const third = { ...req, raw: 'third.wav', sourceHash: hash('third.wav'), expectedRevision: 2, resolution: 'keep-both' as const };
  assert.equal(store.importValue('u', third).created, true);
  assert.equal(store.drafts.list('u').length, 2);
});

test('a matching destination gets a receipt without duplicating its draft', () => {
  const store = fresh();
  const document = store.drafts.create('u', { studio: 'create', fields: { 'hs-bpm': 0 } }, { origin: 'client' });
  const raw = '0';
  const result = store.importValue('u', { storageKey: 'hs-bpm', raw, sourceHash: hash(raw), schemaVersion: 1, expectedRevision: document.revision });
  assert.equal(result.created, false);
  assert.equal(result.document?.id, document.id);
  assert.equal(store.drafts.list('u').length, 1);
});

test('golden browser draft values keep false, zero, null, caption selection and literal stem paths', () => {
  const store = fresh();
  const cases = [
    ['hs-bpm', '0', 0],
    ['hs-instrumental', 'false', false],
    ['cover-studio-lyricsSource', 'null', null],
    ['hs-repaint-regionStart', '30.5', 30.5],
    ['hs-mm3CaptionSource:42', '{"mode":"track","selectedTitle":"Track One"}', { mode: 'track', selectedTitle: 'Track One' }],
    ['hs-storm-caption', '"live caption"', 'live caption'],
    ['hs-stem-sourceUrl', '/references/source.flac', '/references/source.flac'],
    ['hs-sb-model', 'htdemucs', 'htdemucs'],
  ] as const;
  for (const [storageKey, raw, expected] of cases) {
    const result = store.importValue('u', { storageKey, raw, sourceHash: hash(raw), schemaVersion: 1, expectedRevision: 0 });
    assert.deepEqual((result.document?.body as StudioDraftBody).fields[storageKey], expected);
  }
  assert.equal(store.drafts.list('u').length, cases.length);
});

test('draft route checks ownership and stale writes without starting work', async () => {
  const store = fresh();
  const app = express();
  app.use(express.json());
  app.use('/api/studio-drafts', createStudioDraftsRouter(store, req =>
    req.headers.authorization === 'Bearer owner' ? 'owner' : req.headers.authorization === 'Bearer other' ? 'other' : null));
  const server = await new Promise<ReturnType<typeof app.listen>>(resolve => { const s = app.listen(0, () => resolve(s)); });
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}/api/studio-drafts`;
  const request = (url: string, method = 'GET', body?: unknown, token = 'owner') => fetch(base + url, {
    method, headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body),
  });
  try {
    assert.equal((await request('/drafts', 'GET', undefined, 'bad')).status, 401);
    const created = await request('/drafts', 'POST', { body: { studio: 'create', fields: { 'hs-bpm': 0, 'hs-instrumental': false } } });
    assert.equal(created.status, 201);
    const doc = (await created.json()).document;
    assert.equal((await request(`/drafts/${doc.id}`, 'GET', undefined, 'other')).status, 404);
    const saved = await request(`/drafts/${doc.id}`, 'PUT', { expectedRevision: 1, body: { studio: 'create', fields: { 'hs-bpm': 120 } } });
    assert.equal(saved.status, 200);
    const stale = await request(`/drafts/${doc.id}`, 'PUT', { expectedRevision: 1, body: { studio: 'create', fields: { 'hs-bpm': 60 } } });
    assert.equal(stale.status, 409);
    assert.equal((await stale.json()).currentRevision, 2);
    assert.equal(store.drafts.get(doc.id, 'owner').body.fields['hs-bpm'], 120);
    const added = await request('/playlist/commands', 'POST', { expectedRevision: 0, command: { operation: 'add', item: item('orphan') } });
    assert.equal(added.status, 200);
    assert.equal((await added.json()).document.body.items[0].id, 'orphan');
    const conflict = await request('/playlist/commands', 'POST', { expectedRevision: 0, command: { operation: 'clear' } });
    assert.equal(conflict.status, 409);
    assert.equal((await conflict.json()).currentRevision, 1);
    assert.equal((await request('/playlist')).status, 200);
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); }
});

test('changing a source clears stale analysis and caption while retaining user edits', () => {
  const old: StudioDraftBody = { studio: 'cover', fields: {
    'cover-studio-sourceAssetId': 'asset-a', 'cover-studio-analysis': { bpm: 120 },
    'cover-studio-artistCaption': 'derived caption', 'cover-studio-lyrics': 'my edit',
  } };
  const next = { ...old, fields: { ...old.fields, 'cover-studio-sourceAssetId': 'asset-b' } };
  assert.deepEqual(withoutStaleSourceResults(old, next).fields, {
    'cover-studio-sourceAssetId': 'asset-b', 'cover-studio-lyrics': 'my edit',
  });
  assert.deepEqual(withoutStaleSourceResults(old, old), old);
});

test('a missing uploaded source is reported on draft load', () => {
  const store = fresh();
  const body: StudioDraftBody = { studio: 'repaint', fields: { 'hs-repaint-sourceAssetId': 'missing-asset' } };
  const message = store.sourceError('u', body, '.');
  assert.match(message ?? '', /unavailable.*Reupload/);
});

test('captured source identity cannot disagree with its editing field', () => {
  assert.throws(() => validateDraft({ studio: 'cover', sourceAssetId: 'new', fields: { 'cover-studio-sourceAssetId': 'old' } }),
    (e: unknown) => e instanceof WorkflowError && e.status === 400);
  validateDraft({ studio: 'cover', sourceAssetId: 'same', fields: { 'cover-studio-sourceAssetId': 'same' } });
});

test('malformed scalar imports fail without a receipt or a draft', () => {
  const store = fresh();
  const raw = '{"unexpected":true}';
  assert.throws(() => store.importValue('u', {
    storageKey: 'hs-bpm', raw, sourceHash: hash(raw), schemaVersion: 1, expectedRevision: 0,
  }), (e: unknown) => e instanceof WorkflowError && e.status === 400);
  assert.equal(store.drafts.list('u').length, 0);
  assert.equal(store.drafts.receipts('u').length, 0);
});

test('existing Cover and training-to-Create document ids remain readable', () => {
  const db = new Database(':memory:');
  const docs = new WorkflowDocuments(db);
  const store = new StudioDrafts(db, docs);
  for (const kind of ['cover-draft', 'training-audition-create']) {
    const legacy = docs.create('u', kind, { captured: true });
    assert.equal(store.handoff('u', legacy.id).revision, 1);
    assert.throws(() => store.handoff('other', legacy.id), (e: unknown) => e instanceof WorkflowError && e.status === 404);
  }
  const unrelated = docs.create('u', 'other', {});
  assert.throws(() => store.handoff('u', unrelated.id), (e: unknown) => e instanceof WorkflowError && e.status === 404);
});
