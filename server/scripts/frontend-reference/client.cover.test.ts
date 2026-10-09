// client.cover.test.ts — Cover Studio's full chain through the documented
// workflow-job sequence: open, caption, transcribe, score approval (a draft
// document write), render; plus stale revision, rejected input,
// cancellation and auth failure.
//
// Two steps reach outside Node in production: the caption step calls an LLM
// provider and the transcribe step runs the YuE2 transcriber. Both are
// swapped for fixtures through coverWorkflow.ts's overrideCoverWorkflowDeps
// seam. That replaces only the dependency, never a route: every request
// below still goes through the real /api/workflows routes, the real
// cover-* kinds, the real draft documents and the real audio queue, with the
// fake engine standing in for ace-server.

import test from 'node:test';
import assert from 'node:assert/strict';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient, ClientError } from './client.js';
import { makeFixtureWav } from './fixtures.js';

let server: FakeServer;
let client: ReferenceClient;
let clearOverrides: () => void;
const calls = { caption: 0, transcribe: 0 };
/** While set, the fake caption waits until its job is cancelled. */
let holdCaption = false;

const FIXTURE_ABC = 'X:1\nT:fixture\nM:4/4\nL:1/8\nK:C\nCDEF GABc|';

test.before(async () => {
  server = await startFakeServer();
  const { overrideCoverWorkflowDeps } = await import('../../src/services/workflows/coverWorkflow.js');
  clearOverrides = overrideCoverWorkflowDeps({
    caption: async (_artistId, _provider, _model, _force, signal) => {
      calls.caption++;
      if (holdCaption) await new Promise((_, reject) => signal.addEventListener('abort', () => reject(new Error('Cancelled')), { once: true }));
      return { text: 'fixture caption: dusty funk, warm bass' };
    },
    transcribe: async () => { calls.transcribe++; return { abc: FIXTURE_ABC, scoreSource: 'fixture' }; },
  });
  client = new ReferenceClient(server.origin);
  await client.login();
});
test.after(async () => { clearOverrides?.(); await server.close(); });

const controls = {
  bpmOverride: null, bpmCorrection: 1, keyOverride: null, tempoScale: 1, pitchShift: 0, noFsq: false,
  audioCoverStrength: 0.5, coverNoiseStrength: 0, coverNoiseMethod: '', vocalLanguage: 'en', sourceLatentUrl: '',
  timbreOverridePath: '', presetAdapterPath: '', presetReferencePath: '', triggerUseFilename: false, triggerPlacement: 'prepend',
  voices: 'vocal', keepChords: false, tempoMode: 'source', coverBpm: 120, keyShift: 0, cfgScale: 1.5,
  lmAdapterAr: '', lmAdapterNar: '', pairMode: 'base', yue2Pick: {}, captionMode: 'custom', captionTracks: [],
} as const;
const renderInput = (documentId: string, revision: number, caption: string, over: Record<string, unknown> = {}) => ({
  documentId, revision, expectedBackend: 'ace', engineParams: { skipLm: true, seed: 3, randomSeed: false },
  title: 'Fixture', artistName: 'Fixture Artist', targetArtistName: '', lyrics: '', caption, instrumental: true,
  lyricsSource: null, scoreSource: null, analysis: null, settings: {}, controls, ...over,
});

async function openDraft() {
  const asset = await client.uploadAudio(makeFixtureWav(2), 'cover-source.wav');
  const open = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-open', { assetId: asset.asset_id })).id);
  assert.equal(open.status, 'succeeded', JSON.stringify(open.error));
  return { asset, ...(open.result as { documentId: string; revision: number }) };
}

test('Cover: open → caption → transcribe → approve → render, each step on the revision the last returned', async () => {
  const { documentId, revision: r1 } = await openDraft();

  const caption = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-caption',
    { documentId, revision: r1, artistId: 1, provider: 'fixture', model: '', force: false })).id);
  assert.equal(caption.status, 'succeeded', JSON.stringify(caption.error));
  const c = caption.result as { revision: number; caption: string };
  assert.equal(c.caption, 'fixture caption: dusty funk, warm bass');
  assert.equal(c.revision, r1 + 1);

  const transcribe = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-transcribe',
    { documentId, revision: c.revision, force: false })).id);
  assert.equal(transcribe.status, 'succeeded', JSON.stringify(transcribe.error));
  const t = transcribe.result as { revision: number; abc: string; scoreSource: string };
  assert.deepEqual([t.abc, t.scoreSource, t.revision], [FIXTURE_ABC, 'fixture', c.revision + 1]);

  // Approval is the client's own edit of the draft document.
  const { document: before } = await client.getDocument(documentId);
  assert.equal(before.revision, t.revision);
  const approved = await client.putDocument(documentId, t.revision, { ...(before.data as object), approvedAbc: t.abc });
  assert.equal(approved.document.revision, t.revision + 1);
  assert.equal((approved.document.data as { approvedAbc: string }).approvedAbc, FIXTURE_ABC);

  const render = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-render',
    renderInput(documentId, approved.document.revision, c.caption))).id);
  assert.equal(render.status, 'succeeded', JSON.stringify(render.error));
  const r = render.result as { request: Record<string, unknown>; audioIntentId: string; audio: { songIds?: string[] } };
  assert.equal(r.request.taskType, 'cover');
  assert.equal(r.request.style, c.caption);
  assert.match(String(r.request.sourceAudioUrl), /^\/references\//);
  assert.ok(r.audio.songIds?.[0], 'a song row from the fake engine render');
  assert.deepEqual(calls, { caption: 1, transcribe: 1 });
});

test('Cover: a stale draft revision fails the step; the draft keeps the newer write', async () => {
  const { documentId, revision } = await openDraft();
  const { document } = await client.getDocument(documentId);
  await client.putDocument(documentId, revision, { ...(document.data as object), approvedAbc: 'X:1\nK:C\nC|' });
  const stale = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-transcribe', { documentId, revision, force: false })).id);
  assert.equal(stale.status, 'failed');
  assert.match(JSON.stringify(stale.error), /Stale cover draft revision/);
  await assert.rejects(client.putDocument(documentId, revision, document.data), (err: unknown) => err instanceof ClientError && err.status === 409);
  assert.equal((await client.getDocument(documentId)).document.revision, revision + 1);
});

test('Cover: rejected input — a malformed step is 400 at submit, lyrics-less vocal render fails at run', async () => {
  await assert.rejects(client.submitWorkflowJob('cover-open', { assetId: 'not-a-uuid' }), (err: unknown) => err instanceof ClientError && err.status === 400);
  await assert.rejects(client.submitWorkflowJob('cover-transcribe', { documentId: 'x', revision: 0 }), (err: unknown) => err instanceof ClientError && err.status === 400);
  const { documentId, revision } = await openDraft();
  const vocal = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-render',
    renderInput(documentId, revision, 'x', { instrumental: false, lyrics: '  ' }))).id);
  assert.equal(vocal.status, 'failed');
  assert.match(JSON.stringify(vocal.error), /Enter lyrics or enable Instrumental mode/);
});

test('Cover: cancelling a running caption step ends it cancelled and leaves the draft unchanged', async () => {
  const { documentId, revision } = await openDraft();
  holdCaption = true;
  try {
    const job = await client.submitWorkflowJob('cover-caption', { documentId, revision, artistId: 1, provider: 'fixture', model: '', force: true });
    for (let i = 0; i < 50 && (await client.getWorkflowJob(job.id)).status !== 'running'; i++) await new Promise(r => setTimeout(r, 100));
    await client.cancelWorkflowJob(job.id);
    const done = await client.waitForWorkflowJob(job.id);
    assert.equal(done.status, 'cancelled');
  } finally { holdCaption = false; }
  assert.equal((await client.getDocument(documentId)).document.revision, revision);
});

test('Cover: auth failure — submitting without a token is 401', async () => {
  const res = await fetch(`${server.origin}/api/workflows/jobs`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: 'cover-open', idempotencyKey: 'x', input: { assetId: '00000000-0000-4000-8000-000000000000' } }) });
  assert.equal(res.status, 401);
});

test('Safety: no real subprocess, LLM, transcriber or off-origin call', () => {
  assert.deepEqual(server.violations, []);
});
