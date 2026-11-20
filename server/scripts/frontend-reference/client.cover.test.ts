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
    // The YuE2 cases below queue a YuE2 request while ACE is the active backend.
    capability: async () => true,
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

// ── The three client-built bodies that moved to the server ────────────────
//
// The fake engine cannot run a YuE2 render, so these read the request the
// workflow captured on the audio queue (the job's audio intent) instead of
// waiting for audio. That request is exactly what the bundled UI used to build.

const SCORE_ABC = 'X:1\nM:4/4\nL:1/8\nQ:1/4=120\nK:C\n% verse\nV: Vocal\nC8|D8|\n';
const yue2Controls = { ...controls, pairMode: 'base', captionMode: 'custom', tempoMode: 'set', coverBpm: 100, keyShift: 2, keepChords: true };
const queuedRequest = async (jobId: string) => {
  let job = await client.getWorkflowJob(jobId);
  for (let i = 0; i < 100 && !(job as unknown as { audioIntentIds: string[] }).audioIntentIds.length
    && !['failed', 'cancelled', 'succeeded'].includes(job.status); i++) {
    await new Promise(r => setTimeout(r, 100));
    job = await client.getWorkflowJob(jobId);
  }
  const intent = (job as unknown as { audioIntentIds: string[] }).audioIntentIds[0];
  assert.ok(intent, `the job queued audio (${job.status}: ${JSON.stringify(job.error)})`);
  const request = (await client.audioQueueItem(intent)).item.request as Record<string, any>;
  await client.cancelWorkflowJob(jobId); // the render itself is not under test
  return request;
};
const postJson = (path: string, body: unknown) => fetch(`${server.origin}${path}`, { method: 'POST',
  headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${client.token}` }, body: JSON.stringify(body) });

test('Cover: a score-only draft (MusicXML) renders a YuE2 cover with no recording, and no other step takes it', async () => {
  const open = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-open', { score: { abc: `  ${SCORE_ABC}` } })).id);
  assert.equal(open.status, 'succeeded', JSON.stringify(open.error));
  const { documentId, revision } = open.result as { documentId: string; revision: number };
  const { document } = await client.getDocument(documentId);
  assert.deepEqual([(document.data as any).scoreOnly, (document.data as any).approvedAbc], [true, SCORE_ABC.trim()]);

  await assert.rejects(client.submitWorkflowJob('cover-open', { assetId: '00000000-0000-4000-8000-000000000000', score: { abc: 'x' } }),
    (err: unknown) => err instanceof ClientError && err.status === 400);
  const transcribe = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-transcribe', { documentId, revision, force: false })).id);
  assert.equal(transcribe.status, 'failed');
  assert.match(JSON.stringify(transcribe.error), /no asset id/);
  const ace = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-render', renderInput(documentId, revision, 'x'))).id);
  assert.equal(ace.status, 'failed');
  assert.match(JSON.stringify(ace.error), /YuE2 only/);

  const request = await queuedRequest((await client.submitWorkflowJob('cover-render', renderInput(documentId, revision, 'dusty funk',
    { expectedBackend: 'yue2', controls: yue2Controls, targetArtistName: 'Fixture Band' }))).id);
  assert.equal(request.backend, 'yue2');
  assert.equal(request.sourceAudioUrl, undefined, 'no recording');
  assert.equal(request.yue2Abc, SCORE_ABC.trim());
  assert.deepEqual([request.yue2Cover.sourceKind, request.yue2Cover.tempo, request.yue2Cover.keyShift, request.yue2Cover.keepChords],
    ['musicxml', 100, 2, true]);
  assert.deepEqual([request.artist, request.artistName, request.source], ['Fixture Band', 'Fixture Band', 'cover-studio']);
});

test("Cover: an ACE cover mixes the separation's stems on the server, and only for ACE", async () => {
  const { documentId, revision, asset } = await openDraft();
  const { id: separationId } = await client.supersepSeparate(asset.audio_url, 2);
  const stems = { separationId, controls: [{ index: 0, volume: 1, muted: false }, { index: 1, volume: 0.5, muted: true }] };

  const request = await queuedRequest((await client.submitWorkflowJob('cover-render', renderInput(documentId, revision, 'x', { stems }))).id);
  assert.match(String(request.sourceAudioUrl), /^\/references\/[0-9a-f-]+\.wav$/);
  assert.notEqual(request.sourceAudioUrl, asset.audio_url, "the mix replaces the draft's own audio");
  assert.equal(request.taskType, 'cover');

  const unknown = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-render',
    renderInput(documentId, revision, 'x', { stems: { ...stems, separationId: 'nope' } }))).id);
  assert.equal(unknown.status, 'failed');
  assert.match(JSON.stringify(unknown.error), /Stem separation not found/);

  const yue2 = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-render',
    renderInput(documentId, revision, 'x', { stems, expectedBackend: 'yue2', controls: yue2Controls }))).id);
  assert.equal(yue2.status, 'failed');
  assert.match(JSON.stringify(yue2.error), /ACE covers only/);

  // The YuE2 transcription path mixes through its own step and gets an asset to open a draft on.
  const mixed = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-stems', stems)).id);
  assert.equal(mixed.status, 'succeeded', JSON.stringify(mixed.error));
  const { assetId, audioUrl } = mixed.result as { assetId: string; audioUrl: string };
  assert.match(audioUrl, /^\/references\//);
  const reopened = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-open', { assetId })).id);
  assert.equal(reopened.status, 'succeeded', JSON.stringify(reopened.error));
});

test('Cover: Build starts from a draft reference and saves through the server', async () => {
  const open = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-open', { score: { abc: SCORE_ABC } })).id);
  const { documentId, revision } = open.result as { documentId: string; revision: number };
  const cover = renderInput(documentId, revision, 'dusty funk', { expectedBackend: 'yue2', controls: yue2Controls });

  // The body the UI used to build is gone: an empty request and a non-YuE2 cover are refused.
  assert.equal((await postJson('/api/yue2-cover/build', {})).status, 400);
  const wrong = await postJson('/api/yue2-cover/build', { cover: { ...cover, expectedBackend: 'ace' } });
  assert.equal(wrong.status, 400);
  assert.match(((await wrong.json()) as { error: string }).error, /YuE2 backend/);
  assert.equal((await postJson('/api/yue2-cover/build', { cover: { ...cover, revision: revision + 5 } })).status, 409);

  assert.equal((await client.setActiveBackend('yue2')).activeId, 'yue2');
  try {
    const started = await postJson('/api/yue2-cover/build', { cover });
    assert.equal(started.status, 200, await started.clone().text());
    const session = await started.json() as { id: string; params: Record<string, any>; kept: unknown[] };
    assert.equal(session.params.yue2Abc, SCORE_ABC.trim());
    assert.equal(session.params.yue2Cover.sourceKind, 'musicxml');
    assert.equal(session.params.sourceAudioUrl, undefined);

    // Save song takes the session id only; with nothing kept it refuses.
    const empty = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-build-save', { sessionId: session.id })).id);
    assert.equal(empty.status, 'failed');
    assert.match(JSON.stringify(empty.error), /Keep at least one take/);
    const missing = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-build-save',
      { sessionId: '00000000-0000-4000-8000-000000000000' })).id);
    assert.equal(missing.status, 'failed');
  } finally { await client.setActiveBackend('ace'); }
});

test('Safety: no real subprocess, LLM, transcriber or off-origin call', () => {
  assert.deepEqual(server.violations, []);
});
