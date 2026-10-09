// client.test.ts — the reference client exercising Create, Insta-Gen,
// Cover, Repaint, Lego and stem separation against the REAL production
// routes/services, mounted by fakeServer.ts with a fake engine standing in
// for ace-server. One fakeServer per file (module-level singletons like
// generate.ts's in-memory job map are shared across every request it
// receives, same as the real app), cleaned up in test.after.

import test from 'node:test';
import assert from 'node:assert/strict';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient, ClientError } from './client.js';
import { makeFixtureWav } from './fixtures.js';

let server: FakeServer;
let client: ReferenceClient;

test.before(async () => {
  server = await startFakeServer();
  client = new ReferenceClient(server.origin);
  await client.login();
});
test.after(() => server.close());

// ── Create (text2music) ───────────────────────────────────────────────────

test('Create: submit, poll, succeeded — real audio URL and song row', async () => {
  const { jobId } = await client.submitGeneration({
    caption: 'a fake test caption', lyrics: '[Instrumental]', instrumental: true,
    seed: 1, randomSeed: false, skipLm: true,
  });
  const status = await client.waitForGeneration(jobId);
  assert.equal(status.status, 'succeeded');
  const result = status.result as { audioUrls: string[]; songIds: string[] };
  assert.ok(result.audioUrls?.[0]);
  assert.ok(result.songIds?.[0]);
});

test('Create: cancellation ends the job cancelled, not succeeded', async () => {
  const { jobId } = await client.submitGeneration({
    caption: 'cancel me', lyrics: '[Instrumental]', instrumental: true, seed: 2, randomSeed: false, skipLm: true,
  });
  await client.cancelGeneration(jobId);
  const status = await client.waitForGeneration(jobId);
  assert.equal(status.status, 'cancelled');
});

test('Create: auth failure — no bearer token is 401 before any job runs', async () => {
  const res = await fetch(`${server.origin}/api/generate`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ caption: 'x' }),
  });
  assert.equal(res.status, 401);
});

// ── Resolve → durable audio queue (the resolved Create path, not the legacy
// direct-submit one above) ──────────────────────────────────────────────

test('Resolve: preview then submit through the durable audio queue, not legacy direct-submit', async () => {
  const { request } = await client.resolvePreview({
    kind: 'create',
    params: { caption: 'a resolved caption', lyrics: '[Instrumental]', instrumental: true, seed: 7, randomSeed: false, skipLm: true },
  });
  assert.equal(request.expectedBackend, 'ace');
  const { item } = await client.enqueueAudioIntent(request);
  const finished = await client.waitForAudioIntent(String((item as { id: string }).id));
  assert.equal(finished.status, 'succeeded');
  assert.ok(finished.jobId);
});

// ── Insta-Gen (insta-preview, insta-approve, insta-direct) ────────────────

const instaInput = () => ({
  caption: 'a fake insta caption', genres: [], lyricMode: 'instrumental',
  subject: '', randomSubject: true, provider: '', model: '',
  vocalLanguage: 'en', thinking: false, engineParams: { skipLm: true },
  expectedBackend: 'ace', coResident: false, cacheLmCodes: false,
});

test('Insta-Gen: preview creates a document; approve renders the edited lyrics, not the original', async () => {
  const preview = await client.submitWorkflowJob('insta-preview', instaInput());
  const previewJob = await client.waitForWorkflowJob(preview.id);
  assert.equal(previewJob.status, 'succeeded');
  const { documentId, revision, result } = previewJob.result as { documentId: string; revision: number; result: { caption: string; lyrics: string } };
  assert.equal(result.lyrics, '[Instrumental]');

  // Edit before approving — this is the whole point of the edits/result split.
  const putRes = await client.putDocument(documentId, revision, {
    input: instaInput(), result, edits: { lyrics: '[Instrumental]', caption: 'an edited insta caption' },
  });

  const approve = await client.submitWorkflowJob('insta-approve', { documentId, revision: putRes.document.revision });
  const approveJob = await client.waitForWorkflowJob(approve.id);
  assert.equal(approveJob.status, 'succeeded');
  const rendered = approveJob.result as { request: Record<string, unknown>; audio: unknown };
  assert.equal(rendered.request.caption, 'an edited insta caption');
});

test('Insta-Gen: insta-direct resolves and renders in one job', async () => {
  const direct = await client.submitWorkflowJob('insta-direct', instaInput());
  const job = await client.waitForWorkflowJob(direct.id);
  assert.equal(job.status, 'succeeded');
  assert.ok((job.result as { audioIntentId: string }).audioIntentId);
});

test('Insta-Gen: rejected input — an empty caption is 400 before any job is created', async () => {
  const res = await fetch(`${server.origin}/api/workflows/jobs`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${client.token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: 'insta-preview', idempotencyKey: 'bad-insta-1', input: { ...instaInput(), caption: '' } }),
  });
  assert.equal(res.status, 400);
  const body = await res.json() as { issues: Array<{ path: string[] }> };
  assert.ok(body.issues.some(i => i.path.includes('caption')));
});

test('Insta-Gen: a stale document revision on approve fails the job, not the submit', async () => {
  const preview = await client.submitWorkflowJob('insta-preview', instaInput());
  const previewJob = await client.waitForWorkflowJob(preview.id);
  const { documentId, revision } = previewJob.result as { documentId: string; revision: number };
  // Submit accepts it (the schema only checks shape); the mismatch surfaces
  // as the job's own failure once `run()` compares it to the document's
  // actual revision — same distinction the shared envelope docs draw
  // between a submit-time 400 and a job-time error.
  const approve = await client.submitWorkflowJob('insta-approve', { documentId, revision: revision + 1 });
  const job = await client.waitForWorkflowJob(approve.id);
  assert.equal(job.status, 'failed');
  assert.match(job.error ?? '', /Stale preview revision/);
});

// ── Cover (cover-open) ─────────────────────────────────────────────────────
//
// cover-caption and cover-render are BLOCKED for this harness, not isolated:
// coverWorkflow.ts's defaultCaption (coverWorkflow.ts:167) unconditionally
// calls getProvider(provider).call() — a real LLM provider — with no
// availability guard (unlike analyzeWithEssentia's essentiaAvailable()
// check, which is why cover-open below is isolated: fakeServer.ts points
// HOT_STEP_ROOT at an empty temp dir, so the essentia binary path it checks
// never resolves regardless of what happens to be installed on the host
// running the suite — availability is not isolation on its own, and
// safetyGuards.ts's subprocess guard is the backstop if that check is ever
// changed). cover-transcribe's
// defaultTranscribe (coverWorkflow.ts:137) calls the real yue2CoverService,
// the same YuE2 worker 7f-3 (training/streaming) is fixturing — not mine to
// fake here. Neither has a dependency seam reachable from this harness:
// yue2Cover.ts:182 calls registerCoverWorkflows() with no deps at import
// time, and WorkflowJobs.register() throws on re-registering a kind, so
// there is no way to swap in a fake caption/transcribe after that import
// without a production change (e.g. an exported mutable deps holder, or an
// env-gated opt-out of the import-time auto-register). Reported to Lead
// rather than faking a 'succeeded' HTTP response for either step.

test('Cover: open — real metadata/key-BPM analysis, no LLM or remote worker', async () => {
  const asset = await client.uploadAudio(makeFixtureWav(2));
  const open = await client.waitForWorkflowJob((await client.submitWorkflowJob('cover-open', { assetId: asset.asset_id })).id);
  assert.equal(open.status, 'succeeded');
  const openResult = open.result as { documentId: string; revision: number };
  assert.ok(openResult.documentId);
});

// ── Repaint and Lego (layer-render) ────────────────────────────────────────

test('Repaint: render a region of an uploaded source asset', async () => {
  const asset = await client.uploadAudio(makeFixtureWav(2));
  const job = await client.waitForWorkflowJob((await client.submitWorkflowJob('repaint-render', {
    source: { kind: 'asset', id: asset.asset_id, expectedUrl: asset.audio_url },
    expectedBackend: 'ace', engineParams: { skipLm: true },
    regionStart: 0, regionEnd: 1, lyrics: '', styleCaption: 'fake repaint style', sourceName: 'fixture',
    repaintMode: 'balanced', crossfadeFrames: 0,
  })).id);
  assert.equal(job.status, 'succeeded');
  assert.ok((job.result as { audioIntentId: string }).audioIntentId);
});

test('Repaint: rejected input — region end before start is 400 at submit', async () => {
  const asset = await client.uploadAudio(makeFixtureWav(2));
  const res = await fetch(`${server.origin}/api/workflows/jobs`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${client.token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      kind: 'repaint-render', idempotencyKey: 'bad-repaint-1',
      input: {
        source: { kind: 'asset', id: asset.asset_id, expectedUrl: asset.audio_url },
        expectedBackend: 'ace', engineParams: {},
        regionStart: 0, regionEnd: -1, lyrics: '', styleCaption: '', sourceName: '',
        repaintMode: 'balanced', crossfadeFrames: 0,
      },
    }),
  });
  assert.equal(res.status, 400);
});

test('Lego (layer-render): isolate one stem track into its own render', async () => {
  const asset = await client.uploadAudio(makeFixtureWav(2));
  const job = await client.waitForWorkflowJob((await client.submitWorkflowJob('layer-render', {
    source: { kind: 'asset', id: asset.asset_id, expectedUrl: asset.audio_url },
    expectedBackend: 'ace', engineParams: { skipLm: true },
    trackName: 'vocals', buildModel: 'acestep-v15-base-fixture', caption: 'fake vocal layer',
  })).id);
  assert.equal(job.status, 'succeeded');
  assert.ok((job.result as { audioIntentId: string }).audioIntentId);
});

// ── Stem separation (Stem Studio + SuperSep proxy) ─────────────────────────

test('Stem Studio extract: one track, sequential DiT render, polled to done', async () => {
  const asset = await client.uploadAudio(makeFixtureWav(2));
  const { id } = await client.stemExtract({ sourceAudioUrl: asset.audio_url, sourceFileName: asset.filename, tracks: ['vocals'] });
  const progress = await client.waitForStemJob(id);
  assert.equal(progress.status, 'done');
  const result = await client.stemResult(id);
  assert.equal((result.stems as unknown[]).length, 1);
});

test('Stem Studio extract: rejected input — unknown track name is 400', async () => {
  const asset = await client.uploadAudio(makeFixtureWav(2));
  await assert.rejects(
    () => client.stemExtract({ sourceAudioUrl: asset.audio_url, tracks: ['not-a-real-track'] }),
    (err: unknown) => err instanceof ClientError && err.status === 400,
  );
});

test('Stem Studio supersep: separation runs in-process and saves stems', async () => {
  const asset = await client.uploadAudio(makeFixtureWav(2));
  const { id } = await client.stemSupersep({ sourceAudioUrl: asset.audio_url, sourceFileName: asset.filename, level: 0 });
  const progress = await client.waitForStemJob(id);
  assert.equal(progress.status, 'done');
  const result = await client.stemResult(id);
  assert.ok((result.stems as unknown[]).length > 0);
});

test('SuperSep proxy: separate, progress, result, release — thin pass-through to the engine', async () => {
  const asset = await client.uploadAudio(makeFixtureWav(2));
  const { id } = await client.supersepSeparate(asset.audio_url, 0);
  const progress = await client.supersepProgress(id);
  assert.equal(progress.status, 'done');
  const result = await client.supersepResult(id);
  assert.ok(result.stems.length > 0);
  const release = await client.supersepRelease(id);
  assert.equal((release as { ok: boolean }).ok, true);
});

// ── Safety ──────────────────────────────────────────────────────────────
//
// Last, so it covers every test above it. safetyGuards.ts records a
// violation even when production code catches the guard's thrown error and
// degrades gracefully (e.g. essentiaClient.ts treating any failure as a null
// result) — this assertion is what actually fails the suite in that case.

test('Safety: no real subprocess exec or off-origin network call occurred', () => {
  assert.deepEqual(server.violations, []);
});
