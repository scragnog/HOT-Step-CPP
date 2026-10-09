// client.lyric.test.ts — Lyric Studio batch (lyric-batch), the audit group
// dropped from the original 7f-1/7f-2/7f-3 split (Lead's correction after
// the Batch 7 gate). Documented sequence: docs/dev/frontend-studios.md's
// "Lyric Studio batch" section.
//
// BLOCKED, not faked: three of the four item types reach a real external
// dependency with no injectable seam.
//   - generate/refine (lyricWorkflow.ts:186-199): llmService.generateLyrics-
//     Streaming/refineLyricsStreaming call llm/registry.ts's getProvider()
//     then the provider's own real HTTP call (Ollama, OpenAI, ...). No deps
//     parameter on runLyricBatch/execute to inject a fixture provider.
//   - fetch (lyricWorkflow.ts:153-154): genius.fetchLyrics/getArtistImageUrl/
//     getAlbumImageUrl are real calls to api.genius.com, no deps parameter.
//   - generate and refine additionally need a pre-existing Profile/Generation
//     row, which in production only exists after a prior successful generate/
//     refine/profile step — so even their preflight (capture) path can only
//     be reached with a real id, which this harness has no way to create
//     without the same blocked call.
//   Proposed seam, same shape as coverWorkflow.ts's overrideCoverWorkflowDeps
//   and labelingQueue.ts's overrideTrainingRunner: an optional
//   `overrideLyricWorkflowDeps({ llm, genius, profiler })` on
//   lyricWorkflow.ts, defaulted to the real llmService/geniusService/
//   profilerService calls.
//
// profile is partially real: profilerService.buildProfile()'s first line is
// `llmService.getProvider(providerName).defaultModel` (profilerService.ts
// :712), which throws `Unknown LLM provider: <name>` before any analysis or
// network call for a provider name outside the fixed registry
// (llm/registry.ts's `providers` map). lyricItemBase's `provider` field is
// `z.string().min(1)` (studioWorkflows.ts:177) — no enum — so a deliberately
// unknown provider name is valid input, not a validation rejection, and the
// job really runs profilerService.buildProfile() against a real LyricsSet
// before failing for a real, reproducible reason. That is genuine coverage
// of capture -> submit -> execute -> per-item-error -> job result, not a
// workaround standing in for a result this harness cannot produce.
//
// render needs a Generation row (via WrittenSongIntent's generationId), so
// it is blocked transitively by the same generate/refine gap.

import test from 'node:test';
import assert from 'node:assert/strict';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient, ClientError } from './client.js';

let server: FakeServer;
let client: ReferenceClient;

test.before(async () => {
  server = await startFakeServer();
  client = new ReferenceClient(server.origin);
  await client.login();
});
test.after(() => server.close());

type LyricResult = { results: Array<{ index: number; status: 'done' | 'error'; value?: unknown; error?: string }> };

test('Lyric Studio: auth failure — no bearer token is 401 on both capture routes', async () => {
  const batches = await fetch(`${server.origin}/api/lireek/workflow-batches`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ items: [] }),
  });
  assert.equal(batches.status, 401);
  const renders = await fetch(`${server.origin}/api/lireek/workflow-renders`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ intents: [] }),
  });
  assert.equal(renders.status, 401);
});

test('Lyric Studio: rejected input — items must be an array', async () => {
  const res = await fetch(`${server.origin}/api/lireek/workflow-batches`, {
    method: 'POST', headers: { Authorization: `Bearer ${client.token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ items: 'not-an-array' }),
  });
  assert.equal(res.status, 400);
});

test('Lyric Studio: per-item preflight errors stay ordered by index, never reaching execution', async () => {
  // Neither item resolves to a real source, so captureLyricItems turns both
  // into `preflight-error` items (lyricWorkflow.ts:98-101) before the job
  // ever runs — no LLM/Genius call for either.
  const job = await client.waitForWorkflowJob((await client.submitLyricBatch([
    { type: 'profile', targetId: 999999, provider: 'fixture' },
    { type: 'generate', targetId: 999999, provider: 'fixture', count: 999 },
  ])).id);
  assert.equal(job.status, 'succeeded');
  const { results } = job.result as LyricResult;
  assert.equal(results.length, 2);
  assert.equal(results[0].index, 0);
  assert.equal(results[0].status, 'error');
  assert.match(results[0].error!, /not found/);
  assert.equal(results[1].index, 1);
  assert.equal(results[1].status, 'error');
  // The source-id check (lyricWorkflow.ts:81-82) runs before the count
  // check, so a nonexistent targetId reports "not found" even though
  // count=999 is independently invalid too (lyricWorkflow.ts:95-96).
  assert.match(results[1].error!, /not found/);
});

test('Lyric Studio: a real profile item fails for a real reason — an unknown LLM provider, before any network call', async () => {
  const { lyrics_set } = await client.createLyricsSet({
    artist_name: 'Reference Fixture Artist', album: 'Reference Fixture Album',
    songs: [{ title: 'Track 1', album: 'Reference Fixture Album', lyrics: '[Verse]\nfixture line' }],
  });
  const job = await client.waitForWorkflowJob((await client.submitLyricBatch([
    { type: 'profile', targetId: (lyrics_set as { id: number }).id, provider: 'not-a-real-provider' },
  ])).id);
  assert.equal(job.status, 'succeeded');
  const { results } = job.result as LyricResult;
  assert.equal(results.length, 1);
  assert.equal(results[0].status, 'error');
  assert.match(results[0].error!, /Unknown LLM provider/);
});

test('Lyric Studio: an idempotency key reused with a different batch is 409', async () => {
  const key = `lyric-${Date.now()}`;
  await client.submitLyricBatch([{ type: 'profile', targetId: 1, provider: 'fixture' }], key);
  await assert.rejects(
    () => client.submitLyricBatch([{ type: 'profile', targetId: 2, provider: 'fixture' }], key),
    (err: unknown) => err instanceof ClientError && err.status === 409,
  );
});

test('Lyric Studio: cancelling an already-finished job is 409; cancelling an unknown job is 404', async () => {
  const job = await client.waitForWorkflowJob((await client.submitLyricBatch([
    { type: 'profile', targetId: 999999, provider: 'fixture' },
  ])).id);
  assert.equal(job.status, 'succeeded');
  await assert.rejects(
    () => client.cancelWorkflowJob(job.id),
    (err: unknown) => err instanceof ClientError && err.status === 409,
  );
  await assert.rejects(
    () => client.cancelWorkflowJob('00000000-0000-0000-0000-000000000000'),
    (err: unknown) => err instanceof ClientError && err.status === 404,
  );
});

test('Safety: no real subprocess exec or off-origin network call occurred', () => {
  assert.deepEqual(server.violations, []);
});
