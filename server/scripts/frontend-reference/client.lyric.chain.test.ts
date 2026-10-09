// client.lyric.chain.test.ts — Lyric Studio's full chain through the
// documented sequence: fetch an album, profile it, generate lyrics, refine
// one, run a mixed batch, then render a written song; plus a stale source
// revision, cancel-and-retry, and per-item failure.
//
// Genius, the profiler's LLM pass and the lyric LLM are fixtures, set
// through lyricWorkflow.ts's overrideLyricWorkflowDeps. That replaces only
// those dependencies, never a route: every request below goes through the
// real /api/lireek capture routes, the real lyric-batch workflow, the real
// Lyric Studio rows and the real audio queue, with the fake engine standing
// in for ace-server on the render.

import test from 'node:test';
import assert from 'node:assert/strict';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient } from './client.js';
import type { WorkflowJob } from '../../src/contracts/workflow.js';

type Results = { results: Array<{ index: number; status: 'done' | 'error'; value?: any; error?: string }> };

let server: FakeServer;
let client: ReferenceClient;
let clearOverrides: () => void;
const calls = { fetch: 0, profile: 0, generate: 0, refine: 0 };
const historySeen: string[][] = [];
/** Set to make the next refine wait until released (stale-revision test). */
let refineGate: { reached: () => void; release: Promise<void> } | null = null;
/** Set to make generate wait until its job is cancelled (retry test). */
let holdGenerate = false;

test.before(async () => {
  server = await startFakeServer();
  const { overrideLyricWorkflowDeps } = await import('../../src/services/lireek/lyricWorkflow.js');
  clearOverrides = overrideLyricWorkflowDeps({
    fetchLyrics: async (artist, album) => {
      calls.fetch++;
      if (artist === 'Unknown Band') throw new Error('Genius: artist not found');
      return { artist, album: album ?? null, total_songs: 2, songs: [
        { title: 'First Song', album: album ?? null, lyrics: '[Verse]\nfixture line one\nfixture line two' },
        { title: 'Second Song', album: album ?? null, lyrics: '[Chorus]\nfixture chorus' },
      ] };
    },
    getArtistImageUrl: async () => null,
    getAlbumImageUrl: async () => null,
    buildProfile: async (artist) => {
      calls.profile++;
      return { artist, album: null, themes: ['fixtures'], tone_and_mood: 'calm', style_caption: 'fixture style' } as never;
    },
    generateLyricsStreaming: async (_profile, provider, model, _extra, usedSubjects, _bpms, _keys, _titles, _durations, onChunk) => {
      calls.generate++;
      historySeen.push([...(usedSubjects ?? [])]);
      if (holdGenerate) await new Promise(() => { /* until the job is cancelled */ });
      onChunk?.('[Verse]\n');
      const n = calls.generate;
      return { lyrics: `[Verse]\ngenerated ${n}`, provider, model: model || 'fixture-model', title: `Generated ${n}`, subject: `subject ${n}`,
        bpm: 100 + n, key: 'C major', caption: 'fixture caption', caption_mm3: '', caption_yue2: '', duration: 120,
        system_prompt: 'sys', user_prompt: 'user' };
    },
    refineLyricsStreaming: async (lyrics, _artist, title, provider, model) => {
      calls.refine++;
      if (refineGate) { refineGate.reached(); await refineGate.release; }
      return { lyrics: `${lyrics}\n(refined)`, provider, model: model || 'fixture-model', title: `${title} (refined)`, subject: '',
        bpm: 0, key: '', caption: '', caption_mm3: '', duration: 0, system_prompt: 'sys', user_prompt: 'user' };
    },
  });
  client = new ReferenceClient(server.origin);
  await client.login();
});
test.after(async () => { clearOverrides?.(); await server.close(); });

async function batch(items: Array<Record<string, unknown>>): Promise<{ job: WorkflowJob; out: Results }> {
  const job = await client.waitForWorkflowJob((await client.submitLyricBatch(items)).id);
  return { job, out: job.result as Results };
}
const one = (out: Results) => { assert.equal(out.results[0].status, 'done', JSON.stringify(out.results[0])); return out.results[0].value; };

let lyricsSetId = 0, profileId = 0, generationId = 0;

test('Lyric Studio: fetch → profile → generate ×2 → refine, each on the row the last step saved', async () => {
  const fetched = one((await batch([{ type: 'fetch', artist: 'Fixture Artist', album: 'Fixture Album', maxSongs: 5 }])).out);
  assert.equal(fetched.songs_fetched, 2);
  lyricsSetId = fetched.lyrics_set_id;

  profileId = one((await batch([{ type: 'profile', targetId: lyricsSetId, provider: 'fixture' }])).out).id;
  assert.ok(profileId > 0);

  const generated = await batch([{ type: 'generate', targetId: profileId, provider: 'fixture', count: 2 }]);
  assert.equal(generated.job.status, 'succeeded');
  assert.deepEqual(generated.out.results.map(r => r.status), ['done', 'done']);
  const [first, second] = generated.out.results.map(r => r.value);
  assert.notEqual(first.id, second.id);
  // The second generate of the batch is told what the first one wrote.
  assert.ok(historySeen.at(-1)!.includes(first.subject), JSON.stringify(historySeen));
  generationId = first.id;

  const refined = one((await batch([{ type: 'refine', targetId: generationId, provider: 'fixture' }])).out);
  assert.notEqual(refined.id, generationId, 'a refinement is a new generation');
  assert.deepEqual(calls, { fetch: 1, profile: 1, generate: 2, refine: 1 });
});

test('Lyric Studio: a mixed batch keeps per-item failures to their own items, in order', async () => {
  const { job, out } = await batch([
    { type: 'generate', targetId: profileId, provider: 'fixture' },
    { type: 'refine', targetId: 999999, provider: 'fixture' },
    { type: 'fetch', artist: 'Unknown Band' },
    { type: 'profile', targetId: lyricsSetId, provider: 'fixture' },
  ]);
  assert.equal(job.status, 'succeeded', 'per-item errors do not fail the job');
  assert.deepEqual(out.results.map(r => [r.index, r.status]), [[0, 'done'], [1, 'error'], [2, 'error'], [3, 'done']]);
  assert.match(out.results[1].error!, /Generation 999999 not found/);
  assert.match(out.results[2].error!, /artist not found/);
});

test('Lyric Studio: a source edited mid-step fails that item as stale', async () => {
  let reached!: () => void;
  const atRefine = new Promise<void>(resolve => { reached = resolve; });
  let release!: () => void;
  refineGate = { reached, release: new Promise<void>(resolve => { release = resolve; }) };
  try {
    const submitted = await client.submitLyricBatch([{ type: 'refine', targetId: generationId, provider: 'fixture' }]);
    await atRefine;
    // Another client edits the generation the refine was captured from.
    const edit = await fetch(`${server.origin}/api/lireek/generations/${generationId}`, { method: 'PATCH',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: 'Edited elsewhere' }) });
    assert.equal(edit.status, 200, await edit.text());
    release();
    const job = await client.waitForWorkflowJob(submitted.id);
    const r = (job.result as Results).results[0];
    assert.deepEqual([r.status, r.error], ['error', 'Source changed during refinement']);
  } finally { refineGate = null; }
});

test('Lyric Studio: a cancelled batch retried reuses its finished items and runs the rest', async () => {
  const before = calls.generate;
  // Item 0 finishes; item 1 holds until the job is cancelled.
  holdGenerate = true;
  const submitted = await client.submitLyricBatch([
    { type: 'fetch', artist: 'Fixture Artist', album: 'Retry Album' },
    { type: 'generate', targetId: profileId, provider: 'fixture' },
  ]);
  for (let i = 0; i < 100 && calls.generate === before; i++) await new Promise(r => setTimeout(r, 50));
  const generated = calls.generate;
  assert.equal(generated, before + 1, 'the generate item is running');
  await client.cancelWorkflowJob(submitted.id);
  const cancelled = await client.waitForWorkflowJob(submitted.id);
  assert.equal(cancelled.status, 'cancelled');
  holdGenerate = false;
  const fetchesBefore = calls.fetch;
  await client.retryWorkflowJob(submitted.id);
  const retried = await client.waitForWorkflowJob(submitted.id);
  assert.equal(retried.status, 'succeeded');
  assert.deepEqual((retried.result as Results).results.map(r => r.status), ['done', 'done']);
  assert.equal(calls.fetch, fetchesBefore, 'the finished fetch is reused, not refetched');
  assert.equal(calls.generate, generated + 1, 'only the unfinished generate runs again');
});

test('Lyric Studio: render a written song through the audio queue; a bad intent is a per-item error', async () => {
  const res = await fetch(`${server.origin}/api/lireek/workflow-renders`, { method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${client.token}` },
    body: JSON.stringify({ intents: [
      { kind: 'written-song', engine: 'ace', generationId, lyricsSetId, params: { skipLm: true, seed: 5, randomSeed: false }, settings: {} },
      { kind: 'written-song', engine: 'ace', generationId: 999999, lyricsSetId, params: {}, settings: {} },
    ] }) });
  assert.equal(res.status, 201, await res.clone().text());
  const { job: submitted } = await res.json() as { job: WorkflowJob };
  const job = await client.waitForWorkflowJob(submitted.id, 30_000);
  assert.equal(job.status, 'succeeded', JSON.stringify(job.error));
  const [rendered, bad] = (job.result as Results).results;
  assert.equal(rendered.status, 'done', JSON.stringify(rendered));
  assert.deepEqual([rendered.value.generationId, typeof rendered.value.jobId], [generationId, 'string']);
  assert.equal(bad.status, 'error');
  const status = await client.generationStatus(rendered.value.jobId);
  assert.equal(status.status, 'succeeded', 'the render ran on the fake engine');
});

test('Safety: no real subprocess, LLM, Genius or off-origin call', () => {
  assert.deepEqual(server.violations, []);
});
