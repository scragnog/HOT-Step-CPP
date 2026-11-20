// recordingLink.test.ts — a Lyric Studio render is listed under its written
// song whichever audio attempt succeeds, and a render that never reached the
// audio queue is flagged `unqueued` so a client can show it as a failed row.
// Real lireek DB, real audio intent queue and workflow runner; fake engine jobs.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

test('render fails, is retried from the queue and succeeds: the song lists the recording', async () => {
  process.env.DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'recording-link-'));
  const database = await import('../../db/database.js');
  const rows = await import('../../db/lireekDb.js');
  const { captureRenderItems, lyricBatchInput, runLyricBatch } = await import('./lyricWorkflow.js');
  const { linkRenderedRecording } = await import('./recordingLink.js');
  const { WorkflowJobs } = await import('../workflows/workflowJobs.js');
  const { AudioIntentQueue } = await import('../audioQueue/intentQueue.js');
  database.initDb();
  const artist = rows.getOrCreateArtist('Fixture Artist');
  const set = rows.saveLyricsSet(artist.id as number, 'Fixture Album', 1, [{ title: 'Fixture Song', lyrics: 'A verse' }]);
  const profile = rows.saveProfile(set.id as number, 'fixture-provider', 'fixture-model', { artist: 'Fixture Artist' });
  const gen = rows.saveGeneration({ profileId: profile.id as number, provider: 'fixture-provider', model: 'm', lyrics: 'A verse', title: 'Fixture Song' });

  const engineJobs = new Map<string, { status: string; error?: string; result?: unknown }>();
  let next = 0;
  const queue = new AudioIntentQueue({
    db: database.getDb(),
    submit: async () => { const jobId = `job-${++next}`; engineJobs.set(jobId, { status: 'running' }); return { ok: true, jobId }; },
    getJob: id => engineJobs.get(id),
    cancelJob: () => false,
    activeEngine: () => 'ace',
    isEngine: id => id === 'ace',
    onSucceeded: linkRenderedRecording,
  });
  const workflows = new WorkflowJobs({ db: database.getDb(), audioPollMs: 5, audio: {
    enqueue: (input, userId) => queue.enqueue(input, userId), get: id => queue.get(id), cancel: id => queue.cancel(id),
  } });
  workflows.register({ kind: 'lyric-render-test', input: lyricBatchInput, run: runLyricBatch });
  const intent = { kind: 'written-song' as const, engine: 'ace', generationId: gen.id as number, lyricsSetId: set.id as number, params: {}, settings: {} };
  const settle = async () => { for (let i = 0; i < 5; i++) { await queue.tick(); await new Promise(r => setTimeout(r, 15)); } };

  const job = workflows.submit({ kind: 'lyric-render-test', idempotencyKey: 'retry', input: await captureRenderItems([intent]) }, 'user').job;
  for (let i = 0; i < 100 && !workflows.get(job.id).audioIntentIds.length; i++) await new Promise(r => setTimeout(r, 5));
  const audioId = workflows.get(job.id).audioIntentIds[0];
  assert.equal((queue.get(audioId).meta as { lyricLink?: unknown }).lyricLink !== undefined, true);
  await settle();
  engineJobs.set('job-1', { status: 'failed', error: 'engine fell over' });
  await settle();
  await workflows.settled();
  const failed = (workflows.get(job.id).result as { results: Array<{ status: string; unqueued?: boolean }> }).results[0];
  assert.deepEqual([failed.status, failed.unqueued], ['error', undefined], 'the audio row shows this failure itself');
  assert.equal(rows.getAudioGenerations(gen.id as number).length, 0);

  // Retry from the queue: the workflow has finished, the link must not depend on it.
  queue.retry(audioId);
  await settle();
  engineJobs.set('job-2', { status: 'succeeded', result: { audioUrls: ['/audio/retried.wav'] } });
  await settle();
  const listed = rows.getAudioGenerations(gen.id as number);
  assert.deepEqual(listed.map(r => [r.hotstep_job_id, r.audio_url]), [['job-2', '/audio/retried.wav']]);

  // Failures before the audio queue are flagged, and keep the intent for Retry.
  const stale = await captureRenderItems([intent]);
  rows.updateGenerationFields(gen.id as number, { title: 'Renamed after capture' });
  const missing = await captureRenderItems([{ ...intent, generationId: 999_999 }]);
  assert.equal(missing.items[0].type, 'preflight-error');
  assert.equal((missing.items[0] as { intent?: unknown }).intent !== undefined, true);
  const batch = workflows.submit({ kind: 'lyric-render-test', idempotencyKey: 'unqueued',
    input: { items: [...stale.items, ...missing.items] } }, 'user').job;
  await workflows.settled();
  const results = (workflows.get(batch.id).result as { results: Array<{ error?: string; unqueued?: boolean }> }).results;
  assert.match(results[0].error ?? '', /changed since submission/);
  assert.deepEqual(results.map(r => r.unqueued), [true, true]);
  assert.equal(workflows.get(batch.id).audioIntentIds.length, 0);
});
