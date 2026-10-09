import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { WorkflowJobs, type WorkflowDeps } from './workflowJobs.js';
import { WorkflowDocuments } from './revisions.js';
import { createInstaGenKinds, effectiveInstaRequest, type InstaInput, type InstaResult } from './instaGenWorkflow.js';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';

const input: InstaInput = {
  caption: 'Punk Rock', genres: ['Punk Rock'], lyricMode: 'lyrics', subject: '', randomSubject: false,
  provider: '', model: '', vocalLanguage: 'en', thinking: true,
  engineParams: { seed: 42, source: 'wrong-source', bpm: 88 }, expectedBackend: 'ace',
  coResident: false, cacheLmCodes: true,
};
const result: InstaResult = {
  caption: 'Fast punk rock', lyrics: '[Verse 1]\nOld street\n\n[Chorus]\nRun tonight',
  bpm: 140, duration: 190, keyScale: 'D minor', timeSignature: '4', vocalLanguage: 'en',
};

function setup(inspire: (signal: AbortSignal) => Promise<InstaResult> = async () => result, audioStatus: 'pending' | 'succeeded' = 'succeeded') {
  const db = new Database(':memory:');
  const documents = new WorkflowDocuments(db);
  const items = new Map<string, AudioIntentItem>();
  let submitted = 0;
  const audio: WorkflowDeps['audio'] = {
    enqueue: ({ idempotencyKey, request, meta }) => {
      const known = [...items.values()].find(i => i.idempotencyKey === idempotencyKey);
      if (known) return { item: known, created: false };
      const item = { id: `audio-${++submitted}`, idempotencyKey, request, meta: meta || null,
        status: audioStatus, result: { audioUrls: ['/audio/test.wav'], songIds: ['song-1'] }, error: null,
        engine: 'ace', jobId: 'job-1', attempt: 1, previousJobIds: [], waiting: null, cancelRequested: false,
        createdAt: 1, updatedAt: 1 } as AudioIntentItem;
      items.set(item.id, item);
      return { item, created: true };
    },
    get: id => items.get(id)!,
    cancel: id => { const item = items.get(id)!; item.status = 'cancelled'; return item; },
  };
  const jobs = new WorkflowJobs({ db, audio, audioPollMs: 1 });
  let active = 'ace';
  let calls = 0;
  const kinds = createInstaGenKinds({
    documents,
    checkBackend: async captured => { if (captured.expectedBackend !== active) throw new Error('backend changed'); return true; },
    inspire: async (_params, signal) => { calls++; return inspire(signal); },
  });
  for (const kind of kinds) jobs.register(kind);
  const submit = (kind: string, key: string, body: Record<string, unknown>, user = 'u') =>
    jobs.submit({ kind, idempotencyKey: key, input: body }, user);
  return { db, documents, jobs, items, submit, get submitted() { return submitted; }, get calls() { return calls; }, switchBackend: (id: string) => { active = id; } };
}

test('direct and approved preview send the same captured request and one audio item each', async () => {
  const s = setup();
  const direct = s.submit('insta-direct', 'direct', input as unknown as Record<string, unknown>);
  const preview = s.submit('insta-preview', 'preview', input as unknown as Record<string, unknown>);
  await s.jobs.settled();
  const previewResult = s.jobs.get(preview.job.id).result as { documentId: string; revision: number };
  const approved = s.submit('insta-approve', 'approve', previewResult);
  await s.jobs.settled();
  assert.equal(s.jobs.get(approved.job.id).status, 'succeeded');
  assert.deepEqual([...s.items.values()].map(i => i.request), [
    (s.jobs.get(direct.job.id).result as any).request,
    (s.jobs.get(approved.job.id).result as any).request,
  ]);
  assert.deepEqual([...s.items.values()][0].request, [...s.items.values()][1].request);
  assert.equal([...s.items.values()][0].request.title, 'Run tonight');
  assert.equal([...s.items.values()][0].request.source, 'insta-gen');
  assert.equal([...s.items.values()][0].request.expectedBackend, 'ace');
});

test('instrumental and missing metadata keep old request defaults', () => {
  const instrumental = { ...input, lyricMode: 'instrumental' as const };
  const request = effectiveInstaRequest(instrumental, { caption: '', lyrics: '[Instrumental]', vocalLanguage: 'en' });
  assert.equal(request.instrumental, true);
  assert.equal(request.vocalLanguage, undefined);
  assert.equal(request.title, 'Punk Rock');
  assert.equal(request.caption, 'Punk Rock');
  assert.equal(request.bpm, 88, 'missing inspire BPM leaves captured global BPM');
  assert.equal(request.duration, undefined);
  assert.equal(request.source, 'insta-gen');
  const noDuration = effectiveInstaRequest(input, result, undefined, false);
  assert.equal(noDuration.duration, undefined, 'a backend without editable duration never gets the estimate');
});

test('a user-set duration wins over the LM estimate; auto (-1) still falls back to it', () => {
  const userSet = { ...input, engineParams: { ...input.engineParams, duration: 10 } };
  assert.equal(effectiveInstaRequest(userSet, result).duration, 10, 'Custom-Gen duration must not be overwritten by the LM result');
  const auto = { ...input, engineParams: { ...input.engineParams, duration: -1 } };
  assert.equal(effectiveInstaRequest(auto, result).duration, 190, 'the -1 auto sentinel still defers to the LM estimate');
  assert.equal(effectiveInstaRequest(input, result).duration, 190, 'no engineParams duration also falls back to the LM estimate');
});

test('approval uses its document revision; edits and backend changes cannot silently alter an accepted render', async () => {
  const s = setup();
  const preview = s.submit('insta-preview', 'p', input as unknown as Record<string, unknown>);
  await s.jobs.settled();
  const { documentId, revision } = s.jobs.get(preview.job.id).result as { documentId: string; revision: number };
  const old = s.documents.get(documentId, 'u');
  s.documents.update(documentId, 'u', revision, { ...old.data, edits: { lyrics: 'Edited', caption: 'Edited caption' } });
  const stale = s.submit('insta-approve', 'stale', { documentId, revision });
  await s.jobs.settled();
  assert.equal(s.jobs.get(stale.job.id).status, 'failed');
  assert.equal(s.submitted, 0);
  s.switchBackend('minimax-m3');
  const changed = s.submit('insta-approve', 'changed', { documentId, revision: revision + 1 });
  await s.jobs.settled();
  assert.equal(s.jobs.get(changed.job.id).status, 'failed');
  assert.equal(s.submitted, 0);
});

test('the GET -> PUT edits -> approve sequence renders the edited lyrics, not the original preview', async () => {
  const s = setup();
  const preview = s.submit('insta-preview', 'edit-preview', input as unknown as Record<string, unknown>);
  await s.jobs.settled();
  const { documentId, revision } = s.jobs.get(preview.job.id).result as { documentId: string; revision: number };
  // GET: the client reads the full stored body, not just `result`.
  const fetched = s.documents.get(documentId, 'u');
  assert.deepEqual(fetched.data.edits, { lyrics: result.lyrics, caption: result.caption }, 'edits starts as a copy of result');
  // PUT: resend input/result unchanged, only edits differs; data is replaced wholesale.
  const edited = s.documents.update(documentId, 'u', revision, {
    ...fetched.data as object, edits: { lyrics: '[Verse 1]\nNew street\n\n[Chorus]\nEdited tonight', caption: 'Edited caption' },
  });
  const approved = s.submit('insta-approve', 'edit-approve', { documentId, revision: edited.revision });
  await s.jobs.settled();
  assert.equal(s.jobs.get(approved.job.id).status, 'succeeded');
  const request = (s.jobs.get(approved.job.id).result as any).request;
  assert.equal(request.lyrics, '[Verse 1]\nNew street\n\n[Chorus]\nEdited tonight');
  assert.equal(request.title, 'Edited tonight');
});

test('idempotency, cancel, restart interruption and cursor replay hold for Insta-Gen', async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const s = setup(async signal => { await gate; if (signal.aborted) throw new Error('cancelled'); return result; });
  const first = s.submit('insta-direct', 'same', input as unknown as Record<string, unknown>);
  assert.equal(s.submit('insta-direct', 'same', input as unknown as Record<string, unknown>).job.id, first.job.id);
  assert.throws(() => s.submit('insta-direct', 'same', { ...input, caption: 'Changed' }));
  const waiting = s.submit('insta-direct', 'waiting', input as unknown as Record<string, unknown>);
  assert.equal(s.jobs.cancel(waiting.job.id).status, 'cancelled');
  const before = s.jobs.replay(first.job.id, 'u', 0);
  assert.ok(before.events.length >= 2);
  const cancelled = s.jobs.cancel(first.job.id);
  assert.equal(cancelled.cancelRequested, true);
  release();
  await s.jobs.settled();
  assert.equal(s.jobs.get(first.job.id).status, 'cancelled');
  assert.equal(s.submitted, 0);
  const after = s.jobs.replay(first.job.id, 'u', before.events.at(-1)!.seq);
  assert.equal(after.gap, false);
  assert.ok(after.events.some(e => e.type === 'cancel-requested'));

  // A second process sees a running job as interrupted, with no automatic run.
  const stillRunning = setup(async () => new Promise<InstaResult>(() => {}));
  const job = stillRunning.submit('insta-direct', 'restart', input as unknown as Record<string, unknown>).job;
  const replacement = new WorkflowJobs({ db: stillRunning.db, audio: {
    enqueue: () => { throw new Error('must not run'); },
    get: () => { throw new Error('must not run'); },
    cancel: () => { throw new Error('must not run'); },
  } });
  replacement.reconcileAfterRestart();
  assert.equal(replacement.get(job.id).status, 'interrupted');
  assert.ok(replacement.replay(job.id, 'u', 0).events.some(e => (e.data as any)?.status === 'interrupted'));
});
test('cancelling a direct render cancels its queued audio and discards its result', async () => {
  const s = setup(undefined, 'pending');
  const job = s.submit('insta-direct', 'audio-cancel', input as unknown as Record<string, unknown>).job;
  for (let i = 0; i < 50 && s.items.size === 0; i++) await new Promise(r => setTimeout(r, 2));
  assert.equal(s.items.size, 1);
  const ack = s.jobs.cancel(job.id);
  assert.equal(ack.cancelRequested, true);
  await s.jobs.settled();
  assert.equal([...s.items.values()][0].status, 'cancelled');
  assert.equal(s.jobs.get(job.id).status, 'cancelled');
  assert.equal(s.jobs.get(job.id).result, null);
});
