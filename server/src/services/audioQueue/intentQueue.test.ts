import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import express from 'express';
import type { Server } from 'node:http';
import { AudioIntentQueue, AudioQueueError, MAX_IN_FLIGHT, type AudioQueueDeps } from './intentQueue.js';
import { createAudioQueueRouter } from '../../routes/audioQueue.js';

type Job = { status: string; error?: string; result?: unknown };

/** A fake generation queue: records submits, holds jobs, can hold a submit open. */
function harness(db = new Database(':memory:')) {
  const submits: Array<{ body: Record<string, unknown>; userId: string }> = [];
  const cancels: string[] = [];
  const jobs = new Map<string, Job>();
  let active = 'yue2';
  let next = 0;
  let gate: Promise<void> | null = null;
  let reply: ((body: Record<string, unknown>) => { ok: true; jobId: string } | { ok: false; status: number; body: Record<string, unknown> }) | null = null;
  const deps: AudioQueueDeps = {
    db,
    submit: async (body, userId) => {
      submits.push({ body, userId });
      if (gate) await gate;
      if (reply) return reply(body);
      const jobId = `job-${++next}`;
      jobs.set(jobId, { status: 'pending' });
      return { ok: true, jobId };
    },
    getJob: id => jobs.get(id),
    cancelJob: id => { cancels.push(id); const j = jobs.get(id); if (!j) return false; j.status = 'cancelled'; return true; },
    activeEngine: () => active,
    isEngine: id => ['ace', 'minimax-m3', 'yue2'].includes(id),
    now: () => 1000 + submits.length,
  };
  return {
    db, deps, submits, cancels, jobs,
    queue: new AudioIntentQueue(deps),
    setActive: (id: string) => { active = id; },
    hold: () => { let open!: () => void; gate = new Promise(r => { open = r; }); return () => { gate = null; open(); }; },
    setReply: (fn: typeof reply) => { reply = fn; },
  };
}

const req = (extra: Record<string, unknown> = {}) => ({ caption: 'c', lyrics: 'l', seed: 1, expectedBackend: 'yue2', ...extra });

test('double submit with one key queues and renders once; another request under it is refused', async () => {
  const h = harness();
  const a = h.queue.enqueue({ idempotencyKey: 'song-1', request: req() }, 'u1');
  const b = h.queue.enqueue({ idempotencyKey: 'song-1', request: req() }, 'u2');
  assert.equal(a.created, true);
  assert.equal(b.created, false);
  assert.equal(b.item.id, a.item.id);
  assert.throws(() => h.queue.enqueue({ idempotencyKey: 'song-1', request: req({ caption: 'other' }) }, 'u1'),
    (e: unknown) => e instanceof AudioQueueError && e.status === 409);
  await h.queue.tick();
  await h.queue.tick();
  assert.equal(h.submits.length, 1);
  assert.equal(h.queue.list().length, 1);
});

test('two executors on one database never submit the same item twice', async () => {
  const db = new Database(':memory:');
  const one = harness(db);
  const two = harness(db);
  for (let i = 0; i < 3; i++) one.queue.enqueue({ idempotencyKey: `k${i}`, request: req({ seed: i }) }, 'u');
  const release = one.hold();
  const release2 = two.hold();
  const p = Promise.all([one.queue.tick(), two.queue.tick()]);
  release(); release2();
  await p;
  const seeds = [...one.submits, ...two.submits].map(s => s.body.seed).sort();
  assert.deepEqual(seeds, [0, 1, 2]);
});

test('restart: pending stays pending, submitting and submitted become interrupted, terminal untouched, nothing resubmitted', async () => {
  const db = new Database(':memory:');
  const before = harness(db);
  const ids = ['pending', 'submitting', 'submitted', 'succeeded', 'failed', 'cancelled'].map((s, i) => {
    const { item } = before.queue.enqueue({ idempotencyKey: s, request: req({ seed: i }) }, 'u');
    if (s !== 'pending') db.prepare('UPDATE audio_intents SET status = ?, job_id = ? WHERE id = ?').run(s, s === 'submitting' ? null : `old-${s}`, item.id);
    return [s, item.id] as const;
  });
  // New process: the old jobs map is gone.
  const after = harness(db);
  assert.equal(after.queue.reconcileAfterRestart(), 2);
  const status = Object.fromEntries(ids.map(([s, id]) => [s, after.queue.get(id).status]));
  assert.deepEqual(status, { pending: 'pending', submitting: 'interrupted', submitted: 'interrupted', succeeded: 'succeeded', failed: 'failed', cancelled: 'cancelled' });
  assert.match(after.queue.get(ids[2][1]).error!, /old-submitted/);
  assert.match(after.queue.get(ids[1][1]).error!, /may or may not have reached/);
  await after.queue.tick();
  await after.queue.tick();
  assert.deepEqual(after.submits.map(s => s.body.seed), [0]);   // only the never-sent item
});

test('cancel during submit: the job is cancelled as soon as the submit returns', async () => {
  const h = harness();
  const { item } = h.queue.enqueue({ idempotencyKey: 'k', request: req() }, 'u');
  const release = h.hold();
  const pass = h.queue.tick();
  await new Promise(r => setImmediate(r));
  assert.equal(h.queue.get(item.id).status, 'submitting');
  assert.equal(h.queue.cancel(item.id).cancelRequested, true);
  release();
  await pass;
  const done = h.queue.get(item.id);
  assert.equal(done.status, 'cancelled');
  assert.equal(done.jobId, 'job-1');
  assert.deepEqual(h.cancels, ['job-1']);
});

test('cancel during a submit that fails ends cancelled, not failed', async () => {
  const h = harness();
  const { item } = h.queue.enqueue({ idempotencyKey: 'k', request: req() }, 'u');
  h.setReply(() => ({ ok: false, status: 400, body: { error: 'bad' } }));
  const release = h.hold();
  const pass = h.queue.tick();
  await new Promise(r => setImmediate(r));
  h.queue.cancel(item.id);
  release();
  await pass;
  assert.equal(h.queue.get(item.id).status, 'cancelled');
});

test('cancel: pending stops at once; submitted cancels its job; terminal is refused', async () => {
  const h = harness();
  const p = h.queue.enqueue({ idempotencyKey: 'p', request: req() }, 'u').item;
  assert.equal(h.queue.cancel(p.id).status, 'cancelled');
  const s = h.queue.enqueue({ idempotencyKey: 's', request: req() }, 'u').item;
  await h.queue.tick();
  assert.equal(h.queue.get(s.id).status, 'submitted');
  assert.equal(h.queue.cancel(s.id).status, 'cancelled');
  assert.deepEqual(h.cancels, ['job-1']);
  assert.throws(() => h.queue.cancel(s.id), (e: unknown) => e instanceof AudioQueueError && e.status === 409);
  assert.equal(h.submits.length, 1);
});

test('retry renders a failed item again as a new attempt and keeps the old job id', async () => {
  const h = harness();
  const { item } = h.queue.enqueue({ idempotencyKey: 'k', request: req() }, 'u');
  await h.queue.tick();
  h.jobs.get('job-1')!.status = 'failed';
  h.jobs.get('job-1')!.error = 'engine crashed';
  await h.queue.tick();
  assert.equal(h.queue.get(item.id).status, 'failed');
  assert.equal(h.queue.get(item.id).error, 'engine crashed');
  const again = h.queue.retry(item.id);
  assert.equal(again.status, 'pending');
  assert.equal(again.attempt, 2);
  assert.deepEqual(again.previousJobIds, ['job-1']);
  await h.queue.tick();
  assert.equal(h.queue.get(item.id).jobId, 'job-2');
  assert.equal(h.submits.length, 2);
  assert.throws(() => h.queue.retry(item.id), (e: unknown) => e instanceof AudioQueueError && e.status === 409);
});

test('retry is also how an interrupted item renders again, and only then', async () => {
  const db = new Database(':memory:');
  const first = harness(db);
  const { item } = first.queue.enqueue({ idempotencyKey: 'k', request: req() }, 'u');
  await first.queue.tick();
  const second = harness(db);
  second.queue.reconcileAfterRestart();
  await second.queue.tick();
  assert.equal(second.submits.length, 0);
  second.queue.retry(item.id);
  await second.queue.tick();
  assert.equal(second.submits.length, 1);
  assert.deepEqual(second.queue.get(item.id).previousJobIds, ['job-1']);
});

test('backend switch between enqueue and run: the item waits for its engine and runs its snapshot unchanged', async () => {
  const h = harness();
  const request = req({ caption: 'resolved for yue2' });
  const { item } = h.queue.enqueue({ idempotencyKey: 'k', request }, 'u');
  h.setActive('ace');
  await h.queue.tick();
  assert.equal(h.submits.length, 0);
  assert.match(h.queue.get(item.id).waiting!, /yue2 engine; ace is active/);
  h.setActive('yue2');
  await h.queue.tick();
  assert.deepEqual(h.submits[0].body, request);
  assert.equal(h.queue.get(item.id).waiting, null);
});

test('a switch racing the submit (409) puts the item back to wait; nothing ran', async () => {
  const h = harness();
  const { item } = h.queue.enqueue({ idempotencyKey: 'k', request: req() }, 'u');
  h.setReply(() => ({ ok: false, status: 409, body: { error: "Expected backend 'yue2' but the active backend is 'ace'" } }));
  await h.queue.tick();
  const it = h.queue.get(item.id);
  assert.equal(it.status, 'pending');
  assert.match(it.waiting!, /Expected backend/);
  assert.equal(it.jobId, null);
});

test('a request with no engine is pinned to the one active at enqueue', () => {
  const h = harness();
  h.setActive('minimax-m3');
  const { item } = h.queue.enqueue({ idempotencyKey: 'k', request: { caption: 'c' } }, 'u');
  assert.equal(item.engine, 'minimax-m3');
  assert.equal(item.request.expectedBackend, 'minimax-m3');
  assert.throws(() => h.queue.enqueue({ idempotencyKey: 'x', request: { expectedBackend: 'nope' } }, 'u'),
    (e: unknown) => e instanceof AudioQueueError && e.status === 400);
});

test('submitted items follow their jobs: success with result, failure with error, a lost job surfaces', async () => {
  const h = harness();
  const a = h.queue.enqueue({ idempotencyKey: 'a', request: req({ seed: 1 }) }, 'u').item;
  const b = h.queue.enqueue({ idempotencyKey: 'b', request: req({ seed: 2 }) }, 'u').item;
  const c = h.queue.enqueue({ idempotencyKey: 'c', request: req({ seed: 3 }) }, 'u').item;
  await h.queue.tick();
  h.jobs.set('job-1', { status: 'succeeded', result: { audioUrls: ['/audio/x.wav'], songIds: ['s'] } });
  h.jobs.set('job-2', { status: 'running' });
  h.jobs.delete('job-3');
  await h.queue.tick();
  assert.equal(h.queue.get(a.id).status, 'succeeded');
  assert.deepEqual(h.queue.get(a.id).result, { audioUrls: ['/audio/x.wav'], songIds: ['s'] });
  assert.equal(h.queue.get(b.id).status, 'submitted');
  assert.equal(h.queue.get(c.id).status, 'interrupted');
  assert.equal(h.submits.length, 3);
});

test('pause stops submissions but keeps following; resume continues; state persists', async () => {
  const h = harness();
  const a = h.queue.enqueue({ idempotencyKey: 'a', request: req({ seed: 1 }) }, 'u').item;
  await h.queue.tick();
  h.queue.setPaused(true);
  h.queue.enqueue({ idempotencyKey: 'b', request: req({ seed: 2 }) }, 'u');
  h.jobs.get('job-1')!.status = 'succeeded';
  await h.queue.tick();
  assert.equal(h.submits.length, 1);
  assert.equal(h.queue.get(a.id).status, 'succeeded');
  assert.equal(new AudioIntentQueue(h.deps).state().paused, true);
  h.queue.setPaused(false);
  await h.queue.tick();
  assert.equal(h.submits.length, 2);
});

test('a pause that lands while a submit is held open stops the rest of that pass', async () => {
  const h = harness();
  const items = [1, 2, 3].map(seed => h.queue.enqueue({ idempotencyKey: `k${seed}`, request: req({ seed }) }, 'u').item);
  const release = h.hold();
  const pass = h.queue.tick();
  await new Promise(r => setImmediate(r));
  assert.equal(h.submits.length, 1);           // the first submit is in flight
  h.queue.setPaused(true);                     // /pause completes now
  release();
  await pass;
  assert.deepEqual(h.submits.map(s => s.body.seed), [1]);
  assert.equal(h.queue.get(items[0].id).status, 'submitted');
  assert.deepEqual([items[1], items[2]].map(i => h.queue.get(i.id).status), ['pending', 'pending']);
  await h.queue.tick();
  assert.equal(h.submits.length, 1);           // still paused
  h.queue.setPaused(false);
  await h.queue.tick();
  assert.deepEqual(h.submits.map(s => s.body.seed), [1, 2, 3]);
});

test('at most MAX_IN_FLIGHT are submitted at once; engine not ready stops the pass', async () => {
  const h = harness();
  for (let i = 0; i < MAX_IN_FLIGHT + 2; i++) h.queue.enqueue({ idempotencyKey: `k${i}`, request: req({ seed: i }) }, 'u');
  await h.queue.tick();
  assert.equal(h.submits.length, MAX_IN_FLIGHT);
  for (const id of h.jobs.keys()) h.jobs.get(id)!.status = 'succeeded';
  h.setReply(() => ({ ok: false, status: 503, body: { error: 'Engine not ready' } }));
  await h.queue.tick();
  assert.equal(h.submits.length, MAX_IN_FLIGHT + 1);    // one try, then the pass stops
  assert.equal(h.queue.state().counts.pending, 2);
});

test('route: auth, validation, created vs existing, unknown item', async () => {
  const h = harness();
  const app = express();
  app.use(express.json());
  app.use('/api/audio-queue', createAudioQueueRouter(h.queue, r => (r.headers.authorization === 'Bearer good' ? 'u' : null)));
  const server: Server = await new Promise(resolve => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); });
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}/api/audio-queue`;
  const call = (method: string, path: string, body?: unknown, token = 'good') => fetch(base + path, {
    method, headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` }, body: body ? JSON.stringify(body) : undefined });
  try {
    assert.equal((await call('GET', '/state', undefined, 'stale')).status, 401);
    assert.equal((await call('POST', '/items', { request: {} })).status, 400);
    const first = await call('POST', '/items', { idempotencyKey: 'r1', request: req() });
    assert.equal(first.status, 201);
    const again = await call('POST', '/items', { idempotencyKey: 'r1', request: req() });
    assert.equal(again.status, 200);
    assert.equal(((await again.json()) as { created: boolean }).created, false);
    assert.equal((await call('GET', '/items/nope')).status, 404);
    assert.equal((await call('GET', '/items?status=bogus')).status, 400);
    assert.equal(((await (await call('POST', '/pause')).json()) as { paused: boolean }).paused, true);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});
