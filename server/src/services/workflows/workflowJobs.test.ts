import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { z } from 'zod/v4';
import { WorkflowError, WorkflowJobs, type WorkflowDeps, type WorkflowKind } from './workflowJobs.js';
import { WorkflowDocuments, bumpRevision } from './revisions.js';
import { startWorkflowTestServer } from './testServer.js';
import { MAX_EVENTS_PER_JOB } from '../../contracts/workflow.js';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';

/** A fake audio intent queue keyed like the real one. */
function fakeAudio() {
  const items = new Map<string, AudioIntentItem>();
  const byKey = new Map<string, string>();
  const enqueued: string[] = [];
  const audio: WorkflowDeps['audio'] = {
    enqueue: input => {
      const known = byKey.get(input.idempotencyKey);
      if (known) return { item: items.get(known)!, created: false };
      const id = `intent-${items.size + 1}`;
      const item = { id, idempotencyKey: input.idempotencyKey, status: 'pending', request: input.request, meta: input.meta ?? null } as AudioIntentItem;
      items.set(id, item); byKey.set(input.idempotencyKey, id); enqueued.push(input.idempotencyKey);
      return { item, created: true };
    },
    get: id => items.get(id)!,
    cancel: id => { const i = items.get(id)!; i.status = 'cancelled'; return i; },
  };
  return { audio, items, enqueued, finish: (id: string, status: AudioIntentItem['status']) => { items.get(id)!.status = status; } };
}

/** A gate a test opens to let a step continue. */
function gate() {
  let open!: () => void;
  const opened = new Promise<void>(r => { open = r; });
  return { opened, open };
}

function setup(db = new Database(':memory:'), audio = fakeAudio()) {
  let t = 1000;
  const jobs = new WorkflowJobs({ db, audio: audio.audio, now: () => ++t, audioPollMs: 5 });
  return { db, audio, jobs };
}

const echo = (over: Partial<WorkflowKind<{ text: string }>> = {}): WorkflowKind<{ text: string }> => ({
  kind: 'echo', input: z.object({ text: z.string().min(1) }),
  run: async ctx => ({ said: ctx.input.text }),
  ...over,
});

const types = (jobs: WorkflowJobs, id: string) => jobs.replay(id, undefined).events.map(e => e.type === 'status' ? `status:${(e.data as any).status}` : e.type);

test('idempotency is scoped to user, kind and key, and a changed input is a conflict', async () => {
  const { jobs } = setup();
  let runs = 0;
  jobs.register(echo({ run: async ctx => { runs++; return ctx.input.text; } }));
  jobs.register({ ...echo(), kind: 'other' });
  const first = jobs.submit({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } }, 'u1');
  assert.equal(first.created, true);
  const again = jobs.submit({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } }, 'u1');
  assert.deepEqual([again.created, again.job.id], [false, first.job.id]);
  assert.throws(() => jobs.submit({ kind: 'echo', idempotencyKey: 'k', input: { text: 'b' } }, 'u1'),
    (e: unknown) => e instanceof WorkflowError && e.status === 409);
  assert.notEqual(jobs.submit({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } }, 'u2').job.id, first.job.id);
  assert.notEqual(jobs.submit({ kind: 'other', idempotencyKey: 'k', input: { text: 'a' } }, 'u1').job.id, first.job.id);
  await jobs.settled();
  assert.equal(runs, 2);
  assert.equal(jobs.get(first.job.id).status, 'succeeded');
  assert.equal(jobs.get(first.job.id).result, 'a');
  // The input is validated, and only the parsed value is captured.
  assert.throws(() => jobs.submit({ kind: 'echo', idempotencyKey: 'bad', input: { text: '' } }, 'u1'),
    (e: unknown) => e instanceof WorkflowError && e.status === 400);
  assert.throws(() => jobs.submit({ kind: 'missing', idempotencyKey: 'x', input: {} }, 'u1'),
    (e: unknown) => e instanceof WorkflowError && e.status === 400);
  const stripped = jobs.submit({ kind: 'echo', idempotencyKey: 'extra', input: { text: 'a', stray: 1 } }, 'u1');
  assert.deepEqual(stripped.job.input, { text: 'a' });
  await jobs.settled();
});

test('kinds run one at a time by default, oldest first, and a step failure fails only its job', async () => {
  const { jobs } = setup();
  const order: string[] = [];
  const g = gate();
  jobs.register(echo({ run: async ctx => { order.push(ctx.input.text); if (ctx.input.text === 'a') await g.opened; if (ctx.input.text === 'b') throw new Error('boom'); return 1; } }));
  const a = jobs.submit({ kind: 'echo', idempotencyKey: 'a', input: { text: 'a' } }, 'u').job;
  const b = jobs.submit({ kind: 'echo', idempotencyKey: 'b', input: { text: 'b' } }, 'u').job;
  const c = jobs.submit({ kind: 'echo', idempotencyKey: 'c', input: { text: 'c' } }, 'u').job;
  assert.deepEqual([jobs.get(a.id).status, jobs.get(b.id).status], ['running', 'pending']);
  g.open();
  await jobs.settled();
  assert.deepEqual(order, ['a', 'b', 'c']);
  assert.deepEqual([a, b, c].map(j => jobs.get(j.id).status), ['succeeded', 'failed', 'succeeded']);
  assert.equal(jobs.get(b.id).error, 'boom');
});

test('cancel: pending never runs; running is acknowledged, aborted and ends cancelled; finished refuses', async () => {
  const { jobs, audio } = setup();
  const g = gate();
  let sawAbort = false;
  let ran: string[] = [];
  jobs.register(echo({
    run: async ctx => {
      ran.push(ctx.input.text);
      ctx.audio.enqueue('take-1', { caption: ctx.input.text });
      ctx.signal.addEventListener('abort', () => { sawAbort = true; });
      await g.opened;
      ctx.throwIfCancelled();
      return 'done';
    },
  }));
  const running = jobs.submit({ kind: 'echo', idempotencyKey: 'r', input: { text: 'r' } }, 'u').job;
  const queued = jobs.submit({ kind: 'echo', idempotencyKey: 'q', input: { text: 'q' } }, 'u').job;
  assert.equal(jobs.cancel(queued.id).status, 'cancelled');
  const ack = jobs.cancel(running.id);
  assert.deepEqual([ack.status, ack.cancelRequested], ['running', true]);
  assert.equal(sawAbort, true);
  assert.equal(audio.items.get(ack.audioIntentIds[0])!.status, 'cancelled', 'its audio item is cancelled too');
  assert.deepEqual(jobs.cancel(running.id).cancelRequested, true, 'a repeated cancel is a no-op');
  g.open();
  await jobs.settled();
  assert.equal(jobs.get(running.id).status, 'cancelled');
  assert.deepEqual(ran, ['r']);
  assert.deepEqual(types(jobs, running.id), ['status:pending', 'status:running', 'audio', 'cancel-requested', 'status:cancelled']);
  assert.throws(() => jobs.cancel(running.id), (e: unknown) => e instanceof WorkflowError && e.status === 409);

  // Completion first: a cancel after success is refused, and the result stands.
  ran = [];
  const quick = jobs.submit({ kind: 'echo', idempotencyKey: 'quick', input: { text: 'x' } }, 'u').job;
  g.open();
  await jobs.settled();
  assert.equal(jobs.get(quick.id).status, 'succeeded');
  assert.throws(() => jobs.cancel(quick.id), (e: unknown) => e instanceof WorkflowError && e.status === 409);
});

test('cancel racing completion: a step that ignores the signal and returns still ends cancelled, result discarded', async () => {
  const { jobs } = setup();
  const g = gate();
  jobs.register(echo({ run: async () => { await g.opened; return 'late result'; } }));
  const job = jobs.submit({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } }, 'u').job;
  jobs.cancel(job.id);
  g.open();
  await jobs.settled();
  const after = jobs.get(job.id);
  assert.deepEqual([after.status, after.result], ['cancelled', null]);
});

test('a timeout ends the run even when the step ignores the signal; its late result and writes are discarded', async () => {
  const { jobs } = setup();
  const late: string[] = [];
  jobs.register(echo({
    timeoutMs: 5,
    run: async ctx => {
      if (ctx.input.text === 'forever') return new Promise(() => {});
      if (ctx.input.text === 'quick') return 'fast';
      await new Promise(r => setTimeout(r, 30));
      try { ctx.emit('late', {}); } catch (err) { late.push((err as Error).message); }
      return 'late result';
    },
  }));
  const slow = jobs.submit({ kind: 'echo', idempotencyKey: 'slow', input: { text: 'slow' } }, 'u').job;
  const stuck = jobs.submit({ kind: 'echo', idempotencyKey: 'stuck', input: { text: 'forever' } }, 'u').job;
  await jobs.settled();
  for (const id of [slow.id, stuck.id]) {
    assert.equal(jobs.get(id).status, 'failed');
    assert.match(jobs.get(id).error ?? '', /Timed out/);
  }
  await new Promise(r => setTimeout(r, 50));
  assert.deepEqual([jobs.get(slow.id).status, jobs.get(slow.id).result], ['failed', null], 'the late return changed nothing');
  assert.equal(late.length, 1, 'the late emit was refused');
  assert.ok(!types(jobs, slow.id).includes('late'));
  // The never-settling step does not hold its slot: the next job runs.
  const next = jobs.submit({ kind: 'echo', idempotencyKey: 'next', input: { text: 'quick' } }, 'u').job;
  await jobs.settled();
  assert.equal(jobs.get(next.id).status, 'succeeded');
});

test('restart then retry before the old step ends: only the new attempt can write or finish', async () => {
  const db = new Database(':memory:');
  const audio = fakeAudio();
  const oldStep = gate();
  const newStep = gate();
  const refused: string[] = [];
  const kind = (which: 'OLD' | 'NEW', g: ReturnType<typeof gate>) => echo({
    run: async ctx => {
      await g.opened;
      try { ctx.emit('progress', { from: which }); } catch (err) { refused.push(`${which}: ${(err as Error).message}`); }
      return which;
    },
  });
  const before = setup(db, audio);
  before.jobs.register(kind('OLD', oldStep));
  const job = before.jobs.submit({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } }, 'u').job;
  assert.equal(before.jobs.get(job.id).status, 'running');

  const after = setup(db, audio);
  after.jobs.reconcileAfterRestart();
  after.jobs.register(kind('NEW', newStep));
  assert.equal(after.jobs.retry(job.id).status, 'running', 'attempt 2 starts at once');

  oldStep.open();
  await before.jobs.settled();
  assert.equal(after.jobs.get(job.id).status, 'running', 'the old step finishing did not end attempt 2');
  newStep.open();
  await after.jobs.settled();
  const done = after.jobs.get(job.id);
  assert.deepEqual([done.status, done.attempt, done.result], ['succeeded', 2, 'NEW']);
  const progress = after.jobs.replay(job.id, undefined).events.filter(e => e.type === 'progress').map(e => (e.data as { from: string }).from);
  assert.deepEqual(progress, ['NEW']);
  assert.equal(refused.length, 1);
  assert.match(refused[0], /^OLD: /);
});

test('HTTP: a retried job replays from 0 through its current end, not the first attempt\'s', async () => {
  const { jobs } = setup();
  jobs.register(echo({ run: async ctx => { if (ctx.attempt === 1) throw new Error('first try fails'); return 'ok'; } }));
  const server = await startWorkflowTestServer({ jobs });
  try {
    const { job } = jobs.submit({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } }, 'u');
    await jobs.settled();
    assert.equal(jobs.get(job.id).status, 'failed');
    jobs.retry(job.id);
    await jobs.settled();
    const frames = await readFrames(`${server.origin}/api/workflows/jobs/${job.id}/events?after=0`, { Authorization: 'Bearer t' }, () => false);
    assert.equal(frames[0].job.status, 'succeeded');
    const statuses = frames.filter(f => f.event?.type === 'status').map(f => f.event.data.status);
    assert.deepEqual(statuses, ['pending', 'running', 'failed', 'pending', 'running', 'succeeded']);
    assert.equal(frames.at(-1).id, jobs.get(job.id).lastSeq);
  } finally {
    await server.close();
  }
});

test('restart: a running job becomes interrupted and is not rerun; pending starts; retry reruns and reuses its audio', async () => {
  const db = new Database(':memory:');
  const audio = fakeAudio();
  const before = setup(db, audio);
  const hang = gate();
  const runs: number[] = [];
  const kind = (wait: boolean) => echo({
    run: async ctx => {
      runs.push(ctx.attempt);
      const item = ctx.audio.enqueue('render', { caption: ctx.input.text });
      if (wait) await hang.opened;
      return (await ctx.audio.wait(item.id)).status;
    },
  });
  before.jobs.register(kind(true));
  const caught = before.jobs.submit({ kind: 'echo', idempotencyKey: 'caught', input: { text: 'a' } }, 'u').job;
  const waiting = before.jobs.submit({ kind: 'echo', idempotencyKey: 'waiting', input: { text: 'b' } }, 'u').job;
  assert.deepEqual([before.jobs.get(caught.id).status, before.jobs.get(waiting.id).status], ['running', 'pending']);

  // A new process on the same database: reconcile first, then kinds register.
  const after = setup(db, audio);
  assert.equal(after.jobs.reconcileAfterRestart(), 1);
  assert.equal(after.jobs.get(caught.id).status, 'interrupted');
  assert.match(after.jobs.get(caught.id).error ?? '', /retry it/);
  after.jobs.register(kind(false));
  audio.finish('intent-1', 'succeeded');
  await new Promise(r => setTimeout(r, 30));
  audio.finish('intent-2', 'succeeded');
  await after.jobs.settled();
  assert.equal(after.jobs.get(caught.id).status, 'interrupted', 'never rerun by itself');
  assert.equal(after.jobs.get(waiting.id).status, 'succeeded');

  // The old process's step finishing late cannot overwrite the interruption.
  hang.open();
  await before.jobs.settled();
  assert.equal(after.jobs.get(caught.id).status, 'interrupted');

  const retried = after.jobs.retry(caught.id);
  assert.equal(retried.attempt, 2);
  await after.jobs.settled();
  assert.equal(after.jobs.get(caught.id).status, 'succeeded');
  assert.deepEqual(after.jobs.get(caught.id).audioIntentIds, ['intent-1'], 'the retry got the same audio item back');
  assert.equal(audio.enqueued.filter(k => k === `workflow:${caught.id}:render`).length, 1, 'rendered once');
  assert.throws(() => after.jobs.retry(caught.id), (e: unknown) => e instanceof WorkflowError && e.status === 409);
});

test('replay: events after a cursor, and a gap once the cursor falls out of the bounded window', async () => {
  const { jobs } = setup();
  const n = MAX_EVENTS_PER_JOB + 50;
  jobs.register(echo({ run: async ctx => { for (let i = 0; i < n; i++) ctx.emit('progress', { i }); } }));
  const job = jobs.submit({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } }, 'u').job;
  await jobs.settled();
  const total = jobs.get(job.id).lastSeq;
  assert.equal(total, n + 3, 'pending, running, n progress, succeeded');
  const recent = jobs.replay(job.id, 'u', total - 5);
  assert.deepEqual([recent.gap, recent.events.map(e => e.seq)], [false, [total - 4, total - 3, total - 2, total - 1, total]]);
  const stale = jobs.replay(job.id, 'u', 3);
  assert.equal(stale.gap, true);
  assert.equal(stale.events.length, MAX_EVENTS_PER_JOB);
  assert.equal(stale.events[0].seq, total - MAX_EVENTS_PER_JOB + 1);
  assert.deepEqual(jobs.replay(job.id, 'u', total), { job: jobs.get(job.id), events: [], gap: false });
  assert.throws(() => jobs.replay(job.id, 'someone else'), (e: unknown) => e instanceof WorkflowError && e.status === 404);
});

test('documents: one of two writes from the same revision lands; the other gets the current revision', () => {
  const db = new Database(':memory:');
  const docs = new WorkflowDocuments(db);
  const doc = docs.create('u', 'cover-draft', { title: 'a' });
  assert.equal(doc.revision, 1);
  const first = docs.update(doc.id, 'u', 1, { title: 'b' });
  assert.equal(first.revision, 2);
  assert.throws(() => docs.update(doc.id, 'u', 1, { title: 'c' }),
    (e: unknown) => e instanceof WorkflowError && e.status === 409 && e.extra.currentRevision === 2);
  assert.deepEqual(docs.get(doc.id, 'u').data, { title: 'b' }, 'the stale write changed nothing');
  // A merge function runs inside the same check.
  assert.deepEqual(docs.update(doc.id, 'u', 2, cur => ({ ...cur, late: true })).data, { title: 'b', late: true });
  assert.throws(() => docs.get(doc.id, 'other'), (e: unknown) => e instanceof WorkflowError && e.status === 404);
  assert.throws(() => docs.remove(doc.id, 'u', 2), (e: unknown) => e instanceof WorkflowError && e.status === 409);
  docs.remove(doc.id, 'u', 3);
  assert.throws(() => docs.get(doc.id, 'u'), (e: unknown) => e instanceof WorkflowError && e.status === 404);

  // The same check on a domain table, rolled back with the domain change.
  db.exec(`CREATE TABLE builder_projects (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, revision INTEGER NOT NULL DEFAULT 0)`);
  db.prepare(`INSERT INTO builder_projects (id, user_id, title) VALUES ('p', 'u', 'x')`).run();
  const edit = (expected: number, title: string) => db.transaction(() => {
    db.prepare('UPDATE builder_projects SET title = ? WHERE id = ?').run(title, 'p');
    return bumpRevision(db, 'builder_projects', 'p', expected, 'u');
  })();
  assert.equal(edit(0, 'y'), 1);
  assert.throws(() => edit(0, 'z'), (e: unknown) => e instanceof WorkflowError && e.status === 409 && e.extra.currentRevision === 1);
  assert.equal((db.prepare(`SELECT title FROM builder_projects WHERE id = 'p'`).get() as { title: string }).title, 'y');
  assert.throws(() => bumpRevision(db, 'songs', 'p', 0), /not revisioned/);
});

// ── HTTP ────────────────────────────────────────────────────────────────────

/** Read SSE frames until `stop` says so, then disconnect. */
async function readFrames(url: string, headers: Record<string, string>, stop: (frames: any[]) => boolean) {
  const controller = new AbortController();
  const res = await fetch(url, { headers, signal: controller.signal });
  const reader = res.body!.getReader();
  const frames: any[] = [];
  let buffer = '';
  try {
    while (!stop(frames)) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += new TextDecoder().decode(value);
      let at;
      while ((at = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        const data = block.split('\n').find(l => l.startsWith('data: '));
        const id = block.split('\n').find(l => l.startsWith('id: '));
        if (data) frames.push({ ...JSON.parse(data.slice(6)), id: id ? Number(id.slice(4)) : undefined });
      }
    }
  } finally { controller.abort(); }
  return frames;
}

test('HTTP: submit, stream, disconnect, resume with Last-Event-ID without loss or repeats, and conflicts as 409', async () => {
  const db = new Database(':memory:');
  const { jobs } = setup(db);
  const step = gate();
  jobs.register(echo({ run: async ctx => { ctx.emit('progress', { n: 1 }); await step.opened; ctx.emit('progress', { n: 2 }); return 'ok'; } }));
  const server = await startWorkflowTestServer({ db, jobs, userIdOf: req => (req.headers['x-user'] as string) || null });
  const http = { base: `${server.origin}/api/workflows`, close: server.close };
  const h = { 'x-user': 'u', 'Content-Type': 'application/json' };
  try {
    assert.equal((await fetch(`${http.base}/jobs`)).status, 401);
    const post = (body: unknown) => fetch(`${http.base}/jobs`, { method: 'POST', headers: h, body: JSON.stringify(body) });
    const created = await post({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } });
    assert.equal(created.status, 201);
    const { job } = await created.json();
    assert.equal((await post({ kind: 'echo', idempotencyKey: 'k', input: { text: 'a' } })).status, 200);
    assert.equal((await post({ kind: 'echo', idempotencyKey: 'k', input: { text: 'b' } })).status, 409);
    assert.equal((await post({ kind: 'echo', idempotencyKey: 'k2', input: { text: '' } })).status, 400);

    // First connection: snapshot plus events so far, then the client drops.
    const first = await readFrames(`${http.base}/jobs/${job.id}/events`, h, f => f.some(x => x.event?.type === 'progress'));
    assert.equal(first[0].type, 'snapshot');
    const lastSeen = Math.max(...first.filter(f => f.id).map(f => f.id));
    step.open();
    await jobs.settled();
    // Reconnect from the last id seen: exactly the rest, in order, then the stream ends.
    const second = await readFrames(`${http.base}/jobs/${job.id}/events`, { ...h, 'Last-Event-ID': String(lastSeen) }, () => false);
    assert.equal(second[0].type, 'snapshot');
    assert.equal(second[0].gap, false);
    const seqs = [...first, ...second].filter(f => f.id).map(f => f.id);
    assert.deepEqual(seqs, Array.from({ length: jobs.get(job.id).lastSeq }, (_, i) => i + 1));
    assert.equal(second.at(-1).event.data.status, 'succeeded');
    const polled = await (await fetch(`${http.base}/jobs/${job.id}?after=${lastSeen}`, { headers: h })).json();
    assert.deepEqual(polled.events.map((e: any) => e.seq), seqs.filter(s => s > lastSeen));
    assert.equal((await fetch(`${http.base}/jobs/${job.id}`, { headers: { 'x-user': 'other' } })).status, 404);
    assert.equal((await fetch(`${http.base}/jobs/${job.id}/cancel`, { method: 'POST', headers: h })).status, 409);

    const doc = (await (await fetch(`${http.base}/documents`, { method: 'POST', headers: h, body: JSON.stringify({ kind: 'd', data: { a: 1 } }) })).json()).document;
    const put = (expectedRevision: number) => fetch(`${http.base}/documents/${doc.id}`, { method: 'PUT', headers: h, body: JSON.stringify({ expectedRevision, data: { a: expectedRevision } }) });
    assert.equal((await put(1)).status, 200);
    const stale = await put(1);
    assert.equal(stale.status, 409);
    assert.equal((await stale.json()).currentRevision, 2);
  } finally {
    await http.close();
  }
});

test('HTTP: events that happen while a client is connected arrive live', async () => {
  const { jobs } = setup();
  const step = gate();
  jobs.register(echo({ run: async ctx => { await step.opened; ctx.emit('progress', { live: true }); return 'ok'; } }));
  const server = await startWorkflowTestServer({ jobs });
  try {
    const { job } = jobs.submit({ kind: 'echo', idempotencyKey: 'live', input: { text: 'a' } }, 'u');
    setTimeout(step.open, 50);
    const frames = await readFrames(`${server.origin}/api/workflows/jobs/${job.id}/events`, { Authorization: 'Bearer t' }, () => false);
    const live = frames.filter(f => f.event).map(f => f.event.type === 'status' ? f.event.data.status : f.event.type);
    assert.deepEqual(live, ['pending', 'running', 'progress', 'succeeded']);
  } finally {
    await server.close();
  }
});
