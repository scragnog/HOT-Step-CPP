// audioQueue/intentQueue.ts — the durable audio intent queue.
//
// The browser's audio queue (ui/src/stores/audioGenQueueStore.ts) holds its
// pending songs in local storage and submits them itself, so closing the tab
// stops the queue and two tabs can submit the same song twice. This is the
// server-side replacement: each item is a captured /api/generate body stored
// in SQLite with an idempotency key, and an executor in the Node process
// submits it through the same submitGeneration() the HTTP route uses (same
// checks, same job map, same GPU lane — no second scheduler).
//
// Rules this module guarantees:
//   - A request is captured once, at enqueue, and submitted unchanged. Its
//     engine is the engine it was resolved for (expectedBackend). It is only
//     submitted while that engine is active; after a switch it waits and says
//     so. The queue never switches engines itself.
//   - Submission is a claim (pending → submitting, one UPDATE) followed by the
//     submit call, so two executors, or a cancel racing a submit, cannot both
//     act on one item.
//   - Nothing submitted is ever submitted again automatically. After a
//     restart, an item caught submitting or submitted becomes `interrupted`:
//     its job lived in the old process's memory, so whether it rendered is
//     unknown here. Only an explicit retry renders it again.
//   - Two copies of the app can share this table (services/instance.ts). An
//     item belongs to the copy that queued it (owner_copy) and, once claimed,
//     to the process that submitted it (owner_instance). Only that copy claims
//     it, only that process follows its job, and an item whose process is
//     still alive is never interrupted by the other copy. A cancel from the
//     other copy only flags it; the owner acts on the flag.

import { randomUUID } from 'node:crypto';
import type Database from 'better-sqlite3';
import type { AudioIntentItem, AudioIntentStatus, AudioQueueState } from '../../contracts/audioQueue.js';
import type { ImportAudioQueue } from '../../contracts/audioQueue.js';
import { requestVersion } from '../generation/resolve/resolveIntent.js';
import { copyId, instanceAlive, instanceId } from '../instance.js';

export const MAX_IN_FLIGHT = 4;
const TERMINAL: AudioIntentStatus[] = ['succeeded', 'failed', 'cancelled', 'interrupted'];
const JOB_TERMINAL = new Set(['succeeded', 'failed', 'cancelled']);

export class AudioQueueError extends Error {
  constructor(readonly status: 400 | 404 | 409, message: string) { super(message); }
}

export interface AudioQueueDeps {
  db: Database.Database;
  /** submitGeneration from routes/generate.ts. */
  submit: (body: Record<string, unknown>, userId: string) => Promise<
    { ok: true; jobId: string } | { ok: false; status: number; body: Record<string, unknown> }>;
  /** The live job, or undefined once it is no longer held. attempts carries
   *  the engine's own lifecycle timestamps (routes/generate.ts's
   *  GenerationAttempt) — startedAt is set after GPU-lane queue wait, so it
   *  excludes queueing; endedAt is set on every end path, success, failure
   *  or cancel. This is the only trustworthy source for "how long did this
   *  actually render", because it does not depend on this queue's own poll
   *  cadence ever observing the job mid-run. */
  getJob: (jobId: string) => {
    status: string; error?: string; result?: unknown;
    attempts?: Array<{ startedAt: number; endedAt?: number }>;
  } | undefined;
  cancelJob: (jobId: string) => boolean;
  activeEngine: () => string;
  isEngine: (id: string) => boolean;
  now?: () => number;
}

interface Row {
  id: string; idempotency_key: string; user_id: string; status: AudioIntentStatus; engine: string;
  request: string; request_hash: string; meta: string | null; job_id: string | null; attempt: number;
  previous_job_ids: string; waiting: string | null; cancel_requested: number; error: string | null;
  result: string | null; created_at: number; updated_at: number;
  started_at: number | null; elapsed_seconds: number | null;
  owner_copy: string | null; owner_instance: string | null;
}

/** Owned by a live process other than this one: hands off. */
const foreignLive = (r: Pick<Row, 'owner_instance'>) => r.owner_instance !== instanceId() && instanceAlive(r.owner_instance);

export function ensureAudioQueueSchema(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS audio_intents (
      id TEXT PRIMARY KEY,
      idempotency_key TEXT NOT NULL UNIQUE,
      user_id TEXT NOT NULL,
      status TEXT NOT NULL,
      engine TEXT NOT NULL,
      request TEXT NOT NULL,
      request_hash TEXT NOT NULL,
      meta TEXT,
      job_id TEXT,
      attempt INTEGER NOT NULL DEFAULT 1,
      previous_job_ids TEXT NOT NULL DEFAULT '[]',
      waiting TEXT,
      cancel_requested INTEGER NOT NULL DEFAULT 0,
      error TEXT,
      result TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      started_at INTEGER,
      elapsed_seconds INTEGER
    );
    CREATE INDEX IF NOT EXISTS audio_intents_status ON audio_intents(status, created_at);
    CREATE TABLE IF NOT EXISTS audio_queue_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS audio_queue_imports (backup_id TEXT PRIMARY KEY, receipt TEXT NOT NULL);
  `);
  // Migration for a database created before these columns existed.
  for (const [name, type] of [['started_at', 'INTEGER'], ['elapsed_seconds', 'INTEGER'], ['owner_copy', 'TEXT'], ['owner_instance', 'TEXT']]) {
    const row = db.prepare(`SELECT COUNT(*) AS c FROM pragma_table_info('audio_intents') WHERE name = ?`).get(name) as { c: number };
    if (row.c === 0) db.exec(`ALTER TABLE audio_intents ADD COLUMN ${name} ${type}`);
  }
}

/** Server-side status strings that mean "accepted, but the engine has not
 *  actually started this job yet" — mirrors isQueued() in the UI's
 *  generationTimer.ts. Real generation time excludes this wait. */
const JOB_QUEUED = new Set(['pending', 'queued']);

function elapsedSecondsFor(r: Row, now: number): number | null {
  if (r.elapsed_seconds != null) return r.elapsed_seconds;
  // Only an actively-running item ticks live; a terminal row must already
  // have elapsed_seconds frozen by whoever moved it there (follow(),
  // finishCancel, reconcileAfterRestart). Falling through to started_at for
  // a terminal row with no frozen value is how a finished item kept
  // counting up (2s, then 62s a minute later — the F19 round-2 regression).
  if (r.status === 'submitted' && r.started_at != null) return Math.round((now - r.started_at) / 1000);
  return null;
}

/** The job's own running time, from its last attempt's lifecycle timestamps
 *  — never from submission or queue-wait. null when the engine never told us
 *  it started (still queued when the job went terminal, or never ran at
 *  all): showing nothing beats showing queue wait as render time. */
function attemptElapsedSeconds(
  job: { attempts?: Array<{ startedAt: number; endedAt?: number }> } | undefined, fallbackEnd: number,
): number | null {
  const attempt = job?.attempts?.[job.attempts.length - 1];
  if (attempt?.startedAt == null) return null;
  return Math.round(((attempt.endedAt ?? fallbackEnd) - attempt.startedAt) / 1000);
}

function toItem(r: Row, now: number, db: Database.Database): AudioIntentItem {
  const result = r.result ? JSON.parse(r.result) : null;
  const songTitles = (Array.isArray(result?.songIds) ? result.songIds : []).map((id: unknown) => {
    if (typeof id !== 'string') return null;
    const song = db.prepare('SELECT title FROM songs WHERE id = ? AND user_id = ?').get(id, r.user_id) as { title: string } | undefined;
    return song?.title ?? null;
  });
  return {
    id: r.id, idempotencyKey: r.idempotency_key, status: r.status, engine: r.engine,
    request: JSON.parse(r.request), meta: r.meta ? JSON.parse(r.meta) : null,
    jobId: r.job_id, attempt: r.attempt, previousJobIds: JSON.parse(r.previous_job_ids),
    waiting: r.waiting, cancelRequested: r.cancel_requested === 1, error: r.error,
    result, songTitles, createdAt: r.created_at, updatedAt: r.updated_at,
    elapsedSeconds: elapsedSecondsFor(r, now),
  };
}

export class AudioIntentQueue {
  private readonly db: Database.Database;
  private readonly now: () => number;
  private ticking: Promise<void> | null = null;
  private timer: NodeJS.Timeout | null = null;

  constructor(private readonly deps: AudioQueueDeps) {
    this.db = deps.db;
    this.now = deps.now ?? Date.now;
    ensureAudioQueueSchema(this.db);
  }

  // ── Items ─────────────────────────────────────────────────────────────────

  private row(id: string): Row | undefined {
    return this.db.prepare('SELECT * FROM audio_intents WHERE id = ?').get(id) as Row | undefined;
  }

  get(id: string): AudioIntentItem {
    const r = this.row(id);
    if (!r) throw new AudioQueueError(404, `Queue item ${id} not found`);
    return toItem(r, this.now(), this.db);
  }

  dismiss(id: string): void {
    const r = this.row(id);
    if (!r) throw new AudioQueueError(404, `Queue item ${id} not found`);
    if (!TERMINAL.includes(r.status)) throw new AudioQueueError(409, 'Only a finished queue item can be removed');
    this.db.prepare('DELETE FROM audio_intents WHERE id = ?').run(id);
  }

  list(status?: AudioIntentStatus): AudioIntentItem[] {
    const rows = status
      ? this.db.prepare('SELECT * FROM audio_intents WHERE status = ? ORDER BY created_at, rowid').all(status)
      : this.db.prepare('SELECT * FROM audio_intents ORDER BY created_at, rowid').all();
    const now = this.now();
    return (rows as Row[]).map(r => toItem(r, now, this.db));
  }

  /** Import a verified browser backup atomically. A submitted legacy job is
   *  followed by id, never sent again; an unknown job is interrupted.
   *  An import hands the queue back to the server, so it also clears the pause
   *  a rollback export set: 'hold' and 'discard' act on the imported items'
   *  own status, never on new work. Left paused, every later submit would
   *  sit pending with nothing in the UI to say why. */
  importLegacy(input: ImportAudioQueue, userId: string): { backupId: string; imported: number; existing: number; items: { legacyId: string; itemId: string; status: AudioIntentStatus }[] } {
    const prior = this.db.prepare('SELECT receipt FROM audio_queue_imports WHERE backup_id = ?').get(input.backupId) as { receipt: string } | undefined;
    if (prior) { this.setPaused(false); return JSON.parse(prior.receipt); }
    const seen = new Set<string>();
    for (const item of input.items) {
      if (seen.has(item.legacyId)) throw new AudioQueueError(400, `Duplicate legacy item id: ${item.legacyId}`);
      seen.add(item.legacyId);
      const sibling = (item.meta.view as Record<string, unknown> | undefined)?.mm3TakeOf;
      if (item.status === 'pending' && !item.jobId && !item.request && !sibling) {
        throw new AudioQueueError(400, `Item ${item.legacyId} needs a resolved request before import`);
      }
      if (item.request && Object.keys(item.request).length === 0) throw new AudioQueueError(400, `Item ${item.legacyId} has an empty request`);
      if (item.request?.expectedBackend !== undefined && (typeof item.request.expectedBackend !== 'string' || !this.deps.isEngine(item.request.expectedBackend))) {
        throw new AudioQueueError(400, `Item ${item.legacyId} has an invalid expectedBackend`);
      }
    }
    return this.db.transaction(() => {
      const receipt: { backupId: string; imported: number; existing: number; items: { legacyId: string; itemId: string; status: AudioIntentStatus }[] } = {
        backupId: input.backupId, imported: 0, existing: 0, items: [],
      };
      for (const old of input.items) {
        const key = `browser:${old.legacyId}`;
        const existing = this.db.prepare('SELECT * FROM audio_intents WHERE idempotency_key = ?').get(key) as Row | undefined;
        if (existing) {
          if (existing.meta && (JSON.parse(existing.meta) as Record<string, unknown>).legacyId !== old.legacyId) {
            throw new AudioQueueError(409, `Imported item ${old.legacyId} conflicts with an existing queue item`);
          }
          if (old.request && requestVersion({ ...old.request, expectedBackend: existing.engine }) !== existing.request_hash) {
            throw new AudioQueueError(409, `Imported item ${old.legacyId} has changed since its first import`);
          }
          receipt.existing++;
          receipt.items.push({ legacyId: old.legacyId, itemId: existing.id, status: existing.status });
          continue;
        }
        const request = structuredClone(old.request ?? {});
        const engine = typeof request.expectedBackend === 'string' ? request.expectedBackend : this.deps.activeEngine();
        request.expectedBackend = engine;
        const job = old.jobId ? this.deps.getJob(old.jobId) : undefined;
        let status: AudioIntentStatus;
        if (old.status === 'succeeded' || old.status === 'failed') status = old.status;
        else if ((old.meta.view as Record<string, unknown> | undefined)?.mm3TakeOf && !old.jobId) status = 'interrupted';
        else if (old.jobId) status = job ? 'submitted' : 'interrupted';
        else if (old.status !== 'pending') status = 'interrupted';
        else status = input.choice === 'hold' ? 'held' : input.choice === 'discard' ? 'cancelled' : 'pending';
        const id = randomUUID();
        const t = this.now();
        this.db.prepare(`INSERT INTO audio_intents (id, idempotency_key, user_id, status, engine, request, request_hash, meta,
          job_id, error, result, created_at, updated_at, owner_copy, owner_instance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
          id, key, userId, status, engine, JSON.stringify(request), requestVersion(request),
          JSON.stringify({ ...old.meta, legacyId: old.legacyId, backupId: input.backupId }),
          old.jobId ?? null, old.error ?? null, old.result ? JSON.stringify(old.result) : null, t, t,
          copyId(), status === 'submitted' ? instanceId() : null,
        );
        receipt.imported++;
        receipt.items.push({ legacyId: old.legacyId, itemId: id, status });
      }
      this.db.prepare('INSERT INTO audio_queue_imports (backup_id, receipt) VALUES (?, ?)').run(input.backupId, JSON.stringify(receipt));
      this.setPaused(false);
      return receipt;
    })();
  }

  resumeHeld(ids: string[]): AudioIntentItem[] {
    const change = this.db.transaction(() => ids.map(id => {
      const done = this.db.prepare(`UPDATE audio_intents SET status = 'pending', waiting = NULL, owner_copy = ?, updated_at = ? WHERE id = ? AND status = 'held'`)
        .run(copyId(), this.now(), id);
      if (done.changes !== 1) throw new AudioQueueError(409, `Queue item ${id} is not held`);
      return this.get(id);
    }));
    return change();
  }

  /** Queue a request, or return the item already queued under this key. */
  enqueue(input: { idempotencyKey: string; request: Record<string, unknown>; meta?: Record<string, unknown> }, userId: string): { item: AudioIntentItem; created: boolean } {
    const request = structuredClone(input.request);
    const asserted = request.expectedBackend;
    if (asserted !== undefined && (typeof asserted !== 'string' || !this.deps.isEngine(asserted))) {
      throw new AudioQueueError(400, `Unknown engine in expectedBackend: ${String(asserted)}`);
    }
    // Capture the engine now: a request that names none is pinned to the one
    // active at enqueue, so a later switch cannot redirect it.
    const engine = typeof asserted === 'string' ? asserted : this.deps.activeEngine();
    request.expectedBackend = engine;
    const hash = requestVersion(request);

    const existing = this.db.prepare('SELECT * FROM audio_intents WHERE idempotency_key = ?').get(input.idempotencyKey) as Row | undefined;
    if (existing) return this.sameOrConflict(existing, hash);
    const t = this.now();
    const id = randomUUID();
    try {
      this.db.prepare(`INSERT INTO audio_intents (id, idempotency_key, user_id, status, engine, request, request_hash, meta, created_at, updated_at, owner_copy)
        VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)`).run(
        id, input.idempotencyKey, userId, engine, JSON.stringify(request), hash,
        input.meta ? JSON.stringify(input.meta) : null, t, t, copyId());
    } catch (err: any) {
      // Lost a race with another client using the same key.
      if (String(err?.code).startsWith('SQLITE_CONSTRAINT')) {
        const winner = this.db.prepare('SELECT * FROM audio_intents WHERE idempotency_key = ?').get(input.idempotencyKey) as Row;
        return this.sameOrConflict(winner, hash);
      }
      throw err;
    }
    return { item: this.get(id), created: true };
  }

  private sameOrConflict(existing: Row, hash: string): { item: AudioIntentItem; created: false } {
    if (existing.request_hash !== hash) {
      throw new AudioQueueError(409, `Idempotency key '${existing.idempotency_key}' is already used by a different request`);
    }
    return { item: toItem(existing, this.now(), this.db), created: false };
  }

  /** Cancel an item. Pending stops at once; submitting is cancelled as soon as
   *  its submit returns; submitted cancels its job. Terminal items are refused. */
  cancel(id: string): AudioIntentItem {
    const r = this.row(id);
    if (!r) throw new AudioQueueError(404, `Queue item ${id} not found`);
    if (TERMINAL.includes(r.status)) throw new AudioQueueError(409, `Queue item is already ${r.status}`);
    const t = this.now();
    if (r.status === 'pending' || r.status === 'held') {
      // Guarded: the executor may have claimed it a moment ago.
      const done = this.db.prepare(`UPDATE audio_intents SET status = 'cancelled', waiting = NULL, updated_at = ? WHERE id = ? AND status IN ('pending','held')`).run(t, id);
      if (done.changes === 1) return this.get(id);
    }
    this.db.prepare('UPDATE audio_intents SET cancel_requested = 1, updated_at = ? WHERE id = ?').run(t, id);
    const now = this.row(id)!;
    // Another live process holds the job: its follow() sees the flag and cancels it.
    if (now.status === 'submitted' && now.job_id && !foreignLive(now)) this.finishCancel(now);
    return this.get(id);
  }

  private finishCancel(r: Row): void {
    // Read the job before cancelling it — cancelJob() may drop it from the
    // live map once it acknowledges, taking its attempt timestamps with it.
    const job = r.job_id ? this.deps.getJob(r.job_id) : undefined;
    const t = this.now();
    const elapsedSeconds = attemptElapsedSeconds(job, t);
    if (r.job_id) this.deps.cancelJob(r.job_id);
    this.db.prepare(`UPDATE audio_intents SET status = 'cancelled', elapsed_seconds = ?, updated_at = ? WHERE id = ? AND status IN ('submitting','submitted')`)
      .run(elapsedSeconds, t, r.id);
  }

  /** Put a failed, cancelled or interrupted item back in the queue as a new
   *  attempt. This is the only way anything renders twice. */
  retry(id: string): AudioIntentItem {
    const r = this.row(id);
    if (!r) throw new AudioQueueError(404, `Queue item ${id} not found`);
    if (!['failed', 'cancelled', 'interrupted'].includes(r.status)) {
      throw new AudioQueueError(409, `Only a failed, cancelled or interrupted item can be retried; this one is ${r.status}`);
    }
    if (Object.keys(JSON.parse(r.request) as Record<string, unknown>).every(key => key === 'expectedBackend')) {
      throw new AudioQueueError(409, 'This imported item has no resolved request to retry');
    }
    const previous = JSON.parse(r.previous_job_ids) as string[];
    if (r.job_id) previous.push(r.job_id);
    this.db.prepare(`UPDATE audio_intents SET status = 'pending', job_id = NULL, attempt = attempt + 1, previous_job_ids = ?,
      waiting = NULL, cancel_requested = 0, error = NULL, result = NULL, started_at = NULL, elapsed_seconds = NULL,
      owner_copy = ?, owner_instance = NULL, updated_at = ? WHERE id = ?`).run(JSON.stringify(previous), copyId(), this.now(), id);
    return this.get(id);
  }

  // ── Queue state ───────────────────────────────────────────────────────────

  private paused(): boolean {
    const v = this.db.prepare(`SELECT value FROM audio_queue_state WHERE key = 'paused'`).get() as { value: string } | undefined;
    return v?.value === '1';
  }

  /** Pause stops new submissions; what is already submitted carries on. */
  setPaused(paused: boolean): AudioQueueState {
    this.db.prepare(`INSERT INTO audio_queue_state (key, value) VALUES ('paused', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value`).run(paused ? '1' : '0');
    return this.state();
  }

  state(): AudioQueueState {
    const counts: AudioQueueState['counts'] = {};
    for (const row of this.db.prepare('SELECT status, COUNT(*) AS n FROM audio_intents GROUP BY status').all() as Array<{ status: AudioIntentStatus; n: number }>) {
      counts[row.status] = row.n;
    }
    return { paused: this.paused(), maxInFlight: MAX_IN_FLIGHT, counts };
  }

  // ── Restart ───────────────────────────────────────────────────────────────

  /** Run once at startup, before the executor. Never resubmits: an item that
   *  was submitting or submitted when the process stopped is surfaced as
   *  interrupted, because its job (and whether it rendered) died with the old
   *  process. Pending items stay pending; they were never sent. Items another
   *  live process submitted are left alone. */
  reconcileAfterRestart(): number {
    return this.interrupt(r => !foreignLive(r));
  }

  /** Interrupt the submitting/submitted items `which` picks. */
  private interrupt(which: (r: Row) => boolean): number {
    const ids = (this.db.prepare(`SELECT * FROM audio_intents WHERE status IN ('submitting','submitted')`).all() as Row[])
      .filter(which).map(r => r.id);
    if (!ids.length) return 0;
    const t = this.now();
    // The job itself died with the old process, so its attempt timestamps
    // are gone too — started_at (our own "first observed running" mark,
    // persisted separately) is the only timing left. Freeze it now: leaving
    // it live would have elapsedSecondsFor recompute (restart time - started_at)
    // forever, growing on every later read (the "2s, then 62s a minute
    // later" bug). No started_at means it never observed running: null.
    const freeze = `elapsed_seconds = CASE WHEN started_at IS NOT NULL THEN CAST(ROUND((? - started_at) / 1000.0) AS INTEGER) ELSE NULL END`;
    const a = this.db.prepare(`UPDATE audio_intents SET status = 'interrupted', updated_at = ?, ${freeze},
      error = 'The server stopped while this item was being submitted. It may or may not have reached the engine; retry it to render it again.'
      WHERE status = 'submitting' AND id IN (SELECT value FROM json_each(?))`);
    const b = this.db.prepare(`UPDATE audio_intents SET status = 'interrupted', updated_at = ?, ${freeze},
      error = 'The server stopped while job ' || job_id || ' was running, so its outcome is unknown here. Check the library before retrying.'
      WHERE status = 'submitted' AND id IN (SELECT value FROM json_each(?))`);
    const list = JSON.stringify(ids);
    return this.db.transaction(() => a.run(t, t, list).changes + b.run(t, t, list).changes)();
  }

  // ── Executor ──────────────────────────────────────────────────────────────

  start(intervalMs = 1500): void {
    if (this.timer) return;
    this.reconcileAfterRestart();
    this.timer = setInterval(() => { void this.tick(); }, intervalMs);
    this.timer.unref();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  /** One pass: follow submitted jobs, then submit what may go. Passes never
   *  overlap; a call during a pass waits for it. */
  tick(): Promise<void> {
    if (!this.ticking) this.ticking = this.pass().finally(() => { this.ticking = null; });
    return this.ticking;
  }

  private async pass(): Promise<void> {
    this.follow();
    if (this.paused()) return;
    // Only this process's submissions count, and only this copy's (or legacy
    // unowned) pending items are taken: the other copy runs its own queue.
    const inFlight = () => (this.db.prepare(`SELECT COUNT(*) AS n FROM audio_intents WHERE status IN ('submitting','submitted') AND owner_instance = ?`)
      .get(instanceId()) as { n: number }).n;
    const pending = this.db.prepare(`SELECT * FROM audio_intents WHERE status = 'pending' AND (owner_copy IS NULL OR owner_copy = ?) ORDER BY created_at, rowid`)
      .all(copyId()) as Row[];
    const active = this.deps.activeEngine();
    for (const r of pending) {
      if (inFlight() >= MAX_IN_FLIGHT) break;
      if (r.engine !== active) {
        this.setWaiting(r.id, `Waiting for the ${r.engine} engine; ${active} is active`);
        continue;
      }
      const stop = await this.submitOne(r);
      if (stop) break;
    }
  }

  private setWaiting(id: string, waiting: string | null): void {
    this.db.prepare(`UPDATE audio_intents SET waiting = ?, updated_at = ? WHERE id = ? AND status = 'pending' AND IFNULL(waiting,'') <> IFNULL(?,'')`)
      .run(waiting, this.now(), id, waiting);
  }

  /** Claim, submit, record. Returns true when submission should stop for this
   *  pass (the engine is not taking work, or the queue was paused meanwhile). */
  private async submitOne(r: Row): Promise<boolean> {
    // The pause check is part of the claim itself: a pause that lands while
    // an earlier submit in this pass is awaiting stops every later claim.
    const claim = this.db.prepare(`UPDATE audio_intents SET status = 'submitting', waiting = NULL, owner_instance = ?, updated_at = ?
      WHERE id = ? AND status = 'pending' AND cancel_requested = 0
        AND NOT EXISTS (SELECT 1 FROM audio_queue_state WHERE key = 'paused' AND value = '1')`).run(instanceId(), this.now(), r.id);
    if (claim.changes !== 1) return this.paused();
    let result: Awaited<ReturnType<AudioQueueDeps['submit']>>;
    try {
      result = await this.deps.submit(JSON.parse(r.request), r.user_id);
    } catch (err) {
      result = { ok: false, status: 500, body: { error: err instanceof Error ? err.message : String(err) } };
    }
    const now = this.row(r.id)!;
    const t = this.now();
    if (result.ok) {
      this.db.prepare(`UPDATE audio_intents SET status = 'submitted', job_id = ?, updated_at = ? WHERE id = ?`).run(result.jobId, t, r.id);
      if (now.cancel_requested === 1) this.finishCancel({ ...now, job_id: result.jobId });
      return false;
    }
    if (now.cancel_requested === 1) {
      this.db.prepare(`UPDATE audio_intents SET status = 'cancelled', updated_at = ? WHERE id = ?`).run(t, r.id);
      return false;
    }
    const message = String(result.body?.error ?? `Submit failed (${result.status})`);
    if (result.status === 409 || result.status === 503) {
      // Not taken, nothing ran: the engine switched under us, or is not ready.
      // Back to pending, and say why.
      this.db.prepare(`UPDATE audio_intents SET status = 'pending', waiting = ?, updated_at = ? WHERE id = ?`).run(message, t, r.id);
      return true;
    }
    this.db.prepare(`UPDATE audio_intents SET status = 'failed', error = ?, updated_at = ? WHERE id = ?`).run(message, t, r.id);
    return false;
  }

  /** Bring this process's submitted items up to date with their jobs, and
   *  interrupt items whose submitting process has died (the other copy
   *  crashed or stopped). Items a live process holds are its own business. */
  private follow(): void {
    this.interrupt(r => r.owner_instance !== instanceId() && !instanceAlive(r.owner_instance));
    const rows = this.db.prepare(`SELECT * FROM audio_intents WHERE status = 'submitted' AND owner_instance = ?`).all(instanceId()) as Row[];
    for (const r of rows) {
      const job = r.job_id ? this.deps.getJob(r.job_id) : undefined;
      const t = this.now();
      if (!job) {
        // The job vanished from the live map (reset/pruned) without this
        // loop ever seeing it reach a terminal status — same situation as a
        // restart: its own attempt timestamps are gone with it, so freeze
        // from whatever started_at we had observed ourselves, same as
        // reconcileAfterRestart, rather than leave it to tick forever.
        const freeze = `elapsed_seconds = CASE WHEN started_at IS NOT NULL THEN CAST(ROUND((? - started_at) / 1000.0) AS INTEGER) ELSE NULL END`;
        this.db.prepare(`UPDATE audio_intents SET status = 'interrupted', updated_at = ?, ${freeze},
          error = 'Job ' || IFNULL(job_id,'?') || ' is no longer known to the generation queue (it was reset or pruned); its outcome is unknown.'
          WHERE id = ? AND status = 'submitted'`).run(t, t, r.id);
        continue;
      }
      // The moment the engine actually starts this job, not when it was
      // accepted — a deep queue can leave a "submitted" job sitting as
      // job.status 'pending'/'queued' for a while first (issue #64's rule,
      // applied here instead of trusting a client's poll cadence to catch
      // it). Used only for the live ticker below, while still running — the
      // frozen value on terminal comes from the job's own attempt timestamps
      // (attemptElapsedSeconds), not from this.
      if (r.started_at == null && !JOB_QUEUED.has(job.status) && !JOB_TERMINAL.has(job.status)) {
        this.db.prepare(`UPDATE audio_intents SET started_at = ? WHERE id = ? AND started_at IS NULL`).run(t, r.id);
        r.started_at = t;
      }
      if (r.cancel_requested === 1 && !JOB_TERMINAL.has(job.status)) { this.finishCancel(r); continue; }
      if (!JOB_TERMINAL.has(job.status)) continue;
      // Freeze real generation time now, while this item is still in front of
      // us: once status leaves 'submitted' it never comes back through this
      // loop, so this is the only chance to record it. From the job's own
      // attempt timestamps, never from submission — a job that sat queued
      // for 600s and ran for 1s must read 1s, not 601s.
      const elapsedSeconds = attemptElapsedSeconds(job, t);
      this.db.prepare(`UPDATE audio_intents SET status = ?, error = ?, result = ?, elapsed_seconds = ?, updated_at = ? WHERE id = ? AND status = 'submitted'`)
        .run(job.status, job.status === 'failed' ? (job.error ?? 'Generation failed') : null,
          job.result === undefined ? null : JSON.stringify(job.result), elapsedSeconds, t, r.id);
    }
  }
}
