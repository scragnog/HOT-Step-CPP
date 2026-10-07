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

import { randomUUID } from 'node:crypto';
import type Database from 'better-sqlite3';
import type { AudioIntentItem, AudioIntentStatus, AudioQueueState } from '../../contracts/audioQueue.js';
import { requestVersion } from '../generation/resolve/resolveIntent.js';

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
  /** The live job, or undefined once it is no longer held. */
  getJob: (jobId: string) => { status: string; error?: string; result?: unknown } | undefined;
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
}

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
      updated_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS audio_intents_status ON audio_intents(status, created_at);
    CREATE TABLE IF NOT EXISTS audio_queue_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  `);
}

function toItem(r: Row): AudioIntentItem {
  return {
    id: r.id, idempotencyKey: r.idempotency_key, status: r.status, engine: r.engine,
    request: JSON.parse(r.request), meta: r.meta ? JSON.parse(r.meta) : null,
    jobId: r.job_id, attempt: r.attempt, previousJobIds: JSON.parse(r.previous_job_ids),
    waiting: r.waiting, cancelRequested: r.cancel_requested === 1, error: r.error,
    result: r.result ? JSON.parse(r.result) : null, createdAt: r.created_at, updatedAt: r.updated_at,
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
    return toItem(r);
  }

  list(status?: AudioIntentStatus): AudioIntentItem[] {
    const rows = status
      ? this.db.prepare('SELECT * FROM audio_intents WHERE status = ? ORDER BY created_at, rowid').all(status)
      : this.db.prepare('SELECT * FROM audio_intents ORDER BY created_at, rowid').all();
    return (rows as Row[]).map(toItem);
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
      this.db.prepare(`INSERT INTO audio_intents (id, idempotency_key, user_id, status, engine, request, request_hash, meta, created_at, updated_at)
        VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)`).run(
        id, input.idempotencyKey, userId, engine, JSON.stringify(request), hash,
        input.meta ? JSON.stringify(input.meta) : null, t, t);
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
    return { item: toItem(existing), created: false };
  }

  /** Cancel an item. Pending stops at once; submitting is cancelled as soon as
   *  its submit returns; submitted cancels its job. Terminal items are refused. */
  cancel(id: string): AudioIntentItem {
    const r = this.row(id);
    if (!r) throw new AudioQueueError(404, `Queue item ${id} not found`);
    if (TERMINAL.includes(r.status)) throw new AudioQueueError(409, `Queue item is already ${r.status}`);
    const t = this.now();
    if (r.status === 'pending') {
      // Guarded: the executor may have claimed it a moment ago.
      const done = this.db.prepare(`UPDATE audio_intents SET status = 'cancelled', waiting = NULL, updated_at = ? WHERE id = ? AND status = 'pending'`).run(t, id);
      if (done.changes === 1) return this.get(id);
    }
    this.db.prepare('UPDATE audio_intents SET cancel_requested = 1, updated_at = ? WHERE id = ?').run(t, id);
    const now = this.row(id)!;
    if (now.status === 'submitted' && now.job_id) this.finishCancel(now);
    return this.get(id);
  }

  private finishCancel(r: Row): void {
    if (r.job_id) this.deps.cancelJob(r.job_id);
    this.db.prepare(`UPDATE audio_intents SET status = 'cancelled', updated_at = ? WHERE id = ? AND status IN ('submitting','submitted')`).run(this.now(), r.id);
  }

  /** Put a failed, cancelled or interrupted item back in the queue as a new
   *  attempt. This is the only way anything renders twice. */
  retry(id: string): AudioIntentItem {
    const r = this.row(id);
    if (!r) throw new AudioQueueError(404, `Queue item ${id} not found`);
    if (!['failed', 'cancelled', 'interrupted'].includes(r.status)) {
      throw new AudioQueueError(409, `Only a failed, cancelled or interrupted item can be retried; this one is ${r.status}`);
    }
    const previous = JSON.parse(r.previous_job_ids) as string[];
    if (r.job_id) previous.push(r.job_id);
    this.db.prepare(`UPDATE audio_intents SET status = 'pending', job_id = NULL, attempt = attempt + 1, previous_job_ids = ?,
      waiting = NULL, cancel_requested = 0, error = NULL, result = NULL, updated_at = ? WHERE id = ?`).run(JSON.stringify(previous), this.now(), id);
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
   *  process. Pending items stay pending; they were never sent. */
  reconcileAfterRestart(): number {
    const t = this.now();
    const a = this.db.prepare(`UPDATE audio_intents SET status = 'interrupted', updated_at = ?,
      error = 'The server stopped while this item was being submitted. It may or may not have reached the engine; retry it to render it again.'
      WHERE status = 'submitting'`).run(t);
    const b = this.db.prepare(`UPDATE audio_intents SET status = 'interrupted', updated_at = ?,
      error = 'The server stopped while job ' || job_id || ' was running, so its outcome is unknown here. Check the library before retrying.'
      WHERE status = 'submitted'`).run(t);
    return a.changes + b.changes;
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
    const inFlight = () => (this.db.prepare(`SELECT COUNT(*) AS n FROM audio_intents WHERE status IN ('submitting','submitted')`).get() as { n: number }).n;
    const pending = this.db.prepare(`SELECT * FROM audio_intents WHERE status = 'pending' ORDER BY created_at, rowid`).all() as Row[];
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
   *  pass (the engine is not taking work). */
  private async submitOne(r: Row): Promise<boolean> {
    const claim = this.db.prepare(`UPDATE audio_intents SET status = 'submitting', waiting = NULL, updated_at = ?
      WHERE id = ? AND status = 'pending' AND cancel_requested = 0`).run(this.now(), r.id);
    if (claim.changes !== 1) return false;
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

  /** Bring submitted items up to date with their jobs. */
  private follow(): void {
    const rows = this.db.prepare(`SELECT * FROM audio_intents WHERE status = 'submitted'`).all() as Row[];
    for (const r of rows) {
      const job = r.job_id ? this.deps.getJob(r.job_id) : undefined;
      const t = this.now();
      if (!job) {
        this.db.prepare(`UPDATE audio_intents SET status = 'interrupted', updated_at = ?,
          error = 'Job ' || IFNULL(job_id,'?') || ' is no longer known to the generation queue (it was reset or pruned); its outcome is unknown.'
          WHERE id = ? AND status = 'submitted'`).run(t, r.id);
        continue;
      }
      if (r.cancel_requested === 1 && !JOB_TERMINAL.has(job.status)) { this.finishCancel(r); continue; }
      if (!JOB_TERMINAL.has(job.status)) continue;
      this.db.prepare(`UPDATE audio_intents SET status = ?, error = ?, result = ?, updated_at = ? WHERE id = ? AND status = 'submitted'`)
        .run(job.status, job.status === 'failed' ? (job.error ?? 'Generation failed') : null,
          job.result === undefined ? null : JSON.stringify(job.result), t, r.id);
    }
  }
}
