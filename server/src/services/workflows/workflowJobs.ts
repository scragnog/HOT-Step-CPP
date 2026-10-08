// workflows/workflowJobs.ts — durable workflow jobs with a bounded,
// replayable event stream.
//
// A studio registers a kind (registerWorkflowKind) with an input schema and a
// run function; clients submit jobs of that kind through /api/workflows. The
// kind owns the domain policy; this module owns only the lifecycle, built on
// the same rules as the audio intent queue (services/audioQueue/intentQueue.ts):
//   - The input is validated and captured once, at submit. A job runs from
//     that copy, never from client state.
//   - Idempotency is scoped to (user, kind, key): the same key and input
//     returns the existing job, the same key with another input is a 409.
//   - Starting is a claim (pending → running, one guarded UPDATE), so a
//     cancel racing a start, or two passes, cannot both act on one job.
//   - Cancel is acknowledged at once: a pending job is cancelled outright; a
//     running one gets cancelRequested, a 'cancel-requested' event, an aborted
//     signal and its audio items cancelled, and ends `cancelled` without
//     waiting for its step. Terminal jobs refuse a cancel. A timeout ends a
//     run the same way, as `failed`.
//   - Each run holds a claim id. A step's writes and its completion count
//     only while that claim is current, so a step that ignored its signal, or
//     the old process's step after a restart and retry, changes nothing.
//   - Nothing that started is run again automatically. After a restart a
//     job caught running is `interrupted`; only an explicit retry reruns it.
//   - Audio never runs here. A step queues it on the audio intent queue
//     (ctx.audio), which owns GPU admission, under a key scoped to the job.
//     A retried job therefore gets the same audio items back instead of
//     rendering twice, unless the kind puts ctx.attempt in the key.
//   - Every state change is also an event. Events are numbered per job with
//     no holes; only the newest MAX_EVENTS_PER_JOB are kept, and a reader
//     whose cursor fell behind that window is told so (gap) with the job's
//     current state.

import { EventEmitter } from 'node:events';
import { randomUUID } from 'node:crypto';
import type Database from 'better-sqlite3';
import type { z } from 'zod/v4';
import {
  MAX_EVENTS_PER_JOB,
  type SubmitWorkflowJob, type WorkflowEvent, type WorkflowJob, type WorkflowJobStatus, type WorkflowReplay,
} from '../../contracts/workflow.js';
import type { AudioIntentItem } from '../../contracts/audioQueue.js';
import { requestVersion } from '../generation/resolve/resolveIntent.js';

const TERMINAL: WorkflowJobStatus[] = ['succeeded', 'failed', 'cancelled', 'interrupted'];
const AUDIO_TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
const MAX_INPUT_BYTES = 1_000_000;
const MAX_EVENT_BYTES = 64_000;
const DEFAULT_TIMEOUT_MS = 30 * 60_000;

export class WorkflowError extends Error {
  constructor(readonly status: 400 | 404 | 409, message: string, readonly extra: Record<string, unknown> = {}) { super(message); }
}

export interface WorkflowContext<I> {
  jobId: string;
  userId: string;
  input: I;
  attempt: number;
  /** Aborted on cancel or timeout. Pass it to fetches and LLM calls. */
  signal: AbortSignal;
  /** Throws if the job was cancelled or timed out. Call between steps. */
  throwIfCancelled(): void;
  /** Append an event to the job's stream (data is JSON, at most 64 KB). */
  emit(type: string, data?: unknown): void;
  audio: {
    /** Queue a captured /api/generate body on the audio intent queue. `key`
     *  is unique within this job; repeating it returns the same item. */
    enqueue(key: string, request: Record<string, unknown>, meta?: Record<string, unknown>): AudioIntentItem;
    /** Resolve when the item is succeeded, failed, cancelled or interrupted. */
    wait(intentId: string): Promise<AudioIntentItem>;
  };
}

export interface WorkflowKind<I extends Record<string, unknown> = Record<string, unknown>> {
  kind: string;
  /** Validates and normalises the submitted input; the parsed value is what
   *  gets captured and hashed. */
  input: z.ZodType<I>;
  run(ctx: WorkflowContext<I>): Promise<unknown>;
  /** Jobs of this kind running at once in this process. Default 1. */
  maxConcurrent?: number;
  /** A run longer than this is aborted and fails. Default 30 minutes. */
  timeoutMs?: number;
}

export interface WorkflowDeps {
  db: Database.Database;
  audio: {
    enqueue(input: { idempotencyKey: string; request: Record<string, unknown>; meta?: Record<string, unknown> }, userId: string): { item: AudioIntentItem; created: boolean };
    get(id: string): AudioIntentItem;
    cancel(id: string): AudioIntentItem;
  };
  now?: () => number;
  /** Poll interval for ctx.audio.wait. */
  audioPollMs?: number;
}

interface Row {
  id: string; user_id: string; kind: string; idempotency_key: string; status: WorkflowJobStatus;
  input: string; input_hash: string; attempt: number; cancel_requested: number; claim_id: string | null; audio_intent_ids: string;
  result: string | null; error: string | null; last_seq: number;
  created_at: number; started_at: number | null; finished_at: number | null; updated_at: number;
}

export function ensureWorkflowSchema(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS workflow_jobs (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      idempotency_key TEXT NOT NULL,
      status TEXT NOT NULL,
      input TEXT NOT NULL,
      input_hash TEXT NOT NULL,
      attempt INTEGER NOT NULL DEFAULT 1,
      cancel_requested INTEGER NOT NULL DEFAULT 0,
      claim_id TEXT,
      audio_intent_ids TEXT NOT NULL DEFAULT '[]',
      result TEXT,
      error TEXT,
      last_seq INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL,
      started_at INTEGER,
      finished_at INTEGER,
      updated_at INTEGER NOT NULL,
      UNIQUE (user_id, kind, idempotency_key)
    );
    CREATE INDEX IF NOT EXISTS workflow_jobs_status ON workflow_jobs(status, created_at);
    CREATE TABLE IF NOT EXISTS workflow_events (
      job_id TEXT NOT NULL REFERENCES workflow_jobs(id) ON DELETE CASCADE,
      seq INTEGER NOT NULL,
      type TEXT NOT NULL,
      data TEXT,
      at INTEGER NOT NULL,
      PRIMARY KEY (job_id, seq)
    );
  `);
  // Databases from before claim_id was added.
  const hasClaim = (db.prepare(`SELECT COUNT(*) AS c FROM pragma_table_info('workflow_jobs') WHERE name = 'claim_id'`).get() as { c: number }).c;
  if (!hasClaim) db.exec('ALTER TABLE workflow_jobs ADD COLUMN claim_id TEXT');
}

function toJob(r: Row): WorkflowJob {
  return {
    id: r.id, kind: r.kind, idempotencyKey: r.idempotency_key, status: r.status,
    input: JSON.parse(r.input), attempt: r.attempt, cancelRequested: r.cancel_requested === 1,
    audioIntentIds: JSON.parse(r.audio_intent_ids), result: r.result === null ? null : JSON.parse(r.result),
    error: r.error, lastSeq: r.last_seq, createdAt: r.created_at, startedAt: r.started_at,
    finishedAt: r.finished_at, updatedAt: r.updated_at,
  };
}

const message = (err: unknown) => err instanceof Error ? err.message : String(err);

export class WorkflowJobs {
  private readonly db: Database.Database;
  private readonly now: () => number;
  private readonly kinds = new Map<string, WorkflowKind<any>>();
  /** Runs in this process, by job id. */
  private readonly running = new Map<string, { controller: AbortController; kind: string }>();
  private readonly events = new EventEmitter();
  /** Settles when a run (by claim id) has finished and been recorded (tests, shutdown). */
  private readonly runs = new Map<string, Promise<void>>();

  constructor(private readonly deps: WorkflowDeps) {
    this.db = deps.db;
    this.now = deps.now ?? Date.now;
    this.events.setMaxListeners(0);
    ensureWorkflowSchema(this.db);
  }

  register(def: WorkflowKind<any>): void {
    if (this.kinds.has(def.kind)) throw new Error(`Workflow kind '${def.kind}' is already registered`);
    this.kinds.set(def.kind, def);
    this.pump();
  }

  // ── Reading ───────────────────────────────────────────────────────────────

  private row(id: string, userId?: string): Row {
    const r = this.db.prepare('SELECT * FROM workflow_jobs WHERE id = ?').get(id) as Row | undefined;
    if (!r || (userId !== undefined && r.user_id !== userId)) throw new WorkflowError(404, `Workflow job ${id} not found`);
    return r;
  }

  get(id: string, userId?: string): WorkflowJob { return toJob(this.row(id, userId)); }

  list(userId: string, filter: { kind?: string; status?: WorkflowJobStatus } = {}): WorkflowJob[] {
    const rows = this.db.prepare(`SELECT * FROM workflow_jobs WHERE user_id = ?
      AND (? IS NULL OR kind = ?) AND (? IS NULL OR status = ?) ORDER BY created_at DESC, rowid DESC LIMIT 200`)
      .all(userId, filter.kind ?? null, filter.kind ?? null, filter.status ?? null, filter.status ?? null) as Row[];
    return rows.map(toJob);
  }

  /** The job plus every kept event after `after`. */
  replay(id: string, userId: string | undefined, after = 0): WorkflowReplay {
    const job = this.get(id, userId);
    const events = (this.db.prepare('SELECT seq, type, data, at FROM workflow_events WHERE job_id = ? AND seq > ? ORDER BY seq')
      .all(id, after) as Array<{ seq: number; type: string; data: string | null; at: number }>)
      .map(e => ({ seq: e.seq, type: e.type, data: e.data === null ? null : JSON.parse(e.data), at: e.at }));
    const gap = job.lastSeq > after && (events.length === 0 || events[0].seq > after + 1);
    return { job, events, gap };
  }

  /** Live events for one job, after they are stored. Returns unsubscribe. */
  subscribe(id: string, listener: (e: WorkflowEvent) => void): () => void {
    this.events.on(id, listener);
    return () => { this.events.off(id, listener); };
  }

  // ── Events ────────────────────────────────────────────────────────────────

  /** Store an event (inside the caller's transaction, if any). Publishing to
   *  live subscribers happens after the outermost transaction commits. */
  private append(id: string, type: string, data: unknown): WorkflowEvent {
    const text = data === undefined ? null : JSON.stringify(data);
    if (text && text.length > MAX_EVENT_BYTES) throw new Error(`Workflow event '${type}' is larger than ${MAX_EVENT_BYTES} bytes`);
    const at = this.now();
    const { last_seq: seq } = this.db.prepare('UPDATE workflow_jobs SET last_seq = last_seq + 1 WHERE id = ? RETURNING last_seq').get(id) as { last_seq: number };
    this.db.prepare('INSERT INTO workflow_events (job_id, seq, type, data, at) VALUES (?, ?, ?, ?, ?)').run(id, seq, type, text, at);
    this.db.prepare('DELETE FROM workflow_events WHERE job_id = ? AND seq <= ?').run(id, seq - MAX_EVENTS_PER_JOB);
    return { seq, type, data: data ?? null, at };
  }

  private publish(id: string, events: WorkflowEvent[]): void {
    for (const e of events) this.events.emit(id, e);
  }

  /** Run fn in a transaction; fn returns the events it appended for job id. */
  private change(id: string, fn: () => WorkflowEvent[]): void {
    const events = this.db.transaction(fn)();
    this.publish(id, events);
  }

  // ── Commands ──────────────────────────────────────────────────────────────

  submit(input: SubmitWorkflowJob, userId: string): { job: WorkflowJob; created: boolean } {
    const def = this.kinds.get(input.kind);
    if (!def) throw new WorkflowError(400, `Unknown workflow kind '${input.kind}'`);
    const parsed = def.input.safeParse(input.input);
    if (!parsed.success) {
      throw new WorkflowError(400, `Invalid input for ${input.kind}`, {
        issues: parsed.error.issues.map(i => ({ path: i.path.join('.'), message: i.message })),
      });
    }
    const captured = JSON.stringify(parsed.data);
    if (captured.length > MAX_INPUT_BYTES) throw new WorkflowError(400, `Workflow input is larger than ${MAX_INPUT_BYTES} bytes`);
    const hash = requestVersion(JSON.parse(captured));
    const find = () => this.db.prepare('SELECT * FROM workflow_jobs WHERE user_id = ? AND kind = ? AND idempotency_key = ?')
      .get(userId, input.kind, input.idempotencyKey) as Row | undefined;
    const same = (r: Row) => {
      if (r.input_hash !== hash) throw new WorkflowError(409, `Idempotency key '${input.idempotencyKey}' is already used by a different ${input.kind} input`);
      return { job: toJob(r), created: false as const };
    };
    const existing = find();
    if (existing) return same(existing);
    const id = randomUUID();
    const t = this.now();
    try {
      this.change(id, () => {
        this.db.prepare(`INSERT INTO workflow_jobs (id, user_id, kind, idempotency_key, status, input, input_hash, created_at, updated_at)
          VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)`).run(id, userId, input.kind, input.idempotencyKey, captured, hash, t, t);
        return [this.append(id, 'status', { status: 'pending' })];
      });
    } catch (err: any) {
      if (String(err?.code).startsWith('SQLITE_CONSTRAINT')) return same(find()!);
      throw err;
    }
    this.pump();
    return { job: this.get(id), created: true };
  }

  cancel(id: string, userId?: string): WorkflowJob {
    const r = this.row(id, userId);
    if (TERMINAL.includes(r.status)) throw new WorkflowError(409, `Workflow job is already ${r.status}`);
    if (r.cancel_requested === 1) return toJob(r);
    const t = this.now();
    this.change(id, () => {
      // Guarded: a start may have claimed it since the read above.
      const pending = this.db.prepare(`UPDATE workflow_jobs SET status = 'cancelled', cancel_requested = 1, finished_at = ?, updated_at = ?
        WHERE id = ? AND status = 'pending'`).run(t, t, id);
      if (pending.changes === 1) return [this.append(id, 'status', { status: 'cancelled' })];
      const running = this.db.prepare(`UPDATE workflow_jobs SET cancel_requested = 1, updated_at = ?
        WHERE id = ? AND status = 'running' AND cancel_requested = 0`).run(t, id);
      return running.changes === 1 ? [this.append(id, 'cancel-requested', {})] : [];
    });
    const now = this.row(id);
    if (now.status === 'running') {
      this.running.get(id)?.controller.abort(new WorkflowError(409, 'Cancelled'));
      for (const intentId of JSON.parse(now.audio_intent_ids) as string[]) {
        try { this.deps.audio.cancel(intentId); } catch { /* already finished */ }
      }
    }
    return toJob(now);
  }

  /** Put a failed, cancelled or interrupted job back as a new attempt. The
   *  only way a started job runs twice. */
  retry(id: string, userId?: string): WorkflowJob {
    const r = this.row(id, userId);
    if (!['failed', 'cancelled', 'interrupted'].includes(r.status)) {
      throw new WorkflowError(409, `Only a failed, cancelled or interrupted job can be retried; this one is ${r.status}`);
    }
    this.change(id, () => {
      const done = this.db.prepare(`UPDATE workflow_jobs SET status = 'pending', attempt = attempt + 1, cancel_requested = 0, claim_id = NULL,
        result = NULL, error = NULL, started_at = NULL, finished_at = NULL, updated_at = ? WHERE id = ? AND status = ?`)
        .run(this.now(), id, r.status);
      if (done.changes !== 1) throw new WorkflowError(409, 'Workflow job changed while retrying; reload it');
      return [this.append(id, 'status', { status: 'pending', attempt: r.attempt + 1 })];
    });
    this.pump();
    return this.get(id);
  }

  // ── Restart ───────────────────────────────────────────────────────────────

  /** Run once at startup, before any job starts. A job that was running when
   *  the process stopped is interrupted: its step died with the process and
   *  may have half-finished. Pending jobs never started, so they stay pending. */
  reconcileAfterRestart(): number {
    const rows = this.db.prepare(`SELECT id FROM workflow_jobs WHERE status = 'running'`).all() as Array<{ id: string }>;
    const t = this.now();
    const error = 'The server stopped while this job was running. Its last step may or may not have finished; retry it to run it again.';
    for (const { id } of rows) {
      this.change(id, () => {
        this.db.prepare(`UPDATE workflow_jobs SET status = 'interrupted', claim_id = NULL, error = ?, finished_at = ?, updated_at = ? WHERE id = ? AND status = 'running'`)
          .run(error, t, t, id);
        return [this.append(id, 'status', { status: 'interrupted', error })];
      });
    }
    return rows.length;
  }

  // ── Execution ─────────────────────────────────────────────────────────────

  /** Start every pending job whose kind has a free slot, oldest first. */
  pump(): void {
    const pending = this.db.prepare(`SELECT * FROM workflow_jobs WHERE status = 'pending' ORDER BY created_at, rowid`).all() as Row[];
    for (const r of pending) {
      const def = this.kinds.get(r.kind);
      if (!def) continue;
      const busy = [...this.running.values()].filter(x => x.kind === r.kind).length;
      if (busy >= (def.maxConcurrent ?? 1)) continue;
      const t = this.now();
      const claimId = randomUUID();
      let claimed = false;
      this.change(r.id, () => {
        const claim = this.db.prepare(`UPDATE workflow_jobs SET status = 'running', claim_id = ?, started_at = ?, updated_at = ?
          WHERE id = ? AND status = 'pending' AND cancel_requested = 0`).run(claimId, t, t, r.id);
        claimed = claim.changes === 1;
        return claimed ? [this.append(r.id, 'status', { status: 'running', attempt: r.attempt })] : [];
      });
      if (claimed) this.launch(def, r, claimId);
    }
  }

  /** True while `claimId` is the job's current run. Every write a step makes,
   *  and its completion, checks this, so a step that outlived its run (after
   *  a cancel, a timeout, or a restart and retry) cannot touch the job. */
  private owns(id: string, claimId: string): boolean {
    const r = this.row(id);
    return r.status === 'running' && r.claim_id === claimId;
  }

  private launch(def: WorkflowKind<any>, r: Row, claimId: string): void {
    const controller = new AbortController();
    this.running.set(r.id, { controller, kind: r.kind });
    const timeoutMs = def.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${Math.round(timeoutMs / 1000)} s`)), timeoutMs);
    timer.unref?.();
    const signal = controller.signal;
    const live = () => { if (!this.owns(r.id, claimId)) throw new Error(`Workflow job ${r.id} is no longer running this step`); };
    const throwIfCancelled = () => { if (signal.aborted) throw signal.reason; };
    const ctx: WorkflowContext<any> = {
      jobId: r.id, userId: r.user_id, input: JSON.parse(r.input), attempt: r.attempt, signal, throwIfCancelled,
      emit: (type, data) => this.change(r.id, () => { live(); return [this.append(r.id, type, data)]; }),
      audio: {
        enqueue: (key, request, meta) => {
          throwIfCancelled();
          live();
          const { item } = this.deps.audio.enqueue({
            idempotencyKey: `workflow:${r.id}:${key}`, request, meta: { ...meta, workflowJobId: r.id, workflowKey: key },
          }, r.user_id);
          this.change(r.id, () => {
            live();
            const ids = JSON.parse(this.row(r.id).audio_intent_ids) as string[];
            if (ids.includes(item.id)) return [];
            ids.push(item.id);
            this.db.prepare('UPDATE workflow_jobs SET audio_intent_ids = ?, updated_at = ? WHERE id = ?').run(JSON.stringify(ids), this.now(), r.id);
            return [this.append(r.id, 'audio', { intentId: item.id, key })];
          });
          return item;
        },
        wait: async intentId => {
          for (;;) {
            const item = this.deps.audio.get(intentId);
            if (AUDIO_TERMINAL.has(item.status)) return item;
            throwIfCancelled();
            await new Promise<void>((resolve, reject) => {
              const t = setTimeout(() => { signal.removeEventListener('abort', onAbort); resolve(); }, this.deps.audioPollMs ?? 1000);
              const onAbort = () => { clearTimeout(t); reject(signal.reason); };
              signal.addEventListener('abort', onAbort, { once: true });
            });
          }
        },
      },
    };
    type Outcome = { ok: true; result: unknown } | { ok: false; error: string };
    // A cancel or timeout ends the run at once, whether or not the step
    // listens to the signal. A step that ignores it keeps running in the
    // background, fenced off by owns(); its slot is free and its result unused.
    const aborted = new Promise<Outcome>(resolve => signal.addEventListener('abort',
      () => resolve({ ok: false, error: message(signal.reason) }), { once: true }));
    // Started synchronously, as a claim is followed by its first step at once.
    let started: Promise<unknown>;
    try { started = Promise.resolve(def.run(ctx)); } catch (err) { started = Promise.reject(err); }
    const stepped = started.then(
      (result): Outcome => ({ ok: true, result }),
      (err): Outcome => ({ ok: false, error: message(signal.aborted ? signal.reason : err) }));
    const run = Promise.race([stepped, aborted]).then(outcome => {
      clearTimeout(timer);
      if (this.running.get(r.id)?.controller === controller) this.running.delete(r.id);
      this.finish(r.id, claimId, outcome);
    }).finally(() => { this.runs.delete(claimId); this.pump(); });
    this.runs.set(claimId, run);
  }

  private finish(id: string, claimId: string, outcome: { ok: true; result: unknown } | { ok: false; error: string }): void {
    const t = this.now();
    this.change(id, () => {
      if (!this.owns(id, claimId)) return [];
      const end = (status: WorkflowJobStatus, result: string | null, error: string | null) => {
        this.db.prepare(`UPDATE workflow_jobs SET status = ?, result = ?, error = ?, claim_id = NULL, finished_at = ?, updated_at = ?
          WHERE id = ? AND claim_id = ?`).run(status, result, error, t, t, id, claimId);
        return [this.append(id, 'status', error ? { status, error } : { status })];
      };
      if (this.row(id).cancel_requested === 1) return end('cancelled', null, null);
      if (!outcome.ok) return end('failed', null, outcome.error);
      let result: string | null;
      try { result = outcome.result === undefined ? null : JSON.stringify(outcome.result); }
      catch (err) { return end('failed', null, `Result is not JSON: ${message(err)}`); }
      return end('succeeded', result, null);
    });
  }

  /** Resolves when every job running now has finished (tests, shutdown). */
  async settled(): Promise<void> {
    while (this.runs.size) await Promise.all([...this.runs.values()]);
  }
}
