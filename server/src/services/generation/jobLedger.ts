// jobLedger.ts — what happened to a POST /api/generate job, kept past a restart.
//
// The live job (progress, stage, attempts) lives only in routes/generate.ts's
// in-memory map. A server restart empties that map, and a client polling
// /status for a job that was running would get a bare 404 and lose track of
// it. This ledger keeps one row per job in the app database: written when the
// job is queued and again when it ends. The first use in a process marks every
// row still open as interrupted, because its job died with the old process.
// Nothing is ever resubmitted from here.
import type Database from 'better-sqlite3';
import type { GenerationJob } from './jobTypes.js';

/** Kept for a week; long enough for any client still polling. */
const KEEP_MS = 7 * 24 * 60 * 60 * 1000;

export const INTERRUPTED_ERROR =
  'Generation failed: the server restarted while this job was queued or running. It was not resubmitted.';

export interface LedgerEntry {
  id: string;
  userId: string;
  status: 'open' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';
  error: string | null;
  result: GenerationJob['result'] | null;
  createdAt: number;
  finishedAt: number | null;
}

interface Row {
  id: string; user_id: string; status: LedgerEntry['status']; error: string | null;
  result_json: string | null; created_at: number; finished_at: number | null;
}

export class GenerationJobLedger {
  constructor(private readonly db: Database.Database, private readonly now: () => number = Date.now) {
    db.exec(`
      CREATE TABLE IF NOT EXISTS generation_jobs (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        status      TEXT NOT NULL,
        error       TEXT,
        result_json TEXT,
        created_at  INTEGER NOT NULL,
        finished_at INTEGER
      )`);
    const at = this.now();
    // Every open row belongs to a process that has gone: this runs once, before
    // this process queues anything.
    db.prepare(`UPDATE generation_jobs SET status = 'interrupted', error = ?, finished_at = ? WHERE status = 'open'`)
      .run(INTERRUPTED_ERROR, at);
    db.prepare('DELETE FROM generation_jobs WHERE created_at < ?').run(at - KEEP_MS);
  }

  opened(job: Pick<GenerationJob, 'id' | 'userId' | 'createdAt'>): void {
    this.db.prepare(`INSERT OR REPLACE INTO generation_jobs (id, user_id, status, created_at) VALUES (?, ?, 'open', ?)`)
      .run(job.id, job.userId ?? '', job.createdAt);
  }

  /** Record a job's terminal state; a job still active is left open. */
  finished(job: Pick<GenerationJob, 'id' | 'status' | 'error' | 'result'>): void {
    if (job.status !== 'succeeded' && job.status !== 'failed' && job.status !== 'cancelled') return;
    this.db.prepare(`UPDATE generation_jobs SET status = ?, error = ?, result_json = ?, finished_at = ? WHERE id = ?`)
      .run(job.status, job.error ?? null, job.status === 'succeeded' && job.result ? JSON.stringify(job.result) : null,
        this.now(), job.id);
  }

  get(id: string): LedgerEntry | null {
    const row = this.db.prepare('SELECT * FROM generation_jobs WHERE id = ?').get(id) as Row | undefined;
    if (!row) return null;
    return {
      id: row.id, userId: row.user_id, status: row.status, error: row.error,
      result: row.result_json ? JSON.parse(row.result_json) : null,
      createdAt: row.created_at, finishedAt: row.finished_at,
    };
  }
}

/** The /status body for a job the live map no longer holds. An interrupted
 *  job reports `failed` with `interrupted: true`, because every client already
 *  treats `failed` as terminal and shows its error. The error text contains
 *  "failed" on purpose: the browser queue only stops polling on such messages. */
export function ledgerStatusBody(entry: LedgerEntry): Record<string, unknown> {
  const interrupted = entry.status === 'interrupted' || entry.status === 'open';
  const status = interrupted ? 'failed' : entry.status;
  return {
    jobId: entry.id,
    status,
    stage: interrupted ? 'Interrupted' : status === 'succeeded' ? 'Complete' : status === 'cancelled' ? 'Cancelled' : 'Failed',
    progress: status === 'succeeded' ? 100 : 0,
    result: entry.result ?? undefined,
    error: interrupted ? (entry.error ?? INTERRUPTED_ERROR) : entry.error ?? undefined,
    interrupted,
    attempts: [],
    ace_job_id: null,
    ace_phase: null,
    ace_phase_progress: null,
    batch: null,
    mm3_streaming: false,
    mm3_interleaved: null,
    mm3_duration: null,
    mm3_takes: 1,
    mm3_take_seeds: null,
    mm3_ending: null,
  };
}
