// workflows/revisions.ts — optimistic revisions for drafts and projects.
//
// Every write names the revision it was based on. The write and the bump are
// one guarded UPDATE (`WHERE revision = ?`), so of two clients editing from
// the same revision exactly one lands and the other gets a 409 carrying the
// current revision; neither can silently overwrite the other. A late result
// (a render finishing after the user moved on) goes through the same check.
//
// Two forms:
//   - WorkflowDocuments: a generic JSON draft store (cover drafts, previews).
//   - bumpRevision: the same check on a domain table with its own columns and
//     a `revision INTEGER` column (builder_projects). Call it inside the
//     transaction that makes the domain change, so both land or neither does.

import { randomUUID } from 'node:crypto';
import type Database from 'better-sqlite3';
import type { WorkflowDocument } from '../../contracts/workflow.js';
import { WorkflowError } from './workflowJobs.js';

/** Tables bumpRevision may touch: names are interpolated into SQL. */
const REVISIONED_TABLES = new Set(['workflow_documents', 'builder_projects']);

export function revisionConflict(currentRevision: number): WorkflowError {
  return new WorkflowError(409, `Stale revision: this was changed elsewhere and is now at revision ${currentRevision}. Reload it and apply your change again.`, { currentRevision });
}

/** Advance `table`'s row `id` from `expected` to `expected + 1`, or throw a
 *  404 / 409 (with currentRevision). Returns the new revision. */
export function bumpRevision(db: Database.Database, table: string, id: string, expected: number, userId?: string): number {
  if (!REVISIONED_TABLES.has(table)) throw new Error(`Table '${table}' is not revisioned`);
  const owner = userId === undefined ? '' : ' AND user_id = ?';
  const ownerArgs = userId === undefined ? [] : [userId];
  const done = db.prepare(`UPDATE ${table} SET revision = revision + 1 WHERE id = ? AND revision = ?${owner}`).run(id, expected, ...ownerArgs);
  if (done.changes === 1) return expected + 1;
  const current = db.prepare(`SELECT revision FROM ${table} WHERE id = ?${owner}`).get(id, ...ownerArgs) as { revision: number } | undefined;
  if (!current) throw new WorkflowError(404, `${table} row ${id} not found`);
  throw revisionConflict(current.revision);
}

interface Row { id: string; user_id: string; kind: string; revision: number; data: string; created_at: number; updated_at: number }

export function ensureRevisionSchema(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS workflow_documents (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      revision INTEGER NOT NULL DEFAULT 1,
      data TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS workflow_documents_owner ON workflow_documents(user_id, kind, updated_at);
  `);
}

const toDoc = (r: Row): WorkflowDocument => ({
  id: r.id, kind: r.kind, revision: r.revision, data: JSON.parse(r.data), createdAt: r.created_at, updatedAt: r.updated_at,
});

export class WorkflowDocuments {
  private readonly now: () => number;

  constructor(private readonly db: Database.Database, now?: () => number) {
    this.now = now ?? Date.now;
    ensureRevisionSchema(db);
  }

  create(userId: string, kind: string, data: Record<string, unknown>): WorkflowDocument {
    const id = randomUUID();
    const t = this.now();
    this.db.prepare('INSERT INTO workflow_documents (id, user_id, kind, revision, data, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?, ?)')
      .run(id, userId, kind, JSON.stringify(data), t, t);
    return this.get(id, userId);
  }

  get(id: string, userId: string): WorkflowDocument {
    const r = this.db.prepare('SELECT * FROM workflow_documents WHERE id = ? AND user_id = ?').get(id, userId) as Row | undefined;
    if (!r) throw new WorkflowError(404, `Document ${id} not found`);
    return toDoc(r);
  }

  list(userId: string, kind: string): WorkflowDocument[] {
    return (this.db.prepare('SELECT * FROM workflow_documents WHERE user_id = ? AND kind = ? ORDER BY updated_at DESC LIMIT 200')
      .all(userId, kind) as Row[]).map(toDoc);
  }

  /** Replace the data if the document is still at expectedRevision. `data`
   *  may be a function of the stored data, run inside the same transaction
   *  (for server-side merges such as recording a late result). */
  update(id: string, userId: string, expectedRevision: number,
    data: Record<string, unknown> | ((current: Record<string, unknown>) => Record<string, unknown>)): WorkflowDocument {
    return this.db.transaction(() => {
      const current = this.get(id, userId);
      const next = typeof data === 'function' ? data(current.data) : data;
      bumpRevision(this.db, 'workflow_documents', id, expectedRevision, userId);
      this.db.prepare('UPDATE workflow_documents SET data = ?, updated_at = ? WHERE id = ?').run(JSON.stringify(next), this.now(), id);
      return this.get(id, userId);
    })();
  }

  remove(id: string, userId: string, expectedRevision: number): void {
    this.db.transaction(() => {
      bumpRevision(this.db, 'workflow_documents', id, expectedRevision, userId);
      this.db.prepare('DELETE FROM workflow_documents WHERE id = ?').run(id);
    })();
  }
}
