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
import type { z } from 'zod/v4';
import {
  INSTALLATION_OWNER, MAX_TYPED_DOCUMENT_BYTES, documentProvenanceSchema,
  type DocumentImportReceipt, type DocumentProvenance, type DocumentScope, type TypedDocument, type WorkflowDocument,
} from '../../contracts/workflow.js';
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
    CREATE TABLE IF NOT EXISTS document_import_receipts (
      user_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      storage_key TEXT NOT NULL,
      source_hash TEXT NOT NULL,
      document_id TEXT NOT NULL,
      imported_at INTEGER NOT NULL,
      PRIMARY KEY (user_id, kind, storage_key, source_hash)
    );
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

// ── Typed documents ─────────────────────────────────────────────────────────
//
// A domain kind (a preset family, a studio draft) on top of WorkflowDocuments:
// same table, same revision check, with the data held as an envelope
// { schemaVersion, provenance, body }. The domain supplies the body schema;
// this layer owns scope, versioning, provenance and import receipts.

export interface TypedDocumentKind<T extends Record<string, unknown>> {
  kind: string;
  scope: DocumentScope;
  /** The version written by this build. Starts at 1. */
  schemaVersion: number;
  /** The body at schemaVersion. */
  schema: z.ZodType<T>;
  /** Bring a body stored at an older version up to schemaVersion. Without it
   *  only schemaVersion is readable. Runs on read; the stored copy changes
   *  only when the document is next written. */
  upgrade?: (fromVersion: number, body: unknown) => unknown;
}

const typedKinds = new Map<string, TypedDocumentKind<any>>();

/** Declare a typed kind. Domains call this at import; it touches no database.
 *  A kind registers once. */
export function registerDocumentKind<T extends Record<string, unknown>>(def: TypedDocumentKind<T>): TypedDocumentKind<T> {
  if (!/^[a-z][a-z0-9-]{0,60}(\.[a-z0-9-]{1,60})*$/.test(def.kind)) throw new Error(`Bad document kind '${def.kind}'`);
  if (!Number.isInteger(def.schemaVersion) || def.schemaVersion < 1) throw new Error(`Bad schemaVersion for '${def.kind}'`);
  if (typedKinds.has(def.kind)) throw new Error(`Document kind '${def.kind}' is already registered`);
  typedKinds.set(def.kind, def);
  return def;
}

/** True for a kind served only by its domain's routes. */
export function isTypedDocumentKind(kind: string): boolean {
  return typedKinds.has(kind);
}

interface Envelope { schemaVersion: number; provenance: DocumentProvenance; body: unknown }

const invalidBody = (kind: string, issues: Array<{ path: PropertyKey[]; message: string }>) =>
  new WorkflowError(400, `Invalid ${kind}`, { issues: issues.map(i => ({ path: i.path.map(String).join('.'), message: i.message })) });

export class TypedDocuments<T extends Record<string, unknown>> {
  constructor(private readonly docs: WorkflowDocuments, private readonly db: Database.Database,
    readonly def: TypedDocumentKind<T>, private readonly now: () => number = Date.now) {
    ensureRevisionSchema(db);
  }

  /** Who owns this kind's documents for a request by `userId`. */
  owner(userId: string): string {
    return this.def.scope === 'installation' ? INSTALLATION_OWNER : userId;
  }

  create(userId: string, body: unknown, provenance: unknown): TypedDocument<T> {
    return this.read(this.docs.create(this.owner(userId), this.def.kind, this.envelope(body, provenance)));
  }

  get(id: string, userId: string): TypedDocument<T> {
    return this.read(this.mine(id, userId));
  }

  /** A document this build cannot read fails the list with its 409 (naming
   *  it), rather than the list quietly looking empty or complete. */
  list(userId: string): TypedDocument<T>[] {
    return this.docs.list(this.owner(userId), this.def.kind).map(doc => this.read(doc));
  }

  /** Replace the body if the document is still at expectedRevision. `body`
   *  may be a function of the current (upgraded) body, run in the same
   *  transaction. */
  update(id: string, userId: string, expectedRevision: number,
    body: unknown | ((current: T) => unknown), provenance: unknown): TypedDocument<T> {
    // A body this build cannot read is never overwritten: that would lose a
    // newer version's content.
    this.read(this.mine(id, userId));
    return this.read(this.docs.update(id, this.owner(userId), expectedRevision, current =>
      this.envelope(typeof body === 'function' ? (body as (c: T) => unknown)(this.unwrap(current).body) : body, provenance)));
  }

  remove(id: string, userId: string, expectedRevision: number): void {
    this.mine(id, userId);
    this.docs.remove(id, this.owner(userId), expectedRevision);
  }

  /** Import one browser value once. The receipt and the document are written
   *  in one transaction, so a failed import leaves neither and can be retried;
   *  a repeated import of the same key and hash returns the first receipt
   *  (and its document, unless that has since been deleted). */
  importOnce(userId: string, source: { storageKey: string; sourceHash: string }, body: unknown, provenance: Record<string, unknown> = {}):
  { receipt: DocumentImportReceipt; document: TypedDocument<T> | null; created: boolean } {
    const owner = this.owner(userId);
    return this.db.transaction(() => {
      const known = this.db.prepare('SELECT * FROM document_import_receipts WHERE user_id = ? AND kind = ? AND storage_key = ? AND source_hash = ?')
        .get(owner, this.def.kind, source.storageKey, source.sourceHash) as ReceiptRow | undefined;
      if (known) {
        let document: TypedDocument<T> | null = null;
        try { document = this.get(known.document_id, userId); } catch (err) { if (!(err instanceof WorkflowError) || err.status !== 404) throw err; }
        return { receipt: toReceipt(known), document, created: false };
      }
      const document = this.create(userId, body, { ...provenance, origin: 'import', importedFrom: source });
      const at = this.now();
      this.db.prepare('INSERT INTO document_import_receipts (user_id, kind, storage_key, source_hash, document_id, imported_at) VALUES (?, ?, ?, ?, ?, ?)')
        .run(owner, this.def.kind, source.storageKey, source.sourceHash, document.id, at);
      return { receipt: { kind: this.def.kind, ...source, documentId: document.id, importedAt: at }, document, created: true };
    })();
  }

  receipts(userId: string): DocumentImportReceipt[] {
    return (this.db.prepare('SELECT * FROM document_import_receipts WHERE user_id = ? AND kind = ? ORDER BY imported_at')
      .all(this.owner(userId), this.def.kind) as ReceiptRow[]).map(toReceipt);
  }

  /** The document, owned by this user (or the installation) and of this kind. */
  private mine(id: string, userId: string): WorkflowDocument {
    const doc = this.docs.get(id, this.owner(userId));
    if (doc.kind !== this.def.kind) throw new WorkflowError(404, `Document ${id} not found`);
    return doc;
  }

  private envelope(body: unknown, provenance: unknown): Record<string, unknown> {
    const parsedBody = this.def.schema.safeParse(body);
    if (!parsedBody.success) throw invalidBody(this.def.kind, parsedBody.error.issues);
    const parsedProvenance = documentProvenanceSchema.safeParse(provenance);
    if (!parsedProvenance.success) throw invalidBody(`${this.def.kind} provenance`, parsedProvenance.error.issues);
    if (Buffer.byteLength(JSON.stringify(parsedBody.data)) > MAX_TYPED_DOCUMENT_BYTES)
      throw new WorkflowError(400, `${this.def.kind} is larger than ${MAX_TYPED_DOCUMENT_BYTES} bytes`);
    return { schemaVersion: this.def.schemaVersion, provenance: { ...parsedProvenance.data, at: this.now() }, body: parsedBody.data };
  }

  /** The stored envelope, upgraded to the current version, or a 409 that
   *  leaves the stored document alone. */
  private unwrap(data: Record<string, unknown>): Envelope & { body: T; storedSchemaVersion?: number } {
    const env = data as unknown as Envelope;
    const stored = env.schemaVersion;
    const unsupported = () => new WorkflowError(409, `This ${this.def.kind} was saved as version ${stored}, which this build cannot read (it reads version ${this.def.schemaVersion}). It has not been changed.`,
      { reason: 'unsupported-version', schemaVersion: stored, supportedVersion: this.def.schemaVersion });
    if (!Number.isInteger(stored) || stored > this.def.schemaVersion) throw unsupported();
    let body = env.body;
    if (stored < this.def.schemaVersion) {
      if (!this.def.upgrade) throw unsupported();
      body = this.def.upgrade(stored, body);
    }
    const parsed = this.def.schema.safeParse(body);
    if (!parsed.success) throw unsupported();
    return { schemaVersion: this.def.schemaVersion, provenance: env.provenance, body: parsed.data,
      ...(stored !== this.def.schemaVersion ? { storedSchemaVersion: stored } : {}) };
  }

  private read(doc: WorkflowDocument): TypedDocument<T> {
    let env: ReturnType<TypedDocuments<T>['unwrap']>;
    try { env = this.unwrap(doc.data); } catch (err) {
      if (err instanceof WorkflowError) err.extra.documentId = doc.id;
      throw err;
    }
    return { id: doc.id, kind: doc.kind, revision: doc.revision, ...env, createdAt: doc.createdAt, updatedAt: doc.updatedAt };
  }
}

interface ReceiptRow { user_id: string; kind: string; storage_key: string; source_hash: string; document_id: string; imported_at: number }
const toReceipt = (r: ReceiptRow): DocumentImportReceipt => ({
  kind: r.kind, storageKey: r.storage_key, sourceHash: r.source_hash, documentId: r.document_id, importedAt: r.imported_at,
});
