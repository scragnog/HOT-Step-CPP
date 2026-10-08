// contracts/workflow.ts — wire schemas for durable workflow jobs and
// revisioned documents (/api/workflows). zod 4; clients import the types.
//
// A workflow job is one server-owned multi-step operation (a lyric refine
// sequence, an approved preview, a cover analysis). Its input is captured at
// submit and never re-read from the client. Each kind's own module defines
// what the input means; this contract only fixes the envelope.

import { z } from 'zod/v4';

/** POST /api/workflows/jobs */
export const submitWorkflowJobSchema = z.object({
  kind: z.string().min(1).max(100),
  /** Caller-chosen, unique per job within (user, kind). Repeating it with the
   *  same input returns the existing job; with a different input it is a 409. */
  idempotencyKey: z.string().min(1).max(200),
  input: z.record(z.string(), z.unknown()),
});
export type SubmitWorkflowJob = z.infer<typeof submitWorkflowJobSchema>;

export type WorkflowJobStatus =
  | 'pending'      // accepted, not started
  | 'running'      // claimed by this process; cancelRequested may be set
  | 'succeeded' | 'failed' | 'cancelled'
  | 'interrupted'; // the server stopped while it ran; never rerun automatically

export interface WorkflowJob {
  id: string;
  kind: string;
  idempotencyKey: string;
  status: WorkflowJobStatus;
  /** The captured input, as validated by the kind at submit. */
  input: Record<string, unknown>;
  attempt: number;
  /** Set when a cancel was accepted. A running job ends `cancelled` once its
   *  step stops; a result it produced after that is discarded. */
  cancelRequested: boolean;
  /** Audio queue items this job created, in order (see /api/audio-queue). */
  audioIntentIds: string[];
  result: unknown;
  error: string | null;
  /** Sequence number of the newest event. */
  lastSeq: number;
  createdAt: number;
  startedAt: number | null;
  finishedAt: number | null;
  updatedAt: number;
}

/** One entry in a job's event stream. `seq` starts at 1 and has no holes,
 *  but only the newest MAX_EVENTS_PER_JOB entries are kept. */
export interface WorkflowEvent {
  seq: number;
  /** 'status' carries {status, error?}; 'cancel-requested' acknowledges a
   *  cancel; 'audio' carries {intentId}. Kinds add their own types. */
  type: string;
  data: unknown;
  at: number;
}

/** GET /api/workflows/jobs/:id?after=N, and the first SSE frame's shape. */
export interface WorkflowReplay {
  job: WorkflowJob;
  events: WorkflowEvent[];
  /** True when events after the cursor were dropped by the bound: the client
   *  missed them and should rebuild its view from `job`. */
  gap: boolean;
}

export const MAX_EVENTS_PER_JOB = 500;

/** POST /api/workflows/documents */
export const createWorkflowDocumentSchema = z.object({
  kind: z.string().min(1).max(100),
  data: z.record(z.string(), z.unknown()),
});

/** PUT /api/workflows/documents/:id. The write lands only if the stored
 *  revision still equals expectedRevision; otherwise 409 with the current one. */
export const updateWorkflowDocumentSchema = z.object({
  expectedRevision: z.number().int().min(1),
  data: z.record(z.string(), z.unknown()),
});

export interface WorkflowDocument {
  id: string;
  kind: string;
  revision: number;
  data: Record<string, unknown>;
  createdAt: number;
  updatedAt: number;
}

/** Body of every 409 caused by a stale revision. */
export interface RevisionConflict {
  error: string;
  currentRevision: number;
}

// ── Typed documents (Batch 4 domains: presets, preferences, studio drafts) ──
//
// A typed kind is a WorkflowDocuments kind whose data is an envelope around a
// domain body: the body's schema version, where it came from, and the body.
// Domains register the kind (services/workflows/revisions.ts) and serve it
// from their own routes; the generic /documents routes refuse typed kinds so
// nothing bypasses the domain's validation.

/** 'user': the signed-in local user owns it, as every document today.
 *  'installation': one owner for the whole machine, for operational settings
 *  that are not per-user now. */
export type DocumentScope = 'user' | 'installation';

/** Owner id of installation-scoped documents. Never a real user id. */
export const INSTALLATION_OWNER = '@installation';

/** Largest serialized body a typed document accepts. */
export const MAX_TYPED_DOCUMENT_BYTES = 1024 * 1024;

/** Where a body came from. Set by the domain route from what it knows; the
 *  server stamps `at`. An import names the browser key and a hash of the
 *  value it read, which is also the import's receipt. */
export const documentProvenanceSchema = z.object({
  origin: z.enum(['client', 'import', 'server']),
  /** An id the client chose for itself (one browser profile), if any. */
  clientId: z.string().min(1).max(200).optional(),
  importedFrom: z.object({
    storageKey: z.string().min(1).max(500),
    sourceHash: z.string().min(1).max(200),
  }).optional(),
});
export type DocumentProvenance = z.infer<typeof documentProvenanceSchema> & { at: number };

/** A typed document as the domain routes return it. */
export interface TypedDocument<T> {
  id: string;
  kind: string;
  revision: number;
  /** The version `body` is at: always the kind's current version, since an
   *  older stored body is upgraded on read (the stored copy is unchanged
   *  until the next write). */
  schemaVersion: number;
  /** The version as stored, when it was upgraded on read. */
  storedSchemaVersion?: number;
  provenance: DocumentProvenance;
  body: T;
  createdAt: number;
  updatedAt: number;
}

/** One browser value imported once. Retrying the same key and hash returns
 *  the receipt instead of importing again, even if the document has since
 *  been deleted (documentId then names a document that no longer exists). */
export interface DocumentImportReceipt {
  kind: string;
  storageKey: string;
  sourceHash: string;
  documentId: string;
  importedAt: number;
}

/** Body of the 409 for a stored body this server cannot read: a newer
 *  version, or an older one with no upgrade. Nothing is deleted. */
export interface UnsupportedDocumentVersion {
  error: string;
  reason: 'unsupported-version';
  /** The document that could not be read. */
  documentId: string;
  schemaVersion: number;
  supportedVersion: number;
}
