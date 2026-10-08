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
