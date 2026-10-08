import { z } from 'zod/v4';
import type { WorkflowDocument } from './workflow.js';

export const reviewPickSchema = z.object({
  datasetId: z.string().min(1),
  runId: z.string().min(1),
  runRevision: z.number().int().nonnegative(),
  reviewRevision: z.string().length(64),
  step: z.number().int().nonnegative(),
  checkpointDir: z.string().min(1),
  blindLabel: z.string().optional(),
});
export const reviewFinishSchema = z.object({
  entries: z.array(reviewPickSchema.extend({ datasetRevision: z.string().min(1) })).min(1).max(32),
  knee: z.boolean().default(true),
});
export const reviewCleanupSchema = reviewPickSchema.extend({
  choice: z.object({ caches: z.boolean(), otherCheckpoints: z.boolean(), otherRuns: z.boolean(),
    resume: z.boolean(), otherPreviews: z.boolean() }),
});

export const auditionDraftSchema = z.object({
  datasetId: z.string().min(1),
  previewId: z.string().uuid(),
  slot: z.enum(['base', 'adapter']),
  cell: z.enum(['bare', 'adapter']),
});
export const reviewUseSchema = reviewPickSchema.extend({
  narFurther: z.boolean(),
  nar: z.object({ budget: z.number().int().min(10), lrScale: z.number().min(0.05).max(1),
    keepDelta: z.number().min(0), target: z.number().min(0).max(10).nullable(), knee: z.boolean() }),
});
// ── Responses of /api/training/ops/review/* (docs/dev/frontend-training.md) ──
// Described here so a client needs no service import; trainingStarts.test.ts
// checks the service types still fit them.

/** One checkpoint rung of a YuE2 joint run, as review scores it. */
export interface ReviewRungFact {
  step: number;
  dir: string;
  blindLabel: string;
  likeness: number | null;
  corruption: number | null;
  overall: number | null;
  replansPerTake: number;
  previewCount: number;
  planFlags: number;
}

/** GET /ladder/:datasetId/:runId. Echo `dataset.revision`, `runRevision` and
 *  `reviewRevision` back in any later command about this ladder. */
export interface ReviewLadderResponse {
  dataset: { id: string; revision: string | null };
  run: string;
  runRevision: number;
  reviewRevision: string;
  status: 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';
  finished: boolean;
  /** The worker a mirrored run trained on; null for a run on this machine. */
  origin: string | null;
  method: unknown;
  freezePlannerNow: boolean;
  facts: ReviewRungFact[];
  best: ReviewRungFact | null;
}

/** POST /finish-batch. `batch` is the queued finish batch (GET /api/training/yue2-batch/:id). */
export interface ReviewFinishResponse {
  batch: { id: string; status: string; items: unknown[]; currentDatasetId: string | null; lyricTiming: boolean };
  selections: Array<{ runId: string; step: number }>;
}

/** POST /decision. `chain` started decoder training as `jobId`; `skip` needs
 *  nothing; `reject` means a mirrored worker run cannot take a decoder pass here. */
export type ReviewDecisionResponse =
  | { outcome: 'skip' }
  | { outcome: 'reject'; worker?: string }
  | { outcome: 'chain'; jobId: string };

export interface ReviewCleanupItem { count: number; bytes: number; detail?: string[] }
/** What POST /cleanup would delete, by category. Selecting writes nothing. */
export interface ReviewCleanupPlan {
  run: string; step: number; keep: string;
  caches: ReviewCleanupItem;
  otherCheckpoints: ReviewCleanupItem;
  otherRuns: ReviewCleanupItem;
  resume: ReviewCleanupItem;
  otherPreviews: ReviewCleanupItem;
}

/** POST /select: links the rung as the dataset's adapter (a side effect) and
 *  returns the cleanup plan (a preview; nothing is deleted). */
export interface ReviewSelectResponse { linked: true; alreadyLinked: boolean; plan: ReviewCleanupPlan }

/** POST /cleanup: deletes what `choice` selects. */
export interface ReviewCleanupResponse { freedBytes: number; done: string[]; finishError?: string }

/** POST /draft: a Create draft mirroring one audition render. Repeating the
 *  idempotency key returns the same draft. */
export interface AuditionDraftResponse { draftId: string; revision: number }
/** GET /draft/:id. */
export interface AuditionDraftDocument { draft: WorkflowDocument }
