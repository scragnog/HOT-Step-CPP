// contracts/trainingOperation.ts — the shared envelope for Node-owned training
// operations (/api/training/ops/<domain>/...). zod 4; clients import the types.
//
// A training command names everything it depends on when it is accepted, and
// runs from that captured copy:
//   - operation: what it is and its idempotency key;
//   - worker: the machine it runs on, chosen explicitly. The browser's
//     "Train on" selection is only where the client reads the default from;
//     once accepted, a command keeps its worker whatever any client selects
//     later, and operations never travel through the worker proxy;
//   - dataset/sources: the revisions it was built from, checked at accept;
//   - payload: the domain's own body (recipe, preparation, review), defined
//     and versioned by the slice that owns it, not here.

import { z } from 'zod/v4';

export const TRAINING_SNAPSHOT_VERSION = 1;

export const trainingWorkerRefSchema = z.discriminatedUnion('kind', [
  /** This machine. */
  z.object({ kind: z.literal('local') }),
  /** A training worker named in TRAINING_WORKERS. */
  z.object({ kind: z.literal('remote'), name: z.string().min(1).max(100) }),
]);
export type TrainingWorkerRef = z.infer<typeof trainingWorkerRefSchema>;

/** What an operation needs from its worker. A remote worker prepares, trains
 *  and renders previews; review, scoring and finishing always run on this
 *  machine against its own index (services/training/trainingWorkers.ts). */
export const TRAINING_WORKER_CAPABILITIES = ['prepare', 'train', 'preview', 'review'] as const;
export type TrainingWorkerCapability = typeof TRAINING_WORKER_CAPABILITIES[number];
export const REMOTE_WORKER_CAPABILITIES: readonly TrainingWorkerCapability[] = ['prepare', 'train', 'preview'];

/** A dataset as the client saw it. `revision` is opaque: compare, never parse. */
export const trainingDatasetRefSchema = z.object({
  id: z.string().min(1).max(200),
  revision: z.string().min(1).max(200),
});
export type TrainingDatasetRef = z.infer<typeof trainingDatasetRefSchema>;

/** Any other input the operation was built from (a run, checkpoint, preset),
 *  with the revision the client saw. The owning domain checks these. */
export const trainingSourceRefSchema = z.object({
  kind: z.string().min(1).max(50),
  id: z.string().min(1).max(200),
  revision: z.string().min(1).max(200),
});
export type TrainingSourceRef = z.infer<typeof trainingSourceRefSchema>;

export const trainingOperationIdSchema = z.object({
  kind: z.string().min(1).max(100),
  /** Unique per operation within (user, kind); a repeat returns the first. */
  idempotencyKey: z.string().min(1).max(200),
});

/** The envelope for one domain's payload schema. */
export function trainingSnapshotSchema<P extends z.ZodType>(payload: P) {
  return z.object({
    version: z.literal(TRAINING_SNAPSHOT_VERSION),
    operation: trainingOperationIdSchema,
    worker: trainingWorkerRefSchema,
    dataset: trainingDatasetRefSchema.optional(),
    sources: z.array(trainingSourceRefSchema).max(32).default([]),
    payload,
  });
}

export interface TrainingSnapshot<P> {
  version: typeof TRAINING_SNAPSHOT_VERSION;
  operation: { kind: string; idempotencyKey: string };
  worker: TrainingWorkerRef;
  dataset?: TrainingDatasetRef;
  sources: TrainingSourceRef[];
  payload: P;
}

/** Body of a refused operation. `reason` is set for worker and revision
 *  refusals; `currentRevision` for a stale dataset. */
export interface TrainingOperationError {
  error: string;
  reason?: 'unknown-worker' | 'worker-offline' | 'worker-version' | 'unsupported-capability' | 'stale-dataset' | 'missing-dataset';
  currentRevision?: string;
  issues?: Array<{ path: string; message: string }>;
}
