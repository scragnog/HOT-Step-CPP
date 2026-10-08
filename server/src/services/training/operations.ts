// training/operations.ts — hooks for Node-owned training operations.
//
// Each domain (recipes, preparation, review) registers its own handlers here
// and they are served under /api/training/ops/<domain>/..., so no domain edits
// routes/training.ts. A handler accepts a command with acceptTrainingSnapshot:
// the envelope (contracts/trainingOperation.ts) is validated with the domain's
// payload schema, the worker it names is resolved for the capability the
// operation needs, and the dataset revision it was built from must still be
// current. What comes back is the captured snapshot the operation runs from.
//
// This is not a scheduler: a domain runs its work through the existing stage
// runners, workers and workflow jobs, from the captured snapshot.

import { Router, type Request, type Response, type NextFunction } from 'express';
import type { z } from 'zod/v4';
import {
  REMOTE_WORKER_CAPABILITIES, trainingSnapshotSchema,
  type TrainingOperationError, type TrainingSnapshot, type TrainingWorkerCapability, type TrainingWorkerRef,
} from '../../contracts/trainingOperation.js';
import { APP_VERSION } from '../../config.js';
import { getWorker, workerStatus, type WorkerInfo } from './trainingWorkers.js';
import { getDataset } from './datasetsRepo.js';

export class TrainingOperationFailure extends Error {
  constructor(readonly status: 400 | 404 | 409 | 503, readonly body: TrainingOperationError) { super(body.error); }
}

export type ResolvedTrainingWorker =
  | { ref: { kind: 'local' }; remote: null }
  | { ref: { kind: 'remote'; name: string }; remote: WorkerInfo };

/** What the hooks need from the rest of the server; tests pass fakes. */
export interface TrainingOperationDeps {
  getWorker(name: string): WorkerInfo | undefined;
  workerStatus(w: WorkerInfo): Promise<{ online: boolean; version?: string; error?: string }>;
  /** The dataset's current revision, or null when it does not exist. */
  datasetRevision(id: string): string | null;
  appVersion: string;
}

export const defaultTrainingOperationDeps = (): TrainingOperationDeps => ({
  getWorker,
  workerStatus,
  // training_datasets.updated_at moves on every edit of the row (labels,
  // counters, status). Opaque to clients: they echo it back.
  datasetRevision: id => getDataset(id)?.updatedAt ?? null,
  appVersion: APP_VERSION,
});

/** Resolve a worker reference for an operation needing `capability`. A
 *  remote worker must still be configured, online and on this version. */
export async function resolveTrainingWorker(ref: TrainingWorkerRef, capability: TrainingWorkerCapability, deps: TrainingOperationDeps): Promise<ResolvedTrainingWorker> {
  if (ref.kind === 'local') return { ref, remote: null };
  if (!REMOTE_WORKER_CAPABILITIES.includes(capability)) {
    throw new TrainingOperationFailure(409, { reason: 'unsupported-capability', error: `Training worker ${ref.name} cannot ${capability}; that runs on this machine` });
  }
  const w = deps.getWorker(ref.name);
  if (!w) throw new TrainingOperationFailure(409, { reason: 'unknown-worker', error: `No training worker named ${ref.name} is configured any more` });
  const s = await deps.workerStatus(w);
  if (!s.online) throw new TrainingOperationFailure(503, { reason: 'worker-offline', error: `Training worker ${ref.name} is offline${s.error ? `: ${s.error}` : ''}` });
  if (s.version !== deps.appVersion) {
    throw new TrainingOperationFailure(409, { reason: 'worker-version', error: `Training worker ${ref.name} runs ${s.version ?? 'an unknown version'}, this machine ${deps.appVersion}; update it first` });
  }
  return { ref, remote: w };
}

/** Validate an operation body and capture it: envelope plus the domain's
 *  payload schema, worker resolved for `capability`, dataset revision current.
 *  The returned snapshot (not the request) is what the operation runs from. */
export async function acceptTrainingSnapshot<P extends z.ZodType>(
  body: unknown, payload: P, capability: TrainingWorkerCapability, deps: TrainingOperationDeps,
): Promise<{ snapshot: TrainingSnapshot<z.infer<P>>; worker: ResolvedTrainingWorker }> {
  const parsed = trainingSnapshotSchema(payload).safeParse(body);
  if (!parsed.success) {
    throw new TrainingOperationFailure(400, {
      error: 'Invalid training operation',
      issues: parsed.error.issues.map(i => ({ path: i.path.join('.'), message: i.message })),
    });
  }
  const snapshot = parsed.data as TrainingSnapshot<z.infer<P>>;
  if (snapshot.dataset) {
    const current = deps.datasetRevision(snapshot.dataset.id);
    if (current === null) throw new TrainingOperationFailure(404, { reason: 'missing-dataset', error: `Dataset ${snapshot.dataset.id} not found` });
    if (current !== snapshot.dataset.revision) {
      throw new TrainingOperationFailure(409, { reason: 'stale-dataset', currentRevision: current, error: 'The dataset changed since this was prepared; reload it and try again' });
    }
  }
  const worker = await resolveTrainingWorker(snapshot.worker, capability, deps);
  return { snapshot, worker };
}

// ── Domain handlers ─────────────────────────────────────────────────────────

const domains = new Map<string, Router>();

/** Add a domain's routes under /api/training/ops/<domain>. `mount` gets its
 *  own router and the deps; it may be called at import, before the server
 *  starts. A domain registers once. */
export function registerTrainingOperations(domain: string, mount: (router: Router, deps: TrainingOperationDeps) => void, deps = defaultTrainingOperationDeps()): void {
  if (!/^[a-z][a-z0-9-]{0,40}$/.test(domain)) throw new Error(`Bad training operation domain '${domain}'`);
  if (domains.has(domain)) throw new Error(`Training operation domain '${domain}' is already registered`);
  const router = Router();
  mount(router, deps);
  domains.set(domain, router);
}

/** Wrap an async handler: a TrainingOperationFailure becomes its status and
 *  body, anything else a 500. */
export function trainingOp(fn: (req: Request, res: Response) => Promise<unknown>) {
  return (req: Request, res: Response) => {
    fn(req, res).then(value => { if (!res.headersSent) res.json(value); }, (err: unknown) => {
      if (res.headersSent) return;
      if (err instanceof TrainingOperationFailure) { res.status(err.status).json(err.body); return; }
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    });
  };
}

/** Mounted at /api/training/ops by routes/training.ts. Looks domains up per
 *  request, so registration order against the mount does not matter. */
export const trainingOperationsRouter = Router();
trainingOperationsRouter.use('/:domain', (req: Request, res: Response, next: NextFunction) => {
  const router = domains.get(String(req.params.domain));
  if (!router) { res.status(404).json({ error: `Unknown training operation domain '${req.params.domain}'` }); return; }
  router(req, res, next);
});
