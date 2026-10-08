// trainingOperations.ts — client for Node-owned training operations
// (/api/training/ops/<domain>, server/src/services/training/operations.ts).
//
// Operations always go to this machine (never through API_BASE, which follows
// "Train on"). The worker is part of the command: snapshotFor() reads the
// current "Train on" choice once, when the command is built, and the server
// keeps that captured worker whatever this or another client selects later.
import type {
  TrainingDatasetRef, TrainingOperationError, TrainingSnapshot, TrainingSourceRef, TrainingWorkerRef,
} from '../../../server/src/contracts/trainingOperation';
import { getTrainingWorker } from './trainingApi';

export type { TrainingSnapshot, TrainingWorkerRef } from '../../../server/src/contracts/trainingOperation';

export class TrainingOperationRequestError extends Error {
  readonly status: number;
  readonly reason?: TrainingOperationError['reason'];
  readonly currentRevision?: string;
  readonly issues?: TrainingOperationError['issues'];
  constructor(status: number, body: TrainingOperationError) {
    super(body.error);
    this.status = status;
    this.reason = body.reason;
    this.currentRevision = body.currentRevision;
    this.issues = body.issues;
  }
}

/** The worker "Train on" points at now, as an explicit reference. */
export const currentWorkerRef = (): TrainingWorkerRef => {
  const name = getTrainingWorker();
  return name ? { kind: 'remote', name } : { kind: 'local' };
};

/** Build an operation envelope. `worker` defaults to "Train on" now. */
export function snapshotFor<P>(op: {
  kind: string; idempotencyKey: string; payload: P;
  worker?: TrainingWorkerRef; dataset?: TrainingDatasetRef; sources?: TrainingSourceRef[];
}): TrainingSnapshot<P> {
  return {
    version: 1,
    operation: { kind: op.kind, idempotencyKey: op.idempotencyKey },
    worker: op.worker ?? currentWorkerRef(),
    ...(op.dataset ? { dataset: op.dataset } : {}),
    sources: op.sources ?? [],
    payload: op.payload,
  };
}

/** Call a domain's operation route on this machine. Throws
 *  TrainingOperationRequestError with the server's status and reason. */
export async function trainingOperation<T>(domain: string, path: string,
  init: { method?: string; body?: unknown; signal?: AbortSignal; token?: string } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  if (init.body !== undefined) headers['Content-Type'] = 'application/json';
  // Training routes take no token; send one for a domain that runs workflow
  // jobs, which belong to a user.
  if (init.token) headers.Authorization = `Bearer ${init.token}`;
  const res = await fetch(`/api/training/ops/${encodeURIComponent(domain)}${path}`, {
    method: init.method ?? (init.body === undefined ? 'GET' : 'POST'),
    headers,
    body: init.body === undefined ? undefined : JSON.stringify(init.body),
    signal: init.signal,
  });
  const value = await res.json().catch(() => ({}));
  if (!res.ok) throw new TrainingOperationRequestError(res.status, { error: value.error || `Training operation failed (${res.status})`, ...value });
  return value as T;
}
