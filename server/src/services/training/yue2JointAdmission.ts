// yue2JointAdmission.ts — start a joint-training job that a Node operation
// has already decided on, re-checking that decision at the moment of enqueue.
//
// The operation registers a `revalidate` guard and posts its request to the
// ordinary /yue2-joint-train route with a one-shot admission token. The route
// runs every check it always runs, then calls the guard synchronously right
// before queueing, so a cleanup or finish that landed in between is caught
// (409) rather than trained on. A token is single use and never leaves this
// process: an unknown or spent one is refused.
import { randomUUID } from 'crypto';
import { config } from '../../config.js';

export type JointAdmissionResult = { status: number; body: Record<string, unknown> };

const guards = new Map<string, () => void>();
// ponytail: grows by one entry per admitted continuation for the life of the
// process; prune by age if continuations ever become frequent.
const outcomes = new Map<string, Promise<JointAdmissionResult>>();

/** The route's side: take the guard for `token`, once. */
export function takeJointAdmission(token: unknown): (() => void) | undefined {
  if (typeof token !== 'string') return undefined;
  const guard = guards.get(token);
  guards.delete(token);
  return guard;
}

/** Run `guard`, mapping a throw to the 409 message the route returns. */
export function jointAdmissionError(guard: () => void): string | null {
  try { guard(); return null; } catch (err) {
    const body = (err as { body?: { error?: unknown } }).body;
    return typeof body?.error === 'string' ? body.error : err instanceof Error ? err.message : String(err);
  }
}

/** Post `body` to the joint-train route for `datasetId`, admitted only if
 *  `revalidate` still passes at enqueue. The same `idempotencyKey` returns the
 *  first successful (or in-flight) result instead of starting a second job. */
export function admitYue2JointTrain(opts: {
  datasetId: string; body: Record<string, unknown>; idempotencyKey: string; revalidate: () => void;
  post?: (url: string, body: Record<string, unknown>) => Promise<JointAdmissionResult>;
}): Promise<JointAdmissionResult> {
  const known = outcomes.get(opts.idempotencyKey);
  if (known) return known;
  const token = randomUUID();
  guards.set(token, opts.revalidate);
  const url = `http://127.0.0.1:${config.server.port}/api/training/datasets/${encodeURIComponent(opts.datasetId)}/yue2-joint-train`;
  const result = (opts.post ?? postJson)(url, { ...opts.body, admission: token })
    .finally(() => guards.delete(token));
  outcomes.set(opts.idempotencyKey, result);
  // Only a started job is remembered: a refused or failed attempt may be retried.
  result.then(r => { if (r.status !== 200) outcomes.delete(opts.idempotencyKey); },
    () => outcomes.delete(opts.idempotencyKey));
  return result;
}

async function postJson(url: string, body: Record<string, unknown>): Promise<JointAdmissionResult> {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return { status: res.status, body: await res.json().catch(() => ({ error: `HTTP ${res.status}` })) };
}
