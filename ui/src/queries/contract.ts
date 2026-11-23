// contract.ts — project policy for TanStack Query in ui/src.
//
// These are HOT-Step's choices, not library defaults. client.ts applies them
// to every query and mutation; areas override per resource only with the
// helpers here. The full contract, with key schema and consumer manifest,
// is docs/dev/ui-server-state.md.

import type { Query } from '@tanstack/react-query';
import { ApiError } from '@/services/httpClient';

export const QUERY_STALE_MS = 30_000;
export const QUERY_GC_MS = 5 * 60_000;
/** Retries after the first failure, for network errors and 5xx only. */
export const MAX_QUERY_RETRIES = 2;
const RETRY_BASE_MS = 500;
const RETRY_MAX_MS = 4_000;
/** Active job and VRAM polling cadence. */
export const ACTIVE_POLL_MS = 1_500;
/** Ceiling for the doubling interval while a poll keeps failing. */
export const ERROR_POLL_MAX_MS = 30_000;

/** A request that was aborted, by Query or by the caller. Never retried.
 *  The axios path rejects with a CanceledError; the SSE and upload paths
 *  reject with an AbortError — both mean "the caller stopped asking". */
export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
    || (error instanceof Error && (error.name === 'AbortError' || error.name === 'CanceledError'));
}

/** Worth retrying: fetch's network TypeError, an XHR transport failure
 *  (ApiError status 0) or a 5xx. Never an abort or a 4xx: the server has
 *  answered, and asking again gets the same answer. */
export function isRetryableError(error: unknown): boolean {
  if (isAbortError(error)) return false;
  if (error instanceof ApiError) return error.status === 0 || error.status >= 500;
  return error instanceof TypeError;
}

export function retryQuery(failureCount: number, error: unknown): boolean {
  return failureCount < MAX_QUERY_RETRIES && isRetryableError(error);
}

/** Capped exponential: 500 ms, 1 s, 2 s, 4 s, 4 s... */
export function retryDelay(attempt: number): number {
  return Math.min(RETRY_BASE_MS * 2 ** attempt, RETRY_MAX_MS);
}

// errorUpdateCount at each query's last success. fetchFailureCount can't
// drive the backoff: it resets when every fetch starts, so it only ever
// counts one cycle's retries.
const failuresBefore = new WeakMap<object, number>();

/** A `refetchInterval` for a job or status poll. Polls every `ms` while
 *  `isActive(data)` holds and stops for good once it doesn't (terminal stop).
 *  While poll cycles keep failing (each after its own retries) the interval
 *  doubles per consecutive failed cycle, capped at ERROR_POLL_MAX_MS; the
 *  next success resets it. */
export function pollWhile<T>(isActive: (data: T | undefined) => boolean, ms = ACTIVE_POLL_MS) {
  return (query: Pick<Query<T>, 'state'>): number | false => {
    if (!isActive(query.state.data)) return false;
    const { status, errorUpdateCount } = query.state;
    if (status !== 'error') failuresBefore.set(query, errorUpdateCount);
    const failures = errorUpdateCount - (failuresBefore.get(query) ?? 0);
    return failures > 0 ? Math.min(ms * 2 ** failures, ERROR_POLL_MAX_MS) : ms;
  };
}
