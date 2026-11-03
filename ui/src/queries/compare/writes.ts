// writes.ts — Compare studio commands and their cache follow-up.
//
// Scoring is optimistic: the take updates at once, the session's list summary
// follows, and a failed PUT restores both. The server result then wins for
// status/archived/counts (the same rule the old store applied). Session-level
// commands patch the returned session/summary; only create/delete refetch the
// list, because their membership changed.

import {
  useMutation, useQueryClient, type QueryClient, type QueryKey,
} from '@tanstack/react-query';
import {
  compareApi,
  type CompareSession, type CompareSessionSummary, type CreateSessionInput,
  type ScoreInput, type ScoreResult,
} from '../../services/compareApi';
import { resourceKey } from '../keys';
import { queryManager } from '../invalidation';
import { CMP, compareQueries } from './queries';

// ── Pure helpers (exported for testing) ──────────────────────────────────────

/** Local status estimate, same rule as the server: a take counts as scored if
 *  any criterion has a value, keep flag, or non-empty note. */
export function localStatus(session: Pick<CompareSession, 'takes'>): { status: CompareSessionSummary['status']; scored: number; total: number } {
  const total = session.takes.length;
  const scored = session.takes.filter(t =>
    t.scores.some(s => s.value !== null || s.keep !== null || (s.note ?? '').length > 0)).length;
  const status = total > 0 && scored >= total ? 'scored' : scored > 0 ? 'partial' : 'unscored';
  return { status, scored, total };
}

/** Apply score inputs to a session immutably. Fields absent from an input are left alone. */
export function applyScores(session: CompareSession, items: ScoreInput[]): CompareSession {
  const takes = session.takes.map(t => {
    const mine = items.filter(i => i.songId === t.songId);
    if (!mine.length) return t;
    const scores = [...t.scores];
    for (const it of mine) {
      const idx = scores.findIndex(s => s.criterion === it.criterion);
      const base = idx >= 0 ? scores[idx] : { criterion: it.criterion, value: null, keep: null, note: '' };
      const next = { ...base };
      if ('value' in it) next.value = it.value ?? null;
      if ('keep' in it) next.keep = it.keep ?? null;
      if ('note' in it) next.note = it.note ?? '';
      if (idx >= 0) scores[idx] = next; else scores.push(next);
    }
    return { ...t, scores };
  });
  const s = { ...session, takes };
  // Optimistic: mirror the server auto-archive rule (never un-archives).
  const ls = localStatus(s);
  return { ...s, ...ls, archived: session.archived || ls.status === 'scored' };
}

/** Server result wins for status, archived and counts. */
export function reconcileScoreResult<T extends CompareSessionSummary>(session: T, r: ScoreResult): T {
  return { ...session, status: r.status, archived: r.archived, scored: r.scored, total: r.total };
}

/** The list row for a session (drops criteria and takes). */
export function summaryOf(session: CompareSession): CompareSessionSummary {
  const { criteria: _c, takes: _t, ...rest } = session;
  void _c; void _t;
  return rest;
}

// ── Cache follow-up (pure enough to test without React) ─────────────────────

const sessionsPrefix = () => resourceKey(CMP.sessions);

function patchSessionInLists(client: QueryClient, session: CompareSession) {
  const summary = summaryOf(session);
  client.setQueriesData<{ sessions: CompareSessionSummary[] }>(
    { queryKey: sessionsPrefix() },
    old => old && ({ sessions: old.sessions.map(s => (s.id === session.id ? summary : s)) }),
  );
}

/** Replace the cached session and its list row. Used by commands whose
 *  response is the whole session (add/remove take). */
export function patchSession(client: QueryClient, session: CompareSession) {
  client.setQueryData(compareQueries.session(session.id).queryKey, session);
  patchSessionInLists(client, session);
}

/** Merge an archive response into the cached session and list row. */
export function patchSessionSummary(client: QueryClient, summary: CompareSessionSummary) {
  client.setQueryData<CompareSession>(compareQueries.session(summary.id).queryKey, old => old && ({ ...old, ...summary }));
  client.setQueriesData<{ sessions: CompareSessionSummary[] }>(
    { queryKey: sessionsPrefix() },
    old => old && ({ sessions: old.sessions.map(s => (s.id === summary.id ? { ...s, ...summary } : s)) }),
  );
}

export interface ScoreRollback {
  session: CompareSession | undefined;
  lists: Array<[QueryKey, { sessions: CompareSessionSummary[] } | undefined]>;
}

/** Apply scores to the cache before the PUT; returns what to restore. */
export async function beginOptimisticScores(
  client: QueryClient, sessionId: string, items: ScoreInput[],
): Promise<ScoreRollback> {
  const key = compareQueries.session(sessionId).queryKey;
  await client.cancelQueries({ queryKey: key });
  const session = client.getQueryData<CompareSession>(key);
  const lists = client.getQueriesData<{ sessions: CompareSessionSummary[] }>({ queryKey: sessionsPrefix() });
  if (!session) return { session: undefined, lists: [] };
  const next = applyScores(session, items);
  client.setQueryData(key, next);
  patchSessionInLists(client, next);
  return { session, lists };
}

/** Make the server's status/archived/counts authoritative after the PUT. */
export function commitScoreResult(
  client: QueryClient, sessionId: string, result: ScoreResult,
) {
  const key = compareQueries.session(sessionId).queryKey;
  client.setQueryData<CompareSession>(key, old => old && reconcileScoreResult(old, result));
  client.setQueriesData<{ sessions: CompareSessionSummary[] }>(
    { queryKey: sessionsPrefix() },
    old => old && ({ sessions: old.sessions.map(s => (s.id === sessionId ? reconcileScoreResult(s, result) : s)) }),
  );
}

/** Restore the session and every list entry captured by beginOptimisticScores. */
export function rollbackScores(client: QueryClient, rollback: ScoreRollback) {
  if (rollback.session) client.setQueryData(compareQueries.session(rollback.session.id).queryKey, rollback.session);
  for (const [key, data] of rollback.lists) client.setQueryData(key, data);
}

// ── Mutation hooks ──────────────────────────────────────────────────────────

export function useCreateCompareSession() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateSessionInput) => compareApi.createSession(body),
    onSuccess: session => {
      client.setQueryData(compareQueries.session(session.id).queryKey, session);
      void queryManager.invalidate(client, sessionsPrefix());
    },
  });
}

export function useDeleteCompareSession() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => compareApi.deleteSession(id),
    onSuccess: (_result, id) => {
      client.removeQueries({ queryKey: compareQueries.session(id).queryKey, exact: true });
      void queryManager.invalidate(client, sessionsPrefix());
    },
  });
}

export function useArchiveCompareSession() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ id, archived }: { id: string; archived: boolean }) => compareApi.setArchived(id, archived),
    onSuccess: summary => {
      patchSessionSummary(client, summary);
      // Membership changed: the session moves between the archived and
      // non-archived lists. Invalidate both so they refetch with the new filter.
      void queryManager.invalidate(client, sessionsPrefix());
    },
  });
}

export function useRemoveCompareTake() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, songId }: { sessionId: string; songId: string }) =>
      compareApi.removeTake(sessionId, songId),
    onSuccess: session => patchSession(client, session),
  });
}

export function useSetCompareScore() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, items }: { sessionId: string; items: ScoreInput[] }) =>
      compareApi.putScores(sessionId, items),
    onMutate: ({ sessionId, items }) => beginOptimisticScores(client, sessionId, items),
    onError: (_error, _vars, rollback) => { if (rollback) rollbackScores(client, rollback); },
    onSuccess: (result, { sessionId }, rollback) => {
      commitScoreResult(client, sessionId, result);
      // Server auto-archive moved the session between lists — refetch both.
      // Compare with the pre-mutation snapshot: the cache is already
      // optimistically archived. No snapshot means membership is unknown.
      if (rollback?.session?.archived !== result.archived) {
        void queryManager.invalidate(client, sessionsPrefix());
      }
    },
  });
}
