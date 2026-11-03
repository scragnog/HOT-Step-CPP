// queries.ts — Compare studio server snapshots.
//
// Sessions are a local resource (no auth/worker/backend scope). The list is
// filtered by `archived`; the detail carries takes and scores; analysis and
// diff are derived reads with their own long-lived entries so switching
// takes or replaying a session never re-analyzes the same song.

import { keepPreviousData, queryOptions } from '@tanstack/react-query';
import {
  compareApi, searchCompareSongs,
  type CompareAnalysis, type CompareDiff, type CompareSession, type CompareSessionSummary,
} from '../../services/compareApi';
import { serverKey, type Resource } from '../keys';

/** Key names, declared once so invalidation and the factories cannot drift. */
export const CMP = {
  sessions: 'compare/sessions',
  session: 'compare/session',
  songs: 'compare/songs',
  analysis: 'compare/analysis',
  diff: 'compare/diff',
} as const satisfies Record<string, Resource>;

/** Analysis is CPU-heavy server-side and cached there; keep it fresh for a
 *  session's length instead of the default 30 s. */
const DERIVED_STALE_MS = 5 * 60_000;

export const compareQueries = {
  sessions: (opts: { archived?: boolean; group?: string } = {}) => queryOptions({
    queryKey: serverKey(CMP.sessions, { filters: { archived: opts.archived, group: opts.group } }),
    queryFn: ({ signal }) => compareApi.listSessions(opts, { signal }),
  }),

  session: (id: string) => queryOptions({
    queryKey: serverKey(CMP.session, { id }),
    queryFn: ({ signal }) => compareApi.getSession(id, { signal }),
  }),

  /** Library search for the new-session picker. Keeps the previous page while
   *  the next one loads, so typing never blanks the list. */
  songSearch: (q: string, limit: number, offset: number) => queryOptions({
    queryKey: serverKey(CMP.songs, { filters: { q, limit, offset } }),
    queryFn: ({ signal }) => searchCompareSongs(q, limit, offset, { signal }),
    placeholderData: keepPreviousData,
  }),

  analysis: (songId: string) => queryOptions({
    queryKey: serverKey(CMP.analysis, { id: songId }),
    queryFn: ({ signal }) => compareApi.getAnalysis(songId, { signal }),
    staleTime: DERIVED_STALE_MS,
  }),

  diff: (a: string, b: string) => queryOptions({
    queryKey: serverKey(CMP.diff, { filters: { a, b } }),
    queryFn: ({ signal }) => compareApi.getDiff(a, b, { signal }),
    staleTime: DERIVED_STALE_MS,
  }),
};

export type { CompareAnalysis, CompareDiff, CompareSession, CompareSessionSummary };
