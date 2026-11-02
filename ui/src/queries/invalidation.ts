// invalidation.ts — the one global queryManager: the only place in the app
// that invalidates cache entries (P2.4, docs/dev/tanstack-query.md).
//
// Every refetch-after-write goes through this object: store writes declare
// what they dirtied via writeAndInvalidate (writes.ts), the SSE bridge flushes
// its coalesced keys here, and refresh buttons call a named group below. No
// other file calls QueryClient.invalidateQueries — grepping for it finds
// exactly this one (tests excepted).
//
// P2 shape: the groups wrap today's per-area key lists; when P6 folds each
// entity into its own composer folder, a group closes over that composer and
// the old list/helper is deleted. Parameterized invalidations (a write knows
// its worker/id) call `invalidate`/`filter` with the captured scope — the manager
// still owns the actual invalidateQueries call.

import type { InvalidateQueryFilters, QueryClient, QueryKey } from '@tanstack/react-query';
import { resourceKey, SHARED, SONG_VIEWS } from './keys';

/** The one primitive: the only QueryClient.invalidateQueries call in the app. */
export function invalidateKeys(client: QueryClient, keys: QueryKey[]): Promise<unknown> {
  return Promise.all(keys.map(key => client.invalidateQueries({ queryKey: key })));
}

/** For filters that need more than a key prefix (predicates, type narrowing). */
export function invalidateQuery(client: QueryClient, filter: InvalidateQueryFilters): Promise<unknown> {
  return client.invalidateQueries(filter);
}

export const queryManager = {
  /** Every cached view of song rows, across areas (keys.ts's SONG_VIEWS). A
   *  song write, delete, completion or reconnect dirties all of them. */
  songs: {
    views: (c: QueryClient) => invalidateKeys(c, SONG_VIEWS),
  },
  /** The editable .env snapshot, after Settings saved a diff. The key is
   *  named here (not imported from shell-queue) so this file stays the only
   *  invalidation owner without taking an area dependency. */
  env: {
    values: (c: QueryClient) => invalidateKeys(c, [resourceKey('shell-queue/env')]),
  },
  /** A track's lyric sidecars, after a crop or retranscription rewrote them. */
  sidecars: {
    all: (c: QueryClient) => invalidateKeys(c, [resourceKey(SHARED.lrc), resourceKey(SHARED.lyricsJson)]),
  },
  /** Lyric Studio (lireek) snapshots — a purge of generations or profiles
   *  rewrote every cached document in the area (`cover-lyrics/reads.ts`'s
   *  `resource('lireek')`; P6 folds this into the entity composer). */
  coverLyrics: {
    lireek: (c: QueryClient) => invalidateKeys(c, [resourceKey('cover-lyrics/lireek')]),
  },
  /** Invalidate exact keys or prefixes. The one call site for "this write
   *  dirtied these specific entries." */
  invalidate: (c: QueryClient, ...keys: QueryKey[]) => invalidateKeys(c, keys),
  /** A full filter (predicate-narrowed sets, e.g. one worker's reads). */
  filter: (c: QueryClient, filter: InvalidateQueryFilters) => invalidateQuery(c, filter),
  /** Everything, after a write that can replace rows in every area (a settings
   *  backup import). Still routed through the one owner. */
  all: (c: QueryClient) => invalidateQuery(c, {}),

  /** Event map: writes declare what happened; the manager resolves it to the
   *  groups above. Parameterized methods are called directly when the write
   *  knows the id (e.g. a patch). */
  events: {
    'song:written': (c: QueryClient) => queryManager.songs.views(c),
    'sidecars:changed': (c: QueryClient) => queryManager.sidecars.all(c),
    /** A nuke of Lyric Studio generations or profiles (SettingsPanel). */
    'cover-lyrics:lireek-purged': (c: QueryClient) => queryManager.coverLyrics.lireek(c),
  },
};

export type EventRef = keyof typeof queryManager.events;

/** Resolve a list of named events to their invalidations. */
export function invalidateEvents(client: QueryClient, events: readonly EventRef[]): Promise<unknown> {
  return Promise.all(events.map(e => queryManager.events[e](client)));
}
