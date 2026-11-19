// songs.ts — the library and recent-songs snapshots, and every song write's
// cache follow-up. shell-queue owns the canonical songs/song reads
// (docs/dev/ui-server-state.md); other areas query them by the shared keys.

import { queryOptions, type QueryClient } from '@tanstack/react-query';
import { getSongPostProcessingStatus, songApi, type PostProcessJobStatus } from '@/services/api';
import type { Song } from '@/types';
import { pollWhile } from '@/queries/contract';
import { resourceKey, serverKey, sharedKeys, SHARED, SONG_VIEWS } from '@/queries/keys';
import { queryManager } from '@/queries/invalidation';

export const songsOptions = () => queryOptions({
  queryKey: sharedKeys.songs(),
  queryFn: ({ signal }) => songApi.list({ signal }),
});

export const songOptions = (id: string) => queryOptions({
  queryKey: sharedKeys.song(id),
  queryFn: ({ signal }) => songApi.get(id, { signal }),
});

export const recentSongsOptions = (source = 'all') => queryOptions({
  queryKey: serverKey('shell-queue/recent-songs', { auth: true, filters: { source: source === 'all' ? undefined : source } }),
  queryFn: ({ signal }) => songApi.getRecentSongs(source, 50, { signal }),
});

const POST_PROCESS_ACTIVE = new Set(['starting', 'pending', 'running']);

/** One after-the-fact post-processing run. Polls at the active cadence until
 *  the run ends; a null answer (the job aged out of the server's map) keeps
 *  polling, as before. */
export const postProcessStatusOptions = (songId: string, jobId: string) => queryOptions({
  queryKey: serverKey('shell-queue/post-process', { id: songId, filters: { jobId } }),
  queryFn: ({ signal }) => getSongPostProcessingStatus(songId, jobId, { signal }),
  staleTime: 0,
  refetchInterval: pollWhile<PostProcessJobStatus | null>(s => !s || POST_PROCESS_ACTIVE.has(s.status)),
});

const RECENT = resourceKey('shell-queue/recent-songs');

/** Every cached view of the library, across areas (keys.ts). A song write
 *  refetches all of them. Studios' and Cover's feeds are only invalidated,
 *  never edited from here: their shapes belong to their areas. */
export { SONG_VIEWS };

export const invalidateSongs = (client: QueryClient): Promise<unknown> => queryManager.songs.views(client);

type SongList = { songs: Song[] };

/** Edit every cached library list in place (before the refetch lands). */
export function patchLibrary(client: QueryClient, edit: (songs: Song[]) => Song[]): void {
  client.setQueriesData<SongList>({ queryKey: resourceKey(SHARED.songs) }, old => old && { songs: edit(old.songs) });
}

/** Apply a field patch to one song wherever this area caches it (library
 *  lists, its detail, the recent-songs feed), then refetch every song view.
 *  The refetch also cancels any older read still in flight, so it cannot
 *  land over the patch. Rows carry both spellings, so the patch should too. */
export function patchSong(client: QueryClient, id: string, patch: Record<string, unknown>): Promise<unknown> {
  const edit = <T extends { id: string }>(rows: T[]) => rows.map(s => (s.id === id ? { ...s, ...patch } : s));
  patchLibrary(client, edit);
  client.setQueriesData<{ song: Song }>({ queryKey: resourceKey(SHARED.song, {}, id) }, old => old && { song: { ...old.song, ...patch } });
  client.setQueriesData<{ songs: { id: string }[] }>({ queryKey: RECENT }, old => old && { ...old, songs: edit(old.songs) });
  return invalidateSongs(client);
}

/** After a successful delete: drop the rows now, then refetch. */
export function removeSongs(client: QueryClient, ids: string[]): Promise<unknown> {
  const gone = new Set(ids);
  patchLibrary(client, songs => songs.filter(s => !gone.has(s.id)));
  client.setQueriesData<{ songs: { id: string }[] }>({ queryKey: RECENT }, old => old && { ...old, songs: old.songs.filter(s => !gone.has(s.id)) });
  for (const id of ids) client.removeQueries({ queryKey: resourceKey(SHARED.song, {}, id) });
  return invalidateSongs(client);
}
