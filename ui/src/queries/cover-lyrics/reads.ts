import { queryOptions } from '@tanstack/react-query';
import { getAuthToken } from '@/services/axiosClient';
import { apiClient, type ReadOptions } from '@/services/httpClient';
import { queryClient } from '@/queries/client';
import { serverKey, resourceKey, type Resource } from '@/queries/keys';
import { queryManager } from '@/queries/invalidation';

export type Domain = 'lireek' | 'drafts' | 'cover' | 'workflow';
const resource = (domain: Domain): Resource => `cover-lyrics/${domain}`;

/** Raw wire snapshots; callers select or hydrate an editable working copy. */
export function snapshotOptions<T>(domain: Domain, path: string, options: ReadOptions = {}, staleTime?: number) {
  const [pathname, search = ''] = path.split('?');
  const filters = Object.fromEntries(new URLSearchParams(search));
  // The transport sends the bearer whenever a session exists, so the key's
  // auth scope is derived from the same accessor — never a threaded token.
  const key = serverKey(resource(domain), { auth: getAuthToken() !== null, id: pathname, filters });
  return queryOptions({
    queryKey: key,
    queryFn: ({ signal }) => apiClient.get<T>(path, {
      signal: options.signal ? AbortSignal.any([signal, options.signal]) : signal,
    }),
    ...(staleTime === undefined ? {} : { staleTime }),
  });
}

export function readSnapshot<T>(domain: Domain, path: string, options: ReadOptions = {}, staleTime?: number): Promise<T> {
  return queryClient.query(snapshotOptions<T>(domain, path, options, staleTime));
}

/** Cancel before invalidating: an older read must never repopulate a changed document. */
export async function changed(domain: Domain, scope?: { session?: number; api?: string }) {
  const queryKey = resourceKey(resource(domain), scope);
  await queryClient.cancelQueries({ queryKey });
  await queryManager.invalidate(queryClient, queryKey);
}

export async function songsChanged() {
  await queryManager.songs.views(queryClient);
}
