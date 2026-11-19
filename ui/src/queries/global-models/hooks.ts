import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { masteringApi } from '@/services/api';
import { referencesOptions, vramOptions } from './options';
import { queryManager } from '@/queries/invalidation';

export const useVram = () => useQuery(vramOptions());

/** The shared reference-track read plus its upload/delete writes. Every
 *  mastering/timbre UI uses this, so the cache entry and invalidation have one
 *  owner. `mutate` takes per-call `onSuccess` for the caller's selection
 *  side-effect (which reference to select/clear). */
export function useReferenceTracks() {
  const client = useQueryClient();
  const query = useQuery(referencesOptions());
  const refresh = useCallback(() => queryManager.invalidate(client, referencesOptions().queryKey), [client]);
  const upload = useMutation({
    mutationFn: (file: File) => masteringApi.uploadReference(file),
    onSuccess: refresh,
  });
  const remove = useMutation({
    mutationFn: (name: string) => masteringApi.deleteReference(name),
    onSuccess: refresh,
  });
  return {
    references: query.data?.references ?? [],
    query,
    refresh,
    upload,
    remove,
    uploading: upload.isPending,
    deleting: remove.isPending,
  };
}
