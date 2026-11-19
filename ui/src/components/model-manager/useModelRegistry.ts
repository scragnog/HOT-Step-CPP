// useModelRegistry.ts — Hook for fetching and querying the model registry

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { registryOptions } from '@/queries/global-models/options';
import type { RegistryFile } from '@/types';
import { isPresent } from '@/utils/guards';

export function useModelRegistry() {
  const query = useQuery(registryOptions());
  const registry = query.data ?? null;

  const installedFiles = useMemo(() => {
    if (!registry) return new Set<string>();
    return new Set(registry.files.filter(f => f.installed).map(f => f.id));
  }, [registry]);

  /** fileIds is data from model-registry.json, so a hand-edited catalogue can
   *  ship a pack without it. One empty pack is a better failure than a blank
   *  page (issue #120). */
  const getPackFiles = (packId: string): RegistryFile[] => {
    const pack = registry?.packs.find(p => p.id === packId);
    if (!pack || !registry) return [];
    if (!Array.isArray(pack.fileIds)) {
      console.warn(`[ModelRegistry] Pack "${packId}" has no fileIds array — nothing to download.`);
      return [];
    }
    return pack.fileIds.map(id => registry.files.find(f => f.id === id)).filter(isPresent) as RegistryFile[];
  };

  return { registry, loading: query.isPending, error: query.error?.message ?? null, installedFiles, getPackFiles };
}
