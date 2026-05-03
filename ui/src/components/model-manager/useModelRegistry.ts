// useModelRegistry.ts — Hook for fetching and querying the model registry

import { useState, useEffect, useCallback, useMemo } from 'react';
import { modelManagerApi } from '../../services/api';
import type { ModelRegistry, RegistryFile, StarterPack } from '../../types';

export interface PackStatus {
  total: number;
  installed: number;
  missing: number;
  totalSize: number;
  remainingSize: number;
}

export function useModelRegistry() {
  const [registry, setRegistry] = useState<ModelRegistry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRegistry = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await modelManagerApi.registry();
      setRegistry(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch registry');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRegistry();
  }, [fetchRegistry]);

  const installedFiles = useMemo(() => {
    if (!registry) return new Set<string>();
    return new Set(registry.files.filter(f => f.installed).map(f => f.filename));
  }, [registry]);

  const getFile = useCallback((id: string): RegistryFile | undefined => {
    return registry?.files.find(f => f.id === id);
  }, [registry]);

  const getPackFiles = useCallback((packId: string): RegistryFile[] => {
    const pack = registry?.packs.find(p => p.id === packId);
    if (!pack || !registry) return [];
    return pack.fileIds.map(id => registry.files.find(f => f.id === id)).filter(Boolean) as RegistryFile[];
  }, [registry]);

  const packStatus = useCallback((packId: string): PackStatus => {
    const files = getPackFiles(packId);
    const installed = files.filter(f => f.installed).length;
    return {
      total: files.length,
      installed,
      missing: files.length - installed,
      totalSize: files.reduce((a, f) => a + f.sizeBytes, 0),
      remainingSize: files.filter(f => !f.installed).reduce((a, f) => a + f.sizeBytes, 0),
    };
  }, [getPackFiles]);

  return {
    registry,
    loading,
    error,
    refresh: fetchRegistry,
    installedFiles,
    getFile,
    getPackFiles,
    packStatus,
  };
}
