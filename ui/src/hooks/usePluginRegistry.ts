import { useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { pluginsOptions } from '@/queries/global-models/options';
import { queryManager } from '@/queries/invalidation';
import type { PluginRegistry, PluginInfo } from '@/types/pluginTypes';
const EMPTY_REGISTRY: PluginRegistry = { solvers: [], schedulers: [], guidance: [] };

export function usePluginRegistry() {
  const client = useQueryClient();
  const query = useQuery(pluginsOptions());
  const registry = query.data ?? EMPTY_REGISTRY;
  const reload = useCallback(() => queryManager.invalidate(client, pluginsOptions().queryKey), [client]);
  const findSolver = useCallback((name: string): PluginInfo | undefined => registry.solvers.find(s => s.name === name), [registry]);
  const findScheduler = useCallback((name: string): PluginInfo | undefined => registry.schedulers.find(s => s.name === name), [registry]);
  const findGuidance = useCallback((name: string): PluginInfo | undefined => registry.guidance.find(s => s.name === name), [registry]);
  return { registry, loading: query.isPending, error: query.error?.message ?? null, reload, findSolver, findScheduler, findGuidance };
}
