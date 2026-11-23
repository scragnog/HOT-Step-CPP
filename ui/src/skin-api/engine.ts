// engine.ts — backends, what they can do, their models, VRAM and the engine log.

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useBackendStore } from '../stores/backendStore';
import { backendsOptions, backendModelsOptions, capabilitiesOptions, vramOptions } from '../queries/global-models/options';
import { BACKEND_PREFIX } from '../app/paramStore';
import { useEventSource } from '../hooks/useEventSource';
import type { SkinBackend, SkinCapabilities, SkinLogLine, SkinModelCatalogue, SkinVram } from './types';

/** Every generation backend the server knows. */
export function useBackends(): { backends: SkinBackend[]; activeId: string } {
  const q = useQuery(backendsOptions());
  const activeId = useBackendStore(s => s.activeBackendId) || 'ace';
  const backends = useMemo(() => (q.data?.backends ?? []).map(b => ({ id: b.id, name: b.displayName, active: b.id === activeId })), [q.data, activeId]);
  return { backends, activeId };
}

/** The backend renders go to right now, or undefined until the list has loaded. */
export function useActiveBackend(): SkinBackend | undefined {
  const { backends } = useBackends();
  return backends.find(b => b.active);
}

export const backend = {
  /** Make `id` the active backend. Per-backend settings follow it, as in classic. */
  switch: (id: string): Promise<void> => useBackendStore.getState().switchBackend(id),
};

/** What a backend supports. Defaults to the active backend. */
export function useCapabilities(backendId?: string): SkinCapabilities | undefined {
  const active = useBackendStore(s => s.activeBackendId) || 'ace';
  const id = backendId ?? active;
  const q = useQuery(capabilitiesOptions(id));
  return useMemo(() => q.data && ({
    backend: q.data.backend, up: q.data.up,
    maxDuration: q.data.core.duration.max,
    supportsBpm: q.data.core.bpm, supportsKeyScale: q.data.core.keyscale,
    supportsNegativePrompt: q.data.core.negativePrompt, maxBatch: q.data.core.batch.max,
    extensionIds: q.data.extensions.map(e => `${BACKEND_PREFIX}${e.key}`),
  }), [q.data]);
}

/** The model files a backend offers, by bucket. Fill model selects from this. */
export function useBackendModels(backendId?: string): SkinModelCatalogue | undefined {
  const active = useBackendStore(s => s.activeBackendId) || 'ace';
  const id = backendId ?? active;
  const q = useQuery(backendModelsOptions(id));
  return useMemo(() => q.data && ({ backend: q.data.backend, buckets: q.data.buckets, adapters: q.data.adapters ?? [] }), [q.data]);
}

/** GPU memory, refreshed while mounted. */
export function useVram(): SkinVram | undefined {
  const q = useQuery(vramOptions());
  return useMemo(() => q.data && ({ usedMb: q.data.used_mb, totalMb: q.data.total_mb, freeMb: q.data.free_mb }), [q.data]);
}

/** The last 500 engine and server log lines. Opens the log stream only while mounted. */
export function useEngineLog(): { lines: readonly SkinLogLine[]; connected: boolean } {
  const { lines, connected } = useEventSource('/api/logs', true);
  return { lines, connected };
}
