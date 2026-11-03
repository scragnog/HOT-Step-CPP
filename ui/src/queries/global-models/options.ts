import { queryOptions, type QueryClient } from '@tanstack/react-query';
import { ApiClient } from '../../services/httpClient';
import { adapterApi, vstApi, type VstPlugin, type VstChainEntry } from '../../services/api';
import { serverKey, sharedKeys, scopeOf, resourceKey, type ServerKey } from '../keys';
import { pollWhile } from '../contract';
import { queryManager } from '../invalidation';
import type { PluginRegistry } from '../../types/pluginTypes';
import type { ModelRegistry, AceModels } from '../../types';
import type { BackendCapabilities, BackendInfo, BackendModelCatalogue } from '../../stores/backendStore';

export const adapterScanOptions = (folder: string) => queryOptions({
  queryKey: sharedKeys.adapterScan(folder),
  queryFn: ({ signal }) => adapterApi.scan(folder, { signal }),
  staleTime: 0,
});

/** Which plugin editor windows are still running. Polled only while at least
 *  one GUI is open; the observer lives in the VST store. */
export const openGuisOptions = () => queryOptions({
  queryKey: serverKey('global-models/vst-open-guis'),
  queryFn: ({ signal }) => vstApi.openGuis({ signal }),
  staleTime: 0,
});

/** The installed VST3 plugins, shared by the chain dropdown and its projection. */
export const vstScanOptions = () => localOptions<{ plugins: VstPlugin[] }>('vst-scan', '/vst/scan');
/** The persisted active VST chain. */
export const vstChainOptions = () => localOptions<{ plugins: VstChainEntry[] }>('vst-chain', '/vst/chain');

/** Every LM adapter under adapters/lm, for the shell and Lyric Studio pickers.
 *  `folder` narrows the scan (empty means the default adapters/lm root). */
export const lmAdaptersOptions = (folder = '') => read<Awaited<ReturnType<typeof adapterApi.lmList>>>(
  serverKey('global-models/lm-adapters', { filters: { folder } }),
  `/adapters/lm${folder ? `?folder=${encodeURIComponent(folder)}` : ''}`,
);

// Each request uses the API origin and backend captured in its key.
function read<T>(key: ServerKey, path: string) {
  return queryOptions({
    queryKey: key,
    queryFn: ({ queryKey, signal }) => new ApiClient({ baseUrl: scopeOf(queryKey)!.api }).get<T>(path, { signal }),
  });
}

export function localOptions<T>(name: string, path: string, filters = {}) {
  return read<T>(serverKey(`global-models/${name}`, { filters }), path);
}

export const pluginsOptions = () => read<PluginRegistry>(sharedKeys.plugins(), '/plugins');
export const registryOptions = () => localOptions<ModelRegistry>('registry', '/model-manager/registry');
export const backendsOptions = () => localOptions<{ backends: BackendInfo[]; activeId: string }>('backends', '/backends');
export const backendModelsOptions = (backend: string) => read<BackendModelCatalogue>(
  serverKey('global-models/backend-models', { backend }), `/backends/models?backend=${encodeURIComponent(backend)}`,
);
export const capabilitiesOptions = (backend: string) => queryOptions({
  ...read<BackendCapabilities>(sharedKeys.capabilities(backend), `/capabilities?backend=${encodeURIComponent(backend)}`),
  refetchInterval: pollWhile<BackendCapabilities>(data => !data?.up, 10_000),
});
export function usableCapabilities(previous: BackendCapabilities | undefined, next: BackendCapabilities) {
  return next.up === false && previous?.up ? previous : next;
}
export interface VramInfo { used_mb: number; total_mb: number; free_mb: number }
export interface LoadedModel { label: string; mb: number; in_use: boolean }
export const vramOptions = () => queryOptions({
  ...read<VramInfo>(sharedKeys.vram(), '/logs/vram'),
  refetchInterval: pollWhile<VramInfo>(() => true),
});
export const loadedModelsOptions = () => queryOptions({
  ...read<{ loaded: LoadedModel[] }>(sharedKeys.loadedModels(), '/logs/models-loaded'),
  refetchInterval: pollWhile<{ loaded: LoadedModel[] }>(() => true, 2000),
});
export const aceModelsOptions = () => queryOptions({
  ...localOptions<AceModels>('ace-models', '/models'),
  refetchInterval: pollWhile<AceModels>(data => !data?.models?.dit?.length, 3000),
});
export interface ReferenceTrack { name: string; size: number; url: string }
export const referencesOptions = () => localOptions<{ references: ReferenceTrack[] }>('references', '/mastering/references');
export interface PpVaeStatus { available: boolean }
export const ppVaeOptions = () => localOptions<PpVaeStatus>('pp-vae', '/models/pp-vae');
export interface StableStepStatus {
  available: boolean;
  backends?: { gguf: boolean };
  tokenizer: boolean;
  ggufWeights: boolean;
}
export const stableStepOptions = () => localOptions<StableStepStatus>('stablestep', '/models/stablestep');
export interface StableStepAdapterList { adapters: Array<{ name: string; sizeMb: number }> }
export const stableStepAdaptersOptions = () => localOptions<StableStepAdapterList>('sa3-adapters', '/models/stablestep/adapters');
export interface Mm3LmAdapterScales {
  scale: number; scaleAttn: number; scaleMlp: number; scaleEarly: number; scaleMid: number; scaleLate: number;
}
export interface Mm3LmAdapterEntry {
  /** Path relative to the mm3-lm-adapters root — what the request carries. */
  file: string;
  name?: string;
  trigger?: string;
  /** false when the run recorded a trigger it never trained. Absent on every
   *  sidecar written before the field existed, and absent means trained. */
  triggerPrepend?: boolean;
  rank?: number;
  dataset?: string;
  encoder?: string;
  trainedSteps?: number;
  recommendedScales?: Partial<Mm3LmAdapterScales>;
  notes?: string;
}
export interface Mm3LmAdapterCatalogue {
  adapters: Mm3LmAdapterEntry[];
  defaultScales: Mm3LmAdapterScales;
  dir: string;
}
export const mm3AdaptersOptions = () => localOptions<Mm3LmAdapterCatalogue>('mm3-adapters', '/mm3/lm-adapters');
export interface MonitorStatus { running: boolean; paused: boolean; position: number; duration: number }
export const monitorOptions = () => queryOptions({
  ...localOptions<MonitorStatus>('monitor', '/vst/monitor/status'),
  staleTime: 0,
  refetchInterval: (query: Parameters<ReturnType<typeof pollWhile<MonitorStatus>>>[0]) =>
    pollWhile<MonitorStatus>(() => true, query.state.data?.running ? 300 : 2000)(query),
});

// File installation affects every backend and each availability probe. Prefixes
// include inactive backend caches so returning to a backend cannot reuse them.
export const modelChangeKeys = () => [
  resourceKey('global-models/registry'), resourceKey('global-models/backend-models'),
  resourceKey('global-models/ace-models'), resourceKey('capabilities'),
  resourceKey('loaded-models'), resourceKey('vram'),
  resourceKey('global-models/pp-vae'), resourceKey('global-models/stablestep'),
  resourceKey('global-models/sa3-adapters'), resourceKey('global-models/mm3-adapters'),
];
export async function invalidateModels(client: QueryClient) {
  await queryManager.invalidate(client, ...modelChangeKeys());
}
