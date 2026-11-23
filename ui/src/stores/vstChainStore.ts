import { projectQuery } from '@/queries/global-models/projection';
import { vstPresetsApi } from '@/queries/global-models/presets';
import type { MonitorStatus } from '@/queries/global-models/options';
import { queryClient } from '@/queries/client';
import type { ServerKey } from '@/queries/keys';
import { monitorOptions, openGuisOptions, vstChainOptions, vstScanOptions } from '@/queries/global-models/options';
import { queryManager } from '@/queries/invalidation';
import { keysPlan, writeAndInvalidate } from '@/queries/writes';
import { apiClient } from '@/services/httpClient';
// vstChainStore.ts — Zustand store for VST3 post-processing chain
//
// Manages the global VST3 plugin chain: scanning, ordering, enable/disable,
// and GUI launching. Chain state is persisted server-side.

import { create } from 'zustand';
import { QueryObserver } from '@tanstack/react-query';
import { vstApi, type VstPlugin, type VstChainEntry } from '@/services/api';
import { hashImportValue, type VstChainPresetBody } from '@/services/preferencesApi';
import { PresetCollection, type PresetCollectionSnapshot } from '@/services/presetCollection';
import { readJson } from '@/utils/storage/storageJson';

// ── Presets (named chain snapshots) ──────────────────────────────────────────
//
// Presets live on the server (preferences vst-chain, installation scope: one
// shared owner per machine, same as the active chain), as a PresetCollection
// keyed by document id; two presets may share a name. The browser key is
// imported once (a same-name collision waits for the user's keep-both or
// replace) and never written to again.

export const PRESETS_KEY = 'vst-chain-presets';
export const MIGRATED_FLAG = 'vst-chain-presets:server-migrated';

function loadLegacyPresetsFromStorage(): Record<string, VstChainEntry[]> {
  const v = readJson<Record<string, VstChainEntry[]>>(PRESETS_KEY, {});
  return v && typeof v === 'object' && !Array.isArray(v) ? v : {};
}

export function createVstPresetCollection(): PresetCollection<VstChainPresetBody> {
  return new PresetCollection<VstChainPresetBody>({
    api: vstPresetsApi,
    family: 'vst-chain',
    migratedFlag: MIGRATED_FLAG,
    legacyItems: () => Promise.all(Object.entries(loadLegacyPresetsFromStorage()).map(async ([name, entries]) => {
      const body: VstChainPresetBody = { name, entries: entries as VstChainPresetBody['entries'] };
      return { storageKey: `${PRESETS_KEY}:${name}`, sourceHash: await hashImportValue(JSON.stringify(body)), name, body };
    })),
  });
}

interface VstChainState {
  // Available plugins (from scan)
  plugins: VstPlugin[];
  scanning: boolean;
  scanError: string | null;

  // Active chain (persisted on server)
  chain: VstChainEntry[];
  chainLoaded: boolean;

  // UIDs whose native GUI windows are currently open
  openGuiUids: string[];
  // set when any GUI is opened while monitoring — cleared on restart
  pendingGuiChanges: boolean;

  // Monitor (real-time playback)
  monitoring: boolean;
  monitorPaused: boolean;
  monitorPosition: number;
  monitorDuration: number;
  // track currently loaded in monitor — needed for restart
  monitorTrackPath: string;

  // Named chain snapshots, persisted server-side. `presetSnapshot` mirrors
  // `presetCollection`; render notices and act on entries through it.
  presetCollection: PresetCollection<VstChainPresetBody>;
  presetSnapshot: PresetCollectionSnapshot<VstChainPresetBody>;

  // Actions
  scanPlugins: () => Promise<void>;
  loadChain: () => Promise<void>;
  addToChain: (plugin: VstPlugin) => Promise<void>;
  removeFromChain: (uid: string) => Promise<void>;
  toggleEnabled: (uid: string) => Promise<void>;
  reorderChain: (fromIndex: number, toIndex: number) => Promise<void>;
  openGui: (plugin: VstChainEntry) => Promise<void>;
  closeGui: (uid: string) => void;
  chainEnabled: () => boolean;
  startMonitor: (trackPath: string) => Promise<void>;
  stopMonitor: () => Promise<void>;
  restartMonitor: () => Promise<void>;
  pauseMonitor: () => Promise<void>;
  resumeMonitor: () => Promise<void>;
  switchMonitorTrack: (trackPath: string) => Promise<void>;
  seekMonitor: (position: number) => Promise<void>;
  pollMonitorStatus: () => Promise<void>;
  loadPresets: () => Promise<void>;
  /** Save the current chain under `name`: updates the one preset with that
   *  name, or creates a new one (also when several share it). */
  savePreset: (name: string) => Promise<void>;
  /** Apply a preset, by key (document id). */
  loadPreset: (key: string) => Promise<void>;
  deletePreset: (key: string) => Promise<void>;
}

let guiPoll: { observer: QueryObserver<{ uids: string[] }, Error, { uids: string[] }, { uids: string[] }, ServerKey>; unsubscribe: () => void; stopped: boolean } | null = null;

/** Follow the open-GUI read until every editor window has closed. */
function startGuiPoll() {
  if (guiPoll) return;
  const startedAt = Date.now();
  const observer = new QueryObserver(queryClient, {
    ...openGuisOptions(),
    refetchInterval: 1500,
    refetchIntervalInBackground: true,
  });
  // Registered before subscribe(), which can call the listener synchronously:
  // a stop from inside that first call must still tear this poll down.
  const poll = { observer, unsubscribe: () => {}, stopped: false };
  guiPoll = poll;
  const unsubscribe = observer.subscribe(result => {
    if (poll.stopped || !result.data) return;
    // Only a read that finished after this poll started counts. subscribe()
    // replays the cached entry, and that older snapshot stays in `data` while
    // the first fetch runs; acting on it would clear the marker openGui just set.
    if (result.isFetching || result.dataUpdatedAt < startedAt) return;
    const open = useVstChainStore.getState().openGuiUids.filter(u => result.data!.uids.includes(u));
    useVstChainStore.setState({ openGuiUids: open });
    if (open.length === 0) stopGuiPoll();
  });
  poll.unsubscribe = unsubscribe;
  if (poll.stopped) unsubscribe();
}
function stopGuiPoll() {
  if (!guiPoll) return;
  guiPoll.stopped = true;
  guiPoll.unsubscribe();
  guiPoll = null;
}

/** A preset collection whose changes are mirrored into the store. */
function attachPresets(set: (partial: Partial<VstChainState>) => void) {
  const presetCollection = createVstPresetCollection();
  presetCollection.subscribe(() => set({ presetSnapshot: presetCollection.getSnapshot() }));
  return { presetCollection, presetSnapshot: presetCollection.getSnapshot() };
}

export const useVstChainStore = create<VstChainState>((set, get) => {
  return {
  plugins: [],
  scanning: false,
  scanError: null,
  chain: [],
  chainLoaded: false,
  openGuiUids: [],
  pendingGuiChanges: false,
  monitoring: false,
  monitorPaused: false,
  monitorPosition: 0,
  monitorDuration: 0,
  monitorTrackPath: '',
  ...attachPresets(set),

  loadPresets: () => get().presetCollection.load(),

  scanPlugins: async () => {
    set({ scanning: true, scanError: null });
    try {
      await queryManager.invalidate(queryClient, vstScanOptions().queryKey);
      await queryClient.query(vstScanOptions());
      set({ scanning: false });
    } catch (err: any) {
      set({ scanning: false, scanError: err.message });
    }
  },

  loadChain: async () => {
    try {
      await queryClient.query(vstChainOptions());
    } catch {
      set({ chainLoaded: true });
    }
  },

  addToChain: async (plugin: VstPlugin) => {
    const { chain } = get();
    // Don't add duplicates
    if (chain.some(p => p.uid === plugin.uid)) return;

    const entry: VstChainEntry = {
      uid: plugin.uid,
      name: plugin.name,
      vendor: plugin.vendor,
      path: plugin.path,
      enabled: true,
      statePath: '',
    };
    const newChain = [...chain, entry];
    set({ chain: newChain });

    try {
      await updateCachedChain(newChain);
    } catch { /* transport failure already toasted by the global handler */ }
  },

  removeFromChain: async (uid: string) => {
    const { chain } = get();
    const newChain = chain.filter(p => p.uid !== uid);
    set({ chain: newChain });

    try {
      await updateCachedChain(newChain);
    } catch { /* transport failure already toasted by the global handler */ }
  },

  toggleEnabled: async (uid: string) => {
    const { chain } = get();
    const newChain = chain.map(p =>
      p.uid === uid ? { ...p, enabled: !p.enabled } : p
    );
    set({ chain: newChain });

    try {
      await updateCachedChain(newChain);
    } catch { /* transport failure already toasted by the global handler */ }
  },

  reorderChain: async (fromIndex: number, toIndex: number) => {
    const { chain } = get();
    const newChain = [...chain];
    const [moved] = newChain.splice(fromIndex, 1);
    newChain.splice(toIndex, 0, moved);
    set({ chain: newChain });

    try {
      await updateCachedChain(newChain);
    } catch { /* transport failure already toasted by the global handler */ }
  },

  openGui: async (plugin: VstChainEntry) => {
    try {
      await vstApi.openGui(plugin.path, plugin.uid);
      set(s => ({
        openGuiUids: [...s.openGuiUids.filter(u => u !== plugin.uid), plugin.uid],
        // flag that state files may be dirty if monitor is running
        pendingGuiChanges: s.monitoring ? true : s.pendingGuiChanges,
      }));
      // Poll until every editor window has been closed, so the open marker
      // clears when the user closes the window.
      startGuiPoll();
    } catch { /* transport failure already toasted by the global handler */ }
  },

  closeGui: (uid: string) => {
    let lastOne = false;
    set(s => {
      const openGuiUids = s.openGuiUids.filter(u => u !== uid);
      lastOne = openGuiUids.length === 0;
      return { openGuiUids };
    });
    // The user closed the last editor before the next poll noticed; stop now
    // instead of leaving the observer ticking forever.
    if (lastOne) stopGuiPoll();
  },

  chainEnabled: () => {
    return get().chain.some(p => p.enabled);
  },

  startMonitor: async (trackPath: string) => {
    try {
      await writeAndInvalidate(queryClient, () => vstApi.monitorStart(trackPath),
        keysPlan(monitorOptions().queryKey));
      set({ monitoring: true, monitorPaused: false, monitorTrackPath: trackPath, pendingGuiChanges: false });
    } catch { /* transport failure already toasted by the global handler */ }
  },

  stopMonitor: async () => {
    try {
      await writeAndInvalidate(queryClient, () => vstApi.monitorStop(),
        keysPlan(monitorOptions().queryKey));
      set({ monitoring: false, monitorPaused: false, monitorPosition: 0, pendingGuiChanges: false });
    } catch { /* transport failure already toasted by the global handler */ }
  },

  restartMonitor: async () => {
    const { monitorTrackPath } = get();
    if (!monitorTrackPath) return;
    try {
      await writeAndInvalidate(queryClient, () => apiClient.post('/vst/monitor/restart'),
        keysPlan(monitorOptions().queryKey));
      set({ monitorPaused: false, monitorPosition: 0, pendingGuiChanges: false });
    } catch {
      // Fallback: manual stop + start (the failed restart is already toasted).
      try {
        await writeAndInvalidate(queryClient, () => vstApi.monitorStop(),
          keysPlan(monitorOptions().queryKey));
        await writeAndInvalidate(queryClient, () => vstApi.monitorStart(monitorTrackPath),
          keysPlan(monitorOptions().queryKey));
        set({ monitoring: true, monitorPaused: false, monitorPosition: 0, pendingGuiChanges: false });
      } catch { /* each failed step toasted by the global handler */ }
    }
  },

  pauseMonitor: async () => {
    try {
      await writeAndInvalidate(queryClient, () => apiClient.post('/vst/monitor/pause'),
        keysPlan(monitorOptions().queryKey));
      set({ monitorPaused: true });
    } catch { /* transport failure already toasted by the global handler */ }
  },

  resumeMonitor: async () => {
    try {
      await writeAndInvalidate(queryClient, () => apiClient.post('/vst/monitor/resume'),
        keysPlan(monitorOptions().queryKey));
      set({ monitorPaused: false });
    } catch { /* transport failure already toasted by the global handler */ }
  },

  switchMonitorTrack: async (trackPath: string) => {
    try {
      await writeAndInvalidate(queryClient, () => vstApi.monitorSwitch(trackPath),
        keysPlan(monitorOptions().queryKey));
      set({ monitorTrackPath: trackPath });
    } catch { /* transport failure already toasted by the global handler */ }
  },

  seekMonitor: async (position: number) => {
    try {
      await writeAndInvalidate(queryClient, () => vstApi.monitorSeek(position),
        keysPlan(monitorOptions().queryKey));
    } catch { /* transport failure already toasted by the global handler */ }
  },

  pollMonitorStatus: async () => {
    try { await queryClient.query(monitorOptions()); }
    catch { /* Keep the last successful scoped projection. */ }
  },

  savePreset: async (name: string) => {
    const { chain, presetCollection } = get();
    const body: VstChainPresetBody = { name, entries: chain.map(p => ({ ...p })) as VstChainPresetBody['entries'] };
    const sameName = presetCollection.getSnapshot().entries.filter(e => e.body.name === name);
    if (sameName.length === 1) presetCollection.update(sameName[0]!.key, body);
    else await presetCollection.create(body);
  },

  loadPreset: async (key: string) => {
    const preset = get().presetCollection.getSnapshot().entries.find(e => e.key === key)?.body.entries;
    if (!preset) return;
    set({ chain: preset });
    try {
      await updateCachedChain(preset);
    } catch { /* transport failure already toasted by the global handler */ }
  },

  deletePreset: (key: string) => get().presetCollection.remove(key),
  };
});

/** Test-only: a fresh preset collection, as in a newly opened browser. */
export function _resetVstPresetsForTests(): void {
  useVstChainStore.setState(attachPresets(useVstChainStore.setState));
}

async function updateCachedChain(plugins: VstChainEntry[]) {
  const key = vstChainOptions().queryKey;
  await queryClient.cancelQueries({ queryKey: key });
  const result = await writeAndInvalidate(queryClient,
    () => apiClient.put<{ plugins: VstChainEntry[] }>('/vst/chain', { plugins }),
    keysPlan(key));
  queryClient.setQueryData(key, result);
  return result;
}

const detachVstProjections = [
  projectQuery<{ plugins: VstPlugin[] }>(queryClient,
    vstScanOptions().queryKey, data => useVstChainStore.setState({ plugins: data.plugins })),
  projectQuery<{ plugins: VstChainEntry[] }>(queryClient,
    vstChainOptions().queryKey, data => useVstChainStore.setState({ chain: data.plugins ?? [], chainLoaded: true })),
  projectQuery<MonitorStatus>(queryClient, monitorOptions().queryKey, data => useVstChainStore.setState({
    monitoring: data.running ?? false, monitorPaused: data.paused ?? false,
    monitorPosition: data.position ?? 0, monitorDuration: data.duration ?? 0,
  })),
];
if (import.meta.hot) import.meta.hot.dispose(() => detachVstProjections.forEach(stop => stop()));
