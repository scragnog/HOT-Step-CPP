// vstChainStore.ts — Zustand store for VST3 post-processing chain
//
// Manages the global VST3 plugin chain: scanning, ordering, enable/disable,
// and GUI launching. Chain state is persisted server-side.

import { create } from 'zustand';
import { vstApi, type VstPlugin, type VstChainEntry } from '../services/api';
import { hashImportValue, type VstChainPresetBody } from '../services/preferencesApi';
import { PresetCollection, type PresetCollectionSnapshot } from '../services/presetCollection';

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
  try {
    const v = JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}');
    return v && typeof v === 'object' && !Array.isArray(v) ? v : {};
  } catch { return {}; }
}

export function createVstPresetCollection(): PresetCollection<VstChainPresetBody> {
  return new PresetCollection<VstChainPresetBody>({
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
      const { plugins } = await vstApi.scan();
      set({ plugins, scanning: false });
    } catch (err: any) {
      set({ scanning: false, scanError: err.message });
    }
  },

  loadChain: async () => {
    try {
      const { plugins } = await vstApi.getChain();
      set({ chain: plugins || [], chainLoaded: true });
    } catch {
      set({ chain: [], chainLoaded: true });
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
      const result = await vstApi.updateChain(newChain);
      set({ chain: result.plugins });
    } catch (err) {
      console.error('[VST] Failed to update chain:', err);
    }
  },

  removeFromChain: async (uid: string) => {
    const { chain } = get();
    const newChain = chain.filter(p => p.uid !== uid);
    set({ chain: newChain });

    try {
      const result = await vstApi.updateChain(newChain);
      set({ chain: result.plugins });
    } catch (err) {
      console.error('[VST] Failed to update chain:', err);
    }
  },

  toggleEnabled: async (uid: string) => {
    const { chain } = get();
    const newChain = chain.map(p =>
      p.uid === uid ? { ...p, enabled: !p.enabled } : p
    );
    set({ chain: newChain });

    try {
      const result = await vstApi.updateChain(newChain);
      set({ chain: result.plugins });
    } catch (err) {
      console.error('[VST] Failed to update chain:', err);
    }
  },

  reorderChain: async (fromIndex: number, toIndex: number) => {
    const { chain } = get();
    const newChain = [...chain];
    const [moved] = newChain.splice(fromIndex, 1);
    newChain.splice(toIndex, 0, moved);
    set({ chain: newChain });

    try {
      const result = await vstApi.updateChain(newChain);
      set({ chain: result.plugins });
    } catch (err) {
      console.error('[VST] Failed to update chain:', err);
    }
  },

  openGui: async (plugin: VstChainEntry) => {
    try {
      await vstApi.openGui(plugin.path, plugin.uid);
      set(s => ({
        openGuiUids: [...s.openGuiUids.filter(u => u !== plugin.uid), plugin.uid],
        // flag that state files may be dirty if monitor is running
        pendingGuiChanges: s.monitoring ? true : s.pendingGuiChanges,
      }));
    } catch (err) {
      console.error('[VST] Failed to open GUI:', err);
    }
  },

  closeGui: (uid: string) => {
    set(s => ({ openGuiUids: s.openGuiUids.filter(u => u !== uid) }));
  },

  chainEnabled: () => {
    return get().chain.some(p => p.enabled);
  },

  startMonitor: async (trackPath: string) => {
    try {
      await vstApi.monitorStart(trackPath);
      set({ monitoring: true, monitorPaused: false, monitorTrackPath: trackPath, pendingGuiChanges: false });
    } catch (err) {
      console.error('[VST] Failed to start monitor:', err);
    }
  },

  stopMonitor: async () => {
    try {
      await vstApi.monitorStop();
      set({ monitoring: false, monitorPaused: false, monitorPosition: 0, pendingGuiChanges: false });
    } catch (err) {
      console.error('[VST] Failed to stop monitor:', err);
    }
  },

  restartMonitor: async () => {
    const { monitorTrackPath } = get();
    if (!monitorTrackPath) return;
    try {
      // Stop first, then restart with the same track
      await fetch('/api/vst/monitor/restart', { method: 'POST' });
      set({ monitorPaused: false, monitorPosition: 0, pendingGuiChanges: false });
    } catch (err) {
      console.error('[VST] Failed to restart monitor:', err);
      // Fallback: manual stop + start
      try {
        await vstApi.monitorStop();
        await vstApi.monitorStart(monitorTrackPath);
        set({ monitoring: true, monitorPaused: false, monitorPosition: 0, pendingGuiChanges: false });
      } catch (e) {
        console.error('[VST] Restart fallback also failed:', e);
      }
    }
  },

  pauseMonitor: async () => {
    try {
      const res = await fetch('/api/vst/monitor/pause', { method: 'POST' });
      if (res.ok) set({ monitorPaused: true });
    } catch (err) {
      console.error('[VST] Failed to pause monitor:', err);
    }
  },

  resumeMonitor: async () => {
    try {
      const res = await fetch('/api/vst/monitor/resume', { method: 'POST' });
      if (res.ok) set({ monitorPaused: false });
    } catch (err) {
      console.error('[VST] Failed to resume monitor:', err);
    }
  },

  switchMonitorTrack: async (trackPath: string) => {
    try {
      await vstApi.monitorSwitch(trackPath);
      set({ monitorTrackPath: trackPath });
    } catch (err) {
      console.error('[VST] Failed to switch monitor track:', err);
    }
  },

  seekMonitor: async (position: number) => {
    try {
      await vstApi.monitorSeek(position);
    } catch (err) {
      console.error('[VST] Failed to seek monitor:', err);
    }
  },

  pollMonitorStatus: async () => {
    // Hit the endpoint directly (not via vstApi) to read the raw paused field.
    try {
      const res = await fetch("/api/vst/monitor/status");
      const data = await res.json();
      set({
        monitoring: data.running ?? false,
        monitorPaused: data.paused ?? false,
        monitorPosition: data.position ?? 0,
        monitorDuration: data.duration ?? 0,
      });
    } catch {
      set({ monitoring: false, monitorPaused: false, monitorPosition: 0, monitorDuration: 0 });
    }
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
      const result = await vstApi.updateChain(preset);
      set({ chain: result.plugins });
    } catch (err) {
      console.error('[VST] Failed to apply preset chain:', err);
    }
  },

  deletePreset: (key: string) => get().presetCollection.remove(key),
  };
});

/** Test-only: a fresh preset collection, as in a newly opened browser. */
export function _resetVstPresetsForTests(): void {
  useVstChainStore.setState(attachPresets(useVstChainStore.setState));
}
