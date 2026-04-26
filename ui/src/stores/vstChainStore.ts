// vstChainStore.ts — Zustand store for VST3 post-processing chain
//
// Manages the global VST3 plugin chain: scanning, ordering, enable/disable,
// and GUI launching. Chain state is persisted server-side.

import { create } from 'zustand';
import { vstApi, type VstPlugin, type VstChainEntry } from '../services/api';

interface VstChainState {
  // Available plugins (from scan)
  plugins: VstPlugin[];
  scanning: boolean;
  scanError: string | null;

  // Active chain (persisted on server)
  chain: VstChainEntry[];
  chainLoaded: boolean;

  // Monitor (real-time playback)
  monitoring: boolean;
  monitorPosition: number;
  monitorDuration: number;

  // Actions
  scanPlugins: () => Promise<void>;
  loadChain: () => Promise<void>;
  addToChain: (plugin: VstPlugin) => Promise<void>;
  removeFromChain: (uid: string) => Promise<void>;
  toggleEnabled: (uid: string) => Promise<void>;
  reorderChain: (fromIndex: number, toIndex: number) => Promise<void>;
  openGui: (plugin: VstChainEntry) => Promise<void>;
  chainEnabled: () => boolean;
  startMonitor: (trackPath: string) => Promise<void>;
  stopMonitor: () => Promise<void>;
  switchMonitorTrack: (trackPath: string) => Promise<void>;
  seekMonitor: (position: number) => Promise<void>;
  pollMonitorStatus: () => Promise<void>;
}

export const useVstChainStore = create<VstChainState>((set, get) => ({
  plugins: [],
  scanning: false,
  scanError: null,
  chain: [],
  chainLoaded: false,
  monitoring: false,
  monitorPosition: 0,
  monitorDuration: 0,

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
    } catch (err) {
      console.error('[VST] Failed to open GUI:', err);
    }
  },

  chainEnabled: () => {
    return get().chain.some(p => p.enabled);
  },

  startMonitor: async (trackPath: string) => {
    try {
      await vstApi.monitorStart(trackPath);
      set({ monitoring: true });
    } catch (err) {
      console.error('[VST] Failed to start monitor:', err);
    }
  },

  stopMonitor: async () => {
    try {
      await vstApi.monitorStop();
      set({ monitoring: false });
    } catch (err) {
      console.error('[VST] Failed to stop monitor:', err);
    }
  },

  switchMonitorTrack: async (trackPath: string) => {
    try {
      await vstApi.monitorSwitch(trackPath);
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
    try {
      const { running, position, duration } = await vstApi.monitorStatus();
      set({ monitoring: running, monitorPosition: position, monitorDuration: duration });
    } catch {
      set({ monitoring: false, monitorPosition: 0, monitorDuration: 0 });
    }
  },
}));
