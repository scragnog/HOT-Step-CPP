// vstChainStore.ts — Zustand store for VST3 post-processing chain
//
// Manages the global VST3 plugin chain: scanning, ordering, enable/disable,
// and GUI launching. Chain state is persisted server-side.

import { create } from 'zustand';
import { vstApi, type VstPlugin, type VstChainEntry } from '../services/api';

// ── Preset helpers (named chain snapshots, persisted to localStorage) ────────

const PRESETS_KEY = 'vst-chain-presets';

function loadPresetsFromStorage(): Record<string, VstChainEntry[]> {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}'); } catch { return {}; }
}

function savePresetsToStorage(presets: Record<string, VstChainEntry[]>): void {
  try { localStorage.setItem(PRESETS_KEY, JSON.stringify(presets)); } catch {}
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

  // Named chain snapshots — persisted to localStorage
  presets: Record<string, VstChainEntry[]>;

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
  savePreset: (name: string) => void;
  loadPreset: (name: string) => Promise<void>;
  deletePreset: (name: string) => void;
}

export const useVstChainStore = create<VstChainState>((set, get) => ({
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
  presets: loadPresetsFromStorage(),

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

  savePreset: (name: string) => {
    const { chain, presets } = get();
    const newPresets = { ...presets, [name]: chain.map(p => ({ ...p })) };
    set({ presets: newPresets });
    savePresetsToStorage(newPresets);
  },

  loadPreset: async (name: string) => {
    const { presets } = get();
    const preset = presets[name];
    if (!preset) return;
    set({ chain: preset });
    try {
      const result = await vstApi.updateChain(preset);
      set({ chain: result.plugins });
    } catch (err) {
      console.error('[VST] Failed to apply preset chain:', err);
    }
  },

  deletePreset: (name: string) => {
    const { presets } = get();
    const newPresets = { ...presets };
    delete newPresets[name];
    set({ presets: newPresets });
    savePresetsToStorage(newPresets);
  },
}));
