// vstChainStore.ts — Zustand store for VST3 post-processing chain
//
// Manages the global VST3 plugin chain: scanning, ordering, enable/disable,
// and GUI launching. Chain state is persisted server-side.

import { create } from 'zustand';
import { vstApi, type VstPlugin, type VstChainEntry } from '../services/api';
import { hashImportValue, preferencesApi, type VstChainPresetBody } from '../services/preferencesApi';

// ── Preset helpers (named chain snapshots) ───────────────────────────────────
//
// Presets now live on the server (preferences.vst-chain-preset, installation
// scope — one shared owner per machine, same as the active chain). The
// browser key is kept as a one-time import source and is never written to
// again; `presets` is hydrated from the server on first use.

const PRESETS_KEY = 'vst-chain-presets';
const MIGRATED_FLAG = 'vst-chain-presets:server-migrated';

function loadLegacyPresetsFromStorage(): Record<string, VstChainEntry[]> {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY) || '{}'); } catch { return {}; }
}

/** Import every legacy localStorage preset once (per name+content). Runs at
 *  most once per browser profile; failures are retried on the next load
 *  since the flag is only set after the import call returns. */
async function importLegacyPresetsOnce(): Promise<void> {
  if (localStorage.getItem(MIGRATED_FLAG)) return;
  const legacy = loadLegacyPresetsFromStorage();
  const names = Object.keys(legacy);
  if (names.length > 0) {
    const items = await Promise.all(names.map(async name => {
      const body: VstChainPresetBody = { name, entries: legacy[name]! as VstChainPresetBody['entries'] };
      const raw = JSON.stringify(body);
      return { storageKey: `${PRESETS_KEY}:${name}`, sourceHash: await hashImportValue(raw), name, body };
    }));
    // A name collision from a prior partial import keeps both rather than guessing which is current.
    await preferencesApi.presets.import('vst-chain', items.map(i => ({ ...i, resolution: 'keep-both' as const })));
  }
  try { localStorage.setItem(MIGRATED_FLAG, '1'); } catch {}
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

  // Named chain snapshots — persisted server-side (preferences.vst-chain-preset)
  presets: Record<string, VstChainEntry[]>;
  presetsLoaded: boolean;
  /** Document id + revision per preset name, for update/delete. Not public API. */
  presetDocs: Record<string, { id: string; revision: number }>;

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
  savePreset: (name: string) => Promise<void>;
  loadPreset: (name: string) => Promise<void>;
  deletePreset: (name: string) => Promise<void>;
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
  presets: loadLegacyPresetsFromStorage(),
  presetsLoaded: false,
  presetDocs: {},

  loadPresets: async () => {
    try {
      await importLegacyPresetsOnce();
      const { documents } = await preferencesApi.presets.list<VstChainPresetBody>('vst-chain');
      const presets: Record<string, VstChainEntry[]> = {};
      const presetDocs: Record<string, { id: string; revision: number }> = {};
      for (const d of documents) { presets[d.body.name] = d.body.entries; presetDocs[d.body.name] = { id: d.id, revision: d.revision }; }
      set({ presets, presetDocs, presetsLoaded: true });
    } catch (err) {
      console.error('[VST] Failed to load presets:', err);
      set({ presetsLoaded: true });
    }
  },

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
    const { chain, presets, presetDocs } = get();
    const entries = chain.map(p => ({ ...p }));
    set({ presets: { ...presets, [name]: entries } });
    try {
      const existing = presetDocs[name];
      const body: VstChainPresetBody = { name, entries };
      const { document } = existing
        ? await preferencesApi.presets.update('vst-chain', existing.id, existing.revision, body)
        : await preferencesApi.presets.create('vst-chain', body);
      set(s => ({ presetDocs: { ...s.presetDocs, [name]: { id: document.id, revision: document.revision } } }));
    } catch (err) {
      console.error('[VST] Failed to save preset:', err);
    }
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

  deletePreset: async (name: string) => {
    const { presets, presetDocs } = get();
    const newPresets = { ...presets };
    delete newPresets[name];
    set({ presets: newPresets });
    const existing = presetDocs[name];
    if (!existing) return;
    try {
      await preferencesApi.presets.remove('vst-chain', existing.id, existing.revision);
      set(s => { const docs = { ...s.presetDocs }; delete docs[name]; return { presetDocs: docs }; });
    } catch (err) {
      console.error('[VST] Failed to delete preset:', err);
    }
  },
}));

useVstChainStore.getState().loadPresets();
