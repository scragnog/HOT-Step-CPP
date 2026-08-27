// backendStore.ts — Zustand store for the multi-backend registry
//
// Backends are the registered generation engines behind /api/backends — ACE-Step
// 1.5 today, future engines (e.g. MiniMax-Music3) later. See
// docs/plans/multi-backend-architecture.md §4.5. Almost every install only ever
// has ONE backend registered ('ace'); BackendToggle stays hidden until a second
// one exists, so this store is inert (one entry, no UI) for the common case.
//
// Fetch-on-demand, not polled: callers refetch capabilities explicitly (on
// mount / on backend switch) rather than the store running an interval loop.

import { create } from 'zustand';

// ── Active-backend mirror ──
//
// The server owns the active backend (persisted in its `settings` table), but
// it is one fetch away and the store is constructed synchronously at import.
// globalParamsStore hydrates its PER-BACKEND fields (solver / scheduler /
// guidance, plugin + extension params) from this id at that same moment, so
// booting on 'ace' and correcting a tick later would flash ACE's sampler
// settings — and, worse, a knob moved inside that window would be written to
// ACE's slot while MiniMax-Music3 was the active backend.
//
// So mirror the last known id locally and start from it. fetchBackends()
// overwrites it with the server's answer immediately; the two only ever
// disagree if the DB was changed by another client.
const ACTIVE_KEY = 'hs-activeBackend';

function readActiveMirror(): string {
  try { return localStorage.getItem(ACTIVE_KEY) || 'ace'; } catch { return 'ace'; }
}

function writeActiveMirror(id: string): void {
  try { localStorage.setItem(ACTIVE_KEY, id); } catch { /* full / blocked */ }
}

// ── Types (mirror server/src/services/backends/types.ts — §4.2) ──

export interface BackendInfo {
  id: string;
  displayName: string;
  resourcePool: 'gpu' | 'remote';
  active: boolean;
}

export interface BackendCoreCapabilities {
  /** `auto`: the backend picks the length itself when none is asked for (MM3's
   *  planner LM stops on its own EOS token). Optional because a manifest cached
   *  from a server older than the flag has no such field. */
  duration: { max: number; auto?: boolean };
  bpm: boolean;
  keyscale: boolean;
  negativePrompt: boolean;
  batch: { max: number };
  seed: boolean;
  // Open — backends may report extra core-ish knobs (mirrors the server-side
  // index signature in backends/types.ts). Nothing here is guaranteed to
  // exist; check with `=== true` / `?.`, never assume presence.
  [key: string]: unknown;
}

export interface BackendFeatureCapabilities {
  /** Backend exposes a selectable model catalogue. Separate from `lm` —
   *  MiniMax-Music3 has selectable weights (a quant ladder) but no ACE-style
   *  LM/CoT stage, and gating the Models cluster on `lm` hid its picker. */
  models: boolean;
  lm: boolean;
  plugins: boolean;
  /** The Lua sampler plugins (solvers, schedulers, guidance) actually RUN on
   *  this backend's denoiser. Distinct from `plugins`, which selects WHICH
   *  Generation dropdown renders — MiniMax-Music3 needs the generic one for its
   *  own knobs while still running the plugins, so it reports
   *  `plugins: false, samplerPlugins: true`. Optional here because a backend
   *  registered by an older server won't send it. */
  samplerPlugins?: boolean;
  adapters: boolean;
  /** The backend exposes runtime LM LoRA adapters (picker + strength dials on
   *  its language/planner stage). Separate from `adapters`, which gates ACE's
   *  DiT adapter stack UI. Optional here because a backend registered by an
   *  older server won't send it. */
  lmAdapters?: boolean;
  /** Model-agnostic post stages (VST chain, reference mastering) run for this
   *  backend. Separate from `plugins`, which also gates the ACE-VAE-coupled
   *  stages (PP-VAE re-encode, Spectral Lifter). */
  postProcess: boolean;
  /** StableStep / SA3 refinement is available for this backend's output. */
  stableStep: boolean;
  /** Whisper transcription of the rendered audio (backend-agnostic). */
  whisper: boolean;
  /** LRC timestamps from the model's own generation-time attention. */
  lyricTimestamps: boolean;
  cover: boolean;
  repaint: boolean;
  lego: boolean;
  extract: boolean;
  streaming: boolean;
  training: boolean;
  midi: boolean;
  stems: boolean;
  understand: boolean;
  conceptSteering: boolean;
}

/** Backend-specific knob schema — rendered generically by PluginControls,
 *  same shape as the Lua plugin param schema (types/pluginTypes.ts). */
/** Which top-bar cluster a declared knob belongs in. Absent means
 *  'generation' — the behaviour before groups existed, and what a manifest
 *  cached from an older server will look like. */
export type BackendExtensionGroup = 'generation' | 'lm';

export interface BackendExtensionParam {
  key: string;
  /** Top-bar cluster this knob renders in. Optional: older manifests have no
   *  such field and every knob falls back to the Generation panel. */
  group?: BackendExtensionGroup;
  type: 'slider' | 'select' | 'toggle' | 'text';
  label: string;
  hint?: string;
  transform?: string;
  default?: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  visible_when?: { key: string; equals: string };
}

export interface BackendCapabilities {
  backend: string;
  up: boolean;
  core: BackendCoreCapabilities;
  features: BackendFeatureCapabilities;
  extensions: BackendExtensionParam[];
}

/** Backend-shaped model catalogue (GET /api/backends/models). Buckets differ
 *  per backend — ACE reports {lm,dit,vae,embedding} file names, MiniMax-Music3
 *  reports {lm,synth} quant ladders — so consumers render what they are given
 *  rather than branching on backend id. */
export interface BackendModelCatalogue {
  backend: string;
  /** True when the backend holds model choice as engine state and accepts
   *  POST /api/backends/models. False (e.g. ACE) means the picker is owned
   *  elsewhere — per-request model names on generate. */
  selectable: boolean;
  buckets: Record<string, string[]>;
  adapters?: string[];
  lmAdapters?: string[];
  /** What is actually in force right now, per bucket. */
  defaults?: Record<string, unknown>;
  /** Optional cosmetic per-option metadata, bucket -> option -> {label,bytes}. */
  meta?: Record<string, Record<string, { label?: string; bytes?: number }>>;
}

interface BackendState {
  backends: BackendInfo[];
  activeBackendId: string;
  /** Per-backend capability manifest, keyed by backend id. */
  capabilities: Record<string, BackendCapabilities>;
  /** Per-backend model catalogue, keyed by backend id. */
  models: Record<string, BackendModelCatalogue>;
  loading: boolean;
  error: string | null;

  fetchBackends: () => Promise<void>;
  fetchCapabilities: (id?: string) => Promise<void>;
  fetchModels: (id?: string) => Promise<void>;
  /** Post a bucket->value selection. Returns true on success. Refetches the
   *  catalogue so `defaults` reflects what the engine actually resolved. */
  selectModels: (selection: Record<string, string>, id?: string) => Promise<boolean>;
  switchBackend: (id: string) => Promise<void>;
}

export const useBackendStore = create<BackendState>((set, get) => ({
  backends: [],
  // Last known active id (mirrored to localStorage, see above) so the
  // per-backend UI state hydrates correctly on the very first render.
  // Falls back to 'ace' — the server's default and every pre-multi-backend
  // install; overwritten by fetchBackends() once the registry responds.
  activeBackendId: readActiveMirror(),
  capabilities: {},
  models: {},
  loading: false,
  error: null,

  fetchBackends: async () => {
    set({ loading: true, error: null });
    try {
      const res = await fetch('/api/backends');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: { backends: BackendInfo[]; activeId: string } = await res.json();
      const activeId = data.activeId || 'ace';
      writeActiveMirror(activeId);
      set({
        backends: Array.isArray(data.backends) ? data.backends : [],
        activeBackendId: activeId,
        loading: false,
      });
    } catch (err) {
      // Advisory — a failed fetch must not break generation. Every consumer
      // (getGlobalParams, BackendToggle) treats an empty/single-entry list as
      // "no multi-backend UI", which is exactly the pre-feature behaviour.
      console.warn('[Backends] fetch failed:', err instanceof Error ? err.message : String(err));
      set({ loading: false, error: err instanceof Error ? err.message : String(err) });
    }
  },

  fetchCapabilities: async (id) => {
    const backendId = id || get().activeBackendId;
    if (!backendId) return;
    try {
      const res = await fetch(`/api/capabilities?backend=${encodeURIComponent(backendId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: BackendCapabilities = await res.json();
      set({ capabilities: { ...get().capabilities, [backendId]: data } });
    } catch (err) {
      console.warn(`[Backends] capabilities fetch failed for "${backendId}":`, err instanceof Error ? err.message : String(err));
    }
  },

  fetchModels: async (id) => {
    const backendId = id || get().activeBackendId;
    if (!backendId) return;
    try {
      const res = await fetch(`/api/backends/models?backend=${encodeURIComponent(backendId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: BackendModelCatalogue = await res.json();
      set({ models: { ...get().models, [backendId]: data } });
    } catch (err) {
      // Advisory, like fetchCapabilities: an empty catalogue renders as
      // "no models found", which is the honest degrade.
      console.warn(`[Backends] models fetch failed for "${backendId}":`, err instanceof Error ? err.message : String(err));
    }
  },

  selectModels: async (selection, id) => {
    const backendId = id || get().activeBackendId;
    if (!backendId) return false;
    try {
      const res = await fetch('/api/backends/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: backendId, selection }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || `HTTP ${res.status}`);
      }
      // Re-read rather than trusting the request: the engine may fall back to
      // a different quant if the requested file vanished.
      await get().fetchModels(backendId);
      return true;
    } catch (err) {
      console.warn('[Backends] model selection failed:', err instanceof Error ? err.message : String(err));
      set({ error: err instanceof Error ? err.message : String(err) });
      return false;
    }
  },

  switchBackend: async (id) => {
    set({ error: null });
    try {
      const res = await fetch('/api/backends/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: { activeId: string } = await res.json();
      writeActiveMirror(data.activeId);
      set({
        activeBackendId: data.activeId,
        backends: get().backends.map(b => ({ ...b, active: b.id === data.activeId })),
      });
      // Refetch on success — the new active backend's manifest may not be
      // cached yet (or may be stale from a previous session).
      await get().fetchCapabilities(data.activeId);
      await get().fetchModels(data.activeId);
    } catch (err) {
      console.warn('[Backends] switch failed:', err instanceof Error ? err.message : String(err));
      set({ error: err instanceof Error ? err.message : String(err) });
    }
  },
}));
