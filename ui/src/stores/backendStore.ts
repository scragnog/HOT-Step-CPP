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
import { queryClient } from '../queries/client';
import { backendsOptions, backendModelsOptions, capabilitiesOptions, usableCapabilities } from '../queries/global-models/options';
import { queryManager } from '../queries/invalidation';
import { keysPlan, writeAndInvalidate } from '../queries/writes';
import { scopeOf } from '../queries/keys';
import { apiClient } from '../services/httpClient';
import { projectQuery } from '../queries/global-models/projection';
import type { Query } from '@tanstack/react-query';

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
  duration: { max: number; auto?: boolean; editable?: boolean };
  captionSource?: 'none' | 'mm3-tracks' | 'yue2-dataset';
  bpm: boolean;
  keyscale: boolean;
  negativePrompt: boolean;
  batch: { max: number };
  seed: boolean;
  // ── Forward-declared optional fields (docs/plans/yue2/01-seam-design.md
  // §4.2) ── typed here so callers get a real type instead of `unknown`
  // through the index signature below. Every field stays OPTIONAL: a
  // manifest from a backend (or an older server) that doesn't set it simply
  // omits it, and nothing may assume presence. None of these are MM3- or
  // YuE2-specific — any backend can declare any of them.
  /** Shape the backend wants its prompt in. `'structured'` = a headed
   *  template (MM3's Structured Caption); `'freeform'` = a plain style +
   *  lyrics prompt (ACE, and YuE2's declared shape). Cosmetic in this slice —
   *  nothing branches on it yet (that's the CreatePanel/MetadataSection wiring
   *  in the doc above, not done here). */
  captionFormat?: 'freeform' | 'structured';
  /** Backend honors a time-signature core param, the same sense `bpm`/
   *  `keyscale` above report support for those two. */
  timeSignature?: boolean;
  /** What this backend's "language" field actually controls: `'vocal'` (sung
   *  language) or `'lyrics'` (the lyrics text's own language). Hint only. */
  languageMeans?: 'vocal' | 'lyrics';
  /** True only when the backend is reachable and ITS OWN weight files
   *  specifically weren't found — never for engine-down or corrupt-file
   *  cases (see GlobalParamBar's models-missing banner). Promoted out of the
   *  index signature below since it's now read generically, not just by MM3. */
  modelsMissing?: boolean;
  /** Paired with `modelsMissing` — a short user-facing explanation (e.g. "both
   *  GGUFs (~24 GB)") shown instead of a generic fallback when present. */
  modelsMissingHint?: string;
  /** Plain-text license line for this backend's weights, e.g. YuE2's "CC
   *  BY-NC 4.0, but the authors have clarified…" notice. Cosmetic only — not
   *  a legal control, just something to surface to the user. */
  license?: string;
  /** Short muted notice to pair with the backend picker — e.g. a weights-
   *  licence reminder with an upstream contact address. Rendered
   *  verbatim by BackendToggle.tsx: never truncated, reworded, or paraphrased. */
  notice?: string;
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
  /** The LM adapter is ENGINE STATE (POST /api/backends/models, `lmAdapter`
   *  bucket) rather than a per-request parameter — YuE2 merges the delta into
   *  the resident weights at load, so picking one evicts the model instead of
   *  riding along on the next generation. Optional: absent means the
   *  per-request kind (MiniMax-Music3). */
  lmAdapterSelectable?: boolean;
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
  /** The render is force-aligned against its own lyrics afterwards, for a
   *  backend whose model cannot time them itself (YuE2). Opt-in per render —
   *  gp.yue2AlignLyrics. */
  forcedAlignment: boolean;
  cover: boolean;
  /** Audio-to-audio transform without a lead sheet or vocal conditioning. */
  coverAudioTransform?: boolean;
  /** Cover by replaying a reference's encoded codes (MM3 planks). */
  coverCodeForcing?: boolean;
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
export type BackendExtensionGroup = 'generation' | 'lm' | 'adapter';

export interface BackendExtensionParam {
  key: string;
  /** Top-bar cluster this knob renders in. Optional: older manifests have no
   *  such field and every knob falls back to the Generation panel. */
  group?: BackendExtensionGroup;
  /** Accordion inside the cluster; absent = rendered flat. A section renders where its first knob is declared. */
  section?: string;
  section_hint?: string;
  type: 'slider' | 'select' | 'toggle' | 'text';
  /** text only: render a multi-line box (lead sheets, anything pasted). */
  multiline?: boolean;
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
  /** Plain-text weights-licence line (e.g. CC BY-NC 4.0 notice); optional, additive. */
  license?: string;
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
  /** Optional per-entry detail for `lmAdapters`, keyed by the same reference
   *  the list carries. `trigger` is the load-bearing one: a YuE2 NAR adapter
   *  does nothing unless its trigger word leads the style prompt. */
  lmAdapterMeta?: Record<string, {
    label?: string;
    runName?: string;
    trigger?: string;
    triggerInferred?: boolean;
    /** Typed in the Adapters dropdown: the file, its run and its dataset carry none. */
    triggerManual?: boolean;
    rank?: number;
    steps?: number;
    bytes?: number;
    /** HeartMuLa: the adapter carries MLP factors, so its MLP dial does something. */
    mlp?: boolean;
    /** HeartMuLa: the variant the adapter was trained on (its home base for re-base). */
    variant?: string;
    dataset?: string;
    final?: boolean;
    /** Mean of the last 20 logged step losses up to this checkpoint. */
    loss?: number;
    /** Held-out minted_val at this checkpoint's step, AR only. */
    valLoss?: number;
  }>;
  /** What is actually in force right now, per bucket. */
  defaults?: Record<string, unknown>;
  /** Optional per-option metadata. A disabledReason prevents selection. */
  meta?: Record<string, Record<string, { label?: string; bytes?: number; disabledReason?: string }>>;
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

// One entry per selectModels call still in flight, in call order. Each call
// removes only its own entry: two saves may carry the same value for a key, so
// value equality cannot say whose it is.
const _pending = new Map<string, Array<Record<string, string>>>();

/** Selections posted by selectModels whose refetch has not landed yet, later
 *  calls winning, so a reader of `defaults` can lay them over the stale
 *  values. Empty otherwise. */
export function pendingSelection(backendId: string): Readonly<Record<string, string>> {
  return Object.assign({}, ...(_pending.get(backendId) ?? []));
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
      const data = await queryClient.query(backendsOptions());
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
      await queryClient.query(capabilitiesOptions(backendId));
    } catch (err) {
      console.warn(`[Backends] capabilities fetch failed for "${backendId}":`, err instanceof Error ? err.message : String(err));
    }
  },

  fetchModels: async (id) => {
    const backendId = id || get().activeBackendId;
    if (!backendId) return;
    try {
      await queryClient.query(backendModelsOptions(backendId));
    } catch (err) {
      // Advisory, like fetchCapabilities: an empty catalogue renders as
      // "no models found", which is the honest degrade.
      console.warn(`[Backends] models fetch failed for "${backendId}":`, err instanceof Error ? err.message : String(err));
    }
  },

  selectModels: async (selection, id) => {
    const backendId = id || get().activeBackendId;
    if (!backendId) return false;
    // Until the refetch lands, `defaults` still holds the previous pick; a job
    // queued in that window must see this selection (pendingSelection).
    const mine = { ...selection };
    _pending.set(backendId, [...(_pending.get(backendId) ?? []), mine]);
    try {
      await queryClient.cancelQueries({ queryKey: backendModelsOptions(backendId).queryKey });
      // Re-read rather than trusting the request: the engine may fall back to
      // a different quant if the requested file vanished. Silent: every caller
      // renders its own inline note on !ok (the adapter-dropdown legs, the
      // backend-models dropdown, PresetSettingsModal) or swallows deliberately
      // (yue2CaptionSource), so their line is the sole notifier.
      await writeAndInvalidate(queryClient,
        () => apiClient.post('/backends/models', { backend: backendId, selection }, { silent: true }),
        keysPlan(backendModelsOptions(backendId).queryKey));
      await get().fetchModels(backendId);
      return true;
    } catch {
      // No global toast — the write is silent and every caller owns its own
      // inline note (or swallows deliberately).
      return false;
    } finally {
      const rest = (_pending.get(backendId) ?? []).filter(p => p !== mine);
      if (rest.length) _pending.set(backendId, rest);
      else _pending.delete(backendId);
    }
  },

  switchBackend: async (id) => {
    set({ error: null });
    try {
      await queryClient.cancelQueries({ queryKey: backendsOptions().queryKey });
      // The plan depends on the write's RESULT (the server resolves the
      // requested id to the active backend), so capture it in a holder: the
      // fn runs first, and the plan reads what the write saw. Refetch on
      // success — the new active backend's manifest may not be cached yet (or
      // may be stale from a previous session). The manager still owns the
      // invalidateQueries call.
      let activeId = id;
      const data = await writeAndInvalidate(queryClient,
        async () => {
          const res = await apiClient.post<{ activeId: string }>('/backends/active', { id });
          activeId = res.activeId;
          return res;
        },
        client => queryManager.invalidate(
          client,
          backendsOptions().queryKey,
          capabilitiesOptions(activeId).queryKey,
          backendModelsOptions(activeId).queryKey,
        ));
      queryClient.setQueryData(backendsOptions().queryKey, old => old && ({
        ...old, activeId: data.activeId,
        backends: old.backends.map(b => ({ ...b, active: b.id === data.activeId })),
      }));
      writeActiveMirror(data.activeId);
      set({
        activeBackendId: data.activeId,
        backends: get().backends.map(b => ({ ...b, active: b.id === data.activeId })),
      });
      await get().fetchCapabilities(data.activeId);
      await get().fetchModels(data.activeId);
    } catch {
      // Transport failure already toasted by the global handler.
    }
  },
}));

// Legacy selectors remain projections of canonical, backend-scoped cache data.
// This listener never fetches and is removed before HMR replaces the module.
function applyBackendProjection(query: Query) {
  const { queryKey, state } = query;
  if (scopeOf(queryKey)?.api !== apiClient.baseUrl || state.data === undefined) return;
  const backend = scopeOf(queryKey)?.backend;
  if (queryKey[1] === 'capabilities' && backend) {
    const data = state.data as BackendCapabilities;
    const previous = useBackendStore.getState().capabilities;
    const usable = usableCapabilities(previous[backend], data);
    if (usable === previous[backend]) return;
    useBackendStore.setState({ capabilities: { ...previous, [backend]: usable } });
  }
  if (queryKey[1] === 'global-models/backend-models' && backend) {
    useBackendStore.setState(s => ({ models: { ...s.models, [backend]: state.data as BackendModelCatalogue } }));
  }
}
queryClient.getQueryCache().getAll().forEach(applyBackendProjection);
const detachQueryProjection = queryClient.getQueryCache().subscribe(event => {
  if (event.type === 'updated' && event.action.type === 'success') applyBackendProjection(event.query);
  if (event.type === 'removed' && scopeOf(event.query.queryKey)?.api === apiClient.baseUrl) {
    const backend = scopeOf(event.query.queryKey)?.backend;
    if (!backend) return;
    const field = event.query.queryKey[1] === 'capabilities' ? 'capabilities'
      : event.query.queryKey[1] === 'global-models/backend-models' ? 'models' : null;
    if (field) {
      const next = { ...useBackendStore.getState()[field] };
      delete next[backend];
      useBackendStore.setState({ [field]: next });
    }
  }
});
const detachBackends = projectQuery<{ backends: BackendInfo[]; activeId: string }>(queryClient,
  backendsOptions().queryKey, data => {
    const activeId = data.activeId || 'ace';
    writeActiveMirror(activeId);
    useBackendStore.setState({ backends: data.backends, activeBackendId: activeId });
  });
if (import.meta.hot) import.meta.hot.dispose(() => { detachQueryProjection(); detachBackends(); });
