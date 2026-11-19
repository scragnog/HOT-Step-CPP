// lyricEditorSettings.ts — the lyric editor's browser-persisted settings.
//
// Two kinds live here:
//
//   editorPrefs   browser-wide view choices (zoom, Follow, Auto save), moved
//                 from `hs-lyricEditor*` keys to versioned ones, with the old
//                 keys read once and migrated;
//   realign       the Repair timings settings, kept per track because a sparse
//                 rap verse and a legato ballad do not want the same pass.
//
// Per-track keys are `realign-<track scope>:1` — namespace, scope, version, as
// `versionedStore` lays them out. The scope comes from the audio URL
// (`trackScope`), which is the identity the editor and the server already use.
//
// Defaults are data, not an if-chain at every call site: `DEFAULT_REALIGN_SETTINGS`
// is what the repair popover opens on the first time and what a missing or
// half-written value is filled from.

import { createVersionedStore, type VersionedStore } from './versionedStore';

/** What the Repair timings pass is allowed to change. */
export interface RealignSettings {
  /** Copy word offsets from an earlier occurrence of the same line. */
  repeats: boolean;
  /** Shorten a tail that runs into the next onset or over silence. */
  cutBack: boolean;
  /** Move a low-confidence onset onto a voiced rise nearby. */
  snap: boolean;
  /** Push a word that starts before the previous one ends. */
  monotonic: boolean;
}

export const DEFAULT_REALIGN_SETTINGS: RealignSettings = {
  repeats: true,
  cutBack: true,
  snap: true,
  monotonic: true,
};

const REALIGN_VERSION = 1;
const realignStores = new Map<string, VersionedStore<RealignSettings>>();

function realignStore(scope: string): VersionedStore<RealignSettings> {
  const cached = realignStores.get(scope);
  if (cached) return cached;
  const store = createVersionedStore<RealignSettings>({
    namespace: 'realign',
    version: REALIGN_VERSION,
    scope,
    // Every field is optional in what is stored: a key written by an older
    // build, or a partly-written one, is filled from the defaults rather than
    // rejected wholesale.
    migrate: value => {
      if (!value || typeof value !== 'object') return null;
      const raw = value as Record<string, unknown>;
      const pick = (key: keyof RealignSettings) =>
        (typeof raw[key] === 'boolean' ? raw[key] : DEFAULT_REALIGN_SETTINGS[key]);
      return {
        repeats: pick('repeats'),
        cutBack: pick('cutBack'),
        snap: pick('snap'),
        monotonic: pick('monotonic'),
      };
    },
  });
  realignStores.set(scope, store);
  return store;
}

export function loadRealignSettings(scope: string): RealignSettings {
  return realignStore(scope).read() ?? { ...DEFAULT_REALIGN_SETTINGS };
}

export function saveRealignSettings(scope: string, settings: RealignSettings): void {
  realignStore(scope).write({ ...settings });
}

/** The old flag keys held `'1'`/`'0'`; JSON turns those into numbers. */
function asBoolean(value: unknown): boolean | null {
  if (value === true || value === 1 || value === '1') return true;
  if (value === false || value === 0 || value === '0') return false;
  return null;
}

interface BooleanPref {
  read(fallback: boolean): boolean;
  write(value: boolean): void;
}

function flagPreference(namespace: string, legacyKey: string): BooleanPref {
  const store = createVersionedStore<boolean>({
    namespace,
    version: 1,
    legacyKeys: [legacyKey],
    migrate: asBoolean,
  });
  return {
    read(fallback) { return store.read() ?? fallback; },
    write(value) { store.write(value); },
  };
}

const zoomStore = createVersionedStore<number>({
  namespace: 'lyric-editor-zoom',
  version: 1,
  legacyKeys: ['hs-lyricEditorZoom'],
  migrate: value => {
    const zoom = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(zoom) && zoom > 0 ? zoom : null;
  },
});

/**
 * The editor's browser-wide view choices. `read` takes the caller's fallback
 * so the layout constants stay with the layout, not with storage.
 */
export const editorPrefs = {
  follow: flagPreference('lyric-editor-follow', 'hs-lyricEditorFollow'),
  autosave: flagPreference('lyric-editor-autosave', 'hs-lyricEditorAutosave'),
  zoom: {
    read(fallback: number) { return zoomStore.read() ?? fallback; },
    write(value: number) { zoomStore.write(value); },
  },
};
