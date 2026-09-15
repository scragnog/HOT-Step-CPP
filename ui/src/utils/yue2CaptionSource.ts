/**
 * yue2CaptionSource.ts — where a YuE2 render's caption comes from.
 *
 * The sibling of mm3CaptionSource.ts, and the differences are all consequences
 * of one thing: a YuE2 AR adapter is trained on WHOLE SONGS with that song's
 * own caption, under `--caption-dropout 0.5`. Every caption in the adapter's
 * training dataset is therefore an in-distribution prompt for it, and picking
 * one steers the render towards that track's character rather than the album's
 * average. So the same three-way control applies:
 *
 *   auto    the dataset track whose tempo is nearest this song's. Ties go to
 *           the first in dataset order.
 *   track   a specific dataset track the user picked, by name.
 *   custom  (default) the caption the user typed, exactly as before this existed.
 *
 * Two things differ from MM3 on purpose:
 *
 *   1. The choice is keyed by ADAPTER PATH, not by lyrics-set or generation id.
 *      A caption list belongs to the adapter's training dataset, and the same
 *      adapter is in force across every song until it is switched — YuE2 merges
 *      the delta into the resident LM at load, so the adapter genuinely is
 *      global state (see Yue2LmAdapterDropdown).
 *   2. `custom` is the DEFAULT, where MM3 defaults to `auto`. MM3's control only
 *      appears after a Send-to-Create handoff, so it can never take a caption
 *      box the user typed into. This one appears the moment a captioned adapter
 *      is selected, and silently replacing a typed caption would be theft.
 *
 * The caption a track contributes is `styled`: the caption plus the
 * "<genre>, <bpm> BPM, key of <key>." tail the trainer appended to it. The
 * server composes it with the same function the trainer used
 * (server/src/services/backends/yue2/style.ts) — the client never rebuilds that
 * sentence, because a near-miss is off-distribution in exactly the way this
 * feature exists to avoid. The trigger word is NOT part of it: generate.ts
 * wraps whatever the caption box holds in the adapter's own style template.
 */

import { useBackendStore } from '../stores/backendStore';

/** The registered id of the YuE2 backend (server/src/services/backends/yue2/index.ts). */
export const YUE2_BACKEND_ID = 'yue2';

export type Yue2CaptionMode = 'auto' | 'track' | 'custom';

/** One song from the adapter's training dataset, as `sources[]` records it in
 *  the dataset's `yue2_preprocess.json`. `bpm` is a string there, not a number —
 *  it is a sidecar field, not a measurement. */
export interface Yue2SourceTrack {
  name: string;
  caption: string;
  genre?: string;
  bpm?: string | number;
  key?: string;
  /** Caption + metadata tail, composed server-side. Absent for a server that
   *  predates it, in which case the bare caption is the honest fallback. */
  styled?: string;
}

/** The per-adapter choice. Absent from storage means `{ mode: 'custom' }`. */
export interface Yue2CaptionSelection {
  mode: Yue2CaptionMode;
  /** Only meaningful for mode 'track'. */
  selectedName?: string;
  /** What the caption box held before a dataset caption took it over, so
   *  switching back to Custom returns the user's own words rather than leaving
   *  them holding the dataset track's. */
  customCaption?: string;
}

/** `hs-yue2CaptionSource:<absolute adapter path>` → Yue2CaptionSelection */
export const YUE2_CAPTION_SOURCE_PREFIX = 'hs-yue2CaptionSource:';
/** `hs-yue2SourceTracks:<absolute adapter path>` → Yue2SourceTrack[] */
export const YUE2_SOURCE_TRACKS_PREFIX = 'hs-yue2SourceTracks:';

// ── Storage ──────────────────────────────────────────────────────────────────

function _read<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? null : (JSON.parse(raw) as T);
  } catch {
    return null;
  }
}

function _write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage full or unavailable — the choice simply won't persist */
  }
}

export function readYue2CaptionSelection(adapterPath: string): Yue2CaptionSelection {
  if (!adapterPath) return { mode: 'custom' };
  const stored = _read<Yue2CaptionSelection>(YUE2_CAPTION_SOURCE_PREFIX + adapterPath);
  if (!stored || (stored.mode !== 'auto' && stored.mode !== 'track' && stored.mode !== 'custom')) {
    return { mode: 'custom' };
  }
  return stored;
}

export function writeYue2CaptionSelection(adapterPath: string, sel: Yue2CaptionSelection): void {
  if (!adapterPath) return;
  _write(YUE2_CAPTION_SOURCE_PREFIX + adapterPath, sel);
}

export function readYue2SourceTracks(adapterPath: string): Yue2SourceTrack[] {
  if (!adapterPath) return [];
  return _read<Yue2SourceTrack[]>(YUE2_SOURCE_TRACKS_PREFIX + adapterPath) ?? [];
}

export function cacheYue2SourceTracks(adapterPath: string, tracks: Yue2SourceTrack[]): void {
  if (!adapterPath) return;
  _write(YUE2_SOURCE_TRACKS_PREFIX + adapterPath, tracks);
}

/**
 * The adapter's dataset captions, from the server.
 *
 * `GET /api/training/yue2-adapter-captions?adapter=<path>` answers
 * `{ tracks: Yue2SourceTrack[] }`. The adapter path is the key because it is the
 * only handle the UI has: `lmAdapterMeta` carries a dataset NAME, not an id, and
 * matching on that would be a guess. The server walks adapter path → run
 * manifest → the `yue2_preprocess.json` that run trained against → `sources[]`
 * with a non-empty caption. An adapter whose dataset or manifest is gone answers
 * with an empty list, which reads here as "no picker".
 */
export async function fetchYue2SourceTracks(adapterPath: string): Promise<Yue2SourceTrack[]> {
  if (!adapterPath) return [];
  try {
    const res = await fetch(
      `/api/training/yue2-adapter-captions?adapter=${encodeURIComponent(adapterPath)}`);
    if (!res.ok) return [];
    const data = await res.json() as { tracks?: Yue2SourceTrack[] };
    const tracks = (data.tracks ?? []).filter(t => t && t.name && (t.styled || t.caption));
    cacheYue2SourceTracks(adapterPath, tracks);
    return tracks;
  } catch {
    return [];
  }
}

/** Cached first, server second. The cache is what lets the non-React render
 *  paths resolve a pick synchronously: a selection can only be non-custom if
 *  the picker ran, and the picker is what fills the cache. */
export async function ensureYue2SourceTracks(adapterPath: string): Promise<Yue2SourceTrack[]> {
  const cached = readYue2SourceTracks(adapterPath);
  if (cached.length) return cached;
  return fetchYue2SourceTracks(adapterPath);
}

/** Which adapter's training dataset the caption list comes from, absolute path,
 *  '' for the base model. The catalogue is the one authority — it is what the
 *  picker POSTs to and re-reads, so a mirror of our own could only ever
 *  disagree.
 *
 *  THE AR IS PREFERRED, and the order is not arbitrary: the AR half is the one
 *  trained on each song's own caption under caption dropout, so its dataset's
 *  captions are the in-distribution prompts this feature exists to offer. The
 *  NAR is the fallback for a stack that has only that half.
 *
 *  `lmAdapter` is read last and only for compatibility: the picker carried one
 *  slot until the AR/NAR split (58aff871), and a client reading the old key
 *  against a split catalogue gets '' — which is how this control silently
 *  stopped appearing in Create with both halves plainly loaded. */
export function yue2CaptionAdapterPath(defaults: Record<string, unknown> | undefined): string {
  const pick = (k: string): string => {
    const v = defaults?.[k];
    return typeof v === 'string' ? v.trim() : '';
  };
  return pick('lmAdapterAr') || pick('lmAdapterNar') || pick('lmAdapter');
}

export function activeYue2AdapterPath(): string {
  const catalogue = useBackendStore.getState().models[YUE2_BACKEND_ID];
  return yue2CaptionAdapterPath(catalogue?.defaults as Record<string, unknown> | undefined);
}

// ── Resolution ───────────────────────────────────────────────────────────────

/** The caption text a track contributes to the box. */
export function yue2TrackCaption(track: Yue2SourceTrack): string {
  return (track.styled || track.caption || '').trim();
}

/** The manifest's `bpm` is whatever the sidecar said, so it can be "128",
 *  "128.5", "~128" or nothing at all. */
export function yue2TrackBpm(track: Yue2SourceTrack): number | undefined {
  const n = typeof track.bpm === 'number' ? track.bpm : parseFloat(String(track.bpm ?? ''));
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

/** The dataset track whose tempo is nearest `bpm`. Ties go to the earlier
 *  track, hence the strictly-smaller test; a song with no tempo, or a dataset
 *  whose tracks carry none, falls back to the first — there is nothing to be
 *  nearest to. */
export function pickNearestBpmTrack(
  tracks: Yue2SourceTrack[],
  bpm: number | undefined,
): Yue2SourceTrack | null {
  if (tracks.length === 0) return null;
  if (!bpm || bpm <= 0) return tracks[0];

  let best: Yue2SourceTrack | null = null;
  let bestDist = Infinity;
  for (const track of tracks) {
    const tempo = yue2TrackBpm(track);
    if (tempo === undefined) continue;
    const dist = Math.abs(tempo - bpm);
    if (dist < bestDist) { bestDist = dist; best = track; }
  }
  return best ?? tracks[0];
}

export interface Yue2ResolvedCaption {
  /** The caption to render with. */
  caption: string;
  /** The mode that actually applied — a pick whose track is no longer in the
   *  dataset degrades to one that works, so this is not always the stored mode. */
  mode: Yue2CaptionMode;
  /** Dataset track the caption came from, for the "From dataset track:" line. */
  fromName?: string;
}

/**
 * Resolve the caption for one render.
 *
 * `own` is the user's caption — the caption box, or a written song's own. Every
 * degradation lands back on it: an empty dataset, a picked track that has since
 * gone, a track row with nothing in it.
 */
export function resolveYue2Caption(
  own: string,
  bpm: number | undefined,
  tracks: Yue2SourceTrack[],
  sel: Yue2CaptionSelection,
): Yue2ResolvedCaption {
  const mine = (own || '').trim();
  if (sel.mode === 'custom') return { caption: mine, mode: 'custom' };

  if (sel.mode === 'track' && sel.selectedName) {
    const hit = tracks.find(track => track.name === sel.selectedName);
    const caption = hit ? yue2TrackCaption(hit) : '';
    if (caption) return { caption, mode: 'track', fromName: hit!.name };
    // The dataset changed under the choice — fall through to auto.
  }

  const nearest = pickNearestBpmTrack(tracks, bpm);
  const caption = nearest ? yue2TrackCaption(nearest) : '';
  if (caption) return { caption, mode: 'auto', fromName: nearest!.name };
  return { caption: mine, mode: 'custom' };
}

/** Resolve straight from storage — the form the non-React render paths use.
 *  Returns the song's own caption whenever nothing has been picked, so callers
 *  can use it unconditionally on this backend. */
export function resolveYue2CaptionForGeneration(
  gen: { bpm?: number; caption?: string | null },
): Yue2ResolvedCaption {
  const adapter = activeYue2AdapterPath();
  const own = gen.caption || '';
  if (!adapter) return { caption: own.trim(), mode: 'custom' };
  return resolveYue2Caption(own, gen.bpm, readYue2SourceTracks(adapter), readYue2CaptionSelection(adapter));
}

// ── Album presets ────────────────────────────────────────────────────────────

/** Apply an album preset's two YuE2 halves to the engine.
 *
 *  NOT a request param, unlike MM3's `mm3LmAdapter`: on this backend the
 *  adapter is merged into the resident LM at load, so the selection IS engine
 *  state and the only way to set it is the same POST the picker makes
 *  (Yue2LmAdapterDropdown). A preset that only wrote a param would change
 *  nothing.
 *
 *  Both halves travel together, and an ABSENT half is sent as '' rather than
 *  omitted — the same rule the MM3 path follows for the same reason: the
 *  selection persists across sessions, so an album with no NAR would otherwise
 *  keep rendering through the previous album's. Scales are left alone; they are
 *  the user's dials, not the album's.
 *
 *  Returns true when the engine accepted the selection. Never throws. */
export async function applyYue2PresetAdapters(
  preset: { yue2_ar_adapter_path?: string | null; yue2_nar_adapter_path?: string | null } | null | undefined,
): Promise<boolean> {
  const ar = String(preset?.yue2_ar_adapter_path ?? '').trim();
  const nar = String(preset?.yue2_nar_adapter_path ?? '').trim();
  try {
    return await useBackendStore.getState().selectModels(
      { lmAdapterAr: ar, lmAdapterNar: nar }, YUE2_BACKEND_ID);
  } catch {
    return false;
  }
}
