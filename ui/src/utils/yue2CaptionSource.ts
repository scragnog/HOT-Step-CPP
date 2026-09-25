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
 *   auto    (default) the dataset track whose tempo is nearest this song's.
 *           Ties go to the first in dataset order.
 *   track   a specific dataset track the user picked, by name.
 *   custom  the caption the user typed, exactly as before this existed.
 *
 * Keyed by training DATASET id, not by adapter path. It used to be keyed by
 * adapter path — the handle was convenient, since an adapter is what the
 * engine actually holds — but that broke two ways: a run folder moved by hand
 * (the join-a-run-folder feature, f3273078) changes the adapter path under a
 * caption list that had nothing to do with the move, and the cleanup job that
 * deletes a run's prepared cache took the captions with it even though the
 * dataset the album is built on never went anywhere. The dataset id is the one
 * handle that survives both: an album's `training_datasets.lyrics_set_id`
 * links it directly, and it never moves or gets swept.
 *
 * `auto` is the default, as on MM3. A training caption is what reliably lands
 * in the album's character, so it is the right thing to reach for first — but
 * unlike MM3's control, this one can appear over a caption box the user has
 * already typed into. So whoever applies the default stashes what was in the
 * box first (`customCaption`), and switching to Custom hands those words back
 * rather than leaving the user holding a dataset track's.
 *
 * The caption a track contributes is `styled`: for joint adapters it is the
 * exact prepared style, including any trained trigger opener; for Legacy it
 * includes the trainer's genre/BPM/key tail. generate.ts avoids adding an
 * opener twice when the selected caption already carries it.
 */

import { useBackendStore } from '../stores/backendStore';

/** The registered id of the YuE2 backend (server/src/services/backends/yue2/index.ts). */
export const YUE2_BACKEND_ID = 'yue2';

export type Yue2CaptionMode = 'auto' | 'track' | 'custom';

/** One song from the dataset, as `GET /api/training/yue2-dataset-captions`
 *  answers it. `bpm` is a string there, not a number — it is a sidecar field,
 *  not a measurement. */
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

/** The per-dataset choice. Absent from storage means `{ mode: 'auto' }`. */
export interface Yue2CaptionSelection {
  mode: Yue2CaptionMode;
  /** Only meaningful for mode 'track'. */
  selectedName?: string;
  /** What the caption box held before a dataset caption took it over, so
   *  switching back to Custom returns the user's own words rather than leaving
   *  them holding the dataset track's. */
  customCaption?: string;
}

/** What Send-to-Create hands the Create panel so it can offer the same
 *  three-way control with no server call of its own. Mirrors
 *  Mm3CaptionSourcesHandoff. */
export interface Yue2CaptionSourcesHandoff extends Yue2CaptionSelection {
  datasetId: string;
  datasetName: string;
  tracks: Yue2SourceTrack[];
}

/** `hs-yue2CaptionSource:ds:<dataset id>` → Yue2CaptionSelection */
export const YUE2_CAPTION_SOURCE_PREFIX = 'hs-yue2CaptionSource:ds:';
/** `hs-yue2SourceTracks3:ds:<dataset id>` → Yue2SourceTrack[]
 *
 *  The `3` is a cache bust: the previous prefixes were keyed by adapter path,
 *  and a client that already cached one of those under the old key must not
 *  go on reading it now that the key means a dataset id. */
export const YUE2_SOURCE_TRACKS_PREFIX = 'hs-yue2SourceTracks3:ds:';
/** `hs-yue2DatasetForLyricsSet:<lyrics set id>` → dataset id.
 *
 *  A written song only carries a lyrics-set id, not a dataset id — the link
 *  lives server-side (`training_datasets.lyrics_set_id`). This is the local
 *  cache of that lookup, filled whenever `fetchYue2CaptionSource` resolves by
 *  lyrics-set, so the two non-React render paths can resolve a dataset id
 *  synchronously without re-asking the server on every song. */
export const YUE2_DATASET_FOR_LYRICS_SET_PREFIX = 'hs-yue2DatasetForLyricsSet:';
/** `hs-yue2CaptionDataset` → dataset id (or '' for none). The Create panel's
 *  current dataset — set by the Send-to-Create handoff or picked directly in
 *  the panel's own Dataset dropdown. */
export const YUE2_CAPTION_DATASET_KEY = 'hs-yue2CaptionDataset';
/** `hs-yue2CaptionSources` → Yue2CaptionSourcesHandoff (Send-to-Create handoff) */
export const YUE2_CAPTION_SOURCES_KEY = 'hs-yue2CaptionSources';

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

/** The choice when nothing has been stored for this dataset. Automatic, so a
 *  freshly selected dataset renders under its own captions without the user
 *  having to find the picker. With no dataset there is nothing to be
 *  automatic about, and the caption box is the only source there is. */
export function defaultYue2CaptionSelection(datasetId: string): Yue2CaptionSelection {
  // Automatic only when an adapter is in force: its training captions are the
  // in-distribution prompts this feature exists to offer. A base-model render
  // keeps the caption the user wrote, even with a dataset linked. Both render
  // paths apply the album's halves before resolving, so the engine's resident
  // pick is the album's here (useAudioGeneration, audioGenQueueStore).
  const adapter = yue2CaptionAdapterPath(
    useBackendStore.getState().models[YUE2_BACKEND_ID]?.defaults as Record<string, unknown> | undefined);
  return { mode: datasetId && adapter ? 'auto' : 'custom' };
}

/** True when this dataset has a stored choice, i.e. the user has been through
 *  the picker. Lets a caller tell "chose Automatic" from "never chose", which
 *  `readYue2CaptionSelection` collapses into the same answer. */
export function hasStoredYue2CaptionSelection(datasetId: string): boolean {
  if (!datasetId) return false;
  const stored = _read<Yue2CaptionSelection>(YUE2_CAPTION_SOURCE_PREFIX + datasetId);
  return !!stored && (stored.mode === 'auto' || stored.mode === 'track' || stored.mode === 'custom');
}

export function readYue2CaptionSelection(datasetId: string): Yue2CaptionSelection {
  if (!datasetId) return { mode: 'custom' };
  const stored = _read<Yue2CaptionSelection>(YUE2_CAPTION_SOURCE_PREFIX + datasetId);
  if (!stored || (stored.mode !== 'auto' && stored.mode !== 'track' && stored.mode !== 'custom')) {
    return defaultYue2CaptionSelection(datasetId);
  }
  return stored;
}

export function writeYue2CaptionSelection(datasetId: string, sel: Yue2CaptionSelection): void {
  if (!datasetId) return;
  _write(YUE2_CAPTION_SOURCE_PREFIX + datasetId, sel);
}

export function readYue2SourceTracks(datasetId: string): Yue2SourceTrack[] {
  if (!datasetId) return [];
  return _read<Yue2SourceTrack[]>(YUE2_SOURCE_TRACKS_PREFIX + datasetId) ?? [];
}

export function cacheYue2SourceTracks(datasetId: string, tracks: Yue2SourceTrack[]): void {
  if (!datasetId) return;
  _write(YUE2_SOURCE_TRACKS_PREFIX + datasetId, tracks);
}

function readYue2DatasetForLyricsSet(lyricsSetId: number): string {
  return _read<string>(YUE2_DATASET_FOR_LYRICS_SET_PREFIX + lyricsSetId) ?? '';
}

function writeYue2DatasetForLyricsSet(lyricsSetId: number, datasetId: string): void {
  _write(YUE2_DATASET_FOR_LYRICS_SET_PREFIX + lyricsSetId, datasetId);
}

export function readYue2CaptionDataset(): string {
  return _read<string>(YUE2_CAPTION_DATASET_KEY) ?? '';
}

export function readYue2CaptionSources(): Yue2CaptionSourcesHandoff | null {
  const stored = _read<Yue2CaptionSourcesHandoff>(YUE2_CAPTION_SOURCES_KEY);
  if (!stored || !Array.isArray(stored.tracks)) return null;
  return stored;
}

export function clearYue2CaptionSources(): void {
  try { localStorage.removeItem(YUE2_CAPTION_SOURCES_KEY); } catch { /* ignore */ }
}

/**
 * The dataset's captioned tracks, from the server.
 *
 * `GET /api/training/yue2-dataset-captions` accepts one of `dataset` (id),
 * `lyricsSet` (a lyrics-set id, resolved server-side through
 * `training_datasets.lyrics_set_id`) or `adapter` (an absolute adapter path,
 * resolved through the run that trained it — kept for the one caller that
 * only has a path: the Create panel's "whatever the engine is holding").
 * Answers `{ datasetId, datasetSlug, datasetName, tracks }`; unlinked or
 * unknown answers all-empty with `tracks: []`, which reads here as "no
 * picker".
 */
export async function fetchYue2CaptionSource(
  by: { dataset?: string; lyricsSet?: number; adapter?: string },
): Promise<{ datasetId: string; datasetName: string; tracks: Yue2SourceTrack[] }> {
  const params = new URLSearchParams();
  if (by.dataset) params.set('dataset', by.dataset);
  else if (by.lyricsSet) params.set('lyricsSet', String(by.lyricsSet));
  else if (by.adapter) params.set('adapter', by.adapter);
  else return { datasetId: '', datasetName: '', tracks: [] };

  try {
    const res = await fetch(`/api/training/yue2-dataset-captions?${params.toString()}`);
    if (!res.ok) return { datasetId: '', datasetName: '', tracks: [] };
    const data = await res.json() as {
      datasetId?: string; datasetName?: string; tracks?: Yue2SourceTrack[];
    };
    const datasetId = data.datasetId || '';
    const datasetName = data.datasetName || '';
    const tracks = (data.tracks ?? []).filter(t => t && t.name && (t.styled || t.caption));
    if (datasetId) cacheYue2SourceTracks(datasetId, tracks);
    if (by.lyricsSet) writeYue2DatasetForLyricsSet(by.lyricsSet, datasetId);
    return { datasetId, datasetName, tracks };
  } catch {
    return { datasetId: '', datasetName: '', tracks: [] };
  }
}

/** Cache first, server second — except for `by.adapter`, which always asks the
 *  server: it is one cheap lookup and there is no local cache keyed by adapter
 *  path any more to check.
 *
 *  The cache is what lets the non-React render paths resolve a pick
 *  synchronously — and since Automatic is the default, every render path on
 *  this backend has to fill it before resolving, whether or not the user ever
 *  opened the picker. Both do (audioGenQueueStore, useAudioGeneration). */
export async function ensureYue2CaptionSource(
  by: { dataset?: string; lyricsSet?: number; adapter?: string },
): Promise<{ datasetId: string; datasetName: string; tracks: Yue2SourceTrack[] }> {
  if (by.adapter && !by.dataset && !by.lyricsSet) return fetchYue2CaptionSource(by);

  const datasetId = by.dataset || (by.lyricsSet ? readYue2DatasetForLyricsSet(by.lyricsSet) : '');
  if (datasetId) {
    const cached = readYue2SourceTracks(datasetId);
    if (cached.length) return { datasetId, datasetName: '', tracks: cached };
  }
  return fetchYue2CaptionSource(by);
}

/** Which adapter the engine is currently holding, by half — AR preferred, NAR
 *  the fallback for a stack that has only that half. Used solely to resolve a
 *  dataset id for the Create panel when nothing else has chosen one yet (via
 *  `ensureYue2CaptionSource({ adapter })`); the caption list itself is never
 *  looked up by this path once a dataset is known.
 *
 *  `lmAdapter` is read last and only for compatibility: the picker carried one
 *  slot until the AR/NAR split (58aff871), and a client reading the old key
 *  against a split catalogue gets ''. */
export function yue2CaptionAdapterPath(defaults: Record<string, unknown> | undefined): string {
  const pick = (k: string): string => {
    const v = defaults?.[k];
    return typeof v === 'string' ? v.trim() : '';
  };
  return pick('lmAdapterAr') || pick('lmAdapterNar') || pick('lmAdapter');
}

/** The album's own adapter, from its preset — the one the render WILL use, not
 *  the one the engine happens to hold right now. Still adapter-keyed, because
 *  the adapter selection is genuinely engine state (see applyYue2PresetAdapters
 *  below) — this only decides WHICH halves to merge, not which dataset's
 *  captions to offer; the caption side is resolved separately, by
 *  lyrics-set id, once the merge is in force. */
export function yue2PresetAdapterPath(
  preset: { yue2_ar_adapter_path?: string | null; yue2_nar_adapter_path?: string | null } | null | undefined,
): string {
  const ar = String(preset?.yue2_ar_adapter_path ?? '').trim();
  const nar = String(preset?.yue2_nar_adapter_path ?? '').trim();
  return ar || nar;
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
 *  `lyricsSetId` is the album's, when there is one: the lyrics-set → dataset
 *  map resolves it to a dataset id (filled by a prior
 *  `fetchYue2CaptionSource({ lyricsSet })` call, which every render path makes
 *  before this runs). With no lyrics-set id — Create, Cover Studio — the
 *  Create panel's own chosen dataset (`hs-yue2CaptionDataset`) applies
 *  instead, so a song sent from Create still resolves the same way the panel
 *  showed it. No datasetId either way lands on the song's own caption, so
 *  callers can use this unconditionally on this backend. */
export function resolveYue2CaptionForGeneration(
  gen: { id?: number; bpm?: number; caption?: string | null },
  lyricsSetId?: number,
): Yue2ResolvedCaption {
  const datasetId = lyricsSetId ? readYue2DatasetForLyricsSet(lyricsSetId) : readYue2CaptionDataset();
  const own = gen.caption || '';
  if (!datasetId) return { caption: own.trim(), mode: 'custom' };
  return resolveYue2Caption(
    own, gen.bpm, readYue2SourceTracks(datasetId), readYue2SongSelection(datasetId, gen.id));
}

// ── Per-song choice (Lyric Studio) ───────────────────────────────────────────
//
// The selection above is keyed by DATASET, which is right for Create: one
// caption box, one dataset in force, no song to hang the choice on. A written
// song is different — the card shows one song and MM3 lets each song pick its
// own source track, so YuE2 does too.
//
// Keyed by BOTH dataset and song. The caption list belongs to one dataset, so
// a choice made under one album cannot mean anything under another's; storing
// it per song alone would silently apply a Crash Test Dummies track title to
// a Green Day render. Falls back to the per-dataset choice when this song has
// never been touched, so a preference set in Create still leads.

function yue2SongKey(datasetId: string, genId: number): string {
  return `${YUE2_CAPTION_SOURCE_PREFIX}song:${genId}:${datasetId}`;
}

/** True when THIS song has its own stored choice, as opposed to falling back
 *  to the dataset-level default. Lets a caller apply a different default (e.g.
 *  Automatic only when the album has an adapter) without that default being
 *  mistaken for a real stored 'custom' pick on the next render. */
export function hasStoredYue2SongSelection(datasetId: string, genId?: number): boolean {
  if (!datasetId || typeof genId !== 'number') return false;
  const stored = _read<Yue2CaptionSelection>(yue2SongKey(datasetId, genId));
  return !!stored && (stored.mode === 'auto' || stored.mode === 'track' || stored.mode === 'custom');
}

export function readYue2SongSelection(datasetId: string, genId?: number): Yue2CaptionSelection {
  if (!datasetId) return { mode: 'custom' };
  if (typeof genId === 'number') {
    const stored = _read<Yue2CaptionSelection>(yue2SongKey(datasetId, genId));
    if (stored && (stored.mode === 'auto' || stored.mode === 'track' || stored.mode === 'custom')) {
      return stored;
    }
  }
  return readYue2CaptionSelection(datasetId);
}

export function writeYue2SongSelection(
  datasetId: string, genId: number, sel: Yue2CaptionSelection,
): void {
  if (!datasetId || typeof genId !== 'number') return;
  _write(yue2SongKey(datasetId, genId), sel);
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
