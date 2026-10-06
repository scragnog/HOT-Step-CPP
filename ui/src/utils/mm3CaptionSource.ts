/**
 * mm3CaptionSource.ts — where a MiniMax-Music3 render's caption comes from.
 *
 * Rendering a NEW song's lyrics under one of the artist's OWN training tracks'
 * Structured Captions — word for word, not a fresh caption in the same style —
 * is the most reliable way to get a song that sounds like the band and, in
 * particular, the most reliable way to get a natural ending. So the MM3 caption
 * used for a written song is a CHOICE, not simply the caption the LLM wrote for
 * it:
 *
 *   auto    (default) the source track whose tempo is nearest this song's,
 *           among the album's tracks that have an MM3 caption. Ties go to the
 *           first in album order.
 *   track   a specific source track the user picked, by title.
 *   custom  the generation's own `caption_mm3`, exactly as before this existed.
 *
 * The choice lives in localStorage — there is no column for it — keyed by
 * generation id. The album's captioned tracks are cached in memory, keyed by
 * lyrics-set id, so the two non-React render paths (the Lyric Studio audio
 * queue and Send-to-Create) can resolve a caption synchronously.
 *
 * Nothing here touches ACE-Step: every entry point is gated on the MM3 backend
 * being the active one.
 */

export type Mm3CaptionMode = 'auto' | 'track' | 'custom';

/** One album source track that carries an MM3 Structured Caption. */
export interface Mm3SourceTrack {
  title: string;
  bpm?: number;
  caption: string;
}

/** The per-generation choice. Absent from storage means `{ mode: 'auto' }`. */
export interface Mm3CaptionSelection {
  mode: Mm3CaptionMode;
  /** Only meaningful for mode 'track'. */
  selectedTitle?: string;
}

/** What Send-to-Create hands the Create panel so it can offer the same control
 *  with no server call of its own. */
export interface Mm3CaptionSourcesHandoff extends Mm3CaptionSelection {
  /** The generation's own caption_mm3 — what "Custom" starts from. */
  customCaption: string;
  /** Album tracks that have an MM3 caption, in album order. */
  tracks: Mm3SourceTrack[];
}

/** `hs-mm3CaptionSource:<generation id>` → Mm3CaptionSelection */
export const MM3_CAPTION_SOURCE_PREFIX = 'hs-mm3CaptionSource:';
/** `hs-mm3SourceTracks:<lyrics set id>` → Mm3SourceTrack[] */
export const MM3_SOURCE_TRACKS_PREFIX = 'hs-mm3SourceTracks:';
/** `hs-mm3CaptionSources` → Mm3CaptionSourcesHandoff (Send-to-Create handoff) */
export const MM3_CAPTION_SOURCES_KEY = 'hs-mm3CaptionSources';

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

// The album's captioned tracks are a copy of server data, so they live in
// memory: every render path calls ensureMm3SourceTracks first, which refetches
// on a miss. They used to sit in localStorage, one entry per album ever opened,
// and filled the quota. Clear what earlier builds left there.
const sourceTracks = new Map<number, Mm3SourceTrack[]>();
try {
  for (const key of Object.keys(localStorage)) {
    if (key.startsWith(MM3_SOURCE_TRACKS_PREFIX)) localStorage.removeItem(key);
  }
} catch { /* storage unavailable */ }

export function readMm3CaptionSelection(generationId: number): Mm3CaptionSelection {
  const stored = _read<Mm3CaptionSelection>(MM3_CAPTION_SOURCE_PREFIX + generationId);
  if (!stored || (stored.mode !== 'auto' && stored.mode !== 'track' && stored.mode !== 'custom')) {
    return { mode: 'auto' };
  }
  return stored;
}

export function writeMm3CaptionSelection(generationId: number, sel: Mm3CaptionSelection): void {
  _write(MM3_CAPTION_SOURCE_PREFIX + generationId, sel);
}

export function readMm3SourceTracks(lyricsSetId: number | undefined): Mm3SourceTrack[] {
  if (!lyricsSetId) return [];
  return sourceTracks.get(lyricsSetId) ?? [];
}

/** The render paths are plain modules that may run before the album's detail
 *  was ever opened in this browser (a queue resumed after a reload, a song
 *  card rendered straight from the list). Fetch and cache the album's
 *  captioned tracks then, so "Automatic" never silently degrades to the
 *  song's own caption just because the cache was cold. */
export async function ensureMm3SourceTracks(lyricsSetId: number | undefined): Promise<Mm3SourceTrack[]> {
  if (!lyricsSetId) return [];
  const cached = readMm3SourceTracks(lyricsSetId);
  if (cached.length) return cached;
  try {
    const { lireekApi } = await import('../services/lireekApi');
    const set = await lireekApi.getLyricsSet(lyricsSetId);
    const songs = typeof set.songs === 'string' ? JSON.parse(set.songs) : (set.songs || []);
    const tracks = collectMm3SourceTracks(songs);
    cacheMm3SourceTracks(lyricsSetId, tracks);
    return tracks;
  } catch {
    return [];
  }
}

/** Cache the album's captioned tracks so the render paths can resolve without
 *  an API call. Called whenever an album's detail data is loaded. */
export function cacheMm3SourceTracks(lyricsSetId: number, tracks: Mm3SourceTrack[]): void {
  sourceTracks.set(lyricsSetId, tracks);
}

/** Keep only the source songs that actually carry an MM3 caption, in album
 *  order — the order is the documented tie-break for "nearest tempo". */
export function collectMm3SourceTracks(
  songs: Array<{ title?: string; bpm?: number; mm3Caption?: string }>,
): Mm3SourceTrack[] {
  const out: Mm3SourceTrack[] = [];
  for (const s of songs || []) {
    const caption = (s.mm3Caption || '').trim();
    if (!caption) continue;
    out.push({ title: s.title || 'Untitled', bpm: s.bpm, caption });
  }
  return out;
}

export function readMm3CaptionSources(): Mm3CaptionSourcesHandoff | null {
  const stored = _read<Mm3CaptionSourcesHandoff>(MM3_CAPTION_SOURCES_KEY);
  if (!stored || !Array.isArray(stored.tracks)) return null;
  return stored;
}

export function clearMm3CaptionSources(): void {
  try { localStorage.removeItem(MM3_CAPTION_SOURCES_KEY); } catch { /* ignore */ }
}

// ── Resolution ───────────────────────────────────────────────────────────────

/** The source track whose tempo is nearest `bpm`.
 *
 *  Ties go to the earlier track, which is why this walks the array in order and
 *  only takes a STRICTLY smaller distance. A song with no tempo, or an album
 *  whose captioned tracks have none, falls back to the first track — there is
 *  nothing to be nearest to. */
export function pickNearestBpmTrack(
  tracks: Mm3SourceTrack[],
  bpm: number | undefined,
): Mm3SourceTrack | null {
  if (tracks.length === 0) return null;
  if (!bpm || bpm <= 0) return tracks[0];

  let best: Mm3SourceTrack | null = null;
  let bestDist = Infinity;
  for (const track of tracks) {
    if (!track.bpm || track.bpm <= 0) continue;
    const dist = Math.abs(track.bpm - bpm);
    if (dist < bestDist) { bestDist = dist; best = track; }
  }
  return best ?? tracks[0];
}

export interface Mm3ResolvedCaption {
  /** The caption to render with. May be empty when nothing is available. */
  caption: string;
  /** The mode that actually applied — an unavailable choice degrades to one
   *  that works, so this is not always the stored mode. */
  mode: Mm3CaptionMode;
  /** Source track the caption came from, for the "From dataset track:" line. */
  fromTitle?: string;
}

/**
 * Resolve the caption for one written song.
 *
 * Degradation is deliberate and always downward to something renderable: a
 * picked track that is no longer in the album, or an album with no captioned
 * tracks at all, ends up on the song's own caption rather than on nothing.
 */
export function resolveMm3Caption(
  gen: { bpm?: number; caption_mm3?: string | null },
  tracks: Mm3SourceTrack[],
  sel: Mm3CaptionSelection,
): Mm3ResolvedCaption {
  const own = (gen.caption_mm3 || '').trim();

  // Custom means the song's OWN MM3 caption. A song written before that field
  // existed has none, and the ACE caption is not a substitute on this backend
  // (measured off-genre), so an empty Custom falls through to the dataset.
  if (sel.mode === 'custom' && own) return { caption: own, mode: 'custom' };

  if (sel.mode === 'track' && sel.selectedTitle) {
    const hit = tracks.find(track => track.title === sel.selectedTitle);
    if (hit) return { caption: hit.caption, mode: 'track', fromTitle: hit.title };
    // The album changed under the choice — fall through to auto.
  }

  const nearest = pickNearestBpmTrack(tracks, gen.bpm);
  if (nearest) return { caption: nearest.caption, mode: 'auto', fromTitle: nearest.title };
  return { caption: own, mode: 'custom' };
}

/** Resolve straight from storage — the form the non-React render paths use. */
export function resolveMm3CaptionForGeneration(
  gen: { id?: number; bpm?: number; caption_mm3?: string | null },
  lyricsSetId: number | undefined,
): Mm3ResolvedCaption {
  const sel = gen.id ? readMm3CaptionSelection(gen.id) : { mode: 'auto' as const };
  return resolveMm3Caption(gen, readMm3SourceTracks(lyricsSetId), sel);
}
