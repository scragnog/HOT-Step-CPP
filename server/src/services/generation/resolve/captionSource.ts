// resolve/captionSource.ts — which caption a render uses, per engine.
//
// Server-side port of ui/src/utils/captionForBackend.ts, mm3CaptionSource.ts
// and yue2CaptionSource.ts (resolution functions only). Behaviour must match
// those files exactly, including every fallback, so the same inputs give the
// same caption whichever side resolves it. The browser's stored selections are
// NOT read here: callers pass the selection explicitly.

import type {
  CaptionSourceMode, Mm3CaptionSelection, Mm3SourceTrack, Yue2CaptionSelection, Yue2SourceTrack,
} from '../../../contracts/resolution.js';

export const MM3_ENGINE_ID = 'minimax-m3';
export const YUE2_ENGINE_ID = 'yue2';

// ── MM3 ──────────────────────────────────────────────────────────────────────

/** Album songs that carry an MM3 caption, in album order (the tie-break order). */
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

/** Nearest tempo; ties to the earlier track; no tempo anywhere → the first. */
export function pickNearestMm3Track(tracks: Mm3SourceTrack[], bpm: number | undefined): Mm3SourceTrack | null {
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

export interface ResolvedCaption {
  caption: string;
  mode: CaptionSourceMode;
  fromTrack?: string;
}

export function resolveMm3Caption(
  gen: { bpm?: number; caption_mm3?: string | null },
  tracks: Mm3SourceTrack[],
  sel: Mm3CaptionSelection,
): ResolvedCaption {
  const own = (gen.caption_mm3 || '').trim();
  if (sel.mode === 'custom' && own) return { caption: own, mode: 'custom' };
  if (sel.mode === 'track' && sel.selectedTitle) {
    const hit = tracks.find(track => track.title === sel.selectedTitle);
    if (hit) return { caption: hit.caption, mode: 'track', fromTrack: hit.title };
  }
  const nearest = pickNearestMm3Track(tracks, gen.bpm);
  if (nearest) return { caption: nearest.caption, mode: 'auto', fromTrack: nearest.title };
  return { caption: own, mode: 'custom' };
}

/** A stored MM3 choice as the browser reads it: anything unrecognised is auto. */
export function normalizeMm3Selection(sel: Mm3CaptionSelection | null | undefined): Mm3CaptionSelection {
  return sel && (sel.mode === 'auto' || sel.mode === 'track' || sel.mode === 'custom') ? sel : { mode: 'auto' };
}

// ── YuE2 ─────────────────────────────────────────────────────────────────────

export function yue2TrackCaption(track: Yue2SourceTrack): string {
  return (track.styled || track.caption || '').trim();
}

export function yue2TrackBpm(track: Yue2SourceTrack): number | undefined {
  const n = typeof track.bpm === 'number' ? track.bpm : parseFloat(String(track.bpm ?? ''));
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

export function pickNearestYue2Track(tracks: Yue2SourceTrack[], bpm: number | undefined): Yue2SourceTrack | null {
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

export function resolveYue2Caption(
  own: string,
  bpm: number | undefined,
  tracks: Yue2SourceTrack[],
  sel: Yue2CaptionSelection,
): ResolvedCaption {
  const mine = (own || '').trim();
  if (sel.mode === 'custom') return { caption: mine, mode: 'custom' };
  if (sel.mode === 'track' && sel.selectedName) {
    const hit = tracks.find(track => track.name === sel.selectedName);
    const caption = hit ? yue2TrackCaption(hit) : '';
    if (caption) return { caption, mode: 'track', fromTrack: hit!.name };
  }
  const nearest = pickNearestYue2Track(tracks, bpm);
  const caption = nearest ? yue2TrackCaption(nearest) : '';
  if (caption) return { caption, mode: 'auto', fromTrack: nearest!.name };
  return { caption: mine, mode: 'custom' };
}

/** The choice when none is given: auto only with a dataset AND an adapter in
 *  force (yue2CaptionSource.ts defaultYue2CaptionSelection). */
export function defaultYue2Selection(datasetId: string, adapterInForce: boolean): Yue2CaptionSelection {
  return { mode: datasetId && adapterInForce ? 'auto' : 'custom' };
}

/** A given YuE2 choice, or the default when absent or unrecognised. No
 *  dataset means custom, whatever was asked for. */
export function effectiveYue2Selection(
  datasetId: string, sel: Yue2CaptionSelection | null | undefined, adapterInForce: boolean,
): Yue2CaptionSelection {
  if (!datasetId) return { mode: 'custom' };
  if (sel && (sel.mode === 'auto' || sel.mode === 'track' || sel.mode === 'custom')) return sel;
  return defaultYue2Selection(datasetId, adapterInForce);
}

/** Which adapter path the pick holds: AR, then NAR, then the legacy key. */
export function yue2CaptionAdapterPath(defaults: Record<string, unknown> | undefined): string {
  const pick = (k: string): string => {
    const v = defaults?.[k];
    return typeof v === 'string' ? v.trim() : '';
  };
  return pick('lmAdapterAr') || pick('lmAdapterNar') || pick('lmAdapter');
}

/** The pick a queued song renders with: the picker's paths and dials, with the
 *  album preset's two halves in place of the paths when there is a preset. */
export function yue2PickAtEnqueue(
  defaults: Record<string, unknown> | undefined,
  preset: { yue2_ar_adapter_path?: string | null; yue2_nar_adapter_path?: string | null } | null | undefined,
): Record<string, string | number> {
  const pick: Record<string, string | number> = {};
  for (const [k, v] of Object.entries(defaults ?? {})) {
    if (/^lmAdapter(Ar|Nar)(Scale(Attn|Mlp|Early|Mid|Late)?)?$/.test(k) && (typeof v === 'string' || typeof v === 'number')) {
      pick[k] = v;
    }
  }
  if (preset) {
    pick.lmAdapterAr = String(preset.yue2_ar_adapter_path ?? '').trim();
    pick.lmAdapterNar = String(preset.yue2_nar_adapter_path ?? '').trim();
  }
  return pick;
}

// ── Precedence across engines (captionForBackend.ts) ─────────────────────────

export interface SongCaptions {
  id?: number;
  bpm?: number;
  caption?: string | null;
  caption_mm3?: string | null;
  caption_yue2?: string | null;
}

export interface CaptionContext {
  mm3?: { tracks: Mm3SourceTrack[]; selection: Mm3CaptionSelection };
  /** `datasetId` '' means no dataset is linked. */
  yue2?: { datasetId: string; tracks: Yue2SourceTrack[]; selection: Yue2CaptionSelection };
  renderingAs?: boolean;
}

export interface CaptionResolution extends Partial<ResolvedCaption> {
  caption: string;
  /** Which branch of the precedence produced the caption. */
  source: 'album-track' | 'song-mm3' | 'dataset-track' | 'song-yue2' | 'song';
}

/** captionForBackend, branch for branch. On MM3 the song's MM3 caption or an
 *  album track; on YuE2 a Render-as dataset pick, then the song's own YuE2
 *  caption, then a dataset pick; otherwise (and on every empty result) the
 *  ACE caption. */
export function captionForEngine(gen: SongCaptions, engine: string, ctx: CaptionContext): CaptionResolution {
  if (engine === MM3_ENGINE_ID) {
    const r = resolveMm3Caption(gen, ctx.mm3?.tracks ?? [], ctx.mm3?.selection ?? { mode: 'auto' });
    if (r.caption.trim()) return { ...r, source: r.mode === 'custom' ? 'song-mm3' : 'album-track' };
  }
  const yue2 = ctx.yue2;
  const resolveYue2 = (): ResolvedCaption => (yue2 && yue2.datasetId
    ? resolveYue2Caption(gen.caption || '', gen.bpm, yue2.tracks, yue2.selection)
    : { caption: (gen.caption || '').trim(), mode: 'custom' });
  if (engine === YUE2_ENGINE_ID && ctx.renderingAs) {
    const r = resolveYue2();
    if (r.mode !== 'custom' && r.caption.trim()) return { ...r, source: 'dataset-track' };
  }
  if (engine === YUE2_ENGINE_ID) {
    if ((gen.caption_yue2 || '').trim()) return { caption: (gen.caption_yue2 || '').trim(), source: 'song-yue2' };
    const r = resolveYue2();
    if (r.caption.trim()) return { ...r, source: r.mode === 'custom' ? 'song' : 'dataset-track' };
  }
  return { caption: gen.caption || '', source: 'song' };
}
