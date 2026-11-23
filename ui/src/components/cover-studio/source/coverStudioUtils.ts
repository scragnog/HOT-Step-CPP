import { readJson } from '@/utils/storage/storageJson';
import { writeJson } from '@/utils/storage/storageJson';
// coverStudioUtils.ts — Shared helpers for Cover Studio

// ── Persistence ─────────────────────────────────────────────────────────
export const STORAGE_PREFIX = 'cover-studio-';
export const TRACK_CACHE_KEY = 'cover-studio-trackCache';

export function persist(key: string, value: any) {
  try { writeJson(STORAGE_PREFIX + key, value); } catch {}
}

export function restore<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + key);
    return raw !== null ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

// ── Track cache ─────────────────────────────────────────────────────────
export interface TrackCacheEntry {
  artist: string;
  title: string;
  lyrics: string;
  bpm: number;
  key: string;
  scale?: string;
  duration: number | null;
  album?: string;
}

export function getTrackCache(): Record<string, TrackCacheEntry> {
  return readJson<Record<string, TrackCacheEntry>>(TRACK_CACHE_KEY, {});
}

export function saveTrackCacheEntry(filename: string, entry: Partial<TrackCacheEntry>) {
  try {
    const cache = getTrackCache();
    cache[filename] = { ...(cache[filename] || {}), ...entry } as TrackCacheEntry;
    writeJson(TRACK_CACHE_KEY, cache);
  } catch {}
}

// ── Types ───────────────────────────────────────────────────────────────
export interface AudioMetadata {
  artist: string;
  title: string;
  album: string;
  duration: number | null;
}

export interface AudioAnalysis {
  bpm: number;
  key: string;
  scale?: string;
}

// ── Music theory helpers ────────────────────────────────────────────────
export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/** All 24 standard musical keys (12 major + 12 minor). */
export const ALL_KEYS = [
  ...NOTE_NAMES.map(n => `${n} major`),
  ...NOTE_NAMES.map(n => `${n} minor`),
];
const NOTE_ALIASES: Record<string, number> = {
  'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
  'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
};

export function transposeKey(keyStr: string, semitones: number): string {
  if (!keyStr || semitones === 0) return keyStr;
  const parts = keyStr.trim().split(/\s+/);
  const noteIndex = NOTE_ALIASES[parts[0]];
  if (noteIndex === undefined) return keyStr;
  const newIndex = ((noteIndex + semitones) % 12 + 12) % 12;
  const quality = parts.slice(1).join(' ');
  return quality ? `${NOTE_NAMES[newIndex]} ${quality}` : NOTE_NAMES[newIndex];
}

// ── Cover source and render state ───────────────────────────────────────
/** An uploaded source whose draft is still being opened. A library send has neither an asset nor a
 *  draft and is not pending: its draft is registered when the render starts. */
export const sourceDraftPending = (p: { sourceMode: string; sourceAssetId: string; coverDraftId: string; coverDraftRevision: number }) =>
  p.sourceMode === 'audio' && !!p.sourceAssetId && (!p.coverDraftId || !p.coverDraftRevision);

/** Whether the render that started `started` (its workflow job id, '' when the start itself failed)
 *  still owns the busy state: a newer render's job id in `current` means it does not. */
export const renderOwnsBusy = (started: string, current: string) => started ? current === started : current === '';

// ── YuE2 cover content ──────────────────────────────────────────────────
/** What a YuE2 cover still needs before it can render, or null when ready.
 *  The style caption is required: with it empty, an instrumental sends YuE2
 *  only "Instrumental, no vocals, ..." and the model invents a voice anyway
 *  (gate renders 6e90dbed, 6f78f660). Covers on other backends keep their
 *  own rules. */
export function yue2CoverContentMissing(p: { caption: string; lyrics: string; instrumental: boolean }): string | null {
  if (!p.caption.trim()) return 'Write a YuE2 style caption';
  if (!p.instrumental && !p.lyrics.trim()) return 'Enter lyrics or enable Instrumental mode';
  return null;
}
