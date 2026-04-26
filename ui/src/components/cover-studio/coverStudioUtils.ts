// coverStudioUtils.ts — Shared helpers for Cover Studio

// ── Persistence ─────────────────────────────────────────────────────────
export const STORAGE_PREFIX = 'cover-studio-';
export const TRACK_CACHE_KEY = 'cover-studio-trackCache';

export function persist(key: string, value: any) {
  try { localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value)); } catch {}
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
  try { return JSON.parse(localStorage.getItem(TRACK_CACHE_KEY) || '{}'); } catch { return {}; }
}

export function saveTrackCacheEntry(filename: string, entry: Partial<TrackCacheEntry>) {
  try {
    const cache = getTrackCache();
    cache[filename] = { ...(cache[filename] || {}), ...entry } as TrackCacheEntry;
    localStorage.setItem(TRACK_CACHE_KEY, JSON.stringify(cache));
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
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
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
