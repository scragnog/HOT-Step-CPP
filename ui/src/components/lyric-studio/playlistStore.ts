/**
 * playlistStore.ts — localStorage-backed play queue for Lyric Studio.
 *
 * Stores a list of PlaylistItems under `lireek-playQueue`.
 * Provides a React hook `usePlaylist()` with automatic reactivity via
 * a custom event (`lireek-playlist-change`) + window storage events.
 */

import { useCallback, useSyncExternalStore } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

export interface PlaylistItem {
  id: string;
  title: string;
  audioUrl: string;
  masteredAudioUrl?: string;
  noAdapterAudioUrl?: string;
  artistName?: string;
  coverUrl?: string;
  duration?: number; // seconds
  style?: string;
  /** Preserved so M/O toggle works when playing from playlist */
  generationParams?: any;
}

// ── Storage ──────────────────────────────────────────────────────────────────

const STORAGE_KEY = 'lireek-playQueue';
const CHANGE_EVENT = 'lireek-playlist-change';

let _snapshot: PlaylistItem[] | null = null;
let _persistTimer: ReturnType<typeof setTimeout> | null = null;

function read(): PlaylistItem[] {
  if (_snapshot) return _snapshot;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    _snapshot = raw ? JSON.parse(raw) : [];
  } catch {
    _snapshot = [];
  }
  return _snapshot!;
}

/** Debounced persistence — for high-frequency ops (drag reorder). */
function _persistPlaylist(): void {
  if (_persistTimer) clearTimeout(_persistTimer);
  _persistTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(_snapshot || []));
    } catch (e) {
      console.error('[Playlist] localStorage write failed (quota?):', e);
    }
  }, 500);
}

/** Force-flush persistence immediately (for clear, reorder — infrequent ops). */
function _persistPlaylistNow(): void {
  if (_persistTimer) { clearTimeout(_persistTimer); _persistTimer = null; }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(_snapshot || []));
  } catch (e) {
    console.error('[Playlist] localStorage write failed (quota?):', e);
  }
}

function write(items: PlaylistItem[], immediate = false): void {
  _snapshot = items;
  if (immediate) _persistPlaylistNow(); else _persistPlaylist();
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

// ── Public API ───────────────────────────────────────────────────────────────

export function getPlaylist(): PlaylistItem[] { return read(); }

export function addToPlaylist(item: PlaylistItem): void {
  const list = read();
  if (list.some(i => i.id === item.id)) return;
  write([...list, item], true);  // persist immediately — playlist changes must not be lost
}

export function removeFromPlaylist(id: string): void {
  write(read().filter(i => i.id !== id));
}

export function clearPlaylist(): void { write([], true); }

export function isInPlaylist(id: string): boolean {
  return read().some(i => i.id === id);
}

/**
 * Patch fields on an item already in the playlist.
 *
 * addToPlaylist() early-returns on a duplicate id, so before this there was no
 * way at all to refresh an entry: a playlist item was whatever it happened to
 * be when it was added, forever.
 */
export function updatePlaylistItem(id: string, patch: Partial<PlaylistItem>): void {
  const list = read();
  if (!list.some(i => i.id === id)) return;
  write(list.map(i => (i.id === id ? { ...i, ...patch } : i)), true);
}

export function reorderPlaylist(items: PlaylistItem[]): void { write(items, true); }

export function moveItem(id: string, direction: 'up' | 'down'): void {
  const list = [...read()];
  const idx = list.findIndex(i => i.id === id);
  if (idx < 0) return;
  const target = direction === 'up' ? idx - 1 : idx + 1;
  if (target < 0 || target >= list.length) return;
  [list[idx], list[target]] = [list[target], list[idx]];
  write(list);
}

// ── React Hook ───────────────────────────────────────────────────────────────

function subscribe(cb: () => void): () => void {
  const onCustom = () => cb();
  const onStorage = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY) { _snapshot = null; cb(); }
  };
  window.addEventListener(CHANGE_EVENT, onCustom);
  window.addEventListener('storage', onStorage);
  return () => {
    window.removeEventListener(CHANGE_EVENT, onCustom);
    window.removeEventListener('storage', onStorage);
  };
}

function getSnapshot(): PlaylistItem[] { return read(); }

export function usePlaylist() {
  const items = useSyncExternalStore(subscribe, getSnapshot);

  const add = useCallback((item: PlaylistItem) => addToPlaylist(item), []);
  const remove = useCallback((id: string) => removeFromPlaylist(id), []);
  const clear = useCallback(() => clearPlaylist(), []);
  const isIn = useCallback((id: string) => items.some(i => i.id === id), [items]);
  const move = useCallback((id: string, dir: 'up' | 'down') => moveItem(id, dir), []);
  const reorder = useCallback((newItems: PlaylistItem[]) => reorderPlaylist(newItems), []);

  return { items, add, remove, clear, isIn, move, reorder };
}

// ── Staying in step with post-processing ─────────────────────────────────────
//
// Playlist entries are snapshots in localStorage, and nothing used to update
// them. So removing a track's post-processed version left the playlist still
// advertising a mastered file that had just been deleted: the row kept
// offering "Remove Post-Processed Version" (making a revert that had actually
// succeeded look like it did nothing) and playback still had a dead URL to
// reach for.
//
// Registered at module scope so it applies whether or not the playlist sidebar
// is open.
window.addEventListener('song-postprocessed', (e: Event) => {
  const { songId, masteredAudioUrl } = (e as CustomEvent).detail || {};
  if (songId && masteredAudioUrl) updatePlaylistItem(songId, { masteredAudioUrl });
});

window.addEventListener('song-postprocess-reverted', (e: Event) => {
  const { songId } = (e as CustomEvent).detail || {};
  if (songId) updatePlaylistItem(songId, { masteredAudioUrl: '' });
});
