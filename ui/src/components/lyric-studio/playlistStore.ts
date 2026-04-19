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

function write(items: PlaylistItem[]): void {
  _snapshot = items;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

// ── Public API ───────────────────────────────────────────────────────────────

export function getPlaylist(): PlaylistItem[] { return read(); }

export function addToPlaylist(item: PlaylistItem): void {
  const list = read();
  if (list.some(i => i.id === item.id)) return;
  write([...list, item]);
}

export function removeFromPlaylist(id: string): void {
  write(read().filter(i => i.id !== id));
}

export function clearPlaylist(): void { write([]); }

export function isInPlaylist(id: string): boolean {
  return read().some(i => i.id === id);
}

export function reorderPlaylist(items: PlaylistItem[]): void { write(items); }

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
