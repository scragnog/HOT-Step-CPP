/**
 * playlistStore.ts — ordered Lyric Studio playlist with explicit browser import.
 *
 * Stores a list of PlaylistItems under `lireek-playQueue`.
 * Provides a React hook `usePlaylist()` with automatic reactivity via
 * a custom event (`lireek-playlist-change`) + window storage events.
 */

import { useCallback, useEffect, useSyncExternalStore } from 'react';
import { useAuth } from '../../context/AuthContext';
import { studioDraftsApi, type PlaylistDocument } from '../../services/studioDraftsApi';

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
const SERVER_POINTER_KEY = 'lireek-playQueue-server-document';

let _snapshot: PlaylistItem[] | null = null;
let _persistTimer: ReturnType<typeof setTimeout> | null = null;
let serverToken: string | null = null;
let serverDocument: PlaylistDocument | null = null;
let connected = false;
let initialized = false;
let connecting: Promise<void> | null = null;
let pending: unknown[] = [];
let draining = false;
let status = { needsImport: false, replacingServer: false, saveError: '', saving: false };

function announce(): void { window.dispatchEvent(new CustomEvent(CHANGE_EVENT)); }
function setStatus(next: typeof status): void { status = next; announce(); }

async function drain(): Promise<void> {
  if (draining || !connected || !serverToken || status.saveError) return;
  draining = true;
  setStatus({ ...status, saving: true });
  try {
    while (pending.length) {
      const result = await studioDraftsApi.playlistCommand(serverToken, serverDocument?.revision ?? 0, pending[0]);
      serverDocument = result.document;
      pending.shift();
      if (!pending.length) { _snapshot = result.document.body.items; announce(); }
    }
  } catch (error) {
    setStatus({ ...status, saveError: error instanceof Error ? error.message : String(error), saving: false });
  } finally {
    draining = false;
    if (!status.saveError) setStatus({ ...status, saving: false });
  }
}

async function connect(token: string): Promise<void> {
  if (serverToken === token && (initialized || connecting)) return connecting ?? Promise.resolve();
  serverToken = token;
  connected = false;
  connecting = (async () => {
    try {
      if (_persistTimer) _persistPlaylistNow();
      const result = await studioDraftsApi.playlist(token);
      serverDocument = result.document;
      const legacy = localStorage.getItem(STORAGE_KEY);
      const pointer = localStorage.getItem(SERVER_POINTER_KEY);
      if (legacy !== null && (!pointer || !result.document)) {
        initialized = true;
        setStatus({ needsImport: true, replacingServer: !!result.document, saveError: '', saving: false });
        return;
      }
      connected = true;
      initialized = true;
      _snapshot = result.document?.body.items ?? [];
      setStatus({ needsImport: false, replacingServer: false, saveError: '', saving: false });
    } catch (error) {
      initialized = false;
      setStatus({ ...status, saveError: error instanceof Error ? error.message : String(error) });
    } finally { connecting = null; }
  })();
  return connecting;
}

async function importBrowserPlaylist(): Promise<void> {
  if (!serverToken) return;
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw === null) return;
  try {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(raw));
    const sourceHash = `sha256:${Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('')}`;
    const result = await studioDraftsApi.importValue(serverToken, {
      storageKey: STORAGE_KEY, raw, sourceHash, schemaVersion: 1,
      expectedRevision: serverDocument?.revision ?? 0,
      ...(serverDocument ? { resolution: 'replace' as const } : {}),
    });
    if (!result.document) throw new Error('This browser snapshot was previously imported and its saved copy was deleted. Change the browser playlist before importing again.');
    serverDocument = result.document as PlaylistDocument;
    connected = true;
    _snapshot = serverDocument.body.items;
    setStatus({ needsImport: false, replacingServer: false, saveError: '', saving: false });
    try { localStorage.setItem(SERVER_POINTER_KEY, serverDocument.id); }
    catch (error) { console.warn('[Playlist] Browser migration marker was not saved:', error); }
  } catch (error) {
    setStatus({ ...status, saveError: error instanceof Error ? error.message : String(error) });
  }
}

async function retryPlaylistSave(): Promise<void> {
  if (!serverToken) return;
  try {
    if (!connected) { initialized = false; await connect(serverToken); return; }
    serverDocument = (await studioDraftsApi.playlist(serverToken)).document;
    status = { ...status, saveError: '' };
    await drain();
    announce();
  } catch (error) {
    setStatus({ ...status, saveError: error instanceof Error ? error.message : String(error) });
  }
}

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

function write(items: PlaylistItem[], immediate = false, command?: unknown): void {
  _snapshot = items;
  if (connected && command) { pending.push(command); void drain(); }
  else if (immediate) _persistPlaylistNow(); else _persistPlaylist();
  announce();
}

// ── Public API ───────────────────────────────────────────────────────────────

export function getPlaylist(): PlaylistItem[] { return read(); }

export function addToPlaylist(item: PlaylistItem): void {
  const list = read();
  if (list.some(i => i.id === item.id)) return;
  write([...list, item], true, { operation: 'add', item });
}

export function removeFromPlaylist(id: string): void {
  write(read().filter(i => i.id !== id), false, { operation: 'remove', id });
}

export function clearPlaylist(): void { write([], true, { operation: 'clear' }); }

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
  write(list.map(i => (i.id === id ? { ...i, ...patch } : i)), true, { operation: 'update', id, patch });
}

export function reorderPlaylist(items: PlaylistItem[]): void { write(items, true, { operation: 'reorder', ids: items.map(i => i.id) }); }

export function moveItem(id: string, direction: 'up' | 'down'): void {
  const list = [...read()];
  const idx = list.findIndex(i => i.id === id);
  if (idx < 0) return;
  const target = direction === 'up' ? idx - 1 : idx + 1;
  if (target < 0 || target >= list.length) return;
  [list[idx], list[target]] = [list[target], list[idx]];
  write(list, false, { operation: 'reorder', ids: list.map(i => i.id) });
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
  const { token } = useAuth();
  useEffect(() => { if (token) void connect(token); }, [token]);
  const items = useSyncExternalStore(subscribe, getSnapshot);
  const currentStatus = useSyncExternalStore(subscribe, () => status);

  const add = useCallback((item: PlaylistItem) => addToPlaylist(item), []);
  const remove = useCallback((id: string) => removeFromPlaylist(id), []);
  const clear = useCallback(() => clearPlaylist(), []);
  const isIn = useCallback((id: string) => items.some(i => i.id === id), [items]);
  const move = useCallback((id: string, dir: 'up' | 'down') => moveItem(id, dir), []);
  const reorder = useCallback((newItems: PlaylistItem[]) => reorderPlaylist(newItems), []);

  return { items, add, remove, clear, isIn, move, reorder,
    needsImport: currentStatus.needsImport, replacingServer: currentStatus.replacingServer,
    saveError: currentStatus.saveError, saving: currentStatus.saving,
    importBrowserPlaylist, retryPlaylistSave };
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
