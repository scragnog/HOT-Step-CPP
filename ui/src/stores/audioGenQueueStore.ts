/**
 * audioGenQueueStore.ts — Module-level singleton for sequential audio generation.
 *
 * Manages a queue of audio generation jobs with:
 * - Artist-batched execution (reorders pending items to minimize adapter switches)
 * - Progress polling and status tracking
 * - localStorage persistence — queue survives page reloads / HMR
 * - Resume on reload — in-flight jobs resume polling automatically
 *
 * Adapted for the C++ engine which accepts adapter params directly in the generate call
 * (no separate loadLora/unloadLora endpoints).
 *
 * Components subscribe via `useAudioGenQueue()` which uses `useSyncExternalStore`.
 */

import { useSyncExternalStore, useEffect, useRef, useCallback } from 'react';
import { lireekApi } from '../services/lireekApi';
import { generateApi, songApi } from '../services/api';
import { writePersistedState } from '../hooks/usePersistedState';
import type { Generation, AlbumPreset } from '../services/lireekApi';
import { addToPlaylist } from '../components/lyric-studio/playlistStore';
import type { GenerationParams } from '../types';
import { resolveDuration } from '../utils/estimateDuration';
import { createGenerationTimer, getGenerationTimeoutMinutes } from '../utils/generationTimer';

// ── Types ────────────────────────────────────────────────────────────────────

export type AudioQueueStatus = 'pending' | 'loading-adapter' | 'generating' | 'succeeded' | 'failed';

export interface AudioQueueItem {
  id: string;
  generation: Generation;
  artistId: number;
  artistName: string;
  artistImageUrl?: string;
  /** Track cover art URL — populated when the song resolves with cover_url from the DB.
   *  Preferred over artistImageUrl for playback backdrop / playlist thumbnail. */
  coverUrl?: string;
  preset: AlbumPreset | null;
  profileId: number;
  lyricsSetId: number;
  /** Snapshot of getGlobalParams() captured at enqueue time — same as Create page */
  globalParams: Partial<GenerationParams>;
  status: AudioQueueStatus;
  jobId?: string;
  progress?: number;
  stage?: string;
  elapsed?: number;
  error?: string;
  audioUrl?: string;
  songId?: string;
  masteredAudioUrl?: string;
  audioDuration?: number;
}

export interface AudioGenQueueState {
  items: AudioQueueItem[];
  completionCounter: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
// (mergeCreatePanelSettings removed — we now use getGlobalParams() snapshot
// passed in at enqueue time, identical to the Create page path.)

// ── Persistence (IndexedDB — no 5MB cap) ─────────────────────────────────────
// localStorage has a hard 5MB browser limit that large queues (600+ items with
// lyrics/prompts) easily exceed. IndexedDB has virtually unlimited storage
// (~50% of available disk space). Writes are async and non-blocking.

const IDB_NAME = 'lireek-queue-store';
const IDB_STORE = 'queue';
const IDB_KEY = 'state';
const LS_KEY = 'lireek-audio-gen-queue'; // legacy localStorage key for migration

let _persistTimer: ReturnType<typeof setTimeout> | null = null;

function _openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(IDB_STORE)) {
        req.result.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function _idbGet<T>(key: string): Promise<T | undefined> {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const store = tx.objectStore(IDB_STORE);
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result as T | undefined);
    req.onerror = () => reject(req.error);
  });
}

async function _idbSet(key: string, value: unknown): Promise<void> {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

/** Prepare queue data for persistence. Strips globalParams (only needed at
 *  submit time) to reduce storage churn — not for quota reasons anymore. */
function _dataForStorage(): { items: AudioQueueItem[]; completionCounter: number } {
  return {
    items: _state.items.map(item => {
      if (item.globalParams && Object.keys(item.globalParams).length > 0) {
        const { globalParams, ...rest } = item;
        return rest as AudioQueueItem;
      }
      return item;
    }),
    completionCounter: _state.completionCounter,
  };
}

/** Debounced persistence — avoids writes on every 2.5s poll tick. */
function _persist(): void {
  if (_persistTimer) clearTimeout(_persistTimer);
  _persistTimer = setTimeout(() => {
    _idbSet(IDB_KEY, _dataForStorage()).catch(e =>
      console.error('[AudioGenQueue] IDB write failed:', e));
  }, 2000);
}

/** Force-flush persistence immediately (status transitions). */
function _persistNow(): void {
  if (_persistTimer) { clearTimeout(_persistTimer); _persistTimer = null; }
  _idbSet(IDB_KEY, _dataForStorage()).catch(e =>
    console.error('[AudioGenQueue] IDB write failed:', e));
}

/** Sanitize restored items — reset in-flight items to pending. */
function _sanitizeItems(items: AudioQueueItem[]): AudioQueueItem[] {
  return items.map((item: AudioQueueItem) => {
    if (item.status === 'loading-adapter' || item.status === 'generating') {
      return {
        ...item,
        status: 'pending' as AudioQueueStatus,
        stage: 'Reconnecting…',
        progress: undefined,
        elapsed: undefined,
      };
    }
    return item;
  });
}

/** Async restore from IndexedDB (with one-time localStorage migration). */
async function _restoreFromIDB(): Promise<void> {
  try {
    // Try IDB first
    let data = await _idbGet<{ items: AudioQueueItem[]; completionCounter: number }>(IDB_KEY);

    // One-time migration from localStorage → IDB
    if (!data) {
      try {
        const raw = localStorage.getItem(LS_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          data = { items: parsed.items || [], completionCounter: parsed.completionCounter || 0 };
          // Write to IDB and remove from localStorage
          await _idbSet(IDB_KEY, data);
          localStorage.removeItem(LS_KEY);
          console.log('[AudioGenQueue] Migrated', data.items.length, 'items from localStorage to IndexedDB');
        }
      } catch (e) {
        console.warn('[AudioGenQueue] localStorage migration failed:', e);
      }
    }

    if (data && data.items && data.items.length > 0) {
      _state = {
        items: _sanitizeItems(data.items),
        completionCounter: data.completionCounter || 0,
      };
      // Notify UI of restored state
      _state = { ..._state, items: [..._state.items] };
      _listeners.forEach(fn => fn());
    }
  } catch (e) {
    console.error('[AudioGenQueue] IDB restore failed:', e);
  }
}

// ── Module-level singleton ───────────────────────────────────────────────────
// NOTE: _state and _listeners MUST be declared before _idbReady because
// _restoreFromIDB() writes to _state and notifies _listeners.

let _state: AudioGenQueueState = { items: [], completionCounter: 0 };
const _listeners = new Set<() => void>();

/** Ready gate — resolves when IDB restore is complete.
 *  resumeQueue() awaits this before processing items. */
const _idbReady: Promise<void> = _restoreFromIDB();

function _emit(immediate = false) {
  _state = { ..._state, items: [..._state.items] };
  if (immediate) _persistNow(); else _persist();
  _listeners.forEach(fn => fn());
}

function _getSnapshot(): AudioGenQueueState { return _state; }
function _subscribe(listener: () => void): () => void {
  _listeners.add(listener);
  return () => _listeners.delete(listener);
}

let _nextId = 0;
function _genId(): string { return `aq-${Date.now()}-${_nextId++}`; }

// ── Resume tracking ──────────────────────────────────────────────────────────

let _resumeCalled = false;

// ── Public API ───────────────────────────────────────────────────────────────

export async function enqueueAudioGen(
  gen: Generation,
  opts: { artistId: number; artistName: string; artistImageUrl?: string; profileId: number; lyricsSetId: number },
  globalParams: Partial<GenerationParams>,
  token: string,
): Promise<void> {
  let preset: AlbumPreset | null = null;
  try {
    const res = await lireekApi.getPreset(opts.lyricsSetId);
    preset = res.preset;
  } catch { /* no preset configured */ }

  const item: AudioQueueItem = {
    id: _genId(),
    generation: gen,
    artistId: opts.artistId,
    artistName: opts.artistName,
    artistImageUrl: opts.artistImageUrl,
    preset,
    profileId: opts.profileId,
    lyricsSetId: opts.lyricsSetId,
    globalParams,
    status: 'pending',
  };

  _state.items.push(item);
  _emit(true);
  _processQueue(token);
}

export function removeFromAudioQueue(id: string): void {
  _state.items = _state.items.filter(i => i.id !== id);
  _emit(true);
}

/** Force-dismiss an active/generating item (user clicked X).
 *  Also calls the server cancel API to stop the generation and C++ engine. */
export function forceFailQueueItem(id: string): void {
  const item = _state.items.find(i => i.id === id);
  if (item && (item.status === 'generating' || item.status === 'loading-adapter')) {
    // Cancel on the server → triggers abort controller → cancels C++ engine job
    if (item.jobId) {
      generateApi.cancel(item.jobId).catch(() => {});
    }
    item.status = 'failed';
    item.error = 'Cancelled by user';
    item.stage = undefined;
    item.progress = undefined;
    _emit(true);
  }
}

export function clearFinishedFromAudioQueue(): void {
  _state.items = _state.items.filter(i => i.status !== 'succeeded' && i.status !== 'failed');
  _emit(true);
}

// ── Manual queue API (for Cover Studio and other non-Lyric-Studio modules) ───

/** Add a pre-built item to the queue (no Lireek API calls, no preset lookup). */
export function addManualQueueItem(opts: {
  title: string;
  artistName?: string;
  caption?: string;
}): string {
  const id = _genId();
  const item: AudioQueueItem = {
    id,
    generation: {
      id: 0, profile_id: 0, provider: 'cover-studio', model: '',
      title: opts.title, caption: opts.caption || '', lyrics: '',
      created_at: new Date().toISOString(),
    },
    artistId: 0,
    artistName: opts.artistName || '',
    preset: null,
    profileId: 0,
    lyricsSetId: 0,
    globalParams: {},
    status: 'generating',
    stage: 'Submitting...',
  };
  _state.items.push(item);
  _emit(true);
  return id;
}

/** Update progress/stage of a manually-added queue item. */
export function updateManualQueueItem(id: string, update: {
  title?: string;
  jobId?: string;
  progress?: number;
  stage?: string;
  elapsed?: number;
  status?: AudioQueueStatus;
}): void {
  const item = _state.items.find(i => i.id === id);
  if (!item) return;
  if (update.title !== undefined) item.generation.title = update.title;
  if (update.jobId !== undefined) item.jobId = update.jobId;
  if (update.progress !== undefined) item.progress = update.progress;
  if (update.stage !== undefined) item.stage = update.stage;
  if (update.elapsed !== undefined) item.elapsed = update.elapsed;
  if (update.status !== undefined) item.status = update.status;
  _emit(true);
}

/** Mark a manually-added queue item as succeeded with audio results. */
export function completeManualQueueItem(id: string, result: {
  audioUrl: string;
  songId?: string;
  masteredAudioUrl?: string;
  audioDuration?: number;
}): void {
  const item = _state.items.find(i => i.id === id);
  if (!item) return;
  item.status = 'succeeded';
  item.audioUrl = result.audioUrl;
  if (result.songId) item.songId = result.songId;
  if (result.masteredAudioUrl) item.masteredAudioUrl = result.masteredAudioUrl;
  if (result.audioDuration) item.audioDuration = result.audioDuration;
  item.progress = 100;
  item.stage = 'Complete!';
  _state.completionCounter++;
  _emit(true);
  _maybeAutoAddToPlaylist(item);

  // Notify App.tsx so Library updates in real-time
  if (result.songId) _notifySongCreated(result.songId);

  // If server didn't provide duration, probe the audio file
  if (!item.audioDuration && result.audioUrl) {
    _probeAudioDuration(id, result.audioUrl);
  }
}

/** Mark a manually-added queue item as failed. */
export function failManualQueueItem(id: string, error: string): void {
  const item = _state.items.find(i => i.id === id);
  if (!item) return;
  item.status = 'failed';
  item.error = error;
  item.progress = undefined;
  item.stage = undefined;
  _emit(true);
}

/** Force-reset the server's generation queue and mark all local active items as failed. */
export async function resetServerQueue(): Promise<{ cancelled: number; drained: number }> {
  const result = await generateApi.resetQueue();
  // Mark all local active items as failed
  for (const item of _state.items) {
    if (item.status === 'pending' || item.status === 'loading-adapter' || item.status === 'generating') {
      item.status = 'failed';
      item.error = 'Queue reset';
      item.stage = undefined;
      item.progress = undefined;
    }
  }
  _emit(true);
  return result;
}
// ── Auto-add to playlist ─────────────────────────────────────────────────────

const SEND_TO_PLAYLIST_KEY = 'hs-sendToPlaylist';

/** Read the "Send To Playlist" toggle from localStorage. */
export function getSendToPlaylist(): boolean {
  try {
    const raw = localStorage.getItem(SEND_TO_PLAYLIST_KEY);
    return raw ? JSON.parse(raw) === true : false;
  } catch { return false; }
}

/** Write the "Send To Playlist" toggle to localStorage. */
export function setSendToPlaylist(enabled: boolean): void {
  localStorage.setItem(SEND_TO_PLAYLIST_KEY, JSON.stringify(enabled));
}

/** If the toggle is on, auto-add the completed item to the playlist. */
function _maybeAutoAddToPlaylist(item: AudioQueueItem): void {
  if (!getSendToPlaylist()) return;
  const resolvedId = item.songId || item.id;
  addToPlaylist({
    id: resolvedId,
    title: item.generation.title || 'Untitled',
    audioUrl: item.audioUrl || '',
    masteredAudioUrl: item.masteredAudioUrl || '',
    artistName: item.artistName || '',
    coverUrl: item.coverUrl || item.artistImageUrl || '',
    duration: item.audioDuration || 0,
  });
}

// ── Song-created notification ────────────────────────────────────────────────

/**
 * After a queue item completes with a songId, fetch the full song from the API
 * and dispatch a CustomEvent so App.tsx can add it to the library state.
 * Also backfills the queue item's coverUrl from the song's cover_url so the
 * playback backdrop shows the track's cover art instead of the artist image.
 */
async function _notifySongCreated(songId: string): Promise<void> {
  try {
    const { song } = await songApi.get(songId);
    if (song) {
      window.dispatchEvent(new CustomEvent('song-created', { detail: { song } }));
      // Backfill coverUrl into the queue item so the playback track uses cover art
      const coverUrl = song.coverUrl || song.cover_url;
      if (coverUrl) {
        const item = _state.items.find(i => i.songId === songId);
        if (item && !item.coverUrl) {
          item.coverUrl = coverUrl;
          _emit(true);
        }
      }
    }
  } catch {
    // Non-fatal — the song is saved, it'll appear on next reload
    console.warn('[AudioQueue] Could not fetch song for library notification:', songId);
  }
}

// ── Audio duration probing ───────────────────────────────────────────────────

/** Probe an audio URL with a hidden Audio element to get the real duration.
 *  Fetches the file as a blob first to avoid race conditions when served
 *  via tunnels/proxies where the file may not be fully ready yet. */
async function _probeAudioDuration(itemId: string, url: string): Promise<void> {
  let blobUrl: string | undefined;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    const blob = await res.blob();
    blobUrl = URL.createObjectURL(blob);

    const audio = new Audio();
    audio.preload = 'metadata';

    audio.onloadedmetadata = () => {
      const dur = audio.duration;
      if (dur && isFinite(dur) && dur > 0) {
        const item = _state.items.find(i => i.id === itemId);
        if (item) {
          item.audioDuration = Math.round(dur);
          _emit(true);
        }
      }
      cleanup();
    };

    audio.onerror = () => {
      cleanup();
    };

    audio.src = blobUrl;

    function cleanup() {
      audio.src = '';
      audio.onloadedmetadata = null;
      audio.onerror = null;
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        blobUrl = undefined;
      }
    }
  } catch {
    // Non-fatal — duration simply won't be updated
    if (blobUrl) URL.revokeObjectURL(blobUrl);
  }
}

// ── Simple generation API (for Create page) ─────────────────────────────────

/**
 * Enqueue a simple generation from the Create page.
 * Submits to the generate API, tracks progress in the shared queue,
 * and calls onSongCreated for each resulting song.
 *
 * This replaces useGenerationStore for unified queue management.
 */
export async function enqueueSimpleGen(
  params: Record<string, any>,
  token: string,
  onSongCreated?: (song: any) => void,
): Promise<void> {
  const title = (params.title as string) || 'Untitled';
  const id = _genId();
  const item: AudioQueueItem = {
    id,
    generation: {
      id: 0, profile_id: 0, provider: 'create', model: '',
      title, caption: (params.caption as string) || '', lyrics: (params.lyrics as string) || '',
      created_at: new Date().toISOString(),
    },
    artistId: 0,
    artistName: '',
    preset: null,
    profileId: 0,
    lyricsSetId: 0,
    globalParams: params,
    status: 'generating',
    stage: 'Submitting…',
  };
  _state.items.push(item);
  _emit(true);

  try {
    const res = await generateApi.submit(params as any, token);
    item.jobId = res.jobId;
    item.stage = 'Queued…';
    _emit(true);

    // Poll until done. Timer ignores server-queue wait — only counts real
    // generation time so deep queues don't inflate elapsed / trip the timeout.
    const timer = createGenerationTimer();
    while (true) {
      await new Promise(r => setTimeout(r, 1500));
      try {
        const status = await generateApi.status(res.jobId);
        const t = timer.tick(status.status);
        item.progress = status.progress !== undefined
          ? Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100))
          : undefined;
        item.stage = status.stage || 'Generating…';
        item.elapsed = t.elapsed;
        _emit();  // progress tick — debounced persistence

        if (status.status === 'succeeded') {
          const audioUrl = status.result?.audioUrls?.[0] || '';
          const songIds = status.result?.songIds || [];
          const masteredUrl = status.result?.masteredAudioUrl;
          item.status = 'succeeded';
          item.audioUrl = audioUrl;
          item.songId = songIds[0];
          item.masteredAudioUrl = masteredUrl;
          item.audioDuration = status.result?.duration;
          item.progress = 100;
          item.stage = 'Complete!';
          _state.completionCounter++;
          _emit(true);
          _maybeAutoAddToPlaylist(item);

          // If server didn't provide duration, probe the audio file
          if (!item.audioDuration && audioUrl) {
            _probeAudioDuration(item.id, audioUrl);
          }

          // Fetch and deliver songs to the library
          if (onSongCreated) {
            for (const songId of songIds) {
              try {
                const { song } = await songApi.get(songId);
                onSongCreated(song);
              } catch { /* non-fatal */ }
            }
          }
          return;
        }
        if (status.status === 'failed' || status.status === 'cancelled') {
          throw new Error(status.error || (status.status === 'cancelled' ? 'Cancelled' : 'Generation failed'));
        }
        // Safety: configurable timeout (default 30 min), measured from
        // generation start — queue wait does not count.
        if (t.timedOut) {
          throw new Error(`Generation timed out after ${getGenerationTimeoutMinutes()} minutes`);
        }
      } catch (e) {
        if ((e as Error).message.includes('failed') || (e as Error).message.includes('timed out') || (e as Error).message.includes('Cancelled')) {
          throw e;
        }
        // Transient network error — keep polling
      }
    }
  } catch (err) {
    item.status = 'failed';
    item.error = (err as Error).message;
    item.progress = undefined;
    item.stage = undefined;
    _emit(true);
  }
}

export async function resumeQueue(token: string): Promise<void> {
  if (_resumeCalled) return;
  _resumeCalled = true;

  // Wait for IndexedDB restore to complete before processing
  await _idbReady;

  _pruneDeletedSongs(token);

  const hasPending = _state.items.some(i => i.status === 'pending');
  if (hasPending) {
    _processQueue(token);
  }
}

/**
 * Drop succeeded items whose song no longer exists in the DB — the queue is
 * persisted in IndexedDB, so entries survive nukes/deletes done before this
 * page load or from another tab. Candidates are captured BEFORE the fetch so
 * an item that completes mid-fetch can never be pruned by a stale id list.
 */
async function _pruneDeletedSongs(token: string): Promise<void> {
  const candidates = _state.items
    .filter(i => i.status === 'succeeded' && i.songId)
    .map(i => i.id);
  if (candidates.length === 0) return;

  try {
    const { ids } = await songApi.listIds(token);
    const existing = new Set(ids);
    const candidateSet = new Set(candidates);
    const before = _state.items.length;
    _state.items = _state.items.filter(i =>
      !candidateSet.has(i.id) || existing.has(i.songId!)
    );
    if (_state.items.length !== before) {
      console.log(`[AudioGenQueue] Pruned ${before - _state.items.length} queue entries for deleted songs`);
      _emit(true);
    }
  } catch {
    // Non-fatal — server unreachable, keep entries as-is
  }
}

// ── Queue runner ─────────────────────────────────────────────────────────────

let _running = false;

async function _processQueue(token: string): Promise<void> {
  if (_running) return;
  _running = true;

  try {
    while (true) {
      const pending = _state.items.filter(i => i.status === 'pending');
      if (pending.length === 0) break;

      // Artist batching: prefer items with same adapter as last completed
      const next = pending[0];

      try {
        // If the item has a jobId from a previous session, try to reconnect
        // to the server job (browser-only reload). If the server doesn't know
        // about it (404 = full restart), clear the jobId and re-submit.
        if (next.jobId) {
          const reconnected = await _tryReconnect(next, token);
          if (reconnected) {
            // _tryReconnect already set the item's status, audioUrl, songId, etc.
            // Just ensure it's counted and notified.
            if (next.status === 'succeeded') {
              _state.completionCounter++;
              if (next.songId) _notifySongCreated(next.songId);
              _maybeAutoAddToPlaylist(next);
            }
            _emit(true);
            continue;
          }
          // Server doesn't know about this job — clear and re-submit
          console.log(`[AudioQueue] Job ${next.jobId} not found on server — re-submitting`);
          next.jobId = undefined;
          next.stage = 'Re-submitting…';
          _emit(true);
        }

        await _executeItem(next, token);
        next.status = 'succeeded';
        _state.completionCounter++;
        // Notify App.tsx so Library updates in real-time
        if (next.songId) _notifySongCreated(next.songId);
        _maybeAutoAddToPlaylist(next);
      } catch (err) {
        next.status = 'failed';
        next.error = (err as Error).message;
        console.error(`[AudioQueue] Item ${next.id} failed:`, (err as Error).message);
      }
      _emit(true);
    }
  } finally {
    _running = false;
    console.log('[AudioQueue] Queue processor stopped');
  }
}

/**
 * Try to reconnect to a server job from a previous browser session.
 * Returns true if the job was found and polled to completion, false if
 * the server doesn't know about it (needs re-submit).
 */
async function _tryReconnect(item: AudioQueueItem, _token: string): Promise<boolean> {
  const jobId = item.jobId!;
  console.log(`[AudioQueue] Attempting to reconnect to job ${jobId}...`);

  try {
    const status = await generateApi.status(jobId);
    // Server knows about this job — reconnect!
    if (status.status === 'succeeded') {
      // Already done — just collect the results
      const audioUrl = status.result?.audioUrls?.[0];
      const songId = status.result?.songIds?.[0];
      const masteredUrl = status.result?.masteredAudioUrl;
      if (audioUrl) {
        item.audioUrl = audioUrl;
        if (songId) item.songId = songId;
        if (masteredUrl) item.masteredAudioUrl = masteredUrl;
        if (status.result?.duration) item.audioDuration = status.result.duration;
      }
      item.status = 'succeeded';
      item.progress = 100;
      item.stage = 'Complete!';
      console.log(`[AudioQueue] Reconnected to job ${jobId} — already succeeded`);
      return true;
    }
    if (status.status === 'failed' || status.status === 'cancelled') {
      item.status = 'failed';
      item.error = status.error || (status.status === 'cancelled' ? 'Cancelled' : 'Failed');
      console.log(`[AudioQueue] Reconnected to job ${jobId} — ${status.status}`);
      return true; // Don't re-submit a failed job
    }

    // Still running — resume polling
    console.log(`[AudioQueue] Reconnected to job ${jobId} — resuming poll (status=${status.status})`);
    item.status = 'generating';
    item.stage = status.stage || 'Reconnected…';
    item.progress = status.progress;
    _emit(true);

    await _pollUntilDone(item, _token);

    // _pollUntilDone sets audioUrl etc. on success, or throws on failure.
    // Mark succeeded here (poll doesn't set status).
    item.status = 'succeeded';
    item.progress = 100;
    item.stage = 'Complete!';

    // Resolve audio generation in Lireek DB if applicable
    if (item.audioUrl && jobId) {
      try {
        await lireekApi.resolveAudioGeneration(jobId, item.audioUrl);
      } catch { /* non-fatal */ }
    }

    return true;
  } catch (err) {
    // 404 or network error — server doesn't know about this job
    const msg = (err as Error).message || '';
    if (msg.includes('404') || msg.includes('not found') || msg.includes('Job not found')) {
      console.log(`[AudioQueue] Job ${jobId} not found on server (server restarted?)`);
      return false;
    }
    // Transient network error — try once more
    console.warn(`[AudioQueue] Reconnect probe failed for ${jobId}:`, msg);
    return false;
  }
}

async function _executeItem(item: AudioQueueItem, token: string): Promise<void> {
  const gen = item.generation;
  const preset = item.preset;

  // 1) Start with globalParams snapshot — identical to Create page's getGlobalParams().
  //    This includes ALL engine params: inference, guidance, solver, DCW, latent,
  //    LM, adapter (global), mastering, trigger word, etc.
  const params: Record<string, any> = { ...item.globalParams };

  // 2) Overlay content fields from the written song
  params.lyrics = gen.lyrics || '';
  params.caption = gen.caption || '';
  params.title = gen.title || '';
  params.instrumental = false;
  params.duration = resolveDuration(gen.duration, gen.lyrics || '', gen.bpm || 120);
  if (gen.bpm) params.bpm = gen.bpm;
  if (gen.key) params.keyScale = gen.key;
  if (item.artistName) params.artist = item.artistName;
  if (gen.subject) params.subject = gen.subject;

  // 2b) Per-song params that the Create page sends from its local state.
  //     Read from the same localStorage keys to maintain parity.
  //     Fallbacks match CreatePanel defaults: '' (auto) for timesig, 'en' for language.
  if (!params.timeSignature) {
    try {
      const stored = localStorage.getItem('hs-timeSignature');
      params.timeSignature = stored ? JSON.parse(stored) : '';
    } catch { params.timeSignature = ''; }
  }
  if (!params.vocalLanguage) {
    try {
      const stored = localStorage.getItem('hs-vocalLanguage');
      params.vocalLanguage = stored ? JSON.parse(stored) : 'en';
    } catch { params.vocalLanguage = 'en'; }
  }

  // 2c) Settings flags — Create page adds these from App settings.
  //     Read from localStorage (same key as App.tsx / SettingsPanel).
  try {
    const settingsRaw = localStorage.getItem('ace-settings');
    if (settingsRaw) {
      const appSettings = JSON.parse(settingsRaw);
      params.coResident = appSettings.coResident;
      params.cacheLmCodes = appSettings.cacheLmCodes;
      params.parallelWhisper = appSettings.parallelWhisper;
      params.parallelQualityEval = appSettings.parallelQualityEval;
      params.parallelCoverArt = appSettings.parallelCoverArt;
      // Pass timeout to server so server-side wall-clock limit matches user preference
      if (typeof appSettings.generationTimeoutMinutes === 'number') {
        params.generationTimeoutMinutes = appSettings.generationTimeoutMinutes;
      }
    }
  } catch { /* ignore parse errors */ }

  // 3) Adapter override from album preset (path from preset, scale/groups from globalParams)
  if (preset?.adapter_path) {
    item.status = 'loading-adapter';
    item.stage = `Preparing adapter for ${item.artistName}…`;
    _emit(true);

    // Update the top bar to reflect the adapter being used
    writePersistedState('hs-adapter', preset.adapter_path);

    // Override adapter path from preset; scale, group scales, mode, and
    // trigger word settings are already correct from globalParams.
    params.loraPath = preset.adapter_path;
    // loraStack supersedes loraPath in translateParams (req.adapters wins in
    // the engine), and getGlobalParams folds even a single top-bar adapter
    // into a stack — so the stack must be replaced too, or whatever was
    // loaded in the top panel keeps playing instead of the preset adapter.
    params.loraStack = [{
      path: preset.adapter_path,
      scale: typeof params.loraScale === 'number' ? params.loraScale : 1.0,
    }];

    // Re-derive trigger word from the PRESET adapter filename (globalParams
    // has the trigger word for the GLOBAL adapter, which may differ).
    const settingsRaw = localStorage.getItem('ace-settings');
    const triggerSettings = settingsRaw ? JSON.parse(settingsRaw) : {};
    const useFilename = triggerSettings.triggerUseFilename === true;
    const placement = (triggerSettings.triggerPlacement as 'prepend' | 'append' | 'replace') || 'prepend';
    // Clear the snapshot's trigger words first: translateParams prefers the
    // plural triggerWords (derived from the global adapter stack) over the
    // singular triggerWord set below, so stale globals must not survive.
    delete params.triggerWord;
    delete params.triggerWords;
    delete params.triggerPlacement;
    if (useFilename) {
      const fileName = preset.adapter_path.replace(/\\/g, '/').split('/').pop() || '';
      const triggerWord = fileName.replace(/\.safetensors$/i, '');
      if (triggerWord) {
        params.triggerWord = triggerWord;
        params.triggerWords = [triggerWord];
        params.triggerPlacement = placement;
      }
    }
  }

  // 4) Mastering reference from album preset (does NOT force-enable — respects global toggle)
  if (preset?.reference_track_path) {
    // Update the top bar to reflect the mastering reference
    writePersistedState('hs-masteringReference', preset.reference_track_path);
    writePersistedState('hs-timbreReference', true);

    params.masteringReference = preset.reference_track_path;
    // Timbre: use dedicated timbre path from globalParams if set,
    // otherwise default to preset reference track
    if (typeof item.globalParams?.timbreReference === 'string' && item.globalParams.timbreReference) {
      params.timbreReference = item.globalParams.timbreReference;  // dedicated timbre audio path
    } else {
      params.timbreReference = true;  // use preset reference track as timbre
    }

    // Randomize Timbre: pick a random track from the same folder instead of the exact file
    try {
      const raw = localStorage.getItem('lireek-randomizeTimbreRef');
      if (raw && JSON.parse(raw) === true) {
        params.randomizeTimbreRef = true;
      }
    } catch { /* ignore */ }
  }

  // 5) Submit generation
  params.taskType = 'text2music';
  params.source = 'lyric-studio';
  item.status = 'generating';
  item.stage = 'Submitting to audio engine…';
  _emit(true);

  const res = await generateApi.submit(params as any, token);
  const jobId = res.jobId;
  item.jobId = jobId;
  _emit(true); // persist jobId immediately

  // 6) Link audio to Lireek generation
  if (jobId) {
    await lireekApi.linkAudio(gen.id, jobId);
  }

  // 7) Poll until done
  await _pollUntilDone(item, token);
}

async function _pollUntilDone(item: AudioQueueItem, _token: string): Promise<void> {
  const jobId = item.jobId!;
  item.stage = 'Generating audio…';
  // Resume the clock from any persisted elapsed (reconnect after reload);
  // a fresh item starts counting only once the engine picks the job up.
  const timer = createGenerationTimer({ resumeElapsedSec: item.elapsed });
  _emit(true);

  while (true) {
    await new Promise(r => setTimeout(r, 2500));
    try {
      const status = await generateApi.status(jobId);
      const t = timer.tick(status.status);
      item.progress = status.progress !== undefined
        ? Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100))
        : undefined;
      item.stage = status.stage || 'Generating…';
      item.elapsed = t.elapsed;
      _emit();  // progress tick — debounced persistence

      if (status.status === 'succeeded') {
        const audioUrl = status.result?.audioUrls?.[0];
        const songId = status.result?.songIds?.[0];
        const masteredUrl = status.result?.masteredAudioUrl;
        if (audioUrl) {
          item.audioUrl = audioUrl;
          if (songId) item.songId = songId;
          if (masteredUrl) item.masteredAudioUrl = masteredUrl;
          if (status.result?.duration) item.audioDuration = status.result.duration;
          _emit(true);
          // If server didn't provide duration, probe the audio file
          if (!item.audioDuration) _probeAudioDuration(item.id, audioUrl);
        }
        // Resolve audio generation in Lireek DB
        if (audioUrl && jobId) {
          try {
            await lireekApi.resolveAudioGeneration(jobId, audioUrl);
          } catch { /* non-fatal */ }
        }
        return;
      }
      if (status.status === 'failed' || status.status === 'cancelled') throw new Error(status.error || (status.status === 'cancelled' ? 'Cancelled' : 'Generation failed'));

      // Safety: configurable timeout (default 30 min), measured from
      // generation start — queue wait does not count.
      if (t.timedOut) {
        throw new Error(`Generation timed out after ${getGenerationTimeoutMinutes()} minutes`);
      }
    } catch (e) {
      const msg = (e as Error).message;
      if ((msg.includes('failed') || msg.includes('Cancelled') || msg.includes('timed out')) && !msg.includes('fetch')) {
        throw e;
      }
      // Transient network error — keep polling
    }
  }
}



// ── React hooks ──────────────────────────────────────────────────────────────

export function useAudioGenQueue(token?: string): AudioGenQueueState {
  useEffect(() => {
    // No items guard — _state is empty at this point because IDB restore is
    // async. resumeQueue itself awaits _idbReady before checking for work.
    if (token) {
      resumeQueue(token);
    }
  }, [token]);

  return useSyncExternalStore(_subscribe, _getSnapshot, _getSnapshot);
}

/** Resume queue on mount (if items exist).  Does NOT subscribe to queue state. */
export function useResumeQueue(token?: string): void {
  useEffect(() => {
    if (token) {
      resumeQueue(token);
    }
  }, [token]);
}

/**
 * Subscribe to a derived slice of queue state.  The component only re-renders
 * when the selected value changes (Object.is equality).
 *
 * @example
 *   const activeCount = useAudioGenQueueSelector(s =>
 *     s.items.filter(i => i.status === 'generating').length
 *   );
 */
export function useAudioGenQueueSelector<T>(selector: (state: AudioGenQueueState) => T): T {
  const selectorRef = useRef(selector);
  selectorRef.current = selector;

  const selectedRef = useRef<T>(selector(_state));

  const getSelectedSnapshot = useCallback(() => {
    const next = selectorRef.current(_state);
    if (Object.is(selectedRef.current, next)) return selectedRef.current;
    selectedRef.current = next;
    return next;
  }, []);

  return useSyncExternalStore(_subscribe, getSelectedSnapshot, getSelectedSnapshot);
}
