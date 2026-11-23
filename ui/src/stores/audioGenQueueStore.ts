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
import { generateApi, withIntentTimeout, fetchAudioBlob } from '../services/api';
import type { Generation, AlbumPreset } from '../services/lireekApi';
import { addToPlaylist } from './playlistStore';
import type { GenerationParams, GenerationJob } from '../types';
import { MM3_BACKEND_ID } from '../utils/captionForBackend';
import { yue2CaptionAdapterPath, yue2PickAtEnqueue, YUE2_BACKEND_ID } from '../utils/yue2CaptionSource';
import { useLmAdapterEnabled } from '../utils/lmAdapterPref';
import { masterToAlbumEnabled } from '../utils/masterToAlbumPref';
import { useBackendStore } from './backendStore';
import type { CreateIntent, WrittenSongIntent } from '../../../server/src/contracts/resolution';
import { readMm3CaptionSelection } from '../utils/mm3CaptionSource';
import { readYue2SongSelection, readYue2DatasetForLyricsSet } from '../utils/yue2CaptionSource';
import { getUseLlmDuration } from '../utils/estimateDuration';
import { audioQueueApi } from '../services/audioQueueApi';
import { queryClient } from '../queries/client';
import { ACTIVE_POLL_MS } from '../queries/contract';
import {
  readAudioQueue, readGenerationStatus, readSong, readWorkflowJob,
} from '../queries/shell-queue/queue';
import { workflowApi } from '../services/workflowApi';
import { lyricWorkflowApi } from '../services/lyricWorkflowApi';
import type { WorkflowJob } from '../../../server/src/contracts/workflow';
import { exportQueueBackup, type QueueBackup, type QueueBackupStore } from './audioQueueMigration';
import { decideQueueOwnerAction } from './audioQueueOwnerDecision';
import type { AudioIntentItem, ImportAudioQueue } from '../../../server/src/contracts/audioQueue';

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
  /** The album this renders AS: its preset, adapters and caption source, and
   *  the album the recording is listed under. */
  lyricsSetId: number;
  /** The lyrics' own album, when it differs from `lyricsSetId` ("Render as"). */
  sourceLyricsSetId?: number;
  /** Snapshot of getGlobalParams() captured at enqueue time — same as Create page */
  globalParams: Partial<GenerationParams>;
  /** YuE2 model pick captured at enqueue (yue2PickAtEnqueue): the adapters
   *  this song renders with, whatever the picker says when it runs (#204).
   *  Absent on items queued before this existed; they take it at run time. */
  yue2Pick?: Record<string, string | number>;
  status: AudioQueueStatus;
  jobId?: string;
  /** Manual rows run by a workflow job (Insta-Gen); cancel goes there. */
  workflowJobId?: string;
  serverQueueId?: string;
  progress?: number;
  stage?: string;
  elapsed?: number;
  /** Lead job id of the server batch this item rendered in (YuE2 queue
   *  coalescing); the queue boxes items sharing one. */
  batchId?: string;
  /** 'lm' when the batch only shared an ACE LM pass and rendered one by one. */
  batchKind?: string;
  /** Items in the submit wave this item went up with (ACE coalescing hint). */
  lmWave?: number;
  error?: string;
  audioUrl?: string;
  songId?: string;
  masteredAudioUrl?: string;
  noAdapterAudioUrl?: string;
  audioDuration?: number;
  /** MiniMax-Music3 "play while rendering": the ENGINE has confirmed it will
   *  serve this job's audio live. Only true once the job reaches the engine
   *  and streaming was both requested and accepted — the Create panel shows
   *  its Listen affordance off this, never off the request flag, so a decline
   *  simply means no player rather than a button that 409s. */
  mm3Streaming?: boolean;
  /** True when the engine renders windows WHILE it plans (audio in seconds).
   *  False means it fell back to the serial path and audio starts once
   *  planning finishes — still a stream, just a later one. Undefined until the
   *  engine has decided. */
  mm3Interleaved?: boolean;
  /** Resolved render length in seconds (engine-side), for the streaming card's
   *  progress and the play bar's duration. */
  mm3Duration?: number;
  /** ── Ensemble takes ──
   *  One render can produce several DIFFERENT songs from the same prompt. Each
   *  becomes its OWN queue entry so it can be watched, streamed and played
   *  independently — a single entry holding three songs is a batch, not three
   *  tracks.
   *
   *  The entries share one `jobId`. Exactly one of them (`mm3Take === 0`, the
   *  one that was enqueued) polls it; the siblings are created by the store
   *  when the engine reports the take count, and are driven from that poll.
   *  `mm3TakeOf` being set is what marks an entry as a non-polling sibling. */
  mm3Take?: number;
  /** Total takes in this render. Present on every entry of the group. */
  mm3TakeCount?: number;
  /** Set on siblings only — the id of the entry that owns the polling. */
  mm3TakeOf?: string;
  /** This take's seed, lossless (decimal string). */
  mm3TakeSeed?: string;
  /** A Lyric Studio render that failed before it reached the server queue,
   *  projected from its lyric-render job (refreshFailedRenders). Retry submits
   *  `intent` again; X and Clear Done dismiss `key`. */
  failedRender?: { key: string; intent?: WrittenSongIntent };
}

export interface AudioGenQueueState {
  items: AudioQueueItem[];
  completionCounter: number;
  /** Number of never-submitted items found in storage on this page load and
   *  held back rather than started. Zero once the user answers, and zero on any
   *  load where there was nothing waiting. See resumeQueue(). */
  awaitingResume: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
// (mergeCreatePanelSettings removed — we now use getGlobalParams() snapshot
// passed in at enqueue time, identical to the Create page path.)

// ── MM3 live audio ───────────────────────────────────────────────────────────

/** Copy the MiniMax-Music3 streaming flags off a status poll onto the item.
 *  Every loop that polls a job for a row needs this, or the Create page shows
 *  no player while the engine streams; hence a function, not two statements. */
function _captureMm3Stream(item: AudioQueueItem, status: GenerationJob): void {
  if (status.mm3_streaming === true && !item.mm3Streaming) item.mm3Streaming = true;
  if (status.mm3_interleaved !== null && status.mm3_interleaved !== undefined) {
    item.mm3Interleaved = status.mm3_interleaved;
  }
  if (status.mm3_duration) item.mm3Duration = status.mm3_duration;
  _expandTakes(item, status);
}

/** Split an ensemble render into one queue entry PER TAKE.
 *
 *  Runs the first time the engine reports a take count above one, which is as
 *  soon as it has the job — long before any audio. That timing is the whole
 *  point: the entries have to exist while the takes are still rendering, so
 *  each can be watched and listened to as it grows. Creating them at the end
 *  would just be a batch that reveals itself once there is nothing left to
 *  watch.
 *
 *  Only the ORIGINAL entry polls the job. The siblings carry `mm3TakeOf` and
 *  are driven from that one poll (_syncTakeSiblings), because K entries polling
 *  one job id would be K times the requests for one answer.
 *
 *  Idempotent: a poll tick runs several times a second and must not keep
 *  adding entries. */
function _expandTakes(item: AudioQueueItem, status: GenerationJob): void {
  const takes = Number(status.mm3_takes ?? 1);
  if (takes <= 1 || item.mm3TakeOf || item.mm3TakeCount) return;

  const seeds = status.mm3_take_seeds ?? [];
  item.mm3Take = 0;
  item.mm3TakeCount = takes;
  item.mm3TakeSeed = seeds[0];
  // One render, so one box in the queue, like a coalesced YuE2 batch.
  item.batchId ??= item.jobId;

  const at = _state.items.findIndex(i => i.id === item.id);
  if (at < 0) return;
  if (_state.items.some(i => i.mm3TakeOf === item.id)) return;
  const siblings: AudioQueueItem[] = [];
  for (let t = 1; t < takes; t++) {
    siblings.push({
      ...item,
      // A distinct id, because everything downstream keys on it — but the SAME
      // jobId, because there is one render behind all of them.
      id: `${item.id}:take${t}`,
      mm3Take: t,
      mm3TakeCount: takes,
      mm3TakeOf: item.id,
      mm3TakeSeed: seeds[t],
      // Nothing of take 0's output belongs to a sibling.
      audioUrl: undefined,
      songId: undefined,
      masteredAudioUrl: undefined,
      noAdapterAudioUrl: undefined,
      audioDuration: undefined,
    });
  }
  _state.items.splice(at + 1, 0, ...siblings);
  console.log(`[MM3 Takes] ${item.jobId}: ${takes} takes -> ${takes} queue entries`);
}

/** Mirror the polling entry's live state onto its take siblings. They share one
 *  render, so progress, stage and streaming flags are common to all of them;
 *  only the finished OUTPUT differs, and that is assigned per take on success. */
function _syncTakeSiblings(item: AudioQueueItem, status?: GenerationJob): void {
  if (!item.mm3TakeCount || item.mm3TakeCount <= 1) return;
  // Natural-ending candidates: the server's take count SHRINKS once the planner
  // has dropped the capped candidates. Any sibling past the surviving count is
  // a plan that never reached an ending and will never get audio; say so on
  // its card instead of leaving it spinning until the render finishes.
  const survived = status ? Number(status.mm3_takes ?? item.mm3TakeCount) : item.mm3TakeCount;
  for (const s of _state.items) {
    if (s.mm3TakeOf !== item.id) continue;
    if ((s.mm3Take ?? 0) >= survived) {
      if (s.status !== 'failed') {
        s.status = 'failed';
        s.progress = 0;
        s.stage = 'No natural ending';
        s.error = 'Reached the length cap without ending; dropped by Require Natural Ending';
      }
      continue;
    }
    s.status = item.status;
    s.progress = item.progress;
    s.stage = item.stage;
    s.elapsed = item.elapsed;
    s.error = item.error;
    s.jobId = item.jobId;
    s.mm3Streaming = item.mm3Streaming;
    s.mm3Interleaved = item.mm3Interleaved;
    s.mm3Duration = item.mm3Duration;
  }
}

// ── The v1.3.x browser queue (IndexedDB, read only) ──────────────────────────
// Releases before v1.4.0 ran the queue in the browser and kept it here. The
// server queue owns every render now; this store is only read once, by the
// one-shot import below, and never written or deleted: it and its backups are
// the user's only copy of what they had queued.

const IDB_NAME = 'lireek-queue-store';
const IDB_STORE = 'queue';
const IDB_KEY = 'state';
const LS_KEY = 'lireek-audio-gen-queue'; // legacy localStorage key for migration
export const OWNER_KEY = 'lireek-audio-queue-owner-v1';
export const BACKUP_KEY = 'server-migration-backup-v1';
/** The exact resolved import request, persisted before the POST so a reload can replay it verbatim. */
export const PENDING_IMPORT_KEY = `${BACKUP_KEY}:pending-import`;
/** 'browser' is the never-decided state: no key, or a v1.3.x/rolled-back
 *  'browser' value, which the load-time decision imports like any other. */
type QueueOwner = 'browser' | 'migrating' | 'server';
function queueOwner(): QueueOwner {
  const value = localStorage.getItem(OWNER_KEY);
  return value === 'server' || value === 'migrating' ? value : 'browser';
}

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
          // Keep the legacy source until an explicit, verified server import.
          await _idbSet(IDB_KEY, data);
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
        // Never restored from storage. resumeQueue() decides it fresh on every
        // page load, from what it actually finds waiting.
        awaitingResume: 0,
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

let _state: AudioGenQueueState = { items: [], completionCounter: 0, awaitingResume: 0 };
/** Rows addManualQueueItem made on this page; refreshServerProjection keeps
 *  them until the server queue has their render. */
const _manualIds = new Set<string>();
const _listeners = new Set<() => void>();

/** Ready gate — resolves when IDB restore is complete.
 *  The owner decision awaits this before reading what is pending. */
const _idbReady: Promise<void> = _restoreFromIDB().then(() => { void _hydrateFinishedItems(); });

function migrationStore(): QueueBackupStore {
  return {
    readLegacy: () => localStorage.getItem(LS_KEY),
    writeLegacy: raw => raw === null ? localStorage.removeItem(LS_KEY) : localStorage.setItem(LS_KEY, raw),
    readIndexedDb: async () => (await _idbGet(IDB_KEY)) ?? null,
    writeIndexedDb: value => _idbSet(IDB_KEY, value),
    saveBackup: backup => _idbSet(`${BACKUP_KEY}:${backup.id}`, backup),
    readBackup: async id => (await _idbGet<QueueBackup>(`${BACKUP_KEY}:${id}`)) ?? null,
  };
}

function downloadQueueExport(name: string, value: unknown): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

function backupItems(backup: QueueBackup): AudioQueueItem[] {
  const parse = (value: unknown): AudioQueueItem[] => value && typeof value === 'object'
    && Array.isArray((value as { items?: unknown }).items)
    ? (value as { items: AudioQueueItem[] }).items : [];
  const legacy = backup.legacyRaw ? JSON.parse(backup.legacyRaw) : null;
  const byId = new Map<string, AudioQueueItem>();
  for (const item of parse(legacy)) if (typeof item.id === 'string') byId.set(item.id, item);
  for (const item of parse(backup.indexedDb)) if (typeof item.id === 'string') byId.set(item.id, item);
  return [...byId.values()];
}

async function resolvedRequestForImport(item: AudioQueueItem): Promise<Record<string, unknown> | undefined> {
  if (item.mm3TakeOf) return undefined;
  if (item.jobId || item.status === 'succeeded' || item.status === 'failed') return undefined;
  if (item.generation.provider === 'create') return item.globalParams as Record<string, unknown>;
  if (item.generation.id && item.lyricsSetId) {
    return (await generateApi.previewIntent(writtenSongIntentFor(item))).request;
  }
  throw new Error(`Queue item ${item.id} has no render request; review it before migration`);
}

/** The written-song intent for a queue item: everything the server resolver
 *  cannot read itself (browser settings, the backend pick and dials, caption
 *  selections) captured now. */
function writtenSongIntentFor(item: AudioQueueItem): WrittenSongIntent {
  const backendId = typeof item.globalParams.backend === 'string'
    ? item.globalParams.backend : useBackendStore.getState().activeBackendId;
  const appSettings = JSON.parse(localStorage.getItem('ace-settings') || '{}') as Record<string, unknown>;
  const readSetting = (key: string, fallback: string) => {
    try { return JSON.parse(localStorage.getItem(key) || '') as string; }
    catch { return fallback; }
  };
  const datasetId = backendId === YUE2_BACKEND_ID ? readYue2DatasetForLyricsSet(item.lyricsSetId) : '';
  return {
    kind: 'written-song', engine: backendId, generationId: item.generation.id,
    lyricsSetId: item.lyricsSetId, sourceLyricsSetId: item.sourceLyricsSetId,
    artistName: item.artistName, params: item.globalParams,
    settings: {
      useLlmDuration: getUseLlmDuration(), useLmAdapter: useLmAdapterEnabled(),
      triggerUseFilename: appSettings.triggerUseFilename === true,
      triggerPlacement: appSettings.triggerPlacement === 'append' || appSettings.triggerPlacement === 'replace'
        ? appSettings.triggerPlacement : 'prepend',
      randomizeTimbreRef: localStorage.getItem('lireek-randomizeTimbreRef') === 'true',
      masterToAlbumTrack: masterToAlbumEnabled(),
      timeSignature: readSetting('hs-timeSignature', ''),
      vocalLanguage: readSetting('hs-vocalLanguage', 'en'), app: appSettings,
    },
    ...(backendId === MM3_BACKEND_ID ? { mm3Selection: readMm3CaptionSelection(item.generation.id) } : {}),
    ...(datasetId ? { yue2Selection: readYue2SongSelection(datasetId, item.generation.id,
      !!yue2CaptionAdapterPath(item.yue2Pick ?? yue2PickAtEnqueue(item.preset))) } : {}),
    ...(backendId === YUE2_BACKEND_ID ? {
      yue2Pick: item.yue2Pick ?? yue2PickAtEnqueue(item.preset),
      yue2Defaults: useBackendStore.getState().models[YUE2_BACKEND_ID]?.defaults ?? {},
    } : {}),
  };
}

/** Finished items are already reflected in the library; nothing needs to
 *  re-run them, so skip importing them into the live server queue. Items an
 *  earlier import already landed (a profile rolled back to the browser in
 *  v1.4.x) and whose server row still exists are left there, where Retry
 *  reaches them; sending them again would fail the whole import if a written
 *  song now resolves differently. A receipt is history, not proof: a row the
 *  user has since dismissed is gone, so that item is imported again. */
async function _resolveImportItems(backup: QueueBackup): Promise<ImportAudioQueue['items']> {
  const prior = await _idbGet<{ items?: { legacyId: string; itemId: string }[] }>(`${BACKUP_KEY}:receipt`);
  const live = prior?.items?.length ? new Set((await readAudioQueue()).items.map(i => i.id)) : new Set<string>();
  const landed = new Set(prior?.items?.filter(i => live.has(i.itemId)).map(i => i.legacyId));
  const items: ImportAudioQueue['items'] = [];
  for (const item of backupItems(backup).filter(i => i.status !== 'succeeded' && i.status !== 'failed' && !landed.has(i.id))) {
    items.push({
      legacyId: item.id,
      request: await resolvedRequestForImport(item),
      meta: { view: item, ...(item.generation.provider !== 'create' && item.generation.id && item.lyricsSetId
        ? { lyricLink: { generationId: item.generation.id, lyricsSetId: item.lyricsSetId } } : {}) },
      status: item.status, jobId: item.jobId,
      error: item.error,
    });
  }
  return items;
}

/** POSTs an already-resolved, already-persisted import payload. Only called
 *  once OWNER_KEY is `'migrating'` and `PENDING_IMPORT_KEY` holds this exact
 *  payload — on failure this deliberately does NOT revert OWNER_KEY. Once an
 *  import has started it may already have reached the server, so the queue
 *  stays fenced until a retry lands it. */
async function _submitImport(token: string, payload: ImportAudioQueue): Promise<Awaited<ReturnType<typeof audioQueueApi.importLegacy>>> {
  const receipt = await audioQueueApi.importLegacy(payload);
  await _idbSet(`${BACKUP_KEY}:receipt`, receipt);
  localStorage.setItem(OWNER_KEY, 'server');
  _lastToken = token;
  _startServerProjection(token);
  return receipt;
}

/** One-shot import of a v1.3.x browser queue. The browser originals are kept;
 *  the backup is saved and read back before anything is sent. */
export async function migrateAudioQueue(token: string, choice: 'hold' | 'resume' | 'discard' = 'hold', download = true): Promise<{
  backup: QueueBackup; receipt: Awaited<ReturnType<typeof audioQueueApi.importLegacy>>;
}> {
  await _idbReady;
  if (queueOwner() !== 'browser') throw new Error('The queue is already migrating or server-owned');
  const backup = await exportQueueBackup(migrationStore());
  if (download) downloadQueueExport(`audio-queue-backup-${backup.id}.json`, backup);
  localStorage.setItem(OWNER_KEY, 'migrating');
  const items = await _resolveImportItems(backup);
  const payload: ImportAudioQueue = { backupId: backup.id, choice, items };
  await _idbSet(PENDING_IMPORT_KEY, payload);
  const receipt = await _submitImport(token, payload);
  return { backup, receipt };
}

/** Recovers an interrupted migration (OWNER_KEY stuck at `'migrating'` —
 *  a reload or crash during `migrateAudioQueue`). Replays the exact payload
 *  persisted before the original POST, so it is idempotent even if that
 *  POST already reached the server: `intentQueue.importLegacy` caches by
 *  `backupId` (returns the same receipt instantly) and, failing that, by
 *  each item's `legacyId` (already-imported items come back as `existing`,
 *  never duplicated). If the payload was never persisted — the original
 *  attempt died before its first network call — nothing could have reached
 *  the server, so it is safe to rebuild fresh from the still-intact backup.
 *  On failure this also does NOT revert OWNER_KEY away from `'migrating'`;
 *  the caller (the banner's Retry button, or the silent first attempt on
 *  load) decides what to show. */
export async function retryQueueMigration(token: string): Promise<void> {
  await _idbReady;
  if (queueOwner() !== 'migrating') throw new Error('No interrupted migration to retry');
  let payload = await _idbGet<ImportAudioQueue>(PENDING_IMPORT_KEY);
  if (!payload) {
    const backup = await exportQueueBackup(migrationStore());
    payload = { backupId: backup.id, choice: 'resume', items: await _resolveImportItems(backup) };
    await _idbSet(PENDING_IMPORT_KEY, payload);
  }
  await _submitImport(token, payload);
}

let _serverProjectionBusy = false;
/** A refresh was asked for while one was in flight (an enqueue racing the
 *  poll). The running read may predate the change, so read once more after
 *  it rather than wait for the next tick, which an idle queue spaces out. */
let _serverProjectionAgain = false;
/** Applies are serialized: two snapshots can resolve close together, and the
 *  reducer compares each row against its previous status to deliver songs
 *  exactly once. */
let _applyChain: Promise<void> = Promise.resolve();

/** Song ids already handed to the library (the `song-created` event or an
 *  enqueue's onSongCreated) this page. The server projection and an
 *  enqueue's own waiter can both see one render finish; the library, and
 *  every effect hung off it, must hear about it once. */
const _deliveredSongIds = new Set<string>();
function _claimSongDelivery(songId: string): boolean {
  if (_deliveredSongIds.has(songId)) return false;
  _deliveredSongIds.add(songId);
  return true;
}

function serverItemTitle(server: AudioIntentItem, take: number): string {
  const view = server.meta?.view as AudioQueueItem | undefined;
  return server.songTitles?.[take] ?? view?.generation?.title ?? String(server.request.title ?? 'Untitled');
}

function projectServerItem(server: AudioIntentItem, previous?: AudioQueueItem): AudioQueueItem {
  const view = server.meta?.view as AudioQueueItem | undefined;
  const base = view?.generation ? view : {
    id: server.id,
    generation: {
      id: 0, profile_id: 0, provider: 'server', model: '',
      title: String(server.request.title ?? 'Untitled'),
      caption: String(server.request.caption ?? ''), lyrics: String(server.request.lyrics ?? ''),
      created_at: new Date(server.createdAt).toISOString(),
    },
    artistId: 0, artistName: '', preset: null, profileId: 0, lyricsSetId: 0,
    globalParams: server.request,
  } as AudioQueueItem;
  const status: AudioQueueStatus = server.status === 'succeeded' ? 'succeeded'
    : ['failed', 'cancelled', 'interrupted'].includes(server.status) ? 'failed'
      : ['held', 'pending'].includes(server.status) ? 'pending' : 'generating';
  const result = server.result as { audioUrls?: string[]; songIds?: string[]; masteredAudioUrls?: string[];
    noAdapterAudioUrls?: string[]; durations?: number[]; duration?: number } | null;
  const take = base.mm3Take ?? 0;
  return {
    // A server-built view (a Lyric Studio workflow render) has no browser row id.
    ...base, ...previous, id: base.id ?? server.id, serverQueueId: server.id, status,
    generation: { ...(previous?.generation ?? base.generation), title: serverItemTitle(server, take) },
    jobId: server.jobId ?? undefined,
    globalParams: base.globalParams ?? server.request,
    stage: server.waiting ?? (server.status === 'held' ? 'Waiting for resume' : status === 'succeeded' ? 'Complete!' : previous?.stage),
    error: server.error ?? undefined,
    progress: status === 'succeeded' ? 100 : previous?.progress,
    audioUrl: result?.audioUrls?.[take] ?? previous?.audioUrl ?? base.audioUrl,
    songId: result?.songIds?.[take] ?? previous?.songId ?? base.songId,
    masteredAudioUrl: result?.masteredAudioUrls?.[take] ?? previous?.masteredAudioUrl ?? base.masteredAudioUrl,
    noAdapterAudioUrl: result?.noAdapterAudioUrls?.[take] ?? previous?.noAdapterAudioUrl ?? base.noAdapterAudioUrl,
    audioDuration: result?.durations?.[take] ?? result?.duration ?? previous?.audioDuration ?? base.audioDuration,
    // Server-computed (intentQueue's follow() loop) so it is right regardless
    // of whether a client ever observed the job mid-run — see F19.
    elapsed: server.elapsedSeconds ?? previous?.elapsed ?? base.elapsed,
  };
}

/** The extra rows of a multi-song render (MM3 variations, a YuE2 or ACE
 *  batch). The server queue holds one intent per render, so the siblings
 *  _expandTakes made live only here: rebuilt from the last snapshot's rows on
 *  every refresh, or the next refresh drops them. A finished render fans out
 *  by its song count too, so a reload keeps one row per song. */
function projectTakeSiblings(item: AudioQueueItem, server: AudioIntentItem, previous: Map<string, AudioQueueItem>): AudioQueueItem[] {
  const result = server.result as { audioUrls?: string[]; songIds?: string[]; masteredAudioUrls?: string[];
    noAdapterAudioUrls?: string[]; durations?: number[] } | null;
  const songIds = server.status === 'succeeded' ? result?.songIds ?? [] : [];
  const count = Math.max(item.mm3TakeCount ?? 1, songIds.length);
  if (item.mm3TakeOf || count <= 1) return [];
  item.mm3Take = 0;
  item.mm3TakeCount = count;
  item.batchId ??= item.jobId ?? item.id;
  const siblings: AudioQueueItem[] = [];
  for (let t = 1; t < count; t++) {
    const id = `${item.id}:take${t}`;
    const s: AudioQueueItem = previous.get(id) ?? {
      ...item, id, mm3Take: t, mm3TakeSeed: undefined, audioUrl: undefined, songId: undefined,
      masteredAudioUrl: undefined, noAdapterAudioUrl: undefined, audioDuration: undefined, coverUrl: undefined,
    };
    Object.assign(s, { mm3TakeCount: count, mm3TakeOf: item.id, batchId: item.batchId, jobId: item.jobId,
      serverQueueId: item.serverQueueId, elapsed: item.elapsed });
    s.generation = { ...s.generation, title: serverItemTitle(server, t) };
    if (songIds[t]) {
      Object.assign(s, { status: 'succeeded', progress: 100, stage: 'Complete!', error: undefined, songId: songIds[t],
        audioUrl: result?.audioUrls?.[t] ?? s.audioUrl,
        masteredAudioUrl: result?.masteredAudioUrls?.[t] ?? s.masteredAudioUrl,
        noAdapterAudioUrl: result?.noAdapterAudioUrls?.[t] ?? s.noAdapterAudioUrl,
        audioDuration: result?.durations?.[t] ?? s.audioDuration });
    } else if (item.status === 'succeeded') {
      Object.assign(s, { status: 'failed', progress: 0, stage: 'No natural ending',
        error: 'Reached the length cap without ending; dropped by Require Natural Ending' });
    } else {
      // Live progress: _syncTakeSiblings refines this from the job poll below.
      Object.assign(s, { status: item.status, progress: item.progress, stage: item.stage, error: item.error });
    }
    siblings.push(s);
  }
  return siblings;
}

/** Reconcile one server snapshot into the local rows. The Query projection
 *  hook calls this for every successful read; manual refresh calls it too.
 *  Applies are serialized so two snapshots cannot both compare against the
 *  same pre-completion rows. */
export function applyServerItems(items: AudioIntentItem[]): Promise<void> {
  const run = _applyChain.then(() => applyServerItemsNow(items));
  _applyChain = run;
  return run;
}

async function applyServerItemsNow(items: AudioIntentItem[]): Promise<void> {
  try {
    const previous = new Map(_state.items.map(item => [item.id, item]));
    // A studio's own row (addManualQueueItem) lives only in this tab until its
    // render reaches the server queue: a Cover workflow queued behind another
    // has no intent until the first finishes. Keep it until a server item
    // carries its job or workflow, or every refresh wipes a queued row.
    const claimed = new Set(items.flatMap(server => [server.jobId, server.meta?.workflowJobId]).filter(Boolean));
    const local = _state.items.filter(item => _manualIds.has(item.id) && !item.serverQueueId
      && !claimed.has(item.jobId) && !claimed.has(item.workflowJobId));
    _state.items = items.flatMap(server => {
      const id = (server.meta?.view as AudioQueueItem | undefined)?.id ?? server.id;
      const item = projectServerItem(server, previous.get(id));
      return [item, ...projectTakeSiblings(item, server, previous)];
    }).concat(local, _failedRenders);
    _heldIds.clear();
    for (const server of items) if (server.status === 'held') {
      const item = _state.items.find(entry => entry.serverQueueId === server.id);
      if (item) _heldIds.add(item.id);
    }
    _state.awaitingResume = _heldIds.size;
    for (const server of items) {
      if (!server.jobId || !['submitted', 'submitting'].includes(server.status)) continue;
      const item = _state.items.find(entry => entry.serverQueueId === server.id);
      if (!item) continue;
      try {
        const job = await readGenerationStatus(server.jobId);
        item.progress = job.progress === undefined ? item.progress : Math.min(100, (job.progress > 1 ? job.progress / 100 : job.progress) * 100);
        item.stage = job.stage || item.stage;
        _captureMm3Stream(item, job);
        if (job.batch?.lead) { item.batchId = job.batch.lead; item.batchKind = job.batch.kind; }
        _syncTakeSiblings(item, job);
      } catch { /* server queue owns retry and interruption state */ }
    }
    for (const item of _state.items) {
      if (item.status !== 'succeeded' || previous.get(item.id)?.status === 'succeeded') continue;
      _state.completionCounter++;
      _maybeAutoAddToPlaylist(item);
      if (item.songId) _notifySongCreated(item.songId);
    }
    _emit();
  } catch (err) {
    console.warn('[AudioQueue] Server projection apply failed:', err);
  }
}

/** Read the server queue now and reconcile it. Reads are Query-cached, so a
 *  command that refreshes while the projection hook is already fetching joins
 *  that request. Cadence belongs to the hook; this is the event-driven path. */
export async function refreshServerProjection(): Promise<void> {
  if (queueOwner() !== 'server') return;
  if (_serverProjectionBusy) { _serverProjectionAgain = true; return; }
  _serverProjectionBusy = true;
  try {
    const { items } = await readAudioQueue();
    await applyServerItems(items);
  } catch (err) {
    console.warn('[AudioQueue] Server projection refresh failed:', err);
  } finally {
    _serverProjectionBusy = false;
    if (_serverProjectionAgain) { _serverProjectionAgain = false; void refreshServerProjection(); }
  }
}

/** Ownership changed or the app resumed: enable the projection hook and take
 *  one read now. The cadence itself is Query's refetchInterval. */
function _startServerProjection(token: string): void {
  _lastToken = token;
  _emit();
  void refreshServerProjection();
  void refreshFailedRenders();
}

// ── Lyric Studio renders that never reached the queue ───────────────────────
// A render can fail before the server queues any audio: the song or preset
// changed before its job ran, or it never resolved at all. No queue row
// carries that error, so these rows come from the render jobs themselves and
// survive a reload the same way. Dismissing one is this browser's choice and
// is kept here; the job is left as it is.

const DISMISSED_RENDERS_KEY = 'lireek-failed-renders-dismissed-v1';
let _failedRenders: AudioQueueItem[] = [];

function _dismissedRenders(): string[] {
  try { return JSON.parse(localStorage.getItem(DISMISSED_RENDERS_KEY) || '[]') as string[]; }
  catch { return []; }
}

function _dismissRenders(keys: string[]): void {
  if (!keys.length) return;
  // ponytail: newest 500 kept; the job list they filter holds 200 jobs at most.
  localStorage.setItem(DISMISSED_RENDERS_KEY, JSON.stringify([..._dismissedRenders(), ...keys].slice(-500)));
  const gone = (row: AudioQueueItem) => !!row.failedRender && keys.includes(row.failedRender.key);
  _failedRenders = _failedRenders.filter(row => !gone(row));
  _state.items = _state.items.filter(row => !gone(row));
  _emit();
}

type RenderJobItem = { type: string; intent?: WrittenSongIntent; view?: Partial<AudioQueueItem> };
type RenderJobResult = { results?: Array<{ index: number; status: string; error?: string; unqueued?: boolean }> } | null;

/** Failed rows for one render job: each item flagged `unqueued`, or every
 *  item of a job that failed or was interrupted before it queued anything. */
export function failedRenderRows(job: WorkflowJob, dismissed: Set<string>): AudioQueueItem[] {
  const items = (job.input as { items?: RenderJobItem[] }).items ?? [];
  const failures = job.status === 'succeeded'
    ? ((job.result as RenderJobResult)?.results ?? []).filter(r => r.status === 'error' && r.unqueued)
      .map(r => ({ index: r.index, error: r.error ?? 'Render failed' }))
    : (job.status === 'failed' || job.status === 'interrupted') && job.audioIntentIds.length === 0
      ? items.flatMap((item, index) => item.intent ? [{ index, error: job.error ?? `Render ${job.status}` }] : [])
      : [];
  return failures.flatMap(({ index, error }) => {
    const key = `${job.id}:${index}`;
    const item = items[index];
    if (dismissed.has(key) || !item) return [];
    const view = item.view ?? {};
    const generationId = item.intent?.generationId ?? 0;
    return [{
      artistId: 0, artistName: '', preset: null, profileId: 0, lyricsSetId: item.intent?.lyricsSetId ?? 0,
      ...view,
      id: `failed-render:${key}`,
      generation: {
        id: generationId, profile_id: 0, provider: 'lyric-studio', model: '', lyrics: '',
        created_at: new Date(job.createdAt).toISOString(), ...view.generation,
        title: view.generation?.title || `Written song ${generationId}`,
      },
      globalParams: item.intent?.params ?? {},
      status: 'failed', error, failedRender: { key, intent: item.intent },
    } as AudioQueueItem];
  });
}

/** Re-read the render jobs and rebuild their failed rows. Called on load and
 *  whenever a render job the UI follows finishes; not on every poll, since it
 *  reads whole jobs. */
export async function refreshFailedRenders(): Promise<void> {
  if (queueOwner() !== 'server') return;
  try {
    const { jobs } = await lyricWorkflowApi.listRenders();
    const dismissed = new Set(_dismissedRenders());
    _failedRenders = jobs.flatMap(job => failedRenderRows(job, dismissed));
    _state.items = _state.items.filter(row => !row.failedRender).concat(_failedRenders);
    _emit();
  } catch (err) {
    console.warn('[AudioQueue] Failed render read failed:', err);
  }
}

/** Submit failed renders again. The server captures each intent afresh, so a
 *  song or preset that changed renders as it is now. */
/** Rows whose resubmission is in flight; a second Retry click skips them. */
const _retryingRenders = new Set<string>();
/** POST /lireek/workflow-renders takes 1 to 200 intents. */
const RENDER_BATCH_MAX = 200;

function _retryFailedRenders(rows: AudioQueueItem[]): number {
  const todo = rows.filter(row => row.failedRender?.intent && !_retryingRenders.has(row.failedRender.key));
  for (let at = 0; at < todo.length; at += RENDER_BATCH_MAX) {
    const chunk = todo.slice(at, at + RENDER_BATCH_MAX);
    const keys = chunk.map(row => row.failedRender!.key);
    for (const key of keys) _retryingRenders.add(key);
    const submitted = lyricWorkflowApi.submitRenders(chunk.map(row => row.failedRender!.intent!));
    void submitted.catch(() => {}).finally(() => { for (const key of keys) _retryingRenders.delete(key); });
    void submitted
      // Dismissed only once the server holds the new job: a refused submit
      // queued nothing, so its rows stay for another Retry.
      .then(job => { _dismissRenders(keys); return lyricWorkflowApi.follow(job.id, { onEvent: () => {} }); })
      .catch(err => console.warn('[AudioQueue] Failed render retry did not finish:', err))
      .finally(() => { void refreshServerProjection(); void refreshFailedRenders(); });
  }
  return todo.length;
}


export function getAudioQueueOwner(): QueueOwner { return queueOwner(); }

/** Test-only: resets every module-level singleton this file keeps (state,
 *  the resume/owner-decision call-once guards, timers) so each test in the
 *  migration test suite starts from a cold load. Never called by the app —
 *  production has exactly one page lifetime per import. */
export function _queueItemsForTests(): AudioQueueItem[] { return _state.items; }

export function _resetAudioQueueForTests(items: AudioQueueItem[] = []): void {
  _state = { items, completionCounter: 0, awaitingResume: 0 };
  _manualIds.clear();
  _resumeCalled = false;
  _lastToken = null;
  _ownerDecision = null;
  _heldIds.clear();
  _serverProjectionBusy = false;
  _serverProjectionAgain = false;
  _applyChain = Promise.resolve();
  _deliveredSongIds.clear();
  _failedRenders = [];
  _retryingRenders.clear();
  queryClient.clear();
}

window.addEventListener('storage', event => {
  if (event.key === OWNER_KEY && event.newValue === 'server' && _lastToken) {
    _startServerProjection(_lastToken);
  }
});

/** Song ids already reconciled against their row this session — including the
 *  ones that came back with no master, so a track that genuinely has none is
 *  not re-fetched on every sweep. */
const _hydratedSongIds = new Set<string>();

/** Pull a finished queue item's missing fields off its song ROW.
 *
 *  A queue item only ever knew what the JOB result told it, and the job result
 *  reported one mastered URL and one duration for a whole render — nothing at
 *  all on MM3, which used to fold the master into audioUrls. So a finished card
 *  played WITHOUT the unmastered/mastered switch that the identical track gets
 *  from Recent Songs or the Library, and every take of an ensemble claimed the
 *  same length. The row has all of it, per song, and it is the same row those
 *  other two views read — so asking it is what makes the three agree. */
async function _hydrateItemFromSong(item: AudioQueueItem): Promise<boolean> {
  const songId = item.songId;
  if (!songId || _hydratedSongIds.has(songId)) return false;
  _hydratedSongIds.add(songId);
  try {
    const { song } = await readSong(songId);
    if (!song) return false;
    let dirty = false;
    const mastered = song.masteredAudioUrl || (song as any).mastered_audio_url;
    if (mastered && mastered !== item.masteredAudioUrl) { item.masteredAudioUrl = mastered; dirty = true; }
    const noAdapter = song.noAdapterAudioUrl || (song as any).noadapter_audio_url;
    if (noAdapter && noAdapter !== item.noAdapterAudioUrl) { item.noAdapterAudioUrl = noAdapter; dirty = true; }
    const raw = song.audioUrl || (song as any).audio_url;
    if (raw && raw !== item.audioUrl) { item.audioUrl = raw; dirty = true; }
    const dur = Number(song.duration);
    if (Number.isFinite(dur) && dur > 0 && dur !== item.audioDuration) { item.audioDuration = dur; dirty = true; }
    const coverUrl = song.coverUrl || (song as any).cover_url;
    if (coverUrl && !item.coverUrl) { item.coverUrl = coverUrl; dirty = true; }
    return dirty;
  } catch {
    // The row may be gone (deleted from the Library) — leave the item as it is.
    _hydratedSongIds.delete(songId);
    return false;
  }
}

/** Reconcile the finished items a restored queue came back with. Newest first
 *  and bounded, because this runs on every page load and the interesting items
 *  are the ones still on screen. */
async function _hydrateFinishedItems(limit = 80): Promise<void> {
  const todo = _state.items
    .filter(i => i.status === 'succeeded' && i.songId && !_hydratedSongIds.has(i.songId))
    .slice(-limit)
    .reverse();
  if (todo.length === 0) return;
  let dirty = false;
  // Four at a time: enough to finish a full queue quickly, few enough that a
  // page load does not fire eighty simultaneous reads at the server.
  const workers = Array.from({ length: Math.min(4, todo.length) }, async () => {
    for (;;) {
      const item = todo.pop();
      if (!item) return;
      if (await _hydrateItemFromSong(item)) dirty = true;
    }
  });
  await Promise.all(workers);
  if (dirty) _emit();
}

function _emit() {
  _state = { ..._state, items: [..._state.items] };
  _listeners.forEach(fn => fn());
}

// Queue items cache the mastered URL from the generation that produced them,
// and the queue is persisted, so a finished item outlives the page. A
// post-processing pass (or a revert) run later has to reach it, or the item's
// menu keeps deciding from a stale flag — offering a run the server refuses
// and hiding the removal that would fix it.
function _syncMastered(songId: string, masteredAudioUrl: string): void {
  let touched = false;
  for (const item of _state.items) {
    if (item.songId === songId || item.id === songId) {
      item.masteredAudioUrl = masteredAudioUrl;
      touched = true;
    }
  }
  if (touched) _emit();
}

window.addEventListener('song-postprocessed', (e: Event) => {
  const { songId, masteredAudioUrl } = (e as CustomEvent).detail || {};
  if (songId && masteredAudioUrl) _syncMastered(songId, masteredAudioUrl);
});

window.addEventListener('song-postprocess-reverted', (e: Event) => {
  const { songId } = (e as CustomEvent).detail || {};
  if (songId) _syncMastered(songId, '');
});

// Apply menu renames immediately; server snapshots also read current song titles.
window.addEventListener('song-renamed', (e: Event) => {
  const { songId, title } = (e as CustomEvent).detail || {};
  if (!songId || !title) return;
  let touched = false;
  _state.items = _state.items.map(item => {
    if (item.songId !== songId && item.id !== songId) return item;
    touched = true;
    return { ...item, generation: { ...item.generation, title } };
  });
  if (touched) _emit();
});

function _getSnapshot(): AudioGenQueueState { return _state; }
function _subscribe(listener: () => void): () => void {
  _listeners.add(listener);
  return () => _listeners.delete(listener);
}

let _nextId = 0;
function _genId(): string { return `aq-${Date.now()}-${_nextId++}`; }

// ── Resume tracking ──────────────────────────────────────────────────────────

let _resumeCalled = false;

/** Rows whose server item is 'held', waiting on the user's Resume or Discard.
 *
 *  Kept out of AudioQueueStatus deliberately: a sixth status would have to be
 *  handled by every render site that switches on it, and a missed one shows a
 *  blank row. These items stay 'pending'. */
const _heldIds = new Set<string>();

/** True while restored items are waiting on the user's answer. */
export function isHeldQueueItem(id: string): boolean { return _heldIds.has(id); }

/** Last auth token the queue was handed. Lets actions that aren't wired to the
 *  auth context — the Retry button — reach the server queue. */
let _lastToken: string | null = null;

// ── Public API ───────────────────────────────────────────────────────────────

type LyricQueueOpts = { artistId: number; artistName: string; artistImageUrl?: string; profileId: number; lyricsSetId: number; sourceLyricsSetId?: number };

/** Capture a written song as an intent the server resolves itself
 *  (resolveWrittenSongIntent), for POST /api/lireek/workflow-renders: the
 *  browser settings, backend pick and caption selections it cannot read. */
export async function captureWrittenSongIntent(
  gen: Generation, opts: LyricQueueOpts, globalParams: Partial<GenerationParams>,
): Promise<WrittenSongIntent> {
  return withIntentTimeout(writtenSongIntentFor(await newLyricQueueItem(gen, opts, globalParams)));
}

async function newLyricQueueItem(
  gen: Generation, opts: LyricQueueOpts, globalParams: Partial<GenerationParams>,
): Promise<AudioQueueItem> {
  let preset: AlbumPreset | null = null;
  try {
    const res = await lireekApi.getPreset(opts.lyricsSetId);
    preset = res.preset;
  } catch { /* no preset configured */ }
  // The picker's dials ride in the pick; fetch them if the picker never has.
  if (!useBackendStore.getState().models[YUE2_BACKEND_ID]) {
    await useBackendStore.getState().fetchModels(YUE2_BACKEND_ID);
  }

  return {
    id: _genId(),
    generation: gen,
    artistId: opts.artistId,
    artistName: opts.artistName,
    artistImageUrl: opts.artistImageUrl,
    preset,
    profileId: opts.profileId,
    lyricsSetId: opts.lyricsSetId,
    sourceLyricsSetId: opts.sourceLyricsSetId,
    globalParams,
    yue2Pick: yue2PickAtEnqueue(preset),
    status: 'pending',
  };
}

export function removeFromAudioQueue(id: string): void {
  const failed = _state.items.find(item => item.id === id)?.failedRender;
  if (failed) { _dismissRenders([failed.key]); return; }
  if (queueOwner() === 'server') {
    const serverId = _state.items.find(item => item.id === id)?.serverQueueId;
    if (serverId && _lastToken) {
      void audioQueueApi.cancel(serverId)
        .catch(() => null)
        .then(() => audioQueueApi.dismiss(serverId))
        .then(() => refreshServerProjection());
    }
    return;
  }
  _state.items = _state.items.filter(i => i.id !== id);
  if (_heldIds.delete(id)) _state.awaitingResume = _heldIds.size;
  _emit();
}

/** Force-dismiss an active/generating item (user clicked X).
 *  Also calls the server cancel API to stop the generation and C++ engine. */
export function forceFailQueueItem(id: string): void {
  const item = _state.items.find(i => i.id === id);
  // A workflow-backed manual row (Insta-Gen): cancelling the workflow job also
  // cancels its audio intents, whichever side owns the queue.
  const workflowJobId = item?.workflowJobId;
  if (queueOwner() === 'server' && !workflowJobId) {
    const serverId = item?.serverQueueId;
    if (serverId && _lastToken) void audioQueueApi.cancel(serverId).then(() => refreshServerProjection());
    return;
  }
  if (item && (item.status === 'generating' || item.status === 'loading-adapter')) {
    if (workflowJobId) {
      if (_lastToken) workflowApi.cancel(workflowJobId).catch(() => {});
    } else if (item.jobId) {
      // Cancel on the server → triggers abort controller → cancels C++ engine job
      generateApi.cancel(item.jobId).catch(() => {});
    }
    item.status = 'failed';
    item.error = 'Cancelled by user';
    item.stage = undefined;
    item.progress = undefined;
    _emit();
  }
}

export function clearFinishedFromAudioQueue(): void {
  _dismissRenders(_state.items.flatMap(item => item.failedRender ? [item.failedRender.key] : []));
  if (queueOwner() === 'server') {
    // Take rows share their render's server item; it goes once, with its owner
    // row, and the projection drops the takes with it.
    if (_lastToken) void Promise.all(_state.items.filter(item => item.serverQueueId && !item.mm3TakeOf
      && (item.status === 'succeeded' || item.status === 'failed'))
      .map(item => audioQueueApi.dismiss(item.serverQueueId!)))
      .then(() => refreshServerProjection());
    return;
  }
  _state.items = _state.items.filter(i => i.status !== 'succeeded' && i.status !== 'failed');
  _emit();
}

/** Put every failed server item back in the queue, and submit again every
 *  Lyric Studio render that failed before reaching it. */
export function retryFailedInAudioQueue(): number {
  if (queueOwner() !== 'server' || !_lastToken) return 0;
  // A failed take row (a dropped candidate) belongs to its owner's render.
  const failed = _state.items.filter(item => item.status === 'failed' && item.serverQueueId && !item.mm3TakeOf);
  const renders = _state.items.filter(item => item.failedRender?.intent);
  void Promise.all(failed.map(item => audioQueueApi.retry(item.serverQueueId!)))
    .then(() => refreshServerProjection());
  return failed.length + _retryFailedRenders(renders);
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
  _manualIds.add(id);
  _state.items.push(item);
  _emit();
  return id;
}

/** Update progress/stage of a manually-added queue item. */
export function updateManualQueueItem(id: string, update: {
  title?: string;
  jobId?: string;
  /** Workflow job behind the row; X cancels it through workflowApi. */
  workflowJob?: { id: string; token: string };
  progress?: number;
  stage?: string;
  elapsed?: number;
  status?: AudioQueueStatus;
}): void {
  const item = _state.items.find(i => i.id === id);
  if (!item) return;
  if (update.title !== undefined) item.generation.title = update.title;
  if (update.jobId !== undefined) item.jobId = update.jobId;
  if (update.workflowJob) {
    const following = item.workflowJobId === update.workflowJob.id;
    item.workflowJobId = update.workflowJob.id;
    _lastToken = update.workflowJob.token;
    if (!following) _followWorkflowRender(id, update.workflowJob.id);
  }
  if (update.progress !== undefined) item.progress = update.progress;
  if (update.stage !== undefined) item.stage = update.stage;
  if (update.elapsed !== undefined) item.elapsed = update.elapsed;
  if (update.status !== undefined) item.status = update.status;
  _emit();
}

/** A studio's workflow-backed row (Cover, Auto-Gen) hears only its workflow's
 *  own status, and in a browser-owned queue nothing else polls the render that
 *  workflow queued, so the row showed "Generating…" with no stage or step.
 *  Follow the workflow's newest audio item to its engine job, as the
 *  server-owned projection does (refreshServerProjection), until the row ends. */
function _followWorkflowRender(id: string, workflowJobId: string): void {
  const ended = (status: AudioQueueStatus) => status === 'succeeded' || status === 'failed';
  const tick = async (): Promise<boolean> => {
    const item = _state.items.find(i => i.id === id);
    if (!item || item.workflowJobId !== workflowJobId || ended(item.status)) return false;
    if (queueOwner() === 'server') return true; // the projection fills it in
    try {
      const { job } = await readWorkflowJob(workflowJobId);
      const intentId = job.audioIntentIds[job.audioIntentIds.length - 1];
      const intentItem = intentId ? (await readAudioQueue()).items.find(i => i.id === intentId) : undefined;
      // Real generation time is the server's (intentQueue's follow() loop),
      // same source projectServerItem reads — never this loop's own poll
      // cadence, which can miss a fast job entirely (F19).
      if (intentItem?.elapsedSeconds != null) item.elapsed = intentItem.elapsedSeconds;
      const jobId = intentItem?.jobId;
      if (!jobId) return true;
      const status = await readGenerationStatus(jobId);
      if (ended(item.status)) return false;
      if (status.progress !== undefined) item.progress = Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100));
      if (status.stage) item.stage = status.stage;
      _emit();
    } catch { /* the studio's own workflow poll reports failures */ }
    return true;
  };
  const loop = async () => { if (await tick()) setTimeout(loop, 1500); };
  void loop();
}

/** Mark a manually-added queue item as succeeded with audio results. */
export function completeManualQueueItem(id: string, result: {
  audioUrl: string;
  songId?: string;
  masteredAudioUrl?: string;
  noAdapterAudioUrl?: string;
  audioDuration?: number;
  /** The intent queue's own elapsed time for this render (finished.elapsedSeconds
   *  from ctx.audio.wait) — authoritative, unlike this loop's own poll cadence. */
  elapsedSeconds?: number | null;
}): void {
  const item = _state.items.find(i => i.id === id);
  if (!item) return;
  item.status = 'succeeded';
  item.audioUrl = result.audioUrl;
  if (result.songId) item.songId = result.songId;
  if (result.masteredAudioUrl) item.masteredAudioUrl = result.masteredAudioUrl;
  if (result.noAdapterAudioUrl) item.noAdapterAudioUrl = result.noAdapterAudioUrl;
  if (result.audioDuration) item.audioDuration = result.audioDuration;
  if (result.elapsedSeconds != null) item.elapsed = result.elapsedSeconds;
  item.progress = 100;
  item.stage = 'Complete!';
  _state.completionCounter++;
  _emit();
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
  _emit();
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
  _emit();
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
    noAdapterAudioUrl: item.noAdapterAudioUrl || '',
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
    const { song } = await readSong(songId);
    if (song) {
      if (_claimSongDelivery(songId)) window.dispatchEvent(new CustomEvent('song-created', { detail: { song } }));
      // This IS the row the restore sweep would have asked for.
      _hydratedSongIds.add(songId);
      const item = _state.items.find(i => i.songId === songId);
      if (item) {
        let dirty = false;
        // Backfill coverUrl into the queue item so the playback track uses cover art
        const coverUrl = song.coverUrl || song.cover_url;
        if (coverUrl && !item.coverUrl) { item.coverUrl = coverUrl; dirty = true; }
        // The song ROW is the authority on which takes exist: it stores the raw
        // render and its master in separate columns, which is what the playbar's
        // unmastered/mastered switch keys off. Backfilling from it means a queue
        // card offers the same switch as every other place the track is played,
        // whatever the job result happened to report.
        const mastered = song.masteredAudioUrl || (song as any).mastered_audio_url;
        if (mastered && !item.masteredAudioUrl) { item.masteredAudioUrl = mastered; dirty = true; }
        const noAdapter = song.noAdapterAudioUrl || (song as any).noadapter_audio_url;
        if (noAdapter && !item.noAdapterAudioUrl) { item.noAdapterAudioUrl = noAdapter; dirty = true; }
        // The raw render too: a job result that reported the master as its audio
        // would otherwise leave both sides of the switch on the same file.
        const raw = song.audioUrl || (song as any).audio_url;
        if (raw && raw !== item.audioUrl) { item.audioUrl = raw; dirty = true; }
        // Per-take length, from the row that measured it off the WAV header.
        const dur = Number(song.duration);
        if (Number.isFinite(dur) && dur > 0 && dur !== item.audioDuration) {
          item.audioDuration = dur;
          dirty = true;
        }
        if (dirty) _emit();
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
    const blob = await fetchAudioBlob(url);
    if (!blob) return;
    blobUrl = URL.createObjectURL(blob);

    const audio = new Audio();
    audio.preload = 'metadata';

    audio.onloadedmetadata = () => {
      const dur = audio.duration;
      if (dur && isFinite(dur) && dur > 0) {
        const item = _state.items.find(i => i.id === itemId);
        if (item) {
          item.audioDuration = Math.round(dur);
          _emit();
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

/** A Create render's YuE2 pick, captured now (#319): the picker's defaults
 *  with any save still in flight, so a Generate clicked mid-save renders what
 *  the picker shows. A one-shot Generate (it always carries an intent) takes
 *  the picker over a pick its params inherited by reusing a song, since the
 *  picker is Create's only adapter control. A build save keeps its session's
 *  own dials and fills the rest from the picker. */
export function withCreateYue2Pick(request: Record<string, unknown>, oneShot: boolean): Record<string, unknown> {
  const backend = typeof request.backend === 'string' ? request.backend : useBackendStore.getState().activeBackendId;
  if (backend !== YUE2_BACKEND_ID) return request;
  const own = request.yue2Pick as Record<string, string | number> | undefined;
  const live = yue2PickAtEnqueue(null);
  return { ...request, yue2Pick: oneShot || !own ? live : { ...live, ...own } };
}

/**
 * Enqueue a Create render on the server queue and call onSongCreated for each
 * song it produces. The server resolves the request: `intent` when the caller
 * has one (CreatePanel), else a plain Create intent over `params`, which also
 * supply the row's title, caption and lyrics. `oneShot` picks the YuE2 rule in
 * withCreateYue2Pick; a build save passes false to keep its session's dials.
 */
export async function enqueueSimpleGen(
  params: Record<string, any>,
  token: string,
  onSongCreated?: (song: any) => void,
  intent?: CreateIntent,
  oneShot = !!intent,
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
  await ensureQueueOwnerDecided(token);
  if (queueOwner() !== 'server') throw new Error('Audio queue migration is in progress');
  const preview = await generateApi.previewIntent(intent ?? { kind: 'create', params });
  const request = withCreateYue2Pick(preview.request as Record<string, unknown>, oneShot);
  await audioQueueApi.enqueue(id, request, { view: item });
  _lastToken = token;
  await refreshServerProjection();
  if (onSongCreated) {
    void (async () => {
      for (;;) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        // Rides the projection's reads: a list younger than one poll is reused.
        const server = (await readAudioQueue(ACTIVE_POLL_MS)).items.find(entry => entry.idempotencyKey === id);
        if (!server || ['failed', 'cancelled', 'interrupted'].includes(server.status)) return;
        if (server.status !== 'succeeded') continue;
        const ids = (server.result?.songIds as string[] | undefined) ?? [];
        for (const songId of ids) {
          // Claim only once the row is in hand: a failed read leaves the
          // song for the projection's own delivery.
          try {
            const { song } = await readSong(songId);
            if (_claimSongDelivery(songId)) onSongCreated(song);
          } catch { /* projection still updates library */ }
        }
        return;
      }
    })();
  }
}

/** One-time, silent owner decision on load. No pending items goes straight to
 *  server-owned — nothing to lose. Pending items (a v1.3.x browser queue, or a
 *  profile rolled back to the browser in v1.4.x) are imported via the `resume`
 *  choice, no download and no held-for-review step. A stuck `'migrating'`
 *  retries via `retryQueueMigration`. Either way, a failure leaves the banner
 *  as the only sign anything happened. Idempotent and safe to call from both
 *  `resumeQueue()` and the banner. */
let _ownerDecision: Promise<void> | null = null;
export function ensureQueueOwnerDecided(token: string): Promise<void> {
  if (!_ownerDecision) {
    _ownerDecision = (async () => {
      await _idbReady;
      const hasPending = _state.items.some(item => item.status !== 'succeeded' && item.status !== 'failed');
      const action = decideQueueOwnerAction(localStorage.getItem(OWNER_KEY), hasPending);
      if (action === 'none') return;
      if (action === 'set-server') { localStorage.setItem(OWNER_KEY, 'server'); _emit(); return; }
      if (action === 'retry') {
        try { await retryQueueMigration(token); }
        catch (e) { console.error('[AudioGenQueue] migration retry failed, staying fenced:', e); }
        return;
      }
      try { await migrateAudioQueue(token, 'resume', false); }
      catch (e) { console.error('[AudioGenQueue] automatic migration failed:', e); }
    })();
  }
  return _ownerDecision;
}

export async function resumeQueue(token: string): Promise<void> {
  if (_resumeCalled) return;
  _resumeCalled = true;
  _lastToken = token;
  await ensureQueueOwnerDecided(token);
  if (queueOwner() === 'server') _startServerProjection(token);
}

/** User answered the restored-queue prompt with Resume: release the server's
 *  held items. */
export function resumeRestoredQueue(token?: string): void {
  if (_state.awaitingResume === 0 || queueOwner() !== 'server') return;
  const t = token ?? _lastToken;
  if (!t) return;
  const ids = _state.items.filter(item => _heldIds.has(item.id) && item.serverQueueId).map(item => item.serverQueueId!);
  void audioQueueApi.resumeHeld(ids).then(() => refreshServerProjection());
}

/** User answered the restored-queue prompt with Discard. Drop only the items
 *  that were held. Anything added since is untouched. */
export function discardRestoredQueue(): void {
  if (_state.awaitingResume === 0 || queueOwner() !== 'server' || !_lastToken) return;
  const ids = _state.items.filter(item => _heldIds.has(item.id) && item.serverQueueId).map(item => item.serverQueueId!);
  void Promise.all(ids.map(id => audioQueueApi.cancel(id).then(() => audioQueueApi.dismiss(id))))
    .then(() => refreshServerProjection());
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

/** The current queue owner, reactive to migration/rollback transitions. The
 *  projection hook uses it to enable itself only while the server owns the
 *  queue. */
export function useAudioQueueOwner(): 'browser' | 'migrating' | 'server' {
  return useSyncExternalStore(_subscribe, queueOwner, queueOwner);
}

/** True while a row still needs the server projection followed at the active
 *  cadence. Includes studio-only rows that have no server item yet. */
export function hasActiveQueueWork(): boolean {
  return _state.items.some(item => item.status === 'pending' || item.status === 'loading-adapter' || item.status === 'generating');
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
