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

import { useSyncExternalStore, useEffect } from 'react';
import { lireekApi } from '../services/lireekApi';
import { generateApi } from '../services/api';
import { writePersistedState } from '../hooks/usePersistedState';
import type { Generation, AlbumPreset } from '../services/lireekApi';
import type { GenerationParams } from '../types';
import { resolveDuration } from '../utils/estimateDuration';

// ── Types ────────────────────────────────────────────────────────────────────

export type AudioQueueStatus = 'pending' | 'loading-adapter' | 'generating' | 'succeeded' | 'failed';

export interface AudioQueueItem {
  id: string;
  generation: Generation;
  artistId: number;
  artistName: string;
  artistImageUrl?: string;
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

// ── Persistence ──────────────────────────────────────────────────────────────

const STORAGE_KEY = 'lireek-audio-gen-queue';

function _persist(): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      items: _state.items,
      completionCounter: _state.completionCounter,
    }));
  } catch { /* quota exceeded etc */ }
}

function _restore(): AudioGenQueueState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { items: [], completionCounter: 0 };
    const parsed = JSON.parse(raw);
    const items: AudioQueueItem[] = (parsed.items || []).map((item: AudioQueueItem) => {
      if (item.status === 'loading-adapter' ||
          (item.status === 'generating' && !item.jobId)) {
        return { ...item, status: 'pending' as AudioQueueStatus, stage: undefined, progress: undefined };
      }
      return item;
    });
    return { items, completionCounter: parsed.completionCounter || 0 };
  } catch {
    return { items: [], completionCounter: 0 };
  }
}

// ── Module-level singleton ───────────────────────────────────────────────────

let _state: AudioGenQueueState = _restore();
const _listeners = new Set<() => void>();

function _emit() {
  _state = { ..._state, items: [..._state.items] };
  _persist();
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

const _resumedJobIds = new Set<string>();
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
  _emit();
  _processQueue(token);
}

export function removeFromAudioQueue(id: string): void {
  _state.items = _state.items.filter(i => i.id !== id);
  _emit();
}

/** Force-dismiss an active/generating item (user clicked X) */
export function forceFailQueueItem(id: string): void {
  const item = _state.items.find(i => i.id === id);
  if (item && (item.status === 'generating' || item.status === 'loading-adapter')) {
    item.status = 'failed';
    item.error = 'Dismissed by user';
    item.stage = undefined;
    item.progress = undefined;
    _emit();
  }
}

export function clearFinishedFromAudioQueue(): void {
  _state.items = _state.items.filter(i => i.status !== 'succeeded' && i.status !== 'failed');
  _emit();
}

export function resumeQueue(token: string): void {
  if (_resumeCalled) return;
  _resumeCalled = true;

  let didFix = false;
  for (const item of _state.items) {
    // Reset items that were mid-processing but have no job ID (interrupted before submission)
    if ((item.status === 'generating' || item.status === 'loading-adapter') && !item.jobId) {
      item.status = 'pending';
      item.stage = undefined;
      item.progress = undefined;
      didFix = true;
    }
  }
  if (didFix) _emit();

  const hasPending = _state.items.some(i => i.status === 'pending');
  const inFlight = _state.items.filter(i => i.status === 'generating' && i.jobId);

  const resumePromises: Promise<void>[] = [];
  for (const item of inFlight) {
    if (_resumedJobIds.has(item.jobId!)) continue;
    _resumedJobIds.add(item.jobId!);
    resumePromises.push(_resumePolling(item, token));
  }

  if (hasPending) {
    if (resumePromises.length > 0) {
      Promise.all(resumePromises).then(() => _processQueue(token));
    } else {
      _processQueue(token);
    }
  }
}

// ── Queue runner ─────────────────────────────────────────────────────────────

let _running = false;

async function _processQueue(token: string): Promise<void> {
  if (_running) return;
  _running = true;

  while (true) {
    const pending = _state.items.filter(i => i.status === 'pending');
    if (pending.length === 0) break;

    // Artist batching: prefer items with same adapter as last completed
    const next = pending[0];

    try {
      await _executeItem(next, token);
      next.status = 'succeeded';
      _state.completionCounter++;
    } catch (err) {
      next.status = 'failed';
      next.error = (err as Error).message;
    }
    _emit();
  }

  _running = false;
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
    }
  } catch { /* ignore parse errors */ }

  // 3) Adapter override from album preset (path from preset, scale/groups from globalParams)
  if (preset?.adapter_path) {
    item.status = 'loading-adapter';
    item.stage = `Preparing adapter for ${item.artistName}…`;
    _emit();

    // Update the top bar to reflect the adapter being used
    writePersistedState('hs-adapter', preset.adapter_path);

    // Override adapter path from preset; scale, group scales, mode, and
    // trigger word settings are already correct from globalParams.
    params.loraPath = preset.adapter_path;

    // Re-derive trigger word from the PRESET adapter filename (globalParams
    // has the trigger word for the GLOBAL adapter, which may differ).
    const settingsRaw = localStorage.getItem('ace-settings');
    const triggerSettings = settingsRaw ? JSON.parse(settingsRaw) : {};
    const useFilename = triggerSettings.triggerUseFilename === true;
    const placement = (triggerSettings.triggerPlacement as 'prepend' | 'append' | 'replace') || 'prepend';
    if (useFilename) {
      const fileName = preset.adapter_path.replace(/\\/g, '/').split('/').pop() || '';
      const triggerWord = fileName.replace(/\.safetensors$/i, '');
      if (triggerWord) {
        params.triggerWord = triggerWord;
        params.triggerPlacement = placement;
      }
    } else {
      // Preset adapter but no filename trigger — clear any global trigger word
      delete params.triggerWord;
      delete params.triggerPlacement;
    }
  }

  // 4) Mastering reference from album preset (does NOT force-enable — respects global toggle)
  if (preset?.reference_track_path) {
    // Update the top bar to reflect the mastering reference
    writePersistedState('hs-masteringReference', preset.reference_track_path);
    writePersistedState('hs-timbreReference', true);

    params.masteringReference = preset.reference_track_path;
    params.timbreReference = true;
  }

  // 5) Submit generation
  params.taskType = 'text2music';
  params.source = 'lyric-studio';
  item.status = 'generating';
  item.stage = 'Submitting to audio engine…';
  _emit();

  const res = await generateApi.submit(params as any, token);
  const jobId = res.jobId;
  item.jobId = jobId;
  _emit(); // persist jobId immediately

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
  const startTime = item.elapsed ? Date.now() - item.elapsed * 1000 : Date.now();
  _emit();

  while (true) {
    await new Promise(r => setTimeout(r, 2500));
    try {
      const status = await generateApi.status(jobId);
      item.progress = status.progress !== undefined
        ? Math.min(100, Math.max(0, (status.progress > 1 ? status.progress / 100 : status.progress) * 100))
        : undefined;
      item.stage = status.stage || 'Generating…';
      item.elapsed = Math.round((Date.now() - startTime) / 1000);
      _emit();

      if (status.status === 'succeeded') {
        const audioUrl = status.result?.audioUrls?.[0];
        const songId = status.result?.songIds?.[0];
        const masteredUrl = status.result?.masteredAudioUrl;
        if (audioUrl) {
          item.audioUrl = audioUrl;
          if (songId) item.songId = songId;
          if (masteredUrl) item.masteredAudioUrl = masteredUrl;
          if (status.result?.duration) item.audioDuration = status.result.duration;
          _emit();
        }
        // Resolve audio generation in Lireek DB
        if (audioUrl && jobId) {
          try {
            await lireekApi.resolveAudioGeneration(jobId, audioUrl);
          } catch { /* non-fatal */ }
        }
        return;
      }
      if (status.status === 'failed') throw new Error(status.error || 'Generation failed');
    } catch (e) {
      if ((e as Error).message.includes('failed') && !(e as Error).message.includes('fetch')) {
        throw e;
      }
    }
  }
}

async function _resumePolling(item: AudioQueueItem, token: string): Promise<void> {
  try {
    await _pollUntilDone(item, token);
    item.status = 'succeeded';
    _state.completionCounter++;
  } catch (err) {
    item.status = 'failed';
    item.error = (err as Error).message;
  }
  _emit();
}

// ── React hook ───────────────────────────────────────────────────────────────

export function useAudioGenQueue(token?: string): AudioGenQueueState {
  useEffect(() => {
    if (token && _state.items.length > 0) {
      resumeQueue(token);
    }
  }, [token]);

  return useSyncExternalStore(_subscribe, _getSnapshot, _getSnapshot);
}
