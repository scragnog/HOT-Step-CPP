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
import { generateApi, songApi } from '../services/api';
import type { Generation, AlbumPreset } from '../services/lireekApi';

// ── Types ────────────────────────────────────────────────────────────────────

export type AudioQueueStatus = 'pending' | 'loading-adapter' | 'generating' | 'succeeded' | 'failed';

export interface AudioQueueItem {
  id: string;
  generation: Generation;
  artistId: number;
  artistName: string;
  preset: AlbumPreset | null;
  profileId: number;
  lyricsSetId: number;
  status: AudioQueueStatus;
  jobId?: string;
  progress?: number;
  stage?: string;
  elapsed?: number;
  error?: string;
  audioUrl?: string;
}

export interface AudioGenQueueState {
  items: AudioQueueItem[];
  completionCounter: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function readPersisted(key: string): any {
  try {
    const raw = localStorage.getItem(key);
    return raw !== null ? JSON.parse(raw) : undefined;
  } catch { return undefined; }
}

function mergeCreatePanelSettings(params: Record<string, any>): void {
  // Read ALL generation params from Create panel's hs-* localStorage keys.
  // Only content fields (lyrics/caption/bpm/key/duration) and adapter/reference
  // are excluded — those come from the album preset + written song.
  const map: [string, string][] = [
    // DiT settings
    ['hs-inferenceSteps', 'inferenceSteps'],
    ['hs-guidanceScale', 'guidanceScale'],
    ['hs-shift', 'shift'],
    ['hs-inferMethod', 'inferMethod'],
    ['hs-scheduler', 'scheduler'],
    ['hs-guidanceMode', 'guidanceMode'],
    // Seed
    ['hs-seed', 'seed'],
    ['hs-randomSeed', 'randomSeed'],
    ['hs-batchSize', 'batchSize'],
    // LM
    ['hs-skipLm', 'skipLm'],
    ['hs-useCotCaption', 'useCotCaption'],
    ['hs-lmTemperature', 'lmTemperature'],
    ['hs-lmCfgScale', 'lmCfgScale'],
    ['hs-lmTopK', 'lmTopK'],
    ['hs-lmTopP', 'lmTopP'],
    ['hs-lmNegativePrompt', 'lmNegativePrompt'],
    // Models
    ['hs-ditModel', 'ditModel'],
    ['hs-lmModel', 'lmModel'],
    ['hs-vaeModel', 'vaeModel'],
    // Solver sub-params
    ['hs-storkSubsteps', 'storkSubsteps'],
    ['hs-beatStability', 'beatStability'],
    ['hs-frequencyDamping', 'frequencyDamping'],
    ['hs-temporalSmoothing', 'temporalSmoothing'],
    // Guidance sub-params
    ['hs-apgMomentum', 'apgMomentum'],
    ['hs-apgNormThreshold', 'apgNormThreshold'],
    // Language
    ['hs-vocalLanguage', 'vocalLanguage'],
    // Adapter mode (runtime vs merge — engine setting, not preset)
    ['hs-adapterMode', 'adapterMode'],
  ];
  for (const [storageKey, paramKey] of map) {
    const val = readPersisted(storageKey);
    if (val !== undefined && val !== null) {
      params[paramKey] = val;
    }
  }
}

function applyTriggerWord(params: Record<string, any>, adapterPath: string): void {
  const useFilename = localStorage.getItem('ace-globalTriggerUseFilename') === 'true';
  const placement = (localStorage.getItem('ace-globalTriggerPlacement') as 'prepend' | 'append' | 'replace') || 'prepend';
  if (!useFilename) return;
  const fileName = adapterPath.replace(/\\/g, '/').split('/').pop() || '';
  const triggerWord = fileName.replace(/\.safetensors$/i, '');
  if (!triggerWord) return;
  const current = ((params.caption as string) || '').trim();
  if (current.toLowerCase().includes(triggerWord.toLowerCase())) return;
  if (placement === 'replace') { params.caption = triggerWord; }
  else if (placement === 'append') { params.caption = current ? `${current}, ${triggerWord}` : triggerWord; }
  else { params.caption = current ? `${triggerWord}, ${current}` : triggerWord; }
}

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
  opts: { artistId: number; artistName: string; profileId: number; lyricsSetId: number },
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
    preset,
    profileId: opts.profileId,
    lyricsSetId: opts.lyricsSetId,
    status: 'pending',
  };

  _state.items.push(item);
  _emit();
  _processQueue(token);
}

export function removeFromAudioQueue(id: string): void {
  _state.items = _state.items.filter(i => i.id !== id || i.status !== 'pending');
  _emit();
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
    if (item.status === 'generating' && !item.jobId) {
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

  // 1) Build base params
  const params: Record<string, any> = {
    lyrics: gen.lyrics || '',
    caption: gen.caption || '',
    title: gen.title || '',
    instrumental: false,
    duration: gen.duration || 180,
  };
  if (gen.bpm) params.bpm = gen.bpm;
  if (gen.key) params.keyScale = gen.key;

  // 2) Merge persisted CreatePanel settings
  mergeCreatePanelSettings(params);

  // 3) Adapter — in cpp engine, we pass directly to generate
  if (preset?.adapter_path) {
    item.status = 'loading-adapter';
    item.stage = `Preparing adapter for ${item.artistName}…`;
    _emit();

    params.loraPath = preset.adapter_path;
    params.loraScale = preset.adapter_scale ?? 1.0;
    if (preset.adapter_group_scales) {
      params.adapterGroupScales = preset.adapter_group_scales;
    }

    // Trigger word
    applyTriggerWord(params, preset.adapter_path);
  }

  // 4) Reference Track — timbre conditioning + mastering
  if (preset?.reference_track_path) {
    params.sourceAudioUrl = preset.reference_track_path;
    params.audioCoverStrength = preset.audio_cover_strength ?? 0.5;
    params.masteringEnabled = true;
    params.masteringReference = preset.reference_track_path;
  }

  // 5) Submit generation
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

async function _pollUntilDone(item: AudioQueueItem, token: string): Promise<void> {
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
        if (audioUrl) {
          item.audioUrl = audioUrl;
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
