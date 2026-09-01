/**
 * postProcessStore.ts — after-the-fact post-processing runs.
 *
 * "Run through Post-Processing" exists for the render made with the global PP
 * toggle off that turned out to be a keeper. The chain is the same one a
 * generation runs; only the trigger is different.
 *
 * State lives here rather than in the menu component because the menu unmounts
 * the moment it is clicked, and because every surface showing that track wants
 * the same spinner. Follows the useSyncExternalStore pattern of
 * playbackStore/abCompareStore.
 */

import { useSyncExternalStore } from 'react';
import type { Song } from '../types';
import { runSongPostProcessing, getSongPostProcessingStatus } from '../services/api';
import { useGlobalParamsStore } from './globalParamsStore';

export interface PostProcessRun {
  songId: string;
  jobId?: string;
  status: 'starting' | 'pending' | 'running' | 'succeeded' | 'failed';
  stage: string;
  error?: string;
  masteredAudioUrl?: string;
}

// ── State ────────────────────────────────────────────────────────────────────

let _runs: Record<string, PostProcessRun> = {};

const _listeners = new Set<() => void>();
const _pollers = new Map<string, ReturnType<typeof setInterval>>();

function notify(): void {
  _listeners.forEach(cb => cb());
}

function setRun(songId: string, patch: Partial<PostProcessRun>): void {
  const prev = _runs[songId] || { songId, status: 'starting' as const, stage: '' };
  _runs = { ..._runs, [songId]: { ...prev, ...patch, songId } };
  notify();
}

function clearRun(songId: string): void {
  const { [songId]: _drop, ...rest } = _runs;
  _runs = rest;
  notify();
}

function subscribe(cb: () => void): () => void {
  _listeners.add(cb);
  return () => { _listeners.delete(cb); };
}

function getSnapshot(): Record<string, PostProcessRun> {
  return _runs;
}

// ── Eligibility ──────────────────────────────────────────────────────────────

export interface PpAvailability {
  enabled: boolean;
  /** Why not, for the disabled item's tooltip. Empty when enabled. */
  reason: string;
}

/**
 * Whether this song can be post-processed after the fact.
 *
 * The chain is NOT idempotent — mastering a mastered file overcooks it — so a
 * song that already carries a mastered version is refused. Nothing is lost by
 * being strict here: a song only ever gets a mastered_audio_url when a stage
 * actually ran, so an empty one really does mean "never processed".
 *
 * The server enforces the same rule (services/generation/rePostProcess.ts);
 * this is the UI half, so the menu can explain itself instead of failing on
 * click.
 */
export function canPostProcess(song: Song | null | undefined): PpAvailability {
  if (!song) return { enabled: false, reason: 'No song selected' };

  const mastered = song.masteredAudioUrl || (song as any).mastered_audio_url || '';
  if (mastered) {
    return { enabled: false, reason: 'Already post-processed — running the chain again would overcook it' };
  }

  const audioUrl = song.audioUrl || (song as any).audio_url || '';
  if (!audioUrl) return { enabled: false, reason: 'No audio file' };
  if (!audioUrl.toLowerCase().endsWith('.wav')) {
    return { enabled: false, reason: 'Needs the raw WAV render' };
  }

  const run = _runs[song.id];
  if (run && (run.status === 'starting' || run.status === 'pending' || run.status === 'running')) {
    return { enabled: false, reason: 'Post-processing is already running' };
  }

  return { enabled: true, reason: '' };
}

// ── Runner ───────────────────────────────────────────────────────────────────

const POLL_MS = 1500;

function stopPolling(songId: string): void {
  const timer = _pollers.get(songId);
  if (timer) { clearInterval(timer); _pollers.delete(songId); }
}

function startPolling(songId: string, jobId: string): void {
  stopPolling(songId);
  const timer = setInterval(async () => {
    try {
      const status = await getSongPostProcessingStatus(songId, jobId);
      if (!status) return; // job aged out of the server's map — keep the last state
      setRun(songId, {
        status: status.status,
        stage: status.stage,
        error: status.error,
        masteredAudioUrl: status.masteredAudioUrl,
      });

      if (status.status === 'succeeded') {
        stopPolling(songId);
        // Let every list holding this song swap to the processed file.
        window.dispatchEvent(new CustomEvent('song-postprocessed', {
          detail: { songId, masteredAudioUrl: status.masteredAudioUrl },
        }));
        // Leave the succeeded state up briefly so the surface can show it.
        setTimeout(() => clearRun(songId), 4000);
      } else if (status.status === 'failed') {
        stopPolling(songId);
      }
    } catch (err: any) {
      stopPolling(songId);
      setRun(songId, { status: 'failed', stage: 'Failed', error: err.message });
    }
  }, POLL_MS);
  _pollers.set(songId, timer);
}

/**
 * Queue a post-processing pass for `song` using the PP settings currently in
 * the panel, with the master toggle forced on.
 *
 * The payload is the whole getGlobalParams() object rather than a hand-picked
 * set of PP keys. A whitelist here would go stale the first time someone adds a
 * PP knob, and a silently-dropped param is this codebase's most-repeated bug.
 * The server replaces the fields that belong to the song rather than the form
 * (instrumental, caption, seed) before it runs anything.
 */
export async function startPostProcessing(song: Song): Promise<void> {
  const availability = canPostProcess(song);
  if (!availability.enabled) throw new Error(availability.reason);

  setRun(song.id, { status: 'starting', stage: 'Queueing...', error: undefined });

  try {
    const params = useGlobalParamsStore.getState().getGlobalParams({ postProcessingEnabled: true });
    const { jobId, stage } = await runSongPostProcessing(song.id, params as Record<string, any>);
    setRun(song.id, { jobId, status: 'pending', stage: stage || 'Queued' });
    startPolling(song.id, jobId);
  } catch (err: any) {
    setRun(song.id, { status: 'failed', stage: 'Failed', error: err.message });
    throw err;
  }
}

/** Drop a failed run from the UI once the user has read the error. */
export function dismissPostProcessing(songId: string): void {
  stopPolling(songId);
  clearRun(songId);
}

// ── Hooks ────────────────────────────────────────────────────────────────────

export function usePostProcessRuns(): Record<string, PostProcessRun> {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/** The active run for one song, or undefined. */
export function usePostProcessRun(songId: string | undefined): PostProcessRun | undefined {
  const runs = usePostProcessRuns();
  return songId ? runs[songId] : undefined;
}
