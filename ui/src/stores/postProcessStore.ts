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
import { QueryObserver } from '@tanstack/react-query';
import type { Song } from '@/types';
import { runSongPostProcessing, revertSongPostProcessing } from '@/services/api';
import { queryClient } from '@/queries/client';
import { postProcessStatusOptions } from '@/queries/shell-queue/songs';
import { useGlobalParamsStore } from './globalParamsStore';
import { forgetPeaks } from '@/audio/peaksClient';

export interface PostProcessRun {
  songId: string;
  /** Kept on the run so the activity panel can name it without holding the
   *  whole song — the row the click came from may be unmounted by then. */
  title: string;
  jobId?: string;
  status: 'starting' | 'pending' | 'running' | 'succeeded' | 'failed';
  stage: string;
  error?: string;
  masteredAudioUrl?: string;
  /** The run was refused because a processed version already exists. The dock
   *  offers the remedy inline, since that is where the refusal is seen. */
  alreadyProcessed?: boolean;
}

// ── State ────────────────────────────────────────────────────────────────────

let _runs: Record<string, PostProcessRun> = {};

const _listeners = new Set<() => void>();
/** Per song, the unsubscribe of the Query observer polling its run. */
const _pollers = new Map<string, () => void>();

function notify(): void {
  _listeners.forEach(cb => cb());
}

function setRun(songId: string, patch: Partial<PostProcessRun>): void {
  const prev = _runs[songId] || { songId, title: '', status: 'starting' as const, stage: '' };
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

/** How long a completed run stays on the dock before retiring itself. Long
 *  enough to notice the tick, short enough that a batch drains on its own. */
const SUCCESS_LINGER_MS = 4000;

function stopPolling(songId: string): void {
  const unsubscribe = _pollers.get(songId);
  if (unsubscribe) { unsubscribe(); _pollers.delete(songId); }
}

function startPolling(songId: string, jobId: string): void {
  stopPolling(songId);
  // Query owns the reads: one request at a time, at the active cadence, and
  // the options stop polling once the run ends. Unsubscribing on the first
  // terminal answer keeps the completion event to one.
  const observer = new QueryObserver(queryClient, postProcessStatusOptions(songId, jobId));
  _pollers.set(songId, observer.subscribe(result => {
    if (result.status === 'error') {
      stopPolling(songId);
      setRun(songId, { status: 'failed', stage: 'Failed', error: result.error.message });
      return;
    }
    const status = result.data;
    if (!status) { stopPolling(songId); clearRun(songId); return; } // job pruned (404): nothing left to poll
    setRun(songId, {
      status: status.status,
      stage: status.stage,
      error: status.error,
      masteredAudioUrl: status.masteredAudioUrl,
    });

    if (status.status === 'succeeded') {
      stopPolling(songId);
      // Every pass writes the same <base>_mastered.wav URL, so the peaks
      // cached for an earlier pass would otherwise draw over this one.
      if (status.masteredAudioUrl) forgetPeaks(status.masteredAudioUrl);
      // Let every list holding this song swap to the processed file.
      window.dispatchEvent(new CustomEvent('song-postprocessed', {
        detail: { songId, masteredAudioUrl: status.masteredAudioUrl },
      }));
      // Show the tick briefly, then retire the run so the dock empties
      // itself and disappears. Safe to do now that the dock exists: the run
      // was visible for its whole duration, and the track itself carries the
      // lasting evidence (a mastered version, and the M/O switch).
      //
      // FAILURES ARE NOT CLEARED HERE, deliberately. A success that
      // disappears is tidy; an error that disappears is a bug you never find
      // out about.
      setTimeout(() => {
        if (_runs[songId]?.status === 'succeeded') clearRun(songId);
      }, SUCCESS_LINGER_MS);
    } else if (status.status === 'failed') {
      stopPolling(songId);
    }
  }));
}

/**
 * Send one run to the server and start polling it.
 *
 * Shared by the menu path and the failure toast's retry, so both build the same
 * payload and leave the same state behind. The payload is the whole
 * getGlobalParams() object rather than a hand-picked set of PP keys: a
 * whitelist here would go stale the first time someone adds a PP knob, and a
 * silently-dropped param is this codebase's most-repeated bug. The server
 * replaces the fields that belong to the song rather than the form
 * (instrumental, caption, seed) before it runs anything.
 */
async function queuePostProcessingRun(songId: string, title: string): Promise<void> {
  // Clear the old job id: it belongs to the run that failed, and the toast
  // dedupe keys on it, so a failed retry must not reuse it.
  setRun(songId, { title, jobId: undefined, status: 'starting', stage: 'Queueing...', error: undefined });
  try {
    const params = useGlobalParamsStore.getState().getGlobalParams({ postProcessingEnabled: true });
    const { jobId, stage } = await runSongPostProcessing(songId, params as Record<string, any>);
    setRun(songId, { jobId, status: 'pending', stage: stage || 'Queued' });
    startPolling(songId, jobId);
  } catch (err: unknown) {
    setRun(songId, {
      status: 'failed',
      stage: 'Failed',
      error: err instanceof Error ? err.message : 'Post-processing failed',
    });
    throw err;
  }
}

/**
 * Queue a post-processing pass for `song` using the PP settings currently in
 * the panel, with the master toggle forced on.
 */
export async function startPostProcessing(song: Song): Promise<void> {
  const availability = canPostProcess(song);
  if (!availability.enabled) throw new Error(availability.reason);

  try {
    await queuePostProcessingRun(song.id, song.title || 'Untitled');
  } catch (err: any) {
    // A stale song object is the usual reason for this refusal — the list was
    // fetched before the pass finished. Correct every surface holding this
    // song, so "Remove Post-Processed Version" appears instead of staying
    // hidden behind the same stale flag that allowed the click.
    if (err?.alreadyProcessed && err.masteredAudioUrl) {
      window.dispatchEvent(new CustomEvent('song-postprocessed', {
        detail: { songId: song.id, masteredAudioUrl: err.masteredAudioUrl },
      }));
    }
    setRun(song.id, { alreadyProcessed: !!err?.alreadyProcessed });
    throw err;
  }
}

/**
 * Run a failed pass again, after its dependency was enabled from the toast.
 * The server re-checks eligibility, so a stale local copy of the song cannot
 * start a run it would refuse.
 */
export async function retryPostProcessing(songId: string, title?: string): Promise<void> {
  await queuePostProcessingRun(songId, title || _runs[songId]?.title || 'Untitled');
}

// ── Failure triage ───────────────────────────────────────────────────────────

export type PpFailureKind = 'resources' | 'dependency' | 'other';

export interface PpDependencyInfo {
  /** The setting `enablePpDependency` turns on before the retry. */
  id: 'ppVae' | 'stableStep' | 'spectral' | 'naturalizer' | 'plugin';
  label: string;
}

const RESOURCE_RE = /out of memory|\boom\b|vram|gpu memory|cuda|allocation|alloc|insufficient memory|not enough memory|resources? (limit|exhaust)|memory (limit|exhaust)/i;
const DEPENDENCY_RE = /not installed|missing model|model .*missing|missing .*model|pp-?vae|stable-?step|sa3|vst|plugin|no stage produced audio|dependency|weights|tokenizer/i;

/**
 * Why a failed pass failed, as far as its message can say.
 *
 * Resources are checked first: a memory message usually also names the model it
 * was loading, and freeing VRAM is the only thing that helps there. A missing
 * description is an input problem, not an install — enabling the stage again
 * would fail the same way, so it is not treated as a dependency.
 */
export function ppFailureKind(message: string): PpFailureKind {
  if (RESOURCE_RE.test(message)) return 'resources';
  if (/needs a description of this track|genre or style/i.test(message)) return 'other';
  return DEPENDENCY_RE.test(message) ? 'dependency' : 'other';
}

/** The stage a dependency failure names, when it has a switch to turn on. */
export function ppDependency(message: string): PpDependencyInfo | null {
  // "Nothing ran" lists the stages it tried, not a switch that is off; the
  // remedy is installing what is missing, not re-enabling what was requested.
  if (/no stage produced audio/i.test(message)) return null;
  // A missing model file is installed in Model Manager; turning the stage on
  // again cannot conjure it.
  if (/not installed|missing model|model .*missing|missing .*model/i.test(message)) return null;
  if (/pp-?vae/i.test(message)) return { id: 'ppVae', label: 'PP-VAE' };
  if (/stable-?step|sa3/i.test(message)) return { id: 'stableStep', label: 'StableStep' };
  if (/spectral/i.test(message)) return { id: 'spectral', label: 'Spectral Lifter' };
  if (/naturalizer/i.test(message)) return { id: 'naturalizer', label: 'Vocal Naturalizer' };
  if (/plugin/i.test(message)) return { id: 'plugin', label: 'post-process plugin' };
  // A VST chain has no switch of its own; it is configured in its editor.
  return null;
}

/** Turn on the stage a failure named, so the retry can include it. */
export function enablePpDependency(dependency: PpDependencyInfo): void {
  const gp = useGlobalParamsStore.getState();
  switch (dependency.id) {
    case 'ppVae': gp.setPpVaeReencode(true); break;
    case 'stableStep': gp.setStableStepOn(true); break;
    case 'spectral': gp.setSpectralLifterEnabled(true); break;
    case 'naturalizer': gp.setVocalNaturalizerEnabled(true); break;
    case 'plugin': gp.setPostprocessEnabled(true); break;
  }
}

/**
 * Remove a song's post-processed version so it can be run again.
 *
 * Clears any local run record too — the "complete" badge describes a file that
 * no longer exists once this returns.
 */
export async function revertPostProcessing(song: Song): Promise<boolean> {
  return revertPostProcessingById(song.id, song.title);
}

/**
 * Same as revertPostProcessing but keyed by id — for callers like the activity
 * dock that hold a run record rather than a Song.
 */
export async function revertPostProcessingById(songId: string, title?: string): Promise<boolean> {
  try {
    const { removed } = await revertSongPostProcessing(songId);
    stopPolling(songId);
    clearRun(songId);
    // Announced even when the server had nothing left to remove. `removed:
    // false` means some other surface got there first, so this client is the
    // stale one — exactly the case that most needs correcting.
    window.dispatchEvent(new CustomEvent('song-postprocess-reverted', {
      detail: { songId },
    }));
    return removed;
  } catch (err: any) {
    // A revert that fails silently is indistinguishable from one that worked
    // but did not redraw — which is precisely how a successful playlist revert
    // read as a dead button. Put it where runs report themselves.
    setRun(songId, {
      title: title || _runs[songId]?.title || 'Untitled',
      status: 'failed',
      stage: 'Failed',
      error: `Could not remove the processed version: ${err.message}`,
      alreadyProcessed: false,
    });
    throw err;
  }
}

/** Drop a finished run from the activity panel. */
export function dismissRun(songId: string): void {
  stopPolling(songId);
  clearRun(songId);
}

// ── Hooks ────────────────────────────────────────────────────────────────────

export function usePostProcessRuns(): Record<string, PostProcessRun> {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/** Runs worth showing in the activity panel, newest interaction first. */
export function usePostProcessRunList(): PostProcessRun[] {
  const runs = usePostProcessRuns();
  return Object.values(runs);
}

/** The active run for one song, or undefined. */
export function usePostProcessRun(songId: string | undefined): PostProcessRun | undefined {
  const runs = usePostProcessRuns();
  return songId ? runs[songId] : undefined;
}
