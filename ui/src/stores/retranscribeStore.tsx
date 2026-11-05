/**
 * retranscribeStore.tsx — in-flight Whisper retranscriptions.
 *
 * "Retranscribe Lyrics" runs synchronously on the server for anywhere from ten
 * seconds to a few minutes on CPU, and the menu that starts it unmounts on the
 * click. State therefore lives here rather than in the menu: every surface
 * showing the track gets the same spinner, and the outcome arrives as a toast
 * even if the user has navigated away.
 *
 * The toast names the model, because the server picks it by priority and the UI
 * has no way to know which one that is, and it carries a Cancel button: the
 * request is abortable and the server kills whisper-cli when the response
 * closes (POST /api/songs/:id/retranscribe).
 *
 * This is deliberately NOT the generation queue. That queue is the single GPU
 * lane; whisper-cli is CPU-only and must not sit in front of, or behind, a
 * render.
 *
 * Follows the useSyncExternalStore pattern of playbackStore/postProcessStore.
 */

import { useCallback, useSyncExternalStore } from 'react';
import { toast } from 'react-toastify';
import type { Song } from '../types';
import { retranscribeLyrics, whisperModels } from '../services/api';
import { ApiError } from '../services/httpClient';
import { queryClient } from '../queries/client';
import { queryManager } from '../queries/invalidation';
import { invalidateSongs } from '../queries/shell-queue/songs';

/** Song ids with a request in flight, and when each started. */
let _running: Record<string, number> = {};

/**
 * A track whose timings were placed by hand answers 409 on the first click.
 * Remembering the click lets the second one carry `force`, which is the confirm:
 * one click to be told, one to mean it.
 */
const _forceOnce: Record<string, number> = {};

/** The abort handle per running song, so the toast's Cancel button can reach it. */
const _controllers = new Map<string, AbortController>();

const _listeners = new Set<() => void>();

function notify(): void {
  _listeners.forEach(cb => cb());
}

function subscribe(cb: () => void): () => void {
  _listeners.add(cb);
  return () => { _listeners.delete(cb); };
}

function getSnapshot(): Record<string, number> {
  return _running;
}

function setRunning(songId: string, startedAt: number | null): void {
  const rest = { ..._running };
  delete rest[songId];
  _running = startedAt === null ? rest : { ...rest, [songId]: startedAt };
  notify();
}

/**
 * A track's retranscription: whether one is running, and the two ways to act on
 * it. `data` is what a surface renders — true across every surface, so the
 * spinner does not disappear when the menu closes — and `start` and `cancel` are
 * what it calls. Both types below are inferred from the hook, never retyped, so
 * a new handler cannot be added without the consumers seeing it.
 */
export function useRetranscribe(song: Song) {
  const running = useSyncExternalStore(subscribe, getSnapshot);
  const start = useCallback(() => startRetranscribe(song), [song]);
  const cancel = useCallback(() => cancelRetranscribe(song.id), [song.id]);
  return {
    data: { transcribing: running[song.id] !== undefined },
    start, cancel,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type Retranscribe = ReturnType<typeof useRetranscribe>;
/** What a component may call: the handlers, without the state. */
export type RetranscribeHandlers = Omit<Retranscribe, 'data'>;

/** Stop a running retranscription. A no-op once it has finished. */
function cancelRetranscribe(songId: string): void {
  _controllers.get(songId)?.abort();
}

/**
 * Start a retranscription and report it: a toast that stays up for the whole
 * run, naming the model the server will use and offering Cancel, then a success
 * toast with the counts or the server's reason for failing. A second click
 * while one is running is ignored.
 */
async function startRetranscribe(song: Song): Promise<void> {
  if (_running[song.id] !== undefined) return;

  const controller = new AbortController();
  _controllers.set(song.id, controller);
  setRunning(song.id, Date.now());

  const title = song.title || 'track';
  // The model is picked server-side by priority, so ask which one before the
  // toast claims it. A failed lookup only costs the label.
  const info = await whisperModels().catch(() => null);
  const modelName = info?.label || info?.model || '';

  const toastId = toast.loading(
    <div className="flex items-center gap-3">
      <span>
        Transcribing lyrics: {title}
        {modelName ? ` (${modelName})` : ''}
      </span>
      <button
        type="button"
        className="shrink-0 rounded-md border border-red-500 px-2 py-0.5 text-xs font-semibold text-red-500 hover:bg-red-500/10 dark:border-red-400/70 dark:text-red-400"
        onClick={() => cancelRetranscribe(song.id)}
      >
        Cancel
      </button>
    </div>,
    { closeButton: false, autoClose: false },
  );

  try {
    const forceAt = _forceOnce[song.id];
    const force = forceAt !== undefined && Date.now() - forceAt < 60_000;
    delete _forceOnce[song.id];
    const result = await retranscribeLyrics(song.id, force
      ? { signal: controller.signal, force: true }
      : { signal: controller.signal });
    void invalidateSongs(queryClient);
    void queryManager.sidecars.all(queryClient);
    const used = result.model ? ` with ${modelName || result.model}` : '';
    toast.update(toastId, {
      render: `Lyrics transcribed${used}: ${result.lineCount} lines, ${result.wordCount} words`,
      type: 'success',
      isLoading: false,
      autoClose: 6000,
    });
  } catch (err: unknown) {
    if (controller.signal.aborted) {
      toast.update(toastId, {
        render: 'Transcription cancelled',
        type: 'error',
        isLoading: false,
        autoClose: 4000,
      });
    } else if (err instanceof ApiError && err.status === 409) {
      _forceOnce[song.id] = Date.now();
      toast.update(toastId, {
        // The server's sentence is the only copy: one message, from the route
        // that decided to refuse, so the two can never drift apart again.
        render: err?.message || "This track's lyric timings were edited by hand. Click Retranscribe again to overwrite them.",
        type: 'warning',
        isLoading: false,
        autoClose: 12000,
      });
    } else {
      const message = (err instanceof Error && err.message) || 'Transcription failed';
      toast.update(toastId, {
        render: `Transcription failed: ${message}`,
        type: 'error',
        isLoading: false,
        autoClose: 12000,
      });
    }
  } finally {
    _controllers.delete(song.id);
    setRunning(song.id, null);
  }
}
