/**
 * streamingStore.ts — Module-level singleton store for LLM streaming + queue.
 *
 * Lives outside React component lifecycle so SSE connections and queue state
 * survive navigation between panels.
 *
 * Components subscribe via `useStreamingStore()` which uses `useSyncExternalStore`.
 */

import { useSyncExternalStore } from 'react';
import { skipThinking } from '../services/lireekApi';

// ── Types ────────────────────────────────────────────────────────────────────

export type QueueItemType = 'profile' | 'generate' | 'refine';

export interface QueueItem {
  id: string;
  type: QueueItemType;
  targetId: number;
  label: string;
  provider: string;
  model?: string;
  status: 'pending' | 'running' | 'done' | 'error';
  error?: string;
  count?: number;
  countCompleted?: number;
}

export interface StreamingState {
  text: string;
  phase: string;
  done: boolean;
  visible: boolean;
  currentLabel: string;
  queue: QueueItem[];
}

// ── Module-level singleton state ─────────────────────────────────────────────

let _state: StreamingState = {
  text: '', phase: '', done: false, visible: false, currentLabel: '', queue: [],
};

const _listeners = new Set<() => void>();

function _emit() {
  _state = { ..._state };
  _listeners.forEach(fn => fn());
}

function _getSnapshot(): StreamingState { return _state; }

function _subscribe(listener: () => void): () => void {
  _listeners.add(listener);
  return () => _listeners.delete(listener);
}

// ── Streaming control ────────────────────────────────────────────────────────

function resetStream(label: string) {
  _state.text = '';
  _state.phase = '';
  _state.done = false;
  _state.visible = true;
  _state.currentLabel = label;
  _emit();
}

function appendChunk(text: string) {
  _state.text += text;
  _emit();
}

function setPhase(phase: string) {
  _state.phase = phase;
  _state.text += `\n--- ${phase} ---\n`;
  _emit();
}

function finishStream() {
  _state.done = true;
  _emit();
}

interface StreamCallbacks {
  onChunk: (text: string) => void;
  onPhase: (phase: string) => void;
  onResult?: (data: any) => void;
  onError?: (msg: string) => void;
}

function makeCallbacks(onResult?: (data: any) => void, onError?: (msg: string) => void): StreamCallbacks {
  return { onChunk: appendChunk, onPhase: setPhase, onResult, onError };
}

// ── SSE consumer helper ──────────────────────────────────────────────────────

async function consumeSSE(url: string, body: any, callbacks: StreamCallbacks): Promise<void> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEventType = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();

      // Track named event types: "event: chunk", "event: phase", "event: complete"
      if (trimmed.startsWith('event: ')) {
        currentEventType = trimmed.slice(7).trim();
        continue;
      }

      if (trimmed.startsWith('data: ')) {
        try {
          const data = JSON.parse(trimmed.slice(6));
          const eventType = currentEventType || '';
          currentEventType = ''; // Reset after use

          if (data.error) {
            callbacks.onError?.(data.error);
            throw new Error(data.error);
          }

          // Dispatch by named event type first, then fall back to inline field detection
          switch (eventType) {
            case 'phase':
              callbacks.onPhase(data.phase || data.text || '');
              break;
            case 'chunk':
              callbacks.onChunk(data.text || data.chunk || '');
              break;
            case 'complete':
            case 'result':
              callbacks.onResult?.(data);
              break;
            case 'error':
              callbacks.onError?.(data.error || data.message || 'Unknown error');
              break;
            default:
              // Fallback: detect by inline fields (for servers that don't send event: lines)
              if (data.phase) callbacks.onPhase(data.phase);
              else if (data.text || data.chunk) callbacks.onChunk(data.text || data.chunk);
              else if (data.result) callbacks.onResult?.(data.result);
              break;
          }
        } catch (e) {
          // Re-throw server errors, only swallow JSON parse failures
          if (e instanceof Error && !e.message.includes('JSON')) throw e;
        }
      }
    }
  }
}

// ── Queue system ─────────────────────────────────────────────────────────────

let _queueRunning = false;

function _nextId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
}

export function addToQueue(item: Omit<QueueItem, 'id' | 'status'>): void {
  _state.queue.push({ ...item, id: _nextId(), status: 'pending' });
  _emit();
  _processQueue();
}

export function addBulkToQueue(items: Omit<QueueItem, 'id' | 'status'>[]): void {
  for (const item of items) {
    _state.queue.push({ ...item, id: _nextId(), status: 'pending' });
  }
  _emit();
  _processQueue();
}

export function removeFromQueue(id: string): void {
  _state.queue = _state.queue.filter(q => q.id !== id || q.status === 'running');
  _emit();
}

export function clearQueue(): void {
  _state.queue = _state.queue.filter(q => q.status === 'running');
  _emit();
}

async function _processQueue(): Promise<void> {
  if (_queueRunning) return;
  _queueRunning = true;

  while (true) {
    const next = _state.queue.find(q => q.status === 'pending');
    if (!next) break;

    next.status = 'running';
    _emit();

    try {
      await _executeQueueItem(next);
      next.status = 'done';
    } catch (err) {
      next.status = 'error';
      next.error = (err as Error).message;
    }
    _emit();
  }

  _queueRunning = false;
}

async function _executeQueueItem(item: QueueItem): Promise<void> {
  const totalCount = item.count || 1;

  for (let i = 0; i < totalCount; i++) {
    const runLabel = totalCount > 1
      ? `${item.label} (${i + 1}/${totalCount})`
      : item.label;
    resetStream(runLabel);
    item.countCompleted = i;
    _emit();

    switch (item.type) {
      case 'profile':
        await consumeSSE(
          `/api/lireek/lyrics-sets/${item.targetId}/build-profile-stream`,
          { provider_name: item.provider, model: item.model },
          makeCallbacks(),
        );
        break;
      case 'generate':
        await consumeSSE(
          `/api/lireek/profiles/${item.targetId}/generate-stream`,
          { provider_name: item.provider, model: item.model },
          makeCallbacks(),
        );
        break;
      case 'refine':
        await consumeSSE(
          `/api/lireek/generations/${item.targetId}/refine-stream`,
          { provider_name: item.provider, model: item.model },
          makeCallbacks(),
        );
        break;
    }
    finishStream();
  }
  item.countCompleted = totalCount;
  _emit();
}

// ── Standalone streaming (non-queue, immediate) ──────────────────────────────

export async function startStreamBuildProfile(
  lyricsSetId: number,
  req: { provider: string; model?: string },
  onComplete?: () => void,
): Promise<void> {
  resetStream('Building profile…');
  try {
    await consumeSSE(
      `/api/lireek/lyrics-sets/${lyricsSetId}/build-profile-stream`,
      { provider_name: req.provider, model: req.model },
      makeCallbacks(() => onComplete?.(), (msg) => { _state.text += `\n⚠ Error: ${msg}`; _emit(); }),
    );
  } catch (err) {
    _state.text += `\n⚠ Error: ${(err as Error).message}`;
    _emit();
  } finally {
    finishStream();
  }
}

export async function startStreamGenerate(
  _profileId: number,
  req: { profile_id: number; provider: string; model?: string; extra_instructions?: string },
  onComplete?: () => void,
): Promise<void> {
  resetStream('Generating lyrics…');
  try {
    await consumeSSE(
      `/api/lireek/profiles/${req.profile_id}/generate-stream`,
      { provider_name: req.provider, model: req.model, extra_instructions: req.extra_instructions },
      makeCallbacks(() => onComplete?.(), (msg) => { _state.text += `\n⚠ Error: ${msg}`; _emit(); }),
    );
  } catch (err) {
    _state.text += `\n⚠ Error: ${(err as Error).message}`;
    _emit();
  } finally {
    finishStream();
  }
}

export async function startStreamRefine(
  generationId: number,
  req: { provider: string; model?: string },
  onComplete?: () => void,
): Promise<void> {
  resetStream('Refining lyrics…');
  try {
    await consumeSSE(
      `/api/lireek/generations/${generationId}/refine-stream`,
      { provider_name: req.provider, model: req.model },
      makeCallbacks(() => onComplete?.(), (msg) => { _state.text += `\n⚠ Error: ${msg}`; _emit(); }),
    );
  } catch (err) {
    _state.text += `\n⚠ Error: ${(err as Error).message}`;
    _emit();
  } finally {
    finishStream();
  }
}

export function doSkipThinking(): void {
  skipThinking().catch(() => {});
}

// ── React hook ───────────────────────────────────────────────────────────────

export function useStreamingStore(): StreamingState {
  return useSyncExternalStore(_subscribe, _getSnapshot, _getSnapshot);
}
