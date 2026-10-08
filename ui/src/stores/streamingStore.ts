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
import { lyricWorkflowApi, runLyricOperation, type LyricBatchRequest, type LyricBatchResult } from '../services/lyricWorkflowApi';
import type { WorkflowJob } from '../../../server/src/contracts/workflow';

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
  userSubject?: string;
  /** Suppress LLM reasoning for this run (see CallOptions.noThink server-side). */
  noThink?: boolean;
  jobId?: string;
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

// Cap streaming text to prevent unbounded memory growth from thinking models
const MAX_STREAM_TEXT = 200_000; // ~200KB trailing window

let _state: StreamingState = {
  text: '', phase: '', done: false, visible: false, currentLabel: '', queue: [],
};

const _listeners = new Set<() => void>();

/** Immediate emit — used for infrequent state changes (queue status, stream start/finish) */
function _emit() {
  _state = { ..._state };
  _listeners.forEach(fn => fn());
}

/**
 * Throttled emit — used during streaming to batch React re-renders.
 * Without this, every chunk (hundreds/sec) triggers a full re-render,
 * creating a render storm that exhausts browser memory.
 */
let _emitScheduled = false;
function _emitThrottled() {
  if (_emitScheduled) return;
  _emitScheduled = true;
  requestAnimationFrame(() => {
    _emitScheduled = false;
    _emit();
  });
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
  // Trim to trailing window if text exceeds cap (prevents OOM from thinking models)
  if (_state.text.length > MAX_STREAM_TEXT) {
    _state.text = '…(earlier output trimmed)…\n' + _state.text.slice(-MAX_STREAM_TEXT);
  }
  _emitThrottled();
}

function setPhase(phase: string) {
  _state.phase = phase;
  _state.text += `\n--- ${phase} ---\n`;
  _emitThrottled();
}

function finishStream() {
  _state.done = true;
  _emit();
}

// ── Queue system ─────────────────────────────────────────────────────────────

let _queueRunning = false;
let _lastToken: string | null = null;
const _following = new Set<string>();

function _nextId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
}

export function addToQueue(item: Omit<QueueItem, 'id' | 'status'>, token: string): void {
  _lastToken = token;
  _state.queue.push({ ...item, id: _nextId(), status: 'pending' });
  _emit();
  _processQueue();
}

export function addBulkToQueue(items: Omit<QueueItem, 'id' | 'status'>[], token: string): void {
  _lastToken = token;
  for (const item of items) {
    _state.queue.push({ ...item, id: _nextId(), status: 'pending' });
  }
  _emit();
  _processQueue();
}

export function removeFromQueue(id: string): void {
  const item = _state.queue.find(q => q.id === id);
  if (item?.jobId && _lastToken && item.status === 'running' &&
      !_state.queue.some(q => q.id !== id && q.jobId === item.jobId && q.status === 'running')) {
    void lyricWorkflowApi.cancel(_lastToken, item.jobId);
  }
  _state.queue = _state.queue.filter(q => q.id !== id);
  _emit();
}

export function clearQueue(): void {
  _state.queue = _state.queue.filter(q => q.status === 'running' || q.status === 'pending');
  _emit();
}

async function _processQueue(): Promise<void> {
  if (_queueRunning) return;
  if (!_lastToken) return;
  _queueRunning = true;
  const pending = _state.queue.filter(q => q.status === 'pending');
  if (pending.length === 0) { _queueRunning = false; return; }
  const expanded = pending.flatMap(item => Array.from({ length: item.count || 1 }, () => item));
  const requests: LyricBatchRequest[] = pending.map(item => ({
    type: item.type, targetId: item.targetId, provider: item.provider, model: item.model,
    userSubject: item.userSubject, noThink: item.noThink, count: item.count || 1,
  }));
  try {
    const job = await lyricWorkflowApi.submitBatch(_lastToken, requests);
    for (const item of pending) { item.jobId = job.id; item.status = 'running'; }
    _emit();
    _following.add(job.id);
    try { await followQueueJob(_lastToken, job, expanded); }
    finally { _following.delete(job.id); }
  } catch (err) {
    for (const item of pending) { item.status = 'error'; item.error = (err as Error).message; }
    _emit();
  } finally {
    _queueRunning = false;
    if (_state.queue.some(q => q.status === 'pending')) void _processQueue();
  }
}

/** Reattach after a page reload. The service replays the bounded event log. */
export async function restoreLyricQueue(token: string): Promise<void> {
  _lastToken = token;
  const { jobs } = await lyricWorkflowApi.list(token);
  for (const job of jobs) {
    if (job.status !== 'pending' && job.status !== 'running') continue;
    if (_following.has(job.id)) continue;
    const raw = (job.input as { items?: Array<{ type: QueueItemType | 'fetch' | 'render'; sourceId?: number; provider?: string; model?: string }> }).items || [];
    if (!raw.length || raw.some(item => !['profile', 'generate', 'refine'].includes(item.type))) continue;
    const items = raw.map(item => ({ id: _nextId(), type: item.type as QueueItemType, targetId: item.sourceId!,
      label: `${item.type}: ${item.sourceId}`, provider: item.provider || '', model: item.model,
      status: 'running' as const, jobId: job.id }));
    _state.queue.push(...items);
    _emit();
    _following.add(job.id);
    void followQueueJob(token, job, items).finally(() => { _following.delete(job.id); });
  }
}

async function followQueueJob(token: string, job: WorkflowJob, expanded: QueueItem[]): Promise<void> {
  const status = await lyricWorkflowApi.follow(token, job.id, {
    onSnapshot: (snapshot, gap) => {
      if (gap) setPhase('Earlier progress was trimmed; waiting for current status');
      if (snapshot.status === 'interrupted') {
        for (const item of expanded) { item.status = 'error'; item.error = 'Interrupted by server restart'; }
        _emit();
      }
    },
    onEvent: event => {
      const data = event.data as { index?: number; phase?: string; text?: string; error?: string } | null;
      const item = data?.index === undefined ? null : expanded[data.index];
      if (event.type === 'item-start' && item) resetStream(item.label);
      if (event.type === 'phase' && data?.phase) setPhase(data.phase);
      if (event.type === 'chunk' && data?.text) appendChunk(data.text);
      if (event.type === 'item-result' && item) { item.countCompleted = (item.countCompleted || 0) + 1; if (item.countCompleted >= (item.count || 1) && item.status !== 'error') item.status = 'done'; finishStream(); _emit(); }
      if (event.type === 'item-error' && item) { item.status = 'error'; item.error = data?.error || 'Item failed'; item.countCompleted = (item.countCompleted || 0) + 1; finishStream(); _emit(); }
    },
  });
  if (status === 'succeeded') {
    const final = await lyricWorkflowApi.get(token, job.id);
    const results = (final.job.result as LyricBatchResult | null)?.results || [];
    for (const result of results) {
      const item = expanded[result.index];
      if (!item) continue;
      if (result.status === 'error') { item.status = 'error'; item.error = result.error; }
      else if (item.status !== 'error') item.status = 'done';
      item.countCompleted = Math.max(item.countCompleted || 0, item.count || 1);
    }
    _emit();
  }
  if (status !== 'succeeded') {
    for (const item of expanded) if (item.status === 'running') { item.status = 'error'; item.error = `Workflow ${status}`; }
    _emit();
  }
}

// ── Standalone streaming (non-queue, immediate) ──────────────────────────────

export async function startStreamBuildProfile(
  lyricsSetId: number,
  req: { provider: string; model?: string },
  token: string,
  onComplete?: () => void,
): Promise<void> {
  resetStream('Building profile…');
  try {
    await runLyricOperation(token, { type: 'profile', targetId: lyricsSetId, provider: req.provider, model: req.model },
      { onChunk: appendChunk, onPhase: setPhase, onResult: () => onComplete?.() });
  } catch (err) {
    _state.text += `\n⚠ Error: ${(err as Error).message}`;
    _emit();
    throw err;
  } finally {
    finishStream();
  }
}

export async function startStreamGenerate(
  _profileId: number,
  req: { profile_id: number; provider: string; model?: string; extra_instructions?: string; user_subject?: string; no_think?: boolean; count?: number },
  token: string,
  onComplete?: () => void,
): Promise<void> {
  resetStream('Generating lyrics…');
  try {
    await runLyricOperation(token, { type: 'generate', targetId: req.profile_id, provider: req.provider, model: req.model,
      extraInstructions: req.extra_instructions, userSubject: req.user_subject, noThink: req.no_think, count: req.count },
      { onChunk: appendChunk, onPhase: setPhase, onResult: () => onComplete?.() });
  } catch (err) {
    _state.text += `\n⚠ Error: ${(err as Error).message}`;
    _emit();
    throw err;
  } finally {
    finishStream();
  }
}

export async function startStreamRefine(
  generationId: number,
  req: { provider: string; model?: string },
  token: string,
  onComplete?: () => void,
): Promise<void> {
  resetStream('Refining lyrics…');
  try {
    await runLyricOperation(token, { type: 'refine', targetId: generationId, provider: req.provider, model: req.model },
      { onChunk: appendChunk, onPhase: setPhase, onResult: () => onComplete?.() });
  } catch (err) {
    _state.text += `\n⚠ Error: ${(err as Error).message}`;
    _emit();
    throw err;
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
