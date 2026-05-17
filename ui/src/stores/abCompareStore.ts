/**
 * abCompareStore.ts — A/B comparison state for pinning two tracks and
 * opening a parameter diff modal.
 *
 * Uses useSyncExternalStore (same pattern as playbackStore/playlistStore).
 * Playback is delegated to playbackStore.enterABMode().
 */

import { useSyncExternalStore, useRef, useCallback } from 'react';
import type { PlaybackTrack } from './playbackStore';
import { enterABMode } from './playbackStore';

// ── Types ────────────────────────────────────────────────────────────────────

export interface ABCompareState {
  trackA: PlaybackTrack | null;
  trackB: PlaybackTrack | null;
  isModalOpen: boolean;
}

// ── State ────────────────────────────────────────────────────────────────────

let _state: ABCompareState = {
  trackA: null,
  trackB: null,
  isModalOpen: false,
};

// ── Reactivity ───────────────────────────────────────────────────────────────

const _listeners = new Set<() => void>();

function notify(): void {
  _listeners.forEach(cb => cb());
}

function setState(updates: Partial<ABCompareState>): void {
  _state = { ..._state, ...updates };
  notify();
}

function subscribe(cb: () => void): () => void {
  _listeners.add(cb);
  return () => _listeners.delete(cb);
}

function getSnapshot(): ABCompareState {
  return _state;
}

// ── Public API ───────────────────────────────────────────────────────────────

export function setTrackA(track: PlaybackTrack | null): void {
  setState({ trackA: track });
}

export function setTrackB(track: PlaybackTrack | null): void {
  setState({ trackB: track });
}

export function openModal(): void {
  if (_state.trackA && _state.trackB) {
    setState({ isModalOpen: true });
  }
}

export function closeModal(): void {
  setState({ isModalOpen: false });
}

export function clear(): void {
  setState({ trackA: null, trackB: null, isModalOpen: false });
}

/** Start A/B dual playback via playbackStore */
export function playAB(): void {
  if (!_state.trackA || !_state.trackB) return;
  enterABMode(_state.trackA, _state.trackB);
}

// ── React Hooks ──────────────────────────────────────────────────────────────

export function useABCompare(): ABCompareState {
  return useSyncExternalStore(subscribe, getSnapshot);
}

export function useABCompareSelector<T>(selector: (state: ABCompareState) => T): T {
  const selectorRef = useRef(selector);
  selectorRef.current = selector;

  const selectedRef = useRef<T>(selector(_state));

  const getSelectedSnapshot = useCallback(() => {
    const next = selectorRef.current(_state);
    if (Object.is(selectedRef.current, next)) return selectedRef.current;
    selectedRef.current = next;
    return next;
  }, []);

  return useSyncExternalStore(subscribe, getSelectedSnapshot);
}
