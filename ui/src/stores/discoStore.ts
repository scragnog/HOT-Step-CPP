// discoStore.ts — Disco mode state + beat detection engine.
//
// Uses the same useSyncExternalStore pattern as playbackStore.
// Reads kick drum energy from the registered audioMotion-analyzer instance
// via getEnergy(60, 150). Uses onset detection (not raw energy level) to
// detect actual beat hits — the pulse spikes on kick transients and decays fast.

import { useSyncExternalStore, useRef, useCallback } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

interface DiscoState {
  discoMode: boolean;
  pulseIntensity: number;  // 0.0–1.0, onset-detected pulse
}

// ── localStorage ─────────────────────────────────────────────────────────────

const DISCO_PREFS_KEY = 'disco-prefs';

function loadDiscoPrefs(): { discoMode: boolean } {
  try {
    const raw = localStorage.getItem(DISCO_PREFS_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { discoMode: false };
}

function saveDiscoPrefs(discoMode: boolean): void {
  localStorage.setItem(DISCO_PREFS_KEY, JSON.stringify({ discoMode }));
}

// ── audioMotion Registration ─────────────────────────────────────────────────

let _audioMotion: any = null;

export function registerAudioMotion(instance: any): void {
  _audioMotion = instance;
}

export function unregisterAudioMotion(): void {
  _audioMotion = null;
}

// ── State ────────────────────────────────────────────────────────────────────

const prefs = loadDiscoPrefs();

let _state: DiscoState = {
  discoMode: prefs.discoMode,
  pulseIntensity: 0,
};

// ── Reactivity (useSyncExternalStore) ────────────────────────────────────────

const _listeners = new Set<() => void>();

function notify(): void {
  _listeners.forEach(cb => cb());
}

function setState(updates: Partial<DiscoState>): void {
  _state = { ..._state, ...updates };
  notify();
}

function subscribe(cb: () => void): () => void {
  _listeners.add(cb);
  return () => _listeners.delete(cb);
}

function getSnapshot(): DiscoState {
  return _state;
}

// ── Beat Detection Loop (Onset Detection) ────────────────────────────────────

let _rafId: number | null = null;
let _lastFrameTime = 0;
let _debugFrameCount = 0;

// Onset detection state
let _runningAvg = 0;         // Slow-moving average of bass energy
let _pulse = 0;              // Current pulse value (0–1), decays per frame
let _lastHitTime = 0;        // Prevent double-triggers within refractory period

// Tuning knobs
const AVG_SMOOTHING = 0.93;  // How slow the running average adapts (higher = slower)
const ONSET_MULT = 1.4;      // Energy must exceed avg * this to trigger a hit
const ONSET_FLOOR = 0.08;    // Minimum absolute delta to count as a hit
const REFRACTORY_MS = 80;    // Min ms between hits (prevents double-triggers)
const DECAY_RATE = 0.88;     // Per-frame decay (lower = faster drop) — ~150ms to near-zero at 60fps
const HIT_STRENGTH = 2.5;    // Multiplier on the delta to scale the pulse spike

function beatDetectionLoop(timestamp: number): void {
  if (!_state.discoMode) {
    _rafId = null;
    _pulse = 0;
    setState({ pulseIntensity: 0 });
    return;
  }

  const dt = _lastFrameTime ? timestamp - _lastFrameTime : 16;
  _lastFrameTime = timestamp;

  let rawEnergy = 0;
  if (_audioMotion) {
    try {
      rawEnergy = _audioMotion.getEnergy(60, 150) ?? 0;
    } catch {
      rawEnergy = 0;
    }
  }

  // Update running average (slow-moving baseline)
  _runningAvg = _runningAvg * AVG_SMOOTHING + rawEnergy * (1 - AVG_SMOOTHING);

  // Onset detection: is current energy significantly above the running average?
  const delta = rawEnergy - _runningAvg;
  const threshold = _runningAvg * (ONSET_MULT - 1) + ONSET_FLOOR;
  const timeSinceHit = timestamp - _lastHitTime;

  if (delta > threshold && timeSinceHit > REFRACTORY_MS) {
    // HIT! Spike the pulse proportional to how much we exceeded the threshold
    const hitIntensity = Math.min(1.0, (delta / threshold) * HIT_STRENGTH * 0.5);
    _pulse = Math.max(_pulse, hitIntensity);
    _lastHitTime = timestamp;
  }

  // Decay the pulse each frame
  _pulse *= DECAY_RATE;
  if (_pulse < 0.01) _pulse = 0;

  // Debug: log once per second
  _debugFrameCount++;
  if (_debugFrameCount % 60 === 0) {
    console.log(`[Disco] raw=${rawEnergy.toFixed(3)} avg=${_runningAvg.toFixed(3)} delta=${delta.toFixed(3)} pulse=${_pulse.toFixed(3)}`);
  }

  // Only notify if meaningfully changed
  if (Math.abs(_pulse - _state.pulseIntensity) > 0.005) {
    _state = { ..._state, pulseIntensity: _pulse };
    notify();
  }

  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function startLoop(): void {
  if (_rafId !== null) return;
  _lastFrameTime = 0;
  _runningAvg = 0;
  _pulse = 0;
  _lastHitTime = 0;
  _debugFrameCount = 0;
  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function stopLoop(): void {
  if (_rafId !== null) {
    cancelAnimationFrame(_rafId);
    _rafId = null;
  }
  _pulse = 0;
  _runningAvg = 0;
  setState({ pulseIntensity: 0 });
}

// ── Public API ───────────────────────────────────────────────────────────────

export function setDiscoMode(on: boolean): void {
  setState({ discoMode: on });
  saveDiscoPrefs(on);
  if (on) {
    startLoop();
  } else {
    stopLoop();
  }
}

export function toggleDiscoMode(): void {
  setDiscoMode(!_state.discoMode);
}

export function setDiscoPlaying(isPlaying: boolean): void {
  if (_state.discoMode && isPlaying) {
    startLoop();
  } else if (!isPlaying) {
    stopLoop();
  }
}

// ── React Hooks ──────────────────────────────────────────────────────────────

export function useDiscoSelector<T>(selector: (state: DiscoState) => T): T {
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

export function usePulseIntensity(): number {
  return useDiscoSelector(s => s.pulseIntensity);
}

export function useDiscoMode(): boolean {
  return useDiscoSelector(s => s.discoMode);
}
