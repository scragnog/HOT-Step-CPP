// discoStore.ts — Disco mode state + multi-stem beat detection engine.
//
// SYNC STRATEGY: Pre-analyzed disco data is loaded as a tiny JSON file
// (~30-60 KB) generated server-side during stem extraction. Energy per
// ~16ms window is already computed. During playback, we look up energy
// at the main player's currentTime — a pure array index. Zero sync
// issues, zero Web Audio API overhead.
//
// Fallback chain:
//   1. Disco JSON (preferred — tiny, instant, server-analyzed)
//   2. audioMotion analyser (basic kick detection from full mix)
//
// Three stem channels:
//   1. KICK   — punch/impact → scale pulse + rainbow border
//   2. SNARE  — crack/flash → full-viewport white flash
//   3. HI-HAT — shimmer → floating particles

import { useSyncExternalStore, useRef, useCallback } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

interface DiscoState {
  discoMode: boolean;
  kickPulse: number;     // 0.0–1.0
  snarePulse: number;    // 0.0–1.0
  hihatEnergy: number;   // 0.0–1.0 (continuous)
  discoDataLoaded: boolean;
}

/** Server-generated disco data — matches disco-analyzer.ts output */
interface DiscoData {
  version: number;
  fps: number;
  duration: number;
  kick: number[];
  snare: number[];
  hihat: number[];
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

// ── Disco Data Management ────────────────────────────────────────────────────

let _discoData: DiscoData | null = null;
let _discoDataUrl = '';

// Track main player state
let _mainCurrentTime = 0;

/** Look up normalised energy at a given time */
function getEnergyAtTime(energyArray: number[] | undefined, fps: number, timeSec: number): number {
  if (!energyArray || energyArray.length === 0) return -1;
  const idx = Math.floor(timeSec * fps);
  if (idx < 0 || idx >= energyArray.length) return 0;
  return energyArray[idx];
}

/** Load disco data JSON from server */
function loadDiscoData(url: string): void {
  if (url === _discoDataUrl) return;
  _discoDataUrl = url;

  if (!url) {
    _discoData = null;
    setState({ discoDataLoaded: false });
    console.log('[Disco] Disco data unloaded');
    return;
  }

  console.log(`[Disco] Loading disco data: ${url}`);

  fetch(url)
    .then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((data: DiscoData) => {
      if (_discoDataUrl !== url) return; // URL changed while loading
      _discoData = data;
      setState({ discoDataLoaded: true });
      const totalWindows = Math.max(data.kick?.length || 0, data.snare?.length || 0, data.hihat?.length || 0);
      console.log(`[Disco] Disco data ready: ${totalWindows} windows, ${data.duration?.toFixed(1)}s, fps=${data.fps}`);
    })
    .catch(err => {
      console.warn('[Disco] Disco data load failed:', err);
      _discoData = null;
      setState({ discoDataLoaded: false });
    });
}

// ── Public API ───────────────────────────────────────────────────────────────

/** Set the disco data URL — this is the primary data source */
export function setDiscoDataUrl(url: string): void {
  loadDiscoData(url);
}

/** Legacy: kept for backward compat. Stem URLs are no longer used for analysis. */
export function setStemUrls(_urls: { kick?: string; snare?: string; hihat?: string }): void {
  // No-op — analysis now comes from disco data JSON
}

/** Legacy alias */
export function setKickStemUrl(_url: string): void {
  // No-op
}

/** Update main player time — called from App.tsx on every currentTime change */
export function updateMainTime(time: number): void {
  _mainCurrentTime = time;
}

/** Sync main player state (play/pause) */
export function syncStems(action: 'play' | 'pause' | 'seek', time?: number): void {
  if (time !== undefined && (action === 'play' || action === 'seek')) {
    _mainCurrentTime = time;
  }
}

/** Legacy alias */
export function syncKickStem(action: 'play' | 'pause' | 'seek', time?: number): void {
  syncStems(action, time);
}

// ── Audio: Full Mix Fallback ─────────────────────────────────────────────────

let _fallbackAnalyser: AnalyserNode | null = null;
let _fallbackFreqData: Uint8Array<ArrayBuffer> | null = null;
let _fallbackBinStart = 0;
let _fallbackBinEnd = 0;

export function registerAudioMotion(instance: any): void {
  try {
    const ctx: AudioContext = instance.audioCtx;
    _fallbackAnalyser = ctx.createAnalyser();
    _fallbackAnalyser.fftSize = 2048;
    _fallbackAnalyser.smoothingTimeConstant = 0.15;
    const sources = instance.connectedSources;
    if (sources?.[0]) sources[0].connect(_fallbackAnalyser);
    _fallbackFreqData = new Uint8Array(_fallbackAnalyser.frequencyBinCount);
    const binHz = ctx.sampleRate / _fallbackAnalyser.fftSize;
    _fallbackBinStart = Math.floor(50 / binHz);
    _fallbackBinEnd = Math.ceil(180 / binHz);
  } catch { /* non-fatal */ }
}

export function unregisterAudioMotion(): void {
  _fallbackAnalyser = null;
  _fallbackFreqData = null;
}

function readFallbackEnergy(): number {
  if (!_fallbackAnalyser || !_fallbackFreqData) return 0;
  _fallbackAnalyser.getByteFrequencyData(_fallbackFreqData);
  let sum = 0, count = 0;
  for (let i = _fallbackBinStart; i <= _fallbackBinEnd && i < _fallbackFreqData.length; i++) {
    sum += _fallbackFreqData[i]; count++;
  }
  return count > 0 ? (sum / count) / 255 : 0;
}

// ── State ────────────────────────────────────────────────────────────────────

const prefs = loadDiscoPrefs();

let _state: DiscoState = {
  discoMode: prefs.discoMode,
  kickPulse: 0,
  snarePulse: 0,
  hihatEnergy: 0,
  discoDataLoaded: false,
};

// ── Reactivity (useSyncExternalStore) ────────────────────────────────────────

const _listeners = new Set<() => void>();
function notify(): void { _listeners.forEach(cb => cb()); }
function setState(updates: Partial<DiscoState>): void {
  _state = { ..._state, ...updates };
  notify();
}
function subscribe(cb: () => void): () => void {
  _listeners.add(cb);
  return () => _listeners.delete(cb);
}

// ── Beat Detection Loop ──────────────────────────────────────────────────────

let _rafId: number | null = null;
let _debugFrameCount = 0;

// Per-stem pulse accumulators
let _kickPulse = 0;
let _snarePulse = 0;
let _hihatSmoothed = 0;

// Kick: threshold detection
const KICK_THRESHOLD = 0.12;
const KICK_ATTACK = 0.7;
const KICK_DECAY = 0.88;

// Snare: threshold detection (snappier than kick)
const SNARE_THRESHOLD = 0.10;
const SNARE_ATTACK = 0.8;
const SNARE_DECAY = 0.82;

// Hi-hat: continuous energy (smoothed for particle spawning)
const HIHAT_SMOOTH = 0.3;

// Fallback mode: adaptive normalisation
let _rollingMin = 1.0;
let _rollingMax = 0.0;
const FB_MIN_DECAY = 0.002;
const FB_MAX_DECAY = 0.005;
const FB_MIN_RANGE = 0.03;
const FB_ATTACK = 0.6;
const FB_DECAY = 0.85;

function beatDetectionLoop(): void {
  if (!_state.discoMode) {
    _rafId = null;
    _kickPulse = 0;
    _snarePulse = 0;
    _hihatSmoothed = 0;
    setState({ kickPulse: 0, snarePulse: 0, hihatEnergy: 0 });
    return;
  }

  const t = _mainCurrentTime;
  const fps = _discoData?.fps || 60;

  // ── Kick ──
  const kickEnergy = _discoData ? getEnergyAtTime(_discoData.kick, fps, t) : -1;
  const usingDiscoData = kickEnergy >= 0;

  if (usingDiscoData) {
    if (kickEnergy > KICK_THRESHOLD) {
      const target = Math.min(1.0, (kickEnergy - KICK_THRESHOLD) / (1 - KICK_THRESHOLD));
      _kickPulse += (target - _kickPulse) * KICK_ATTACK;
    } else {
      _kickPulse *= KICK_DECAY;
    }
  } else {
    // Fallback: use full mix for kick detection
    const raw = readFallbackEnergy();
    if (raw < _rollingMin) _rollingMin = raw; else _rollingMin += FB_MIN_DECAY;
    if (raw > _rollingMax) _rollingMax = raw; else _rollingMax -= FB_MAX_DECAY;
    if (_rollingMax - _rollingMin < FB_MIN_RANGE) {
      const mid = (_rollingMax + _rollingMin) / 2;
      _rollingMin = mid - FB_MIN_RANGE / 2;
      _rollingMax = mid + FB_MIN_RANGE / 2;
    }
    const norm = Math.max(0, Math.min(1, (raw - _rollingMin) / (_rollingMax - _rollingMin)));
    if (norm > _kickPulse) _kickPulse += (norm - _kickPulse) * FB_ATTACK;
    else _kickPulse *= FB_DECAY;
  }

  // ── Snare (disco data only) ──
  if (_discoData) {
    const snareEnergy = getEnergyAtTime(_discoData.snare, fps, t);
    if (snareEnergy >= 0) {
      if (snareEnergy > SNARE_THRESHOLD) {
        const target = Math.min(1.0, (snareEnergy - SNARE_THRESHOLD) / (1 - SNARE_THRESHOLD));
        _snarePulse += (target - _snarePulse) * SNARE_ATTACK;
      } else {
        _snarePulse *= SNARE_DECAY;
      }
    }
  }

  // ── Hi-Hat (disco data only) ──
  if (_discoData) {
    const hihatEnergy = getEnergyAtTime(_discoData.hihat, fps, t);
    if (hihatEnergy >= 0) {
      _hihatSmoothed = _hihatSmoothed * HIHAT_SMOOTH + hihatEnergy * (1 - HIHAT_SMOOTH);
    }
  }

  // Clamp tiny values
  if (_kickPulse < 0.01) _kickPulse = 0;
  if (_snarePulse < 0.01) _snarePulse = 0;
  if (_hihatSmoothed < 0.01) _hihatSmoothed = 0;

  // Debug: log twice per second
  _debugFrameCount++;
  if (_debugFrameCount % 30 === 0) {
    const mode = usingDiscoData ? 'DISCO' : 'FALLBACK';
    console.log(`[Disco] [${mode}] t=${t.toFixed(2)}s kick=${_kickPulse.toFixed(3)} snare=${_snarePulse.toFixed(3)} hihat=${_hihatSmoothed.toFixed(3)}`);
  }

  // Notify subscribers if anything changed
  const kickChanged = Math.abs(_kickPulse - _state.kickPulse) > 0.003;
  const snareChanged = Math.abs(_snarePulse - _state.snarePulse) > 0.003;
  const hihatChanged = Math.abs(_hihatSmoothed - _state.hihatEnergy) > 0.003;

  if (kickChanged || snareChanged || hihatChanged) {
    _state = {
      ..._state,
      kickPulse: _kickPulse,
      snarePulse: _snarePulse,
      hihatEnergy: _hihatSmoothed,
    };
    notify();
  }

  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function startLoop(): void {
  if (_rafId !== null) return;
  _rollingMin = 1.0;
  _rollingMax = 0.0;
  _kickPulse = 0;
  _snarePulse = 0;
  _hihatSmoothed = 0;
  _debugFrameCount = 0;
  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function stopLoop(): void {
  if (_rafId !== null) {
    cancelAnimationFrame(_rafId);
    _rafId = null;
  }
  _kickPulse = 0;
  _snarePulse = 0;
  _hihatSmoothed = 0;
  setState({ kickPulse: 0, snarePulse: 0, hihatEnergy: 0 });
}

// ── Public Mode API ──────────────────────────────────────────────────────────

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
  if (_state.discoMode && isPlaying) startLoop();
  else if (!isPlaying) stopLoop();
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

/** Legacy alias — returns kickPulse */
export function usePulseIntensity(): number {
  return useDiscoSelector(s => s.kickPulse);
}

export function useDiscoMode(): boolean {
  return useDiscoSelector(s => s.discoMode);
}

export function useDiscoDataLoaded(): boolean {
  return useDiscoSelector(s => s.discoDataLoaded);
}

/** @deprecated Use useDiscoDataLoaded instead */
export function useKickStemLoaded(): boolean {
  return useDiscoDataLoaded();
}

export function useSnarePulse(): number {
  return useDiscoSelector(s => s.snarePulse);
}

export function useHihatEnergy(): number {
  return useDiscoSelector(s => s.hihatEnergy);
}
