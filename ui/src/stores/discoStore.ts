// discoStore.ts — Disco mode state + beat detection engine.
//
// Two modes of beat detection:
// 1. KICK STEM (preferred) — plays an isolated kick drum WAV silently via a
//    hidden Audio element, connected to a dedicated AnalyserNode. The signal
//    is near-silence between hits and spikes sharply on kicks. Simple threshold.
// 2. FULL MIX FALLBACK — reads from audioMotion's AnalyserNode via the shared
//    audio graph. Uses adaptive normalisation to extract dynamics from a noisy
//    signal. Works but much less reactive.

import { useSyncExternalStore, useRef, useCallback } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

interface DiscoState {
  discoMode: boolean;
  pulseIntensity: number;  // 0.0–1.0
  kickStemLoaded: boolean; // true when using isolated kick stem (not fallback)
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

// ── Audio: Kick Stem (preferred path) ────────────────────────────────────────

let _kickAudio: HTMLAudioElement | null = null;
let _kickCtx: AudioContext | null = null;
let _kickAnalyser: AnalyserNode | null = null;
let _kickSource: MediaElementAudioSourceNode | null = null;
let _kickFreqData: Uint8Array | null = null;
let _kickStemUrl: string = '';
let _kickBinStart = 0;
let _kickBinEnd = 0;
let _mainIsPlaying = false;  // Track main player state for auto-play on stem load
let _mainCurrentTime = 0;

/** Load a kick stem URL for beat detection. Pass '' to unload. */
export function setKickStemUrl(url: string): void {
  // Skip if already loaded
  if (url === _kickStemUrl) return;
  _kickStemUrl = url;

  if (!url || url.startsWith('extracting:')) {
    // Unload
    _kickAudio?.pause();
    _kickAudio = null;
    _kickSource = null;  // Can't disconnect — MediaElementSource is permanent
    _kickAnalyser = null;
    _kickFreqData = null;
    setState({ kickStemLoaded: false });
    console.log('[Disco] Kick stem unloaded');
    return;
  }

  // Create or reuse Audio element
  if (!_kickAudio) {
    _kickAudio = new Audio();
    _kickAudio.crossOrigin = 'anonymous';
    // Volume must be non-zero for MediaElementSource to send signal
    // in all browsers. We don't connect to ctx.destination so it's inaudible.
    _kickAudio.volume = 1;
    _kickAudio.preload = 'auto';
  }

  _kickAudio.src = url;
  _kickAudio.load();

  // Create AudioContext + AnalyserNode on first use
  if (!_kickCtx) {
    _kickCtx = new AudioContext();
  }

  // Connect audio element to analyser (only once — MediaElementSource is permanent)
  if (!_kickSource) {
    _kickSource = _kickCtx.createMediaElementSource(_kickAudio);
    _kickAnalyser = _kickCtx.createAnalyser();
    _kickAnalyser.fftSize = 2048;
    _kickAnalyser.smoothingTimeConstant = 0.1; // Very low — we want raw transients
    _kickSource.connect(_kickAnalyser);
    // DON'T connect to ctx.destination — we don't want to hear the kick stem
    // (the main WaveSurfer already plays the full mix)

    _kickFreqData = new Uint8Array(_kickAnalyser.frequencyBinCount);

    // Compute FFT bin indices for kick drum range (40–200Hz)
    const binHz = _kickCtx.sampleRate / _kickAnalyser.fftSize;
    _kickBinStart = Math.floor(40 / binHz);
    _kickBinEnd = Math.ceil(200 / binHz);
    console.log(`[Disco] Kick analyser: bins ${_kickBinStart}-${_kickBinEnd} (${(binHz * _kickBinStart).toFixed(0)}-${(binHz * _kickBinEnd).toFixed(0)}Hz)`);
  }

  setState({ kickStemLoaded: true });
  console.log(`[Disco] Kick stem loaded: ${url}`);

  // If the main player is already playing, auto-start the kick stem
  // (handles the case where stem loads after playback has already started)
  if (_mainIsPlaying) {
    console.log(`[Disco] Main player already playing — auto-starting kick stem at ${_mainCurrentTime.toFixed(1)}s`);
    _kickAudio.currentTime = _mainCurrentTime;
    if (_kickCtx?.state === 'suspended') _kickCtx.resume();
    _kickAudio.play().catch(() => {});
  }
}

/** Sync kick stem playback with main player */
export function syncKickStem(action: 'play' | 'pause' | 'seek', time?: number): void {
  // Track main player state so we can auto-start on stem load
  if (action === 'play') {
    _mainIsPlaying = true;
    if (time !== undefined) _mainCurrentTime = time;
  } else if (action === 'pause') {
    _mainIsPlaying = false;
  } else if (action === 'seek' && time !== undefined) {
    _mainCurrentTime = time;
  }

  if (!_kickAudio || !_kickStemUrl) return;

  // Resume AudioContext on first interaction (Chrome autoplay policy)
  if (_kickCtx?.state === 'suspended') {
    _kickCtx.resume();
  }

  switch (action) {
    case 'play':
      if (time !== undefined) _kickAudio.currentTime = time;
      _kickAudio.play().catch(() => {}); // Ignore autoplay errors
      break;
    case 'pause':
      _kickAudio.pause();
      break;
    case 'seek':
      if (time !== undefined) _kickAudio.currentTime = time;
      break;
  }
}

function readKickStemEnergy(): number {
  if (!_kickAnalyser || !_kickFreqData) return -1; // -1 = not available

  _kickAnalyser.getByteFrequencyData(_kickFreqData);

  let sum = 0;
  let count = 0;
  for (let i = _kickBinStart; i <= _kickBinEnd && i < _kickFreqData.length; i++) {
    sum += _kickFreqData[i];
    count++;
  }
  return count > 0 ? (sum / count) / 255 : 0;
}

// ── Audio: Full Mix Fallback ─────────────────────────────────────────────────

let _audioMotion: any = null;
let _fallbackAnalyser: AnalyserNode | null = null;
let _fallbackFreqData: Uint8Array | null = null;
let _fallbackBinStart = 0;
let _fallbackBinEnd = 0;

export function registerAudioMotion(instance: any): void {
  _audioMotion = instance;
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
  _audioMotion = null;
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
  pulseIntensity: 0,
  kickStemLoaded: false,
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
function getSnapshot(): DiscoState { return _state; }

// ── Beat Detection Loop ──────────────────────────────────────────────────────

let _rafId: number | null = null;
let _debugFrameCount = 0;
let _pulse = 0;

// Kick stem mode: simple threshold (the signal is clean!)
const KICK_THRESHOLD = 0.12;   // Energy above this = kick hit
const KICK_ATTACK = 0.7;       // How fast pulse rises (0-1, higher = snappier)
const KICK_DECAY = 0.88;       // Per-frame decay (~180ms to near-zero at 60fps)

// Fallback mode: adaptive normalisation (same as before)
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
    _pulse = 0;
    setState({ pulseIntensity: 0 });
    return;
  }

  // Try kick stem first, fall back to full mix
  const kickEnergy = readKickStemEnergy();
  const usingKick = kickEnergy >= 0;

  if (usingKick) {
    // ── KICK STEM MODE: simple threshold detection ──
    if (kickEnergy > KICK_THRESHOLD) {
      const target = Math.min(1.0, (kickEnergy - KICK_THRESHOLD) / (1 - KICK_THRESHOLD));
      _pulse += (target - _pulse) * KICK_ATTACK;
    } else {
      _pulse *= KICK_DECAY;
    }
  } else {
    // ── FALLBACK MODE: adaptive normalisation ──
    const raw = readFallbackEnergy();
    if (raw < _rollingMin) _rollingMin = raw; else _rollingMin += FB_MIN_DECAY;
    if (raw > _rollingMax) _rollingMax = raw; else _rollingMax -= FB_MAX_DECAY;
    if (_rollingMax - _rollingMin < FB_MIN_RANGE) {
      const mid = (_rollingMax + _rollingMin) / 2;
      _rollingMin = mid - FB_MIN_RANGE / 2;
      _rollingMax = mid + FB_MIN_RANGE / 2;
    }
    const norm = Math.max(0, Math.min(1, (raw - _rollingMin) / (_rollingMax - _rollingMin)));
    if (norm > _pulse) _pulse += (norm - _pulse) * FB_ATTACK;
    else _pulse *= FB_DECAY;
  }

  if (_pulse < 0.01) _pulse = 0;

  // Debug: log twice per second
  _debugFrameCount++;
  if (_debugFrameCount % 30 === 0) {
    const mode = usingKick ? 'KICK' : 'FALLBACK';
    const energy = usingKick ? kickEnergy : readFallbackEnergy();
    console.log(`[Disco] [${mode}] energy=${energy.toFixed(3)} pulse=${_pulse.toFixed(3)}`);
  }

  // Notify subscribers
  if (Math.abs(_pulse - _state.pulseIntensity) > 0.003) {
    _state = { ..._state, pulseIntensity: _pulse };
    notify();
  }

  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function startLoop(): void {
  if (_rafId !== null) return;
  _rollingMin = 1.0;
  _rollingMax = 0.0;
  _pulse = 0;
  _debugFrameCount = 0;
  _rafId = requestAnimationFrame(beatDetectionLoop);
}

function stopLoop(): void {
  if (_rafId !== null) {
    cancelAnimationFrame(_rafId);
    _rafId = null;
  }
  _pulse = 0;
  setState({ pulseIntensity: 0 });
  // Pause kick stem when stopping
  syncKickStem('pause');
}

// ── Public API ───────────────────────────────────────────────────────────────

export function setDiscoMode(on: boolean): void {
  setState({ discoMode: on });
  saveDiscoPrefs(on);
  if (on) startLoop(); else stopLoop();
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

export function usePulseIntensity(): number {
  return useDiscoSelector(s => s.pulseIntensity);
}

export function useDiscoMode(): boolean {
  return useDiscoSelector(s => s.discoMode);
}

export function useKickStemLoaded(): boolean {
  return useDiscoSelector(s => s.kickStemLoaded);
}
