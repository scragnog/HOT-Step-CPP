// discoStore.ts — Disco mode state + multi-stem beat detection engine.
//
// Three stem channels, each with its own HTMLAudioElement + AnalyserNode:
//   1. KICK   — punch/impact → scale pulse on panels
//   2. SNARE  — crack/flash → full-viewport white flash
//   3. HI-HAT — shimmer → floating particles
//
// Fallback: if no stems available, uses audioMotion's AnalyserNode for
// basic kick detection from the full mix.

import { useSyncExternalStore, useRef, useCallback } from 'react';

// ── Types ────────────────────────────────────────────────────────────────────

interface DiscoState {
  discoMode: boolean;
  kickPulse: number;     // 0.0–1.0 (legacy: also exposed as pulseIntensity)
  snarePulse: number;    // 0.0–1.0
  hihatEnergy: number;   // 0.0–1.0 (continuous energy, not pulsed)
  kickStemLoaded: boolean;
  snareStemLoaded: boolean;
  hihatStemLoaded: boolean;
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

// ── Audio: Multi-Stem Channels ───────────────────────────────────────────────

interface StemChannel {
  audio: HTMLAudioElement | null;
  analyser: AnalyserNode | null;
  source: MediaElementAudioSourceNode | null;
  freqData: Uint8Array | null;
  url: string;
  binStart: number;
  binEnd: number;
  loaded: boolean;
}

// Shared AudioContext for all stem channels
let _stemCtx: AudioContext | null = null;

// Track main player state for auto-play on stem load
let _mainIsPlaying = false;
let _mainCurrentTime = 0;

// Frequency ranges for each stem type
const STEM_FREQ_RANGES = {
  kick:  { lo: 40,   hi: 200  },
  snare: { lo: 200,  hi: 2000 },
  hihat: { lo: 6000, hi: 16000 },
} as const;

function createChannel(): StemChannel {
  return {
    audio: null, analyser: null, source: null, freqData: null,
    url: '', binStart: 0, binEnd: 0, loaded: false,
  };
}

const _stems: Record<'kick' | 'snare' | 'hihat', StemChannel> = {
  kick:  createChannel(),
  snare: createChannel(),
  hihat: createChannel(),
};

type StemKey = keyof typeof _stems;

function ensureContext(): AudioContext {
  if (!_stemCtx) _stemCtx = new AudioContext();
  return _stemCtx;
}

/** Load a stem URL for a specific channel. Pass '' to unload. */
function loadStemChannel(key: StemKey, url: string): void {
  const ch = _stems[key];
  if (url === ch.url) return;
  ch.url = url;

  if (!url || url.startsWith('extracting:')) {
    ch.audio?.pause();
    ch.audio = null;
    ch.source = null;
    ch.analyser = null;
    ch.freqData = null;
    ch.loaded = false;
    updateStemLoadedState();
    console.log(`[Disco] ${key} stem unloaded`);
    return;
  }

  // Create or reuse Audio element
  if (!ch.audio) {
    ch.audio = new Audio();
    ch.audio.crossOrigin = 'anonymous';
    ch.audio.volume = 1; // Must be non-zero for signal
    ch.audio.preload = 'auto';
  }

  ch.audio.src = url;
  ch.audio.load();

  const ctx = ensureContext();

  // Connect audio → analyser (only once — MediaElementSource is permanent)
  if (!ch.source) {
    ch.source = ctx.createMediaElementSource(ch.audio);
    ch.analyser = ctx.createAnalyser();
    ch.analyser.fftSize = 2048;
    ch.analyser.smoothingTimeConstant = key === 'hihat' ? 0.3 : 0.1;
    ch.source.connect(ch.analyser);
    // Don't connect to ctx.destination — inaudible analysis only

    ch.freqData = new Uint8Array(ch.analyser.frequencyBinCount);

    const binHz = ctx.sampleRate / ch.analyser.fftSize;
    const range = STEM_FREQ_RANGES[key];
    ch.binStart = Math.floor(range.lo / binHz);
    ch.binEnd = Math.ceil(range.hi / binHz);
    console.log(`[Disco] ${key} analyser: bins ${ch.binStart}-${ch.binEnd} (${(binHz * ch.binStart).toFixed(0)}-${(binHz * ch.binEnd).toFixed(0)}Hz)`);
  }

  ch.loaded = true;
  updateStemLoadedState();
  console.log(`[Disco] ${key} stem loaded: ${url}`);

  // Auto-start if main player is already playing
  if (_mainIsPlaying) {
    console.log(`[Disco] Main player already playing — auto-starting ${key} stem at ${_mainCurrentTime.toFixed(1)}s`);
    ch.audio.currentTime = _mainCurrentTime;
    if (_stemCtx?.state === 'suspended') _stemCtx.resume();
    ch.audio.play().catch(() => {});
  }
}

function updateStemLoadedState(): void {
  setState({
    kickStemLoaded: _stems.kick.loaded,
    snareStemLoaded: _stems.snare.loaded,
    hihatStemLoaded: _stems.hihat.loaded,
  });
}

function readStemEnergy(key: StemKey): number {
  const ch = _stems[key];
  if (!ch.analyser || !ch.freqData) return -1;

  ch.analyser.getByteFrequencyData(ch.freqData);

  let sum = 0, count = 0;
  for (let i = ch.binStart; i <= ch.binEnd && i < ch.freqData.length; i++) {
    sum += ch.freqData[i];
    count++;
  }
  return count > 0 ? (sum / count) / 255 : 0;
}

// ── Public Stem API ──────────────────────────────────────────────────────────

/** Load all drum stem URLs at once. Pass '' for any stem to unload it. */
export function setStemUrls(urls: { kick?: string; snare?: string; hihat?: string }): void {
  if (urls.kick !== undefined) loadStemChannel('kick', urls.kick);
  if (urls.snare !== undefined) loadStemChannel('snare', urls.snare);
  if (urls.hihat !== undefined) loadStemChannel('hihat', urls.hihat);
}

/** Legacy: load just the kick stem URL */
export function setKickStemUrl(url: string): void {
  loadStemChannel('kick', url);
}

/** Sync all stem playback with main player */
export function syncStems(action: 'play' | 'pause' | 'seek', time?: number): void {
  // Track main player state
  if (action === 'play') {
    _mainIsPlaying = true;
    if (time !== undefined) _mainCurrentTime = time;
  } else if (action === 'pause') {
    _mainIsPlaying = false;
  } else if (action === 'seek' && time !== undefined) {
    _mainCurrentTime = time;
  }

  // Resume AudioContext on first interaction
  if (_stemCtx?.state === 'suspended') _stemCtx.resume();

  for (const key of ['kick', 'snare', 'hihat'] as StemKey[]) {
    const ch = _stems[key];
    if (!ch.audio || !ch.url) continue;

    switch (action) {
      case 'play':
        if (time !== undefined) ch.audio.currentTime = time;
        ch.audio.play().catch(() => {});
        break;
      case 'pause':
        ch.audio.pause();
        break;
      case 'seek':
        if (time !== undefined) ch.audio.currentTime = time;
        break;
    }
  }
}

/** Legacy alias */
export function syncKickStem(action: 'play' | 'pause' | 'seek', time?: number): void {
  syncStems(action, time);
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
  kickPulse: 0,
  snarePulse: 0,
  hihatEnergy: 0,
  kickStemLoaded: false,
  snareStemLoaded: false,
  hihatStemLoaded: false,
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
const SNARE_DECAY = 0.82;  // Faster decay — snare is crackly

// Hi-hat: continuous energy (smoothed for particle spawning)
const HIHAT_SMOOTH = 0.3;  // Smoothing factor (0=no smooth, 1=frozen)

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

  // ── Kick ──
  const kickEnergy = readStemEnergy('kick');
  const usingKick = kickEnergy >= 0;

  if (usingKick) {
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

  // ── Snare ──
  const snareEnergy = readStemEnergy('snare');
  if (snareEnergy >= 0) {
    if (snareEnergy > SNARE_THRESHOLD) {
      const target = Math.min(1.0, (snareEnergy - SNARE_THRESHOLD) / (1 - SNARE_THRESHOLD));
      _snarePulse += (target - _snarePulse) * SNARE_ATTACK;
    } else {
      _snarePulse *= SNARE_DECAY;
    }
  }

  // ── Hi-Hat ──
  const hihatEnergy = readStemEnergy('hihat');
  if (hihatEnergy >= 0) {
    _hihatSmoothed = _hihatSmoothed * HIHAT_SMOOTH + hihatEnergy * (1 - HIHAT_SMOOTH);
  }

  // Clamp tiny values to zero
  if (_kickPulse < 0.01) _kickPulse = 0;
  if (_snarePulse < 0.01) _snarePulse = 0;
  if (_hihatSmoothed < 0.01) _hihatSmoothed = 0;

  // Debug: log twice per second
  _debugFrameCount++;
  if (_debugFrameCount % 30 === 0) {
    const mode = usingKick ? 'STEMS' : 'FALLBACK';
    console.log(`[Disco] [${mode}] kick=${_kickPulse.toFixed(3)} snare=${_snarePulse.toFixed(3)} hihat=${_hihatSmoothed.toFixed(3)}`);
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
  syncStems('pause');
}

// ── Public API ───────────────────────────────────────────────────────────────

export function setDiscoMode(on: boolean): void {
  setState({ discoMode: on });
  saveDiscoPrefs(on);
  if (on) {
    startLoop();
    // If main player is already playing, resume stems so beat detection gets signal
    if (_mainIsPlaying) {
      syncStems('play', _mainCurrentTime);
    }
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

export function useKickStemLoaded(): boolean {
  return useDiscoSelector(s => s.kickStemLoaded);
}

export function useSnarePulse(): number {
  return useDiscoSelector(s => s.snarePulse);
}

export function useHihatEnergy(): number {
  return useDiscoSelector(s => s.hihatEnergy);
}
