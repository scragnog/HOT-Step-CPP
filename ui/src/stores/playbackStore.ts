/**
 * playbackStore.ts — Unified playback state for the entire application.
 *
 * Single source of truth for: current track, track list context, transport
 * state (play/pause/seek/next/prev), volume, shuffle, repeat, mastered toggle,
 * and buffering/error states.
 *
 * Orchestrates two WaveSurfer instances (original + mastered) via registered
 * imperative handles — App.tsx renders the DOM, this store controls behavior.
 *
 * Uses useSyncExternalStore for React reactivity (same pattern as playlistStore).
 * Persists preferences + track list to localStorage.
 */

import { useSyncExternalStore, useRef, useCallback } from 'react';
import type { WaveformPlayerHandle } from '../components/player/WaveformPlayer';
import { useVstChainStore } from './vstChainStore';
import {
  mm3StreamSnapshot, mm3StreamSubscribe, mm3StreamToggle, mm3StreamPause, mm3StreamSelect, mm3StreamPlay,
  mm3StreamSeek, mm3StreamSetVolume,
} from './mm3StreamStore';

/** VST monitoring and the global player are mutually exclusive — they'd play
 *  two tracks at once. Whenever global playback starts, stop the VST monitor. */
function stopVstMonitorIfActive(): void {
  try {
    const vst = useVstChainStore.getState();
    if (vst.monitoring) vst.stopMonitor();
  } catch { /* vst store not initialized yet */ }
}

// ── The stream deck ─────────────────────────────────────────────────────────
//
// A fourth deck alongside original / mastered / no-adapter, except that its
// audio is not a file and not a media element: it is a Web Audio graph fed by
// mm3StreamStore as the engine emits finished windows. Everything below that
// touches playback asks `isStreamDeck()` first and delegates.
//
// It is a delegation and not an abstraction on purpose. Making WaveSurfer play a
// track that has no URL, no length and no seekable end would mean fighting every
// assumption it is built on; four `if` statements in the transport is the whole
// cost of not doing that.

function isStreamDeck(): boolean {
  return !!_state.currentTrack?.streamJobId;
}

// ── Types ────────────────────────────────────────────────────────────────────

/** Canonical track representation — normalized, always number duration */
export interface PlaybackTrack {
  id: string;
  title: string;
  audioUrl: string;
  masteredAudioUrl?: string;
  /** No-adapter reference render (bare-DiT low-step output, unprocessed) */
  noAdapterAudioUrl?: string;
  kickStemUrl?: string;
  snareStemUrl?: string;
  hihatStemUrl?: string;
  discoDataUrl?: string;
  artistName?: string;
  coverUrl?: string;
  duration?: number;                  // Always seconds, never string
  style?: string;
  lyrics?: string;
  caption?: string;
  generationParams?: Record<string, any>;
  /** MiniMax-Music3 live render. When set, this track's audio does not exist as
   *  a file yet — it is arriving window by window into mm3StreamStore, and the
   *  transport below delegates there instead of to the WaveSurfer decks. The
   *  track becomes an ordinary one the moment the render is saved. */
  streamJobId?: string;
  streamTake?: number;
}

export type PlaybackSource =
  | 'library'
  | 'playlist'
  | 'lireek-recent'
  | 'lireek-recordings'
  | 'lireek-queue'
  | 'cover-studio'
  | 'direct';

export interface PlaybackState {
  // ── Track Navigation ──
  currentTrack: PlaybackTrack | null;
  trackList: PlaybackTrack[];
  trackIndex: number;
  source: PlaybackSource;

  // ── Playback Status ──
  isPlaying: boolean;
  isLoading: boolean;
  loadError: string | null;
  currentTime: number;
  duration: number;

  // ── Player Bar Visibility ──
  // True when a track has been loaded for playback. Stays true when paused.
  // Only set to false by an explicit stop() call, which collapses the player bar.
  playerActive: boolean;

  // ── Variant Switch (no-adapter / original / mastered) ──
  playMastered: boolean;
  hasMastered: boolean;
  /** Third deck: no-adapter reference render. Never both playMastered and
   *  playNoAdapter — 'original' is the state where both are false. */
  playNoAdapter: boolean;
  hasNoAdapter: boolean;
  currentAudioUrl: string | null;

  // ── Preferences (persisted) ──
  volume: number;
  shuffle: boolean;
  repeat: 'none' | 'all' | 'one';
  playbackRate: number;
  spectrumEnabled: boolean;

  // ── Trim Mode ──
  trimMode: boolean;
  trimInPoint: number | null;
  trimOutPoint: number | null;
  trimClickCount: number;  // 0 = waiting for IN, 1 = waiting for OUT, 2 = both set

  // ── A/B Comparison Mode ──
  abMode: boolean;
  abTrackA: PlaybackTrack | null;
  abTrackB: PlaybackTrack | null;
}

// ── Track Converters (delegated to playbackConverters.ts) ────────────────────
export {
  songToTrack,
  playlistItemToTrack,
  recentSongToTrack,
  unifiedRecentSongToTrack,
  audioQueueItemToTrack,
} from './playbackConverters';

// ── localStorage Keys ────────────────────────────────────────────────────────

const PREFS_KEY = 'playback-prefs';
const TRACKLIST_KEY = 'playback-tracklist';
const MAX_PERSISTED_TRACKS = 50;

// ── Persisted Preferences ────────────────────────────────────────────────────

interface PersistedPrefs {
  volume: number;
  shuffle: boolean;
  repeat: 'none' | 'all' | 'one';
  playbackRate: number;
  spectrumEnabled: boolean;
}

function loadPrefs(): PersistedPrefs {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) return { ...DEFAULT_PREFS, ...JSON.parse(raw) };
  } catch { /* ignore */ }
  // Migrate legacy volume key
  const legacyVol = localStorage.getItem('volume');
  if (legacyVol) {
    const v = parseFloat(legacyVol);
    if (!isNaN(v)) return { ...DEFAULT_PREFS, volume: v };
  }
  return DEFAULT_PREFS;
}

function savePrefs(p: PersistedPrefs): void {
  localStorage.setItem(PREFS_KEY, JSON.stringify(p));
  // Keep legacy key in sync for any remaining consumers
  localStorage.setItem('volume', String(p.volume));
}

const DEFAULT_PREFS: PersistedPrefs = {
  volume: 0.8,
  shuffle: false,
  repeat: 'none',
  playbackRate: 1.0,
  spectrumEnabled: false,
};

// ── Persisted Track List ─────────────────────────────────────────────────────

interface PersistedTrackList {
  trackList: PlaybackTrack[];
  trackIndex: number;
  source: PlaybackSource;
}

function loadTrackList(): PersistedTrackList | null {
  try {
    const raw = localStorage.getItem(TRACKLIST_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.trackList)) return parsed;
  } catch { /* ignore */ }
  return null;
}

function saveTrackList(tl: PersistedTrackList): void {
  // Strip heavy text fields (lyrics/caption/generationParams) — URLs are cheap,
  // and a restored list only needs to be playable, not fully hydrated.
  const slim: PlaybackTrack[] = tl.trackList.slice(0, MAX_PERSISTED_TRACKS).map(t => ({
    id: t.id, title: t.title, audioUrl: t.audioUrl,
    masteredAudioUrl: t.masteredAudioUrl,
    noAdapterAudioUrl: t.noAdapterAudioUrl, kickStemUrl: t.kickStemUrl,
    snareStemUrl: t.snareStemUrl, hihatStemUrl: t.hihatStemUrl,
    discoDataUrl: t.discoDataUrl, artistName: t.artistName,
    coverUrl: t.coverUrl, duration: t.duration, style: t.style,
  }));
  try {
    localStorage.setItem(TRACKLIST_KEY, JSON.stringify({ ...tl, trackList: slim }));
  } catch {
    // QuotaExceededError — clear the key and retry with just the current track
    try {
      localStorage.removeItem(TRACKLIST_KEY);
      localStorage.setItem(TRACKLIST_KEY, JSON.stringify({ ...tl, trackList: slim.slice(0, 1) }));
    } catch { /* give up gracefully — playback still works, just won't persist */ }
  }
}

// ── WaveSurfer Handle Registration ───────────────────────────────────────────

// Store ref OBJECTS (not .current values) so we always dereference at call time.
// This is critical because useImperativeHandle may not have committed yet when
// the useEffect in App.tsx runs registerPlayers.
type WsRef = { current: WaveformPlayerHandle | null };

let _wsOriginalRef: WsRef = { current: null };
let _wsAltRef: WsRef = { current: null };
let _wsNoAdapterRef: WsRef = { current: null };

export function registerPlayers(
  orig: WsRef,
  alt: WsRef,
  noAdapter?: WsRef
): void {
  _wsOriginalRef = orig;
  _wsAltRef = alt;
  if (noAdapter) _wsNoAdapterRef = noAdapter;
}

export function getActiveMediaElement(): HTMLMediaElement | null {
  if (_state.playNoAdapter && _wsNoAdapterRef.current) return _wsNoAdapterRef.current.getMediaElement();
  if (_state.playMastered && _wsAltRef.current) return _wsAltRef.current.getMediaElement();
  if (_wsOriginalRef.current) return _wsOriginalRef.current.getMediaElement();
  return null;
}

// ── State ────────────────────────────────────────────────────────────────────

const prefs = loadPrefs();
const restored = loadTrackList();

let _state: PlaybackState = {
  currentTrack: restored ? restored.trackList[restored.trackIndex] || null : null,
  trackList: restored?.trackList || [],
  trackIndex: restored?.trackIndex || 0,
  source: restored?.source || 'direct',

  isPlaying: false,
  isLoading: false,
  loadError: null,
  currentTime: 0,
  duration: 0,

  playerActive: false,

  playMastered: false,
  hasMastered: false,
  playNoAdapter: false,
  hasNoAdapter: false,
  currentAudioUrl: null,

  trimMode: false,
  trimInPoint: null,
  trimOutPoint: null,
  trimClickCount: 0,

  abMode: false,
  abTrackA: null,
  abTrackB: null,

  ...prefs,
};

// ── Reactivity (useSyncExternalStore) ────────────────────────────────────────

const CHANGE_EVENT = 'playback-state-change';

let _listeners = new Set<() => void>();

function notify(): void {
  _listeners.forEach(cb => cb());
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

function setState(updates: Partial<PlaybackState>): void {
  _state = { ..._state, ...updates };
  notify();
}

function subscribe(cb: () => void): () => void {
  _listeners.add(cb);
  return () => _listeners.delete(cb);
}

function getSnapshot(): PlaybackState {
  return _state;
}

// ── Internal Helpers ─────────────────────────────────────────────────────────

let _retryTimer: ReturnType<typeof setTimeout> | null = null;
let _retryCount = 0;
const MAX_RETRIES = 30;
const RETRY_INTERVAL = 150;

/**
 * When switching from one playing track to another, we suppress the
 * isPlaying→false transition so the visualiser doesn't collapse and
 * re-expand.  Cleared on playback start, failure, or manual pause.
 */
let _suppressPlayFalse = false;

/**
 * Auto-advance past a track that could not be played.
 *
 * These used to be bare setTimeout(() => next(), n) calls that nothing held a
 * handle to, so one scheduled against a track you had already moved on from
 * still fired and skipped a track for no visible reason — and several could be
 * in flight at once. Now: only one is ever pending, it is cancelled when a new
 * track loads, and it checks it is still sitting on the track that failed
 * before doing anything.
 */
let _autoAdvanceTimer: ReturnType<typeof setTimeout> | null = null;

function cancelAutoAdvance(): void {
  if (_autoAdvanceTimer) { clearTimeout(_autoAdvanceTimer); _autoAdvanceTimer = null; }
}

function scheduleAutoAdvance(reason: string, delayMs: number): void {
  cancelAutoAdvance();
  const forTrackId = _state.currentTrack?.id ?? null;
  _autoAdvanceTimer = setTimeout(() => {
    _autoAdvanceTimer = null;
    const nowId = _state.currentTrack?.id ?? null;
    if (nowId !== forTrackId) {
      console.log(`[Playback] auto-advance (${reason}) dropped — track changed since it was scheduled`);
      return;
    }
    next(`auto:${reason}`);
  }, delayMs);
}

/** Attempt to start all loaded players. Retries if the AUDIBLE track isn't ready. */
function startBothPlayers(): void {
  if (_retryTimer) { clearTimeout(_retryTimer); _retryTimer = null; }

  const wsOrig = _wsOriginalRef.current;
  const wsAlt = _wsAltRef.current;
  const wsNoAdapter = _wsNoAdapterRef.current;

  // Start original if ready
  let origReady = false;
  if (wsOrig) {
    const m = wsOrig.getMediaElement();
    if (m && m.readyState >= 2) {
      if (m.paused) wsOrig.play();
      origReady = true;
    }
  }

  // Start alt if ready (mastered track)
  let altReady = false;
  if (wsAlt && _state.hasMastered) {
    const m = wsAlt.getMediaElement();
    if (m && m.readyState >= 2) {
      if (m.paused) wsAlt.play();
      altReady = true;
    }
  }

  // Start no-adapter deck if ready
  let noAdapterReady = false;
  if (wsNoAdapter && _state.hasNoAdapter) {
    const m = wsNoAdapter.getMediaElement();
    if (m && m.readyState >= 2) {
      if (m.paused) wsNoAdapter.play();
      noAdapterReady = true;
    }
  }

  // The AUDIBLE track must be ready before we declare success.
  const audibleReady = _state.playNoAdapter ? noAdapterReady
    : _state.playMastered ? altReady : origReady;

  if (audibleReady) {
    _retryCount = 0;
    _suppressPlayFalse = false;
    // Arm the once-per-track finish guard here rather than in loadTrack. Doing
    // it at load time cleared it before the outgoing player's trailing finish
    // arrived, so that event looked like a fresh track ending and skipped one.
    _lastFinishedTrackId = null;
    setState({ isPlaying: true, isLoading: false, loadError: null });
    return;
  }

  // Audible track not ready — retry
  _retryCount++;
  if (_retryCount <= MAX_RETRIES) {
    _retryTimer = setTimeout(startBothPlayers, RETRY_INTERVAL);
  } else {
    // Mastered/no-adapter track failed but original is ready — fall back to original
    if ((_state.playMastered || _state.playNoAdapter) && origReady) {
      console.warn('[Playback] Selected variant failed to load, falling back to original');
      _retryCount = 0;
      _suppressPlayFalse = false;
      setState({
        playMastered: false,
        playNoAdapter: false,
        currentAudioUrl: _state.currentTrack?.audioUrl || null,
        isPlaying: true,
        isLoading: false,
        loadError: null,
      });
      applyVolumes();
      return;
    }

    // Total failure — auto-advance if we're in a multi-track context
    _retryCount = 0;
    _suppressPlayFalse = false;
    console.error(
      `[Playback] gave up waiting for "${_state.currentTrack?.title}" after ${MAX_RETRIES * RETRY_INTERVAL}ms`
      + ` (playMastered=${_state.playMastered} playNoAdapter=${_state.playNoAdapter} origReady=${origReady} altReady=${altReady})`
    );
    setState({
      isPlaying: false,
      isLoading: false,
      loadError: 'Audio failed to load. The file may be missing or inaccessible.',
    });

    // Auto-advance to next track after a short delay (like a skip)
    if (_state.trackList.length > 1) {
      scheduleAutoAdvance('load-timeout', 500);
    }
  }
}

function applyVolumes(): void {
  if (isStreamDeck()) { mm3StreamSetVolume(_state.volume); return; }
  const audible = _state.playNoAdapter ? 'noadapter' : _state.playMastered ? 'mastered' : 'original';
  _wsOriginalRef.current?.setVolume(audible === 'original' ? _state.volume : 0);
  _wsAltRef.current?.setVolume(audible === 'mastered' ? _state.volume : 0);
  _wsNoAdapterRef.current?.setVolume(audible === 'noadapter' ? _state.volume : 0);
}

function applyPlaybackRate(): void {
  _wsOriginalRef.current?.setPlaybackRate(_state.playbackRate);
  _wsAltRef.current?.setPlaybackRate(_state.playbackRate);
  _wsNoAdapterRef.current?.setPlaybackRate(_state.playbackRate);
}

function persistTrackList(): void {
  saveTrackList({
    trackList: _state.trackList,
    trackIndex: _state.trackIndex,
    source: _state.source,
  });
}

function persistPrefs(): void {
  savePrefs({
    volume: _state.volume,
    shuffle: _state.shuffle,
    repeat: _state.repeat,
    playbackRate: _state.playbackRate,
    spectrumEnabled: _state.spectrumEnabled,
  });
}

// Track the ID being loaded to prevent stale ready callbacks
let _loadingTrackId: string | null = null;

/** Core play logic — loads audio into WaveSurfer instances */
function loadTrack(track: PlaybackTrack): void {
  // Auto-exit A/B mode when playing a regular track
  if (_state.abMode) {
    setState({ abMode: false, abTrackA: null, abTrackB: null });
  }
  // A live render has no file to load. Take over the bar, silence the three
  // file decks so a previous track cannot keep playing underneath, and let the
  // mirror below drive time/duration/isPlaying from the stream.
  if (track.streamJobId) {
    _loadingTrackId = track.id;
    cancelAutoAdvance();
    _suppressPlayFalse = false;
    _wsOriginalRef.current?.pause();
    _wsAltRef.current?.pause();
    _wsNoAdapterRef.current?.pause();
    // Point the player at THIS card's take. Every take of the render is already
    // buffering, so this is a switch, not a start — the song you pick is
    // audible from wherever you last left it, not from the beginning.
    mm3StreamSelect(track.streamJobId, track.streamTake ?? 0);
    // pbPlay means PLAY, so start it — selecting alone would leave a click on a
    // paused card switching the audio silently. Harmless when it is already
    // running (mm3StreamPlay returns early), and when the chosen take has no
    // audio yet it records the intent so its first window starts on arrival.
    mm3StreamPlay();
    const snap = mm3StreamSnapshot();
    setState({
      currentTrack: track,
      playerActive: true,
      isLoading: false,
      loadError: null,
      isPlaying: snap.playing,
      currentTime: snap.position,
      duration: snap.expected || snap.received,
      hasMastered: false, playMastered: false,
      hasNoAdapter: false, playNoAdapter: false,
      currentAudioUrl: null,
    });
    mm3StreamSetVolume(_state.volume);
    return;
  }

  // Validate: skip tracks with no audio URL
  if (!track.audioUrl) {
    console.error('[Playback] Track has no audioUrl, skipping:', track.title);
    _suppressPlayFalse = false;
    setState({
      currentTrack: track,
      isPlaying: false,
      isLoading: false,
      loadError: 'No audio file available for this track.',
    });
    // Auto-advance if in a multi-track context
    if (_state.trackList.length > 1) {
      scheduleAutoAdvance('no-audio-url', 300);
    }
    return;
  }

  _loadingTrackId = track.id;
  cancelAutoAdvance();
  _retryCount = 0;
  if (_retryTimer) { clearTimeout(_retryTimer); _retryTimer = null; }

  const hasMastered = !!track.masteredAudioUrl;
  const hasNoAdapter = !!track.noAdapterAudioUrl;
  const useMastered = hasMastered;

  // If already playing, suppress the isPlaying→false transition so the
  // visualiser stays expanded during the track switch.
  const wasPlaying = _state.isPlaying;
  if (wasPlaying) _suppressPlayFalse = true;

  setState({
    currentTrack: track,
    playerActive: true,      // Mark player bar as active (stays open on pause)
    isLoading: true,
    isPlaying: wasPlaying,   // keep true during track-to-track switch
    loadError: null,
    currentTime: 0,
    duration: 0,
    hasMastered,
    playMastered: useMastered,
    hasNoAdapter,
    playNoAdapter: false,    // never default to the reference render
    currentAudioUrl: useMastered ? track.masteredAudioUrl! : track.audioUrl,
  });

  // Load into WaveSurfer
  _wsOriginalRef.current?.loadUrl(track.audioUrl);
  if (_wsAltRef.current && hasMastered) {
    _wsAltRef.current.loadUrl(track.masteredAudioUrl!);
  } else {
    // This track has no mastered version, so nothing overwrites the alt player —
    // it would otherwise keep running the PREVIOUS track's audio, silent because
    // its volume is zeroed, until that audio ended and fired its own finish.
    _wsAltRef.current?.pause();
  }
  if (_wsNoAdapterRef.current && hasNoAdapter) {
    _wsNoAdapterRef.current.loadUrl(track.noAdapterAudioUrl!);
  } else {
    // Same stale-audio guard as the alt deck above.
    _wsNoAdapterRef.current?.pause();
  }

  applyVolumes();
  applyPlaybackRate();
}

// ── Stream mirror ───────────────────────────────────────────────────────────
//
// While the current track is a live render, the bar's clock IS the stream's.
// Mirrored rather than polled so the position stays in step with the audio
// graph rather than with a React interval, and guarded on the job id so a
// finished stream cannot keep writing over a track the user has moved on to.
mm3StreamSubscribe(() => {
  const track = _state.currentTrack;
  if (!track?.streamJobId) return;
  const snap = mm3StreamSnapshot();
  if (snap.jobId !== track.streamJobId) return;
  // A different take of the same render is audible — its transport is not this
  // card's, and writing it here would make one card show another's playhead.
  if (snap.take !== (track.streamTake ?? 0)) return;
  const duration = snap.expected || snap.received;
  if (
    snap.playing === _state.isPlaying &&
    Math.abs(snap.position - _state.currentTime) < 0.05 &&
    Math.abs(duration - _state.duration) < 0.05
  ) return;
  setState({ isPlaying: snap.playing, currentTime: snap.position, duration });
});

// ── Public API ───────────────────────────────────────────────────────────────

/** Play a single track with no navigation context */
export function play(track: PlaybackTrack): void {
  stopVstMonitorIfActive();
  setState({
    trackList: [track],
    trackIndex: 0,
    source: 'direct',
  });
  persistTrackList();
  loadTrack(track);
}

/** Play a track within a navigable list */
export function playFromList(
  track: PlaybackTrack,
  list: PlaybackTrack[],
  source: PlaybackSource
): void {
  stopVstMonitorIfActive();
  const idx = list.findIndex(t => t.id === track.id);
  setState({
    trackList: list,
    trackIndex: idx >= 0 ? idx : 0,
    source,
  });
  persistTrackList();
  loadTrack(track);
}

/** Toggle play/pause on both WaveSurfer instances */
export function togglePlay(): void {
  if (!_state.currentTrack) return;
  // Resuming (currently paused → about to play) stops the VST monitor so the two
  // don't play at once. Pausing must NOT — the monitor-start flow pauses us first.
  if (!_state.isPlaying) stopVstMonitorIfActive();
  if (isStreamDeck()) { mm3StreamToggle(); return; }
  // Pause no longer collapses the player bar — only stop() does that.
  // Clear suppress guard so isPlaying state updates naturally.
  _suppressPlayFalse = false;
  _wsOriginalRef.current?.playPause();
  if (_state.hasMastered) _wsAltRef.current?.playPause();
  if (_state.hasNoAdapter) _wsNoAdapterRef.current?.playPause();
}

/** Stop playback entirely — pauses audio, resets position, and collapses the player bar */
export function stop(): void {
  // Stops LISTENING, not rendering: the engine keeps going and still saves the
  // finished file. Deliberately does not close the stream — reopening is not
  // possible (the engine consumes chunks as they are read), so a stop that
  // dropped the audio would be unrecoverable.
  if (isStreamDeck()) mm3StreamPause();
  _suppressPlayFalse = false;
  if (_retryTimer) { clearTimeout(_retryTimer); _retryTimer = null; }
  _retryCount = 0;

  // Pause all WaveSurfer instances (don't call playPause — we want guaranteed pause)
  const wsOrig = _wsOriginalRef.current;
  const wsAlt = _wsAltRef.current;
  const wsNoAdapter = _wsNoAdapterRef.current;
  if (wsOrig) {
    const m = wsOrig.getMediaElement();
    if (m && !m.paused) wsOrig.playPause();
  }
  if (wsAlt) {
    const m = wsAlt.getMediaElement();
    if (m && !m.paused) wsAlt.playPause();
  }
  if (wsNoAdapter) {
    const m = wsNoAdapter.getMediaElement();
    if (m && !m.paused) wsNoAdapter.playPause();
  }

  setState({
    playerActive: false,
    isPlaying: false,
    currentTrack: null,
    currentTime: 0,
    duration: 0,
    isLoading: false,
    loadError: null,
    hasMastered: false,
    playMastered: false,
    hasNoAdapter: false,
    playNoAdapter: false,
    currentAudioUrl: null,
    trimMode: false,
    trimInPoint: null,
    trimOutPoint: null,
    trimClickCount: 0,
    abMode: false,
    abTrackA: null,
    abTrackB: null,
  });
}

/** Seek all WaveSurfer instances to a time in seconds */
export function seek(time: number): void {
  if (isStreamDeck()) { mm3StreamSeek(time); return; }
  const wsOrig = _wsOriginalRef.current;
  const wsAlt = _wsAltRef.current;
  const wsNoAdapter = _wsNoAdapterRef.current;
  if (wsOrig) {
    const d = wsOrig.getDuration();
    if (d > 0) wsOrig.seekTo(time / d);
  }
  if (wsAlt) {
    const d = wsAlt.getDuration();
    if (d > 0) wsAlt.seekTo(time / d);
  }
  if (wsNoAdapter) {
    const d = wsNoAdapter.getDuration();
    if (d > 0) wsNoAdapter.seekTo(time / d);
  }
}

/** Advance to next track in trackList.
 *  `reason` is diagnostic only. This is also wired directly to the skip button,
 *  so a non-string argument (a click event) just means the user pressed it. */
export function next(reason?: unknown): void {
  const why = typeof reason === 'string' ? reason : 'user';
  const { trackList, trackIndex, shuffle, repeat } = _state;
  if (trackList.length === 0) return;

  let nextIdx: number;
  if (shuffle) {
    if (trackList.length === 1) {
      nextIdx = 0;
    } else {
      // Pick random index excluding current
      do { nextIdx = Math.floor(Math.random() * trackList.length); }
      while (nextIdx === trackIndex && trackList.length > 1);
    }
  } else {
    nextIdx = trackIndex + 1;
    if (nextIdx >= trackList.length) {
      if (repeat === 'all') {
        nextIdx = 0;
      } else {
        // End of list, no repeat — stop
        setState({ isPlaying: false });
        return;
      }
    }
  }

  const nextTrack = trackList[nextIdx];
  if (!nextTrack) return;
  console.log(`[Playback] next(${why}) ${trackIndex}→${nextIdx} of ${trackList.length}: "${nextTrack.title}"`);
  setState({ trackIndex: nextIdx });
  persistTrackList();
  loadTrack(nextTrack);
}

/** Go to previous track (or restart if >3s into current) */
export function previous(): void {
  const { trackList, trackIndex, currentTime } = _state;

  // If >3s into current track, restart it
  if (currentTime > 3) {
    seek(0);
    return;
  }

  if (trackList.length === 0) return;

  let prevIdx = trackIndex - 1;
  if (prevIdx < 0) {
    if (_state.repeat === 'all') {
      prevIdx = trackList.length - 1;
    } else {
      prevIdx = 0; // Stay at start
    }
  }

  const prevTrack = trackList[prevIdx];
  if (!prevTrack) return;
  setState({ trackIndex: prevIdx });
  persistTrackList();
  loadTrack(prevTrack);
}

export type PlaybackVariant = 'noadapter' | 'original' | 'mastered';

/** Switch the audible deck between the no-adapter reference, the raw
 *  (unmastered) output, and the mastered output. Position-syncs the target
 *  deck to the currently audible one, so the A/B comparison is seamless. */
export function setPlaybackVariant(variant: PlaybackVariant): void {
  const track = _state.currentTrack;
  if (!track || _state.abMode) return;
  if (variant === 'mastered' && !_state.hasMastered) return;
  if (variant === 'noadapter' && !_state.hasNoAdapter) return;

  const current: PlaybackVariant = _state.playNoAdapter ? 'noadapter'
    : _state.playMastered ? 'mastered' : 'original';
  if (variant === current) return;

  const deckFor = (v: PlaybackVariant) =>
    v === 'noadapter' ? _wsNoAdapterRef.current
    : v === 'mastered' ? _wsAltRef.current
    : _wsOriginalRef.current;

  const activeWs = deckFor(current);
  const targetWs = deckFor(variant);
  if (!targetWs) return;

  // Sync target position from the currently audible deck. The reference render
  // skips auto-trim so its duration can differ — sync by absolute seconds.
  if (activeWs) {
    const activeTime = activeWs.getCurrentTime();
    const targetDur = targetWs.getDuration();
    if (targetDur > 0) targetWs.seekTo(Math.min(activeTime, targetDur) / targetDur);
  }

  // Keep every loaded deck rolling (browser may have paused a shadow deck)
  if (_state.isPlaying) {
    for (const v of ['original', 'mastered', 'noadapter'] as PlaybackVariant[]) {
      if (v === 'mastered' && !_state.hasMastered) continue;
      if (v === 'noadapter' && !_state.hasNoAdapter) continue;
      const ws = deckFor(v);
      const m = ws?.getMediaElement();
      if (ws && m?.paused) ws.play();
    }
  }

  setState({
    playMastered: variant === 'mastered',
    playNoAdapter: variant === 'noadapter',
    currentAudioUrl: variant === 'mastered' ? (track.masteredAudioUrl || track.audioUrl)
      : variant === 'noadapter' ? (track.noAdapterAudioUrl || track.audioUrl)
      : track.audioUrl,
  });

  applyVolumes();
}

/** Toggle between mastered and original audio */
export function toggleMastered(): void {
  const wsOrig = _wsOriginalRef.current;
  const wsAlt = _wsAltRef.current;
  if (!_state.hasMastered || !wsOrig || !wsAlt || !_state.currentTrack) return;

  const wantMastered = !_state.playMastered;

  // Sync position from active → inactive
  const activeWs = _state.playMastered ? wsAlt : wsOrig;
  const inactiveWs = _state.playMastered ? wsOrig : wsAlt;
  const activeDur = activeWs.getDuration();
  const activeTime = activeWs.getCurrentTime();
  if (activeDur > 0) {
    const inactiveDur = inactiveWs.getDuration();
    if (inactiveDur > 0) inactiveWs.seekTo(activeTime / inactiveDur);
  }

  // Ensure both tracks are actually playing (browser may have paused the shadow)
  const newActiveWs = wantMastered ? wsAlt : wsOrig;
  const newShadowWs = wantMastered ? wsOrig : wsAlt;
  const newActiveMedia = newActiveWs.getMediaElement();
  if (newActiveMedia?.paused) newActiveWs.play();
  const newShadowMedia = newShadowWs.getMediaElement();
  if (newShadowMedia?.paused) newShadowWs.play();

  setState({
    playMastered: wantMastered,
    playNoAdapter: false,
    currentAudioUrl: wantMastered
      ? _state.currentTrack.masteredAudioUrl || _state.currentTrack.audioUrl
      : _state.currentTrack.audioUrl,
  });

  // Volume swap applied reactively via applyVolumes in the setState notification
  applyVolumes();
}

// ── A/B Comparison Mode ─────────────────────────────────────────────────────

/** Enter A/B comparison mode — loads Track A into original, Track B into alt */
export function enterABMode(trackA: PlaybackTrack, trackB: PlaybackTrack): void {
  _retryCount = 0;
  if (_retryTimer) { clearTimeout(_retryTimer); _retryTimer = null; }
  _loadingTrackId = trackA.id;

  // Suppress play-false transition like a track switch
  if (_state.isPlaying) _suppressPlayFalse = true;

  setState({
    abMode: true,
    abTrackA: trackA,
    abTrackB: trackB,
    currentTrack: trackA,
    trackList: [trackA, trackB],
    trackIndex: 0,
    source: 'direct',
    hasMastered: true,       // Tell system alt player is active
    playMastered: false,     // A is audible (original = A)
    hasNoAdapter: false,     // third deck unused in A/B mode
    playNoAdapter: false,
    currentAudioUrl: trackA.audioUrl,
    isLoading: true,
    isPlaying: _state.isPlaying,
    loadError: null,
    currentTime: 0,
    duration: 0,
  });

  // Load audio into WaveSurfer instances
  _wsOriginalRef.current?.loadUrl(trackA.audioUrl);
  _wsAltRef.current?.loadUrl(trackB.audioUrl);
  _wsNoAdapterRef.current?.pause();

  applyVolumes();
  applyPlaybackRate();
}

/** Exit A/B comparison mode */
export function exitABMode(): void {
  setState({
    abMode: false,
    abTrackA: null,
    abTrackB: null,
  });
}

/** Toggle between Track A and Track B — wraps toggleMastered with metadata swap */
export function toggleAB(): void {
  if (!_state.abMode || !_state.abTrackA || !_state.abTrackB) return;
  toggleMastered();  // Swaps volumes + syncs position

  // After toggleMastered, playMastered has been flipped
  const nowB = _state.playMastered;
  setState({
    currentTrack: nowB ? _state.abTrackB : _state.abTrackA,
    currentAudioUrl: nowB ? _state.abTrackB!.audioUrl : _state.abTrackA!.audioUrl,
  });
}

// ── WaveSurfer Event Handlers ────────────────────────────────────────────────
// Called from App.tsx WaveformPlayer callbacks

/** Original track became ready */
export function handleOriginalReady(duration: number): void {
  // Ignore stale ready events from previously loaded tracks
  if (_loadingTrackId && _state.currentTrack && _loadingTrackId !== _state.currentTrack.id) return;
  setState({ duration });
  startBothPlayers();
}

/** Mastered (alt) track became ready — sync position then start */
export function handleAltReady(duration: number): void {
  if (_loadingTrackId && _state.currentTrack && _loadingTrackId !== _state.currentTrack.id) return;
  // Sync alt position with original
  const wsOrig = _wsOriginalRef.current;
  const wsAlt = _wsAltRef.current;
  if (wsAlt && wsOrig) {
    const origDur = wsOrig.getDuration();
    const origTime = wsOrig.getCurrentTime();
    if (origDur > 0 && duration > 0) {
      wsAlt.seekTo(origTime / duration);
    }
  }
  startBothPlayers();
}

/** No-adapter reference deck became ready — sync position then start */
export function handleNoAdapterReady(duration: number): void {
  if (_loadingTrackId && _state.currentTrack && _loadingTrackId !== _state.currentTrack.id) return;
  const wsOrig = _wsOriginalRef.current;
  const wsNoAdapter = _wsNoAdapterRef.current;
  if (wsNoAdapter && wsOrig) {
    const origDur = wsOrig.getDuration();
    const origTime = wsOrig.getCurrentTime();
    if (origDur > 0 && duration > 0) {
      wsNoAdapter.seekTo(Math.min(origTime, duration) / duration);
    }
  }
  startBothPlayers();
}

/** Called by WaveSurfer onTimeUpdate */
export function setCurrentTime(t: number): void {
  // Avoid excessive re-renders — only update if changed meaningfully
  if (Math.abs(t - _state.currentTime) > 0.05) {
    _state = { ..._state, currentTime: t };
    notify();
  }
}

/** Called by WaveSurfer onPlayChange */
export function setIsPlaying(playing: boolean): void {
  // During track-to-track transitions, suppress the false event so the
  // visualiser doesn't briefly collapse.
  if (!playing && _suppressPlayFalse) return;
  if (_state.isPlaying !== playing) {
    setState({ isPlaying: playing });
  }
}

/** Called by WaveSurfer onFinish.
 *
 *  Both players are mounted for the whole life of the app and each fires its own
 *  finish event, so a single song ending produces up to two calls here. This used
 *  to be de-duplicated with a 100ms wall-clock debounce, which is only a guess:
 *  an original and its master are separately encoded and their ends can land
 *  further apart than that, and a player still holding the PREVIOUS track can
 *  finish at an arbitrary moment. Either case advanced the playlist twice.
 *
 *  Instead: only the audible player may drive the playlist, and a given track may
 *  advance the playlist at most once. */
let _lastFinishedTrackId: string | null = null;
export function handleFinish(which: 'original' | 'alt' | 'noadapter' = 'original'): void {
  const audible: 'original' | 'alt' | 'noadapter' = _state.playNoAdapter ? 'noadapter'
    : _state.playMastered ? 'alt' : 'original';
  if (which !== audible) return;

  // Re-pointing a player at a new URL makes it emit one last finish for the
  // audio it is dropping. That arrives synchronously inside the load we just
  // triggered, so it carries the OUTGOING track's position against the incoming
  // track's not-yet-known duration — 237.2s/0.0s. A real end-of-track finish
  // only happens on a track that has loaded and reached its end.
  if (_state.isLoading || _state.duration <= 0) return;

  if (_state.repeat === 'one') {
    // Repeat current track
    _wsOriginalRef.current?.seekTo(0);
    _wsOriginalRef.current?.play();
    if (_state.hasMastered && _wsAltRef.current) {
      _wsAltRef.current.seekTo(0);
      _wsAltRef.current.play();
    }
    if (_state.hasNoAdapter && _wsNoAdapterRef.current) {
      _wsNoAdapterRef.current.seekTo(0);
      _wsNoAdapterRef.current.play();
    }
    return;
  }

  // Advance at most once per track, so a duplicate or late finish can never
  // jump two songs ahead. Reset in loadTrack, so replaying the same track works.
  const finishedId = _state.currentTrack?.id ?? null;
  if (finishedId !== null && finishedId === _lastFinishedTrackId) {
    console.log('[Playback] finish ignored — this track already advanced the playlist');
    return;
  }
  _lastFinishedTrackId = finishedId;

  // Only auto-advance when playing from a playlist, or when repeat-all is active.
  // Individual tracks (library, recent, direct, etc.) just stop at the end.
  const shouldAdvance = _state.repeat === 'all'
    || (_state.source === 'playlist' && _state.trackList.length > 1);

  if (shouldAdvance) {
    next('finish');
  } else {
    setState({ isPlaying: false });
  }
}

// ── Setters for Preferences ──────────────────────────────────────────────────

export function setVolume(v: number): void {
  setState({ volume: v });
  applyVolumes();
  persistPrefs();
}

export function setPlaybackRate(r: number): void {
  setState({ playbackRate: r });
  applyPlaybackRate();
  persistPrefs();
}

export function setShuffle(v: boolean): void {
  setState({ shuffle: v });
  persistPrefs();
}

export function setRepeat(mode: 'none' | 'all' | 'one'): void {
  setState({ repeat: mode });
  persistPrefs();
}

export function cycleRepeat(): void {
  const modes: Array<'none' | 'all' | 'one'> = ['none', 'all', 'one'];
  const idx = modes.indexOf(_state.repeat);
  setRepeat(modes[(idx + 1) % modes.length]);
}

export function setSpectrumEnabled(v: boolean): void {
  setState({ spectrumEnabled: v });
  persistPrefs();
}

// ── Trim Mode ────────────────────────────────────────────────────────────────

/** Toggle trim mode on/off */
export function setTrimMode(on: boolean): void {
  setState({
    trimMode: on,
    trimInPoint: null,
    trimOutPoint: null,
    trimClickCount: 0,
  });
}

/** Handle a click on the waveform during trim mode */
export function handleTrimClick(timeSec: number): void {
  const { trimClickCount, trimInPoint, trimOutPoint, duration } = _state;
  if (trimClickCount === 0) {
    // First click = set IN point
    setState({ trimInPoint: timeSec, trimClickCount: 1 });
  } else if (trimClickCount === 1) {
    // Second click = set OUT point (swap if needed)
    let inP = _state.trimInPoint ?? 0;
    let outP = timeSec;
    if (inP > outP) [inP, outP] = [outP, inP];
    setState({ trimInPoint: inP, trimOutPoint: outP, trimClickCount: 2 });
  } else {
    // Subsequent clicks: move nearest marker
    const inP = trimInPoint ?? 0;
    const outP = trimOutPoint ?? duration;
    const distToIn = Math.abs(timeSec - inP);
    const distToOut = Math.abs(timeSec - outP);
    if (distToIn <= distToOut) {
      let newIn = timeSec;
      let newOut = outP;
      if (newIn > newOut) [newIn, newOut] = [newOut, newIn];
      setState({ trimInPoint: newIn, trimOutPoint: newOut });
    } else {
      let newIn = inP;
      let newOut = timeSec;
      if (newIn > newOut) [newIn, newOut] = [newOut, newIn];
      setState({ trimInPoint: newIn, trimOutPoint: newOut });
    }
  }
}

/** Reload the current track's audio (after crop) */
export function reloadCurrentTrack(newDuration?: number): void {
  const track = _state.currentTrack;
  if (!track) return;
  // Force cache-bust by appending timestamp
  const bustCache = (url: string) => {
    const base = url.split('?')[0];
    return `${base}?_t=${Date.now()}`;
  };
  _wsOriginalRef.current?.loadUrl(bustCache(track.audioUrl));
  if (_wsAltRef.current && track.masteredAudioUrl) {
    _wsAltRef.current.loadUrl(bustCache(track.masteredAudioUrl));
  }
  if (_wsNoAdapterRef.current && track.noAdapterAudioUrl) {
    _wsNoAdapterRef.current.loadUrl(bustCache(track.noAdapterAudioUrl));
  }
  if (newDuration !== undefined) {
    setState({ duration: newDuration });
  }
  // Exit trim mode
  setTrimMode(false);
}

// ── React Hooks ──────────────────────────────────────────────────────────────

/**
 * Subscribe to the full playback state.
 * @deprecated Prefer usePlaybackSelector(s => s.field) — subscribing to the
 * full state causes re-renders on every currentTime tick (~20×/sec).
 */
export function usePlayback(): PlaybackState {
  return useSyncExternalStore(subscribe, getSnapshot);
}

/**
 * Subscribe to a slice of playback state.  The component only re-renders when
 * the selected value changes (Object.is equality).
 *
 * @example
 *   const isPlaying = usePlaybackSelector(s => s.isPlaying);
 *   const trackId   = usePlaybackSelector(s => s.currentTrack?.id ?? null);
 */
export function usePlaybackSelector<T>(selector: (state: PlaybackState) => T): T {
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
