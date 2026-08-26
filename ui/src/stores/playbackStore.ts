/**
 * playbackStore.ts — what is playing, and what plays next.
 *
 * Track list, navigation, shuffle and repeat, trim mode, A/B pairs, and the
 * preferences that persist. It decides WHAT to play. Making sound is
 * audio/engine.ts's job and this file does not reach past its front door.
 *
 * It used to be otherwise. This store held three WaveSurfer decks, ran them in
 * lockstep with volume choosing which was audible, and carried the machinery
 * that arrangement needed: a load token, per-deck load stamps, a 60-try retry
 * loop, a fallback-to-original path, a suppressed-pause flag and an arbiter to
 * decide which of four things owned the speakers. All of it existed because
 * loading a track meant downloading and decoding up to 96 MB before a sound
 * came out, and because "is this deck ready" had no honest answer.
 *
 * The engine streams from disk and reports readiness truthfully, so none of
 * that survived the move. What is left is a playlist.
 *
 * The live MM3 render is still a separate player (mm3StreamStore) because a
 * track being written as you listen has no file, no length and no seekable
 * end. Exactly one of the two owns the audio at a time, and setAudible is the
 * only way that changes.
 */

import { useSyncExternalStore, useRef, useCallback } from 'react';
import { useVstChainStore } from './vstChainStore';
import {
  mm3StreamSnapshot, mm3StreamSubscribe, mm3StreamToggle, mm3StreamSelect, mm3StreamPlay,
  mm3StreamSilence,
  mm3StreamSeek, mm3StreamSetVolume,
} from './mm3StreamStore';
import { setPitchRatio } from '../audio/pitchShift';
import { forgetPeaks } from '../audio/peaksClient';
import * as engine from '../audio/engine';

/** VST monitoring and the global player are mutually exclusive — they'd play
 *  two tracks at once. Whenever global playback starts, stop the VST monitor. */
function stopVstMonitorIfActive(): void {
  try {
    const vst = useVstChainStore.getState();
    if (vst.monitoring) vst.stopMonitor();
  } catch { /* vst store not initialized yet */ }
}

// ── Who is audible ──────────────────────────────────────────────────────────
//
// Two players: the engine (files) and mm3StreamStore (a render arriving window
// by window). Ownership is state rather than something inferred from whichever
// track happens to be selected — inferring it is what left a stream playing
// underneath a file track and routed the transport to a player that was not
// making any sound.
type Audible = { kind: 'none' } | { kind: 'file' } | { kind: 'stream'; jobId: string; take: number };

let _audible: Audible = { kind: 'none' };

/** Hand the audio to exactly one player, silencing the others first. Every
 *  transition goes through here, so two things sounding at once is not a state
 *  this store can reach. */
function setAudible(next: Audible): void {
  const from = _audible.kind === 'stream' ? `stream:${_audible.take}` : _audible.kind;
  const to = next.kind === 'stream' ? `stream:${next.take}` : next.kind;
  if (next.kind !== 'file') engine.pause();
  if (next.kind !== 'stream') mm3StreamSilence();
  // The VST monitor is its own player and loses to anything the user starts.
  if (next.kind !== 'none') stopVstMonitorIfActive();
  _audible = next;
  if (from !== to) console.log(`[Playback] audible: ${from} -> ${to}`);
}

function isStreamDeck(): boolean {
  return _audible.kind === 'stream';
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
   *  transport below delegates there instead of to the engine. The track
   *  becomes an ordinary one the moment the render is saved. */
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
  /** Third variant: the no-adapter reference render. Never both playMastered
   *  and playNoAdapter — 'original' is the state where both are false. */
  playNoAdapter: boolean;
  hasNoAdapter: boolean;
  currentAudioUrl: string | null;

  // ── Preferences (persisted) ──
  volume: number;
  shuffle: boolean;
  repeat: 'none' | 'all' | 'one';
  playbackRate: number;
  /** 44.1 kHz replay of a 48 kHz render: pitch down ×0.91875 through the
   *  WSOLA shifter, tempo left where it was. Not persisted — it is a
   *  diagnostic, and a sticky one would silently detune every future listen. */
  pitch441: boolean;
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

const DEFAULT_PREFS: PersistedPrefs = {
  volume: 0.8,
  shuffle: false,
  repeat: 'none',
  playbackRate: 1.0,
  spectrumEnabled: false,
};

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

  pitch441: false,

  ...prefs,
};

// ── Reactivity (useSyncExternalStore) ────────────────────────────────────────

const CHANGE_EVENT = 'playback-state-change';

const _listeners = new Set<() => void>();

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

// ── Auto-advance past a track that could not be played ───────────────────────
//
// Only one is ever pending, it is cancelled when a new track loads, and it
// checks it is still sitting on the track that failed before doing anything.
// Bare setTimeout(() => next()) calls used to skip a track you had already
// moved on from, several of them in flight at once.

let _autoAdvanceTimer: ReturnType<typeof setTimeout> | null = null;

function cancelAutoAdvance(): void {
  if (_autoAdvanceTimer) { clearTimeout(_autoAdvanceTimer); _autoAdvanceTimer = null; }
}

function scheduleAutoAdvance(reason: string, delayMs: number): void {
  cancelAutoAdvance();
  const forTrackId = _state.currentTrack?.id ?? null;
  _autoAdvanceTimer = setTimeout(() => {
    _autoAdvanceTimer = null;
    if ((_state.currentTrack?.id ?? null) !== forTrackId) {
      console.log(`[Playback] auto-advance (${reason}) dropped — track changed since it was scheduled`);
      return;
    }
    next(`auto:${reason}`);
  }, delayMs);
}

// ── Engine mirror ────────────────────────────────────────────────────────────
//
// The engine is the truth about what the audio is doing; this copies it into
// the shape the UI already reads. One direction only — nothing here writes
// back — so there is no loop to reason about.

/** Errors are reported once per track, not once per notification. */
let _reportedErrorFor: string | null = null;

engine.subscribe(() => {
  if (_audible.kind !== 'file') return;
  const e = engine.getSnapshot();
  // A late notification about a track we have already left.
  if (e.trackId && _state.currentTrack && e.trackId !== _state.currentTrack.id) return;

  const updates: Partial<PlaybackState> = {
    isPlaying: e.playing,
    isLoading: e.loading,
    loadError: e.error,
    currentTime: e.currentTime,
    duration: e.duration || _state.currentTrack?.duration || 0,
    hasMastered: e.available.includes('mastered'),
    hasNoAdapter: e.available.includes('noAdapter'),
    playMastered: e.variant === 'mastered',
    playNoAdapter: e.variant === 'noAdapter',
    currentAudioUrl: e.src,
  };

  // Only notify when something actually moved. The frame channel drives the
  // playhead; this path exists for the parts of the UI that read state.
  const changed = (Object.keys(updates) as Array<keyof PlaybackState>)
    .some(k => !Object.is(_state[k], updates[k]));
  if (changed) setState(updates);

  if (e.error && e.trackId && e.trackId !== _reportedErrorFor) {
    _reportedErrorFor = e.trackId;
    console.error(`[Playback] "${_state.currentTrack?.title}": ${e.error}`);
    // A file that is not there should not hold up a playlist.
    if (_state.trackList.length > 1) scheduleAutoAdvance('load-error', 500);
  }
});

engine.onEnded(() => {
  if (_audible.kind !== 'file') return;
  handleTrackEnded();
});

// ── Loading ──────────────────────────────────────────────────────────────────

/** Core play logic — hands the track to whichever player can play it. */
function loadTrack(track: PlaybackTrack): void {
  // Auto-exit A/B mode when playing a regular track
  if (_state.abMode) {
    setState({ abMode: false, abTrackA: null, abTrackB: null });
  }
  cancelAutoAdvance();
  _reportedErrorFor = null;

  // A live render has no file to load. Hand the audio to the stream player and
  // let the mirror below drive time/duration/isPlaying from it.
  if (track.streamJobId) {
    setAudible({ kind: 'stream', jobId: track.streamJobId, take: track.streamTake ?? 0 });
    // Point the player at THIS card's take. Every take of the render is already
    // buffering, so this is a switch, not a start — the song you pick is
    // audible from wherever you last left it, not from the beginning.
    mm3StreamSelect(track.streamJobId, track.streamTake ?? 0);
    // pbPlay means PLAY, so start it — selecting alone would leave a click on a
    // paused card switching the audio silently.
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

  if (!track.audioUrl) {
    console.error('[Playback] Track has no audioUrl, skipping:', track.title);
    setState({
      currentTrack: track,
      isPlaying: false,
      isLoading: false,
      loadError: 'No audio file available for this track.',
    });
    if (_state.trackList.length > 1) scheduleAutoAdvance('no-audio-url', 300);
    return;
  }

  setAudible({ kind: 'file' });
  setState({
    currentTrack: track,
    playerActive: true,      // the bar stays open on pause; only stop() closes it
    isLoading: true,
    loadError: null,
    currentTime: 0,
    duration: track.duration ?? 0,
  });

  engine.load(
    {
      id: track.id,
      title: track.title,
      original: track.audioUrl,
      mastered: track.masteredAudioUrl || undefined,
      noAdapter: track.noAdapterAudioUrl || undefined,
    },
    { autoplay: true }
  );
  engine.setVolume(_state.volume);
  engine.setRate(_state.playbackRate);
}

// ── Stream mirror ───────────────────────────────────────────────────────────
//
// While the current track is a live render, the bar's clock IS the stream's.
// Mirrored rather than polled so the position stays in step with the audio
// graph, and guarded on the job id so a finished stream cannot keep writing
// over a track the user has moved on to.
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
  setState({ trackList: [track], trackIndex: 0, source: 'direct' });
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
  setState({ trackList: list, trackIndex: idx >= 0 ? idx : 0, source });
  persistTrackList();
  loadTrack(track);
}

export function togglePlay(): void {
  const track = _state.currentTrack;
  if (!track) return;
  // Resuming stops the VST monitor so the two don't play at once. Pausing must
  // NOT — the monitor-start flow pauses us first.
  if (!_state.isPlaying) stopVstMonitorIfActive();
  if (isStreamDeck()) { mm3StreamToggle(); return; }
  // A track restored from localStorage on page load is selected but has never
  // been handed to the engine. Pressing play on it means "play this", not
  // "toggle the nothing that is loaded".
  if (engine.getSnapshot().trackId !== track.id) { loadTrack(track); return; }
  engine.toggle();
}

/** Stop playback entirely — silences every player, resets position, and
 *  collapses the player bar. */
export function stop(): void {
  // Stops LISTENING, not rendering: an MM3 render keeps going and still saves
  // the finished file. Deliberately does not close the stream — reopening is
  // impossible (the engine consumes chunks as they are read), so a stop that
  // dropped the audio would be unrecoverable.
  setAudible({ kind: 'none' });
  // Let go of the file too, so a song you are no longer listening to stops
  // holding a buffered stream.
  engine.unload();
  cancelAutoAdvance();

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

/** Seek to a time in seconds */
export function seek(time: number): void {
  if (isStreamDeck()) { mm3StreamSeek(time); return; }
  engine.seek(time);
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
      do { nextIdx = Math.floor(Math.random() * trackList.length); }
      while (nextIdx === trackIndex && trackList.length > 1);
    }
  } else {
    nextIdx = trackIndex + 1;
    if (nextIdx >= trackList.length) {
      if (repeat === 'all') {
        nextIdx = 0;
      } else {
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

  if (currentTime > 3) { seek(0); return; }
  if (trackList.length === 0) return;

  let prevIdx = trackIndex - 1;
  if (prevIdx < 0) {
    prevIdx = _state.repeat === 'all' ? trackList.length - 1 : 0;
  }

  const prevTrack = trackList[prevIdx];
  if (!prevTrack) return;
  setState({ trackIndex: prevIdx });
  persistTrackList();
  loadTrack(prevTrack);
}

/** A track played to its natural end. */
function handleTrackEnded(): void {
  if (_state.repeat === 'one') {
    engine.seek(0);
    void engine.play();
    return;
  }

  // Only auto-advance when playing from a playlist, or when repeat-all is
  // active. Individual tracks (library, recent, direct) stop at the end.
  const shouldAdvance = _state.repeat === 'all'
    || (_state.source === 'playlist' && _state.trackList.length > 1);

  if (shouldAdvance) next('finish');
  else setState({ isPlaying: false });
}

// ── Variants ─────────────────────────────────────────────────────────────────

export type PlaybackVariant = 'noadapter' | 'original' | 'mastered';

const TO_ENGINE: Record<PlaybackVariant, engine.Variant> = {
  noadapter: 'noAdapter',
  original: 'original',
  mastered: 'mastered',
};

/**
 * Switch the audible render between the no-adapter reference, the raw output
 * and the mastered output, holding the playhead where it is.
 *
 * This re-points one element rather than crossfading between decks that were
 * already loaded, so there is a short gap. The decks bought their instant
 * switch with three decoded copies of the song held in memory for the whole
 * time you listened to it, which is a bad trade for a button most people press
 * a handful of times.
 */
export function setPlaybackVariant(variant: PlaybackVariant): void {
  if (!_state.currentTrack || _state.abMode || isStreamDeck()) return;
  engine.setVariant(TO_ENGINE[variant]);
}

/** Toggle between mastered and original audio */
export function toggleMastered(): void {
  if (!_state.hasMastered) return;
  setPlaybackVariant(_state.playMastered ? 'original' : 'mastered');
}

// ── A/B Comparison Mode ─────────────────────────────────────────────────────

/** Enter A/B comparison mode — two tracks, one transport, toggle between them
 *  at the same position. */
export function enterABMode(trackA: PlaybackTrack, trackB: PlaybackTrack): void {
  setState({
    abMode: true,
    abTrackA: trackA,
    abTrackB: trackB,
    currentTrack: trackA,
    trackList: [trackA, trackB],
    trackIndex: 0,
    source: 'direct',
    playerActive: true,
  });
  setAudible({ kind: 'file' });
  engine.load(
    { id: trackA.id, title: trackA.title, original: trackA.audioUrl },
    // Both sides play their unmastered render, or the comparison is between
    // two different things.
    { autoplay: true, variant: 'original' }
  );
}

export function exitABMode(): void {
  setState({ abMode: false, abTrackA: null, abTrackB: null });
}

/** Toggle between Track A and Track B, keeping the playhead. */
export function toggleAB(): void {
  const { abMode, abTrackA, abTrackB, currentTrack } = _state;
  if (!abMode || !abTrackA || !abTrackB) return;

  const goingToB = currentTrack?.id === abTrackA.id;
  const target = goingToB ? abTrackB : abTrackA;
  const at = _state.currentTime;

  setState({ currentTrack: target, trackIndex: goingToB ? 1 : 0 });
  engine.load(
    { id: target.id, title: target.title, original: target.audioUrl },
    { autoplay: _state.isPlaying, startAt: at, variant: 'original' }
  );
}

// ── Setters for Preferences ──────────────────────────────────────────────────

export function setVolume(v: number): void {
  setState({ volume: v });
  if (isStreamDeck()) mm3StreamSetVolume(v);
  else engine.setVolume(v);
  persistPrefs();
}

export function setPlaybackRate(r: number): void {
  setState({ playbackRate: r });
  engine.setRate(r);
  persistPrefs();
}

/** 44.1 kHz replay of a 48 kHz render — pitch only, tempo held.
 *
 *  Deliberately not done by slowing the transport, which is how it used to
 *  work: that dragged tempo along with the pitch and put two variables into a
 *  comparison meant to isolate one. */
export const SR_44K_48K_RATIO = 44100 / 48000;   // 0.91875

export function setPitch441(v: boolean): void {
  setState({ pitch441: v });
  setPitchRatio(v ? SR_44K_48K_RATIO : 1);
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

export function setTrimMode(on: boolean): void {
  setState({ trimMode: on, trimInPoint: null, trimOutPoint: null, trimClickCount: 0 });
}

/** Handle a click on the waveform during trim mode */
export function handleTrimClick(timeSec: number): void {
  const { trimClickCount, trimInPoint, trimOutPoint, duration } = _state;
  if (trimClickCount === 0) {
    setState({ trimInPoint: timeSec, trimClickCount: 1 });
  } else if (trimClickCount === 1) {
    let inP = _state.trimInPoint ?? 0;
    let outP = timeSec;
    if (inP > outP) [inP, outP] = [outP, inP];
    setState({ trimInPoint: inP, trimOutPoint: outP, trimClickCount: 2 });
  } else {
    // Subsequent clicks move whichever marker is nearer.
    const inP = trimInPoint ?? 0;
    const outP = trimOutPoint ?? duration;
    const moveIn = Math.abs(timeSec - inP) <= Math.abs(timeSec - outP);
    let newIn = moveIn ? timeSec : inP;
    let newOut = moveIn ? outP : timeSec;
    if (newIn > newOut) [newIn, newOut] = [newOut, newIn];
    setState({ trimInPoint: newIn, trimOutPoint: newOut });
  }
}

/** Reload the current track's audio after it has been rewritten on disk (a
 *  crop, say). The file keeps its URL, so both the browser's cache and the
 *  peaks cache have to be told. */
export function reloadCurrentTrack(newDuration?: number): void {
  const track = _state.currentTrack;
  if (!track) return;

  const bust = (url: string) => `${url.split('?')[0]}?_t=${Date.now()}`;
  for (const url of [track.audioUrl, track.masteredAudioUrl, track.noAdapterAudioUrl]) {
    if (url) forgetPeaks(url);
  }

  setAudible({ kind: 'file' });
  engine.load(
    {
      id: track.id,
      title: track.title,
      original: bust(track.audioUrl),
      mastered: track.masteredAudioUrl ? bust(track.masteredAudioUrl) : undefined,
      noAdapter: track.noAdapterAudioUrl ? bust(track.noAdapterAudioUrl) : undefined,
    },
    { autoplay: _state.isPlaying }
  );

  if (newDuration !== undefined) setState({ duration: newDuration });
  setTrimMode(false);
}

// ── React Hooks ──────────────────────────────────────────────────────────────

/**
 * Subscribe to the full playback state.
 * @deprecated Prefer usePlaybackSelector(s => s.field) — subscribing to the
 * full state re-renders on every currentTime tick.
 */
export function usePlayback(): PlaybackState {
  return useSyncExternalStore(subscribe, getSnapshot);
}

/**
 * Subscribe to a slice of playback state. The component only re-renders when
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
