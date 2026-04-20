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

import { useCallback, useSyncExternalStore } from 'react';
import type { WaveformPlayerHandle } from '../components/player/WaveformPlayer';
import type { Song } from '../types';
import type { PlaylistItem } from '../components/lyric-studio/playlistStore';

// ── Types ────────────────────────────────────────────────────────────────────

/** Canonical track representation — normalized, always number duration */
export interface PlaybackTrack {
  id: string;
  title: string;
  audioUrl: string;
  masteredAudioUrl?: string;
  artistName?: string;
  coverUrl?: string;
  duration?: number;                  // Always seconds, never string
  style?: string;
  lyrics?: string;
  caption?: string;
  generationParams?: Record<string, any>;
}

export type PlaybackSource =
  | 'library'
  | 'playlist'
  | 'lireek-recent'
  | 'lireek-recordings'
  | 'lireek-queue'
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

  // ── Mastered/Original Toggle ──
  playMastered: boolean;
  hasMastered: boolean;
  currentAudioUrl: string | null;

  // ── Preferences (persisted) ──
  volume: number;
  shuffle: boolean;
  repeat: 'none' | 'all' | 'one';
  playbackRate: number;
  spectrumEnabled: boolean;
}

// ── Conversion Helpers ───────────────────────────────────────────────────────

function coerceDuration(d: string | number | undefined | null): number | undefined {
  if (d == null) return undefined;
  if (typeof d === 'number') return d;
  if (typeof d === 'string') {
    if (d.includes(':')) {
      const [m, s] = d.split(':').map(Number);
      return (m || 0) * 60 + (s || 0);
    }
    const n = parseFloat(d);
    return isNaN(n) ? undefined : n;
  }
  return undefined;
}

export function songToTrack(song: Song): PlaybackTrack {
  return {
    id: song.id,
    title: song.title || 'Untitled',
    audioUrl: song.audioUrl || song.audio_url || '',
    masteredAudioUrl: song.masteredAudioUrl || song.mastered_audio_url || '',
    artistName: song.artistName || '',
    coverUrl: song.coverUrl || song.cover_url || '',
    duration: coerceDuration(song.duration),
    style: song.style || '',
    lyrics: song.lyrics || '',
    caption: song.caption || '',
    generationParams: song.generationParams || song.generation_params as any,
  };
}

export function playlistItemToTrack(item: PlaylistItem): PlaybackTrack {
  return {
    id: item.id,
    title: item.title || 'Untitled',
    audioUrl: item.audioUrl || '',
    masteredAudioUrl: item.masteredAudioUrl || '',
    artistName: item.artistName || '',
    coverUrl: item.coverUrl || '',
    duration: coerceDuration(item.duration),
    style: item.style || '',
    generationParams: item.generationParams,
  };
}

// Generic converter for RecentSong-shaped objects (lireekApi types)
export function recentSongToTrack(rs: {
  ag_id?: number;
  hotstep_job_id?: string;
  song_title?: string;
  audio_url?: string;
  mastered_audio_url?: string;
  artist_name?: string;
  cover_url?: string;
  album_image?: string;
  artist_image?: string;
  duration?: number;
  caption?: string;
  lyrics?: string;
}): PlaybackTrack {
  return {
    id: rs.hotstep_job_id || `recent-${rs.ag_id}`,
    title: rs.song_title || 'Untitled',
    audioUrl: rs.audio_url || '',
    masteredAudioUrl: rs.mastered_audio_url || '',
    artistName: rs.artist_name || '',
    coverUrl: rs.cover_url || rs.album_image || rs.artist_image || '',
    duration: coerceDuration(rs.duration),
    caption: rs.caption || '',
    lyrics: rs.lyrics || '',
  };
}

// Generic converter for AudioQueueItem-shaped objects
export function audioQueueItemToTrack(item: {
  id: string;
  songId?: string;
  audioUrl?: string;
  masteredAudioUrl?: string;
  artistName?: string;
  artistImageUrl?: string;
  audioDuration?: number;
  generation: { title?: string; caption?: string };
}): PlaybackTrack {
  return {
    id: item.songId || item.id,
    title: item.generation.title || 'Untitled',
    audioUrl: item.audioUrl || '',
    masteredAudioUrl: item.masteredAudioUrl || '',
    artistName: item.artistName || '',
    coverUrl: item.artistImageUrl || '',
    duration: coerceDuration(item.audioDuration),
    caption: item.generation.caption || '',
  };
}

// ── localStorage Keys ────────────────────────────────────────────────────────

const PREFS_KEY = 'playback-prefs';
const TRACKLIST_KEY = 'playback-tracklist';
const MAX_PERSISTED_TRACKS = 500;

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
  const limited = {
    ...tl,
    trackList: tl.trackList.slice(0, MAX_PERSISTED_TRACKS),
  };
  localStorage.setItem(TRACKLIST_KEY, JSON.stringify(limited));
}

// ── WaveSurfer Handle Registration ───────────────────────────────────────────

// Store ref OBJECTS (not .current values) so we always dereference at call time.
// This is critical because useImperativeHandle may not have committed yet when
// the useEffect in App.tsx runs registerPlayers.
type WsRef = { current: WaveformPlayerHandle | null };

let _wsOriginalRef: WsRef = { current: null };
let _wsAltRef: WsRef = { current: null };

export function registerPlayers(
  orig: WsRef,
  alt: WsRef
): void {
  _wsOriginalRef = orig;
  _wsAltRef = alt;
}

export function getActiveMediaElement(): HTMLMediaElement | null {
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

  playMastered: false,
  hasMastered: false,
  currentAudioUrl: null,

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

/** Attempt to start both players. Retries if the AUDIBLE track isn't ready. */
function startBothPlayers(): void {
  if (_retryTimer) { clearTimeout(_retryTimer); _retryTimer = null; }

  const wsOrig = _wsOriginalRef.current;
  const wsAlt = _wsAltRef.current;

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

  // The AUDIBLE track must be ready before we declare success.
  // playMastered=true → alt is audible, playMastered=false → original is audible.
  const audibleReady = _state.playMastered ? altReady : origReady;

  if (audibleReady) {
    _retryCount = 0;
    setState({ isPlaying: true, isLoading: false, loadError: null });
    return;
  }

  // Audible track not ready — retry
  _retryCount++;
  if (_retryCount <= MAX_RETRIES) {
    _retryTimer = setTimeout(startBothPlayers, RETRY_INTERVAL);
  } else {
    _retryCount = 0;
    setState({
      isLoading: false,
      loadError: 'Audio failed to load. The file may be missing or inaccessible.',
    });
  }
}

function applyVolumes(): void {
  _wsOriginalRef.current?.setVolume(_state.playMastered ? 0 : _state.volume);
  _wsAltRef.current?.setVolume(_state.playMastered ? _state.volume : 0);
}

function applyPlaybackRate(): void {
  _wsOriginalRef.current?.setPlaybackRate(_state.playbackRate);
  _wsAltRef.current?.setPlaybackRate(_state.playbackRate);
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
  _loadingTrackId = track.id;
  _retryCount = 0;
  if (_retryTimer) { clearTimeout(_retryTimer); _retryTimer = null; }

  const hasMastered = !!track.masteredAudioUrl;
  const useMastered = hasMastered;

  setState({
    currentTrack: track,
    isLoading: true,
    isPlaying: false,
    loadError: null,
    currentTime: 0,
    duration: 0,
    hasMastered,
    playMastered: useMastered,
    currentAudioUrl: useMastered ? track.masteredAudioUrl! : track.audioUrl,
  });

  // Load into WaveSurfer
  _wsOriginalRef.current?.loadUrl(track.audioUrl);
  if (_wsAltRef.current && hasMastered) _wsAltRef.current.loadUrl(track.masteredAudioUrl!);

  applyVolumes();
  applyPlaybackRate();
}

// ── Public API ───────────────────────────────────────────────────────────────

/** Play a single track with no navigation context */
export function play(track: PlaybackTrack): void {
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
  _wsOriginalRef.current?.playPause();
  if (_state.hasMastered) _wsAltRef.current?.playPause();
}

/** Seek both WaveSurfer instances to a time in seconds */
export function seek(time: number): void {
  const wsOrig = _wsOriginalRef.current;
  const wsAlt = _wsAltRef.current;
  if (wsOrig) {
    const d = wsOrig.getDuration();
    if (d > 0) wsOrig.seekTo(time / d);
  }
  if (wsAlt) {
    const d = wsAlt.getDuration();
    if (d > 0) wsAlt.seekTo(time / d);
  }
}

/** Advance to next track in trackList */
export function next(): void {
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
    currentAudioUrl: wantMastered
      ? _state.currentTrack.masteredAudioUrl || _state.currentTrack.audioUrl
      : _state.currentTrack.audioUrl,
  });

  // Volume swap applied reactively via applyVolumes in the setState notification
  applyVolumes();
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
  if (_state.isPlaying !== playing) {
    setState({ isPlaying: playing });
  }
}

/** Called by WaveSurfer onFinish */
export function handleFinish(): void {
  if (_state.repeat === 'one') {
    // Repeat current track
    _wsOriginalRef.current?.seekTo(0);
    _wsOriginalRef.current?.play();
    if (_state.hasMastered && _wsAltRef.current) {
      _wsAltRef.current.seekTo(0);
      _wsAltRef.current.play();
    }
    return;
  }

  if (_state.repeat === 'all' || _state.trackList.length > 1) {
    next();
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

// ── React Hook ───────────────────────────────────────────────────────────────

export function usePlayback(): PlaybackState {
  return useSyncExternalStore(subscribe, getSnapshot);
}
