// engine.ts — the play system.
//
// One <audio> element, created in JavaScript and owned here for the life of the
// page. You hand it a track, it plays it. Nothing in this file knows about
// React, Zustand, playlists, the library, or what a waveform looks like.
//
// WHY THIS EXISTS
//
// The old player ran three WaveSurfer decks in lockstep, each of which
// downloaded its whole file and decoded it into a Float32 AudioBuffer to draw a
// waveform. A mastered render is 96 MB of 32-bit float, so a track cost a few
// hundred megabytes and several seconds before a sound came out. Everything
// else followed from that: because loading was slow and its readiness signal
// was ambiguous, starting a track needed a retry loop, a load token, per-deck
// load stamps and a fallback path, and because the decks could disagree about
// which of them was audible, there was an arbiter on top of that.
//
// None of it is here. The peaks come from the server, so nothing is decoded and
// the waveform is ~25 KB of JSON. The element streams the file over Range
// requests. play() returns a promise that tells the truth about whether audio
// started, so there is nothing to poll and nothing to retry.
//
// WHAT IT DELIBERATELY DOES NOT DO
//
// Only one file is loaded at a time. Switching variants re-points the same
// element, which costs a short gap where the old player had none. That gap is
// the price of never again holding three decoded copies of a song, and of the
// variant switch being four lines instead of a subsystem.

import { fetchPeaks, peaksCached, type Peaks } from './peaksClient';
import { pitchAttachElement, pitchResume } from './pitchShift';

export type Variant = 'original' | 'mastered' | 'noAdapter';

export interface EngineTrack {
  id: string;
  title?: string;
  /** `original` is required; a track with no audio is not a track. */
  original: string;
  mastered?: string;
  noAdapter?: string;
}

export interface EngineState {
  trackId: string | null;
  title: string;
  /** Which variant is playing, and which ones this track has at all. */
  variant: Variant;
  available: Variant[];
  /** URL currently pointed at, for anything that needs to fetch alongside. */
  src: string | null;
  playing: boolean;
  /** True between load() and the element having enough data to play. */
  loading: boolean;
  /** Set when the element refused the file. Cleared by the next load. */
  error: string | null;
  currentTime: number;
  duration: number;
  /** Waveform envelope for the current src, or null while it is in flight or
   *  if the server could not produce one. */
  peaks: Peaks | null;
  volume: number;
  rate: number;
}

const state: EngineState = {
  trackId: null,
  title: '',
  variant: 'original',
  available: [],
  src: null,
  playing: false,
  loading: false,
  error: null,
  currentTime: 0,
  duration: 0,
  peaks: null,
  volume: 0.8,
  rate: 1,
};

// ── Subscribers ──────────────────────────────────────────────────────────────
//
// Two channels, because the two kinds of consumer want opposite things. React
// wants a stable snapshot and as few updates as it can get away with; the
// waveform's playhead wants a number every frame and must never make React
// re-render to get it.

type Listener = () => void;
const listeners = new Set<Listener>();
type FrameListener = (time: number, duration: number) => void;
const frameListeners = new Set<FrameListener>();

/** Snapshot identity changes only when something in it changed, so
 *  useSyncExternalStore does not tear or loop. */
let snapshot: EngineState = { ...state };

function emit(): void {
  snapshot = { ...state };
  for (const fn of listeners) fn();
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

export function getSnapshot(): EngineState {
  return snapshot;
}

/** Per-frame position, for canvas playheads and lyric highlighting. Delivered
 *  from one shared rAF loop that only runs while audio is playing. */
export function subscribeFrame(fn: FrameListener): () => void {
  frameListeners.add(fn);
  return () => { frameListeners.delete(fn); };
}

// ── The element ──────────────────────────────────────────────────────────────

let el: HTMLAudioElement | null = null;

/** The one media element. Created on first use and never replaced, which is
 *  what lets it be captured for the audio graph exactly once — a second
 *  createMediaElementSource on the same element throws and cannot be undone. */
function element(): HTMLAudioElement {
  if (el) return el;

  const a = new Audio();
  a.preload = 'auto';
  a.volume = state.volume;
  // Same-origin, so no CORS dance. Explicitly not autoplay: every start goes
  // through play() so there is one place that knows whether audio began.
  a.addEventListener('play', () => { state.playing = true; emit(); startFrames(); });
  a.addEventListener('pause', () => { state.playing = false; emit(); });
  a.addEventListener('ended', () => {
    state.playing = false;
    emit();
    for (const fn of endListeners) fn();
  });
  a.addEventListener('durationchange', () => {
    if (Number.isFinite(a.duration) && a.duration > 0) {
      state.duration = a.duration;
      emit();
    }
  });
  a.addEventListener('canplay', () => {
    if (state.loading) { state.loading = false; emit(); }
  });
  a.addEventListener('error', () => {
    const code = a.error?.code;
    const why = code === 4 ? 'file missing or not decodable'
      : code === 2 ? 'network error'
      : code === 3 ? 'decode error'
      : 'playback aborted';
    // An error on an element we have already moved off is not this track's
    // problem. Only report when the element is still pointed where we put it.
    if (!a.currentSrc || !state.src || a.currentSrc.endsWith(state.src)) {
      state.error = why;
      state.loading = false;
      state.playing = false;
      emit();
      console.warn(`[engine] ${why}: ${state.src}`);
    }
  });

  el = a;
  // Join the shared audio graph once. Everything downstream — the spectrum
  // analyser, the pitch shifter — reads from there.
  pitchAttachElement(a);
  return a;
}

/** The element, for the analyser and anything else that needs the real node.
 *  Callers must not set src or call play/pause on it. */
export function getElement(): HTMLAudioElement {
  return element();
}

// ── Track-ended notification ────────────────────────────────────────────────

type EndListener = () => void;
const endListeners = new Set<EndListener>();

/** Fires when a track plays to its natural end. Whoever owns the playlist
 *  decides what happens next; the engine has no opinion. */
export function onEnded(fn: EndListener): () => void {
  endListeners.add(fn);
  return () => { endListeners.delete(fn); };
}

// ── Frame loop ───────────────────────────────────────────────────────────────

let frameHandle: number | null = null;
let lastEmittedTime = 0;

function startFrames(): void {
  if (frameHandle !== null) return;
  const tick = () => {
    const a = el;
    if (!a || a.paused) { frameHandle = null; return; }

    const t = a.currentTime;
    for (const fn of frameListeners) fn(t, state.duration);

    // React only hears about it ten times a second. The playhead is already
    // smooth via the frame channel, and re-rendering the tree at 60 Hz to move
    // a timestamp is how a player ends up feeling heavy.
    if (Math.abs(t - lastEmittedTime) >= 0.1) {
      lastEmittedTime = t;
      state.currentTime = t;
      emit();
    }

    frameHandle = requestAnimationFrame(tick);
  };
  frameHandle = requestAnimationFrame(tick);
}

// ── Loading ──────────────────────────────────────────────────────────────────

/** Invalidates in-flight peak fetches when the track changes under them. This
 *  is the only async thing in the engine, so it is the only token needed. */
let loadToken = 0;

/** The track the engine is holding, so a variant switch does not need the
 *  caller to hand it back. */
let current: EngineTrack | null = null;

function urlFor(track: EngineTrack, variant: Variant): string {
  if (variant === 'mastered' && track.mastered) return track.mastered;
  if (variant === 'noAdapter' && track.noAdapter) return track.noAdapter;
  return track.original;
}

function availableOf(track: EngineTrack): Variant[] {
  const out: Variant[] = ['original'];
  if (track.mastered) out.push('mastered');
  if (track.noAdapter) out.push('noAdapter');
  return out;
}

/** Point the element at a URL and, separately, go and get its waveform.
 *  `startAt` and `resume` carry a variant switch across the reload. */
function pointAt(url: string, opts: { startAt: number; resume: boolean; token: number }): void {
  const a = element();

  state.src = url;
  state.error = null;
  state.loading = true;
  state.peaks = peaksCached(url);
  if (state.peaks) state.duration = state.peaks.duration;
  emit();

  // Setting src is the whole load. The browser streams the file with Range
  // requests from here on; nothing is downloaded or decoded by us.
  a.src = url;
  a.playbackRate = state.rate;
  a.volume = state.volume;

  if (opts.startAt > 0) {
    // currentTime cannot be set until the element knows the file's length.
    const seekWhenReady = () => {
      a.currentTime = opts.startAt;
      a.removeEventListener('loadedmetadata', seekWhenReady);
    };
    a.addEventListener('loadedmetadata', seekWhenReady);
  }

  if (opts.resume) void play();

  void fetchPeaks(url).then((peaks) => {
    if (opts.token !== loadToken) return;   // moved on since
    state.peaks = peaks;
    // The server's duration is exact and usually arrives before the element
    // has parsed enough of the file to have its own.
    if (peaks && peaks.duration > 0 && !state.duration) state.duration = peaks.duration;
    emit();
  });
}

/**
 * Load a track and, unless told otherwise, start it.
 *
 * There is no readiness handshake and nothing to retry. If the file is not
 * there, the element says so and `state.error` carries it; the caller decides
 * whether that means skip to the next track.
 */
export function load(
  track: EngineTrack,
  opts: { autoplay?: boolean; startAt?: number; variant?: Variant } = {}
): void {
  const token = ++loadToken;
  const available = availableOf(track);

  state.trackId = track.id;
  state.title = track.title ?? '';
  state.available = available;
  // A new track starts on its default variant: mastered when there is one,
  // because that is what people expect to hear, and never the no-adapter
  // reference, which is a diagnostic. A caller can override — the A/B
  // comparison needs both sides on the same variant to be a fair test.
  state.variant = opts.variant && available.includes(opts.variant)
    ? opts.variant
    : track.mastered ? 'mastered' : 'original';
  state.currentTime = opts.startAt ?? 0;
  state.duration = 0;
  lastEmittedTime = 0;

  current = track;
  pointAt(urlFor(track, state.variant), {
    startAt: opts.startAt ?? 0,
    resume: opts.autoplay !== false,
    token,
  });
}

/**
 * Switch between the mastered, unmastered and no-adapter renders of the track
 * that is already loaded, keeping the playhead and the play state.
 *
 * The gap here is one file swap on the same element. The old three-deck design
 * had no gap and cost three decoded copies of the song for it.
 */
export function setVariant(variant: Variant): void {
  if (!current || variant === state.variant) return;
  if (!state.available.includes(variant)) return;

  const a = element();
  const at = a.currentTime;
  const wasPlaying = !a.paused;

  state.variant = variant;
  pointAt(urlFor(current, variant), { startAt: at, resume: wasPlaying, token: ++loadToken });
}

// ── Transport ────────────────────────────────────────────────────────────────

/** Start, or resume. Resolves true when audio is actually coming out.
 *
 *  A rejected play() is nearly always the autoplay policy, which means the page
 *  has had no gesture yet. That is reported, not retried — retrying a policy
 *  decision just loses the error. */
export async function play(): Promise<boolean> {
  const a = element();
  // Our own AudioContext is in the path now, and a suspended one is silence
  // however healthy the element looks.
  pitchResume();
  try {
    await a.play();
    return true;
  } catch (err) {
    const name = (err as Error)?.name;
    if (name === 'AbortError') return false;   // superseded by another load
    state.error = name === 'NotAllowedError'
      ? 'Click anywhere in the page, then press play — the browser wants a gesture first.'
      : String(err);
    state.playing = false;
    emit();
    console.warn('[engine] play refused:', err);
    return false;
  }
}

export function pause(): void {
  el?.pause();
}

export function toggle(): void {
  const a = element();
  if (a.paused) void play(); else a.pause();
}

/** Pause and rewind, without unloading. The player bar collapsing is the
 *  caller's business. */
export function stop(): void {
  const a = element();
  a.pause();
  a.currentTime = 0;
  state.currentTime = 0;
  lastEmittedTime = 0;
  emit();
}

export function seek(seconds: number): void {
  const a = element();
  const dur = state.duration || a.duration;
  if (!Number.isFinite(dur) || dur <= 0) return;
  const t = Math.max(0, Math.min(dur - 0.05, seconds));
  a.currentTime = t;
  state.currentTime = t;
  lastEmittedTime = t;
  emit();
  for (const fn of frameListeners) fn(t, dur);
}

export function seekFraction(fraction: number): void {
  seek(fraction * (state.duration || 0));
}

export function setVolume(v: number): void {
  const vol = Math.max(0, Math.min(1, v));
  state.volume = vol;
  if (el) el.volume = vol;
  emit();
}

export function setRate(rate: number): void {
  state.rate = rate;
  if (el) {
    el.playbackRate = rate;
    // The speed buttons hold pitch. Detuning by rate is what the 44.1 kHz
    // toggle used to do, and it dragged tempo along with it; pitch is the
    // shifter's job, not the transport's.
    (el as HTMLAudioElement & { preservesPitch?: boolean }).preservesPitch = true;
  }
  emit();
}

/** Release the file without tearing down the element or the audio graph.
 *  Used when the player bar closes, so a song you are no longer listening to
 *  stops holding a buffered stream. */
export function unload(): void {
  loadToken++;
  const a = element();
  a.pause();
  a.removeAttribute('src');
  a.load();                      // drops what the element had buffered
  current = null;
  Object.assign(state, {
    trackId: null, title: '', src: null, playing: false, loading: false,
    error: null, currentTime: 0, duration: 0, peaks: null,
    available: [], variant: 'original' as Variant,
  });
  lastEmittedTime = 0;
  emit();
}
