// mm3StreamStore.ts — a transport for a MiniMax-Music3 render that is still rendering.
//
// MM3 renders in 200-frame windows, and a window's audio is FINAL the moment it
// is vocoded and cropped (engine mm3-pipeline.h §4). With the sliding pipeline
// the engine dispatches those windows WHILE the planner is still planning, so
// audio exists seconds into a render. It arrives as a chunked body of
// concatenated self-contained WAVs on GET /api/generate/mm3/stream/:jobId.
//
// This is not a "preview player" — it is the deck the main play bar drives while
// the track is still being written. It therefore has to behave like one:
// play/pause, seek anywhere inside the audio received so far, a position that
// only moves forward, a volume, and a waveform that grows.
//
// ── Why a module singleton ──────────────────────────────────────────────────
//
// THE ENGINE ALLOWS EXACTLY ONE READER PER JOB. Chunks are consumed as they are
// written, so a second connection would silently steal half the song from the
// first; the engine answers 409 instead. The audio therefore cannot live in
// component state — the transport is shown in the generations grid, the queue
// card and the play bar at once. One AudioContext, one connection, N readers.
//
// ── The timeline ────────────────────────────────────────────────────────────
//
// Decoded windows are kept as segments with exact frame offsets, and playback is
// described by an ANCHOR: frame `anchorFrame` is due to sound at context time
// `anchorTime`. Position is then
//
//     anchorFrame + max(0, ctx.currentTime - anchorTime) * rate
//
// The max(0, …) is load-bearing three times over: it holds the playhead still
// during the initial pre-buffer, it holds it still during a rebuffer stall, and
// it makes "re-anchor into the future" the single primitive that expresses
// starting, seeking and stalling alike. Scheduling is in FRAMES, never in
// seconds accumulated from `ab.duration`, which would drift across a hundred
// windows.
//
// ── Why this is not useStreamAudio ──────────────────────────────────────────
//
// useStreamAudio plays STORM: a chain of INDEPENDENT generations, beat-matched
// and CROSSFADED into each other, where the cut point is a musical decision.
// These segments are consecutive spans of ONE signal with exact sample offsets
// and must HARD-SPLICE — any crossfade would sum two neighbours across a
// boundary that is meant to be a plain concatenation. Two more differences a
// copy of that file would get wrong:
//
//   * THE SAMPLE RATE COMES FROM THE CHUNK. useStreamAudio hardcodes 48 kHz for
//     ACE. MM3 is 44.1 kHz, and a context at the wrong rate makes
//     decodeAudioData resample every window — after which the exact frame counts
//     this whole file depends on are no longer exact. Read it from the WAV
//     header, before the context is created.
//   * ON UNDERRUN IT STALLS, IT DOES NOT DROP. Whether the render outruns
//     playback is a per-configuration fact: measured on a 5090 at 30 steps, a
//     q8_0 fresh plan sustains 1.06x realtime but an f16 one manages 0.74x and
//     WILL be caught. Then the timeline is re-anchored forward and everything
//     still plays in full — a pause, not a hole in the track.
//
// The stream is an ADDITIONAL output. The finished WAV is written, saved and
// added to the library exactly as it is with streaming off, so a failure here
// costs a preview and nothing else.

import { useSyncExternalStore } from 'react';
import { extractWav, wavFrameCount, wavSampleRate } from '../utils/wavStream';

export interface Mm3StreamState {
  /** The job whose stream is open (or was last opened). */
  jobId: string | null;
  /** The network stream is open — i.e. more audio is still coming. */
  connected: boolean;
  /** The engine closed the stream: what is here is the whole track. */
  done: boolean;
  /** The transport is running. */
  playing: boolean;
  /** Playhead, seconds. */
  position: number;
  /** Seconds of audio received — the seekable limit, and how much of the
   *  track is finished. */
  received: number;
  /** The render's full length in seconds, as the engine resolved it. Known
   *  before any audio arrives, which is what lets a card show real progress
   *  instead of a spinner. */
  expected: number;
  /** Windows received. */
  chunks: number;
  /** Times the renderer was caught and playback had to rebuffer. */
  underruns: number;
  /** The browser refused to start audio without a gesture. */
  needsGesture: boolean;
  volume: number;
  error: string | null;
  /** Bumped whenever the waveform grows, so canvases can redraw cheaply without
   *  the peaks themselves being React state. */
  peaksVersion: number;
}

/** Pre-buffer before playback starts, seconds; also the amount re-buffered
 *  after a stall. Long enough to ride out a slow window, short enough that
 *  "press generate, hear music" is still true. */
const HEADROOM = 5;

/** Waveform resolution. 40 buckets/s is ~14 k floats for a six-minute track —
 *  finer than the bar width any canvas here can draw, and cheap. */
const PEAK_RATE = 40;

const INITIAL: Mm3StreamState = {
  jobId: null, connected: false, done: false, playing: false,
  position: 0, received: 0, expected: 0, chunks: 0, underruns: 0,
  needsGesture: false, volume: 1.0, error: null, peaksVersion: 0,
};

let state: Mm3StreamState = INITIAL;
const listeners = new Set<() => void>();

function emit(patch: Partial<Mm3StreamState>): void {
  state = { ...state, ...patch };
  listeners.forEach(fn => fn());
}

interface Seg {
  buf: AudioBuffer;
  /** First frame of this segment in the finished track. */
  start: number;
  frames: number;
}

let ac: AudioContext | null = null;
let gain: GainNode | null = null;
let abort: AbortController | null = null;
let raf = 0;

let segs: Seg[] = [];
let totalFrames = 0;
let rate = 0;

/** Waveform envelope, one max-abs value per 1/PEAK_RATE second. */
let peaks = new Float32Array(0);
let peakCount = 0;

/** The anchor (see the header). */
let anchorFrame = 0;
let anchorTime = 0;
/** Segments already handed to the graph under the CURRENT anchor. Cleared by
 *  every re-anchor, because a re-anchor moves when everything is due. */
let scheduled = new Set<number>();
let sources: AudioBufferSourceNode[] = [];
/** Playhead while paused. */
let pausedFrame = 0;
let playing = false;
/** INTENT to play, as distinct from `playing` (actually sounding). They differ
 *  for the whole gap between opening the stream and the first window arriving —
 *  press play in that window and there is nothing to start yet, so the intent is
 *  recorded and honoured when audio appears. The reverse also matters: pausing
 *  while waiting must survive the first window's auto-start. */
let wantPlay = false;

let netOpen = false;
let volume = 1.0;

/** Jobs already auto-opened, so a remount does not reopen a stream the user
 *  deliberately stopped. */
const autoOpened = new Set<string>();
/** Failed auto-open attempts per job. The engine can legitimately say "not yet"
 *  for a moment (the job is queued behind another on the one GPU worker), and a
 *  single attempt that lost that race would leave a silent card for the whole
 *  render. Bounded, because a permanent 409 must not loop. */
const attempts = new Map<string, number>();
const MAX_AUTO_ATTEMPTS = 3;

// ── Position ────────────────────────────────────────────────────────────────

function positionFrames(): number {
  if (!ac || !playing) return pausedFrame;
  const p = anchorFrame + Math.max(0, ac.currentTime - anchorTime) * rate;
  return Math.min(p, totalFrames);
}

function tick(): void {
  // RE-ARM UNCONDITIONALLY. `ac` does not exist until the first window lands,
  // which on a fresh plan is ten seconds after the stream opens — an early
  // `return` here (as the first cut had) kills the loop before there is
  // anything to report and never restarts it, so the playhead never moves for
  // the whole render while the audio plays perfectly.
  if (ac) {
    const pf = positionFrames();
    // The end of a finished stream is the end of the track. Without this the
    // transport sits at 100% still claiming to play, and the play button would
    // do nothing when pressed.
    if (playing && state.done && totalFrames > 0 && pf >= totalFrames) {
      playing = false;
      wantPlay = false;
      pausedFrame = totalFrames;
      emit({ playing: false, position: totalFrames / (rate || 1) });
    } else {
      emit({ position: pf / (rate || 1) });
    }
  }
  raf = requestAnimationFrame(tick);
}

// ── Scheduling ──────────────────────────────────────────────────────────────

function stopSources(): void {
  for (const s of sources) { try { s.stop(); } catch { /* already ended */ } }
  sources = [];
  scheduled = new Set();
}

/** Hand one segment to the graph under the current anchor, skipping any part
 *  already behind the playhead. */
function scheduleSeg(i: number): void {
  if (!ac || !gain || scheduled.has(i)) return;
  const seg = segs[i];
  if (!seg || seg.start + seg.frames <= anchorFrame) return;   // entirely played
  const at = anchorTime + (seg.start - anchorFrame) / rate;
  const offsetFrames = Math.max(0, anchorFrame - seg.start);
  const src = ac.createBufferSource();
  src.buffer = seg.buf;
  src.connect(gain);
  // No ramp, no fade: consecutive spans of one signal, spliced at the sample
  // the engine cropped them to.
  src.start(Math.max(at, ac.currentTime), offsetFrames / rate);
  sources.push(src);
  src.onended = () => { sources = sources.filter(s => s !== src); };
  scheduled.add(i);
}

/** Re-anchor: frame `frame` becomes due `delay` seconds from now, and everything
 *  from there is rescheduled. The one primitive behind start, seek and rebuffer. */
function anchorAt(frame: number, delay: number): void {
  if (!ac) return;
  stopSources();
  anchorFrame = Math.max(0, Math.min(frame, totalFrames));
  anchorTime = ac.currentTime + delay;
  for (let i = 0; i < segs.length; i++) scheduleSeg(i);
}

// ── Waveform ────────────────────────────────────────────────────────────────

function appendPeaks(buf: AudioBuffer): void {
  const per = Math.max(1, Math.round(buf.sampleRate / PEAK_RATE));
  const n = Math.ceil(buf.length / per);
  if (peakCount + n > peaks.length) {
    const grown = new Float32Array(Math.max(peaks.length * 2, peakCount + n + 1024));
    grown.set(peaks.subarray(0, peakCount));
    peaks = grown;
  }
  // Max-abs across both channels, so a hard-panned moment is not drawn as silence.
  const L = buf.getChannelData(0);
  const R = buf.numberOfChannels > 1 ? buf.getChannelData(1) : L;
  for (let b = 0; b < n; b++) {
    const from = b * per;
    const to = Math.min(from + per, buf.length);
    let m = 0;
    for (let i = from; i < to; i++) {
      const a = Math.abs(L[i]) > Math.abs(R[i]) ? Math.abs(L[i]) : Math.abs(R[i]);
      if (a > m) m = a;
    }
    peaks[peakCount++] = m;
  }
}

/** The waveform so far. Deliberately NOT React state — it is thousands of floats
 *  that a canvas reads directly; `peaksVersion` is the render trigger. */
export function mm3StreamPeaks(): { peaks: Float32Array; count: number; rate: number } {
  return { peaks, count: peakCount, rate: PEAK_RATE };
}

// ── Connection ──────────────────────────────────────────────────────────────

function teardown(): void {
  netOpen = false;
  playing = false;
  cancelAnimationFrame(raf);
  stopSources();
  try { ac?.close(); } catch { /* already closed */ }
  ac = null;
  gain = null;
  segs = [];
  totalFrames = 0;
  rate = 0;
  peaks = new Float32Array(0);
  peakCount = 0;
  anchorFrame = 0;
  anchorTime = 0;
  pausedFrame = 0;
}

async function open(jobId: string, expected: number, autoPlay: boolean): Promise<void> {
  if (netOpen) return;
  // A previous render's context may still be around (its tail played out and was
  // never stopped). Start clean: the new stream has its own timeline.
  teardown();
  netOpen = true;
  wantPlay = autoPlay;
  emit({ ...INITIAL, jobId, expected, volume, connected: true });

  // Logged because the failure mode of this feature is SILENCE, and silence is
  // indistinguishable from "the engine has not produced a window yet".
  console.log(`[MM3 Stream] opening ${jobId} (expecting ${expected.toFixed(1)}s)`);
  const ctrl = new AbortController();
  abort = ctrl;
  raf = requestAnimationFrame(tick);

  let chunks = 0, underruns = 0;

  try {
    const res = await fetch(`/api/generate/mm3/stream/${encodeURIComponent(jobId)}`, { signal: ctrl.signal });
    if (!res.ok || !res.body) {
      // 409 is the engine declining (not a streaming job, already finished,
      // already has a reader). Surface its own sentence — those are written to
      // be read.
      const body = await res.text().catch(() => '');
      let msg = `Stream unavailable (HTTP ${res.status})`;
      try { const j = JSON.parse(body) as { error?: string }; if (j?.error) msg = j.error; } catch { /* not JSON */ }
      throw new Error(msg);
    }

    const reader = res.body.getReader();
    let buf: Uint8Array<ArrayBufferLike> = new Uint8Array(0);

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!netOpen) break;

      const merged = new Uint8Array(buf.length + value.length);
      merged.set(buf); merged.set(value, buf.length);
      buf = merged;

      for (;;) {
        const w = extractWav(buf);
        if (!w) break;
        buf = w.remaining;

        const r = wavSampleRate(w.data);
        const n = wavFrameCount(w.data);
        if (!r || !n) continue;

        // First chunk: build the context AT THE STREAM'S RATE (see the header).
        if (!ac) {
          const ctx = new AudioContext({ sampleRate: r });
          const g = ctx.createGain();
          g.gain.value = volume;
          g.connect(ctx.destination);
          ac = ctx;
          gain = g;
          rate = r;
          // Autoplay policy: a context created without a user gesture starts
          // suspended. Try to resume; if the browser still says no, say so
          // rather than sitting silently on a full buffer.
          if (ctx.state === 'suspended') {
            try { await ctx.resume(); } catch { /* needs a gesture */ }
          }
          emit({ needsGesture: ctx.state === 'suspended' });
        }
        const ctx = ac;
        if (!ctx) break;

        // decodeAudioData detaches the ArrayBuffer it is given, so hand it a
        // copy — `buf` may still alias the same backing store.
        const copy = w.data.slice();
        let ab: AudioBuffer;
        try {
          ab = await ctx.decodeAudioData(copy.buffer as ArrayBuffer);
        } catch {
          continue;  // a torn chunk costs one window, not the stream
        }
        if (!netOpen) break;

        const seg: Seg = { buf: ab, start: totalFrames, frames: n };
        segs.push(seg);
        totalFrames += n;
        appendPeaks(ab);
        chunks++;

        if (chunks === 1) {
          console.log(`[MM3 Stream] first window: ${(n / rate).toFixed(2)}s @ ${rate} Hz, context ${ctx.state}`);
          // Start the transport one headroom from now. The playhead holds at 0
          // until then (positionFrames' max(0, …)) — the pre-buffer the
          // reference Space uses.
          if (wantPlay) {
            playing = true;
            anchorAt(0, HEADROOM);
          }
        } else if (playing) {
          const dueAt = anchorTime + (seg.start - anchorFrame) / rate;
          if (dueAt < ctx.currentTime) {
            // The renderer has been caught. Re-anchor so this window and every
            // later one still play IN FULL, one headroom from now: a stall
            // rather than a hole. Starting it part-way in would keep the clock
            // and lose audio, and this is a track someone is deciding whether
            // to keep.
            underruns++;
            anchorAt(seg.start, HEADROOM);
          } else {
            scheduleSeg(segs.length - 1);
          }
        }

        emit({
          chunks, underruns, playing,
          received: totalFrames / rate,
          peaksVersion: state.peaksVersion + 1,
        });
      }
    }
    console.log(`[MM3 Stream] closed: ${chunks} window(s), ${(totalFrames / (rate || 1)).toFixed(1)}s, ${underruns} rebuffer(s)`);
    // The engine is done, so `received` IS the length. Replacing `expected`
    // here is what makes a card that finished early (EOS) stop showing a gap it
    // will never fill.
    emit({ done: true, connected: false, expected: totalFrames / (rate || 1) });
  } catch (e: unknown) {
    const err = e as Error;
    if (err?.name !== 'AbortError') {
      console.warn(`[MM3 Stream] ${jobId}: ${err?.message || String(e)}`);
      emit({ error: err?.message || String(e), connected: false });
      // Nothing arrived at all — most likely the job had not reached the engine
      // yet. Retry a couple of times before giving up.
      if (chunks === 0) {
        const n = (attempts.get(jobId) ?? 0) + 1;
        attempts.set(jobId, n);
        if (n < MAX_AUTO_ATTEMPTS) {
          setTimeout(() => { autoOpened.delete(jobId); mm3StreamEnsure(jobId, expected, autoPlay); }, 2000);
        }
      }
    }
  } finally {
    // The scheduled tail keeps playing; only the NETWORK side stops here.
    netOpen = false;
    abort = null;
    emit({ connected: false });
  }
}

// ── Public API ──────────────────────────────────────────────────────────────

/** Open the stream for `jobId` if nothing is playing it yet. Idempotent, and it
 *  will not reopen a stream the user stopped on purpose. A null jobId is NOT a
 *  teardown: a render that has just finished still has audio to play. */
export function mm3StreamEnsure(jobId: string | null, expected = 0, autoPlay = true): void {
  if (!jobId || netOpen || autoOpened.has(jobId)) return;
  autoOpened.add(jobId);
  void open(jobId, expected, autoPlay);
}

/** Open because the user asked — after they stopped, or after the auto-open gave
 *  up. Bypasses the once-only guard and resets the retry budget. */
export function mm3StreamStart(jobId: string, expected = 0): void {
  if (netOpen) return;
  autoOpened.add(jobId);
  attempts.delete(jobId);
  void open(jobId, expected, true);
}

export function mm3StreamPlay(): void {
  // Recorded even when there is nothing to play yet — the first window will
  // honour it. This is what makes "press play while it says waiting…" work.
  wantPlay = true;
  if (!ac || playing) return;
  if (ac.state === 'suspended') {
    void ac.resume().then(() => emit({ needsGesture: ac?.state === 'suspended' }));
  }
  playing = true;
  // A short delay rather than zero: start() in the immediate past drops the
  // attack of the first segment.
  anchorAt(pausedFrame, 0.05);
  emit({ playing: true });
}

export function mm3StreamPause(): void {
  wantPlay = false;
  if (!playing) return;
  pausedFrame = positionFrames();
  playing = false;
  stopSources();
  emit({ playing: false, position: pausedFrame / (rate || 1) });
}

export function mm3StreamToggle(): void {
  if (playing) mm3StreamPause(); else mm3StreamPlay();
}

/** True once there is audio to play. The generations card shows a live render
 *  before this is true (so the user sees it coming) but its transport only
 *  means anything after. */
export function mm3StreamHasAudio(): boolean {
  return totalFrames > 0;
}

/** Seek inside the audio received so far. Beyond that there is nothing to play,
 *  so it clamps rather than pretending. */
export function mm3StreamSeek(sec: number): void {
  if (!ac || !rate) return;
  const frame = Math.max(0, Math.min(sec * rate, totalFrames));
  pausedFrame = frame;
  if (playing) anchorAt(frame, 0.05);
  emit({ position: frame / rate });
}

export function mm3StreamSetVolume(v: number): void {
  volume = v;
  if (gain) gain.gain.value = v;
  emit({ volume: v });
}

/** Retry after the browser refused audio without a gesture. Call from a click. */
export function mm3StreamResume(): void {
  if (!ac) return;
  void ac.resume().then(() => emit({ needsGesture: ac?.state === 'suspended' }));
}

/** Close the stream and drop the audio. The render is unaffected — this ends a
 *  preview, it does not cancel a job. */
export function mm3StreamStop(): void {
  abort?.abort();
  abort = null;
  teardown();
  emit({ ...INITIAL, volume });
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

export function useMm3StreamAudio(): Mm3StreamState {
  return useSyncExternalStore(subscribe, () => state, () => state);
}

export function mm3StreamSnapshot(): Mm3StreamState {
  return state;
}

/** Notified on every state change — for stores that MIRROR this one (the play
 *  bar) rather than components that render it. */
export function mm3StreamSubscribe(fn: () => void): () => void {
  return subscribe(fn);
}

/** The live graph, for the spectrum analyser.
 *
 *  Returned as (context, node) rather than as a media element because there is
 *  no media element — and deliberately NOT routed through one. The analyser
 *  attaches a second audioMotion instance to THIS context with
 *  `connectSpeakers: false`, so it only observes: the audio path stays
 *  gain → destination at the stream's own 44.1 kHz, which is what keeps the
 *  window splices sample-exact. Null until the first window has been decoded. */
export function mm3StreamGraph(): { ctx: AudioContext; node: AudioNode } | null {
  return ac && gain ? { ctx: ac, node: gain } : null;
}
