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
  /** Which take of that render is AUDIBLE. An ensemble render streams every
   *  take at once — they all buffer, one plays — so this says which one the
   *  transport below is describing. 0 for an ordinary render. */
  take: number;
  /** How many takes this render is streaming. 1 for an ordinary render. */
  takeCount: number;
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
  jobId: null, take: 0, takeCount: 1, connected: false, done: false, playing: false,
  position: 0, received: 0, expected: 0, chunks: 0, underruns: 0,
  needsGesture: false, volume: 1.0, error: null, peaksVersion: 0,
};

let state: Mm3StreamState = INITIAL;
const listeners = new Set<() => void>();

function emit(patch: Partial<Mm3StreamState>): void {
  state = { ...state, ...patch };
  listeners.forEach(fn => fn());
}

/** Republish the ACTIVE receiver's shape. Called after anything that changes
 *  which take you are hearing, or that take's contents — the transport fields
 *  (playing, position) belong to the player and are left alone. */
function emitActive(extra: Partial<Mm3StreamState> = {}): void {
  const r = cur();
  if (!r) { emit(extra); return; }
  emit({
    jobId: r.jobId, take: r.take, takeCount: takeCountFor(r.jobId),
    connected: r.connected, done: r.done, received: r.received, expected: r.expected,
    chunks: r.chunks, underruns: r.underruns, error: r.error,
    peaksVersion: r.peaksVersion,
    ...extra,
  });
}

function takeCountFor(jobId: string): number {
  let n = 0;
  for (const r of receivers.values()) if (r.jobId === jobId) n++;
  return Math.max(1, n);
}

interface Seg {
  buf: AudioBuffer;
  /** First frame of this segment in the finished track. */
  start: number;
  frames: number;
}

// ── Receivers and the player ────────────────────────────────────────────────
//
// An ensemble render streams K takes AT ONCE, and they all have to be read:
// the engine drops a stream nobody is draining once it buffers past a
// threshold, so an unread take loses its live audio entirely. But three songs
// through the speakers at once is not listening, it is noise.
//
// So the store splits in two. A RECEIVER per take owns the network read and
// the decoded audio — segments, length, waveform — and nothing else. The
// PLAYER is a singleton: one AudioContext, one gain, one set of scheduled
// sources, pointed at whichever receiver is currently selected. Switching
// takes re-anchors the player onto a different receiver's buffer; every take
// keeps its own playhead, so coming back to one resumes where you left it.
//
// Memory: a decoded three-minute stereo take is ~63 MB, so four of them is
// ~250 MB. That is the price of being able to hear any of them instantly, and
// it is released when the group is torn down.
interface Receiver {
  key: string;      // `${jobId}:${take}`
  jobId: string;
  take: number;
  segs: Seg[];
  totalFrames: number;
  rate: number;
  /** Waveform envelope, one max-abs value per 1/PEAK_RATE second. */
  peaks: Float32Array;
  peakCount: number;
  /** Playhead for THIS take while it is not the audible one. */
  pausedFrame: number;
  connected: boolean;
  done: boolean;
  received: number;
  expected: number;
  chunks: number;
  underruns: number;
  error: string | null;
  abort: AbortController | null;
  netOpen: boolean;
  peaksVersion: number;
}

const receivers = new Map<string, Receiver>();
/** The receiver the PLAYER is pointed at — the one you can hear. */
let activeKey: string | null = null;

const rkey = (jobId: string, take: number): string => `${jobId}:${take}`;

function newReceiver(jobId: string, take: number, expected: number): Receiver {
  return {
    key: rkey(jobId, take), jobId, take,
    segs: [], totalFrames: 0, rate: 0,
    peaks: new Float32Array(0), peakCount: 0, pausedFrame: 0,
    connected: false, done: false, received: 0, expected,
    chunks: 0, underruns: 0, error: null, abort: null, netOpen: false,
    peaksVersion: 0,
  };
}

/** The audible receiver, or null before anything is open. */
function cur(): Receiver | null {
  return activeKey ? receivers.get(activeKey) ?? null : null;
}

let ac: AudioContext | null = null;
let gain: GainNode | null = null;
let raf = 0;

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

let volume = 1.0;

/** True while ANY take of the current group is still reading. */
export function mm3StreamReceiving(): boolean {
  for (const r of receivers.values()) if (r.netOpen) return true;
  return false;
}

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
  const r = cur();
  if (!ac || !playing || !r) return pausedFrame;
  const p = anchorFrame + Math.max(0, ac.currentTime - anchorTime) * (r.rate || 1);
  return Math.min(p, r.totalFrames);
}

function tick(): void {
  // RE-ARM UNCONDITIONALLY. `ac` does not exist until the first window lands,
  // which on a fresh plan is ten seconds after the stream opens — an early
  // `return` here (as the first cut had) kills the loop before there is
  // anything to report and never restarts it, so the playhead never moves for
  // the whole render while the audio plays perfectly.
  const r = cur();
  if (ac && r) {
    const pf = positionFrames();
    // The end of a finished stream is the end of the track. Without this the
    // transport sits at 100% still claiming to play, and the play button would
    // do nothing when pressed.
    if (playing && r.done && r.totalFrames > 0 && pf >= r.totalFrames) {
      playing = false;
      wantPlay = false;
      pausedFrame = r.totalFrames;
      r.pausedFrame = r.totalFrames;
      emit({ playing: false, position: r.totalFrames / (r.rate || 1) });
    } else {
      emit({ position: pf / (r.rate || 1) });
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
  const r = cur();
  if (!ac || !gain || !r || scheduled.has(i)) return;
  const seg = r.segs[i];
  if (!seg || seg.start + seg.frames <= anchorFrame) return;   // entirely played
  const at = anchorTime + (seg.start - anchorFrame) / r.rate;
  let offsetFrames = Math.max(0, anchorFrame - seg.start);
  // A window whose due time has already passed — decoded a beat late — must
  // not simply be started NOW: that shifts it later than the audio before it
  // ended, which is a hole. Skip into the buffer by however long it is late,
  // so the splice still lands where the previous window stopped. Being late by
  // more than a window is an underrun and is handled by re-anchoring instead.
  let startAt = at;
  if (at < ac.currentTime) {
    offsetFrames += (ac.currentTime - at) * r.rate;
    startAt = ac.currentTime;
  }
  if (offsetFrames >= seg.buf.length) return;   // entirely in the past
  const src = ac.createBufferSource();
  src.buffer = seg.buf;
  src.connect(gain);
  // No ramp, no fade: consecutive spans of one signal, spliced at the sample
  // the engine cropped them to.
  src.start(startAt, offsetFrames / r.rate);
  sources.push(src);
  src.onended = () => { sources = sources.filter(s => s !== src); };
  scheduled.add(i);
}

/** Re-anchor: frame `frame` becomes due `delay` seconds from now, and everything
 *  from there is rescheduled. The one primitive behind start, seek and rebuffer. */
function anchorAt(frame: number, delay: number): void {
  const r = cur();
  if (!ac || !r) return;
  stopSources();
  anchorFrame = Math.max(0, Math.min(frame, r.totalFrames));
  anchorTime = ac.currentTime + delay;
  for (let i = 0; i < r.segs.length; i++) scheduleSeg(i);
}

// ── Waveform ────────────────────────────────────────────────────────────────

function appendPeaks(r: Receiver, buf: AudioBuffer): void {
  const per = Math.max(1, Math.round(buf.sampleRate / PEAK_RATE));
  const n = Math.ceil(buf.length / per);
  if (r.peakCount + n > r.peaks.length) {
    const grown = new Float32Array(Math.max(r.peaks.length * 2, r.peakCount + n + 1024));
    grown.set(r.peaks.subarray(0, r.peakCount));
    r.peaks = grown;
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
    r.peaks[r.peakCount++] = m;
  }
}

/** The waveform so far. Deliberately NOT React state — it is thousands of floats
 *  that a canvas reads directly; `peaksVersion` is the render trigger.
 *
 *  With no arguments it is the AUDIBLE take, which is what the play bar draws.
 *  Pass a job and take to draw a background one — an ensemble's cards each show
 *  their own song growing, and they are not the one you are listening to. */
export function mm3StreamPeaks(jobId?: string, take = 0): { peaks: Float32Array; count: number; rate: number } {
  const r = jobId ? receivers.get(rkey(jobId, take)) : cur();
  return r
    ? { peaks: r.peaks, count: r.peakCount, rate: PEAK_RATE }
    : { peaks: new Float32Array(0), count: 0, rate: PEAK_RATE };
}

/** One take's live progress, for its own card: how much audio has arrived and
 *  how much is coming. Null until that take's stream is open. */
export function mm3StreamTakeState(jobId: string, take: number): {
  received: number; expected: number; done: boolean; connected: boolean;
  chunks: number; error: string | null; audible: boolean;
} | null {
  const r = receivers.get(rkey(jobId, take));
  if (!r) return null;
  return {
    received: r.received, expected: r.expected, done: r.done, connected: r.connected,
    chunks: r.chunks, error: r.error, audible: activeKey === r.key,
  };
}

// ── Connection ──────────────────────────────────────────────────────────────

/** Drop the player AND every receiver. Called when a DIFFERENT render's stream
 *  opens, and by mm3StreamStop — never between takes of the same render, which
 *  is the whole reason they coexist. */
function teardown(): void {
  playing = false;
  cancelAnimationFrame(raf);
  raf = 0;
  stopSources();
  try { ac?.close(); } catch { /* already closed */ }
  ac = null;
  gain = null;
  for (const r of receivers.values()) {
    r.abort?.abort();
    r.abort = null;
    r.netOpen = false;
  }
  receivers.clear();
  activeKey = null;
  anchorFrame = 0;
  anchorTime = 0;
  pausedFrame = 0;
}

async function open(jobId: string, take: number, expected: number, autoPlay: boolean): Promise<void> {
  const key = rkey(jobId, take);
  if (receivers.get(key)?.netOpen) return;
  // A DIFFERENT render's audio is finished with — its tail played out and was
  // never stopped. Start clean. Takes of the SAME render live alongside each
  // other, which is the whole point, so only a change of job tears down.
  const existing = cur();
  if (existing && existing.jobId !== jobId) teardown();

  const r = newReceiver(jobId, take, expected);
  r.netOpen = true;
  r.connected = true;
  receivers.set(key, r);
  // The first take opened becomes the audible one; later takes buffer silently
  // until selected. Auto-play intent belongs to the player, not the receiver.
  if (!activeKey || !receivers.has(activeKey)) {
    activeKey = key;
    wantPlay = autoPlay;
    emit({ ...INITIAL, jobId, take, takeCount: takeCountFor(jobId), expected, volume, connected: true });
  } else {
    emitActive();
  }

  // Logged because the failure mode of this feature is SILENCE, and silence is
  // indistinguishable from "the engine has not produced a window yet".
  console.log(`[MM3 Stream] opening ${jobId} take ${take} (expecting ${expected.toFixed(1)}s)`);
  const ctrl = new AbortController();
  r.abort = ctrl;
  if (!raf) raf = requestAnimationFrame(tick);

  let chunks = 0, underruns = 0;

  try {
    const url = `/api/generate/mm3/stream/${encodeURIComponent(jobId)}${take > 0 ? `?take=${take}` : ''}`;
    const res = await fetch(url, { signal: ctrl.signal });
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
      if (!r.netOpen) break;

      const merged = new Uint8Array(buf.length + value.length);
      merged.set(buf); merged.set(value, buf.length);
      buf = merged;

      for (;;) {
        const w = extractWav(buf);
        if (!w) break;
        buf = w.remaining;

        const sr = wavSampleRate(w.data);
        const n = wavFrameCount(w.data);
        if (!sr || !n) continue;

        // First chunk: build the context AT THE STREAM'S RATE (see the header).
        // Shared by every take — they are windows of the same engine at the same
        // rate, and a second context would mean a second clock to splice against.
        if (!ac) {
          const ctx = new AudioContext({ sampleRate: sr });
          const g = ctx.createGain();
          g.gain.value = volume;
          g.connect(ctx.destination);
          ac = ctx;
          gain = g;
          // Autoplay policy: a context created without a user gesture starts
          // suspended. Try to resume; if the browser still says no, say so
          // rather than sitting silently on a full buffer.
          if (ctx.state === 'suspended') {
            try { await ctx.resume(); } catch { /* needs a gesture */ }
          }
          emit({ needsGesture: ctx.state === 'suspended' });
        }
        r.rate = sr;
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
        if (!r.netOpen) break;

        // THE TIMELINE IS THE DECODED AUDIO, not what the header claimed.
        //
        // Windows were placed using the WAV header's frame count while the
        // thing actually played is the decoded buffer. Any difference between
        // the two — a decoder that trims, a header that rounds — is inserted as
        // SILENCE at every seam, because the next window is scheduled from the
        // header's idea of where this one ended. The engine's bytes are gapless
        // (check-mm3-stream proves the streamed PCM is byte-identical to the
        // saved file), so a hole between windows can only have come from here.
        const frames = ab.length || n;
        if (chunks === 0 && frames !== n) {
          console.warn(`[MM3 Stream] window header says ${n} frames, decoded ${frames} `
            + '— using the decoded length (the difference would be an audible gap per window)');
        }
        const seg: Seg = { buf: ab, start: r.totalFrames, frames };
        r.segs.push(seg);
        r.totalFrames += frames;
        appendPeaks(r, ab);
        chunks++;
        r.chunks = chunks;
        r.received = r.totalFrames / (r.rate || 1);
        r.peaksVersion++;

        // Everything below drives the TRANSPORT, so it applies only to the take
        // you are hearing. The others accumulate silently — that is what makes
        // switching to one instant instead of restarting it.
        const audible = activeKey === r.key;

        if (chunks === 1) {
          console.log(`[MM3 Stream] ${jobId} take ${take} first window: `
            + `${(n / r.rate).toFixed(2)}s @ ${r.rate} Hz, context ${ctx.state}`);
          // Start the transport one headroom from now. The playhead holds at 0
          // until then (positionFrames' max(0, …)) — the pre-buffer the
          // reference Space uses.
          if (audible && wantPlay) {
            playing = true;
            anchorAt(0, HEADROOM);
          }
        } else if (audible && playing) {
          const dueAt = anchorTime + (seg.start - anchorFrame) / r.rate;
          if (dueAt < ctx.currentTime) {
            // The renderer has been caught. Re-anchor so this window and every
            // later one still play IN FULL, one headroom from now: a stall
            // rather than a hole. Starting it part-way in would keep the clock
            // and lose audio, and this is a track someone is deciding whether
            // to keep.
            underruns++;
            r.underruns = underruns;
            anchorAt(seg.start, HEADROOM);
          } else {
            scheduleSeg(r.segs.length - 1);
          }
        }

        // A background take still has to publish, or its card would never show
        // it growing — which is the one thing an ensemble stream is for.
        if (audible) emitActive({ playing });
        else emit({ peaksVersion: state.peaksVersion + 1 });
      }
    }
    console.log(`[MM3 Stream] ${jobId} take ${take} closed: ${chunks} window(s), `
      + `${(r.totalFrames / (r.rate || 1)).toFixed(1)}s, ${underruns} rebuffer(s)`);
    // The engine is done, so `received` IS the length. Replacing `expected`
    // here is what makes a card that finished early (EOS) stop showing a gap it
    // will never fill — and takes DO finish at different lengths, because each
    // hits its own EOS.
    r.done = true;
    r.connected = false;
    r.expected = r.totalFrames / (r.rate || 1);
    if (activeKey === r.key) emitActive();
    else emit({ peaksVersion: state.peaksVersion + 1 });
  } catch (e: unknown) {
    const err = e as Error;
    if (err?.name !== 'AbortError') {
      console.warn(`[MM3 Stream] ${jobId} take ${take}: ${err?.message || String(e)}`);
      r.error = err?.message || String(e);
      r.connected = false;
      if (activeKey === r.key) emitActive();
      // Nothing arrived at all — most likely the job had not reached the engine
      // yet. Retry a couple of times before giving up. Keyed per TAKE, because
      // one take losing the race must not spend another take's budget.
      if (chunks === 0) {
        const n = (attempts.get(key) ?? 0) + 1;
        attempts.set(key, n);
        if (n < MAX_AUTO_ATTEMPTS) {
          setTimeout(() => {
            autoOpened.delete(key);
            receivers.delete(key);
            mm3StreamEnsure(jobId, expected, autoPlay, take);
          }, 2000);
        }
      }
    }
  } finally {
    // The scheduled tail keeps playing; only the NETWORK side stops here.
    r.netOpen = false;
    r.abort = null;
    r.connected = false;
    if (activeKey === r.key) emit({ connected: false });
  }
}

// ── Public API ──────────────────────────────────────────────────────────────

/** Open the stream for `jobId` if nothing is playing it yet. Idempotent, and it
 *  will not reopen a stream the user stopped on purpose. A null jobId is NOT a
 *  teardown: a render that has just finished still has audio to play. */
export function mm3StreamEnsure(jobId: string | null, expected = 0, autoPlay = true, take = 0): void {
  if (!jobId) return;
  const key = rkey(jobId, take);
  if (autoOpened.has(key) || receivers.get(key)?.netOpen) return;
  autoOpened.add(key);
  void open(jobId, take, expected, autoPlay);
}

/** Open EVERY take of an ensemble render at once.
 *
 *  All of them must be read, not just the one being listened to: the engine
 *  drops a stream nobody is draining once it buffers past a threshold, so an
 *  unopened take silently loses its live audio (the saved file is unaffected).
 *  Only take 0 is allowed to auto-play; the rest buffer until selected. */
export function mm3StreamEnsureTakes(jobId: string | null, takes: number, expected = 0, autoPlay = true): void {
  if (!jobId) return;
  for (let t = 0; t < Math.max(1, takes); t++) {
    mm3StreamEnsure(jobId, expected, autoPlay && t === 0, t);
  }
}

/** Make a take the audible one. The others keep receiving.
 *
 *  Each take remembers its own playhead, so switching away and back resumes
 *  where you were rather than restarting. Playback INTENT carries across: if
 *  you were listening, you are still listening, just to a different song. */
export function mm3StreamSelect(jobId: string, take: number): void {
  const key = rkey(jobId, take);
  const next = receivers.get(key);
  if (!next || activeKey === key) return;
  const prev = cur();
  if (prev) prev.pausedFrame = playing ? positionFrames() : pausedFrame;
  const wasPlaying = playing;
  stopSources();
  playing = false;
  activeKey = key;
  pausedFrame = Math.min(next.pausedFrame, next.totalFrames);
  emitActive({ playing: false, position: pausedFrame / (next.rate || 1) });
  if (wasPlaying && next.totalFrames > 0) mm3StreamPlay();
  console.log(`[MM3 Stream] now hearing ${jobId} take ${take}`);
}

/** Which take is audible, so a card can show itself as the one playing. */
export function mm3StreamActiveTake(): { jobId: string; take: number } | null {
  const r = cur();
  return r ? { jobId: r.jobId, take: r.take } : null;
}

/** Open because the user asked — after they stopped, or after the auto-open gave
 *  up. Bypasses the once-only guard and resets the retry budget. */
export function mm3StreamStart(jobId: string, expected = 0, take = 0): void {
  const key = rkey(jobId, take);
  if (receivers.get(key)?.netOpen) return;
  autoOpened.add(key);
  attempts.delete(key);
  void open(jobId, take, expected, true);
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
  const r = cur();
  if (r) r.pausedFrame = pausedFrame;
  emit({ playing: false, position: pausedFrame / (r?.rate || 1) });
}

export function mm3StreamToggle(): void {
  if (playing) mm3StreamPause(); else mm3StreamPlay();
}

/** True once there is audio to play. The generations card shows a live render
 *  before this is true (so the user sees it coming) but its transport only
 *  means anything after. */
export function mm3StreamHasAudio(): boolean {
  return (cur()?.totalFrames ?? 0) > 0;
}

/** Seek inside the audio received so far. Beyond that there is nothing to play,
 *  so it clamps rather than pretending. */
export function mm3StreamSeek(sec: number): void {
  const r = cur();
  if (!ac || !r || !r.rate) return;
  const frame = Math.max(0, Math.min(sec * r.rate, r.totalFrames));
  pausedFrame = frame;
  r.pausedFrame = frame;
  if (playing) anchorAt(frame, 0.05);
  emit({ position: frame / r.rate });
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
/** Close EVERY take's stream and drop the audio. The render is unaffected —
 *  this ends a preview, it does not cancel a job. */
export function mm3StreamStop(): void {
  for (const r of receivers.values()) {
    r.abort?.abort();
    r.abort = null;
    r.netOpen = false;
  }
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
