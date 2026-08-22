// mm3StreamStore.ts — play a MiniMax-Music3 render while it is still rendering.
//
// MM3 renders in 200-frame windows, and a window's audio is FINAL the moment it
// is vocoded and cropped (engine mm3-pipeline.h §4). With the sliding pipeline
// the engine dispatches those windows WHILE the planner is still planning, so
// audio exists seconds into a render. It arrives as a chunked body of
// concatenated self-contained WAVs on GET /api/generate/mm3/stream/:jobId.
//
// ── Why a module singleton and not a hook's useState ────────────────────────
//
// THE ENGINE ALLOWS EXACTLY ONE READER PER JOB. Chunks are consumed as they are
// written, so a second connection would silently steal half the song from the
// first; the engine answers 409 instead. The player therefore cannot be
// per-component state — the queue card it lives in is rendered by
// InlineAudioQueue, which the ActivitySidebar mounts on several pages, and two
// simultaneous mounts would race for the one stream. One AudioContext, one
// connection, N subscribers.
//
// ── Why this is not useStreamAudio ──────────────────────────────────────────
//
// They look similar and are not the same problem. useStreamAudio plays STORM: a
// chain of INDEPENDENT generations, beat-matched and CROSSFADED into each other,
// where the cut point is a musical decision. These chunks are consecutive spans
// of ONE continuous signal with exact sample offsets, and they must HARD-SPLICE:
// any crossfade would sum two neighbours across a boundary that is supposed to
// be a plain concatenation. Three things this gets right that a copy of the
// STORM path would not:
//
//   1. THE SAMPLE RATE COMES FROM THE CHUNK. useStreamAudio hardcodes
//      `new AudioContext({ sampleRate: 48000 })` for ACE. MM3 is 44.1 kHz, and a
//      context at the wrong rate makes decodeAudioData resample every chunk —
//      after which the exact frame counts the splice depends on are no longer
//      exact. The rate is read from the first WAV header, before the context.
//   2. SCHEDULING IS IN SAMPLES. Each chunk starts at
//      `base + cumulativeFrames / rate`. Accumulating `ab.duration` instead
//      would accumulate float error across a hundred windows.
//   3. ON UNDERRUN IT STALLS, IT DOES NOT DROP. Whether the render outruns
//      playback is a per-configuration fact — measured on a 5090 at 30 steps, a
//      q8_0 fresh plan sustains 1.06x realtime but an f16 one manages 0.74x and
//      WILL be caught. When that happens the whole timeline shifts forward so
//      every later window still plays in full: a pause, not a hole in a track
//      someone is deciding whether to keep.
//
// The stream is an ADDITIONAL output. The finished WAV is written, saved and
// added to the library exactly as it is with streaming off, so a failure in
// this file costs a preview and nothing else.

import { useSyncExternalStore } from 'react';
import { extractWav, wavFrameCount, wavSampleRate } from '../utils/wavStream';

export interface Mm3StreamState {
  /** The job whose stream is open (or was last opened). */
  jobId: string | null;
  /** The network stream is open. */
  isPlaying: boolean;
  /** Something is still going on — the stream is open, or audio is still
   *  scheduled and audible. A finished render still has its tail to play. */
  active: boolean;
  /** The browser refused to start audio without a gesture. Show a button. */
  needsGesture: boolean;
  /** Windows received. */
  chunks: number;
  /** Seconds of audio received. */
  received: number;
  /** Seconds of audio played. */
  position: number;
  /** Seconds of audio scheduled ahead of the playhead. */
  ahead: number;
  /** Times the renderer was caught and playback had to rebuffer. */
  underruns: number;
  /** The engine closed the stream. */
  done: boolean;
  volume: number;
  /** Pre-buffer, seconds. */
  headroom: number;
  error: string | null;
}

const DEFAULT_HEADROOM = 5;

const INITIAL: Mm3StreamState = {
  jobId: null, isPlaying: false, active: false, needsGesture: false,
  chunks: 0, received: 0, position: 0, ahead: 0, underruns: 0, done: false,
  volume: 1.0, headroom: DEFAULT_HEADROOM, error: null,
};

let state: Mm3StreamState = INITIAL;
const listeners = new Set<() => void>();

function emit(patch: Partial<Mm3StreamState>): void {
  state = { ...state, ...patch };
  listeners.forEach(fn => fn());
}

let ac: AudioContext | null = null;
let gain: GainNode | null = null;
let abort: AbortController | null = null;
let raf = 0;
let sources: AudioBufferSourceNode[] = [];
/** Context time the first chunk starts at. 0 until the first chunk lands. */
let base = 0;
/** Frames (per channel) already scheduled — the splice cursor. */
let cursor = 0;
let rate = 0;
/** The network side is open. NOT the same as playback: the scheduled tail
 *  outlives the connection, and conflating the two is what makes a second
 *  render's play button dead. */
let netOpen = false;
let volume = 1.0;
let headroom = DEFAULT_HEADROOM;
/** Jobs already auto-started, so a remount does not reopen a stream the user
 *  deliberately stopped. */
const autoStarted = new Set<string>();
/** Failed auto-open attempts per job. The engine can legitimately say "not
 *  yet" for a moment (the job is queued behind another on the one GPU worker),
 *  and a single attempt that lost that race would leave the user with a silent
 *  card for the whole render. Bounded, because a permanent 409 must not loop. */
const attempts = new Map<string, number>();
const MAX_AUTO_ATTEMPTS = 3;

function tick(): void {
  if (!ac) return;
  const r = rate || 1;
  const scheduled = cursor / r;
  const ahead = base > 0 ? Math.max(0, base + scheduled - ac.currentTime) : 0;
  emit({
    // Audio CONSUMED, not wall time since `base` — the two differ by every
    // rebuffer stall, and only this one is monotonic.
    position: base > 0 ? Math.max(0, scheduled - ahead) : 0,
    ahead,
    active: netOpen || ahead > 0,
  });
  raf = requestAnimationFrame(tick);
}

function teardown(): void {
  netOpen = false;
  cancelAnimationFrame(raf);
  for (const s of sources) { try { s.stop(); } catch { /* already ended */ } }
  sources = [];
  try { ac?.close(); } catch { /* already closed */ }
  ac = null;
  gain = null;
  base = 0;
  cursor = 0;
  rate = 0;
}

async function open(jobId: string): Promise<void> {
  if (netOpen) return;
  // A previous render's context may still be around (its tail played out and
  // was never explicitly stopped). Start clean: the new stream has a different
  // origin, and reusing the old base would place every chunk of it in the past.
  teardown();
  netOpen = true;
  emit({ ...INITIAL, jobId, volume, headroom, isPlaying: true, active: true });

  // Logged because the failure mode of this feature is SILENCE, and silence is
  // indistinguishable from "the engine has not produced a window yet". These
  // four lines are the difference between a bug report and a diagnosis.
  console.log(`[MM3 Stream] opening ${jobId}`);
  const ctrl = new AbortController();
  abort = ctrl;
  raf = requestAnimationFrame(tick);

  let chunks = 0, frames = 0, underruns = 0;

  try {
    const res = await fetch(`/api/generate/mm3/stream/${encodeURIComponent(jobId)}`, { signal: ctrl.signal });
    if (!res.ok || !res.body) {
      // 409 is the engine declining (not a streaming job, already finished,
      // already has a reader). Surface its own sentence — those messages are
      // written to be read.
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

        // First chunk: build the context AT THE STREAM'S RATE (see the header),
        // then set the playback origin one headroom ahead.
        if (!ac) {
          const ctx = new AudioContext({ sampleRate: r });
          const g = ctx.createGain();
          g.gain.value = volume;
          g.connect(ctx.destination);
          ac = ctx;
          gain = g;
          rate = r;
          // Autoplay policy: a context created without a user gesture starts
          // suspended. Try to resume, and if the browser still says no, say so
          // rather than sitting silently with a full buffer.
          if (ctx.state === 'suspended') {
            try { await ctx.resume(); } catch { /* needs a gesture */ }
          }
          emit({ needsGesture: ctx.state === 'suspended' });
          base = ctx.currentTime + headroom;
        }
        const ctx = ac;
        const g = gain;
        if (!ctx || !g) break;

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

        // Underrun: this window is due before it arrived, i.e. the renderer has
        // been caught. Shift the ENTIRE timeline forward by the deficit plus one
        // headroom, so this window and every later one still play in full.
        // Already-scheduled windows are unaffected; only future starts move.
        if (base + cursor / rate < ctx.currentTime) {
          const deficit = ctx.currentTime - (base + cursor / rate);
          base += deficit + Math.max(1, headroom);
          underruns++;
        }

        const at = base + cursor / rate;
        const src = ctx.createBufferSource();
        src.buffer = ab;
        src.connect(g);
        // No ramp, no fade: consecutive spans of one signal, spliced at the
        // sample the engine cropped them to.
        src.start(at);
        sources.push(src);
        src.onended = () => { sources = sources.filter(s => s !== src); };

        // The cursor advances by the HEADER's frame count, not ab.length: they
        // agree when the context rate matches (which is why it is built from the
        // header), and the header is the engine's arithmetic, not the browser's.
        cursor += n;
        chunks++;
        frames += n;
        if (chunks === 1) {
          console.log(`[MM3 Stream] first window: ${(n / rate).toFixed(2)}s @ ${rate} Hz, ` +
                      `context ${ctx.state}, starts in ${(at - ctx.currentTime).toFixed(1)}s`);
        }
        emit({ chunks, received: frames / rate, underruns });
      }
    }
    console.log(`[MM3 Stream] closed: ${chunks} window(s), ${(frames / (rate || 1)).toFixed(1)}s audio, ${underruns} rebuffer(s)`);
    emit({ done: true });
  } catch (e: unknown) {
    const err = e as Error;
    if (err?.name !== 'AbortError') {
      console.warn(`[MM3 Stream] ${jobId}: ${err?.message || String(e)}`);
      emit({ error: err?.message || String(e) });
      // Nothing arrived at all — most likely the job had not reached the engine
      // yet. Retry a couple of times before giving up and leaving the message.
      if (chunks === 0) {
        const n = (attempts.get(jobId) ?? 0) + 1;
        attempts.set(jobId, n);
        if (n < MAX_AUTO_ATTEMPTS) {
          setTimeout(() => {
            autoStarted.delete(jobId);
            mm3StreamEnsure(jobId);
          }, 2000);
        }
      }
    }
  } finally {
    // The scheduled buffers keep playing to the end of what was received; only
    // the NETWORK side stops here. Tearing the context down would cut the last
    // few seconds off every render — the tail is real audio. tick() keeps
    // running off `ac` and reports `active: false` once it has drained.
    netOpen = false;
    abort = null;
    emit({ isPlaying: false });
  }
}

/** Open the stream for `jobId` if nothing is playing it yet.
 *
 *  Called from an effect, so it must be idempotent and must not reopen a stream
 *  the user stopped on purpose — hence `autoStarted`. A null jobId is NOT a
 *  teardown: a render that has just finished still has seconds of audio
 *  scheduled, and yanking the context would cut its ending off. */
export function mm3StreamEnsure(jobId: string | null): void {
  if (!jobId || netOpen || autoStarted.has(jobId)) return;
  autoStarted.add(jobId);
  void open(jobId);
}

/** Open the stream because the user asked — after they stopped it, or after
 *  the auto-open gave up. Bypasses `autoStarted` and resets its retry budget. */
export function mm3StreamStart(jobId: string): void {
  if (netOpen) return;
  autoStarted.add(jobId);
  attempts.delete(jobId);
  void open(jobId);
}

/** Retry after the browser refused to start audio without a gesture. MUST be
 *  called from a click handler. */
export function mm3StreamResume(): void {
  if (!ac) return;
  void ac.resume().then(() => emit({ needsGesture: ac?.state === 'suspended' }));
}

/** Stop listening. The render is unaffected — this closes a preview, it does
 *  not cancel a job. */
export function mm3StreamStop(): void {
  abort?.abort();
  abort = null;
  teardown();
  emit({ ...INITIAL, volume, headroom });
}

export function mm3StreamSetVolume(v: number): void {
  volume = v;
  if (gain) gain.gain.value = v;
  emit({ volume: v });
}

/** Only takes effect on the NEXT stream — the origin of an in-flight one is
 *  fixed, and moving it would misplace every chunk already scheduled. */
export function mm3StreamSetHeadroom(s: number): void {
  headroom = s;
  emit({ headroom: s });
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

export function useMm3StreamAudio(): Mm3StreamState {
  return useSyncExternalStore(subscribe, () => state, () => state);
}
