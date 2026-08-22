// useMm3StreamAudio.ts — play a MiniMax-Music3 render while it is still rendering.
//
// MM3 renders in 200-frame windows with a 100-frame hop, and a window's audio
// is FINAL the moment it has been vocoded and cropped (engine
// mm3-pipeline.h §4). The engine emits each one as a self-contained WAV on
// GET /mm3/stream; the Node server pipes that through unchanged at
// GET /api/generate/mm3/stream/:jobId. This hook reads that byte stream,
// decodes each WAV and schedules it in Web Audio.
//
// ── Why this is NOT useStreamAudio ──────────────────────────────────────────
//
// They look similar and are not the same problem. useStreamAudio plays STORM:
// a chain of INDEPENDENT generations, beat-matched and CROSSFADED into each
// other, where the cut point is a musical decision. These chunks are
// consecutive spans of ONE continuous signal with exact sample offsets, and
// they must HARD-SPLICE: any crossfade at all would sum two neighbours across
// a boundary that is supposed to be a plain concatenation, and the seam would
// be audible. Sharing that scheduler to save a file would have shipped that.
//
// Three things this gets right that a copy of the STORM path would not:
//
//   1. THE SAMPLE RATE COMES FROM THE CHUNK. useStreamAudio hardcodes
//      `new AudioContext({ sampleRate: 48000 })` for ACE. MM3 is 44.1 kHz. A
//      context at the wrong rate makes decodeAudioData resample every chunk,
//      and then the exact frame counts the splice arithmetic depends on stop
//      being exact. The rate is read from the first WAV header, before the
//      context is created.
//   2. SCHEDULING IS IN SAMPLES, NOT SECONDS. Each chunk starts at
//      `base + cumulativeFrames / rate`, with `cumulativeFrames` accumulated
//      from the header frame counts. Accumulating `ab.duration` instead would
//      accumulate float error across a hundred windows.
//   3. HEADROOM IS A PRE-BUFFER, NOT A CROSSFADE. `base` is set to
//      `currentTime + headroom` on the first chunk, mirroring the reference
//      Space, and is pushed forward again if the renderer ever falls behind.
//
// ON UNDERRUN, STALL — DO NOT DROP. Whether the render outruns playback is a
// per-configuration fact, not a given: measured on a 5090 at 30 steps, a
// q8_0 fresh plan sustains 1.00x realtime and an AR-cache hit 1.98x, but an
// f16 fresh plan manages 0.74x and WILL be caught. When that happens the whole
// timeline is shifted forward — every later window plays in full, one
// headroom later. The alternative (start the late window part-way in, so it
// still ends where it was due) keeps the clock but puts a hole in the song,
// and this is a preview of a track someone is deciding whether to keep.
//
// The stream is an ADDITIONAL output. The finished WAV is still written, saved
// and added to the library exactly as it is with streaming off — nothing here
// is on the path to a saved song, so a failure in this file costs a preview.

import { useCallback, useEffect, useRef, useState } from 'react';
import { extractWav, wavFrameCount, wavSampleRate } from '../utils/wavStream';

export interface Mm3StreamState {
  /** A stream is open and audio is scheduled (or about to be). */
  isPlaying: boolean;
  /** Windows received from the engine. */
  chunks: number;
  /** Seconds of audio received so far. */
  received: number;
  /** Seconds elapsed since the first chunk started playing. */
  position: number;
  /** Seconds of audio scheduled ahead of the playhead. 0 means the player has
   *  caught up with the renderer and the next window will arrive late. */
  ahead: number;
  /** Chunks that arrived after their scheduled start time — i.e. the render
   *  fell behind playback. Surfaced rather than hidden: it is the one number
   *  that says "raise the headroom". */
  underruns: number;
  /** The engine finished and closed the stream. */
  done: boolean;
  /** Something is still going on — the stream is open, or audio is still
   *  scheduled and audible. The mount condition for the player: a render that
   *  has FINISHED still has its tail playing out, and yanking the transport
   *  away mid-tail would leave audio running with no way to stop it. */
  active: boolean;
  volume: number;
  /** Pre-buffer, seconds. */
  headroom: number;
  error: string | null;
}

/** Matches the reference Space's default pre-buffer. Long enough to ride out a
 *  slow window, short enough that "press play, hear music" still feels true. */
const DEFAULT_HEADROOM = 5;

const INITIAL: Mm3StreamState = {
  isPlaying: false, chunks: 0, received: 0, position: 0, ahead: 0,
  underruns: 0, done: false, active: false, volume: 1.0, headroom: DEFAULT_HEADROOM, error: null,
};

export function useMm3StreamAudio() {
  const [state, setState] = useState<Mm3StreamState>(INITIAL);

  const acRef        = useRef<AudioContext | null>(null);
  const gainRef      = useRef<GainNode | null>(null);
  const abortRef     = useRef<AbortController | null>(null);
  const rafRef       = useRef(0);
  /** The NETWORK side is open. Goes false when the body ends, which is not the
   *  same as playback ending — the scheduled tail plays on. Keeping these two
   *  as one flag is what would make a second render's Listen button dead. */
  const netRef       = useRef(false);
  const sourcesRef   = useRef<AudioBufferSourceNode[]>([]);
  /** Context time the first chunk starts at. 0 until the first chunk lands. */
  const baseRef      = useRef(0);
  /** Frames (per channel) already scheduled — the splice cursor. */
  const cursorRef    = useRef(0);
  /** Seconds `base` has been pushed forward by rebuffering. Playback position
   *  is derived from audio consumed, not from `base`, so that a stall does not
   *  make the readout jump backwards. */
  const stalledRef   = useRef(0);
  const rateRef      = useRef(0);
  const volumeRef    = useRef(1.0);
  const headroomRef  = useRef(DEFAULT_HEADROOM);

  const tick = useCallback(() => {
    const ac = acRef.current;
    // Driven by the AudioContext's existence, NOT by the network flag: after
    // the body ends there are still seconds of scheduled audio to report on.
    if (!ac) return;
    const rate = rateRef.current || 1;
    const scheduled = cursorRef.current / rate;
    const endOfBuffer = baseRef.current + scheduled;
    const ahead = baseRef.current > 0 ? Math.max(0, endOfBuffer - ac.currentTime) : 0;
    setState(p => ({
      ...p,
      // Audio CONSUMED, not wall time since `base` — the two differ by every
      // rebuffer stall, and only this one is monotonic.
      position: baseRef.current > 0 ? Math.max(0, scheduled - ahead) : 0,
      ahead,
      active: netRef.current || ahead > 0,
    }));
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const teardown = useCallback(() => {
    netRef.current = false;
    cancelAnimationFrame(rafRef.current);
    for (const s of sourcesRef.current) { try { s.stop(); } catch { /* already ended */ } }
    sourcesRef.current = [];
    try { acRef.current?.close(); } catch { /* already closed */ }
    acRef.current = null;
    gainRef.current = null;
    baseRef.current = 0;
    cursorRef.current = 0;
    stalledRef.current = 0;
    rateRef.current = 0;
  }, []);

  /** Open the stream and start playing. MUST be called from a user gesture —
   *  the AudioContext is created here, and browsers refuse to start one
   *  otherwise. */
  const start = useCallback(async (jobId: string) => {
    if (netRef.current) return;
    // A previous render's context may still be around (its tail played out and
    // was never explicitly stopped). Start clean — the new stream almost
    // certainly has a different origin, and reusing the old base would place
    // every chunk of this render in the past.
    teardown();
    netRef.current = true;
    setState({ ...INITIAL, volume: volumeRef.current, headroom: headroomRef.current, isPlaying: true, active: true });

    const abort = new AbortController();
    abortRef.current = abort;
    rafRef.current = requestAnimationFrame(tick);

    let chunks = 0;
    let frames = 0;
    let underruns = 0;

    try {
      const res = await fetch(`/api/generate/mm3/stream/${encodeURIComponent(jobId)}`, { signal: abort.signal });
      if (!res.ok || !res.body) {
        // 409 is the engine declining (not a streaming job, already finished,
        // already has a reader). Report the engine's own sentence — those
        // messages are written to be read.
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
        if (!netRef.current) break;

        const merged = new Uint8Array(buf.length + value.length);
        merged.set(buf); merged.set(value, buf.length);
        buf = merged;

        for (;;) {
          const w = extractWav(buf);
          if (!w) break;
          buf = w.remaining;

          const rate = wavSampleRate(w.data);
          const n = wavFrameCount(w.data);
          if (!rate || !n) continue;

          // First chunk: build the context AT THE STREAM'S RATE (see the
          // header note), then set the playback origin one headroom ahead.
          if (!acRef.current) {
            const ac = new AudioContext({ sampleRate: rate });
            const g = ac.createGain();
            g.gain.value = volumeRef.current;
            g.connect(ac.destination);
            acRef.current = ac;
            gainRef.current = g;
            rateRef.current = rate;
            baseRef.current = ac.currentTime + headroomRef.current;
          }
          const ac = acRef.current;
          const g = gainRef.current;
          if (!ac || !g) break;

          // decodeAudioData detaches the ArrayBuffer it is given, so hand it a
          // copy — `buf` may still alias the same backing store.
          const copy = w.data.slice();
          let ab: AudioBuffer;
          try {
            ab = await ac.decodeAudioData(copy.buffer as ArrayBuffer);
          } catch {
            continue;  // a torn chunk costs one window, not the stream
          }
          if (!netRef.current) break;

          // Underrun: this window is due before it arrived, i.e. the renderer
          // has been caught. Shift the ENTIRE timeline forward by the deficit
          // plus one headroom, so this window and every later one still play in
          // full — a stall, not a hole. Already-scheduled windows are unaffected
          // (they have played); only future start times move.
          if (baseRef.current + cursorRef.current / rate < ac.currentTime) {
            const deficit = ac.currentTime - (baseRef.current + cursorRef.current / rate);
            const rebuffer = deficit + Math.max(1, headroomRef.current);
            baseRef.current += rebuffer;
            stalledRef.current += rebuffer;
            underruns++;
          }

          const at = baseRef.current + cursorRef.current / rate;
          const src = ac.createBufferSource();
          src.buffer = ab;
          src.connect(g);
          // No ramp, no fade: consecutive spans of one signal, spliced at the
          // sample the engine cropped them to.
          src.start(at);
          sourcesRef.current.push(src);
          src.onended = () => {
            sourcesRef.current = sourcesRef.current.filter(s => s !== src);
          };

          // The cursor advances by the HEADER's frame count, not ab.length:
          // they agree when the context rate matches (which is the point of
          // building it from the header), and the header is the engine's own
          // arithmetic rather than the browser's.
          cursorRef.current += n;
          chunks++;
          frames += n;
          setState(p => ({ ...p, chunks, received: frames / rate, underruns }));
        }
      }
      setState(p => ({ ...p, done: true }));
    } catch (e: unknown) {
      const err = e as Error;
      if (err?.name !== 'AbortError') setState(p => ({ ...p, error: err?.message || String(e) }));
    } finally {
      // The scheduled buffers keep playing to the end of what was received;
      // only the NETWORK side stops here. Tearing the context down would cut
      // the last few seconds off every render — the tail is real audio.
      netRef.current = false;
      abortRef.current = null;
      setState(p => ({ ...p, isPlaying: false }));
      // tick() keeps running off acRef and reports `active` false once the
      // tail has drained, which is what unmounts the player.
    }
  }, [tick, teardown]);

  /** Stop playback and drop the connection. The render is unaffected — this
   *  closes a preview, it does not cancel a job. */
  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    teardown();
    setState(p => ({ ...INITIAL, volume: p.volume, headroom: p.headroom }));
  }, [teardown]);

  const setVolume = useCallback((v: number) => {
    volumeRef.current = v;
    if (gainRef.current) gainRef.current.gain.value = v;
    setState(p => ({ ...p, volume: v }));
  }, []);

  /** Only takes effect on the NEXT start — the origin of an in-flight stream
   *  is already fixed, and moving it would misplace every scheduled chunk. */
  const setHeadroom = useCallback((s: number) => {
    headroomRef.current = s;
    setState(p => ({ ...p, headroom: s }));
  }, []);

  useEffect(() => () => { abortRef.current?.abort(); teardown(); }, [teardown]);

  return { ...state, start, stop, setVolume, setHeadroom };
}
