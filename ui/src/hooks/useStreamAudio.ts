import { useRef, useState, useCallback, useEffect } from 'react';
// Moved out to utils/wavStream.ts when the MM3 player needed the same parser —
// same function, same behaviour, one copy.
import { extractWav } from '../utils/wavStream';

interface StreamState {
  isPlaying: boolean;
  currentSlot: number;   // WAVs received from server
  playingSlot: number;   // slot currently playing in Web Audio
  currentTime: number;
  bufferedTime: number;
  volume: number;
  maxBuffer: number;
  bufferPaused: boolean;
  estimatedSlotDuration: number;
  detectedBpm: number;              // measured BPM per slot (0=unknown)
  detectedKey: string;              // measured key per slot (''=unknown)
  isRecording: boolean;
  recordingTime: number;           // seconds elapsed since record started
  error: string | null;
}

const DEFAULT_MAX_BUFFER = 900;



// ── Goertzel + chroma key detection ─────────────────────────────────────────
function goertzel(data: Float32Array, freq: number, sr: number, len: number): number {
  const omega = 2 * Math.PI * freq / sr, c = 2 * Math.cos(omega);
  let s1 = 0, s2 = 0;
  for (let i = 0; i < len; i++) { const s = data[i] + c * s1 - s2; s2 = s1; s1 = s; }
  return s1 * s1 + s2 * s2 - c * s1 * s2;
}
function computeChroma(data: Float32Array, sr: number): Float32Array {
  const ch = new Float32Array(12), len = Math.min(data.length, 16384);
  const base = [130.81,138.59,146.83,155.56,164.81,174.61,185,196,207.65,220,233.08,246.94];
  for (let n = 0; n < 12; n++) {
    let e = 0;
    for (let o = 0; o < 4; o++) { const f = base[n] * Math.pow(2, o); if (f < 4200) e += goertzel(data, f, sr, len); }
    ch[n] = e;
  }
  const mx = Math.max(...Array.from(ch)); if (mx > 0) for (let i = 0; i < 12; i++) ch[i] /= mx;
  return ch;
}
function pearson(a: Float32Array, b: number[]): number {
  let sx=0,sy=0,sxy=0,sx2=0,sy2=0;
  for (let i=0;i<12;i++){sx+=a[i];sy+=b[i];sxy+=a[i]*b[i];sx2+=a[i]*a[i];sy2+=b[i]*b[i];}
  const d=Math.sqrt((12*sx2-sx*sx)*(12*sy2-sy*sy)); return d===0?0:(12*sxy-sx*sy)/d;
}
const KEYS=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const MAJ=[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88];
const MIN=[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17];
function detectKey(ab: AudioBuffer): string {
  const ch = computeChroma(ab.getChannelData(0), ab.sampleRate);
  let best=-Infinity, bk='C', bm='major';
  for (let k=0;k<12;k++) {
    const rot=new Float32Array(12); for(let i=0;i<12;i++) rot[i]=ch[(i+k)%12];
    const mj=pearson(rot,MAJ), mn=pearson(rot,MIN);
    if(mj>best){best=mj;bk=KEYS[k];bm='major';}
    if(mn>best){best=mn;bk=KEYS[k];bm='minor';}
  }
  return bm==='major'?bk:bk+'m';
}
function detectBpm(ab: AudioBuffer): { bpm: number; firstBeat: number } {
  const raw = ab.getChannelData(0), sr = ab.sampleRate, fs = 1024, hs = 512;

  // First-difference high-pass — attenuates kick body (~50-150Hz),
  // emphasizes snare crack (3-8kHz). Coefficient 0.97 ≈ cutoff ~750Hz.
  const data = new Float32Array(raw.length);
  for (let i = 1; i < raw.length; i++) data[i] = raw[i] - 0.97 * raw[i-1];

  // RMS envelope on HF-filtered signal — snares now dominate
  const en: number[] = [];
  for (let i = 0; i + fs < data.length; i += hs) {
    let e = 0; for (let j = 0; j < fs; j++) e += data[i+j] * data[i+j];
    en.push(Math.sqrt(e / fs));
  }

  // Onset detection — tighter min-spacing (snares never faster than 0.15s)
  const win = 16, onsets: number[] = [];
  for (let i = win; i < en.length - 1; i++) {
    let m = 0; for (let j = i - win; j < i; j++) m += en[j]; m /= win;
    if (en[i] > m * 1.5 && en[i] >= en[i-1] && en[i] > en[i+1]) {
      const t = i * hs / sr;
      if (onsets.length === 0 || t - onsets[onsets.length-1] > 0.15) onsets.push(t);
    }
  }

  if (onsets.length < 4) return { bpm: 0, firstBeat: 0 };

  // BPM — fundamental only, no half/double aliasing, 2-BPM bins
  const bpms: number[] = [];
  for (let i = 1; i < onsets.length; i++) {
    const iv = onsets[i] - onsets[i-1], b = 60 / iv;
    if (b >= 60 && b <= 200) bpms.push(b);
  }
  if (!bpms.length) return { bpm: 0, firstBeat: 0 };
  const h: Record<number, number> = {};
  for (const b of bpms) { const bin = Math.round(b / 2) * 2; h[bin] = (h[bin] || 0) + 1; }
  const bpm = parseInt(Object.entries(h).sort((a, b2) => b2[1] - a[1])[0][0]);

  return { bpm, firstBeat: onsets[0] ?? 0 };
}

export function useStreamAudio(streamId = 'default') {
  const [state, setState] = useState<StreamState>({
    isPlaying: false, currentSlot: 0, playingSlot: 0,
    currentTime: 0, bufferedTime: 0,
    volume: 1.0, maxBuffer: DEFAULT_MAX_BUFFER, bufferPaused: false, estimatedSlotDuration: 180,
    detectedBpm: 0, detectedKey: '',
    isRecording: false, recordingTime: 0, error: null,
  });

  const acRef          = useRef<AudioContext|null>(null);
  const gainRef        = useRef<GainNode|null>(null);
  const nextRef        = useRef(0);
  const startRef       = useRef(0);
  const abortRef       = useRef<AbortController|null>(null);
  const rafRef         = useRef(0);
  const aliveRef       = useRef(false);
  const xfadeBeatsRef  = useRef(4);
  const bpmRef         = useRef(120);
  const slotN          = useRef(0);
  const maxBufferRef   = useRef(DEFAULT_MAX_BUFFER);
  const serverPausedRef    = useRef(false);
  const crossfadeGainRef   = useRef(1.0);  // DJ crossfader multiplier (0..1)
  const volumeRef          = useRef(1.0);   // mirror of volume state for gain calculations
  const mediaRecorderRef   = useRef<MediaRecorder|null>(null);
  const recordedChunksRef  = useRef<Blob[]>([]);
  const recordingTimerRef  = useRef<ReturnType<typeof setInterval>|null>(null);
  const recordingDestRef   = useRef<MediaStreamAudioDestinationNode|null>(null);
  // Track when each slot starts playing in AudioContext time
  const slotTimesRef      = useRef<Array<{ slot: number; t: number }>>([]);
  const slotDurationsRef  = useRef<number[]>([]); // rolling last 5 slot durations

  const tick = useCallback(() => {
    const ac = acRef.current;
    if (!ac || !aliveRef.current) return;
    const currentTime  = Math.max(0, ac.currentTime - startRef.current);
    const bufferedTime = Math.max(0, nextRef.current - startRef.current);
    const ahead        = nextRef.current - ac.currentTime;

    // Compute which slot is currently audible
    let playing = 0;
    for (const { slot, t } of slotTimesRef.current) {
      if (t <= ac.currentTime) playing = slot;
      else break;
    }

    setState(p => ({ ...p, currentTime, bufferedTime, bufferPaused: serverPausedRef.current, playingSlot: playing }));

    // Resume gate — runs every RAF frame, breaks the deadlock
    if (serverPausedRef.current && ahead <= maxBufferRef.current * 0.8) {
      serverPausedRef.current = false;
      fetch('/api/generate/storm/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ streamId, stream_pause: false }),
      }).catch(() => {});
    }

    rafRef.current = requestAnimationFrame(tick);
  }, [streamId]);

  const start = useCallback(async (params: Record<string, unknown>) => {
    const ac = new AudioContext({ sampleRate: 48000 });
    acRef.current = ac;
    startRef.current = ac.currentTime + 0.3;
    nextRef.current  = startRef.current;
    slotN.current    = 0;
    slotTimesRef.current = [];
    slotDurationsRef.current = [];
    serverPausedRef.current = false;

    const g = ac.createGain();
    volumeRef.current = state.volume;
    crossfadeGainRef.current = 1.0;
    g.gain.value = state.volume;
    g.connect(ac.destination);
    gainRef.current = g;

    const abort = new AbortController();
    abortRef.current = abort;
    aliveRef.current = true;

    if (typeof params.bpm === 'number') bpmRef.current = params.bpm || 120;

    setState(p => ({
      ...p, isPlaying: true, currentSlot: 0, playingSlot: 0,
      currentTime: 0, bufferedTime: 0, bufferPaused: false, error: null,
    }));
    rafRef.current = requestAnimationFrame(tick);

    try {
      const res = await fetch('/api/generate/storm/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...params, streamId }),
        signal: abort.signal,
      });
      if (!res.ok || !res.body) throw new Error(`Stream failed: ${res.status}`);

      const reader = res.body.getReader();
      let buf: Uint8Array<ArrayBufferLike> = new Uint8Array(0);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const m = new Uint8Array(buf.length + value.length);
        m.set(buf); m.set(value, buf.length); buf = m;

        while (true) {
          const w = extractWav(buf);
          if (!w) break;
          buf = w.remaining;
          const copy = w.data.slice();
          let ab: AudioBuffer;
          try { ab = await ac.decodeAudioData(copy.buffer); } catch { continue; }

          // ── Detect BPM + key FIRST so crossfade uses this slot's tempo ──────
          const { bpm: measuredBpm, firstBeat } = detectBpm(ab);
          const measuredKey = detectKey(ab);
          if (measuredBpm > 0) bpmRef.current = measuredBpm;

          // ── Schedule crossfade with bar-boundary quantization ─────────────
          // Snapping the cut point to a bar boundary means the outgoing track
          // always ends on a musically correct beat — snares line up.
          const beatLen  = 60 / Math.max(60, bpmRef.current);
          const barLen   = 4 * beatLen;
          const minXfade = xfadeBeatsRef.current * beatLen;
          const dur      = ab.duration;
          const isFirst  = slotN.current === 0;

          const sg  = ac.createGain(); sg.connect(g);
          const src = ac.createBufferSource(); src.buffer = ab; src.connect(sg);
          const t0  = nextRef.current;
          src.start(t0);

          // Find last bar boundary that still leaves >= minXfade of tail
          let bodyEndOffset: number;
          if (minXfade > 0 && dur > minXfade * 2 && firstBeat > 0 && barLen > 0) {
            const latestCut = dur - minXfade;
            const barsAvail = Math.floor((latestCut - firstBeat) / barLen);
            const snapped   = firstBeat + Math.max(0, barsAvail) * barLen;
            bodyEndOffset   = (snapped > minXfade && snapped <= latestCut) ? snapped : latestCut;
          } else {
            bodyEndOffset = minXfade > 0 && dur > minXfade * 2 ? dur - minXfade : dur;
          }

          const bodyEnd  = t0 + bodyEndOffset;
          const xfadeDur = t0 + dur - bodyEnd;

          if (xfadeDur > 0.1) {
            sg.gain.setValueAtTime(0, t0);
            sg.gain.linearRampToValueAtTime(1, t0 + (isFirst ? 0.3 : Math.min(xfadeDur, minXfade || xfadeDur)));
            sg.gain.setValueAtTime(1, bodyEnd);
            sg.gain.linearRampToValueAtTime(0, t0 + dur);
            nextRef.current = bodyEnd;
          } else {
            sg.gain.setValueAtTime(1, t0);
            nextRef.current = t0 + dur;
          }

          slotN.current++;
          slotTimesRef.current.push({ slot: slotN.current, t: t0 });
          const durs = slotDurationsRef.current;
          durs.push(dur); if (durs.length > 5) durs.shift();
          const avgDur = Math.round(durs.reduce((a, b) => a + b, 0) / durs.length);
          setState(p => ({
            ...p, currentSlot: slotN.current, estimatedSlotDuration: avgDur,
            detectedBpm: measuredBpm > 0 ? measuredBpm : p.detectedBpm,
            detectedKey: measuredKey || p.detectedKey,
          }));

          // Pause signal — resume handled by tick()
          if (ac) {
            const ahead = nextRef.current - ac.currentTime;
            if (!serverPausedRef.current && ahead > maxBufferRef.current) {
              serverPausedRef.current = true;
              sendControl('stream_pause', true);
            }
          }
        }
      }
    } catch (e: unknown) {
      const err = e as Error;
      if (err.name !== 'AbortError') setState(p => ({ ...p, error: err.message }));
    } finally {
      aliveRef.current = false;
      cancelAnimationFrame(rafRef.current);
      // Auto-stop recording when stream ends
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (recordingTimerRef.current) { clearInterval(recordingTimerRef.current); recordingTimerRef.current = null; }
      setState(p => ({ ...p, isPlaying: false, bufferPaused: false, isRecording: false, recordingTime: 0 }));
      try { ac.close(); } catch {}
      acRef.current = null; gainRef.current = null; recordingDestRef.current = null;
    }
  }, [state.volume, tick]);

  const stop = useCallback(async () => {
    aliveRef.current = false;
    cancelAnimationFrame(rafRef.current);
    try { await fetch('/api/generate/storm/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ streamId }),
    }); } catch {}
    abortRef.current?.abort();
    try { acRef.current?.close(); } catch {}
    acRef.current = null; gainRef.current = null;
    slotTimesRef.current = [];
    slotDurationsRef.current = [];
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (recordingTimerRef.current) { clearInterval(recordingTimerRef.current); recordingTimerRef.current = null; }
    setState(p => ({ ...p, isPlaying: false, currentSlot: 0, playingSlot: 0, currentTime: 0, bufferedTime: 0, bufferPaused: false, isRecording: false, recordingTime: 0, detectedBpm: 0, detectedKey: '' }));
  }, [streamId]);

  const setVolume     = useCallback((v: number) => { volumeRef.current = v; setState(p => ({ ...p, volume: v })); if (gainRef.current) gainRef.current.gain.value = v * crossfadeGainRef.current; }, []);
  const setXfadeBeats = useCallback((v: number) => { xfadeBeatsRef.current = v; }, []);
  const setBpm        = useCallback((v: number) => { bpmRef.current = v; }, []);
  const setMaxBuffer  = useCallback((s: number) => { maxBufferRef.current = s; setState(p => ({ ...p, maxBuffer: s })); }, []);

  const shiftNextSlot = useCallback((ms: number) => {
    if (nextRef.current > 0) nextRef.current += ms / 1000;
  }, []);

  const getNextBeatTime = useCallback((bpm: number): number => {
    const ac = acRef.current;
    if (!ac) return 0;
    const beatLen = 60 / Math.max(60, bpm);
    const elapsed = ac.currentTime - startRef.current;
    const beatsElapsed = elapsed / beatLen;
    const nextBeat = Math.ceil(beatsElapsed + 0.1) * beatLen;
    return startRef.current + nextBeat;
  }, []);

  const setCrossfadeGain = useCallback((v: number) => {
    crossfadeGainRef.current = v;
    if (gainRef.current) gainRef.current.gain.value = volumeRef.current * v;
  }, []);

  const startRecording = useCallback(() => {
    const ac  = acRef.current;
    const g   = gainRef.current;
    if (!ac || !g || !aliveRef.current) { console.warn('[Record] no active stream'); return; }
    if (mediaRecorderRef.current) return; // already recording

    // Tap the master gain → MediaStreamDestination
    const dest = ac.createMediaStreamDestination();
    g.connect(dest);
    recordingDestRef.current = dest;

    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : 'audio/webm';

    const chunks: Blob[] = [];
    recordedChunksRef.current = chunks;

    const mr = new MediaRecorder(dest.stream, { mimeType });
    mr.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    mr.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, { type: mimeType });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      const ts   = new Date().toISOString().slice(0,19).replace(/[T:]/g, '-');
      a.download = `storm-set-${ts}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      recordedChunksRef.current = [];
      mediaRecorderRef.current  = null;
    };

    mr.start(1000); // chunk every 1s so onstop has data
    mediaRecorderRef.current = mr;
    setState(p => ({ ...p, isRecording: true, recordingTime: 0 }));

    let elapsed = 0;
    recordingTimerRef.current = setInterval(() => {
      elapsed += 1;
      setState(p => ({ ...p, recordingTime: elapsed }));
    }, 1000);
    console.log('[Record] started');
  }, []);

  const stopRecording = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (!mr || mr.state === 'inactive') return;
    mr.stop(); // triggers onstop → download
    if (recordingDestRef.current && gainRef.current) {
      try { gainRef.current.disconnect(recordingDestRef.current); } catch {}
      recordingDestRef.current = null;
    }
    if (recordingTimerRef.current) { clearInterval(recordingTimerRef.current); recordingTimerRef.current = null; }
    setState(p => ({ ...p, isRecording: false, recordingTime: 0 }));
    console.log('[Record] stopped — downloading');
  }, []);

  const sendControl = useCallback(async (key: string, value: unknown) => {
    try {
      await fetch('/api/generate/storm/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ streamId, [key]: value }),
      });
    } catch {}
  }, [streamId]);

  useEffect(() => () => {
    aliveRef.current = false;
    cancelAnimationFrame(rafRef.current);
    abortRef.current?.abort();
    try { acRef.current?.close(); } catch {}
  }, []);

  return { ...state, start, stop, setVolume, setXfadeBeats, setBpm, setMaxBuffer, sendControl, startRecording, stopRecording, setCrossfadeGain, shiftNextSlot, getNextBeatTime };
}