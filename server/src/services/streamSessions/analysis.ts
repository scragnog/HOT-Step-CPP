// streamSessions/analysis.ts — BPM and key of one stream chunk.
//
// The one copy of the analysis STORM has always run in the browser
// (ui/src/hooks/useStreamAudio.ts imports it from here). Node runs it on each
// chunk it streams; the client runs it on the decoded buffer for its crossfade
// timing. Both see channel 0 at the chunk's own rate, so they agree. Pure: no
// Node or DOM imports.

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

/** Key name ('A', 'F#m'): chroma of the first 16384 samples against Krumhansl profiles. */
export function detectKey(data: Float32Array, sampleRate: number): string {
  const ch = computeChroma(data, sampleRate);
  let best=-Infinity, bk='C', bm='major';
  for (let k=0;k<12;k++) {
    const rot=new Float32Array(12); for(let i=0;i<12;i++) rot[i]=ch[(i+k)%12];
    const mj=pearson(rot,MAJ), mn=pearson(rot,MIN);
    if(mj>best){best=mj;bk=KEYS[k];bm='major';}
    if(mn>best){best=mn;bk=KEYS[k];bm='minor';}
  }
  return bm==='major'?bk:bk+'m';
}

/** BPM (0 = unknown) and the first onset in seconds. */
export function detectBpm(raw: Float32Array, sr: number): { bpm: number; firstBeat: number } {
  const fs = 1024, hs = 512;

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
