/**
 * pitchShiftWorklet.js — pitch shift without touching tempo.
 *
 * Loaded by pitchShift.ts as an AudioWorklet module. Plain ES2017, no imports:
 * Vite copies this file to the output as an asset rather than bundling it, so
 * it has to stand on its own.
 *
 * WHAT IT DOES
 * ------------
 * A media element can change speed and let pitch follow, or change speed at a
 * fixed pitch. Neither gives you the third thing: pitch moved, speed left
 * alone. That needs a real shifter, so this is one.
 *
 * Two stages, ratio r (0.91875 for a 48 kHz render heard at 44.1 kHz):
 *
 *   1. WSOLA time-compression by r — same pitch, r x the length. Overlap-add
 *      of Hann-windowed input frames at a fixed synthesis hop, where each
 *      frame's read position is nudged within +/- SEEK to the offset that best
 *      correlates with the natural continuation of the previous frame. That
 *      alignment is what keeps waveforms in phase across the splice; a plain
 *      OLA at these hops would comb-filter badly.
 *   2. A 16-tap windowed-sinc resampler reading that stream at rate r — undoes
 *      the length change and drops the pitch by r along the way.
 *
 * r frames of stage-1 output per output frame, and 1 input frame per r of
 * stage-1 output: exactly 1:1 in to out, so the node never drifts against the
 * clock.
 *
 * The nominal analysis position advances along a float timeline, and each
 * frame's search offset is measured from that nominal rather than from the
 * previous chosen position — so the +/- SEEK nudges cannot accumulate.
 *
 * Correlation runs on the mid signal and the winning offset is applied to both
 * channels. Per-channel offsets would shear the stereo image.
 *
 * At ratio 1 the whole thing is bypassed sample-for-sample, so ordinary
 * playback keeps its exact samples and zero added latency. Switching in or out
 * of bypass changes the latency by ~50 ms, which would click, so mode changes
 * happen behind a short gain fade.
 */

// ── Windowed-sinc interpolation table ───────────────────────────────────────
const TAPS = 16;
const HALF = TAPS >> 1;
const PHASES = 512;
const SINC = new Float32Array(PHASES * TAPS);

(function buildSincTable() {
  for (let p = 0; p < PHASES; p++) {
    const frac = p / PHASES;
    const off = p * TAPS;
    let sum = 0;
    for (let j = 0; j < TAPS; j++) {
      const tau = (j - HALF + 1) - frac;          // [-HALF, HALF]
      const s = tau === 0 ? 1 : Math.sin(Math.PI * tau) / (Math.PI * tau);
      // Blackman window over the tap span; zero at both ends.
      const t = (tau / HALF + 1) / 2;
      const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * t) + 0.08 * Math.cos(4 * Math.PI * t);
      const v = s * w;
      SINC[off + j] = v;
      sum += v;
    }
    // Normalise each phase to unity DC gain, or the interpolator ripples.
    for (let j = 0; j < TAPS; j++) SINC[off + j] /= sum;
  }
})();

// Ring capacities. Powers of two so indexing is a mask, and far larger than
// the working set (a frame plus the search span) so nothing can lap itself.
const IN_CAP = 1 << 15;
const IN_MASK = IN_CAP - 1;
const S_CAP = 1 << 15;
const S_MASK = S_CAP - 1;

const MAX_CH = 2;
const FADE_FRAMES = 512;   // ~11 ms at 48 kHz — covers a mode switch

class PitchShiftProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    const sr = sampleRate;
    // ~43 ms frame, 50 % overlap. Long enough to hold the bass together,
    // short enough not to smear transients.
    this.win = Math.round((sr * 0.0427) / 2) * 2;
    this.hop = this.win >> 1;
    this.seek = Math.round(sr * 0.005);   // +/- 5 ms alignment search
    this.corr = Math.round(sr * 0.010);   // 10 ms correlation window

    this.window = new Float32Array(this.win);
    for (let n = 0; n < this.win; n++) {
      this.window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / this.win);
    }

    this.inBuf = [];
    this.sBuf = [];
    for (let c = 0; c < MAX_CH; c++) {
      this.inBuf.push(new Float32Array(IN_CAP));
      this.sBuf.push(new Float32Array(S_CAP));
    }
    this.midBuf = new Float32Array(IN_CAP);
    this.tmpl = new Float32Array(this.corr);

    this.ratio = 1;         // requested shift
    this.active = false;    // is the DSP path currently the audible one
    this.want = false;
    this.gain = 1;
    this.fade = 0;          // -1 fading out, +1 fading in, 0 steady

    this.resetDsp();

    this.port.onmessage = (e) => {
      const d = e.data;
      if (!d || d.type !== 'ratio') return;
      const r = Number(d.value);
      if (!isFinite(r) || r <= 0.25 || r >= 4) return;
      this.ratio = r;
      this.want = Math.abs(r - 1) > 1e-6;
    };
  }

  resetDsp() {
    this.inWrite = 0;
    this.sWrite = 0;
    this.sPos = 0;
    this.rPos = 0;
    this.aNom = this.seek;   // start clear of the ring's left edge
    this.firstFrame = true;
    this.tmplValid = false;
    for (let c = 0; c < MAX_CH; c++) {
      this.inBuf[c].fill(0);
      this.sBuf[c].fill(0);
    }
    this.midBuf.fill(0);
  }

  /** Best analysis offset for the next frame: the one whose signal best
   *  matches where the previous frame was heading. Normalised correlation, so
   *  a loud offset cannot win on level alone. */
  findDelta() {
    const { seek, corr, midBuf, tmpl } = this;
    const base = Math.round(this.aNom);
    let best = 0;
    let bestScore = -Infinity;
    for (let d = -seek; d <= seek; d++) {
      const p = base + d;
      let dot = 0;
      let energy = 0;
      for (let n = 0; n < corr; n++) {
        const v = midBuf[(p + n) & IN_MASK];
        dot += v * tmpl[n];
        energy += v * v;
      }
      const score = dot / Math.sqrt(energy + 1e-9);
      if (score > bestScore) { bestScore = score; best = d; }
    }
    return best;
  }

  /** Overlap-add one frame into the stage-1 stream. Returns false when the
   *  input ring does not hold enough yet. */
  produceFrame(nch) {
    const { win, hop, seek, corr } = this;
    const base = Math.round(this.aNom);
    if (this.inWrite < base + seek + win) return false;

    const a = base + (this.tmplValid ? this.findDelta() : 0);

    // The first half of the frame already carries the previous frame's tail;
    // only the fresh half needs clearing before we accumulate into it.
    const zFrom = this.firstFrame ? this.sPos : this.sPos + hop;
    const zTo = this.sPos + win;
    for (let i = zFrom; i < zTo; i++) {
      const si = i & S_MASK;
      for (let c = 0; c < nch; c++) this.sBuf[c][si] = 0;
    }

    for (let n = 0; n < win; n++) {
      const w = this.window[n];
      const si = (this.sPos + n) & S_MASK;
      const ii = (a + n) & IN_MASK;
      for (let c = 0; c < nch; c++) this.sBuf[c][si] += w * this.inBuf[c][ii];
    }

    this.sWrite = this.sPos + hop;
    this.sPos += hop;

    // Where this frame would have carried on to, had we not hopped — the
    // template the next frame gets matched against.
    for (let n = 0; n < corr; n++) {
      this.tmpl[n] = this.midBuf[(a + hop + n) & IN_MASK];
    }
    this.tmplValid = true;
    this.firstFrame = false;
    this.aNom += hop / this.ratio;
    return true;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const frames = output[0].length;
    const nOut = output.length;
    const nIn = input ? input.length : 0;

    // Nothing upstream — emit silence and start clean when it returns.
    // A short channel means the same thing: the render quantum carries no
    // audio, and copying it would leave whatever was in the buffer before.
    if (nIn === 0 || input[0].length !== frames) {
      for (let c = 0; c < nOut; c++) output[c].fill(0);
      if (this.active) this.resetDsp();
      return true;
    }

    const nch = Math.min(MAX_CH, nIn, nOut);

    // Mode changes ride a fade so the ~50 ms latency step cannot click.
    if (this.want !== this.active && this.fade === 0) this.fade = -1;

    if (this.active) {
      // Stage 0 — take the input in.
      for (let c = 0; c < nch; c++) {
        const src = input[c];
        const dst = this.inBuf[c];
        for (let i = 0; i < frames; i++) dst[(this.inWrite + i) & IN_MASK] = src[i];
      }
      const mid = this.midBuf;
      if (nch >= 2) {
        const l = input[0];
        const r = input[1];
        for (let i = 0; i < frames; i++) mid[(this.inWrite + i) & IN_MASK] = 0.5 * (l[i] + r[i]);
      } else {
        const l = input[0];
        for (let i = 0; i < frames; i++) mid[(this.inWrite + i) & IN_MASK] = l[i];
      }
      this.inWrite += frames;

      // Stage 1 — keep the compressed stream ahead of the read head.
      const needTo = this.rPos + frames * this.ratio + HALF + 2;
      while (this.sWrite < needTo) {
        if (!this.produceFrame(nch)) break;
      }

      // Stage 2 — resample it back to length, dropping the pitch on the way.
      for (let i = 0; i < frames; i++) {
        const n0 = Math.floor(this.rPos);
        if (n0 + HALF >= this.sWrite) {
          // Priming, or starved. Hold silence rather than repeat.
          for (let c = 0; c < nch; c++) output[c][i] = 0;
          continue;
        }
        const off = (((this.rPos - n0) * PHASES) | 0) * TAPS;
        const bi = n0 - HALF + 1;
        for (let c = 0; c < nch; c++) {
          const buf = this.sBuf[c];
          let acc = 0;
          for (let j = 0; j < TAPS; j++) acc += SINC[off + j] * buf[(bi + j) & S_MASK];
          output[c][i] = acc;
        }
        this.rPos += this.ratio;
      }
    } else {
      for (let c = 0; c < nch; c++) output[c].set(input[c]);
    }

    for (let c = nch; c < nOut; c++) output[c].fill(0);

    // ── Fade / mode switch ──
    if (this.fade !== 0) {
      const step = 1 / FADE_FRAMES;
      for (let i = 0; i < frames; i++) {
        this.gain = Math.max(0, Math.min(1, this.gain + this.fade * step));
        for (let c = 0; c < nch; c++) output[c][i] *= this.gain;
      }
      if (this.fade < 0 && this.gain <= 0) {
        this.active = this.want;
        this.resetDsp();
        this.fade = 1;
      } else if (this.fade > 0 && this.gain >= 1) {
        this.fade = 0;
      }
    }

    return true;
  }
}

registerProcessor('hot-pitch-shift', PitchShiftProcessor);
