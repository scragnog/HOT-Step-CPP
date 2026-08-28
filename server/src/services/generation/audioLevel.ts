// generation/audioLevel.ts — Shared gain staging, metering and limiting
//
// One place that owns audio level for the whole post-processing chain. Before
// this existed, every stage improvised: Gain Offset hard-clamped to int16,
// mastering.cpp clamped in its WAV writer, lufsNormalize hard-clipped at its
// "true-peak" ceiling, and mixWavBuffers scaled the whole mix on peak. Four
// implementations, none of them limiting, each one baking its damage into a
// 16-bit file for the next stage to read.
//
// The rule this module exists to enforce:
//
//   The chain must never destroy peaks. Exactly one stage limits, and it is
//   the last stage that can.
//
// A stage with audio-modifying work after it uses OverflowMode 'passthrough'
// and is allowed to hand on peaks above 0 dBFS in float. Only the final stage
// limits — and when that final stage is a user VST chain (Ozone) or
// mastering.exe, that tool owns limiting and the chain adds none of its own.
//
// The engine already supports this: audio_encode_wav_f32 (engine/src/audio-io.h)
// coerces NaN/Inf but does not clamp, and read_wav_buf (engine/src/wav.h) reads
// PCM16 / PCM24 / float32 including WAVE_FORMAT_EXTENSIBLE.
//
// See docs/plans/2026-08-28-post-processing-gain-staging.md.

import fs from 'fs';

// ── Types ───────────────────────────────────────────────────────────────────

export type WavSampleFormat = 's16' | 's24' | 'f32';

export interface FloatWav {
  /** Per-channel samples as floats. Not clamped — may exceed [-1, 1]. */
  channels: Float32Array[];
  sampleRate: number;
  numChannels: number;
  /** Format the buffer was decoded from, so a stage can round-trip in kind. */
  sourceFormat: WavSampleFormat;
}

export interface LevelMetrics {
  /** Highest absolute sample value, linear. */
  samplePeak: number;
  /** Highest absolute value of the 4x oversampled signal, linear. */
  truePeak: number;
  samplePeakDb: number;
  truePeakDb: number;
  /** Integrated loudness, ITU-R BS.1770-4. -Infinity for silence. */
  lufs: number;
  /** RMS across all channels, linear. */
  rms: number;
  rmsDb: number;
  /** Sample peak over RMS, in dB. Low values indicate loudness-war density. */
  crestDb: number;
  /** Samples pinned at the signal's own peak value. */
  clippedSamples: number;
  /** Longest run of consecutive samples pinned at the peak — the signature of
   *  a hard clip. A limiter leaves isolated samples touching its ceiling; a
   *  clipper leaves flat tops. Measured against the signal's OWN peak, not
   *  against full scale, so it stays meaningful in a float file that carries
   *  peaks above 0 dBFS mid-chain. */
  longestClipRun: number;
  durationSec: number;
}

/** How a stage handles a signal that exceeds full scale.
 *
 *  - 'passthrough': leave it. Correct whenever another audio-modifying stage
 *    runs afterwards, and the only correct choice when the user's own VST
 *    chain is doing the mastering.
 *  - 'limit': look-ahead limiter to the ceiling. For the last stage that can
 *    limit, when nothing downstream owns it.
 *  - 'clip': hard clip. Present so the old behaviour stays reachable and
 *    testable; nothing should choose it by default. */
export type OverflowMode = 'passthrough' | 'limit' | 'clip';

export interface LimitOptions {
  /** Peak ceiling in dBFS. Default -1. */
  ceilingDb?: number;
  /** Look-ahead in milliseconds. Default 5. */
  lookaheadMs?: number;
  /** Release time in milliseconds. Default 50. */
  releaseMs?: number;
}

export interface LimitResult {
  engaged: boolean;
  /** Largest gain reduction applied, in dB (positive number). */
  maxReductionDb: number;
  peakBefore: number;
  peakAfter: number;
}

// ── dB helpers ──────────────────────────────────────────────────────────────

export function dbToLin(db: number): number {
  return Math.pow(10, db / 20);
}

/** Linear amplitude to dBFS. Returns -Infinity for zero. */
export function linToDb(lin: number): number {
  return lin > 0 ? 20 * Math.log10(lin) : -Infinity;
}

// ── WAV decode ──────────────────────────────────────────────────────────────
//
// Proper chunk walking (not a byte scan for 'fmt '/'data', which can match
// inside a LIST or metadata chunk) and WAVE_FORMAT_EXTENSIBLE resolution.

const FORMAT_PCM = 1;
const FORMAT_FLOAT = 3;
const FORMAT_EXTENSIBLE = 0xfffe;

export function parseWav(buf: Buffer): FloatWav {
  if (buf.length < 44 ||
      buf.toString('ascii', 0, 4) !== 'RIFF' ||
      buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('[Level] Not a valid RIFF/WAVE file');
  }

  let audioFormat = 0, numChannels = 0, sampleRate = 0, bitsPerSample = 0;
  let dataOffset = -1, dataSize = 0;

  let offset = 12;
  while (offset + 8 <= buf.length) {
    const chunkId = buf.toString('ascii', offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    const body = offset + 8;

    if (chunkId === 'fmt ' && body + 16 <= buf.length) {
      audioFormat   = buf.readUInt16LE(body);
      numChannels   = buf.readUInt16LE(body + 2);
      sampleRate    = buf.readUInt32LE(body + 4);
      bitsPerSample = buf.readUInt16LE(body + 14);
      // WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of the
      // SubFormat GUID, 24 bytes into the fmt body.
      if (audioFormat === FORMAT_EXTENSIBLE && body + 26 <= buf.length) {
        audioFormat = buf.readUInt16LE(body + 24);
      }
    } else if (chunkId === 'data') {
      dataOffset = body;
      dataSize = Math.min(chunkSize, buf.length - body);
      break;
    }

    offset = body + chunkSize + (chunkSize % 2); // chunks are word-aligned
  }

  if (dataOffset < 0 || sampleRate <= 0 || numChannels <= 0) {
    throw new Error('[Level] WAV missing fmt or data chunk');
  }

  let sourceFormat: WavSampleFormat;
  if (audioFormat === FORMAT_FLOAT && bitsPerSample === 32) sourceFormat = 'f32';
  else if (audioFormat === FORMAT_PCM && bitsPerSample === 24) sourceFormat = 's24';
  else if (audioFormat === FORMAT_PCM && bitsPerSample === 16) sourceFormat = 's16';
  else throw new Error(`[Level] Unsupported WAV format (fmt=${audioFormat}, ${bitsPerSample}-bit)`);

  const bytesPerSample = bitsPerSample >> 3;
  const frameBytes = bytesPerSample * numChannels;
  const frames = Math.floor(dataSize / frameBytes);

  const channels: Float32Array[] = [];
  for (let ch = 0; ch < numChannels; ch++) channels.push(new Float32Array(frames));

  for (let i = 0; i < frames; i++) {
    const frameStart = dataOffset + i * frameBytes;
    for (let ch = 0; ch < numChannels; ch++) {
      const p = frameStart + ch * bytesPerSample;
      let v: number;
      if (sourceFormat === 'f32') {
        v = buf.readFloatLE(p);
      } else if (sourceFormat === 's24') {
        const raw = buf[p] | (buf[p + 1] << 8) | (buf[p + 2] << 16);
        v = ((raw & 0x800000) ? raw - 0x1000000 : raw) / 8388608;
      } else {
        v = buf.readInt16LE(p) / 32768;
      }
      channels[ch][i] = v;
    }
  }

  return { channels, sampleRate, numChannels, sourceFormat };
}

export function readWav(filePath: string): FloatWav {
  return parseWav(fs.readFileSync(filePath));
}

// ── WAV encode ──────────────────────────────────────────────────────────────

/** Encode to WAV. 'f32' does NOT clamp — above-full-scale peaks survive, which
 *  is the whole point of carrying the chain in float. 's16'/'s24' clamp,
 *  because they have to. */
export function encodeWav(wav: FloatWav, format: WavSampleFormat): Buffer {
  const { channels, sampleRate, numChannels } = wav;
  const frames = channels[0]?.length ?? 0;
  const bitsPerSample = format === 'f32' ? 32 : format === 's24' ? 24 : 16;
  const bytesPerSample = bitsPerSample >> 3;
  const dataSize = frames * numChannels * bytesPerSample;

  const out = Buffer.alloc(44 + dataSize);
  out.write('RIFF', 0, 'ascii');
  out.writeUInt32LE(36 + dataSize, 4);
  out.write('WAVE', 8, 'ascii');
  out.write('fmt ', 12, 'ascii');
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(format === 'f32' ? FORMAT_FLOAT : FORMAT_PCM, 20);
  out.writeUInt16LE(numChannels, 22);
  out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * numChannels * bytesPerSample, 28);
  out.writeUInt16LE(numChannels * bytesPerSample, 32);
  out.writeUInt16LE(bitsPerSample, 34);
  out.write('data', 36, 'ascii');
  out.writeUInt32LE(dataSize, 40);

  let p = 44;
  for (let i = 0; i < frames; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const v = channels[ch][i];
      if (format === 'f32') {
        out.writeFloatLE(Number.isFinite(v) ? v : 0, p);
      } else if (format === 's24') {
        const c = Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0));
        const s = Math.max(-8388608, Math.min(8388607, Math.round(c * 8388607)));
        const u = s < 0 ? s + 0x1000000 : s;
        out[p] = u & 0xff;
        out[p + 1] = (u >> 8) & 0xff;
        out[p + 2] = (u >> 16) & 0xff;
      } else {
        const c = Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0));
        out.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(c * 32767))), p);
      }
      p += bytesPerSample;
    }
  }

  return out;
}

export function writeWav(filePath: string, wav: FloatWav, format: WavSampleFormat): void {
  fs.writeFileSync(filePath, encodeWav(wav, format));
}

// ── True peak ───────────────────────────────────────────────────────────────
//
// 4x oversampled peak via a 4-phase Hann-windowed sinc interpolator. This is
// NOT the tabulated BS.1770-4 Annex 2 filter — it is a windowed sinc of
// similar order, which is adequate for the decision this chain actually makes
// (is there enough headroom, and did a stage create inter-sample overs). It is
// not calibrated for compliance metering, and is not claimed to be.
//
// The reported true peak is max(samplePeak, oversampledPeak), so it can never
// come back lower than the plain sample peak regardless of filter phase.

const TP_PHASES = 4;
const TP_TAPS = 16; // per phase

let tpKernel: Float32Array[] | null = null;

function truePeakKernel(): Float32Array[] {
  if (tpKernel) return tpKernel;

  const total = TP_TAPS * TP_PHASES;
  const h = new Float64Array(total);
  const center = (total - 1) / 2;
  for (let i = 0; i < total; i++) {
    const x = (i - center) / TP_PHASES;          // in original-sample units
    const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
    const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (total - 1));
    h[i] = sinc * hann;
  }

  // Split into polyphase branches and normalise each to unity DC gain, so a
  // constant signal interpolates to itself rather than picking up ripple.
  const phases: Float32Array[] = [];
  for (let p = 0; p < TP_PHASES; p++) {
    const branch = new Float32Array(TP_TAPS);
    let sum = 0;
    for (let k = 0; k < TP_TAPS; k++) {
      branch[k] = h[k * TP_PHASES + p];
      sum += branch[k];
    }
    if (Math.abs(sum) > 1e-12) {
      for (let k = 0; k < TP_TAPS; k++) branch[k] /= sum;
    }
    phases.push(branch);
  }

  tpKernel = phases;
  return phases;
}

/** Fraction of the sample peak below which a sample cannot host the true peak.
 *
 *  An interpolated value is bounded by (local max sample) x (sum of |taps|)
 *  for its branch. Only oversampling around samples within 6 dB of the sample
 *  peak turns an O(64 x samples) sweep into a few thousand windows on a real
 *  track — without it, metering 4 minutes of stereo costs tens of seconds and
 *  per-stage logging is unaffordable.
 *
 *  0.5 is deliberately looser than the branch tap sum requires. A tone at
 *  exactly SR/4 sampled on its zero crossings is the worst realistic case and
 *  hides 3 dB; the margin here is double that. `audioLevel.selftest.ts` checks
 *  the thresholded search against an exhaustive one rather than trusting the
 *  reasoning. */
const TP_CANDIDATE_RATIO = 0.5;

/** Highest absolute value of the 4x oversampled signal, linear.
 *  Never returns less than the plain sample peak. */
export function truePeak(channels: Float32Array[]): number {
  const phases = truePeakKernel();
  const half = TP_TAPS >> 1;

  let peak = 0;
  for (const ch of channels) {
    for (let i = 0; i < ch.length; i++) {
      const a = Math.abs(ch[i]);
      if (a > peak) peak = a;
    }
  }
  if (peak === 0) return 0;

  const threshold = peak * TP_CANDIDATE_RATIO;
  let tp = peak;

  for (const ch of channels) {
    const n = ch.length;
    for (let i = 0; i < n; i++) {
      if (Math.abs(ch[i]) < threshold) continue;
      for (let p = 0; p < TP_PHASES; p++) {
        const branch = phases[p];
        let acc = 0;
        for (let k = 0; k < TP_TAPS; k++) {
          const idx = i + k - half;
          if (idx >= 0 && idx < n) acc += branch[k] * ch[idx];
        }
        const a = Math.abs(acc);
        if (a > tp) tp = a;
      }
    }
  }

  return tp;
}

// ── K-weighting and integrated LUFS (ITU-R BS.1770-4) ───────────────────────
//
// Moved here from lufsNormalize.ts so the metering and the normalizer share
// one implementation. Coefficients: BS.1770-4 Table 1 for 48 kHz, pyLoudnorm
// for 44.1 kHz; other rates fall back to the 48 kHz set.

interface BiquadCoeffs { b0: number; b1: number; b2: number; a1: number; a2: number }

function getKWeightingCoeffs(sampleRate: number): [BiquadCoeffs, BiquadCoeffs] {
  if (sampleRate === 44100) {
    return [
      { b0: 1.5308412300498355, b1: -2.6509799951547297, b2: 1.1690790799215869,
        a1: -1.6636551132560204, a2: 0.7125954280732254 },
      { b0: 1.0, b1: -2.0, b2: 1.0, a1: -1.9891696736297957, a2: 0.9891990357870394 },
    ];
  }
  return [
    { b0: 1.53512485958697, b1: -2.69169618940638, b2: 1.19839281085285,
      a1: -1.69065929318241, a2: 0.73248077421585 },
    { b0: 1.0, b1: -2.0, b2: 1.0, a1: -1.99004745483398, a2: 0.99007225036621 },
  ];
}

function applyBiquad(samples: Float32Array, c: BiquadCoeffs): Float32Array {
  const out = new Float32Array(samples.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < samples.length; i++) {
    const x0 = samples[i];
    const y0 = c.b0 * x0 + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    out[i] = y0;
    x2 = x1; x1 = x0;
    y2 = y1; y1 = y0;
  }
  return out;
}

function applyKWeighting(samples: Float32Array, sampleRate: number): Float32Array {
  const [shelf, hp] = getKWeightingCoeffs(sampleRate);
  return applyBiquad(applyBiquad(samples, shelf), hp);
}

/** Integrated loudness in LUFS per ITU-R BS.1770-4. -Infinity for silence. */
export function integratedLufs(channels: Float32Array[], sampleRate: number): number {
  const numChannels = channels.length;
  if (numChannels === 0 || channels[0].length === 0) return -Infinity;

  const kWeighted = channels.map(ch => applyKWeighting(ch, sampleRate));

  const blockSamples = Math.round(sampleRate * 0.4);  // 400 ms
  const hopSamples = Math.round(sampleRate * 0.1);    // 100 ms hop (75% overlap)
  const totalSamples = kWeighted[0].length;
  const numBlocks = Math.max(0, Math.floor((totalSamples - blockSamples) / hopSamples) + 1);
  if (numBlocks === 0) return -Infinity;

  const blockLoudness = new Float64Array(numBlocks);
  for (let b = 0; b < numBlocks; b++) {
    const start = b * hopSamples;
    const end = Math.min(start + blockSamples, totalSamples);
    let blockPower = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const kw = kWeighted[ch];
      let chPower = 0;
      for (let i = start; i < end; i++) chPower += kw[i] * kw[i];
      blockPower += chPower / blockSamples;   // channel weight 1.0 for mono/stereo
    }
    blockLoudness[b] = blockPower > 0 ? -0.691 + 10 * Math.log10(blockPower) : -Infinity;
  }

  // Absolute gate at -70 LUFS, then relative gate at (ungated mean - 10 dB).
  const ungated: number[] = [];
  for (let b = 0; b < numBlocks; b++) if (blockLoudness[b] > -70) ungated.push(b);
  if (ungated.length === 0) return -Infinity;

  const meanPower = (blocks: number[]) => {
    let sum = 0;
    for (const b of blocks) sum += Math.pow(10, (blockLoudness[b] + 0.691) / 10);
    return -0.691 + 10 * Math.log10(sum / blocks.length);
  };

  const relativeGate = meanPower(ungated) - 10;
  const gated = ungated.filter(b => blockLoudness[b] > relativeGate);
  if (gated.length === 0) return -Infinity;

  return meanPower(gated);
}

// ── Metering ────────────────────────────────────────────────────────────────

/** Full level report. `truePeak` is the expensive part (4x oversampling over
 *  every sample); pass `{ skipTruePeak: true }` for a cheap probe. */
export function measureLevels(wav: FloatWav, opts: { skipTruePeak?: boolean } = {}): LevelMetrics {
  const { channels, sampleRate } = wav;
  const frames = channels[0]?.length ?? 0;

  let samplePeak = 0;
  let sumSq = 0;
  let count = 0;

  for (const ch of channels) {
    for (let i = 0; i < ch.length; i++) {
      const v = ch[i];
      const a = Math.abs(v);
      if (a > samplePeak) samplePeak = a;
      sumSq += v * v;
      count++;
    }
  }

  // Flat-top detection, against the signal's own peak rather than full scale.
  //
  // The earlier version counted everything at or above 0.9999. That reads
  // correctly on a finished 16-bit file and is nonsense mid-chain: a float
  // file legitimately carrying +3 dBFS peaks has tens of thousands of samples
  // above 0.9999 that are merely loud, and it reported them as clipping.
  //
  // Pinning at the peak is the thing that actually distinguishes a clipper
  // from a limiter. A limiter leaves isolated samples touching its ceiling; a
  // clipper leaves runs. This also still catches 16-bit clipping, where the
  // peak IS 32767/32768 and the flat tops sit exactly on it.
  let clippedSamples = 0;
  let longestClipRun = 0;
  if (samplePeak > 0) {
    const eps = samplePeak * 1e-4;
    for (const ch of channels) {
      let currentRun = 0;
      for (let i = 0; i < ch.length; i++) {
        if (Math.abs(Math.abs(ch[i]) - samplePeak) <= eps) {
          clippedSamples++;
          currentRun++;
          if (currentRun > longestClipRun) longestClipRun = currentRun;
        } else {
          currentRun = 0;
        }
      }
    }
  }

  const rms = count > 0 ? Math.sqrt(sumSq / count) : 0;
  const tp = opts.skipTruePeak ? samplePeak : truePeak(channels);

  return {
    samplePeak,
    truePeak: tp,
    samplePeakDb: linToDb(samplePeak),
    truePeakDb: linToDb(tp),
    lufs: integratedLufs(channels, sampleRate),
    rms,
    rmsDb: linToDb(rms),
    crestDb: rms > 0 ? linToDb(samplePeak) - linToDb(rms) : 0,
    clippedSamples,
    longestClipRun,
    durationSec: sampleRate > 0 ? frames / sampleRate : 0,
  };
}

export function measureFile(filePath: string, opts?: { skipTruePeak?: boolean }): LevelMetrics {
  return measureLevels(readWav(filePath), opts);
}

/** Run length at which pinned-at-peak samples are reported as flat-topping.
 *
 *  Not zero, and not three. Any signal has a sample or two sitting on its own
 *  peak, and a limiter holding a ceiling through a loud passage can leave a
 *  short run there legitimately. A clipper leaves runs an order of magnitude
 *  longer — a real render measured 59 against a limiter's 0. Eight sits in the
 *  gap with room on both sides. */
export const FLAT_TOP_RUN = 8;

const fmtDb = (db: number) => (Number.isFinite(db) ? db.toFixed(1) : '-inf');

/** One-line level summary for the generation log. */
export function formatLevels(m: LevelMetrics, label: string): string {
  const clip = m.longestClipRun >= FLAT_TOP_RUN
    ? ` | FLAT-TOPPED: ${m.clippedSamples} samples pinned at peak, longest run ${m.longestClipRun}`
    : '';
  return `${label}: peak ${fmtDb(m.samplePeakDb)} dBFS, true peak ${fmtDb(m.truePeakDb)} dBTP, `
    + `${fmtDb(m.lufs)} LUFS, crest ${m.crestDb.toFixed(1)} dB${clip}`;
}

// ── Limiter ─────────────────────────────────────────────────────────────────

/**
 * Look-ahead peak limiter, in place.
 *
 * Replaces the hard clip that lufsNormalize.ts called a true-peak limiter.
 * The gain envelope is derived from the maximum across channels at each
 * sample, so stereo image is preserved: both channels always take the same
 * gain.
 *
 * The envelope is a sliding minimum over the look-ahead window, smoothed with
 * separate attack and release coefficients, then hard-clamped against the
 * per-sample requirement. That final clamp is what makes the ceiling a
 * guarantee rather than an aspiration — it only engages if the smoothing
 * overshot, which is brief and rare, and it is still strictly better than
 * clipping the raw signal.
 *
 * Because the envelope already accounts for the next `lookaheadMs` of signal,
 * gain is down before the peak arrives and the output needs no delay.
 */
export function limitPeaks(wav: FloatWav, opts: LimitOptions = {}): LimitResult {
  const ceilingDb = opts.ceilingDb ?? -1;
  const lookaheadMs = opts.lookaheadMs ?? 5;
  const releaseMs = opts.releaseMs ?? 50;

  const ceiling = dbToLin(ceilingDb);
  const { channels, sampleRate } = wav;
  const frames = channels[0]?.length ?? 0;
  if (frames === 0) {
    return { engaged: false, maxReductionDb: 0, peakBefore: 0, peakAfter: 0 };
  }

  // Per-sample gain the signal demands, from the loudest channel.
  const demand = new Float32Array(frames);
  let peakBefore = 0;
  for (let i = 0; i < frames; i++) {
    let a = 0;
    for (const ch of channels) {
      const v = Math.abs(ch[i]);
      if (v > a) a = v;
    }
    if (a > peakBefore) peakBefore = a;
    demand[i] = a > ceiling ? ceiling / a : 1;
  }

  if (peakBefore <= ceiling) {
    return { engaged: false, maxReductionDb: 0, peakBefore, peakAfter: peakBefore };
  }

  // Sliding minimum over [i, i + lookahead], via a monotonic deque.
  const look = Math.max(1, Math.round((sampleRate * lookaheadMs) / 1000));
  const env = new Float32Array(frames);
  const deque = new Int32Array(frames);
  let head = 0, tail = 0;
  for (let i = frames - 1; i >= 0; i--) {
    while (tail > head && demand[deque[tail - 1]] >= demand[i]) tail--;
    deque[tail++] = i;
    while (deque[head] > i + look) head++;
    env[i] = demand[deque[head]];
  }

  // Attack fast enough to reach the target within the look-ahead window.
  const attackSamples = Math.max(1, look / 3);
  const releaseSamples = Math.max(1, (sampleRate * releaseMs) / 1000);
  const attackCoeff = Math.exp(-1 / attackSamples);
  const releaseCoeff = Math.exp(-1 / releaseSamples);

  // Start at the envelope's first value rather than unity. Ramping down from
  // 1.0 makes the safety clamp engage on the opening crests, which pins a run
  // of samples exactly at the ceiling — a flat top of the limiter's own
  // making, at the one moment it is least necessary.
  let g = env[0];
  let minGain = g;
  for (let i = 0; i < frames; i++) {
    const target = env[i];
    const coeff = target < g ? attackCoeff : releaseCoeff;
    g = target + (g - target) * coeff;
    // Ceiling guarantee: never let smoothing leave the sample above the wall.
    if (g > demand[i]) g = demand[i];
    if (g < minGain) minGain = g;
    for (const ch of channels) ch[i] *= g;
  }

  let peakAfter = 0;
  for (const ch of channels) {
    for (let i = 0; i < frames; i++) {
      const a = Math.abs(ch[i]);
      if (a > peakAfter) peakAfter = a;
    }
  }

  return {
    engaged: true,
    maxReductionDb: -linToDb(minGain),
    peakBefore,
    peakAfter,
  };
}

// ── Gain ────────────────────────────────────────────────────────────────────

export interface ApplyGainResult {
  gainDb: number;
  mode: OverflowMode;
  before: LevelMetrics;
  after: LevelMetrics;
  limit?: LimitResult;
}

/** Apply gain to every channel, then handle overflow per `mode`. In place. */
export function applyGain(wav: FloatWav, gainDb: number, mode: OverflowMode,
                          limitOpts?: LimitOptions): LimitResult | undefined {
  if (gainDb !== 0) {
    const lin = dbToLin(gainDb);
    for (const ch of wav.channels) {
      for (let i = 0; i < ch.length; i++) ch[i] *= lin;
    }
  }

  if (mode === 'limit') {
    return limitPeaks(wav, limitOpts);
  }
  if (mode === 'clip') {
    const ceiling = dbToLin(limitOpts?.ceilingDb ?? 0);
    let engaged = false;
    let peakBefore = 0, peakAfter = 0;
    for (const ch of wav.channels) {
      for (let i = 0; i < ch.length; i++) {
        const a = Math.abs(ch[i]);
        if (a > peakBefore) peakBefore = a;
        if (a > ceiling) {
          ch[i] = ch[i] > 0 ? ceiling : -ceiling;
          engaged = true;
        }
        const b = Math.abs(ch[i]);
        if (b > peakAfter) peakAfter = b;
      }
    }
    return { engaged, maxReductionDb: 0, peakBefore, peakAfter };
  }
  // 'passthrough' — deliberately nothing. A later stage owns the ceiling.
  return undefined;
}

/**
 * Read a WAV, apply gain, handle overflow, write it back.
 *
 * `outFormat` defaults to the source format, so this is safe to drop into an
 * existing 16-bit stage. Pass 'f32' once the chain is carrying float, which is
 * what makes 'passthrough' mean anything.
 */
export function applyGainToFile(
  filePath: string,
  gainDb: number,
  mode: OverflowMode,
  opts: { outFormat?: WavSampleFormat; limit?: LimitOptions; skipTruePeak?: boolean } = {},
): ApplyGainResult {
  const wav = readWav(filePath);
  const before = measureLevels(wav, { skipTruePeak: opts.skipTruePeak });
  const limit = applyGain(wav, gainDb, mode, opts.limit);
  const after = measureLevels(wav, { skipTruePeak: opts.skipTruePeak });
  writeWav(filePath, wav, opts.outFormat ?? wav.sourceFormat);
  return { gainDb, mode, before, after, limit };
}
