// audioQualityEvaluator.ts — Audio quality scoring for post-generation analysis
//
// Ported from jeankassio/JK-AceStep-Nodes AudioQualityEvaluator (MIT License).
// Pure TypeScript — no librosa/numpy/external dependencies.
//
// Three metrics (matching original weights):
//   1. Metallic Sound  (40%) — spectral rolloff at 85th percentile
//   2. Word Cuts       (40%) — spectral flux discontinuities (z-score)
//   3. Noise / Hiss    (20%) — zero-crossing rate
//
// Usage:
//   const result = evaluateAudioQuality('/path/to/file.wav');
//   console.log(result.score);  // 0.0–1.0

import fs from 'fs';

// ── Types ───────────────────────────────────────────────────────────────────

export interface QualityResult {
  score: number;        // 0.0–1.0 overall weighted score
  metallic: number;     // 0.0–1.0 sub-score
  wordCuts: number;     // 0.0–1.0 sub-score
  noise: number;        // 0.0–1.0 sub-score
  raw: {
    rolloffHz: number;
    severeCuts: number;
    moderateCuts: number;
    severePct: number;
    moderatePct: number;
    zcr: number;
  };
}

// ── FFT (Radix-2 Cooley-Tukey) ──────────────────────────────────────────────

/** In-place radix-2 FFT. Arrays must be power-of-2 length. */
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }
  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = -2 * Math.PI / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < halfLen; j++) {
        const a = i + j;
        const b = a + halfLen;
        const tRe = curRe * re[b] - curIm * im[b];
        const tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

// ── STFT ────────────────────────────────────────────────────────────────────

/** Compute magnitude spectrogram via Short-Time Fourier Transform. */
function stft(
  samples: Float32Array, nFft: number, hopLength: number
): Float64Array[] {
  const numFrames = Math.max(0, Math.floor((samples.length - nFft) / hopLength) + 1);
  const frames: Float64Array[] = [];

  // Pre-compute Hann window
  const window = new Float64Array(nFft);
  for (let i = 0; i < nFft; i++) {
    window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nFft - 1)));
  }

  const freqBins = (nFft >> 1) + 1;
  for (let f = 0; f < numFrames; f++) {
    const offset = f * hopLength;
    const re = new Float64Array(nFft);
    const im = new Float64Array(nFft);

    // Apply window
    for (let i = 0; i < nFft; i++) {
      re[i] = (offset + i < samples.length ? samples[offset + i] : 0) * window[i];
    }

    fft(re, im);

    // Magnitude (only positive frequencies)
    const mag = new Float64Array(freqBins);
    for (let i = 0; i < freqBins; i++) {
      mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    }
    frames.push(mag);
  }

  return frames;
}

// ── WAV Parsing ─────────────────────────────────────────────────────────────

interface WavInfo {
  samples: Float32Array;  // mono, normalised to [-1, 1]
  sampleRate: number;
}

function parseWav(buf: Buffer): WavInfo {
  // Find 'fmt ' chunk
  let fmtOffset = -1;
  for (let i = 12; i < buf.length - 8; i++) {
    if (buf[i] === 0x66 && buf[i+1] === 0x6D && buf[i+2] === 0x74 && buf[i+3] === 0x20) {
      fmtOffset = i + 8;
      break;
    }
  }
  if (fmtOffset < 0) throw new Error('No fmt chunk in WAV');

  const audioFormat = buf.readUInt16LE(fmtOffset);
  const numChannels = buf.readUInt16LE(fmtOffset + 2);
  const sampleRate = buf.readUInt32LE(fmtOffset + 4);
  const bitsPerSample = buf.readUInt16LE(fmtOffset + 14);

  // Find 'data' chunk
  let dataOffset = -1;
  let dataSize = 0;
  for (let i = 12; i < buf.length - 8; i++) {
    if (buf[i] === 0x64 && buf[i+1] === 0x61 && buf[i+2] === 0x74 && buf[i+3] === 0x61) {
      dataSize = buf.readUInt32LE(i + 4);
      dataOffset = i + 8;
      break;
    }
  }
  if (dataOffset < 0) throw new Error('No data chunk in WAV');

  const bytesPerSample = bitsPerSample >> 3;
  const totalSamples = Math.min(
    Math.floor(dataSize / (bytesPerSample * numChannels)),
    Math.floor((buf.length - dataOffset) / (bytesPerSample * numChannels))
  );

  // Read and downmix to mono Float32
  const mono = new Float32Array(totalSamples);

  for (let i = 0; i < totalSamples; i++) {
    let sum = 0;
    for (let ch = 0; ch < numChannels; ch++) {
      const pos = dataOffset + (i * numChannels + ch) * bytesPerSample;
      let sample: number;
      if (audioFormat === 3 || bitsPerSample === 32) {
        // 32-bit float
        sample = buf.readFloatLE(pos);
      } else if (bitsPerSample === 24) {
        const s = (buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16));
        sample = ((s & 0x800000) ? s - 0x1000000 : s) / 8388608;
      } else {
        // 16-bit
        sample = buf.readInt16LE(pos) / 32768;
      }
      sum += sample;
    }
    mono[i] = sum / numChannels;
  }

  return { samples: mono, sampleRate };
}

// ── Metric 1: Metallic Sound (Spectral Rolloff) ────────────────────────────

function scoreMetallic(frames: Float64Array[], sampleRate: number, nFft: number): { score: number; rolloffHz: number } {
  if (frames.length === 0) return { score: 0.5, rolloffHz: 0 };

  const freqBins = frames[0].length;
  const rollPercent = 0.85;
  let rolloffSum = 0;

  for (const mag of frames) {
    // Total energy in this frame
    let totalEnergy = 0;
    for (let i = 0; i < freqBins; i++) totalEnergy += mag[i] * mag[i];

    // Find bin where cumulative energy reaches 85%
    const threshold = totalEnergy * rollPercent;
    let cumulative = 0;
    let rolloffBin = freqBins - 1;
    for (let i = 0; i < freqBins; i++) {
      cumulative += mag[i] * mag[i];
      if (cumulative >= threshold) {
        rolloffBin = i;
        break;
      }
    }

    rolloffSum += (rolloffBin * sampleRate) / nFft;
  }

  const meanRolloff = rolloffSum / frames.length;

  // Scoring thresholds (matching original)
  let score: number;
  if (meanRolloff <= 7000) score = 1.0;
  else if (meanRolloff <= 8000) score = 0.8;
  else if (meanRolloff <= 9000) score = 0.5;
  else if (meanRolloff <= 10000) score = 0.2;
  else if (meanRolloff <= 11000) score = 0.05;
  else score = 0.01;

  return { score, rolloffHz: meanRolloff };
}

// ── Metric 2: Word Cuts (Spectral Flux) ─────────────────────────────────────

interface WordCutsResult {
  score: number;
  severeCuts: number;
  moderateCuts: number;
  severePct: number;
  moderatePct: number;
}

function scoreWordCuts(frames: Float64Array[]): WordCutsResult {
  if (frames.length < 2) {
    return { score: 0.5, severeCuts: 0, moderateCuts: 0, severePct: 0, moderatePct: 0 };
  }

  const freqBins = frames[0].length;
  const numFlux = frames.length - 1;
  const flux = new Float64Array(numFlux);

  // Compute spectral flux (L2 norm of frame-to-frame difference)
  for (let f = 0; f < numFlux; f++) {
    let sum = 0;
    for (let b = 0; b < freqBins; b++) {
      const diff = frames[f + 1][b] - frames[f][b];
      sum += diff * diff;
    }
    flux[f] = Math.sqrt(sum);
  }

  // Compute mean and std
  let meanFlux = 0;
  for (let i = 0; i < numFlux; i++) meanFlux += flux[i];
  meanFlux /= numFlux;

  let variance = 0;
  for (let i = 0; i < numFlux; i++) {
    const d = flux[i] - meanFlux;
    variance += d * d;
  }
  const stdFlux = Math.sqrt(variance / numFlux);

  if (stdFlux < 1e-6) {
    return { score: 0.0, severeCuts: -1, moderateCuts: 0, severePct: 0, moderatePct: 0 };
  }

  // Count severe (z > 4.0) and moderate (3.0 < z ≤ 4.0) discontinuities
  let severe = 0;
  let moderate = 0;
  for (let i = 0; i < numFlux; i++) {
    const z = (flux[i] - meanFlux) / stdFlux;
    if (z > 4.0) severe++;
    else if (z > 3.0) moderate++;
  }

  const severePct = (severe / numFlux) * 100;
  const moderatePct = (moderate / numFlux) * 100;

  // Scoring (matching original thresholds)
  let score: number;
  if (severePct < 0.05) {
    score = moderatePct < 0.8 ? 1.0 : moderatePct < 1.2 ? 0.95 : 0.90;
  } else if (severePct < 0.10) {
    score = moderatePct < 0.8 ? 0.85 : moderatePct < 1.2 ? 0.80 : 0.75;
  } else if (severePct < 0.15) {
    score = 0.65;
  } else if (severePct < 0.25) {
    score = 0.45;
  } else if (severePct < 0.40) {
    score = 0.25;
  } else {
    score = 0.10;
  }

  return { score, severeCuts: severe, moderateCuts: moderate, severePct, moderatePct };
}

// ── Metric 3: Noise / Hiss (Zero-Crossing Rate) ────────────────────────────

function scoreNoise(samples: Float32Array): { score: number; zcr: number } {
  if (samples.length < 2) return { score: 0.5, zcr: 0 };

  let crossings = 0;
  for (let i = 1; i < samples.length; i++) {
    if ((samples[i] >= 0) !== (samples[i - 1] >= 0)) crossings++;
  }
  const zcr = crossings / (samples.length - 1);

  // Scoring (matching original thresholds)
  let score: number;
  if (zcr >= 0.05 && zcr <= 0.12) score = 1.0;
  else if ((zcr >= 0.03 && zcr < 0.05) || (zcr > 0.12 && zcr <= 0.18)) score = 0.7;
  else if (zcr < 0.02) score = 0.4;
  else score = 0.3;

  return { score, zcr };
}

// ── Main evaluator ──────────────────────────────────────────────────────────

/**
 * Evaluate audio quality of a WAV file.
 * Returns a QualityResult with overall score (0–1) and per-metric breakdown.
 */
export function evaluateAudioQuality(wavPath: string): QualityResult {
  const buf = fs.readFileSync(wavPath);
  const { samples, sampleRate } = parseWav(buf);

  if (samples.length === 0) {
    return {
      score: 0, metallic: 0, wordCuts: 0, noise: 0,
      raw: { rolloffHz: 0, severeCuts: 0, moderateCuts: 0, severePct: 0, moderatePct: 0, zcr: 0 },
    };
  }

  // STFT parameters (matching original: n_fft=2048, hop=512)
  const nFft = 2048;
  const hopLength = 512;
  const frames = stft(samples, nFft, hopLength);

  // Metric 1: Metallic (40%)
  const met = scoreMetallic(frames, sampleRate, nFft);

  // Metric 2: Word Cuts (40%)
  const wc = scoreWordCuts(frames);

  // Metric 3: Noise (20%)
  const ns = scoreNoise(samples);

  // Weighted total
  const score = met.score * 0.40 + wc.score * 0.40 + ns.score * 0.20;

  return {
    score,
    metallic: met.score,
    wordCuts: wc.score,
    noise: ns.score,
    raw: {
      rolloffHz: met.rolloffHz,
      severeCuts: wc.severeCuts,
      moderateCuts: wc.moderateCuts,
      severePct: wc.severePct,
      moderatePct: wc.moderatePct,
      zcr: ns.zcr,
    },
  };
}

/**
 * Format a QualityResult as a human-readable log string.
 */
export function formatQualityLog(result: QualityResult, label: string): string {
  const r = result.raw;
  return `[Quality] ${label}: ${result.score.toFixed(3)} | ` +
    `Metallic=${result.metallic.toFixed(2)} WordCuts=${result.wordCuts.toFixed(2)} Noise=${result.noise.toFixed(2)} | ` +
    `Raw[Roll:${r.rolloffHz.toFixed(0)}Hz Cuts:${r.severeCuts}/${r.moderateCuts} ` +
    `(${r.severePct.toFixed(2)}%/${r.moderatePct.toFixed(2)}%) ZCR:${r.zcr.toFixed(3)}]`;
}
