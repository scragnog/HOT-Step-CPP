// audioActivity.ts — where the singing is, and the windows Whisper should run on.
//
// Whisper's encoder is 30 s and the CLI is a single process per file, so a long
// track is either one slow call that can be killed by a timeout, or several
// calls whose boundaries land wherever a fixed clock puts them. A fixed
// boundary cuts a word in half and the two halves are transcribed twice.
//
// This measures a short-term RMS envelope and returns windows that start and
// end in a quiet gap, capped at a length the CLI can finish. It reads WAV
// directly (the app's renders are WAV) and answers one whole-file window for
// anything it cannot decode, so a missing decode is a slower run, never a
// failed one.

import fs from 'fs';
import path from 'path';
import { parseWavHeader } from './audioCrop.js';

export interface AudioWindow {
  /** Seconds from the start of the file. */
  startSec: number;
  endSec: number;
}

/** Envelope resolution. 50 ms is short enough to find a consonant gap. */
const HOP_SEC = 0.05;
/** RMS frame length. */
const FRAME_SEC = 0.1;

export interface DecodedWav {
  samples: Float32Array;
  rate: number;
}

/** The audio format tag from a WAV fmt chunk (1 = PCM, 3 = IEEE float). */
function wavFormatTag(buf: Buffer): number {
  for (let offset = 12; offset + 8 <= Math.min(buf.length, 256);) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'fmt ') return buf.readUInt16LE(offset + 8);
    offset += 8 + size + (size & 1);
  }
  return 1;
}

/** Mono mix of a WAV file's samples, or null when it is not a WAV we can read. */
export function decodeWavMono(filePath: string): DecodedWav | null {
  try {
    const buf = fs.readFileSync(filePath);
    const info = parseWavHeader(buf);
    const bytesPerSample = info.bitsPerSample / 8;
    const frameSize = bytesPerSample * info.numChannels;
    if (!frameSize || !info.sampleRate) return null;
    const total = Math.floor(info.dataSize / frameSize);
    if (total <= 0) return null;
    const float = wavFormatTag(buf) === 3;
    const samples = new Float32Array(total);
    for (let i = 0; i < total; i++) {
      const base = info.dataOffset + i * frameSize;
      let sum = 0;
      for (let ch = 0; ch < info.numChannels; ch++) {
        const off = base + ch * bytesPerSample;
        if (off + bytesPerSample > buf.length) break;
        if (info.bitsPerSample === 16) {
          sum += buf.readInt16LE(off) / 32768;
        } else if (info.bitsPerSample === 24) {
          const raw = buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16);
          sum += ((raw & 0x800000 ? raw - 0x1000000 : raw) / 8388608);
        } else if (info.bitsPerSample === 32) {
          sum += float ? buf.readFloatLE(off) : buf.readInt32LE(off) / 2147483648;
        }
      }
      samples[i] = sum / info.numChannels;
    }
    return { samples, rate: info.sampleRate };
  } catch {
    return null;
  }
}

/** Short-term RMS per hop, normalised 0..1. Empty when the audio is too short. */
export function energyEnvelope(samples: Float32Array, rate: number): number[] {
  const frame = Math.max(1, Math.round(FRAME_SEC * rate));
  const hop = Math.max(1, Math.round(HOP_SEC * rate));
  const env: number[] = [];
  for (let start = 0; start + frame <= samples.length; start += hop) {
    let sum = 0;
    for (let i = start; i < start + frame; i++) sum += samples[i] * samples[i];
    env.push(Math.sqrt(sum / frame));
  }
  return env;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Voiced runs, padded, with short gaps closed. Empty for a silent file. */
function voicedSpans(env: number[], duration: number): AudioWindow[] {
  const peak = env.reduce((m, v) => Math.max(m, v), 0);
  if (peak < 1e-6) return [];
  const threshold = Math.max(peak * 0.02, median(env) * 0.25, 1e-5);
  const pad = 0.35;
  const joinGap = 0.7;
  const spans: AudioWindow[] = [];
  let open = -1;
  for (let i = 0; i < env.length; i++) {
    const voiced = env[i] >= threshold;
    if (voiced && open < 0) open = i;
    if (!voiced && open >= 0) {
      spans.push({ startSec: open * HOP_SEC, endSec: i * HOP_SEC });
      open = -1;
    }
  }
  if (open >= 0) spans.push({ startSec: open * HOP_SEC, endSec: Math.min(duration, env.length * HOP_SEC) });

  const merged: AudioWindow[] = [];
  for (const span of spans) {
    const padded = {
      startSec: Math.max(0, span.startSec - pad),
      endSec: Math.min(duration, span.endSec + pad),
    };
    if (padded.endSec - padded.startSec < 0.3) continue;
    const last = merged[merged.length - 1];
    if (last && padded.startSec - last.endSec <= joinGap) last.endSec = padded.endSec;
    else merged.push(padded);
  }
  return merged;
}

/** Split one long run at its quietest point near the cap, so no word is cut. */
function chunkSpan(span: AudioWindow, env: number[], maxWindowSec: number): AudioWindow[] {
  const out: AudioWindow[] = [];
  let start = span.startSec;
  while (span.endSec - start > maxWindowSec) {
    const target = start + maxWindowSec;
    const from = Math.floor((target - maxWindowSec * 0.25) / HOP_SEC);
    const to = Math.floor(target / HOP_SEC);
    let cut = target;
    let quietest = Infinity;
    for (let i = Math.max(0, from); i <= Math.min(env.length - 1, to); i++) {
      if (env[i] < quietest) {
        quietest = env[i];
        cut = i * HOP_SEC;
      }
    }
    if (cut <= start + 1) cut = target;
    out.push({ startSec: start, endSec: cut });
    start = cut;
  }
  if (span.endSec - start > 0.2) out.push({ startSec: start, endSec: span.endSec });
  return out;
}

export interface WindowPlan {
  windows: AudioWindow[];
  /** True when the plan came from measured vocal activity. */
  gated: boolean;
}

/**
 * Windows for a Whisper run. A file with no measurable activity, or one this
 * cannot decode, gets a single window covering whatever duration the caller
 * knows (0 means "the whole file" to the runner).
 */
export function planWhisperWindows(
  filePath: string,
  opts: { maxWindowSec?: number; durationSec?: number } = {},
): WindowPlan {
  const maxWindowSec = Math.max(10, opts.maxWindowSec ?? 90);
  const ext = path.extname(filePath).toLowerCase();
  if (ext !== '.wav') return { windows: [{ startSec: 0, endSec: opts.durationSec ?? 0 }], gated: false };

  const decoded = decodeWavMono(filePath);
  if (!decoded) return { windows: [{ startSec: 0, endSec: opts.durationSec ?? 0 }], gated: false };
  const duration = decoded.samples.length / decoded.rate;
  if (!(duration > 0)) return { windows: [{ startSec: 0, endSec: 0 }], gated: false };

  const env = energyEnvelope(decoded.samples, decoded.rate);
  const spans = voicedSpans(env, duration);
  if (spans.length === 0) return { windows: [{ startSec: 0, endSec: duration }], gated: false };

  const windows = spans.flatMap(span => chunkSpan(span, env, maxWindowSec));
  // A plan with one window that covers the file is not worth a crop.
  if (windows.length === 1 && windows[0].startSec <= 0.01 && windows[0].endSec >= duration - 0.01) {
    return { windows: [{ startSec: 0, endSec: 0 }], gated: true };
  }
  return { windows, gated: true };
}

/** The envelope of one file, for the statistics pass. Empty when unreadable. */
export function fileEnvelope(filePath: string): number[] {
  const decoded = decodeWavMono(filePath);
  return decoded ? energyEnvelope(decoded.samples, decoded.rate) : [];
}

export { HOP_SEC };
