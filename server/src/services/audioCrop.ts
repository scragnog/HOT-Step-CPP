/**
 * audioCrop.ts — Manual crop service for audio files.
 *
 * Crops a WAV file to a specified [inPoint, outPoint] range in seconds.
 * Also handles companion LRC files by filtering out-of-range lines and
 * shifting remaining timestamps by -inPoint.
 *
 * Used by the manual trim/crop UI feature. WAV parsing follows the same
 * pattern as autoTrim.ts.
 */

import fs from 'fs';
import { bumpRevision } from './lyricsRevision.js';
import { formatLrcTimestamp, shiftWordStamps, splitLrcLine } from './lrcExport.js';

// ── WAV parsing ──────────────────────────────────────────────────────────────

export interface WavInfo {
  sampleRate: number;
  numChannels: number;
  bitsPerSample: number;
  dataOffset: number;
  dataSize: number;
}

export function parseWavHeader(buf: Buffer): WavInfo {
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Not a valid WAV file');
  }

  let offset = 12;
  let fmtFound = false;
  let sampleRate = 0;
  let numChannels = 0;
  let bitsPerSample = 0;
  let dataOffset = 0;
  let dataSize = 0;

  while (offset < buf.length - 8) {
    const chunkId = buf.toString('ascii', offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);

    if (chunkId === 'fmt ') {
      numChannels = buf.readUInt16LE(offset + 10);
      sampleRate = buf.readUInt32LE(offset + 12);
      bitsPerSample = buf.readUInt16LE(offset + 22);
      fmtFound = true;
    } else if (chunkId === 'data') {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }
    offset += 8 + chunkSize;
    if (chunkSize % 2 !== 0) offset++;
  }

  if (!fmtFound || dataOffset === 0) {
    throw new Error('WAV file missing fmt or data chunk');
  }

  return { sampleRate, numChannels, bitsPerSample, dataOffset, dataSize };
}

/**
 * Read a WAV file's duration in seconds from its header (cheap — reads only the
 * first 8 KB, enough to find the fmt + data chunks). Returns 0 on any failure
 * or for non-WAV input. Used to backfill song duration when the LM/request did
 * not provide one (e.g. repaint/cover tasks where the LM is skipped).
 */
export function wavDurationSec(filePath: string): number {
  try {
    const fd = fs.openSync(filePath, 'r');
    const head = Buffer.alloc(8192);
    const n = fs.readSync(fd, head, 0, 8192, 0);
    fs.closeSync(fd);
    const info = parseWavHeader(head.subarray(0, n));
    const bytesPerFrame = info.numChannels * (info.bitsPerSample / 8);
    if (info.sampleRate <= 0 || bytesPerFrame <= 0) return 0;
    return info.dataSize / bytesPerFrame / info.sampleRate;
  } catch {
    return 0;
  }
}

// ── WAV crop ─────────────────────────────────────────────────────────────────

export interface CropResult {
  newDurationSec: number;
}

/**
 * Crop a WAV file in place to the [inPointSec, outPointSec] range.
 *
 * Applies a 5ms cosine crossfade at both cut points to prevent clicks.
 * Overwrites the file.
 */
export function cropWavFile(
  wavPath: string,
  inPointSec: number,
  outPointSec: number,
  fadeOutSec = 0.005,
): CropResult {
  const buf = fs.readFileSync(wavPath);
  const info = parseWavHeader(buf);
  const bytesPerSample = info.bitsPerSample / 8;
  const frameSize = bytesPerSample * info.numChannels;
  const totalSamples = Math.floor(info.dataSize / frameSize);

  // Clamp to valid range
  const inSample = Math.max(0, Math.floor(inPointSec * info.sampleRate));
  const outSample = Math.min(totalSamples, Math.floor(outPointSec * info.sampleRate));

  if (outSample <= inSample) {
    throw new Error('Invalid crop range: outPoint must be after inPoint');
  }

  const croppedSamples = outSample - inSample;
  const minSamples = info.sampleRate; // 1 second minimum
  if (croppedSamples < minSamples) {
    throw new Error(`Crop result too short (${(croppedSamples / info.sampleRate).toFixed(2)}s). Minimum is 1 second.`);
  }

  // Build new buffer: header + cropped PCM data
  const newDataSize = croppedSamples * frameSize;
  const newBuf = Buffer.alloc(info.dataOffset + newDataSize);

  // Copy header (everything up to data start)
  buf.copy(newBuf, 0, 0, info.dataOffset);

  // Copy PCM data from inSample to outSample
  const srcStart = info.dataOffset + inSample * frameSize;
  const srcEnd = info.dataOffset + outSample * frameSize;
  buf.copy(newBuf, info.dataOffset, srcStart, srcEnd);

  // Apply 5ms cosine crossfade at the cut points to prevent clicks
  const fadeSamples = Math.min(Math.floor(info.sampleRate * 0.005), Math.floor(croppedSamples / 4));

  if (fadeSamples > 0) {
    // Fade-in at the start of the cropped region
    for (let i = 0; i < fadeSamples; i++) {
      const gain = 0.5 * (1 - Math.cos(Math.PI * i / fadeSamples)); // 0 → 1
      for (let ch = 0; ch < info.numChannels; ch++) {
        const off = info.dataOffset + i * frameSize + ch * bytesPerSample;
        if (off + bytesPerSample > newBuf.length) continue;
        applySampleGain(newBuf, off, info.bitsPerSample, gain);
      }
    }

    // Fade-out at the end of the cropped region (callers trimming a render to
    // length pass a longer fadeOutSec so the cut sounds like an ending)
    const outFade = Math.max(fadeSamples, Math.min(Math.floor(info.sampleRate * fadeOutSec), Math.floor(croppedSamples / 4)));
    for (let i = 0; i < outFade; i++) {
      const sampleIdx = croppedSamples - outFade + i;
      const gain = 0.5 * (1 + Math.cos(Math.PI * i / outFade)); // 1 → 0
      for (let ch = 0; ch < info.numChannels; ch++) {
        const off = info.dataOffset + sampleIdx * frameSize + ch * bytesPerSample;
        if (off + bytesPerSample > newBuf.length) continue;
        applySampleGain(newBuf, off, info.bitsPerSample, gain);
      }
    }
  }

  // Update RIFF chunk size (file size - 8)
  newBuf.writeUInt32LE(newBuf.length - 8, 4);

  // Update data chunk size
  newBuf.writeUInt32LE(newDataSize, info.dataOffset - 4);

  // Write back
  fs.writeFileSync(wavPath, newBuf);

  const newDurationSec = croppedSamples / info.sampleRate;
  return { newDurationSec };
}

/** Apply a gain multiplier to a single sample at the given buffer offset */
function applySampleGain(buf: Buffer, off: number, bitsPerSample: number, gain: number): void {
  if (bitsPerSample === 16) {
    const val = buf.readInt16LE(off);
    buf.writeInt16LE(Math.round(val * gain), off);
  } else if (bitsPerSample === 32) {
    const val = buf.readFloatLE(off);
    buf.writeFloatLE(val * gain, off);
  } else if (bitsPerSample === 24) {
    const raw = buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16);
    let val = raw > 0x7FFFFF ? raw - 0x1000000 : raw;
    val = Math.round(val * gain);
    buf[off] = val & 0xFF;
    buf[off + 1] = (val >> 8) & 0xFF;
    buf[off + 2] = (val >> 16) & 0xFF;
  }
}

// ── LRC crop ─────────────────────────────────────────────────────────────────

/**
 * Crop an LRC file in place: remove lines outside [inPointSec, outPointSec],
 * then shift all remaining timestamps by -inPointSec.
 */
export function cropLrcFile(
  lrcPath: string,
  inPointSec: number,
  outPointSec: number,
): void {
  const raw = fs.readFileSync(lrcPath, 'utf-8');
  const lines = raw.replace(/\r/g, '').split('\n');
  const result: string[] = [];

  for (const line of lines) {
    const parsed = splitLrcLine(line);
    if (!parsed) {
      // Non-timestamped line (metadata like [ti:], [ar:], etc.) — keep as-is
      if (line.trim().length > 0) result.push(line);
      continue;
    }

    // A line may carry several stamps (a returning chorus), and word stamps
    // inside the text have to move with the line, or an enhanced file keeps
    // pre-crop word times.
    const text = shiftWordStamps(parsed.text, -inPointSec);
    for (const time of parsed.times) {
      // Filter: keep only lines within the crop range
      if (time < inPointSec || time > outPointSec) continue;

      // Shift timestamp by -inPointSec. Shared formatter, so rounding at a
      // centisecond boundary cannot emit `.100` here.
      result.push(`${formatLrcTimestamp(Math.max(0, time - inPointSec))} ${text}`);
    }
  }

  fs.writeFileSync(lrcPath, result.join('\n') + '\n', 'utf-8');
}

// ── Word-level lyrics crop ───────────────────────────────────────────────────

/**
 * Crop a `.lyrics.json` file in place, mirroring `cropLrcFile`.
 *
 * The player prefers this sidecar over the `.lrc`, so cropping only the LRC
 * left the karaoke highlighting — and every export read through the JSON —
 * pointing at the pre-crop timeline. Lines and words outside the range are
 * dropped; the rest shift by `-inPointSec`.
 *
 * A file that is missing or not shaped like word timings is left alone: a crop
 * must not fail over a sidecar.
 */
export function cropLyricsJsonFile(
  jsonPath: string,
  inPointSec: number,
  outPointSec: number,
): void {
  const data = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
  if (!Array.isArray(data?.lines)) return;

  type Timed = { start?: number; end?: number };
  const shift = (t: number) => Math.max(0, t - inPointSec);
  const inRange = (t: unknown): t is number =>
    typeof t === 'number' && Number.isFinite(t) && t >= inPointSec && t <= outPointSec;

  data.lines = data.lines
    .filter((line: Timed) => inRange(line?.start))
    .map((line: Timed & { words?: Timed[] }) => ({
      ...line,
      start: shift(line.start as number),
      ...(typeof line.end === 'number' ? { end: shift(line.end) } : {}),
      ...(Array.isArray(line.words)
        ? {
            words: line.words
              .filter(w => inRange(w?.start))
              .map(w => ({
                ...w,
                start: shift(w.start as number),
                ...(typeof w.end === 'number' ? { end: shift(w.end) } : {}),
              })),
          }
        : {}),
    }));

  // Cropping moves the whole timeline, so any editor open on the old one must
  // be refused when it next saves.
  bumpRevision(data);
  fs.writeFileSync(jsonPath, JSON.stringify(data, null, 2));
}
