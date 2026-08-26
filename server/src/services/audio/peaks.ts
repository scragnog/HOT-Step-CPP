// peaks.ts — waveform peak extraction for the player.
//
// The player used to draw its waveform by downloading the whole audio file into
// the browser and running decodeAudioData on it. For a 4:34 mastered render
// that is a 96 MB fetch, a 96 MB ArrayBuffer and a 96 MB Float32 AudioBuffer,
// per deck, retained for as long as the track is loaded. It is why the tab ran
// out of memory, and why clicking a track took seconds to make a sound.
//
// So the peaks are computed here, once, and cached. The browser gets ~20 KB of
// JSON and points an <audio> element straight at the URL, which streams over
// Range requests the way the browser has always been good at.
//
// Everything the app renders is WAV, so there is no decoder here — the header
// says where the samples are and how wide they are, and we stride over them.
// Anything else (mp3/flac, which only exist as explicit downloads) reports
// unsupported and the UI draws a flat strip rather than failing.

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';

/** Bumped when the on-disk shape changes, which invalidates every cache file. */
const PEAKS_VERSION = 1;

/** Buckets per file, regardless of length. A 1600 px waveform at 2 px a bar is
 *  800 bars, so this is two samples per bar at the widest the player ever gets
 *  and it costs ~20 KB of JSON. */
const BUCKETS = 2000;

/** Read the file in 4 MB bites. The whole point of this module is to not hold a
 *  96 MB buffer in memory, and that applies on this side of the wire too. */
const READ_CHUNK = 4 * 1024 * 1024;

export interface PeaksData {
  version: number;
  /** Seconds. Authoritative — the UI uses it before the media element knows. */
  duration: number;
  sampleRate: number;
  channels: number;
  /** Per-bucket minimum and maximum of the channel-averaged signal, -1..1. */
  min: number[];
  max: number[];
  /** Set when the format has no decoder here. min/max are empty. */
  unsupported?: string;
  /** Cache validity — recomputed when the source file no longer matches. */
  srcSize: number;
  srcMtimeMs: number;
}

// ── WAV header ───────────────────────────────────────────────────────────────

interface WavFormat {
  audioFormat: number;   // 1 = PCM, 3 = IEEE float
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
  dataOffset: number;
  dataLength: number;
}

/** Walk the RIFF chunks for `fmt ` and `data`. Chunks other than those two
 *  (LIST, fact, id3 …) are skipped rather than assumed absent — a 44-byte
 *  header is the common case, not a guarantee. */
function readWavHeader(fd: number, fileSize: number): WavFormat | { error: string } {
  const head = Buffer.alloc(Math.min(65536, fileSize));
  fs.readSync(fd, head, 0, head.length, 0);

  if (head.toString('ascii', 0, 4) !== 'RIFF' || head.toString('ascii', 8, 12) !== 'WAVE') {
    return { error: `not a WAV file (${head.toString('ascii', 0, 4)})` };
  }

  let audioFormat = 0, channels = 0, sampleRate = 0, bitsPerSample = 0;
  let dataOffset = -1, dataLength = 0;
  let off = 12;

  while (off + 8 <= head.length) {
    const id = head.toString('ascii', off, off + 4);
    const len = head.readUInt32LE(off + 4);
    const body = off + 8;

    if (id === 'fmt ' && body + 16 <= head.length) {
      audioFormat = head.readUInt16LE(body);
      channels = head.readUInt16LE(body + 2);
      sampleRate = head.readUInt32LE(body + 4);
      bitsPerSample = head.readUInt16LE(body + 14);
      // WAVE_FORMAT_EXTENSIBLE hides the real format code in the first two
      // bytes of the SubFormat GUID.
      if (audioFormat === 0xfffe && body + 26 <= head.length) {
        audioFormat = head.readUInt16LE(body + 24);
      }
    } else if (id === 'data') {
      dataOffset = body;
      // A zero or 0xFFFFFFFF length means the writer was streaming and never
      // came back to patch it. Trust the file size instead.
      dataLength = (len === 0 || len === 0xffffffff)
        ? fileSize - body
        : Math.min(len, fileSize - body);
      break;
    }

    off = body + len + (len % 2);   // RIFF chunks pad to even boundaries
  }

  if (dataOffset < 0) return { error: 'no data chunk found' };
  if (!channels || !sampleRate || !bitsPerSample) return { error: 'no fmt chunk found' };

  return { audioFormat, channels, sampleRate, bitsPerSample, dataOffset, dataLength };
}

/** A reader for one sample, normalized to -1..1, or null if we cannot decode
 *  this width and format combination. */
function sampleReader(fmt: WavFormat): ((buf: Buffer, at: number) => number) | null {
  const { audioFormat, bitsPerSample } = fmt;
  if (audioFormat === 1) {
    switch (bitsPerSample) {
      case 8:  return (b, i) => (b[i] - 128) / 128;             // 8-bit PCM is unsigned
      case 16: return (b, i) => b.readInt16LE(i) / 32768;
      // Buffer has no readInt24LE, so assemble the three bytes and sign-extend
      // by shifting the top bit up to bit 31 and back down again.
      case 24: return (b, i) => (((b[i] | (b[i + 1] << 8) | (b[i + 2] << 16)) << 8) >> 8) / 8388608;
      case 32: return (b, i) => b.readInt32LE(i) / 2147483648;
    }
  } else if (audioFormat === 3) {
    if (bitsPerSample === 32) return (b, i) => b.readFloatLE(i);
    if (bitsPerSample === 64) return (b, i) => b.readDoubleLE(i);
  }
  return null;
}

// ── Peak computation ─────────────────────────────────────────────────────────

function computeFromFile(filePath: string): PeaksData {
  const stat = fs.statSync(filePath);
  const base = { version: PEAKS_VERSION, srcSize: stat.size, srcMtimeMs: stat.mtimeMs };
  const unsupported = (reason: string, sampleRate = 44100, channels = 2): PeaksData => ({
    ...base, duration: 0, sampleRate, channels, min: [], max: [], unsupported: reason,
  });

  if (path.extname(filePath).toLowerCase() !== '.wav') {
    return unsupported(`unsupported extension ${path.extname(filePath)}`);
  }

  const fd = fs.openSync(filePath, 'r');
  try {
    const header = readWavHeader(fd, stat.size);
    if ('error' in header) return unsupported(header.error);

    const read = sampleReader(header);
    if (!read) {
      return unsupported(
        `unsupported format ${header.audioFormat}/${header.bitsPerSample}-bit`,
        header.sampleRate, header.channels
      );
    }

    const bytesPerSample = header.bitsPerSample / 8;
    const frameSize = bytesPerSample * header.channels;
    const totalFrames = Math.floor(header.dataLength / frameSize);
    const duration = totalFrames / header.sampleRate;

    const min = new Array<number>(BUCKETS).fill(0);
    const max = new Array<number>(BUCKETS).fill(0);
    if (totalFrames === 0) {
      return {
        ...base, duration: 0,
        sampleRate: header.sampleRate, channels: header.channels, min, max,
      };
    }

    // Bucket by absolute frame index rather than by a running counter, so
    // rounding cannot drift a long file's last bucket off the end.
    const framesPerBucket = totalFrames / BUCKETS;

    const buf = Buffer.allocUnsafe(READ_CHUNK - (READ_CHUNK % frameSize));
    let framePos = 0;
    let filePos = header.dataOffset;
    const dataEnd = header.dataOffset + totalFrames * frameSize;

    while (filePos < dataEnd) {
      const want = Math.min(buf.length, dataEnd - filePos);
      const got = fs.readSync(fd, buf, 0, want, filePos);
      if (got <= 0) break;
      const frames = Math.floor(got / frameSize);
      if (frames === 0) break;

      for (let f = 0; f < frames; f++) {
        const at = f * frameSize;
        let sum = 0;
        for (let c = 0; c < header.channels; c++) sum += read(buf, at + c * bytesPerSample);
        const v = sum / header.channels;

        let b = Math.floor((framePos + f) / framesPerBucket);
        if (b >= BUCKETS) b = BUCKETS - 1;
        if (v < min[b]) min[b] = v;
        if (v > max[b]) max[b] = v;
      }

      framePos += frames;
      filePos += frames * frameSize;
    }

    const round = (n: number) => Math.round(Math.max(-1, Math.min(1, n)) * 1000) / 1000;
    return {
      ...base,
      duration,
      sampleRate: header.sampleRate,
      channels: header.channels,
      min: min.map(round),
      max: max.map(round),
    };
  } finally {
    fs.closeSync(fd);
  }
}

// ── Disk cache ───────────────────────────────────────────────────────────────

function cacheDir(): string {
  const dir = path.join(config.data.dir, 'peaks');
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

/** One cache file per source, named after its path relative to the data dir so
 *  the mapping stays readable when something goes wrong. */
function cachePathFor(filePath: string): string {
  const rel = path.relative(config.data.dir, filePath).replace(/[\\/]/g, '__');
  return path.join(cacheDir(), `${rel}.peaks.json`);
}

/** Peaks for one audio file, from cache when the source has not changed.
 *  Never throws — a failure comes back as an `unsupported` payload so the UI
 *  can draw something and, more importantly, still play the track. */
export function getPeaks(filePath: string): PeaksData {
  let stat: fs.Stats;
  try {
    stat = fs.statSync(filePath);
  } catch {
    return {
      version: PEAKS_VERSION, duration: 0, sampleRate: 44100, channels: 2,
      min: [], max: [], unsupported: 'file not found', srcSize: 0, srcMtimeMs: 0,
    };
  }

  const cacheFile = cachePathFor(filePath);
  try {
    const cached = JSON.parse(fs.readFileSync(cacheFile, 'utf8')) as PeaksData;
    if (
      cached.version === PEAKS_VERSION &&
      cached.srcSize === stat.size &&
      Math.abs(cached.srcMtimeMs - stat.mtimeMs) < 1
    ) {
      return cached;
    }
  } catch { /* no cache, or a corrupt one — recompute */ }

  const started = Date.now();
  let data: PeaksData;
  try {
    data = computeFromFile(filePath);
  } catch (err) {
    data = {
      version: PEAKS_VERSION, duration: 0, sampleRate: 44100, channels: 2,
      min: [], max: [], unsupported: String(err), srcSize: stat.size, srcMtimeMs: stat.mtimeMs,
    };
  }

  try {
    fs.writeFileSync(cacheFile, JSON.stringify(data));
  } catch (err) {
    console.warn('[Peaks] could not cache', cacheFile, err);
  }

  console.log(
    `[Peaks] ${path.basename(filePath)} ${data.duration.toFixed(1)}s in ${Date.now() - started}ms`
    + (data.unsupported ? ` (${data.unsupported})` : '')
  );
  return data;
}

/** Compute and cache in the background, for the moment a render is saved, so
 *  the first listen costs nothing at all. Failures are logged, never thrown —
 *  the lazy path picks them up. */
export function precomputePeaks(filePath: string): void {
  setImmediate(() => {
    try { getPeaks(filePath); }
    catch (err) { console.warn('[Peaks] precompute failed for', filePath, err); }
  });
}
