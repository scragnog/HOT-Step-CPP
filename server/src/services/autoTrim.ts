/**
 * autoTrim.ts — Silence-detection auto-trimming for generated audio.
 *
 * After generating audio with a duration buffer, this service scans the WAV
 * from the end backwards to find the first sustained silence gap (≥1 second).
 * It trims at the start of that gap and applies a short fade-out, producing
 * a clean song ending even when the model would otherwise cut off mid-phrase
 * or restart after a natural ending.
 *
 * Algorithm:
 * 1. Parse WAV PCM data (16-bit or 32-bit float)
 * 2. Compute RMS energy in 100ms windows from the end backwards
 * 3. Find first sustained silence gap (≥1s below -40dB threshold)
 * 4. Only accept trim points after originalDuration * 0.8
 * 5. Trim at the start of the gap + apply fade-out
 * 6. If no qualifying gap: trim at originalDuration + fade-out
 */

import fs from 'fs';

// ── WAV parsing helpers ──────────────────────────────────────────────────────

interface WavInfo {
  sampleRate: number;
  numChannels: number;
  bitsPerSample: number;
  dataOffset: number;    // byte offset of PCM data
  dataSize: number;      // byte size of PCM data
}

function parseWavHeader(buf: Buffer): WavInfo {
  // Standard RIFF WAV header
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
    // Pad to even boundary
    if (chunkSize % 2 !== 0) offset++;
  }

  if (!fmtFound || dataOffset === 0) {
    throw new Error('WAV file missing fmt or data chunk');
  }

  return { sampleRate, numChannels, bitsPerSample, dataOffset, dataSize };
}

/** Read a sample (any channel, mono-mixed) as a float in [-1, 1] */
function readSampleMono(buf: Buffer, info: WavInfo, sampleIndex: number): number {
  const bytesPerSample = info.bitsPerSample / 8;
  const frameSize = bytesPerSample * info.numChannels;
  const frameOffset = info.dataOffset + sampleIndex * frameSize;

  let sum = 0;
  for (let ch = 0; ch < info.numChannels; ch++) {
    const off = frameOffset + ch * bytesPerSample;
    if (off + bytesPerSample > buf.length) return 0;

    if (info.bitsPerSample === 16) {
      sum += buf.readInt16LE(off) / 32768;
    } else if (info.bitsPerSample === 32) {
      sum += buf.readFloatLE(off);
    } else if (info.bitsPerSample === 24) {
      const val = (buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16));
      sum += (val > 0x7FFFFF ? val - 0x1000000 : val) / 8388608;
    }
  }
  return sum / info.numChannels;
}

// ── RMS computation ──────────────────────────────────────────────────────────

/** Compute RMS of a window of samples */
function computeWindowRms(buf: Buffer, info: WavInfo, startSample: number, windowSamples: number, totalSamples: number): number {
  let sumSq = 0;
  const end = Math.min(startSample + windowSamples, totalSamples);
  const count = end - startSample;
  if (count <= 0) return 0;

  for (let i = startSample; i < end; i++) {
    const s = readSampleMono(buf, info, i);
    sumSq += s * s;
  }
  return Math.sqrt(sumSq / count);
}

// ── Core trim logic ──────────────────────────────────────────────────────────

export interface AutoTrimResult {
  trimmed: boolean;
  originalDurationSec: number;
  trimmedDurationSec: number;
  trimPointSec: number;
}

/**
 * Auto-trim silence from the end of a WAV file, in place.
 *
 * Strategy (two-pass):
 * 1. Strip trailing silence from the very end of the file.
 * 2. Scan the buffer zone (originalDuration ± margin) for a sustained
 *    silence gap (≥2 seconds) — this catches the "ends, pauses, restarts"
 *    pattern by trimming at the gap before the restart.
 * 3. If no gap found in the buffer zone, trim at the last meaningful audio
 *    or fall back to originalDuration with a fade.
 *
 * @param wavPath         Path to the WAV file (will be overwritten if trimmed)
 * @param originalDuration The user's requested duration in seconds (before buffer)
 * @param fadeMs          Fade-out duration in milliseconds (default 500)
 * @returns               Info about what was trimmed
 */
export function autoTrimSilence(
  wavPath: string,
  originalDuration: number,
  fadeMs: number = 500,
): AutoTrimResult {
  const buf = fs.readFileSync(wavPath);
  const info = parseWavHeader(buf);
  const bytesPerSample = info.bitsPerSample / 8;
  const frameSize = bytesPerSample * info.numChannels;
  const totalSamples = Math.floor(info.dataSize / frameSize);
  const totalDurationSec = totalSamples / info.sampleRate;

  // Window size: 100ms
  const windowSamples = Math.floor(info.sampleRate * 0.1);
  const totalWindows = Math.floor(totalSamples / windowSamples);

  // Silence threshold: -50dB (stricter than -40dB to avoid catching quiet passages)
  const silenceThresholdLinear = Math.pow(10, -50 / 20);  // ≈ 0.00316

  // ── Pass 1: Find the last window with meaningful audio ─────────────────
  // This strips any trailing silence/noise at the very end of the file.
  let lastAudioWindow = totalWindows - 1;
  for (let w = totalWindows - 1; w >= 0; w--) {
    const windowStart = w * windowSamples;
    const rms = computeWindowRms(buf, info, windowStart, windowSamples, totalSamples);
    if (rms >= silenceThresholdLinear) {
      lastAudioWindow = w;
      break;
    }
  }
  // "Effective end" is where audio content actually stops
  const effectiveEndSample = (lastAudioWindow + 1) * windowSamples;
  const effectiveEndSec = effectiveEndSample / info.sampleRate;

  // ── Pass 2: Scan the buffer zone for a sustained silence gap ───────────
  // Only consider gaps that START after (originalDuration - 5s) to avoid
  // trimming musical breaks deep within the song.
  const bufferZoneStartSec = Math.max(0, originalDuration - 5);
  const bufferZoneStartWindow = Math.floor(bufferZoneStartSec / 0.1);
  const minGapWindows = 20;  // 2 seconds at 100ms windows

  let trimSample = -1;
  let consecutiveSilent = 0;

  // Scan backwards from effective end to find a gap in the buffer zone
  const effectiveEndWindow = Math.min(lastAudioWindow + 1, totalWindows);
  for (let w = effectiveEndWindow - 1; w >= bufferZoneStartWindow; w--) {
    const windowStart = w * windowSamples;
    const rms = computeWindowRms(buf, info, windowStart, windowSamples, totalSamples);

    if (rms < silenceThresholdLinear) {
      consecutiveSilent++;
    } else {
      if (consecutiveSilent >= minGapWindows) {
        // Found a qualifying gap — trim at this audio content's end
        // (w is the last window with audio, trim after it)
        trimSample = (w + 1) * windowSamples;
        break;
      }
      consecutiveSilent = 0;
    }
  }

  // ── Decide final trim point ────────────────────────────────────────────
  if (trimSample < 0) {
    // No silence gap found in the buffer zone.
    if (effectiveEndSec <= originalDuration + 1) {
      // Audio ends at or before original duration — no trim needed
      // (or the model ran out of content naturally)
      return {
        trimmed: false,
        originalDurationSec: totalDurationSec,
        trimmedDurationSec: totalDurationSec,
        trimPointSec: totalDurationSec,
      };
    }
    // Audio fills the entire buffer — trim at effective end
    // (strip trailing silence only, respect the song's natural length)
    trimSample = Math.min(effectiveEndSample, totalSamples);
  }

  const trimTimeSec = trimSample / info.sampleRate;

  // Don't trim if the trim point is essentially at the end already (within 0.5s)
  if (totalDurationSec - trimTimeSec < 0.5) {
    return {
      trimmed: false,
      originalDurationSec: totalDurationSec,
      trimmedDurationSec: totalDurationSec,
      trimPointSec: totalDurationSec,
    };
  }

  // Apply fade-out before the trim point
  const fadeSamples = Math.floor((fadeMs / 1000) * info.sampleRate);
  const fadeStart = Math.max(0, trimSample - fadeSamples);

  // Create new buffer with trimmed data
  const newDataSize = trimSample * frameSize;
  const newBuf = Buffer.alloc(info.dataOffset + newDataSize);

  // Copy header + data up to trim point
  buf.copy(newBuf, 0, 0, Math.min(info.dataOffset + newDataSize, buf.length));

  // Apply fade-out in the new buffer
  for (let i = fadeStart; i < trimSample; i++) {
    const fadePos = (i - fadeStart) / fadeSamples; // 0..1
    const gain = Math.cos(fadePos * Math.PI * 0.5); // cosine fade: 1→0

    for (let ch = 0; ch < info.numChannels; ch++) {
      const off = info.dataOffset + i * frameSize + ch * bytesPerSample;
      if (off + bytesPerSample > newBuf.length) continue;

      if (info.bitsPerSample === 16) {
        const val = newBuf.readInt16LE(off);
        newBuf.writeInt16LE(Math.round(val * gain), off);
      } else if (info.bitsPerSample === 32) {
        const val = newBuf.readFloatLE(off);
        newBuf.writeFloatLE(val * gain, off);
      } else if (info.bitsPerSample === 24) {
        const raw = newBuf[off] | (newBuf[off + 1] << 8) | (newBuf[off + 2] << 16);
        let val = raw > 0x7FFFFF ? raw - 0x1000000 : raw;
        val = Math.round(val * gain);
        newBuf[off] = val & 0xFF;
        newBuf[off + 1] = (val >> 8) & 0xFF;
        newBuf[off + 2] = (val >> 16) & 0xFF;
      }
    }
  }

  // Update RIFF chunk size (file size - 8)
  newBuf.writeUInt32LE(newBuf.length - 8, 4);

  // Update data chunk size
  newBuf.writeUInt32LE(newDataSize, info.dataOffset - 4);

  // Write back
  fs.writeFileSync(wavPath, newBuf);

  return {
    trimmed: true,
    originalDurationSec: totalDurationSec,
    trimmedDurationSec: trimTimeSec,
    trimPointSec: trimTimeSec,
  };
}

