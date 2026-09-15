/**
 * audioConvert.ts — Convert non-WAV/MP3 audio to WAV using ffmpeg.
 *
 * The C++ engine (audio-io.h) only decodes WAV and MP3. Source audio from
 * Cover Studio may be FLAC, M4A, OGG, etc. This module converts on-demand.
 *
 * FFmpeg is resolved via getFFmpegPath():
 *   - Portable mode: server/ffmpeg.exe (bundled in release)
 *   - Dev mode: ffmpeg-static npm package
 *
 * Converted files are cached alongside the original so re-runs skip conversion.
 */
import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { getFFmpegPath } from '../config.js';


/** Extensions the C++ engine can decode natively */
const ENGINE_NATIVE_EXTS = new Set(['.wav', '.mp3']);

/**
 * Check if a WAV file is in the format the C++ engine can decode:
 * PCM (format=1), 16-bit, any sample rate / channel count.
 *
 * Returns true if the WAV is engine-compatible, false if it needs conversion
 * (e.g. 24-bit, 32-bit float, or non-PCM formats like ADPCM).
 */
function isEngineCompatibleWav(filePath: string): boolean {
  try {
    // Read enough of the header to find the fmt chunk
    const fd = fs.openSync(filePath, 'r');
    const hdr = Buffer.alloc(128);
    fs.readSync(fd, hdr, 0, 128, 0);
    fs.closeSync(fd);

    // Validate RIFF/WAVE container
    if (hdr.toString('ascii', 0, 4) !== 'RIFF' || hdr.toString('ascii', 8, 12) !== 'WAVE') {
      return false;
    }

    // Search for the "fmt " chunk (usually at offset 12, but not always)
    for (let offset = 12; offset < 100; offset++) {
      if (hdr.toString('ascii', offset, offset + 4) === 'fmt ') {
        // fmt chunk found — read audio format and bits per sample
        const fmtOffset = offset + 8; // skip "fmt " + chunk size (4 bytes)
        const audioFormat = hdr.readUInt16LE(fmtOffset);       // 1 = PCM, 3 = IEEE float
        const bitsPerSample = hdr.readUInt16LE(fmtOffset + 14); // offset 14 within fmt data

        if (audioFormat === 1 && bitsPerSample === 16) {
          return true;  // PCM 16-bit — engine can decode this
        }

        // Log why it's incompatible
        const sampleRate = hdr.readUInt32LE(fmtOffset + 4);
        const channels = hdr.readUInt16LE(fmtOffset + 2);
        console.log(`[audioConvert] WAV needs conversion: format=${audioFormat} bits=${bitsPerSample} rate=${sampleRate} ch=${channels}`);
        return false;
      }
    }

    // No fmt chunk found — not a valid WAV
    return false;
  } catch {
    // Can't read header — safer to convert
    return false;
  }
}

/**
 * Ensure the audio file at `filePath` is in a format the engine can decode.
 * - MP3: returned as-is (engine decodes natively)
 * - WAV 16-bit PCM: returned as-is
 * - WAV 24/32-bit, float, or non-PCM: converted to 48 kHz stereo PCM16 via ffmpeg
 * - Other formats (FLAC, OGG, M4A, etc.): converted via ffmpeg
 *
 * Converted files are cached as `<original>.engine.wav` next to the original.
 */
export function ensureEngineFormat(filePath: string, cacheDir?: string): Buffer {
  const ext = path.extname(filePath).toLowerCase();

  // MP3: always engine-compatible
  if (ext === '.mp3') {
    return fs.readFileSync(filePath);
  }

  // WAV: only compatible if 16-bit PCM
  if (ext === '.wav' && isEngineCompatibleWav(filePath)) {
    return fs.readFileSync(filePath);
  }

  // Everything else needs conversion (or WAV with non-16-bit format)

  // Check for cached conversion.
  //
  // The cache sits beside the source by default, which is right for the two
  // folders this was written for (data/references, data/audio) and WRONG for a
  // user's dataset: a training run would leave a ~50 MB .engine.wav next to
  // every track, in a folder the dataset scanner walks, so they come back as
  // untitled unlabelled rows. Callers working outside our own data dirs pass
  // a scratch `cacheDir` instead.
  const wavPath = cacheDir
    ? path.join(cacheDir, path.basename(filePath) + '.engine.wav')
    : filePath + '.engine.wav';
  if (cacheDir) fs.mkdirSync(cacheDir, { recursive: true });
  if (fs.existsSync(wavPath)) {
    console.log(`[audioConvert] Using cached conversion: ${path.basename(wavPath)}`);
    return fs.readFileSync(wavPath);
  }

  // Convert via ffmpeg
  const ffmpegPath = getFFmpegPath();
  if (!ffmpegPath) {
    // Last resort: if it's a WAV we can't convert, return it anyway and hope for the best
    if (ext === '.wav') {
      console.warn(`[audioConvert] ffmpeg not available — returning non-16-bit WAV as-is (engine may reject it)`);
      return fs.readFileSync(filePath);
    }
    throw new Error('ffmpeg not available — cannot convert non-WAV/MP3 audio');
  }

  const label = ext === '.wav' ? 'non-16-bit WAV' : ext.slice(1).toUpperCase();
  console.log(`[audioConvert] Converting ${path.basename(filePath)} (${label}) → WAV (48kHz/16-bit/stereo)...`);
  const t0 = Date.now();

  try {
    execFileSync(ffmpegPath, [
      '-y',                   // overwrite output
      '-i', filePath,         // input
      '-ar', '48000',         // 48 kHz
      '-ac', '2',             // stereo
      '-c:a', 'pcm_s16le',   // 16-bit PCM WAV
      '-f', 'wav',            // WAV container
      wavPath,                // output
    ], {
      timeout: 120_000,       // 2 min max
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  } catch (err: any) {
    // Clean up partial output
    try { fs.unlinkSync(wavPath); } catch {}
    const stderr = err.stderr?.toString()?.slice(-500) || '';
    throw new Error(`ffmpeg conversion failed: ${stderr || err.message}`);
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const size = (fs.statSync(wavPath).size / 1024 / 1024).toFixed(1);
  console.log(`[audioConvert] Done in ${elapsed}s → ${path.basename(wavPath)} (${size} MB)`);

  return fs.readFileSync(wavPath);
}

/**
 * Apply tempo-scaling and/or pitch-shifting to a WAV buffer.
 *
 * The C++ engine doesn't support these natively (the Python ACE-Step backend
 * does via release_task). So for HOT-Step CPP covers we pre-process the
 * source audio with ffmpeg before feeding it to the engine.
 *
 * @param srcBuffer  WAV/MP3 buffer (engine-compatible format)
 * @param tempoScale >1 = faster, <1 = slower (pitch-preserving time-stretch)
 * @param pitchShift Semitones: +N = higher, -N = lower (tempo-preserving)
 * @returns          Processed WAV buffer (48 kHz stereo PCM16)
 */
export function timeStretchPitchShift(
  srcBuffer: Buffer,
  tempoScale: number,
  pitchShift: number,
): Buffer {
  const ffmpegPath = getFFmpegPath();
  if (!ffmpegPath) {
    throw new Error('ffmpeg not available — cannot apply tempo/pitch changes');
  }

  // Write source to temp file
  const tmpDir = path.join(process.cwd(), 'data', 'tmp');
  if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });
  const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const tmpIn = path.join(tmpDir, `stretch_in_${id}.wav`);
  const tmpOut = path.join(tmpDir, `stretch_out_${id}.wav`);

  try {
    fs.writeFileSync(tmpIn, srcBuffer);

    // Build ffmpeg filter chain.
    //
    // Key insight: asetrate+aresample changes BOTH pitch AND tempo together.
    // To keep them independent, we compute a single combined atempo correction:
    //   - pitchFactor = 2^(semitones/12)
    //   - asetrate+aresample shifts pitch by pitchFactor but also speeds up by pitchFactor
    //   - To undo that speed change AND apply desired tempoScale:
    //     effectiveTempo = tempoScale / pitchFactor
    //   - If only pitch (tempoScale=1): atempo=1/pitchFactor → compensates speed change
    //   - If only tempo (pitchShift=0): atempo=tempoScale → standard time-stretch
    //   - If both: correctly combines both adjustments
    const filters: string[] = [];
    const pitchFactor = pitchShift !== 0 ? Math.pow(2, pitchShift / 12) : 1.0;

    // Step 1: Pitch via asetrate+aresample (also changes tempo by pitchFactor)
    if (pitchShift !== 0) {
      filters.push(`asetrate=48000*${pitchFactor.toFixed(6)}`);
      filters.push('aresample=48000');
    }

    // Step 2: Combined atempo — undo pitch's tempo side-effect + apply desired tempo
    const effectiveTempo = tempoScale / pitchFactor;
    if (Math.abs(effectiveTempo - 1.0) > 0.001) {
      // ffmpeg atempo range: 0.5–100.0. Chain for values outside this range.
      let remaining = effectiveTempo;
      while (remaining < 0.5) {
        filters.push('atempo=0.5');
        remaining /= 0.5;
      }
      while (remaining > 100.0) {
        filters.push('atempo=100.0');
        remaining /= 100.0;
      }
      filters.push(`atempo=${remaining.toFixed(6)}`);
    }

    if (filters.length === 0) {
      // No processing needed — return original
      return srcBuffer;
    }

    const filterStr = filters.join(',');
    console.log(`[audioConvert] Applying tempo=${tempoScale}x, pitch=${pitchShift}st → filter: ${filterStr}`);
    const t0 = Date.now();

    execFileSync(ffmpegPath, [
      '-y',
      '-i', tmpIn,
      '-af', filterStr,
      '-ar', '48000',
      '-ac', '2',
      '-c:a', 'pcm_s16le',
      '-f', 'wav',
      tmpOut,
    ], {
      timeout: 180_000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const outBuf = fs.readFileSync(tmpOut);
    console.log(`[audioConvert] Tempo/pitch done in ${elapsed}s → ${(outBuf.length / 1024 / 1024).toFixed(1)} MB`);
    return outBuf;
  } finally {
    // Cleanup temp files
    try { fs.unlinkSync(tmpIn); } catch {}
    try { fs.unlinkSync(tmpOut); } catch {}
  }
}
