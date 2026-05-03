/**
 * audioConvert.ts — Convert non-WAV/MP3 audio to WAV using bundled ffmpeg.
 *
 * The C++ engine (audio-io.h) only decodes WAV and MP3. Source audio from
 * Cover Studio may be FLAC, M4A, OGG, etc. This module converts on-demand
 * using ffmpeg-static (self-contained, no system dependency).
 *
 * Converted files are cached alongside the original so re-runs skip conversion.
 */
import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// ffmpeg-static exports the path to the bundled binary
// @ts-ignore — ffmpeg-static default export is a string path at runtime
import ffmpegPathImport from 'ffmpeg-static';
const ffmpegPath: string | null = ffmpegPathImport as unknown as string | null;

/** Extensions the C++ engine can decode natively */
const ENGINE_NATIVE_EXTS = new Set(['.wav', '.mp3']);

/**
 * Ensure the audio file at `filePath` is in a format the engine can decode.
 * If it's already WAV or MP3, returns the original buffer unchanged.
 * Otherwise, converts to 48 kHz stereo WAV via ffmpeg and returns that buffer.
 *
 * Converted files are cached as `<original>.engine.wav` next to the original.
 */
export function ensureEngineFormat(filePath: string): Buffer {
  const ext = path.extname(filePath).toLowerCase();

  // Already engine-compatible — read and return
  if (ENGINE_NATIVE_EXTS.has(ext)) {
    return fs.readFileSync(filePath);
  }

  // Check for cached conversion
  const wavPath = filePath + '.engine.wav';
  if (fs.existsSync(wavPath)) {
    console.log(`[audioConvert] Using cached conversion: ${path.basename(wavPath)}`);
    return fs.readFileSync(wavPath);
  }

  // Convert via ffmpeg-static
  if (!ffmpegPath) {
    throw new Error('ffmpeg-static not available — cannot convert non-WAV/MP3 audio');
  }

  console.log(`[audioConvert] Converting ${path.basename(filePath)} → WAV (48kHz stereo)...`);
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
  if (!ffmpegPath) {
    throw new Error('ffmpeg-static not available — cannot apply tempo/pitch changes');
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
