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
