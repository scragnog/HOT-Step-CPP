// audioSlice.ts — a short WAV for one alignment window.
//
// Whisper's CLI is given a real file per window rather than `--offset-t`, so a
// timestamp it reports is relative to that file and nothing has to be known
// about how a particular whisper.cpp build treats offsets. The CTC forced
// aligner works on a crop for the same reason: a selection is aligned against
// the audio under it, not against the whole track with the rest of the lyrics
// held back.
//
// ffmpeg does the decoding, because a render may be 24-bit PCM or an MP3 and
// the rest of the server already treats ffmpeg as the decoder of record.

import { execFile } from 'child_process';
import { promisify } from 'util';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { config, getFFmpegPath } from '../config.js';

const execFileAsync = promisify(execFile);

/** Where windows are written. Cleared as soon as the run that made them ends. */
export function sliceDir(): string {
  return path.join(config.data.dir, 'tmp', 'lyric-slices');
}

/**
 * Write `[startSec, endSec)` of `inputPath` as a 48 kHz stereo PCM16 WAV and
 * return its path. The caller owns the file and should pass it to
 * `removeSlice` when done.
 */
export async function sliceToWav(
  inputPath: string,
  startSec: number,
  endSec: number,
  opts: { signal?: AbortSignal } = {},
): Promise<string> {
  const dir = sliceDir();
  fs.mkdirSync(dir, { recursive: true });
  const out = path.join(dir, `${Date.now().toString(36)}-${crypto.randomUUID()}.wav`);
  const ffmpeg = getFFmpegPath();
  if (!ffmpeg) throw new Error('ffmpeg is not available, so an audio range cannot be cut');
  const start = Math.max(0, startSec);
  const duration = Math.max(0.2, endSec - start);
  try {
    await execFileAsync(ffmpeg, [
      '-y', '-v', 'error',
      '-ss', start.toFixed(3),
      '-t', duration.toFixed(3),
      '-i', inputPath,
      '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le', '-f', 'wav',
      out,
    ], { timeout: 120_000, maxBuffer: 8 * 1024 * 1024, ...(opts.signal ? { signal: opts.signal } : {}) });
  } catch (err: any) {
    fs.rmSync(out, { force: true });
    throw new Error(`Cannot cut ${start.toFixed(2)}s..${endSec.toFixed(2)}s from the render: ${err?.stderr?.toString()?.slice(-300) || err.message}`);
  }
  return out;
}

export function removeSlice(file: string): void {
  fs.rmSync(file, { force: true });
}
