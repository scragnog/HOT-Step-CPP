// library/importTrack.ts — bring an audio file you already have into the library
//
// The post-processing chain only runs on a song row whose audio_url is a raw
// WAV in data/audio (services/generation/rePostProcess.ts enforces exactly
// that). A FLAC on disk is neither, so "run my own track through the chain"
// had no way in. This is that way in: convert the file into the same shape a
// render has, put it where renders live, and insert the row. Everything that
// already works on a song — post-processing, stems, export, playlists — then
// works on it too, with no engine involved.

import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { randomUUID } from 'crypto';
import { config, getFFmpegPath } from '../../config.js';
import { getDb } from '../../db/database.js';
import { convertToWav } from '../../routes/mastering.js';
import { getPeaks } from '../audio/peaks.js';
import { read as readAudioMeta } from '../training/audioMeta.js';

const execFileAsync = promisify(execFile);

/** What the picker offers and the route accepts. Anything ffmpeg can decode
 *  would work; this is the list worth advertising. */
export const IMPORT_EXTENSIONS = [
  '.wav', '.mp3', '.flac', '.m4a', '.mp4', '.aac', '.ogg', '.opus', '.webm', '.aiff', '.aif',
];

/** The rates renders actually come in at — MiniMax-Music3 decodes to 44.1 kHz,
 *  YuE2 and ACE to 48 kHz, and the chain reads the header either way. An
 *  import at one of these passes through unresampled. */
const NATIVE_RATES = new Set([44100, 48000]);

/** Where anything else lands. SA3 (StableStep) is native 44.1 kHz, so an
 *  oddball rate costs one resample here instead of one inside every stage. */
const FALLBACK_RATE = 44100;

/**
 * Write `src` out as the file shape a raw render has: stereo 16-bit PCM WAV at
 * a rate the pipeline already sees.
 *
 * 16-bit PCM is not an arbitrary choice — it is what every raw render in
 * data/audio is, what the C++ engine decodes (audio-io.h, and see
 * isEngineCompatibleWav in services/audioConvert.ts), and what the waveform
 * reader draws. A 24-bit or float import would play and post-process but come
 * out of peaks.ts as `unsupported`, i.e. a track with no waveform.
 *
 * Without ffmpeg (a portable build missing the bundled binary) MP3 still gets
 * in through mp3-codec and a WAV is taken as it is; the rest need ffmpeg and
 * say so.
 */
async function toRenderShapedWav(src: string, dest: string, srcRate: number): Promise<void> {
  const ffmpeg = getFFmpegPath();
  if (!ffmpeg) {
    await convertToWav(src, dest);
    return;
  }
  const rate = NATIVE_RATES.has(srcRate) ? srcRate : FALLBACK_RATE;
  await execFileAsync(ffmpeg, [
    '-y', '-i', src,
    '-map', '0:a:0',
    '-ac', '2', '-ar', String(rate), '-c:a', 'pcm_s16le',
    dest,
  ], { timeout: 600_000 });
}

/** The title to show, from embedded tags when the file carries them and from
 *  the filename when it does not. "01 - Track.flac" is a worse name than the
 *  tags it ships with, and a file with no tags at all is common enough that
 *  the filename has to stay the fallback. */
function importTitle(meta: { artist: string; title: string }, originalName: string): string {
  const stem = path.basename(originalName, path.extname(originalName)).trim();
  if (!meta.title) return stem || 'Untitled';
  return meta.artist ? `${meta.artist} - ${meta.title}` : meta.title;
}

export interface ImportOptions {
  userId: string;
  /** Where the uploaded bytes currently sit. Left alone — the caller owns it. */
  sourcePath: string;
  /** The name the user's file had, used for the extension and the fallback title. */
  originalName: string;
}

/**
 * Import one audio file and return its new song row.
 *
 * Throws on an unsupported extension or a failed conversion; a half-written
 * destination is removed first, so a failure leaves nothing behind in the
 * audio directory.
 */
export async function importTrackFile(opts: ImportOptions): Promise<any> {
  const { userId, sourcePath, originalName } = opts;

  const ext = path.extname(originalName).toLowerCase();
  if (!IMPORT_EXTENSIONS.includes(ext)) {
    throw new Error(`Unsupported file type "${ext || originalName}". Accepted: ${IMPORT_EXTENSIONS.join(', ')}`);
  }

  // Tags come off the ORIGINAL: the WAV we are about to write carries none.
  const meta = await readAudioMeta(sourcePath);

  const songId = randomUUID();
  const filename = `${songId}.wav`;
  const destPath = path.join(config.data.audioDir, filename);
  fs.mkdirSync(config.data.audioDir, { recursive: true });

  try {
    await toRenderShapedWav(sourcePath, destPath, meta.sampleRate);
  } catch (err: any) {
    try { if (fs.existsSync(destPath)) fs.unlinkSync(destPath); } catch { /* ignore */ }
    throw new Error(`Could not convert ${originalName}: ${err?.message || err}`);
  }

  if (!fs.existsSync(destPath) || fs.statSync(destPath).size === 0) {
    try { if (fs.existsSync(destPath)) fs.unlinkSync(destPath); } catch { /* ignore */ }
    throw new Error(`Conversion produced no audio for ${originalName}`);
  }

  // Measured from the file we wrote, which also warms the waveform cache so
  // the first play does not wait on it. The tag duration is the fallback for
  // a header we could not read.
  const peaks = getPeaks(destPath);
  const duration = peaks.duration > 0 ? peaks.duration : meta.duration;

  const title = importTitle(meta, originalName);
  const generationParams = {
    source: 'import',
    importedFrom: originalName,
    importedAt: new Date().toISOString(),
    ...(meta.artist ? { artist: meta.artist } : {}),
    ...(meta.album ? { album: meta.album } : {}),
  };

  getDb().prepare(`
    INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                       duration, bpm, tags, generation_params, backend)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    songId, userId, title, '', meta.genre || '', '', `/audio/${filename}`,
    duration, meta.bpm || 0, JSON.stringify([]), JSON.stringify(generationParams), 'import',
  );

  console.log(`[Import] ${originalName} → ${filename} ("${title}", ${duration.toFixed(1)}s)`);

  return getDb().prepare('SELECT * FROM songs WHERE id = ?').get(songId);
}
