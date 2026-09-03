// training/essentiaClient.ts — direct execFile of the Essentia CLI
//
// Deliberately does NOT call our own POST /api/analyze over HTTP (D13) — this is
// a batch path that runs hundreds of times per job. The internals mirror
// routes/analyze.ts: transcode unsupported containers via ffmpeg first, then run
// essentia_streaming_extractor_music and read the JSON it drops.
//
// Essentia writes progress to stderr even on success, so success is decided by
// "did the output file appear", never by the exit code.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.5

import fs from 'fs';
import os from 'os';
import path from 'path';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { config, getFFmpegPath } from '../../config.js';

export interface EssentiaResult {
  bpm: number | null;
  key: string;      // raw tonal key, e.g. "G"
  scale: string;    // raw scale, e.g. "major"
  loudness: number | null;
  danceability: number | null;
}

/** Extensions Essentia decodes natively — anything else goes via ffmpeg. */
const NATIVE_EXTS = ['.wav', '.mp3', '.flac', '.aiff', '.aif'];

export function essentiaAvailable(): boolean {
  try {
    return !!config.essentia.bin && fs.existsSync(config.essentia.bin);
  } catch {
    return false;
  }
}

/** `"G" + "major"` → `"G Major"` — the shape the real Side-Step corpus uses. */
export function essentiaKeyString(key: string, scale: string): string {
  if (!key) return '';
  const s = (scale || '').trim();
  const titled = s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : '';
  return titled ? `${key} ${titled}` : key;
}

function run(bin: string, args: string[], timeoutMs: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    execFile(bin, args, { timeout: timeoutMs, maxBuffer: 10 * 1024 * 1024, signal }, (error) => {
      // Resolve regardless — the caller decides on the presence of the output file.
      if (error && (error as NodeJS.ErrnoException).name === 'AbortError') {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

/**
 * BPM / key / loudness / danceability for one file.
 * Returns `null` on ANY failure — a broken file must never abort a label job.
 */
export async function analyzeWithEssentia(
  audioPath: string,
  signal?: AbortSignal,
): Promise<EssentiaResult | null> {
  if (!essentiaAvailable()) return null;

  const stamp = `${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
  const tmpJson = path.join(os.tmpdir(), `hs_training_essentia_${stamp}.json`);
  let tmpWav: string | null = null;
  let inputPath = audioPath;

  try {
    const ext = path.extname(audioPath).toLowerCase();
    if (!NATIVE_EXTS.includes(ext)) {
      const ffmpeg = getFFmpegPath();
      if (!ffmpeg) {
        console.warn(`[Training] Essentia skipped for ${path.basename(audioPath)} — no ffmpeg to convert ${ext}`);
        return null;
      }
      tmpWav = path.join(os.tmpdir(), `hs_training_essentia_${stamp}.wav`);
      await run(ffmpeg, ['-y', '-i', audioPath, '-ar', '44100', '-ac', '2', tmpWav], 120_000, signal);
      if (!fs.existsSync(tmpWav)) {
        console.warn(`[Training] ffmpeg produced no WAV for ${path.basename(audioPath)}`);
        return null;
      }
      inputPath = tmpWav;
    }

    await run(config.essentia.bin, [inputPath, tmpJson], 120_000, signal);
    if (!fs.existsSync(tmpJson)) return null;

    const data = JSON.parse(fs.readFileSync(tmpJson, 'utf-8')) as any;
    const rawBpm = data?.rhythm?.bpm;
    // key_edma is the 2.1_beta5+ field the shipped builds emit; key_key/key_scale
    // is the older layout, kept so a hand-dropped 2.1_beta2 extractor still yields a key.
    const tonal = data?.tonal ?? {};
    const keyData = tonal.key_edma ?? (typeof tonal.key_key === 'string'
      ? { key: tonal.key_key, scale: tonal.key_scale } : {});
    const loud = data?.lowlevel?.average_loudness;
    const dance = data?.highlevel?.danceability?.all?.danceable;

    return {
      bpm: typeof rawBpm === 'number' && Number.isFinite(rawBpm) ? Math.round(rawBpm) : null,
      key: typeof keyData.key === 'string' ? keyData.key : '',
      scale: typeof keyData.scale === 'string' ? keyData.scale : '',
      loudness: typeof loud === 'number' && Number.isFinite(loud) ? loud : null,
      danceability: typeof dance === 'number' && Number.isFinite(dance) ? dance : null,
    };
  } catch (err: any) {
    // Cancellation kills the child via the AbortSignal; that is a null result,
    // not an error — the job loop notices `signal.aborted` on the next check.
    if (err?.name !== 'AbortError') {
      console.warn(`[Training] Essentia failed for ${path.basename(audioPath)}: ${err?.message || err}`);
    }
    return null;
  } finally {
    try { fs.unlinkSync(tmpJson); } catch { /* never existed */ }
    if (tmpWav) { try { fs.unlinkSync(tmpWav); } catch { /* never existed */ } }
  }
}
