// generation/sourceAudio.ts — Source audio and timbre reference preparation
//
// Handles loading, format conversion, and pre-processing of source audio
// for cover/repaint tasks and timbre reference conditioning.

import fs from 'fs';
import path from 'path';
import { config } from '../../config.js';
import { mapPath } from '../../services/pathMapper.js';
import { ensureEngineFormat, timeStretchPitchShift } from '../../services/audioConvert.js';
import { readHslat, latentFrameCount, latentDuration } from '../../services/latentFormat.js';
import { convertToWav } from '../../routes/mastering.js';
import { readDuration } from '../training/audioMeta.js';

type LogFn = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => void;

/** Load and prepare source audio for cover/repaint tasks */
export function loadSourceAudio(
  sourceAudioUrl: string | undefined,
  jobId: string,
  log: LogFn
): Buffer | undefined {
  if (!sourceAudioUrl) return undefined;

  const resolvedUrl = mapPath(sourceAudioUrl) || sourceAudioUrl;
  const srcPath = resolvedUrl.startsWith('/references/')
    ? path.join(config.data.dir, 'references', resolvedUrl.replace('/references/', ''))
    : resolvedUrl.startsWith('/audio/')
      ? path.join(config.data.audioDir, resolvedUrl.replace('/audio/', ''))
      : path.isAbsolute(resolvedUrl)
        ? resolvedUrl
        : path.join(config.data.dir, resolvedUrl);

  log('DEBUG', `[Synth Phase] Looking for source audio at: ${srcPath}`);

  if (!fs.existsSync(srcPath)) {
    // Warning-and-continue submitted a cover with no source at all, which the
    // engine rejected as a bare "Generation failed" (#130). Fail here, with
    // the path, so a stale upload reference is diagnosable from the UI.
    log('ERROR', `[Synth Phase] Source audio not found: ${srcPath}`);
    throw new Error(`Source audio file is missing (${path.basename(srcPath)}). The upload it referred to is gone; re-upload the track.`);
  }

  try {
    const buf = ensureEngineFormat(srcPath);
    log('INFO', `[Synth Phase] Source audio (cover): ${srcPath} (${(buf.length / 1024 / 1024).toFixed(1)} MB)`);
    return buf;
  } catch (convErr: any) {
    log('WARNING', `[Synth Phase] Audio conversion failed: ${convErr.message}`);
    return fs.readFileSync(srcPath);
  }
}

/** Load source latent (skips VAE encode) */
export function loadSourceLatent(
  sourceLatentUrl: string | undefined,
  log: LogFn
): Buffer | undefined {
  if (!sourceLatentUrl) return undefined;

  const latentPath = sourceLatentUrl.startsWith('/audio/')
    ? path.join(config.data.audioDir, sourceLatentUrl.replace('/audio/', ''))
    : path.isAbsolute(sourceLatentUrl) ? sourceLatentUrl : path.join(config.data.dir, sourceLatentUrl);

  if (!fs.existsSync(latentPath)) {
    log('WARNING', `[Latent] Source latent not found: ${latentPath}`);
    return undefined;
  }

  try {
    const fileContents = fs.readFileSync(latentPath);
    const parsed = readHslat(fileContents);
    const rawLatent = parsed.rawLatent;
    if (rawLatent.length % 256 !== 0) {
      log('WARNING', `[Latent] Invalid latent file size (${rawLatent.length} bytes), ignoring`);
      return undefined;
    }
    log('INFO', `[Latent] Source latent loaded: ${latentPath} (${latentFrameCount(rawLatent)} frames, ${latentDuration(rawLatent).toFixed(1)}s)`);
    return rawLatent;
  } catch (latErr: any) {
    log('WARNING', `[Latent] Failed to read source latent: ${latErr.message}`);
    return undefined;
  }
}

/** Apply tempo/pitch pre-processing to source audio */
export function applyTempoAndPitch(
  srcAudioBuf: Buffer,
  tempoScale: number | undefined,
  pitchShift: number | undefined,
  log: LogFn
): Buffer {
  if ((!tempoScale || tempoScale === 1.0) && (!pitchShift || pitchShift === 0)) return srcAudioBuf;

  try {
    log('INFO', `[Synth Phase] Pre-processing source audio: tempo=${tempoScale ?? 1.0}x, pitch=${pitchShift ?? 0}st`);
    const result = timeStretchPitchShift(srcAudioBuf, tempoScale ?? 1.0, pitchShift ?? 0);
    log('INFO', `[Synth Phase] Pre-processed source audio: ${(result.length / 1024 / 1024).toFixed(1)} MB`);
    return result;
  } catch (err: any) {
    log('WARNING', `[Synth Phase] Tempo/pitch pre-processing failed: ${err.message}`);
    return srcAudioBuf;
  }
}

/** Longest reference audio /synth will accept, in seconds.
 *
 *  The engine rejects MAX_T_LATENT frames or more — 15000 frames at 1920
 *  samples each is exactly 600 s at 48 kHz (hot-step-server.cpp). Two seconds
 *  of margin absorbs resample rounding on 44.1 kHz sources. */
const TIMBRE_MAX_SECONDS = 598;

/** Resolve and load timbre reference audio */
export async function loadTimbreReference(
  params: any,
  masteringRef: string | undefined,
  seed: number | undefined,
  jobId: string,
  log: LogFn
): Promise<Buffer | undefined> {
  const rawTimbre = params.timbreReference;
  const timbreRef = (rawTimbre === true && typeof masteringRef === 'string')
    ? masteringRef
    : (typeof rawTimbre === 'string' ? rawTimbre : undefined);

  log('DEBUG', `[Synth Phase] timbreRef=${timbreRef}, masteringRef=${masteringRef}`);
  if (!timbreRef) return undefined;

  const mappedRef = mapPath(timbreRef) || timbreRef;
  let refPath = mappedRef.startsWith('/references/')
    ? path.join(config.data.dir, 'references', mappedRef.replace('/references/', ''))
    : path.isAbsolute(mappedRef)
      ? mappedRef
      : path.join(config.data.dir, 'references', mappedRef);

  // Randomize timbre reference
  //
  // The pick has to respect the engine's limit. /synth rejects reference audio
  // of MAX_T_LATENT frames or more (hot-step-server.cpp: T_audio / 1920 >=
  // 15000, i.e. 600 s at 48 kHz) with a 413, and an album with one long track
  // in it would then fail at whatever rate the roll happened to land there —
  // a Goldie "Timeless" render died on the 21-minute title track. So walk
  // forward from the seeded index and take the first candidate that fits,
  // rather than failing the render on a pick we could see was too long.
  if (params.randomizeTimbreRef) {
    try {
      const refDir = path.dirname(refPath);
      const audioExts = new Set(['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']);
      const candidates = fs.readdirSync(refDir)
        .filter(f => audioExts.has(path.extname(f).toLowerCase()))
        .sort();

      if (candidates.length > 1) {
        const s = seed ?? 0;
        const start = Math.abs(s) % candidates.length;

        let picked = '';
        let pickedIdx = -1;
        const skipped: string[] = [];
        // One probe per candidate at worst, and normally exactly one: the
        // seeded pick is usually already short enough.
        for (let n = 0; n < candidates.length; n++) {
          const i = (start + n) % candidates.length;
          const cand = path.join(refDir, candidates[i]);
          const dur = await readDuration(cand);
          // dur === 0 means the duration could not be read. Let it through —
          // the engine is the backstop, and refusing everything unreadable
          // would be worse than the occasional 413.
          if (dur === 0 || dur < TIMBRE_MAX_SECONDS) {
            picked = cand;
            pickedIdx = i;
            break;
          }
          skipped.push(`${candidates[i]} (${(dur / 60).toFixed(1)} min)`);
        }

        if (picked) {
          if (skipped.length) {
            log('INFO', `[Timbre] Skipped ${skipped.length} over-length candidate(s): ${skipped.join(', ')}`);
          }
          log('INFO', `[Timbre] Randomized: picked "${candidates[pickedIdx]}" (${pickedIdx + 1}/${candidates.length}, seed=${s}) from ${refDir}`);
          refPath = picked;
        } else {
          log('WARNING', `[Timbre] Every candidate in ${refDir} is ${TIMBRE_MAX_SECONDS / 60} min or longer — using original reference`);
        }
      } else {
        log('INFO', `[Timbre] Randomize enabled but only ${candidates.length} audio file(s) in ${refDir} — using original`);
      }
    } catch (dirErr: any) {
      log('WARNING', `[Timbre] Randomize failed (using original): ${dirErr.message}`);
    }
  }

  log('DEBUG', `[Synth Phase] Looking for timbre ref at: ${refPath}`);
  if (!fs.existsSync(refPath)) {
    log('ERROR', `[Synth Phase] Timbre reference file not found: ${refPath}`);
    throw new Error(`Timbre reference file is missing (${path.basename(refPath)}). The upload it referred to is gone; re-upload it or clear the reference.`);
  }

  const refExt = path.extname(refPath).toLowerCase();
  let readPath = refPath;
  let tempWav: string | undefined;

  if (refExt !== '.wav' && refExt !== '.mp3') {
    try {
      tempWav = path.join(config.data.dir, `timbre_temp_${jobId}.wav`);
      log('INFO', `[Synth Phase] Converting timbre ref ${refExt} → WAV via ffmpeg`);
      await convertToWav(refPath, tempWav);
      readPath = tempWav;
    } catch (convErr: any) {
      log('WARNING', `[Synth Phase] Timbre ref conversion failed (${convErr.message}), sending raw file`);
      readPath = refPath;
      tempWav = undefined;
    }
  }

  const buf = fs.readFileSync(readPath);
  log('INFO', `[Synth Phase] Timbre reference: ${refPath} (${(buf.length / 1024 / 1024).toFixed(1)} MB)`);

  if (tempWav && fs.existsSync(tempWav)) {
    try { fs.unlinkSync(tempWav); } catch {}
  }

  return buf;
}
