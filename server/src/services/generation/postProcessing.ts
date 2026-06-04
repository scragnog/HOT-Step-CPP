// generation/postProcessing.ts — Post-generation processing chain
//
// PP-VAE re-encode, Spectral Lifter, Vocal Naturalizer, VST chain, mastering,
// and optional Audio Quality Evaluation.
// Operates on a COPY of the raw WAV — raw generation is never modified.

import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { config } from '../../config.js';
import { aceClient } from '../../services/aceClient.js';
import { runMastering } from '../../routes/mastering.js';
import { applyVstChain } from '../../routes/vst.js';
import { runVocalNaturalizer, type NaturalizerParams } from './vocalNaturalizer.js';
import { evaluateAudioQuality, formatQualityLog, type QualityResult } from './audioQualityEvaluator.js';

type LogFn = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => void;
type StageFn = (stage: string) => void;

interface PostProcessParams {
  postProcessingEnabled?: boolean;
  ppVaeReencode?: boolean;
  ppVaeBlend?: number;
  ppVaeUseOnnx?: boolean;
  spectralLifterEnabled?: boolean;
  slDenoiseStrength?: number;
  slNoiseFloor?: number;
  slHfMix?: number;
  slTransientBoost?: number;
  slShimmerReduction?: number;
  masteringEnabled?: boolean;
  masteringReference?: string;
  // Vocal Naturalizer
  vocalNaturalizerEnabled?: boolean;
  naturalizeAmount?: number;
  natVibratoRate?: number;
  natVibratoDepth?: number;
  natFormantStrength?: number;
  natMetallicReduction?: number;
  natQuantizationMask?: number;
  natTransitionSmooth?: number;
  // Context — used to skip naturalizer on instrumentals
  instrumental?: boolean;
  // Pre-VST gain offset (dB)
  gainOffsetDb?: number;
  // Audio Quality Evaluator
  qualityEvalEnabled?: boolean;
  qualityEvalTarget?: 'unmastered' | 'mastered' | 'both';
  // LUFS Normalization (final stage after mastering)
  lufsEnabled?: boolean;
  lufsTarget?: number;       // target integrated LUFS (e.g. -14)
  // Pipeline parallelism
  parallelQualityEval?: boolean;
}

/** Quality scores for a single track (unmastered, mastered, or both). */
export interface TrackQualityScores {
  unmastered?: QualityResult;
  mastered?: QualityResult;
}

/** Result of the full post-processing chain. */
export interface PostProcessResult {
  masteredUrls: string[];
  qualityScores: TrackQualityScores[];
  timing: Array<{ name: string; ms: number }>;
}

/** Run the full post-processing chain on a list of audio files. */
export async function runPostProcessingChain(
  audioUrls: string[],
  params: PostProcessParams,
  totalTracks: number,
  jobId: string,
  log: LogFn,
  setStage: StageFn
): Promise<PostProcessResult> {
  const ppMasterOn = params.postProcessingEnabled !== false;
  const ppVaeOn = ppMasterOn && !!params.ppVaeReencode;
  const spectralLifterOn = ppMasterOn && !!params.spectralLifterEnabled;
  const masteringRef = params.masteringReference;
  const masteringOn = ppMasterOn && !!masteringRef && !!params.masteringEnabled;
  const masteredUrls: string[] = [];
  const qualityScores: TrackQualityScores[] = [];
  const timing: Array<{ name: string; ms: number }> = [];
  const qeOn = !!params.qualityEvalEnabled;
  const qeTarget = params.qualityEvalTarget || 'unmastered';

  for (let i = 0; i < audioUrls.length; i++) {
    const audioUrl = audioUrls[i];
    const audioFilename = path.basename(audioUrl);
    const rawWavPath = path.join(config.data.audioDir, audioFilename);

    if (!rawWavPath.endsWith('.wav')) { masteredUrls.push(''); continue; }

    const ext2 = path.extname(audioFilename);
    const base2 = path.basename(audioFilename, ext2);
    const processedFilename = `${base2}_mastered${ext2}`;
    const processedPath = path.join(config.data.audioDir, processedFilename);

    fs.copyFileSync(rawWavPath, processedPath);
    let anyStageRan = false;
    const trackQuality: TrackQualityScores = {};

    // ── Quality Evaluation: Unmastered (before any PP) ──
    // When parallelQualityEval is enabled, fire QE concurrently with PP-VAE
    // (they operate on different files: QE reads rawWavPath, PP-VAE reads processedPath)
    let qePrePromise: Promise<void> | undefined;
    const runQePre = async () => {
      if (!(qeOn && (qeTarget === 'unmastered' || qeTarget === 'both'))) return;
      const qeStart = performance.now();
      try {
        if (!params.parallelQualityEval) {
          setStage(`Quality check (unmastered)${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
        }
        const result = evaluateAudioQuality(rawWavPath);
        trackQuality.unmastered = result;
        log('INFO', formatQualityLog(result, `Unmastered ${audioFilename}`));
      } catch (qeErr: any) {
        log('WARNING', `[Quality] Unmastered eval failed (non-fatal): ${qeErr.message}`);
      }
      const qeMs = Math.round(performance.now() - qeStart);
      if (qeMs > 50) timing.push({ name: 'Quality Eval (pre)', ms: qeMs });
    };

    if (params.parallelQualityEval) {
      // Fire and continue — will be awaited before mastered QE
      qePrePromise = runQePre();
    } else {
      await runQePre();
    }

    if (ppVaeOn) {
      const ppVaeStart = performance.now();
      setStage(`PP-VAE Re-encode${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
      try {
        const wavBuf = fs.readFileSync(processedPath);
        const blend = params.ppVaeBlend ?? 0;
        const processed = await aceClient.submitPpVaeReencode(wavBuf, blend, params.ppVaeUseOnnx);
        fs.writeFileSync(processedPath, processed);
        anyStageRan = true;
        log('INFO', `[PP-VAE] Re-encoded ${audioFilename}`);
      } catch (ppErr: any) {
        log('WARNING', `[PP-VAE] Failed (non-fatal): ${ppErr.message}`);
      }
      timing.push({ name: 'PP-VAE Re-encode', ms: Math.round(performance.now() - ppVaeStart) });
    }

    if (spectralLifterOn) {
      const slStart = performance.now();
      setStage(`Spectral Lifter${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
      try {
        const wavBuf = fs.readFileSync(processedPath);
        const slParams = {
          denoise_strength: params.slDenoiseStrength ?? 0.3,
          noise_floor: params.slNoiseFloor ?? 0.1,
          hf_mix: params.slHfMix ?? 0.0,
          transient_boost: params.slTransientBoost ?? 0.0,
          shimmer_reduction: params.slShimmerReduction ?? 6.0,
        };
        const processed = await aceClient.submitSpectralLifter(wavBuf, slParams);
        fs.writeFileSync(processedPath, processed);
        anyStageRan = true;
        log('INFO', `[Spectral Lifter] Applied to ${audioFilename}`);
      } catch (slErr: any) {
        log('WARNING', `[Spectral Lifter] Failed (non-fatal): ${slErr.message}`);
      }
      timing.push({ name: 'Spectral Lifter', ms: Math.round(performance.now() - slStart) });
    }

    // ── Vocal Naturalizer (between Spectral Lifter and VST Chain) ──
    const natOn = ppMasterOn && !!params.vocalNaturalizerEnabled && !params.instrumental;
    if (natOn) {
      const natStart = performance.now();
      try {
        const natParams: NaturalizerParams = {
          amount: params.naturalizeAmount ?? 0.5,
          vibratoRate: params.natVibratoRate ?? 4.5,
          vibratoDepth: params.natVibratoDepth ?? 1.0,
          formantStrength: params.natFormantStrength ?? 1.0,
          metallicReduction: params.natMetallicReduction ?? 1.0,
          quantizationMask: params.natQuantizationMask ?? 0.0,
          transitionSmooth: params.natTransitionSmooth ?? 1.0,
        };
        const applied = await runVocalNaturalizer(
          processedPath, natParams, log, setStage, i, audioUrls.length
        );
        if (applied) {
          anyStageRan = true;
          log('INFO', `[Vocal Naturalizer] Applied to ${processedFilename}`);
        }
      } catch (natErr: any) {
        log('WARNING', `[Vocal Naturalizer] Failed (non-fatal): ${natErr.message}`);
      }
      timing.push({ name: 'Vocal Naturalizer', ms: Math.round(performance.now() - natStart) });
    }

    // ── Pre-VST Gain Offset ──
    const gainDb = params.gainOffsetDb ?? 0;
    if (ppMasterOn && gainDb !== 0) {
      const gainStart = performance.now();
      setStage(`Gain offset ${gainDb > 0 ? '+' : ''}${gainDb} dB${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
      try {
        const buf = fs.readFileSync(processedPath);
        // Parse WAV: find 'data' chunk
        let dataOffset = -1;
        for (let off = 36; off < buf.length - 8; off++) {
          if (buf[off] === 0x64 && buf[off+1] === 0x61 && buf[off+2] === 0x74 && buf[off+3] === 0x61) {
            dataOffset = off;
            break;
          }
        }
        if (dataOffset >= 0) {
          const dataSize = buf.readUInt32LE(dataOffset + 4);
          const pcmStart = dataOffset + 8;
          const audioFormat = buf.readUInt16LE(20);
          const bitsPerSample = buf.readUInt16LE(34);
          const linearGain = Math.pow(10, gainDb / 20);

          if (audioFormat === 1 && bitsPerSample === 16) {
            // PCM 16-bit
            for (let p = pcmStart; p + 1 < pcmStart + dataSize && p + 1 < buf.length; p += 2) {
              let sample = buf.readInt16LE(p) * linearGain;
              sample = Math.max(-32768, Math.min(32767, Math.round(sample)));
              buf.writeInt16LE(sample, p);
            }
          } else if (audioFormat === 3 && bitsPerSample === 32) {
            // IEEE float 32-bit
            for (let p = pcmStart; p + 3 < pcmStart + dataSize && p + 3 < buf.length; p += 4) {
              buf.writeFloatLE(buf.readFloatLE(p) * linearGain, p);
            }
          }
          // else: unsupported format, skip silently

          fs.writeFileSync(processedPath, buf);
          anyStageRan = true;
          log('INFO', `[Gain] Applied ${gainDb > 0 ? '+' : ''}${gainDb} dB to ${processedFilename}`);
        }
      } catch (gainErr: any) {
        log('WARNING', `[Gain] Offset failed (non-fatal): ${gainErr.message}`);
      }
      const gainMs = Math.round(performance.now() - gainStart);
      if (gainMs > 10) timing.push({ name: 'Gain Offset', ms: gainMs });
    }

    if (ppMasterOn) {
      const vstStart = performance.now();
      setStage(`Applying VST chain${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
      try {
        const applied = await applyVstChain(processedPath);
        if (applied) {
          anyStageRan = true;
          log('INFO', `[VST] Chain applied to ${processedFilename}`);
        }
      } catch (vstErr: any) {
        log('WARNING', `[VST] Chain failed (non-fatal): ${vstErr.message}`);
      }
      const vstMs = Math.round(performance.now() - vstStart);
      if (vstMs > 50) timing.push({ name: 'VST Chain', ms: vstMs });
    }

    if (masteringOn && masteringRef) {
      const masterStart = performance.now();
      setStage(`Mastering${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
      try {
        const refPath = masteringRef.startsWith('/references/')
          ? path.join(config.data.dir, 'references', masteringRef.replace('/references/', ''))
          : path.isAbsolute(masteringRef)
            ? masteringRef
            : path.join(config.data.dir, 'references', masteringRef);
        const tempMastered = processedPath + '.mastered.wav';
        await runMastering(processedPath, refPath, tempMastered);
        fs.renameSync(tempMastered, processedPath);
        anyStageRan = true;
        log('INFO', `[Mastering] Applied to ${processedFilename}`);
      } catch (masterErr: any) {
        log('WARNING', `[Mastering] Failed (non-fatal): ${masterErr.message}`);
      }
      timing.push({ name: 'Mastering', ms: Math.round(performance.now() - masterStart) });
    }

    // ── LUFS Normalization (final audio-modifying stage) ──
    const lufsOn = ppMasterOn && masteringOn && !!params.lufsEnabled && params.lufsTarget !== undefined;
    if (lufsOn && params.lufsTarget !== undefined) {
      const lufsStart = performance.now();
      setStage(`LUFS normalization${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
      try {
        const { normalizeLufs } = await import('./lufsNormalize.js');
        const result = normalizeLufs(processedPath, params.lufsTarget);
        anyStageRan = true;
        log('INFO',
          `[LUFS] ${processedFilename}: ${result.measuredLufs.toFixed(1)} → ${result.targetLufs.toFixed(1)} LUFS ` +
          `(${result.appliedGainDb > 0 ? '+' : ''}${result.appliedGainDb.toFixed(1)} dB` +
          `${result.limiterActive ? ', limiter active' : ''})` +
          ` | Peak: ${(20 * Math.log10(Math.max(result.peakBefore, 1e-10))).toFixed(1)} → ${(20 * Math.log10(Math.max(result.peakAfter, 1e-10))).toFixed(1)} dBFS`
        );
      } catch (lufsErr: any) {
        log('WARNING', `[LUFS] Normalization failed (non-fatal): ${lufsErr.message}`);
      }
      timing.push({ name: 'LUFS Normalize', ms: Math.round(performance.now() - lufsStart) });
    }

    // ── Quality Evaluation: Mastered (after all PP stages) ──
    // Ensure pre-QE (if deferred) has completed before we proceed
    if (qePrePromise) await qePrePromise;
    if (qeOn && (qeTarget === 'mastered' || qeTarget === 'both') && anyStageRan) {
      const qePostStart = performance.now();
      try {
        setStage(`Quality check (mastered)${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
        const result = evaluateAudioQuality(processedPath);
        trackQuality.mastered = result;
        log('INFO', formatQualityLog(result, `Mastered ${processedFilename}`));
      } catch (qeErr: any) {
        log('WARNING', `[Quality] Mastered eval failed (non-fatal): ${qeErr.message}`);
      }
      const qePostMs = Math.round(performance.now() - qePostStart);
      if (qePostMs > 50) timing.push({ name: 'Quality Eval (post)', ms: qePostMs });
    }

    qualityScores.push(trackQuality);

    if (anyStageRan) {
      masteredUrls.push(`/audio/${processedFilename}`);
    } else {
      try { fs.unlinkSync(processedPath); } catch {}
      masteredUrls.push('');
    }
  }

  return { masteredUrls, qualityScores, timing };
}
