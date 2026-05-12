// generation/postProcessing.ts — Post-generation processing chain
//
// PP-VAE re-encode, Spectral Lifter, Vocal Naturalizer, VST chain, mastering,
// and optional Audio Quality Evaluation.
// Operates on a COPY of the raw WAV — raw generation is never modified.

import fs from 'fs';
import path from 'path';
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
  // Audio Quality Evaluator
  qualityEvalEnabled?: boolean;
  qualityEvalTarget?: 'unmastered' | 'mastered' | 'both';
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
    if (qeOn && (qeTarget === 'unmastered' || qeTarget === 'both')) {
      try {
        setStage(`Quality check (unmastered)${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
        const result = evaluateAudioQuality(rawWavPath);
        trackQuality.unmastered = result;
        log('INFO', formatQualityLog(result, `Unmastered ${audioFilename}`));
      } catch (qeErr: any) {
        log('WARNING', `[Quality] Unmastered eval failed (non-fatal): ${qeErr.message}`);
      }
    }

    if (ppVaeOn) {
      setStage(`PP-VAE Re-encode${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
      try {
        const wavBuf = fs.readFileSync(processedPath);
        const blend = params.ppVaeBlend ?? 0;
        const processed = await aceClient.submitPpVaeReencode(wavBuf, blend);
        fs.writeFileSync(processedPath, processed);
        anyStageRan = true;
        log('INFO', `[PP-VAE] Re-encoded ${audioFilename}`);
      } catch (ppErr: any) {
        log('WARNING', `[PP-VAE] Failed (non-fatal): ${ppErr.message}`);
      }
    }

    if (spectralLifterOn) {
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
    }

    // ── Vocal Naturalizer (between Spectral Lifter and VST Chain) ──
    const natOn = ppMasterOn && !!params.vocalNaturalizerEnabled && !params.instrumental;
    if (natOn) {
      try {
        const natParams: NaturalizerParams = {
          amount: params.naturalizeAmount ?? 0.5,
          vibratoRate: params.natVibratoRate ?? 4.5,
          vibratoDepth: params.natVibratoDepth ?? 1.0,
          formantStrength: params.natFormantStrength ?? 1.0,
          metallicReduction: params.natMetallicReduction ?? 1.0,
          quantizationMask: params.natQuantizationMask ?? 1.0,
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
    }

    if (ppMasterOn) {
      try {
        const applied = await applyVstChain(processedPath);
        if (applied) {
          anyStageRan = true;
          log('INFO', `[VST] Chain applied to ${processedFilename}`);
        }
      } catch (vstErr: any) {
        log('WARNING', `[VST] Chain failed (non-fatal): ${vstErr.message}`);
      }
    }

    if (masteringOn && masteringRef) {
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
    }

    // ── Quality Evaluation: Mastered (after all PP stages) ──
    if (qeOn && (qeTarget === 'mastered' || qeTarget === 'both') && anyStageRan) {
      try {
        setStage(`Quality check (mastered)${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);
        const result = evaluateAudioQuality(processedPath);
        trackQuality.mastered = result;
        log('INFO', formatQualityLog(result, `Mastered ${processedFilename}`));
      } catch (qeErr: any) {
        log('WARNING', `[Quality] Mastered eval failed (non-fatal): ${qeErr.message}`);
      }
    }

    qualityScores.push(trackQuality);

    if (anyStageRan) {
      masteredUrls.push(`/audio/${processedFilename}`);
    } else {
      try { fs.unlinkSync(processedPath); } catch {}
      masteredUrls.push('');
    }
  }

  return { masteredUrls, qualityScores };
}
