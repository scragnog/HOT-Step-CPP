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
import { sa3ModelsInstalled, tokenizeForSa3, buildStableStepPrompt } from '../sa3Tokenizer.js';
import { wavDurationSec } from '../audioCrop.js';

type LogFn = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', msg: string) => void;
type StageFn = (stage: string) => void;

interface PostProcessParams {
  postProcessingEnabled?: boolean;
  ppVaeReencode?: boolean;
  ppVaeBlend?: number;
  ppVaeUseOnnx?: boolean;
  // StableStep — SA3 (Stable Audio 3) SDEdit refine of the instrumental
  stableStepOn?: boolean;
  stableStep?: boolean;          // preset/settings-file alias for stableStepOn
  stableStepStrength?: number;   // 0..1 init noise level (default 0.3)
  /** Per-track captions (parallel to audioUrls) used to build the SA3 prompt.
   *  Populated by the generate route from the LM results. */
  stableStepCaptions?: string[];
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
  const stableStepOn = ppMasterOn && !!(params.stableStepOn ?? params.stableStep);
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

    // ── StableStep: SA3 SDEdit refine (before PP-VAE) ──
    // Instrumental gens: refine the whole mix through the SA3 model.
    // Vocal gens: SuperSep-split into vocals + instrumental, SA3-refine the
    // instrumental, PP-VAE the vocals, then recombine sample-wise in Node.
    if (stableStepOn) {
      const ssStart = performance.now();
      try {
        if (!sa3ModelsInstalled()) {
          log('WARNING', '[StableStep] SA3 models not installed (models/onnx/sa3) — skipping');
        } else {
          const strength = params.stableStepStrength ?? 0.3;
          const caption = params.stableStepCaptions?.[i] || '';
          const durationSec = wavDurationSec(processedPath);
          const prompt = buildStableStepPrompt(caption, durationSec);
          const { ids, nTokens } = await tokenizeForSa3(prompt);
          log('INFO', `[StableStep] Prompt (${nTokens} tokens): ${prompt}`);
          const suffix = totalTracks > 1 ? ` (${i + 1}/${totalTracks})` : '';

          if (params.instrumental) {
            // Whole-mix refine — no stems needed
            setStage(`StableStep: refining instrumental${suffix}...`);
            const wavBuf = fs.readFileSync(processedPath);
            const refined = await aceClient.submitSa3Refine(wavBuf, {
              tokens: ids, nTokens, strength,
            });
            fs.writeFileSync(processedPath, refined);
          } else {
            // Full stem workflow: separate → refine instrumental / clean vocals → recombine
            setStage(`StableStep: separating stems${suffix}...`);
            const srcBuf = fs.readFileSync(processedPath);
            const sepId = await aceClient.submitSuperSepSeparate(srcBuf, 0 /* BASIC: 6 stems */);

            // Poll separation to completion (GPU-serialized with other engine work)
            const sepDeadline = Date.now() + 30 * 60_000;
            for (;;) {
              const prog = await aceClient.superSepProgress(sepId);
              if (prog.status === 'done') break;
              if (prog.status === 'failed' || prog.status === 'cancelled') {
                throw new Error(`SuperSep ${prog.status}: ${prog.error || prog.message || 'unknown'}`);
              }
              if (Date.now() > sepDeadline) throw new Error('SuperSep separation timed out');
              await new Promise(r => setTimeout(r, 500));
            }

            const sepResult = await aceClient.superSepResult(sepId);
            const stems = sepResult.stems;
            const isVocal = (s: { category: string }) => s.category === 'vocals';
            const hasVocalStem = stems.some(s => isVocal(s) && !s.hidden);

            if (!hasVocalStem) {
              // No vocal energy detected — refine the whole mix directly
              log('INFO', '[StableStep] No vocal stem found — refining full mix');
              setStage(`StableStep: refining instrumental${suffix}...`);
              const refined = await aceClient.submitSa3Refine(srcBuf, {
                tokens: ids, nTokens, strength,
              });
              fs.writeFileSync(processedPath, refined);
            } else {
              // Engine-side recombine gives 48 kHz WAVs for both halves:
              // instrumental = everything except vocals, vocals = solo vocals.
              const instControls = stems.map(s => ({
                index: s.index, volume: 1.0, muted: s.hidden || isVocal(s),
              }));
              const vocalControls = stems.map(s => ({
                index: s.index, volume: 1.0, muted: s.hidden || !isVocal(s),
              }));
              const instBuf = await aceClient.superSepRecombine(sepId, instControls);
              const vocalBuf = await aceClient.superSepRecombine(sepId, vocalControls);

              setStage(`StableStep: refining instrumental${suffix}...`);
              const refinedInst = await aceClient.submitSa3Refine(instBuf, {
                tokens: ids, nTokens, strength,
              });

              setStage(`StableStep: processing vocals${suffix}...`);
              let cleanVocals = vocalBuf;
              try {
                cleanVocals = await aceClient.submitPpVaeReencode(vocalBuf, 0.0);
              } catch (vErr: any) {
                log('WARNING', `[StableStep] Vocal PP-VAE failed, using raw vocal stem: ${vErr.message}`);
              }

              setStage(`StableStep: recombining${suffix}...`);
              const mixed = mixWavBuffers(refinedInst, cleanVocals);
              fs.writeFileSync(processedPath, mixed);
            }
          }

          anyStageRan = true;
          log('INFO', `[StableStep] Refined ${audioFilename} (strength=${strength})`);
        }
      } catch (ssErr: any) {
        log('WARNING', `[StableStep] Failed (non-fatal): ${ssErr.message}`);
      }
      timing.push({ name: 'StableStep', ms: Math.round(performance.now() - ssStart) });
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

// ── WAV mix helpers (StableStep recombine) ──────────────────────────────────
// No shared float-WAV parse/encode helper exists in server/src (audioCrop.ts
// keeps its header parser private and operates in-place), so StableStep uses
// this minimal local implementation: 16-bit PCM + 32-bit float, stereo/mono.

interface ParsedWavAudio {
  sampleRate: number;
  numChannels: number;
  /** Interleaved samples, normalized to [-1, 1] floats. */
  samples: Float32Array;
}

function parseWavToFloat(buf: Buffer): ParsedWavAudio {
  if (buf.length < 44 ||
      buf.toString('ascii', 0, 4) !== 'RIFF' ||
      buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Not a valid WAV file');
  }

  let offset = 12;
  let audioFormat = 0, numChannels = 0, sampleRate = 0, bitsPerSample = 0;
  let dataOffset = -1, dataSize = 0;

  while (offset + 8 <= buf.length) {
    const chunkId = buf.toString('ascii', offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    if (chunkId === 'fmt ') {
      audioFormat = buf.readUInt16LE(offset + 8);
      numChannels = buf.readUInt16LE(offset + 10);
      sampleRate = buf.readUInt32LE(offset + 12);
      bitsPerSample = buf.readUInt16LE(offset + 22);
    } else if (chunkId === 'data') {
      dataOffset = offset + 8;
      dataSize = Math.min(chunkSize, buf.length - dataOffset);
      break;
    }
    offset += 8 + chunkSize + (chunkSize % 2);
  }

  if (dataOffset < 0 || sampleRate <= 0 || numChannels <= 0) {
    throw new Error('WAV file missing fmt or data chunk');
  }

  let samples: Float32Array;
  if (audioFormat === 1 && bitsPerSample === 16) {
    const n = Math.floor(dataSize / 2);
    samples = new Float32Array(n);
    for (let s = 0; s < n; s++) {
      samples[s] = buf.readInt16LE(dataOffset + s * 2) / 32768;
    }
  } else if (audioFormat === 3 && bitsPerSample === 32) {
    const n = Math.floor(dataSize / 4);
    samples = new Float32Array(n);
    for (let s = 0; s < n; s++) {
      samples[s] = buf.readFloatLE(dataOffset + s * 4);
    }
  } else {
    throw new Error(`Unsupported WAV format (fmt=${audioFormat}, ${bitsPerSample}-bit)`);
  }

  return { sampleRate, numChannels, samples };
}

function encodeWav16(samples: Float32Array, sampleRate: number, numChannels: number): Buffer {
  const dataSize = samples.length * 2;
  const out = Buffer.alloc(44 + dataSize);
  out.write('RIFF', 0, 'ascii');
  out.writeUInt32LE(36 + dataSize, 4);
  out.write('WAVE', 8, 'ascii');
  out.write('fmt ', 12, 'ascii');
  out.writeUInt32LE(16, 16);                                   // fmt chunk size
  out.writeUInt16LE(1, 20);                                    // PCM
  out.writeUInt16LE(numChannels, 22);
  out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * numChannels * 2, 28);         // byte rate
  out.writeUInt16LE(numChannels * 2, 32);                      // block align
  out.writeUInt16LE(16, 34);                                   // bits per sample
  out.write('data', 36, 'ascii');
  out.writeUInt32LE(dataSize, 40);
  for (let s = 0; s < samples.length; s++) {
    const v = Math.max(-32768, Math.min(32767, Math.round(samples[s] * 32767)));
    out.writeInt16LE(v, 44 + s * 2);
  }
  return out;
}

/** Sum two WAV buffers sample-wise (missing tail treated as silence) with a
 *  peak guard: if |sum| exceeds 0.999 the whole mix is scaled down to fit.
 *  Both inputs must share sample rate and channel count. Returns 16-bit PCM. */
function mixWavBuffers(a: Buffer, b: Buffer): Buffer {
  const wa = parseWavToFloat(a);
  const wb = parseWavToFloat(b);
  if (wa.sampleRate !== wb.sampleRate) {
    throw new Error(`Sample rate mismatch (${wa.sampleRate} vs ${wb.sampleRate})`);
  }
  if (wa.numChannels !== wb.numChannels) {
    throw new Error(`Channel count mismatch (${wa.numChannels} vs ${wb.numChannels})`);
  }

  const n = Math.max(wa.samples.length, wb.samples.length);
  const mixed = new Float32Array(n);
  let peak = 0;
  for (let s = 0; s < n; s++) {
    const v = (s < wa.samples.length ? wa.samples[s] : 0)
            + (s < wb.samples.length ? wb.samples[s] : 0);
    mixed[s] = v;
    const av = Math.abs(v);
    if (av > peak) peak = av;
  }
  if (peak > 0.999) {
    const scale = 0.999 / peak;
    for (let s = 0; s < n; s++) mixed[s] *= scale;
  }
  return encodeWav16(mixed, wa.sampleRate, wa.numChannels);
}
