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
/** Fired once per track when Whisper stem-mode is active: stemPath is a temp
 *  WAV of the isolated vocal stem, or null when no stem could be produced
 *  (caller should fall back to full-mix transcription). The caller owns
 *  deleting the temp file. */
type VocalStemFn = (trackIdx: number, stemPath: string | null) => void;

// StableStep GGML backend files — 4 GGUFs at the models dir root (the ONNX
// set lives in <models>/onnx/sa3 and is checked via sa3ModelsInstalled()).
// tokenizer.json in onnx/sa3 is required for BOTH backends (Node tokenizes).
// Keep in sync with SA3_GGUF_FILES in routes/models.ts.
const SA3_GGUF_FILES = [
  'sa3-dit-BF16.gguf',
  'sa3-same-enc-F16.gguf',
  'sa3-same-dec-F16.gguf',
  'sa3-text-enc-BF16.gguf',
];

/** True if the SA3 GGML backend appears installed (4 root GGUFs + tokenizer). */
function sa3GgufInstalled(): boolean {
  const modelsDir = config.aceServer.models;
  return fs.existsSync(path.join(modelsDir, 'onnx', 'sa3', 'tokenizer.json'))
      && SA3_GGUF_FILES.every(f => fs.existsSync(path.join(modelsDir, f)));
}

// BS-Roformer-Leap "Xe" pair (huggingface.co/pcunwa/BS-Roformer-Leap). Both are
// single-stem models differing only in target_instrument, and both run through
// the engine's native GGML BS-RoFormer (bs-roformer-ggml.h), not ONNX Runtime.
// Keep in sync with BS_MODEL_LEAP_XE_* in engine/src/supersep.cpp.
const LEAP_XE_FILES = ['bs_leap_xe_voc-F32.gguf', 'bs_leap_xe_inst-F32.gguf'];

/** True if both Leap Xe checkpoints are present, enabling SUPERSEP_STABLESTEP. */
function leapXeInstalled(): boolean {
  const modelsDir = config.aceServer.models;
  return LEAP_XE_FILES.every(f => fs.existsSync(path.join(modelsDir, 'supersep', f)));
}

// SuperSepLevel values from engine/src/supersep.h.
const SEP_LEVEL_VOCALS_ONLY = 4;  // single 6-stem pass, instrumental = mix − vocals
const SEP_LEVEL_STABLESTEP  = 5;  // dual Leap Xe pass, both stems neural

// SA3's native sample rate (engine/src/sa3-refine.h: SA3_SR 44100). A source
// already at this rate needs no resampling anywhere in the StableStep path.
const SA3_NATIVE_SR = 44100;

export interface PostProcessParams {
  postProcessingEnabled?: boolean;
  ppVaeReencode?: boolean;
  ppVaeBlend?: number;
  ppVaeUseOnnx?: boolean;
  // StableStep — SA3 (Stable Audio 3) SDEdit refine of the instrumental
  stableStepOn?: boolean;
  stableStep?: boolean;          // preset/settings-file alias for stableStepOn
  stableStepStrength?: number;   // 0..1 init noise level (default 0.3)
  stableStepSteps?: number;      // sampler steps (engine default 8, clamped 1..64)
  /** Engine backend for the SA3 refine: 'onnx' (ONNX Runtime/TensorRT),
   *  'gguf' (GGML — CUDA/Vulkan/CPU) or 'auto' (engine picks, default). */
  stableStepBackend?: 'auto' | 'onnx' | 'gguf';
  /** Lua plugin routing for the SA3 sampler loop — same registry as the ACE
   *  Generation dropdowns. Absent = the original pingpong/euler path.
   *  Naming a solver replaces the pingpong renoise; SA3 has no CFG uncond, so
   *  APG-family guidance is a pass-through (engine/src/sa3-refine.h). */
  stableStepSolver?: string;
  stableStepScheduler?: string;
  stableStepGuidanceMode?: string;
  stableStepGuidanceScale?: number;
  stableStepPluginParams?: Record<string, string | number>;
  /** StableStep DoRA adapters (models/sa3-adapters/<name>.gguf) merged into
   *  the SA3 DiT at load with per-adapter strength. Forces the GGUF backend. */
  stableStepAdapters?: Array<{ name: string; scale: number }>;
  /** Re-encode the isolated vocal stem through the PP-VAE before recombining.
   *  Default OFF — the PP-VAE is a lossy 60:1 autoencoder round trip: measured
   *  against a resample-only control it costs ~2 dB across the midrange rising
   *  to ~6 dB above 16 kHz, halves the energy above 10 kHz, and drops
   *  input/output coherence below 0.1 above 4 kHz (i.e. the top octaves are
   *  resynthesised, not reproduced). It does smooth AS1.5 vocal fizz, so it
   *  stays available — but opt-in. Off = the original AS1.5 vocal stem is
   *  recombined with the SA3-refined instrumental untouched. */
  stableStepVocalPpVae?: boolean;
  /** Preserve source dynamics: engine-side windowed envelope match of the
   *  refined audio to the pre-refine source (counters mastered-density
   *  "loudness war" character from adapters trained on commercial masters). */
  stableStepPreserveDynamics?: boolean;
  /** Source blending: 'off' | 'crossover' (source lows + refined highs at a
   *  spectral crossover) | 'mix' (full-band wet/dry). */
  stableStepBlendMode?: 'off' | 'crossover' | 'mix';
  stableStepCrossoverHz?: number;      // crossover center (default 250)
  stableStepCrossoverWidthHz?: number; // transition width (default 200)
  stableStepMix?: number;              // 0 = pure source .. 1 = pure refined
  /** SA3 refine RNG seed. Undefined = engine picks a random seed per refine.
   *  Populated by the generate route: follows the resolved generation seed by
   *  default, or a fixed user override (stableStepSeedFollowsDit=false). */
  stableStepSeed?: number;
  /** Per-track captions (parallel to audioUrls) used to build the SA3 prompt.
   *  Populated by the generate route from the LM results. */
  stableStepCaptions?: string[];
  /** Whisper "Isolate vocals first" toggle — when set (and onVocalStem is
   *  provided), a SuperSep split runs even if StableStep won't consume it. */
  whisperIsolateVocals?: boolean;
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

// ── Shared vocal separation ─────────────────────────────────────────────────

/** Result of a VOCALS_ONLY SuperSep split (stems are 44.1 kHz WAVs). */
interface VocalSeparation {
  sepId: string;
  stems: Array<{ index: number; category: string; hidden: boolean }>;
  vocalIndex: number;
  vocalBuf: Buffer;
  instBuf: Buffer;
}

/** Run a 2-stem (Vocals + Instrumental) split via the engine's SuperSep API and
 *  fetch both stems. Returns null when the split produced no usable
 *  vocal/instrumental pair (silent stems are dropped engine-side). Throws on
 *  separation failure/timeout.
 *
 *  `level` picks the separation strategy — see the SuperSepLevel constants. */
async function separateVocals(
  srcBuf: Buffer,
  level: number = SEP_LEVEL_VOCALS_ONLY,
): Promise<VocalSeparation | null> {
  const sepId = await aceClient.submitSuperSepSeparate(srcBuf, level);

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
  const vocalStem = stems.find(s => s.category === 'vocals' && !s.hidden);
  const instStem = stems.find(s => s.category === 'instruments' && !s.hidden);
  if (!vocalStem || !instStem) return null;

  const instBuf = await aceClient.superSepStem(sepId, instStem.index);
  const vocalBuf = await aceClient.superSepStem(sepId, vocalStem.index);
  return { sepId, stems, vocalIndex: vocalStem.index, vocalBuf, instBuf };
}

/** Run the full post-processing chain on a list of audio files.
 *  onVocalStem (optional): requests the isolated vocal stem for Whisper
 *  transcription — see VocalStemFn. Fired per track as soon as the shared
 *  SuperSep split completes, so CPU transcription overlaps later GPU stages. */
export async function runPostProcessingChain(
  audioUrls: string[],
  params: PostProcessParams,
  totalTracks: number,
  jobId: string,
  log: LogFn,
  setStage: StageFn,
  onVocalStem?: VocalStemFn
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

    // ── Shared vocal separation (StableStep stem workflow + Whisper isolation) ──
    // 2-stem vocal/instrumental split, run at most ONCE per track. Consumers:
    //   - StableStep (vocal gens): SA3-refines the instrumental stem and
    //     recombines it with the vocal stem.
    //   - Whisper isolation (onVocalStem): the vocal stem is written to a temp
    //     WAV and handed to the caller BEFORE the SA3 refine so CPU
    //     transcription overlaps the GPU work.
    // Deliberately NOT gated on ppMasterOn: the split is a service for
    // Whisper, not an audio-modifying PP stage (anyStageRan untouched).
    const sa3Available = sa3ModelsInstalled() || sa3GgufInstalled();
    // The split is NOT optional for a vocal track: SA3 refines instrumental
    // only, so handing it a full mix makes it hallucinate replacement vocals.
    // It is a correctness requirement, not a quality nicety.
    const stableStepWantsStems = stableStepOn && sa3Available && !params.instrumental;
    const whisperWantsStems = !!onVocalStem && !!params.whisperIsolateVocals && !params.instrumental;
    let vocalSep: VocalSeparation | null = null;

    if (stableStepWantsStems || whisperWantsStems) {
      const sepStart = performance.now();
      // StableStep gets the dual Leap Xe pass when both checkpoints are
      // installed: the vocal that is re-applied after the refine comes from the
      // vocal-target model and the instrumental fed to SA3 comes from the
      // instrumental-target model, so neither stem is a mix-minus residual.
      // Whisper-only splits stay on the single 6-stem pass — one model load
      // instead of two, and transcription does not need that precision.
      const useLeap = stableStepWantsStems && leapXeInstalled();
      const sepLevel = useLeap ? SEP_LEVEL_STABLESTEP : SEP_LEVEL_VOCALS_ONLY;
      if (stableStepWantsStems && !useLeap) {
        log('INFO', '[SuperSep] Leap Xe models not installed — using the 6-stem '
          + 'BS-RoFormer pass (instrumental derived as mix − vocals)');
      }
      setStage(`Separating vocals${totalTracks > 1 ? ` (${i+1}/${totalTracks})` : ''}...`);

      // A previous track's refine can leave the SA3 DiT resident (~2.8 GB),
      // while BS-RoFormer wants ~2.6 GB CONTIGUOUS for its attention matrix.
      // Fragmentation, not total free VRAM, is what fails there — so evict SA3
      // before the split rather than after an allocation failure. It reloads
      // on demand when the refine runs a few lines below. Best-effort by
      // design: in_use modules are skipped and any error here is non-fatal,
      // because the separation may well succeed without the eviction.
      if (stableStepWantsStems) {
        try {
          const loaded = await aceClient.listLoadedModels();
          for (const m of loaded.filter(x => x.label.toLowerCase().includes('sa3') && !x.in_use)) {
            if (await aceClient.unloadLabel(m.label)) {
              log('INFO', `[StableStep] Evicted ${m.label} (${m.mb.toFixed(0)} MB) before vocal separation`);
            }
          }
        } catch (evErr: any) {
          log('DEBUG', `[StableStep] SA3 eviction before separation failed (non-fatal): ${evErr?.message ?? evErr}`);
        }
      }

      try {
        log('INFO', `[SuperSep] Vocal split via level ${sepLevel}`
          + `${useLeap ? ' (dual BS-Roformer-Leap Xe)' : ' (BS-RoFormer 6-stem)'}`);
        vocalSep = await separateVocals(fs.readFileSync(processedPath), sepLevel);
        if (!vocalSep) {
          log('INFO', '[SuperSep] No vocal/instrumental split (no vocal energy detected)');
        }
      } catch (sepErr: any) {
        log('WARNING', `[SuperSep] Vocal separation failed (non-fatal): ${sepErr.message}`);
      }
      timing.push({ name: 'Vocal Separation', ms: Math.round(performance.now() - sepStart) });
    }

    // Hand the vocal stem to the Whisper callback (null = fall back to full mix)
    if (onVocalStem) {
      let stemPath: string | null = null;
      if (vocalSep) {
        try {
          stemPath = processedPath + '.vocalstem.tmp.wav';
          fs.writeFileSync(stemPath, vocalSep.vocalBuf);
        } catch (stemErr: any) {
          log('WARNING', `[SuperSep] Failed to write vocal stem for Whisper: ${stemErr.message}`);
          stemPath = null;
        }
      }
      onVocalStem(i, stemPath);
    }

    // ── StableStep: SA3 SDEdit refine (before PP-VAE) ──
    // Instrumental gens: refine the whole mix through the SA3 model.
    // Vocal gens: consume the shared split above — SA3-refine the
    // instrumental, optionally PP-VAE the vocals (off by default, see
    // stableStepVocalPpVae), then recombine sample-wise in Node.
    if (stableStepOn) {
      const ssStart = performance.now();
      try {
        if (!sa3Available) {
          log('WARNING', '[StableStep] SA3 models not installed (neither models/onnx/sa3 nor root GGUFs) — skipping');
        } else {
          // Engine backend: 'onnx' | 'gguf' forces one; undefined = engine auto.
          const backend = (params.stableStepBackend === 'onnx' || params.stableStepBackend === 'gguf')
            ? params.stableStepBackend : undefined;
          const adapters = (params.stableStepAdapters ?? []).filter(a => a && a.name && a.scale !== 0);
          const envMatch = params.stableStepPreserveDynamics !== false;
          const blendMode = params.stableStepBlendMode ?? 'off';
          const blendOpts = blendMode === 'mix'
            ? { mix: Math.min(1, Math.max(0, params.stableStepMix ?? 1)) }
            : blendMode === 'crossover'
              ? { bandBlend: true, bandFreq: params.stableStepCrossoverHz ?? 250,
                  bandWidth: params.stableStepCrossoverWidthHz ?? 200 }
              : {};
          if (blendMode !== 'off') {
            log('INFO', `[StableStep] Source blend: ${JSON.stringify(blendOpts)}`);
          }
          if (adapters.length > 0) {
            log('INFO', `[StableStep] Adapters: ${adapters.map(a => `${a.name}@${a.scale}`).join(', ')}`);
          }
          const strength = params.stableStepStrength ?? 0.3;
          // Lua plugin routing + step count. Spread into every submitSa3Refine
          // call below so the three refine branches (whole-mix, no-split,
          // stem-split) cannot drift apart. Each field is omitted unless set,
          // which is what keeps the engine on its bit-identical default path.
          const sa3PluginOpts = {
            ...(params.stableStepSteps !== undefined ? { steps: params.stableStepSteps } : {}),
            ...(params.stableStepSolver ? { solver: params.stableStepSolver } : {}),
            ...(params.stableStepScheduler ? { scheduler: params.stableStepScheduler } : {}),
            ...(params.stableStepGuidanceMode ? { guidanceMode: params.stableStepGuidanceMode } : {}),
            ...(params.stableStepGuidanceScale && params.stableStepGuidanceScale > 1
              ? { guidanceScale: params.stableStepGuidanceScale } : {}),
            ...(params.stableStepPluginParams && Object.keys(params.stableStepPluginParams).length > 0
              ? { pluginParams: params.stableStepPluginParams } : {}),
          };
          if (sa3PluginOpts.solver || sa3PluginOpts.scheduler || sa3PluginOpts.guidanceMode) {
            log('INFO', `[StableStep] Plugins: solver=${sa3PluginOpts.solver ?? '(pingpong)'} `
              + `scheduler=${sa3PluginOpts.scheduler ?? '(sa3 logsnr)'} `
              + `guidance=${sa3PluginOpts.guidanceMode ?? '(none)'}`);
          }
          const ssSeed = params.stableStepSeed;
          if (ssSeed !== undefined) log('INFO', `[StableStep] Seed: ${ssSeed}`);
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
              tokens: ids, nTokens, strength, backend, adapters, envMatch, seed: ssSeed,
              ...blendOpts, ...sa3PluginOpts,
            });
            fs.writeFileSync(processedPath, refined);
          } else if (!vocalSep) {
            // No vocal/instrumental split available (no vocal energy detected,
            // or the separation failed) — refine the whole mix directly.
            log('INFO', '[StableStep] No vocal/instrumental split — refining full mix');
            setStage(`StableStep: refining instrumental${suffix}...`);
            const srcBuf = fs.readFileSync(processedPath);
            const refined = await aceClient.submitSa3Refine(srcBuf, {
              tokens: ids, nTokens, strength, backend, adapters, envMatch, seed: ssSeed,
              ...blendOpts, ...sa3PluginOpts,
            });
            fs.writeFileSync(processedPath, refined);
          } else {
            // Stems from the shared split above come back at 44.1 kHz, which is
            // also SA3's native rate (sa3-refine.h SA3_SR 44100).
            //
            // The 48 kHz round-trip below exists only because ACE's own output
            // is 48 kHz and the final mix has to match it. When the SOURCE is
            // already 44.1 kHz (MiniMax-Music3), every rate in this branch
            // agrees natively: SA3 in and out, both stems, and the mix. That
            // path needs no out_sr override and no vocal resample at all —
            // which also means it never touches PP-VAE, an ACE-only model.
            const vs = vocalSep;
            const srcRate = parseWavToFloat(fs.readFileSync(processedPath)).sampleRate;
            const nativeRate = srcRate === SA3_NATIVE_SR;
            if (nativeRate) {
              log('INFO', `[StableStep] Source is ${srcRate} Hz — SA3 native; `
                + 'stems, refine and mix all stay at that rate (no resample)');
            }

            setStage(`StableStep: refining instrumental${suffix}...`);
            const refinedInst = await aceClient.submitSa3Refine(vs.instBuf, {
              tokens: ids, nTokens, strength, backend, adapters, envMatch, seed: ssSeed,
              // Omitted on the native path: the engine defaults out_sr to the
              // input rate, so asking for anything would force a resample.
              ...(nativeRate ? {} : { outSr: 48000 }),
              ...blendOpts, ...sa3PluginOpts,
            });

            setStage(`StableStep: processing vocals${suffix}...`);

            // The raw stem is 44.1 kHz and has to reach 48 kHz to mix with the
            // refined instrumental. /pp-vae-reencode?blend=1.0 returns straight
            // after the engine's 48 kHz decode, before any VAE work (see
            // hot-step-server.cpp), so it is a pure resample that preserves
            // level to within 0.01 dB — measured, not assumed.
            const rawVocalsAt48k = async (): Promise<Buffer> => {
              try {
                return await aceClient.submitPpVaeReencode(vs.vocalBuf, 1.0);
              } catch (rErr: any) {
                // That endpoint 501s when no PP-VAE model is installed, even at
                // blend=1.0. Fall back to a solo-stem recombine — it resamples,
                // but ALSO peak-normalises to -1 dBFS (supersep_recombine in
                // supersep.cpp), which would push the vocal several dB above
                // the instrumental. Undo that by matching the source RMS.
                log('WARNING', `[StableStep] 48 kHz resample via PP-VAE passthrough failed `
                  + `(${rErr.message}) — falling back to stem recombine`);
                const recombined = await aceClient.superSepRecombine(vs.sepId,
                  vs.stems.map(s => ({ index: s.index, volume: 1.0, muted: s.index !== vs.vocalIndex })));
                return matchRms(recombined, vs.vocalBuf);
              }
            };

            let cleanVocals: Buffer;
            if (nativeRate) {
              // Both stems and the refined instrumental are already at the
              // source rate — the vocal stem goes back in untouched. PP-VAE
              // vocal cleanup is deliberately skipped here even when asked for:
              // it is an ACE VAE round-trip that would also force 48 kHz.
              if (params.stableStepVocalPpVae) {
                log('INFO', '[StableStep] Vocal PP-VAE skipped — ACE-only model, and the '
                  + 'native-rate path needs no resample');
              }
              cleanVocals = vs.vocalBuf;
            } else if (params.stableStepVocalPpVae) {
              try {
                cleanVocals = await aceClient.submitPpVaeReencode(vs.vocalBuf, 0.0); // 48 kHz out
              } catch (vErr: any) {
                log('WARNING', `[StableStep] Vocal PP-VAE failed, using raw vocal stem: ${vErr.message}`);
                cleanVocals = await rawVocalsAt48k();
              }
            } else {
              log('INFO', '[StableStep] Vocal PP-VAE off — remixing the original AS1.5 vocal stem (48 kHz resample only)');
              cleanVocals = await rawVocalsAt48k();
            }

            setStage(`StableStep: recombining${suffix}...`);
            const mixed = mixWavBuffers(refinedInst, cleanVocals);
            fs.writeFileSync(processedPath, mixed);
          }

          anyStageRan = true;
          log('INFO', `[StableStep] Refined ${audioFilename} (strength=${strength}, backend=${backend ?? 'auto'})`);
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
        const { normalizeLufs, formatLufsLog } = await import('./lufsNormalize.js');
        const result = normalizeLufs(processedPath, params.lufsTarget);
        anyStageRan = true;
        log('INFO', formatLufsLog(result, processedFilename));
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

/** Rescale `wav` so its RMS matches `reference`'s, capped so the result never
 *  exceeds the reference peak. Used to undo the -1 dBFS peak normalisation that
 *  supersep_recombine applies, which would otherwise change the vocal level
 *  relative to the instrumental. Sample rates may differ. */
function matchRms(wav: Buffer, reference: Buffer): Buffer {
  const w = parseWavToFloat(wav);
  const r = parseWavToFloat(reference);
  const rms = (s: Float32Array) => {
    let acc = 0;
    for (let i = 0; i < s.length; i++) acc += s[i] * s[i];
    return s.length ? Math.sqrt(acc / s.length) : 0;
  };
  const wRms = rms(w.samples);
  const rRms = rms(r.samples);
  if (wRms < 1e-8 || rRms < 1e-8) return wav;

  let gain = rRms / wRms;
  let wPeak = 0, rPeak = 0;
  for (let i = 0; i < w.samples.length; i++) wPeak = Math.max(wPeak, Math.abs(w.samples[i]));
  for (let i = 0; i < r.samples.length; i++) rPeak = Math.max(rPeak, Math.abs(r.samples[i]));
  if (wPeak * gain > rPeak + 0.01) gain = rPeak / (wPeak + 1e-8);

  const out = new Float32Array(w.samples.length);
  for (let i = 0; i < w.samples.length; i++) out[i] = w.samples[i] * gain;
  return encodeWav16(out, w.sampleRate, w.numChannels);
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
