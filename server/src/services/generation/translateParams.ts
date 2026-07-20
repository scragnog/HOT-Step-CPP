// generation/translateParams.ts — Translate frontend params to AceRequest format
//
// Pure function with zero side effects. Maps UI-facing parameter names
// to the AceRequest schema expected by the ace-server engine.

import type { AceRequest } from '../../services/aceClient.js';
import { mapPath } from '../../services/pathMapper.js';
import { parseAdapterSections, stripAdapterDirectives } from './adapterSections.js';

/** Samples per compiled timestep gain curve (t ∈ [0,1], uniform). */
const GAIN_CURVE_SAMPLES = 33;

/**
 * Compile an active-timestep window into a gain curve g(t) sampled uniformly
 * over flow-matching t ∈ [0,1] (t=1 noise → t=0 clean).
 *
 * Edges use smoothstep ramps of width `soft` CENTERED on the window bounds
 * (g=0.5 exactly at start/end). Two adapters sharing a boundary — e.g. a
 * "structure" expert on [0.5, 1] and a "timbre" expert on [0, 0.5] — therefore
 * crossfade with gains summing to exactly 1 across the transition.
 */
function windowToGainCurve(start: number, end: number, soft = 0.1): number[] {
  const lo = Math.min(start, end);
  const hi = Math.max(start, end);
  const smooth = (x: number) => {
    const c = Math.max(0, Math.min(1, x));
    return c * c * (3 - 2 * c);
  };
  // Ramp centered on an edge: 0 at edge-soft/2, 0.5 at edge, 1 at edge+soft/2.
  const rise = (t: number, edge: number) =>
    soft > 0 ? smooth((t - (edge - soft / 2)) / soft) : (t >= edge ? 1 : 0);
  const curve: number[] = [];
  for (let i = 0; i < GAIN_CURVE_SAMPLES; i++) {
    const t = i / (GAIN_CURVE_SAMPLES - 1);
    // Window bounds at the domain ends need no ramp (nothing to fade to).
    const gLo = lo <= 0 ? 1 : rise(t, lo);
    const gHi = hi >= 1 ? 1 : 1 - rise(t, hi);
    curve.push(gLo * gHi);
  }
  return curve;
}

/** Translate frontend params to AceRequest format */
export function translateParams(params: any): AceRequest {
  const req: AceRequest = {
    caption: params.prompt || params.songDescription || params.caption || params.style || '',
  };

  // Lyrics / instrumental
  if (params.instrumental) {
    req.lyrics = '[Instrumental]';
  } else if (params.lyrics) {
    req.lyrics = params.lyrics;
  }

  // Metadata
  if (params.bpm) req.bpm = params.bpm;
  if (params.duration) {
    const buffer = (params.autoTrimEnabled && params.durationBuffer) ? params.durationBuffer : 0;
    req.duration = params.duration + buffer;
  }
  if (params.keyScale) req.keyscale = params.keyScale;
  if (params.timeSignature) {
    const ts = String(params.timeSignature);
    req.timesignature = ts.includes('/') ? ts.split('/')[0] : ts;
  }
  if (params.vocalLanguage) req.vocal_language = params.vocalLanguage;

  // Seed (DiT / generation phase)
  if (params.randomSeed) {
    req.seed = Math.floor(Math.random() * 2_147_483_647);
  } else if (params.seed !== undefined) {
    req.seed = params.seed;
  }

  // LM Seed — independent from the seed above, unless tied to it via
  // lmSeedFollowsDit (default true — matches the engine's original
  // behavior: locked seed -> both deterministic, random -> both random).
  // When tied, lm_seed is left unset entirely so the engine's own fallback
  // (lm_seed defaults to the DiT seed when absent) does the tying — this
  // correctly follows a *randomized* seed too, since req.seed is already
  // resolved above by this point.
  const lmSeedFollowsDit = params.lmSeedFollowsDit !== false; // default true
  if (!lmSeedFollowsDit && params.lmSeed !== undefined) {
    req.lm_seed = params.lmSeed;
  }

  // Batch
  if (params.batchSize) req.lm_batch_size = params.batchSize;

  // LM params
  if (params.lmTemperature !== undefined) req.lm_temperature = params.lmTemperature;
  if (params.lmCfgScale !== undefined) req.lm_cfg_scale = params.lmCfgScale;
  if (params.lmTopP !== undefined) req.lm_top_p = params.lmTopP;
  if (params.lmTopK !== undefined) req.lm_top_k = params.lmTopK;
  if (params.negative_prompt) req.negative_prompt = params.negative_prompt;
  if (params.lmNegativePrompt) req.lm_negative_prompt = params.lmNegativePrompt;

  // DiT params
  if (params.inferenceSteps) req.inference_steps = params.inferenceSteps;
  if (params.guidanceScale !== undefined) req.guidance_scale = params.guidanceScale;
  if (params.shift !== undefined) req.shift = params.shift;
  if (params.inferMethod) req.infer_method = params.inferMethod;
  if (params.scheduler) req.scheduler = params.scheduler;
  if (params.guidanceMode) req.guidance_mode = params.guidanceMode;

  // Cover/repaint
  if (params.taskType) req.task_type = params.taskType;
  if (params.audioCoverStrength !== undefined) req.audio_cover_strength = params.audioCoverStrength;
  if (params.coverNoiseStrength !== undefined) req.cover_noise_strength = params.coverNoiseStrength;
  if (params.coverNoiseMethod) req.cover_noise_method = params.coverNoiseMethod;
  if (params.repaintingStart !== undefined) req.repainting_start = params.repaintingStart;
  if (params.repaintingEnd !== undefined) req.repainting_end = params.repaintingEnd;
  if (params.seedStrength !== undefined) req.seed_strength = params.seedStrength;
  if (params.evictLm) req.evict_lm = true;
  if (params.vaeChunk) req.vae_chunk = params.vaeChunk;
  if (params.batchCfg !== undefined) req.batch_cfg = params.batchCfg ? 1 : 0;
  if (params.trackName) req.track = params.trackName;

  // CoT
  if (params.useCotCaption !== undefined) req.use_cot_caption = params.useCotCaption;

  // Model routing
  if (params.ditModel) req.synth_model = params.ditModel;
  if (params.lmModel) req.lm_model = params.lmModel;
  if (params.vaeModel) req.vae_model = params.vaeModel;
  if (params.embeddingModel) req.emb_model = params.embeddingModel;
  if (params.loraPath) req.adapter = mapPath(params.loraPath);
  if (params.loraScale !== undefined) req.adapter_scale = params.loraScale;
  // Multi-adapter stack: when the UI supplies a list, it supersedes the single
  // adapter — each entry is mapped to an engine path and carries its own scale.
  // The engine merges them (or sums runtime deltas) with per-adapter scaling.
  if (Array.isArray(params.loraStack) && params.loraStack.length > 0) {
    req.adapters = params.loraStack
      .filter((a: { path?: string }) => a && a.path)
      .map((a: { path: string; scale?: number; stepStart?: number; stepEnd?: number; stepSoft?: number; gainCurve?: number[] }) => {
        const entry: { name: string; scale: number; gain_curve?: number[] } = {
          name: mapPath(a.path) as string,
          scale: a.scale ?? 1.0,
        };
        // Timestep-dependent gain (interval experts / MoE mixing): an explicit
        // curve wins; otherwise an active-t window [stepStart, stepEnd] compiles
        // to one. A full-range window (0..1) means "always on" — no curve.
        if (Array.isArray(a.gainCurve) && a.gainCurve.length > 0) {
          entry.gain_curve = a.gainCurve.map((g) => Math.max(0, Number(g) || 0));
        } else {
          const s = a.stepStart ?? 0;
          const e = a.stepEnd ?? 1;
          if (s > 0 || e < 1) {
            entry.gain_curve = windowToGainCurve(s, e, a.stepSoft ?? 0.1);
          }
        }
        return entry;
      });
  }
  if (params.adapterGroupScales) req.adapter_group_scales = params.adapterGroupScales;
  if (params.adapterMode) req.adapter_mode = params.adapterMode;
  if (params.adapterRuntimeQuant) req.adapter_runtime_quant = params.adapterRuntimeQuant;
  if (params.adapterMergeLowVram) req.adapter_merge_lowvram = true;

  // Per-section adapter masking (regional LoRA): parse inline [Section]{k=v} directives
  // from the lyrics into a per-section weight table, strip them from the lyrics sent to
  // the engine, and force runtime mode (merge can't vary per-frame). Only applied with a
  // 2+ adapter stack; a no-directive lyric leaves everything untouched.
  if (Array.isArray(params.loraStack) && params.loraStack.length >= 2 && req.lyrics) {
    const parsed = parseAdapterSections(
      req.lyrics,
      params.loraStack,
      params.adapterStackMode || 'blend',
      params.adapterStackBudget ?? 0.75,
    );
    if (parsed.sections && parsed.sections.length > 0) {
      req.lyrics = parsed.lyrics;
      req.adapter_sections = parsed.sections;
      req.adapter_mode = 'runtime';
      if (params.adapterSectionAlignAt !== undefined) req.adapter_section_align_at = params.adapterSectionAlignAt;
      if (params.adapterSectionIsolation !== undefined) req.adapter_section_isolation = params.adapterSectionIsolation;
    }
  } else if (req.lyrics) {
    // Gate not met (0–1 adapters / Simple mode): directives can't apply, but they
    // must STILL be stripped — otherwise `[Verse]{x=0.9}` reaches the LM/encoder
    // as garbage tokens.
    req.lyrics = stripAdapterDirectives(req.lyrics);
  }
  // Timestep-dependent adapter gating (interval experts / MoE mixing) rides the
  // engine's per-section mask machinery. When any stacked adapter carries a gain
  // curve but lyric directives produced no sections, synthesize a single
  // whole-song section carrying the stack scales, and force runtime mode (merge
  // bakes weights once; gains vary per step). P2 alignment stays off naturally:
  // it requires a section token map, which only directive parsing builds.
  if (
    Array.isArray(req.adapters) &&
    req.adapters.some((a) => a.gain_curve && a.gain_curve.length > 0) &&
    (!req.adapter_sections || req.adapter_sections.length === 0)
  ) {
    req.adapter_sections = [{ weights: req.adapters.map((a) => a.scale), size: 1 }];
    req.adapter_mode = 'runtime';
  }
  // Basin re-base: rebaseSource is a DiT model NAME (engine resolves to its path).
  // Only meaningful alongside an adapter; engine ignores it otherwise.
  if (params.loraPath && params.rebaseSource && params.rebaseBeta) {
    req.rebase_source = params.rebaseSource;
    req.rebase_beta = params.rebaseBeta;
  }

  // Trigger word(s): every loaded adapter contributes its trigger. triggerWords
  // is the full list; fall back to the single triggerWord for older callers.
  const triggerWords: string[] = (Array.isArray(params.triggerWords) && params.triggerWords.length)
    ? params.triggerWords
    : (params.triggerWord ? [params.triggerWord] : []);
  if (triggerWords.length && params.triggerPlacement && params.loraPath) {
    const tw = triggerWords.join(', ');
    const caption = req.caption || '';
    switch (params.triggerPlacement) {
      case 'prepend': req.caption = caption ? `${tw}, ${caption}` : tw; break;
      case 'append':  req.caption = caption ? `${caption}, ${tw}` : tw; break;
      case 'replace': req.caption = tw; break;
    }
  }

  // Solver sub-parameters
  if (params.storkSubsteps !== undefined) req.stork_substeps = params.storkSubsteps;
  if (params.beatStability !== undefined) req.beat_stability = params.beatStability;
  if (params.frequencyDamping !== undefined) req.frequency_damping = params.frequencyDamping;
  if (params.temporalSmoothing !== undefined) req.temporal_smoothing = params.temporalSmoothing;

  // Guidance sub-parameters
  if (params.apgMomentum !== undefined) req.apg_momentum = params.apgMomentum;
  if (params.apgNormThreshold !== undefined) req.apg_norm_threshold = params.apgNormThreshold;

  // DCW
  if (params.dcwEnabled !== undefined) req.dcw_enabled = params.dcwEnabled;
  if (params.dcwMode) req.dcw_mode = params.dcwMode;
  if (params.dcwScaler !== undefined) req.dcw_scaler = params.dcwScaler;
  if (params.dcwHighScaler !== undefined) req.dcw_high_scaler = params.dcwHighScaler;

  // Latent post-processing
  if (params.latentShift !== undefined) req.latent_shift = params.latentShift;
  if (params.latentRescale !== undefined) req.latent_rescale = params.latentRescale;
  if (params.customTimesteps) req.custom_timesteps = params.customTimesteps;
  if (params.cfgCutoffRatio !== undefined) req.cfg_cutoff_ratio = params.cfgCutoffRatio;
  if (params.lmCfgCutoffRatio !== undefined) req.lm_cfg_cutoff_ratio = params.lmCfgCutoffRatio;
  if (params.cacheRatio !== undefined) req.cache_ratio = params.cacheRatio;

  // Post-VAE spectral denoiser
  if (params.denoiseStrength !== undefined) req.denoise_strength = params.denoiseStrength;
  if (params.denoiseSmoothing !== undefined) req.denoise_smoothing = params.denoiseSmoothing;
  if (params.denoiseMix !== undefined) req.denoise_mix = params.denoiseMix;

  // LSS: Latent Spectral Suppressor (pre-VAE latent channel gate)
  if (params.lssStrength !== undefined) req.lss_strength = params.lssStrength;
  if (params.lssVarThresh !== undefined) req.lss_var_thresh = params.lssVarThresh;
  if (params.lssDcRemove !== undefined) req.lss_dc_remove = params.lssDcRemove;

  // Lua plugin dynamic params (passthrough from UI)
  if (params.pluginParams && Object.keys(params.pluginParams).length > 0) {
    req.plugin_params = params.pluginParams;
  }

  // Postprocess plugin (replaces built-in VAE tiled decoder with Lua plugin)
  if (params.postprocessPlugin) {
    req.postprocess_plugin = params.postprocessPlugin;
  }

  // VAE backend selection (ONNX Runtime / TensorRT)
  if (params.useOrtVae) req.use_ort_vae = true;

  // Streaming pipeline (DEMON-style ring buffer)
  if (params.streamMode) req.stream_mode = true;
  if (params.streamDepth !== undefined) req.stream_depth = params.streamDepth;
  if (params.streamChunkDir) req.stream_chunk_dir = params.streamChunkDir;

  return req;
}
