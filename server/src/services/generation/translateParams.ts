// generation/translateParams.ts — Translate frontend params to AceRequest format
//
// Pure function with zero side effects. Maps UI-facing parameter names
// to the AceRequest schema expected by the ace-server engine.

import type { AceRequest } from '../../services/aceClient.js';
import { mapPath } from '../../services/pathMapper.js';

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

  // Seed
  if (params.randomSeed) {
    req.seed = Math.floor(Math.random() * 2_147_483_647);
  } else if (params.seed !== undefined) {
    req.seed = params.seed;
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
  if (params.adapterGroupScales) req.adapter_group_scales = params.adapterGroupScales;
  if (params.adapterMode) req.adapter_mode = params.adapterMode;

  // Trigger word
  if (params.triggerWord && params.triggerPlacement && params.loraPath) {
    const tw = params.triggerWord;
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
