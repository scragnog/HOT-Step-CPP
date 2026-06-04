// globalParamsStore.ts -- Zustand store replacing GlobalParamsContext
//
// Migrated from React Context + usePersistedState to Zustand with
// per-field localStorage keys.  Consumers use selectors:
//   const val = useGlobalParamsStore(s => s.fieldName);
//
// Uses the SAME hs-* localStorage keys -- zero migration needed.

import { create } from 'zustand';
import type { GenerationParams } from '../types';
import { DEFAULT_SETTINGS, type AppSettings } from '../components/settings/SettingsPanel';

// -- Types --

export interface AdapterGroupScales {
  self_attn: number;
  cross_attn: number;
  mlp: number;
  cond_embed: number;
}

// -- Per-key localStorage adapter --

function readKey<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v !== null ? JSON.parse(v) : fallback;
  } catch { return fallback; }
}

function writeKey<T>(key: string, value: T): void {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* full */ }
}

// -- Store --

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const useGlobalParamsStore = create<any>()((set, get) => ({
  // -- State (initialised from localStorage) --
  ditModel: readKey("hs-ditModel", ''),
  lmModel: readKey("hs-lmModel", ''),
  vaeModel: readKey("hs-vaeModel", ''),
  embeddingModel: readKey("hs-embeddingModel", ''),
  adapter: readKey("hs-adapter", ''),
  adapterScale: readKey("hs-adapterScale", 1.0),
  adapterMode: readKey("hs-adapterMode", 'runtime'),
  adapterGroupScales: readKey("hs-adapterGroupScales", {
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0, time_embed: 0.0, proj_in: 0.0,
  }),
  adapterFolder: readKey("hs-adapterFolder", ''),
  advancedAdapters: readKey("hs-advancedAdapters", false),
  adaptersOpen: readKey("hs-adaptersOpen", false),
  inferenceSteps: readKey("hs-inferenceSteps", 12),
  guidanceScale: readKey("hs-guidanceScale", 9.0),
  cfgCutoffRatio: readKey("hs-cfgCutoffRatio", 1.0),
  lmCfgCutoffRatio: readKey("hs-lmCfgCutoffRatio", 1.0),
  cacheRatio: readKey("hs-cacheRatio", 0),
  shift: readKey("hs-shift", 3.0),
  inferMethod: readKey("hs-inferMethod", 'euler'),
  scheduler: readKey("hs-scheduler", 'linear'),
  guidanceMode: readKey("hs-guidanceMode", 'apg'),
  seed: readKey("hs-seed", 42),
  randomSeed: readKey("hs-randomSeed", true),
  batchSize: readKey("hs-batchSize", 1),
  storkSubsteps: readKey("hs-storkSubsteps", 10),
  beatStability: readKey("hs-beatStability", 0.25),
  frequencyDamping: readKey("hs-frequencyDamping", 0.4),
  temporalSmoothing: readKey("hs-temporalSmoothing", 0.13),
  apgMomentum: readKey("hs-apgMomentum", 0.75),
  apgNormThreshold: readKey("hs-apgNormThreshold", 2.5),
  dcwEnabled: readKey("hs-dcwEnabled", false),
  dcwMode: readKey("hs-dcwMode", 'double'),
  dcwLowScaler: readKey("hs-dcwLowScaler", readKey("hs-dcwScaler", 0.2)),
  dcwHighScaler: readKey("hs-dcwHighScaler", 0.2),
  latentShift: readKey("hs-latentShift", 0.0),
  latentRescale: readKey("hs-latentRescale", 1.0),
  customTimesteps: readKey("hs-customTimesteps", ''),
  denoiseStrength: readKey("hs-denoiseStrength", 0.0),
  denoiseSmoothing: readKey("hs-denoiseSmoothing", 0.7),
  denoiseMix: readKey("hs-denoiseMix", 0.25),
  autoTrimEnabled: readKey("hs-autoTrimEnabled", false),
  durationBuffer: readKey("hs-durationBuffer", 15),
  autoTrimFadeMs: readKey("hs-autoTrimFadeMs", 2000),
  skipLm: readKey("hs-skipLm", false),
  skipLrc: readKey("hs-skipLrc", false),
  useCotCaption: readKey("hs-useCotCaption", true),
  lmTemperature: readKey("hs-lmTemperature", 0.8),
  lmCfgScale: readKey("hs-lmCfgScale", 2.2),
  lmTopK: readKey("hs-lmTopK", 0),
  lmTopP: readKey("hs-lmTopP", 0.92),
  lmNegativePrompt: readKey("hs-lmNegativePrompt", 'NO USER INPUT'),
  lmCodesStrength: readKey("hs-lmCodesStrength", 1.0),
  postProcessingEnabled: readKey("hs-postProcessingEnabled", true),
  spectralLifterEnabled: readKey("hs-spectralLifterEnabled", false),
  slDenoiseStrength: readKey("hs-slDenoiseStrength", 0.3),
  slNoiseFloor: readKey("hs-slNoiseFloor", 0.1),
  slHfMix: readKey("hs-slHfMix", 0.0),
  slTransientBoost: readKey("hs-slTransientBoost", 0.0),
  slShimmerReduction: readKey("hs-slShimmerReduction", 6.0),
  masteringEnabled: readKey("hs-masteringEnabled", false),
  masteringReference: readKey("hs-masteringReference", ''),
  timbreReference: readKey("hs-timbreReference", false),
  timbreAudioPath: readKey("hs-timbreAudioPath", ''),
  vocalNaturalizerEnabled: readKey("hs-vocalNaturalizerEnabled", false),
  gainOffsetDb: readKey("hs-gainOffsetDb", 0),
  naturalizeAmount: readKey("hs-naturalizeAmount", 0.5),
  natVibratoRate: readKey("hs-natVibratoRate", 4.5),
  natVibratoDepth: readKey("hs-natVibratoDepth", 1.0),
  natFormantStrength: readKey("hs-natFormantStrength", 1.0),
  natMetallicReduction: readKey("hs-natMetallicReduction", 1.0),
  natQuantizationMask: readKey("hs-natQuantizationMask", 0.0),
  natTransitionSmooth: readKey("hs-natTransitionSmooth", 1.0),
  ppVaeReencode: readKey("hs-ppVaeReencode", false),
  ppVaeBlend: readKey("hs-ppVaeBlend", 0.0),
  ppVaeUseOnnx: readKey("hs-ppVaeUseOnnx", true),
  coverArtEnabled: readKey("hs-coverArtEnabled", false),
  coverArtSubject: readKey("hs-coverArtSubject", ''),
  qualityEvalEnabled: readKey("hs-qualityEvalEnabled", false),
  qualityEvalTarget: readKey("hs-qualityEvalTarget", 'unmastered'),

  // Dynamic Lua plugin params
  pluginParams: readKey('hs-pluginParams', {} as Record<string, string>),

  // Whisper Lyrics Transcription
  whisperLyricsEnabled: readKey("hs-whisperLyricsEnabled", false),
  whisperModel: readKey("hs-whisperModel", ''),
  whisperLanguage: readKey("hs-whisperLang", 'auto'),
  whisperBeamSize: readKey("hs-whisperBeam", 5),
  whisperIsolateVocals: readKey("hs-whisperIsolate", false),

  // Postprocess plugin (replaces built-in VAE tiled decoder)
  postprocessEnabled: readKey('hs-postprocessEnabled', false),
  postprocessPlugin: readKey('hs-postprocessPlugin', ''),

  // VAE backend selection (ONNX Runtime / TensorRT)
  useOrtVae: readKey('hs-useOrtVae', false),

  // LUFS Normalization
  lufsEnabled: readKey('hs-lufsEnabled', false),
  lufsPreset: readKey('hs-lufsPreset', 'spotify'),
  lufsTarget: readKey('hs-lufsTarget', -14),

  // -- Actions --
  setDitModel: (v: any) => { set({ ditModel: v }); writeKey("hs-ditModel", v); },
  setLmModel: (v: any) => { set({ lmModel: v }); writeKey("hs-lmModel", v); },
  setVaeModel: (v: any) => {
    set({ vaeModel: v });
    writeKey("hs-vaeModel", v);
    // Auto-detect ORT backend from file extension
    const isOnnx = /\.onnx$/i.test(v || '');
    set({ useOrtVae: isOnnx });
    writeKey('hs-useOrtVae', isOnnx);
  },
  setEmbeddingModel: (v: any) => { set({ embeddingModel: v }); writeKey("hs-embeddingModel", v); },
  setAdapter: (v: any) => { set({ adapter: v }); writeKey("hs-adapter", v); },
  setAdapterScale: (v: any) => { set({ adapterScale: v }); writeKey("hs-adapterScale", v); },
  setAdapterMode: (v: any) => { set({ adapterMode: v }); writeKey("hs-adapterMode", v); },
  setAdapterGroupScales: (v: any) => { set({ adapterGroupScales: v }); writeKey("hs-adapterGroupScales", v); },
  setAdapterFolder: (v: any) => { set({ adapterFolder: v }); writeKey("hs-adapterFolder", v); },
  setAdvancedAdapters: (v: any) => { set({ advancedAdapters: v }); writeKey("hs-advancedAdapters", v); },
  setAdaptersOpen: (v: any) => { set({ adaptersOpen: v }); writeKey("hs-adaptersOpen", v); },
  setInferenceSteps: (v: any) => { set({ inferenceSteps: v }); writeKey("hs-inferenceSteps", v); },
  setGuidanceScale: (v: any) => { set({ guidanceScale: v }); writeKey("hs-guidanceScale", v); },
  setCfgCutoffRatio: (v: any) => { set({ cfgCutoffRatio: v }); writeKey("hs-cfgCutoffRatio", v); },
  setLmCfgCutoffRatio: (v: any) => { set({ lmCfgCutoffRatio: v }); writeKey("hs-lmCfgCutoffRatio", v); },
  setCacheRatio: (v: any) => { set({ cacheRatio: v }); writeKey("hs-cacheRatio", v); },
  setShift: (v: any) => { set({ shift: v }); writeKey("hs-shift", v); },
  setInferMethod: (v: any) => { set({ inferMethod: v }); writeKey("hs-inferMethod", v); },
  setScheduler: (v: any) => { set({ scheduler: v }); writeKey("hs-scheduler", v); },
  setGuidanceMode: (v: any) => { set({ guidanceMode: v }); writeKey("hs-guidanceMode", v); },
  setSeed: (v: any) => { set({ seed: v }); writeKey("hs-seed", v); },
  setRandomSeed: (v: any) => {
    set({ randomSeed: v }); writeKey("hs-randomSeed", v);
    // When disabling random, snap seed away from -1 (the random sentinel)
    if (!v && get().seed === -1) {
      set({ seed: 42 }); writeKey("hs-seed", 42);
    }
  },
  setBatchSize: (v: any) => { set({ batchSize: v }); writeKey("hs-batchSize", v); },
  setStorkSubsteps: (v: any) => { set({ storkSubsteps: v }); writeKey("hs-storkSubsteps", v); },
  setBeatStability: (v: any) => { set({ beatStability: v }); writeKey("hs-beatStability", v); },
  setFrequencyDamping: (v: any) => { set({ frequencyDamping: v }); writeKey("hs-frequencyDamping", v); },
  setTemporalSmoothing: (v: any) => { set({ temporalSmoothing: v }); writeKey("hs-temporalSmoothing", v); },
  setApgMomentum: (v: any) => { set({ apgMomentum: v }); writeKey("hs-apgMomentum", v); },
  setApgNormThreshold: (v: any) => { set({ apgNormThreshold: v }); writeKey("hs-apgNormThreshold", v); },
  setDcwEnabled: (v: any) => { set({ dcwEnabled: v }); writeKey("hs-dcwEnabled", v); },
  setDcwMode: (v: any) => { set({ dcwMode: v }); writeKey("hs-dcwMode", v); },
  setDcwLowScaler: (v: any) => { set({ dcwLowScaler: v }); writeKey("hs-dcwLowScaler", v); },
  setDcwHighScaler: (v: any) => { set({ dcwHighScaler: v }); writeKey("hs-dcwHighScaler", v); },
  setLatentShift: (v: any) => { set({ latentShift: v }); writeKey("hs-latentShift", v); },
  setLatentRescale: (v: any) => { set({ latentRescale: v }); writeKey("hs-latentRescale", v); },
  setCustomTimesteps: (v: any) => { set({ customTimesteps: v }); writeKey("hs-customTimesteps", v); },
  setDenoiseStrength: (v: any) => { set({ denoiseStrength: v }); writeKey("hs-denoiseStrength", v); },
  setDenoiseSmoothing: (v: any) => { set({ denoiseSmoothing: v }); writeKey("hs-denoiseSmoothing", v); },
  setDenoiseMix: (v: any) => { set({ denoiseMix: v }); writeKey("hs-denoiseMix", v); },
  setAutoTrimEnabled: (v: any) => { set({ autoTrimEnabled: v }); writeKey("hs-autoTrimEnabled", v); },
  setDurationBuffer: (v: any) => { set({ durationBuffer: v }); writeKey("hs-durationBuffer", v); },
  setAutoTrimFadeMs: (v: any) => { set({ autoTrimFadeMs: v }); writeKey("hs-autoTrimFadeMs", v); },
  setSkipLm: (v: any) => { set({ skipLm: v }); writeKey("hs-skipLm", v); },
  setSkipLrc: (v: any) => { set({ skipLrc: v }); writeKey("hs-skipLrc", v); },
  setUseCotCaption: (v: any) => { set({ useCotCaption: v }); writeKey("hs-useCotCaption", v); },
  setLmTemperature: (v: any) => { set({ lmTemperature: v }); writeKey("hs-lmTemperature", v); },
  setLmCfgScale: (v: any) => { set({ lmCfgScale: v }); writeKey("hs-lmCfgScale", v); },
  setLmTopK: (v: any) => { set({ lmTopK: v }); writeKey("hs-lmTopK", v); },
  setLmTopP: (v: any) => { set({ lmTopP: v }); writeKey("hs-lmTopP", v); },
  setLmNegativePrompt: (v: any) => { set({ lmNegativePrompt: v }); writeKey("hs-lmNegativePrompt", v); },
  setLmCodesStrength: (v: any) => { set({ lmCodesStrength: v }); writeKey("hs-lmCodesStrength", v); },
  setPostProcessingEnabled: (v: any) => { set({ postProcessingEnabled: v }); writeKey("hs-postProcessingEnabled", v); },
  setSpectralLifterEnabled: (v: any) => { set({ spectralLifterEnabled: v }); writeKey("hs-spectralLifterEnabled", v); },
  setSlDenoiseStrength: (v: any) => { set({ slDenoiseStrength: v }); writeKey("hs-slDenoiseStrength", v); },
  setSlNoiseFloor: (v: any) => { set({ slNoiseFloor: v }); writeKey("hs-slNoiseFloor", v); },
  setSlHfMix: (v: any) => { set({ slHfMix: v }); writeKey("hs-slHfMix", v); },
  setSlTransientBoost: (v: any) => { set({ slTransientBoost: v }); writeKey("hs-slTransientBoost", v); },
  setSlShimmerReduction: (v: any) => { set({ slShimmerReduction: v }); writeKey("hs-slShimmerReduction", v); },
  setMasteringEnabled: (v: any) => { set({ masteringEnabled: v }); writeKey("hs-masteringEnabled", v); },
  setMasteringReference: (v: any) => { set({ masteringReference: v }); writeKey("hs-masteringReference", v); },
  setTimbreReference: (v: any) => { set({ timbreReference: v }); writeKey("hs-timbreReference", v); },
  setTimbreAudioPath: (v: any) => { set({ timbreAudioPath: v }); writeKey("hs-timbreAudioPath", v); },
  setVocalNaturalizerEnabled: (v: any) => { set({ vocalNaturalizerEnabled: v }); writeKey("hs-vocalNaturalizerEnabled", v); },
  setGainOffsetDb: (v: any) => { set({ gainOffsetDb: v }); writeKey("hs-gainOffsetDb", v); },
  setNaturalizeAmount: (v: any) => { set({ naturalizeAmount: v }); writeKey("hs-naturalizeAmount", v); },
  setNatVibratoRate: (v: any) => { set({ natVibratoRate: v }); writeKey("hs-natVibratoRate", v); },
  setNatVibratoDepth: (v: any) => { set({ natVibratoDepth: v }); writeKey("hs-natVibratoDepth", v); },
  setNatFormantStrength: (v: any) => { set({ natFormantStrength: v }); writeKey("hs-natFormantStrength", v); },
  setNatMetallicReduction: (v: any) => { set({ natMetallicReduction: v }); writeKey("hs-natMetallicReduction", v); },
  setNatQuantizationMask: (v: any) => { set({ natQuantizationMask: v }); writeKey("hs-natQuantizationMask", v); },
  setNatTransitionSmooth: (v: any) => { set({ natTransitionSmooth: v }); writeKey("hs-natTransitionSmooth", v); },
  setPpVaeReencode: (v: any) => { set({ ppVaeReencode: v }); writeKey("hs-ppVaeReencode", v); },
  setPpVaeBlend: (v: any) => { set({ ppVaeBlend: v }); writeKey("hs-ppVaeBlend", v); },
  setPpVaeUseOnnx: (v: any) => { set({ ppVaeUseOnnx: v }); writeKey("hs-ppVaeUseOnnx", v); },
  setCoverArtEnabled: (v: any) => { set({ coverArtEnabled: v }); writeKey("hs-coverArtEnabled", v); },
  setCoverArtSubject: (v: any) => { set({ coverArtSubject: v }); writeKey("hs-coverArtSubject", v); },
  setQualityEvalEnabled: (v: any) => { set({ qualityEvalEnabled: v }); writeKey("hs-qualityEvalEnabled", v); },
  setQualityEvalTarget: (v: any) => { set({ qualityEvalTarget: v }); writeKey("hs-qualityEvalTarget", v); },
  setWhisperLyricsEnabled: (v: any) => { set({ whisperLyricsEnabled: v }); writeKey("hs-whisperLyricsEnabled", v); },
  setWhisperModel: (v: any) => { set({ whisperModel: v }); writeKey("hs-whisperModel", v); },
  setWhisperLanguage: (v: any) => { set({ whisperLanguage: v }); writeKey("hs-whisperLang", v); },
  setWhisperBeamSize: (v: any) => { set({ whisperBeamSize: v }); writeKey("hs-whisperBeam", v); },
  setWhisperIsolateVocals: (v: any) => { set({ whisperIsolateVocals: v }); writeKey("hs-whisperIsolate", v); },
  setPostprocessEnabled: (v: any) => { set({ postprocessEnabled: v }); writeKey("hs-postprocessEnabled", v); },
  setPostprocessPlugin: (v: any) => { set({ postprocessPlugin: v }); writeKey("hs-postprocessPlugin", v); },
  setUseOrtVae: (v: any) => { set({ useOrtVae: v }); writeKey('hs-useOrtVae', v); },
  setLufsEnabled: (v: any) => { set({ lufsEnabled: v }); writeKey('hs-lufsEnabled', v); },
  setLufsPreset: (v: any) => {
    set({ lufsPreset: v });
    writeKey('hs-lufsPreset', v);
    // Auto-set target from preset
    const presetTargets: Record<string, number> = {
      spotify: -14, apple: -16, ebu: -23, club: -8,
    };
    if (v !== 'custom' && presetTargets[v] !== undefined) {
      set({ lufsTarget: presetTargets[v] });
      writeKey('hs-lufsTarget', presetTargets[v]);
    }
  },
  setLufsTarget: (v: any) => { set({ lufsTarget: v }); writeKey('hs-lufsTarget', v); },

  // Plugin param helpers
  setPluginParam: (key: string, value: string) => {
    const prev = get().pluginParams;
    const next = { ...prev, [key]: value };
    set({ pluginParams: next });
    writeKey('hs-pluginParams', next);
  },
  resetPluginParams: (pluginName: string) => {
    const prev = get().pluginParams;
    const next: Record<string, string> = {};
    for (const [k, v] of Object.entries(prev)) {
      if (!k.startsWith(pluginName + ':')) next[k] = v as string;
    }
    set({ pluginParams: next });
    writeKey('hs-pluginParams', next);
  },

  // -- Derived: assemble generation params --
  getGlobalParams: (): Partial<GenerationParams> => {
    const s = get();
    const settings: AppSettings = readKey('ace-settings', DEFAULT_SETTINGS);
    const triggerWord: string = settings.triggerUseFilename && s.adapter
      ? (s.adapter.split(/[\\\/]/).pop()?.replace(/\.safetensors$/i, '') || '')
      : '';

    return {
      ditModel: s.ditModel, lmModel: s.lmModel, vaeModel: s.vaeModel, embeddingModel: s.embeddingModel,
      loraPath: s.adapter, loraScale: s.adapterScale,
      adapterGroupScales: s.adapter ? s.adapterGroupScales : undefined,
      adapterMode: s.adapter ? s.adapterMode : 'merge',
      triggerWord: triggerWord || undefined,
      triggerPlacement: triggerWord ? settings.triggerPlacement : undefined,
      inferenceSteps: s.inferenceSteps, guidanceScale: s.guidanceScale, shift: s.shift,
      cfgCutoffRatio: s.cfgCutoffRatio < 1.0 ? s.cfgCutoffRatio : undefined,
      lmCfgCutoffRatio: s.lmCfgCutoffRatio < 1.0 ? s.lmCfgCutoffRatio : undefined,
      cacheRatio: s.cacheRatio > 0 ? s.cacheRatio : undefined,
      inferMethod: s.inferMethod, scheduler: s.scheduler, guidanceMode: s.guidanceMode,
      seed: s.seed, randomSeed: s.randomSeed, batchSize: s.batchSize,
      storkSubsteps: (s.inferMethod === 'stork2' || s.inferMethod === 'stork4') ? s.storkSubsteps : undefined,
      beatStability: s.inferMethod === 'jkass_fast' ? s.beatStability : undefined,
      frequencyDamping: s.inferMethod === 'jkass_fast' ? s.frequencyDamping : undefined,
      temporalSmoothing: s.inferMethod === 'jkass_fast' ? s.temporalSmoothing : undefined,
      apgMomentum: s.guidanceMode === 'apg' ? s.apgMomentum : undefined,
      apgNormThreshold: s.guidanceMode === 'apg' ? s.apgNormThreshold : undefined,
      skipLm: s.skipLm, useCotCaption: s.useCotCaption,
      skipLrc: s.skipLrc || undefined,
      lmTemperature: s.lmTemperature, lmCfgScale: s.lmCfgScale,
      lmTopK: s.lmTopK, lmTopP: s.lmTopP, lmNegativePrompt: s.lmNegativePrompt,
      audioCoverStrength: (!s.skipLm && s.lmCodesStrength < 1.0) ? s.lmCodesStrength : undefined,
      postProcessingEnabled: s.postProcessingEnabled,
      spectralLifterEnabled: s.postProcessingEnabled ? s.spectralLifterEnabled : false,
      slDenoiseStrength: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slDenoiseStrength : undefined,
      slNoiseFloor: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slNoiseFloor : undefined,
      slHfMix: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slHfMix : undefined,
      slTransientBoost: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slTransientBoost : undefined,
      slShimmerReduction: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slShimmerReduction : undefined,
      masteringEnabled: s.postProcessingEnabled ? s.masteringEnabled : false,
      masteringReference: (s.postProcessingEnabled && s.masteringEnabled) ? s.masteringReference : undefined,
      timbreReference: s.timbreAudioPath
        ? s.timbreAudioPath
        : (s.postProcessingEnabled && s.masteringEnabled && s.timbreReference && s.masteringReference) ? true : undefined,
      dcwEnabled: s.dcwEnabled,
      dcwMode: s.dcwEnabled ? s.dcwMode : undefined,
      // Route the correct scaler to dcw_scaler based on mode:
      // low/double/pix use dcwLowScaler, high uses dcwHighScaler
      dcwScaler: s.dcwEnabled
        ? (s.dcwMode === 'high' ? s.dcwHighScaler * 0.02 : s.dcwLowScaler * 0.05)
        : undefined,
      dcwHighScaler: (s.dcwEnabled && s.dcwMode === 'double') ? s.dcwHighScaler * 0.02 : undefined,
      latentShift: s.latentShift !== 0 ? s.latentShift : undefined,
      latentRescale: s.latentRescale !== 1 ? s.latentRescale : undefined,
      customTimesteps: s.customTimesteps || undefined,
      denoiseStrength: s.denoiseStrength > 0 ? s.denoiseStrength : undefined,
      denoiseSmoothing: s.denoiseStrength > 0 ? s.denoiseSmoothing : undefined,
      denoiseMix: s.denoiseStrength > 0 ? s.denoiseMix : undefined,
      pluginParams: Object.keys(s.pluginParams).length > 0 ? s.pluginParams : undefined,
      autoTrimEnabled: s.autoTrimEnabled || undefined,
      durationBuffer: s.autoTrimEnabled ? s.durationBuffer : undefined,
      autoTrimFadeMs: s.autoTrimEnabled ? s.autoTrimFadeMs : undefined,
      vocalNaturalizerEnabled: s.postProcessingEnabled ? s.vocalNaturalizerEnabled : false,
      gainOffsetDb: (s.postProcessingEnabled && s.gainOffsetDb !== 0) ? s.gainOffsetDb : undefined,
      naturalizeAmount: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.naturalizeAmount : undefined,
      natVibratoRate: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natVibratoRate : undefined,
      natVibratoDepth: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natVibratoDepth : undefined,
      natFormantStrength: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natFormantStrength : undefined,
      natMetallicReduction: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natMetallicReduction : undefined,
      natQuantizationMask: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natQuantizationMask : undefined,
      natTransitionSmooth: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natTransitionSmooth : undefined,
      ppVaeReencode: (s.postProcessingEnabled && s.ppVaeReencode) || undefined,
      ppVaeBlend: (s.postProcessingEnabled && s.ppVaeReencode && s.ppVaeBlend > 0) ? s.ppVaeBlend : undefined,
      ppVaeUseOnnx: (s.postProcessingEnabled && s.ppVaeReencode) ? s.ppVaeUseOnnx : undefined,
      coverArtEnabled: (s.postProcessingEnabled && s.coverArtEnabled) || undefined,
      coverArtSubject: (s.postProcessingEnabled && s.coverArtEnabled && s.coverArtSubject) ? s.coverArtSubject : undefined,
      qualityEvalEnabled: (s.postProcessingEnabled && s.qualityEvalEnabled) || undefined,
      qualityEvalTarget: (s.postProcessingEnabled && s.qualityEvalEnabled) ? s.qualityEvalTarget : undefined,
      postprocessPlugin: (s.postProcessingEnabled && s.postprocessEnabled && s.postprocessPlugin) ? s.postprocessPlugin : undefined,
      lufsEnabled: (s.postProcessingEnabled && s.masteringEnabled && s.lufsEnabled) || undefined,
      lufsTarget: (s.postProcessingEnabled && s.masteringEnabled && s.lufsEnabled) ? s.lufsTarget : undefined,
      useOrtVae: s.useOrtVae || undefined,
      whisperLyricsEnabled: s.whisperLyricsEnabled,
      whisperModel: s.whisperLyricsEnabled ? s.whisperModel : undefined,
      whisperLanguage: s.whisperLyricsEnabled ? s.whisperLanguage : undefined,
      whisperBeamSize: s.whisperLyricsEnabled ? s.whisperBeamSize : undefined,
      whisperIsolateVocals: s.whisperLyricsEnabled ? s.whisperIsolateVocals : undefined,
    };
  },
}));
