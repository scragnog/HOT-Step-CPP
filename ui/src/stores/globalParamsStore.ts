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
  time_embed: number;
  proj_in: number;
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
  // Multi-adapter stack: a list of { path, scale } applied together, each with
  // its own scale. When non-empty it supersedes the single `adapter`. Group
  // scales, adapter mode and basin re-base apply globally to the whole stack.
  adapterStack: readKey("hs-adapterStack", [] as { path: string; scale: number }[]),
  // Stack scaling mode:
  //  'sum'   — each entry's `scale` is its absolute scale; the engine sums them
  //            (can deliberately over-drive: Σ scale may exceed 1).
  //  'blend' — each entry's `scale` is a relative weight; the effective scales
  //            are normalised to the budget so Σ effective = adapterStackBudget,
  //            keeping combined strength constant as adapters are added.
  adapterStackMode: readKey("hs-adapterStackMode", 'blend'),
  adapterStackBudget: readKey("hs-adapterStackBudget", 0.75),
  // Per-section masking (P2) tuning: alignment step fraction, and 0..1 regional
  // self-attention isolation (moderate default) to stop sections inheriting the
  // first section's voice.
  adapterSectionAlignAt: readKey("hs-adapterSectionAlignAt", 0.55),
  adapterSectionIsolation: readKey("hs-adapterSectionIsolation", 0.5),
  adapterMode: readKey("hs-adapterMode", 'runtime'),
  // Runtime adapter delta VRAM precision: 'bf16' (full), 'q8_0' (~½), 'q4_k' (~¼).
  // Lets many stacked adapters fit in VRAM; runtime mode only.
  adapterRuntimeQuant: readKey("hs-adapterRuntimeQuant", 'bf16'),
  // Merge (low VRAM): re-encode merged weights to the base's native quant instead
  // of F32 promotion (~¼ the merged-DiT VRAM on a Q8 base). Merge mode only.
  adapterMergeLowVram: readKey("hs-adapterMergeLowVram", false),
  adapterGroupScales: readKey("hs-adapterGroupScales", {
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0, time_embed: 0.0, proj_in: 0.0,
  }),
  // Basin re-base: nudge a cross-base adapter into the basin of the model it was
  // trained on. rebaseSource = DiT model name (the adapter's "home base").
  // Merge mode only. Remembered across sessions.
  rebaseSource: readKey("hs-rebaseSource", ''),
  rebaseBeta: readKey("hs-rebaseBeta", 0.75),
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
  // LM Seed — independent of the DiT/generation seed above, unless tied
  // via lmSeedFollowsDit (default true = original tied behavior).
  lmSeed: readKey("hs-lmSeed", 42),
  lmSeedFollowsDit: readKey("hs-lmSeedFollowsDit", true),
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
  lssStrength: readKey("hs-lssStrength", 0.0),
  lssVarThresh: readKey("hs-lssVarThresh", 0.15),
  lssDcRemove: readKey("hs-lssDcRemove", true),
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
  stableStepOn: readKey("hs-stableStepOn", false),
  stableStepStrength: readKey("hs-stableStepStrength", 0.3),
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
  setAdapterStack: (v: any) => { set({ adapterStack: v }); writeKey("hs-adapterStack", v); },
  // Add/remove an adapter path from the stack (idempotent toggle).
  toggleAdapterInStack: (path: string, scale = 1.0) => {
    const cur: { path: string; scale: number }[] = get().adapterStack || [];
    const exists = cur.some(a => a.path === path);
    const next = exists ? cur.filter(a => a.path !== path) : [...cur, { path, scale }];
    set({ adapterStack: next }); writeKey("hs-adapterStack", next);
  },
  // Set the per-adapter scale (sum mode) or relative weight (blend mode) for one entry.
  setAdapterStackScale: (path: string, scale: number) => {
    const cur: { path: string; scale: number }[] = get().adapterStack || [];
    const next = cur.map(a => (a.path === path ? { ...a, scale } : a));
    set({ adapterStack: next }); writeKey("hs-adapterStack", next);
  },
  // Set one entry's active-timestep window (timestep-dependent adapters).
  // Axis is remaining-steps fraction: 1 = first step (noise), 0 = last step
  // (clean). Evaluated per STEP by the engine (gain_domain 'steps'), so 50%
  // means half the steps even on shift-skewed schedules.
  // Full range [0,1] means "always on" and is stripped to keep entries clean.
  setAdapterStackWindow: (path: string, stepStart: number, stepEnd: number) => {
    const cur: { path: string; scale: number; stepStart?: number; stepEnd?: number }[] = get().adapterStack || [];
    const full = stepStart <= 0 && stepEnd >= 1;
    const next = cur.map(a => {
      if (a.path !== path) return a;
      if (full) { const { stepStart: _s, stepEnd: _e, ...rest } = a; return rest; }
      return { ...a, stepStart, stepEnd };
    });
    set({ adapterStack: next }); writeKey("hs-adapterStack", next);
  },
  setAdapterStackMode: (v: any) => { set({ adapterStackMode: v }); writeKey("hs-adapterStackMode", v); },
  setAdapterStackBudget: (v: any) => { set({ adapterStackBudget: v }); writeKey("hs-adapterStackBudget", v); },
  setAdapterSectionAlignAt: (v: any) => { set({ adapterSectionAlignAt: v }); writeKey("hs-adapterSectionAlignAt", v); },
  setAdapterSectionIsolation: (v: any) => { set({ adapterSectionIsolation: v }); writeKey("hs-adapterSectionIsolation", v); },
  setAdapterMode: (v: any) => { set({ adapterMode: v }); writeKey("hs-adapterMode", v); },
  setAdapterRuntimeQuant: (v: any) => { set({ adapterRuntimeQuant: v }); writeKey("hs-adapterRuntimeQuant", v); },
  setAdapterMergeLowVram: (v: any) => { set({ adapterMergeLowVram: v }); writeKey("hs-adapterMergeLowVram", v); },
  setAdapterGroupScales: (v: any) => { set({ adapterGroupScales: v }); writeKey("hs-adapterGroupScales", v); },
  setRebaseSource: (v: any) => { set({ rebaseSource: v }); writeKey("hs-rebaseSource", v); },
  setRebaseBeta: (v: any) => { set({ rebaseBeta: v }); writeKey("hs-rebaseBeta", v); },
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
  setLmSeed: (v: any) => { set({ lmSeed: v }); writeKey("hs-lmSeed", v); },
  setLmSeedFollowsDit: (v: any) => { set({ lmSeedFollowsDit: v }); writeKey("hs-lmSeedFollowsDit", v); },
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
  setLssStrength: (v: any) => { set({ lssStrength: v }); writeKey("hs-lssStrength", v); },
  setLssVarThresh: (v: any) => { set({ lssVarThresh: v }); writeKey("hs-lssVarThresh", v); },
  setLssDcRemove: (v: any) => { set({ lssDcRemove: v }); writeKey("hs-lssDcRemove", v); },
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
  setStableStepOn: (v: any) => { set({ stableStepOn: v }); writeKey("hs-stableStepOn", v); },
  setStableStepStrength: (v: any) => { set({ stableStepStrength: v }); writeKey("hs-stableStepStrength", v); },
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

    // Effective adapter stack: the multi-adapter list when in Advanced mode,
    // otherwise the single `adapter` folded into a one-element stack. `primary`
    // drives the single-adapter features (trigger word, basin re-base, group
    // scales) which remain keyed on the first adapter. Gated on advancedAdapters
    // to match the badge/UI — a stack persisted from a previous Advanced session
    // must not override the Simple-mode selection.
    const isStack = !!(s.advancedAdapters && s.adapterStack && s.adapterStack.length > 0);
    const rawStack: { path: string; scale: number; stepStart?: number; stepEnd?: number }[] = isStack
      ? s.adapterStack
      : (s.adapter ? [{ path: s.adapter, scale: s.adapterScale }] : []);

    // Blend mode (multi-adapter stacks only): treat each entry's `scale` as a
    // relative weight and normalise so the effective scales sum to the budget,
    // keeping combined strength constant regardless of how many are stacked.
    // Sum mode (and the single-adapter fallback) sends the raw scales as-is.
    // Blend only applies with 2+ adapters — a single adapter's strength is just
    // its own scale, sent as-is.
    let stack = rawStack;
    if (isStack && s.adapterStackMode === 'blend' && rawStack.length >= 2) {
      const budget = s.adapterStackBudget ?? 0.75;
      const sumW = rawStack.reduce((acc, e) => acc + (e.scale || 0), 0);
      // Spread each entry so per-adapter extras (stepStart/stepEnd timestep
      // window, etc.) survive the blend re-scale.
      stack = sumW > 0
        ? rawStack.map(e => ({ ...e, scale: +(budget * (e.scale || 0) / sumW).toFixed(4) }))
        : rawStack.map(e => ({ ...e, scale: +(budget / rawStack.length).toFixed(4) }));
    }
    const primary = stack[0]?.path || '';
    // Timestep windows force runtime mode server-side regardless of the selected
    // adapter mode — runtime-only knobs must flow whenever they're present.
    const hasStepWindows = stack.some(
      (e: { stepStart?: number; stepEnd?: number }) => e.stepStart !== undefined || e.stepEnd !== undefined,
    );

    // Trigger words: one per loaded adapter (every adapter in the stack
    // contributes its filename trigger, not just the first).
    const triggerWords: string[] = settings.triggerUseFilename
      ? stack
          .map(e => e.path.split(/[\\\/]/).pop()?.replace(/\.safetensors$/i, '') || '')
          .filter(Boolean)
      : [];
    const triggerWord = triggerWords.join(', ');

    return {
      ditModel: s.ditModel, lmModel: s.lmModel, vaeModel: s.vaeModel, embeddingModel: s.embeddingModel,
      loraPath: primary, loraScale: stack[0]?.scale ?? 1.0,
      // Multi-adapter stack (>1 entry) — sent alongside loraPath; the engine
      // prefers the stack and applies each adapter with its own scale.
      loraStack: stack.length > 0 ? stack : undefined,
      // Stack scaling mode + budget — reused for per-section masking transforms.
      adapterStackMode: s.adapterStackMode,
      adapterStackBudget: s.adapterStackBudget,
      // Per-section masking tuning (only meaningful with a 2+ adapter stack).
      adapterSectionAlignAt: stack.length >= 2 ? s.adapterSectionAlignAt : undefined,
      adapterSectionIsolation: stack.length >= 2 ? s.adapterSectionIsolation : undefined,
      adapterGroupScales: primary ? s.adapterGroupScales : undefined,
      adapterMode: primary ? s.adapterMode : 'merge',
      // Runtime delta quantization (VRAM saver) — relevant in both runtime modes
      // (lowrank still stores full-size re-base corrections / Conv1d fallbacks).
      // Also sent when timestep windows are active: they force runtime mode
      // server-side even from Merge, and gating on the *selected* mode silently
      // killed the knob there (full BF16 deltas, 2×8 GB — the 32 GB bug).
      adapterRuntimeQuant: (primary && (s.adapterMode === 'runtime' || s.adapterMode === 'runtime_lowrank' || hasStepWindows))
        ? s.adapterRuntimeQuant : undefined,
      // Merge low-VRAM storage (native-quant re-encode) — only relevant in merge mode.
      adapterMergeLowVram: (primary && s.adapterMode !== 'runtime' && s.adapterMergeLowVram) ? true : undefined,
      // Basin re-base: only sent with an adapter and a chosen source. Works in
      // both merge and runtime modes (runtime folds the nudge into the delta sum);
      // the engine skips it on the per-section masking path.
      rebaseSource: (primary && s.rebaseSource) ? s.rebaseSource : undefined,
      rebaseBeta: (primary && s.rebaseSource) ? s.rebaseBeta : undefined,
      triggerWord: triggerWord || undefined,
      triggerWords: triggerWords.length ? triggerWords : undefined,
      triggerPlacement: triggerWords.length ? settings.triggerPlacement : undefined,
      inferenceSteps: s.inferenceSteps, guidanceScale: s.guidanceScale, shift: s.shift,
      cfgCutoffRatio: s.cfgCutoffRatio < 1.0 ? s.cfgCutoffRatio : undefined,
      lmCfgCutoffRatio: s.lmCfgCutoffRatio < 1.0 ? s.lmCfgCutoffRatio : undefined,
      cacheRatio: s.cacheRatio > 0 ? s.cacheRatio : undefined,
      inferMethod: s.inferMethod, scheduler: s.scheduler, guidanceMode: s.guidanceMode,
      seed: s.seed, randomSeed: s.randomSeed,
      lmSeed: s.lmSeed, lmSeedFollowsDit: s.lmSeedFollowsDit,
      batchSize: s.batchSize,
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
      lssStrength: s.lssStrength > 0 ? s.lssStrength : undefined,
      lssVarThresh: s.lssStrength > 0 ? s.lssVarThresh : undefined,
      lssDcRemove: s.lssStrength > 0 ? s.lssDcRemove : undefined,
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
      stableStepOn: (s.postProcessingEnabled && s.stableStepOn) || undefined,
      stableStepStrength: (s.postProcessingEnabled && s.stableStepOn) ? s.stableStepStrength : undefined,
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
