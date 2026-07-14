// GlobalParamsContext.tsx -- Thin compatibility wrapper
//
// The actual state now lives in globalParamsStore (Zustand).
// This file provides backward-compatible exports so existing consumers
// can migrate incrementally.
//
// NEW CODE should import from '../stores/globalParamsStore' directly
// and use selectors: useGlobalParamsStore(s => s.fieldName)

import React from 'react';
import { useGlobalParamsStore } from '../stores/globalParamsStore';
import type { AdapterGroupScales } from '../stores/globalParamsStore';
import type { GenerationParams } from '../types';

// Re-export types that consumers may import from here
export type { AdapterGroupScales } from '../stores/globalParamsStore';

// Re-export the Zustand store for direct selector usage
export { useGlobalParamsStore } from '../stores/globalParamsStore';

// ── Legacy interface (matches the old context shape) ──

export interface GlobalParams {
  // Models
  ditModel: string; setDitModel: (v: string) => void;
  lmModel: string; setLmModel: (v: string) => void;
  vaeModel: string; setVaeModel: (v: string) => void;
  embeddingModel: string; setEmbeddingModel: (v: string) => void;
  // Adapters
  adapter: string; setAdapter: (v: string) => void;
  adapterScale: number; setAdapterScale: (v: number) => void;
  // Multi-adapter stack: { path, scale }[] applied together (supersedes `adapter`)
  adapterStack: { path: string; scale: number }[];
  setAdapterStack: (v: { path: string; scale: number }[]) => void;
  toggleAdapterInStack: (path: string, scale?: number) => void;
  setAdapterStackScale: (path: string, scale: number) => void;
  // Stack scaling: 'sum' (raw scales summed) or 'blend' (weights normalised to budget)
  adapterStackMode: string; setAdapterStackMode: (v: string) => void;
  adapterStackBudget: number; setAdapterStackBudget: (v: number) => void;
  // Per-section masking (P2) tuning
  adapterSectionAlignAt: number; setAdapterSectionAlignAt: (v: number) => void;
  adapterSectionIsolation: number; setAdapterSectionIsolation: (v: number) => void;
  adapterMode: string; setAdapterMode: (v: string) => void;
  // Runtime delta VRAM precision: 'bf16' | 'q8_0' | 'q4_k' (runtime mode only)
  adapterRuntimeQuant: string; setAdapterRuntimeQuant: (v: string) => void;
  adapterGroupScales: AdapterGroupScales;
  setAdapterGroupScales: (v: AdapterGroupScales) => void;
  // Basin re-base (cross-base adapter support)
  rebaseSource: string; setRebaseSource: (v: string) => void;
  rebaseBeta: number; setRebaseBeta: (v: number) => void;
  adapterFolder: string; setAdapterFolder: (v: string) => void;
  advancedAdapters: boolean; setAdvancedAdapters: (v: boolean) => void;
  adaptersOpen: boolean; setAdaptersOpen: (v: boolean) => void;
  // Generation
  inferenceSteps: number; setInferenceSteps: (v: number) => void;
  guidanceScale: number; setGuidanceScale: (v: number) => void;
  cfgCutoffRatio: number; setCfgCutoffRatio: (v: number) => void;
  lmCfgCutoffRatio: number; setLmCfgCutoffRatio: (v: number) => void;
  cacheRatio: number; setCacheRatio: (v: number) => void;
  shift: number; setShift: (v: number) => void;
  inferMethod: string; setInferMethod: (v: string) => void;
  scheduler: string; setScheduler: (v: string) => void;
  guidanceMode: string; setGuidanceMode: (v: string) => void;
  seed: number; setSeed: (v: number) => void;
  randomSeed: boolean; setRandomSeed: (v: boolean) => void;
  // LM Seed — independent from seed/randomSeed above; drives LM-phase
  // sampling. When lmSeedFollowsDit is true, it's tied to `seed` instead.
  lmSeed: number; setLmSeed: (v: number) => void;
  lmSeedFollowsDit: boolean; setLmSeedFollowsDit: (v: boolean) => void;
  batchSize: number; setBatchSize: (v: number) => void;
  // Solver sub-params
  storkSubsteps: number; setStorkSubsteps: (v: number) => void;
  beatStability: number; setBeatStability: (v: number) => void;
  frequencyDamping: number; setFrequencyDamping: (v: number) => void;
  temporalSmoothing: number; setTemporalSmoothing: (v: number) => void;
  // Guidance sub-params
  apgMomentum: number; setApgMomentum: (v: number) => void;
  apgNormThreshold: number; setApgNormThreshold: (v: number) => void;
  // Plugin params
  pluginParams: Record<string, string>;
  setPluginParam: (key: string, value: string) => void;
  resetPluginParams: (pluginName: string) => void;
  // DCW
  dcwEnabled: boolean; setDcwEnabled: (v: boolean) => void;
  dcwMode: string; setDcwMode: (v: string) => void;
  dcwLowScaler: number; setDcwLowScaler: (v: number) => void;
  dcwHighScaler: number; setDcwHighScaler: (v: number) => void;
  // Latent
  latentShift: number; setLatentShift: (v: number) => void;
  latentRescale: number; setLatentRescale: (v: number) => void;
  customTimesteps: string; setCustomTimesteps: (v: string) => void;
  // Denoiser
  denoiseStrength: number; setDenoiseStrength: (v: number) => void;
  denoiseSmoothing: number; setDenoiseSmoothing: (v: number) => void;
  denoiseMix: number; setDenoiseMix: (v: number) => void;
  // Auto-trim
  autoTrimEnabled: boolean; setAutoTrimEnabled: (v: boolean) => void;
  durationBuffer: number; setDurationBuffer: (v: number) => void;
  autoTrimFadeMs: number; setAutoTrimFadeMs: (v: number) => void;
  // LM
  skipLm: boolean; setSkipLm: (v: boolean) => void;
  skipLrc: boolean; setSkipLrc: (v: boolean) => void;
  useCotCaption: boolean; setUseCotCaption: (v: boolean) => void;
  lmTemperature: number; setLmTemperature: (v: number) => void;
  lmCfgScale: number; setLmCfgScale: (v: number) => void;
  lmTopK: number; setLmTopK: (v: number) => void;
  lmTopP: number; setLmTopP: (v: number) => void;
  lmNegativePrompt: string; setLmNegativePrompt: (v: string) => void;
  lmCodesStrength: number; setLmCodesStrength: (v: number) => void;
  // Post-processing
  postProcessingEnabled: boolean; setPostProcessingEnabled: (v: boolean) => void;
  spectralLifterEnabled: boolean; setSpectralLifterEnabled: (v: boolean) => void;
  slDenoiseStrength: number; setSlDenoiseStrength: (v: number) => void;
  slNoiseFloor: number; setSlNoiseFloor: (v: number) => void;
  slHfMix: number; setSlHfMix: (v: number) => void;
  slTransientBoost: number; setSlTransientBoost: (v: number) => void;
  slShimmerReduction: number; setSlShimmerReduction: (v: number) => void;
  masteringEnabled: boolean; setMasteringEnabled: (v: boolean) => void;
  masteringReference: string; setMasteringReference: (v: string) => void;
  timbreReference: boolean; setTimbreReference: (v: boolean) => void;
  timbreAudioPath: string; setTimbreAudioPath: (v: string) => void;
  // Naturalizer
  vocalNaturalizerEnabled: boolean; setVocalNaturalizerEnabled: (v: boolean) => void;
  // Pre-VST gain offset
  gainOffsetDb: number; setGainOffsetDb: (v: number) => void;
  naturalizeAmount: number; setNaturalizeAmount: (v: number) => void;
  natVibratoRate: number; setNatVibratoRate: (v: number) => void;
  natVibratoDepth: number; setNatVibratoDepth: (v: number) => void;
  natFormantStrength: number; setNatFormantStrength: (v: number) => void;
  natMetallicReduction: number; setNatMetallicReduction: (v: number) => void;
  natQuantizationMask: number; setNatQuantizationMask: (v: number) => void;
  natTransitionSmooth: number; setNatTransitionSmooth: (v: number) => void;
  // PP-VAE
  ppVaeReencode: boolean; setPpVaeReencode: (v: boolean) => void;
  ppVaeBlend: number; setPpVaeBlend: (v: number) => void;
  ppVaeUseOnnx: boolean; setPpVaeUseOnnx: (v: boolean) => void;
  // Cover Art
  coverArtEnabled: boolean; setCoverArtEnabled: (v: boolean) => void;
  coverArtSubject: string; setCoverArtSubject: (v: string) => void;
  // Quality Evaluator
  qualityEvalEnabled: boolean; setQualityEvalEnabled: (v: boolean) => void;
  qualityEvalTarget: string; setQualityEvalTarget: (v: string) => void;
  // Postprocess plugin
  postprocessEnabled: boolean; setPostprocessEnabled: (v: boolean) => void;
  postprocessPlugin: string; setPostprocessPlugin: (v: string) => void;
  // LUFS Normalization
  lufsEnabled: boolean; setLufsEnabled: (v: boolean) => void;
  lufsPreset: string; setLufsPreset: (v: string) => void;
  lufsTarget: number; setLufsTarget: (v: number) => void;
  // VAE backend
  useOrtVae: boolean; setUseOrtVae: (v: boolean) => void;
  // Whisper Lyrics
  whisperLyricsEnabled: boolean; setWhisperLyricsEnabled: (v: boolean) => void;
  whisperModel: string; setWhisperModel: (v: string) => void;
  whisperLanguage: string; setWhisperLanguage: (v: string) => void;
  whisperBeamSize: number; setWhisperBeamSize: (v: number) => void;
  whisperIsolateVocals: boolean; setWhisperIsolateVocals: (v: boolean) => void;
  // Derived
  getGlobalParams: () => Partial<GenerationParams>;
}

// ── Provider (now a no-op wrapper — state lives in Zustand) ──

export const GlobalParamsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return <>{children}</>;
};

/**
 * Backward-compatible hook — returns the full store state.
 * @deprecated Prefer useGlobalParamsStore(s => s.fieldName) for selective subscriptions.
 * This hook triggers a re-render on ANY store change.
 */
export function useGlobalParams(): GlobalParams {
  return useGlobalParamsStore() as GlobalParams;
}
