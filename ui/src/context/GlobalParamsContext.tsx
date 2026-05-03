// GlobalParamsContext.tsx — Centralized engine configuration state
//
// All "global" generation parameters that apply across all modes
// (Create, Lyric Studio, Cover Studio, etc.) live here.
//
// Uses the SAME `hs-*` localStorage keys as before — zero migration.
// Lyric Studio's direct localStorage reads continue to work.

import React, { createContext, useContext, useMemo, useCallback } from 'react';
import { usePersistedState } from '../hooks/usePersistedState';
import { DEFAULT_SETTINGS, type AppSettings } from '../components/settings/SettingsPanel';
import type { GenerationParams } from '../types';

// ── Types ────────────────────────────────────────────────────────────────────

export interface AdapterGroupScales {
  self_attn: number;
  cross_attn: number;
  mlp: number;
  cond_embed: number;
}

export interface GlobalParams {
  // ── Models ──
  ditModel: string;
  setDitModel: (v: string) => void;
  lmModel: string;
  setLmModel: (v: string) => void;
  vaeModel: string;
  setVaeModel: (v: string) => void;
  embeddingModel: string;
  setEmbeddingModel: (v: string) => void;

  // ── Adapters ──
  adapter: string;
  setAdapter: (v: string) => void;
  adapterScale: number;
  setAdapterScale: (v: number) => void;
  adapterMode: string;
  setAdapterMode: (v: string) => void;
  adapterGroupScales: AdapterGroupScales;
  setAdapterGroupScales: (v: AdapterGroupScales) => void;
  adapterFolder: string;
  setAdapterFolder: (v: string) => void;
  advancedAdapters: boolean;
  setAdvancedAdapters: (v: boolean) => void;
  adaptersOpen: boolean;
  setAdaptersOpen: (v: boolean) => void;

  // ── Generation Settings ──
  inferenceSteps: number;
  setInferenceSteps: (v: number) => void;
  guidanceScale: number;
  setGuidanceScale: (v: number) => void;
  shift: number;
  setShift: (v: number) => void;
  inferMethod: string;
  setInferMethod: (v: string) => void;
  scheduler: string;
  setScheduler: (v: string) => void;
  guidanceMode: string;
  setGuidanceMode: (v: string) => void;
  seed: number;
  setSeed: (v: number) => void;
  randomSeed: boolean;
  setRandomSeed: (v: boolean) => void;
  batchSize: number;
  setBatchSize: (v: number) => void;

  // Solver sub-params
  storkSubsteps: number;
  setStorkSubsteps: (v: number) => void;
  beatStability: number;
  setBeatStability: (v: number) => void;
  frequencyDamping: number;
  setFrequencyDamping: (v: number) => void;
  temporalSmoothing: number;
  setTemporalSmoothing: (v: number) => void;

  // Guidance sub-params
  apgMomentum: number;
  setApgMomentum: (v: number) => void;
  apgNormThreshold: number;
  setApgNormThreshold: (v: number) => void;

  // DCW
  dcwEnabled: boolean;
  setDcwEnabled: (v: boolean) => void;
  dcwMode: string;
  setDcwMode: (v: string) => void;
  dcwScaler: number;
  setDcwScaler: (v: number) => void;
  dcwHighScaler: number;
  setDcwHighScaler: (v: number) => void;

  // Latent post-processing
  latentShift: number;
  setLatentShift: (v: number) => void;
  latentRescale: number;
  setLatentRescale: (v: number) => void;
  customTimesteps: string;
  setCustomTimesteps: (v: string) => void;

  // Post-VAE spectral denoiser
  denoiseStrength: number;
  setDenoiseStrength: (v: number) => void;
  denoiseSmoothing: number;
  setDenoiseSmoothing: (v: number) => void;
  denoiseMix: number;
  setDenoiseMix: (v: number) => void;

  // ── Duration Buffer / Auto-Trim ──
  autoTrimEnabled: boolean;
  setAutoTrimEnabled: (v: boolean) => void;
  durationBuffer: number;
  setDurationBuffer: (v: number) => void;
  autoTrimFadeMs: number;
  setAutoTrimFadeMs: (v: number) => void;

  // ── LM / Thinking ──
  skipLm: boolean;
  setSkipLm: (v: boolean) => void;
  useCotCaption: boolean;
  setUseCotCaption: (v: boolean) => void;
  lmTemperature: number;
  setLmTemperature: (v: number) => void;
  lmCfgScale: number;
  setLmCfgScale: (v: number) => void;
  lmTopK: number;
  setLmTopK: (v: number) => void;
  lmTopP: number;
  setLmTopP: (v: number) => void;
  lmNegativePrompt: string;
  setLmNegativePrompt: (v: string) => void;
  lmCodesStrength: number;
  setLmCodesStrength: (v: number) => void;

  // ── Post-Processing (master toggle + individual stages) ──
  postProcessingEnabled: boolean;
  setPostProcessingEnabled: (v: boolean) => void;
  spectralLifterEnabled: boolean;
  setSpectralLifterEnabled: (v: boolean) => void;
  slDenoiseStrength: number;
  setSlDenoiseStrength: (v: number) => void;
  slNoiseFloor: number;
  setSlNoiseFloor: (v: number) => void;
  slHfMix: number;
  setSlHfMix: (v: number) => void;
  slTransientBoost: number;
  setSlTransientBoost: (v: number) => void;
  slShimmerReduction: number;
  setSlShimmerReduction: (v: number) => void;
  masteringEnabled: boolean;
  setMasteringEnabled: (v: boolean) => void;
  masteringReference: string;
  setMasteringReference: (v: string) => void;
  timbreReference: boolean;
  setTimbreReference: (v: boolean) => void;

  // PP-VAE re-encode
  ppVaeReencode: boolean;
  setPpVaeReencode: (v: boolean) => void;
  ppVaeBlend: number;
  setPpVaeBlend: (v: number) => void;

  // ── Derived ──
  /** Assemble all engine params for a generation request */
  getGlobalParams: () => Partial<GenerationParams>;
}

// ── Context ──────────────────────────────────────────────────────────────────

const GlobalParamsCtx = createContext<GlobalParams | null>(null);

export function useGlobalParams(): GlobalParams {
  const ctx = useContext(GlobalParamsCtx);
  if (!ctx) throw new Error('useGlobalParams must be used inside <GlobalParamsProvider>');
  return ctx;
}

// ── Provider ─────────────────────────────────────────────────────────────────

export const GlobalParamsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Models
  const [ditModel, setDitModel] = usePersistedState('hs-ditModel', '');
  const [lmModel, setLmModel] = usePersistedState('hs-lmModel', '');
  const [vaeModel, setVaeModel] = usePersistedState('hs-vaeModel', '');
  const [embeddingModel, setEmbeddingModel] = usePersistedState('hs-embeddingModel', '');

  // Adapters
  const [adapter, setAdapter] = usePersistedState('hs-adapter', '');
  const [adapterScale, setAdapterScale] = usePersistedState('hs-adapterScale', 1.0);
  const [adapterMode, setAdapterMode] = usePersistedState('hs-adapterMode', 'runtime');
  const [adapterGroupScales, setAdapterGroupScales] = usePersistedState<AdapterGroupScales>('hs-adapterGroupScales', {
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0,
  });
  const [adapterFolder, setAdapterFolder] = usePersistedState('hs-adapterFolder', '');
  const [advancedAdapters, setAdvancedAdapters] = usePersistedState('hs-advancedAdapters', false);
  const [adaptersOpen, setAdaptersOpen] = usePersistedState('hs-adaptersOpen', false);

  // Generation settings
  const [inferenceSteps, setInferenceSteps] = usePersistedState('hs-inferenceSteps', 12);
  const [guidanceScale, setGuidanceScale] = usePersistedState('hs-guidanceScale', 9.0);
  const [shift, setShift] = usePersistedState('hs-shift', 3.0);
  const [inferMethod, setInferMethod] = usePersistedState('hs-inferMethod', 'euler');
  const [scheduler, setScheduler] = usePersistedState('hs-scheduler', 'linear');
  const [guidanceMode, setGuidanceMode] = usePersistedState('hs-guidanceMode', 'apg');
  const [seed, setSeed] = usePersistedState('hs-seed', -1);
  const [randomSeed, setRandomSeed] = usePersistedState('hs-randomSeed', true);
  const [batchSize, setBatchSize] = usePersistedState('hs-batchSize', 1);

  // Solver sub-params
  const [storkSubsteps, setStorkSubsteps] = usePersistedState('hs-storkSubsteps', 10);
  const [beatStability, setBeatStability] = usePersistedState('hs-beatStability', 0.25);
  const [frequencyDamping, setFrequencyDamping] = usePersistedState('hs-frequencyDamping', 0.4);
  const [temporalSmoothing, setTemporalSmoothing] = usePersistedState('hs-temporalSmoothing', 0.13);

  // Guidance sub-params
  const [apgMomentum, setApgMomentum] = usePersistedState('hs-apgMomentum', 0.75);
  const [apgNormThreshold, setApgNormThreshold] = usePersistedState('hs-apgNormThreshold', 2.5);

  // DCW
  const [dcwEnabled, setDcwEnabled] = usePersistedState('hs-dcwEnabled', false);
  const [dcwMode, setDcwMode] = usePersistedState('hs-dcwMode', 'double');
  const [dcwScaler, setDcwScaler] = usePersistedState('hs-dcwScaler', 0.2);
  const [dcwHighScaler, setDcwHighScaler] = usePersistedState('hs-dcwHighScaler', 0.2);

  // Latent post-processing
  const [latentShift, setLatentShift] = usePersistedState('hs-latentShift', 0.0);
  const [latentRescale, setLatentRescale] = usePersistedState('hs-latentRescale', 1.0);
  const [customTimesteps, setCustomTimesteps] = usePersistedState('hs-customTimesteps', '');

  // Post-VAE spectral denoiser
  const [denoiseStrength, setDenoiseStrength] = usePersistedState('hs-denoiseStrength', 0.0);
  const [denoiseSmoothing, setDenoiseSmoothing] = usePersistedState('hs-denoiseSmoothing', 0.7);
  const [denoiseMix, setDenoiseMix] = usePersistedState('hs-denoiseMix', 0.25);

  // Duration buffer / Auto-trim (disabled by default)
  const [autoTrimEnabled, setAutoTrimEnabled] = usePersistedState('hs-autoTrimEnabled', false);
  const [durationBuffer, setDurationBuffer] = usePersistedState('hs-durationBuffer', 15);
  const [autoTrimFadeMs, setAutoTrimFadeMs] = usePersistedState('hs-autoTrimFadeMs', 2000);

  // LM / Thinking
  const [skipLm, setSkipLm] = usePersistedState('hs-skipLm', false);
  const [useCotCaption, setUseCotCaption] = usePersistedState('hs-useCotCaption', true);
  const [lmTemperature, setLmTemperature] = usePersistedState('hs-lmTemperature', 0.8);
  const [lmCfgScale, setLmCfgScale] = usePersistedState('hs-lmCfgScale', 2.2);
  const [lmTopK, setLmTopK] = usePersistedState('hs-lmTopK', 0);
  const [lmTopP, setLmTopP] = usePersistedState('hs-lmTopP', 0.92);
  const [lmNegativePrompt, setLmNegativePrompt] = usePersistedState('hs-lmNegativePrompt', 'NO USER INPUT');
  const [lmCodesStrength, setLmCodesStrength] = usePersistedState('hs-lmCodesStrength', 1.0);

  // Post-processing — master toggle + individual stages
  const [postProcessingEnabled, setPostProcessingEnabled] = usePersistedState('hs-postProcessingEnabled', true);
  const [spectralLifterEnabled, setSpectralLifterEnabled] = usePersistedState('hs-spectralLifterEnabled', false);
  const [slDenoiseStrength, setSlDenoiseStrength] = usePersistedState('hs-slDenoiseStrength', 0.3);
  const [slNoiseFloor, setSlNoiseFloor] = usePersistedState('hs-slNoiseFloor', 0.1);
  const [slHfMix, setSlHfMix] = usePersistedState('hs-slHfMix', 0.0);
  const [slTransientBoost, setSlTransientBoost] = usePersistedState('hs-slTransientBoost', 0.0);
  const [slShimmerReduction, setSlShimmerReduction] = usePersistedState('hs-slShimmerReduction', 6.0);
  const [masteringEnabled, setMasteringEnabled] = usePersistedState('hs-masteringEnabled', false);
  const [masteringReference, setMasteringReference] = usePersistedState('hs-masteringReference', '');
  const [timbreReference, setTimbreReference] = usePersistedState('hs-timbreReference', false);

  // PP-VAE re-encode (spectral cleanup via post-processing VAE)
  const [ppVaeReencode, setPpVaeReencode] = usePersistedState('hs-ppVaeReencode', false);
  const [ppVaeBlend, setPpVaeBlend] = usePersistedState('hs-ppVaeBlend', 0.0);

  // Trigger word settings — read from shared settings (same key as App.tsx)
  const [settings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

  // Assemble all global params for generation
  const getGlobalParams = useCallback((): Partial<GenerationParams> => {
    // Compute trigger word from adapter filename
    const triggerWord = settings.triggerUseFilename && adapter
      ? (adapter.split(/[\\/]/).pop()?.replace(/\.safetensors$/i, '') || '')
      : '';

    return {
      // Models
      ditModel,
      lmModel,
      vaeModel,
      embeddingModel,

      // Adapter
      loraPath: adapter,
      loraScale: adapterScale,
      adapterGroupScales: adapter ? adapterGroupScales : undefined,
      adapterMode: adapter ? adapterMode : 'merge',

      // Trigger word
      triggerWord: triggerWord || undefined,
      triggerPlacement: triggerWord ? settings.triggerPlacement : undefined,

      // Generation
      inferenceSteps,
      guidanceScale,
      shift,
      inferMethod,
      scheduler,
      guidanceMode,
      seed,
      randomSeed,
      batchSize,

      // Solver sub-params (conditional)
      storkSubsteps: (inferMethod === 'stork2' || inferMethod === 'stork4') ? storkSubsteps : undefined,
      beatStability: inferMethod === 'jkass_fast' ? beatStability : undefined,
      frequencyDamping: inferMethod === 'jkass_fast' ? frequencyDamping : undefined,
      temporalSmoothing: inferMethod === 'jkass_fast' ? temporalSmoothing : undefined,

      // Guidance sub-params (conditional)
      apgMomentum: guidanceMode === 'apg' ? apgMomentum : undefined,
      apgNormThreshold: guidanceMode === 'apg' ? apgNormThreshold : undefined,

      // LM
      skipLm,
      useCotCaption,
      lmTemperature,
      lmCfgScale,
      lmTopK,
      lmTopP,
      lmNegativePrompt,

      // LM Codes Strength — fraction of DiT steps guided by LM codes
      audioCoverStrength: (!skipLm && lmCodesStrength < 1.0) ? lmCodesStrength : undefined,

      // Post-processing — master toggle sent to server to gate the entire chain
      // All PP stages also gated client-side per field below
      postProcessingEnabled,
      spectralLifterEnabled: postProcessingEnabled ? spectralLifterEnabled : false,
      slDenoiseStrength: (postProcessingEnabled && spectralLifterEnabled) ? slDenoiseStrength : undefined,
      slNoiseFloor: (postProcessingEnabled && spectralLifterEnabled) ? slNoiseFloor : undefined,
      slHfMix: (postProcessingEnabled && spectralLifterEnabled) ? slHfMix : undefined,
      slTransientBoost: (postProcessingEnabled && spectralLifterEnabled) ? slTransientBoost : undefined,
      slShimmerReduction: (postProcessingEnabled && spectralLifterEnabled) ? slShimmerReduction : undefined,
      masteringEnabled: postProcessingEnabled ? masteringEnabled : false,
      masteringReference: (postProcessingEnabled && masteringEnabled) ? masteringReference : undefined,
      timbreReference: (postProcessingEnabled && masteringEnabled && timbreReference && masteringReference) ? true : undefined,

      // DCW
      dcwEnabled,
      dcwMode: dcwEnabled ? dcwMode : undefined,
      dcwScaler: dcwEnabled ? dcwScaler * 0.05 : undefined,
      dcwHighScaler: (dcwEnabled && dcwMode === 'double') ? dcwHighScaler * 0.02 : undefined,

      // Latent post-processing
      latentShift: latentShift !== 0 ? latentShift : undefined,
      latentRescale: latentRescale !== 1 ? latentRescale : undefined,
      customTimesteps: customTimesteps || undefined,

      // Post-VAE spectral denoiser
      denoiseStrength: denoiseStrength > 0 ? denoiseStrength : undefined,
      denoiseSmoothing: denoiseStrength > 0 ? denoiseSmoothing : undefined,
      denoiseMix: denoiseStrength > 0 ? denoiseMix : undefined,

      // Duration buffer / auto-trim
      autoTrimEnabled: autoTrimEnabled || undefined,
      durationBuffer: autoTrimEnabled ? durationBuffer : undefined,
      autoTrimFadeMs: autoTrimEnabled ? autoTrimFadeMs : undefined,

      // PP-VAE re-encode
      ppVaeReencode: (postProcessingEnabled && ppVaeReencode) || undefined,
      ppVaeBlend: (postProcessingEnabled && ppVaeReencode && ppVaeBlend > 0) ? ppVaeBlend : undefined,
    };
  }, [
    ditModel, lmModel, vaeModel, embeddingModel,
    adapter, adapterScale, adapterMode, adapterGroupScales,
    inferenceSteps, guidanceScale, shift, inferMethod, scheduler, guidanceMode,
    seed, randomSeed, batchSize,
    storkSubsteps, beatStability, frequencyDamping, temporalSmoothing,
    apgMomentum, apgNormThreshold,
    dcwEnabled, dcwMode, dcwScaler, dcwHighScaler,
    latentShift, latentRescale, customTimesteps,
    denoiseStrength, denoiseSmoothing, denoiseMix,
    autoTrimEnabled, durationBuffer, autoTrimFadeMs,
    skipLm, useCotCaption, lmTemperature, lmCfgScale, lmTopK, lmTopP, lmNegativePrompt, lmCodesStrength,
     spectralLifterEnabled, slDenoiseStrength, slNoiseFloor, slHfMix, slTransientBoost, slShimmerReduction,
    masteringEnabled, masteringReference, timbreReference,
    ppVaeReencode, ppVaeBlend,
    postProcessingEnabled,
    settings,
  ]);

  const value = useMemo<GlobalParams>(() => ({
    // Models
    ditModel, setDitModel, lmModel, setLmModel, vaeModel, setVaeModel, embeddingModel, setEmbeddingModel,
    // Adapters
    adapter, setAdapter, adapterScale, setAdapterScale,
    adapterMode, setAdapterMode, adapterGroupScales, setAdapterGroupScales,
    adapterFolder, setAdapterFolder, advancedAdapters, setAdvancedAdapters,
    adaptersOpen, setAdaptersOpen,
    // Generation
    inferenceSteps, setInferenceSteps, guidanceScale, setGuidanceScale,
    shift, setShift, inferMethod, setInferMethod, scheduler, setScheduler,
    guidanceMode, setGuidanceMode, seed, setSeed, randomSeed, setRandomSeed,
    batchSize, setBatchSize,
    // Solver sub-params
    storkSubsteps, setStorkSubsteps, beatStability, setBeatStability,
    frequencyDamping, setFrequencyDamping, temporalSmoothing, setTemporalSmoothing,
    // Guidance sub-params
    apgMomentum, setApgMomentum, apgNormThreshold, setApgNormThreshold,
    // DCW
    dcwEnabled, setDcwEnabled, dcwMode, setDcwMode,
    dcwScaler, setDcwScaler, dcwHighScaler, setDcwHighScaler,
    // Latent post-processing
    latentShift, setLatentShift, latentRescale, setLatentRescale,
    customTimesteps, setCustomTimesteps,
    // Denoiser
    denoiseStrength, setDenoiseStrength, denoiseSmoothing, setDenoiseSmoothing,
    denoiseMix, setDenoiseMix,
    // Duration buffer / auto-trim
    autoTrimEnabled, setAutoTrimEnabled, durationBuffer, setDurationBuffer,
    autoTrimFadeMs, setAutoTrimFadeMs,
    // LM
    skipLm, setSkipLm, useCotCaption, setUseCotCaption,
    lmTemperature, setLmTemperature, lmCfgScale, setLmCfgScale,
    lmTopK, setLmTopK, lmTopP, setLmTopP,
    lmNegativePrompt, setLmNegativePrompt,
    lmCodesStrength, setLmCodesStrength,
    // Post-processing — master toggle + Spectral Lifter (native C++)
    postProcessingEnabled, setPostProcessingEnabled,
    spectralLifterEnabled, setSpectralLifterEnabled,
    slDenoiseStrength, setSlDenoiseStrength,
    slNoiseFloor, setSlNoiseFloor,
    slHfMix, setSlHfMix,
    slTransientBoost, setSlTransientBoost,
    slShimmerReduction, setSlShimmerReduction,
    masteringEnabled, setMasteringEnabled,
    masteringReference, setMasteringReference,
    timbreReference, setTimbreReference,
    // PP-VAE
    ppVaeReencode, setPpVaeReencode,
    ppVaeBlend, setPpVaeBlend,
    // Derived
    getGlobalParams,
  }), [
    ditModel, setDitModel, lmModel, setLmModel, vaeModel, setVaeModel, embeddingModel, setEmbeddingModel,
    adapter, setAdapter, adapterScale, setAdapterScale,
    adapterMode, setAdapterMode, adapterGroupScales, setAdapterGroupScales,
    adapterFolder, setAdapterFolder, advancedAdapters, setAdvancedAdapters,
    adaptersOpen, setAdaptersOpen,
    inferenceSteps, setInferenceSteps, guidanceScale, setGuidanceScale,
    shift, setShift, inferMethod, setInferMethod, scheduler, setScheduler,
    guidanceMode, setGuidanceMode, seed, setSeed, randomSeed, setRandomSeed,
    batchSize, setBatchSize,
    storkSubsteps, setStorkSubsteps, beatStability, setBeatStability,
    frequencyDamping, setFrequencyDamping, temporalSmoothing, setTemporalSmoothing,
    apgMomentum, setApgMomentum, apgNormThreshold, setApgNormThreshold,
    dcwEnabled, setDcwEnabled, dcwMode, setDcwMode,
    dcwScaler, setDcwScaler, dcwHighScaler, setDcwHighScaler,
    latentShift, setLatentShift, latentRescale, setLatentRescale,
    customTimesteps, setCustomTimesteps,
    denoiseStrength, setDenoiseStrength, denoiseSmoothing, setDenoiseSmoothing,
    denoiseMix, setDenoiseMix,
    autoTrimEnabled, setAutoTrimEnabled, durationBuffer, setDurationBuffer,
    autoTrimFadeMs, setAutoTrimFadeMs,
    skipLm, setSkipLm, useCotCaption, setUseCotCaption,
    lmTemperature, setLmTemperature, lmCfgScale, setLmCfgScale,
    lmTopK, setLmTopK, lmTopP, setLmTopP,
    lmNegativePrompt, setLmNegativePrompt, lmCodesStrength, setLmCodesStrength,
    spectralLifterEnabled, setSpectralLifterEnabled,
    postProcessingEnabled, setPostProcessingEnabled,
    slDenoiseStrength, setSlDenoiseStrength,
    slNoiseFloor, setSlNoiseFloor,
    slHfMix, setSlHfMix,
    slTransientBoost, setSlTransientBoost,
    slShimmerReduction, setSlShimmerReduction,
    masteringEnabled, setMasteringEnabled,
    masteringReference, setMasteringReference,
    timbreReference, setTimbreReference,
    ppVaeReencode, setPpVaeReencode,
    ppVaeBlend, setPpVaeBlend,
    getGlobalParams,
  ]);

  return (
    <GlobalParamsCtx.Provider value={value}>
      {children}
    </GlobalParamsCtx.Provider>
  );
};
