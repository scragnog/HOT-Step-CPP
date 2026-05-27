// useAssistantActions.ts — Hook for applying LLM-suggested settings changes
//
// Maps action field names to GlobalParamsContext setters, provides
// preview diffs and one-click apply functionality.

import { useGlobalParams } from '../context/GlobalParamsContext';
import { writePersistedState } from './usePersistedState';
import type { AssistantAction } from '../services/assistantApi';

export interface ActionDiff {
  key: string;
  label: string;
  from: any;
  to: any;
}

/** Human-readable labels for parameter field names */
const LABEL_MAP: Record<string, string> = {
  // Models
  ditModel: 'DiT Model',
  lmModel: 'LM Model',
  vaeModel: 'VAE Model',
  // Adapter
  adapter: 'Adapter',
  adapterScale: 'Adapter Scale',
  adapterMode: 'Adapter Mode',
  // Generation
  inferMethod: 'Solver',
  inferenceSteps: 'Steps',
  guidanceScale: 'Guidance Scale',
  shift: 'Shift',
  scheduler: 'Scheduler',
  guidanceMode: 'Guidance Mode',
  seed: 'Seed',
  randomSeed: 'Random Seed',
  batchSize: 'Batch Size',
  // Solver sub-params
  storkSubsteps: 'STORK Substeps',
  beatStability: 'Beat Stability',
  frequencyDamping: 'Frequency Damping',
  temporalSmoothing: 'Temporal Smoothing',
  // Guidance sub-params
  apgMomentum: 'APG Momentum',
  apgNormThreshold: 'APG Norm Threshold',
  // LM
  skipLm: 'Skip LM',
  useCotCaption: 'CoT Caption',
  lmTemperature: 'LM Temperature',
  lmCfgScale: 'LM CFG Scale',
  lmTopK: 'LM Top-K',
  lmTopP: 'LM Top-P',
  lmNegativePrompt: 'LM Negative Prompt',
  // Post-processing
  postProcessingEnabled: 'Post-Processing',
  spectralLifterEnabled: 'Spectral Lifter',
  slDenoiseStrength: 'SL Denoise',
  slNoiseFloor: 'SL Noise Floor',
  slHfMix: 'SL HF Mix',
  slTransientBoost: 'SL Transient Boost',
  slShimmerReduction: 'SL Shimmer Reduction',
  masteringEnabled: 'Mastering',
  masteringReference: 'Mastering Reference',
  // Denoiser
  denoiseStrength: 'Denoise Strength',
  denoiseSmoothing: 'Denoise Smoothing',
  denoiseMix: 'Denoise Mix',
  // PP-VAE
  ppVaeReencode: 'PP-VAE Re-encode',
  ppVaeBlend: 'PP-VAE Blend',
  // DCW
  dcwEnabled: 'DCW Enabled',
  dcwMode: 'DCW Mode',
  dcwLowScaler: 'DCW Low Scaler',
  dcwHighScaler: 'DCW High Scaler',
  // Latent
  latentShift: 'Latent Shift',
  latentRescale: 'Latent Rescale',
  customTimesteps: 'Custom Timesteps',
  // Duration
  autoTrimEnabled: 'Auto-Trim',
  durationBuffer: 'Duration Buffer',
  autoTrimFadeMs: 'Trim Fade',
  // Content fields (localStorage)
  caption: 'Style Description',
  lyrics: 'Lyrics',
  instrumental: 'Instrumental',
  bpm: 'BPM',
  duration: 'Duration (s)',
  keyScale: 'Key / Scale',
  timeSignature: 'Time Signature',
  vocalLanguage: 'Vocal Language',
};

export function useAssistantActions() {
  const gp = useGlobalParams();

  // Map field names → setter functions
  const setterMap: Record<string, (v: any) => void> = {
    // Models
    ditModel: gp.setDitModel,
    lmModel: gp.setLmModel,
    vaeModel: gp.setVaeModel,
    // Adapter
    adapter: gp.setAdapter,
    adapterScale: gp.setAdapterScale,
    adapterMode: gp.setAdapterMode,
    adapterGroupScales: gp.setAdapterGroupScales,
    // Generation
    inferMethod: gp.setInferMethod,
    inferenceSteps: gp.setInferenceSteps,
    guidanceScale: gp.setGuidanceScale,
    shift: gp.setShift,
    scheduler: gp.setScheduler,
    guidanceMode: gp.setGuidanceMode,
    seed: gp.setSeed,
    randomSeed: gp.setRandomSeed,
    batchSize: gp.setBatchSize,
    // Solver sub-params
    storkSubsteps: gp.setStorkSubsteps,
    beatStability: gp.setBeatStability,
    frequencyDamping: gp.setFrequencyDamping,
    temporalSmoothing: gp.setTemporalSmoothing,
    // Guidance sub-params
    apgMomentum: gp.setApgMomentum,
    apgNormThreshold: gp.setApgNormThreshold,
    // LM
    skipLm: gp.setSkipLm,
    useCotCaption: gp.setUseCotCaption,
    lmTemperature: gp.setLmTemperature,
    lmCfgScale: gp.setLmCfgScale,
    lmTopK: gp.setLmTopK,
    lmTopP: gp.setLmTopP,
    lmNegativePrompt: gp.setLmNegativePrompt,
    // Post-processing
    postProcessingEnabled: gp.setPostProcessingEnabled,
    spectralLifterEnabled: gp.setSpectralLifterEnabled,
    slDenoiseStrength: gp.setSlDenoiseStrength,
    slNoiseFloor: gp.setSlNoiseFloor,
    slHfMix: gp.setSlHfMix,
    slTransientBoost: gp.setSlTransientBoost,
    slShimmerReduction: gp.setSlShimmerReduction,
    masteringEnabled: gp.setMasteringEnabled,
    masteringReference: gp.setMasteringReference,
    // Denoiser
    denoiseStrength: gp.setDenoiseStrength,
    denoiseSmoothing: gp.setDenoiseSmoothing,
    denoiseMix: gp.setDenoiseMix,
    // PP-VAE
    ppVaeReencode: gp.setPpVaeReencode,
    ppVaeBlend: gp.setPpVaeBlend,
    // DCW
    dcwEnabled: gp.setDcwEnabled,
    dcwMode: gp.setDcwMode,
    dcwLowScaler: gp.setDcwLowScaler,
    dcwHighScaler: gp.setDcwHighScaler,
    // Latent
    latentShift: gp.setLatentShift,
    latentRescale: gp.setLatentRescale,
    customTimesteps: gp.setCustomTimesteps,
    // Duration
    autoTrimEnabled: gp.setAutoTrimEnabled,
    durationBuffer: gp.setDurationBuffer,
    autoTrimFadeMs: gp.setAutoTrimFadeMs,
    // Content fields (stored in localStorage, synced via StorageEvent)
    caption: (v: any) => writePersistedState('hs-caption', v),
    lyrics: (v: any) => writePersistedState('hs-lyrics', v),
    instrumental: (v: any) => writePersistedState('hs-instrumental', v),
    bpm: (v: any) => writePersistedState('hs-bpm', v),
    duration: (v: any) => writePersistedState('hs-duration', v),
    keyScale: (v: any) => writePersistedState('hs-keyScale', v),
    timeSignature: (v: any) => writePersistedState('hs-timeSignature', v),
    vocalLanguage: (v: any) => writePersistedState('hs-vocalLanguage', v),
  };

  /** Apply a list of actions to the global params */
  function applyActions(actions: AssistantAction[]): number {
    let applied = 0;
    for (const action of actions) {
      const setter = setterMap[action.set];
      if (setter) {
        setter(action.value);
        applied++;
      } else {
        console.warn(`[Assistant] Unknown action field: ${action.set}`);
      }
    }
    return applied;
  }

  /** Preview actions as a diff array (before applying) */
  function previewActions(actions: AssistantAction[]): ActionDiff[] {
    const engineParams = gp.getGlobalParams();
    // Also read content fields from localStorage for preview diffs
    const readLS = <T,>(key: string, fallback: T): T => {
      try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
    };
    const current: Record<string, any> = {
      ...engineParams,
      caption: readLS('hs-caption', ''),
      lyrics: readLS('hs-lyrics', ''),
      instrumental: readLS('hs-instrumental', false),
      bpm: readLS('hs-bpm', 0),
      duration: readLS('hs-duration', -1),
      keyScale: readLS('hs-keyScale', ''),
      timeSignature: readLS('hs-timeSignature', ''),
      vocalLanguage: readLS('hs-vocalLanguage', 'en'),
    };
    return actions
      .filter(a => a.set in setterMap)
      .map(a => {
        const fromVal = current[a.set];
        // For lyrics, truncate the preview to keep the diff card readable
        const fmt = (v: any) => typeof v === 'string' && v.length > 80 ? v.slice(0, 77) + '...' : v;
        return {
          key: a.set,
          label: LABEL_MAP[a.set] || a.set,
          from: fmt(fromVal),
          to: fmt(a.value),
        };
      });
  }

  return { applyActions, previewActions };
}
