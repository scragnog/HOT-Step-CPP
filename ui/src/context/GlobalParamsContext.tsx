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

  // ── Mastering ──
  masteringEnabled: boolean;
  setMasteringEnabled: (v: boolean) => void;
  masteringReference: string;
  setMasteringReference: (v: string) => void;
  timbreReference: boolean;
  setTimbreReference: (v: boolean) => void;

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
  const [dcwMode, setDcwMode] = usePersistedState('hs-dcwMode', 'low');
  const [dcwScaler, setDcwScaler] = usePersistedState('hs-dcwScaler', 1.0);
  const [dcwHighScaler, setDcwHighScaler] = usePersistedState('hs-dcwHighScaler', 0.0);

  // LM / Thinking
  const [skipLm, setSkipLm] = usePersistedState('hs-skipLm', false);
  const [useCotCaption, setUseCotCaption] = usePersistedState('hs-useCotCaption', true);
  const [lmTemperature, setLmTemperature] = usePersistedState('hs-lmTemperature', 0.8);
  const [lmCfgScale, setLmCfgScale] = usePersistedState('hs-lmCfgScale', 2.2);
  const [lmTopK, setLmTopK] = usePersistedState('hs-lmTopK', 0);
  const [lmTopP, setLmTopP] = usePersistedState('hs-lmTopP', 0.92);
  const [lmNegativePrompt, setLmNegativePrompt] = usePersistedState('hs-lmNegativePrompt', 'NO USER INPUT');

  // Mastering
  const [masteringEnabled, setMasteringEnabled] = usePersistedState('hs-masteringEnabled', false);
  const [masteringReference, setMasteringReference] = usePersistedState('hs-masteringReference', '');
  const [timbreReference, setTimbreReference] = usePersistedState('hs-timbreReference', false);

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

      // Mastering
      masteringEnabled,
      masteringReference: masteringEnabled ? masteringReference : undefined,
      timbreReference: (masteringEnabled && timbreReference && masteringReference) ? true : undefined,

      // DCW
      dcwEnabled,
      dcwMode: dcwEnabled ? dcwMode : undefined,
      dcwScaler: dcwEnabled ? dcwScaler * 0.02 : undefined,
      dcwHighScaler: (dcwEnabled && dcwMode === 'double') ? dcwHighScaler * 0.02 : undefined,
    };
  }, [
    ditModel, lmModel, vaeModel,
    adapter, adapterScale, adapterMode, adapterGroupScales,
    inferenceSteps, guidanceScale, shift, inferMethod, scheduler, guidanceMode,
    seed, randomSeed, batchSize,
    storkSubsteps, beatStability, frequencyDamping, temporalSmoothing,
    apgMomentum, apgNormThreshold,
    dcwEnabled, dcwMode, dcwScaler, dcwHighScaler,
    skipLm, useCotCaption, lmTemperature, lmCfgScale, lmTopK, lmTopP, lmNegativePrompt,
    masteringEnabled, masteringReference, timbreReference,
    settings,
  ]);

  const value = useMemo<GlobalParams>(() => ({
    // Models
    ditModel, setDitModel, lmModel, setLmModel, vaeModel, setVaeModel,
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
    // LM
    skipLm, setSkipLm, useCotCaption, setUseCotCaption,
    lmTemperature, setLmTemperature, lmCfgScale, setLmCfgScale,
    lmTopK, setLmTopK, lmTopP, setLmTopP,
    lmNegativePrompt, setLmNegativePrompt,
    // Mastering
    masteringEnabled, setMasteringEnabled,
    masteringReference, setMasteringReference,
    timbreReference, setTimbreReference,
    // Derived
    getGlobalParams,
  }), [
    ditModel, setDitModel, lmModel, setLmModel, vaeModel, setVaeModel,
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
    skipLm, setSkipLm, useCotCaption, setUseCotCaption,
    lmTemperature, setLmTemperature, lmCfgScale, setLmCfgScale,
    lmTopK, setLmTopK, lmTopP, setLmTopP,
    lmNegativePrompt, setLmNegativePrompt,
    masteringEnabled, setMasteringEnabled,
    masteringReference, setMasteringReference,
    timbreReference, setTimbreReference,
    getGlobalParams,
  ]);

  return (
    <GlobalParamsCtx.Provider value={value}>
      {children}
    </GlobalParamsCtx.Provider>
  );
};
