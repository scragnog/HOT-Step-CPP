import React, { createContext, useContext } from 'react';
import { usePersistedState } from '../hooks/usePersistedState';
import { DEFAULT_SETTINGS, type AppSettings } from '../components/settings/SettingsPanel';

const SETTINGS_KEY = 'ace-settings';

interface CreateContextType {
  title: string; setTitle: (v: string) => void;
  caption: string; setCaption: (v: string) => void;
  lyrics: string; setLyrics: (v: string) => void;
  instrumental: boolean; setInstrumental: (v: boolean) => void;
  bpm: number; setBpm: (v: number) => void;
  keyScale: string; setKeyScale: (v: string) => void;
  timeSignature: string; setTimeSignature: (v: string) => void;
  duration: number; setDuration: (v: number) => void;
  vocalLanguage: string; setVocalLanguage: (v: string) => void;
  inferenceSteps: number; setInferenceSteps: (v: number) => void;
  guidanceScale: number; setGuidanceScale: (v: number) => void;
  shift: number; setShift: (v: number) => void;
  inferMethod: string; setInferMethod: (v: string) => void;
  scheduler: string; setScheduler: (v: string) => void;
  guidanceMode: string; setGuidanceMode: (v: string) => void;
  seed: number; setSeed: (v: number) => void;
  randomSeed: boolean; setRandomSeed: (v: boolean) => void;
  batchSize: number; setBatchSize: (v: number) => void;
  useCotCaption: boolean; setUseCotCaption: (v: boolean) => void;
  storkSubsteps: number; setStorkSubsteps: (v: number) => void;
  beatStability: number; setBeatStability: (v: number) => void;
  frequencyDamping: number; setFrequencyDamping: (v: number) => void;
  temporalSmoothing: number; setTemporalSmoothing: (v: number) => void;
  apgMomentum: number; setApgMomentum: (v: number) => void;
  apgNormThreshold: number; setApgNormThreshold: (v: number) => void;
  skipLm: boolean; setSkipLm: (v: boolean) => void;
  lmTemperature: number; setLmTemperature: (v: number) => void;
  lmCfgScale: number; setLmCfgScale: (v: number) => void;
  lmTopK: number; setLmTopK: (v: number) => void;
  lmTopP: number; setLmTopP: (v: number) => void;
  lmNegativePrompt: string; setLmNegativePrompt: (v: string) => void;
  ditModel: string; setDitModel: (v: string) => void;
  lmModel: string; setLmModel: (v: string) => void;
  vaeModel: string; setVaeModel: (v: string) => void;
  adapter: string; setAdapter: (v: string) => void;
  adapterScale: number; setAdapterScale: (v: number) => void;
  adapterGroupScales: any; setAdapterGroupScales: (v: any) => void;
  adapterMode: string; setAdapterMode: (v: string) => void;
  advancedAdapters: boolean; setAdvancedAdapters: (v: boolean) => void;
  adapterFolder: string; setAdapterFolder: (v: string) => void;
  adaptersOpen: boolean; setAdaptersOpen: (v: boolean) => void;
  settings: AppSettings; setLocalSettings: (v: any) => void;
  masteringEnabled: boolean; setMasteringEnabled: (v: boolean) => void;
  masteringReference: string; setMasteringReference: (v: string) => void;
  timbreReference: boolean; setTimbreReference: (v: boolean) => void;
}

const CreateContext = createContext<CreateContextType | undefined>(undefined);

export const CreateProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [title, setTitle] = usePersistedState('hs-title', '');
  const [caption, setCaption] = usePersistedState('hs-caption', '');
  const [lyrics, setLyrics] = usePersistedState('hs-lyrics', '');
  const [instrumental, setInstrumental] = usePersistedState('hs-instrumental', false);

  // Metadata
  const [bpm, setBpm] = usePersistedState('hs-bpm', 0);
  const [keyScale, setKeyScale] = usePersistedState('hs-keyScale', '');
  const [timeSignature, setTimeSignature] = usePersistedState('hs-timeSignature', '');
  const [duration, setDuration] = usePersistedState('hs-duration', -1);
  const [vocalLanguage, setVocalLanguage] = usePersistedState('hs-vocalLanguage', 'en');

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
  const [useCotCaption, setUseCotCaption] = usePersistedState('hs-useCotCaption', true);

  // Solver sub-params
  const [storkSubsteps, setStorkSubsteps] = usePersistedState('hs-storkSubsteps', 10);
  const [beatStability, setBeatStability] = usePersistedState('hs-beatStability', 0.25);
  const [frequencyDamping, setFrequencyDamping] = usePersistedState('hs-frequencyDamping', 0.4);
  const [temporalSmoothing, setTemporalSmoothing] = usePersistedState('hs-temporalSmoothing', 0.13);

  // Guidance sub-params
  const [apgMomentum, setApgMomentum] = usePersistedState('hs-apgMomentum', 0.75);
  const [apgNormThreshold, setApgNormThreshold] = usePersistedState('hs-apgNormThreshold', 2.5);

  // LM toggle
  const [skipLm, setSkipLm] = usePersistedState('hs-skipLm', false);

  // LM settings
  const [lmTemperature, setLmTemperature] = usePersistedState('hs-lmTemperature', 0.8);
  const [lmCfgScale, setLmCfgScale] = usePersistedState('hs-lmCfgScale', 2.2);
  const [lmTopK, setLmTopK] = usePersistedState('hs-lmTopK', 0);
  const [lmTopP, setLmTopP] = usePersistedState('hs-lmTopP', 0.92);
  const [lmNegativePrompt, setLmNegativePrompt] = usePersistedState('hs-lmNegativePrompt', 'NO USER INPUT');

  // Models
  const [ditModel, setDitModel] = usePersistedState('hs-ditModel', '');
  const [lmModel, setLmModel] = usePersistedState('hs-lmModel', '');
  const [vaeModel, setVaeModel] = usePersistedState('hs-vaeModel', '');
  const [adapter, setAdapter] = usePersistedState('hs-adapter', '');
  const [adapterScale, setAdapterScale] = usePersistedState('hs-adapterScale', 1.0);
  const [adapterGroupScales, setAdapterGroupScales] = usePersistedState('hs-adapterGroupScales', {
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0,
  });
  const [adapterMode, setAdapterMode] = usePersistedState('hs-adapterMode', 'runtime');

  // Adapter accordion state
  const [advancedAdapters, setAdvancedAdapters] = usePersistedState('hs-advancedAdapters', false);
  const [adapterFolder, setAdapterFolder] = usePersistedState('hs-adapterFolder', '');
  const [adaptersOpen, setAdaptersOpen] = usePersistedState('hs-adaptersOpen', true);

  // Trigger word settings
  const [settings, setLocalSettings] = usePersistedState<AppSettings>(SETTINGS_KEY, DEFAULT_SETTINGS);

  // Mastering
  const [masteringEnabled, setMasteringEnabled] = usePersistedState('hs-masteringEnabled', false);
  const [masteringReference, setMasteringReference] = usePersistedState('hs-masteringReference', '');
  const [timbreReference, setTimbreReference] = usePersistedState('hs-timbreReference', false);

  return (
    <CreateContext.Provider value={{
      title, setTitle,
      caption, setCaption,
      lyrics, setLyrics,
      instrumental, setInstrumental,
      bpm, setBpm,
      keyScale, setKeyScale,
      timeSignature, setTimeSignature,
      duration, setDuration,
      vocalLanguage, setVocalLanguage,
      inferenceSteps, setInferenceSteps,
      guidanceScale, setGuidanceScale,
      shift, setShift,
      inferMethod, setInferMethod,
      scheduler, setScheduler,
      guidanceMode, setGuidanceMode,
      seed, setSeed,
      randomSeed, setRandomSeed,
      batchSize, setBatchSize,
      useCotCaption, setUseCotCaption,
      storkSubsteps, setStorkSubsteps,
      beatStability, setBeatStability,
      frequencyDamping, setFrequencyDamping,
      temporalSmoothing, setTemporalSmoothing,
      apgMomentum, setApgMomentum,
      apgNormThreshold, setApgNormThreshold,
      skipLm, setSkipLm,
      lmTemperature, setLmTemperature,
      lmCfgScale, setLmCfgScale,
      lmTopK, setLmTopK,
      lmTopP, setLmTopP,
      lmNegativePrompt, setLmNegativePrompt,
      ditModel, setDitModel,
      lmModel, setLmModel,
      vaeModel, setVaeModel,
      adapter, setAdapter,
      adapterScale, setAdapterScale,
      adapterGroupScales, setAdapterGroupScales,
      adapterMode, setAdapterMode,
      advancedAdapters, setAdvancedAdapters,
      adapterFolder, setAdapterFolder,
      adaptersOpen, setAdaptersOpen,
      settings, setLocalSettings,
      masteringEnabled, setMasteringEnabled,
      masteringReference, setMasteringReference,
      timbreReference, setTimbreReference,
    }}>
      {children}
    </CreateContext.Provider>
  );
};

export const useCreateContext = () => {
  const context = useContext(CreateContext);
  if (context === undefined) {
    throw new Error('useCreateContext must be used within a CreateProvider');
  }
  return context;
};
