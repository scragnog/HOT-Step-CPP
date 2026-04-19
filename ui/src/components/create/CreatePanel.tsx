// CreatePanel.tsx — The composition panel that wires sections together
//
// Ported to Tailwind styling, matching hot-step-9000's panel layout.
// Each section is a separate module.

import React, { useEffect, useRef } from 'react';
import { Zap, Loader2, Download, Upload } from 'lucide-react';
import { usePersistedState } from '../../hooks/usePersistedState';
import { ContentSection } from './ContentSection';
import { MetadataSection } from './MetadataSection';
import { GenerationSettings } from './GenerationSettings';
import { ModelSelector } from './ModelSelector';
import { AdaptersAccordion } from './AdaptersAccordion';
import { MasteringSection } from './MasteringSection';
import { DEFAULT_SETTINGS, type AppSettings } from '../settings/SettingsPanel';
import type { GenerationParams, Song } from '../../types';

interface CreatePanelProps {
  onGenerate: (params: GenerationParams) => void;
  isGenerating: boolean;
  reuseData?: { song: Song; timestamp: number } | null;
}

export const CreatePanel: React.FC<CreatePanelProps> = ({ onGenerate, isGenerating, reuseData }) => {
  // Content
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
  const [adaptersOpen, setAdaptersOpen] = usePersistedState('hs-adaptersOpen', false);

  // Trigger word settings — read from shared settings (same key as App.tsx)
  const [settings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

  // Mastering
  const [masteringEnabled, setMasteringEnabled] = usePersistedState('hs-masteringEnabled', false);
  const [masteringReference, setMasteringReference] = usePersistedState('hs-masteringReference', '');

  // Reuse data
  useEffect(() => {
    if (!reuseData) return;
    const gp = reuseData.song.generationParams;
    if (!gp) return;

    if (reuseData.song.caption || gp.caption) setCaption(reuseData.song.caption || gp.caption || '');
    if (reuseData.song.lyrics || gp.lyrics) setLyrics(reuseData.song.lyrics || gp.lyrics || '');
    if (reuseData.song.style || gp.style) setCaption(reuseData.song.style || gp.style || '');
    if (gp.bpm) setBpm(gp.bpm);
    if (gp.keyScale) setKeyScale(gp.keyScale);
    if (gp.timeSignature) setTimeSignature(gp.timeSignature);
    if (gp.duration) setDuration(typeof gp.duration === 'string' ? parseFloat(gp.duration) : gp.duration);
    if (gp.inferenceSteps) setInferenceSteps(gp.inferenceSteps);
    if (gp.guidanceScale !== undefined) setGuidanceScale(gp.guidanceScale);
    if (gp.seed !== undefined) setSeed(gp.seed);
  }, [reuseData?.timestamp]);

  // ── JSON Import / Export ────────────────────────────────────────────────
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = () => {
    const preset: Record<string, unknown> = {
      _format: 'hot-step-preset',
      _version: 1,
      caption, lyrics, instrumental,
      bpm, duration, keyScale, timeSignature, vocalLanguage,
      inferenceSteps, guidanceScale, shift,
      inferMethod, scheduler, guidanceMode,
      seed, randomSeed, batchSize,
      useCotCaption, skipLm,
      lmTemperature, lmCfgScale, lmTopK, lmTopP, lmNegativePrompt,
      ditModel, lmModel, vaeModel,
      adapter, adapterScale, adapterGroupScales, adapterMode,
      masteringEnabled, masteringReference,
      // Solver sub-params
      storkSubsteps, beatStability, frequencyDamping, temporalSmoothing,
      // Guidance sub-params
      apgMomentum, apgNormThreshold,
    };
    const blob = new Blob([JSON.stringify(preset, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const slug = (caption || 'preset').slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
    a.download = `${slug}_params.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const p = JSON.parse(reader.result as string);
        // Content
        if (p.caption !== undefined) setCaption(p.caption);
        if (p.lyrics !== undefined) setLyrics(p.lyrics);
        if (p.instrumental !== undefined) setInstrumental(p.instrumental);
        // Metadata
        if (p.bpm !== undefined) setBpm(p.bpm);
        if (p.duration !== undefined) setDuration(p.duration);
        if (p.keyScale !== undefined) setKeyScale(p.keyScale);
        if (p.timeSignature !== undefined) setTimeSignature(p.timeSignature);
        if (p.vocalLanguage !== undefined) setVocalLanguage(p.vocalLanguage);
        // Generation
        if (p.inferenceSteps !== undefined) setInferenceSteps(p.inferenceSteps);
        if (p.guidanceScale !== undefined) setGuidanceScale(p.guidanceScale);
        if (p.shift !== undefined) setShift(p.shift);
        if (p.inferMethod !== undefined) setInferMethod(p.inferMethod);
        if (p.scheduler !== undefined) setScheduler(p.scheduler);
        if (p.guidanceMode !== undefined) setGuidanceMode(p.guidanceMode);
        if (p.seed !== undefined) setSeed(p.seed);
        if (p.randomSeed !== undefined) setRandomSeed(p.randomSeed);
        if (p.batchSize !== undefined) setBatchSize(p.batchSize);
        if (p.useCotCaption !== undefined) setUseCotCaption(p.useCotCaption);
        if (p.skipLm !== undefined) setSkipLm(p.skipLm);
        // LM
        if (p.lmTemperature !== undefined) setLmTemperature(p.lmTemperature);
        if (p.lmCfgScale !== undefined) setLmCfgScale(p.lmCfgScale);
        if (p.lmTopK !== undefined) setLmTopK(p.lmTopK);
        if (p.lmTopP !== undefined) setLmTopP(p.lmTopP);
        if (p.lmNegativePrompt !== undefined) setLmNegativePrompt(p.lmNegativePrompt);
        // Models
        if (p.ditModel !== undefined) setDitModel(p.ditModel);
        if (p.lmModel !== undefined) setLmModel(p.lmModel);
        if (p.vaeModel !== undefined) setVaeModel(p.vaeModel);
        if (p.adapter !== undefined) setAdapter(p.adapter);
        if (p.adapterScale !== undefined) setAdapterScale(p.adapterScale);
        if (p.adapterGroupScales !== undefined) setAdapterGroupScales(p.adapterGroupScales);
        if (p.adapterMode !== undefined) setAdapterMode(p.adapterMode);
        // Mastering
        if (p.masteringEnabled !== undefined) setMasteringEnabled(p.masteringEnabled);
        if (p.masteringReference !== undefined) setMasteringReference(p.masteringReference);
        // Solver sub-params
        if (p.storkSubsteps !== undefined) setStorkSubsteps(p.storkSubsteps);
        if (p.beatStability !== undefined) setBeatStability(p.beatStability);
        if (p.frequencyDamping !== undefined) setFrequencyDamping(p.frequencyDamping);
        if (p.temporalSmoothing !== undefined) setTemporalSmoothing(p.temporalSmoothing);
        // Guidance sub-params
        if (p.apgMomentum !== undefined) setApgMomentum(p.apgMomentum);
        if (p.apgNormThreshold !== undefined) setApgNormThreshold(p.apgNormThreshold);
      } catch (err) {
        console.error('[Preset Import] Invalid JSON:', err);
      }
    };
    reader.readAsText(file);
    // Reset input so re-importing the same file works
    e.target.value = '';
  };

  const handleGenerate = () => {
    // Compute trigger word from adapter filename
    const triggerWord = settings.triggerUseFilename && adapter
      ? (adapter.split(/[\\/]/).pop()?.replace(/\.safetensors$/i, '') || '')
      : '';

    const params: GenerationParams = {
      caption,
      lyrics: instrumental ? '[Instrumental]' : lyrics,
      instrumental,
      bpm,
      duration,
      keyScale,
      timeSignature,
      vocalLanguage,
      inferenceSteps,
      guidanceScale,
      shift,
      inferMethod,
      scheduler,
      guidanceMode,
      seed,
      randomSeed,
      batchSize,
      lmTemperature,
      lmCfgScale,
      lmTopK,
      lmTopP,
      lmNegativePrompt,
      useCotCaption,
      skipLm,
      ditModel,
      lmModel,
      vaeModel,
      loraPath: adapter,
      loraScale: adapterScale,
      adapterGroupScales: adapter ? adapterGroupScales : undefined,
      adapterMode: adapter ? adapterMode : 'merge',
      triggerWord: triggerWord || undefined,
      triggerPlacement: triggerWord ? settings.triggerPlacement : undefined,
      taskType: 'text2music',
      masteringEnabled,
      masteringReference: masteringEnabled ? masteringReference : undefined,
      // Solver sub-params (only when relevant solver is active)
      storkSubsteps: (inferMethod === 'stork2' || inferMethod === 'stork4') ? storkSubsteps : undefined,
      beatStability: inferMethod === 'jkass_fast' ? beatStability : undefined,
      frequencyDamping: inferMethod === 'jkass_fast' ? frequencyDamping : undefined,
      temporalSmoothing: inferMethod === 'jkass_fast' ? temporalSmoothing : undefined,
      // Guidance sub-params (only when APG is active)
      apgMomentum: guidanceMode === 'apg' ? apgMomentum : undefined,
      apgNormThreshold: guidanceMode === 'apg' ? apgNormThreshold : undefined,
    };
    onGenerate(params);
  };

  return (
    <div className="h-full flex flex-col bg-zinc-50 dark:bg-suno-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Create</h2>
        <div className="flex items-center gap-1.5">
          <button onClick={handleExport} title="Export preset"
            className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-emerald-400 transition-colors">
            <Upload size={14} />
          </button>
          <button onClick={() => fileInputRef.current?.click()} title="Import preset"
            className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-sky-400 transition-colors">
            <Download size={14} />
          </button>
          <input ref={fileInputRef} type="file" accept=".json" className="hidden"
            onChange={handleImport} />
          <span className="text-xs text-zinc-500 font-medium ml-1">text2music</span>
        </div>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto hide-scrollbar px-4 py-3 space-y-1">
        <ContentSection
          caption={caption} onCaptionChange={setCaption}
          lyrics={lyrics} onLyricsChange={setLyrics}
          instrumental={instrumental} onInstrumentalChange={setInstrumental}
        />

        <MetadataSection
          bpm={bpm} onBpmChange={setBpm}
          keyScale={keyScale} onKeyScaleChange={setKeyScale}
          timeSignature={timeSignature} onTimeSignatureChange={setTimeSignature}
          duration={duration} onDurationChange={setDuration}
          vocalLanguage={vocalLanguage} onVocalLanguageChange={setVocalLanguage}
        />

        <ModelSelector
          ditModel={ditModel} onDitModelChange={setDitModel}
          lmModel={lmModel} onLmModelChange={setLmModel}
          vaeModel={vaeModel} onVaeModelChange={setVaeModel}
        />

        <AdaptersAccordion
          isOpen={adaptersOpen}
          onToggle={() => setAdaptersOpen(!adaptersOpen)}
          advancedAdapters={advancedAdapters}
          onAdvancedAdaptersChange={setAdvancedAdapters}
          adapter={adapter}
          onAdapterChange={setAdapter}
          adapterScale={adapterScale}
          onAdapterScaleChange={setAdapterScale}
          adapterMode={adapterMode}
          onAdapterModeChange={setAdapterMode}
          adapterGroupScales={adapterGroupScales}
          onAdapterGroupScalesChange={setAdapterGroupScales}
          adapterFolder={adapterFolder}
          onAdapterFolderChange={setAdapterFolder}
          triggerUseFilename={settings.triggerUseFilename}
          triggerPlacement={settings.triggerPlacement}
        />

        <GenerationSettings
          inferenceSteps={inferenceSteps} onInferenceStepsChange={setInferenceSteps}
          guidanceScale={guidanceScale} onGuidanceScaleChange={setGuidanceScale}
          shift={shift} onShiftChange={setShift}
          inferMethod={inferMethod} onInferMethodChange={setInferMethod}
          scheduler={scheduler} onSchedulerChange={setScheduler}
          guidanceMode={guidanceMode} onGuidanceModeChange={setGuidanceMode}
          seed={seed} onSeedChange={setSeed}
          randomSeed={randomSeed} onRandomSeedChange={setRandomSeed}
          batchSize={batchSize} onBatchSizeChange={setBatchSize}
          skipLm={skipLm} onSkipLmChange={setSkipLm}
          lmTemperature={lmTemperature} onLmTemperatureChange={setLmTemperature}
          lmCfgScale={lmCfgScale} onLmCfgScaleChange={setLmCfgScale}
          lmTopK={lmTopK} onLmTopKChange={setLmTopK}
          lmTopP={lmTopP} onLmTopPChange={setLmTopP}
          lmNegativePrompt={lmNegativePrompt} onLmNegativePromptChange={setLmNegativePrompt}
          useCotCaption={useCotCaption} onUseCotCaptionChange={setUseCotCaption}
          storkSubsteps={storkSubsteps} onStorkSubstepsChange={setStorkSubsteps}
          beatStability={beatStability} onBeatStabilityChange={setBeatStability}
          frequencyDamping={frequencyDamping} onFrequencyDampingChange={setFrequencyDamping}
          temporalSmoothing={temporalSmoothing} onTemporalSmoothingChange={setTemporalSmoothing}
          apgMomentum={apgMomentum} onApgMomentumChange={setApgMomentum}
          apgNormThreshold={apgNormThreshold} onApgNormThresholdChange={setApgNormThreshold}
        />

        <MasteringSection
          masteringEnabled={masteringEnabled}
          onMasteringEnabledChange={setMasteringEnabled}
          masteringReference={masteringReference}
          onMasteringReferenceChange={setMasteringReference}
        />
      </div>

      {/* Generate button */}
      <div className="px-4 py-3 border-t border-zinc-200 dark:border-white/5">
        <button
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white font-semibold text-sm transition-all duration-200 hover:shadow-lg hover:shadow-pink-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={handleGenerate}
          disabled={isGenerating || (!caption.trim() && !lyrics.trim() && !instrumental)}
        >
          {isGenerating ? (
            <>
              <Loader2 size={18} className="spinner" />
              Generating...
            </>
          ) : (
            <>
              <Zap size={18} />
              Generate
            </>
          )}
        </button>
      </div>
    </div>
  );
};
