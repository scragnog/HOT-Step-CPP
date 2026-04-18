// CreatePanel.tsx — The composition panel that wires sections together
//
// Ported to Tailwind styling, matching hot-step-9000's panel layout.
// Each section is a separate module.

import React, { useEffect } from 'react';
import { Zap, Loader2 } from 'lucide-react';
import { usePersistedState } from '../../hooks/usePersistedState';
import { ContentSection } from './ContentSection';
import { MetadataSection } from './MetadataSection';
import { GenerationSettings } from './GenerationSettings';
import { ModelSelector } from './ModelSelector';
import { MasteringSection } from './MasteringSection';
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

  const handleGenerate = () => {
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
      taskType: 'text2music',
      masteringEnabled,
      masteringReference: masteringEnabled ? masteringReference : undefined,
    };
    onGenerate(params);
  };

  return (
    <div className="h-full flex flex-col bg-zinc-50 dark:bg-suno-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Create</h2>
        <span className="text-xs text-zinc-500 font-medium">text2music</span>
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
          adapter={adapter} onAdapterChange={setAdapter}
          adapterScale={adapterScale} onAdapterScaleChange={setAdapterScale}
          adapterGroupScales={adapterGroupScales} onAdapterGroupScalesChange={setAdapterGroupScales}
          adapterMode={adapterMode} onAdapterModeChange={setAdapterMode}
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
