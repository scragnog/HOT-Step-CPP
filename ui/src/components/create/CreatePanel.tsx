// CreatePanel.tsx — The composition panel that wires sections together
//
// This is NOT a monolith — each section is a separate module.
// This file just composes them and manages the form state.

import React from 'react';
import { usePersistedState } from '../../hooks/usePersistedState';
import { ContentSection } from './ContentSection';
import { MetadataSection } from './MetadataSection';
import { GenerationSettings } from './GenerationSettings';
import { ModelSelector } from './ModelSelector';
import type { GenerationParams } from '../../types';
import './CreatePanel.css';

interface CreatePanelProps {
  onGenerate: (params: GenerationParams) => void;
  isGenerating: boolean;
}

export const CreatePanel: React.FC<CreatePanelProps> = ({ onGenerate, isGenerating }) => {
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
  const [inferMethod, setInferMethod] = usePersistedState('hs-inferMethod', 'ode');
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
  const [adapter, setAdapter] = usePersistedState('hs-adapter', '');
  const [adapterScale, setAdapterScale] = usePersistedState('hs-adapterScale', 1.0);

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
      loraPath: adapter,
      loraScale: adapterScale,
      taskType: 'text2music',
    };
    onGenerate(params);
  };

  return (
    <div className="create-panel">
      <div className="create-panel-header">
        <h2 className="create-panel-title">Create</h2>
      </div>

      <div className="create-panel-scroll">
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
          adapter={adapter} onAdapterChange={setAdapter}
          adapterScale={adapterScale} onAdapterScaleChange={setAdapterScale}
        />

        <GenerationSettings
          inferenceSteps={inferenceSteps} onInferenceStepsChange={setInferenceSteps}
          guidanceScale={guidanceScale} onGuidanceScaleChange={setGuidanceScale}
          shift={shift} onShiftChange={setShift}
          inferMethod={inferMethod} onInferMethodChange={setInferMethod}
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
      </div>

      {/* Generate button */}
      <div className="create-panel-footer">
        <button
          className="btn btn-generate btn-lg w-full"
          onClick={handleGenerate}
          disabled={isGenerating || (!caption.trim() && !lyrics.trim() && !instrumental)}
        >
          {isGenerating ? (
            <>
              <span className="spinner">⟳</span>
              Generating...
            </>
          ) : (
            <>⚡ Generate</>
          )}
        </button>
      </div>
    </div>
  );
};
