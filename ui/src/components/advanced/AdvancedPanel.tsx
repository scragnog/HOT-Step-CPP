import React from 'react';
import { X } from 'lucide-react';
import { ModelSelector } from '../create/ModelSelector';
import { GenerationSettings } from '../create/GenerationSettings';
import { MasteringSection } from '../create/MasteringSection';
import { useCreateState } from '../../hooks/useCreateState';

interface AdvancedPanelProps {
  onClose?: () => void;
}

export const AdvancedPanel: React.FC<AdvancedPanelProps> = ({ onClose }) => {
  const {
    inferenceSteps, setInferenceSteps,
    guidanceScale, setGuidanceScale,
    shift, setShift,
    inferMethod, setInferMethod,
    scheduler, setScheduler,
    guidanceMode, setGuidanceMode,
    seed, setSeed,
    randomSeed, setRandomSeed,
    batchSize, setBatchSize,
    skipLm, setSkipLm,
    lmTemperature, setLmTemperature,
    lmCfgScale, setLmCfgScale,
    lmTopK, setLmTopK,
    lmTopP, setLmTopP,
    lmNegativePrompt, setLmNegativePrompt,
    useCotCaption, setUseCotCaption,
    storkSubsteps, setStorkSubsteps,
    beatStability, setBeatStability,
    frequencyDamping, setFrequencyDamping,
    temporalSmoothing, setTemporalSmoothing,
    apgMomentum, setApgMomentum,
    apgNormThreshold, setApgNormThreshold,
    ditModel, setDitModel,
    lmModel, setLmModel,
    vaeModel, setVaeModel,
    masteringEnabled, setMasteringEnabled,
    masteringReference, setMasteringReference,
    timbreReference, setTimbreReference,
  } = useCreateState();

  return (
    <div className="h-full flex flex-col bg-zinc-50 dark:bg-suno-panel">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Advanced Options</h2>
        {onClose && (
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-red-400 transition-colors">
            <X size={16} />
          </button>
        )}
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto hide-scrollbar px-4 py-3 space-y-1">
        <ModelSelector
          ditModel={ditModel} onDitModelChange={setDitModel}
          lmModel={lmModel} onLmModelChange={setLmModel}
          vaeModel={vaeModel} onVaeModelChange={setVaeModel}
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
          timbreReference={timbreReference}
          onTimbreReferenceChange={setTimbreReference}
        />
      </div>
    </div>
  );
};
