// GenerationSettings.tsx — DiT + LM inference parameters
//
// Accordion containing: steps, guidance, shift, seed, solver, batch, LM params.

import React from 'react';
import { Accordion } from '../shared/Accordion';
import { Slider } from '../shared/Slider';

interface GenerationSettingsProps {
  // DiT
  inferenceSteps: number; onInferenceStepsChange: (v: number) => void;
  guidanceScale: number; onGuidanceScaleChange: (v: number) => void;
  shift: number; onShiftChange: (v: number) => void;
  inferMethod: string; onInferMethodChange: (v: string) => void;
  // Seed
  seed: number; onSeedChange: (v: number) => void;
  randomSeed: boolean; onRandomSeedChange: (v: boolean) => void;
  // Batch
  batchSize: number; onBatchSizeChange: (v: number) => void;
  // LM
  skipLm: boolean; onSkipLmChange: (v: boolean) => void;
  lmTemperature: number; onLmTemperatureChange: (v: number) => void;
  lmCfgScale: number; onLmCfgScaleChange: (v: number) => void;
  lmTopK: number; onLmTopKChange: (v: number) => void;
  lmTopP: number; onLmTopPChange: (v: number) => void;
  lmNegativePrompt: string; onLmNegativePromptChange: (v: string) => void;
  useCotCaption: boolean; onUseCotCaptionChange: (v: boolean) => void;
}

export const GenerationSettings: React.FC<GenerationSettingsProps> = (props) => {
  return (
    <Accordion title="Generation Settings" defaultOpen={false}>
      {/* DiT settings */}
      <Slider label="Inference Steps" value={props.inferenceSteps}
        onChange={props.onInferenceStepsChange} min={1} max={100} step={1} showInput />

      <Slider label="Guidance Scale" value={props.guidanceScale}
        onChange={props.onGuidanceScaleChange} min={0} max={20} step={0.1} showInput />

      <Slider label="Shift" value={props.shift}
        onChange={props.onShiftChange} min={0} max={10} step={0.1} showInput />

      {/* Solver */}
      <div>
        <label className="label">Solver</label>
        <select className="input select" value={props.inferMethod}
          onChange={e => props.onInferMethodChange(e.target.value)}>
          <option value="ode">ODE (Euler)</option>
          <option value="sde">SDE (Stochastic)</option>
        </select>
      </div>

      {/* Seed */}
      <div>
        <div className="flex items-center justify-between">
          <label className="label" style={{ marginBottom: 0 }}>Seed</label>
          <label className="toggle-row" style={{ fontSize: 'var(--text-xs)' }}>
            <input type="checkbox" checked={props.randomSeed}
              onChange={e => props.onRandomSeedChange(e.target.checked)} />
            <span>Random</span>
          </label>
        </div>
        {!props.randomSeed && (
          <input type="number" className="input" value={props.seed}
            onChange={e => props.onSeedChange(parseInt(e.target.value) || -1)} />
        )}
      </div>

      {/* Batch */}
      <Slider label="Batch Size" value={props.batchSize}
        onChange={props.onBatchSizeChange} min={1} max={9} step={1} />

      <div className="divider" />

      {/* LM Toggle — the big switch */}
      <div className="flex items-center justify-between">
        <div style={{ fontWeight: 600, fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
          LM / Thinking
        </div>
        <label className="toggle-row" style={{ fontSize: 'var(--text-xs)' }}>
          <input type="checkbox" checked={!props.skipLm}
            onChange={e => props.onSkipLmChange(!e.target.checked)} />
          <span>{props.skipLm ? 'Disabled' : 'Enabled'}</span>
        </label>
      </div>

      {/* LM Settings — dimmed when skipped */}
      {!props.skipLm && (
        <>
          {/* CoT Caption */}
          <label className="toggle-row">
            <input type="checkbox" checked={props.useCotCaption}
              onChange={e => props.onUseCotCaptionChange(e.target.checked)} />
            <span>Chain-of-Thought Caption</span>
          </label>

          <Slider label="Temperature" value={props.lmTemperature}
            onChange={props.onLmTemperatureChange} min={0} max={2} step={0.01} showInput />

          <Slider label="CFG Scale" value={props.lmCfgScale}
            onChange={props.onLmCfgScaleChange} min={0} max={10} step={0.1} showInput />

          <Slider label="Top-K" value={props.lmTopK}
            onChange={props.onLmTopKChange} min={0} max={200} step={1} showInput />

          <Slider label="Top-P" value={props.lmTopP}
            onChange={props.onLmTopPChange} min={0} max={1} step={0.01} showInput />

          <div>
            <label className="label">Negative Prompt</label>
            <input className="input" value={props.lmNegativePrompt}
              onChange={e => props.onLmNegativePromptChange(e.target.value)}
              placeholder="NO USER INPUT" />
          </div>
        </>
      )}
    </Accordion>
  );
};

