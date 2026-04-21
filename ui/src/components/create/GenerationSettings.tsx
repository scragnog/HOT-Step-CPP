// GenerationSettings.tsx — DiT + LM inference parameters
// Ported to Tailwind accordion styling matching hot-step-9000.

import React from 'react';
import { RotateCcw } from 'lucide-react';
import { Slider } from '../shared/Slider';
import { ToggleSwitch } from '../shared/ToggleSwitch';

const selectClasses = "w-full px-3 py-2 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer";
const inputClasses = "w-full px-3 py-2 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

interface GenerationSettingsProps {
  // DiT
  inferenceSteps: number; onInferenceStepsChange: (v: number) => void;
  guidanceScale: number; onGuidanceScaleChange: (v: number) => void;
  shift: number; onShiftChange: (v: number) => void;
  inferMethod: string; onInferMethodChange: (v: string) => void;
  scheduler: string; onSchedulerChange: (v: string) => void;
  guidanceMode: string; onGuidanceModeChange: (v: string) => void;
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
  // Solver sub-params
  storkSubsteps: number; onStorkSubstepsChange: (v: number) => void;
  beatStability: number; onBeatStabilityChange: (v: number) => void;
  frequencyDamping: number; onFrequencyDampingChange: (v: number) => void;
  temporalSmoothing: number; onTemporalSmoothingChange: (v: number) => void;
  // Guidance sub-params
  apgMomentum: number; onApgMomentumChange: (v: number) => void;
  apgNormThreshold: number; onApgNormThresholdChange: (v: number) => void;
}

export const GenerationSettings: React.FC<GenerationSettingsProps> = (props) => {
  // Resolve scheduler dropdown value from the composite string representation
  const schedulerKey = props.scheduler.startsWith('composite') ? 'composite'
    : props.scheduler.startsWith('beta:') ? 'beta'
    : props.scheduler.startsWith('power:') ? 'power'
    : props.scheduler;

  return (
    <div className="space-y-1">
      {/* DiT Settings Section */}
      <div className="pt-3 border-t border-white/5">
        <div className="px-3 py-2.5">
          <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">Generation Settings</span>
        </div>

        <div className="px-3 pb-3 space-y-3">
          <Slider label="Inference Steps" value={props.inferenceSteps}
            onChange={props.onInferenceStepsChange} min={1} max={100} step={1} showInput />

          <Slider label="Guidance Scale" value={props.guidanceScale}
            onChange={props.onGuidanceScaleChange} min={0} max={20} step={0.1} showInput />

          {/* Shift with Auto toggle */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Shift</label>
              <button
                onClick={() => {
                  if (props.shift === -1) {
                    props.onShiftChange(3.0); // restore default
                  } else {
                    props.onShiftChange(-1);
                  }
                }}
                className={`text-[10px] font-bold px-2 py-0.5 rounded-full transition-all ${
                  props.shift === -1
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'bg-zinc-800 text-zinc-500 border border-white/5 hover:text-zinc-300 hover:border-white/10'
                }`}
              >
                Auto
              </button>
            </div>
            {props.shift === -1 ? (
              <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-cyan-500/5 border border-cyan-500/10 text-xs text-cyan-400/80">
                <span>Adaptive shift based on duration &amp; step count</span>
              </div>
            ) : (
              <Slider label="" value={props.shift}
                onChange={props.onShiftChange} min={0} max={10} step={0.1} showInput />
            )}
          </div>

          {/* Solver */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Solver</label>
            <select className={selectClasses} value={props.inferMethod}
              onChange={e => props.onInferMethodChange(e.target.value)}>
              <optgroup label="── Single Evaluation (1 NFE) ──">
                <option value="euler">Euler (ODE)</option>
                <option value="dpm2m">DPM++ 2M</option>
                <option value="dpm3m">DPM++ 3M</option>
                <option value="dpm2m_ada">DPM++ 2M Adaptive</option>
                <option value="jkass_fast">JKASS Fast</option>
                <option value="stork2">STORK 2</option>
                <option value="stork4">STORK 4</option>
                <option value="sde">SDE (Stochastic)</option>
              </optgroup>
              <optgroup label="── Multi Evaluation ──">
                <option value="heun">Heun (2 NFE)</option>
                <option value="jkass_quality">JKASS Quality (2 NFE)</option>
                <option value="rk4">RK4 (4 NFE)</option>
                <option value="rk5">RK5 (6 NFE)</option>
                <option value="dopri5">DOPRI5 Adaptive (7+ NFE)</option>
                <option value="dop853">DOP853 (13 NFE)</option>
              </optgroup>
            </select>
          </div>

          {/* ── JKASS Fast Sub-Controls ── */}
          {props.inferMethod === 'jkass_fast' && (
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 space-y-3 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">JKASS Fast Controls</span>
                <button type="button" onClick={() => {
                  props.onBeatStabilityChange(0.25);
                  props.onFrequencyDampingChange(0.4);
                  props.onTemporalSmoothingChange(0.13);
                }} className="flex items-center gap-1 text-[10px] text-amber-400 hover:text-amber-300 transition-colors">
                  <RotateCcw size={10} /> Reset
                </button>
              </div>
              <Slider label="Beat Stability" value={props.beatStability}
                onChange={props.onBeatStabilityChange} min={0} max={1} step={0.01} showInput />
              <Slider label="Frequency Damping" value={props.frequencyDamping}
                onChange={props.onFrequencyDampingChange} min={0} max={5} step={0.1} showInput />
              <Slider label="Temporal Smoothing" value={props.temporalSmoothing}
                onChange={props.onTemporalSmoothingChange} min={0} max={1} step={0.01} showInput />
            </div>
          )}

          {/* ── STORK Sub-Steps ── */}
          {(props.inferMethod === 'stork2' || props.inferMethod === 'stork4') && (
            <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3 space-y-3 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-cyan-400 uppercase tracking-wider">STORK Controls</span>
                <button type="button" onClick={() => props.onStorkSubstepsChange(10)}
                  className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 transition-colors">
                  <RotateCcw size={10} /> Reset
                </button>
              </div>
              <Slider label="Sub-Steps" value={props.storkSubsteps}
                onChange={props.onStorkSubstepsChange} min={2} max={50} step={1} showInput />
              <p className="text-[10px] text-zinc-500">Chebyshev sub-iterations per step. Higher = more stability work (default: 10)</p>
            </div>
          )}

          {/* Scheduler */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Schedule</label>
            <select className={selectClasses} value={schedulerKey}
              onChange={e => {
                const v = e.target.value;
                if (v === 'beta') props.onSchedulerChange('beta:0.50:0.70');
                else if (v === 'power') props.onSchedulerChange('power:2.00');
                else if (v === 'composite') props.onSchedulerChange('composite:bong_tangent+linear:0.50:0.50');
                else props.onSchedulerChange(v);
              }}>
              <option value="linear">Linear (Default)</option>
              <option value="beta57">Beta 57 (RES4LYF)</option>
              <option value="beta">Beta (Custom)</option>
              <option value="cosine">Cosine</option>
              <option value="power">Power</option>
              <option value="ddim_uniform">DDIM Uniform (Log-SNR)</option>
              <option value="sgm_uniform">SGM / Karras (ρ=7)</option>
              <option value="bong_tangent">Tangent (Front-loaded)</option>
              <option value="linear_quadratic">Linear-Quadratic</option>
              <option value="composite">Composite (2-Stage)</option>
            </select>
          </div>

          {/* Guidance Mode */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Guidance</label>
            <select className={selectClasses} value={props.guidanceMode}
              onChange={e => props.onGuidanceModeChange(e.target.value)}>
              <option value="apg">APG (Default)</option>
              <option value="cfg_pp">CFG++</option>
              <option value="dynamic_cfg">Dynamic CFG</option>
              <option value="rescaled_cfg">Rescaled CFG</option>
            </select>
          </div>

          {/* Seed */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Seed</label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input type="checkbox" checked={props.randomSeed}
                  onChange={e => props.onRandomSeedChange(e.target.checked)}
                  className="rounded border-zinc-600 bg-zinc-800 text-pink-500 focus:ring-pink-500/20" />
                <span className="text-xs text-zinc-500">Random</span>
              </label>
            </div>
            {!props.randomSeed && (
              <input type="number" className={inputClasses} value={props.seed}
                onChange={e => props.onSeedChange(parseInt(e.target.value) || -1)} />
            )}
          </div>

          {/* Batch */}
          <Slider label="Batch Size" value={props.batchSize}
            onChange={props.onBatchSizeChange} min={1} max={9} step={1} />
        </div>
      </div>

      {/* LM Settings Section */}
      <div className="pt-3 border-t border-white/5">
        <div className="flex items-center justify-between px-3 py-2.5">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">LM / Thinking</span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
              props.skipLm ? 'bg-zinc-700 text-zinc-400' : 'bg-purple-500/20 text-purple-400'
            }`}>
              {props.skipLm ? 'OFF' : 'ON'}
            </span>
          </div>
        </div>

        <div className="px-3 pb-3 space-y-3">
          {/* LM Toggle */}
          <ToggleSwitch
            checked={!props.skipLm}
            onChange={v => props.onSkipLmChange(!v)}
            label="Enable LM Conditioning"
          />

          {!props.skipLm && (
            <>
              {/* CoT Caption */}
              <ToggleSwitch
                checked={props.useCotCaption}
                onChange={props.onUseCotCaptionChange}
                label="Chain-of-Thought Caption"
              />

              <Slider label="Temperature" value={props.lmTemperature}
                onChange={props.onLmTemperatureChange} min={0} max={2} step={0.01} showInput />

              <Slider label="CFG Scale" value={props.lmCfgScale}
                onChange={props.onLmCfgScaleChange} min={0} max={10} step={0.1} showInput />

              <Slider label="Top-K" value={props.lmTopK}
                onChange={props.onLmTopKChange} min={0} max={200} step={1} showInput />

              <Slider label="Top-P" value={props.lmTopP}
                onChange={props.onLmTopPChange} min={0} max={1} step={0.01} showInput />

              <div>
                <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Negative Prompt</label>
                <input className={inputClasses} value={props.lmNegativePrompt}
                  onChange={e => props.onLmNegativePromptChange(e.target.value)}
                  placeholder="NO USER INPUT" />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
