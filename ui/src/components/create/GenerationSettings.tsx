// GenerationSettings.tsx — DiT + LM inference parameters
// Ported to Tailwind accordion styling matching hot-step-9000.

import React, { useState } from 'react';
import { ChevronDown, RotateCcw } from 'lucide-react';
import { Slider } from '../shared/Slider';

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
  const [ditOpen, setDitOpen] = useState(false);
  const [lmOpen, setLmOpen] = useState(false);

  // Resolve scheduler dropdown value from the composite string representation
  const schedulerKey = props.scheduler.startsWith('composite') ? 'composite'
    : props.scheduler.startsWith('beta:') ? 'beta'
    : props.scheduler.startsWith('power:') ? 'power'
    : props.scheduler;

  return (
    <div className="space-y-1 pt-3 border-t border-white/5">
      {/* DiT Settings Accordion */}
      <button
        onClick={() => setDitOpen(!ditOpen)}
        className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/5 transition-colors"
      >
        <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Generation Settings</span>
        <ChevronDown size={14} className={`text-zinc-500 transition-transform duration-200 ${ditOpen ? 'rotate-180' : ''}`} />
      </button>

      {ditOpen && (
        <div className="px-3 pb-3 space-y-3">
          <Slider label="Inference Steps" value={props.inferenceSteps}
            onChange={props.onInferenceStepsChange} min={1} max={100} step={1} showInput />

          <Slider label="Guidance Scale" value={props.guidanceScale}
            onChange={props.onGuidanceScaleChange} min={0} max={20} step={0.1} showInput />

          <Slider label="Shift" value={props.shift}
            onChange={props.onShiftChange} min={0} max={10} step={0.1} showInput />

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

          {/* ── Beta (Custom) Sub-Controls ── */}
          {props.scheduler.startsWith('beta:') && (() => {
            const parts = props.scheduler.split(':');
            const alpha = parseFloat(parts[1] || '0.5');
            const betaParam = parseFloat(parts[2] || '0.7');
            const updateBeta = (a: number, b: number) => {
              props.onSchedulerChange(`beta:${a.toFixed(2)}:${b.toFixed(2)}`);
            };
            return (
              <div className="rounded-xl border border-teal-500/20 bg-teal-500/5 p-3 space-y-3 transition-all">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-semibold text-teal-400 uppercase tracking-wider">Beta Distribution</span>
                  <button type="button" onClick={() => updateBeta(0.5, 0.7)}
                    className="flex items-center gap-1 text-[10px] text-teal-400 hover:text-teal-300 transition-colors">
                    <RotateCcw size={10} /> Reset
                  </button>
                </div>
                <Slider label="Alpha (α)" value={alpha}
                  onChange={v => updateBeta(v, betaParam)} min={0.1} max={2.0} step={0.05} showInput />
                <Slider label="Beta (β)" value={betaParam}
                  onChange={v => updateBeta(alpha, v)} min={0.1} max={2.0} step={0.05} showInput />
                <p className="text-[10px] text-zinc-500">Lower α = more density at edges. Lower β = front-loaded (structural focus).</p>
              </div>
            );
          })()}

          {/* ── Power Sub-Controls ── */}
          {props.scheduler.startsWith('power:') && (() => {
            const exponent = parseFloat(props.scheduler.split(':')[1] || '2.0');
            return (
              <div className="rounded-xl border border-orange-500/20 bg-orange-500/5 p-3 space-y-3 transition-all">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-semibold text-orange-400 uppercase tracking-wider">Power Law</span>
                  <button type="button" onClick={() => props.onSchedulerChange('power:2.00')}
                    className="flex items-center gap-1 text-[10px] text-orange-400 hover:text-orange-300 transition-colors">
                    <RotateCcw size={10} /> Reset
                  </button>
                </div>
                <Slider label="Exponent" value={exponent}
                  onChange={v => props.onSchedulerChange(`power:${v.toFixed(2)}`)} min={0.25} max={4.0} step={0.05} showInput />
                <p className="text-[10px] text-zinc-500">p&gt;1 = front-loaded (structure), p=1 = linear, p&lt;1 = back-loaded (detail).</p>
              </div>
            );
          })()}

          {/* ── Composite Sub-Controls ── */}
          {props.scheduler.startsWith('composite') && (() => {
            const parts = props.scheduler.split(':');
            const schedulerPair = (parts[1] || 'bong_tangent+linear').split('+');
            const stageA = schedulerPair[0] || 'bong_tangent';
            const stageB = schedulerPair[1] || 'linear';
            const crossover = parseFloat(parts[2] || '0.5');
            const split = parseFloat(parts[3] || '0.5');
            const update = (a: string, b: string, c: number, s: number) => {
              props.onSchedulerChange(`composite:${a}+${b}:${c.toFixed(2)}:${s.toFixed(2)}`);
            };
            return (
              <div className="rounded-xl border border-purple-500/20 bg-purple-500/5 p-3 space-y-3 transition-all">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-semibold text-purple-400 uppercase tracking-wider">Composite (2-Stage)</span>
                  <button type="button" onClick={() => update('bong_tangent', 'linear', 0.5, 0.5)}
                    className="flex items-center gap-1 text-[10px] text-purple-400 hover:text-purple-300 transition-colors">
                    <RotateCcw size={10} /> Reset
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[10px] text-purple-400 mb-1">Stage A</label>
                    <select className={selectClasses} value={stageA}
                      onChange={e => update(e.target.value, stageB, crossover, split)}>
                      <option value="linear">Linear</option>
                      <option value="beta57">Beta 57</option>
                      <option value="cosine">Cosine</option>
                      <option value="ddim_uniform">DDIM</option>
                      <option value="sgm_uniform">SGM</option>
                      <option value="bong_tangent">Tangent</option>
                      <option value="linear_quadratic">Lin-Quad</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[10px] text-purple-400 mb-1">Stage B</label>
                    <select className={selectClasses} value={stageB}
                      onChange={e => update(stageA, e.target.value, crossover, split)}>
                      <option value="linear">Linear</option>
                      <option value="beta57">Beta 57</option>
                      <option value="cosine">Cosine</option>
                      <option value="ddim_uniform">DDIM</option>
                      <option value="sgm_uniform">SGM</option>
                      <option value="bong_tangent">Tangent</option>
                      <option value="linear_quadratic">Lin-Quad</option>
                    </select>
                  </div>
                </div>
                <Slider label="Crossover" value={crossover}
                  onChange={v => update(stageA, stageB, v, split)} min={0.1} max={0.9} step={0.05} showInput />
                <Slider label="Split" value={split}
                  onChange={v => update(stageA, stageB, crossover, v)} min={0.1} max={0.9} step={0.05} showInput />
              </div>
            );
          })()}

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

          {/* ── APG Sub-Controls ── */}
          {props.guidanceMode === 'apg' && (
            <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-3 space-y-3 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider">APG Parameters</span>
                <button type="button" onClick={() => {
                  props.onApgMomentumChange(0.75);
                  props.onApgNormThresholdChange(2.5);
                }} className="flex items-center gap-1 text-[10px] text-blue-400 hover:text-blue-300 transition-colors">
                  <RotateCcw size={10} /> Reset
                </button>
              </div>
              <Slider label="Momentum" value={props.apgMomentum}
                onChange={props.onApgMomentumChange} min={0} max={1} step={0.01} showInput />
              <Slider label="Norm Threshold" value={props.apgNormThreshold}
                onChange={props.onApgNormThresholdChange} min={0} max={10} step={0.1} showInput />
              <p className="text-[10px] text-zinc-500">Momentum smooths guidance across steps. Norm threshold clips gradient magnitude per channel.</p>
            </div>
          )}

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
      )}

      {/* LM Settings Accordion */}
      <button
        onClick={() => setLmOpen(!lmOpen)}
        className="w-full flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">LM / Thinking</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
            props.skipLm ? 'bg-zinc-700 text-zinc-400' : 'bg-purple-500/20 text-purple-400'
          }`}>
            {props.skipLm ? 'OFF' : 'ON'}
          </span>
        </div>
        <ChevronDown size={14} className={`text-zinc-500 transition-transform duration-200 ${lmOpen ? 'rotate-180' : ''}`} />
      </button>

      {lmOpen && (
        <div className="px-3 pb-3 space-y-3">
          {/* LM Toggle */}
          <label className="flex items-center gap-2.5 cursor-pointer">
            <input type="checkbox" checked={!props.skipLm}
              onChange={e => props.onSkipLmChange(!e.target.checked)}
              className="rounded border-zinc-600 bg-zinc-800 text-pink-500 focus:ring-pink-500/20" />
            <span className="text-sm text-zinc-400">Enable LM Conditioning</span>
          </label>

          {!props.skipLm && (
            <>
              {/* CoT Caption */}
              <label className="flex items-center gap-2.5 cursor-pointer">
                <input type="checkbox" checked={props.useCotCaption}
                  onChange={e => props.onUseCotCaptionChange(e.target.checked)}
                  className="rounded border-zinc-600 bg-zinc-800 text-pink-500 focus:ring-pink-500/20" />
                <span className="text-sm text-zinc-400">Chain-of-Thought Caption</span>
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
                <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Negative Prompt</label>
                <input className={inputClasses} value={props.lmNegativePrompt}
                  onChange={e => props.onLmNegativePromptChange(e.target.value)}
                  placeholder="NO USER INPUT" />
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};
