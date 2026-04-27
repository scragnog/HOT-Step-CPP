// GenerationDropdown.tsx — DiT generation settings for the global param bar
//
// Adapted from the DiT section of create/GenerationSettings.tsx.
// Reads from GlobalParamsContext instead of props.

import React from 'react';
import { RotateCcw, ChevronDown } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { Slider } from '../shared/Slider';
import { ToggleSwitch } from './BarSection';
import { formatScheduler } from './modelLabels';
import { usePersistedState } from '../../hooks/usePersistedState';

const selectClasses = "w-full px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer";
const inputClasses = "w-full px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

export const GenerationDropdown: React.FC = () => {
  const gp = useGlobalParams();
  const [compositeOpen, setCompositeOpen] = usePersistedState('hs-genAccordion-composite', false);
  const [dcwOpen, setDcwOpen] = usePersistedState('hs-genAccordion-dcw', false);
  const [latentOpen, setLatentOpen] = usePersistedState('hs-genAccordion-latent', false);
  const [denoiserOpen, setDenoiserOpen] = usePersistedState('hs-genAccordion-denoiser', false);
  const [autoTrimOpen, setAutoTrimOpen] = usePersistedState('hs-genAccordion-autotrim', false);

  // Resolve scheduler dropdown value from the composite string representation
  const schedulerKey = gp.scheduler.startsWith('composite') ? 'composite'
    : gp.scheduler.startsWith('beta:') ? 'beta'
    : gp.scheduler.startsWith('power:') ? 'power'
    : gp.scheduler;

  return (
    <div className="space-y-3">
      <Slider label="Inference Steps" value={gp.inferenceSteps}
        onChange={gp.setInferenceSteps} min={1} max={100} step={1} showInput />

      <Slider label="Guidance Scale" value={gp.guidanceScale}
        onChange={gp.setGuidanceScale} min={0} max={20} step={0.1} showInput />

      {/* Shift with Auto toggle */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Shift</label>
          <button
            onClick={() => {
              if (gp.shift === -1) {
                gp.setShift(3.0);
              } else {
                gp.setShift(-1);
              }
            }}
            className={`text-[10px] font-bold px-2 py-0.5 rounded-full transition-all ${
              gp.shift === -1
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                : 'bg-zinc-800 text-zinc-500 border border-white/5 hover:text-zinc-300 hover:border-white/10'
            }`}
          >
            Auto
          </button>
        </div>
        {gp.shift === -1 ? (
          <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-cyan-500/5 border border-cyan-500/10 text-xs text-cyan-400/80">
            <span>Adaptive shift based on duration &amp; step count</span>
          </div>
        ) : (
          <Slider label="" value={gp.shift}
            onChange={gp.setShift} min={0} max={10} step={0.1} showInput />
        )}
      </div>

      {/* Solver */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Solver</label>
        <select className={selectClasses} value={gp.inferMethod}
          onChange={e => gp.setInferMethod(e.target.value)}>
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
                <option value="gl2s">Gauss-Legendre 2s (6 NFE)</option>
                <option value="rfsolver">RF-Solver (2 NFE)</option>
          </optgroup>
        </select>
      </div>

      {/* ── JKASS Fast Sub-Controls ── */}
      {gp.inferMethod === 'jkass_fast' && (
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 space-y-3 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">JKASS Fast Controls</span>
            <button type="button" onClick={() => {
              gp.setBeatStability(0.25);
              gp.setFrequencyDamping(0.4);
              gp.setTemporalSmoothing(0.13);
            }} className="flex items-center gap-1 text-[10px] text-amber-400 hover:text-amber-300 transition-colors">
              <RotateCcw size={10} /> Reset
            </button>
          </div>
          <Slider label="Beat Stability" value={gp.beatStability}
            onChange={gp.setBeatStability} min={0} max={1} step={0.01} showInput />
          <Slider label="Frequency Damping" value={gp.frequencyDamping}
            onChange={gp.setFrequencyDamping} min={0} max={5} step={0.1} showInput />
          <Slider label="Temporal Smoothing" value={gp.temporalSmoothing}
            onChange={gp.setTemporalSmoothing} min={0} max={1} step={0.01} showInput />
        </div>
      )}

      {/* ── STORK Sub-Steps ── */}
      {(gp.inferMethod === 'stork2' || gp.inferMethod === 'stork4') && (
        <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3 space-y-3 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-cyan-400 uppercase tracking-wider">STORK Controls</span>
            <button type="button" onClick={() => gp.setStorkSubsteps(10)}
              className="flex items-center gap-1 text-[10px] text-cyan-400 hover:text-cyan-300 transition-colors">
              <RotateCcw size={10} /> Reset
            </button>
          </div>
          <Slider label="Sub-Steps" value={gp.storkSubsteps}
            onChange={gp.setStorkSubsteps} min={2} max={50} step={1} showInput />
          <p className="text-[10px] text-zinc-500">Chebyshev sub-iterations per step. Higher = more stability work (default: 10)</p>
        </div>
      )}

      {/* Scheduler */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Schedule</label>
        <select className={selectClasses} value={schedulerKey}
          onChange={e => {
            const v = e.target.value;
            if (v === 'beta') gp.setScheduler('beta:0.50:0.70');
            else if (v === 'power') gp.setScheduler('power:2.00');
            else if (v === 'composite') gp.setScheduler('composite:bong_tangent+linear:0.50:0.50');
            else gp.setScheduler(v);
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
      {gp.scheduler.startsWith('beta:') && (() => {
        const parts = gp.scheduler.split(':');
        const alpha = parseFloat(parts[1] || '0.5');
        const betaParam = parseFloat(parts[2] || '0.7');
        const updateBeta = (a: number, b: number) => {
          gp.setScheduler(`beta:${a.toFixed(2)}:${b.toFixed(2)}`);
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
      {gp.scheduler.startsWith('power:') && (() => {
        const exponent = parseFloat(gp.scheduler.split(':')[1] || '2.0');
        return (
          <div className="rounded-xl border border-orange-500/20 bg-orange-500/5 p-3 space-y-3 transition-all">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-orange-400 uppercase tracking-wider">Power Law</span>
              <button type="button" onClick={() => gp.setScheduler('power:2.00')}
                className="flex items-center gap-1 text-[10px] text-orange-400 hover:text-orange-300 transition-colors">
                <RotateCcw size={10} /> Reset
              </button>
            </div>
            <Slider label="Exponent" value={exponent}
              onChange={v => gp.setScheduler(`power:${v.toFixed(2)}`)} min={0.25} max={4.0} step={0.05} showInput />
            <p className="text-[10px] text-zinc-500">p&gt;1 = front-loaded (structure), p=1 = linear, p&lt;1 = back-loaded (detail).</p>
          </div>
        );
      })()}

      {/* ── Composite Sub-Controls (Accordion) ── */}
      {gp.scheduler.startsWith('composite') && (() => {
        const parts = gp.scheduler.split(':');
        const schedulerPair = (parts[1] || 'bong_tangent+linear').split('+');
        const stageA = schedulerPair[0] || 'bong_tangent';
        const stageB = schedulerPair[1] || 'linear';
        const crossover = parseFloat(parts[2] || '0.5');
        const split = parseFloat(parts[3] || '0.5');
        const update = (a: string, b: string, c: number, s: number) => {
          gp.setScheduler(`composite:${a}+${b}:${c.toFixed(2)}:${s.toFixed(2)}`);
        };
        return (
          <div className="rounded-xl border border-purple-500/20 bg-purple-500/5 transition-all overflow-hidden">
            <button
              type="button"
              onClick={() => setCompositeOpen(!compositeOpen)}
              className="w-full flex items-center justify-between px-3 py-2 hover:bg-purple-500/5 transition-colors"
            >
              <div className="flex items-center gap-2">
                <ChevronDown size={12} className={`text-purple-400 transition-transform duration-200 ${compositeOpen ? 'rotate-180' : ''}`} />
                <span className="text-[10px] font-semibold text-purple-400 uppercase tracking-wider">Composite (2-Stage)</span>
              </div>
              <button type="button" onClick={(e) => { e.stopPropagation(); update('bong_tangent', 'linear', 0.5, 0.5); }}
                className="flex items-center gap-1 text-[10px] text-purple-400 hover:text-purple-300 transition-colors">
                <RotateCcw size={10} /> Reset
              </button>
            </button>
            {compositeOpen && (
              <div className="px-3 pb-3 space-y-3">
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
            )}
          </div>
        );
      })()}

      {/* Guidance Mode */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Guidance</label>
        <select className={selectClasses} value={gp.guidanceMode}
          onChange={e => gp.setGuidanceMode(e.target.value)}>
          <option value="apg">APG (Default)</option>
          <option value="cfg_pp">CFG++</option>
          <option value="dynamic_cfg">Dynamic CFG</option>
          <option value="rescaled_cfg">Rescaled CFG</option>
        </select>
      </div>

      {/* ── APG Sub-Controls ── */}
      {gp.guidanceMode === 'apg' && (
        <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-3 space-y-3 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider">APG Parameters</span>
            <button type="button" onClick={() => {
              gp.setApgMomentum(0.75);
              gp.setApgNormThreshold(2.5);
            }} className="flex items-center gap-1 text-[10px] text-blue-400 hover:text-blue-300 transition-colors">
              <RotateCcw size={10} /> Reset
            </button>
          </div>
          <Slider label="Momentum" value={gp.apgMomentum}
            onChange={gp.setApgMomentum} min={0} max={1} step={0.01} showInput />
          <Slider label="Norm Threshold" value={gp.apgNormThreshold}
            onChange={gp.setApgNormThreshold} min={0} max={10} step={0.1} showInput />
          <p className="text-[10px] text-zinc-500">Momentum smooths guidance across steps. Norm threshold clips gradient magnitude per channel.</p>
        </div>
      )}

      {/* ── DCW Correction (Accordion with checkbox in title) ── */}
      <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 transition-all overflow-hidden">
        <button
          type="button"
          onClick={() => setDcwOpen(!dcwOpen)}
          className="w-full flex items-center justify-between px-3 py-2 hover:bg-emerald-500/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ChevronDown size={12} className={`text-emerald-400 transition-transform duration-200 ${dcwOpen ? 'rotate-180' : ''}`} />
            <div className="flex items-center gap-1.5" onClick={e => e.stopPropagation()}>
              <ToggleSwitch checked={gp.dcwEnabled} onChange={gp.setDcwEnabled} accentColor="emerald" />
              <span className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider">DCW Correction</span>
            </div>
          </div>
          {gp.dcwEnabled && (
            <span type="button" onClick={(e) => {
              e.stopPropagation();
              gp.setDcwMode('double');
              gp.setDcwScaler(0.2);
              gp.setDcwHighScaler(0.2);
            }} className="flex items-center gap-1 text-[10px] text-emerald-400 hover:text-emerald-300 transition-colors cursor-pointer">
              <RotateCcw size={10} /> Reset
            </span>
          )}
        </button>
        {dcwOpen && gp.dcwEnabled && (
          <div className="px-3 pb-3 space-y-3">
            <div>
              <label className="block text-[10px] text-emerald-400 mb-1">Correction Mode</label>
              <select className={selectClasses} value={gp.dcwMode}
                onChange={e => gp.setDcwMode(e.target.value)}>
                <option value="low">Low-Frequency (Recommended)</option>
                <option value="high">High-Frequency</option>
                <option value="double">Both (Low + High)</option>
                <option value="pix">Pixel-Space (No Wavelets)</option>
              </select>
            </div>
            <Slider label="Correction Scaler" value={gp.dcwScaler}
              onChange={gp.setDcwScaler} min={0} max={1} step={0.01} showInput />
            {gp.dcwMode === 'double' && (
              <Slider label="High-Freq Scaler" value={gp.dcwHighScaler}
                onChange={gp.setDcwHighScaler} min={0} max={1} step={0.01} showInput />
            )}
            <p className="text-[10px] text-zinc-500">
              Wavelet-domain SNR-t bias correction (CVPR 2026). Scaler is dynamically modulated by timestep.
            </p>
          </div>
        )}
      </div>

      {/* ── Duration Buffer / Auto-Trim (Accordion with toggle) ── */}
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 transition-all overflow-hidden">
        <button
          type="button"
          onClick={() => setAutoTrimOpen(!autoTrimOpen)}
          className="w-full flex items-center justify-between px-3 py-2 hover:bg-amber-500/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ChevronDown size={12} className={`text-amber-400 transition-transform duration-200 ${autoTrimOpen ? 'rotate-180' : ''}`} />
            <div className="flex items-center gap-1.5" onClick={e => e.stopPropagation()}>
              <ToggleSwitch checked={gp.autoTrimEnabled} onChange={gp.setAutoTrimEnabled} accentColor="amber" />
              <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">Auto-Trim Endings</span>
            </div>
          </div>
          {gp.autoTrimEnabled && (
            <span onClick={(e) => {
              e.stopPropagation();
              gp.setDurationBuffer(15);
            }} className="flex items-center gap-1 text-[10px] text-amber-400 hover:text-amber-300 transition-colors cursor-pointer">
              <RotateCcw size={10} /> Reset
            </span>
          )}
        </button>
        {autoTrimOpen && gp.autoTrimEnabled && (
          <div className="px-3 pb-3 space-y-3">
            <Slider label="Duration Buffer (seconds)" value={gp.durationBuffer}
              onChange={gp.setDurationBuffer} min={5} max={30} step={1} showInput />
            <p className="text-[10px] text-zinc-500">
              Generates extra audio beyond the requested duration, then trims at the natural song ending.
              Prevents abrupt cut-offs by giving the model breathing room to conclude.
            </p>
          </div>
        )}
      </div>

      {/* ── Latent Post-Processing (Accordion) ── */}
      <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 transition-all overflow-hidden">
        <button
          type="button"
          onClick={() => setLatentOpen(!latentOpen)}
          className="w-full flex items-center justify-between px-3 py-2 hover:bg-indigo-500/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ChevronDown size={12} className={`text-indigo-400 transition-transform duration-200 ${latentOpen ? 'rotate-180' : ''}`} />
            <span className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider">Latent Post-Processing</span>
          </div>
          <span onClick={(e) => {
            e.stopPropagation();
            gp.setLatentShift(0);
            gp.setLatentRescale(1);
            gp.setCustomTimesteps('');
          }} className="flex items-center gap-1 text-[10px] text-indigo-400 hover:text-indigo-300 transition-colors cursor-pointer">
            <RotateCcw size={10} /> Reset
          </span>
        </button>
        {latentOpen && (
          <div className="px-3 pb-3 space-y-3">
            <Slider label="Latent Shift" value={gp.latentShift}
              onChange={gp.setLatentShift} min={-2} max={2} step={0.01} showInput />
            <Slider label="Latent Rescale" value={gp.latentRescale}
              onChange={gp.setLatentRescale} min={0.1} max={3} step={0.01} showInput />
            <div>
              <label className="block text-[10px] text-indigo-400 mb-1">Custom Timesteps</label>
              <input className={inputClasses} value={gp.customTimesteps}
                onChange={e => gp.setCustomTimesteps(e.target.value)}
                placeholder="0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0" />
              <p className="text-[10px] text-zinc-500 mt-1">CSV of descending floats. Overrides schedule + step count when set.</p>
            </div>
          </div>
        )}
      </div>

      {/* ── Post-VAE Spectral Denoiser (Accordion with toggle) ── */}
      <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 transition-all overflow-hidden">
        <button
          type="button"
          onClick={() => setDenoiserOpen(!denoiserOpen)}
          className="w-full flex items-center justify-between px-3 py-2 hover:bg-amber-500/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ChevronDown size={12} className={`text-amber-400 transition-transform duration-200 ${denoiserOpen ? 'rotate-180' : ''}`} />
            <div className="flex items-center gap-1.5" onClick={e => e.stopPropagation()}>
              <ToggleSwitch checked={gp.denoiseStrength > 0} onChange={(on) => gp.setDenoiseStrength(on ? 0.5 : 0)} accentColor="amber" />
              <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">Denoiser</span>
            </div>
          </div>
          {gp.denoiseStrength > 0 && (
            <span onClick={(e) => {
              e.stopPropagation();
              gp.setDenoiseStrength(0.0);
              gp.setDenoiseSmoothing(0.7);
              gp.setDenoiseMix(0.25);
            }} className="flex items-center gap-1 text-[10px] text-amber-400 hover:text-amber-300 transition-colors cursor-pointer">
              <RotateCcw size={10} /> Reset
            </span>
          )}
        </button>
        {denoiserOpen && gp.denoiseStrength > 0 && (
          <div className="px-3 pb-3 space-y-3">
            <Slider label="Strength" value={gp.denoiseStrength}
              onChange={gp.setDenoiseStrength} min={0.01} max={1} step={0.01} showInput />
            <Slider label="Smoothing" value={gp.denoiseSmoothing}
              onChange={gp.setDenoiseSmoothing} min={0} max={1} step={0.01} showInput />
            <Slider label="Mix" value={gp.denoiseMix}
              onChange={gp.setDenoiseMix} min={0} max={1} step={0.01} showInput />
            <p className="text-[10px] text-zinc-500">
              Spectral gate removes VAE fuzz after decode. Higher strength = more aggressive noise suppression.
            </p>
          </div>
        )}
      </div>

      {/* Seed */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Seed</label>
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-zinc-500">Random</span>
            <ToggleSwitch checked={gp.randomSeed} onChange={gp.setRandomSeed} accentColor="sky" />
          </div>
        </div>
        {!gp.randomSeed && (
          <input type="number" className={inputClasses} value={gp.seed}
            onChange={e => gp.setSeed(parseInt(e.target.value) || -1)} />
        )}
      </div>

      {/* Batch */}
      <Slider label="Batch Size" value={gp.batchSize}
        onChange={gp.setBatchSize} min={1} max={9} step={1} />
    </div>
  );
};

/** Summary badge for the Generation section */
export const GenerationBadge: React.FC = () => {
  const gp = useGlobalParams();

  const solverLabels: Record<string, string> = {
    euler: 'Euler', heun: 'Heun', dpm2m: 'DPM++2M', dpm3m: 'DPM++3M',
    rk4: 'RK4', rk5: 'RK5', sde: 'SDE', jkass_fast: 'JKASS',
    jkass_quality: 'JKASSq', stork2: 'STORK2', stork4: 'STORK4',
    dopri5: 'DOPRI5', dop853: 'DOP853', dpm2m_ada: 'DPM++A', gl2s: 'GL2s', rfsolver: 'RFSolv',
  };
  const guidanceLabels: Record<string, string> = {
    apg: 'APG', cfg_pp: 'CFG++', dynamic_cfg: 'DynCFG', rescaled_cfg: 'rCFG',
  };

  const solver = solverLabels[gp.inferMethod] || gp.inferMethod;
  const guidance = guidanceLabels[gp.guidanceMode] || gp.guidanceMode;
  const schedule = formatScheduler(gp.scheduler);
  const shiftLabel = gp.shift === -1 ? 'Auto' : gp.shift.toFixed(1);
  const seedLabel = gp.randomSeed ? 'Rnd' : 'Fix';

  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {gp.inferenceSteps}s · {solver} · {schedule} · {guidance} {gp.guidanceScale.toFixed(1)} · σ{shiftLabel} · {seedLabel}
    </span>
  );
};
