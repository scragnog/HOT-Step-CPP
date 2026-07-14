// GenerationDropdown.tsx — DiT generation settings for the global param bar
//
// Adapted from the DiT section of create/GenerationSettings.tsx.
// Reads from GlobalParamsContext instead of props.

import React, { useState, useEffect, useCallback, useMemo } from 'react';
// Seed input uses local string state to avoid parseInt("-") → NaN → -1 snap-back
import { useTranslation } from 'react-i18next';
import { RotateCcw, ChevronDown, Music2, Upload, Trash2, Zap, Save } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { Slider } from '../shared/Slider';
import { ToggleSwitch } from './BarSection';
import { formatScheduler, formatReferenceName } from './modelLabels';
import { usePersistedState } from '../../hooks/usePersistedState';
import { masteringApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { usePluginRegistry } from '../../hooks/usePluginRegistry';
import { PluginControls } from './PluginControls';
import { SeedManagerDrawer } from './SeedManagerDrawer';

const selectClasses = "w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer";
const inputClasses = "w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

export const GenerationDropdown: React.FC = () => {
  const gp = useGlobalParams();
  const { t } = useTranslation();
  const { registry, findSolver, findScheduler, findGuidance } = usePluginRegistry();
  const [compositeOpen, setCompositeOpen] = usePersistedState('hs-genAccordion-composite', false);
  const [dcwOpen, setDcwOpen] = usePersistedState('hs-genAccordion-dcw', false);
  const [latentOpen, setLatentOpen] = usePersistedState('hs-genAccordion-latent', false);
  const [denoiserOpen, setDenoiserOpen] = usePersistedState('hs-genAccordion-denoiser', false);
  const [autoTrimOpen, setAutoTrimOpen] = usePersistedState('hs-genAccordion-autotrim', false);
  const [perfOpen, setPerfOpen] = usePersistedState('hs-genAccordion-perf', false);
  const [timbreOpen, setTimbreOpen] = usePersistedState('hs-genAccordion-timbre', false);
  const { token } = useAuth();

  // ── Timbre reference file management ──
  interface ReferenceTrack { name: string; size: number; url: string; }
  const [timbreRefs, setTimbreRefs] = useState<ReferenceTrack[]>([]);
  const [timbreUploading, setTimbreUploading] = useState(false);
  const [seedDrawerOpen, setSeedDrawerOpen] = useState(false);

  useEffect(() => {
    masteringApi.listReferences()
      .then(data => setTimbreRefs(data.references))
      .catch(() => {});
  }, []);

  const handleTimbreUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !token) return;
    try {
      setTimbreUploading(true);
      const result = await masteringApi.uploadReference(file, token);
      gp.setTimbreAudioPath(result.name);
      const data = await masteringApi.listReferences();
      setTimbreRefs(data.references);
    } catch (err) {
      console.error('[Timbre] Upload failed:', err);
    } finally {
      setTimbreUploading(false);
      e.target.value = '';
    }
  }, [token, gp]);

  // Resolve scheduler dropdown value from the composite string representation
  const schedulerKey = gp.scheduler.startsWith('composite') ? 'composite'
    : gp.scheduler.startsWith('beta:') ? 'beta'
    : gp.scheduler.startsWith('power:') ? 'power'
    : gp.scheduler;

  return (
    <div className="space-y-3">
      <Slider label="Inference Steps" value={gp.inferenceSteps}
        onChange={gp.setInferenceSteps} min={1} max={200} step={1} showInput />

      <Slider label="Guidance Scale" value={gp.guidanceScale}
        onChange={gp.setGuidanceScale} min={0} max={20} step={0.1} showInput />

      {/* ── Performance / Speed Boosts (Accordion, closed by default) ── */}
      <div className={`rounded-xl border transition-all overflow-hidden ${
        (gp.cfgCutoffRatio < 1 || gp.lmCfgCutoffRatio < 1 || gp.cacheRatio > 0)
          ? 'border-amber-500/20 bg-amber-500/5'
          : 'border-zinc-200 dark:border-white/10 bg-zinc-100/30 dark:bg-zinc-800/30'
      }`}>
        <button
          type="button"
          onClick={() => setPerfOpen(!perfOpen)}
          className="w-full flex items-center justify-between px-3 py-2 hover:bg-amber-500/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ChevronDown size={12} className={`text-amber-400 transition-transform duration-200 ${perfOpen ? 'rotate-180' : ''}`} />
            <Zap size={14} className={(gp.cfgCutoffRatio < 1 || gp.lmCfgCutoffRatio < 1 || gp.cacheRatio > 0) ? 'text-amber-400' : 'text-zinc-500'} />
            <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider">Performance</span>
          </div>
          {(gp.cfgCutoffRatio < 1 || gp.lmCfgCutoffRatio < 1 || gp.cacheRatio > 0) && (
            <span className="text-[10px] text-amber-400/60 font-mono">
              {gp.cfgCutoffRatio < 1 ? `CFG ${Math.round(gp.cfgCutoffRatio * 100)}%` : ''}
              {gp.cfgCutoffRatio < 1 && gp.lmCfgCutoffRatio < 1 ? ' · ' : ''}
              {gp.lmCfgCutoffRatio < 1 ? `LM ${Math.round(gp.lmCfgCutoffRatio * 100)}%` : ''}
              {(gp.cfgCutoffRatio < 1 || gp.lmCfgCutoffRatio < 1) && gp.cacheRatio > 0 ? ' · ' : ''}
              {gp.cacheRatio > 0 ? `Cache ${Math.round(gp.cacheRatio * 100)}%` : ''}
            </span>
          )}
        </button>
        {perfOpen && (
          <div className="px-3 pb-3 space-y-3 border-t border-zinc-200 dark:border-white/5">
            <Slider label="CFG Cutoff" value={gp.cfgCutoffRatio}
              onChange={gp.setCfgCutoffRatio} min={0} max={1} step={0.05} showInput />
            <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
              Ratio of DiT steps using full guidance. Lower = faster but may reduce prompt adherence. 0.5 ≈ 20% speedup.
            </p>
            <Slider label="LM CFG Cutoff" value={gp.lmCfgCutoffRatio}
              onChange={gp.setLmCfgCutoffRatio} min={0.3} max={1} step={0.05} showInput />
            <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
              Fraction of LM audio code tokens using guidance. Lower = faster but may reduce prompt adherence. 0.7 = ~15% LM speedup.
            </p>
            <Slider label="Step Cache" value={gp.cacheRatio}
              onChange={gp.setCacheRatio} min={0} max={0.7} step={0.05} showInput />
            <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
              Skip redundant forward passes by reusing velocity. Higher = faster but may reduce quality. Try 0.3–0.5.
            </p>
            <button type="button" onClick={() => { gp.setCfgCutoffRatio(1); gp.setLmCfgCutoffRatio(1); gp.setCacheRatio(0); }}
              className="flex items-center gap-1 text-[10px] text-amber-400 hover:text-amber-300 transition-colors">
              <RotateCcw size={10} /> Reset to defaults
            </button>
          </div>
        )}
      </div>

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
                : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border border-zinc-200 dark:border-white/5 hover:text-zinc-700 dark:text-zinc-300 hover:border-zinc-300 dark:border-white/10'
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
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('gen.solver')}</label>
        <select className={selectClasses} value={gp.inferMethod}
          onChange={e => gp.setInferMethod(e.target.value)}>
          {registry.solvers.length > 0 ? (
            <>
              <optgroup label="── Single Evaluation (1 NFE) ──">
                {registry.solvers.filter(s => (s.nfe ?? 1) === 1).map(s => (
                  <option key={s.name} value={s.name}>{s.display}</option>
                ))}
              </optgroup>
              <optgroup label="── Multi Evaluation ──">
                {registry.solvers.filter(s => (s.nfe ?? 1) > 1).map(s => (
                  <option key={s.name} value={s.name}>{s.display} ({s.nfe} NFE)</option>
                ))}
              </optgroup>
              {registry.solvers.some(s => (s.nfe ?? 1) === 0) && (
                <optgroup label="── Adaptive (Variable NFE) ──">
                  {registry.solvers.filter(s => (s.nfe ?? 1) === 0).map(s => (
                    <option key={s.name} value={s.name}>{s.display}</option>
                  ))}
                </optgroup>
              )}
            </>
          ) : (
            <>
              {/* Fallback while registry is loading */}
              <option value="euler">Euler (ODE)</option>
              <option value="heun">Heun (2 NFE)</option>
              <option value="dpm2m">DPM++ 2M</option>
              <option value="rk4">RK4 (4 NFE)</option>
            </>
          )}
        </select>
        {/* Solver description from Lua plugin metadata */}
        {(() => {
          const solver = findSolver(gp.inferMethod);
          return solver?.description ? (
            <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{solver.description}</p>
          ) : null;
        })()}
      </div>

      {/* ── Dynamic Solver Controls ── */}
      {(() => {
        const solver = findSolver(gp.inferMethod);
        if (!solver || solver.params.length === 0) return null;
        return (
          <PluginControls
            pluginName={solver.name}
            displayName={solver.display}
            accent={solver.accent}
            params={solver.params}
            values={gp.pluginParams}
            onChange={gp.setPluginParam}
            onReset={() => gp.resetPluginParams(solver.name)}
          />
        );
      })()}

      {/* Scheduler */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('gen.schedule')}</label>
        <select className={selectClasses} value={schedulerKey}
          onChange={e => {
            const v = e.target.value;
            if (v === 'beta') gp.setScheduler('beta:0.50:0.70');
            else if (v === 'power') gp.setScheduler('power:2.00');
            else if (v === 'composite') gp.setScheduler('composite:bong_tangent+linear:0.50:0.50');
            else gp.setScheduler(v);
          }}>
          {registry.schedulers.length > 0 ? (
            <>
              {registry.schedulers.map(s => (
                <option key={s.name} value={s.name}>{s.display}</option>
              ))}
              {/* Synthetic entries: parameterized schedules handled by the UI */}
              <option value="beta">Beta (Custom)</option>
              <option value="power">Power</option>
              <option value="composite">Composite (2-Stage)</option>
            </>
          ) : (
            <>
              {/* Fallback while registry is loading */}
              <option value="linear">Linear (Default)</option>
              <option value="cosine">Cosine</option>
              <option value="ddim_uniform">DDIM Uniform</option>
              <option value="sgm_uniform">SGM / Karras</option>
              <option value="bong_tangent">Tangent</option>
              <option value="linear_quadratic">Linear-Quadratic</option>
              <option value="composite">Composite (2-Stage)</option>
            </>
          )}
        </select>
        {/* Scheduler description from Lua plugin metadata */}
        {(() => {
          const sched = registry.schedulers.find(s => s.name === schedulerKey);
          return sched?.description ? (
            <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{sched.description}</p>
          ) : null;
        })()}
      </div>

      {/* ── Dynamic Scheduler Controls ── */}
      {(() => {
        const sched = findScheduler(schedulerKey);
        if (!sched || !sched.params || sched.params.length === 0) return null;
        return (
          <PluginControls
            pluginName={sched.name}
            displayName={sched.display}
            accent={sched.accent}
            params={sched.params}
            values={gp.pluginParams}
            onChange={gp.setPluginParam}
            onReset={() => gp.resetPluginParams(sched.name)}
          />
        );
      })()}

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
                      {registry.schedulers.length > 0 ? (
                        registry.schedulers.map(s => (
                          <option key={s.name} value={s.name}>{s.display}</option>
                        ))
                      ) : (
                        <>
                          <option value="linear">Linear</option>
                          <option value="cosine">Cosine</option>
                          <option value="ddim_uniform">DDIM</option>
                          <option value="sgm_uniform">SGM</option>
                          <option value="bong_tangent">Tangent</option>
                          <option value="linear_quadratic">Lin-Quad</option>
                        </>
                      )}
                    </select>
                  </div>
                  <div>
                    <label className="block text-[10px] text-purple-400 mb-1">Stage B</label>
                    <select className={selectClasses} value={stageB}
                      onChange={e => update(stageA, e.target.value, crossover, split)}>
                      {registry.schedulers.length > 0 ? (
                        registry.schedulers.map(s => (
                          <option key={s.name} value={s.name}>{s.display}</option>
                        ))
                      ) : (
                        <>
                          <option value="linear">Linear</option>
                          <option value="cosine">Cosine</option>
                          <option value="ddim_uniform">DDIM</option>
                          <option value="sgm_uniform">SGM</option>
                          <option value="bong_tangent">Tangent</option>
                          <option value="linear_quadratic">Lin-Quad</option>
                        </>
                      )}
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
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('gen.guidance')}</label>
        <select className={selectClasses} value={gp.guidanceMode}
          onChange={e => gp.setGuidanceMode(e.target.value)}>
          {registry.guidance.length > 0 ? (
            registry.guidance.map(g => (
              <option key={g.name} value={g.name}>{g.display}</option>
            ))
          ) : (
            <>
              <option value="apg">APG (Default)</option>
              <option value="cfg_pp">CFG++</option>
              <option value="dynamic_cfg">Dynamic CFG</option>
              <option value="rescaled_cfg">Rescaled CFG</option>
            </>
          )}
        </select>
        {/* Guidance description from Lua plugin metadata */}
        {(() => {
          const guide = findGuidance(gp.guidanceMode);
          return guide?.description ? (
            <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{guide.description}</p>
          ) : null;
        })()}
      </div>

      {/* ── APG Sub-Controls (native C++ path — always show for APG) ── */}
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

      {/* ── Dynamic Guidance Controls (non-APG) ── */}
      {gp.guidanceMode !== 'apg' && (() => {
        const guide = findGuidance(gp.guidanceMode);
        if (!guide || guide.params.length === 0) return null;
        return (
          <PluginControls
            pluginName={guide.name}
            displayName={guide.display}
            accent={guide.accent}
            params={guide.params}
            values={gp.pluginParams}
            onChange={gp.setPluginParam}
            onReset={() => gp.resetPluginParams(guide.name)}
          />
        );
      })()}

      {/* ── Timbre Conditioning (Accordion with file picker) ── */}
      <div className={`rounded-xl border transition-all overflow-hidden ${gp.timbreAudioPath ? 'border-teal-500/20 bg-teal-500/5' : 'border-zinc-200 dark:border-white/10 bg-zinc-100/30 dark:bg-zinc-800/30'}`}>
        <button
          type="button"
          onClick={() => setTimbreOpen(!timbreOpen)}
          className="w-full flex items-center justify-between px-3 py-2 hover:bg-teal-500/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ChevronDown size={12} className={`text-teal-400 transition-transform duration-200 ${timbreOpen ? 'rotate-180' : ''}`} />
            <Music2 size={14} className={gp.timbreAudioPath ? 'text-teal-400' : 'text-zinc-500'} />
            <span className="text-[10px] font-semibold text-teal-400 uppercase tracking-wider">Timbre Reference</span>
          </div>
          {gp.timbreAudioPath ? (
            <span className="text-[10px] text-teal-400/60 font-mono truncate max-w-[120px]">
              {formatReferenceName(gp.timbreAudioPath)}
            </span>
          ) : gp.timbreReference && gp.masteringReference ? (
            <span className="text-[10px] text-zinc-500 font-mono">Mastering ref</span>
          ) : (
            <span className="text-[10px] text-zinc-600 font-mono">None</span>
          )}
        </button>
        {timbreOpen && (
          <div className="px-3 pb-3 space-y-3 border-t border-zinc-200 dark:border-white/5">
            <p className="text-[10px] text-zinc-500 leading-relaxed mt-2">
              Set a dedicated audio track for timbre conditioning. The reference is VAE-encoded
              and fed into the DiT during synthesis, guiding tone and texture.
              If not set, the mastering reference is used when the timbre toggle is enabled.
            </p>

            {/* Reference selector */}
            {timbreRefs.length > 0 ? (
              <select
                className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-teal-500/50 focus:ring-1 focus:ring-teal-500/20 outline-none transition-colors cursor-pointer"
                value={gp.timbreAudioPath}
                onChange={e => gp.setTimbreAudioPath(e.target.value)}
              >
                <option value="">None (use mastering ref if enabled)</option>
                {timbreRefs.map(r => (
                  <option key={r.name} value={r.name}>
                    {r.name} ({r.size < 1024 * 1024 ? `${(r.size / 1024).toFixed(1)} KB` : `${(r.size / (1024 * 1024)).toFixed(1)} MB`})
                  </option>
                ))}
              </select>
            ) : (
              <div className="text-xs text-zinc-500 italic px-1">
                No reference tracks uploaded yet
              </div>
            )}

            {/* Selected file info + clear */}
            {gp.timbreAudioPath && (
              <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-teal-500/5 border border-teal-500/10">
                <Music2 size={14} className="text-teal-400 flex-shrink-0" />
                <span className="text-xs text-teal-300 truncate flex-1">{gp.timbreAudioPath}</span>
                <button
                  onClick={() => gp.setTimbreAudioPath('')}
                  className="p-1 rounded hover:bg-red-500/10 text-zinc-500 hover:text-red-400 transition-colors flex-shrink-0"
                  title={t('gen.clearTimbreRef')}
                >
                  <Trash2 size={12} />
                </button>
              </div>
            )}

            {/* Upload button */}
            <div className="flex items-center gap-2">
              <input
                type="file"
                accept="audio/*"
                id="timbre-ref-upload-gen"
                className="hidden"
                onChange={handleTimbreUpload}
              />
              <label
                htmlFor="timbre-ref-upload-gen"
                className={`flex items-center gap-2 px-3 py-2 text-xs font-semibold rounded-xl border cursor-pointer transition-all ${
                  timbreUploading
                    ? 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border-zinc-200 dark:border-white/5 cursor-wait'
                    : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border-zinc-300 dark:border-white/10 hover:border-teal-500/30 hover:text-teal-400'
                }`}
              >
                {timbreUploading ? (
                  <><span className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" /> Uploading...</>
                ) : (
                  <><Upload size={14} /> {t('gen.uploadReference')}</>
                )}
              </label>
            </div>
          </div>
        )}
      </div>

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
            <span onClick={(e) => {
              e.stopPropagation();
              gp.setDcwMode('double');
              gp.setDcwLowScaler(0.2);
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
              <div className="relative group/dcw">
                <select className={selectClasses} value={gp.dcwMode}
                  onChange={e => gp.setDcwMode(e.target.value)}>
                  <option value="low">Low-Frequency</option>
                  <option value="high">High-Frequency</option>
                  <option value="double">Both (Low + High)</option>
                  <option value="pix">Pixel-Space (No Wavelets)</option>
                </select>
                <div className="mt-1.5 text-[10px] text-zinc-500 leading-relaxed">
                  {gp.dcwMode === 'low' && '🎵 Corrects low-frequency wavelet bands — tightens bass, kick and rhythm without touching treble.'}
                  {gp.dcwMode === 'high' && '✨ Corrects high-frequency wavelet bands — sharpens hi-hats, vocals and presence.'}
                  {gp.dcwMode === 'double' && '🎛️ Independent correction on both low and high bands with separate scalers.'}
                  {gp.dcwMode === 'pix' && '📐 Applies correction directly in latent space, bypassing wavelet decomposition. More uniform but less targeted.'}
                </div>
              </div>
            </div>
            {(gp.dcwMode === 'low' || gp.dcwMode === 'double' || gp.dcwMode === 'pix') && (
              <Slider label={gp.dcwMode === 'double' ? 'Low-Freq Scaler' : 'Scaler'} value={gp.dcwLowScaler}
                onChange={gp.setDcwLowScaler} min={0} max={1} step={0.01} showInput />
            )}
            {(gp.dcwMode === 'high' || gp.dcwMode === 'double') && (
              <Slider label={gp.dcwMode === 'double' ? 'High-Freq Scaler' : 'Scaler'} value={gp.dcwHighScaler}
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
              gp.setAutoTrimFadeMs(2000);
            }} className="flex items-center gap-1 text-[10px] text-amber-400 hover:text-amber-300 transition-colors cursor-pointer">
              <RotateCcw size={10} /> Reset
            </span>
          )}
        </button>
        {autoTrimOpen && gp.autoTrimEnabled && (
          <div className="px-3 pb-3 space-y-3">
            <Slider label="Duration Buffer (seconds)" value={gp.durationBuffer}
              onChange={gp.setDurationBuffer} min={5} max={30} step={1} showInput />
            <Slider label="Fade-Out (seconds)" value={gp.autoTrimFadeMs / 1000}
              onChange={(v: number) => gp.setAutoTrimFadeMs(Math.round(v * 1000))} min={0.5} max={5} step={0.1} showInput />
            <p className="text-[10px] text-zinc-500">
              Generates extra audio beyond the requested duration, then trims at the natural song ending.
              Fade-out only applies when no clean ending is detected (forced trim at original duration).
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
      <div className="relative">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Generation Seed</label>
            <button onClick={() => setSeedDrawerOpen(true)} title="Seed Manager"
              className="text-zinc-500 hover:text-amber-400 transition-colors">
              <Save size={12} />
            </button>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-zinc-500">Random</span>
            <ToggleSwitch checked={gp.randomSeed} onChange={gp.setRandomSeed} accentColor="sky" />
          </div>
        </div>
        {!gp.randomSeed && (
          <SeedInput value={gp.seed} onChange={gp.setSeed} className={inputClasses} />
        )}
        <p className="text-[10px] text-zinc-500 mt-1">
          Drives audio synthesis (DiT). Varies per track during batch generation. See LM Seed for caption/lyrics/code sampling.
        </p>
        <SeedManagerDrawer
          isOpen={seedDrawerOpen}
          onClose={() => setSeedDrawerOpen(false)}
          currentSeed={gp.seed}
          onLoad={(seed) => { gp.setSeed(seed); gp.setRandomSeed(false); setSeedDrawerOpen(false); }}
          onLoadRandom={(seed) => { gp.setSeed(seed); gp.setRandomSeed(false); }}
        />
      </div>

      {/* Batch */}
      <Slider label="Batch Size" value={gp.batchSize}
        onChange={gp.setBatchSize} min={1} max={9} step={1} />
    </div>
  );
};

/** Seed input with local string buffer — prevents parseInt("-") snap-back */
const SeedInput: React.FC<{ value: number; onChange: (v: number) => void; className: string }> = ({ value, onChange, className }) => {
  const [local, setLocal] = useState(String(value));
  useEffect(() => { setLocal(String(value)); }, [value]);
  const commit = () => { onChange(parseInt(local) || 42); };
  return (
    <input type="number" className={className} value={local}
      onChange={e => setLocal(e.target.value)}
      onBlur={commit}
      onKeyDown={e => { if (e.key === 'Enter') commit(); }}
    />
  );
};

/** Summary badge for the Generation section */
export const GenerationBadge: React.FC = () => {
  const gp = useGlobalParams();
  const { registry } = usePluginRegistry();

  const solver = useMemo(() => {
    const s = registry.solvers.find(p => p.name === gp.inferMethod);
    return s?.display || gp.inferMethod;
  }, [registry.solvers, gp.inferMethod]);

  const guidance = useMemo(() => {
    const g = registry.guidance.find(p => p.name === gp.guidanceMode);
    return g?.display || gp.guidanceMode;
  }, [registry.guidance, gp.guidanceMode]);

  const schedule = formatScheduler(gp.scheduler);
  const shiftLabel = gp.shift === -1 ? 'Auto' : gp.shift.toFixed(1);
  const seedLabel = gp.randomSeed ? 'Rnd' : 'Fix';

  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {gp.inferenceSteps}s · {solver} · {schedule} · {guidance} {gp.guidanceScale.toFixed(1)} · σ{shiftLabel} · Seed {seedLabel}
    </span>
  );
};
