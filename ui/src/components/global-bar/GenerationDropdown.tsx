// GenerationDropdown.tsx — DiT generation settings for the global param bar
//
// Adapted from the DiT section of create/GenerationSettings.tsx.
// Reads from GlobalParamsContext instead of props.

import React, { useState, useEffect, useCallback, useMemo } from 'react';
// Seed input uses local string state to avoid parseInt("-") → NaN → -1 snap-back
import { useTranslation } from 'react-i18next';
import { RotateCcw, ChevronDown, Music2, Upload, Trash2, Zap } from 'lucide-react';
import { useGlobalParams, useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { Slider } from '../shared/Slider';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { formatScheduler, formatReferenceName } from './modelLabels';
import { usePersistedState } from '../../hooks/usePersistedState';
import { masteringApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { usePluginRegistry } from '../../hooks/usePluginRegistry';
import { PluginControls } from './PluginControls';
import { SeedControl } from './SeedControl';

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
  const [lssOpen, setLssOpen] = usePersistedState('hs-genAccordion-lss', false);

  // LSS params — new fields use direct store selectors (GlobalParamsContext is a legacy shim)
  const lssStrength = useGlobalParamsStore((s: any) => s.lssStrength);
  const lssVarThresh = useGlobalParamsStore((s: any) => s.lssVarThresh);
  const lssDcRemove = useGlobalParamsStore((s: any) => s.lssDcRemove);
  const setLssStrength = useGlobalParamsStore((s: any) => s.setLssStrength);
  const setLssVarThresh = useGlobalParamsStore((s: any) => s.setLssVarThresh);
  const setLssDcRemove = useGlobalParamsStore((s: any) => s.setLssDcRemove);
  const [autoTrimOpen, setAutoTrimOpen] = usePersistedState('hs-genAccordion-autotrim', false);
  const [perfOpen, setPerfOpen] = usePersistedState('hs-genAccordion-perf', false);
  const [timbreOpen, setTimbreOpen] = usePersistedState('hs-genAccordion-timbre', false);
  const { token } = useAuth();

  // ── Timbre reference file management ──
  interface ReferenceTrack { name: string; size: number; url: string; }
  const [timbreRefs, setTimbreRefs] = useState<ReferenceTrack[]>([]);
  const [timbreUploading, setTimbreUploading] = useState(false);

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

  // Hoisted because a plugin's own description is now the picker label's hover
  // card rather than a paragraph under the <select>, so the label needs the
  // metadata as well as the body.
  const solverMeta = findSolver(gp.inferMethod);
  const schedMeta = registry.schedulers.find(sc => sc.name === schedulerKey);
  const guideMeta = findGuidance(gp.guidanceMode);

  // The four DCW modes differ enough that one blurb cannot cover them all, so
  // this hover card follows the selection.
  const DCW_MODE_INFO: Record<string, string> = {
    low: 'Corrects low-frequency wavelet bands - tightens bass, kick and rhythm without touching treble.',
    high: 'Corrects high-frequency wavelet bands - sharpens hi-hats, vocals and presence.',
    double: 'Independent correction on both low and high bands, with separate scalers for each.',
    pix: 'Applies correction directly in latent space, bypassing wavelet decomposition. More uniform but less targeted.',
  };

  return (
    <div className="space-y-3">
      <Slider label="Inference Steps" value={gp.inferenceSteps}
        onChange={gp.setInferenceSteps} min={1} max={300} step={1} showInput
        infoMeta="default 12 · range 1-300"
        info="How many denoising steps the DiT takes to turn noise into audio. More steps take longer to render; past roughly 40 on ACE-Step, extra steps stop improving quality." />

      <Slider label="Guidance Scale" value={gp.guidanceScale}
        onChange={gp.setGuidanceScale} min={0} max={20} step={0.1} showInput
        infoMeta="default 9.0 · range 0-20"
        info="Classifier-free guidance strength: how hard the DiT is pushed toward matching the caption and lyrics. Higher follows the prompt more closely but can sound over-processed; lower drifts further from the prompt but sounds more natural." />

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
            <ParamLabel label="Performance" underline={false}
              className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider"
              info="Three ways to buy speed with quality. All three are off by default, and all three are safe to leave alone." />
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
              onChange={gp.setCfgCutoffRatio} min={0} max={1} step={0.05} showInput
              infoMeta="0–1 · default 1"
              info="Ratio of DiT steps using full guidance. Lower = faster but may reduce prompt adherence. 0.5 ≈ 20% speedup." />
            <Slider label="LM CFG Cutoff" value={gp.lmCfgCutoffRatio}
              onChange={gp.setLmCfgCutoffRatio} min={0.3} max={1} step={0.05} showInput
              infoMeta="0.3–1 · default 1"
              info="Fraction of LM audio code tokens using guidance. Lower = faster but may reduce prompt adherence. 0.7 = ~15% LM speedup." />
            <Slider label="Step Cache" value={gp.cacheRatio}
              onChange={gp.setCacheRatio} min={0} max={0.7} step={0.05} showInput
              infoMeta="0–0.7 · default 0"
              info="Skip redundant forward passes by reusing velocity. Higher = faster but may reduce quality. Try 0.3–0.5." />
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
          <ParamLabel label="Shift" className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
            info="Timestep shift (sigma). Biases the schedule toward the noisy end, trading fine detail for structure. Auto derives it from the duration and the step count." />
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

      {/* Attention Reach (DiT sliding-window override) — UI REMOVED, PARKED.
          The engine parameter still exists and still defaults to the model's own
          128; it is simply not exposed. Widening it broke every generation that
          used a heavily-converged LM planner adapter, and the cause was never
          identified. Investigation, rig and full results:
          docs/plans/attention-drift/. To resume, restore this control — the
          store field, context type and translateParams mapping are all intact. */}

      {/* Solver */}
      <div>
        <ParamLabel label={t('gen.solver')} info={solverMeta?.description}
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
          rootClassName="flex mb-1.5" />
        <StyledSelect accent="sky" className={selectClasses} value={gp.inferMethod}
          onChange={gp.setInferMethod}
          options={registry.solvers.length > 0 ? [
            ...registry.solvers.filter(s => (s.nfe ?? 1) === 1)
              .map(s => ({ value: s.name, label: s.display })),
            ...registry.solvers.filter(s => (s.nfe ?? 1) > 1)
              .map(s => ({ value: s.name, label: `${s.display} (${s.nfe} NFE)` })),
            ...registry.solvers.filter(s => (s.nfe ?? 1) === 0)
              .map(s => ({ value: s.name, label: s.display })),
          ] : [
            /* Fallback while registry is loading */
            { value: 'euler', label: 'Euler (ODE)' },
            { value: 'heun', label: 'Heun (2 NFE)' },
            { value: 'dpm2m', label: 'DPM++ 2M' },
            { value: 'rk4', label: 'RK4 (4 NFE)' },
          ]} />
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
        <ParamLabel label={t('gen.schedule')} info={schedMeta?.description}
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
          rootClassName="flex mb-1.5" />
        <StyledSelect accent="sky" className={selectClasses} value={schedulerKey}
          onChange={(v: string) => {
            if (v === 'beta') gp.setScheduler('beta:0.50:0.70');
            else if (v === 'power') gp.setScheduler('power:2.00');
            else if (v === 'composite') gp.setScheduler('composite:bong_tangent+linear:0.50:0.50');
            else gp.setScheduler(v);
          }}
          options={registry.schedulers.length > 0 ? [
            ...registry.schedulers.map(s => ({ value: s.name, label: s.display })),
            /* Synthetic entries: parameterized schedules handled by the UI */
            { value: 'beta', label: 'Beta (Custom)' },
            { value: 'power', label: 'Power' },
            { value: 'composite', label: 'Composite (2-Stage)' },
          ] : [
            /* Fallback while registry is loading */
            { value: 'linear', label: 'Linear (Default)' },
            { value: 'cosine', label: 'Cosine' },
            { value: 'ddim_uniform', label: 'DDIM Uniform' },
            { value: 'sgm_uniform', label: 'SGM / Karras' },
            { value: 'bong_tangent', label: 'Tangent' },
            { value: 'linear_quadratic', label: 'Linear-Quadratic' },
            { value: 'composite', label: 'Composite (2-Stage)' },
          ]} />
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
              <ParamLabel label="Beta Distribution" underline={false}
                className="text-[10px] font-semibold text-teal-400 uppercase tracking-wider"
                info="Lower alpha puts more density at the edges. Lower beta front-loads the schedule, favouring structure over detail." />
              <button type="button" onClick={() => updateBeta(0.5, 0.7)}
                className="flex items-center gap-1 text-[10px] text-teal-400 hover:text-teal-300 transition-colors">
                <RotateCcw size={10} /> Reset
              </button>
            </div>
            <Slider label="Alpha (α)" value={alpha}
              onChange={v => updateBeta(v, betaParam)} min={0.1} max={2.0} step={0.05} showInput
              infoMeta="default 0.5 · range 0.1-2.0"
              info="Shape of the beta distribution used to space schedule steps. Lower values put more steps at the edges of the schedule (start and end); higher values spread them more evenly." />
            <Slider label="Beta (β)" value={betaParam}
              onChange={v => updateBeta(alpha, v)} min={0.1} max={2.0} step={0.05} showInput
              infoMeta="default 0.7 · range 0.1-2.0"
              info="Second shape parameter of the beta distribution. Lower values front-load the schedule toward structure; higher values shift steps toward the detail end." />
          </div>
        );
      })()}

      {/* ── Power Sub-Controls ── */}
      {gp.scheduler.startsWith('power:') && (() => {
        const exponent = parseFloat(gp.scheduler.split(':')[1] || '2.0');
        return (
          <div className="rounded-xl border border-orange-500/20 bg-orange-500/5 p-3 space-y-3 transition-all">
            <div className="flex items-center justify-between">
              <ParamLabel label="Power Law" underline={false}
                className="text-[10px] font-semibold text-orange-400 uppercase tracking-wider"
                info="Above 1 front-loads the schedule (structure). At 1 it is linear. Below 1 it back-loads (detail)." />
              <button type="button" onClick={() => gp.setScheduler('power:2.00')}
                className="flex items-center gap-1 text-[10px] text-orange-400 hover:text-orange-300 transition-colors">
                <RotateCcw size={10} /> Reset
              </button>
            </div>
            <Slider label="Exponent" value={exponent}
              onChange={v => gp.setScheduler(`power:${v.toFixed(2)}`)} min={0.25} max={4.0} step={0.05} showInput
              infoMeta="default 2.0 · range 0.25-4.0"
              info="Exponent of the power-law schedule. Above 1 front-loads steps toward the noisy end, favouring structure; at 1 the schedule is linear; below 1 it back-loads toward the clean end, favouring detail." />
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
                    <ParamLabel label="Stage A" className="text-[10px] text-purple-400" rootClassName="flex mb-1"
                      info="The scheduler that runs before the crossover point." />
                    <StyledSelect accent="purple" className={selectClasses} value={stageA}
                      onChange={(v: string) => update(v, stageB, crossover, split)}
                      options={registry.schedulers.length > 0 ? (
                        registry.schedulers.map(s => ({ value: s.name, label: s.display }))
                      ) : [
                        { value: 'linear', label: 'Linear' },
                        { value: 'cosine', label: 'Cosine' },
                        { value: 'ddim_uniform', label: 'DDIM' },
                        { value: 'sgm_uniform', label: 'SGM' },
                        { value: 'bong_tangent', label: 'Tangent' },
                        { value: 'linear_quadratic', label: 'Lin-Quad' },
                      ]} />
                  </div>
                  <div>
                    <ParamLabel label="Stage B" className="text-[10px] text-purple-400" rootClassName="flex mb-1"
                      info="The scheduler that runs after the crossover point." />
                    <StyledSelect accent="purple" className={selectClasses} value={stageB}
                      onChange={(v: string) => update(stageA, v, crossover, split)}
                      options={registry.schedulers.length > 0 ? (
                        registry.schedulers.map(s => ({ value: s.name, label: s.display }))
                      ) : [
                        { value: 'linear', label: 'Linear' },
                        { value: 'cosine', label: 'Cosine' },
                        { value: 'ddim_uniform', label: 'DDIM' },
                        { value: 'sgm_uniform', label: 'SGM' },
                        { value: 'bong_tangent', label: 'Tangent' },
                        { value: 'linear_quadratic', label: 'Lin-Quad' },
                      ]} />
                  </div>
                </div>
                <Slider label="Crossover" value={crossover}
                  onChange={v => update(stageA, stageB, v, split)} min={0.1} max={0.9} step={0.05} showInput
                  infoMeta="default 0.5 · range 0.1-0.9"
                  info="How gradually the schedule blends from Stage A to Stage B around the split point. Lower is a harder cut; higher blends more steps between the two schedulers." />
                <Slider label="Split" value={split}
                  onChange={v => update(stageA, stageB, crossover, v)} min={0.1} max={0.9} step={0.05} showInput
                  infoMeta="default 0.5 · range 0.1-0.9"
                  info="Where in the step sequence the schedule crosses from Stage A to Stage B, as a fraction of the total steps." />
              </div>
            )}
          </div>
        );
      })()}

      {/* Guidance Mode */}
      <div>
        <ParamLabel label={t('gen.guidance')} info={guideMeta?.description}
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
          rootClassName="flex mb-1.5" />
        <StyledSelect accent="sky" className={selectClasses} value={gp.guidanceMode}
          onChange={gp.setGuidanceMode}
          options={registry.guidance.length > 0 ? (
            registry.guidance.map(g => ({ value: g.name, label: g.display }))
          ) : [
            { value: 'apg', label: 'APG (Default)' },
            { value: 'cfg_pp', label: 'CFG++' },
            { value: 'dynamic_cfg', label: 'Dynamic CFG' },
            { value: 'rescaled_cfg', label: 'Rescaled CFG' },
          ]} />
      </div>

      {/* ── APG Sub-Controls (native C++ path — always show for APG) ── */}
      {gp.guidanceMode === 'apg' && (
        <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-3 space-y-3 transition-all">
          <div className="flex items-center justify-between">
            <ParamLabel label="APG Parameters" underline={false}
              className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider"
              info="Momentum smooths guidance across steps. Norm threshold clips gradient magnitude per channel." />
            <button type="button" onClick={() => {
              gp.setApgMomentum(0.75);
              gp.setApgNormThreshold(2.5);
            }} className="flex items-center gap-1 text-[10px] text-blue-400 hover:text-blue-300 transition-colors">
              <RotateCcw size={10} /> Reset
            </button>
          </div>
          <Slider label="Momentum" value={gp.apgMomentum}
            onChange={gp.setApgMomentum} min={0} max={1} step={0.01} showInput
            infoMeta="default 0.75 · range 0-1"
            info="Smooths the guidance signal across sampling steps by blending in the previous step's guidance. Higher values carry over more from prior steps, damping step-to-step jitter." />
          <Slider label="Norm Threshold" value={gp.apgNormThreshold}
            onChange={gp.setApgNormThreshold} min={0} max={10} step={0.1} showInput
            infoMeta="default 2.5 · range 0-10"
            info="Caps the guidance vector's magnitude per channel. Lower values clip more aggressively, holding guidance back; higher values let larger guidance vectors through unclipped." />
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
            <ParamLabel label="Timbre Reference" underline={false}
              className="text-[10px] font-semibold text-teal-400 uppercase tracking-wider"
              info="A dedicated audio track for timbre conditioning. The reference is VAE-encoded and fed into the DiT during synthesis, guiding tone and texture. With none set, the mastering reference is used instead whenever the timbre toggle is on." />
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
            {/* Reference selector */}
            {timbreRefs.length > 0 ? (
              <div>
                <ParamLabel label="Reference Track" className="text-[10px] text-teal-400" rootClassName="flex mb-1"
                  info="Which uploaded track feeds the DiT's timbre conditioning. None uses the mastering reference instead, when Mastering is on and its 'Also use as timbre reference' switch is on." />
                <StyledSelect
                  accent="teal"
                  className="w-full"
                  value={gp.timbreAudioPath}
                  onChange={gp.setTimbreAudioPath}
                  options={[
                    { value: '', label: 'None (use mastering ref if enabled)' },
                    ...timbreRefs.map(r => ({
                      value: r.name,
                      label: `${r.name} (${r.size < 1024 * 1024 ? `${(r.size / 1024).toFixed(1)} KB` : `${(r.size / (1024 * 1024)).toFixed(1)} MB`})`,
                    })),
                  ]}
                />
              </div>
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
              <Toggle checked={gp.dcwEnabled} onChange={gp.setDcwEnabled} accent="emerald" />
              <ParamLabel label="DCW Correction" underline={false}
                className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider"
                info="Wavelet-domain SNR-t bias correction (CVPR 2026). The scaler is dynamically modulated by timestep." />
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
              <ParamLabel label="Correction Mode" info={DCW_MODE_INFO[gp.dcwMode]}
                className="text-[10px] text-emerald-400" rootClassName="flex mb-1" />
              <StyledSelect accent="emerald" className={selectClasses} value={gp.dcwMode}
                onChange={gp.setDcwMode}
                options={[
                  { value: 'low', label: 'Low-Frequency' },
                  { value: 'high', label: 'High-Frequency' },
                  { value: 'double', label: 'Both (Low + High)' },
                  { value: 'pix', label: 'Pixel-Space (No Wavelets)' },
                ]} />
            </div>
            {(gp.dcwMode === 'low' || gp.dcwMode === 'double' || gp.dcwMode === 'pix') && (
              <Slider label={gp.dcwMode === 'double' ? 'Low-Freq Scaler' : 'Scaler'} value={gp.dcwLowScaler}
                onChange={gp.setDcwLowScaler} min={0} max={1} step={0.01} showInput
                infoMeta="default 0.2 · range 0-1"
                info="How strongly the correction is applied to the low-frequency band. Higher applies more correction; 0 turns that band off." />
            )}
            {(gp.dcwMode === 'high' || gp.dcwMode === 'double') && (
              <Slider label={gp.dcwMode === 'double' ? 'High-Freq Scaler' : 'Scaler'} value={gp.dcwHighScaler}
                onChange={gp.setDcwHighScaler} min={0} max={1} step={0.01} showInput
                infoMeta="default 0.2 · range 0-1"
                info="How strongly the correction is applied to the high-frequency band. Higher applies more correction; 0 turns that band off." />
            )}
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
              <Toggle checked={gp.autoTrimEnabled} onChange={gp.setAutoTrimEnabled} accent="amber" />
              <ParamLabel label="Auto-Trim Endings" underline={false}
                className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider"
                info="Generates extra audio beyond the requested duration, then trims at the natural song ending. The fade-out only applies when no clean ending is found and the trim is forced at the original duration." />
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
              onChange={gp.setDurationBuffer} min={5} max={30} step={1} showInput
              infoMeta="default 15 · range 5-30"
              info="Extra seconds rendered beyond the requested duration, giving the trim search room to find a natural ending before the requested length is forced." />
            <Slider label="Fade-Out (seconds)" value={gp.autoTrimFadeMs / 1000}
              onChange={(v: number) => gp.setAutoTrimFadeMs(Math.round(v * 1000))} min={0.5} max={5} step={0.1} showInput
              infoMeta="default 2 · range 0.5-5"
              info="How long the fade-out runs when no natural ending is found and the track is cut at the requested length." />
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
              onChange={gp.setLatentShift} min={-2} max={2} step={0.01} showInput
              infoMeta="default 0 · range -2-2"
              info="Added to every value in the DiT's output latents before the VAE decodes them." />
            <Slider label="Latent Rescale" value={gp.latentRescale}
              onChange={gp.setLatentRescale} min={0.1} max={3} step={0.01} showInput
              infoMeta="default 1 · range 0.1-3"
              info="Multiplies every latent value before the shift above is added." />
            <div>
              <ParamLabel label="Custom Timesteps" className="text-[10px] text-indigo-400" rootClassName="flex mb-1"
                info="CSV of descending floats. Overrides the schedule and the step count when set." />
              <input className={inputClasses} value={gp.customTimesteps}
                onChange={e => gp.setCustomTimesteps(e.target.value)}
                placeholder="0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0" />
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
              <Toggle checked={gp.denoiseStrength > 0} onChange={(on) => gp.setDenoiseStrength(on ? 0.5 : 0)} accent="amber" />
              <ParamLabel label="Denoiser" underline={false}
                className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider"
                info="Spectral gate that removes VAE fuzz after decode. Higher strength = more aggressive noise suppression." />
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
              onChange={gp.setDenoiseStrength} min={0.01} max={1} step={0.01} showInput
              infoMeta="default 0.5 · range 0.01-1"
              info="How aggressively the spectral gate suppresses VAE fuzz. Higher removes more noise but can dull detail." />
            <Slider label="Smoothing" value={gp.denoiseSmoothing}
              onChange={gp.setDenoiseSmoothing} min={0} max={1} step={0.01} showInput
              infoMeta="default 0.7 · range 0-1"
              info="How sharp or smooth the gate's cutoff is. 0 is a sharp gate; 1 is very smooth." />
            <Slider label="Mix" value={gp.denoiseMix}
              onChange={gp.setDenoiseMix} min={0} max={1} step={0.01} showInput
              infoMeta="default 0.25 · range 0-1"
              info="Blends the denoised signal back with the original. 0 is fully dry (original); 1 is fully denoised." />
          </div>
        )}
      </div>

      {/* ── LSS: Latent Spectral Suppressor (Accordion with toggle) ── */}
      <div className="rounded-xl border border-teal-500/20 bg-teal-500/5 transition-all overflow-hidden">
        <button
          type="button"
          onClick={() => setLssOpen(!lssOpen)}
          className="w-full flex items-center justify-between px-3 py-2 hover:bg-teal-500/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ChevronDown size={12} className={`text-teal-400 transition-transform duration-200 ${lssOpen ? 'rotate-180' : ''}`} />
            <div className="flex items-center gap-1.5" onClick={e => e.stopPropagation()}>
              <Toggle checked={lssStrength > 0} onChange={(on) => setLssStrength(on ? 0.65 : 0)} accent="teal" />
              <ParamLabel label="LSS" underline={false}
                className="text-[10px] font-semibold text-teal-400 uppercase tracking-wider"
                info="Latent Spectral Suppressor (MDMAchine): gates quiet latent channels before VAE decode. Channels below the variance threshold are attenuated toward 1 minus strength." />
            </div>
          </div>
          {lssStrength > 0 && (
            <span onClick={(e) => {
              e.stopPropagation();
              setLssStrength(0.0);
              setLssVarThresh(0.15);
              setLssDcRemove(true);
            }} className="flex items-center gap-1 text-[10px] text-teal-400 hover:text-teal-300 transition-colors cursor-pointer">
              <RotateCcw size={10} /> Reset
            </span>
          )}
        </button>
        {lssOpen && lssStrength > 0 && (
          <div className="px-3 pb-3 space-y-3">
            <Slider label="Strength" value={lssStrength}
              onChange={setLssStrength} min={0.01} max={1} step={0.01} showInput
              infoMeta="default 0.65 · range 0.01-1"
              info="How hard quiet latent channels are attenuated. The attenuation floor is 1 minus this value, so 1.0 pulls the quietest channels to silence." />
            <Slider label="Var Threshold" value={lssVarThresh}
              onChange={setLssVarThresh} min={0.01} max={0.5} step={0.01} showInput
              infoMeta="default 0.15 · range 0.01-0.5"
              info="Channels whose variance falls below this, relative to the loudest channel, are treated as quiet and attenuated." />
            <div className="flex items-center justify-between">
              <ParamLabel label="DC Remove" className="text-xs text-zinc-500" rootClassName="flex"
                info="Removes each latent channel's DC offset (constant bias) in addition to the variance gating above. On by default." />
              <Toggle checked={lssDcRemove} onChange={setLssDcRemove} accent="teal" />
            </div>
          </div>
        )}
      </div>

      {/* Seed — shared with the generic backend cluster (SeedControl.tsx) so
          the two can never drift apart. */}
      <SeedControl inputClasses={inputClasses} />

      {/* Batch */}
      <Slider label="Batch Size" value={gp.batchSize}
        onChange={gp.setBatchSize} min={1} max={9} step={1}
        infoMeta="default 1 · range 1-9"
        info="How many takes to render from one request. The LM plans that many variations, then the DiT renders each in turn as its own song. With the LM off, or for cover and repaint tasks, extra takes reuse the same plan with a different random seed each." />
    </div>
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
