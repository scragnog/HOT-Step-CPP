// LmThinkingDropdown.tsx — LM / Thinking settings for the global param bar
//
// The on/off toggle is in the bar header (ToggleSwitch).
// This dropdown only shows the detailed LM parameters when LM is enabled.

import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Save } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { Slider } from '../shared/Slider';
import { ParamLabel } from '../shared/ParamLabel';
import { ToggleSwitch } from './BarSection';
import { SeedManagerDrawer } from './SeedManagerDrawer';
import type { LmRepMode } from '../../types';

/** One line each on what the mode costs you, since the audible trade-off is
 *  the whole point of the choice. */
const REP_MODE_HELP: Record<LmRepMode, string> = {
  presence:
    'Penalises every distinct code in the window once, however often it recurred. Cannot tell a stuck loop from ordinary musical restatement, so strengths that break loops also flatten structure.',
  frequency:
    'Penalty rises with how often each code recurred (capped at 8). Same as Presence for one-off codes, so a loop is hit far harder than the section around it — try a much lower penalty here.',
  dry:
    'Penalises only codes that would extend a verbatim recent cycle, growing exponentially with match length. Leaves non-verbatim restatement alone. Raise Min Match if sustained textures get chewed up.',
};

const inputClasses ="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

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

export const LmThinkingDropdown: React.FC = () => {
  const gp = useGlobalParams();
  const { t } = useTranslation();
  const [seedDrawerOpen, setSeedDrawerOpen] = useState(false);

  if (gp.skipLm) {
    return (
      <div className="text-xs text-zinc-500 italic text-center py-2">
        {t('lm.disabled')}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* CoT Caption */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-zinc-600 dark:text-zinc-400">{t('lm.cotCaption')}</span>
        <ToggleSwitch checked={gp.useCotCaption} onChange={gp.setUseCotCaption} accentColor="purple" />
      </div>

      <Slider label="Temperature" value={gp.lmTemperature}
        onChange={gp.setLmTemperature} min={0} max={2} step={0.01} showInput />

      <Slider label="CFG Scale" value={gp.lmCfgScale}
        onChange={gp.setLmCfgScale} min={0} max={10} step={0.1} showInput />

      <Slider label="Top-K" value={gp.lmTopK}
        onChange={gp.setLmTopK} min={0} max={200} step={1} showInput />

      <Slider label="Top-P" value={gp.lmTopP}
        onChange={gp.setLmTopP} min={0} max={1} step={0.01} showInput />

      {/* Anti-loop: windowed repetition penalty on audio-code sampling.
          1.0 = off. Breaks the stuck-loop failure mode (planner adapters
          sharpen the code distribution into repetition attractors). */}
      <Slider label="Repetition Penalty" value={gp.lmRepPenalty}
        onChange={gp.setLmRepPenalty} min={1.0} max={1.5} step={0.01} showInput />
      {gp.lmRepPenalty > 1.0 && (
        <>
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Rep. Mode</label>
            <select className={inputClasses} value={gp.lmRepMode}
              onChange={e => gp.setLmRepMode(e.target.value as LmRepMode)}>
              <option value="presence">Presence (legacy)</option>
              <option value="frequency">Frequency (count-scaled)</option>
              <option value="dry">DRY (verbatim-loop breaker)</option>
            </select>
            <p className="mt-1.5 text-[11px] leading-snug text-zinc-500">{REP_MODE_HELP[gp.lmRepMode] ?? REP_MODE_HELP.presence}</p>
          </div>

          <Slider label={`Rep. Window (codes) — ${(gp.lmRepWindow / 5).toFixed(1)}s`} value={gp.lmRepWindow}
            onChange={gp.setLmRepWindow} min={8} max={256} step={8} showInput />

          {gp.lmRepMode === 'dry' && (
            <>
              <Slider label="DRY Base" value={gp.lmDryBase}
                onChange={gp.setLmDryBase} min={1.05} max={4} step={0.05} showInput />
              <Slider label={`DRY Min Match — ${(gp.lmDryMinLen / 5).toFixed(1)}s`} value={gp.lmDryMinLen}
                onChange={gp.setLmDryMinLen} min={2} max={32} step={1} showInput />
            </>
          )}
        </>
      )}

      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('lm.negativePrompt')}</label>
        <input className={inputClasses} value={gp.lmNegativePrompt}
          onChange={e => gp.setLmNegativePrompt(e.target.value)}
          placeholder="NO USER INPUT" />
      </div>

      {/* LM codes window: how far into the DiT schedule the codes condition
          generation. Ratio = fraction of the step budget; Step Count = an
          absolute number of steps that stays fixed when the budget changes.
          Both are sent as audio_cover_strength (steps convert at build). */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">LM Codes</label>
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-zinc-500">Step Count</span>
            <ToggleSwitch checked={gp.lmCodesMode === 'steps'}
              onChange={v => gp.setLmCodesMode(v ? 'steps' : 'ratio')} accentColor="sky" />
          </div>
        </div>
        {gp.lmCodesMode === 'steps' ? (
          <>
            <Slider label={`Codes Steps — first ${Math.min(gp.lmCodesSteps, gp.inferenceSteps)} of ${gp.inferenceSteps}`}
              value={gp.lmCodesSteps} onChange={gp.setLmCodesSteps}
              min={0} max={gp.inferenceSteps} step={1} showInput />
            <p className="mt-1.5 text-[11px] leading-snug text-zinc-500">
              Codes condition exactly this many steps, regardless of the Steps setting. At the maximum they apply to every step.
            </p>
          </>
        ) : (
          <Slider label="Strength" value={gp.lmCodesStrength}
            onChange={gp.setLmCodesStrength} min={0} max={1} step={0.05} showInput />
        )}
      </div>

      {/* LM Seed — independent from the Generation (DiT) seed by default,
          unless "Use DiT Seed" is on, which ties lm_seed to the DiT seed
          (the original engine behavior: locked seed -> both deterministic,
          random -> both random). */}
      <div className="relative">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">LM Seed</label>
            <button onClick={() => setSeedDrawerOpen(true)} title="Seed Manager"
              className="text-zinc-500 hover:text-amber-400 transition-colors">
              <Save size={12} />
            </button>
          </div>
          <div className="flex items-center gap-1.5">
            <ParamLabel label="Use DiT Seed" className="text-xs text-zinc-500"
              info={gp.lmSeedFollowsDit
                ? 'Tied to the Generation seed — a locked seed makes both deterministic, a random one makes both random.'
                : 'Drives caption, lyric and audio-code sampling independently of the Generation seed.'} />
            <ToggleSwitch checked={gp.lmSeedFollowsDit} onChange={gp.setLmSeedFollowsDit} accentColor="sky" />
          </div>
        </div>
        {!gp.lmSeedFollowsDit && (
          <SeedInput value={gp.lmSeed} onChange={gp.setLmSeed} className={inputClasses} />
        )}
        <SeedManagerDrawer
          isOpen={seedDrawerOpen}
          onClose={() => setSeedDrawerOpen(false)}
          currentSeed={gp.lmSeed}
          onLoad={(seed) => { gp.setLmSeed(seed); gp.setLmSeedFollowsDit(false); setSeedDrawerOpen(false); }}
          onLoadRandom={(seed) => { gp.setLmSeed(seed); gp.setLmSeedFollowsDit(false); }}
        />
      </div>
    </div>
  );
};

/** Summary badge for the LM / Thinking section */
export const LmThinkingBadge: React.FC = () => {
  const { skipLm, useCotCaption, lmTemperature, lmCfgScale, lmCodesStrength, lmCodesMode, lmCodesSteps, inferenceSteps, lmSeedFollowsDit } = useGlobalParams();

  if (skipLm) return null;

  const seedLabel = lmSeedFollowsDit ? 'DiT' : 'Fix';
  const csLabel = lmCodesMode === 'steps'
    ? (lmCodesSteps < inferenceSteps ? ` · CS ${lmCodesSteps}st` : '')
    : (lmCodesStrength < 1.0 ? ` · CS ${lmCodesStrength.toFixed(2)}` : '');

  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {useCotCaption ? 'CoT · ' : ''}T{lmTemperature.toFixed(2)} · CFG {lmCfgScale.toFixed(1)}{csLabel} · Seed {seedLabel}
    </span>
  );
};
