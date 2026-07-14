// LmThinkingDropdown.tsx — LM / Thinking settings for the global param bar
//
// The on/off toggle is in the bar header (ToggleSwitch).
// This dropdown only shows the detailed LM parameters when LM is enabled.

import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Save } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { Slider } from '../shared/Slider';
import { ToggleSwitch } from './BarSection';
import { SeedManagerDrawer } from './SeedManagerDrawer';

const inputClasses = "w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

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

      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('lm.negativePrompt')}</label>
        <input className={inputClasses} value={gp.lmNegativePrompt}
          onChange={e => gp.setLmNegativePrompt(e.target.value)}
          placeholder="NO USER INPUT" />
      </div>

      <Slider label="LM Codes Strength" value={gp.lmCodesStrength}
        onChange={gp.setLmCodesStrength} min={0} max={1} step={0.05} showInput />

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
            <span className="text-xs text-zinc-500">Use DiT Seed</span>
            <ToggleSwitch checked={gp.lmSeedFollowsDit} onChange={gp.setLmSeedFollowsDit} accentColor="sky" />
          </div>
        </div>
        {!gp.lmSeedFollowsDit && (
          <SeedInput value={gp.lmSeed} onChange={gp.setLmSeed} className={inputClasses} />
        )}
        <p className="text-[10px] text-zinc-500 mt-1">
          {gp.lmSeedFollowsDit
            ? 'Tied to the Generation seed — locked seed means both are deterministic, random means both are random.'
            : 'Drives caption/lyrics/audio-code sampling independently of the Generation seed.'}
        </p>
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
  const { skipLm, useCotCaption, lmTemperature, lmCfgScale, lmCodesStrength, lmSeedFollowsDit } = useGlobalParams();

  if (skipLm) return null;

  const seedLabel = lmSeedFollowsDit ? 'DiT' : 'Fix';

  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {useCotCaption ? 'CoT · ' : ''}T{lmTemperature.toFixed(2)} · CFG {lmCfgScale.toFixed(1)}{lmCodesStrength < 1.0 ? ` · CS ${lmCodesStrength.toFixed(2)}` : ''} · Seed {seedLabel}
    </span>
  );
};
