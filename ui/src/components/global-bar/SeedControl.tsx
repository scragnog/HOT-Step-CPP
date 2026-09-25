// SeedControl.tsx — the shared "Generation Seed" block
//
// Extracted from GenerationDropdown so the ACE cluster and the generic
// backend cluster render the SAME control rather than two that drift apart.
// It writes to the same globalParams fields (seed / randomSeed), so a seed set
// under one backend is the seed used by the other — which is what "the seed"
// means to a user.

import React, { useEffect, useState } from 'react';
import { Minus, Plus, Save } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { ToggleSwitch } from './BarSection';
import { SeedManagerDrawer } from './SeedManagerDrawer';
import { ParamLabel } from '../shared/ParamLabel';

// Matches the seed range used elsewhere in the app (StormLiveControls, TrainDitForm).
const MAX_SEED = 2147483647;

/** Seed input with local string buffer — prevents parseInt("-") snap-back. */
const SeedInput: React.FC<{ value: number; onChange: (v: number) => void; className: string }> = ({
  value, onChange, className,
}) => {
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

export const SeedControl: React.FC<{ inputClasses: string; hint?: string }> = ({ inputClasses, hint }) => {
  const { t } = useTranslation();
  const gp = useGlobalParams();
  const [seedDrawerOpen, setSeedDrawerOpen] = useState(false);
  const stepButtonClasses = 'flex-shrink-0 p-1.5 rounded-lg text-zinc-500 hover:text-amber-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:text-zinc-500 disabled:hover:bg-transparent';

  return (
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
          <ParamLabel label="Random" className="text-xs text-zinc-500"
            info={hint ?? 'Drives audio synthesis (DiT). Varies per track during batch generation. See LM Seed for caption, lyrics and code sampling.'} />
          <ToggleSwitch checked={gp.randomSeed} onChange={gp.setRandomSeed} accentColor="sky" />
        </div>
      </div>
      {!gp.randomSeed && (
        <div className="flex items-center gap-1">
          <button type="button" title={t('seed.decrement')} aria-label={t('seed.decrement')}
            disabled={gp.randomSeed} className={stepButtonClasses}
            onClick={() => gp.setSeed(Math.max(0, gp.seed - 1))}>
            <Minus size={12} />
          </button>
          <SeedInput value={gp.seed} onChange={gp.setSeed} className={inputClasses} />
          <button type="button" title={t('seed.increment')} aria-label={t('seed.increment')}
            disabled={gp.randomSeed} className={stepButtonClasses}
            onClick={() => gp.setSeed(Math.min(MAX_SEED, gp.seed + 1))}>
            <Plus size={12} />
          </button>
        </div>
      )}
      <SeedManagerDrawer
        isOpen={seedDrawerOpen}
        onClose={() => setSeedDrawerOpen(false)}
        currentSeed={gp.seed}
        onLoad={(seed) => { gp.setSeed(seed); gp.setRandomSeed(false); setSeedDrawerOpen(false); }}
        onLoadRandom={(seed) => { gp.setSeed(seed); gp.setRandomSeed(false); }}
      />
    </div>
  );
};
