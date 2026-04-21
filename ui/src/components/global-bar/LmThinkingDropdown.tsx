// LmThinkingDropdown.tsx — LM / Thinking settings for the global param bar
//
// The on/off toggle is in the bar header (ToggleSwitch).
// This dropdown only shows the detailed LM parameters when LM is enabled.

import React from 'react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { Slider } from '../shared/Slider';

const inputClasses = "w-full px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

export const LmThinkingDropdown: React.FC = () => {
  const gp = useGlobalParams();

  if (gp.skipLm) {
    return (
      <div className="text-xs text-zinc-500 italic text-center py-2">
        LM conditioning is disabled. Toggle it on in the bar above.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* CoT Caption */}
      <label className="flex items-center gap-2.5 cursor-pointer">
        <input type="checkbox" checked={gp.useCotCaption}
          onChange={e => gp.setUseCotCaption(e.target.checked)}
          className="rounded border-zinc-600 bg-zinc-800 text-pink-500 focus:ring-pink-500/20" />
        <span className="text-sm text-zinc-400">Chain-of-Thought Caption</span>
      </label>

      <Slider label="Temperature" value={gp.lmTemperature}
        onChange={gp.setLmTemperature} min={0} max={2} step={0.01} showInput />

      <Slider label="CFG Scale" value={gp.lmCfgScale}
        onChange={gp.setLmCfgScale} min={0} max={10} step={0.1} showInput />

      <Slider label="Top-K" value={gp.lmTopK}
        onChange={gp.setLmTopK} min={0} max={200} step={1} showInput />

      <Slider label="Top-P" value={gp.lmTopP}
        onChange={gp.setLmTopP} min={0} max={1} step={0.01} showInput />

      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Negative Prompt</label>
        <input className={inputClasses} value={gp.lmNegativePrompt}
          onChange={e => gp.setLmNegativePrompt(e.target.value)}
          placeholder="NO USER INPUT" />
      </div>
    </div>
  );
};

/** Summary badge for the LM / Thinking section */
export const LmThinkingBadge: React.FC = () => {
  const { skipLm, useCotCaption } = useGlobalParams();

  return (
    <div className="flex items-center gap-1.5">
      {!skipLm && useCotCaption && (
        <span className="text-[10px] text-purple-400/60">CoT</span>
      )}
    </div>
  );
};
