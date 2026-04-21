// LmThinkingDropdown.tsx — LM / Thinking settings for the global param bar
//
// Adapted from the LM section of create/GenerationSettings.tsx.
// Reads from GlobalParamsContext instead of props.

import React from 'react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { Slider } from '../shared/Slider';

const inputClasses = "w-full px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

export const LmThinkingDropdown: React.FC = () => {
  const gp = useGlobalParams();

  return (
    <div className="space-y-3">
      {/* LM Toggle */}
      <label className="flex items-center gap-2.5 cursor-pointer">
        <input type="checkbox" checked={!gp.skipLm}
          onChange={e => gp.setSkipLm(!e.target.checked)}
          className="rounded border-zinc-600 bg-zinc-800 text-pink-500 focus:ring-pink-500/20" />
        <span className="text-sm text-zinc-400">Enable LM Conditioning</span>
      </label>

      {!gp.skipLm && (
        <>
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
        </>
      )}
    </div>
  );
};

/** Summary badge for the LM / Thinking section */
export const LmThinkingBadge: React.FC = () => {
  const { skipLm, useCotCaption } = useGlobalParams();

  return (
    <div className="flex items-center gap-1.5">
      <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
        skipLm ? 'bg-zinc-700 text-zinc-400' : 'bg-purple-500/20 text-purple-400'
      }`}>
        {skipLm ? 'OFF' : 'ON'}
      </span>
      {!skipLm && useCotCaption && (
        <span className="text-[10px] text-purple-400/60">CoT</span>
      )}
    </div>
  );
};
