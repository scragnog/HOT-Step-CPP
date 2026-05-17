// RepaintSettings.tsx — Settings panel + Generate button for Repaint Studio
//
// Controls: repaint mode, crossfade, style caption, generate/cancel.

import React from 'react';
import { Loader2, Paintbrush, X } from 'lucide-react';

interface RepaintSettingsProps {
  // Repaint parameters
  repaintMode: string;
  onRepaintModeChange: (mode: string) => void;
  crossfadeFrames: number;
  onCrossfadeFramesChange: (f: number) => void;
  styleCaption: string;
  onStyleCaptionChange: (s: string) => void;

  // Generation state
  canGenerate: boolean;
  isGenerating: boolean;
  genProgress: number;
  genStage: string;
  onGenerate: () => void;
  onCancel: () => void;
}

const REPAINT_MODES = [
  { value: 'conservative', label: 'Conservative', desc: 'Preserves more of the original character', ratio: 0.7 },
  { value: 'balanced', label: 'Balanced', desc: 'Default blend of original and new content', ratio: 0.5 },
  { value: 'aggressive', label: 'Aggressive', desc: 'More creative freedom in the repainted region', ratio: 0.3 },
];

export const RepaintSettings: React.FC<RepaintSettingsProps> = ({
  repaintMode,
  onRepaintModeChange,
  crossfadeFrames,
  onCrossfadeFramesChange,
  styleCaption,
  onStyleCaptionChange,
  canGenerate,
  isGenerating,
  genProgress,
  genStage,
  onGenerate,
  onCancel,
}) => {

  // Crossfade in seconds (25Hz frame rate)
  const crossfadeSec = (crossfadeFrames / 25).toFixed(2);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-white/5">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Paintbrush size={14} className="text-pink-400" />
          Repaint Settings
        </h3>
      </div>

      <div className="flex-1 p-4 space-y-5">
        {/* Repaint Mode */}
        <div>
          <label className="text-xs font-medium text-zinc-400 mb-2 block">Repaint Mode</label>
          <div className="space-y-1.5">
            {REPAINT_MODES.map(mode => (
              <button
                key={mode.value}
                onClick={() => onRepaintModeChange(mode.value)}
                className={`w-full text-left px-3 py-2 rounded-lg border transition-all text-xs ${
                  repaintMode === mode.value
                    ? 'border-pink-500/40 bg-pink-500/10 text-white'
                    : 'border-white/5 bg-white/[0.02] text-zinc-400 hover:border-white/10 hover:bg-white/5'
                }`}
              >
                <div className="font-medium">{mode.label}</div>
                <div className="text-[10px] opacity-60 mt-0.5">{mode.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Crossfade */}
        <div>
          <label className="text-xs font-medium text-zinc-400 mb-1.5 block">
            Boundary Crossfade
            <span className="ml-2 text-zinc-600 font-mono">{crossfadeSec}s ({crossfadeFrames} frames)</span>
          </label>
          <input
            type="range"
            min={0}
            max={30}
            step={1}
            value={crossfadeFrames}
            onChange={e => onCrossfadeFramesChange(parseInt(e.target.value))}
            className="w-full accent-pink-500"
          />
          <div className="flex justify-between text-[10px] text-zinc-600 mt-0.5">
            <span>Hard cut</span>
            <span>Smooth blend</span>
          </div>
        </div>

        {/* Style Caption */}
        <div>
          <label className="text-xs font-medium text-zinc-400 mb-1.5 block">
            Style Description
          </label>
          <textarea
            value={styleCaption}
            onChange={e => onStyleCaptionChange(e.target.value)}
            className="w-full h-24 resize-none bg-black/20 border border-white/10 rounded-xl
              px-3 py-2 text-xs text-white placeholder-zinc-600 focus:outline-none
              focus:border-pink-500 transition-colors leading-relaxed"
            placeholder="Style from original song will be used if left empty..."
          />
        </div>
      </div>

      {/* Generate / Cancel */}
      <div className="flex-shrink-0 p-4 border-t border-white/5">
        {isGenerating ? (
          <div className="space-y-3">
            {/* Progress bar */}
            <div className="relative h-2 bg-white/5 rounded-full overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-pink-500 to-purple-500 rounded-full transition-all duration-500"
                style={{ width: `${genProgress}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-400 flex items-center gap-2">
                <Loader2 size={12} className="animate-spin text-pink-400" />
                {genStage || 'Generating...'}
              </span>
              <span className="font-mono text-zinc-500">{genProgress}%</span>
            </div>
            <button
              onClick={onCancel}
              className="w-full px-4 py-2 rounded-xl border border-red-500/30 text-red-400 text-xs
                font-medium hover:bg-red-500/10 transition-colors flex items-center justify-center gap-2"
            >
              <X size={12} /> Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={onGenerate}
            disabled={!canGenerate}
            className="w-full px-4 py-3 rounded-xl font-semibold text-sm transition-all
              bg-gradient-to-r from-pink-500 to-purple-500 text-white
              hover:from-pink-400 hover:to-purple-400 hover:shadow-lg hover:shadow-pink-500/20
              active:scale-[0.98]
              disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-none
              flex items-center justify-center gap-2"
          >
            <Paintbrush size={16} />
            Repaint Region
          </button>
        )}
      </div>
    </div>
  );
};
