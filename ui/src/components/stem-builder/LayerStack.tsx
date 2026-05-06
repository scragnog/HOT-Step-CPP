// LayerStack.tsx — Iterative composition layer history
//
// Shows the stack of stems generated so far in a Stem Builder session.
// Each layer shows the track name, caption, and a play button.
// The "Use as New Source" button enables iterative composition.

import React from 'react';
import { Play, Layers, ArrowUp } from 'lucide-react';

export interface LayerInfo {
  trackName: string;
  caption: string;
  audioUrl: string;
  songId?: string;
  timestamp: number;
}

/** Display labels with emoji icons (mirrors TrackPicker) */
const TRACK_EMOJI: Record<string, string> = {
  vocals: '🎤', backing_vocals: '🎙️', drums: '🥁', bass: '🎵',
  guitar: '🎸', keyboard: '🎹', percussion: '🪘', strings: '🎻',
  synth: '🎛️', fx: '🔊', brass: '🎺', woodwinds: '🪕',
};

interface LayerStackProps {
  layers: LayerInfo[];
  sourceFileName: string;
  onPlayLayer: (layer: LayerInfo) => void;
  onUseAsSource: (layer: LayerInfo) => void;
}

export const LayerStack: React.FC<LayerStackProps> = ({
  layers,
  sourceFileName,
  onPlayLayer,
  onUseAsSource,
}) => {
  if (layers.length === 0) {
    return null;
  }

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2">
        <Layers size={13} className="text-amber-400" />
        <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
          Layer Stack ({layers.length})
        </span>
      </div>

      <div className="flex flex-col gap-1">
        {/* Layers — most recent first */}
        {[...layers].reverse().map((layer, idx) => (
          <div
            key={layer.timestamp}
            className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06] group hover:border-amber-500/20 transition-colors"
          >
            <span className="text-sm flex-shrink-0">
              {TRACK_EMOJI[layer.trackName] || '🎵'}
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-medium text-zinc-300 capitalize truncate">
                {layer.trackName.replace('_', ' ')}
              </div>
              {layer.caption && (
                <div className="text-[10px] text-zinc-600 truncate">{layer.caption}</div>
              )}
            </div>
            <button
              type="button"
              onClick={() => onPlayLayer(layer)}
              className="flex-shrink-0 w-6 h-6 rounded-full bg-white/[0.06] flex items-center justify-center hover:bg-amber-500/20 transition-colors"
              title="Play this layer"
            >
              <Play size={10} className="text-zinc-400 group-hover:text-amber-400" />
            </button>
            {idx === 0 && (
              <button
                type="button"
                onClick={() => onUseAsSource(layer)}
                className="flex-shrink-0 px-2 py-0.5 rounded text-[10px] font-medium bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 transition-colors"
                title="Use this output as the source for the next layer"
              >
                <ArrowUp size={10} className="inline mr-0.5" />
                Source
              </button>
            )}
          </div>
        ))}

        {/* Original source */}
        <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-white/[0.02] border border-dashed border-white/[0.06]">
          <span className="text-sm flex-shrink-0">📁</span>
          <div className="flex-1 min-w-0">
            <div className="text-[10px] text-zinc-600 truncate">
              Original: {sourceFileName || 'backing track'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
