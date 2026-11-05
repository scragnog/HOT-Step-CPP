// LyricPeaks.tsx — the waveform above the timeline, drawn for the visible window.
//
// The canvas and its wrapper are own refs here rather than the modal's, because
// the only thing that needs them is the hook that paints into them. Sizing and
// sampling belong to `useVisiblePeaks`; this file is the strip it paints on.

import React, { useRef } from 'react';
import { PEAKS_H } from './timingModel';
import { useVisiblePeaks, type PeaksLike } from './useVisiblePeaks';

export interface LyricPeaksProps {
  peaks: PeaksLike | null | undefined;
  total: number;
  pxPerSec: number;
  scrollLeft: number;
}

export const LyricPeaks: React.FC<LyricPeaksProps> = ({ peaks, total, pxPerSec, scrollLeft }) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useVisiblePeaks({
    wrapRef, canvasRef, peaks, total, pxPerSec, scrollLeft, height: PEAKS_H,
  });

  return (
    <div
      ref={wrapRef}
      className="shrink-0 border-b border-zinc-200 bg-zinc-100 dark:border-white/10 dark:bg-black/40"
      style={{ height: PEAKS_H }}
    >
      <canvas ref={canvasRef} className="block" />
    </div>
  );
};

export default LyricPeaks;
