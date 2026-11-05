// useVisiblePeaks.ts — draw only the part of the waveform that is on screen.
//
// A canvas as wide as a four-minute track at 120 px/s would be 28,800 px of ink
// per device pixel, so the peaks are sampled for the visible window each time
// the window moves. Reads the same bucket arrays the player's waveform uses.
//
// This one returns nothing, unlike the editor's other hooks: it paints and has
// no state of its own, so there is no `data` and no handlers to hand out. The
// refs and the values to paint arrive as options.

import { useEffect, type RefObject } from 'react';

export interface PeaksLike {
  min: ArrayLike<number>;
  max: ArrayLike<number>;
}

export interface VisiblePeaksOptions {
  wrapRef: RefObject<HTMLDivElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  peaks: PeaksLike | null | undefined;
  total: number;
  pxPerSec: number;
  scrollLeft: number;
  height: number;
}

export function useVisiblePeaks({
  wrapRef, canvasRef, peaks, total, pxPerSec, scrollLeft, height,
}: VisiblePeaksOptions) {
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const cssW = wrap.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(cssW * dpr));
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, height);
    if (!peaks || peaks.min.length === 0 || total <= 0) return;
    const mid = height / 2;
    ctx.fillStyle = 'rgba(161, 161, 170, 0.75)';
    const buckets = peaks.min.length;
    for (let x = 0; x < cssW; x += 2) {
      const time = (scrollLeft + x) / pxPerSec;
      const b = Math.min(buckets - 1, Math.max(0, Math.floor((time / total) * buckets)));
      const top = mid - peaks.max[b] * (mid - 1);
      const bottom = mid - peaks.min[b] * (mid - 1);
      ctx.fillRect(x, top, 1, Math.max(1, bottom - top));
    }
  }, [wrapRef, canvasRef, peaks, total, pxPerSec, scrollLeft, height]);
}
