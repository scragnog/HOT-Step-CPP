// Waveform.tsx — the play bar's waveform, drawn from server-computed peaks.
//
// This is a view. It reads the engine and paints; it never starts, stops or
// loads anything, and the engine does not know it exists. Deleting this file
// would cost the app its waveform and nothing else.
//
// It replaces a WaveSurfer instance per variant. WaveSurfer's own drawing was
// never the problem — its loader was, because getting a waveform out of it
// meant letting it download and decode the file. With peaks arriving as ~25 KB
// of JSON there is nothing left for it to do that a canvas cannot.
//
// The playhead moves on the engine's frame channel rather than through React,
// so a moving cursor costs one canvas blit rather than a re-render of the tree.

import React, { useEffect, useRef } from 'react';
import { useSyncExternalStore } from 'react';
import {
  subscribe, getSnapshot, subscribeFrame, seekFraction,
} from '../../audio/engine';

interface WaveformProps {
  height?: number;
  /** Trim mode's IN/OUT points, in seconds. The region outside them is dimmed
   *  and the boundaries get a coloured edge. */
  trimIn?: number | null;
  trimOut?: number | null;
}

const BAR_W = 2;
const BAR_GAP = 1;

export const Waveform: React.FC<WaveformProps> = ({ height = 56, trimIn, trimOut }) => {
  const state = useSyncExternalStore(subscribe, getSnapshot);
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  /** The waveform without any progress colouring, drawn once per peaks/size
   *  change. Every frame is then a blit plus one clipped gradient fill. */
  const baseRef = useRef<HTMLCanvasElement | null>(null);
  const hoverRef = useRef<number | null>(null);
  const [hoverX, setHoverX] = React.useState<number | null>(null);

  const { peaks, duration, currentTime } = state;

  // ── Base layer: the bars, in their unplayed colour ────────────────────────
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const draw = () => {
      const cssW = wrap.clientWidth;
      if (cssW <= 0) return;
      const dpr = window.devicePixelRatio || 1;

      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${height}px`;

      const base = baseRef.current ?? document.createElement('canvas');
      baseRef.current = base;
      base.width = canvas.width;
      base.height = canvas.height;

      const ctx = base.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, height);

      const mid = height / 2;
      const step = BAR_W + BAR_GAP;
      const bars = Math.max(1, Math.floor(cssW / step));

      ctx.fillStyle = 'rgba(113, 113, 122, 0.45)';

      if (!peaks || peaks.min.length === 0) {
        // No envelope: either it is still in flight or the server could not
        // decode the file. Either way a flat line reads as "waveform pending",
        // and the track plays regardless.
        ctx.fillStyle = 'rgba(113, 113, 122, 0.25)';
        ctx.fillRect(0, mid - 0.5, cssW, 1);
      } else {
        const buckets = peaks.min.length;
        for (let b = 0; b < bars; b++) {
          // Every bucket contributes to some bar, so a long track's transients
          // survive instead of being sampled past.
          const from = Math.floor((b / bars) * buckets);
          const to = Math.max(from + 1, Math.floor(((b + 1) / bars) * buckets));
          let lo = 0, hi = 0;
          for (let i = from; i < to && i < buckets; i++) {
            if (peaks.min[i] < lo) lo = peaks.min[i];
            if (peaks.max[i] > hi) hi = peaks.max[i];
          }
          const top = mid - hi * (mid - 2);
          const bottom = mid - lo * (mid - 2);
          ctx.fillRect(b * step, top, BAR_W, Math.max(2, bottom - top));
        }
      }

      paint();
    };

    draw();

    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [peaks, height]);

  // ── Composite: base, progress gradient, trim dimming, playhead ────────────
  const paint = React.useCallback((timeOverride?: number) => {
    const canvas = canvasRef.current;
    const base = baseRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !base || !wrap) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = wrap.clientWidth;
    if (cssW <= 0) return;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(base, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const dur = duration || 0;
    const t = timeOverride ?? currentTime;
    const playedX = dur > 0 ? Math.min(cssW, (t / dur) * cssW) : 0;

    // Colour the played bars by painting a gradient only where the base layer
    // already put ink. Cheaper and sharper than redrawing every bar.
    if (playedX > 0) {
      const grad = ctx.createLinearGradient(0, 0, cssW, 0);
      grad.addColorStop(0, '#ec4899');
      grad.addColorStop(1, '#a855f7');
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, playedX, height);
      ctx.clip();
      ctx.globalCompositeOperation = 'source-atop';
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, playedX, height);
      ctx.restore();
    }

    // Trim mode: dim everything that would be cut, mark the boundaries.
    if (dur > 0 && (trimIn != null || trimOut != null)) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
      if (trimIn != null && trimIn > 0) {
        ctx.fillRect(0, 0, (trimIn / dur) * cssW, height);
      }
      if (trimOut != null && trimOut < dur) {
        const x = (trimOut / dur) * cssW;
        ctx.fillRect(x, 0, cssW - x, height);
      }
      if (trimIn != null) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.9)';
        ctx.fillRect((trimIn / dur) * cssW - 1, 0, 2, height);
      }
      if (trimOut != null) {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
        ctx.fillRect((trimOut / dur) * cssW - 1, 0, 2, height);
      }
    }

    // Playhead.
    if (dur > 0 && t > 0) {
      ctx.fillStyle = '#ec4899';
      ctx.fillRect(playedX - 1, 0, 2, height);
    }

    // Hover line, the WaveSurfer hover plugin's job. The timestamp itself is a
    // DOM label below, which is easier to read than canvas text.
    const hx = hoverRef.current;
    if (hx != null) {
      ctx.fillStyle = 'rgba(168, 85, 247, 0.9)';
      ctx.fillRect(hx - 1, 0, 2, height);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [duration, currentTime, height, trimIn, trimOut]);

  // Repaint on any state the composite depends on.
  useEffect(() => { paint(); }, [paint]);

  // Playhead at frame rate, off the React path entirely.
  useEffect(() => subscribeFrame((t) => paint(t)), [paint]);

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div
      ref={wrapRef}
      className="relative w-full cursor-pointer select-none"
      style={{ height }}
      onClick={(e) => {
        const wrap = wrapRef.current;
        if (!wrap || !duration) return;
        const rect = wrap.getBoundingClientRect();
        seekFraction((e.clientX - rect.left) / rect.width);
      }}
      onMouseMove={(e) => {
        const wrap = wrapRef.current;
        if (!wrap) return;
        const x = e.clientX - wrap.getBoundingClientRect().left;
        hoverRef.current = x;
        setHoverX(x);
        paint();
      }}
      onMouseLeave={() => {
        hoverRef.current = null;
        setHoverX(null);
        paint();
      }}
    >
      <canvas ref={canvasRef} className="block w-full" />
      {hoverX != null && duration > 0 && (
        <div
          className="absolute top-0 px-1 py-0.5 text-[10px] font-mono rounded bg-zinc-900 text-zinc-200 pointer-events-none"
          style={{ left: Math.min(hoverX + 4, (wrapRef.current?.clientWidth ?? 0) - 40) }}
        >
          {fmt((hoverX / Math.max(1, wrapRef.current?.clientWidth ?? 1)) * duration)}
        </div>
      )}
    </div>
  );
};
