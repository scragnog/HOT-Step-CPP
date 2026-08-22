// StreamWaveform.tsx — the play bar's waveform for a track that is still rendering.
//
// WaveSurfer is not usable here and that is not a shortcut. It is built around a
// media element with a known duration and a fully-downloadable source; a live
// render has no URL, no final length until the planner stops, and an end that
// moves every few seconds. Feeding it a growing peaks array means re-`load()`ing
// on every window, which resets playback. So: a canvas, drawing the envelope
// mm3StreamStore accumulates as windows land.
//
// It draws three regions, and the difference between the last two is the point
// of the whole feature:
//
//   played      pink → purple gradient, matching the normal waveform's progress
//   rendered    dim, seekable — audio that exists and can be jumped to
//   not yet     a flat baseline over the span the render has not reached
//
// The width of the canvas is the FULL expected track, not the part received, so
// the waveform grows into place instead of rescaling under the playhead every
// time a window arrives.

import React, { useEffect, useRef } from 'react';
import { useMm3StreamAudio, mm3StreamPeaks, mm3StreamSeek } from '../../stores/mm3StreamStore';

interface StreamWaveformProps {
  height?: number;
}

export const StreamWaveform: React.FC<StreamWaveformProps> = ({ height = 56 }) => {
  const s = useMm3StreamAudio();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Redraw on new audio (peaksVersion) and on every position tick. Both come
  // through the store, so there is no second animation loop competing with the
  // one that drives the transport.
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = wrap.clientWidth;
    if (cssW <= 0) return;
    if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${height}px`;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, height);

    const { peaks, count, rate } = mm3StreamPeaks();
    // Total span drawn. Before the first window `expected` is all we have; once
    // the engine closes the stream it IS the received length, so the bars stop
    // moving the moment the render ends.
    const total = Math.max(s.expected, s.received, 0.001);
    const mid = height / 2;
    const barW = 2, gap = 1, step = barW + gap;
    const bars = Math.floor(cssW / step);

    const grad = ctx.createLinearGradient(0, 0, cssW, 0);
    grad.addColorStop(0, '#ec4899');
    grad.addColorStop(1, '#a855f7');

    for (let b = 0; b < bars; b++) {
      const x = b * step;
      const tFrom = (b / bars) * total;
      const tTo = ((b + 1) / bars) * total;

      // Peak bucket range for this bar.
      const iFrom = Math.floor(tFrom * rate);
      const iTo = Math.min(count, Math.ceil(tTo * rate));
      let amp = 0;
      for (let i = iFrom; i < iTo; i++) if (peaks[i] > amp) amp = peaks[i];

      const rendered = tFrom < s.received;
      const played = tTo <= s.position || (tFrom < s.position && s.position < tTo);

      if (!rendered) {
        // Not generated yet — a baseline, so the bar reads as "coming" rather
        // than as a silent passage in the music.
        ctx.fillStyle = 'rgba(113, 113, 122, 0.18)';
        ctx.fillRect(x, mid - 0.5, barW, 1);
        continue;
      }

      const h = Math.max(2, amp * (height - 4));
      ctx.fillStyle = played ? grad : 'rgba(113, 113, 122, 0.45)';
      ctx.fillRect(x, mid - h / 2, barW, h);
    }

    // Playhead.
    if (s.position > 0) {
      const px = (s.position / total) * cssW;
      ctx.fillStyle = '#ec4899';
      ctx.fillRect(px - 1, 0, 2, height);
    }

    // The boundary between rendered and not — the edge that advances as the
    // engine works. Amber to match the card's "generating" treatment.
    if (s.received < total - 0.05) {
      const rx = (s.received / total) * cssW;
      ctx.fillStyle = 'rgba(249, 115, 22, 0.55)';
      ctx.fillRect(rx, 0, 1.5, height);
    }
  }, [s.peaksVersion, s.position, s.received, s.expected, height]);

  return (
    <div
      ref={wrapRef}
      className="relative w-full cursor-pointer select-none"
      style={{ height }}
      onClick={(e) => {
        const wrap = wrapRef.current;
        if (!wrap) return;
        const rect = wrap.getBoundingClientRect();
        const frac = (e.clientX - rect.left) / rect.width;
        const total = Math.max(s.expected, s.received, 0.001);
        // Seeking past the render is meaningless — the store clamps, and the
        // click reads as "take me to the newest audio", which is what a user
        // clicking into the unrendered region actually wants.
        mm3StreamSeek(frac * total);
      }}
      title="Click to seek within the audio rendered so far"
    >
      <canvas ref={canvasRef} className="block w-full" />
    </div>
  );
};
