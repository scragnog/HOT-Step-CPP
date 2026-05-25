// DiscoPulseWrapper.tsx — Beat-reactive panel wrapper for disco mode.
//
// Wraps a UI panel to apply scale transform + multi-coloured box-shadow glow
// that pulses with the kick drum frequency. Each instance receives a unique
// glow colour and optional stagger delay for a left-to-right ripple effect.
//
// Uses direct DOM style manipulation (via ref) to avoid React re-renders at
// 60fps — only the disco store notifies at ~60fps, and we read the value in
// a useEffect that writes straight to the DOM element.

import React, { useRef, useEffect } from 'react';
import { usePulseIntensity, useDiscoMode } from '../../stores/discoStore';

interface DiscoPulseWrapperProps {
  /** Neon glow colour for this panel (hex, e.g. '#ff1493') */
  glowColor: string;
  /** Stagger delay in ms — creates a left-to-right ripple effect */
  staggerMs?: number;
  /** Extra CSS classes to pass through */
  className?: string;
  /** Inline styles to pass through */
  style?: React.CSSProperties;
  children: React.ReactNode;
}

export const DiscoPulseWrapper: React.FC<DiscoPulseWrapperProps> = ({
  glowColor,
  staggerMs = 0,
  className = '',
  style,
  children,
}) => {
  const discoMode = useDiscoMode();
  const pulseIntensity = usePulseIntensity();
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Apply visual effects directly to DOM to avoid React re-render churn
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;

    if (!discoMode || pulseIntensity < 0.01) {
      el.style.transform = '';
      el.style.boxShadow = '';
      return;
    }

    // Apply with stagger delay
    const timer = staggerMs > 0
      ? setTimeout(() => applyPulse(el, pulseIntensity, glowColor), staggerMs)
      : (applyPulse(el, pulseIntensity, glowColor), null);

    return () => {
      if (timer !== null) clearTimeout(timer);
    };
  }, [pulseIntensity, discoMode, staggerMs, glowColor]);

  // Clean up styles when disco mode is toggled off
  useEffect(() => {
    if (!discoMode) {
      const el = wrapperRef.current;
      if (el) {
        el.style.transform = '';
        el.style.boxShadow = '';
      }
    }
  }, [discoMode]);

  return (
    <div
      ref={wrapperRef}
      className={`disco-pulse-wrapper ${className}`}
      style={style}
      data-disco-active={discoMode || undefined}
    >
      {children}
    </div>
  );
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function applyPulse(el: HTMLElement, intensity: number, color: string): void {
  // Scale: 1.0 → 1.012 at full intensity
  const scale = 1 + intensity * 0.012;
  el.style.transform = `scale(${scale})`;

  // Multi-layer glow: tight inner + mid spread + wide ambient
  const tight = Math.round(4 * intensity);
  const mid = Math.round(12 * intensity);
  const wide = Math.round(24 * intensity);
  el.style.boxShadow = [
    `0 0 ${tight}px 0 ${color}`,
    `0 0 ${mid}px 0 ${hexToRgba(color, 0.6)}`,
    `0 0 ${wide}px 0 ${hexToRgba(color, 0.25)}`,
  ].join(', ');
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
