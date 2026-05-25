// DiscoPulseWrapper.tsx — Beat-reactive panel wrapper for disco mode.
//
// Wraps a UI panel to apply scale transform + multi-coloured INSET box-shadow
// glow that pulses with kick drum transients. Uses inset shadows because
// regular box-shadow is clipped by overflow:hidden on parent containers.
//
// Uses direct DOM style manipulation (via ref) to avoid React re-renders at
// 60fps — the disco store notifies at ~60fps, and we write straight to the
// DOM element.

import React, { useRef, useEffect } from 'react';
import { usePulseIntensity, useDiscoMode } from '../../stores/discoStore';

interface DiscoPulseWrapperProps {
  /** Neon glow colour for this panel (hex, e.g. '#ff1493') */
  glowColor: string;
  /** Stagger delay in ms — creates a left-to-right ripple via CSS transition-delay */
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

  // Apply visual effects directly to DOM to avoid React re-render churn.
  // This runs on every pulseIntensity change (~60fps when active).
  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;

    if (!discoMode || pulseIntensity < 0.005) {
      el.style.transform = '';
      el.style.boxShadow = '';
      el.style.filter = '';
      return;
    }

    applyPulse(el, pulseIntensity, glowColor);
  }, [pulseIntensity, discoMode, glowColor]);

  // Clean up styles when disco mode is toggled off
  useEffect(() => {
    if (!discoMode) {
      const el = wrapperRef.current;
      if (el) {
        el.style.transform = '';
        el.style.boxShadow = '';
        el.style.filter = '';
      }
    }
  }, [discoMode]);

  return (
    <div
      ref={wrapperRef}
      className={`disco-pulse-wrapper ${className}`}
      style={{
        ...style,
        transitionDelay: staggerMs > 0 ? `${staggerMs}ms` : undefined,
      }}
      data-disco-active={discoMode || undefined}
    >
      {children}
    </div>
  );
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function applyPulse(el: HTMLElement, intensity: number, color: string): void {
  // Scale: 1.0 → 1.04 at full intensity (4% — clearly visible pop)
  const scale = 1 + intensity * 0.04;
  el.style.transform = `scale(${scale})`;

  // INSET multi-layer glow — not clipped by overflow:hidden on parents!
  const tight = Math.round(8 * intensity);
  const mid = Math.round(20 * intensity);
  const wide = Math.round(40 * intensity);
  el.style.boxShadow = [
    `inset 0 0 ${tight}px ${Math.round(2 * intensity)}px ${color}`,
    `inset 0 0 ${mid}px ${Math.round(4 * intensity)}px ${hexToRgba(color, 0.45)}`,
    `inset 0 0 ${wide}px ${Math.round(6 * intensity)}px ${hexToRgba(color, 0.15)}`,
    // Outer glow for edge-positioned panels (player bar, sidebar edges)
    `0 0 ${Math.round(12 * intensity)}px ${Math.round(2 * intensity)}px ${hexToRgba(color, 0.5)}`,
  ].join(', ');

  // Combined filter: brightness pop + slow hue drift
  const brightness = 1 + intensity * 0.15;
  const hueShift = Math.sin(performance.now() / 4000) * 15;
  el.style.filter = `brightness(${brightness}) hue-rotate(${hueShift.toFixed(1)}deg)`;
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
