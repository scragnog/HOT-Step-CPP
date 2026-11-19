// DiscoPulseWrapper.tsx — iOS 18-style animated rainbow border for disco mode.
//
// Uses a spinning conic-gradient masked to a border ring around the panel.
// CSS mask-composite: exclude cuts out the interior, so the gradient is
// ONLY visible as a glowing border — no layout changes or margin tricks needed.
// Also applies a subtle scale pulse on drum hits for a physical "jump" feel.
//
// Structure:
//   <wrapper>          ← position: relative, isolation: isolate, transform: scale()
//     <glow>           ← spinning gradient, masked to border ring
//       <glow-bg>      ← oversized spinning conic-gradient
//     </glow>
//     <content>        ← children (unchanged layout)
//   </wrapper>

import React, { useRef, useEffect } from 'react';
import { usePulseIntensity, useSnarePulse, useDiscoMode } from '@/stores/discoStore';

interface DiscoPulseWrapperProps {
  /** Base hue for this panel's glow (0-360). Each panel gets a different hue. */
  hue?: number;
  /** Which drum stem drives this panel's pulse. Default: 'kick' */
  stem?: 'kick' | 'snare';
  /** Extra CSS classes to pass through to the content wrapper */
  className?: string;
  /** Inline styles for the content wrapper */
  style?: React.CSSProperties;
  children: React.ReactNode;
}

export const DiscoPulseWrapper: React.FC<DiscoPulseWrapperProps> = ({
  hue = 145,
  stem = 'kick',
  className = '',
  style,
  children,
}) => {
  const discoMode = useDiscoMode();
  const kickPulse = usePulseIntensity();
  const snarePulse = useSnarePulse();
  const wrapperRef = useRef<HTMLDivElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);

  // Modulate glow + scale via direct DOM writes (avoids React re-renders at 60fps)
  useEffect(() => {
    const wrapper = wrapperRef.current;
    const glow = glowRef.current;
    if (!wrapper || !glow) return;

    if (!discoMode) {
      glow.style.opacity = '0';
      wrapper.style.transform = '';
      return;
    }

    const pulse = stem === 'snare' ? snarePulse : kickPulse;

    // Power curve: low-intensity moments are subtle, hits POP
    const intensity = Math.pow(pulse, 0.6);

    // Glow opacity
    glow.style.opacity = String(Math.min(1, intensity * 1.3));

    // Scale pulse: subtle but physical — 1.0 → 1.008 on hard hits
    const scale = 1 + intensity * 0.008;
    wrapper.style.transform = `scale(${scale})`;
  }, [kickPulse, snarePulse, discoMode, stem]);

  // Clean up on disco mode toggle off
  useEffect(() => {
    if (!discoMode) {
      if (glowRef.current) glowRef.current.style.opacity = '0';
      if (wrapperRef.current) wrapperRef.current.style.transform = '';
    }
  }, [discoMode]);

  return (
    <div
      ref={wrapperRef}
      className={`disco-wrapper ${discoMode ? 'disco-active' : ''} ${className}`}
      style={style}
    >
      {/* Spinning gradient border ring */}
      <div
        ref={glowRef}
        className="disco-glow"
        style={{ opacity: 0 }}
      >
        <div
          className="disco-glow-bg"
          style={{
            '--disco-hue': hue,
          } as React.CSSProperties}
        />
      </div>

      {/* Actual panel content — no wrapper div needed, children render directly */}
      {children}
    </div>
  );
};
