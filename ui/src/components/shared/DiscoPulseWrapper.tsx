// DiscoPulseWrapper.tsx — iOS 18-style animated rainbow border for disco mode.
//
// Uses a spinning conic-gradient behind the panel content, with a blurred
// white clip layer creating a soft luminous edge glow. Pulse intensity
// from the disco store modulates the glow opacity and scale.
//
// Structure:
//   <wrapper>          ← position: relative, isolation: isolate
//     <glow>           ← overflow: hidden, clips the spinning gradient
//       <glow-bg>      ← oversized spinning conic-gradient
//     </glow>
//     <white-clip>     ← blurred white layer for soft edge glow
//     <content>        ← children with slight inset so glow peeks around edges
//   </wrapper>

import React, { useRef, useEffect } from 'react';
import { usePulseIntensity, useDiscoMode } from '../../stores/discoStore';

interface DiscoPulseWrapperProps {
  /** Base hue for this panel's glow (0-360). Each panel gets a different hue
   *  so they're independently coloured while the gradient rotates through
   *  neighbouring hues via conic-gradient. */
  hue?: number;
  /** Extra CSS classes to pass through to the content wrapper */
  className?: string;
  /** Inline styles for the content wrapper */
  style?: React.CSSProperties;
  children: React.ReactNode;
}

export const DiscoPulseWrapper: React.FC<DiscoPulseWrapperProps> = ({
  hue = 145,
  className = '',
  style,
  children,
}) => {
  const discoMode = useDiscoMode();
  const pulseIntensity = usePulseIntensity();
  const glowRef = useRef<HTMLDivElement>(null);
  const whiteClipRef = useRef<HTMLDivElement>(null);

  // Modulate glow intensity via direct DOM writes (avoids React re-renders at 60fps)
  useEffect(() => {
    const glow = glowRef.current;
    const whiteClip = whiteClipRef.current;
    if (!glow || !whiteClip) return;

    if (!discoMode) {
      glow.style.opacity = '0';
      whiteClip.style.opacity = '0';
      return;
    }

    // Glow opacity: ramp from 0 → 1 with pulse
    // Use a power curve so low-intensity moments are subtle and hits POP
    const intensity = Math.pow(pulseIntensity, 0.7);
    glow.style.opacity = String(Math.min(1, intensity * 1.2));
    whiteClip.style.opacity = String(Math.min(1, intensity * 0.9));

    // Scale the glow slightly on hits for extra punch
    const scale = 1 + intensity * 0.03;
    glow.style.transform = `scale(${scale})`;
  }, [pulseIntensity, discoMode]);

  // Clean up on disco mode toggle
  useEffect(() => {
    if (!discoMode) {
      const glow = glowRef.current;
      const whiteClip = whiteClipRef.current;
      if (glow) { glow.style.opacity = '0'; glow.style.transform = ''; }
      if (whiteClip) whiteClip.style.opacity = '0';
    }
  }, [discoMode]);

  return (
    <div
      className={`disco-wrapper ${discoMode ? 'disco-active' : ''}`}
      style={{ position: 'relative', isolation: 'isolate' } as React.CSSProperties}
    >
      {/* Spinning rainbow gradient — clipped to card bounds */}
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

      {/* Blurred white layer — creates soft luminous edge */}
      <div
        ref={whiteClipRef}
        className="disco-white-clip"
        style={{ opacity: 0 }}
      />

      {/* Actual panel content — slightly inset so glow peeks around edges */}
      <div className={`disco-content ${className}`} style={style}>
        {children}
      </div>
    </div>
  );
};
