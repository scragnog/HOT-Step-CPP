// SpectrumAnalyzer.tsx — audioMotion-analyzer wrapper for real-time spectrum visualization
//
// Uses the "roundBars + bar-level colorMode" preset from audioMotion demos.
// Connects to the audio source via an HTMLMediaElement from wavesurfer.
//
// IMPORTANT: This component must stay mounted once created — never conditionally
// render it. audioMotion calls createMediaElementSource() which permanently
// redirects the audio element through the Web Audio API graph. Unmounting
// would call destroy(), disconnecting that graph and breaking audio playback.
// Use the `visible` prop to show/hide instead.

import { useEffect, useRef } from 'react';
import AudioMotionAnalyzer from 'audiomotion-analyzer';

interface SpectrumAnalyzerProps {
  /** The HTMLMediaElement to analyze (from wavesurfer's getMediaElement) */
  mediaElement: HTMLMediaElement | null;
  /** Whether the analyzer is visible */
  visible: boolean;
  /** Whether audio is currently playing */
  isPlaying: boolean;
}

export const SpectrumAnalyzer: React.FC<SpectrumAnalyzerProps> = ({
  mediaElement,
  visible,
  isPlaying,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const analyzerRef = useRef<AudioMotionAnalyzer | null>(null);
  const connectedElementRef = useRef<HTMLMediaElement | null>(null);

  // Initialize analyzer — only once per media element, never destroy
  useEffect(() => {
    if (!containerRef.current || !mediaElement) return;

    // Already connected to this element — nothing to do
    if (analyzerRef.current && connectedElementRef.current === mediaElement) return;

    // Media element changed — we need to reconnect.
    // Note: we do NOT destroy the old instance here because that would
    // disconnect the MediaElementSourceNode and break audio.
    // audioMotion can handle reconnecting via connectInput.
    if (analyzerRef.current) {
      try {
        analyzerRef.current.connectInput(mediaElement);
      } catch {
        // If reconnect fails, try creating fresh
        // (this shouldn't normally happen)
      }
      connectedElementRef.current = mediaElement;
      return;
    }

    try {
      const analyzer = new AudioMotionAnalyzer(containerRef.current, {
        source: mediaElement,
        mode: 1,              // 1/48th octave bands — very dense thin bars
        roundBars: true,
        colorMode: 'bar-level',
        gradient: 'prism',
        barSpace: 0.25,       // tight spacing between bars
        reflexRatio: 0.5,     // bottom half mirrors top — bars grow up + down
        reflexAlpha: 1,       // full opacity reflection (not faded)
        reflexBright: 1,      // match brightness
        bgAlpha: 0,           // transparent — dark bg shows through
        overlay: true,
        showPeaks: false,
        smoothing: 0.7,
        showScaleX: false,
        showScaleY: false,
        maxFPS: 60,
      });

      analyzerRef.current = analyzer;
      connectedElementRef.current = mediaElement;
    } catch (err) {
      console.error('[SpectrumAnalyzer] Failed to initialize:', err);
    }

    // Intentionally NO cleanup — we never destroy the audioMotion instance
    // because that would disconnect the MediaElementSourceNode permanently.
  }, [mediaElement]);

  // Toggle animation based on visibility + play state
  useEffect(() => {
    if (!analyzerRef.current) return;
    if (visible && isPlaying) {
      analyzerRef.current.start();
    } else if (!visible) {
      // Stop animation loop when hidden for performance
      analyzerRef.current.stop();
    }
  }, [visible, isPlaying]);

  return (
    <div
      ref={containerRef}
      className="w-full flex-shrink-0 bg-zinc-950"
      style={{
        height: visible ? 75 : 0,
        marginBottom: visible ? 15 : 0,
        overflow: 'hidden',
        transition: 'height 0.2s ease-in-out, margin-bottom 0.2s ease-in-out',
        borderBottom: visible ? '1px solid rgba(168, 85, 247, 0.25)' : 'none',
      }}
    />
  );
};
