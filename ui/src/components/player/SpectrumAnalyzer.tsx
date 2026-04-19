// SpectrumAnalyzer.tsx — audioMotion-analyzer wrapper for real-time spectrum visualization
//
// Uses the "roundBars + bar-level colorMode" preset from audioMotion demos.
// Connects to the audio source via an HTMLMediaElement from wavesurfer.

import { useEffect, useRef } from 'react';
import AudioMotionAnalyzer from 'audiomotion-analyzer';

interface SpectrumAnalyzerProps {
  /** The HTMLMediaElement to analyze (from wavesurfer's getMediaElement) */
  mediaElement: HTMLMediaElement | null;
  /** Whether audio is currently playing */
  isPlaying: boolean;
}

export const SpectrumAnalyzer: React.FC<SpectrumAnalyzerProps> = ({
  mediaElement,
  isPlaying,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const analyzerRef = useRef<AudioMotionAnalyzer | null>(null);
  const connectedElementRef = useRef<HTMLMediaElement | null>(null);

  // Initialize analyzer
  useEffect(() => {
    if (!containerRef.current) return;

    // Don't create until we have a media element
    if (!mediaElement) return;

    // Already connected to this element
    if (analyzerRef.current && connectedElementRef.current === mediaElement) return;

    // Destroy previous instance if media element changed
    if (analyzerRef.current) {
      try { analyzerRef.current.destroy(); } catch { /* ignore */ }
      analyzerRef.current = null;
      connectedElementRef.current = null;
    }

    try {
      const analyzer = new AudioMotionAnalyzer(containerRef.current, {
        source: mediaElement,
        mode: 6,              // 1/6th octave bands
        roundBars: true,
        colorMode: 'bar-level',
        gradient: 'prism',
        barSpace: 0.4,
        bgAlpha: 0,           // transparent — dark bg shows through
        overlay: true,
        showPeaks: true,
        smoothing: 0.7,
        reflexRatio: 0.3,    // subtle reflection
        reflexAlpha: 0.2,
        reflexBright: 0.8,
        showScaleX: false,
        showScaleY: false,
        // Performance
        maxFPS: 60,
      });

      analyzerRef.current = analyzer;
      connectedElementRef.current = mediaElement;
    } catch (err) {
      console.error('[SpectrumAnalyzer] Failed to initialize:', err);
    }

    return () => {
      if (analyzerRef.current) {
        try { analyzerRef.current.destroy(); } catch { /* ignore */ }
        analyzerRef.current = null;
        connectedElementRef.current = null;
      }
    };
  }, [mediaElement]);

  // Toggle animation based on play state
  useEffect(() => {
    if (!analyzerRef.current) return;
    if (isPlaying) {
      analyzerRef.current.start();
    }
    // Don't stop — let bars decay naturally when paused
  }, [isPlaying]);

  return (
    <div
      ref={containerRef}
      className="w-full flex-shrink-0 bg-zinc-950"
      style={{ height: 80 }}
    />
  );
};
