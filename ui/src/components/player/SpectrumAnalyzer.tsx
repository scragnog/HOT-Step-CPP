// SpectrumAnalyzer.tsx — audioMotion-analyzer wrapper for real-time spectrum visualization
//
// Uses the "roundBars + bar-level colorMode" preset from audioMotion demos.
// Connects to the audio source via an HTMLMediaElement from wavesurfer.
//
// IMPORTANT: This component must stay mounted once created — never conditionally
// render it. destroy() would tear down its end of the audio graph. Use the
// `visible` prop to show/hide instead.
//
// It does NOT capture the media element. pitchShift.ts owns the decks' route to
// the speakers (an element can only be captured once, and the pitch shifter has
// to sit in that route), so this reads the tap node it exposes. That also means
// the analyser shows what you are actually hearing, pitch shift included, and
// that it never needs reconnecting when the deck changes — the tap does not.

import { useEffect, useRef } from 'react';
import AudioMotionAnalyzer from 'audiomotion-analyzer';
import { registerAudioMotion } from '../../stores/discoStore';
import { pitchAttachElement, pitchTapNode } from '../../audio/pitchShift';

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
    if (!containerRef.current || !mediaElement) {
      console.log('[SpectrumAnalyzer] Skip init: container=', !!containerRef.current, 'media=', !!mediaElement);
      return;
    }

    // Make sure this deck is in the shared chain, then read the one node that
    // carries every deck. A later deck changes nothing here.
    pitchAttachElement(mediaElement);
    const source = pitchTapNode();
    if (analyzerRef.current) {
      connectedElementRef.current = mediaElement;
      return;
    }
    if (!source) {
      console.error('[SpectrumAnalyzer] no audio graph to analyse');
      return;
    }

    try {
      const analyzer = new AudioMotionAnalyzer(containerRef.current, {
        source,
        // The tap already feeds the speakers; a second path would double it.
        connectSpeakers: false,
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
      registerAudioMotion(analyzer);
      console.log('[SpectrumAnalyzer] Created + registered audioMotion');
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
    } else {
      // Stop animation loop when hidden OR when visible but paused
      analyzerRef.current.stop();
    }
  }, [visible, isPlaying]);

  return (
    <div
      ref={containerRef}
      className="w-full flex-shrink-0 bg-white dark:bg-zinc-950"
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
