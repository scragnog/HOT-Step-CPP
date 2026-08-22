// StreamSpectrum.tsx — the spectrum analyser for a track that is still rendering.
//
// A SECOND audioMotion instance, not a reroute of the existing one, and that is
// the whole design.
//
// SpectrumAnalyzer.tsx attaches to an HTMLMediaElement via
// createMediaElementSource, which permanently redirects that element's audio
// through audioMotion's own AudioContext — which is exactly why that file warns
// never to unmount it. A live render has no media element at all: its audio is a
// Web Audio graph at the stream's own 44.1 kHz. Pushing it through a media
// element to satisfy the existing component would mean a MediaStream hop, a
// resample, and a second copy of the audio to keep from double-playing.
//
// So this instance is built on the STREAM's context with `connectSpeakers:
// false`. It observes the same gain node the speakers hear and outputs nothing
// itself, so the audio path is untouched — still gain → destination at 44.1 kHz,
// which is what keeps the window splices sample-exact.
//
// Same visual settings as SpectrumAnalyzer, deliberately: switching to a
// streaming track should not change what the analyser looks like.

import React, { useEffect, useRef } from 'react';
import AudioMotionAnalyzer from 'audiomotion-analyzer';
import { useMm3StreamAudio, mm3StreamGraph } from '../../stores/mm3StreamStore';

interface StreamSpectrumProps {
  visible: boolean;
}

export const StreamSpectrum: React.FC<StreamSpectrumProps> = ({ visible }) => {
  const s = useMm3StreamAudio();
  const containerRef = useRef<HTMLDivElement>(null);
  const analyzerRef = useRef<AudioMotionAnalyzer | null>(null);
  const boundRef = useRef<AudioNode | null>(null);

  // Bind once per stream graph. A new render builds a new AudioContext, so the
  // old instance is destroyed here — unlike the media-element analyser, that is
  // safe: this one drives no audio, it only watches.
  useEffect(() => {
    const graph = mm3StreamGraph();
    if (!containerRef.current || !graph) return;
    if (analyzerRef.current && boundRef.current === graph.node) return;

    if (analyzerRef.current) {
      try { analyzerRef.current.destroy(); } catch { /* already gone */ }
      analyzerRef.current = null;
    }

    try {
      const analyzer = new AudioMotionAnalyzer(containerRef.current, {
        audioCtx: graph.ctx,
        source: graph.node,
        // The stream store already connects gain → destination. Letting
        // audioMotion connect to the speakers too would play everything twice.
        connectSpeakers: false,
        mode: 1,
        roundBars: true,
        colorMode: 'bar-level',
        gradient: 'prism',
        barSpace: 0.25,
        reflexRatio: 0.5,
        reflexAlpha: 1,
        reflexBright: 1,
        bgAlpha: 0,
        overlay: true,
        showPeaks: false,
        smoothing: 0.7,
        showScaleX: false,
        showScaleY: false,
        maxFPS: 60,
      });
      analyzerRef.current = analyzer;
      boundRef.current = graph.node;
      console.log('[StreamSpectrum] bound to the live render graph');
    } catch (err) {
      console.error('[StreamSpectrum] failed to initialise:', err);
    }
    // s.chunks: the graph does not exist until the first window has been
    // decoded, so this has to re-run when one arrives.
  }, [s.chunks, s.jobId]);

  useEffect(() => {
    const a = analyzerRef.current;
    if (!a) return;
    if (visible && s.playing) a.start(); else a.stop();
  }, [visible, s.playing]);

  useEffect(() => () => {
    try { analyzerRef.current?.destroy(); } catch { /* already gone */ }
    analyzerRef.current = null;
    boundRef.current = null;
  }, []);

  return (
    <div
      ref={containerRef}
      className="w-full flex-shrink-0 bg-white dark:bg-zinc-950"
      style={{
        height: visible ? 75 : 0,
        marginBottom: visible ? 15 : 0,
        overflow: 'hidden',
        transition: 'height 0.2s ease-in-out, margin-bottom 0.2s ease-in-out',
        borderBottom: visible ? '1px solid rgba(249, 115, 22, 0.35)' : 'none',
      }}
    />
  );
};
