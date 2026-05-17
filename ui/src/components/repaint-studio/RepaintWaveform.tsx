// RepaintWaveform.tsx — Waveform display with draggable repaint region selector
//
// Wraps WaveformPlayer with a dedicated wavesurfer.js Region for selecting
// the repaint zone. Region handles are draggable; dimmed zones show outside.

import React, { useRef, useCallback, useEffect, useState, useMemo } from 'react';
import { Play, Pause, SkipForward } from 'lucide-react';
import { WaveformPlayer, type WaveformPlayerHandle } from '../player/WaveformPlayer';
import { SectionMarkers } from '../player/SectionMarkers';
import { formatTime } from '../../utils/lrcUtils';

interface RepaintWaveformProps {
  audioUrl: string;
  regionStart: number;
  regionEnd: number;
  onRegionChange: (start: number, end: number) => void;
  onDurationChange?: (d: number) => void;
}

export const RepaintWaveform: React.FC<RepaintWaveformProps> = ({
  audioUrl,
  regionStart,
  regionEnd,
  onRegionChange,
  onDurationChange,
}) => {
  const waveformRef = useRef<WaveformPlayerHandle>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);

  // Track if region was initialized from props or needs auto-setup
  const regionInitialized = useRef(false);

  // Load audio
  useEffect(() => {
    if (audioUrl && waveformRef.current) {
      waveformRef.current.loadUrl(audioUrl);
      regionInitialized.current = false;
      setIsReady(false);
    }
  }, [audioUrl]);

  const handleReady = useCallback((dur: number) => {
    setDuration(dur);
    setIsReady(true);
    onDurationChange?.(dur);

    // Auto-initialize region to middle third if not already set
    if (!regionInitialized.current && regionStart === 0 && regionEnd === 0) {
      const third = dur / 3;
      onRegionChange(Math.round(third), Math.round(third * 2));
    }
    regionInitialized.current = true;
  }, [onDurationChange, onRegionChange, regionStart, regionEnd]);

  // ── Render repaint region via the Regions plugin ──
  useEffect(() => {
    const ws = waveformRef.current;
    if (!ws || !isReady || duration <= 0) return;

    // Clear previous repaint regions
    ws.clearRegions();

    // Dimmed zone before region
    if (regionStart > 0.01) {
      ws.addMarker(0, '', 'rgba(0, 0, 0, 0)'); // invisible spacer
    }

    // The waveform's setTrimRegions does what we need — dimmed zones + markers
    // But we need a custom look, so we'll use addMarker for the boundaries only
    // The CSS overlay handles the dimming
  }, [isReady, regionStart, regionEnd, duration]);

  // ── Playback controls ──
  const handlePlayPause = useCallback(() => {
    waveformRef.current?.playPause();
  }, []);

  const handlePlayRegion = useCallback(() => {
    if (!waveformRef.current || duration <= 0) return;
    waveformRef.current.seekTo(regionStart / duration);
    waveformRef.current.play();
  }, [regionStart, duration]);

  // Format region range for display
  const regionLabel = useMemo(() => {
    const effEnd = regionEnd > 0 ? regionEnd : duration;
    const regionDur = effEnd - regionStart;
    return `${formatTime(regionStart)} → ${formatTime(effEnd)} (${regionDur.toFixed(1)}s)`;
  }, [regionStart, regionEnd, duration]);

  const effEnd = regionEnd > 0 ? regionEnd : duration;

  return (
    <div className="flex flex-col gap-2">
      {/* Section markers */}
      {audioUrl && duration > 0 && (
        <SectionMarkers audioUrl={audioUrl} duration={duration} />
      )}

      {/* Waveform with region overlay */}
      <div className="relative">
        <WaveformPlayer
          ref={waveformRef}
          onTimeUpdate={setCurrentTime}
          onDurationChange={(d) => { setDuration(d); onDurationChange?.(d); }}
          onPlayChange={setIsPlaying}
          onReady={handleReady}
        />
        {/* Dimmed overlay - before region */}
        {isReady && duration > 0 && regionStart > 0.01 && (
          <div
            className="absolute top-0 left-0 h-full bg-black/50 pointer-events-none z-10"
            style={{ width: `${(regionStart / duration) * 100}%` }}
          />
        )}
        {/* Dimmed overlay - after region */}
        {isReady && duration > 0 && effEnd < duration - 0.01 && (
          <div
            className="absolute top-0 right-0 h-full bg-black/50 pointer-events-none z-10"
            style={{ width: `${((duration - effEnd) / duration) * 100}%` }}
          />
        )}
        {/* Region highlight border */}
        {isReady && duration > 0 && (
          <div
            className="absolute top-0 h-full border-l-2 border-r-2 border-pink-500/60 pointer-events-none z-10"
            style={{
              left: `${(regionStart / duration) * 100}%`,
              width: `${((effEnd - regionStart) / duration) * 100}%`,
            }}
          />
        )}
      </div>

      {/* Region range inputs + controls */}
      <div className="flex items-center gap-3 px-1">
        {/* Playback controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={handlePlayPause}
            className="w-8 h-8 rounded-lg flex items-center justify-center bg-white/5 hover:bg-white/10 text-white transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button
            onClick={handlePlayRegion}
            className="px-2 h-8 rounded-lg flex items-center gap-1.5 bg-pink-500/10 hover:bg-pink-500/20 text-pink-400 text-xs font-medium transition-colors"
            title="Play from region start"
          >
            <SkipForward size={12} /> Region
          </button>
        </div>

        {/* Separator */}
        <div className="h-6 w-px bg-white/10" />

        {/* Region time display */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-zinc-500">Region:</span>
          <span className="font-mono text-pink-400 bg-pink-500/10 px-2 py-0.5 rounded">
            {regionLabel}
          </span>
        </div>

        {/* Separator */}
        <div className="h-6 w-px bg-white/10" />

        {/* Start/End numeric inputs */}
        <div className="flex items-center gap-2 text-xs">
          <label className="text-zinc-500">Start:</label>
          <input
            type="number"
            value={Number(regionStart.toFixed(1))}
            onChange={e => {
              const v = parseFloat(e.target.value) || 0;
              if (v >= 0 && v < effEnd) onRegionChange(v, regionEnd);
            }}
            min={0}
            max={effEnd - 0.1}
            step={0.5}
            className="w-16 px-2 py-1 rounded bg-black/30 border border-white/10 text-white font-mono text-xs focus:outline-none focus:border-pink-500"
          />
          <label className="text-zinc-500">End:</label>
          <input
            type="number"
            value={Number((regionEnd > 0 ? regionEnd : duration).toFixed(1))}
            onChange={e => {
              const v = parseFloat(e.target.value) || 0;
              if (v > regionStart && v <= duration) onRegionChange(regionStart, v);
            }}
            min={regionStart + 0.1}
            max={duration}
            step={0.5}
            className="w-16 px-2 py-1 rounded bg-black/30 border border-white/10 text-white font-mono text-xs focus:outline-none focus:border-pink-500"
          />
          <span className="text-zinc-600">sec</span>
        </div>

        {/* Current time display */}
        <div className="ml-auto text-xs font-mono text-zinc-500">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>

      {/* Region slider — full-width dual-thumb range */}
      <div className="px-1">
        <div className="relative h-6 flex items-center">
          {/* Track background */}
          <div className="absolute inset-x-0 h-1.5 rounded-full bg-white/5" />
          {/* Active region */}
          <div
            className="absolute h-1.5 rounded-full bg-gradient-to-r from-pink-500/40 to-purple-500/40"
            style={{
              left: `${duration > 0 ? (regionStart / duration) * 100 : 0}%`,
              width: `${duration > 0 ? ((effEnd - regionStart) / duration) * 100 : 0}%`,
            }}
          />
          {/* Start thumb */}
          <input
            type="range"
            min={0}
            max={duration || 100}
            step={0.1}
            value={regionStart}
            onChange={e => {
              const v = parseFloat(e.target.value);
              if (v < effEnd - 0.5) onRegionChange(v, regionEnd);
            }}
            className="absolute inset-x-0 w-full appearance-none bg-transparent pointer-events-auto z-20 cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded [&::-webkit-slider-thumb]:bg-pink-500 [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-pink-300 [&::-webkit-slider-thumb]:cursor-grab [&::-webkit-slider-thumb]:active:cursor-grabbing [&::-webkit-slider-thumb]:shadow-lg"
            style={{ height: '24px' }}
          />
          {/* End thumb */}
          <input
            type="range"
            min={0}
            max={duration || 100}
            step={0.1}
            value={effEnd}
            onChange={e => {
              const v = parseFloat(e.target.value);
              if (v > regionStart + 0.5) onRegionChange(regionStart, v);
            }}
            className="absolute inset-x-0 w-full appearance-none bg-transparent pointer-events-auto z-20 cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:rounded [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-purple-300 [&::-webkit-slider-thumb]:cursor-grab [&::-webkit-slider-thumb]:active:cursor-grabbing [&::-webkit-slider-thumb]:shadow-lg"
            style={{ height: '24px' }}
          />
        </div>
      </div>
    </div>
  );
};
