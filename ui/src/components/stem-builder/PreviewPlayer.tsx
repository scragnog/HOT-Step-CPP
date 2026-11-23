// PreviewPlayer.tsx — Dual-track preview player for Stem Builder
//
// Plays source audio and generated stem simultaneously, with independent
// volume controls. Replaces the broken `new Audio().play()` pattern with
// a visible, controllable player in the UI.
//
// Features:
//   - Synced playback of source + stem (play/pause/seek together)
//   - Independent volume sliders per track
//   - Solo/mute toggles
//   - Click-to-seek progress bar
//   - Auto-play on new stem URL

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, Square, Volume2, VolumeX, X } from 'lucide-react';
import { formatClock } from '@/utils/format/format';

interface PreviewPlayerProps {
  sourceUrl: string;
  stemUrl: string;
  stemLabel: string;      // e.g. "drums", "guitar"
  onClose: () => void;
}

export const PreviewPlayer: React.FC<PreviewPlayerProps> = ({
  sourceUrl,
  stemUrl,
  stemLabel,
  onClose,
}) => {
  const sourceRef = useRef<HTMLAudioElement>(null);
  const stemRef = useRef<HTMLAudioElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const [sourceVol, setSourceVol] = useState(0.7);
  const [stemVol, setStemVol] = useState(1.0);
  const [sourceMuted, setSourceMuted] = useState(false);
  const [stemMuted, setStemMuted] = useState(false);

  // Track which sources are loaded
  const [sourceLoaded, setSourceLoaded] = useState(false);
  const [stemLoaded, setStemLoaded] = useState(false);

  // ── Apply volumes ──
  useEffect(() => {
    if (sourceRef.current) {
      sourceRef.current.volume = sourceMuted ? 0 : sourceVol;
    }
  }, [sourceVol, sourceMuted]);

  useEffect(() => {
    if (stemRef.current) {
      stemRef.current.volume = stemMuted ? 0 : stemVol;
    }
  }, [stemVol, stemMuted]);

  // ── Time update ──
  useEffect(() => {
    const stem = stemRef.current;
    if (!stem) return;
    const onTime = () => setCurrentTime(stem.currentTime);
    const onDur = () => setDuration(stem.duration || 0);
    const onEnd = () => { setIsPlaying(false); setCurrentTime(0); };
    stem.addEventListener('timeupdate', onTime);
    stem.addEventListener('loadedmetadata', onDur);
    stem.addEventListener('ended', onEnd);
    return () => {
      stem.removeEventListener('timeupdate', onTime);
      stem.removeEventListener('loadedmetadata', onDur);
      stem.removeEventListener('ended', onEnd);
    };
  }, [stemUrl]);

  // ── Auto-play when both tracks are loaded ──
  useEffect(() => {
    if (sourceLoaded && stemLoaded && sourceRef.current && stemRef.current) {
      // Small delay to ensure both elements are ready
      const t = setTimeout(() => {
        sourceRef.current?.play().catch(() => {});
        stemRef.current?.play().catch(() => {});
        setIsPlaying(true);
      }, 100);
      return () => clearTimeout(t);
    }
  }, [sourceLoaded, stemLoaded]);

  // ── Reset loaded state when URLs change ──
  useEffect(() => { setSourceLoaded(false); }, [sourceUrl]);
  useEffect(() => { setStemLoaded(false); setCurrentTime(0); setIsPlaying(false); }, [stemUrl]);

  // ── Synced play/pause ──
  const togglePlay = useCallback(() => {
    const src = sourceRef.current;
    const stm = stemRef.current;
    if (!src || !stm) return;

    if (isPlaying) {
      src.pause();
      stm.pause();
      setIsPlaying(false);
    } else {
      // Sync positions before playing
      src.currentTime = stm.currentTime;
      src.play().catch(() => {});
      stm.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [isPlaying]);

  const handleStop = useCallback(() => {
    const src = sourceRef.current;
    const stm = stemRef.current;
    if (src) { src.pause(); src.currentTime = 0; }
    if (stm) { stm.pause(); stm.currentTime = 0; }
    setIsPlaying(false);
    setCurrentTime(0);
  }, []);

  // ── Seek ──
  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const t = pct * duration;

    if (sourceRef.current) sourceRef.current.currentTime = t;
    if (stemRef.current) stemRef.current.currentTime = t;
    setCurrentTime(t);
  }, [duration]);

  const formatTime = (s: number) => formatClock(s);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="rounded-xl bg-zinc-900/80 border border-amber-500/15 p-3 flex flex-col gap-2">
      {/* Hidden audio elements */}
      <audio
        ref={sourceRef}
        src={sourceUrl}
        preload="auto"
        onCanPlayThrough={() => setSourceLoaded(true)}
      />
      <audio
        ref={stemRef}
        src={stemUrl}
        preload="auto"
        onCanPlayThrough={() => setStemLoaded(true)}
      />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="text-[11px] font-semibold text-amber-400 uppercase tracking-wider">
            Preview
          </div>
          <div className="text-[10px] text-zinc-500">
            Source + {stemLabel}
          </div>
        </div>
        <button
          type="button"
          onClick={() => { handleStop(); onClose(); }}
          className="w-5 h-5 rounded flex items-center justify-center hover:bg-white/10 transition-colors"
          title="Close preview"
        >
          <X size={12} className="text-zinc-500" />
        </button>
      </div>

      {/* Transport + Progress */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={togglePlay}
          className="w-8 h-8 rounded-lg bg-amber-500/15 hover:bg-amber-500/25 flex items-center justify-center transition-colors"
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying
            ? <Pause size={14} className="text-amber-400" />
            : <Play size={14} className="text-amber-400 ml-0.5" />
          }
        </button>
        <button
          type="button"
          onClick={handleStop}
          className="w-8 h-8 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] flex items-center justify-center transition-colors"
          title="Stop"
        >
          <Square size={12} className="text-zinc-500" />
        </button>

        {/* Progress bar */}
        <div
          className="flex-1 h-2 rounded-full bg-white/[0.06] cursor-pointer group relative"
          onClick={handleSeek}
        >
          <div
            className="h-full rounded-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-100 relative"
            style={{ width: `${progress}%` }}
          >
            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-amber-400 shadow-md opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        </div>

        <span className="text-[10px] text-zinc-500 font-mono w-16 text-right">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>

      {/* Volume controls */}
      <div className="flex items-center gap-3">
        {/* Source volume */}
        <div className="flex items-center gap-1.5 flex-1">
          <button
            type="button"
            onClick={() => setSourceMuted(m => !m)}
            className="flex-shrink-0 w-5 h-5 rounded flex items-center justify-center hover:bg-white/10 transition-colors"
            title={sourceMuted ? 'Unmute source' : 'Mute source'}
          >
            {sourceMuted
              ? <VolumeX size={12} className="text-zinc-600" />
              : <Volume2 size={12} className="text-zinc-500" />
            }
          </button>
          <span className="text-[10px] text-zinc-600 w-10 flex-shrink-0">Source</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={sourceVol}
            onChange={e => setSourceVol(parseFloat(e.target.value))}
            className="flex-1 h-1 accent-zinc-500 cursor-pointer"
          />
        </div>

        {/* Stem volume */}
        <div className="flex items-center gap-1.5 flex-1">
          <button
            type="button"
            onClick={() => setStemMuted(m => !m)}
            className="flex-shrink-0 w-5 h-5 rounded flex items-center justify-center hover:bg-white/10 transition-colors"
            title={stemMuted ? 'Unmute stem' : 'Mute stem'}
          >
            {stemMuted
              ? <VolumeX size={12} className="text-amber-800" />
              : <Volume2 size={12} className="text-amber-400" />
            }
          </button>
          <span className="text-[10px] text-amber-400/70 w-10 flex-shrink-0 capitalize truncate">
            {stemLabel}
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={stemVol}
            onChange={e => setStemVol(parseFloat(e.target.value))}
            className="flex-1 h-1 accent-amber-500 cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
};
