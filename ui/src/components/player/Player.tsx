// Player.tsx — Bottom audio player bar
// Ported from hot-step-9000: 3-section layout (info | controls | volume).

import React from 'react';
import {
  Play, Pause, SkipBack, SkipForward,
  Shuffle, Repeat, Repeat1,
  Volume2, VolumeX,
  RotateCcw, Trash2, Download,
  Music, Sparkles, Activity, ListMusic, Scissors,
} from 'lucide-react';
import type { Song } from '../../types';

interface PlayerProps {
  currentSong: Song | null;
  isPlaying: boolean;
  onTogglePlay: () => void;
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
  onNext: () => void;
  onPrevious: () => void;
  volume: number;
  onVolumeChange: (v: number) => void;
  playbackRate: number;
  onPlaybackRateChange: (r: number) => void;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isShuffle: boolean;
  onToggleShuffle: () => void;
  repeatMode: 'none' | 'all' | 'one';
  onToggleRepeat: () => void;
  onReusePrompt?: () => void;
  onDelete?: () => void;
  onDownload?: () => void;
  playMastered: boolean;
  onToggleMastered: () => void;
  spectrumEnabled: boolean;
  onToggleSpectrum: () => void;
  showPlaylist: boolean;
  playlistCount: number;
  onTogglePlaylist: () => void;
  trimMode: boolean;
  onToggleTrimMode: () => void;
}

const formatTime = (s: number) => {
  if (!s || !isFinite(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
};

export const Player: React.FC<PlayerProps> = ({
  currentSong,
  isPlaying,
  onTogglePlay,
  currentTime,
  duration,
  onSeek: _onSeek,
  onNext,
  onPrevious,
  volume,
  onVolumeChange,
  playbackRate,
  onPlaybackRateChange,
  isShuffle,
  onToggleShuffle,
  repeatMode,
  onToggleRepeat,
  onReusePrompt,
  onDelete,
  onDownload,
  playMastered,
  onToggleMastered,
  spectrumEnabled,
  onToggleSpectrum,
  showPlaylist,
  playlistCount,
  onTogglePlaylist,
  trimMode,
  onToggleTrimMode,
}) => {


  if (!currentSong) {
    return (
      <div className="h-14 flex-shrink-0 bg-zinc-950 flex items-center justify-center">
        <span className="text-sm text-zinc-600">Select a song to play</span>
      </div>
    );
  }

  return (
    <div className="h-14 flex-shrink-0 bg-zinc-950 flex items-center px-4 gap-4">
      {/* Left: Song Info */}
      <div className="flex items-center gap-3 w-[240px] flex-shrink-0">
        <div className="w-12 h-12 rounded-lg bg-zinc-800 flex items-center justify-center overflow-hidden flex-shrink-0">
          {currentSong.coverUrl ? (
            <img src={currentSong.coverUrl} alt="" className="w-full h-full object-cover" />
          ) : (
            <Music size={20} className="text-zinc-600" />
          )}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-medium text-white truncate">{currentSong.title || 'Untitled'}</div>
          <div className="text-xs text-zinc-500 truncate">
            {currentSong.caption || currentSong.style || ''}
          </div>
        </div>
      </div>

      {/* Center: Transport Controls */}
      <div className="flex-1 flex items-center justify-center gap-3 max-w-[500px] mx-auto">
        <span className="text-[10px] text-zinc-500 font-mono w-10 text-right flex-shrink-0">{formatTime(currentTime)}</span>
        <button
          onClick={onToggleShuffle}
          className={`p-1.5 rounded-lg transition-colors ${isShuffle ? 'text-pink-400' : 'text-zinc-500 hover:text-white'}`}
          title="Shuffle"
        >
          <Shuffle size={16} />
        </button>
        <button onClick={onPrevious} className="p-1.5 rounded-lg text-zinc-400 hover:text-white transition-colors">
          <SkipBack size={18} />
        </button>
        <button
          onClick={onTogglePlay}
          className="w-9 h-9 rounded-full bg-white flex items-center justify-center hover:scale-105 transition-transform"
        >
          {isPlaying
            ? <Pause size={18} className="text-black" fill="black" />
            : <Play size={18} className="text-black ml-0.5" fill="black" />
          }
        </button>
        <button onClick={onNext} className="p-1.5 rounded-lg text-zinc-400 hover:text-white transition-colors">
          <SkipForward size={18} />
        </button>
        <button
          onClick={onToggleRepeat}
          className={`p-1.5 rounded-lg transition-colors ${repeatMode !== 'none' ? 'text-pink-400' : 'text-zinc-500 hover:text-white'}`}
          title={repeatMode === 'one' ? 'Repeat One' : repeatMode === 'all' ? 'Repeat All' : 'Repeat Off'}
        >
          {repeatMode === 'one' ? <Repeat1 size={16} /> : <Repeat size={16} />}
        </button>
        <button
          onClick={onToggleSpectrum}
          className={`p-1.5 rounded-lg transition-colors ${spectrumEnabled ? 'text-purple-400' : 'text-zinc-500 hover:text-white'}`}
          title={spectrumEnabled ? 'Spectrum Analyzer On' : 'Spectrum Analyzer Off'}
        >
          <Activity size={16} />
        </button>
        <span className="text-[10px] text-zinc-500 font-mono w-10 flex-shrink-0">{formatTime(duration)}</span>
      </div>

      {/* Right: Volume + Actions */}
      <div className="flex items-center gap-3 w-[280px] flex-shrink-0 justify-end">
        {/* Playback Rate */}
        <button
          onClick={() => {
            const rates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
            const idx = rates.indexOf(playbackRate);
            onPlaybackRateChange(rates[(idx + 1) % rates.length]);
          }}
          className="text-xs text-zinc-500 hover:text-white px-1.5 py-0.5 rounded font-mono transition-colors"
          title="Playback Speed"
        >
          {playbackRate}x
        </button>

        {/* Mastered toggle — only when song has mastered version */}
        {currentSong.masteredAudioUrl && (
          <button
            onClick={onToggleMastered}
            className={`p-1.5 rounded-lg transition-all ${
              playMastered
                ? 'text-amber-400 bg-amber-500/10 shadow-[0_0_8px_rgba(245,158,11,0.15)]'
                : 'text-zinc-500 hover:text-amber-400 hover:bg-amber-500/5'
            }`}
            title={playMastered ? 'Playing mastered • Click for original' : 'Playing original • Click for mastered'}
          >
            <Sparkles size={15} />
          </button>
        )}

        {/* Trim / Crop toggle */}
        <button
          onClick={onToggleTrimMode}
          className={`p-1.5 rounded-lg transition-all ${
            trimMode
              ? 'text-cyan-400 bg-cyan-500/10 shadow-[0_0_8px_rgba(6,182,212,0.15)]'
              : 'text-zinc-500 hover:text-cyan-400 hover:bg-cyan-500/5'
          }`}
          title={trimMode ? 'Exit Trim Mode' : 'Trim / Crop Audio'}
        >
          <Scissors size={15} />
        </button>

        {/* Playlist toggle */}
        <button
          onClick={onTogglePlaylist}
          className={`p-1.5 rounded-lg transition-all relative ${
            showPlaylist
              ? 'text-pink-400 bg-pink-500/10'
              : 'text-zinc-500 hover:text-pink-400 hover:bg-pink-500/5'
          }`}
          title={showPlaylist ? 'Hide Playlist' : 'Show Playlist'}
        >
          <ListMusic size={15} />
          {playlistCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 min-w-[14px] h-[14px] px-0.5 rounded-full bg-pink-500 text-[8px] font-bold text-white flex items-center justify-center">
              {playlistCount}
            </span>
          )}
        </button>

        {/* Volume */}
        <div className="flex items-center gap-1.5 group flex-shrink-0">
          <button
            onClick={() => onVolumeChange(volume > 0 ? 0 : 0.8)}
            className="text-zinc-400 hover:text-white transition-colors"
          >
            {volume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={volume}
            onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
            className="w-20 min-w-[80px]"
          />
        </div>

        {/* Quick Actions */}
        {onReusePrompt && (
          <button
            onClick={onReusePrompt}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-white hover:bg-white/5 transition-colors"
            title="Reuse Prompt"
          >
            <RotateCcw size={14} />
          </button>
        )}
        {onDownload && (
          <button
            onClick={onDownload}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
            title="Download"
          >
            <Download size={14} />
          </button>
        )}
        {onDelete && (
          <button
            onClick={onDelete}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
            title="Delete"
          >
            <Trash2 size={14} />
          </button>
        )}
      </div>
    </div>
  );
};
