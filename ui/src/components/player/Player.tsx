// Player.tsx — Bottom audio player bar
// Ported from hot-step-9000: 3-section layout (info | controls | volume).

import React from 'react';
import {
  Play, Pause, SkipBack, SkipForward, Square,
  Shuffle, Repeat, Repeat1,
  Volume2, VolumeX,
  RotateCcw, Trash2, Download,
  Music, Activity, ListMusic, Scissors, X,
} from 'lucide-react';
import { DiscoIcon } from './DiscoIcon';
import { useTranslation } from 'react-i18next';
import type { Song } from '../../types';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';

interface PlayerProps {
  currentSong: Song | null;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStop: () => void;
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
  onNext: () => void;
  onPrevious: () => void;
  volume: number;
  onVolumeChange: (v: number) => void;
  playbackRate: number;
  onPlaybackRateChange: (r: number) => void;
  pitch441: boolean;
  onTogglePitch441: () => void;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isShuffle: boolean;
  onToggleShuffle: () => void;
  repeatMode: 'none' | 'all' | 'one';
  onToggleRepeat: () => void;
  onReusePrompt?: () => void;
  onDelete?: () => void;
  onDownload?: () => void;
  playMastered: boolean;
  playNoAdapter: boolean;
  onSetPlaybackVariant: (v: 'noadapter' | 'original' | 'mastered') => void;
  onDownloadVariant?: (v: 'noadapter' | 'original' | 'mastered') => void;
  spectrumEnabled: boolean;
  onToggleSpectrum: () => void;
  showPlaylist: boolean;
  playlistCount: number;
  onTogglePlaylist: () => void;
  trimMode: boolean;
  onToggleTrimMode: () => void;
  discoMode: boolean;
  onToggleDisco: () => void;
  abMode: boolean;
  abActiveLabel: 'A' | 'B';
  onToggleAB: () => void;
  onExitABMode: () => void;
}

const formatTime = (s: number) => {
  if (!s || !isFinite(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
};

/** One segment of the playbar's variant switch: a label that selects the
 *  variant, plus an inline download icon that grabs that exact render.
 *  Two sibling buttons rather than a nested one — a button inside a button
 *  is invalid HTML and swallows the inner click in some browsers. */
const VariantSegment: React.FC<{
  active: boolean;
  label: string;
  hint: string;
  downloadHint: string;
  activeClass: string;
  idleClass: string;
  onSelect: () => void;
  onDownload?: () => void;
}> = ({ active, label, hint, downloadHint, activeClass, idleClass, onSelect, onDownload }) => (
  <div className={`flex items-center transition-colors ${active ? activeClass : idleClass}`}>
    <button
      onClick={onSelect}
      className="pl-2 py-1 text-[10px] font-medium"
      title={hint}
    >
      {label}
    </button>
    {onDownload ? (
      <button
        onClick={(e) => { e.stopPropagation(); onDownload(); }}
        className="pl-1 pr-2 py-1 opacity-50 hover:opacity-100 transition-opacity"
        title={downloadHint}
        aria-label={downloadHint}
      >
        <Download size={10} />
      </button>
    ) : (
      <span className="pr-2" />
    )}
  </div>
);

export const Player: React.FC<PlayerProps> = ({
  currentSong,
  isPlaying,
  onTogglePlay,
  onStop,
  currentTime,
  duration,
  onSeek: _onSeek,
  onNext,
  onPrevious,
  volume,
  onVolumeChange,
  playbackRate,
  onPlaybackRateChange,
  pitch441,
  onTogglePitch441,
  isShuffle,
  onToggleShuffle,
  repeatMode,
  onToggleRepeat,
  onReusePrompt,
  onDelete,
  onDownload,
  playMastered,
  playNoAdapter,
  onSetPlaybackVariant,
  onDownloadVariant,
  spectrumEnabled,
  onToggleSpectrum,
  showPlaylist,
  playlistCount,
  onTogglePlaylist,
  trimMode,
  onToggleTrimMode,
  discoMode,
  onToggleDisco,
  abMode,
  abActiveLabel,
  onToggleAB,
  onExitABMode,
}) => {
  const { t } = useTranslation();
  const { isDisguised, disguiseTitle } = useDisguiseMode();


  if (!currentSong) {
    return (
      <div className="h-14 flex-shrink-0 bg-white dark:bg-zinc-950 flex items-center justify-center">
        <span className="text-sm text-zinc-600">{t('player.selectSong')}</span>
      </div>
    );
  }

  return (
    <div className="h-14 flex-shrink-0 bg-white dark:bg-zinc-950 flex items-center px-4 gap-4">
      {/* Left: Song Info */}
      <div className="flex items-center gap-3 w-[240px] flex-shrink-0">
        <div className="w-12 h-12 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center overflow-hidden flex-shrink-0">
          {currentSong.coverUrl ? (
            <img src={currentSong.coverUrl} alt="" className="w-full h-full object-cover" />
          ) : (
            <Music size={20} className="text-zinc-600" />
          )}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-medium text-zinc-900 dark:text-white truncate">{disguiseTitle(currentSong.title || 'Untitled')}</div>
          <div className="text-xs text-zinc-500 truncate">
            {isDisguised ? '' : (currentSong.caption || currentSong.style || '')}
          </div>
        </div>
      </div>

      {/* Center: Transport Controls */}
      <div className="flex-1 flex items-center justify-center gap-3 max-w-[500px] mx-auto">
        <span className="text-[10px] text-zinc-500 font-mono w-10 text-right flex-shrink-0">{formatTime(currentTime)}</span>
        <button
          onClick={onToggleShuffle}
          className={`p-1.5 rounded-lg transition-colors ${isShuffle ? 'text-pink-400' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'}`}
          title={t('player.shuffle')}
        >
          <Shuffle size={16} />
        </button>
        <button onClick={onPrevious} className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors">
          <SkipBack size={18} />
        </button>
        <button
          onClick={onStop}
          className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
          title={t('player.stop')}
        >
          <Square size={14} fill="currentColor" />
        </button>
        <button
          onClick={onTogglePlay}
          className="w-9 h-9 rounded-full bg-zinc-900 dark:bg-white flex items-center justify-center hover:scale-105 transition-transform shadow-md"
        >
          {isPlaying
            ? <Pause size={18} className="text-white dark:text-black" fill="currentColor" />
            : <Play size={18} className="text-white dark:text-black ml-0.5" fill="currentColor" />
          }
        </button>
        <button onClick={onNext} className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors">
          <SkipForward size={18} />
        </button>
        <button
          onClick={onToggleRepeat}
          className={`p-1.5 rounded-lg transition-colors ${repeatMode !== 'none' ? 'text-pink-400' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'}`}
          title={repeatMode === 'one' ? t('player.repeatOne') : repeatMode === 'all' ? t('player.repeatAll') : t('player.repeatOff')}
        >
          {repeatMode === 'one' ? <Repeat1 size={16} /> : <Repeat size={16} />}
        </button>
        <button
          onClick={onToggleSpectrum}
          className={`p-1.5 rounded-lg transition-colors ${spectrumEnabled ? 'text-purple-400' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'}`}
          title={spectrumEnabled ? t('player.spectrumOn') : t('player.spectrumOff')}
        >
          <Activity size={16} />
        </button>
        <button
          onClick={onToggleDisco}
          className={`p-1.5 rounded-lg transition-colors ${discoMode ? 'text-yellow-400 animate-pulse' : 'text-zinc-500 hover:text-yellow-400 hover:bg-yellow-500/5'}`}
          title={discoMode ? 'Disco mode ON' : 'Disco mode OFF'}
        >
          <DiscoIcon size={16} />
        </button>
        <span className="text-[10px] text-zinc-500 font-mono w-10 flex-shrink-0">{formatTime(duration)}</span>
      </div>

      {/* Right: Volume + Actions — min-width keeps the layout stable, but the
          cluster may grow when the text variant switch is present */}
      <div className="flex items-center gap-3 min-w-[280px] flex-shrink-0 justify-end">
        {/* Playback Rate */}
        <button
          onClick={() => {
            const rates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
            const idx = rates.indexOf(playbackRate);
            onPlaybackRateChange(rates[(idx + 1) % rates.length]);
          }}
          className="text-xs text-zinc-500 hover:text-zinc-900 dark:hover:text-white px-1.5 py-0.5 rounded font-mono transition-colors"
          title={t('player.playbackSpeed')}
        >
          {playbackRate}x
        </button>

        {/* 44.1 kHz replay test — the pitch drop a 48 kHz render takes when it
            is clocked out at 44.1 kHz, without the slowdown that comes with it
            on real hardware. Tempo held on purpose: a track that is both lower
            AND slower is much harder to A/B than one that is only lower. */}
        <button
          onClick={onTogglePitch441}
          className={`text-xs px-1.5 py-0.5 rounded font-mono transition-colors ${
            pitch441
              ? 'text-amber-400 bg-amber-500/10'
              : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
          }`}
          title={pitch441
            ? 'Pitched down ~1.47 semitones (44.1 kHz clock), tempo unchanged — click for 48 kHz'
            : 'Hear the 48 kHz render at 44.1 kHz pitch (down ~1.47 semitones, same tempo)'}
        >
          {pitch441 ? '44.1k' : '48k'}
        </button>

        {/* Variant switch — no-adapter reference / unmastered / mastered.
            Segments appear only when the song carries that variant; hidden
            entirely (like the old icon toggle) when neither extra exists. */}
        {!abMode && (currentSong.masteredAudioUrl || currentSong.noAdapterAudioUrl) && (
          <div className="flex rounded-lg overflow-hidden border border-zinc-300 dark:border-white/10 flex-shrink-0">
            {currentSong.noAdapterAudioUrl && (
              <VariantSegment
                active={playNoAdapter}
                label={t('player.variantNoAdapter')}
                hint={t('player.variantNoAdapterHint')}
                downloadHint={t('player.variantDownloadHint', {
                  version: t('player.variantNoAdapter'),
                  defaultValue: `Download the ${t('player.variantNoAdapter')} version`,
                })}
                activeClass="bg-purple-500/15 text-purple-400 shadow-[0_0_8px_rgba(168,85,247,0.15)]"
                idleClass="bg-white dark:bg-zinc-900 text-zinc-500 hover:text-purple-400"
                onSelect={() => onSetPlaybackVariant('noadapter')}
                onDownload={onDownloadVariant && (() => onDownloadVariant('noadapter'))}
              />
            )}
            <VariantSegment
              active={!playMastered && !playNoAdapter}
              label={t('player.variantUnmastered')}
              hint={t('player.variantUnmasteredHint')}
              downloadHint={t('player.variantDownloadHint', {
                version: t('player.variantUnmastered'),
                defaultValue: `Download the ${t('player.variantUnmastered')} version`,
              })}
              activeClass="bg-zinc-200 dark:bg-zinc-700 text-zinc-900 dark:text-white"
              idleClass="bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
              onSelect={() => onSetPlaybackVariant('original')}
              onDownload={onDownloadVariant && (() => onDownloadVariant('original'))}
            />
            {currentSong.masteredAudioUrl && (
              <VariantSegment
                active={playMastered}
                label={t('player.variantMastered')}
                hint={t('player.variantMasteredHint')}
                downloadHint={t('player.variantDownloadHint', {
                  version: t('player.variantMastered'),
                  defaultValue: `Download the ${t('player.variantMastered')} version`,
                })}
                activeClass="bg-amber-500/15 text-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.15)]"
                idleClass="bg-white dark:bg-zinc-900 text-zinc-500 hover:text-amber-400"
                onSelect={() => onSetPlaybackVariant('mastered')}
                onDownload={onDownloadVariant && (() => onDownloadVariant('mastered'))}
              />
            )}
          </div>
        )}

        {/* A/B toggle — only in A/B comparison mode */}
        {abMode && (
          <>
            <button
              onClick={onToggleAB}
              className={`px-2 py-1 rounded-lg text-xs font-bold transition-all ${
                abActiveLabel === 'A'
                  ? 'bg-blue-500/15 text-blue-400 border border-blue-500/25 shadow-[0_0_8px_rgba(59,130,246,0.15)]'
                  : 'bg-orange-500/15 text-orange-400 border border-orange-500/25 shadow-[0_0_8px_rgba(249,115,22,0.15)]'
              }`}
              title={`Playing Track ${abActiveLabel} — click to switch`}
            >
              {abActiveLabel}
            </button>
            <button
              onClick={onExitABMode}
              className="p-1 rounded-lg text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              title="Exit A/B mode"
            >
              <X size={14} />
            </button>
          </>
        )}

        {/* Trim / Crop toggle */}
        <button
          onClick={onToggleTrimMode}
          className={`p-1.5 rounded-lg transition-all ${
            trimMode
              ? 'text-cyan-400 bg-cyan-500/10 shadow-[0_0_8px_rgba(6,182,212,0.15)]'
              : 'text-zinc-500 hover:text-cyan-400 hover:bg-cyan-500/5'
          }`}
          title={trimMode ? t('player.exitTrimMode') : t('player.trimCrop')}
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
          title={showPlaylist ? t('player.hidePlaylist') : t('player.showPlaylist')}
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
            className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
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
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors"
            title={t('player.edit')}
          >
            <RotateCcw size={14} />
          </button>
        )}
        {onDownload && (
          <button
            onClick={onDownload}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
            title={t('player.download')}
          >
            <Download size={14} />
          </button>
        )}
        {onDelete && (
          <button
            onClick={onDelete}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
            title={t('player.delete')}
          >
            <Trash2 size={14} />
          </button>
        )}
      </div>
    </div>
  );
};
