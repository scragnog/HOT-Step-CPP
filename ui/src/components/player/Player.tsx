// Player.tsx — Bottom audio player bar
// Ported from hot-step-9000: 3-section layout (info | controls | volume).

import React from 'react';
import {
  Play, Pause, SkipBack, SkipForward, Square,
  Shuffle, Repeat, Repeat1,
  Volume2, VolumeX,
  RotateCcw, Trash2, Download,
  Activity, Scissors, X,
} from 'lucide-react';
import { DiscoIcon } from './DiscoIcon';
import { useTranslation } from 'react-i18next';
import { isPitchShiftReady } from '../../audio/pitchShift';
import type { Song } from '../../types';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';
import { CoverImage } from '../shared/CoverImage';
import { HintButton } from '../shared/HintButton';
import { ParamLabel } from '../shared/ParamLabel';
import { useIsMobile } from '../shared/BottomSheet';
import { FullScreenPlayer } from './FullScreenPlayer';

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
    <HintButton
      label={label}
      info={hint}
      className="pl-2 py-1 text-[10px] font-medium"
      onClick={onSelect}
      ariaLabel={label}
    >
      {label}
    </HintButton>
    {onDownload ? (
      <HintButton
        label={label}
        info={downloadHint}
        onClick={(e) => { e.stopPropagation(); onDownload(); }}
        className="pl-1 pr-2 py-1 opacity-50 hover:opacity-100 transition-opacity"
        ariaLabel={downloadHint}
      >
        <Download size={10} />
      </HintButton>
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
  onSeek,
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
  const isMobile = useIsMobile();
  const [expanded, setExpanded] = React.useState(false);
  React.useEffect(() => { if (!isMobile) setExpanded(false); }, [isMobile]);

  if (!currentSong) {
    return (
      <div className="h-14 flex-shrink-0 bg-white dark:bg-zinc-950 flex items-center justify-center">
        <span className="text-sm text-zinc-600">{t('player.selectSong')}</span>
      </div>
    );
  }

  return (
    <div className={[
      'flex-shrink-0 bg-white dark:bg-zinc-950 px-4',
      // Desktop: single row, fixed height
      'min-[1024px]:flex min-[1024px]:items-center min-[1024px]:h-14 min-[1024px]:gap-4',
      // Mobile: two rows stacked, auto height
      'max-[1023px]:flex max-[1023px]:flex-col max-[1023px]:py-1.5 max-[1023px]:gap-1',
    ].join(' ')}>
      {isMobile && expanded && (
        <FullScreenPlayer
          song={currentSong}
          isPlaying={isPlaying}
          currentTime={currentTime}
          duration={duration}
          onTogglePlay={onTogglePlay}
          onNext={onNext}
          onPrevious={onPrevious}
          onSeek={onSeek}
          isShuffle={isShuffle}
          onToggleShuffle={onToggleShuffle}
          repeatMode={repeatMode}
          onToggleRepeat={onToggleRepeat}
          onClose={() => setExpanded(false)}
        />
      )}
      {/* Left: Song Info, row 1 on mobile (tap opens the full-screen player on phones) */}
      <div
        className={`flex items-center gap-3 min-[1024px]:w-[240px] min-[1024px]:flex-shrink-0 max-[1023px]:w-full max-[1023px]:min-w-0${isMobile ? ' cursor-pointer' : ''}`}
        onClick={isMobile ? () => setExpanded(true) : undefined}
        role={isMobile ? 'button' : undefined}
        aria-label={isMobile ? t('player.fullScreen.open') : undefined}
      >
        <div className="w-12 h-12 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center overflow-hidden flex-shrink-0">
          <CoverImage url={currentSong.coverUrl} seed={currentSong.id} iconSize={20} />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-medium text-zinc-900 dark:text-white truncate">{disguiseTitle(currentSong.title || 'Untitled')}</div>
          <div className="text-xs text-zinc-500 truncate">
            {isDisguised ? '' : (currentSong.caption || currentSong.style || '')}
          </div>
        </div>
      </div>

      {/* Row 2 on mobile: transport + right controls together */}
      <div className="max-[1023px]:flex max-[1023px]:items-center max-[1023px]:gap-2 max-[1023px]:w-full min-[1024px]:contents">

      {/* Center: Transport Controls */}
      <div className="flex-1 flex items-center justify-center gap-3 min-[1024px]:max-w-[500px] min-[1024px]:mx-auto max-[1023px]:flex-1 max-[1023px]:min-w-0 max-[1023px]:gap-2">
        <span className="text-[10px] text-zinc-500 font-mono w-10 text-right flex-shrink-0">{formatTime(currentTime)}</span>
        <HintButton
          label={t('player.shuffle')}
          info={t('player.shuffleInfo', 'Shuffles the play queue. Off: plays it in order. The queue order itself is not changed, only the order it is walked in.')}
          onClick={onToggleShuffle}
          className={`p-1.5 rounded-lg transition-colors ${isShuffle ? 'text-pink-400' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'}`}
          ariaLabel={t('player.shuffle')}
        >
          <Shuffle size={16} />
        </HintButton>
        <HintButton
          label={t('player.previous', 'Previous track')}
          info={t('player.previousInfo', 'Plays the track before this one in the queue. At the start of the queue it goes to the last track.')}
          onClick={onPrevious}
          className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
          ariaLabel={t('player.previous', 'Previous track')}
        >
          <SkipBack size={18} />
        </HintButton>
        <HintButton
          label={t('player.stop')}
          info={t('player.stopInfo', 'Stops playback and returns the playhead to the start of the track. Pause keeps the position, this does not.')}
          onClick={onStop}
          className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
          ariaLabel={t('player.stop')}
        >
          <Square size={14} fill="currentColor" />
        </HintButton>
        <HintButton
          label={isPlaying ? t('player.pause', 'Pause') : t('player.play', 'Play')}
          info={isPlaying
            ? t('player.pauseInfo', 'Pauses where it is, so the playhead stays put. Space does the same anywhere that is not a text field.')
            : t('player.playInfo', 'Plays from the playhead. Space does the same anywhere that is not a text field.')}
          onClick={onTogglePlay}
          className="w-9 h-9 rounded-full bg-zinc-900 dark:bg-white flex items-center justify-center hover:scale-105 transition-transform shadow-md"
          ariaLabel={isPlaying ? t('player.pause', 'Pause') : t('player.play', 'Play')}
        >
          {isPlaying
            ? <Pause size={18} className="text-white dark:text-black" fill="currentColor" />
            : <Play size={18} className="text-white dark:text-black ml-0.5" fill="currentColor" />
          }
        </HintButton>
        <HintButton
          label={t('player.next', 'Next track')}
          info={t('player.nextInfo', 'Plays the next track in the queue. At the end of the queue it wraps to the first track.')}
          onClick={onNext}
          className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
          ariaLabel={t('player.next', 'Next track')}
        >
          <SkipForward size={18} />
        </HintButton>
        <HintButton
          label={repeatMode === 'one' ? t('player.repeatOne') : repeatMode === 'all' ? t('player.repeatAll') : t('player.repeatOff')}
          info={t('player.repeatInfo', 'Cycles repeat off, all and one. All loops the queue; one loops the current track; off stops at the end of the queue.')}
          onClick={onToggleRepeat}
          className={`p-1.5 rounded-lg transition-colors ${repeatMode !== 'none' ? 'text-pink-400' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'}`}
          ariaLabel={t('player.repeat', 'Repeat')}
        >
          {repeatMode === 'one' ? <Repeat1 size={16} /> : <Repeat size={16} />}
        </HintButton>
        <HintButton
          label={spectrumEnabled ? t('player.spectrumOn') : t('player.spectrumOff')}
          info={t('player.spectrumInfo', 'Draws a live frequency spectrum beside the player. It reads the same audio output, so it costs a little CPU while playing and nothing while paused.')}
          onClick={onToggleSpectrum}
          className={`p-1.5 rounded-lg transition-colors max-[1023px]:hidden ${spectrumEnabled ? 'text-purple-400' : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'}`}
          ariaLabel={t('player.spectrum', 'Spectrum')}
        >
          <Activity size={16} />
        </HintButton>
        <HintButton
          label={discoMode ? t('player.discoOn', 'Disco mode on') : t('player.discoOff', 'Disco mode off')}
          info={t('player.discoInfo', 'Tints the player with colours that follow the music. Presentation only: it changes nothing about the render or the audio.')}
          onClick={onToggleDisco}
          className={`p-1.5 rounded-lg transition-colors max-[1023px]:hidden ${discoMode ? 'text-yellow-400 animate-pulse' : 'text-zinc-500 hover:text-yellow-400 hover:bg-yellow-500/5'}`}
          ariaLabel={t('player.disco', 'Disco mode')}
        >
          <DiscoIcon size={16} />
        </HintButton>
        <span className="text-[10px] text-zinc-500 font-mono w-10 flex-shrink-0">{formatTime(duration)}</span>
      </div>

      {/* Right: Volume + Actions — min-width keeps the layout stable, but the
          cluster may grow when the text variant switch is present */}
      <div className="flex items-center gap-3 min-[1024px]:min-w-[280px] min-[1024px]:flex-shrink-0 max-[1023px]:gap-2 justify-end">
        {/* Playback Rate — hidden on mobile */}
        <HintButton
          label={t('player.playbackSpeed')}
          info={t('player.playbackSpeedInfo', 'Cycles 0.5x, 0.75x, 1x, 1.25x, 1.5x and 2x. Pitch is corrected, so a slower speed is not also a lower key. The lyric editor shares this speed.')}
          onClick={() => {
            const rates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
            const idx = rates.indexOf(playbackRate);
            onPlaybackRateChange(rates[(idx + 1) % rates.length]);
          }}
          className="text-xs text-zinc-500 hover:text-zinc-900 dark:hover:text-white px-1.5 py-0.5 rounded font-mono transition-colors max-[1023px]:hidden"
          ariaLabel={t('player.playbackSpeed')}
        >
          {playbackRate}x
        </HintButton>

        {/* 44.1 kHz replay test — the pitch drop a 48 kHz render takes when it
            is clocked out at 44.1 kHz, without the slowdown that comes with it
            on real hardware. Tempo held on purpose: a track that is both lower
            AND slower is much harder to A/B than one that is only lower. */}
        <HintButton
          label={t('player.pitchTest', '44.1 kHz replay test')}
          info={!isPitchShiftReady()
            ? t('player.pitchUnavailable', 'The pitch shifter is not available in this browser, so the replay test cannot run. The console says why.')
            : pitch441
              ? t('player.pitchOnInfo', 'Pitched down about 1.47 semitones, as a 48 kHz render sounds clocked out at 44.1 kHz, with the tempo held. Click for 48 kHz.')
              : t('player.pitchOffInfo', 'Hear the 48 kHz render at 44.1 kHz pitch, about 1.47 semitones lower, at the same tempo. A way to hear that pitch drop without a slower track.')}
          onClick={onTogglePitch441}
          className={`text-xs px-1.5 py-0.5 rounded font-mono transition-colors max-[1023px]:hidden ${
            pitch441
              ? 'text-amber-400 bg-amber-500/10'
              : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
          }`}
          ariaLabel={t('player.pitchTest', '44.1 kHz replay test')}
        >
          {pitch441 ? '44.1k' : '48k'}
        </HintButton>

        {/* Variant switch — hidden on mobile (space-constrained), visible on desktop */}
        {!abMode && (currentSong.masteredAudioUrl || currentSong.noAdapterAudioUrl) && (
          <div className="flex rounded-lg overflow-hidden border border-zinc-300 dark:border-white/10 flex-shrink-0 max-[1023px]:hidden">
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
            <HintButton
              label={t('player.abSwitch', 'Switch A/B')}
              info={t('player.abSwitchInfo', 'Switches between the two comparison tracks without leaving the player. The label shows which one is playing.')}
              onClick={onToggleAB}
              className={`px-2 py-1 rounded-lg text-xs font-bold transition-all ${
                abActiveLabel === 'A'
                  ? 'bg-blue-500/15 text-blue-400 border border-blue-500/25 shadow-[0_0_8px_rgba(59,130,246,0.15)]'
                  : 'bg-orange-500/15 text-orange-400 border border-orange-500/25 shadow-[0_0_8px_rgba(249,115,22,0.15)]'
              }`}
              ariaLabel={t('player.abSwitch', 'Switch A/B')}
            >
              {abActiveLabel}
            </HintButton>
            <HintButton
              label={t('player.exitAB', 'Exit A/B')}
              info={t('player.exitABInfo', 'Leaves A/B comparison and returns the player to the normal single-track controls.')}
              onClick={onExitABMode}
              className="p-1 rounded-lg text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              ariaLabel={t('player.exitAB', 'Exit A/B')}
            >
              <X size={14} />
            </HintButton>
          </>
        )}

        {/* Trim / Crop toggle — hidden on mobile */}
        <HintButton
          label={trimMode ? t('player.exitTrimMode') : t('player.trimCrop')}
          info={t('player.trimInfo', 'Shows the trim controls above the waveform: click an in point and an out point, then crop. Cropping rewrites the audio file and cuts its lyric sidecars to the new timeline, so it cannot be undone.')}
          onClick={onToggleTrimMode}
          className={`p-1.5 rounded-lg transition-all max-[1023px]:hidden ${
            trimMode
              ? 'text-cyan-400 bg-cyan-500/10 shadow-[0_0_8px_rgba(6,182,212,0.15)]'
              : 'text-zinc-500 hover:text-cyan-400 hover:bg-cyan-500/5'
          }`}
          ariaLabel={trimMode ? t('player.exitTrimMode') : t('player.trimCrop')}
        >
          <Scissors size={15} />
        </HintButton>


        {/* Volume */}
        <div className="flex items-center gap-1.5 group flex-shrink-0">
          <HintButton
            label={volume === 0 ? t('player.unmute', 'Unmute') : t('player.mute', 'Mute')}
            info={t('player.muteInfo', 'Mutes the player, or restores the volume to 80 percent. The lyric editor shares this volume.')}
            onClick={() => onVolumeChange(volume > 0 ? 0 : 0.8)}
            className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
            ariaLabel={volume === 0 ? t('player.unmute', 'Unmute') : t('player.mute', 'Mute')}
          >
            {volume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </HintButton>
          <ParamLabel
            label={t('player.volume', 'Volume')}
            info={t('player.volumeInfo', 'Player volume, shared with the lyric editor because they use the same audio element. Quiet is often better than muted when placing a syllable by ear.')}
          >
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={volume}
              onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
              className="min-[1024px]:w-20 min-[1024px]:min-w-[80px] max-[1023px]:w-14"
              aria-label={t('player.volume', 'Volume')}
            />
          </ParamLabel>
        </div>

        {/* Quick Actions, hidden on mobile */}
        {onReusePrompt && (
          <HintButton
            label={t('player.edit')}
            info={t('player.editInfo', 'Loads this render\u2019s parameters back into Custom-Gen so the same prompt and settings can be generated again, with changes.')}
            onClick={onReusePrompt}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors max-[1023px]:hidden"
            ariaLabel={t('player.edit')}
          >
            <RotateCcw size={14} />
          </HintButton>
        )}
        {onDownload && (
          <HintButton
            label={t('player.download')}
            info={t('player.downloadInfo', 'Downloads this track in the format and version set on the Settings page. When a download folder is configured the file is saved there instead, with the matching .lrc beside it.')}
            onClick={onDownload}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors max-[1023px]:hidden"
            ariaLabel={t('player.download')}
          >
            <Download size={14} />
          </HintButton>
        )}
        {onDelete && (
          <HintButton
            label={t('player.delete')}
            info={t('player.deleteInfo', 'Deletes the track, its audio file and its lyric sidecars. This cannot be undone.')}
            onClick={onDelete}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors max-[1023px]:hidden"
            ariaLabel={t('player.delete')}
          >
            <Trash2 size={14} />
          </HintButton>
        )}
      </div>
      {/* /row-2 wrapper */}
      </div>
    </div>
  );
};
