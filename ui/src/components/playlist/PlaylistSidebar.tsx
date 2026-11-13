/**
 * PlaylistSidebar.tsx — Global playlist panel, rendered as a right-side sidebar.
 *
 * Replaces the floating playlist. Lives in App.tsx so it's accessible from any view.
 * Track list rendering adapted from the original FloatingPlaylist.
 */

import React, { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Play, X, Trash2, ChevronUp, ChevronDown,
  Music, ListPlus, ListMusic, Square, DownloadCloud, Loader2,
} from 'lucide-react';
import { usePlaylist, type PlaylistItem } from '../lyric-studio/playlistStore';
import { playFromList, togglePlay, playlistItemToTrack, usePlaybackSelector } from '../../stores/playbackStore';
import { downloadTrack, downloadAll } from '../../utils/downloadTrack';
import { useAuth } from '../../context/AuthContext';
import type { Song } from '../../types';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';
import { SongActionsMenu, songFromPlaylistItem } from '../shared/SongActionsMenu';
import { CoverImage } from '../shared/CoverImage';

/** Fraction of row width a swipe must travel to trigger an action. */
const SWIPE_THRESHOLD = 0.3;

interface SwipeableRowProps {
  item: PlaylistItem;
  onSwipeLeft?: (item: PlaylistItem) => void;
  onSwipeRight?: (item: PlaylistItem) => void;
  children: React.ReactNode;
}

/**
 * Wraps a playlist row with horizontal swipe gesture support (touch + mouse).
 * Drag state is tracked via refs to avoid re-renders during tracking;
 * only the visual swipe delta lives in state.
 */
const SwipeableRow: React.FC<SwipeableRowProps> = ({ item, onSwipeLeft, onSwipeRight, children }) => {
  const [swipeDelta, setSwipeDelta] = useState(0);
  const dragState = useRef({
    startX: 0,
    startY: 0,
    rowWidth: 0,
    isHorizontalSwipe: false,
    isTracking: false,
  });
  // True once the pointer has moved far enough to count as a drag rather than a click.
  const wasDragged = useRef(false);
  const rowRef = useRef<HTMLDivElement>(null);

  /** Begin tracking a swipe from the given pointer position. */
  const startDrag = useCallback((clientX: number, clientY: number) => {
    const rowEl = rowRef.current;
    const rowWidth = rowEl ? rowEl.getBoundingClientRect().width : 0;
    dragState.current = {
      startX: clientX,
      startY: clientY,
      rowWidth,
      isHorizontalSwipe: false,
      isTracking: true,
    };
    wasDragged.current = false;
  }, []);

  /** Track pointer movement, clamping to the row width. */
  const moveDrag = useCallback((clientX: number, clientY: number) => {
    const state = dragState.current;
    if (!state.isTracking) return;
    const deltaX = clientX - state.startX;
    const deltaY = clientY - state.startY;

    // If vertical movement dominates, cancel swipe tracking and let
    // the browser handle default vertical scrolling.
    if (!state.isHorizontalSwipe) {
      if (Math.abs(deltaY) > Math.abs(deltaX)) {
        state.isTracking = false;
        setSwipeDelta(0);
        return;
      }
      state.isHorizontalSwipe = true;
    }

    // Mark as a real drag once the pointer moves more than a few pixels.
    if (!wasDragged.current && (Math.abs(deltaX) > 5 || Math.abs(deltaY) > 5)) {
      wasDragged.current = true;
    }

    const clamped = Math.max(-state.rowWidth, Math.min(state.rowWidth, deltaX));
    setSwipeDelta(clamped);
  }, []);

  /** End tracking, fire the callback if the threshold was crossed, snap back. */
  const endDrag = useCallback((clientX: number) => {
    const state = dragState.current;
    if (!state.isTracking) return;
    state.isTracking = false;

    const deltaX = clientX - state.startX;
    const threshold = state.rowWidth * SWIPE_THRESHOLD;

    if (deltaX < -threshold && onSwipeLeft) {
      onSwipeLeft(item);
    } else if (deltaX > threshold && onSwipeRight) {
      onSwipeRight(item);
    }

    // Always snap back to origin.
    setSwipeDelta(0);
  }, [item, onSwipeLeft, onSwipeRight]);

  // ── Touch handlers ──────────────────────────────────────────────────────────
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const touch = e.touches[0];
    startDrag(touch.clientX, touch.clientY);
  }, [startDrag]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    const touch = e.touches[0];
    moveDrag(touch.clientX, touch.clientY);
  }, [moveDrag]);

  const handleTouchEnd = useCallback((e: React.TouchEvent) => {
    const touch = e.changedTouches[0];
    endDrag(touch.clientX);
  }, [endDrag]);

  // ── Mouse handlers (desktop drag) ───────────────────────────────────────────
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    // Left button only — leave right-click (context menu) untouched.
    if (e.button !== 0) return;
    startDrag(e.clientX, e.clientY);

    const onMove = (ev: MouseEvent) => {
      moveDrag(ev.clientX, ev.clientY);
      // Only lock the cursor / selection once it's a real drag, so plain
      // clicks feel normal.
      if (wasDragged.current) {
        document.body.style.cursor = 'grabbing';
        document.body.style.userSelect = 'none';
      }
    };
    const onUp = (ev: MouseEvent) => {
      endDrag(ev.clientX);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [startDrag, moveDrag, endDrag]);

  // Suppress the click that follows a drag so it doesn't trigger click-to-play.
  const handleClickCapture = useCallback((e: React.MouseEvent) => {
    if (wasDragged.current) {
      e.stopPropagation();
      e.preventDefault();
    }
  }, []);

  const threshold = dragState.current.rowWidth * SWIPE_THRESHOLD;
  const showLeftBg = threshold > 0 && swipeDelta <= -threshold;
  const showRightBg = threshold > 0 && swipeDelta >= threshold;

  return (
    <div
      className="relative overflow-hidden"
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      onMouseDown={handleMouseDown}
      onClickCapture={handleClickCapture}
    >
      {/* Left-swipe background — revealed on the right as the row moves left. */}
      <div
        className={`absolute inset-0 flex items-center justify-end pr-4 transition-colors duration-150 ${showLeftBg ? 'bg-red-500/20' : 'bg-transparent'}`}
      >
        <div className={`flex items-center gap-1 text-red-400 text-[10px] font-medium transition-opacity duration-150 ${showLeftBg ? 'opacity-100' : 'opacity-0'}`}>
          <Trash2 className="w-3 h-3" />
          <span>Delete</span>
        </div>
      </div>
      {/* Right-swipe background — revealed on the left as the row moves right. */}
      <div
        className={`absolute inset-0 flex items-center justify-start pl-4 transition-colors duration-150 ${showRightBg ? 'bg-emerald-500/20' : 'bg-transparent'}`}
      >
        <div className={`flex items-center gap-1 text-emerald-400 text-[10px] font-medium transition-opacity duration-150 ${showRightBg ? 'opacity-100' : 'opacity-0'}`}>
          <X className="w-3 h-3" />
          <span>Remove</span>
        </div>
      </div>
      {/* Row content — shifts horizontally by the swipe delta. */}
      <div ref={rowRef} style={{ transform: `translateX(${swipeDelta}px)` }} className="relative z-10">
        {children}
      </div>
    </div>
  );
};

interface PlaylistSidebarProps {
  /** Omitted when embedded — the host section owns the header. */
  onClose?: () => void;
  /** Rendered inside a tabbed section: no header row, actions move to the footer. */
  embedded?: boolean;
  /** Enable horizontal swipe gestures on playlist rows (mobile). */
  swipeEnabled?: boolean;
  /** Called when a row is swiped left past the threshold (delete). */
  onSwipeLeft?: (item: PlaylistItem) => void;
  /** Called when a row is swiped right past the threshold (remove). */
  onSwipeRight?: (item: PlaylistItem) => void;
}

export const PlaylistSidebar: React.FC<PlaylistSidebarProps> = ({
  onClose,
  embedded = false,
  swipeEnabled = false,
  onSwipeLeft,
  onSwipeRight,
}) => {
  const playlist = usePlaylist();
  const { t } = useTranslation();
  const currentSongId = usePlaybackSelector(s => s.currentTrack?.id ?? null);
  const { disguiseArtist, disguiseTitle } = useDisguiseMode();
  const [downloading, setDownloading] = useState(false);
  const { token } = useAuth();

  /** Convert a PlaylistItem to a Song shape for the download utility. */
  const playlistItemToSong = useCallback((item: PlaylistItem): Song => ({
    id: item.id,
    title: item.title || 'Untitled',
    style: item.style || '',
    caption: item.style || '',
    lyrics: '',
    audioUrl: item.audioUrl,
    masteredAudioUrl: item.masteredAudioUrl || '',
    coverUrl: item.coverUrl || '',
    duration: item.duration || 0,
    artistName: item.artistName || '',
    tags: [],
  }), []);

  const handleDownloadAll = useCallback(async () => {
    if (downloading || playlist.items.length === 0 || !token) return;
    setDownloading(true);
    try {
      const songs = playlist.items.map(playlistItemToSong);
      const failures = await downloadAll(songs);
      if (failures.length) window.alert(`Some downloads failed:\n${failures.join('\n')}`);
    } catch (err) {
      window.alert(`Download failed: ${(err as Error).message}`);
    } finally {
      setDownloading(false);
    }
  }, [playlist.items, downloading, playlistItemToSong, token]);

  const handlePlay = useCallback((item: PlaylistItem) => {
    const allTracks = playlist.items.map(playlistItemToTrack);
    const clickedTrack = playlistItemToTrack(item);
    // The current track pauses/resumes in place; reloading it would restart it.
    if (clickedTrack.id === currentSongId) { togglePlay(); return; }
    playFromList(clickedTrack, allTracks, 'playlist');
  }, [playlist.items, currentSongId]);

  // Play All always (re)installs the playlist as the queue and starts at the top.
  const handlePlayAll = useCallback(() => {
    playFromList(playlistItemToTrack(playlist.items[0]), playlist.items.map(playlistItemToTrack), 'playlist');
  }, [playlist.items]);

  const totalDuration = useMemo(() => {
    const total = playlist.items.reduce((sum, i) => sum + (i.duration || 0), 0);
    if (total <= 0) return '0:00';
    return `${Math.floor(total / 60)}:${String(Math.floor(total % 60)).padStart(2, '0')}`;
  }, [playlist.items]);

  return (
    <div className={`flex flex-col h-full ${embedded ? '' : 'bg-zinc-50/80 dark:bg-zinc-950/80 backdrop-blur-sm'}`}>
      {/* Header — the embedded form gets its header from the enclosing tab strip */}
      {!embedded && (
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-white/5 flex-shrink-0">
        <div className="flex items-center gap-2">
          <ListMusic className="w-4 h-4 text-pink-400" />
          <span className="text-xs font-bold text-zinc-800 dark:text-zinc-200 uppercase tracking-wider">{t('playlist.title')}</span>
          {playlist.items.length > 0 && (
            <span className="text-[10px] text-zinc-500 font-normal">
              {playlist.items.length} · {totalDuration}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {playlist.items.length > 0 && (
            <button onClick={handleDownloadAll} disabled={downloading}
              className="p-1 rounded-md text-zinc-600 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-40"
              title={downloading ? 'Downloading...' : 'Download all tracks'}>
              {downloading
                ? <Loader2 className="w-3 h-3 animate-spin" />
                : <DownloadCloud className="w-3 h-3" />}
            </button>
          )}
          {playlist.items.length > 0 && (
            <button onClick={playlist.clear}
              className="p-1 rounded-md text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              title={t('playlist.clearPlaylist')}>
              <Trash2 className="w-3 h-3" />
            </button>
          )}
          {onClose && (
            <button onClick={onClose}
              className="p-1 rounded-md text-zinc-500 hover:text-white hover:bg-white/10 transition-colors"
              title={t('playlist.closePlaylist')}>
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>
      )}

      {playlist.needsImport && (
        <div className="px-3 py-2 border-b border-zinc-200 dark:border-white/10 text-xs text-zinc-600 dark:text-zinc-300">
          <p>Browser playlist found. Import it to save the ordered list on this installation.</p>
          <button className="mt-1 text-pink-500 hover:text-pink-400 underline" onClick={() => void playlist.importBrowserPlaylist()}>
            {playlist.replacingServer ? 'Import and replace saved playlist' : 'Import browser playlist'}{playlist.saveError ? ' (retry)' : ''}
          </button>
        </div>
      )}
      {playlist.saveError && (
        <div className="px-3 py-2 border-b border-red-300 dark:border-red-900 text-xs text-red-600 dark:text-red-300" role="alert">
          <p>Playlist save failed: {playlist.saveError}. Your local edit is still visible.</p>
          {!playlist.needsImport && <button className="mt-1 underline" onClick={() => void playlist.retryPlaylistSave()}>Reload and reapply</button>}
        </div>
      )}
      {playlist.saving && <div className="px-3 py-1 text-xs text-zinc-500" role="status">Saving playlist…</div>}

      {/* Track list */}
      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-hide">
        {playlist.items.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-6 py-8 gap-3">
            <div className="w-14 h-14 rounded-full bg-white dark:bg-zinc-900 flex items-center justify-center border border-zinc-200 dark:border-white/5">
              <Music className="w-6 h-6 text-zinc-700" />
            </div>
            <div>
              <p className="text-xs text-zinc-500 font-medium mb-1">{t('playlist.empty')}</p>
              <p className="text-[10px] text-zinc-600 leading-relaxed max-w-[200px]">
                Click the <ListPlus className="w-3 h-3 inline text-pink-400" /> button next to any track to add it here.
              </p>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-white/5">
            {playlist.items.map((item, idx) => {
              const isPlaying = currentSongId === item.id;
              const rowClassName = `group flex items-center gap-2 px-3 py-2 hover:bg-white/[0.04] transition-colors ${isPlaying ? 'bg-pink-500/10' : ''}`;
              const rowContent = (
                <>
                  {/* Track number / play button */}
                  <div className="w-6 flex-shrink-0 flex items-center justify-center">
                    <span className={`text-[10px] font-mono group-hover:hidden ${isPlaying ? 'text-pink-400 font-bold' : 'text-zinc-600'}`}>
                      {idx + 1}
                    </span>
                    <button onClick={() => handlePlay(item)}
                      className="hidden group-hover:flex w-5 h-5 items-center justify-center rounded-full hover:bg-white/10 transition-colors"
                      title={isPlaying ? 'Now Playing' : 'Play'}>
                      {isPlaying
                        ? <Square className="w-2.5 h-2.5 text-pink-400" />
                        : <Play className="w-2.5 h-2.5 text-white ml-0.5" />}
                    </button>
                  </div>

                  {/* Cover art */}
                  <div className="w-9 h-9 rounded-md overflow-hidden flex-shrink-0 bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center">
                    <CoverImage url={item.coverUrl} seed={item.id} iconSize={14} />
                  </div>

                  {/* Title / Artist */}
                  <div className="flex-1 min-w-0 cursor-pointer" onClick={() => handlePlay(item)}>
                    <p className={`text-[11px] font-medium truncate leading-tight ${isPlaying ? 'text-pink-300' : 'text-zinc-800 dark:text-zinc-200'}`}>
                      {disguiseTitle(item.title || 'Untitled')}
                    </p>
                    {item.artistName && (
                      <p className="text-[9px] text-zinc-500 truncate leading-tight">{disguiseArtist(item.artistName)}</p>
                    )}
                  </div>

                  {/* Actions — appear on hover between title and duration */}
                  <div className="hidden group-hover:flex items-center gap-0 flex-shrink-0">
                    {/* Download moved onto the shared menu; up/down/remove stay
                        out here because they reorder the playlist rather than
                        acting on the song. */}
                    <SongActionsMenu
                      song={songFromPlaylistItem(item)}
                      size={12}
                      className="!p-0.5"
                      onDownload={() => downloadTrack(
                        { ...songFromPlaylistItem(item), artistName: item.artistName || '' } as any,
                        { artistName: item.artistName || '' },
                      )}
                    />
                    <button onClick={() => playlist.move(item.id, 'up')} disabled={idx === 0}
                      className="p-0.5 text-zinc-600 hover:text-white transition-colors disabled:opacity-20" title={t('playlist.moveUp')}>
                      <ChevronUp className="w-3 h-3" />
                    </button>
                    <button onClick={() => playlist.move(item.id, 'down')} disabled={idx === playlist.items.length - 1}
                      className="p-0.5 text-zinc-600 hover:text-white transition-colors disabled:opacity-20" title={t('playlist.moveDown')}>
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    <button onClick={() => playlist.remove(item.id)}
                      className="p-0.5 text-zinc-600 hover:text-red-400 transition-colors ml-0.5" title={t('playlist.remove')}>
                      <X className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Duration — always visible, pinned right */}
                  {item.duration && item.duration > 0 && (
                    <span className="text-[9px] text-zinc-600 font-mono flex-shrink-0 tabular-nums w-8 text-right ml-auto">
                      {Math.floor(item.duration / 60)}:{String(Math.floor(item.duration % 60)).padStart(2, '0')}
                    </span>
                  )}
                </>
              );

              if (!swipeEnabled) {
                return (
                  <div key={item.id} className={rowClassName}>
                    {rowContent}
                  </div>
                );
              }

              return (
                <SwipeableRow key={item.id} item={item} onSwipeLeft={onSwipeLeft} onSwipeRight={onSwipeRight}>
                  <div className={rowClassName}>
                    {rowContent}
                  </div>
                </SwipeableRow>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      {playlist.items.length > 0 && (
        <div className="flex items-center justify-between px-4 py-2 border-t border-zinc-200 dark:border-white/5 flex-shrink-0">
          <button onClick={handlePlayAll}
            className="flex items-center gap-1.5 text-[10px] font-semibold text-pink-400 hover:text-pink-300 transition-colors">
            <Play className="w-3 h-3" /> {t('playlist.playAll')}
          </button>
          <div className="flex items-center gap-1">
            <span className="text-[9px] text-zinc-600 font-mono">{totalDuration}</span>
            {embedded && (
              <>
                <button onClick={handleDownloadAll} disabled={downloading}
                  className="p-1 rounded-md text-zinc-600 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-40"
                  title={downloading ? 'Downloading...' : 'Download all tracks'}>
                  {downloading
                    ? <Loader2 className="w-3 h-3 animate-spin" />
                    : <DownloadCloud className="w-3 h-3" />}
                </button>
                <button onClick={playlist.clear}
                  className="p-1 rounded-md text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                  title={t('playlist.clearPlaylist')}>
                  <Trash2 className="w-3 h-3" />
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
