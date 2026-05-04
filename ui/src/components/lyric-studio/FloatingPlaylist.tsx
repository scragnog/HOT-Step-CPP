/**
 * FloatingPlaylist.tsx — Winamp-style draggable, resizable floating playlist window.
 *
 * Persists position, size, and open/minimized state in localStorage.
 * Lives at the Lyric Studio layout level so it floats above everything.
 */

import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import {
  Play, X, Minus, Maximize2,
  ListPlus, Trash2, ChevronUp, ChevronDown,
  Music, ListMusic, Square, Download,
} from 'lucide-react';
import type { Song } from '../../types';
import { usePlaylist, type PlaylistItem } from './playlistStore';
import { playFromList, playlistItemToTrack, usePlayback } from '../../stores/playbackStore';
import { DownloadModal } from '../shared/DownloadModal';

// ── Window state persistence ─────────────────────────────────────────────────

const WINDOW_KEY = 'lireek-playlistWindow';

interface WindowState {
  x: number; y: number; width: number; height: number; open: boolean; minimized: boolean;
}

const DEFAULT_STATE: WindowState = {
  x: -1, y: -1, width: 340, height: 420, open: false, minimized: false,
};

function loadWindowState(): WindowState {
  try {
    const raw = localStorage.getItem(WINDOW_KEY);
    if (!raw) return DEFAULT_STATE;
    return { ...DEFAULT_STATE, ...JSON.parse(raw) };
  } catch { return DEFAULT_STATE; }
}

function saveWindowState(state: WindowState): void {
  localStorage.setItem(WINDOW_KEY, JSON.stringify(state));
}

// ── Component ────────────────────────────────────────────────────────────────

export const FloatingPlaylist: React.FC = () => {
  const playlist = usePlaylist();
  const pb = usePlayback();
  const currentSongId = pb.currentTrack?.id ?? null;
  const [windowState, setWindowState] = useState<WindowState>(loadWindowState);
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const windowRef = useRef<HTMLDivElement>(null);
  const dragStart = useRef({ x: 0, y: 0, wx: 0, wy: 0 });
  const resizeStart = useRef({ x: 0, y: 0, w: 0, h: 0 });

  useEffect(() => {
    if (windowState.open && windowState.x === -1) {
      const x = Math.max(50, window.innerWidth - windowState.width - 40);
      const y = Math.max(50, window.innerHeight - windowState.height - 140);
      const newState = { ...windowState, x, y };
      setWindowState(newState);
      saveWindowState(newState);
    }
  }, [windowState.open]);

  const updateWindow = useCallback((updates: Partial<WindowState>) => {
    setWindowState(prev => {
      const next = { ...prev, ...updates };
      saveWindowState(next);
      return next;
    });
  }, []);

  // ── Dragging ─────────────────────────────────────────────────────────────
  const handleDragStart = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    e.preventDefault();
    setIsDragging(true);
    dragStart.current = { x: e.clientX, y: e.clientY, wx: windowState.x, wy: windowState.y };
  }, [windowState.x, windowState.y]);

  useEffect(() => {
    if (!isDragging) return;
    const handleMove = (e: MouseEvent) => {
      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;
      updateWindow({
        x: Math.max(0, Math.min(window.innerWidth - 100, dragStart.current.wx + dx)),
        y: Math.max(0, Math.min(window.innerHeight - 40, dragStart.current.wy + dy)),
      });
    };
    const handleUp = () => setIsDragging(false);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => { window.removeEventListener('mousemove', handleMove); window.removeEventListener('mouseup', handleUp); };
  }, [isDragging, updateWindow]);

  // ── Resizing ─────────────────────────────────────────────────────────────
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);
    resizeStart.current = { x: e.clientX, y: e.clientY, w: windowState.width, h: windowState.height };
  }, [windowState.width, windowState.height]);

  useEffect(() => {
    if (!isResizing) return;
    const handleMove = (e: MouseEvent) => {
      updateWindow({
        width: Math.max(260, resizeStart.current.w + (e.clientX - resizeStart.current.x)),
        height: Math.max(200, resizeStart.current.h + (e.clientY - resizeStart.current.y)),
      });
    };
    const handleUp = () => setIsResizing(false);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => { window.removeEventListener('mousemove', handleMove); window.removeEventListener('mouseup', handleUp); };
  }, [isResizing, updateWindow]);

  // ── Play via main player ─────────────────────────────────────────────────
  const handlePlayMain = useCallback((item: PlaylistItem) => {
    const allTracks = playlist.items.map(playlistItemToTrack);
    const clickedTrack = playlistItemToTrack(item);
    playFromList(clickedTrack, allTracks, 'playlist');
  }, [playlist.items]);

  const totalDuration = useMemo(() => {
    const total = playlist.items.reduce((sum, i) => sum + (i.duration || 0), 0);
    if (total <= 0) return '';
    return `${Math.floor(total / 60)}:${String(Math.floor(total % 60)).padStart(2, '0')}`;
  }, [playlist.items]);

  // ── Toggle button (always visible) ───────────────────────────────────────
  if (!windowState.open) {
    return (
      <button onClick={() => updateWindow({ open: true, minimized: false })}
        className="fixed bottom-28 right-4 z-40 group" title="Open Playlist">
        <div className="relative">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-600 to-purple-700 shadow-lg shadow-pink-500/30 flex items-center justify-center hover:scale-110 transition-transform">
            <ListMusic className="w-5 h-5 text-white" />
          </div>
          {playlist.items.length > 0 && (
            <span className="absolute -top-1 -right-1 min-w-[18px] h-[18px] px-1 rounded-full bg-pink-500 text-[10px] font-bold text-white flex items-center justify-center shadow">
              {playlist.items.length}
            </span>
          )}
        </div>
      </button>
    );
  }

  // ── Minimized bar ────────────────────────────────────────────────────────
  if (windowState.minimized) {
    return (
      <div className="fixed z-40 select-none" style={{ left: windowState.x, top: windowState.y, width: windowState.width }}>
        <div className="flex items-center justify-between px-3 py-1.5 bg-gradient-to-r from-zinc-900 to-zinc-800 rounded-lg border border-white/10 shadow-xl cursor-move"
          onMouseDown={handleDragStart}>
          <span className="flex items-center gap-1.5 text-[11px] font-semibold text-zinc-300">
            <ListMusic className="w-3.5 h-3.5 text-pink-400" /> Playlist
            <span className="text-zinc-500">({playlist.items.length})</span>
          </span>
          <div className="flex items-center gap-0.5">
            <button onClick={() => updateWindow({ minimized: false })} className="p-1 text-zinc-500 hover:text-white transition-colors" title="Restore">
              <Maximize2 className="w-3 h-3" />
            </button>
            <button onClick={() => updateWindow({ open: false })} className="p-1 text-zinc-500 hover:text-red-400 transition-colors" title="Close">
              <X className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Full window ──────────────────────────────────────────────────────────
  return (
    <div ref={windowRef} className="fixed z-40 flex flex-col select-none"
      style={{ left: windowState.x, top: windowState.y, width: windowState.width, height: windowState.height, pointerEvents: 'auto' }}>
      {/* Title bar */}
      <div className="flex items-center justify-between px-3 py-2 bg-gradient-to-r from-zinc-900 via-zinc-850 to-zinc-900 rounded-t-xl border border-b-0 border-white/10 cursor-move flex-shrink-0"
        onMouseDown={handleDragStart}>
        <span className="flex items-center gap-1.5 text-xs font-bold text-zinc-200 tracking-wide">
          <ListMusic className="w-4 h-4 text-pink-400" /> PLAYLIST
          {playlist.items.length > 0 && (
            <span className="text-[10px] font-normal text-zinc-500 ml-1">
              {playlist.items.length} track{playlist.items.length !== 1 ? 's' : ''}
              {totalDuration && <> · {totalDuration}</>}
            </span>
          )}
        </span>
        <div className="flex items-center gap-0.5">
          {playlist.items.length > 0 && (
            <button onClick={playlist.clear} className="p-1 text-zinc-600 hover:text-red-400 transition-colors" title="Clear all">
              <Trash2 className="w-3 h-3" />
            </button>
          )}
          <button onClick={() => updateWindow({ minimized: true })} className="p-1 text-zinc-500 hover:text-white transition-colors" title="Minimize">
            <Minus className="w-3 h-3" />
          </button>
          <button onClick={() => updateWindow({ open: false })} className="p-1 text-zinc-500 hover:text-red-400 transition-colors" title="Close">
            <X className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Track list */}
      <div className="flex-1 min-h-0 overflow-y-auto bg-black/90 backdrop-blur-sm border-x border-white/10 scrollbar-hide">
        {playlist.items.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-6 py-8 gap-2">
            <div className="w-12 h-12 rounded-full bg-zinc-900 flex items-center justify-center border border-white/5">
              <Music className="w-5 h-5 text-zinc-600" />
            </div>
            <p className="text-xs text-zinc-500 font-medium">Playlist empty</p>
            <p className="text-[10px] text-zinc-600 leading-relaxed max-w-[200px]">
              Click the <ListPlus className="w-3 h-3 inline text-pink-400" /> button next to any track to add it here.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-white/5">
            {playlist.items.map((item, idx) => {
              const isItemPlaying = currentSongId === item.id;
              return (
                <div key={item.id}
                  className={`group relative flex items-center gap-2 px-2.5 py-1.5 hover:bg-white/[0.04] transition-colors ${isItemPlaying ? 'bg-pink-500/10' : ''}`}>
                  <div className="w-6 flex-shrink-0 flex items-center justify-center">
                    <span className={`text-[10px] font-mono group-hover:hidden ${isItemPlaying ? 'text-pink-400 font-bold' : 'text-zinc-600'}`}>
                      {idx + 1}
                    </span>
                    <button onClick={() => handlePlayMain(item)}
                      className="hidden group-hover:flex w-5 h-5 items-center justify-center rounded-full hover:bg-white/10 transition-colors"
                      title={isItemPlaying ? 'Now Playing' : 'Play'}>
                      {isItemPlaying ? <Square className="w-2.5 h-2.5 text-pink-400" /> : <Play className="w-2.5 h-2.5 text-white ml-0.5" />}
                    </button>
                  </div>

                  {item.coverUrl ? (
                    <div className="w-8 h-8 rounded overflow-hidden flex-shrink-0 bg-zinc-800">
                      <img src={item.coverUrl} alt="" className="w-full h-full object-cover" />
                    </div>
                  ) : (
                    <div className="w-8 h-8 rounded flex-shrink-0 bg-zinc-800 flex items-center justify-center">
                      <Music className="w-3 h-3 text-zinc-600" />
                    </div>
                  )}

                  <div className="flex-1 min-w-0 cursor-pointer" onClick={() => handlePlayMain(item)}>
                    <p className={`text-[11px] font-medium truncate leading-tight ${isItemPlaying ? 'text-pink-300' : 'text-zinc-200'}`}>
                      {item.title || 'Untitled'}
                    </p>
                    {item.artistName && (
                      <p className="text-[9px] text-zinc-500 truncate leading-tight">{item.artistName}</p>
                    )}
                  </div>

                  {/* Action buttons — appear on hover between title and duration */}
                  <div className="hidden group-hover:flex items-center gap-0 flex-shrink-0">
                    <button onClick={(e) => {
                        e.stopPropagation();
                        setDownloadSong({
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
                        });
                      }}
                      className="p-0.5 text-zinc-600 hover:text-emerald-400 transition-colors" title="Download">
                      <Download className="w-3 h-3" />
                    </button>
                    <button onClick={() => playlist.move(item.id, 'up')} disabled={idx === 0}
                      className="p-0.5 text-zinc-600 hover:text-white transition-colors disabled:opacity-20" title="Move up">
                      <ChevronUp className="w-3 h-3" />
                    </button>
                    <button onClick={() => playlist.move(item.id, 'down')} disabled={idx === playlist.items.length - 1}
                      className="p-0.5 text-zinc-600 hover:text-white transition-colors disabled:opacity-20" title="Move down">
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    <button onClick={() => playlist.remove(item.id)}
                      className="p-0.5 text-zinc-600 hover:text-red-400 transition-colors ml-0.5" title="Remove">
                      <X className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Duration — always visible, pinned right */}
                  {item.duration && item.duration > 0 && (
                    <span className="text-[9px] text-zinc-600 font-mono flex-shrink-0 tabular-nums w-8 text-right">
                      {Math.floor(item.duration / 60)}:{String(Math.floor(item.duration % 60)).padStart(2, '0')}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer bar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-gradient-to-r from-zinc-900 to-zinc-800 rounded-b-xl border border-t-0 border-white/10 flex-shrink-0">
        {playlist.items.length > 0 ? (
          <button onClick={() => { if (playlist.items.length > 0) handlePlayMain(playlist.items[0]); }}
            className="flex items-center gap-1.5 text-[10px] font-semibold text-pink-400 hover:text-pink-300 transition-colors">
            <Play className="w-3 h-3" /> Play All
          </button>
        ) : (
          <span className="text-[10px] text-zinc-600">No tracks</span>
        )}
        <span className="text-[9px] text-zinc-600 font-mono">{totalDuration || '0:00'}</span>
      </div>

      {/* Resize handle */}
      <div className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize" onMouseDown={handleResizeStart}>
        <svg width="10" height="10" viewBox="0 0 10 10" className="absolute bottom-1.5 right-1.5 text-zinc-600 hover:text-zinc-400 transition-colors">
          <path d="M9 1L1 9M9 5L5 9M9 9L9 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </div>

      {/* Download Modal */}
      {downloadSong && (
        <DownloadModal
          song={downloadSong}
          isOpen={!!downloadSong}
          onClose={() => setDownloadSong(null)}
          artistName={downloadSong.artistName}
        />
      )}
    </div>
  );
};

// ── Toggle Button (for use anywhere) ─────────────────────────────────────────

export const PlaylistToggleButton: React.FC<{ className?: string }> = ({ className = '' }) => {
  const playlist = usePlaylist();
  const handleToggle = () => {
    const state = loadWindowState();
    saveWindowState({ ...state, open: !state.open, minimized: false });
    window.dispatchEvent(new CustomEvent('lireek-playlist-window-change'));
  };

  return (
    <button onClick={handleToggle}
      className={`relative p-1.5 rounded-md text-zinc-500 hover:text-pink-400 hover:bg-pink-500/10 transition-colors ${className}`}
      title="Toggle Playlist">
      <ListMusic className="w-4 h-4" />
      {playlist.items.length > 0 && (
        <span className="absolute -top-0.5 -right-0.5 min-w-[14px] h-[14px] px-0.5 rounded-full bg-pink-500 text-[8px] font-bold text-white flex items-center justify-center">
          {playlist.items.length}
        </span>
      )}
    </button>
  );
};
