// SongList.tsx — Song library display
// Ported from hot-step-9000 visual design with Tailwind.

import React from 'react';
import { Play, Pause, Trash2, RotateCcw, Music, MoreHorizontal, Download } from 'lucide-react';
import type { Song } from '../../types';

interface SongListProps {
  songs: Song[];
  currentSongId?: string;
  onPlay: (song: Song) => void;
  onDelete: (song: Song) => void;
  onSelect?: (song: Song) => void;
  onReuse?: (song: Song) => void;
  onDownload?: (song: Song) => void;
}

export const SongList: React.FC<SongListProps> = ({
  songs, currentSongId, onPlay, onDelete, onSelect, onReuse, onDownload,
}) => {
  if (songs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-zinc-500">
        <Music size={48} className="mb-4 opacity-30" />
        <div className="text-lg font-medium text-zinc-400">No songs yet</div>
        <div className="text-sm text-zinc-600 mt-1">Create your first track to see it here</div>
      </div>
    );
  }

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-white">Library</h3>
        <span className="text-xs text-zinc-500 font-medium">
          {songs.length} song{songs.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Song Grid */}
      <div className="space-y-1">
        {songs.map(song => (
          <SongItem
            key={song.id}
            song={song}
            isActive={currentSongId === song.id}
            onPlay={() => onPlay(song)}
            onSelect={() => onSelect?.(song)}
            onDelete={() => onDelete(song)}
            onReuse={() => onReuse?.(song)}
            onDownload={() => onDownload?.(song)}
          />
        ))}
      </div>
    </div>
  );
};

interface SongItemProps {
  song: Song;
  isActive: boolean;
  onPlay: () => void;
  onSelect?: () => void;
  onDelete: () => void;
  onReuse?: () => void;
  onDownload?: () => void;
}

const SongItem: React.FC<SongItemProps> = ({
  song, isActive, onPlay, onSelect, onDelete, onReuse, onDownload,
}) => {
  const [showMenu, setShowMenu] = React.useState(false);

  const formatDuration = (val: string | number | undefined) => {
    if (!val) return '--:--';
    if (typeof val === 'string') return val;
    const m = Math.floor(val / 60);
    const s = Math.floor(val % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const gp = song.generationParams;

  return (
    <div
      className={`
        group relative flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer transition-all duration-200
        ${isActive
          ? 'bg-pink-500/10 border border-pink-500/20'
          : 'hover:bg-white/5 border border-transparent'
        }
      `}
      onClick={onSelect}
    >
      {/* Cover Art Thumbnail */}
      <div className="relative w-11 h-11 rounded-lg bg-zinc-800 flex-shrink-0 flex items-center justify-center overflow-hidden">
        {song.coverUrl ? (
          <img src={song.coverUrl} alt="" className="w-full h-full object-cover" />
        ) : (
          <Music size={18} className="text-zinc-600" />
        )}
        {/* Play overlay on hover */}
        <button
          onClick={(e) => { e.stopPropagation(); onPlay(); }}
          className="absolute inset-0 flex items-center justify-center bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity"
        >
          {isActive ? <Pause size={16} className="text-white" /> : <Play size={16} className="text-white ml-0.5" />}
        </button>
      </div>

      {/* Song Info */}
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium truncate ${isActive ? 'text-pink-400' : 'text-zinc-200'}`}>
          {song.title || 'Untitled'}
        </div>
        <div className="text-xs text-zinc-500 truncate mt-0.5">
          {song.caption || song.style || 'No description'}
        </div>
      </div>

      {/* Metadata Badges */}
      <div className="hidden xl:flex items-center gap-2 flex-shrink-0">
        {gp?.bpm && (
          <span className="text-[10px] text-zinc-500 font-medium px-1.5 py-0.5 rounded bg-zinc-800/50">
            {gp.bpm} BPM
          </span>
        )}
        {gp?.keyScale && (
          <span className="text-[10px] text-zinc-500 font-medium px-1.5 py-0.5 rounded bg-zinc-800/50">
            {gp.keyScale}
          </span>
        )}
      </div>

      {/* Duration */}
      <span className="text-xs text-zinc-500 font-mono flex-shrink-0 w-12 text-right">
        {formatDuration(song.duration)}
      </span>

      {/* Actions */}
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 relative">
        <button
          onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu); }}
          className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/10 transition-colors"
        >
          <MoreHorizontal size={16} />
        </button>

        {/* Context Menu */}
        {showMenu && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setShowMenu(false)} />
            <div className="absolute right-0 top-full mt-1 z-50 bg-zinc-900 border border-white/10 rounded-xl shadow-xl py-1 min-w-[160px]">
              {onReuse && (
                <button
                  onClick={(e) => { e.stopPropagation(); onReuse(); setShowMenu(false); }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                >
                  <RotateCcw size={14} /> Reuse Prompt
                </button>
              )}
              {onDownload && (
                <button
                  onClick={(e) => { e.stopPropagation(); onDownload(); setShowMenu(false); }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                >
                  <Download size={14} /> Download
                </button>
              )}
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(); setShowMenu(false); }}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
              >
                <Trash2 size={14} /> Delete
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
