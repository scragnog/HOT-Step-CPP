// SongList.tsx — Song library display with bulk selection + source filtering
// Ported from hot-step-9000 visual design with Tailwind.

import React, { useState, useCallback, useMemo } from 'react';
import {
  Play, Pause, Trash2, RotateCcw, Music, MoreHorizontal,
  Download, CheckSquare, Square, MinusSquare, X, Pencil, ListPlus,
} from 'lucide-react';
import type { Song } from '../../types';
import { togglePlay, usePlayback } from '../../stores/playbackStore';

// ── Source filter definitions ────────────────────────────────────────────────

type SourceFilter = 'all' | 'create' | 'lyric-studio' | 'cover-studio';

const SOURCE_FILTERS: { id: SourceFilter; label: string; color: string }[] = [
  { id: 'all',           label: 'All',           color: 'text-zinc-300 bg-white/10 border-white/10' },
  { id: 'create',        label: 'Create',        color: 'text-violet-300 bg-violet-500/15 border-violet-500/25' },
  { id: 'lyric-studio',  label: 'Lyric Studio',  color: 'text-pink-300 bg-pink-500/15 border-pink-500/25' },
  { id: 'cover-studio',  label: 'Cover Studio',  color: 'text-cyan-300 bg-cyan-500/15 border-cyan-500/25' },
];

function getSongSource(song: Song): string {
  return (song.generationParams as any)?.source || (song.generation_params as any)?.source || 'create';
}

// ── Component ────────────────────────────────────────────────────────────────

interface SongListProps {
  songs: Song[];
  currentSongId?: string;
  onPlay: (song: Song) => void;
  onDelete: (song: Song) => void;
  onBulkDelete?: (ids: string[]) => void;
  onSelect?: (song: Song) => void;
  onReuse?: (song: Song) => void;
  onDownload?: (song: Song) => void;
  onRename?: (song: Song, newTitle: string) => void;
  onAddToPlaylist?: (song: Song) => void;
}

export const SongList: React.FC<SongListProps> = ({
  songs, currentSongId, onPlay, onDelete, onBulkDelete, onSelect, onReuse, onDownload, onRename, onAddToPlaylist,
}) => {
  const playback = usePlayback();
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all');

  // Client-side source filtering
  const filteredSongs = useMemo(() => {
    if (sourceFilter === 'all') return songs;
    return songs.filter(s => getSongSource(s) === sourceFilter);
  }, [songs, sourceFilter]);

  // Count songs per source for filter badges
  const sourceCounts = useMemo(() => {
    const counts: Record<string, number> = { all: songs.length, create: 0, 'lyric-studio': 0, 'cover-studio': 0 };
    for (const s of songs) {
      const src = getSongSource(s);
      if (src in counts) counts[src]++;
      else counts.create++; // default to create for untagged
    }
    return counts;
  }, [songs]);

  const toggleSelection = useCallback((id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelectedIds(new Set(filteredSongs.map(s => s.id)));
  }, [filteredSongs]);

  const deselectAll = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  const exitSelectionMode = useCallback(() => {
    setSelectionMode(false);
    setSelectedIds(new Set());
  }, []);

  const handleBulkDelete = useCallback(() => {
    if (selectedIds.size === 0) return;
    onBulkDelete?.(Array.from(selectedIds));
    exitSelectionMode();
  }, [selectedIds, onBulkDelete, exitSelectionMode]);

  if (songs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-zinc-500">
        <Music size={48} className="mb-4 opacity-30" />
        <div className="text-lg font-medium text-zinc-400">No songs yet</div>
        <div className="text-sm text-zinc-600 mt-1">Create your first track to see it here</div>
      </div>
    );
  }

  const allSelected = selectedIds.size === filteredSongs.length && filteredSongs.length > 0;
  const someSelected = selectedIds.size > 0 && !allSelected;

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-bold text-white">Library</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500 font-medium">
            {filteredSongs.length}{sourceFilter !== 'all' ? ` / ${songs.length}` : ''} song{filteredSongs.length !== 1 ? 's' : ''}
          </span>
          {!selectionMode ? (
            <button
              onClick={() => setSelectionMode(true)}
              className="text-xs text-zinc-500 hover:text-zinc-300 px-2 py-1 rounded-lg hover:bg-white/5 transition-colors"
              title="Select tracks"
            >
              Select
            </button>
          ) : (
            <button
              onClick={exitSelectionMode}
              className="text-xs text-zinc-400 hover:text-white px-2 py-1 rounded-lg hover:bg-white/5 transition-colors"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Source Filter Tabs */}
      <div className="flex items-center gap-1.5 mb-4 overflow-x-auto scrollbar-hide">
        {SOURCE_FILTERS.map(f => {
          const isActive = sourceFilter === f.id;
          const count = sourceCounts[f.id] || 0;
          return (
            <button
              key={f.id}
              onClick={() => { setSourceFilter(f.id); exitSelectionMode(); }}
              className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold
                border transition-all duration-200 whitespace-nowrap
                ${isActive
                  ? f.color
                  : 'text-zinc-500 bg-transparent border-transparent hover:text-zinc-300 hover:bg-white/5'
                }
              `}
            >
              {f.label}
              {count > 0 && (
                <span className={`
                  min-w-[18px] h-4 px-1 rounded-full text-[10px] font-bold
                  flex items-center justify-center
                  ${isActive ? 'bg-white/15' : 'bg-zinc-800 text-zinc-500'}
                `}>
                  {count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Bulk Action Bar */}
      {selectionMode && (
        <div className="flex items-center gap-2 mb-3 px-3 py-2.5 rounded-xl bg-zinc-800/80 border border-white/5">
          <button
            onClick={allSelected ? deselectAll : selectAll}
            className="p-1 rounded text-zinc-400 hover:text-white transition-colors"
            title={allSelected ? 'Deselect all' : 'Select all'}
          >
            {allSelected
              ? <CheckSquare size={18} className="text-pink-400" />
              : someSelected
                ? <MinusSquare size={18} className="text-pink-400" />
                : <Square size={18} />
            }
          </button>

          <span className="text-xs text-zinc-400 font-medium flex-1">
            {selectedIds.size === 0
              ? 'Select tracks'
              : `${selectedIds.size} selected`
            }
          </span>

          {selectedIds.size > 0 && (
            <button
              onClick={handleBulkDelete}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 hover:text-red-300 text-xs font-semibold transition-all"
            >
              <Trash2 size={13} />
              Delete {selectedIds.size}
            </button>
          )}
        </div>
      )}

      {/* Empty state for filter */}
      {filteredSongs.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
          <Music size={32} className="mb-3 opacity-20" />
          <div className="text-sm text-zinc-500">No {SOURCE_FILTERS.find(f => f.id === sourceFilter)?.label} songs</div>
        </div>
      )}

      {/* Song List */}
      <div className="space-y-1">
        {filteredSongs.map(song => (
          <SongItem
            key={song.id}
            song={song}
            isActive={currentSongId === song.id}
            isPlaying={currentSongId === song.id && playback.isPlaying}
            selectionMode={selectionMode}
            isSelected={selectedIds.has(song.id)}
            onToggleSelect={() => toggleSelection(song.id)}
            onPlay={() => onPlay(song)}
            onSelect={() => onSelect?.(song)}
            onDelete={() => onDelete(song)}
            onReuse={() => onReuse?.(song)}
            onDownload={() => onDownload?.(song)}
            onRename={onRename ? (newTitle) => onRename(song, newTitle) : undefined}
            onAddToPlaylist={onAddToPlaylist ? () => onAddToPlaylist(song) : undefined}
            showSourceBadge={sourceFilter === 'all'}
          />
        ))}
      </div>
    </div>
  );
};

interface SongItemProps {
  song: Song;
  isActive: boolean;
  isPlaying: boolean;
  selectionMode: boolean;
  isSelected: boolean;
  onToggleSelect: () => void;
  onPlay: () => void;
  onSelect?: () => void;
  onDelete: () => void;
  onReuse?: () => void;
  onDownload?: () => void;
  onRename?: (newTitle: string) => void;
  onAddToPlaylist?: () => void;
  showSourceBadge?: boolean;
}

const SongItem: React.FC<SongItemProps> = ({
  song, isActive, isPlaying, selectionMode, isSelected, onToggleSelect,
  onPlay, onSelect, onDelete, onReuse, onDownload, onRename, onAddToPlaylist, showSourceBadge,
}) => {
  const [showMenu, setShowMenu] = React.useState(false);
  const [editing, setEditing] = React.useState(false);
  const [editTitle, setEditTitle] = React.useState(song.title || '');
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const commitRename = () => {
    const trimmed = editTitle.trim();
    if (trimmed && trimmed !== (song.title || '')) {
      onRename?.(trimmed);
    }
    setEditing(false);
  };

  const formatDuration = (val: string | number | undefined) => {
    if (!val) return '--:--';
    if (typeof val === 'string') return val;
    const m = Math.floor(val / 60);
    const s = Math.floor(val % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const gp = song.generationParams;

  const handleClick = () => {
    if (selectionMode) {
      onToggleSelect();
    } else {
      onSelect?.();
    }
  };

  return (
    <div
      className={`
        group relative flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer transition-all duration-200
        ${isSelected && selectionMode
          ? 'bg-pink-500/10 border border-pink-500/20'
          : isActive
            ? 'bg-pink-500/10 border border-pink-500/20'
            : 'hover:bg-white/5 border border-transparent'
        }
      `}
      onClick={handleClick}
    >
      {/* Selection Checkbox */}
      {selectionMode && (
        <div className="flex-shrink-0">
          {isSelected
            ? <CheckSquare size={18} className="text-pink-400" />
            : <Square size={18} className="text-zinc-600" />
          }
        </div>
      )}

      {/* Cover Art Thumbnail */}
      <div className="relative w-11 h-11 rounded-lg bg-zinc-800 flex-shrink-0 flex items-center justify-center overflow-hidden">
        {song.coverUrl ? (
          <img src={song.coverUrl} alt="" className="w-full h-full object-cover" />
        ) : (
          <Music size={18} className="text-zinc-600" />
        )}
        {/* Play overlay on hover (not in selection mode) */}
        {!selectionMode && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (isActive) {
                // Toggle play/pause via playback store — don't reload the track
                togglePlay();
              } else {
                onPlay();
              }
            }}
            className="absolute inset-0 flex items-center justify-center bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            {isPlaying ? <Pause size={16} className="text-white" /> : <Play size={16} className="text-white ml-0.5" />}
          </button>
        )}
      </div>

      {/* Song Info */}
      <div className="flex-1 min-w-0">
        {editing ? (
          <input
            ref={inputRef}
            className="w-full text-sm font-medium bg-zinc-800 border border-pink-500/40 rounded-lg px-2 py-0.5 text-zinc-200 outline-none focus:border-pink-500"
            value={editTitle}
            onChange={e => setEditTitle(e.target.value)}
            onBlur={commitRename}
            onKeyDown={e => {
              if (e.key === 'Enter') commitRename();
              if (e.key === 'Escape') { setEditTitle(song.title || ''); setEditing(false); }
            }}
            onClick={e => e.stopPropagation()}
          />
        ) : (
          <div className="flex items-center gap-1 group/title">
            <div className={`text-sm font-medium truncate ${isActive ? 'text-pink-400' : 'text-zinc-200'}`}>
              {song.title || 'Untitled'}
            </div>
            {onRename && !selectionMode && (
              <button
                onClick={e => { e.stopPropagation(); setEditTitle(song.title || ''); setEditing(true); }}
                className="flex-shrink-0 p-0.5 rounded text-zinc-600 hover:text-zinc-300 opacity-0 group-hover/title:opacity-100 transition-opacity"
                title="Rename"
              >
                <Pencil size={11} />
              </button>
            )}
          </div>
        )}
        <div className="text-xs text-zinc-500 truncate mt-0.5">
          {song.style || song.caption || 'No description'}
        </div>
      </div>

      {/* Metadata Badges */}
      <div className="hidden xl:flex items-center gap-2 flex-shrink-0">
        {showSourceBadge && (() => {
          const src = getSongSource(song);
          const cfg = SOURCE_FILTERS.find(f => f.id === src);
          if (!cfg || src === 'create') return null;
          return (
            <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border ${cfg.color}`}>
              {cfg.label}
            </span>
          );
        })()}
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

      {/* Actions (hidden in selection mode) */}
      {!selectionMode && (
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 relative">
          {onAddToPlaylist && (
            <button
              onClick={(e) => { e.stopPropagation(); onAddToPlaylist(); }}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-pink-400 hover:bg-white/10 transition-colors"
              title="Add to Playlist"
            >
              <ListPlus size={15} />
            </button>
          )}
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
      )}
    </div>
  );
};
