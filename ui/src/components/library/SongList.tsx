// SongList.tsx — Song library display with bulk selection + source filtering
// Ported from hot-step-9000 visual design with Tailwind.

import React, { useState, useCallback, useMemo } from 'react';
import {
  Play, Pause, Trash2, RotateCcw, Music, MoreHorizontal,
  Download, CheckSquare, Square, MinusSquare, X, Pencil, ListPlus, Image,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Song } from '../../types';
import { togglePlay, usePlayback } from '../../stores/playbackStore';

// ── Source filter definitions ────────────────────────────────────────────────

type SourceFilter = 'all' | 'create' | 'insta-gen' | 'lyric-studio' | 'cover-studio';

const SOURCE_FILTERS: { id: SourceFilter; label: string; color: string }[] = [
  { id: 'all',           label: 'All',           color: 'text-zinc-700 dark:text-zinc-300 bg-white/10 border-zinc-300 dark:border-white/10' },
  { id: 'create',        label: 'Custom-Gen',   color: 'text-violet-300 bg-violet-500/15 border-violet-500/25' },
  { id: 'insta-gen',     label: 'Auto-Gen',      color: 'text-fuchsia-300 bg-fuchsia-500/15 border-fuchsia-500/25' },
  { id: 'lyric-studio',  label: 'Lyric Studio',  color: 'text-pink-300 bg-pink-500/15 border-pink-500/25' },
  { id: 'cover-studio',  label: 'Cover Studio',  color: 'text-cyan-300 bg-cyan-500/15 border-cyan-500/25' },
];

function getSongSource(song: Song): string {
  return (song.generationParams as any)?.source || (song.generation_params as any)?.source || 'create';
}

/**
 * Trigger cover art generation for a song, poll until complete,
 * then dispatch a CustomEvent so App.tsx can update the song list.
 */
function triggerCoverArtGeneration(song: Song): void {
  const params = song.generationParams || song.generation_params as any;
  fetch('/api/cover-art/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      songId: song.id,
      title: song.title || '',
      style: song.style || params?.style || '',
      lyrics: song.lyrics || params?.lyrics || '',
      subject: params?.subject || '',
    }),
  }).then(async r => {
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      console.error('[CoverArt] Generate failed:', d);
      return;
    }
    const { jobId } = await r.json();
    console.log('[CoverArt] Generation started for', song.title || song.id);

    // Poll job status until succeeded/failed
    const poll = setInterval(async () => {
      try {
        const jr = await fetch(`/api/cover-art/generate/${jobId}`);
        if (!jr.ok) return;
        const job = await jr.json();
        if (job.status === 'succeeded') {
          clearInterval(poll);
          console.log('[CoverArt] Cover ready:', job.result?.coverUrl);
          // Notify App.tsx to update the song in state
          window.dispatchEvent(new CustomEvent('cover-art-updated', {
            detail: { songId: song.id, coverUrl: job.result?.coverUrl },
          }));
        } else if (job.status === 'failed') {
          clearInterval(poll);
          console.error('[CoverArt] Generation failed:', job.error);
        }
      } catch { /* network error — keep polling */ }
    }, 2000);

    // Safety: stop polling after 5 minutes
    setTimeout(() => clearInterval(poll), 300_000);
  }).catch(err => console.error('[CoverArt]', err));
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
  /** Show source filter tabs — true for Library page, false for Create page */
  showFilters?: boolean;
  /** Layout mode — 'list' for compact rows, 'grid' for rich cards */
  viewMode?: 'list' | 'grid';
  /** Override the header title */
  title?: string;
}

export const SongList: React.FC<SongListProps> = ({
  songs, currentSongId, onPlay, onDelete, onBulkDelete, onSelect, onReuse, onDownload, onRename, onAddToPlaylist,
  showFilters = true, viewMode = 'list', title = 'Library',
}) => {
  const { t } = useTranslation();
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
    const counts: Record<string, number> = { all: songs.length, create: 0, 'insta-gen': 0, 'lyric-studio': 0, 'cover-studio': 0 };
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
        <div className="text-lg font-medium text-zinc-600 dark:text-zinc-400">{t('library.noSongsYet')}</div>
        <div className="text-sm text-zinc-600 mt-1">{t('library.createFirstTrack')}</div>
      </div>
    );
  }

  const allSelected = selectedIds.size === filteredSongs.length && filteredSongs.length > 0;
  const someSelected = selectedIds.size > 0 && !allSelected;

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-bold text-white">{title}</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500 font-medium">
            {filteredSongs.length}{showFilters && sourceFilter !== 'all' ? ` / ${songs.length}` : ''} song{filteredSongs.length !== 1 ? 's' : ''}
          </span>
          {!selectionMode ? (
            <button
              onClick={() => setSelectionMode(true)}
              className="text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 px-2 py-1 rounded-lg hover:bg-white/5 transition-colors"
              title={t('library.selectTracks')}
            >
              {t('library.select')}
            </button>
          ) : (
            <button
              onClick={exitSelectionMode}
              className="text-xs text-zinc-600 dark:text-zinc-400 hover:text-white px-2 py-1 rounded-lg hover:bg-white/5 transition-colors"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Source Filter Tabs — only on Library page */}
      {showFilters && (
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
                    : 'text-zinc-500 bg-transparent border-transparent hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-white/5'
                  }
                `}
              >
                {f.label}
                {count > 0 && (
                  <span className={`
                    min-w-[18px] h-4 px-1 rounded-full text-[10px] font-bold
                    flex items-center justify-center
                    ${isActive ? 'bg-white/15' : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500'}
                  `}>
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Bulk Action Bar */}
      {selectionMode && (
        <div className="flex items-center gap-2 mb-3 px-3 py-2.5 rounded-xl bg-zinc-100/80 dark:bg-zinc-800/80 border border-zinc-200 dark:border-white/5">
          <button
            onClick={allSelected ? deselectAll : selectAll}
            className="p-1 rounded text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
            title={allSelected ? t('library.deselectAll') : t('library.selectAll')}
          >
            {allSelected
              ? <CheckSquare size={18} className="text-pink-400" />
              : someSelected
                ? <MinusSquare size={18} className="text-pink-400" />
                : <Square size={18} />
            }
          </button>

          <span className="text-xs text-zinc-600 dark:text-zinc-400 font-medium flex-1">
            {selectedIds.size === 0
              ? t('library.selectTracks')
              : t('library.selected', { count: selectedIds.size })
            }
          </span>

          {selectedIds.size > 0 && (
            <button
              onClick={handleBulkDelete}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 hover:text-red-300 text-xs font-semibold transition-all"
            >
              <Trash2 size={13} />
              {t('library.deleteCount', { count: selectedIds.size })}
            </button>
          )}
        </div>
      )}

      {/* Empty state */}
      {filteredSongs.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
          <Music size={32} className="mb-3 opacity-20" />
          <div className="text-sm text-zinc-500">
            {showFilters ? t('library.noFilterSongs', { filter: SOURCE_FILTERS.find(f => f.id === sourceFilter)?.label }) : t('library.noSongsYet')}
          </div>
        </div>
      )}

      {/* Grid View */}
      {viewMode === 'grid' && filteredSongs.length > 0 && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
          {filteredSongs.map(song => (
            <SongCard
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
            />
          ))}
        </div>
      )}

      {/* List View */}
      {viewMode === 'list' && filteredSongs.length > 0 && (
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
              showSourceBadge={showFilters && sourceFilter === 'all'}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ── Quality Score Badge ──────────────────────────────────────────────────────

interface ParsedQuality {
  score: number;
  metallic: number;
  wordCuts: number;
  noise: number;
}

function parseQualityScores(raw?: string): { unmastered?: ParsedQuality; mastered?: ParsedQuality } | null {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch { return null; }
}

function qualityColor(score: number): string {
  if (score >= 0.8) return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
  if (score >= 0.5) return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
  return 'text-red-400 bg-red-500/10 border-red-500/20';
}

const QualityBadge: React.FC<{ song: Song }> = ({ song }) => {
  const scores = parseQualityScores(song.quality_scores);
  if (!scores) return null;

  const [showTooltip, setShowTooltip] = React.useState(false);

  const renderBadge = (q: ParsedQuality, label?: string) => (
    <span
      className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${qualityColor(q.score)} cursor-help`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      {label ? `${label}: ` : ''}{Math.round(q.score * 100)}%
    </span>
  );

  const renderTooltip = () => {
    if (!showTooltip) return null;
    const entries: Array<{ label: string; q: ParsedQuality }> = [];
    if (scores.unmastered) entries.push({ label: 'Unmastered', q: scores.unmastered });
    if (scores.mastered) entries.push({ label: 'Mastered', q: scores.mastered });

    return (
      <div className="absolute bottom-full left-0 mb-1.5 z-50 p-2.5 rounded-lg bg-zinc-900 border border-white/10 shadow-xl min-w-[160px]">
        {entries.map(({ label, q }) => (
          <div key={label} className="mb-2 last:mb-0">
            <div className="text-[10px] font-bold text-zinc-300 mb-1">{label}</div>
            <div className="space-y-0.5">
              <div className="flex justify-between text-[10px]">
                <span className="text-zinc-500">Overall</span>
                <span className={qualityColor(q.score).split(' ')[0]}>{Math.round(q.score * 100)}%</span>
              </div>
              <div className="flex justify-between text-[10px]">
                <span className="text-zinc-500">Metallic</span>
                <span className={qualityColor(q.metallic).split(' ')[0]}>{Math.round(q.metallic * 100)}%</span>
              </div>
              <div className="flex justify-between text-[10px]">
                <span className="text-zinc-500">Word Cuts</span>
                <span className={qualityColor(q.wordCuts).split(' ')[0]}>{Math.round(q.wordCuts * 100)}%</span>
              </div>
              <div className="flex justify-between text-[10px]">
                <span className="text-zinc-500">Noise</span>
                <span className={qualityColor(q.noise).split(' ')[0]}>{Math.round(q.noise * 100)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const hasBoth = scores.unmastered && scores.mastered;

  return (
    <div className="relative inline-flex items-center gap-1">
      {scores.unmastered && renderBadge(scores.unmastered, hasBoth ? 'Raw' : undefined)}
      {scores.mastered && renderBadge(scores.mastered, hasBoth ? 'PP' : undefined)}
      {renderTooltip()}
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
  const { t } = useTranslation();
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
      <div className="relative w-11 h-11 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex-shrink-0 flex items-center justify-center overflow-hidden">
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
            className="absolute inset-0 flex items-center justify-center bg-black/30 dark:bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity"
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
            className="w-full text-sm font-medium bg-zinc-100 dark:bg-zinc-800 border border-pink-500/40 rounded-lg px-2 py-0.5 text-zinc-800 dark:text-zinc-200 outline-none focus:border-pink-500"
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
            <div className={`text-sm font-medium truncate ${isActive ? 'text-pink-400' : 'text-zinc-800 dark:text-zinc-200'}`}>
              {song.title || 'Untitled'}
            </div>
            {onRename && !selectionMode && (
              <button
                onClick={e => { e.stopPropagation(); setEditTitle(song.title || ''); setEditing(true); }}
                className="flex-shrink-0 p-0.5 rounded text-zinc-600 hover:text-zinc-700 dark:hover:text-zinc-300 opacity-0 group-hover/title:opacity-100 transition-opacity"
                title={t('library.rename')}
              >
                <Pencil size={11} />
              </button>
            )}
          </div>
        )}
        <div className="text-xs text-zinc-500 truncate mt-0.5">
          {song.style || song.caption || t('library.noDescription')}
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
          <span className="text-[10px] text-zinc-500 font-medium px-1.5 py-0.5 rounded bg-zinc-100/50 dark:bg-zinc-800/50">
            {gp.bpm} BPM
          </span>
        )}
        {gp?.keyScale && (
          <span className="text-[10px] text-zinc-500 font-medium px-1.5 py-0.5 rounded bg-zinc-100/50 dark:bg-zinc-800/50">
            {gp.keyScale}
          </span>
        )}
        <QualityBadge song={song} />
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
              className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-pink-400 hover:bg-white/10 transition-colors"
              title={t('library.addToPlaylist')}
            >
              <ListPlus size={15} />
            </button>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu); }}
            className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <MoreHorizontal size={16} />
          </button>

          {/* Context Menu */}
          {showMenu && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowMenu(false)} />
              <div className="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-xl shadow-xl py-1 min-w-[160px]">

                {onReuse && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onReuse(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                  >
                    <RotateCcw size={14} /> {t('library.reusePrompt')}
                  </button>
                )}
                {onDownload && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onDownload(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                  >
                    <Download size={14} /> {t('library.download')}
                  </button>
                )}
                <button
                  onClick={(e) => { e.stopPropagation(); onDelete(); setShowMenu(false); }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                >
                  <Trash2 size={14} /> {t('library.delete')}
                </button>

                {/* Cover Art Generation */}
                <div className="border-t border-zinc-200 dark:border-white/5 my-1" />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowMenu(false);
                    triggerCoverArtGeneration(song);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-violet-400 hover:bg-violet-500/10 transition-colors"
                >
                  <Image size={14} /> {song.coverUrl ? 'Regenerate Cover Art' : 'Generate Cover Art'}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

// ── SongCard — Grid view card ────────────────────────────────────────────────

interface SongCardProps {
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
}

const SongCard: React.FC<SongCardProps> = ({
  song, isActive, isPlaying, selectionMode, isSelected, onToggleSelect,
  onPlay, onSelect, onDelete, onReuse, onDownload, onAddToPlaylist,
}) => {
  const { t } = useTranslation();
  const [showMenu, setShowMenu] = React.useState(false);

  const formatDuration = (val: string | number | undefined) => {
    if (!val) return '--:--';
    if (typeof val === 'string') return val;
    const m = Math.floor(val / 60);
    const s = Math.floor(val % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const gp = song.generationParams || song.generation_params as any;
  const bpm = song.bpm || gp?.bpm;
  const keyScale = song.key_scale || gp?.keyScale;
  const model = song.dit_model || gp?.ditModel || '';
  const modelShort = model ? model.split('/').pop()?.replace(/\.gguf$/, '').substring(0, 20) : '';

  const formatDate = (d: string | Date | undefined) => {
    if (!d) return '';
    const date = typeof d === 'string' ? new Date(d) : d;
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffH = Math.floor(diffMs / 3600000);
    if (diffH < 1) return 'Just now';
    if (diffH < 24) return `${diffH}h ago`;
    const diffD = Math.floor(diffH / 24);
    if (diffD < 7) return `${diffD}d ago`;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  const handleClick = () => {
    if (selectionMode) onToggleSelect();
    else onSelect?.();
  };

  return (
    <div
      className={`
        group relative rounded-xl border overflow-hidden cursor-pointer
        transition-all duration-200 hover:scale-[1.02] hover:shadow-lg hover:shadow-black/20
        ${isSelected && selectionMode
          ? 'border-pink-500/40 bg-pink-500/5 ring-1 ring-pink-500/20'
          : isActive
            ? 'border-pink-500/30 bg-pink-500/5'
            : 'border-zinc-200 dark:border-white/5 bg-zinc-50/80 dark:bg-zinc-900/50 hover:border-zinc-300 dark:hover:border-zinc-300 dark:border-white/10 hover:bg-zinc-100/50 dark:hover:bg-zinc-100/50 dark:bg-zinc-800/50'
        }
      `}
      onClick={handleClick}
    >
      {/* Album Art Area */}
      <div className="relative aspect-square bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center overflow-hidden">
        {song.coverUrl || song.cover_url ? (
          <img src={song.coverUrl || song.cover_url} alt="" className="w-full h-full object-cover" />
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Music size={32} className="text-zinc-700" />
          </div>
        )}

        {/* Play overlay */}
        {!selectionMode && (
          <button
            onClick={(e) => { e.stopPropagation(); if (isActive) togglePlay(); else onPlay(); }}
            className="absolute inset-0 flex items-center justify-center bg-black/20 dark:bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <div className="w-12 h-12 rounded-full bg-white/15 backdrop-blur-sm flex items-center justify-center border border-white/20">
              {isPlaying
                ? <Pause size={20} className="text-white" />
                : <Play size={20} className="text-white ml-0.5" />
              }
            </div>
          </button>
        )}

        {/* Selection checkbox */}
        {selectionMode && (
          <div className="absolute top-2 left-2">
            {isSelected
              ? <CheckSquare size={20} className="text-pink-400 drop-shadow" />
              : <Square size={20} className="text-white/60 drop-shadow" />
            }
          </div>
        )}

        {/* Duration badge */}
        <div className="absolute bottom-2 right-2 px-1.5 py-0.5 rounded-md bg-black/20 dark:bg-black/40 dark:bg-black/70 backdrop-blur-sm text-[10px] font-mono text-white/80">
          {formatDuration(song.duration)}
        </div>

        {/* More menu */}
        {!selectionMode && (
          <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu); }}
              className="w-7 h-7 rounded-full bg-black/20 dark:bg-black/50 backdrop-blur-sm flex items-center justify-center text-white/70 hover:text-white hover:bg-black/20 dark:bg-black/40 dark:bg-black/70 transition-colors"
            >
              <MoreHorizontal size={14} />
            </button>
            {showMenu && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setShowMenu(false)} />
                <div className="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-xl shadow-xl py-1 min-w-[150px]">
                  {onReuse && (
                    <button onClick={(e) => { e.stopPropagation(); onReuse(); setShowMenu(false); }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors">
                      <RotateCcw size={12} /> {t('library.reusePrompt')}
                    </button>
                  )}
                  {onAddToPlaylist && (
                    <button onClick={(e) => { e.stopPropagation(); onAddToPlaylist(); setShowMenu(false); }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors">
                      <ListPlus size={12} /> {t('library.addToPlaylist')}
                    </button>
                  )}
                  {onDownload && (
                    <button onClick={(e) => { e.stopPropagation(); onDownload(); setShowMenu(false); }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors">
                      <Download size={12} /> {t('library.download')}
                    </button>
                  )}
                  <button onClick={(e) => { e.stopPropagation(); onDelete(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 transition-colors">
                    <Trash2 size={12} /> {t('library.delete')}
                  </button>

                  {/* Cover Art Generation */}
                  <div className="border-t border-zinc-200 dark:border-white/5 my-1" />
                  <button onClick={(e) => {
                    e.stopPropagation();
                    setShowMenu(false);
                    triggerCoverArtGeneration(song);
                  }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-violet-400 hover:bg-violet-500/10 transition-colors">
                    <Image size={12} /> {song.coverUrl || song.cover_url ? 'Regenerate Cover' : 'Generate Cover'}
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Card Body */}
      <div className="px-3 py-2.5">
        {/* Title */}
        <div className={`text-sm font-semibold truncate ${isActive ? 'text-pink-400' : 'text-zinc-800 dark:text-zinc-200'}`}>
          {song.title || 'Untitled'}
        </div>

        {/* Style / Caption */}
        <div className="text-[11px] text-zinc-500 truncate mt-0.5 leading-tight">
          {song.style || song.caption || t('library.noDescription')}
        </div>

        {/* Metadata Row */}
        <div className="flex items-center gap-1.5 mt-2 flex-wrap">
          {bpm && (
            <span className="text-[9px] text-zinc-500 font-medium px-1.5 py-0.5 rounded bg-zinc-100/80 dark:bg-zinc-800/80 border border-zinc-200 dark:border-white/5">
              {bpm} BPM
            </span>
          )}
          {keyScale && (
            <span className="text-[9px] text-zinc-500 font-medium px-1.5 py-0.5 rounded bg-zinc-100/80 dark:bg-zinc-800/80 border border-zinc-200 dark:border-white/5">
              {keyScale}
            </span>
          )}
          {modelShort && (
            <span className="text-[9px] text-violet-400/60 font-medium px-1.5 py-0.5 rounded bg-violet-500/5 border border-violet-500/10 truncate max-w-[100px]" title={model}>
              {modelShort}
            </span>
          )}
          <QualityBadge song={song} />
        </div>

        {/* Date */}
        <div className="text-[10px] text-zinc-600 mt-1.5">
          {formatDate(song.created_at || song.createdAt)}
        </div>
      </div>
    </div>
  );
};
