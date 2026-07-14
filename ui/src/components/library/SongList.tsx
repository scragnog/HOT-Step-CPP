// SongList.tsx — Song library display with bulk selection + source filtering
// Ported from hot-step-9000 visual design with Tailwind.

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import ReactDOM from 'react-dom';
import {
  Play, Pause, Trash2, RotateCcw, Music, MoreHorizontal,
  Download, CheckSquare, Square, MinusSquare, X, Pencil, ListPlus, Image,
  LayoutGrid, List as ListIcon, Table2, ArrowLeftRight, Upload, Mic2, Loader2,
  Check, Columns3, ChevronLeft, ChevronRight, Disc3, Tags,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { Song } from '../../types';
import { togglePlay, usePlaybackSelector } from '../../stores/playbackStore';
import { songToTrack } from '../../stores/playbackStore';
import { useABCompareSelector, setTrackA, setTrackB, playAB, openModal as openABModal, clear as clearAB } from '../../stores/abCompareStore';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';
import { downloadAll } from '../../utils/downloadTrack';
import { HoverFullText } from '../shared/HoverFullText';
import { openCoverArtPrompt } from './CoverArtPromptModal';

// ── Source filter definitions ────────────────────────────────────────────────

type SourceFilter = 'all' | 'create' | 'insta-gen' | 'lyric-studio' | 'cover-studio' | 'repaint' | 'stem-builder' | 'builder';

const SOURCE_FILTERS: { id: SourceFilter; label: string; color: string }[] = [
  { id: 'all',           label: 'All',           color: 'text-zinc-700 dark:text-zinc-300 bg-white/10 border-zinc-300 dark:border-white/10' },
  { id: 'create',        label: 'Custom-Gen',   color: 'text-violet-300 bg-violet-500/15 border-violet-500/25' },
  { id: 'insta-gen',     label: 'Auto-Gen',      color: 'text-fuchsia-300 bg-fuchsia-500/15 border-fuchsia-500/25' },
  { id: 'lyric-studio',  label: 'Lyric Studio',  color: 'text-pink-300 bg-pink-500/15 border-pink-500/25' },
  { id: 'cover-studio',  label: 'Cover Studio',  color: 'text-cyan-300 bg-cyan-500/15 border-cyan-500/25' },
  { id: 'repaint',       label: 'Repaint',       color: 'text-amber-300 bg-amber-500/15 border-amber-500/25' },
  { id: 'stem-builder',  label: 'Stem Build',    color: 'text-emerald-300 bg-emerald-500/15 border-emerald-500/25' },
  { id: 'builder',       label: 'Song Builder',  color: 'text-indigo-300 bg-indigo-500/15 border-indigo-500/25' },
];

function getSongSource(song: Song): string {
  return (song.generationParams as any)?.source || (song.generation_params as any)?.source || 'create';
}

// ── Pagination ───────────────────────────────────────────────────────────────
const PAGE_SIZE_KEY = 'hs-library-pageSize';
const PAGE_SIZE_OPTIONS: (number | 'all')[] = [20, 40, 60, 80, 100, 'all'];

function loadPageSize(): number | 'all' {
  try {
    const v = localStorage.getItem(PAGE_SIZE_KEY);
    if (v === 'all') return 'all';
    const n = v ? parseInt(v, 10) : NaN;
    if ((PAGE_SIZE_OPTIONS as (number | string)[]).includes(n)) return n;
  } catch { /* ignore */ }
  return 40;
}
function savePageSize(v: number | 'all') {
  try { localStorage.setItem(PAGE_SIZE_KEY, String(v)); } catch { /* ignore */ }
}

/**
 * Manual per-track cover art (#67): open the prompt modal so the user can edit
 * the prompt before generating. The modal handles generation + polling and
 * dispatches `cover-art-updated` when done. Auto-generate-after-creation is a
 * separate server-side path and is unaffected.
 */
function triggerCoverArtGeneration(song: Song): void {
  openCoverArtPrompt(song);
}

// ── Portal Menu ──────────────────────────────────────────────────────────────
// Renders dropdown menus at document.body level to escape overflow clipping.

interface PortalMenuProps {
  anchorRef: React.RefObject<HTMLElement | null>;
  onClose: () => void;
  children: React.ReactNode;
}

const PortalMenu: React.FC<PortalMenuProps> = ({ anchorRef, onClose, children }) => {
  const [pos, setPos] = React.useState<{ top: number; left: number; flipped: boolean }>({ top: 0, left: 0, flipped: false });

  React.useLayoutEffect(() => {
    if (!anchorRef.current) return;
    const rect = anchorRef.current.getBoundingClientRect();
    const menuH = 360; // max estimated height
    const spaceBelow = window.innerHeight - rect.bottom;
    const flipped = spaceBelow < menuH && rect.top > spaceBelow;
    setPos({
      top: flipped ? rect.top : rect.bottom + 4,
      left: Math.max(8, rect.right - 168), // align right edge, min 8px from left
      flipped,
    });
  }, [anchorRef]);

  return ReactDOM.createPortal(
    <>
      <div className="fixed inset-0 z-[9998]" onClick={onClose} />
      <div
        className="fixed z-[9999] bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-xl shadow-xl py-1 min-w-[160px]"
        style={{
          top: pos.flipped ? undefined : pos.top,
          bottom: pos.flipped ? (window.innerHeight - pos.top + 4) : undefined,
          left: pos.left,
        }}
      >
        {children}
      </div>
    </>,
    document.body
  );
};

// ── Component ────────────────────────────────────────────────────────────────

// ── Reusable pagination controls (rendered top + bottom) ─────────────────────
const PaginationControls: React.FC<{
  pageSize: number | 'all';
  page: number;
  totalPages: number;
  onChangePageSize: (v: number | 'all') => void;
  onPage: (updater: (p: number) => number) => void;
}> = ({ pageSize, page, totalPages, onChangePageSize, onPage }) => (
  <div className="flex items-center gap-3 text-xs">
    <div className="flex items-center gap-1.5">
      <span className="text-zinc-500">Per page</span>
      <div className="flex items-center rounded-lg border border-zinc-200 dark:border-white/10 overflow-hidden">
        {PAGE_SIZE_OPTIONS.map(opt => (
          <button
            key={String(opt)}
            onClick={() => onChangePageSize(opt)}
            className={`px-2.5 py-1 transition-colors ${
              pageSize === opt
                ? 'bg-pink-500/20 text-pink-400'
                : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
            }`}
          >
            {opt === 'all' ? 'All' : opt}
          </button>
        ))}
      </div>
    </div>
    {pageSize !== 'all' && totalPages > 1 && (
      <div className="flex items-center gap-2 text-zinc-500">
        <button
          onClick={() => onPage(p => Math.max(1, p - 1))}
          disabled={page <= 1}
          className="p-1 rounded hover:text-zinc-300 hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title="Previous page"
        >
          <ChevronLeft size={16} />
        </button>
        <span className="tabular-nums whitespace-nowrap">Page {page} / {totalPages}</span>
        <button
          onClick={() => onPage(p => Math.min(totalPages, p + 1))}
          disabled={page >= totalPages}
          className="p-1 rounded hover:text-zinc-300 hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          title="Next page"
        >
          <ChevronRight size={16} />
        </button>
      </div>
    )}
  </div>
);

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
  /** Load this track's source audio + title/lyrics/style into Cover Studio. */
  onSendToCover?: (song: Song) => void;
  /** Open the metadata editor for this track. */
  onEditMetadata?: (song: Song) => void;
  /** Show source filter tabs — true for Library page, false for Create page */
  showFilters?: boolean;
  /** Layout mode — 'list' for compact rows, 'grid' for rich cards, 'table' for data-dense */
  viewMode?: 'list' | 'grid' | 'table';
  /** Override the header title */
  title?: string;
}

export const SongList: React.FC<SongListProps> = ({
  songs, currentSongId, onPlay, onDelete, onBulkDelete, onSelect, onReuse, onDownload, onRename, onAddToPlaylist,
  onSendToCover, onEditMetadata, showFilters = true, viewMode = 'list', title = 'Library',
}) => {
  const { t } = useTranslation();
  const isPlaying = usePlaybackSelector(s => s.isPlaying);
  const abTrackAId = useABCompareSelector(s => s.trackA?.id ?? null);
  const abTrackBId = useABCompareSelector(s => s.trackB?.id ?? null);
  const hasBothAB = !!abTrackAId && !!abTrackBId;
  const abTrackATitle = useABCompareSelector(s => s.trackA?.title ?? '');
  const abTrackBTitle = useABCompareSelector(s => s.trackB?.title ?? '');
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all');

  // Internal view mode — persisted per context (title), initialized from prop
  const storageKey = `hs-songlist-viewMode-${title}`;
  const [activeViewMode, setActiveViewMode] = useState<'list' | 'grid' | 'table'>(() => {
    try { const v = localStorage.getItem(storageKey); if (v === 'list' || v === 'grid' || v === 'table') return v; } catch {}
    return viewMode ?? 'list';
  });
  const changeViewMode = useCallback((m: 'list' | 'grid' | 'table') => {
    setActiveViewMode(m);
    try { localStorage.setItem(storageKey, m); } catch {}
  }, [storageKey]);

  // Client-side source filtering
  const filteredSongs = useMemo(() => {
    if (sourceFilter === 'all') return songs;
    return songs.filter(s => getSongSource(s) === sourceFilter);
  }, [songs, sourceFilter]);

  // Count songs per source for filter badges. Keys are derived from
  // SOURCE_FILTERS so adding a new source can't drift out of sync.
  const sourceCounts = useMemo(() => {
    const counts: Record<string, number> = { all: songs.length };
    for (const f of SOURCE_FILTERS) if (f.id !== 'all') counts[f.id] = 0;
    for (const s of songs) {
      const src = getSongSource(s);
      if (src in counts && src !== 'all') counts[src]++;
      else counts.create++; // default to create for untagged / unknown
    }
    return counts;
  }, [songs]);

  // ── Pagination ─────────────────────────────────────────────────────────────
  const [pageSize, setPageSizeState] = useState<number | 'all'>(() => loadPageSize());
  const [page, setPage] = useState(1);
  const changePageSize = useCallback((v: number | 'all') => {
    setPageSizeState(v); savePageSize(v); setPage(1);
  }, []);

  const totalPages = pageSize === 'all' ? 1 : Math.max(1, Math.ceil(filteredSongs.length / pageSize));
  // Reset to page 1 when the filter changes; clamp if the list shrinks.
  useEffect(() => { setPage(1); }, [sourceFilter]);
  useEffect(() => { setPage(p => Math.min(p, totalPages)); }, [totalPages]);

  const pagedSongs = useMemo(() => {
    if (pageSize === 'all') return filteredSongs;
    const start = (page - 1) * pageSize;
    return filteredSongs.slice(start, start + pageSize);
  }, [filteredSongs, page, pageSize]);

  const paginationProps = { pageSize, page, totalPages, onChangePageSize: changePageSize, onPage: setPage };

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

  const [bulkDownloading, setBulkDownloading] = useState(false);

  const handleBulkDownload = useCallback(async () => {
    if (selectedIds.size === 0 || bulkDownloading) return;
    const selected = filteredSongs.filter(s => selectedIds.has(s.id));
    if (selected.length === 0) return;
    setBulkDownloading(true);
    try {
      await downloadAll(selected);
    } finally {
      setBulkDownloading(false);
    }
  }, [selectedIds, filteredSongs, bulkDownloading]);

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

          {/* View mode toggle */}
          <div className="flex items-center rounded-lg border border-zinc-200 dark:border-white/10 overflow-hidden">
            {([['grid', LayoutGrid], ['list', ListIcon], ['table', Table2]] as const).map(([mode, Icon]) => (
              <button
                key={mode}
                onClick={() => changeViewMode(mode as 'list' | 'grid' | 'table')}
                className={`p-1.5 transition-colors ${
                  activeViewMode === mode
                    ? 'bg-pink-500/20 text-pink-400'
                    : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
                }`}
                title={mode.charAt(0).toUpperCase() + mode.slice(1) + ' view'}
              >
                <Icon size={14} />
              </button>
            ))}
          </div>

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
            <>
              <button
                onClick={handleBulkDownload}
                disabled={bulkDownloading}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 hover:text-emerald-300 text-xs font-semibold transition-all disabled:opacity-40"
              >
                {bulkDownloading
                  ? <><Loader2 size={13} className="animate-spin" /> Downloading…</>
                  : <><Download size={13} /> {t('library.downloadCount', { count: selectedIds.size })}</>}
              </button>
              <button
                onClick={handleBulkDelete}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 hover:text-red-300 text-xs font-semibold transition-all"
              >
                <Trash2 size={13} />
                {t('library.deleteCount', { count: selectedIds.size })}
              </button>
            </>
          )}
        </div>
      )}

      {/* A/B Comparison Bar */}
      {(abTrackAId || abTrackBId) && (
        <div className="mb-3 px-4 py-3 rounded-xl bg-gradient-to-r from-blue-500/5 via-zinc-900/0 to-orange-500/5 border border-white/10 flex items-center gap-3">
          <ArrowLeftRight size={16} className="text-pink-400 flex-shrink-0" />
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${
              abTrackAId ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' : 'bg-zinc-800 text-zinc-500 border-white/5'
            }`}>
              A {abTrackAId && <span className="font-normal truncate max-w-[100px]">{abTrackATitle}</span>}
            </span>
            <span className="text-[10px] text-zinc-600">vs</span>
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold border ${
              abTrackBId ? 'bg-orange-500/10 text-orange-400 border-orange-500/20' : 'bg-zinc-800 text-zinc-500 border-white/5'
            }`}>
              B {abTrackBId && <span className="font-normal truncate max-w-[100px]">{abTrackBTitle}</span>}
            </span>
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {hasBothAB && (
              <>
                <button
                  onClick={playAB}
                  className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[10px] font-semibold bg-pink-500/10 text-pink-400 border border-pink-500/20 hover:bg-pink-500/20 transition-colors"
                >
                  <Play size={10} /> Play A/B
                </button>
                <button
                  onClick={openABModal}
                  className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[10px] font-semibold bg-violet-500/10 text-violet-400 border border-violet-500/20 hover:bg-violet-500/20 transition-colors"
                >
                  <ArrowLeftRight size={10} /> Compare
                </button>
              </>
            )}
            <button
              onClick={clearAB}
              className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              title="Clear A/B"
            >
              <X size={12} />
            </button>
          </div>
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

      {/* Top pagination — grid/list (table view renders it inline beside Columns) */}
      {activeViewMode !== 'table' && filteredSongs.length > 0 && (
        <div className="flex justify-end mb-3">
          <PaginationControls {...paginationProps} />
        </div>
      )}

      {/* Grid View */}
      {activeViewMode === 'grid' && filteredSongs.length > 0 && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
          {pagedSongs.map(song => (
            <SongCard
              key={song.id}
              song={song}
              isActive={currentSongId === song.id}
              isPlaying={currentSongId === song.id && isPlaying}
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
              onSendToCover={onSendToCover ? () => onSendToCover(song) : undefined}
              onEditMetadata={onEditMetadata ? () => onEditMetadata(song) : undefined}
              abTrackAId={abTrackAId}
              abTrackBId={abTrackBId}
            />
          ))}
        </div>
      )}

      {/* List View */}
      {activeViewMode === 'list' && filteredSongs.length > 0 && (
        <div className="space-y-1">
          {pagedSongs.map(song => (
            <SongItem
              key={song.id}
              song={song}
              isActive={currentSongId === song.id}
              isPlaying={currentSongId === song.id && isPlaying}
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
              onSendToCover={onSendToCover ? () => onSendToCover(song) : undefined}
              onEditMetadata={onEditMetadata ? () => onEditMetadata(song) : undefined}
              showSourceBadge={showFilters && sourceFilter === 'all'}
              abTrackAId={abTrackAId}
              abTrackBId={abTrackBId}
            />
          ))}
        </div>
      )}

      {/* Table View */}
      {activeViewMode === 'table' && filteredSongs.length > 0 && (
        <SongTable
          songs={pagedSongs}
          currentSongId={currentSongId}
          isPlaying={isPlaying}
          selectionMode={selectionMode}
          selectedIds={selectedIds}
          onToggleSelect={toggleSelection}
          onPlay={onPlay}
          onSelect={onSelect}
          onDelete={onDelete}
          onReuse={onReuse}
          onDownload={onDownload}
          onRename={onRename}
          onSendToCover={onSendToCover}
          showSourceBadge={showFilters && sourceFilter === 'all'}
          abTrackAId={abTrackAId}
          abTrackBId={abTrackBId}
          topRight={<PaginationControls {...paginationProps} />}
        />
      )}

      {/* Bottom pagination bar */}
      {filteredSongs.length > 0 && (
        <div className="flex justify-end mt-4 pt-3 border-t border-zinc-200 dark:border-white/5">
          <PaginationControls {...paginationProps} />
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
  onSendToCover?: () => void;
  onEditMetadata?: () => void;
  showSourceBadge?: boolean;
  abTrackAId?: string | null;
  abTrackBId?: string | null;
}

const SongItem: React.FC<SongItemProps> = ({
  song, isActive, isPlaying, selectionMode, isSelected, onToggleSelect,
  onPlay, onSelect, onDelete, onReuse, onDownload, onRename, onAddToPlaylist, onSendToCover, onEditMetadata, showSourceBadge,
  abTrackAId, abTrackBId,
}) => {
  const { t } = useTranslation();
  const [showMenu, setShowMenu] = React.useState(false);
  const [editing, setEditing] = React.useState(false);
  const [editTitle, setEditTitle] = React.useState(song.title || '');
  const inputRef = React.useRef<HTMLInputElement>(null);
  const menuBtnRef = React.useRef<HTMLButtonElement>(null);
  const { isDisguised, disguiseTitle } = useDisguiseMode();

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
        ${song.id === abTrackAId ? 'border-l-2 !border-l-blue-500' : song.id === abTrackBId ? 'border-l-2 !border-l-orange-500' : ''}
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
              {disguiseTitle(song.title || 'Untitled')}
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
          {isDisguised ? '' : (song.style || song.caption || t('library.noDescription'))}
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
            ref={menuBtnRef}
            onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu); }}
            className="p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <MoreHorizontal size={16} />
          </button>

          {/* Context Menu — portaled to body to escape overflow clipping */}
          {showMenu && (
            <PortalMenu anchorRef={menuBtnRef} onClose={() => setShowMenu(false)}>
                {onReuse && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onReuse(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                  >
                    <RotateCcw size={14} /> {t('library.edit')}
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
                {onSendToCover && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onSendToCover(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-cyan-400 hover:bg-cyan-500/10 transition-colors"
                  >
                    <Disc3 size={14} /> {t('library.sendToCover', 'Send to Cover Studio')}
                  </button>
                )}
                {onEditMetadata && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onEditMetadata(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-amber-400 hover:bg-amber-500/10 transition-colors"
                  >
                    <Tags size={14} /> {t('metadata.editTitle', 'Edit Metadata')}
                  </button>
                )}
                <button
                  onClick={(e) => { e.stopPropagation(); onDelete(); setShowMenu(false); }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                >
                  <Trash2 size={14} /> {t('library.delete')}
                </button>

                {/* Export Params */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowMenu(false);
                    const params = song.generationParams || song.generation_params || {};
                    const exportData = { _format: 'hot-step-preset', _version: 1, ...params, title: song.title || '', caption: (params as any).caption || song.style || '', lyrics: (params as any).lyrics || song.lyrics || '' };
                    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${(song.title || 'song').slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}_params.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                >
                  <Upload size={14} /> {t('library.exportParams')}
                </button>

                {/* Retranscribe Lyrics */}
                <button
                  onClick={async (e) => {
                    e.stopPropagation();
                    setShowMenu(false);
                    try {
                      const { retranscribeLyrics } = await import('../../services/api');
                      const result = await retranscribeLyrics(song.id);
                      console.log(`[Retranscribe] ${result.wordCount} words, ${result.lineCount} lines`);
                    } catch (err: any) {
                      console.error('[Retranscribe] Failed:', err.message);
                    }
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-sky-400 hover:bg-sky-500/10 transition-colors"
                >
                  <Mic2 size={14} /> Retranscribe Lyrics
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

                {/* A/B Comparison */}
                <div className="border-t border-zinc-200 dark:border-white/5 my-1" />
                <button
                  onClick={(e) => { e.stopPropagation(); setTrackA(songToTrack(song)); setShowMenu(false); }}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors ${
                    song.id === abTrackAId ? 'text-blue-400 bg-blue-500/10' : 'text-blue-400/70 hover:bg-blue-500/10 hover:text-blue-400'
                  }`}
                >
                  <ArrowLeftRight size={14} /> Set as Track A
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); setTrackB(songToTrack(song)); setShowMenu(false); }}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors ${
                    song.id === abTrackBId ? 'text-orange-400 bg-orange-500/10' : 'text-orange-400/70 hover:bg-orange-500/10 hover:text-orange-400'
                  }`}
                >
                  <ArrowLeftRight size={14} /> Set as Track B
                </button>
            </PortalMenu>
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
  onSendToCover?: () => void;
  onEditMetadata?: () => void;
  abTrackAId?: string | null;
  abTrackBId?: string | null;
}

const SongCard: React.FC<SongCardProps> = ({
  song, isActive, isPlaying, selectionMode, isSelected, onToggleSelect,
  onPlay, onSelect, onDelete, onReuse, onDownload, onRename, onAddToPlaylist, onSendToCover, onEditMetadata,
  abTrackAId: _abTrackAId, abTrackBId: _abTrackBId,
}) => {
  const { t } = useTranslation();
  const [showMenu, setShowMenu] = React.useState(false);
  const menuBtnRef = React.useRef<HTMLButtonElement>(null);
  const [editing, setEditing] = React.useState(false);
  const [editTitle, setEditTitle] = React.useState(song.title || '');
  const renameInputRef = React.useRef<HTMLInputElement>(null);
  const { isDisguised, disguiseTitle } = useDisguiseMode();

  React.useEffect(() => {
    if (editing && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
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

  const formatDate = (d: string | Date | undefined) => {
    if (!d) return '';
    // SQLite datetime('now') stores UTC without 'Z' suffix — force UTC interpretation
    const date = typeof d === 'string'
      ? new Date(d.includes('T') || d.includes('Z') ? d : d.replace(' ', 'T') + 'Z')
      : d;
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
        group relative rounded-xl border overflow-hidden cursor-pointer aspect-square
        transition-all duration-200 hover:scale-[1.02] hover:shadow-lg hover:shadow-black/20
        ${isSelected && selectionMode
          ? 'border-pink-500/40 bg-pink-500/5 ring-1 ring-pink-500/20'
          : isActive
            ? 'border-pink-500/30 bg-pink-500/5'
            : 'border-zinc-200 dark:border-white/5 bg-zinc-50/80 dark:bg-zinc-900/50 hover:border-zinc-300 dark:hover:border-white/10 hover:bg-zinc-100/50 dark:hover:bg-zinc-800/50'
        }
      `}
      onClick={handleClick}
    >
      {/* Full-bleed image */}
      <div className="absolute inset-0 bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center">
        {song.coverUrl || song.cover_url ? (
          <img src={song.coverUrl || song.cover_url} alt="" className="w-full h-full object-cover" />
        ) : (
          <Music size={32} className="text-zinc-700" />
        )}
      </div>

      {/* Play overlay */}
      {!selectionMode && (
        <button
          onClick={(e) => { e.stopPropagation(); if (isActive) togglePlay(); else onPlay(); }}
          className="absolute inset-0 z-10 flex items-center justify-center bg-black/20 dark:bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity"
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
        <div className="absolute top-2 left-2 z-20">
          {isSelected
            ? <CheckSquare size={20} className="text-pink-400 drop-shadow" />
            : <Square size={20} className="text-white/60 drop-shadow" />
          }
        </div>
      )}

      {/* Duration badge */}
      <div className="absolute top-2 left-2 z-10 px-1.5 py-0.5 rounded-md bg-black/40 backdrop-blur-sm text-[10px] font-mono text-white/80"
        style={selectionMode ? { left: '2rem' } : undefined}
      >
        {formatDuration(song.duration)}
      </div>

      {/* More menu */}
      {!selectionMode && (
        <div className="absolute top-2 right-2 z-20 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            ref={menuBtnRef}
            onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu); }}
            className="w-7 h-7 rounded-full bg-black/40 backdrop-blur-sm flex items-center justify-center text-white/70 hover:text-white hover:bg-black/60 transition-colors"
          >
            <MoreHorizontal size={14} />
          </button>
          {showMenu && (
            <PortalMenu anchorRef={menuBtnRef} onClose={() => setShowMenu(false)}>
                {onReuse && (
                  <button onClick={(e) => { e.stopPropagation(); onReuse(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors">
                    <RotateCcw size={12} /> {t('library.edit')}
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
                {onSendToCover && (
                  <button onClick={(e) => { e.stopPropagation(); onSendToCover(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-cyan-400 hover:bg-cyan-500/10 transition-colors">
                    <Disc3 size={12} /> {t('library.sendToCover', 'Send to Cover Studio')}
                  </button>
                )}
                {onEditMetadata && (
                  <button onClick={(e) => { e.stopPropagation(); onEditMetadata(); setShowMenu(false); }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-amber-400 hover:bg-amber-500/10 transition-colors">
                    <Tags size={12} /> {t('metadata.editTitle', 'Edit Metadata')}
                  </button>
                )}
                <button onClick={(e) => { e.stopPropagation(); onDelete(); setShowMenu(false); }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 transition-colors">
                  <Trash2 size={12} /> {t('library.delete')}
                </button>

                {/* Export Params */}
                <button onClick={(e) => {
                  e.stopPropagation();
                  setShowMenu(false);
                  const params = song.generationParams || song.generation_params || {};
                  const exportData = { _format: 'hot-step-preset', _version: 1, ...params, title: song.title || '', caption: (params as any).caption || song.style || '', lyrics: (params as any).lyrics || song.lyrics || '' };
                  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `${(song.title || 'song').slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}_params.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors">
                  <Upload size={12} /> {t('library.exportParams')}
                </button>

                {/* Retranscribe Lyrics */}
                <button onClick={async (e) => {
                  e.stopPropagation();
                  setShowMenu(false);
                  try {
                    const { retranscribeLyrics } = await import('../../services/api');
                    const result = await retranscribeLyrics(song.id);
                    console.log(`[Retranscribe] ${result.wordCount} words, ${result.lineCount} lines`);
                  } catch (err: any) {
                    console.error('[Retranscribe] Failed:', err.message);
                  }
                }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-sky-400 hover:bg-sky-500/10 transition-colors">
                  <Mic2 size={12} /> Retranscribe Lyrics
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
            </PortalMenu>
          )}
        </div>
      )}

      {/* Info overlay — gradient from bottom */}
      <div className="absolute inset-x-0 bottom-0 z-10 p-3 pt-10 bg-gradient-to-t from-black/80 via-black/50 to-transparent pointer-events-none">
        {/* Title — editable */}
        {editing ? (
          <div className="pointer-events-auto">
            <input
              ref={renameInputRef}
              className="w-full text-sm font-semibold bg-black/40 backdrop-blur-sm border border-pink-500/40 rounded-lg px-2 py-0.5 text-white outline-none focus:border-pink-500"
              value={editTitle}
              onChange={e => setEditTitle(e.target.value)}
              onBlur={commitRename}
              onKeyDown={e => {
                if (e.key === 'Enter') commitRename();
                if (e.key === 'Escape') { setEditTitle(song.title || ''); setEditing(false); }
              }}
              onClick={e => e.stopPropagation()}
            />
          </div>
        ) : (
          <div className="flex items-center gap-1 group/title pointer-events-auto">
            <div className={`text-sm font-semibold drop-shadow-sm truncate ${isActive ? 'text-pink-400' : 'text-white'}`}>
              {disguiseTitle(song.title || 'Untitled')}
            </div>
            {onRename && !selectionMode && (
              <button
                onClick={e => { e.stopPropagation(); setEditTitle(song.title || ''); setEditing(true); }}
                className="flex-shrink-0 p-0.5 rounded text-white/50 hover:text-white opacity-0 group-hover/title:opacity-100 transition-opacity"
                title={t('library.rename')}
              >
                <Pencil size={11} />
              </button>
            )}
          </div>
        )}

        {/* Style / Caption — wraps, max 3 lines */}
        <div className="text-[11px] text-white/70 mt-0.5 leading-tight line-clamp-3">
          {isDisguised ? '' : (song.style || song.caption || t('library.noDescription'))}
        </div>

        {/* Quality + Date row */}
        <div className="flex items-center gap-2 mt-1.5 pointer-events-auto">
          <QualityBadge song={song} />
          <span className="text-[10px] text-white/50">
            {formatDate(song.created_at || song.createdAt)}
          </span>
        </div>
      </div>
    </div>
  );
};

// ── TableTitleCell — Inline-editable title for table rows ─────────────────────

const TableTitleCell: React.FC<{
  song: Song;
  isActive: boolean;
  onRename?: (newTitle: string) => void;
  disguiseTitle: (t: string) => string;
}> = ({ song, isActive, onRename, disguiseTitle }) => {
  const { t } = useTranslation();
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

  if (editing) {
    return (
      <input
        ref={inputRef}
        className="w-full text-xs font-medium bg-zinc-800 border border-pink-500/40 rounded px-1.5 py-0.5 text-zinc-200 outline-none focus:border-pink-500"
        value={editTitle}
        onChange={e => setEditTitle(e.target.value)}
        onBlur={commitRename}
        onKeyDown={e => {
          if (e.key === 'Enter') commitRename();
          if (e.key === 'Escape') { setEditTitle(song.title || ''); setEditing(false); }
        }}
        onClick={e => e.stopPropagation()}
      />
    );
  }

  return (
    <span className="flex items-center gap-1 group/title min-w-0">
      <span className={`font-medium truncate ${isActive ? 'text-pink-400' : 'text-zinc-200'}`}>
        {disguiseTitle(song.title || 'Untitled')}
      </span>
      {onRename && (
        <button
          onClick={e => { e.stopPropagation(); setEditTitle(song.title || ''); setEditing(true); }}
          className="flex-shrink-0 p-0.5 rounded text-zinc-600 hover:text-zinc-300 opacity-0 group-hover/title:opacity-100 transition-opacity"
          title={t('library.rename')}
        >
          <Pencil size={10} />
        </button>
      )}
    </span>
  );
};

// ── Style cell — truncated, with a hover tooltip showing the full text + copy ──
const TableStyleCell: React.FC<{ text: string }> = ({ text }) => (
  <HoverFullText text={text} className="text-zinc-500 truncate block" />
);

// ── SongTable — Table view with resizable columns ────────────────────────────

interface SongTableProps {
  songs: Song[];
  currentSongId?: string;
  isPlaying: boolean;
  selectionMode: boolean;
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onPlay: (song: Song) => void;
  onSelect?: (song: Song) => void;
  onDelete: (song: Song) => void;
  onReuse?: (song: Song) => void;
  onDownload?: (song: Song) => void;
  onRename?: (song: Song, newTitle: string) => void;
  onSendToCover?: (song: Song) => void;
  showSourceBadge?: boolean;
  abTrackAId?: string | null;
  abTrackBId?: string | null;
  /** Rendered on the right of the Columns toolbar row (e.g. pagination). */
  topRight?: React.ReactNode;
}

const SOURCE_BADGE_MAP: Record<string, { label: string; cls: string }> = {
  'insta-gen': { label: 'Auto-Gen', cls: 'text-fuchsia-300 bg-fuchsia-500/15 border-fuchsia-500/25' },
  'lyric-studio': { label: 'Lyric', cls: 'text-pink-300 bg-pink-500/15 border-pink-500/25' },
  'cover-studio': { label: 'Cover', cls: 'text-cyan-300 bg-cyan-500/15 border-cyan-500/25' },
  'repaint': { label: 'Repaint', cls: 'text-amber-300 bg-amber-500/15 border-amber-500/25' },
  'stem-builder': { label: 'Stem', cls: 'text-emerald-300 bg-emerald-500/15 border-emerald-500/25' },
  'builder': { label: 'Builder', cls: 'text-indigo-300 bg-indigo-500/15 border-indigo-500/25' },
};

// Column definitions — id, label, default width, min width, alignment.
// defaultHidden = not shown until the user enables it via the Columns menu.
type ColAlign = 'left' | 'center' | 'right';
interface ColDef { id: string; label: string; defaultW: number; minW: number; align: ColAlign; resizable: boolean; defaultHidden?: boolean }

const BASE_COLS: ColDef[] = [
  { id: 'thumb',   label: '',        defaultW: 40,  minW: 32,  align: 'left', resizable: false },
  { id: 'title',   label: 'Title',   defaultW: 320, minW: 80,  align: 'left', resizable: true },
  { id: 'style',   label: 'Style',   defaultW: 320, minW: 80,  align: 'left', resizable: true },
  { id: 'bpm',     label: 'BPM',     defaultW: 60,  minW: 40,  align: 'left', resizable: true },
  { id: 'key',     label: 'Key',     defaultW: 60,  minW: 40,  align: 'left', resizable: true },
  { id: 'model',   label: 'Model',   defaultW: 110, minW: 60,  align: 'left', resizable: true },
  { id: 'quality', label: 'Quality', defaultW: 70,  minW: 50,  align: 'left', resizable: true },
  { id: 'time',    label: 'Time',    defaultW: 56,  minW: 44,  align: 'left', resizable: true },
  { id: 'date',    label: 'Date',    defaultW: 80,  minW: 50,  align: 'left', resizable: true },
  // ── Generation data (hidden by default; enable via Columns menu) ──
  { id: 'seed',      label: 'Seed',      defaultW: 96,  minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'lmSeed',    label: 'LM Seed',   defaultW: 96,  minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'steps',     label: 'Steps',     defaultW: 56,  minW: 44, align: 'left', resizable: true, defaultHidden: true },
  { id: 'cfg',       label: 'CFG',       defaultW: 56,  minW: 44, align: 'left', resizable: true, defaultHidden: true },
  { id: 'shift',     label: 'Shift',     defaultW: 56,  minW: 44, align: 'left', resizable: true, defaultHidden: true },
  { id: 'solver',    label: 'Solver',    defaultW: 110, minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'scheduler', label: 'Scheduler', defaultW: 110, minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'guidanceMode', label: 'Guidance', defaultW: 100, minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'lora',      label: 'LoRA',      defaultW: 130, minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'loraScale', label: 'LoRA Scale', defaultW: 80, minW: 50, align: 'left', resizable: true, defaultHidden: true },
  { id: 'lm',        label: 'LM Model',  defaultW: 120, minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'vae',       label: 'VAE',       defaultW: 110, minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'timesig',   label: 'Time Sig',  defaultW: 68,  minW: 48, align: 'left', resizable: true, defaultHidden: true },
  { id: 'lang',      label: 'Lang',      defaultW: 60,  minW: 44, align: 'left', resizable: true, defaultHidden: true },
  { id: 'task',      label: 'Task',      defaultW: 100, minW: 60, align: 'left', resizable: true, defaultHidden: true },
  { id: 'instrumental', label: 'Instr.', defaultW: 60, minW: 48, align: 'center', resizable: true, defaultHidden: true },
  { id: 'mastered',  label: 'Mastered',  defaultW: 76,  minW: 56, align: 'center', resizable: true, defaultHidden: true },
  { id: 'seedType',  label: 'Seed Type', defaultW: 80,  minW: 56, align: 'left', resizable: true, defaultHidden: true },
  { id: 'lmSeedType', label: 'LM Seed Type', defaultW: 90, minW: 60, align: 'left', resizable: true, defaultHidden: true },
  // Holds up to 5 action buttons (edit/download/delete/A/B). Must be wide
  // enough that the fixed-layout <td overflow-hidden> never clips them — the
  // column is non-resizable so the user can't widen it manually (#58).
  { id: 'actions', label: 'Actions', defaultW: 158, minW: 158, align: 'right', resizable: false },
];

const SOURCE_COL: ColDef = { id: 'source', label: 'Source', defaultW: 80, minW: 50, align: 'left', resizable: true };
const SELECT_COL: ColDef = { id: 'select', label: '', defaultW: 32, minW: 32, align: 'left', resizable: false };
// Flexible spacer inserted before the actions column. It carries no width of
// its own and absorbs all leftover table width, which pins `actions` to the
// right edge while the resizable middle columns share the space between (#58).
const SPACER_COL: ColDef = { id: 'spacer', label: '', defaultW: 0, minW: 0, align: 'left', resizable: false };

const STORAGE_KEY = 'hs-table-colWidths';

function loadColWidths(): Record<string, number> {
  try { const v = localStorage.getItem(STORAGE_KEY); return v ? JSON.parse(v) : {}; } catch { return {}; }
}

function saveColWidths(w: Record<string, number>) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(w)); } catch { /* ignore */ }
}

// ── Column visibility (user-customizable columns) ────────────────────────────
const COL_VIS_KEY = 'hs-table-colVisibility';
// Structural columns the user can't hide — the table needs them to function.
const MANDATORY_COLS = new Set(['select', 'thumb', 'title', 'spacer', 'actions']);

function loadColVisibility(): Record<string, boolean> {
  try { const v = localStorage.getItem(COL_VIS_KEY); return v ? JSON.parse(v) : {}; } catch { return {}; }
}
function saveColVisibility(v: Record<string, boolean>) {
  try { localStorage.setItem(COL_VIS_KEY, JSON.stringify(v)); } catch { /* ignore */ }
}

/** Whether a column shows when the user hasn't explicitly toggled it. */
function colDefaultVisible(id: string): boolean {
  if (MANDATORY_COLS.has(id)) return true;
  if (id === 'source') return true; // shown whenever the source column applies
  return !BASE_COLS.find(c => c.id === id)?.defaultHidden;
}

/** Filename without path or model extension — for LoRA / LM / VAE cells. */
function baseName(p?: string): string {
  if (!p) return '';
  const file = p.replace(/\\/g, '/').split('/').pop() || '';
  return file.replace(/\.(safetensors|gguf|ckpt|pt|onnx|bin)$/i, '');
}

const SongTable: React.FC<SongTableProps> = ({
  songs, currentSongId, isPlaying, selectionMode, selectedIds, onToggleSelect,
  onPlay, onSelect, onDelete, onReuse, onDownload, onRename, onSendToCover, showSourceBadge,
  abTrackAId, abTrackBId, topRight,
}) => {
  const { t } = useTranslation();
  const { isDisguised, disguiseTitle } = useDisguiseMode();

  // Persisted per-column visibility (default visible). Mandatory cols ignore it.
  const [colVisible, setColVisible] = useState<Record<string, boolean>>(() => loadColVisibility());
  const [colMenuOpen, setColMenuOpen] = useState(false);
  const isColVisible = (id: string) => MANDATORY_COLS.has(id) || (colVisible[id] ?? colDefaultVisible(id));
  const toggleCol = (id: string) => setColVisible(prev => {
    const cur = prev[id] ?? colDefaultVisible(id);
    const next = { ...prev, [id]: !cur };
    saveColVisibility(next);
    return next;
  });

  // Columns the user can show/hide (Source only exists in the unfiltered view).
  const toggleableCols = useMemo(() => {
    const ids = BASE_COLS.filter(c => !MANDATORY_COLS.has(c.id)).map(c => ({ id: c.id, label: c.label }));
    if (showSourceBadge) ids.splice(1, 0, { id: 'source', label: 'Source' }); // after Style
    return ids;
  }, [showSourceBadge]);

  // Build column list based on current flags + visibility
  const columns = useMemo(() => {
    const cols: ColDef[] = [];
    if (selectionMode) cols.push(SELECT_COL);
    // Insert base cols, injecting source col after style if needed, and a
    // flexible spacer just before actions so actions hugs the right edge.
    for (const c of BASE_COLS) {
      if (c.id === 'actions') cols.push(SPACER_COL);
      if (!isColVisible(c.id)) continue;
      cols.push(c);
      if (c.id === 'style' && showSourceBadge && isColVisible('source')) cols.push(SOURCE_COL);
    }
    return cols;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionMode, showSourceBadge, colVisible]);

  // Persisted column widths
  const [colWidths, setColWidths] = useState<Record<string, number>>(() => {
    const saved = loadColWidths();
    const merged: Record<string, number> = {};
    for (const c of [...BASE_COLS, SOURCE_COL, SELECT_COL]) {
      merged[c.id] = saved[c.id] ?? c.defaultW;
    }
    return merged;
  });

  const getW = (id: string) => colWidths[id] ?? BASE_COLS.find(c => c.id === id)?.defaultW ?? 80;

  // Drag-resize handler
  const handleResizeStart = useCallback((colId: string, minW: number, startX: number) => {
    const startW = getW(colId);
    const onMove = (e: MouseEvent) => {
      const newW = Math.max(minW, startW + (e.clientX - startX));
      setColWidths(prev => {
        const next = { ...prev, [colId]: newW };
        saveColWidths(next);
        return next;
      });
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [colWidths]);

  const formatDuration = (val: string | number | undefined) => {
    if (!val) return '--:--';
    if (typeof val === 'string') return val;
    const m = Math.floor(val / 60);
    const s = Math.floor(val % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const formatDate = (d: string | Date | undefined) => {
    if (!d) return '';
    // SQLite datetime('now') stores UTC without 'Z' suffix — force UTC interpretation
    const date = typeof d === 'string'
      ? new Date(d.includes('T') || d.includes('Z') ? d : d.replace(' ', 'T') + 'Z')
      : d;
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffH = Math.floor(diffMs / 3600000);
    if (diffH < 1) return 'Just now';
    if (diffH < 24) return `${diffH}h ago`;
    const diffD = Math.floor(diffH / 24);
    if (diffD < 7) return `${diffD}d ago`;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  // Sum of fixed column widths (spacer is flexible, contributes nothing).
  // Used as the table's min width so it scrolls once the fixed columns no
  // longer fit; otherwise the table fills 100% and the spacer takes the slack.
  const totalW = columns.reduce((s, c) => s + (c.id === 'spacer' ? 0 : getW(c.id)), 0);

  return (
    <div>
      {/* Toolbar: Columns menu (left) + pagination / extras (right) */}
      <div className="flex items-center justify-between gap-3 mb-2">
        <div className="relative">
          <button
            onClick={() => setColMenuOpen(o => !o)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-white/10 text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-white/5 transition-colors"
            title="Choose columns"
          >
            <Columns3 size={13} /> Columns
          </button>
          {colMenuOpen && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setColMenuOpen(false)} />
              <div className="absolute left-0 top-full mt-1 z-50 min-w-[180px] max-h-80 overflow-y-auto py-1 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 shadow-xl">
                <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-400 dark:text-zinc-500">Show columns</div>
                {toggleableCols.map(c => (
                  <button
                    key={c.id}
                    onClick={() => toggleCol(c.id)}
                    className="w-full flex items-center gap-2.5 px-3 py-1.5 text-xs text-zinc-700 dark:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                  >
                    {isColVisible(c.id)
                      ? <CheckSquare size={14} className="text-pink-400 flex-shrink-0" />
                      : <Square size={14} className="text-zinc-500 flex-shrink-0" />}
                    {c.label}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
        {topRight && <div className="flex-shrink-0">{topRight}</div>}
      </div>
      <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-white/5">
      <table className="text-xs" style={{ tableLayout: 'fixed', width: '100%', minWidth: totalW }}>
        <colgroup>
          {columns.map(c => (
            <col key={c.id} style={{ width: c.id === 'spacer' ? 'auto' : getW(c.id) }} />
          ))}
        </colgroup>
        <thead>
          <tr className="bg-zinc-100/80 dark:bg-zinc-800/80 text-zinc-500 text-left">
            {columns.map(c => (
              <th
                key={c.id}
                className={`relative px-2 py-2.5 font-semibold select-none ${
                  c.align === 'center' ? 'text-center' : c.align === 'right' ? 'text-right' : ''
                } ${
                  // Actions is a frozen column — pinned to the right edge, always
                  // visible regardless of how the other columns are sized/scrolled (#58).
                  c.id === 'actions'
                    ? 'sticky right-0 z-20 bg-zinc-100 dark:bg-zinc-800 border-l border-zinc-200 dark:border-white/10'
                    : ''
                }`}
              >
                {c.label}
                {c.resizable && (
                  <div
                    className="absolute top-0 right-0 w-1.5 h-full cursor-col-resize z-10 group/resize hover:bg-pink-500/30"
                    onMouseDown={(e) => { e.preventDefault(); handleResizeStart(c.id, c.minW, e.clientX); }}
                  >
                    <div className="w-px h-full mx-auto bg-zinc-300 dark:bg-zinc-600 group-hover/resize:bg-pink-400 transition-colors" />
                  </div>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {songs.map(song => {
            const isActive = currentSongId === song.id;
            const gp = song.generationParams || song.generation_params as any;
            const bpm = song.bpm || gp?.bpm;
            const keyScale = song.key_scale || gp?.keyScale;
            const model = song.dit_model || gp?.ditModel || '';
            const modelShort = model ? model.split('/').pop()?.replace(/\.gguf$/, '').substring(0, 16) : '';
            const src = (gp as any)?.source || 'create';
            const badge = SOURCE_BADGE_MAP[src];
            const loraName = baseName(gp?.loraPath);
            const lmName = baseName(gp?.lmModel);
            const vaeName = baseName(gp?.vaeModel);
            const isMastered = !!(song.masteredAudioUrl || song.mastered_audio_url);

            // Build cell map
            const cells: Record<string, React.ReactNode> = {
              select: selectionMode ? (
                selectedIds.has(song.id)
                  ? <CheckSquare size={15} className="text-pink-400" />
                  : <Square size={15} className="text-zinc-600" />
              ) : null,

              thumb: (
                <div className="relative w-8 h-8 rounded bg-zinc-100 dark:bg-zinc-800 overflow-hidden flex items-center justify-center">
                  {song.coverUrl || song.cover_url ? (
                    <img src={song.coverUrl || song.cover_url} alt="" className="w-full h-full object-cover" />
                  ) : (
                    <Music size={12} className="text-zinc-600" />
                  )}
                  {!selectionMode && (
                    <button
                      onClick={(e) => { e.stopPropagation(); if (isActive) togglePlay(); else onPlay(song); }}
                      className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      {isActive && isPlaying
                        ? <Pause size={10} className="text-white" />
                        : <Play size={10} className="text-white ml-0.5" />
                      }
                    </button>
                  )}
                </div>
              ),

              title: <TableTitleCell song={song} isActive={isActive} onRename={onRename ? (newTitle) => onRename(song, newTitle) : undefined} disguiseTitle={disguiseTitle} />,
              style: <TableStyleCell text={isDisguised ? '—' : (song.style || song.caption || '—')} />,
              source: badge ? <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full border ${badge.cls}`}>{badge.label}</span> : null,
              bpm: <span className="text-zinc-500 font-mono">{bpm || '—'}</span>,
              key: <span className="text-zinc-500">{keyScale || '—'}</span>,
              model: <span className="text-violet-400/60 truncate block" title={model}>{modelShort || '—'}</span>,
              quality: <QualityBadge song={song} />,
              time: <span className="text-zinc-500 font-mono">{formatDuration(song.duration)}</span>,
              date: <span className="text-zinc-600">{formatDate(song.created_at || song.createdAt)}</span>,
              // ── Generation data (hidden by default) ──
              seed: <span className="text-zinc-500 font-mono truncate block">{gp?.seed ?? '—'}</span>,
              lmSeed: <span className="text-zinc-500 font-mono truncate block">{gp?.lmSeed ?? '—'}</span>,
              steps: <span className="text-zinc-500 font-mono">{gp?.inferenceSteps ?? '—'}</span>,
              cfg: <span className="text-zinc-500 font-mono">{gp?.guidanceScale ?? '—'}</span>,
              shift: <span className="text-zinc-500 font-mono">{gp?.shift ?? '—'}</span>,
              solver: <span className="text-zinc-500 truncate block" title={gp?.inferMethod || ''}>{gp?.inferMethod || '—'}</span>,
              scheduler: <span className="text-zinc-500 truncate block" title={gp?.scheduler || ''}>{gp?.scheduler || '—'}</span>,
              guidanceMode: <span className="text-zinc-500 truncate block" title={gp?.guidanceMode || ''}>{gp?.guidanceMode || '—'}</span>,
              lora: <span className="text-zinc-500 truncate block" title={gp?.loraPath || ''}>{loraName || '—'}</span>,
              loraScale: <span className="text-zinc-500 font-mono">{loraName && gp?.loraScale != null ? gp.loraScale : '—'}</span>,
              lm: <span className="text-zinc-500 truncate block" title={gp?.lmModel || ''}>{lmName || '—'}</span>,
              vae: <span className="text-zinc-500 truncate block" title={gp?.vaeModel || ''}>{vaeName || '—'}</span>,
              timesig: <span className="text-zinc-500 font-mono">{gp?.timeSignature || song.time_signature || '—'}</span>,
              lang: <span className="text-zinc-500">{gp?.vocalLanguage || '—'}</span>,
              task: <span className="text-zinc-500 truncate block" title={gp?.taskType || ''}>{gp?.taskType || '—'}</span>,
              instrumental: <span className="text-zinc-500">{gp?.instrumental ? 'Yes' : (gp?.instrumental === false ? 'No' : '—')}</span>,
              mastered: isMastered ? <Check size={13} className="text-emerald-400 inline" /> : <span className="text-zinc-600">—</span>,
              seedType: <span className="text-zinc-500">{gp?.randomSeed === true ? 'Random' : (gp?.randomSeed === false ? 'Fixed' : '—')}</span>,
              lmSeedType: <span className="text-zinc-500">{gp?.lmSeedFollowsDit === true ? 'Tied to DiT' : (gp?.lmSeedFollowsDit === false ? 'Fixed' : '—')}</span>,
              actions: (
                <div className="flex items-center justify-end gap-0.5 flex-nowrap flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                  {onReuse && (
                    <button onClick={(e) => { e.stopPropagation(); onReuse(song); }}
                      className="p-1 rounded text-zinc-500 hover:text-white hover:bg-white/10 transition-colors" title={t('library.edit')}>
                      <RotateCcw size={12} />
                    </button>
                  )}
                  {onDownload && (
                    <button onClick={(e) => { e.stopPropagation(); onDownload(song); }}
                      className="p-1 rounded text-zinc-500 hover:text-white hover:bg-white/10 transition-colors" title={t('library.download')}>
                      <Download size={12} />
                    </button>
                  )}
                  {onSendToCover && (
                    <button onClick={(e) => { e.stopPropagation(); onSendToCover(song); }}
                      className="p-1 rounded text-zinc-500 hover:text-cyan-400 hover:bg-cyan-500/10 transition-colors" title={t('library.sendToCover', 'Send to Cover Studio')}>
                      <Disc3 size={12} />
                    </button>
                  )}
                  <button onClick={(e) => { e.stopPropagation(); onDelete(song); }}
                    className="p-1 rounded text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors" title={t('library.delete')}>
                    <Trash2 size={12} />
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); setTrackA(songToTrack(song)); }}
                    className={`p-1 rounded transition-colors ${song.id === abTrackAId ? 'text-blue-400 bg-blue-500/10' : 'text-zinc-500 hover:text-blue-400 hover:bg-blue-500/10'}`}
                    title="Set as Track A">
                    <span className="text-[9px] font-bold">A</span>
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); setTrackB(songToTrack(song)); }}
                    className={`p-1 rounded transition-colors ${song.id === abTrackBId ? 'text-orange-400 bg-orange-500/10' : 'text-zinc-500 hover:text-orange-400 hover:bg-orange-500/10'}`}
                    title="Set as Track B">
                    <span className="text-[9px] font-bold">B</span>
                  </button>
                </div>
              ),
            };

            return (
              <tr
                key={song.id}
                className={`group border-t border-zinc-100 dark:border-white/5 cursor-pointer transition-colors ${
                  selectedIds.has(song.id) && selectionMode
                    ? 'bg-pink-500/10'
                    : isActive
                      ? 'bg-pink-500/5'
                      : 'hover:bg-white/5'
                }`}
                onClick={() => selectionMode ? onToggleSelect(song.id) : onSelect?.(song)}
              >
                {columns.map(c => (
                  <td
                    key={c.id}
                    className={`px-2 py-2 overflow-hidden ${
                      c.align === 'center' ? 'text-center' : c.align === 'right' ? 'text-right' : ''
                    } ${
                      // Frozen actions column — opaque bg (matches the transparent
                      // default row) so scrolled cells slide cleanly underneath (#58).
                      c.id === 'actions'
                        ? 'sticky right-0 z-10 bg-white dark:bg-zinc-950 group-hover:bg-zinc-50 dark:group-hover:bg-zinc-900 border-l border-zinc-200 dark:border-white/10'
                        : ''
                    }`}
                  >
                    {cells[c.id]}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    </div>
  );
};


