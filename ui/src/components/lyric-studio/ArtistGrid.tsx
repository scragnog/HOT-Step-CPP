import React, { useState, useEffect, useRef } from 'react';
import { Music, Plus, RefreshCw, Trash2, MoreVertical, Link2, Search, PenLine, ChevronDown } from 'lucide-react';
import type { Artist } from '../../services/lireekApi';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';

interface ArtistGridProps {
  artists: Artist[];
  loading: boolean;
  onSelectArtist: (artist: Artist) => void;
  onAddNew: () => void;
  onAddManual: () => void;
  onDelete: (artist: Artist) => void;
  onRefreshImage: (artist: Artist) => void;
  onSetImage?: (artist: Artist, url: string) => void;
}

export const ArtistGrid: React.FC<ArtistGridProps> = ({
  artists, loading, onSelectArtist, onAddNew, onAddManual, onDelete, onRefreshImage, onSetImage,
}) => {
  const { disguiseArtist, disguiseImageUrl } = useDisguiseMode();
  const [menuOpenId, setMenuOpenId] = useState<number | null>(null);
  const [imageErrors, setImageErrors] = useState<Set<number>>(new Set());
  const [addMenuOpen, setAddMenuOpen] = useState(false);
  const gridRef = useRef<HTMLDivElement>(null);
  const addBtnRef = useRef<HTMLDivElement>(null);

  // Close context menus on click outside
  useEffect(() => {
    if (menuOpenId === null && !addMenuOpen) return;
    const handler = (e: MouseEvent) => {
      if (gridRef.current && !gridRef.current.contains(e.target as Node)) {
        setMenuOpenId(null);
      }
      if (addBtnRef.current && !addBtnRef.current.contains(e.target as Node)) {
        setAddMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [menuOpenId, addMenuOpen]);

  const gradient = (name: string) => {
    const hash = name.split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
    const h1 = Math.abs(hash) % 360;
    const h2 = (h1 + 40) % 360;
    return `linear-gradient(135deg, hsl(${h1}, 70%, 35%), hsl(${h2}, 60%, 25%))`;
  };

  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 p-6">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="aspect-[3/4] rounded-2xl bg-white/5 animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Music className="w-7 h-7 text-pink-400" />
          Lyric Studio
        </h1>
        <div className="relative" ref={addBtnRef}>
          <button
            onClick={() => setAddMenuOpen(!addMenuOpen)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-semibold transition-all hover:scale-105 shadow-lg shadow-pink-600/20"
          >
            <Plus className="w-4 h-4" />
            Add Artist
            <ChevronDown className="w-3.5 h-3.5 opacity-70" />
          </button>
          {addMenuOpen && (
            <div className="absolute right-0 top-full mt-1 z-30 min-w-[180px] rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 shadow-2xl py-1 animate-in fade-in slide-in-from-top-1">
              <button
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                onClick={() => { setAddMenuOpen(false); onAddNew(); }}
              >
                <Search className="w-3.5 h-3.5" />
                Fetch from Genius
              </button>
              <button
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                onClick={() => { setAddMenuOpen(false); onAddManual(); }}
              >
                <PenLine className="w-3.5 h-3.5" />
                Add Manually
              </button>
            </div>
          )}
        </div>
      </div>

      {artists.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Music className="w-10 h-10 text-zinc-600" />
          </div>
          <h2 className="text-lg font-semibold text-zinc-600 dark:text-zinc-400 mb-2">No artists yet</h2>
          <p className="text-sm text-zinc-500 max-w-sm mb-6">
            Start by fetching lyrics from Genius or adding an artist manually.
          </p>
          <div className="flex items-center gap-3">
            <button
              onClick={onAddNew}
              className="flex items-center gap-2 px-5 py-3 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-semibold transition-all"
            >
              <Search className="w-4 h-4" />
              Fetch from Genius
            </button>
            <button
              onClick={onAddManual}
              className="flex items-center gap-2 px-5 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-zinc-300 dark:border-white/10 text-zinc-700 dark:text-zinc-300 hover:text-white text-sm font-semibold transition-all"
            >
              <PenLine className="w-4 h-4" />
              Add Manually
            </button>
          </div>
        </div>
      ) : (
        <div ref={gridRef} className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
          {artists.map((artist, idx) => (
            <div
              key={artist.id}
              className={`group relative aspect-[3/4] rounded-2xl cursor-pointer transition-all duration-300 hover:scale-[1.03] hover:shadow-2xl hover:shadow-pink-500/10 ls2-card-in ls2-stagger-${Math.min(idx + 1, 11)} ${menuOpenId === artist.id ? 'z-30' : ''}`}
              onClick={() => onSelectArtist(artist)}
            >
              {/* Image clip wrapper — overflow-hidden here so the context menu can extend beyond the card */}
              <div className="absolute inset-0 rounded-2xl overflow-hidden">
                {/* Background image or gradient */}
                {(() => {
                  const dUrl = disguiseImageUrl(artist.image_url, artist.name);
                  const dName = disguiseArtist(artist.name);
                  return dUrl && !imageErrors.has(artist.id) ? (
                  <img
                    src={dUrl}
                    alt={dName}
                    className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                    onError={() => setImageErrors(prev => new Set(prev).add(artist.id))}
                  />
                ) : (
                  <div
                    className="absolute inset-0 flex items-center justify-center"
                    style={{ background: gradient(dName) }}
                  >
                    <span className="text-5xl font-black text-white/20 select-none">
                      {dName.charAt(0).toUpperCase()}
                    </span>
                  </div>
                );
                })()}

                {/* Dark overlay gradient */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/30 to-transparent" />
              </div>

              {/* Content */}
              <div className="absolute inset-x-0 bottom-0 p-4">
                <h3 className="text-base font-bold text-white truncate mb-1 drop-shadow-lg">
                  {disguiseArtist(artist.name)}
                </h3>
                <p className="text-xs text-zinc-700 dark:text-zinc-300/80">
                  {artist.lyrics_set_count ?? 0} album{(artist.lyrics_set_count ?? 0) !== 1 ? 's' : ''}
                </p>
              </div>

              {/* Hover glow ring */}
              <div className="absolute inset-0 rounded-2xl ring-1 ring-white/10 group-hover:ring-pink-500/40 transition-all duration-300" />

              {/* Context menu button */}
              <button
                className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/20 dark:bg-black/50 text-white/60 hover:text-white hover:bg-black/20 dark:bg-black/40 dark:bg-black/70 opacity-0 group-hover:opacity-100 transition-all z-10"
                onClick={(e) => {
                  e.stopPropagation();
                  setMenuOpenId(menuOpenId === artist.id ? null : artist.id);
                }}
              >
                <MoreVertical className="w-4 h-4" />
              </button>

              {/* Context menu */}
              {menuOpenId === artist.id && (
                <div
                  className="absolute top-10 right-2 z-20 min-w-[160px] rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 shadow-2xl py-1 animate-in fade-in slide-in-from-top-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                    onClick={() => { onRefreshImage(artist); setMenuOpenId(null); }}
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Re-fetch from Genius
                  </button>
                  {onSetImage && (
                    <button
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                      onClick={() => {
                        setMenuOpenId(null);
                        const url = prompt(`Paste an image URL for ${artist.name}:`, artist.image_url || '');
                        if (url && url.trim()) onSetImage(artist, url.trim());
                      }}
                    >
                      <Link2 className="w-3.5 h-3.5" />
                      Set Custom Image
                    </button>
                  )}
                  <div className="border-t border-zinc-200 dark:border-white/5 my-1" />
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-colors"
                    onClick={() => { onDelete(artist); setMenuOpenId(null); }}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Delete Artist
                  </button>
                </div>
              )}
            </div>
          ))}

          {/* Add new card — dropdown */}
          <div
            className="relative aspect-[3/4] rounded-2xl border-2 border-dashed border-zinc-300 dark:border-white/10 hover:border-pink-500/30 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 hover:bg-white/[0.02] group"
            onClick={() => setAddMenuOpen(!addMenuOpen)}
          >
            <div className="w-12 h-12 rounded-full bg-white/5 group-hover:bg-pink-500/10 flex items-center justify-center mb-3 transition-colors">
              <Plus className="w-6 h-6 text-zinc-500 group-hover:text-pink-400 transition-colors" />
            </div>
            <span className="text-sm text-zinc-500 group-hover:text-zinc-700 dark:text-zinc-300 font-medium transition-colors">
              Add Artist
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
