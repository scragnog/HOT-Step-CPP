import React, { useState, useEffect, useRef } from 'react';
import { Disc3, Plus, FileText, MoreVertical, RefreshCw, Link2, Trash2, Search, PenLine, ChevronDown, Sparkles } from 'lucide-react';
import type { LyricsSet, SongLyric } from '../../services/lireekApi';

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

interface AlbumGridProps {
  albums: LyricsSet[];
  loading: boolean;
  artistName: string;
  onSelectAlbum: (album: LyricsSet) => void;
  onAddAlbum: () => void;
  onAddManual: () => void;
  onDeleteAlbum: (album: LyricsSet) => void;
  onRefreshImage?: (album: LyricsSet) => void;
  onSetImage?: (album: LyricsSet, url: string) => void;
  onCuratedProfile?: () => void;
}

export const AlbumGrid: React.FC<AlbumGridProps> = ({
  albums, loading, artistName, onSelectAlbum, onAddAlbum, onAddManual, onDeleteAlbum, onRefreshImage, onSetImage, onCuratedProfile,
}) => {
  const [imageErrors, setImageErrors] = useState<Set<number>>(new Set());
  const [menuOpenId, setMenuOpenId] = useState<number | null>(null);
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
    const hash = (name || 'album').split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
    const h1 = Math.abs(hash) % 360;
    const h2 = (h1 + 30) % 360;
    return `linear-gradient(135deg, hsl(${h1}, 50%, 25%), hsl(${h2}, 40%, 18%))`;
  };

  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 p-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="aspect-square rounded-2xl bg-white/5 animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-white">Albums</h2>
          <p className="text-sm text-zinc-400 mt-0.5">{artistName}</p>
        </div>
        <div className="flex items-center gap-3">
          {onCuratedProfile && albums.length >= 2 && (
            <button
              onClick={onCuratedProfile}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-purple-600/80 hover:bg-purple-500 text-white text-sm font-semibold transition-all hover:scale-105 shadow-lg shadow-purple-600/20"
            >
              <Sparkles className="w-4 h-4" />
              Curated Profile
            </button>
          )}
          <div className="relative" ref={addBtnRef}>
            <button
              onClick={() => setAddMenuOpen(!addMenuOpen)}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-semibold transition-all hover:scale-105 shadow-lg shadow-pink-600/20"
            >
              <Plus className="w-4 h-4" />
              Add Album
              <ChevronDown className="w-3.5 h-3.5 opacity-70" />
            </button>
            {addMenuOpen && (
              <div className="absolute right-0 top-full mt-1 z-30 min-w-[180px] rounded-xl bg-zinc-900 border border-white/10 shadow-2xl py-1 animate-in fade-in slide-in-from-top-1">
                <button
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                  onClick={() => { setAddMenuOpen(false); onAddAlbum(); }}
                >
                  <Search className="w-3.5 h-3.5" />
                  Fetch from Genius
                </button>
                <button
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                  onClick={() => { setAddMenuOpen(false); onAddManual(); }}
                >
                  <PenLine className="w-3.5 h-3.5" />
                  Add Manually
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {albums.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Disc3 className="w-8 h-8 text-zinc-600" />
          </div>
          <h3 className="text-base font-semibold text-zinc-400 mb-2">No albums yet</h3>
          <p className="text-sm text-zinc-500 max-w-xs mb-4">
            Fetch lyrics for an album or add one manually.
          </p>
          <div className="flex items-center gap-3">
            <button
              onClick={onAddAlbum}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-semibold transition-all"
            >
              <Search className="w-4 h-4" />
              Fetch from Genius
            </button>
            <button
              onClick={onAddManual}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-zinc-300 hover:text-white text-sm font-semibold transition-all"
            >
              <PenLine className="w-4 h-4" />
              Add Manually
            </button>
          </div>
        </div>
      ) : (
        <div ref={gridRef} className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {albums.map((album, idx) => {
            const songCount = album.total_songs ?? (album.songs ? parseSongs(album.songs).length : 0);
            return (
              <div
                key={album.id}
                className={`group relative aspect-square rounded-2xl overflow-hidden cursor-pointer transition-all duration-300 hover:scale-[1.03] hover:shadow-2xl hover:shadow-indigo-500/10 ls2-card-in ls2-stagger-${Math.min(idx + 1, 11)}`}
                onClick={() => onSelectAlbum(album)}
              >
                {/* Album cover art or gradient */}
                {album.image_url && !imageErrors.has(album.id) ? (
                  <img
                    src={album.image_url}
                    alt={album.album || 'Album'}
                    className="absolute inset-0 w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                    onError={() => setImageErrors(prev => new Set(prev).add(album.id))}
                  />
                ) : (
                  <>
                    <div
                      className="absolute inset-0"
                      style={{ background: gradient(album.album || String(album.id)) }}
                    />
                    {/* Decorative vinyl record */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60%] h-[60%] rounded-full border border-white/5 opacity-20">
                      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[30%] h-[30%] rounded-full border border-white/10" />
                    </div>
                  </>
                )}

                {/* Overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />

                {/* Content */}
                <div className="absolute inset-x-0 bottom-0 p-4">
                  <h3 className="text-sm font-bold text-white truncate mb-2 drop-shadow-lg">
                    {album.album || 'Top Songs'}
                  </h3>
                  <div className="flex items-center gap-3 text-xs text-zinc-300/70">
                    <span className="flex items-center gap-1">
                      <FileText className="w-3 h-3" />
                      {songCount} songs
                    </span>
                  </div>
                </div>

                {/* Hover ring */}
                <div className="absolute inset-0 rounded-2xl ring-1 ring-white/10 group-hover:ring-indigo-500/40 transition-all duration-300" />

                {/* Context menu button */}
                <button
                  className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/50 text-white/60 hover:text-white hover:bg-black/70 opacity-0 group-hover:opacity-100 transition-all z-10"
                  onClick={(e) => {
                    e.stopPropagation();
                    setMenuOpenId(menuOpenId === album.id ? null : album.id);
                  }}
                >
                  <MoreVertical className="w-4 h-4" />
                </button>

                {/* Context menu */}
                {menuOpenId === album.id && (
                  <div
                    className="absolute top-10 right-2 z-20 min-w-[160px] rounded-xl bg-zinc-900 border border-white/10 shadow-2xl py-1 animate-in fade-in slide-in-from-top-1"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {onRefreshImage && (
                      <button
                        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                        onClick={() => { onRefreshImage(album); setMenuOpenId(null); }}
                      >
                        <RefreshCw className="w-3.5 h-3.5" />
                        Re-fetch from Genius
                      </button>
                    )}
                    {onSetImage && (
                      <button
                        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 hover:text-white transition-colors"
                        onClick={() => {
                          setMenuOpenId(null);
                          const url = prompt(`Paste an image URL for "${album.album || 'Album'}":`, album.image_url || '');
                          if (url && url.trim()) onSetImage(album, url.trim());
                        }}
                      >
                        <Link2 className="w-3.5 h-3.5" />
                        Set Custom Image
                      </button>
                    )}
                    <div className="border-t border-white/5 my-1" />
                    <button
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-colors"
                      onClick={() => { onDeleteAlbum(album); setMenuOpenId(null); }}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete Album
                    </button>
                  </div>
                )}
              </div>
            );
          })}

          {/* Add new album card — dropdown */}
          <div
            className="relative aspect-square rounded-2xl border-2 border-dashed border-white/10 hover:border-pink-500/30 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 hover:bg-white/[0.02] group"
            onClick={() => setAddMenuOpen(!addMenuOpen)}
          >
            <div className="w-10 h-10 rounded-full bg-white/5 group-hover:bg-pink-500/10 flex items-center justify-center mb-2 transition-colors">
              <Plus className="w-5 h-5 text-zinc-500 group-hover:text-pink-400 transition-colors" />
            </div>
            <span className="text-xs text-zinc-500 group-hover:text-zinc-300 font-medium transition-colors">
              Add Album
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
