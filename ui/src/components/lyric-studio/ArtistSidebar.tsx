import React, { useRef, useEffect, useCallback, useMemo, useState } from 'react';
import { ChevronLeft, Search, X } from 'lucide-react';
import type { Artist } from '../../services/lireekApi';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';

interface ArtistSidebarProps {
  artists: Artist[];
  selectedArtistId: number;
  onSelectArtist: (artist: Artist) => void;
  onBack: () => void;
  artistIdsWithAdapters?: Set<number>;
}

const SCROLL_KEY = 'ls-artist-sidebar-scroll';
const FILTER_KEY = 'ls-artist-sidebar-filter';

/** Lowercase + strip diacritics, so "Motorhead" finds "Motörhead". */
const normalize = (s: string) =>
  s.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase();

/** Sort key: a leading "The " is ignored, so "The Beach Boys" files under B. */
const sortKey = (name: string) => normalize(name).replace(/^the\s+/, '').trim();

export const ArtistSidebar: React.FC<ArtistSidebarProps> = ({
  artists, selectedArtistId, onSelectArtist, onBack,
  artistIdsWithAdapters,
}) => {
  const { disguiseArtist, disguiseImageUrl } = useDisguiseMode();
  const [imageErrors, setImageErrors] = React.useState<Set<number>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Filter text survives navigating into an album and back (the sidebar remounts)
  const [query, setQuery] = useState(() => {
    try { return sessionStorage.getItem(FILTER_KEY) ?? ''; } catch { return ''; }
  });

  // Sort + filter run on the REAL names — disguise mode is display-only
  const visibleArtists = useMemo(() => {
    const sorted = [...artists].sort((a, b) =>
      sortKey(a.name).localeCompare(sortKey(b.name), undefined, { numeric: true })
    );
    const q = normalize(query.trim());
    if (!q) return sorted;
    return sorted.filter(a => {
      const n = normalize(a.name);
      return n.includes(q) || n.replace(/^the\s+/, '').includes(q);
    });
  }, [artists, query]);

  const setFilter = useCallback((value: string) => {
    setQuery(value);
    try { sessionStorage.setItem(FILTER_KEY, value); } catch { /* ignore */ }
    // Jump back to the top of the (now different) list
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, []);

  // Restore scroll position on mount
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    try {
      const saved = sessionStorage.getItem(SCROLL_KEY);
      if (saved) el.scrollTop = Number(saved);
    } catch { /* ignore */ }
  }, []);

  // Debounced save on scroll
  const handleScroll = useCallback(() => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      try {
        const el = scrollRef.current;
        if (el) sessionStorage.setItem(SCROLL_KEY, String(el.scrollTop));
      } catch { /* ignore */ }
    }, 150);
  }, []);

  const gradient = (name: string) => {
    const hash = name.split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
    const h1 = Math.abs(hash) % 360;
    return `hsl(${h1}, 50%, 30%)`;
  };

  return (
    <div className="flex flex-col h-full bg-zinc-50 dark:bg-zinc-950/50">
      {/* Back button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 px-4 py-3 text-sm text-zinc-600 dark:text-zinc-400 hover:text-white hover:bg-white/5 border-b border-zinc-200 dark:border-white/5 transition-colors"
      >
        <ChevronLeft className="w-4 h-4" />
        All Artists
      </button>

      {/* Name filter */}
      <div className="px-3 py-2 border-b border-zinc-200 dark:border-white/5">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-500 pointer-events-none" />
          <input
            type="text"
            value={query}
            onChange={(e) => setFilter(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') { e.stopPropagation(); setFilter(''); } }}
            placeholder="Filter artists..."
            className="w-full pl-8 pr-7 py-1.5 text-xs rounded-md bg-white dark:bg-white/5 border border-zinc-200 dark:border-white/10 text-zinc-700 dark:text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-pink-500/50"
          />
          {query && (
            <button
              onClick={() => setFilter('')}
              title="Clear filter"
              className="absolute right-1.5 top-1/2 -translate-y-1/2 p-0.5 rounded text-zinc-500 hover:text-zinc-300 hover:bg-white/10"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      {/* Artist list */}
      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto py-2">
        {visibleArtists.length === 0 && (
          <p className="px-4 py-6 text-xs text-zinc-500 text-center">
            No artists match that filter
          </p>
        )}
        {visibleArtists.map((artist) => {
          const isSelected = artist.id === selectedArtistId;
          const hasAdapter = !artistIdsWithAdapters || artistIdsWithAdapters.size === 0 || artistIdsWithAdapters.has(artist.id);
          return (
            <button
              key={artist.id}
              className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-all ${
                isSelected
                  ? 'bg-pink-500/10 border-l-2 border-pink-500'
                  : 'hover:bg-white/5 border-l-2 border-transparent'
              }`}
              style={!isSelected && !hasAdapter ? { backgroundColor: 'rgba(220, 38, 38, 0.08)' } : undefined}
              onClick={() => onSelectArtist(artist)}
            >
              {/* Mini avatar */}
              <div className="w-8 h-8 flex-shrink-0 rounded-lg overflow-hidden">
                {(() => {
                  const dUrl = disguiseImageUrl(artist.image_url, artist.name);
                  const dName = disguiseArtist(artist.name);
                  return dUrl && !imageErrors.has(artist.id) ? (
                  <img
                    src={dUrl}
                    alt={dName}
                    className="w-full h-full object-cover"
                    onError={() => setImageErrors(prev => new Set(prev).add(artist.id))}
                  />
                ) : (
                  <div
                    className="w-full h-full flex items-center justify-center text-xs font-bold text-white/60"
                    style={{ backgroundColor: gradient(dName) }}
                  >
                    {dName.charAt(0).toUpperCase()}
                  </div>
                );
                })()}
              </div>

              <div className="min-w-0 flex-1">
                <p className={`text-sm font-medium truncate ${isSelected ? 'text-pink-400' : 'text-zinc-700 dark:text-zinc-300'}`}>
                  {disguiseArtist(artist.name)}
                </p>
                <p className="text-[11px] text-zinc-500">
                  {artist.lyrics_set_count ?? 0} album{(artist.lyrics_set_count ?? 0) !== 1 ? 's' : ''}
                </p>
              </div>
            </button>
          );
        })}
      </div>

      {/* Artist count */}
      <div className="px-4 py-2 border-t border-zinc-200 dark:border-white/5 text-[10px] text-zinc-600 text-center">
        {query.trim()
          ? `${visibleArtists.length} of ${artists.length} artists`
          : `${artists.length} artist${artists.length !== 1 ? 's' : ''}`}
      </div>
    </div>
  );
};
