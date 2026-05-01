import React, { useRef, useEffect, useCallback } from 'react';
import { ChevronLeft } from 'lucide-react';
import type { Artist } from '../../services/lireekApi';

interface ArtistSidebarProps {
  artists: Artist[];
  selectedArtistId: number;
  onSelectArtist: (artist: Artist) => void;
  onBack: () => void;
  artistIdsWithAdapters?: Set<number>;
}

const SCROLL_KEY = 'ls-artist-sidebar-scroll';

export const ArtistSidebar: React.FC<ArtistSidebarProps> = ({
  artists, selectedArtistId, onSelectArtist, onBack,
  artistIdsWithAdapters,
}) => {
  const [imageErrors, setImageErrors] = React.useState<Set<number>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

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
    <div className="flex flex-col h-full bg-zinc-950/50">
      {/* Back button */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 px-4 py-3 text-sm text-zinc-400 hover:text-white hover:bg-white/5 border-b border-white/5 transition-colors"
      >
        <ChevronLeft className="w-4 h-4" />
        All Artists
      </button>

      {/* Artist list */}
      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto py-2">
        {artists.map((artist) => {
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
                {artist.image_url && !imageErrors.has(artist.id) ? (
                  <img
                    src={artist.image_url}
                    alt={artist.name}
                    className="w-full h-full object-cover"
                    onError={() => setImageErrors(prev => new Set(prev).add(artist.id))}
                  />
                ) : (
                  <div
                    className="w-full h-full flex items-center justify-center text-xs font-bold text-white/60"
                    style={{ backgroundColor: gradient(artist.name) }}
                  >
                    {artist.name.charAt(0).toUpperCase()}
                  </div>
                )}
              </div>

              <div className="min-w-0 flex-1">
                <p className={`text-sm font-medium truncate ${isSelected ? 'text-pink-400' : 'text-zinc-300'}`}>
                  {artist.name}
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
      <div className="px-4 py-2 border-t border-white/5 text-[10px] text-zinc-600 text-center">
        {artists.length} artist{artists.length !== 1 ? 's' : ''}
      </div>
    </div>
  );
};
