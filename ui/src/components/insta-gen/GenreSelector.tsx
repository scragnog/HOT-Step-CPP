// GenreSelector.tsx — Multi-select genre dropdown with categorised groups
//
// Combobox-style: selected genres appear as removable chips above a
// searchable, categorised dropdown list.

import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { ChevronDown, X, Search } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { GENRE_TAXONOMY, ALL_GENRES } from '../../data/genres';

interface GenreSelectorProps {
  selected: string[];
  onChange: (genres: string[]) => void;
}

export const GenreSelector: React.FC<GenreSelectorProps> = ({ selected, onChange }) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Focus search on open
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => searchRef.current?.focus(), 50);
    } else {
      setSearch('');
    }
  }, [isOpen]);

  const toggle = useCallback((genre: string) => {
    if (selected.includes(genre)) {
      onChange(selected.filter(g => g !== genre));
    } else {
      onChange([...selected, genre]);
    }
  }, [selected, onChange]);

  const clearAll = useCallback(() => {
    onChange([]);
  }, [onChange]);

  const randomize = useCallback(() => {
    // Pick 2-4 random genres from the full list
    const count = 2 + Math.floor(Math.random() * 3); // 2, 3, or 4
    const shuffled = [...ALL_GENRES].sort(() => Math.random() - 0.5);
    onChange(shuffled.slice(0, count));
  }, [onChange]);

  // Filtered taxonomy based on search
  const filteredTaxonomy = useMemo(() => {
    if (!search.trim()) return GENRE_TAXONOMY;
    const q = search.toLowerCase();
    return GENRE_TAXONOMY
      .map(cat => ({
        ...cat,
        genres: cat.genres.filter(g => g.toLowerCase().includes(q)),
      }))
      .filter(cat => cat.genres.length > 0);
  }, [search]);

  return (
    <div ref={containerRef} className="relative">
      {/* Label */}
      <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400 mb-1.5">
        {t('instaGen.genreLabel')}
      </label>

      {/* Selected chips + trigger */}
      <div
        className="min-h-[42px] w-full rounded-xl border border-zinc-300 dark:border-white/10 bg-zinc-50 dark:bg-white/5 px-3 py-2 cursor-pointer hover:border-pink-400/50 transition-colors flex flex-wrap items-center gap-1.5"
        onClick={() => setIsOpen(!isOpen)}
      >
        {selected.length === 0 && (
          <span className="text-zinc-400 dark:text-zinc-500 text-sm select-none">
            {t('instaGen.genrePlaceholder')}
          </span>
        )}
        {selected.map(genre => (
          <span
            key={genre}
            className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-pink-500/15 text-pink-400 border border-pink-500/20 hover:bg-pink-500/25 transition-colors"
          >
            {genre}
            <button
              onClick={(e) => { e.stopPropagation(); toggle(genre); }}
              className="hover:text-pink-200 transition-colors"
            >
              <X size={12} />
            </button>
          </span>
        ))}
        <div className="ml-auto flex items-center gap-1.5 flex-shrink-0">
          <button
            onClick={(e) => { e.stopPropagation(); randomize(); }}
            className="text-xs text-zinc-400 hover:text-violet-400 transition-colors"
            title="Random genres"
          >
            Random
          </button>
          {selected.length > 0 && (
            <button
              onClick={(e) => { e.stopPropagation(); clearAll(); }}
              className="text-xs text-zinc-400 hover:text-red-400 transition-colors"
              title={t('instaGen.clearAll')}
            >
              {t('instaGen.clearAll')}
            </button>
          )}
          <ChevronDown
            size={16}
            className={`text-zinc-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
          />
        </div>
      </div>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 mt-1 w-full max-h-[380px] rounded-xl border border-zinc-300 dark:border-white/10 bg-white dark:bg-zinc-900 shadow-2xl overflow-hidden flex flex-col animate-in fade-in slide-in-from-top-1 duration-150">
          {/* Search input */}
          <div className="flex items-center gap-2 px-3 py-2.5 border-b border-zinc-200 dark:border-white/5">
            <Search size={14} className="text-zinc-400 flex-shrink-0" />
            <input
              ref={searchRef}
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t('instaGen.genrePlaceholder')}
              className="flex-1 bg-transparent text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 dark:placeholder:text-zinc-500 outline-none"
              onKeyDown={(e) => {
                if (e.key === 'Escape') setIsOpen(false);
              }}
            />
          </div>

          {/* Category list */}
          <div className="flex-1 overflow-y-auto py-1">
            {filteredTaxonomy.length === 0 && (
              <div className="px-4 py-6 text-center text-sm text-zinc-400">
                No genres match &ldquo;{search}&rdquo;
              </div>
            )}
            {filteredTaxonomy.map(category => (
              <div key={category.name}>
                {/* Category header */}
                <div className="px-3 py-1.5 text-xs font-semibold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider sticky top-0 bg-white dark:bg-zinc-900">
                  {category.icon} {category.name}
                </div>
                {/* Genre items */}
                <div className="px-2 pb-1 flex flex-wrap gap-1">
                  {category.genres.map(genre => {
                    const isSelected = selected.includes(genre);
                    return (
                      <button
                        key={genre}
                        onClick={() => toggle(genre)}
                        className={`
                          px-2.5 py-1 rounded-lg text-xs font-medium transition-all duration-150
                          ${isSelected
                            ? 'bg-pink-500/20 text-pink-400 border border-pink-500/30 shadow-sm shadow-pink-500/10'
                            : 'bg-zinc-100 dark:bg-white/5 text-zinc-600 dark:text-zinc-300 border border-transparent hover:bg-zinc-200 dark:hover:bg-white/10 hover:text-zinc-900 dark:hover:text-white'
                          }
                        `}
                      >
                        {genre}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
