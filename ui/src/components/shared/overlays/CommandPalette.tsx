// CommandPalette.tsx: Ctrl/Cmd+K quick jump to studios and songs.
// Studios and the already-loaded library songs are ranked locally with
// utils/fuzzy.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Search, LayoutGrid, Music } from 'lucide-react';
import { fuzzyRank } from '@/utils/fuzzy';
import type { Song } from '@/types';

/** Event any component can dispatch to open the palette (mobile search icon). */
export const OPEN_PALETTE_EVENT = 'hs-open-palette';

/** Views from the sidebar, with their i18n label keys. */
export const PALETTE_VIEWS: ReadonlyArray<{ view: string; labelKey: string }> = [
  { view: 'insta-gen', labelKey: 'sidebar.instaGen' },
  { view: 'create', labelKey: 'sidebar.create' },
  { view: 'library', labelKey: 'sidebar.library' },
  { view: 'lyric-studio', labelKey: 'sidebar.lyricStudio' },
  { view: 'cover-studio', labelKey: 'sidebar.coverStudio' },
  { view: 'repaint', labelKey: 'sidebar.repaint' },
  { view: 'stem-studio', labelKey: 'sidebar.stemStudio' },
  { view: 'compare-studio', labelKey: 'sidebar.compare' },
  { view: 'stem-builder', labelKey: 'sidebar.stemBuilder' },
  { view: 'song-builder', labelKey: 'sidebar.songBuilder' },
  { view: 'storm', labelKey: 'sidebar.storm' },
  { view: 'midi-studio', labelKey: 'sidebar.midiStudio' },
  { view: 'training-studio', labelKey: 'sidebar.trainingStudio' },
  { view: 'settings', labelKey: 'sidebar.settings' },
];

type Item =
  | { kind: 'view'; key: string; label: string; view: string }
  | { kind: 'song'; key: string; label: string; sub: string; song: Song };

interface Props {
  onClose: () => void;
  onNavigate: (view: string) => void;
  onPickSong: (songId: string) => void;
  songs: Song[];
}

export const CommandPalette: React.FC<Props> = ({ onClose, onNavigate, onPickSong, songs }) => {
  const { t } = useTranslation();
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const items: Item[] = useMemo(() => {
    const views: Item[] = PALETTE_VIEWS.map(v => ({ kind: 'view', key: `v:${v.view}`, label: t(v.labelKey), view: v.view }));
    const rankedViews = fuzzyRank(views, query, i => i.label);
    if (!query.trim()) return rankedViews;
    const songItems: Item[] = songs.map(s => ({
      kind: 'song', key: `s:${s.id}`, label: s.title || 'Untitled', sub: s.style || '', song: s,
    }));
    return [...rankedViews, ...fuzzyRank(songItems, query, i => i.label).slice(0, 20)];
  }, [query, songs, t]);

  useEffect(() => { setActive(0); }, [query, songs]);
  useEffect(() => {
    listRef.current?.querySelector(`[data-idx="${active}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [active]);

  const choose = (item: Item | undefined) => {
    if (!item) return;
    onClose();
    if (item.kind === 'view') onNavigate(item.view);
    else onPickSong(item.song.id);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(a => Math.min(items.length - 1, a + 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(a => Math.max(0, a - 1)); }
    else if (e.key === 'Enter') { e.preventDefault(); choose(items[active]); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); onClose(); }
  };

  return (
    <div className="fixed inset-0 z-[210] flex items-start justify-center bg-black/60 pt-[max(10vh,env(safe-area-inset-top))] px-3" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={t('palette.title')}
        className="w-[560px] max-w-full rounded-xl bg-white dark:bg-suno-panel border border-zinc-200 dark:border-zinc-800 shadow-2xl overflow-hidden"
        onClick={e => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="flex items-center gap-2 px-3 border-b border-zinc-200 dark:border-zinc-800">
          <Search size={16} className="text-zinc-500 flex-shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder={t('palette.placeholder')}
            aria-label={t('palette.placeholder')}
            aria-controls="hs-palette-list"
            aria-activedescendant={items[active] ? `hs-pal-${active}` : undefined}
            className="flex-1 bg-transparent py-3 text-sm outline-none"
          />
          <kbd className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 font-mono text-zinc-500">Esc</kbd>
        </div>
        <ul id="hs-palette-list" ref={listRef} role="listbox" className="max-h-[50vh] overflow-y-auto py-1">
          {items.length === 0 && (
            <li className="px-4 py-3 text-xs text-zinc-500">{t('palette.empty')}</li>
          )}
          {items.map((item, idx) => (
            <li
              key={item.key}
              id={`hs-pal-${idx}`}
              data-idx={idx}
              role="option"
              aria-selected={idx === active}
              onMouseMove={() => setActive(idx)}
              onClick={() => choose(item)}
              className={`flex items-center gap-3 px-4 py-2.5 cursor-pointer text-sm ${idx === active ? 'bg-pink-500/10 text-pink-500 dark:text-pink-300' : ''}`}
            >
              {item.kind === 'view' ? <LayoutGrid size={14} className="flex-shrink-0" /> : <Music size={14} className="flex-shrink-0" />}
              <span className="truncate">{item.label}</span>
              {item.kind === 'song' && item.sub && <span className="ml-auto text-xs text-zinc-500 truncate max-w-[45%]">{item.sub}</span>}
              {item.kind === 'view' && <span className="ml-auto text-[10px] uppercase text-zinc-500">{t('palette.studio')}</span>}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
