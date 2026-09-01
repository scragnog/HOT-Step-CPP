import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronRight, ListOrdered, Code2, Download, Clock, Shuffle, Brain } from 'lucide-react';
import type { Artist } from '../../services/lireekApi';
import { TripleProviderSelector, type ModelSelections, loadSelections, saveSelections } from './ProviderSelector';
import { LLM_DURATION_KEY } from '../../utils/estimateDuration';
import { USE_LM_ADAPTER_KEY } from '../../utils/lmAdapterPref';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';

// ── Persisted state hook ────────────────────────────────────────────────────
function useLocalPersistedState<T>(key: string, defaultValue: T): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [state, setState] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw !== null ? JSON.parse(raw) : defaultValue;
    } catch { return defaultValue; }
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);

  return [state, setState];
}


// ── Props ────────────────────────────────────────────────────────────────────

interface ArtistPageSidebarProps {
  artist?: Artist;
  albumCount?: number;
  onOpenQueue: () => void;
  onOpenPromptEditor: () => void;
}

export const ArtistPageSidebar: React.FC<ArtistPageSidebarProps> = ({
  artist, albumCount, onOpenQueue, onOpenPromptEditor,
}) => {
  const [imageError, setImageError] = useState(false);
  const { t } = useTranslation();
  const { disguiseArtist, disguiseImageUrl } = useDisguiseMode();

  // ── LLM Models ──
  const [modelSelections, setModelSelections] = useState<ModelSelections>(loadSelections);
  const [llmExpanded, setLlmExpanded] = useState(false);

  // ── Download filename prepend ──
  const [filenamePrepend, setFilenamePrepend] = useLocalPersistedState<string>('lireek-downloadFilenamePrepend', '');

  // ── LLM Duration toggle ──
  const [useLlmDuration, setUseLlmDuration] = useLocalPersistedState<boolean>(LLM_DURATION_KEY, true);

  // ── Randomize Timbre Reference ──
  const [randomizeTimbre, setRandomizeTimbre] = useLocalPersistedState<boolean>('lireek-randomizeTimbreRef', false);

  // ── Use LM Adapter ──
  // Off by default: a trained DiT adapter on the stock planner has been
  // sounding better than the same DiT adapter paired with its LM adapter.
  const [useLmAdapter, setUseLmAdapter] = useLocalPersistedState<boolean>(USE_LM_ADAPTER_KEY, false);

  const gradient = (name: string) => {
    const hash = name.split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
    const h1 = Math.abs(hash) % 360;
    const h2 = (h1 + 40) % 360;
    return `linear-gradient(180deg, hsl(${h1}, 50%, 20%) 0%, hsl(${h2}, 40%, 12%) 100%)`;
  };

  return (
    <div className="flex flex-col h-full bg-zinc-50 dark:bg-zinc-950/50 overflow-hidden">
      {/* Artist header — only shown when artist context available */}
      {artist && (
        <>
          <div className="relative flex-shrink-0">
            {(() => {
              const dUrl = disguiseImageUrl(artist.image_url, artist.name);
              return dUrl && !imageError ? (
              <img
                src={dUrl}
                alt={disguiseArtist(artist.name)}
                className="w-full aspect-[16/9] object-cover"
                onError={() => setImageError(true)}
              />
            ) : (
              <div className="w-full aspect-[16/9]" style={{ background: gradient(disguiseArtist(artist.name)) }} />
            );
            })()}
            <div className="absolute inset-0 bg-gradient-to-t from-zinc-950 via-transparent to-transparent" />
          </div>
          <div className="px-4 py-3 -mt-6 relative z-10 flex-shrink-0">
            <h2 className="text-base font-bold text-white leading-tight">{disguiseArtist(artist.name)}</h2>
            <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">
              {(albumCount ?? 0)} album{(albumCount ?? 0) !== 1 ? 's' : ''}
            </p>
          </div>
        </>
      )}

      {/* Action buttons */}
      <div className="px-4 py-2 flex gap-2 flex-shrink-0">
        <button
          onClick={onOpenQueue}
          className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-pink-600/20 hover:bg-pink-600/30 text-pink-400 text-xs font-semibold transition-colors"
        >
          <ListOrdered className="w-3.5 h-3.5" />
          {t('lyric.bulkOperations')}
        </button>
        <button
          onClick={onOpenPromptEditor}
          className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-600 dark:text-zinc-400 hover:text-white text-xs transition-colors"
          title={t('lyric.editSystemPrompts')}
        >
          <Code2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Scrollable settings area */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-4 pb-4 space-y-3">

        {/* ── Download Filename Prepend ─────────────────────────── */}
        <div>
          <div
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-zinc-200 dark:border-white/5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold"
          >
            <span className="flex items-center gap-1.5">
              <Download className="w-3 h-3" />
              {t('lyric.filenamePrepend')}
            </span>
          </div>
          <div className="mt-2 px-1">
            <input
              type="text"
              value={filenamePrepend}
              onChange={e => setFilenamePrepend(e.target.value)}
              placeholder="e.g. MyLabel - "
              className="w-full bg-zinc-200 dark:bg-black/20 border border-zinc-300 dark:border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-pink-500 transition-colors"
            />
            <p className="text-[10px] text-zinc-600 mt-1 leading-tight">
              Prepended to download filenames, e.g. <span className="text-zinc-500">{filenamePrepend || '...'}</span>Artist - Song.flac
            </p>
          </div>
        </div>

        {/* ── LLM Duration Override ───────────────────────────── */}
        <div>
          <label
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-zinc-200 dark:border-white/5 cursor-pointer hover:bg-white/[0.06] transition-colors"
          >
            <span className="flex items-center gap-1.5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold">
              <Clock className="w-3 h-3" />
              {t('lyric.llmDuration')}
            </span>
            <input
              type="checkbox"
              checked={useLlmDuration}
              onChange={e => setUseLlmDuration(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-100 dark:bg-zinc-800 text-pink-500 focus:ring-pink-500 focus:ring-offset-0 cursor-pointer"
            />
          </label>
          <p className="text-[10px] text-zinc-600 mt-1 px-1 leading-tight">
            {useLlmDuration
              ? <>Uses the LLM's estimated duration — may overshoot, causing "double song" artifacts.</>
              : <>Uses calculated duration from lyrics + BPM — tighter fit, less wasted generation.</>
            }
          </p>
        </div>

        {/* ── Randomize Timbre Reference ────────────────────────── */}
        <div>
          <label
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-zinc-200 dark:border-white/5 cursor-pointer hover:bg-white/[0.06] transition-colors"
          >
            <span className="flex items-center gap-1.5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold">
              <Shuffle className="w-3 h-3" />
              {t('lyric.randomizeTimbre')}
            </span>
            <input
              type="checkbox"
              checked={randomizeTimbre}
              onChange={e => setRandomizeTimbre(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-100 dark:bg-zinc-800 text-amber-500 focus:ring-amber-500 focus:ring-offset-0 cursor-pointer"
            />
          </label>
          <p className="text-[10px] text-zinc-600 mt-1 px-1 leading-tight">
            {randomizeTimbre
              ? <>Picks a random track from the reference folder as timbre conditioner — prevents riff leakage from a single reference.</>
              : <>Uses the exact reference track set in the album preset for timbre conditioning.</>
            }
          </p>
        </div>

        {/* ── Use LM Adapter ────────────────────────────────────── */}
        <div>
          <label
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-zinc-200 dark:border-white/5 cursor-pointer hover:bg-white/[0.06] transition-colors"
          >
            <span className="flex items-center gap-1.5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold">
              <Brain className="w-3 h-3" />
              {t('lyric.useLmAdapter')}
            </span>
            <input
              type="checkbox"
              checked={useLmAdapter}
              onChange={e => setUseLmAdapter(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-100 dark:bg-zinc-800 text-violet-500 focus:ring-violet-500 focus:ring-offset-0 cursor-pointer"
            />
          </label>
          <p className="text-[10px] text-zinc-600 mt-1 px-1 leading-tight">
            {useLmAdapter
              ? t('lyric.useLmAdapterOn')
              : t('lyric.useLmAdapterOff')
            }
          </p>
        </div>

        {/* "Generate All Audio" lived here. It queued every written song in the library
            with no way to narrow it down, and has been replaced by the Generate Audio tab
            in Bulk Operations, which does the same job per artist/album. */}

        {/* ── LLM Models ──────────────────────────────────────────── */}
        <div>
          <button
            onClick={() => setLlmExpanded(!llmExpanded)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] hover:bg-white/[0.06] border border-zinc-200 dark:border-white/5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold transition-colors"
          >
            <span>{t('lyric.llmModels')}</span>
            {llmExpanded
              ? <ChevronDown className="w-3.5 h-3.5" />
              : <ChevronRight className="w-3.5 h-3.5" />
            }
          </button>
          {llmExpanded && (
            <div className="mt-2 animate-in slide-in-from-top-1 duration-150">
              <TripleProviderSelector
                selections={modelSelections}
                onSelectionsChange={(sel) => {
                  setModelSelections(sel);
                  saveSelections(sel);
                }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
