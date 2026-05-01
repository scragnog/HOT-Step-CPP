import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight, ListOrdered, Code2, Download, Clock } from 'lucide-react';
import type { Artist } from '../../services/lireekApi';
import { TripleProviderSelector, type ModelSelections, loadSelections, saveSelections } from './ProviderSelector';
import { LLM_DURATION_KEY } from '../../utils/estimateDuration';

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

  // ── LLM Models ──
  const [modelSelections, setModelSelections] = useState<ModelSelections>(loadSelections);
  const [llmExpanded, setLlmExpanded] = useState(false);

  // ── Download filename prepend ──
  const [filenamePrepend, setFilenamePrepend] = useLocalPersistedState<string>('lireek-downloadFilenamePrepend', '');

  // ── LLM Duration toggle ──
  const [useLlmDuration, setUseLlmDuration] = useLocalPersistedState<boolean>(LLM_DURATION_KEY, true);

  const gradient = (name: string) => {
    const hash = name.split('').reduce((a, c) => ((a << 5) - a + c.charCodeAt(0)) | 0, 0);
    const h1 = Math.abs(hash) % 360;
    const h2 = (h1 + 40) % 360;
    return `linear-gradient(180deg, hsl(${h1}, 50%, 20%) 0%, hsl(${h2}, 40%, 12%) 100%)`;
  };

  return (
    <div className="flex flex-col h-full bg-zinc-950/50 overflow-hidden">
      {/* Artist header — only shown when artist context available */}
      {artist && (
        <>
          <div className="relative flex-shrink-0">
            {artist.image_url && !imageError ? (
              <img
                src={artist.image_url}
                alt={artist.name}
                className="w-full aspect-[16/9] object-cover"
                onError={() => setImageError(true)}
              />
            ) : (
              <div className="w-full aspect-[16/9]" style={{ background: gradient(artist.name) }} />
            )}
            <div className="absolute inset-0 bg-gradient-to-t from-zinc-950 via-transparent to-transparent" />
          </div>
          <div className="px-4 py-3 -mt-6 relative z-10 flex-shrink-0">
            <h2 className="text-base font-bold text-white leading-tight">{artist.name}</h2>
            <p className="text-xs text-zinc-400 mt-0.5">
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
          Bulk Operations
        </button>
        <button
          onClick={onOpenPromptEditor}
          className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white text-xs transition-colors"
          title="Edit System Prompts"
        >
          <Code2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Scrollable settings area */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-4 pb-4 space-y-3">

        {/* ── Download Filename Prepend ─────────────────────────── */}
        <div>
          <div
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-white/5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold"
          >
            <span className="flex items-center gap-1.5">
              <Download className="w-3 h-3" />
              Filename Prepend
            </span>
          </div>
          <div className="mt-2 px-1">
            <input
              type="text"
              value={filenamePrepend}
              onChange={e => setFilenamePrepend(e.target.value)}
              placeholder="e.g. MyLabel - "
              className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-pink-500 transition-colors"
            />
            <p className="text-[10px] text-zinc-600 mt-1 leading-tight">
              Prepended to download filenames, e.g. <span className="text-zinc-500">{filenamePrepend || '...'}</span>Artist - Song.flac
            </p>
          </div>
        </div>

        {/* ── LLM Duration Override ───────────────────────────── */}
        <div>
          <label
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] border border-white/5 cursor-pointer hover:bg-white/[0.06] transition-colors"
          >
            <span className="flex items-center gap-1.5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold">
              <Clock className="w-3 h-3" />
              LLM Duration
            </span>
            <input
              type="checkbox"
              checked={useLlmDuration}
              onChange={e => setUseLlmDuration(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 text-pink-500 focus:ring-pink-500 focus:ring-offset-0 cursor-pointer"
            />
          </label>
          <p className="text-[10px] text-zinc-600 mt-1 px-1 leading-tight">
            {useLlmDuration
              ? <>Uses the LLM's estimated duration — may overshoot, causing "double song" artifacts.</>
              : <>Uses calculated duration from lyrics + BPM — tighter fit, less wasted generation.</>
            }
          </p>
        </div>

        {/* ── LLM Models ──────────────────────────────────────────── */}
        <div>
          <button
            onClick={() => setLlmExpanded(!llmExpanded)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] hover:bg-white/[0.06] border border-white/5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold transition-colors"
          >
            <span>LLM Models</span>
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
