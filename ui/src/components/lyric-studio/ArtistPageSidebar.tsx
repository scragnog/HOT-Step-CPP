import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight, ListOrdered, Code2, Download, Zap } from 'lucide-react';
import type { Artist } from '../../services/lireekApi';
import { TripleProviderSelector, type ModelSelections, loadSelections, saveSelections } from './ProviderSelector';
import { EditableSlider } from '../shared/EditableSlider';
import { ScaleOverridePresets } from '../shared/ScaleOverridePresets';

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

// ── Toggle ──────────────────────────────────────────────────────────────────
const Toggle: React.FC<{ on: boolean; onClick: () => void }> = ({ on, onClick }) => (
  <button
    type="button"
    onClick={onClick}
    className={`w-9 h-[18px] rounded-full flex items-center transition-colors duration-200 px-0.5 border border-white/10 ${on ? 'bg-pink-600' : 'bg-black/40'} cursor-pointer`}
  >
    <div className={`w-3.5 h-3.5 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${on ? 'translate-x-[18px]' : 'translate-x-0'}`} />
  </button>
);

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

  // ── Adapter Scale Override — same localStorage keys read by audioGenQueueStore ──
  const [overrideExpanded, setOverrideExpanded] = useState(false);
  const [globalScaleOverrideEnabled, setGlobalScaleOverrideEnabled] = useLocalPersistedState('hs-globalScaleOverride', false);
  const [globalOverallScale, setGlobalOverallScale] = useLocalPersistedState('hs-globalOverallScale', 1.0);
  const [globalGroupScales, setGlobalGroupScales] = useLocalPersistedState<{ self_attn: number; cross_attn: number; mlp: number; cond_embed: number }>('hs-globalGroupScales', { self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0 });

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
        {/* ── Adapter Scale Override ───────────────────────────────── */}
        <div>
          <button
            onClick={() => setOverrideExpanded(!overrideExpanded)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.03] hover:bg-white/[0.06] border border-white/5 text-[11px] text-zinc-500 uppercase tracking-wider font-semibold transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Zap className="w-3 h-3" />
              Scale Overrides
              {globalScaleOverrideEnabled && (
                <span className="text-[9px] font-bold px-1 py-0.5 rounded bg-amber-500/20 text-amber-400 normal-case tracking-normal animate-pulse">
                  ON
                </span>
              )}
            </span>
            {overrideExpanded
              ? <ChevronDown className="w-3.5 h-3.5" />
              : <ChevronRight className="w-3.5 h-3.5" />
            }
          </button>
          {overrideExpanded && (
            <div className="mt-2 space-y-3 animate-in slide-in-from-top-1 duration-150">
              {/* Enable toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[11px] font-medium text-zinc-300">Enable Override</span>
                  <p className="text-[10px] text-zinc-600 leading-tight">
                    Overrides all per-album adapter scales
                  </p>
                </div>
                <Toggle on={globalScaleOverrideEnabled} onClick={() => setGlobalScaleOverrideEnabled(!globalScaleOverrideEnabled)} />
              </div>

              {/* Sliders — always visible (greyed when disabled) */}
              <div className={!globalScaleOverrideEnabled ? 'opacity-40 pointer-events-none' : ''}>
                {/* Scale Override Presets */}
                <ScaleOverridePresets
                  currentOverallScale={globalOverallScale}
                  currentGroupScales={globalGroupScales}
                  onLoad={(overall, groups) => {
                    setGlobalOverallScale(overall);
                    setGlobalGroupScales(groups);
                  }}
                  compact
                />
                <EditableSlider
                  label="Overall Scale"
                  value={globalOverallScale}
                  min={0} max={4} step={0.05}
                  onChange={setGlobalOverallScale}
                  formatDisplay={(v) => v.toFixed(2)}
                />
                <div className="space-y-1 mt-2 pl-2 border-l-2 border-amber-500/20">
                  <EditableSlider
                    label="Self-Attn"
                    value={globalGroupScales.self_attn}
                    min={0} max={4} step={0.05}
                    onChange={(v) => setGlobalGroupScales(prev => ({ ...prev, self_attn: v }))}
                    formatDisplay={(v) => v.toFixed(2)}
                    helpText="Controls how audio frames relate to each other over time"
                  />
                  <EditableSlider
                    label="Cross-Attn"
                    value={globalGroupScales.cross_attn}
                    min={0} max={4} step={0.05}
                    onChange={(v) => setGlobalGroupScales(prev => ({ ...prev, cross_attn: v }))}
                    formatDisplay={(v) => v.toFixed(2)}
                    helpText="How strongly the text prompt shapes the output vs. the adapter"
                  />
                  <EditableSlider
                    label="MLP"
                    value={globalGroupScales.mlp}
                    min={0} max={4} step={0.05}
                    onChange={(v) => setGlobalGroupScales(prev => ({ ...prev, mlp: v }))}
                    formatDisplay={(v) => v.toFixed(2)}
                    helpText="Controls the adapter's stored timbre, tonal texture, and sonic character"
                  />
                  <EditableSlider
                    label="Cond"
                    value={globalGroupScales.cond_embed}
                    min={0} max={4} step={0.05}
                    onChange={(v) => setGlobalGroupScales(prev => ({ ...prev, cond_embed: v }))}
                    formatDisplay={(v) => v.toFixed(2)}
                    helpText="Controls how the adapter reshapes text/style prompt interpretation"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

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
