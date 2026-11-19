import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { seedsOptions, type SavedSeed } from '@/queries/shell-queue/queue';
import { queryManager } from '@/queries/invalidation';
import { apiClient } from '@/services/httpClient';
// SeedManagerDrawer.tsx — Seed save/load popover for STORM Live Controls
// MDMAchine / A&E Concepts 2026
// GPL v3 — safe for public repo
//
// Renders as a floating panel anchored to the seed row.
// Triggered by a 💾 button added next to the existing 🎲 / 🔒 buttons.
//
// Features:
//   - List all saved seeds with search filter
//   - Click a seed → fires onLoad(seed) immediately (no confirm)
//   - Star toggle for favorites, shown first
//   - Save current seed with optional name + description
//   - Delete with single-click (pill turns red, second click confirms)
//   - Imports existing ComfyUI SeedSaver files with zero conversion

import React from 'react';
import { X, Star, Trash2, Save, Shuffle, Loader2 } from 'lucide-react';
import { HintButton } from '@/components/shared/controls/HintButton';
import { formatDay } from '@/utils/format/dateUtils';
import { plural } from '@/utils/format/plural';

// ─── Types ────────────────────────────────────────────────────────────────────

interface SeedManagerDrawerProps {
  isOpen:       boolean;
  onClose:      () => void;
  currentSeed:  number;
  /** Called when user clicks a saved seed — apply it immediately */
  onLoad:       (seed: number) => void;
  /** Called when user hits the random-saved-seed button */
  onLoadRandom: (seed: number) => void;
}

// ─── Subcomponents ────────────────────────────────────────────────────────────

// ─── Main component ───────────────────────────────────────────────────────────

export const SeedManagerDrawer: React.FC<SeedManagerDrawerProps> = ({
  isOpen, onClose, currentSeed, onLoad, onLoadRandom,
}) => {
  const client = useQueryClient();
  const options = seedsOptions();
  const query = useQuery({ ...options, enabled: isOpen });
  const seeds = query.data?.seeds ?? [];
  const loading = query.isPending;
  const refresh = () => queryManager.invalidate(client, options.queryKey);
  const [filter,      setFilter]      = React.useState('');
  const [saveName,    setSaveName]    = React.useState('');
  const [saveDesc,    setSaveDesc]    = React.useState('');
  const [deleteConfirm, setDeleteConfirm] = React.useState<string | null>(null);
  const [flashLoaded, setFlashLoaded] = React.useState<string | null>(null);
  // Read failures surface inline (GET errors never toast — the interceptor's
  // method filter); command failures are toasted by the transport. Each request
  // is single-use, so it lives in its mutation body rather than an extracted
  // helper.
  const [error,       setError]       = React.useState<string | null>(null);

  const saveMut = useMutation({ mutationFn: ({ name, seed, description }: { name: string; seed: number; description: string }) => apiClient.post('/seeds', { name, seed, description }) });
  const deleteMut = useMutation({ mutationFn: (name: string) => apiClient.delete(`/seeds/${encodeURIComponent(name)}`) });
  const favoriteMut = useMutation({ mutationFn: (name: string) => apiClient.post<{ favorite: boolean }>(`/seeds/${encodeURIComponent(name)}/favorite`).then(d => d.favorite) });
  const randomMut = useMutation({
    mutationFn: () => apiClient.get<{ seed: number; name: string }>('/seeds/random'),
    onSuccess: r => {
      onLoadRandom(r.seed);
      setFlashLoaded(r.name);
      setTimeout(() => setFlashLoaded(null), 1200);
    },
  });

  React.useEffect(() => { if (query.error) setError(query.error.message); }, [query.error]);

  // Clear delete confirm when filter changes
  React.useEffect(() => { setDeleteConfirm(null); }, [filter]);

  if (!isOpen) return null;

  // Sort: favorites first, then alpha
  const filtered = seeds
    .filter(s =>
      !filter ||
      s.name.toLowerCase().includes(filter.toLowerCase()) ||
      s.description.toLowerCase().includes(filter.toLowerCase()) ||
      s.tags.some(t => t.toLowerCase().includes(filter.toLowerCase()))
    )
    .sort((a, b) => {
      if (a.favorite !== b.favorite) return a.favorite ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

  const handleLoad = (s: SavedSeed) => {
    onLoad(s.seed);
    setFlashLoaded(s.name);
    setTimeout(() => setFlashLoaded(null), 1200);
  };

  const handleSave = () => {
    const name = saveName.trim() || `seed_${currentSeed}`;
    setError(null);
    saveMut.mutate(
      { name, seed: currentSeed, description: saveDesc.trim() },
      { onSuccess: () => { void refresh(); setSaveName(''); setSaveDesc(''); } },
    );
  };

  const handleDelete = (name: string) => {
    if (deleteConfirm !== name) { setDeleteConfirm(name); return; }
    setDeleteConfirm(null);
    deleteMut.mutate(name, { onSuccess: () => void refresh() });
  };

  const handleFavorite = (name: string) => {
    favoriteMut.mutate(name, { onSuccess: () => void refresh() });
  };

  return (
    <>
      {/* Backdrop — click to close */}
      <div className="fixed inset-0 z-40 bg-black/40" onClick={onClose} />

      {/* Drawer panel */}
      <div
        className="fixed z-50 w-72 bg-zinc-900 rounded-xl border border-zinc-700 shadow-2xl overflow-hidden"
        style={{ bottom: '3.5rem', right: '1rem' }}   // anchors above seed row; adjust if needed
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800">
          <span className="text-[11px] font-bold text-zinc-200">🎲 Seed Manager</span>
          <div className="flex items-center gap-1">
            <HintButton onClick={() => randomMut.mutate()} disabled={randomMut.isPending}
              label="Load a random saved seed"
              info="Picks one of your saved seeds at random and applies it to the current field."
              ariaLabel="Load a random saved seed"
              className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors flex items-center gap-0.5 disabled:opacity-50 disabled:cursor-not-allowed">
              {randomMut.isPending ? <Loader2 size={9} className="animate-spin" /> : <Shuffle size={9} />} random
            </HintButton>
            <button onClick={onClose} aria-label="Close seed manager"
              className="p-0.5 rounded text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800 transition-colors">
              <X size={12} />
            </button>
          </div>
        </div>

        {/* Save current seed */}
        <div className="px-3 py-2 border-b border-zinc-800 space-y-1.5">
          <div className="text-[9px] text-zinc-600 uppercase tracking-wider">Save current seed</div>
          <div className="flex items-center gap-1">
            <div className="flex-1 text-[9px] text-amber-400 tabular-nums font-mono bg-zinc-800 rounded px-1.5 py-0.5 border border-zinc-700">
              {currentSeed}
            </div>
          </div>
          <input
            type="text"
            value={saveName}
            onChange={e => setSaveName(e.target.value)}
            placeholder={`name (default: seed_${currentSeed})`}
            onKeyDown={e => { if (e.key === 'Enter') handleSave(); }}
            className="w-full px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[9px] text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-amber-500/40 transition-colors"
          />
          <input
            type="text"
            value={saveDesc}
            onChange={e => setSaveDesc(e.target.value)}
            placeholder="description (optional)"
            onKeyDown={e => { if (e.key === 'Enter') handleSave(); }}
            className="w-full px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[9px] text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-amber-500/40 transition-colors"
          />
          <button
            onClick={handleSave}
            disabled={saveMut.isPending}
            className="flex items-center gap-1 text-[9px] px-2 py-0.5 rounded bg-amber-600 hover:bg-amber-500 text-white font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {saveMut.isPending ? <Loader2 size={9} className="animate-spin" /> : <Save size={9} />}
            Save
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="px-3 py-1.5 text-[9px] text-red-400 bg-red-900/20 border-b border-zinc-800">
            ❌ {error}
          </div>
        )}

        {/* Search */}
        <div className="px-3 py-2 border-b border-zinc-800">
          <input
            type="text"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="search seeds…"
            className="w-full px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[9px] text-zinc-300 placeholder:text-zinc-600 outline-none focus:border-zinc-500 transition-colors"
          />
        </div>

        {/* Seed list */}
        <div className="max-h-48 overflow-y-auto hide-scrollbar">
          {loading && (
            <div className="flex items-center justify-center py-4 gap-1.5 text-zinc-600">
              <Loader2 size={12} className="animate-spin" />
              <span className="text-[9px]">loading…</span>
            </div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="text-center py-4 text-[9px] text-zinc-700">
              {seeds.length === 0 ? 'no seeds saved yet' : 'no matches'}
            </div>
          )}

          {!loading && filtered.map(s => (
            <div
              key={s.name}
              className={`group flex items-center gap-1.5 px-3 py-1.5 hover:bg-zinc-800/60 transition-colors border-b border-zinc-800/40 last:border-0 ${
                flashLoaded === s.name ? 'bg-green-900/20' : ''
              }`}
            >
              {/* Favorite star */}
              <button
                onClick={() => handleFavorite(s.name)}
                // A double-click would queue two toggles — the second undoes the first.
                disabled={favoriteMut.isPending && favoriteMut.variables === s.name}
                className={`flex-shrink-0 transition-colors ${
                  s.favorite ? 'text-amber-400 hover:text-amber-300' : 'text-zinc-700 hover:text-zinc-500'
                }`}
                title={s.favorite ? 'Remove from favorites' : 'Add to favorites'}
              >
                <Star size={9} fill={s.favorite ? 'currentColor' : 'none'} />
              </button>

              {/* Load button — name + seed value */}
              <button
                onClick={() => handleLoad(s)}
                className="flex-1 text-left min-w-0"
                title={`Load seed ${s.seed}${s.description ? ` — ${s.description}` : ''}`}
              >
                <div className="flex items-baseline gap-1.5 min-w-0">
                  <span className={`text-[9px] font-medium truncate ${
                    flashLoaded === s.name ? 'text-green-400' : 'text-zinc-300 group-hover:text-white'
                  }`}>
                    {flashLoaded === s.name ? '✓ loaded' : s.name}
                  </span>
                  <span className="text-[8px] text-zinc-600 tabular-nums shrink-0 font-mono">
                    {s.seed}
                  </span>
                </div>
                {(s.description || s.saved_at) && (
                  <div className="flex items-center gap-1 mt-0.5">
                    {s.description && (
                      <span className="text-[7px] text-zinc-600 truncate">{s.description}</span>
                    )}
                    {s.saved_at && (
                      <span className="text-[7px] text-zinc-700 shrink-0">{formatDay(s.saved_at)}</span>
                    )}
                  </div>
                )}
              </button>

              {/* Delete */}
              <button
                onClick={() => handleDelete(s.name)}
                className={`flex-shrink-0 transition-all rounded px-1 py-0.5 ${
                  deleteConfirm === s.name
                    ? 'text-red-300 bg-red-900/40 text-[8px] font-medium'
                    : 'text-zinc-700 hover:text-red-400 opacity-0 group-hover:opacity-100'
                }`}
                title={deleteConfirm === s.name ? 'Click again to confirm delete' : 'Delete seed'}
              >
                {deleteConfirm === s.name ? '✕ sure?' : <Trash2 size={9} />}
              </button>
            </div>
          ))}
        </div>

        {/* Footer count */}
        <div className="px-3 py-1.5 border-t border-zinc-800">
          <span className="text-[8px] text-zinc-700">
            {seeds.length} {plural(seeds.length, 'seed')} saved
            {seeds.some(s => s.favorite) ? ` · ${seeds.filter(s => s.favorite).length} ★` : ''}
          </span>
        </div>
      </div>
    </>
  );
};
