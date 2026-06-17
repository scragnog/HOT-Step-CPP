// VstChainDropdown.tsx — VST3 Post-Processing chain panel for the global bar
//
// Shows the plugin chain, lets you add/remove/reorder plugins,
// toggle enable/disable, and launch native plugin GUIs.
//
// Changes over stock:
//   - PluginSearch stays open after add; shows added count; Done button
//   - ChainRow: green dot when GUI open, restart-to-apply hint while monitoring
//   - Pending changes banner when monitor is running and a GUI was opened
//   - Preset section: save / load / delete named chain snapshots
//   - Status poll lives in MonitorBar now (no own interval here)

import React, { useEffect, useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Plus, Trash2, ExternalLink, Search,
  ChevronUp, ChevronDown, Power, Headphones, Square,
  BookmarkPlus, Check, RefreshCw, AlertTriangle,
} from 'lucide-react';
import { useVstChainStore } from '../../stores/vstChainStore';
import { usePlaybackSelector, togglePlay } from '../../stores/playbackStore';

// Format seconds as mm:ss
function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

// ── Plugin Search Dropdown ──────────────────────────────────
// Stays open after adds. Done button closes.

const PluginSearch: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const { plugins, scanning, scanPlugins, addToChain, chain } = useVstChainStore();
  const { t } = useTranslation();
  const [filter, setFilter] = useState('');
  const [addedUids, setAddedUids] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (plugins.length === 0 && !scanning) scanPlugins();
  }, [plugins.length, scanning, scanPlugins]);

  const filtered = useMemo(() => {
    const q = filter.toLowerCase();
    return (plugins || []).filter(p =>
      !q ||
      p.name.toLowerCase().includes(q) ||
      p.vendor.toLowerCase().includes(q) ||
      p.subcategories.toLowerCase().includes(q)
    );
  }, [plugins, filter]);

  const inChain = useMemo(() => new Set((chain || []).map(p => p.uid)), [chain]);

  const handleAdd = (plugin: typeof plugins[number]) => {
    addToChain(plugin);
    setAddedUids(s => new Set([...s, plugin.uid]));
  };

  return (
    <div className="space-y-2">
      {/* Search input */}
      <div className="relative">
        <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
        <input
          type="text"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="Search plugins..."
          autoFocus
          className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20 outline-none transition-colors"
        />
      </div>

      {/* Plugin list */}
      <div className="max-h-48 overflow-y-auto space-y-0.5 scrollbar-thin">
        {scanning ? (
          <div className="flex items-center gap-2 justify-center py-4 text-zinc-500">
            <span className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs">{t('vst.scanningPlugins')}</span>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-xs text-zinc-600 text-center py-3 italic">
            {plugins.length === 0 ? t('vst.noPluginsFound') : t('vst.noMatches')}
          </div>
        ) : (
          filtered.map(plugin => {
            const already = inChain.has(plugin.uid);
            const justAdded = addedUids.has(plugin.uid);
            return (
              <button
                key={plugin.uid}
                disabled={already}
                onClick={() => !already && handleAdd(plugin)}
                className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left transition-colors ${
                  already ? 'opacity-40 cursor-not-allowed' : 'hover:bg-violet-500/10 cursor-pointer'
                }`}
              >
                {justAdded
                  ? <Check size={12} className="flex-shrink-0 text-emerald-400" />
                  : <Plus size={12} className={`flex-shrink-0 ${already ? 'text-zinc-600' : 'text-violet-400'}`} />
                }
                <div className="min-w-0 flex-1">
                  <div className="text-xs text-zinc-800 dark:text-zinc-200 truncate">{plugin.name}</div>
                  <div className="text-[10px] text-zinc-500 truncate">{plugin.vendor} · {plugin.subcategories}</div>
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* Rescan + Done */}
      <div className="flex items-center justify-between pt-1">
        <button
          onClick={scanPlugins}
          disabled={scanning}
          className="text-[10px] text-zinc-600 hover:text-violet-400 transition-colors"
        >
          {scanning ? t('vst.scanning') : `Rescan (${plugins.length} found)`}
        </button>
        <button
          onClick={onClose}
          className="text-[10px] font-semibold text-violet-400 hover:text-violet-300 transition-colors px-2 py-0.5 rounded-lg hover:bg-violet-500/10"
        >
          {addedUids.size > 0 ? `Done (+${addedUids.size})` : 'Done'}
        </button>
      </div>
    </div>
  );
};

// ── Preset Manager ──────────────────────────────────────────
// Save / load / delete named chain snapshots (localStorage).

const PresetManager: React.FC = () => {
  const { presets, savePreset, loadPreset, deletePreset, chain } = useVstChainStore();
  const [saving, setSaving] = useState(false);
  const [newName, setNewName] = useState('');
  const names = Object.keys(presets);

  const handleSave = () => {
    const name = newName.trim();
    if (!name) return;
    savePreset(name);
    setNewName('');
    setSaving(false);
  };

  if (names.length === 0 && !saving && chain.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {names.length > 0 && (
        <select
          className="flex-1 min-w-0 px-2 py-1 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-white/10 text-[10px] text-zinc-600 dark:text-zinc-400 cursor-pointer focus:border-violet-500/50 outline-none transition-colors"
          defaultValue=""
          onChange={e => { if (e.target.value) { loadPreset(e.target.value); e.target.value = ''; } }}
        >
          <option value="" disabled>Load preset…</option>
          {names.map(n => <option key={n} value={n}>{n}</option>)}
        </select>
      )}

      {names.length > 0 && (
        <select
          className="px-2 py-1 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-white/10 text-[10px] text-zinc-600 dark:text-zinc-400 cursor-pointer focus:border-red-500/50 outline-none transition-colors"
          defaultValue=""
          onChange={e => { if (e.target.value) { deletePreset(e.target.value); e.target.value = ''; } }}
        >
          <option value="" disabled>Delete…</option>
          {names.map(n => <option key={n} value={n}>✕ {n}</option>)}
        </select>
      )}

      {chain.length > 0 && (
        saving ? (
          <div className="flex items-center gap-1 flex-1 min-w-0">
            <input
              type="text"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleSave(); if (e.key === 'Escape') setSaving(false); }}
              placeholder="Preset name…"
              autoFocus
              className="flex-1 min-w-0 px-2 py-1 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-violet-500/40 text-[10px] text-zinc-800 dark:text-zinc-200 placeholder-zinc-500 outline-none"
            />
            <button
              onClick={handleSave}
              disabled={!newName.trim()}
              className="p-1 rounded text-emerald-400 hover:text-emerald-300 disabled:opacity-30 transition-colors"
            >
              <Check size={12} />
            </button>
          </div>
        ) : (
          <button
            onClick={() => setSaving(true)}
            title="Save current chain as preset"
            className="p-1.5 rounded-lg hover:bg-violet-500/10 text-zinc-500 hover:text-violet-400 transition-colors flex-shrink-0"
          >
            <BookmarkPlus size={12} />
          </button>
        )
      )}
    </div>
  );
};

// ── Chain Entry Row ─────────────────────────────────────────

const ChainRow: React.FC<{
  entry: { uid: string; name: string; vendor: string; path: string; enabled: boolean; statePath: string };
  index: number;
  total: number;
}> = ({ entry, index, total }) => {
  const { toggleEnabled, removeFromChain, reorderChain, openGui, openGuiUids, monitoring } = useVstChainStore();
  const { t } = useTranslation();
  const guiOpen = openGuiUids.includes(entry.uid);

  return (
    <div
      className={`flex items-center gap-1.5 px-2 py-1.5 rounded-lg border transition-all ${
        entry.enabled
          ? 'bg-violet-500/5 border-violet-500/20'
          : 'bg-zinc-100/50 dark:bg-zinc-800/50 border-zinc-200 dark:border-white/5 opacity-50'
      }`}
    >
      {/* Drag handle placeholder + reorder buttons */}
      <div className="flex flex-col gap-0 flex-shrink-0">
        <button
          onClick={() => reorderChain(index, index - 1)}
          disabled={index === 0}
          className="p-0 text-zinc-600 hover:text-zinc-700 dark:text-zinc-300 disabled:opacity-20 transition-colors"
        >
          <ChevronUp size={10} />
        </button>
        <button
          onClick={() => reorderChain(index, index + 1)}
          disabled={index >= total - 1}
          className="p-0 text-zinc-600 hover:text-zinc-700 dark:text-zinc-300 disabled:opacity-20 transition-colors"
        >
          <ChevronDown size={10} />
        </button>
      </div>

      {/* Order number */}
      <span className="text-[10px] text-zinc-600 font-mono w-3 text-center flex-shrink-0">
        {index + 1}
      </span>

      {/* Plugin name (green dot when its GUI window is open) */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-zinc-800 dark:text-zinc-200 truncate">{entry.name}</span>
          {guiOpen && <span title="GUI window open" className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />}
        </div>
        <div className="text-[10px] text-zinc-500 truncate">{entry.vendor}</div>
      </div>

      {/* Actions */}
      <button
        onClick={() => openGui(entry)}
        title={monitoring ? 'Open GUI — restart monitor to apply changes' : (guiOpen ? 'Bring GUI to front' : t('vst.openPluginUi'))}
        className={`p-1 rounded transition-colors flex-shrink-0 ${
          guiOpen ? 'text-emerald-400 hover:bg-emerald-500/10' : 'text-zinc-500 hover:text-violet-400 hover:bg-violet-500/10'
        }`}
      >
        <ExternalLink size={12} />
      </button>

      <button
        onClick={() => toggleEnabled(entry.uid)}
        title={entry.enabled ? 'Disable' : 'Enable'}
        className={`p-1 rounded transition-colors flex-shrink-0 ${
          entry.enabled
            ? 'hover:bg-amber-500/10 text-emerald-400 hover:text-amber-400'
            : 'hover:bg-emerald-500/10 text-zinc-600 hover:text-emerald-400'
        }`}
      >
        <Power size={12} />
      </button>

      <button
        onClick={() => removeFromChain(entry.uid)}
        title={t('vst.removeFromChain')}
        className="p-1 rounded hover:bg-red-500/10 text-zinc-500 hover:text-red-400 transition-colors flex-shrink-0"
      >
        <Trash2 size={11} />
      </button>
    </div>
  );
};

// ── Main Dropdown ───────────────────────────────────────────

export const VstChainDropdown: React.FC = () => {
  const {
    chain, chainLoaded, loadChain, monitoring,
    startMonitor, stopMonitor, restartMonitor,
    monitorPosition, monitorDuration, seekMonitor,
    pendingGuiChanges,
  } = useVstChainStore();
  const currentTrack = usePlaybackSelector(s => s.currentTrack);
  const isPlaying = usePlaybackSelector(s => s.isPlaying);
  const presets = useVstChainStore(s => s.presets);
  const [showSearch, setShowSearch] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const { t } = useTranslation();

  useEffect(() => {
    if (!chainLoaded) loadChain();
  }, [chainLoaded, loadChain]);

  const safeChain = chain || [];
  const enabledCount = safeChain.filter(p => p.enabled).length;
  const hasTrack = !!currentTrack?.audioUrl;

  const handleMonitorToggle = async () => {
    if (monitoring) {
      await stopMonitor();
    } else if (hasTrack) {
      // Pause browser playback to avoid double audio
      if (isPlaying) togglePlay();
      await startMonitor(currentTrack!.audioUrl);
    }
  };

  const handleRestart = async () => {
    setRestarting(true);
    await restartMonitor();
    setRestarting(false);
  };

  return (
    <div className="space-y-3">
      {/* Presets */}
      {(safeChain.length > 0 || Object.keys(presets).length > 0) && (
        <div className="space-y-1">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Presets</div>
          <PresetManager />
        </div>
      )}

      {/* Pending changes banner — GUI was opened while monitoring */}
      {monitoring && pendingGuiChanges && (
        <div className="flex items-center gap-2 px-2.5 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
          <AlertTriangle size={12} className="text-amber-400 flex-shrink-0" />
          <span className="text-[10px] text-amber-300 flex-1 leading-relaxed">
            GUI changes won't be heard until the monitor restarts — it loads state files at startup.
          </span>
          <button
            onClick={handleRestart}
            disabled={restarting}
            className="flex items-center gap-1 text-[10px] font-semibold text-amber-300 hover:text-amber-200 transition-colors flex-shrink-0 disabled:opacity-50"
          >
            <RefreshCw size={11} className={restarting ? 'animate-spin' : ''} />
            {restarting ? 'Restarting…' : 'Restart'}
          </button>
        </div>
      )}

      {/* Current chain */}
      {safeChain.length > 0 ? (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">
              Plugin Chain ({enabledCount}/{safeChain.length} active)
            </div>
            {monitoring && !pendingGuiChanges && (
              <button
                onClick={handleRestart}
                disabled={restarting}
                title="Restart monitor to reload plugin state"
                className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-violet-400 transition-colors disabled:opacity-50"
              >
                <RefreshCw size={10} className={restarting ? 'animate-spin' : ''} />
                reload
              </button>
            )}
          </div>
          {safeChain.map((entry, i) => (
            <ChainRow key={entry.uid} entry={entry} index={i} total={safeChain.length} />
          ))}
        </div>
      ) : (
        <div className="text-xs text-zinc-500 italic text-center py-2">
          {t('vst.noPluginsInChain')}
        </div>
      )}

      {/* Monitor button */}
      {enabledCount > 0 && (
        <button
          onClick={handleMonitorToggle}
          disabled={!monitoring && !hasTrack}
          className={`w-full flex items-center justify-center gap-2 px-3 py-2.5 text-xs font-semibold rounded-xl border transition-all ${
            monitoring
              ? 'bg-violet-500/15 border-violet-500/40 text-violet-300 hover:bg-violet-500/25'
              : hasTrack
                ? 'bg-zinc-100 dark:bg-zinc-800 border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-400 hover:border-violet-500/30 hover:text-violet-400'
                : 'bg-zinc-100/50 dark:bg-zinc-800/50 border-zinc-200 dark:border-white/5 text-zinc-600 cursor-not-allowed'
          }`}
        >
          {monitoring ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-500" />
              </span>
              <Square size={12} />
              {t('vst.stopMonitor')}
            </>
          ) : (
            <>
              <Headphones size={14} />
              {hasTrack ? t('vst.monitorWithVst') : t('vst.playTrackFirst')}
            </>
          )}
        </button>
      )}

      {/* Transport bar during monitoring (MonitorBar is the persistent version) */}
      {monitoring && (monitorDuration ?? 0) > 0 && (
        <div className="space-y-1">
          <input
            type="range"
            min={0}
            max={monitorDuration}
            step={0.5}
            value={monitorPosition ?? 0}
            onChange={e => seekMonitor(parseFloat(e.target.value))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, rgb(139 92 246) ${((monitorPosition ?? 0) / monitorDuration) * 100}%, rgb(63 63 70) ${((monitorPosition ?? 0) / monitorDuration) * 100}%)`,
            }}
          />
          <div className="flex justify-between text-[10px] text-zinc-500 font-mono">
            <span>{formatTime(monitorPosition ?? 0)}</span>
            <span>{formatTime(monitorDuration)}</span>
          </div>
        </div>
      )}

      {/* Add plugin */}
      {showSearch ? (
        <div className="border-t border-zinc-200 dark:border-white/5 pt-2">
          <PluginSearch onClose={() => setShowSearch(false)} />
        </div>
      ) : (
        <button
          onClick={() => setShowSearch(true)}
          className="w-full flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border border-zinc-300 dark:border-white/10 hover:border-violet-500/30 hover:text-violet-400 transition-all"
        >
          <Plus size={14} /> {t('vst.addPlugin')}
        </button>
      )}

      {/* Info */}
      <p className="text-[10px] text-zinc-600 leading-relaxed">
        VST3 plugins are applied to generated audio in chain order.
        Click <ExternalLink size={9} className="inline" /> to open the plugin's native UI
        and configure settings — they're saved automatically when you close the window.
      </p>
    </div>
  );
};

// ── Badge ───────────────────────────────────────────────────

export const VstChainBadge: React.FC = () => {
  const { chain, monitoring } = useVstChainStore();
  const enabled = (chain || []).filter(p => p.enabled);

  if (enabled.length === 0 && !monitoring) return null;

  return (
    <span className="flex items-center gap-1.5 text-[10px] text-violet-400/60 font-mono truncate">
      {monitoring && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-violet-500" />
        </span>
      )}
      {monitoring ? 'monitoring' : `${enabled.length} plugin${enabled.length !== 1 ? 's' : ''}`}
    </span>
  );
};
