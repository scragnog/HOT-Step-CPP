// VstChainDropdown.tsx — VST3 Post-Processing chain panel for the global bar
//
// Shows the plugin chain, lets you add/remove/reorder plugins,
// toggle enable/disable, and launch native plugin GUIs.

import React, { useEffect, useState, useMemo } from 'react';
import {
  Plus, Trash2, ExternalLink, Search,
  ChevronUp, ChevronDown, Power, Headphones, Square,
} from 'lucide-react';
import { useVstChainStore } from '../../stores/vstChainStore';
import { usePlayback, togglePlay } from '../../stores/playbackStore';
// VstPlugin type reserved for future use
// import type { VstPlugin } from '../../services/api';

// Format seconds as mm:ss
function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

// ── Plugin Search Dropdown ──────────────────────────────────

const PluginSearch: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const { plugins, scanning, scanPlugins, addToChain, chain } = useVstChainStore();
  const [filter, setFilter] = useState('');

  useEffect(() => {
    if (plugins.length === 0 && !scanning) {
      scanPlugins();
    }
  }, [plugins.length, scanning, scanPlugins]);

  const filtered = useMemo(() => {
    if (!filter) return plugins;
    const q = filter.toLowerCase();
    return plugins.filter(
      p => p.name.toLowerCase().includes(q) ||
           p.vendor.toLowerCase().includes(q) ||
           p.subcategories.toLowerCase().includes(q)
    );
  }, [plugins, filter]);

  const inChain = useMemo(() => new Set(chain.map(p => p.uid)), [chain]);

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
          className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-zinc-800 border border-white/10 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20 outline-none transition-colors"
        />
      </div>

      {/* Plugin list */}
      <div className="max-h-48 overflow-y-auto space-y-0.5 scrollbar-thin">
        {scanning ? (
          <div className="flex items-center gap-2 justify-center py-4 text-zinc-500">
            <span className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs">Scanning VST3 plugins...</span>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-xs text-zinc-600 text-center py-3 italic">
            {plugins.length === 0 ? 'No VST3 plugins found' : 'No matches'}
          </div>
        ) : (
          filtered.map(plugin => (
            <button
              key={plugin.uid}
              disabled={inChain.has(plugin.uid)}
              onClick={() => { addToChain(plugin); onClose(); }}
              className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left transition-colors ${
                inChain.has(plugin.uid)
                  ? 'opacity-30 cursor-not-allowed'
                  : 'hover:bg-violet-500/10 cursor-pointer'
              }`}
            >
              <Plus size={12} className={`flex-shrink-0 ${inChain.has(plugin.uid) ? 'text-zinc-600' : 'text-violet-400'}`} />
              <div className="min-w-0 flex-1">
                <div className="text-xs text-zinc-200 truncate">{plugin.name}</div>
                <div className="text-[10px] text-zinc-500 truncate">{plugin.vendor} · {plugin.subcategories}</div>
              </div>
            </button>
          ))
        )}
      </div>

      {/* Rescan button */}
      <button
        onClick={scanPlugins}
        disabled={scanning}
        className="w-full text-[10px] text-zinc-600 hover:text-violet-400 transition-colors py-1"
      >
        {scanning ? 'Scanning...' : `Rescan (${plugins.length} found)`}
      </button>
    </div>
  );
};

// ── Chain Entry Row ─────────────────────────────────────────

// ChainRow uses inline type annotation below

const ChainRow: React.FC<{
  entry: { uid: string; name: string; vendor: string; path: string; enabled: boolean; statePath: string };
  index: number;
  total: number;
}> = ({ entry, index, total }) => {
  const { toggleEnabled, removeFromChain, reorderChain, openGui } = useVstChainStore();

  return (
    <div
      className={`flex items-center gap-1.5 px-2 py-1.5 rounded-lg border transition-all ${
        entry.enabled
          ? 'bg-violet-500/5 border-violet-500/20'
          : 'bg-zinc-800/50 border-white/5 opacity-50'
      }`}
    >
      {/* Drag handle placeholder + reorder buttons */}
      <div className="flex flex-col gap-0 flex-shrink-0">
        <button
          onClick={() => reorderChain(index, index - 1)}
          disabled={index === 0}
          className="p-0 text-zinc-600 hover:text-zinc-300 disabled:opacity-20 transition-colors"
        >
          <ChevronUp size={10} />
        </button>
        <button
          onClick={() => reorderChain(index, index + 1)}
          disabled={index >= total - 1}
          className="p-0 text-zinc-600 hover:text-zinc-300 disabled:opacity-20 transition-colors"
        >
          <ChevronDown size={10} />
        </button>
      </div>

      {/* Order number */}
      <span className="text-[10px] text-zinc-600 font-mono w-3 text-center flex-shrink-0">
        {index + 1}
      </span>

      {/* Plugin name */}
      <div className="min-w-0 flex-1">
        <div className="text-xs text-zinc-200 truncate">{entry.name}</div>
        <div className="text-[10px] text-zinc-500 truncate">{entry.vendor}</div>
      </div>

      {/* Actions */}
      <button
        onClick={() => openGui(entry)}
        title="Open plugin UI"
        className="p-1 rounded hover:bg-violet-500/10 text-zinc-500 hover:text-violet-400 transition-colors flex-shrink-0"
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
        title="Remove from chain"
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
    startMonitor, stopMonitor, pollMonitorStatus,
    monitorPosition, monitorDuration, seekMonitor,
  } = useVstChainStore();
  const { currentTrack, isPlaying } = usePlayback();
  const [showSearch, setShowSearch] = useState(false);

  useEffect(() => {
    if (!chainLoaded) loadChain();
  }, [chainLoaded, loadChain]);

  // Poll monitor status while monitoring (300ms for smooth transport)
  useEffect(() => {
    if (!monitoring) return;
    const id = setInterval(() => pollMonitorStatus(), 300);
    return () => clearInterval(id);
  }, [monitoring, pollMonitorStatus]);

  const enabledCount = chain.filter(p => p.enabled).length;
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

  return (
    <div className="space-y-3">
      {/* Current chain */}
      {chain.length > 0 ? (
        <div className="space-y-1">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-1">
            Plugin Chain ({enabledCount}/{chain.length} active)
          </div>
          {chain.map((entry, i) => (
            <ChainRow key={entry.uid} entry={entry} index={i} total={chain.length} />
          ))}
        </div>
      ) : (
        <div className="text-xs text-zinc-500 italic text-center py-2">
          No plugins in chain. Add one below.
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
                ? 'bg-zinc-800 border-white/10 text-zinc-400 hover:border-violet-500/30 hover:text-violet-400'
                : 'bg-zinc-800/50 border-white/5 text-zinc-600 cursor-not-allowed'
          }`}
        >
          {monitoring ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-500" />
              </span>
              <Square size={12} />
              Stop Monitor
            </>
          ) : (
            <>
              <Headphones size={14} />
              {hasTrack ? 'Monitor with VST Chain' : 'Play a track first'}
            </>
          )}
        </button>
      )}

      {/* Transport bar during monitoring */}
      {monitoring && monitorDuration > 0 && (
        <div className="space-y-1">
          <input
            type="range"
            min={0}
            max={monitorDuration}
            step={0.5}
            value={monitorPosition}
            onChange={e => seekMonitor(parseFloat(e.target.value))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, rgb(139 92 246) ${(monitorPosition / monitorDuration) * 100}%, rgb(63 63 70) ${(monitorPosition / monitorDuration) * 100}%)`,
            }}
          />
          <div className="flex justify-between text-[10px] text-zinc-500 font-mono">
            <span>{formatTime(monitorPosition)}</span>
            <span>{formatTime(monitorDuration)}</span>
          </div>
        </div>
      )}

      {/* Add plugin */}
      {showSearch ? (
        <div className="border-t border-white/5 pt-2">
          <PluginSearch onClose={() => setShowSearch(false)} />
        </div>
      ) : (
        <button
          onClick={() => setShowSearch(true)}
          className="w-full flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-xl bg-zinc-800 text-zinc-400 border border-white/10 hover:border-violet-500/30 hover:text-violet-400 transition-all"
        >
          <Plus size={14} /> Add Plugin
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
  const enabled = chain.filter(p => p.enabled);

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
