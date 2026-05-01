// AdaptersDropdown.tsx — Adapter configuration UI for the global param bar
//
// Adapted from create/AdaptersAccordion.tsx to read from GlobalParamsContext.
// Self-contained with Simple and Advanced modes, file browser, group scales.

import React, { useState, useCallback } from 'react';
import { FolderOpen, X, Tag, Search, Circle, ChevronDown, RotateCcw } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { adapterApi } from '../../services/api';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { Slider } from '../shared/Slider';
import { DEFAULT_SETTINGS, type AppSettings } from '../settings/SettingsPanel';
import type { AdapterFile } from '../../types';

// Select styling applied inline where needed

const GROUP_INFO = [
  { key: 'self_attn' as const,  label: 'Self-Attn',    help: 'How audio frames relate to each other over time' },
  { key: 'cross_attn' as const, label: 'Cross-Attn',   help: 'How strongly your text prompt shapes the output' },
  { key: 'mlp' as const,        label: 'MLP',          help: 'Timbre, tonal texture, and sonic character' },
  { key: 'cond_embed' as const, label: 'Conditioning', help: 'How the adapter reshapes text/style interpretation' },
];

function deriveTriggerWord(adapterPath: string): string {
  if (!adapterPath) return '';
  const filename = adapterPath.split(/[\\/]/).pop() || '';
  return filename.replace(/\.safetensors$/i, '');
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export const AdaptersDropdown: React.FC = () => {
  const gp = useGlobalParams();
  const [settings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

  // Internal state
  const [adapterFiles, setAdapterFiles] = useState<AdapterFile[]>([]);
  const [showGroupScales, setShowGroupScales] = usePersistedState('hs-adapterAccordion-groupScales', false);
  const [fileBrowserOpen, setFileBrowserOpen] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  const fileBrowserMode = gp.advancedAdapters ? 'folder' as const : 'file' as const;
  const triggerWord = deriveTriggerWord(gp.adapter);
  const adapterFilename = gp.adapter ? gp.adapter.split(/[\\/]/).pop() || '' : '';
  const allDefault = GROUP_INFO.every(g => gp.adapterGroupScales[g.key] === 1.0);

  const handleGroupScaleChange = (key: keyof typeof gp.adapterGroupScales, value: number) => {
    gp.setAdapterGroupScales({ ...gp.adapterGroupScales, [key]: value });
  };

  const handleScan = useCallback(async (folder?: string) => {
    const dir = folder || gp.adapterFolder;
    if (!dir) return;
    setScanning(true);
    setScanError(null);
    try {
      const result = await adapterApi.scan(dir);
      setAdapterFiles(result.files);
      if (result.files.length === 0) {
        setScanError('No .safetensors files found in this folder');
      }
    } catch (err: any) {
      setScanError(err?.message || 'Failed to scan folder');
    } finally {
      setScanning(false);
    }
  }, [gp.adapterFolder]);

  const handleBrowseSelect = (path: string) => {
    setFileBrowserOpen(false);
    if (gp.advancedAdapters) {
      gp.setAdapterFolder(path);
      handleScan(path);
    } else {
      gp.setAdapter(path);
    }
  };

  return (
    <div className="space-y-3">
      {/* Simple / Advanced toggle */}
      <div className="flex rounded-xl overflow-hidden border border-white/10">
        <button
          onClick={() => gp.setAdvancedAdapters(false)}
          className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
            !gp.advancedAdapters ? 'bg-zinc-700 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
          }`}
        >
          Simple
        </button>
        <button
          onClick={() => gp.setAdvancedAdapters(true)}
          className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
            gp.advancedAdapters ? 'bg-zinc-700 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
          }`}
        >
          Advanced
        </button>
      </div>

      {/* ═══ SIMPLE MODE ═══ */}
      {!gp.advancedAdapters && (
        <>
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Adapter Path</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={gp.adapter}
                onChange={(e) => gp.setAdapter(e.target.value)}
                placeholder="Path to .safetensors file..."
                className="flex-1 px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-zinc-200 placeholder-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors font-mono text-xs"
              />
              <button
                onClick={() => setFileBrowserOpen(true)}
                className="px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 transition-colors"
                title="Browse for adapter file"
              >
                <FolderOpen size={14} />
              </button>
            </div>
          </div>

          {gp.adapter && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
              <Circle size={8} fill="#10b981" className="text-emerald-500 flex-shrink-0" />
              <span className="text-xs text-emerald-400 font-medium truncate flex-1" title={gp.adapter}>
                {adapterFilename}
              </span>
              <button
                onClick={() => gp.setAdapter('')}
                className="text-zinc-500 hover:text-zinc-300 transition-colors flex-shrink-0"
                title="Clear adapter"
              >
                <X size={12} />
              </button>
            </div>
          )}

          {gp.adapter && settings.triggerUseFilename && triggerWord && (
            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-pink-500/10 border border-pink-500/20">
              <Tag size={10} className="text-pink-400 flex-shrink-0" />
              <span className="text-[10px] text-pink-400 font-medium">{triggerWord}</span>
              <span className="text-[10px] text-zinc-500">({settings.triggerPlacement})</span>
            </div>
          )}
        </>
      )}

      {/* ═══ ADVANCED MODE ═══ */}
      {gp.advancedAdapters && (
        <>
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Adapter Folder</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={gp.adapterFolder}
                onChange={(e) => gp.setAdapterFolder(e.target.value)}
                placeholder="Path to folder with adapters..."
                className="flex-1 px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-zinc-200 placeholder-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors font-mono text-xs"
              />
              <button
                onClick={() => handleScan()}
                disabled={!gp.adapterFolder || scanning}
                className="px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="Scan folder"
              >
                <Search size={14} className={scanning ? 'animate-spin' : ''} />
              </button>
              <button
                onClick={() => setFileBrowserOpen(true)}
                className="px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 transition-colors"
                title="Browse for folder"
              >
                <FolderOpen size={14} />
              </button>
            </div>
          </div>

          {scanError && (
            <div className="text-xs text-amber-400/70 px-1">{scanError}</div>
          )}

          {adapterFiles.length > 0 && (
            <div className="rounded-xl bg-zinc-900/50 border border-white/5 overflow-hidden" style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {adapterFiles.map((file) => {
                const isActive = gp.adapter === file.path;
                return (
                  <button
                    key={file.path}
                    onClick={() => gp.setAdapter(isActive ? '' : file.path)}
                    className={`w-full flex items-center gap-2 px-3 py-2 text-left transition-colors ${
                      isActive ? 'bg-emerald-500/10 border-l-2 border-emerald-500' : 'hover:bg-white/5 border-l-2 border-transparent'
                    }`}
                  >
                    {isActive ? (
                      <Circle size={8} fill="#10b981" className="text-emerald-500 flex-shrink-0" />
                    ) : (
                      <Circle size={8} className="text-zinc-600 flex-shrink-0" />
                    )}
                    <span className={`text-xs truncate flex-1 ${isActive ? 'text-emerald-400 font-medium' : 'text-zinc-400'}`}>
                      {file.name}
                    </span>
                    <span className="text-zinc-600 flex-shrink-0" style={{ fontSize: '10px' }}>
                      {formatSize(file.size)}
                    </span>
                  </button>
                );
              })}
            </div>
          )}

          {gp.adapter && (
            <>
              <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                <Circle size={8} fill="#10b981" className="text-emerald-500 flex-shrink-0" />
                <span className="text-xs text-emerald-400 font-medium truncate flex-1" title={gp.adapter}>
                  {adapterFilename}
                </span>
                <button
                  onClick={() => gp.setAdapter('')}
                  className="text-zinc-500 hover:text-zinc-300 transition-colors flex-shrink-0"
                  title="Deselect adapter"
                >
                  <X size={12} />
                </button>
              </div>

              {settings.triggerUseFilename && triggerWord && (
                <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-pink-500/10 border border-pink-500/20">
                  <Tag size={10} className="text-pink-400 flex-shrink-0" />
                  <span className="text-[10px] text-pink-400 font-medium">{triggerWord}</span>
                  <span className="text-[10px] text-zinc-500">({settings.triggerPlacement})</span>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ═══ SHARED CONTROLS (when adapter selected) ═══ */}
      {gp.adapter && (
        <>
          {/* Adapter Scale */}
          <Slider label="Adapter Scale" value={gp.adapterScale}
            onChange={gp.setAdapterScale} min={0} max={4} step={0.05} showInput />

          {/* Loading Mode */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Loading Mode</label>
            <div className="flex rounded-xl overflow-hidden border border-white/10">
              <button
                onClick={() => gp.setAdapterMode('merge')}
                className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                  gp.adapterMode === 'merge' ? 'bg-zinc-700 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Merge
              </button>
              <button
                onClick={() => gp.setAdapterMode('runtime')}
                className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                  gp.adapterMode === 'runtime' ? 'bg-pink-600 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Runtime LoRA ⚡
              </button>
            </div>
            <p className="text-[10px] text-zinc-600 mt-1">
              {gp.adapterMode === 'runtime'
                ? 'Fast: keeps base weights quantized, applies adapter at inference (~5s load)'
                : 'Classic: merges adapter into weights (slow for K-quant models, ~127s)'}
            </p>
          </div>

          {/* Group Scales */}
          <button
            onClick={() => setShowGroupScales(!showGroupScales)}
            className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <ChevronDown size={12} className={`transition-transform duration-200 ${showGroupScales ? 'rotate-180' : ''}`} />
            Group Scales
            {!allDefault && (
              <span className="w-1.5 h-1.5 rounded-full bg-pink-500" title="Group scales modified" />
            )}
          </button>

          {showGroupScales && (
            <div className="rounded-xl bg-zinc-800/50 border border-white/5 p-3 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">Layer Scales</span>
                <button type="button" onClick={() => gp.setAdapterGroupScales({ self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0 })}
                  className="flex items-center gap-1 text-[10px] text-zinc-400 hover:text-zinc-300 transition-colors">
                  <RotateCcw size={10} /> Reset
                </button>
              </div>
              {GROUP_INFO.map(({ key, label, help }) => (
                <div key={key}>
                  <Slider label={label} value={gp.adapterGroupScales[key]}
                    onChange={v => handleGroupScaleChange(key, v)} min={0} max={4} step={0.05} showInput />
                  <p className="text-[10px] text-zinc-600 mt-0.5 -mb-1">{help}</p>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* File Browser Modal */}
      <FileBrowserModal
        open={fileBrowserOpen}
        onClose={() => setFileBrowserOpen(false)}
        onSelect={handleBrowseSelect}
        mode={fileBrowserMode}
        startPath={gp.advancedAdapters ? gp.adapterFolder : undefined}
        filter="adapters"
        title={gp.advancedAdapters ? 'Select Adapter Folder' : 'Select Adapter File'}
      />
    </div>
  );
};

/** Summary badge for the Adapters section */
export const AdaptersBadge: React.FC = () => {
  const { adapter, adapterScale } = useGlobalParams();
  const filename = adapter ? adapter.split(/[\\/]/).pop()?.replace(/\.safetensors$/i, '') || '' : '';

  return (
    <div className="flex items-center gap-1.5">
      {adapter ? (
        <>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
          <span className="text-[10px] text-emerald-400 font-mono">{filename}</span>
          <span className="text-[10px] text-zinc-600">×{adapterScale.toFixed(2)}</span>
        </>
      ) : (
        <span className="text-[10px] text-zinc-600">None</span>
      )}
    </div>
  );
};
