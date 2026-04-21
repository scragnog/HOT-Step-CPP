// AdaptersAccordion.tsx — Adapter selection and configuration
//
// Self-contained accordion with Simple and Advanced modes.
// Simple Mode:  path input + browse + scale + loading mode
// Advanced Mode: folder scan + file list + group scales
//
// Replaces the adapter section that was previously inside ModelSelector.

import React, { useState, useCallback } from 'react';
import { ChevronDown, FolderOpen, X, Tag, Search, Circle } from 'lucide-react';
import { adapterApi } from '../../services/api';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { useLanguage } from '../../context/LanguageContext';
import type { AdapterFile } from '../../types';

interface AdapterGroupScales {
  self_attn: number;
  cross_attn: number;
  mlp: number;
  cond_embed: number;
}

interface AdaptersAccordionProps {
  // Mode
  advancedAdapters: boolean;
  onAdvancedAdaptersChange: (val: boolean) => void;

  // Adapter selection (shared between modes)
  adapter: string;
  onAdapterChange: (path: string) => void;
  adapterScale: number;
  onAdapterScaleChange: (val: number) => void;
  adapterMode: string;
  onAdapterModeChange: (val: string) => void;

  // Group scales (advanced only)
  adapterGroupScales: AdapterGroupScales;
  onAdapterGroupScalesChange: (v: AdapterGroupScales) => void;

  // Folder scanning (advanced)
  adapterFolder: string;
  onAdapterFolderChange: (val: string) => void;

  // Trigger word display (read from settings)
  triggerUseFilename: boolean;
  triggerPlacement: 'prepend' | 'append' | 'replace';
  onTriggerPlacementChange?: (v: 'prepend' | 'append' | 'replace') => void;
}

const GROUP_INFO = [
  { key: 'self_attn' as const,  label: 'Self-Attn',    help: 'How audio frames relate to each other over time' },
  { key: 'cross_attn' as const, label: 'Cross-Attn',   help: 'How strongly your text prompt shapes the output' },
  { key: 'mlp' as const,        label: 'MLP',          help: 'Timbre, tonal texture, and sonic character' },
  { key: 'cond_embed' as const, label: 'Conditioning', help: 'How the adapter reshapes text/style interpretation' },
];

/** Extract trigger word from adapter path — filename without extension, underscores kept */
function deriveTriggerWord(adapterPath: string): string {
  if (!adapterPath) return '';
  const filename = adapterPath.split(/[\\/]/).pop() || '';
  return filename.replace(/\.safetensors$/i, '');
}

/** Format byte count */
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export const AdaptersAccordion: React.FC<AdaptersAccordionProps> = ({
  advancedAdapters, onAdvancedAdaptersChange,
  adapter, onAdapterChange,
  adapterScale, onAdapterScaleChange,
  adapterMode, onAdapterModeChange,
  adapterGroupScales, onAdapterGroupScalesChange,
  adapterFolder, onAdapterFolderChange,
  triggerUseFilename, triggerPlacement, onTriggerPlacementChange,
}) => {
  const { t } = useLanguage();
  // Internal state
  const [adapterFiles, setAdapterFiles] = useState<AdapterFile[]>([]);
  const [showGroupScales, setShowGroupScales] = useState(false);
  const [fileBrowserOpen, setFileBrowserOpen] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  // The FileBrowserModal mode depends on the adapter mode
  // Simple: select a file
  // Advanced: select a folder
  const fileBrowserMode = advancedAdapters ? 'folder' as const : 'file' as const;

  const triggerWord = deriveTriggerWord(adapter);
  const adapterFilename = adapter ? adapter.split(/[\\/]/).pop() || '' : '';

  const handleGroupScaleChange = (key: keyof AdapterGroupScales, value: number) => {
    onAdapterGroupScalesChange({ ...adapterGroupScales, [key]: value });
  };

  const allDefault = GROUP_INFO.every(g => adapterGroupScales[g.key] === 1.0);

  // Scan folder for adapter files (Advanced mode)
  const handleScan = useCallback(async (folder?: string) => {
    const dir = folder || adapterFolder;
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
  }, [adapterFolder]);

  // Handle file browser selection
  const handleBrowseSelect = (path: string) => {
    setFileBrowserOpen(false);
    if (advancedAdapters) {
      // Folder mode — set folder path and auto-scan
      onAdapterFolderChange(path);
      handleScan(path);
    } else {
      // File mode — set adapter path directly
      onAdapterChange(path);
    }
  };

  return (
    <div className="space-y-3 pt-3 border-t border-white/5">
      <div className="flex items-center gap-2 px-3 py-2.5">
        <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">{t('section_adapters')}</span>
        {adapter && (
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" title="Adapter active" />
        )}
      </div>

      <div className="px-3 pb-3 space-y-3">
        {/* Simple / Advanced toggle */}
        <div className="flex rounded-xl overflow-hidden border border-white/10">
          <button
            onClick={() => onAdvancedAdaptersChange(false)}
            className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
              !advancedAdapters ? 'bg-zinc-700 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
            }`}
          >
            Simple
          </button>
          <button
            onClick={() => onAdvancedAdaptersChange(true)}
            className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
              advancedAdapters ? 'bg-zinc-700 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
            }`}
          >
            Advanced
          </button>
        </div>

        {/* ═══ SIMPLE MODE ═══ */}
        {!advancedAdapters && (
          <>
            {/* Path input + Browse */}
            <div>
              <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Adapter Path</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={adapter}
                  onChange={(e) => onAdapterChange(e.target.value)}
                  placeholder="Path to .safetensors file..."
                  className="flex-1 px-3 py-2 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 placeholder-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors font-mono text-xs"
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

            {/* Selected adapter indicator */}
            {adapter && (
              <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                <Circle size={8} fill="#10b981" className="text-emerald-500 flex-shrink-0" />
                <span className="text-xs text-emerald-400 font-medium truncate flex-1" title={adapter}>
                  {adapterFilename}
                </span>
                <button
                  onClick={() => onAdapterChange('')}
                  className="text-zinc-500 hover:text-zinc-300 transition-colors flex-shrink-0"
                  title="Clear adapter"
                >
                  <X size={12} />
                </button>
              </div>
            )}

            {/* Trigger word tag + placement selector */}
            {adapter && triggerUseFilename && triggerWord && (
              <div className="space-y-1.5">
                <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-pink-500/10 border border-pink-500/20">
                  <Tag size={10} className="text-pink-400 flex-shrink-0" />
                  <span className="text-[10px] text-pink-400 font-medium flex-1">{triggerWord}</span>
                </div>
                {onTriggerPlacementChange && (
                  <div className="flex rounded-lg overflow-hidden border border-white/10">
                    {(['prepend', 'append', 'replace'] as const).map((p) => (
                      <button
                        key={p}
                        onClick={() => onTriggerPlacementChange(p)}
                        className={`flex-1 px-2 py-1 text-[10px] font-medium transition-colors capitalize ${
                          triggerPlacement === p ? 'bg-pink-600 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                        }`}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Adapter Scale */}
            {adapter && (
              <>
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Adapter Scale</label>
                    <span className="text-xs text-zinc-400 font-mono">{adapterScale.toFixed(2)}</span>
                  </div>
                  <input type="range" value={adapterScale}
                    onChange={e => onAdapterScaleChange(parseFloat(e.target.value))}
                    min={0} max={4} step={0.05} className="w-full" />
                </div>

                {/* Loading Mode */}
                <div>
                  <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Loading Mode</label>
                  <div className="flex rounded-xl overflow-hidden border border-white/10">
                    <button
                      onClick={() => onAdapterModeChange('merge')}
                      className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                        adapterMode === 'merge' ? 'bg-zinc-700 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                      }`}
                    >
                      Merge
                    </button>
                    <button
                      onClick={() => onAdapterModeChange('runtime')}
                      className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                        adapterMode === 'runtime' ? 'bg-pink-600 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                      }`}
                    >
                      Runtime LoRA ⚡
                    </button>
                  </div>
                  <p className="text-[10px] text-zinc-600 mt-1">
                    {adapterMode === 'runtime'
                      ? 'Fast: keeps base weights quantized, applies adapter at inference (~5s load)'
                      : 'Classic: merges adapter into weights (slow for K-quant models, ~127s)'}
                  </p>
                </div>
              </>
            )}
          </>
        )}

        {/* ═══ ADVANCED MODE ═══ */}
        {advancedAdapters && (
          <>
            {/* Folder path + Scan + Browse */}
            <div>
              <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Adapter Folder</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={adapterFolder}
                  onChange={(e) => onAdapterFolderChange(e.target.value)}
                  placeholder="Path to folder with adapters..."
                  className="flex-1 px-3 py-2 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 placeholder-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors font-mono text-xs"
                />
                <button
                  onClick={() => handleScan()}
                  disabled={!adapterFolder || scanning}
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

            {/* Scan error */}
            {scanError && (
              <div className="text-xs text-amber-400/70 px-1">
                {scanError}
              </div>
            )}

            {/* Available adapters list */}
            {adapterFiles.length > 0 && (
              <div className="rounded-xl bg-zinc-900/50 border border-white/5 overflow-hidden" style={{ maxHeight: '200px', overflowY: 'auto' }}>
                {adapterFiles.map((file) => {
                  const isActive = adapter === file.path;
                  return (
                    <button
                      key={file.path}
                      onClick={() => onAdapterChange(isActive ? '' : file.path)}
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
                      <span
                        className={`text-[10px] font-semibold flex-shrink-0 ${isActive ? 'text-emerald-500' : 'text-zinc-600'}`}
                      >
                        {isActive ? 'Active' : 'Select'}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}

            {/* Selected adapter details (Advanced) */}
            {adapter && (
              <>
                {/* Selected indicator */}
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                  <Circle size={8} fill="#10b981" className="text-emerald-500 flex-shrink-0" />
                  <span className="text-xs text-emerald-400 font-medium truncate flex-1" title={adapter}>
                    {adapterFilename}
                  </span>
                  <button
                    onClick={() => onAdapterChange('')}
                    className="text-zinc-500 hover:text-zinc-300 transition-colors flex-shrink-0"
                    title="Deselect adapter"
                  >
                    <X size={12} />
                  </button>
                </div>

                {/* Trigger word tag + placement selector */}
                {triggerUseFilename && triggerWord && (
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-pink-500/10 border border-pink-500/20">
                      <Tag size={10} className="text-pink-400 flex-shrink-0" />
                      <span className="text-[10px] text-pink-400 font-medium flex-1">{triggerWord}</span>
                    </div>
                    {onTriggerPlacementChange && (
                      <div className="flex rounded-lg overflow-hidden border border-white/10">
                        {(['prepend', 'append', 'replace'] as const).map((p) => (
                          <button
                            key={p}
                            onClick={() => onTriggerPlacementChange(p)}
                            className={`flex-1 px-2 py-1 text-[10px] font-medium transition-colors capitalize ${
                              triggerPlacement === p ? 'bg-pink-600 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                            }`}
                          >
                            {p}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Adapter Scale */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Adapter Scale</label>
                    <span className="text-xs text-zinc-400 font-mono">{adapterScale.toFixed(2)}</span>
                  </div>
                  <input type="range" value={adapterScale}
                    onChange={e => onAdapterScaleChange(parseFloat(e.target.value))}
                    min={0} max={4} step={0.05} className="w-full" />
                </div>

                {/* Loading Mode */}
                <div>
                  <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Loading Mode</label>
                  <div className="flex rounded-xl overflow-hidden border border-white/10">
                    <button
                      onClick={() => onAdapterModeChange('merge')}
                      className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                        adapterMode === 'merge' ? 'bg-zinc-700 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                      }`}
                    >
                      Merge
                    </button>
                    <button
                      onClick={() => onAdapterModeChange('runtime')}
                      className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
                        adapterMode === 'runtime' ? 'bg-pink-600 text-white' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                      }`}
                    >
                      Runtime LoRA ⚡
                    </button>
                  </div>
                  <p className="text-[10px] text-zinc-600 mt-1">
                    {adapterMode === 'runtime'
                      ? 'Fast: keeps base weights quantized, applies adapter at inference (~5s load)'
                      : 'Classic: merges adapter into weights (slow for K-quant models, ~127s)'}
                  </p>
                </div>

                {/* Group Scales Toggle (Advanced only) */}
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
                  <div className="rounded-xl bg-zinc-900/50 border border-white/5 p-3 space-y-3">
                    {GROUP_INFO.map(({ key, label, help }) => (
                      <div key={key}>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-xs text-zinc-500" title={help}>{label}</label>
                          <span className={`text-xs font-mono ${
                            adapterGroupScales[key] === 1.0 ? 'text-zinc-600' : 'text-pink-400'
                          }`}>
                            {adapterGroupScales[key].toFixed(2)}
                          </span>
                        </div>
                        <input
                          type="range"
                          value={adapterGroupScales[key]}
                          onChange={e => handleGroupScaleChange(key, parseFloat(e.target.value))}
                          min={0} max={4} step={0.05}
                          className="w-full"
                        />
                      </div>
                    ))}
                    <div className="text-center text-[10px] text-zinc-600 mt-1">
                      Scale changes apply on next generation (DiT reload)
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>

      {/* File Browser Modal */}
      <FileBrowserModal
        open={fileBrowserOpen}
        onClose={() => setFileBrowserOpen(false)}
        onSelect={handleBrowseSelect}
        mode={fileBrowserMode}
        startPath={advancedAdapters ? adapterFolder : undefined}
        filter="adapters"
        title={advancedAdapters ? 'Select Adapter Folder' : 'Select Adapter File'}
      />
    </div>
  );
};
