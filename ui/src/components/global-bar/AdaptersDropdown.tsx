// AdaptersDropdown.tsx — Adapter configuration UI for the global param bar
//
// Adapted from create/AdaptersAccordion.tsx to read from GlobalParamsContext.
// Self-contained with Simple and Advanced modes, file browser, group scales.

import React, { useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { FolderOpen, X, Tag, Search, Circle, ChevronDown, RotateCcw } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { adapterApi, modelApi } from '../../services/api';
import { FileBrowserModal } from '../shared/FileBrowserModal';
import { Slider } from '../shared/Slider';
import { ModelSelect, getModelFormat } from './ModelSelect';
import { formatDitModel } from './modelLabels';
import { DEFAULT_SETTINGS, type AppSettings } from '../settings/SettingsPanel';
import type { AdapterFile } from '../../types';

// Select styling applied inline where needed

const GROUP_INFO = [
  { key: 'self_attn' as const,  label: 'Self-Attn',    help: 'How audio frames relate to each other over time' },
  { key: 'cross_attn' as const, label: 'Cross-Attn',   help: 'How strongly your text prompt shapes the output' },
  { key: 'mlp' as const,        label: 'MLP',          help: 'Timbre, tonal texture, and sonic character' },
  { key: 'cond_embed' as const, label: 'Conditioning', help: 'How the adapter reshapes text/style interpretation' },
  { key: 'time_embed' as const, label: 'Timestep',     help: 'How the adapter modifies noise-schedule understanding (0 = skip)' },
  { key: 'proj_in' as const,    label: 'Proj-In',      help: 'Input patchification layer — how latent tokens enter the model (0 = skip)' },
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
  const { t } = useTranslation();
  const [settings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

  // Internal state
  const [adapterFiles, setAdapterFiles] = useState<AdapterFile[]>([]);
  const [showGroupScales, setShowGroupScales] = usePersistedState('hs-adapterAccordion-groupScales', false);
  const [fileBrowserOpen, setFileBrowserOpen] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [ditModels, setDitModels] = useState<string[]>([]);

  // DiT model list for the basin re-base "home base" selector.
  useEffect(() => {
    modelApi.list().then(m => setDitModels(m?.models?.dit || [])).catch(() => {});
  }, []);

  const fileBrowserMode = gp.advancedAdapters ? 'folder' as const : 'file' as const;
  // In advanced mode the stack drives everything; in simple mode the single adapter does.
  const stack: { path: string; scale: number; stepStart?: number; stepEnd?: number }[] = gp.adapterStack || [];

  // Any stack entry with a timestep window forces runtime mode server-side —
  // several knobs below change visibility/meaning when this is true.
  const stackHasWindows = stack.some(e => e.stepStart !== undefined || e.stepEnd !== undefined);

  // Timestep window helpers. Store fields stepStart/stepEnd are flow-matching t
  // (1 = noise, 0 = clean); the UI shows "% of denoising" (0% = first step),
  // so display = (1 − t) flipped: startPct derives from stepEnd and vice versa.
  const winStartPct = (e: { stepEnd?: number }) => Math.round((1 - (e.stepEnd ?? 1)) * 100);
  const winEndPct = (e: { stepStart?: number }) => Math.round((1 - (e.stepStart ?? 0)) * 100);
  const setWindowPct = (path: string, sPct: number, ePct: number) => {
    const lo = Math.max(0, Math.min(100, Math.min(sPct, ePct)));
    const hi = Math.max(0, Math.min(100, Math.max(sPct, ePct)));
    gp.setAdapterStackWindow(path, 1 - hi / 100, 1 - lo / 100);
  };
  const primaryPath = gp.advancedAdapters ? (stack[0]?.path || '') : gp.adapter;
  const hasAdapter = gp.advancedAdapters ? stack.length > 0 : !!gp.adapter;
  const triggerWord = deriveTriggerWord(primaryPath);
  // Every stacked adapter contributes its trigger word (matches what is injected
  // into the caption server-side).
  const stackTriggerWords = stack.map(e => deriveTriggerWord(e.path)).filter(Boolean).join(', ');
  const adapterFilename = gp.adapter ? gp.adapter.split(/[\\/]/).pop() || '' : '';
  const fileLabel = (p: string) => p.split(/[\\/]/).pop() || p;

  // Blend mode: per-adapter sliders are relative weights, normalised so the
  // effective scales sum to the combined-strength budget. effectiveScale maps a
  // row's weight to the scale actually sent to the engine (mirrors the store).
  const isBlend = gp.adapterStackMode === 'blend';
  // Sum/Blend distinction only matters with 2+ adapters; a single adapter just
  // has a "Strength".
  const multiStack = stack.length >= 2;
  const stackWeightSum = stack.reduce((acc, e) => acc + (e.scale || 0), 0);
  const effectiveScale = (weight: number) => {
    if (!isBlend || !multiStack) return weight;
    const budget = gp.adapterStackBudget ?? 0.75;
    return stackWeightSum > 0 ? (budget * (weight || 0)) / stackWeightSum : budget / Math.max(1, stack.length);
  };
  const GROUP_DEFAULTS: Record<string, number> = { self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0, time_embed: 0.0, proj_in: 0.0 };
  const allDefault = GROUP_INFO.every(g => gp.adapterGroupScales[g.key] === (GROUP_DEFAULTS[g.key] ?? 1.0));

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
      <div className="flex rounded-xl overflow-hidden border border-zinc-300 dark:border-white/10">
        <button
          onClick={() => gp.setAdvancedAdapters(false)}
          className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
            !gp.advancedAdapters ? 'bg-zinc-200 dark:bg-zinc-700 text-white' : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
          }`}
        >
          {t('common.simple')}
        </button>
        <button
          onClick={() => gp.setAdvancedAdapters(true)}
          className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors ${
            gp.advancedAdapters ? 'bg-zinc-200 dark:bg-zinc-700 text-white' : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
          }`}
        >
          {t('common.advanced')}
        </button>
      </div>

      {/* ═══ SIMPLE MODE ═══ */}
      {!gp.advancedAdapters && (
        <>
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('adapter.adapterPath')}</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={gp.adapter}
                onChange={(e) => gp.setAdapter(e.target.value)}
                placeholder="Path to .safetensors file..."
                className="flex-1 px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors font-mono text-xs"
              />
              <button
                onClick={() => setFileBrowserOpen(true)}
                className="px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors"
                title={t('adapter.browseFile')}
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
                className="text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors flex-shrink-0"
                title={t('adapter.clearAdapter')}
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
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('adapter.adapterFolder')}</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={gp.adapterFolder}
                onChange={(e) => gp.setAdapterFolder(e.target.value)}
                placeholder="Path to folder with adapters..."
                className="flex-1 px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors font-mono text-xs"
              />
              <button
                onClick={() => handleScan()}
                disabled={!gp.adapterFolder || scanning}
                className="px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title={t('adapter.scanFolder')}
              >
                <Search size={14} className={scanning ? 'animate-spin' : ''} />
              </button>
              <button
                onClick={() => setFileBrowserOpen(true)}
                className="px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors"
                title={t('adapter.browseFolder')}
              >
                <FolderOpen size={14} />
              </button>
            </div>
          </div>

          {scanError && (
            <div className="text-xs text-amber-400/70 px-1">{scanError}</div>
          )}

          {adapterFiles.length > 0 && (
            <div className="rounded-xl bg-zinc-50/80 dark:bg-zinc-900/50 border border-zinc-200 dark:border-white/5 overflow-hidden" style={{ maxHeight: '200px', overflowY: 'auto' }}>
              {adapterFiles.map((file) => {
                const isActive = stack.some(a => a.path === file.path);
                return (
                  <button
                    key={file.path}
                    onClick={() => gp.toggleAdapterInStack(file.path, 1.0)}
                    className={`w-full flex items-center gap-2 px-3 py-2 text-left transition-colors ${
                      isActive ? 'bg-emerald-500/10 border-l-2 border-emerald-500' : 'hover:bg-white/5 border-l-2 border-transparent'
                    }`}
                  >
                    {isActive ? (
                      <Circle size={8} fill="#10b981" className="text-emerald-500 flex-shrink-0" />
                    ) : (
                      <Circle size={8} className="text-zinc-600 flex-shrink-0" />
                    )}
                    <span className={`text-xs truncate flex-1 ${isActive ? 'text-emerald-400 font-medium' : 'text-zinc-600 dark:text-zinc-400'}`}>
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

          {/* Selected stack: one row per adapter with its own scale + remove. */}
          {stack.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-zinc-600 dark:text-zinc-400 uppercase tracking-wider">
                  {t('adapter.adapterStack', 'Adapter Stack')} ({stack.length})
                </span>
                <button type="button" onClick={() => gp.setAdapterStack([])}
                  className="text-[10px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors">
                  {t('adapter.clearAll', 'Clear all')}
                </button>
              </div>

              {/* Sum / Blend toggle — only meaningful with 2+ adapters */}
              {multiStack && (
                <>
                  <div className="flex rounded-xl overflow-hidden border border-zinc-300 dark:border-white/10">
                    <button
                      type="button"
                      onClick={() => gp.setAdapterStackMode('blend')}
                      className={`flex-1 px-2.5 py-1.5 text-xs font-medium transition-colors ${
                        isBlend ? 'bg-emerald-600 text-white' : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                      }`}
                    >
                      Blend
                    </button>
                    <button
                      type="button"
                      onClick={() => gp.setAdapterStackMode('sum')}
                      className={`flex-1 px-2.5 py-1.5 text-xs font-medium transition-colors ${
                        !isBlend ? 'bg-amber-600 text-white' : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                      }`}
                    >
                      Sum
                    </button>
                  </div>
                  <p className="text-[10px] text-zinc-600 -mt-1">
                    {isBlend
                      ? 'Per-adapter sliders are relative weights; effective scales are normalised so they sum to the combined strength below. Keeps total strength constant as you add adapters.'
                      : 'Per-adapter sliders are absolute scales, summed directly. Σ can exceed 1 to deliberately over-drive the stack.'}
                  </p>

                  {/* Combined strength budget (blend mode only) */}
                  {isBlend && (
                    <Slider label={t('adapter.combinedStrength', 'Combined Strength (Σ)')} value={gp.adapterStackBudget}
                      onChange={gp.setAdapterStackBudget} min={0} max={4} step={0.05} showInput />
                  )}
                </>
              )}

              {stack.map((entry, i) => (
                <div key={entry.path} className="px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-zinc-500 flex-shrink-0">{i + 1}.</span>
                    <span className="text-xs text-emerald-400 font-medium truncate flex-1" title={entry.path}>
                      {fileLabel(entry.path)}
                    </span>
                    {multiStack && isBlend && (
                      <span className="text-[10px] text-zinc-500 font-mono flex-shrink-0" title="Effective scale sent to the engine">
                        → {effectiveScale(entry.scale).toFixed(3)}
                      </span>
                    )}
                    <button
                      onClick={() => gp.toggleAdapterInStack(entry.path)}
                      className="text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors flex-shrink-0"
                      title={t('adapter.deselectAdapter')}
                    >
                      <X size={12} />
                    </button>
                  </div>
                  <Slider
                    label={!multiStack
                      ? t('adapter.strength', 'Strength')
                      : isBlend ? t('adapter.weight', 'Weight') : t('adapter.adapterScale', 'Adapter Scale')}
                    value={entry.scale}
                    onChange={v => gp.setAdapterStackScale(entry.path, v)} min={0} max={4} step={0.05} showInput />
                  {/* Timestep window (interval experts): which slice of denoising this
                      adapter is active in. 0% = first step (structure), 100% = last
                      (texture/detail). Any non-full window forces runtime mode. */}
                  <div className="flex items-center gap-1.5"
                    title={t('adapter.timestepWindowHint',
                      'Active slice of the denoising process. Early steps shape structure/rhythm, late steps shape timbre/detail. Windows crossfade where adapters meet. Forces runtime mode.')}>
                    <span className="text-[10px] text-zinc-500 flex-shrink-0">
                      {t('adapter.timestepWindow', 'Active phase')}
                    </span>
                    <input type="number" min={0} max={100} step={5} value={winStartPct(entry)}
                      onChange={e => setWindowPct(entry.path, Number(e.target.value), winEndPct(entry))}
                      className="w-12 px-1 py-0.5 text-[10px] text-right rounded bg-white dark:bg-black/30 border border-zinc-200 dark:border-white/10 text-zinc-700 dark:text-zinc-300" />
                    <span className="text-[10px] text-zinc-500">–</span>
                    <input type="number" min={0} max={100} step={5} value={winEndPct(entry)}
                      onChange={e => setWindowPct(entry.path, winStartPct(entry), Number(e.target.value))}
                      className="w-12 px-1 py-0.5 text-[10px] text-right rounded bg-white dark:bg-black/30 border border-zinc-200 dark:border-white/10 text-zinc-700 dark:text-zinc-300" />
                    <span className="text-[10px] text-zinc-500">%</span>
                    {(entry.stepStart !== undefined || entry.stepEnd !== undefined) && (
                      <button type="button"
                        onClick={() => gp.setAdapterStackWindow(entry.path, 0, 1)}
                        className="text-[10px] text-amber-400/80 hover:text-amber-300 flex-shrink-0"
                        title={t('adapter.timestepWindowReset', 'Reset to always active')}>
                        ⏱ {t('adapter.timestepWindowClear', 'clear')}
                      </button>
                    )}
                  </div>
                </div>
              ))}

              {stackHasWindows && (
                <div className="px-2.5 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <p className="text-[10px] text-amber-400/90 leading-relaxed m-0">
                    {t('adapter.timestepWindowVram',
                      'Timestep windows force Runtime mode: each adapter holds its own full-size deltas in VRAM. Set Adapter VRAM below to Q8 ½ or Q4 ¼ to keep this affordable.')}
                  </p>
                </div>
              )}

              {settings.triggerUseFilename && stackTriggerWords && (
                <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-pink-500/10 border border-pink-500/20">
                  <Tag size={10} className="text-pink-400 flex-shrink-0" />
                  <span className="text-[10px] text-pink-400 font-medium">{stackTriggerWords}</span>
                  <span className="text-[10px] text-zinc-500">({settings.triggerPlacement})</span>
                </div>
              )}

              {/* Per-section masking hint (2+ adapters) */}
              {multiStack && (
                <div className="px-2.5 py-2 rounded-lg bg-sky-500/5 border border-sky-500/15 space-y-1">
                  <div className="text-[10px] font-semibold text-sky-300/80 uppercase tracking-wider">Per-section influence</div>
                  <p className="text-[10px] text-zinc-500 leading-relaxed">
                    Vary each adapter by lyric section — add a directive after a section header,
                    keyed by trigger word. Forces runtime mode.
                  </p>
                  <pre className="text-[9px] text-zinc-400 font-mono whitespace-pre-wrap leading-snug bg-black/20 rounded p-1.5 m-0">{`[Verse]{${stackTriggerWords.split(', ').map((w, i) => `${w}=${i === 0 ? '1' : '0'}`).join('; ')}}
[Chorus]{${stackTriggerWords.split(', ').map((w, i) => `${w}=${i === 0 ? '0' : '1'}`).join('; ')}}`}</pre>

                  <div className="pt-1 space-y-2">
                    <Slider label="Alignment Timing" value={gp.adapterSectionAlignAt}
                      onChange={gp.setAdapterSectionAlignAt} min={0.2} max={0.85} step={0.05} showInput />
                    <p className="text-[9px] text-zinc-500 leading-relaxed -mt-1">
                      When section boundaries snap to the model's real timing. Earlier locks section
                      identity sooner (less first-adapter bias); too early = fuzzier boundaries.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* ═══ SHARED CONTROLS (when adapter selected) ═══ */}
      {hasAdapter && (
        <>
          {/* Adapter Scale — simple mode only; the advanced stack has per-row scales */}
          {!gp.advancedAdapters && (
            <Slider label="Adapter Scale" value={gp.adapterScale}
              onChange={gp.setAdapterScale} min={0} max={4} step={0.05} showInput />
          )}

          {/* Loading Mode */}
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('adapter.loadingMode')}</label>
            <div className="flex rounded-xl overflow-hidden border border-zinc-300 dark:border-white/10">
              <button
                type="button"
                onClick={() => gp.setAdapterMode('merge')}
                className={`flex-1 px-2.5 py-1.5 text-xs font-medium transition-colors ${
                  gp.adapterMode === 'merge' ? 'bg-amber-600 text-white' : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                Merge
              </button>
              <button
                type="button"
                onClick={() => gp.setAdapterMode('runtime')}
                className={`flex-1 px-2.5 py-1.5 text-xs font-medium transition-colors ${
                  gp.adapterMode === 'runtime' ? 'bg-pink-600 text-white' : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                Runtime ⚡
              </button>
              <button
                type="button"
                onClick={() => gp.setAdapterMode('runtime_lowrank')}
                className={`flex-1 px-2.5 py-1.5 text-xs font-medium transition-colors ${
                  gp.adapterMode === 'runtime_lowrank' ? 'bg-violet-600 text-white' : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                Low-Rank 🪶
              </button>
            </div>
            <p className="text-[10px] text-zinc-600 mt-1">
              {gp.adapterMode === 'runtime_lowrank'
                ? 'Applies raw adapter factors per-step, never materializing full deltas — lowest VRAM (LoRA & LoKr; DoRA needs Merge). Basin re-base still works.'
                : gp.adapterMode === 'runtime'
                ? 'Keeps base weights intact, applies adapter per-step. Same quality, slower inference, saves VRAM.'
                : 'Merges adapter at F32 precision. Best quality, fast inference, but uses more VRAM during synthesis.'}
            </p>
          </div>

          {/* Merge VRAM — merge mode only (storage precision of the merged weights) */}
          {gp.adapterMode === 'merge' && (
            <div>
              <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
                {t('adapter.mergeVram', 'Merge VRAM')}
              </label>
              <div className="flex rounded-xl overflow-hidden border border-zinc-300 dark:border-white/10">
                {([
                  { v: false, label: 'HQ',      sub: 'Merged weights stored as F32 (best quality, ~4× VRAM on a Q8 base)' },
                  { v: true,  label: 'Low ¼',   sub: 'Merged weights re-encoded to the base\'s native quant' },
                ] as const).map(opt => (
                  <button
                    key={String(opt.v)}
                    type="button"
                    onClick={() => gp.setAdapterMergeLowVram(opt.v)}
                    title={opt.sub}
                    className={`flex-1 px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      gp.adapterMergeLowVram === opt.v
                        ? 'bg-sky-600 text-white'
                        : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-zinc-600 mt-1">
                HQ keeps merged weights at F32 (a Q8 base grows ~4× in VRAM). Low re-encodes them
                back to the base&apos;s native quant — base-model VRAM, one extra quantization
                round-trip. FP4 bases always use the low path.
              </p>
            </div>
          )}

          {/* Adapter Quantization — runtime modes (quantizes the in-VRAM full-size
              deltas; in Low-Rank mode that's the re-base correction + Conv1d fallbacks).
              Also shown when timestep windows are set: windows force runtime mode
              server-side, so this knob governs VRAM even from Merge/Low-Rank. */}
          {(gp.adapterMode === 'runtime' || gp.adapterMode === 'runtime_lowrank' || stackHasWindows) && (
            <div>
              <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
                {t('adapter.runtimeQuant', 'Adapter VRAM')}
              </label>
              <div className="flex rounded-xl overflow-hidden border border-zinc-300 dark:border-white/10">
                {([
                  { v: 'bf16', label: 'Full',   sub: 'BF16' },
                  { v: 'q8_0', label: 'Q8 ½',   sub: 'Q8_0' },
                  { v: 'q4_0', label: 'Q4 ¼',   sub: 'Q4_0' },
                ] as const).map(opt => (
                  <button
                    key={opt.v}
                    type="button"
                    onClick={() => gp.setAdapterRuntimeQuant(opt.v)}
                    title={opt.sub}
                    className={`flex-1 px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      gp.adapterRuntimeQuant === opt.v
                        ? 'bg-sky-600 text-white'
                        : 'bg-white dark:bg-zinc-900 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-zinc-600 mt-1">
                Quantizes the runtime adapter deltas in VRAM (nothing written to disk). Q8 halves /
                Q4 quarters VRAM per adapter — lets more stacked adapters fit. Small quality cost;
                safe when the base model is already 4-bit (NVFP4).
              </p>
            </div>
          )}

          {/* Basin re-base (cross-base adapter support) — merge AND runtime modes
              (runtime folds the nudge into the delta sum; per-section masking skips it).
              Home base must be a SafeTensors model (nudge reads F32 weights), so
              the selector is filtered to safetensors DiT models only. */}
          <div className="rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 p-3 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-semibold text-zinc-600 dark:text-zinc-400 uppercase tracking-wider">Basin Re-base</span>
                  {gp.rebaseSource && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" title="Re-base active" />}
                </div>
                {gp.rebaseSource && (
                  <button type="button" onClick={() => gp.setRebaseSource('')}
                    className="text-[10px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors">
                    Off
                  </button>
                )}
              </div>
              <label className="block text-[10px] text-zinc-500">Adapter trained on (home base)</label>
              <ModelSelect
                id="rebase-source-select"
                value={gp.rebaseSource}
                onChange={gp.setRebaseSource}
                options={ditModels.filter(m => getModelFormat(m) === 'safetensors')}
                formatLabel={formatDitModel}
                placeholder="Off — apply adapter as-is"
              />
              {gp.rebaseSource && (
                <>
                  <Slider label="Re-base Strength (β)" value={gp.rebaseBeta}
                    onChange={gp.setRebaseBeta} min={0} max={1} step={0.05} showInput />
                  <p className="text-[10px] text-zinc-600">
                    Nudges the loaded base toward the adapter's home base so a heavy cross-base adapter
                    stays coherent at full strength. β=1 = home-base behavior; lower keeps more of the
                    loaded base's character.
                  </p>
                </>
              )}
          </div>

          {/* Group Scales */}
          <button
            onClick={() => setShowGroupScales(!showGroupScales)}
            className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
          >
            <ChevronDown size={12} className={`transition-transform duration-200 ${showGroupScales ? 'rotate-180' : ''}`} />
            {t('adapter.groupScales')}
            {!allDefault && (
              <span className="w-1.5 h-1.5 rounded-full bg-pink-500" title="Group scales modified" />
            )}
          </button>

          {showGroupScales && (
            <div className="rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 p-3 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-zinc-600 dark:text-zinc-400 uppercase tracking-wider">{t('adapter.layerScales')}</span>
                <button type="button" onClick={() => gp.setAdapterGroupScales({ self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0, time_embed: 0.0, proj_in: 0.0 })}
                  className="flex items-center gap-1 text-[10px] text-zinc-600 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors">
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
        title={gp.advancedAdapters ? t('adapter.selectAdapterFolder') : t('adapter.selectAdapterFile')}
      />
    </div>
  );
};

/** Summary badge for the Adapters section */
export const AdaptersBadge: React.FC = () => {
  const { adapter, adapterScale, adapterStack, advancedAdapters } = useGlobalParams();
  const stack = adapterStack || [];
  const useStack = advancedAdapters && stack.length > 0;
  const shortName = (p: string) => p.split(/[\\/]/).pop()?.replace(/\.safetensors$/i, '') || '';

  if (useStack) {
    const first = shortName(stack[0].path);
    return (
      <div className="flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
        <span className="text-[10px] text-emerald-400 font-mono truncate max-w-[120px]" title={stack.map(a => a.path).join('\n')}>
          {first}
        </span>
        {stack.length > 1 && <span className="text-[10px] text-zinc-600">+{stack.length - 1}</span>}
      </div>
    );
  }

  const filename = adapter ? shortName(adapter) : '';
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
