// SourcePanel.tsx — Left panel: source audio upload + metadata + analysis
import React, { useCallback, useState, useEffect } from 'react';
import { Upload, Music, Loader2, X, Layers, Volume2, FolderOpen } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { AudioMetadata, AudioAnalysis } from './coverStudioUtils';
import { ALL_KEYS } from './coverStudioUtils';
import { SEPARATION_LEVELS } from '../../services/supersepApi';
import { LatentImport, type LatentMetadata } from '../shared/LatentImport';
import { masteringApi } from '../../services/api';
import { VOCAL_LANGUAGES } from '../../constants/languages';
import { ProviderSelector } from '../lyric-studio/ProviderSelector';

interface SourcePanelProps {
  sourceFileName: string;
  metadata: AudioMetadata | null;
  analysis: AudioAnalysis | null;
  isUploading: boolean;
  isAnalyzing: boolean;
  onFileSelected: (file: File) => void;
  onClear: () => void;
  bpmCorrection: number;
  onBpmCorrectionChange: (v: number) => void;
  bpmOverride: number | null;
  onBpmOverrideChange: (v: number | null) => void;
  keyOverride: string | null;
  onKeyOverrideChange: (v: string | null) => void;
  vocalLanguage: string;
  onVocalLanguageChange: (v: string) => void;
  advancedMode: boolean;
  onAdvancedModeChange: (v: boolean) => void;
  sepLevel: number;
  onSepLevelChange: (v: number) => void;
  isSeparating: boolean;
  sepProgress: number;
  sepMessage: string;
  sourceAudioUrl: string;
  onSeparate: () => void;
  hasStems: boolean;
  onConfigureStems: () => void;
  // Latent import
  sourceLatentUrl: string;
  onLatentLoaded: (url: string, meta: LatentMetadata) => void;
  onLatentClear: () => void;
  // Timbre reference override
  timbreOverridePath: string;
  onTimbreOverridePathChange: (v: string) => void;
  token: string | null;
  // Caption LLM selector
  coverCaptionProvider: string;
  coverCaptionModel: string;
  onCoverCaptionProviderChange: (provider: string) => void;
  onCoverCaptionModelChange: (model: string) => void;
}

export const SourcePanel: React.FC<SourcePanelProps> = ({
  sourceFileName, metadata, analysis, isUploading, isAnalyzing,
  onFileSelected, onClear,
  bpmCorrection, onBpmCorrectionChange,
  bpmOverride, onBpmOverrideChange,
  keyOverride, onKeyOverrideChange,
  vocalLanguage, onVocalLanguageChange,
  advancedMode, onAdvancedModeChange,
  sepLevel, onSepLevelChange,
  isSeparating, sepProgress, sepMessage,
  sourceAudioUrl, onSeparate,
  hasStems, onConfigureStems,
  sourceLatentUrl, onLatentLoaded, onLatentClear,
  timbreOverridePath, onTimbreOverridePathChange, token,
  coverCaptionProvider, coverCaptionModel,
  onCoverCaptionProviderChange, onCoverCaptionModelChange,
}) => {
  const { t } = useTranslation();
  const [isDragging, setIsDragging] = useState(false);
  const [isTimbreDragging, setIsTimbreDragging] = useState(false);
  const [isTimbreUploading, setIsTimbreUploading] = useState(false);
  const [showTimbreBrowser, setShowTimbreBrowser] = useState(false);
  interface ReferenceTrack { name: string; size: number; url: string; }
  const [timbreRefs, setTimbreRefs] = useState<ReferenceTrack[]>([]);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const timbreFileRef = React.useRef<HTMLInputElement>(null);

  // Load references when browser opens
  useEffect(() => {
    if (showTimbreBrowser) {
      masteringApi.listReferences()
        .then(data => setTimbreRefs(data.references))
        .catch(() => {});
    }
  }, [showTimbreBrowser]);

  const handleTimbreUpload = useCallback(async (file: File) => {
    if (!token) return;
    try {
      setIsTimbreUploading(true);
      const result = await masteringApi.uploadReference(file, token);
      onTimbreOverridePathChange(result.name);
      // Refresh list if browser is open
      if (showTimbreBrowser) {
        const data = await masteringApi.listReferences();
        setTimbreRefs(data.references);
      }
    } catch (err) {
      console.error('[Timbre] Upload failed:', err);
    } finally {
      setIsTimbreUploading(false);
    }
  }, [token, onTimbreOverridePathChange, showTimbreBrowser]);

  const handleTimbreDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsTimbreDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && /\.(mp3|wav|flac|ogg|m4a|opus|aac)$/i.test(file.name)) {
      handleTimbreUpload(file);
    }
  }, [handleTimbreUpload]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && /\.(mp3|wav|flac|ogg|m4a|opus|aac)$/i.test(file.name)) {
      onFileSelected(file);
    }
  }, [onFileSelected]);

  const correctedBpm = bpmOverride != null ? bpmOverride : (analysis?.bpm ? Math.round(analysis.bpm * bpmCorrection) : null);
  const effectiveKey = keyOverride || analysis?.key || null;
  const bpmIsOverridden = bpmOverride != null;

  return (
    <div className="w-[320px] flex-shrink-0 overflow-y-auto scrollbar-hide border-r border-zinc-200 dark:border-white/5 p-4 space-y-4">
      {/* Upload zone */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Upload className="w-4 h-4 text-cyan-400" />
          {t('cover.sourceAudio')}
        </div>
        <div
          onDrop={handleDrop}
          onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-300
            ${isDragging
              ? 'border-cyan-400 bg-cyan-500/10 scale-[1.02]'
              : sourceFileName
                ? 'border-cyan-500/30 bg-cyan-500/5'
                : 'border-zinc-300 dark:border-zinc-700 hover:border-cyan-400/50 hover:bg-cyan-500/5'}
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp3,.wav,.flac,.ogg,.m4a,.opus,.aac"
            className="hidden"
            onChange={e => { if (e.target.files?.[0]) onFileSelected(e.target.files[0]); }}
          />
          {isUploading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              <span className="text-xs text-cyan-400">{t('cover.uploading')}</span>
            </div>
          ) : isAnalyzing ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-8 h-8 text-teal-400 animate-spin" />
              <span className="text-xs text-teal-400">{t('cover.analyzingBpmKey')}</span>
            </div>
          ) : sourceFileName ? (
            <div className="flex flex-col items-center gap-2">
              <Music className="w-8 h-8 text-cyan-400" />
              <span className="text-xs text-zinc-600 dark:text-zinc-400 truncate max-w-full">{sourceFileName}</span>
              <span className="text-[10px] text-zinc-500">{t('cover.clickOrDropReplace')}</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-8 h-8 text-zinc-600 dark:text-zinc-400" />
              <span className="text-xs text-zinc-500">{t('cover.dropAudioOrBrowse')}</span>
              <span className="text-[10px] text-zinc-600">MP3, WAV, FLAC, OGG, M4A</span>
            </div>
          )}
        </div>

        {/* Latent import (alternative to audio) */}
        <LatentImport
          latentUrl={sourceLatentUrl}
          onLatentLoaded={onLatentLoaded}
          onClear={onLatentClear}
        />
      </div>

      {/* Timbre Reference Override */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Volume2 className="w-4 h-4 text-teal-400" />
          {t('cover.timbreRef')}
          <span className="text-[10px] font-normal text-zinc-500">(optional)</span>
        </div>

        {/* Drag-and-drop / click zone */}
        <div
          onDrop={handleTimbreDrop}
          onDragOver={e => { e.preventDefault(); setIsTimbreDragging(true); }}
          onDragLeave={() => setIsTimbreDragging(false)}
          onClick={() => timbreFileRef.current?.click()}
          className={`
            relative border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all duration-300
            ${isTimbreDragging
              ? 'border-teal-400 bg-teal-500/10 scale-[1.02]'
              : timbreOverridePath
                ? 'border-teal-500/30 bg-teal-500/5'
                : 'border-zinc-300 dark:border-zinc-700 hover:border-teal-400/50 hover:bg-teal-500/5'}
          `}
        >
          <input
            ref={timbreFileRef}
            type="file"
            accept=".mp3,.wav,.flac,.ogg,.m4a,.opus,.aac"
            className="hidden"
            onChange={e => {
              if (e.target.files?.[0]) handleTimbreUpload(e.target.files[0]);
              e.target.value = '';
            }}
          />
          {isTimbreUploading ? (
            <div className="flex flex-col items-center gap-1">
              <Loader2 className="w-6 h-6 text-teal-400 animate-spin" />
              <span className="text-[10px] text-teal-400">Uploading...</span>
            </div>
          ) : timbreOverridePath ? (
            <div className="flex flex-col items-center gap-1">
              <Volume2 className="w-6 h-6 text-teal-400" />
              <span className="text-[10px] text-zinc-600 dark:text-zinc-400 truncate max-w-full">{timbreOverridePath}</span>
              <span className="text-[9px] text-zinc-500">{t('cover.clickOrDropReplace')}</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-1">
              <Upload className="w-6 h-6 text-zinc-600 dark:text-zinc-400" />
              <span className="text-[10px] text-zinc-500">Drop timbre reference or click to upload</span>
              <span className="text-[9px] text-zinc-600">MP3, WAV, FLAC, OGG</span>
            </div>
          )}
        </div>

        {/* Action row: Browse + Clear */}
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => { e.stopPropagation(); setShowTimbreBrowser(!showTimbreBrowser); }}
            className="flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg bg-teal-500/10 hover:bg-teal-500/20 text-teal-400 text-[10px] font-medium transition-colors border border-teal-500/20"
          >
            <FolderOpen className="w-3 h-3" />
            Browse References
          </button>
          {timbreOverridePath && (
            <button
              onClick={() => onTimbreOverridePathChange('')}
              className="flex items-center gap-1 px-2 py-1.5 rounded-lg text-[10px] font-medium text-zinc-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
            >
              <X className="w-3 h-3" />
              {t('cover.clearTimbreRef')}
            </button>
          )}
        </div>

        {/* Reference browser modal */}
        {showTimbreBrowser && (
          <div className="rounded-xl bg-black/5 dark:bg-white/5 border border-zinc-200 dark:border-white/10 overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-200 dark:border-white/5">
              <span className="text-[10px] font-semibold text-zinc-500 uppercase">Reference Tracks</span>
              <button onClick={() => setShowTimbreBrowser(false)} className="text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300">
                <X className="w-3 h-3" />
              </button>
            </div>
            <div className="max-h-[160px] overflow-y-auto scrollbar-hide">
              {timbreRefs.length === 0 ? (
                <div className="px-3 py-4 text-center text-[10px] text-zinc-500">
                  No references uploaded yet. Drop an audio file above to add one.
                </div>
              ) : (
                timbreRefs.map(ref => (
                  <button
                    key={ref.name}
                    onClick={() => { onTimbreOverridePathChange(ref.name); setShowTimbreBrowser(false); }}
                    className={`w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-teal-500/10 transition-colors ${
                      timbreOverridePath === ref.name ? 'bg-teal-500/15 border-l-2 border-teal-400' : ''
                    }`}
                  >
                    <Volume2 className="w-3 h-3 text-teal-400 flex-shrink-0" />
                    <span className="text-xs text-zinc-700 dark:text-zinc-300 truncate flex-1">{ref.name}</span>
                    <span className="text-[9px] text-zinc-500 flex-shrink-0">
                      {ref.size < 1024 * 1024 ? `${(ref.size / 1024).toFixed(0)} KB` : `${(ref.size / (1024 * 1024)).toFixed(1)} MB`}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        )}

        {/* Selected timbre info badge */}
        {timbreOverridePath && (
          <div className="rounded-lg bg-teal-500/5 border border-teal-500/20 px-3 py-1.5 flex items-center gap-2">
            <Volume2 className="w-3 h-3 text-teal-400 flex-shrink-0" />
            <span className="text-[10px] text-zinc-500 truncate flex-1">{timbreOverridePath}</span>
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-teal-900/30 text-teal-400">TIMBRE</span>
          </div>
        )}

        {!timbreOverridePath && (
          <p className="text-[10px] text-zinc-500 leading-tight">
            {t('cover.timbreRefHelp')}
          </p>
        )}
      </div>

      {/* Metadata display */}
      {metadata && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-medium text-zinc-500 uppercase">{t('cover.metadata')}</span>
            <button onClick={onClear} className="text-zinc-500 hover:text-red-400 transition-colors" title="Clear">
              <X className="w-3 h-3" />
            </button>
          </div>
          <div className="rounded-lg bg-black/5 dark:bg-white/5 p-3 space-y-1">
            {metadata.artist && <MetaRow label="Artist" value={metadata.artist} />}
            {metadata.title && <MetaRow label="Title" value={metadata.title} />}
            {metadata.album && <MetaRow label="Album" value={metadata.album} />}
            {metadata.duration != null && (
              <MetaRow label="Duration" value={`${Math.floor(metadata.duration / 60)}:${String(Math.floor(metadata.duration % 60)).padStart(2, '0')}`} />
            )}
          </div>
        </div>
      )}

      {/* Analysis display */}
      {analysis && (
        <div className="space-y-2">
          <span className="text-[10px] font-medium text-zinc-500 uppercase">{t('cover.analysis')}</span>
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-lg bg-gradient-to-br from-cyan-500/10 to-teal-500/10 border border-cyan-500/20 p-3 text-center">
              <span className="text-[10px] text-zinc-500 block">BPM</span>
              <span className={`text-lg font-bold ${bpmIsOverridden ? 'text-amber-400' : 'text-cyan-400'}`}>{correctedBpm ?? analysis.bpm}</span>
              {(bpmIsOverridden || bpmCorrection !== 1) && (
                <span className="text-[9px] text-zinc-500 block">
                  (detected: {analysis.bpm})
                </span>
              )}
            </div>
            <div className="rounded-lg bg-gradient-to-br from-teal-500/10 to-emerald-500/10 border border-teal-500/20 p-3 text-center">
              <span className="text-[10px] text-zinc-500 block">Key</span>
              <span className="text-lg font-bold text-teal-400">{effectiveKey || analysis.key}</span>
              {keyOverride && (
                <span className="text-[9px] text-zinc-500 block">
                  (detected: {analysis.key})
                </span>
              )}
            </div>
          </div>
          {/* BPM correction — Essentia sometimes halves or doubles the tempo */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 whitespace-nowrap">{t('cover.tempoFix')}</span>
            <div className="flex gap-1 flex-1">
              {([
                { label: '÷2', value: 0.5 },
                { label: 'Detected', value: 1 },
                { label: '×2', value: 2 },
              ] as const).map(opt => (
                <button
                  key={opt.value}
                  onClick={() => { onBpmCorrectionChange(opt.value); onBpmOverrideChange(null); }}
                  className={`flex-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all ${
                    bpmCorrection === opt.value && !bpmIsOverridden
                      ? 'bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-500/40'
                      : 'bg-white/5 text-zinc-500 hover:bg-white/10 hover:text-zinc-700 dark:text-zinc-300'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
          {/* Free-text BPM override */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 whitespace-nowrap">{t('cover.custom')}</span>
            <input
              type="number"
              min={20}
              max={300}
              placeholder={String(analysis?.bpm ? Math.round(analysis.bpm * bpmCorrection) : 120)}
              value={bpmOverride ?? ''}
              onChange={e => {
                const v = e.target.value.trim();
                onBpmOverrideChange(v ? parseInt(v, 10) || null : null);
              }}
              className={`flex-1 px-2 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 border text-xs outline-none transition-colors tabular-nums ${
                bpmIsOverridden
                  ? 'border-amber-500/40 text-amber-300 focus:border-amber-500/60 focus:ring-1 focus:ring-amber-500/20'
                  : 'border-zinc-300 dark:border-white/10 text-zinc-700 dark:text-zinc-300 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/20'
              }`}
            />
            {bpmIsOverridden && (
              <button onClick={() => onBpmOverrideChange(null)} className="text-zinc-500 hover:text-red-400 transition-colors" title="Clear override">
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
          {/* Key override — Essentia sometimes gets the wrong key */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 whitespace-nowrap">{t('cover.keyFix')}</span>
            <select
              value={keyOverride || ''}
              onChange={e => onKeyOverrideChange(e.target.value || null)}
              className={`flex-1 px-2 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 border text-xs outline-none transition-colors cursor-pointer ${
                keyOverride
                  ? 'border-teal-500/40 text-teal-300 focus:border-teal-500/60 focus:ring-1 focus:ring-teal-500/20'
                  : 'border-zinc-300 dark:border-white/10 text-zinc-700 dark:text-zinc-300 focus:border-teal-500/50 focus:ring-1 focus:ring-teal-500/20'
              }`}
            >
              <option value="">Detected{analysis?.key ? ` (${analysis.key})` : ''}</option>
              {ALL_KEYS.map(k => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </div>
          {/* Vocal language */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 whitespace-nowrap">{t('cover.language')}</span>
            <select
              value={vocalLanguage}
              onChange={e => onVocalLanguageChange(e.target.value)}
              className="flex-1 px-2 py-1 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-xs text-zinc-700 dark:text-zinc-300 outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/20 transition-colors cursor-pointer"
            >
              {VOCAL_LANGUAGES.map(l => (
                <option key={l.value} value={l.value}>{l.label}</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Advanced Mode */}
      <div className="border-t border-zinc-200 dark:border-white/5 pt-4 space-y-3">
        <button
          onClick={() => onAdvancedModeChange(!advancedMode)}
          className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
            advancedMode
              ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
              : 'bg-white/5 text-zinc-600 dark:text-zinc-400 border border-zinc-300 dark:border-white/10 hover:border-zinc-400 dark:hover:border-white/20'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          {t('cover.advancedMode')}{advancedMode ? ' (On)' : ''}
        </button>

        {advancedMode && (
          <div className="space-y-2">
            <select
              value={sepLevel}
              onChange={(e) => onSepLevelChange(parseInt(e.target.value))}
              className="w-full px-2 py-1.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-xs text-zinc-700 dark:text-zinc-300 focus:outline-none focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/20 transition-colors cursor-pointer"
            >
              {SEPARATION_LEVELS.map(l => (
                <option key={l.value} value={l.value}>{l.label} — {l.description}</option>
              ))}
            </select>

            {hasStems ? (
              <>
                <button
                  onClick={onConfigureStems}
                  className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-gradient-to-r from-purple-500 to-cyan-500 text-white text-xs font-semibold shadow-lg hover:shadow-cyan-500/25 transition-all"
                >
                  🎛️ Configure Stems
                </button>
                {/* Splitting on its own changes nothing: the stems are
                    recombined at full volume at generation time, which is the
                    source audio again. The mixer is where the feature lives,
                    and the button silently becoming a different button did not
                    say so (issue #119). */}
                <p className="text-[10px] text-zinc-500 leading-tight">
                  Stems are ready. Mute or lower one in the mixer to keep it out of
                  the cover. Left untouched they recombine at full volume, which is
                  the original audio.
                </p>
              </>
            ) : (
              <button
                onClick={onSeparate}
                disabled={isSeparating || !sourceAudioUrl}
                className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 text-xs font-medium transition-colors disabled:opacity-50"
              >
                {isSeparating ? (
                  <>
                    <Loader2 className="w-3 h-3 animate-spin" />
                    {Math.round(sepProgress * 100)}% — {sepMessage}
                  </>
                ) : (
                  '✂ Split Stems'
                )}
              </button>
            )}
            {!hasStems && (
              <p className="text-[10px] text-zinc-500 leading-tight">
                Step 1 of 2. Separates the source, then a mixer opens here for
                muting or attenuating individual stems.
              </p>
            )}
          </div>
        )}
      </div>

      {/* Caption LLM */}
      <div className="border-t border-zinc-200 dark:border-white/5 pt-4 space-y-2">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <span className="text-base">✨</span>
          Caption LLM
        </div>
        <ProviderSelector
          selectedProvider={coverCaptionProvider}
          selectedModel={coverCaptionModel}
          onProviderChange={onCoverCaptionProviderChange}
          onModelChange={onCoverCaptionModelChange}
          label="Caption LLM"
          compact
        />
        <p className="text-[10px] text-zinc-500 leading-tight">
          Used to auto-generate a style description when no caption is found for the selected artist.
        </p>
      </div>
    </div>
  );
};

const MetaRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex items-center gap-2">
    <span className="text-[10px] text-zinc-500 w-14 flex-shrink-0">{label}</span>
    <span className="text-xs text-zinc-700 dark:text-zinc-300 truncate">{value}</span>
  </div>
);
