// StemStudio.tsx — Main Stem Studio orchestrator
//
// Composes: SourceSelector, TrackSelector, StemMixer, RecentExtractions
// Manages extraction state, polls progress, loads results into mixer.
//
// Extract mode forces a base/SFT DiT model and disables adapters,
// LM thinking, and post-processing regardless of global bar settings.
//
// Layout follows the same pattern as CoverStudio / LyricStudio:
// flex columns with border dividers, resizable right sidebar.

import React, { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, Info, Clock, ListOrdered } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { modelApi } from '../../services/api';
import {
  submitExtraction, submitSupersep, waitForExtraction, getExtractResult,
  getStemUrl, getDownloadAllUrl, TRACK_CATEGORIES,
  type ExtractProgress, type ExtractJobResult,
} from '../../services/stemStudioApi';
import { type SeparationLevel } from '../../services/supersepApi';
import { StemMixer, type StemControl, type MixerStemInfo } from '../shared/StemMixer';
import { SourceSelector } from './SourceSelector';
import { TrackSelector } from './TrackSelector';
import { RecentExtractions } from './RecentExtractions';
import { Section } from '../shared/ActivitySidebar';
import { InlineAudioQueue } from '../lyric-studio/InlineAudioQueue';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useAudioGenQueueSelector } from '../../stores/audioGenQueueStore';

/** Filter DiT model list to only pure base models (no merge/sft/turbo) */
function getBaseModels(ditModels: string[]): string[] {
  return ditModels.filter(m => {
    const lower = m.toLowerCase();
    // Only allow acestep-v15-base-* and acestep-v15-xl-base-*
    return lower.startsWith('acestep-v15-base-') || lower.startsWith('acestep-v15-xl-base-');
  });
}

export const StemStudio: React.FC = () => {
  const { t } = useTranslation();
  // Source audio (persisted across sessions for testing convenience)
  const [sourceAudioUrl, _setSourceAudioUrl] = useState(() => localStorage.getItem('hs-stem-sourceUrl') || '');
  const [sourceFileName, _setSourceFileName] = useState(() => localStorage.getItem('hs-stem-sourceFile') || '');
  const setSourceAudioUrl = (url: string) => { localStorage.setItem('hs-stem-sourceUrl', url); _setSourceAudioUrl(url); };
  const setSourceFileName = (name: string) => { localStorage.setItem('hs-stem-sourceFile', name); _setSourceFileName(name); };

  // Track selection
  const [selectedTracks, setSelectedTracks] = useState<string[]>(['vocals', 'drums', 'bass', 'guitar']);
  const [mode, setMode] = useState<'extract' | 'supersep'>('extract');
  const [sepLevel, setSepLevel] = useState<SeparationLevel>(() => {
    const stored = localStorage.getItem('hs-stem-sepLevel');
    return (stored != null ? parseInt(stored) : 1) as SeparationLevel;
  });

  // Optional enhancement
  const [style, setStyle] = useState('');
  const [lyrics, setLyrics] = useState('');

  // Model selection — extract requires base/SFT model
  const [baseModels, setBaseModels] = useState<string[]>([]);
  const [extractModel, setExtractModel] = useState<string>('');
  const [modelsLoading, setModelsLoading] = useState(true);

  // Extraction state
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractProgress, setExtractProgress] = useState<ExtractProgress | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  // Results
  const [mixerStems, setMixerStems] = useState<MixerStemInfo[] | null>(null);
  const [stemControls, setStemControls] = useState<StemControl[]>([]);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Sidebar — shared width with other studio pages
  const [sidebarWidth, setSidebarWidth] = usePersistedState('hs-activitySidebarWidth', 320);
  const queueCount = useAudioGenQueueSelector(s =>
    s.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length
  );

  // Fetch available base models on mount — restore persisted selection
  useEffect(() => {
    modelApi.list()
      .then(data => {
        const base = getBaseModels(data.models.dit || []);
        setBaseModels(base);
        if (base.length > 0) {
          const stored = localStorage.getItem('hs-stem-extractModel');
          setExtractModel(stored && base.includes(stored) ? stored : base[0]);
        }
      })
      .catch(err => console.error('[StemStudio] Failed to load models:', err))
      .finally(() => setModelsLoading(false));
  }, []);

  // Persist extract model selection
  useEffect(() => {
    if (extractModel) localStorage.setItem('hs-stem-extractModel', extractModel);
  }, [extractModel]);

  // Persist sep level selection
  useEffect(() => {
    localStorage.setItem('hs-stem-sepLevel', String(sepLevel));
  }, [sepLevel]);

  const handleSourceChange = useCallback((url: string, fileName: string, meta?: { style?: string; lyrics?: string }) => {
    setSourceAudioUrl(url);
    setSourceFileName(fileName);
    if (meta?.style) setStyle(meta.style);
    if (meta?.lyrics) setLyrics(meta.lyrics);
  }, []);

  const handleSupersep = useCallback(async () => {
    if (!sourceAudioUrl) return;

    setIsExtracting(true);
    setExtractProgress(null);

    try {
      const jobId = await submitSupersep({
        sourceAudioUrl,
        sourceFileName,
        level: sepLevel,
      });

      setActiveJobId(jobId);

      const result = await waitForExtraction(jobId, (progress) => {
        setExtractProgress(progress);
      });

      loadResultIntoMixer(jobId, result);
      setRefreshTrigger(prev => prev + 1);

    } catch (err: any) {
      console.error('SuperSep failed:', err);
      setExtractProgress({
        status: 'failed',
        progress: 0,
        currentTrack: '',
        completedStems: [],
        totalTracks: 0,
        error: err.message,
      });
    } finally {
      setIsExtracting(false);
    }
  }, [sourceAudioUrl, sourceFileName, sepLevel]);

  const handleExtract = useCallback(async () => {
    if (!sourceAudioUrl) return;

    // Route based on mode
    if (mode === 'supersep') {
      return handleSupersep();
    }

    // Extract mode — needs tracks + model
    if (selectedTracks.length === 0 || !extractModel) return;

    setIsExtracting(true);
    setExtractProgress(null);

    try {
      const ditSettings = {
        ditModel: extractModel,
        loraPath: '',
        loraScale: 0,
        seed: -1,
      };

      const jobId = await submitExtraction({
        sourceAudioUrl,
        sourceFileName,
        tracks: selectedTracks,
        style: style || undefined,
        lyrics: lyrics || undefined,
        ditSettings,
      });

      setActiveJobId(jobId);

      const result = await waitForExtraction(jobId, (progress) => {
        setExtractProgress(progress);
      });

      loadResultIntoMixer(jobId, result);
      setRefreshTrigger(prev => prev + 1);

    } catch (err: any) {
      console.error('Extraction failed:', err);
      setExtractProgress({
        status: 'failed',
        progress: 0,
        currentTrack: '',
        completedStems: [],
        totalTracks: selectedTracks.length,
        error: err.message,
      });
    } finally {
      setIsExtracting(false);
    }
  }, [sourceAudioUrl, sourceFileName, selectedTracks, style, lyrics, extractModel, mode, handleSupersep]);

  const loadResultIntoMixer = useCallback((jobId: string, result: ExtractJobResult) => {
    const stems: MixerStemInfo[] = result.stems.map((s, idx) => ({
      name: s.trackName,
      category: (s as any).category || TRACK_CATEGORIES[s.trackName] || 'other',
      audioUrl: s.audioUrl || getStemUrl(jobId, s.trackName),
      index: idx,
      stage: (s as any).stage,
    }));
    setMixerStems(stems);
    setStemControls(stems.map(s => ({ index: s.index, volume: 1.0, muted: false })));
  }, []);

  const handleSelectPastJob = useCallback(async (jobId: string) => {
    try {
      setActiveJobId(jobId);
      const result = await getExtractResult(jobId);
      loadResultIntoMixer(jobId, result);
    } catch (err) {
      console.error('Failed to load past job:', err);
    }
  }, [loadResultIntoMixer]);

  const handleDownloadStem = useCallback((stem: MixerStemInfo) => {
    if (!activeJobId) return;
    const a = document.createElement('a');
    a.href = getStemUrl(activeJobId, stem.name);
    a.download = `${stem.name}.wav`;
    a.click();
  }, [activeJobId]);

  const handleDownloadAll = useCallback(() => {
    if (!activeJobId) return;
    const a = document.createElement('a');
    a.href = getDownloadAllUrl(activeJobId);
    a.download = 'stems.zip';
    a.click();
  }, [activeJobId]);

  // Sidebar resize handler — identical to CoverStudio
  const handleSidebarResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = sidebarWidth;
    const onMove = (ev: MouseEvent) => {
      const newW = Math.min(700, Math.max(240, startW + startX - ev.clientX));
      setSidebarWidth(newW);
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [sidebarWidth, setSidebarWidth]);

  return (
    <div className="flex flex-col w-full h-full overflow-hidden">

      {/* Main workspace */}
      <div className="flex-1 flex overflow-hidden">

        {/* Left — Source Audio + Optional Fields */}
        <div className="flex flex-col gap-4 p-4 overflow-y-auto border-r border-zinc-200 dark:border-white/5 flex-shrink-0" style={{ width: 300 }}>
          <SourceSelector
            sourceAudioUrl={sourceAudioUrl}
            sourceFileName={sourceFileName}
            onSourceChange={handleSourceChange}
          />

          {/* Optional: style hint + lyrics */}
          <div className="border-t border-zinc-200 dark:border-white/5 pt-3">
            <details>
              <summary className="text-xs text-zinc-500 cursor-pointer font-medium">{t('stem.optionalStyleLyrics')}</summary>
              <div className="flex flex-col gap-2 mt-2.5">
                <label className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">{t('stem.styleHint')}</label>
                <input
                  type="text"
                  value={style}
                  onChange={e => setStyle(e.target.value)}
                  placeholder="e.g. indie rock, distorted guitar, raw vocals"
                  className="px-2.5 py-2 rounded-md border border-white/[0.08] bg-white/[0.04] text-zinc-700 dark:text-zinc-300 text-xs outline-none focus:border-purple-500/40 transition-colors"
                />
                <label className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">{t('stem.lyrics')}</label>
                <textarea
                  value={lyrics}
                  onChange={e => setLyrics(e.target.value)}
                  placeholder="Paste lyrics here to improve vocal extraction..."
                  className="px-2.5 py-2 rounded-md border border-white/[0.08] bg-white/[0.04] text-zinc-700 dark:text-zinc-300 text-xs outline-none resize-y font-[inherit] focus:border-purple-500/40 transition-colors"
                  rows={4}
                />
              </div>
            </details>
          </div>
        </div>

        {/* Center — Track Selection + Mixer */}
        <div className="flex-1 flex flex-col gap-4 p-4 overflow-y-auto border-r border-zinc-200 dark:border-white/5">
          {/* Model selector for Extract */}
          {mode === 'extract' && !modelsLoading && baseModels.length === 0 && (
            <div style={styles.noModelsWarning}>
              <AlertTriangle size={14} />
              <span>No base/SFT models found. Extract requires a non-turbo DiT model. Use <strong>Get More Models</strong> in the sidebar to download one.</span>
            </div>
          )}
          {mode === 'extract' && baseModels.length > 0 && (
            <div style={styles.modelSelector}>
              <Info size={13} style={{ color: '#a78bfa', flexShrink: 0 }} />
              <span style={styles.modelLabel}>{t('stem.extractModel')}</span>
              <select
                value={extractModel}
                onChange={e => setExtractModel(e.target.value)}
                style={styles.modelSelect}
                disabled={isExtracting}
              >
                {baseModels.map(m => (
                  <option key={m} value={m}>{m.replace(/\.gguf$/i, '')}</option>
                ))}
              </select>
            </div>
          )}

          <TrackSelector
            selectedTracks={selectedTracks}
            onTracksChange={setSelectedTracks}
            mode={mode}
            onModeChange={setMode}
            onExtract={handleExtract}
            isExtracting={isExtracting}
            canExtract={mode === 'supersep' ? !!sourceAudioUrl : (!!sourceAudioUrl && !!extractModel)}
            sepLevel={sepLevel}
            onSepLevelChange={setSepLevel}
          />

          {/* Progress */}
          {extractProgress && (extractProgress.status === 'extracting' || extractProgress.status === 'separating' || extractProgress.status === 'saving') && (
            <div style={styles.progressSection}>
              <div style={styles.progressLabel}>
                {extractProgress.status === 'separating'
                  ? `Separating... ${extractProgress.sepMessage || ''}`
                  : extractProgress.status === 'saving'
                    ? `Saving stems (${extractProgress.completedStems.length}/${extractProgress.totalTracks})`
                    : `Extracting ${extractProgress.currentTrack} (${extractProgress.completedStems.length + 1}/${extractProgress.totalTracks})`
                }
              </div>
              <div style={styles.progressBar}>
                <div style={{ ...styles.progressFill, width: `${extractProgress.progress}%` }} />
              </div>
              <span style={styles.progressPercent}>{extractProgress.progress}%</span>
            </div>
          )}

          {/* Error */}
          {extractProgress?.status === 'failed' && (
            <div style={styles.errorMsg}>
              ❌ Extraction failed: {extractProgress.error}
            </div>
          )}

          {/* Mixer */}
          {mixerStems && activeJobId && (
            <div className="mt-2">
              <StemMixer
                jobId={activeJobId}
                stems={mixerStems}
                controls={stemControls}
                onControlsChange={setStemControls}
                onDownloadStem={handleDownloadStem}
                onDownloadAll={handleDownloadAll}
              />
            </div>
          )}
        </div>

        {/* Resize handle */}
        <div
          className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-purple-500/20 active:bg-purple-500/30 transition-colors"
          onMouseDown={handleSidebarResize}
        >
          <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-purple-400 transition-colors" />
        </div>

        {/* Right — Recent Extractions + Queue */}
        <div className="h-full flex-shrink-0 border-l border-zinc-200 dark:border-white/5 overflow-hidden flex flex-col" style={{ width: sidebarWidth }}>
          <Section
            title={t('stem.recentExtractions')}
            icon={<Clock className="w-3 h-3" />}
            defaultOpen={true}
          >
            <RecentExtractions
              onSelectJob={handleSelectPastJob}
              activeJobId={activeJobId || undefined}
              refreshTrigger={refreshTrigger}
            />
          </Section>

          <Section
            title={t('stem.queue')}
            icon={<ListOrdered className="w-3 h-3" />}
            count={queueCount}
            countColor="bg-purple-500/20 text-purple-300"
            defaultOpen={true}
          >
            <InlineAudioQueue />
          </Section>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  noModelsWarning: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 12px',
    borderRadius: 8,
    background: 'rgba(239,68,68,0.1)',
    border: '1px solid rgba(239,68,68,0.25)',
    color: '#ef4444',
    fontSize: 12,
    fontWeight: 500,
  },
  modelSelector: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '6px 10px',
    borderRadius: 8,
    background: 'rgba(167,139,250,0.06)',
    border: '1px solid rgba(167,139,250,0.15)',
  },
  modelLabel: {
    fontSize: 11,
    fontWeight: 600,
    color: '#a78bfa',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
    whiteSpace: 'nowrap' as const,
  },
  modelSelect: {
    flex: 1,
    padding: '5px 10px',
    borderRadius: 12,
    border: '1px solid rgba(255,255,255,0.1)',
    background: '#27272a',
    color: '#d4d4d8',
    fontSize: 12,
    outline: 'none',
    cursor: 'pointer',
    transition: 'border-color 0.15s ease',
  },
  progressSection: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '10px 0',
  },
  progressLabel: {
    fontSize: 12,
    color: '#a78bfa',
    fontWeight: 600,
    whiteSpace: 'nowrap',
  },
  progressBar: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    background: 'rgba(255,255,255,0.06)',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
    background: 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
    transition: 'width 0.3s ease',
  },
  progressPercent: {
    fontSize: 11,
    color: '#888',
    fontFamily: 'monospace',
    width: 32,
    textAlign: 'right',
  },
  errorMsg: {
    padding: '10px 14px',
    borderRadius: 8,
    background: 'rgba(239,68,68,0.1)',
    border: '1px solid rgba(239,68,68,0.2)',
    color: '#ef4444',
    fontSize: 12,
  },
};

export default StemStudio;
