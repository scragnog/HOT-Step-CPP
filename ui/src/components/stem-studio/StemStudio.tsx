// StemStudio.tsx — Main Stem Studio orchestrator
//
// Composes: SourceSelector, TrackSelector, StemMixer, RecentExtractions
// Manages extraction state, polls progress, loads results into mixer.
//
// Extract mode forces a base/SFT DiT model and disables adapters,
// LM thinking, and post-processing regardless of global bar settings.

import React, { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, Info } from 'lucide-react';
import { modelApi } from '../../services/api';
import {
  submitExtraction, waitForExtraction, getExtractResult,
  getStemUrl, getDownloadAllUrl, TRACK_CATEGORIES,
  type ExtractProgress, type ExtractJobResult,
} from '../../services/stemStudioApi';
import { StemMixer, type StemControl, type MixerStemInfo } from '../shared/StemMixer';
import { SourceSelector } from './SourceSelector';
import { TrackSelector } from './TrackSelector';
import { RecentExtractions } from './RecentExtractions';

/** Filter DiT model list to only pure base models (no merge/sft/turbo) */
function getBaseModels(ditModels: string[]): string[] {
  return ditModels.filter(m => {
    const lower = m.toLowerCase();
    // Only allow acestep-v15-base-* and acestep-v15-xl-base-*
    return lower.startsWith('acestep-v15-base-') || lower.startsWith('acestep-v15-xl-base-');
  });
}

export const StemStudio: React.FC = () => {
  // Source audio
  const [sourceAudioUrl, setSourceAudioUrl] = useState('');
  const [sourceFileName, setSourceFileName] = useState('');

  // Track selection
  const [selectedTracks, setSelectedTracks] = useState<string[]>(['vocals', 'drums', 'bass', 'guitar']);
  const [mode, setMode] = useState<'extract' | 'supersep'>('extract');

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

  // Fetch available base models on mount
  useEffect(() => {
    modelApi.list()
      .then(data => {
        const base = getBaseModels(data.models.dit || []);
        setBaseModels(base);
        if (base.length > 0) setExtractModel(base[0]);
      })
      .catch(err => console.error('[StemStudio] Failed to load models:', err))
      .finally(() => setModelsLoading(false));
  }, []);

  const handleSourceChange = useCallback((url: string, fileName: string) => {
    setSourceAudioUrl(url);
    setSourceFileName(fileName);
  }, []);

  const handleExtract = useCallback(async () => {
    if (!sourceAudioUrl || selectedTracks.length === 0 || !extractModel) return;

    setIsExtracting(true);
    setExtractProgress(null);
    setMixerStems(null);

    try {
      // Force extract-specific settings — ignores global bar
      const ditSettings = {
        ditModel: extractModel,       // forced base/SFT model
        loraPath: '',                  // no adapter
        loraScale: 0,                 // no adapter
        seed: -1,                     // always random for extract
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

      // Poll for progress
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
  }, [sourceAudioUrl, sourceFileName, selectedTracks, style, lyrics, extractModel]);

  const loadResultIntoMixer = useCallback((jobId: string, result: ExtractJobResult) => {
    const stems: MixerStemInfo[] = result.stems.map((s, idx) => ({
      name: s.trackName,
      category: TRACK_CATEGORIES[s.trackName] || 'other',
      audioUrl: getStemUrl(jobId, s.trackName),
      index: idx,
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

  return (
    <div style={styles.outerContainer}>
      <div style={styles.layout}>
        {/* Left column — Source Audio */}
        <div style={styles.leftCol}>
          <SourceSelector
            sourceAudioUrl={sourceAudioUrl}
            sourceFileName={sourceFileName}
            onSourceChange={handleSourceChange}
          />

          {/* Optional: style hint + lyrics */}
          <div style={styles.optionalSection}>
            <details>
              <summary style={styles.optionalSummary}>Optional: Style & Lyrics (improves extraction)</summary>
              <div style={styles.optionalFields}>
                <label style={styles.fieldLabel}>Style Hint</label>
                <input
                  type="text"
                  value={style}
                  onChange={e => setStyle(e.target.value)}
                  placeholder="e.g. indie rock, distorted guitar, raw vocals"
                  style={styles.textInput}
                />
                <label style={styles.fieldLabel}>Lyrics</label>
                <textarea
                  value={lyrics}
                  onChange={e => setLyrics(e.target.value)}
                  placeholder="Paste lyrics here to improve vocal extraction..."
                  style={styles.textarea}
                  rows={4}
                />
              </div>
            </details>
          </div>
        </div>

        {/* Center column — Track Selection + Mixer */}
        <div style={styles.centerCol}>
          {/* Model selector for Extract */}
          {!modelsLoading && baseModels.length === 0 && (
            <div style={styles.noModelsWarning}>
              <AlertTriangle size={14} />
              <span>No base/SFT models found. Extract requires a non-turbo DiT model. Use <strong>Get More Models</strong> in the sidebar to download one.</span>
            </div>
          )}
          {baseModels.length > 0 && (
            <div style={styles.modelSelector}>
              <Info size={13} style={{ color: '#a78bfa', flexShrink: 0 }} />
              <span style={styles.modelLabel}>Extract Model</span>
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
            canExtract={!!sourceAudioUrl && !!extractModel}
          />

          {/* Progress */}
          {extractProgress && extractProgress.status === 'extracting' && (
            <div style={styles.progressSection}>
              <div style={styles.progressLabel}>
                Extracting {extractProgress.currentTrack} ({extractProgress.completedStems.length + 1}/{extractProgress.totalTracks})
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
            <div style={styles.mixerSection}>
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

        {/* Right column — Recent Extractions */}
        <div style={styles.rightCol}>
          <RecentExtractions
            onSelectJob={handleSelectPastJob}
            activeJobId={activeJobId || undefined}
            refreshTrigger={refreshTrigger}
          />
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  outerContainer: {
    height: '100%',
    padding: 16,
    overflowY: 'auto',
  },
  layout: {
    display: 'grid',
    gridTemplateColumns: '280px 1fr 240px',
    gap: 16,
    maxWidth: 1400,
    margin: '0 auto',
    height: '100%',
  },
  leftCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    padding: 16,
    borderRadius: 12,
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.06)',
    overflowY: 'auto',
  },
  centerCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    padding: 16,
    borderRadius: 12,
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.06)',
    overflowY: 'auto',
  },
  rightCol: {
    display: 'flex',
    flexDirection: 'column',
    padding: 12,
    borderRadius: 12,
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.06)',
    overflowY: 'auto',
  },
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
  mixerSection: {
    marginTop: 8,
  },
  optionalSection: {
    borderTop: '1px solid rgba(255,255,255,0.05)',
    paddingTop: 12,
  },
  optionalSummary: {
    fontSize: 12,
    color: '#888',
    cursor: 'pointer',
    fontWeight: 500,
  },
  optionalFields: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    marginTop: 10,
  },
  fieldLabel: {
    fontSize: 11,
    fontWeight: 600,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  textInput: {
    padding: '8px 10px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.08)',
    background: 'rgba(255,255,255,0.04)',
    color: '#d4d4d4',
    fontSize: 12,
    outline: 'none',
  },
  textarea: {
    padding: '8px 10px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.08)',
    background: 'rgba(255,255,255,0.04)',
    color: '#d4d4d4',
    fontSize: 12,
    outline: 'none',
    resize: 'vertical',
    fontFamily: 'inherit',
  },
};

export default StemStudio;
