// TrackSelector.tsx — Toggle grid for selecting stem extraction tracks
// Supports two modes: Extract (pick individual tracks) and SuperSep (pick separation level)
import React from 'react';
import { useTranslation } from 'react-i18next';
import { EXTRACT_TRACKS, TRACK_LABELS, TRACK_CATEGORIES } from '../../services/stemStudioApi';
import { SEPARATION_LEVELS, type SeparationLevel } from '../../services/supersepApi';
import { ToggleSwitch } from '../global-bar/BarSection';

interface TrackSelectorProps {
  selectedTracks: string[];
  onTracksChange: (tracks: string[]) => void;
  mode: 'extract' | 'supersep';
  onModeChange: (mode: 'extract' | 'supersep') => void;
  onExtract: () => void;
  isExtracting: boolean;
  canExtract: boolean;
  // SuperSep-specific
  sepLevel: SeparationLevel;
  onSepLevelChange: (level: SeparationLevel) => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  vocals: '#e879f9',
  instruments: '#60a5fa',
  drums: '#f97316',
  other: '#a3a3a3',
};

const CATEGORY_ACCENTS: Record<string, 'pink' | 'emerald' | 'sky' | 'purple' | 'amber'> = {
  vocals: 'pink',
  instruments: 'sky',
  drums: 'amber',
  other: 'purple',
};

export const TrackSelector: React.FC<TrackSelectorProps> = ({
  selectedTracks, onTracksChange, mode, onModeChange,
  onExtract, isExtracting, canExtract,
  sepLevel, onSepLevelChange,
}) => {
  const { t } = useTranslation();
  const toggleTrack = (track: string) => {
    if (selectedTracks.includes(track)) {
      onTracksChange(selectedTracks.filter(t => t !== track));
    } else {
      onTracksChange([...selectedTracks, track]);
    }
  };

  const selectAll = () => onTracksChange([...EXTRACT_TRACKS]);
  const clearAll = () => onTracksChange([]);

  // Group tracks by category for display
  const grouped = EXTRACT_TRACKS.reduce<Record<string, string[]>>((acc, t) => {
    const cat = TRACK_CATEGORIES[t] || 'other';
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(t);
    return acc;
  }, {});

  const isSupersep = mode === 'supersep';

  // Button label depends on mode
  const buttonLabel = isExtracting
    ? '⏳ Processing...'
    : isSupersep
      ? '▶ Separate Audio'
      : `▶ Extract ${selectedTracks.length} Track${selectedTracks.length !== 1 ? 's' : ''}`;

  // Can proceed?
  const canProceed = isSupersep
    ? canExtract  // only needs source audio
    : canExtract && selectedTracks.length > 0;  // needs source + tracks

  return (
    <div style={styles.container}>
      {/* Mode selector */}
      <div style={styles.modeRow}>
        <button
          onClick={() => onModeChange('extract')}
          style={{
            ...styles.modeBtn,
            background: mode === 'extract' ? 'rgba(167,139,250,0.15)' : 'transparent',
            color: mode === 'extract' ? '#a78bfa' : '#888',
            borderColor: mode === 'extract' ? 'rgba(167,139,250,0.3)' : 'rgba(255,255,255,0.08)',
          }}
        >
          🎵 Extract (DiT)
        </button>
        <button
          onClick={() => onModeChange('supersep')}
          style={{
            ...styles.modeBtn,
            background: mode === 'supersep' ? 'rgba(34,197,94,0.15)' : 'transparent',
            color: mode === 'supersep' ? '#22c55e' : '#888',
            borderColor: mode === 'supersep' ? 'rgba(34,197,94,0.3)' : 'rgba(255,255,255,0.08)',
          }}
        >
          🧠 SuperSep (ONNX)
        </button>
      </div>

      {/* SuperSep: Separation level dropdown */}
      {isSupersep && (
        <div style={styles.sepLevelSection}>
          <h3 style={styles.sectionTitle}>{t('stem.separationLevel')}</h3>
          <select
            value={sepLevel}
            onChange={e => onSepLevelChange(parseInt(e.target.value) as SeparationLevel)}
            style={styles.sepLevelSelect}
            disabled={isExtracting}
          >
            {SEPARATION_LEVELS.map(l => (
              <option key={l.value} value={l.value}>
                {l.label} — {l.description}
              </option>
            ))}
          </select>
          <p style={styles.sepLevelHint}>
            SuperSep uses ONNX neural networks to separate the audio into stems.
            Higher levels produce more stems but take longer.
          </p>
        </div>
      )}

      {/* Extract: Track selection grid */}
      {!isSupersep && (
        <>
          <h3 style={styles.sectionTitle}>{t('stem.selectTracks')}</h3>

          {/* Quick actions */}
          <div style={styles.quickActions}>
            <button onClick={selectAll} style={styles.quickBtn}>{t('stem.selectAllTracks')}</button>
            <button onClick={clearAll} style={styles.quickBtn}>{t('stem.clearTracks')}</button>
            <span style={styles.selectedCount}>{selectedTracks.length} {t('stem.selected')}</span>
          </div>

          {/* Track grid — Row 1: Vocals, Drums, Other | Row 2: Instruments */}
          <div style={styles.trackGrid}>
            {/* Row 1 */}
            <div style={styles.trackRow}>
              {(['vocals', 'drums', 'other'] as const).map(cat => {
                const tracks = grouped[cat];
                if (!tracks) return null;
                return (
                  <div key={cat} style={styles.categoryGroup}>
                    <div style={{ ...styles.categoryLabel, color: CATEGORY_COLORS[cat] }}>
                      <span>●</span> {cat.charAt(0).toUpperCase() + cat.slice(1)}
                    </div>
                    <div style={styles.categoryTracks}>
                      {tracks.map(track => (
                        <div key={track} style={styles.trackItem}>
                          <ToggleSwitch
                            checked={selectedTracks.includes(track)}
                            onChange={() => toggleTrack(track)}
                            accentColor={CATEGORY_ACCENTS[cat] || 'purple'}
                          />
                          <span style={{
                            ...styles.trackLabel,
                            color: selectedTracks.includes(track) ? '#d4d4d4' : '#888',
                          }}>
                            {TRACK_LABELS[track] || track}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
            {/* Row 2 — Instruments */}
            {grouped['instruments'] && (
              <div style={styles.categoryGroup}>
                <div style={{ ...styles.categoryLabel, color: CATEGORY_COLORS['instruments'] }}>
                  <span>●</span> Instruments
                </div>
                <div style={styles.categoryTracks}>
                  {grouped['instruments'].map(track => (
                    <div key={track} style={styles.trackItem}>
                      <ToggleSwitch
                        checked={selectedTracks.includes(track)}
                        onChange={() => toggleTrack(track)}
                        accentColor={CATEGORY_ACCENTS['instruments'] || 'sky'}
                      />
                      <span style={{
                        ...styles.trackLabel,
                        color: selectedTracks.includes(track) ? '#d4d4d4' : '#888',
                      }}>
                        {TRACK_LABELS[track] || track}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* Action button */}
      <button
        onClick={onExtract}
        disabled={!canProceed || isExtracting}
        style={{
          ...styles.extractBtn,
          background: isSupersep
            ? 'linear-gradient(135deg, #22c55e, #16a34a)'
            : 'linear-gradient(135deg, #8b5cf6, #6366f1)',
          opacity: (!canProceed || isExtracting) ? 0.5 : 1,
          cursor: (!canProceed || isExtracting) ? 'not-allowed' : 'pointer',
        }}
      >
        {buttonLabel}
      </button>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  modeRow: {
    display: 'flex',
    gap: 6,
  },
  modeBtn: {
    flex: 1,
    padding: '8px 12px',
    borderRadius: 8,
    border: '1px solid',
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    transition: 'all 0.15s ease',
  },
  sectionTitle: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
    color: '#d4d4d4',
    letterSpacing: '0.02em',
  },
  // SuperSep level
  sepLevelSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  sepLevelSelect: {
    padding: '8px 12px',
    borderRadius: 10,
    border: '1px solid rgba(255,255,255,0.1)',
    background: '#27272a',
    color: '#d4d4d8',
    fontSize: 13,
    outline: 'none',
    cursor: 'pointer',
    transition: 'border-color 0.15s ease',
  },
  sepLevelHint: {
    margin: 0,
    fontSize: 11,
    color: '#666',
    lineHeight: 1.5,
  },
  // Extract tracks
  quickActions: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  quickBtn: {
    padding: '4px 10px',
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.08)',
    background: 'rgba(255,255,255,0.04)',
    color: '#aaa',
    fontSize: 11,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.1s ease',
  },
  selectedCount: {
    fontSize: 11,
    color: '#666',
    marginLeft: 'auto',
  },
  trackGrid: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  trackRow: {
    display: 'flex',
    gap: 16,
    flexWrap: 'wrap' as const,
  },
  categoryGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  categoryTracks: {
    display: 'flex',
    flexWrap: 'wrap' as const,
    gap: 4,
  },
  categoryLabel: {
    fontSize: 10,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    padding: '2px 0',
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
  trackItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '4px 8px',
    borderRadius: 6,
    cursor: 'pointer',
    transition: 'background 0.1s ease',
  },
  trackLabel: {
    fontSize: 13,
    fontWeight: 500,
    transition: 'color 0.1s ease',
  },
  extractBtn: {
    padding: '10px 16px',
    borderRadius: 10,
    border: 'none',
    color: '#fff',
    fontSize: 14,
    fontWeight: 600,
    transition: 'all 0.15s ease',
    marginTop: 4,
  },
};

