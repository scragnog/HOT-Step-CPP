// StemMixer.tsx — Interactive stem mixer for SuperSep
//
// Displays separated stems with per-stem volume/mute controls,
// categorized by type (vocals, instruments, drums, other).
// Supports real-time preview via Web Audio API.

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import type { StemInfo } from '../../services/supersepApi';
import { getStemAudioUrl, recombineStems } from '../../services/supersepApi';

interface StemControl {
  index: number;
  volume: number;
  muted: boolean;
}

interface StemMixerProps {
  jobId: string;
  stems: StemInfo[];
  onRecombine?: (audioBlob: Blob) => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  vocals: '#e879f9',      // fuchsia
  instruments: '#60a5fa', // blue
  drums: '#f97316',       // orange
  other: '#a3a3a3',       // gray
};

const CATEGORY_ORDER = ['vocals', 'instruments', 'drums', 'other'];

export const StemMixer: React.FC<StemMixerProps> = ({ jobId, stems, onRecombine }) => {
  const [controls, setControls] = useState<StemControl[]>(() =>
    stems.map((s) => ({ index: s.index, volume: 1.0, muted: false }))
  );
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecombining, setIsRecombining] = useState(false);
  const [soloIndex, setSoloIndex] = useState<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodesRef = useRef<Map<number, { source: AudioBufferSourceNode; gain: GainNode }>>(new Map());
  const buffersRef = useRef<Map<number, AudioBuffer>>(new Map());
  const [loadedStems, setLoadedStems] = useState<Set<number>>(new Set());
  const [loadingStems, setLoadingStems] = useState(false);

  // Group stems by category
  const groupedStems = useMemo(() => {
    const groups: Record<string, StemInfo[]> = {};
    for (const s of stems) {
      const cat = s.category || 'other';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(s);
    }
    return CATEGORY_ORDER.filter(c => groups[c]).map(c => ({ category: c, stems: groups[c] }));
  }, [stems]);

  const updateControl = useCallback((index: number, partial: Partial<StemControl>) => {
    setControls(prev => prev.map(c => c.index === index ? { ...c, ...partial } : c));
  }, []);

  const toggleMute = useCallback((index: number) => {
    setControls(prev => prev.map(c => c.index === index ? { ...c, muted: !c.muted } : c));
  }, []);

  const toggleSolo = useCallback((index: number) => {
    setSoloIndex(prev => prev === index ? null : index);
  }, []);

  // Load audio buffers for preview
  const loadBuffers = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate: 44100 });
    }
    const ctx = audioContextRef.current;
    setLoadingStems(true);

    const loaded = new Set<number>();
    for (const stem of stems) {
      if (buffersRef.current.has(stem.index)) {
        loaded.add(stem.index);
        continue;
      }
      try {
        const url = getStemAudioUrl(jobId, stem.index);
        const res = await fetch(url);
        const arrayBuf = await res.arrayBuffer();
        const audioBuf = await ctx.decodeAudioData(arrayBuf);
        buffersRef.current.set(stem.index, audioBuf);
        loaded.add(stem.index);
      } catch (err) {
        console.warn(`Failed to load stem ${stem.index}:`, err);
      }
    }
    setLoadedStems(loaded);
    setLoadingStems(false);
  }, [jobId, stems]);

  // Start/stop preview playback
  const togglePlayback = useCallback(async () => {
    if (isPlaying) {
      // Stop
      sourceNodesRef.current.forEach(({ source }) => {
        try { source.stop(); } catch {}
      });
      sourceNodesRef.current.clear();
      setIsPlaying(false);
      return;
    }

    // Load if needed
    if (loadedStems.size < stems.length) {
      await loadBuffers();
    }

    const ctx = audioContextRef.current;
    if (!ctx) return;
    if (ctx.state === 'suspended') await ctx.resume();

    // Create source + gain for each stem
    const startTime = ctx.currentTime + 0.05;
    for (const stem of stems) {
      const buf = buffersRef.current.get(stem.index);
      if (!buf) continue;

      const source = ctx.createBufferSource();
      source.buffer = buf;

      const gain = ctx.createGain();
      const ctrl = controls.find(c => c.index === stem.index);
      const effectiveMuted = ctrl?.muted || (soloIndex !== null && soloIndex !== stem.index);
      gain.gain.value = effectiveMuted ? 0 : (ctrl?.volume ?? 1.0);

      source.connect(gain);
      gain.connect(ctx.destination);
      source.start(startTime);

      sourceNodesRef.current.set(stem.index, { source, gain });
      source.onended = () => {
        sourceNodesRef.current.delete(stem.index);
        if (sourceNodesRef.current.size === 0) {
          setIsPlaying(false);
        }
      };
    }
    setIsPlaying(true);
  }, [isPlaying, stems, controls, soloIndex, loadedStems, loadBuffers]);

  // Update gain nodes in real-time when controls change
  useEffect(() => {
    sourceNodesRef.current.forEach((node, idx) => {
      const ctrl = controls.find(c => c.index === idx);
      const effectiveMuted = ctrl?.muted || (soloIndex !== null && soloIndex !== idx);
      node.gain.gain.setValueAtTime(
        effectiveMuted ? 0 : (ctrl?.volume ?? 1.0),
        audioContextRef.current?.currentTime ?? 0
      );
    });
  }, [controls, soloIndex]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      sourceNodesRef.current.forEach(({ source }) => {
        try { source.stop(); } catch {}
      });
      audioContextRef.current?.close();
    };
  }, []);

  const handleRecombine = useCallback(async () => {
    setIsRecombining(true);
    try {
      const stemControls = controls.map(c => ({
        index: c.index,
        volume: (soloIndex !== null && soloIndex !== c.index) ? 0 : (c.muted ? 0 : c.volume),
        muted: c.muted || (soloIndex !== null && soloIndex !== c.index),
      }));
      const blob = await recombineStems(jobId, stemControls);
      onRecombine?.(blob);
    } catch (err) {
      console.error('Recombine failed:', err);
    } finally {
      setIsRecombining(false);
    }
  }, [jobId, controls, soloIndex, onRecombine]);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>🎛️ Stem Mixer</h3>
        <div style={styles.headerButtons}>
          <button
            onClick={togglePlayback}
            disabled={loadingStems}
            style={{ ...styles.button, ...(isPlaying ? styles.buttonActive : {}) }}
          >
            {loadingStems ? '⏳ Loading...' : isPlaying ? '⏹ Stop' : '▶ Preview'}
          </button>
          <button
            onClick={handleRecombine}
            disabled={isRecombining}
            style={{ ...styles.button, ...styles.buttonPrimary }}
          >
            {isRecombining ? '⏳ Mixing...' : '🔀 Recombine'}
          </button>
        </div>
      </div>

      <div style={styles.stemList}>
        {groupedStems.map(({ category, stems: catStems }) => (
          <div key={category} style={styles.categoryGroup}>
            <div style={{
              ...styles.categoryLabel,
              borderColor: CATEGORY_COLORS[category] || '#666',
            }}>
              <span style={{ color: CATEGORY_COLORS[category] }}>●</span>
              {' '}{category.charAt(0).toUpperCase() + category.slice(1)}
              <span style={styles.stemCount}>({catStems.length})</span>
            </div>

            {catStems.map(stem => {
              const ctrl = controls.find(c => c.index === stem.index)!;
              const isSoloed = soloIndex === stem.index;
              const effectiveMuted = ctrl.muted || (soloIndex !== null && !isSoloed);

              return (
                <div
                  key={stem.index}
                  style={{
                    ...styles.stemRow,
                    opacity: effectiveMuted ? 0.5 : 1,
                  }}
                >
                  <div style={styles.stemInfo}>
                    <span style={styles.stemName}>{stem.name}</span>
                    <span style={styles.stageBadge}>S{stem.stage}</span>
                  </div>

                  <div style={styles.stemControls}>
                    <button
                      onClick={() => toggleMute(stem.index)}
                      style={{
                        ...styles.tinyButton,
                        background: ctrl.muted ? '#ef4444' : '#333',
                      }}
                      title={ctrl.muted ? 'Unmute' : 'Mute'}
                    >M</button>

                    <button
                      onClick={() => toggleSolo(stem.index)}
                      style={{
                        ...styles.tinyButton,
                        background: isSoloed ? '#eab308' : '#333',
                        color: isSoloed ? '#000' : '#fff',
                      }}
                      title={isSoloed ? 'Unsolo' : 'Solo'}
                    >S</button>

                    <input
                      type="range"
                      min="0"
                      max="200"
                      value={Math.round(ctrl.volume * 100)}
                      onChange={(e) => updateControl(stem.index, { volume: parseInt(e.target.value) / 100 })}
                      style={styles.slider}
                    />
                    <span style={styles.volumeLabel}>
                      {Math.round(ctrl.volume * 100)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

// ── Styles ──────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: 'rgba(0,0,0,0.3)',
    borderRadius: 12,
    padding: 16,
    border: '1px solid rgba(255,255,255,0.1)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    margin: 0,
    fontSize: 16,
    fontWeight: 600,
    color: '#e5e5e5',
  },
  headerButtons: {
    display: 'flex',
    gap: 8,
  },
  button: {
    padding: '6px 14px',
    borderRadius: 8,
    border: '1px solid rgba(255,255,255,0.15)',
    background: 'rgba(255,255,255,0.08)',
    color: '#e5e5e5',
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 500,
    transition: 'all 0.15s ease',
  },
  buttonActive: {
    background: '#ef4444',
    borderColor: '#ef4444',
    color: '#fff',
  },
  buttonPrimary: {
    background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
    borderColor: '#8b5cf6',
    color: '#fff',
  },
  stemList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 12,
  },
  categoryGroup: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: 4,
  },
  categoryLabel: {
    fontSize: 11,
    fontWeight: 700,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    color: '#a3a3a3',
    padding: '4px 0',
    borderBottom: '1px solid',
  },
  stemCount: {
    fontWeight: 400,
    color: '#666',
    marginLeft: 4,
  },
  stemRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '6px 8px',
    borderRadius: 8,
    background: 'rgba(255,255,255,0.03)',
    transition: 'opacity 0.15s ease',
  },
  stemInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    minWidth: 140,
  },
  stemName: {
    fontSize: 13,
    color: '#d4d4d4',
    fontWeight: 500,
  },
  stageBadge: {
    fontSize: 10,
    padding: '1px 5px',
    borderRadius: 4,
    background: 'rgba(255,255,255,0.08)',
    color: '#888',
    fontWeight: 600,
  },
  stemControls: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  tinyButton: {
    width: 24,
    height: 24,
    borderRadius: 4,
    border: 'none',
    color: '#fff',
    fontSize: 10,
    fontWeight: 700,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background 0.1s ease',
  },
  slider: {
    width: 120,
    height: 4,
    cursor: 'pointer',
    accentColor: '#8b5cf6',
  },
  volumeLabel: {
    fontSize: 11,
    color: '#888',
    width: 38,
    textAlign: 'right' as const,
    fontFamily: 'monospace',
  },
};

export default StemMixer;
