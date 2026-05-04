// StemMixer.tsx — Shared interactive stem mixer
//
// Controlled component: parent owns volume/mute state.
// Supports real-time preview via Web Audio API.
// Decoupled from any specific stem source — accepts audio URLs directly.

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { Download } from 'lucide-react';

export interface MixerStemInfo {
  name: string;
  category: 'vocals' | 'instruments' | 'drums' | 'other';
  audioUrl: string;    // direct URL to WAV/audio file
  index: number;
  stage?: number;      // optional — used by SuperSep stems
}

export interface StemControl {
  index: number;
  volume: number;
  muted: boolean;
}

interface StemMixerProps {
  jobId: string;
  stems: MixerStemInfo[];
  controls: StemControl[];
  onControlsChange: (controls: StemControl[]) => void;
  onClose?: () => void;
  onDownloadStem?: (stem: MixerStemInfo) => void;
  onDownloadAll?: () => void;
}

const CATEGORY_COLORS: Record<string, string> = {
  vocals: '#e879f9',      // fuchsia
  instruments: '#60a5fa', // blue
  drums: '#f97316',       // orange
  other: '#a3a3a3',       // gray
};

const CATEGORY_ORDER = ['vocals', 'instruments', 'drums', 'other'];

export const StemMixer: React.FC<StemMixerProps> = ({ jobId, stems, controls, onControlsChange, onClose, onDownloadStem, onDownloadAll }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [soloIndex, setSoloIndex] = useState<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodesRef = useRef<Map<number, { source: AudioBufferSourceNode; gain: GainNode }>>(new Map());
  const buffersRef = useRef<Map<number, AudioBuffer>>(new Map());
  const [loadedStems, setLoadedStems] = useState<Set<number>>(new Set());
  const [loadingStems, setLoadingStems] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const playbackOffsetRef = useRef(0);
  const playbackStartTimeRef = useRef(0);
  const animationFrameRef = useRef<number>();

  // Reset buffers AND stop playback when stems change (different job loaded)
  useEffect(() => {
    // Stop any active playback — null onended first to prevent stale callbacks
    sourceNodesRef.current.forEach(({ source }) => {
      source.onended = null;
      try { source.stop(); } catch {}
    });
    sourceNodesRef.current.clear();
    setIsPlaying(false);

    // Clear buffers
    buffersRef.current.clear();
    setLoadedStems(new Set());
    setDuration(0);
    setCurrentTime(0);
    playbackOffsetRef.current = 0;
  }, [jobId]);

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };


  // Group stems by category
  const groupedStems = useMemo(() => {
    const groups: Record<string, MixerStemInfo[]> = {};
    for (const s of stems) {
      const cat = s.category || 'other';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(s);
    }
    return CATEGORY_ORDER.filter(c => groups[c]).map(c => ({ category: c, stems: groups[c] }));
  }, [stems]);

  const updateControl = useCallback((index: number, partial: Partial<StemControl>) => {
    onControlsChange(controls.map(c => c.index === index ? { ...c, ...partial } : c));
  }, [controls, onControlsChange]);

  const toggleMute = useCallback((index: number) => {
    onControlsChange(controls.map(c => c.index === index ? { ...c, muted: !c.muted } : c));
  }, [controls, onControlsChange]);

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
        const url = stem.audioUrl;
        const res = await fetch(url);
        const arrayBuf = await res.arrayBuffer();
        const audioBuf = await ctx.decodeAudioData(arrayBuf);
        buffersRef.current.set(stem.index, audioBuf);
        if (stem.index === stems[0].index) {
          setDuration(audioBuf.duration);
        }

        loaded.add(stem.index);
      } catch (err) {
        console.warn(`Failed to load stem ${stem.index}:`, err);
      }
    }
    setLoadedStems(loaded);
    setLoadingStems(false);
  }, [stems]);


  const updateProgress = useCallback(() => {
    if (audioContextRef.current && isPlaying) {
      const elapsed = audioContextRef.current.currentTime - playbackStartTimeRef.current;
      let newTime = playbackOffsetRef.current + elapsed;
      if (duration > 0 && newTime >= duration) {
         newTime = 0;
         setIsPlaying(false);
         playbackOffsetRef.current = 0;
      } else {
         setCurrentTime(newTime);
         animationFrameRef.current = requestAnimationFrame(updateProgress);
      }
    }
  }, [isPlaying, duration]);

  useEffect(() => {
    if (isPlaying) {
      animationFrameRef.current = requestAnimationFrame(updateProgress);
    } else {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    }
    return () => { if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current); };
  }, [isPlaying, updateProgress]);

  // Start/stop preview playback
  const togglePlayback = useCallback(async () => {
    if (isPlaying) {
      // Pause
      sourceNodesRef.current.forEach(({ source }) => {
        try { source.stop(); } catch {}
      });
      sourceNodesRef.current.clear();
      setIsPlaying(false);
      if (audioContextRef.current) {
         playbackOffsetRef.current += (audioContextRef.current.currentTime - playbackStartTimeRef.current);
      }
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
      source.start(startTime, playbackOffsetRef.current);

      sourceNodesRef.current.set(stem.index, { source, gain });
      source.onended = () => {
        sourceNodesRef.current.delete(stem.index);
        if (sourceNodesRef.current.size === 0) {
          setIsPlaying(false);
          playbackOffsetRef.current = 0;
          setCurrentTime(0);
        }
      };
    }
    playbackStartTimeRef.current = ctx.currentTime;
    setIsPlaying(true);
  }, [isPlaying, stems, controls, soloIndex, loadedStems, loadBuffers, duration]);


  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    playbackOffsetRef.current = time;

    // If playing, stop old sources and immediately restart from new position
    // (avoids the stale-closure problem of setTimeout + togglePlayback)
    if (isPlaying && audioContextRef.current) {
      // Null onended on old sources BEFORE stopping — prevents stale callbacks
      // from deleting new source nodes (same index) out of the map
      sourceNodesRef.current.forEach(({ source }) => {
        source.onended = null;
        try { source.stop(); } catch {}
      });
      sourceNodesRef.current.clear();

      const ctx = audioContextRef.current;
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
        source.start(startTime, time);
        sourceNodesRef.current.set(stem.index, { source, gain });
        source.onended = () => {
          sourceNodesRef.current.delete(stem.index);
          if (sourceNodesRef.current.size === 0) {
            setIsPlaying(false);
            playbackOffsetRef.current = 0;
            setCurrentTime(0);
          }
        };
      }
      playbackStartTimeRef.current = ctx.currentTime;
    }
  }, [isPlaying, stems, controls, soloIndex]);

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

  const content = (
    <div style={{ ...styles.container, width: '100%', maxWidth: '600px', maxHeight: '90vh', display: 'flex', flexDirection: 'column' }}>
      <div style={styles.header}>
                <h3 style={styles.title}>🎛️ Stem Mixer</h3>
        {onClose && (
           <button onClick={onClose} className="text-zinc-400 hover:text-white transition-colors" style={{ marginLeft: 'auto', marginRight: 16 }}>✕</button>
        )}
        <div style={styles.headerButtons}>
          <button
            onClick={togglePlayback}
            disabled={loadingStems}
            style={{ ...styles.button, ...(isPlaying ? styles.buttonActive : {}) }}
          >
            {loadingStems ? '⏳ Loading...' : isPlaying ? '⏹ Stop' : '▶ Preview'}
          </button>
          {onDownloadAll && (
            <button
              onClick={onDownloadAll}
              style={{ ...styles.button, display: 'flex', alignItems: 'center', gap: 4 }}
              title="Download all stems as ZIP"
            >
              <Download size={13} /> All
            </button>
          )}
        </div>
      </div>


      {duration > 0 && (
        <div style={{ padding: '16px 0', borderBottom: '1px solid rgba(255,255,255,0.05)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 12, color: '#a3a3a3', fontFamily: 'monospace', width: 40, textAlign: 'right' }}>{formatTime(currentTime)}</span>
          <input 
            type="range" 
            min="0" max={duration} step="0.1" 
            value={currentTime} 
            onChange={handleSeek}
            style={{ flex: 1, accentColor: '#ec4899', height: 4, cursor: 'pointer' }} 
          />
          <span style={{ fontSize: 12, color: '#a3a3a3', fontFamily: 'monospace', width: 40 }}>{formatTime(duration)}</span>
        </div>
      )}

      <div style={{ ...styles.stemList, overflowY: 'auto' }}>
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
                    {stem.stage !== undefined && (
                      <span style={styles.stageBadge}>S{stem.stage}</span>
                    )}
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

                    {onDownloadStem && (
                      <button
                        onClick={() => onDownloadStem(stem)}
                        style={{ ...styles.tinyButton, background: 'transparent', color: '#888' }}
                        title={`Download ${stem.name}`}
                      >
                        <Download size={12} />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );

  if (!onClose) return content;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }} onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
       {content}
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
