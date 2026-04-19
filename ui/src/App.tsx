// App.tsx — Root application component
//
// Ported from hot-step-9000: 3-panel resizable layout with
// Sidebar | CreatePanel | SongList | RightSidebar | Player.

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './context/AuthContext';
import { useGenerationStore } from './stores/useGenerationStore';
import { usePersistedState } from './hooks/usePersistedState';
import { songApi } from './services/api';
import { Sidebar } from './components/sidebar/Sidebar';
import { CreatePanel } from './components/create/CreatePanel';
import { SongList } from './components/library/SongList';
import { JobQueue } from './components/queue/JobQueue';
import { Player } from './components/player/Player';
import { WaveformPlayer, type WaveformPlayerHandle } from './components/player/WaveformPlayer';
import { SectionMarkers } from './components/player/SectionMarkers';
import { SpectrumAnalyzer } from './components/player/SpectrumAnalyzer';
import { RightSidebar } from './components/details/RightSidebar';
import { Toast, type ToastType } from './components/shared/Toast';
import { ConfirmDialog } from './components/shared/ConfirmDialog';
import { DownloadModal } from './components/shared/DownloadModal';
import { SettingsPanel, type AppSettings, DEFAULT_SETTINGS } from './components/settings/SettingsPanel';
import type { Song, GenerationParams } from './types';

const App: React.FC = () => {
  const { token, isLoading } = useAuth();
  const [activeView, setActiveView] = useState('create');
  const [songs, setSongs] = useState<Song[]>([]);
  const [currentSong, setCurrentSong] = useState<Song | null>(null);
  const [selectedSong, setSelectedSong] = useState<Song | null>(null);

  // Resizable panel widths (persisted)
  const [createPanelWidth, setCreatePanelWidth] = usePersistedState('ace-createPanelWidth', 490);
  const [rightSidebarWidth, setRightSidebarWidth] = usePersistedState('ace-rightSidebarWidth', 360);
  const [showRightSidebar, setShowRightSidebar] = useState(true);

  // Settings state (persisted)
  const [settings, setSettings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

  // Player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(() => {
    const stored = localStorage.getItem('volume');
    return stored ? parseFloat(stored) : 0.8;
  });
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isShuffle, setIsShuffle] = useState(false);
  const [repeatMode, setRepeatMode] = useState<'none' | 'all' | 'one'>('none');
  const wavesurferRef = useRef<WaveformPlayerHandle>(null);
  const currentSongIdRef = useRef<string | null>(null);
  const [playMastered, setPlayMastered] = useState(false);
  const [currentAudioUrl, setCurrentAudioUrl] = useState<string | null>(null);

  // Spectrum analyzer state (persisted)
  const [spectrumEnabled, setSpectrumEnabled] = usePersistedState('ace-spectrum-enabled', false);
  const [spectrumMediaEl, setSpectrumMediaEl] = useState<HTMLMediaElement | null>(null);

  // Toast state
  const [toast, setToast] = useState<{ message: string; type: ToastType; isVisible: boolean }>({
    message: '', type: 'success', isVisible: false,
  });

  // Confirm dialog state
  const [confirmDialog, setConfirmDialog] = useState<{
    title: string; message: string; confirmLabel?: string; danger?: boolean; onConfirm: () => void;
  } | null>(null);

  const [isShutdown, setIsShutdown] = useState(false);

  // Download modal state
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);

  const showToast = (message: string, type: ToastType = 'success') => {
    setToast({ message, type, isVisible: true });
  };

  // Song created callback — add to library
  const handleSongCreated = useCallback((song: Song) => {
    setSongs(prev => [song, ...prev.filter(s => s.id !== song.id)]);
  }, []);

  // Generation store
  const genStore = useGenerationStore(handleSongCreated);

  // Load songs on mount
  useEffect(() => {
    if (!token) return;
    songApi.list(token)
      .then(({ songs }) => setSongs(songs))
      .catch(err => console.error('[App] Failed to load songs:', err));
  }, [token]);

  // Refresh songs (kept for future use — e.g. after external DB changes)
  // const refreshSongsList = useCallback(() => {
  //   if (!token) return;
  //   songApi.list(token)
  //     .then(({ songs }) => setSongs(songs))
  //     .catch(err => console.error('[App] Refresh failed:', err));
  // }, [token]);

  // Handle generation
  const handleGenerate = useCallback((params: GenerationParams) => {
    if (!token) return;
    // Inject settings into generation params
    const enrichedParams = {
      ...params,
      coResident: settings.coResident,
      cacheLmCodes: settings.cacheLmCodes,
    };
    genStore.submit(enrichedParams, token).catch(err => {
      console.error('[App] Generation failed:', err);
    });
  }, [token, genStore, settings]);

  // Handle delete
  const handleDelete = useCallback(async (song: Song) => {
    if (!token) return;
    await songApi.delete(song.id, token);
    setSongs(prev => prev.filter(s => s.id !== song.id));
    if (currentSong?.id === song.id) setCurrentSong(null);
    if (selectedSong?.id === song.id) setSelectedSong(null);
    showToast(`Deleted "${song.title}"`, 'success');
  }, [token, currentSong, selectedSong]);

  // Handle bulk delete
  const handleBulkDelete = useCallback((ids: string[]) => {
    if (!token || ids.length === 0) return;
    setConfirmDialog({
      title: `Delete ${ids.length} song${ids.length !== 1 ? 's' : ''}?`,
      message: `This will permanently delete ${ids.length} track${ids.length !== 1 ? 's' : ''} and their audio files from disk.`,
      confirmLabel: `Delete ${ids.length}`,
      danger: true,
      onConfirm: async () => {
        setConfirmDialog(null);
        try {
          await songApi.bulkDelete(ids, token);
          const idSet = new Set(ids);
          setSongs(prev => prev.filter(s => !idSet.has(s.id)));
          if (currentSong && idSet.has(currentSong.id)) setCurrentSong(null);
          if (selectedSong && idSet.has(selectedSong.id)) setSelectedSong(null);
          showToast(`Deleted ${ids.length} track${ids.length !== 1 ? 's' : ''}`, 'success');
        } catch (err: any) {
          showToast(`Delete failed: ${err.message}`, 'error');
        }
      },
    });
  }, [token, currentSong, selectedSong]);

  // Reuse handler
  const [reuseData, setReuseData] = useState<{ song: Song; timestamp: number } | null>(null);
  const handleReuse = useCallback((song: Song) => {
    setReuseData({ song, timestamp: Date.now() });
    setActiveView('create');
  }, []);

  // Song update handler
  const handleSongUpdate = useCallback((updatedSong: Song) => {
    setSongs(prev => prev.map(s => s.id === updatedSong.id ? updatedSong : s));
    if (currentSong?.id === updatedSong.id) setCurrentSong(updatedSong);
    if (selectedSong?.id === updatedSong.id) setSelectedSong(updatedSong);
  }, [currentSong, selectedSong]);

  // Rename handler — PATCH title to server + update local state
  const handleRename = useCallback(async (song: Song, newTitle: string) => {
    try {
      const { song: updated } = await songApi.update(song.id, { title: newTitle }, token!);
      const normalized = { ...song, ...updated, title: newTitle };
      handleSongUpdate(normalized);
    } catch (err: any) {
      showToast(`Rename failed: ${err.message}`, 'error');
    }
  }, [token, handleSongUpdate]);

  // ── Player Logic (wavesurfer-based) ─────────────────────────
  const playSong = useCallback((song: Song) => {
    const ws = wavesurferRef.current;
    if (!ws) return;

    // Always show song details when interacting with a song
    setSelectedSong(song);
    setShowRightSidebar(true);

    if (currentSongIdRef.current === song.id) {
      // Toggle play/pause for same song
      ws.playPause();
      return;
    }

    // New song
    currentSongIdRef.current = song.id;
    setCurrentSong(song);
    // Auto-select mastered if available
    const useMastered = !!(song.masteredAudioUrl);
    setPlayMastered(useMastered);
    const url = useMastered ? song.masteredAudioUrl! : song.audioUrl;
    setCurrentAudioUrl(url);
    ws.loadUrl(url);
  }, []);

  // Auto-play when wavesurfer finishes loading the new URL
  const handleWaveformReady = useCallback((_dur: number) => {
    wavesurferRef.current?.play();
  }, []);

  const toggleMastered = useCallback(() => {
    const ws = wavesurferRef.current;
    if (!ws || !currentSong) return;

    const wantMastered = !playMastered;
    setPlayMastered(wantMastered);

    const newUrl = wantMastered && currentSong.masteredAudioUrl
      ? currentSong.masteredAudioUrl
      : currentSong.audioUrl;

    // Save position, load new URL, restore position after ready
    const pos = ws.getCurrentTime();
    const dur = ws.getDuration();
    setCurrentAudioUrl(newUrl);
    ws.loadUrl(newUrl);
    // Seek back after load — wavesurfer 'ready' event fires before our onReady callback
    const seekBack = () => {
      if (dur > 0) ws.seekTo(pos / dur);
      ws.play();
    };
    // Use a one-shot listener approach via timeout — wavesurfer re-decodes on load
    setTimeout(seekBack, 300);
  }, [currentSong, playMastered]);

  const togglePlay = useCallback(() => {
    if (!currentSong) return;
    wavesurferRef.current?.playPause();
  }, [currentSong]);

  const handleSeek = useCallback((time: number) => {
    const ws = wavesurferRef.current;
    const dur = ws?.getDuration() ?? 0;
    if (ws && dur > 0) ws.seekTo(time / dur);
  }, []);

  const playNext = useCallback(() => {
    if (!currentSong || songs.length === 0) return;
    const idx = songs.findIndex(s => s.id === currentSong.id);
    const next = isShuffle
      ? songs[Math.floor(Math.random() * songs.length)]
      : songs[(idx + 1) % songs.length];
    if (next) playSong(next);
  }, [currentSong, songs, isShuffle, playSong]);

  const playPrevious = useCallback(() => {
    if (!currentSong || songs.length === 0) return;
    const idx = songs.findIndex(s => s.id === currentSong.id);
    const prev = songs[(idx - 1 + songs.length) % songs.length];
    if (prev) playSong(prev);
  }, [currentSong, songs, playSong]);

  // Handle wavesurfer finish (replaces 'ended' event)
  const handleWaveformFinish = useCallback(() => {
    if (repeatMode === 'one') {
      wavesurferRef.current?.seekTo(0);
      wavesurferRef.current?.play();
    } else if (repeatMode === 'all' || songs.length > 1) {
      playNext();
    } else {
      setIsPlaying(false);
    }
  }, [repeatMode, playNext, songs.length]);

  // Sync volume to localStorage
  useEffect(() => {
    localStorage.setItem('volume', String(volume));
  }, [volume]);

  // ── Render ────────────────────────────────────────────────

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-suno text-zinc-400">
        <div className="text-center">
          <div className="text-4xl mb-4">⚡</div>
          <div className="text-lg font-medium">Loading HOT-Step...</div>
        </div>
      </div>
    );
  }

  if (isShutdown) {
    return (
      <div className="fixed inset-0 z-[200] flex flex-col items-center justify-center bg-black text-white">
        <div className="text-6xl mb-6">👋</div>
        <h1 className="text-2xl font-bold mb-2">HOT-Step CPP has shut down</h1>
        <p className="text-zinc-400">You may now close this browser tab.</p>
      </div>
    );
  }

  // Main content renderer
  const renderContent = () => {
    if (activeView === 'settings') {
      return (
        <div className="flex-1 overflow-y-auto">
          <SettingsPanel
            settings={settings}
            onSettingsChange={setSettings}
          />
        </div>
      );
    }

    if (activeView === 'library') {
      return (
        <div className="flex-1 overflow-y-auto">
          <SongList
            songs={songs}
            currentSongId={currentSong?.id}
            onPlay={playSong}
            onDelete={handleDelete}
            onBulkDelete={handleBulkDelete}
            onSelect={(s) => { setSelectedSong(s); setShowRightSidebar(true); }}
            onReuse={handleReuse}
            onDownload={setDownloadSong}
            onRename={handleRename}
          />
        </div>
      );
    }

    // Default: create view
    return (
      <div className="flex flex-1 overflow-hidden">
        {/* Create Panel — resizable */}
        <div
          className="flex-shrink-0 h-full border-r border-zinc-200 dark:border-white/5"
          style={{ width: createPanelWidth }}
        >
          <CreatePanel
            onGenerate={handleGenerate}
            isGenerating={genStore.isGenerating}
            reuseData={reuseData}
          />
        </div>

        {/* Left resize handle */}
        <div
          className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
          onMouseDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startW = createPanelWidth;
            const onMove = (ev: MouseEvent) => {
              const newW = Math.min(700, Math.max(360, startW + (ev.clientX - startX)));
              setCreatePanelWidth(newW);
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
          }}
        >
          <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
        </div>

        {/* Song List + Queue */}
        <div className="flex-1 min-w-0 overflow-y-auto">
          <JobQueue
            jobs={genStore.jobs}
            onCancel={genStore.cancel}
            onClearCompleted={genStore.clearCompleted}
          />
          <SongList
            songs={songs}
            currentSongId={currentSong?.id}
            onPlay={playSong}
            onDelete={handleDelete}
            onBulkDelete={handleBulkDelete}
            onSelect={(s) => { setSelectedSong(s); setShowRightSidebar(true); }}
            onReuse={handleReuse}
            onDownload={setDownloadSong}
            onRename={handleRename}
          />
        </div>

        {/* Right Sidebar — resizable */}
        {showRightSidebar && selectedSong && (
          <>
            {/* Right resize handle */}
            <div
              className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startW = rightSidebarWidth;
                const onMove = (ev: MouseEvent) => {
                  const newW = Math.min(600, Math.max(280, startW + startX - ev.clientX));
                  setRightSidebarWidth(newW);
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
              }}
            >
              <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
            </div>
            <div
              className="flex-shrink-0 h-full bg-zinc-50 dark:bg-suno-panel z-10 border-l border-zinc-200 dark:border-white/5"
              style={{ width: rightSidebarWidth }}
            >
              <RightSidebar
                song={selectedSong}
                onClose={() => setShowRightSidebar(false)}
                onReuse={handleReuse}
                onDelete={handleDelete}
                onPlay={playSong}
                isPlaying={isPlaying && currentSong?.id === selectedSong?.id}
                onDownload={setDownloadSong}
              />
            </div>
          </>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-screen bg-white dark:bg-suno text-zinc-900 dark:text-white font-sans antialiased selection:bg-pink-500/30 transition-all duration-300">
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          activeView={activeView}
          onViewChange={setActiveView}
          onQuit={() => {
            setConfirmDialog({
              title: 'Quit HOT-Step CPP',
              message: 'Are you sure you wish to shut down HOT-Step CPP? This will stop the engine and all servers.',
              confirmLabel: 'Shut Down',
              danger: true,
              onConfirm: async () => {
                setConfirmDialog(null);
                try { await fetch('/api/shutdown', { method: 'POST' }); } catch { /* shutting down */ }
                setIsShutdown(true);
              },
            });
          }}
        />

        <main className="flex-1 flex overflow-hidden relative">
          {renderContent()}
        </main>
      </div>

      {/* ── Bottom Player Area: Markers → Waveform → Transport ── */}
      <div className="flex-shrink-0 bg-zinc-950 border-t border-white/5">
        <SectionMarkers audioUrl={currentAudioUrl ?? undefined} duration={duration} />
        {spectrumEnabled && (
          <SpectrumAnalyzer mediaElement={spectrumMediaEl} isPlaying={isPlaying} />
        )}
        <WaveformPlayer
          ref={wavesurferRef}
          volume={volume}
          playbackRate={playbackRate}
          onTimeUpdate={setCurrentTime}
          onDurationChange={setDuration}
          onPlayChange={setIsPlaying}
          onFinish={handleWaveformFinish}
          onReady={(dur) => {
            handleWaveformReady(dur);
            // Capture the media element for the spectrum analyzer
            setSpectrumMediaEl(wavesurferRef.current?.getMediaElement() ?? null);
          }}
        />
        <Player
          currentSong={currentSong}
          isPlaying={isPlaying}
          onTogglePlay={togglePlay}
          currentTime={currentTime}
          duration={duration}
          onSeek={handleSeek}
          onNext={playNext}
          onPrevious={playPrevious}
          volume={volume}
          onVolumeChange={setVolume}
          playbackRate={playbackRate}
          onPlaybackRateChange={setPlaybackRate}
          audioRef={wavesurferRef as any}
          isShuffle={isShuffle}
          onToggleShuffle={() => setIsShuffle(!isShuffle)}
          repeatMode={repeatMode}
          onToggleRepeat={() => setRepeatMode(prev => prev === 'none' ? 'all' : prev === 'all' ? 'one' : 'none')}
          onReusePrompt={() => currentSong && handleReuse(currentSong)}
          onDelete={() => currentSong && handleDelete(currentSong)}
          onDownload={() => currentSong && setDownloadSong(currentSong)}
          playMastered={playMastered}
          onToggleMastered={toggleMastered}
          spectrumEnabled={spectrumEnabled}
          onToggleSpectrum={() => setSpectrumEnabled(!spectrumEnabled)}
        />
      </div>

      {/* Modals */}
      <Toast
        message={toast.message}
        type={toast.type}
        isVisible={toast.isVisible}
        onClose={() => setToast(prev => ({ ...prev, isVisible: false }))}
      />
      <ConfirmDialog
        isOpen={confirmDialog !== null}
        title={confirmDialog?.title ?? ''}
        message={confirmDialog?.message ?? ''}
        confirmLabel={confirmDialog?.confirmLabel}
        danger={confirmDialog?.danger}
        onConfirm={() => confirmDialog?.onConfirm()}
        onCancel={() => setConfirmDialog(null)}
      />
      {downloadSong && (
        <DownloadModal
          song={downloadSong}
          isOpen={true}
          onClose={() => setDownloadSong(null)}
          defaultFormat={settings.downloadFormat}
          defaultMp3Bitrate={settings.downloadMp3Bitrate}
          defaultOpusBitrate={settings.downloadOpusBitrate}
        />
      )}
    </div>
  );
};

export default App;
