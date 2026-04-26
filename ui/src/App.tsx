// App.tsx — Root application component
//
// Ported from hot-step-9000: 3-panel resizable layout with
// Sidebar | CreatePanel | SongList | RightSidebar | Player.
//
// Playback is managed by playbackStore — App.tsx just renders WaveSurfer
// DOM hosts and wires their callbacks to the store.

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './context/AuthContext';
import { GlobalParamsProvider, useGlobalParams } from './context/GlobalParamsContext';
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
import { TerminalPanel } from './components/terminal/TerminalPanel';
import { LyricStudioV2 } from './components/lyric-studio/LyricStudioV2';
import { CoverStudio } from './components/cover-studio/CoverStudio';
import { GlobalParamBar } from './components/global-bar/GlobalParamBar';
import { PlaylistSidebar } from './components/playlist/PlaylistSidebar';
import {
  usePlayback,
  registerPlayers,
  getActiveMediaElement,
  handleOriginalReady,
  handleAltReady,
  handleFinish as pbHandleFinish,
  setCurrentTime as pbSetCurrentTime,
  setIsPlaying as pbSetIsPlaying,
  togglePlay as pbTogglePlay,
  seek as pbSeek,
  next as pbNext,
  previous as pbPrevious,
  setVolume as pbSetVolume,
  setPlaybackRate as pbSetPlaybackRate,
  setShuffle as pbSetShuffle,
  cycleRepeat as pbCycleRepeat,
  setSpectrumEnabled as pbSetSpectrumEnabled,
  toggleMastered as pbToggleMastered,
  songToTrack,
  playFromList,
} from './stores/playbackStore';
import type { Song, GenerationParams } from './types';
import { usePlaylist, addToPlaylist, type PlaylistItem } from './components/lyric-studio/playlistStore';

/** Derive top-level view from the browser URL */
function viewFromUrl(path = window.location.pathname): string {
  if (path.startsWith('/lyric-studio')) return 'lyric-studio';
  if (path.startsWith('/cover-studio')) return 'cover-studio';
  if (path.startsWith('/library')) return 'library';
  if (path.startsWith('/settings')) return 'settings';
  return 'create';
}

/** Map view names to URL paths */
function urlForView(view: string): string {
  if (view === 'lyric-studio') {
    // Restore the last deep URL (artist/album/tab) if we have one
    try {
      const saved = localStorage.getItem('hs-lastLyricStudioUrl');
      if (saved) return saved;
    } catch { /* ignore */ }
    return '/lyric-studio';
  }
  if (view === 'cover-studio') return '/cover-studio';
  if (view === 'library') return '/library';
  if (view === 'settings') return '/settings';
  return '/';
}

/** Inner app content — must be rendered inside GlobalParamsProvider */
const AppContent: React.FC = () => {
  const { token, isLoading } = useAuth();
  const [activeView, setActiveView] = useState(() => viewFromUrl());
  const [songs, setSongs] = useState<Song[]>([]);
  const [selectedSong, setSelectedSong] = useState<Song | null>(null);

  // Resizable panel widths (persisted)
  const [createPanelWidth, setCreatePanelWidth] = usePersistedState('ace-createPanelWidth', 490);
  const [rightSidebarWidth, setRightSidebarWidth] = usePersistedState('ace-rightSidebarWidth', 360);
  const [showRightSidebar, setShowRightSidebar] = useState(true);

  // Terminal panel state (persisted)
  const [showTerminal, setShowTerminal] = usePersistedState('ace-showTerminal', false);
  const [terminalWidth, setTerminalWidth] = usePersistedState('ace-terminalWidth', 450);

  // Settings state (persisted)
  const [settings, setSettings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

  // Playlist sidebar (persisted)
  const [showPlaylist, setShowPlaylist] = usePersistedState('ace-showPlaylist', false);
  const [playlistWidth, setPlaylistWidth] = usePersistedState('ace-playlistWidth', 300);
  const playlistData = usePlaylist();

  // ── Playback (from unified store) ──
  const pb = usePlayback();
  const currentSong = pb.currentTrack as (Song | null);  // PlaybackTrack is Song-compatible for rendering
  const wavesurferRef = useRef<WaveformPlayerHandle>(null);
  const wavesurferAltRef = useRef<WaveformPlayerHandle>(null);

  // Register WaveSurfer ref objects with playback store.
  // We pass the refs themselves (not .current) so the store always gets
  // the latest handle even if useImperativeHandle hasn't committed yet.
  useEffect(() => {
    registerPlayers(wavesurferRef, wavesurferAltRef);
  }, []);

  // Track spectrum analyzer media element — updates on mastered toggle
  const [spectrumMediaEl, setSpectrumMediaEl] = useState<HTMLMediaElement | null>(null);
  useEffect(() => {
    const el = getActiveMediaElement();
    if (el) setSpectrumMediaEl(el);
  }, [pb.playMastered, pb.currentTrack]);

  // Toast state
  const [toast, setToast] = useState<{ message: string; type: ToastType; isVisible: boolean }>({
    message: '', type: 'success', isVisible: false,
  });

  // Confirm dialog state
  const [confirmDialog, setConfirmDialog] = useState<{
    title: string; message: string; confirmLabel?: string; danger?: boolean; onConfirm: () => void;
  } | null>(null);

  const [isShutdown, setIsShutdown] = useState(false);

  // ── URL-based routing ──────────────────────────────────────
  const navigateTo = useCallback((view: string) => {
    // Save deep Lyric Studio URL before leaving so we can restore it
    if (activeView === 'lyric-studio' && view !== 'lyric-studio') {
      try { localStorage.setItem('hs-lastLyricStudioUrl', window.location.pathname); } catch { /* ignore */ }
    }
    setActiveView(view);
    const url = urlForView(view);
    if (window.location.pathname !== url) {
      window.history.pushState(null, '', url);
    }
  }, [activeView]);

  // Handle browser back/forward
  useEffect(() => {
    const handlePopState = () => setActiveView(viewFromUrl());
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Download modal state
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);

  const showToast = (message: string, type: ToastType = 'success') => {
    setToast({ message, type, isVisible: true });
  };

  // Song created callback — add to library
  const handleSongCreated = useCallback((song: Song) => {
    setSongs(prev => [song, ...prev.filter(s => s.id !== song.id)]);
  }, []);

  // Play a song from the library (used by SongList, RightSidebar)
  const playSong = useCallback((song: Song) => {
    setSelectedSong(song);
    setShowRightSidebar(true);
    playFromList(songToTrack(song), songs.map(songToTrack), 'library');
  }, [songs]);

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

  // Global params from context
  const globalParams = useGlobalParams();

  // Handle generation — merges content params (from CreatePanel) with global engine params (from context)
  const handleGenerate = useCallback((contentParams: Partial<GenerationParams>) => {
    if (!token) return;
    // Merge: global engine params + content-specific params + settings
    const engineParams = globalParams.getGlobalParams();
    const enrichedParams = {
      ...engineParams,
      ...contentParams,
      coResident: settings.coResident,
      cacheLmCodes: settings.cacheLmCodes,
    };
    genStore.submit(enrichedParams as GenerationParams, token).catch(err => {
      console.error('[App] Generation failed:', err);
    });
  }, [token, genStore, settings, globalParams]);

  // Handle delete
  const handleDelete = useCallback(async (song: Song) => {
    if (!token) return;
    await songApi.delete(song.id, token);
    setSongs(prev => prev.filter(s => s.id !== song.id));
    if (selectedSong?.id === song.id) setSelectedSong(null);
    showToast(`Deleted "${song.title}"`, 'success');
  }, [token, selectedSong]);

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
          if (selectedSong && idSet.has(selectedSong.id)) setSelectedSong(null);
          showToast(`Deleted ${ids.length} track${ids.length !== 1 ? 's' : ''}`, 'success');
        } catch (err: any) {
          showToast(`Delete failed: ${err.message}`, 'error');
        }
      },
    });
  }, [token, selectedSong]);

  // Reuse handler
  const [reuseData, setReuseData] = useState<{ song: Song; timestamp: number } | null>(null);
  const handleReuse = useCallback((song: Song) => {
    setReuseData({ song, timestamp: Date.now() });
    navigateTo('create');
  }, [navigateTo]);

  // Song update handler
  const handleSongUpdate = useCallback((updatedSong: Song) => {
    setSongs(prev => prev.map(s => s.id === updatedSong.id ? updatedSong : s));
    if (selectedSong?.id === updatedSong.id) setSelectedSong(updatedSong);
  }, [selectedSong]);

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

  // ── Player Logic ──
  // All playback state and logic lives in playbackStore.
  // App.tsx just wires WaveSurfer DOM callbacks to the store.
  // (No inline playSong/playNext/playPrevious/toggleMastered etc.)

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

    if (activeView === 'lyric-studio') {
      return (
        <div className="flex-1 overflow-hidden">
          <LyricStudioV2 />
        </div>
      );
    }

    if (activeView === 'cover-studio') {
      return (
        <div className="flex-1 overflow-hidden">
          <CoverStudio />
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
            onAddToPlaylist={(song) => {
              addToPlaylist({
                id: song.id,
                title: song.title || 'Untitled',
                audioUrl: song.audioUrl || song.audio_url || '',
                masteredAudioUrl: song.masteredAudioUrl || song.mastered_audio_url || '',
                artistName: song.artistName || '',
                coverUrl: song.coverUrl || song.cover_url || '',
                duration: typeof song.duration === 'number' ? song.duration : undefined,
                style: song.style || '',
              });
              showToast('Added to playlist', 'success');
            }}
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
            activeJobCount={genStore.activeJobCount}
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
            onAddToPlaylist={(song) => {
              addToPlaylist({
                id: song.id,
                title: song.title || 'Untitled',
                audioUrl: song.audioUrl || song.audio_url || '',
                masteredAudioUrl: song.masteredAudioUrl || song.mastered_audio_url || '',
                artistName: song.artistName || '',
                coverUrl: song.coverUrl || song.cover_url || '',
                duration: typeof song.duration === 'number' ? song.duration : undefined,
                style: song.style || '',
              });
              showToast('Added to playlist', 'success');
            }}
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
                onPlay={(song) => playFromList(songToTrack(song), songs.map(songToTrack), 'library')}
                isPlaying={pb.isPlaying && pb.currentTrack?.id === selectedSong?.id}
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
      {/* Global Parameter Bar — full width, above everything */}
      <GlobalParamBar />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          activeView={activeView}
          onViewChange={navigateTo}
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
          showTerminal={showTerminal}
          onToggleTerminal={() => setShowTerminal(prev => !prev)}
        />

        <main className="flex-1 flex overflow-hidden relative">
          {renderContent()}
        </main>

        {/* Playlist Sidebar — right of main, left of terminal */}
        {showPlaylist && (
          <>
            <div
              className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startW = playlistWidth;
                const onMove = (ev: MouseEvent) => {
                  const newW = Math.min(600, Math.max(220, startW + startX - ev.clientX));
                  setPlaylistWidth(newW);
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
              className="flex-shrink-0 h-full border-l border-white/5"
              style={{ width: playlistWidth }}
            >
              <PlaylistSidebar onClose={() => setShowPlaylist(false)} />
            </div>
          </>
        )}

        {/* Terminal Panel — far right, resizable */}
        {showTerminal && (
          <>
            <div
              className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-emerald-500/20 active:bg-emerald-500/30 transition-colors"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startW = terminalWidth;
                const onMove = (ev: MouseEvent) => {
                  const newW = Math.min(900, Math.max(300, startW + startX - ev.clientX));
                  setTerminalWidth(newW);
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
              <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-emerald-400 transition-colors" />
            </div>
            <div
              className="flex-shrink-0 h-full border-l border-white/5"
              style={{ width: terminalWidth }}
            >
              <TerminalPanel onClose={() => setShowTerminal(false)} />
            </div>
          </>
        )}
      </div>

      {/* ── Bottom Player Area: Markers → Waveform → Transport ── */}
      <div className="flex-shrink-0 bg-zinc-950 border-t border-white/5">
        {/* Collapsible visualisation area — animates up when playing, down when paused/stopped.
            Uses CSS Grid 0fr→1fr trick so the transition tracks actual content height perfectly,
            unlike max-height which over-shoots and makes the expand feel instant. */}
        <div
          style={{
            display: 'grid',
            gridTemplateRows: pb.isPlaying ? '1fr' : '0fr',
            opacity: pb.isPlaying ? 1 : 0,
            transition: 'grid-template-rows 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease',
          }}
        >
          <div style={{ overflow: 'hidden', minHeight: 0 }}>
          <SectionMarkers audioUrl={pb.currentAudioUrl ?? undefined} duration={pb.duration} />
          <SpectrumAnalyzer
            mediaElement={spectrumMediaEl}
            visible={pb.spectrumEnabled && pb.isPlaying}
            isPlaying={pb.isPlaying}
          />
          {/* Dual waveform: original + mastered stacked, opacity-switched */}
          <div className="relative" style={{ height: 56 }}>
            <div style={{
              position: 'absolute', inset: 0,
              opacity: pb.playMastered ? 0 : 1,
              pointerEvents: pb.playMastered ? 'none' : 'auto',
              transition: 'opacity 0.15s ease',
            }}>
              <WaveformPlayer
                ref={wavesurferRef}
                volume={pb.playMastered ? 0 : pb.volume}
                playbackRate={pb.playbackRate}
                onTimeUpdate={pbSetCurrentTime}
                onDurationChange={() => {}}
                onPlayChange={pbSetIsPlaying}
                onFinish={pbHandleFinish}
                onReady={(dur) => {
                  handleOriginalReady(dur);
                  if (!pb.playMastered) {
                    setSpectrumMediaEl(wavesurferRef.current?.getMediaElement() ?? null);
                  }
                }}
              />
            </div>
            <div style={{
              position: 'absolute', inset: 0,
              opacity: pb.playMastered ? 1 : 0,
              pointerEvents: pb.playMastered ? 'auto' : 'none',
              transition: 'opacity 0.15s ease',
            }}>
              <WaveformPlayer
                ref={wavesurferAltRef}
                volume={pb.playMastered ? pb.volume : 0}
                playbackRate={pb.playbackRate}
                onTimeUpdate={pbSetCurrentTime}
                onDurationChange={() => {}}
                onPlayChange={pbSetIsPlaying}
                onFinish={pbHandleFinish}
                onReady={(dur) => {
                  handleAltReady(dur);
                  if (pb.playMastered) {
                    setSpectrumMediaEl(wavesurferAltRef.current?.getMediaElement() ?? null);
                  }
                }}
              />
            </div>
          </div>
          </div>
          {/* /inner overflow wrapper */}
        </div>
        <Player
          currentSong={currentSong}
          isPlaying={pb.isPlaying}
          onTogglePlay={pbTogglePlay}
          currentTime={pb.currentTime}
          duration={pb.duration}
          onSeek={pbSeek}
          onNext={pbNext}
          onPrevious={pbPrevious}
          volume={pb.volume}
          onVolumeChange={pbSetVolume}
          playbackRate={pb.playbackRate}
          onPlaybackRateChange={pbSetPlaybackRate}
          audioRef={wavesurferRef as any}
          isShuffle={pb.shuffle}
          onToggleShuffle={() => pbSetShuffle(!pb.shuffle)}
          repeatMode={pb.repeat}
          onToggleRepeat={pbCycleRepeat}
          onReusePrompt={() => currentSong && handleReuse(currentSong as Song)}
          onDelete={() => currentSong && handleDelete(currentSong as Song)}
          onDownload={() => currentSong && setDownloadSong(currentSong as Song)}
          playMastered={pb.playMastered}
          onToggleMastered={pbToggleMastered}
          spectrumEnabled={pb.spectrumEnabled}
          onToggleSpectrum={() => pbSetSpectrumEnabled(!pb.spectrumEnabled)}
          showPlaylist={showPlaylist}
          playlistCount={playlistData.items.length}
          onTogglePlaylist={() => setShowPlaylist(prev => !prev)}
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

/** Root App — wraps content in GlobalParamsProvider */
const App: React.FC = () => (
  <GlobalParamsProvider>
    <AppContent />
  </GlobalParamsProvider>
);

export default App;
