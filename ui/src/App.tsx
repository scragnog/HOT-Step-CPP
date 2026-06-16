// App.tsx — Root application component
//
// Ported from hot-step-9000: 3-panel resizable layout with
// Sidebar | CreatePanel | SongList | RightSidebar | Player.
//
// Playback is managed by playbackStore — App.tsx just renders WaveSurfer
// DOM hosts and wires their callbacks to the store.

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from './context/AuthContext';
import { GlobalParamsProvider, useGlobalParams } from './context/GlobalParamsContext';
import { usePersistedState } from './hooks/usePersistedState';
import { useTheme } from './hooks/useTheme';
import { songApi } from './services/api';
import { Sidebar } from './components/sidebar/Sidebar';
import { CreatePanel } from './components/create/CreatePanel';
import { SongList } from './components/library/SongList';
import { enqueueSimpleGen, useResumeQueue, useAudioGenQueueSelector } from './stores/audioGenQueueStore';
import { ActivitySidebar } from './components/shared/ActivitySidebar';
import { Player } from './components/player/Player';
import { WaveformPlayer, type WaveformPlayerHandle } from './components/player/WaveformPlayer';
import { SectionMarkers } from './components/player/SectionMarkers';
import { LyricsBar } from './components/player/LyricsBar';
import { TrimControls } from './components/player/TrimControls';
import { SpectrumAnalyzer } from './components/player/SpectrumAnalyzer';
import { RightSidebar } from './components/details/RightSidebar';
import { Toast, type ToastType } from './components/shared/Toast';
import { ConfirmDialog } from './components/shared/ConfirmDialog';
import { downloadTrack } from './utils/downloadTrack';
import { SettingsPanel, type AppSettings, DEFAULT_SETTINGS } from './components/settings/SettingsPanel';
import { TerminalPanel } from './components/terminal/TerminalPanel';
import { AssistantPanel } from './components/assistant/AssistantPanel';
import { LyricStudioV2 } from './components/lyric-studio/LyricStudioV2';
import { CoverStudio } from './components/cover-studio/CoverStudio';
import { StemStudio } from './components/stem-studio/StemStudio';
import { StemBuilder } from './components/stem-builder/StemBuilder';
import { RepaintStudio } from './components/repaint-studio/RepaintStudio';
import { GlobalParamBar } from './components/global-bar/GlobalParamBar';
import { InstaGenPanel } from './components/insta-gen/InstaGenPanel';
import { PlaylistSidebar } from './components/playlist/PlaylistSidebar';
import {
  usePlaybackSelector,
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
  setTrimMode as pbSetTrimMode,
  handleTrimClick as pbHandleTrimClick,
  reloadCurrentTrack as pbReloadCurrentTrack,
  toggleAB as pbToggleAB,
  exitABMode as pbExitABMode,
  stop as pbStop,
  songToTrack,
  playFromList,
} from './stores/playbackStore';
import type { Song, GenerationParams } from './types';
import { usePlaylist, addToPlaylist } from './components/lyric-studio/playlistStore';
import { DisguiseModeProvider } from './hooks/useDisguiseMode';
import { ABCompareModal } from './components/shared/ABCompareModal';
import { useABCompareSelector, playAB, openModal as openABModal, clear as clearAB } from './stores/abCompareStore';
import { useDiscoMode, toggleDiscoMode, setDiscoPlaying, setDiscoDataUrl, syncStems, updateMainTime } from './stores/discoStore';
import { DiscoPulseWrapper } from './components/shared/DiscoPulseWrapper';

import { HiHatParticles } from './components/shared/HiHatParticles';

/** Derive top-level view from the browser URL */
function viewFromUrl(path = window.location.pathname): string {
  if (path.startsWith('/insta-gen')) return 'insta-gen';
  if (path.startsWith('/lyric-studio')) return 'lyric-studio';
  if (path.startsWith('/cover-studio')) return 'cover-studio';
  if (path.startsWith('/stem-studio')) return 'stem-studio';
  if (path.startsWith('/stem-builder')) return 'stem-builder';
  if (path.startsWith('/repaint')) return 'repaint';
  if (path.startsWith('/library')) return 'library';
  if (path.startsWith('/settings')) return 'settings';
  return 'create';
}

/** Map view names to URL paths */
function urlForView(view: string): string {
  if (view === 'insta-gen') return '/insta-gen';
  if (view === 'lyric-studio') {
    // Restore the last deep URL (artist/album/tab) if we have one
    try {
      const saved = localStorage.getItem('hs-lastLyricStudioUrl');
      if (saved) return saved;
    } catch { /* ignore */ }
    return '/lyric-studio';
  }
  if (view === 'cover-studio') return '/cover-studio';
  if (view === 'stem-studio') return '/stem-studio';
  if (view === 'stem-builder') return '/stem-builder';
  if (view === 'repaint') return '/repaint';
  if (view === 'library') return '/library';
  if (view === 'settings') return '/settings';
  return '/';
}

/** Restarting overlay — polls /api/health and reloads when the server is back */
const RestartingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [status, setStatus] = useState(t('app.restarting.stopping'));
  const [dots, setDots] = useState('');

  // Animate dots
  useEffect(() => {
    const iv = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(iv);
  }, []);

  // Poll for server health
  useEffect(() => {
    let cancelled = false;
    let attempt = 0;

    // Wait a moment before polling (give the server time to die)
    const startDelay = setTimeout(() => {
      setStatus(t('app.restarting.waiting'));
      const poll = setInterval(async () => {
        attempt++;
        try {
          const res = await fetch('/api/health', { signal: AbortSignal.timeout(2000) });
          if (res.ok && !cancelled) {
            clearInterval(poll);
            setStatus(t('app.restarting.reconnected'));
            setTimeout(() => window.location.reload(), 500);
          }
        } catch {
          // Still down — keep polling
          if (attempt > 5 && !cancelled) {
            setStatus(t('app.restarting.waiting'));
          }
        }
      }, 2000);

      // Cleanup
      return () => { cancelled = true; clearInterval(poll); };
    }, 3000);

    return () => { cancelled = true; clearTimeout(startDelay); };
  }, []);

  return (
    <div className="fixed inset-0 z-[200] flex flex-col items-center justify-center bg-white dark:bg-black text-zinc-900 dark:text-white">
      {/* Animated spinner */}
      <div className="relative mb-8">
        <div className="w-16 h-16 rounded-full border-4 border-amber-500/20" />
        <div className="absolute inset-0 w-16 h-16 rounded-full border-4 border-transparent border-t-amber-400 animate-spin" />
      </div>
      <h1 className="text-2xl font-bold mb-2">{t('app.restarting.title')}</h1>
      <p className="text-zinc-600 dark:text-zinc-400 text-lg">{status}{dots}</p>
      <p className="text-zinc-600 text-sm mt-4">{t('app.restarting.message')}</p>
    </div>
  );
};

/** Compact A/B comparison bar — shows above the player when both tracks are pinned */
const ABMiniBar: React.FC = () => {
  const trackA = useABCompareSelector(s => s.trackA);
  const trackB = useABCompareSelector(s => s.trackB);
  if (!trackA || !trackB) return null;

  return (
    <div className="flex items-center gap-2 px-4 py-1.5 bg-gradient-to-r from-blue-500/5 via-transparent to-orange-500/5 border-t border-white/5">
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold bg-blue-500/10 text-blue-400 border border-blue-500/20">
        A <span className="font-normal truncate max-w-[80px]">{trackA.title}</span>
      </span>
      <span className="text-[9px] text-zinc-600">vs</span>
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold bg-orange-500/10 text-orange-400 border border-orange-500/20">
        B <span className="font-normal truncate max-w-[80px]">{trackB.title}</span>
      </span>
      <div className="flex-1" />
      <button
        onClick={playAB}
        className="px-2 py-1 rounded-md text-[9px] font-semibold bg-pink-500/10 text-pink-400 border border-pink-500/20 hover:bg-pink-500/20 transition-colors"
      >
        ▶ Play A/B
      </button>
      <button
        onClick={openABModal}
        className="px-2 py-1 rounded-md text-[9px] font-semibold bg-violet-500/10 text-violet-400 border border-violet-500/20 hover:bg-violet-500/20 transition-colors"
      >
        ↔ Compare
      </button>
      <button
        onClick={clearAB}
        className="p-1 rounded-md text-zinc-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
        title="Clear A/B"
      >
        ✕
      </button>
    </div>
  );
};

/** Extract the source tag from a song's generation params */
function getSongSource(song: Song): string {
  return (song.generationParams as any)?.source || (song.generation_params as any)?.source || 'create';
}

/** Inner app content — must be rendered inside GlobalParamsProvider */
const AppContent: React.FC = () => {
  const { t } = useTranslation();
  const { token, isLoading } = useAuth();
  const [activeView, setActiveView] = useState(() => viewFromUrl());
  const [songs, setSongs] = useState<Song[]>([]);
  const [selectedSong, setSelectedSong] = useState<Song | null>(null);

  // Resizable panel widths (persisted)
  const [createPanelWidth, setCreatePanelWidth] = usePersistedState('ace-createPanelWidth', 490);
  const [rightSidebarWidth, setRightSidebarWidth] = usePersistedState('ace-rightSidebarWidth', 360);
  const [activitySidebarWidth, setActivitySidebarWidth] = usePersistedState('hs-activitySidebarWidth', 320);
  const [showRightSidebar, setShowRightSidebar] = useState(true);

  // Terminal panel state (persisted)
  const [showTerminal, setShowTerminal] = usePersistedState('ace-showTerminal', false);
  const [terminalWidth, setTerminalWidth] = usePersistedState('ace-terminalWidth', 450);

  // Assistant panel state (persisted)
  const [showAssistant, setShowAssistant] = usePersistedState('hs-showAssistant', false);
  const [assistantWidth, setAssistantWidth] = usePersistedState('hs-assistantWidth', 420);

  // Settings state (persisted)
  const [settings, setSettings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

  // Theme state
  const { theme, toggleTheme } = useTheme();

  // Playlist sidebar (persisted)
  const [showPlaylist, setShowPlaylist] = usePersistedState('ace-showPlaylist', false);
  const [playlistWidth, setPlaylistWidth] = usePersistedState('ace-playlistWidth', 300);
  const playlistData = usePlaylist();

  // ── Playback (from unified store — each selector only re-renders when its value changes) ──
  const currentTrack = usePlaybackSelector(s => s.currentTrack);
  const currentSong = currentTrack as (Song | null);  // PlaybackTrack is Song-compatible for rendering
  const isPlaying = usePlaybackSelector(s => s.isPlaying);
  const playerActive = usePlaybackSelector(s => s.playerActive);
  const currentTime = usePlaybackSelector(s => s.currentTime);
  const duration = usePlaybackSelector(s => s.duration);
  const volume = usePlaybackSelector(s => s.volume);
  const playbackRate = usePlaybackSelector(s => s.playbackRate);
  const playMastered = usePlaybackSelector(s => s.playMastered);
  const spectrumEnabled = usePlaybackSelector(s => s.spectrumEnabled);
  const shuffle = usePlaybackSelector(s => s.shuffle);
  const repeat = usePlaybackSelector(s => s.repeat);
  const trimMode = usePlaybackSelector(s => s.trimMode);
  const trimInPoint = usePlaybackSelector(s => s.trimInPoint);
  const trimOutPoint = usePlaybackSelector(s => s.trimOutPoint);
  const trimClickCount = usePlaybackSelector(s => s.trimClickCount);
  const abMode = usePlaybackSelector(s => s.abMode);
  const abActiveLabel = usePlaybackSelector(s => s.playMastered && s.abMode ? 'B' as const : 'A' as const);
  const wavesurferRef = useRef<WaveformPlayerHandle>(null);
  const wavesurferAltRef = useRef<WaveformPlayerHandle>(null);
  const discoMode = useDiscoMode();

  // Disco hue assignments per panel (0-360)
  const DISCO = {
    sidebar:      120,  // green
    createPanel:  330,  // hot pink
    songGrid:     195,  // deep sky blue
    activity:     270,  // purple
    playlist:     20,   // orange
    assistant:    175,  // cyan
    terminal:     135,  // green
    player:       45,   // gold
    rightSidebar: 300,  // magenta
  };

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
  }, [playMastered, currentTrack, isPlaying]);

  // Sync disco beat detection with playback state
  useEffect(() => {
    setDiscoPlaying(isPlaying);
    syncStems(isPlaying ? 'play' : 'pause', currentTime);
  }, [isPlaying]);

  // Feed current time to disco store every frame — this is the sync backbone.
  // Pre-analyzed stems look up energy at this time, so it's always in sync
  // with the main player regardless of seek, pause, rate changes.
  useEffect(() => {
    updateMainTime(currentTime);
  }, [currentTime]);

  // Load disco data JSON when track changes — trigger on-demand extraction if missing
  useEffect(() => {
    const discoUrl = currentTrack?.discoDataUrl || '';
    setDiscoDataUrl(discoUrl);

    // If disco extraction is on and this track has no disco data, trigger extraction
    if (settings.discoKickExtract && currentTrack?.id && !discoUrl) {
      console.log(`[Disco] Track ${currentTrack.id} has no disco data — triggering extraction`);
      let cancelled = false;
      (async () => {
        try {
          const res = await fetch(`/api/songs/${currentTrack.id}/extract-kick`, { method: 'POST' });
          const data = await res.json();
          if (cancelled) return;
          if (data.status === 'exists' && data.discoDataUrl) {
            console.log(`[Disco] Disco data already exists for ${currentTrack.id}`);
            setDiscoDataUrl(data.discoDataUrl);
            return;
          }
          if (data.status !== 'started') return;
          console.log(`[Disco] Stem extraction started for ${currentTrack.id}`);

          // Poll until disco data URL appears on the song
          for (let i = 0; i < 360 && !cancelled; i++) {
            await new Promise(r => setTimeout(r, 5000));
            if (cancelled) return;
            try {
              const songRes = await fetch(`/api/songs/${currentTrack.id}`);
              const songData = await songRes.json();
              const s = songData.song || songData;
              const ddUrl = s.disco_data_url || '';
              if (ddUrl) {
                console.log(`[Disco] Disco data ready for ${currentTrack.id}: ${ddUrl}`);
                setDiscoDataUrl(ddUrl);
                return;
              }
            } catch { /* keep polling */ }
          }
        } catch (err) {
          console.warn('[Disco] On-demand extraction failed:', err);
        }
      })();
      return () => { cancelled = true; };
    }
  }, [currentTrack?.id, currentTrack?.discoDataUrl, settings.discoKickExtract]);

  // ── Trim mode waveform click handler ──
  const handleWaveformClick = useCallback((timeSec: number) => {
    if (trimMode) {
      pbHandleTrimClick(timeSec);
    }
  }, [trimMode]);

  // Wrap seek to also sync kick stem
  const handleSeek = useCallback((time: number) => {
    pbSeek(time);
    syncStems('seek', time);
  }, []);

  // Toast state
  const [toast, setToast] = useState<{ message: string; type: ToastType; isVisible: boolean }>({
    message: '', type: 'success', isVisible: false,
  });

  // Confirm dialog state
  const [confirmDialog, setConfirmDialog] = useState<{
    title: string; message: string; confirmLabel?: string; danger?: boolean; onConfirm: () => void;
  } | null>(null);

  const [isShutdown, setIsShutdown] = useState(false);
  const [isRestarting, setIsRestarting] = useState(false);

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

  // Direct download helper (reads settings from localStorage)
  const handleDownload = useCallback((song: Song) => downloadTrack(song), []);

  const showToast = (message: string, type: ToastType = 'success') => {
    setToast({ message, type, isVisible: true });
  };

  // Disco mode toggle — prompt to enable stem extraction if needed
  const handleToggleDisco = useCallback(() => {
    if (discoMode) {
      // Turning off — always allowed
      toggleDiscoMode();
      return;
    }
    // Turning on — check if stem extraction is enabled
    if (!settings.discoKickExtract) {
      setConfirmDialog({
        title: '🪩 Enable Disco Mode?',
        message: 'Disco mode uses separated drum stems (kick, snare, hi-hat) for beat-reactive visuals. This requires extracting drum stems after each generation (~60-90s extra per song). Enable stem extraction and activate disco mode?',
        confirmLabel: 'Enable & Activate',
        onConfirm: () => {
          setSettings(prev => ({ ...prev, discoKickExtract: true }));
          toggleDiscoMode();
          showToast('🪩 Disco mode enabled! Drum stems will be extracted for new songs.', 'success');
        },
      });
    } else {
      toggleDiscoMode();
    }
  }, [discoMode, settings.discoKickExtract]);

  // Song created callback — add to library + trigger sidebar refresh
  const [songCreatedCount, setSongCreatedCount] = useState(0);
  const handleSongCreated = useCallback((song: Song) => {
    setSongs(prev => [song, ...prev.filter(s => s.id !== song.id)]);
    setSongCreatedCount(c => c + 1);

    // Auto-extract drum stems + disco data for disco mode if setting enabled
    if (settings.discoKickExtract && song.id && !song.disco_data_url) {
      console.log(`[Disco] Auto-extracting drum stems for song ${song.id}`);
      fetch(`/api/songs/${song.id}/extract-kick`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          if (data.status === 'started') {
            console.log(`[Disco] Stem extraction + analysis started for ${song.id}`);
          }
        })
        .catch(err => console.warn('[Disco] Stem extraction failed:', err));
    }
  }, [settings.discoKickExtract]);

  // Play a song from the library (used by SongList, RightSidebar)
  const playSong = useCallback((song: Song) => {
    setSelectedSong(song);
    setShowRightSidebar(true);
    playFromList(songToTrack(song), songs.map(songToTrack), 'library');
  }, [songs]);

  // Shared audio generation queue — resume on mount, subscribe only to active count
  useResumeQueue(token || undefined);
  const activeJobCount = useAudioGenQueueSelector(s =>
    s.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length
  );

  // Track the active streaming job ID — for SSE connection in CreatePanel
  const streamJobId = useAudioGenQueueSelector(s => {
    const active = s.items.find(i =>
      i.status === 'generating' && i.jobId && (i.globalParams as any)?.streamMode === true
    );
    return active?.jobId || null;
  });

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
      source: 'create',
      coResident: settings.coResident,
      cacheLmCodes: settings.cacheLmCodes,
      parallelWhisper: settings.parallelWhisper,
      parallelQualityEval: settings.parallelQualityEval,
      parallelCoverArt: settings.parallelCoverArt,
    };

    enqueueSimpleGen(
      enrichedParams,
      token,
      handleSongCreated,
    );
  }, [token, settings, globalParams, handleSongCreated]);

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

  // Listen for cover art completion events (fired by SongList context menu)
  useEffect(() => {
    const handler = (e: Event) => {
      const { songId, coverUrl } = (e as CustomEvent).detail;
      if (!songId || !coverUrl) return;
      setSongs(prev => prev.map(s =>
        s.id === songId ? { ...s, coverUrl, cover_url: coverUrl } : s
      ));
      // Also update selected song if it's the one that got cover art
      setSelectedSong(prev =>
        prev?.id === songId ? { ...prev, coverUrl, cover_url: coverUrl } as Song : prev
      );
    };
    window.addEventListener('cover-art-updated', handler);
    return () => window.removeEventListener('cover-art-updated', handler);
  }, []);

  // Listen for song-created events (fired by audioGenQueueStore on completion)
  // This covers Lyric Studio, Cover Studio, and resumed queue items.
  useEffect(() => {
    const handler = (e: Event) => {
      const { song } = (e as CustomEvent).detail;
      if (!song) return;
      handleSongCreated(song);
    };
    window.addEventListener('song-created', handler);
    return () => window.removeEventListener('song-created', handler);
  }, [handleSongCreated]);

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
      <div className="flex h-screen items-center justify-center bg-white dark:bg-suno text-zinc-600 dark:text-zinc-400">
        <div className="text-center">
          <div className="text-4xl mb-4">⚡</div>
          <div className="text-lg font-medium">{t('app.loading')}</div>
        </div>
      </div>
    );
  }

  if (isShutdown) {
    return (
      <div className="fixed inset-0 z-[200] flex flex-col items-center justify-center bg-white dark:bg-black text-zinc-900 dark:text-white">
        <div className="text-6xl mb-6">👋</div>
        <h1 className="text-2xl font-bold mb-2">{t('app.shutdown.title')}</h1>
        <p className="text-zinc-600 dark:text-zinc-400">{t('app.shutdown.message')}</p>
      </div>
    );
  }

  if (isRestarting) {
    return <RestartingOverlay />;
  }
  // Main content renderer
  const renderContent = () => {
    if (activeView === 'settings') {
      return (
        <DiscoPulseWrapper hue={DISCO.activity} className="flex-1 overflow-y-auto">
          <SettingsPanel
            settings={settings}
            onSettingsChange={setSettings}
            onNukeComplete={() => { setSongs([]); setSelectedSong(null); }}
          />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'lyric-studio') {
      return (
        <DiscoPulseWrapper hue={DISCO.songGrid} className="flex-1 overflow-hidden">
          <LyricStudioV2 />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'cover-studio') {
      return (
        <DiscoPulseWrapper hue={DISCO.activity} className="flex-1 overflow-hidden">
          <CoverStudio />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'stem-studio') {
      return (
        <DiscoPulseWrapper hue={DISCO.assistant} className="flex-1 overflow-hidden">
          <StemStudio />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'stem-builder') {
      return (
        <DiscoPulseWrapper hue={DISCO.assistant} className="flex-1 overflow-hidden">
          <StemBuilder />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'repaint') {
      return (
        <DiscoPulseWrapper hue={DISCO.createPanel} className="flex-1 overflow-hidden">
          <RepaintStudio />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'insta-gen') {
      return (
        <div className="flex flex-1 overflow-hidden">
          {/* Insta-Gen Panel — resizable */}
          <DiscoPulseWrapper hue={DISCO.createPanel}
            className="flex-shrink-0 h-full border-r border-zinc-200 dark:border-white/5"
            style={{ width: createPanelWidth }}
          >
            <InstaGenPanel
              onSongCreated={handleSongCreated}
              activeJobCount={activeJobCount}
              onNavigate={navigateTo}
            />
          </DiscoPulseWrapper>

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

          {/* Song List — "Generations" (filtered to insta-gen only) */}
          <DiscoPulseWrapper hue={DISCO.songGrid} className="flex-1 min-w-0 overflow-y-auto">
            <SongList
              songs={songs.filter(s => getSongSource(s) === 'insta-gen')}
              currentSongId={currentSong?.id}
              onPlay={playSong}
              onDelete={handleDelete}
              onBulkDelete={handleBulkDelete}
              onSelect={(s) => { setSelectedSong(s); setShowRightSidebar(true); }}
              onReuse={handleReuse}
              onDownload={handleDownload}
              onRename={handleRename}
              showFilters={false}
              viewMode="grid"
              title={t('app.generations')}
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
                showToast(t('app.addedToPlaylist'), 'success');
              }}
            />
          </DiscoPulseWrapper>

          {/* Resize handle — Activity Sidebar */}
          <div
            className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
            onMouseDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startW = activitySidebarWidth;
              const onMove = (ev: MouseEvent) => {
                const newW = Math.min(700, Math.max(240, startW + startX - ev.clientX));
                setActivitySidebarWidth(newW);
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
          {/* Activity Sidebar — Recent Songs + Queue */}
          <DiscoPulseWrapper hue={DISCO.activity} className="h-full flex-shrink-0 border-l border-zinc-200 dark:border-white/5 overflow-hidden" style={{ width: activitySidebarWidth }}>
            <ActivitySidebar
              source="insta-gen"
              showToast={showToast}
              refreshKey={songCreatedCount}
              compact={activitySidebarWidth < 380}
            />
          </DiscoPulseWrapper>

          {/* Right Sidebar — Song detail (same as Create) */}
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
              <DiscoPulseWrapper hue={DISCO.rightSidebar}
                className="flex-shrink-0 h-full bg-zinc-50 dark:bg-suno z-10 border-l border-zinc-200 dark:border-white/5"
                style={{ width: rightSidebarWidth }}
              >
                <RightSidebar
                  song={selectedSong}
                  onClose={() => setShowRightSidebar(false)}
                  onReuse={handleReuse}
                  onDelete={handleDelete}
                  onPlay={(song) => playFromList(songToTrack(song), songs.map(songToTrack), 'library')}
                  isPlaying={isPlaying && currentTrack?.id === selectedSong?.id}
                  onDownload={handleDownload}
                  onRename={handleRename}
                />
              </DiscoPulseWrapper>
            </>
          )}
        </div>
      );
    }

    if (activeView === 'library') {
      return (
        <div className="flex flex-1 overflow-hidden">
          <DiscoPulseWrapper hue={DISCO.songGrid} className="flex-1 min-w-0 overflow-y-auto">
            <SongList
              songs={songs}
              currentSongId={currentSong?.id}
              onPlay={playSong}
              onDelete={handleDelete}
              onBulkDelete={handleBulkDelete}
              onSelect={(s) => { setSelectedSong(s); setShowRightSidebar(true); }}
              onReuse={handleReuse}
              onDownload={handleDownload}
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
                showToast(t('app.addedToPlaylist'), 'success');
              }}
            />
          </DiscoPulseWrapper>

          {/* Right Sidebar — Song detail */}
          {showRightSidebar && selectedSong && (
            <>
              {/* Resize handle */}
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
              <DiscoPulseWrapper hue={DISCO.rightSidebar}
                className="flex-shrink-0 h-full bg-zinc-50 dark:bg-suno z-10 border-l border-zinc-200 dark:border-white/5"
                style={{ width: rightSidebarWidth }}
              >
                <RightSidebar
                  song={selectedSong}
                  onClose={() => setShowRightSidebar(false)}
                  onReuse={handleReuse}
                  onDelete={handleDelete}
                  onPlay={(song) => playFromList(songToTrack(song), songs.map(songToTrack), 'library')}
                  isPlaying={isPlaying && currentTrack?.id === selectedSong?.id}
                  onDownload={handleDownload}
                  onRename={handleRename}
                />
              </DiscoPulseWrapper>
            </>
          )}
        </div>
      );
    }

    // Default: create view
    return (
      <div className="flex flex-1 overflow-hidden">
        {/* Create Panel — resizable */}
        <DiscoPulseWrapper hue={DISCO.createPanel} className="flex-shrink-0 h-full border-r border-zinc-200 dark:border-white/5" style={{ width: createPanelWidth }}>
          <CreatePanel
            onGenerate={handleGenerate}
            activeJobCount={activeJobCount}
            reuseData={reuseData}
            streamJobId={streamJobId}
          />
        </DiscoPulseWrapper>

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

        {/* Song List + Queue (filtered to create/custom-gen only) */}
        <DiscoPulseWrapper hue={DISCO.songGrid} className="flex-1 min-w-0 overflow-y-auto">
          <SongList
            songs={songs.filter(s => getSongSource(s) === 'create')}
            currentSongId={currentSong?.id}
            onPlay={playSong}
            onDelete={handleDelete}
            onBulkDelete={handleBulkDelete}
            onSelect={(s) => { setSelectedSong(s); setShowRightSidebar(true); }}
            onReuse={handleReuse}
            onDownload={handleDownload}
            onRename={handleRename}
            showFilters={false}
            viewMode="grid"
            title={t('app.generations')}
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
              showToast(t('app.addedToPlaylist'), 'success');
            }}
          />
        </DiscoPulseWrapper>

        {/* Resize handle — Activity Sidebar */}
        <div
          className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
          onMouseDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startW = activitySidebarWidth;
            const onMove = (ev: MouseEvent) => {
              const newW = Math.min(700, Math.max(240, startW + startX - ev.clientX));
              setActivitySidebarWidth(newW);
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
        {/* Activity Sidebar — Recent Songs + Queue */}
        <DiscoPulseWrapper hue={DISCO.activity} className="h-full flex-shrink-0 border-l border-zinc-200 dark:border-white/5 overflow-hidden" style={{ width: activitySidebarWidth }}>
          <ActivitySidebar
            source="create"
            showToast={showToast}
            refreshKey={songCreatedCount}
            compact={activitySidebarWidth < 380}
          />
        </DiscoPulseWrapper>

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
            <DiscoPulseWrapper hue={DISCO.rightSidebar}
              className="flex-shrink-0 h-full bg-zinc-50 dark:bg-suno z-10 border-l border-zinc-200 dark:border-white/5"
              style={{ width: rightSidebarWidth }}
            >
              <RightSidebar
                song={selectedSong}
                onClose={() => setShowRightSidebar(false)}
                onReuse={handleReuse}
                onDelete={handleDelete}
                onPlay={(song) => playFromList(songToTrack(song), songs.map(songToTrack), 'library')}
                isPlaying={isPlaying && currentTrack?.id === selectedSong?.id}
                onDownload={handleDownload}
                onRename={handleRename}
              />
            </DiscoPulseWrapper>
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
        <DiscoPulseWrapper hue={DISCO.sidebar}>
        <Sidebar
          activeView={activeView}
          onViewChange={navigateTo}
          onQuit={() => {
            setConfirmDialog({
              title: t('app.quitDialog.title'),
              message: t('app.quitDialog.message'),
              confirmLabel: t('app.quitDialog.confirm'),
              danger: true,
              onConfirm: async () => {
                setConfirmDialog(null);
                try { await fetch('/api/shutdown', { method: 'POST' }); } catch { /* shutting down */ }
                setIsShutdown(true);
              },
            });
          }}
          onRestart={() => {
            setConfirmDialog({
              title: t('app.restartDialog.title'),
              message: t('app.restartDialog.message'),
              confirmLabel: t('app.restartDialog.confirm'),
              danger: false,
              onConfirm: async () => {
                setConfirmDialog(null);
                setIsRestarting(true);
                try { await fetch('/api/shutdown/restart', { method: 'POST' }); } catch { /* restarting */ }
              },
            });
          }}
          theme={theme}
          onToggleTheme={toggleTheme}
          showTerminal={showTerminal}
          onToggleTerminal={() => setShowTerminal(prev => !prev)}
          showAssistant={showAssistant}
          onToggleAssistant={() => setShowAssistant(prev => !prev)}
        />
        </DiscoPulseWrapper>

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
            <DiscoPulseWrapper hue={DISCO.playlist}
              className="flex-shrink-0 h-full border-l border-zinc-200 dark:border-white/5"
              style={{ width: playlistWidth }}
            >
              <PlaylistSidebar onClose={() => setShowPlaylist(false)} />
            </DiscoPulseWrapper>
          </>
        )}

        {/* Assistant Panel — between playlist and terminal, resizable */}
        {showAssistant && (
          <>
            <div
              className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-violet-500/20 active:bg-violet-500/30 transition-colors"
              onMouseDown={(e) => {
                e.preventDefault();
                const startX = e.clientX;
                const startW = assistantWidth;
                const onMove = (ev: MouseEvent) => {
                  const newW = Math.min(700, Math.max(320, startW + startX - ev.clientX));
                  setAssistantWidth(newW);
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
              <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-violet-400 transition-colors" />
            </div>
            <DiscoPulseWrapper hue={DISCO.assistant}
              className="flex-shrink-0 h-full border-l border-zinc-200 dark:border-white/5"
              style={{ width: assistantWidth }}
            >
              <AssistantPanel onClose={() => setShowAssistant(false)} activeView={activeView} />
            </DiscoPulseWrapper>
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
            <DiscoPulseWrapper hue={DISCO.terminal}
              className="flex-shrink-0 h-full border-l border-zinc-200 dark:border-white/5"
              style={{ width: terminalWidth }}
            >
              <TerminalPanel onClose={() => setShowTerminal(false)} />
            </DiscoPulseWrapper>
          </>
        )}
      </div>

      {/* ── Bottom Player Area: Markers → Waveform → Transport ── */}
      <DiscoPulseWrapper hue={DISCO.player} className="flex-shrink-0 bg-white dark:bg-zinc-950 border-t border-zinc-200 dark:border-white/5" style={{ position: 'relative' }}>
        {/* Collapsible visualisation area — animates up when playing, down when paused/stopped.
            Uses CSS Grid 0fr→1fr trick so the transition tracks actual content height perfectly,
            unlike max-height which over-shoots and makes the expand feel instant. */}
        <div
          style={{
            display: 'grid',
            gridTemplateRows: (playerActive || trimMode) ? '1fr' : '0fr',
            opacity: (playerActive || trimMode) ? 1 : 0,
            transition: 'grid-template-rows 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease',
          }}
        >
          <div style={{ overflow: 'hidden', minHeight: 0 }}>
          <SectionMarkers audioUrl={currentTrack?.audioUrl ?? undefined} duration={duration} />
          <SpectrumAnalyzer
            mediaElement={spectrumMediaEl}
            visible={spectrumEnabled && playerActive}
            isPlaying={isPlaying}
          />
          {trimMode && (
            <TrimControls
              trimInPoint={trimInPoint}
              trimOutPoint={trimOutPoint}
              trimClickCount={trimClickCount}
              duration={duration}
              songId={currentTrack?.id ?? null}
              audioUrl={currentTrack?.audioUrl ?? null}
              wavesurferRef={wavesurferRef}
              wavesurferAltRef={wavesurferAltRef}
              onReload={pbReloadCurrentTrack}
              onCancel={() => pbSetTrimMode(false)}
            />
          )}
          {/* Dual waveform: original + mastered stacked, opacity-switched */}
          <div className="relative" style={{ height: 56 }}>
            <div style={{
              position: 'absolute', inset: 0,
              opacity: playMastered ? 0 : 1,
              pointerEvents: playMastered ? 'none' : 'auto',
              transition: 'opacity 0.15s ease',
            }}>
              <WaveformPlayer
                ref={wavesurferRef}
                volume={playMastered ? 0 : volume}
                playbackRate={playbackRate}
                onTimeUpdate={pbSetCurrentTime}
                onDurationChange={() => {}}
                onPlayChange={pbSetIsPlaying}
                onFinish={pbHandleFinish}
                onWaveformClick={handleWaveformClick}
                onReady={(dur) => {
                  handleOriginalReady(dur);
                  if (!playMastered) {
                    setSpectrumMediaEl(wavesurferRef.current?.getMediaElement() ?? null);
                  }
                }}
              />
            </div>
            <div style={{
              position: 'absolute', inset: 0,
              opacity: playMastered ? 1 : 0,
              pointerEvents: playMastered ? 'auto' : 'none',
              transition: 'opacity 0.15s ease',
            }}>
              <WaveformPlayer
                ref={wavesurferAltRef}
                volume={playMastered ? volume : 0}
                playbackRate={playbackRate}
                onTimeUpdate={pbSetCurrentTime}
                onDurationChange={() => {}}
                onPlayChange={pbSetIsPlaying}
                onFinish={pbHandleFinish}
                onWaveformClick={handleWaveformClick}
                onReady={(dur) => {
                  handleAltReady(dur);
                  if (playMastered) {
                    setSpectrumMediaEl(wavesurferAltRef.current?.getMediaElement() ?? null);
                  }
                }}
              />
            </div>
          </div>
          </div>
          {/* /inner overflow wrapper */}
        </div>
        <LyricsBar
          audioUrl={currentTrack?.audioUrl ?? undefined}
          currentTime={currentTime}
          isPlaying={isPlaying}
        />
        {/* Global A/B comparison mini-bar — visible from any view when both tracks are pinned */}
        <ABMiniBar />
        <Player
          currentSong={currentSong}
          isPlaying={isPlaying}
          onTogglePlay={pbTogglePlay}
          onStop={pbStop}
          currentTime={currentTime}
          duration={duration}
          onSeek={handleSeek}
          onNext={pbNext}
          onPrevious={pbPrevious}
          volume={volume}
          onVolumeChange={pbSetVolume}
          playbackRate={playbackRate}
          onPlaybackRateChange={pbSetPlaybackRate}
          audioRef={wavesurferRef as any}
          isShuffle={shuffle}
          onToggleShuffle={() => pbSetShuffle(!shuffle)}
          repeatMode={repeat}
          onToggleRepeat={pbCycleRepeat}
          onReusePrompt={() => currentSong && handleReuse(currentSong as Song)}
          onDelete={() => currentSong && handleDelete(currentSong as Song)}
          onDownload={() => currentSong && downloadTrack(currentSong as Song)}
          playMastered={playMastered}
          onToggleMastered={pbToggleMastered}
          spectrumEnabled={spectrumEnabled}
          onToggleSpectrum={() => pbSetSpectrumEnabled(!spectrumEnabled)}
          showPlaylist={showPlaylist}
          playlistCount={playlistData.items.length}
          onTogglePlaylist={() => setShowPlaylist(prev => !prev)}
          trimMode={trimMode}
          onToggleTrimMode={() => pbSetTrimMode(!trimMode)}
          abMode={abMode}
          abActiveLabel={abActiveLabel}
          onToggleAB={pbToggleAB}
          onExitABMode={pbExitABMode}
          discoMode={discoMode}
          onToggleDisco={handleToggleDisco}
        />
        <HiHatParticles />
      </DiscoPulseWrapper>

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

      <ABCompareModal />
    </div>
  );
};

/** Root App — wraps content in GlobalParamsProvider + DisguiseModeProvider */
const App: React.FC = () => (
  <GlobalParamsProvider>
    <DisguiseModeProvider>
      <AppContent />
    </DisguiseModeProvider>
  </GlobalParamsProvider>
);

export default App;
