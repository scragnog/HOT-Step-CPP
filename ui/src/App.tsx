// App.tsx — Root application component
//
// Ported from hot-step-9000: 3-panel resizable layout with
// Sidebar | CreatePanel | SongList | RightSidebar | Player.
//
// Playback is managed by playbackStore — App.tsx just renders WaveSurfer
// DOM hosts and wires their callbacks to the store.

import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from './context/AuthContext';
import { GlobalParamsProvider, useGlobalParams } from './context/GlobalParamsContext';
import { usePersistedState } from './hooks/usePersistedState';
import { useTheme } from './hooks/useTheme';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { songApi, shutdownApi } from './services/api';
import { apiClient } from './services/httpClient';
import { useJobProgressWatcher, requestNotifyPermissionFromGesture } from './hooks/useJobProgress';
import { bridgeEvents } from './queries/sseBridge';
import { SONG_VIEWS, invalidateSongs, patchLibrary, patchSong, removeSongs, songOptions, songsOptions } from './queries/shell-queue/songs';
import { waitForQuery } from './queries/wait';
import { Sidebar } from './components/sidebar/Sidebar';
import { CreatePanel } from './components/create/CreatePanel';
import { SongList } from './components/library/SongList';
import { StreamWaveform } from './components/player/StreamWaveform';
import { mm3StreamEnsureTakes, useMm3StreamAudio } from './stores/mm3StreamStore';
import { enqueueSimpleGen, useResumeQueue, useAudioGenQueueSelector, clearFinishedFromAudioQueue } from './stores/audioGenQueueStore';
import { useAudioQueueProjection } from './queries/shell-queue/projection';
import { NoModelsBanner } from './components/shared/NoModelsBanner';
import { RestoredQueueBanner } from './components/shared/RestoredQueueBanner';
import { AudioQueueMigrationBanner } from './components/shared/AudioQueueMigrationBanner';
import { ErrorBoundary } from './components/shared/ErrorBoundary';
import { PostProcessDock } from './components/shared/PostProcessActivity';
import { ActivitySidebar } from './components/shared/ActivitySidebar';
import { Player } from './components/player/Player';
import { Waveform } from './components/player/Waveform';
import { getElement as getPlayerElement, subscribeFrame } from './audio/engine';
import { SectionMarkers } from './components/player/SectionMarkers';
import { LyricsBar } from './components/player/LyricsBar';
import { TrimControls } from './components/player/TrimControls';
import { SpectrumAnalyzer } from './components/player/SpectrumAnalyzer';
import { StreamSpectrum } from './components/player/StreamSpectrum';
import { RightSidebar } from './components/details/RightSidebar';
import { MetadataEditorModal } from './components/details/MetadataEditorModal';
import { CoverArtPromptModal } from './components/library/CoverArtPromptModal';
import { ImportTracksModal } from './components/library/ImportTracksModal';
import { Toast, type ToastType } from './components/shared/Toast';
import { HotkeyHelpModal } from './components/shared/HotkeyHelpModal';
import { CommandPalette, OPEN_PALETTE_EVENT } from './components/shared/CommandPalette';
import { useGlobalHotkeys } from './hooks/useGlobalHotkeys';
import { ConfirmDialog } from './components/shared/ConfirmDialog';
import { downloadTrack, downloadTrackVersion } from './utils/downloadTrack';
import { SettingsPanel, type AppSettings, DEFAULT_SETTINGS } from './components/settings/SettingsPanel';
import { AssistantPanel } from './components/assistant/AssistantPanel';
import { LyricStudioV2 } from './components/lyric-studio/LyricStudioV2';
import { CoverStudio } from './components/cover-studio/CoverStudio';
import { StemStudio } from './components/stem-studio/StemStudio';
import { StemBuilder } from './components/stem-builder/StemBuilder';
import { RepaintStudio } from './components/repaint-studio/RepaintStudio';
import { SongBuilder } from './components/song-builder/SongBuilder';
import { StormPage } from './components/storm/StormPage';
import { MidiStudio } from './components/midi-studio/MidiStudio';
import { TrainingStudio } from './components/training-studio/TrainingStudio';
import { CompareStudio } from './components/compare-studio/CompareStudio';
import { setPendingCompareSongs } from './services/compareApi';
import { GlobalParamBar } from './components/global-bar/GlobalParamBar';
import { InstaGenPanel } from './components/insta-gen/InstaGenPanel';
import {
  usePlaybackSelector,
  togglePlay as pbTogglePlay,
  seek as pbSeek,
  next as pbNext,
  previous as pbPrevious,
  setVolume as pbSetVolume,
  setPlaybackRate as pbSetPlaybackRate,
  setPitch441 as pbSetPitch441,
  setShuffle as pbSetShuffle,
  cycleRepeat as pbCycleRepeat,
  setSpectrumEnabled as pbSetSpectrumEnabled,
  setPlaybackVariant as pbSetPlaybackVariant,
  setTrimMode as pbSetTrimMode,
  handleTrimClick as pbHandleTrimClick,
  reloadCurrentTrack as pbReloadCurrentTrack,
  toggleAB as pbToggleAB,
  exitABMode as pbExitABMode,
  stop as pbStop,
  songToTrack,
  playFromList,
  play as pbPlay,
} from './stores/playbackStore';
import type { Song, GenerationParams } from './types';
import type { CreateIntent } from '../../server/src/contracts/resolution';
import { addToPlaylist } from './components/lyric-studio/playlistStore';
import { DisguiseModeProvider } from './hooks/useDisguiseMode';
import { ABCompareModal } from './components/shared/ABCompareModal';
import { useABCompareSelector, playAB, openModal as openABModal, clear as clearAB } from './stores/abCompareStore';
import { useDiscoMode, toggleDiscoMode, setDiscoPlaying, setDiscoDataUrl, syncStems, updateMainTime } from './stores/discoStore';
import { DiscoPulseWrapper } from './components/shared/DiscoPulseWrapper';

import { HiHatParticles } from './components/shared/HiHatParticles';
import { subscribeEventSource } from './services/sharedEventSource';

// Inlined: the file could not be fetched from a server that has just died.
import serverOfflineImage from './assets/server-offline.webp?inline';
/** Derive top-level view from the browser URL */
function viewFromUrl(path = window.location.pathname): string {
  if (path.startsWith('/insta-gen')) return 'insta-gen';
  if (path.startsWith('/lyric-studio')) return 'lyric-studio';
  if (path.startsWith('/cover-studio')) return 'cover-studio';
  if (path.startsWith('/stem-studio')) return 'stem-studio';
  if (path.startsWith('/stem-builder')) return 'stem-builder';
  if (path.startsWith('/song-builder')) return 'song-builder';
  if (path.startsWith('/storm')) return 'storm';
  if (path.startsWith('/midi-studio')) return 'midi-studio';
  if (path.startsWith('/training-studio')) return 'training-studio';
  if (path.startsWith('/compare')) return 'compare-studio';
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
  if (view === 'song-builder') return '/song-builder';
  if (view === 'storm') return '/storm';
  if (view === 'midi-studio') return '/midi-studio';
  if (view === 'training-studio') {
    // Restore the last deep URL (dataset/phase) if we have one
    try {
      const saved = localStorage.getItem('hs-lastTrainingStudioUrl');
      if (saved) return saved;
    } catch { /* ignore */ }
    return '/training-studio';
  }
  if (view === 'compare-studio') return '/compare';
  if (view === 'repaint') return '/repaint';
  if (view === 'library') return '/library';
  if (view === 'settings') return '/settings';
  return '/';
}

/** Poll after either kind of disconnect, then reload once the server answers. */
function useServerHealthReload(startDelayMs: number, onWaiting?: () => void, onReconnected?: () => void): void {
  useEffect(() => {
    let cancelled = false;
    let checking = false;
    let poll: number | undefined;
    let reload: number | undefined;
    const start = window.setTimeout(() => {
      onWaiting?.();
      poll = window.setInterval(async () => {
        if (checking) return;
        checking = true;
        try {
          const res = await fetch('/api/health', { signal: AbortSignal.timeout(2000) });
          if (res.ok && !cancelled) {
            window.clearInterval(poll);
            onReconnected?.();
            reload = window.setTimeout(() => window.location.reload(), 500);
          }
        } catch { /* The server is still offline. */ }
        finally { checking = false; }
      }, 2000);
    }, startDelayMs);
    return () => {
      cancelled = true;
      window.clearTimeout(start);
      window.clearInterval(poll);
      window.clearTimeout(reload);
    };
  }, [startDelayMs, onWaiting, onReconnected]);
}

// Which recent-songs feed the activity column shows, per view. Views that are
// not listed — library, settings, training — show every source.
const ACTIVITY_SOURCE: Record<string, string | undefined> = {
  'create': 'create',
  'insta-gen': 'insta-gen',
  'lyric-studio': 'lyric-studio',
  'cover-studio': 'cover-studio',
  'repaint': 'repaint',
};

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

  const onWaiting = useCallback(() => setStatus(t('app.restarting.waiting')), [t]);
  const onReconnected = useCallback(() => setStatus(t('app.restarting.reconnected')), [t]);
  useServerHealthReload(3000, onWaiting, onReconnected);

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

const ServerOfflineOverlay: React.FC = () => {
  useServerHealthReload(0);
  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black">
      <img src={serverOfflineImage} alt="Server offline. Please restart or close tab" className="max-w-[90vw] max-h-[90vh] object-contain" />
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
/** A stable empty library, so memo and callback deps don't churn before load. */
const NO_SONGS: Song[] = [];

function getSongSource(song: Song): string {
  return (song.generationParams as any)?.source || (song.generation_params as any)?.source || 'create';
}

/** Inner app content — must be rendered inside GlobalParamsProvider */
// Tab title "(N) 43% · HOT-Step" + done notification while the tab is hidden.
// Own component so the queue subscription re-renders only this, not AppContent.
const JobProgressWatcher: React.FC = () => {
  const { t } = useTranslation();
  const doneBody = useCallback((name: string) => t('jobs.doneBody', { name }), [t]);
  useJobProgressWatcher(t('jobs.doneTitle'), doneBody);
  return null;
};

const AppContent: React.FC = () => {
  const { t } = useTranslation();
  // sessionEpoch re-renders this component when the auth-scoped keys move.
  const { token, sessionEpoch, isLoading } = useAuth();
  void sessionEpoch;
  const queryClient = useQueryClient();
  const [activeView, setActiveView] = useState(() => viewFromUrl());
  const songsQuery = useQuery({ ...songsOptions(), enabled: !!token });
  const songs = songsQuery.data?.songs ?? NO_SONGS;
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [selectedSong, setSelectedSong] = useState<Song | null>(null);

  // Resizable panel widths (persisted)
  const [createPanelWidth, setCreatePanelWidth] = usePersistedState('ace-createPanelWidth', 490);
  const [rightSidebarWidth, setRightSidebarWidth] = usePersistedState('ace-rightSidebarWidth', 360);
  const [activitySidebarWidth, setActivitySidebarWidth] = usePersistedState('hs-activitySidebarWidth', 420);
  const [showRightSidebar, setShowRightSidebar] = useState(true);

  // Assistant panel state (persisted)
  const [showAssistant, setShowAssistant] = usePersistedState('hs-showAssistant', false);
  const [assistantWidth, setAssistantWidth] = usePersistedState('hs-assistantWidth', 420);

  // Settings state (persisted)
  const [settings, setSettings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);

  // Theme state
  const { theme, toggleTheme } = useTheme();


  // ── Playback (from unified store — each selector only re-renders when its value changes) ──
  const currentTrack = usePlaybackSelector(s => s.currentTrack);
  /** The bar is driving a live MiniMax-Music3 render rather than a file. */
  const isStreamTrack = !!currentTrack?.streamJobId;
  const currentSong = currentTrack as (Song | null);  // PlaybackTrack is Song-compatible for rendering
  const isPlaying = usePlaybackSelector(s => s.isPlaying);
  // Read inside an effect that must NOT re-run when playback state changes —
  // the auto-play decision is made once, when the stream opens.
  const isPlayingRef = useRef(isPlaying);
  isPlayingRef.current = isPlaying;
  const playerActive = usePlaybackSelector(s => s.playerActive);
  const currentTime = usePlaybackSelector(s => s.currentTime);
  const duration = usePlaybackSelector(s => s.duration);
  const volume = usePlaybackSelector(s => s.volume);
  const playbackRate = usePlaybackSelector(s => s.playbackRate);
  const pitch441 = usePlaybackSelector(s => s.pitch441);
  const playMastered = usePlaybackSelector(s => s.playMastered);
  const playNoAdapter = usePlaybackSelector(s => s.playNoAdapter);
  const spectrumEnabled = usePlaybackSelector(s => s.spectrumEnabled);
  const shuffle = usePlaybackSelector(s => s.shuffle);
  const repeat = usePlaybackSelector(s => s.repeat);
  const trimMode = usePlaybackSelector(s => s.trimMode);
  const trimInPoint = usePlaybackSelector(s => s.trimInPoint);
  const trimOutPoint = usePlaybackSelector(s => s.trimOutPoint);
  const trimClickCount = usePlaybackSelector(s => s.trimClickCount);
  const abMode = usePlaybackSelector(s => s.abMode);
  const abActiveLabel = usePlaybackSelector(s => s.playMastered && s.abMode ? 'B' as const : 'A' as const);
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
    storm:        210,  // electric blue
    trainingStudio: 60,   // amber
  };

  // The analyser's source. One element for the life of the page now, so there
  // is nothing to re-point when the variant or the track changes.
  const spectrumMediaEl = getPlayerElement();

  // Sync disco beat detection with playback state
  useEffect(() => {
    setDiscoPlaying(isPlaying);
    syncStems(isPlaying ? 'play' : 'pause', currentTime);
  }, [isPlaying]);

  // Feed current time to the disco store every frame — this is the sync
  // backbone. Pre-analyzed stems look up energy at this time, so it stays in
  // step with the audio through seeks, pauses and rate changes.
  //
  // The engine's frame channel really is per-frame, and deliberately does not
  // go through React. The store's currentTime, which updates ten times a
  // second, covers the live MM3 render, where there is no engine to ask.
  useEffect(() => subscribeFrame((t) => updateMainTime(t)), []);
  useEffect(() => {
    if (isStreamTrack) updateMainTime(currentTime);
  }, [currentTime, isStreamTrack]);

  // Load disco data JSON when track changes — trigger on-demand extraction if missing
  useEffect(() => {
    const discoUrl = currentTrack?.discoDataUrl || '';
    setDiscoDataUrl(discoUrl);

    // If disco extraction is on and this track has no disco data, trigger extraction
    if (settings.discoKickExtract && currentTrack?.id && !discoUrl) {
      console.log(`[Disco] Track ${currentTrack.id} has no disco data — triggering extraction`);
      const controller = new AbortController();
      const jobId = currentTrack.id;
      (async () => {
        try {
          const data = await apiClient.post<{ status?: string; discoDataUrl?: string }>(
            `/songs/${jobId}/extract-kick`,
            undefined,
            { signal: controller.signal, silent: true },
          );
          if (controller.signal.aborted) return;
          if (data.status === 'exists' && data.discoDataUrl) {
            console.log(`[Disco] Disco data already exists for ${jobId}`);
            setDiscoDataUrl(data.discoDataUrl);
            return;
          }
          if (data.status !== 'started') return;
          console.log(`[Disco] Stem extraction started for ${jobId}`);

          // Poll the canonical song snapshot until the URL lands. The waiter
          // owns the cadence, the cache entry and cancellation.
          const { song } = await waitForQuery<{ song: { disco_data_url?: string } }>(songOptions(jobId), s => !s.song.disco_data_url, ({ song: s }) => {
            if (s.disco_data_url) setDiscoDataUrl(s.disco_data_url);
          }, { pollMs: 5000, timeoutMs: 30 * 60 * 1000, signal: controller.signal });
          console.log(`[Disco] Disco data ready for ${jobId}: ${song.disco_data_url}`);
          // The library list now carries a new disco URL; refresh it so the
          // player and other views see the same snapshot this wait resolved on.
          void invalidateSongs(queryClient);
        } catch (err) {
          if ((err as Error)?.name !== 'AbortError') console.warn('[Disco] On-demand extraction failed:', err);
        }
      })();
      return () => controller.abort();
    }
  }, [currentTrack?.id, currentTrack?.discoDataUrl, settings.discoKickExtract, queryClient]);

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
  const [isOffline, setIsOffline] = useState(false);

  // Presence beacon: the launcher asks /api/health whether any tab is already
  // open before opening a new one. The server counts each tab's /api/logs
  // stream, the one the Terminal reads, so a tab holds a single connection
  // (Chrome allows six per host across all tabs). SSE reconnects at the
  // network layer, so a background tab (whose timers Chrome throttles) still
  // re-registers within seconds of the server coming back; the shared stream
  // reopens itself after a non-200 (Vite proxy error while the server is down).
  useEffect(() => {
    let checking = false;
    let disposed = false;
    const confirmOffline = async () => {
      if (checking) return;
      checking = true;
      try {
        const res = await fetch('/api/health', { signal: AbortSignal.timeout(2000) });
        if (!res.ok && !disposed) setIsOffline(true);
      } catch {
        if (!disposed) setIsOffline(true);
      } finally { checking = false; }
    };
    const unsubscribe = subscribeEventSource('/api/logs', { error: () => void confirmOffline() }, { reconnect: true });
    return () => { disposed = true; unsubscribe(); };
  }, []);

  // Settings' own "Restart now" button posts to /api/shutdown/restart from deep
  // inside the panel and cannot reach this state directly, so it announces the
  // restart instead. Without this the page sat there against a dead server and
  // never reloaded when it came back (issue #153).
  useEffect(() => {
    const onRestarting = () => setIsRestarting(true);
    // A refused restart (the server is still up) drops the overlay again.
    const onRestartCancelled = () => setIsRestarting(false);
    window.addEventListener('hotstep:restarting', onRestarting);
    window.addEventListener('hotstep:restart-cancelled', onRestartCancelled);
    return () => {
      window.removeEventListener('hotstep:restarting', onRestarting);
      window.removeEventListener('hotstep:restart-cancelled', onRestartCancelled);
    };
  }, []);

  // ── URL-based routing ──────────────────────────────────────
  const navigateTo = useCallback((view: string) => {
    // Save deep Lyric Studio URL before leaving so we can restore it
    if (activeView === 'lyric-studio' && view !== 'lyric-studio') {
      try { localStorage.setItem('hs-lastLyricStudioUrl', window.location.pathname); } catch { /* ignore */ }
    }
    // Same idea for Training Studio: dataset/phase deep link.
    if (activeView === 'training-studio' && view !== 'training-studio') {
      try { localStorage.setItem('hs-lastTrainingStudioUrl', window.location.pathname); } catch { /* ignore */ }
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

  // downloadTrack.ts saves into the DOWNLOAD_DIR folder when one is set; say where it went.
  useEffect(() => {
    const onSaved = (e: Event) => {
      const { path, error } = (e as CustomEvent<{ path?: string; error?: string }>).detail;
      setToast(error
        ? { message: t('download.saveFailed', 'Download failed: {{error}}', { error }), type: 'error', isVisible: true }
        : { message: t('download.savedTo', 'Saved to {{path}}', { path }), type: 'success', isVisible: true });
    };
    window.addEventListener('hotstep:download-saved', onSaved);
    return () => window.removeEventListener('hotstep:download-saved', onSaved);
  }, [t]);

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

  // Song created callback — show it in the library now, then refetch every
  // song view (library, details, recent feeds).
  const handleSongCreated = useCallback((song: Song) => {
    patchLibrary(queryClient, prev => [song, ...prev.filter(s => s.id !== song.id)]);
    void invalidateSongs(queryClient);

    // Auto-extract drum stems + disco data for disco mode if setting enabled
    if (settings.discoKickExtract && song.id && !song.disco_data_url) {
      console.log(`[Disco] Auto-extracting drum stems for song ${song.id}`);
      apiClient.post<{ status?: string }>(`/songs/${song.id}/extract-kick`, undefined, { silent: true })
        .then(data => {
          if (data.status === 'started') {
            console.log(`[Disco] Stem extraction + analysis started for ${song.id}`);
          }
        })
        .catch(err => console.warn('[Disco] Stem extraction failed:', err));
    }
  }, [settings.discoKickExtract, queryClient]);

  // Play a song from the library (used by SongList, RightSidebar)
  const playSong = useCallback((song: Song) => {
    if (currentTrack?.id === song.id) {
      pbTogglePlay();
      return;
    }
    // A live render is played on its own, not inside a list: it is not in
    // `songs` (it has no row yet), so playFromList would place it at index 0 of
    // somebody else's list and skip/next would jump into the library the moment
    // its tail ended. There is also nothing to show in the details sidebar.
    if (song.streamJobId) {
      pbPlay(songToTrack(song));
      return;
    }
    setSelectedSong(song);
    setShowRightSidebar(true);
    playFromList(songToTrack(song), songs.map(songToTrack), 'library');
  }, [currentTrack?.id, songs]);

  // Notification click: open the finished song in the details panel.
  useEffect(() => {
    const onOpen = async (e: Event) => {
      const songId = (e as CustomEvent).detail?.songId as string | undefined;
      if (!songId) return;
      try {
        const { song } = await songApi.get(songId);
        if (song) { setSelectedSong(song); setShowRightSidebar(true); }
      } catch { /* song gone */ }
    };
    window.addEventListener('hotstep:open-song', onOpen);
    return () => window.removeEventListener('hotstep:open-song', onOpen);
  }, []);

  // Shared audio generation queue — resume on mount, subscribe only to active count
  useResumeQueue(token || undefined);
  useAudioQueueProjection(token || undefined);
  const activeJobCount = useAudioGenQueueSelector(s =>
    s.items.filter(i => i.status === 'pending' || i.status === 'loading-adapter' || i.status === 'generating').length
  );

  // ── The live MiniMax-Music3 render, as a track ──
  //
  // A streaming render is a real, playable track before its file exists, so it
  // is stood up as a synthetic Song and prepended to the generations grid. It
  // carries no audioUrl: playbackStore sees `streamJobId` and drives
  // mm3StreamStore instead of loading a URL. The moment the render is saved the
  // real row arrives from the server and this one is dropped — matched on the
  // job id, so the two can never both be on screen.
  const streamingItem = useAudioGenQueueSelector(s =>
    s.items.find(i => i.status === 'generating' && i.jobId && i.mm3Streaming === true) ?? null
  );
  const mm3StreamState = useMm3StreamAudio();
  // ONE CARD PER TAKE. An ensemble render produces several different songs at
  // once and they all stream simultaneously, so each gets its own card from the
  // moment the engine reports the take count — clicking one makes it the
  // audible take while the rest keep buffering. A single card would turn three
  // songs into a batch you only meet when it is over.
  const streamingSongs = useMemo<Song[]>(() => {
    if (!streamingItem?.jobId) return [];
    const p = (streamingItem.globalParams || {}) as Record<string, any>;
    const takes = Math.max(1, streamingItem.mm3TakeCount ?? 1);
    const base = streamingItem.generation?.title || p.title || 'Generating…';
    return Array.from({ length: takes }, (_, t) => ({
      id: `mm3-stream:${streamingItem.jobId}:${t}`,
      streamJobId: streamingItem.jobId,
      streamTake: t,
      title: takes > 1 ? `${base} (take ${t + 1})` : base,
      lyrics: p.lyrics || '',
      style: p.caption || p.prompt || '',
      caption: p.caption || p.prompt || '',
      audioUrl: '',
      duration: streamingItem.mm3Duration || 0,
      tags: [],
      artistName: streamingItem.artistName || '',
      isGenerating: true,
    } as Song));
  }, [streamingItem?.jobId, streamingItem?.mm3Duration, streamingItem?.mm3TakeCount,
      streamingItem?.generation?.title, streamingItem?.artistName, streamingItem?.globalParams]);
  const streamingSong = streamingSongs[0] ?? null;

  // Open the stream as soon as the engine confirms one. Here rather than in the
  // queue sidebar because App is always mounted and the sidebar is not — a
  // render started from Create and then watched from the Library page must
  // still play. mm3StreamEnsure is idempotent and remembers jobs it has already
  // opened, so it will not reopen one the user deliberately stopped.
  //
  // It only auto-plays when the bar is IDLE. Starting a stream over a track the
  // user is listening to would put two songs through the speakers at once; the
  // card is there to be clicked instead.
  // Every take, not just the first: the engine DROPS a stream nobody is
  // draining once it buffers past a threshold, so an unopened take loses its
  // live audio (its saved file is unaffected). Only take 0 may auto-play.
  useEffect(() => {
    mm3StreamEnsureTakes(
      streamingItem?.jobId ?? null,
      streamingItem?.mm3TakeCount ?? 1,
      streamingItem?.mm3Duration ?? 0,
      !isPlayingRef.current,
    );
  }, [streamingItem?.jobId, streamingItem?.mm3Duration, streamingItem?.mm3TakeCount]);

  // ...and hand the PLAY BAR to it the moment there is audio.
  //
  // Without this the stream plays while the bar still points at whatever was
  // there before: the grid card does not highlight, the waveform never appears
  // (it renders off currentTrack), and the transport controls the wrong track.
  // Audio starting and the bar knowing about it have to be the same event.
  const tookOverRef = useRef<string | null>(null);
  useEffect(() => {
    const jobId = mm3StreamState.jobId;
    if (!jobId || mm3StreamState.chunks === 0 || !streamingSong) return;
    if (tookOverRef.current === jobId) return;
    if (!mm3StreamState.playing) return;   // it did not auto-play; leave the bar alone
    tookOverRef.current = jobId;
    pbPlay(songToTrack(streamingSong));
  }, [mm3StreamState.jobId, mm3StreamState.chunks, mm3StreamState.playing, streamingSong]);


  // The library is the shared songs query above. When the server comes back
  // after a drop (the shared /api/logs stream reopens), refetch every song
  // view: songs may have changed while this tab could not hear about it. The
  // log lines themselves are not parsed here.
  useEffect(() => bridgeEvents(queryClient, {
    url: '/api/logs',
    reconnect: true,
    parse: () => undefined,
    map: () => undefined,
    reconnectInvalidate: SONG_VIEWS,
  }), [queryClient]);

  // Global params from context
  const globalParams = useGlobalParams();

  // Handle generation — merges content params (from CreatePanel) with global engine params (from context)
  const handleGenerate = useCallback((contentParams: Partial<GenerationParams>, contentIntent?: CreateIntent) => {
    if (!token) return;
    // First Generate click is a user gesture: ask once for done notifications.
    void requestNotifyPermissionFromGesture();
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
      contentIntent ? { ...contentIntent, params: { ...engineParams, ...contentIntent.params,
        source: 'create', coResident: settings.coResident, cacheLmCodes: settings.cacheLmCodes,
        parallelWhisper: settings.parallelWhisper, parallelQualityEval: settings.parallelQualityEval,
        parallelCoverArt: settings.parallelCoverArt } } : undefined,
    );
  }, [token, settings, globalParams, handleSongCreated]);

  // Handle delete
  const handleDelete = useCallback(async (song: Song) => {
    if (!token) return;
    await songApi.delete(song.id);
    void removeSongs(queryClient, [song.id]);
    if (selectedSong?.id === song.id) setSelectedSong(null);
    showToast(`Deleted "${song.title}"`, 'success');
  }, [token, selectedSong, queryClient]);

  // Global desktop hotkeys (see utils/hotkeys.ts); "?" opens the cheat sheet.
  const [showHotkeyHelp, setShowHotkeyHelp] = useState(false);
  const openHotkeyHelp = useCallback(() => setShowHotkeyHelp(true), []);
  const closeHotkeyHelp = useCallback(() => setShowHotkeyHelp(false), []);
  const [showPalette, setShowPalette] = useState(false);
  const openPalette = useCallback(() => setShowPalette(true), []);
  const closePalette = useCallback(() => setShowPalette(false), []);
  useEffect(() => {
    window.addEventListener(OPEN_PALETTE_EVENT, openPalette);
    return () => window.removeEventListener(OPEN_PALETTE_EVENT, openPalette);
  }, [openPalette]);
  useGlobalHotkeys(openHotkeyHelp, openPalette);

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
          await songApi.bulkDelete(ids);
          const idSet = new Set(ids);
          void removeSongs(queryClient, ids);
          if (selectedSong && idSet.has(selectedSong.id)) setSelectedSong(null);
          showToast(`Deleted ${ids.length} track${ids.length !== 1 ? 's' : ''}`, 'success');
        } catch (err: any) {
          showToast(`Delete failed: ${err.message}`, 'error');
        }
      },
    });
  }, [token, selectedSong, queryClient]);

  // Reuse handler
  const [reuseData, setReuseData] = useState<{ song: Song; timestamp: number } | null>(null);
  const handleReuse = useCallback((song: Song) => {
    setReuseData({ song, timestamp: Date.now() });
    navigateTo('create');
  }, [navigateTo]);

  // Send-to-Cover handler — loads a library track into Cover Studio (#61)
  const [coverSource, setCoverSource] = useState<{ song: Song; timestamp: number } | null>(null);
  const handleSendToCover = useCallback((song: Song) => {
    setCoverSource({ song, timestamp: Date.now() });
    navigateTo('cover-studio');
  }, [navigateTo]);

  // Metadata editor (#60)
  const [metadataEditSong, setMetadataEditSong] = useState<Song | null>(null);
  const handleEditMetadata = useCallback((song: Song) => setMetadataEditSong(song), []);

  // Song update handler
  const handleSongUpdate = useCallback((updatedSong: Song) => {
    patchLibrary(queryClient, prev => prev.map(s => s.id === updatedSong.id ? updatedSong : s));
    void invalidateSongs(queryClient);
    if (selectedSong?.id === updatedSong.id) setSelectedSong(updatedSong);
  }, [selectedSong, queryClient]);

  // Listen for cover art completion events (fired by SongList context menu)
  useEffect(() => {
    const handler = (e: Event) => {
      const { songId, coverUrl } = (e as CustomEvent).detail;
      if (!songId || !coverUrl) return;
      void patchSong(queryClient, songId, { coverUrl, cover_url: coverUrl });
      // Also update selected song if it's the one that got cover art
      setSelectedSong(prev =>
        prev?.id === songId ? { ...prev, coverUrl, cover_url: coverUrl } as Song : prev
      );
    };
    window.addEventListener('cover-art-updated', handler);
    return () => window.removeEventListener('cover-art-updated', handler);
  }, [queryClient]);

  // Listen for post-processing completion (fired by postProcessStore after a
  // "Run Post-Processing" finishes). The song now has a mastered file, which
  // the player prefers and which makes the menu item disable itself.
  useEffect(() => {
    const handler = (e: Event) => {
      const { songId, masteredAudioUrl } = (e as CustomEvent).detail;
      if (!songId || !masteredAudioUrl) return;
      void patchSong(queryClient, songId, { masteredAudioUrl, mastered_audio_url: masteredAudioUrl });
      setSelectedSong(prev =>
        prev?.id === songId
          ? { ...prev, masteredAudioUrl, mastered_audio_url: masteredAudioUrl } as Song
          : prev
      );
    };
    window.addEventListener('song-postprocessed', handler);
    return () => window.removeEventListener('song-postprocessed', handler);
  }, [queryClient]);

  // ...and the reverse, when a processed version is thrown away.
  useEffect(() => {
    const handler = (e: Event) => {
      const { songId } = (e as CustomEvent).detail || {};
      if (!songId) return;
      void patchSong(queryClient, songId, { masteredAudioUrl: '', mastered_audio_url: '' });
      setSelectedSong(prev =>
        prev?.id === songId
          ? { ...prev, masteredAudioUrl: '', mastered_audio_url: '' } as Song
          : prev
      );
    };
    window.addEventListener('song-postprocess-reverted', handler);
    return () => window.removeEventListener('song-postprocess-reverted', handler);
  }, [queryClient]);

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
      const { song: updated } = await songApi.update(song.id, { title: newTitle });
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
  if (isOffline) {
    return <ServerOfflineOverlay />;
  }
  // Main content renderer
  const renderContent = () => {
    if (activeView === 'settings') {
      return (
        <DiscoPulseWrapper hue={DISCO.activity} className="flex-1 overflow-y-auto">
          <SettingsPanel
            settings={settings}
            onSettingsChange={setSettings}
            onNukeComplete={() => {
              patchLibrary(queryClient, () => []);
              void invalidateSongs(queryClient);
              setSelectedSong(null);
              // The queue is a persisted singleton store outside the song
              // cache, so it survives the nuke unless explicitly cleared.
              clearFinishedFromAudioQueue();
            }}
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
          <CoverStudio coverSource={coverSource} />
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

    if (activeView === 'storm') {
      return (
        <DiscoPulseWrapper hue={DISCO.storm} className="flex-1 overflow-hidden">
          <StormPage onGenerate={handleGenerate} activeJobCount={activeJobCount} />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'midi-studio') {
      return (
        <DiscoPulseWrapper hue={DISCO.playlist} className="flex-1 overflow-hidden">
          <MidiStudio />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'training-studio') {
      return (
        <DiscoPulseWrapper hue={DISCO.trainingStudio} className="flex-1 overflow-hidden">
          <TrainingStudio />
        </DiscoPulseWrapper>
      );
    }

    if (activeView === 'compare-studio') {
      return (
        <DiscoPulseWrapper hue={DISCO.playlist} className="flex-1 overflow-hidden">
          <CompareStudio />
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

    if (activeView === 'song-builder') {
      return (
        <DiscoPulseWrapper hue={DISCO.createPanel} className="flex-1 overflow-hidden">
          <SongBuilder />
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
              onReuse={handleReuse} onSendToCover={handleSendToCover} onEditMetadata={handleEditMetadata}
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
                  noAdapterAudioUrl: song.noAdapterAudioUrl || song.noadapter_audio_url || '',
                  artistName: song.artistName || '',
                  coverUrl: song.coverUrl || song.cover_url || '',
                  duration: typeof song.duration === 'number' ? song.duration : undefined,
                  style: song.style || '',
                });
                showToast(t('app.addedToPlaylist'), 'success');
              }}
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
                className="flex-shrink-0 h-full bg-zinc-50 dark:bg-suno z-10 border-l border-zinc-200 dark:border-white/5 max-xl:max-w-[26vw]"
                style={{ width: rightSidebarWidth }}
              >
                <RightSidebar
                  song={selectedSong}
                  onClose={() => setShowRightSidebar(false)}
                  onReuse={handleReuse} onSendToCover={handleSendToCover} onEditMetadata={handleEditMetadata}
                  onDelete={handleDelete}
                  onPlay={playSong}
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
              onCompare={(ids) => { setPendingCompareSongs(ids); navigateTo('compare-studio'); }}
              onExportZip={async (ids) => {
                try {
                  const res = await fetch('/api/songs/export-zip', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
                    body: JSON.stringify({ ids }),
                  });
                  if (!res.ok) throw new Error((await res.json().catch(() => null))?.error || `HTTP ${res.status}`);
                  const blob = await res.blob();
                  const name = /filename="([^"]+)"/.exec(res.headers.get('Content-Disposition') || '')?.[1] || 'hotstep-library.zip';
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
                  setTimeout(() => URL.revokeObjectURL(url), 10_000);
                } catch (e) {
                  showToast(t('library.exportZipFailed', 'Zip export failed: {{error}}', { error: (e as Error).message }), 'error');
                }
              }}
              onSelect={(s) => { setSelectedSong(s); setShowRightSidebar(true); }}
              onReuse={handleReuse} onSendToCover={handleSendToCover} onEditMetadata={handleEditMetadata}
              onDownload={handleDownload}
              onRename={handleRename}
              showImport
              onAddToPlaylist={(song) => {
                addToPlaylist({
                  id: song.id,
                  title: song.title || 'Untitled',
                  audioUrl: song.audioUrl || song.audio_url || '',
                  masteredAudioUrl: song.masteredAudioUrl || song.mastered_audio_url || '',
                  noAdapterAudioUrl: song.noAdapterAudioUrl || song.noadapter_audio_url || '',
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
                className="flex-shrink-0 h-full bg-zinc-50 dark:bg-suno z-10 border-l border-zinc-200 dark:border-white/5 max-xl:max-w-[26vw]"
                style={{ width: rightSidebarWidth }}
              >
                <RightSidebar
                  song={selectedSong}
                  onClose={() => setShowRightSidebar(false)}
                  onReuse={handleReuse} onSendToCover={handleSendToCover} onEditMetadata={handleEditMetadata}
                  onDelete={handleDelete}
                  onPlay={playSong}
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
    // Layout: side-by-side on wide landscape, stacked vertically on narrow (<1024px)
    // and on portrait screens >= 1024px (e.g. 1080x1920 at 100%).
    return (
      <div className="flex flex-1 min-w-0 min-h-0
        min-[1024px]:flex-row max-[1023px]:flex-col
        min-[1024px]:overflow-clip max-[1023px]:overflow-visible
        portrait-wide:flex-col
        portrait-wide:overflow-visible">
        {/* Create Panel — resizable on wide landscape, full-width on narrow/portrait.
            Tailwind's ! prefix generates !important which overrides inline style. */}
        <DiscoPulseWrapper
          hue={DISCO.createPanel}
          className="flex-shrink-0 min-w-0 border-zinc-200 dark:border-white/5
            min-[1024px]:h-full min-[1024px]:border-r min-[1024px]:max-xl:max-w-[32vw]
            max-[1023px]:!w-full max-[1023px]:border-b max-[1023px]:h-auto max-[1023px]:border-r-0
            portrait-wide:!w-full portrait-wide:border-b portrait-wide:border-r-0 portrait-wide:h-auto portrait-wide:max-w-none"
          style={{ width: createPanelWidth }}
        >
          <CreatePanel
            onGenerate={handleGenerate}
            activeJobCount={activeJobCount}
            reuseData={reuseData}
          />
        </DiscoPulseWrapper>

        {/* Left resize handle — hidden on narrow and portrait */}
        <div
          className="flex-shrink-0 w-1.5 cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors
            min-[1024px]:h-full max-[1023px]:hidden
            portrait-wide:hidden"
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
            songs={[
              // The live render first: it is the newest thing here, and it is
              // playable before its file exists. Dropped automatically once the
              // real row arrives, because the queue item stops being 'generating'.
              ...streamingSongs,
              ...songs.filter(s => getSongSource(s) === 'create'),
            ]}
            currentSongId={currentSong?.id}
            onPlay={playSong}
            onDelete={handleDelete}
            onBulkDelete={handleBulkDelete}
            onSelect={(s) => { setSelectedSong(s); setShowRightSidebar(true); }}
            onReuse={handleReuse} onSendToCover={handleSendToCover} onEditMetadata={handleEditMetadata}
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
                noAdapterAudioUrl: song.noAdapterAudioUrl || song.noadapter_audio_url || '',
                artistName: song.artistName || '',
                coverUrl: song.coverUrl || song.cover_url || '',
                duration: typeof song.duration === 'number' ? song.duration : undefined,
                style: song.style || '',
              });
              showToast(t('app.addedToPlaylist'), 'success');
            }}
          />
        </DiscoPulseWrapper>


        {/* Right Sidebar — resizable; hidden on narrow / portrait stacked layout */}
        {showRightSidebar && selectedSong && (
          <>
            {/* Right resize handle */}
            <div
              className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors
                max-[1023px]:hidden portrait-wide:hidden"
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
              className="flex-shrink-0 h-full bg-zinc-50 dark:bg-suno z-10 border-l border-zinc-200 dark:border-white/5
                max-xl:max-w-[26vw] max-[1023px]:hidden portrait-wide:hidden"
              style={{ width: rightSidebarWidth }}
            >
              <RightSidebar
                song={selectedSong}
                onClose={() => setShowRightSidebar(false)}
                onReuse={handleReuse} onSendToCover={handleSendToCover} onEditMetadata={handleEditMetadata}
                onDelete={handleDelete}
                onPlay={playSong}
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
    <div className="flex flex-col h-[100dvh] bg-white dark:bg-suno text-zinc-900 dark:text-white font-sans antialiased selection:bg-pink-500/30 transition-all duration-300">
      {/* Global Parameter Bar — full width, above everything.
          The mobile compact row (hamburger + Settings) lives inside
          GlobalParamBar so both desktop and mobile bars share the same
          backdrop and z-level. */}
      <JobProgressWatcher />
      <GlobalParamBar onOpenMobileNav={() => setMobileNavOpen(true)} />

      {/* Asks before running a generation queue restored from a previous
          session, rather than starting it unattended (issue #100). Renders
          nothing when there is nothing waiting. */}
      <NoModelsBanner />
      <RestoredQueueBanner />
      <AudioQueueMigrationBanner />

      <div className="flex-1 min-h-0 flex overflow-clip">
        <DiscoPulseWrapper hue={DISCO.sidebar}>
        <Sidebar
          activeView={activeView}
          onViewChange={navigateTo}
          mobileOpen={mobileNavOpen}
          onMobileClose={() => setMobileNavOpen(false)}
          onQuit={() => {
            setConfirmDialog({
              title: t('app.quitDialog.title'),
              message: t('app.quitDialog.message'),
              confirmLabel: t('app.quitDialog.confirm'),
              danger: true,
              onConfirm: async () => {
                setConfirmDialog(null);
                try { await shutdownApi.quit(); } catch { /* shutting down */ }
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
                try { await shutdownApi.restart(); } catch { /* restarting */ }
              },
            });
          }}
          theme={theme}
          onToggleTheme={toggleTheme}
          showAssistant={showAssistant}
          onToggleAssistant={() => setShowAssistant(prev => !prev)}
        />
        </DiscoPulseWrapper>

        {/* Left stack — everything the activity column does NOT cover. The
            player lives in here rather than spanning the window, so the
            column runs the full height and the transport stops at its edge. */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-clip">
        <main className="flex-1 min-w-0 min-h-0 flex min-[1024px]:overflow-clip max-[1023px]:overflow-y-auto portrait-wide:overflow-y-auto relative">
          <ErrorBoundary resetKey={activeView}>
            {renderContent()}
          </ErrorBoundary>
        </main>

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
              {/* Two analysers, one visible at a time. The media-element one must
                  stay mounted forever (it owns a MediaElementSourceNode that carries
                  the file decks' audio — see SpectrumAnalyzer's header), so it is
                  collapsed rather than unmounted while a live render is playing. */}
              <SpectrumAnalyzer
                mediaElement={spectrumMediaEl}
                visible={spectrumEnabled && playerActive && !isStreamTrack}
                isPlaying={isPlaying}
              />
              <StreamSpectrum visible={spectrumEnabled && playerActive && isStreamTrack} />
              {trimMode && (
                <TrimControls
                  trimInPoint={trimInPoint}
                  trimOutPoint={trimOutPoint}
                  trimClickCount={trimClickCount}
                  songId={currentTrack?.id ?? null}
                  audioUrl={currentTrack?.audioUrl ?? null}
                  onReload={pbReloadCurrentTrack}
                  onCancel={() => pbSetTrimMode(false)}
                />
              )}
              {/* One waveform. The live-render canvas replaces it rather than
                  stacking: a track still being written has no file to draw peaks
                  from, so its envelope comes from mm3StreamStore as windows land. */}
              <div className="relative" style={{ height: 56 }}>
                {isStreamTrack ? (
                  <StreamWaveform height={56} />
                ) : (
                  <Waveform
                    height={56}
                    trimIn={trimMode ? trimInPoint : null}
                    trimOut={trimMode ? trimOutPoint : null}
                    onClickTime={trimMode ? pbHandleTrimClick : undefined}
                  />
                )}
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
            {/* Post-processing runs for minutes with no other on-screen trace.
                Docked at the root so it is visible from every view, Library
                included. */}
            <PostProcessDock />

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
              pitch441={pitch441}
              onTogglePitch441={() => pbSetPitch441(!pitch441)}
              isShuffle={shuffle}
              onToggleShuffle={() => pbSetShuffle(!shuffle)}
              repeatMode={repeat}
              onToggleRepeat={pbCycleRepeat}
              onReusePrompt={() => currentSong && handleReuse(currentSong as Song)}
              onDelete={() => currentSong && handleDelete(currentSong as Song)}
              onDownload={() => currentSong && downloadTrack(currentSong as Song)}
              playMastered={playMastered}
              playNoAdapter={playNoAdapter}
              onSetPlaybackVariant={pbSetPlaybackVariant}
              onDownloadVariant={(v) => currentSong && downloadTrackVersion(currentSong as Song, v)}
              spectrumEnabled={spectrumEnabled}
              onToggleSpectrum={() => pbSetSpectrumEnabled(!spectrumEnabled)}
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
        </div>

        {/* Activity column — Playlist/Recent + Queue + Terminal. One instance for
            every view, so the queue and the engine log are never more than a
            glance away regardless of which studio is open. */}
        <div
          className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors max-[1023px]:hidden portrait-wide:hidden"
          onMouseDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startW = activitySidebarWidth;
            const onMove = (ev: MouseEvent) => {
              setActivitySidebarWidth(Math.min(900, Math.max(240, startW + startX - ev.clientX)));
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
        <DiscoPulseWrapper hue={DISCO.activity}
          className="h-full flex-shrink-0 border-l border-zinc-200 dark:border-white/5 overflow-hidden max-xl:max-w-[30vw] max-[1023px]:hidden portrait-wide:hidden"
          style={{ width: activitySidebarWidth }}
        >
          <ActivitySidebar
            source={ACTIVITY_SOURCE[activeView]}
            showToast={showToast}
            compact={activitySidebarWidth < 380}
            showTerminal
          />
        </DiscoPulseWrapper>

        {/* Assistant Panel — right of the activity column, resizable */}
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
              className="flex-shrink-0 h-full border-l border-zinc-200 dark:border-white/5 max-xl:max-w-[26vw]"
              style={{ width: assistantWidth }}
            >
              <AssistantPanel onClose={() => setShowAssistant(false)} activeView={activeView} />
            </DiscoPulseWrapper>
          </>
        )}

      </div>

      {/* Modals */}
      {showHotkeyHelp && <HotkeyHelpModal onClose={closeHotkeyHelp} />}
      {showPalette && (
        <CommandPalette
          onClose={closePalette}
          onNavigate={navigateTo}
          songs={songs}
          onPickSong={(id) => {
            const song = songs.find(s => s.id === id);
            if (song) {
              navigateTo('library');
              playFromList(songToTrack(song), songs.map(songToTrack), 'library');
              setSelectedSong(song);
            } else {
              navigateTo('library');
            }
          }}
        />
      )}
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

      {/* Per-track cover art prompt modal (#67) — single instance, driven by window events */}
      <CoverArtPromptModal />

      {/* "Upload into Library" — single instance, opened from the Library header */}
      <ImportTracksModal />

      {metadataEditSong && token && (
        <MetadataEditorModal
          song={metadataEditSong}
          onClose={() => setMetadataEditSong(null)}
          onSaved={(s) => { handleSongUpdate(s); setMetadataEditSong(null); }}
        />
      )}
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
