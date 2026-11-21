// useAppRuntime.ts — the headless runtime every shell shares.
//
// Everything here ran inside App.tsx's AppContent before shells existed: queue
// resume and projection, opening live MM3 takes and handing them the play bar,
// song-cache patches from window events, the library refetch on reconnect,
// offline/restart detection, and the settings flags a Create render carries.
// A shell calls this once and renders however it likes. Must sit inside
// AuthProvider, QueryClientProvider and GlobalParamsProvider.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { probeServerHealth } from '../services/shellTransport';
import { subscribeEventSource } from '../services/sharedEventSource';
import { useAuth } from '../context/AuthContext';
import { useGlobalParams } from '../context/GlobalParamsContext';
import { usePersistedState } from '../hooks/usePersistedState';
import { apiClient } from '../services/httpClient';
import { bridgeEvents } from '../queries/sseBridge';
import { SONG_VIEWS, invalidateSongs, patchLibrary, patchSong } from '../queries/shell-queue/songs';
import { useAudioQueueProjection } from '../queries/shell-queue/projection';
import { useResumeQueue, useAudioGenQueueSelector } from '../stores/audioGenQueueStore';
import { mm3StreamEnsureTakes, useMm3StreamAudio } from '../stores/mm3StreamStore';
import { usePlaybackSelector, play as pbPlay, songToTrack } from '../stores/playbackStore';
import { DEFAULT_SETTINGS, type AppSettings } from './appSettings';
import type { UpdateRestartAttempt } from '../services/updatesApi';
import type { GenerationParams, Song } from '../types';

export interface AppRuntimeOptions {
  /** A shell that keeps its own copy of a song (a details panel) hears every
   *  cache patch here too. Read through a ref, so it may change every render. */
  onSongPatch?: (songId: string, patch: Partial<Song>) => void;
}

export function useAppRuntime({ onSongPatch }: AppRuntimeOptions = {}) {
  const { token } = useAuth();
  const queryClient = useQueryClient();
  const [settings, setSettings] = usePersistedState<AppSettings>('ace-settings', DEFAULT_SETTINGS);
  const isPlaying = usePlaybackSelector(s => s.isPlaying);
  // Read inside an effect that must NOT re-run when playback state changes —
  // the auto-play decision is made once, when the stream opens.
  const isPlayingRef = useRef(isPlaying);
  isPlayingRef.current = isPlaying;
  const patchRef = useRef(onSongPatch);
  patchRef.current = onSongPatch;

  const [isRestarting, setIsRestarting] = useState(false);
  const [isOffline, setIsOffline] = useState(false);
  const [updateAttempt, setUpdateAttempt] = useState<UpdateRestartAttempt | null>(null);

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
        const res = await probeServerHealth();
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
  // queue sidebar because the shell is always mounted and the sidebar is not — a
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

  // The library is the shared songs query. When the server comes back
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

  // Global params from context, merged with the content params a render carries
  // (from CreatePanel) and the settings flags. The Build by section tab starts
  // its session from the same params.
  const globalParams = useGlobalParams();
  const enrichCreateParams = useCallback((contentParams: Partial<GenerationParams>) => ({
    ...globalParams.getGlobalParams(),
    ...contentParams,
    source: 'create',
    coResident: settings.coResident,
    cacheLmCodes: settings.cacheLmCodes,
    parallelWhisper: settings.parallelWhisper,
    parallelQualityEval: settings.parallelQualityEval,
    parallelCoverArt: settings.parallelCoverArt,
  }), [globalParams, settings]);

  // Listen for cover art completion events (fired by SongList context menu)
  useEffect(() => {
    const handler = (e: Event) => {
      const { songId, coverUrl } = (e as CustomEvent).detail;
      if (!songId || !coverUrl) return;
      const patch = { coverUrl, cover_url: coverUrl };
      void patchSong(queryClient, songId, patch);
      patchRef.current?.(songId, patch);
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
      const patch = { masteredAudioUrl, mastered_audio_url: masteredAudioUrl };
      void patchSong(queryClient, songId, patch);
      patchRef.current?.(songId, patch);
    };
    window.addEventListener('song-postprocessed', handler);
    return () => window.removeEventListener('song-postprocessed', handler);
  }, [queryClient]);

  // ...and the reverse, when a processed version is thrown away.
  useEffect(() => {
    const handler = (e: Event) => {
      const { songId } = (e as CustomEvent).detail || {};
      if (!songId) return;
      const patch = { masteredAudioUrl: '', mastered_audio_url: '' };
      void patchSong(queryClient, songId, patch);
      patchRef.current?.(songId, patch);
    };
    window.addEventListener('song-postprocess-reverted', handler);
    return () => window.removeEventListener('song-postprocess-reverted', handler);
  }, [queryClient]);

  // A rename from the shared song menu (SongActionsMenu) — already saved.
  useEffect(() => {
    const handler = (e: Event) => {
      const { songId, title } = (e as CustomEvent).detail || {};
      if (!songId || !title) return;
      void patchSong(queryClient, songId, { title });
      patchRef.current?.(songId, { title });
    };
    window.addEventListener('song-renamed', handler);
    return () => window.removeEventListener('song-renamed', handler);
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

  return {
    settings, setSettings,
    isOffline, isRestarting, setIsRestarting, updateAttempt, setUpdateAttempt,
    activeJobCount, streamingSongs, streamingSong,
    handleSongCreated, enrichCreateParams,
  };
}
