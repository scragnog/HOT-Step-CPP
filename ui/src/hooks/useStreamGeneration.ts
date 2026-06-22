// useStreamGeneration.ts — React hook for streaming audio preview via SSE
//
// Connects to GET /api/generate/stream/:jobId once a generation job is running.
// Receives 'preview' events with WAV file paths, fetches them, decodes to
// AudioBuffers. Each preview REPLACES the previous one (same full track at
// increasing quality as denoising progresses).
//
// Exposes: status, previews[], play(), pause(), stop()

import { useState, useEffect, useRef, useCallback } from 'react';

export interface StreamPreview {
  url: string;
  step: number;
  totalSteps: number;
  slot: number;
  /** AudioBuffer decoded from the preview WAV (null if not yet loaded) */
  buffer: AudioBuffer | null;
}

export interface StreamStatus {
  status: string;
  stage: string;
  progress: number;
}

export interface StreamGenerationState {
  /** Whether the SSE connection is active */
  connected: boolean;
  /** Latest status from the server */
  status: StreamStatus | null;
  /** All received previews in order */
  previews: StreamPreview[];
  /** Whether audio is currently playing */
  playing: boolean;
  /** Whether the generation is complete */
  done: boolean;
  /** Error message if generation failed */
  error: string | null;
  /** Final result (audio URLs etc.) when done */
  result: any | null;
}

export function useStreamGeneration(jobId: string | null) {
  const [state, setState] = useState<StreamGenerationState>({
    connected: false,
    status: null,
    previews: [],
    playing: false,
    done: false,
    error: null,
    result: null,
  });

  const esRef = useRef<EventSource | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const isPlayingRef = useRef(false);
  const latestBufferRef = useRef<AudioBuffer | null>(null);
  const previewsRef = useRef<StreamPreview[]>([]);
  const playbackOffsetRef = useRef(0);  // where in the track the user was listening

  // ── Ensure AudioContext exists ──────────────────────────────────────
  const ensureAudioCtx = useCallback((): AudioContext => {
    if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
      audioCtxRef.current = new AudioContext();
    }
    return audioCtxRef.current;
  }, []);

  // ── Stop current playback (internal) ────────────────────────────────
  const stopCurrentSource = useCallback(() => {
    if (currentSourceRef.current) {
      try { currentSourceRef.current.stop(); } catch { /* already stopped */ }
      currentSourceRef.current = null;
    }
  }, []);

  // ── Play the latest buffer from a given offset ──────────────────────
  const playLatestFrom = useCallback((offset: number) => {
    const buffer = latestBufferRef.current;
    if (!buffer) return;

    const ctx = ensureAudioCtx();
    if (ctx.state === 'suspended') ctx.resume();

    stopCurrentSource();

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    // Clamp offset to buffer duration
    const safeOffset = Math.min(Math.max(0, offset), buffer.duration - 0.1);
    source.start(0, safeOffset);
    playbackOffsetRef.current = safeOffset;

    // Track playback position so we can resume from the right spot on next preview
    const startedAt = ctx.currentTime;
    currentSourceRef.current = source;

    source.onended = () => {
      currentSourceRef.current = null;
      // Calculate where playback ended
      const elapsed = ctx.currentTime - startedAt;
      playbackOffsetRef.current = safeOffset + elapsed;

      if (isPlayingRef.current && playbackOffsetRef.current < buffer.duration - 0.5) {
        // Buffer ended prematurely (shouldn't happen), don't auto-replay
      } else {
        // Reached end of track — stop playing
        isPlayingRef.current = false;
        setState(prev => ({ ...prev, playing: false }));
        playbackOffsetRef.current = 0;
      }
    };
  }, [ensureAudioCtx, stopCurrentSource]);

  // ── Fetch and decode a preview WAV file ─────────────────────────────
  const fetchPreview = useCallback(async (url: string): Promise<AudioBuffer | null> => {
    try {
      const res = await fetch(url);
      if (!res.ok) return null;
      const arrayBuffer = await res.arrayBuffer();
      const ctx = ensureAudioCtx();
      return await ctx.decodeAudioData(arrayBuffer);
    } catch (err) {
      console.warn('[StreamGen] Failed to decode preview:', url, err);
      return null;
    }
  }, [ensureAudioCtx]);

  // ── Public controls ─────────────────────────────────────────────────
  const play = useCallback(() => {
    if (isPlayingRef.current) return;
    isPlayingRef.current = true;
    setState(prev => ({ ...prev, playing: true }));
    playLatestFrom(0);
  }, [playLatestFrom]);

  const pause = useCallback(() => {
    if (!isPlayingRef.current) return;
    // Save current position before stopping
    if (currentSourceRef.current && audioCtxRef.current) {
      // We can't query position directly, but we track it via startedAt
    }
    isPlayingRef.current = false;
    setState(prev => ({ ...prev, playing: false }));
    stopCurrentSource();
  }, [stopCurrentSource]);

  const stop = useCallback(() => {
    isPlayingRef.current = false;
    setState(prev => ({ ...prev, playing: false }));
    stopCurrentSource();
    playbackOffsetRef.current = 0;
  }, [stopCurrentSource]);

  // ── Full cleanup: stop audio + close context ────────────────────────
  const fullCleanup = useCallback(() => {
    isPlayingRef.current = false;
    stopCurrentSource();
    playbackOffsetRef.current = 0;
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    latestBufferRef.current = null;
  }, [stopCurrentSource]);

  // ── SSE connection ──────────────────────────────────────────────────
  useEffect(() => {
    if (!jobId) {
      // jobId went null → generation ended or was never started
      // Stop any playing audio
      fullCleanup();
      previewsRef.current = [];
      setState({
        connected: false, status: null, previews: [],
        playing: false, done: false, error: null, result: null,
      });
      return;
    }

    // Reset state for new job
    previewsRef.current = [];
    latestBufferRef.current = null;
    playbackOffsetRef.current = 0;

    const es = new EventSource(`/api/generate/stream/${jobId}`);
    esRef.current = es;

    es.addEventListener('open', () => {
      setState(prev => ({ ...prev, connected: true }));
    });

    es.addEventListener('status', (event) => {
      try {
        const data = JSON.parse(event.data);
        setState(prev => ({
          ...prev,
          status: {
            status: data.status,
            stage: data.stage || '',
            progress: data.progress || 0,
          },
        }));
      } catch { /* ignore */ }
    });

    es.addEventListener('preview', async (event) => {
      try {
        const data = JSON.parse(event.data);
        const preview: StreamPreview = {
          url: data.url,
          step: data.step,
          totalSteps: data.totalSteps,
          slot: data.slot,
          buffer: null,
        };

        // Add to list
        previewsRef.current = [...previewsRef.current, preview];
        setState(prev => ({
          ...prev,
          previews: [...previewsRef.current],
        }));

        // Fetch and decode the WAV
        const buffer = await fetchPreview(data.url);
        if (buffer) {
          preview.buffer = buffer;
          latestBufferRef.current = buffer;

          // If currently playing, seamlessly switch to the new (better quality) buffer
          // Resume from approximately where the user was in the track
          if (isPlayingRef.current && audioCtxRef.current) {
            // Calculate current playback position
            let currentPos = playbackOffsetRef.current;
            if (currentSourceRef.current) {
              // Rough estimate based on when we started
              currentPos = playbackOffsetRef.current;
              // We'll use the tracked offset since Web Audio doesn't expose currentTime per source
            }
            playLatestFrom(Math.min(currentPos, buffer.duration - 0.5));
          }

          setState(prev => ({
            ...prev,
            previews: [...previewsRef.current],
          }));
        }
      } catch { /* ignore */ }
    });

    es.addEventListener('done', (event) => {
      try {
        const data = JSON.parse(event.data);
        // Stop preview playback — user will use the normal player for the final track
        stop();
        setState(prev => ({
          ...prev,
          done: true,
          result: data.result,
        }));
      } catch { /* ignore */ }
      es.close();
    });

    es.addEventListener('error', (event) => {
      const messageEvent = event as MessageEvent;
      if (messageEvent.data) {
        try {
          const data = JSON.parse(messageEvent.data);
          setState(prev => ({
            ...prev,
            error: data.error || 'Generation failed',
            done: true,
          }));
        } catch { /* ignore */ }
      }
      stop();
      setState(prev => ({ ...prev, connected: false }));
      es.close();
    });

    return () => {
      es.close();
      esRef.current = null;
      // Stop audio when SSE disconnects
      stop();
      setState(prev => ({ ...prev, connected: false }));
    };
  }, [jobId, fetchPreview, playLatestFrom, stop, fullCleanup]);

  // Cleanup AudioContext on unmount
  useEffect(() => {
    return () => fullCleanup();
  }, [fullCleanup]);

  return {
    ...state,
    play,
    pause,
    stop,
  };
}
