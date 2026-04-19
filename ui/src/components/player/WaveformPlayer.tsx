// WaveformPlayer.tsx — wavesurfer.js wrapper for Bars mode waveform visualization
//
// Renders a full-width waveform using wavesurfer.js with:
//   - Bars mode (barWidth, barGap, barRadius)
//   - Hover plugin (purple line + timestamp label)
//   - Regions plugin (section markers from LRC, added externally)
//
// Exposes imperative API via ref for parent control.

import {
  useEffect,
  useRef,
  useImperativeHandle,
  forwardRef,
  useCallback,
} from 'react';
import WaveSurfer from 'wavesurfer.js';
import Hover from 'wavesurfer.js/dist/plugins/hover.esm.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';

export interface WaveformPlayerHandle {
  play: () => void;
  pause: () => void;
  playPause: () => void;
  seekTo: (fraction: number) => void;
  setVolume: (v: number) => void;
  setPlaybackRate: (r: number) => void;
  loadUrl: (url: string) => void;
  getCurrentTime: () => number;
  getDuration: () => number;
  getMediaElement: () => HTMLMediaElement | null;
  /** Add section markers via the Regions plugin */
  clearRegions: () => void;
  addMarker: (time: number, label: string, color?: string) => void;
}

interface WaveformPlayerProps {
  /** Called on every timeupdate tick */
  onTimeUpdate?: (currentTime: number) => void;
  /** Called when duration is known */
  onDurationChange?: (duration: number) => void;
  /** Called when play state changes */
  onPlayChange?: (isPlaying: boolean) => void;
  /** Called when audio finishes */
  onFinish?: () => void;
  /** Called when audio is ready to play */
  onReady?: (duration: number) => void;
  /** Initial volume 0–1 */
  volume?: number;
  /** Initial playback rate */
  playbackRate?: number;
}

export const WaveformPlayer = forwardRef<WaveformPlayerHandle, WaveformPlayerProps>(
  (
    {
      onTimeUpdate,
      onDurationChange,
      onPlayChange,
      onFinish,
      onReady,
      volume = 0.8,
      playbackRate = 1.0,
    },
    ref
  ) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WaveSurfer | null>(null);
    const regionsRef = useRef<RegionsPlugin | null>(null);
    const pendingUrlRef = useRef<string | null>(null);

    // Stable callback refs to avoid re-creating wavesurfer on callback changes
    const onTimeUpdateRef = useRef(onTimeUpdate);
    onTimeUpdateRef.current = onTimeUpdate;
    const onDurationChangeRef = useRef(onDurationChange);
    onDurationChangeRef.current = onDurationChange;
    const onPlayChangeRef = useRef(onPlayChange);
    onPlayChangeRef.current = onPlayChange;
    const onFinishRef = useRef(onFinish);
    onFinishRef.current = onFinish;
    const onReadyRef = useRef(onReady);
    onReadyRef.current = onReady;

    // Initialize wavesurfer once
    useEffect(() => {
      if (!containerRef.current) return;

      const regions = RegionsPlugin.create();
      regionsRef.current = regions;

      const ws = WaveSurfer.create({
        container: containerRef.current,
        height: 56,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        cursorWidth: 2,
        cursorColor: '#ec4899',
        // Use a canvas gradient for the progress color
        waveColor: 'rgba(113, 113, 122, 0.4)',
        progressColor: '#ec4899',
        backend: 'MediaElement',
        normalize: true,
        interact: true,
        hideScrollbar: true,
        autoScroll: false,
        plugins: [
          Hover.create({
            lineColor: '#a855f7',
            lineWidth: 2,
            labelBackground: '#18181b',
            labelColor: '#e4e4e7',
            labelSize: '11px',
          }),
          regions,
        ],
      });

      // Apply gradient to the progress wave after creation
      // wavesurfer v7 supports CanvasGradient for progressColor
      const canvas = containerRef.current.querySelector('canvas');
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
          gradient.addColorStop(0, '#ec4899');
          gradient.addColorStop(1, '#a855f7');
          ws.setOptions({ progressColor: gradient });
        }
      }

      ws.setVolume(volume);

      // Events
      ws.on('timeupdate', (t: number) => onTimeUpdateRef.current?.(t));
      ws.on('ready', (dur: number) => {
        onDurationChangeRef.current?.(dur);
        onReadyRef.current?.(dur);
        // Apply gradient now that canvas is sized
        const c = containerRef.current?.querySelector('canvas');
        if (c) {
          const ctx2 = c.getContext('2d');
          if (ctx2) {
            const g = ctx2.createLinearGradient(0, 0, c.width, 0);
            g.addColorStop(0, '#ec4899');
            g.addColorStop(1, '#a855f7');
            ws.setOptions({ progressColor: g });
          }
        }
      });
      ws.on('play', () => onPlayChangeRef.current?.(true));
      ws.on('pause', () => onPlayChangeRef.current?.(false));
      ws.on('finish', () => onFinishRef.current?.());

      wsRef.current = ws;

      // If a URL was requested before init completed, load it now
      if (pendingUrlRef.current) {
        ws.load(pendingUrlRef.current);
        pendingUrlRef.current = null;
      }

      return () => {
        ws.destroy();
        wsRef.current = null;
        regionsRef.current = null;
      };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Sync volume changes
    useEffect(() => {
      wsRef.current?.setVolume(volume);
    }, [volume]);

    // Sync playback rate changes
    useEffect(() => {
      wsRef.current?.setPlaybackRate(playbackRate);
    }, [playbackRate]);

    // Imperative API
    const loadUrl = useCallback((url: string) => {
      if (wsRef.current) {
        regionsRef.current?.clearRegions();
        wsRef.current.load(url);
      } else {
        pendingUrlRef.current = url;
      }
    }, []);

    useImperativeHandle(
      ref,
      () => ({
        play: () => wsRef.current?.play(),
        pause: () => wsRef.current?.pause(),
        playPause: () => wsRef.current?.playPause(),
        seekTo: (fraction: number) => wsRef.current?.seekTo(fraction),
        setVolume: (v: number) => wsRef.current?.setVolume(v),
        setPlaybackRate: (r: number) => wsRef.current?.setPlaybackRate(r),
        loadUrl,
        getCurrentTime: () => wsRef.current?.getCurrentTime() ?? 0,
        getDuration: () => wsRef.current?.getDuration() ?? 0,
        getMediaElement: () => wsRef.current?.getMediaElement() ?? null,
        clearRegions: () => regionsRef.current?.clearRegions(),
        addMarker: (time: number, label: string, color?: string) => {
          regionsRef.current?.addRegion({
            start: time,
            content: label,
            color: color ?? 'rgba(168, 85, 247, 0.4)',
            drag: false,
            resize: false,
          });
        },
      }),
      [loadUrl]
    );

    return (
      <div
        ref={containerRef}
        className="w-full cursor-pointer"
        style={{ height: 56 }}
      />
    );
  }
);

WaveformPlayer.displayName = 'WaveformPlayer';
