/**
 * useAbEngine.ts - Web Audio A/B engine for the Compare studio.
 *
 * One AudioContext (created/resumed on a user gesture for mobile Safari).
 * Every decoded take plays from its own BufferSource started at the same
 * context time and offset, so switching keeps one shared playhead. Each take
 * has a GainNode; switching ramps gains over ~10 ms. At most MAX_DECODED
 * buffers are kept (LRU eviction, the active take is never evicted).
 */

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { dbToLinear, levelMatchGainsDb } from './levelMatch';
import { pauseGlobalPlayback, registerExternalPlayer } from '@/stores/playbackStore';
import { errorMessage } from '@/utils/format/errorMessage';
import { useGlobalEvent } from '@/hooks/useGlobalEvent';

export interface AbTake {
  id: string;
  url: string;
  /** Integrated loudness from analysis, used for level match. */
  integratedLufs?: number | null;
}

export interface LoopRegion { start: number; end: number; }

export interface AbEngine {
  /** Call from a click/tap handler before play on mobile. */
  unlock: () => Promise<void>;
  play: () => Promise<void>;
  pause: () => void;
  toggle: () => Promise<void>;
  seek: (sec: number) => void;
  setActive: (id: string) => void;
  setLoop: (loop: LoopRegion | null) => void;
  setLevelMatch: (on: boolean) => void;
  /** rAF-friendly playhead: subscribe + getter, no React re-render per frame. */
  subscribeTime: (cb: (t: number) => void) => () => void;
  getCurrentTime: () => number;
  useCurrentTime: () => number;
  activeId: string | null;
  isPlaying: boolean;
  loop: LoopRegion | null;
  levelMatch: boolean;
  duration: number;
  decoded: string[];
  loading: string[];
  errors: Record<string, string>;
  /** Linear gain applied per take (level match), for display. */
  gainsDb: Record<string, number>;
}

export const MAX_DECODED = 4;
const RAMP_SEC = 0.01;
const START_LEAD_SEC = 0.03;

interface Voice { src: AudioBufferSourceNode; gain: GainNode; }

/**
 * Loop plan for one take. Every take must loop on the same cycle
 * (loop.end - loop.start), so loopEnd is never clamped to the take length.
 * A take shorter than loop.end is padded with silence up to padTo, so it
 * goes quiet until the shared loop restarts instead of running its own cycle.
 */
export function voiceLoopPlan(loop: LoopRegion, bufDuration: number): { loopStart: number; loopEnd: number; padTo: number } {
  return { loopStart: loop.start, loopEnd: loop.end, padTo: Math.max(bufDuration, loop.end) };
}

/** Position of a native looping source after `elapsed` seconds from `offset`. */
export function loopedSourcePosition(offset: number, elapsed: number, loopStart: number, loopEnd: number): number {
  const pos = offset + elapsed;
  if (loopEnd <= loopStart || pos < loopEnd) return pos;
  return loopStart + ((pos - loopStart) % (loopEnd - loopStart));
}

/** Map an elapsed play time to a position, wrapping inside the loop region. */
export function wrapPosition(pos: number, loop: LoopRegion | null): number {
  if (!loop || loop.end <= loop.start) return pos;
  if (pos < loop.end) return pos;
  const len = loop.end - loop.start;
  return loop.start + ((pos - loop.start) % len);
}

export function useAbEngine(takes: AbTake[]): AbEngine {
  const ctxRef = useRef<AudioContext | null>(null);
  const masterRef = useRef<GainNode | null>(null);
  const buffers = useRef(new Map<string, AudioBuffer>());
  const lru = useRef<string[]>([]);
  const pending = useRef(new Map<string, Promise<AudioBuffer | null>>());
  const voices = useRef(new Map<string, Voice>());
  const takeGains = useRef(new Map<string, GainNode>());

  // Transport: position = startOffset + (ctx.currentTime - startCtxTime), wrapped.
  const startCtxTime = useRef(0);
  const startOffset = useRef(0);
  const pausedAt = useRef(0);
  const playingRef = useRef(false);
  const activeRef = useRef<string | null>(null);
  const loopRef = useRef<LoopRegion | null>(null);
  const levelMatchRef = useRef(false);
  const takesRef = useRef(takes);
  takesRef.current = takes;

  const [activeId, setActiveId] = useState<string | null>(takes[0]?.id ?? null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loop, setLoopState] = useState<LoopRegion | null>(null);
  const [levelMatch, setLevelMatchState] = useState(false);
  const [decoded, setDecoded] = useState<string[]>([]);
  const [loading, setLoading] = useState<string[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [duration, setDuration] = useState(0);

  if (activeRef.current === null && takes[0]) activeRef.current = takes[0].id;
  useEffect(() => {
    if (activeId && !takes.some(t => t.id === activeId)) {
      const next = takes[0]?.id ?? null;
      activeRef.current = next;
      setActiveId(next);
    } else if (!activeId && takes[0]) {
      setActiveId(takes[0].id);
    }
  }, [takes, activeId]);

  const gainsDb = useMemo(() => {
    const m: Record<string, number | null | undefined> = {};
    for (const t of takes) m[t.id] = t.integratedLufs;
    return levelMatchGainsDb(m);
  }, [takes]);
  const gainsDbRef = useRef(gainsDb);
  gainsDbRef.current = gainsDb;

  // ── Time subscription ───────────────────────────────────────────────────
  const timeListeners = useRef(new Set<(t: number) => void>());
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);

  const getCurrentTime = useCallback((): number => {
    const ctx = ctxRef.current;
    if (!playingRef.current || !ctx) return pausedAt.current;
    const pos = startOffset.current + (ctx.currentTime - startCtxTime.current);
    return Math.max(0, wrapPosition(pos, loopRef.current));
  }, []);

  const emitTime = useCallback(() => {
    const t = getCurrentTime();
    lastTimeRef.current = t;
    timeListeners.current.forEach(cb => cb(t));
  }, [getCurrentTime]);

  const tick = useCallback(() => {
    emitTime();
    const ctx = ctxRef.current;
    // Stop at end when not looping.
    if (playingRef.current && ctx && !loopRef.current) {
      let maxDur = 0;
      buffers.current.forEach(b => { maxDur = Math.max(maxDur, b.duration); });
      if (maxDur > 0 && getCurrentTime() >= maxDur) {
        stopVoices();
        playingRef.current = false;
        pausedAt.current = 0;
        setIsPlaying(false);
        emitTime();
      }
    }
    rafRef.current = playingRef.current ? requestAnimationFrame(tick) : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [emitTime, getCurrentTime]);

  const startRaf = useCallback(() => {
    if (rafRef.current === null) rafRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const subscribeTime = useCallback((cb: (t: number) => void) => {
    timeListeners.current.add(cb);
    cb(getCurrentTime());
    return () => { timeListeners.current.delete(cb); };
  }, [getCurrentTime]);

  // ── Context and gains ───────────────────────────────────────────────────
  const ensureCtx = useCallback((): AudioContext => {
    if (!ctxRef.current) {
      const Ctor: typeof AudioContext = window.AudioContext
        ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new Ctor();
      const master = ctx.createGain();
      master.connect(ctx.destination);
      ctxRef.current = ctx;
      masterRef.current = master;
    }
    return ctxRef.current;
  }, []);

  const unlock = useCallback(async () => {
    const ctx = ensureCtx();
    if (ctx.state !== 'running') {
      try { await ctx.resume(); } catch { /* resumes on next gesture */ }
    }
  }, [ensureCtx]);

  const targetGain = useCallback((id: string): number => {
    if (id !== activeRef.current) return 0;
    return levelMatchRef.current ? dbToLinear(gainsDbRef.current[id] ?? 0) : 1;
  }, []);

  const takeGain = useCallback((id: string): GainNode => {
    const ctx = ensureCtx();
    let g = takeGains.current.get(id);
    if (!g) {
      g = ctx.createGain();
      g.gain.value = targetGain(id);
      g.connect(masterRef.current!);
      takeGains.current.set(id, g);
    }
    return g;
  }, [ensureCtx, targetGain]);

  const rampGains = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const now = ctx.currentTime;
    takeGains.current.forEach((g, id) => {
      g.gain.cancelScheduledValues(now);
      g.gain.setValueAtTime(g.gain.value, now);
      g.gain.linearRampToValueAtTime(targetGain(id), now + RAMP_SEC);
    });
  }, [targetGain]);

  // ── Voices ──────────────────────────────────────────────────────────────
  function stopVoice(id: string) {
    const v = voices.current.get(id);
    if (!v) return;
    try { v.src.onended = null; v.src.stop(); } catch { /* already stopped */ }
    v.src.disconnect();
    voices.current.delete(id);
  }
  function stopVoices() {
    for (const id of [...voices.current.keys()]) stopVoice(id);
  }

  /** Start one take's source at context time `when` and buffer position `offset`. */
  const startVoice = useCallback((id: string, when: number, offset: number) => {
    const ctx = ctxRef.current;
    const buf = buffers.current.get(id);
    if (!ctx || !buf) return;
    stopVoice(id);
    const src = ctx.createBufferSource();
    let playBuf = buf;
    const lp = loopRef.current;
    if (lp && lp.end > lp.start) {
      const plan = voiceLoopPlan(lp, buf.duration);
      if (plan.padTo > buf.duration) {
        // Pad with silence so this take shares the master loop cycle.
        const len = Math.ceil(plan.padTo * buf.sampleRate) + 1;
        playBuf = ctx.createBuffer(buf.numberOfChannels, len, buf.sampleRate);
        for (let c = 0; c < buf.numberOfChannels; c++) playBuf.copyToChannel(buf.getChannelData(c), c);
      }
      src.loop = true;
      src.loopStart = plan.loopStart;
      src.loopEnd = plan.loopEnd;
    }
    src.buffer = playBuf;
    const gain = takeGain(id);
    src.connect(gain);
    if (offset < playBuf.duration) src.start(when, offset);
    voices.current.set(id, { src, gain });
  }, [takeGain]);

  /** (Re)start all decoded takes at a shared future context time. */
  const startAll = useCallback((offset: number) => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    stopVoices();
    const when = ctx.currentTime + START_LEAD_SEC;
    startCtxTime.current = when;
    startOffset.current = offset;
    buffers.current.forEach((_b, id) => startVoice(id, when, offset));
  }, [startVoice]);

  /** Join a newly decoded take into running playback at the shared playhead. */
  const joinVoice = useCallback((id: string) => {
    const ctx = ctxRef.current;
    if (!ctx || !playingRef.current) return;
    const when = ctx.currentTime + START_LEAD_SEC;
    const pos = startOffset.current + (when - startCtxTime.current);
    startVoice(id, when, wrapPosition(pos, loopRef.current));
  }, [startVoice]);

  // ── Decode with LRU ─────────────────────────────────────────────────────
  const touch = useCallback((id: string) => {
    lru.current = [id, ...lru.current.filter(x => x !== id)];
  }, []);

  const evict = useCallback(() => {
    while (buffers.current.size > MAX_DECODED) {
      const victim = [...lru.current].reverse().find(id => id !== activeRef.current && buffers.current.has(id));
      if (!victim) break;
      stopVoice(victim);
      buffers.current.delete(victim);
      lru.current = lru.current.filter(x => x !== victim);
    }
    setDecoded([...buffers.current.keys()]);
  }, []);

  const recomputeDuration = useCallback(() => {
    let d = 0;
    buffers.current.forEach(b => { d = Math.max(d, b.duration); });
    setDuration(d);
  }, []);

  const decode = useCallback((id: string): Promise<AudioBuffer | null> => {
    const have = buffers.current.get(id);
    if (have) { touch(id); return Promise.resolve(have); }
    const inflight = pending.current.get(id);
    if (inflight) return inflight;
    const take = takesRef.current.find(t => t.id === id);
    if (!take) return Promise.resolve(null);
    const ctx = ensureCtx();
    setLoading(l => [...l, id]);
    const p = fetch(take.url)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); })
      .then(ab => new Promise<AudioBuffer>((res, rej) => { ctx.decodeAudioData(ab, res, rej); }))
      .then(buf => {
        if (!takesRef.current.some(t => t.id === id)) return null;
        buffers.current.set(id, buf);
        touch(id);
        evict();
        recomputeDuration();
        if (buffers.current.has(id)) joinVoice(id);
        return buf;
      })
      .catch((e: unknown) => {
        setErrors(er => ({ ...er, [id]: errorMessage(e) }));
        return null;
      })
      .finally(() => {
        pending.current.delete(id);
        setLoading(l => l.filter(x => x !== id));
      });
    pending.current.set(id, p);
    return p;
  }, [ensureCtx, touch, evict, recomputeDuration, joinVoice]);

  // Preload the first few takes (decode needs a context; creating one is fine,
  // it just stays suspended until a gesture resumes it).
  useEffect(() => {
    for (const t of takes.slice(0, MAX_DECODED)) void decode(t.id);
    // Drop buffers for takes that are gone.
    for (const id of [...buffers.current.keys()]) {
      if (!takes.some(t => t.id === id)) {
        stopVoice(id);
        buffers.current.delete(id);
        takeGains.current.get(id)?.disconnect();
        takeGains.current.delete(id);
        lru.current = lru.current.filter(x => x !== id);
      }
    }
    setDecoded([...buffers.current.keys()]);
    recomputeDuration();
  }, [takes, decode, recomputeDuration]);

  // ── Transport ───────────────────────────────────────────────────────────
  const play = useCallback(async () => {
    await unlock();
    if (playingRef.current) return;
    // Only one player audible: silence the global player bar first.
    pauseGlobalPlayback();
    if (activeRef.current) await decode(activeRef.current);
    startAll(pausedAt.current);
    playingRef.current = true;
    setIsPlaying(true);
    startRaf();
  }, [unlock, decode, startAll, startRaf]);

  const pause = useCallback(() => {
    if (!playingRef.current) return;
    pausedAt.current = getCurrentTime();
    stopVoices();
    playingRef.current = false;
    setIsPlaying(false);
    emitTime();
  }, [getCurrentTime, emitTime]);

  const toggle = useCallback(async () => {
    if (playingRef.current) pause(); else await play();
  }, [play, pause]);

  const seek = useCallback((sec: number) => {
    const t = Math.max(0, sec);
    if (playingRef.current) startAll(t); else pausedAt.current = t;
    emitTime();
  }, [startAll, emitTime]);

  const setActive = useCallback((id: string) => {
    if (id === activeRef.current) return;
    activeRef.current = id;
    setActiveId(id);
    touch(id);
    rampGains();
    if (!buffers.current.has(id)) void decode(id);
  }, [touch, rampGains, decode]);

  const setLoop = useCallback((lp: LoopRegion | null) => {
    const valid = lp && lp.end > lp.start ? lp : null;
    const pos = getCurrentTime();
    loopRef.current = valid;
    setLoopState(valid);
    let target = pos;
    if (valid && (pos < valid.start || pos >= valid.end)) target = valid.start;
    if (playingRef.current) startAll(target); else pausedAt.current = target;
    emitTime();
  }, [getCurrentTime, startAll, emitTime]);

  const setLevelMatch = useCallback((on: boolean) => {
    levelMatchRef.current = on;
    setLevelMatchState(on);
    rampGains();
  }, [rampGains]);

  // Global playback starting pauses this engine (and vice versa in play()).
  useEffect(() => registerExternalPlayer(() => { if (playingRef.current) pause(); }), [pause]);

  // Re-apply gains when analysis loudness arrives.
  useEffect(() => { rampGains(); }, [gainsDb, rampGains]);

  // Tab background: the browser may suspend the context. Pause cleanly so the
  // playhead stays correct, and resume the context on return.
  useGlobalEvent('visibilitychange', () => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (document.hidden) {
      if (playingRef.current) pause();
    } else if (ctx.state === 'suspended') {
      void ctx.resume().catch(() => { /* needs a gesture */ });
    }
  }, { target: 'document' });

  // Cleanup on unmount.
  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    playingRef.current = false;
    stopVoices();
    takeGains.current.forEach(g => g.disconnect());
    takeGains.current.clear();
    buffers.current.clear();
    timeListeners.current.clear();
    const ctx = ctxRef.current;
    ctxRef.current = null;
    masterRef.current = null;
    if (ctx) void ctx.close().catch(() => { /* already closed */ });
  }, []);

  // Stable store subscribe/snapshot so useSyncExternalStore never resubscribes per render.
  const storeSubscribe = useCallback(
    (cb: () => void) => subscribeTime(() => cb()),
    [subscribeTime],
  );
  const storeSnapshot = useCallback(() => lastTimeRef.current, []);
  const useCurrentTime = useCallback(
    // eslint-disable-next-line react-hooks/rules-of-hooks
    () => useSyncExternalStore(storeSubscribe, storeSnapshot),
    [storeSubscribe, storeSnapshot],
  );

  return {
    unlock, play, pause, toggle, seek, setActive, setLoop, setLevelMatch,
    subscribeTime, getCurrentTime, useCurrentTime,
    activeId, isPlaying, loop, levelMatch, duration, decoded, loading, errors, gainsDb,
  };
}
