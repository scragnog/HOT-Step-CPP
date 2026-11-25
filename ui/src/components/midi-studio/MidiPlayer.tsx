import { useQuery, useQueryClient } from '@tanstack/react-query';
import { midiNotesOptions, midiJobsOptions } from '@/queries/studios/midi';
import { queryManager } from '@/queries/invalidation';
// MidiPlayer.tsx — live piano roll + synced playback for MIDI Studio
//
// Plays the ORIGINAL track (audio element) and the transcribed MIDI (WebAudio
// synth) in sync, with an equal-power crossfade slider between them — hear
// either or both. In live mode it consumes the job's SSE event stream, so
// notes appear on the roll (and become playable) while transcription is still
// running; for finished jobs it loads notes.json and offers the same player.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Pause, Play, Radio } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  getMidiStreamUrl, channelLabel,
} from '@/services/midiStudioApi';
import { MidiSynth, type PlayNote } from './midiSynth';
import { QueryError } from '@/components/shared/feedback/QueryError';
import { formatClock } from '@/utils/format/format';

const PPS = 40;          // px per second
const ROLL_H = 230;
const PITCH_MIN = 21, PITCH_MAX = 108;

function familyColor(family: string, alpha = 1): string {
  if (/drum/i.test(family)) return `hsla(0, 70%, 55%, ${alpha})`;
  let h = 0;
  for (let i = 0; i < family.length; i++) h = (h * 31 + family.charCodeAt(i)) % 360;
  return `hsla(${h}, 70%, 55%, ${alpha})`;
}

interface Props {
  jobId: string;
  sourceAudioUrl?: string;
  live: boolean;
}

export const MidiPlayer: React.FC<Props> = ({ jobId, sourceAudioUrl, live }) => {
  const { t } = useTranslation();
  const client = useQueryClient();
  const notesQuery = useQuery({ ...midiNotesOptions(jobId), enabled: !live });

  // NOTE: parent must mount this component with a key unique per (job, mode)
  // — notes accumulate for the component's lifetime, no in-place reset.
  const notesRef = useRef<PlayNote[]>([]);
  const [noteCount, setNoteCount] = useState(0);      // mirrors notesRef length -> redraw
  const [redraw, setRedraw] = useState(0);            // bump from event handlers
  const [isPlaying, setIsPlaying] = useState(false);
  const [crossfade, setCrossfade] = useState(50);
  const [duration, setDuration] = useState(0);
  const [curTime, setCurTime] = useState(0);
  const [frontier, setFrontier] = useState<number | null>(live ? 0 : null);
  const [chunks, setChunks] = useState<{ done: number; total: number; phase?: string } | null>(null);
  const [liveDone, setLiveDone] = useState(!live);
  const [families, setFamilies] = useState<string[]>([]);
  const [muted, setMuted] = useState<Set<string>>(new Set());
  const [soloed, setSoloed] = useState<Set<string>>(new Set());

  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const synthRef = useRef<MidiSynth | null>(null);
  const lastManualScroll = useRef(0);
  const rafRef = useRef(0);

  // family audible = no solos active ? not muted : soloed
  const applyMixer = useCallback((synth: MidiSynth, fams: string[], mutedS: Set<string>, soloS: Set<string>) => {
    for (const f of fams) {
      synth.setFamilyAudible(f, soloS.size > 0 ? soloS.has(f) : !mutedS.has(f));
    }
  }, []);

  const getSynth = useCallback((): MidiSynth => {
    if (!synthRef.current) {
      const synth = new MidiSynth();
      // seed with everything that streamed in before first play — the synth
      // is created lazily on the first user gesture, so notes accumulated in
      // notesRef must be handed over here (this was the "MIDI side silent" bug)
      synth.setAllNotes(notesRef.current);
      synthRef.current = synth;
    }
    return synthRef.current;
  }, []);

  // keep the synth's family buses in sync with mute/solo state
  useEffect(() => {
    if (synthRef.current) applyMixer(synthRef.current, families, muted, soloed);
  }, [families, muted, soloed, applyMixer]);

  const addNotes = useCallback((ns: PlayNote[]) => {
    if (!ns.length) return;
    notesRef.current.push(...ns);
    synthRef.current?.addNotes(ns);
    setFamilies(prev => {
      const s = new Set(prev);
      let changed = false;
      for (const n of ns) if (!s.has(n.family)) { s.add(n.family); changed = true; }
      return changed ? [...s] : prev;
    });
    setNoteCount(notesRef.current.length);
  }, []);

  // ── data source: SSE (live) or notes.json (finished) ──
  useEffect(() => {
    if (live) {
      const open = new Map<number, { pitch: number; time: number; instrument: string }>();
      const es = new EventSource(getMidiStreamUrl(jobId));
      const batch: PlayNote[] = [];
      const flush = () => { if (batch.length) { addNotes(batch.splice(0)); } };
      const flushTimer = window.setInterval(flush, 200);
      es.onmessage = (msg) => {
        try {
          const ev = JSON.parse(msg.data);
          if (ev.type === 'note_start') {
            open.set(ev.index, { pitch: ev.pitch, time: ev.time, instrument: ev.instrument });
          } else if (ev.type === 'note_end') {
            const st = open.get(ev.index);
            if (st) {
              open.delete(ev.index);
              batch.push({ pitch: st.pitch, start: st.time, duration: Math.max(0.03, ev.time - st.time), family: st.instrument });
            }
          } else if (ev.type === 'progress') {
            setChunks({ done: ev.completed, total: ev.total, phase: ev.phase });
            // dettato reports audio seconds; MuScriptor's chunks are 5 s each
            setFrontier(typeof ev.seconds === 'number' ? ev.seconds : ev.completed * 5.0);
          } else if (ev.type === 'status') {
            void queryManager.invalidate(client, midiJobsOptions().queryKey);
            void queryManager.invalidate(client, midiNotesOptions(jobId).queryKey);
            flush();
            setLiveDone(true);
            setFrontier(null);
            es.close();
          }
        } catch { /* ignore malformed */ }
      };
      es.onopen = () => { void queryManager.invalidate(client, midiJobsOptions().queryKey); };
      es.onerror = () => { void queryManager.invalidate(client, midiJobsOptions().queryKey); };
      return () => { window.clearInterval(flushTimer); es.close(); };
    }
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, live]);

  // ── audio element wiring (original track = master clock) ──
  useEffect(() => {
    const parsed = notesQuery.data;
    if (live || !parsed) return;
    const label = new Map(parsed.channels.map(ch => [ch.channel, channelLabel(ch)]));
    const notes = parsed.notes.map(n => ({ pitch: n.pitch, start: n.start, duration: n.duration,
      family: label.get(n.channel) || `ch${n.channel}` }));
    notesRef.current = [];
    synthRef.current?.setAllNotes([]);
    setFamilies([]);
    addNotes(notes);
    setDuration(d => Math.max(d, parsed.durationSec));
  }, [notesQuery.data, live, addNotes]);

  useEffect(() => {
    const el = audioRef.current;
    if (!el || !sourceAudioUrl) return;
    const onMeta = () => setDuration(d => Math.max(d, el.duration || 0));
    const onEnd = () => setIsPlaying(false);
    el.addEventListener('loadedmetadata', onMeta);
    el.addEventListener('ended', onEnd);
    return () => { el.removeEventListener('loadedmetadata', onMeta); el.removeEventListener('ended', onEnd); };
  }, [sourceAudioUrl]);

  useEffect(() => () => { synthRef.current?.dispose(); synthRef.current = null; }, []);

  const togglePlay = useCallback(async () => {
    const el = audioRef.current;
    if (!el || !sourceAudioUrl) return;
    const synth = getSynth();
    synth.attachAudio(el);
    synth.setCrossfade(crossfade / 100);
    applyMixer(synth, families, muted, soloed);
    if (synth.playing) {
      synth.pause();
      setIsPlaying(false);
    } else {
      await synth.play();
      setIsPlaying(true);
    }
  }, [sourceAudioUrl, crossfade, getSynth, applyMixer, families, muted, soloed]);

  const toggleMute = useCallback((f: string) => {
    setMuted(prev => {
      const s = new Set(prev);
      if (s.has(f)) s.delete(f); else s.add(f);
      return s;
    });
  }, []);
  const toggleSolo = useCallback((f: string) => {
    setSoloed(prev => {
      const s = new Set(prev);
      if (s.has(f)) s.delete(f); else s.add(f);
      return s;
    });
  }, []);

  const handleCrossfade = useCallback((v: number) => {
    setCrossfade(v);
    synthRef.current?.setCrossfade(v / 100);
  }, []);

  const seekTo = useCallback((time: number) => {
    const el = audioRef.current;
    if (!el) return;
    if (synthRef.current) synthRef.current.seek(time);
    else el.currentTime = time;
    setCurTime(time);
    setRedraw(r => r + 1);
  }, []);

  // ── canvas drawing + playhead follow ──
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = canvas.width, h = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dark = document.documentElement.classList.contains('dark');
    ctx.clearRect(0, 0, w, h);

    const rowH = h / (PITCH_MAX - PITCH_MIN + 1);
    ctx.fillStyle = dark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.35)';
    ctx.font = '9px sans-serif';
    const grid = notesQuery.data?.grid;
    if (grid) {
      // the MIDI's own beats and bar lines, numbered as a DAW shows them
      ctx.strokeStyle = dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
      for (const b of grid.beats) {
        ctx.beginPath(); ctx.moveTo(b * PPS, 0); ctx.lineTo(b * PPS, h); ctx.stroke();
      }
      ctx.strokeStyle = dark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)';
      const firstBar = grid.bars.findIndex(b => b >= grid.audioShift - 1e-3);
      grid.bars.forEach((b, i) => {
        ctx.beginPath(); ctx.moveTo(b * PPS, 0); ctx.lineTo(b * PPS, h); ctx.stroke();
        if (firstBar >= 0 && i >= firstBar) ctx.fillText(String(i - firstBar + 1), b * PPS + 3, 10);
      });
    } else {
      // 10 s gridlines
      ctx.strokeStyle = dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
      for (let s = 0; s * PPS < w; s += 10) {
        const x = s * PPS;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        ctx.fillText(`${s}s`, x + 3, 10);
      }
    }

    for (const n of notesRef.current) {
      const p = Math.min(PITCH_MAX, Math.max(PITCH_MIN, n.pitch));
      ctx.fillStyle = familyColor(n.family, 0.85);
      ctx.fillRect(n.start * PPS, (PITCH_MAX - p) * rowH, Math.max(1.5, n.duration * PPS), Math.max(1.5, rowH - 0.5));
    }

    // un-transcribed region (live)
    if (frontier !== null && duration > 0) {
      const x = frontier * PPS;
      ctx.fillStyle = dark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
      ctx.fillRect(x, 0, w - x, h);
    }

    // playhead
    const t = synthRef.current?.currentTime ?? curTime;
    ctx.strokeStyle = 'rgba(236,72,153,0.9)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(t * PPS, 0); ctx.lineTo(t * PPS, h); ctx.stroke();
  }, [frontier, duration, curTime, notesQuery.data]);

  // resize canvas to content and redraw on data changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const wantW = Math.max(600, Math.ceil(Math.max(duration, frontier ?? 0, 10) * PPS) + 40);
    if (canvas.width !== wantW) { canvas.width = wantW; canvas.height = ROLL_H; }
    draw();
  }, [noteCount, redraw, duration, frontier, draw]);

  // animation loop while playing: playhead + auto-follow
  useEffect(() => {
    if (!isPlaying) { draw(); return; }
    const loop = () => {
      const tNow = synthRef.current?.currentTime ?? 0;
      setCurTime(tNow);
      draw();
      const sc = scrollRef.current;
      if (sc && Date.now() - lastManualScroll.current > 2500) {
        const target = tNow * PPS - sc.clientWidth * 0.4;
        sc.scrollLeft = Math.max(0, target);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [isPlaying, draw]);

  const fmt = (s: number) => formatClock(s);

  return (
    <div className="mt-3">
      {notesQuery.isError && (
        <QueryError error={notesQuery.error} onRetry={() => void notesQuery.refetch()} className="mb-2" />
      )}
      {sourceAudioUrl && <audio ref={audioRef} src={sourceAudioUrl} preload="metadata" />}

      {/* transport */}
      <div className="flex items-center gap-3 gap-y-2 flex-wrap mb-2">
        <button
          onClick={togglePlay}
          disabled={!sourceAudioUrl}
          className="w-9 h-9 rounded-full bg-purple-600 hover:bg-purple-500 text-white flex items-center justify-center disabled:opacity-40 transition-colors"
          title={isPlaying ? t('midiStudio.player.pause') : t('midiStudio.player.play')}
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
        </button>
        <span className="text-[11px] tabular-nums text-zinc-500 dark:text-zinc-400 w-[86px]">
          {fmt(curTime)} / {fmt(duration)}
        </span>

        {/* crossfade: original <-> MIDI */}
        <div className="flex items-center gap-2">
          <span className={`text-[11px] ${crossfade < 50 ? 'font-semibold text-zinc-800 dark:text-zinc-200' : 'text-zinc-400 dark:text-zinc-500'}`}>
            {t('midiStudio.player.original')}
          </span>
          <input
            type="range" min={0} max={100} value={crossfade}
            onChange={e => handleCrossfade(Number(e.target.value))}
            className="w-36 accent-purple-500"
            title={t('midiStudio.player.crossfade')}
          />
          <span className={`text-[11px] ${crossfade > 50 ? 'font-semibold text-zinc-800 dark:text-zinc-200' : 'text-zinc-400 dark:text-zinc-500'}`}>
            MIDI
          </span>
        </div>

        {live && !liveDone && chunks && (
          <span className="flex items-center gap-1.5 text-[11px] text-purple-500 dark:text-purple-400">
            <Radio size={11} className="animate-pulse" />
            {chunks.phase === 'drums' ? t('midiStudio.statusDrums') : chunks.phase === 'group' ? t('midiStudio.statusGrouping')
              : t('midiStudio.player.liveChunk', { done: chunks.done, total: chunks.total })}
          </span>
        )}
        <span className="text-[11px] text-zinc-400 dark:text-zinc-500">
          {t('midiStudio.player.noteCount', { count: noteCount })}
        </span>
      </div>

      {/* legend + per-track mute/solo (affects the MIDI side only) */}
      {families.length > 0 && (
        <div className="flex flex-wrap gap-x-2 gap-y-1 mb-1.5">
          {families.map(f => {
            const isMuted = muted.has(f);
            const isSolo = soloed.has(f);
            const audible = soloed.size > 0 ? isSolo : !isMuted;
            return (
              <span
                key={f}
                className={`flex items-center gap-1.5 text-[11px] rounded-md px-1.5 py-0.5 border transition-colors ${
                  isSolo
                    ? 'border-purple-500/60 bg-purple-500/10 text-zinc-800 dark:text-zinc-200'
                    : 'border-transparent text-zinc-600 dark:text-zinc-400'
                } ${audible ? '' : 'opacity-45'}`}
              >
                <span className="w-2.5 h-2.5 rounded-sm inline-block flex-shrink-0" style={{ backgroundColor: familyColor(f) }} />
                <span className={isMuted && soloed.size === 0 ? 'line-through' : ''}>{f.replace(/_/g, ' ')}</span>
                <button
                  onClick={() => toggleMute(f)}
                  title={t('midiStudio.player.mute')}
                  className={`w-4 h-4 rounded text-[9px] font-bold leading-none flex items-center justify-center transition-colors ${
                    isMuted ? 'bg-red-500 text-white' : 'bg-zinc-200 dark:bg-white/10 text-zinc-500 hover:bg-zinc-300 dark:hover:bg-white/20'
                  }`}
                >M</button>
                <button
                  onClick={() => toggleSolo(f)}
                  title={t('midiStudio.player.solo')}
                  className={`w-4 h-4 rounded text-[9px] font-bold leading-none flex items-center justify-center transition-colors ${
                    isSolo ? 'bg-amber-500 text-white' : 'bg-zinc-200 dark:bg-white/10 text-zinc-500 hover:bg-zinc-300 dark:hover:bg-white/20'
                  }`}
                >S</button>
              </span>
            );
          })}
        </div>
      )}

      {/* piano roll (click to seek) */}
      <div
        ref={scrollRef}
        onScroll={() => { lastManualScroll.current = Date.now(); }}
        className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/30"
      >
        <canvas
          ref={canvasRef}
          height={ROLL_H}
          className="block cursor-pointer"
          style={{ height: ROLL_H }}
          onClick={e => {
            const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
            seekTo((e.clientX - rect.left) / PPS);
          }}
        />
      </div>
    </div>
  );
};

export default MidiPlayer;
