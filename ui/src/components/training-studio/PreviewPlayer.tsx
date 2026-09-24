// PreviewPlayer.tsx — a scrub-first audio player for listening tests.
//
// The stock <audio> control's seek bar is a few pixels tall and 200 px wide,
// useless for hunting late-song decay. This one is a full-width, tall seek
// bar with the last third shaded (where planner decay shows), a time readout,
// and jump buttons for the passages that matter.
import React, { useEffect, useRef, useState } from 'react';
import { Pause, Play, SkipBack } from 'lucide-react';

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export const PreviewPlayer: React.FC<{ src: string; label?: string; sublabel?: string }> = ({ src, label, sublabel }) => {
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  useEffect(() => {
    const a = audio.current; if (!a) return;
    const onTime = () => setTime(a.currentTime);
    const onMeta = () => setDuration(Number.isFinite(a.duration) ? a.duration : 0);
    const onPlay = () => setPlaying(true), onPause = () => setPlaying(false);
    a.addEventListener('timeupdate', onTime); a.addEventListener('loadedmetadata', onMeta); a.addEventListener('durationchange', onMeta);
    a.addEventListener('play', onPlay); a.addEventListener('pause', onPause); a.addEventListener('ended', onPause);
    return () => { a.removeEventListener('timeupdate', onTime); a.removeEventListener('loadedmetadata', onMeta); a.removeEventListener('durationchange', onMeta); a.removeEventListener('play', onPlay); a.removeEventListener('pause', onPause); a.removeEventListener('ended', onPause); };
  }, [src]);
  const seek = (t: number) => { const a = audio.current; if (!a) return; a.currentTime = Math.max(0, Math.min(duration || 0, t)); setTime(a.currentTime); };
  const toggle = () => { const a = audio.current; if (!a) return; if (a.paused) void a.play(); else a.pause(); };
  const frac = duration > 0 ? time / duration : 0;
  const jumps = duration > 0 ? [0.5, 0.67, 0.85].map(f => ({ f, t: duration * f })) : [];
  return (
    <div className="rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/60 dark:bg-black/20 px-3 py-2">
      <audio ref={audio} src={src} preload="metadata" />
      <div className="flex items-center gap-3">
        <button type="button" onClick={toggle} className="w-9 h-9 shrink-0 rounded-full bg-amber-500 text-black flex items-center justify-center hover:bg-amber-400" aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
        </button>
        <div className="flex-1 min-w-0">
          {(label || sublabel) && <div className="flex items-baseline gap-2 text-[11px] mb-1"><span className="font-semibold text-zinc-800 dark:text-zinc-100">{label}</span><span className="text-zinc-500 truncate">{sublabel}</span></div>}
          <div className="relative h-5 rounded-md bg-zinc-200 dark:bg-zinc-800 overflow-hidden cursor-pointer select-none"
            onPointerDown={e => { const r = e.currentTarget.getBoundingClientRect(); seek(((e.clientX - r.left) / r.width) * duration); e.currentTarget.setPointerCapture(e.pointerId); }}
            onPointerMove={e => { if (e.buttons & 1) { const r = e.currentTarget.getBoundingClientRect(); seek(((e.clientX - r.left) / r.width) * duration); } }}>
            {/* last third: where late-song decay lives */}
            <div className="absolute inset-y-0 right-0 bg-amber-500/15" style={{ width: '33.3%' }} />
            <div className="absolute inset-y-0 left-0 bg-amber-500/70" style={{ width: `${frac * 100}%` }} />
            <div className="absolute inset-y-0 w-0.5 bg-white" style={{ left: `calc(${frac * 100}% - 1px)` }} />
            {jumps.map(j => <div key={j.f} className="absolute inset-y-0 w-px bg-zinc-500/50" style={{ left: `${j.f * 100}%` }} />)}
          </div>
        </div>
        <div className="shrink-0 text-[11px] tabular-nums text-zinc-600 dark:text-zinc-300 w-[88px] text-right">{fmt(time)} / {fmt(duration)}</div>
      </div>
      <div className="mt-1.5 flex items-center gap-1.5 pl-12 text-[10px]">
        <button type="button" onClick={() => seek(0)} className="px-2 py-0.5 rounded border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 flex items-center gap-1"><SkipBack size={10} />start</button>
        <button type="button" onClick={() => seek(time - 15)} className="px-2 py-0.5 rounded border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10">−15 s</button>
        <button type="button" onClick={() => seek(time + 15)} className="px-2 py-0.5 rounded border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10">+15 s</button>
        {jumps.map(j => <button key={j.f} type="button" onClick={() => seek(j.t)} className="px-2 py-0.5 rounded border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10">{fmt(j.t)}</button>)}
        <span className="ml-1 text-zinc-500">shaded = last third</span>
      </div>
    </div>
  );
};

export default PreviewPlayer;
