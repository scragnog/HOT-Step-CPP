// Mm3StreamPlayer.tsx — "Play While Rendering" transport for MiniMax-Music3.
//
// Renders INSIDE the active queue card (InlineAudioQueue), not on the Create
// page: the sidebar that card lives in is mounted on every studio page, so the
// preview follows the generation instead of being stranded on the page it was
// started from. That is where the user is already watching the render.
//
// It drives mm3StreamStore, which is a module singleton — the engine allows one
// reader per job, so the audio machinery cannot be per-component state.
//
// Playback starts on its own (the toggle is called "Play While Rendering"; a
// user who turned it on wants audio, not a second button to find). The only
// button that appears unprompted is the one the browser forces: an AudioContext
// created without a user gesture starts suspended, and if resume() is refused
// the user has to click once.
//
// Deliberately NOT the StreamPlayer next door: that one belongs to ACE's
// shelved streamMode and speaks in DiT steps and preview chunks.

import React from 'react';
import { Headphones, Square, Volume2, Loader2, AlertTriangle, Play } from 'lucide-react';
import {
  useMm3StreamAudio, mm3StreamStop, mm3StreamStart, mm3StreamResume, mm3StreamSetVolume,
} from '../../stores/mm3StreamStore';

interface Mm3StreamPlayerProps {
  /** The job this card is for — the transport only renders for the job the
   *  store is actually streaming, so two active cards cannot both claim it. */
  jobId?: string;
  /** Engine renders windows while it plans (true), fell back to serial so audio
   *  starts after planning (false), or has not decided yet (null/undefined). */
  interleaved?: boolean | null;
}

const mmss = (s: number) => {
  const t = Math.max(0, Math.floor(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
};

export const Mm3StreamPlayer: React.FC<Mm3StreamPlayerProps> = ({ jobId, interleaved }) => {
  const s = useMm3StreamAudio();
  // Another job owns the stream — one reader per job, so this card has nothing
  // to show.
  if (jobId && s.jobId && s.jobId !== jobId) return null;

  // The store is not on this job: either the user stopped listening, or the
  // auto-open exhausted its retries. Offer the way back rather than pretending
  // to be waiting for a window that will never be requested.
  if (jobId && s.jobId !== jobId) {
    return (
      <div className="mt-2 pt-2 border-t border-white/5">
        <button
          onClick={() => mm3StreamStart(jobId)}
          className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-semibold
                     text-emerald-400 hover:text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20
                     border border-emerald-500/20 transition-colors"
        >
          <Headphones size={10} /> Listen while it renders
        </button>
      </div>
    );
  }

  const waiting = !s.error && s.chunks === 0;

  return (
    <div className="mt-2 pt-2 border-t border-white/5 space-y-1.5">
      <div className="flex items-center gap-2">
        <Headphones size={12} className="text-emerald-400 shrink-0" />

        {s.needsGesture ? (
          // The browser refused to start audio. One click fixes it, and saying
          // so beats a full buffer playing silently.
          <button
            onClick={mm3StreamResume}
            className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold
                       text-white bg-emerald-600 hover:bg-emerald-500 transition-colors"
          >
            <Play size={10} /> Tap to play
          </button>
        ) : waiting ? (
          <span className="flex items-center gap-1.5 text-[10px] text-zinc-500">
            <Loader2 size={10} className="animate-spin" />
            {interleaved === false ? 'audio starts once planning finishes…' : 'waiting for the first window…'}
          </span>
        ) : (
          <span className="text-[10px] tabular-nums text-zinc-400">
            {mmss(s.position)} / {mmss(s.received)} · buffer {s.ahead.toFixed(1)}s
          </span>
        )}

        <div className="flex-1" />

        {!waiting && !s.error && (
          <>
            <Volume2 size={11} className="text-zinc-500 shrink-0" />
            <input
              type="range" min={0} max={1} step={0.01} value={s.volume}
              onChange={e => mm3StreamSetVolume(Number(e.target.value))}
              className="w-12 accent-emerald-500"
            />
          </>
        )}
        {s.active && (
          <button
            onClick={mm3StreamStop}
            title="Stop listening (the render continues)"
            className="p-0.5 rounded text-zinc-500 hover:text-zinc-200 transition-colors"
          >
            <Square size={11} />
          </button>
        )}
      </div>

      {/* Buffer meter — a full bar is one headroom's worth of audio scheduled
          ahead of the playhead. Empty means the renderer has been caught. */}
      {!waiting && !s.error && (
        <div className="h-0.5 rounded-full bg-zinc-800 overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-[width] duration-200"
            style={{ width: `${Math.min(100, s.headroom > 0 ? (s.ahead / s.headroom) * 100 : (s.ahead > 0 ? 100 : 0))}%` }}
          />
        </div>
      )}

      {s.error && (
        <div className="flex items-start gap-1 text-[10px] text-amber-400">
          <AlertTriangle size={10} className="mt-0.5 shrink-0" />
          <span>{s.error} — the render itself is unaffected.</span>
        </div>
      )}

      {s.underruns > 0 && (
        <div className="text-[10px] text-amber-400 leading-snug">
          Rebuffered {s.underruns}&times; — rendering is slower than playback, so it pauses rather than
          dropping audio. A q8_0 LM or fewer flow steps fixes it.
        </div>
      )}

      {/* Say WHY it will be slow rather than letting it look broken: on the
          serial fallback the first window cannot exist until the planner is
          done, which on a fresh plan is most of the render. */}
      {interleaved === false && waiting && (
        <p className="text-[10px] text-zinc-600 leading-snug">
          Not enough VRAM to keep both model stacks resident, so the planner has to finish first.
        </p>
      )}
    </div>
  );
};
