// Mm3StreamPlayer.tsx — "Play While Rendering" for MiniMax-Music3.
//
// Sits in the Create panel while an MM3 job that asked for streaming is
// running. Press Listen and the engine's finished windows start playing while
// the rest of the song is still being computed.
//
// Deliberately NOT the StreamPlayer next door: that one belongs to ACE's
// shelved streamMode and speaks in DiT steps and preview chunks. This speaks in
// windows and buffer headroom, and it drives useMm3StreamAudio's hard-splice
// scheduler rather than STORM's crossfading one.
//
// The two numbers worth watching are BUFFER and, if it ever appears, the
// late-window count. Rendering runs faster than realtime, so the buffer should
// climb; if it drains to zero the render has fallen behind playback and the
// headroom needs raising.

import React from 'react';
import { Headphones, Square, Volume2, Loader2, AlertTriangle } from 'lucide-react';

interface Mm3StreamPlayerProps {
  isPlaying: boolean;
  chunks: number;
  received: number;
  position: number;
  ahead: number;
  underruns: number;
  done: boolean;
  /** Stream open, or scheduled audio still playing out. */
  active: boolean;
  /** Engine renders windows while it plans (true), fell back to serial so audio
   *  starts after planning (false), or has not decided yet (null). */
  interleaved: boolean | null;
  volume: number;
  headroom: number;
  error: string | null;
  onStart: () => void;
  onStop: () => void;
  onVolume: (v: number) => void;
  onHeadroom: (s: number) => void;
}

const mmss = (s: number) => {
  const t = Math.max(0, Math.floor(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
};

export const Mm3StreamPlayer: React.FC<Mm3StreamPlayerProps> = ({
  isPlaying, chunks, received, position, ahead, underruns, done, active, interleaved,
  volume, headroom, error, onStart, onStop, onVolume, onHeadroom,
}) => {
  // "Live" = the stream is open, or its tail is still playing out.
  const live = isPlaying || active;
  const waiting = live && chunks === 0;

  return (
    <div className="bg-zinc-100 dark:bg-zinc-900/80 backdrop-blur-sm border border-emerald-500/20 rounded-xl p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-500 dark:text-emerald-400">
          <Headphones size={13} />
          Play While Rendering
        </div>
        {live && (
          <span className="text-[10px] tabular-nums text-zinc-500">
            {chunks} window{chunks === 1 ? '' : 's'} · {mmss(received)} rendered
          </span>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-1.5 text-[11px] text-amber-600 dark:text-amber-400">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          <span>{error} — the render itself is unaffected.</span>
        </div>
      )}

      {!live ? (
        <>
          <button
            onClick={onStart}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-semibold
                       text-white bg-emerald-600 hover:bg-emerald-500 transition-colors"
          >
            <Headphones size={14} />
            Listen now
          </button>
          {/* Say WHY it will be slow rather than letting it look broken: on the
              serial fallback the first window cannot exist until the planner
              is done, which on a fresh plan is most of the render. */}
          {interleaved === false && (
            <p className="text-[10px] text-zinc-500 leading-snug">
              Not enough VRAM to keep both model stacks resident, so audio starts once planning
              finishes rather than a few seconds in. Reusing the planner output, or a smaller LM
              quant, avoids the wait.
            </p>
          )}
          {/* Only adjustable before starting: the playback origin of a live
              stream is already fixed, and moving it would misplace every
              chunk already scheduled against it. */}
          <label className="flex items-center gap-2 text-[10px] text-zinc-500">
            <span className="shrink-0">Buffer</span>
            <input
              type="range" min={0} max={60} step={1} value={headroom}
              onChange={e => onHeadroom(Number(e.target.value))}
              className="flex-1 accent-emerald-500"
            />
            <span className="w-8 text-right tabular-nums">{headroom}s</span>
          </label>
        </>
      ) : (
        <>
          <div className="flex items-center gap-2">
            <button
              onClick={onStop}
              title="Stop listening (the render continues)"
              className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white
                         bg-zinc-200/60 dark:bg-zinc-800/60 transition-colors"
            >
              <Square size={13} />
            </button>
            <div className="flex-1 text-[11px] tabular-nums text-zinc-600 dark:text-zinc-300">
              {waiting ? (
                <span className="flex items-center gap-1.5 text-zinc-500">
                  <Loader2 size={11} className="animate-spin" />
                  waiting for the first window…
                </span>
              ) : (
                <>
                  {mmss(position)} · buffer {ahead.toFixed(1)}s
                  {done && <span className="ml-1.5 text-emerald-500">render complete</span>}
                </>
              )}
            </div>
            <Volume2 size={13} className="text-zinc-500 shrink-0" />
            <input
              type="range" min={0} max={1} step={0.01} value={volume}
              onChange={e => onVolume(Number(e.target.value))}
              className="w-16 accent-emerald-500"
            />
          </div>

          {/* Buffer meter — full bar = one headroom's worth of audio scheduled
              ahead of the playhead. Empty means the renderer has been caught. */}
          <div className="h-1 rounded-full bg-zinc-300/60 dark:bg-zinc-800 overflow-hidden">
            <div
              className="h-full bg-emerald-500 transition-[width] duration-200"
              style={{ width: `${Math.min(100, headroom > 0 ? (ahead / headroom) * 100 : (ahead > 0 ? 100 : 0))}%` }}
            />
          </div>

          {underruns > 0 && (
            <div className="text-[10px] text-amber-600 dark:text-amber-400">
              Paused to rebuffer {underruns}&times; — rendering is running slower than playback, so the
              track is playing with gaps rather than dropping audio. A q8_0 LM, fewer flow steps, or a
              bigger buffer next time all fix it.
            </div>
          )}
        </>
      )}
    </div>
  );
};
