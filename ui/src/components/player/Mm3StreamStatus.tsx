// Mm3StreamStatus.tsx — the queue card's line about a live MiniMax-Music3 render.
//
// INFORMATIONAL ONLY, deliberately. The transport for a streaming track lives
// where every other track's transport lives: its card in the generations grid
// and the play bar at the bottom. Duplicating play/stop/volume here would give
// the same audio two sets of controls that could disagree, and would train the
// user to look in the wrong place for a track that will be an ordinary one a
// minute later.
//
// The one exception is the button the browser forces: an AudioContext created
// without a user gesture starts suspended, and if resume() is refused SOMEONE
// has to be told. That is a fault report, not a transport.

import React from 'react';
import { Headphones, AlertTriangle, Play } from 'lucide-react';
import { useMm3StreamAudio, mm3StreamResume } from '../../stores/mm3StreamStore';

interface Mm3StreamStatusProps {
  /** The job this card is for — the line only speaks for the job the store is
   *  actually streaming. */
  jobId?: string;
  /** Engine renders windows while it plans (true), fell back to serial so audio
   *  starts after planning (false), or has not decided yet (null/undefined). */
  interleaved?: boolean | null;
}

const mmss = (s: number) => {
  const t = Math.max(0, Math.floor(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
};

export const Mm3StreamStatus: React.FC<Mm3StreamStatusProps> = ({ jobId, interleaved }) => {
  const s = useMm3StreamAudio();
  if (jobId && s.jobId !== jobId) return null;

  return (
    <div className="mt-1.5 flex items-start gap-1.5 text-[10px] leading-snug">
      <Headphones size={10} className="mt-0.5 shrink-0 text-emerald-400" />
      <div className="min-w-0 flex-1 text-zinc-500">
        {s.error ? (
          <span className="flex items-start gap-1 text-amber-400">
            <AlertTriangle size={10} className="mt-0.5 shrink-0" />
            {s.error} — the render itself is unaffected.
          </span>
        ) : s.chunks === 0 ? (
          interleaved === false
            ? 'streaming — audio starts once planning finishes (both model stacks would not fit in VRAM)'
            : 'streaming — waiting for the first window…'
        ) : (
          <>
            <span className="tabular-nums text-zinc-400">{mmss(s.received)}</span>
            {s.expected > s.received && <> of <span className="tabular-nums">{mmss(s.expected)}</span></>}
            {' ready to play'}
            {s.underruns > 0 && (
              <span className="text-amber-400">
                {' · rebuffered '}{s.underruns}&times; (rendering slower than playback)
              </span>
            )}
          </>
        )}
      </div>
      {s.needsGesture && (
        <button
          onClick={mm3StreamResume}
          className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold shrink-0
                     text-white bg-emerald-600 hover:bg-emerald-500 transition-colors"
          title="Your browser blocked audio until you interact with the page"
        >
          <Play size={9} /> Enable audio
        </button>
      )}
    </div>
  );
};
