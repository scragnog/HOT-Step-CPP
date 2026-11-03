// LyricTimelineMarks.tsx — where every syllable starts, drawn on the waveform.
//
// The waveform answers "when is the sound" and the lyrics strip answers "which
// line", but neither says where one syllable gives way to the next. When a line
// lights up early the cause is usually a syllable boundary, and that boundary is
// invisible. This paints a tick per word start over the bars: line starts get a
// full-height rule, words a short comb along the bottom, and the syllable being
// sung right now is pink.
//
// Pointer events pass through, so seeking and the waveform's own hover time keep
// working everywhere underneath. Only the static ticks are memoised: the active
// syllable is one extra element, so a moving playhead re-renders two nodes rather
// than every word in the song.

import React, { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { lrcOptions, lyricsJsonOptions } from '../../queries/shell-queue/lyrics';
import { parseLrc } from '../../utils/lrcUtils';
import { findActiveWordIndex, findCurrentLineIndex } from '../../utils/wordLrcUtils';

interface LyricTimelineMarksProps {
  audioUrl?: string;
  duration: number;
  currentTime: number;
}

/** Fraction of the track, clamped. */
function pct(t: number, duration: number): number {
  return Math.max(0, Math.min(100, (t / duration) * 100));
}

/** One tick per time, positioned by fraction of the track. */
const Ticks = React.memo(function Ticks({
  times, duration, className,
}: { times: number[]; duration: number; className: string }) {
  return (
    <>
      {times.map((t, i) => (
        <div key={i} className={className} style={{ left: `${pct(t, duration)}%` }} />
      ))}
    </>
  );
});

export const LyricTimelineMarks: React.FC<LyricTimelineMarksProps> = ({ audioUrl, duration, currentTime }) => {
  // Shared sidecar queries, joined with LyricsBar's and SectionMarkers' reads.
  const wordData = useQuery(lyricsJsonOptions(audioUrl ?? '')).data ?? null;
  const lrc = useQuery(lrcOptions(audioUrl ?? '')).data ?? null;

  const lineStarts = useMemo(() => {
    if (wordData) return wordData.lines.map(l => l.start).filter(Number.isFinite);
    return lrc ? parseLrc(lrc).map(l => l.time) : [];
  }, [wordData, lrc]);

  const wordStarts = useMemo(() => {
    if (!wordData) return [];
    const out: number[] = [];
    for (const line of wordData.lines) {
      for (const w of line.words) if (Number.isFinite(w.start)) out.push(w.start);
    }
    return out;
  }, [wordData]);

  const activeStart = useMemo(() => {
    if (!wordData) return null;
    const lineIdx = findCurrentLineIndex(wordData.lines, currentTime);
    if (lineIdx < 0) return null;
    const words = wordData.lines[lineIdx]?.words ?? [];
    const wordIdx = findActiveWordIndex(words, currentTime);
    return wordIdx >= 0 ? words[wordIdx].start : null;
  }, [wordData, currentTime]);

  if (!duration || (lineStarts.length === 0 && wordStarts.length === 0)) return null;

  return (
    <div className="absolute inset-0 pointer-events-none select-none overflow-hidden">
      <Ticks times={lineStarts} duration={duration} className="absolute top-0 bottom-0 w-px bg-white/10" />
      {wordStarts.length > 0 && (
        <Ticks times={wordStarts} duration={duration} className="absolute bottom-0 w-px h-[7px] bg-pink-400/35" />
      )}
      {activeStart != null && (
        <div
          className="absolute bottom-0 w-px h-3.5 bg-pink-400 shadow-[0_0_6px_rgba(236,72,153,0.9)]"
          style={{ left: `${pct(activeStart, duration)}%` }}
        />
      )}
    </div>
  );
};
