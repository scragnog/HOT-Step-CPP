// SectionMarkers.tsx — Thin row of song structure markers positioned above the waveform
// Parses section markers from LRC files and displays them at their proportional position.
// Ported from hot-step-9000.
//
// Backends without engine-side lyric alignment (YuE2, MM3 without an LRC) never
// write a .lrc, but Whisper's .lyrics.json carries the same [Section] labels per
// line — so fall back to that, exactly as LyricsBar already does.

import React, { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { parseSectionMarkers, type SectionMarker } from '@/utils/lyrics/lrcUtils';
import { lrcOptions, lyricsJsonOptions } from '@/queries/shell-queue/lyrics';

interface SectionMarkersProps {
  audioUrl?: string;
  duration: number;
}

export const SectionMarkers: React.FC<SectionMarkersProps> = ({ audioUrl, duration }) => {
  // Shared sidecar queries, joined with LyricsBar's reads of the same track.
  const lrc = useQuery(lrcOptions(audioUrl ?? '')).data ?? null;
  const json = useQuery(lyricsJsonOptions(audioUrl ?? '')).data ?? null;

  const markers = useMemo<SectionMarker[]>(() => {
    const fromLrc = lrc ? parseSectionMarkers(lrc) : [];
    if (fromLrc.length) return fromLrc;
    const fromJson: SectionMarker[] = [];
    for (const line of json?.lines ?? []) {
      if (line.section && line.section !== fromJson[fromJson.length - 1]?.label) {
        fromJson.push({ time: line.start, label: line.section });
      }
    }
    return fromJson;
  }, [lrc, json]);

  if (markers.length === 0 || !duration) return null;

  // Deduplicate consecutive markers with the same label
  const deduped = markers.filter((m, i) => i === 0 || m.label !== markers[i - 1].label);

  return (
    <div className="relative w-full h-5 bg-black/20 dark:bg-black/40 overflow-hidden select-none flex-shrink-0">
      {deduped.map((marker, i) => {
        const leftPct = (marker.time / duration) * 100;
        const nextTime = i + 1 < deduped.length ? deduped[i + 1].time : duration;
        const widthPct = ((nextTime - marker.time) / duration) * 100;

        return (
          <div
            key={`${marker.label}-${marker.time}`}
            className="absolute top-0 h-full flex items-center"
            style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
          >
            {/* Left edge tick */}
            <div className="absolute left-0 top-0 bottom-0 w-px bg-white/20" />
            {/* Label */}
            <span
              className="text-[9px] font-bold uppercase tracking-wider text-zinc-600 dark:text-zinc-400 pl-1.5 truncate"
              title={marker.label}
            >
              {marker.label}
            </span>
          </div>
        );
      })}
    </div>
  );
};
