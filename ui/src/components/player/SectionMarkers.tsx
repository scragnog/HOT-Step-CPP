// SectionMarkers.tsx — Thin row of song structure markers positioned above the waveform
// Parses section markers from LRC files and displays them at their proportional position.
// Ported from hot-step-9000.

import React, { useMemo, useState, useEffect } from 'react';
import { parseSectionMarkers } from '../../utils/lrcUtils';

interface SectionMarkersProps {
  audioUrl?: string;
  duration: number;
}

export const SectionMarkers: React.FC<SectionMarkersProps> = ({ audioUrl, duration }) => {
  const [fetchedLrc, setFetchedLrc] = useState<string | null>(null);

  useEffect(() => {
    if (!audioUrl) { setFetchedLrc(null); return; }
    let cancelled = false;
    const lrcUrl = audioUrl.replace(/\.\w+$/, '.lrc');
    fetch(lrcUrl)
      .then(res => { if (!res.ok) throw new Error('No LRC'); return res.text(); })
      .then(text => { if (!cancelled && text.includes('[')) setFetchedLrc(text); })
      .catch(() => { if (!cancelled) setFetchedLrc(null); });
    return () => { cancelled = true; };
  }, [audioUrl]);

  const markers = useMemo(() => fetchedLrc ? parseSectionMarkers(fetchedLrc) : [], [fetchedLrc]);

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
