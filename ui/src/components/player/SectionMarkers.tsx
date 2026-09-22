// SectionMarkers.tsx — Thin row of song structure markers positioned above the waveform
// Parses section markers from LRC files and displays them at their proportional position.
// Ported from hot-step-9000.
//
// Backends without engine-side lyric alignment (YuE2, MM3 without an LRC) never
// write a .lrc, but Whisper's .lyrics.json carries the same [Section] labels per
// line — so fall back to that, exactly as LyricsBar already does.

import React, { useState, useEffect } from 'react';
import { parseSectionMarkers, type SectionMarker } from '../../utils/lrcUtils';
import { fetchLyricsJson, sidecarUrl } from '../../utils/wordLrcUtils';

interface SectionMarkersProps {
  audioUrl?: string;
  duration: number;
}

export const SectionMarkers: React.FC<SectionMarkersProps> = ({ audioUrl, duration }) => {
  const [markers, setMarkers] = useState<SectionMarker[]>([]);

  useEffect(() => {
    if (!audioUrl) { setMarkers([]); return; }
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch(sidecarUrl(audioUrl, '.lrc'));
        if (res.ok) {
          const text = await res.text();
          if (cancelled) return;
          if (text.includes('[')) {
            const fromLrc = parseSectionMarkers(text);
            if (fromLrc.length) { setMarkers(fromLrc); return; }
          }
        }
      } catch { /* no LRC — try the word-level lyrics below */ }

      const json = await fetchLyricsJson(audioUrl);
      if (cancelled) return;
      const fromJson: SectionMarker[] = [];
      for (const line of json?.lines ?? []) {
        if (line.section && line.section !== fromJson[fromJson.length - 1]?.label) {
          fromJson.push({ time: line.start, label: line.section });
        }
      }
      setMarkers(fromJson);
    })();

    return () => { cancelled = true; };
  }, [audioUrl]);

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
