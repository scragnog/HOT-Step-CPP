// SectionMarkers.tsx — Thin row of song structure markers positioned above the waveform
// Parses section markers from LRC files and displays them at their proportional position.
// Ported from hot-step-9000.

import React, { useMemo, useState, useEffect } from 'react';

interface SectionMarker {
  time: number;
  label: string;
}

interface SectionMarkersProps {
  audioUrl?: string;
  duration: number;
}

function parseSectionMarkers(raw: string): SectionMarker[] {
  // Step 1: Parse ALL lines — both section markers and lyric lines
  const allEntries: { time: number; text: string; isSection: boolean }[] = [];
  for (const line of raw.replace(/\r/g, '').split('\n')) {
    const match = line.match(/^\[(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)$/);
    if (match) {
      const mins = parseInt(match[1], 10);
      const secs = parseInt(match[2], 10);
      const cs = match[3] ? parseInt(match[3].padEnd(2, '0').slice(0, 2), 10) : 0;
      const text = match[4].trim();
      const time = mins * 60 + secs + cs / 100;
      const isSection = /^\[.*\]$/.test(text);
      allEntries.push({ time, text, isSection });
    }
  }
  allEntries.sort((a, b) => a.time - b.time);

  // Step 2: Extract section markers and determine their real start time.
  // LRC section timestamps can be unreliable — the first actual lyric line
  // AFTER a section marker tells us when that section's vocals really begin.
  const sectionIndices: number[] = [];
  for (let i = 0; i < allEntries.length; i++) {
    if (allEntries[i].isSection) sectionIndices.push(i);
  }

  const markers: SectionMarker[] = [];
  for (let si = 0; si < sectionIndices.length; si++) {
    const idx = sectionIndices[si];
    const entry = allEntries[idx];
    // Clean the label: strip brackets, remove style hints after " - "
    let label = entry.text.slice(1, -1); // remove [ and ]
    const dashIdx = label.indexOf(' - ');
    if (dashIdx !== -1) label = label.slice(0, dashIdx);
    label = label.charAt(0).toUpperCase() + label.slice(1);

    // Find the first actual lyric line between this section and the next section
    const nextSectionIdx = si + 1 < sectionIndices.length
      ? sectionIndices[si + 1]
      : allEntries.length;
    let firstLyricTime: number | null = null;
    for (let j = idx + 1; j < nextSectionIdx; j++) {
      if (!allEntries[j].isSection && allEntries[j].text.length > 0) {
        firstLyricTime = allEntries[j].time;
        break;
      }
    }

    // Use the first lyric timestamp as the real section start if it's
    // significantly later than the marker (>1s gap = instrumental lead-in).
    const realTime = (firstLyricTime !== null && firstLyricTime - entry.time > 1)
      ? firstLyricTime
      : entry.time;

    markers.push({ time: realTime, label });
  }

  // Step 3: If the first marker starts well after 0:00 and there's no explicit Intro,
  // insert a synthetic Intro marker.
  if (markers.length > 0 && markers[0].time > 2) {
    const firstLabel = markers[0].label.toLowerCase();
    if (firstLabel !== 'intro' && firstLabel !== 'introduction') {
      markers.unshift({ time: 0, label: 'Intro' });
    }
  }

  return markers.sort((a, b) => a.time - b.time);
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
    <div className="relative w-full h-5 bg-black/40 overflow-hidden select-none flex-shrink-0">
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
              className="text-[9px] font-bold uppercase tracking-wider text-zinc-400 pl-1.5 truncate"
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
