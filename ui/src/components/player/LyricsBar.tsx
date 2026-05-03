// LyricsBar.tsx — Bottom bar showing synced lyrics one line at a time
// Displayed between the waveform and transport controls when playing.
// Ported from hot-step-9000.

import React, { useMemo, useState, useEffect, useRef } from 'react';
import { ChevronDown, ChevronUp, Music } from 'lucide-react';

interface LrcLine {
    time: number;
    text: string;
}

interface LyricsBarProps {
    audioUrl?: string;
    currentTime: number;
    isPlaying: boolean;
}

function parseLrc(raw: string): LrcLine[] {
    const lines: LrcLine[] = [];
    for (const line of raw.replace(/\r/g, '').split('\n')) {
        const match = line.match(/^\[(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)$/);
        if (match) {
            const mins = parseInt(match[1], 10);
            const secs = parseInt(match[2], 10);
            const cs = match[3] ? parseInt(match[3].padEnd(2, '0').slice(0, 2), 10) : 0;
            const text = match[4].trim();
            if (text && !/^\[.*\]$/.test(text)) {
                lines.push({ time: mins * 60 + secs + cs / 100, text });
            }
        }
    }
    return lines.sort((a, b) => a.time - b.time);
}

function findCurrentIndex(lines: LrcLine[], time: number): number {
    if (lines.length === 0) return -1;
    let lo = 0, hi = lines.length - 1, result = -1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (lines[mid].time <= time) {
            result = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return result;
}

export const LyricsBar: React.FC<LyricsBarProps> = ({ audioUrl, currentTime, isPlaying }) => {
    const [fetchedLrc, setFetchedLrc] = useState<string | null>(null);
    const [expanded, setExpanded] = useState(true);

    // Fetch LRC — stays cached because we never unmount
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

    const lines = useMemo(() => fetchedLrc ? parseLrc(fetchedLrc) : [], [fetchedLrc]);
    const currentIdx = findCurrentIndex(lines, currentTime);

    // Derive display text directly — no stale closure, no animation key
    const displayText = currentIdx >= 0 ? lines[currentIdx]?.text ?? '' : '';

    // Don't render if no LRC data at all
    if (lines.length === 0) return null;

    return (
        <div className="flex-shrink-0 border-t border-white/5 bg-black/80 backdrop-blur-sm z-30 transition-all duration-300">
            {/* Collapse/Expand toggle tab */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-center gap-2 py-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
                <Music size={11} className="text-pink-500/60" />
                <span className="font-medium tracking-wide uppercase text-[10px]">Lyrics</span>
                {expanded ? <ChevronDown size={11} /> : <ChevronUp size={11} />}
            </button>

            {/* Lyrics content */}
            <div
                className="overflow-hidden transition-all duration-300 ease-out"
                style={{ maxHeight: expanded ? '60px' : '0px', opacity: expanded ? 1 : 0 }}
            >
                <div className="px-8 pb-3 flex items-center justify-center">
                    <span
                        className="text-lg md:text-xl font-bold text-white tracking-wide text-center transition-opacity duration-300"
                        style={{
                            textShadow: '0 0 30px rgba(236, 72, 153, 0.4), 0 2px 8px rgba(0,0,0,0.5)',
                        }}
                    >
                        {displayText || '♪ ♪ ♪'}
                    </span>
                </div>
            </div>
        </div>
    );
};
