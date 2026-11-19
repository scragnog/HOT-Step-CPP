// LyricsBar.tsx — the synced lyrics strip under the waveform.
//
// Display only. It shows the current line and, when Whisper data exists, the
// syllable being sung. Editing is a deliberate act, so it is reachable only from
// the pencil in the header and only while the strip is open: the editor is a
// modal with transport, zoom and per-word placement, which a strip this short
// cannot offer. Nothing here writes to the sidecars.

import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronDown, ChevronUp, Music, Pencil } from 'lucide-react';
import { parseLrc, type LrcLine } from '@/utils/lyrics/lrcUtils';
import { findCurrentLineIndex, findActiveWordIndex } from '@/utils/lyrics/wordLrcUtils';
import { lrcOptions, lyricsJsonOptions } from '@/queries/shell-queue/lyrics';
import { cue, getPlaybackState } from '@/stores/playbackStore';
import { seedLinesFromLrc } from './lyric-editor/lrcSeed';
import type { EditLine } from './lyric-editor/timingModel';
import { WordHighlighter } from './WordHighlighter';
import { LyricEditorModal } from './LyricEditorModal';
import { HintButton } from '@/components/shared/controls/HintButton';

interface LyricsBarProps {
    audioUrl?: string;
    currentTime: number;
    isPlaying: boolean;
    /** Whole-track length, for the editor's ruler. */
    duration?: number;
    /** Song title, shown in the editor header. */
    title?: string;
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

export const LyricsBar: React.FC<LyricsBarProps> = ({ audioUrl, currentTime, duration = 0, title }) => {
    const [expanded, setExpanded] = useState(true);
    /** What the editor was opened on, frozen at the click — see `openEditor`. */
    const [editing, setEditing] = useState<{
        audioUrl: string; title?: string; duration: number; seed: EditLine[] | null;
    } | null>(null);

    // Word-level .lyrics.json takes priority; .lrc is the fallback. Both are
    // shared sidecar queries, joined with SectionMarkers' reads of the same track.
    const wordData = useQuery(lyricsJsonOptions(audioUrl ?? '')).data ?? null;
    const lrc = useQuery(lrcOptions(audioUrl ?? '')).data ?? null;
    const fetchedLrc = wordData ? null : lrc;
    // The editor's starting point when the track has line timings but no word
    // timings: one chip per word, inside the line it belongs to.
    const lrcSeed = useMemo(() => seedLinesFromLrc(lrc ?? '', duration), [lrc, duration]);

    const lines = useMemo(() => fetchedLrc ? parseLrc(fetchedLrc) : [], [fetchedLrc]);
    const currentIdx = findCurrentIndex(lines, currentTime);

    // Word-level indices
    const wordLineIdx = wordData ? findCurrentLineIndex(wordData.lines, currentTime) : -1;
    const wordLine = wordData && wordLineIdx >= 0 ? wordData.lines[wordLineIdx] : null;
    const activeWordIdx = wordLine ? findActiveWordIndex(wordLine.words, currentTime) : -1;

    // Section marker — show when the current line's section differs from the previous
    const currentSection = wordLine?.section || null;
    const prevSection = wordData && wordLineIdx > 0 ? wordData.lines[wordLineIdx - 1]?.section : null;
    const showSection = currentSection && currentSection !== prevSection;

    // Derive display text for LRC fallback
    const displayText = currentIdx >= 0 ? lines[currentIdx]?.text ?? '' : '';

    /**
     * Open the editor with this track in the player.
     *
     * The strip can be showing the track that a page reload restored into the
     * bar: it is the current track, the bar draws it, and its file was never
     * handed to the engine. The pencil used to open an editor whose transport
     * had nothing to play at all. Cueing is a no-op when the track is already
     * loaded, so a playhead set in the bar survives the click.
     */
    const openEditor = () => {
        if (!audioUrl) return;
        const track = getPlaybackState().currentTrack;
        if (track && track.audioUrl === audioUrl) cue(track);
        // Frozen here, not read from the live track: a playlist can reach the
        // end of this song behind the modal and load the next one, and the
        // editor must keep the file it opened with. Rebound to the new track it
        // would hold one song's lines while its saver pointed at the other's
        // sidecar.
        // The seed is only used when the sidecar has no words in it, so it is
        // passed unconditionally and the modal decides.
        setEditing({ audioUrl, title, duration, seed: lrcSeed });
    };

    // Don't render if no lyrics data at all
    if (lines.length === 0 && !wordData) return null;

    return (
        <>
            <div className="flex-shrink-0 border-t border-zinc-200 dark:border-white/5 bg-black/80 backdrop-blur-sm z-30 transition-all duration-300 flex flex-col">
                {/* The tab and the editor door share one header, but the door is
                    positioned out of the flow: the title stays centred over the
                    lyric line below instead of being pushed left by the button.
                    Editing needs timed lyrics — word timings, or a line-level
                    `.lrc` whose lines can be split into words — and an open
                    strip, so the pencil is only there when both are true. */}
                <div className="relative flex items-center justify-center">
                    <button
                        onClick={() => setExpanded(!expanded)}
                        className="flex items-center justify-center gap-2 px-16 py-1.5 text-xs text-zinc-500 hover:text-zinc-700 dark:text-zinc-300 transition-colors"
                    >
                        <Music size={11} className="text-pink-500/60" />
                        <span className="font-medium tracking-wide uppercase text-[10px]">Lyrics</span>
                        {expanded ? <ChevronDown size={11} /> : <ChevronUp size={11} />}
                    </button>
                    {expanded && (wordData || lrcSeed.length > 0) && audioUrl && (
                        <HintButton
                            label="Edit word timings"
                            info="Opens the lyric editor: a zoomed timeline with one chip per word, the waveform behind it and the karaoke preview underneath. The pencil appears only when the strip is open and the track has timed lyrics — word timings, or a line-level .lrc whose lines are split into words — so it is never one stray click away."
                            onClick={openEditor}
                            className="flex items-center gap-1 rounded border border-zinc-300 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-black/5 hover:text-zinc-700 dark:border-white/10 dark:text-zinc-400 dark:hover:bg-white/10"
                            rootClassName="absolute right-2 top-1/2 -translate-y-1/2"
                        >
                            <Pencil size={10} />
                            Edit
                        </HintButton>
                    )}
                </div>

                {/* Lyrics content */}
                <div
                    className="overflow-hidden transition-all duration-300 ease-out"
                    style={{ maxHeight: expanded ? '76px' : '0px', opacity: expanded ? 1 : 0 }}
                >
                    <div className="px-8 flex flex-wrap items-center justify-center gap-3">
                        {/* Section badge */}
                        {showSection && (
                            <span
                                className="flex-shrink-0 px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wider uppercase bg-gradient-to-r from-violet-500/25 to-pink-500/25 text-violet-300 border border-violet-500/20"
                                style={{ animation: 'fade-in 0.3s ease-out' }}
                            >
                                {currentSection}
                            </span>
                        )}
                        <span
                            className="text-lg md:text-xl font-bold text-white tracking-wide text-center transition-opacity duration-300"
                            style={{
                                textShadow: wordData ? undefined : '0 0 30px rgba(236, 72, 153, 0.4), 0 2px 8px rgba(0,0,0,0.5)',
                            }}
                        >
                            {wordData && wordLine ? (
                                <WordHighlighter words={wordLine.words} activeWordIndex={activeWordIdx} />
                            ) : (
                                displayText || '♪ ♪ ♪'
                            )}
                        </span>
                    </div>
                </div>
            </div>

            {editing && (                    <LyricEditorModal
                        audioUrl={editing.audioUrl}
                        title={editing.title}
                        duration={editing.duration}
                        currentTime={currentTime}
                        seed={editing.seed}
                        onClose={() => setEditing(null)}
                    />
            )}
        </>
    );
};
