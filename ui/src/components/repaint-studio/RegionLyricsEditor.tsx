// RegionLyricsEditor.tsx — Line-by-line lyrics editor scoped to the repaint region
//
// Shows all LRC lines with context. Lines inside the region are editable,
// lines outside are dimmed for context. Section headers shown as badges.

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { AlertCircle, FileText } from 'lucide-react';
import {
  parseAllLrcLines,
  markRegion,
  reconstructLyrics,
  formatTime,
  type RegionLrcLine,
} from '../../utils/lrcUtils';

interface RegionLyricsEditorProps {
  /** Raw LRC text (null if no LRC available) */
  lrcText: string | null;
  /** Song lyrics from DB (fallback when no LRC) */
  fallbackLyrics: string;
  /** Region start in seconds */
  regionStart: number;
  /** Region end in seconds */
  regionEnd: number;
  /** Track duration for display */
  duration: number;
  /** Called when lyrics change — returns the full reconstructed lyrics string */
  onLyricsChange: (fullLyrics: string) => void;
}

export const RegionLyricsEditor: React.FC<RegionLyricsEditorProps> = ({
  lrcText,
  fallbackLyrics,
  regionStart,
  regionEnd,
  duration,
  onLyricsChange,
}) => {
  // ── LRC mode: line-by-line editing ──
  const allLines = useMemo(() => {
    if (!lrcText) return null;
    return parseAllLrcLines(lrcText);
  }, [lrcText]);

  const markedLines = useMemo(() => {
    if (!allLines) return null;
    const effEnd = regionEnd > 0 ? regionEnd : duration;
    return markRegion(allLines, regionStart, effEnd);
  }, [allLines, regionStart, regionEnd, duration]);

  // Editable texts for in-region lyric lines (not section headers)
  const regionLyricLines = useMemo(() => {
    if (!markedLines) return [];
    return markedLines.filter(l => l.inRegion && l.sectionLabel === null);
  }, [markedLines]);

  const [editedTexts, setEditedTexts] = useState<string[]>([]);

  // Reset edited texts when region or source changes
  useEffect(() => {
    setEditedTexts(regionLyricLines.map(l => l.text));
  }, [regionLyricLines.length, regionStart, regionEnd, lrcText]);

  // Update a single line
  const handleLineEdit = useCallback((index: number, newText: string) => {
    setEditedTexts(prev => {
      const next = [...prev];
      next[index] = newText;
      return next;
    });
  }, []);

  // Reconstruct and emit full lyrics whenever edits change
  useEffect(() => {
    if (!allLines || editedTexts.length === 0) return;
    const effEnd = regionEnd > 0 ? regionEnd : duration;
    const full = reconstructLyrics(allLines as RegionLrcLine[], editedTexts, regionStart, effEnd);
    onLyricsChange(full);
  }, [editedTexts, allLines, regionStart, regionEnd, duration]);

  // ── Fallback mode: plain textarea ──
  const [fallbackText, setFallbackText] = useState(fallbackLyrics);
  useEffect(() => {
    setFallbackText(fallbackLyrics);
  }, [fallbackLyrics]);

  const handleFallbackChange = useCallback((text: string) => {
    setFallbackText(text);
    onLyricsChange(text);
  }, [onLyricsChange]);

  // Count stats
  const inRegionCount = regionLyricLines.length;
  const sectionCount = markedLines?.filter(l => l.inRegion && l.sectionLabel !== null).length ?? 0;

  // ── Render: LRC mode ──
  if (markedLines && markedLines.length > 0) {
    // Find first and last in-region line indices for scroll context
    const firstInRegionIdx = markedLines.findIndex(l => l.inRegion);
    const lastInRegionIdx = markedLines.length - 1 - [...markedLines].reverse().findIndex(l => l.inRegion);

    // Show a few lines of context before/after
    const CONTEXT = 3;
    const showStart = Math.max(0, firstInRegionIdx - CONTEXT);
    const showEnd = Math.min(markedLines.length - 1, lastInRegionIdx + CONTEXT);

    let editableIdx = 0;

    return (
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex-shrink-0 px-4 py-3 border-b border-white/5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText size={14} className="text-pink-400" />
              <span className="text-sm font-medium text-white">Region Lyrics</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-zinc-500">
              <span>{inRegionCount} line{inRegionCount !== 1 ? 's' : ''} in range</span>
              {sectionCount > 0 && (
                <span>{sectionCount} section{sectionCount !== 1 ? 's' : ''}</span>
              )}
            </div>
          </div>
          <p className="text-[11px] text-zinc-600 mt-1">
            Edit the lyrics within the selected region. Lines outside the region are shown for context.
          </p>
        </div>

        {/* Scrollable line-by-line editor */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {/* Indicator if there are hidden lines before */}
          {showStart > 0 && (
            <div className="text-[10px] text-zinc-600 text-center py-1">
              ··· {showStart} earlier line{showStart !== 1 ? 's' : ''} ···
            </div>
          )}

          {markedLines.slice(showStart, showEnd + 1).map((line, i) => {
            const globalIdx = showStart + i;

            // Section header
            if (line.sectionLabel !== null) {
              return (
                <div
                  key={`section-${globalIdx}`}
                  className={`flex items-center gap-2 py-1.5 ${
                    line.inRegion ? '' : 'opacity-30'
                  }`}
                >
                  <span className={`
                    px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider
                    ${line.inRegion
                      ? 'bg-pink-500/20 text-pink-300 border border-pink-500/30'
                      : 'bg-white/5 text-zinc-500 border border-white/5'}
                  `}>
                    {line.sectionLabel}
                  </span>
                  <span className="text-[10px] text-zinc-600 font-mono">
                    {formatTime(line.time)}
                  </span>
                </div>
              );
            }

            // In-region lyric: editable
            if (line.inRegion) {
              const currentEditIdx = editableIdx;
              editableIdx++;
              return (
                <div
                  key={`lyric-${globalIdx}`}
                  className="flex items-start gap-2 group"
                >
                  <span className="flex-shrink-0 text-[10px] text-pink-500/60 font-mono w-10 text-right pt-1.5">
                    {formatTime(line.time)}
                  </span>
                  <input
                    type="text"
                    value={editedTexts[currentEditIdx] ?? line.text}
                    onChange={e => handleLineEdit(currentEditIdx, e.target.value)}
                    className="flex-1 px-2.5 py-1 rounded-lg bg-pink-500/5 border border-pink-500/20 text-white text-sm
                      focus:outline-none focus:border-pink-500/50 focus:bg-pink-500/10 transition-colors
                      placeholder-zinc-600 font-mono"
                    spellCheck={false}
                  />
                </div>
              );
            }

            // Out-of-region lyric: read-only context
            return (
              <div
                key={`ctx-${globalIdx}`}
                className="flex items-start gap-2 opacity-25"
              >
                <span className="flex-shrink-0 text-[10px] text-zinc-600 font-mono w-10 text-right pt-1.5">
                  {formatTime(line.time)}
                </span>
                <span className="flex-1 px-2.5 py-1 text-sm text-zinc-500 font-mono">
                  {line.text}
                </span>
              </div>
            );
          })}

          {/* Indicator if there are hidden lines after */}
          {showEnd < markedLines.length - 1 && (
            <div className="text-[10px] text-zinc-600 text-center py-1">
              ··· {markedLines.length - 1 - showEnd} later line{markedLines.length - 1 - showEnd !== 1 ? 's' : ''} ···
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Render: Fallback mode (no LRC) ──
  return (
    <div className="flex flex-col h-full">
      <div className="flex-shrink-0 px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-2">
          <AlertCircle size={14} className="text-amber-400" />
          <span className="text-sm font-medium text-white">Lyrics</span>
        </div>
        <p className="text-[11px] text-amber-500/60 mt-1">
          No timestamp data available — edit the lyrics for the section you want to change.
          The engine will regenerate the selected time region using these lyrics.
        </p>
      </div>
      <div className="flex-1 p-3">
        <textarea
          value={fallbackText}
          onChange={e => handleFallbackChange(e.target.value)}
          className="w-full h-full resize-none bg-black/20 border border-white/10 rounded-xl
            px-4 py-3 text-sm text-white placeholder-zinc-600 focus:outline-none
            focus:border-pink-500 transition-colors font-mono leading-relaxed"
          placeholder="Paste or type the lyrics for the repaint region..."
          spellCheck={false}
        />
      </div>
    </div>
  );
};
