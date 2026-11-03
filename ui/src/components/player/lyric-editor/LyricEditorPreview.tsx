// LyricEditorPreview.tsx — the line being sung, in the editor's own state.
//
// This is the feedback loop the modal would otherwise hide behind itself: the
// line highlights from the timings on screen, not from the file, so a drag shows
// its result immediately without closing the editor to check.
//
// It is given the lines and the time rather than a prepared line and word list:
// finding what is current is this row's whole job, and the shell should not have
// to compute it in order to decide whether to render it. No line means no row,
// which is a guard at the top rather than a wrapper at the call site.

import React from 'react';
import { lineAtOrBefore, type EditLine } from './timingModel';
import { WordHighlighter } from '../WordHighlighter';

export interface PreviewWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
  source: 'matched' | 'whisper' | 'ad-lib';
}

export interface LyricEditorPreviewProps {
  lines: EditLine[];
  currentTime: number;
}

export const LyricEditorPreview: React.FC<LyricEditorPreviewProps> = ({ lines, currentTime }) => {
  const idx = lineAtOrBefore(lines, currentTime);
  if (idx < 0) return null;
  const line = lines[idx];

  const words: PreviewWord[] = line.words.map(w => ({
    word: w.word,
    start: w.start,
    end: w.end,
    confidence: w.confidence ?? 0,
    source: (w.source as PreviewWord['source']) ?? 'whisper',
  }));
  const activeWordIndex = words.filter(w => w.start <= currentTime).length - 1;

  return (
    <div className="flex items-center justify-center gap-2 border-t border-zinc-200 bg-zinc-50/60 px-3 py-2 dark:border-white/10 dark:bg-white/5">
      {line.section && (
        <span className="rounded-full border border-violet-500/20 bg-violet-500/15 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-violet-300">
          {line.section}
        </span>
      )}
      <span className="text-center text-sm font-semibold text-zinc-800 md:text-base dark:text-zinc-100">
        {words.length > 0
          ? <WordHighlighter words={words} activeWordIndex={activeWordIndex} />
          : line.text}
      </span>
    </div>
  );
};

export default LyricEditorPreview;
