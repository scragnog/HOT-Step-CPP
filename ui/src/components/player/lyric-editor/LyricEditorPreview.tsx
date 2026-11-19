// LyricEditorPreview.tsx — the line being sung, with the View controls beside it.
//
// This is the feedback loop the modal would otherwise hide behind itself: the
// line highlights from the timings on screen, not from the file, so a drag shows
// its result immediately without closing the editor to check.
//
// The View scope used to be a row of its own below the transport. It lives here
// instead because both rows answer the same question — "where am I?" — and the
// karaoke line is the one place the answer is already on screen. Zoom sits at
// the left edge, the line takes the middle, and the three ways of finding a
// place sit at the right edge: the playhead, the selected word, and framing it.
//
// "Jump" and "Frame" both scroll on two axes. The timeline scrolls vertically as
// well as along the track, so centring only horizontally leaves the selected row
// off screen — which is exactly the case after a preview row or a search jumps
// to a line far down the sheet. Both are disabled without a selection: there is
// nothing to jump to, and framing the playhead is what Follow is for.
//
// It is given the lines and the time rather than a prepared line and word list:
// finding what is current is this row's whole job, and the shell should not have
// to compute it in order to decide whether to render it.

import React, { useMemo } from 'react';
import { Crosshair, LocateFixed, Maximize2, Music2, ZoomIn, ZoomOut } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '@/components/shared/controls/HintButton';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { ICON_BTN } from './editorStyles';
import { MAX_PX, MIN_PX, ZOOM_FACTOR, lineAtOrBefore, type EditLine } from './timingModel';
import { WordHighlighter } from '@/components/player/WordHighlighter';

/** The zoom slider's granularity, in pixels per second. */
const ZOOM_SLIDER_STEP = 10;

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
  pxPerSec: number;
  onZoom: (pxPerSec: number) => void;
  onZoomBy: (factor: number) => void;
  onGoToPlayhead: () => void;
  /** Scroll the selected word into view on both axes. */
  onJumpToSelection: () => void;
  /** Zoom to the selected word and scroll to it. */
  onFrameSelection: () => void;
  hasSelection: boolean;
}

export const LyricEditorPreview: React.FC<LyricEditorPreviewProps> = ({
  lines, currentTime, pxPerSec, onZoom, onZoomBy, onGoToPlayhead,
  onJumpToSelection, onFrameSelection, hasSelection,
}) => {
  const { t } = useTranslation();
  const idx = lineAtOrBefore(lines, currentTime);
  const at = idx >= 0 ? lines[idx] : null;
  // A line stops being "active" at its end: past that the track is between
  // lines, and holding the old words up through a solo says the wrong thing.
  const line = at && currentTime <= at.end ? at : null;
  // During an instrumental — before the first line or between two — the row
  // still names the part when the sheet labels it.
  const nextSection = useMemo(
    () => (line ? null : lines.find(l => l.start >= currentTime && l.section?.trim())?.section ?? null),
    [line, lines, currentTime],
  );

  const words: PreviewWord[] = (line?.words ?? []).map(w => ({
    word: w.word,
    start: w.start,
    end: w.end,
    confidence: w.confidence ?? 0,
    source: (w.source as PreviewWord['source']) ?? 'whisper',
  }));
  const activeWordIndex = words.filter(w => w.start <= currentTime).length - 1;

  return (
    <div className="flex min-h-[38px] shrink-0 items-center gap-2 border-t border-zinc-200 bg-zinc-50/60 px-3 py-1.5 text-[11px] text-zinc-600 dark:border-white/10 dark:bg-white/5 dark:text-zinc-300">
      {/* Far left: how much of the track a pixel is worth. */}
      <div className="flex shrink-0 items-center gap-1">
        <HintButton
          label={t('lyrics.editor.zoomOut', 'Zoom out')}
          info={t('lyrics.editor.zoomOutInfo', 'Shows more of the track per pixel, so a whole line fits at once. Ctrl+- does the same from anywhere.')}
          className={ICON_BTN}
          onClick={() => onZoomBy(1 / ZOOM_FACTOR)}
          ariaLabel={t('lyrics.editor.zoomOut', 'Zoom out')}
        >
          <ZoomOut size={12} />
        </HintButton>
        <ParamLabel
          label={t('lyrics.editor.zoom', 'Timeline zoom')}
          info={t('lyrics.editor.zoomHint', 'Pixels per second. At 30 a phrase is a few pixels; at 300 a syllable is wide enough to read and drag. Ctrl+- and Ctrl+= change it, and the chosen zoom is remembered.')}
        >
          <input
            type="range" min={MIN_PX} max={MAX_PX} step={ZOOM_SLIDER_STEP} value={pxPerSec}
            onChange={e => onZoom(Number(e.target.value))}
            className="w-20 accent-pink-500"
            aria-label={t('lyrics.editor.zoom', 'Timeline zoom')}
          />
        </ParamLabel>
        <HintButton
          label={t('lyrics.editor.zoomIn', 'Zoom in')}
          info={t('lyrics.editor.zoomInInfo', 'Shows less of the track per pixel, which is how a syllable gets wide enough to place. Ctrl+= does the same.')}
          className={ICON_BTN}
          onClick={() => onZoomBy(ZOOM_FACTOR)}
          ariaLabel={t('lyrics.editor.zoomIn', 'Zoom in')}
        >
          <ZoomIn size={12} />
        </HintButton>
        <span className="w-9 shrink-0 tabular-nums text-zinc-500">{pxPerSec}</span>
        <span className="shrink-0 text-[10px] text-zinc-500">px/s</span>
      </div>

      {/* Middle: the line being sung, which is what the timings are for. */}
      <div className="flex min-w-0 flex-1 items-center justify-center gap-2 px-2">
        {line ? (
          <>
            {line.section && (
              <span className="shrink-0 rounded-full border border-violet-500/20 bg-violet-500/15 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-violet-300">
                {line.section}
              </span>
            )}
            <span className="min-w-0 truncate text-center text-sm font-semibold text-zinc-800 md:text-base dark:text-zinc-100">
              {words.length > 0
                ? <WordHighlighter words={words} activeWordIndex={activeWordIndex} />
                : line.text}
            </span>
          </>
        ) : (
          // Instrumental time, or the intro before the first line: the row stays
          // on screen (it is also the zoom row) and says what is happening. The
          // next labelled section is the useful name here — nothing is being
          // sung, but the part is still the chorus the label announces.
          <span className="flex min-w-0 items-center gap-1.5 text-zinc-400">
            <Music2 size={13} className="shrink-0" />
            {nextSection && (
              <span className="shrink-0 rounded-full border border-violet-500/20 bg-violet-500/15 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-violet-300">
                {nextSection}
              </span>
            )}
            <span className="truncate">{t('lyrics.editor.instrumental', 'Instrumental')}</span>
          </span>
        )}
      </div>

      {/* Far right: the three ways back to a known place. */}
      <div className="flex shrink-0 items-center gap-1">
        <HintButton
          label={t('lyrics.editor.goToPlayhead', 'Scroll to the playhead')}
          info={t('lyrics.editor.goToPlayheadInfo', 'Brings the view back to the current time after a pan, without turning Follow on. The playhead keeps moving; this only moves the window.')}
          className={ICON_BTN}
          onClick={onGoToPlayhead}
          ariaLabel={t('lyrics.editor.goToPlayhead', 'Scroll to the playhead')}
        >
          <LocateFixed size={12} />
        </HintButton>
        <HintButton
          label={t('lyrics.editor.jumpSelected', 'Jump to the selected word')}
          info={t('lyrics.editor.jumpSelectedInfo', 'Scrolls the timeline to the selected word in both directions — along the track and down to its row — without moving the playhead. Disabled until a word is selected.')}
          className={ICON_BTN}
          onClick={onJumpToSelection}
          disabled={!hasSelection}
          ariaLabel={t('lyrics.editor.jumpSelected', 'Jump to the selected word')}
        >
          <Crosshair size={12} />
        </HintButton>
        <HintButton
          label={t('lyrics.editor.zoomSelected', 'Frame the selected word')}
          info={t('lyrics.editor.zoomSelectedInfo', 'Zooms so the selected word fills most of the window, then jumps to it on both axes, for work at syllable level. Needs a selected word.')}
          className={ICON_BTN}
          onClick={onFrameSelection}
          disabled={!hasSelection}
          ariaLabel={t('lyrics.editor.zoomSelected', 'Frame the selected word')}
        >
          <Maximize2 size={12} />
        </HintButton>
      </div>
    </div>
  );
};

export default LyricEditorPreview;
