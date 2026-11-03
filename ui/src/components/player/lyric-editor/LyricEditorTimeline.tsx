// LyricEditorTimeline.tsx — the scrolling map: ruler, one row per line, a chip
// per word, the drag readout, the playhead, and the paging arrows.
//
// It draws. What a gesture may do to a word is `timingModel`'s rule and where
// the pointer is relative to the window is `useWordGestures`' problem; this
// component only turns the two into elements.
//
// Chips keep a plain `title` instead of the house hover card: they are hundreds
// of drag targets inside a scrolling pane, and a portalled card per chip would
// cost more than it explains. The name and times are on the chip and in its
// aria-label.

import React, { useMemo } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '../../shared/HintButton';
import { PAN_BTN } from './editorStyles';
import {
  BOTTOM_PAD, ROW_H, RULER_H, fmt, rulerTicks, type EditLine, type Selection,
} from './timingModel';
import type { DragLabel, WordGestures } from './useWordGestures';

/** A chip never shrinks below this, so a very short word stays clickable. */
const MIN_CHIP_PX = 10;

export interface LyricEditorTimelineProps {
  lines: EditLine[];
  selected: Selection | null;
  currentTime: number;
  total: number;
  pxPerSec: number;
  scrollerRef: React.RefObject<HTMLDivElement | null>;
  onScrollLeft: (left: number) => void;
  /** The readout for the gesture in flight, from `useWordGestures().data`. */
  dragLabel: DragLabel | null;
  /** The gesture handlers: drag, resize, pan. This row cannot reach the state. */
  gestures: WordGestures;
  canPageLeft: boolean;
  canPageRight: boolean;
  onPage: (dir: -1 | 1) => void;
}

export const LyricEditorTimeline: React.FC<LyricEditorTimelineProps> = ({
  lines, selected, currentTime, total, pxPerSec, scrollerRef, onScrollLeft,
  dragLabel, gestures, canPageLeft, canPageRight, onPage,
}) => {
  const { t } = useTranslation();
  // The ruler is this row's own concern: it is thinned to the current zoom, so
  // the labels exist only where they are drawn.
  const ticks = useMemo(() => rulerTicks(total, pxPerSec), [total, pxPerSec]);

  return (
    <div className="relative flex min-h-[240px] flex-1">
      <div
        ref={scrollerRef}
        onScroll={e => onScrollLeft(e.currentTarget.scrollLeft)}
        className="relative flex-1 overflow-auto"
        title={t('lyrics.editor.panHint', 'Drag the timeline to pan, click to seek')}
      >
        {lines.length === 0 ? (
          <div className="p-6 text-center text-xs text-zinc-500">
            {t('lyrics.editor.noWords', 'This track has no word timings to edit. Run Retranscribe Lyrics first.')}
          </div>
        ) : (
          <div
            className="relative cursor-grab active:cursor-grabbing"
            style={{ width: Math.max(1, total * pxPerSec), height: RULER_H + lines.length * ROW_H + BOTTOM_PAD }}
            onPointerDown={gestures.startPan}
            onPointerMove={gestures.movePan}
            onPointerUp={gestures.endPan}
            onPointerLeave={gestures.cancelPan}
          >
            {ticks.map(time => (
              <div
                key={time}
                className="pointer-events-none absolute top-0 border-l border-zinc-300 text-[9px] tabular-nums text-zinc-500 dark:border-white/15"
                style={{ left: time * pxPerSec, height: RULER_H }}
              >
                <span className="absolute left-1 top-1 whitespace-nowrap">{fmt(time)}</span>
              </div>
            ))}

            {lines.map((line, li) => (
              <div
                key={li}
                className="pointer-events-none absolute left-0 right-0 border-b border-zinc-100 dark:border-white/5"
                style={{ top: RULER_H + li * ROW_H, height: ROW_H }}
              >
                {line.words.length === 0 && (
                  <span
                    className="absolute top-[7px] whitespace-nowrap text-[10px] italic text-zinc-500"
                    style={{ left: line.start * pxPerSec }}
                    title={t('lyrics.editor.untimedLine', 'This line has no word timings, so it can only be kept as it is')}
                  >
                    {line.text}
                  </span>
                )}
                {line.words.map((w, wi) => {
                  const isSelected = selected?.line === li && selected?.word === wi;
                  return (
                    <button
                      key={wi}
                      type="button"
                      onPointerDown={e => gestures.startDrag(e, li, wi)}
                      onPointerMove={gestures.dragMove}
                      onPointerUp={gestures.endDrag}
                      onPointerCancel={gestures.endDrag}
                      aria-pressed={isSelected}
                      aria-label={`${w.word}, ${fmt(w.start)} to ${fmt(w.end)}`}
                      className={`pointer-events-auto absolute top-[3px] h-[22px] cursor-grab overflow-hidden rounded px-1 text-left text-[10px] leading-[22px] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-300 ${
                        isSelected
                          ? 'z-20 bg-pink-500 text-white ring-2 ring-pink-300'
                          : 'bg-pink-500/15 text-pink-300 hover:bg-pink-500/30'
                      }`}
                      style={{ left: w.start * pxPerSec, width: Math.max(MIN_CHIP_PX, (w.end - w.start) * pxPerSec) }}
                      title={`${w.word} · ${fmt(w.start)} → ${fmt(w.end)}`}
                    >
                      <span className="pointer-events-none block truncate">{w.word}</span>
                      {isSelected && (
                        <>
                          <span
                            onPointerDown={e => gestures.startResize(e, li, wi, 'left')}
                            onPointerMove={gestures.resizeMove}
                            onPointerUp={gestures.endResize}
                            onPointerCancel={gestures.endResize}
                            className="pointer-events-auto absolute left-0 top-0 z-30 h-full w-[6px] cursor-ew-resize rounded-l bg-white/80"
                            title={t('lyrics.editor.resizeLeft', 'Drag to move the start of this word')}
                          />
                          <span
                            onPointerDown={e => gestures.startResize(e, li, wi, 'right')}
                            onPointerMove={gestures.resizeMove}
                            onPointerUp={gestures.endResize}
                            onPointerCancel={gestures.endResize}
                            className="pointer-events-auto absolute right-0 top-0 z-30 h-full w-[6px] cursor-ew-resize rounded-r bg-white/80"
                            title={t('lyrics.editor.resizeRight', 'Drag to change how long this word lasts')}
                          />
                        </>
                      )}
                    </button>
                  );
                })}
              </div>
            ))}

            {dragLabel && (
              <>
                <div
                  className="pointer-events-none absolute z-20 w-px bg-pink-300/80"
                  style={{ left: dragLabel.start * pxPerSec, top: RULER_H + dragLabel.line * ROW_H, height: ROW_H }}
                />
                <div
                  className="pointer-events-none absolute z-20 w-px bg-pink-300/40"
                  style={{ left: dragLabel.end * pxPerSec, top: RULER_H + dragLabel.line * ROW_H, height: ROW_H }}
                />
                <div
                  className="pointer-events-none absolute z-30 -translate-x-1/2 whitespace-nowrap rounded bg-pink-600 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-white shadow"
                  style={{ left: dragLabel.start * pxPerSec, top: RULER_H + dragLabel.line * ROW_H - 17 }}
                >
                  {fmt(dragLabel.start)} → {fmt(dragLabel.end)}
                </div>
              </>
            )}

            <div
              className="pointer-events-none absolute top-0 bottom-0 w-px bg-pink-400"
              style={{ left: currentTime * pxPerSec }}
            />
          </div>
        )}
      </div>

      <HintButton
        label={t('lyrics.editor.pageBack', 'Page back')}
        info={t('lyrics.editor.pageBackInfo', 'Scrolls left so the last word on screen becomes the first one, about one window at a time. Drag the timeline for smaller moves.')}
        className={PAN_BTN}
        rootClassName="absolute left-1 top-1/2 z-30 -translate-y-1/2"
        onClick={() => onPage(-1)}
        disabled={!canPageLeft}
        ariaLabel={t('lyrics.editor.pageBack', 'Page back')}
      >
        <ChevronLeft size={18} />
      </HintButton>
      <HintButton
        label={t('lyrics.editor.pageForward', 'Page forward')}
        info={t('lyrics.editor.pageForwardInfo', 'Scrolls right so the first word past the right edge becomes the first one on screen. Jumps to the next window of words instead of a fixed number of seconds.')}
        className={PAN_BTN}
        rootClassName="absolute right-1 top-1/2 z-30 -translate-y-1/2"
        onClick={() => onPage(1)}
        disabled={!canPageRight}
        ariaLabel={t('lyrics.editor.pageForward', 'Page forward')}
      >
        <ChevronRight size={18} />
      </HintButton>
    </div>
  );
};

export default LyricEditorTimeline;
