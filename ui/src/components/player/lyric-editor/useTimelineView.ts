// useTimelineView.ts — zoom, follow, the scroll window and paging.
//
// The editor's view is a few numbers with two behaviours attached: Follow keeps
// the playhead in the window, and zooming re-centres on what the user is
// working on. Paging belongs here too, because it is the same window moving by a
// different rule: a page is "as far as the next window of words", not a number
// of seconds.
//
// Follow is released in one place. A drag and a page both mean the user has
// taken the view, so both switch Follow off and say so through `releaseFollow`;
// that is why `useWordGestures` is handed this function instead of keeping its
// own copy of the toast.
//
// The timeline scrolls on two axes, so `revealWord` and `frameWord` do too: a
// row can be off screen below the fold even when its time is in the window.
// "Frame" is reveal plus zoom, which is why framing a word also jumps to it
// rather than leaving the selected row out of view.
//
// Every distance here is a named constant in `timingModel` rather than a number
// in the middle of an expression: "35% of the window" is a decision, and it
// should be findable and changeable beside the other ones.
//
// The return is `{ data, ...handlers }`, like the editor's other hooks.

import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import {
  ANCHOR_RATIO, clampPx, DEFAULT_PX, FOLLOW_LEAD_PX, FOLLOW_REVEAL_RATIO, FOLLOW_TAIL_PX,
  FRAME_MIN_SPAN, FRAME_SPAN_FACTOR, MIN_SPAN_SEC, PAGE_PAD_PX, ROW_H, RULER_H,
  pageTargetTime, type EditLine,
} from './timingModel';
import { editorPrefs } from '@/storage/lyricEditorSettings';

export interface TimelineViewOptions {
  scrollerRef: React.RefObject<HTMLDivElement | null>;
  /** The words paging steps through. */
  lines: EditLine[];
  currentTime: number;
  total: number;
  /** What zooming should keep in view: the selected word, else the playhead. */
  anchorTime: number;
}

/** Keep a left offset inside the element's scrollable range. */
function clampScrollLeft(el: HTMLElement, left: number): number {
  return Math.max(0, Math.min(Math.max(0, el.scrollWidth - el.clientWidth), left));
}

/** Centre one timeline row vertically, clamped to the scrollable range. */
function scrollLineIntoView(el: HTMLElement, lineIndex: number): void {
  // Rows are no longer all ROW_H: an alignment preview expands one row with a
  // second lane, moving every row below it. Measure the rendered row instead of
  // assuming index * ROW_H, so the row a review click selects is actually centred.
  const row = el.querySelector<HTMLElement>(`[data-line-index="${Math.max(0, lineIndex)}"]`);
  const rowRect = row?.getBoundingClientRect();
  const elRect = el.getBoundingClientRect();
  const y = rowRect
    ? rowRect.top - elRect.top + el.scrollTop + rowRect.height / 2
    : RULER_H + Math.max(0, lineIndex) * ROW_H + ROW_H / 2;
  el.scrollTop = Math.max(
    0,
    Math.min(Math.max(0, el.scrollHeight - el.clientHeight), y - el.clientHeight / 2),
  );
}

export function useTimelineView({
  scrollerRef, lines, currentTime, total, anchorTime,
}: TimelineViewOptions) {
  const { t } = useTranslation();
  const [pxPerSec, setPxPerSecState] = useState(() => clampPx(editorPrefs.zoom.read(DEFAULT_PX)));
  const [scrollLeft, setScrollLeft] = useState(0);
  const [follow, setFollow] = useState(() => editorPrefs.follow.read(true));
  const [viewportPx, setViewportPx] = useState(0);

  // The window's width drives paging, so it is measured rather than assumed:
  // the editor is resizable and the sidebar can fold.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const update = () => setViewportPx(el.clientWidth);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [scrollerRef]);

  // The versioned stores read a legacy `hs-lyricEditor*` key once and write the
  // new one; nothing below knows whether the value came from an old build.
  useEffect(() => { editorPrefs.follow.write(follow); }, [follow]);
  useEffect(() => { editorPrefs.zoom.write(pxPerSec); }, [pxPerSec]);

  const setPxPerSec = useCallback((value: number) => setPxPerSecState(clampPx(value)), []);
  const zoomBy = useCallback((factor: number) => setPxPerSecState(p => clampPx(p * factor)), []);

  const goToPlayhead = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollLeft = clampScrollLeft(el, currentTime * pxPerSec - el.clientWidth * ANCHOR_RATIO);
  }, [scrollerRef, currentTime, pxPerSec]);

  /** Bring a word's time across and its row down, without zooming or seeking. */
  const revealWord = useCallback((time: number, lineIndex: number) => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollLeft = clampScrollLeft(el, time * pxPerSec - el.clientWidth * ANCHOR_RATIO);
    scrollLineIntoView(el, lineIndex);
  }, [scrollerRef, pxPerSec]);

  /**
   * Frame one word: zoom so it fills most of the window, then reveal it on both
   * axes. The zoom effect below re-centres on `anchorTime`; when this word is
   * the anchor that lands on the same place, and when it is not the explicit
   * scroll is what makes the button keep its promise.
   */
  const frameWord = useCallback((start: number, end: number, lineIndex: number) => {
    const el = scrollerRef.current;
    if (!el) return;
    const span = Math.max(FRAME_MIN_SPAN, (end - start) * FRAME_SPAN_FACTOR);
    const next = clampPx(el.clientWidth / Math.max(MIN_SPAN_SEC, span));
    setPxPerSecState(next);
    el.scrollLeft = clampScrollLeft(el, start * next - el.clientWidth * ANCHOR_RATIO);
    scrollLineIntoView(el, lineIndex);
  }, [scrollerRef]);

  /**
   * Follow owns the view, so the moment the user takes it by hand it is switched
   * off and a toast says so. Leaving it on would snap the view back on the next
   * playhead tick and make the gesture feel broken.
   */
  const releaseFollow = useCallback(() => {
    setFollow(false);
    toast.info(t('lyrics.editor.followDisabled', 'Follow has been disabled.'));
  }, [t]);

  /**
   * Page by words, not by pixels: back puts the last visible word at the left
   * edge, forward puts the first word past the right edge there.
   */
  const pageBy = useCallback((dir: -1 | 1) => {
    const el = scrollerRef.current;
    if (!el) return;
    const target = pageTargetTime(lines, dir, {
      scrollLeft: el.scrollLeft, viewportPx: el.clientWidth || viewportPx, pxPerSec,
    });
    if (target == null) return;
    if (follow) releaseFollow();
    el.scrollLeft = Math.max(0, target * pxPerSec - PAGE_PAD_PX);
  }, [scrollerRef, lines, viewportPx, pxPerSec, follow, releaseFollow]);

  const canPageLeft = lines.length > 0 && scrollLeft > 1;
  const canPageRight = pageTargetTime(lines, 1, { scrollLeft, viewportPx, pxPerSec }) != null;

  // Follow the playhead only when it would leave the window; a click on a word
  // near the edge should not yank the view.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el || !follow || total <= 0) return;
    const x = currentTime * pxPerSec;
    if (x < el.scrollLeft + FOLLOW_LEAD_PX || x > el.scrollLeft + el.clientWidth - FOLLOW_TAIL_PX) {
      el.scrollLeft = Math.max(0, x - el.clientWidth * FOLLOW_REVEAL_RATIO);
    }
  }, [currentTime, pxPerSec, follow, total, scrollerRef]);

  // Zooming re-centres on the anchor, which is what a zoom is for: the thing
  // you are working on should stay where you left it.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollLeft = Math.max(0, anchorTime * pxPerSec - el.clientWidth * ANCHOR_RATIO);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pxPerSec]);

  return {
    data: { pxPerSec, scrollLeft, follow, canPageLeft, canPageRight },
    setPxPerSec, zoomBy,
    setScrollLeft, setFollow, goToPlayhead,
    revealWord, frameWord,
    releaseFollow, pageBy,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type TimelineView = ReturnType<typeof useTimelineView>;
/** What a component may call: the handlers, without the state. */
export type TimelineViewHandlers = Omit<TimelineView, 'data'>;
