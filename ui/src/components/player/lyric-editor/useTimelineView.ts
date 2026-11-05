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
// Every distance here is a named constant in `timingModel` rather than a number
// in the middle of an expression: "35% of the window" is a decision, and it
// should be findable and changeable beside the other ones.
//
// The return is `{ data, ...handlers }`, like the editor's other hooks.

import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import {
  ANCHOR_RATIO, clampPx, DEFAULT_PX, FOLLOW_KEY, FOLLOW_LEAD_PX, FOLLOW_REVEAL_RATIO,
  FOLLOW_TAIL_PX, MIN_SPAN_SEC, PAGE_PAD_PX, pageTargetTime, readFlag, readNumber, ZOOM_KEY,
  type EditLine,
} from './timingModel';

export interface TimelineViewOptions {
  scrollerRef: React.RefObject<HTMLDivElement | null>;
  /** The words paging steps through. */
  lines: EditLine[];
  currentTime: number;
  total: number;
  /** What zooming should keep in view: the selected word, else the playhead. */
  anchorTime: number;
}

export function useTimelineView({
  scrollerRef, lines, currentTime, total, anchorTime,
}: TimelineViewOptions) {
  const { t } = useTranslation();
  const [pxPerSec, setPxPerSecState] = useState(() => clampPx(readNumber(ZOOM_KEY, DEFAULT_PX)));
  const [scrollLeft, setScrollLeft] = useState(0);
  const [follow, setFollow] = useState(() => readFlag(FOLLOW_KEY, true));
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

  useEffect(() => {
    try { localStorage.setItem(FOLLOW_KEY, follow ? '1' : '0'); } catch { /* storage blocked */ }
  }, [follow]);
  useEffect(() => {
    try { localStorage.setItem(ZOOM_KEY, String(pxPerSec)); } catch { /* storage blocked */ }
  }, [pxPerSec]);

  const setPxPerSec = useCallback((value: number) => setPxPerSecState(clampPx(value)), []);
  const zoomBy = useCallback((factor: number) => setPxPerSecState(p => clampPx(p * factor)), []);

  const goToPlayhead = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollLeft = Math.max(0, currentTime * pxPerSec - el.clientWidth * ANCHOR_RATIO);
  }, [scrollerRef, currentTime, pxPerSec]);

  /** Fill the window with a span of seconds, for framing one word. */
  const zoomToSpan = useCallback((span: number) => {
    const el = scrollerRef.current;
    if (!el) return;
    setPxPerSecState(clampPx(el.clientWidth / Math.max(MIN_SPAN_SEC, span)));
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
    setPxPerSec, zoomBy, zoomToSpan,
    setScrollLeft, setFollow, goToPlayhead,
    releaseFollow, pageBy,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type TimelineView = ReturnType<typeof useTimelineView>;
/** What a component may call: the handlers, without the state. */
export type TimelineViewHandlers = Omit<TimelineView, 'data'>;
