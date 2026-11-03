// useTimelineView.ts — zoom, follow and scroll position.
//
// The editor's view is a few numbers with two behaviours attached: Follow keeps
// the playhead in the window, and zooming re-centres on what the user is
// working on. Both are cheap, both are easy to get subtly wrong, so they live
// here instead of in the middle of the pointer handlers.

import { useCallback, useEffect, useState } from 'react';
import { clampPx, FOLLOW_KEY, readFlag, readNumber, ZOOM_KEY } from './timingModel';

export interface TimelineViewOptions {
  scrollerRef: React.RefObject<HTMLDivElement | null>;
  currentTime: number;
  total: number;
  /** What zooming should keep in view: the selected word, else the playhead. */
  anchorTime: number;
}

export function useTimelineView({ scrollerRef, currentTime, total, anchorTime }: TimelineViewOptions) {
  const [pxPerSec, setPxPerSecState] = useState(() => clampPx(readNumber(ZOOM_KEY, 120)));
  const [scrollLeft, setScrollLeft] = useState(0);
  const [follow, setFollow] = useState(() => readFlag(FOLLOW_KEY, true));

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
    el.scrollLeft = Math.max(0, currentTime * pxPerSec - el.clientWidth * 0.35);
  }, [scrollerRef, currentTime, pxPerSec]);

  /** Fill the window with a span of seconds, for framing one word. */
  const zoomToSpan = useCallback((span: number) => {
    const el = scrollerRef.current;
    if (!el) return;
    setPxPerSecState(clampPx(el.clientWidth / Math.max(0.1, span)));
  }, [scrollerRef]);

  // Follow the playhead only when it would leave the window; a click on a word
  // near the edge should not yank the view.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el || !follow || total <= 0) return;
    const x = currentTime * pxPerSec;
    if (x < el.scrollLeft + 20 || x > el.scrollLeft + el.clientWidth - 80) {
      el.scrollLeft = Math.max(0, x - el.clientWidth * 0.3);
    }
  }, [currentTime, pxPerSec, follow, total, scrollerRef]);

  // Zooming re-centres on the anchor, which is what a zoom is for: the thing
  // you are working on should stay where you left it.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollLeft = Math.max(0, anchorTime * pxPerSec - el.clientWidth * 0.35);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pxPerSec]);

  return {
    pxPerSec, setPxPerSec, zoomBy, zoomToSpan,
    scrollLeft, setScrollLeft,
    follow, setFollow,
    goToPlayhead,
  };
}
