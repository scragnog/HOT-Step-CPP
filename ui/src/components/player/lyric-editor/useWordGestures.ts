// useWordGestures.ts — dragging a word, resizing its edges, panning the view.
//
// These are pointer bookkeeping, not layout and not rules: a gesture remembers
// where it started and how far the edge auto-scroll moved the view under it,
// then hands the wanted instant to `timingModel`, which decides what is legal.
// One drag is one `record()`, taken on the first move so a click that never
// travels does not fill the undo stack.
//
// Edge auto-scroll lives here too, because it is what lets a drag keep
// travelling after the pointer leaves the window: the animation frame scrolls
// the container and adds the pixels to the gesture's own offset.
//
// The return is `{ data, ...handlers }` like the editor's other hooks. The
// timeline takes the handlers and cannot reach the state; the readout it needs
// (`dragLabel`) is passed to it as a value.

import { useRef, useState } from 'react';
import { seek } from '../../../audio/engine';
import {
  EDGE_SCROLL_PX, EDGE_SCROLL_STEP, MIN_WORD_SEC, PAN_THRESHOLD_PX,
  clampStart, moveWordTo, resizeWordEdge, type EditLine, type Selection,
} from './timingModel';

/** The readout under the pointer while a word is being moved or resized. */
export interface DragLabel {
  line: number;
  start: number;
  end: number;
}

type LinesUpdater = EditLine[] | ((prev: EditLine[]) => EditLine[]);

export interface WordGestureOptions {
  lines: EditLine[];
  setLines: (updater: LinesUpdater) => void;
  /** One undo entry per gesture. */
  record: () => void;
  onSelect: (selection: Selection) => void;
  pxPerSec: number;
  /** Whether lengthening a word pushes the words after it. */
  autoShift: boolean;
  /** Audio duration, so a resize cannot push words past the end. */
  total: number;
  follow: boolean;
  /** Follow is turned off and announced the first time a pan moves. */
  onPanWhileFollowing: () => void;
  scrollerRef: React.RefObject<HTMLDivElement | null>;
}

/** Which way the view should creep at this pointer position, if at all. */
function edgeScrollDelta(clientX: number, rect: DOMRect): number {
  if (clientX < rect.left + EDGE_SCROLL_PX) return -EDGE_SCROLL_STEP;
  if (clientX > rect.right - EDGE_SCROLL_PX) return EDGE_SCROLL_STEP;
  return 0;
}

export function useWordGestures({
  lines, setLines, record, onSelect, pxPerSec, autoShift, total, follow,
  onPanWhileFollowing, scrollerRef,
}: WordGestureOptions) {
  const [dragLabel, setDragLabel] = useState<DragLabel | null>(null);

  // `scrolled` accumulates the pixels an edge auto-scroll moved under the
  // pointer, so a drag keeps travelling past the edge of the window.
  const dragRef = useRef<{
    line: number; word: number; x: number; start: number; span: number; pushed: boolean; scrolled: number;
  } | null>(null);
  const resizeRef = useRef<{
    line: number; word: number; edge: 'left' | 'right'; x: number; start: number; end: number;
    pushed: boolean; scrolled: number;
  } | null>(null);
  const panRef = useRef<{ x: number; left: number; moved: boolean; followReleased: boolean } | null>(null);
  const edgeRef = useRef<{ dx: number } | null>(null);
  const edgeRaf = useRef<number | null>(null);

  /** Keep the view scrolling while a drag sits near an edge. */
  function startEdgeScroll() {
    if (edgeRaf.current !== null) return;
    const tick = () => {
      const el = scrollerRef.current;
      const d = edgeRef.current;
      if (!el || !d || d.dx === 0) { edgeRaf.current = null; return; }
      el.scrollLeft = Math.max(0, el.scrollLeft + d.dx);
      if (dragRef.current) dragRef.current.scrolled += d.dx;
      if (resizeRef.current) resizeRef.current.scrolled += d.dx;
      edgeRaf.current = requestAnimationFrame(tick);
    };
    edgeRaf.current = requestAnimationFrame(tick);
  }

  function stopEdgeScroll() {
    if (edgeRaf.current !== null) cancelAnimationFrame(edgeRaf.current);
    edgeRaf.current = null;
    edgeRef.current = null;
  }

  function trackEdge(clientX: number) {
    const el = scrollerRef.current;
    if (!el) return;
    const dx = edgeScrollDelta(clientX, el.getBoundingClientRect());
    edgeRef.current = { dx };
    if (dx === 0) {
      stopEdgeScroll();
      return;
    }
    startEdgeScroll();
  }

  const startDrag = (e: React.PointerEvent, li: number, wi: number) => {
    const w = lines[li]?.words?.[wi];
    if (!w) return;
    e.preventDefault();
    e.stopPropagation();
    onSelect({ line: li, word: wi });
    dragRef.current = {
      line: li, word: wi, x: e.clientX, start: w.start,
      span: Math.max(MIN_WORD_SEC, w.end - w.start), pushed: false, scrolled: 0,
    };
    setDragLabel({ line: li, start: w.start, end: w.end });
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const dragMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    const line = d ? lines[d.line] : null;
    if (!d || !line) return;
    trackEdge(e.clientX);
    const nextLineStart = d.line + 1 < lines.length ? lines[d.line + 1]?.words[0]?.start ?? Infinity : Infinity;
    const wanted = d.start + (e.clientX - d.x + d.scrolled) / pxPerSec;
    const start = clampStart(line, d.word, wanted, d.span, nextLineStart);
    // One history entry per drag, however many frames it takes.
    if (!d.pushed) { record(); d.pushed = true; }
    setLines(prev => moveWordTo(prev, d.line, d.word, wanted));
    setDragLabel({ line: d.line, start, end: start + d.span });
  };

  const endDrag = () => {
    dragRef.current = null;
    setDragLabel(null);
    stopEdgeScroll();
  };

  const startResize = (e: React.PointerEvent, li: number, wi: number, edge: 'left' | 'right') => {
    const w = lines[li]?.words?.[wi];
    if (!w) return;
    e.preventDefault();
    e.stopPropagation();
    onSelect({ line: li, word: wi });
    resizeRef.current = {
      line: li, word: wi, edge, x: e.clientX, start: w.start, end: w.end, pushed: false, scrolled: 0,
    };
    setDragLabel({ line: li, start: w.start, end: w.end });
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const resizeMove = (e: React.PointerEvent) => {
    const d = resizeRef.current;
    if (!d) return;
    trackEdge(e.clientX);
    const dx = (e.clientX - d.x + d.scrolled) / pxPerSec;
    const res = resizeWordEdge(
      lines, d.line, d.word, d.edge,
      d.edge === 'left' ? d.start + dx : d.end + dx,
      { autoShift, total },
    );
    if (!res) return;
    if (!d.pushed) { record(); d.pushed = true; }
    setLines(res.lines);
    setDragLabel({ line: d.line, start: res.start, end: res.end });
  };

  const endResize = () => {
    resizeRef.current = null;
    setDragLabel(null);
    stopEdgeScroll();
  };

  /** Background drag pans the timeline; a click that never moved seeks. */
  const startPan = (e: React.PointerEvent) => {
    const el = scrollerRef.current;
    if (!el) return;
    panRef.current = { x: e.clientX, left: el.scrollLeft, moved: false, followReleased: false };
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const movePan = (e: React.PointerEvent) => {
    const el = scrollerRef.current;
    const p = panRef.current;
    if (!el || !p) return;
    const dx = e.clientX - p.x;
    if (Math.abs(dx) <= PAN_THRESHOLD_PX) return;
    p.moved = true;
    // Only when Follow is on, and only once per gesture, so one drag never
    // stacks a toast per pointer move.
    if (follow && !p.followReleased) {
      p.followReleased = true;
      onPanWhileFollowing();
    }
    el.scrollLeft = p.left - dx;
  };

  const endPan = (e: React.PointerEvent) => {
    const p = panRef.current;
    panRef.current = null;
    const el = scrollerRef.current;
    if (!p || p.moved || !el) return;
    const rect = el.getBoundingClientRect();
    seek(Math.max(0, (e.clientX - rect.left + el.scrollLeft) / pxPerSec));
  };

  /** The pointer left the pane: forget the pan without seeking. */
  const cancelPan = () => { panRef.current = null; };

  return {
    data: { dragLabel },
    startDrag, dragMove, endDrag,
    startResize, resizeMove, endResize,
    startPan, movePan, endPan, cancelPan,
  };
}

/** The handlers only: what a component renders and calls, without the state. */
export type WordGestures = Omit<ReturnType<typeof useWordGestures>, 'data'>;
