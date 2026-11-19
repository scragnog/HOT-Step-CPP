// useWordGestures.ts — dragging a word, resizing its edges, panning the view.
//
// These are pointer bookkeeping, not layout and not rules: a gesture remembers
// where it started and how far the edge auto-scroll moved the view under it,
// then hands the wanted instant to `timingModel`, which decides what is legal.
// One drag is one `record()`, taken on the first move so a click that never
// travels does not fill the undo stack.
//
// A word picked up out of a multi-word selection carries the whole block: the
// gesture keeps the span it grabbed and asks `dragSelection` for the block each
// frame, from the lines the gesture started with, so the clamp cannot be eaten a
// frame at a time. A press inside a block that never travels is a plain click
// and narrows the selection to the word under the pointer, on release. A
// shift-click is not such a press: it extends the range (across a line break
// too, which is what Loop replays) and carries no block, or the release would
// collapse the extension the same click had just made.
//
// Edge auto-scroll lives here too, because it is what lets a drag keep
// travelling after the pointer leaves the window: the animation frame scrolls
// the container and adds the pixels to the gesture's own offset.
//
// The return is `{ data, ...handlers }` like the editor's other hooks. The
// timeline takes the handlers and cannot reach the state; the readout it needs
// (`dragLabel`) is passed to it as a value.

import { useRef, useState } from 'react';
import { seek } from '@/audio/engine';
import {
  EDGE_SCROLL_PX, EDGE_SCROLL_STEP, MIN_WORD_SEC, PAN_THRESHOLD_PX,
  clampStart, dragSelection, moveWordTo, resizeWordEdge, selectedKeys, selectedRefs,
  type EditLine, type Selection, type SelectionSpan,
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
  /** `extend` is a shift-click: it moves the far end of the selection. */
  onSelect: (selection: Selection, extend: boolean) => void;
  /** The live selection, so a drag knows whether it picked up a block. */
  selection: SelectionSpan | null;
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
  lines, setLines, record, onSelect, selection, pxPerSec, autoShift, total, follow,
  onPanWhileFollowing, scrollerRef,
}: WordGestureOptions) {
  const [dragLabel, setDragLabel] = useState<DragLabel | null>(null);

  // `scrolled` accumulates the pixels an edge auto-scroll moved under the
  // pointer, so a drag keeps travelling past the edge of the window.
  const dragRef = useRef<{
    line: number; word: number; x: number; start: number; span: number;
    /** The block this drag carries, when the word was picked up out of one. */
    block: SelectionSpan | null;
    /** The lines as they were when the gesture began, so a move is not cumulative. */
    base: EditLine[];
    pushed: boolean; scrolled: number;
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

  /**
   * Pick up a word. A word taken out of a multi-word selection carries the
   * whole block; anything else selects what was clicked and moves only it.
   *
   * A click inside a block is not collapsed here, because the pointer may be
   * about to drag it: `endDrag` collapses on release when nothing travelled.
   */
  const startDrag = (e: React.PointerEvent, li: number, wi: number) => {
    const w = lines[li]?.words?.[wi];
    if (!w) return;
    e.preventDefault();
    e.stopPropagation();

    const ref = { line: li, word: wi };
    const inBlock = !e.shiftKey
      && selectedRefs(lines, selection).length > 1
      && selectedKeys(lines, selection).has(`${li}:${wi}`);
    if (!inBlock) onSelect(ref, e.shiftKey);

    dragRef.current = {
      line: li, word: wi, x: e.clientX, start: w.start,
      span: Math.max(MIN_WORD_SEC, w.end - w.start),
      // Only a word picked up out of an existing block carries that block. A
      // shift-click's own span is a range being extended, not a grab: treating
      // it as a block made both its ghost drag move every word and its release
      // collapse the selection to the clicked word, so a selection could never
      // span two lines.
      block: inBlock ? selection! : null,
      base: lines, pushed: false, scrolled: 0,
    };
    setDragLabel({ line: li, start: w.start, end: w.end });
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };

  const dragMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    const line = d ? d.base[d.line] : null;
    if (!d || !line) return;
    trackEdge(e.clientX);
    const delta = (e.clientX - d.x + d.scrolled) / pxPerSec;
    // One history entry per drag, however many frames it takes.
    if (!d.pushed) { record(); d.pushed = true; }

    if (d.block) {
      // The block moves as one, from the lines the gesture started with, so the
      // clamp cannot be eaten a frame at a time.
      const next = dragSelection(d.base, d.block, delta, total);
      setLines(next);
      const moved = next[d.line]?.words?.[d.word];
      if (moved) setDragLabel({ line: d.line, start: moved.start, end: moved.end });
      return;
    }

    const nextLineStart = d.line + 1 < d.base.length ? d.base[d.line + 1]?.words[0]?.start ?? Infinity : Infinity;
    const wanted = d.start + delta;
    const start = clampStart(line, d.word, wanted, d.span, nextLineStart);
    setLines(prev => moveWordTo(prev, d.line, d.word, wanted));
    setDragLabel({ line: d.line, start, end: start + d.span });
  };

  const endDrag = () => {
    const d = dragRef.current;
    // A press inside a block that never travelled is a plain click: it narrows
    // the selection to the word under the pointer.
    if (d && !d.pushed && d.block) onSelect({ line: d.line, word: d.word }, false);
    dragRef.current = null;
    setDragLabel(null);
    stopEdgeScroll();
  };

  const startResize = (e: React.PointerEvent, li: number, wi: number, edge: 'left' | 'right') => {
    const w = lines[li]?.words?.[wi];
    if (!w) return;
    e.preventDefault();
    e.stopPropagation();
    onSelect({ line: li, word: wi }, e.shiftKey);
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
    if (!el || e.button !== 0) return;
    // Capture the PANE, not the content inside it. Scrolling moves the content
    // under the pointer, and a capture target that slides away fires boundary
    // events and loses the gesture after the first small adjustment. The pane
    // itself never moves, so the drag keeps going for as long as the button is
    // held — which is what makes a long pan possible.
    e.preventDefault();
    panRef.current = { x: e.clientX, left: el.scrollLeft, moved: false, followReleased: false };
    el.setPointerCapture?.(e.pointerId);
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
    const max = Math.max(0, el.scrollWidth - el.clientWidth);
    el.scrollLeft = Math.max(0, Math.min(max, p.left - dx));
  };

  const endPan = (e: React.PointerEvent) => {
    const p = panRef.current;
    panRef.current = null;
    const el = scrollerRef.current;
    if (!p || p.moved || !el) return;
    const rect = el.getBoundingClientRect();
    seek(Math.max(0, (e.clientX - rect.left + el.scrollLeft) / pxPerSec));
  };

  /** The gesture was cancelled: forget the pan without seeking. */
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
