// A text box that grows with its text up to a cap, and past the cap shows where you are with a marker instead of a
// scrollbar (the concept's grow and scrollMark, 19-text-boxes.js): the prompt card's box and the lyric sheet. Drawing
// only: it measures the box and places the marker; the text is the part's.

import { useCallback, useEffect, useLayoutEffect, useState, type CSSProperties, type PointerEvent as HandEvent, type RefObject } from 'react';

export interface Grown {
  /** The box's own classes: capped, and which way there is more to read. */
  cls: string;
  /** The marker, while more than a line is out of view (Q-28): where it stands, and the hand that drags it. */
  mark: { style: CSSProperties; drag: boolean; onPointerDown(e: HandEvent<HTMLElement>): void } | null;
  /** Measure again (after the text changed from outside the box). */
  grow(): void;
}

interface State { capped: boolean; up: boolean; down: boolean; top: number; h: number; on: boolean }
const NONE: State = { capped: false, up: false, down: false, top: 0, h: 0, on: false };
const same = (a: State, b: State) => a.capped === b.capped && a.up === b.up && a.down === b.down && a.on === b.on && Math.abs(a.top - b.top) < 0.5 && Math.abs(a.h - b.h) < 0.5;
const lineOf = (t: HTMLElement) => +getComputedStyle(t).lineHeight.replace('px', '') || 20;

/** Sets the box's height for its text, measured at its own height (F-09: in a grid or flex host the host's height would
 *  stretch an auto box), without the page's scroll moving; true when the text runs past the cap. */
function fit(t: HTMLTextAreaElement, cap: number): boolean {
  const html = document.documentElement, host = t.parentElement!, x = scrollX, y = scrollY, keep = t.scrollTop, anchor = html.style.overflowAnchor;
  html.style.overflowAnchor = 'none'; html.style.minHeight = html.scrollHeight + 'px'; host.style.minHeight = host.offsetHeight + 'px';
  const as = t.style.alignSelf; t.style.alignSelf = 'start'; t.style.height = 'auto';
  const h = t.scrollHeight + 2; t.style.height = Math.min(h, cap) + 'px'; t.style.alignSelf = as;
  host.style.minHeight = ''; html.style.minHeight = ''; html.style.overflowAnchor = anchor;
  if (scrollX !== x || scrollY !== y) scrollTo(x, y);
  t.scrollTop = keep;
  return h > cap;
}
function markOf(t: HTMLTextAreaElement, capped: boolean): State {
  if (!capped || t.scrollHeight - t.clientHeight <= lineOf(t)) return { ...NONE, capped };
  const max = t.scrollHeight - t.clientHeight, f = max ? t.scrollTop / max : 0, h = Math.max(28, t.clientHeight / t.scrollHeight * t.clientHeight);
  return { capped, on: true, h, top: t.offsetTop + f * (t.clientHeight - h), up: t.scrollTop > 2, down: t.scrollTop < max - 2 };
}

/** Grows the box `ref` with its text (`text`), up to `cap()` pixels. */
export function useGrow(ref: RefObject<HTMLTextAreaElement | null>, cap: () => number, text: string): Grown {
  const [st, setSt] = useState<State>(NONE), [drag, setDrag] = useState(false);
  const put = useCallback((s: State) => setSt(o => (same(o, s) ? o : s)), []);
  const grow = useCallback(() => { const t = ref.current; if (!t || !t.offsetParent) return; put(markOf(t, fit(t, cap()))); }, [ref, put, cap]);
  useLayoutEffect(grow, [text, grow]);
  useEffect(() => {
    const t = ref.current; if (!t) return;
    /* a box that changes width (the desk re-packed it, the window changed) grows again for its new width */
    let w = -1;
    const ro = new ResizeObserver(es => es.forEach(e => { const nw = Math.round(e.contentRect.width); if (nw !== w) { const first = w < 0; w = nw; if (!first) grow(); } else put(markOf(t, t.classList.contains('capped'))); }));
    ro.observe(t);
    const scroll = () => put(markOf(t, t.classList.contains('capped')));
    t.addEventListener('scroll', scroll); addEventListener('resize', grow);
    return () => { ro.disconnect(); t.removeEventListener('scroll', scroll); removeEventListener('resize', grow); };
  }, [ref, grow, put]);
  const onPointerDown = (e: HandEvent<HTMLElement>) => {
    const t = ref.current; if (!t) return;
    e.preventDefault(); const m = e.currentTarget, y0 = e.clientY, s0 = t.scrollTop;
    m.setPointerCapture(e.pointerId); setDrag(true);
    const move = (ev: PointerEvent) => { t.scrollTop = s0 + (ev.clientY - y0) * (t.scrollHeight - t.clientHeight) / (t.clientHeight - m.offsetHeight || 1); };
    const up = () => { setDrag(false); m.removeEventListener('pointermove', move); m.removeEventListener('pointerup', up); m.removeEventListener('pointercancel', up); };
    m.addEventListener('pointermove', move); m.addEventListener('pointerup', up); m.addEventListener('pointercancel', up);
  };
  return {
    cls: [st.capped ? 'capped' : '', st.on && st.up ? 'more-up' : '', st.on && st.down ? 'more-down' : ''].filter(Boolean).join(' '),
    mark: st.on ? { style: { height: st.h + 'px', top: st.top + 'px' }, drag, onPointerDown } : null,
    grow,
  };
}
