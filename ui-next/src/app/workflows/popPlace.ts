// The popover's placement and closing (17-popover.js placeFloat, closePop), and Escape closing it wherever focus is
// (the top layer). Its frame is pop.tsx.

import { onKey } from '../register/keys.ts';
import { useWfView } from './state.ts';

/** Places a floating panel by its anchor (placeFloat): below when it fits, above when that fits better (or `up` asks
 *  first), never under the top bar, never past the window's edges, 12px clear; `over` is an element it stands clear of
 *  when above (the drafts list stands above the whole go bar: F-18). */
export function placeFloat(el: HTMLElement, anchor: HTMLElement, o: { up?: boolean; over?: Element | null } = {}): void {
  const r = anchor.getBoundingClientRect(), vw = document.documentElement.clientWidth, vh = innerHeight, m = 12, gap = 6;
  const top0 = Math.max(m, (document.querySelector('.top')?.getBoundingClientRect().bottom ?? 0) + 8);
  el.style.maxHeight = ''; el.classList.remove('capped-float');
  const g = document.querySelector('.gobar')?.getBoundingClientRect();
  const floor = g?.height && g.top < vh && g.bottom > 0 && r.bottom <= g.top ? g.top - 8 : vh - m;
  const upTop = o.over ? Math.min(r.top, o.over.getBoundingClientRect().top - 2) : r.top;
  const h = el.offsetHeight, w = el.offsetWidth, below = floor - r.bottom - gap, above = upTop - gap - top0;
  const up = o.up ? !(h > above && below > above) : h > below && above > below;
  const room = Math.max(120, up ? above : below), hh = Math.min(h, room);
  if (h > room) { el.style.maxHeight = room + 'px'; el.classList.add('capped-float'); }
  const y = up ? upTop - gap - hh : r.bottom + gap, x = Math.min(Math.max(r.left, m), Math.max(m, vw - w - m));
  const fixed = getComputedStyle(el).position === 'fixed';
  el.style.top = Math.round(y + (fixed ? 0 : scrollY)) + 'px'; el.style.left = Math.round(x + (fixed ? 0 : scrollX)) + 'px'; el.style.right = 'auto';
  el.classList.toggle('up', up);
}

/** Closes the popover; focus goes back to its button unless told not to. */
export function closePop(refocus = true): void {
  const p = useWfView.getState().pop; if (!p) return;
  useWfView.setState({ pop: null });
  if (refocus && p.anchor.isConnected) p.anchor.focus({ preventScroll: true });
}

/* Escape closes it, wherever focus is (the top layer: 17-popover.js layerOpen) */
onKey({ id: 'wf-pop-escape', order: 20, when: e => e.key === 'Escape' && !!useWfView.getState().pop, run: e => { e.preventDefault(); closePop(); } });

