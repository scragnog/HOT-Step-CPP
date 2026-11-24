// The popover's mechanism (the concept's 17-popover.js; its view is popover.tsx): one floating panel (#pop) that opens
// from a button, takes focus, gives it back when it closes, and closes on a click outside it, on Escape (layers.ts), on
// Tab out of it, or when its button is redrawn away or scrolls out of its list. What it holds is drawn by whoever opens
// it: the Lab offer (adding.ts), a tile's moves (grip-menu.tsx), and later packages' (the adapter list, the presets,
// the drafts, a row's menu).
//
//   openPop(button, () => <MyBody />, { label: ui('...') })     toggles from `button`
//   closePop()                                                   focus back to the button
//
// placeFloat is the one placement every popover and menu uses (C-17, C-42).

import type { ReactNode } from 'react';
import type { Slot } from '../../derive.ts';
import { usePop, type PlaceOpts } from '../store/overlays.ts';
import { layerClosed, shown } from './layers.ts';

export const popHost = (): HTMLElement | null => document.getElementById('pop');
const host = popHost;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Places a floating panel by its anchor: below when it fits, above when that fits (or fits better), otherwise in the
 *  larger room, capped to it with an inner scroll; never under the top bar, never past the window's edges, 12px clear
 *  of them; the go bar, where it is in view, is a floor for anything opened above it. `prefer: 'up'` tries above
 *  first; `over` clears that element's top rather than the anchor's. Works for fixed and absolute layers. */
export function placeFloat(el: HTMLElement, anchor: HTMLElement, o: PlaceOpts = {}): void {
  const r = anchor.getBoundingClientRect(), vw = document.documentElement.clientWidth, vh = innerHeight, m = 12, gap = 6;
  const top0 = Math.max(m, (document.querySelector('.top')?.getBoundingClientRect().bottom ?? 0) + 8);
  el.style.maxHeight = ''; el.classList.remove('capped-float');
  const g = document.querySelector('.gobar')?.getBoundingClientRect();
  const floor = !document.getElementById('app')?.hidden && g?.height && g.top < vh && g.bottom > 0 && r.bottom <= g.top ? g.top - 8 : vh - m;
  const upTop = o.over ? Math.min(r.top, o.over.getBoundingClientRect().top - 2) : r.top;
  const h = el.offsetHeight, w = el.offsetWidth, below = floor - r.bottom - gap, above = upTop - gap - top0;
  const up = o.prefer === 'up' ? !(h > above && below > above) : h > below && above > below;
  const room = Math.max(120, up ? above : below), hh = Math.min(h, room);
  if (h > room) { el.style.maxHeight = room + 'px'; el.classList.add('capped-float'); }
  const y = up ? upTop - gap - hh : r.bottom + gap, x = clamp(o.alignEnd ? r.right - w : r.left, m, Math.max(m, vw - w - m));
  const fixed = getComputedStyle(el).position === 'fixed';
  el.style.top = Math.round(y + (fixed ? 0 : scrollY)) + 'px'; el.style.left = Math.round(x + (fixed ? 0 : scrollX)) + 'px'; el.style.right = 'auto';
  el.classList.toggle('up', up);
}

/** Opens the popover from `anchor` with `body` in it; from the same anchor again, closes it. True if it opened. */
export function openPop(anchor: HTMLElement, body: () => ReactNode, o: { label?: Slot; place?: PlaceOpts } = {}): boolean {
  const s = usePop.getState();
  if (s.body && s.anchor === anchor) { closePop(); return false; }
  if (s.body) closePop(false);
  anchor.setAttribute('aria-expanded', 'true');
  usePop.setState({ anchor, body, label: o.label ?? null, place: o.place ?? {}, n: s.n + 1 });
  return true;
}
/** Closes the popover; `refocus`: focus goes back to its button (unless the user has put it elsewhere). */
export function closePop(refocus = true): void {
  const s = usePop.getState(), el = host(); if (!s.body || !el) return;
  el.hidden = true; el.classList.remove('fixedpop', 'capped-float', 'up'); el.style.maxHeight = '';
  if (s.anchor?.hasAttribute('aria-expanded')) s.anchor.setAttribute('aria-expanded', 'false');
  usePop.setState({ anchor: null, body: null, label: null });
  layerClosed(el, refocus);
}
export const popOpen = (): boolean => !!usePop.getState().body;



/* a popover follows its anchor: it closes when the anchor is redrawn away or scrolls out of its list (C-14) */
document.addEventListener('scroll', () => {
  const a = usePop.getState().anchor; if (!a) return;
  if (!a.isConnected || !shown(a)) { closePop(false); return; }
  const r = a.getBoundingClientRect(), sc = a.closest('.side-body, #browser, .drafts');
  if (sc) { const s = sc.getBoundingClientRect(); if (r.bottom < s.top || r.top > s.bottom) closePop(false); }
}, { capture: true, passive: true });

/* a click outside closes it; the toast speaks for what the popover just did, so its Undo does not close it; a button
   that opens a popover of its own toggles it itself */
const OPENERS = '#ideasBtn, [data-cartadd], [data-plannerpick], [data-additem], #draftsBtn, [data-rowmore], [data-move], [data-refsmore], [data-cartadd2]';
document.addEventListener('click', e => {
  if (!popOpen()) return;
  const t = e.target as Element;
  if (t.closest?.('[data-closepop]') && host()?.contains(t)) { closePop(); return; }
  /* the click that opened it, from whatever button (a strip's named action carries none of the hooks above: RD-01) */
  if (usePop.getState().anchor?.contains(t)) return;
  if (t.isConnected && !t.closest('#pop, #toast') && !t.closest(OPENERS)) closePop();
});

/* a menu in the popover moves with the arrow keys, Home and End (C-13) */
document.addEventListener('keydown', e => {
  const el = host(); if (!el || el.hidden || !el.contains(document.activeElement)) return;
  const items = [...el.querySelectorAll<HTMLElement>('[role="menuitem"]:not(:disabled)')], i = items.indexOf(document.activeElement as HTMLElement);
  if (!items.length || i < 0) return;
  const j = ({ ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: items.length - 1 } as Record<string, number>)[e.key];
  if (j === undefined) return;
  e.preventDefault(); items[(j + items.length) % items.length]!.focus();
});
