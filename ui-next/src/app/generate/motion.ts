// The go bar's and Latest's motion (G-M-06, M-12): the page's motion tokens, reduced motion, whether a card is in
// sight, and the slide of what moves when something is added beside it (the concept's 18-motion.js flipKids,
// 35-songs.js renderSongs). Drawing only: places and times, no value.

import { dockH } from '../shell/placeToast.ts';
import { easing, mtime, reduced } from '../kit/motion.ts';


/** Is an element wholly on screen, between the top bar and the go bar when it sticks to the bottom (above the docked
 *  engine log while it is open, the concept's dockH)? */
export function inView(el: Element): boolean {
  const r = el.getBoundingClientRect(), top = document.querySelector('.top')?.getBoundingClientRect().bottom ?? 0, g = document.querySelector('.gobar')?.getBoundingClientRect();
  const bot = g && g.bottom >= innerHeight - dockH() - 2 ? g.top : innerHeight - dockH();
  return r.top >= top && r.bottom <= bot;
}

/** Where each child of a container stands, relative to the container, by a key. */
export function places(box: Element | null, sel: string, key: (el: HTMLElement) => string): Map<string, DOMRect> {
  const out = new Map<string, DOMRect>(); if (!box) return out;
  const b = box.getBoundingClientRect();
  for (const el of box.querySelectorAll<HTMLElement>(sel)) { if (!el.offsetParent) continue; const r = el.getBoundingClientRect(); out.set(key(el), new DOMRect(r.left - b.left, r.top - b.top, r.width, r.height)); }
  return out;
}
/** Slides each child that moved from where it stood (`was`) to where it stands now. */
export function slide(box: Element | null, sel: string, key: (el: HTMLElement) => string, was: Map<string, DOMRect>, time = '--t-move'): void {
  if (!box || reduced()) return;
  const now = places(box, sel, key);
  for (const el of box.querySelectorAll<HTMLElement>(sel)) {
    const a = was.get(key(el)), b = now.get(key(el)); if (!a || !b) continue;
    const dx = a.left - b.left, dy = a.top - b.top;
    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) el.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], { duration: mtime(time), easing: easing('--e-out') });
  }
}
