// A tier change from the view: the desk shows more or less of itself, the place the user was reading stays where it was
// on screen (Q-14, RC-11), and the controls a link named are rung on arrival (the concept's setTier, deskAnchor and
// ringCtls, 31-shell.js, 11-render.js). The concept's choreographed tier transition (a View Transition in which every
// part present in both tiers is one object) is the motion package's (9); this changes the tier at once.

import { flushSync } from 'react-dom';
import type { Tier } from '../../definitions/types.ts';
import { useDesk } from '../store/desk.ts';
import { reduced } from '../kit/motion.ts';

const ANCHORS = '#bento > .tile[data-m], .lyr-card';
const anchorKey = (el: Element) => (el as HTMLElement).dataset.m ? `#bento > .tile[data-m="${(el as HTMLElement).dataset.m}"]` : '.lyr-card';
/** The place the user is reading, taken before the change: the tile focus is in, else the one nearest under the bars;
 *  the function puts it back where it was on screen (nothing at the top of the page: there is nothing to keep). */
function deskAnchor(): () => void {
  if (scrollY < 1) return () => {};
  const line = Math.max(document.querySelector('.top')?.getBoundingClientRect().bottom ?? 0, document.querySelector('.deskbar')?.getBoundingClientRect().bottom ?? 0);
  let el: Element | null = document.activeElement?.closest?.(ANCHORS) ?? null, best = Infinity;
  if (!el) document.querySelectorAll(ANCHORS).forEach(t => { const r = t.getBoundingClientRect(); if (!r.height || r.bottom <= line + 8 || r.top >= innerHeight) return; const d = Math.abs(r.top - line); if (d < best) { best = d; el = t; } });
  if (!el) return () => {};
  const k = anchorKey(el), top = (el as Element).getBoundingClientRect().top;
  return () => { const n = document.querySelector(k); if (!n) return; const d = n.getBoundingClientRect().top - top; if (Math.abs(d) > 0.5) scrollTo(scrollX, scrollY + d); };
}

const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), select, textarea, [tabindex="0"]';

export function setTierFromView(t: Tier, o: { ring?: string[]; focus?: boolean } = {}): void {
  const s = useDesk.getState(), keep = deskAnchor();
  /* drawn and laid out now, so the anchor is put back before the page is painted; and again once the tiles' own sizes
     have settled the layout (a re-pack on the next frame) */
  flushSync(() => s.setTier(t));
  keep();
  requestAnimationFrame(() => {
    keep();
    if (o.focus) document.querySelector<HTMLElement>(`#tiers [data-tier="${t}"]`)?.focus({ preventScroll: true });
    if (!o.ring?.length) return;
    /* a file's field rings its drop zone, and focus goes into it (the fix round's Q-10) */
    const els = o.ring.map(k => document.querySelector<HTMLElement>(k.startsWith('[') ? `#bento ${k}` : `#bento [data-ctl="${k}"], #bento [data-bind="${k}"]`)).filter((x): x is HTMLElement => !!x).map(el => el.closest<HTMLElement>('label.drop') ?? el);
    els.forEach(el => { el.classList.remove('ring-here'); void el.offsetWidth; el.classList.add('ring-here'); window.setTimeout(() => el.classList.remove('ring-here'), reduced() ? 1200 : 2400); });
    els[0]?.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
    /* the link that named them is gone with the tier: focus goes to the first control it named (Q-28) */
    const a = document.activeElement, lost = !a || a === document.body || !a.isConnected, f = els[0] && (els[0].matches(FOCUSABLE) ? els[0] : els[0].querySelector<HTMLElement>(FOCUSABLE));
    if (lost && f) f.focus({ preventScroll: true });
  });
}
