// A tier change from the view: the desk shows more or less of itself, and the controls a link named are rung on arrival
// (the concept's setTier and ringCtls, 31-shell.js). The concept's choreographed tier transition (a View Transition in
// which every part present in both tiers is one object) is a later work package; this changes the tier at once.

import type { Tier } from '../../definitions/types.ts';
import { useDesk } from '../store/desk.ts';

const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), select, textarea, [tabindex="0"]';
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export function setTierFromView(t: Tier, o: { ring?: string[]; focus?: boolean } = {}): void {
  const s = useDesk.getState();
  s.setTier(t);
  requestAnimationFrame(() => {
    if (o.focus) document.querySelector<HTMLElement>(`#tiers [data-tier="${t}"]`)?.focus({ preventScroll: true });
    if (!o.ring?.length) return;
    const els = o.ring.map(k => document.querySelector<HTMLElement>(k.startsWith('[') ? `#bento ${k}` : `#bento [data-ctl="${k}"], #bento [data-bind="${k}"]`)).filter((x): x is HTMLElement => !!x);
    els.forEach(el => { el.classList.remove('ring-here'); void el.offsetWidth; el.classList.add('ring-here'); window.setTimeout(() => el.classList.remove('ring-here'), reduced() ? 1200 : 2400); });
    els[0]?.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
    /* the link that named them is gone with the tier: focus goes to the first control it named (Q-28) */
    const a = document.activeElement, lost = !a || a === document.body || !a.isConnected, f = els[0] && (els[0].matches(FOCUSABLE) ? els[0] : els[0].querySelector<HTMLElement>(FOCUSABLE));
    if (lost && f) f.focus({ preventScroll: true });
  });
}
