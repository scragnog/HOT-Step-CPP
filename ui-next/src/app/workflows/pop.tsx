// The popover the Workflows panel and the go bar's Drafts open (the concept's 17-popover.js): drawn into the
// skeleton's #pop, placed by the concept's one placement rule, toggled from its own button, closed by Escape or a press
// outside it, focus given back to its button. Package 8 builds the page's popover for every other use; this one is the
// row menu's and the drafts' alone, and folds into package 8's when both are in (notes: deviation 20).

import { useEffect, useLayoutEffect, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { closePop, placeFloat } from './popPlace.ts';
import { useWfView } from './state.ts';

/** The popover's frame: what it holds is drawn by its owner (the row menu, the drafts list). */
export function Pop({ children, label, fixedUp }: { children: ReactNode; label: string; fixedUp?: boolean }) {
  const pop = useWfView(s => s.pop), host = document.getElementById('pop');
  useLayoutEffect(() => {
    const host = document.getElementById('pop');
    if (!host || !pop) return;
    host.hidden = false; host.setAttribute('aria-label', label);
    host.classList.toggle('fixedpop', !!fixedUp);
    placeFloat(host, pop.anchor, fixedUp ? { up: true, over: document.querySelector('.gobar') } : {});
  });
  useEffect(() => {
    const host = document.getElementById('pop');
    if (!host || !pop) return;
    /* it takes focus (the drafts' name field, the menu's first item) */
    (host.querySelector<HTMLElement>('#draftName') ?? host.querySelector<HTMLElement>('[role="menuitem"]'))?.focus({ preventScroll: true });
    /* a press outside it closes it (not one on the toast, which speaks for what it just did, nor on its own button) */
    const out = (e: PointerEvent) => { const t = e.target as Element | null; if (!t?.isConnected || t.closest('#pop, #toast') || pop.anchor.contains(t)) return; closePop(false); };
    /* a menu moves with the arrow keys, Home and End (C-13) */
    const keys = (e: KeyboardEvent) => {
      const items = [...host.querySelectorAll<HTMLElement>('[role="menuitem"]:not(:disabled)')], i = items.indexOf(document.activeElement as HTMLElement);
      if (!items.length || i < 0) return;
      const j = ({ ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: items.length - 1 } as Record<string, number>)[e.key]; if (j === undefined) return;
      e.preventDefault(); items[(j + items.length) % items.length]!.focus();
    };
    document.addEventListener('pointerdown', out, true); host.addEventListener('keydown', keys);
    return () => {
      document.removeEventListener('pointerdown', out, true); host.removeEventListener('keydown', keys);
      host.hidden = true; host.classList.remove('fixedpop', 'capped-float', 'up'); host.style.maxHeight = '';
    };
  }, [pop]);
  return host && pop ? createPortal(children, host) : null;
}
