// What a picker and its list share that is not drawing: a name never broken at its hyphen, and the list's scrolling box
// with no scroll bar, its edges fading where there is more (the concept's nbHyphen, 22-pick.js, and edgeFade,
// 19-text-boxes.js). Not a part of the inventory.

import { useEffect } from 'react';

/** A name such as "Composite (2-Stage)" never breaks at its hyphen (L-28). */
export const nbHyphen = (s: string): string => s.replace(/(\d)-(\w)/g, '$1\u2011$2');

/** The list's scrolling box (found by its selector in the open list) fades at the edges where there is more to see. */
export function useListFade(host: () => HTMLElement | null, selector: string, open: unknown): void {
  useEffect(() => {
    const el = host()?.querySelector<HTMLElement>(selector); if (!el) return;
    el.classList.add('noscroll');
    const upd = () => { const st = el.scrollTop, max = el.scrollHeight - el.clientHeight; el.classList.toggle('fade-t', st > 2); el.classList.toggle('fade-b', st < max - 2); };
    el.addEventListener('scroll', upd, { passive: true });
    const ro = new ResizeObserver(upd); ro.observe(el); upd();
    return () => { el.removeEventListener('scroll', upd); ro.disconnect(); };
  }, [host, selector, open]);
}
