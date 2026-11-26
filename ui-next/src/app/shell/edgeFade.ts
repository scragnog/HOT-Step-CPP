// A scrolling box with no scroll bar: its edges fade where there is more to see (the concept's edgeFade,
// 19-text-boxes.js; the design rule "edge fades, not scroll bars"). Drawing only.

import { useEffect, type RefObject } from 'react';

export function useEdgeFade(ref: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const el = ref.current; if (!el) return;
    el.classList.add('noscroll');
    const upd = () => { const st = el.scrollTop, max = el.scrollHeight - el.clientHeight; el.classList.toggle('fade-t', st > 2); el.classList.toggle('fade-b', st < max - 2); };
    el.addEventListener('scroll', upd, { passive: true });
    const ro = new ResizeObserver(upd), mo = new MutationObserver(upd);
    ro.observe(el); mo.observe(el, { childList: true, subtree: true }); upd();
    return () => { el.removeEventListener('scroll', upd); ro.disconnect(); mo.disconnect(); };
  }, [ref]);
}
