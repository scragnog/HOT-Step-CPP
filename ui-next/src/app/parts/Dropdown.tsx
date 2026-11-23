// The shared dropdown a pick opens (the concept's #dd, 23-dropdown.js), in its LCD variant: a listbox pinned beside
// its button, never taller than the room it has, walked with the arrow keys, Enter or a click chooses, Escape or a
// click outside closes it and focus goes back to its button. It is drawn into the page's one #dd (index.html), as the
// concept's is: shown while open, hidden after, with its variant's class, the choice's name and the button's width.
// Not a part of the inventory: a piece the pick parts share.

import { useEffect, useLayoutEffect, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { LcdFace, useSay, type ParamView } from '../kit/index.ts';

const host = () => document.getElementById('dd');

export function Dropdown({ anchor, p, variant, close }: { anchor: RefObject<HTMLButtonElement | null>; p: ParamView; variant: string; close: (refocus: boolean) => void }) {
  const say = useSay(), label = say(p.label), el = host();
  useLayoutEffect(() => {
    const dd = host(), b = anchor.current; if (!dd || !b) return;
    dd.className = `dd dd-${variant} `; dd.hidden = false; dd.dataset.k = p.key; dd.setAttribute('aria-label', label);
    dd.style.setProperty('--ddw', b.offsetWidth + 'px'); dd.style.maxHeight = '';
    const h = dd.offsetHeight;
    let down: boolean | null = null;
    /* pinned to the window beside its button wherever the page scrolls, never taller than the room it has; the side it
       first opened on is kept, and it closes when its button scrolls out of the window (23-dropdown.js placeDD) */
    const place = () => {
      const r = b.getBoundingClientRect(), docW = document.documentElement.clientWidth, w = dd.offsetWidth, gap = 8, edge = 12;
      if (down !== null && (r.bottom < 0 || r.top > innerHeight || !b.isConnected)) { close(false); return; }
      const below = innerHeight - r.bottom - gap - edge, above = r.top - gap - edge - 56;
      if (down === null) down = h <= below || below >= above;
      const room = Math.max(160, down ? below : above), fit = Math.min(h, room);
      dd.style.maxHeight = h > room ? room + 'px' : '';
      dd.style.top = (down ? r.bottom + gap : r.top - gap - fit) + 'px';
      dd.style.transformOrigin = down ? 'top center' : 'bottom center'; dd.classList.toggle('up', !down);
      dd.style.left = Math.min(docW - w - 16, Math.max(16, r.left + (r.width - w) / 2)) + 'px';
      dd.classList.toggle('scrolls', h > room);
    };
    place();
    let raf = 0;
    const scroll = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; place(); }); };
    addEventListener('scroll', scroll, { passive: true, capture: true });
    (dd.querySelector<HTMLElement>('[aria-selected="true"]') ?? dd.querySelector<HTMLElement>('.opt'))?.focus();
    return () => { dd.hidden = true; removeEventListener('scroll', scroll, { capture: true }); cancelAnimationFrame(raf); };
  }, [anchor, close, label, p.key, variant]);
  useEffect(() => {
    const out = (e: PointerEvent) => { if (!host()?.contains(e.target as Node) && !anchor.current?.contains(e.target as Node)) close(false); };
    document.addEventListener('pointerdown', out, true);
    return () => document.removeEventListener('pointerdown', out, true);
  }, [anchor, close]);
  useEffect(() => {
    const dd = host(); if (!dd) return;
    const key = (e: KeyboardEvent) => {
      const list = [...dd.querySelectorAll<HTMLElement>('.opt')], i = list.indexOf(document.activeElement as HTMLElement);
      const to = ({ ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: list.length - 1 } as Record<string, number>)[e.key];
      if (to !== undefined) { e.preventDefault(); list[Math.max(0, Math.min(list.length - 1, to))]?.focus(); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(true); }
      else if (e.key === 'Tab') close(false);
    };
    dd.addEventListener('keydown', key);
    return () => dd.removeEventListener('keydown', key);
  }, [close]);
  if (!el || !p.options) return null;
  return createPortal(
    <>
      <div className="dd-lcd-title">{label}</div>
      {p.options.map(o => (
        <button key={String(o.value)} type="button" className="opt opt-lcd" role="option" data-opt={String(o.value)} aria-selected={o.on} tabIndex={-1}
          onClick={() => { p.set(o.value); close(true); }}>
          <span><LcdFace text={o.pick} /></span><small>{o.detail ? say(o.detail) : ''}</small>
        </button>
      ))}
    </>,
    el,
  );
}
