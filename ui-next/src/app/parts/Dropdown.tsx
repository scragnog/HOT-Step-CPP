// The shared dropdown a pick opens (the concept's #dd, 23-dropdown.js), in its LCD variant: a listbox pinned beside
// its button, never taller than the room it has, walked with the arrow keys, Enter or a click chooses, Escape or a
// click outside closes it and focus goes back to its button.

import { useEffect, useLayoutEffect, useRef, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import type { BoundParam } from '../../definitions/parts.ts';
import { optionsOf } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import { LcdFace } from './LcdFace.tsx';

export function Dropdown({ anchor, param, variant, label, close }: { anchor: RefObject<HTMLButtonElement | null>; param: BoundParam; variant: string; label: string; close: (refocus: boolean) => void }) {
  const say = useSay(), desk = useDesk(s => s.desk), el = useRef<HTMLDivElement>(null);
  const opts = optionsOf(ix, param.key, desk);
  useLayoutEffect(() => {
    const dd = el.current, b = anchor.current; if (!dd || !b) return;
    const r = b.getBoundingClientRect(), docW = document.documentElement.clientWidth, w = dd.offsetWidth, h = dd.offsetHeight, gap = 8, edge = 12;
    const below = innerHeight - r.bottom - gap - edge, above = r.top - gap - edge - 56, down = h <= below || below >= above;
    const room = Math.max(160, down ? below : above), fit = Math.min(h, room);
    dd.style.maxHeight = h > room ? room + 'px' : '';
    dd.style.top = (down ? r.bottom + gap : r.top - gap - fit) + 'px';
    dd.style.transformOrigin = down ? 'top center' : 'bottom center'; dd.classList.toggle('up', !down);
    dd.style.left = Math.min(docW - w - 16, Math.max(16, r.left + (r.width - w) / 2)) + 'px';
    dd.classList.toggle('scrolls', h > room);
    (dd.querySelector<HTMLElement>('[aria-selected="true"]') ?? dd.querySelector<HTMLElement>('.opt'))?.focus({ preventScroll: true });
  }, [anchor]);
  useEffect(() => {
    const out = (e: PointerEvent) => { if (!el.current?.contains(e.target as Node) && !anchor.current?.contains(e.target as Node)) close(false); };
    document.addEventListener('pointerdown', out, true);
    return () => document.removeEventListener('pointerdown', out, true);
  }, [anchor, close]);
  const key = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const list = [...el.current!.querySelectorAll<HTMLElement>('.opt')], i = list.indexOf(document.activeElement as HTMLElement);
    const to = ({ ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: list.length - 1 } as Record<string, number>)[e.key];
    if (to !== undefined) { e.preventDefault(); list[Math.max(0, Math.min(list.length - 1, to))]?.focus(); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(true); }
    else if (e.key === 'Tab') close(false);
  };
  return createPortal(
    <div ref={el} id="dd" className={`dd dd-${variant} `} role="listbox" aria-label={label} onKeyDown={key}>
      <div className="dd-lcd-title">{label}</div>
      {opts.map(o => (
        <button key={String(o.value)} type="button" className="opt opt-lcd" role="option" data-opt={String(o.value)} aria-selected={o.value === param.value} tabIndex={-1}
          onClick={() => { param.set(o.value); close(true); }}>
          <span><LcdFace text={say(o.lcd ?? o.label)} /></span><small>{say(o.detail ?? '')}</small>
        </button>
      ))}
    </div>,
    document.body,
  );
}
