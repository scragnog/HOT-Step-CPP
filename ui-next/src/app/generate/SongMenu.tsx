// A song card's menu (35-songs.js, overlays/14-song-menu.html #menu): Repaint a region, Reuse its words, the other
// pages a song can be sent to. It opens above its ⋯ button, right edges together, inside the centre column and never
// over the go bar; below only when above lacks room (F-28). Arrows, Home and End move in it; Escape closes it and puts
// focus back on its button; Tab and Shift+Tab close it and go on from its button (S-16); a click outside closes it.

import { Fragment, useEffect, useLayoutEffect, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import { songCard, songMenu, type SongAction } from '../../derive.ts';
import { ui, useSay } from '../i18n/index.ts';
import { useDesk } from '../store/desk.ts';
import { setMenu, useGen, type Card } from '../store/generate.ts';
import { repaint, reuse, sendTo } from './actions.ts';
import { layerClosed, layerOpen } from '../features/layers.ts';

const trigger = (id: string) => document.querySelector<HTMLElement>(`#latest .song[data-id="${id}"] [data-songmenu]`);
const close = (refocus: boolean) => { const id = useGen.getState().menuFor; setMenu(null); if (refocus && id) trigger(id)?.focus({ preventScroll: true }); };

/** Where the menu stands (F-28): above its button when there is room, else below; never over the go bar (it is a
 *  ceiling above the cards and a floor where it sticks below them); right edges together, inside the centre column.
 *  When neither side has room the page moves first, if it can; else it takes the roomier side and scrolls inside. */
function room(b: HTMLElement): { up: number; down: number; r: DOMRect } {
  const r = b.getBoundingClientRect(), gap = 6, top0 = (document.querySelector('.top')?.getBoundingClientRect().bottom ?? 0) + 8;
  const g = document.querySelector('.gobar')?.getBoundingClientRect(), seen = !!g && g.height > 0 && g.bottom > 0 && g.top < innerHeight;
  const ceil = seen && g!.bottom <= r.top ? Math.max(top0, g!.bottom + 8) : top0, floor = seen && g!.top >= r.bottom ? g!.top - 8 : innerHeight - 12;
  return { up: r.top - gap - ceil, down: floor - r.bottom - gap, r };
}
function place(m: HTMLElement, b: HTMLElement): void {
  m.style.left = '0px'; m.style.top = '0px'; m.style.maxHeight = ''; m.style.overflowY = ''; m.classList.remove('up');
  const h = m.offsetHeight, w = m.offsetWidth, gap = 6;
  let k = room(b);
  if (k.up < h && k.down < h) { const can = document.documentElement.scrollHeight - innerHeight - scrollY; if (can > 0) { scrollBy({ top: Math.min(can, h - k.down), behavior: 'instant' }); k = room(b); } }
  const c = document.getElementById('centre')?.getBoundingClientRect(), lo = c ? c.left : 12, hi = c ? c.right : document.documentElement.clientWidth - 12;
  /* neither side has room and the page cannot move: beside its card, level with the button, between the go bar and the
     window's foot (the concept's F-34, inside F-28's bounds), when that is roomier; so it covers neither its card nor
     the go bar (RC-04) */
  const card = b.closest('.song')?.getBoundingClientRect(), side = card ? (hi - card.right - gap >= w ? card.right + gap : card.left - gap - w >= lo ? card.left - gap - w : null) : null;
  if (k.up < h && k.down < h && side !== null) {
    const ceil = k.r.top - gap - k.up, floor = k.r.bottom + gap + k.down, tall = floor - ceil;
    if (tall > Math.max(k.up, k.down)) {
      if (tall < h) { m.style.maxHeight = `${tall}px`; m.style.overflowY = 'auto'; }
      const y = Math.max(ceil, Math.min(k.r.bottom - h, floor - Math.min(h, tall)));
      m.style.left = `${side + scrollX}px`; m.style.top = `${y + scrollY}px`;
      return;
    }
  }
  const up = k.up >= h || (k.down < h && k.up > k.down), fit = up ? k.up : k.down;
  if (fit < h) { m.style.maxHeight = `${Math.max(120, fit)}px`; m.style.overflowY = 'auto'; }
  const mh = Math.min(h, fit < h ? Math.max(120, fit) : h), r = k.r;
  const left = Math.max(lo, Math.min(hi - w, r.right - w)), top = up ? r.top - gap - mh : r.bottom + gap;
  m.style.left = `${left + scrollX}px`; m.style.top = `${top + scrollY}px`; m.classList.toggle('up', up);
}

function Items({ c }: { c: Card }) {
  const say = useSay(), engine = useDesk(s => s.desk.engine), items = songMenu(engine, c.engine);
  const act = (id: SongAction) => { close(false); if (id === 'repaint') repaint(c); else if (id === 'reuse') reuse(c); else sendTo(id, c); };
  return <>{items.map(it => <Fragment key={it.id}>{it.rule ? <hr /> : null}<button type="button" role="menuitem" tabIndex={-1} data-sa={it.id} onClick={() => act(it.id)}>{say(it.says)}{it.small ? <small>{say(it.small)}</small> : null}</button></Fragment>)}</>;
}

export function SongMenu() {
  const say = useSay(), id = useGen(s => s.menuFor), c = useGen(s => s.cards.find(x => x.id === s.menuFor && !x.pending && !x.error));
  useLayoutEffect(() => {
    const m = document.getElementById('menu'); if (!m) return;
    m.hidden = !c;
    if (!c) { m.removeAttribute('aria-label'); m.style.maxHeight = ''; layerClosed(m, false); return; }
    m.setAttribute('aria-label', say(ui('gen.actions', { name: songCard(c).name })));
    const b = trigger(c.id); if (b) place(m, b);
    /* one of the page's layers (features/layers.ts): Escape closes it and focus goes back to its button */
    layerOpen(m, { trigger: b, close: () => close(true) });
    m.querySelector<HTMLElement>('button:not(:disabled)')?.focus({ preventScroll: true });
  }, [c?.id]); // eslint-disable-line react-hooks/exhaustive-deps -- placed and focused when it opens
  /* a click outside closes it (its own button toggles it); focus goes back to its button unless the click put it on
     something else (layers.ts layerClosed: RC-05) */
  useEffect(() => {
    if (!id) return;
    const out = (e: MouseEvent) => {
      const t = e.target as Element, m = document.getElementById('menu'); if (!t.isConnected || t.closest?.('#menu') || t.closest?.('[data-songmenu]')) return;
      close(false); if (m) layerClosed(m);
    };
    document.addEventListener('click', out, true);
    return () => document.removeEventListener('click', out, true);
  }, [id]);
  useEffect(() => {
    const m = document.getElementById('menu'); if (!m) return;
    const key = (e: globalThis.KeyboardEvent) => onKey(e as unknown as KeyboardEvent);
    m.addEventListener('keydown', key);
    return () => m.removeEventListener('keydown', key);
  }, []);
  const host = document.getElementById('menu');
  return host && c ? createPortal(<Items c={c} />, host) : null;
}

function onKey(e: KeyboardEvent): void {
  const m = document.getElementById('menu')!, items = [...m.querySelectorAll<HTMLElement>('[role="menuitem"]:not(:disabled)')], i = items.indexOf(document.activeElement as HTMLElement);
  /* Tab leaves the menu: it closes, and focus goes on from its button, as if the menu had not been there (S-16) */
  if (e.key === 'Tab') { close(true); return; }
  const j = ({ ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: items.length - 1 } as Record<string, number>)[e.key]; if (j === undefined) return;
  e.preventDefault(); items[(j + items.length) % items.length]?.focus();
}
