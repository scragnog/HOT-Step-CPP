// Drag and drop onto the desk (the concept's 30-desk.js, "drag and drop"): a Modules card, a picker drawing, a pedal
// or a side unit dragged onto the desk is added where its preview stood; a tile dragged by its header moves there.
// Delegated listeners on the document; what arrives goes through the store's add and move.
//
// The preview is the thing itself (L-26): a tile's own face, faded, standing at the place nearest the pointer among
// the places the tile could take in the order (Q-01, L-01). The layout is a pure function of the tiles, their order and
// the width (desk/pack.ts), so every place is laid out exactly (desk/layout.ts deskSlots) and the drop lands exactly
// where the preview stood; nothing else moves unpredictably. A moved tile leaves the desk while it travels, the others
// close up behind it, and they slide rather than jump while the preview makes room (M-13); the page never scrolls
// (desk/layout.ts holdScroll, Q-15). A pedal or a side unit makes the Pedalboard glow, or previews it if it is not on
// the desk yet.

import { createRoot } from 'react-dom/client';
import { flushSync } from 'react-dom';
import type { ModuleDef } from '../../definitions/types.ts';
import { deskModel } from '../../derive.ts';
import { BOARD, LEAD, defOf, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { usePicker } from '../store/overlays.ts';
import { Tile } from '../render/Tile.tsx';
import { deskSlots, lastMeasure, packNow } from '../desk/layout.ts';
import { thumbBare, thumbOf } from '../thumbs/index.ts';
import { sayNow } from '../i18n/index.ts';
import { defName } from '../../derive.ts';
import { closePicker } from './picker-keys.ts';
import { sayParked } from './adding.ts';

type Place = { k: number; x: number; y: number; w: number; h: number };
let dragId: string | null = null, ghost: HTMLElement | null = null, ghostKey: string | null = null;
let dropBefore: { k: number; m: string | null } | null = null, dragRects: Place[] | null = null;
/** Every place the dragged tile could take, as the preview chooses among them (the test layer reads it). */
export const currentDragRects = (): Place[] | null => dragRects;

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const bento = () => document.getElementById('bento');
const centre = () => document.getElementById('centre');
const make = () => { const s = useDesk.getState(); return `${deskModel(ix, s.desk, s.mods).make}|${innerWidth}`; };
const defOfItem = (id: string) => defOf(id.includes(':') ? id.split(':')[1]! : id);
const dragKey = () => (dragId?.startsWith('move:') ? dragId.slice(5) : dragId);
/** The tiles on the desk in order (not the lyric sheet, not the one travelling). */
const deskTiles = () => [...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]')].filter(t => t.dataset.m !== LEAD?.id && !t.classList.contains('moving') && !t.classList.contains('ghost')).map(t => t.dataset.m!);
/** The module the tile goes before, to stand at place k among `list` (null: the end; past the last shown tile, before
 *  the next module that is not shown here). */
function beforeAt(list: string[], k: number): string | null {
  if (k < list.length) return list[k]!;
  const s = useDesk.getState(), vis = deskModel(ix, s.desk, s.mods).tiles, last = list[list.length - 1];
  return last ? vis.slice(vis.indexOf(last) + 1).find(x => !list.includes(x) && x !== dragKey()) ?? null : null;
}

/** While dragging, the tiles make room for the preview by sliding, not jumping (M-13). */
function slideRoom(fn: () => void): void {
  const before = new Map([...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]:not(.ghost)')].map(t => [t.dataset.m!, t.getBoundingClientRect()]));
  fn(); packNow(make());
  if (reduced()) return;
  document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]:not(.ghost)').forEach(t => {
    const b = before.get(t.dataset.m!); if (!b) return; const r = t.getBoundingClientRect(), dx = b.left - r.left, dy = b.top - r.top;
    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) t.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], { duration: 240, easing: 'cubic-bezier(.2, .8, .2, 1)' });
  });
}

/** A tile drawn as the desk would draw it, as plain markup (a preview takes no part in the page). */
function drawn(def: ModuleDef): HTMLElement | null {
  const box = document.createElement('div'), root = createRoot(box);
  flushSync(() => root.render(<Tile def={def} />));
  const el = box.firstElementChild?.cloneNode(true) as HTMLElement | undefined;
  root.unmount();
  return el ?? null;
}
/** A preview is a copy of the tile (it keeps its name, as the concept's does: data-gm says it is the preview); it holds
 *  no ids, so none is on the page twice, and a new tile's preview no hooks of its own (the concept's list). */
const HOOKS = ['data-bind', 'data-knob', 'data-drag', 'data-scope', 'data-term', 'data-pedals'];
function ghostCopy(g: HTMLElement, key: string, hooks: boolean): HTMLElement {
  g.classList.add('ghost'); g.dataset.gm = key;
  [g, ...g.querySelectorAll<HTMLElement>('[id]')].forEach(x => x.removeAttribute('id'));
  if (!hooks) g.querySelectorAll(HOOKS.map(a => `[${a}]`).join(', ')).forEach(x => HOOKS.forEach(a => x.removeAttribute(a)));
  return g;
}
/** The tile being moved leaves the desk while it travels: it stays in the page, hidden where it stood (G-M-04). */
function liftOut(t: HTMLElement): void {
  const dw = document.getElementById('deskwrap')!.getBoundingClientRect(), r = t.getBoundingClientRect();
  t.style.setProperty('--mx', r.left - dw.left + 'px'); t.style.setProperty('--my', r.top - dw.top + 'px'); t.style.setProperty('--mw', r.width + 'px'); t.classList.add('moving');
}
function ghostFor(id: string): HTMLElement | null {
  if (id.startsWith('move:')) {
    const m = id.slice(5), t = document.querySelector<HTMLElement>(`#bento > .tile[data-m="${m}"]`); if (!t) return null;
    const g = ghostCopy(t.cloneNode(true) as HTMLElement, m, true);
    t.dataset.mo = m; t.removeAttribute('data-m'); liftOut(t);
    return g;
  }
  const d = defOfItem(id); if (!d) return null;
  if (d.kind !== 'module') {
    /* a pedal or side unit: the Pedalboard on the desk glows; else the Pedalboard is previewed, holding it */
    const post = document.querySelector<HTMLElement>(`#bento > .tile[data-m="${BOARD.id}"]`);
    if (post) { post.classList.add('glow'); return null; }
    const g = drawn(BOARD); if (!g) return null;
    const body = g.querySelector('.tile-body'), engine = useDesk.getState().desk.engine, name = sayNow(defName(d, useDesk.getState().desk));
    if (body) body.innerHTML = `<div class="board"><div class="pedals">${d.kind === 'pedal' ? thumbBare(d, engine, name, 'drag') : thumbOf(d, engine, name, 'drag')}</div></div>`;
    return ghostCopy(g, BOARD.id, false);
  }
  const g = drawn(d); return g ? ghostCopy(g, d.id, false) : null;
}

/** Ends a drag: the preview goes, the moved tile is back in the desk's flow, nothing glows or is lifted. */
function clearDrag(): void {
  ghost?.remove(); ghost = null; dropBefore = null; ghostKey = null; dragRects = null;
  document.querySelectorAll<HTMLElement>('.tile[data-mo]').forEach(t => { t.dataset.m = t.dataset.mo; delete t.dataset.mo; });
  document.querySelectorAll('.tile.moving, .tile.glow').forEach(t => t.classList.remove('moving', 'glow'));
  document.querySelectorAll('.lift').forEach(x => x.classList.remove('lift'));
  document.body.classList.remove('dnd');
  if (!usePicker.getState().closing) usePicker.setState({ peek: false });
}

/* the drag image is the thing itself, a touch larger, lifted on the overlay shadow */
function dragImage(e: DragEvent, src: HTMLElement): void {
  const art = src.querySelector<HTMLElement>('.bthumb, .box') ?? src; if (!e.dataTransfer?.setDragImage || !art.getBoundingClientRect().width) return;
  const r = art.getBoundingClientRect(), wrap = document.createElement('div'), c = art.cloneNode(true) as HTMLElement;
  wrap.className = 'drag-img'; wrap.appendChild(c); c.style.width = r.width + 'px'; c.style.height = r.height + 'px'; document.body.appendChild(wrap);
  e.dataTransfer.setDragImage(wrap, (e.clientX - r.left) * 1.05 + 18, (e.clientY - r.top) * 1.05 + 18);
  setTimeout(() => wrap.remove(), 0);
}

document.addEventListener('dragstart', e => {
  const el = (e.target as Element).closest?.<HTMLElement>('[data-drag]'); if (!el || !e.dataTransfer) return;
  const id = el.dataset.drag!; dragId = id;
  e.dataTransfer.setData('text/plain', id); e.dataTransfer.effectAllowed = 'copyMove';
  if (!id.startsWith('move:')) dragImage(e, el);
  setTimeout(() => {
    if (dragId !== id) return;
    if (!id.startsWith('move:')) el.classList.add('lift');
    document.body.classList.add('dnd');
    if (usePicker.getState().open) flushSync(() => usePicker.setState({ peek: true }));
  }, 0);
});
/* a cancelled move: the tile comes back into its place and its neighbours slide aside for it */
document.addEventListener('dragend', () => { const back = document.querySelector('#bento > .tile.moving'); dragId = null; if (back) slideRoom(clearDrag); else clearDrag(); });

const inCentre = (t: EventTarget | null) => !!(t as Element | null)?.closest?.('#centre');
document.addEventListener('dragover', e => {
  if (!dragId || !inCentre(e.target) || !e.dataTransfer) return;
  e.preventDefault(); e.dataTransfer.dropEffect = dragId.startsWith('move:') ? 'move' : 'copy';
  const g = bento(); if (!g) return;
  /* a moved tile leaves the desk first (the others slide into its place); the desk is measured once, with the preview
     in it, so the preview cannot move its own target */
  if (ghostKey !== dragId) { ghostKey = dragId; const id = dragId; if (id.startsWith('move:')) slideRoom(() => { ghost = ghostFor(id); }); else ghost = ghostFor(id); }
  if (!ghost) return;
  const key = ghost.dataset.gm!, tail = () => g.querySelector(':scope > .desk-empty, :scope > .q-empty, :scope > .moretiles, :scope > .parked');
  if (!ghost.isConnected) { const gh = ghost; slideRoom(() => g.insertBefore(gh, tail())); dragRects = null; dropBefore = null; }
  if (!dragRects) { const it = lastMeasure()?.items.find(x => x.key === key); dragRects = it ? deskSlots(g, key, it) : []; }
  /* the preview stands where the pointer is, else at the nearest place; while the pointer stays inside it, it stays */
  const px = e.clientX + scrollX, py = e.clientY + scrollY, inside = (s: Place) => px >= s.x && px <= s.x + s.w && py >= s.y && py <= s.y + s.h;
  const cur = dropBefore && dragRects.find(s => s.k === dropBefore!.k);
  if (cur && inside(cur)) return;
  const dist = (s: Place) => Math.hypot(Math.max(s.x - px, 0, px - s.x - s.w), Math.max(s.y - py, 0, py - s.y - s.h));
  const pick = dragRects.find(inside) ?? [...dragRects].sort((a, b) => dist(a) - dist(b) || a.k - b.k)[0]; if (!pick || pick.k === dropBefore?.k) return;
  const list = deskTiles().filter(x => x !== key && x !== dragKey()), gh = ghost;
  const at = list[pick.k] ? g.querySelector(`:scope > .tile[data-m="${list[pick.k]}"]:not(.ghost)`) : tail();
  dropBefore = { k: pick.k, m: beforeAt(list, pick.k) };
  slideRoom(() => g.insertBefore(gh, at));
});
/* a drag that enters a new element is accepted there too */
document.addEventListener('dragenter', e => { if (dragId && inCentre(e.target)) e.preventDefault(); });
document.addEventListener('dragleave', e => {
  if (!dragId || !inCentre(e.target) || centre()?.contains(e.relatedTarget as Node | null)) return;
  if (ghost?.isConnected) { const gh = ghost; slideRoom(() => gh.remove()); }
  dropBefore = null;
});
document.addEventListener('drop', e => {
  if (!dragId || !inCentre(e.target)) return;
  e.preventDefault();
  const id = dragId, before = dropBefore ? dropBefore.m : null;
  /* the picker goes from its peek; it does not come back first (G-M-03), and focus goes to what arrived */
  if (usePicker.getState().open) closePicker(false);
  clearDrag(); dragId = null;
  const s = useDesk.getState();
  if (id.startsWith('move:')) {
    const m = id.slice(5);
    flushSync(() => { if (m !== before) s.move(m, before, `.tile[data-m="${m}"] .grip`); });
    packNow(make());
    document.querySelector<HTMLElement>(`.tile[data-m="${m}"] .grip`)?.focus({ preventScroll: true });
    return;
  }
  const d = defOfItem(id); if (!d) return;
  let r: ReturnType<typeof s.add> = { did: 'there', step: null };
  flushSync(() => { r = s.add(d.id, before); });
  packNow(make());
  if (r.did !== 'added') return;
  sayParked(d.id, r.step);
  const sel = d.kind === 'module' ? `.tile[data-m="${d.id}"] .grip` : d.kind === 'unit' ? `[data-xid="${d.id}"] .u-pow` : `[data-pid="${d.id}"] .foot, [data-pid="${d.id}"] .px`;
  /* what arrived is brought into view by the shortest scroll, as the concept's flash does (30-desk.js) */
  requestAnimationFrame(() => { const el = document.querySelector<HTMLElement>(sel); el?.focus({ preventScroll: true }); el?.closest('.tile, .pedal')?.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }); });
});
