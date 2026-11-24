// The desk's own transition (the concept's transition(), 30-desk.js, M-02): structural changes animate only what
// changed. When the modules, the board, the engine or (without a View Transition) the tier change, the store tells this
// file before the page is drawn again; where every tile and pedal stood is taken then, and once the desk has been laid
// out (desk/layout.ts onPacked) each is compared with where it is now:
// - a tile that arrived scales in (--t-base); one that left fades out where it stood (--t-fast), its node put back for
//   the moment in a layer of its own; one that moved slides from its old place to its new one (--t-move);
// - a tile dropped from a drag starts from its preview's box and settles into its place, lifted a little with its shadow
//   dropping (M-13, G-M-03, G-M-04): it never shows at the place it came from;
// - a pedal that joins a board drops onto it (--t-land, the spring) and its LED blooms; one that leaves lifts off; the
//   others slide; the new cables draw out of their jacks and every cable follows the pedals frame by frame;
// - under reduced motion nothing moves: what arrives fades in over 120ms and an arriving tile carries the still ring
//   (S-27); what leaves fades out.
// React draws the new desk as it always does; this only reads where things stood and plays the difference. A scene the
// test layer sets is drawn as it is (still()). Drawing only: places and times, no value.

import { useDesk } from '../store/desk.ts';
import { BOARD } from '../model/index.ts';
import { onPacked } from '../desk/layout.ts';
import { drawOut, followCables } from '../board/cables.ts';
import { booted, easing, mtime, reduced } from '../kit/motion.ts';

type At = { el: HTMLElement; r: DOMRect };
type Snap = { tiles: Map<string, At>; peds: Map<string, At>; added: boolean };
const TILES = '#bento > .tile[data-m]', PEDALS = '[data-pedals] > .pedal[data-pid]';
const board = () => document.querySelector<HTMLElement>('#bento > .tile[data-m] .board');
const rects = (sel: string, key: (el: HTMLElement) => string) => new Map([...document.querySelectorAll<HTMLElement>(sel)].map(el => [key(el), { el, r: el.getBoundingClientRect() }]));
const snap = (added: boolean): Snap => ({ tiles: rects(TILES, el => el.dataset.m!), peds: rects(`#bento ${PEDALS}`, el => el.dataset.pid!), added });

let pending: Snap | null = null;
/* the tile a drop lands, and the preview's box it lands from */
let landing: { id: string; from: DOMRect } | null = null;

/** The node that left, put back for a moment exactly where it stood, in a layer of its own over `host`, then faded:
 *  it is the node React let go of, inert, hidden from screen readers and without ids (no id is on the page twice). */
function exitGhost(old: HTMLElement, r: DOMRect, host: HTMLElement, cls: string): HTMLElement {
  const hr = host.getBoundingClientRect(), layer = document.createElement('div');
  layer.className = cls; layer.setAttribute('aria-hidden', 'true'); layer.inert = true;
  [old, ...old.querySelectorAll<HTMLElement>('[id]')].forEach(x => x.removeAttribute('id'));
  old.style.removeProperty('view-transition-name');
  old.style.cssText += `;position:absolute;margin:0;left:${r.left - hr.left}px;top:${r.top - hr.top}px;width:${r.width}px;height:${r.height}px;transform:none`;
  layer.appendChild(old); host.appendChild(layer); return layer;
}

function play(): void {
  const was = pending; if (!was) return; pending = null;
  const calm = reduced(), out = easing('--e-out'), inn = easing('--e-in'), spring = easing('--e-spring'), land = landing; landing = null;
  const seen = new Set<string>();
  document.querySelectorAll<HTMLElement>(TILES).forEach(t => {
    const m = t.dataset.m!, b = was.tiles.get(m); seen.add(m);
    if (land && m === land.id) {
      if (calm) { t.animate([{ opacity: 0.4 }, { opacity: 1 }], { duration: 120 }); if (was.added) ringStill(t, 'flash'); return; }
      const r = t.getBoundingClientRect(), dx = land.from.left - r.left, dy = land.from.top - r.top, far = Math.hypot(dx, dy) > 16;
      t.animate([{ transform: `translate(${dx}px, ${dy}px) scale(1.02)`, boxShadow: '0 18px 44px #1b172229' }, { transform: 'none', boxShadow: '0 0 0 transparent' }], { duration: far ? mtime('--t-move') : mtime('--t-land'), easing: far ? out : spring });
      return;
    }
    if (!b) {
      t.animate(calm ? [{ opacity: 0 }, { opacity: 1 }] : [{ opacity: 0, transform: 'scale(.96)' }, { opacity: 1, transform: 'none' }], { duration: calm ? 120 : mtime('--t-base'), easing: out, fill: 'backwards' });
      /* under reduced motion an arriving tile has no entrance of its own, so it carries the still ring (S-27) */
      if (calm && was.added) ringStill(t, 'flash');
      if (m === BOARD.id) followCables(board, mtime('--t-base') + 80);
      return;
    }
    if (calm) return;
    const r = t.getBoundingClientRect(), dx = b.r.left - r.left, dy = b.r.top - r.top;
    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) t.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], { duration: mtime('--t-move'), easing: out });
  });
  const wrap = document.getElementById('deskwrap');
  was.tiles.forEach((b, m) => {
    if (seen.has(m) || b.el.isConnected || !wrap) return;
    const layer = exitGhost(b.el, b.r, wrap, 'bento exit-layer');
    b.el.animate(calm ? [{ opacity: 1 }, { opacity: 0 }] : [{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'scale(.97)' }], { duration: calm ? 120 : mtime('--t-fast'), easing: inn, fill: 'forwards' }).finished.then(() => layer.remove(), () => layer.remove());
  });
  playPedals(was, calm, out, inn, spring);
}

/** The pedals on the board: one that joins drops onto it, one that leaves lifts off, the others slide; the cables follow. */
function playPedals(was: Snap, calm: boolean, out: string, inn: string, spring: string): void {
  const pd = document.querySelector<HTMLElement>(`#bento [data-pedals]`);
  if (!pd || !(was.peds.size || pd.querySelector('.pedal[data-pid]'))) return;
  const seen = new Set<string>(); let moved = false;
  pd.querySelectorAll<HTMLElement>(':scope > .pedal[data-pid]').forEach(p => {
    const id = p.dataset.pid!, b = was.peds.get(id); seen.add(id);
    if (!b) {
      if (!was.peds.size) return;
      moved = true; p.classList.add('landing');
      p.animate(calm ? [{ opacity: 0 }, { opacity: 1 }] : [{ opacity: 0, transform: 'translateY(-10px) scale(1.03)', filter: 'drop-shadow(0 14px 12px #0004)' }, { opacity: 1, transform: 'none', filter: 'drop-shadow(0 0 0 #0000)' }], { duration: calm ? 120 : mtime('--t-land'), easing: calm ? out : spring, fill: 'backwards' })
        .finished.then(() => { p.classList.remove('landing'); ringStill(p, 'flash', calm ? 1200 : 700); }, () => {});
      return;
    }
    if (calm) return;
    const r = p.getBoundingClientRect(), dx = b.r.left - r.left, dy = b.r.top - r.top;
    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) { moved = true; p.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], { duration: mtime('--t-move'), easing: out }); }
  });
  const host = pd.closest<HTMLElement>('.board');
  was.peds.forEach((b, id) => {
    if (seen.has(id) || !host || b.el.isConnected || !pd.querySelector('.pedal[data-pid]')) return;
    moved = true;
    const layer = exitGhost(b.el, b.r, host, 'pedals pedal-exit');
    b.el.animate(calm ? [{ opacity: 1 }, { opacity: 0 }] : [{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'translateY(-8px) scale(.97)' }], { duration: calm ? 120 : mtime('--t-fast'), easing: inn, fill: 'forwards' }).finished.then(() => layer.remove(), () => layer.remove());
  });
  if (moved) { if (!calm) drawOut(); followCables(board, Math.max(mtime('--t-land'), mtime('--t-move')) + 80); }
}

/** A class that rings an element for a moment (the tile's and the pedal's flash: a fading ring, or the still outline
 *  under reduced motion). */
function ringStill(el: HTMLElement, cls: string, ms = reduced() ? 1200 : mtime('--t-slow') * 2): void {
  el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls);
  window.setTimeout(() => el.classList.remove(cls), ms);
}

/** Whether the desk's own transition plays: not while the page loads, nor inside a tier change's View Transition. */
const moving = () => booted() && !document.documentElement.classList.contains('tier-morph');

/* the store says the desk is about to change its make-up: where everything stands is taken now, while the page still
   shows it, and played once the desk is laid out (or on the next frame, for a change that lays nothing out) */
useDesk.subscribe((s, p) => {
  const mods = s.mods.join() !== p.mods.join(), board = s.board.join() !== p.board.join();
  if (!mods && !board && s.desk.tier === p.desk.tier && s.desk.engine === p.desk.engine) return;
  const added = mods && s.mods.some(m => !p.mods.includes(m));
  if (pending) { if (added) pending.added = true; return; }
  if (!moving()) return;
  pending = snap(added);
  requestAnimationFrame(play);
});
onPacked(play);

/** Runs fn, which changes the desk's make-up some other way than through the store (a fold opened), as the desk's own
 *  transition: what it adds, removes or moves animates as above. */
export function transition(fn: () => void): void {
  if (!pending && moving()) { pending = snap(false); requestAnimationFrame(play); }
  fn();
}
/** A scene the test layer sets is drawn as it is, as the concept's render() draws it: what was taken is dropped. */
export const still = (): void => { pending = null; landing = null; };

/* a drop: the desk is taken as the user sees it, the preview standing where the tile will go (before the drop clears
   it); a module lands from the preview's box, a pedal or unit lands on the board */
document.addEventListener('drop', e => {
  const g = document.querySelector<HTMLElement>('#bento > .tile.ghost'), id = e.dataTransfer?.getData('text/plain') ?? '';
  if (!g?.isConnected || !moving()) return;
  const pedal = id.includes(':');
  pending = snap(!pedal && !id.startsWith('move:')); requestAnimationFrame(play);
  if (!pedal) landing = { id: id.startsWith('move:') ? id.slice(5) : id, from: g.getBoundingClientRect() };
}, true);
