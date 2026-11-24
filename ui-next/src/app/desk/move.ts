// Moving a tile from the keyboard (the concept's moveTileDir, 30-desk.js): an arrow moves it one visible step that
// way. Up and down by height, left and right within its row, else (at a row's end) the next place in reading order. A
// press that would change nothing visible is refused and said, and nudges the tile, recording no step. A run of presses
// on one grip is one move (the store).

import { defName } from '../../derive.ts';
import { LEAD, defOf } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui } from '../i18n/index.ts';
import { deskSlots, lastMeasure } from './layout.ts';
import { reduced } from '../kit/motion.ts';

type Slot = { k: number; x: number; y: number; w: number; h: number };
const sameSlot = (a: Slot, b: Slot) => Math.abs(a.x - b.x) < 2 && Math.abs(a.y - b.y) < 2;
const tilesNow = () => [...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]')].filter(t => t.dataset.m !== LEAD?.id && !t.classList.contains('moving')).map(t => t.dataset.m!);

function slotToward(now: Slot, slots: Slot[], dir: 'left' | 'right' | 'up' | 'down'): Slot | null {
  const mid = (s: Slot) => [s.x + s.w / 2, s.y + s.h / 2] as const, [cx, cy] = mid(now), moved = slots.filter(s => s.k !== now.k && !sameSlot(s, now));
  const level = (s: Slot) => Math.abs(s.y - now.y) <= 24, nearX = (a: Slot, b: Slot) => Math.abs(mid(a)[0] - cx) - Math.abs(mid(b)[0] - cx);
  let c: Slot[] = [];
  /* up and down: a place whose top is above or below this one's, so the tile's top visibly moves that way (a place in
     the same row where the tile would stand shorter has its middle higher, and is no step up) */
  if (dir === 'up') c = moved.filter(s => mid(s)[1] < cy - 4 && s.y < now.y - 4).sort((a, b) => (mid(b)[1] - mid(a)[1]) || nearX(a, b));
  if (dir === 'down') c = moved.filter(s => mid(s)[1] > cy + 4 && s.y > now.y + 4).sort((a, b) => (mid(a)[1] - mid(b)[1]) || nearX(a, b));
  if (dir === 'left') { c = moved.filter(s => level(s) && mid(s)[0] < cx - 4).sort((a, b) => mid(b)[0] - mid(a)[0]); if (!c.length) c = moved.filter(s => s.k < now.k).sort((a, b) => b.k - a.k); }
  if (dir === 'right') { c = moved.filter(s => level(s) && mid(s)[0] > cx + 4).sort((a, b) => mid(a)[0] - mid(b)[0]); if (!c.length) c = moved.filter(s => s.k > now.k).sort((a, b) => a.k - b.k); }
  return c[0] ?? null;
}

export function moveTile(id: string, dir: 'left' | 'right' | 'up' | 'down'): void {
  const g = document.getElementById('bento'), m = lastMeasure(), it = m?.items.find(x => x.key === id), d = defOf(id);
  if (!g || !it || !d) return;
  const all = tilesNow(), list = all.filter(x => x !== id), slots = deskSlots(g, id, it), now = slots.find(s => s.k === all.indexOf(id));
  if (!now) return;
  const store = useDesk.getState(), name = defName(d, store.desk), to = slotToward(now, slots, dir);
  if (!to) {
    store.say(ui('tile.cantMove', { name, dir: ui(`dir.${dir}`) }));
    const v = { left: [-4, 0], right: [4, 0], up: [0, -4], down: [0, 4] }[dir], el = g.querySelector<HTMLElement>(`:scope > .tile[data-m="${id}"]`);
    if (el && !reduced()) el.animate([{ translate: '0 0' }, { translate: `${v[0]}px ${v[1]}px` }, { translate: '0 0' }], { duration: 160, easing: 'ease-out' });
    return;
  }
  /* the module it goes before, to stand at place k among the other tiles (null: the end) */
  const before = to.k < list.length ? list[to.k]! : null;
  store.move(id, before, `.tile[data-m="${id}"] .grip`);
  requestAnimationFrame(() => {
    const gr = document.querySelector<HTMLElement>(`[data-move="${id}"]`);
    gr?.focus({ preventScroll: true });
    gr?.closest('.tile')?.scrollIntoView({ block: 'nearest' });
    const after = tilesNow();
    useDesk.getState().say(ui('tile.moved', { name, dir: ui(`dir.${dir}`), n: after.indexOf(id) + 1, of: after.length }));
  });
}
