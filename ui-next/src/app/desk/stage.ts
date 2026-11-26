// A tier change staged by what is in view (docs/09, "Rendering"). Drawing a full desk at the new tier, laying it out and
// capturing it (it was a View Transition then) took 100-230 ms before anything moved. When the desk has been laid out at the new
// tier before (taken as that tier was left: desk/layout.ts deskLeaving) and nothing a tile out of view drew there can
// have changed since (what its models read: render/tileTier.ts), the change draws only what will be in view: a tile out
// of view before and after keeps what it was drawn at (a tile the change brings is drawn as its shell), stands in the
// layout as what was kept of it at the new tier, lays nothing out and takes no part in the motion (shell/morph.ts), which no one
// would see; once the motion has started, the held tiles are drawn at the new tier one a frame, nearest first, and each
// is checked against what was kept (a tile that is not the height it was kept at is measured and the desk laid out
// again, as for any change of content). Drawing only: which tiles show, and every value, are the core's as before.

import { flushSync } from 'react-dom';
import type { Tier } from '../../definitions/types.ts';
import { deskModel } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk, type DeskStore } from '../store/desk.ts';
import { deskFold } from '../store/fold.ts';
import { changes } from '../render/reads.ts';
import { heldTiles, holdTiles, readsOfTile, releaseTile, type Hold } from '../render/tileTier.ts';
import { adoptStaged, deskLeaving, deskPlaces, endStage, stagePack, type DeskAt } from './layout.ts';
import type { PackItem } from './pack.ts';
import { tally } from '../tally.ts';

/** What the desk was at a tier when it was left: the store then, the desk's make-up, its layout, and what each tile's
 *  models read. */
interface Left { s: DeskStore; make: string; at: DeskAt; reads: Map<string, Set<string>> }
const left = new Map<Tier, Left>();
const makeOf = (s: DeskStore, tier: Tier) => `${deskModel(ix, { ...s.desk, tier }, s.mods, deskFold.get()).make}|${innerWidth}`;

/** A staged change: the tiles it holds (at their tier, or `later`), what each stands in as, and each one's look. */
export interface Stage { holds: Map<string, Hold>; items: Map<string, PackItem>; looks: Map<string, string> }
let looks = new Map<string, string>(), frame = 0, task = 0;
const tileEl = (id: string) => document.querySelector<HTMLElement>(`#bento > .tile[data-m="${id}"]`);

/** The desk is about to leave its tier: what a change held is drawn first, then what the desk is at this tier is kept. */
export function leaving(): void {
  settle();
  const s = useDesk.getState(), at = deskLeaving();
  if (!at) { left.delete(s.desk.tier); return; }
  const reads = new Map<string, Set<string>>();
  at.items.forEach(it => { const r = it.tile ? readsOfTile(it.key) : null; if (r) reads.set(it.key, r); });
  left.set(s.desk.tier, { s, make: makeOf(s, s.desk.tier), at, reads });
}

/** What a change to `t` can leave undrawn for a moment (null: everything is drawn now): the tiles out of view before and
 *  after it (`reader`: the tile whose place on screen is kept, and its top now), whose layout at `t` was kept and whose
 *  content there cannot have changed. The desk must stand on the same engine, fold and window as when it was left at
 *  `t`; its modules may have changed since (one added, removed or moved). */
export function planStage(t: Tier, reader: { key: string; top: number } | null): Stage | null {
  const L = left.get(t), s = useDesk.getState(), was = L?.make.split('|'), will = makeOf(s, t).split('|');
  if (!L || !was || [1, 3, 4].some(i => was[i] !== will[i])) return null;
  /* where everything will stand: the tiles in the order they now take, each as it was laid out at `t` (one not laid
     out there yet as it is here: it is drawn and measured in the change), what stands before and after them as then */
  const ids = deskModel(ix, { ...s.desk, tier: t }, s.mods, deskFold.get()).tiles, here = left.get(s.desk.tier)?.at.items ?? [];
  const first = L.at.items.findIndex(it => it.tile), last = L.at.items.findLastIndex(it => it.tile);
  const guess = ids.map(id => L.at.items.find(it => it.key === id) ?? here.find(it => it.key === id));
  if (first < 0 || guess.some(it => !it)) return null;
  const places = deskPlaces(L.at, [...L.at.items.slice(0, first), ...guess as PackItem[], ...L.at.items.slice(last + 1)]); if (!places) return null;
  /* what is in view is told from where the desk stood when it was left: within a margin, for what stands above it */
  const d = changes(L.s, s), margin = 120, at = reader && places.get(reader.key);
  /* the page scrolls so that the reader's tile keeps its place on screen */
  const top = (y: number) => y - (at ? at.y - reader!.top : scrollY);
  const seen = (y: number, h: number) => y + h > -margin && y < innerHeight + margin;
  const holds = new Map<string, Hold>(), items = new Map<string, PackItem>(), lk = new Map<string, string>();
  L.at.items.forEach((it, i) => {
    const r = L.reads.get(it.key), p = places.get(it.key);
    if (!it.tile || !L.at.ok[i] || !r || !p || !ids.includes(it.key) || [...r].some(x => d.has(x))) return;
    const el = tileEl(it.key), now = el?.getBoundingClientRect();
    if (seen(top(p.y), p.h) || (now && seen(now.top, now.height))) return;
    holds.set(it.key, el ? s.desk.tier : 'later'); items.set(it.key, it); lk.set(it.key, L.at.looks[i]!);
  });
  return holds.size ? { holds, items, looks: lk } : null;
}

/** Inside the change, before it is drawn: the tiles are held, stand in the layout, and lay nothing out. */
export function hold(st: Stage): void {
  tally('stage.held', st.holds.size);
  looks = st.looks; stagePack(new Map(st.items));
  st.holds.forEach((h, id) => { if (h !== 'later') tileEl(id)?.style.setProperty('content-visibility', 'hidden'); });
  holdTiles(st.holds);
}

/** After the change is laid out: a held tile that stands in view after all is drawn now. */
export function drawInView(): void {
  heldTiles().forEach(id => { const r = tileEl(id)?.getBoundingClientRect(); if (r && r.bottom > 0 && r.top < innerHeight) draw(id); });
}

/** Once the motion has started: the held tiles are drawn, one a frame, nearest the view first. */
export function drawHeld(): void {
  if (frame || task || !heldTiles().length) return;
  /* after a frame's drawing, so that each frame does one tile's */
  const later = () => { frame = requestAnimationFrame(() => { frame = 0; task = window.setTimeout(step); }); };
  const step = () => {
    task = 0;
    const ids = heldTiles(); if (!ids.length) { endStage(); return; }
    const far = (id: string) => { const r = tileEl(id)?.getBoundingClientRect(); return r ? Math.max(0, r.top - innerHeight, -r.bottom) : 0; };
    draw(ids.reduce((a, b) => (far(b) < far(a) ? b : a)));
    if (heldTiles().length) later(); else endStage();
  };
  later();
}

function draw(id: string): void {
  tally('stage.drawn');
  /* shown before it is drawn, so that what its parts measure as they are drawn (the board's rows) is measured */
  tileEl(id)?.style.removeProperty('content-visibility');
  flushSync(() => releaseTile(id));
  const el = tileEl(id); if (el) adoptStaged(el, looks.get(id) ?? '');
}

/** Every held tile is drawn now (before the desk leaves its tier, or when its make-up changes). */
export function settle(): void {
  cancelAnimationFrame(frame); clearTimeout(task); frame = task = 0;
  const ids = heldTiles();
  if (ids.length) { tally('stage.settled', ids.length); ids.forEach(id => tileEl(id)?.style.removeProperty('content-visibility')); flushSync(() => ids.forEach(releaseTile)); ids.forEach(id => { const el = tileEl(id); if (el) adoptStaged(el, looks.get(id) ?? ''); }); }
  endStage();
}

/* a module, the board or the engine that changes while tiles are held: they are drawn first (after the change's own
   drawing, which lays them out as they stand in) */
useDesk.subscribe((s, p) => { if ((s.mods !== p.mods || s.board !== p.board || s.desk.engine !== p.desk.engine) && heldTiles().length) queueMicrotask(settle); });
deskFold.subscribe(() => { if (heldTiles().length) queueMicrotask(settle); });
