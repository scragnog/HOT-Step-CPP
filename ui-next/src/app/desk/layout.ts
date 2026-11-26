// The desk's layout on the page: measuring what each tile needs (the concept's deskMeasure) and placing the tiles
// where the pure layout (pack.ts) puts them (packNow), with the concept's rule for when the widths are kept: a change
// that leaves the desk's make-up alone (the same tiles, order, tier, engine and window width: a value set, a line
// appearing in a tile) keeps every tile's width and column (Q-29); any other change chooses afresh. 11-render.js.
// A re-pack that a tile's content set off slides the tiles it moves (the concept's packDesk, M-02); a change to the
// desk's make-up is the desk's own transition's (features/motion.ts), told when the desk has been laid out (onPacked).
//
// What a tile needs (its natural width and its height at each width it may take) is measured and kept, and the desk is
// measured again only when a tile's can have changed (docs/09, "Rendering"): a tile new on the desk, one whose classes
// or layout attributes changed, one whose markup changed inside (a MutationObserver), one whose body changed size while
// the layout had not moved it (a ResizeObserver), the desk's columns or width, a web font that arrived. Measuring is the
// concept's, whole (every tile cleared, each width set in turn, the heights read); a layout is solved once for the same
// tiles (a small cache), and only what changed is written, so a re-pack that changes nothing touches nothing and a
// removal, a move or a drag measures nothing.

import { easing, mtime, reduced, booted } from '../kit/motion.ts';
import { deskPlace, deskSettle, deskSolve, deskSolveAt, PACK_GAP, PACK_ROW, PACK_STRETCH, type Layout, type PackItem, type Plan, type Rect } from './pack.ts';
import { BOARD, LEAD } from '../model/index.ts';
import { tally } from '../tally.ts';

interface Measured { cols: number; tiles: HTMLElement[]; items: PackItem[] }
let packMake = '', packFresh = 0;
const state = new WeakMap<HTMLElement, { sig: string; plan: Plan[] } | null>();
let meas: Measured | null = null;
/* where each child was last placed (its key's rectangle in the layout) */
const placedAt = new Map<string, Rect>();
/* while a tier change is staged: what each tile it holds needs at the new tier, by key (stage.ts) */
let staged: Map<string, PackItem> | null = null;
const packedFns: (() => void)[] = [];
/** Runs fn after the desk (or the board) has been laid out, once the drawing that set it off is complete. */
export const onPacked = (fn: () => void): void => { packedFns.push(fn); };
/** The desk or the board was laid out: what waits for it runs once the commit that drew it is over. */
export const packed = (): void => { queueMicrotask(() => packedFns.forEach(f => f())); };
/** The last measurement of the desk, for the keyboard moves (deskSlots). */
export const lastMeasure = (): Measured | null => meas;

/** Runs a layout-measuring step without the page height or scroll position moving, even for an instant. */
export function holdScroll(fn: () => void): void {
  const html = document.documentElement, x = scrollX, y = scrollY, a = html.style.overflowAnchor;
  html.style.overflowAnchor = 'none'; html.style.minHeight = html.scrollHeight + 'px';
  try { fn(); } finally { html.style.minHeight = ''; html.style.overflowAnchor = a; if (scrollX !== x || scrollY !== y) scrollTo(x, y); }
}

const packKey = (t: HTMLElement) => t.dataset.m || t.dataset.gm || (t.classList.contains('desk-empty') ? '_empty' : t.className);
/** The tier the desk is drawn at (the centre column's class), and a tile's kept measurement's key at it: what a tile
 *  needs is kept for each tier it has been drawn at. */
const tierOf = (g: HTMLElement) => /\bt-(\w+)/.exec(g.closest('.centre')?.className ?? '')?.[1] ?? '';
const keptKey = (t: HTMLElement, tier: string) => `${packKey(t)}|${tier}`;

/* ---------------------------------------------------------------- what is kept of each tile's measurement */

/** A tile's measurement: its natural width and its height (in rows) at each width measured, for the desk's columns and
 *  width (`geo`) and the tile's own look (its classes and layout attributes) when it was measured. */
interface Kept { el: HTMLElement; look: string; geo: string; nat: number; hs: Map<number, number> }
const kept = new WeakMap<HTMLElement, Map<string, Kept>>();
/** Tiles whose content changed since they were measured. */
let stale = new WeakSet<HTMLElement>();
/** Tiles moved or restyled since the ResizeObserver last spoke (the layout placed them, a drag lifted them, a tier change
 *  named them): a change of their body's size is that, not their content's. */
let placed = new WeakSet<HTMLElement>();
/** What the layout last wrote on each tile. */
const wrote = new WeakMap<HTMLElement, string>();
/* the classes a tile carries for a moment, or the layout's own: none changes what it measures */
const PASSING = new Set(['stretched', 'widened', 'narrowed', 'flash', 'here', 'glow', 'moving', 'ghost', 'lift', 'landing', 'ring-here', 'offscreen', 'turning']);
const lookOf = (t: HTMLElement) => [...t.classList].filter(c => !PASSING.has(c)).sort().join(' ') + '|' + ['m', 'grow', 'fold', 'span', 'stretch', 'fill'].map(k => t.dataset[k] ?? '-').join('|');
/** The desk's child that holds n. */
const tileOf = (g: HTMLElement, n: Node | null): HTMLElement | null => { while (n && n.parentNode !== g) n = n.parentNode; return n instanceof HTMLElement ? n : null; };

const observed = new WeakMap<HTMLElement, MutationObserver>();
/* a change inside a tile (React drew it again, a part set a size) marks the tile. Not: a change of the tile's own
   attributes (the layout's style and classes, a drag's marks: what of them a measurement depends on is the tile's look,
   lookOf); the desk's children coming and going (a new element is measured as new); the board's cables, drawn over it;
   a style set back to what it was (by the first of the records about it: a box measured at another size for a moment
   and set back leaves two), or changed only in what names it for the tier change (view-transition-*) */
const VT_DECL = /view-transition-[a-z]+:[^;]*;?/g;
const sameStyle = (a: string | null, b: string | null) => (a ?? '').replace(VT_DECL, '').trim() === (b ?? '').replace(VT_DECL, '').trim();
function noteMutations(g: HTMLElement, rs: MutationRecord[]): void {
  const was = new Map<Node, string | null>();
  for (const r of rs) if (r.type === 'attributes' && r.attributeName === 'style' && !was.has(r.target)) was.set(r.target, r.oldValue);
  for (const r of rs) {
    if (r.target === g) continue;
    const t = tileOf(g, r.target); if (!t) continue;
    if (r.type === 'attributes' && r.target === t) { placed.add(t); continue; }
    if (r.type === 'attributes' && r.attributeName === 'style' && sameStyle(was.get(r.target)!, (r.target as Element).getAttribute('style'))) continue;
    if (stale.has(t) || (r.target as Element).closest?.('.cables') || r.target.parentElement?.closest('.cables')) continue;
    stale.add(t);
  }
}
function watchContent(g: HTMLElement): void {
  if (observed.has(g)) { noteMutations(g, observed.get(g)!.takeRecords()); return; }
  const mo = new MutationObserver(rs => noteMutations(g, rs));
  mo.observe(g, { childList: true, subtree: true, attributes: true, attributeOldValue: true, characterData: true });
  observed.set(g, mo);
}
/* a web font that arrives changes what every tile needs (one that could not load changes nothing) */
document.fonts?.addEventListener?.('loadingdone', e => { if ((e as FontFaceSetLoadEvent).fontfaces.length) { stale = new WeakSet(); fontGen++; } });
let fontGen = 0;

/** Measures every tile as the concept's deskMeasure does (each tile's natural width, then its height at each width it
 *  may take, every tile of a width set at once), and keeps what each needs. Every tile is measured when one must be:
 *  what one needs can hang on where the others stand (a board's rows are balanced to the pixel, and the desk's columns
 *  share out their fractions of a pixel by what is placed in them), so a tile is only ever measured as the concept
 *  measures it, with every tile cleared at once. */
function measureAll(g: HTMLElement, need: HTMLElement[], cols: number, widths: (t: HTMLElement, nat: number) => number[], geo: string, ks: Map<string, Kept>, stand: [HTMLElement, number[]][] = []): void {
  tally('measure'); tally('measure.tiles', need.length);
  const tier = tierOf(g), rows = (h: number) => Math.max(1, Math.ceil((h + PACK_GAP) / PACK_ROW));
  need.forEach(t => { t.style.gridColumn = ''; t.style.gridRow = ''; t.classList.remove('stretched', 'widened', 'narrowed'); wrote.delete(t); });
  /* a tile a staged tier change holds stands in at the widths it takes at the new tier, its content not laid out
     (stage.ts): the others are placed in the flow exactly as when every tile is measured */
  const span = (j: number) => stand.forEach(([t, ws]) => { t.style.gridColumn = 'span ' + (ws[j] ?? ws[0]); });
  stand.forEach(([t]) => { t.style.gridRow = ''; wrote.delete(t); }); span(0);
  const end = measuring(g);
  const nat = need.map(t => { const cs = getComputedStyle(t), e = cs.gridColumnEnd, sp = [cs.gridColumnStart, e].find(v => /^span/.test(v)); return e === '-1' ? cols : sp ? Math.min(cols, parseInt(sp.slice(5)) || 1) : 1; });
  const ws = need.map((t, i) => widths(t, nat[i]!));
  tally('measure.layouts');
  const hs = need.map(t => [rows(t.offsetHeight)]);
  for (let j = 1; j < 3; j++) {
    const set = need.filter((_, i) => ws[i]![j]);
    if (!set.length) break;
    tally('measure.layouts');
    need.forEach((t, i) => { const o = ws[i]![j]; if (o) { t.style.gridColumn = 'span ' + o; t.classList.toggle('narrowed', o < nat[i]!); t.classList.toggle('widened', o > nat[i]!); } });
    span(j);
    need.forEach((t, i) => { if (ws[i]![j]) hs[i]![j] = rows(t.offsetHeight); });
    set.forEach(t => { t.style.gridColumn = ''; t.classList.remove('narrowed', 'widened'); });
    span(0);
  }
  end();
  need.forEach((t, i) => { ks.set(keptKey(t, tier), { el: t, look: lookOf(t), geo, nat: nat[i]!, hs: new Map(ws[i]!.map((w, j) => [w, hs[i]![j]!])) }); stale.delete(t); });
}

/** Stands the desk's children as the concept's `.bento.measuring > *` does (each at its own height, its row end free)
 *  until the function it returns puts back what they had. Written on the children, not as the class on the desk: a class
 *  on the desk restyles everything in it (its :has rules), 14 ms each way on a full desk. */
function measuring(g: HTMLElement): () => void {
  const kids = [...g.children].filter((k): k is HTMLElement => k instanceof HTMLElement);
  const had = kids.map(k => [k.style.getPropertyValue('align-self'), k.style.getPropertyPriority('align-self'), k.style.getPropertyValue('grid-row-end'), k.style.getPropertyPriority('grid-row-end')] as const);
  kids.forEach(k => { k.style.setProperty('align-self', 'start'); k.style.setProperty('grid-row-end', 'auto', 'important'); });
  return () => kids.forEach((k, i) => {
    const [a, ap, e, ep] = had[i]!;
    if (a) k.style.setProperty('align-self', a, ap); else k.style.removeProperty('align-self');
    if (e) k.style.setProperty('grid-row-end', e, ep); else k.style.removeProperty('grid-row-end');
  });
}

/** The widths a tile may take (its own first), as the concept's deskMeasure offers them. */
function widthsOf(t: HTMLElement, w: number, cols: number, first: HTMLElement | undefined): number[] {
  const o = [w];
  if ('grow' in t.dataset && w < cols && (w === 1 ? cols > 2 : cols === 3)) o.push(w + 1);
  if ('grow' in t.dataset && w === 2 && cols === 4) o.push(4);
  if ('fold' in t.dataset && cols === 3 && w === 2) o.push(1);
  if ('span' in t.dataset && w < cols && !o.includes(cols)) o.push(cols);
  if (t.dataset.m === LEAD?.id && (cols === 3 || cols === 4) && w === 2) o.push(cols);
  if (t.dataset.m === BOARD.id && cols === 4 && w === 4 && first?.dataset.m === LEAD?.id) o.push(2);
  return o;
}

/* the desk's columns, read when its width or its tier is new: the resolved grid-template-columns counts the columns
   the tiles' places add too, so they are read with every tile's place cleared (the concept reads them so), and kept
   while the width and the tier stay */
const grids = new WeakMap<HTMLElement, Map<string, number>>();
const atOf = (g: HTMLElement, width: number) => `${width}|${g.closest('.centre')?.className ?? ''}`;
const tilesOf = (g: HTMLElement) => [...g.children].filter((t): t is HTMLElement => t instanceof HTMLElement && !t.classList.contains('moving'));

/** What every tile needs, measuring only those whose measurement cannot be kept, handed to `place`; while tiles stand
 *  somewhere else for a moment (the columns read, tiles measured) until `place` has put them back, the page's height and
 *  scroll are held. */
function measure(g: HTMLElement, width: number, place: (m: Measured) => void): void {
  const tiles = tilesOf(g), at = atOf(g, width), ks = kept.get(g) ?? new Map<string, Kept>(), tier = tierOf(g), seen = grids.get(g) ?? new Map<string, number>();
  kept.set(g, ks); grids.set(g, seen); watchContent(g);
  let cols = seen.get(at) ?? 0;
  const widths = (t: HTMLElement, nat: number) => widthsOf(t, nat, cols, tiles[0]);
  /* the tiles a staged tier change holds: what each needs at the new tier is what was kept of it there */
  const standIn = (t: HTMLElement) => staged?.get(packKey(t)), held = tiles.filter(t => standIn(t));
  /* a preview (a moved tile's copy) is the tile it copies: its measurement is the tile's (the drag measures nothing) */
  const usable = (t: HTMLElement, k: Kept | undefined, geo: string): k is Kept => !!k && k.geo === geo && k.look === lookOf(t) && !stale.has(t) && !stale.has(k.el)
    && (k.el === t || (t.classList.contains('ghost') && k.el.classList.contains('moving') && k.el.dataset.mo === t.dataset.gm));
  const needOf = (geo: string) => tiles.filter(t => { if (standIn(t)) return false; const k = ks.get(keptKey(t, tier)); return !usable(t, k, geo) || widths(t, k.nat).some(w => !k.hs.has(w)); });
  let need = cols ? needOf(`${cols}|${width}|${fontGen}`) : tiles;
  const items = (): Measured => ({
    cols, tiles,
    items: tiles.map((t, i) => {
      const s = standIn(t); if (s) return s;
      const k = ks.get(keptKey(t, tier))!, ws = widths(t, k.nat);
      return {
        key: packKey(t), opts: ws.map(w => ({ w, h: k.hs.get(w)! })),
        reach: t.dataset.stretch ? Math.round(+t.dataset.stretch / PACK_ROW) : PACK_STRETCH,
        fill: t.classList.contains('desk-empty') ? 2 : t.dataset.m === LEAD?.id ? 1 : 0,
        tile: !!(t.dataset.m || t.dataset.gm) && t.dataset.m !== LEAD?.id, lead: i === 0 && t.dataset.m === LEAD?.id,
      };
    }),
  });
  if (!need.length) { place(items()); return; }
  holdScroll(() => {
    g.style.minHeight = g.offsetHeight + 'px';
    try {
      if (!cols) { tiles.forEach(t => { t.style.gridColumn = ''; t.style.gridRow = ''; wrote.delete(t); }); cols = getComputedStyle(g).gridTemplateColumns.split(' ').length; seen.set(at, cols); need = needOf(`${cols}|${width}|${fontGen}`); }
      if (need.length) measureAll(g, tiles.filter(t => !standIn(t)), cols, widths, `${cols}|${width}|${fontGen}`, ks, held.map(t => [t, standIn(t)!.opts.map(o => o.w)]));
      place(items());
    } finally { g.style.minHeight = ''; }
  });
}

/* ---------------------------------------------------------------- the layout, solved once for the same tiles */

const solved = new Map<string, Layout>();
const itemsKey = (C: number, items: PackItem[]) => C + '#' + items.map(it => `${it.key}:${it.opts.map(o => o.w + 'x' + o.h).join(',')}:${it.reach}:${it.fill}:${+it.tile}${+it.lead}`).join(';');
const remember = (k: string, l: Layout) => { solved.delete(k); solved.set(k, l); if (solved.size > 48) solved.delete(solved.keys().next().value!); return l; };
/** deskSolve, kept for the same columns and tiles (the arrow keys' places, the pack after a move, a re-pack that finds
 *  nothing changed). */
const solve = (C: number, items: PackItem[]): Layout => { const k = itemsKey(C, items), l = solved.get(k); if (l) return remember(k, l); tally('solve'); return remember(k, deskSolve(C, items)); };

/** The desk's make-up: what, when it changes, makes the layout choose its widths afresh. */
export const deskMake = (tier: string, engine: string, mods: string[], qAll: boolean) => [tier, engine, mods.join(), qAll, innerWidth].join('|');

/** A render: note whether the make-up changed (a fresh layout), then place the tiles (or, while `deferPack` runs,
 *  leave them for flushPack). */
export function renderPack(make: string): void {
  if (make !== packMake) packFresh = performance.now();
  packMake = make;
  if (deferring) { deferred = make; return; }
  packNow(make);
}
/* the make-up a render left for flushPack */
let deferring = false, deferred: string | null = null;
/** Runs fn (a drawing of the desk) with the layout left for flushPack: the desk is then measured once, after everything
 *  the drawing set off has run (the parts' own effects, which can change a tile's content), not before them and again
 *  after (a tier change). */
export function deferPack(fn: () => void): void {
  deferring = true;
  try { fn(); } finally { deferring = false; }
}

/** Places every tile. `keep`: keep the widths chosen for this same desk (a content change). Writes only what changed. */
export function packNow(make: string, keep = make === packMake && performance.now() - packFresh > 700): void {
  tally('pack');
  document.querySelectorAll<HTMLElement>('.bento').forEach(g => {
    if (getComputedStyle(g).display !== 'grid') return;
    if (!g.offsetParent) { if (state.get(g)) packFresh = performance.now(); state.set(g, null); return; }
    const width = g.clientWidth;
    measure(g, width, m => placeAll(g, m, width, keep));
  });
  packed();
}

/** Places the tiles where the layout puts them, writing only what changed. */
function placeAll(g: HTMLElement, m: Measured, width: number, keep: boolean): void {
  const sig = width + '|' + m.cols + '|' + m.items.map(it => it.key).join('|');
  const was = state.get(g);
  let res: Layout;
  if (keep && was && was.sig === sig) res = { ...deskSettle(m.cols, m.items, deskPlace(m.cols, m.items, was.plan)), plan: was.plan };
  else res = solve(m.cols, m.items);
  state.set(g, { sig, plan: res.plan }); meas = m;
  res.out.forEach(p => { const it = m.items[p.i]; if (it) placedAt.set(it.key, p); });
  m.tiles.forEach((t, i) => {
    const p = res.out.find(r => r.i === i); if (!p) return; const w0 = m.items[i]!.opts[0]!.w;
    const col = `${p.c + 1} / span ${p.w}`, row = `${p.y + 1} / span ${p.h}`, st = p.h > p.h0, wi = p.w > w0, na = p.w < w0;
    /* what React drew again keeps the layout's classes only if it kept the class attribute: they are checked, not assumed */
    if (wrote.get(t) === col + '|' + row && t.classList.contains('stretched') === st && t.classList.contains('widened') === wi && t.classList.contains('narrowed') === na) return;
    wrote.set(t, col + '|' + row); placed.add(t);
    t.style.gridColumn = col; t.style.gridRow = row;
    t.classList.toggle('stretched', st); t.classList.toggle('widened', wi); t.classList.toggle('narrowed', na);
  });
}

/** A content change (a tile's body changed size, the window resized): re-place, keeping widths while the make-up holds. */
let raf = 0, pend: [string, boolean] = ['', false];
export function packSoon(make: () => string, content: boolean): void {
  const now = make();
  if (!content && now !== packMake) { packFresh = performance.now(); }
  /* a resize that leaves the width alone (the desk's make-up the same) keeps the widths too, as the concept's packDesk
     does: the same desk at the same width gives the same layout (RD-08) */
  const keep = content || now === packMake ? performance.now() - packFresh > 700 : false;
  packMake = now;
  pend = [now, keep];
  if (!raf) raf = requestAnimationFrame(() => { raf = 0; slidePack(() => packNow(...pend)); });
}
/** A re-pack waiting for the next frame runs now (a tier change lays the desk out inside its transition). */
export function flushPack(): void {
  const make = deferred; deferred = null;
  if (raf) { cancelAnimationFrame(raf); raf = 0; if (make === null) { packNow(...pend); return; } }
  if (make !== null) packNow(make);
}
/** The desk laid out now for what its tiles hold now (a tile that grew since the last pack, before the ResizeObserver
 *  has said so): the desk's own transition measures where things end only once they are there. */
export function settlePack(): void { if (raf) { flushPack(); return; } if (packMake) packNow(packMake); }

/** Watches the desk's tiles for a change of size: a tile's body that changed size while the layout had not moved it
 *  re-places the desk (`now` is the desk's make-up), as does the window. */
export function watchDesk(g: HTMLElement, now: () => string): () => void {
  const ro = new ResizeObserver(es => {
    for (const e of es) { const t = tileOf(g, e.target); if (t && !placed.has(t)) stale.add(t); }
    placed = new WeakSet();
    packSoon(now, true);
  });
  const watch = () => g.querySelectorAll('.tile-body').forEach(b => ro.observe(b));
  watch(); const mo = new MutationObserver(watch); mo.observe(g, { childList: true });
  const rs = () => packSoon(now, false);
  addEventListener('resize', rs);
  return () => { ro.disconnect(); mo.disconnect(); removeEventListener('resize', rs); };
}

/** A content re-pack: the tiles it moves slide from where they stood over --t-base (the concept's packDesk); not while
 *  a drag makes room its own way, a tier change runs, or a tile is already moving. */
function slidePack(fn: () => void): void {
  const still = !booted() || reduced() || document.body.classList.contains('dnd') || document.documentElement.classList.contains('tier-morph');
  const tiles = () => [...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]')];
  const was = still ? null : new Map(tiles().map(t => [t.dataset.m!, t.getBoundingClientRect()]));
  fn();
  if (!was) return;
  /* every box is read before any animation starts: one layout, not one a tile */
  tiles().map(t => [t, t.getBoundingClientRect()] as const).forEach(([t, r]) => {
    const b = was.get(t.dataset.m!); if (!b || t.getAnimations().some(a => a.playState === 'running')) return;
    const dx = b.left - r.left, dy = b.top - r.top;
    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) t.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], { duration: mtime('--t-base'), easing: easing('--e-out') });
  });
}

/** Where a tile would stand at each place among the others (index k is a place among the desk's tiles, not before the
 *  lyric sheet), in page pixels, for the arrow keys: the layout is a pure function of the order, so each place is
 *  exactly what a move there gives. */
export function deskSlots(g: HTMLElement, key: string, item: PackItem): { k: number; x: number; y: number; w: number; h: number }[] {
  const m = meas; if (!m) return [];
  const rest = m.items.filter(it => it.key !== key);
  const lead = rest[0]?.lead ? 1 : 0, last = rest.reduce((a, it, i) => (it.tile ? i + 1 : a), lead);
  const gr = g.getBoundingClientRect(), colW = (g.clientWidth - (m.cols - 1) * PACK_GAP) / m.cols, out = [];
  /* each place's layout, the steps the orders share taken once; kept, so the pack after a move solves nothing */
  const all = deskSolveAt(m.cols, rest, item, lead, last); tally('solve', all.length);
  for (let k = lead; k <= last; k++) {
    const items = [...rest.slice(0, k), item, ...rest.slice(k)], res = remember(itemsKey(m.cols, items), all[k - lead]!), p = res.out.find(r => r.i === k)!;
    out.push({ k: k - lead, x: gr.left + scrollX + p.c * (colW + PACK_GAP), y: gr.top + scrollY + p.y * PACK_ROW, w: p.w * colW + (p.w - 1) * PACK_GAP, h: p.h * PACK_ROW - PACK_GAP });
  }
  return out;
}

/* ---------------------------------------------------------------- a tier change staged by what is in view (stage.ts) */

/** The desk as it stood at a tier, taken as that tier is left: what the layout had of each child (its item and look,
 *  and whether that still held), the columns, and where the desk and its columns stood on the page. Enough to lay the
 *  desk out at that tier again without drawing what will be out of view. */
export interface DeskAt { at: string; cols: number; geo: string; items: PackItem[]; looks: string[]; ok: boolean[]; box: { left: number; top: number; colW: number } }

/** The desk as it is now, about to be left for another tier (null: no layout of it is kept). */
export function deskLeaving(): DeskAt | null {
  const g = document.getElementById('bento'), m = meas, ks = g && kept.get(g);
  if (!g || !m || !ks || staged || !g.offsetParent || m.tiles.length !== m.items.length) return null;
  const width = g.clientWidth, at = atOf(g, width), tier = tierOf(g), geo = `${m.cols}|${width}|${fontGen}`;
  if (grids.get(g)?.get(at) !== m.cols) return null;
  watchContent(g);
  const ok = m.tiles.map(t => { const k = ks.get(keptKey(t, tier)); return t.parentElement === g && !stale.has(t) && !!k && k.el === t && k.geo === geo && k.look === lookOf(t); });
  const r = g.getBoundingClientRect();
  return { at, cols: m.cols, geo, items: m.items, looks: m.tiles.map(lookOf), ok, box: { left: r.left + scrollX, top: r.top + scrollY, colW: (width - (m.cols - 1) * PACK_GAP) / m.cols } };
}

/** Where each child of a desk taken by deskLeaving would stand (or each of `items`, laid out on it), in page pixels, by
 *  key (null: its columns or its fonts are not what they were). */
export function deskPlaces(d: DeskAt, items = d.items): Map<string, { x: number; y: number; w: number; h: number }> | null {
  const g = document.getElementById('bento');
  if (!g || grids.get(g)?.get(d.at) !== d.cols || !d.geo.endsWith(`|${fontGen}`)) return null;
  const { left, top, colW } = d.box, out = new Map<string, { x: number; y: number; w: number; h: number }>();
  for (const p of solve(d.cols, items).out) out.set(items[p.i]!.key, { x: left + p.c * (colW + PACK_GAP), y: top + p.y * PACK_ROW, w: p.w * colW + (p.w - 1) * PACK_GAP, h: p.h * PACK_ROW - PACK_GAP });
  return out;
}

/** Lays the desk out with the tiles `items` names standing in for themselves (stage.ts holds them): what each needs at
 *  the new tier is its item, and nothing of it is measured. null: none stands in. */
export function stagePack(items: Map<string, PackItem> | null): void { staged = items; }

/** A held tile is drawn now: what was kept of it at this tier is its measurement again, if it looks as it did then; if
 *  it is not the height its place was given for, it is measured afresh and the desk laid out again (as a content
 *  change). `look`: its look when the tier was left. */
export function adoptStaged(t: HTMLElement, look: string): void {
  const g = t.parentElement, key = packKey(t), item = staged?.get(key);
  t.style.removeProperty('content-visibility');
  if (!g || !item) return;
  staged!.delete(key);
  const ks = kept.get(g), width = g.clientWidth, cols = grids.get(g)?.get(atOf(g, width));
  watchContent(g); placed.add(t);
  if (ks && cols && lookOf(t) === look) { ks.set(keptKey(t, tierOf(g)), { el: t, look, geo: `${cols}|${width}|${fontGen}`, nat: item.opts[0]!.w, hs: new Map(item.opts.map(o => [o.w, o.h])) }); stale.delete(t); }
  const p = placedAt.get(key), rows = Math.max(1, Math.ceil((t.offsetHeight + PACK_GAP) / PACK_ROW));
  if (stale.has(t) || !p || rows !== (t.classList.contains('stretched') ? p.h : p.h0)) { stale.add(t); packSoon(() => packMake, true); }
}

/** The staging is over: no tile stands in, none is hidden. */
export function endStage(): void {
  staged = null;
  document.querySelectorAll<HTMLElement>('#bento > .tile').forEach(t => { if (t.style.getPropertyValue('content-visibility')) t.style.removeProperty('content-visibility'); });
}
