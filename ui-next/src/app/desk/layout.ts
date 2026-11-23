// The desk's layout on the page: measuring what each tile needs (the concept's deskMeasure) and placing the tiles
// where the pure layout (pack.ts) puts them (packNow), with the concept's rule for when the widths are kept: a change
// that leaves the desk's make-up alone (the same tiles, order, tier, engine and window width: a value set, a line
// appearing in a tile) keeps every tile's width and column (Q-29); any other change chooses afresh. 11-render.js.

import { deskPlace, deskSettle, deskSolve, PACK_GAP, PACK_ROW, PACK_STRETCH, type Layout, type PackItem, type Plan } from './pack.ts';

interface Measured { cols: number; tiles: HTMLElement[]; items: PackItem[] }
let packMake = '', packFresh = 0;
const state = new WeakMap<HTMLElement, { sig: string; plan: Plan[] } | null>();
let meas: Measured | null = null;
/** The last measurement of the desk, for the keyboard moves (deskSlots). */
export const lastMeasure = (): Measured | null => meas;

/** Runs a layout-measuring step without the page height or scroll position moving, even for an instant. */
export function holdScroll(fn: () => void): void {
  const html = document.documentElement, x = scrollX, y = scrollY, a = html.style.overflowAnchor;
  html.style.overflowAnchor = 'none'; html.style.minHeight = html.scrollHeight + 'px';
  try { fn(); } finally { html.style.minHeight = ''; html.style.overflowAnchor = a; if (scrollX !== x || scrollY !== y) scrollTo(x, y); }
}

const packKey = (t: HTMLElement) => t.dataset.m || t.dataset.gm || (t.classList.contains('desk-empty') ? '_empty' : t.className);

function measure(g: HTMLElement): Measured {
  const tiles = [...g.children].filter((t): t is HTMLElement => t instanceof HTMLElement && !t.classList.contains('moving'));
  g.classList.add('measuring');
  tiles.forEach(t => { t.style.gridColumn = ''; t.style.gridRow = ''; t.classList.remove('stretched', 'widened', 'narrowed'); });
  const cols = getComputedStyle(g).gridTemplateColumns.split(' ').length;
  const rows = (h: number) => Math.max(1, Math.ceil((h + PACK_GAP) / PACK_ROW));
  const nat = tiles.map(t => { const cs = getComputedStyle(t), e = cs.gridColumnEnd, sp = [cs.gridColumnStart, e].find(v => /^span/.test(v)); return e === '-1' ? cols : sp ? Math.min(cols, parseInt(sp.slice(5)) || 1) : 1; });
  const opts = tiles.map((t, i) => {
    const w = nat[i]!, o = [w];
    if ('grow' in t.dataset && w < cols && (w === 1 ? cols > 2 : cols === 3)) o.push(w + 1);
    if ('grow' in t.dataset && w === 2 && cols === 4) o.push(4);
    if ('fold' in t.dataset && cols === 3 && w === 2) o.push(1);
    if ('span' in t.dataset && w < cols && !o.includes(cols)) o.push(cols);
    if (t.dataset.m === 'lyrics' && (cols === 3 || cols === 4) && w === 2) o.push(cols);
    if (t.dataset.m === 'post' && cols === 4 && w === 4 && tiles[0]?.dataset.m === 'lyrics') o.push(2);
    return o;
  });
  const hs = tiles.map(t => [rows(t.offsetHeight)]);
  for (let j = 1; j < 3; j++) {
    const set = tiles.filter((_, i) => opts[i]![j]);
    if (!set.length) break;
    tiles.forEach((t, i) => { const o = opts[i]![j]; if (o) { t.style.gridColumn = 'span ' + o; t.classList.toggle('narrowed', o < nat[i]!); t.classList.toggle('widened', o > nat[i]!); } });
    tiles.forEach((t, i) => { if (opts[i]![j]) hs[i]![j] = rows(t.offsetHeight); });
    set.forEach(t => { t.style.gridColumn = ''; t.classList.remove('narrowed', 'widened'); });
  }
  g.classList.remove('measuring');
  return {
    cols, tiles,
    items: tiles.map((t, i) => ({
      key: packKey(t), opts: opts[i]!.map((w, j) => ({ w, h: hs[i]![j]! })),
      reach: t.dataset.stretch ? Math.round(+t.dataset.stretch / PACK_ROW) : PACK_STRETCH,
      fill: t.classList.contains('desk-empty') ? 2 : t.dataset.m === 'lyrics' ? 1 : 0,
      tile: !!(t.dataset.m || t.dataset.gm) && t.dataset.m !== 'lyrics', lead: i === 0 && t.dataset.m === 'lyrics',
    })),
  };
}

/** The desk's make-up: what, when it changes, makes the layout choose its widths afresh. */
export const deskMake = (tier: string, engine: string, mods: string[], qAll: boolean) => [tier, engine, mods.join(), qAll, innerWidth].join('|');

/** A render: note whether the make-up changed (a fresh layout), then place the tiles. */
export function renderPack(make: string): void {
  if (make !== packMake) packFresh = performance.now();
  packMake = make;
  packNow(make);
}

/** Places every tile. `keep`: keep the widths chosen for this same desk (a content change). */
export function packNow(make: string, keep = make === packMake && performance.now() - packFresh > 700): void {
  holdScroll(() => document.querySelectorAll<HTMLElement>('.bento').forEach(g => {
    if (getComputedStyle(g).display !== 'grid') return;
    if (!g.offsetParent) { if (state.get(g)) packFresh = performance.now(); state.set(g, null); return; }
    g.style.minHeight = g.offsetHeight + 'px';
    const m = measure(g), sig = g.clientWidth + '|' + m.cols + '|' + m.items.map(it => it.key).join('|');
    const was = state.get(g);
    let res: Layout;
    if (keep && was && was.sig === sig) res = { ...deskSettle(m.cols, m.items, deskPlace(m.cols, m.items, was.plan)), plan: was.plan };
    else res = deskSolve(m.cols, m.items);
    state.set(g, { sig, plan: res.plan }); meas = m;
    m.tiles.forEach((t, i) => {
      const p = res.out.find(r => r.i === i); if (!p) return; const w0 = m.items[i]!.opts[0]!.w;
      t.style.gridColumn = `${p.c + 1} / span ${p.w}`; t.style.gridRow = `${p.y + 1} / span ${p.h}`;
      t.classList.toggle('stretched', p.h > p.h0); t.classList.toggle('widened', p.w > w0); t.classList.toggle('narrowed', p.w < w0);
    });
    g.style.minHeight = '';
  }));
}

/** A content change (a tile's body changed size, the window resized): re-place, keeping widths while the make-up holds. */
let raf = 0;
export function packSoon(make: () => string, content: boolean): void {
  const now = make();
  if (!content && now !== packMake) { packFresh = performance.now(); }
  const keep = content && now === packMake ? performance.now() - packFresh > 700 : false;
  packMake = now;
  if (!raf) raf = requestAnimationFrame(() => { raf = 0; packNow(now, keep); });
}

/** Where a tile would stand at each place among the others (index k is a place among the desk's tiles, not before the
 *  lyric sheet), in page pixels, for the arrow keys: the layout is a pure function of the order, so each place is
 *  exactly what a move there gives. */
export function deskSlots(g: HTMLElement, key: string, item: PackItem): { k: number; x: number; y: number; w: number; h: number }[] {
  const m = meas; if (!m) return [];
  const rest = m.items.filter(it => it.key !== key);
  const lead = rest[0]?.lead ? 1 : 0, last = rest.reduce((a, it, i) => (it.tile ? i + 1 : a), lead);
  const gr = g.getBoundingClientRect(), colW = (g.clientWidth - (m.cols - 1) * PACK_GAP) / m.cols, out = [];
  for (let k = lead; k <= last; k++) {
    const items = [...rest.slice(0, k), item, ...rest.slice(k)], res = deskSolve(m.cols, items), p = res.out.find(r => r.i === k)!;
    out.push({ k: k - lead, x: gr.left + scrollX + p.c * (colW + PACK_GAP), y: gr.top + scrollY + p.y * PACK_ROW, w: p.w * colW + (p.w - 1) * PACK_GAP, h: p.h * PACK_ROW - PACK_GAP });
  }
  return out;
}
