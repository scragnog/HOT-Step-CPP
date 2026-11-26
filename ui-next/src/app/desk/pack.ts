// The desk's layout as a pure function (the concept's deskSpots, deskPlace, deskSettle and deskSolve, 11-render.js),
// with no DOM: what it is given is what the concept measured (deskMeasure), and what it returns is where each tile
// stands, in columns and 2 px rows. The rule a user can hold in their head: tiles read like a page, left to right and
// top to bottom, in the order you put them; each takes the first place after the tile before it where it fits; the
// desk only widens a tile that reads as well wider, folds one, or grows one a little to square an edge, choosing the
// widths that leave the fewest holes. The same tiles in the same order at the same width give the same desk.
// tools/test-view.ts checks it against layouts recorded from the concept page (tools/fixtures/pack.json).

export const PACK_ROW = 2, PACK_GAP = 12;
const PACK_SNAP = 24 / PACK_ROW;
export const PACK_STRETCH = 24 / PACK_ROW;
const PACK_BEAM = 24;

/** A tile as the layout sees it: the widths it may take (its own first) with its height at each, in 2 px rows. */
export interface PackItem {
  key: string;
  opts: { w: number; h: number }[];
  /** How far it may grow down to square an edge, in rows. */
  reach: number;
  /** 2: the empty-desk card (grows as far as it must); 1: the lyric sheet (half again its height); 0: anything else. */
  fill: 0 | 1 | 2;
  /** A tile (a module), as opposed to the empty desk, the fold button or the parked tray. */
  tile: boolean;
  /** The lyric sheet at the head of the desk: it stands in the first column. */
  lead: boolean;
}
export interface Rect { i: number; c: number; w: number; y: number; h: number; h0: number; o: number }
export interface Plan { o: number; c?: number }
export interface Layout { out: Rect[]; bottom: number; blank: number; fillGrow: number; flexGrow: number; holes: number; ragged: number; score?: number; plan: Plan[]; ws?: number[] }

/** The places a tile may take after the one before it: in each column, the highest place at or after the previous
 *  tile in reading order where it fits; a top just above another tile's edge drops to meet it. */
export function deskSpots(C: number, rs: Rect[], cy: number, cc: number, w: number, h: number, lead: boolean): { c: number; y: number }[] {
  const free = (c: number, y: number) => rs.every(r => r.c + r.w <= c || c + w <= r.c || r.y + r.h <= y || y + h <= r.y), out: { c: number; y: number }[] = [];
  for (let c = 0; c + w <= C; c++) {
    if (lead && c) break;
    const ys = [cy, ...rs.filter(r => !(r.c + r.w <= c || c + w <= r.c) && r.y + r.h > cy).map(r => r.y + r.h)].sort((a, b) => a - b);
    for (const y0 of ys) {
      if (y0 === cy && c <= cc) continue;
      if (!free(c, y0)) continue;
      let y = y0;
      rs.forEach(r => [r.y, r.y + r.h].forEach(e => { if (e > y0 && e <= y0 + PACK_SNAP && (y === y0 || e < y) && free(c, e)) y = e; }));
      if (y !== y0 && !rs.some(r => r.y + r.h === y0 && !(r.c + r.w <= c || c + w <= r.c))) y = y0;
      out.push({ c, y }); break;
    }
  }
  return out.sort((a, b) => a.y - b.y || a.c - b.c);
}

/** deskSpots, for the solver's states: the tiles over the columns a place spans are read from `cols` (each column's
 *  tiles in order of their tops), so a place is tested against those tiles only. The same places, in the same order. */
function spotsIn(C: number, rs: Rect[], cols: Rect[][], cy: number, cc: number, w: number, h: number, lead: boolean): { c: number; y: number }[] {
  const out: { c: number; y: number }[] = [];
  for (let c = 0; c + w <= C; c++) {
    if (lead && c) break;
    /* each tile over these columns once: from the first of them it covers */
    const over: Rect[] = []; for (let k = c; k < c + w; k++) for (const r of cols[k]!) if (Math.max(r.c, c) === k) over.push(r);
    const free = (y: number) => { for (const r of over) if (!(r.y + r.h <= y || y + h <= r.y)) return false; return true; };
    const ys = [cy]; for (const r of over) if (r.y + r.h > cy) ys.push(r.y + r.h);
    ys.sort((a, b) => a - b);
    for (const y0 of ys) {
      if (y0 === cy && c <= cc) continue;
      if (!free(y0)) continue;
      /* the nearest free edge of any tile just below drops it there */
      let y = y0;
      for (const r of rs) for (const e of [r.y, r.y + r.h]) if (e > y0 && e <= y0 + PACK_SNAP && (y === y0 || e < y) && free(e)) y = e;
      if (y !== y0 && !over.some(r => r.y + r.h === y0)) y = y0;
      out.push({ c, y }); break;
    }
  }
  return out.sort((a, b) => a.y - b.y || a.c - b.c);
}

/** Places items by a plan (each item's option and column; with no column, the highest place). */
export function deskPlace(C: number, items: PackItem[], plan: Plan[]): Rect[] {
  const rs: Rect[] = []; let cy = 0, cc = -1;
  items.forEach((it, i) => {
    const o = it.opts[plan[i]?.o || 0]!, w = Math.min(C, o.w), h = o.h, sp = deskSpots(C, rs, cy, cc, w, h, it.lead);
    const s = sp.find(x => x.c === plan[i]?.c) || sp[0]!;
    rs.push({ i, c: s.c, w, y: s.y, h, h0: h, o: plan[i]?.o || 0 }); cy = s.y; cc = s.c;
  });
  return rs;
}

/** Squares the edges (each tile grows down to the next tile under it, or the desk's foot, when that is close), then
 *  counts what is left empty. */
export function deskSettle(C: number, items: PackItem[], rs: Rect[]): Omit<Layout, 'plan'> {
  const out = rs.map(r => ({ ...r })), bottom = Math.max(0, ...out.map(r => r.y + r.h));
  let blank = 0, fillGrow = 0, flexGrow = 0, holes = 0, ragged = 0;
  for (const r of out) {
    let gap = Infinity;
    for (let c = r.c; c < r.c + r.w; c++) {
      let nxt = Infinity;
      out.forEach(o => { if (o !== r && o.c <= c && c < o.c + o.w && o.y >= r.y + r.h) nxt = Math.min(nxt, o.y); });
      gap = Math.min(gap, (nxt < Infinity ? nxt : bottom) - (r.y + r.h));
    }
    const fill = items[r.i]!.fill, reach = fill === 2 ? Infinity : fill ? r.h0 * 1.5 : items[r.i]!.reach;
    if (gap > 0 && gap <= reach) { r.h += gap; if (fill) fillGrow += gap * r.w; else if (items[r.i]!.reach > PACK_STRETCH) flexGrow += gap * r.w; else blank += gap * r.w; }
  }
  for (let c = 0; c < C; c++) {
    const col = out.filter(r => r.c <= c && c < r.c + r.w).sort((a, b) => a.y - b.y); let y = 0;
    col.forEach(r => { holes += Math.max(0, r.y - y); y = Math.max(y, r.y + r.h); });
    ragged += bottom - y;
  }
  return { out, bottom, blank, fillGrow, flexGrow, holes, ragged };
}

type St = { rs: Rect[]; cols: Rect[][]; bot: number; key: string; cy: number; cc: number; cost: number; plan: Plan[]; s?: number };
const START = (C: number): St => ({ rs: [], cols: Array.from({ length: C }, () => []), bot: -Infinity, key: '', cy: 0, cc: -1, cost: 0, plan: [] });
const NONE: Layout = { out: [], bottom: 0, blank: 0, fillGrow: 0, flexGrow: 0, holes: 0, ragged: 0, plan: [], ws: [] };

/** One step of the search: item i placed every way it may go after each state of the beam; the best states kept. */
function step(C: number, items: PackItem[], i: number, beam: St[]): St[] {
  const reachOf = (r: Rect) => (items[r.i]!.fill === 2 ? Infinity : items[r.i]!.fill ? r.h0 * 1.5 : items[r.i]!.reach);
  /* the empty height under the tiles, where nothing below can close it: a column's gaps, less what the tile above may
     grow by (cap); cols[c] holds the tiles that cover column c, in order of their tops (then of placing) */
  const dead = (cols: Rect[][], cy: number) => {
    const under = (r: Rect, c: number) => { if (Math.max(cy, r.y + r.h) !== cy) return Infinity; const b = r.y + r.h; for (const o of cols[c]!) if (o !== r && o.y >= b) return Math.min(cy, o.y); return cy; };
    const cap = (r: Rect) => { let g = Infinity; for (let c = r.c; c < r.c + r.w; c++) { const n = under(r, c); g = Math.min(g, n === Infinity ? 0 : n - (r.y + r.h)); } return Math.min(Math.max(0, g), reachOf(r)); };
    let d = 0;
    for (let c = 0; c < C; c++) {
      let y = 0, last: Rect | null = null;
      for (const r of cols[c]!) { if (r.y > y) d += Math.max(0, r.y - y - (last ? cap(last) : 0)); if (r.y + r.h >= y) { y = r.y + r.h; last = r; } }
      if (cy > y) d += Math.max(0, cy - y - (last ? cap(last) : 0));
    }
    return d;
  };
  /* a column's tiles with r added: after every tile whose top is at or above r's, as a stable sort by top puts it */
  const withRect = (col: Rect[], r: Rect) => { let k = col.length; while (k > 0 && col[k - 1]!.y > r.y) k--; return [...col.slice(0, k), r, ...col.slice(k)]; };
  const it = items[i]!, next: St[] = [];
  for (const st of beam) it.opts.forEach((o, oi) => {
    const w = Math.min(C, o.w), sp = spotsIn(C, st.rs, st.cols, st.cy, st.cc, w, o.h, it.lead);
    sp.forEach((s, si) => {
      const r: Rect = { i, c: s.c, w, y: s.y, h: o.h, h0: o.h, o: oi }, rs = [...st.rs, r], cols = st.cols.slice();
      for (let c = s.c; c < s.c + w; c++) cols[c] = withRect(st.cols[c]!, r);
      const cost = st.cost + (oi ? 40 : 0) + (s.y - sp[0]!.y) * 0.5 * w + si * 2, bot = Math.max(st.bot, s.y + o.h);
      const ns: St = { rs, cols, bot, key: (st.key ? st.key + ' ' : '') + s.c + ',' + w + ',' + s.y, cy: s.y, cc: s.c, cost, plan: [...st.plan, { o: oi, c: s.c }] };
      ns.s = dead(cols, s.y) * 4 + bot * C * 0.5 + cost; next.push(ns);
    });
  });
  next.sort((a, b) => a.s! - b.s!);
  const seen = new Set<string>(), out: St[] = [];
  for (const ns of next) { if (seen.has(ns.key)) continue; seen.add(ns.key); out.push(ns); if (out.length >= PACK_BEAM) break; }
  return out;
}
/** The best of the last beam, its edges squared. */
function finish(C: number, items: PackItem[], beam: St[]): Layout {
  let best: Layout | null = null;
  for (const st of beam) {
    const s = deskSettle(C, items, st.rs) as Layout;
    s.score = s.holes * 4 + s.ragged * 1.2 + s.blank * 2 + s.fillGrow * 0.25 + s.flexGrow * 0.5 + s.bottom * C * 0.5 + st.cost; s.plan = st.plan;
    if (!best || s.score < best.score! - 0.01) best = s;
  }
  best!.ws = best!.plan.map(p => p.o);
  return best!;
}

/** Chooses each tile's width and place by a small beam search in reading order: holes inside the desk weigh most,
 *  then empty columns at its foot, then blank surface, then height; a width that is not the tile's own, or a place
 *  lower than the highest one open to it, costs a little. Deterministic. Each state keeps, per column, its tiles in
 *  order of their tops (`cols`) and its lowest edge (`bot`), so a candidate is scored without sorting the desk again. */
export function deskSolve(C: number, items: PackItem[]): Layout {
  if (!items.length) return { ...NONE };
  let beam = [START(C)];
  for (let i = 0; i < items.length; i++) beam = step(C, items, i, beam);
  return finish(C, items, beam);
}

/** deskSolve with `item` put at each place k from `lo` to `hi` among `rest` (the arrow keys and a drag ask where a tile
 *  would stand at every place): the same layouts, the steps the orders share (the tiles before the place) taken once. */
export function deskSolveAt(C: number, rest: PackItem[], item: PackItem, lo: number, hi: number): Layout[] {
  const pre: St[][] = [[START(C)]];
  for (let j = 0; j < hi; j++) pre.push(step(C, rest, j, pre[j]!));
  const out: Layout[] = [];
  for (let k = lo; k <= hi; k++) {
    const items = [...rest.slice(0, k), item, ...rest.slice(k)];
    let beam = pre[k]!;
    for (let i = k; i < items.length; i++) beam = step(C, items, i, beam);
    out.push(finish(C, items, beam));
  }
  return out;
}
