// The board's geometry (27-pedalboard.js): Lab pedals stand on one line, each as tall as its controls rounded up to a
// 32 px step (snapPedals); a chain that needs two rows or more wraps into balanced rows (balanceRows); and the cables,
// measured on screen so they land on the jacks, each link a <g data-k="a>b"> updated in place (fixCables). How things
// are drawn and move; nothing here is about a value. A cable that already exists never redraws; a new one draws out of
// its jacks when the board has just changed (drawOut, features/motion.ts), and one that is no longer needed retracts;
// while pedals slide, the cables follow them frame by frame (followCables).

import { easing, mtime, reduced } from '../kit/motion.ts';
import { packed } from '../desk/layout.ts';
const PED_STEP = 32, JACKY = 36, PLUGW = 10, CABR = 10;

export function snapPedals(board: HTMLElement | null, lab: boolean): void {
  /* a board not drawn (its tile held out of view by a staged tier change) is snapped once it is */
  const pd = board?.querySelector<HTMLElement>('[data-pedals]'); if (!pd || !board!.checkVisibility()) return;
  const ps = [...pd.querySelectorAll<HTMLElement>(':scope > .pedal')];
  ps.forEach(p => { p.style.minHeight = ''; }); balanceRows(pd, ps); packed(); if (!lab) return;
  const hs = ps.map(p => p.offsetHeight); ps.forEach((p, i) => { p.style.minHeight = Math.ceil(hs[i]! / PED_STEP) * PED_STEP + 'px'; });
}

/** The narrowest board that still takes the same number of rows, centred, so no pedal is left alone on the last row. */
function balanceRows(pd: HTMLElement, ps: HTMLElement[]): void {
  pd.style.paddingRight = pd.style.paddingLeft = ''; if (ps.length < 3) return;
  const cs = getComputedStyle(pd), gap = parseFloat(cs.columnGap) || 0, pr = parseFloat(cs.paddingRight) || 0, pl = parseFloat(cs.paddingLeft) || 0, avail = pd.clientWidth - pl - pr, ws = ps.map(p => p.offsetWidth);
  const rows = (W: number) => { let n = 1, x = 0; ws.forEach(w => { if (x && x + gap + w > W + 0.5) { n++; x = w; } else x = x ? x + gap + w : w; }); return n; };
  const n = rows(avail); if (n < 2) return;
  let lo = Math.max(...ws), hi = avail; while (hi - lo > 1) { const m = (lo + hi) / 2; if (rows(m) <= n) hi = m; else lo = m; }
  const spare = Math.floor(avail - hi); pd.style.paddingLeft = (pl + Math.floor(spare / 2)) + 'px'; pd.style.paddingRight = (pr + Math.ceil(spare / 2)) + 'px';
}

interface P { id: string; l: number; r: number; t: number; b: number }
interface Link { k: string; d: string; plugs: [number, number, number][]; label?: [string, number, number, string] }
function cableLinks(ps: P[], W: number, jb: number): Link[] {
  const L = 14, R = W - 14, out: Link[] = [], jy = (p: P) => (jb ? Math.max(p.t + 20, p.b - jb) : p.t + JACKY);
  if (!ps.length) return out;
  const f = ps[0]!, z = ps[ps.length - 1]!;
  out.push({ k: 'in>' + f.id, d: `M0 ${jy(f)}H${f.l - PLUGW}`, plugs: [[f.l - PLUGW, jy(f), 0]], label: ['IN', 4, jy(f) - 10, 'start'] });
  for (let i = 1; i < ps.length; i++) {
    const a = ps[i - 1]!, b = ps[i]!, sx = a.r + PLUGW, ex = b.l - PLUGW, ay = jy(a), by = jy(b);
    let d: string;
    if (Math.abs(a.b - b.b) < 4 && b.l > a.l) { const g = ex - sx, sag = Math.max(8, Math.min(14, g * 0.35)); d = `M${sx} ${ay}C${sx + g * 0.5} ${ay + sag} ${ex - g * 0.5} ${by + sag} ${ex} ${by}`; }
    else {
      const nextTop = Math.min(...ps.filter(p => Math.abs(p.b - b.b) < 4).map(p => p.t));
      const x1 = Math.max(sx + CABR + 2, Math.min(sx + 14, R)), G = (a.b + nextTop) / 2 - 2;
      d = `M${sx} ${ay}H${x1 - CABR}Q${x1} ${ay} ${x1} ${ay + CABR}V${G - CABR}Q${x1} ${G} ${x1 - CABR} ${G}H${L + CABR}Q${L} ${G} ${L} ${G + CABR}V${by - CABR}Q${L} ${by} ${L + CABR} ${by}H${ex}`;
    }
    out.push({ k: a.id + '>' + b.id, d, plugs: [[sx, ay, 1], [ex, by, 0]] });
  }
  out.push({ k: z.id + '>out', d: `M${z.r + PLUGW} ${jy(z)}H${W}`, plugs: [[z.r + PLUGW, jy(z), 1]], label: ['OUT', W - 4, jy(z) - 10, 'end'] });
  return out;
}
const plugSVG = ([x, y, right]: [number, number, number]) => `<rect class="plug" x="${right ? x - 8 : x}" y="${y - 5}" width="8" height="10" rx="2"/><rect class="plug-c" x="${right ? x - 2 : x}" y="${y - 3.5}" width="2" height="7" rx="1"/>`;

type G = SVGGElement & { _d?: string; _gone?: boolean };
let draw = false, follow = 0, until = 0;
/** The next cables drawn are new links of a board that has just changed: they draw out of their jacks. */
export const drawOut = (): void => { draw = true; };
/** The cables are measured on screen, so they follow pedals while those are mid-animation: redrawn every frame for ms. */
export function followCables(board: () => HTMLElement | null, ms: number): void {
  until = performance.now() + ms;
  if (!follow) follow = requestAnimationFrame(function step() { fixCables(board()); follow = performance.now() < until ? requestAnimationFrame(step) : 0; if (!follow) window.setTimeout(() => fixCables(board()), 40); });
}
export function fixCables(board: HTMLElement | null): void {
  const pd = board?.querySelector<HTMLElement>('[data-pedals]'), svg = board?.querySelector<SVGSVGElement>('.cables'); if (!pd || !svg || !board!.checkVisibility()) return;
  const sr = svg.getBoundingClientRect(), W = sr.width; if (!W) return;
  const ps = [...pd.querySelectorAll<HTMLElement>(':scope > .pedal')].map(p => { const r = p.getBoundingClientRect(); return { id: p.dataset.pid!, l: r.left - sr.left, r: r.right - sr.left, t: r.top - sr.top, b: r.bottom - sr.top }; });
  const jb = parseFloat(getComputedStyle(board!).getPropertyValue('--jb')) || 0;
  const out = draw, ease = easing('--e-out'), dur = mtime('--t-base'), delay = mtime('--t-land') + 60; draw = false;
  const links = cableLinks(ps, W, jb), have = new Map([...svg.children].map(g => g as G).filter(g => g.dataset.k && !g._gone).map(g => [g.dataset.k!, g]));
  links.forEach(l => {
    let g = have.get(l.k);
    const plugs = l.plugs.map(plugSVG).join('');
    if (g) {
      have.delete(l.k);
      if (g._d !== l.d) { g._d = l.d; g.querySelectorAll('path').forEach(p => p.setAttribute('d', l.d)); const pl = g.querySelector('.plugs'); if (pl) pl.innerHTML = plugs; const t = g.querySelector('text'); if (t && l.label) { t.setAttribute('x', String(l.label[1])); t.setAttribute('y', String(l.label[2])); } }
      return;
    }
    g = document.createElementNS('http://www.w3.org/2000/svg', 'g') as G; g.dataset.k = l.k; g._d = l.d; g.setAttribute('class', 'lk');
    g.innerHTML = `<path class="cb" d="${l.d}" pathLength="1"/><path class="cb-hi" d="${l.d}" pathLength="1"/><path class="cb-sig" d="${l.d}" pathLength="100"/><g class="plugs">${plugs}</g>${l.label ? `<text class="cb-lbl" x="${l.label[1]}" y="${l.label[2]}" text-anchor="${l.label[3]}">${l.label[0]}</text>` : ''}`;
    svg.appendChild(g);
    if (!out || reduced()) return;
    g.querySelectorAll('.cb, .cb-hi').forEach(p => p.animate([{ strokeDasharray: '1 2', strokeDashoffset: 1 }, { strokeDasharray: '1 2', strokeDashoffset: 0 }], { duration: dur, delay, easing: ease, fill: 'backwards' }));
    g.querySelectorAll('.plugs, .cb-lbl').forEach(p => p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: dur * 0.6, delay, easing: ease, fill: 'backwards' }));
  });
  /* the links stand in the chain's order in the drawing, whatever order they came in */
  const order = links.map(l => [...svg.children].find(g => (g as G).dataset.k === l.k && !(g as G)._gone)).filter((g): g is Element => !!g);
  if (order.some((g, i) => [...svg.children].filter(x => !(x as G)._gone)[i] !== g)) order.forEach(g => svg.appendChild(g));
  have.forEach(g => {
    g._gone = true;
    if (reduced()) { g.remove(); return; }
    g.querySelectorAll('.cb, .cb-hi').forEach(p => p.animate([{ strokeDasharray: '1 2', strokeDashoffset: 0 }, { strokeDasharray: '1 2', strokeDashoffset: -1 }], { duration: mtime('--t-fast'), easing: easing('--e-in'), fill: 'forwards' }));
    g.animate([{ opacity: 1 }, { opacity: 0 }], { duration: mtime('--t-fast'), easing: easing('--e-in'), fill: 'forwards' }).finished.then(() => g.remove(), () => g.remove());
  });
}

/** The board re-balances its rows and redraws its cables when it changes width or the window resizes. */
export function watchBoard(board: HTMLElement | null, lab: () => boolean): () => void {
  if (!board) return () => {};
  let bw = -1;
  const ro = new ResizeObserver(es => { es.forEach(e => { const w = Math.round(e.contentRect.width); if (bw !== w) { bw = w; snapPedals(board, lab()); } }); fixCables(board); });
  ro.observe(board);
  let t = 0; const rs = () => { clearTimeout(t); t = window.setTimeout(() => { snapPedals(board, lab()); fixCables(board); }, 120); requestAnimationFrame(() => fixCables(board)); };
  addEventListener('resize', rs);
  /* a pedal that gains or loses a line inside it (the Mastering pedal's "lent to Timbre" note) is snapped again, as the
     concept's board is drawn again: its 32px step is taken from what it now holds (RD-16) */
  let f = 0; const mo = new MutationObserver(ms => { if (ms.every(m => (m.target as Element).closest?.('.cables'))) return; if (!f) f = requestAnimationFrame(() => { f = 0; snapPedals(board, lab()); fixCables(board); }); });
  mo.observe(board, { childList: true, subtree: true });
  return () => { ro.disconnect(); mo.disconnect(); cancelAnimationFrame(f); removeEventListener('resize', rs); };
}
